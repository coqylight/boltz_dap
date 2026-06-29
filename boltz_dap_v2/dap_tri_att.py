"""
DAP-aware Triangle Attention for Boltz 2 — exact matching edition.

Uses the ORIGINAL attention code path (explicit Q@K^T + bias + softmax_no_cast)
to produce bit-identical results to the baseline. Chunks over the row dim
for memory efficiency (~6.9 GB transient per chunk vs ~0 for SDPA).

Starting node: row-scattered z works directly (all N columns available).
Ending node: needs row_to_col to get all N rows, then operates like starting.
"""

import math
import os
import sys
import torch
from torch import Tensor, nn
from typing import Optional
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import row_to_col, col_to_row, gather
from boltz_distributed.core import get_dap_size, get_dap_rank
from dap_dtype_guard import ensure_bf16_row_to_col_input

from boltz.model.layers.triangular_attention.utils import (
    permute_final_dims,
    chunk_layer,
)
from boltz.model.layers.triangular_attention.primitives import (
    _attention,
)


def _module_compute_dtype(module: nn.Module) -> torch.dtype:
    """Use original attention parameter dtype for pointwise math."""
    return module.layer_norm.weight.dtype


class DAPTriAttStart(nn.Module):
    """DAP wrapper for TriangleAttentionStartingNode.

    Operates on row-scattered z [B, N/dap, N, D].
    Starting node attention: iterate over rows (N/dap, local),
    attend across columns (N, full). Only the bias is gathered.
    Uses the ORIGINAL attention code for exact numerical matching.
    """

    def __init__(self, original_tri_att):
        super().__init__()
        self.inner = original_tri_att

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, chunk_size, use_kernels)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        storage_dtype = x.dtype
        compute_dtype = _module_compute_dtype(self.inner)
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)
        if mask.dtype != x.dtype:
            mask = mask.to(dtype=x.dtype)

        # Layer norm (pointwise)
        x = self.inner.layer_norm(x)

        # Mask bias: [B, N/dap, 1, 1, N]
        mask_bias = self.inner.inf * (mask[..., :, None, None, :] - 1)

        # Triangle bias: gather only H channels (not D)
        local_bias = self.inner.linear(x)
        local_bias = permute_final_dims(local_bias, (2, 0, 1))

        # Gather dim 2 (N/dap -> N): gives [B, H, N, N]
        N = x.shape[2]
        full_bias = gather(local_bias.contiguous(), dim=2, original_size=N)
        del local_bias
        full_bias = full_bias.unsqueeze(-4)  # [B, 1, H, N, N]

        # Use original _chunk / mha path for exact matching
        if chunk_size is not None and not use_kernels:
            mha_inputs = {
                "q_x": x,
                "kv_x": x,
                "tri_bias": full_bias,
                "mask_bias": mask_bias,
                "mask": mask[..., :, None, None, :],
            }
            x = chunk_layer(
                partial(self.inner.mha, use_kernels=use_kernels),
                mha_inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(x.shape[:-2]),
                _out=None,
            )
        else:
            x = self.inner.mha(
                x, x, full_bias, mask_bias,
                mask[..., :, None, None, :],
                use_kernels=use_kernels,
            )

        if x.dtype != storage_dtype:
            x = x.to(dtype=storage_dtype)
        return x


class DAPTriAttEnd(nn.Module):
    """DAP wrapper for TriangleAttentionEndingNode.

    Ending node needs all N rows for keys/queries.
    Strategy: row_to_col -> transpose -> operate like starting node.
    Uses the ORIGINAL attention code for exact numerical matching.
    """

    def __init__(self, original_tri_att):
        super().__init__()
        self.inner = original_tri_att

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, chunk_size, use_kernels)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        original_N = x.shape[2]
        dap_rank = get_dap_rank()
        storage_dtype = x.dtype

        # 1. row_to_col: [B, N/dap, N, D] -> [B, N_pad, N/dap, D]
        ensure_bf16_row_to_col_input(
            x, tag="tri_att_end.row_to_col", rank=dap_rank
        )
        x_col = row_to_col(x)
        mask_col = row_to_col(mask.unsqueeze(-1)).squeeze(-1)

        # Zero out padded positions to prevent them from affecting attention
        N_pad = x_col.shape[1]
        if N_pad > original_N:
            x_col[:, original_N:, :, :] = 0
            mask_col[:, original_N:, :] = 0

        # 2. Transpose for ending node: [B, N/dap, N_pad, D]
        x_t = x_col.transpose(-2, -3)
        mask_t = mask_col.transpose(-1, -2)
        del x_col, mask_col

        compute_dtype = _module_compute_dtype(self.inner)
        if x_t.dtype != compute_dtype:
            x_t = x_t.to(dtype=compute_dtype)
        if mask_t.dtype != x_t.dtype:
            mask_t = mask_t.to(dtype=x_t.dtype)

        # 3. Layer norm (pointwise)
        x_t = self.inner.layer_norm(x_t)

        # 4. Mask bias: [B, N/dap, 1, 1, N_pad]
        mask_bias = self.inner.inf * (mask_t[..., :, None, None, :] - 1)

        # 5. Triangle bias: gather the small bias
        local_bias = self.inner.linear(x_t)
        local_bias = permute_final_dims(local_bias, (2, 0, 1))
        full_bias = gather(local_bias.contiguous(), dim=2, original_size=N_pad)
        del local_bias
        full_bias = full_bias.unsqueeze(-4)  # [B, 1, H, N_pad, N_pad]

        # 6. Use original _chunk / mha path for exact matching
        mask_expanded = mask_t[..., :, None, None, :]
        if chunk_size is not None and not use_kernels:
            mha_inputs = {
                "q_x": x_t,
                "kv_x": x_t,
                "tri_bias": full_bias,
                "mask_bias": mask_bias,
                "mask": mask_expanded,
            }
            x_t = chunk_layer(
                partial(self.inner.mha, use_kernels=use_kernels),
                mha_inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(x_t.shape[:-2]),
                _out=None,
            )
        else:
            x_t = self.inner.mha(
                x_t, x_t, full_bias, mask_bias,
                mask_expanded,
                use_kernels=use_kernels,
            )

        # 7. Transpose back + col_to_row
        x_col_out = x_t.transpose(-2, -3)
        del x_t
        if x_col_out.dtype != storage_dtype:
            x_col_out = x_col_out.to(dtype=storage_dtype)
        x_out = col_to_row(x_col_out)
        del x_col_out

        # 8. Trim padding
        if x_out.shape[2] > original_N:
            x_out = x_out[:, :, :original_N, :]

        if x_out.dtype != storage_dtype:
            x_out = x_out.to(dtype=storage_dtype)
        return x_out
