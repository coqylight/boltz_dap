"""
DAP-aware PairformerNoSeqLayer for Boltz 2.

Flow per layer (row-scattered z [B, N/dap, N, D]):
  z → tri_mul_out(z)    — row-scattered, DAP-wrapped
  z → row_to_col → z_col → tri_mul_in(z_col) → col_to_row → z  — trim padding
  z → DAPTriAttStart(z)  — scattered, gathers only small bias
  z → DAPTriAttEnd(z)    — uses row_to_col internally, gathers only bias
  z → transition_z(z)    — pointwise
"""

import torch
from torch import Tensor, nn
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import row_to_col, col_to_row
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_trimul import DAPTriMulOut, DAPTriMulIn
from dap_tri_att import DAPTriAttStart, DAPTriAttEnd


def get_dropout_mask(dropout_rate, x, training, columnwise=False):
    """Generate dropout mask matching Boltz's pairformer dropout."""
    if not training or dropout_rate == 0.0:
        return 1.0
    shape = list(x.shape)
    if columnwise:
        shape[-2] = 1
    else:
        shape[-3] = 1
    mask = torch.ones(shape, device=x.device, dtype=x.dtype)
    mask = torch.nn.functional.dropout(mask, p=dropout_rate, training=True)
    return mask


class DAPPairformerNoSeqLayer(nn.Module):
    """DAP wrapper for PairformerNoSeqLayer.

    Accepts and returns z in row-scattered form [B, N/dap, N, D].
    No full z gather needed — tri_att gathers only the small bias.
    """

    def __init__(self, original_layer):
        super().__init__()
        self.tri_mul_out = DAPTriMulOut(original_layer.tri_mul_out)
        self.tri_mul_in = DAPTriMulIn(original_layer.tri_mul_in)
        self.tri_att_start = DAPTriAttStart(original_layer.tri_att_start)
        self.tri_att_end = DAPTriAttEnd(original_layer.tri_att_end)
        self.transition_z = original_layer.transition_z
        self.dropout = original_layer.dropout
        self._diag_enabled = False  # toggled externally for profiling
        self._save_subop_checkpoints = False  # set True to save per-sub-op z
        self._subop_data = {}  # populated when _save_subop_checkpoints is True

    def _gather_z_full(self, z_scattered, original_N):
        """Gather scattered z to full tensor for checkpoint comparison."""
        from boltz_distributed.comm import gather as dap_gather
        from boltz_distributed.core import get_dap_size
        dap_size = get_dap_size()
        if dap_size > 1:
            N_padded = ((original_N + dap_size - 1) // dap_size) * dap_size
            z_full = dap_gather(z_scattered.contiguous(), dim=1, original_size=N_padded)
            if N_padded != original_N:
                z_full = z_full[:, :original_N, :original_N, :]
        else:
            z_full = z_scattered[:, :original_N, :original_N, :]
        return z_full

    def forward(
        self,
        z: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tensor:
        """Forward: z [B, N/dap, N, D], pair_mask [B, N/dap, N]."""
        original_N = z.shape[2]
        dap_rank = get_dap_rank() if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized() else 0
        _save = self._save_subop_checkpoints

        # ── Per-sub-op profiling ──
        _diag = self._diag_enabled and dap_rank == 0
        def _pf_prof(label):
            """Reset peak stats, run op, measure exact transient."""
            pass  # placeholder for pre/post pattern

        def _pre():
            if not _diag:
                return 0
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated(0) // (1024*1024)
            torch.cuda.reset_peak_memory_stats(0)
            return alloc

        def _post(label, alloc_before):
            if not _diag:
                return
            torch.cuda.synchronize()
            alloc_after = torch.cuda.memory_allocated(0) // (1024*1024)
            peak = torch.cuda.max_memory_allocated(0) // (1024*1024)
            transient = peak - alloc_before
            print(f"        [PF-OP] {label:16s} | alloc: {alloc_before}→{alloc_after}MB "
                  f"(Δ{alloc_after - alloc_before:+d}) | peak={peak}MB | TRANSIENT={transient}MB",
                  flush=True)

        def _save_checkpoint(label, z_scat):
            if not _save:
                return
            z_full = self._gather_z_full(z_scat, original_N)
            self._subop_data[label] = z_full.cpu().to(torch.bfloat16)
            if dap_rank == 0:
                zf = z_full.float()
                print(f"        [SUBOP-CKP] {label}: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")
            del z_full

        _save_checkpoint("input", z)

        # 1. TriMulOut: row-scattered
        a0 = _pre()
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask, use_kernels=use_kernels)
        _post("tri_mul_out", a0)
        _save_checkpoint("after_tri_mul_out", z)

        # 2. TriMulIn: col-scattered round-trip
        a0 = _pre()
        z_col = row_to_col(z)
        pair_mask_col = row_to_col(pair_mask.unsqueeze(-1)).squeeze(-1)
        # Zero out padded positions (N_pad > original_N) to prevent them from
        # contributing non-zero values through LayerNorm→projection→einsum
        N_pad = z_col.shape[1]
        if N_pad > original_N:
            z_col[:, original_N:, :, :] = 0
            pair_mask_col[:, original_N:, :] = 0
        dropout = get_dropout_mask(self.dropout, z_col, self.training)
        z_col = z_col + dropout * self.tri_mul_in(z_col, mask=pair_mask_col, use_kernels=use_kernels)
        z = col_to_row(z_col)
        del z_col
        if z.shape[2] > original_N:
            z = z[:, :, :original_N, :]
        _post("tri_mul_in", a0)
        _save_checkpoint("after_tri_mul_in", z)

        # 3. TriAttStart: scattered, gathers only bias (H=4 channels)
        a0 = _pre()
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
        )
        _post("tri_att_start", a0)
        _save_checkpoint("after_tri_att_start", z)

        # 4. TriAttEnd: internally uses row_to_col, gathers only bias
        a0 = _pre()
        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
        )
        _post("tri_att_end", a0)
        _save_checkpoint("after_tri_att_end", z)

        # 5. Transition (pointwise)
        a0 = _pre()
        z = z + self.transition_z(z)
        _post("transition_z", a0)
        _save_checkpoint("after_transition", z)

        return z
