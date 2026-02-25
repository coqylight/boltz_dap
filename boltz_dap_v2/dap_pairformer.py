"""
DAP-aware PairformerLayer for Boltz 2 (main pairformer with sequence attention).

Flow per layer (row-scattered z [B, N/dap, N, D]):
  z → tri_mul_out(z)         — row-scattered, DAP-wrapped
  z → row_to_col → z_col → tri_mul_in(z_col) → col_to_row → z
  z → DAPTriAttStart(z)      — scattered, gathers only small bias
  z → DAPTriAttEnd(z)        — internally uses row_to_col for ending
  z → transition_z(z)        — pointwise
  s, z → seq_attention       — gathers only pair bias (H channels), not full z
"""

import torch
from torch import Tensor, nn
from typing import Optional
import pathlib

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import row_to_col, col_to_row, gather, scatter
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_trimul import DAPTriMulOut, DAPTriMulIn
from dap_tri_att import DAPTriAttStart, DAPTriAttEnd
from dap_pairformer_noseq import get_dropout_mask


class DAPPairformerLayer(nn.Module):
    """DAP wrapper for PairformerLayer (with sequence attention).

    z stays row-scattered [B, N/dap, N, D] throughout pair ops.
    Sequence attention gathers only the H-channel pair bias, not full z.
    """

    def __init__(self, original_layer):
        super().__init__()
        self.tri_mul_out = DAPTriMulOut(original_layer.tri_mul_out)
        self.tri_mul_in = DAPTriMulIn(original_layer.tri_mul_in)
        self.tri_att_start = DAPTriAttStart(original_layer.tri_att_start)
        self.tri_att_end = DAPTriAttEnd(original_layer.tri_att_end)
        self.transition_z = original_layer.transition_z

        # Sequence attention (uses pair bias from z)
        self.pre_norm_s = original_layer.pre_norm_s
        self.attention = original_layer.attention
        self.transition_s = original_layer.transition_s
        self.s_post_norm = original_layer.s_post_norm

        self.dropout = original_layer.dropout

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
        layer_idx: int = -1,
    ) -> tuple[Tensor, Tensor]:
        """Forward.

        s: [B, N, 384] — replicated
        z: [B, N/dap, N, D] — row-scattered
        mask: [B, N] — replicated
        pair_mask: [B, N/dap, N] — row-scattered
        """
        dap_size = get_dap_size()
        dap_rank = get_dap_rank()
        original_N = z.shape[2]

        # Sub-op checkpointing: only for layer 0
        # NOTE: gather() is NCCL collective — ALL ranks must call it!
        _subop_dir = os.environ.get("BOLTZ_SAVE_SUBOP_CKPT", "")
        _do_subop = bool(_subop_dir) and layer_idx == 0
        if layer_idx == 0 and dap_rank == 0:
            print(f"[SUBOP-DEBUG] layer_idx={layer_idx} _subop_dir='{_subop_dir}' _do_subop={_do_subop}", flush=True)
        if _do_subop and dap_rank == 0:
            _subop_path = pathlib.Path(_subop_dir)
            _subop_path.mkdir(parents=True, exist_ok=True)

        def _save_z(name):
            """Gather scattered z (collective!) and save on rank 0."""
            if not _do_subop:
                return
            # ALL ranks must call gather (NCCL collective)
            z_full = gather(z.contiguous(), dim=1, original_size=original_N)
            if dap_rank == 0:
                torch.save(z_full.detach().cpu(), pathlib.Path(_subop_dir) / f"{name}.pt")
            del z_full

        def _mem(label):
            pass  # Disabled: use [TIMELINE] logs in dap_trunk.py instead

        _mem("start")

        # === Pair operations (all on scattered z) ===

        # 1. TriMulOut
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask, use_kernels=use_kernels)
        _mem("after tri_mul_out")
        _save_z("after_trimul_out")

        # 2. TriMulIn (col-scattered round-trip)
        z_col = row_to_col(z)
        pair_mask_col = row_to_col(pair_mask.unsqueeze(-1)).squeeze(-1)
        dropout = get_dropout_mask(self.dropout, z_col, self.training)
        z_col = z_col + dropout * self.tri_mul_in(z_col, mask=pair_mask_col, use_kernels=use_kernels)
        z = col_to_row(z_col)
        del z_col
        if z.shape[2] > original_N:
            z = z[:, :, :original_N, :]
        _mem("after tri_mul_in")
        _save_z("after_trimul_in")

        # 3. TriAttStart (scattered, gathers only bias)
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
        )
        _mem("after tri_att_start")
        _save_z("after_triatt_start")

        # 4. TriAttEnd (internally handles row_to_col)
        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z, mask=pair_mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
        )
        _mem("after tri_att_end")
        _save_z("after_triatt_end")

        # 5. Transition (pointwise, chunked to avoid 4×D expansion spike)
        z = z + self.transition_z(z)
        _mem("after transition_z")

        # === Sequence attention (gather only bias, not full z) ===
        # proj_z: z [B, I, J, D] → bias [B, H, I, J] (LayerNorm + Linear(D→H) + Rearrange)
        # Compute bias on scattered z, then gather only H channels (vs D=128)
        if dap_size > 1:
            # z is [B, N/dap, N, D] → proj_z → [B, H, N/dap, N]
            pair_bias = self.attention.proj_z(z)
            # Gather along dim=2 (the scattered I dimension, now after rearrange to H-first)
            pair_bias = gather(pair_bias.contiguous(), dim=2, original_size=original_N)
            # pair_bias is now [B, H, N, N] — full bias, no full z needed!
        else:
            pair_bias = self.attention.proj_z(z)
        _mem("after proj_z+gather")

        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(s.float())

            # Call attention with pre-computed bias (skip internal proj_z)
            B = s_normed.shape[0]
            attn_mod = self.attention
            q = attn_mod.proj_q(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
            k = attn_mod.proj_k(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
            v = attn_mod.proj_v(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
            g = attn_mod.proj_g(s_normed).sigmoid()

            with torch.autocast("cuda", enabled=False):
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (attn_mod.head_dim ** 0.5) + pair_bias.float()
                attn = attn + (1 - mask[:, None, None].float()) * -attn_mod.inf
                attn = attn.softmax(dim=-1)
                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)

            del pair_bias
            o = o.reshape(B, -1, attn_mod.c_s)
            s = s.float() + attn_mod.proj_o(g * o)
            s = s + self.transition_s(s)
            s = self.s_post_norm(s)

        _mem("after seq_attn")
        self._logged = True

        if _do_subop:
            # Clear env var so only first recycling step saves
            os.environ.pop("BOLTZ_SAVE_SUBOP_CKPT", None)

        # z stays scattered
        return s, z
