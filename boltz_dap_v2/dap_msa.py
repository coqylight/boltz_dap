"""
DAP-aware MSALayer for Boltz 2.

The MSALayer contains:
1. pair_weighted_averaging(m, z, mask) — uses z for attention weights
2. msa_transition(m)
3. outer_product_mean(m, msa_mask) — produces z-shaped output
4. pairformer_layer (PairformerNoSeqLayer) — DAP-wrapped

Optimizations in this version:
- PWA: gather only the 8-channel bias (proj_z output), NOT full z
- OPM: scatter a on position dim, keep b full → output naturally scattered
"""

import torch
from torch import Tensor, nn
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import gather, scatter
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_pairformer_noseq import DAPPairformerNoSeqLayer, get_dropout_mask


class DAPMSALayer(nn.Module):
    """DAP wrapper for MSALayer.

    Accepts z in row-scattered form [B, N/dap, N, D].
    Returns (z, m) where z is row-scattered.
    """

    def __init__(self, original_layer):
        """Wrap an existing MSALayer."""
        super().__init__()
        # MSA-specific ops
        self.pair_weighted_averaging = original_layer.pair_weighted_averaging
        self.msa_transition = original_layer.msa_transition
        self.outer_product_mean = original_layer.outer_product_mean
        self.msa_dropout = original_layer.msa_dropout
        self._diag_enabled = False  # toggled by trunk for memory profiling

        # Wrap pairformer with DAP
        self.pairformer_layer = DAPPairformerNoSeqLayer(original_layer.pairformer_layer)

        # Granular checkpoint support
        self._save_gran_ckpts = False
        self._gran_ckpt_data = {}

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
        use_kernels: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with DAP.

        z: [B, N/dap, N, D] — row-scattered
        m: [B, S, N, msa_s] — replicated (MSA sequences, small)
        token_mask: [B, N, N] — full pair mask (replicated)
        msa_mask: [B, S] — replicated
        """
        dap_size = get_dap_size()
        dap_rank = get_dap_rank()
        original_N = z.shape[2]  # full N from the non-scattered dim

        # ── Fine-grained memory logging ──
        _msa_diag = getattr(self, '_diag_enabled', False)
        def _msa_mem(label):
            if not _msa_diag or dap_rank != 0:
                return
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated(0) // (1024*1024)
            peak = torch.cuda.max_memory_allocated(0) // (1024*1024)
            print(f"      [MSA]  alloc= {alloc:5d}MB | peak= {peak:5d}MB | {label}", flush=True)

        _msa_mem("entry")

        # 1. pair_weighted_averaging with scattered z bias
        #    Always use the DAP path (scatter/gather are no-ops when dap_size=1)
        #    to ensure bitwise reproducibility between 1-GPU and multi-GPU runs.
        pwa = self.pair_weighted_averaging
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)

        # Compute proj_z on scattered z → [B, N/dap, N, H]
        # Then gather only the small bias, not full z
        z_normed_scattered = pwa.norm_z(z)
        b_scattered = pwa.proj_z(z_normed_scattered)  # [B, N/dap, N, H=8]
        del z_normed_scattered
        b_full = gather(b_scattered.contiguous(), dim=1,
                       original_size=original_N)  # [B, N, N, 8] — tiny!
        del b_scattered

        # Run PWA manually with pre-computed bias
        m_normed = pwa.norm_m(m)
        pwa_out = _pwa_with_bias(pwa, m_normed, b_full, token_mask,
                                  chunk_heads_pwa)
        del b_full
        m = m + msa_dropout * pwa_out
        del pwa_out

        _msa_mem("after PWA")

        # 2. MSA transition (pointwise on m, no z involved)
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        _msa_mem("after MSA transition")

        # Granular checkpoint: m after PWA + transition (before OPM)
        if self._save_gran_ckpts:
            if dap_rank == 0:
                self._gran_ckpt_data["after_pwa_and_transition_m"] = m.detach().cpu().to(torch.bfloat16)
                # Also save z before OPM (already scattered, gather it)
            z_full_pre = gather(z.contiguous(), dim=1, original_size=original_N) if dap_size > 1 else z
            if dap_rank == 0:
                self._gran_ckpt_data["before_opm_z"] = z_full_pre[:, :original_N, :original_N, :].detach().cpu().to(torch.bfloat16)
            del z_full_pre

        # 3. outer_product_mean — scattered computation (no full [B,N,N,C] on any GPU)
        #    Always use _opm_scattered for bitwise reproducibility
        #    (scatter/gather are no-ops when dap_size=1).
        opm = self.outer_product_mean
        opm_scattered = _opm_scattered(
            opm, m, msa_mask, chunk_size_outer_product
        )
        z = z + opm_scattered
        del opm_scattered

        _msa_mem("after OPM")

        # Granular checkpoint: z after OPM (ALL ranks must call gather)
        if self._save_gran_ckpts:
            z_full = gather(z.contiguous(), dim=1, original_size=original_N)
            if dap_rank == 0:
                self._gran_ckpt_data["after_opm"] = z_full[:, :original_N, :original_N, :].cpu().to(torch.bfloat16)
            del z_full

        # 4. Pairformer layer (DAP-aware)
        if dap_size > 1:
            pair_mask_scattered = scatter(token_mask, dim=1)
        else:
            pair_mask_scattered = token_mask

        # ── Measure exact PF transient ──
        if _msa_diag and dap_rank == 0:
            torch.cuda.synchronize()
            _pf_alloc_before = torch.cuda.memory_allocated(0) // (1024*1024)
            torch.cuda.reset_peak_memory_stats(0)

        # Enable PF sub-op profiling when MSA diagnostics are active
        self.pairformer_layer._diag_enabled = _msa_diag

        # Force use_kernels=False for MSA PF to match PyTorch-native DAP ops
        _msa_use_kernels = False

        z = self.pairformer_layer(
            z, pair_mask_scattered,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=_msa_use_kernels,
        )

        if _msa_diag and dap_rank == 0:
            torch.cuda.synchronize()
            _pf_alloc_after = torch.cuda.memory_allocated(0) // (1024*1024)
            _pf_peak = torch.cuda.max_memory_allocated(0) // (1024*1024)
            _pf_transient = _pf_peak - _pf_alloc_before
            print(f"      [MSA-PF] alloc_before={_pf_alloc_before}MB → alloc_after={_pf_alloc_after}MB | "
                  f"peak_during_PF={_pf_peak}MB | PF_TRANSIENT={_pf_transient}MB | "
                  f"persistent_delta={_pf_alloc_after - _pf_alloc_before}MB", flush=True)

        _msa_mem("after PF layer")

        # Granular checkpoint: z after PF (ALL ranks must call gather)
        if self._save_gran_ckpts:
            z_full = gather(z.contiguous(), dim=1, original_size=original_N)
            if dap_rank == 0:
                self._gran_ckpt_data["after_pf"] = z_full[:, :original_N, :original_N, :].cpu().to(torch.bfloat16)
            del z_full

        return z, m


def _pwa_with_bias(pwa, m_normed, b_full, mask, chunk_heads):
    """Run PairWeightedAveraging with pre-computed z bias.

    Since we already computed b = proj_z(norm_z(z)) and gathered it,
    we skip the norm_z/proj_z inside PWA and just use b_full directly.

    m_normed: [B, S, N, msa_s] — already normed
    b_full: [B, N, N, H] — pre-computed attention bias
    """
    if chunk_heads and not pwa.training:
        # Sequential head computation
        b_full_perm = b_full.permute(0, 3, 1, 2)  # [B, H, N, N]
        b_full_perm = b_full_perm + (1 - mask[:, None]) * -pwa.inf

        for head_idx in range(pwa.num_heads):
            sliced_weight_proj_m = pwa.proj_m.weight[
                head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h, :
            ]
            sliced_weight_proj_g = pwa.proj_g.weight[
                head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h, :
            ]
            sliced_weight_proj_o = pwa.proj_o.weight[
                :, head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h
            ]

            v = m_normed @ sliced_weight_proj_m.T
            v = v.reshape(*v.shape[:3], 1, pwa.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            w = torch.softmax(b_full_perm[:, head_idx:head_idx+1], dim=-1)

            g = m_normed @ sliced_weight_proj_g.T
            g = g.sigmoid()

            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], 1 * pwa.c_h)
            o_chunks = g * o
            if head_idx == 0:
                o_out = o_chunks @ sliced_weight_proj_o.T
            else:
                o_out += o_chunks @ sliced_weight_proj_o.T
        return o_out
    else:
        # All heads at once
        v = pwa.proj_m(m_normed)
        v = v.reshape(*v.shape[:3], pwa.num_heads, pwa.c_h)
        v = v.permute(0, 3, 1, 2, 4)

        b = b_full.permute(0, 3, 1, 2)  # [B, H, N, N]
        b = b + (1 - mask[:, None]) * -pwa.inf
        w = torch.softmax(b, dim=-1)

        g = pwa.proj_g(m_normed)
        g = g.sigmoid()

        o = torch.einsum("bhij,bhsjd->bhsid", w, v)
        o = o.permute(0, 2, 3, 1, 4)
        o = o.reshape(*o.shape[:3], pwa.num_heads * pwa.c_h)
        o = pwa.proj_o(g * o)
        return o


def _opm_scattered(opm, m, mask, chunk_size):
    """Run OuterProductMean with row-scattered output.

    Scatter `a` on position dim, keep `b` full, so einsum produces
    [B, N/dap, N, c_hidden*c_hidden] directly — no full [B, N, N, C]
    tensor is ever allocated.

    m:    [B, S, N, c_in]  — replicated on all ranks
    mask: [B, S, N]        — MSA mask (per-sequence, per-position)
    Returns: [B, N/dap, N, c_out] — row-scattered
    """
    # Expand mask: [B, S, N] → [B, S, N, 1]
    mask_exp = mask.unsqueeze(-1).to(m)

    # Compute projections on full m (replicated)
    m_normed = opm.norm(m)
    a = opm.proj_a(m_normed) * mask_exp  # [B, S, N, c_hidden]
    b = opm.proj_b(m_normed) * mask_exp  # [B, S, N, c_hidden]
    del m_normed

    # Scatter a AND mask on position dim (dim=2, the N dimension)
    # This gives each GPU its local rows of a and the corresponding mask
    a_scattered = scatter(a, dim=2)      # [B, S, N/dap, c_hidden]
    mask_a = scatter(mask_exp, dim=2)    # [B, S, N/dap, 1]
    del a
    # b and mask_b stay full (all j-columns needed)
    mask_b = mask_exp                    # [B, S, N, 1]

    if chunk_size is not None and not opm.training:
        # Compute num_mask_scattered from mask_a × mask_b
        # num_mask[b, i_local, j] = sum_s(mask[b,s,i_local] * mask[b,s,j])
        for i in range(0, mask_a.shape[1], 64):
            chunk_ma = mask_a[:, i : i + 64, :, :]   # [B, 64, N/dap, 1]
            chunk_mb = mask_b[:, i : i + 64, :, :]   # [B, 64, N, 1]
            cross = chunk_ma[:, :, :, None, :] * chunk_mb[:, :, None, :, :]
            # cross: [B, 64, N/dap, N, 1]
            if i == 0:
                num_mask = cross.sum(1)              # [B, N/dap, N, 1]
            else:
                num_mask += cross.sum(1)
            del cross
        num_mask = num_mask.clamp(min=1)

        # Compute in chunks over c_hidden (same as original OPM)
        for i in range(0, opm.c_hidden, chunk_size):
            a_chunk = a_scattered[:, :, :, i:i+chunk_size]
            sliced_weight = opm.proj_o.weight[
                :, i * opm.c_hidden : (i + chunk_size) * opm.c_hidden
            ]
            # einsum: [B,S,N/dap,c_chunk] x [B,S,N,c_h] → [B,N/dap,N,c_chunk*c_h]
            z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask
            if i == 0:
                z_out = z.to(m) @ sliced_weight.T
            else:
                z_out = z_out + z.to(m) @ sliced_weight.T
        z_out = z_out + opm.proj_o.bias
        return z_out
    else:
        # Non-chunked path — use float32 like original
        # num_mask from mask_a_scattered × mask_b_full
        cross = mask_a[:, :, :, None, :] * mask_b[:, :, None, :, :]
        # cross: [B, S, N/dap, N, 1]
        num_mask = cross.sum(1).clamp(min=1)  # [B, N/dap, N, 1]
        del cross

        z = torch.einsum("bsic,bsjd->bijcd", a_scattered.float(), b.float())
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask
        z = opm.proj_o(z.to(m))
        return z

