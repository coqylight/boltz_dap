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


def _opm_output_chunk_size(n_output: int) -> int:
    raw = os.environ.get("BOLTZ_OPM_OUTPUT_CHUNK", "").strip()
    if raw == "" or raw == "0":
        return n_output
    try:
        value = int(raw)
    except ValueError:
        return n_output
    if value <= 0:
        return n_output
    return min(n_output, value)


def _opm_row_chunk_size(n_rows: int) -> int:
    raw = os.environ.get("BOLTZ_OPM_ROW_CHUNK", "16").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 16
    if value <= 0:
        return n_rows
    return min(n_rows, value)


def _pwa_q_chunk_size(n_rows: int) -> int:
    raw = os.environ.get("BOLTZ_PWA_Q_CHUNK", "").strip()
    if raw == "" or raw == "0":
        return n_rows
    try:
        value = int(raw)
    except ValueError:
        return n_rows
    if value <= 0:
        return n_rows
    return min(n_rows, value)


def _msa_sequence_chunk_size(num_s: int) -> int:
    raw = os.environ.get("BOLTZ_MSA_SEQUENCE_CHUNK", "").strip()
    if raw == "":
        raw = os.environ.get("BOLTZ_PWA_S_CHUNK", "4").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, min(num_s, value))


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
            reserved = torch.cuda.memory_reserved(0) // (1024*1024)
            peak = torch.cuda.max_memory_allocated(0) // (1024*1024)
            free, total = torch.cuda.mem_get_info(0)
            free_mb = free // (1024*1024)
            total_mb = total // (1024*1024)
            print(f"      [MSA]  alloc= {alloc:5d}MB | reserved= {reserved:5d}MB | "
                  f"free= {free_mb:5d}/{total_mb:5d}MB | peak= {peak:5d}MB | {label}", flush=True)

        _msa_mem("entry")
        if not (self.training and torch.is_grad_enabled()):
            torch.cuda.empty_cache()
            _msa_mem("after entry cache release")

        # 1. pair_weighted_averaging with scattered z bias
        #    Always use the DAP path (scatter/gather are no-ops when dap_size=1)
        #    to ensure bitwise reproducibility between 1-GPU and multi-GPU runs.
        pwa = self.pair_weighted_averaging
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)

        if self.training and torch.is_grad_enabled():
            m_normed = pwa.norm_m(m)
            pwa_out = _pwa_scattered_streamed(pwa, m_normed, z, token_mask)
            del m_normed
            m = m + msa_dropout * pwa_out
            del pwa_out
        else:
            pwa_local = _pwa_scattered_local_eval(pwa, m, z, token_mask)
            _add_pwa_residual_scattered_(m, pwa_local, original_N)
            del pwa_local
        del msa_dropout

        _msa_mem("after PWA")

        if self.training and torch.is_grad_enabled():
            m = m + self.msa_transition(m, chunk_size_transition_msa)
        else:
            _add_msa_transition_streamed_(
                self.msa_transition,
                m,
                chunk_size_transition_msa,
            )

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
        if not (self.training and torch.is_grad_enabled()):
            _msa_mem("before OPM cache release")
            torch.cuda.empty_cache()
            _msa_mem("after OPM cache release")
        opm = self.outer_product_mean
        if self.training and torch.is_grad_enabled():
            opm_scattered = _opm_scattered(
                opm, m, msa_mask, chunk_size_outer_product
            )
        else:
            opm_scattered = _opm_scattered_s_staged(
                opm, m, msa_mask, chunk_size_outer_product
            )
        if self.training and torch.is_grad_enabled():
            z = z + opm_scattered
        else:
            z.add_(opm_scattered)
        del opm_scattered

        _msa_mem("after OPM")

        # Granular checkpoint: z after OPM (ALL ranks must call gather)
        if self._save_gran_ckpts:
            z_full = gather(z.contiguous(), dim=1, original_size=original_N)
            if dap_rank == 0:
                self._gran_ckpt_data["after_opm"] = z_full[:, :original_N, :original_N, :].cpu().to(torch.bfloat16)
            del z_full

        if not (self.training and torch.is_grad_enabled()):
            torch.cuda.empty_cache()
            _msa_mem("after OPM cache release")

        # 4. Pairformer layer (DAP-aware)
        if dap_size > 1:
            pair_mask_scattered = scatter(token_mask, dim=1)
            del token_mask
        else:
            pair_mask_scattered = token_mask

        _msa_mem("before PF layer")

        # ── Measure exact PF transient ──
        if _msa_diag and dap_rank == 0:
            torch.cuda.synchronize()
            _pf_alloc_before = torch.cuda.memory_allocated(0) // (1024*1024)
            _pf_reserved_before = torch.cuda.memory_reserved(0) // (1024*1024)
            _pf_free_before, _pf_total = torch.cuda.mem_get_info(0)
            torch.cuda.reset_peak_memory_stats(0)

        # Enable PF sub-op profiling when MSA diagnostics are active
        self.pairformer_layer._diag_enabled = _msa_diag

        # Force use_kernels=False for MSA PF to match PyTorch-native DAP ops
        _msa_use_kernels = False

        z = self.pairformer_layer(
            z, pair_mask_scattered,
            chunk_size_tri_attn=chunk_size_tri_attn,
            chunk_size_transition_z=chunk_size_transition_z,
            use_kernels=_msa_use_kernels,
        )

        if _msa_diag and dap_rank == 0:
            torch.cuda.synchronize()
            _pf_alloc_after = torch.cuda.memory_allocated(0) // (1024*1024)
            _pf_reserved_after = torch.cuda.memory_reserved(0) // (1024*1024)
            _pf_free_after, _ = torch.cuda.mem_get_info(0)
            _pf_peak = torch.cuda.max_memory_allocated(0) // (1024*1024)
            _pf_transient = _pf_peak - _pf_alloc_before
            print(f"      [MSA-PF] alloc_before={_pf_alloc_before}MB → alloc_after={_pf_alloc_after}MB | "
                f"reserved={_pf_reserved_before}→{_pf_reserved_after}MB | "
                f"free={_pf_free_before // (1024*1024)}→{_pf_free_after // (1024*1024)}MB "
                f"of {_pf_total // (1024*1024)}MB | peak_during_PF={_pf_peak}MB | PF_TRANSIENT={_pf_transient}MB | "
                  f"persistent_delta={_pf_alloc_after - _pf_alloc_before}MB", flush=True)

        _msa_mem("after PF layer")

        # Granular checkpoint: z after PF (ALL ranks must call gather)
        if self._save_gran_ckpts:
            z_full = gather(z.contiguous(), dim=1, original_size=original_N)
            if dap_rank == 0:
                self._gran_ckpt_data["after_pf"] = z_full[:, :original_N, :original_N, :].cpu().to(torch.bfloat16)
            del z_full

        return z, m


def _pwa_s_chunk_size(num_s: int) -> int:
    """MSA depth chunk for PWA proj_o matmul; full matmul if num_s <= chunk."""
    value = os.environ.get("BOLTZ_PWA_S_CHUNK")
    if value is not None:
        return max(1, min(num_s, int(value)))
    return min(num_s, 32)


def _pwa_proj_o_matmul_s_chunked(o_chunks: Tensor, proj_o_weight_t: Tensor) -> Tensor:
    """Compute PWA output projection in independent MSA-depth chunks."""
    s_dim = o_chunks.shape[1]
    chunk_size = _pwa_s_chunk_size(s_dim)
    if chunk_size >= s_dim:
        return o_chunks @ proj_o_weight_t

    parts: list[Tensor] = []
    for s_start in range(0, s_dim, chunk_size):
        s_end = min(s_start + chunk_size, s_dim)
        parts.append(o_chunks[:, s_start:s_end].contiguous() @ proj_o_weight_t)
    return torch.cat(parts, dim=1)


def _pwa_scattered_local(pwa, m_normed: Tensor, z_scattered: Tensor, mask: Tensor) -> Tensor:
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    if dap_size == 1:
        b_full = pwa.proj_z(pwa.norm_z(z_scattered))
        return _pwa_with_bias(pwa, m_normed, b_full, mask, chunk_heads=True)

    batch_size, num_s, original_n, c_m = m_normed.shape
    local_rows = z_scattered.shape[1]
    q_chunk = _pwa_q_chunk_size(local_rows)
    s_chunk = _pwa_s_chunk_size(num_s)
    row_offset = dap_rank * local_rows

    local_out = torch.zeros(
        batch_size,
        num_s,
        local_rows,
        c_m,
        dtype=m_normed.dtype,
        device=m_normed.device,
    )

    for q_start in range(0, local_rows, q_chunk):
        q_end = min(q_start + q_chunk, local_rows)
        global_start = row_offset + q_start
        if global_start >= original_n:
            continue
        valid_end = min(row_offset + q_end, original_n)
        valid_rows = valid_end - global_start

        z_q = z_scattered[:, q_start : q_start + valid_rows, :, :]
        b_q_all = pwa.proj_z(pwa.norm_z(z_q))
        mask_q = mask[:, global_start:valid_end, :]

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

            b_q = b_q_all[..., head_idx]
            b_q = b_q + (1 - mask_q) * -pwa.inf
            w = torch.softmax(b_q[:, None, :, :], dim=-1)

            for s_start in range(0, num_s, s_chunk):
                s_end = min(s_start + s_chunk, num_s)
                m_s = m_normed[:, s_start:s_end, :, :]

                v_s = m_s @ sliced_weight_proj_m.T
                v_s = v_s.reshape(m_s.shape[0], m_s.shape[1], m_s.shape[2], 1, pwa.c_h)
                v_s = v_s.permute(0, 3, 1, 2, 4)

                g_s = m_s @ sliced_weight_proj_g.T
                g_s = g_s.sigmoid()
                g_s = g_s[:, :, global_start:valid_end, :]

                o_s = torch.einsum("bhij,bhsjd->bhsid", w, v_s)
                o_s = o_s.permute(0, 2, 3, 1, 4)
                o_s = o_s.reshape(o_s.shape[0], o_s.shape[1], o_s.shape[2], pwa.c_h)

                part = (g_s * o_s) @ sliced_weight_proj_o.T
                local_out[:, s_start:s_end, q_start : q_start + valid_rows, :].add_(part)
                del part, o_s, g_s, v_s
            del w, b_q
        del b_q_all, mask_q, z_q

    return local_out


def _pwa_scattered_local_eval(pwa, m: Tensor, z_scattered: Tensor, mask: Tensor) -> Tensor:
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    if dap_size == 1:
        return _pwa_scattered_local(pwa, pwa.norm_m(m), z_scattered, mask)

    batch_size, num_s, original_n, c_m = m.shape
    local_rows = z_scattered.shape[1]
    row_offset = dap_rank * local_rows
    valid_rows = max(0, min(local_rows, original_n - row_offset))
    s_chunk = _msa_sequence_chunk_size(num_s)

    z_local = z_scattered[:, :valid_rows]
    bias = pwa.proj_z(pwa.norm_z(z_local))
    mask_local = mask[:, row_offset : row_offset + valid_rows]
    bias.add_((1 - mask_local).unsqueeze(-1) * -pwa.inf)
    weights = torch.softmax(bias.permute(0, 3, 1, 2), dim=-1)
    del bias, mask_local, z_local

    local_out = torch.zeros(
        batch_size,
        num_s,
        local_rows,
        c_m,
        dtype=m.dtype,
        device=m.device,
    )
    for s_start in range(0, num_s, s_chunk):
        s_end = min(s_start + s_chunk, num_s)
        m_s = pwa.norm_m(m[:, s_start:s_end])
        for head_idx in range(pwa.num_heads):
            proj_m_weight = pwa.proj_m.weight[
                head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h
            ]
            proj_g_weight = pwa.proj_g.weight[
                head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h
            ]
            proj_o_weight = pwa.proj_o.weight[
                :, head_idx * pwa.c_h : (head_idx + 1) * pwa.c_h
            ]

            values = m_s @ proj_m_weight.T
            values = values[:, None]
            gates = (m_s @ proj_g_weight.T).sigmoid()
            gates = gates[:, :, row_offset : row_offset + valid_rows]
            output = torch.einsum(
                "bhij,bhsjd->bhsid",
                weights[:, head_idx : head_idx + 1],
                values,
            )
            output = output[:, 0]
            part = (gates * output) @ proj_o_weight.T
            local_out[
                :, s_start:s_end, :valid_rows
            ].add_(part)
            del gates, output, part, values
        del m_s
    del weights
    return local_out


def _pwa_scattered_streamed(pwa, m_normed: Tensor, z_scattered: Tensor, mask: Tensor) -> Tensor:
    """Run DAP PWA by streaming local query rows and gathering output rows."""
    local_out = _pwa_scattered_local(pwa, m_normed, z_scattered, mask)
    if get_dap_size() == 1:
        return local_out

    original_n = m_normed.shape[2]
    gathered = gather(local_out.permute(0, 2, 1, 3).contiguous(), dim=1, original_size=original_n)
    return gathered.permute(0, 2, 1, 3)


def _add_pwa_residual_scattered_(m: Tensor, local_out: Tensor, original_n: int) -> None:
    if get_dap_size() == 1:
        m.add_(local_out)
        return

    s_chunk = _msa_sequence_chunk_size(m.shape[1])
    for s_start in range(0, m.shape[1], s_chunk):
        s_end = min(s_start + s_chunk, m.shape[1])
        local_chunk = local_out[:, s_start:s_end].permute(0, 2, 1, 3).contiguous()
        full_chunk = gather(local_chunk, dim=1, original_size=original_n)
        m[:, s_start:s_end].add_(full_chunk.permute(0, 2, 1, 3))
        del local_chunk, full_chunk


def _add_msa_transition_streamed_(transition, m: Tensor, chunk_size: int | None) -> None:
    s_chunk = _msa_sequence_chunk_size(m.shape[1])
    for s_start in range(0, m.shape[1], s_chunk):
        s_end = min(s_start + s_chunk, m.shape[1])
        m_chunk = m[:, s_start:s_end]
        update = transition(m_chunk, chunk_size)
        m_chunk.add_(update)
        del update


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

            w = torch.softmax(b_full_perm[:, head_idx:head_idx+1], dim=-1)

            num_s = m_normed.shape[1]
            s_chunk = _pwa_s_chunk_size(num_s)
            block: Tensor | None = None
            for s_start in range(0, num_s, s_chunk):
                s_end = min(s_start + s_chunk, num_s)
                m_s = m_normed[:, s_start:s_end, :, :]

                v_s = m_s @ sliced_weight_proj_m.T
                v_s = v_s.reshape(m_s.shape[0], m_s.shape[1], m_s.shape[2], 1, pwa.c_h)
                v_s = v_s.permute(0, 3, 1, 2, 4)

                g_s = m_s @ sliced_weight_proj_g.T
                g_s = g_s.sigmoid()

                o_s = torch.einsum("bhij,bhsjd->bhsid", w, v_s)
                o_s = o_s.permute(0, 2, 3, 1, 4)
                o_s = o_s.reshape(o_s.shape[0], o_s.shape[1], o_s.shape[2], pwa.c_h)

                o_chunks_s = g_s * o_s
                part = _pwa_proj_o_matmul_s_chunked(o_chunks_s, sliced_weight_proj_o.T)
                if block is None:
                    batch_size, _, n_tokens, out_dim = part.shape
                    block = torch.empty(
                        batch_size,
                        num_s,
                        n_tokens,
                        out_dim,
                        dtype=part.dtype,
                        device=part.device,
                    )
                block[:, s_start:s_end].copy_(part)
                del part
            assert block is not None
            if head_idx == 0:
                o_out = block
            else:
                o_out.add_(block)
            del block
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


def _opm_scattered_s_staged(opm, m: Tensor, mask: Tensor, chunk_size: int | None) -> Tensor:
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    batch_size, num_s, num_tokens, _ = m.shape
    local_rows = (num_tokens + dap_size - 1) // dap_size
    row_start = dap_rank * local_rows
    row_end = min(row_start + local_rows, num_tokens)
    valid_rows = max(0, row_end - row_start)
    s_chunk = _msa_sequence_chunk_size(num_s)

    a_cpu = torch.empty(
        batch_size,
        num_s,
        num_tokens,
        opm.c_hidden,
        dtype=m.dtype,
        device="cpu",
    )
    b = torch.empty(
        batch_size,
        num_s,
        num_tokens,
        opm.c_hidden,
        dtype=m.dtype,
        device=m.device,
    )
    for s_start in range(0, num_s, s_chunk):
        s_end = min(s_start + s_chunk, num_s)
        mask_chunk = mask[:, s_start:s_end].unsqueeze(-1).to(m)
        m_normed = opm.norm(m[:, s_start:s_end])
        a_chunk = opm.proj_a(m_normed).mul_(mask_chunk)
        b_chunk = opm.proj_b(m_normed).mul_(mask_chunk)
        a_cpu[:, s_start:s_end].copy_(a_chunk, non_blocking=False)
        b[:, s_start:s_end].copy_(b_chunk)
        del a_chunk, b_chunk, m_normed, mask_chunk

    mask_exp = mask.unsqueeze(-1).to(m)
    num_mask = None
    for s_start in range(0, num_s, 64):
        s_end = min(s_start + 64, num_s)
        mask_b = mask_exp[:, s_start:s_end]
        mask_a = mask_b[:, :, row_start:row_end]
        if valid_rows < local_rows:
            padded_mask_a = torch.zeros(
                batch_size,
                s_end - s_start,
                local_rows,
                1,
                dtype=m.dtype,
                device=m.device,
            )
            padded_mask_a[:, :, :valid_rows].copy_(mask_a)
            mask_a = padded_mask_a
        cross_sum = (
            mask_a[:, :, :, None, :] * mask_b[:, :, None, :, :]
        ).sum(1)
        if num_mask is None:
            num_mask = cross_sum
        else:
            num_mask.add_(cross_sum)
        del cross_sum
    assert num_mask is not None
    num_mask.clamp_(min=1)

    hidden_chunk = chunk_size or opm.c_hidden
    hidden_chunk = max(1, min(hidden_chunk, opm.c_hidden))
    output_chunk = _opm_output_chunk_size(opm.proj_o.out_features)
    row_chunk = _opm_row_chunk_size(local_rows)
    z_out = None
    for hidden_start in range(0, opm.c_hidden, hidden_chunk):
        hidden_end = min(hidden_start + hidden_chunk, opm.c_hidden)
        a_full = a_cpu[..., hidden_start:hidden_end].to(
            device=m.device,
            non_blocking=False,
        )
        a_local = a_full[:, :, row_start:row_end]
        if valid_rows < local_rows:
            padded_a_local = torch.zeros(
                batch_size,
                num_s,
                local_rows,
                hidden_end - hidden_start,
                dtype=m.dtype,
                device=m.device,
            )
            padded_a_local[:, :, :valid_rows].copy_(a_local)
            a_local = padded_a_local

        sliced_weight = opm.proj_o.weight[
            :,
            hidden_start * opm.c_hidden : hidden_end * opm.c_hidden,
        ]
        for row_chunk_start in range(0, local_rows, row_chunk):
            row_chunk_end = min(row_chunk_start + row_chunk, local_rows)
            z_tile = torch.einsum(
                "bsic,bsjd->bijcd",
                a_local[:, :, row_chunk_start:row_chunk_end],
                b,
            )
            z_tile = z_tile.reshape(*z_tile.shape[:3], -1)
            z_tile.div_(num_mask[:, row_chunk_start:row_chunk_end])
            if z_out is None:
                z_out = torch.empty(
                    batch_size,
                    local_rows,
                    num_tokens,
                    opm.proj_o.out_features,
                    dtype=z_tile.dtype,
                    device=z_tile.device,
                )
            for output_start in range(0, opm.proj_o.out_features, output_chunk):
                output_end = min(output_start + output_chunk, opm.proj_o.out_features)
                z_proj = z_tile @ sliced_weight[output_start:output_end].T
                z_out_tile = z_out[
                    :,
                    row_chunk_start:row_chunk_end,
                    :,
                    output_start:output_end,
                ]
                if hidden_start == 0:
                    z_out_tile.copy_(z_proj)
                else:
                    z_out_tile.add_(z_proj)
                del z_proj
            del z_tile
        del a_full, a_local

    assert z_out is not None
    z_out.add_(opm.proj_o.bias.to(dtype=z_out.dtype, device=z_out.device))
    return z_out


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
        output_chunk = _opm_output_chunk_size(opm.proj_o.out_features)
        z_out = None
        for i in range(0, opm.c_hidden, chunk_size):
            a_chunk = a_scattered[:, :, :, i:i+chunk_size]
            sliced_weight = opm.proj_o.weight[
                :, i * opm.c_hidden : (i + chunk_size) * opm.c_hidden
            ]
            # einsum: [B,S,N/dap,c_chunk] x [B,S,N,c_h] → [B,N/dap,N,c_chunk*c_h]
            z = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
            z = z.reshape(*z.shape[:3], -1)
            z = z / num_mask
            z = z.to(m)

            if z_out is None:
                z_out = torch.empty(
                    *z.shape[:3],
                    opm.proj_o.out_features,
                    dtype=z.dtype,
                    device=z.device,
                )

            for o in range(0, opm.proj_o.out_features, output_chunk):
                o_end = min(o + output_chunk, opm.proj_o.out_features)
                z_proj = z @ sliced_weight[o:o_end].T
                if i == 0:
                    z_out[..., o:o_end] = z_proj
                else:
                    z_out[..., o:o_end].add_(z_proj)
                del z_proj
            del z

        z_out.add_(opm.proj_o.bias.to(dtype=z_out.dtype, device=z_out.device))
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

