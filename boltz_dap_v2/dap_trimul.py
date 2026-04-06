"""
DAP-aware Triangle Multiplication for Boltz 2.

v2: Uses broadcast-based chunking instead of all-gather.
Each rank broadcasts its local shard one at a time, so we never
materialise the full [B, N, N, D] tensor on any single GPU.

Key pattern:
- TriMulOut: z is row-scattered [B, N/dap, N, D]
  • broadcast b from each rank → compute partial einsum for that j-range
  • output is row-scattered [B, N/dap, N, D]

- TriMulIn: z is col-scattered [B, N, N/dap, D]
  • broadcast a from each rank → compute partial einsum for that i-range
  • output is col-scattered [B, N, N/dap, D]

Optional second-axis micro-chunking (same math as one einsum; lowers peak VRAM):
  BOLTZ_TRIMUL_MICRO_J — max j-width per einsum inside each rank block (TriMulOut)
  BOLTZ_TRIMUL_MICRO_I — max i-height per einsum inside each rank block (TriMulIn)
  Unset → min(span, 128). Set to a large value to effectively disable.
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.core import get_dap_size, get_dap_rank, is_dap_initialized

from boltz.model.layers import initialize as init

# ── Profiling helpers ──────────────────────────────────────────────
def _m():
    return torch.cuda.memory_allocated() / (1024**2)

def _p():
    return torch.cuda.max_memory_allocated() / (1024**2)

DIAG = True  # set False to silence profiling

def _log(prefix, tag, a0=None):
    if not DIAG:
        return _m()
    a = _m(); p = _p()
    delta = f"Δ{a - a0:+.0f}" if a0 is not None else ""
    print(f"          [{prefix}] {tag:40s} | alloc={a:8.0f}MB | peak={p:8.0f}MB | {delta}", flush=True)
    return a


def _micro_chunk_size(axis: str, span: int) -> int:
    """Sub-chunk length along j (out) or i (in) within one DAP rank block [1, span].

    axis: 'j' for TriangleMultiplicationOutgoing, 'i' for Incoming.
    """
    if span <= 0:
        return 1
    key = f"BOLTZ_TRIMUL_MICRO_{axis.upper()}"
    raw = os.environ.get(key)
    if raw is not None and str(raw).strip() != "":
        return max(1, min(span, int(raw)))
    return max(1, min(span, 128))


class DAPTriMulOut(nn.Module):
    """Triangle Multiplication Outgoing with DAP (broadcast-chunked).

    Input:  z row-scattered [B, N/dap, N, D]
    Output: z row-scattered [B, N/dap, N, D]
    """

    def __init__(self, original_module):
        super().__init__()
        self.inner = original_module

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, use_kernels=use_kernels)

        dap_rank = get_dap_rank()
        P = "TRI-MUL-OUT"
        # x: [B, N/dap, N, D] (row-scattered)
        B, N_local, N_full, D_z = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        # ── Projections ──
        x_normed = self.inner.norm_in(x)
        x_in = x_normed
        x_proj = self.inner.p_in(x_normed) * self.inner.g_in(x_normed).sigmoid()
        x_proj = x_proj * mask.unsqueeze(-1)
        a, b = torch.chunk(x_proj.float(), 2, dim=-1)
        del x_proj, x_normed
        a0 = _log(P, "after projections + chunk", a0)

        # ── Gating (computed while b is still local) ──
        g_out = self.inner.g_out(x_in).sigmoid()
        del x_in
        a0 = _log(P, "after g_out", a0)

        # ── Broadcast-chunked einsum ──
        D = a.shape[-1]
        out = torch.zeros(B, N_local, N_full, D, dtype=a.dtype, device=a.device)
        b_contig = b.contiguous()
        b_recv = torch.empty_like(b)
        a0 = _log(P, "after alloc out + buffers", a0)

        for src in range(dap_size):
            if src == dap_rank:
                b_chunk = b_contig
                dist.broadcast(b_chunk, src=src)  # send ours
            else:
                dist.broadcast(b_recv, src=src)   # receive theirs
                b_chunk = b_recv

            j_start = src * N_local
            j_end = min(j_start + N_local, N_full)
            j_span = j_end - j_start
            micro_j = _micro_chunk_size("j", j_span)
            for jj in range(j_start, j_end, micro_j):
                jj_e = min(jj + micro_j, j_end)
                r0 = jj - j_start
                r1 = jj_e - j_start
                b_sub = b_chunk[:, r0:r1, :, :]
                out[:, :, jj:jj_e, :] = torch.einsum(
                    "bikd,bjkd->bijd", a, b_sub
                )
            a0 = _log(P, f"einsum src={src} j=[{j_start}:{j_end}] micro={micro_j}", a0)

        del a, b, b_contig, b_recv
        a0 = _log(P, "after cleanup", a0)

        out = self.inner.p_out(self.inner.norm_out(out)) * g_out
        _log(P, "after output gating", a0)
        return out


class DAPTriMulIn(nn.Module):
    """Triangle Multiplication Incoming with DAP (broadcast-chunked).

    Input:  z col-scattered [B, N, N/dap, D]
    Output: z col-scattered [B, N, N/dap, D]
    """

    def __init__(self, original_module):
        super().__init__()
        self.inner = original_module

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, use_kernels=use_kernels)

        dap_rank = get_dap_rank()
        P = "TRI-MUL-IN "
        # x: [B, N, N/dap, D] (col-scattered)
        B, N_full, N_local, D_z = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        # ── Projections ──
        x_normed = self.inner.norm_in(x)
        x_in = x_normed
        x_proj = self.inner.p_in(x_normed) * self.inner.g_in(x_normed).sigmoid()
        x_proj = x_proj * mask.unsqueeze(-1)
        a, b = torch.chunk(x_proj.float(), 2, dim=-1)
        del x_proj, x_normed
        a0 = _log(P, "after projections + chunk", a0)

        # ── Gating ──
        g_out = self.inner.g_out(x_in).sigmoid()
        del x_in
        a0 = _log(P, "after g_out", a0)

        # ── Broadcast-chunked einsum ──
        # einsum "bkid,bkjd->bijd": contract over k=N_full
        # a has i=N_local (scattered cols), b has j=N_local (scattered cols)
        # We need full i range → broadcast a from each rank
        D = a.shape[-1]
        out = torch.zeros(B, N_full, N_local, D, dtype=a.dtype, device=a.device)
        a_contig = a.contiguous()
        a_recv = torch.empty_like(a)
        a0 = _log(P, "after alloc out + buffers", a0)

        for src in range(dap_size):
            if src == dap_rank:
                a_chunk = a_contig
                dist.broadcast(a_chunk, src=src)
            else:
                dist.broadcast(a_recv, src=src)
                a_chunk = a_recv

            i_start = src * N_local
            i_end = min(i_start + N_local, N_full)
            i_span = i_end - i_start
            micro_i = _micro_chunk_size("i", i_span)
            for ii in range(i_start, i_end, micro_i):
                ii_e = min(ii + micro_i, i_end)
                r0 = ii - i_start
                r1 = ii_e - i_start
                a_sub = a_chunk[:, :, r0:r1, :]
                out[:, ii:ii_e, :, :] = torch.einsum(
                    "bkid,bkjd->bijd", a_sub, b
                )
            a0 = _log(P, f"einsum src={src} i=[{i_start}:{i_end}] micro={micro_i}", a0)

        del a, b, a_contig, a_recv
        a0 = _log(P, "after cleanup", a0)

        out = self.inner.p_out(self.inner.norm_out(out)) * g_out
        _log(P, "after output gating", a0)
        return out
