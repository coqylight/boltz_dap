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
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

DIAG = os.environ.get("BOLTZ_TRIMUL_DIAG", "0") == "1"


def _trimul_k_tile_size(n_contract: int) -> int:
    """Einsum contracts over the full N (k) axis; process k in tiles to avoid
    holding full a,b float tensors at once (major VRAM win for N≈8k+).

    BOLTZ_TRIMUL_K_TILE:
      unset / "0" / "" -> auto (512 when n_contract > 2048, else full k)
      positive int -> tile size (clamped to n_contract)
    """
    raw = os.environ.get("BOLTZ_TRIMUL_K_TILE", "").strip()
    if raw == "" or raw == "0":
        if n_contract >= 12000:
            return min(n_contract, 128)
        if n_contract >= 8000:
            return min(n_contract, 256)
        return min(n_contract, 512) if n_contract > 2048 else n_contract
    try:
        v = int(raw)
    except ValueError:
        if n_contract >= 12000:
            return min(n_contract, 128)
        if n_contract >= 8000:
            return min(n_contract, 256)
        return min(n_contract, 512) if n_contract > 2048 else n_contract
    if v <= 0:
        return n_contract
    return min(n_contract, v)


def _trimul_local_tile_size(n_contract: int, n_local: int) -> int:
    """Chunk the broadcasted local row/col shard for large-N cases."""
    raw = os.environ.get("BOLTZ_TRIMUL_LOCAL_TILE", "").strip()
    if raw == "" or raw == "0":
        if n_contract >= 12000:
            return min(n_local, 128)
        if n_contract >= 8000:
            return min(n_local, 256)
        return n_local
    try:
        v = int(raw)
    except ValueError:
        if n_contract >= 12000:
            return min(n_local, 128)
        if n_contract >= 8000:
            return min(n_local, 256)
        return n_local
    if v <= 0:
        return n_local
    return min(n_local, v)


def _trimul_output_tile_size(n_full: int) -> int:
    """Chunk the final norm_out -> p_out projection over sequence positions.

    This avoids materializing a second full output tensor during F.linear.
    """
    raw = os.environ.get("BOLTZ_TRIMUL_OUT_TILE", "").strip()
    if raw == "" or raw == "0":
        if n_full >= 12000:
            return 128
        if n_full >= 8000:
            return 256
        return n_full
    try:
        v = int(raw)
    except ValueError:
        if n_full >= 12000:
            return 128
        if n_full >= 8000:
            return 256
        return n_full
    if v <= 0:
        return n_full
    return min(n_full, v)


def _log(prefix, tag, a0=None):
    if not DIAG:
        return _m()
    a = _m(); p = _p()
    delta = f"Δ{a - a0:+.0f}" if a0 is not None else ""
    print(f"          [{prefix}] {tag:40s} | alloc={a:8.0f}MB | peak={p:8.0f}MB | {delta}", flush=True)
    return a


def _module_compute_dtype(module: nn.Module) -> torch.dtype:
    """Use the original layer's parameter dtype for pointwise math."""
    return module.norm_in.weight.dtype


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
        B, N_local, N_full, _ = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        storage_dtype = x.dtype
        compute_dtype = _module_compute_dtype(self.inner)
        if x.dtype != compute_dtype:
            a0 = _log(P, f"stream input casts {storage_dtype} -> {compute_dtype}", a0)

        k_tile = _trimul_k_tile_size(N_full)
        local_tile = _trimul_local_tile_size(N_full, N_local)
        # After p_in: [..., 2*D_z]; chunk splits into two [..., D_z] halves.
        # Accumulator must be D_z-wide — not D_z//2.
        out = None

        # Contract over k in tiles: never keep full normalized x, a, or b.
        for k0 in range(0, N_full, k_tile):
            k1 = min(k0 + k_tile, N_full)
            x_sl = x[:, :, k0:k1, :]
            if x_sl.dtype != compute_dtype:
                x_sl = x_sl.to(dtype=compute_dtype)
            x_sl = self.inner.norm_in(x_sl)
            m_sl = mask[:, :, k0:k1]
            if m_sl.dtype != x_sl.dtype:
                m_sl = m_sl.to(dtype=x_sl.dtype)
            x_proj = self.inner.p_in(x_sl) * self.inner.g_in(x_sl).sigmoid()
            x_proj = x_proj * m_sl.unsqueeze(-1)
            a, b = torch.chunk(x_proj.float(), 2, dim=-1)
            del x_sl, x_proj

            d_pair = a.shape[-1]
            if out is None:
                out = torch.zeros(
                    B, N_local, N_full, d_pair, dtype=torch.float32, device=x.device
                )
                a0 = _log(P, f"alloc out (k_tile={k_tile}, d_pair={d_pair})", a0)

            b_contig = b.contiguous()

            for src in range(dap_size):
                j_start = src * N_local
                j_end = min(j_start + N_local, N_full)
                jl = j_end - j_start
                for j0 in range(0, jl, local_tile):
                    j1 = min(j0 + local_tile, jl)
                    if src == dap_rank:
                        b_chunk = b_contig[:, j0:j1, :, :].contiguous()
                    else:
                        b_chunk = torch.empty(
                            B,
                            j1 - j0,
                            k1 - k0,
                            d_pair,
                            dtype=b_contig.dtype,
                            device=b_contig.device,
                        )
                    dist.broadcast(b_chunk, src=src)

                    out[:, :, j_start + j0:j_start + j1, :] += torch.einsum(
                        "bikd,bjkd->bijd",
                        a,
                        b_chunk,
                    )
                    a0 = _log(
                        P,
                        f"k[{k0}:{k1}] src={src} j=[{j_start + j0}:{j_start + j1}]",
                        a0,
                    )
                    del b_chunk

            del a, b, b_contig

        a0 = _log(P, "after k-tile + cleanup", a0)

        out_tile = _trimul_output_tile_size(N_full)
        for j0 in range(0, N_full, out_tile):
            j1 = min(j0 + out_tile, N_full)
            out_chunk = out[:, :, j0:j1, :]
            normed = F.layer_norm(
                out_chunk,
                self.inner.norm_out.normalized_shape,
                self.inner.norm_out.weight,
                self.inner.norm_out.bias,
                self.inner.norm_out.eps,
            )
            projected = F.linear(
                normed,
                self.inner.p_out.weight,
                self.inner.p_out.bias,
            )
            x_gate = x[:, :, j0:j1, :]
            if x_gate.dtype != compute_dtype:
                x_gate = x_gate.to(dtype=compute_dtype)
            x_gate = self.inner.norm_in(x_gate)
            g_out = self.inner.g_out(x_gate).sigmoid()
            projected = projected * g_out.to(projected.dtype)
            out[:, :, j0:j1, :] = projected
            del normed, projected, x_gate, g_out
        _log(P, f"after output gating (tile={out_tile})", a0)

        # Keep the persistent pair activations in the original storage dtype
        # activation. Otherwise row_to_col/col_to_row would need a second
        # full-size float32 buffer (~2.27 GiB for N≈8712, D=128).
        if out.dtype != storage_dtype:
            out = out.to(dtype=storage_dtype)
            _log(P, f"after cast to {storage_dtype}", a0)
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
        B, N_full, N_local, _ = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        storage_dtype = x.dtype
        compute_dtype = _module_compute_dtype(self.inner)
        if x.dtype != compute_dtype:
            a0 = _log(P, f"stream input casts {storage_dtype} -> {compute_dtype}", a0)

        k_tile = _trimul_k_tile_size(N_full)
        local_tile = _trimul_local_tile_size(N_full, N_local)
        out = None

        # einsum "bkid,bkjd->bijd": contract over k (dim1 of x)
        for k0 in range(0, N_full, k_tile):
            k1 = min(k0 + k_tile, N_full)
            x_sl = x[:, k0:k1, :, :]
            if x_sl.dtype != compute_dtype:
                x_sl = x_sl.to(dtype=compute_dtype)
            x_sl = self.inner.norm_in(x_sl)
            m_sl = mask[:, k0:k1, :]
            if m_sl.dtype != x_sl.dtype:
                m_sl = m_sl.to(dtype=x_sl.dtype)
            x_proj = self.inner.p_in(x_sl) * self.inner.g_in(x_sl).sigmoid()
            x_proj = x_proj * m_sl.unsqueeze(-1)
            a, b = torch.chunk(x_proj.float(), 2, dim=-1)
            del x_sl, x_proj

            d_pair = a.shape[-1]
            if out is None:
                out = torch.zeros(
                    B, N_full, N_local, d_pair, dtype=torch.float32, device=x.device
                )
                a0 = _log(P, f"alloc out (k_tile={k_tile}, d_pair={d_pair})", a0)

            a_contig = a.contiguous()

            for src in range(dap_size):
                i_start = src * N_local
                i_end = min(i_start + N_local, N_full)
                il = i_end - i_start
                for i0 in range(0, il, local_tile):
                    i1 = min(i0 + local_tile, il)
                    if src == dap_rank:
                        a_chunk = a_contig[:, :, i0:i1, :].contiguous()
                    else:
                        a_chunk = torch.empty(
                            B,
                            k1 - k0,
                            i1 - i0,
                            d_pair,
                            dtype=a_contig.dtype,
                            device=a_contig.device,
                        )
                    dist.broadcast(a_chunk, src=src)

                    out[:, i_start + i0:i_start + i1, :, :] += torch.einsum(
                        "bkid,bkjd->bijd",
                        a_chunk,
                        b,
                    )
                    a0 = _log(
                        P,
                        f"k[{k0}:{k1}] src={src} i=[{i_start + i0}:{i_start + i1}]",
                        a0,
                    )
                    del a_chunk

            del a, b, a_contig
        a0 = _log(P, "after k-tile + cleanup", a0)

        out_tile = _trimul_output_tile_size(N_full)
        for i0 in range(0, N_full, out_tile):
            i1 = min(i0 + out_tile, N_full)
            out_chunk = out[:, i0:i1, :, :]
            normed = F.layer_norm(
                out_chunk,
                self.inner.norm_out.normalized_shape,
                self.inner.norm_out.weight,
                self.inner.norm_out.bias,
                self.inner.norm_out.eps,
            )
            projected = F.linear(
                normed,
                self.inner.p_out.weight,
                self.inner.p_out.bias,
            )
            x_gate = x[:, i0:i1, :, :]
            if x_gate.dtype != compute_dtype:
                x_gate = x_gate.to(dtype=compute_dtype)
            x_gate = self.inner.norm_in(x_gate)
            g_out = self.inner.g_out(x_gate).sigmoid()
            projected = projected * g_out.to(projected.dtype)
            out[:, i0:i1, :, :] = projected
            del normed, projected, x_gate, g_out
        _log(P, f"after output gating (tile={out_tile})", a0)

        # Keep the persistent pair activations in the original storage dtype
        # activation. Otherwise row_to_col/col_to_row would need a second
        # full-size float32 buffer (~2.27 GiB for N≈8712, D=128).
        if out.dtype != storage_dtype:
            out = out.to(dtype=storage_dtype)
            _log(P, f"after cast to {storage_dtype}", a0)
        return out
