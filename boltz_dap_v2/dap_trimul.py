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

DIAG = os.environ.get("BOLTZ_TRIMUL_DIAG", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _trimul_k_tile_size(n_contract: int) -> int:
    """Einsum contracts over the full N (k) axis; process k in tiles to avoid
    holding full a,b float tensors at once (major VRAM win for N≈8k+).

    BOLTZ_TRIMUL_K_TILE:
      unset / \"0\" / \"\" → auto (512 when n_contract > 2048, else full k)
      positive int → tile size (clamped to n_contract)
    """
    raw = os.environ.get("BOLTZ_TRIMUL_K_TILE", "").strip()
    if raw == "" or raw == "0":
        return min(n_contract, 512) if n_contract > 2048 else n_contract
    try:
        v = int(raw)
    except ValueError:
        return min(n_contract, 512) if n_contract > 2048 else n_contract
    if v <= 0:
        return n_contract
    return min(n_contract, v)


def _trimul_output_tile_size(n_output: int, env_name: str) -> int:
    """Tile the output axis so full-plane fp32 accumulators need not coexist."""
    raw = os.environ.get(env_name, os.environ.get("BOLTZ_TRIMUL_OUTPUT_TILE", "")).strip()
    if raw == "" or raw == "0":
        return n_output
    try:
        v = int(raw)
    except ValueError:
        return n_output
    if v <= 0:
        return n_output
    return min(n_output, v)


def _trimul_final_tile_size(n_output: int) -> int:
    """Tile final projection/gating over pair axes; LayerNorm is feature-local."""
    raw = os.environ.get("BOLTZ_TRIMUL_FINAL_TILE", "").strip()
    if raw == "" or raw == "0":
        return min(n_output, 1024) if n_output > 2048 else n_output
    try:
        v = int(raw)
    except ValueError:
        return min(n_output, 1024) if n_output > 2048 else n_output
    if v <= 0:
        return n_output
    return min(n_output, v)


def _trimul_result_stage(requested: str = "gpu") -> str:
    raw = requested.strip().lower()
    if raw in {"1", "true", "yes", "on", "cpu"} and not torch.is_grad_enabled():
        return "cpu"
    return "gpu"


def _trimul_source_diag_enabled() -> bool:
    return os.environ.get("BOLTZ_TRIMUL_SOURCE_DIAG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _allocate_result(shape, dtype, device, result_stage: str) -> Tensor:
    result_device = "cpu" if result_stage == "cpu" else device
    return torch.empty(*shape, dtype=dtype, device=result_device)


def _store_result_slice(result: Tensor, index, value: Tensor) -> None:
    if result.device.type == "cpu":
        result[index].copy_(value.detach().cpu())
    else:
        result[index] = value


def _scale_slice(scale, i0: int, i1: int, j0: int, j1: int):
    if not torch.is_tensor(scale):
        return scale
    i_sel = slice(None) if scale.shape[1] == 1 else slice(i0, i1)
    j_sel = slice(None) if scale.shape[2] == 1 else slice(j0, j1)
    return scale[:, i_sel, j_sel, :]


def _add_residual_from_result_(x: Tensor, result: Tensor, scale, prefix: str) -> Tensor:
    i_tile = _trimul_final_tile_size(result.shape[1])
    j_tile = _trimul_final_tile_size(result.shape[2])
    a0 = _log(prefix, f"residual add from {result.device.type} result")
    for i0 in range(0, result.shape[1], i_tile):
        i1 = min(i0 + i_tile, result.shape[1])
        for j0 in range(0, result.shape[2], j_tile):
            j1 = min(j0 + j_tile, result.shape[2])
            update = result[:, i0:i1, j0:j1, :]
            if update.device != x.device:
                update = update.to(device=x.device, non_blocking=False)
            scale_part = _scale_slice(scale, i0, i1, j0, j1)
            if torch.is_tensor(scale_part):
                update = update * scale_part
            elif scale_part != 1.0:
                update = update * scale_part
            x[:, i0:i1, j0:j1, :].add_(update)
            del update
    _log(prefix, "after residual add", a0)
    return x


def _log(prefix, tag, a0=None):
    if not DIAG:
        return _m()
    a = _m(); p = _p()
    delta = f"Δ{a - a0:+.0f}" if a0 is not None else ""
    print(f"          [{prefix}] {tag:40s} | alloc={a:8.0f}MB | peak={p:8.0f}MB | {delta}", flush=True)
    return a


class DAPTriMulOut(nn.Module):
    """Triangle Multiplication Outgoing with DAP (broadcast-chunked).

    Input:  z row-scattered [B, N/dap, N, D]
    Output: z row-scattered [B, N/dap, N, D]
    """

    def __init__(self, original_module):
        super().__init__()
        self.inner = original_module

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        use_kernels: bool = False,
        *,
        _result_stage: str = "gpu",
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, use_kernels=use_kernels)

        dap_rank = get_dap_rank()
        P = "TRI-MUL-OUT"
        # x: [B, N/dap, N, D] (row-scattered)
        B, N_local, N_full, D_z = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        k_tile = _trimul_k_tile_size(N_full)
        output_tile = _trimul_output_tile_size(N_local, "BOLTZ_TRIMUL_OUT_OUTPUT_TILE")
        final_tile = _trimul_final_tile_size(N_full)
        result_stage = _trimul_result_stage(_result_stage)

        if output_tile < N_local:
            result = None
            a0 = _log(P, f"stream output tile={output_tile}, k_tile={k_tile}, final_tile={final_tile}", a0)

            for i0 in range(0, N_local, output_tile):
                i1 = min(i0 + output_tile, N_local)
                x_normed_out = self.inner.norm_in(x[:, i0:i1, :, :])
                out_tile = None

                for k0 in range(0, N_full, k_tile):
                    k1 = min(k0 + k_tile, N_full)
                    x_normed_a = x_normed_out[:, :, k0:k1, :]
                    x_proj_a = self.inner.p_in(x_normed_a) * self.inner.g_in(x_normed_a).sigmoid()
                    x_proj_a = x_proj_a * mask[:, i0:i1, k0:k1].unsqueeze(-1)
                    a, _ = torch.chunk(x_proj_a.float(), 2, dim=-1)
                    del x_proj_a

                    x_normed_b = self.inner.norm_in(x[:, :, k0:k1, :])
                    x_proj_b = self.inner.p_in(x_normed_b) * self.inner.g_in(x_normed_b).sigmoid()
                    x_proj_b = x_proj_b * mask[:, :, k0:k1].unsqueeze(-1)
                    _, b = torch.chunk(x_proj_b.float(), 2, dim=-1)
                    del x_normed_b, x_proj_b

                    d_pair = a.shape[-1]
                    if out_tile is None:
                        out_tile = torch.zeros(
                            B, i1 - i0, N_full, d_pair, dtype=torch.float32, device=x.device
                        )
                        a0 = _log(P, f"alloc out tile i[{i0}:{i1}]", a0)

                    b_contig = b.contiguous()
                    b_recv = torch.empty_like(b_contig)

                    for src in range(dap_size):
                        if src == dap_rank:
                            b_chunk = b_contig
                            dist.broadcast(b_chunk, src=src)
                        else:
                            dist.broadcast(b_recv, src=src)
                            b_chunk = b_recv

                        j_start = src * N_local
                        j_end = min(j_start + N_local, N_full)
                        jl = j_end - j_start
                        out_tile[:, :, j_start:j_end, :] += torch.einsum(
                            "bikd,bjkd->bijd",
                            a,
                            b_chunk[:, :jl, :, :],
                        )
                        if _trimul_source_diag_enabled():
                            a0 = _log(P, f"output i[{i0}:{i1}] k[{k0}:{k1}] src={src}", a0)

                    del a, b, b_contig, b_recv
                for j0 in range(0, N_full, final_tile):
                    j1 = min(j0 + final_tile, N_full)
                    g_out_part = self.inner.g_out(x_normed_out[:, :, j0:j1, :]).sigmoid()
                    final_part = self.inner.p_out(
                        self.inner.norm_out(out_tile[:, :, j0:j1, :])
                    ) * g_out_part
                    if result is None:
                        result = _allocate_result(
                            (B, N_local, N_full, final_part.shape[-1]),
                            final_part.dtype,
                            x.device,
                            result_stage,
                        )
                        a0 = _log(P, f"alloc streamed result ({result_stage})", a0)
                    _store_result_slice(result, (slice(None), slice(i0, i1), slice(j0, j1), slice(None)), final_part)
                    del g_out_part, final_part
                del x_normed_out, out_tile

            _log(P, "after streamed output", a0)
            return result

        # ── Norm + output gating (needs full x_normed; same as stock TriMul) ──
        x_normed = self.inner.norm_in(x)
        g_out = self.inner.g_out(x_normed).sigmoid()
        a0 = _log(P, "after norm_in + g_out", a0)

        # After p_in: [..., 2*D_z]; chunk splits into two [..., D_z] halves (same as stock TriMul).
        # Accumulator must be D_z-wide — not D_z//2 (that was a sizing bug).
        out = None

        # ── Contract over k in tiles: never keep full a,b simultaneously ──
        for k0 in range(0, N_full, k_tile):
            k1 = min(k0 + k_tile, N_full)
            m_sl = mask[:, :, k0:k1]
            x_sl = x_normed[:, :, k0:k1, :]
            x_proj = self.inner.p_in(x_sl) * self.inner.g_in(x_sl).sigmoid()
            x_proj = x_proj * m_sl.unsqueeze(-1)
            a, b = torch.chunk(x_proj.float(), 2, dim=-1)
            del x_proj

            d_pair = a.shape[-1]
            if out is None:
                out = torch.zeros(
                    B, N_local, N_full, d_pair, dtype=torch.float32, device=x.device
                )
                a0 = _log(P, f"alloc out (k_tile={k_tile}, d_pair={d_pair})", a0)

            b_contig = b.contiguous()
            b_recv = torch.empty_like(b_contig)

            for src in range(dap_size):
                if src == dap_rank:
                    b_chunk = b_contig
                    dist.broadcast(b_chunk, src=src)
                else:
                    dist.broadcast(b_recv, src=src)
                    b_chunk = b_recv

                j_start = src * N_local
                j_end = min(j_start + N_local, N_full)
                jl = j_end - j_start
                # x_sl is already k-tiled; b_chunk's k axis is local 0..(k1-k0). Do not index with k0:k1.
                out[:, :, j_start:j_end, :] += torch.einsum(
                    "bikd,bjkd->bijd",
                    a,
                    b_chunk[:, :jl, :, :],
                )
                a0 = _log(P, f"k[{k0}:{k1}] src={src} j=[{j_start}:{j_end}]", a0)

            del a, b, b_contig, b_recv

        del x_normed
        a0 = _log(P, "after k-tile + cleanup", a0)

        out = self.inner.p_out(self.inner.norm_out(out)) * g_out
        _log(P, "after output gating", a0)
        return out

    def forward_with_residual(
        self,
        x: Tensor,
        mask: Tensor,
        residual_scale,
        use_kernels: bool = False,
    ) -> Tensor:
        if get_dap_size() <= 1:
            return x + residual_scale * self.inner(x, mask, use_kernels=use_kernels)
        result_stage = os.environ.get("BOLTZ_TRIMUL_RESULT_STAGE", "cpu")
        result = self.forward(x, mask, use_kernels=use_kernels, _result_stage=result_stage)
        try:
            return _add_residual_from_result_(x, result, residual_scale, "TRI-MUL-OUT")
        finally:
            del result


class DAPTriMulIn(nn.Module):
    """Triangle Multiplication Incoming with DAP (broadcast-chunked).

    Input:  z col-scattered [B, N, N/dap, D]
    Output: z col-scattered [B, N, N/dap, D]
    """

    def __init__(self, original_module):
        super().__init__()
        self.inner = original_module

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        use_kernels: bool = False,
        *,
        _result_stage: str = "gpu",
    ) -> Tensor:
        dap_size = get_dap_size()
        if dap_size <= 1:
            return self.inner(x, mask, use_kernels=use_kernels)

        dap_rank = get_dap_rank()
        P = "TRI-MUL-IN "
        # x: [B, N, N/dap, D] (col-scattered)
        B, N_full, N_local, D_z = x.shape

        torch.cuda.reset_peak_memory_stats()
        a0 = _log(P, "entry")

        k_tile = _trimul_k_tile_size(N_full)
        output_tile = _trimul_output_tile_size(N_full, "BOLTZ_TRIMUL_IN_OUTPUT_TILE")
        final_tile = _trimul_final_tile_size(N_local)
        result_stage = _trimul_result_stage(_result_stage)

        if output_tile < N_full:
            result = None
            a0 = _log(P, f"stream output tile={output_tile}, k_tile={k_tile}, final_tile={final_tile}", a0)

            for i0 in range(0, N_full, output_tile):
                i1 = min(i0 + output_tile, N_full)
                x_normed_out = self.inner.norm_in(x[:, i0:i1, :, :])
                out_tile = None

                for k0 in range(0, N_full, k_tile):
                    k1 = min(k0 + k_tile, N_full)
                    x_normed_k = self.inner.norm_in(x[:, k0:k1, :, :])
                    x_proj = self.inner.p_in(x_normed_k) * self.inner.g_in(x_normed_k).sigmoid()
                    x_proj = x_proj * mask[:, k0:k1, :].unsqueeze(-1)
                    a, b = torch.chunk(x_proj.float(), 2, dim=-1)
                    del x_normed_k, x_proj

                    d_pair = a.shape[-1]
                    if out_tile is None:
                        out_tile = torch.zeros(
                            B, i1 - i0, N_local, d_pair, dtype=torch.float32, device=x.device
                        )
                        a0 = _log(P, f"alloc out tile i[{i0}:{i1}]", a0)

                    a_contig = a.contiguous()
                    a_recv = torch.empty_like(a_contig)

                    for src in range(dap_size):
                        if src == dap_rank:
                            a_chunk = a_contig
                            dist.broadcast(a_chunk, src=src)
                        else:
                            dist.broadcast(a_recv, src=src)
                            a_chunk = a_recv

                        src_i_start = src * N_local
                        src_i_end = min(src_i_start + N_local, N_full)
                        overlap_start = max(i0, src_i_start)
                        overlap_end = min(i1, src_i_end)
                        if overlap_start >= overlap_end:
                            continue

                        local_start = overlap_start - src_i_start
                        local_end = overlap_end - src_i_start
                        tile_start = overlap_start - i0
                        tile_end = overlap_end - i0
                        out_tile[:, tile_start:tile_end, :, :] += torch.einsum(
                            "bkid,bkjd->bijd",
                            a_chunk[:, :, local_start:local_end, :],
                            b,
                        )

                    del a, b, a_contig, a_recv
                for j0 in range(0, N_local, final_tile):
                    j1 = min(j0 + final_tile, N_local)
                    g_out_part = self.inner.g_out(x_normed_out[:, :, j0:j1, :]).sigmoid()
                    final_part = self.inner.p_out(
                        self.inner.norm_out(out_tile[:, :, j0:j1, :])
                    ) * g_out_part
                    if result is None:
                        result = _allocate_result(
                            (B, N_full, N_local, final_part.shape[-1]),
                            final_part.dtype,
                            x.device,
                            result_stage,
                        )
                        a0 = _log(P, f"alloc streamed result ({result_stage})", a0)
                    _store_result_slice(result, (slice(None), slice(i0, i1), slice(j0, j1), slice(None)), final_part)
                    del g_out_part, final_part
                del x_normed_out, out_tile

            _log(P, "after streamed output", a0)
            return result

        # ── Norm + gating (full x_normed); same as stock incoming ──
        x_normed = self.inner.norm_in(x)
        g_out = self.inner.g_out(x_normed).sigmoid()
        a0 = _log(P, "after norm_in + g_out", a0)

        out = None

        # einsum "bkid,bkjd->bijd": contract over k (dim1 of x); tile k to limit a,b peak
        for k0 in range(0, N_full, k_tile):
            k1 = min(k0 + k_tile, N_full)
            m_sl = mask[:, k0:k1, :]
            x_sl = x_normed[:, k0:k1, :, :]
            x_proj = self.inner.p_in(x_sl) * self.inner.g_in(x_sl).sigmoid()
            x_proj = x_proj * m_sl.unsqueeze(-1)
            a, b = torch.chunk(x_proj.float(), 2, dim=-1)
            del x_proj

            d_pair = a.shape[-1]
            if out is None:
                out = torch.zeros(
                    B, N_full, N_local, d_pair, dtype=torch.float32, device=x.device
                )
                a0 = _log(P, f"alloc out (k_tile={k_tile}, d_pair={d_pair})", a0)

            a_contig = a.contiguous()
            a_recv = torch.empty_like(a_contig)

            for src in range(dap_size):
                if src == dap_rank:
                    a_chunk = a_contig
                    dist.broadcast(a_chunk, src=src)
                else:
                    dist.broadcast(a_recv, src=src)
                    a_chunk = a_recv

                i_start = src * N_local
                i_end = min(i_start + N_local, N_full)
                il = i_end - i_start
                # b is already the k-tile [B, k_tile_len, N_local, d]; do not slice dim1 with k0:k1.
                out[:, i_start:i_end, :, :] += torch.einsum(
                    "bkid,bkjd->bijd",
                    a_chunk[:, :, :il, :],
                    b,
                )
                a0 = _log(P, f"k[{k0}:{k1}] src={src} i=[{i_start}:{i_end}]", a0)

            del a, b, a_contig, a_recv
        del x_normed
        a0 = _log(P, "after k-tile + cleanup", a0)

        out = self.inner.p_out(self.inner.norm_out(out)) * g_out
        _log(P, "after output gating", a0)
        return out

    def forward_with_residual(
        self,
        x: Tensor,
        mask: Tensor,
        residual_scale,
        use_kernels: bool = False,
    ) -> Tensor:
        if get_dap_size() <= 1:
            return x + residual_scale * self.inner(x, mask, use_kernels=use_kernels)
        result_stage = os.environ.get("BOLTZ_TRIMUL_RESULT_STAGE", "cpu")
        result = self.forward(x, mask, use_kernels=use_kernels, _result_stage=result_stage)
        try:
            return _add_residual_from_result_(x, result, residual_scale, "TRI-MUL-IN ")
        finally:
            del result
