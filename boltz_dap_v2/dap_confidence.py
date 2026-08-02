"""
DAP-aware Confidence Module for Boltz 2.

Distributes ALL confidence computation across GPUs:
1. Scatter z early (before pre-PF ops)
2. Pre-pairformer ops computed per-chunk on each GPU
3. DAP pairformer (all GPUs)
4. Gather z → confidence heads (GPU 0)

Usage:
    Called from dap_trunk.py's dap_forward() instead of the original
    model.confidence_module() call.
"""

import torch
from torch import Tensor
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boltz_distributed.comm import scatter, gather
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_pairformer import DAPPairformerLayer

# Diagnostic: sequential inner-call index (set by outer loop, read after broadcast to debug shape corruption)
_DEBUG_CONF_CALL_IDX = [0]


def _local_cuda_device() -> torch.device:
    """Return the process-local CUDA device selected by DAP initialization."""
    return torch.device("cuda", torch.cuda.current_device())


def _process_ram_mb() -> tuple[int | None, int | None, int | None]:
    """Return process RSS plus cgroup memory usage/limit in MB when available."""
    rss_mb = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass

    cgroup_used_mb = None
    cgroup_limit_mb = None
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            cgroup_used_mb = int(f.read().strip()) // (1024 * 1024)
        with open("/sys/fs/cgroup/memory.max") as f:
            raw_limit = f.read().strip()
            if raw_limit != "max":
                cgroup_limit_mb = int(raw_limit) // (1024 * 1024)
    except OSError:
        pass

    return rss_mb, cgroup_used_mb, cgroup_limit_mb


def _dict_tensors_to_cpu(obj):
    """Recursively move tensors in nested dict to CPU (for pair_chains_iptm)."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.is_cuda else obj
    if isinstance(obj, dict):
        return {k: _dict_tensors_to_cpu(v) for k, v in obj.items()}
    return obj


def _stage_confidence_feats(feats: dict, conf, dap_rank: int) -> dict[str, Tensor]:
    """Keep only confidence inputs, with one CPU copy on global rank 0."""
    required = {
        "asym_id",
        "atom_pad_mask",
        "atom_to_token",
        "frames_idx",
        "mol_type",
        "token_pad_mask",
        "token_to_rep_atom",
    }
    if conf.add_z_input_to_z:
        required.update(
            {
                "contact_conditioning",
                "contact_threshold",
                "entity_id",
                "residue_index",
                "sym_id",
                "token_bonds",
                "token_index",
            }
        )
        if conf.bond_type_feature:
            required.add("type_bonds")
        if (
            hasattr(conf, "rel_pos")
            and hasattr(conf.rel_pos, "cyclic_pos_enc")
            and conf.rel_pos.cyclic_pos_enc
        ):
            required.add("cyclic_period")

    feats_cpu = {}
    for key, value in list(feats.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if dap_rank == 0 and key in required:
            cpu_value = value.cpu() if value.is_cuda else value
            feats[key] = cpu_value
            feats_cpu[key] = cpu_value
        else:
            del feats[key]
    return feats_cpu


def _stream_contact_probability(
    pred_distogram_logits: Tensor,
    device: torch.device,
    row_chunk: int,
) -> Tensor:
    """Compute contact probability without materializing full logits on GPU."""
    batch_size, num_tokens, num_tokens_2, num_bins = pred_distogram_logits.shape
    if num_tokens != num_tokens_2 or num_bins != 64:
        raise ValueError(
            "Expected distogram logits with shape [B, N, N, 64], got "
            f"{tuple(pred_distogram_logits.shape)}"
        )
    row_chunk = max(1, row_chunk)
    prob_contact = torch.empty(
        batch_size,
        num_tokens,
        num_tokens,
        dtype=pred_distogram_logits.dtype,
        device=device,
    )
    contacts = torch.zeros(
        (1, 1, 1, num_bins),
        dtype=pred_distogram_logits.dtype,
        device=device,
    )
    contacts[:, :, :, :20] = 1.0
    for row_start in range(0, num_tokens, row_chunk):
        row_end = min(row_start + row_chunk, num_tokens)
        logits_rows = pred_distogram_logits[:, row_start:row_end].to(
            device=device, non_blocking=False
        )
        probability_rows = torch.nn.functional.softmax(logits_rows, dim=-1)
        prob_contact[:, row_start:row_end].copy_(
            (probability_rows * contacts).sum(-1)
        )
        del logits_rows, probability_rows
    return prob_contact


def inject_dap_into_confidence(confidence_module):
    """Replace confidence module's pairformer layers with DAP wrappers.

    confidence_module.pairformer_stack is a PairformerModule
    with .layers = ModuleList of PairformerLayer.
    """
    pf = confidence_module.pairformer_stack
    if hasattr(pf, '_orig_mod'):
        pf = pf._orig_mod

    num_layers = len(pf.layers)
    for i in range(num_layers):
        pf.layers[i] = DAPPairformerLayer(pf.layers[i])

    dap_rank = get_dap_rank()
    if dap_rank == 0:
        print(f"  ✓ Wrapped {num_layers} confidence pairformer layers with DAP")

    return confidence_module


def load_confidence_pre_pf_weights(model, device):
    """Load confidence pre-PF sub-module weights onto a GPU.

    These are small (~10 MB total): LayerNorms, small Linears, Embeddings.
    Called for GPU 1+ so they can compute pre-PF ops on their z chunk.
    """
    conf = model.confidence_module

    # Move pre-PF modules to device
    conf.s_inputs_norm.to(device)
    if not conf.no_update_s:
        conf.s_norm.to(device)
    conf.z_norm.to(device)
    conf.s_to_z.to(device)
    conf.s_to_z_transpose.to(device)

    if conf.add_s_input_to_s:
        conf.s_input_to_s.to(device)

    if conf.add_s_to_z_prod:
        conf.s_to_z_prod_in1.to(device)
        conf.s_to_z_prod_in2.to(device)
        conf.s_to_z_prod_out.to(device)

    if conf.add_z_input_to_z:
        conf.rel_pos.to(device)
        conf.token_bonds.to(device)
        if conf.bond_type_feature:
            conf.token_bonds_type.to(device)
        conf.contact_conditioning.to(device)

    conf.dist_bin_pairwise_embed.to(device)
    # Move boundaries buffer
    conf.boundaries = conf.boundaries.to(device)

    # PAE head weights needed for distributed PAE computation (Phase 3a)
    heads = conf.confidence_heads
    if heads.use_separate_heads:
        heads.to_pae_intra_logits.to(device)
        heads.to_pae_inter_logits.to(device)
    else:
        heads.to_pae_logits.to(device)


def run_confidence_dap(
    model,
    s_inputs: Tensor,
    s: Tensor,
    z_holder,
    x_pred: Tensor,
    feats: dict,
    pred_distogram_logits: Tensor,
    multiplicity: int = 1,
    run_sequentially: bool = True,
    use_kernels: bool = False,
    write_full_pae: bool = True,
    write_full_pde: bool = True,
):
    """Run the confidence module with DAP on ALL operations.

    All GPUs: scatter z early, compute pre-PF ops per-chunk, run DAP PF.
    GPU 0: gather z, run confidence heads.

    Parameters match model.confidence_module.forward().
    """
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    conf = model.confidence_module
    local_device = _local_cuda_device()

    def _barrier() -> None:
        try:
            torch.distributed.barrier(device_ids=[dap_rank])
        except TypeError:
            torch.distributed.barrier()

    # Extract z from z_holder early (before multiplicity branch)
    # NOTE: on non-primary ranks, z may be None (only rank 0 holds full z).
    # The scatter phase inside each recursive call distributes z from rank 0.
    z = z_holder[0] if isinstance(z_holder, list) else z_holder

    # Handle sequential processing of multiple samples
    if run_sequentially and multiplicity > 1:
        # Only rank 0 has z; assert batch=1 only there
        if dap_rank == 0:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
        if dap_rank == 0:
            # Rank 0: avoid holding 25 full outputs in memory (OOM on hexamer).
            # Run first sample, pre-allocate merged buffers, then fill in remaining samples.
            _DEBUG_CONF_CALL_IDX[0] = 0
            x_pred_0 = x_pred[0:1]
            out_0 = run_confidence_dap(
                model, s_inputs, s, z, x_pred_0, feats, pred_distogram_logits,
                multiplicity=1, run_sequentially=False, use_kernels=use_kernels,
                write_full_pae=write_full_pae, write_full_pde=write_full_pde,
            )
            # Keep merged on CPU so GPU only holds one confidence run at a time (avoids OOM on hexamer).
            merged = {}
            pair_chains_list = []
            for key in out_0:
                val0 = out_0[key]
                if val0 is None:
                    merged[key] = None
                elif key == "pair_chains_iptm":
                    pair_chains_list.append(_dict_tensors_to_cpu(val0))
                elif isinstance(val0, torch.Tensor):
                    # val0 has leading batch dim (1, ...); merged stacks over multiplicity without that dim.
                    merged[key] = torch.empty(
                        (multiplicity,) + val0.shape[1:],
                        dtype=val0.dtype,
                        device="cpu",
                    )
                    merged[key][0].copy_(val0[0].cpu())
                else:
                    merged[key] = [val0]
            del out_0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for sample_idx in range(1, multiplicity):
                _DEBUG_CONF_CALL_IDX[0] = sample_idx
                x_pred_i = x_pred[sample_idx : sample_idx + 1]
                out_i = run_confidence_dap(
                    model, s_inputs, s, z, x_pred_i, feats, pred_distogram_logits,
                    multiplicity=1, run_sequentially=False, use_kernels=use_kernels,
                    write_full_pae=write_full_pae, write_full_pde=write_full_pde,
                )
                for key in out_i:
                    vali = out_i[key]
                    if vali is None:
                        pass
                    elif key == "pair_chains_iptm":
                        pair_chains_list.append(_dict_tensors_to_cpu(vali))
                    elif isinstance(vali, torch.Tensor):
                        merged[key][sample_idx].copy_(vali[0].cpu())
                del out_i
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # Build final out_dict: merged tensors as-is; pair_chains_iptm from list
            out_dict = {}
            for key in merged:
                if merged[key] is None:
                    out_dict[key] = None
                elif isinstance(merged[key], torch.Tensor):
                    out_dict[key] = merged[key]
                else:
                    out_dict[key] = merged[key][0]  # fallback single value
            if pair_chains_list:
                pair_chains_iptm = {}
                for chain_idx1 in pair_chains_list[0]:
                    chains_iptm = {}
                    for chain_idx2 in pair_chains_list[0][chain_idx1]:
                        chains_iptm[chain_idx2] = torch.cat(
                            [pair_chains_list[i][chain_idx1][chain_idx2] for i in range(multiplicity)],
                            dim=0,
                        )
                    pair_chains_iptm[chain_idx1] = chains_iptm
                out_dict["pair_chains_iptm"] = pair_chains_iptm
            return out_dict
        else:
            # Non-primary ranks: run all calls (no output collection)
            for sample_idx in range(multiplicity):
                _DEBUG_CONF_CALL_IDX[0] = sample_idx
                x_pred_i = x_pred  # empty on non-zero ranks
                run_confidence_dap(
                    model, s_inputs, s, z, x_pred_i, feats, pred_distogram_logits,
                    multiplicity=1, run_sequentially=False, use_kernels=use_kernels,
                    write_full_pae=write_full_pae, write_full_pde=write_full_pde,
                )
            return {}

    # ── Memory logging helper ──────────────────────────────────────────
    import time as _time
    _conf_t0 = _time.time()
    _conf_diag = os.environ.get("BOLTZ_CONFIDENCE_DIAG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    def _cmem(label):
        if not _conf_diag:
            return
        torch.cuda.synchronize(local_device)
        alloc = torch.cuda.memory_allocated(local_device) // (1024 * 1024)
        resv = torch.cuda.memory_reserved(local_device) // (1024 * 1024)
        free_cuda, total_cuda = torch.cuda.mem_get_info(local_device)
        free_mb = free_cuda // (1024 * 1024)
        elapsed = _time.time() - _conf_t0
        rss_mb, cgroup_used_mb, cgroup_limit_mb = _process_ram_mb()
        ram = f" | rss={rss_mb:6d}MB" if rss_mb is not None else ""
        if cgroup_used_mb is not None:
            if cgroup_limit_mb is None:
                ram += f" | cgroup={cgroup_used_mb:6d}MB/max"
            else:
                ram += f" | cgroup={cgroup_used_mb:6d}/{cgroup_limit_mb}MB"
        print(f"    [CONF R{dap_rank}]  {elapsed:6.1f}s | alloc={alloc:6d}MB | resv={resv:6d}MB | free={free_mb:6d}MB{ram} | {label}", flush=True)

    _cmem("conf entry")

    feats_cpu = _stage_confidence_feats(feats, conf, dap_rank)
    torch.cuda.empty_cache()
    _cmem(
        f"confidence feats staged (rank0_cpu_keys={len(feats_cpu)}, "
        f"remaining_keys={len(feats)})"
    )

    def _rank0_feat(name: str, dtype: Optional[torch.dtype] = None) -> Tensor:
        tensor = feats[name]
        if dtype is None:
            return tensor.to(device=local_device, non_blocking=True)
        return tensor.to(device=local_device, dtype=dtype, non_blocking=True)

    # ══════════════════════════════════════════════════════════════════
    # Phase 0: Scatter z + broadcast small data to all GPUs
    # ══════════════════════════════════════════════════════════════════

    # z was already extracted from z_holder above (before multiplicity branch)
    z_on_cpu = getattr(z, "device", None) and str(z.device).startswith("cpu")

    if dap_rank == 0:
        N = z.shape[1]
        B = z.shape[0]
        D_z = z.shape[3]
        D_s = s.shape[2]
        shape_tensor = torch.tensor(
            [B, N, D_z, D_s], dtype=torch.long, device=local_device
        )
    else:
        shape_tensor = torch.zeros(4, dtype=torch.long, device=local_device)

    torch.distributed.broadcast(shape_tensor, src=0)
    # On rank 1: compare GPU .tolist() vs .cpu().tolist() before using (to pinpoint .tolist() vs broadcast/sync cause)
    if _conf_diag and dap_rank == 1:
        _gpu_list = shape_tensor.tolist()
    raw_list = shape_tensor.cpu().tolist()
    if _conf_diag and dap_rank == 1:
        if _gpu_list != raw_list:
            print(
                f"    [CONF SHAPE diag] rank=1 call_idx={_DEBUG_CONF_CALL_IDX[0]} GPU_tolist={_gpu_list} CPU_tolist={raw_list} MISMATCH",
                flush=True,
            )
    # Use raw_list (from .cpu().tolist()) for reliable Python ints
    B, N, D_z, D_s = int(raw_list[0]), int(raw_list[1]), int(raw_list[2]), int(raw_list[3])
    # Guard against corrupted/overflowed dimensions (e.g. from bad broadcast on non-zero ranks)
    if not (0 < B <= 1000 and 0 < N <= 100000 and 0 < D_z <= 2048 and 0 < D_s <= 2048):
        raise ValueError(
            f"Invalid shape after broadcast: B={B} N={N} D_z={D_z} D_s={D_s} (dap_rank={dap_rank}). "
            "Check that z_holder is correct on rank 0 and that sequential calls pass z per sample."
        )
    # Diagnostic: compare Rank 0 vs Rank 1 shape after broadcast (to pinpoint .tolist() vs buffer/sync cause)
    if _conf_diag and dap_rank in (0, 1):
        _diag_call = _DEBUG_CONF_CALL_IDX[0]
        print(
            f"    [CONF SHAPE diag] rank={dap_rank} call_idx={_diag_call} raw_list={raw_list} -> B={B} N={N} D_z={D_z} D_s={D_s}",
            flush=True,
        )

    # Pad N to be divisible by dap_size
    N_padded = ((N + dap_size - 1) // dap_size) * dap_size
    chunk_N = N_padded // dap_size
    row_start = dap_rank * chunk_N
    row_end = row_start + chunk_N

    # Scatter z: each GPU gets [B, chunk_N, N, D_z]
    # When z is on CPU (Rank 0), scatter chunk-by-chunk so full z is never on GPU (avoids OOM).
    if dap_rank == 0:
        if z_on_cpu:
            if N_padded != N:
                z = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, N_padded - N))
            for r in range(1, dap_size):
                start = r * chunk_N
                end = start + chunk_N
                chunk = z[:, start:end, :, :].contiguous()
                chunk_bf16 = chunk.bfloat16().cuda()
                torch.distributed.send(chunk_bf16, dst=r)
                del chunk_bf16, chunk
            chunk0 = z[:, :chunk_N, :, :].contiguous()
            z_chunk = chunk0.bfloat16().cuda().float()
            del chunk0
        else:
            if N_padded != N:
                z_padded = torch.nn.functional.pad(z, (0, 0, 0, 0, 0, N_padded - N))
            else:
                z_padded = z
            z_bf16 = z_padded.bfloat16()
            del z_padded
            for r in range(1, dap_size):
                start = r * chunk_N
                end = start + chunk_N
                chunk = z_bf16[:, start:end, :, :].contiguous()
                torch.distributed.send(chunk, dst=r)
            z_chunk = z_bf16[:, :chunk_N, :, :].contiguous().float()
            del z_bf16, z
        if isinstance(z_holder, list):
            z_holder[0] = None
        torch.cuda.empty_cache()
        _cmem("after z scatter (full z freed)")
    else:
        z_chunk = torch.empty(B, chunk_N, N, D_z, dtype=torch.bfloat16, device=local_device)
        torch.distributed.recv(z_chunk, src=0)
        z_chunk = z_chunk.float()
        torch.cuda.empty_cache()  # Release stale trunk-era reserved memory

    # Broadcast small 1D data: s, s_inputs, mask
    if dap_rank != 0:
        s = torch.empty(B, N, D_s, dtype=torch.float32, device=local_device)
        s_inputs = torch.empty(B, N, D_s, dtype=torch.float32, device=local_device)
        mask = torch.empty(B, N, dtype=torch.float32, device=local_device)
    else:
        mask = _rank0_feat("token_pad_mask", torch.float32)
    torch.distributed.broadcast(s, src=0)
    torch.distributed.broadcast(s_inputs, src=0)
    torch.distributed.broadcast(mask, src=0)

    # Scatter N² feats entries needed for pre-PF ops
    # Helper: scatter rows of a [B,N,N,...] tensor
    def _scatter_rows(full_tensor_or_none, name, dtype=torch.float32):
        """Scatter rows of an N² tensor, staging rank-0 chunks to GPU only as needed."""
        if dap_rank == 0:
            full = full_tensor_or_none
            # Broadcast ndim and last dim so other ranks can allocate
            info = torch.tensor([full.dim(), full.shape[-1] if full.dim() == 4 else 0],
                                device=local_device, dtype=torch.long)
        else:
            info = torch.zeros(2, dtype=torch.long, device=local_device)
        torch.distributed.broadcast(info, src=0)
        ndim, last_d = info.tolist()

        if dap_rank == 0:
            if N_padded != N:
                if ndim == 4:
                    full = torch.nn.functional.pad(full, (0, 0, 0, 0, 0, N_padded - N))
                else:
                    full = torch.nn.functional.pad(full, (0, 0, 0, N_padded - N))
            for r in range(1, dap_size):
                rs = r * chunk_N
                re = rs + chunk_N
                chunk = full[:, rs:re].contiguous()
                chunk_send = chunk.to(device=local_device, dtype=dtype, non_blocking=True)
                torch.distributed.send(chunk_send, dst=r)
                del chunk, chunk_send
            chunk = full[:, row_start:row_end].contiguous().to(
                device=local_device, dtype=dtype, non_blocking=True
            )
            del full
            torch.cuda.empty_cache()
            return chunk
        else:
            if ndim == 4:
                chunk = torch.empty(B, chunk_N, N, int(last_d), dtype=dtype, device=local_device)
            else:
                chunk = torch.empty(B, chunk_N, N, dtype=dtype, device=local_device)
            torch.distributed.recv(chunk, src=0)
            return chunk

    feats_chunk = {}

    if conf.add_z_input_to_z:
        # rel_pos needs 1D feats (full, for both row & col indexing)
        for key in ["asym_id", "residue_index", "entity_id", "sym_id", "token_index"]:
            if dap_rank == 0:
                t = _rank0_feat(key, torch.float32)
            else:
                t = torch.empty(B, N, dtype=torch.float32, device=local_device)
            torch.distributed.broadcast(t, src=0)
            feats_chunk[key] = t

        if hasattr(conf, 'rel_pos') and hasattr(conf.rel_pos, 'cyclic_pos_enc') and conf.rel_pos.cyclic_pos_enc:
            if dap_rank == 0:
                t = _rank0_feat("cyclic_period", torch.float32)
            else:
                t = torch.empty(B, N, dtype=torch.float32, device=local_device)
            torch.distributed.broadcast(t, src=0)
            feats_chunk["cyclic_period"] = t

        # token_bonds [B,N,N] or [B,N,N,1] → scatter rows
        feats_chunk["token_bonds"] = _scatter_rows(
            feats["token_bonds"] if dap_rank == 0 else None, "token_bonds")

        # type_bonds [B,N,N] → scatter rows (if needed)
        if conf.bond_type_feature:
            feats_chunk["type_bonds"] = _scatter_rows(
                feats["type_bonds"] if dap_rank == 0 else None, "type_bonds")

        # contact_conditioning [B,N,N,C] → scatter rows
        feats_chunk["contact_conditioning"] = _scatter_rows(
            feats["contact_conditioning"] if dap_rank == 0 else None, "contact_conditioning")

        # contact_threshold [B,N,N] → scatter rows
        feats_chunk["contact_threshold"] = _scatter_rows(
            feats["contact_threshold"] if dap_rank == 0 else None, "contact_threshold")

    # Broadcast x_pred_repr for distance bins (small: [B, N, 3])
    if dap_rank == 0:
        rep_atom_idx = feats["token_to_rep_atom"].argmax(dim=-1).to(
            device=local_device, dtype=torch.long, non_blocking=True
        )
        rep_atom_idx = rep_atom_idx.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            Bx, mult, N_atoms, _ = x_pred.shape
            x_pred = x_pred.reshape(Bx * mult, N_atoms, -1)
        x_pred = x_pred.to(device=local_device, non_blocking=True)
        x_pred_repr = torch.gather(
            x_pred,
            1,
            rep_atom_idx.unsqueeze(-1).expand(-1, -1, x_pred.shape[-1]),
        )
        del rep_atom_idx
        # x_pred_repr is [B, N, 3] — small
    else:
        x_pred_repr = torch.empty(B, N, 3, dtype=torch.float32, device=local_device)
    torch.distributed.broadcast(x_pred_repr, src=0)

    if dap_rank == 0:
        torch.cuda.empty_cache()

    _cmem("after scatter + broadcast")

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Distributed pre-PF ops (all GPUs, on z_chunk)
    # ══════════════════════════════════════════════════════════════════

    # Norms (per-element, works on chunk)
    s_inputs_n = conf.s_inputs_norm(s_inputs)
    if not conf.no_update_s:
        s = conf.s_norm(s)

    if conf.add_s_input_to_s:
        s = s + conf.s_input_to_s(s_inputs_n)

    z_chunk = conf.z_norm(z_chunk)

    # Relative position encoding (per-chunk rows)
    if conf.add_z_input_to_z:
        # Build chunked feats for rel_pos: it indexes [:, :, None] - [:, None, :]
        # We create a feats dict that makes rel_pos produce [B, chunk_N, N, D]
        rel_feats = {}
        for key in ["asym_id", "residue_index", "entity_id", "sym_id", "token_index"]:
            if key in feats_chunk:
                rel_feats[key] = feats_chunk[key]
        if "cyclic_period" in feats_chunk:
            rel_feats["cyclic_period"] = feats_chunk["cyclic_period"]

        # Manually compute rel_pos for chunk rows
        # rel_pos uses feats[key][:, :, None] - feats[key][:, None, :]
        # For chunk rows, we need feats[key][:, row_start:row_end, None] - feats[key][:, None, :]
        # We create a modified feats where the "row" dimension is the chunk
        chunk_rel_feats = {}
        for key in ["asym_id", "residue_index", "entity_id", "sym_id", "token_index"]:
            if key in rel_feats:
                # Pad row dimension if needed
                full_feat = rel_feats[key]  # [B, N]
                if N_padded != N:
                    full_feat = torch.nn.functional.pad(full_feat, (0, N_padded - N))
                chunk_feat_rows = full_feat[:, row_start:row_end]  # [B, chunk_N]
                # Create a "fake" full feats that when used as [:, :, None] gives chunk rows
                # We'll compute manually instead
                chunk_rel_feats[key] = (chunk_feat_rows, rel_feats[key])  # (rows, cols)
        if "cyclic_period" in rel_feats:
            chunk_rel_feats["cyclic_period"] = rel_feats["cyclic_period"]

        # Compute rel_pos per-chunk manually (mirrors RelativePositionEncoder.forward)
        rp = conf.rel_pos
        rows = {}
        cols = {}
        for key in ["asym_id", "residue_index", "entity_id", "sym_id", "token_index"]:
            if key in chunk_rel_feats:
                rows[key], cols[key] = chunk_rel_feats[key]

        b_same_chain = torch.eq(rows["asym_id"][:, :, None], cols["asym_id"][:, None, :])
        b_same_residue = torch.eq(rows["residue_index"][:, :, None], cols["residue_index"][:, None, :])
        b_same_entity = torch.eq(rows["entity_id"][:, :, None], cols["entity_id"][:, None, :])

        d_residue = rows["residue_index"][:, :, None] - cols["residue_index"][:, None, :]

        if hasattr(rp, 'cyclic_pos_enc') and rp.cyclic_pos_enc and "cyclic_period" in chunk_rel_feats:
            period_feat = chunk_rel_feats["cyclic_period"]
            period = torch.where(period_feat > 0, period_feat, torch.zeros_like(period_feat) + 10000)
            # period is [B, N], need to broadcast for chunk rows
            d_residue = (d_residue - period[:, None, :] * torch.round(d_residue / period[:, None, :])).long()

        d_residue = torch.clip(d_residue + rp.r_max, 0, 2 * rp.r_max)
        d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * rp.r_max + 1).long()
        from torch.nn.functional import one_hot
        a_rel_pos = one_hot(d_residue, 2 * rp.r_max + 2)

        d_token = torch.clip(
            rows["token_index"][:, :, None] - cols["token_index"][:, None, :] + rp.r_max,
            0, 2 * rp.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * rp.r_max + 1,
        ).long()
        a_rel_token = one_hot(d_token, 2 * rp.r_max + 2)

        d_chain = torch.clip(
            rows["sym_id"][:, :, None] - cols["sym_id"][:, None, :] + rp.s_max,
            0, 2 * rp.s_max,
        )
        fix_check = rp.fix_sym_check if hasattr(rp, 'fix_sym_check') else False
        d_chain = torch.where(
            (~b_same_entity) if fix_check else b_same_chain,
            torch.zeros_like(d_chain) + 2 * rp.s_max + 1,
            d_chain,
        ).long()
        a_rel_chain = one_hot(d_chain, 2 * rp.s_max + 2)

        rel_pos_chunk = rp.linear_layer(
            torch.cat([a_rel_pos.float(), a_rel_token.float(),
                       b_same_entity.unsqueeze(-1).float(), a_rel_chain.float()], dim=-1)
        )  # [B, chunk_N, N, D]
        z_chunk = z_chunk + rel_pos_chunk
        del rel_pos_chunk, a_rel_pos, a_rel_token, a_rel_chain, d_residue, d_token, d_chain
        del b_same_chain, b_same_residue, b_same_entity

        # token_bonds (per-chunk rows)
        z_chunk = z_chunk + conf.token_bonds(feats_chunk["token_bonds"].unsqueeze(-1) if feats_chunk["token_bonds"].dim() == 3 else feats_chunk["token_bonds"])
        if conf.bond_type_feature:
            z_chunk = z_chunk + conf.token_bonds_type(feats_chunk["type_bonds"].long())

        # contact_conditioning (per-chunk rows)
        if "contact_conditioning" in feats_chunk:
            cc_feats = {
                "contact_conditioning": feats_chunk["contact_conditioning"],
                "contact_threshold": feats_chunk["contact_threshold"],
            }
            z_chunk = z_chunk + conf.contact_conditioning(cc_feats)
            del cc_feats

    # Repeat-interleave for multiplicity (on s)
    s = s.repeat_interleave(multiplicity, 0)

    # Outer product: s_to_z(s_inputs)[:, rows, None, :] + s_to_z_transpose(s_inputs)[:, None, :, :]
    s_z = conf.s_to_z(s_inputs_n)  # [B, N, D]
    s_z_t = conf.s_to_z_transpose(s_inputs_n)  # [B, N, D]
    # For chunk rows, slice s_z to chunk
    if N_padded != N:
        s_z_padded = torch.nn.functional.pad(s_z, (0, 0, 0, N_padded - N))
    else:
        s_z_padded = s_z
    z_chunk = z_chunk + s_z_padded[:, row_start:row_end, None, :] + s_z_t[:, None, :, :]
    del s_z_padded

    if conf.add_s_to_z_prod:
        p1 = conf.s_to_z_prod_in1(s_inputs_n)  # [B, N, D]
        p2 = conf.s_to_z_prod_in2(s_inputs_n)  # [B, N, D]
        if N_padded != N:
            p1 = torch.nn.functional.pad(p1, (0, 0, 0, N_padded - N))
        z_chunk = z_chunk + conf.s_to_z_prod_out(
            p1[:, row_start:row_end, None, :] * p2[:, None, :, :]
        )
        del p1, p2

    del s_z, s_z_t

    # Repeat for multiplicity
    z_chunk = z_chunk.repeat_interleave(multiplicity, 0)
    s_inputs_n = s_inputs_n.repeat_interleave(multiplicity, 0)

    # Distance bins (per-chunk)
    # x_pred_repr is [B, N, 3] — compute cdist for chunk rows only
    if N_padded != N:
        x_repr_padded = torch.nn.functional.pad(x_pred_repr, (0, 0, 0, N_padded - N))
    else:
        x_repr_padded = x_pred_repr
    x_repr_padded = x_repr_padded.repeat_interleave(multiplicity, 0)
    x_pred_repr_full = x_pred_repr.repeat_interleave(multiplicity, 0)
    d_chunk = torch.cdist(
        x_repr_padded[:, row_start:row_end],  # [B, chunk_N, 3]
        x_pred_repr_full,  # [B, N, 3]
    )  # [B, chunk_N, N]
    distogram_chunk = (d_chunk.unsqueeze(-1) > conf.boundaries).sum(dim=-1).long()
    distogram_chunk = conf.dist_bin_pairwise_embed(distogram_chunk)
    z_chunk = z_chunk + distogram_chunk
    del distogram_chunk, x_repr_padded

    # Compute mask for chunk
    mask = mask.repeat_interleave(multiplicity, 0)
    if N_padded != N:
        mask_padded = torch.nn.functional.pad(mask, (0, N_padded - N))
    else:
        mask_padded = mask
    pair_mask_chunk = mask_padded[:, row_start:row_end].unsqueeze(-1) * mask.unsqueeze(1)
    del feats_chunk, mask_padded

    _cmem("pre-PF done, PF start")

    # Phase 1 broadcasts are done; rank 0 features stay on CPU until Phase 3 heads.
    # Release reserved-but-unused CUDA blocks on ALL ranks
    torch.cuda.empty_cache()
    _cmem("feats offloaded + cache cleared")

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: DAP Pairformer (all GPUs) — unchanged
    # ══════════════════════════════════════════════════════════════════

    pf = conf.pairformer_stack
    if hasattr(pf, '_orig_mod'):
        pf = pf._orig_mod

    from boltz.data import const
    if not pf.training:
        if N > 2000:
            chunk_size_tri_attn = 16  # 9MME: 128 → ~44GB transient, 16 → ~5.5GB
        else:
            chunk_size_tri_attn = 128
    else:
        chunk_size_tri_attn = None

    for li, layer in enumerate(pf.layers):
        s, z_chunk = layer(
            s, z_chunk, mask, pair_mask_chunk,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )
        _cmem(f"  conf PF layer[{li}]")

    _cmem("PF done")

    # Phase 3 only needs a small subset of features; keep dense setup tensors on CPU.
    if dap_rank == 0:
        torch.cuda.empty_cache()
        _cmem("post-PF cache cleared, feats remain CPU-backed")

    phase3_feats: dict[str, Tensor] = {}

    def _phase3_source(name: str):
        if name in feats_cpu:
            return feats_cpu[name]
        return feats[name]

    def _phase3_feat(name: str, dtype: Optional[torch.dtype] = None) -> Tensor:
        tensor = phase3_feats.get(name)
        if tensor is None:
            source = _phase3_source(name)
            tensor = source.to(device=local_device, non_blocking=True)
            phase3_feats[name] = tensor
        if dtype is not None and tensor.dtype != dtype:
            return tensor.to(dtype=dtype)
        return tensor

    def _phase3_feat_dict(names: list[str]) -> dict:
        staged = {}
        for name in names:
            if name not in feats_cpu and name not in feats:
                continue
            value = _phase3_source(name)
            staged[name] = _phase3_feat(name) if isinstance(value, torch.Tensor) else value
        return staged

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Distributed confidence heads — stream PAE, chunk PDE
    # ══════════════════════════════════════════════════════════════════

    heads = conf.confidence_heads
    B_conf = z_chunk.shape[0]
    row_valid = max(0, min(chunk_N, N - row_start))

    def _broadcast_token_feat(name: str, dtype: torch.dtype) -> Tensor:
        if dap_rank == 0:
            tensor = _phase3_feat(name, dtype=dtype)
        else:
            tensor = torch.empty(B, N, dtype=dtype, device=local_device)
        torch.distributed.broadcast(tensor, src=0)
        return tensor

    from boltz.data import const
    from boltz.model.layers.confidence_utils import (
        compute_collinear_mask,
        tm_function,
    )

    asym_id_token = _broadcast_token_feat("asym_id", torch.long)
    token_pad_mask_base = _broadcast_token_feat("token_pad_mask", torch.float32)
    token_type_base = _broadcast_token_feat("mol_type", torch.long)
    asym_id_rep = asym_id_token.repeat_interleave(multiplicity, 0)
    token_pad_mask_m = token_pad_mask_base.repeat_interleave(multiplicity, 0).float()
    token_type = token_type_base.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

    def _phase3_atom_token_index() -> Tensor:
        source = _phase3_source("atom_to_token")
        return source.argmax(dim=-1).to(device=local_device, dtype=torch.long, non_blocking=True)

    def _compute_frame_mask_sparse() -> Tensor:
        frames_idx_true = _phase3_feat("frames_idx").long()
        atom_pad_mask_base = _phase3_feat("atom_pad_mask", torch.float32)
        atom_token_idx = _phase3_atom_token_index().clamp_(0, N - 1)
        asym_id_atom = torch.gather(asym_id_token, 1, atom_token_idx)

        B_atom, _, _ = x_pred.shape
        pred_atom_coords = x_pred.reshape(B_atom // multiplicity, multiplicity, -1, 3)
        frames_idx_pred = (
            frames_idx_true.clone()
            .repeat_interleave(multiplicity, 0)
            .reshape(B_atom // multiplicity, multiplicity, -1, 3)
        )

        for batch_idx, pred_atom_coord in enumerate(pred_atom_coords):
            token_idx = 0
            atom_idx = 0
            for chain_id in torch.unique(asym_id_token[batch_idx]):
                mask_chain_token = (asym_id_token[batch_idx] == chain_id) * token_pad_mask_base[batch_idx]
                mask_chain_atom = (asym_id_atom[batch_idx] == chain_id) * atom_pad_mask_base[batch_idx]
                num_tokens = int(mask_chain_token.sum().item())
                num_atoms = int(mask_chain_atom.sum().item())
                if (
                    token_type_base[batch_idx, token_idx] != const.chain_type_ids["NONPOLYMER"]
                    or num_atoms < 3
                ):
                    token_idx += num_tokens
                    atom_idx += num_atoms
                    continue
                chain_atom_mask = mask_chain_atom.bool()
                chain_coords = pred_atom_coord[:, chain_atom_mask]
                dist_mat = ((chain_coords[:, None, :, :] - chain_coords[:, :, None, :]) ** 2).sum(-1) ** 0.5
                resolved_pair = 1 - (
                    atom_pad_mask_base[batch_idx][chain_atom_mask][None, :]
                    * atom_pad_mask_base[batch_idx][chain_atom_mask][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
                frames = torch.cat(
                    [indices[:, :, 1:2], indices[:, :, 0:1], indices[:, :, 2:3]], dim=2
                ) + atom_idx
                try:
                    frames_idx_pred[batch_idx, :, token_idx : token_idx + num_atoms, :] = frames
                except Exception as exc:
                    print(f"Failed to process {feats.get('pdb_id', '<unknown>')} due to {exc}")
                token_idx += num_tokens
                atom_idx += num_atoms

        device = frames_idx_pred.device
        frames_expanded = pred_atom_coords[
            torch.arange(0, B_atom // multiplicity, 1, device=device)[:, None, None, None],
            torch.arange(0, multiplicity, 1, device=device)[None, :, None, None],
            frames_idx_pred,
        ].reshape(-1, 3, 3)
        mask_collinear_pred = compute_collinear_mask(
            frames_expanded[:, 1] - frames_expanded[:, 0],
            frames_expanded[:, 1] - frames_expanded[:, 2],
        ).reshape(B_atom // multiplicity, multiplicity, -1)
        return (mask_collinear_pred * token_pad_mask_base[:, None, :]).reshape(-1, N).float()

    if dap_rank == 0:
        try:
            maski = _compute_frame_mask_sparse()
        except Exception as exc:
            print(f"Error in streamed PAE/PTM frame mask: {exc}")
            maski = torch.zeros(B_conf, N, dtype=torch.float32, device=local_device)
    else:
        maski = torch.empty(B_conf, N, dtype=torch.float32, device=local_device)
    torch.distributed.broadcast(maski, src=0)

    z_pae = z_chunk[:, :row_valid, :, :]
    if heads.use_separate_heads:
        row_asym = asym_id_rep[:, row_start : row_start + row_valid]
        same_chain_chunk = row_asym.unsqueeze(-1) == asym_id_rep.unsqueeze(1)
        pae_logits_chunk = heads.to_pae_intra_logits(z_pae)
        m_same = same_chain_chunk.to(dtype=pae_logits_chunk.dtype).unsqueeze(-1)
        pae_logits_chunk.mul_(m_same)
        pae_inter_chunk = heads.to_pae_inter_logits(z_pae)
        pae_logits_chunk.addcmul_(pae_inter_chunk, 1.0 - m_same)
        del pae_inter_chunk, m_same, same_chain_chunk, row_asym
    else:
        pae_logits_chunk = heads.to_pae_logits(z_pae)
    del z_pae

    num_pae_bins = pae_logits_chunk.shape[-1]
    pae_bin_width = 32.0 / num_pae_bins
    pae_bounds = torch.arange(
        start=0.5 * pae_bin_width,
        end=32.0,
        step=pae_bin_width,
        device=local_device,
        dtype=torch.float32,
    )
    pae_probs = torch.nn.functional.softmax(pae_logits_chunk, dim=-1)
    del pae_logits_chunk
    pae_chunk_value = torch.sum(
        pae_probs * pae_bounds.view(*((1,) * (pae_probs.ndim - 1)), num_pae_bins),
        dim=-1,
    ).float()

    pae_full_cpu: Optional[Tensor] = None
    if write_full_pae:
        _pae_stream_pin = bool(torch.cuda.is_available())
        if dap_rank == 0:
            pae_full_cpu = torch.empty((B_conf, N, N), dtype=torch.float32, device="cpu", pin_memory=_pae_stream_pin)
            if row_valid > 0:
                pae_full_cpu[:, row_start : row_start + row_valid].copy_(
                    pae_chunk_value.cpu(), non_blocking=_pae_stream_pin
                )
            for src_rank in range(1, dap_size):
                src_start = src_rank * chunk_N
                src_valid = max(0, min(chunk_N, N - src_start))
                if src_valid <= 0:
                    continue
                recv_buf = torch.empty(B_conf, src_valid, N, dtype=torch.float32, device=local_device)
                torch.distributed.recv(recv_buf, src=src_rank)
                pae_full_cpu[:, src_start : src_start + src_valid].copy_(
                    recv_buf.cpu(), non_blocking=_pae_stream_pin
                )
                del recv_buf
        elif row_valid > 0:
            torch.distributed.send(pae_chunk_value.contiguous(), dst=0)

    N_res = token_pad_mask_m.sum(dim=-1, keepdim=True)
    tm_values = tm_function(pae_bounds.unsqueeze(0), N_res).view(B_conf, 1, 1, num_pae_bins)
    tm_expected_chunk = torch.sum(pae_probs.float() * tm_values, dim=-1)
    del pae_probs, tm_values

    row_slice = slice(row_start, row_start + row_valid)
    maski_rows = maski[:, row_slice]
    token_pad_rows = token_pad_mask_m[:, row_slice]
    asym_rows = asym_id_rep[:, row_slice]
    ligand_rows = is_ligand_token[:, row_slice]
    protein_rows = is_protein_token[:, row_slice]

    def _chunk_max_score(pair_mask: Tensor) -> Tensor:
        numer = (tm_expected_chunk * pair_mask).sum(dim=-1)
        denom = pair_mask.sum(dim=-1)
        score = numer / (denom + 1e-5)
        local_max = score.max(dim=1).values if score.shape[1] > 0 else torch.zeros(B_conf, device=local_device)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
        return local_max

    pair_mask_ptm = maski_rows[:, :, None] * token_pad_mask_m[:, None, :] * token_pad_rows[:, :, None]
    ptm = _chunk_max_score(pair_mask_ptm)
    pair_mask_iptm = pair_mask_ptm * (asym_id_rep[:, None, :] != asym_rows[:, :, None])
    iptm = _chunk_max_score(pair_mask_iptm)
    ligand_iptm_mask = pair_mask_iptm * (
        (ligand_rows[:, :, None] * is_protein_token[:, None, :])
        + (protein_rows[:, :, None] * is_ligand_token[:, None, :])
    )
    ligand_iptm = _chunk_max_score(ligand_iptm_mask)
    protein_iptm_mask = pair_mask_iptm * (protein_rows[:, :, None] * is_protein_token[:, None, :])
    protein_iptm = _chunk_max_score(protein_iptm_mask)

    pair_chains_iptm = {}
    asym_ids_list = [int(idx) for idx in torch.unique(asym_id_rep).tolist()]
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = pair_mask_ptm * (
                (asym_id_rep[:, None, :] == idx1) * (asym_rows[:, :, None] == idx2)
            )
            chain_iptm[idx2] = _chunk_max_score(mask_pair_chain)
            del mask_pair_chain
        pair_chains_iptm[idx1] = chain_iptm
    del pair_mask_ptm, pair_mask_iptm, ligand_iptm_mask, protein_iptm_mask
    del tm_expected_chunk, pae_chunk_value, maski, maski_rows, token_pad_rows
    del ligand_rows, protein_rows
    torch.cuda.empty_cache()
    _cmem("streamed PAE summaries done")

    # Gather d_chunk → full d (collective, small)
    d_full = gather(d_chunk.contiguous(), dim=1, original_size=N)
    del d_chunk

    # 3b. Chunked PDE on Rank 0 (never gather full z — avoids OOM on hexamer)
    # For each column block r_start:r_end_col: R0 gathers rows + z_sym, runs PDE heads.
    # Optional: BOLTZ_DAP_KEEP_PDE_LOGITS=1 (--keep_pde_logits) keeps full logits on CPU; default off (saves RAM).
    keep_pde_logits = os.environ.get("BOLTZ_DAP_KEEP_PDE_LOGITS", "0") == "1"
    D_z_chunk = z_chunk.shape[3]
    if dap_rank == 0 and heads.use_separate_heads:
        is_same_chain = (asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)).float()
        is_different_chain = 1.0 - is_same_chain

    token_pair_mask: Optional[Tensor] = None
    token_interface_pair_mask: Optional[Tensor] = None
    numer_pde = denom_pde = numer_ipde = denom_ipde = None
    pde_full_cpu: Optional[Tensor] = None
    pde_logits_cpu_chunks: Optional[list] = None
    _pde_stream_pin = False
    if dap_rank == 0:
        from boltz.model.layers.confidence_utils import compute_aggregated_metric as _pde_expectation

        prob_contact = _stream_contact_probability(
            pred_distogram_logits,
            local_device,
            int(os.environ.get("BOLTZ_CONF_DISTOGRAM_ROW_CHUNK", "16")),
        ).repeat_interleave(multiplicity, 0)
        token_pad_mask_m = token_pad_mask_base.repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask_m.unsqueeze(-1)
            * token_pad_mask_m.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask_m.shape[1], device=token_pad_mask_m.device
                ).unsqueeze(0)
            )
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        asym_id_rep = asym_id_token.repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id_rep.unsqueeze(-1) != asym_id_rep.unsqueeze(-2)
        )
        numer_pde = torch.zeros(B, device=local_device, dtype=torch.float32)
        denom_pde = torch.zeros(B, device=local_device, dtype=torch.float32)
        numer_ipde = torch.zeros(B, device=local_device, dtype=torch.float32)
        denom_ipde = torch.zeros(B, device=local_device, dtype=torch.float32)
        _pde_stream_pin = bool(torch.cuda.is_available())
        if write_full_pde:
            pde_full_cpu = torch.empty((B_conf, N, N), dtype=torch.float32, device="cpu", pin_memory=_pde_stream_pin)
        pde_logits_cpu_chunks = [] if keep_pde_logits else None

    for r_idx in range(dap_size):
        _cmem(f"chunked PDE r_idx={r_idx}/{dap_size} start")
        r_start = r_idx * chunk_N
        r_end = r_start + chunk_N
        # Last column chunk may be shorter when N is not divisible (e.g. N=1557 → 1170:1557 has 387 cols)
        r_end_col = min(r_end, N)
        col_chunk_size = r_end_col - r_start

        # Row chunk size for this r_idx: last rank can have fewer rows when N_padded != N
        row_chunk_r = (N - (dap_size - 1) * chunk_N) if (r_idx == dap_size - 1 and N_padded != N) else chunk_N

        if dap_rank == 0:
            z_row = torch.empty(B, row_chunk_r, N, D_z_chunk, dtype=z_chunk.dtype, device=z_chunk.device)
        _barrier()
        # Send only row_chunk_r rows: last rank has z_chunk [B, 390, N, D] but only 387 are valid (row_chunk_r)
        if dap_rank == r_idx and r_idx != 0:
            torch.distributed.send(z_chunk[:, :row_chunk_r, :, :].contiguous(), dst=0)
        if dap_rank == 0:
            if r_idx == 0:
                z_row.copy_(z_chunk)
            else:
                torch.distributed.recv(z_row, src=r_idx)
        _barrier()
        # Gather columns [r_start:r_end_col] from all ranks → z_col [B, N, col_chunk_size, D]
        if dap_rank == 0:
            z_col_parts = []
            for k in range(dap_size):
                row_k = (N - (dap_size - 1) * chunk_N) if (k == dap_size - 1 and N_padded != N) else chunk_N
                if k == 0:
                    z_col_parts.append(z_chunk[:, :, r_start:r_end_col, :].clone())
                else:
                    buf = torch.empty(B, row_k, col_chunk_size, D_z_chunk, dtype=z_chunk.dtype, device=local_device)
                    torch.distributed.recv(buf, src=k)
                    z_col_parts.append(buf)
            z_col = torch.cat(z_col_parts, dim=1)
            del z_col_parts
            if N_padded != N:
                z_col = z_col[:, :N, :, :]
        else:
            # Send only row_self rows so recv buffer (B, row_k, col_chunk_size, D) on R0 matches
            row_self = (N - (dap_size - 1) * chunk_N) if (dap_rank == dap_size - 1 and N_padded != N) else chunk_N
            torch.distributed.send(
                z_chunk[:, :row_self, r_start:r_end_col, :].contiguous(), dst=0
            )
        if dap_rank == 0:
            z_sym_chunk = z_row + z_col.permute(0, 2, 1, 3)
            del z_row, z_col
            if heads.use_separate_heads:
                pde_intra_c = heads.to_pde_intra_logits(z_sym_chunk)
                pde_inter_c = heads.to_pde_inter_logits(z_sym_chunk)
                m_same = is_same_chain[:, r_start:r_end_col, :].unsqueeze(-1)
                m_diff = is_different_chain[:, r_start:r_end_col, :].unsqueeze(-1)
                pde_c = pde_intra_c * m_same + pde_inter_c * m_diff
                del pde_intra_c, pde_inter_c, m_same, m_diff
            else:
                pde_c = heads.to_pde_logits(z_sym_chunk)
            del z_sym_chunk
            # On-the-fly: expected PDE scalar field + gPDE/giPDE (same formulas as confidencev2) without
            # full [B,N,N,bins] pde_logits on GPU; optional CPU chunks for logits if KEEP flag set.
            # Layout: pde_c is [B, j_block, i, bins]; torch.cat(..., dim=1) stacks j → equivalent to
            # pde_logits[b, j, i, :]. token_pair_mask is [b, i, j] so use pde_ij = pde_cm.permute(0,2,1).
            pde_cm = _pde_expectation(pde_c, end=32)
            pde_ij = pde_cm.float().permute(0, 2, 1)
            m_blk = token_pair_mask[:, :, r_start:r_end_col].float()
            iface_blk = token_interface_pair_mask[:, :, r_start:r_end_col].float()
            numer_pde += (pde_ij * m_blk).sum(dim=(1, 2))
            denom_pde += m_blk.sum(dim=(1, 2))
            numer_ipde += (pde_ij * iface_blk).sum(dim=(1, 2))
            denom_ipde += iface_blk.sum(dim=(1, 2))
            if pde_full_cpu is not None:
                pde_full_cpu[:, :, r_start:r_end_col].copy_(pde_ij.cpu(), non_blocking=_pde_stream_pin)
            if pde_logits_cpu_chunks is not None:
                pde_logits_cpu_chunks.append(pde_c.detach().cpu())
            del pde_c, pde_cm, pde_ij, m_blk, iface_blk
        _cmem(f"chunked PDE r_idx={r_idx}/{dap_size} done")
    del z_chunk
    pde_logits: Optional[Tensor] = None
    complex_pde: Optional[Tensor] = None
    complex_ipde: Optional[Tensor] = None
    pde: Optional[Tensor] = None
    if dap_rank == 0:
        complex_pde = numer_pde / denom_pde
        complex_ipde = numer_ipde / (denom_ipde + 1e-5)
        pde = pde_full_cpu
        if keep_pde_logits and pde_logits_cpu_chunks:
            pde_logits = torch.cat(pde_logits_cpu_chunks, dim=1)
            del pde_logits_cpu_chunks
            if N_padded != N:
                pde_logits = pde_logits[:, :N, :N, :].contiguous()
        # Masks / contact probabilities are only used inside chunked PDE.
        del prob_contact, token_pad_pair_mask
        del token_pair_mask, token_interface_pair_mask, asym_id_rep
    _barrier()
    _cmem("chunked PDE done")

    # 3c. GPU 0: run metrics (no full z)
    if dap_rank == 0:
        torch.cuda.empty_cache()
        out_dict = {}

        if conf.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = None  # not kept to save memory

        _cmem("PDE done")

        # s-only heads
        resolved_logits = heads.to_resolved_logits(s)
        plddt_logits = heads.to_plddt_logits(s)

        # ── Metric aggregation (from original ConfidenceHeads.forward) ──
        from boltz.data import const
        from boltz.model.layers.confidence_utils import (
            compute_aggregated_metric,
        )

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        if heads.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)
            token_pad_mask = token_pad_mask_base.repeat_interleave(multiplicity, 0)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)

            is_contact = (d_full < 8).float()
            is_different_chain_metric = (
                asym_id_token.unsqueeze(-1) != asym_id_token.unsqueeze(-2)
            ).float()
            is_different_chain_metric = is_different_chain_metric.repeat_interleave(multiplicity, 0)
            token_interface_mask = torch.max(
                is_contact * is_different_chain_metric * (1 - is_ligand_token).unsqueeze(-1),
                dim=-1,
            ).values
            token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
            iplddt_weight = (
                is_ligand_token * ligand_weight
                + token_interface_mask * interface_weight
                + token_non_interface_mask * non_interface_weight
            )
            complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / torch.sum(
                token_pad_mask * iplddt_weight, dim=-1
            )
        else:
            from torch.nn.functional import pad as nn_pad
            B_h, N_h, _ = resolved_logits.shape
            resolved_logits = resolved_logits.reshape(B_h, N_h, heads.max_num_atoms_per_token, 2)
            arange_max = torch.arange(heads.max_num_atoms_per_token).reshape(1, 1, -1).to(resolved_logits.device)
            atom_to_token_base = _phase3_feat("atom_to_token")
            atom_pad_mask_base = _phase3_feat("atom_pad_mask")
            max_atoms_mask = atom_to_token_base.sum(1).unsqueeze(-1) > arange_max
            resolved_logits = resolved_logits[:, max_atoms_mask.squeeze(0)]
            resolved_logits = nn_pad(resolved_logits, (0, 0, 0, int(atom_pad_mask_base.shape[1] - atom_pad_mask_base.sum().item())), value=0)
            plddt_logits = plddt_logits.reshape(B_h, N_h, heads.max_num_atoms_per_token, -1)
            plddt_logits = plddt_logits[:, max_atoms_mask.squeeze(0)]
            plddt_logits = nn_pad(plddt_logits, (0, 0, 0, int(atom_pad_mask_base.shape[1] - atom_pad_mask_base.sum().item())), value=0)
            atom_pad_mask = atom_pad_mask_base.repeat_interleave(multiplicity, 0)
            plddt = compute_aggregated_metric(plddt_logits)
            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(dim=-1)
            token_type_f = token_type_base.float()
            atom_to_token = atom_to_token_base.float()
            chain_id_token = asym_id_token.float()
            atom_type = torch.bmm(atom_to_token, token_type_f.unsqueeze(-1)).squeeze(-1)
            is_ligand_atom = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
            d_atom = torch.cdist(x_pred, x_pred)
            is_contact = (d_atom < 8).float()
            chain_id_atom = torch.bmm(atom_to_token, chain_id_token.unsqueeze(-1)).squeeze(-1)
            is_different_chain_metric = (chain_id_atom.unsqueeze(-1) != chain_id_atom.unsqueeze(-2)).float()
            atom_interface_mask = torch.max(
                is_contact * is_different_chain_metric * (1 - is_ligand_atom).unsqueeze(-1), dim=-1
            ).values
            atom_non_interface_mask = (1 - atom_interface_mask) * (1 - is_ligand_atom)
            iplddt_weight = (
                is_ligand_atom * ligand_weight
                + atom_interface_mask * interface_weight
                + atom_non_interface_mask * non_interface_weight
            )
            complex_iplddt = (plddt * atom_pad_mask_base * iplddt_weight).sum(dim=-1) / torch.sum(
                atom_pad_mask_base * iplddt_weight, dim=-1
            )

        # gPDE / giPDE / pde field: already computed during chunked PDE (same formulas as confidencev2).

        out_dict.update(dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
            ptm=ptm,
            iptm=iptm,
            ligand_iptm=ligand_iptm,
            protein_iptm=protein_iptm,
            pair_chains_iptm=pair_chains_iptm,
        ))
        if write_full_pae:
            out_dict["pae"] = pae_full_cpu

        return out_dict
    else:
        return {}
