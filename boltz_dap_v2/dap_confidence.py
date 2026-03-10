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
):
    """Run the confidence module with DAP on ALL operations.

    All GPUs: scatter z early, compute pre-PF ops per-chunk, run DAP PF.
    GPU 0: gather z, run confidence heads.

    Parameters match model.confidence_module.forward().
    """
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    conf = model.confidence_module

    # Extract z from z_holder early (before multiplicity branch)
    # NOTE: on non-primary ranks, z may be None (only rank 0 holds full z).
    # The scatter phase inside each recursive call distributes z from rank 0.
    z = z_holder[0] if isinstance(z_holder, list) else z_holder

    # Handle sequential processing of multiple samples
    if run_sequentially and multiplicity > 1:
        # Only rank 0 has z; assert batch=1 only there
        if dap_rank == 0:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
        out_dicts = []
        for sample_idx in range(multiplicity):
            # Rank 0: slice x_pred for this sample; others: pass empty tensor
            if dap_rank == 0:
                x_pred_i = x_pred[sample_idx : sample_idx + 1]
            else:
                x_pred_i = x_pred  # empty tensor on non-primary ranks
            out_dicts.append(
                run_confidence_dap(
                    model,
                    s_inputs, s, z,  # pass z directly (not list) to avoid clearing
                    x_pred_i,
                    feats,
                    pred_distogram_logits,
                    multiplicity=1,
                    run_sequentially=False,
                    use_kernels=use_kernels,
                )
            )
        # Merge outputs (only rank 0 has meaningful outputs)
        if dap_rank == 0:
            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key]:
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1]:
                            chains_iptm[chain_idx2] = torch.cat(
                                [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                                dim=0,
                            )
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm
            return out_dict
        else:
            # Non-primary ranks return last result (not used by caller)
            return out_dicts[-1] if out_dicts else {}

    # ── Memory logging helper ──────────────────────────────────────────
    import time as _time
    _conf_t0 = _time.time()
    def _cmem(label):
        torch.cuda.synchronize()
        dev = dap_rank
        alloc = torch.cuda.memory_allocated(dev) // (1024 * 1024)
        resv = torch.cuda.memory_reserved(dev) // (1024 * 1024)
        free_cuda, total_cuda = torch.cuda.mem_get_info(dev)
        free_mb = free_cuda // (1024 * 1024)
        elapsed = _time.time() - _conf_t0
        print(f"    [CONF R{dap_rank}]  {elapsed:6.1f}s | alloc={alloc:6d}MB | resv={resv:6d}MB | free={free_mb:6d}MB | {label}", flush=True)

    _cmem("conf entry")

    # ══════════════════════════════════════════════════════════════════
    # Phase 0: Scatter z + broadcast small data to all GPUs
    # ══════════════════════════════════════════════════════════════════

    # z was already extracted from z_holder above (before multiplicity branch)

    if dap_rank == 0:
        N = z.shape[1]
        B = z.shape[0]
        D_z = z.shape[3]
        D_s = s.shape[2]
        shape_tensor = torch.tensor([B, N, D_z, D_s], device=z.device)
    else:
        shape_tensor = torch.zeros(4, dtype=torch.long, device=f'cuda:{dap_rank}')

    torch.distributed.broadcast(shape_tensor, src=0)
    B, N, D_z, D_s = shape_tensor.tolist()
    N, D_z, D_s = int(N), int(D_z), int(D_s)
    B = int(B)

    # Pad N to be divisible by dap_size
    N_padded = ((N + dap_size - 1) // dap_size) * dap_size
    chunk_N = N_padded // dap_size
    row_start = dap_rank * chunk_N
    row_end = row_start + chunk_N

    # Scatter z: each GPU gets [B, chunk_N, N, D_z]
    if dap_rank == 0:
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
        # Clear holder to break the caller's last reference to full z
        if isinstance(z_holder, list):
            z_holder[0] = None
        torch.cuda.empty_cache()
        _cmem("after z scatter (full z freed)")
    else:
        device = torch.device(f'cuda:{dap_rank}')
        z_chunk = torch.empty(B, chunk_N, N, D_z, dtype=torch.bfloat16, device=device)
        torch.distributed.recv(z_chunk, src=0)
        z_chunk = z_chunk.float()
        torch.cuda.empty_cache()  # Release stale trunk-era reserved memory

    # Broadcast small 1D data: s, s_inputs, mask
    if dap_rank != 0:
        device = torch.device(f'cuda:{dap_rank}')
        s = torch.empty(B, N, D_s, dtype=torch.float32, device=device)
        s_inputs = torch.empty(B, N, D_s, dtype=torch.float32, device=device)
        mask = torch.empty(B, N, dtype=torch.float32, device=device)
    else:
        mask = feats["token_pad_mask"].float()
    torch.distributed.broadcast(s, src=0)
    torch.distributed.broadcast(s_inputs, src=0)
    torch.distributed.broadcast(mask, src=0)

    # Scatter N² feats entries needed for pre-PF ops
    # Helper: scatter rows of a [B,N,N,...] tensor
    def _scatter_rows(full_tensor_or_none, name, dtype=torch.float32):
        """Scatter rows of an N² tensor: GPU 0 sends chunk rows to each GPU."""
        if dap_rank == 0:
            full = full_tensor_or_none
            # Broadcast ndim and last dim so other ranks can allocate
            info = torch.tensor([full.dim(), full.shape[-1] if full.dim() == 4 else 0],
                                device=full.device, dtype=torch.long)
        else:
            info = torch.zeros(2, dtype=torch.long, device=f'cuda:{dap_rank}')
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
                torch.distributed.send(full[:, rs:re].contiguous().to(dtype), dst=r)
            chunk = full[:, row_start:row_end].contiguous().to(dtype)
            del full
            return chunk
        else:
            if ndim == 4:
                chunk = torch.empty(B, chunk_N, N, int(last_d), dtype=dtype, device=f'cuda:{dap_rank}')
            else:
                chunk = torch.empty(B, chunk_N, N, dtype=dtype, device=f'cuda:{dap_rank}')
            torch.distributed.recv(chunk, src=0)
            return chunk

    feats_chunk = {}

    if conf.add_z_input_to_z:
        # rel_pos needs 1D feats (full, for both row & col indexing)
        for key in ["asym_id", "residue_index", "entity_id", "sym_id", "token_index"]:
            if dap_rank == 0:
                t = feats[key].float()
            else:
                t = torch.empty(B, N, dtype=torch.float32, device=f'cuda:{dap_rank}')
            torch.distributed.broadcast(t, src=0)
            feats_chunk[key] = t

        if hasattr(conf, 'rel_pos') and hasattr(conf.rel_pos, 'cyclic_pos_enc') and conf.rel_pos.cyclic_pos_enc:
            if dap_rank == 0:
                t = feats["cyclic_period"].float()
            else:
                t = torch.empty(B, N, dtype=torch.float32, device=f'cuda:{dap_rank}')
            torch.distributed.broadcast(t, src=0)
            feats_chunk["cyclic_period"] = t

        # token_bonds [B,N,N] or [B,N,N,1] → scatter rows
        feats_chunk["token_bonds"] = _scatter_rows(
            feats["token_bonds"].float() if dap_rank == 0 else None, "token_bonds")

        # type_bonds [B,N,N] → scatter rows (if needed)
        if conf.bond_type_feature:
            feats_chunk["type_bonds"] = _scatter_rows(
                feats["type_bonds"].float() if dap_rank == 0 else None, "type_bonds")

        # contact_conditioning [B,N,N,C] → scatter rows
        feats_chunk["contact_conditioning"] = _scatter_rows(
            feats["contact_conditioning"].float() if dap_rank == 0 else None, "contact_conditioning")

        # contact_threshold [B,N,N] → scatter rows
        feats_chunk["contact_threshold"] = _scatter_rows(
            feats["contact_threshold"].float() if dap_rank == 0 else None, "contact_threshold")

    # Broadcast x_pred_repr for distance bins (small: [B, N, 3])
    if dap_rank == 0:
        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        if len(x_pred.shape) == 4:
            Bx, mult, N_atoms, _ = x_pred.shape
            x_pred = x_pred.reshape(Bx * mult, N_atoms, -1)
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        # x_pred_repr is [B, N, 3] — small
    else:
        x_pred_repr = torch.empty(B, N, 3, dtype=torch.float32, device=f'cuda:{dap_rank}')
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

    _cmem("pre-PF done, PF start")

    # ── Offload feats to CPU on rank 0 before PF ──
    # Phase 1 broadcasts are done; feats only needed post-PF for Phase 3.
    # R0 has 43GB alloc + 59.5GB reserved → only 18.8GB free.
    # Offloading feats (~10GB) + empty_cache (reclaims 16.5GB reserved gap)
    # gives R0 ~45GB free — enough for tri-mul's 21.7GB transient.
    feats_cpu = {}
    if dap_rank == 0:
        for key in list(feats.keys()):
            if isinstance(feats[key], torch.Tensor) and feats[key].is_cuda:
                feats_cpu[key] = feats[key].cpu()
                feats[key] = feats_cpu[key]  # replace with CPU ref
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

    # ── Reload feats to GPU on rank 0 for Phase 3 heads ──
    if dap_rank == 0 and feats_cpu:
        for key in feats_cpu:
            feats[key] = feats_cpu[key].cuda()
        del feats_cpu
        torch.cuda.empty_cache()
        _cmem("feats reloaded to GPU")

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Distributed confidence heads — PAE on chunks, PDE on gathered z
    # ══════════════════════════════════════════════════════════════════

    heads = conf.confidence_heads

    # 3a. Compute PAE logits on z_chunk BEFORE gathering z (saves ~592 MB)
    if heads.use_separate_heads:
        pae_intra_chunk = heads.to_pae_intra_logits(z_chunk)  # [B, N/dap, N, bins]
        pae_inter_chunk = heads.to_pae_inter_logits(z_chunk)
        # We'll apply intra/inter masks after gather (need full asym_id)
        pae_intra_logits = gather(pae_intra_chunk.contiguous(), dim=1, original_size=N)
        pae_inter_logits = gather(pae_inter_chunk.contiguous(), dim=1, original_size=N)
        del pae_intra_chunk, pae_inter_chunk
    else:
        pae_chunk = heads.to_pae_logits(z_chunk)  # [B, N/dap, N, 64]
        pae_logits = gather(pae_chunk.contiguous(), dim=1, original_size=N)  # [B, N, N, 64]
        del pae_chunk

    _cmem("PAE computed + gathered")

    # 3b. Gather z for PDE (needs z + z.T — requires all rows)
    z = gather(z_chunk.contiguous(), dim=1, original_size=N)  # [B, N, N, 128]
    del z_chunk

    # Gather d_chunk → full d (collective)
    d_full = gather(d_chunk.contiguous(), dim=1, original_size=N)
    del d_chunk

    _cmem("z + d gathered")

    # 3c. GPU 0: compute PDE, free z, then run metrics
    if dap_rank == 0:
        out_dict = {}

        if conf.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        # Apply intra/inter masks for separate heads PAE
        if heads.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token.unsqueeze(-1) == asym_id_token.unsqueeze(-2)
            is_different_chain = ~is_same_chain
            pae_logits = (pae_intra_logits * is_same_chain.float().unsqueeze(-1)
                         + pae_inter_logits * is_different_chain.float().unsqueeze(-1))
            del pae_intra_logits, pae_inter_logits

        # Compute PDE logits from z + z.T, then free z immediately
        z_sym = z + z.transpose(1, 2)
        if not conf.return_latent_feats:
            del z  # free z now — it's not needed anymore
        if heads.use_separate_heads:
            pde_intra = heads.to_pde_intra_logits(z_sym)
            pde_inter = heads.to_pde_inter_logits(z_sym)
            del z_sym
            pde_logits = (pde_intra * is_same_chain.float().unsqueeze(-1)
                         + pde_inter * is_different_chain.float().unsqueeze(-1))
            del pde_intra, pde_inter
        else:
            pde_logits = heads.to_pde_logits(z_sym)
            del z_sym

        _cmem("PDE done, z freed")

        # s-only heads
        resolved_logits = heads.to_resolved_logits(s)
        plddt_logits = heads.to_plddt_logits(s)

        # ── Metric aggregation (from original ConfidenceHeads.forward) ──
        from boltz.data import const
        from boltz.model.layers.confidence_utils import (
            compute_aggregated_metric,
            compute_ptms,
        )

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()

        if heads.token_level_confidence:
            plddt = compute_aggregated_metric(plddt_logits)
            token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
            complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(dim=-1)

            is_contact = (d_full < 8).float()
            is_different_chain_metric = (
                feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
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
            max_atoms_mask = feats["atom_to_token"].sum(1).unsqueeze(-1) > arange_max
            resolved_logits = resolved_logits[:, max_atoms_mask.squeeze(0)]
            resolved_logits = nn_pad(resolved_logits, (0, 0, 0, int(feats["atom_pad_mask"].shape[1] - feats["atom_pad_mask"].sum().item())), value=0)
            plddt_logits = plddt_logits.reshape(B_h, N_h, heads.max_num_atoms_per_token, -1)
            plddt_logits = plddt_logits[:, max_atoms_mask.squeeze(0)]
            plddt_logits = nn_pad(plddt_logits, (0, 0, 0, int(feats["atom_pad_mask"].shape[1] - feats["atom_pad_mask"].sum().item())), value=0)
            atom_pad_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)
            plddt = compute_aggregated_metric(plddt_logits)
            complex_plddt = (plddt * atom_pad_mask).sum(dim=-1) / atom_pad_mask.sum(dim=-1)
            token_type_f = feats["mol_type"].float()
            atom_to_token = feats["atom_to_token"].float()
            chain_id_token = feats["asym_id"].float()
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
            complex_iplddt = (plddt * feats["atom_pad_mask"] * iplddt_weight).sum(dim=-1) / torch.sum(
                feats["atom_pad_mask"] * iplddt_weight, dim=-1
            )

        # gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = torch.nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(pred_distogram_prob.device)
        contacts[:, :, :, :20] = 1.0
        prob_contact = (pred_distogram_prob * contacts).sum(-1)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1) * token_pad_mask.unsqueeze(-2)
            * (1 - torch.eye(token_pad_mask.shape[1], device=token_pad_mask.device).unsqueeze(0))
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(dim=(1, 2))
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2))
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

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
        ))
        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        try:
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feats, multiplicity
            )
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm
        except Exception as e:
            print(f"Error in compute_ptms: {e}")
            out_dict["ptm"] = torch.zeros_like(complex_plddt)
            out_dict["iptm"] = torch.zeros_like(complex_plddt)
            out_dict["ligand_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["protein_iptm"] = torch.zeros_like(complex_plddt)
            out_dict["pair_chains_iptm"] = torch.zeros_like(complex_plddt)

        return out_dict
    else:
        return {}
