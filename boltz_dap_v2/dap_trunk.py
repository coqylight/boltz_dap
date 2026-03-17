"""
DAP Trunk Wrapper for Boltz 2.

This module monkey-patches the Boltz 2 model to use DAP-aware layers
in the trunk loop. The approach:
1. ALL GPUs: Run input embedding, z_init, recycling (small ops)
2. ALL GPUs: Scatter z, run template/MSA/pairformer with DAP layers
3. ALL GPUs: Gather z back to full
4. GPU 0 ONLY: Run post-trunk (distogram, diffusion, structure, confidence)
   Non-primary GPUs skip post-trunk entirely — those modules
   aren't even loaded on their GPUs.

No model duplication — non-primary GPUs only have trunk weights.
Activations (z) are sharded across GPUs during the trunk.
"""

import torch
from torch import Tensor, nn
from typing import Optional
import time
import json
from datetime import datetime

import sys
import os

# Late-imported in _run_template_dap when dap_size > 1
# from dap_tri_att import DAPTriAttStart, DAPTriAttEnd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import scatter, gather, row_to_col, col_to_row
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_msa import DAPMSALayer
from dap_pairformer import DAPPairformerLayer
from dap_pairformer_noseq import DAPPairformerNoSeqLayer
from dap_confidence import inject_dap_into_confidence, run_confidence_dap


def inject_dap_into_model(model):
    """Inject DAP wrappers into a Boltz 2 model in-place.

    Replaces:
    - msa_module.layers[i] → DAPMSALayer (wraps MSALayer)
    - pairformer_module.layers[i] → DAPPairformerLayer (wraps PairformerLayer)
    - template_module.pairformer.layers[i] → DAPPairformerNoSeqLayer (wraps PairformerNoSeqLayer)

    Returns the model with a modified forward function.
    """
    dap_rank = get_dap_rank()
    dap_size = get_dap_size()

    if dap_size <= 1:
        print("[DAP] DAP size <= 1, wrapping layers anyway for bitwise reproducibility")

    # 1. Wrap MSA module layers
    if hasattr(model, 'msa_module'):
        msa = model.msa_module
        if hasattr(msa, '_orig_mod'):
            msa = msa._orig_mod
        for i in range(len(msa.layers)):
            original_layer = msa.layers[i]
            msa.layers[i] = DAPMSALayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(msa.layers)} MSA layers")

    # 2. Wrap main pairformer layers
    if hasattr(model, 'pairformer_module'):
        pf = model.pairformer_module
        if hasattr(pf, '_orig_mod'):
            pf = pf._orig_mod
        for i in range(len(pf.layers)):
            original_layer = pf.layers[i]
            pf.layers[i] = DAPPairformerLayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(pf.layers)} pairformer layers")

    # 3. Wrap template module's pairformer layers with DAP
    if hasattr(model, 'template_module') and model.use_templates:
        tmpl = model.template_module
        if hasattr(tmpl, '_orig_mod'):
            tmpl = tmpl._orig_mod
        pf_noseq = tmpl.pairformer
        for i in range(len(pf_noseq.layers)):
            original_layer = pf_noseq.layers[i]
            pf_noseq.layers[i] = DAPPairformerNoSeqLayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(pf_noseq.layers)} template pairformer layers")

    # 4. Wrap confidence module's pairformer layers
    if hasattr(model, 'confidence_module') and model.confidence_prediction:
        inject_dap_into_confidence(model.confidence_module)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped confidence pairformer layers")

    # 5. Store original forward and install DAP-aware forward
    model._original_forward = model.forward
    model.forward = _make_dap_forward(model)

    # 6. CRITICAL: Re-apply eval() so that newly created DAP wrapper modules
    #    (which default to training=True as new nn.Module instances) inherit
    #    training=False. Without this, dropout masks are randomly applied
    #    during inference, causing massive output divergence.
    if not model.training:
        model.eval()

    if dap_rank == 0:
        print(f"  [DAP] Installed DAP forward pass (scatter z before trunk, gather after)")

    return model


def _zs_checkpoint(label, z_scattered, s, original_N):
    """Save full z/s tensors from scattered z for comparison with baseline.
    
    z_scattered is [B, N/dap, N_padded, D]. We gather to full z.
    Returns dict with full tensors (bf16, CPU) for exact comparison.
    """
    dap_rank = get_dap_rank()
    dap_size = get_dap_size()
    
    if dap_size > 1:
        N_padded = ((original_N + dap_size - 1) // dap_size) * dap_size
        z_full = gather(z_scattered.contiguous(), dim=1, original_size=N_padded)
        if N_padded != original_N:
            z_full = z_full[:, :original_N, :original_N, :]
    else:
        z_full = z_scattered[:, :original_N, :original_N, :]
    
    zf = z_full.float()
    sf = s.float()
    
    if dap_rank == 0:
        print(f"    [CKP] [{label}]  z: mean={zf.mean():.6f} std={zf.std():.4f} "
              f"absmax={zf.abs().max():.4f}  |  s: mean={sf.mean():.6f} std={sf.std():.4f} "
              f"absmax={sf.abs().max():.4f}")
    
    result = {
        "z": z_full.cpu().to(torch.bfloat16),
        "s": s.cpu().to(torch.bfloat16),
    }
    del z_full
    return result


def _make_dap_forward(model):
    """Create a DAP-aware forward function that wraps the original.

    Key modifications:
    - After z_init is computed, scatter it across GPUs
    - After the trunk loop (template + MSA + pairformer), gather z back
    - Everything else (distogram, diffusion, structure, confidence) uses full z
    """
    original_forward = model._original_forward

    def dap_forward(
        feats,
        recycling_steps=0,
        num_sampling_steps=None,
        multiplicity_diffusion_train=1,
        diffusion_samples=1,
        max_parallel_samples=None,
        run_confidence_sequentially=False,
    ):
        dap_size = get_dap_size()
        dap_rank = get_dap_rank()
        _t0 = time.time()

        # ── Peak tracker: records every checkpoint and detects peak changes ──
        _timeline = []  # list of (elapsed, alloc, peak, label)
        _peak_changes = []  # list of (label, old_peak, new_peak) — when peak increases

        def _mem_log(label):
            """Log memory checkpoint and detect if this checkpoint set a new peak."""
            if dap_rank != 0:
                return
            elapsed = time.time() - _t0
            alloc = torch.cuda.memory_allocated() // 1024**2
            peak = torch.cuda.max_memory_allocated() // 1024**2
            prev_peak = _timeline[-1][2] if _timeline else 0
            _timeline.append((elapsed, alloc, peak, label))
            marker = ""
            if peak > prev_peak:
                _peak_changes.append((label, prev_peak, peak))
                marker = f"  ◀◀ NEW PEAK (+{peak - prev_peak}MB)"
            print(f"    [TIMELINE] {elapsed:7.1f}s | alloc={alloc:>6d}MB | peak={peak:>6d}MB | {label}{marker}")

        def _print_peak_summary():
            """Print definitive summary of where the peak was set."""
            if dap_rank != 0 or not _timeline:
                return
            final_peak = _timeline[-1][2]
            print(f"\n{'='*72}")
            print(f"  PEAK SUMMARY  (final peak = {final_peak} MB)")
            print(f"{'='*72}")
            if _peak_changes:
                for label, old, new in _peak_changes:
                    print(f"    {old:>6d}MB → {new:>6d}MB  (+{new-old:>5d}MB)  at: {label}")
                # The last peak change is the one that set the final peak
                winner = _peak_changes[-1]
                print(f"\n  ▶▶ DEFINITIVE PEAK SET BY: \"{winner[0]}\"")
                print(f"     {winner[1]}MB → {winner[2]}MB (+{winner[2]-winner[1]}MB)")
            else:
                print("    No peak changes detected (peak was 0 throughout)")
            print(f"{'='*72}\n")

        # NOTE: We no longer short-circuit for dap_size <= 1.
        # scatter/gather are no-ops when dap_size=1, so the DAP path
        # produces identical results — but we need the checkpoint-saving
        # code below to run for divergence diagnosis.

        # ── Input embedding ──────────────────────────────────────────────
        # s_inputs [B, N, token_s] is small, computed on all GPUs.
        # z_init [B, N, N, C] is large. We compute it on all GPUs because
        # scatter() is an all-to-all collective requiring all ranks to
        # participate. We immediately scatter and delete the full tensor
        # so the transient peak is brief.
        with torch.set_grad_enabled(
            model.training and model.structure_prediction_training
        ):
            if dap_rank == 0:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{'='*72}")
                print(f"  MEMORY TIMELINE  (run started {ts})")
                print(f"{'='*72}")
                torch.cuda.reset_peak_memory_stats()
            _mem_log("start")

            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            _mem_log("after input_embedder + s_init")

            # Compute masks (small, all GPUs)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            # ── Fine-grained checkpoint dict ──
            _checkpoints = {}

            # ── Compute z_init in DIRECT-LOCAL mode ──────────────────────
            # Each GPU computes ONLY its local shard of the N×N pairwise
            # tensor, never allocating the full N×N. This reduces peak
            # memory from O(N²) to O(N²/dap_size) per GPU.
            # Previous approach (scatter-first) still built full N×N
            # intermediates during rel_pos (one_hot expansions ≈ 30GB for
            # N=4642), causing OOM.
            
            original_N = s_inputs.shape[1]
            split_size = (original_N + dap_size - 1) // dap_size
            N_padded = split_size * dap_size
            row_start = dap_rank * split_size
            row_end = min(row_start + split_size, original_N)
            local_rows = row_end - row_start
            need_row_pad = split_size - local_rows  # >0 only for last rank when N not divisible

            def _pad_dim1(t):
                """Pad tensor along dim=1 if this rank has fewer rows than split_size."""
                if need_row_pad == 0:
                    return t
                pad_shape = list(t.shape)
                pad_shape[1] = need_row_pad
                return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype, device=t.device)], dim=1)

            # Step 1: Outer product — only local rows × all columns
            _z1 = model.z_init_1(s_inputs)  # [B, N, C] — small (~1MB)
            _z2 = model.z_init_2(s_inputs)  # [B, N, C] — small (~1MB)
            z_init_scattered = _pad_dim1(
                _z1[:, row_start:row_end, None, :] + _z2[:, None, :, :]
            )  # [B, split_size, N, C] — only local shard, never full N×N
            del _z1, _z2
            _mem_log("after z_init outer product (direct local)")

            # Sub-checkpoint: s_inputs
            _checkpoints["sub/s_inputs"] = {
                "s": s_inputs.cpu().to(torch.bfloat16),
                "z": torch.zeros(1),
            }
            if dap_rank == 0:
                sf = s_inputs.float()
                print(f"    [CKP] [sub/s_inputs]  s_inputs: mean={sf.mean():.6f} std={sf.std():.4f} absmax={sf.abs().max():.4f}  shape={list(sf.shape)}")

            # Sub-checkpoint: s_init
            _checkpoints["sub/s_init"] = {
                "s": s_init.cpu().to(torch.bfloat16),
                "z": torch.zeros(1),
            }
            if dap_rank == 0:
                sf = s_init.float()
                print(f"    [CKP] [sub/s_init]  s_init: mean={sf.mean():.6f} std={sf.std():.4f} absmax={sf.abs().max():.4f}  shape={list(sf.shape)}")

            # Sub-checkpoint: z_outer_product
            _checkpoints["sub/z_outer_product"] = _zs_checkpoint("sub/z_outer_product", z_init_scattered, s_init, original_N)

            # Step 2: Relative position encoding — local rows only
            # INLINE computation to avoid full N×N intermediates.
            # (model.rel_pos(feats) would create [B, N, N, 66] one_hot ×3
            #  which is ~15GB for N=4642. Local rows: only ~7GB for 2 GPUs.)
            from torch.nn.functional import one_hot as _one_hot
            _rp = model.rel_pos
            _asym_r = feats["asym_id"][:, row_start:row_end]
            _asym_c = feats["asym_id"]
            _b_same_chain = torch.eq(_asym_r[:, :, None], _asym_c[:, None, :])

            _res_r = feats["residue_index"][:, row_start:row_end]
            _res_c = feats["residue_index"]
            _b_same_res = torch.eq(_res_r[:, :, None], _res_c[:, None, :])

            _ent_r = feats["entity_id"][:, row_start:row_end]
            _ent_c = feats["entity_id"]
            _b_same_ent = torch.eq(_ent_r[:, :, None], _ent_c[:, None, :])

            _d_res = _res_r[:, :, None] - _res_c[:, None, :]
            if _rp.cyclic_pos_enc and torch.any(feats["cyclic_period"] > 0):
                _period = torch.where(
                    feats["cyclic_period"] > 0,
                    feats["cyclic_period"],
                    torch.zeros_like(feats["cyclic_period"]) + 10000,
                )
                _d_res = (_d_res - _period * torch.round(_d_res / _period)).long()
            _d_res = torch.clip(_d_res + _rp.r_max, 0, 2 * _rp.r_max)
            _d_res = torch.where(_b_same_chain, _d_res, torch.zeros_like(_d_res) + 2 * _rp.r_max + 1)
            _a_rel_pos = _one_hot(_d_res, 2 * _rp.r_max + 2)
            del _d_res

            _tok_r = feats["token_index"][:, row_start:row_end]
            _tok_c = feats["token_index"]
            _d_tok = torch.clip(_tok_r[:, :, None] - _tok_c[:, None, :] + _rp.r_max, 0, 2 * _rp.r_max)
            _d_tok = torch.where(_b_same_chain & _b_same_res, _d_tok, torch.zeros_like(_d_tok) + 2 * _rp.r_max + 1)
            _a_rel_tok = _one_hot(_d_tok, 2 * _rp.r_max + 2)
            del _d_tok, _b_same_res

            _sym_r = feats["sym_id"][:, row_start:row_end]
            _sym_c = feats["sym_id"]
            _d_chain = torch.clip(_sym_r[:, :, None] - _sym_c[:, None, :] + _rp.s_max, 0, 2 * _rp.s_max)
            _cond = (~_b_same_ent) if _rp.fix_sym_check else _b_same_chain
            _d_chain = torch.where(_cond, torch.zeros_like(_d_chain) + 2 * _rp.s_max + 1, _d_chain)
            _a_rel_chain = _one_hot(_d_chain, 2 * _rp.s_max + 2)
            del _d_chain, _b_same_chain

            _rpe_local = _rp.linear_layer(
                torch.cat([
                    _a_rel_pos.float(),
                    _a_rel_tok.float(),
                    _b_same_ent.unsqueeze(-1).float(),
                    _a_rel_chain.float(),
                ], dim=-1)
            )
            del _a_rel_pos, _a_rel_tok, _b_same_ent, _a_rel_chain

            z_init_scattered = z_init_scattered + _pad_dim1(_rpe_local)
            del _rpe_local

            # For diffusion_conditioning (rank 0 only, later): recompute
            # full rel_pos on CPU. rel_pos weights are tiny (~70KB) so
            # the CPU↔GPU transfer is negligible.
            if dap_rank == 0:
                with torch.no_grad():
                    _rp_feats_cpu = {k: feats[k].cpu() for k in
                                     ("asym_id", "residue_index", "entity_id",
                                      "token_index", "sym_id", "cyclic_period")}
                    model.rel_pos.cpu()
                    relative_position_encoding = model.rel_pos(_rp_feats_cpu)
                    model.rel_pos.cuda()
                    del _rp_feats_cpu
            else:
                relative_position_encoding = None
            _mem_log("after rel_pos_enc (direct local)")

            # Sub-checkpoint: z after rel_pos
            _checkpoints["sub/z_after_rel_pos"] = _zs_checkpoint("sub/z_after_rel_pos", z_init_scattered, s_init, original_N)

            # Step 3: Token bonds — local rows only
            _tb_local = feats["token_bonds"][:, row_start:row_end, :].float()
            z_init_scattered = z_init_scattered + _pad_dim1(model.token_bonds(_tb_local))
            del _tb_local
            if model.bond_type_feature:
                _tbt_local = feats["type_bonds"][:, row_start:row_end, :].long()
                z_init_scattered = z_init_scattered + _pad_dim1(model.token_bonds_type(_tbt_local))
                del _tbt_local
            _mem_log("after token_bonds (direct local)")

            # Sub-checkpoint: z after token_bonds
            _checkpoints["sub/z_after_token_bonds"] = _zs_checkpoint("sub/z_after_token_bonds", z_init_scattered, s_init, original_N)

            # Step 4: Contact conditioning — local rows only
            cc_feats_local = {
                "contact_conditioning": _pad_dim1(feats["contact_conditioning"][:, row_start:row_end]),
                "contact_threshold": _pad_dim1(feats["contact_threshold"][:, row_start:row_end]),
            }
            _z_component = model.contact_conditioning(cc_feats_local)
            del cc_feats_local
            z_init_scattered = z_init_scattered + _z_component
            del _z_component
            torch.cuda.empty_cache()
            _mem_log("after contact_conditioning (direct local, all z_init done)")

            # Sub-checkpoint: z after contact_conditioning (= full z_init)
            _checkpoints["sub/z_after_contact_cond"] = _zs_checkpoint("sub/z_after_contact_cond", z_init_scattered, s_init, original_N)

            # Initialize recycling tensors as SCATTERED
            s = torch.zeros_like(s_init)
            z_scattered = torch.zeros_like(z_init_scattered)
            pair_mask_scattered = scatter(pair_mask, dim=1)

            # Full init checkpoint (same as sub/z_after_contact_cond)
            cp = _zs_checkpoint("init", z_init_scattered, s_init, original_N)
            _checkpoints["init"] = cp

            if model.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        model.training
                        and model.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        if (
                            model.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Recycling: s is full (small), z stays scattered
                        s = s_init + model.s_recycle(model.s_norm(s))
                        z_scattered = z_init_scattered + model.z_recycle(
                            model.z_norm(z_scattered)
                        )

                        cp = _zs_checkpoint(f"R{i}/after_recycle", z_scattered, s, original_N)
                        _checkpoints[f"R{i}/after_recycle"] = cp

                        # Template module
                        if model.use_templates:
                            tmpl = model.template_module
                            if hasattr(tmpl, '_orig_mod') and not model.training:
                                tmpl = tmpl._orig_mod
                            z_scattered = _run_template_dap(
                                tmpl, z_scattered, feats, pair_mask,
                                model.use_kernels, original_N, _mem_log,
                                recycle_idx=i,
                            )
                            _mem_log("after template_module")

                            cp = _zs_checkpoint(f"R{i}/after_template", z_scattered, s, original_N)
                            _checkpoints[f"R{i}/after_template"] = cp

                        # MSA module
                        # CRITICAL: Original Boltz2 does z = z + msa_module(z).
                        # msa_module internally modifies z (z += opm, z = pf(z)),
                        # returns z_modified, and the outer call adds z_before again.
                        # _run_msa_dap must match this: z_final = z_before + msa_result.
                        msa = model.msa_module
                        if hasattr(msa, '_orig_mod') and not model.training:
                            msa = msa._orig_mod

                        z_before_msa = z_scattered
                        z_msa_out = _run_msa_dap(
                            msa, z_scattered, s_inputs, feats,
                            pair_mask, model.use_kernels,
                            mem_log=_mem_log,
                            _msa_diag=(i == 0),
                        )

                        # Save msa/z_out_residual granular checkpoint
                        # ALL ranks must participate in gather (collective op)
                        if i == 0:
                            _z_msa_full = gather(z_msa_out.contiguous(), dim=1, original_size=N_padded)
                            if dap_rank == 0:
                                _msa_gran = getattr(_run_msa_dap, '_gran_ckpts', {})
                                _msa_gran["msa/z_out_residual"] = _z_msa_full[:, :original_N, :original_N, :].cpu().to(torch.bfloat16)
                                _run_msa_dap._gran_ckpts = _msa_gran
                            del _z_msa_full

                        z_scattered = z_before_msa + z_msa_out
                        del z_before_msa, z_msa_out
                        _mem_log("after msa_module")

                        cp = _zs_checkpoint(f"R{i}/after_msa", z_scattered, s, original_N)
                        _checkpoints[f"R{i}/after_msa"] = cp

                        # Merge MSA granular checkpoints into granular_ckpts.pt
                        if i == 0 and dap_rank == 0 and hasattr(_run_msa_dap, '_gran_ckpts') and _run_msa_dap._gran_ckpts:
                            import os as _os
                            _out_dir = _os.environ.get('BOLTZ_OUT_DIR', '')
                            if _out_dir:
                                _gran_path = _os.path.join(_out_dir, 'granular_ckpts.pt')
                                if _os.path.exists(_gran_path):
                                    _existing = torch.load(_gran_path, map_location='cpu')
                                else:
                                    _existing = {}
                                # Trim padded tensors to original_N
                                for _gk, _gv in _run_msa_dap._gran_ckpts.items():
                                    if _gv.dim() == 4 and _gv.shape[1] > original_N:
                                        _gv = _gv[:, :original_N, :original_N, :]
                                    _existing[_gk] = _gv
                                torch.save(_existing, _gran_path)
                                print(f"      [GRAN] Merged {len(_run_msa_dap._gran_ckpts)} MSA checkpoints → {len(_existing)} total in {_gran_path}")
                                del _existing
                            _run_msa_dap._gran_ckpts = {}  # clear

                        # Pairformer module
                        pf = model.pairformer_module
                        if hasattr(pf, '_orig_mod') and not model.training:
                            pf = pf._orig_mod

                        s, z_scattered = _run_pairformer_dap(
                            pf, s, z_scattered, mask, pair_mask,
                            model.use_kernels,
                            mem_log=_mem_log
                        )
                        _mem_log("after pairformer_module")

                        cp = _zs_checkpoint(f"R{i}/after_pairformer", z_scattered, s, original_N)
                        _checkpoints[f"R{i}/after_pairformer"] = cp

                # ── OFFLOAD TRUNK WEIGHTS TO CPU FIRST ─────────────────────
                # Free ~22GB of model weights BEFORE gathering full z,
                # otherwise trunk weights + full z exceed GPU memory for 9MME.
                trunk_module_names = [
                    "input_embedder", "s_init", "z_init_1", "z_init_2",
                    "rel_pos", "token_bonds", "contact_conditioning",
                    "s_recycle", "z_recycle", "s_norm", "z_norm",
                    "msa_module", "pairformer_module", "template_module",
                ]
                if model.bond_type_feature:
                    trunk_module_names.append("token_bonds_type")
                for name in trunk_module_names:
                    if hasattr(model, name):
                        getattr(model, name).cpu()
                torch.cuda.empty_cache()
                _mem_log("after trunk offload to CPU")

                # ── GATHER z back to full and TRIM to original_N ──
                z = gather(z_scattered.contiguous(), dim=1, original_size=N_padded)
                del z_scattered
                _mem_log("after gather z (full z restored)")
                # Trim padding back to original sequence length
                if N_padded != original_N:
                    z = z[:, :original_N, :original_N, :]

                # Non-rank-0 GPUs: free the full z immediately
                # (only rank 0 needs it for post-trunk modules)
                if dap_rank != 0:
                    del z
                    z = None
                    torch.cuda.empty_cache()

                # Save checkpoints (rank 0 only)
                if dap_rank == 0:
                    import os as _os
                    _out_dir = _os.environ.get('BOLTZ_OUT_DIR', '')
                    if _out_dir:
                        _ckpt_path = _os.path.join(_out_dir, 'trunk_checkpoints.pt')
                        torch.save(_checkpoints, _ckpt_path)
                        _ckpt_size = _os.path.getsize(_ckpt_path) / (1024**2)
                        print(f"    [CKP] Saved {len(_checkpoints)} full-tensor checkpoints to {_ckpt_path} ({_ckpt_size:.0f} MB)")

            if dap_rank == 0:
                pdistogram = model.distogram_module(z)
                dict_out = {"pdistogram": pdistogram, "s": s}

                # Offload distogram after use
                model.distogram_module.cpu()
                torch.cuda.empty_cache()
                _mem_log("after distogram (offloaded)")

                # Move z to CPU BEFORE diffusion_conditioning to free ~5.5GB
                # (z_cond = cat(z, rel_pos) would need ~11GB extra on GPU)
                z_cpu = z.cpu()
                del z
                torch.cuda.empty_cache()
                _mem_log("after z offloaded to CPU")
            else:
                dict_out = {"s": s}

            # GPU 0: runs distogram, diffusion, structure (GPU 1 waits)
            # Both GPUs: participate in confidence pairformer DAP

            if (
                model.run_trunk_and_structure
                and ((not model.training) or model.confidence_prediction)
                and (not model.skip_run_structure)
            ):
                if dap_rank == 0:
                    # ── Inlined diffusion_conditioning with chunked Transitions ──
                    _mem_log("before diffusion_conditioning (inlined)")
                    dc = model.diffusion_conditioning

                    # ① PairwiseConditioning — chunked along rows to avoid OOM
                    # Without chunking, transition hidden expansion [N,N,512] = 22GB
                    _mem_log("  dc: before pairwise_conditioner")
                    pw = dc.pairwise_conditioner
                    N = z_cpu.shape[1]
                    chunk_size = 512  # Process 512 rows at a time

                    # Allocate output z_cond on GPU
                    # Determine output dim by probing with a small tensor
                    _probe_in = torch.zeros(1, 1, 1, z_cpu.shape[-1] + relative_position_encoding.shape[-1],
                                           dtype=z_cpu.dtype, device="cuda")
                    _probe_out = pw.dim_pairwise_init_proj(_probe_in)
                    out_dim = _probe_out.shape[-1]
                    del _probe_in, _probe_out
                    z_cond = torch.empty(1, N, N, out_dim, dtype=z_cpu.dtype, device="cuda")

                    for row_start in range(0, N, chunk_size):
                        row_end = min(row_start + chunk_size, N)
                        # Slice z_cpu and rel_pos on CPU, concat, move chunk to GPU
                        chunk_in = torch.cat(
                            (z_cpu[:, row_start:row_end],
                             relative_position_encoding[:, row_start:row_end]),
                            dim=-1
                        ).cuda()
                        chunk_out = pw.dim_pairwise_init_proj(chunk_in)
                        del chunk_in
                        # Apply transitions in-place per chunk
                        for transition in pw.transitions:
                            chunk_out = transition(chunk_out) + chunk_out
                        z_cond[:, row_start:row_end] = chunk_out
                        del chunk_out

                    del relative_position_encoding  # Free CPU tensor
                    torch.cuda.empty_cache()
                    _mem_log("  dc: after pairwise_conditioner (row-chunked)")

                    # Offload pairwise_conditioner weights — no longer needed
                    pw.cpu()
                    torch.cuda.empty_cache()
                    _mem_log("  dc: pairwise_conditioner offloaded")

                    # ② Token transformer biases — row-chunked like pairwise_conditioner
                    # 24 projections of z_cond [B,N,N,128]→[B,N,N,8] each, cat to [B,N,N,192]
                    _mem_log("  dc: before token_trans_bias")
                    layers = dc.token_trans_proj_z
                    n_layers = len(layers)
                    # Probe output dim per layer
                    _p_in = torch.zeros(1, 1, 1, out_dim, dtype=z_cond.dtype, device="cuda")
                    _p_out = layers[0](_p_in)
                    per_layer_dim = _p_out.shape[-1]
                    del _p_in, _p_out
                    total_bias_dim = n_layers * per_layer_dim

                    token_trans_bias = torch.empty(1, N, N, total_bias_dim, dtype=z_cond.dtype, device="cuda")
                    chunk_size_ttb = 256  # Smaller chunks for 24 projections
                    for row_start in range(0, N, chunk_size_ttb):
                        row_end = min(row_start + chunk_size_ttb, N)
                        z_chunk = z_cond[:, row_start:row_end]  # [1, chunk, N, 128]
                        token_trans_bias[:, row_start:row_end] = torch.cat(
                            [layer(z_chunk) for layer in layers], dim=-1
                        )
                        del z_chunk
                    _mem_log("  dc: after token_trans_bias (row-chunked)")

                    # Offload token_trans_proj_z weights
                    for layer in dc.token_trans_proj_z:
                        layer.cpu()
                    # Offload token_trans_bias to CPU before atom_encoder (~8.2GB freed)
                    token_trans_bias_cpu = token_trans_bias.cpu()
                    del token_trans_bias
                    # Offload structure_module + confidence weights (not needed until later)
                    # These are still on GPU from model init, taking ~10-15GB
                    if hasattr(model, 'structure_module'):
                        model.structure_module.cpu()
                    if hasattr(model, 'confidence_module'):
                        model.confidence_module.cpu()
                    # Offload atom_enc/dec_proj_z (tiny but every MB counts)
                    for layer in dc.atom_enc_proj_z:
                        layer.cpu()
                    for layer in dc.atom_dec_proj_z:
                        layer.cpu()
                    torch.cuda.empty_cache()
                    _mem_log("  dc: struct/conf/bias all offloaded before atom_enc")

                    # ③ AtomEncoder — pre-compute z_to_p in row-chunks to avoid
                    # materializing full z.float() (11GB for N=4642)
                    _mem_log("  dc: before atom_encoder (chunked z_to_p)")
                    
                    ae = dc.atom_encoder
                    z_to_p_trans = ae.z_to_p_trans
                    
                    # Probe output dim of z_to_p_trans
                    _probe = torch.zeros(1, 1, 1, z_cond.shape[-1], dtype=torch.float32, device="cuda")
                    z_to_p_dim = z_to_p_trans(_probe).shape[-1]
                    del _probe
                    
                    # Pre-compute z_to_p in row-chunks
                    z_to_p_full = torch.empty(1, N, N, z_to_p_dim, dtype=torch.float32, device="cuda")
                    chunk_ae = 256
                    for row_start in range(0, N, chunk_ae):
                        row_end = min(row_start + chunk_ae, N)
                        z_chunk = z_cond[:, row_start:row_end].float()  # [1, chunk, N, 128] fp32
                        z_to_p_full[:, row_start:row_end] = z_to_p_trans(z_chunk)
                        del z_chunk
                    _mem_log("  dc: z_to_p pre-computed (row-chunked)")
                    
                    # Free z_cond — no longer needed (z_to_p already computed)
                    del z_cond
                    torch.cuda.empty_cache()
                    _mem_log("  dc: z_cond freed before atom_enc forward")
                    
                    # Monkey-patch z_to_p_trans to return pre-computed result
                    # nn.Module.__setattr__ rejects non-Module, so wrap in a Module
                    class _PrecomputedZToP(torch.nn.Module):
                        def __init__(self, result):
                            super().__init__()
                            self._result = result
                        def forward(self, x):
                            return self._result
                    
                    original_z_to_p_trans = ae.z_to_p_trans
                    ae.z_to_p_trans = _PrecomputedZToP(z_to_p_full)
                    
                    # Create a dummy z — z_to_p_trans is monkey-patched so z.float()
                    # result is discarded. Just needs valid shape for other parts of forward.
                    dummy_z = torch.zeros(1, N, N, 1, dtype=torch.float32, device="cuda")
                    
                    q, c, p, to_keys = ae(
                        feats=feats, s_trunk=s, z=dummy_z,
                    )
                    
                    # Restore original z_to_p_trans
                    ae.z_to_p_trans = original_z_to_p_trans
                    del z_to_p_full, dummy_z
                    torch.cuda.empty_cache()
                    _mem_log("  dc: after atom_encoder")




                    # Offload diffusion_conditioning weights
                    dc.cpu()
                    torch.cuda.empty_cache()
                    _mem_log("after diffusion_conditioning (offloaded)")

                    # z is already on CPU (offloaded before diffusion_conditioning)
                    dict_out["z"] = z_cpu
                    _mem_log("  mem_opt: z already on CPU, stored in dict_out")

                    # token_trans_bias already on CPU (offloaded before atom_encoder)
                    _mem_log("  mem_opt: token_trans_bias already on CPU")

                    # Reload atom_enc/dec_proj_z for bias computation
                    for layer in dc.atom_enc_proj_z:
                        layer.cuda()
                    for layer in dc.atom_dec_proj_z:
                        layer.cuda()

                    # Compute atom biases (p is from atom_encoder, on GPU)
                    atom_enc_bias = torch.cat([layer(p) for layer in dc.atom_enc_proj_z], dim=-1)
                    atom_dec_bias = torch.cat([layer(p) for layer in dc.atom_dec_proj_z], dim=-1)
                    del p
                    # Offload atom proj z after use
                    for layer in dc.atom_enc_proj_z:
                        layer.cpu()
                    for layer in dc.atom_dec_proj_z:
                        layer.cpu()
                    torch.cuda.empty_cache()
                    _mem_log("  dc: atom biases computed, proj offloaded")

                    diffusion_conditioning = {
                        "q": q, "c": c, "to_keys": to_keys,
                        "atom_enc_bias": atom_enc_bias,
                        "atom_dec_bias": atom_dec_bias,
                        "token_trans_bias": token_trans_bias_cpu,
                    }

                    _mem_log("before structure_module.sample")

                # ── Distributed Diffusion Sampling ──────────────────────
                # Instead of rank 0 running ALL diffusion samples alone,
                # broadcast conditioning to all ranks and split samples.
                # Each rank runs an independent subset → ~4× memory savings.
                import torch.distributed as _tdist

                if dap_size > 1 and diffusion_samples > 1:
                    # ── Phase 1: Broadcast conditioning data to all ranks ──
                    _mem_log("  dist_diff: broadcasting conditioning")

                    # s and s_inputs are already replicated on all ranks after trunk.
                    # (s is replicated by pairformer DAP, s_inputs computed on all ranks)
                    # No broadcast needed for either.

                    # Broadcast diffusion_conditioning dict
                    if dap_rank == 0:
                        # Send shape metadata first
                        dc_meta = {}
                        for k, v in diffusion_conditioning.items():
                            if isinstance(v, torch.Tensor):
                                dc_meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
                        import pickle
                        meta_bytes = pickle.dumps(dc_meta)
                        meta_size = torch.tensor([len(meta_bytes)], dtype=torch.long, device="cuda")
                    else:
                        meta_size = torch.tensor([0], dtype=torch.long, device="cuda")
                        diffusion_conditioning = {}

                    _tdist.broadcast(meta_size, src=0)

                    if dap_rank == 0:
                        meta_tensor = torch.frombuffer(bytearray(meta_bytes), dtype=torch.uint8).cuda()
                    else:
                        meta_tensor = torch.empty(meta_size.item(), dtype=torch.uint8, device="cuda")
                    _tdist.broadcast(meta_tensor, src=0)

                    if dap_rank != 0:
                        import pickle
                        dc_meta = pickle.loads(meta_tensor.cpu().numpy().tobytes())
                    del meta_tensor

                    # Broadcast each conditioning tensor
                    _dtype_map = {
                        "torch.float32": torch.float32, "torch.float16": torch.float16,
                        "torch.bfloat16": torch.bfloat16, "torch.int64": torch.int64,
                    }
                    for k, meta in dc_meta.items():
                        shape = meta["shape"]
                        dtype = _dtype_map.get(meta["dtype"], torch.float32)
                        is_cpu = "cpu" in meta["device"]

                        if dap_rank != 0:
                            diffusion_conditioning[k] = torch.empty(shape, dtype=dtype, device="cuda")

                        # If on CPU (e.g. token_trans_bias), move to GPU for broadcast
                        if dap_rank == 0 and is_cpu:
                            diffusion_conditioning[k] = diffusion_conditioning[k].cuda()

                        _tdist.broadcast(diffusion_conditioning[k], src=0)

                        # Move back to CPU if it was CPU-offloaded (saves GPU memory)
                        if is_cpu:
                            diffusion_conditioning[k] = diffusion_conditioning[k].cpu()

                    _mem_log("  dist_diff: conditioning broadcast complete")

                    # ── Phase 2: Split samples across ranks ──
                    # With FK steering, multiplicity is expanded by num_particles.
                    # Split at the sample level (before FK expansion) to keep
                    # FK particle groups intact on each rank.
                    base_per_rank = diffusion_samples // dap_size
                    remainder = diffusion_samples % dap_size
                    local_samples = base_per_rank + (1 if dap_rank < remainder else 0)

                    _mem_log(f"  dist_diff: rank {dap_rank} running {local_samples}/{diffusion_samples} samples")

                    # ── Phase 3: Each rank runs structure_module.sample() ──
                    model.structure_module.cuda()
                    _mem_log("  dist_diff: struct module reloaded to GPU")

                    if local_samples > 0:
                        with torch.autocast("cuda", enabled=False):
                            local_structure_output = model.structure_module.sample(
                                s_trunk=s.float(),
                                s_inputs=s_inputs.float(),
                                feats=feats,
                                num_sampling_steps=num_sampling_steps,
                                atom_mask=feats["atom_pad_mask"].float(),
                                multiplicity=local_samples,
                                max_parallel_samples=max_parallel_samples,
                                steering_args=getattr(model, 'steering_args', None),
                                diffusion_conditioning=diffusion_conditioning,
                            )
                        local_coords = local_structure_output["sample_atom_coords"]
                    else:
                        # This rank has no samples (edge case: more GPUs than samples)
                        n_atoms = feats["atom_pad_mask"].shape[1]
                        local_coords = torch.empty(0, n_atoms, 3, dtype=torch.float32, device="cuda")

                    _mem_log("  dist_diff: local sampling complete")
                    model.structure_module.cpu()
                    # Free conditioning on non-primary ranks
                    if dap_rank != 0:
                        del diffusion_conditioning
                    torch.cuda.empty_cache()

                    # ── Phase 4: Gather sample_atom_coords to rank 0 ──
                    # Collect local_coords sizes from all ranks
                    local_count = torch.tensor([local_coords.shape[0]], dtype=torch.long, device="cuda")
                    all_counts = [torch.tensor([0], dtype=torch.long, device="cuda") for _ in range(dap_size)]
                    _tdist.all_gather(all_counts, local_count)
                    counts = [c.item() for c in all_counts]

                    if dap_rank == 0:
                        # Allocate buffer for all samples
                        total_samples = sum(counts)
                        n_atoms = local_coords.shape[1]
                        _coord_dtype = local_coords.dtype
                        all_coords = torch.empty(total_samples, n_atoms, 3, dtype=_coord_dtype, device="cuda")
                        offset = 0
                        all_coords[0:counts[0]] = local_coords
                        del local_coords  # free rank 0's local copy
                        offset += counts[0]
                        for src_rank in range(1, dap_size):
                            if counts[src_rank] > 0:
                                recv_buf = torch.empty(counts[src_rank], n_atoms, 3, dtype=_coord_dtype, device="cuda")
                                _tdist.recv(recv_buf, src=src_rank)
                                all_coords[offset:offset + counts[src_rank]] = recv_buf
                                del recv_buf
                            offset += counts[src_rank]
                    else:
                        if local_coords.shape[0] > 0:
                            _tdist.send(local_coords, dst=0)
                        del local_coords

                    torch.cuda.empty_cache()

                    if dap_rank == 0:
                        structure_output = {"sample_atom_coords": all_coords, "diff_token_repr": None}
                        dict_out.update(structure_output)
                        del all_coords

                    _mem_log("  dist_diff: gather complete")

                else:
                    # Single GPU or single sample — original path
                    if dap_rank == 0:
                        # Reload structure_module to GPU (was offloaded before atom_encoder)
                        model.structure_module.cuda()
                        _mem_log("  struct module reloaded to GPU")

                        # Structure module (rank 0 only, then offload)
                        with torch.autocast("cuda", enabled=False):
                            structure_output = model.structure_module.sample(
                                s_trunk=s.float(),
                                s_inputs=s_inputs.float(),
                                feats=feats,
                                num_sampling_steps=num_sampling_steps,
                                atom_mask=feats["atom_pad_mask"].float(),
                                multiplicity=diffusion_samples,
                                max_parallel_samples=max_parallel_samples,
                                steering_args=getattr(model, 'steering_args', None),
                                diffusion_conditioning=diffusion_conditioning,
                            )
                            dict_out.update(structure_output)

                        _mem_log("after structure_module.sample")
                        model.structure_module.cpu()
                        del diffusion_conditioning
                        torch.cuda.empty_cache()

                if dap_rank == 0:
                    # When DAP>1: keep z on CPU so run_confidence_dap can scatter-from-CPU (avoids OOM).
                    # When single GPU: need z on GPU for model.confidence_module().
                    if dap_size > 1:
                        dict_out["z"] = z_cpu
                        _mem_log("after structure_module (offloaded, z kept on CPU for confidence DAP)")
                    else:
                        z = z_cpu.cuda()
                        dict_out["z"] = z
                        del z_cpu
                        _mem_log("after structure_module (offloaded, z restored)")

                    if model.predict_bfactor:
                        dict_out["bfactors_logits"] = model.bfactor_module(s)
                        model.bfactor_module.cpu()
                        torch.cuda.empty_cache()

                # Sync before confidence DAP (GPU 1 was waiting)
                if dap_size > 1:
                    # Release stale CUDA reserved memory from trunk on ALL ranks
                    # (ranks 1-3 peaked at ~70GB during trunk PF, still reserved)
                    torch.cuda.empty_cache()
                    import torch.distributed as tdist
                    tdist.barrier()

                # Confidence module with DAP (ALL GPUs participate)
                if model.confidence_prediction:
                    if dap_size > 1:
                        # Grab conf-specific data before offloading dict_out
                        if dap_rank == 0:
                            conf_x_pred = dict_out.get("sample_atom_coords", feats["coords"])
                            # .contiguous() on [B,N,N] slice (9 MB) so full pdistogram
                            # [B,N,N,64] (590 MB) can be freed by CPU offload
                            conf_pdist = dict_out["pdistogram"][:, :, :, 0].contiguous()
                        else:
                            conf_x_pred = torch.empty(0)
                            conf_pdist = torch.empty(0)

                        # ── Offload ALL dict_out to CPU ──
                        # z/s/pdistogram/structure_outputs all go to CPU.
                        # The local vars z, s, s_inputs still hold GPU refs.
                        if dap_rank == 0:
                            for key in list(dict_out.keys()):
                                if isinstance(dict_out[key], torch.Tensor) and dict_out[key].is_cuda:
                                    dict_out[key] = dict_out[key].cpu()
                            torch.cuda.empty_cache()

                        _mem_log("before confidence (dict_out offloaded)")

                        # Reload confidence_module to GPU (was offloaded before atom_encoder)
                        if hasattr(model, 'confidence_module'):
                            model.confidence_module.cuda()
                        _mem_log("  confidence module reloaded to GPU")

                        # Pass z via mutable list — run_confidence_dap clears z_holder[0] after scatter.
                        # Fix (job 353074): Rank 1,2,3 never have z_cpu in scope → use [None] to avoid
                        # UnboundLocalError. Only Rank 0 has z_cpu (kept on CPU for scatter-from-CPU in DAP).
                        if dap_rank == 0:
                            z_holder = [z_cpu]
                        else:
                            z_holder = [None]
                        confidence_output = run_confidence_dap(
                            model,
                            s_inputs=s_inputs,
                            s=s,
                            z_holder=z_holder,
                            x_pred=conf_x_pred,
                            feats=feats,
                            pred_distogram_logits=conf_pdist,
                            multiplicity=diffusion_samples,
                            run_sequentially=run_confidence_sequentially,
                            use_kernels=model.use_kernels,
                        )
                        # Release remaining local GPU refs
                        del s, s_inputs, conf_x_pred, conf_pdist, z_holder

                        _mem_log("after confidence_module")
                    else:
                        confidence_output = model.confidence_module(
                            s_inputs=s_inputs.detach(),
                            s=s.detach(),
                            z=z.detach(),
                            x_pred=(
                                dict_out["sample_atom_coords"].detach()
                                if not model.skip_run_structure
                                else feats["coords"].repeat_interleave(diffusion_samples, 0)
                            ),
                            feats=feats,
                            pred_distogram_logits=dict_out["pdistogram"][:, :, :, 0].detach(),
                            multiplicity=diffusion_samples,
                            run_sequentially=run_confidence_sequentially,
                            use_kernels=model.use_kernels,
                        )
                    if dap_rank == 0:
                        dict_out.update(confidence_output)
                        # Move any CPU-offloaded tensors back to GPU for writer.
                        # Keep large confidence tensors on CPU to avoid OOM (e.g. hexamer 13 samples).
                        _confidence_cpu_keys = frozenset({
                            "pae_logits", "pde_logits", "pde", "pae", "plddt",
                            "complex_pde", "complex_pae", "complex_plddt", "complex_iplddt", "complex_ipde",
                            "ptm", "iptm", "ligand_iptm", "protein_iptm",
                            "s_conf", "z_conf",
                        })
                        for key in list(dict_out.keys()):
                            if key in _confidence_cpu_keys:
                                continue
                            if isinstance(dict_out[key], torch.Tensor) and not dict_out[key].is_cuda:
                                dict_out[key] = dict_out[key].cuda(0)
                    model.confidence_module.cpu()
                    torch.cuda.empty_cache()

        _print_peak_summary()

        # Save memory timeline as JSON for graphing
        if dap_rank == 0 and _timeline:
            _out_dir = os.environ.get('BOLTZ_OUT_DIR', '')
            if _out_dir:
                _mem_path = os.path.join(_out_dir, 'mem_timeline.json')
                _records = [{"step": lbl, "alloc_mb": a, "peak_mb": p, "elapsed_s": round(t, 2)} for t, a, p, lbl in _timeline]
                with open(_mem_path, 'w') as _mf:
                    json.dump(_records, _mf, indent=2)
                print(f"    [MEM] Saved {len(_records)} memory timeline records to {_mem_path}")

        return dict_out

    return dap_forward


def _run_template_dap(tmpl_module, z_scattered, feats, pair_mask, use_kernels, original_N, mem_log=None, recycle_idx=0):
    """Run template module with DAP-scattered z.

    Computes template features in SCATTERED form (N/dap rows × N cols)
    so no GPU ever holds a full N×N tensor. Diagnostic gathers are gated
    behind BOLTZ_TEMPLATE_DEBUG to avoid OOM on large inputs.

    z_scattered: [B, N/dap, N_padded, D] — row-scattered
    pair_mask: [B, N, N] — full (unscattered)
    Returns: z_scattered + template_output (still row-scattered)
    """
    from boltz.data import const
    from torch.nn.functional import one_hot

    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    _tmpl_debug = os.environ.get("BOLTZ_TEMPLATE_DEBUG", "0") == "1"

    # Granular checkpoint dict (R0 only, rank 0 only, debug only)
    _gran_ckpts = {} if _tmpl_debug else None

    # Load template features (all on full N — these are per-residue, small)
    res_type = feats["template_restype"]            # [B, T, N, C]
    frame_rot = feats["template_frame_rot"]         # [B, T, N, 3, 3]
    frame_t = feats["template_frame_t"]             # [B, T, N, 3]
    frame_mask = feats["template_mask_frame"]       # [B, T, N]
    cb_coords = feats["template_cb"]                # [B, T, N, 3]
    ca_coords = feats["template_ca"]                # [B, T, N, 3]
    cb_mask = feats["template_mask_cb"]             # [B, T, N]
    visibility_ids = feats["visibility_ids"]        # [B, T, N] — per-template!
    template_mask = feats["template_mask"].any(dim=2).float()  # [B, T]
    num_templates = template_mask.sum(dim=1).clamp(min=1)      # [B]

    B, T = res_type.shape[:2]
    N = original_N

    if mem_log:
        mem_log("  template: loaded per-residue features")

    # ── Scatter row-dimension inputs BEFORE feature computation ──
    # Each GPU computes only its row shard of the N×N features.
    # Column dimension stays full (N) — row dimension becomes N/dap.
    if dap_size > 1:
        def _scat_bt(x, last_dims):
            """Scatter on the N (residue) dimension for [B, T, N, ...] tensors."""
            shape = x.shape  # [B, T, N, ...]
            x_flat = x.reshape(B * T, N, *shape[3:])  # [B*T, N, ...]
            x_scat = scatter(x_flat, dim=1)  # [B*T, N/dap, ...]
            return x_scat.reshape(B, T, x_scat.shape[1], *shape[3:])

        cb_coords_scat = _scat_bt(cb_coords, 3)      # [B, T, N/dap, 3]
        ca_coords_scat = _scat_bt(ca_coords, 3)      # [B, T, N/dap, 3]
        frame_rot_scat = _scat_bt(frame_rot, (3, 3)) # [B, T, N/dap, 3, 3]
        frame_t_scat = _scat_bt(frame_t, 3)          # [B, T, N/dap, 3]
        frame_mask_scat = _scat_bt(frame_mask.unsqueeze(-1), 1).squeeze(-1)  # [B, T, N/dap]
        cb_mask_scat = _scat_bt(cb_mask.unsqueeze(-1), 1).squeeze(-1)        # [B, T, N/dap]
        res_type_scat = _scat_bt(res_type, res_type.shape[-1])               # [B, T, N/dap, C]
        vis_ids_scat = _scat_bt(visibility_ids.unsqueeze(-1), 1).squeeze(-1) # [B, T, N/dap]
        N_scat = cb_coords_scat.shape[2]
    else:
        cb_coords_scat = cb_coords
        ca_coords_scat = ca_coords
        frame_rot_scat = frame_rot
        frame_t_scat = frame_t
        frame_mask_scat = frame_mask
        cb_mask_scat = cb_mask
        res_type_scat = res_type
        vis_ids_scat = visibility_ids
        N_scat = N

    if mem_log:
        mem_log("  template: scattered row-dim coords")

    # ── Compute features in scattered form: [B,T,N/dap,N,...] ──
    # Row dim = scattered (N/dap), col dim = full (N). No full N×N anywhere.
    # Pairwise masks: [B, T, N/dap, N, 1]
    b_cb_mask = (cb_mask_scat[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
    b_frame_mask = (frame_mask_scat[:, :, :, None] * frame_mask[:, :, None, :])[..., None]

    # Template pair mask: [B, T, N/dap, N]
    tmlp_pair_mask = (vis_ids_scat[:, :, :, None] == visibility_ids[:, :, None, :]).float()

    if mem_log:
        mem_log("  template: computed pairwise masks (scattered)")

    # Compute template pair features in scattered form
    with torch.autocast(device_type="cuda", enabled=False):
        # Distogram: cdist(scattered_rows, all_cols) → [B, T, N/dap, N]
        cb_dists = torch.cdist(cb_coords_scat, cb_coords)
        boundaries = torch.linspace(tmpl_module.min_dist, tmpl_module.max_dist,
                                     tmpl_module.num_bins - 1).to(cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=tmpl_module.num_bins)  # [B,T,N/dap,N,bins]
        del cb_dists

        if mem_log:
            mem_log("  template: distogram computed (scattered)")

        # Unit vector: scattered frame × full ca → [B, T, N/dap, N, 3]
        frame_rot_t = frame_rot_scat.unsqueeze(3).transpose(-1, -2)  # [B,T,N/dap,1,3,3]
        frame_t_exp = frame_t_scat.unsqueeze(3).unsqueeze(-1)         # [B,T,N/dap,1,3,1]
        ca_exp = ca_coords.unsqueeze(2).unsqueeze(-1)                 # [B,T,1,N,3,1]
        vector = torch.matmul(frame_rot_t, (ca_exp - frame_t_exp))   # [B,T,N/dap,N,3,1]
        norm = torch.norm(vector, dim=-2, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)  # [B,T,N/dap,N,3]
        del frame_rot_t, frame_t_exp, ca_exp, vector, norm

        if mem_log:
            mem_log("  template: unit_vector computed (scattered)")

        # Concatenate and project: a_tij [B, T, N/dap, N, template_dim]
        a_tij = torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
        a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)
        del distogram, b_cb_mask, unit_vector, b_frame_mask

        res_type_i = res_type_scat[:, :, :, None].expand(-1, -1, -1, N, -1)  # [B,T,N/dap,N,C]
        res_type_j = res_type[:, :, None, :].expand(-1, -1, N_scat, -1, -1)  # [B,T,N/dap,N,C]
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        del res_type_i, res_type_j
        a_tij = tmpl_module.a_proj(a_tij)  # [B, T, N/dap, N, template_dim]

    # Free scattered coords (no longer needed)
    del cb_coords_scat, ca_coords_scat, frame_rot_scat, frame_t_scat
    del frame_mask_scat, cb_mask_scat, res_type_scat, vis_ids_scat, tmlp_pair_mask

    _ri = recycle_idx  # alias for compact logging

    if mem_log:
        mem_log("  template: a_tij computed (scattered, no full N×N)")

    # Save upstream template checkpoints (debug only — requires gather)
    if _tmpl_debug and recycle_idx == 0:
        # Gather a_tij for comparison (collective)
        _a_full = gather(a_tij.reshape(B * T, *a_tij.shape[2:]).contiguous(), dim=1)
        if dap_rank == 0:
            _af = _a_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] a_tij: mean={_af.mean():.6f} std={_af.std():.4f} absmax={_af.abs().max():.4f} (gathered)")
            _gran_ckpts["tmpl/a_tij"] = _a_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _a_full

    # a_tij is already [B, T, N/dap, N, D] — no scatter needed!

    # z_scattered is [B, N/dap, N_padded, D]. Trim to original_N for template ops
    # and project: z_proj(z_norm(z_scattered[:,None])) → [B, 1, N/dap, N, template_dim]
    z_for_tmpl = z_scattered[:, :, :original_N, :]  # [B, N/dap, N, D]
    z_proj_out = tmpl_module.z_proj(tmpl_module.z_norm(z_for_tmpl[:, None]))

    if mem_log:
        mem_log("  template: z_proj computed")

    # Save z_proj output (debug only — requires gather)
    if _tmpl_debug and recycle_idx == 0:
        _zp_bt = z_proj_out.reshape(-1, *z_proj_out.shape[2:])  # [B*1, N/dap, N, D]
        _zp_full = gather(_zp_bt.contiguous(), dim=1)
        if dap_rank == 0:
            _zpf = _zp_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] z_proj_out: mean={_zpf.mean():.6f} std={_zpf.std():.4f} absmax={_zpf.abs().max():.4f} (gathered)")
            _gran_ckpts["tmpl/z_proj_out"] = _zp_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _zp_full, _zp_bt

    v = z_proj_out + a_tij
    del a_tij, z_proj_out

    # Log v_input stats (debug only)
    if _tmpl_debug:
        _v_full = gather(v.reshape(B * T, *v.shape[2:]).contiguous(), dim=1)
        if dap_rank == 0:
            _vf = _v_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] v_input: mean={_vf.mean():.6f} std={_vf.std():.4f} absmax={_vf.abs().max():.4f} (gathered)")
            if recycle_idx == 0:
                _gran_ckpts["tmpl/v_input"] = _v_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _v_full

    if mem_log:
        mem_log("  template: v = z_proj + a_tij")

    # Prepare pair_mask for scattered template PF
    # pair_mask is [B, N, N]. We need it scattered: [B, N/dap, N]
    pair_mask_tmpl = pair_mask[:, :, :original_N]  # ensure original_N
    if dap_size > 1:
        pair_mask_tmpl = scatter(pair_mask_tmpl, dim=1)  # [B, N/dap, N]

    # Expand mask for T templates: [B*T, N/dap, N]
    pair_mask_tmpl = pair_mask_tmpl[:, None].expand(-1, T, -1, -1)
    pair_mask_tmpl = pair_mask_tmpl.reshape(B * T, *pair_mask_tmpl.shape[2:])

    # v: [B, T, N/dap, N, template_dim] → [B*T, N/dap, N, template_dim]
    v = v.view(B * T, *v.shape[2:])

    if mem_log:
        mem_log("  template: v scattered, entering PF")

    # Run DAP-wrapped pairformer (2 layers)
    # Set chunk size
    if not tmpl_module.pairformer.training:
        if original_N > const.chunk_size_threshold:
            # For very long sequences, use even smaller chunks to prevent OOM
            # Q@K^T shape: [chunk, H, N, N] × 4 bytes
            chunk_size_tri_attn = 8 if original_N > 2000 else 32
        else:
            chunk_size_tri_attn = 128
    else:
        chunk_size_tri_attn = None

    pf_input = v
    from boltz.model.layers.pairformer import get_dropout_mask as _pfnoseq_dropout

    def _save_subop_gather(label, z_scat, li):
        """Gather scattered z for sub-op checkpoint (debug only, collective)."""
        if not _tmpl_debug:
            return
        _full = gather(z_scat.contiguous(), dim=1)
        if dap_rank == 0 and recycle_idx == 0:
            _trimmed = _full[:, :N, :N, :].cpu().to(torch.bfloat16)
            _gran_ckpts[f"tmpl/pf{li}/{label}"] = _trimmed
            _fl = _full[:, :N, :N, :].float()
            print(f"        [SUBOP pf{li}] {label}: mean={_fl.mean():.6f} std={_fl.std():.4f} absmax={_fl.abs().max():.4f}")
        del _full

    for _li, layer in enumerate(tmpl_module.pairformer.layers):
        # IMPORTANT: Force use_kernels=False so both 1-GPU (original layers)
        # and 2-GPU (DAP-wrapped layers) use the same PyTorch-native code path.
        _tmpl_use_kernels = False
        _save_subop_gather("input", pf_input, _li)

        # 1. tri_mul_out
        dropout = _pfnoseq_dropout(layer.dropout, pf_input, layer.training)
        pf_input = pf_input + dropout * layer.tri_mul_out(
            pf_input, mask=pair_mask_tmpl, use_kernels=_tmpl_use_kernels
        )
        if mem_log:
            mem_log(f"  template: PF[{_li}] after tri_mul_out")
        _save_subop_gather("after_tri_mul_out", pf_input, _li)

        # 2. tri_mul_in — needs row_to_col transpose for DAP
        if dap_size > 1:
            z_col = row_to_col(pf_input)
            pair_mask_col = row_to_col(pair_mask_tmpl.unsqueeze(-1)).squeeze(-1)
            # Zero out padded positions
            N_pad = z_col.shape[1]
            if N_pad > original_N:
                z_col[:, original_N:, :, :] = 0
                pair_mask_col[:, original_N:, :] = 0
            dropout = _pfnoseq_dropout(layer.dropout, z_col, layer.training)
            z_col = z_col + dropout * layer.tri_mul_in(
                z_col, mask=pair_mask_col, use_kernels=_tmpl_use_kernels
            )
            pf_input = col_to_row(z_col)
            del z_col
            if pf_input.shape[2] > original_N:
                pf_input = pf_input[:, :, :original_N, :]
        else:
            dropout = _pfnoseq_dropout(layer.dropout, pf_input, layer.training)
            pf_input = pf_input + dropout * layer.tri_mul_in(
                pf_input, mask=pair_mask_tmpl, use_kernels=_tmpl_use_kernels
            )
        if mem_log:
            mem_log(f"  template: PF[{_li}] after tri_mul_in")
        _save_subop_gather("after_tri_mul_in", pf_input, _li)

        # 3. tri_att_start (already DAP-wrapped via DAPPairformerNoSeqLayer injection)
        dropout = _pfnoseq_dropout(layer.dropout, pf_input, layer.training)
        pf_input = pf_input + dropout * layer.tri_att_start(
            pf_input, mask=pair_mask_tmpl, chunk_size=chunk_size_tri_attn,
            use_kernels=_tmpl_use_kernels,
        )
        if mem_log:
            mem_log(f"  template: PF[{_li}] after tri_att_start")
        _save_subop_gather("after_tri_att_start", pf_input, _li)

        # 4. tri_att_end (already DAP-wrapped via DAPPairformerNoSeqLayer injection)
        dropout = _pfnoseq_dropout(layer.dropout, pf_input, layer.training, columnwise=True)
        pf_input = pf_input + dropout * layer.tri_att_end(
            pf_input, mask=pair_mask_tmpl, chunk_size=chunk_size_tri_attn,
            use_kernels=_tmpl_use_kernels,
        )
        if mem_log:
            mem_log(f"  template: PF[{_li}] after tri_att_end")
        _save_subop_gather("after_tri_att_end", pf_input, _li)

        # 5. transition_z
        pf_input = pf_input + layer.transition_z(pf_input)
        if mem_log:
            mem_log(f"  template: PF[{_li}] after transition")
        _save_subop_gather("after_transition", pf_input, _li)

        # Log PF layer output (debug only — gather is collective)
        if _tmpl_debug:
            _pf_full = gather(pf_input.contiguous(), dim=1)
            if dap_rank == 0:
                _pff = _pf_full[:, :N, :N, :].float()
                print(f"      [R{_ri}/TMPL] v_after_pf{_li}: mean={_pff.mean():.6f} std={_pff.std():.4f} absmax={_pff.abs().max():.4f} (gathered)")
                if recycle_idx == 0:
                    _gran_ckpts[f"tmpl/v_after_pf{_li}"] = _pf_full[:, :N, :N, :].cpu().to(torch.bfloat16)
            del _pf_full
        if mem_log:
            mem_log(f"  template: PF layer[{_li}] done")

    # v = v + pf_output (residual)
    v = v + pf_input
    del pf_input

    if mem_log:
        mem_log("  template: PF residual added")

    # Log v_residual (debug only)
    if _tmpl_debug:
        _vr_full = gather(v.contiguous(), dim=1)
        if dap_rank == 0:
            _vrf = _vr_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] v_residual: mean={_vrf.mean():.6f} std={_vrf.std():.4f} absmax={_vrf.abs().max():.4f} (gathered)")
            if recycle_idx == 0:
                _gran_ckpts["tmpl/v_residual"] = _vr_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _vr_full

    # Post-PF: norm, reshape, aggregate over templates
    v = tmpl_module.v_norm(v)
    v = v.view(B, T, *v.shape[1:])  # [B, T, N/dap, N, template_dim]

    # Granular: v_norm (debug only)
    if _tmpl_debug:
        _vn_full = gather(v.reshape(B * T, *v.shape[2:]).contiguous(), dim=1)
        if dap_rank == 0 and recycle_idx == 0:
            _gran_ckpts["tmpl/v_norm"] = _vn_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _vn_full

    # Aggregate templates
    tmask = template_mask[:, :, None, None, None]
    ntemplates = num_templates[:, None, None, None]
    u = (v * tmask).sum(dim=1) / ntemplates.to(v)  # [B, N/dap, N, template_dim]
    del v

    if mem_log:
        mem_log("  template: aggregated over templates")

    # Log u_agg (debug only)
    if _tmpl_debug:
        _u_full = gather(u.contiguous(), dim=1)
        if dap_rank == 0:
            _uf = _u_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] u_agg: mean={_uf.mean():.6f} std={_uf.std():.4f} absmax={_uf.abs().max():.4f} (gathered)")
            if recycle_idx == 0:
                _gran_ckpts["tmpl/u_agg"] = _u_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _u_full

    # Project back to z dim
    u = tmpl_module.u_proj(tmpl_module.relu(u))  # [B, N/dap, N, token_z]

    if mem_log:
        mem_log("  template: u_proj computed")

    # Log u_proj (debug only)
    if _tmpl_debug:
        _u_full = gather(u.contiguous(), dim=1)
        if dap_rank == 0:
            _uf = _u_full[:, :N, :N, :].float()
            print(f"      [R{_ri}/TMPL] u_proj: mean={_uf.mean():.6f} std={_uf.std():.4f} absmax={_uf.abs().max():.4f} (gathered)")
            if recycle_idx == 0:
                _gran_ckpts["tmpl/u_proj"] = _u_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _u_full

    # Pad dim 2 to match z_scattered's padded N if needed
    if u.shape[2] < z_scattered.shape[2]:
        pad_n = z_scattered.shape[2] - u.shape[2]
        u = torch.nn.functional.pad(u, (0, 0, 0, pad_n))

    # Add to z_scattered
    z_scattered = z_scattered + u
    del u

    if mem_log:
        mem_log("  template: added to z_scattered")

    # Granular: z_final (debug only)
    if _tmpl_debug:
        _z_full = gather(z_scattered.contiguous(), dim=1)
        if dap_rank == 0 and recycle_idx == 0:
            _gran_ckpts["tmpl/z_final"] = _z_full[:, :N, :N, :].cpu().to(torch.bfloat16)
        del _z_full

    # Save granular checkpoints to disk (debug only)
    if _tmpl_debug and recycle_idx == 0 and dap_rank == 0:
        _out_dir = os.environ.get('BOLTZ_OUT_DIR', '')
        if _out_dir and _gran_ckpts:
            _gran_path = os.path.join(_out_dir, 'granular_ckpts.pt')
            torch.save(_gran_ckpts, _gran_path)
            _gran_size = os.path.getsize(_gran_path) / (1024**2)
            print(f"      [GRAN] Saved {len(_gran_ckpts)} granular checkpoints to {_gran_path} ({_gran_size:.0f} MB)")

    return z_scattered


def _run_msa_dap(msa_module, z_scattered, s_inputs, feats, full_pair_mask, use_kernels, mem_log=None, _msa_diag=False):
    """Run MSA module with DAP-scattered z.

    z_scattered: [B, N/dap, N, D]
    """
    # Set chunk sizes (same logic as original, with OOM-safe overrides)
    N = z_scattered.shape[2]  # full N
    if not msa_module.training:
        from boltz.data import const
        if N > const.chunk_size_threshold:
            chunk_heads_pwa = True
            chunk_size_transition_z = 64
            chunk_size_transition_msa = 32
            chunk_size_outer_product = 4
            # For very long sequences, reduce tri_att chunk to prevent
            # Q@K^T score matrix OOM: [chunk, H, N, N] × 4 bytes
            chunk_size_tri_attn = 16 if N > 2000 else 128
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = 512
    else:
        chunk_heads_pwa = False
        chunk_size_transition_z = None
        chunk_size_transition_msa = None
        chunk_size_outer_product = None
        chunk_size_tri_attn = None

    # Prepare MSA features
    from boltz.data import const
    msa = feats["msa"]
    msa = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
    has_deletion = feats["has_deletion"].unsqueeze(-1)
    deletion_value = feats["deletion_value"].unsqueeze(-1)
    is_paired = feats["msa_paired"].unsqueeze(-1)
    msa_mask = feats["msa_mask"]
    token_mask = feats["token_pad_mask"].float()
    token_mask_2d = token_mask[:, :, None] * token_mask[:, None, :]

    if msa_module.use_paired_feature:
        m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
    else:
        m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

    if msa_module.subsample_msa:
        msa_indices = torch.randperm(msa.shape[1])[:msa_module.num_subsampled_msa]
        m = m[:, msa_indices]
        msa_mask = msa_mask[:, msa_indices]

    m = msa_module.msa_proj(m)
    m = m + msa_module.s_proj(s_inputs).unsqueeze(1)

    # Run MSA blocks with DAP layers
    for i in range(msa_module.msa_blocks):
        # Enable fine-grained diagnostics for layers 0,1 on first trunk cycle
        layer = msa_module.layers[i]
        layer._diag_enabled = _msa_diag and (i <= 1)

        # Enable granular checkpoints on block 0 when diagnostics are active
        if _msa_diag and i == 0:
            layer._save_gran_ckpts = True
            layer._gran_ckpt_data = {}

        z_scattered, m = layer(
            z_scattered, m, token_mask_2d, msa_mask,
            chunk_heads_pwa,
            chunk_size_transition_z,
            chunk_size_transition_msa,
            chunk_size_outer_product,
            chunk_size_tri_attn,
            use_kernels,
        )

        # Collect granular checkpoints — dap_msa.py already gathered & saved to CPU
        # NOTE: _gran_ckpt_data is only populated on rank 0, so we must reset
        # _save_gran_ckpts unconditionally to prevent rank 1 from doing extra
        # gathers in subsequent recycles.
        if _msa_diag and i == 0:
            _dap_rank = get_dap_rank()
            if _dap_rank == 0 and hasattr(layer, '_gran_ckpt_data') and layer._gran_ckpt_data:
                _gran_ckpts = getattr(_run_msa_dap, '_gran_ckpts', {})
                for key, val in layer._gran_ckpt_data.items():
                    _gran_ckpts[f"msa/blk0/{key}"] = val  # already CPU bf16
                _run_msa_dap._gran_ckpts = _gran_ckpts
            layer._save_gran_ckpts = False
            layer._gran_ckpt_data = {}

        if mem_log:
            mem_log(f"  msa_module.layer[{i}]")

    return z_scattered


def _run_pairformer_dap(pf_module, s, z_scattered, mask, full_pair_mask, use_kernels, mem_log=None):
    """Run pairformer module with DAP-scattered z.

    z_scattered: [B, N/dap, N, D]
    s: [B, N, 384] — replicated
    """
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    pair_mask_scattered = scatter(full_pair_mask, dim=1) if dap_size > 1 else full_pair_mask

    # Set chunk sizes for large N (same logic as MSA module)
    N = z_scattered.shape[2]
    if not pf_module.training:
        from boltz.data import const
        if N > const.chunk_size_threshold:
            # Reduce chunk size for very long sequences to prevent OOM
            # in Q@K^T attention score computation
            chunk_size_tri_attn = 16 if N > 2000 else 128
        else:
            chunk_size_tri_attn = 512
    else:
        chunk_size_tri_attn = None

    # Per-layer checkpoint saving (controlled by env var)
    _layer_dir = os.environ.get("BOLTZ_SAVE_LAYER_CKPT", "")
    if _layer_dir and dap_rank == 0:
        import pathlib
        pathlib.Path(_layer_dir).mkdir(parents=True, exist_ok=True)

    num_layers = len(pf_module.layers)
    for i, layer in enumerate(pf_module.layers):
        s, z_scattered = layer(
            s, z_scattered, mask, pair_mask_scattered,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=use_kernels,
            layer_idx=i,
        )

        # Save z every 4th layer + last layer for divergence analysis
        # gather() is NCCL collective — ALL ranks must call it
        if _layer_dir and (i % 4 == 0 or i == num_layers - 1):
            z_full = gather(z_scattered.contiguous(), dim=1, original_size=N)
            if dap_rank == 0:
                torch.save(z_full.detach().cpu(), f"{_layer_dir}/layer_{i:03d}.pt")
            del z_full

        # Log every 8th layer to avoid spam (48 layers total)
        if mem_log and (i % 8 == 0 or i == num_layers - 1):
            mem_log(f"  pairformer.layer[{i}]")

    # Clear env var so only R0 (first recycling step) saves per-layer checkpoints
    if _layer_dir:
        os.environ.pop("BOLTZ_SAVE_LAYER_CKPT", None)

    # Save z at end of each recycling step (BOLTZ_SAVE_RECYCLE_CKPT)
    _recycle_dir = os.environ.get("BOLTZ_SAVE_RECYCLE_CKPT", "")
    if _recycle_dir:
        _rc = int(os.environ.get("_BOLTZ_RECYCLE_CTR", "0"))
        # gather is collective — all ranks must call
        z_full = gather(z_scattered.contiguous(), dim=1, original_size=N)
        if dap_rank == 0:
            import pathlib
            pathlib.Path(_recycle_dir).mkdir(parents=True, exist_ok=True)
            torch.save(z_full.detach().cpu(), f"{_recycle_dir}/recycle_{_rc:02d}.pt")
        del z_full
        os.environ["_BOLTZ_RECYCLE_CTR"] = str(_rc + 1)

    return s, z_scattered
