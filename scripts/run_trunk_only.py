#!/usr/bin/env python
"""
Trunk-only z/s extraction for single-GPU baseline comparison.

Saves FULL z/s tensors at every checkpoint for exact element-wise comparison with DAP.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run --project /path/to/boltz python run_trunk_only.py \
        /path/to/input.yaml --out_dir /path/to/output --seed 42 --use_msa_server
"""

import gc
import os
import sys
import warnings
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import torch


def _log_stats(label, z, s):
    """Print z/s stats inline."""
    zf, sf = z.float(), s.float()
    print(f"    [{label}]  z: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}"
          f"  |  s: mean={sf.mean():.6f} std={sf.std():.4f} absmax={sf.abs().max():.4f}")


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--cache", type=click.Path(), default="~/.boltz")
@click.option("--recycling_steps", type=int, default=3)
@click.option("--use_msa_server", is_flag=True)
@click.option("--seed", type=int, default=None)
def main(data, out_dir, cache, recycling_steps, use_msa_server, seed):
    """Extract z/s from Boltz2 trunk with full tensor checkpoints."""

    device = torch.device('cuda:0')

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)
        # Force deterministic cuBLAS GEMMs (same results across runs/GPUs)
        # NOTE: do NOT use torch.use_deterministic_algorithms(True) — it disables
        # Flash Attention and causes OOM on long sequences.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"  [SEED] Set torch/numpy seed={seed} + deterministic cuBLAS/cuDNN")

    data = Path(data)
    out_dir = Path(out_dir)
    cache = Path(cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BOLTZ2 TRUNK-ONLY — FULL TENSOR CHECKPOINT LOGGING")
    print(f"{'='*70}\n")

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    from boltz.main import (
        Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs,
        BoltzSteeringParams, BoltzProcessedInput, process_inputs,
        filter_inputs_structure,
    )
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data import const

    # Process inputs
    print("[1/4] Processing inputs...")
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    process_inputs(
        data=[data], out_dir=out_dir, ccd_path=ccd_path, mol_dir=mol_dir,
        use_msa_server=use_msa_server, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", boltz2=True,
        preprocessing_threads=1, max_msa_seqs=8192,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=out_dir)
    if not filtered_manifest.records:
        print("No records."); return

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
        template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
    )

    # Load model
    print("[2/4] Loading model...")
    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt", strict=True,
        predict_args={"recycling_steps": recycling_steps, "sampling_steps": 200,
                      "diffusion_samples": 1, "max_parallel_samples": 1,
                      "write_confidence_summary": True, "write_full_pae": False,
                      "write_full_pde": False},
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=True,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()

    # Only load trunk modules to GPU
    trunk_modules = [
        "input_embedder", "s_init", "z_init_1", "z_init_2",
        "rel_pos", "token_bonds", "contact_conditioning",
        "s_recycle", "z_recycle", "s_norm", "z_norm",
        "msa_module", "pairformer_module", "template_module",
        "distogram_module",
    ]
    if model.bond_type_feature:
        trunk_modules.append("token_bonds_type")

    for name in trunk_modules:
        if hasattr(model, name):
            getattr(model, name).to(device)

    print(f"  ✓ Trunk modules loaded to {device}")

    # Setup data
    print("[3/4] Loading data...")
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=2,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    def _move(x, dev):
        if isinstance(x, torch.Tensor): return x.to(dev)
        elif isinstance(x, dict): return {k: _move(v, dev) for k, v in x.items()}
        elif isinstance(x, list): return [_move(v, dev) for v in x]
        return x

    # Run trunk only
    print("[4/4] Running trunk with full tensor checkpoints...")
    checkpoints = {}  # label -> {"z": tensor, "s": tensor}

    for batch_idx, batch in enumerate(dataloader):
        feats = _move(batch, device)
        N = feats["token_pad_mask"].shape[-1]
        print(f"  N={N}")

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()

        with torch.no_grad(), torch.set_grad_enabled(False):
            # === Input embedding ===
            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)

            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            z_init = (
                model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = model.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            z_init = z_init + model.token_bonds(feats["token_bonds"].float())
            if model.bond_type_feature:
                z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
            z_init = z_init + model.contact_conditioning(feats)

            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            _log_stats("init", z_init, s_init)
            checkpoints["init"] = {"z": z_init.cpu().to(torch.bfloat16),
                                   "s": s_init.cpu().to(torch.bfloat16)}

            # === Recycling loop ===
            granular = {}  # granular template checkpoints for R0
            for i in range(recycling_steps + 1):
                s = s_init + model.s_recycle(model.s_norm(s))
                z = z_init + model.z_recycle(model.z_norm(z))

                label = f"R{i}/after_recycle"
                _log_stats(label, z, s)
                checkpoints[label] = {"z": z.cpu().to(torch.bfloat16),
                                      "s": s.cpu().to(torch.bfloat16)}

                # Template — expanded with granular checkpoints
                if model.use_templates:
                    from torch.nn.functional import one_hot

                    def _ts(label, t):
                        """Memory-efficient stats: compute on bf16 directly."""
                        print(f"      [R{i}/TMPL] {label}: mean={t.float().mean():.6f} std={t.float().std():.4f} absmax={t.abs().max().float():.4f}")

                    def _gran(label, t):
                        """Save granular checkpoint (R0 only)."""
                        if i == 0:
                            granular[label] = t.cpu().to(torch.bfloat16)
                            tf = t.float()
                            print(f"      [GRAN] {label}: mean={tf.mean():.6f} std={tf.std():.4f} absmax={tf.abs().max():.4f}")

                    tmpl = model.template_module
                    if hasattr(tmpl, '_orig_mod'):
                        tmpl = tmpl._orig_mod

                    # --- Sub-op 1: Feature computation ---
                    _res_type = feats["template_restype"]
                    _frame_rot = feats["template_frame_rot"]
                    _frame_t = feats["template_frame_t"]
                    _frame_mask = feats["template_mask_frame"]
                    _cb_coords = feats["template_cb"]
                    _ca_coords = feats["template_ca"]
                    _cb_mask = feats["template_mask_cb"]
                    _vis_ids = feats["visibility_ids"]
                    _tmask = feats["template_mask"].any(dim=2).float()
                    _ntmpl = _tmask.sum(dim=1).clamp(min=1)

                    _B, _T = _res_type.shape[:2]
                    _N = z.shape[1]

                    _b_cb = (_cb_mask[:, :, :, None] * _cb_mask[:, :, None, :])[..., None]
                    _b_frm = (_frame_mask[:, :, :, None] * _frame_mask[:, :, None, :])[..., None]
                    _tpm = (_vis_ids[:, :, :, None] == _vis_ids[:, :, None, :]).float()

                    with torch.autocast(device_type="cuda", enabled=False):
                        _dists = torch.cdist(_cb_coords, _cb_coords)
                        _bounds = torch.linspace(tmpl.min_dist, tmpl.max_dist, tmpl.num_bins - 1).to(_dists.device)
                        _dgram = (_dists[..., None] > _bounds).sum(dim=-1).long()
                        _dgram = one_hot(_dgram, num_classes=tmpl.num_bins)

                        _fr = _frame_rot.unsqueeze(2).transpose(-1, -2)
                        _ft = _frame_t.unsqueeze(2).unsqueeze(-1)
                        _ca = _ca_coords.unsqueeze(3).unsqueeze(-1)
                        _vec = torch.matmul(_fr, (_ca - _ft))
                        _nrm = torch.norm(_vec, dim=-1, keepdim=True)
                        _uv = torch.where(_nrm > 0, _vec / _nrm, torch.zeros_like(_vec)).squeeze(-1)

                        _a = torch.cat([_dgram, _b_cb, _uv, _b_frm], dim=-1)
                        _a = _a * _tpm.unsqueeze(-1)
                        _ri = _res_type[:, :, :, None].expand(-1, -1, -1, _N, -1)
                        _rj = _res_type[:, :, None, :].expand(-1, -1, _N, -1, -1)
                        _a = torch.cat([_a, _ri, _rj], dim=-1)
                        _a = tmpl.a_proj(_a)

                    _ts("a_tij", _a)
                    _gran("tmpl/a_tij", _a)

                    # --- Sub-op 2: v = z_proj(z_norm(z)) + a_tij ---
                    _pm = pair_mask[:, None].expand(-1, _T, -1, -1).reshape(_B * _T, *pair_mask.shape[1:])
                    _z_proj = tmpl.z_proj(tmpl.z_norm(z[:, None]))
                    _gran("tmpl/z_proj_out", _z_proj)

                    _v = _z_proj + _a
                    _v = _v.view(_B * _T, *_v.shape[2:])
                    del _a, _z_proj

                    _ts("v_input", _v)
                    _gran("tmpl/v_input", _v)

                    # --- Sub-op 3: Pairformer layers ---
                    from boltz.data import const as _const
                    if not tmpl.pairformer.training:
                        if _N > _const.chunk_size_threshold:
                            _chunk_tri = 128
                        else:
                            _chunk_tri = 512
                    else:
                        _chunk_tri = None

                    _pf_out = _v
                    for _li, _layer in enumerate(tmpl.pairformer.layers):
                        # Expanded per-sub-op with granular checkpoints
                        from boltz.model.layers.pairformer import get_dropout_mask as _gdm

                        # 1. TriMulOut
                        _dropout = _gdm(_layer.dropout, _pf_out, _layer.training)
                        _pf_out = _pf_out + _dropout * _layer.tri_mul_out(
                            _pf_out, mask=_pm, use_kernels=model.use_kernels)
                        _gran(f"tmpl/pf{_li}/after_tri_mul_out", _pf_out)

                        # 2. TriMulIn
                        _dropout = _gdm(_layer.dropout, _pf_out, _layer.training)
                        _pf_out = _pf_out + _dropout * _layer.tri_mul_in(
                            _pf_out, mask=_pm, use_kernels=model.use_kernels)
                        _gran(f"tmpl/pf{_li}/after_tri_mul_in", _pf_out)

                        # 3. TriAttStart
                        _dropout = _gdm(_layer.dropout, _pf_out, _layer.training)
                        _pf_out = _pf_out + _dropout * _layer.tri_att_start(
                            _pf_out, mask=_pm, chunk_size=_chunk_tri, use_kernels=model.use_kernels)
                        _gran(f"tmpl/pf{_li}/after_tri_att_start", _pf_out)

                        # 4. TriAttEnd
                        _dropout = _gdm(_layer.dropout, _pf_out, _layer.training, columnwise=True)
                        _pf_out = _pf_out + _dropout * _layer.tri_att_end(
                            _pf_out, mask=_pm, chunk_size=_chunk_tri, use_kernels=model.use_kernels)
                        _gran(f"tmpl/pf{_li}/after_tri_att_end", _pf_out)

                        # 5. Transition
                        _pf_out = _pf_out + _layer.transition_z(_pf_out)
                        _gran(f"tmpl/pf{_li}/after_transition", _pf_out)

                        _ts(f"v_after_pf{_li}", _pf_out)
                        _gran(f"tmpl/v_after_pf{_li}", _pf_out)

                    # --- Sub-op 4: Residual + norm ---
                    _v = _v + _pf_out
                    del _pf_out
                    _ts("v_residual", _v)
                    _gran("tmpl/v_residual", _v)

                    _v = tmpl.v_norm(_v)
                    _v = _v.view(_B, _T, *_v.shape[1:])
                    _ts("v_norm", _v)
                    _gran("tmpl/v_norm", _v)

                    # --- Sub-op 5: Template aggregation ---
                    _tmask_e = _tmask[:, :, None, None, None]
                    _ntmpl_e = _ntmpl[:, None, None, None]
                    _u = (_v * _tmask_e).sum(dim=1) / _ntmpl_e.to(_v)
                    del _v
                    _ts("u_agg", _u)
                    _gran("tmpl/u_agg", _u)

                    # --- Sub-op 6: Output projection ---
                    _u = tmpl.u_proj(tmpl.relu(_u))
                    _ts("u_proj", _u)
                    _gran("tmpl/u_proj", _u)

                    z = z + _u
                    del _u
                    torch.cuda.empty_cache()
                    _gran("tmpl/z_final", z)

                    label = f"R{i}/after_template"
                    _log_stats(label, z, s)
                    checkpoints[label] = {"z": z.cpu().to(torch.bfloat16),
                                          "s": s.cpu().to(torch.bfloat16)}

                # MSA — use original msa_module call (no inlining = no extra z.clone())
                # Granular checkpoints saved via _save_gran_ckpts in MSALayer.forward()
                torch.cuda.empty_cache()
                msa = model.msa_module
                if hasattr(msa, '_orig_mod'):
                    msa = msa._orig_mod

                # Enable granular checkpoints on block 0 for R0
                if i == 0:
                    msa.layers[0]._save_gran_ckpts = True
                    msa.layers[0]._gran_ckpt_data = {}

                # Original Boltz2 call
                _msa_result = msa(z, s_inputs, feats, use_kernels=model.use_kernels)
                if i == 0:
                    _gran("msa/z_out_residual", _msa_result)
                z = z + _msa_result
                del _msa_result

                # Collect granular checkpoints from block 0
                if i == 0 and hasattr(msa.layers[0], '_gran_ckpt_data') and msa.layers[0]._gran_ckpt_data:
                    for _gk, _gv in msa.layers[0]._gran_ckpt_data.items():
                        _gran(f"msa/blk0/{_gk}", _gv)
                    msa.layers[0]._save_gran_ckpts = False
                    msa.layers[0]._gran_ckpt_data = {}

                label = f"R{i}/after_msa"
                _log_stats(label, z, s)
                checkpoints[label] = {"z": z.cpu().to(torch.bfloat16),
                                      "s": s.cpu().to(torch.bfloat16)}

                # Pairformer — with per-layer checkpoints (every 4th layer)
                pf = model.pairformer_module
                if hasattr(pf, '_orig_mod'):
                    pf = pf._orig_mod

                s, z = pf(s, z, mask=mask, pair_mask=pair_mask,
                          use_kernels=model.use_kernels)

                label = f"R{i}/after_pairformer"
                _log_stats(label, z, s)
                checkpoints[label] = {"z": z.cpu().to(torch.bfloat16),
                                      "s": s.cpu().to(torch.bfloat16)}

                print(f"  Recycle {i} done, peak={torch.cuda.max_memory_allocated(device)//1024**2}MB")

        t1 = time.time()
        peak = torch.cuda.max_memory_allocated(device) // (1024**2)
        print(f"\n  Trunk time: {t1-t0:.1f}s, Peak: {peak}MB ({peak/1024:.1f}GB)")

        # Save
        ckpt_path = out_dir / "trunk_checkpoints.pt"
        torch.save(checkpoints, str(ckpt_path))
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024**2)
        print(f"  ✓ Saved {len(checkpoints)} full-tensor checkpoints to {ckpt_path} ({ckpt_size_mb:.0f} MB)")

        # Save granular template checkpoints
        if granular:
            gran_path = out_dir / "granular_ckpts.pt"
            torch.save(granular, str(gran_path))
            gran_size_mb = os.path.getsize(gran_path) / (1024**2)
            print(f"  ✓ Saved {len(granular)} granular checkpoints to {gran_path} ({gran_size_mb:.0f} MB)")

        # Also save final z/s for backward compat
        zs_path = out_dir / "zs_tensors.pt"
        torch.save({"z": z.cpu(), "s": s.cpu()}, str(zs_path))
        print(f"  ✓ Saved final z/s to {zs_path}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
