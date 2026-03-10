#!/usr/bin/env python
"""
Baseline sub-op checkpoint script.

Runs the original Boltz 2 template pairformer (single GPU, no DAP)
and saves z after each sub-op (tri_mul_out, tri_mul_in, tri_att_start,
tri_att_end, transition) for the first PF layer of recycle 0.

Usage:
    uv run --project /path/to/boltz python run_baseline_subops.py \
        /path/to/input.yaml --out_dir /path/to/output --seed 42
"""

import gc
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from functools import partial

import click
import torch

from boltz.main import (
    Boltz2DiffusionParams,
    PairformerArgsV2,
    MSAModuleArgs,
    BoltzSteeringParams,
    BoltzProcessedInput,
    process_inputs,
    filter_inputs_structure,
)
from boltz.model.models.boltz2 import Boltz2
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data import const
from torch.nn.functional import one_hot


def run_pf_layer_with_checkpoints(layer, z, mask, chunk_size_tri_attn, use_kernels):
    """Run a PairformerNoSeqLayer and save z after each sub-op."""
    checkpoints = {}
    checkpoints["input"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] input: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    # 1. TriMulOut
    dropout_mask = 1.0  # eval mode, dropout = 0
    z = z + dropout_mask * layer.tri_mul_out(z, mask=mask, use_kernels=use_kernels)
    checkpoints["after_tri_mul_out"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] after_tri_mul_out: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    # 2. TriMulIn
    z = z + dropout_mask * layer.tri_mul_in(z, mask=mask, use_kernels=use_kernels)
    checkpoints["after_tri_mul_in"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] after_tri_mul_in: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    # 3. TriAttStart
    z = z + dropout_mask * layer.tri_att_start(
        z, mask=mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
    )
    checkpoints["after_tri_att_start"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] after_tri_att_start: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    # 4. TriAttEnd
    z = z + dropout_mask * layer.tri_att_end(
        z, mask=mask, chunk_size=chunk_size_tri_attn, use_kernels=use_kernels,
    )
    checkpoints["after_tri_att_end"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] after_tri_att_end: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    # 5. Transition
    z = z + layer.transition_z(z, chunk_size=128)
    checkpoints["after_transition"] = z.cpu().to(torch.bfloat16)
    zf = z.float()
    print(f"        [SUBOP-CKP] after_transition: mean={zf.mean():.6f} std={zf.std():.4f} absmax={zf.abs().max():.4f}")

    return z, checkpoints


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--cache", type=click.Path(), default="~/.boltz")
@click.option("--use_msa_server", is_flag=True)
@click.option("--seed", type=int, default=None)
def main(data, out_dir, cache, use_msa_server, seed):
    data = Path(data)
    out_dir = Path(out_dir)
    cache = Path(cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)
        print(f"  [SEED] Set seed={seed}")

    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    print("[1/4] Processing inputs...")
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=[data], out_dir=out_dir, ccd_path=ccd_path, mol_dir=mol_dir,
        use_msa_server=use_msa_server, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", boltz2=True, preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered = filter_inputs_structure(manifest=manifest, outdir=out_dir)
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered, targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa", constraints_dir=None,
        template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
    )

    print("[2/4] Loading model...")
    checkpoint = cache / "boltz2_conf.ckpt"
    model = Boltz2.load_from_checkpoint(
        checkpoint, strict=True,
        predict_args={"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1,
                       "max_parallel_samples": 1, "write_confidence_summary": True,
                       "write_full_pae": False, "write_full_pde": False},
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    model.to(device)
    print(f"  Model loaded ({torch.cuda.memory_allocated(device)/1024**2:.0f} MB)")

    print("[3/4] Running trunk up to template PF layer 0...")
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest, target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir, mol_dir=mol_dir, num_workers=2,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    data_module.setup("predict")

    for batch in data_module.predict_dataloader():
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Input embedding
        s_inputs = model.input_embedder(batch)
        s_init = model.s_init(s_inputs)
        mask = batch["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        N = mask.shape[1]

        z = model.z_init_1(s_inputs)[:, :, None] + model.z_init_2(s_inputs)[:, None, :]
        z = z + model.rel_pos(batch) + model.token_bonds(batch["token_bonds"].float())
        if model.bond_type_feature:
            z = z + model.token_bonds_type(batch["type_bonds"].long())
        z = z + model.contact_conditioning(batch)

        # Recycle init
        s = torch.zeros_like(s_init)
        z_recycle = torch.zeros_like(z)

        # Recycle 0
        s = s_init + model.s_recycle(model.s_norm(s))
        z = z + model.z_recycle(model.z_norm(z_recycle))
        del z_recycle

        # Template features (same as _run_template_dap but on full z)
        tmpl = model.template_module
        if hasattr(tmpl, '_orig_mod'):
            tmpl = tmpl._orig_mod

        res_type = batch["template_restype"]
        frame_rot = batch["template_frame_rot"]
        frame_t = batch["template_frame_t"]
        frame_mask = batch["template_mask_frame"]
        cb_coords = batch["template_cb"]
        ca_coords = batch["template_ca"]
        cb_mask = batch["template_mask_cb"]
        visibility_ids = batch["visibility_ids"]
        template_mask = batch["template_mask"].any(dim=2).float()
        num_templates = template_mask.sum(dim=1).clamp(min=1)
        B, T = res_type.shape[:2]

        b_cb_mask = (cb_mask[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
        b_frame_mask = (frame_mask[:, :, :, None] * frame_mask[:, :, None, :])[..., None]
        tmlp_pair_mask = (visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]).float()

        with torch.autocast(device_type="cuda", enabled=False):
            cb_dists = torch.cdist(cb_coords, cb_coords)
            boundaries = torch.linspace(tmpl.min_dist, tmpl.max_dist, tmpl.num_bins - 1).to(cb_dists.device)
            distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
            distogram = one_hot(distogram, num_classes=tmpl.num_bins)
            frame_rot_t = frame_rot.unsqueeze(2).transpose(-1, -2)
            frame_t_exp = frame_t.unsqueeze(2).unsqueeze(-1)
            ca_exp = ca_coords.unsqueeze(3).unsqueeze(-1)
            vector = torch.matmul(frame_rot_t, (ca_exp - frame_t_exp))
            norm = torch.norm(vector, dim=-1, keepdim=True)
            unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector)).squeeze(-1)
            a_tij = torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
            a_tij = a_tij * tmlp_pair_mask.unsqueeze(-1)
            res_type_i = res_type[:, :, :, None].expand(-1, -1, -1, N, -1)
            res_type_j = res_type[:, :, None, :].expand(-1, -1, N, -1, -1)
            a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
            a_tij = tmpl.a_proj(a_tij)

        # Save upstream template checkpoints for comparison
        tmpl_ckpts = {}

        # Save a_tij (full N) — matches DAP's a_tij_full
        af = a_tij.float()
        print(f"      [TMPL] a_tij: mean={af.mean():.6f} std={af.std():.4f} absmax={af.abs().max():.4f}")
        tmpl_ckpts["a_tij_full"] = a_tij.cpu().to(torch.bfloat16)

        # Compute z_proj and a_tij separately (to checkpoint individually)
        z_proj_out = tmpl.z_proj(tmpl.z_norm(z[:, None]))
        zpf = z_proj_out.float()
        print(f"      [TMPL] z_proj_out: mean={zpf.mean():.6f} std={zpf.std():.4f} absmax={zpf.abs().max():.4f}")
        # Reshape to [B*1, N, N, D] for comparable shape
        tmpl_ckpts["z_proj_out"] = z_proj_out.reshape(-1, *z_proj_out.shape[2:]).cpu().to(torch.bfloat16)

        v = z_proj_out + a_tij
        del a_tij, z_proj_out
        vf = v.float()
        print(f"      [TMPL] v_input: mean={vf.mean():.6f} std={vf.std():.4f} absmax={vf.abs().max():.4f}")

        pair_mask_tmpl = pair_mask[:, None].expand(-1, T, -1, -1).reshape(B * T, N, N)
        v = v.view(B * T, N, N, -1)

        # Save v_input (after reshape to B*T)
        tmpl_ckpts["v_input"] = v.cpu().to(torch.bfloat16)

        # Save upstream checkpoints
        tmpl_path = out_dir / "template_upstream_ckpts.pt"
        torch.save(tmpl_ckpts, str(tmpl_path))
        print(f"      Saved {len(tmpl_ckpts)} upstream checkpoints to {tmpl_path}")
        del tmpl_ckpts

        # Set chunk size
        if N > const.chunk_size_threshold:
            chunk_size_tri_attn = 128
        else:
            chunk_size_tri_attn = 512

        # Run FIRST pairformer layer with per-sub-op checkpoints
        print("\n  Running template PF layer 0 with sub-op checkpoints...")
        layer0 = tmpl.pairformer.layers[0]
        v, subop_data = run_pf_layer_with_checkpoints(
            layer0, v, pair_mask_tmpl, chunk_size_tri_attn, use_kernels=False
        )

        print(f"\n[4/4] Saving sub-op checkpoints...")
        subop_path = out_dir / "subop_checkpoints.pt"
        torch.save(subop_data, str(subop_path))
        print(f"  Saved {len(subop_data)} sub-op checkpoints to {subop_path}")
        break


if __name__ == "__main__":
    main()
