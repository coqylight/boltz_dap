#!/usr/bin/env python
"""
Single-GPU Boltz2 baseline with z/s tensor saving.

Runs the ORIGINAL Boltz2 model (no DAP, no wrappers) on a single GPU,
with deterministic seeding and z/s tensor saving for comparison with DAP.

Usage:
    python run_boltz_baseline.py /path/to/input.yaml \
        --out_dir /path/to/output --seed 42 --use_msa_server
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


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--cache", type=click.Path(), default="~/.boltz")
@click.option("--recycling_steps", type=int, default=3)
@click.option("--sampling_steps", type=int, default=200)
@click.option("--diffusion_samples", type=int, default=1)
@click.option("--use_msa_server", is_flag=True)
@click.option("--seed", type=int, default=None, help="Random seed for deterministic runs")
def main(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    use_msa_server: bool = False,
    seed: int = None,
):
    """Run original Boltz2 on single GPU with z/s saving."""

    device = torch.device('cuda:0')

    # Deterministic seeding
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)
        print(f"  [SEED] Set torch/numpy seed={seed}")

    # Paths
    data = Path(data)
    out_dir = Path(out_dir)
    cache = Path(cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BOLTZ 2 BASELINE INFERENCE (single GPU, no DAP)")
    print(f"{'='*70}")
    print(f"Input: {data}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    # Suppress warnings
    warnings.filterwarnings("ignore", ".*Tensor Cores.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    # Import Boltz modules
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
    from boltz.data.write.writer import BoltzWriter

    print("[1/5] Processing input data...")

    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    process_inputs(
        data=[data],
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=out_dir)

    if not filtered_manifest.records:
        print("No predictions needed.")
        return

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
        template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None,
    )

    print(f"  ✓ Processed {len(filtered_manifest.records)} input(s)")

    # Load model
    print("\n[2/5] Loading Boltz2 model...")

    checkpoint = cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs()
    steering_args = BoltzSteeringParams()

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()
    model = model.to(device)
    print(f"  ✓ Model loaded to {device}")

    # Hook to intercept z/s after trunk
    _captured = {}
    _orig_distogram_forward = model.distogram_module.forward
    def _hook_distogram(z):
        _captured['z'] = z.detach().cpu()
        return _orig_distogram_forward(z)
    model.distogram_module.forward = _hook_distogram

    # Also hook s through s_post_norm or pairformer output
    # The pairformer_module.forward returns (s, z), and s is used later
    _orig_pf_forward = model.pairformer_module.forward
    def _hook_pf(*args, **kwargs):
        s, z = _orig_pf_forward(*args, **kwargs)
        _captured['s_pf'] = s.detach().cpu()
        return s, z
    model.pairformer_module.forward = _hook_pf

    # Data module
    print("\n[3/5] Setting up data...")
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=2,
        constraints_dir=processed.constraints_dir,
    )

    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
        write_embeddings=False,
    )

    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    def _move_to_device(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: _move_to_device(v, device) for k, v in x.items()}
        elif isinstance(x, list):
            return [_move_to_device(v, device) for v in x]
        return x

    print("\n[4/5] Running inference...")
    for batch_idx, batch in enumerate(dataloader):
        batch = _move_to_device(batch, device)

        N = batch.get("token_pad_mask", torch.tensor([])).shape[-1]
        print(f"  Batch {batch_idx}: N={N}")

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()

        with torch.no_grad():
            pred_dict = model.predict_step(batch, batch_idx)

        t1 = time.time()
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  Time: {t1-t0:.1f}s, Peak memory: {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")

        if pred_dict.get("exception", False):
            print(f"  ✗ OOM during inference!")
            continue

        # Write CIF
        pred_writer.write_on_batch_end(
            trainer=None, pl_module=None,
            prediction=pred_dict, batch_indices=None,
            batch=batch, batch_idx=batch_idx, dataloader_idx=0,
        )

        # Save z/s tensors
        zs_data = {}
        if pred_dict.get("z") is not None:
            zs_data["z"] = pred_dict["z"].cpu()
        elif "z" in _captured:
            zs_data["z"] = _captured["z"]

        if pred_dict.get("s") is not None:
            zs_data["s"] = pred_dict["s"].cpu()
        elif "s_pf" in _captured:
            zs_data["s"] = _captured["s_pf"]

        if zs_data:
            zs_path = out_dir / "zs_tensors.pt"
            torch.save(zs_data, str(zs_path))
            print(f"  ✓ Saved z/s to {zs_path}")
            for k, v in zs_data.items():
                print(f"    {k}: shape={list(v.shape)}, mean={v.float().mean():.4f}, std={v.float().std():.4f}")

    # Check output
    print(f"\n[5/5] Checking output...")
    cif_files = list((out_dir / "predictions").rglob("*.cif"))
    if cif_files:
        print(f"  ✓ CIF file: {cif_files[0]}")
    else:
        print(f"  ✗ No CIF file found")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
