#!/usr/bin/env python
"""
Boltz 2 with Proper FastFold-style DAP.

Runs Boltz 2 inference with Dynamic Axial Parallelism:
- Pair representation z is SCATTERED across GPUs (no model duplication)
- Triangle multiplication intermediates are halved per GPU
- All-to-all communication for row↔col scatter switching

Usage:
    torchrun --nproc_per_node=2 run_boltz_dap_v2.py \
        /path/to/input.yaml --out_dir /path/to/output

Requirements:
    - 2+ GPUs on the same node (NVLink recommended)
    - boltz environment activated
"""

import gc
import os
import sys
import warnings
import threading
import time
import subprocess as sp
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import torch
import torch.distributed as dist


class GPUMonitor:
    """Monitor GPU memory during inference."""

    def __init__(self, log_file, interval=0.5):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.max_memory = {}
        self.thread = None

    def _get_gpu_memory(self):
        result = sp.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        mem_info = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 5:
                gpu_id = int(parts[0].strip())
                used = int(parts[1].strip())
                total = int(parts[2].strip())
                util = int(parts[3].strip())
                temp = int(parts[4].strip())
                mem_info.append((gpu_id, used, total, util, temp))
        return mem_info

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor(self):
        with open(self.log_file, 'w') as f:
            f.write("timestamp,gpu_id,mem_used_mb,mem_total_mb,util_pct,temp_c\n")
            start_time = time.time()
            while self.running:
                elapsed = time.time() - start_time
                mem_info = self._get_gpu_memory()
                for gpu_id, used, total, util, temp in mem_info:
                    f.write(f"{elapsed:.1f},{gpu_id},{used},{total},{util},{temp}\n")
                    if gpu_id not in self.max_memory or used > self.max_memory[gpu_id]:
                        self.max_memory[gpu_id] = used
                f.flush()
                time.sleep(self.interval)

    def report(self):
        print(f"\n{'='*60}")
        print("GPU PEAK MEMORY USAGE")
        print(f"{'='*60}")
        for gpu_id, max_mem in sorted(self.max_memory.items()):
            print(f"  GPU {gpu_id}: Peak = {max_mem} MB ({max_mem/1024:.1f} GB)")
        print(f"{'='*60}")


@click.command()
@click.argument("data", type=click.Path(exists=True))
@click.option("--out_dir", type=click.Path(), required=True)
@click.option("--cache", type=click.Path(), default="~/.boltz")
@click.option("--recycling_steps", type=int, default=3)
@click.option("--sampling_steps", type=int, default=200)
@click.option("--diffusion_samples", type=int, default=1)
@click.option("--use_msa_server", is_flag=True)
@click.option("--no_kernels", is_flag=True, help="Disable cuequivariance CUDA kernels (use PyTorch-native ops)")
@click.option("--use_flex_attention", is_flag=True, help="Use FlexAttention for triangle attention (memory/throughput)")
@click.option("--use_flex_attention_chunked", is_flag=True, help="Use chunked FlexAttention for DAP (experimental; avoids 112GB OOM)")
@click.option("--use_potentials", is_flag=True, help="Enable FK steering + physical guidance potentials")
@click.option("--write_full_pae/--no_write_full_pae", default=True, help="Dump full PAE matrix to npz (default: on)")
@click.option("--write_full_pde/--no_write_full_pde", default=True, help="Dump full PDE matrix to npz (default: on)")
@click.option("--seed", type=int, default=None, help="Random seed for deterministic runs")
@click.option("--skip_processing", is_flag=True, help="Reuse existing out_dir/processed without running process_inputs")
def main(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    use_msa_server: bool = False,
    no_kernels: bool = False,
    use_flex_attention: bool = False,
    use_flex_attention_chunked: bool = False,
    use_potentials: bool = False,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    seed: int = None,
    skip_processing: bool = False,
):
    """Run Boltz 2 with proper FastFold-style DAP (no model duplication)."""

    # Initialize DAP
    from boltz_distributed import init_dap, get_dap_size, get_dap_rank

    init_dap()

    dap_rank = get_dap_rank()
    dap_size = get_dap_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')

    # Deterministic seeding for controlled A/B testing
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed)
        # Force deterministic cuBLAS GEMMs (same results across GPUs with same inputs)
        # NOTE: do NOT use torch.use_deterministic_algorithms(True) — it disables
        # Flash Attention and causes OOM on long sequences.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if dap_rank == 0:
            print(f"  [SEED] Set torch/numpy seed={seed} + deterministic cuBLAS/cuDNN")

    # Paths
    data = Path(data)
    out_dir = Path(out_dir)
    cache = Path(cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    import os as _os
    _os.environ['BOLTZ_OUT_DIR'] = str(out_dir)
    log_file = out_dir / "gpu_memory.log"

    def rank_print(msg):
        if dap_rank == 0:
            print(msg)

    rank_print(f"\n{'='*70}")
    rank_print(f"BOLTZ 2 DAP v2 INFERENCE ({dap_size} GPUs)")
    rank_print(f"{'='*70}")
    rank_print(f"Input: {data}")
    rank_print(f"Output: {out_dir}")
    rank_print(f"No model duplication — activations sharded across GPUs")
    rank_print(f"{'='*70}\n")

    # Start GPU monitoring (rank 0 only)
    monitor = None
    if dap_rank == 0:
        monitor = GPUMonitor(str(log_file))
        monitor.start()

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

    rank_print("[1/6] Processing input data...")

    # Process inputs (only on rank 0)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    processed_manifest = out_dir / "processed" / "manifest.json"
    if dap_rank == 0:
        if skip_processing:
            if not processed_manifest.exists():
                raise FileNotFoundError(
                    f"--skip_processing was set, but processed manifest was not found: {processed_manifest}"
                )
            rank_print("  ✓ Reusing existing processed inputs (skip_processing=True)")
        else:
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

    dist.barrier()

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=out_dir)

    if not filtered_manifest.records:
        rank_print("No predictions needed.")
        dist.destroy_process_group()
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

    rank_print(f"  ✓ Processed {len(filtered_manifest.records)} input(s)")

    # Load model to CPU first (ALL ranks load from checkpoint to CPU)
    rank_print("\n[2/6] Loading Boltz2 model...")

    checkpoint = cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs()
    steering_args = BoltzSteeringParams()
    if use_potentials:
        steering_args.fk_steering = True
        steering_args.physical_guidance_update = True
        rank_print(f"  ✓ Potentials enabled: FK steering + physical guidance")

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=not no_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model.eval()

    rank_print(f"  ✓ Model loaded to CPU (all ranks)")

    # ── Optional: FlexAttention for triangle attention (before DAP injection) ──
    if use_flex_attention_chunked:
        try:
            from flex_attention_patch_chunked import patch_triangle_attention
            n_patched = patch_triangle_attention(model)
            rank_print(f"  ✓ FlexAttention (chunked) patched onto {n_patched} TriangleAttention layers")
        except Exception as e:
            import traceback
            rank_print(f"  ⚠ FlexAttention chunked patch skipped: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
    elif use_flex_attention:
        try:
            from flex_attention_patch import patch_triangle_attention
            n_patched = patch_triangle_attention(model)
            rank_print(f"  ✓ FlexAttention patched onto {n_patched} TriangleAttention layers")
        except Exception as e:
            import traceback
            rank_print(f"  ⚠ FlexAttention patch skipped: {e}")
            rank_print("  FlexAttention patch traceback (to locate 'duplicate template name'):")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()

    # ── Selective GPU placement ──────────────────────────────────────────
    # GPU 0: gets the FULL model (trunk + post-trunk)
    # GPU 1+: gets ONLY trunk modules (input_embedder, msa, pairformer,
    #         template, recycling). Post-trunk stays on CPU and is never used.
    rank_print(f"\n[3/6] Placing modules on GPUs (selective, no duplication)...")

    # Trunk modules needed on ALL GPUs (for DAP):
    trunk_module_names = [
        "input_embedder", "s_init", "z_init_1", "z_init_2",
        "rel_pos", "token_bonds", "contact_conditioning",
        "s_recycle", "z_recycle", "s_norm", "z_norm",
        "msa_module", "pairformer_module", "template_module",
        "distogram_module",
    ]
    # Also include bond_type_feature related modules if present
    if model.bond_type_feature:
        trunk_module_names.append("token_bonds_type")

    if dap_rank == 0:
        # GPU 0: move EVERYTHING to GPU
        model = model.to(device)
        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  GPU 0: Full model loaded ({mem_after:.0f} MB)")
    else:
        # GPU 1+: move ONLY trunk modules to GPU, keep rest on CPU
        for name in trunk_module_names:
            if hasattr(model, name):
                getattr(model, name).to(device)

        # Also load confidence pairformer stack for DAP participation
        if hasattr(model, 'confidence_module') and model.confidence_prediction:
            model.confidence_module.pairformer_stack.to(device)
            # Load pre-PF weights too (z_norm, rel_pos, s_to_z, etc. — ~10 MB)
            from dap_confidence import load_confidence_pre_pf_weights
            load_confidence_pre_pf_weights(model, device)
            print(f"  GPU {dap_rank}: Confidence PF + pre-PF weights loaded for DAP")

        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  GPU {dap_rank}: Trunk + confidence PF loaded ({mem_after:.0f} MB)")
        print(f"  GPU {dap_rank}: Other post-trunk modules (structure, diffusion) stay on CPU")

    # Inject DAP wrappers
    rank_print(f"\n[4/6] Injecting DAP wrappers...")

    from dap_trunk import inject_dap_into_model
    model = inject_dap_into_model(model)

    rank_print(f"  ✓ DAP injection complete")

    # Create data module
    rank_print(f"\n[5/6] Running inference with DAP...")

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

    # Create prediction writer (only rank 0 writes)
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format="mmcif",
        boltz2=True,
        write_embeddings=False,
    )

    # Run inference manually (no Trainer — we control the DAP ourselves)
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    def _move_to_device(x, device):
        """Recursively move tensors in nested dicts/lists to device."""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: _move_to_device(v, device) for k, v in x.items()}
        elif isinstance(x, list):
            return [_move_to_device(v, device) for v in x]
        return x

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device (recursively handles nested dicts)
        batch = _move_to_device(batch, device)

        rank_print(f"  Running batch {batch_idx}...")
        N = batch.get("token_pad_mask", torch.tensor([])).shape[-1] if "token_pad_mask" in batch else 0
        rank_print(f"    Sequence length: {N}")
        mem_before = torch.cuda.memory_allocated(device) / 1024**2
        rank_print(f"    Memory before forward: {mem_before:.0f} MB")

        torch.cuda.reset_peak_memory_stats(device)

        if dap_rank == 0:
            # Rank 0: full predict_step (calls DAP forward → structure → confidence)
            with torch.no_grad():
                pred_dict = model.predict_step(batch, batch_idx)
        else:
            # Non-primary: just call forward to participate in DAP trunk comms
            # The DAP forward returns early after the trunk on non-primary ranks
            with torch.no_grad():
                _ = model(
                    batch,
                    recycling_steps=recycling_steps,
                    num_sampling_steps=sampling_steps,
                    diffusion_samples=diffusion_samples,
                    max_parallel_samples=1,
                    run_confidence_sequentially=True,
                )
            pred_dict = None

        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"    GPU {dap_rank}: Memory after forward: {mem_after:.0f} MB")
        print(f"    GPU {dap_rank}: Peak memory: {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")

        # Barrier to sync all GPUs before next batch
        dist.barrier()

        if pred_dict is not None and pred_dict.get("exception", False):
            rank_print(f"    ✗ OOM during inference!")
            continue

        # Only rank 0 writes output
        if dap_rank == 0 and pred_dict is not None:
            pred_writer.write_on_batch_end(
                trainer=None,
                pl_module=None,
                prediction=pred_dict,
                batch_indices=None,
                batch=batch,
                batch_idx=batch_idx,
                dataloader_idx=0,
            )

            # Save z and s tensors for comparison
            if pred_dict.get("s") is not None and pred_dict.get("z") is not None:
                zs_path = out_dir / "zs_tensors.pt"
                torch.save({
                    "s": pred_dict["s"].cpu(),
                    "z": pred_dict["z"].cpu(),
                }, str(zs_path))
                rank_print(f"  ✓ Saved z/s tensors to {zs_path}")

    # Stop monitoring
    if monitor:
        monitor.stop()
        monitor.report()

    # Check output
    rank_print(f"\n[6/6] Checking output...")

    if dap_rank == 0:
        cif_files = list((out_dir / "predictions").rglob("*.cif"))
        if cif_files:
            rank_print(f"  ✓ CIF file: {cif_files[0]}")
        else:
            rank_print(f"  ✗ No CIF file found")

    rank_print(f"\n{'='*70}")
    rank_print("COMPLETE")
    rank_print(f"{'='*70}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
