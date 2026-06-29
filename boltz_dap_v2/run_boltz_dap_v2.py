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
import hashlib
import json
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


def _sha256_file(path: Path) -> str:
    """Return SHA256 hash for an input file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_processing_fingerprint(data: Path, use_msa_server: bool) -> dict:
    """Build a reproducibility fingerprint for processed inputs."""
    return {
        "schema_version": 1,
        "data_path": str(data.resolve()),
        "data_sha256": _sha256_file(data),
        "boltz2": True,
        "use_msa_server": bool(use_msa_server),
        "msa_server_url": "https://api.colabfold.com",
        "msa_pairing_strategy": "greedy",
        "max_msa_seqs": 8192,
    }


def _diff_fingerprint(old: dict, new: dict) -> str:
    """Return human-readable changed keys between two fingerprints."""
    changed = []
    all_keys = sorted(set(old.keys()) | set(new.keys()))
    for key in all_keys:
        if old.get(key) != new.get(key):
            changed.append(f"{key}: old={old.get(key)!r}, new={new.get(key)!r}")
    return "; ".join(changed)


def _is_oom_error(err: BaseException) -> bool:
    """Return True when an exception indicates CUDA OOM."""
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _build_prediction_dict(model, batch: dict, out: dict) -> dict:
    """Mirror Boltz2.predict_step output without using its soft-OOM path."""
    pred_dict = {"exception": False}
    if "keys_dict_batch" in model.predict_args:
        for key in model.predict_args["keys_dict_batch"]:
            pred_dict[key] = batch[key]

    pred_dict["masks"] = batch["atom_pad_mask"]
    pred_dict["token_masks"] = batch["token_pad_mask"]
    pred_dict["s"] = out["s"]
    pred_dict["z"] = out["z"]

    if "keys_dict_out" in model.predict_args:
        for key in model.predict_args["keys_dict_out"]:
            pred_dict[key] = out[key]

    pred_dict["coords"] = out["sample_atom_coords"]
    if model.confidence_prediction:
        pred_dict["pde"] = out["pde"]
        pred_dict["plddt"] = out["plddt"]
        pred_dict["confidence_score"] = (
            4 * out["complex_plddt"]
            + (
                out["iptm"]
                if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"]))
                else out["ptm"]
            )
        ) / 5
        pred_dict["complex_plddt"] = out["complex_plddt"]
        pred_dict["complex_iplddt"] = out["complex_iplddt"]
        pred_dict["complex_pde"] = out["complex_pde"]
        pred_dict["complex_ipde"] = out["complex_ipde"]
        if model.alpha_pae > 0:
            pred_dict["pae"] = out["pae"]
            pred_dict["ptm"] = out["ptm"]
            pred_dict["iptm"] = out["iptm"]
            pred_dict["ligand_iptm"] = out["ligand_iptm"]
            pred_dict["protein_iptm"] = out["protein_iptm"]
            pred_dict["pair_chains_iptm"] = out["pair_chains_iptm"]
    if model.affinity_prediction:
        pred_dict["affinity_pred_value"] = out["affinity_pred_value"]
        pred_dict["affinity_probability_binary"] = out["affinity_probability_binary"]
        if model.affinity_ensemble:
            pred_dict["affinity_pred_value1"] = out["affinity_pred_value1"]
            pred_dict["affinity_probability_binary1"] = out["affinity_probability_binary1"]
            pred_dict["affinity_pred_value2"] = out["affinity_pred_value2"]
            pred_dict["affinity_probability_binary2"] = out["affinity_probability_binary2"]
    return pred_dict


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
@click.option(
    "--run-profile",
    type=click.Choice(["prod", "parity", "debug"], case_sensitive=False),
    default="prod",
    show_default=True,
    help="Preset for runtime behavior: prod(min overhead), parity(deterministic + original math path), debug(keep diagnostics).",
)
@click.option("--use_potentials", is_flag=True, help="Enable FK steering + physical guidance potentials")
@click.option("--write_full_pae/--no_write_full_pae", default=False, help="Dump full PAE matrix to npz (default: off)")
@click.option("--write_full_pde/--no_write_full_pde", default=False, help="Dump full PDE matrix to npz (default: off)")
@click.option(
    "--save_trunk_checkpoints/--no_save_trunk_checkpoints",
    default=False,
    help="Save large trunk_checkpoints.pt debug artifact (default: off)",
)
@click.option(
    "--save_granular_checkpoints/--no_save_granular_checkpoints",
    default=False,
    help="Save granular_ckpts.pt debug artifact (default: off)",
)
@click.option("--dc_pairwise_chunk_size", type=int, default=512, help="Row chunk size for diffusion pairwise conditioner")
@click.option("--dc_token_bias_chunk_size", type=int, default=256, help="Row chunk size for diffusion token_trans_bias")
@click.option("--dc_atom_encoder_chunk_size", type=int, default=256, help="Row chunk size for diffusion atom encoder z_to_p")
@click.option(
    "--keep_pde_logits",
    is_flag=True,
    help="After chunked PDE, also concatenate full pde_logits on CPU (~large). Default off to save memory; pde field is always built.",
)
@click.option("--seed", type=int, default=None, help="Random seed for deterministic runs")
@click.option("--skip_processing", is_flag=True, help="Reuse existing out_dir/processed without running process_inputs")
@click.option(
    "--processed-dir",
    "--processed_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Reuse processed inputs from this directory while writing predictions to out_dir.",
)
@click.option(
    "--template-t-chunk-size",
    type=int,
    default=None,
    help="Boltz2 template_module.template_t_chunk_size (pairformer along T); lowers VRAM on large templates.",
)
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
    run_profile: str = "prod",
    use_potentials: bool = False,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    save_trunk_checkpoints: bool = False,
    save_granular_checkpoints: bool = False,
    dc_pairwise_chunk_size: int = 512,
    dc_token_bias_chunk_size: int = 256,
    dc_atom_encoder_chunk_size: int = 256,
    keep_pde_logits: bool = False,
    seed: int = None,
    skip_processing: bool = False,
    processed_dir: str | None = None,
    template_t_chunk_size: int | None = None,
):
    """Run Boltz 2 with proper FastFold-style DAP (no model duplication)."""

    # Initialize DAP
    from boltz_distributed import init_dap, get_dap_size, get_dap_rank

    init_dap()

    dap_rank = get_dap_rank()
    dap_size = get_dap_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')

    run_profile = run_profile.lower()
    if run_profile == "parity":
        # Parity mode pins execution to deterministic, PyTorch-native paths.
        if seed is None:
            seed = 0
        no_kernels = True
        use_flex_attention = False
        use_flex_attention_chunked = False
        # Disable heavy debug artifacts to avoid changing peak memory behavior.
        save_trunk_checkpoints = False
        save_granular_checkpoints = False
        write_full_pae = False
        write_full_pde = False

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
    _os.environ['BOLTZ_SAVE_TRUNK_CKPT'] = "1" if save_trunk_checkpoints else "0"
    _os.environ['BOLTZ_SAVE_GRAN_CKPT'] = "1" if save_granular_checkpoints else "0"
    _os.environ['BOLTZ_DC_PAIRWISE_CHUNK'] = str(dc_pairwise_chunk_size)
    _os.environ['BOLTZ_DC_TOKEN_BIAS_CHUNK'] = str(dc_token_bias_chunk_size)
    _os.environ['BOLTZ_DC_ATOM_ENCODER_CHUNK'] = str(dc_atom_encoder_chunk_size)
    _os.environ['BOLTZ_DAP_KEEP_PDE_LOGITS'] = "1" if keep_pde_logits else "0"
    log_file = out_dir / "gpu_memory.log"

    def rank_print(msg):
        if dap_rank == 0:
            print(msg)

    rank_print(f"\n{'='*70}")
    rank_print(f"BOLTZ 2 DAP v2 INFERENCE ({dap_size} GPUs)")
    rank_print(f"{'='*70}")
    rank_print(f"Profile: {run_profile}")
    rank_print(f"Input: {data}")
    processed_root = Path(processed_dir).expanduser() if processed_dir else out_dir / "processed"

    rank_print(f"Output: {out_dir}")
    if processed_dir:
        rank_print(f"Processed input dir: {processed_root}")
    rank_print(f"No model duplication — activations sharded across GPUs")
    rank_print(
        "Trunk checkpoints: "
        + ("on" if save_trunk_checkpoints else "off")
        + " | granular checkpoints: "
        + ("on" if save_granular_checkpoints else "off")
        + f" | diffusion chunks: pw={dc_pairwise_chunk_size},"
        + f" ttb={dc_token_bias_chunk_size}, ae={dc_atom_encoder_chunk_size}"
    )
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
        _apply_template_t_chunk_size,
    )
    from boltz.model.models.boltz2 import Boltz2
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzWriter

    rank_print("[1/6] Processing input data...")

    # Process inputs (only on rank 0)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    processed_manifest = processed_root / "manifest.json"
    fingerprint_path = processed_root / "input_fingerprint.json"
    expected_fingerprint = _build_processing_fingerprint(data, use_msa_server)
    processing_error = None
    if dap_rank == 0:
        try:
            if processed_dir and not skip_processing:
                raise ValueError(
                    "--processed_dir is only valid with --skip_processing. "
                    "Remove --processed_dir to regenerate processed inputs in out_dir."
                )
            if skip_processing:
                if not processed_manifest.exists():
                    raise FileNotFoundError(
                        "--skip_processing was set, but processed manifest was not "
                        f"found: {processed_manifest}"
                    )
                if not fingerprint_path.exists():
                    raise FileNotFoundError(
                        "--skip_processing was set, but input fingerprint is missing. "
                        "Re-run once without --skip_processing to create it."
                    )
                stored_fingerprint = json.loads(fingerprint_path.read_text())
                if stored_fingerprint != expected_fingerprint:
                    diff_msg = _diff_fingerprint(stored_fingerprint, expected_fingerprint)
                    raise RuntimeError(
                        "Processed inputs were generated from different input/settings. "
                        f"Changed fields: {diff_msg}. Re-run without --skip_processing."
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
                fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
                fingerprint_path.write_text(
                    json.dumps(expected_fingerprint, indent=2, sort_keys=True) + "\n"
                )
        except Exception as err:
            processing_error = f"{type(err).__name__}: {err}"

    error_box = [processing_error]
    dist.broadcast_object_list(error_box, src=0)
    if error_box[0] is not None:
        if monitor:
            monitor.stop()
        dist.destroy_process_group()
        raise RuntimeError(f"Input processing preflight failed: {error_box[0]}")

    # Load manifest
    manifest = Manifest.load(processed_manifest)
    filtered_manifest = filter_inputs_structure(manifest=manifest, outdir=out_dir)

    if not filtered_manifest.records:
        rank_print("No predictions needed.")
        dist.destroy_process_group()
        return

    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_root / "structures",
        msa_dir=processed_root / "msa",
        constraints_dir=(processed_root / "constraints") if (processed_root / "constraints").exists() else None,
        template_dir=(processed_root / "templates") if (processed_root / "templates").exists() else None,
        extra_mols_dir=(processed_root / "mols") if (processed_root / "mols").exists() else None,
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
    _apply_template_t_chunk_size(model, template_t_chunk_size)
    if template_t_chunk_size is not None:
        rank_print(f"  ✓ template_t_chunk_size={template_t_chunk_size} (template pairformer)")

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

        # For very large systems, force stronger MSA streaming to keep VRAM headroom.
        # This only changes execution chunking (not model math).
        if N >= 8000:
            current_pwa_chunk = int(os.environ.get("BOLTZ_PWA_S_CHUNK", "0"))
            if current_pwa_chunk <= 0 or current_pwa_chunk > 2:
                os.environ["BOLTZ_PWA_S_CHUNK"] = "2"
                rank_print("    [OOM-GUARD] BOLTZ_PWA_S_CHUNK set to 2 for N>=8000")
            current_opm_out_chunk = int(os.environ.get("BOLTZ_OPM_OUT_CHUNK", "0"))
            if current_opm_out_chunk <= 0 or current_opm_out_chunk > 8:
                os.environ["BOLTZ_OPM_OUT_CHUNK"] = "8"
                rank_print("    [OOM-GUARD] BOLTZ_OPM_OUT_CHUNK set to 8 for N>=8000")
            current_trimul_out_tile = int(os.environ.get("BOLTZ_TRIMUL_OUT_TILE", "0"))
            if current_trimul_out_tile <= 0 or current_trimul_out_tile > 256:
                os.environ["BOLTZ_TRIMUL_OUT_TILE"] = "256"
                rank_print("    [OOM-GUARD] BOLTZ_TRIMUL_OUT_TILE set to 256 for N>=8000")
            current_tri_att_chunk = int(os.environ.get("BOLTZ_TRI_ATT_CHUNK", "0") or "0")
            if current_tri_att_chunk <= 0 or current_tri_att_chunk > 2:
                os.environ["BOLTZ_TRI_ATT_CHUNK"] = "2"
                rank_print("    [OOM-GUARD] BOLTZ_TRI_ATT_CHUNK set to 2 for N>=8000")

        mem_before = torch.cuda.memory_allocated(device) / 1024**2
        rank_print(f"    Memory before forward: {mem_before:.0f} MB")

        torch.cuda.reset_peak_memory_stats(device)

        pred_dict = None

        try:
            # All ranks follow the exact same DAP forward path. Rank 0 only
            # converts the returned tensors into the writer/predict payload.
            with torch.no_grad():
                model_out = model(
                    batch,
                    recycling_steps=recycling_steps,
                    num_sampling_steps=sampling_steps,
                    diffusion_samples=diffusion_samples,
                    max_parallel_samples=1,
                    run_confidence_sequentially=True,
                )
            if dap_rank == 0:
                pred_dict = _build_prediction_dict(model, batch, model_out)
        except RuntimeError as err:
            if _is_oom_error(err):
                print(f"    GPU {dap_rank}: Runtime OOM in batch {batch_idx}; failing fast")
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"DAP inference OOM on rank {dap_rank}, batch {batch_idx}"
                ) from err
            else:
                raise

        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"    GPU {dap_rank}: Memory after forward: {mem_after:.0f} MB")
        print(f"    GPU {dap_rank}: Peak memory: {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")

        # Barrier to sync all GPUs before next batch
        dist.barrier()

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
