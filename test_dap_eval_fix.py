"""
Quick verification: does calling .eval() on the DAP layer fix the divergence?

Run: torchrun --nproc_per_node=2 --standalone test_dap_eval_fix.py 2>&1 | tee /project/engvimmune/gleeai/boltz_dap/eval_fix_output.log
"""

import os
import sys
import torch
import torch.distributed as dist
from dataclasses import asdict
from pathlib import Path

PROJ = "/project/engvimmune/gleeai"
sys.path.insert(0, os.path.join(PROJ, "boltz_dap/boltz_dap_v2"))
sys.path.insert(0, os.path.join(PROJ, "boltz_dap"))
sys.path.insert(0, os.path.join(PROJ, "boltz/src"))

from boltz_distributed.core import init_dap, get_dap_size, get_dap_rank
from boltz_distributed.comm import scatter, gather

def rank_print(*a, **k):
    if get_dap_rank() == 0:
        print(*a, **k, flush=True)

def compare(name, a, b):
    if a.shape != b.shape:
        rank_print(f"  ❌ {name}: SHAPE {list(a.shape)} vs {list(b.shape)}")
        return
    diff = (a.float() - b.float()).abs()
    m = diff.mean().item(); mx = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    v = "✅" if mx < 1e-3 else ("⚠️ " if mx < 1e-1 else "❌")
    rank_print(f"  {v} {name:40s} mean={m:.2e} max={mx:.2e} cos={cos:.6f}")

def trim_gather(z_scat, N):
    full = gather(z_scat.contiguous(), dim=1)
    if full.shape[1] > N:
        full = full[:, :N, :N, :]
    return full


def main():
    init_dap()
    # Suppress profiling noise
    import dap_trimul; dap_trimul.DIAG = False

    rank_print("Loading model...")
    from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    cache = Path("~/.boltz").expanduser()
    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt", strict=True,
        predict_args={"recycling_steps": 3, "sampling_steps": 200,
                      "diffusion_samples": 1, "max_parallel_samples": 1,
                      "write_confidence_summary": True, "write_full_pae": False,
                      "write_full_pde": False},
        map_location=f"cuda:{get_dap_rank()}",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    rank_print("Model loaded.\n")

    # Get template layer
    tmpl = model.template_module
    if hasattr(tmpl, '_orig_mod'):
        tmpl = tmpl._orig_mod
    layer = tmpl.pairformer.layers[0]

    from dap_pairformer_noseq import DAPPairformerNoSeqLayer

    D = layer.tri_mul_out.norm_in.normalized_shape[0]
    N = 64

    torch.manual_seed(123)
    z_init = torch.randn(1, N, N, D, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)
    z_scat = scatter(z_init, dim=1)
    pm_scat = scatter(pm, dim=1)

    # --- Original layer (reference) ---
    rank_print("=" * 70)
    rank_print("TEST: DAP layer with .eval() vs without .eval()")
    rank_print("=" * 70)

    with torch.no_grad():
        z_orig = layer(z_init.clone(), pair_mask=pm, use_kernels=False)

    # --- DAP layer WITHOUT .eval() (the BUG) ---
    dap_layer_buggy = DAPPairformerNoSeqLayer(layer)
    rank_print(f"\n  dap_layer_buggy.training = {dap_layer_buggy.training}  ← THIS IS THE BUG")
    with torch.no_grad():
        z_dap_buggy = dap_layer_buggy(z_scat.clone(), pm_scat, use_kernels=False)
    z_dap_buggy_full = trim_gather(z_dap_buggy, N)

    # --- DAP layer WITH .eval() (the FIX) ---
    dap_layer_fixed = DAPPairformerNoSeqLayer(layer)
    dap_layer_fixed.eval()  # THE FIX
    rank_print(f"  dap_layer_fixed.training = {dap_layer_fixed.training}  ← FIXED")
    with torch.no_grad():
        z_dap_fixed = dap_layer_fixed(z_scat.clone(), pm_scat, use_kernels=False)
    z_dap_fixed_full = trim_gather(z_dap_fixed, N)

    # --- Compare ---
    rank_print(f"\n--- Results ---")
    compare("BUGGY: orig vs DAP(training=True) ", z_orig, z_dap_buggy_full)
    compare("FIXED: orig vs DAP(training=False)", z_orig, z_dap_fixed_full)

    # Also test main PairformerLayer
    rank_print(f"\n--- Main PairformerLayer ---")
    from dap_pairformer import DAPPairformerLayer
    pf = model.pairformer_module
    if hasattr(pf, '_orig_mod'):
        pf = pf._orig_mod
    pf_layer = pf.layers[0]

    D_z = pf_layer.tri_mul_out.norm_in.normalized_shape[0]
    D_s = pf_layer.attention.proj_z.weight.shape[1]
    torch.manual_seed(456)
    z_pf = torch.randn(1, N, N, D_z, device="cuda", dtype=torch.float32)
    s_pf = torch.randn(1, N, D_s, device="cuda", dtype=torch.float32)
    mask_pf = torch.ones(1, N, device="cuda", dtype=torch.float32)

    z_scat_pf = scatter(z_pf, dim=1)
    pm_scat_pf = scatter(pm, dim=1)

    with torch.no_grad():
        s_orig, z_orig_pf = pf_layer(s_pf.clone(), z_pf.clone(), mask_pf, pm, use_kernels=False)

    dap_pf_buggy = DAPPairformerLayer(pf_layer)
    rank_print(f"\n  dap_pf_buggy.training = {dap_pf_buggy.training}  ← BUG")
    with torch.no_grad():
        s_buggy, z_buggy = dap_pf_buggy(s_pf.clone(), z_scat_pf.clone(), mask_pf, pm_scat_pf, use_kernels=False)
    z_buggy_full = trim_gather(z_buggy, N)

    dap_pf_fixed = DAPPairformerLayer(pf_layer)
    dap_pf_fixed.eval()
    rank_print(f"  dap_pf_fixed.training = {dap_pf_fixed.training}  ← FIXED")
    with torch.no_grad():
        s_fixed, z_fixed = dap_pf_fixed(s_pf.clone(), z_scat_pf.clone(), mask_pf, pm_scat_pf, use_kernels=False)
    z_fixed_full = trim_gather(z_fixed, N)

    rank_print(f"\n--- PairformerLayer Results ---")
    compare("BUGGY: orig z vs DAP(training=True) ", z_orig_pf, z_buggy_full)
    compare("FIXED: orig z vs DAP(training=False)", z_orig_pf, z_fixed_full)
    compare("BUGGY: orig s vs DAP(training=True) ", s_orig, s_buggy)
    compare("FIXED: orig s vs DAP(training=False)", s_orig, s_fixed)

    rank_print(f"\n{'=' * 70}")
    rank_print("DONE — If FIXED rows show ✅, the eval() fix resolves the DAP bug!")
    rank_print("=" * 70)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
