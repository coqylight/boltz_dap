#!/bin/bash
#SBATCH --job-name=kern_prec
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/kernel_precision_%j.log
#SBATCH --mail-type=END,FAIL

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
DAP_PROJECT="/project/engvimmune/gleeai/boltz_dap"
YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
PROCESSED_SRC="/project/engvimmune/gleeai/boltz_output/ref_subop_v2/processed"
BASE="/project/engvimmune/gleeai/boltz_output"
SEED=42

echo "========================================"
echo "KERNEL PRECISION COMPARISON TEST"
echo "========================================"
echo "Comparing 3 configs (all 1-GPU, same seed $SEED):"
echo "  1. No kernels (PyTorch native) — baseline"
echo "  2. Kernels with default precision (TF32)"
echo "  3. Kernels with IEEE f32 precision"
echo ""

# ── CONFIG 1: No kernels ──────────────────────────
echo "================================================"
echo "CONFIG 1: No kernels (PyTorch native)"
echo "================================================"
OUT1="$BASE/prec_test_nokern"
rm -rf "$OUT1"
mkdir -p "$OUT1"
cp -r "$PROCESSED_SRC" "$OUT1/processed"

BOLTZ_OUT_DIR="$OUT1" \
uv run --project "$BOLTZ_PROJECT" \
    python /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$OUT1" \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --seed $SEED \
    --no_kernels

echo "  ✓ Config 1 done"
echo ""

# ── CONFIG 2: Kernels with DEFAULT precision (TF32) ──
echo "================================================"
echo "CONFIG 2: Kernels with default precision (TF32)"
echo "================================================"
OUT2="$BASE/prec_test_kern_tf32"
rm -rf "$OUT2"
mkdir -p "$OUT2"
cp -r "$PROCESSED_SRC" "$OUT2/processed"

BOLTZ_OUT_DIR="$OUT2" \
BOLTZ_KERNEL_PRECISION="default" \
uv run --project "$BOLTZ_PROJECT" \
    python /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$OUT2" \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --seed $SEED

echo "  ✓ Config 2 done"
echo ""

# ── CONFIG 3: Kernels with IEEE f32 precision ──
echo "================================================"
echo "CONFIG 3: Kernels with IEEE f32 precision"
echo "================================================"
OUT3="$BASE/prec_test_kern_ieee"
rm -rf "$OUT3"
mkdir -p "$OUT3"
cp -r "$PROCESSED_SRC" "$OUT3/processed"

BOLTZ_OUT_DIR="$OUT3" \
BOLTZ_KERNEL_PRECISION="ieee" \
uv run --project "$BOLTZ_PROJECT" \
    python /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$OUT3" \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --seed $SEED

echo "  ✓ Config 3 done"
echo ""

# ── COMPARE ──────────────────────────────────────────
echo "================================================"
echo "COMPARISON: Trunk checkpoints z/s"
echo "================================================"

uv run --project "$BOLTZ_PROJECT" python3 - <<'PYEOF'
import torch

configs = {
    "no_kernels": "/project/engvimmune/gleeai/boltz_output/prec_test_nokern",
    "kern_tf32":  "/project/engvimmune/gleeai/boltz_output/prec_test_kern_tf32",
    "kern_ieee":  "/project/engvimmune/gleeai/boltz_output/prec_test_kern_ieee",
}

# Load trunk checkpoints
ckpts = {}
for name, path in configs.items():
    try:
        ckpts[name] = torch.load(f"{path}/trunk_checkpoints.pt", map_location="cpu", weights_only=True)
        print(f"  Loaded {name}: {len(ckpts[name])} checkpoints")
    except Exception as e:
        print(f"  FAILED {name}: {e}")

# Load granular checkpoints
grans = {}
for name, path in configs.items():
    try:
        grans[name] = torch.load(f"{path}/granular_ckpts.pt", map_location="cpu", weights_only=True)
        print(f"  Loaded {name} granular: {len(grans[name])} keys")
    except Exception as e:
        print(f"  FAILED {name} granular: {e}")

baseline = "no_kernels"

def compare_tensor(r, d):
    for dim in range(r.dim()):
        if r.shape[dim] != d.shape[dim]:
            ms = min(r.shape[dim], d.shape[dim])
            r = r.narrow(dim, 0, ms)
            d = d.narrow(dim, 0, ms)
    diff = (r.float() - d.float()).abs()
    n_diff = (diff > 0).sum().item()
    pct = 100.0 * n_diff / diff.numel() if diff.numel() > 0 else 0
    return diff.mean().item(), diff.max().item(), n_diff, diff.numel(), pct

# Compare trunk checkpoints
print("\n" + "=" * 130)
print(f"{'Key':<40} {'tensor':<6} | {'kern_tf32 vs baseline':^40} | {'kern_ieee vs baseline':^40}")
print(f"{'':40} {'':6} | {'mean':>10} {'max':>10} {'%diff':>8} {'match':>6} | {'mean':>10} {'max':>10} {'%diff':>8} {'match':>6}")
print("-" * 130)

for key in sorted(ckpts.get(baseline, {}).keys()):
    for subk in ['z', 's']:
        ref = ckpts[baseline][key]
        if isinstance(ref, dict) and subk in ref:
            rv = ref[subk]
            cols = []
            for comp in ["kern_tf32", "kern_ieee"]:
                if comp in ckpts and key in ckpts[comp]:
                    d = ckpts[comp][key]
                    if isinstance(d, dict) and subk in d:
                        mean, mx, nd, nt, pct = compare_tensor(rv, d[subk])
                        ident = "✅" if mx == 0 else "❌"
                        cols.append(f"{mean:>10.8f} {mx:>10.6f} {pct:>7.2f}% {ident:>6}")
                    else:
                        cols.append(f"{'N/A':>40}")
                else:
                    cols.append(f"{'N/A':>40}")
            print(f"{key:<40} {subk:<6} | {cols[0]} | {cols[1]}")

# Compare granular checkpoints
print("\n" + "=" * 130)
print("GRANULAR CHECKPOINT COMPARISON")
print(f"{'Key':<40}       | {'kern_tf32 vs baseline':^40} | {'kern_ieee vs baseline':^40}")
print(f"{'':46} | {'mean':>10} {'max':>10} {'%diff':>8} {'match':>6} | {'mean':>10} {'max':>10} {'%diff':>8} {'match':>6}")
print("-" * 130)

for key in sorted(grans.get(baseline, {}).keys()):
    rv = grans[baseline][key]
    cols = []
    for comp in ["kern_tf32", "kern_ieee"]:
        if comp in grans and key in grans[comp]:
            mean, mx, nd, nt, pct = compare_tensor(rv, grans[comp][key])
            ident = "✅" if mx == 0 else "❌"
            cols.append(f"{mean:>10.8f} {mx:>10.6f} {pct:>7.2f}% {ident:>6}")
        else:
            cols.append(f"{'N/A':>40}")
    print(f"{key:<40}       | {cols[0]} | {cols[1]}")

# Compare z/s final tensors
print("\n" + "=" * 130)
print("FINAL z/s TENSOR COMPARISON")
print("-" * 130)
for comp in ["kern_tf32", "kern_ieee"]:
    try:
        ref_zs = torch.load(f"{configs[baseline]}/zs_tensors.pt", map_location="cpu", weights_only=True)
        comp_zs = torch.load(f"{configs[comp]}/zs_tensors.pt", map_location="cpu", weights_only=True)
        for t in ['z', 's']:
            mean, mx, nd, nt, pct = compare_tensor(ref_zs[t], comp_zs[t])
            cos = torch.nn.functional.cosine_similarity(
                ref_zs[t].float().flatten().unsqueeze(0),
                comp_zs[t].float().flatten().unsqueeze(0)
            ).item()
            ident = "✅" if mx == 0 else "❌"
            print(f"  {comp:15} {t}: {ident}  mean_diff={mean:.8f}  max_diff={mx:.6f}  {pct:.2f}% differ  cosine={cos:.8f}")
    except Exception as e:
        print(f"  {comp:15}: FAILED - {e}")

# Compare CIF files
print("\n" + "=" * 130)
print("CIF FILE COMPARISON")
print("-" * 130)
import os
cif_base = f"{configs[baseline]}/predictions/1LP3_pentamer_from_tetramer/1LP3_pentamer_from_tetramer_model_0.cif"
for comp in ["kern_tf32", "kern_ieee"]:
    cif_comp = f"{configs[comp]}/predictions/1LP3_pentamer_from_tetramer/1LP3_pentamer_from_tetramer_model_0.cif"
    if os.path.exists(cif_base) and os.path.exists(cif_comp):
        with open(cif_base) as f1, open(cif_comp) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        n_diff = sum(1 for a, b in zip(lines1, lines2) if a != b)
        ident = "✅ IDENTICAL" if n_diff == 0 else f"❌ {n_diff} lines differ"
        print(f"  {comp:15} vs {baseline}: {ident} (total lines: {len(lines1)})")
    else:
        print(f"  {comp:15}: CIF file not found")

print("\n" + "=" * 130)
print("ALL DONE")
PYEOF
