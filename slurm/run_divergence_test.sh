#!/bin/bash
# Quick divergence diagnosis: run DAP script with 1 GPU (no scatter/gather)
# to get reference trunk_checkpoints.pt, then compare per-checkpoint against
# the 2-GPU checkpoints we already have.

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
REF_OUT="$BASE/ref_1gpu_seed42"
DAP_CKPT="$BASE/dap_predict_seed42/trunk_checkpoints.pt"
BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"

echo "=============================================="
echo "STEP 1: Run DAP code with 1 GPU (reference)"
echo "=============================================="

# Copy processed data from the existing DAP run
mkdir -p "$REF_OUT"
if [ -d "$BASE/dap_predict_seed42/processed" ]; then
    cp -r "$BASE/dap_predict_seed42/processed" "$REF_OUT/processed" 2>/dev/null || true
elif [ -d "$BASE/baseline_predict_seed42/boltz_results_1LP3_pentamer_from_tetramer/processed" ]; then
    cp -r "$BASE/baseline_predict_seed42/boltz_results_1LP3_pentamer_from_tetramer/processed" "$REF_OUT/processed" 2>/dev/null || true
fi

# Run with 1 GPU using torchrun (needed for dist init, but dap_size=1 means no scatter/gather)
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=1 --standalone \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$REF_OUT" \
    --seed 42 \
    --use_msa_server

echo ""
echo "=============================================="
echo "STEP 2: Compare per-checkpoint (1GPU vs 2GPU)"
echo "=============================================="

REF_CKPT="$REF_OUT/trunk_checkpoints.pt"

uv run --project "$BOLTZ_PROJECT" python3 - <<'PYEOF'
import torch

ref_path = "/project/engvimmune/gleeai/boltz_output/ref_1gpu_seed42/trunk_checkpoints.pt"
dap_path = "/project/engvimmune/gleeai/boltz_output/dap_predict_seed42/trunk_checkpoints.pt"

print(f"Reference (1-GPU): {ref_path}")
print(f"DAP (2-GPU):       {dap_path}")

ref = torch.load(ref_path, map_location="cpu", weights_only=True)
dap = torch.load(dap_path, map_location="cpu", weights_only=True)

ref_keys = sorted(ref.keys())
dap_keys = sorted(dap.keys())
common = [k for k in ref_keys if k in dap_keys]

print(f"\nRef checkpoints: {ref_keys}")
print(f"DAP checkpoints: {dap_keys}")
print(f"Common:          {len(common)}")

print(f"\n{'='*140}")
print(f"{'Checkpoint':<28}| {'z_identical':>11} {'z_diff_mean':>12} {'z_diff_max':>12} {'z_cos':>10} | {'s_identical':>11} {'s_diff_mean':>12} {'s_diff_max':>12} {'s_cos':>10}")
print("-" * 140)

first_z_div = None
first_s_div = None

for key in common:
    zr = ref[key]["z"].float()
    zd = dap[key]["z"].float()
    N = min(zr.shape[1], zd.shape[1])
    zr, zd = zr[:, :N, :N], zd[:, :N, :N]
    
    z_identical = torch.equal(ref[key]["z"][:, :N, :N], dap[key]["z"][:, :N, :N])
    z_diff = (zr - zd).abs()
    z_cos = torch.nn.functional.cosine_similarity(zr.flatten(), zd.flatten(), dim=0).item()
    
    sr = ref[key]["s"].float()
    sd = dap[key]["s"].float()
    Ns = min(sr.shape[1], sd.shape[1])
    sr, sd = sr[:, :Ns], sd[:, :Ns]
    
    s_identical = torch.equal(ref[key]["s"][:, :Ns], dap[key]["s"][:, :Ns])
    s_diff = (sr - sd).abs()
    s_cos = torch.nn.functional.cosine_similarity(sr.flatten(), sd.flatten(), dim=0).item()
    
    z_mark = "✅" if z_identical else "❌"
    s_mark = "✅" if s_identical else "❌"
    
    if not z_identical and first_z_div is None:
        first_z_div = key
    if not s_identical and first_s_div is None:
        first_s_div = key
    
    print(f"{key:<28}| {z_mark:>11} {z_diff.mean():>12.8f} {z_diff.max():>12.8f} {z_cos:>10.8f} | {s_mark:>11} {s_diff.mean():>12.8f} {s_diff.max():>12.8f} {s_cos:>10.8f}")

print(f"\n{'='*140}")
print("FIRST DIVERGENCE POINTS")
print(f"{'='*140}")
print(f"  z: {first_z_div if first_z_div else 'No divergence detected ✅'}")
print(f"  s: {first_s_div if first_s_div else 'No divergence detected ✅'}")

if first_z_div:
    zr = ref[first_z_div]["z"].float()
    zd = dap[first_z_div]["z"].float()
    N = min(zr.shape[1], zd.shape[1])
    zr, zd = zr[:, :N, :N], zd[:, :N, :N]
    diff = (zr - zd).abs()
    print(f"\n  First z divergence '{first_z_div}' details:")
    print(f"    Elements differing: {(diff > 0).sum().item()} / {diff.numel()}")
    print(f"    Mean diff:   {diff.mean():.10f}")
    print(f"    Max diff:    {diff.max():.10f}")
    print(f"    Diff>1e-5:   {(diff > 1e-5).sum().item()}")
    print(f"    Diff>1e-3:   {(diff > 1e-3).sum().item()}")
    print(f"    Diff>0.01:   {(diff > 0.01).sum().item()}")
    print(f"    Diff>0.1:    {(diff > 0.1).sum().item()}")

    # Check previous checkpoint to see if the divergence is new
    prev_key = common[max(0, common.index(first_z_div) - 1)]
    if prev_key != first_z_div:
        zr_prev = ref[prev_key]["z"].float()
        zd_prev = dap[prev_key]["z"].float()
        N_prev = min(zr_prev.shape[1], zd_prev.shape[1])
        prev_identical = torch.equal(ref[prev_key]["z"][:, :N_prev, :N_prev], dap[prev_key]["z"][:, :N_prev, :N_prev])
        print(f"\n  Previous checkpoint '{prev_key}': {'✅ identical' if prev_identical else '❌ already diverged'}")

PYEOF

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
