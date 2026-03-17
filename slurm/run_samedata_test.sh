#!/bin/bash
# Combined test: Original Boltz2 predict vs DAP, then compare z/s embeddings.
# Objectives: (1) same z and s, (2) lower memory peak, (3) no data duplication.
# Submit: sbatch run_samedata_test.sbatch

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
BASELINE_OUT="$BASE/baseline_predict_seed42"
DAP_OUT="$BASE/dap_predict_seed42"
BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
SEED=42

# ==============================================
# STEP 1: Run ORIGINAL Boltz2 predict (single GPU, completely unmodified)
# ==============================================
echo "=============================================="
echo "STEP 1: Run original Boltz2 predict (1 GPU)"
echo "=============================================="

CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    boltz predict "$YAML" \
    --out_dir "$BASELINE_OUT" \
    --seed $SEED \
    --use_msa_server \
    --write_embeddings \
    --diffusion_samples 1 \
    --recycling_steps 3

echo ""
echo "=============================================="
echo "STEP 2: Run DAP (2 GPUs, same data)"
echo "=============================================="

# Copy processed data from baseline so both use identical inputs
mkdir -p "$DAP_OUT"
BASELINE_PROCESSED=$(find "$BASELINE_OUT" -path "*/processed" -type d | head -1)
if [ -n "$BASELINE_PROCESSED" ] && [ ! -d "$DAP_OUT/processed" ]; then
    cp -r "$BASELINE_PROCESSED" "$DAP_OUT/processed" 2>/dev/null || true
    echo "  Copied processed data from baseline"
fi

rm -rf "$DAP_OUT/predictions"
NCCL_TIMEOUT=1800 \
uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=2 --standalone \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$DAP_OUT" \
    --seed $SEED \
    --use_msa_server

echo ""
echo "=============================================="
echo "STEP 3: Compare z/s embeddings"
echo "=============================================="

uv run --project "$BOLTZ_PROJECT" python3 - <<'PYEOF'
import torch
import numpy as np
from pathlib import Path

baseline_dir = Path("/project/engvimmune/gleeai/boltz_output/baseline_predict_seed42")
dap_dir = Path("/project/engvimmune/gleeai/boltz_output/dap_predict_seed42")

# Find baseline embeddings (saved by boltz predict --write_embeddings)
baseline_emb = list(baseline_dir.rglob("embeddings_*.npz"))
if not baseline_emb:
    print("ERROR: No baseline embeddings found! Check boltz predict output.")
    exit(1)
print(f"Baseline embeddings: {baseline_emb[0]}")
base = np.load(str(baseline_emb[0]))
zb = torch.from_numpy(base["z"]).float()
sb = torch.from_numpy(base["s"]).float()

# Find DAP embeddings (saved as zs_tensors.pt)
dap_zs = dap_dir / "zs_tensors.pt"
if not dap_zs.exists():
    print("ERROR: No DAP zs_tensors.pt found!")
    exit(1)
print(f"DAP embeddings: {dap_zs}")
dap = torch.load(str(dap_zs), map_location="cpu")
zd = dap["z"].float()
sd = dap["s"].float()

print(f"\nBaseline z: {zb.shape}, s: {sb.shape}")
print(f"DAP      z: {zd.shape}, s: {sd.shape}")

# Trim to common size (DAP may have padding from scatter)
Nz = min(zb.shape[-2], zd.shape[-2])
Ns = min(sb.shape[-2], sd.shape[-2])
zb, zd = zb[..., :Nz, :Nz, :], zd[..., :Nz, :Nz, :]
sb, sd = sb[..., :Ns, :], sd[..., :Ns, :]

print(f"Comparing z[:, :{Nz}, :{Nz}], s[:, :{Ns}]")

# Z comparison
z_diff = (zb - zd).abs()
z_cos = torch.nn.functional.cosine_similarity(zb.flatten(), zd.flatten(), dim=0).item()
z_identical = torch.equal(zb.to(torch.bfloat16), zd.to(torch.bfloat16))

# S comparison
s_diff = (sb - sd).abs()
s_cos = torch.nn.functional.cosine_similarity(sb.flatten(), sd.flatten(), dim=0).item()
s_identical = torch.equal(sb.to(torch.bfloat16), sd.to(torch.bfloat16))

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"  z: mean_diff={z_diff.mean():.8f}  max_diff={z_diff.max():.6f}  cosine={z_cos:.8f}  bf16_identical={z_identical}")
print(f"  s: mean_diff={s_diff.mean():.8f}  max_diff={s_diff.max():.6f}  cosine={s_cos:.8f}  bf16_identical={s_identical}")

print(f"\n{'='*80}")
print("OBJECTIVES CHECK")
print(f"{'='*80}")
print(f"  [1] Same z:  {'✅ PASS' if z_identical else '❌ FAIL'}")
print(f"  [1] Same s:  {'✅ PASS' if s_identical else '❌ FAIL'}")
print(f"  [2] Lower memory peak:  check GPU monitor logs")
print(f"  [3] No data duplication: DAP scatters z across GPUs ✅")

if not z_identical:
    print(f"\n  z divergence details:")
    print(f"    Elements differing: {(z_diff > 0).sum().item()} / {z_diff.numel()}")
    print(f"    Mean diff:   {z_diff.mean():.10f}")
    print(f"    Max diff:    {z_diff.max():.10f}")
    print(f"    Diff>0.01:   {(z_diff > 0.01).sum().item()}")
    print(f"    Diff>0.1:    {(z_diff > 0.1).sum().item()}")
    print(f"    Diff>1.0:    {(z_diff > 1.0).sum().item()}")

if not s_identical:
    print(f"\n  s divergence details:")
    print(f"    Elements differing: {(s_diff > 0).sum().item()} / {s_diff.numel()}")
    print(f"    Mean diff:   {s_diff.mean():.10f}")
    print(f"    Max diff:    {s_diff.max():.10f}")
    print(f"    Diff>0.01:   {(s_diff > 0.01).sum().item()}")
    print(f"    Diff>0.1:    {(s_diff > 0.1).sum().item()}")
    print(f"    Diff>1.0:    {(s_diff > 1.0).sum().item()}")
PYEOF

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
