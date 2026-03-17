#!/bin/bash
# Controlled A/B test: SDPA vs FlexAttention with SAME seed
# Run this inside an srun allocation with 2 GPUs

set -e

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
SCRIPT="/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py"
SEED=42

echo "============================================================"
echo "CONTROLLED A/B TEST: SDPA vs FlexAttention (seed=$SEED)"
echo "============================================================"

# ─── Run 1: SDPA only (FlexAttention OFF) ───
echo ""
echo ">>> RUN 1: SDPA (FlexAttention OFF, seed=$SEED)"
echo "============================================================"
export BOLTZ_USE_FLEX_ATTENTION=0
torchrun --nproc_per_node=2 --standalone "$SCRIPT" \
    "$YAML" \
    --out_dir "$BASE/zs_test_sdpa" \
    --seed $SEED \
    --use_msa_server

echo ""
echo ">>> RUN 2: FlexAttention ON (seed=$SEED)"
echo "============================================================"
export BOLTZ_USE_FLEX_ATTENTION=1
torchrun --nproc_per_node=2 --standalone "$SCRIPT" \
    "$YAML" \
    --out_dir "$BASE/zs_test_flex" \
    --seed $SEED \
    --use_msa_server

echo ""
echo ">>> COMPARING z/s tensors..."
echo "============================================================"
python3 - <<'PYEOF'
import torch
import sys

sdpa = torch.load("/project/engvimmune/gleeai/boltz_output/zs_test_sdpa/zs_tensors.pt", 
                  map_location="cpu", weights_only=True)
flex = torch.load("/project/engvimmune/gleeai/boltz_output/zs_test_flex/zs_tensors.pt",
                  map_location="cpu", weights_only=True)

print("=" * 60)
print("Z (PAIR REPRESENTATION) COMPARISON")
print("=" * 60)
z_s, z_f = sdpa["z"], flex["z"]
print(f"  Shape: {list(z_s.shape)}")
print(f"  SDPA  dtype={z_s.dtype}, mean={z_s.float().mean():.6f}, std={z_s.float().std():.6f}")
print(f"  Flex  dtype={z_f.dtype}, mean={z_f.float().mean():.6f}, std={z_f.float().std():.6f}")
diff_z = (z_s.float() - z_f.float()).abs()
print(f"  Diff  mean={diff_z.mean():.6e}, max={diff_z.max():.6e}")
rel = diff_z / (z_s.float().abs() + 1e-8)
print(f"  Rel   mean={rel.mean():.6e}, max={rel.max():.6e}")

print()
print("=" * 60)
print("S (SINGLE REPRESENTATION) COMPARISON")
print("=" * 60)
s_s, s_f = sdpa["s"], flex["s"]
print(f"  Shape: {list(s_s.shape)}")
print(f"  SDPA  dtype={s_s.dtype}, mean={s_s.float().mean():.6f}, std={s_s.float().std():.6f}")
print(f"  Flex  dtype={s_f.dtype}, mean={s_f.float().mean():.6f}, std={s_f.float().std():.6f}")
diff_s = (s_s.float() - s_f.float()).abs()
print(f"  Diff  mean={diff_s.mean():.6e}, max={diff_s.max():.6e}")
rel_s = diff_s / (s_s.float().abs() + 1e-8)
print(f"  Rel   mean={rel_s.mean():.6e}, max={rel_s.max():.6e}")

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
if diff_z.mean() < 1e-3 and diff_s.mean() < 1e-3:
    print("  ✓ z and s are nearly identical — FlexAttention does NOT corrupt trunk output")
    print("    Structural differences are from diffusion sampling randomness")
elif diff_z.mean() < 0.1:
    print("  ⚠ Small differences in z/s — may compound in diffusion but trunk is approximately correct")
else:
    print("  ✗ Large z/s differences — FlexAttention corrupts trunk output")
PYEOF

echo ""
echo "Done. Check CIF files in:"
echo "  SDPA: $BASE/zs_test_sdpa/predictions/"
echo "  Flex: $BASE/zs_test_flex/predictions/"
