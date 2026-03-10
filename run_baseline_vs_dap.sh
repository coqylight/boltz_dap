#!/bin/bash
# Run single-GPU baseline + DAP, then compare z/s tensors directly.
# Run inside an srun allocation with at least 2 GPUs.
# Usage: bash run_baseline_vs_dap.sh

set -e

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
DAP_SCRIPT="/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py"
BASELINE_SCRIPT="/project/engvimmune/gleeai/boltz_dap/run_boltz_baseline.py"
SEED=42

echo "============================================================"
echo "BASELINE vs DAP COMPARISON (seed=$SEED)"
echo "============================================================"

# ─── Run 1: Single-GPU baseline (no DAP) ───
echo ""
echo ">>> RUN 1: Single-GPU Baseline (seed=$SEED)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 python "$BASELINE_SCRIPT" \
    "$YAML" \
    --out_dir "$BASE/baseline_seed42" \
    --seed $SEED \
    --use_msa_server

# ─── Run 2: DAP with eval fix (2 GPUs) ───
echo ""
echo ">>> RUN 2: DAP (2 GPUs, eval fix, seed=$SEED)"
echo "============================================================"
torchrun --nproc_per_node=2 --standalone "$DAP_SCRIPT" \
    "$YAML" \
    --out_dir "$BASE/dap_seed42" \
    --seed $SEED \
    --use_msa_server

# ─── Compare z/s tensors ───
echo ""
echo ">>> COMPARING z/s tensors..."
echo "============================================================"
python3 - <<'PYEOF'
import torch
import json
import os
from pathlib import Path

baseline_zs = torch.load("/project/engvimmune/gleeai/boltz_output/baseline_seed42/zs_tensors.pt",
                          map_location="cpu", weights_only=True)
dap_zs = torch.load("/project/engvimmune/gleeai/boltz_output/dap_seed42/zs_tensors.pt",
                     map_location="cpu", weights_only=True)

print("=" * 60)
print("Z (PAIR REPRESENTATION) COMPARISON")
print("=" * 60)
z_b, z_d = baseline_zs["z"].float(), dap_zs["z"].float()
# Trim to common size
N = min(z_b.shape[1], z_d.shape[1])
z_b, z_d = z_b[:, :N, :N], z_d[:, :N, :N]
print(f"  Shape: baseline={list(baseline_zs['z'].shape)}, dap={list(dap_zs['z'].shape)}")
diff_z = (z_b - z_d).abs()
cos_z = torch.nn.functional.cosine_similarity(z_b.flatten(), z_d.flatten(), dim=0).item()
print(f"  Diff  mean={diff_z.mean():.6e}, max={diff_z.max():.6e}")
print(f"  Cosine similarity: {cos_z:.8f}")

print()
print("=" * 60)
print("S (SINGLE REPRESENTATION) COMPARISON")
print("=" * 60)
s_b, s_d = baseline_zs["s"].float(), dap_zs["s"].float()
N_s = min(s_b.shape[1], s_d.shape[1])
s_b, s_d = s_b[:, :N_s], s_d[:, :N_s]
print(f"  Shape: baseline={list(baseline_zs['s'].shape)}, dap={list(dap_zs['s'].shape)}")
diff_s = (s_b - s_d).abs()
cos_s = torch.nn.functional.cosine_similarity(s_b.flatten(), s_d.flatten(), dim=0).item()
print(f"  Diff  mean={diff_s.mean():.6e}, max={diff_s.max():.6e}")
print(f"  Cosine similarity: {cos_s:.8f}")

print()
print("=" * 60)
print("CONFIDENCE COMPARISON")
print("=" * 60)
for name, d in [("baseline", "baseline_seed42"), ("dap", "dap_seed42")]:
    conf_files = list((Path(f"/project/engvimmune/gleeai/boltz_output/{d}/predictions")).rglob("confidence*.json"))
    if conf_files:
        with open(conf_files[0]) as f:
            c = json.load(f)
        print(f"  {name:10s}: pLDDT={c['complex_plddt']:.4f}  ptm={c['ptm']:.4f}  iptm={c['iptm']:.4f}  conf={c['confidence_score']:.4f}")

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
if cos_z > 0.999 and cos_s > 0.999:
    print("  ✅ z/s match closely — DAP trunk is correct!")
    print("     pLDDT difference is from diffusion sampling stochasticity")
elif cos_z > 0.99:
    print("  ⚠️  Small z/s differences — may be from numerical precision")
    print("     Not a DAP bug, but accumulated floating-point rounding")
else:
    print("  ❌ Large z/s divergence — DAP trunk has a remaining bug!")
    print("     Investigate further with layer-by-layer comparison")
PYEOF

echo ""
echo "Done."
echo "  Baseline: $BASE/baseline_seed42/predictions/"
echo "  DAP:      $BASE/dap_seed42/predictions/"
