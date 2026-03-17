#!/bin/bash
# Test: does use_kernels affect z/s in the original Boltz2?
# Run the SAME boltz predict with identical input, seed, and model —
# only difference is --no_kernels flag.
set -e
export PYTHONUNBUFFERED=1

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
SEED=42

echo "=============================================="
echo "RUN 1: boltz predict WITH kernels (default)"
echo "=============================================="
OUT1="$BASE/kernel_test_with"
rm -rf "$OUT1"
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    boltz predict "$YAML" \
    --out_dir "$OUT1" \
    --write_embeddings \
    --seed $SEED \
    --use_msa_server

echo ""
echo "=============================================="
echo "RUN 2: boltz predict WITHOUT kernels (--no_kernels)"
echo "=============================================="
OUT2="$BASE/kernel_test_without"
rm -rf "$OUT2"
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    boltz predict "$YAML" \
    --out_dir "$OUT2" \
    --write_embeddings \
    --no_kernels \
    --seed $SEED \
    --use_msa_server

echo ""
echo "=============================================="
echo "COMPARE: z and s embeddings"
echo "=============================================="

uv run --project "$BOLTZ_PROJECT" python3 - <<'PYEOF'
import numpy as np
import os, glob

base = "/project/engvimmune/gleeai/boltz_output"

# Find embedding files
emb1_files = sorted(glob.glob(f"{base}/kernel_test_with/**/embeddings_*.npz", recursive=True))
emb2_files = sorted(glob.glob(f"{base}/kernel_test_without/**/embeddings_*.npz", recursive=True))

print(f"With kernels:    {len(emb1_files)} embedding files")
print(f"Without kernels: {len(emb2_files)} embedding files")

if not emb1_files or not emb2_files:
    print("ERROR: No embedding files found!")
    exit(1)

e1 = np.load(emb1_files[0])
e2 = np.load(emb2_files[0])

for key in ['z', 's']:
    if key not in e1 or key not in e2:
        print(f"  {key}: missing in one or both")
        continue
    a = e1[key].astype(np.float32)
    b = e2[key].astype(np.float32)
    diff = np.abs(a - b)
    identical = np.array_equal(e1[key], e2[key])
    n_diff = np.sum(diff > 0)
    
    print(f"\n  {key} (shape {a.shape}):")
    print(f"    Bitwise identical: {'✅ YES' if identical else '❌ NO'}")
    print(f"    Elements differing: {n_diff} / {a.size} ({100*n_diff/a.size:.2f}%)")
    print(f"    Mean diff: {diff.mean():.10f}")
    print(f"    Max diff:  {diff.max():.10f}")
    if not identical:
        cos = np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"    Cosine similarity: {cos:.10f}")
PYEOF

echo ""
echo "ALL DONE"
