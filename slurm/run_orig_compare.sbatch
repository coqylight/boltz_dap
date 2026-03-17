#!/bin/bash
#SBATCH --job-name=orig_cmp
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/orig_compare_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
OUT_B="/project/engvimmune/gleeai/boltz_output/orig_no_kernels"
OUT_C="/project/engvimmune/gleeai/boltz_output/orig_kernels_tf32"
OUT_D="/project/engvimmune/gleeai/boltz_output/orig_kernels_ieee"

echo "=============================================="
echo "Original Boltz2 — 3 configurations"
echo "Input: $INPUT"
echo "=============================================="

# -----------------------------------------------
# B: Original Boltz2, NO kernels
# -----------------------------------------------
echo ""
echo "=============================================="
echo "RUN B: No kernels (PyTorch native)"
echo "Output: $OUT_B"
echo "=============================================="

uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir "$OUT_B" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --no_kernels \
    --override

echo "  ✓ Run B complete"

# -----------------------------------------------
# C: Original Boltz2, kernels with TF32 precision
# -----------------------------------------------
echo ""
echo "=============================================="
echo "RUN C: Kernels with TF32 precision"
echo "Output: $OUT_C"
echo "=============================================="

BOLTZ_KERNEL_PRECISION=tf32 uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir "$OUT_C" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override

echo "  ✓ Run C complete"

# -----------------------------------------------
# D: Original Boltz2, kernels with IEEE f32 precision
# -----------------------------------------------
echo ""
echo "=============================================="
echo "RUN D: Kernels with IEEE f32 precision"
echo "Output: $OUT_D"
echo "=============================================="

BOLTZ_KERNEL_PRECISION=ieee uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir "$OUT_D" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override

echo "  ✓ Run D complete"

# -----------------------------------------------
# Compare CIF files
# -----------------------------------------------
echo ""
echo "=============================================="
echo "Comparing CIF files"
echo "=============================================="

CIF_B="$OUT_B/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"
CIF_C="$OUT_C/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"
CIF_D="$OUT_D/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"

echo ""
echo "B vs C (no kernels vs kernels TF32):"
if diff -q "$CIF_B" "$CIF_C" > /dev/null 2>&1; then echo "  IDENTICAL"; else diff "$CIF_B" "$CIF_C" | wc -l; echo "  lines differ"; fi

echo ""
echo "B vs D (no kernels vs kernels IEEE):"
if diff -q "$CIF_B" "$CIF_D" > /dev/null 2>&1; then echo "  IDENTICAL"; else diff "$CIF_B" "$CIF_D" | wc -l; echo "  lines differ"; fi

echo ""
echo "C vs D (kernels TF32 vs kernels IEEE):"
if diff -q "$CIF_C" "$CIF_D" > /dev/null 2>&1; then echo "  IDENTICAL"; else diff "$CIF_C" "$CIF_D" | wc -l; echo "  lines differ"; fi

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
