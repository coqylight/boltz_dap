#!/bin/bash
#SBATCH --job-name=orig_cd
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/orig_cd_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
OUT_C="/project/engvimmune/gleeai/boltz_output/orig_kernels_default"
OUT_D="/project/engvimmune/gleeai/boltz_output/orig_kernels_ieee"

echo "=============================================="
echo "Original Boltz2 — Runs C & D (kernels only)"
echo "Run B (no kernels) already completed at orig_no_kernels/"
echo "=============================================="

# -----------------------------------------------
# C: Original Boltz2, kernels with DEFAULT precision (TF32)
#    BOLTZ_KERNEL_PRECISION=default → precision=None → uses TF32
# -----------------------------------------------
echo ""
echo "=============================================="
echo "RUN C: Kernels with DEFAULT precision (TF32)"
echo "Output: $OUT_C"
echo "=============================================="

BOLTZ_KERNEL_PRECISION=default uv run --project /project/engvimmune/gleeai/boltz \
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
#    BOLTZ_KERNEL_PRECISION=ieee → precision="ieee"
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
# Compare all 3 CIF files (B is already saved)
# -----------------------------------------------
echo ""
echo "=============================================="
echo "Comparing CIF files"
echo "=============================================="

CIF_B="/project/engvimmune/gleeai/boltz_output/orig_no_kernels/boltz_results_1LP3_pentamer/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"
CIF_C="$OUT_C/boltz_results_1LP3_pentamer/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"
CIF_D="$OUT_D/boltz_results_1LP3_pentamer/predictions/1LP3_pentamer/1LP3_pentamer_model_0.cif"

echo ""
echo "B vs C (no kernels vs kernels default/TF32):"
if diff -q "$CIF_B" "$CIF_C" > /dev/null 2>&1; then echo "  IDENTICAL"; else echo "  $(diff "$CIF_B" "$CIF_C" | wc -l) lines differ"; fi

echo ""
echo "B vs D (no kernels vs kernels IEEE):"
if diff -q "$CIF_B" "$CIF_D" > /dev/null 2>&1; then echo "  IDENTICAL"; else echo "  $(diff "$CIF_B" "$CIF_D" | wc -l) lines differ"; fi

echo ""
echo "C vs D (kernels default/TF32 vs kernels IEEE):"
if diff -q "$CIF_C" "$CIF_D" > /dev/null 2>&1; then echo "  IDENTICAL"; else echo "  $(diff "$CIF_C" "$CIF_D" | wc -l) lines differ"; fi

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
