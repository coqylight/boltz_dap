#!/bin/bash
#SBATCH --job-name=oom_profile
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/oom_profile_%j.log

# ============================================================================
# Profile original Boltz2 OOM – pentamer + hexamer on single GPU
# boltz2.py is ALREADY patched with _lg() / _dump_oom() via direct edits.
# After profiling, restore with: git checkout boltz/src/boltz/model/models/boltz2.py
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BOLTZ="/project/engvimmune/gleeai/boltz"
OUT="/project/engvimmune/gleeai/boltz_output"

echo "=============================================="
echo "RUN 1: PENTAMER (N=2595)"
echo "=============================================="

uv run --project "$BOLTZ" \
    boltz predict /project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml \
    --out_dir "$OUT/oom_profile_pentamer" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 || true

echo ""
echo "=============================================="
echo "PENTAMER COMPLETE"
echo "=============================================="

echo ""
echo "=============================================="
echo "RUN 2: HEXAMER (N=3114)"
echo "=============================================="

uv run --project "$BOLTZ" \
    boltz predict /project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml \
    --out_dir "$OUT/oom_profile_hexamer" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 || true

echo ""
echo "=============================================="
echo "HEXAMER COMPLETE"
echo "=============================================="

echo ""
echo "ALL DONE — restore with: git checkout boltz/src/boltz/model/models/boltz2.py"
