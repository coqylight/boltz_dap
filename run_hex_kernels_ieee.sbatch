#!/bin/bash
#SBATCH --job-name=hex_ieee
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/hex_kernels_ieee_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOLTZ_KERNEL_PRECISION=ieee

echo "=== Hexamer: Original Boltz2, Kernels IEEE f32 ==="
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict /project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml \
    --out_dir /project/engvimmune/gleeai/boltz_output/hex_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override

echo "DONE"
