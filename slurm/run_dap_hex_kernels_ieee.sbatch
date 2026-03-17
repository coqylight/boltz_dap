#!/bin/bash
#SBATCH --job-name=dap_hex_ie
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/dap_hex_kernels_ieee_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export BOLTZ_KERNEL_PRECISION=ieee

echo "=== DAP-2 Hexamer: Kernels IEEE f32 ==="
srun uv run --project /project/engvimmune/gleeai/boltz torchrun --nproc_per_node=2 --master_port=29502 \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    /project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml \
    --out_dir /project/engvimmune/gleeai/boltz_output/dap_hex_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1

echo "DONE"
