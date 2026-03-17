#!/bin/bash
#SBATCH --job-name=dap_diverge
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/dap_diverge_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

# Divergence diagnosis: 1-GPU reference vs 2-GPU DAP, per-sub-op checkpoint comparison.

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /project/engvimmune/gleeai/boltz_dap
exec > >(tee /project/engvimmune/gleeai/boltz_output/dap_diverge.log) 2>&1

bash /project/engvimmune/gleeai/boltz_dap/run_divergence_test.sh
