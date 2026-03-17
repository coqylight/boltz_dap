#!/bin/bash
#SBATCH --job-name=analyze_div
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/analyze_divergence_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/analyze_divergence_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

DAP_DIR=/project/engvimmune/gleeai/boltz_dap
cd "$DAP_DIR"
uv run --project /project/engvimmune/gleeai/boltz python "${DAP_DIR}/scripts/analyze_divergence.py"
