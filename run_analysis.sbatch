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

cd /project/engvimmune/gleeai/boltz_dap
uv run --project /project/engvimmune/gleeai/boltz python analyze_divergence.py
