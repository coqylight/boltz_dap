#!/bin/bash
#SBATCH --job-name=subop_diag
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/subop_diag_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

cd /project/engvimmune/gleeai/boltz_dap
exec > >(tee /project/engvimmune/gleeai/boltz_output/subop_diag.log) 2>&1
bash /project/engvimmune/gleeai/boltz_dap/run_subop_test.sh
