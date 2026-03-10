#!/bin/bash
#SBATCH --job-name=kernel_test
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/kernel_test_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

cd /project/engvimmune/gleeai/boltz_dap
exec > >(tee /project/engvimmune/gleeai/boltz_output/kernel_test.log) 2>&1
bash /project/engvimmune/gleeai/boltz_dap/run_kernel_test.sh
