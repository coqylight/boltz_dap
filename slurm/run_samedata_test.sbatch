#!/bin/bash
#SBATCH --job-name=dap_samedata
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/dap_samedata_test_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

# Use original Boltz2 predict as baseline, compare z/s with DAP.
# Objectives: (1) same z and s, (2) lower memory, (3) no data duplication.

cd /project/engvimmune/gleeai/boltz_dap
exec > >(tee /project/engvimmune/gleeai/boltz_output/dap_samedata_test.log) 2>&1

bash /project/engvimmune/gleeai/boltz_dap/run_samedata_test.sh
