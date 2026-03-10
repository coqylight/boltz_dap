#!/bin/bash
#SBATCH --job-name=boltz_dap_pentamer
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/pentamer_from_tetramer_fa3_test2/slurm_%j.out
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/pentamer_from_tetramer_fa3_test2/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

# Create output directory
mkdir -p /project/engvimmune/gleeai/boltz_output/pentamer_from_tetramer_fa3_test2

# Activate environment
source /project/engvimmune/gleeai/envs/boltz_uv/bin/activate
export CUDA_HOME=/project/engvimmune/gleeai/envs/boltz
export PATH=$CUDA_HOME/bin:$PATH

# Run DAP inference with detailed logging
cd /project/engvimmune/gleeai
torchrun --nproc_per_node=2 boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    boltz_input/1LP3_pentamer_from_tetramer.yaml \
    --out_dir boltz_output/pentamer_from_tetramer_fa3_test2 \
    --cache ~/.boltz \
    --use_msa_server \
    2>&1 | tee boltz_output/pentamer_from_tetramer_fa3_test2/detailed_log.txt
