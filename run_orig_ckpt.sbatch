#!/bin/bash
#SBATCH --job-name=orig_ckpt
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/orig_ckpt_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
CKPT_DIR="/project/engvimmune/gleeai/boltz_output/orig_trunk_ckpts"
mkdir -p "$CKPT_DIR"

echo "=============================================="
echo "Original Boltz2 — 3 configs with z/s checkpointing"
echo "=============================================="

# -----------------------------------------------
# B: No kernels
# -----------------------------------------------
echo ""
echo "=== RUN B: No kernels ==="
BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/no_kernels.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_ckpt_no_kernels \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --no_kernels \
    --override
echo "  ✓ Run B complete"

# -----------------------------------------------
# C: Kernels DEFAULT (TF32)
# -----------------------------------------------
echo ""
echo "=== RUN C: Kernels DEFAULT (TF32) ==="
BOLTZ_KERNEL_PRECISION=default BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/kernels_tf32.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_ckpt_kernels_tf32 \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override
echo "  ✓ Run C complete"

# -----------------------------------------------
# D: Kernels IEEE f32
# -----------------------------------------------
echo ""
echo "=== RUN D: Kernels IEEE f32 ==="
BOLTZ_KERNEL_PRECISION=ieee BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/kernels_ieee.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_ckpt_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override
echo "  ✓ Run D complete"

# -----------------------------------------------
# Compare z/s across configs
# -----------------------------------------------
echo ""
echo "=============================================="
echo "Comparing trunk z/s checkpoints"
echo "=============================================="
python3 -c "
import torch

files = {
    'B_no_kernels': '$CKPT_DIR/no_kernels.pt',
    'C_kernels_tf32': '$CKPT_DIR/kernels_tf32.pt',
    'D_kernels_ieee': '$CKPT_DIR/kernels_ieee.pt',
}

data = {k: torch.load(v, weights_only=True) for k, v in files.items()}

# Compare final recycling step (R3)
pairs = [('B_no_kernels', 'C_kernels_tf32'), ('B_no_kernels', 'D_kernels_ieee'), ('C_kernels_tf32', 'D_kernels_ieee')]
for a, b in pairs:
    print(f'\n{a} vs {b}:')
    for tensor_name in ['R3/after_pairformer/z', 'R3/after_pairformer/s']:
        t_a = data[a][tensor_name].float()
        t_b = data[b][tensor_name].float()
        diff = (t_a - t_b).abs()
        match = torch.equal(t_a, t_b)
        short = tensor_name.split('/')[-1]
        print(f'  {short}: identical={match} mean_diff={diff.mean():.8f} max_diff={diff.max():.8f}')

print('\nDONE')
"
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
