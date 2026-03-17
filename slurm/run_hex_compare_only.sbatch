#!/bin/bash
#SBATCH --job-name=hex-compare
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/hex_compare_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/hex_compare_%j.log
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

source /project/engvimmune/gleeai/boltz/.venv/bin/activate

python3 - << 'PYEOF'
import torch, gc, os, glob

orig_base = '/project/engvimmune/gleeai/boltz_output/orig_hex_trunk_ckpts'
dap_base = '/project/engvimmune/gleeai/boltz_output'
v2_ckpt = '/project/engvimmune/gleeai/boltz_output/v2_hex_recycle_ckpts'

# ======= 1. Original vs DAP (same config, R3) =======
print('='*70)
print('  HEXAMER: Original vs DAP — Same Config (R3)')
print('='*70)
print(f'{"Comparison":>35} | {"z_mean":>10} | {"z_max":>10} | {"s_mean":>10} | {"s_max":>10}')
print('-'*85)

pairs = [
    ('no_kernels', 'dap_hex_no_kernels', 'No-Kernels'),
    ('kernels_tf32', 'dap_hex_kernels_tf32', 'Kernels-TF32'),
    ('kernels_ieee', 'dap_hex_kernels_ieee', 'Kernels-IEEE'),
]

for orig_cfg, dap_dir, label in pairs:
    orig = torch.load(f'{orig_base}/{orig_cfg}.pt', map_location='cpu', weights_only=False)
    dap = torch.load(f'{dap_base}/{dap_dir}/zs_tensors.pt', map_location='cpu', weights_only=False)
    z_orig = orig['R3/after_pairformer/z'].float()
    z_dap = dap['z'].float()
    s_orig = orig['R3/after_pairformer/s'].float()
    s_dap = dap['s'].float()
    z_diff = (z_orig - z_dap).abs()
    s_diff = (s_orig - s_dap).abs()
    print(f'{label + ": Orig vs DAP":>35} | {z_diff.mean():.6f} | {z_diff.max():>10.4f} | {s_diff.mean():.6f} | {s_diff.max():>10.4f}')
    del orig, dap, z_orig, z_dap, s_orig, s_dap, z_diff, s_diff; gc.collect()

# ======= 2. Per-recycling-step (Orig R0-R3 vs DAP final) =======
print()
print('='*70)
print('  HEXAMER: Per-Recycling-Step (Orig Ri vs DAP final z)')
print('='*70)
for orig_cfg, dap_dir, label in pairs:
    orig = torch.load(f'{orig_base}/{orig_cfg}.pt', map_location='cpu', weights_only=False)
    dap = torch.load(f'{dap_base}/{dap_dir}/zs_tensors.pt', map_location='cpu', weights_only=False)
    z_dap = dap['z'].float()
    del dap; gc.collect()
    print(f'  {label}:')
    for r in range(4):
        z_orig = orig[f'R{r}/after_pairformer/z'].float()
        diff = (z_orig - z_dap).abs()
        print(f'    R{r}: z_mean={diff.mean():.6f}  z_max={diff.max():.4f}')
        del z_orig, diff; gc.collect()
    del orig, z_dap; gc.collect()

# ======= 3. Original cross-config (from v2 recycle ckpts) =======
print()
print('='*70)
print('  HEXAMER: Original Configs Cross-Comparison (v2 R3)')
print('='*70)
print(f'{"Comparison":>35} | {"z_mean":>10} | {"z_max":>10}')
print('-'*65)
cfgs = {}
for c in ['orig_nk', 'orig_ieee', 'orig_tf32']:
    files = sorted(glob.glob(f'{v2_ckpt}/{c}/recycle_*.pt'))
    cfgs[c] = files[-1] if files else None

for a, b, label in [('orig_nk','orig_ieee','nk vs ieee'),('orig_nk','orig_tf32','nk vs tf32'),('orig_ieee','orig_tf32','ieee vs tf32')]:
    if cfgs[a] and cfgs[b]:
        za = torch.load(cfgs[a], map_location='cpu', weights_only=True).float()
        zb = torch.load(cfgs[b], map_location='cpu', weights_only=True).float()
        diff = (za - zb).abs()
        print(f'{label:>35} | {diff.mean():.6f} | {diff.max():>10.4f}')
        del za, zb, diff; gc.collect()
    else:
        print(f'{label:>35} | SKIPPED')

# ======= 4. DAP cross-config =======
print()
print('='*70)
print('  HEXAMER: DAP Configs Cross-Comparison (final z)')
print('='*70)
print(f'{"Comparison":>35} | {"z_mean":>10} | {"z_max":>10}')
print('-'*65)

dap_files = {
    'nk': f'{dap_base}/dap_hex_no_kernels/zs_tensors.pt',
    'ieee': f'{dap_base}/dap_hex_kernels_ieee/zs_tensors.pt',
    'tf32': f'{dap_base}/dap_hex_kernels_tf32/zs_tensors.pt',
}

for a, b, label in [('nk','ieee','DAP nk vs ieee'),('nk','tf32','DAP nk vs tf32'),('ieee','tf32','DAP ieee vs tf32')]:
    da = torch.load(dap_files[a], map_location='cpu', weights_only=False)
    db = torch.load(dap_files[b], map_location='cpu', weights_only=False)
    z_diff = (da['z'].float() - db['z'].float()).abs()
    s_diff = (da['s'].float() - db['s'].float()).abs()
    print(f'{label:>35} | {z_diff.mean():.6f} | {z_diff.max():>10.4f}')
    del da, db, z_diff, s_diff; gc.collect()

# ======= 5. Orig per-recycling cross-config =======
print()
print('='*70)
print('  HEXAMER: Original Per-Recycling-Step Cross-Config')
print('='*70)
for r in range(4):
    print(f'  R{r}:')
    for a_cfg, b_cfg, label in [('no_kernels','kernels_ieee','nk vs ieee'),('no_kernels','kernels_tf32','nk vs tf32'),('kernels_ieee','kernels_tf32','ieee vs tf32')]:
        orig_a = torch.load(f'{orig_base}/{a_cfg}.pt', map_location='cpu', weights_only=False)
        orig_b = torch.load(f'{orig_base}/{b_cfg}.pt', map_location='cpu', weights_only=False)
        za = orig_a[f'R{r}/after_pairformer/z'].float()
        zb = orig_b[f'R{r}/after_pairformer/z'].float()
        diff = (za - zb).abs()
        print(f'    {label:>15}: z_mean={diff.mean():.6f}  z_max={diff.max():.4f}')
        del orig_a, orig_b, za, zb, diff; gc.collect()

print('\nDONE')
PYEOF
