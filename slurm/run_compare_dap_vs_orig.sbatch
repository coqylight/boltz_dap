#!/bin/bash
#SBATCH --job-name=cmp_dap_vs
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_dap_vs_orig_%j.log

export PYTHONUNBUFFERED=1

echo "=============================================="
echo "Comparing DAP hexamer vs Original Boltz2 hexamer z/s"
echo "=============================================="

uv run --project /project/engvimmune/gleeai/boltz python3 -c "
import torch, gc

orig_dir = '/project/engvimmune/gleeai/boltz_output/orig_hex_trunk_ckpts'
dap_map = {
    'DAP_no_kernels': '/project/engvimmune/gleeai/boltz_output/dap_hex_no_kernels/trunk_checkpoints.pt',
    'DAP_kernels_tf32': '/project/engvimmune/gleeai/boltz_output/dap_hex_kernels_tf32/trunk_checkpoints.pt',
    'DAP_kernels_ieee': '/project/engvimmune/gleeai/boltz_output/dap_hex_kernels_ieee/trunk_checkpoints.pt',
}
orig_map = {
    'Orig_no_kernels': f'{orig_dir}/no_kernels.pt',
    'Orig_kernels_tf32': f'{orig_dir}/kernels_tf32.pt',
    'Orig_kernels_ieee': f'{orig_dir}/kernels_ieee.pt',
}

def extract_zs(data, key_prefix):
    \"\"\"Extract z and s tensors from either DAP or Orig format.\"\"\"
    # DAP format: data['R3/after_pairformer'] = {'z': tensor, 's': tensor}
    # Orig format: data['R3/after_pairformer/z'] = tensor, data['R3/after_pairformer/s'] = tensor
    z_key = f'{key_prefix}/z'
    s_key = f'{key_prefix}/s'
    if z_key in data:
        return data[z_key], data[s_key]
    elif key_prefix in data:
        val = data[key_prefix]
        if isinstance(val, dict):
            return val['z'], val['s']
    return None, None

# 1. DAP vs Orig (same kernel config)
print('=== DAP vs Original Boltz2 (same kernel config) ===')
pairs = [
    ('DAP_no_kernels', 'Orig_no_kernels'),
    ('DAP_kernels_tf32', 'Orig_kernels_tf32'),
    ('DAP_kernels_ieee', 'Orig_kernels_ieee'),
]
for dap_name, orig_name in pairs:
    print(f'\n{dap_name} vs {orig_name}:')
    dap = torch.load(dap_map[dap_name], weights_only=True, map_location='cpu')
    orig = torch.load(orig_map[orig_name], weights_only=True, map_location='cpu')
    for rn in ['R0', 'R3']:
        z_dap, s_dap = extract_zs(dap, f'{rn}/after_pairformer')
        z_orig, s_orig = extract_zs(orig, f'{rn}/after_pairformer')
        if z_dap is None or z_orig is None:
            print(f'  {rn}: could not extract z/s')
            continue
        z_dap, z_orig = z_dap.float(), z_orig.float()
        s_dap, s_orig = s_dap.float(), s_orig.float()
        if z_dap.shape != z_orig.shape:
            print(f'  {rn}/z: SHAPE MISMATCH dap={z_dap.shape} orig={z_orig.shape}')
        else:
            zd = (z_dap - z_orig).abs()
            sd = (s_dap - s_orig).abs()
            print(f'  {rn}/z: identical={torch.equal(z_dap, z_orig)} mean={zd.mean():.8f} max={zd.max():.8f}')
            print(f'  {rn}/s: identical={torch.equal(s_dap, s_orig)} mean={sd.mean():.8f} max={sd.max():.8f}')
    del dap, orig; gc.collect()

# 2. DAP configs vs each other (R3 only)
print()
print('=== DAP configs vs each other (R3) ===')
dap_pairs = [
    ('DAP_no_kernels', 'DAP_kernels_tf32'),
    ('DAP_no_kernels', 'DAP_kernels_ieee'),
    ('DAP_kernels_tf32', 'DAP_kernels_ieee'),
]
for name_a, name_b in dap_pairs:
    print(f'\n{name_a} vs {name_b}:')
    a = torch.load(dap_map[name_a], weights_only=True, map_location='cpu')
    b = torch.load(dap_map[name_b], weights_only=True, map_location='cpu')
    z_a, s_a = extract_zs(a, 'R3/after_pairformer')
    z_b, s_b = extract_zs(b, 'R3/after_pairformer')
    if z_a is not None and z_b is not None:
        z_a, z_b = z_a.float(), z_b.float()
        s_a, s_b = s_a.float(), s_b.float()
        if z_a.shape != z_b.shape:
            print(f'  z: SHAPE MISMATCH')
        else:
            zd = (z_a - z_b).abs()
            sd = (s_a - s_b).abs()
            print(f'  z: identical={torch.equal(z_a, z_b)} mean={zd.mean():.8f} max={zd.max():.8f}')
            print(f'  s: identical={torch.equal(s_a, s_b)} mean={sd.mean():.8f} max={sd.max():.8f}')
    del a, b; gc.collect()

print('\nALL DONE')
"

echo "=============================================="
echo "COMPLETE"
echo "=============================================="
