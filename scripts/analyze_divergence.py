#!/usr/bin/env python
"""
Memory-efficient checkpoint analysis.
Loads tensors one at a time to avoid OOM.
"""
import torch
import torch.nn.functional as F
import gc, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else '/project/engvimmune/gleeai/boltz_output/baseline_ckpt_seed42/trunk_checkpoints.pt'
DAP = sys.argv[2] if len(sys.argv) > 2 else '/project/engvimmune/gleeai/boltz_output/dap_ckpt_seed42/trunk_checkpoints.pt'

def load_one(path, key):
    """Load just one checkpoint from the file."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    result = data[key]
    del data
    gc.collect()
    return result

# 1. Verify R0/after_recycle is identical
print("=" * 70)
print("1. Verifying R0/after_recycle is identical...")
b = load_one(BASE, 'R0/after_recycle')
d = load_one(DAP, 'R0/after_recycle')
zb0 = b['z'].float()
zd0 = d['z'].float()
print(f"   z diff max: {(zb0 - zd0).abs().max():.8f}")
print(f"   s diff max: {(b['s'].float() - d['s'].float()).abs().max():.8f}")
del b, d; gc.collect()

# 2. Analyze template output
print("\n" + "=" * 70)
print("2. Analyzing template output...")
bt = load_one(BASE, 'R0/after_template')
dt = load_one(DAP, 'R0/after_template')
zbt = bt['z'].float()
zdt = dt['z'].float()

tmpl_base = zbt - zb0  # template output in baseline
tmpl_dap = zdt - zd0   # template output in DAP
tmpl_diff = (tmpl_base - tmpl_dap).abs()

print(f"   Template output (base): mean={tmpl_base.mean():.6f}, std={tmpl_base.std():.6f}")
print(f"   Template output (DAP):  mean={tmpl_dap.mean():.6f}, std={tmpl_dap.std():.6f}")
print(f"   Diff: mean={tmpl_diff.mean():.6f}, max={tmpl_diff.max():.4f}")
cos = F.cosine_similarity(tmpl_base.flatten().unsqueeze(0), tmpl_dap.flatten().unsqueeze(0)).item()
print(f"   Cosine similarity: {cos:.6f}")
print(f"   Nonzero (base): {(tmpl_base.abs() > 1e-6).sum().item()} / {tmpl_base.numel()}")
print(f"   Nonzero (DAP):  {(tmpl_dap.abs() > 1e-6).sum().item()} / {tmpl_dap.numel()}")

# Check if one is a scaled version of the other
if tmpl_base.abs().max() > 0 and tmpl_dap.abs().max() > 0:
    ratio = tmpl_base / tmpl_dap.clamp(min=1e-8)
    valid = (tmpl_dap.abs() > 0.01) & (tmpl_base.abs() > 0.01)
    if valid.sum() > 100:
        ratios = ratio[valid]
        print(f"   Ratio (base/dap) where both >0.01: mean={ratios.mean():.4f}, std={ratios.std():.4f}")

del zb0, zd0, tmpl_base, tmpl_dap, tmpl_diff; gc.collect()

# 3. Check MSA residual hypothesis
print("\n" + "=" * 70)
print("3. Checking MSA outer residual hypothesis...")
print("   baseline does: z = z + msa_module(z)")
print("   DAP might do:  z = _run_msa_dap(z)  (missing outer +z)")

bm = load_one(BASE, 'R0/after_msa')
dm = load_one(DAP, 'R0/after_msa')
zbm = bm['z'].float()
zdm = dm['z'].float()
del bm, dm; gc.collect()

# If DAP is missing the outer residual, then:
#   baseline_after_msa = z_before + msa_out = z_before + (z_before + internal_deltas)
#                      = 2 * z_before + internal_deltas
#   dap_after_msa = z_before + internal_deltas (only internal residuals from msa layers)
#   So: baseline - dap = z_before
residual = zbm - zdm
print(f"   (base_msa - dap_msa): mean={residual.mean():.6f}, std={residual.std():.4f}")
print(f"   z_before_msa (base): mean={zbt.mean():.6f}, std={zbt.std():.4f}")
cos_res = F.cosine_similarity(residual.flatten().unsqueeze(0), zbt.flatten().unsqueeze(0)).item()
print(f"   Cosine similarity (base-dap) vs z_before: {cos_res:.6f}")
diff_res = (residual - zbt).abs()
print(f"   Mean abs diff: {diff_res.mean():.6f}, max: {diff_res.max():.4f}")
print(f"   ** If close to 0, DAP is missing z = z + msa(z) **")

# Also try with DAP's z_before_msa
print(f"\n   Using DAP z_before (to account for template diff):")
print(f"   z_before_msa (dap): mean={zdt.mean():.6f}, std={zdt.std():.4f}")
cos_res2 = F.cosine_similarity(residual.flatten().unsqueeze(0), zdt.flatten().unsqueeze(0)).item()
print(f"   Cosine similarity (base-dap) vs dap_z_before: {cos_res2:.6f}")
diff_res2 = (residual - zdt).abs()
print(f"   Mean abs diff: {diff_res2.mean():.6f}, max: {diff_res2.max():.4f}")

# Check: baseline_msa = dap_msa + z_before_msa_dap?
check = zbm - (zdm + zdt)
print(f"\n   baseline_msa - (dap_msa + dap_z_before):")
print(f"   Mean: {check.mean():.6f}, max: {check.abs().max():.4f}")
print(f"   ** If ~0, both template diff + missing residual explain everything **")

print("\n" + "=" * 70)
print("DONE")
