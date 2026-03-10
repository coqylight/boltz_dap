#!/usr/bin/env python3
"""Compare checkpoint files key-by-key. Handles both formats:
  - Trunk checkpoints: key -> {"z": tensor, "s": tensor}
  - Granular checkpoints: key -> tensor (bare z tensor, no s)
"""
import sys
import torch

base_path = sys.argv[1]
dap_path = sys.argv[2]

print(f"Baseline: {base_path}")
print(f"DAP:      {dap_path}")

base = torch.load(base_path, map_location='cpu', weights_only=False)
dap = torch.load(dap_path, map_location='cpu', weights_only=False)

base_keys = sorted(base.keys())
dap_keys = sorted(dap.keys())

print(f"\nBaseline keys ({len(base_keys)}): {base_keys}")
print(f"DAP keys      ({len(dap_keys)}):  {dap_keys}")

common = sorted(set(base_keys) & set(dap_keys))
only_base = sorted(set(base_keys) - set(dap_keys))
only_dap = sorted(set(dap_keys) - set(base_keys))
print(f"Common keys   ({len(common)}):  {common}")
if only_base:
    print(f"Only in baseline: {only_base}")
if only_dap:
    print(f"Only in DAP:      {only_dap}")

# Detect format
sample = base[common[0]] if common else None
is_dict_format = isinstance(sample, dict) and "z" in sample

if is_dict_format:
    # Trunk checkpoint format: {"z": tensor, "s": tensor}
    print(f"\nFormat: trunk (z + s per key)")
    print(f"\n{'Key':<28}|  {'z_mean_diff':>12} {'z_max_diff':>12} {'z_cosine':>8} {'z_base_mean':>12} {'z_dap_mean':>12} |  {'s_mean_diff':>12} {'s_max_diff':>12} {'s_cosine':>8} {'s_base_mean':>12} {'s_dap_mean':>12} {'':>3}")
    print("-" * 180)

    z_divergence = False
    s_divergence = False

    for key in common:
        b, d = base[key], dap[key]
        zb, zd = b["z"].float(), d["z"].float()
        N = min(zb.shape[1], zd.shape[1])
        zb, zd = zb[:, :N, :N], zd[:, :N, :N]
        z_diff = (zb - zd).abs()
        z_cos = torch.nn.functional.cosine_similarity(zb.flatten(), zd.flatten(), dim=0).item()

        sb, sd = b["s"].float(), d["s"].float()
        Ns = min(sb.shape[1], sd.shape[1])
        sb, sd = sb[:, :Ns], sd[:, :Ns]
        s_diff = (sb - sd).abs()
        s_cos = torch.nn.functional.cosine_similarity(sb.flatten(), sd.flatten(), dim=0).item()

        status = "✅" if z_diff.max() <= 2.0 and s_diff.max() <= 1.0 else "❌"
        if z_diff.max() > 10: z_divergence = True
        if s_diff.max() > 10: s_divergence = True
        print(f"{key:<28}|  {z_diff.mean():>12.6f} {z_diff.max():>12.4f} {z_cos:>8.6f} {zb.mean():>12.6f} {zd.mean():>12.6f} |  {s_diff.mean():>12.6f} {s_diff.max():>12.4f} {s_cos:>8.6f} {sb.mean():>12.6f} {sd.mean():>12.6f} {status}")

    print()
    if z_divergence: print("⚠️  z DIVERGENCE detected (max diff > 10)")
    if s_divergence: print("⚠️  s DIVERGENCE detected (max diff > 10)")
    if not z_divergence and not s_divergence: print("✅ All checkpoints match within tolerance!")

else:
    # Granular checkpoint format: key -> tensor (bare tensor)
    print(f"\nFormat: granular (bare tensors)")
    print(f"\n{'Key':<36}| {'shape':>24} | {'mean_diff':>12} {'max_diff':>12} {'cosine':>10} | {'base_mean':>12} {'dap_mean':>12} | {'base_std':>10} {'dap_std':>10} {'':>3}")
    print("-" * 180)

    divergence = False
    for key in common:
        bv = base[key]
        dv = dap[key]

        # Handle both tensor and non-tensor values
        if not isinstance(bv, torch.Tensor) or not isinstance(dv, torch.Tensor):
            print(f"{key:<36}| {'(non-tensor)':>24} | skipped")
            continue

        bf, df = bv.float(), dv.float()

        # Trim to common size if needed
        if bf.dim() >= 2 and bf.shape != df.shape:
            mins = [min(a, b) for a, b in zip(bf.shape, df.shape)]
            slices = tuple(slice(0, m) for m in mins)
            bf, df = bf[slices], df[slices]

        diff = (bf - df).abs()
        cos = torch.nn.functional.cosine_similarity(bf.flatten(), df.flatten(), dim=0).item()

        status = "✅" if diff.max() <= 2.0 else "❌"
        if diff.max() > 10: divergence = True

        shape_str = str(list(bf.shape))
        print(f"{key:<36}| {shape_str:>24} | {diff.mean():>12.6f} {diff.max():>12.4f} {cos:>10.6f} | {bf.mean():>12.6f} {df.mean():>12.6f} | {bf.std():>10.4f} {df.std():>10.4f} {status}")

    print()
    if divergence: print("⚠️  DIVERGENCE detected (max diff > 10)")
    else: print("✅ All granular checkpoints match within tolerance!")
