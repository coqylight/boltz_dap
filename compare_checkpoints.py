#!/usr/bin/env python3
"""Compare z/s tensor checkpoints between Original Boltz2 and DAP.

Usage:
  python compare_checkpoints.py
"""
import torch
import os

TARGETS = ["9JGM", "9J09", "9JFS", "9E74", "9KGG"]
ORIG_BASE = "/project/engvimmune/gleeai/kaggle/diag_original"
DAP_BASE = "/project/engvimmune/gleeai/kaggle/diag_dap"

print(f"{'='*90}")
print(f"  DAP vs Original Boltz2 — Tensor Checkpoint Comparison")
print(f"{'='*90}")

for target in TARGETS:
    orig_path = os.path.join(ORIG_BASE, target, "trunk_checkpoints.pt")
    dap_path = os.path.join(DAP_BASE, target, "trunk_checkpoints.pt")
    
    if not os.path.exists(orig_path):
        print(f"\n[{target}] MISSING original checkpoints: {orig_path}")
        continue
    if not os.path.exists(dap_path):
        print(f"\n[{target}] MISSING DAP checkpoints: {dap_path}")
        continue
    
    orig = torch.load(orig_path, map_location="cpu", weights_only=False)
    dap = torch.load(dap_path, map_location="cpu", weights_only=False)
    
    print(f"\n{'='*90}")
    print(f"  Target: {target}")
    print(f"{'='*90}")
    
    # Separate sub-steps from full pipeline stages
    stages = sorted(set(orig.keys()) & set(dap.keys()))
    sub_stages = [s for s in stages if s.startswith("sub/")]
    full_stages = [s for s in stages if not s.startswith("sub/")]
    
    # --- Sub-step comparison (z_init breakdown) ---
    if sub_stages:
        print(f"\n  --- z_init Sub-Step Breakdown ---")
        print(f"  {'Stage':<30s}  {'Tensor':>8s}  {'RMSD':>10s}  {'MaxDiff':>10s}  {'RelErr':>10s}  {'Match?':>8s}")
        print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        
        for stage in sub_stages:
            o = orig[stage]
            d = dap[stage]
            
            # Check s (might be s_inputs or s_init)
            os_ = o["s"].float()
            ds = d["s"].float()
            if os_.shape == ds.shape and os_.numel() > 1:
                s_diff = (os_ - ds)
                s_rmsd = s_diff.pow(2).mean().sqrt().item()
                s_max = s_diff.abs().max().item()
                s_rel = s_diff.abs().mean().item() / (os_.abs().mean().item() + 1e-10)
                s_match = "OK" if s_rmsd < 0.001 else "DIFF"
                print(f"  {stage:<30s}  {'s':>8s}  {s_rmsd:>10.6f}  {s_max:>10.4f}  {s_rel:>10.8f}  {s_match:>8s}")
            
            # Check z (might be placeholder)
            oz = o["z"].float()
            dz = d["z"].float()
            if oz.numel() > 1 and dz.numel() > 1:
                if oz.shape != dz.shape:
                    min_n = min(oz.shape[1], dz.shape[1])
                    oz = oz[:, :min_n, :min_n, :]
                    dz = dz[:, :min_n, :min_n, :]
                z_diff = (oz - dz)
                z_rmsd = z_diff.pow(2).mean().sqrt().item()
                z_max = z_diff.abs().max().item()
                z_rel = z_diff.abs().mean().item() / (oz.abs().mean().item() + 1e-10)
                z_match = "OK" if z_rmsd < 0.001 else "DIFF"
                print(f"  {stage:<30s}  {'z':>8s}  {z_rmsd:>10.6f}  {z_max:>10.4f}  {z_rel:>10.8f}  {z_match:>8s}")
    
    # --- Full pipeline comparison ---
    if full_stages:
        print(f"\n  --- Full Pipeline Stages ---")
        print(f"  {'Stage':<25s}  {'z RMSD':>10s}  {'z MaxDiff':>10s}  {'z RelErr':>10s}  "
              f"{'s RMSD':>10s}  {'s MaxDiff':>10s}  {'s RelErr':>10s}  {'Match?':>8s}")
        print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        
        for stage in full_stages:
            oz = orig[stage]["z"].float()
            dz = dap[stage]["z"].float()
            os_ = orig[stage]["s"].float()
            ds = dap[stage]["s"].float()
            
            if oz.shape != dz.shape:
                min_n = min(oz.shape[1], dz.shape[1])
                oz = oz[:, :min_n, :min_n, :]
                dz = dz[:, :min_n, :min_n, :]
            if os_.shape != ds.shape:
                min_n = min(os_.shape[1], ds.shape[1])
                os_ = os_[:, :min_n, :]
                ds = ds[:, :min_n, :]
            
            z_diff = (oz - dz)
            z_rmsd = z_diff.pow(2).mean().sqrt().item()
            z_max = z_diff.abs().max().item()
            z_rel = z_diff.abs().mean().item() / (oz.abs().mean().item() + 1e-10)
            
            s_diff = (os_ - ds)
            s_rmsd = s_diff.pow(2).mean().sqrt().item()
            s_max = s_diff.abs().max().item()
            s_rel = s_diff.abs().mean().item() / (os_.abs().mean().item() + 1e-10)
            
            match = "OK" if z_rmsd < 0.01 and s_rmsd < 0.01 else "DIFF"
            
            print(f"  {stage:<25s}  {z_rmsd:>10.6f}  {z_max:>10.4f}  {z_rel:>10.6f}  "
                  f"{s_rmsd:>10.6f}  {s_max:>10.4f}  {s_rel:>10.6f}  {match:>8s}")

print(f"\n{'='*90}")
print("  Sub-step match threshold: RMSD < 0.001 (stricter)")
print("  Full pipeline match threshold: RMSD < 0.01 for both z and s")
print(f"{'='*90}")
