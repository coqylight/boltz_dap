#!/usr/bin/env python3
"""Per-chain CA RMSD comparison (proper chain IDs: A, B, C, D, E)."""

import numpy as np
from scipy.spatial.distance import cdist

BASE = "/project/engvimmune/gleeai/boltz_output"
DIR_SDPA = f"{BASE}/pentamer_from_tetramer_fa3_test2/predictions/1LP3_pentamer_from_tetramer"
DIR_FLEX = f"{BASE}/pentamer_flexattn_test/predictions/1LP3_pentamer_from_tetramer"

def parse_ca_by_chain(cif_path):
    """Extract CA atoms grouped by chain ID (column 10 = auth_asym_id)."""
    chains = {}
    with open(cif_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            if parts[3] != "CA":
                continue
            chain = parts[9]   # auth_asym_id (A, B, C, D, E)
            x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
            plddt = float(parts[17])
            if chain not in chains:
                chains[chain] = {"coords": [], "plddt": []}
            chains[chain]["coords"].append([x, y, z])
            chains[chain]["plddt"].append(plddt)
    for c in chains:
        chains[c]["coords"] = np.array(chains[c]["coords"])
        chains[c]["plddt"] = np.array(chains[c]["plddt"])
    return chains

def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid body superposition (no reflections)."""
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.eye(3)
    sign_mat[2, 2] = np.sign(d)
    R = Vt.T @ sign_mat @ U.T
    P_rot = P @ R.T
    return np.sqrt(((P_rot - Q) ** 2).sum(axis=1).mean())

# ──────────────────────────────────────────
print("Parsing CIF files...")
sdpa = parse_ca_by_chain(f"{DIR_SDPA}/1LP3_pentamer_from_tetramer_model_0.cif")
flex = parse_ca_by_chain(f"{DIR_FLEX}/1LP3_pentamer_from_tetramer_model_0.cif")

print(f"\nSDPA chains: {sorted(sdpa.keys())}, CA per chain: {[len(sdpa[c]['coords']) for c in sorted(sdpa)]}")
print(f"Flex chains: {sorted(flex.keys())}, CA per chain: {[len(flex[c]['coords']) for c in sorted(flex)]}")

# ──────────────────────────────────────────
# 1. Per-chain CA RMSD (each chain superposed independently)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("1. PER-CHAIN CA RMSD (independently superposed)")
print("=" * 60)

# Template chains: A, B, C, D have template guidance
# Chain E: no template (5th chain, built from scratch)
template_chains = {"A", "B", "C", "D"}

for c in sorted(sdpa.keys()):
    rmsd = kabsch_rmsd(sdpa[c]["coords"], flex[c]["coords"])
    has_template = "TEMPLATE" if c in template_chains else "NO TEMPLATE"
    plddt_s = sdpa[c]["plddt"].mean()
    plddt_f = flex[c]["plddt"].mean()
    print(f"  Chain {c} ({has_template}): RMSD={rmsd:.2f}A | "
          f"pLDDT: SDPA={plddt_s:.1f}, Flex={plddt_f:.1f} (Δ={plddt_f-plddt_s:+.1f})")

# ──────────────────────────────────────────
# 2. Global RMSD (all chains together)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. GLOBAL CA RMSD (all chains superposed together)")
print("=" * 60)

all_sdpa = np.vstack([sdpa[c]["coords"] for c in sorted(sdpa)])
all_flex = np.vstack([flex[c]["coords"] for c in sorted(flex)])
global_rmsd = kabsch_rmsd(all_sdpa, all_flex)
print(f"  Global RMSD: {global_rmsd:.2f} A")

# ──────────────────────────────────────────
# 3. Template-only RMSD (A-D, should be more similar)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TEMPLATE CHAINS ONLY (A-D) CA RMSD")
print("=" * 60)

tmpl_sdpa = np.vstack([sdpa[c]["coords"] for c in sorted(template_chains)])
tmpl_flex = np.vstack([flex[c]["coords"] for c in sorted(template_chains)])
tmpl_rmsd = kabsch_rmsd(tmpl_sdpa, tmpl_flex)
print(f"  Template chains (A-D) RMSD: {tmpl_rmsd:.2f} A")

# ──────────────────────────────────────────
# 4. Intra-chain distance consistency  
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. INTRA-CHAIN CA DISTANCE CONSISTENCY")
print("=" * 60)
print("  (Same fold = same internal distances, regardless of orientation)")

for c in sorted(sdpa.keys()):
    dm_s = cdist(sdpa[c]["coords"], sdpa[c]["coords"])
    dm_f = cdist(flex[c]["coords"], flex[c]["coords"])
    diff = np.abs(dm_s - dm_f)
    has_template = "TEMPLATE" if c in template_chains else "NO TEMPLATE"
    print(f"  Chain {c} ({has_template}): "
          f"mean_diff={diff.mean():.2f}A, max_diff={diff.max():.2f}A")

# ──────────────────────────────────────────
# 5. Inter-chain distance (chain center-of-mass)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CHAIN CENTER-OF-MASS DISTANCES")
print("=" * 60)

for label, data in [("SDPA", sdpa), ("Flex", flex)]:
    coms = {}
    for c in sorted(data.keys()):
        coms[c] = data[c]["coords"].mean(axis=0)
    print(f"\n  {label}:")
    chains_list = sorted(coms.keys())
    for i, c1 in enumerate(chains_list):
        for j, c2 in enumerate(chains_list):
            if j <= i:
                continue
            d = np.linalg.norm(coms[c1] - coms[c2])
            print(f"    {c1}-{c2}: {d:.1f} A")

# ──────────────────────────────────────────
# 6. Quality comparison summary
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("6. QUALITY METRICS COMPARISON")
print("=" * 60)

plddt_sdpa = np.load(f"{DIR_SDPA}/plddt_1LP3_pentamer_from_tetramer_model_0.npz")
plddt_flex = np.load(f"{DIR_FLEX}/plddt_1LP3_pentamer_from_tetramer_model_0.npz")
pae_sdpa = np.load(f"{DIR_SDPA}/pae_1LP3_pentamer_from_tetramer_model_0.npz")
pae_flex = np.load(f"{DIR_FLEX}/pae_1LP3_pentamer_from_tetramer_model_0.npz")
pde_sdpa = np.load(f"{DIR_SDPA}/pde_1LP3_pentamer_from_tetramer_model_0.npz")
pde_flex = np.load(f"{DIR_FLEX}/pde_1LP3_pentamer_from_tetramer_model_0.npz")

print(f"  pLDDT: SDPA={plddt_sdpa['plddt'].mean():.4f}, "
      f"Flex={plddt_flex['plddt'].mean():.4f} "
      f"(higher=better, Flex {'better' if plddt_flex['plddt'].mean() > plddt_sdpa['plddt'].mean() else 'worse'})")
print(f"  PAE:   SDPA={pae_sdpa['pae'].mean():.4f}, "
      f"Flex={pae_flex['pae'].mean():.4f} "
      f"(lower=better, Flex {'better' if pae_flex['pae'].mean() < pae_sdpa['pae'].mean() else 'worse'})")
print(f"  PDE:   SDPA={pde_sdpa['pde'].mean():.4f}, "
      f"Flex={pde_flex['pde'].mean():.4f} "
      f"(lower=better, Flex {'better' if pde_flex['pde'].mean() < pde_sdpa['pde'].mean() else 'worse'})")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
