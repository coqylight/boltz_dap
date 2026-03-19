#!/usr/bin/env python3
"""Summarize chain-pair interaction metrics from Boltz2 outputs.

Reads confidence JSONs plus full PAE/PDE npz files and writes:
1) per-model chain-pair table
2) mean-over-models chain-pair table
3) top-confidence model chain-pair table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _find_record_id(pred_dir: Path) -> str:
    subdirs = [p for p in pred_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 prediction subdir in {pred_dir}, found {len(subdirs)}"
        )
    return subdirs[0].name


def _load_chain_info(out_dir: Path, record_id: str):
    record_path = out_dir / "processed" / "records" / f"{record_id}.json"
    data = json.loads(record_path.read_text())
    chains = data["chains"]
    # Sort by chain_id to match pair_chains_iptm indexing
    chains = sorted(chains, key=lambda x: int(x["chain_id"]))
    names = [c["chain_name"] for c in chains]
    lengths = [int(c["num_residues"]) for c in chains]
    return names, lengths


def _chain_slices(lengths):
    starts = [0]
    for ln in lengths[:-1]:
        starts.append(starts[-1] + ln)
    return [(s, s + ln) for s, ln in zip(starts, lengths)]


def _mean_pair_block(mat: np.ndarray, s1, e1, s2, e2) -> float:
    # Average both directions to reduce asymmetry effects.
    b12 = mat[s1:e1, s2:e2]
    b21 = mat[s2:e2, s1:e1]
    return float((b12.mean() + b21.mean()) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Boltz run output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pred_root = out_dir / "predictions"
    record_id = _find_record_id(pred_root)
    pred_dir = pred_root / record_id

    names, lengths = _load_chain_info(out_dir, record_id)
    slices = _chain_slices(lengths)
    n_tokens = sum(lengths)

    conf_paths = sorted(glob.glob(str(pred_dir / "confidence_*_model_*.json")))
    if not conf_paths:
        raise RuntimeError(f"No confidence json files found in {pred_dir}")

    rows = []
    by_pair = defaultdict(list)
    model_scores = {}

    for conf_path in conf_paths:
        m = re.search(r"_model_(\d+)\.json$", conf_path)
        if not m:
            continue
        model_idx = int(m.group(1))
        conf = json.loads(Path(conf_path).read_text())
        model_scores[model_idx] = float(conf["confidence_score"])

        pae_path = pred_dir / f"pae_{record_id}_model_{model_idx}.npz"
        pde_path = pred_dir / f"pde_{record_id}_model_{model_idx}.npz"
        if not pae_path.exists() or not pde_path.exists():
            raise RuntimeError(
                f"Missing full PAE/PDE for model {model_idx}: {pae_path.name}, {pde_path.name}"
            )

        pae = np.load(pae_path)["pae"]
        pde = np.load(pde_path)["pde"]
        if pae.shape != (n_tokens, n_tokens) or pde.shape != (n_tokens, n_tokens):
            raise RuntimeError(
                f"Unexpected matrix shape for model {model_idx}: "
                f"pae={pae.shape}, pde={pde.shape}, expected {(n_tokens, n_tokens)}"
            )

        pair_iptm = conf["pair_chains_iptm"]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s1, e1 = slices[i]
                s2, e2 = slices[j]
                mean_pae = _mean_pair_block(pae, s1, e1, s2, e2)
                mean_pde = _mean_pair_block(pde, s1, e1, s2, e2)
                iptm_ij = float(pair_iptm[str(i)][str(j)])
                row = {
                    "model": model_idx,
                    "chain_i": names[i],
                    "chain_j": names[j],
                    "pair_iptm": iptm_ij,
                    "mean_pae": mean_pae,
                    "mean_ipde": mean_pde,
                }
                rows.append(row)
                by_pair[(names[i], names[j])].append(row)

    # Write per-model table
    per_model_path = pred_dir / "chain_pair_metrics_all_models.tsv"
    with per_model_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "chain_i", "chain_j", "pair_iptm", "mean_pae", "mean_ipde"],
            delimiter="\t",
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["model"], x["chain_i"], x["chain_j"])):
            w.writerow(r)

    # Mean-over-models table
    mean_path = pred_dir / "chain_pair_metrics_mean.tsv"
    with mean_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chain_i",
                "chain_j",
                "pair_iptm_mean",
                "pair_iptm_std",
                "mean_pae_mean",
                "mean_pae_std",
                "mean_ipde_mean",
                "mean_ipde_std",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for (ci, cj), vals in sorted(by_pair.items()):
            iptm_vals = np.array([v["pair_iptm"] for v in vals], dtype=np.float64)
            pae_vals = np.array([v["mean_pae"] for v in vals], dtype=np.float64)
            ipde_vals = np.array([v["mean_ipde"] for v in vals], dtype=np.float64)
            w.writerow(
                {
                    "chain_i": ci,
                    "chain_j": cj,
                    "pair_iptm_mean": float(iptm_vals.mean()),
                    "pair_iptm_std": float(iptm_vals.std()),
                    "mean_pae_mean": float(pae_vals.mean()),
                    "mean_pae_std": float(pae_vals.std()),
                    "mean_ipde_mean": float(ipde_vals.mean()),
                    "mean_ipde_std": float(ipde_vals.std()),
                }
            )

    # Top-confidence model table
    top_model = max(model_scores.items(), key=lambda x: x[1])[0]
    top_path = pred_dir / "chain_pair_metrics_top_model.tsv"
    with top_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "chain_i", "chain_j", "pair_iptm", "mean_pae", "mean_ipde"],
            delimiter="\t",
        )
        w.writeheader()
        for r in sorted(
            [x for x in rows if x["model"] == top_model],
            key=lambda x: (x["chain_i"], x["chain_j"]),
        ):
            w.writerow(r)

    print(f"[OK] record_id: {record_id}")
    print(f"[OK] models: {len(model_scores)}")
    print(f"[OK] top_model_by_confidence: {top_model} (score={model_scores[top_model]:.6f})")
    print(f"[OK] wrote: {per_model_path}")
    print(f"[OK] wrote: {mean_path}")
    print(f"[OK] wrote: {top_path}")


if __name__ == "__main__":
    main()
