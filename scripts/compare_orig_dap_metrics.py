#!/usr/bin/env python3
"""Compare original Boltz2 vs DAP metrics for one target.

Outputs:
- model_level_metric_deltas.tsv
- chain_pair_metric_deltas_mean.tsv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _find_record_id(pred_root: Path) -> str:
    subdirs = [p for p in pred_root.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 prediction subdir in {pred_root}, found {len(subdirs)}"
        )
    return subdirs[0].name


def _load_chain_info(out_dir: Path, record_id: str):
    rec = json.loads((out_dir / "processed" / "records" / f"{record_id}.json").read_text())
    chains = sorted(rec["chains"], key=lambda x: int(x["chain_id"]))
    names = [c["chain_name"] for c in chains]
    lengths = [int(c["num_residues"]) for c in chains]
    return names, lengths


def _chain_slices(lengths):
    starts = [0]
    for ln in lengths[:-1]:
        starts.append(starts[-1] + ln)
    return [(s, s + ln) for s, ln in zip(starts, lengths)]


def _pair_block_mean(mat: np.ndarray, s1, e1, s2, e2) -> float:
    b12 = mat[s1:e1, s2:e2]
    b21 = mat[s2:e2, s1:e1]
    return float((b12.mean() + b21.mean()) / 2.0)


def _model_idx(path: Path) -> int:
    m = re.search(r"_model_(\d+)\.json$", path.name)
    if not m:
        raise RuntimeError(f"Cannot parse model index from {path}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_out", required=True)
    ap.add_argument("--dap_out", required=True)
    ap.add_argument("--report_dir", default=None)
    args = ap.parse_args()

    orig_out = Path(args.orig_out)
    dap_out = Path(args.dap_out)
    report_dir = Path(args.report_dir) if args.report_dir else (dap_out / "comparison_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    orig_pred_root = orig_out / "predictions"
    dap_pred_root = dap_out / "predictions"
    orig_record = _find_record_id(orig_pred_root)
    dap_record = _find_record_id(dap_pred_root)
    if orig_record != dap_record:
        raise RuntimeError(f"Record mismatch: orig={orig_record}, dap={dap_record}")
    record_id = orig_record

    names, lengths = _load_chain_info(orig_out, record_id)
    slices = _chain_slices(lengths)
    n_tokens = sum(lengths)

    orig_pred_dir = orig_pred_root / record_id
    dap_pred_dir = dap_pred_root / record_id

    orig_conf_paths = sorted(glob.glob(str(orig_pred_dir / "confidence_*_model_*.json")))
    dap_conf_paths = sorted(glob.glob(str(dap_pred_dir / "confidence_*_model_*.json")))
    if len(orig_conf_paths) != len(dap_conf_paths):
        raise RuntimeError(
            f"Different model counts: orig={len(orig_conf_paths)}, dap={len(dap_conf_paths)}"
        )

    orig_confs = {_model_idx(Path(p)): json.loads(Path(p).read_text()) for p in orig_conf_paths}
    dap_confs = {_model_idx(Path(p)): json.loads(Path(p).read_text()) for p in dap_conf_paths}
    common_models = sorted(set(orig_confs.keys()) & set(dap_confs.keys()))
    if not common_models:
        raise RuntimeError("No common model indices to compare")

    # 1) Model-level summary deltas
    metric_keys = [
        "confidence_score",
        "complex_plddt",
        "complex_iplddt",
        "ptm",
        "iptm",
        "complex_pde",
        "complex_ipde",
    ]
    model_rows = []
    for m in common_models:
        o = orig_confs[m]
        d = dap_confs[m]
        row = {"model": m}
        for k in metric_keys:
            ov = float(o[k])
            dv = float(d[k])
            row[f"orig_{k}"] = ov
            row[f"dap_{k}"] = dv
            row[f"delta_{k}"] = dv - ov
        model_rows.append(row)

    model_out = report_dir / "model_level_metric_deltas.tsv"
    model_fields = ["model"] + [f"orig_{k}" for k in metric_keys] + [f"dap_{k}" for k in metric_keys] + [f"delta_{k}" for k in metric_keys]
    with model_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=model_fields, delimiter="\t")
        w.writeheader()
        for r in model_rows:
            w.writerow(r)

    # 2) Chain-pair means over models for iPTM / mean PAE / mean iPDE
    by_pair = defaultdict(list)
    for m in common_models:
        oc = orig_confs[m]["pair_chains_iptm"]
        dc = dap_confs[m]["pair_chains_iptm"]
        opae = np.load(orig_pred_dir / f"pae_{record_id}_model_{m}.npz")["pae"]
        dpae = np.load(dap_pred_dir / f"pae_{record_id}_model_{m}.npz")["pae"]
        opde = np.load(orig_pred_dir / f"pde_{record_id}_model_{m}.npz")["pde"]
        dpde = np.load(dap_pred_dir / f"pde_{record_id}_model_{m}.npz")["pde"]

        if opae.shape != (n_tokens, n_tokens) or dpae.shape != (n_tokens, n_tokens):
            raise RuntimeError(f"Unexpected PAE shape for model {m}")
        if opde.shape != (n_tokens, n_tokens) or dpde.shape != (n_tokens, n_tokens):
            raise RuntimeError(f"Unexpected PDE shape for model {m}")

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s1, e1 = slices[i]
                s2, e2 = slices[j]
                rec = {
                    "orig_iptm": float(oc[str(i)][str(j)]),
                    "dap_iptm": float(dc[str(i)][str(j)]),
                    "orig_mean_pae": _pair_block_mean(opae, s1, e1, s2, e2),
                    "dap_mean_pae": _pair_block_mean(dpae, s1, e1, s2, e2),
                    "orig_mean_ipde": _pair_block_mean(opde, s1, e1, s2, e2),
                    "dap_mean_ipde": _pair_block_mean(dpde, s1, e1, s2, e2),
                }
                by_pair[(names[i], names[j])].append(rec)

    pair_out = report_dir / "chain_pair_metric_deltas_mean.tsv"
    with pair_out.open("w", newline="") as f:
        fields = [
            "chain_i",
            "chain_j",
            "orig_iptm_mean",
            "dap_iptm_mean",
            "delta_iptm_mean",
            "orig_mean_pae_mean",
            "dap_mean_pae_mean",
            "delta_mean_pae_mean",
            "orig_mean_ipde_mean",
            "dap_mean_ipde_mean",
            "delta_mean_ipde_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for (ci, cj), vals in sorted(by_pair.items()):
            o_iptm = np.array([v["orig_iptm"] for v in vals], dtype=np.float64)
            d_iptm = np.array([v["dap_iptm"] for v in vals], dtype=np.float64)
            o_pae = np.array([v["orig_mean_pae"] for v in vals], dtype=np.float64)
            d_pae = np.array([v["dap_mean_pae"] for v in vals], dtype=np.float64)
            o_ipde = np.array([v["orig_mean_ipde"] for v in vals], dtype=np.float64)
            d_ipde = np.array([v["dap_mean_ipde"] for v in vals], dtype=np.float64)
            w.writerow(
                {
                    "chain_i": ci,
                    "chain_j": cj,
                    "orig_iptm_mean": float(o_iptm.mean()),
                    "dap_iptm_mean": float(d_iptm.mean()),
                    "delta_iptm_mean": float((d_iptm - o_iptm).mean()),
                    "orig_mean_pae_mean": float(o_pae.mean()),
                    "dap_mean_pae_mean": float(d_pae.mean()),
                    "delta_mean_pae_mean": float((d_pae - o_pae).mean()),
                    "orig_mean_ipde_mean": float(o_ipde.mean()),
                    "dap_mean_ipde_mean": float(d_ipde.mean()),
                    "delta_mean_ipde_mean": float((d_ipde - o_ipde).mean()),
                }
            )

    print(f"[OK] compared models: {len(common_models)}")
    print(f"[OK] record_id: {record_id}")
    print(f"[OK] wrote: {model_out}")
    print(f"[OK] wrote: {pair_out}")


if __name__ == "__main__":
    main()
