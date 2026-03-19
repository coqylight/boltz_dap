#!/usr/bin/env python3
"""Generate a compact markdown report for the hexamer reproduction run."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


def _find_record_id(pred_root: Path) -> str:
    subdirs = [p for p in pred_root.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 prediction subdir in {pred_root}, found {len(subdirs)}"
        )
    return subdirs[0].name


def _read_tsv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _parse_peak_from_log(log_path: Path):
    if not log_path.exists():
        return []
    peaks = []
    pat = re.compile(r"GPU (\d+): Peak = (\d+) MB")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            peaks.append((int(m.group(1)), int(m.group(2))))
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pred_root = out_dir / "predictions"
    record_id = _find_record_id(pred_root)
    pred_dir = pred_root / record_id

    cifs = sorted(glob.glob(str(pred_dir / "*.cif")))
    confs = sorted(glob.glob(str(pred_dir / "confidence_*.json")))
    pae_npz = sorted(glob.glob(str(pred_dir / "pae_*.npz")))
    pde_npz = sorted(glob.glob(str(pred_dir / "pde_*.npz")))

    mean_tsv = pred_dir / "chain_pair_metrics_mean.tsv"
    top_tsv = pred_dir / "chain_pair_metrics_top_model.tsv"
    mean_rows = _read_tsv(mean_tsv) if mean_tsv.exists() else []
    top_rows = _read_tsv(top_tsv) if top_tsv.exists() else []

    top_conf = None
    if confs:
        all_conf = []
        for p in confs:
            d = json.loads(Path(p).read_text())
            m = re.search(r"_model_(\d+)\.json$", os.path.basename(p))
            model = int(m.group(1)) if m else -1
            all_conf.append((model, float(d["confidence_score"]), d))
        top_conf = max(all_conf, key=lambda x: x[1])

    strongest = None
    weakest = None
    if mean_rows:
        strongest = max(mean_rows, key=lambda r: _to_float(r["pair_iptm_mean"]) or -1e9)
        weakest = min(mean_rows, key=lambda r: _to_float(r["pair_iptm_mean"]) or 1e9)

    job_logs = sorted(out_dir.parent.glob("hexamer_af3_defaults_*.log"), key=lambda p: p.stat().st_mtime)
    peaks = _parse_peak_from_log(job_logs[-1]) if job_logs else []

    lines = []
    lines.append("# Hexamer Reproduction Report")
    lines.append("")
    lines.append(f"- `record_id`: `{record_id}`")
    lines.append(f"- `out_dir`: `{out_dir}`")
    lines.append(f"- `cif_count`: `{len(cifs)}`")
    lines.append(f"- `confidence_json_count`: `{len(confs)}`")
    lines.append(f"- `full_pae_npz_count`: `{len(pae_npz)}`")
    lines.append(f"- `full_pde_npz_count`: `{len(pde_npz)}`")
    lines.append("")

    if top_conf:
        model, score, d = top_conf
        lines.append("## Top Model")
        lines.append("")
        lines.append(f"- `model_{model}`")
        lines.append(f"- `confidence_score`: `{score:.6f}`")
        lines.append(f"- `complex_plddt`: `{float(d['complex_plddt']):.6f}`")
        lines.append(f"- `iptm`: `{float(d['iptm']):.6f}`")
        lines.append(f"- `complex_ipde`: `{float(d['complex_ipde']):.6f}`")
        lines.append("")

    if strongest and weakest:
        lines.append("## Chain-Pair Summary")
        lines.append("")
        lines.append(
            f"- Strongest pair by mean iPTM: `{strongest['chain_i']}-{strongest['chain_j']}` "
            f"(iPTM=`{_to_float(strongest['pair_iptm_mean']):.6f}`, "
            f"mean PAE=`{_to_float(strongest['mean_pae_mean']):.6f}`, "
            f"mean iPDE=`{_to_float(strongest['mean_ipde_mean']):.6f}`)"
        )
        lines.append(
            f"- Weakest pair by mean iPTM: `{weakest['chain_i']}-{weakest['chain_j']}` "
            f"(iPTM=`{_to_float(weakest['pair_iptm_mean']):.6f}`, "
            f"mean PAE=`{_to_float(weakest['mean_pae_mean']):.6f}`, "
            f"mean iPDE=`{_to_float(weakest['mean_ipde_mean']):.6f}`)"
        )
        lines.append("")
        lines.append("## Mean Chain-Pair Table")
        lines.append("")
        lines.append("| Pair | mean iPTM | mean PAE | mean iPDE |")
        lines.append("|------|-----------|----------|-----------|")
        for r in sorted(mean_rows, key=lambda r: _to_float(r["pair_iptm_mean"]) or -1e9, reverse=True):
            lines.append(
                f"| `{r['chain_i']}-{r['chain_j']}` | "
                f"{_to_float(r['pair_iptm_mean']):.6f} | "
                f"{_to_float(r['mean_pae_mean']):.6f} | "
                f"{_to_float(r['mean_ipde_mean']):.6f} |"
            )
        lines.append("")

    if peaks:
        lines.append("## GPU Peak Memory")
        lines.append("")
        for gpu_id, peak_mb in peaks:
            lines.append(f"- GPU {gpu_id}: `{peak_mb} MB` (`{peak_mb/1024:.1f} GB`)")
        lines.append("")

    lines.append("## Pass Criteria")
    lines.append("")
    lines.append(f"- 25 CIF files generated: `{'yes' if len(cifs) == 25 else 'no'}`")
    lines.append(f"- 25 confidence JSON files generated: `{'yes' if len(confs) == 25 else 'no'}`")
    lines.append(f"- 25 full PAE files generated: `{'yes' if len(pae_npz) == 25 else 'no'}`")
    lines.append(f"- 25 full PDE files generated: `{'yes' if len(pde_npz) == 25 else 'no'}`")
    lines.append(f"- Chain-pair summary generated: `{'yes' if mean_tsv.exists() and top_tsv.exists() else 'no'}`")
    lines.append("")

    out_path = out_dir / "repro_report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote report: {out_path}")


if __name__ == "__main__":
    main()
