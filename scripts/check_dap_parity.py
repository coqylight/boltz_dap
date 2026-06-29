#!/usr/bin/env python3
"""Compare baseline and DAP z/s tensors for math parity smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_tensor_dict(path: Path) -> dict[str, torch.Tensor]:
    # mmap avoids eagerly materializing giant storages into RAM.
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a dict payload")
    if "z" not in payload or "s" not in payload:
        raise KeyError(f"{path} must contain 'z' and 's' tensors")
    return payload


def _stats_chunked(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
    chunk_elems: int = 8_000_000,
) -> dict[str, float | bool]:
    """Compute parity stats in chunks to cap peak host memory."""
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if flat_a.numel() != flat_b.numel():
        raise ValueError("Tensor element counts do not match")

    total_abs = 0.0
    total_numel = int(flat_a.numel())
    max_abs = 0.0
    max_b_abs = 0.0
    allclose_ok = True

    for start in range(0, total_numel, chunk_elems):
        end = min(start + chunk_elems, total_numel)
        a_chunk = flat_a[start:end].float()
        b_chunk = flat_b[start:end].float()
        diff = (a_chunk - b_chunk).abs()
        total_abs += float(diff.sum().item())
        max_abs = max(max_abs, float(diff.max().item()))
        max_b_abs = max(max_b_abs, float(b_chunk.abs().max().item()))
        if allclose_ok and not torch.allclose(a_chunk, b_chunk, atol=atol, rtol=rtol):
            allclose_ok = False

    mean_abs = total_abs / total_numel if total_numel > 0 else 0.0
    max_rel = max_abs / max_b_abs if max_b_abs > 0 else max_abs
    return {
        "allclose": allclose_ok,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline zs_tensors.pt")
    parser.add_argument("--dap", type=Path, required=True, help="DAP zs_tensors.pt")
    parser.add_argument("--atol", type=float, default=5e-4, help="absolute tolerance")
    parser.add_argument("--rtol", type=float, default=5e-4, help="relative tolerance")
    parser.add_argument("--summary-json", type=Path, default=None, help="optional JSON summary output")
    args = parser.parse_args()

    baseline = _load_tensor_dict(args.baseline)
    dap = _load_tensor_dict(args.dap)

    summary: dict[str, object] = {
        "baseline": str(args.baseline),
        "dap": str(args.dap),
        "atol": args.atol,
        "rtol": args.rtol,
        "tensors": {},
    }

    failed = False
    for name in ("z", "s"):
        a = baseline[name]
        b = dap[name]
        if tuple(a.shape) != tuple(b.shape):
            failed = True
            summary["tensors"][name] = {
                "shape_match": False,
                "baseline_shape": list(a.shape),
                "dap_shape": list(b.shape),
            }
            continue

        stats = _stats_chunked(a, b, atol=args.atol, rtol=args.rtol)
        ok = bool(stats["allclose"])
        summary["tensors"][name] = {
            "shape_match": True,
            "shape": list(a.shape),
            **stats,
        }
        failed = failed or (not ok)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
