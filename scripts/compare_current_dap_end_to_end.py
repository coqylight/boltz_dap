#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dap", type=Path, required=True)
    parser.add_argument("--max-tol", type=float, default=1e-2)
    parser.add_argument("--mean-tol", type=float, default=1e-4)
    parser.add_argument("--cos-tol", type=float, default=0.99999)
    return parser.parse_args()


def compare_tensor(
    name: str,
    baseline: torch.Tensor,
    dap: torch.Tensor,
    max_tol: float,
    mean_tol: float,
    cos_tol: float,
) -> None:
    if baseline.shape != dap.shape:
        raise AssertionError(f"{name} shape mismatch: {baseline.shape} != {dap.shape}")
    baseline = baseline.float()
    dap = dap.float()
    diff = (baseline - dap).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    cosine = torch.nn.functional.cosine_similarity(
        baseline.reshape(1, -1), dap.reshape(1, -1)
    ).item()
    print(
        f"{name}: shape={tuple(baseline.shape)} mean={mean_diff:.6e} "
        f"max={max_diff:.6e} cos={cosine:.8f}",
        flush=True,
    )
    if mean_diff > mean_tol or max_diff > max_tol or cosine < cos_tol:
        raise AssertionError(
            f"{name} exceeded parity envelope: mean={mean_diff:.6e}, "
            f"max={max_diff:.6e}, cos={cosine:.8f}"
        )


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern} under {root}, found {matches}")
    return matches[0]


def atom_coordinates(cif_path: Path) -> dict[tuple[str, int, str], np.ndarray]:
    structure = gemmi.read_structure(str(cif_path))
    if len(structure) != 1:
        raise AssertionError(f"Expected one model in {cif_path}, found {len(structure)}")
    coordinates = {}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                key = (chain.name, residue.seqid.num, atom.name)
                coordinates[key] = np.array(
                    [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
                )
    if not coordinates:
        raise AssertionError(f"No atoms found in {cif_path}")
    values = np.stack(list(coordinates.values()))
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite coordinates found in {cif_path}")
    return coordinates


def aligned_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    reference_centered = reference - reference.mean(axis=0)
    mobile_centered = mobile - mobile.mean(axis=0)
    covariance = mobile_centered.T @ reference_centered
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ correction @ right
    aligned = mobile_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((reference_centered - aligned) ** 2, axis=1))))


def load_confidence(root: Path) -> dict:
    path = find_one(root / "predictions", "confidence_*.json")
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    baseline_tensors = torch.load(
        args.baseline / "zs_tensors.pt", map_location="cpu", weights_only=True
    )
    dap_tensors = torch.load(
        args.dap / "zs_tensors.pt", map_location="cpu", weights_only=True
    )
    parity_errors = []
    for name in ("s", "z"):
        if name not in baseline_tensors or name not in dap_tensors:
            raise AssertionError(f"Missing {name} in saved trunk tensors")
        try:
            compare_tensor(
                name,
                baseline_tensors[name],
                dap_tensors[name],
                args.max_tol,
                args.mean_tol,
                args.cos_tol,
            )
        except AssertionError as exc:
            parity_errors.append(str(exc))
            print(f"PARITY FAILURE: {exc}", flush=True)

    baseline_cif = find_one(args.baseline / "predictions", "*_model_0.cif")
    dap_cif = find_one(args.dap / "predictions", "*_model_0.cif")
    baseline_atoms = atom_coordinates(baseline_cif)
    dap_atoms = atom_coordinates(dap_cif)
    if baseline_atoms.keys() != dap_atoms.keys():
        missing = sorted(baseline_atoms.keys() - dap_atoms.keys())[:5]
        extra = sorted(dap_atoms.keys() - baseline_atoms.keys())[:5]
        raise AssertionError(f"CIF topology differs: missing={missing}, extra={extra}")

    atom_keys = sorted(baseline_atoms)
    baseline_coordinates = np.stack([baseline_atoms[key] for key in atom_keys])
    dap_coordinates = np.stack([dap_atoms[key] for key in atom_keys])
    raw_rmsd = float(
        np.sqrt(np.mean(np.sum((baseline_coordinates - dap_coordinates) ** 2, axis=1)))
    )
    fit_rmsd = aligned_rmsd(baseline_coordinates, dap_coordinates)
    chains = sorted({key[0] for key in atom_keys})
    residues = {(key[0], key[1]) for key in atom_keys}
    print(
        f"cif: chains={len(chains)} residues={len(residues)} atoms={len(atom_keys)} "
        f"raw_rmsd={raw_rmsd:.6f}A aligned_rmsd={fit_rmsd:.6f}A",
        flush=True,
    )

    baseline_confidence = load_confidence(args.baseline)
    dap_confidence = load_confidence(args.dap)
    for key in ("confidence_score", "ptm", "complex_plddt", "complex_pde"):
        baseline_value = float(baseline_confidence[key])
        dap_value = float(dap_confidence[key])
        print(
            f"confidence {key}: baseline={baseline_value:.6f} "
            f"dap={dap_value:.6f} delta={dap_value - baseline_value:+.6f}",
            flush=True,
        )

    print(f"baseline_cif={baseline_cif}", flush=True)
    print(f"dap_cif={dap_cif}", flush=True)
    if parity_errors:
        raise AssertionError("; ".join(parity_errors))
    print("Current original-vs-DAP end-to-end validation passed", flush=True)


if __name__ == "__main__":
    main()