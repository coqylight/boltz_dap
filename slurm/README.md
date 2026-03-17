# SLURM / job scripts

This directory contains `.sbatch` and `.sh` scripts for running DAP and comparison jobs on an HPC cluster (e.g. SLURM).

- **Submit from repo root**, e.g.:
  ```bash
  sbatch slurm/run_hex_dap4.sbatch
  ```
- Job scripts use `DAP_DIR` (or hardcoded paths) to find the repo; ensure `DAP_DIR` points to the `boltz_dap` repository root if you clone elsewhere.
- Python helpers (e.g. `compare_checkpoints.py`, `plot_memory.py`) live in **`scripts/`**; job scripts reference them as `${DAP_DIR}/scripts/...`.
