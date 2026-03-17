# Auxiliary scripts

Helper Python scripts for comparison, analysis, and testing. Used by some jobs in `slurm/` or run standalone.

- **compare_*** — checkpoint/structure comparison (e.g. baseline vs DAP, granular checkpoints)
- **analyze_***, **diag_*** — divergence analysis, checkpoint diagnostics
- **plot_memory.py** — memory timeline plotting (used by `slurm/run_tmpl_shard_verify.sbatch`)
- **run_boltz_baseline.py**, **run_trunk_only.py**, **run_oom_profile.py** — single-GPU/original Boltz runs for baselines
- **test_*** — DAP layer tests, FlexAttention vs SDPA, etc.

Run from repo root, e.g. `python scripts/compare_checkpoints.py ...` or let slurm jobs call them via `${DAP_DIR}/scripts/...`.
