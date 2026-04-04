# Auxiliary scripts

Helper Python scripts for comparison, analysis, and testing. Used by some jobs in `slurm/` or run standalone.

- **compare_*** — checkpoint/structure comparison (e.g. baseline vs DAP, granular checkpoints)
- **analyze_***, **diag_*** — divergence analysis, checkpoint diagnostics
- **plot_memory.py** — memory timeline plotting (used by `slurm/run_tmpl_shard_verify.sbatch`)
- **run_boltz_baseline.py**, **run_trunk_only.py**, **run_oom_profile.py** — single-GPU/original Boltz runs for baselines
- **run_cp_examples_with_dap.py** — runs `boltz-cp/examples/*.yaml` with `boltz_dap`, rewriting relative MSA/template paths to absolute paths first
- **test_*** — DAP layer tests, FlexAttention vs SDPA, etc.

Run from repo root, e.g. `python scripts/compare_checkpoints.py ...` or let slurm jobs call them via `${DAP_DIR}/scripts/...`.

Example:
`python scripts/run_cp_examples_with_dap.py --examples prot multimer ligand --use-msa-server-mode auto`

GPU batch example:
`EXAMPLES="prot multimer ligand" sbatch slurm/run_cp_examples_with_dap.sbatch`
