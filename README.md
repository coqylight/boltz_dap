# Boltz-DAP: Distributed Axial Parallelism for Boltz 2

> Run [Boltz 2](https://github.com/jwohlwend/boltz) protein structure prediction on large complexes (hexamers, pentamers) that OOM on a single GPU.

DAP (**D**ynamic **A**xial **P**arallelism) shards the pair representation `z [B, N, N, D]` across multiple GPUs along the row dimension, so no single GPU ever holds the full N×N tensor. This reduces peak memory proportionally to the number of GPUs — **4 GPUs → ~4× less memory per GPU**.

## Why?

Original Boltz 2 holds the full pair tensor on **1 GPU**. For large complexes:

| Complex | N (tokens) | Original Boltz 2 | DAP (4 GPUs) |
|---------|-----------|------------------|--------------|
| Trimer (3 × 519 aa) | ~1,557 | ⚠️ Tight | ✅ ~12 GB/GPU |
| Pentamer (5 × 519 aa) | ~2,595 | ❌ OOM | ✅ ~36 GB/GPU |
| Hexamer (6 × 519 aa) | ~3,114 | ❌ OOM | ✅ ~45 GB/GPU |

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  ALL GPUs: Input embedding → z_init [B, N, N, 128]  │
└──────────────────────┬──────────────────────────────┘
                       ▼
              scatter(z, dim=1)
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
   GPU 0: z₀       GPU 1: z₁     GPU 2: z₂    ...
   [B,N/P,N,D]    [B,N/P,N,D]   [B,N/P,N,D]
         │             │              │
         ▼             ▼              ▼
   ┌──────────────────────────────────────────┐
   │  Trunk Loop (48 Pairformer layers):      │
   │    • TriMulOut  (broadcast-chunked)      │
   │    • TriMulIn   (row↔col + broadcast)    │
   │    • TriAttStart (gather only H-bias)    │
   │    • TriAttEnd   (row↔col + attention)   │
   │    • Transition  (pointwise, no comm)    │
   │    • SeqAttn     (gather only pair bias) │
   └──────────────────────────────────────────┘
         │             │              │
         ▼             ▼              ▼
              gather(z, dim=1)
                       ▼
        z_full [B, N, N, 128]  (GPU 0 only)
                       ▼
         Distogram → Diffusion → Confidence
```

The full `z` is only materialized at scatter/gather boundaries. The entire trunk loop operates on smaller shards.

## Quick Start

### Prerequisites

- **2+ GPUs** on the same node (NVLink recommended)
- Python 3.10+, PyTorch 2.x with CUDA
- [Boltz 2](https://github.com/jwohlwend/boltz) installed (`pip install boltz`)

### Running

```bash
# 4 GPUs
torchrun --nproc_per_node=4 boltz_dap_v2/run_boltz_dap_v2.py \
    input.yaml \
    --out_dir ./output \
    --cache ~/.boltz

# 2 GPUs
torchrun --nproc_per_node=2 boltz_dap_v2/run_boltz_dap_v2.py \
    input.yaml \
    --out_dir ./output \
    --cache ~/.boltz
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--out_dir` | (required) | Output directory |
| `--cache` | `~/.boltz` | Model weights cache |
| `--recycling_steps` | 3 | Number of recycling iterations |
| `--sampling_steps` | 200 | Diffusion sampling steps |
| `--diffusion_samples` | 1 | Number of diffusion samples |
| `--no_kernels` | off | Disable cuequivariance CUDA kernels |
| `--seed` | None | Random seed for reproducibility |

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=boltz-dap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=1:00:00

srun torchrun --nproc_per_node=4 \
    boltz_dap_v2/run_boltz_dap_v2.py \
    input.yaml \
    --out_dir ./output \
    --cache ~/.boltz \
    --recycling_steps 3
```

## Project Structure

```
boltz_dap/
├── boltz_dap_v2/                    # DAP-aware layer wrappers
│   ├── run_boltz_dap_v2.py          # Entry point (replaces `boltz predict`)
│   ├── dap_trunk.py                 # Main forward: scatter → trunk → gather
│   ├── dap_pairformer.py            # PairformerLayer wrapper (with seq attention)
│   ├── dap_pairformer_noseq.py      # PairformerLayer wrapper (for templates)
│   ├── dap_trimul.py                # Triangle multiplication (broadcast-chunked)
│   ├── dap_tri_att.py               # Triangle attention (gather only bias)
│   ├── dap_msa.py                   # MSA module wrapper
│   └── dap_confidence.py            # Confidence module wrapper
├── boltz_distributed/               # Communication primitives
│   ├── core.py                      # init_dap(), get_dap_rank(), get_dap_size()
│   ├── comm.py                      # scatter, gather, row_to_col, col_to_row
│   └── wrappers.py                  # Helper wrappers
└── README.md
```

## Key Design Decisions

### Zero Boltz 2 Modifications

DAP **does not modify any original Boltz 2 source code**. Instead, it monkey-patches the model at runtime:

```python
# dap_trunk.py
inject_dap_into_model(model)  # Wraps each layer with DAP-aware version
```

The original `boltz/` package remains untouched. All weights are identical.

### Broadcast-Chunked Triangle Multiplication

The hardest operation to distribute. Instead of all-gathering the full tensor (which would defeat the purpose), each GPU broadcasts its shard one at a time:

```python
# Each GPU broadcasts b_chunk, others compute partial output
for src in range(dap_size):
    dist.broadcast(b_chunk, src=src)       # One shard at a time
    out[:, :, j_start:j_end, :] = einsum(  # Fill j-columns
        "bikd,bjkd->bijd", a, b_chunk
    )
```

Peak memory stays at ~2× shard size vs full N×N.

### Bias-Only Gathering

For triangle attention and sequence attention, only the small **bias tensor** `[B, H, N, N]` (H ≈ 4–16) is gathered, not the full `z [B, N, N, 128]`. This reduces communication by ~8–32×.

## Numerical Accuracy

DAP produces results with minor floating-point differences from single-GPU Boltz 2, due to different operation ordering in distributed reductions. Structure predictions (LDDT, TM-score) are statistically equivalent.

## References

- [Boltz 2](https://github.com/jwohlwend/boltz) — Base model
- [FastFold](https://github.com/hpcaitech/FastFold) — DAP communication primitives (adapted)
- [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) — Triangle operations architecture

## License

This DAP wrapper follows the same license as Boltz 2.
