# Boltz-DAP: Distributed Axial Parallelism for Boltz-2

> Run [Boltz-2](https://github.com/jwohlwend/boltz) protein structure prediction on large complexes (>2,000 amino acid residues) across multiple GPUs without OOM.

DAP (**D**ynamic **A**xial **P**arallelism) shards the pair representation `z [B, N, N, D]` across multiple GPUs along the row dimension, so no single GPU ever holds the full N×N tensor. This reduces peak memory proportionally to the number of GPUs — **4 GPUs → ~4× less memory per GPU**.

## Why?

Original Boltz-2 holds the full pair representation tensor on **1 GPU**. For large complexes (>2,000 residues), this leads to CUDA out-of-memory (OOM) errors in consumer grade GPUs (VRAM < 48 GB). DAP enables Boltz-2 to run on **multiple GPUs** without OOM, even for large complexes like adeno-associated virus (AAV) hexamers.

| Complex | N (tokens) | Original Boltz-2 | DAP (4 GPUs) |
|---------|-----------|------------------|--------------|
| AAV2 VP3 Trimer (3 × 519 aa) | ~1,557 | ⚠️ Tight | ✅ ~12 GB/GPU |
| AAV2 VP3 Pentamer (5 × 519 aa) | ~2,595 | ❌ OOM | ✅ ~36 GB/GPU |
| AAV2 VP3 Hexamer (6 × 519 aa) | ~3,114 | ❌ OOM | ✅ ~45 GB/GPU |

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

**First time here?** See **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** for a step-by-step guide: install Boltz-2, clone this repo, prepare input YAML, and run DAP.

### Prerequisites

- **2+ GPUs** on the same node (NVLink recommended)
- Python 3.10+, PyTorch 2.x with CUDA
- [Boltz-2](https://github.com/jwohlwend/boltz) installed (`pip install boltz`)

### Tested environment

| Item | Used in development |
|------|---------------------|
| GPU | NVIDIA H800 (2, 4, or 8 per node) |
| CUDA | Compatible with PyTorch 2.x |
| GPU counts tested | 2, 4, 8 GPUs (e.g. trimer/hexamer on 4; 9MME N≈4642 on 8) |
| Settings tested | **Boltz2 default**: `recycling_steps=3`, `sampling_steps=200`, `diffusion_samples=1` · **AF3 default**: `recycling_steps=10`, `sampling_steps=200`, `diffusion_samples=25` |
| Workloads | AAV2 VP3 Trimer (e.g. 3×519 aa), AAV2 VP3 Hexamer (6×~519 aa, 25 samples with `--use_flex_attention_chunked`), 9MME (N≈4642, 8 GPUs) |

Other GPU models (A100, V100, etc.) should work with 2+ GPUs; memory per GPU scales with shard size.

**Example log file:** [example_hexamer_25cif_full.log](example_hexamer_25cif_full.log) — full run that produced 25 CIF files (AAV2 VP3 Hexamer, 4 GPUs, `--use_flex_attention_chunked`, AF3 defaults). Large (~8.8 MB) but useful as a reference.

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
| `--recycling_steps` | 3 | Number of recycling iterations (AF3-style default) |
| `--sampling_steps` | 200 | Diffusion sampling steps |
| `--diffusion_samples` | 1 | Number of diffusion samples |
| `--use_msa_server` | off | Use MSA server (e.g. ColabFold) for MSA generation |
| `--no_kernels` | off | Disable cuequivariance CUDA kernels (PyTorch-native triangle attention) |
| `--use_flex_attention` | off | Use FlexAttention for triangle attention (memory/throughput; may need chunked on large N) |
| `--use_flex_attention_chunked` | off | Chunked FlexAttention for DAP (avoids OOM on 25-sample hexamer; numerically matches original) |
| `--use_potentials` | off | Enable FK steering + physical guidance potentials |
| `--seed` | None | Random seed for reproducibility |

**Confidence** (pLDDT, pTM, iPTM, PAE, PDE) is always computed when the model supports it; no flag required.

For a full **prediction guide** (entrypoint, launch, input data, CLI options, pipeline stages), see [docs/boltz2_dap_prediction.md](docs/boltz2_dap_prediction.md).

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
├── docs/                            # Getting started, prediction guide
├── scripts/                         # Auxiliary Python scripts (compare, analyze, test, etc.)
├── slurm/                           # SLURM job scripts (.sbatch, .sh) for HPC runs
└── README.md
```

## Key Design Decisions

### Zero Boltz-2 Modifications

DAP **does not modify any original Boltz-2 source code**. Instead, it monkey-patches the model at runtime:

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

DAP produces results with minor floating-point differences from single-GPU Boltz-2, due to different operation ordering in distributed reductions. Structure predictions (LDDT, TM-score) are statistically equivalent.

## References
If you found this project useful, please cite:
- [Boltz-2](https://github.com/jwohlwend/boltz) — Base model
- [FastFold](https://github.com/hpcaitech/FastFold) — DAP communication primitives (adapted)
- [AlphaFold 3](https://github.com/google-deepmind/alphafold3) — Triangle operations architecture (adapted)

We would also appreciate it if you could cite this repository in any work that uses or builds upon it. A formal citation will be provided in a preprint describing our implementation, benchmarks, and results on our AAV multimer structure prediction with this approach.

## License

This DAP wrapper follows the same MIT license as Boltz-2.

## Further Advancement

For any inquiries, please email {gleeai, wjkimab}@connect.ust.hk, we would be happy to help with anything we could.

## Acknowledgements

We sincerely thank:
- the original Boltz-2 team for fully open-sourcing their state-of-the-art biomolecular structure prediction models,
- the FastFold team for their open-source distributed communication utilities,
- the AlphaFold 3 team for open-sourcing their inference code and model weights,
- the deep learning for biomolecular interaction modeling and the broader AI for Science communities for their ongoing contributions in this exciting field, and
- the developers and maintainers of all the packages used in this project!

This project was developed with generous compute support in HKUST HPC4 and SuperPOD from The Hong Kong University of Science and Technology (HKUST). This work was conducted at the lab of Prof. Bonnie Danqing Zhu in the Department of Chemical and Biological Engineering (CBE). 

We note the parallel development of [Fold-CP](https://github.com/NVIDIA-Digital-Bio/boltz-cp) by the team at NVIDIA Digital Bio, which also enables multi-GPU Boltz-2 inference (and also training) with a different approach. We look forward to comparing and learning from each other's implementations!