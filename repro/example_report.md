# Example Hexamer Reproduction Report

This is an example report generated from a successful hexamer reproduction run using:

- 4 GPUs
- `recycling_steps=10`
- `sampling_steps=200`
- `diffusion_samples=25`
- `--use_potentials`
- full PAE/PDE export enabled

## Top-Level Outputs

- `cif_count`: `25`
- `confidence_json_count`: `25`
- `full_pae_npz_count`: `25`
- `full_pde_npz_count`: `25`

## Top Model

- `model_0`
- `confidence_score`: `0.775005`
- `complex_plddt`: `0.797229`
- `iptm`: `0.686110`
- `complex_ipde`: `8.160539`

## Chain-Pair Summary

- Strongest pair by mean iPTM: `B-C`
  - mean iPTM: `0.761294`
  - mean PAE: `14.031643`
  - mean iPDE: `7.874321`
- Weakest pair by mean iPTM: `D-F`
  - mean iPTM: `0.540622`
  - mean PAE: `22.398532`
  - mean iPDE: `8.691650`

## Mean Chain-Pair Table

| Pair | mean iPTM | mean PAE | mean iPDE |
|------|-----------|----------|-----------|
| `B-C` | 0.761294 | 14.031643 | 7.874321 |
| `B-E` | 0.739180 | 14.808537 | 6.414675 |
| `B-D` | 0.736938 | 15.394539 | 7.041559 |
| `A-B` | 0.731333 | 14.347603 | 8.088197 |
| `A-D` | 0.726938 | 15.082121 | 6.659903 |
| `C-E` | 0.724514 | 15.715700 | 7.404843 |
| `C-F` | 0.718400 | 15.658672 | 7.239850 |
| `A-C` | 0.678069 | 15.803120 | 9.924179 |
| `A-F` | 0.669671 | 17.556960 | 9.294849 |
| `C-D` | 0.629618 | 19.659589 | 8.502178 |
| `B-F` | 0.610829 | 19.857050 | 8.426006 |
| `A-E` | 0.610331 | 19.679058 | 8.365817 |
| `D-E` | 0.557733 | 21.234803 | 7.560715 |
| `E-F` | 0.542672 | 21.913624 | 8.376318 |
| `D-F` | 0.540622 | 22.398532 | 8.691650 |

## GPU Peak Memory

- GPU 0: `68839 MB` (`67.2 GB`)
- GPU 1: `58523 MB` (`57.2 GB`)
- GPU 2: `58523 MB` (`57.2 GB`)
- GPU 3: `58523 MB` (`57.2 GB`)

## Pass Criteria

- 25 CIF files generated: `yes`
- 25 confidence JSON files generated: `yes`
- 25 full PAE files generated: `yes`
- 25 full PDE files generated: `yes`
- Chain-pair summary generated: `yes`
