# NVIDIA boltz-cp vs 우리 boltz_dap 비교

[NVIDIA-Digital-Bio/boltz-cp](https://github.com/NVIDIA-Digital-Bio/boltz-cp) (Fold-CP)와 우리 **boltz_dap**은 둘 다 **Boltz-2를 여러 GPU에 나눠서 큰 complex를 돌리기 위한** 프로젝트지만, 접근 방식과 지원 범위가 다릅니다.

---

## 1. 공통점 (비슷한 목표)

| 항목 | 공통 |
|------|------|
| **목표** | 단일 대형 protein complex를 여러 GPU에 나눠서 inference (OOM 방지, 확장) |
| **모델** | Boltz-2 |
| **실행** | `torchrun` / SLURM, multi-process |
| **Attention** | FlexAttention 지원 (그들: 공식 옵션, 우리: 패치) |

---

## 2. 병렬화 방식 (가장 큰 차이)

| 항목 | **NVIDIA boltz-cp** | **우리 boltz_dap** |
|------|----------------------|---------------------|
| **이름** | **Context Parallelism (CP)** | **Distributed Attention Parallelism (DAP)** |
| **구현** | PyTorch **DTensor** + **2D mesh** (cp 행×열) | **1D row sharding** + `gather`/`scatter` (torch.distributed) |
| **그리드** | `size_cp` = **완전제곱만** (4, 9, 16) → 2D CP mesh | **임의 GPU 수** (예: 4) → 1D로 pair 행만 분할 |
| **쉐딩** | 텐서를 2D로 나눠 각 rank가 일부만 보유 (DTensor) | pair 표현 **행**만 나눔: `z` [B, N/dap, N, D] per rank |
| **통신** | DTensor가 필요한 곳에서 자동 all-gather/reduce | Triangle attention 등에서 **bias만 gather** 후 다시 scatter (최소 통신) |

- **CP**: “context” 차원을 2D로 잘라서 DTensor로 분산. 공식 논문 [Fold-CP](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/fold_cp.pdf) 참고.
- **DAP**: pair의 **행**만 1D로 나누고, 연산 가능한 부분은 로컬, 필요한 부분만 gather해서 원본과 동일 수식 유지.

---

## 3. 입력·파이프라인

| 항목 | **boltz-cp** | **boltz_dap** |
|------|----------------|----------------|
| **입력** | **preprocessed 전용** (manifest.json, structures/, msa/, templates/ 등 디렉터리) | **YAML 설정 파일** (기존 Boltz와 동일), MSA 서버, template 등 |
| **전처리** | 미리 전처리된 데이터 필요 | `run_boltz_dap_v2.py`가 입력 YAML만 받아서 기존처럼 동작 |
| **Confidence** | **미지원** (`write_confidence_summary=False`) | **지원** (PAE, pLDDT, 25샘플 등) |
| **Potentials** | **미지원** | **지원** (`--use_potentials`) |
| **Template** | 가중치는 로드하지만 **분산 TemplateModule 미구현** | 지원 (DAP으로 처리) |
| **출력** | mmCIF/PDB (CP rank 0만 작성) | CIF 등 (confidence 포함, 25샘플 가능) |

우리 쪽이 **지금 당장 쓰는 관점**에서는 파이프라인이 더 완전함 (confidence, potentials, template, YAML 입력).

---

## 4. Attention / FlexAttention

| 항목 | **boltz-cp** | **boltz_dap** |
|------|----------------|----------------|
| **Triangle attention** | `--triattn_backend`: `cueq`, `trifast`, `reference` | 원본 Boltz TriangleAttention + **선택적으로** FlexAttention **패치** |
| **SDPA with bias** | `--sdpa_with_bias_backend`: `reference`, `torch_flex_attn` (ring-attention 등) | Pairformer 등은 기존 구현; Triangle만 FlexAttention 패치 |
| **Flex 사용 방식** | 옵션으로 **공식 통합** (backend 선택) | **Monkey-patch**로 TriangleAttention만 교체 (DAP 시 청크 경로 + 전체 [N,N] bias로 원본과 동일 결과) |

둘 다 FlexAttention을 쓰지만, CP는 배포용 백엔드 선택, 우리는 DAP과 맞추기 위해 패치로 삽입하는 구조.

---

## 5. 요구 사항·실행

| 항목 | **boltz-cp** | **boltz_dap** |
|------|----------------|----------------|
| **PyTorch** | 2.9+ | 2.x (현재 사용 버전에 맞춤) |
| **GPU 수** | 최소 4, `size_cp` 완전제곱 (4,9,16) | 2 이상 임의 (예: 4) |
| **실행 예** | `torchrun ... main.py predict /path/to/preprocessed --size_dp 1 --size_cp 4` | `torchrun ... run_boltz_dap_v2.py input.yaml --out_dir ... --use_flex_attention_chunked` |

---

## 6. 정리: 비슷한 점과 다른 점

- **비슷한 점**
  - Boltz-2 단일 대형 complex를 여러 GPU로 나눠서 돌리기.
  - FlexAttention 사용.
  - Multi-GPU로 메모리·확장성 확보.

- **다른 점**
  - **병렬 모델**: CP = 2D DTensor mesh, DAP = 1D row sharding + gather/scatter.
  - **완성도**: CP는 PoC 수준, confidence/potentials/template 분산 미지원; DAP은 confidence·potentials·25샘플·CIF까지 지원.
  - **입력**: CP = preprocessed 전용, DAP = YAML 등 기존 Boltz와 동일한 입력.
  - **유연성**: CP = GPU 수 완전제곱, DAP = GPU 수 자유.

NVIDIA는 [공식 Boltz에 CP를 올리기 위한 Draft PR](https://github.com/jwohlwend/boltz/pull/658)을 진행 중이므로, 나중에는 upstream에 CP가 들어가고, 우리 DAP은 **1D sharding + 전체 파이프라인(confidence, potentials)** 에 특화된 대안으로 남는 구도라고 보면 됩니다.
