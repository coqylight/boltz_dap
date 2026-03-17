# cuequivariance chunked 행/열 지원 여부 — 검색 결과 요약

## 검색 범위

- NVIDIA cuEquivariance 공식 문서 (triangle_multiplicative_update, triangle_attention)
- 릴리즈 노트, GitHub 이슈
- "chunk", "leading batch dimensions", "partial rows" 관련 설명

---

## 1. triangle_multiplicative_update

**문서**: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_multiplicative_update.html

- **입력 `x`**: 공식 명시 shape = **(B, N, N, D)**
  - B = batch size  
  - N = sequence length  
  - D = hidden dimension  
- **반환**: **(batch_size, seq_len, seq_len, hidden_dim)** → 역시 전체 N×N.
- **Notes**  
  - "the API supports **variable number of leading batch dimensions**"  
  - 즉 **(B1, B2, ..., N, N, D)** 처럼 **앞쪽에 배치 차원만** 더 둘 수 있다는 의미.  
  - **두 번째·세 번째 차원이 N이 아닌 chunk_N**(예: 행만 쪼갠 (B, chunk_N, N, D))을 지원한다는 내용은 **없음**.

**결론**:  
- 공식 API는 **(..., N, N, D)** 형태만 정의.  
- **chunk된 행/열 (B, chunk_N, N, D) 또는 (B, N, chunk_N, D)에 대한 지원은 문서에 없음.**  
- 따라서 **chunk된 행/열 입력은 공식 지원하지 않는 것으로 보는 것이 맞음.**

---

## 2. triangle_attention

**문서**: https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html

- **q**: (B, N, H, Q, D) 또는 B=1일 때 (N, H, Q, D)  
- **k, v**: (B, N, H, K, D)  
- **bias**: (B, 1, H, Q, K)  
- **mask**: (B, N, 1, 1, K)  
- 예제: `seq_len=128`로 **Q, K 모두 전체 시퀀스 길이** 사용.
- **Notes**: Blackwell 등 최적화에서 "sequence length N must be a multiple of 8" 등 **전체 N 기준** 설명만 있음.

**결론**:  
- **Q, K 모두 전체 시퀀스(전체 N)** 를 가정한 API.  
- **chunk된 Q 또는 K**(예: query만 chunk_N개)를 지원한다는 기술은 **문서에 없음.**

---

## 3. "Variable leading batch dimensions" (릴리즈 노트 0.6.1)

- "Variable number of **leading** batch dimensions" 지원 추가.
- 의미: **(B1, B2, ..., N, N, D)** 처럼 **앞쪽 배치 차원**을 여러 개 둘 수 있음.
- **행/열 차원 N을 chunk_N으로 줄이는 것**과는 무관.

---

## 4. 종합

| 항목 | 공식 문서 상 shape | chunk된 행/열 지원 여부 |
|------|---------------------|--------------------------|
| **triangle_multiplicative_update** | x: (B, N, N, D) | **미언급 → 지원 안 하는 것으로 해석** |
| **triangle_attention** | q/k/v: (B, N, H, Q/K, D), Q·K = 전체 N | **미언급 → 지원 안 하는 것으로 해석** |

- **정리**: 현재 공식 문서와 예제만 보면, cuequivariance는 **전체 N×N(또는 전체 Q×K)** 를 요구하는 API로만 설명되어 있고, **chunk된 행/열을 받는 모드는 문서상으로는 지원하지 않는 것**으로 보는 것이 타당함.
- DAP에서 **행을 쪼갠 (B, chunk_N, N, D)** 를 그대로 넘기는 것은 공식 사용법이 아니므로, **지금처럼 DAP에서는 equivariance kernel을 끄고 PyTorch-native(또는 FlexAttention)만 쓰는 전략이 문서와 맞음.**

추후 NVIDIA에 chunked API 지원 여부를 직접 문의하거나, 소스/릴리즈 노트에서 "chunk" / "partial" / "row slice" 관련 설명이 나오면 이 문서를 갱신하면 됨.

---

## 5. 타 GitHub / Hugging Face에서 chunked 행·열을 cuequivariance로 처리한 사례 검색

### 검색 쿼리

- `cuequivariance chunked row column`, `triangle_multiplicative_update chunk N/dap shard`
- `OpenFold Boltz cuequivariance use_kernels chunk`, `FastFold DAP cuequivariance`
- `AlphaFold2 triangular attention chunk_layer use_kernels`, Hugging Face boltz cuequivariance

### 결과 요약

- **NVIDIA/cuEquivariance**  
  - 이슈/README에는 chunk된 행·열을 kernel에 넘기는 사용 예나 API 설명 없음.  
  - TorchInductor stride 이슈, backward 비연속 텐서 이슈 등만 문서화.

- **OpenFold3 (openfold-3.readthedocs.io)**  
  - `triangle_multiplicative_update` / `triangle_attention`에 cuequivariance 사용.  
  - 짧은 시퀀스(`CUEQ_TRIATTN_FALLBACK_THRESHOLD`)나 hidden_dim > 128이면 **DeepSpeed/PyTorch로 fallback**.  
  - **“chunk된 입력을 cuequivariance에 넣는다”** 기술은 없음. 지원되는 shape면 전체 텐서로 kernel 호출, 아니면 fallback.

- **jwohlwend/boltz PR #43 (Added chunking for larger crops inference)**  
  - PairFormer triangular attention, MSA transition 등에 **chunking 추가** (crop > 512 토큰 시 자동).  
  - 메모리 절감용 **chunk_layer 스타일**로 동작.  
  - PR/코드에서 **cuequivariance를 chunked 입력으로 호출**한다는 언급 없음.  
  - 일반적으로 이런 chunking은 **행 단위로 나눈 뒤 PyTorch 경로(use_kernels=False)** 로 한 chunk씩 호출하는 패턴.

- **FastFold / JAX**  
  - “multi-device JAX에서는 triangle attention에 `shard_map` 등으로 manual sharding이 필요하다”는 식의 설명 있음.  
  - **kernel 자체가 chunk/shard 입력을 받는지**에 대한 구체적 사례는 검색에서 나오지 않음.

- **Hugging Face boltz-community/boltz-2**  
  - Boltz2 체크포인트/모델만 있고, cuequivariance와 chunked 연동에 대한 문서나 코드 예시는 검색되지 않음.

### 결론 (타 repo 기준)

- **공개된 GitHub/Hugging Face 사례 중에서는 “chunk된 행/열만 cuequivariance kernel에 넘겨서 처리했다”고 볼 만한 예는 없음.**  
- OpenFold3·Boltz 모두:
  - cuequivariance는 **지원되는 전체 shape**에 대해 사용하고,
  - 그렇지 않거나 메모리 절약이 필요하면 **chunking + PyTorch(또는 다른) fallback**을 쓰는 구조로 보임.  
- 따라서 **chunked row/column을 cuequivariance로 처리한 선례는 찾지 못한 상태**로 정리하는 것이 맞음.
