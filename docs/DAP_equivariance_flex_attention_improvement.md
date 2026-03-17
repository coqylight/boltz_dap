# DAP Boltz2 개선 방안: Equivariance Kernel & Flex Attention

## 현재 상태 요약

| 구간 | Equivariance kernel | Flex Attention |
|------|---------------------|----------------|
| Template PF | **강제 False** (dap_trunk `_tmpl_use_kernels=False`) | 미적용 |
| MSA PF | **강제 False** (dap_msa `_msa_use_kernels=False`) | 미적용 |
| Trunk PF | `model.use_kernels` 전달하나, **TriMul은 DAP 전용 einsum 경로**만 사용 (kernel 미호출) | **미적용** (patch 호출 없음) |
| Confidence PF | `model.use_kernels` 전달, chunk 입력으로 kernel 동작 불명 | 미적용 |

- **TriMul (Triangle Multiplication)**: DAP 경로에서는 `DAPTriMulOut` / `DAPTriMulIn`이 **전용 broadcast-chunked einsum**만 사용하고, `inner(..., use_kernels=True)`는 호출하지 않음 → cuequivariance kernel 미사용.
- **TriAtt (Triangle Attention)**: DAP이 chunk된 `[B, N/dap, N, D]`를 넘기고, cuequivariance kernel은 보통 전체 `[B, N, N, D]` 가정 → chunk 지원 여부 불명이면 kernel 사용 위험.

---

## 1. Flex Attention 활용 방안

### 1.1 현황

- `boltz_dap_v2/flex_attention_patch.py`가 이미 존재하며, **Triangle Attention의 `mha.forward`를 FlexAttention으로 교체**하는 패치를 제공.
- **chunked(4D)** / **non-chunked(5D)** 두 경로 모두 구현되어 있어, DAP에서 chunk 단위로 호출되는 attention과 형태가 맞음.
- 장점: full QK^T 행렬을 만들지 않아 **메모리 절감**, Triton 기반 `torch.compile(flex_attention)`로 **속도 개선 가능**, PyTorch 기본 제공으로 GPU 간 동작 일관.

### 1.2 적용 방법

- **run_boltz_dap_v2.py**에서 모델 로드 직후, **DAP wrapper 주입 전**에 한 번만 패치 적용:

```python
# Model 로드 직후 (inject_dap_into_model 전)
try:
    from flex_attention_patch import patch_triangle_attention
    n_patched = patch_triangle_attention(model)
    rank_print(f"  ✓ FlexAttention patched onto {n_patched} TriangleAttention layers")
except Exception as e:
    rank_print(f"  ⚠ FlexAttention patch skipped: {e}")
```

- DAP 주입 시 `DAPTriAttStart` / `DAPTriAttEnd`가 `self.inner.mha(...)`를 호출하므로, 위 패치가 적용된 `inner`가 그대로 FlexAttention 경로를 타게 됨.
- **옵션**: CLI에 `--use_flex_attention` 플래그를 두고, 켜진 경우에만 `patch_triangle_attention` 호출하면 기존 동작과 A/B 테스트 가능.

### 1.3 검증

- `test_flex_vs_sdpa.py`로 수치 일치·메모리·시간 비교 가능. DAP과 함께 쓸 때는 hexamer/trimer 1회 inference로 OOM/정확도 한 번 더 확인 권장.

---

## 2. Equivariance Kernel 활용 방안

### 2.1 TriMul (Triangle Multiplication)

- **현재**: DAP은 `DAPTriMulOut` / `DAPTriMulIn`에서 **행/열 chunk별로 broadcast + einsum**만 수행. `inner`(cuequivariance `triangle_multiplicative_update`)는 **호출하지 않음**.
- **cuequivariance**의 triangle multiplicative update는 보통 **row-wise / column-wise**로 동작하므로, 이론상 **한 번에 [B, chunk_N, N, D]만 넘겨도** 동작 가능할 수 있음 (해당 chunk 행/열만 처리).
- **제안**:
  1. **cuequivariance_torch** 문서/소스에서 `triangle_multiplicative_update`가 **chunk된 행/열 입력**을 지원하는지 확인.
  2. 지원하면: `DAPTriMulOut`에서 `use_kernels=True`일 때 `inner(x, mask, use_kernels=True)`를 호출하는 경로 추가 (현재는 항상 수동 einsum). 단, `x`가 이미 `[B, N/dap, N, D]`이므로, kernel이 “일부 행만” 받는 것을 허용하는지가 관건.
  3. 지원하지 않으면: upstream에 chunked API 요청하거나, **한 rank에서만 gather 후 kernel 호출 → scatter** 방식은 메모리 부담이 커서 비추천.

### 2.2 TriAtt (Triangle Attention)

- **현재**: DAP은 `DAPTriAttStart` / `DAPTriAttEnd`에서 **bias만 gather**하고, attention 연산은 chunk된 `x`로 `inner.mha(...)` 호출. 여기서 `use_kernels=True`를 넘기면 `kernel_triangular_attn`이 호출됨.
- **문제**: `kernel_triangular_attn`이 **전체 [B, N, N, D]** 형태를 가정하면, chunk된 입력으로는 잘못된 결과나 크래시 가능.
- **제안**:
  1. cuequivariance의 triangle attention이 **chunked Q/K/V**를 지원하는지 문서/소스 확인.
  2. 지원하지 않으면 DAP에서는 **Triangle Attention에 한해 use_kernels=False 유지**가 안전. (이미 Template/MSA는 강제 False.)
  3. 지원하면: DAP 경로에서 `use_kernels=True` 전달 시 해당 chunked API 사용하도록 분기 추가.

### 2.3 Template / MSA에서 kernel 켜기

- 현재는 **의도적으로** PyTorch-native만 사용해 “1-GPU와 2+-GPU가 동일 코드 경로”로 맞춰 둔 상태.
- Equivariance kernel을 여기서 다시 켜려면:
  - 해당 구간에서 **chunked 입력**에 대한 kernel 지원이 있어야 하고,
  - **수치 일치 검증** (기존 PyTorch-native vs kernel)을 DAP 설정으로 다시 해야 함.
- 우선순위는 **Trunk / Confidence**에서의 kernel 활용이 더 효과가 클 가능성이 큼.

---

## 3. 권장 순서

1. **Flex Attention**
   - `run_boltz_dap_v2.py`에 `patch_triangle_attention(model)` 호출(옵션 플래그 권장) 추가.
   - trimer/hexamer 1회씩 돌려서 메모리·속도·정확도 확인 후, 문제 없으면 기본 활성화.

2. **Equivariance – TriMul**
   - cuequivariance_torch의 triangle multiplicative update가 **row/column chunk** 입력을 지원하는지 확인.
   - 지원하면 `DAPTriMulOut`/`DAPTriMulIn`에 `use_kernels=True` 분기 추가 후, 단일 GPU·다중 GPU 비교 검증.

3. **Equivariance – TriAtt**
   - triangle attention의 **chunked 지원** 여부 확인.
   - 지원 시에만 DAP에서 `use_kernels=True` 전달; 미지원이면 DAP은 계속 False 유지.

4. **Template / MSA**
   - Trunk/Confidence에서 kernel이 안정된 뒤, 필요하면 Template/MSA에 chunked kernel 지원 여부 확인 후 검토.

---

## 4. 요약

- **Flex Attention**: 이미 chunked/5D 대응 패치가 있으므로, **run에서 패치만 호출**하면 DAP과 함께 사용 가능. 메모리·속도 개선 기대.
- **Equivariance**: DAP은 **chunk 단위 연산**이라, cuequivariance가 **chunked 입력**을 지원하는지가 전제. TriMul은 chunk 단위 호출 가능성이 있고, TriAtt은 문서 확인 필요. 지원 확인 후 TriMul → TriAtt 순으로 DAP 경로에 `use_kernels=True`를 조건부로 넣는 방식이 현실적.
