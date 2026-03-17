# Triangle Attention 구현이 어떻게 돌아가는지 (쉬운 설명)

## 1. "SDPA를 쓰나?" → **아니요**

- **SDPA** = PyTorch의 `torch.nn.functional.scaled_dot_product_attention` (그리고 그 뒤에 붙는 cuDNN/Flash 등 백엔드).
- Boltz2의 **Triangle Attention**은 이걸 전혀 호출하지 않습니다.
- 대신 Boltz2 **전용** 구현이 두 가지 있고, 그중 하나가 선택됩니다.

---

## 2. Triangle Attention이 실제로 호출하는 것

Boltz2에서 Triangle Attention은 한 개의 **공통 진입점**만 있습니다.

- 모듈: `TriangleAttention` (Starting / Ending 노드)
- 그 안의 **실제 attention 계산**은 전부 `self.mha` 한 군데에서 이뤄집니다.
- `self.mha` = `Attention` 클래스 (primitives.py).

그래서 “지금 SDPA 쓰냐?”라고 묻는 건, 사실 **“그 `Attention`(self.mha) 안에서 뭘 쓰냐?”**를 묻는 것과 같습니다.

---

## 3. `Attention.forward()` 안의 두 갈래 (코드 그대로)

`primitives.py`의 `Attention.forward()`는 대략 이렇게 생겼습니다:

```python
def forward(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
    q, k, v = self._prep_qkv(q_x, kv_x, ...)   # Q, K, V 프로젝션

    if use_kernels:
        # 갈래 ①: cuequivariance CUDA 커널
        o = kernel_triangular_attn(q, k, v, tri_bias=tri_bias, mask=mask, scale=...)
    else:
        # 갈래 ②: PyTorch로만 구현한 "naive" 경로
        biases = [mask_bias, tri_bias]
        o = _attention(q, k, v, biases)

    o = self._wrap_up(o, q_x)
    return o
```

즉, **항상** 다음 둘 중 하나만 사용합니다.

- **갈래 ①** `use_kernels=True` → `kernel_triangular_attn` (cuequivariance)
- **갈래 ②** `use_kernels=False` → `_attention` (PyTorch만 사용)

**SDPA는 이 두 갈래 어디에도 없습니다.**

---

## 4. 갈래 ② `_attention`이 하는 일 (지금 DAP이 쓰는 경로)

`_attention` 함수는 이렇게 생겼습니다:

```python
def _attention(query, key, value, biases):
    # query: [..., H, Q, C],  key: [..., H, K, C]
    key = permute_final_dims(key, (1, 0))   # [..., H, C, K]

    # ★ 여기서 (Q×C) @ (C×K) = (Q×K) 행렬을 통째로 만든다
    a = torch.matmul(query, key)            # [..., H, Q, K]

    for b in biases:
        a += b
    a = softmax_no_cast(a, -1)

    a = torch.matmul(a, value)              # [..., H, Q, C]
    return a
```

- **중요한 점**: `torch.matmul(query, key)` 때문에 **Q×K 크기의 attention 행렬을 한 번에 만듭니다.**
- N이 크면 이 행렬이 매우 커지고, 그래서 메모리를 많이 씁니다.
- 이걸 “**naive**” 구현이라고 부르는 이유는, “최적화된 attention 커널” 없이 **일반적인 행렬 곱 + bias + softmax**만 쓰기 때문입니다.
- PyTorch **기본 연산**만 쓴다는 의미이지, “PyTorch 공식 SDPA API”를 쓴다는 뜻은 아닙니다.

정리하면:

- **SDPA** = PyTorch가 제공하는 **또 다른** attention API (Flash/cuDNN 등 백엔드).
- **지금 여기** = 그 API가 아니라, **Boltz2가 직접 만든** `_attention` (matmul + bias + softmax).

---

## 5. 갈래 ① cuequivariance 커널

- `use_kernels=True`일 때만 들어가는 경로입니다.
- `kernel_triangular_attn` → 내부에서 `cuequivariance_torch.primitives.triangle.triangle_attention` 호출.
- 이건 **전용 CUDA 커널**이라, 보통은 full Q×K 행렬을 메모리에 안 올리고 더 효율적으로 계산합니다.
- 대신 **전체 시퀀스(전체 N)** 를 한 번에 받는 걸 가정한 구현일 가능성이 큽니다.
- DAP에서는 **행을 쪼갠 chunk**만 넘기기 때문에, 이 커널을 그대로 쓰기엔 설계가 안 맞아서, 지금은 Template/MSA에서 `use_kernels=False`로 강제하고, Trunk/Confidence도 사실상 이 “naive” 쪽으로 타는 상황입니다.

---

## 6. Flex Attention 패치가 바꾸는 것

- Flex Attention 패치는 `**Attention.forward` 자체를 교체**합니다.
- 즉, 위에서 말한 **두 갈래(① kernel_triangular_attn / ② _attention)를 타기 전에**,  
“Triangle Attention용 **세 번째 구현**”으로 들어가게 만듭니다.
- 그 세 번째 구현에서는:
  - **full Q×K 행렬을 만들지 않고**,
  - PyTorch의 `flex_attention`(Triton 기반)을 쓰면서,
  - `score_mod`로 bias만 넣어 주는 방식으로 같은 수식(attention + triangular bias)을 구현합니다.
- 그래서 “**SDPA**로 바꾼다”가 아니라, “**naive _attention (full QK^T)** 대신 **FlexAttention**을 쓰게 한다”라고 보면 됩니다.

---

## 7. 한 줄로 정리


| 이름                         | 뭔가?                                   | Boltz2 Triangle Attention에서 쓰나?                               |
| -------------------------- | ------------------------------------- | ------------------------------------------------------------- |
| **SDPA**                   | PyTorch 기본 제공 attention API (Flash 등) | **안 씀**                                                       |
| **_attention (naive)**     | Boltz2가 직접 만든 구현 (matmul로 Q×K 행렬 생성)  | **지금 DAP이 쓰는 경로**                                             |
| **kernel_triangular_attn** | cuequivariance 전용 CUDA 커널             | DAP에서는 거의 안 씀 (Template/MSA는 꺼짐, Trunk/Confidence는 chunk 때문에) |
| **FlexAttention (패치)**     | full Q×K 없이 attention 계산              | `--use_flex_attention` 켜면 **이걸로 대체**                          |


그래서 “지금은 SDPA만 쓰나?”가 아니라,  
**“지금은 Boltz2 전용 naive 구현(_attention)만 쓰고, Flex 패치를 켜면 그걸 FlexAttention으로 바꾼다”**가 맞습니다.