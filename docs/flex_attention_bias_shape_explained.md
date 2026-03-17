# FlexAttention 청크 경로 vs 원본 DAP: bias shape 차이 설명

## 질문

> "원본 DAP는 full_bias [N,N]를 써서 원본과 같고, 우리 FlexAttention 청크는 앞쪽 C행 [C,J]만 있어서 q_idx ∈ [C,J)는 0으로 채우고, 그 구간 출력이 달라진다" — 이 부분이 잘 이해가 안 된다.

---

## 1. Attention에서 쓰는 shape 한 줄 정리

- **입력**: 이 rank의 pair representation 일부 → `q` shape `[B, I, H, J, c_h]` (I = 이 rank 행 개수, J = 전체 열 개수 = N).
- **Score**: `score[b, h, q_idx, kv_idx]` = (query 위치 `q_idx`, key 위치 `kv_idx`)에 대한 attention logit.
- **Bias**: 이 score에 더해지는 값. 원래 수식은  
  `score[b,h,q_idx,kv_idx] += bias[b,h,q_idx,kv_idx]`  
  이므로, **모든 (q_idx, kv_idx) 쌍**에 대해 bias가 하나씩 필요하다.
- 따라서 **한 배치 원소당 bias shape는 [J, J] (= [N, N])** 이어야 한다.  
  즉, "q_idx ∈ [0, J), kv_idx ∈ [0, J)" 전부에 대해 값이 있어야 원본과 동일하게 계산할 수 있다.

---

## 2. 원본 DAP 경로 (dap_tri_att.py) — 왜 “전체 [N,N] bias”를 쓰는지

1. **Gather로 full_bias 생성**
   - 각 rank는 자기 행만으로 `local_bias`를 만들고, **gather**해서 **전체 N×N**을 만든다.
   - 그래서 `full_bias` shape는 `[B, 1, H, N, N]` 이다.  
     → **모든 (row, col) = (0..N-1, 0..N-1)** 에 대한 값이 다 들어 있다.

2. **chunk_layer에 그대로 전달**
   - `chunk_layer(..., mha_inputs={"q_x": x, "tri_bias": full_bias, ...})`
   - `chunk_layer`는 **앞쪽 batch 차원만** 잘라서 레이어에 넘긴다.  
     (예: `no_batch_dims=2`면 `(B, I)`만 쪼갬.)
   - `tri_bias`는 `[B, 1, H, N, N]` 이므로, 잘리는 것은 `(B, I)` 뿐이고 **마지막 두 차원 (N, N)은 그대로** 유지된다.
   - 따라서 한 청크에 들어가는 `tri_bias` shape는  
     `[chunk_size, 1, H, N, N]`  
     → 청크 안의 **각 배치 원소마다 [N, N] 전체**를 갖고 있다.

3. **그래서 원본과 같다**
   - 내부 `Attention.forward`는 `score [..., H, J, J]`에 `bias [..., H, J, J]`를 더한다.
   - 여기서 J = N 이고, bias가 **정확히 [N, N]** 이므로  
     `q_idx ∈ [0, N)`, `kv_idx ∈ [0, N)` 전부에 대해 올바른 bias가 들어가고,  
     **계산 결과가 원본 Boltz2와 동일**해진다.

요약: **원본 DAP는 “청크를 나눠도, 각 청크에 들어가는 tri_bias가 항상 [N, N] 전체”**이기 때문에,  
“전체 [N, N] bias를 사용한다”고 말하는 것이고, 그래서 원본과 같은 결과가 나온다.

---

## 3. 우리 FlexAttention 청크 경로 — 왜 “[C,J]만 있고, q_idx ∈ [C,J)는 0”이 되는지

1. **같은 full_bias를 받지만, 우리는 “행만” 잘라 쓴다**
   - 패치된 `mha.forward`도 `tri_bias`로 **같은 full_bias `[B, 1, H, N, N]`** 를 받을 수 있다.
   - 하지만 우리는 **메모리/연산을 줄이려고**  
     “이 rank의 행”을 또 **C개 단위 서브청크**로 나눠서 FlexAttention을 여러 번 호출한다.
   - 그때 우리가 쓰는 slice는:
     - `tri_c = tri_bias[:, :, :, row_start+start : row_start+end, :]`
     - 즉 **행만** `row_start+start` ~ `row_start+end` (길이 C)로 자른다.
   - 따라서 **한 서브청크에서 실제로 쓰는 bias shape는 `[B, 1, H, C, J]`**  
     → 배치/헤드 합치면 **실제 있는 건 “C개 행 × J개 열” = [C, J]** 뿐이다.

2. **FlexAttention이 요구하는 shape**
   - 한 번 호출할 때 우리가 넘기는 `q_c` shape는 `[B*C, H, J, c_h]`  
     → score shape는 `[B*C, H, J, J]` 이고,  
     **score_mod(b, h, q_idx, kv_idx)** 는 **q_idx ∈ [0, J), kv_idx ∈ [0, J)** 전부에 대해 불린다.
   - 따라서 “원본과 동일하게” 하려면  
     **bias도 [B*C, H, J, J]** 형태로,  
     **모든 (q_idx, kv_idx)** 에 대해 값이 있어야 한다.

3. **우리가 가진 것과 필요한 것**
   - **가진 것**: `[B*C, H, C, J]` — 즉 **q_idx ∈ [0, C)** 에 대해서만** bias 값이 있음.
   - **필요한 것**: `[B*C, H, J, J]` — **q_idx ∈ [0, J)** 전부.
   - 그래서 **q_idx ∈ [C, J)** 구간에는 “원본에서 쓰여야 할 bias 값”을 우리는 전혀 갖고 있지 않다.  
     (서브청크에서 행을 C개만 잘라왔기 때문.)

4. **그래서 0으로 채운다 (현재 패치)**
   - 인덱스 OOB를 피하려면 `score_mod`에 넘기는 bias를 `[B*C, H, J, J]`로 만들어야 하고,
   - 우리가 아는 건 앞쪽 C개 행뿐이므로,  
     **앞쪽 C개 행만** `_tri + _mask`로 채우고,  
     **q_idx ∈ [C, J)** 구간은 **0으로 채운다**.
   - 즉:
     - **q_idx ∈ [0, C)** → 원본과 같은 bias 사용 → **이 구간 출력은 원본과 같다.**
     - **q_idx ∈ [C, J)** → 원본은 여기서 0이 아닌 bias를 쓰는데, 우리는 0을 씀 → **이 구간 출력은 원본과 다르다.**

그래서  
“우리 FlexAttention 청크 경로는 **이 rank의 서브청크에 대해서만 앞쪽 C행 [C, J]** 만 갖고 있어서,  
**q_idx ∈ [C, J)에 해당하는 bias 값을 알 수 없고**, 0으로 채울 수밖에 없고,  
→ **그 구간의 attention 출력이 원본과 달라진다**”  
가 맞는 설명이다.

---

## 4. 그림으로 보면

```
원본 DAP (chunk_layer):
  full_bias [B, 1, H, N, N]
       ↓ chunk_layer는 (B,I)만 자름 → (N,N)은 그대로
  한 청크의 tri_bias: [chunk_size, 1, H, N, N]
  → 각 배치 원소가 [N,N] 전체 사용 → 원본과 동일

우리 FlexAttention 청크:
  같은 full_bias [B, 1, H, N, N]을 받지만
  행만 자름: tri_c = full_bias[:,:,:, row_start+start:row_start+end, :]
       ↓
  [B, 1, H, C, J]  (C개 행, J개 열만 있음)
  → score는 [B*C, H, J, J] 인데, bias는 [B*C, H, C, J]만 있음
  → q_idx ∈ [C, J) 구간은 “원본에서 쓰는 bias”를 모름 → 0으로 채움
  → 그 구간 출력이 원본과 다름
```

---

## 5. 한 줄 요약

- **원본 DAP**: chunk_layer가 **batch 차원만** 자르기 때문에, **tri_bias의 마지막 두 차원 [N, N]은 항상 통째로** 각 청크에 들어가서, “청크마다 전체 [N, N] bias”를 쓴다 → 원본과 동일.
- **우리 FlexAttention 청크 (이전)**: 행을 C개만 잘라 쓰면 bias는 [C, J]뿐이라 q_idx ∈ [C, J)는 0으로 채워지고 그 구간만 원본과 달라짐.
- **우리 FlexAttention 청크 (개선)**: DAP처럼 **청크마다 전체 [N, N] bias**를 쓰면 된다. `tri_bias [B, 1, H, N, N]`를 서브청크의 C개 배치에 expand해 `[B*C, H, J, J]`로 두고 mask만 더하면 원본과 동일. 트레이드오프는 메모리: bias가 [B*C, H, J, J]로 커짐 (예: C=32, J=3114 → 약 4.6 GB).
