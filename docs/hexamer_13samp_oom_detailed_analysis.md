# Hexamer 13 samples OOM 상세 분석

## 1. 로그 요약 (Job 354019)

- **설정**: diffusion_samples=13, 4 GPU DAP, run_confidence_sequentially=True
- **결과**: 13개 샘플에 대한 **confidence는 모두 완료** (call_idx=0 ~ 12, PDE done까지 진행)
- **실패 시점**: 13번째 샘플 confidence 직후, `[TIMELINE] ... after confidence_module` 로그 다음
- **에러 메시지**: `| WARNING: ran out of memory, skipping batch` (boltz2.py predict_step의 except)
- **최종 GPU 0 Peak**: 66175 MB (64.6 GB) — 80GB 미만이지만 OOM 발생

## 2. 실패 직전 로그 순서

```
[CONF R0] ... PDE done                    ← 13번째 샘플 confidence 완료
GPU 1/2/3: Memory after forward / Peak
[TIMELINE] 4998.4s | alloc= 9503MB | peak= 25568MB | after confidence_module
| WARNING: ran out of memory, skipping batch
GPU 0: Memory after forward: 5513 MB
GPU 0: Peak memory: 47881 MB (46.8 GB)
✗ OOM during inference!
```

즉, **confidence 13회는 모두 끝난 뒤**, 그 다음 단계에서 OOM이 난다.

## 3. 12samp vs 13samp 비교

| 항목 | 12samp (354018, 성공) | 13samp (354019, 실패) |
|------|------------------------|------------------------|
| GPU 0 최종 Peak | 77623 MB (75.8 GB) | 66175 MB (64.6 GB) |
| Confidence 완료 | 12회 모두 완료 | 13회 모두 완료 |
| CIF 생성 | ✓ | ✗ (OOM으로 중단) |

13samp가 12samp보다 **피크 메모리 수치는 더 낮음**. 따라서 “13회가 더 많이 써서”가 아니라, **confidence 직후에 하는 일**에서 OOM이 난다.

## 4. 근본 원인: confidence 결과를 전부 GPU로 올리기

### 4.1 코드 경로

1. **dap_confidence.py**  
   - Rank 0에서 13개 샘플을 순차 실행하고, 결과를 **CPU** `merged` 버퍼에만 쌓음.  
   - `merged[key]` 예: `(13, N, N, D)` (pde_logits 등), device=`"cpu"`.  
   - 따라서 **confidence 구간 자체는 GPU에 13개를 동시에 두지 않음**.

2. **dap_trunk.py** (1012–1038행 근처)  
   - `run_confidence_dap()` 반환값을 `dict_out.update(confidence_output)` 로 넣음.  
   - 그 다음:
     ```python
     # Move any CPU-offloaded tensors back to GPU for writer
     for key in list(dict_out.keys()):
         if isinstance(dict_out[key], torch.Tensor) and not dict_out[key].is_cuda:
             dict_out[key] = dict_out[key].cuda(0)
     ```
   - 즉, **CPU에 있던 confidence 결과(pde, pae, plddt 등)를 전부 GPU 0으로 올린다.**

### 4.2 왜 13samp에서만 OOM인가

- `confidence_output`에는 **merged** 형태의 큰 텐서가 들어 있음.  
  - 예: pde_logits `(13, 3114, 3114, 64)` float32  
  - 13 × 3114 × 3114 × 4 ≈ **약 1.5GB** (이것만으로는 크지 않지만, pae, plddt, pair_chains_iptm 등 **여러 키**가 있고, 전체 합이 큼).
- 12samp: `(12, N, N, D)` 등 → GPU로 올린 뒤에도 80GB 안에 들어가거나 단편화가 덜함.
- 13samp: `(13, N, N, D)` 등 → **한 번에 GPU로 올리면**  
  - 필요한 연속 블록이 커지거나  
  - 이미 사용 중인 GPU 메모리(모델, 중간 버퍼)와 합쳐져 80GB를 넘거나  
  - 단편화로 인해 그만큼의 연속 할당이 실패함.

그래서 **“13회 돌리는 구간”이 아니라, “13개짜리 merged를 전부 .cuda(0) 하는 구간”**에서 OOM이 발생한다.

## 5. 요약

| 항목 | 내용 |
|------|------|
| **실패 위치** | `dap_trunk.py`: `dict_out.update(confidence_output)` 직후, `dict_out[key].cuda(0)` 루프 |
| **원인** | 13개 샘플에 대한 confidence 결과(merged, CPU)를 **전부 GPU 0으로 이동**하면서 할당 실패 또는 피크 초과 |
| **12samp는 되는 이유** | 12개 분량은 같은 코드 경로로 GPU에 올려도 한계 내에 들어가거나 단편화가 덜함 |
| **13samp가 더 피크가 낮게 찍힌 이유** | OOM이 나서 그 시점에서 중단되었고, 12samp는 그 다음 단계(CIF 등)까지 진행해 더 높은 피크가 기록됨 |

## 6. 권장 수정

- **confidence 결과는 writer가 GPU를 요구하지 않는 한 GPU로 옮기지 않기**  
  - `dap_trunk.py`에서 “Move any CPU-offloaded tensors back to GPU for writer” 루프에 **예외** 추가:  
    - `key`가 `pde`, `pae`, `plddt`, `complex_pde`, `complex_pae`, `pae_logits`, `pde_logits` 등 **큰 confidence 텐서**이면 ` .cuda(0)` 하지 않고 CPU에 유지.
- 또는 **writer가 CPU 텐서를 받도록 수정**하고, 위 루프에서 confidence 관련 키는 제외.
- 이렇게 하면 13samp도 12samp와 동일하게 “confidence는 CPU에만 유지, 이후 단계만 GPU 사용”이 되어 OOM을 피할 수 있음.

## 7. 참고: OOM이 잡히는 위치

- **boltz2.py** `predict_step()` (1056–1128행):  
  `self(batch, ...)` 호출 중 `RuntimeError` (e.g. `out of memory`) 발생 시  
  `"| WARNING: ran out of memory, skipping batch"` 출력 후 `return {"exception": True}`.  
- 따라서 스택 상으로는 **forward → confidence 완료 → dict_out.update → .cuda(0) 루프** 중 한 지점에서 OOM이 나고, 그 예외가 `predict_step`까지 전파된 것이다.
