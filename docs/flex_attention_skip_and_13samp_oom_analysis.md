# FlexAttention 스킵 원인 및 Hexamer 13samp OOM 분석

## 1. FlexAttention이 적용되지 않은 이유 ("duplicate template name")

### 1.1 확인된 사실

- **354018 (hexamer 12samp), 354019 (hexamer 13samp), 354017 (trimer)** 로그 모두 동일 메시지:
  - `⚠ FlexAttention patch skipped: duplicate template name`
- 즉 `--use_flex_attention`을 켜도 **패치는 예외로 인해 한 번도 적용되지 않음**.

### 1.2 예외가 나는 위치

- `run_boltz_dap_v2.py` 280~287행:
  - `patch_triangle_attention(model)` 호출 시 `except Exception as e`에서 `e`가 **"duplicate template name"**.
- 예외는 **패치 적용 단계**(모델 로드 직후, GPU 이동 전)에서 발생.

### 1.3 "duplicate template name" 원인 (확정)

- **발생 위치**: `flex_attention_patch.py`를 **import할 때** 실행되는  
  `_compiled_flex_attention = torch.compile(flex_attention)` (구 코드 24행).
- **스택**:
  - `torch.compile(flex_attention)`  
  → `torch._dynamo.optimize`  
  → inductor 로드  
  → `torch/_inductor/kernel/flex_attention.py` 546행: `flex_attention_template = TritonTemplate(...)`  
  → `torch/_inductor/select_algorithm.py` 1695행:  
    `assert name not in self.all_templates, "duplicate template name"`  
  → **AssertionError: duplicate template name**
- **의미**: PyTorch inductor가 FlexAttention용 Triton 템플릿을 등록할 때, 같은 이름이 이미 `all_templates`에 있어서 assertion이 발생함. (같은 프로세스에서 inductor가 한 번 로드되면서 이미 해당 템플릿이 등록된 뒤, 우리가 `torch.compile(flex_attention)`으로 다시 등록을 시도하는 상황으로 추정.)

### 1.4 적용한 수정

- **`flex_attention_patch.py`**: import 시점의 `torch.compile(flex_attention)` 제거.  
  `flex_attention`을 그대로 사용하도록 `_flex_attention_fn = flex_attention` 로 변경.  
  → import 시 더 이상 inductor TritonTemplate 등록이 일어나지 않아 "duplicate template name" 제거.
- **`run_boltz_dap_v2.py`**: FlexAttention 패치 실패 시 `traceback.print_exc()` 로 전체 스택을 로그에 남기도록 예외 로깅 추가 (원인 추적용).

---

## 2. Hexamer 13samp(354019)가 실패한 이유

### 2.1 "저번에는 성공했을 텐데?"에 대한 답

- **13samp는 이전에도 성공한 적이 없음.**
- **353867** (이전 13samp run):  
  동일하게 **"ran out of memory, skipping batch"** → **"OOM during inference!"** → **"No CIF file found"** 로 실패.
- 즉 12samp(354018)만 성공하고, **13samp(353867, 354019)는 두 번 모두 Confidence 단계에서 OOM**으로 실패.

### 2.2 실패 시점 (354019 기준)

- 로그 순서:
  1. 첫 번째 샘플에 대한 Confidence가 **정상 완료**:  
     `chunked PDE r_idx=3/4 done` → `PDE done`  
     `[TIMELINE] 4998.4s | ... after confidence_module`
  2. 직후:  
     `WARNING: ran out of memory, skipping batch`  
     `OOM during inference!`
- 따라서 **첫 번째 샘플의 confidence는 끝났고, 그 다음 단계**(두 번째 샘플 confidence 시작 직전/직후, 또는 그 사이의 merge/정리 단계)에서 OOM 발생.

### 2.3 12samp는 되고 13samp만 실패하는 이유 (가능 원인)

- **multiplicity**만 12 vs 13 차이이고, 나머지 설정·입력은 동일.
- `dap_confidence.py`에서:
  - `merged` 버퍼는 `(multiplicity,) + shape` 로 **CPU**에 할당 → 13일 때 약 8% 더 큰 CPU 메모리만 사용. GPU OOM과는 직접 연관 없음.
  - 다만 **GPU 상**에서 multiplicity에 비례해 커지는 부분이 있음:
    - 예: `token_to_rep_atom.repeat_interleave(multiplicity, 0)`, `s.repeat_interleave(multiplicity, 0)`, `z_chunk.repeat_interleave(multiplicity, 0)` 등은 **이미 1 sample씩 순차 실행**하는 루프 안에서는 `multiplicity=1`로 호출되므로, 루프 내부에서는 12/13에 따른 차이가 없어야 함.
- 따라서 가능한 설명:
  1. **메모리 단편화**
     - 12번째 confidence까지는 피크가 80GB 아래로 유지되다가, 13번째에서 조금만 더 쓰면 OOM이 나는 상황은 아님(실제로는 **첫 샘플 직후**에 OOM).
     - 대신 **첫 샘플 직후**에 “두 번째 샘플 준비”나 “merge 버퍼에 복사” 시 일시적으로 필요한 GPU 메모리가, 13일 때 약간 더 크거나 레이아웃이 달라서 연속 블록이 부족해질 수 있음.
  2. **multiplicity에 묵시적으로 의존하는 GPU 텐서**
     - 코드 상으로는 merged는 CPU이나, **다른 경로**에서 multiplicity가 13일 때만 더 큰 GPU 텐서가 할당되거나, 13일 때 한 번 더 반복되는 루프가 있을 수 있음.  
     → `dap_confidence.py` 전체에서 `multiplicity`가 사용되는 모든 지점을 다시 검토해, **GPU에 (13, ...) 형태로 할당되는 부분이 있는지** 확인하는 것이 좋음.
  3. **비결정적 동작**
     - 같은 13samp라도 노드/메모리 상태에 따라 첫 샘플 직후에만 가끔 OOM이 날 수 있음.  
     → 13samp를 한두 번 더 돌려 보거나, 12samp + 1samp 두 번 돌려서 13개를 모으는 방식으로 우회 가능.

### 2.4 권장 조치

- **당장**: 13개가 필요하면 **12samp + 1samp** (또는 8+5 등)처럼 나눠서 돌리고, 결과만 합쳐서 사용.
- **원인 추적**:  
  - `dap_confidence.py`에서 `multiplicity`가 붙는 모든 텐서가 **정말 CPU 또는 multiplicity=1 구간에서만** 쓰이는지 확인.  
  - 첫 번째 confidence 직후 구간에 **임시 로그**(예: `torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()`)를 넣어, 12samp와 13samp에서 “직후 피크”가 얼마나 차이 나는지 비교.

---

## 3. 요약 표

| 항목 | 내용 |
|------|------|
| FlexAttention 스킵 | 모든 run에서 `patch_triangle_attention()` 예외로 스킵. 메시지: "duplicate template name". |
| 예상 원인 | import 시 `torch.compile(flex_attention)` + Boltz2의 `template_module` compile과의 이름/캐시 충돌 가능성. |
| 13samp 성공 이력 | 없음. 353867, 354019 모두 Confidence 구간 OOM으로 실패. |
| 13samp OOM 시점 | 첫 번째 샘플 confidence 완료 직후(두 번째 샘플 시작 전/직후 또는 merge 단계). |
| 권장 | Flex: import 시 compile 제거 또는 지연, traceback 추가. 13samp: 12+1 등 분할 실행으로 우회, multiplicity 관련 GPU 할당 재검토. |
