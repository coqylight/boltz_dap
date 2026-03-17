# 대형 protein complex 여러 GPU로 나눠서 inference — 오픈 소스 현황

## 질문

Boltz2처럼 **엄청 큰 protein complex 하나**를 **여러 GPU에 나눠서**(pair dimension 등 shard) inference하는 **오픈 GitHub / Hugging Face 사례**가 있는지.

---

## 1. Boltz2 공식 (jwohlwend/boltz)

- **Multi-GPU**: `--devices N` + `CUDA_VISIBLE_DEVICES` 지원.
- **의미**: **입력이 여러 complex일 때** 각 complex를 서로 다른 GPU에 배치하는 **data parallel**.
- **단일 complex를 여러 GPU로 쪼개기**: **미지원**.
  - Issue #460: "GPU parallelization only worked when there were **multiple complexes** in the input, with each complex assigned to a different GPU. **I do not think a single complex can be split across multiple GPUs.**"
  - Issue #51: "we do not support **model parallelism**, devices should be used to run many predictions in parallel."
- 대형 단일 complex는 **chunking**(PR #43, crop>512 토큰 시 자동) + 큰 GPU(48–80GB) 또는 NIM/TensorRT로만 대응.

**정리**: 공식 Boltz2에는 **단일 대형 complex를 여러 GPU에 나눠서 inference하는 기능 없음.**

---

## 2. FastFold (hpcaitech/FastFold)

- **대상**: **AlphaFold2** (Boltz2 아님).
- **DAP (Dynamic Axial Parallelism)**: pair/sequence dimension을 나눠서 **한 개 시퀀스(단일 complex)를 여러 GPU에 분산**.
- 학습·추론 모두 multi-GPU, long sequence 7.5–9.5× 속도 향상, 512 GPU까지 스케일 등.
- 오픈 소스: GitHub (Apache-2.0).

**정리**: **AlphaFold2**에 대해서는 “단일 대형 complex를 여러 GPU로 나눠서 inference”하는 **오픈 사례가 있음 (FastFold).**  
**Boltz2**에 대해서는 해당 없음.

---

## 3. AlphaFold3 (google-deepmind/alphafold3)

- Issue #64: "Can AlphaFold3 predict extremely large complexes using **multiple GPUs**?"
- **단일 대형 complex를 여러 GPU로 나누는 공식 지원은 없음.**  
  대형 complex는 unified memory 등 단일 GPU 메모리 최적화 위주.

---

## 4. 기타 (OpenFold, MassiveFold, ParallelFold 등)

- **OpenFold**: PyTorch 재구현, 메모리 효율 강조. **단일 complex를 multi-GPU로 쪼개는 DAP/쉬딩** 설명은 검색에서 안 나옴.
- **MassiveFold, ParallelFold**: 여러 구조를 여러 GPU에 나누거나, CPU/GPU 파이프라인 분리 등. **단일 complex pair dimension shard** 오픈 구현은 검색되지 않음.

---

## 5. Boltz2 + multi-GPU (단일 complex 분산) 오픈 사례

- 검색 쿼리: `Boltz2 multi-GPU distributed inference`, `Boltz2 DAP shard`, `Boltz2 pair dimension parallel` 등.
- **결과**:  
  - 공식 Boltz는 “여러 complex → 여러 GPU”만 지원.  
  - **Boltz2를 단일 대형 complex 기준으로 여러 GPU에 나눠서 inference하는 공개 GitHub/Hugging Face 프로젝트는 (우리 boltz_dap 제외) 찾지 못함.**

---

## 6. 종합

| 모델        | 단일 대형 complex를 여러 GPU로 나눠서 inference하는 오픈 구현 |
|-------------|--------------------------------------------------------------|
| **AlphaFold2** | ✅ **FastFold** (DAP, hpcaitech/FastFold)                     |
| **AlphaFold3** | ❌ 공식/오픈 없음                                            |
| **Boltz2**     | ❌ 공식 미지원. **오픈으로는 (검색 기준) coqylight/boltz_dap만 해당.** |

즉, **Boltz2로 “엄청 큰 protein complex 하나를 여러 GPU에 나눠서 inference”하는 오픈 GitHub/Hugging Face 사례는, 우리 레포(coqylight/boltz_dap)를 제외하면 아직 없어 보인다.**  
AlphaFold2만 FastFold에서 비슷한 역할(DAP 기반 단일 시퀀스 multi-GPU)을 하고 있다.
