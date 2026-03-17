# boltz_dap 리포지터리 업데이트 현황

GitHub [coqylight/boltz_dap](https://github.com/coqylight/boltz_dap) 기준으로, 로컬에서 추가·수정했지만 아직 반영되지 않은 항목 정리.

---

## 1. 이미 리포에 있는 기능 (README에만 안 적혀 있던 것)

| 기능 | 코드 | README 반영 |
|------|------|-------------|
| **Confidence** (pLDDT, pTM, PAE 등) | `dap_confidence.py` + `dap_trunk.py` (이미 커밋됨) | ✅ 이번에 명시 |
| **Potentials** (FK steering, physical guidance) | `run_boltz_dap_v2.py` `--use_potentials` (이미 커밋됨) | ✅ 이번에 Options 표에 추가 |
| **MSA 서버** | `--use_msa_server` (이미 커밋됨) | ✅ 이번에 Options 표에 추가 |
| **AF3-style 기본값** | `recycling_steps=3`, `sampling_steps=200` 등 (기본값으로 이미 사용 중) | ✅ 표에 "(AF3-style default)" 설명 추가 |

→ **Confidence / Potentials / MSA 서버**는 코드에는 이미 있고, README Options 테이블에만 빠져 있었음. 위처럼 README를 수정해 두었음.

---

## 2. 로컬에만 있고 아직 커밋/푸시 안 된 것

### 2.1 수정된 파일 (git status M)

| 파일 | 내용 |
|------|------|
| `boltz_dap_v2/dap_trunk.py` | Confidence DAP, OOM 대응 등 최신 트렁크 로직 |
| `boltz_dap_v2/flex_attention_patch.py` | FlexAttention 패치 (import 시 compile 제거 등) |
| `boltz_dap_v2/run_boltz_dap_v2.py` | `--use_flex_attention`, `--use_flex_attention_chunked` 플래그 및 패치 로딩 블록 |

### 2.2 새 파일 (git status ??)

| 경로 | 설명 |
|------|------|
| `boltz_dap_v2/flex_attention_patch_chunked.py` | 청크 단위 FlexAttention (full [N,N] bias, FLEX_DAP_CHUNK=128), OOM 회피·수치 동일 |
| `boltz_dap_v2/backup/` | 백업용 (README.md, flex_attention_patch_baseline.py) |
| `boltz_dap_v2/test_flex_patch_traceback.py` | FlexAttention 패치 디버깅용 |
| `docs/` 전체 | 아래 8개 문서 |

### 2.3 docs/ 디렉터리 (전부 미추적)

| 문서 | 내용 |
|------|------|
| `flex_attention_bias_shape_explained.md` | FlexAttention bias shape, DAP와 수치 일치 설명 |
| `boltz_cp_vs_boltz_dap_comparison.md` | NVIDIA boltz-cp vs 우리 DAP 비교 |
| `DAP_equivariance_flex_attention_improvement.md` | DAP + equivariance + FlexAttention 개선 방향 |
| `Triangle_attention_implementation_explained.md` | Triangle attention 구현 설명 |
| `cuequivariance_chunk_support_findings.md` | cuequivariance 청크 지원 조사 |
| `flex_attention_skip_and_13samp_oom_analysis.md` | FlexAttention skip 및 13샘플 OOM 분석 |
| `hexamer_13samp_oom_detailed_analysis.md` | Hexamer 13샘플 OOM 상세 분석 |
| `multi_gpu_large_complex_landscape.md` | 대형 복합체 멀티 GPU 현황 |

---

## 3. 리포 밖에만 있는 참고 자료 (boltz_dap 미포함)

- `/project/engvimmune/gleeai/What_we_learned_from_each_job.txt` — job별 학습 정리 (워크스페이스 루트)
- `/project/engvimmune/gleeai/boltz_output/hexamer_25samp_354380_vs_354432_comparison.md` — 25샘플 실행 비교
- `/project/engvimmune/gleeai/boltz_output/hexamer_*_failure_analysis.md` — 실패 job 분석

원하면 이들 중 일부를 `boltz_dap/docs/` 로 복사해 넣고 커밋할 수 있음.

---

## 4. GitHub에 반영하려면 할 일 (체크리스트)

1. **README.md**  
   - Options 표 보강분이 로컬에 있음 → 커밋 후 푸시하면 됨.

2. **FlexAttention 관련**  
   - `run_boltz_dap_v2.py`, `flex_attention_patch.py`, `dap_trunk.py` 변경분  
   - `flex_attention_patch_chunked.py` 새 파일  
   → 원하면 한 번에 커밋 (예: "Add FlexAttention and chunked FlexAttention options").

3. **문서**  
   - `docs/` 전체를 `git add docs/` 후 커밋 (예: "Add docs: FlexAttention, boltz-cp comparison, OOM analyses").

4. **백업/테스트**  
   - `backup/`, `test_flex_patch_traceback.py` 는 필요 시에만 추가 (선택).

요약: **Confidence / Potentials / AF 기본값**은 코드에 이미 있고, README만 보완했음. **아직 리포에 안 올라간 것**은 FlexAttention 플래그·패치·청크 구현, 그리고 `docs/` 전부와 README 옵션 표 수정분임.
