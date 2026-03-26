# Stage v7.13 Full Execution Report (KO)

## 1) 실행 개요
- run root: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205`
- 실행 목적: v7.12 피드백 반영 후, boundary(P8)에서
  - Gate 정렬 유지(recall)
  - split/marker 안정성 1.0 복구
  - TIME_NON_MONOTONIC/immutable terms 병목 완화
  - split-only 대비 V3 추가 이득 재검증
- 실행 구간(로그 타임스탬프 기준): `2026-03-03 05:42 KST ~ 11:32 KST` (약 5시간 50분)
- 최종 평가 stage: `stage_feedback_v7_13_boundary_test_v8` (unseen, 60 pairs)

## 2) 반영한 코드/설정 변경
### 2.1 러너 로직
- 파일: `/disk/chojm/exp/runners/common.py`
- 주요 반영:
1. immutable scope term 탐지 헬퍼 추가 (`_immutable_scope_term_locations`).
2. Gate 단계에서 immutable term 감지 시
   - locked prefix 복원,
   - 필요 시 suffix 1회 regen,
   - 재검사 수행.
3. best-of 이후 Gate 재검사(`planning_boundary_gate_recheck_after_bestof`) 추가.
4. finalize 직전 split marker 누락 시 deterministic split 재적용 재시도(`planning_split_marker_retry_on_finalize`) 추가.
5. split/immutable 관련 로깅 확장:
   - `split_retry_triggered`, `split_marker_lost_stage`
   - `immutable_terms_detected`, `immutable_terms_locations`, `immutable_terms_fix_action`

### 2.2 집계/분석
- 파일: `/disk/chojm/exp/analysis/summarize.py`
- 신규 집계 컬럼:
  - `split_retry_triggered_rate`
  - `split_marker_lost_stage_nonempty_rate`
  - `immutable_terms_detected_rate`

### 2.3 실험 설정
- 파일: `/disk/chojm/exp/config/experiment_feedback_v7_13.yaml`
- 핵심 플래그:
  - `planning_boundary_immutable_terms_regen: true`
  - `planning_split_marker_retry_on_finalize: true`
  - `planning_boundary_gate_recheck_after_bestof: true`
  - `planning_boundary_suffix_regen_attempts: 1`
- stage 구성:
  - smoke(5 pairs), debug(30), boundary_dev(144), boundary_test_v8(60)

### 2.4 실행/정리 스크립트
- 파일: `/disk/chojm/exp/runners/run_feedback_v7_13_with_cleanup.sh`
- 역할:
  - stage 순차 실행
  - integrity/summary/dual_eval/paired/gate_alignment/diagnostics 자동 수행
  - 종료 시 vLLM 종료 + GPU 해제 로그 저장

## 3) 공정성/무결성 체크
- evaluator lock SHA 기록: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/runs/runner_logs/evaluator_lock_sha256.txt`
- pair integrity:
  - smoke/debug/dev/test_v8 모두 `passed=true`, `duplicates=0`
  - 증빙: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/runs/runner_logs/integrity_*`
- 동일 예산 유지:
  - `max_iterations=5`, `max_timeout=120`, `saga_max_retries=2`

## 4) 단계별 결과 요약
(모드: runtime / strict, 단위: success_rate)

| stage | pairs | V0 runtime | V0_SPLIT_ONLY runtime | V3 runtime | V0 strict | V0_SPLIT_ONLY strict | V3 strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke | 5 | 0.0000 | 0.2000 | 0.6000 | 0.0000 | 0.2000 | 0.6000 |
| debug | 30 | 0.0333 | 0.2333 | 0.4667 | 0.0000 | 0.1667 | 0.3667 |
| boundary_dev | 144 | 0.0139 | 0.2361 | 0.4236 | 0.0000 | 0.2153 | 0.3403 |
| boundary_test_v8 (final unseen) | 60 | 0.0000 | 0.2667 | 0.4667 | 0.0000 | 0.2000 | 0.4167 |

원본 요약 파일:
- `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/dual_eval/metrics_runtime.csv`
- `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/dual_eval/metrics_strict.csv`

## 5) 최종 unseen(test_v8) 상세 결과
### 5.1 핵심 지표
- V0
  - runtime success `0.0000`
  - strict success `0.0000`
- V0_SPLIT_ONLY
  - runtime success `0.2667`
  - strict success `0.2000`
- V3_PREFIX_SPLIT
  - runtime success `0.4667`
  - strict success `0.4167`
  - avg violation(runtime/strict): `1.6333 / 2.0833`
  - state_consistent(runtime): `0.7833`
  - immutable_after_split(runtime): `0.7667`
  - split_applied / marker_survived: `1.0000 / 1.0000`
  - wall_time p50/p95: `43.45s / 51.70s`
  - llm_call_count_mean: `2.9667`

### 5.2 paired 통계 (split-only vs V3)
- runtime:
  - wins/loss/ties = `20 / 8 / 32`
  - mean diff `+0.2000`
  - CI95 `[0.0333, 0.3667]`
  - p-value `0.0357`
  - 파일: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/paired_runtime_v0_split_only_vs_v3_prefix_split.json`
- strict:
  - wins/loss/ties = `21 / 8 / 31`
  - mean diff `+0.2167`
  - CI95 `[0.0500, 0.3833]`
  - p-value `0.0241`
  - 파일: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/paired_strict_v0_split_only_vs_v3_prefix_split.json`

## 6) Gate/병목 진단
### 6.1 Gate alignment
- strict_pass & gate_fail(FN) = `0`
- gate_recall_on_strict_pass = `1.0000`
- gate_precision_on_gate_pass = `0.5319`
- 해석:
  - 정렬 recall은 유지됐지만 precision이 낮아, gate 통과안 중 strict 실패 비율이 여전히 큼.
- 파일: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/gate_alignment_v7_13_boundary_test_v8.md`

### 6.2 split/debug
- split_applied, marker_survived: V3 `1.0 / 1.0`
- split mode(V3):
  - SYSTEM_CONSTRUCTED `42`
  - REAL_CROSSING_FOUND `14`
  - FAILED_NO_CANDIDATE `4`
- 파일: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/split_debug_v7_13_boundary_test_v8.md`

### 6.3 strict 실패 상위(V3, test_v8)
- `event[7] exceeds photo_time`: 20
- `event[6] exceeds photo_time`: 18
- `Immutable past appears modified by disruption terms`: 14
- `State timeline is inconsistent at disruption boundary`: 13
- `event[5] exceeds photo_time`: 12

(집계 소스: `rescored_strict.jsonl`에서 variant=`V3_PREFIX_SPLIT` violation count)

### 6.4 state timeline 상세
- strict `State timeline ...` 매치 행: `13`
- reason 분포:
  - UNKNOWN `10`
  - IN_TRANSIT_MISMATCH `2`
  - TIME_PARSE_FAIL `1`
- 파일: `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/state_timeline_failure_breakdown_v7_13_boundary_test_v8.md`

## 7) v7.13 DoD 판정 (final=unseen test_v8)
피드백에서 제시된 v7.13 DoD 기준으로 평가:

| 항목 | 목표 | 실측(V3, test_v8) | 판정 |
|---|---:|---:|---|
| runtime success | >= 0.80 | 0.4667 | MISS |
| strict success | >= 0.40 | 0.4167 | PASS |
| state_consistent_given_applicable | >= 0.55 | 0.7833 | PASS |
| immutable_after_split_given_applicable | >= 0.75 | 0.7667 | PASS |
| split_applied_runtime_rate | == 1.0 | 1.0000 | PASS |
| split_marker_survived_rate | == 1.0 | 1.0000 | PASS |
| wall_time_p50 | <= 40s | 43.45s | MISS |
| llm_call_count_mean | <= 3.0 | 2.9667 | PASS |
| gate_recall_on_strict_pass | >= 0.95 | 1.0000 | PASS |
| gate_precision_on_gate_pass | >= 0.80 | 0.5319 | MISS |
| fallback_rate | <= 0.15 | 0.2333 | MISS |

## 8) 결론
1. v7.13은 split-only 대비 V3의 추가 이득을 unseen `test_v8`에서 runtime/strict 모두 유의하게 재확인했다.
2. split/marker 안정성은 final stage에서 `1.0/1.0`으로 복구했다.
3. 다만 DoD 기준으로는 `runtime success`, `wall_time p50`, `gate precision`, `fallback rate`가 미달이다.
4. strict 관점의 잔여 병목은 `photo_time`, `immutable terms`, `state timeline`이며, 특히 gate precision 개선이 다음 라운드의 1순위다.

## 9) 다음 라운드 권고(v7.14)
1. Gate precision 향상: strict_fail & gate_pass(FP) 케이스 유형별로 gate check를 더 좁게 정의.
2. fallback 절감: `FAILED_NO_CANDIDATE` 및 gate fail 상위 reason에 대한 deterministic 보정 강화.
3. photo_time 전용 deterministic pass를 gate 직전 1회 고정해 strict/photo_time tail 축소.
4. latency 개선: `llm_call_mean`은 유지하면서 postproc 분기 단축으로 wall p50을 40s 이하로 절감.

## 10) 주요 산출물 경로
- 최종 보고서(본 파일):
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/stage_v7_13_full_execution_report_ko.md`
- 최종 stage 요약:
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/dual_eval/metrics_runtime.csv`
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/dual_eval/metrics_strict.csv`
- paired stats:
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/paired_runtime_v0_split_only_vs_v3_prefix_split.json`
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/results/summary/v7_13_boundary_test_v8/paired_strict_v0_split_only_vs_v3_prefix_split.json`
- gate/split/state 진단:
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/gate_alignment_v7_13_boundary_test_v8.md`
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/split_debug_v7_13_boundary_test_v8.md`
  - `/disk/chojm/experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205/reports/state_timeline_failure_breakdown_v7_13_boundary_test_v8.md`
