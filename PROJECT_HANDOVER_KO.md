# RLM+Saga 실험 인수인계 문서 (상세)

## 1. 프로젝트 목적
- 목적: `RLM`에 Saga 계층(트랜잭션/검증/복구/불변성 관리)을 붙였을 때, 교란(disruption) 상황에서의 안정성과 성능이 개선되는지 검증.
- 핵심 가설:
  1. split/compensation만으로도 boundary 성능이 오른다.
  2. prefix-lock + saga repair가 strict 기준에서 추가 이득을 만든다.
  3. 다만 state/immutable 정합성과 latency를 함께 관리해야 실사용 수준으로 수렴한다.

## 2. 지금까지 수행 절차(순서)
- v7.1~v7.3: boundary split 적용 자체 디버그, split applied=0 문제 해소.
- v7.4~v7.6: holdout/evaluator lock/paired 통계 체계 고정, split-only vs V3 분해.
- v7.7~v7.9: state/immutable 병목 식별, strict 상위 위반 정리, latency 원인 분해.
- v7.10~v7.12: gate alignment 개선, strict-pass & gate-fail 구조 결함 해소.
- v7.13: unseen test_v8에서 split-only 대비 V3 strict 추가 이득 유의 재확인.
- v7.14(현재): photo_time deterministic repair + gate FP 분해를 반영하는 코드 준비 단계. 실행은 중단되어 완주 안 됨.

## 3. 현재 코드/설정 상태
- 코드 루트: `code/exp/`
- 최신 준비된 설정:
  - `code/exp/config/experiment_feedback_v7_14.yaml`
  - `code/exp/config/specific_ids_v7_14_test_v9.json`
- 최신 러너:
  - `code/exp/runners/run_feedback_v7_14_with_cleanup.sh`
- 주요 신규/변경 포인트:
  - `code/exp/runners/common.py`
    - `PHOTO_TIME_EXCEEDED`를 gate reason에 포함
    - deterministic photo_time repair 함수가 메트릭 반환하도록 확장
    - gate loop/targeted retry/hard gate에서 photo_time 메트릭 수집
    - 결과 row에 photo_time 관련 컬럼 추가
  - `code/exp/analysis/gate_fp_breakdown.py` 신규
    - gate-pass & strict-fail(FP) 원인 분해 리포트 생성

## 4. 마지막으로 “완주된” 실험 기준
- 안정적으로 해석 가능한 최신 완주 라운드: v7.13
- 핵심 참고 run (원본 실행 환경 기준 경로):
  - `experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205`
- 본 보관본에 포함된 핵심 리포트 (`evidence/v7_13_boundary_test_v8/reports/`):
  - `evidence/v7_13_boundary_test_v8/reports/stage_v7_13_full_execution_report_ko.md`
  - `evidence/v7_13_boundary_test_v8/reports/gate_alignment_v7_13_boundary_test_v8.md`
  - `evidence/v7_13_boundary_test_v8/reports/state_timeline_failure_breakdown_v7_13_boundary_test_v8.md`

## 5. 실험 데이터/산출물 위치 규칙
- 각 run 폴더 공통 구조:
  - `reports/`
  - `results/raw/`
  - `results/summary/`
  - `runs/runner_logs/`
  - `runs/vllm_server_logs/` (nvidia-smi 증빙 텍스트)

## 6. 새 채팅/새 서버에서 바로 재개하는 절차
1. 이 폴더를 새 서버로 복사.
2. Python venv 생성 후 필요한 패키지 설치.
3. vLLM 서버 기동(8GPU/TP=8).
4. `code/exp/runners/run_feedback_v7_14_with_cleanup.sh` 실행.
5. 완료 후 `reports/`와 `results/summary/`에서 DoD PASS/MISS 확인.

## 7. v7.14 진행 목표(요약)
- photo_time 위반을 deterministic pass로 감소.
- gate precision 개선(Recall 1.0 유지).
- fallback/wall-time 감소.
- unseen test_v9 기준으로 runtime/strict/운영지표 동시 점검.

## 8. 리스크 및 주의
- Gate와 strict evaluator의 정의 불일치가 재발하면 해석 불가.
- test set을 이미 본 뒤 튜닝하면 일반화 주장 약화됨(새 unseen holdout 고정 필요).
- latency는 토큰보다 LLM 호출/재생성 루프에 크게 좌우됨.

## 9. 즉시 점검 체크리스트
- [ ] split_applied_runtime_rate == 1.0
- [ ] split_marker_survived_rate == 1.0
- [ ] gate_recall_on_strict_pass >= 0.95
- [ ] gate_precision_on_gate_pass 개선 추세
- [ ] photo_time strict 위반 top 비중 하락
- [ ] fallback_rate 하락
- [ ] wall_time p50 <= 목표

## 10. 포함된 파일 인덱스
- run 목록/원본/용량: `manifests/run_index.csv`
- 전체 체크섬: `manifests/SHA256SUMS.txt`
- 본 문서: `PROJECT_HANDOVER_KO.md`
