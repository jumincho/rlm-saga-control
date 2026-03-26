# rlm-saga-control

> 🧊 **휴면(dormant) 중인 연구 파일럿입니다.**

## 무엇을 보려던 연구였나

언어모델에게 어떤 일정/계획 같은 시나리오를 풀게 하는데, 도중에 예상치 못한 사건(교란)이 끼어들면 계획이 흐트러집니다. 이 프로젝트는 그 위에 **트랜잭션 시스템에서 빌려 온 안전망(Saga 스타일 제어)** 을 얹으면, 흐트러진 부분만 깔끔하게 고치면서 나머지 결정은 유지할 수 있을지를 본 연구입니다.

비교한 것은 크게 셋이었습니다.

- 그대로 두는 쪽 — 모델이 알아서 한다
- 가볍게만 분기 처리하는 쪽 — 교란된 부분 앞뒤로 잘라 따로 처리한다
- 더 본격적으로 제어하는 쪽 — 앞부분 잠금(prefix lock), 검증, 복구, 불변성 관리까지 함께 한다

평가는 단순한 성공률만이 아니라, "교란 전 결정이 망가지지 않았나", "수정이 일관되게 일어났나", "지연이 얼마나 늘었나" 같은 운영 지표까지 같이 봤습니다.

## 무엇을 알아냈나

- **더 본격적인 제어 쪽이 분명히 더 좋았습니다.** 마지막으로 완주한 비교(처음 보는 시나리오 위에서의 짝지은 비교)에서, 그냥 분기만 하는 쪽보다 정확도가 의미 있게 더 높았습니다.
- **하지만 운영 지표 기준을 모두 만족하진 못했습니다.** 정확도는 올라갔지만 지연 시간, 잘못된 통과(false positive) 비율, fallback 호출 빈도 같은 항목들에서 자체 설정한 목표선을 못 넘긴 것이 여러 개 남았습니다.
- **즉 "되긴 되는데 아직 다듬어지지 않은" 상태**입니다. 다음 라운드에서 다듬으려던 코드와 설정까지는 준비됐지만, 거기까지 실제로 돌려 보진 못했습니다.

자세한 숫자가 궁금하시면:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

## 왜 잠시 멈춰 두는가

긍정 신호 자체는 분명히 있고, 다음 단계로 갈 코드와 가설도 정리돼 있습니다. 다만 "이만하면 운영 가능한 수준" 이라고 말하려면 한 라운드 더 다듬어야 하고, 그건 지금보다 더 큰 환경(여러 GPU, 별도 서빙 인프라)이 필요합니다. 그래서 지금은 잠시 묶어 두고, 자극이 생기면 다음 라운드부터 다시 깨우는 편이 자연스럽다고 판단했습니다.

## 다시 들여다볼 때는 어디부터

- [`PROJECT_HANDOVER_KO.md`](PROJECT_HANDOVER_KO.md) — 인수인계 문서 (가장 먼저 읽으면 좋은 글)
- [`code/exp/README.md`](code/exp/README.md) — 코드 트리 진입점
- [`evidence/`](evidence/) — 마지막 완주 비교의 핵심 결과물들 (paired 비교, 게이트 정렬 리포트, 실패 분해)

다시 살릴 때 권장하는 우선순위: 게이트 정확도 개선 → 결정적 복구 로직 보강 → fallback 빈도 감소 → 지연 절감.

## 코드 어디에 뭐가 있나

| 파일 | 하는 일 |
|---|---|
| [`code/exp/runners/common.py`](code/exp/runners/common.py) | 실험 루프의 공통 부품: 게이트, 재시도, 복구 함수 |
| [`code/exp/runners/run_paired.py`](code/exp/runners/run_paired.py) | 짝지은 비교 실험 러너 |
| [`code/exp/analysis/gate_fp_breakdown.py`](code/exp/analysis/gate_fp_breakdown.py) | "게이트는 통과했지만 엄격 평가에선 실패한 경우" 의 원인 분해 |
| [`code/exp/analysis/state_timeline_failure_breakdown.py`](code/exp/analysis/state_timeline_failure_breakdown.py) | 시점별 상태 일관성 위반 분해 |
| [`code/exp/analysis/immutability_alignment_report.py`](code/exp/analysis/immutability_alignment_report.py) | 바뀌면 안 되는 부분이 잘 보존됐는지 점검 |
| [`code/exp/bench/scorer.py`](code/exp/bench/scorer.py) | 결과 채점 (느슨한 기준 / 엄격한 기준) |

## 폴더 지도

```
.
├── code/exp/              실험 코드 (러너, 분석, 벤치, 설정)
├── evidence/              마지막 완주 비교의 대표 산출물
├── manifests/             포함된 run 의 목록과 체크섬
├── closure_reports/       종료 보고서 (한국어 / 영문)
└── PROJECT_HANDOVER_KO.md 인수인계 문서
```

코드 안에 절대 경로(`/disk/...`) 가정이 박혀 있고 멀티 GPU + 별도 서빙 환경을 전제로 합니다. 새 환경에서 그대로 안 돌 가능성이 높으니 "다시 이해해서 깨우는 출발점" 으로 봐 주세요.

## 환경

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# 별도 서빙 환경(예: vLLM, 멀티 GPU) 필요
```

## 상태

🧊 **휴면 중** — 분명한 긍정 신호와 다음 단계 계획이 모두 정리된 상태에서 멈춰 있습니다.
