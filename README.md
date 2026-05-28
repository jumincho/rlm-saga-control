<div align="center">

# rlm-saga-control

**LLM 시나리오에 트랜잭션식 Saga 제어를 얹으면 교란이 깨끗하게 복구되는가**
**Does a Saga-style control layer over an LLM cleanly recover plans from mid-run disturbance?**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

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

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — 본문과 종료 보고서, 코드 row 에 등장하는 내부 용어(세 가지 regime, gate / gate FP, 불변성, prefix lock, fallback, strict vs loose, `SAGA_*` 환경변수)를 일반어로 정리한 사전
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

코드 안에 절대 경로(`/disk/...`) 가정과 멀티 GPU + 별도 서빙 환경 전제가 박혀 있어, 새 환경에서 그대로는 돌지 않습니다. 재실행보다는 "다시 이해해서 깨우는 출발점" 으로 적합합니다.

## 환경

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# 별도 서빙 환경(예: vLLM, 멀티 GPU) 필요
```

부트스트랩과 러너의 절대 경로 가정은 다음 환경변수로 덮어쓸 수 있습니다.
다만 wrapper `run_*_with_cleanup.sh` 들은 과거 실행 기록으로 그대로 두었고,
실제 재실행을 시도한다면 `bootstrap_env.sh` 가 시작점입니다.

| 환경변수 | 의미 | 기본값 |
| --- | --- | --- |
| `SAGA_EXPERIMENTS_ROOT` | run 디렉토리가 생성되는 위치 | `/disk/chojm/experiments` |
| `SAGA_LLM_REQUIREMENTS` | bootstrap 시 설치할 requirements.txt | `/disk/chojm/SagaLLM/requirements.txt` |
| `SAGA_RLM_PACKAGE` | editable 설치할 RLM 패키지 위치 | `/disk/chojm/rlm` |
| `RLM_SAGA_RUN_ROOT_BASE` | runner Python의 `run_root` 기본값 (YAML보다 우선) | YAML의 `paths.run_root_base` |

## 상태

🧊 **휴면 중** — 분명한 긍정 신호와 다음 단계 계획이 모두 정리된 상태에서 멈춰 있습니다.

---

<a name="english"></a>

## English

> 🧊 **Dormant research pilot.**

### What this set out to test

When you ask a language model to solve a scenario (e.g., a schedule or plan) and an unexpected event disturbs the run partway, the plan unravels. This project layered **Saga-style control borrowed from transactional systems** on top of the LLM to see whether the perturbed part could be cleanly repaired while keeping the rest of the decisions intact.

Three compared regimes:

- Hands-off — let the model handle it.
- Light branching — split the disturbance out and handle it separately.
- Full control — prefix lock, verification, recovery, and invariant management together.

Evaluation went beyond plain success rates: "are pre-disturbance decisions preserved?", "is the repair consistent?", "how much extra latency?" and similar operational metrics.

### What it found

- **The full-control regime was clearly better.** In the last completed paired comparison on unseen scenarios, accuracy was meaningfully higher than the light-branching regime.
- **But not all operational targets were met.** Accuracy improved, but several self-imposed targets — latency, false-positive rate, fallback-call frequency — were not cleared.
- So: **"works but not yet polished."** The code and configs for the next refinement round were prepared, but never actually run.

Full numbers:

- 🇰🇷 [`closure_reports/project_closure_report_ko_20260327.md`](closure_reports/project_closure_report_ko_20260327.md)
- 🇬🇧 [`closure_reports/project_closure_report_20260327.md`](closure_reports/project_closure_report_20260327.md)

### Why it's on hold

There is a clear positive signal and a planned next step, but calling the system "ops-grade" needs one more refinement round — and that needs a bigger environment (multi-GPU, separate serving infrastructure). Parking it now and reopening it when the next round is feasible is the natural call.

### Where to look first when revisiting

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — Decoder ring for the internal vocabulary that survived into the source tree and the closure reports (the three regimes, gate / gate FP, immutability, prefix lock, fallback, strict vs loose, the `SAGA_*` env vars).
- [`PROJECT_HANDOVER_KO.md`](PROJECT_HANDOVER_KO.md) — handover document. Read this first.
- [`code/exp/README.md`](code/exp/README.md) — entry into the code tree.
- [`evidence/`](evidence/) — key outputs of the last completed paired comparison (paired comparisons, gate alignment reports, failure decompositions).

Recommended priority when reopening: improve gate accuracy → strengthen deterministic recovery logic → reduce fallback frequency → trim latency.

### Code map

| File | What it does |
|---|---|
| [`code/exp/runners/common.py`](code/exp/runners/common.py) | Shared experiment-loop primitives: gates, retries, recovery |
| [`code/exp/runners/run_paired.py`](code/exp/runners/run_paired.py) | Paired-comparison experiment runner |
| [`code/exp/analysis/gate_fp_breakdown.py`](code/exp/analysis/gate_fp_breakdown.py) | Decomposition of "gate-pass but strict-fail" cases |
| [`code/exp/analysis/state_timeline_failure_breakdown.py`](code/exp/analysis/state_timeline_failure_breakdown.py) | Per-step state-consistency violation breakdown |
| [`code/exp/analysis/immutability_alignment_report.py`](code/exp/analysis/immutability_alignment_report.py) | Checks that what should not change has not changed |
| [`code/exp/bench/scorer.py`](code/exp/bench/scorer.py) | Result scoring (loose / strict criteria) |

### Folder map

```
.
├── code/exp/              experiment code (runners, analysis, bench, config)
├── evidence/              representative outputs of the last completed comparison
├── manifests/             included-run lists and checksums
├── closure_reports/       closure reports (KO / EN)
└── PROJECT_HANDOVER_KO.md handover doc
```

The code embeds absolute paths (`/disk/...`) and assumes a multi-GPU + separate serving environment, so it will not run as-is in a new environment. The archive is better used as a starting point for understanding and reawakening than for direct re-runs.

### Environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
# requires a separate serving environment (e.g., vLLM, multi-GPU)
```

The bootstrap script and Python runners honor a few environment-variable
overrides for the original author's absolute-path defaults. The wrapper
`run_*_with_cleanup.sh` shell scripts are preserved as historical run
records and have not been parameterized; the practical entry point for a
re-execution attempt is `bootstrap_env.sh`.

| Variable | Meaning | Default |
| --- | --- | --- |
| `SAGA_EXPERIMENTS_ROOT` | Where run directories are created | `/disk/chojm/experiments` |
| `SAGA_LLM_REQUIREMENTS` | requirements.txt installed at bootstrap | `/disk/chojm/SagaLLM/requirements.txt` |
| `SAGA_RLM_PACKAGE` | Path of the RLM package installed in editable mode | `/disk/chojm/rlm` |
| `RLM_SAGA_RUN_ROOT_BASE` | Python runner `run_root` default (takes precedence over YAML) | YAML `paths.run_root_base` |

### Status

🧊 **Dormant** — a clear positive signal and a prepared next step, paused before that round was run.

### License

Released under [CC BY-NC 4.0](./LICENSE).
