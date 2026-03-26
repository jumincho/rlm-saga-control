# 프로젝트 종료 보고서

## 이 프로젝트는 무엇이었나

이 저장소는 `RLM`에 Saga 스타일의 제어 계층을 붙였을 때, 교란이 있는 계획 문제에서 안정성과 성능이 개선되는지를 검증한 실험 계열의 handover 번들이다.

핵심은 단순한 프롬프트 실험이나 일반적인 에이전트 튜닝이 아니었다. 이 프로젝트는 트랜잭션적 제어, 검증, 복구, split 처리, prefix lock, retry policy 같은 Saga 성격의 장치를 붙이면 disruption boundary를 넘는 상황에서도 계획의 불변성과 품질을 더 잘 유지할 수 있는지를 보려는 연구였다.

프로젝트 내부 표현을 쓰면, plain `RLM`과 split-only baseline보다 더 강한 Saga-style variant가 hard boundary task에서 실제로 더 나은지를 검증하는 프로젝트였다.

## 핵심 가설

종료 시점까지 발전한 가설은 대체로 다음과 같았다.

- 단순 split/compensation만으로도 boundary 성능은 plain `RLM`보다 개선될 것이다.
- prefix lock, validation, repair까지 포함한 더 강한 Saga 계층은 split-only보다도 추가 이득을 만들 것이다.
- 하지만 그 이득이 의미 있으려면 state consistency, immutability, gate alignment, latency까지 함께 관리되어야 한다.

즉, 이 프로젝트는 단순 성공률 게임이 아니라 “운영 가능한 품질을 유지하면서 더 잘 푸는가”를 묻는 프로젝트였다.

## 무엇을 비교했나

코드 안에는 여러 variant가 있지만, 최종 실험선에서 중요한 것은 주로 아래 셋이었다.

- `V0`: plain `RLM`
- `V0_SPLIT_ONLY`: split 처리만 있는 baseline
- `V3_PREFIX_SPLIT`: prefix lock, split handling, validation, repair logic이 포함된 더 강한 Saga 스타일 variant

실험은 주로 `realm` benchmark의 `P8` boundary-crossing 시나리오에서 수행되었고, runtime 평가와 strict offline 평가를 함께 사용했다.

## 무엇을 측정했나

이 프로젝트는 최종 성공률만 보지 않았다.

중요하게 추적한 지표는 다음과 같다.

- runtime success
- strict success
- gate recall / gate precision
- split applied rate
- split marker survived rate
- state consistency
- immutability after split
- fallback rate
- wall-clock latency
- LLM call count

이 점이 중요하다. 이 프로젝트의 주장은 “조금 더 맞춘다”가 아니라 “불변성과 운영 지표를 함께 지키면서 더 나아진다”였기 때문이다.

## 프로젝트가 어떻게 진행되었나

보존된 handover 문서를 기준으로 보면 대략 다음 순서로 진행되었다.

1. `v7.1` ~ `v7.3`
   boundary split가 실제로 적용되는지부터 디버그했고, 초기의 `split applied = 0` 계열 문제를 해결했다.

2. `v7.4` ~ `v7.6`
   holdout, evaluator, paired-statistics 체계를 고정하고 split-only 이득과 더 강한 Saga 이득을 분리해서 보기 시작했다.

3. `v7.7` ~ `v7.9`
   state/immutability 병목, strict evaluator 실패 패턴, latency 원인을 집중적으로 분석했다.

4. `v7.10` ~ `v7.12`
   gate alignment를 개선했고, “strict pass인데 gate fail” 같은 구조적 불일치를 줄이려 했다.

5. `v7.13`
   마지막으로 완주된 핵심 실험이다. unseen `test_v8`에서 `V3_PREFIX_SPLIT`이 `V0_SPLIT_ONLY`보다 여전히 우세하다는 점을 paired stats까지 포함해 보여 주었다.

6. `v7.14`
   deterministic `photo_time` repair와 gate false-positive 분해를 더 밀어붙이기 위한 코드와 설정은 준비되었지만, 실제 완주 실험은 끝나지 않았다.

## 가장 중요한 완주 결과

이 보관본에서 가장 중요한 완주 run은 다음이다.

- `experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205`

이 run이 핵심인 이유는 다음과 같다.

- end-to-end로 완주되었고
- 진단 리포트와 paired statistics가 함께 있으며
- unseen `boundary_test_v8`를 포함하고
- 실험 설계가 후반부의 더 성숙한 형태를 반영하고 있기 때문이다

## v7.13이 보여 준 것

최종 unseen `boundary_test_v8`에서 보고된 핵심 결과는 다음과 같았다.

- `V0`
  - runtime success: `0.0000`
  - strict success: `0.0000`

- `V0_SPLIT_ONLY`
  - runtime success: `0.2667`
  - strict success: `0.2000`

- `V3_PREFIX_SPLIT`
  - runtime success: `0.4667`
  - strict success: `0.4167`

그리고 `V0_SPLIT_ONLY` 대비 `V3_PREFIX_SPLIT`의 unseen `test_v8` paired 비교는 유의한 양의 결과를 보였다.

- runtime mean diff: `+0.2000`, 95% CI `[0.0333, 0.3667]`, `p = 0.0357`
- strict mean diff: `+0.2167`, 95% CI `[0.0500, 0.3833]`, `p = 0.0241`

즉, 마지막 완주 unseen 실험에서는 더 강한 Saga-style variant가 split-only보다 실제로 더 좋다는 긍정 신호가 있었다.

## 그런데 왜 여기서 멈췄나

방향성은 좋았지만, 프로젝트는 자기 자신의 운영 기준을 완전히 통과하지는 못했다.

보존된 `v7.13` 최종 보고서는 몇 가지 DoD 항목을 명시적으로 MISS로 표시한다.

- runtime success 목표 미달
- wall-time 목표 미달
- gate precision 목표 미달
- fallback-rate 목표 미달

반면에 다음은 좋았다.

- gate recall on strict pass: `1.0000`
- split applied rate: `1.0000`
- split marker survived rate: `1.0000`
- strict success는 목표치를 넘겼다

따라서 이 프로젝트는 “아무것도 안 됐다”가 아니라 “의미 있는 개선은 있었지만, 운영 품질 기준까지는 아직 못 갔다”라고 정리하는 것이 맞다.

## 종료 시점의 핵심 병목

최종 완주 증거와 handover 문서를 종합하면, 반복적으로 남아 있던 병목은 대체로 이렇다.

- `photo_time` 위반
- immutable past 훼손 또는 disruption-term bleed
- disruption boundary 주변의 state timeline inconsistency
- gate precision 저하, 즉 gate는 통과했지만 strict에서는 실패하는 경우가 많음
- fallback과 latency 부담

즉, 이 프로젝트는 “split 자체가 도움이 되는가”라는 초반 질문은 어느 정도 지나왔고, “더 강한 시스템이 정말 더 낫고 운영적으로도 깔끔한가”라는 더 어려운 질문에서 멈춘 상태였다.

## v7.14의 위치

`v7.14`는 다음 단계가 무엇이었는지를 보여 준다는 점에서 중요하지만, 완주된 증거로 취급하면 안 된다.

보존된 문서 기준으로 `v7.14`는:

- 코드 준비 완료
- deterministic `photo_time` repair 반영
- gate false-positive breakdown 반영
- 하지만 실제 full run 완주는 미완

따라서 정직한 해석은 다음과 같다.

- `v7.13`이 마지막 완주 증거다
- `v7.14`는 마지막 준비 상태이지만 미완이다

## 최종 해석

종료 시점에서 가장 방어 가능한 결론은 다음이다.

Saga 스타일의 더 강한 제어를 `RLM`에 붙이면 plain `RLM`이나 split-only baseline보다 실제 개선이 있었고, 마지막 완주 unseen 실험에서도 그 추가 이득이 관측되었다. 하지만 gate precision, fallback behavior, `photo_time`, state consistency edge case, latency 문제가 충분히 해결되지 않아, 시스템은 유망하지만 아직 마무리되지 않은 상태에서 멈췄다.

즉, “실패”보다는 강한 긍정 신호가 있는 미완 프로젝트에 가깝다.

## 이 보관본이 남기는 것

이번 종료 보관본은 원래의 handover 저장소보다 훨씬 작게 만든다.

남기는 것은 다음이다.

- `code/exp` 아래의 실험 프레임워크 전체
- 원래 top-level README와 handover 문서
- run manifest
- 마지막 완주 run의 대표 evidence 파일 몇 개
- 이 종료 보고서의 영문/한글판

반대로 전체 `experiments/` 트리는 보관하지 않는다.

이것은 의도적인 선택이다. 목적은 전체 역사를 보관하는 것이 아니라, 이 프로젝트가 무엇이었고 어디까지 갔는지를 이해할 수 있는 최소 runnable/inspectable bundle을 남기는 것이다.

## 중요한 이식성 주의사항

보존된 코드는 그대로는 완전한 portable package가 아니다.

많은 스크립트와 설정이 다음과 같은 서버 로컬 절대 경로를 가정한다.

- `/disk/chojm/exp/...`
- `/disk/chojm/experiments/...`
- `/disk/chojm/.cache/...`

또한 메인 러너 스크립트는 다음을 전제로 한다.

- vLLM 서빙 환경
- 환경변수 기반 credential 주입
- 8 GPU 실행 경로

따라서 이 코드는 “바로 실행되는 제품”이 아니라 “다시 이해하고 재가동할 수 있는 실험 프레임워크”로 이해하는 편이 맞다.

## 미래에 다시 연다면

미래에 재개한다면 다음 순서가 자연스럽다.

1. `v7.13`을 마지막 완주 기준선으로 삼고
2. `v7.14`를 준비된 다음 단계로 이어받아
3. 우선순위를
   gate precision,
   deterministic `photo_time` repair,
   fallback 감소,
   latency 절감
   에 두고
4. unseen holdout discipline을 그대로 유지해야 한다

반대로, headline success rate만 보고 운영 지표를 무시하는 재개 방식은 바람직하지 않다.

## 먼저 읽어야 할 대표 evidence

이 slim archive 안에서 가장 먼저 읽어볼 가치가 큰 파일은 다음이다.

- 원래 handover 문서
- `v7.13` full execution report
- `boundary_test_v8`의 `gate_alignment` 보고서
- `boundary_test_v8`의 `state_timeline_failure_breakdown` 보고서
- `v7_13_boundary_test_v8`의 `metrics_runtime.csv`, `metrics_strict.csv`
- `V0_SPLIT_ONLY` vs `V3_PREFIX_SPLIT` paired JSON 파일

이 정도면 처음 보는 사람도 “어디까지는 됐고 어디가 막혔는지”를 충분히 이해할 수 있다.

## 종료 권고

현재 상태의 이 프로젝트는 보관 후 종료해도 무리가 없다.

그 이유는 다음과 같다.

- 실험의 큰 흐름이 이해 가능하고
- 마지막 완주 증거가 꽤 유의미하며
- 다음 단계가 무엇이었는지도 문서에 남아 있고
- 남은 병목이 명확하며
- 원래 handover 저장소는 역사적 산출물이 과도하게 많기 때문이다

가장 적절한 역사적 요약은 다음 한 문장이다.

이 프로젝트는 부분적으로 성공한 robustness 실험이었다. 더 강한 Saga-style 제어는 마지막 완주 unseen 실험에서 더 단순한 baseline보다 실제 개선을 보였지만, 운영 품질 기준을 완전히 만족시키지는 못해 유망하지만 미완의 상태에서 종료되었다.
