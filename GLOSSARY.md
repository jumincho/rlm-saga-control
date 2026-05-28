# Glossary

The closure reports, the code rows, and the analysis tables in this repo
share a small private vocabulary that isn't self-explanatory if you come
in cold. This is the decoder ring.

## The three regimes

Every paired comparison in this project lines three things up against the
same disturbed sample stream:

| Regime | What is on the LLM | What gets recorded |
|---|---|---|
| **Hands-off** (`V0`) | Plain RLM, no Saga layer. Whatever the model emits is what we score. | The baseline column in every paired table. |
| **Light branching** (`V1`, `V2`) | A transactional wrapper (`V1`) and a rule-based runtime validator (`V2`). The validator can `accept` / `augment` (feedback + retry) / `reject` a candidate plan. No prefix lock, no deterministic recovery. | The intermediate column — how much you get from "just adding a validator". |
| **Full Saga control** (`V3` family) | `V2` + retry policy + **prefix lock** + boundary-crossing **split** + **deterministic recovery** routines (photo_time repair, timeline normalization, immutability-guarded best-of selection) + fallback. | The "full control" column. Variants like `V3_PREFIX`, `V3_PREFIX_SPLIT`, `V3_PREFIX_NO_SPLIT`, `V3_BASE` exist as ablations to attribute the gain to prefix-lock vs split vs both. |

The closure-report claim is that **full Saga control was meaningfully more
accurate than light branching** on unseen scenarios in paired comparison,
but did not clear the self-set operational targets (latency, false-positive
rate, fallback frequency).

## "Gate" and gate accuracy

The **boundary gate** is the runtime check that decides whether the
agent's candidate plan can be committed at the disruption boundary.
The check fires only on samples flagged "boundary-crossing applicable"
(notably REALM-Bench P8). The gate looks at five things and emits a
pass/fail with one or more reasons:

| Gate fail reason | What it means |
|---|---|
| `NON_MONOTONIC` | Per-actor timeline isn't monotonic across the alert minute (events overlap or run backwards). |
| `MISSING_BOUNDARY` | No event sits at the alert minute as the boundary handover. |
| `STATE_MISMATCH` | Per-actor state at the alert minute is inconsistent (e.g., same actor in two places). |
| `IMMUTABLE_TERMS` | Pre-alert events were modified — disruption terms leaked into the immutable past. |
| `MARKER_LOST` | `boundary_split_pre` / `boundary_split_post` notes survived the split application but not the final commit. |
| `PHOTO_TIME_EXCEEDED` | P8-specific: a scheduled event ends after `constraints.photo_time`. (Added in v7.14.) |

"**Gate accuracy**" is shorthand for two metrics over the gate's pass /
fail decision vs the offline strict judge: `gate_recall_on_strict_pass`
(fraction of strict-passing samples the gate also passed) and
`gate_precision_on_gate_pass` (fraction of gate-passing samples the
strict judge also accepts).

**Gate FP** = "gate-pass but strict-fail" = the dangerous direction; the
gate said the plan was committable but the strict scorer doesn't accept
it. `analysis/gate_fp_breakdown.py` decomposes those rows by the
top strict violations and the dominant state-failure reasons. The
operational target the closure report calls "not yet cleared" is
**lowering the gate FP rate** without losing recall.

## Immutability

In a disturbed scenario, the **immutable** part is the prefix of events
that already happened before the alert minute. The whole point of Saga
control is to leave that prefix alone and only repair the post-alert
remainder. Concretely "immutable" means three things, checked together:

1. The set of pre-alert events (events ending at or before the alert
   minute) is byte-equal to the locked-prefix snapshot taken at gate time
   (after canonicalization: per-actor sort, normalized text).
2. None of the disruption-related keywords (delay, traffic, closure,
   reroute, detour, reschedule, …) leak into pre-alert event notes.
3. Per-actor windows remain monotonic across the boundary.

`immutable_prefix_ok` is the soft check (no disruption term leak);
`immutable_prefix_after_split_ok` is the stricter check that *also*
requires the boundary split markers to be present so the pre/post
partition is well-defined. The closure report's "immutable preservation"
column is the joint pass rate of these two.

## Prefix lock

The **prefix lock** is the V3 mechanism that enforces "do not touch the
pre-alert events". At gate time the runner hashes the canonicalized
pre-alert prefix (the **locked prefix snapshot**, stamped into the per-
row `locked_prefix_snapshot` field) and refuses to commit any candidate
whose prefix hash drifts from that. The lock is what makes V3 "more than
just split"; ablations `V3_PREFIX` vs `V3_PREFIX_NO_SPLIT` exist
specifically to attribute the gain to the lock by itself.

## Recovery / deterministic recovery

When the gate fails, the runner doesn't immediately give up. It runs a
sequence of **deterministic recovery** routines — fixed transformations
the model isn't involved in — and re-tries the gate. The major ones the
diagnostics record are:

- `boundary_canonicalization_applied` — pin the boundary handover event
  to the alert minute exactly.
- `post_boundary_monotonic_fix_applied` — shift post-alert events to
  restore per-actor monotonicity.
- `timeline_norm_applied` (`timeline_norm_total_shift_minutes`,
  `timeline_norm_overlap_fixes_count`, `timeline_norm_location_fix_applied`)
  — generic time-shift / overlap-fix pass.
- `missing_boundary_event_autofixed` — when no event sits at the
  alert minute, synthesize a minimal one.
- `immutable_scope_sanitize_applied` (`immutable_scope_terms_removed_count`)
  — remove disruption terms that crept into pre-alert notes.
- `photo_time` deterministic repair (added in v7.14) — collapse
  schedule to fit `constraints.photo_time` without re-asking the model.

The "Recovery" column in the closure report is the rate at which the
gate eventually passed *because of* one of these fixes (the
`boundary_gate_pass_after_fix_iter` and `boundary_gate_fix_*` rates).

## Fallback / fallback call frequency

If, after all the deterministic recovery attempts plus a bounded
number of LLM-driven retries, the gate is still failing, V3 falls back
to V2-style behavior (light branching: split-only, no prefix lock)
and commits that. **`v3_fallback_to_split_only_rate`** is the
fraction of samples where that fallback triggered. "Fallback call
frequency" in the closure-report's operational table is the same
number. It is one of the targets that did *not* clear — V3's
fallback rate stayed too high to call the system "operations ready".

## Strict vs loose scoring (`bench/scorer.py`)

`exp/bench/scorer.py` is the single source of truth for what counts as
a successful plan repair. Its behavior pivots on a *policy*:

| Policy | When used | What is hard-failed |
|---|---|---|
| `runtime_v3` | The Saga validator live during the run | Nothing — recoverable violations are reported but not hard-failed. |
| `runtime_p8_hard_v1` | Auto-selected for P8 (boundary-crossing) samples in the runtime path | Disruption / immutable / state — same as strict, but live. |
| `strict_v1` | The offline judge that produces the closure-report numbers | Disruption (`crossing_split_applicable` and `disruption_applicable`), immutable prefix, state-at-alert consistency. |
| `relaxed_v1` | A sanity-check between runtime and strict | Disruption is hard, immutable and state are soft. Used to confirm strict failures aren't just strict being severe. |

In closure-report English:

- **Loose scoring** = the runtime view. Forgives boundary-state issues
  that could plausibly still recover; this is what the agent feels.
- **Strict scoring** = the offline view. Treats `crossing_split_applied`,
  `immutable_prefix_ok`, and `state_at_alert_consistent` as hard
  constraints. This is what the closure report's accuracy column means.

`analysis/summarize_dual_eval.py` re-scores the same raw JSONL under
all three modes so the headline tables can show the gap between them.

## External package dependencies (the `SAGA_*` env vars)

This repository is the **experiment-side** code (`code/exp/`). The
machinery it drives lives in two external packages that the v7.* runs
expected to be installed into the run's venv:

- **`SagaLLM`** — the Saga transaction / validator / recovery library
  the V1 / V2 / V3 variants wrap RLM with. Its requirements file path
  is pointed at by `SAGA_LLM_REQUIREMENTS` (default
  `/disk/chojm/SagaLLM/requirements.txt`).
- **`RLM`** — the underlying scenario reasoner. Its source root path
  is pointed at by `SAGA_RLM_PACKAGE` (default `/disk/chojm/rlm`) and
  the bootstrap installs it in editable mode (`pip install -e`).

Neither package is vendored here. If you want to actually re-run, you
need both checked out at compatible commits, the requirements file
present, and the env vars below pointing at them.

## Environment variables and `/disk/chojm/...` paths

The codebase has hard-coded absolute paths from the original author's
machine. They are all overridable via environment variables.

| Variable | What it points at | Default in code |
|---|---|---|
| `SAGA_EXPERIMENTS_ROOT` | The directory where each per-run folder is created (`rlm_saga_v1_<timestamp>/`). | `/disk/chojm/experiments` |
| `SAGA_LLM_REQUIREMENTS` | The requirements.txt installed at bootstrap (SagaLLM's). | `/disk/chojm/SagaLLM/requirements.txt` |
| `SAGA_RLM_PACKAGE` | The path to the RLM package to install editable. | `/disk/chojm/rlm` |
| `RLM_SAGA_RUN_ROOT_BASE` | Where Python runners (`run_baseline.py`, `run_extension.py`, `run_paired.py`) create their run roots. Takes precedence over `paths.run_root_base` in the YAML. | YAML `paths.run_root_base` (which itself defaults to `/disk/chojm/experiments`). |

`/disk/chojm/...` is the original author's filesystem layout on the
serving host. The repo will not run as-is in a fresh environment;
those env vars are the seam where you'd retarget it.

## `run_paired.py` — paired-comparison semantics

The headline-table runner. For every (seed, sample) it runs **V0
first**, then each of the selected extension variants on the *same*
`injected` sample object, so the resulting baseline.jsonl and
extension.jsonl are guaranteed to pair up on
`(sample_id, seed, stage)`. `analysis/check_pair_integrity.py` is the
gate that verifies that contract before any paired statistic gets
quoted (no duplicates, full intersection, equal unique-key counts).
The `--variants` list always includes V0 plus one or more extensions;
the v7.13 last completed paired round used `V0 + V3_PREFIX_SPLIT` as
its canonical pair.

## Closure reports

The two long-form writeups in `closure_reports/` are the canonical
project narrative; the artifacts under `evidence/` are the
representative outputs that the reports cite.
