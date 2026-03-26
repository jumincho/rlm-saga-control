# Project Closure Report

## What This Project Was

This repository is a handover bundle for a line of experiments asking whether adding a Saga-style control layer to `RLM` improves robustness under disruption-heavy planning tasks.

The central idea was not generic “prompting” or “agent tuning.” The project tested whether transaction-like controls, validation, repair, split handling, prefix locking, and retry policy could make the system more stable when plans cross a disruption boundary and must preserve invariants.

In the language used inside the project, the important question was whether richer Saga-style variants could outperform plain `RLM` and simpler split-only baselines on hard boundary tasks.

## The Core Hypothesis

The working hypothesis evolved, but by the final phase it was roughly this:

- simple split/compensation should already improve boundary performance over plain `RLM`
- adding stronger Saga controls such as prefix lock, validation, and repair should improve further
- however, those gains only matter if state consistency, immutability, gate alignment, and latency remain acceptable

So the project was not only about success rate. It was about success rate under operational and invariant constraints.

## What The System Compared

The codebase defines several variants, but the final experiment line mostly centered on these:

- `V0`: plain `RLM`
- `V0_SPLIT_ONLY`: baseline with split handling only
- `V3_PREFIX_SPLIT`: stronger Saga-style variant with prefix lock, split handling, validation, and repair logic

The experiments were run mainly on `P8` boundary-crossing scenarios using the `realm` benchmark loader, with runtime and strict offline evaluation.

## What Was Measured

The project tracked more than final success.

Important metrics included:

- runtime success
- strict success
- gate recall and gate precision
- split applied rate
- split marker survived rate
- state consistency
- immutability after split
- fallback rate
- wall-clock latency
- LLM call count

This matters because the project’s claim was operational, not just academic. A variant that solves more cases but breaks invariants or explodes latency was not considered sufficient.

## How The Project Evolved

From the preserved handover notes, the sequence was:

1. `v7.1` to `v7.3`
   Debugged whether boundary split was being applied at all and fixed the early “split applied = 0” class of failures.

2. `v7.4` to `v7.6`
   Locked holdout/evaluator/paired-statistics protocol and separated split-only gains from stronger Saga-style gains.

3. `v7.7` to `v7.9`
   Focused on state and immutability bottlenecks, strict-evaluator failures, and latency decomposition.

4. `v7.10` to `v7.12`
   Improved gate alignment and reduced the “strict pass but gate fail” mismatch structure.

5. `v7.13`
   Produced the last fully completed and interpretable run, including unseen `test_v8` evidence showing that `V3_PREFIX_SPLIT` still beat `V0_SPLIT_ONLY`.

6. `v7.14`
   Prepared code and config for further deterministic `photo_time` repair and gate false-positive analysis, but this phase was not completed.

## The Most Important Completed Result

The single most important completed run preserved by this bundle is:

- `experiments/rlm_saga_v1_feedback_v7_13_full_20260303_054205`

That run is the best final evidence for the project because:

- it completed end to end
- it included diagnostics and paired statistics
- it evaluated on unseen `boundary_test_v8`
- it reflected the later, more mature experimental design

## What v7.13 Showed

On the final unseen `boundary_test_v8` stage, the key reported results were:

- `V0`
  - runtime success: `0.0000`
  - strict success: `0.0000`

- `V0_SPLIT_ONLY`
  - runtime success: `0.2667`
  - strict success: `0.2000`

- `V3_PREFIX_SPLIT`
  - runtime success: `0.4667`
  - strict success: `0.4167`

Paired comparisons for `V0_SPLIT_ONLY` vs `V3_PREFIX_SPLIT` on unseen `test_v8` were positive:

- runtime mean difference: `+0.2000`, 95% CI `[0.0333, 0.3667]`, `p = 0.0357`
- strict mean difference: `+0.2167`, 95% CI `[0.0500, 0.3833]`, `p = 0.0241`

This means the project did produce real positive evidence for the stronger Saga-style variant over split-only on the final completed unseen test.

## Why The Project Still Stopped

Although the direction of the results was positive, the project did not reach a “done” state by its own operational standards.

The preserved `v7.13` final report explicitly marked several DoD items as missed:

- runtime success target was missed
- wall-time target was missed
- gate precision target was missed
- fallback-rate target was missed

The same report shows:

- gate recall on strict pass was excellent: `1.0000`
- split applied rate was `1.0000`
- split marker survived rate was `1.0000`
- strict success crossed the stated target

So the project did not fail in the sense of “no benefit.” It failed in the sense of “not yet robust enough and not yet operationally clean enough.”

## The Main Bottlenecks At Closure

The final completed evidence and handover notes point to a short list of recurring bottlenecks:

- `photo_time` violations
- immutable-past corruption or disruption-term bleed
- state timeline inconsistency around the disruption boundary
- low gate precision, meaning too many gate-pass but strict-fail cases
- fallback and latency overhead

In other words, the project had already solved the earliest “does split even help?” question, but had not fully solved the harder “can the stronger system be both better and operationally clean?” question.

## The Status Of v7.14

`v7.14` is important because it shows where the project intended to go next, but it should not be mistaken for completed evidence.

The preserved notes describe `v7.14` as:

- code-prepared
- configured for new deterministic `photo_time` repair
- extended for gate false-positive breakdown
- not fully run to completion

So the honest interpretation is:

- `v7.13` is the last completed, interpretable evidence
- `v7.14` is the latest prepared but incomplete code state

## The Best Final Interpretation

The strongest defensible conclusion at project closure is:

Adding stronger Saga-style control to `RLM` did produce meaningful improvement over plain `RLM` and over split-only in the last completed unseen experiment, but the system still fell short of its own operational targets because gate precision, fallback behavior, `photo_time`, state consistency edge cases, and latency remained unresolved enough to block a clean finish.

That is stronger than “nothing worked,” but weaker than “the system is ready.”

## What This Bundle Preserves

This closure archive is intentionally much slimmer than the original handover repository.

It preserves:

- the experiment framework under `code/exp`
- the original top-level README and handover note
- the run manifest
- a small set of representative final evidence files from the last completed run
- this closure report in English and Korean

It does not preserve the full historical `experiments/` tree.

That is deliberate. The goal is to retain the runnable framework and the final interpretation, not every intermediate artifact.

## Important Portability Caveat

The preserved code is not fully portable without edits.

Many scripts and configs assume server-local absolute paths such as:

- `/disk/chojm/exp/...`
- `/disk/chojm/experiments/...`
- `/disk/chojm/.cache/...`

The framework also assumes:

- a vLLM-serving setup
- environment-variable credentials
- an 8-GPU launch path in its main runner scripts

So the preserved code is suitable for archival understanding and careful reactivation, but not for immediate plug-and-play execution on a fresh machine.

## If Someone Wanted To Reopen The Project

A future restart should begin from this framing:

1. treat `v7.13` as the last completed evidence baseline
2. treat `v7.14` as a prepared but unfinished next step
3. focus first on:
   gate precision,
   deterministic `photo_time` repair,
   fallback reduction,
   and latency reduction
4. keep unseen holdout discipline fixed

The wrong way to restart would be to ignore the operational metrics and look only at headline success-rate gains.

## Included Evidence Worth Reading First

Within the slim archive, the highest-value evidence files are:

- the original handover note
- the `v7.13` full execution report
- the `gate_alignment` report for `boundary_test_v8`
- the `state_timeline_failure_breakdown` report for `boundary_test_v8`
- the `metrics_runtime.csv` and `metrics_strict.csv` files for `v7_13_boundary_test_v8`
- the paired comparison JSON files for `V0_SPLIT_ONLY` vs `V3_PREFIX_SPLIT`

Those files are enough for a newcomer to understand both the positive signal and the remaining blockers.

## Final Closure Recommendation

This project is reasonable to archive and retire in its current state.

Why:

- the main experimental arc is understandable
- the last completed evidence is informative
- the next step was known but not executed
- the remaining blockers are clear
- the original bundle contains far more historical material than is needed for future understanding

The right historical summary is:

This was a partially successful robustness experiment. Stronger Saga-style control improved boundary-task outcomes over simpler baselines, including on the last completed unseen test, but the system still missed important operational quality targets, so the project stopped at a promising but unfinished stage.
