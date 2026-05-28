"""Post-run analysis: re-scoring, paired stats, and diagnostic breakdowns.

Modules in this package take the raw `baseline.jsonl` + `extension.jsonl`
that runners produce and turn them into the per-variant summary tables and
diagnostic reports that the closure report quotes — paired win/loss with
sign-test and bootstrap CI, gate alignment against the strict offline
evaluator, gate false-positive breakdown, immutability-prefix taxonomy,
state-timeline failure breakdown, and the headline aggregate summary. No
network calls; pure functions over JSONL on disk.
"""
