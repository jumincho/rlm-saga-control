"""RLM+Saga experiment package.

This package owns the experiment-side code for the rlm-saga-control study:
benchmark loaders (`bench/`), per-variant runners (`runners/`), and the
analysis scripts that turn raw run JSONL into the closure-report tables
(`analysis/`). The package presupposes an external `rlm` package (the LLM
scenario reasoner) and a separate Saga/RLM serving environment configured
via the `SAGA_*` and `RLM_SAGA_*` environment variables documented in the
top-level README and GLOSSARY.
"""
