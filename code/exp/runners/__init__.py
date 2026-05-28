"""Experiment runners: drive baseline, extension, and paired comparisons.

Each runner module is a thin `__main__` over `common.py`, which holds the
heavy shared machinery (config loading, sample loading, failure injection,
the per-sample run loop with gate/retry/recovery, and the JSONL writer).
`run_baseline.py` runs plain RLM (V0); `run_extension.py` runs the Saga
variants (V1/V2/V3); `run_paired.py` runs V0 plus selected extensions on
the same sample stream so the result tables can be paired record-by-record.
`run_all.py` chains baseline → extension → summary for a single stage.
The `run_*_with_cleanup.sh` bash wrappers next to these are historical
launch scripts preserved as a record of how each round was actually run.
"""
