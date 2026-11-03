# Reproduction of published results

Layer 3 of the correctness standard. Each source paper's experiments are rerun with the same
data, settings and metrics, and must match the reported numbers to the precision the paper
reports.

- `scripts/<name>.py` defines `run()`, which rebuilds one experiment's inputs and settings
  and returns a dict of metrics. Downloads go to `reproduction/.cache/` (not committed).
- `fixtures/<name>.json` stores the published values and their precision (see
  CONTRIBUTING.md). Optional black-box cross-check outputs from maintained public packages
  are stored here too.

`tests/test_reproduction.py` runs every fixture; it is marked `slow`.
