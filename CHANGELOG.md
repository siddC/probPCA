# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/); the API is frozen at v1.0.

## 0.1.0 (unreleased)

### Added

- `BaseFactorization`, the base estimator implementing the shared API contract: `fit`,
  `transform`, `fit_transform`, `inverse_transform`, `score` (mean log-likelihood per
  observed entry), `score_samples` and `sample`, with NaN or boolean masks for missing
  entries and float32 or float64 input.
- `ConvergenceMonitor` (relative-tolerance stopping rule with patience, and a monotonicity
  guard that raises `MonotonicityWarning`), and floors for logs and divisions.
- `metrics`: `match_components`, `principal_angles`.
- `utils.estimator_checks`: API contract, monotone objective, mask equivalence, pickling,
  recovery and stationarity checks.
- Test harness for the six layers of the correctness standard, with a model registry and
  reproduction fixtures.
- CI, tag-triggered trusted publishing to PyPI, documentation on Read the Docs.
