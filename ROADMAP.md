# Roadmap

Release order is build priority. Each model joins the package when it passes all six layers
of the correctness standard (CONTRIBUTING.md). No dates are set.

## Scaffold (done)

- Base estimator implementing the API contract: validation, NaN and mask handling, float32
  and float64, scoring per observed entry, sampling
- Convergence monitor with the stopping rule and the monotonicity guard
- Metrics for matching fitted components to the truth
- Estimator checks and the six-layer test harness, run on a rank-one test double
- CI (lint, types, tests, slow tests, wheel smoke test, docs), tag-triggered release to PyPI

## Models

- **v0.1:** `PPCA`, `GLMPCA`.
- **v0.2:** `MPPCA`, `LogisticPCA`, `BayesianPCA`.
- **v0.3:** `BayesianMPPCA`.
- **v0.4:** `BayesianGLMPCA`.
- **v1.0:** API freeze, complete model documentation pages, JOSS submission.

## CI matrix

The matrix follows boolmf: ubuntu-latest on Python 3.12, 3.13 and 3.14 until v0.2.

- After v0.2, CI adds `macos-latest` and `windows-latest` for every supported Python version.
- After v0.3, CI adds Python 3.15.
