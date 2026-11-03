# Contributing to probPCA

## Setup

```bash
git clone https://github.com/siddC/probPCA && cd probPCA
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Before pushing:

```bash
ruff check .
mypy
pytest -m "not slow"   # fast suite; `pytest` runs everything, including recovery over 10 seeds
sphinx-build -W -b html docs docs/_build/html
```

`main` is protected: every change lands through a pull request, and the `ci-passed` check
must be green.

## Conventions

- **Orientation (scikit-learn):** `X` has shape (n_samples, n_features). `transform` returns
  per-sample latent values of shape (n_samples, k) and `components_` has shape
  (k, n_features). Papers that use features as rows or write V ≈ WHᵀ are converted in each
  derivation, and every docstring states the mapping.
- **Missing data:** `mask` is a boolean array the shape of `X`, True meaning observed. Every
  likelihood, update and score sums over observed entries only. NaN in `X` is accepted and
  converted to a mask. The base class does this; models receive `mask=None` (complete data)
  or a boolean array.
- **Determinism:** `random_state` controls every random choice. `n_init` restarts keep the
  run with the best final objective.
- **Precision:** float64 by default, float32 opt-in by passing float32 data. Logs, digamma and
  divisions are floored at `_core.EPS` (1e-12).
- **Objectives:** `objective_` is always minimized (negative log-likelihood, negative
  log-posterior or negative ELBO). Record it with `_core.ConvergenceMonitor`, which warns on
  any increase.
- **Acceleration:** numba for inner loops (a required dependency); JAX only behind the
  `[jax]` or `[mcmc]` extras.

## Adding a model

1. Write the model's documentation page first: generative model, objective, derivation of
   the updates in this package's orientation, algorithm, limiting cases, complexity and
   references.
2. Subclass `probpca.BaseFactorization` and implement its five hooks (`_fit`, `_transform`,
   `_inverse_transform`, `_score_samples`, `_sample`). `_fit` sets every attribute in
   `FITTED_ATTRIBUTES`.
3. Add a simulator to `probpca.datasets` that returns the data and the ground truth.
4. Add a `ModelSpec` to `tests/_registry.py`. That alone runs the model through layers 1, 4, 5
   and 6. List its limiting cases in the spec (layer 2).
5. Add a reproduction script and fixture (layer 3, below).
6. Add an entry to `CHANGELOG.md`.

### The correctness standard

| Layer | What must hold | Where |
| --- | --- | --- |
| 1. Monotone objective | Objective never increases (relative tolerance 1e-10), complete and masked data, plus random inputs | `tests/test_monotone.py` |
| 2. Limiting cases | The model reduces to its documented special case | `ModelSpec.limits` |
| 3. Reproduction | The source paper's experiments match the published numbers to the paper's precision | `reproduction/` |
| 4. Recovery | Factors (after Hungarian matching) and rank recovered on simulated data, 10 seeds | `tests/test_recovery.py` |
| 5. Stationarity | KKT conditions at convergence | `ModelSpec.stationarity` |
| 6. Conformance | scikit-learn `check_estimator`, pickling, float32 and float64, mask contract | `tests/test_sklearn.py` |

Papers usually report summary metrics rather than fitted factors, and rarely state their
random initialization, so layer 3 confirms the algorithm reaches the paper's results while
layers 1, 2, 4 and 5 carry the exact correctness checks.

### Reproduction fixtures

`reproduction/scripts/<name>.py` defines `run()`, which rebuilds one experiment's inputs and
settings and returns a dict of metrics. `reproduction/fixtures/<name>.json` stores the
published values:

```json
{
  "model": "ModelName",
  "paper": "Author et al. 2020, Table 2",
  "script": "<name>",
  "published": {"heldout_loglik": -1.234},
  "rtol": 0.0005
}
```

## Clean-room rule

Algorithms are implemented from the papers only: no author code is sought, read or
translated. Where a maintained public package already implements the same model, it may be
run as a black-box cross-check on shared inputs, never as a source. This also keeps the
code free of GPL obligations, since several of those packages are GPL.
