# probPCA

[![CI](https://github.com/siddC/probPCA/actions/workflows/ci.yml/badge.svg)](https://github.com/siddC/probPCA/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/probpca/badge/?version=latest)](https://probpca.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Python 3.12–3.14](https://img.shields.io/badge/python-3.12%E2%80%933.14-blue.svg)

Probabilistic and Bayesian PCA: PPCA, mixtures of PPCA, GLM-PCA, logistic PCA and Bayesian PCA with automatic rank selection, missing data and held-out likelihood, with a scikit-learn API.

> **Status: pre-alpha.** The shared engine, the API contract and the test harness for the
> correctness standard are in place. The first models land in v0.1; see the
> [roadmap](ROADMAP.md).

## Why probPCA

Every model is probabilistic PCA or a generalization of it: a likelihood matched to the
data type (Gaussian, counts, binary), mixtures for curved data, and Bayesian versions
that choose the rank. Missing entries are handled under the model, not imputed first,
and every model reports a held-out log-likelihood. Plain PCA, factor analysis and
Minka's automatic dimensionality stay in scikit-learn; Ilin & Raiko's VB-PCA stays in
vbpca-py.

The fitted solution is a subspace (a point on the Grassmann manifold).

## Models

| Class | Model | Inference | Source papers | Release |
| --- | --- | --- | --- | --- |
| `PPCA` | x = mu + Lᵀz + e, isotropic Gaussian noise | Closed form; EM with masks | Tipping & Bishop 1999a; Roweis 1998 | v0.1 |
| `GLMPCA` | Poisson, NB, Bernoulli, binomial, multinomial; offsets and covariates | Fisher scoring with a small ridge penalty | Townes 2019; Townes et al. 2019; Collins, Dasgupta & Schapire 2001 | v0.1 |
| `MPPCA` | Mixture of PPCA models | EM | Tipping & Bishop 1999b | v0.2 |
| `LogisticPCA` | Bernoulli; projection of the saturated model's natural parameters | MM | Landgraf & Lee 2020; Schein, Saul & Ungar 2003 | v0.2 |
| `BayesianPCA` | PPCA with ARD priors on the loading vectors | MAP by EM with type-II ML (default); variational Bayes | Bishop 1999a, 1999b; Oba et al. 2003 | v0.2 |
| `BayesianMPPCA` | Mixture of factor analysers with ARD and a Dirichlet prior | Variational Bayes with birth and death | Ghahramani & Beal 2000 | v0.3 |
| `BayesianGLMPCA` | Exponential-family PCA with priors on scores and loadings | NUTS (NumPyro, `[mcmc]` extra) | Mohamed, Heller & Ghahramani 2008 | v0.4 |

Full citations are on each model's documentation page.

## One API for every model

```python
model = SomeModel(n_components=8, random_state=0)
Z = model.fit_transform(X, mask=observed)   # X: (n_samples, n_features); Z: (n_samples, k)
model.components_                           # (n_components_, n_features)
model.score(X_test, mask=heldout)           # mean held-out log-likelihood per observed entry
model.sample(100, random_state=1)           # synthetic data from the fitted model
```

| Member | Contract |
| --- | --- |
| `fit(X, y=None, mask=None)` | Fit the model; returns self |
| `transform(X, mask=None)` | Latent values for new samples, components held fixed |
| `fit_transform(X, mask=None)` | Fit, then return the latent values of the training samples |
| `inverse_transform(Z)` | Mean of the observation model on the data scale |
| `score(X, mask=None)` | Mean log-likelihood per observed entry (posterior predictive for Bayesian models) |
| `score_samples(X, mask=None)` | Log-likelihood per sample |
| `sample(n_samples, random_state=None)` | Synthetic data from the fitted model |
| `components_`, `n_components_` | Fitted components; effective number after any ARD pruning |
| `objective_`, `n_iter_`, `converged_` | Minimized objective per iteration, iterations used, convergence flag |
| `loglik_` | Final training log-likelihood |

Missing entries are NaN in `X` or `False` in `mask`; every likelihood, update and score sums
over observed entries only. float64 by default; float32 input stays float32.

## Correctness

No estimator ships without passing six layers of tests:

1. **Monotone objective**: MM and EM objectives never increase, ELBOs never decrease.
2. **Limiting cases**: each model reduces to its documented special case.
3. **Reproduction**: each source paper's experiments are rerun and must match the published
   numbers to the precision the paper reports.
4. **Recovery**: factors and rank are recovered from data simulated by the model, over 10 seeds.
5. **Stationarity**: KKT conditions hold at convergence.
6. **Conformance**: scikit-learn's estimator checks, pickling, float32 and float64.

Models are implemented from their papers only (no author code is read or translated).
Maintained public packages may be run as black-box cross-checks on shared inputs.

## Install

probPCA is not on PyPI yet. Until the first release:

```bash
pip install "git+https://github.com/siddC/probPCA"
```

Optional extras: `[jax]` for the JAX backend and `[mcmc]` (NumPyro) for the models that
sample. Python 3.12 to 3.14.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Releases follow [RELEASING.md](RELEASING.md).

## Citation

If you use probPCA, please cite it; [CITATION.cff](CITATION.cff) has the details. A JOSS
paper will follow v1.0.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
