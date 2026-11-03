"""Checks that every probpca estimator must pass.

They implement the generic parts of the correctness standard (see the docs): the API
contract, monotone objectives (layer 1), recovery from simulated data (layer 4), stationarity
(layer 5) and conformance (layer 6). Limiting cases (layer 2) and reproduction of published
results (layer 3) are specific to each model and live in its tests.

Each check raises ``AssertionError`` with an explanation when it fails.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.base import clone

from .._base import FITTED_ATTRIBUTES
from ..metrics import match_components

__all__ = [
    "check_api_contract",
        "check_mask_equivalence",
    "check_monotone_objective",
    "check_pickle",
    "check_recovery",
    "check_stationarity",
    "run_generic_checks",
]


def _with_nan(X, mask):
    X = np.array(X, dtype=np.float64, copy=True)
    X[~mask] = np.nan
    return X


def check_api_contract(estimator, X, mask=None):
    """Fit ``estimator`` and check every member of the API contract.

    Checks the fitted attributes and their shapes, and the shapes and finiteness of what
    ``transform``, ``fit_transform``, ``inverse_transform``, ``score``, ``score_samples`` and
    ``sample`` return.
    """
    X = np.asarray(X)
    n, p = X.shape
    est = clone(estimator)
    Z = est.fit_transform(X, mask=mask)
    for attr in FITTED_ATTRIBUTES:
        assert hasattr(est, attr), f"fitted estimator lacks {attr}"
    k = est.n_components_
    assert est.components_.shape == (k, p), (
        f"components_ has shape {est.components_.shape}, expected {(k, p)}"
    )
    if hasattr(est, "mixing_"):
        assert est.mixing_.shape == (p, k), (
            f"mixing_ has shape {est.mixing_.shape}, expected {(p, k)}"
        )
    assert Z.shape == (n, k), f"fit_transform returned shape {Z.shape}, expected {(n, k)}"
    assert est.transform(X, mask=mask).shape == (n, k)
    assert est.inverse_transform(Z).shape == (n, p)

    obj = est.objective_
    assert obj.ndim == 1 and obj.size >= 1, "objective_ must be a non-empty 1-D array"
    assert np.isfinite(obj).all(), "objective_ has non-finite values"
    assert 1 <= est.n_iter_, "n_iter_ must be at least 1"
    assert isinstance(est.converged_, (bool, np.bool_)), "converged_ must be a bool"
    assert np.isfinite(est.loglik_), "loglik_ is not finite"

    ss = est.score_samples(X, mask=mask)
    assert ss.shape == (n,) and np.isfinite(ss).all(), "score_samples must be finite, shape (n,)"
    n_obs = n * p if mask is None else int(np.asarray(mask).sum())
    np.testing.assert_allclose(
        est.score(X, mask=mask), ss.sum() / n_obs, rtol=1e-10,
        err_msg="score must be the mean log-likelihood per observed entry",
    )
    S = est.sample(5, random_state=0)
    assert S.shape == (5, p), f"sample(5) returned shape {S.shape}, expected {(5, p)}"
    np.testing.assert_array_equal(
        S, est.sample(5, random_state=0), err_msg="sample must be reproducible"
    )
    return est


def check_monotone_objective(estimator, X, mask=None, *, rtol=1e-10):
    """Layer 1: the objective never increases between iterations (relative tolerance ``rtol``).

    ``objective_`` is the minimized objective (negative log-likelihood, negative
    log-posterior or negative ELBO), so every step must be non-increasing.
    """
    est = clone(estimator).fit(X, mask=mask)
    obj = est.objective_
    steps = np.diff(obj)
    scale = np.maximum(np.abs(obj[:-1]), np.finfo(np.float64).tiny)
    bad = np.flatnonzero(steps > rtol * scale)
    assert bad.size == 0, (
        f"objective increased at iterations {(bad + 1).tolist()[:10]}: "
        f"{[(obj[i], obj[i + 1]) for i in bad[:3]]}"
    )
    return est


def check_mask_equivalence(estimator, X, mask, *, rtol=1e-10):
    """Passing NaN for missing entries and passing a mask give the same fit."""
    mask = np.asarray(mask, dtype=bool)
    a = clone(estimator).fit(X, mask=mask)
    b = clone(estimator).fit(_with_nan(X, mask))
    np.testing.assert_allclose(a.components_, b.components_, rtol=rtol, atol=1e-12)
    np.testing.assert_allclose(a.objective_, b.objective_, rtol=rtol)
    # values at unobserved entries must not influence the fit
    X_noise = np.array(X, dtype=np.float64, copy=True)
    X_noise[~mask] = np.abs(X_noise).max() * 7.0 + 3.0
    c = clone(estimator).fit(X_noise, mask=mask)
    np.testing.assert_allclose(
        a.components_, c.components_, rtol=rtol, atol=1e-12,
        err_msg="values at unobserved entries changed the fit",
    )


def check_pickle(estimator, X, mask=None):
    """A pickled and restored estimator gives the same output."""
    est = clone(estimator).fit(X, mask=mask)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_array_equal(est.transform(X, mask=mask), restored.transform(X, mask=mask))
    assert est.score(X, mask=mask) == restored.score(X, mask=mask)


def check_recovery(make_estimator, simulate, *, seeds=None, min_similarity=0.95,
                   metric="correlation"):
    """Layer 4: recover the true components of data simulated from the model.

    Parameters
    ----------
    make_estimator : callable
        ``make_estimator(seed)`` returns an unfitted estimator.
    simulate : callable
        ``simulate(seed)`` returns ``(X, mask, true_components)``; ``mask`` may be None.
    seeds : iterable of int, default=None
        Seeds to run; None means ``range(10)``.
    min_similarity : float, default=0.95
        Smallest matched absolute similarity accepted for every component, on every seed.
    metric : {"correlation", "cosine"}, default="correlation"
        Similarity used to match components.
    """
    worst = []
    for seed in range(10) if seeds is None else seeds:
        X, mask, truth = simulate(seed)
        est = make_estimator(seed).fit(X, mask=mask)
        truth = np.asarray(truth)
        assert est.n_components_ == truth.shape[0], (
            f"seed {seed}: inferred {est.n_components_} components, truth has {truth.shape[0]}"
        )
        _, _, sim = match_components(est.components_, truth, metric=metric)
        worst.append(float(np.min(np.abs(sim))))
    assert min(worst) >= min_similarity, (
        f"worst matched similarity {min(worst):.4f} < {min_similarity} (per seed: {worst})"
    )
    return worst


def check_stationarity(params, grad, *, nonnegative=False, scale=1.0, tol=1e-6):
    """Layer 5: first-order (KKT) conditions at a fitted point.

    ``grad`` is the gradient of the minimized objective at ``params``. For unconstrained
    parameters it must vanish. For nonnegative parameters it must vanish on free (positive)
    coordinates and be nonnegative on coordinates held at zero. "Vanish" means
    ``|grad| <= tol * scale``, where ``scale`` is the size of the terms the gradient is made
    of (for a gradient split into positive and negative parts, the largest positive part),
    so the check does not depend on the units of the data.
    """
    params = np.asarray(params, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    bound = tol * scale
    free = params > 0 if nonnegative else np.ones(params.shape, dtype=bool)
    if free.any():
        worst = np.abs(grad[free]).max()
        assert worst <= bound, (
            f"gradient on free coordinates reaches {worst:.3g} > {bound:.3g} "
            f"({worst / scale:.3g} relative to scale {scale:.3g})"
        )
    if (~free).any():
        lowest = grad[~free].min()
        assert lowest >= -bound, f"gradient on zero coordinates reaches {lowest:.3g} < {-bound:.3g}"


def run_generic_checks(estimator, X, mask=None):
    """Run every model-independent check on ``estimator`` with data ``X``.

    ``mask`` should leave some entries unobserved, so the missing-data path is checked.
    """
    check_api_contract(estimator, X)
    check_monotone_objective(estimator, X)
    check_pickle(estimator, X)
    if mask is not None:
        check_api_contract(estimator, X, mask=mask)
        check_monotone_objective(estimator, X, mask=mask)
        check_mask_equivalence(estimator, X, mask)
