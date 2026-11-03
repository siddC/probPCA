"""Test double for the base class: a rank-one Gaussian model, X = w hᵀ + noise.

It is not a model of this package. It exists so the API contract, the estimator checks and
the six test layers can be exercised before any real model lands, and it stays as a fast
check of the harness itself. Fit by alternating least squares over observed entries, which
never increases the profiled negative log-likelihood.
"""

import numpy as np
from sklearn.utils import check_random_state

from probpca import BaseFactorization
from probpca._base import FITTED_ATTRIBUTES
from probpca._core import EPS, ConvergenceMonitor, n_observed


class ToyRankOne(BaseFactorization):
    def __init__(self, max_iter=200, tol=1e-10, random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _weights(X, mask):
        return np.ones(X.shape) if mask is None else mask.astype(np.float64)

    def _solve_w(self, X, M, h):
        return (M * X) @ h / np.maximum(M @ (h * h), EPS)

    def _fit(self, X, mask):
        X = X.astype(np.float64)
        M = self._weights(X, mask)
        n_obs = n_observed(X, mask)
        rng = check_random_state(self.random_state)
        h = rng.standard_normal(X.shape[1])
        h /= np.linalg.norm(h)
        monitor = ConvergenceMonitor(tol=self.tol, max_iter=self.max_iter)
        while True:
            w = self._solve_w(X, M, h)
            h = (M * X).T @ w / np.maximum(M.T @ (w * w), EPS)
            norm = np.linalg.norm(h)
            if norm > 0:
                h, w = h / norm, w * norm
            rss = float(np.sum(M * (X - np.outer(w, h)) ** 2))
            sigma2 = max(rss / n_obs, EPS)
            if monitor.update(0.5 * n_obs * (np.log(2 * np.pi * sigma2) + 1.0)):
                break
        self.components_ = h[None, :]
        self.n_components_ = 1
        if "mixing_" in FITTED_ATTRIBUTES:  # ICA packages also report the mixing matrix
            self.mixing_ = np.linalg.pinv(self.components_)
        self.noise_variance_ = sigma2
        self.latent_scale_ = float(np.sqrt(np.mean(w * w)))
        self.objective_ = monitor.history
        self.n_iter_ = monitor.n_iter
        self.converged_ = monitor.converged
        return w[:, None]

    def _transform(self, X, mask):
        X = X.astype(np.float64)
        return self._solve_w(X, self._weights(X, mask), self.components_[0])[:, None]

    def _inverse_transform(self, Z):
        return Z @ self.components_

    def _score_samples(self, X, mask):
        X = X.astype(np.float64)
        M = self._weights(X, mask)
        R = X - self._inverse_transform(self._transform(X, mask))
        s2 = self.noise_variance_
        return -0.5 * np.sum(M * (np.log(2 * np.pi * s2) + R * R / s2), axis=1)

    def _sample(self, n_samples, random_state):
        w = random_state.normal(0.0, self.latent_scale_, size=(n_samples, 1))
        noise = random_state.normal(0.0, np.sqrt(self.noise_variance_),
                                    size=(n_samples, self.components_.shape[1]))
        return self._inverse_transform(w) + noise


def simulate(seed, n=80, p=30, missing=0.1):
    """Rank-one Gaussian data with a random mask; returns (X, mask, true components)."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(p)
    h /= np.linalg.norm(h)
    w = rng.normal(0.0, 5.0, n)
    X = np.outer(w, h) + rng.normal(0.0, 0.1, (n, p))
    mask = rng.random((n, p)) >= missing
    return X, mask, h[None, :]


def stationarity(est, X, mask):
    """Gradient of the residual sum of squares in h at the fitted point (w from transform)."""
    M = np.ones(X.shape) if mask is None else mask.astype(np.float64)
    h = est.components_[0]
    w = est.transform(X, mask=mask)[:, 0]
    grad = -2.0 * (M * (X - np.outer(w, h))).T @ w
    scale = 2.0 * float(np.max((M * np.abs(X)).T @ np.abs(w)))
    return [(h, grad, False, scale)]
