"""Objective tracking, the stopping rule and the monotonicity guard used by every solver."""

from __future__ import annotations

import warnings

import numpy as np

from ..exceptions import MonotonicityWarning


class ConvergenceMonitor:
    """Record a minimized objective and decide when to stop.

    Every solver in this package minimizes its objective: a negative log-likelihood, a
    negative log-posterior or a negative ELBO. MM and EM steps never increase it and
    coordinate-ascent variational steps never decrease the ELBO, so any increase beyond
    ``monotone_rtol`` signals a bug and raises a :class:`~probpca.exceptions.MonotonicityWarning`.

    Parameters
    ----------
    tol : float, default=1e-6
        Stop when the relative change of the objective stays below ``tol`` for ``patience``
        consecutive updates.
    max_iter : int, default=1000
        Stop after this many updates whether or not the objective has converged.
    patience : int, default=3
        Number of consecutive small changes required to declare convergence.
    monotone : bool, default=True
        Check that the objective never increases. Turn off only for solvers that are not
        monotone by design, such as samplers.
    monotone_rtol : float, default=1e-10
        Relative increase tolerated before warning, to absorb floating-point rounding.

    Attributes
    ----------
    history : ndarray of shape (n_updates,)
        Objective after each update.
    n_iter : int
        Number of updates recorded.
    converged : bool
        Whether the stopping rule was met before ``max_iter``.
    violations : list of int
        Updates at which the objective increased beyond ``monotone_rtol``.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        patience: int = 3,
        monotone: bool = True,
        monotone_rtol: float = 1e-10,
    ) -> None:
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}.")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}.")
        self.tol = tol
        self.max_iter = max_iter
        self.patience = patience
        self.monotone = monotone
        self.monotone_rtol = monotone_rtol
        self._history: list[float] = []
        self._small = 0
        self.converged = False
        self.violations: list[int] = []

    @property
    def history(self) -> np.ndarray:
        return np.asarray(self._history, dtype=np.float64)

    @property
    def n_iter(self) -> int:
        return len(self._history)

    def update(self, value: float) -> bool:
        """Record the objective after one update; return True when the solver should stop."""
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError(f"objective became {value} at update {self.n_iter + 1}.")
        if self._history:
            prev = self._history[-1]
            scale = max(abs(prev), np.finfo(np.float64).tiny)
            if self.monotone and value - prev > self.monotone_rtol * scale:
                self.violations.append(self.n_iter + 1)
                warnings.warn(
                    f"objective increased from {prev!r} to {value!r} at update "
                    f"{self.n_iter + 1}; the solver should be monotone.",
                    MonotonicityWarning,
                    stacklevel=2,
                )
            self._small = self._small + 1 if abs(prev - value) <= self.tol * scale else 0
        self._history.append(value)
        if self._small >= self.patience:
            self.converged = True
            return True
        return self.n_iter >= self.max_iter
