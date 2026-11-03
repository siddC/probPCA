"""Warnings raised by probpca."""

from sklearn.exceptions import ConvergenceWarning

__all__ = ["ConvergenceWarning", "MonotonicityWarning"]


class MonotonicityWarning(RuntimeWarning):
    """An objective that should never increase did increase.

    MM, EM and coordinate-ascent variational solvers are monotone by construction, so this
    warning points to a bug in a derivation or its implementation. The test suite turns it
    into an error.
    """
