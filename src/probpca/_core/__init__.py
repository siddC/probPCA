"""Engine shared by the estimators in this package: validation, numerics and convergence."""

from ._convergence import ConvergenceMonitor
from ._numerics import EPS, safe_divide, safe_log
from ._validation import n_observed, validate_X_mask

__all__ = [
    "EPS",
    "ConvergenceMonitor",
    "n_observed",
    "safe_divide",
    "safe_log",
    "validate_X_mask",
]
