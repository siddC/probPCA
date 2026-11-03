"""Timing benchmarks for the shared engine (run with `asv run`)."""

import numpy as np

from probpca import BaseFactorization
from probpca._core import ConvergenceMonitor, validate_X_mask


class TimeValidation:
    params = [(500, 2000), (5000, 20000)]
    param_names = ["shape"]

    def setup(self, shape):
        rng = np.random.default_rng(0)
        self.X = rng.random(shape)
        self.mask = rng.random(shape) > 0.1

    def time_complete(self, shape):
        validate_X_mask(BaseFactorization(), self.X, None, reset=True)

    def time_masked(self, shape):
        validate_X_mask(BaseFactorization(), self.X, self.mask, reset=True)


class TimeMonitor:
    def time_10k_updates(self):
        mon = ConvergenceMonitor(tol=0.0, max_iter=10_000)
        for i in range(10_000):
            mon.update(1.0 / (i + 1))
