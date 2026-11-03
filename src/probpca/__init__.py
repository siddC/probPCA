"""probpca: probabilistic and Bayesian PCA.

Probabilistic PCA and its generalizations as statistical models: mixtures,
exponential-family likelihoods for counts and binary data, and Bayesian versions
with automatic rank selection, all with missing data and held-out likelihood.

Every estimator shares one API: ``X`` is (n_samples, n_features), missing entries are NaN or
``mask=False``, ``score`` is the mean log-likelihood per observed entry, and ``sample`` draws
from the fitted model. See https://probpca.readthedocs.io for the model pages.
"""

from . import datasets, metrics
from ._base import BaseFactorization

__version__ = "0.1.0.dev0"

__all__ = ["BaseFactorization", "datasets", "metrics", "__version__"]
