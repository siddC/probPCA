probPCA
=======

Probabilistic and Bayesian PCA: PPCA, mixtures of PPCA, GLM-PCA, logistic PCA and Bayesian PCA with automatic rank selection, missing data and held-out likelihood, with a scikit-learn API.

.. note::

   Pre-alpha. The shared engine, the API contract and the test harness are in place; the
   first models land in v0.1.

Every model is probabilistic PCA or a generalization of it: a likelihood matched to the
data type (Gaussian, counts, binary), mixtures for curved data, and Bayesian versions
that choose the rank. Missing entries are handled under the model, not imputed first,
and every model reports a held-out log-likelihood. Plain PCA, factor analysis and
Minka's automatic dimensionality stay in scikit-learn; Ilin & Raiko's VB-PCA stays in
vbpca-py.

The fitted solution is a subspace (a point on the Grassmann manifold).

.. toctree::
   :maxdepth: 2

   install
   models
   api_contract
   correctness
   auto_examples/index
   api
