API contract
============

Every estimator derives from :class:`probpca.BaseFactorization` and behaves the same way.

Orientation
-----------

``X`` has shape (n_samples, n_features), as in scikit-learn. ``transform`` returns
per-sample latent values of shape (n_samples, k), and ``components_`` has shape
(k, n_features). Papers that use features as rows are converted in each model's derivation.

Missing data
------------

Pass missing entries as NaN in ``X`` or as ``False`` in a boolean ``mask`` of the same shape.
An entry is observed only if it is not NaN and the mask (when given) is True there. Every
likelihood, update and score sums over observed entries only, so values at unobserved
entries never influence a fit. For held-out scoring, fit with a mask and score with its
complement.

Members
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Contract
   * - ``fit(X, y=None, mask=None)``
     - Fit the model; returns self
   * - ``transform(X, mask=None)``
     - Latent values for new samples, with the fitted components held fixed
   * - ``fit_transform(X, mask=None)``
     - Fit, then return latent values for the training samples
   * - ``inverse_transform(Z)``
     - Mean parameter on the data scale (the reconstruction)
   * - ``score(X, mask=None)``
     - Mean log-likelihood per observed entry; for Bayesian models, the posterior-predictive
       log-likelihood
   * - ``score_samples(X, mask=None)``
     - Log-likelihood per sample
   * - ``sample(n_samples, random_state=None)``
     - Synthetic data from the fitted model (posterior predictive for Bayesian models)
   * - ``components_``, ``n_components_``
     - Fitted components; effective number of components after any ARD pruning
   * - ``objective_``, ``n_iter_``, ``converged_``
     - Minimized objective per iteration (negative log-likelihood, negative log-posterior or
       negative ELBO), iterations used, convergence flag
   * - ``loglik_``
     - Final training log-likelihood
   * - Bayesian extras
     - Posterior means and variances of factors (``components_var_``), ``elbo_`` trace,
       ``relevance_`` (ARD precisions)

Numerics
--------

float64 by default; float32 input stays float32. Logs, digamma and divisions are floored at
1e-12. ``random_state`` controls every random choice, and ``n_init`` restarts keep the run
with the best final objective.
