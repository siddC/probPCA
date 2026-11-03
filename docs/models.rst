Models
======

Every model states its likelihood, fits it with an objective that provably improves at each
iteration, handles missing entries and reports a held-out log-likelihood. Each model gets its
own page here when it is released.

.. list-table::
   :header-rows: 1
   :widths: 18 30 22 22 8

   * - Class
     - Model
     - Inference
     - Source papers
     - Release
   * - ``PPCA``
     - x = mu + Lᵀz + e, isotropic Gaussian noise
     - Closed form; EM with masks
     - Tipping & Bishop 1999a; Roweis 1998
     - v0.1
   * - ``GLMPCA``
     - Poisson, NB, Bernoulli, binomial, multinomial; offsets and covariates
     - Fisher scoring with a small ridge penalty
     - Townes 2019; Townes et al. 2019; Collins, Dasgupta & Schapire 2001
     - v0.1
   * - ``MPPCA``
     - Mixture of PPCA models
     - EM
     - Tipping & Bishop 1999b
     - v0.2
   * - ``LogisticPCA``
     - Bernoulli; projection of the saturated model's natural parameters
     - MM
     - Landgraf & Lee 2020; Schein, Saul & Ungar 2003
     - v0.2
   * - ``BayesianPCA``
     - PPCA with ARD priors on the loading vectors
     - MAP by EM with type-II ML (default); variational Bayes
     - Bishop 1999a, 1999b; Oba et al. 2003
     - v0.2
   * - ``BayesianMPPCA``
     - Mixture of factor analysers with ARD and a Dirichlet prior
     - Variational Bayes with birth and death
     - Ghahramani & Beal 2000
     - v0.3
   * - ``BayesianGLMPCA``
     - Exponential-family PCA with priors on scores and loadings
     - NUTS (NumPyro, ``[mcmc]`` extra)
     - Mohamed, Heller & Ghahramani 2008
     - v0.4
