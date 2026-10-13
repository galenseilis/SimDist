# Distributions

The `simdist.dists` module contains every concrete distribution. Every class
accepts an optional `rng: numpy.random.Generator` and defaults to a fresh
generator. Parameters are validated in the constructor and invalid values raise
`ValueError`.

For a quick overview of parameters, see the [catalogue](index.md#distribution-catalogue).

::: simdist.dists
    options:
      show_root_heading: false
      members:
        - Bernoulli
        - Rademacher
        - Categorical
        - BetaBinomial
        - Binomial
        - Empirical
        - DiscreteUniform
        - Hypergeometric
        - NegativeHypergeometric
        - PoissonBinomial
        - FisherNoncentralHypergeometric
        - WalleniusNoncentralHypergeometric
        - Benford
        - Geometric
        - NegativeBinomial
        - Poisson
        - Zipf
        - IdealSoliton
        - RobustSoliton
        - ContinuousUniform
        - Triangular
        - Normal
        - LogNormal
        - Exponential
        - Laplace
        - Logistic
        - Cauchy
        - Gamma
        - Beta
        - ChiSquared
        - StudentT
        - Weibull
        - Rayleigh
        - Pareto
        - VonMises
      members_order: source
      heading_level: 2
