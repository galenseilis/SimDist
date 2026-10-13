# Reference

Reference pages are **information-oriented**: they describe the API precisely
and completely. Use them to look up a signature, an argument, or what a method
returns.

The library is split into three modules:

| Module | Contents |
| --- | --- |
| [`simdist.core`](core.md) | The `Distribution` base class and the combinators `Constant`, `Degenerate`, `Transform`, `Compose`, `Min`, `Max`, `Range`, and `FiniteMixture`. |
| [`simdist.dists`](dists.md) | All concrete distributions. |
| [`simdist.rejectors`](rejectors.md) | `Reject` and the predicates `is_negative` and `outside_interval`. |

## Distribution catalogue

### Discrete, finite support

| Class | Parameters | Notes |
| --- | --- | --- |
| `Bernoulli` | `p` | Two-point: 0 or 1. |
| `Rademacher` | `p` | Two-point: −1 or 1. |
| `Categorical` | `probabilities`, `support=None` | General finite support. |
| `BetaBinomial` | `n`, `alpha`, `beta` | |
| `Binomial` | `num`, `prob` | |
| `Empirical` | `support=None`, `weights=None` | Resamples observed data. |
| `DiscreteUniform` | `lower=0`, `upper=1` | |
| `Hypergeometric` | `ngood`, `nbad`, `nsample` | |
| `NegativeHypergeometric` | `ngood`, `nbad`, `nfailures` | |
| `PoissonBinomial` | `probs` | |
| `FisherNoncentralHypergeometric` | `nwhite`, `nblack`, `nsample`, `odds` | |
| `WalleniusNoncentralHypergeometric` | `nwhite`, `nblack`, `nsample`, `odds` | |
| `Benford` | `base=10` | Leading-digit distribution. |

### Discrete, infinite support

| Class | Parameters | Notes |
| --- | --- | --- |
| `Geometric` | `prob` | |
| `NegativeBinomial` | `rate`, `prob` | |
| `Poisson` | `rate` | Sums with other `Poisson` distributions. |
| `Zipf` | `alpha` | `alpha` must be greater than 1. |
| `IdealSoliton` | `k` | |
| `RobustSoliton` | `k`, `c=0.1`, `delta=0.05` | |

### Absolutely continuous

| Class | Parameters | Notes |
| --- | --- | --- |
| `ContinuousUniform` | `lower`, `upper` | |
| `Triangular` | `low=0`, `high=1`, `mode=None` | |
| `Normal` | `mu`, `sigma` | |
| `LogNormal` | `mu`, `sigma` | |
| `Exponential` | `rate` | |
| `Laplace` | `location`, `scale` | |
| `Logistic` | `location`, `scale` | |
| `Cauchy` | `location`, `scale` | Mean and variance are undefined. |
| `Gamma` | `shape`, `scale` | |
| `Beta` | `alpha`, `beta` | |
| `ChiSquared` | `df` | |
| `StudentT` | `df` | |
| `Weibull` | `shape`, `scale=1` | |
| `Rayleigh` | `scale=1` | |
| `Pareto` | `shape`, `scale=1` | |
| `VonMises` | `mu`, `kappa` | Circular distribution. |

See [`simdist.dists`](dists.md) for full signatures, validation rules, and the
methods each distribution implements.
