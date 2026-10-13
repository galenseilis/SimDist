# SimDist

**Probability distributions for simulations in Python.**

SimDist is a small, dependency-light (NumPy only) library of probability
distributions for building simulations. Distributions are plain objects that
you can sample, query for moments, and combine with ordinary arithmetic.

```python
import numpy as np
from simdist import dists

service_time = dists.Exponential(rate=2.0, rng=np.random.default_rng(0))
service_time.sample()          # one draw
service_time.mean()            # 0.5
service_time.batch_sample(size=5)
```

## Where to start

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Learn SimDist from scratch with a guided, hands-on lesson.

    [:octicons-arrow-right-24: Your first distribution](tutorials/your-first-distribution.md)

-   :material-hammer-wrench:{ .lg .middle } **How-to guides**

    ---

    Practical recipes for common simulation tasks.

    [:octicons-arrow-right-24: Truncate a distribution](how-to/rejection-sampling.md)

-   :material-book-open-variant:{ .lg .middle } **Reference**

    ---

    Complete API documentation for every class and function.

    [:octicons-arrow-right-24: Distributions API](reference/dists.md)

-   :material-lightbulb-on-outline:{ .lg .middle } **Explanation**

    ---

    Understand the design and the ideas behind the library.

    [:octicons-arrow-right-24: Architecture](explanation/architecture.md)

</div>

## Installing

```bash
uv add simdist
```

Or with pip:

```bash
pip install simdist
```

See [Install SimDist](how-to/install.md) for full details.

## What is inside

- **Discrete distributions** with finite support: `Bernoulli`, `Rademacher`,
  `Categorical`, `Binomial`, `BetaBinomial`, `Empirical`, `DiscreteUniform`,
  `Hypergeometric`, `NegativeHypergeometric`, `PoissonBinomial`, the
  noncentral hypergeometric pair (`FisherNoncentralHypergeometric`,
  `WalleniusNoncentralHypergeometric`), and `Benford`.
- **Discrete distributions** with infinite support: `Geometric`,
  `NegativeBinomial`, `Poisson`, `Zipf`, `IdealSoliton`, `RobustSoliton`.
- **Absolutely continuous distributions**: `ContinuousUniform`, `Triangular`,
  `Normal`, `LogNormal`, `Exponential`, `Laplace`, `Logistic`, `Cauchy`,
  `Gamma`, `Beta`, `ChiSquared`, `StudentT`, `Weibull`, `Rayleigh`, `Pareto`,
  and `VonMises`.
- **Combinators** for building new distributions: `Constant`, `Degenerate`,
  `Transform`, `Compose`, `Min`, `Max`, `Range`, and `FiniteMixture`.
- **Rejection sampling** helpers: `Reject`, `is_negative`, `outside_interval`.

!!! note "The documentation framework"
    These pages follow the [Diátaxis](https://diataxis.fr/) framework:
    tutorials for learning, how-to guides for doing, reference for looking
    things up, and explanation for understanding.
