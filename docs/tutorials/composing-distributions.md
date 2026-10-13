# Composing distributions

SimDist distributions are designed to be combined. In this tutorial you will
build up a small service-time model from simpler pieces, using arithmetic,
mixtures, and the composition operator.

## 1. Set up

```python
import numpy as np
from simdist import dists
from simdist.core import Constant, FiniteMixture, Max
```

## 2. Adding two Poisson distributions

When two distributions have a known rule for a combined distribution, SimDist
uses it. The sum of independent Poisson random variables is again Poisson, and
that behaviour is built in.

```python
a = dists.Poisson(rate=2.0)
b = dists.Poisson(rate=3.0)
combined = a + b

type(combined).__name__
```

```
'Poisson'
```

```python
combined.rate
```

```
5.0
```

In general, arithmetic on distributions produces a `Transform`: a distribution
whose sampling applies the operation to independent draws of its operands.
The `+`, `-`, `*`, `/` operators and `abs()` all work this way.

```python
base = dists.Normal(mu=10.0, sigma=2.0, rng=np.random.default_rng(0))
shifted = base + Constant(5)

type(shifted).__name__
```

```
'Transform'
```

```python
shifted.sample()
```

```
15.251460442186787
```

!!! note
    A `Transform` does not guess a closed-form distribution; it simply applies
    the operation to samples. That makes it general, but its moments are not
    available.

## 3. Building a mixture

Sometimes the population is heterogeneous: most jobs are fast, a few are slow.
A `FiniteMixture` draws a component according to `weights`, then samples from
that component. Weights are normalised for you.

```python
rng = np.random.default_rng(42)
fast = dists.Exponential(rate=4.0, rng=rng)
slow = dists.Exponential(rate=1.0, rng=rng)
service = FiniteMixture(weights=[0.7, 0.3], dists=[fast, slow], rng=rng)

service.mean()
```

```
0.475
```

The mean is exactly the weighted average of the component means:
`0.7 * 0.25 + 0.3 * 1.0 = 0.475`.

## 4. Composing distributions with `__call__`

Calling a distribution with another distribution (or an iterable of them)
builds a `Compose`. Each time it is sampled, it draws the parameters from the
provided distributions and then constructs `type(self)` from those parameters.

Here is a hierarchical model: the mean of a normal distribution is itself
uniform between 0 and 5, and its standard deviation is exponential.

```python
hier = dists.Normal()([dists.ContinuousUniform(0.0, 5.0), dists.Exponential(2.0)])

type(hier).__name__
```

```
'Compose'
```

```python
hier.sample()
```

```
2.046467806819607
```

Because the parameters change on every draw, the result is a distribution over
distributions, not a fixed normal.

## 5. Reducing several draws

The combinators `Min`, `Max`, and `Range` take the relevant order statistic of
several independent draws. For instance, the time until the *slower* of two
independent servers responds is the maximum of two response times.

```python
server_a = dists.Normal(mu=0.0, sigma=1.0, rng=np.random.default_rng(0))
server_b = dists.Normal(mu=0.0, sigma=1.0, rng=np.random.default_rng(1))
last_to_respond = Max([server_a, server_b])

last_to_respond.sample()
```

```
0.345584192064786
```

## 6. Where to go next

- See the full toolbox in the [Core API reference](../reference/core.md).
- Learn how to shape a distribution with [rejection sampling](../how-to/rejection-sampling.md).
- Read [Architecture](../explanation/architecture.md) to understand how
  combinators are implemented.
