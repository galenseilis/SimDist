# Your first distribution

In this tutorial you will install SimDist, create a distribution, draw samples
from it, and inspect its properties. By the end you will have a mental model of
how the library is organised.

This tutorial takes about ten minutes. Every step is meant to be typed into a
Python session, so start one now.

## 1. Install SimDist

If you use [uv](https://docs.astral.sh/uv/):

```bash
uv add simdist
```

If you use pip:

```bash
pip install simdist
```

If you are working from a clone of the repository, sync the environment
instead:

```bash
uv sync
uv run python
```

## 2. Import a distribution

SimDist is organised into three modules: `dists` for concrete distributions,
`core` for the base class and combinators, and `rejectors` for rejection
sampling. Concrete distributions live in `simdist.dists`.

```python
import numpy as np
from simdist import dists
```

## 3. Create a distribution

Distributions are objects. Parameters are passed to the constructor, and the
optional `rng` argument takes a `numpy.random.Generator`. Passing a seeded
generator makes your results reproducible.

Let's model the time between arrivals at a queue with an exponential
distribution with rate 2.0 (a mean of 0.5 time units).

```python
from simdist import dists

wait = dists.Exponential(rate=2.0, rng=np.random.default_rng(0))
```

If you omit `rng`, SimDist creates a fresh generator for you, so you never have
to worry about a `None` generator.

## 4. Draw samples

Call `sample()` to draw a single value.

```python
wait.sample()
```

```
0.3399659519844548
```

Call it repeatedly to get a sequence. Use `batch_sample(size=...)` to draw
many at once.

```python
draws = wait.batch_sample(size=5)
draws
```

```
[0.5097985507329323, 0.009903331294527676, 0.0011346633401140911,
 0.2751714363195241, 0.8149702173291926]
```

Because we seeded the generator with `0`, you will get these same numbers if
you run the code yourself.

## 5. Read off properties

Distributions expose the quantities you would expect from a probability
textbook. Not every distribution implements every quantity; the ones that are
undefined (the mean of a Cauchy distribution, for example) raise
`NotImplementedError`.

```python
wait.mean()
```

```
0.5
```

```python
wait.variance()
```

```
0.25
```

```python
wait.median(), wait.mode(), wait.skewness(), wait.excess_kurtosis()
```

```
(0.34657359027997264, 0, 2, 6)
```

A boolean describes whether the distribution is continuous (has a density)
or discrete (has a probability mass function):

```python
wait.is_absolutely_continuous()
```

```
True
```

Other characterisations, like infinite divisibility, are implemented by
distributions for which they apply:

```python
dists.Poisson(rate=2.0).is_infinitely_divisible()
```

```
True
```

## 6. Where to go next

You now know the core loop: construct, sample, inspect. From here you can:

- Browse the [Reference](../reference/dists.md) to find another distribution.
- Follow [Composing distributions](composing-distributions.md) to build larger
  models.
- Read [Architecture](../explanation/architecture.md) to see how it all fits
  together.
