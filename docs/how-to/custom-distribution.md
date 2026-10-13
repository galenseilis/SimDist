# Write your own distribution

You can add a distribution by subclassing `simdist.core.Distribution`. Only
`sample` is required; every other method has a sensible fallback (it raises
`NotImplementedError` if it cannot be computed).

This guide implements a `LogUniform` distribution — uniform on a logarithmic
scale — on the interval `[low, high]`.

## 1. Implement the class

```python
import numpy as np
from simdist.core import Distribution


class LogUniform(Distribution):
    """Uniform distribution on a logarithmic scale."""

    def __init__(self, low=1.0, high=10.0, rng=None):
        if low <= 0:
            raise ValueError("low must be positive.")
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high
        self.rng = np.random.default_rng() if rng is None else rng

    def sample(self, context=None):
        log_low, log_high = np.log(self.low), np.log(self.high)
        return float(np.exp(self.rng.uniform(log_low, log_high)))

    def pdf(self, x):
        if x < self.low or x > self.high:
            return 0.0
        return 1.0 / (x * np.log(self.high / self.low))

    def cdf(self, x):
        if x < self.low:
            return 0.0
        if x > self.high:
            return 1.0
        return np.log(x / self.low) / np.log(self.high / self.low)

    def mean(self):
        return (self.high - self.low) / np.log(self.high / self.low)

    def median(self):
        return np.sqrt(self.low * self.high)

    def is_absolutely_continuous(self):
        return True
```

## 2. Use it

```python
d = LogUniform(1.0, 100.0, rng=np.random.default_rng(0))

d.mean(), d.median()
```

```
(21.497..., 10.0)
```

```python
draws = np.array(d.batch_sample(size=100_000))
float(draws.mean()), float(draws.min()) >= 1.0, float(draws.max()) <= 100.0
```

```
(21.4..., True, True)
```

## 3. Conventions to follow

- **Validate parameters** in `__init__` and raise `ValueError` for invalid
  input. Consistency here is what lets callers trust a distribution.
- **Accept an optional `rng`** and default it to `np.random.default_rng()`.
- **Implement `sample(self, context=None)`.** The optional `context` is a
  dictionary of simulation state that some distributions use; ignore it if you
  do not need it.
- **Return plain Python floats or ints** from methods where practical. The
  built-in distributions do this so that results compose cleanly.
- **Implement `is_absolutely_continuous`** to report whether the distribution
  has a density (continuous) or a probability mass function (discrete).
- **Use `typing.override`** on methods that override the base class, matching
  the style of the built-in distributions.

```python
from typing import override

class MyDistribution(Distribution):
    @override
    def sample(self, context=None):
        ...
```

Once your class works, it automatically gains the arithmetic operators,
`batch_sample`, and compatibility with `Compose`, `FiniteMixture`, and `Reject`.
