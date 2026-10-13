# Truncate a distribution

Rejection sampling shapes a distribution by repeatedly drawing candidates until
one satisfies a predicate. SimDist exposes this through `Reject`, which wraps
any existing distribution.

```python
from functools import partial

import numpy as np
from simdist import dists
from simdist.rejectors import Reject, is_negative, outside_interval
```

A `Reject` takes a source distribution and a `reject(candidate, context)`
predicate that returns `True` when a candidate should be *discarded*.

## Keep only non-negative values

`is_negative` is a ready-made predicate that rejects negative candidates. This
turns a distribution on the whole real line into one supported on
`[0, +∞)`.

```python
from simdist.rejectors import Reject, is_negative

positive_normal = Reject(
    dists.Normal(mu=0.0, sigma=1.0, rng=np.random.default_rng(0)),
    is_negative,
)
draws = np.array(positive_normal.batch_sample(size=20_000))
```

```python
float(draws.min()), float(draws.mean()), float(draws.std())
```

```
(0.0001, 0.804, 0.602)
```

The mean and standard deviation match the known half-normal values
`0.798` and `0.603`.

## Restrict to an interval

`outside_interval` rejects candidates that fall outside `[lower, upper]`. Its
signature matches the `reject(candidate, context)` convention, so bind the
bounds with `functools.partial`.

```python
from functools import partial
from simdist.rejectors import Reject, outside_interval

band = Reject(
    dists.Normal(0.0, 1.0, rng=np.random.default_rng(0)),
    partial(outside_interval, lower=-1.0, upper=1.0),
)

xs = band.batch_sample(size=20_000)
min(xs), max(xs)
```

```
(-1.0, 1.0)
```

## Write a custom rejection rule

A rejection rule is any callable with the signature `reject(candidate, context)`
that returns `True` to reject. For example, to sample uniformly on
`[-1, 1]` while excluding a band around zero:

```python
from simdist import dists
from simdist.rejectors import Reject

def near_zero(candidate, context=None):
    return abs(candidate) < 0.25

gapped = Reject(
    dists.ContinuousUniform(-1.0, 1.0, rng=np.random.default_rng(0)),
    reject=near_zero,
)
```

```python
xs = gapped.batch_sample(size=20_000)
min(xs), max(xs)
```

```
(-1.0, 1.0)
```

!!! warning
    Rejection sampling loops until a candidate is accepted. If the acceptance
    region has (near) zero probability under the proposal, sampling will be
    very slow or will never terminate. Choose a proposal whose support covers
    the region you want.
