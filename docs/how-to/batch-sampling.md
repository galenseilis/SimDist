# Draw a batch of samples

## Draw many samples at once

Every distribution provides `batch_sample`, which returns a Python list of
independent draws.

```python
import numpy as np
from simdist import dists

dist = dists.Normal(mu=5.0, sigma=2.0, rng=np.random.default_rng(0))
dist.batch_sample(size=5)
```

```
[5.2514604421867865, 4.735790273417396, 6.280845300886564,
 5.20980023430608, 3.9286612536777783]
```

The default `size` is `10_000`.

## Get a NumPy array

`batch_sample` returns a list. Wrap it in `numpy.array` when you want vectorised
post-processing:

```python
draws = np.array(dist.batch_sample(size=10_000))
float(draws.mean()), float(draws.std())
```

```
(5.012, 1.997)
```

## Make samples reproducible

Pass a seeded `numpy.random.Generator` to the constructor. Two distributions
built with the same seed produce the same stream of samples.

```python
a = dists.Normal(0, 1, rng=np.random.default_rng(123)).batch_sample(size=3)
b = dists.Normal(0, 1, rng=np.random.default_rng(123)).batch_sample(size=3)

a == b
```

```
True
```

If you omit `rng`, SimDist creates a fresh generator from the global NumPy
entropy source, so those samples are not reproducible. For simulations you
care about, always pass a seeded generator.

## Pass a sampling context

`sample` and `batch_sample` accept an optional `context` argument that is
forwarded to the distribution. Built-in distributions ignore it, but
`Degenerate` and custom distributions may use it to compute values that depend
on simulation state.

```python
from simdist.core import Degenerate

doubling = Degenerate(lambda context: 2 * context["x"])
doubling.sample({"x": 21})
```

```
42
```
