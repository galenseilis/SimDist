# SimDist

Probability distributions for simulations in Python.

SimDist is a small, NumPy-only library of probability distributions for
building simulations. Distributions are objects you can sample, query for
moments, and combine with ordinary arithmetic.

```python
import numpy as np
from simdist import dists

service_time = dists.Exponential(rate=2.0, rng=np.random.default_rng(0))
service_time.sample()          # one draw
service_time.mean()            # 0.5
service_time.batch_sample(size=5)
```

## Installation

```bash
uv add simdist
# or
pip install simdist
```

Requires Python 3.12 or newer.

## Documentation

Full documentation is published at **<https://galenseilis.github.io/SimDist/>**,
organised with the [Diátaxis](https://diataxis.fr/) framework:

- [Tutorials](https://galenseilis.github.io/SimDist/tutorials/) — learn by doing.
- [How-to guides](https://galenseilis.github.io/SimDist/how-to/) — recipes for common tasks.
- [Reference](https://galenseilis.github.io/SimDist/reference/) — the complete API.
- [Explanation](https://galenseilis.github.io/SimDist/explanation/) — design and architecture.

## Development

```bash
uv sync                 # install runtime + dev dependencies
uv run pytest           # run the test suite
uv run ruff check src tests
uv sync --group docs    # install the documentation toolchain
uv run mkdocs serve     # preview the docs at http://127.0.0.1:8000
```

## License

Apache-2.0. See [LICENSE](LICENSE).
