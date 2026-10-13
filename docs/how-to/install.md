# Install SimDist

SimDist requires **Python 3.12 or newer** and depends only on NumPy.

## With uv (recommended)

Add SimDist to a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add simdist
```

## With pip

```bash
pip install simdist
```

## From source

Clone the repository and sync the development environment:

```bash
git clone https://github.com/galenseilis/SimDist.git
cd SimDist
uv sync
```

`uv sync` creates a virtual environment, installs the runtime and development
dependencies, and installs SimDist itself in editable mode.

## Verifying the installation

```python
from simdist import dists

print(dists.Normal().mean())
```

```
0.0
```

## Importing from submodules

`simdist` is organised into submodules and `simdist/__init__.py` deliberately
does **not** re-export any names. Import from the submodule you need:

```python
from simdist import dists          # concrete distributions
from simdist import core           # base class and combinators
from simdist import rejectors      # rejection sampling helpers
```

or import individual names directly:

```python
from simdist.dists import Normal, Exponential
from simdist.core import FiniteMixture
```
