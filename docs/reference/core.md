# Core API

The `simdist.core` module defines the `Distribution` base class and the
combinators used to build distributions from other distributions.

## `Distribution`

The abstract base class. Subclasses must implement `sample`. All other methods
have defaults that either compute a result or raise `NotImplementedError`.

::: simdist.core.Distribution
    options:
      members:
        - sample
        - batch_sample
        - pdf
        - cdf
        - folded_cdf
        - quantile
        - mean
        - median
        - mode
        - variance
        - standard_deviation
        - skewness
        - excess_kurtosis
        - entropy
        - moment_generating_function
        - characteristic_function
        - expected_shortfall
        - is_infinitely_divisible
        - is_absolutely_continuous
      heading_level: 3

### Operator overloading

`Distribution` implements `+`, `-`, `*`, `/`, and `abs`. These return a
[`Transform`](#transform) whose sampling applies the operation to independent
draws of the operands. `__call__` returns either a `Compose` or a `Transform`,
depending on its argument.

## Combinators

### `Degenerate`

::: simdist.core.Degenerate
    options:
      heading_level: 4

### `Constant`

::: simdist.core.Constant
    options:
      heading_level: 4

### `Transform`

::: simdist.core.Transform
    options:
      heading_level: 4

### `Compose`

::: simdist.core.Compose
    options:
      heading_level: 4

### `Min`, `Max`, and `Range`

::: simdist.core.Min
    options:
      heading_level: 4

::: simdist.core.Max
    options:
      heading_level: 4

::: simdist.core.Range
    options:
      heading_level: 4

### `FiniteMixture`

::: simdist.core.FiniteMixture
    options:
      heading_level: 4
