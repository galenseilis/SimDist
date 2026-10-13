"""Simulation-compatible probability distributions."""

import numbers
import operator
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self, override

import numpy as np

__all__ = [
    "Distribution",
    "Degenerate",
    "Constant",
    "Transform",
    "Compose",
    "Min",
    "Max",
    "Range",
    "FiniteMixture",
]


class Distribution(ABC):
    """Definition of simulation-compatible distributions."""

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def sample(self, context: dict[Any, Any] | None = None) -> Any:
        """Sample from distribution."""
        ...

    def batch_sample(self, context: dict[Any, Any] | None = None, size: int = 10_000):
        return [self.sample(context) for _ in range(size)]

    def __abs__(self):
        return Transform((self,), operator.abs)

    def __add__(self, other: Self):
        """
        Add two distributions such that sampling is the sum of the samples.
        """
        dist = _dist_cast(other)
        return Transform((self, dist), operator.add)

    def __sub__(self, other: Self):
        """
        Subtract two distributions such that sampling is the difference of the samples.
        """
        dist = _dist_cast(other)
        return Transform((self, dist), operator.sub)

    def __mul__(self, other: Self):
        """
        Multiply two distributions such that sampling is the product of the samples.
        """
        dist = _dist_cast(other)
        return Transform((self, dist), operator.mul)

    def __truediv__(self, other: Self):
        """
        Divide two distributions such that sampling is the ratio of the samples.
        """
        dist = _dist_cast(other)
        return Transform((self, dist), operator.truediv)

    def __call__(self, other: Any):
        """Compose this distribution's class with sampled parameters.

        If ``other`` is a :class:`Distribution`, or an iterable of
        distributions, a :class:`Compose` distribution is returned that
        samples parameters from ``other`` and rebuilds ``type(self)`` from
        them. If ``other`` is a callable, it is instead applied as a
        :class:`Transform`.
        """
        if isinstance(other, Distribution):
            return Compose(type(self), (other,))
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            components = tuple(other)
            if all(isinstance(component, Distribution) for component in components):
                return Compose(type(self), components)
        if callable(other):
            return Transform((self,), other)
        raise TypeError(
            f"Cannot compose {self!r} with {other!r}; expected a Distribution, "
            "an iterable of Distributions, or a callable."
        )

    def pdf(self, x: float) -> float:  # pylint: disable=C0103
        """Probability density function or
        probability mass function."""
        _ = x
        raise NotImplementedError("Method `pdf` not implemented.")

    def cdf(self, x: float) -> float:  # pylint: disable=C0103
        """Cumulative distribution function.

        https://en.wikipedia.org/wiki/Cumulative_distribution_function
        """
        _ = x
        raise NotImplementedError("Method `cdf` not implemented")

    # INFO: https://en.wikipedia.org/wiki/Cumulative_distribution_function#Folded_cumulative_distribution
    def folded_cdf(self, x: float) -> float:
        p = self.cdf(x)
        if p <= 0.5:
            return p
        else:
            return 1 - p

    def quantile(self, p: float) -> float:  # pylint: disable=C0103
        """Quantile function.

        https://en.wikipedia.org/wiki/Quantile
        https://en.wikipedia.org/wiki/Quantile_function
        """
        _ = p
        raise NotImplementedError("Method `quantile` not implemented.")

    def mean(self) -> float:
        """Expected value.

        https://en.wikipedia.org/wiki/Expected_value
        """
        raise NotImplementedError("Method `mean` not implemented")

    def median(self) -> float:
        """Median.

        https://en.wikipedia.org/wiki/Median
        """
        raise NotImplementedError("Method `median` not implemented.")

    def mode(self) -> float:
        """Mode.

        https://en.wikipedia.org/wiki/Mode_(statistics)
        """
        raise NotImplementedError()

    def variance(self) -> float:
        """Variance.

        https://en.wikipedia.org/wiki/Variance
        """
        raise NotImplementedError()

    def standard_deviation(self) -> float:
        """Standard deviation.

        https://en.wikipedia.org/wiki/Standard_deviation
        """
        raise NotImplementedError()

    def mean_absolute_difference(self) -> float:
        """Mean absolute difference (MAD).

        https://en.wikipedia.org/wiki/Mean_absolute_difference
        """
        raise NotImplementedError()

    def skewness(self) -> float:
        """Skewness."""
        raise NotImplementedError()

    def excess_kurtosis(self) -> float:
        """Excess kurtosis."""
        raise NotImplementedError()

    def entropy(self) -> float:
        """Entropy."""
        raise NotImplementedError()

    def moment_generating_function(self, t: float) -> float:  # pylint: disable=C0103
        """Moment generating function (MGF)."""
        _ = t
        raise NotImplementedError()

    def fisher_information(self):
        """Fisher information."""
        raise NotImplementedError()

    def characteristic_function(self, t: float) -> float:
        """Characteristic function."""
        _ = t
        raise NotImplementedError()

    def expected_shortfall(self, p: float) -> float:  # pylint: disable=C0103
        """Expected shortfall."""
        _ = p
        raise NotImplementedError()

    def is_infinitely_divisible(self) -> bool:
        """Determine whether distribution is infinitely-divisible.

        https://en.wikipedia.org/wiki/Infinite_divisibility_(probability)
        """
        raise NotImplementedError()

    def is_absolutely_continuous(self) -> bool:
        """Determine whether distribution is absolutely continuous."""
        raise NotImplementedError()

    def is_absolutely_continous(self) -> bool:
        """Deprecated misspelling of :meth:`is_absolutely_continuous`."""
        warnings.warn(
            "`is_absolutely_continous` is misspelled; use "
            "`is_absolutely_continuous` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_absolutely_continuous()

    def limit_dist(self, var_limits: dict[str, Any]) -> Any:
        """Return limit distribution."""
        _ = var_limits
        raise NotImplementedError()


class Degenerate(Distribution):
    """Degenerate distribution."""

    def __init__(self, func: Callable[[Any], Any]):
        self.func: Callable[[Any], Any] = func

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.func})"

    @override
    def sample(self, context: dict[Any, Any] | None = None) -> float | Distribution:
        """Sample from distribution."""
        return self.func(context)


class Constant(Distribution):
    def __init__(self, value: float | Distribution):
        self.value: float | Distribution = value

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"

    @override
    def sample(self, context: Any | None = None) -> float | Distribution:
        _ = context
        return self.value

    @override
    def mean(self) -> float:
        if isinstance(self.value, numbers.Number):
            return float(self.value)
        return self.value.mean()

    @override
    def variance(self) -> float:
        if isinstance(self.value, numbers.Number):
            return 0.0
        return self.value.variance()

    @override
    def is_infinitely_divisible(self) -> bool:
        return self.value == 0


def _dist_cast(obj: Any) -> Distribution:
    """Cast object to a distribution."""
    if isinstance(obj, numbers.Number):
        return Degenerate(func=lambda context: obj)
    if isinstance(obj, Distribution):
        return obj
    if callable(obj):
        return Degenerate(func=obj)
    if isinstance(obj, str):
        return Degenerate(func=lambda context: obj)

    raise ValueError(f"Could not cast {obj} to type `Distribution`.")


class Transform(Distribution):
    """Combines samples of two or more distributions via an operator.

    This implicitly induces a change of variables.
    """

    def __init__(
        self,
        dists: tuple[Distribution, ...],
        transform: Callable[..., Any],
    ):
        self.dists: Iterable[Distribution] = dists
        self.transform: Callable[..., Any] = transform

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.dists}, {self.transform})"

    @override
    def sample(self, context: Any | None = None) -> float:
        """Sample from distribution."""
        samples: list[Any] = [dist.sample(context) for dist in self.dists]
        return self.transform(*samples)


class Compose(Distribution):
    """Composite distribution."""

    def __init__(self, dist_cls: type[Distribution], dists: Iterable[Distribution]):
        self.dist_cls: type[Distribution] = dist_cls
        self.dists: Iterable[Distribution] = dists

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.dist_cls}, {self.dists})"

    @override
    def sample(self, context: Any | None = None) -> Any:
        """Composite sampling."""
        component_samples: list[Any] = [dist.sample(context) for dist in self.dists]
        sampled_dist = self.dist_cls(*component_samples)
        return sampled_dist.sample(context)


class Min(Distribution):
    """Distribution takes the minimum of samples from multiple distributions."""

    def __init__(self, dists: Iterable[Distribution]):
        self.dists: Iterable[Distribution] = dists

    @override
    def sample(self, context: Any | None = None) -> float:
        samples: list[float] = [dist.sample(context) for dist in self.dists]
        return min(samples)


class Max(Distribution):
    """Distribution takes the maximum of samples from multiple distributions."""

    def __init__(self, dists: Iterable[Distribution]):
        self.dists: Iterable[Distribution] = dists

    @override
    def sample(self, context: Any | None = None) -> float:
        samples: list[float] = [dist.sample(context) for dist in self.dists]
        return max(samples)


class Range(Distribution):
    """Distribution takes the range of samples from multiple distributions."""

    def __init__(self, dists: Iterable[Distribution]):
        self.dists: Iterable[Distribution] = dists

    @override
    def sample(self, context: Any | None = None) -> float:
        samples: list[float] = [dist.sample(context) for dist in self.dists]
        return max(samples) - min(samples)


class FiniteMixture(Distribution):
    """Finite mixture distribution.

    Finite convex combination of probability distributions.

    Sampling draws a component index according to ``weights`` and then
    samples from the selected component.
    """

    def __init__(
        self,
        weights: Sequence[float],
        dists: Sequence[Distribution],
        rng: np.random.Generator | None = None,
    ):
        if len(weights) != len(dists):
            raise ValueError(f"{len(weights)=} must equal {len(dists)=}.")
        if len(dists) == 0:
            raise ValueError("A finite mixture requires at least one component.")
        if any(w < 0 for w in weights):
            raise ValueError(f"Mixture weights must be non-negative, got {weights=}.")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError(
                f"Mixture weights must sum to a positive value, got {weights=}."
            )
        self.weights: list[float] = [w / total for w in weights]
        self.dists: list[Distribution] = list(dists)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.weights}, {self.dists})"

    @override
    def sample(self, context: Any | None = None) -> Any:
        """Sample from the mixture."""
        index = int(self.rng.choice(len(self.dists), p=self.weights))
        return self.dists[index].sample(context)

    @override
    def mean(self) -> float:
        """Expected value, via the law of total expectation."""
        components = []
        for weight, dist in zip(self.weights, self.dists, strict=True):
            components.append(weight * dist.mean())
        return float(sum(components))

    @override
    def variance(self) -> float:
        """Variance, via the law of total variance."""
        mean = self.mean()
        components = []
        for weight, dist in zip(self.weights, self.dists, strict=True):
            components.append(weight * (dist.variance() + (dist.mean() - mean) ** 2))
        return float(sum(components))
