"""Simulation-compatible probability distributions."""

import numbers
import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable, Self, override

__all__ = ["Distribution", "Degenerate", "Transform", "Compose", "Min", "Max", "Range"]


class Distribution(ABC):
    """Definition of simulation-compatible distributions."""

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def sample(self, context: dict[Any, Any] | None = None) -> Any:
        """Sample from distribution."""
        ...

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

    def __call__(self, other: Self):
        """Overloaded call method.

        If `other` is of type `Distribution`, or is an iterable containing
        only elements of type `Distribution`, then it will attempt to
        compose the distributions. Note that this may fail silently until
        a sample is taken from the resulting distribution if the number of
        parameters in the class of `self` does not match the number of distributions
        representing in `other`.

        If the above is not true but other is nonthesless callable,
        then it will attempt to use it as a transform instead.
        """

        return Compose(self, (other,))


    def pdf(self, x: float) -> float:  # pylint: disable=C0103
        """Probability density function or
        probability mass function."""
        _ = x
        raise NotImplementedError("Method `pdf` not implemented.")

    def cdf(self, x: float) -> float:  # pylint: disable=C0103
        """Cumulative distribution function."""
        _ = x
        raise NotImplementedError("Method `cdf` not implemented")

    def quantile(self, p: float) -> float:  # pylint: disable=C0103
        """Quantile function"""
        _ = p
        raise NotImplementedError("Method `quantile` not implemented.")

    def mean(self) -> float:
        """Expected value."""
        raise NotImplementedError("Method `mean` not implemented")

    def median(self) -> float:
        """Median."""
        raise NotImplementedError("Method `median` not implemented.")

    def mode(self) -> float:
        """Mode."""
        raise NotImplementedError()

    def variance(self) -> float:
        """Variance."""
        raise NotImplementedError()

    def standard_deviation(self) -> float:
        """Standard deviation."""
        raise NotImplementedError()

    def mean_absolute_deviation(self) -> float:
        """Mean absolute deviation (MAD)."""
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

    def characteristic_function(self, t: float) -> float:
        """Characteristic function."""
        _ = t
        raise NotImplementedError()

    def expected_shortfall(self, p: float) -> float:  # pylint: disable=C0103
        """Expected shortfall."""
        _ = p
        raise NotImplementedError()

    def is_infinitely_divisible(self) -> bool:
        """Determine whether distribution is infinitely-divisible."""
        raise NotImplementedError()

    def limit_dist(self, var_limits: dict[str, Any]):
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
        self.value: float | Distribution  = value

    @override
    def sample(self, context: Any | None = None) -> float | Distribution:
        _ = context
        return self.value

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
    """A distribution that combines the samples of two or more other distributions via an operator.

    This implicitly induces a change of variables.
    """

    def __init__(
        self,
        dists: tuple[Distribution],
        transform: Callable[[tuple[Distribution]], Any],
    ):
        self.dists: Iterable[Distribution] = dists
        self.transform: Callable[[tuple[Distribution]], Any] = transform

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

    def __init__(self, dist_cls: Distribution, dists: Iterable[Distribution]):
        self.dist_cls: Distribution = dist_cls
        self.dists: Iterable[Distribution] = dists

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
    """

    # TODO: Implement
