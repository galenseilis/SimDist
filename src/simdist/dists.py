"""Probability distributions."""

##############
# $0 IMPORTS #
##############

from collections.abc import Iterable
from typing import Any, override

import numpy as np

from .core import Distribution

# TODO: Add section comment to distinguish univariate vs multivariate distributions.
#############################
# $1 DISCRETE DISTRIBUTIONS #
#############################

#######################
# $1.1 FINITE SUPPORT #
#######################

# TODO: Bernoulli
# TODO: Rademacher
# TODO: binomial
# TODO: beta-binomial
# TODO: discrete uniform
# TODO: hypergeometric
# TODO: negative hypergeometric
# TODO: Poisson binomial
# TODO: Fisher's noncentral hypergeometric
# TODO: Wallenius' noncentral hypergeometric
# TODO: Benford's law?
# TODO: ideal soliton
# TODO: robust soliton

#########################
# $1.2 INFINITE SUPPORT #
#########################

##########################################
# $2 ABSOLUTELY CONTINUOUS DISTRIBUTIONS #
##########################################

class ContinuousUniform(Distribution):
    """Continuous uniform distribution."""

    def __init__(
        self, lower: float = 0, upper: float = 1, rng: np.random.Generator | None = None
    ) -> None:
        if not (lower <= upper):
            raise ValueError(f"{lower=} was not lower than {upper=}.")
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self.lower: float = lower
        self.upper: float = upper

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"

    @override
    def sample(self, context: Any | None = None) -> float:
        """Sample from distribution."""
        _ = context
        return self.rng.uniform(self.lower, self.upper)

    @classmethod
    def fit(cls, data: Iterable[int | float]):
        """Fit distribution model."""
        return ContinuousUniform(lower=min(data), upper=max(data))


class Exponential(Distribution):
    """Exponential distribution."""

    def __init__(self, rate: float, rng: np.random.Generator | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"{rate=} was not greater than zero.")
        self.rate: float = rate
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(rate={self.rate})"

    @override
    def sample(self, context: Any | None = None):
        """Sample from distribution."""
        _ = context
        return self.rng.exponential(1 / self.rate)

    @classmethod
    def fit(cls, data):
        """Fit distribution to data."""
        return Exponential(rate=1 / np.mean(data))

    @override
    def pdf(self, x: float) -> float:
        return self.rate * np.exp(-self.rate * x)

    @override
    def cdf(self, x: float) -> float:
        return 1 - np.exp(-self.rate * x)

    @override
    def mean(self) -> float:
        return 1 / self.rate

    @override
    def median(self) -> float:
        return np.log(2) / self.rate

    @override
    def mode(self) -> float:
        return 0

    @override
    def variance(self) -> float:
        return 1 / np.square(self.rate)

    @override
    def standard_deviation(self) -> float:
        return 1 / self.rate

    @override
    def skewness(self) -> float:
        return 2

    @override
    def excess_kurtosis(self) -> float:
        return 6

    @override
    def entropy(self) -> float:
        return 1 - np.log(self.rate)

    @override
    def moment_generating_function(self, t: float):  # pylint: disable=C0103
        if t < self.rate:
            return self.rate / (self.rate - t)
        raise ValueError("The argument t must be less than the rate.")

    @override
    def expected_shortfall(self, p: float) -> float:
        if p < 0:
            raise ValueError(f"{p=} must be non-negative.")
        if p >= 1:
            raise ValueError(f"{p=} must be less than one.")
        return -(np.log(1 - p) + 1) / self.rate


class Gamma(Distribution):
    infinite_divisible: bool = True

    def __init__(
        self, shape: float, scale: float, rng: np.random.Generator | None = None
    ) -> None:
        if shape <= 0:
            raise ValueError(f"{shape=} must be positive.")
        if scale <= 0:
            raise ValueError(f"{scale=} must be positive.")
        self.shape: float = shape
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return self.rng.gamma(self.shape, self.scale)

    def fit(cls, data):
        log_data: float = np.log(data)
        mean_data: float = np.mean(data)
        theta_hat: float = np.mean(data * np.log(data)) - mean_data * np.mean(log_data)
        k_hat: float = mean_data / theta_hat
        return Gamma(shape=k_hat, scale=theta_hat)

    @override
    def mean(self) -> float:
        return self.shape * self.scale

    # TODO: Find better approxmation.
    # https://en.wikipedia.org/wiki/Gamma_distribution#Median_approximations_and_bounds
    @override
    def median(self) -> float:
        return (
            self.shape
            - 1 / 3
            + 8 / (405 * self.shape)
            + 184 / (25515 * self.shape**2)
            + 2248 / (3444525 * self.shape**3)
            - 19006408 / (15345358875 * self.shape**4)
        )

    @override
    def mode(self) -> float:
        return (self.shape - 1) * self.scale if self.shape >= 1 else 0

    @override
    def variance(self) -> float:
        return self.shape * self.scale**2

    @override
    def skewness(self) -> float:
        return 2 / np.sqrt(self.shape)

    @override
    def excess_kurtosis(self) -> float:
        return 6 / self.shape
