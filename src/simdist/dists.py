"""Probability distributions."""

##############
# $0 IMPORTS #
##############

from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, Final

import numpy as np

from .core import Distribution

####################
# $1 CONFIGURATION #
####################

__all__: Final[list[str]] = [
    # Discrete, finite support.
    "Bernoulli",
    "Rademacher",
    "BetaBinomial",
    "Binomial",
    "Empirical",
    "DiscreteUniform",
    "Hypergeometric",
    "NegativeHypergeometric",
    "PoissonBinomial",
    "FisherNoncentralHypergeometric",
    "WalleniusNoncentralHypergeometric",
    "Benford",
    # Discrete, infinite support.
    "Geometric",
    "NegativeBinomial",
    "Poisson",
    "IdealSoliton",
    "RobustSoliton",
    # Absolutely continuous.
    "ContinuousUniform",
    "Exponential",
    "Gamma",
    "VonMises",
]

###############################
# $2 UNIVARIATE DISTRIBUTIONS #
###############################

###############################
# $2.1 DISCRETE DISTRIBUTIONS #
###############################

#########################
# $2.1.0 FINITE SUPPORT #
#########################


def _check_probability(p: float, name: str = "p") -> None:
    """Raise if ``p`` is not in the closed unit interval."""
    if not 0 <= p <= 1:
        raise ValueError(f"Probability parameter {name}={p} must be in [0, 1].")


class Bernoulli(Distribution):
    """Bernoulli distribution."""

    support: list[int] = [0, 1]  # INFO: Exhaustive list; not interval.

    def __init__(self, p: float, rng: np.random.Generator | None = None):
        _check_probability(p)
        self.p: float = p
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    @override
    def sample(self, context: Any | None = None):
        _ = context
        return int(self.rng.choice(a=self.support, p=[1 - self.p, self.p]))

    @override
    def mean(self) -> float:
        return self.p

    @override
    def variance(self) -> float:
        return self.p * (1 - self.p)

    @override
    def is_infinitely_divisible(self) -> bool:
        return self.p in (0.0, 1.0)

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


class Rademacher(Distribution):
    """Rademacher distribution: ``+1`` or ``-1``."""

    support: list[int] = [-1, 1]  # INFO: Exhaustive list; not interval.

    def __init__(self, p: float, rng: np.random.Generator | None = None):
        _check_probability(p)
        self.p: float = p
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None):
        _ = context
        return int(self.rng.choice(a=self.support, p=[1 - self.p, self.p]))

    @override
    def mean(self) -> float:
        return 2 * self.p - 1

    @override
    def variance(self) -> float:
        return 4 * self.p * (1 - self.p)


class BetaBinomial(Distribution):
    """Beta-binomial distribution."""

    def __init__(
        self,
        n: int,
        alpha: float,
        beta: float,
        rng: np.random.Generator | None = None,
    ):
        if n != int(n):
            raise ValueError(f"Parameter n={n} must be an integer")
        if n < 0:
            raise ValueError(f"Parameter n={n} must be non-negative.")
        if alpha <= 0:
            raise ValueError(f"Parameter alpha={alpha} must be positive.")
        if beta <= 0:
            raise ValueError(f"Parameter beta={beta} must be positive.")
        self.n: int = int(n)
        self.alpha: float = alpha
        self.beta: float = beta
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None):
        _ = context
        p = self.rng.beta(self.alpha, self.beta)
        return int(self.rng.binomial(self.n, p))

    @override
    def mean(self) -> float:
        return self.n * self.alpha / (self.alpha + self.beta)

    @override
    def variance(self) -> float:
        alpha, beta, n = self.alpha, self.beta, self.n
        return (
            n
            * alpha
            * beta
            * (alpha + beta + n)
            / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )


class Binomial(Distribution):
    """Binomial distribution."""

    def __init__(
        self,
        num: int,
        prob: float,
        rng: np.random.Generator | None = None,
    ):
        if num != int(num):
            raise ValueError(f"Parameter num={num} must be an integer")
        if num < 0:
            raise ValueError(f"Parameter num={num} must be non-negative.")
        _check_probability(prob, name="prob")
        self.num: int = int(num)
        self.prob: float = prob
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return int(self.rng.binomial(self.num, self.prob))

    @override
    def mean(self) -> float:
        return self.num * self.prob

    @override
    def variance(self) -> float:
        return self.num * self.prob * (1 - self.prob)

    @override
    def is_infinitely_divisible(self) -> bool:
        return False


class Empirical(Distribution):
    """Empirical distribution over a finite, possibly weighted, sample."""

    def __init__(
        self,
        support: Any | None = None,
        weights: Any | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self.support: Any = support
        self.weights: Any = weights

    @override
    def sample(self, context: Any | None = None):
        _ = context
        if self.support is None:
            raise ValueError(
                "Cannot sample from an empirical distribution with no support."
            )
        return self.rng.choice(a=self.support, p=self.weights)

    @override
    def mean(self) -> float:
        if self.support is None:
            raise ValueError("Cannot compute the mean without a support.")
        if self.weights is None:
            return float(np.mean(self.support))
        return float(np.average(self.support, weights=self.weights))


class DiscreteUniform(Distribution):
    """Discrete uniform distribution over ``[lower, upper]`` (inclusive).

    .. note::
        ``numpy.random.Generator.integers`` draws from a half-open interval, so
        the ``upper`` endpoint is handled explicitly here.
    """

    def __init__(
        self,
        lower: int = 0,
        upper: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if lower != int(lower) or upper != int(upper):
            raise ValueError("Both `lower` and `upper` must be integers.")
        if upper < lower:
            raise ValueError(
                f"Parameter upper={upper} must not be below lower={lower}."
            )
        self.lower: int = int(lower)
        self.upper: int = int(upper)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.integers(self.lower, self.upper + 1))

    @override
    def mean(self) -> float:
        return (self.lower + self.upper) / 2

    @override
    def variance(self) -> float:
        n = self.upper - self.lower + 1
        return (n**2 - 1) / 12

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


class Hypergeometric(Distribution):
    """Hypergeometric distribution.

    Models the number of successes in ``nsample`` draws *without*
    replacement from a population of ``ngood`` successes and ``nbad``
    failures.
    """

    def __init__(
        self,
        ngood: int,
        nbad: int,
        nsample: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (("ngood", ngood), ("nbad", nbad), ("nsample", nsample)):
            if value != int(value):
                raise ValueError(f"Parameter {name}={value} must be an integer.")
            if value < 0:
                raise ValueError(f"Parameter {name}={value} must be non-negative.")
        if nsample > ngood + nbad:
            raise ValueError(
                f"nsample={nsample} cannot exceed ngood + nbad={ngood + nbad}."
            )
        self.ngood: int = int(ngood)
        self.nbad: int = int(nbad)
        self.nsample: int = int(nsample)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.hypergeometric(self.ngood, self.nbad, self.nsample))

    @override
    def mean(self) -> float:
        total = self.ngood + self.nbad
        return self.nsample * self.ngood / total

    @override
    def variance(self) -> float:
        total = self.ngood + self.nbad
        p = self.ngood / total
        if total <= 1:
            return 0.0
        return self.nsample * p * (1 - p) * (total - self.nsample) / (total - 1)

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


class NegativeHypergeometric(Distribution):
    """Negative hypergeometric distribution.

    Draw without replacement from a population of ``ngood`` successes and
    ``nbad`` failures until ``nfailures`` failures have been observed, and
    count the number of successes drawn.
    """

    def __init__(
        self,
        ngood: int,
        nbad: int,
        nfailures: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (
            ("ngood", ngood),
            ("nbad", nbad),
            ("nfailures", nfailures),
        ):
            if value != int(value):
                raise ValueError(f"Parameter {name}={value} must be an integer.")
            if value < 0:
                raise ValueError(f"Parameter {name}={value} must be non-negative.")
        if nfailures == 0:
            raise ValueError("nfailures must be positive.")
        if nfailures > nbad:
            raise ValueError(
                f"nfailures={nfailures} cannot exceed the number of "
                f"failures, nbad={nbad}."
            )
        self.ngood: int = int(ngood)
        self.nbad: int = int(nbad)
        self.nfailures: int = int(nfailures)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        good = self.ngood
        bad = self.nbad
        failures = 0
        successes = 0
        while failures < self.nfailures:
            if self.rng.random() < good / (good + bad):
                good -= 1
                successes += 1
            else:
                bad -= 1
                failures += 1
        return successes

    @override
    def mean(self) -> float:
        return self.nfailures * self.ngood / (self.nbad + 1)

    @override
    def variance(self) -> float:
        ngood, nbad, r = self.ngood, self.nbad, self.nfailures
        denominator = (nbad + 1) ** 2 * (nbad + 2)
        return (
            r * (ngood + nbad + 1) * (nbad - r + 1) * ngood / denominator
            if nbad - r + 1 >= 0
            else 0.0
        )

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


class PoissonBinomial(Distribution):
    """Poisson binomial distribution.

    The number of successes among ``len(probs)`` independent Bernoulli trials
    with (possibly) distinct success probabilities.
    """

    def __init__(
        self,
        probs: Sequence[float],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.probs: list[float] = [float(p) for p in probs]
        for p in self.probs:
            _check_probability(p, name="p")
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        draws = self.rng.random(len(self.probs)) < self.probs
        return int(np.sum(draws))

    @override
    def mean(self) -> float:
        return float(sum(self.probs))

    @override
    def variance(self) -> float:
        return float(sum(p * (1 - p) for p in self.probs))

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


class FisherNoncentralHypergeometric(Distribution):
    """Fisher's noncentral hypergeometric distribution.

    Draw ``nsample`` balls *without* replacement from an urn with ``nwhite``
    white and ``nblack`` black balls, where white balls are ``odds`` times as
    likely to be drawn as black balls. Sampling is performed by inverting the
    (normalized) probability mass function.
    """

    def __init__(
        self,
        nwhite: int,
        nblack: int,
        nsample: int,
        odds: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (
            ("nwhite", nwhite),
            ("nblack", nblack),
            ("nsample", nsample),
        ):
            if value != int(value):
                raise ValueError(f"Parameter {name}={value} must be an integer.")
            if value < 0:
                raise ValueError(f"Parameter {name}={value} must be non-negative.")
        if nsample > nwhite + nblack:
            raise ValueError(
                f"nsample={nsample} cannot exceed nwhite + nblack={nwhite + nblack}."
            )
        if odds <= 0:
            raise ValueError(f"Parameter odds={odds} must be positive.")
        self.nwhite: int = int(nwhite)
        self.nblack: int = int(nblack)
        self.nsample: int = int(nsample)
        self.odds: float = odds
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self._support, self._probs = self._build_pmf()

    def _build_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        lower = max(0, self.nsample - self.nblack)
        upper = min(self.nsample, self.nwhite)
        xs = np.arange(lower, upper + 1)
        log_odds = math.log(self.odds)
        logs = np.empty(xs.shape, dtype=float)
        for i, x in enumerate(xs):
            x = int(x)
            logs[i] = (
                math.log(math.comb(self.nwhite, x))
                + math.log(math.comb(self.nblack, self.nsample - x))
                + x * log_odds
            )
        logs -= logs.max()
        probs = np.exp(logs)
        probs /= probs.sum()
        return xs, probs

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.choice(self._support, p=self._probs))

    @override
    def mean(self) -> float:
        return float(np.sum(self._support * self._probs))


class WalleniusNoncentralHypergeometric(Distribution):
    """Wallenius' noncentral hypergeometric distribution.

    Simulated by drawing ``nsample`` balls *without* replacement from an urn
    with ``nwhite`` white and ``nblack`` black balls, where at each draw the
    probability of selecting a white ball is proportional to ``odds`` times the
    number of white balls remaining.
    """

    def __init__(
        self,
        nwhite: int,
        nblack: int,
        nsample: int,
        odds: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (
            ("nwhite", nwhite),
            ("nblack", nblack),
            ("nsample", nsample),
        ):
            if value != int(value):
                raise ValueError(f"Parameter {name}={value} must be an integer.")
            if value < 0:
                raise ValueError(f"Parameter {name}={value} must be non-negative.")
        if nsample > nwhite + nblack:
            raise ValueError(
                f"nsample={nsample} cannot exceed nwhite + nblack={nwhite + nblack}."
            )
        if odds <= 0:
            raise ValueError(f"Parameter odds={odds} must be positive.")
        self.nwhite: int = int(nwhite)
        self.nblack: int = int(nblack)
        self.nsample: int = int(nsample)
        self.odds: float = odds
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        white = self.nwhite
        black = self.nblack
        drawn_white = 0
        for _ in range(self.nsample):
            white_weight = self.odds * white
            total = white_weight + black
            if self.rng.random() * total < white_weight:
                white -= 1
                drawn_white += 1
            else:
                black -= 1
        return drawn_white


class Benford(Distribution):
    """Benford's law for the leading significant digit, in a given ``base``."""

    def __init__(
        self,
        base: int = 10,
        rng: np.random.Generator | None = None,
    ) -> None:
        if base != int(base):
            raise ValueError(f"Parameter base={base} must be an integer.")
        if base < 2:
            raise ValueError(f"Parameter base={base} must be at least 2.")
        self.base: int = int(base)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        digits = np.arange(1, self.base)
        probs = np.log(1 + 1 / digits) / np.log(self.base)
        self._digits = digits
        self._probs = probs / probs.sum()

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.choice(self._digits, p=self._probs))

    @override
    def mean(self) -> float:
        return float(np.sum(self._digits * self._probs))

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


###########################
# $2.1.1 INFINITE SUPPORT #
###########################


class Geometric(Distribution):
    """Geometric distribution: number of trials until the first success."""

    infinite_divisible: bool = True

    def __init__(self, prob: float, rng: np.random.Generator | None = None):
        if not 0 < prob <= 1:
            raise ValueError(f"Probability parameter prob={prob} must be in (0, 1].")
        self.prob: float = prob
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return int(self.rng.geometric(self.prob))

    @override
    def mean(self) -> float:
        return 1 / self.prob

    @override
    def variance(self) -> float:
        return (1 - self.prob) / self.prob**2

    @override
    def is_infinitely_divisible(self) -> bool:
        return True


class NegativeBinomial(Distribution):
    """Negative binomial distribution."""

    def __init__(
        self, rate: float, prob: float, rng: np.random.Generator | None = None
    ):
        if rate <= 0:
            raise ValueError(f"Parameter rate={rate} must be positive.")
        if not 0 < prob <= 1:
            raise ValueError(f"Probability parameter prob={prob} must be in (0, 1].")
        self.rate: float = rate
        self.prob: float = prob
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        """Sample from this distribution."""
        _ = context
        return int(self.rng.negative_binomial(self.rate, self.prob))

    @override
    def mean(self) -> float:
        return self.rate * (1 - self.prob) / self.prob

    @override
    def variance(self) -> float:
        return self.rate * (1 - self.prob) / self.prob**2

    @override
    def is_infinitely_divisible(self) -> bool:
        return True


class Poisson(Distribution):
    """Poisson distribution."""

    def __init__(self, rate: float, rng: np.random.Generator | None = None):
        if rate < 0:
            raise ValueError(f"Parameter rate={rate} must be non-negative.")
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self.rate: float = rate

    @override
    def __add__(self, other: Distribution):
        if isinstance(other, Poisson):
            return Poisson(self.rate + other.rate)
        else:
            return super().__add__(other)

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return int(self.rng.poisson(self.rate))

    @override
    def mean(self) -> float:
        return self.rate

    @override
    def variance(self) -> float:
        return self.rate

    @override
    def is_infinitely_divisible(self) -> bool:
        return True


class IdealSoliton(Distribution):
    """Ideal soliton distribution over degrees ``1..K``.

    Used by Luby transform (fountain) codes.
    """

    def __init__(self, k: int, rng: np.random.Generator | None = None) -> None:
        if k != int(k):
            raise ValueError(f"Parameter k={k} must be an integer.")
        if k < 1:
            raise ValueError(f"Parameter k={k} must be positive.")
        self.k: int = int(k)
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        degrees = np.arange(1, self.k + 1)
        probs = np.empty(self.k, dtype=float)
        probs[0] = 1 / self.k
        if self.k > 1:
            probs[1:] = 1 / (degrees[1:] * (degrees[1:] - 1))
        self._degrees = degrees
        self._probs = probs / probs.sum()

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.choice(self._degrees, p=self._probs))


class RobustSoliton(Distribution):
    """Robust soliton distribution over degrees ``1..K``.

    Adds a spike to the ideal soliton distribution; see Luby (2002). The
    parameters ``c`` and ``delta`` control the failure probability of the
    associated fountain code.
    """

    def __init__(
        self,
        k: int,
        c: float = 0.1,
        delta: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        if k != int(k):
            raise ValueError(f"Parameter k={k} must be an integer.")
        if k < 1:
            raise ValueError(f"Parameter k={k} must be positive.")
        if c <= 0:
            raise ValueError(f"Parameter c={c} must be positive.")
        if not 0 < delta < 1:
            raise ValueError(f"Parameter delta={delta} must be in (0, 1).")
        self.k: int = int(k)
        self.c: float = c
        self.delta: float = delta
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

        r = c * math.log(k / delta) * math.sqrt(k)
        spike = max(1, min(self.k, int(k / r)))
        degrees = np.arange(1, self.k + 1)
        rho = np.empty(self.k, dtype=float)
        rho[0] = 1 / self.k
        if self.k > 1:
            rho[1:] = 1 / (degrees[1:] * (degrees[1:] - 1))
        tau = np.zeros(self.k, dtype=float)
        for d in range(1, self.k + 1):
            if d < spike:
                tau[d - 1] = r / (d * self.k)
            elif d == spike:
                tau[d - 1] = r * math.log(r / delta) / self.k
        probs = rho + tau
        self._degrees = degrees
        self._probs = probs / probs.sum()

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.choice(self._degrees, p=self._probs))


##########################################
# $4 ABSOLUTELY CONTINUOUS DISTRIBUTIONS #
##########################################


class ContinuousUniform(Distribution):
    """Continuous uniform distribution."""

    def __init__(
        self, lower: float = 0, upper: float = 1, rng: np.random.Generator | None = None
    ) -> None:
        if not (lower <= upper):
            raise ValueError(f"lower={lower} was not lower than upper={upper}.")
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng
        self.lower: float = lower
        self.upper: float = upper

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(lower={self.lower}, upper={self.upper})"

    @override
    def sample(self, context: Any | None = None) -> float:
        """Sample from this distribution."""
        _ = context
        return float(self.rng.uniform(self.lower, self.upper))

    @classmethod
    def fit(cls, data: Iterable[int | float]):
        """Fit distribution model to data."""
        return ContinuousUniform(lower=min(data), upper=max(data))

    @override
    def mean(self) -> float:
        return (self.lower + self.upper) / 2

    @override
    def variance(self) -> float:
        return (self.upper - self.lower) ** 2 / 12

    @override
    def is_infinitely_divisible(self) -> bool:
        return False

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Exponential(Distribution):
    """Exponential distribution."""

    def __init__(self, rate: float, rng: np.random.Generator | None = None) -> None:
        if rate <= 0:
            raise ValueError(f"rate={rate} was not greater than zero.")
        self.rate: float = rate
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(rate={self.rate})"

    @override
    def sample(self, context: Any | None = None):
        """Sample from distribution."""
        _ = context
        return float(self.rng.exponential(1 / self.rate))

    @classmethod
    def fit(cls, data):
        """Fit distribution to some data.

        Returns an instance of the exponential distribution
        with estimated parameters.
        """
        return Exponential(rate=1 / np.mean(data))

    @override
    def pdf(self, x: float) -> float:
        """Calculate the PDF of this exponential distribution."""
        if x < 0:
            return 0.0
        return float(self.rate * np.exp(-self.rate * x))

    @override
    def cdf(self, x: float) -> float:
        """Calculate the CDF of this exponential distribution."""
        if x < 0:
            return 0.0
        return float(1 - np.exp(-self.rate * x))

    @override
    def mean(self) -> float:
        """Calculate the mean of this exponential distribution."""
        return 1 / self.rate

    @override
    def median(self) -> float:
        """Calculate the median of this exponential distribution."""
        return float(np.log(2) / self.rate)

    @override
    def mode(self) -> float:
        """Calculate the mode of this exponential distribution."""
        return 0

    @override
    def variance(self) -> float:
        """Calculate the variance of this exponential distribution."""
        return 1 / np.square(self.rate)

    @override
    def standard_deviation(self) -> float:
        """Calculate the standard deviation of this exponential distribution."""
        return 1 / self.rate

    @override
    def skewness(self) -> float:
        """Calculate the skewness of this exponential distribution."""
        return 2

    @override
    def excess_kurtosis(self) -> float:
        """Calculate the excess kurtosis of this exponential distribution."""
        return 6

    @override
    def entropy(self) -> float:
        """Calculate the differential entropy of this exponential distribution."""
        return float(1 - np.log(self.rate))

    @override
    def moment_generating_function(self, t: float):  # pylint: disable=C0103
        """Calculate the moment-generating function of this exponential distribution."""
        if t < self.rate:
            return self.rate / (self.rate - t)
        raise ValueError("The argument t must be less than the rate.")

    @override
    def expected_shortfall(self, p: float) -> float:
        """Calculate the expected shortfall of this exponential distribution."""
        if p < 0:
            raise ValueError(f"p={p} must be non-negative.")
        if p >= 1:
            raise ValueError(f"p={p} must be less than one.")
        return float(-(np.log(1 - p) + 1) / self.rate)

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


def _regularized_lower_incomplete_gamma(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function ``P(a, x)``.

    Uses the series representation for ``x < a + 1`` and the continued
    fraction representation of ``Q(a, x) = 1 - P(a, x)`` otherwise (see
    Numerical Recipes, section 6.2).
    """
    if x <= 0:
        return 0.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_prefactor)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(log_prefactor) * h


def _gamma_median(shape: float, scale: float) -> float:
    """Numerically accurate median of a Gamma(``shape``, ``scale``)."""
    lo = 0.0
    hi = (shape + 10) * scale + 10 * math.sqrt(shape) * scale
    while _regularized_lower_incomplete_gamma(shape, hi / scale) < 0.5:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if _regularized_lower_incomplete_gamma(shape, mid / scale) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class Gamma(Distribution):
    """The gamma distribution."""

    infinite_divisible: bool = True

    def __init__(
        self, shape: float, scale: float, rng: np.random.Generator | None = None
    ) -> None:
        if shape <= 0:
            raise ValueError(f"shape={shape} must be positive.")
        if scale <= 0:
            raise ValueError(f"scale={scale} must be positive.")
        self.shape: float = shape
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        """Sample from this gamma distribution."""
        _ = context
        return float(self.rng.gamma(self.shape, self.scale))

    @classmethod
    def fit(cls, data):
        """Fit this distribution to data to create a new instance.

        Uses the approximation
        ``theta_hat = E[X log X] - E[X] E[log X]`` and ``k_hat = E[X] / theta_hat``.
        """
        log_data: float = np.log(data)
        mean_data: float = np.mean(data)
        theta_hat: float = np.mean(data * np.log(data)) - mean_data * np.mean(log_data)
        k_hat: float = mean_data / theta_hat
        return Gamma(shape=k_hat, scale=theta_hat)

    @override
    def mean(self) -> float:
        """Calculate the mean of this gamma distribution."""
        return self.shape * self.scale

    @override
    def median(self) -> float:
        """Calculate the median of this Gamma distribution."""
        return _gamma_median(self.shape, self.scale)

    @override
    def mode(self) -> float:
        """Calculate the mode of this gamma distribution."""
        return (self.shape - 1) * self.scale if self.shape >= 1 else 0

    @override
    def variance(self) -> float:
        """Calculate variance of this gamma distribution."""
        return self.shape * self.scale**2

    @override
    def skewness(self) -> float:
        """Calculate the skewness of this gamma distribution."""
        return float(2 / np.sqrt(self.shape))

    @override
    def excess_kurtosis(self) -> float:
        """Calculate excess kurtosis of this gamma distribution."""
        return float(6 / self.shape)

    @override
    def is_infinitely_divisible(self) -> bool:
        return True

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class VonMises(Distribution):
    """Von Mises distribution on the circle."""

    def __init__(
        self,
        mu: float,
        kappa: float,
        rng: np.random.Generator | None = None,
    ):
        if kappa < 0:
            raise ValueError(f"Parameter kappa={kappa} must be nonnegative.")
        self.mu: float = mu
        self.kappa: float = kappa
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.vonmises(self.mu, self.kappa))

    @override
    def mean(self) -> float:
        return self.mu

    @override
    def median(self) -> float:
        return self.mu

    @override
    def mode(self) -> float:
        return self.mu

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


##############################
# MULTIVARIATE DISTRIBUTIONS #
##############################
