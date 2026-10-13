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
    "Categorical",
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
    "Zipf",
    "IdealSoliton",
    "RobustSoliton",
    # Absolutely continuous.
    "ContinuousUniform",
    "Triangular",
    "Normal",
    "LogNormal",
    "Exponential",
    "Laplace",
    "Logistic",
    "Cauchy",
    "Gamma",
    "Beta",
    "ChiSquared",
    "StudentT",
    "Weibull",
    "Rayleigh",
    "Pareto",
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


class Categorical(Distribution):
    """Categorical distribution over an explicit, finite support.

    A generalisation of :class:`Bernoulli`/:class:`Rademacher` to more than
    two outcomes. Probabilities are normalised to sum to one.
    """

    def __init__(
        self,
        probabilities: Sequence[float],
        support: Sequence[Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        probs = [float(p) for p in probabilities]
        if len(probs) == 0:
            raise ValueError("A categorical distribution needs at least one category.")
        if any(p < 0 for p in probs):
            raise ValueError(
                f"Probabilities must be non-negative, got {probabilities=}."
            )
        total = float(sum(probs))
        if total <= 0:
            raise ValueError("Probabilities must sum to a positive value.")
        self.probabilities: list[float] = [p / total for p in probs]
        if support is None:
            self.support: list[Any] = list(range(len(probs)))
        else:
            self.support = list(support)
            if len(self.support) != len(self.probabilities):
                raise ValueError(
                    f"len(support)={len(self.support)} must equal "
                    f"len(probabilities)={len(self.probabilities)}."
                )
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self.probabilities}, {self.support})"

    @override
    def sample(self, context: Any | None = None) -> Any:
        _ = context
        index = int(self.rng.choice(len(self.support), p=self.probabilities))
        return self.support[index]

    @override
    def mean(self) -> float:
        return float(np.average(self.support, weights=self.probabilities))

    @override
    def variance(self) -> float:
        mean = self.mean()
        return float(
            sum(
                p * (float(s) - mean) ** 2
                for p, s in zip(self.probabilities, self.support, strict=True)
            )
        )

    @override
    def mode(self) -> Any:
        return self.support[int(np.argmax(self.probabilities))]

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


##########################################
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


class Zipf(Distribution):
    """Zipf (zeta) distribution on the positive integers.

    ``P(k)`` is proportional to ``k ** -alpha`` for ``k = 1, 2, ...``.
    """

    def __init__(self, alpha: float, rng: np.random.Generator | None = None) -> None:
        if alpha <= 1:
            raise ValueError(f"Parameter alpha={alpha} must be greater than 1.")
        self.alpha: float = alpha
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> int:
        _ = context
        return int(self.rng.zipf(self.alpha))

    @override
    def mode(self) -> int:
        return 1

    @override
    def is_absolutely_continuous(self) -> bool:
        return False


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


class Triangular(Distribution):
    """Triangular distribution on ``[low, high]`` with peak at ``mode``."""

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        mode: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if high <= low:
            raise ValueError(f"high={high} must be greater than low={low}.")
        if mode is None:
            mode = (low + high) / 2
        if not low <= mode <= high:
            raise ValueError(f"mode={mode} must lie within [{low}, {high}].")
        self.low: float = low
        self.high: float = high
        self._mode: float = mode
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(low={self.low}, high={self.high}, "
            f"mode={self._mode})"
        )

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.triangular(self.low, self._mode, self.high))

    @override
    def pdf(self, x: float) -> float:
        a, b, c = self.low, self.high, self._mode
        if x < a or x > b:
            return 0.0
        if c == a:
            return 2 * (b - x) / (b - a) ** 2
        if c == b:
            return 2 * (x - a) / (b - a) ** 2
        if x < c:
            return 2 * (x - a) / ((b - a) * (c - a))
        if x > c:
            return 2 * (b - x) / ((b - a) * (b - c))
        return 2 / (b - a)

    @override
    def cdf(self, x: float) -> float:
        a, b, c = self.low, self.high, self._mode
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        if c == a:
            return 1 - (b - x) ** 2 / (b - a) ** 2
        if c == b:
            return (x - a) ** 2 / (b - a) ** 2
        if x <= c:
            return (x - a) ** 2 / ((b - a) * (c - a))
        return 1 - (b - x) ** 2 / ((b - a) * (b - c))

    @override
    def mean(self) -> float:
        return (self.low + self.high + self._mode) / 3

    @override
    def variance(self) -> float:
        a, b, c = self.low, self.high, self._mode
        return (a**2 + b**2 + c**2 - a * b - a * c - b * c) / 18

    @override
    def median(self) -> float:
        a, b, c = self.low, self.high, self._mode
        if c >= (a + b) / 2:
            return a + math.sqrt((b - a) * (c - a) / 2)
        return b - math.sqrt((b - a) * (b - c) / 2)

    @override
    def mode(self) -> float:
        return self._mode

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Normal(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma={sigma} must be positive.")
        self.mu: float = mu
        self.sigma: float = sigma
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}(mu={self.mu}, sigma={self.sigma})"

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.normal(self.mu, self.sigma))

    @override
    def pdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2 * math.pi))

    @override
    def cdf(self, x: float) -> float:
        z = (x - self.mu) / (self.sigma * math.sqrt(2))
        return 0.5 * (1 + math.erf(z))

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
    def variance(self) -> float:
        return self.sigma**2

    @override
    def standard_deviation(self) -> float:
        return self.sigma

    @override
    def skewness(self) -> float:
        return 0.0

    @override
    def excess_kurtosis(self) -> float:
        return 0.0

    @override
    def entropy(self) -> float:
        return 0.5 * math.log(2 * math.pi * math.e * self.sigma**2)

    @override
    def is_infinitely_divisible(self) -> bool:
        return True

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class LogNormal(Distribution):
    """Log-normal distribution: ``X = exp(N(mu, sigma ** 2))``."""

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma={sigma} must be positive.")
        self.mu: float = mu
        self.sigma: float = sigma
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.lognormal(self.mu, self.sigma))

    @override
    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        log_x = math.log(x)
        z = (log_x - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (x * self.sigma * math.sqrt(2 * math.pi))

    @override
    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        z = (math.log(x) - self.mu) / (self.sigma * math.sqrt(2))
        return 0.5 * math.erfc(-z)

    @override
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    @override
    def variance(self) -> float:
        return (math.exp(self.sigma**2) - 1) * math.exp(2 * self.mu + self.sigma**2)

    @override
    def median(self) -> float:
        return math.exp(self.mu)

    @override
    def mode(self) -> float:
        return math.exp(self.mu - self.sigma**2)

    @override
    def skewness(self) -> float:
        return (math.exp(self.sigma**2) + 2) * math.sqrt(math.exp(self.sigma**2) - 1)

    @override
    def excess_kurtosis(self) -> float:
        s2 = self.sigma**2
        return math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 6

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Laplace(Distribution):
    """Laplace (double exponential) distribution."""

    def __init__(
        self,
        location: float = 0.0,
        scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale={scale} must be positive.")
        self.location: float = location
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.laplace(self.location, self.scale))

    @override
    def pdf(self, x: float) -> float:
        return math.exp(-abs(x - self.location) / self.scale) / (2 * self.scale)

    @override
    def cdf(self, x: float) -> float:
        if x < self.location:
            return 0.5 * math.exp((x - self.location) / self.scale)
        return 1 - 0.5 * math.exp(-(x - self.location) / self.scale)

    @override
    def mean(self) -> float:
        return self.location

    @override
    def median(self) -> float:
        return self.location

    @override
    def mode(self) -> float:
        return self.location

    @override
    def variance(self) -> float:
        return 2 * self.scale**2

    @override
    def skewness(self) -> float:
        return 0.0

    @override
    def excess_kurtosis(self) -> float:
        return 3.0

    @override
    def entropy(self) -> float:
        return 1 + math.log(2 * self.scale)

    @override
    def is_infinitely_divisible(self) -> bool:
        return True

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Logistic(Distribution):
    """Logistic distribution."""

    def __init__(
        self,
        location: float = 0.0,
        scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale={scale} must be positive.")
        self.location: float = location
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.logistic(self.location, self.scale))

    @override
    def pdf(self, x: float) -> float:
        z = abs(x - self.location) / self.scale
        e = math.exp(-z)
        return e / (self.scale * (1 + e) ** 2)

    @override
    def cdf(self, x: float) -> float:
        z = (x - self.location) / self.scale
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        e = math.exp(z)
        return e / (1 + e)

    @override
    def mean(self) -> float:
        return self.location

    @override
    def median(self) -> float:
        return self.location

    @override
    def mode(self) -> float:
        return self.location

    @override
    def variance(self) -> float:
        return math.pi**2 * self.scale**2 / 3

    @override
    def skewness(self) -> float:
        return 0.0

    @override
    def excess_kurtosis(self) -> float:
        return 1.2

    @override
    def entropy(self) -> float:
        return math.log(self.scale) + 2

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Cauchy(Distribution):
    """Cauchy (Lorentz) distribution.

    The mean and variance are undefined, so those methods are intentionally
    left unimplemented.
    """

    def __init__(
        self,
        location: float = 0.0,
        scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale={scale} must be positive.")
        self.location: float = location
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.standard_cauchy() * self.scale + self.location)

    @override
    def pdf(self, x: float) -> float:
        z = (x - self.location) / self.scale
        return 1 / (math.pi * self.scale * (1 + z**2))

    @override
    def cdf(self, x: float) -> float:
        z = (x - self.location) / self.scale
        return 0.5 + math.atan(z) / math.pi

    @override
    def median(self) -> float:
        return self.location

    @override
    def mode(self) -> float:
        return self.location

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Beta(Distribution):
    """Beta distribution on the unit interval ``[0, 1]``."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"Parameter alpha={alpha} must be positive.")
        if beta <= 0:
            raise ValueError(f"Parameter beta={beta} must be positive.")
        self.alpha: float = alpha
        self.beta: float = beta
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.beta(self.alpha, self.beta))

    @override
    def pdf(self, x: float) -> float:
        if x < 0 or x > 1:
            return 0.0
        log_beta = (
            math.lgamma(self.alpha)
            + math.lgamma(self.beta)
            - math.lgamma(self.alpha + self.beta)
        )
        if x == 0:
            return 0.0 if self.alpha > 1 else float("inf")
        if x == 1:
            return 0.0 if self.beta > 1 else float("inf")
        exponent = (self.alpha - 1) * math.log(x) + (self.beta - 1) * math.log1p(-x)
        return math.exp(exponent - log_beta)

    @override
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @override
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    @override
    def mode(self) -> float:
        a, b = self.alpha, self.beta
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        if a <= 1 < b:
            return 0.0
        if a > 1 and b <= 1:
            return 1.0
        raise NotImplementedError("The beta distribution has no unique mode here.")

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class ChiSquared(Distribution):
    """Chi-squared distribution with ``df`` degrees of freedom."""

    def __init__(self, df: float, rng: np.random.Generator | None = None) -> None:
        if df <= 0:
            raise ValueError(f"df={df} must be positive.")
        self.df: float = df
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.chisquare(self.df))

    @override
    def pdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        k = self.df / 2
        return math.exp(
            (k - 1) * math.log(x) - x / 2 - k * math.log(2) - math.lgamma(k)
        )

    @override
    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return _regularized_lower_incomplete_gamma(self.df / 2, x / 2)

    @override
    def mean(self) -> float:
        return self.df

    @override
    def variance(self) -> float:
        return 2 * self.df

    @override
    def median(self) -> float:
        return _gamma_median(self.df / 2, 2.0)

    @override
    def mode(self) -> float:
        return max(self.df - 2, 0.0)

    @override
    def skewness(self) -> float:
        return math.sqrt(8 / self.df)

    @override
    def excess_kurtosis(self) -> float:
        return 12 / self.df

    @override
    def is_infinitely_divisible(self) -> bool:
        return True

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class StudentT(Distribution):
    """Student's t distribution with ``df`` degrees of freedom."""

    def __init__(self, df: float, rng: np.random.Generator | None = None) -> None:
        if df <= 0:
            raise ValueError(f"df={df} must be positive.")
        self.df: float = df
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.standard_t(self.df))

    @override
    def pdf(self, x: float) -> float:
        df = self.df
        log_norm = (
            math.lgamma((df + 1) / 2)
            - math.lgamma(df / 2)
            - 0.5 * math.log(df * math.pi)
        )
        return math.exp(log_norm - (self.df + 1) / 2 * math.log1p(x**2 / self.df))

    @override
    def mean(self) -> float:
        if self.df > 1:
            return 0.0
        raise ValueError(f"The mean is undefined for df={self.df} <= 1.")

    @override
    def variance(self) -> float:
        if self.df > 2:
            return self.df / (self.df - 2)
        if self.df > 1:
            return float("inf")
        raise ValueError(f"The variance is undefined for df={self.df} <= 1.")

    @override
    def median(self) -> float:
        return 0.0

    @override
    def mode(self) -> float:
        return 0.0

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Weibull(Distribution):
    """Weibull distribution with ``shape`` and ``scale``."""

    def __init__(
        self,
        shape: float,
        scale: float = 1.0,
        rng: np.random.Generator | None = None,
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
        _ = context
        return float(self.rng.weibull(self.shape) * self.scale)

    @override
    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        k, scale = self.shape, self.scale
        z = x / scale
        return (k / scale) * z ** (k - 1) * math.exp(-(z**k))

    @override
    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-((x / self.scale) ** self.shape))

    @override
    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.shape)

    @override
    def variance(self) -> float:
        k = self.shape
        return self.scale**2 * (math.gamma(1 + 2 / k) - math.gamma(1 + 1 / k) ** 2)

    @override
    def median(self) -> float:
        return self.scale * math.log(2) ** (1 / self.shape)

    @override
    def mode(self) -> float:
        k = self.shape
        if k > 1:
            return self.scale * ((k - 1) / k) ** (1 / k)
        return 0.0

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Rayleigh(Distribution):
    """Rayleigh distribution with scale parameter ``scale``."""

    def __init__(
        self, scale: float = 1.0, rng: np.random.Generator | None = None
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale={scale} must be positive.")
        self.scale: float = scale
        self.rng: np.random.Generator = np.random.default_rng() if rng is None else rng

    @override
    def sample(self, context: Any | None = None) -> float:
        _ = context
        return float(self.rng.rayleigh(self.scale))

    @override
    def pdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        s = self.scale
        return (x / s**2) * math.exp(-(x**2) / (2 * s**2))

    @override
    def cdf(self, x: float) -> float:
        if x < 0:
            return 0.0
        return 1 - math.exp(-(x**2) / (2 * self.scale**2))

    @override
    def mean(self) -> float:
        return self.scale * math.sqrt(math.pi / 2)

    @override
    def variance(self) -> float:
        return (2 - math.pi / 2) * self.scale**2

    @override
    def median(self) -> float:
        return self.scale * math.sqrt(2 * math.log(2))

    @override
    def mode(self) -> float:
        return self.scale

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


class Pareto(Distribution):
    """Pareto (type I) distribution with tail index ``shape`` and ``scale``."""

    def __init__(
        self,
        shape: float,
        scale: float = 1.0,
        rng: np.random.Generator | None = None,
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
        _ = context
        return float(self.scale * (1 + self.rng.pareto(self.shape)))

    @override
    def pdf(self, x: float) -> float:
        if x < self.scale:
            return 0.0
        a, xm = self.shape, self.scale
        return a * xm**a / x ** (a + 1)

    @override
    def cdf(self, x: float) -> float:
        if x < self.scale:
            return 0.0
        return 1 - (self.scale / x) ** self.shape

    @override
    def mean(self) -> float:
        if self.shape > 1:
            return self.shape * self.scale / (self.shape - 1)
        return float("inf")

    @override
    def variance(self) -> float:
        a = self.shape
        if a > 2:
            return self.scale**2 * a / ((a - 1) ** 2 * (a - 2))
        return float("inf")

    @override
    def median(self) -> float:
        return self.scale * 2 ** (1 / self.shape)

    @override
    def mode(self) -> float:
        return self.scale

    @override
    def is_absolutely_continuous(self) -> bool:
        return True


##############################
# MULTIVARIATE DISTRIBUTIONS #
##############################
