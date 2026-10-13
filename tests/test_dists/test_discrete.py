"""Tests for discrete distributions in :mod:`simdist.dists`."""

import numpy as np
import pytest

from simdist import dists
from simdist.core import Constant, Transform


def draw(dist, n=10_000):
    return np.array([dist.sample() for _ in range(n)])


# --------------------------------------------------------------------------- #
# Bernoulli
# --------------------------------------------------------------------------- #
def test_bernoulli_support_and_moments():
    dist = dists.Bernoulli(0.3, np.random.default_rng(0))
    xs = draw(dist)
    assert set(np.unique(xs)) <= {0, 1}
    assert xs.mean() == pytest.approx(0.3, rel=0.1)
    assert xs.var() == pytest.approx(0.21, rel=0.2)
    assert dist.mean() == pytest.approx(0.3)
    assert dist.variance() == pytest.approx(0.21)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_bernoulli_invalid(p):
    with pytest.raises(ValueError):
        dists.Bernoulli(p)


def test_bernoulli_infinite_divisibility():
    assert dists.Bernoulli(1.0).is_infinitely_divisible()
    assert not dists.Bernoulli(0.5).is_infinitely_divisible()


def test_bernoulli_repr():
    assert repr(dists.Bernoulli(0.3)) == "Bernoulli(p=0.3)"


# --------------------------------------------------------------------------- #
# Rademacher
# --------------------------------------------------------------------------- #
def test_rademacher_support_and_moments():
    dist = dists.Rademacher(0.25, np.random.default_rng(0))
    xs = draw(dist)
    assert set(np.unique(xs)) <= {-1, 1}
    assert dist.mean() == pytest.approx(2 * 0.25 - 1)
    assert dist.variance() == pytest.approx(4 * 0.25 * 0.75)
    assert xs.mean() == pytest.approx(-0.5, abs=0.1)


def test_rademacher_invalid():
    with pytest.raises(ValueError):
        dists.Rademacher(2.0)


# --------------------------------------------------------------------------- #
# BetaBinomial
# --------------------------------------------------------------------------- #
def test_beta_binomial_support_and_moments():
    dist = dists.BetaBinomial(10, 2.0, 3.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 10
    assert dist.mean() == pytest.approx(10 * 2 / 5)
    assert dist.variance() == pytest.approx(6.0)
    assert xs.mean() == pytest.approx(dist.mean(), rel=0.1)


@pytest.mark.parametrize(
    ("n", "alpha", "beta"),
    [(1.5, 1.0, 1.0), (-1, 1.0, 1.0), (10, 0.0, 1.0), (10, 1.0, -1.0)],
)
def test_beta_binomial_invalid(n, alpha, beta):
    with pytest.raises(ValueError):
        dists.BetaBinomial(n, alpha, beta)


# --------------------------------------------------------------------------- #
# Binomial
# --------------------------------------------------------------------------- #
def test_binomial_support_and_moments():
    dist = dists.Binomial(20, 0.4, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 20
    assert dist.mean() == pytest.approx(8.0)
    assert dist.variance() == pytest.approx(4.8)
    assert xs.mean() == pytest.approx(8.0, rel=0.1)
    assert not dist.is_infinitely_divisible()


@pytest.mark.parametrize(("num", "prob"), [(1.5, 0.5), (-1, 0.5), (10, 1.5)])
def test_binomial_invalid(num, prob):
    with pytest.raises(ValueError):
        dists.Binomial(num, prob)


# --------------------------------------------------------------------------- #
# Empirical
# --------------------------------------------------------------------------- #
def test_empirical_sampling_and_mean():
    dist = dists.Empirical([1, 2, 3, 4], rng=np.random.default_rng(0))
    xs = draw(dist)
    assert set(np.unique(xs)) <= {1, 2, 3, 4}
    assert dist.mean() == pytest.approx(2.5)


def test_empirical_weighted():
    dist = dists.Empirical([0, 10], weights=[0.9, 0.1], rng=np.random.default_rng(0))
    xs = draw(dist)
    assert (xs == 0).mean() == pytest.approx(0.9, abs=0.05)
    assert dist.mean() == pytest.approx(1.0)


def test_empirical_without_support_raises():
    with pytest.raises(ValueError):
        dists.Empirical().sample()
    with pytest.raises(ValueError):
        dists.Empirical().mean()


# --------------------------------------------------------------------------- #
# DiscreteUniform
# --------------------------------------------------------------------------- #
def test_discrete_uniform_support_and_moments():
    dist = dists.DiscreteUniform(1, 6, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() == 1 and xs.max() == 6
    assert dist.mean() == pytest.approx(3.5)
    assert dist.variance() == pytest.approx((36 - 1) / 12)
    assert not dist.is_absolutely_continuous()


def test_discrete_uniform_invalid():
    with pytest.raises(ValueError):
        dists.DiscreteUniform(5, 1)
    with pytest.raises(ValueError):
        dists.DiscreteUniform(0.5, 3)


def test_discrete_uniform_repr_and_not_continuous():
    dist = dists.DiscreteUniform(1, 6)
    assert repr(dist) == "DiscreteUniform(lower=1, upper=6)"
    assert not dist.is_absolutely_continuous()


# --------------------------------------------------------------------------- #
# Hypergeometric
# --------------------------------------------------------------------------- #
def test_hypergeometric_support_and_moments():
    dist = dists.Hypergeometric(30, 20, 10, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 10
    assert dist.mean() == pytest.approx(6.0)
    assert xs.mean() == pytest.approx(6.0, rel=0.1)
    assert not dist.is_absolutely_continuous()


def test_hypergeometric_invalid():
    with pytest.raises(ValueError):
        dists.Hypergeometric(10, 10, 25)
    with pytest.raises(ValueError):
        dists.Hypergeometric(-1, 10, 5)
    with pytest.raises(ValueError):
        dists.Hypergeometric(10, -1, 5)


def test_hypergeometric_variance_and_degenerate_total():
    dist = dists.Hypergeometric(30, 20, 10)
    assert dist.variance() == pytest.approx(1.95918367, rel=1e-6)
    assert dists.Hypergeometric(0, 1, 0).variance() == 0.0
    with pytest.raises(ValueError):
        dists.Hypergeometric(10.5, 10, 5)


# --------------------------------------------------------------------------- #
# NegativeHypergeometric
# --------------------------------------------------------------------------- #
def test_negative_hypergeometric_support_and_moments():
    dist = dists.NegativeHypergeometric(20, 30, 5, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 20
    assert dist.mean() == pytest.approx(5 * 20 / 31)
    assert xs.mean() == pytest.approx(dist.mean(), rel=0.1)
    assert not dist.is_absolutely_continuous()


def test_negative_hypergeometric_invalid():
    with pytest.raises(ValueError):
        dists.NegativeHypergeometric(10, 5, 0)
    with pytest.raises(ValueError):
        dists.NegativeHypergeometric(10, 5, 6)
    with pytest.raises(ValueError):
        dists.NegativeHypergeometric(10.5, 5, 3)
    with pytest.raises(ValueError):
        dists.NegativeHypergeometric(10, -1, 3)


def test_negative_hypergeometric_variance_positive():
    dist = dists.NegativeHypergeometric(20, 30, 5)
    assert dist.variance() > 0.0
    xs = draw(dist)
    assert xs.var() == pytest.approx(dist.variance(), rel=0.25)


# --------------------------------------------------------------------------- #
# PoissonBinomial
# --------------------------------------------------------------------------- #
def test_poisson_binomial_support_and_moments():
    probs = [0.1, 0.2, 0.5, 0.9]
    dist = dists.PoissonBinomial(probs, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= len(probs)
    assert dist.mean() == pytest.approx(sum(probs))
    assert dist.variance() == pytest.approx(sum(p * (1 - p) for p in probs))
    assert xs.mean() == pytest.approx(sum(probs), rel=0.1)
    assert not dist.is_absolutely_continuous()


def test_poisson_binomial_invalid():
    with pytest.raises(ValueError):
        dists.PoissonBinomial([0.5, 1.5])


# --------------------------------------------------------------------------- #
# Noncentral hypergeometric distributions
# --------------------------------------------------------------------------- #
def test_fisher_noncentral_hypergeometric_support_and_pmf():
    dist = dists.FisherNoncentralHypergeometric(20, 20, 10, 2.0)
    # The stored PMF must be normalized and supported correctly.
    assert dist._probs.sum() == pytest.approx(1.0)
    assert dist._support[0] == 0
    assert dist._support[-1] == 10
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 10
    assert dist.mean() == pytest.approx(dist._support @ dist._probs)


def test_fisher_odds_increases_white_draws():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    low = dists.FisherNoncentralHypergeometric(20, 20, 10, 1.0, rng_a)
    high = dists.FisherNoncentralHypergeometric(20, 20, 10, 4.0, rng_b)
    assert high.mean() > low.mean()


def test_fisher_invalid():
    with pytest.raises(ValueError):
        dists.FisherNoncentralHypergeometric(10, 10, 25, 1.0)
    with pytest.raises(ValueError):
        dists.FisherNoncentralHypergeometric(10, 10, 5, 0.0)
    with pytest.raises(ValueError):
        dists.FisherNoncentralHypergeometric(10.5, 10, 5, 1.0)
    with pytest.raises(ValueError):
        dists.FisherNoncentralHypergeometric(10, -1, 5, 1.0)


def test_wallenius_noncentral_hypergeometric_support():
    dist = dists.WalleniusNoncentralHypergeometric(
        20, 20, 10, 2.0, np.random.default_rng(0)
    )
    xs = draw(dist)
    assert xs.min() >= 0 and xs.max() <= 10


def test_wallenius_invalid():
    with pytest.raises(ValueError):
        dists.WalleniusNoncentralHypergeometric(10, 10, 25, 1.0)
    with pytest.raises(ValueError):
        dists.WalleniusNoncentralHypergeometric(10, 10, 5, -1.0)
    with pytest.raises(ValueError):
        dists.WalleniusNoncentralHypergeometric(10, 10.5, 5, 1.0)
    with pytest.raises(ValueError):
        dists.WalleniusNoncentralHypergeometric(10, -1, 5, 1.0)


# --------------------------------------------------------------------------- #
# Benford
# --------------------------------------------------------------------------- #
def test_benford_leading_digit():
    dist = dists.Benford(rng=np.random.default_rng(0))
    xs = draw(dist)
    assert set(np.unique(xs)) <= set(range(1, 10))
    assert (xs == 1).mean() == pytest.approx(np.log10(2), abs=0.02)
    assert dist.mean() == pytest.approx(3.440, abs=1e-3)


def test_benford_invalid_base():
    with pytest.raises(ValueError):
        dists.Benford(base=1)
    with pytest.raises(ValueError):
        dists.Benford(base=2.5)


def test_benford_is_not_absolutely_continuous():
    assert not dists.Benford().is_absolutely_continuous()


# --------------------------------------------------------------------------- #
# Geometric, NegativeBinomial, Poisson
# --------------------------------------------------------------------------- #
def test_geometric_support_and_moments():
    dist = dists.Geometric(0.25, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 1
    assert dist.mean() == pytest.approx(4.0)
    assert dist.variance() == pytest.approx(0.75 / 0.25**2)
    assert dist.is_infinitely_divisible()


def test_geometric_invalid():
    for p in [0.0, 1.5]:
        with pytest.raises(ValueError):
            dists.Geometric(p)


def test_negative_binomial_support_and_moments():
    dist = dists.NegativeBinomial(3, 0.5, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0
    assert dist.mean() == pytest.approx(3.0)
    assert xs.mean() == pytest.approx(3.0, rel=0.15)


def test_negative_binomial_invalid():
    with pytest.raises(ValueError):
        dists.NegativeBinomial(0, 0.5)
    with pytest.raises(ValueError):
        dists.NegativeBinomial(3, 0.0)


def test_negative_binomial_variance_and_divisibility():
    dist = dists.NegativeBinomial(3, 0.5)
    assert dist.variance() == pytest.approx(6.0)
    assert dist.is_infinitely_divisible()


def test_poisson_support_moments_and_addition():
    dist = dists.Poisson(2.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0
    assert dist.mean() == pytest.approx(2.0)
    assert dist.variance() == pytest.approx(2.0)
    assert xs.mean() == pytest.approx(2.0, rel=0.1)
    assert dist.is_infinitely_divisible()
    combined = dist + dists.Poisson(3.0)
    assert isinstance(combined, dists.Poisson)
    assert combined.rate == pytest.approx(5.0)


def test_poisson_addition_with_non_poisson_uses_transform():
    dist = dists.Poisson(2.0, np.random.default_rng(0))
    combined = dist + Constant(1)
    assert isinstance(combined, Transform)
    assert combined.sample() >= 1.0


def test_poisson_invalid():
    with pytest.raises(ValueError):
        dists.Poisson(-1.0)


# --------------------------------------------------------------------------- #
# Soliton distributions
# --------------------------------------------------------------------------- #
def test_ideal_soliton_support_and_probability():
    dist = dists.IdealSoliton(50, np.random.default_rng(0))
    xs = draw(dist, n=20_000)
    assert xs.min() >= 1 and xs.max() <= 50
    assert (xs == 1).mean() == pytest.approx(1 / 50, abs=0.01)


def test_ideal_soliton_invalid():
    with pytest.raises(ValueError):
        dists.IdealSoliton(0)
    with pytest.raises(ValueError):
        dists.IdealSoliton(2.5)


def test_robust_soliton_support_and_normalization():
    dist = dists.RobustSoliton(50, c=0.1, delta=0.05, rng=np.random.default_rng(0))
    assert dist._probs.sum() == pytest.approx(1.0)
    xs = draw(dist, n=20_000)
    assert xs.min() >= 1 and xs.max() <= 50


def test_robust_soliton_invalid():
    with pytest.raises(ValueError):
        dists.RobustSoliton(0)
    with pytest.raises(ValueError):
        dists.RobustSoliton(2.5)
    with pytest.raises(ValueError):
        dists.RobustSoliton(10, c=0.0)
    with pytest.raises(ValueError):
        dists.RobustSoliton(10, delta=1.5)
