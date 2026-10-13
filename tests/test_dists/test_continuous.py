"""Tests for absolutely continuous distributions in :mod:`simdist.dists`."""

import numpy as np
import pytest

from simdist import dists
from simdist.dists import _regularized_lower_incomplete_gamma


def draw(dist, n=10_000):
    return np.array([dist.sample() for _ in range(n)])


def test_regularized_lower_incomplete_gamma():
    # Non-positive x has zero mass.
    assert _regularized_lower_incomplete_gamma(1.0, 0.0) == 0.0
    # Series branch (x < a + 1): P(1, 1) = 1 - e^-1.
    assert _regularized_lower_incomplete_gamma(1.0, 1.0) == pytest.approx(
        1 - np.exp(-1), rel=1e-12
    )
    # Continued fraction branch (x >= a + 1): P(2, 5) = 1 - 6 e^-5.
    assert _regularized_lower_incomplete_gamma(2.0, 5.0) == pytest.approx(
        1 - 6 * np.exp(-5), rel=1e-10
    )


# --------------------------------------------------------------------------- #
# ContinuousUniform
# --------------------------------------------------------------------------- #
def test_continuous_uniform_support_and_moments():
    dist = dists.ContinuousUniform(2.0, 6.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 2.0 and xs.max() <= 6.0
    assert dist.mean() == pytest.approx(4.0)
    assert dist.variance() == pytest.approx((6 - 2) ** 2 / 12)
    assert dist.is_absolutely_continuous()
    assert not dist.is_infinitely_divisible()


def test_continuous_uniform_repr_and_fit():
    dist = dists.ContinuousUniform(2.0, 6.0)
    assert repr(dist) == "ContinuousUniform(lower=2.0, upper=6.0)"
    fitted = dists.ContinuousUniform.fit([1.0, 5.0, 3.0])
    assert fitted.lower == 1.0
    assert fitted.upper == 5.0


def test_continuous_uniform_invalid():
    with pytest.raises(ValueError):
        dists.ContinuousUniform(6.0, 2.0)


# --------------------------------------------------------------------------- #
# Exponential
# --------------------------------------------------------------------------- #
def test_exponential_support_and_moments():
    dist = dists.Exponential(2.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0.0
    assert dist.mean() == pytest.approx(0.5)
    assert dist.variance() == pytest.approx(0.25)
    assert dist.standard_deviation() == pytest.approx(0.5)
    assert xs.mean() == pytest.approx(0.5, rel=0.1)
    assert dist.is_absolutely_continuous()


def test_exponential_pdf_cdf_and_moments():
    dist = dists.Exponential(2.0)
    assert repr(dist) == "Exponential(rate=2.0)"
    assert dist.pdf(0.0) == pytest.approx(2.0)
    assert dist.pdf(1.0) == pytest.approx(2 * np.exp(-2))
    assert dist.pdf(-1.0) == pytest.approx(0.0)
    assert dist.cdf(1.0) == pytest.approx(1 - np.exp(-2))
    assert dist.cdf(-1.0) == pytest.approx(0.0)
    assert dist.median() == pytest.approx(np.log(2) / 2)
    assert dist.mode() == pytest.approx(0.0)
    assert dist.skewness() == pytest.approx(2.0)
    assert dist.excess_kurtosis() == pytest.approx(6.0)
    assert dist.entropy() == pytest.approx(1 - np.log(2))
    assert dist.moment_generating_function(1.0) == pytest.approx(2 / (2 - 1))
    assert dist.expected_shortfall(0.5) == pytest.approx(-(np.log(0.5) + 1) / 2)


def test_exponential_mgf_and_shortfall_validation():
    dist = dists.Exponential(2.0)
    with pytest.raises(ValueError):
        dist.moment_generating_function(3.0)
    with pytest.raises(ValueError):
        dist.expected_shortfall(-0.1)
    with pytest.raises(ValueError):
        dist.expected_shortfall(1.0)


def test_exponential_fit():
    fitted = dists.Exponential.fit([1.0, 2.0, 3.0, 4.0])
    assert isinstance(fitted, dists.Exponential)
    assert fitted.rate == pytest.approx(1 / 2.5)


def test_exponential_invalid():
    for rate in [0.0, -1.0]:
        with pytest.raises(ValueError):
            dists.Exponential(rate)


# --------------------------------------------------------------------------- #
# Gamma
# --------------------------------------------------------------------------- #
def test_gamma_support_and_moments():
    dist = dists.Gamma(2.0, 3.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0.0
    assert dist.mean() == pytest.approx(6.0)
    assert dist.variance() == pytest.approx(18.0)
    assert dist.mode() == pytest.approx(3.0)
    assert dist.skewness() == pytest.approx(2 / np.sqrt(2))
    assert dist.excess_kurtosis() == pytest.approx(3.0)
    assert dist.is_absolutely_continuous()
    assert xs.mean() == pytest.approx(6.0, rel=0.1)


def test_gamma_median_matches_exponential_case():
    # Gamma(shape=1, scale=theta) is Exponential(rate=1/theta); the median is
    # theta * ln(2).
    dist = dists.Gamma(1.0, 2.0)
    assert dist.median() == pytest.approx(2 * np.log(2), rel=1e-6)


def test_gamma_median_monotonic_in_shape():
    medians = [dists.Gamma(shape, 1.0).median() for shape in (1.0, 2.0, 5.0)]
    assert medians[0] < medians[1] < medians[2]


def test_gamma_is_absolutely_continuous():
    dist = dists.Gamma(2.0, 1.0)
    assert dist.is_absolutely_continuous()
    assert dist.is_infinitely_divisible()


def test_gamma_fit_is_classmethod():
    rng = np.random.default_rng(0)
    data = rng.gamma(shape=3.0, scale=2.0, size=50_000)
    fitted = dists.Gamma.fit(data)
    assert isinstance(fitted, dists.Gamma)
    assert fitted.shape == pytest.approx(3.0, rel=0.1)
    assert fitted.scale == pytest.approx(2.0, rel=0.1)


@pytest.mark.parametrize(("shape", "scale"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
def test_gamma_invalid(shape, scale):
    with pytest.raises(ValueError):
        dists.Gamma(shape, scale)


# --------------------------------------------------------------------------- #
# VonMises
# --------------------------------------------------------------------------- #
def test_von_mises_samples_with_default_rng():
    # Regression test: VonMises used to store ``rng=None`` and crash on sample.
    dist = dists.VonMises(0.0, 1.0)
    xs = draw(dist)
    assert xs.min() >= -np.pi
    assert xs.max() <= np.pi


def test_von_mises_moments():
    dist = dists.VonMises(0.5, 2.0, np.random.default_rng(0))
    assert dist.mean() == pytest.approx(0.5)
    assert dist.median() == pytest.approx(0.5)
    assert dist.mode() == pytest.approx(0.5)
    assert dist.is_absolutely_continuous()


def test_von_mises_invalid():
    with pytest.raises(ValueError):
        dists.VonMises(0.0, -1.0)
