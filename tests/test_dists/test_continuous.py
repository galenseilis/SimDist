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


# --------------------------------------------------------------------------- #
# Triangular
# --------------------------------------------------------------------------- #
def test_triangular_support_and_moments():
    dist = dists.Triangular(0.0, 6.0, 3.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0.0 and xs.max() <= 6.0
    assert dist.mean() == pytest.approx(3.0)
    assert dist.mode() == 3.0
    assert dist.variance() == pytest.approx((36 + 9 - 18) / 18)
    assert dist.is_absolutely_continuous()


def test_triangular_pdf_cdf_integrate_to_one():
    dist = dists.Triangular(0.0, 1.0, 0.25)
    assert dist.cdf(0.0) == pytest.approx(0.0)
    assert dist.cdf(1.0) == pytest.approx(1.0)
    assert dist.pdf(0.25) == pytest.approx(2.0)
    a = 2 / (1.0 * 0.25)
    assert dist.pdf(-1.0) == 0.0
    assert dist.pdf(2.0) == 0.0
    # Midpoint of the left slope.
    assert dist.pdf(0.125) == pytest.approx(0.125 * a)


def test_triangular_symmetric_median_and_default_mode():
    dist = dists.Triangular(0.0, 1.0)
    assert dist.mode() == pytest.approx(0.5)
    assert dist.median() == pytest.approx(0.5)
    assert dist.mean() == pytest.approx(0.5)


def test_triangular_degenerate_modes():
    low = dists.Triangular(0.0, 1.0, 0.0)
    high = dists.Triangular(0.0, 1.0, 1.0)
    assert low.pdf(0.0) == pytest.approx(2.0)
    assert high.pdf(1.0) == pytest.approx(2.0)
    assert dists.Triangular(0.0, 1.0, 1.0).median() == pytest.approx(np.sqrt(0.5))


def test_triangular_invalid():
    with pytest.raises(ValueError):
        dists.Triangular(1.0, 1.0)
    with pytest.raises(ValueError):
        dists.Triangular(0.0, 1.0, mode=2.0)


# --------------------------------------------------------------------------- #
# Normal
# --------------------------------------------------------------------------- #
def test_normal_moments_pdf_cdf():
    dist = dists.Normal(2.0, 3.0, np.random.default_rng(0))
    xs = draw(dist)
    assert dist.mean() == pytest.approx(2.0)
    assert dist.variance() == pytest.approx(9.0)
    assert dist.standard_deviation() == pytest.approx(3.0)
    assert dist.median() == pytest.approx(2.0)
    assert dist.mode() == pytest.approx(2.0)
    assert dist.skewness() == pytest.approx(0.0)
    assert dist.excess_kurtosis() == pytest.approx(0.0)
    assert dist.entropy() == pytest.approx(0.5 * np.log(2 * np.pi * np.e * 9))
    assert xs.mean() == pytest.approx(2.0, abs=0.15)
    assert dist.is_infinitely_divisible()
    assert dist.is_absolutely_continuous()


def test_normal_pdf_cdf_values():
    dist = dists.Normal(0.0, 1.0)
    assert dist.pdf(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi))
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.cdf(1.0) == pytest.approx(0.841344746, abs=1e-8)


def test_normal_invalid():
    with pytest.raises(ValueError):
        dists.Normal(0.0, 0.0)


# --------------------------------------------------------------------------- #
# LogNormal
# --------------------------------------------------------------------------- #
def test_log_normal_support_and_moments():
    dist = dists.LogNormal(0.0, 0.5, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() > 0
    assert dist.mean() == pytest.approx(np.exp(0.125))
    assert dist.median() == pytest.approx(1.0)
    assert dist.mode() == pytest.approx(np.exp(-0.25))
    assert dist.variance() == pytest.approx((np.exp(0.25) - 1) * np.exp(0.25))
    assert xs.mean() == pytest.approx(dist.mean(), rel=0.1)


def test_log_normal_cdf_and_invalid():
    dist = dists.LogNormal(0.0, 1.0)
    assert dist.cdf(0.0) == 0.0
    assert dist.cdf(np.e) == pytest.approx(0.841344746, abs=1e-8)
    with pytest.raises(ValueError):
        dists.LogNormal(0.0, -1.0)


# --------------------------------------------------------------------------- #
# Laplace
# --------------------------------------------------------------------------- #
def test_laplace_support_and_moments():
    dist = dists.Laplace(1.0, 2.0, np.random.default_rng(0))
    assert dist.mean() == pytest.approx(1.0)
    assert dist.median() == pytest.approx(1.0)
    assert dist.mode() == pytest.approx(1.0)
    assert dist.variance() == pytest.approx(8.0)
    assert dist.skewness() == pytest.approx(0.0)
    assert dist.excess_kurtosis() == pytest.approx(3.0)
    assert dist.entropy() == pytest.approx(1 + np.log(4))
    assert dist.is_infinitely_divisible()
    assert dist.is_absolutely_continuous()


def test_laplace_pdf_cdf():
    dist = dists.Laplace(0.0, 2.0)
    assert dist.pdf(0.0) == pytest.approx(0.25)
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.cdf(-2.0) == pytest.approx(0.5 * np.exp(-1))
    assert dist.cdf(2.0) == pytest.approx(1 - 0.5 * np.exp(-1))


def test_laplace_invalid():
    with pytest.raises(ValueError):
        dists.Laplace(0.0, 0.0)


# --------------------------------------------------------------------------- #
# Logistic
# --------------------------------------------------------------------------- #
def test_logistic_moments():
    dist = dists.Logistic(1.0, 2.0, np.random.default_rng(0))
    assert dist.mean() == pytest.approx(1.0)
    assert dist.median() == pytest.approx(1.0)
    assert dist.mode() == pytest.approx(1.0)
    assert dist.variance() == pytest.approx(np.pi**2 * 4 / 3)
    assert dist.skewness() == pytest.approx(0.0)
    assert dist.excess_kurtosis() == pytest.approx(1.2)
    assert dist.entropy() == pytest.approx(np.log(2) + 2)
    assert dist.is_absolutely_continuous()


def test_logistic_pdf_cdf_symmetry():
    dist = dists.Logistic(0.0, 1.0)
    assert dist.pdf(0.0) == pytest.approx(0.25)
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.pdf(1.0) == pytest.approx(dist.pdf(-1.0))
    assert dist.cdf(1.0) == pytest.approx(1 - dist.cdf(-1.0))


def test_logistic_invalid():
    with pytest.raises(ValueError):
        dists.Logistic(0.0, -1.0)


# --------------------------------------------------------------------------- #
# Cauchy
# --------------------------------------------------------------------------- #
def test_cauchy_support_and_location():
    dist = dists.Cauchy(3.0, 2.0, np.random.default_rng(0))
    assert dist.median() == pytest.approx(3.0)
    assert dist.mode() == pytest.approx(3.0)
    assert dist.is_absolutely_continuous()


def test_cauchy_pdf_cdf():
    dist = dists.Cauchy(0.0, 1.0)
    assert dist.pdf(0.0) == pytest.approx(1 / np.pi)
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.cdf(1.0) == pytest.approx(0.75)
    assert dist.cdf(-1.0) == pytest.approx(0.25)


def test_cauchy_has_no_mean_or_variance():
    dist = dists.Cauchy(0.0, 1.0)
    with pytest.raises(NotImplementedError):
        dist.mean()
    with pytest.raises(NotImplementedError):
        dist.variance()


def test_cauchy_invalid():
    with pytest.raises(ValueError):
        dists.Cauchy(0.0, 0.0)


# --------------------------------------------------------------------------- #
# Beta
# --------------------------------------------------------------------------- #
def test_beta_support_and_moments():
    dist = dists.Beta(2.0, 3.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0.0 and xs.max() <= 1.0
    assert dist.mean() == pytest.approx(0.4)
    assert dist.variance() == pytest.approx(0.04)
    assert dist.mode() == pytest.approx(1 / 3)


def test_beta_pdf_values():
    dist = dists.Beta(2.0, 3.0)
    assert dist.pdf(0.5) == pytest.approx(1.5)
    assert dist.pdf(0.0) == 0.0
    assert dist.pdf(1.0) == 0.0
    assert dist.pdf(-0.5) == 0.0
    assert dist.pdf(1.5) == 0.0


def test_beta_mode_edge_cases():
    assert dists.Beta(0.5, 2.0).mode() == pytest.approx(0.0)
    assert dists.Beta(2.0, 0.5).mode() == pytest.approx(1.0)


def test_beta_invalid():
    with pytest.raises(ValueError):
        dists.Beta(0.0, 1.0)
    with pytest.raises(ValueError):
        dists.Beta(1.0, -1.0)


# --------------------------------------------------------------------------- #
# ChiSquared
# --------------------------------------------------------------------------- #
def test_chi_squared_support_and_moments():
    dist = dists.ChiSquared(4.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0
    assert dist.mean() == pytest.approx(4.0)
    assert dist.variance() == pytest.approx(8.0)
    assert dist.mode() == pytest.approx(2.0)
    assert dist.skewness() == pytest.approx(np.sqrt(2))
    assert dist.excess_kurtosis() == pytest.approx(3.0)
    assert dist.is_infinitely_divisible()
    assert dist.is_absolutely_continuous()


def test_chi_squared_cdf_and_median_against_reference():
    dist = dists.ChiSquared(2.0)
    # Chi-squared with 2 df is Exp(1/2): cdf(x) = 1 - exp(-x/2).
    assert dist.cdf(2.0) == pytest.approx(1 - np.exp(-1), rel=1e-9)
    assert dist.median() == pytest.approx(2 * np.log(2), rel=1e-6)


def test_chi_squared_invalid():
    with pytest.raises(ValueError):
        dists.ChiSquared(0.0)


# --------------------------------------------------------------------------- #
# StudentT
# --------------------------------------------------------------------------- #
def test_student_t_moments_and_pdf():
    dist = dists.StudentT(5.0, np.random.default_rng(0))
    assert dist.mean() == pytest.approx(0.0)
    assert dist.variance() == pytest.approx(5 / 3)
    assert dist.median() == pytest.approx(0.0)
    assert dist.mode() == pytest.approx(0.0)
    # pdf(0) = Gamma(3) / (Gamma(2.5) * sqrt(5 * pi)).
    expected = 0.37960669
    assert dist.pdf(0.0) == pytest.approx(expected, abs=1e-6)
    assert dist.is_absolutely_continuous()


def test_student_t_undefined_moments():
    with pytest.raises(ValueError):
        dists.StudentT(1.0).mean()
    assert dists.StudentT(1.5).variance() == float("inf")
    with pytest.raises(ValueError):
        dists.StudentT(1.0).variance()


def test_student_t_invalid():
    with pytest.raises(ValueError):
        dists.StudentT(0.0)


# --------------------------------------------------------------------------- #
# Weibull
# --------------------------------------------------------------------------- #
def test_weibull_support_and_moments():
    dist = dists.Weibull(1.5, 2.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0
    assert dist.mean() == pytest.approx(2 * 0.9027453, rel=1e-3)
    assert dist.median() == pytest.approx(2 * np.log(2) ** (1 / 1.5))
    assert dist.mode() == pytest.approx(2 * (0.5 / 1.5) ** (1 / 1.5))
    assert dist.is_absolutely_continuous()


def test_weibull_pdf_cdf():
    dist = dists.Weibull(1.0, 1.0)
    # shape=1 is Exponential(rate=1/scale).
    assert dist.cdf(1.0) == pytest.approx(1 - np.exp(-1))
    assert dist.pdf(0.0) == pytest.approx(1.0)
    assert dist.pdf(-1.0) == 0.0


def test_weibull_mode_at_zero_for_shape_le_one():
    assert dists.Weibull(0.5, 2.0).mode() == 0.0


@pytest.mark.parametrize(("shape", "scale"), [(0.0, 1.0), (1.0, 0.0)])
def test_weibull_invalid(shape, scale):
    with pytest.raises(ValueError):
        dists.Weibull(shape, scale)


# --------------------------------------------------------------------------- #
# Rayleigh
# --------------------------------------------------------------------------- #
def test_rayleigh_support_and_moments():
    dist = dists.Rayleigh(2.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 0
    assert dist.mean() == pytest.approx(2 * np.sqrt(np.pi / 2))
    assert dist.variance() == pytest.approx((2 - np.pi / 2) * 4)
    assert dist.median() == pytest.approx(2 * np.sqrt(2 * np.log(2)))
    assert dist.mode() == pytest.approx(2.0)
    assert dist.is_absolutely_continuous()


def test_rayleigh_cdf_and_invalid():
    dist = dists.Rayleigh(1.0)
    assert dist.cdf(0.0) == 0.0
    assert dist.cdf(1.0) == pytest.approx(1 - np.exp(-0.5))
    with pytest.raises(ValueError):
        dists.Rayleigh(0.0)


# --------------------------------------------------------------------------- #
# Pareto
# --------------------------------------------------------------------------- #
def test_pareto_support_and_moments():
    dist = dists.Pareto(3.0, 2.0, np.random.default_rng(0))
    xs = draw(dist)
    assert xs.min() >= 2.0
    assert dist.mean() == pytest.approx(3.0)
    assert dist.variance() == pytest.approx(3.0)
    assert dist.median() == pytest.approx(2 * 2 ** (1 / 3))
    assert dist.mode() == pytest.approx(2.0)
    assert dist.is_absolutely_continuous()


def test_pareto_cdf_and_heavy_tail():
    dist = dists.Pareto(1.0, 1.0)
    assert dist.cdf(0.5) == 0.0
    assert dist.cdf(2.0) == pytest.approx(0.5)
    assert dist.mean() == float("inf")


def test_pareto_variance_infinite_for_small_shape():
    assert dists.Pareto(2.0, 1.0).variance() == float("inf")


@pytest.mark.parametrize(("shape", "scale"), [(0.0, 1.0), (1.0, 0.0)])
def test_pareto_invalid(shape, scale):
    with pytest.raises(ValueError):
        dists.Pareto(shape, scale)
