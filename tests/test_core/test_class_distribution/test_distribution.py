"""Tests for the ``Distribution`` base class and core combinators."""

import operator

import numpy as np
import pytest

from simdist import dists
from simdist.core import (
    Compose,
    Constant,
    Degenerate,
    Distribution,
    FiniteMixture,
    Max,
    Min,
    Range,
    Transform,
)


def test_distribution_is_abstract():
    with pytest.raises(TypeError):
        Distribution()  # type: ignore[abstract]


def test_base_repr_uses_class_name():
    assert repr(Min([Constant(1), Constant(2)])) == "Min"
    assert repr(Max([Constant(1), Constant(2)])) == "Max"
    assert repr(Range([Constant(1), Constant(2)])) == "Range"


def test_combinator_reprs():
    transform = Transform((Constant(1),), abs)
    assert repr(transform).startswith("Transform(")
    compose = Compose(dists.Exponential, (Constant(1),))
    assert repr(compose).startswith("Compose(")
    mixture = FiniteMixture([1], [Constant(0)])
    assert repr(mixture).startswith("FiniteMixture(")


def test_call_builds_composite_distribution():
    composed = dists.Exponential(1.0)(dists.ContinuousUniform(1.0, 2.0))
    assert isinstance(composed, Compose)
    assert composed.sample() >= 0.0


def test_call_accepts_iterable_of_distributions():
    composed = dists.Gamma(1.0, 1.0)(
        [dists.ContinuousUniform(1.0, 2.0), dists.ContinuousUniform(1.0, 2.0)]
    )
    assert isinstance(composed, Compose)
    assert composed.sample() >= 0.0


def test_call_with_callable_returns_transform():
    transformed = dists.Exponential(1.0)(lambda x: x + 1)
    assert isinstance(transformed, Transform)
    assert transformed.sample() >= 1.0


def test_call_with_invalid_argument_raises():
    with pytest.raises(TypeError):
        dists.Exponential(1.0)(3)


def test_dist_cast_handles_numbers_callables_strings_and_errors():
    assert (Constant(1) + 4).sample() == 5
    assert (Constant(1) + (lambda context: 4)).sample() == 5
    assert (Constant("a") + "b").sample() == "ab"
    with pytest.raises(ValueError):
        Constant(1) + object()


def test_constant_wrapping_distribution_delegates_moments():
    wrapped = Constant(dists.Exponential(2.0))
    assert wrapped.sample().rate == pytest.approx(2.0)
    assert wrapped.mean() == pytest.approx(0.5)
    assert wrapped.variance() == pytest.approx(0.25)


def test_repr_uses_class_name():
    assert repr(Constant(1)) == "Constant(1)"
    assert repr(Degenerate(abs)) == f"Degenerate({abs})"


def test_degenerate_samples_callable_with_context():
    dist = Degenerate(lambda context: context["x"] * 2)
    assert dist.sample({"x": 3}) == 6


def test_constant_sample_and_moments():
    dist = Constant(4.5)
    assert dist.sample() == 4.5
    assert dist.mean() == pytest.approx(4.5)
    assert dist.variance() == pytest.approx(0.0)


def test_constant_infinite_divisibility():
    assert Constant(0).is_infinitely_divisible()
    assert not Constant(1).is_infinitely_divisible()


def test_batch_sample_default_and_size():
    dist = Constant(3)
    assert dist.batch_sample(size=5) == [3, 3, 3, 3, 3]
    assert len(dist.batch_sample()) == 10_000


def test_arithmetic_operators_with_distributions():
    a, b = Constant(6), Constant(2)
    assert (a + b).sample() == 8
    assert (a - b).sample() == 4
    assert (a * b).sample() == 12
    assert (a / b).sample() == 3
    assert abs(Constant(-7)).sample() == 7


def test_arithmetic_operators_with_scalars():
    a = Constant(6)
    assert (a + 2).sample() == 8
    assert (a - 2).sample() == 4
    assert (a * 2).sample() == 12
    assert (a / 2).sample() == 3


def test_transform_applies_operator_to_samples():
    dist = Transform((Constant(2), Constant(3)), operator.mul)
    assert dist.sample() == 6


def test_min_max_range():
    dists_ = [Constant(1), Constant(5), Constant(3)]
    assert Min(dists_).sample() == 1
    assert Max(dists_).sample() == 5
    assert Range(dists_).sample() == 4


def test_compose_rebuilds_distribution_from_samples():
    rate = dists.ContinuousUniform(1.0, 2.0, np.random.default_rng(0))
    composed = Compose(dists.Exponential, (rate,))
    sampled = composed.sample()
    assert 0.0 <= sampled


def test_folded_cdf():
    exp = dists.Exponential(1.0)
    assert exp.folded_cdf(0.0) == pytest.approx(0.0)
    assert exp.folded_cdf(0.5) == pytest.approx(1 - np.exp(-0.5))
    # Above the median the folded CDF reflects back.
    assert exp.folded_cdf(2.0) == pytest.approx(np.exp(-2.0))


def test_unimplemented_methods_raise():
    dist = Constant(1)
    for call in [
        lambda: dist.pdf(0.0),
        lambda: dist.cdf(0.0),
        lambda: dist.quantile(0.5),
        lambda: dist.median(),
        lambda: dist.mode(),
        lambda: dist.standard_deviation(),
        lambda: dist.skewness(),
        lambda: dist.excess_kurtosis(),
        lambda: dist.entropy(),
        lambda: dist.moment_generating_function(0.0),
        lambda: dist.characteristic_function(0.0),
        lambda: dist.expected_shortfall(0.5),
        lambda: dist.is_absolutely_continuous(),
        lambda: dist.limit_dist({}),
    ]:
        with pytest.raises(NotImplementedError):
            call()


class _MinimalDistribution(Distribution):
    """Concrete subclass exposing only the abstract ``sample`` method."""

    def sample(self, context=None):
        return 0.0


def test_base_defaults_raise_for_minimal_subclass():
    dist = _MinimalDistribution()
    for call in [
        lambda: dist.pdf(0.0),
        lambda: dist.cdf(0.0),
        lambda: dist.quantile(0.5),
        lambda: dist.mean(),
        lambda: dist.median(),
        lambda: dist.mode(),
        lambda: dist.variance(),
        lambda: dist.standard_deviation(),
        lambda: dist.mean_absolute_difference(),
        lambda: dist.skewness(),
        lambda: dist.excess_kurtosis(),
        lambda: dist.entropy(),
        lambda: dist.fisher_information(),
        lambda: dist.is_infinitely_divisible(),
        lambda: dist.is_absolutely_continuous(),
    ]:
        with pytest.raises(NotImplementedError):
            call()


def test_is_absolutely_continuous_correct_spelling():
    assert dists.Exponential(1.0).is_absolutely_continuous()
    assert not dists.Bernoulli(0.5).is_absolutely_continuous()


def test_misspelled_is_absolutely_continous_deprecated():
    exp = dists.Exponential(1.0)
    with pytest.warns(DeprecationWarning):
        assert exp.is_absolutely_continous() is True


def test_finite_mixture_normalizes_weights():
    mixture = FiniteMixture([2, 2], [Constant(0), Constant(1)])
    assert mixture.weights == pytest.approx([0.5, 0.5])


def test_finite_mixture_moments_of_constants():
    mixture = FiniteMixture([1, 3], [Constant(0), Constant(4)])
    assert mixture.mean() == pytest.approx(3.0)
    expected_variance = 0.25 * (0 - 3) ** 2 + 0.75 * (4 - 3) ** 2
    assert mixture.variance() == pytest.approx(expected_variance)


def test_finite_mixture_sampling_only_returns_component_values():
    mixture = FiniteMixture(
        [1, 1], [Constant(0), Constant(10)], np.random.default_rng(0)
    )
    samples = {mixture.sample() for _ in range(100)}
    assert samples <= {0, 10}
    assert samples == {0, 10}


def test_finite_mixture_validation():
    with pytest.raises(ValueError):
        FiniteMixture([1, 1], [Constant(0)])
    with pytest.raises(ValueError):
        FiniteMixture([], [])
    with pytest.raises(ValueError):
        FiniteMixture([-1, 1], [Constant(0), Constant(1)])
    with pytest.raises(ValueError):
        FiniteMixture([0, 0], [Constant(0), Constant(1)])
