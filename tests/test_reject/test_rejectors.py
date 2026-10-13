"""Tests for rejection sampling helpers in :mod:`simdist.rejectors`."""

import functools

import numpy as np
import pytest

from simdist import dists
from simdist.rejectors import Reject, is_negative, outside_interval


def test_is_negative():
    assert is_negative(-1.0)
    assert not is_negative(1.0)


def test_is_negative_warns_when_context_passed():
    with pytest.warns(UserWarning):
        assert not is_negative(1.0, context={"unused": True})


def test_outside_interval_bounds():
    assert outside_interval(-1.0, None, lower=0, upper=1)
    assert outside_interval(2.0, None, lower=0, upper=1)
    assert not outside_interval(0.5, None, lower=0, upper=1)
    # ``upper=None`` means unbounded above.
    assert not outside_interval(100.0, None, lower=0, upper=None)


def test_reject_with_is_negative_produces_non_negative():
    base = dists.ContinuousUniform(-1.0, 1.0, np.random.default_rng(0))
    dist = Reject(base, is_negative)
    xs = np.array([dist.sample() for _ in range(1_000)])
    assert xs.min() >= 0.0


def test_reject_with_partial_outside_interval():
    base = dists.ContinuousUniform(-1.0, 1.0, np.random.default_rng(0))
    reject = functools.partial(outside_interval, lower=0.0, upper=0.5)
    dist = Reject(base, reject)
    xs = np.array([dist.sample() for _ in range(1_000)])
    assert xs.min() >= 0.0
    assert xs.max() <= 0.5


def test_reject_repr():
    base = dists.ContinuousUniform(0.0, 1.0)
    assert repr(Reject(base, is_negative)).startswith("Reject(")
