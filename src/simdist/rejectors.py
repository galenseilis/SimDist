from typing import Any, override, Callable
import warnings

from .core import Distribution

__all__ = ["Reject", "is_negative", "outside_interval"]


class Reject(Distribution):
    """Add rejection sampling to a distribution.

    For example, lower truncation of a distribution
    to zero can restrict a real support distribution to
    a non-negative real support distribution.
    """

    def __init__(self, dist: Distribution, reject: Callable[[Any, Any], bool]):
        self.dist: Distribution = dist
        self.reject: Callable[[Any, Any], bool] = reject

    @override
    def __repr__(self):
        return f"RejectDistribution({self.dist}, {self.reject})"

    @override
    def sample(self, context: Any | None = None):
        """Rejection sample from distribution."""
        while True:
            candidate = self.dist.sample(context)
            if not self.reject(candidate, context):
                return candidate


def is_negative(candidate: float, context: Any | None = None) -> bool:  # pylint: disable=W0613
    """Reject negative candidates.

    Ignores context.
    """
    if __debug__:
        if context is not None:
            warnings.warn(
                f"`is_negative` does not use context, but context was passed."
            )
    if candidate < 0:
        return True
    return False


def outside_interval(
    candidate: float,
    lower: float = 0,
    upper: float | None = None,
    context: Any | None = None,
) -> bool:  # pylint: disable=W0613
    """Truncate candidates to an interval.

    Ignores context.
    """
    if upper is None:
        upper = float("inf")
    if __debug__:
        if context is not None:
            warnings.warn(
                f"`outside_interval` does not use context, but context was passed."
            )
    if candidate < lower:
        return True
    if candidate > upper:
        return True
    return False
