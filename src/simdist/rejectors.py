import warnings
from collections.abc import Callable
from typing import Any, Final, override

from .core import Distribution

__all__: Final[list[str]] = ["Reject", "is_negative", "outside_interval"]


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
        return f"{self.__class__.__name__}({self.dist}, {self.reject})"

    @override
    def sample(self, context: Any | None = None) -> float | Distribution:
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
                "`is_negative` does not use context, but context was passed.",
                stacklevel=2,
            )
    if candidate < 0:
        return True
    return False


def outside_interval(
    candidate: float,
    context: Any | None = None,  # pylint: disable=W0613
    lower: float = 0,
    upper: float | None = None,
) -> bool:
    """Truncate candidates to an interval.

    The signature matches the ``reject(candidate, context)`` calling convention
    used by :class:`Reject`, so it can be passed directly or via
    :func:`functools.partial` with the ``lower``/``upper`` keyword arguments.
    Ignores context.
    """
    _ = context
    if upper is None:
        upper = float("inf")
    if candidate < lower:
        return True
    if candidate > upper:
        return True
    return False
