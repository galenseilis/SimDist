# Design decisions

This page records *why* SimDist is built the way it is. If you are deciding
whether to use it, or how to extend it, these are the trade-offs to weigh.

## NumPy is the only dependency

SimDist deliberately avoids heavier numerical stacks. Convenience functions
that would otherwise come from SciPy — the noncentral hypergeometric
distributions, the incomplete gamma function used by `Gamma.median` and
`ChiSquared.cdf`, the soliton distributions — are implemented directly with
`numpy` and the standard library's `math`.

The benefit is a small, predictable install that is easy to vendor into a
simulation environment. The cost is that we implement and maintain numerical
routines ourselves, so the library is intentionally narrow in scope.

## Simulations, not statistics

SimDist is built for *drawing samples inside simulations*, not for fitting and
statistical inference. The organising abstraction is therefore `sample`, and
everything else is optional. This is why:

- The base class is small and `sample` is the only required method.
- Moment computations are optional and raise when unavailable, rather than
  being approximated.
- There is no attempt to provide `fit` for every distribution, only where it is
  straightforward.

## A sampling context

`sample(self, context=None)` threads a dictionary of simulation state through
composition. The built-in distributions ignore it, but it lets a distribution
depend on values that are only known at sampling time — for example, other
variables in a simulation — without capturing them at construction.

This keeps distribution objects reusable across contexts while still allowing
stateful simulators to inject dependencies at draw time.

## Explicit randomness

Every distribution accepts a `rng: numpy.random.Generator` and defaults to a
fresh `numpy.random.default_rng()`. Passing a seeded generator makes a
simulation reproducible. Making `rng` a constructor argument — rather than a
genderless global — means two distributions can be seeded independently, and it
avoids the classic bug where a `None` generator silently uses the global state.

## Structural composition over closure detection

The `+`, `-`, `*`, `/` operators do not try to derive the distribution of a
sum from the operands' families. They build a `Transform` that applies the
operation to samples. A distribution such as `Poisson` opts in to a closed form
by overriding the operator and calling `super()` when it does not apply.

This makes the default behaviour predictable and avoids a large, fragile table
of "if both operands are X, the result is Y" rules.

## `simdist/__init__.py` does not re-export names

The package root is intentionally empty apart from a banner. You import from
`simdist.dists`, `simdist.core`, or `simdist.rejectors`. There are two reasons:

- It makes the origin of each name obvious, which matters when a library
  contains many distributions with similar names.
- It keeps the top-level namespace free so the modules can be reorganised
  without breaking `import simdist` behaviour.

## Backwards compatibility

The method name `is_absolutely_continuous` is the correct spelling. The
original misspelling `is_absolutely_continous` is retained as a deprecated
alias that emits a `DeprecationWarning`, so existing code continues to run
while new code can adopt the correct spelling.
