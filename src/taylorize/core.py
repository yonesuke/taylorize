"""Core: a small, dependency-free truncated Taylor series type.

A :class:`Taylor` represents the normalized Taylor polynomial of a smooth
function ``x(t)`` around ``t = 0``::

    x(t) = c[0] + c[1] t + c[2] t^2 + ...

with ``c[k] = x^(k)(0) / k!``.  Higher-order coefficients are produced lazily
from a recurrence rule, so any order can be requested after the fact.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

__all__ = [
    "Taylor",
    "variable",
    "constant",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "power",
    "derivative",
    "taylor_series",
    "ode_step",
]


class Taylor:
    """A truncated Taylor polynomial with lazily computed coefficients."""

    __slots__ = ("c", "_rule")

    def __init__(self, coeffs, _rule: Optional[Callable[[int], object]] = None) -> None:
        if isinstance(coeffs, Taylor):
            self.c: List[object] = list(coeffs.c)
            self._rule = coeffs._rule if _rule is None else _rule
        elif isinstance(coeffs, (list, tuple)):
            self.c = list(coeffs)
            self._rule = _rule
        else:
            self.c = [coeffs]
            self._rule = _rule
        if not self.c:
            raise ValueError("Taylor needs at least one coefficient.")

    # -- coefficient access --------------------------------------------------
    @property
    def primal(self):
        """The constant term ``c[0]``."""
        return self.c[0]

    @property
    def order(self) -> int:
        """Currently materialized truncation order."""
        return len(self.c) - 1

    def ensure(self, n: int) -> "Taylor":
        """Materialize coefficients up to degree ``n`` (lazily if possible)."""
        cur = len(self.c) - 1
        if n > cur:
            if self._rule is None:
                self.c.extend([0] * (n - cur))
            else:
                for k in range(cur + 1, n + 1):
                    self.c.append(self._rule(k))
        return self

    def coeff(self, k: int):
        if k < 0:
            raise IndexError(k)
        if k >= len(self.c):
            if self._rule is None:
                return 0
            self.ensure(k)
        return self.c[k]

    def __getitem__(self, k):
        if isinstance(k, slice):
            if k.stop is not None and k.stop > len(self.c):
                self.ensure(k.stop - 1)
            return self.c[k]
        return self.coeff(k)

    def eval(self, h, order: Optional[int] = None):
        """Evaluate the polynomial at ``h`` with Horner's scheme."""
        if order is not None:
            self.ensure(order)
            coeffs = self.c[: order + 1]
        else:
            coeffs = self.c
        acc = 0
        for c in reversed(coeffs):
            acc = acc * h + c
        return acc

    def derivative(self, k: int):
        """Return the unnormalized ``k``-th derivative at 0: ``k! * c[k]``."""
        return self.coeff(k) * math.factorial(k)

    def diff(self) -> "Taylor":
        """Formal derivative ``x'(t)`` as a new lazy Taylor."""
        return Taylor([self.coeff(1)], _rule=lambda k: (k + 1) * self.coeff(k + 1))

    def integrate(self, c0=0) -> "Taylor":
        """Formal integral with constant term ``c0``."""
        return Taylor([c0], _rule=lambda k: self.coeff(k - 1) / k)

    def __repr__(self) -> str:
        return "Taylor([%s])" % ", ".join(repr(v) for v in self.c)

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, Taylor):
            return Taylor([self.c[0] + other], _rule=lambda k: self.coeff(k))
        out = Taylor(
            [self.primal + other.primal],
            _rule=lambda k: self.coeff(k) + other.coeff(k),
        )
        out.ensure(max(self.order, other.order))
        return out

    __radd__ = __add__

    def __neg__(self):
        return Taylor([-self.primal], _rule=lambda k: -self.coeff(k))

    def __sub__(self, other):
        return self + (-_as_taylor(other))

    def __rsub__(self, other):
        return _as_taylor(other) - self

    def __mul__(self, other):
        if not isinstance(other, Taylor):
            return Taylor([self.primal * other], _rule=lambda k: self.coeff(k) * other)
        out = Taylor(
            [self.primal * other.primal],
            _rule=lambda k: sum(
                self.coeff(j) * other.coeff(k - j) for j in range(k + 1)
            ),
        )
        out.ensure(max(self.order, other.order))
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Taylor):
            if other == 0:
                raise ZeroDivisionError("division of Taylor by zero")
            return Taylor([self.primal / other], _rule=lambda k: self.coeff(k) / other)
        y0 = other.primal
        if y0 == 0:
            raise ZeroDivisionError("division by Taylor with zero constant term")
        out = Taylor([self.primal / y0])
        out._rule = lambda k: (
            self.coeff(k)
            - sum(other.coeff(j) * out.coeff(k - j) for j in range(1, k + 1))
        ) / y0
        out.ensure(max(self.order, other.order))
        return out

    def __rtruediv__(self, other):
        return _as_taylor(other) / self

    def __pow__(self, p):
        return power(self, p)

    # -- elementary functions ------------------------------------------------
    def exp(self) -> "Taylor":
        out = Taylor([math.exp(self.primal)])
        out._rule = (
            lambda k: sum(j * self.coeff(j) * out.coeff(k - j) for j in range(1, k + 1))
            / k
        )
        out.ensure(self.order)
        return out

    def log(self) -> "Taylor":
        x0 = self.primal
        if x0 <= 0:
            raise ValueError("log requires a positive constant term.")
        out = Taylor([math.log(x0)])
        out._rule = lambda k: (
            self.coeff(k)
            - sum(j * out.coeff(j) * self.coeff(k - j) for j in range(1, k)) / k
        ) / x0
        out.ensure(self.order)
        return out

    def sqrt(self) -> "Taylor":
        return power(self, 0.5)

    def sin_cos(self):
        s = Taylor([math.sin(self.primal)])
        c = Taylor([math.cos(self.primal)])
        s._rule = (
            lambda k: sum(j * self.coeff(j) * c.coeff(k - j) for j in range(1, k + 1))
            / k
        )
        c._rule = (
            lambda k: -sum(j * self.coeff(j) * s.coeff(k - j) for j in range(1, k + 1))
            / k
        )
        s.ensure(self.order)
        c.ensure(self.order)
        return s, c

    def sin(self) -> "Taylor":
        return self.sin_cos()[0]

    def cos(self) -> "Taylor":
        return self.sin_cos()[1]

    def tan(self) -> "Taylor":
        return self.sin() / self.cos()


# -- helpers -----------------------------------------------------------------
def _as_taylor(x) -> Taylor:
    return x if isinstance(x, Taylor) else Taylor([x])


def variable(x0, order: Optional[int] = None) -> Taylor:
    """The independent variable, ``t = x0 + h``."""
    t = Taylor([x0, 1] if order != 0 else [x0])
    if order:
        t.ensure(order)
    return t


def constant(c, order: Optional[int] = None) -> Taylor:
    t = Taylor([c])
    if order:
        t.ensure(order)
    return t


def power(x, p) -> Taylor:
    """``x ** p`` for a Taylor base and scalar (or Taylor) exponent."""
    if isinstance(p, Taylor):
        return (p * x.log()).exp()
    if not isinstance(x, Taylor):
        return x ** p
    if isinstance(p, int):
        if p == 0:
            return constant(1, order=x.order)
        if p < 0:
            return constant(1) / power(x, -p)
        result = constant(1)
        base = x
        e = p
        while e:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result
    x0 = x.primal
    p_plus_1 = p + 1
    out = Taylor([x0 ** p])
    out._rule = lambda k: (
        sum((j * p_plus_1 - k) * x.coeff(j) * out.coeff(k - j) for j in range(1, k + 1))
        / (k * x0)
    )
    out.ensure(x.order)
    return out


def exp(x):
    return _as_taylor(x).exp()


def log(x):
    return _as_taylor(x).log()


def sqrt(x):
    return _as_taylor(x).sqrt()


def sin(x):
    return _as_taylor(x).sin()


def cos(x):
    return _as_taylor(x).cos()


def tan(x):
    return _as_taylor(x).tan()


# -- JAX-style transforms ----------------------------------------------------
def derivative(f: Callable, k: int = 1) -> Callable:
    """Transform ``f`` into a function returning its ``k``-th derivative."""
    if k < 0:
        raise ValueError("k must be >= 0")

    def grad_k(x0):
        return f(variable(x0, order=k)).derivative(k)

    grad_k.__name__ = getattr(f, "__name__", "grad_k")
    return grad_k


def taylor_series(f: Callable, order: int) -> Callable:
    """Transform ``f`` into a function returning ``[c_0, ..., c_order]``."""
    if order < 0:
        raise ValueError("order must be >= 0")

    def series_fn(x0):
        return list(f(variable(x0, order=order)).ensure(order).c)

    series_fn.__name__ = getattr(f, "__name__", "series_fn")
    return series_fn


def ode_step(f: Callable, order: int = 10) -> Callable:
    """Transform a vector field ``f`` into an ``order``-th Taylor step.

    The returned function ``step(x, dt, t=0.0)`` advances the state ``x``
    (a scalar or a sequence) by a single Taylor-method step of size ``dt``.
    The vector field is evaluated once per step on lazy Taylor jets.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    def step(x, dt, t=0.0):
        scalar = not isinstance(x, (list, tuple))
        xs = [x] if scalar else list(x)
        jets = [Taylor([v]) for v in xs]
        rhs = [f(jets[0])] if scalar else list(f(jets))
        for jet, r in zip(jets, rhs):
            jet._rule = (lambda k, r=r: r.coeff(k - 1) / k)
        out = [jet.eval(dt, order=order) for jet in jets]
        return out[0] if scalar else out

    step.__name__ = "step"
    return step
