"""Core: the minimal lazy Taylor ring.

This module knows *nothing* about elementary or special functions,
host-library interop, or coefficient backends.  It provides exactly:

- the ring operations of formal power series (``+``, ``-``, ``*``, and
  division by a unit series, which is part of the ring structure),
- the formal derivative :meth:`Taylor.diff` and Picard integral
  :meth:`Taylor.integrate`,
- the ODE wiring primitive :meth:`Taylor.bind_ode`, from which every
  function in :mod:`taylorize.functions` is built,
- a *pending* state: a Taylor whose coefficients must not be read until
  its ODE has been wired (reading raises instead of silently returning 0).

Coefficients are opaque objects; the core only assumes they support
``+``, ``*`` and division by (small) integers.  ``float``, ``Fraction``,
``mpmath``, ``sympy`` and array scalars all qualify.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

__all__ = ["Taylor", "variable", "constant"]


def _pending_rule(k: int):
    raise RuntimeError(
        "Taylor coefficients were read before bind_ode() wired the ODE. "
        "This usually means a coupled system was evaluated during construction."
    )


class Taylor:
    """A lazy truncated Taylor series ``x(h) = sum_k c[k] h^k``,
    ``c[k] = x^(k)(0) / k!``, with coefficients produced on demand."""

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

    # -- construction --------------------------------------------------------
    @classmethod
    def pending(cls, primal) -> "Taylor":
        """A Taylor with a known constant term whose ODE is not wired yet.

        Reading any higher coefficient raises until :meth:`bind_ode` is
        called.  Coupled systems are built by creating pending states,
        forming their derivative expressions, and then wiring them.
        """
        return cls([primal], _rule=_pending_rule)

    def bind_ode(self, dy: "Taylor") -> "Taylor":
        """Wire the ODE ``y' = dy`` to this series: ``c[k] = dy[k-1] / k``."""
        self._rule = lambda k: dy.coeff(k - 1) / k
        return self

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
        """Formal derivative ``x'(h)`` as a new lazy Taylor."""
        return Taylor([self.coeff(1)], _rule=lambda k: (k + 1) * self.coeff(k + 1))

    def integrate(self, c0=0) -> "Taylor":
        """Formal integral with constant term ``c0`` (Picard operator)."""
        return Taylor([c0]).bind_ode(self)

    def __repr__(self) -> str:
        return "Taylor([%s])" % ", ".join(repr(v) for v in self.c)

    # -- ring arithmetic -------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, Taylor):
            return Taylor([self.c[0] + other], _rule=lambda k: self.coeff(k))
        return Taylor(
            [self.primal + other.primal],
            _rule=lambda k: self.coeff(k) + other.coeff(k),
        )

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
        return Taylor(
            [self.primal * other.primal],
            _rule=lambda k: sum(
                self.coeff(j) * other.coeff(k - j) for j in range(k + 1)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        # Division by a unit series is part of the power-series ring:
        # it uses only ring operations, so it belongs to the core.
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
        return out

    def __rtruediv__(self, other):
        return _as_taylor(other) / self

    def __pow__(self, p):
        if isinstance(p, int):
            if p == 0:
                return constant(1, order=self.order)
            if p < 0:
                return constant(1) / (self ** (-p))
            result, base, e = constant(1), self, p
            while e:
                if e & 1:
                    result = result * base
                base = base * base
                e >>= 1
            return result
        # Non-integer / Taylor exponents are functions, not ring operations:
        # they live in taylorize.functions (deferred import, no cycle).
        from .functions import power

        return power(self, p)


# -- helpers -------------------------------------------------------------------
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
