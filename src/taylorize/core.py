"""Core: the minimal lazy non-commutative / skew power-series ring.

This module knows *nothing* about elementary or special functions,
host-library interop, or coefficient backends.  It provides:

- Level 0: Non-commutative formal power-series ring (``+``, ``-``, ``*``,
  ``rdiv`` / ``ldiv``, ``shift``, integer ``**``, ``compose``), with optional
  Ore/skew ``derivation`` support.
- Level 1: Algebraic / implicit equations ``P(h, Y) = 0`` via
  :meth:`Taylor.solve_algebraic`.
- Level 2: Liouville extensions (:meth:`Taylor.diff`, :meth:`Taylor.euler_diff`,
  :meth:`Taylor.integrate`, :meth:`Taylor.exp_integrate_left`,
  :meth:`Taylor.exp_integrate_right`, :meth:`Taylor.log_diff_left`,
  :meth:`Taylor.log_diff_right`).
- Level 3: Holonomic / Fuchsian regular singular equations
  ``I(theta) Y = RHS`` via :meth:`Taylor.bind_euler_ode`.
- Level 4: High-order & differentially algebraic ODEs via
  :meth:`Taylor.bind_ode` and the universal :meth:`Taylor.bind_recurrence`.

Coefficients are opaque objects of type ``T``; the core only assumes they
support ``+``, ``-``, ``*``, and division by small integers (or an explicit
inverse callback).  ``float``, ``Fraction``, ``mpmath``, ``sympy``, array
scalars, matrices, and non-commutative operators all qualify.
"""
from __future__ import annotations

import math
from typing import Callable, Generic, Iterable, Optional, Sequence, TypeVar, Union, overload

__all__ = ["Taylor", "variable", "constant", "Derivation"]

T = TypeVar("T")
U = TypeVar("U")
Derivation = Callable[[T], T]


def _sum(items: Iterable[T], zero: T) -> T:
    """Fold ``items`` with ``+`` without requiring ``0 + x`` when non-empty."""
    it = iter(items)
    try:
        acc = next(it)
    except StopIteration:
        return zero
    for x in it:
        acc = acc + x  # type: ignore[operator]
    return acc


def _pending_rule(k: int):
    raise RuntimeError(
        f"Taylor coefficient c[{k}] was read before bind_ode()/bind_recurrence() wired it. "
        "This usually means a coupled system was evaluated during construction."
    )


def _merge_derivation(
    d1: Optional[Derivation[T]],
    d2: Optional[Derivation[T]],
) -> Optional[Derivation[T]]:
    if d1 is None:
        return d2
    if d2 is None or d1 is d2 or d1 == d2:
        return d1
    raise ValueError("Cannot combine Taylor series with different derivations.")


def _is_zero(val) -> bool:
    try:
        return bool(val == 0)
    except (TypeError, ValueError):
        return False


class Taylor(Generic[T]):
    """A lazy truncated formal power series ``sum_{k>=0} c[k] h^k`` over any
    (possibly non-commutative or skew) ring ``T``, with coefficients produced
    on demand."""

    __slots__ = ("c", "_rule", "_derivation")
    __iter__ = None

    def __init__(
        self,
        coeffs: Union["Taylor[T]", Sequence[T], T],
        _rule: Optional[Callable[[int], T]] = None,
        derivation: Optional[Derivation[T]] = None,
    ) -> None:
        if isinstance(coeffs, Taylor):
            self.c: list[T] = list(coeffs.c)
            self._rule: Optional[Callable[[int], T]] = coeffs._rule if _rule is None else _rule
            self._derivation: Optional[Derivation[T]] = (
                coeffs._derivation if derivation is None else derivation
            )
        else:
            self.c = list(coeffs) if isinstance(coeffs, (list, tuple)) else [coeffs]  # type: ignore[arg-type]
            self._rule = _rule
            self._derivation = derivation
        if not self.c:
            raise ValueError("Taylor needs at least one coefficient.")

    # -- Base: Lazy Engine & Pending Knot-Tying -------------------------------
    @classmethod
    def pending(
        cls,
        *initial_coeffs: T,
        derivation: Optional[Derivation[T]] = None,
    ) -> "Taylor[T]":
        """A Taylor with known initial coefficient(s) whose recurrence is not wired yet.

        Reading any higher coefficient raises until :meth:`bind_recurrence`,
        :meth:`bind_ode`, or :meth:`bind_euler_ode` is called.
        """
        if not initial_coeffs:
            raise ValueError("pending() requires at least one initial coefficient.")
        return cls(list(initial_coeffs), _rule=_pending_rule, derivation=derivation)

    def bind_recurrence(self, rule: Callable[[int], T]) -> "Taylor[T]":
        """Universal recurrence binder: defines ``c[k] = rule(k)`` for ``k >= len(c)``."""
        self._rule = rule
        return self

    @property
    def primal(self) -> T:
        """The constant term ``c[0]``."""
        return self.c[0]

    @property
    def order(self) -> int:
        """Currently materialized truncation order."""
        return len(self.c) - 1

    @property
    def zero_elem(self) -> T:
        """Additive zero in the coefficient ring ``T``."""
        return self.c[0] * 0  # type: ignore[operator]

    def ensure(self, n: int) -> "Taylor[T]":
        """Materialize coefficients up to degree ``n`` (lazily if possible)."""
        for k in range(len(self.c), n + 1):
            self.c.append(self.zero_elem if self._rule is None else self._rule(k))
        return self

    def coeff(self, k: int) -> T:
        """Return ``c[k]`` (returning ``zero_elem`` for ``k < 0``)."""
        if k < 0:
            return self.zero_elem
        if k >= len(self.c):
            return self.zero_elem if self._rule is None else self.ensure(k).c[k]
        return self.c[k]

    @overload
    def __getitem__(self, k: int) -> T: ...

    @overload
    def __getitem__(self, k: slice) -> list[T]: ...

    def __getitem__(self, k: Union[int, slice]) -> Union[T, list[T]]:
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
        acc = coeffs[-1]
        for c in reversed(coeffs[:-1]):
            acc = acc * h + c  # type: ignore[operator]
        return acc

    def derivative(self, k: int) -> T:
        """Return the unnormalized ``k``-th derivative at 0: ``k! * c[k]``."""
        return self.coeff(k) * math.factorial(k)  # type: ignore[operator]

    def __repr__(self) -> str:
        return "Taylor([%s])" % ", ".join(repr(v) for v in self.c)

    # =========================================================================
    # LEVEL 0: Non-Commutative Ring Operations (with Skew/Ore & Shift)
    # =========================================================================
    def __add__(self, other: Union["Taylor[T]", T, int]) -> "Taylor[T]":
        if not isinstance(other, Taylor):
            return Taylor(
                [self.c[0] + other],  # type: ignore[operator]
                _rule=lambda k: self.coeff(k),
                derivation=self._derivation,
            )
        delta = _merge_derivation(self._derivation, other._derivation)
        return Taylor(
            [self.coeff(0) + other.coeff(0)],  # type: ignore[operator]
            _rule=lambda k: self.coeff(k) + other.coeff(k),  # type: ignore[operator]
            derivation=delta,
        )

    def __radd__(self, other: Union[T, int]) -> "Taylor[T]":
        if isinstance(other, int) and other == 0:
            return self
        return Taylor(
            [other + self.c[0]],  # type: ignore[operator]
            _rule=lambda k: self.coeff(k),
            derivation=self._derivation,
        )

    def __neg__(self) -> "Taylor[T]":
        return Taylor(
            [-self.c[0]],  # type: ignore[operator]
            _rule=lambda k: -self.coeff(k),  # type: ignore[operator]
            derivation=self._derivation,
        )

    def __sub__(self, other: Union["Taylor[T]", T, int]) -> "Taylor[T]":
        return self + (-other)  # type: ignore[operator]

    def __rsub__(self, other: Union[T, int]) -> "Taylor[T]":
        return other + (-self)  # type: ignore[operator]

    def __mul__(self, other: Union["Taylor[T]", T, int]) -> "Taylor[T]":
        if not isinstance(other, Taylor):
            if self._derivation is None:
                return Taylor(
                    [self.c[0] * other],  # type: ignore[operator]
                    _rule=lambda k: self.coeff(k) * other,  # type: ignore[operator]
                )
            return self * Taylor([other], derivation=self._derivation)  # type: ignore[list-item]

        delta = _merge_derivation(self._derivation, other._derivation)
        c0 = self.coeff(0) * other.coeff(0)  # type: ignore[operator]
        if delta is None:
            def mul_rule(k: int) -> T:
                j_min = 0 if other._rule is not None else max(0, k - len(other.c) + 1)
                j_max = k if self._rule is not None else min(k, len(self.c) - 1)
                terms: list[T] = []
                for j in range(j_min, j_max + 1):
                    if j < len(self.c) and _is_zero(self.c[j]):
                        continue
                    if k - j < len(other.c) and _is_zero(other.c[k - j]):
                        continue
                    aj = self.coeff(j)
                    if _is_zero(aj):
                        continue
                    bj = other.coeff(k - j)
                    if _is_zero(bj):
                        continue
                    terms.append(aj * bj)  # type: ignore[operator]
                return _sum(terms, zero=c0 * 0)  # type: ignore[operator]

            return Taylor(
                [c0],
                _rule=mul_rule,
            )

        def skew_rule(k: int) -> T:
            terms: list[T] = []
            j_max = k if self._rule is not None else min(k, len(self.c) - 1)
            for j in range(j_max + 1):
                if j < len(self.c) and _is_zero(self.c[j]):
                    continue
                if j == k and _is_zero(other.c[0]):
                    continue
                aj = self.coeff(j)
                if _is_zero(aj):
                    continue
                m_min = 0 if other._rule is not None else max(0, k - j - len(other.c) + 1)
                for m in range(m_min, k - j + 1):
                    binom = ((-1) ** m) * math.comb(j + m - 1, m) if j > 0 else (1 if m == 0 else 0)
                    if binom != 0:
                        idx_b = k - j - m
                        if idx_b < len(other.c) and _is_zero(other.c[idx_b]):
                            continue
                        bm = other.coeff(idx_b)
                        if _is_zero(bm):
                            continue
                        for _ in range(m):
                            bm = delta(bm)
                        if _is_zero(bm):
                            continue
                        terms.append(aj * bm * binom)  # type: ignore[operator]
            return _sum(terms, zero=c0 * 0)  # type: ignore[operator]

        return Taylor(
            [c0],
            _rule=skew_rule,
            derivation=delta,
        )

    def __rmul__(self, other: Union[T, int]) -> "Taylor[T]":
        return Taylor(
            [other * self.c[0]],  # type: ignore[operator]
            _rule=lambda k: other * self.coeff(k),  # type: ignore[operator]
            derivation=self._derivation,
        )

    def rdiv(
        self,
        other: Union["Taylor[T]", T, int],
        inv_b0: Optional[Callable[[T], T]] = None,
    ) -> "Taylor[T]":
        """Right division ``A * B^{-1}`` (solves ``C * B = A``)."""
        if not isinstance(other, Taylor):
            if inv_b0 is None and _is_zero(other):
                raise ZeroDivisionError("division of Taylor by zero")
            return Taylor(
                [self.c[0] / other],  # type: ignore[operator]
                _rule=lambda k: self.coeff(k) / other,  # type: ignore[operator]
                derivation=self._derivation,
            )
        delta = _merge_derivation(self._derivation, other._derivation)
        b0 = other.coeff(0)
        if inv_b0 is None and _is_zero(b0):
            raise ZeroDivisionError("division by Taylor with zero constant term")
        inv: Callable[[T], T] = (lambda x: x / b0) if inv_b0 is None else inv_b0  # type: ignore[operator]
        out: Taylor[T] = Taylor([inv(self.coeff(0))], derivation=delta)
        return out.bind_recurrence(
            lambda k: inv(
                self.coeff(k)
                - _sum(
                    (out.coeff(k - j) * other.coeff(j) for j in range(1, k + 1)),  # type: ignore[misc]
                    zero=self.zero_elem,
                )
            )
        )

    def ldiv(
        self,
        other: "Taylor[T]",
        inv_b0: Optional[Callable[[T], T]] = None,
    ) -> "Taylor[T]":
        """Left division ``B^{-1} * A`` (solves ``B * C = A``, where ``self`` is ``B``)."""
        delta = _merge_derivation(self._derivation, other._derivation)
        b0 = self.coeff(0)
        if inv_b0 is None and _is_zero(b0):
            raise ZeroDivisionError("division by Taylor with zero constant term")
        inv: Callable[[T], T] = (lambda x: x / b0) if inv_b0 is None else inv_b0  # type: ignore[operator]
        out: Taylor[T] = Taylor([inv(other.coeff(0))], derivation=delta)
        return out.bind_recurrence(
            lambda k: inv(
                other.coeff(k)
                - _sum(
                    (self.coeff(j) * out.coeff(k - j) for j in range(1, k + 1)),  # type: ignore[misc]
                    zero=self.zero_elem,
                )
            )
        )

    __truediv__ = rdiv

    def __rtruediv__(self, other: Union[T, int]) -> "Taylor[T]":
        return Taylor([other], derivation=self._derivation).rdiv(self)  # type: ignore[list-item]

    def shift(self, m: int) -> "Taylor[T]":
        """Multiply by ``h^m`` (``m > 0``) or divide by ``h^{-m}`` (``m < 0``, L'Hopital shift)."""
        return Taylor(
            [self.coeff(-m)],
            _rule=lambda k: self.coeff(k - m),
            derivation=self._derivation,
        )

    def __pow__(self, p):
        if isinstance(p, int):
            if p == 0:
                return Taylor([self.c[0] ** 0], derivation=self._derivation)  # type: ignore[operator]
            if p < 0:
                return Taylor([self.c[0] ** 0], derivation=self._derivation).rdiv(self ** (-p))  # type: ignore[operator]
            res: Optional[Taylor[T]] = None
            base: Taylor[T] = self
            e = p
            while e:
                if e & 1:
                    res = base if res is None else res * base
                base = base * base
                e >>= 1
            assert res is not None
            return res
        # Non-integer / Taylor exponents are functions, not ring operations:
        # they live in taylorize.functions (deferred import, no cycle).
        from .functions import power

        return power(self, p)

    def compose(self, g: "Taylor[U]") -> "Taylor[U]":
        """Formal composition ``f(g(h))`` for any ``g`` with ``g(0) = 0``."""
        unit_u: U = g.c[0] * 0 + 1  # type: ignore[operator]
        powers: list[Taylor[U]] = [Taylor([unit_u], derivation=g._derivation), g]

        def g_pow(j: int) -> "Taylor[U]":
            while len(powers) <= j:
                powers.append(powers[-1] * g)
            return powers[j]

        return Taylor(
            [self.coeff(0) * unit_u],  # type: ignore[operator]
            _rule=lambda k: _sum(
                (self.coeff(j) * g_pow(j).coeff(k) for j in range(1, k + 1)),  # type: ignore[misc]
                zero=g.zero_elem,
            ),
            derivation=g._derivation,
        )

    # =========================================================================
    # LEVEL 1: Algebraic / Implicit Extensions P(h, Y) = 0
    # =========================================================================
    @classmethod
    def solve_algebraic(
        cls,
        residual_fn: Callable[["Taylor[T]"], "Taylor[T]"],
        y0: T,
        inv_jacobian: Callable[[T], T],
        derivation: Optional[Derivation[T]] = None,
    ) -> "Taylor[T]":
        """Solve ``P(h, Y) = 0`` given ``y0`` and ``inv_jacobian = (dP/dY(0, y0))^{-1}``."""
        y: Taylor[T] = cls.pending(y0, derivation=derivation)

        def rule(k: int) -> T:
            y_trial = cls(y.c + [y.zero_elem], derivation=derivation)
            defect = residual_fn(y_trial).coeff(k)
            return -inv_jacobian(defect)  # type: ignore[operator]

        return y.bind_recurrence(rule)

    # =========================================================================
    # LEVEL 2: Liouville Extensions (diff, integrate, non-commutative exp_int)
    # =========================================================================
    def diff(self) -> "Taylor[T]":
        """Formal derivative ``x'(h)`` as a new lazy Taylor."""
        return Taylor(
            [self.coeff(1)],
            _rule=lambda k: (k + 1) * self.coeff(k + 1),  # type: ignore[operator]
            derivation=self._derivation,
        )

    def euler_diff(self) -> "Taylor[T]":
        """Euler derivative ``theta(x) = h * x'(h)`` as a new lazy Taylor."""
        return Taylor(
            [self.zero_elem],
            _rule=lambda k: k * self.coeff(k),  # type: ignore[operator]
            derivation=self._derivation,
        )

    def integrate(self, c0: Optional[T] = None) -> "Taylor[T]":
        """Formal integral with constant term ``c0`` (Picard operator)."""
        init = self.zero_elem if c0 is None else c0
        return Taylor.pending(init, derivation=self._derivation).bind_ode(self)

    def exp_integrate_left(self, y0: T) -> "Taylor[T]":
        """Solve ``Y' = self * Y`` with ``Y(0) = y0``."""
        y: Taylor[T] = Taylor.pending(y0, derivation=self._derivation)
        return y.bind_ode(self * y)

    def exp_integrate_right(self, y0: T) -> "Taylor[T]":
        """Solve ``Y' = Y * self`` with ``Y(0) = y0``."""
        y: Taylor[T] = Taylor.pending(y0, derivation=self._derivation)
        return y.bind_ode(y * self)

    def log_diff_left(self) -> "Taylor[T]":
        """Left logarithmic derivative ``Y' * Y^{-1}``."""
        return self.diff().rdiv(self)

    def log_diff_right(self) -> "Taylor[T]":
        """Right logarithmic derivative ``Y^{-1} * Y'``."""
        return self.ldiv(self.diff())

    # =========================================================================
    # LEVEL 3: Holonomic / Fuchsian Regular Singular Equations I(theta) Y = RHS
    # =========================================================================
    def bind_euler_ode(
        self,
        indicial: Callable[[int], Union[T, int]],
        rhs: "Taylor[T]",
        inv_indicial: Optional[Callable[[int, T], T]] = None,
    ) -> "Taylor[T]":
        """Bind a Fuchsian regular-singular ODE ``I(theta) Y = rhs``."""
        solve_k: Callable[[int, T], T] = (
            (lambda k, r: r / indicial(k)) if inv_indicial is None else inv_indicial  # type: ignore[operator]
        )
        return self.bind_recurrence(lambda k: solve_k(k, rhs.coeff(k)))

    # =========================================================================
    # LEVEL 4: Differentially Algebraic & High-Order Non-Commutative ODEs
    # =========================================================================
    def bind_ode(self, dy: "Taylor[T]", order: int = 1) -> "Taylor[T]":
        """Wire the ``order``-th order ODE ``y^(order) = dy`` to this series."""
        if order < 1:
            raise ValueError("order must be >= 1")
        if len(self.c) > order:
            raise RuntimeError(
                "bind_ode() cannot re-wire a Taylor after higher coefficients are materialized."
            )
        if len(self.c) < order:
            raise ValueError(
                f"bind_ode(order={order}) requires at least {order} initial coefficient(s); "
                f"got {len(self.c)}."
            )
        return self.bind_recurrence(
            lambda k: dy.coeff(k - order) / math.perm(k, order)  # type: ignore[operator]
        )


def variable(
    x0: T,
    order: Optional[int] = None,
    unit: Optional[T] = None,
) -> Taylor[T]:
    """The independent variable, ``t = x0 + unit * h``."""
    u: T = (x0 * 0 + 1) if unit is None else unit  # type: ignore[operator]
    t: Taylor[T] = Taylor([x0, u] if order != 0 else [x0])
    if order:
        t.ensure(order)
    return t


def constant(c: T, order: Optional[int] = None) -> Taylor[T]:
    t: Taylor[T] = Taylor([c])
    if order:
        t.ensure(order)
    return t
