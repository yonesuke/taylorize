"""Tests for Levels 0-4 of the non-commutative / skew Taylor core."""
from fractions import Fraction

import pytest

from taylorize import Taylor, constant, variable
from taylorize.functions import bessel_j0, bessel_j01, painleve_I


# ---------------------------------------------------------------------------
# Minimal free associative algebra over Fraction for non-commutative tests
# ---------------------------------------------------------------------------
class NCExpr:
    """Element of the free associative Q-algebra Q<X, Y, ...>."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict[tuple[str, ...], Fraction] | None = None) -> None:
        self.terms: dict[tuple[str, ...], Fraction] = {
            w: c for w, c in (terms or {}).items() if c != 0
        }

    @classmethod
    def gen(cls, name: str) -> "NCExpr":
        return cls({(name,): Fraction(1)})

    @classmethod
    def scalar(cls, val: int | Fraction) -> "NCExpr":
        return cls({(): Fraction(val)})

    def _coerce(self, other) -> "NCExpr":
        if isinstance(other, NCExpr):
            return other
        if isinstance(other, (int, Fraction)):
            return NCExpr.scalar(other)
        return NotImplemented

    def __add__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        out = dict(self.terms)
        for w, c in o.terms.items():
            out[w] = out.get(w, Fraction(0)) + c
        return NCExpr(out)

    __radd__ = __add__

    def __neg__(self) -> "NCExpr":
        return NCExpr({w: -c for w, c in self.terms.items()})

    def __sub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else self + (-o)

    def __rsub__(self, other):
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else o + (-self)

    def __mul__(self, other):
        if isinstance(other, (int, Fraction)):
            return NCExpr({w: c * other for w, c in self.terms.items()})
        if not isinstance(other, NCExpr):
            return NotImplemented
        out: dict[tuple[str, ...], Fraction] = {}
        for w1, c1 in self.terms.items():
            for w2, c2 in other.terms.items():
                w = w1 + w2
                out[w] = out.get(w, Fraction(0)) + c1 * c2
        return NCExpr(out)

    def __rmul__(self, other):
        if isinstance(other, (int, Fraction)):
            return NCExpr({w: other * c for w, c in self.terms.items()})
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            return NCExpr({w: c / other for w, c in self.terms.items()})
        if isinstance(other, NCExpr) and set(other.terms.keys()) <= {()}:
            c0 = other.terms.get((), Fraction(0))
            return NCExpr({w: c / c0 for w, c in self.terms.items()})
        return NotImplemented

    def __pow__(self, p: int) -> "NCExpr":
        if p == 0:
            return NCExpr.scalar(1)
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, Fraction)):
            return self == NCExpr.scalar(other)
        return isinstance(other, NCExpr) and self.terms == other.terms

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        return " + ".join(
            f"({c})*{''.join(w) if w else '1'}" for w, c in sorted(self.terms.items())
        )


# ---------------------------------------------------------------------------
# Level 0: Non-commutative ring, rdiv/ldiv, shift, compose, skew derivation
# ---------------------------------------------------------------------------
def test_level0_noncommutative_division_and_shift():
    X, Y = NCExpr.gen("X"), NCExpr.gen("Y")
    one = NCExpr.scalar(1)

    A = Taylor([one, X, Y])
    B = Taylor([one, Y, X])

    # Right division: (A / B) * B == A
    q_right = A.rdiv(B)
    prod_right = q_right * B
    assert [prod_right.coeff(k) for k in range(5)] == [A.coeff(k) for k in range(5)]

    # Left division: B * B.ldiv(A) == A
    q_left = B.ldiv(A)
    prod_left = B * q_left
    assert [prod_left.coeff(k) for k in range(5)] == [A.coeff(k) for k in range(5)]

    # Distinctness of left vs right division in non-commutative ring
    assert q_right.coeff(2) != q_left.coeff(2)

    # Shift (multiply by h^m and L'Hopital divide by h^m)
    s = Taylor([Fraction(0), Fraction(0), Fraction(3), Fraction(5)])
    assert [s.shift(-2).coeff(k) for k in range(3)] == [Fraction(3), Fraction(5), Fraction(0)]
    assert [s.shift(1).coeff(k) for k in range(4)] == [Fraction(0), Fraction(0), Fraction(0), Fraction(3)]


def test_level0_mismatched_derivation_raises():
    a = Taylor([1.0], derivation=lambda x: x)
    b = Taylor([2.0], derivation=lambda x: 2 * x)
    with pytest.raises(ValueError):
        _ = a + b
    with pytest.raises(ValueError):
        _ = a * b


# ---------------------------------------------------------------------------
# Level 1: Algebraic / implicit equations P(h, Y) = 0
# ---------------------------------------------------------------------------
def test_level1_catalan_numbers():
    # C(h) = 1 + h * C(h)^2  =>  P(h, Y) = Y - 1 - h * Y^2 = 0, dP/dY(0, 1) = 1
    h = variable(Fraction(0))
    catalan = Taylor.solve_algebraic(
        lambda y: y - Fraction(1) - h * (y * y),
        y0=Fraction(1),
        inv_jacobian=lambda d: d,
    )
    assert [catalan.coeff(k) for k in range(7)] == [
        Fraction(1),
        Fraction(1),
        Fraction(2),
        Fraction(5),
        Fraction(14),
        Fraction(42),
        Fraction(132),
    ]


def test_level1_skew_kdv_pseudodifferential_operator():
    sp = pytest.importorskip("sympy")
    x = sp.Symbol("x")
    u = sp.Function("u")(x)
    deriv = lambda expr: sp.diff(expr, x)

    # Operator L = 1 + d^{-2} * u, solve Y^2 = L in skew ring where h = d^{-1}
    d_inv2 = Taylor([sp.Integer(0), sp.Integer(0), sp.Integer(1)], derivation=deriv)
    L = Taylor([sp.Integer(1)], derivation=deriv) + d_inv2 * Taylor([u], derivation=deriv)
    Y = Taylor.solve_algebraic(
        lambda y: y * y - L,
        y0=sp.Integer(1),
        inv_jacobian=lambda d: d / 2,
        derivation=deriv,
    )
    expected = [
        sp.Integer(1),
        sp.Integer(0),
        u / 2,
        -sp.diff(u, x),
        -(u**2) / 8 + sp.Rational(3, 2) * sp.diff(u, x, 2),
        sp.Rational(3, 4) * u * sp.diff(u, x) - 2 * sp.diff(u, x, 3),
    ]
    residual = Y * Y - L
    for k, exp_k in enumerate(expected):
        assert sp.simplify(Y.coeff(k) - exp_k) == 0
        assert sp.simplify(residual.coeff(k)) == 0


# ---------------------------------------------------------------------------
# Level 2: Non-commutative Dyson & Baker-Campbell-Hausdorff
# ---------------------------------------------------------------------------
def test_level2_dyson_and_bch():
    A, B = NCExpr.gen("A"), NCExpr.gen("B")
    one = NCExpr.scalar(1)

    # Dyson / Mould expansion: Y' = (A + h B) * Y, Y(0) = 1
    Y = Taylor([A, B]).exp_integrate_left(one)
    assert Y.coeff(0) == one
    assert Y.coeff(1) == A
    assert Y.coeff(2) == (A * A) / 2 + B / 2
    assert Y.coeff(3) == (A * A * A) / 6 + (A * B) / 6 + (B * A) / 3

    # Exact Baker-Campbell-Hausdorff: Z(h) = log(exp(hX) exp(hY)) via formal compose
    X, Y_gen = NCExpr.gen("X"), NCExpr.gen("Y")
    exp_hX = Taylor([X]).exp_integrate_left(one)
    exp_hY = Taylor([Y_gen]).exp_integrate_left(one)
    P = exp_hX * exp_hY
    log1p = Taylor([Fraction(0)]).bind_recurrence(
        lambda k: Fraction((-1) ** (k - 1), k)
    )
    Z = log1p.compose(P - one)

    assert Z.coeff(1) == X + Y_gen
    assert Z.coeff(2) == (X * Y_gen - Y_gen * X) / 2
    assert Z.coeff(3) == (
        X * X * Y_gen
        + X * Y_gen * Y_gen
        + Y_gen * X * X
        + Y_gen * Y_gen * X
        - 2 * X * Y_gen * X
        - 2 * Y_gen * X * Y_gen
    ) / 12

    # Non-commutative logarithmic derivatives: (P' * P^{-1}) * P == P' and P * (P^{-1} * P') == P'
    ld_left = P.log_diff_left() * P
    ld_right = P * P.log_diff_right()
    dp = P.diff()
    for k in range(4):
        assert ld_left.coeff(k) == dp.coeff(k)
        assert ld_right.coeff(k) == dp.coeff(k)


# ---------------------------------------------------------------------------
# Level 3: Holonomic / Fuchsian singular equations
# ---------------------------------------------------------------------------
def test_level3_bessel_j0_at_singular_origin():
    j0 = bessel_j0(variable(Fraction(0)))
    assert [j0.coeff(k) for k in range(7)] == [
        Fraction(1),
        Fraction(0),
        Fraction(-1, 4),
        Fraction(0),
        Fraction(1, 64),
        Fraction(0),
        Fraction(-1, 2304),
    ]

    # Composed argument g(t) = 2t with g(0) = 0 also works at origin
    j0_2t, j1_2t = bessel_j01(2 * variable(Fraction(0)))
    assert j0_2t.coeff(2) == Fraction(-1)
    assert j1_2t.coeff(1) == Fraction(1)

    mp = pytest.importorskip("mpmath").mp
    mp.dps = 30
    val = bessel_j0(2 * variable(mp.mpf(0))).eval(mp.mpf("0.25"), order=30)
    assert abs(val - mp.besselj(0, mp.mpf("0.5"))) < mp.mpf("1e-25")


# ---------------------------------------------------------------------------
# Level 4: High-order ODEs (direct bind_ode without auxiliary state)
# ---------------------------------------------------------------------------
def test_level4_high_order_odes():
    # Direct 2nd-order Painleve I with chain rule on composed variable
    x = 2 * variable(Fraction(0))
    y = painleve_I(x, Fraction(1), Fraction(0))
    # y(2t) = 1 + 3*(2t)^2 + (1/6)*(2t)^3 + 3*(2t)^4 + ...
    assert [y.coeff(k) for k in range(5)] == [
        Fraction(1),
        Fraction(0),
        Fraction(12),
        Fraction(4, 3),
        Fraction(48),
    ]

    # Direct 3rd-order ODE: y''' = y, y(0)=1, y'(0)=1, y''(0)=1 => c = [1, 1, 1/2, ...]
    y3 = Taylor.pending(Fraction(1), Fraction(1), Fraction(1, 2))
    y3.bind_ode(y3, order=3)
    assert [y3.coeff(k) for k in range(6)] == [
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 6),
        Fraction(1, 24),
        Fraction(1, 120),
    ]
