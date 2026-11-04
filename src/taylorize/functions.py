"""Functions, defined by the differential equations they satisfy.

Every function in this module is a *declaration*, not a recurrence:
the constructors :func:`define_ode_function` and
:func:`define_ode_system` take the derivative law ``dy/dx = rhs(x, y)``
plus a way to obtain the value at the expansion point (from
:mod:`taylorize.backends`), and wire it into the core with
``Taylor.bind_ode``.  Adding a function is a few lines here -- the core
never changes.

Conventions and known limits:

- ``rhs`` receives the *pending* output series; it may use its primal
  and combine it algebraically, but reading higher coefficients during
  construction raises (the core's pending guard), which catches wiring
  mistakes instead of silently using zeros.
- Systems whose derivative law divides by the variable itself
  (Bessel's ``J1/x``) or by the state (``sqrt`` at 0) are singular at
  the origin and require a nonzero expansion point.  Treating those
  points exactly needs the implicit/holonomic layer, which this module
  deliberately does not fake.
"""
from __future__ import annotations

from .backends import backend_constant, eval_initial, special_initial
from .core import Taylor

__all__ = [
    "define_ode_function",
    "define_ode_system",
    "FUNCTIONS",
    "reciprocal",
    "exp",
    "log",
    "sin",
    "cos",
    "sin_cos",
    "sinh",
    "cosh",
    "sinh_cosh",
    "tan",
    "tanh",
    "atan",
    "asin",
    "sqrt",
    "power",
    "erf",
    "airy",
    "airy_ai",
    "airy_bi",
    "bessel_j01",
    "jacobi_elliptic",
    "weierstrass_p",
    "painleve_I",
    "painleve_II",
]


# -- the two constructors -------------------------------------------------------
def define_ode_function(rhs, init):
    """Define ``y(x)`` by ``dy/dx = rhs(x, y)`` and ``y(x0) = init(x0)``.

    ``rhs`` is an expression in the jets ``x`` and ``y`` built from core
    ring operations (and other functions).  The returned callable maps a
    scalar to a scalar (via ``init``'s backend) and a Taylor to a Taylor.
    """

    def func(x):
        if not isinstance(x, Taylor):
            return init(x)
        y = Taylor.pending(init(x.primal))
        y.bind_ode(rhs(x, y) * x.diff())
        return y

    return func


def define_ode_system(rhs_system, init_system):
    """Define a coupled system ``y_i(x)`` by ``dy_i/dx = rhs_i(x, ys)``
    and ``ys(x0) = init_system(x0)`` (a sequence of scalars)."""

    def func(x, *params):
        if not isinstance(x, Taylor):
            return init_system(x, *params)
        ys = [Taylor.pending(v) for v in init_system(x.primal, *params)]
        dx = x.diff()
        for y, dy in zip(ys, rhs_system(x, ys, *params)):
            y.bind_ode(dy * dx)
        return ys

    return func


def _named(func, name):
    func.__name__ = name
    return func


# -- elementary functions ---------------------------------------------------------
reciprocal = _named(
    define_ode_function(lambda x, y: -y * y, lambda x0: 1 / x0), "reciprocal"
)
exp = _named(define_ode_function(lambda x, y: y, lambda x0: eval_initial("exp", x0)), "exp")
log = _named(define_ode_function(lambda x, y: 1 / x, lambda x0: eval_initial("log", x0)), "log")

sin_cos = _named(
    define_ode_system(
        lambda x, ys: [ys[1], -ys[0]],
        lambda x0: [eval_initial("sin", x0), eval_initial("cos", x0)],
    ),
    "sin_cos",
)


def sin(x):
    if not isinstance(x, Taylor):
        return eval_initial("sin", x)
    return sin_cos(x)[0]


def cos(x):
    if not isinstance(x, Taylor):
        return eval_initial("cos", x)
    return sin_cos(x)[1]


sinh_cosh = _named(
    define_ode_system(
        lambda x, ys: [ys[1], ys[0]],
        lambda x0: [eval_initial("sinh", x0), eval_initial("cosh", x0)],
    ),
    "sinh_cosh",
)


def sinh(x):
    if not isinstance(x, Taylor):
        return eval_initial("sinh", x)
    return sinh_cosh(x)[0]


def cosh(x):
    if not isinstance(x, Taylor):
        return eval_initial("cosh", x)
    return sinh_cosh(x)[1]


tan = _named(define_ode_function(lambda x, y: 1 + y * y, lambda x0: eval_initial("tan", x0)), "tan")
tanh = _named(
    define_ode_function(lambda x, y: 1 - y * y, lambda x0: eval_initial("tanh", x0)), "tanh"
)
atan = _named(
    define_ode_function(lambda x, y: 1 / (1 + x * x), lambda x0: eval_initial("atan", x0)), "atan"
)
sqrt = _named(
    define_ode_function(lambda x, y: 1 / (2 * y), lambda x0: eval_initial("sqrt", x0)), "sqrt"
)
asin = _named(
    define_ode_function(
        lambda x, y: 1 / sqrt(1 - x * x), lambda x0: eval_initial("asin", x0)
    ),
    "asin",
)


def power(x, p):
    """``x ** p`` for scalar or Taylor ``p`` (domain: ``x0 != 0``)."""
    if isinstance(p, Taylor):
        return exp(p * log(x))
    if isinstance(p, int):
        return x ** p
    if not isinstance(x, Taylor):
        return x ** p
    y = Taylor.pending(x.primal ** p)
    y.bind_ode((p * y / x) * x.diff())
    return y


# -- special functions ------------------------------------------------------------
def erf(x):
    if not isinstance(x, Taylor):
        return eval_initial("erf", x)
    c = 2 / eval_initial("sqrt", backend_constant("pi", x.primal))
    y = Taylor.pending(eval_initial("erf", x.primal))
    y.bind_ode((c * exp(-(x * x))) * x.diff())
    return y


airy = _named(
    define_ode_system(
        lambda x, ys: [ys[1], x * ys[0]],
        lambda x0: [special_initial("airyai", x0), special_initial("airyaiprime", x0)],
    ),
    "airy",
)


def airy_ai(x):
    if not isinstance(x, Taylor):
        return special_initial("airyai", x)
    return airy(x)[0]


airy_bi_system = _named(
    define_ode_system(
        lambda x, ys: [ys[1], x * ys[0]],
        lambda x0: [special_initial("airybi", x0), special_initial("airybiprime", x0)],
    ),
    "airy_bi_system",
)


def airy_bi(x):
    if not isinstance(x, Taylor):
        return special_initial("airybi", x)
    return airy_bi_system(x)[0]


bessel_j01 = _named(
    define_ode_system(
        lambda x, ys: [-ys[1], ys[0] - ys[1] / x],
        lambda x0: [special_initial("besselj", x0, 0), special_initial("besselj", x0, 1)],
    ),
    "bessel_j01",
)

jacobi_elliptic = _named(
    define_ode_system(
        lambda x, ys, m: [ys[1] * ys[2], -ys[0] * ys[2], -m * ys[0] * ys[1]],
        lambda x0, m: list(special_initial("ellip_sn_cn_dn", x0, m)),
    ),
    "jacobi_elliptic",
)


def weierstrass_p(x, u0, v0, g2):
    """Weierstrass ℘ as a system with *explicit* initial values at ``x0``:
    ``u' = v``, ``v' = 6u^2 - g2/2``.  Supplying (u0, v0) is the honest
    interface for a function with poles and no canonical value at 0."""
    if not isinstance(x, Taylor):
        return (u0, v0)
    u = Taylor.pending(u0)
    v = Taylor.pending(v0)
    dx = x.diff()
    u.bind_ode(v * dx)
    v.bind_ode((6 * u * u - g2 / 2) * dx)
    return u, v


def painleve_I(x, y0, dy0):
    """P_I: ``y'' = 6y^2 + x``, with explicit initial values at ``x0``."""
    if not isinstance(x, Taylor):
        return (y0, dy0)
    y = Taylor.pending(y0)
    v = Taylor.pending(dy0)
    dx = x.diff()
    y.bind_ode(v * dx)
    v.bind_ode((6 * y * y + x) * dx)
    return y, v


def painleve_II(x, y0, dy0, alpha=0):
    """P_II: ``y'' = 2y^3 + x y + alpha``, explicit initial values."""
    if not isinstance(x, Taylor):
        return (y0, dy0)
    y = Taylor.pending(y0)
    v = Taylor.pending(dy0)
    dx = x.diff()
    y.bind_ode(v * dx)
    v.bind_ode((2 * y * y * y + x * y + alpha) * dx)
    return y, v


# -- registry + host-library dispatch ----------------------------------------------
FUNCTIONS = {
    "reciprocal": reciprocal,
    "exp": exp,
    "log": log,
    "sin": sin,
    "cos": cos,
    "sin_cos": sin_cos,
    "sinh": sinh,
    "cosh": cosh,
    "sinh_cosh": sinh_cosh,
    "tan": tan,
    "tanh": tanh,
    "atan": atan,
    "asin": asin,
    "sqrt": sqrt,
    "power": power,
    "erf": erf,
    "airy": airy,
    "airy_ai": airy_ai,
    "airy_bi": airy_bi,
    "bessel_j01": bessel_j01,
    "jacobi_elliptic": jacobi_elliptic,
    "weierstrass_p": weierstrass_p,
    "painleve_I": painleve_I,
    "painleve_II": painleve_II,
}
