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

import math

from .backends import backend_constant, eval_initial, special_initial
from .core import Taylor

# -- the constructors -----------------------------------------------------------
def define_ode_function(rhs, init, name: str | None = None):
    """Define ``y(x)`` by ``dy/dx = rhs(x, y, *params)`` and ``y(x0) = init(x0, *params)``.

    ``rhs`` is an expression in the jets ``x`` and ``y`` built from core
    ring operations (and other functions).  ``init`` may be a callable or a
    backend function name.  The returned callable maps a scalar to a scalar
    (via ``init``'s backend) and a Taylor to a Taylor.
    """
    if isinstance(init, str):
        init_name = init
        name = name or init_name
        init = lambda x0, *params: eval_initial(init_name, x0, *params)

    def func(x, *params):
        if not isinstance(x, Taylor):
            return init(x, *params)
        y = Taylor.pending(init(x.primal, *params))
        return y.bind_ode(rhs(x, y, *params) * x.diff())

    if name:
        func.__name__ = name
    return func


def define_ode_system(rhs_system, init_system, name: str | None = None):
    """Define a coupled system ``y_i(x)`` by ``dy_i/dx = rhs_i(x, ys, *params)``
    and ``ys(x0) = init_system(x0, *params)`` (a sequence of scalars)."""

    def func(x, *params):
        if not isinstance(x, Taylor):
            return init_system(x, *params)
        ys = [Taylor.pending(v) for v in init_system(x.primal, *params)]
        dx = x.diff()
        for y, dy in zip(ys, rhs_system(x, ys, *params)):
            y.bind_ode(dy * dx)
        return ys

    if name:
        func.__name__ = name
    return func


def _project(system, idx: int, init, name: str):
    """Project component ``idx`` from a coupled ODE system with scalar shortcut."""
    if isinstance(init, str):
        init_name = init
        init = lambda x0, *params: eval_initial(init_name, x0, *params)

    def func(x, *params):
        return system(x, *params)[idx] if isinstance(x, Taylor) else init(x, *params)

    func.__name__ = name
    return func


def _ivp2(x, y0, dy0, accel):
    """Wire a second-order ODE ``y''(x) = accel(x, y)`` with explicit ``(y0, dy0)``.

    Uses Level-4 ``bind_ode(..., order=2)`` directly with no auxiliary state.
    """
    if not isinstance(x, Taylor):
        return y0
    y = Taylor.pending(y0, dy0)
    dx = x.diff()
    ddx = dx.diff()
    return y.bind_ode(accel(x, y) * (dx * dx) + y.diff() * ddx, order=2)


# -- elementary functions ---------------------------------------------------------
reciprocal = define_ode_function(lambda x, y: -y * y, lambda x0: 1 / x0, "reciprocal")
exp = define_ode_function(lambda x, y: y, "exp")
expm1 = define_ode_function(lambda x, y: 1 + y, "expm1")
log = define_ode_function(lambda x, y: 1 / x, "log")
log1p = define_ode_function(lambda x, y: 1 / (1 + x), "log1p")

sin_cos = define_ode_system(
    lambda x, ys: [ys[1], -ys[0]],
    lambda x0: [eval_initial("sin", x0), eval_initial("cos", x0)],
    "sin_cos",
)
sin = _project(sin_cos, 0, "sin", "sin")
cos = _project(sin_cos, 1, "cos", "cos")

sinh_cosh = define_ode_system(
    lambda x, ys: [ys[1], ys[0]],
    lambda x0: [eval_initial("sinh", x0), eval_initial("cosh", x0)],
    "sinh_cosh",
)
sinh = _project(sinh_cosh, 0, "sinh", "sinh")
cosh = _project(sinh_cosh, 1, "cosh", "cosh")

tan = define_ode_function(lambda x, y: 1 + y * y, "tan")
tanh = define_ode_function(lambda x, y: 1 - y * y, "tanh")
atan = define_ode_function(lambda x, y: 1 / (1 + x * x), "atan")
atanh = define_ode_function(lambda x, y: 1 / (1 - x * x), "atanh")
sqrt = define_ode_function(lambda x, y: 1 / (2 * y), "sqrt")
asin = define_ode_function(lambda x, y: 1 / sqrt(1 - x * x), "asin")
acos = define_ode_function(lambda x, y: -1 / sqrt(1 - x * x), "acos")
asinh = define_ode_function(lambda x, y: 1 / sqrt(1 + x * x), "asinh")
acosh = define_ode_function(lambda x, y: 1 / sqrt(x * x - 1), "acosh")

_scalar_power = define_ode_function(lambda x, y, p: p * y / x, lambda x0, p: x0 ** p, "power")


def power(x, p):
    """``x ** p`` for scalar or Taylor ``p`` (domain: ``x0 != 0``)."""
    if isinstance(p, Taylor):
        return exp(p * log(x))
    if isinstance(p, int):
        return x ** p
    return _scalar_power(x, p)


# -- special functions ------------------------------------------------------------
erf = define_ode_function(
    lambda x, y: (2 / eval_initial("sqrt", backend_constant("pi", x.primal))) * exp(-(x * x)),
    "erf",
)
erfc = define_ode_function(
    lambda x, y: -(2 / eval_initial("sqrt", backend_constant("pi", x.primal))) * exp(-(x * x)),
    "erfc",
)
erfi = define_ode_function(
    lambda x, y: (2 / eval_initial("sqrt", backend_constant("pi", x.primal))) * exp(x * x),
    lambda x0: special_initial("erfi", x0),
    "erfi",
)
dawsn = define_ode_function(
    lambda x, y: 1 - 2 * x * y,
    lambda x0: special_initial("dawsn", x0),
    "dawsn",
)

fresnel = define_ode_system(
    lambda x, ys: [
        sin((backend_constant("pi", x.primal) / 2) * (x * x)),
        cos((backend_constant("pi", x.primal) / 2) * (x * x)),
    ],
    lambda x0: list(special_initial("fresnel", x0)),
    "fresnel",
)
fresnel_s = _project(fresnel, 0, lambda x0: special_initial("fresnel", x0)[0], "fresnel_s")
fresnel_c = _project(fresnel, 1, lambda x0: special_initial("fresnel", x0)[1], "fresnel_c")

ellip_ke = define_ode_system(
    lambda m, ys: [
        (ys[1] - (1 - m) * ys[0]) / (2 * m * (1 - m)),
        (ys[1] - ys[0]) / (2 * m),
    ],
    lambda m0: list(special_initial("ellip_ke", m0)),
    "ellip_ke",
)
ellipk = _project(ellip_ke, 0, lambda m0: special_initial("ellip_ke", m0)[0], "ellipk")
ellipe = _project(ellip_ke, 1, lambda m0: special_initial("ellip_ke", m0)[1], "ellipe")

si = define_ode_function(lambda x, y: sin(x) / x, lambda x0: special_initial("si", x0), "si")
ci = define_ode_function(lambda x, y: cos(x) / x, lambda x0: special_initial("ci", x0), "ci")


def sici(x):
    """Sine and cosine integrals ``(Si(x), Ci(x))`` (domain: ``x0 != 0``)."""
    return (si(x), ci(x))


dilog = define_ode_function(
    lambda x, y: -log(1 - x) / x,
    lambda x0: special_initial("dilog", x0),
    "dilog",
)


def spence(x):
    """SciPy-convention dilogarithm ``spence(x) = Li_2(1 - x)``."""
    return dilog(1 - x)


airy = define_ode_system(
    lambda x, ys: [ys[1], x * ys[0]],
    lambda x0: [special_initial("airyai", x0), special_initial("airyaiprime", x0)],
    "airy",
)
airy_ai = _project(airy, 0, lambda x0: special_initial("airyai", x0), "airy_ai")

airy_bi_system = define_ode_system(
    lambda x, ys: [ys[1], x * ys[0]],
    lambda x0: [special_initial("airybi", x0), special_initial("airybiprime", x0)],
    "airy_bi_system",
)
airy_bi = _project(airy_bi_system, 0, lambda x0: special_initial("airybi", x0), "airy_bi")


def _bessel_j0_at_zero(zero_like):
    """Fuchsian expansion of J_0(u) around the regular singular point u = 0:
    ``theta^2 J_0 = -u^2 J_0``."""
    one = zero_like * 0 + 1
    y = Taylor.pending(one)
    return y.bind_euler_ode(lambda k: k * k, -(y.shift(2)))


def _bessel_jn_at_zero(n: int, zero_like):
    """Fuchsian expansion of integer-order ``J_n(u)`` at ``u = 0``."""
    if n == 0:
        return _bessel_j0_at_zero(zero_like)
    one = zero_like * 0 + 1
    zero = zero_like * 0
    lead = one / ((1 << n) * math.factorial(n))
    y = Taylor.pending(*([zero] * n + [lead]))
    return y.bind_euler_ode(lambda k: k * k - n * n, -(y.shift(2)))


_bessel_j01_regular = define_ode_system(
    lambda x, ys: [-ys[1], ys[0] - ys[1] / x],
    lambda x0: [special_initial("besselj", x0, 0), special_initial("besselj", x0, 1)],
    "bessel_j01",
)


def bessel_j01(x):
    """Bessel functions ``(J_0(x), J_1(x))``, including regular-singular ``x0 == 0``."""
    if not isinstance(x, Taylor):
        if x == 0:
            return [x * 0 + 1, x * 0]
        return [special_initial("besselj", x, 0), special_initial("besselj", x, 1)]
    if x.primal == 0:
        j0_std = _bessel_j0_at_zero(x.primal)
        j1_std = -j0_std.diff()
        return [j0_std.compose(x), j1_std.compose(x)]
    return _bessel_j01_regular(x)


def bessel_j0(x):
    """Bessel function ``J_0(x)``, including regular-singular ``x0 == 0``."""
    if not isinstance(x, Taylor):
        return x * 0 + 1 if x == 0 else special_initial("besselj", x, 0)
    if x.primal == 0:
        return _bessel_j0_at_zero(x.primal).compose(x)
    return _bessel_j01_regular(x)[0]


_bessel_j_regular = define_ode_system(
    lambda x, ys, v: [ys[1], -ys[1] / x - (1 - (v * v) / (x * x)) * ys[0]],
    lambda x0, v: list(special_initial("besselj_pair", x0, v)),
    "bessel_j",
)
_bessel_y_regular = define_ode_system(
    lambda x, ys, v: [ys[1], -ys[1] / x - (1 - (v * v) / (x * x)) * ys[0]],
    lambda x0, v: list(special_initial("bessely_pair", x0, v)),
    "bessel_y",
)
_bessel_i_regular = define_ode_system(
    lambda x, ys, v: [ys[1], -ys[1] / x + (1 + (v * v) / (x * x)) * ys[0]],
    lambda x0, v: list(special_initial("besseli_pair", x0, v)),
    "bessel_i",
)
_bessel_k_regular = define_ode_system(
    lambda x, ys, v: [ys[1], -ys[1] / x + (1 + (v * v) / (x * x)) * ys[0]],
    lambda x0, v: list(special_initial("besselk_pair", x0, v)),
    "bessel_k",
)


def bessel_j(v, x):
    """Bessel function of the first kind ``J_v(x)`` (integer ``v >= 0`` works at ``x0 == 0``)."""
    if not isinstance(x, Taylor):
        return special_initial("besselj", x, v)
    if x.primal == 0 and isinstance(v, int) and v >= 0:
        return _bessel_jn_at_zero(v, x.primal).compose(x)
    return _bessel_j_regular(x, v)[0]


def bessel_y(v, x):
    """Bessel function of the second kind ``Y_v(x)`` (domain: ``x0 != 0``)."""
    if not isinstance(x, Taylor):
        return special_initial("bessely_pair", x, v)[0]
    return _bessel_y_regular(x, v)[0]


def bessel_i(v, x):
    """Modified Bessel function of the first kind ``I_v(x)`` (domain: ``x0 != 0``)."""
    if not isinstance(x, Taylor):
        return special_initial("besseli_pair", x, v)[0]
    return _bessel_i_regular(x, v)[0]


def bessel_k(v, x):
    """Modified Bessel function of the second kind ``K_v(x)`` (domain: ``x0 != 0``)."""
    if not isinstance(x, Taylor):
        return special_initial("besselk_pair", x, v)[0]
    return _bessel_k_regular(x, v)[0]


def _hyp2f1_at_zero(a, b, c, zero_like):
    one = zero_like * 0 + 1
    y = Taylor.pending(one)
    dy = y.euler_diff()
    rhs = (dy.euler_diff() + (a + b) * dy + (a * b) * y).shift(1)
    return y.bind_euler_ode(lambda k: k * (k + c - 1), rhs)


_hyp2f1_regular = define_ode_system(
    lambda z, ys, a, b, c: [
        ys[1],
        (a * b * ys[0] - (c - (a + b + 1) * z) * ys[1]) / (z * (1 - z)),
    ],
    lambda z0, a, b, c: list(special_initial("hyp2f1_pair", z0, a, b, c)),
    "hyp2f1",
)


def hyp2f1(a, b, c, z):
    """Gauss hypergeometric function ``2F1(a, b; c; z)``, including regular-singular ``z0 == 0``."""
    if not isinstance(z, Taylor):
        return special_initial("hyp2f1_pair", z, a, b, c)[0]
    if z.primal == 0:
        return _hyp2f1_at_zero(a, b, c, z.primal).compose(z)
    return _hyp2f1_regular(z, a, b, c)[0]


def _hyp1f1_at_zero(a, c, zero_like):
    one = zero_like * 0 + 1
    y = Taylor.pending(one)
    rhs = (y.euler_diff() + a * y).shift(1)
    return y.bind_euler_ode(lambda k: k * (k + c - 1), rhs)


_hyp1f1_regular = define_ode_system(
    lambda z, ys, a, c: [ys[1], (a * ys[0] - (c - z) * ys[1]) / z],
    lambda z0, a, c: list(special_initial("hyp1f1_pair", z0, a, c)),
    "hyp1f1",
)


def hyp1f1(a, c, z):
    """Confluent hypergeometric function ``1F1(a; c; z)``, including regular-singular ``z0 == 0``."""
    if not isinstance(z, Taylor):
        return special_initial("hyp1f1_pair", z, a, c)[0]
    if z.primal == 0:
        return _hyp1f1_at_zero(a, c, z.primal).compose(z)
    return _hyp1f1_regular(z, a, c)[0]


_legendre_sys = define_ode_system(
    lambda x, ys, n: [ys[1], (2 * x * ys[1] - n * (n + 1) * ys[0]) / (1 - x * x)],
    lambda x0, n: list(special_initial("legendre_pair", x0, n)),
    "legendre",
)
_chebyt_sys = define_ode_system(
    lambda x, ys, n: [ys[1], (x * ys[1] - n * n * ys[0]) / (1 - x * x)],
    lambda x0, n: list(special_initial("chebyt_pair", x0, n)),
    "chebyt",
)
_hermite_sys = define_ode_system(
    lambda x, ys, n: [ys[1], 2 * x * ys[1] - 2 * n * ys[0]],
    lambda x0, n: list(special_initial("hermite_pair", x0, n)),
    "hermite",
)
_laguerre_sys = define_ode_system(
    lambda x, ys, n: [ys[1], ((x - 1) * ys[1] - n * ys[0]) / x],
    lambda x0, n: list(special_initial("laguerre_pair", x0, n)),
    "laguerre",
)


def legendre(n, x):
    """Legendre function/polynomial ``P_n(x)`` (domain: ``x0 != +-1``)."""
    if not isinstance(x, Taylor):
        return special_initial("legendre_pair", x, n)[0]
    return _legendre_sys(x, n)[0]


def chebyt(n, x):
    """Chebyshev polynomial of the first kind ``T_n(x)`` (domain: ``x0 != +-1``)."""
    if not isinstance(x, Taylor):
        return special_initial("chebyt_pair", x, n)[0]
    return _chebyt_sys(x, n)[0]


def hermite(n, x):
    """Physicists' Hermite polynomial ``H_n(x)``."""
    if not isinstance(x, Taylor):
        return special_initial("hermite_pair", x, n)[0]
    return _hermite_sys(x, n)[0]


def laguerre(n, x):
    """Laguerre polynomial ``L_n(x)`` (domain: ``x0 != 0``)."""
    if not isinstance(x, Taylor):
        return special_initial("laguerre_pair", x, n)[0]
    return _laguerre_sys(x, n)[0]


jacobi_elliptic = define_ode_system(
    lambda x, ys, m: [ys[1] * ys[2], -ys[0] * ys[2], -m * ys[0] * ys[1]],
    lambda x0, m: list(special_initial("ellip_sn_cn_dn", x0, m)),
    "jacobi_elliptic",
)


def weierstrass_p(x, u0, v0, g2):
    """Weierstrass ℘ with explicit initial values ``(u0, v0)`` at ``x0``:
    ``u'' = 6u^2 - g2/2``."""
    return _ivp2(x, u0, v0, lambda x, u: 6 * u * u - g2 / 2)


def painleve_I(x, y0, dy0):
    """P_I: ``y'' = 6y^2 + x``, with explicit initial values at ``x0``."""
    return _ivp2(x, y0, dy0, lambda x, y: 6 * y * y + x)


def painleve_II(x, y0, dy0, alpha=0):
    """P_II: ``y'' = 2y^3 + x y + alpha``, explicit initial values."""
    return _ivp2(x, y0, dy0, lambda x, y: 2 * y * y * y + x * y + alpha)


# -- registry + exports -----------------------------------------------------------
FUNCTIONS = {
    fn.__name__: fn
    for fn in (
        reciprocal, exp, expm1, log, log1p, sin, cos, sin_cos, sinh, cosh, sinh_cosh,
        tan, tanh, atan, atanh, asin, acos, asinh, acosh, sqrt, power,
        erf, erfc, erfi, dawsn, fresnel, fresnel_s, fresnel_c,
        ellip_ke, ellipk, ellipe, si, ci, sici, dilog, spence,
        airy, airy_ai, airy_bi,
        bessel_j0, bessel_j01, bessel_j, bessel_y, bessel_i, bessel_k,
        hyp2f1, hyp1f1, legendre, chebyt, hermite, laguerre,
        jacobi_elliptic, weierstrass_p, painleve_I, painleve_II,
    )
}

__all__ = ["define_ode_function", "define_ode_system", "FUNCTIONS", *FUNCTIONS]
