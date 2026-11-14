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
log = define_ode_function(lambda x, y: 1 / x, "log")

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
sqrt = define_ode_function(lambda x, y: 1 / (2 * y), "sqrt")
asin = define_ode_function(lambda x, y: 1 / sqrt(1 - x * x), "asin")

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
        reciprocal, exp, log, sin, cos, sin_cos, sinh, cosh, sinh_cosh,
        tan, tanh, atan, asin, sqrt, power, erf, airy, airy_ai, airy_bi,
        bessel_j0, bessel_j01, jacobi_elliptic, weierstrass_p, painleve_I, painleve_II,
    )
}

__all__ = ["define_ode_function", "define_ode_system", "FUNCTIONS", *FUNCTIONS]
