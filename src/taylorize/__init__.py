"""taylorize — lazy truncated Taylor series for higher-order automatic
differentiation.

Architecture (one-way dependencies; see docs/design/ode-defined-functions.md):

- :mod:`taylorize.core` -- the minimal lazy power-series ring plus the
  ``bind_ode`` wiring primitive.  No functions, no backends, no host hooks.
- :mod:`taylorize.functions` -- every elementary/special function,
  declared by the ODE it satisfies and built with ``bind_ode``.
- :mod:`taylorize.backends` -- how initial (0th-order) values are
  obtained per coefficient backend (float/math, mpmath, sympy, exact).
- :mod:`taylorize.transforms` -- ``derivative`` / ``taylor_series``.
- :mod:`taylorize.interop` -- numpy/torch protocol dispatch dynamically wired.
"""
from . import interop as _interop  # noqa: F401  (registers host-library interop hooks)
from .core import Derivation, Taylor, constant, variable
from .functions import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    bessel_i,
    bessel_j,
    bessel_j0,
    bessel_k,
    bessel_y,
    chebyt,
    ci,
    cos,
    dawsn,
    dilog,
    ellipe,
    ellipk,
    erf,
    erfc,
    erfi,
    exp,
    expm1,
    fresnel_c,
    fresnel_s,
    hermite,
    hyp1f1,
    hyp2f1,
    laguerre,
    legendre,
    log,
    log1p,
    power,
    si,
    sin,
    spence,
    sqrt,
    tan,
)
from .transforms import derivative, taylor_series

__version__ = "0.0.3"

__all__ = [
    "Taylor", "Derivation", "variable", "constant",
    "exp", "expm1", "log", "log1p", "sin", "cos", "tan",
    "sqrt", "power", "asin", "acos", "atan", "asinh", "acosh", "atanh",
    "erf", "erfc", "erfi", "dawsn", "fresnel_s", "fresnel_c",
    "ellipk", "ellipe", "si", "ci", "dilog", "spence",
    "bessel_j0", "bessel_j", "bessel_y", "bessel_i", "bessel_k",
    "hyp2f1", "hyp1f1", "legendre", "chebyt", "hermite", "laguerre",
    "derivative", "taylor_series",
]
