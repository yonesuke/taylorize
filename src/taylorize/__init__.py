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
from .core import Taylor, constant, variable
from .functions import (
    asin,
    atan,
    cos,
    erf,
    exp,
    log,
    power,
    sin,
    sqrt,
    tan,
)
from .transforms import derivative, taylor_series

__version__ = "0.0.1"

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
    "asin",
    "atan",
    "erf",
    "derivative",
    "taylor_series",
]
