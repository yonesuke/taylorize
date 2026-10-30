"""taylorize — lazy truncated Taylor series for higher-order automatic
differentiation and Taylor-method ODE integration.

A :class:`Taylor` is the normalized Taylor polynomial of a smooth function
around a point.  Higher-order coefficients are computed lazily, so arbitrary
orders can be requested *after* the expression has been built.

This is the minimal 0.0.1 seed of the project.
"""
from .core import (
    Taylor,
    constant,
    cos,
    derivative,
    exp,
    log,
    ode_step,
    power,
    sin,
    sqrt,
    tan,
    taylor_series,
    variable,
)

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
    "derivative",
    "taylor_series",
    "ode_step",
]
