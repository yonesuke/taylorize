"""Function transforms: turning plain functions into derivative or
series functions.  These only use the core (a variable jet in, its
coefficients out), so they live next to it, not with the function table.
"""
from __future__ import annotations

from typing import Callable

from .core import variable

__all__ = ["derivative", "taylor_series"]


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
