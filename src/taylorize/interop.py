"""Host-library interop: using Taylor objects inside numpy / torch /
mpmath / sympy expressions.

This module registers the dispatch table and injects the interop hooks
(``__array_ufunc__`` and ``__torch_function__``) onto the core :class:`Taylor`
class dynamically, keeping the core completely free of host-library concepts.
"""
from __future__ import annotations

from .core import Taylor
from .functions import FUNCTIONS

__all__ = ["dispatched_names", "_DISPATCH"]

# Mapping from host-library function names to ODE-defined functions
_DISPATCH: dict = {
    name: fn
    for name, fn in FUNCTIONS.items()
    if name in (
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "power",
        "erf",
    )
}
# Common NumPy / PyTorch aliases
_DISPATCH.update(
    {
        "arcsin": FUNCTIONS["asin"],
        "arctan": FUNCTIONS["atan"],
    }
)


def _array_ufunc_impl(self, ufunc, method, *inputs, **kwargs):
    if method != "__call__" or kwargs.get("out") is not None:
        return NotImplemented
    fn = _DISPATCH.get(getattr(ufunc, "__name__", ""))
    return fn(*inputs) if fn is not None else NotImplemented


def _torch_function_impl(cls, func, types, args=(), kwargs=None):
    if kwargs and kwargs.get("out") is not None:
        return NotImplemented
    fn = _DISPATCH.get(getattr(func, "__name__", ""))
    return fn(*args) if fn is not None else NotImplemented


# Inject interop hooks into Taylor
Taylor.__array_ufunc__ = _array_ufunc_impl
Taylor.__torch_function__ = classmethod(_torch_function_impl)


def dispatched_names():
    """Names currently dispatchable through numpy/torch protocols."""
    return sorted(_DISPATCH)
