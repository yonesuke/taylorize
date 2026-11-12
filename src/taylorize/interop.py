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

_DISPATCH_NAMES = (
    "exp", "log", "sqrt", "sin", "cos", "tan",
    "sinh", "cosh", "tanh", "asin", "atan", "power", "erf",
)
_DISPATCH: dict = {name: FUNCTIONS[name] for name in _DISPATCH_NAMES}
_DISPATCH.update({"arcsin": FUNCTIONS["asin"], "arctan": FUNCTIONS["atan"]})


def _dispatch_call(func, args, kwargs):
    if kwargs and kwargs.get("out") is not None:
        return NotImplemented
    fn = _DISPATCH.get(getattr(func, "__name__", ""))
    return fn(*args) if fn is not None else NotImplemented


def _array_ufunc_impl(self, ufunc, method, *inputs, **kwargs):
    return _dispatch_call(ufunc, inputs, kwargs) if method == "__call__" else NotImplemented


def _torch_function_impl(cls, func, types, args=(), kwargs=None):
    return _dispatch_call(func, args, kwargs)


# Inject interop hooks into Taylor
Taylor.__array_ufunc__ = _array_ufunc_impl
Taylor.__torch_function__ = classmethod(_torch_function_impl)


def dispatched_names():
    """Names currently dispatchable through numpy/torch protocols."""
    return sorted(_DISPATCH)
