"""Host-library interop: using Taylor objects inside numpy / torch /
mpmath / sympy expressions.

This module registers the dispatch table and injects the interop hooks
(``__array_ufunc__`` and ``__torch_function__``) onto the core :class:`Taylor`
class dynamically, keeping the core completely free of host-library concepts.
"""
from __future__ import annotations

import sys

from .core import Taylor
from .functions import FUNCTIONS

__all__ = ["dispatched_names", "_DISPATCH"]

_DISPATCH_NAMES = (
    "exp", "log", "sqrt", "sin", "cos", "tan",
    "sinh", "cosh", "tanh", "asin", "atan", "power", "erf",
)
_DISPATCH: dict = {name: FUNCTIONS[name] for name in _DISPATCH_NAMES}
_DISPATCH.update(
    {
        "arcsin": FUNCTIONS["asin"],
        "arctan": FUNCTIONS["atan"],
        "airyai": FUNCTIONS["airy_ai"],
        "airybi": FUNCTIONS["airy_bi"],
    }
)
_MPMATH_PATCHED = False


def _patch_mpmath(mpmath=None) -> None:
    global _MPMATH_PATCHED
    if _MPMATH_PATCHED:
        return
    mpmath = mpmath or sys.modules.get("mpmath")
    mp = getattr(mpmath, "mp", None)
    if mp is None:
        return
    _MPMATH_PATCHED = True
    for name, fn in _DISPATCH.items():
        orig = getattr(mp, name, None)
        if callable(orig):
            wrapped = (
                lambda f, o: lambda x, *a, **kw: f(x, *a) if isinstance(x, Taylor) else o(x, *a, **kw)
            )(fn, orig)
            setattr(mp, name, wrapped)
            if hasattr(mpmath, name):
                setattr(mpmath, name, wrapped)


class _MpmathImportHook:
    def find_spec(self, fullname, path, target=None):
        if fullname == "mpmath":
            sys.meta_path.remove(self)
            import importlib.machinery

            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                orig_exec = spec.loader.exec_module

                def exec_module(module):
                    orig_exec(module)
                    _patch_mpmath(module)

                spec.loader.exec_module = exec_module
            return spec
        return None


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
if "mpmath" in sys.modules:
    _patch_mpmath()
else:
    sys.meta_path.insert(0, _MpmathImportHook())


def dispatched_names():
    """Names currently dispatchable through numpy/torch/mpmath protocols."""
    return sorted(_DISPATCH)
