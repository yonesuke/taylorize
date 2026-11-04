"""Coefficient backends: how to get 0th-order (primal) values.

The core never evaluates a named function.  When a function defined in
:mod:`taylorize.functions` is applied to a Taylor series, only its value
at the expansion point -- a single scalar in the coefficient backend --
is needed from outside.  This module resolves that value from the
backend the coefficient belongs to:

- ``float`` / ``complex`` coefficients -> :mod:`math` / :mod:`cmath`,
  with a documented fallback to ``scipy.special`` for special functions
  that the standard library lacks (Bessel, Airy, Jacobi elliptic).
- ``mpmath`` coefficients -> the active mpmath context (arbitrary precision).
- ``sympy`` coefficients -> sympy itself (symbolic initial values).
- exact rational coefficients -> only operations that stay rational are
  available; transcendental initial values raise an informative error
  instead of silently degrading to float.

Adding a backend means teaching this module how to resolve names in it;
neither the core nor the function definitions change.
"""
from __future__ import annotations

import cmath
import math
import types
from fractions import Fraction

__all__ = ["namespace_for", "eval_initial", "special_initial", "backend_constant"]


def namespace_for(x0):
    """Return the math namespace that natively understands scalar ``x0``."""
    if hasattr(x0, "__array_namespace__"):
        try:
            return x0.__array_namespace__()
        except Exception:
            pass
    ctx = getattr(x0, "context", None) or getattr(x0, "ctx", None)
    if ctx is not None:
        return ctx
    mod_root = type(x0).__module__.split(".")[0]
    if mod_root not in ("builtins", "numbers", "fractions", "decimal"):
        import sys

        mod = sys.modules.get(mod_root)
        if mod is not None:
            # The owning library itself first (numpy, mpmath, sympy all
            # expose exp/sin/...); a ``.math`` submodule is only a
            # fallback -- and for numpy it is a deprecated alias of the
            # stdlib module, which would silently scalarize arrays.
            if hasattr(mod, "sin") and hasattr(mod, "exp"):
                return mod
            math_sub = getattr(mod, "math", None)
            if math_sub is not None and hasattr(math_sub, "sin"):
                return math_sub
            if hasattr(mod, "sin"):
                return mod
    return cmath if isinstance(x0, complex) else math


def _exact_fraction_initial(name: str, x0: Fraction):
    """Exact initial values that stay rational, or ``None``.

    A Fraction must never be silently demoted through ``math``: only a
    demonstrably rational value at the requested point is acceptable.
    """
    zero, one = Fraction(0), Fraction(1)
    at_zero = {
        "exp": one,
        "sin": zero,
        "cos": one,
        "sinh": zero,
        "cosh": one,
        "tan": zero,
        "tanh": zero,
        "atan": zero,
        "asin": zero,
        "erf": zero,
    }
    if x0 == 0 and name in at_zero:
        return at_zero[name]
    if name == "log" and x0 == 1:
        return zero
    if name == "sqrt" and x0 >= 0:
        num_root = math.isqrt(x0.numerator)
        den_root = math.isqrt(x0.denominator)
        if num_root * num_root == x0.numerator and den_root * den_root == x0.denominator:
            return Fraction(num_root, den_root)
    return None


def eval_initial(name: str, x0, *extra):
    """Evaluate the function ``name`` at the scalar ``x0`` in its backend,
    as ``name(x0, *extra)``.

    Raises AttributeError with an explicit message when the backend
    cannot provide the value -- the fallback policy is part of the API.
    """
    if isinstance(x0, Fraction):
        exact = _exact_fraction_initial(name, x0)
        if exact is not None:
            return exact
        raise AttributeError(
            f"the exact rational backend does not provide a rational {name!r} "
            f"initial value at {x0!r}; use float/mpmath/sympy coefficients, "
            f"or supply the initial value explicitly"
        )
    ns = namespace_for(x0)
    fn = getattr(ns, name, None)
    if fn is not None:
        return fn(x0, *extra)
    method = getattr(x0, name, None)
    if callable(method):
        return method(*extra)
    raise AttributeError(
        f"backend {ns!r} does not provide {name!r} for an initial value at "
        f"{x0!r}; use coefficients from a backend that implements it "
        f"(e.g. mpmath or sympy), or supply the initial value explicitly"
    )


def special_initial(kind: str, x0, *params):
    """Initial values for special functions whose calling conventions
    differ per backend (Bessel, Airy, Jacobi elliptic).

    Resolution order: the coefficient's own backend first, then
    ``scipy.special`` for plain float/complex coefficients.
    """
    ns = namespace_for(x0)
    if kind == "besselj":
        (n,) = params
        fn = getattr(ns, "besselj", None)
        if fn is not None:
            return fn(n, x0)
        if isinstance(x0, (float, complex)):
            from scipy.special import jv

            return jv(n, x0)
    elif kind in ("airyai", "airyaiprime", "airybi", "airybiprime"):
        base = kind[:6]  # "airyai" or "airybi"
        idx = 0 if kind == base else 1  # scipy .airy() order: Ai, Ai', Bi, Bi'
        if base == "airybi":
            idx += 2
        fn = getattr(ns, kind, None)
        if fn is not None:
            return fn(x0)
        if idx % 2 == 1:
            base_fn = getattr(ns, base, None)
            if base_fn is not None:  # mpmath convention: airyai(x, derivative=1)
                try:
                    return base_fn(x0, derivative=1)
                except TypeError:
                    pass
        if isinstance(x0, (float, complex)):
            from scipy.special import airy

            return airy(x0)[idx]
    elif kind == "ellip_sn_cn_dn":
        (m,) = params
        if x0 == 0:
            return (x0 * 0, x0 * 0 + 1, x0 * 0 + 1)
        fn = getattr(ns, "ellipfun", None)
        if fn is not None:
            return tuple(fn(k, x0, m=m) for k in ("sn", "cn", "dn"))
        if isinstance(x0, (float, complex)):
            from scipy.special import ellipj

            sn, cn, dn, _ph = ellipj(x0, m)
            return (sn, cn, dn)
    raise AttributeError(
        f"backend {ns!r} does not provide {kind!r} initial values at {x0!r}; "
        f"use mpmath/sympy coefficients, install scipy for float coefficients, "
        f"or supply the initial values explicitly"
    )


def backend_constant(name: str, like):
    """A named constant (``pi``, ...) in the backend of the scalar ``like``."""
    if isinstance(like, Fraction):
        raise AttributeError(f"the exact rational backend does not provide constant {name!r}")
    ns = namespace_for(like)
    val = getattr(ns, name, None)
    function_types = (
        types.FunctionType,
        types.BuiltinFunctionType,
        types.MethodType,
    )
    if val is not None and not isinstance(val, function_types):
        # Some contexts (notably mpmath) expose constants as lazy scalar
        # objects that are technically callable; they are still constants.
        mpf = getattr(ns, "mpf", None)
        if callable(mpf):
            try:
                return mpf(val)
            except (TypeError, ValueError):
                pass
        return val
    if name == "pi":
        return math.pi
    raise AttributeError(f"backend {ns!r} does not provide constant {name!r}")
