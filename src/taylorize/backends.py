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


_FRACTION_AT_ZERO = {
    "exp": Fraction(1), "expm1": Fraction(0), "log1p": Fraction(0),
    "cos": Fraction(1), "cosh": Fraction(1),
    "sin": Fraction(0), "sinh": Fraction(0), "tan": Fraction(0),
    "tanh": Fraction(0), "atan": Fraction(0), "asin": Fraction(0),
    "asinh": Fraction(0), "atanh": Fraction(0),
    "erf": Fraction(0), "erfc": Fraction(1), "erfi": Fraction(0),
}


def _exact_fraction_initial(name: str, x0: Fraction):
    """Exact initial values that stay rational, or ``None``.

    A Fraction must never be silently demoted through ``math``: only a
    demonstrably rational value at the requested point is acceptable.
    """
    if x0 == 0 and name in _FRACTION_AT_ZERO:
        return _FRACTION_AT_ZERO[name]
    if name == "log" and x0 == 1:
        return Fraction(0)
    if name == "sqrt" and x0 >= 0:
        nr, dr = math.isqrt(x0.numerator), math.isqrt(x0.denominator)
        if nr * nr == x0.numerator and dr * dr == x0.denominator:
            return Fraction(nr, dr)
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
    if isinstance(x0, complex):
        if name == "expm1":
            return cmath.exp(x0) - 1
        if name == "log1p":
            return cmath.log(1 + x0)
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
    differ per backend (Bessel, Airy, Jacobi elliptic, Fresnel, etc.).

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
    elif kind in ("besselj_pair", "bessely_pair", "besseli_pair", "besselk_pair"):
        (v,) = params
        mp_name = kind.split("_")[0]  # besselj / bessely / besseli / besselk
        fn = getattr(ns, mp_name, None)
        if fn is not None:
            val = fn(v, x0)
            vm1 = fn(v - 1, x0)
            if mp_name == "besselk":
                der = -vm1 - (v / x0) * val
            else:
                der = vm1 - (v / x0) * val
            return (val, der)
        if isinstance(x0, (float, complex)):
            import scipy.special as sp

            val_fn = {"besselj": sp.jv, "bessely": sp.yv, "besseli": sp.iv, "besselk": sp.kv}[mp_name]
            der_fn = {"besselj": sp.jvp, "bessely": sp.yvp, "besseli": sp.ivp, "besselk": sp.kvp}[mp_name]
            return (val_fn(v, x0), der_fn(v, x0))
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
    elif kind == "ellip_ke":
        fk = getattr(ns, "ellipk", None)
        fe = getattr(ns, "ellipe", None)
        if fk is not None and fe is not None:
            return (fk(x0), fe(x0))
        if isinstance(x0, (float, complex)):
            from scipy.special import ellipe, ellipk

            return (ellipk(x0), ellipe(x0))
    elif kind == "erfi":
        if x0 == 0:
            return x0 * 0
        fn = getattr(ns, "erfi", None)
        if fn is not None:
            return fn(x0)
        if isinstance(x0, (float, complex)):
            from scipy.special import erfi

            return erfi(x0)
    elif kind == "dawsn":
        if x0 == 0:
            return x0 * 0
        fn = getattr(ns, "dawsn", None)
        if fn is not None:
            return fn(x0)
        erfi_fn = getattr(ns, "erfi", None)
        if erfi_fn is not None:
            pi_val = backend_constant("pi", x0)
            return ns.sqrt(pi_val) / 2 * ns.exp(-(x0 * x0)) * erfi_fn(x0)
        if isinstance(x0, (float, complex)):
            from scipy.special import dawsn

            return dawsn(x0)
    elif kind == "fresnel":
        if x0 == 0:
            return (x0 * 0, x0 * 0)
        fs, fc = getattr(ns, "fresnels", None), getattr(ns, "fresnelc", None)
        if fs is not None and fc is not None:
            return (fs(x0), fc(x0))
        if isinstance(x0, (float, complex)):
            from scipy.special import fresnel

            return fresnel(x0)
    elif kind in ("si", "ci"):
        if kind == "si" and x0 == 0:
            return x0 * 0
        fn = getattr(ns, kind, None)
        if fn is not None:
            return fn(x0)
        if isinstance(x0, (float, complex)):
            from scipy.special import sici

            s, c = sici(x0)
            return s if kind == "si" else c
    elif kind == "dilog":
        if x0 == 0:
            return x0 * 0
        fn = getattr(ns, "polylog", None)
        if fn is not None:
            return fn(2, x0)
        if isinstance(x0, (float, complex)):
            from scipy.special import spence

            return spence(1 - x0)
    elif kind == "hyp2f1_pair":
        a, b, c = params
        fn = getattr(ns, "hyp2f1", None)
        if fn is not None:
            return (fn(a, b, c, x0), fn(a + 1, b + 1, c + 1, x0) * a * b / c)
        if isinstance(x0, (float, complex)):
            from scipy.special import hyp2f1

            return (hyp2f1(a, b, c, x0), hyp2f1(a + 1, b + 1, c + 1, x0) * a * b / c)
    elif kind == "hyp1f1_pair":
        a, c = params
        fn = getattr(ns, "hyp1f1", None)
        if fn is not None:
            return (fn(a, c, x0), fn(a + 1, c + 1, x0) * a / c)
        if isinstance(x0, (float, complex)):
            from scipy.special import hyp1f1

            return (hyp1f1(a, c, x0), hyp1f1(a + 1, c + 1, x0) * a / c)
    elif kind == "legendre_pair":
        (n,) = params
        fn = getattr(ns, "legendre", None)
        if fn is not None:
            p0 = fn(n, x0)
            if n == 0:
                return (p0, x0 * 0)
            pm = fn(n - 1, x0)
            return (p0, n * (x0 * p0 - pm) / (x0 * x0 - 1))
        if isinstance(x0, (float, complex)):
            from scipy.special import eval_legendre

            p0 = eval_legendre(n, x0)
            if n == 0:
                return (p0, x0 * 0)
            pm = eval_legendre(n - 1, x0)
            return (p0, n * (x0 * p0 - pm) / (x0 * x0 - 1))
    elif kind == "chebyt_pair":
        (n,) = params
        fn = getattr(ns, "chebyt", None)
        if fn is not None:
            t0 = fn(n, x0)
            if n == 0:
                return (t0, x0 * 0)
            tm = fn(n - 1, x0)
            return (t0, n * (x0 * t0 - tm) / (x0 * x0 - 1))
        if isinstance(x0, (float, complex)):
            from scipy.special import eval_chebyt, eval_chebyu

            return (eval_chebyt(n, x0), n * eval_chebyu(n - 1, x0) if n != 0 else x0 * 0)
    elif kind == "hermite_pair":
        (n,) = params
        fn = getattr(ns, "hermite", None)
        if fn is not None:
            return (fn(n, x0), 2 * n * fn(n - 1, x0) if n != 0 else x0 * 0)
        if isinstance(x0, (float, complex)):
            from scipy.special import eval_hermite

            return (eval_hermite(n, x0), 2 * n * eval_hermite(n - 1, x0) if n != 0 else x0 * 0)
    elif kind == "laguerre_pair":
        (n,) = params
        fn = getattr(ns, "laguerre", None)
        if fn is not None:
            try:
                return (fn(n, 0, x0), -fn(n - 1, 1, x0) if n != 0 else x0 * 0)
            except TypeError:
                return (fn(n, x0), -getattr(ns, "assoc_laguerre")(n - 1, 1, x0) if n != 0 else x0 * 0)
        if isinstance(x0, (float, complex)):
            from scipy.special import eval_genlaguerre, eval_laguerre

            return (eval_laguerre(n, x0), -eval_genlaguerre(n - 1, 1, x0) if n != 0 else x0 * 0)
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
