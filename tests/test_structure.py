"""Tests for the layered architecture: the core stays minimal and
backend-free, functions are ODE declarations, and new functions can be
added from user code without touching the library."""
import subprocess
import sys
from fractions import Fraction

import pytest

from taylorize import Taylor, variable
from taylorize.functions import define_ode_function


def test_core_import_is_backend_free():
    code = (
        "import sys; import taylorize.core; "
        "bad = [m for m in ('numpy', 'mpmath', 'sympy', 'scipy') if m in sys.modules]; "
        "assert not bad, bad; "
        "import taylorize.core as c; assert not hasattr(c, 'exp'); print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_pending_guard_raises():
    y = Taylor.pending(1.0)
    with pytest.raises(RuntimeError):
        y.coeff(2)


def test_exact_rational_coefficients():
    from taylorize.functions import tan

    c = tan(variable(Fraction(0))).ensure(7).c
    assert c == [
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(1, 3),
        Fraction(0),
        Fraction(2, 15),
        Fraction(0),
        Fraction(17, 315),
    ]


def test_user_defined_function_from_ode():
    # logistic: y' = y(1-y), y(0) = 1/2 -- one line, no library changes.
    logistic = define_ode_function(lambda x, y: y * (1 - y), lambda x0: 1 / (1 + __import__("math").exp(-x0)))
    c = logistic(variable(0.0)).ensure(5).c
    # sigma^(k)(0)/k!: sigma'=1/4, sigma'''=-1/8 -> c3=-1/48, sigma^5=1/4 -> c5=1/480
    assert abs(c[1] - 0.25) < 1e-15
    assert abs(c[3] - (-1 / 48)) < 1e-15
    assert abs(c[5] - (1 / 480)) < 1e-12


def test_mpmath_backend_high_precision():
    mp = pytest.importorskip("mpmath").mp
    mp.dps = 50
    from taylorize.functions import erf, exp, sin

    val = exp(sin(variable(mp.mpf(0)))).eval(mp.mpf("0.5"), order=60)
    assert abs(val - mp.exp(mp.sin(mp.mpf("0.5")))) < mp.mpf("1e-40")
    assert abs(erf(variable(mp.mpf(0))).eval(mp.mpf(1), order=80) - mp.erf(1)) < mp.mpf("1e-40")


def test_sympy_backend_symbolic_coefficients():
    sp = pytest.importorskip("sympy")
    from taylorize.functions import exp

    c = exp(variable(sp.Integer(0))).ensure(4).c
    assert [sp.simplify(v - sp.Rational(1, __import__("math").factorial(k))) for k, v in enumerate(c)] == [0] * 5


def test_bessel_and_airy_mpmath():
    mp = pytest.importorskip("mpmath").mp
    mp.dps = 30
    from taylorize.functions import airy, bessel_j01

    j0, _j1 = bessel_j01(variable(mp.mpf(1)))
    assert abs(j0.eval(mp.mpf("0.01"), order=25) - mp.besselj(0, mp.mpf("1.01"))) < mp.mpf("1e-20")
    ai, _aip = airy(variable(mp.mpf(1)))
    assert abs(ai.eval(mp.mpf("0.01"), order=25) - mp.airyai(mp.mpf("1.01"))) < mp.mpf("1e-20")


def test_numpy_dispatch():
    np = pytest.importorskip("numpy")
    from taylorize.functions import exp

    y = np.exp(variable(0.0))
    assert isinstance(y, Taylor) and abs(y.coeff(3) - 1 / 6) < 1e-15
