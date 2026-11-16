from fractions import Fraction

import mpmath as mp
import numpy as np

from taylorize import (
    acos,
    acosh,
    asinh,
    atanh,
    bessel_i,
    bessel_j,
    bessel_k,
    bessel_y,
    chebyt,
    ci,
    dawsn,
    dilog,
    ellipe,
    ellipk,
    erfc,
    erfi,
    expm1,
    fresnel_c,
    fresnel_s,
    hermite,
    hyp1f1,
    hyp2f1,
    laguerre,
    legendre,
    log1p,
    si,
    spence,
    variable,
)


def _check_mpmath_eval(taylor_expr, ref_fn, x0_str, h_str="0.05", order=30, tol=1e-35):
    mp.mp.dps = 50
    x0 = mp.mpf(x0_str)
    h = mp.mpf(h_str)
    got = taylor_expr.eval(h, order=order)
    expected = ref_fn(x0 + h)
    assert abs(got - expected) < tol


def test_elementary_extensions_and_numpy_dispatch():
    mp.mp.dps = 50

    # Exact rational expansions at 0
    z = variable(Fraction(0))
    assert expm1(z).ensure(4).c == [Fraction(0), Fraction(1), Fraction(1, 2), Fraction(1, 6), Fraction(1, 24)]
    assert log1p(z).ensure(4).c == [Fraction(0), Fraction(1), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 4)]
    assert atanh(z).ensure(5).c == [Fraction(0), Fraction(1), Fraction(0), Fraction(1, 3), Fraction(0), Fraction(1, 5)]
    assert asinh(z).ensure(5).c == [Fraction(0), Fraction(1), Fraction(0), Fraction(-1, 6), Fraction(0), Fraction(3, 40)]

    # High-precision mpmath checks
    _check_mpmath_eval(acos(variable(mp.mpf("0.3"))), mp.acos, "0.3")
    _check_mpmath_eval(asinh(variable(mp.mpf("0.4"))), mp.asinh, "0.4")
    _check_mpmath_eval(acosh(variable(mp.mpf("1.5"))), mp.acosh, "1.5", order=35)
    _check_mpmath_eval(atanh(variable(mp.mpf("0.2"))), mp.atanh, "0.2")
    _check_mpmath_eval(expm1(variable(mp.mpf("0.25"))), mp.expm1, "0.25")
    _check_mpmath_eval(log1p(variable(mp.mpf("0.25"))), mp.log1p, "0.25")

    # NumPy ufunc dispatch
    x = variable(0.3)
    h = 0.05
    assert abs(np.arccos(x).eval(h, order=14) - np.arccos(0.3 + h)) < 1e-14
    assert abs(np.arcsinh(x).eval(h, order=14) - np.arcsinh(0.3 + h)) < 1e-14
    assert abs(np.arccosh(variable(1.5)).eval(h, order=14) - np.arccosh(1.5 + h)) < 1e-14
    assert abs(np.arctanh(x).eval(h, order=14) - np.arctanh(0.3 + h)) < 1e-14
    assert abs(np.expm1(x).eval(h, order=14) - np.expm1(0.3 + h)) < 1e-14
    assert abs(np.log1p(x).eval(h, order=14) - np.log1p(0.3 + h)) < 1e-14


def test_erfc_erfi_dawsn_fresnel():
    mp.mp.dps = 50
    _check_mpmath_eval(erfc(variable(mp.mpf("0.3"))), mp.erfc, "0.3")
    _check_mpmath_eval(erfi(variable(mp.mpf("0.0"))), mp.erfi, "0.0")
    _check_mpmath_eval(erfi(variable(mp.mpf("0.3"))), mp.erfi, "0.3")
    _check_mpmath_eval(
        dawsn(variable(mp.mpf("0.4"))),
        lambda z: mp.sqrt(mp.pi) / 2 * mp.exp(-z * z) * mp.erfi(z),
        "0.4",
    )
    _check_mpmath_eval(fresnel_s(variable(mp.mpf("0.0"))), mp.fresnels, "0.0")
    _check_mpmath_eval(fresnel_c(variable(mp.mpf("0.0"))), mp.fresnelc, "0.0")
    _check_mpmath_eval(fresnel_s(variable(mp.mpf("0.5"))), mp.fresnels, "0.5")
    _check_mpmath_eval(fresnel_c(variable(mp.mpf("0.5"))), mp.fresnelc, "0.5")

    # mpmath dispatch
    x = variable(mp.mpf("0.2"))
    h = mp.mpf("0.05")
    assert abs(mp.erfc(x).eval(h, order=30) - mp.erfc(mp.mpf("0.25"))) < 1e-30
    assert abs(mp.erfi(x).eval(h, order=30) - mp.erfi(mp.mpf("0.25"))) < 1e-30
    assert abs(mp.fresnels(x).eval(h, order=30) - mp.fresnels(mp.mpf("0.25"))) < 1e-30
    assert abs(mp.fresnelc(x).eval(h, order=30) - mp.fresnelc(mp.mpf("0.25"))) < 1e-30

    # float at x0 = 0 works without scipy
    assert abs(dawsn(variable(0.0)).eval(0.1, order=14) - float(mp.sqrt(mp.pi) / 2 * mp.exp(-0.01) * mp.erfi(0.1))) < 1e-14


def test_ellipk_ellipe():
    mp.mp.dps = 50
    m = variable(mp.mpf("0.3"))
    _check_mpmath_eval(ellipk(m), mp.ellipk, "0.3", h_str="0.02", order=25)
    _check_mpmath_eval(ellipe(m), mp.ellipe, "0.3", h_str="0.02", order=25)
    assert abs(mp.ellipk(m).eval(mp.mpf("0.02"), order=25) - mp.ellipk(mp.mpf("0.32"))) < 1e-30
    assert abs(mp.ellipe(m).eval(mp.mpf("0.02"), order=25) - mp.ellipe(mp.mpf("0.32"))) < 1e-30


def test_bessel_family():
    # Exact J_2 at regular-singular origin x0 = 0
    j2 = bessel_j(2, variable(Fraction(0)))
    assert j2.ensure(6).c == [
        Fraction(0),
        Fraction(0),
        Fraction(1, 8),
        Fraction(0),
        Fraction(-1, 96),
        Fraction(0),
        Fraction(1, 3072),
    ]

    mp.mp.dps = 50
    x = variable(mp.mpf("1.5"))
    _check_mpmath_eval(bessel_j(mp.mpf("0.5"), x), lambda z: mp.besselj(mp.mpf("0.5"), z), "1.5")
    _check_mpmath_eval(bessel_y(1, x), lambda z: mp.bessely(1, z), "1.5")
    _check_mpmath_eval(bessel_i(1, x), lambda z: mp.besseli(1, z), "1.5")
    _check_mpmath_eval(bessel_k(1, x), lambda z: mp.besselk(1, z), "1.5")

    # mpmath dispatch with Taylor in 2nd argument
    h = mp.mpf("0.05")
    assert abs(mp.besselj(2, x).eval(h, order=30) - mp.besselj(2, mp.mpf("1.55"))) < 1e-30


def test_sici_and_dilog():
    mp.mp.dps = 50
    _check_mpmath_eval(si(variable(mp.mpf("1.2"))), mp.si, "1.2")
    _check_mpmath_eval(ci(variable(mp.mpf("1.2"))), mp.ci, "1.2")
    _check_mpmath_eval(dilog(variable(mp.mpf("0.4"))), lambda z: mp.polylog(2, z), "0.4")
    _check_mpmath_eval(spence(variable(mp.mpf("0.6"))), lambda z: mp.polylog(2, 1 - z), "0.6")

    assert abs(mp.si(variable(mp.mpf("1.2"))).eval(mp.mpf("0.05"), order=30) - mp.si(mp.mpf("1.25"))) < 1e-30
    assert abs(mp.ci(variable(mp.mpf("1.2"))).eval(mp.mpf("0.05"), order=30) - mp.ci(mp.mpf("1.25"))) < 1e-30


def test_hypergeometric_and_orthogonal_polynomials():
    # Exact rational 2F1(1, 1; 2; z) = -log(1 - z) / z = sum z^k / (k + 1)
    f = hyp2f1(Fraction(1), Fraction(1), Fraction(2), variable(Fraction(0)))
    assert f.ensure(5).c == [Fraction(1, k + 1) for k in range(6)]

    # Exact rational 1F1(1; 2; z) = (e^z - 1) / z = sum z^k / (k + 1)!
    g = hyp1f1(Fraction(1), Fraction(2), variable(Fraction(0)))
    assert g.ensure(4).c == [Fraction(1), Fraction(1, 2), Fraction(1, 6), Fraction(1, 24), Fraction(1, 120)]

    mp.mp.dps = 50
    _check_mpmath_eval(
        hyp2f1(mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("1.0"), variable(mp.mpf("0.25"))),
        lambda z: mp.hyp2f1(mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("1.0"), z),
        "0.25",
    )
    _check_mpmath_eval(
        hyp1f1(mp.mpf("0.5"), mp.mpf("1.5"), variable(mp.mpf("0.3"))),
        lambda z: mp.hyp1f1(mp.mpf("0.5"), mp.mpf("1.5"), z),
        "0.3",
    )

    x = variable(mp.mpf("0.4"))
    _check_mpmath_eval(legendre(3, x), lambda z: mp.legendre(3, z), "0.4")
    _check_mpmath_eval(chebyt(4, x), lambda z: mp.chebyt(4, z), "0.4")
    _check_mpmath_eval(hermite(4, x), lambda z: mp.hermite(4, z), "0.4")
    _check_mpmath_eval(laguerre(3, x), lambda z: mp.laguerre(3, 0, z), "0.4")

    # mpmath dispatch
    h = mp.mpf("0.05")
    assert abs(mp.hyp2f1(1, 2, 3, variable(mp.mpf("0.2"))).eval(mp.mpf("0.01"), order=30) - mp.hyp2f1(1, 2, 3, mp.mpf("0.21"))) < 1e-30
    assert abs(mp.legendre(3, x).eval(h, order=30) - mp.legendre(3, mp.mpf("0.45"))) < 1e-30
