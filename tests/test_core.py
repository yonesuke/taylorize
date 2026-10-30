import math

from taylorize import derivative, exp, ode_step, sin, taylor_series


def test_derivative_of_exp():
    assert abs(derivative(exp, k=5)(0.0) - 1.0) < 1e-12


def test_sin_series_matches_factorials():
    c = taylor_series(sin, order=9)(0.0)
    expected = [0.0, 1.0, 0.0, -1 / 6, 0.0, 1 / 120, 0.0, -1 / 5040, 0.0, 1 / 362880]
    for a, b in zip(c, expected):
        assert abs(a - b) < 1e-15


def test_ode_step_harmonic_oscillator():
    def ho(u):
        return [u[1], -u[0]]

    step = ode_step(ho, order=20)
    x, v = step([0.0, 1.0], 0.3)
    assert abs(x - math.sin(0.3)) < 1e-15
    assert abs(v - math.cos(0.3)) < 1e-15
