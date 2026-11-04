from taylorize import derivative, exp, sin, taylor_series


def test_derivative_of_exp():
    assert abs(derivative(exp, k=5)(0.0) - 1.0) < 1e-12


def test_sin_series_matches_factorials():
    c = taylor_series(sin, order=9)(0.0)
    expected = [0.0, 1.0, 0.0, -1 / 6, 0.0, 1 / 120, 0.0, -1 / 5040, 0.0, 1 / 362880]
    for a, b in zip(c, expected):
        assert abs(a - b) < 1e-15
