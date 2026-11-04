"""Minimal examples for taylorize 0.0.1.

Run with::

    uv run python examples/basic.py
"""
from taylorize import derivative, exp, sin, taylor_series


# 1) Higher-order derivative of an ordinary function.
def f(x):
    return exp(sin(x))


d5 = derivative(f, k=5)
print("f^(5)(0)            =", d5(0.0))  # expected: -8.0


# 2) Normalized Taylor coefficients of sin(x) up to order 9.
coeffs = taylor_series(sin, order=9)(0.0)
print("sin coeffs [0..9]   =", coeffs)
