"""Minimal examples for taylorize 0.0.1.

Run with::

    uv run python examples/basic.py
"""
from taylorize import derivative, exp, ode_step, sin, taylor_series


# 1) Higher-order derivative of an ordinary function.
def f(x):
    return exp(sin(x))


d5 = derivative(f, k=5)
print("f^(5)(0)            =", d5(0.0))  # expected: -8.0


# 2) Normalized Taylor coefficients of sin(x) up to order 9.
coeffs = taylor_series(sin, order=9)(0.0)
print("sin coeffs [0..9]   =", coeffs)


# 3) One 16th-order Taylor step of the simple pendulum.
def pendulum(u):
    theta, omega = u[0], u[1]
    return [omega, -sin(theta)]


step16 = ode_step(pendulum, order=16)
print("pendulum dt=0.1     ->", step16([1.0, 0.0], 0.1))
