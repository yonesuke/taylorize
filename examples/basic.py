"""Minimal examples for taylorize 0.0.2.

Run with::

    uv run python examples/basic.py
"""
import numpy as np
from mpmath import mp

from taylorize import derivative, taylor_series, variable

# 1) Higher-order derivative using NumPy functions directly.
d5 = derivative(lambda x: np.exp(np.sin(x)), k=5)
print("f^(5)(0)            =", d5(0.0))  # expected: -8.0

# 2) Normalized Taylor coefficients of np.sin(x) up to order 9.
coeffs = taylor_series(np.sin, order=9)(0.0)
print("sin coeffs [0..9]   =", coeffs)

# 3) 50-digit arbitrary-precision expansion with mpmath.
mp.dps = 50
x = variable(mp.mpf(0))
err = abs(mp.exp(mp.sin(x)).eval(mp.mpf("0.5"), order=60) - mp.exp(mp.sin(mp.mpf("0.5"))))
print("50-digit mpmath err =", float(err))
