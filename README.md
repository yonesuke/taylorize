# taylorize

[![PyPI](https://img.shields.io/pypi/v/taylorize)](https://pypi.org/project/taylorize/)
[![Python versions](https://img.shields.io/pypi/pyversions/taylorize)](https://pypi.org/project/taylorize/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![publish](https://github.com/yonesuke/taylorize/actions/workflows/publish.yml/badge.svg)](https://github.com/yonesuke/taylorize/actions/workflows/publish.yml)

Lazy truncated Taylor series for **higher-order automatic differentiation** and
**arbitrary-precision Taylor-method ODE integration**.

`taylorize` represents a smooth function by its normalized Taylor coefficients
`c[k] = x^(k)(0) / k!` and computes them **lazily**, so you can build an
expression once and then ask for any order afterwards.

> Status: **0.0.2 (pre-alpha seed)**, published on [PyPI](https://pypi.org/project/taylorize/).
> The API is minimal and will change.

## Install

```bash
pip install taylorize
# or
uv add taylorize
```

## Quickstart

```python
from fractions import Fraction
from taylorize import Taylor, derivative, taylor_series, variable, exp, sin, tan
from taylorize.functions import define_ode_function

def f(x):
    return exp(sin(x))

# 5th derivative of exp(sin(x)) at 0
print(derivative(f, k=5)(0.0))          # -8.0

# Exact rational Taylor coefficients of tan(x) up to order 7
print(tan(variable(Fraction(0))).ensure(7).c)
# [Fraction(0, 1), Fraction(1, 1), Fraction(0, 1), Fraction(1, 3), ...]

# Define any function by the ODE it satisfies: y' = y(1 - y), y(0) = 1/2
logistic = define_ode_function(lambda x, y: y * (1 - y), lambda x0: 0.5)
print(taylor_series(logistic, order=5)(0.0))

# Integrate a coupled ODE system (simple pendulum: theta'' = -sin(theta))
theta = Taylor.pending(1.0)
omega = Taylor.pending(0.0)
theta.bind_ode(omega)
omega.bind_ode(-sin(theta))
print(theta.eval(0.1, order=16))
```

See [`examples/basic.py`](examples/basic.py).

## Architecture

The core object is a lazy truncated power series in a nilpotent direction `h`:

```
x(h) = c[0] + c[1] h + c[2] h^2 + ... + c[N] h^N
```

- **`taylorize.core`** — Pure lazy power-series ring (`+`, `-`, `*`, `/`), Picard integration, and `Taylor.pending` / `bind_ode` wiring. Zero knowledge of elementary functions or coefficient backends.
- **`taylorize.functions`** — Elementary and special functions (`exp`, `sin`, `tan`, `erf`, Airy, Bessel, Jacobi elliptic, Weierstrass $\wp$, Painlevé I/II) declared directly via the differential equations they satisfy (`define_ode_function`, `define_ode_system`).
- **`taylorize.backends`** — Resolves 0th-order initial values across `float`/`complex` (`math`/`cmath`/`scipy.special`), exact `Fraction`, arbitrary-precision `mpmath`, symbolic `sympy`, and `numpy`.
- **`taylorize.transforms`** — Function transforms (`derivative`, `taylor_series`).
- **`taylorize.interop`** — Host-library protocol dispatch (`__array_ufunc__`, `__torch_function__`).

## Roadmap

- Implicit / holonomic ODE layer for singular expansion points at the origin
- Adaptive-step Taylor ODE solver
- JAX `jvp` / jet interop

## License

MIT
