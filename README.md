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

> Status: **0.0.1 (pre-alpha seed)**, published on [PyPI](https://pypi.org/project/taylorize/).
> The API is minimal and will change.

## Install

```bash
pip install taylorize
# or
uv add taylorize
```

## Quickstart

```python
from taylorize import derivative, taylor_series, exp, sin, ode_step

def f(x):
    return exp(sin(x))

# 5th derivative of exp(sin(x)) at 0
print(derivative(f, k=5)(0.0))          # -8.0

# Taylor coefficients of sin(x)
print(taylor_series(sin, order=9)(0.0)) # [0, 1, 0, -1/6, ...]

# One 16th-order Taylor step of the simple pendulum
def pendulum(u):
    theta, omega = u[0], u[1]
    return [omega, -sin(theta)]

step = ode_step(pendulum, order=16)
print(step([1.0, 0.0], 0.1))
```

See [`examples/basic.py`](examples/basic.py).

## Idea

The core object is a truncated power series in a nilpotent direction `h`:

```
x(h) = c[0] + c[1] h + c[2] h^2 + ... + c[N] h^N
```

Every elementary function is just a recurrence on these coefficients
(e.g. `exp`: `c[k] = (1/k) * sum(j * x[j] * c[k-j])`), and the chain rule turns
any composition into ring operations on series. The same machinery integrates
ODEs one Taylor step at a time.

## Roadmap

- Backend-agnostic coefficients (`math`, `numpy`, `mpmath`, `torch`, `jax`, `sympy`)
- Automatic interception of `math.*` / `np.*` / `torch.*` / `sp.*`
- Functions defined by ODEs (`from_ode`) and coupled systems
- Special functions (`erf`, Airy, Bessel, Jacobi, ...)
- Adaptive-step Taylor ODE solver

## License

MIT
