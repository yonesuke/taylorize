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

Use your existing **NumPy** or **mpmath** functions directly—no need to import `sin` or `exp` from `taylorize`:

```python
import numpy as np
from taylorize import Taylor, derivative, taylor_series

# 5th derivative of np.exp(np.sin(x)) at 0
print(derivative(lambda x: np.exp(np.sin(x)), k=5)(0.0))  # -8.0

# Normalized Taylor coefficients of np.sin(x) up to order 9
print(taylor_series(np.sin, order=9)(0.0))  # [0.0, 1.0, 0.0, -1/6, ...]

# Coupled ODE system (simple pendulum: theta'' = -sin(theta))
theta, omega = Taylor.pending(1.0), Taylor.pending(0.0)
theta.bind_ode(omega)
omega.bind_ode(-np.sin(theta))
print(theta.eval(0.1, order=16))
```

Arbitrary-precision Taylor expansions with **mpmath**:

```python
from mpmath import mp
from taylorize import variable

mp.dps = 50
x = variable(mp.mpf(0))

# 60th-order Taylor evaluation at x = 0.5 in 50-digit precision
val = mp.exp(mp.sin(x)).eval(mp.mpf("0.5"), order=60)
print(abs(val - mp.exp(mp.sin(mp.mpf("0.5")))))  # < 1e-47
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
