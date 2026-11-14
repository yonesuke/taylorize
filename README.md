# taylorize

[![PyPI](https://img.shields.io/pypi/v/taylorize)](https://pypi.org/project/taylorize/)
[![Python versions](https://img.shields.io/pypi/pyversions/taylorize)](https://pypi.org/project/taylorize/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![publish](https://github.com/yonesuke/taylorize/actions/workflows/publish.yml/badge.svg)](https://github.com/yonesuke/taylorize/actions/workflows/publish.yml)

Lazy truncated Taylor series for **higher-order automatic differentiation**,
**arbitrary-precision Taylor-method ODE integration**, and
**non-commutative / skew formal power series**.

`taylorize` represents a formal series `sum_{k>=0} c[k] h^k` (for smooth functions,
`c[k] = x^(k)(0) / k!`) and computes coefficients **lazily**, so you can build an
expression or recurrence once and query any order afterwards.

> Status: **0.0.3 (pre-alpha)**, published on [PyPI](https://pypi.org/project/taylorize/).
> The API is evolving.

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

Exact rational expansions at regular singular points, higher-order ODEs, and
non-commutative / skew rings:

```python
from fractions import Fraction
from taylorize import bessel_j0, variable
from taylorize.functions import painleve_I

# Fuchsian expansion of Bessel J_0 at the regular singular origin x = 0
j0 = bessel_j0(variable(Fraction(0)))
print([j0.coeff(k) for k in range(5)])  # [1, 0, -1/4, 0, 1/64]

# Direct 2nd-order ODE (Painlevé I: y'' = 6y^2 + x) without auxiliary variables
y = painleve_I(variable(Fraction(0)), Fraction(1), Fraction(0))
print([y.coeff(k) for k in range(6)])  # [1, 0, 3, 1/6, 3, 1/10]
```

See [`examples/basic.py`](examples/basic.py) and
[`examples/noncommutative_and_singular.py`](examples/noncommutative_and_singular.py).

## Architecture

The core object `Taylor[T]` is a lazy formal power series over any (possibly
non-commutative or skew) coefficient ring `T`:

```
x(h) = c[0] + c[1] h + c[2] h^2 + ... + c[N] h^N
```

- **`taylorize.core`** — Pure lazy non-commutative / Ore-skew power-series ring with zero external dependencies:
  - *Level 0 (Ring)*: `+`, `-`, `*`, `rdiv` / `ldiv`, `shift`, integer `**`, `compose`, optional `derivation`
  - *Level 1 (Algebraic / Implicit)*: `Taylor.solve_algebraic` for `P(h, Y) = 0`
  - *Level 2 (Liouville)*: `diff`, `euler_diff`, `integrate`, `exp_integrate_left` / `exp_integrate_right`, `log_diff_left` / `log_diff_right`
  - *Level 3 (Holonomic / Fuchsian)*: `bind_euler_ode` for regular singular equations `I(theta) Y = RHS`
  - *Level 4 (ODEs & Recurrences)*: `bind_ode(dy, order=...)` and universal `bind_recurrence`
- **`taylorize.functions`** — Elementary and special functions (`exp`, `sin`, `tan`, `erf`, Airy, Bessel including singular `x0 = 0`, Jacobi elliptic, Weierstrass $\wp$, Painlevé I/II) declared via the differential equations they satisfy.
- **`taylorize.backends`** — Resolves 0th-order initial values across `float`/`complex` (`math`/`cmath`/`scipy.special`), exact `Fraction`, arbitrary-precision `mpmath`, symbolic `sympy`, and `numpy`.
- **`taylorize.transforms`** — Function transforms (`derivative`, `taylor_series`).
- **`taylorize.interop`** — Host-library protocol dispatch (`__array_ufunc__`, `__torch_function__`, `mpmath` import hook).

## Roadmap

- Adaptive-step Taylor ODE solver
- JAX `jvp` / jet interop

## License

MIT
