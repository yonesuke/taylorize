"""Advanced examples for taylorize 0.1.0:
non-commutative Lie / Dyson expansions, skew pseudo-differential operators,
Fuchsian singular-point expansions, and direct higher-order ODEs.

Run with::

    uv run python examples/noncommutative_and_singular.py
"""
from fractions import Fraction
import sympy as sp

from taylorize import Taylor, bessel_j0, variable
from taylorize.functions import painleve_I

# 1) Exact Bessel J_0(x) at the regular singular origin x = 0 (Level 3: bind_euler_ode)
j0 = bessel_j0(variable(Fraction(0)))
print("1) Bessel J_0(0) coeffs [0..6]:", [j0.coeff(k) for k in range(7)])

# 2) Painleve I y'' = 6y^2 + x via direct 2nd-order bind_ode (Level 4)
y_pi = painleve_I(variable(Fraction(0)), Fraction(1), Fraction(0))
print("2) Painleve I coeffs [0..5]:   ", [y_pi.coeff(k) for k in range(6)])

# 3) Non-commutative 2x2 matrix Dyson expansion Y' = (A + h B) * Y (Level 2)
A = sp.Matrix([[0, 1], [-1, 0]])
B = sp.Matrix([[0, 0], [1, 0]])
eye2 = sp.eye(2)
Y_dyson = Taylor([A, B]).exp_integrate_left(eye2)
print("3) Matrix Dyson c[2]:\n", Y_dyson.coeff(2))

# 4) KdV pseudo-differential operator (1 + d^-2 u)^{1/2} via Skew Level 0 + Level 1
x = sp.Symbol("x")
u = sp.Function("u")(x)
deriv = lambda expr: sp.diff(expr, x)

d_inv2 = Taylor([sp.Integer(0), sp.Integer(0), sp.Integer(1)], derivation=deriv)
L = Taylor([sp.Integer(1)], derivation=deriv) + d_inv2 * Taylor([u], derivation=deriv)
sqrt_L = Taylor.solve_algebraic(
    lambda y: y * y - L,
    y0=sp.Integer(1),
    inv_jacobian=lambda d: d / 2,
    derivation=deriv,
)
print("4) KdV (1 + d^-2 u)^(1/2) coeffs [0..5]:")
for k in range(6):
    print(f"   d^-{k}: {sqrt_L.coeff(k)}")
