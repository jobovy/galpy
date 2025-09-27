import sympy as sp
import numpy as np
import time
from sympy.utilities.autowrap import ufuncify
import numba

# -----------------------------
# 1. Define expression
# -----------------------------
x, y = sp.symbols('x y')
expr = 1 / ((1 + x / 2.0) * (1 + (x / 2.0) ** 2))

# -----------------------------
# 2. Prepare functions
# -----------------------------
# a) Raw SymPy
def sympy_eval(xs):
    return [expr.subs({x: xi}).evalf() for xi in (xs)]

# b) Lambdify
f_lambdify = sp.lambdify((x,), expr, "numpy")

# c) Ufuncify (C compiled)
f_ufunc = ufuncify((x), expr)

# d) Lambdify + numba
f_numba = numba.njit()(f_lambdify)


# Use numexpr as the backend
f_numexpr = sp.lambdify(x, expr, modules='numexpr')

# -----------------------------
# 3. Generate test data
# -----------------------------
N_small = 1000      # scalar-loop style
N_large = 200_000   # large array

xs_small = np.random.rand(N_small)
ys_small = np.random.rand(N_small)

xs_large = np.random.rand(N_large)
ys_large = np.random.rand(N_large)

# -----------------------------
# 4. Benchmark: scalar loop
# -----------------------------
# print("=== Scalar loop benchmark (1000 points) ===")
# t0 = time.time(); res = sympy_eval(xs_small); t1 = time.time()
# print("SymPy subs+evalf:", t1-t0, "s")

# t0 = time.time(); res = [f_lambdify(xi) for xi in zip(xs_small)]; t1 = time.time()
# print("lambdify (numpy) loop:", t1-t0, "s")

# t0 = time.time(); res = [f_ufunc(xi) for xi in zip(xs_small)]; t1 = time.time()
# print("ufuncify loop:", t1-t0, "s")

# t0 = time.time(); res = [f_numba(xi) for xi in zip(xs_small)]; t1 = time.time()
# print("lambdify + numba loop:", t1-t0, "s")

# -----------------------------
# 5. Benchmark: large array
# -----------------------------
print("\n=== Large array benchmark (200k points) ===")
t0 = time.time(); res = f_lambdify(xs_large); t1 = time.time()
print("lambdify (numpy) vectorized:", t1-t0, "s")

t0 = time.time(); res = f_numexpr(xs_large); t1 = time.time()
print("lambdify + numexpr vectorized:", t1-t0, "s")

t0 = time.time(); res = f_ufunc(xs_large); t1 = time.time()
print("ufuncify (C compiled) vectorized:", t1-t0, "s")

t0 = time.time(); res = f_numba(xs_large); t1 = time.time()
print("lambdify + numba vectorized:", t1-t0, "s")
