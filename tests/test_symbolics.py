import sympy as sp

def func():
    r_prime = sp.symbols('r_prime', real=True, positive=True)
    # Define constants
    a = -1.08908545324446e-5
    b = -4.47725354458807e-17
    c = 3.21700694652131e-13
    d = -6.23190204929647e-10

    # Define M(r_prime) with Min
    M = (
        a * r_prime**3 +
        b * sp.Min(1221.5, r_prime)**6 +
        c * sp.Min(1221.5, r_prime)**5 +
        d * sp.Min(1221.5, r_prime)**4
    )
    return M, r_prime


# Define symbols
r = sp.symbols('r', real=True, positive=True)

M, r_prime = func()


# Reference radius for potential zero
r_ref = 0  # or another finite value if you want

# Define potential as integral from r_ref to r
Phi = -1 * sp.integrate(M / r_prime**2, (r_prime, r_ref, r))

# Phi_1 = -1 * sp.integrate(Phi * r_prime**2, (r_prime, r_ref, r))

print("Phi(r) =", Phi)
