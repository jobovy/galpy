import os
import sys

# Get current directory
curdir = os.getcwd()
print("Current directory:", curdir)

# Insert into sys.path if not already there
if curdir not in sys.path:
    sys.path.insert(0, curdir)


import numpy
import matplotlib.pyplot as plt

# import pytest

# from galpy.actionAngle import actionAngleIsochroneApprox
# from galpy.df import chen24spraydf, fardal15spraydf, streamdf, streamspraydf
# from galpy.orbit import Orbit
from galpy.potential import EarthPREMPotential

# from galpy.util import conversion  # for unit conversions
# from galpy.util import coords


import sympy


def earth_PREM_density_sym():
    # R, z = sympy.symbols('R z')
    # r2 = R**2.0 + z**2.0
    # r = sympy.sqrt(r2)
    r = sympy.Symbol("r", real=True, positive=True)
    # R = sympy.symbols('r', real=True, positive=True)
    EARTH_RADIUS_KM = 6371.0  # Earth's radius in km
    x = r / EARTH_RADIUS_KM

    rho = sympy.Piecewise(
        (13.0885 - 8.8381 * x**2, r < 1221.5),
        (
            12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3,
            (r >= 1221.5) & (r < 3480.0),
        ),
        # (
        #     7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3,
        #     (r >= 3480.0) & (r < 5701.0),
        # ),
        # (5.3197 - 1.4836 * x, (r >= 5701.0) & (r < 5771.0)),
        # (11.2494 - 8.0298 * x, (r >= 5771.0) & (r < 5971.0)),
        # (7.1089 - 3.8045 * x, (r >= 5971.0) & (r < 6151.0)),
        # (2.6910 + 0.6924 * x, (r >= 6151.0) & (r < 6346.6)),
        # (2.6910 + 0.6924 * x, (r >= 6346.6) & (r < 6356.0)),
        # (2.6, r >= 6356.0),
        (0, True),  # for r > 6371.0 and safety
    )
    return rho, r


def enclosed_mass():
    # Load symbolic density and radial variable
    dens_sym, r_sym = earth_PREM_density_sym()
    # , R_sym, z_sym

    # Define symbolic radial variable r (independent of R, z)
    r = sympy.Symbol("r", real=True, positive=True)
    r_1 = sympy.Symbol("r_1", real=True, positive=True)
    EARTH_RADIUS_KM = 6371.0
    x = r / EARTH_RADIUS_KM


    # Compute enclosed mass symbolically
    integrand = 4 * sympy.pi * r_sym**2 * dens_sym
    M_r = sympy.integrate(integrand, (r_sym, 0, r_sym), conds="none")

    return M_r.simplify(), r_sym


x = 1000.0
# Example usage:
M_sym, r_prime = enclosed_mass()

r = sympy.Symbol("r", real=True, positive=True)
# r_prime = sympy.symbols('r_prime', real=True, positive=True)

# Define symbolic enclosed mass M(r_prime)
# M = sympy.Function('M')(r_prime)  # replace with your enclosed mass expression


# integrand = M_sym / r_prime**2
# print("sympy.integrate... ")
# Phi = sympy.integrate(integrand, (r_prime, r_prime, 0))
# print("Phi(r) =", Phi)

# print("Enclosed mass M(r):")
# print(sympy.latex(M_sym))
# print("M_sym.evalf(subs=)")
# print(M_sym.evalf(subs={r: x}))
# r_sym = sympy.Symbol('r', real=True, positive=True)

# # Create a NumPy-compatible function
# M_func = sympy.lambdify(r_sym, M_sym, modules=["numpy"])

# print(M_func(6371.0))        # Earth surface
# # print(M_func(numpy.linspace(0, 6371.0, 100)))  # Array of values
# # Create an array of radius values from 0 to Earth's radius (6371 km)
# r_values = numpy.linspace(0, 6371.0, 1000)

# # Evaluate enclosed mass at each radius (unit: g/cm^3 × km^3)
# mass_values = M_func(r_values)

# # Convert to kilograms if desired
# mass_kg = numpy.array(mass_values) * 1e12  # 1 g/cm³ * km³ = 1e12 kg

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(r_values, mass_kg, color='blue')
# plt.xlabel("Radius $r$ (km)")
# plt.ylabel("Enclosed Mass $M(r)$ (kg)")
# plt.title("Enclosed Mass vs. Radius in the Earth (PREM model)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


pot = EarthPREMPotential()

# print('pot._rdens')
# print(pot._rdens(r=100000))
# print(pot._rdens(r=6370.9))

# # def _rforce(self, r, t=0.0):
# #     return -self._rawmass(r) / r**2

# # def _r2deriv(self, r, t=0.0):
# #     return -2 * self._rawmass(r) / r**3.0 + 4.0 * numpy.pi * self._rawdens(r)
# print("pot._rforce")
# print(pot._rforce(r=1000.0))
# print(pot._rforce(r=0.00001))

# print("pot._rforce_sym")
# print(pot._rforce_sym(r=1000.0))
# print(pot._rforce_sym(r=0.00001))

# print('pot._r2deriv')
# print(pot._r2deriv(r=1000.0))
# print(pot._r2deriv(r=0.0000001))

# print('pot._r2deriv_sym')
# print(pot._r2deriv_sym(r=1000.0))
# print(pot._r2deriv_sym(r=0.0000001))

def test_rawmass():
    pot = EarthPREMPotential()
    print(pot._mass(R=pot.R))
    assert pot._mass(R=0.) == 0
