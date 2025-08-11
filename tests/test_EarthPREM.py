import os
import sys

# Get current directory
curdir = os.getcwd()
print("Current directory:", curdir)

# Insert into sys.path if not already there
if curdir not in sys.path:
    sys.path.insert(0, curdir)


import numpy

# import matplotlib.pyplot as plt

# # import pytest

# # from galpy.actionAngle import actionAngleIsochroneApprox
# # from galpy.df import chen24spraydf, fardal15spraydf, streamdf, streamspraydf
# # from galpy.orbit import Orbit
from galpy.potential import EarthPREMPotential
from galpy.potential.EarthPREMPotential import EARTH_RADIUS_KM

# from galpy.util import conversion  # for unit conversions
# from galpy.util import coords


import sympy


def test_density():
    """test the density"""
    pot = EarthPREMPotential()

    assert numpy.fabs(pot._dens(R=0, z=0) - 13.0885) <= 1e-6, (
        f"Calculated density at (R=0, z=0) " + "is not equal to 13.0885. "
    )
    assert numpy.fabs(pot._dens(R=EARTH_RADIUS_KM - 1, z=0) - 2.6) <= 1e-6, (
        "Calculated enclosed mass at earth radius - 1 km (R=6370.0, z=0)"
        + "is not equal to 2.6. "
    )
    assert numpy.fabs(pot._dens(R=0, z=EARTH_RADIUS_KM - 1) - 2.6) <= 1e-6, (
        "Calculated enclosed mass at earth radius - 1 km (R=6370.0, z=0)"
        + "is not equal to 2.6. "
    )


def test_enclosed_mass():
    """test the enclosed mass"""
    pot = EarthPREMPotential()
    # print(f"enclosed mass at r = 0: {pot._mass(R=0, z=0)} g")
    assert pot._mass(R=0, z=0) == 0, (
        "Calculated enclosed mass at (R=0, z=0)" + "is not equal to 0. "
    )
    assert numpy.fabs(pot._mass(R=EARTH_RADIUS_KM, z=0) - 5977886716366.892) <= 1e1, (
        "Calculated enclosed mass at earth radius (R=6371.0, z=0)"
        + "is not close to 5.98e+12 g. "
    )
    assert numpy.fabs(pot._mass(R=0, z=EARTH_RADIUS_KM) - 5977886716366.892) <= 1e1, (
        "Calculated enclosed mass at earth radius (R=0, z=6371.0)"
        + "is not close to 5.98e+12 g. "
    )
    assert (
        numpy.fabs(pot._mass(R=10 * EARTH_RADIUS_KM, z=0) - 5977886716366.892) <= 1e1
    ), (
        "Calculated enclosed mass at 10 times earth radius (R=10*6371.0, z=0)"
        + "is not close to 5.98e+12 g. "
    )
    assert (
        numpy.fabs(pot._mass(R=0, z=10 * EARTH_RADIUS_KM) - 5977886716366.892) <= 1e1
    ), (
        "Calculated enclosed mass at 10 times earth radius (R=0, z=10*6371.0)"
        + "is not close to 5.98e+12 g. "
    )


def test_potential():
    """test the enclosed mass"""
    pot = EarthPREMPotential()
    # print(pot._revaluate(r=1))
    # print(pot._revaluate(r=EARTH_RADIUS_KM))
    # print(pot._revaluate(r=10 * EARTH_RADIUS_KM))
    potential_1km = -54.824980048099455
    potential_EARTH_RADIUS = -938296455.2451564
    potential_10EARTH_RADIUS = -93829645.52451564
    assert (
        numpy.fabs(pot._evaluate(R=1, z=0) - potential_1km) <= 0.1
    ), "Calculated potential at (R=1, z=0) is not right. "
    assert (
        numpy.fabs(pot._evaluate(R=EARTH_RADIUS_KM, z=0) - potential_EARTH_RADIUS) <= 1
    ), "Calculated potential at earth radius is not right. "
    assert (
        numpy.fabs(pot._evaluate(R=0, z=EARTH_RADIUS_KM) - potential_EARTH_RADIUS) <= 1
    ), "Calculated potential at earth radius is not right. "
    assert (
        numpy.fabs(
            pot._evaluate(R=10 * EARTH_RADIUS_KM, z=0) - potential_10EARTH_RADIUS
        )
        <= 1
    ), "Calculated potential at 10 times earth radius (R=10*6371.0, z=0) is not right. "
    assert (
        numpy.fabs(
            pot._evaluate(R=0, z=10 * EARTH_RADIUS_KM) - potential_10EARTH_RADIUS
        )
        <= 1
    ), "Calculated potential at 10 times earth radius (R=0, z=10*6371.0) is not right. "


def test_r2deriv():
    """test the enclosed mass"""
    pot = EarthPREMPotential()
    r_vals = [0.01, EARTH_RADIUS_KM - 1, 10 * (EARTH_RADIUS_KM - 1)]
    # d²Φ/dr² in two different expressions. Either one is good
    expr = (
        -sympy.diff(pot.rawMass_sym, pot.r, 2) / pot.r
        + 2 * 4 * sympy.pi * pot.dens_sym
        - 2 * pot.rawMass_sym / pot.r**3.0
    )
    # expr = (
    #     -sympy.diff(pot.rawMass_sym, pot.r, 2) / pot.r
    #     + 2.0 * sympy.diff(pot.rawMass_sym, pot.r, 1) / pot.r**2.0
    #     - 2 * pot.rawMass_sym / pot.r**3.0
    # )
    for r in r_vals:
        r2deriv_num = float(expr.evalf(subs={pot.r: r}))
        # print(f"r = {r} km")
        # print(f"r2deriv_num = {r2deriv_num}")
        # print(f"pot._r2deriv(r=r) = {pot._r2deriv(r=r)}")
        assert (
            numpy.fabs(pot._r2deriv(r=r) - r2deriv_num) <= 0.1
        ), f"Calculated potential at (R={r:.2e}, z=0) is not right. "


# if __name__ == "__main__":
    # test_density()
    # test_enclosed_mass()
    # test_potential()
    # test_r2deriv()
