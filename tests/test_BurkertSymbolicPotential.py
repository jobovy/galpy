# if there is ModuleNotFoundError
# $env:PYTHONPATH = "N:\Yu0702\galpy;$env:PYTHONPATH"

import numpy

from galpy.potential.SymbolicSphericalPotential import SymbolicSphericalPotential
from galpy.util import conversion
from galpy.util._optional_deps import _SYMPY_LOADED

if _SYMPY_LOADED:
    import sympy


import pytest

from galpy.potential.BurkertPotential import BurkertPotential

# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt


###############################################################################
#   BurkertSymbolicPotential.py: SymbolicPotential with a Burkert density
###############################################################################
class BurkertSymbolicPotential(SymbolicSphericalPotential):
    """BurkertSymbolicPotential.py: Potential with a Burkert density, implemented with sympy.

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{(1+r/a)\\,(1+[r/a]^2)}

    """

    def __init__(self, amp=1.0, a=2.0, normalize=False, ro=None, vo=None):
        """
        Initialize a Burkert-density potential [1]_.

        Parameters
        ----------
        amp : float or Quantity
            Amplitude to be applied to the potential. Can be a Quantity with units of mass density or Gxmass density.
        a : float or Quantity
            Scale radius.
        normalize : bool or float, optional
            If True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1. Default is False.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2025-08-30 - Implemented based on BurkertPotential (Jo Bovy 2020) using SymbolicSphericalPotential - Yuzhe Zhang (Uni Mainz)

        References
        ----------
        .. [1] Burkert (1995), Astrophysical Journal, 447, L25. ADS: https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B.
        """

        a = conversion.parse_length(a, ro=ro, vo=vo)
        self.a = a
        self._scale = self.a

        # define sympy variable radius r
        self.r = sympy.Symbol("r", real=True)
        # Define rho(r)
        amp = sympy.Rational(amp)
        self.a = sympy.Rational(self.a)
        rho_expr = 1 / ((1 + self.r / self.a) * (1 + (self.r / self.a) ** 2))
        # # Make it a function of r
        # dens = sympy.Lambda(self.r, rho_expr)

        SymbolicSphericalPotential.__init__(self, dens=rho_expr, amp=amp)

        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        return None

    def _surfdens(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        x = r / self.a
        Rpa = numpy.sqrt(R**2.0 + self.a**2.0)
        Rma = numpy.sqrt(R**2.0 - self.a**2.0 + 0j)
        if Rma == 0:
            za = z / self.a
            return (
                self.a**2.0
                / 2.0
                * (
                    (
                        2.0
                        - 2.0 * numpy.sqrt(za**2.0 + 1)
                        + numpy.sqrt(2.0) * za * numpy.arctan(za / numpy.sqrt(2.0))
                    )
                    / z
                    + numpy.sqrt(2 * za**2.0 + 2.0)
                    * numpy.arctanh(za / numpy.sqrt(2.0 * (za**2.0 + 1)))
                    / numpy.sqrt(self.a**2.0 + z**2.0)
                )
            )
        else:
            return (
                self.a**2.0
                * (
                    numpy.arctan(z / x / Rma) / Rma
                    + numpy.arctanh(z / x / Rpa) / Rpa
                    - numpy.arctan(z / Rma) / Rma
                    + numpy.arctan(z / Rpa) / Rpa
                ).real
            )


@pytest.fixture(
    params=[
        {"amp": 2.1341897, "a": 1.13124},
        {"amp": 1.0, "a": 1.0},
    ]
)  #
def params(request):
    return request.param


@pytest.fixture
def pots(params):
    pot_num = BurkertPotential(**params)
    pot_sym = BurkertSymbolicPotential(**params)
    return pot_num, pot_sym


@pytest.mark.parametrize("r", numpy.logspace(-4, 10, num=4))
def test_revaluate(pots, r):
    num, sym = pots

    expected = num._revaluate(r)
    v_sym = sym._revaluate(r)

    assert v_sym == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("r", numpy.logspace(-1, 10, num=4))
def test_rforce(pots, r):
    num, sym = pots

    expected = num._rforce(r)
    v_sym = sym._rforce(r)

    assert v_sym == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("r", numpy.logspace(-3, 10, num=4))
def test_r2deriv(pots, r):
    num, sym = pots

    expected = num._r2deriv(r)
    v_sym = sym._r2deriv(r)

    assert v_sym == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("r", [0, 1e-3, 1e-1, 1e1, 1e2, 1e6, 1e8, 1e10])
def test_rdens(pots, r):
    num, sym = pots

    expected = num._rdens(r)
    v_sym = sym._rdens(r)

    assert v_sym == pytest.approx(expected, rel=1e-12)


# def test_r():
#     """test and compare the efficiency of numerical / symbolic calculation"""
#     import time
#     # "amp": 2.1341897, "a": 1.13124
#     # pot_num = BurkertPotential(amp =2.1341897, a=1.13124)
#     pot_sym = BurkertSymbolicPotential(amp =2.1341897, a=1.13124)
#     r_vals = numpy.logspace(0, 10, 2)

#     print(f"\n")
#     # t0 = time.time()
#     # val = pot_num._revaluate(r=r_vals)
#     # val = pot_num._rforce(r=r_vals)
#     # val = pot_num._r2deriv(r=r_vals)
#     # val = pot_num._rdens(r=r_vals)
#     # t1 = time.time()
#     # print(f"Numerical vectorized calculation time: {(t1 - t0):.6f} s")

#     t0 = time.time()
#     val = pot_sym._revaluate(r=r_vals)
#     val = pot_sym._rforce(r=r_vals)
#     val = pot_sym._r2deriv(r=r_vals)
#     val = pot_sym._rdens(r=r_vals)
#     t1 = time.time()
#     print(f"Symbolic vectorized evaluation time: {t1 - t0:.4f} seconds")


def _test_efficiency():
    """test and compare the efficiency of numerical / symbolic calculation"""
    import time

    pot_num = BurkertPotential(amp=2.1341897, a=1.13124)
    pot_sym = BurkertSymbolicPotential(amp=2.1341897, a=1.13124, backend="numpy")
    r_vals = numpy.logspace(0, 10, 1_000_000)

    print(f"\n")
    t0 = time.time()
    val = pot_num._revaluate(r=r_vals)
    val = pot_num._rforce(r=r_vals)
    val = pot_num._r2deriv(r=r_vals)
    val = pot_num._rdens(r=r_vals)
    t1 = time.time()
    print(f"Numerical vectorized calculation time: {(t1 - t0):.6f} s")

    t0 = time.time()
    val = pot_sym._revaluate(r=r_vals)
    val = pot_sym._rforce(r=r_vals)
    val = pot_sym._r2deriv(r=r_vals)
    val = pot_sym._rdens(r=r_vals)
    t1 = time.time()
    print(f"Symbolic vectorized evaluation time: {t1 - t0:.4f} seconds")

    t0 = time.time()
    for r in r_vals:
        val = pot_num._revaluate(r=r)
        val = pot_num._rforce(r=r)
        val = pot_num._r2deriv(r=r)
        val = pot_num._rdens(r=r)
    t1 = time.time()
    print(f"Numerical loop evaluation time: {t1 - t0:.4f} seconds")

    t0 = time.time()
    for r in r_vals:
        val = pot_sym._revaluate(r=r)
        val = pot_sym._rforce(r=r)
        val = pot_sym._r2deriv(r=r)
        val = pot_sym._rdens(r=r)
    t1 = time.time()
    print(f"Symbolic loop evaluation time: {t1 - t0:.4f} seconds")


# if __name__ == "__main__":
