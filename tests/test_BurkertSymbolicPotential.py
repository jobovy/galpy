# if there is ModuleNotFoundError
import os
import sys

# Get current directory
curdir = os.getcwd()
print("Current directory:", curdir)

# Insert into sys.path if not already there
if curdir not in sys.path:
    sys.path.insert(0, curdir)

import numpy

from galpy.util._optional_deps import _SYMPY_LOADED
from galpy.util import conversion

from galpy.potential.SymbolicSphericalPotential import SymbolicSphericalPotential

if _SYMPY_LOADED:
    import sympy

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
        rho_expr = 1 / ((1 + self.r / self.a) * (1 + (self.r / self.a) ** 2))
        # # Make it a function of r
        # dens = sympy.Lambda(self.r, rho_expr)

        SymbolicSphericalPotential.__init__(self, dens=rho_expr)

        if normalize or (
            isinstance(normalize, (int, float)) and not isinstance(normalize, bool)
        ):  # pragma: no cover
            self.normalize(normalize)
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        return None

    # def _revaluate(self, r, t=0.0):
    #     """Potential as a function of r and time"""
    #     x = r / self.a
    #     return (
    #         -(self.a**2.0)
    #         * numpy.pi
    #         * (
    #             -numpy.pi / x
    #             + 2.0 * (1.0 / x + 1) * numpy.arctan(1 / x)
    #             + (1.0 / x + 1) * numpy.log((1.0 + 1.0 / x) ** 2.0 / (1.0 + 1 / x**2.0))
    #             + special.xlogy(2.0 / x, 1.0 + x**2.0)
    #         )
    #     )

    # # Previous way, not stable as r -> infty
    # # return -self.a**2.*numpy.pi/x*(-numpy.pi+2.*(1.+x)*numpy.arctan(1/x)
    # #                                +2.*(1.+x)*numpy.log(1.+x)
    # #                                +(1.-x)*numpy.log(1.+x**2.))

    # def _rforce(self, r, t=0.0):
    #     x = r / self.a
    #     return (
    #         self.a
    #         * numpy.pi
    #         / x**2.0
    #         * (
    #             numpy.pi
    #             - 2.0 * numpy.arctan(1.0 / x)
    #             - 2.0 * numpy.log(1.0 + x)
    #             - numpy.log(1.0 + x**2.0)
    #         )
    #     )

    # def _r2deriv(self, r, t=0.0):
    #     x = r / self.a
    #     return (
    #         4.0 * numpy.pi / (1.0 + x**2.0) / (1.0 + x)
    #         + 2.0 * self._rforce(r) / x / self.a
    #     )

    # def _rdens(self, r, t=0.0):
    #     x = r / self.a
    #     return 1.0 / (1.0 + x) / (1.0 + x**2.0)

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


import pytest

# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy
# import sympy

from galpy.potential.BurkertPotential import BurkertPotential

def test_this():
    pot_sym = BurkertSymbolicPotential(amp=1.0, a=1.0)

    print(pot_sym._rdens(0.0))

@pytest.fixture(
    params=[{"amp": 1.0, "a": 1.0}, {"amp": 2.0, "a": 1.5}, {"amp": 3.5, "a": 0.5}]
)
def params(request):
    return request.param


@pytest.fixture
def pots(params):
    pot_num = BurkertPotential(**params)
    pot_sym = BurkertSymbolicPotential(**params)
    return pot_num, pot_sym


def test_rdens_limit_r_zero(pots, params):
    num, sym = pots
    v_num = num._rdens(0.0)
    v_sym = sym._rdens(0.0)
    expected = params["amp"]
    assert v_num == pytest.approx(expected, rel=1e-12)
    assert v_sym == pytest.approx(expected, rel=1e-12)

@pytest.mark.parametrize("r", [1e2, 1e3, 1e4])
def test_rdens_limit_r_infinity(pots, params, r):
    num, sym = pots
    a, amp = params["a"], params["amp"]

    # multiply by r^3, should approach amp * a^3
    v_num = num._rdens(r) * r**3
    v_sym = sym._rdens(r) * r**3
    expected = amp * a**3

    assert v_num == pytest.approx(expected, rel=1e-6)
    assert v_sym == pytest.approx(expected, rel=1e-6)


# def test_density():
#     """test the density"""
#     pot = EarthPREMPotential()

#     assert numpy.fabs(pot._dens(R=0, z=0) - 13.0885) <= 1e-6, (
#         f"Calculated density at (R=0, z=0) " + "is not equal to 13.0885. "
#     )
#     assert numpy.fabs(pot._dens(R=EARTH_RADIUS_KM - 1, z=0) - 2.6) <= 1e-6, (
#         "Calculated enclosed mass at earth radius - 1 km (R=6370.0, z=0)"
#         + "is not equal to 2.6. "
#     )
#     assert numpy.fabs(pot._dens(R=0, z=EARTH_RADIUS_KM - 1) - 2.6) <= 1e-6, (
#         "Calculated enclosed mass at earth radius - 1 km (R=6370.0, z=0)"
#         + "is not equal to 2.6. "
#     )


# def test_enclosed_mass():
#     """test the enclosed mass"""
#     pot = EarthPREMPotential()
#     # print(f"enclosed mass at r = 0: {pot._mass(R=0, z=0)} g")
#     assert pot._mass(R=0, z=0) == 0, (
#         "Calculated enclosed mass at (R=0, z=0)" + "is not equal to 0. "
#     )
#     assert numpy.fabs(pot._mass(R=EARTH_RADIUS_KM, z=0) - 5977886716366.892) <= 1e1, (
#         "Calculated enclosed mass at earth radius (R=6371.0, z=0)"
#         + "is not close to 5.98e+12 g. "
#     )
#     assert numpy.fabs(pot._mass(R=0, z=EARTH_RADIUS_KM) - 5977886716366.892) <= 1e1, (
#         "Calculated enclosed mass at earth radius (R=0, z=6371.0)"
#         + "is not close to 5.98e+12 g. "
#     )
#     assert (
#         numpy.fabs(pot._mass(R=10 * EARTH_RADIUS_KM, z=0) - 5977886716366.892) <= 1e1
#     ), (
#         "Calculated enclosed mass at 10 times earth radius (R=10*6371.0, z=0)"
#         + "is not close to 5.98e+12 g. "
#     )
#     assert (
#         numpy.fabs(pot._mass(R=0, z=10 * EARTH_RADIUS_KM) - 5977886716366.892) <= 1e1
#     ), (
#         "Calculated enclosed mass at 10 times earth radius (R=0, z=10*6371.0)"
#         + "is not close to 5.98e+12 g. "
#     )


# def test_potential():
#     """test the calculation of potential"""
#     pot = EarthPREMPotential()
#     r_vals = [EARTH_RADIUS_KM, 10 * EARTH_RADIUS_KM]
#     potential_EARTH_RADIUS = -938296455.2451564
#     potential_10EARTH_RADIUS = -93829645.52451564
#     pot_vals = [potential_EARTH_RADIUS, potential_10EARTH_RADIUS]
#     for i, r in enumerate(r_vals):
#         assert (
#             numpy.fabs(pot._evaluate(R=r, z=0) - pot_vals[i]) <= 1
#         ), f"Calculated potential at (R={r:.2e}, z=0) is not right. "


# def test_r2deriv():
#     """test the calculation of r2deriv"""
#     pot = EarthPREMPotential()
#     r_vals = [0.01, EARTH_RADIUS_KM - 1, 10 * (EARTH_RADIUS_KM - 1)]
#     # d²Φ/dr² in two different expressions. Either one is good
#     expr = (
#         -sympy.diff(pot.rawMass, pot.r, 2) / pot.r
#         + 2 * 4 * sympy.pi * pot.dens
#         - 2 * pot.rawMass / pot.r**3.0
#     )
#     expr = (
#         -sympy.diff(pot.rawMass, pot.r, 2) / pot.r
#         + 2.0 * sympy.diff(pot.rawMass, pot.r, 1) / pot.r**2.0
#         - 2 * pot.rawMass / pot.r**3.0
#     )
#     for r in r_vals:
#         # r2deriv obtained by manually doing the differential
#         r2deriv_val = float(expr.evalf(subs={pot.r: r}))
#         assert (
#             numpy.fabs(pot._r2deriv(r=r) - r2deriv_val) <= 0.1
#         ), f"Calculated potential at (R={r:.2e}, z=0) is not right. "


# def efficiency_test_r2deriv():
#     import time

#     """test the efficiency of symbolic calculation"""
#     pot = EarthPREMPotential()
#     r_vals = [0.01, EARTH_RADIUS_KM - 1, 10 * (EARTH_RADIUS_KM - 1)]
#     for r in r_vals:
#         t0 = time.time()
#         for i in range(100):
#             val = pot._r2deriv(r=r)
#         t1 = time.time()
#         print(f"Symbolic calculation time: {(t1 - t0) / 100:.6f} s")


# def plot_potential_num_sym():
    # """
    # make plots of earth potential with symbolic and numeric methods
    # """
    # ballRadius = EARTH_RADIUS_KM
    # rho_0 = 5.52
    # amp = 2 * numpy.pi * rho_0 / 3.0
    # earth_approx = HomogeneousSpherePotential(
    #     amp=amp,
    #     R=ballRadius,
    # )
    # # print(f"earth_approx.R = {earth_approx.R}")
    # extend = 3 * EARTH_RADIUS_KM
    # num_grid = 100
    # R_vals = numpy.linspace(0.01, extend, num=num_grid)
    # z_val = 0

    # earthPREM_sym = EarthPREMPotential()
    # show_plot = True
    # if show_plot:
    #     # font size
    #     plt.rc("font", size=11)  # Default text
    #     plt.rc("legend", fontsize=11)  # Legend

    #     fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
    #     gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

    #     # # fix the margins
    #     # fig.subplots_adjust(
    #     #     top=0.8, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
    #     # )

    #     ax_dens = fig.add_subplot(gs[0, 0])

    #     ax_dens.plot(
    #         R_vals,
    #         # solidBall._dens(R=r_vals, z=zero_vals),
    #         numpy.array(list(map(lambda R: earthPREM_sym._dens(R=R, z=z_val), R_vals))),
    #         # label="by SymbolicSphericalPotential",
    #         color="tab:orange",
    #         alpha=1,
    #         linestyle="--",
    #     )
    #     ax_dens.set_xlabel("radius (km)")
    #     ax_dens.set_ylabel(f"Density (g/cm^3)")

    #     #############################################################################
    #     # put a mark of script information on the figure
    #     # Get the script name and path automatically
    #     script_path = os.path.abspath(__file__)

    #     # Add the annotation to the figure
    #     plt.annotate(
    #         f"Generated by: {script_path}",
    #         xy=(0.02, 0.02),
    #         xycoords="figure fraction",
    #         fontsize=3,
    #         color="gray",
    #     )
    #     plt.show()

    #     fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

    #     gs = gridspec.GridSpec(nrows=2, ncols=2)  # create grid for multiple figures

    #     # to specify heights and widths of subfigures
    #     # width_ratios = [1, 1]
    #     # height_ratios = [1]
    #     # gs = gridspec.GridSpec(nrows=1, ncols=2, \
    #     #   width_ratios=width_ratios, height_ratios=height_ratios)  # create grid for multiple figures

    #     # fix the margins
    #     fig.subplots_adjust(
    #         top=0.8, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
    #     )

    #     ax_dens = fig.add_subplot(gs[0, 0])
    #     ax_pot = fig.add_subplot(gs[0, 1], sharex=ax_dens)  #
    #     ax_R2deriv = fig.add_subplot(gs[1, 0], sharex=ax_dens)  #
    #     ax_Rforce = fig.add_subplot(gs[1, 1], sharex=ax_dens)  #
    #     ax_list = [ax_dens, ax_pot, ax_R2deriv, ax_Rforce]
    #     ax_dens.plot(
    #         R_vals,
    #         # solidBall._dens(R=r_vals, z=zero_vals),
    #         amp
    #         * numpy.array(
    #             list(map(lambda R: earth_approx._dens(R=R, z=z_val), R_vals))
    #         ),
    #         label="approximated by HomogeneousSpherePotential",
    #         color="tab:blue",
    #         alpha=1,
    #         linestyle="-",
    #     )
    #     ax_dens.plot(
    #         R_vals,
    #         # solidBall._dens(R=r_vals, z=zero_vals),
    #         numpy.array(list(map(lambda R: earthPREM_sym._dens(R=R, z=z_val), R_vals))),
    #         label="by SymbolicSphericalPotential",
    #         color="tab:orange",
    #         alpha=1,
    #         linestyle="--",
    #     )

    #     ax_dens.set_xlabel("R")
    #     ax_dens.set_ylabel(f"Density (z={z_val})")

    #     ax_pot.plot(
    #         R_vals,
    #         amp
    #         * numpy.array(
    #             list(map(lambda R: earth_approx._evaluate(R=R, z=z_val), R_vals))
    #         ),
    #         label="approximated by HomogeneousSpherePotential",
    #         color="tab:blue",
    #         alpha=1,
    #         linestyle="-",
    #     )
    #     ax_pot.plot(
    #         R_vals,
    #         # solidBall._evaluate(R=r_vals, z=zero_vals),
    #         numpy.array(
    #             list(map(lambda R: earthPREM_sym._evaluate(R=R, z=z_val), R_vals))
    #         ),
    #         label="by SymbolicSphericalPotential",
    #         color="tab:orange",
    #         alpha=1,
    #         linestyle="--",
    #     )
    #     ax_pot.set_xlabel("R")
    #     ax_pot.set_ylabel(f"Potential (z={z_val})")

    #     ax_R2deriv.plot(
    #         R_vals,
    #         amp
    #         * numpy.array(
    #             list(map(lambda R: earth_approx._R2deriv(R=R, z=z_val), R_vals))
    #         ),
    #         label="approximated by HomogeneousSpherePotential",
    #         color="tab:blue",
    #         alpha=1,
    #         linestyle="-",
    #     )
    #     ax_R2deriv.plot(
    #         R_vals,
    #         # solidBall._evaluate(R=r_vals, z=zero_vals),
    #         numpy.array(
    #             list(map(lambda R: earthPREM_sym._R2deriv(R=R, z=z_val), R_vals))
    #         ),
    #         label="by SymbolicSphericalPotential",
    #         color="tab:orange",
    #         alpha=1,
    #         linestyle="--",
    #     )

    #     ax_R2deriv.set_xlabel("R")
    #     ax_R2deriv.set_ylabel(f"R2deriv (z={z_val})")

    #     ax_Rforce.plot(
    #         R_vals,
    #         amp
    #         * numpy.array(
    #             list(map(lambda R: earth_approx._Rforce(R=R, z=z_val), R_vals))
    #         ),
    #         label="approximated by HomogeneousSpherePotential",
    #         color="tab:blue",
    #         alpha=1,
    #         linestyle="-",
    #     )
    #     ax_Rforce.plot(
    #         R_vals,
    #         # solidBall._evaluate(R=r_vals, z=zero_vals),
    #         numpy.array(
    #             list(map(lambda R: earthPREM_sym._Rforce(R=R, z=z_val), R_vals))
    #         ),
    #         label="by SymbolicSphericalPotential",
    #         color="tab:orange",
    #         alpha=1,
    #         linestyle="--",
    #     )

    #     ax_Rforce.set_xlabel("R")
    #     ax_Rforce.set_ylabel(f"Rforce (z={z_val})")

    #     ax_list[0].legend(
    #         # fontsize=8,
    #         bbox_to_anchor=(1.5, 1.8),
    #         # ha="center",
    #         # va="bottom",
    #     )
    #     # set grid and put figure index
    #     for i, ax in enumerate(ax_list):
    #         ax.grid()

    #         # xleft, xright = ax.get_xlim()
    #         # ybottom, ytop = ax.get_ylim()
    #         ax.text(
    #             -0.013,
    #             1.02,
    #             s="(" + chr(i + ord("a")) + ")",
    #             transform=ax.transAxes,
    #             ha="right",
    #             va="bottom",
    #             color="k",
    #         )

    #     #############################################################################
    #     # put a mark of script information on the figure
    #     # Get the script name and path automatically
    #     script_path = os.path.abspath(__file__)

    #     # Add the annotation to the figure
    #     plt.annotate(
    #         f"Generated by: {script_path}",
    #         xy=(0.02, 0.02),
    #         xycoords="figure fraction",
    #         fontsize=3,
    #         color="gray",
    #     )
    #     # #############################################################################

    #     # plt.tight_layout()
    #     # plt.savefig('example figure - one-column.png', transparent=False)
    #     plt.show()


# if __name__ == "__main__":
