# # if there is ModuleNotFoundError: No module named 'galpy.potential', use:
# import os
# import sys

# # Get current directory
# curdir = os.getcwd()
# # print("Current directory:", curdir)

# # Insert into sys.path if not already there
# if curdir not in sys.path:
#     sys.path.insert(0, curdir)

import numpy
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sympy

from galpy.potential.SymbolicSphericalPotential import SymbolicSphericalPotential
from galpy.potential import EarthPREMPotential
from galpy.potential.EarthPREMPotential import EARTH_RADIUS_KM
from galpy.potential.HomogeneousSpherePotential import HomogeneousSpherePotential


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
    """test the calculation of potential"""
    pot = EarthPREMPotential()
    r_vals = [EARTH_RADIUS_KM, 10 * EARTH_RADIUS_KM]
    potential_EARTH_RADIUS = -938296455.2451564
    potential_10EARTH_RADIUS = -93829645.52451564
    pot_vals = [potential_EARTH_RADIUS, potential_10EARTH_RADIUS]
    for i, r in enumerate(r_vals) :
        assert (
            numpy.fabs(pot._evaluate(R=r, z=0) - pot_vals[i]) <= 1
        ), f"Calculated potential at (R={r:.2e}, z=0) is not right. "


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


def efficiency_test_r2deriv():
    """test the efficiency of symbolic calculation"""
    pot = EarthPREMPotential()
    r_vals = [0.01, EARTH_RADIUS_KM - 1, 10 * (EARTH_RADIUS_KM - 1)]
    for r in r_vals:
        t0 = time.time()
        for i in range(100):
            val = pot._r2deriv(r=r)
        t1 = time.time()
        print(f"Symbolic calculation time: {(t1 - t0) / 100:.6f} s")


def plot_earthPotential_num_sym():
    """
    make plots of earth potential with symbolic and numeric methods
    """
    ballRadius = EARTH_RADIUS_KM
    rho_0 = 5.52
    amp = 2 * numpy.pi * rho_0 / 3.0
    earth_approx = HomogeneousSpherePotential(
        amp=amp,
        R=ballRadius,
    )
    # print(f"earth_approx.R = {earth_approx.R}")
    extend = 3 * EARTH_RADIUS_KM
    num_grid = 100
    R_vals = numpy.linspace(0.01, extend, num=num_grid)
    z_val = 0

    earthPREM_sym = EarthPREMPotential()
    show_plot = True
    if show_plot:
        # font size
        plt.rc("font", size=11)  # Default text
        plt.rc("legend", fontsize=11)  # Legend

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure
        gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

        # # fix the margins
        # fig.subplots_adjust(
        #     top=0.8, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
        # )

        ax_dens = fig.add_subplot(gs[0, 0])

        ax_dens.plot(
            R_vals,
            # solidBall._dens(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: earthPREM_sym._dens(R=R, z=z_val), R_vals))),
            # label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )
        ax_dens.set_xlabel("radius (km)")
        ax_dens.set_ylabel(f"Denisty (g/cm^3)")

        #############################################################################
        # put a mark of script information on the figure
        # Get the script name and path automatically
        script_path = os.path.abspath(__file__)

        # Add the annotation to the figure
        plt.annotate(
            f"Generated by: {script_path}",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=3,
            color="gray",
        )
        plt.show()

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

        gs = gridspec.GridSpec(nrows=2, ncols=2)  # create grid for multiple figures

        # to specify heights and widths of subfigures
        # width_ratios = [1, 1]
        # height_ratios = [1]
        # gs = gridspec.GridSpec(nrows=1, ncols=2, \
        #   width_ratios=width_ratios, height_ratios=height_ratios)  # create grid for multiple figures

        # fix the margins
        fig.subplots_adjust(
            top=0.8, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
        )

        ax_dens = fig.add_subplot(gs[0, 0])
        ax_pot = fig.add_subplot(gs[0, 1], sharex=ax_dens)  #
        ax_R2deriv = fig.add_subplot(gs[1, 0], sharex=ax_dens)  #
        ax_Rforce = fig.add_subplot(gs[1, 1], sharex=ax_dens)  #
        ax_list = [ax_dens, ax_pot, ax_R2deriv, ax_Rforce]
        ax_dens.plot(
            R_vals,
            # solidBall._dens(R=r_vals, z=zero_vals),
            amp
            * numpy.array(
                list(map(lambda R: earth_approx._dens(R=R, z=z_val), R_vals))
            ),
            label="approximated by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_dens.plot(
            R_vals,
            # solidBall._dens(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: earthPREM_sym._dens(R=R, z=z_val), R_vals))),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_dens.set_xlabel("R")
        ax_dens.set_ylabel(f"Denisty (z={z_val})")

        ax_pot.plot(
            R_vals,
            amp
            * numpy.array(
                list(map(lambda R: earth_approx._evaluate(R=R, z=z_val), R_vals))
            ),
            label="approximated by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_pot.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(
                list(map(lambda R: earthPREM_sym._evaluate(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )
        ax_pot.set_xlabel("R")
        ax_pot.set_ylabel(f"Potential (z={z_val})")

        ax_R2deriv.plot(
            R_vals,
            amp
            * numpy.array(
                list(map(lambda R: earth_approx._R2deriv(R=R, z=z_val), R_vals))
            ),
            label="approximated by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_R2deriv.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(
                list(map(lambda R: earthPREM_sym._R2deriv(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_R2deriv.set_xlabel("R")
        ax_R2deriv.set_ylabel(f"R2deriv (z={z_val})")

        ax_Rforce.plot(
            R_vals,
            amp
            * numpy.array(
                list(map(lambda R: earth_approx._Rforce(R=R, z=z_val), R_vals))
            ),
            label="approximated by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_Rforce.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(
                list(map(lambda R: earthPREM_sym._Rforce(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_Rforce.set_xlabel("R")
        ax_Rforce.set_ylabel(f"Rforce (z={z_val})")

        ax_list[0].legend(
            # fontsize=8,
            bbox_to_anchor=(1.5, 1.8),
            # ha="center",
            # va="bottom",
        )
        # set grid and put figure index
        for i, ax in enumerate(ax_list):
            ax.grid()

            # xleft, xright = ax.get_xlim()
            # ybottom, ytop = ax.get_ylim()
            ax.text(
                -0.013,
                1.02,
                s="(" + chr(i + ord("a")) + ")",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="k",
            )

        #############################################################################
        # put a mark of script information on the figure
        # Get the script name and path automatically
        script_path = os.path.abspath(__file__)

        # Add the annotation to the figure
        plt.annotate(
            f"Generated by: {script_path}",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=3,
            color="gray",
        )
        # #############################################################################

        # plt.tight_layout()
        # plt.savefig('example figure - one-column.png', transparent=False)
        plt.show()


def plot_HomogeneousSpherePotential():
    amp = 1
    rho_0 = 3.0 * amp / (2 * numpy.pi)
    solidBall = HomogeneousSpherePotential()
    R_vals = numpy.linspace(0, 10, num=100)
    show_plot = True
    if show_plot:
        # font size
        plt.rc("font", size=11)  # Default text

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

        gs = gridspec.GridSpec(nrows=2, ncols=2)  # create grid for multiple figures

        # fix the margins
        fig.subplots_adjust(
            top=0.913, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
        )

        ax_dens = fig.add_subplot(gs[0, 0])
        ax_pot = fig.add_subplot(gs[0, 1], sharex=ax_dens)  #
        ax_R2deriv = fig.add_subplot(gs[1, 0], sharex=ax_dens)  #
        ax_Rforce = fig.add_subplot(gs[1, 1], sharex=ax_dens)  #
        ax_list = [ax_dens, ax_pot, ax_R2deriv, ax_Rforce]
        ax_dens.plot(
            R_vals,
            # solidBall._dens(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: solidBall._dens(R=R, z=0.0), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )

        ax_dens.set_xlabel("R")
        ax_dens.set_ylabel("Denisty")

        ax_pot.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: solidBall._evaluate(R=R, z=0.0), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )

        ax_pot.set_xlabel("R")
        ax_pot.set_ylabel("Potential")

        ax_R2deriv.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: solidBall._R2deriv(R=R, z=0.0), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )

        ax_R2deriv.set_xlabel("R")
        ax_R2deriv.set_ylabel("R2deriv")

        ax_Rforce.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: solidBall._Rforce(R=R, z=0.0), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )

        ax_Rforce.set_xlabel("R")
        ax_Rforce.set_ylabel("Rforce")

        # set grid and put figure index
        for i, ax in enumerate(ax_list):
            ax.grid()
            # xleft, xright = ax.get_xlim()
            # ybottom, ytop = ax.get_ylim()
            ax.text(
                -0.013,
                1.02,
                s="(" + chr(i + ord("a")) + ")",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="k",
            )
        # ha = 'left' or 'right'
        # va = 'top' or 'bottom'

        #############################################################################
        # put a mark of script information on the figure
        # Get the script name and path automatically
        script_path = os.path.abspath(__file__)

        # Add the annotation to the figure
        plt.annotate(
            f"Generated by: {script_path}",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=3,
            color="gray",
        )
        # #############################################################################
        plt.show()


def plot_homoSphere_by_num_and_sym():
    """
    plot properities of a homogeneous sphere by numeric and symbolic methods
    """
    ballRadius = 1.1
    amp = 1
    rho_0 = 3.0 * amp / (2 * numpy.pi)
    solidBall = HomogeneousSpherePotential(
        amp=amp,
        R=ballRadius,
    )
    r_vals = numpy.linspace(0.01, 10, num=100)
    R_vals = numpy.linspace(0.01, 10, num=100)
    z_val = 0.5
    pieces = [
        (rho_0, 0, ballRadius),
        (0.0, ballRadius, sympy.oo),
    ]

    def dens_sym():
        r = sympy.Symbol("r", real=True, positive=True)
        # Build the Piecewise arguments from the list
        pw_args = []
        for expr, rmin, rmax in pieces:
            cond = (r >= rmin) & (r < rmax)
            pw_args.append((expr, cond))

        # Add fallback condition for r >= EARTH_RADIUS_KM (density = 0)
        pw_args.append((0, True))

        dens_sym = sympy.Piecewise(*pw_args)
        return dens_sym, r

    solidBall_sym = SymbolicSphericalPotential(dens=dens_sym)
    show_plot = True
    if show_plot:
        # font size
        plt.rc("font", size=11)  # Default text

        fig = plt.figure(figsize=(6.0, 4.0), dpi=150)  # initialize a figure

        gs = gridspec.GridSpec(nrows=2, ncols=2)  # create grid for multiple figures

        # fix the margins
        fig.subplots_adjust(
            top=0.8, bottom=0.156, left=0.119, right=0.972, hspace=0.725, wspace=0.5
        )

        ax_dens = fig.add_subplot(gs[0, 0])
        ax_pot = fig.add_subplot(gs[0, 1], sharex=ax_dens)  #
        ax_R2deriv = fig.add_subplot(gs[1, 0], sharex=ax_dens)  #
        ax_Rforce = fig.add_subplot(gs[1, 1], sharex=ax_dens)  #
        ax_list = [ax_dens, ax_pot, ax_R2deriv, ax_Rforce]
        ax_dens.plot(
            R_vals,
            numpy.array(list(map(lambda R: solidBall._dens(R=R, z=z_val), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_dens.plot(
            R_vals,
            numpy.array(list(map(lambda R: solidBall_sym._dens(R=R, z=z_val), R_vals))),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_dens.set_xlabel("R")
        ax_dens.set_ylabel(f"Denisty (z={z_val})")

        ax_pot.plot(
            R_vals,
            numpy.array(list(map(lambda R: solidBall._evaluate(R=R, z=z_val), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_pot.plot(
            R_vals,
            numpy.array(
                list(map(lambda R: solidBall_sym._evaluate(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )
        ax_pot.set_xlabel("R")
        ax_pot.set_ylabel(f"Potential (z={z_val})")

        ax_R2deriv.plot(
            R_vals,
            numpy.array(list(map(lambda R: solidBall._R2deriv(R=R, z=z_val), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_R2deriv.plot(
            R_vals,
            numpy.array(
                list(map(lambda R: solidBall_sym._R2deriv(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_R2deriv.set_xlabel("R")
        ax_R2deriv.set_ylabel(f"R2deriv (z={z_val})")

        ax_Rforce.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(list(map(lambda R: solidBall._Rforce(R=R, z=z_val), R_vals))),
            label="by HomogeneousSpherePotential",
            color="tab:blue",
            alpha=1,
            linestyle="-",
        )
        ax_Rforce.plot(
            R_vals,
            # solidBall._evaluate(R=r_vals, z=zero_vals),
            numpy.array(
                list(map(lambda R: solidBall_sym._Rforce(R=R, z=z_val), R_vals))
            ),
            label="by SymbolicSphericalPotential",
            color="tab:orange",
            alpha=1,
            linestyle="--",
        )

        ax_Rforce.set_xlabel("R")
        ax_Rforce.set_ylabel(f"Rforce (z={z_val})")

        ax_list[0].legend(
            bbox_to_anchor=(1.5, 1.8),
        )
        # set grid and put figure index
        for i, ax in enumerate(ax_list):
            ax.grid()

            # xleft, xright = ax.get_xlim()
            # ybottom, ytop = ax.get_ylim()
            ax.text(
                -0.013,
                1.02,
                s="(" + chr(i + ord("a")) + ")",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color="k",
            )

        #############################################################################
        # put a watermark of script information on the figure
        # Get the script name and path automatically
        script_path = os.path.abspath(__file__)

        # Add the annotation to the figure
        plt.annotate(
            f"Generated by: {script_path}",
            xy=(0.02, 0.02),
            xycoords="figure fraction",
            fontsize=3,
            color="gray",
        )
        # #############################################################################
        # plt.savefig('example figure - one-column.png', transparent=False)
        plt.show()


# if __name__ == "__main__":
