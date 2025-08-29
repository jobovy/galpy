# Test the functions in galpy/util/__init__.py
# import sys
import numpy
import pytest

from galpy import orbit, potential


# Test run for two different rtol/atol values for 2d orbit integration on a Kepler pot. for all algorithms available
# test is passed if the difference in the orbital reconstruction is not exactly zero for all sampling points and
# if the orbital energy loss is smaller for the more precise rtol/atol orbit reconstruction
def test_2d_tol_integration():
    from galpy import orbit

    ttol_vec = [1e-12, 1e-6]
    times = numpy.linspace(
        0.0, 10.0, 250
    )  # with this time stepping, rk6_c results will not be affected by changes in rtol/atol
    integrators = [
        "dopr54_c",
        "odeint",
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
    ]
    # only use the simplest normalised KeplerPotential
    pot = potential.KeplerPotential(amp=1.0, normalize=True)
    # initialise list of resulting orbits for each integrator for each rtol/atol (10 integrators, 2 tols)
    o_lists = [[] * len(ttol_vec) for j in range(len(integrators))]
    Delta_r = numpy.zeros(
        [len(integrators)]
    )  # reconstruction differences in position summed over all time steps
    Delta_E = numpy.zeros(
        [len(integrators)]
    )  # energy differences summed over all time steps

    for cnt_int in numpy.arange(
        len(integrators)
    ):  # loop over all possible integration algorithms
        for cnt_tol in numpy.arange(len(ttol_vec)):
            # initialise a test orbit with few rounds, integrate trajectory, append to list of orbits
            o = orbit.Orbit([1.0, 0.1, 0.8])  # Orbit([R, vR, vT])
            o.integrate(
                times,
                pot,
                method=integrators[cnt_int],
                rtol=ttol_vec[cnt_tol],
                atol=ttol_vec[cnt_tol],
            )
            o_lists[cnt_int].append(o)

        # make test for differing reconstruction precision and energy loss along the orbits
        Delta_r[cnt_int] = numpy.sum(
            numpy.abs(o_lists[cnt_int][0].r(times) - o_lists[cnt_int][1].r(times))
        )
        Delta_E[cnt_int] = numpy.sum(
            numpy.abs(o_lists[cnt_int][0].E(times) - o_lists[cnt_int][1].E(times))
        )

        # if special integrators yield same reconstructions
        if integrators[cnt_int] == "rk6_c":
            assert Delta_r[cnt_int] == 0.0, (
                f"{integrators[cnt_int]} orbit integration is unexpectedly sensitive to rtol/atol - position difference"
            )
            assert Delta_E[cnt_int] == 0.0, (
                f"{integrators[cnt_int]} orbit integration is unexpectedly sensitive to rtol/atol - energy difference"
            )
        else:  # for all other integration routines check that differences are moderate as expected
            assert Delta_r[cnt_int] > 0.0, (
                f"{integrators[cnt_int]} orbit integration unexpectedly not sensitive to rtol/atol - position difference"
            )
            assert Delta_r[cnt_int] < 0.1, (
                f"{integrators[cnt_int]} orbit integration has worse than expected reconstruction precision - position difference"
            )
            assert Delta_E[cnt_int] > 0.0, (
                f"{integrators[cnt_int]} orbit integration unexpectedly not sensitive to rtol/atol - energy difference"
            )
            assert Delta_E[cnt_int] < 0.1, (
                f"{integrators[cnt_int]} orbit integration has worse than expected reconstruction precision - energy difference"
            )

    return None
