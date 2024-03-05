# Tests of dynamical friction implementation
import sys

import pytest

PY3 = sys.version > "3"
import numpy

from galpy import potential
from galpy.util import galpyWarning


def test_ChandrasekharDynamicalFrictionForce_constLambda():
    # Test that the ChandrasekharDynamicalFrictionForce with constant Lambda
    # agrees with analytical solutions for circular orbits:
    # assuming that a mass remains on a circular orbit in an isothermal halo
    # with velocity dispersion sigma and for constant Lambda:
    # r_final^2 - r_initial^2 = -0.604 ln(Lambda) GM/sigma t
    # (e.g., B&T08, p. 648)
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    const_lnLambda = 7.0
    r_init = 2.0
    dt = 2.0 / conversion.time_in_Gyr(vo, ro)
    # Compute
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs, const_lnLambda=const_lnLambda, dens=lp
    )  # don't provide sigmar, so it gets computed using galpy.df.jeans
    o = Orbit([r_init, 0.0, 1.0, 0.0, 0.0, 0.0])
    ts = numpy.linspace(0.0, dt, 1001)
    o.integrate(ts, [lp, cdfc], method="odeint")
    r_pred = numpy.sqrt(
        o.r() ** 2.0 - 0.604 * const_lnLambda * GMs * numpy.sqrt(2.0) * dt
    )
    assert (
        numpy.fabs(r_pred - o.r(ts[-1])) < 0.01
    ), "ChandrasekharDynamicalFrictionForce with constant lnLambda for circular orbits does not agree with analytical prediction"
    return None


def test_ChandrasekharDynamicalFrictionForce_varLambda():
    # Test that dynamical friction with variable Lambda for small r ranges
    # gives ~ the same result as using a constant Lambda that is the mean of
    # the variable lambda
    # Also tests that giving an axisymmetric list of potentials for the
    # density works
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    r_init = 3.0
    dt = 2.0 / conversion.time_in_Gyr(vo, ro)
    # Compute evolution with variable ln Lambda
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=lambda r: 1.0 / numpy.sqrt(2.0),
    )
    o = Orbit([r_init, 0.0, 1.0, 0.0, 0.0, 0.0])
    ts = numpy.linspace(0.0, dt, 1001)
    o.integrate(ts, [potential.MWPotential2014, cdf], method="odeint")
    lnLs = numpy.array(
        [
            cdf.lnLambda(r, v)
            for (r, v) in zip(
                o.r(ts), numpy.sqrt(o.vx(ts) ** 2.0 + o.vy(ts) ** 2.0 + o.vz(ts) ** 2.0)
            )
        ]
    )
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        const_lnLambda=numpy.mean(lnLs),
        dens=potential.MWPotential2014,
        sigmar=lambda r: 1.0 / numpy.sqrt(2.0),
    )
    oc = o()
    oc.integrate(ts, [potential.MWPotential2014, cdfc], method="odeint")
    assert (
        numpy.fabs(oc.r(ts[-1]) - o.r(ts[-1])) < 0.05
    ), "ChandrasekharDynamicalFrictionForce with variable lnLambda for a short radial range is not close to the calculation using a constant lnLambda"
    return None


def test_ChandrasekharDynamicalFrictionForce_evaloutsideminrmaxr():
    # Test that dynamical friction returns the expected force when evaluating
    # outside of the [minr,maxr] range over which sigmar is interpolated:
    # 0 at r < minr
    # using sigmar(r) for r > maxr
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    # Compute evolution with variable ln Lambda
    sigmar = lambda r: 1.0 / r
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=2.0,
    )
    # cdf 2 for checking r > maxr of cdf
    cdf2 = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=4.0,
    )
    v = [0.1, 0.0, 0.0]
    # r < minr
    assert (
        numpy.fabs(cdf.Rforce(0.1, 0.0, v=v)) < 1e-16
    ), "potential.ChandrasekharDynamicalFrictionForce at r < minr not equal to zero"
    assert (
        numpy.fabs(cdf.zforce(0.1, 0.0, v=v)) < 1e-16
    ), "potential.ChandrasekharDynamicalFrictionForce at r < minr not equal to zero"
    # r > maxr
    assert (
        numpy.fabs(cdf.Rforce(3.0, 0.0, v=v) - cdf2.Rforce(3.0, 0.0, v=v)) < 1e-10
    ), "potential.ChandrasekharDynamicalFrictionForce at r > maxr not as expected"
    assert (
        numpy.fabs(cdf.zforce(3.0, 0.0, v=v) - cdf2.zforce(3.0, 0.0, v=v)) < 1e-10
    ), "potential.ChandrasekharDynamicalFrictionForce at r > maxr not as expected"
    return None


def test_ChandrasekharDynamicalFrictionForce_pickling():
    # Test that ChandrasekharDynamicalFrictionForce objects can/cannot be
    # pickled as expected
    import pickle

    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    # sigmar internally computed, should be able to be pickled
    # Compute evolution with variable ln Lambda
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs, rhm=0.125, dens=potential.MWPotential2014, minr=0.5, maxr=2.0
    )
    pickled = pickle.dumps(cdf)
    cdfu = pickle.loads(pickled)
    # Test a few values
    assert (
        numpy.fabs(
            cdf.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
            - cdfu.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of ChandrasekharDynamicalFrictionForce object does not work as expected"
    assert (
        numpy.fabs(
            cdf.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
            - cdfu.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of ChandrasekharDynamicalFrictionForce object does not work as expected"
    # Not providing dens = Logarithmic should also work
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs, rhm=0.125, minr=0.5, maxr=2.0
    )
    pickled = pickle.dumps(cdf)
    cdfu = pickle.loads(pickled)
    # Test a few values
    assert (
        numpy.fabs(
            cdf.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
            - cdfu.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of ChandrasekharDynamicalFrictionForce object does not work as expected"
    assert (
        numpy.fabs(
            cdf.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
            - cdfu.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of ChandrasekharDynamicalFrictionForce object does not work as expected"

    # Providing sigmar as a lambda function gives AttributeError
    sigmar = lambda r: 1.0 / r
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=2.0,
    )
    if PY3:
        with pytest.raises(AttributeError) as excinfo:
            pickled = pickle.dumps(cdf)
    else:
        with pytest.raises(pickle.PicklingError) as excinfo:
            pickled = pickle.dumps(cdf)
    return None


# Test whether dynamical friction in C works (compare to Python, which is
# tested below; put here because a test of many potentials)
def test_dynamfric_c():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import McMillan17
    from galpy.potential.Potential import _check_c

    # Basic parameters for the test
    times = numpy.linspace(0.0, -100.0, 1001)  # ~3 Gyr at the Solar circle
    integrator = "dop853_c"
    py_integrator = "dop853"
    # Define all of the potentials (by hand, because need reasonable setup)
    MWPotential3021 = copy.deepcopy(potential.MWPotential2014)
    MWPotential3021[2] *= 1.5  # Increase mass by 50%
    pots = [
        potential.LogarithmicHaloPotential(normalize=1),
        potential.LogarithmicHaloPotential(normalize=1.3, q=0.9, b=0.7),  # nonaxi
        potential.NFWPotential(normalize=1.0, a=1.5),
        potential.MiyamotoNagaiPotential(normalize=0.02, a=10.0, b=10.0),
        potential.MiyamotoNagaiPotential(normalize=0.6, a=0.0, b=3.0),  # special case
        potential.PowerSphericalPotential(alpha=2.3, normalize=2.0),
        potential.DehnenSphericalPotential(normalize=4.0, alpha=1.2),
        potential.DehnenCoreSphericalPotential(normalize=4.0),
        potential.HernquistPotential(normalize=1.0, a=3.5),
        potential.JaffePotential(normalize=1.0, a=20.5),
        potential.DoubleExponentialDiskPotential(normalize=0.2, hr=3.0, hz=0.6),
        potential.FlattenedPowerPotential(normalize=3.0),
        potential.FlattenedPowerPotential(normalize=3.0, alpha=0),  # special case
        potential.IsochronePotential(normalize=2.0),
        potential.PowerSphericalPotentialwCutoff(normalize=0.3, rc=10.0),
        potential.PlummerPotential(normalize=0.6, b=3.0),
        potential.PseudoIsothermalPotential(normalize=0.1, a=3.0),
        potential.BurkertPotential(normalize=0.2, a=2.5),
        potential.TriaxialHernquistPotential(normalize=1.0, a=3.5, b=0.8, c=0.9),
        potential.TriaxialNFWPotential(normalize=1.0, a=1.5, b=0.8, c=0.9),
        potential.TriaxialJaffePotential(normalize=1.0, a=20.5, b=0.8, c=1.4),
        potential.PerfectEllipsoidPotential(normalize=0.3, a=3.0, b=0.7, c=1.5),
        potential.PerfectEllipsoidPotential(
            normalize=0.3, a=3.0, b=0.7, c=1.5, pa=3.0, zvec=[0.0, 1.0, 0.0]
        ),  # rotated
        potential.HomogeneousSpherePotential(
            normalize=0.02, R=82.0 / 8
        ),  # make sure to go to dens = 0 part,
        potential.interpSphericalPotential(
            rforce=potential.HomogeneousSpherePotential(normalize=0.02, R=82.0 / 8.0),
            rgrid=numpy.linspace(0.0, 82.0 / 8.0, 201),
        ),
        potential.TriaxialGaussianPotential(
            normalize=0.03, sigma=4.0, b=0.8, c=1.5, pa=3.0, zvec=[1.0, 0.0, 0.0]
        ),
        potential.SCFPotential(
            Acos=numpy.array([[[1.0]]]),
            normalize=1.0,
            a=3.5,  # same as Hernquist
        ),
        potential.SCFPotential(
            Acos=numpy.array([[[1.0, 0.0], [0.3, 0.0]]]),  # nonaxi
            Asin=numpy.array([[[0.0, 0.0], [1e-1, 0.0]]]),
            normalize=1.0,
            a=3.5,
        ),
        MWPotential3021,
        McMillan17,  # SCF + DiskSCF
    ]
    # tolerances in log10
    tol = {}
    tol["default"] = -7.0
    # Following are a little more difficult
    tol["DoubleExponentialDiskPotential"] = -4.5
    tol["TriaxialHernquistPotential"] = -6.0
    tol["TriaxialNFWPotential"] = -6.0
    tol["TriaxialJaffePotential"] = -6.0
    tol["MWPotential3021"] = -6.0
    tol["HomogeneousSpherePotential"] = -6.0
    tol["interpSphericalPotential"] = -6.0  # == HomogeneousSpherePotential
    tol["McMillan17"] = -6.0
    for p in pots:
        if not _check_c(p, dens=True):
            continue  # dynamfric not in C!
        pname = type(p).__name__
        if pname == "list":
            if (
                isinstance(p[0], potential.PowerSphericalPotentialwCutoff)
                and len(p) > 1
                and isinstance(p[1], potential.MiyamotoNagaiPotential)
                and len(p) > 2
                and isinstance(p[2], potential.NFWPotential)
            ):
                pname = "MWPotential3021"  # Must be!
            else:
                pname = "McMillan17"
        # print(pname)
        if pname in list(tol.keys()):
            ttol = tol[pname]
        else:
            ttol = tol["default"]
        # Setup orbit, ~ LMC
        o = Orbit(
            [5.13200034, 1.08033051, 0.23323391, -3.48068653, 0.94950884, -1.54626091]
        )
        # Setup dynamical friction object
        if pname == "McMillan17":
            cdf = potential.ChandrasekharDynamicalFrictionForce(
                GMs=0.5553870441722593, rhm=5.0 / 8.0, dens=p, maxr=500.0 / 8, nr=101
            )
            ttimes = numpy.linspace(0.0, -30.0, 1001)  # ~1 Gyr at the Solar circle
        else:
            cdf = potential.ChandrasekharDynamicalFrictionForce(
                GMs=0.5553870441722593, rhm=5.0 / 8.0, dens=p, maxr=500.0 / 8, nr=201
            )
            ttimes = times
        # Integrate in C
        o.integrate(ttimes, p + cdf, method=integrator)
        # Integrate in Python
        op = o()
        op.integrate(ttimes, p + cdf, method=py_integrator)
        # Compare r (most important)
        assert (
            numpy.amax(numpy.fabs(o.r(ttimes) - op.r(ttimes))) < 10**ttol
        ), f"Dynamical friction in C does not agree with dynamical friction in Python for potential {pname}"
    return None


# Test that r < minr in ChandrasekharDynamFric works properly
def test_dynamfric_c_minr():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, -100.0, 1001)  # ~3 Gyr at the Solar circle
    integrator = "dop853_c"
    pot = potential.LogarithmicHaloPotential(normalize=1)
    # Setup orbit, ~ LMC
    o = Orbit(
        [5.13200034, 1.08033051, 0.23323391, -3.48068653, 0.94950884, -1.54626091]
    )
    # Setup dynamical friction object, with minr = 130 st always 0 for this orbit
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.5553870441722593,
        rhm=5.0 / 8.0,
        dens=pot,
        minr=130.0 / 8.0,
        maxr=500.0 / 8,
    )
    # Integrate in C with dynamical friction
    o.integrate(times, pot + cdf, method=integrator)
    # Integrate in C without dynamical friction
    op = o()
    op.integrate(times, pot, method=integrator)
    # Compare r (most important)
    assert (
        numpy.amax(numpy.fabs(o.r(times) - op.r(times))) < 10**-8.0
    ), "Dynamical friction in C does not properly use minr"
    return None


# Test that when an orbit reaches r < minr, a warning is raised to alert the user
def test_dynamfric_c_minr_warning():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 100.0, 1001)  # ~3 Gyr at the Solar circle
    integrator = "dop853_c"
    pot = potential.LogarithmicHaloPotential(normalize=1)
    # Setup orbit
    o = Orbit()
    # Setup dynamical friction object, with minr = 1, should thus reach it
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.5553870441722593, rhm=5.0 / 8.0, dens=pot, minr=1.0
    )
    # Integrate, should raise warning
    with pytest.warns(galpyWarning) as record:
        o.integrate(times, pot + cdf, method=integrator)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Orbit integration with ChandrasekharDynamicalFrictionForce entered domain where r < minr and ChandrasekharDynamicalFrictionForce is turned off; initialize ChandrasekharDynamicalFrictionForce with a smaller minr to avoid this if you wish (but note that you want to turn it off close to the center for an object that sinks all the way to r=0, to avoid numerical instabilities)"
        )
    assert raisedWarning, "Integrating an orbit that goes to r < minr with dynamical friction should have raised a warning, but didn't"
    return None
