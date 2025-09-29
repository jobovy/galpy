# Tests of fuzzy dark matter dynamical friction implementation
import sys

import pytest

PY3 = sys.version > "3"
PY_GE_314 = sys.version_info >= (3, 14)
import numpy

from galpy import potential
from galpy.util import galpyWarning


def test_FDMDynamicalFrictionForce_central_limit():
    # test that FDM dynamical friction in the central limit (i.e. when kr << 1)
    # agrees with analytical solutions for circular orbits in logarithmic potentials
    # assuming a constant velocity
    # r_pred = r0 * exp(- 0.5 * G * M_obj * vcirc * (m/hbar)**2 / 3 * t)

    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**6.0 / conversion.mass_in_msol(vo, ro)
    r0 = 0.001
    vc = 1.0
    m = 1e-99
    mhbar = (
        conversion.parse_mass(m, ro=ro, vo=vo)
        / conversion._GHBARINKM3S3KPC2
        * ro**2
        * vo**3
    )

    tau_pred = 3 / (GMs * vc * (mhbar) ** 2)  # analytical orbital time
    t = numpy.linspace(0.0, 2 * tau_pred, 1001)
    r_pred = r0 * numpy.exp(-t / tau_pred)  # analytical solution

    from galpy.potential import FDMDynamicalFrictionForce, LogarithmicHaloPotential

    Loghalo = LogarithmicHaloPotential(normalize=1.0)
    o = Orbit([r0, 0.0, vc, 0.0, 0.0, 0.0])
    fdf = FDMDynamicalFrictionForce(GMs=GMs, dens=Loghalo, m=m)
    o.integrate(t, Loghalo + fdf, method="dop853_c")

    # Compare to analytical solution
    assert numpy.amax(numpy.fabs(o.r(t) - r_pred)) / r0 < 0.001, (
        "FDMDynamicalFrictionForce in the central limit does not agree with analytical solution for circular orbits in logarithmic potentials"
    )

    # Also run this test using the Python implementation, but for less time
    t = numpy.linspace(0.0, 2 * tau_pred / 5, 1001)
    r_pred = r0 * numpy.exp(-t / tau_pred)  # analytical solution
    o.integrate(t, Loghalo + fdf, method="dop853")

    # Compare to analytical solution
    assert numpy.amax(numpy.fabs(o.r(t) - r_pred)) / r0 < 0.001, (
        "FDMDynamicalFrictionForce in the central limit does not agree with analytical solution for circular orbits in logarithmic potentials"
    )
    return None


def test_FDMDynamicalFrictionForce_const_FDMfactor():
    # test that FDM dynamical friction with a constant FDM factor
    # agrees with the analytical solution for circular orbits in logarithmic potentials
    # assuming a constant velocity
    # r_pred = np.sqrt(-2*GMs*const_FDMfactor*t/vc + r0**2)

    from scipy import special as sp

    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**6.0 / conversion.mass_in_msol(vo, ro)
    r0 = 0.2
    vc = 1.0
    m = 1e-99
    mhbar = (
        conversion.parse_mass(m, ro=ro, vo=vo)
        / conversion._GHBARINKM3S3KPC2
        * ro**2
        * vo**3
    )
    const_kr = mhbar * r0 * vc
    const_FDMfactor = (
        -sp.sici(const_kr)[1]
        + numpy.log(const_kr)
        + numpy.euler_gamma
        + (numpy.sin(const_kr) / (const_kr))
        - 1
    )
    tau_pred = vc * r0**2 / (2 * GMs * const_FDMfactor)  # analytical orbital time
    t = numpy.linspace(0.0, 0.99 * tau_pred, 1001)
    r_pred = numpy.sqrt(
        -2 * GMs * const_FDMfactor * t / vc + r0**2
    )  # analytical solution

    from galpy.potential import FDMDynamicalFrictionForce, LogarithmicHaloPotential

    Loghalo = LogarithmicHaloPotential(normalize=1.0)

    o = Orbit([r0, 0.0, vc, 0.0, 0.0, 0.0])
    fdf = FDMDynamicalFrictionForce(
        GMs=GMs, dens=Loghalo, m=m, const_FDMfactor=const_FDMfactor
    )
    o.integrate(t, Loghalo + fdf, method="dop853")

    # Compare to analytical solution
    assert numpy.amax(numpy.fabs(o.r(t) - r_pred)) / r0 < 0.001, (
        "FDMDynamicalFrictionForce with constant FDM factor does not agree with analytical solution for circular orbits in logarithmic potentials"
    )
    return None


def test_FDMDynamicalFrictionForce_const_FDMfactor_c():
    # test that FDM dynamical friction with a constant FDM factor
    # agrees with the analytical solution for circular orbits in logarithmic potentials
    # assuming a constant velocity
    # r_pred = np.sqrt(-2*GMs*const_FDMfactor*t/vc + r0**2)

    from scipy import special as sp

    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**6.0 / conversion.mass_in_msol(vo, ro)
    r0 = 0.2
    vc = 1.0
    m = 1e-99
    mhbar = (
        conversion.parse_mass(m, ro=ro, vo=vo)
        / conversion._GHBARINKM3S3KPC2
        * ro**2
        * vo**3
    )
    const_kr = mhbar * r0 * vc
    const_FDMfactor = (
        -sp.sici(const_kr)[1]
        + numpy.log(const_kr)
        + numpy.euler_gamma
        + (numpy.sin(const_kr) / (const_kr))
        - 1
    )
    tau_pred = vc * r0**2 / (2 * GMs * const_FDMfactor)  # analytical orbital time
    t = numpy.linspace(0.0, 0.99 * tau_pred, 1001)
    r_pred = numpy.sqrt(
        -2 * GMs * const_FDMfactor * t / vc + r0**2
    )  # analytical solution

    from galpy.potential import FDMDynamicalFrictionForce, LogarithmicHaloPotential

    Loghalo = LogarithmicHaloPotential(normalize=1.0)

    o = Orbit([r0, 0.0, vc, 0.0, 0.0, 0.0])
    fdf = FDMDynamicalFrictionForce(
        GMs=GMs, dens=Loghalo, m=m, const_FDMfactor=const_FDMfactor
    )
    o.integrate(t, Loghalo + fdf, method="dop853_c")

    # Compare to analytical solution
    assert numpy.amax(numpy.fabs(o.r(t) - r_pred)) / r0 < 0.001, (
        "FDMDynamicalFrictionForce with constant FDM factor does not agree with analytical solution for circular orbits in logarithmic potentials"
    )
    return None


def test_FDMDynamicalFrictionForce_classicalregime():
    # test that FDM dynamical friction in the classical regime (i.e. when kr >>1)
    # agrees with Chandrasekhar dynamical friction

    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    r_init = 3.0
    dt = 2.0 / conversion.time_in_Gyr(vo, ro)
    halo = potential.NFWPotential(normalize=1.0, a=1.5)
    # Compute evolution with FDM dynamical friction
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=GMs,
        dens=halo,
        sigmar=lambda r: 1.0 / numpy.sqrt(2.0),
        m=4e-98,  # 4e-21 eV
    )
    o = Orbit([r_init, 0.0, 1.0, 0.0, 0.0, 0.0])
    ts = numpy.linspace(0.0, dt, 1001)
    o.integrate(ts, [halo, fdf], method="odeint")
    # Compare to Chandrasekhar dynamical friction
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs,
        dens=halo,
        sigmar=lambda r: 1.0 / numpy.sqrt(2.0),
    )
    oc = o()
    oc.integrate(ts, [halo, cdfc], method="odeint")
    assert numpy.fabs(oc.r(ts[-1]) - o.r(ts[-1])) < 0.05, (
        "FDMDynamicalFrictionForce in the classical regime does not agree with Chandrasekhar dynamical friction"
    )
    return None


def test_FDMDynamicalFrictionForce_evaloutsideminrmaxr():
    # Test that FDM dynamical friction returns the expected force when evaluating
    # outside of the [minr,maxr] range over which sigmar is interpolated:
    # 0 at r < minr
    # using sigmar(r) for r > maxr
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    # Compute evolution with variable ln Lambda
    sigmar = lambda r: 1.0 / r
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=2.0,
    )
    # fdf 2 for checking r > maxr of fdf
    fdf2 = potential.FDMDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=4.0,
    )
    v = [0.1, 0.0, 0.0]
    # r < minr
    assert numpy.fabs(fdf.Rforce(0.1, 0.0, v=v)) < 1e-16, (
        "potential.FDMDynamicalFrictionForce at r < minr not equal to zero"
    )
    assert numpy.fabs(fdf.zforce(0.1, 0.0, v=v)) < 1e-16, (
        "potential.FDMDynamicalFrictionForce at r < minr not equal to zero"
    )
    # r > maxr
    assert numpy.fabs(fdf.Rforce(3.0, 0.0, v=v) - fdf2.Rforce(3.0, 0.0, v=v)) < 1e-10, (
        "potential.FDMDynamicalFrictionForce at r > maxr not as expected"
    )
    assert numpy.fabs(fdf.zforce(3.0, 0.0, v=v) - fdf2.zforce(3.0, 0.0, v=v)) < 1e-10, (
        "potential.FDMDynamicalFrictionForce at r > maxr not as expected"
    )
    return None


def test_FDMDynamicalFrictionForce_pickling():
    # Test that FDMDynamicalFrictionForce objects can/cannot be
    # pickled as expected
    import pickle

    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    # sigmar internally computed, should be able to be pickled
    # Compute evolution with variable ln Lambda
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=GMs, rhm=0.125, dens=potential.MWPotential2014, minr=0.5, maxr=2.0
    )
    pickled = pickle.dumps(fdf)
    fdfu = pickle.loads(pickled)
    # Test a few values
    assert (
        numpy.fabs(
            fdf.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
            - fdfu.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of FDMDynamicalFrictionForce object does not work as expected"
    assert (
        numpy.fabs(
            fdf.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
            - fdfu.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of FDMDynamicalFrictionForce object does not work as expected"
    # Not providing dens = Logarithmic should also work
    fdf = potential.FDMDynamicalFrictionForce(GMs=GMs, rhm=0.125, minr=0.5, maxr=2.0)
    pickled = pickle.dumps(fdf)
    fdfu = pickle.loads(pickled)
    # Test a few values
    assert (
        numpy.fabs(
            fdf.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
            - fdfu.Rforce(1.0, 0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of FDMDynamicalFrictionForce object does not work as expected"
    assert (
        numpy.fabs(
            fdf.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
            - fdfu.zforce(2.0, -0.2, v=[1.0, 1.0, 0.0])
        )
        < 1e-10
    ), "Pickling of FDMDynamicalFrictionForce object does not work as expected"

    # Providing sigmar as a lambda function gives AttributeError
    sigmar = lambda r: 1.0 / r
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=GMs,
        rhm=0.125,
        dens=potential.MWPotential2014,
        sigmar=sigmar,
        minr=0.5,
        maxr=2.0,
    )
    if PY3 and not PY_GE_314:
        with pytest.raises(AttributeError) as excinfo:
            pickled = pickle.dumps(fdf)
    else:
        with pytest.raises(pickle.PicklingError) as excinfo:
            pickled = pickle.dumps(fdf)
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
        potential.DehnenSphericalPotential(normalize=4.0, alpha=1.2),
        potential.DehnenCoreSphericalPotential(normalize=4.0),
        potential.PlummerPotential(normalize=0.6, b=3.0),
        potential.HomogeneousSpherePotential(normalize=0.02, R=82.0 / 8),
        MWPotential3021,
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
        if pname in list(tol.keys()):
            ttol = tol[pname]
        else:
            ttol = tol["default"]
        # Setup orbit, ~ LMC
        o = Orbit(
            [5.13200034, 1.08033051, 0.23323391, -3.48068653, 0.94950884, -1.54626091]
        )
        # Setup dynamical friction object
        fdf = potential.FDMDynamicalFrictionForce(
            GMs=0.5553870441722593,
            rhm=5.0 / 8.0,
            dens=p,
            m=3e-102,
            maxr=500.0 / 8,
            nr=201,
        )
        ttimes = times
        # Integrate in C
        o.integrate(ttimes, p + fdf, method=integrator)
        # Integrate in Python
        op = o()
        op.integrate(ttimes, p + fdf, method=py_integrator)
        # Compare r (most important)
        assert numpy.amax(numpy.fabs(o.r(ttimes) - op.r(ttimes))) < 10**ttol, (
            f"Dynamical friction in C does not agree with dynamical friction in Python for potential {pname}"
        )
    return None


# Test that r < minr in FDMDynamFric works properly
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
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=0.5553870441722593,
        rhm=5.0 / 8.0,
        dens=pot,
        minr=130.0 / 8.0,
        maxr=500.0 / 8,
    )
    # Integrate in C with dynamical friction
    o.integrate(times, pot + fdf, method=integrator)
    # Integrate in C without dynamical friction
    op = o()
    op.integrate(times, pot, method=integrator)
    # Compare r (most important)
    assert numpy.amax(numpy.fabs(o.r(times) - op.r(times))) < 10**-8.0, (
        "Dynamical friction in C does not properly use minr"
    )
    return None


# Test that when an orbit reaches r < minr, a warning is raised to alert the user
def test_dynamfric_c_minr_warning():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 100.0, 1001)  # ~3 Gyr at the Solar circle
    integrator = "dop853_c"
    pot = potential.LogarithmicHaloPotential(normalize=1)
    # Setup orbit
    o = Orbit([0.5, 0.1, 0.0, 0.0, 0.0, 0.0])
    # Setup dynamical friction object, with minr = 1, should thus reach it
    fdf = potential.FDMDynamicalFrictionForce(
        GMs=0.5553870441722593, rhm=5.0 / 8.0, dens=pot, minr=1.0
    )
    # Integrate, should raise warning
    with pytest.warns(galpyWarning) as record:
        o.integrate(times, pot + fdf, method=integrator)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Orbit integration with FDMDynamicalFrictionForce entered domain where r < minr and FDMDynamicalFrictionForce is turned off; initialize FDMDynamicalFrictionForce with a smaller minr to avoid this if you wish (but note that you want to turn it off close to the center for an object that sinks all the way to r=0, to avoid numerical instabilities)"
        )
    assert raisedWarning, (
        "Integrating an orbit that goes to r < minr with dynamical friction should have raised a warning, but didn't"
    )
    return None
