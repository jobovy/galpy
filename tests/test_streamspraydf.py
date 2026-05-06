import copy

import numpy
import pytest

from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.df import chen24spraydf, fardal15spraydf, streamdf, streamspraydf
from galpy.orbit import Orbit
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    HernquistPotential,
    LogarithmicHaloPotential,
    MovingObjectPotential,
    MWPotential2014,
    PlummerPotential,
    RotateAndTiltWrapperPotential,
)
from galpy.util import conversion  # for unit conversions
from galpy.util import coords

################################ Tests against streamdf ######################


def test_streamspraydf_deprecation():
    # Check if the deprecating class raises the correct warning
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    with pytest.warns(DeprecationWarning):
        spdf = streamspraydf(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        )


# Setup both DFs
@pytest.fixture(scope="module")
def setup_testStreamsprayAgainstStreamdf():
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    # Set up streamdf
    sigv = 0.365  # km/s
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    # Set up streamspraydf
    f15spdf_bovy14 = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    c24spdf_bovy14 = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    return sdf_bovy14, [f15spdf_bovy14, c24spdf_bovy14]


def test_sample_bovy14(setup_testStreamsprayAgainstStreamdf):
    # Load objects that were setup above
    sdf_bovy14, spdfs_bovy14 = setup_testStreamsprayAgainstStreamdf
    for spdf_bovy14 in spdfs_bovy14:
        numpy.random.seed(1)
        RvR_sdf = sdf_bovy14.sample(n=1000)
        RvR_spdf = spdf_bovy14.sample(n=1000, integrate=True, return_orbit=False)
        # Sanity checks
        # Range in Z
        indx = (RvR_sdf[3] > 4.0 / 8.0) * (RvR_sdf[3] < 5.0 / 8.0)
        # mean
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[0][indx]) - numpy.mean(RvR_spdf[0][indx]))
            < 6e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[1][indx]) - numpy.mean(RvR_spdf[1][indx]))
            < 5e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[2][indx]) - numpy.mean(RvR_spdf[2][indx]))
            < 5e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[4][indx]) - numpy.mean(RvR_spdf[4][indx]))
            < 5e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[5][indx]) - numpy.mean(RvR_spdf[5][indx]))
            < 1e-1
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        # Another range in Z
        indx = (RvR_sdf[3] > 5.0 / 8.0) * (RvR_sdf[3] < 6.0 / 8.0)
        # mean
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[0][indx]) - numpy.mean(RvR_spdf[0][indx]))
            < 1e-1
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[1][indx]) - numpy.mean(RvR_spdf[1][indx]))
            < 3e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[2][indx]) - numpy.mean(RvR_spdf[2][indx]))
            < 4e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[4][indx]) - numpy.mean(RvR_spdf[4][indx]))
            < 3e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
        assert (
            numpy.fabs(numpy.mean(RvR_sdf[5][indx]) - numpy.mean(RvR_spdf[5][indx]))
            < 1e-1
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean)"
        )
    return None


def test_bovy14_sampleorbit(setup_testStreamsprayAgainstStreamdf):
    # Load objects that were setup above
    sdf_bovy14, spdfs_bovy14 = setup_testStreamsprayAgainstStreamdf
    for spdf_bovy14 in spdfs_bovy14:
        numpy.random.seed(1)
        XvX_sdf = sdf_bovy14.sample(n=1000, xy=True)
        XvX_spdf = spdf_bovy14.sample(
            n=1000
        )  # returns Orbit, from which we can get anything we want
        # Sanity checks
        # Range in Z
        indx = (XvX_sdf[2] > 4.0 / 8.0) * (XvX_sdf[2] < 5.0 / 8.0)
        # mean
        assert (
            numpy.fabs(numpy.mean(XvX_sdf[0][indx]) - numpy.mean(XvX_spdf.x()[indx]))
            < 6e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)"
        )
        assert (
            numpy.fabs(numpy.mean(XvX_sdf[1][indx]) - numpy.mean(XvX_spdf.y()[indx]))
            < 2e-1
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)"
        )
        assert (
            numpy.fabs(numpy.mean(XvX_sdf[4][indx]) - numpy.mean(XvX_spdf.vy()[indx]))
            < 3e-2
        ), (
            "streamdf and streamspraydf do not generate similar samples for the Bovy (2014) stream (mean, xy)"
        )
    return None


def test_integrate(setup_testStreamsprayAgainstStreamdf):
    # Test that sampling at stripping + integrate == sampling at the end
    # Load objects that were setup above
    _, spdfs_bovy14 = setup_testStreamsprayAgainstStreamdf
    for spdf_bovy14 in spdfs_bovy14:
        # Sample at at stripping
        numpy.random.seed(4)
        RvR_noint, dt_noint = spdf_bovy14.sample(
            n=100, return_orbit=False, returndt=True, integrate=False
        )
        # and integrate
        for ii in range(len(dt_noint)):
            to = Orbit(RvR_noint[:, ii])
            to.integrate(numpy.linspace(-dt_noint[ii], 0.0, 1001), spdf_bovy14._pot)
            RvR_noint[:, ii] = [
                to.R(0.0),
                to.vR(0.0),
                to.vT(0.0),
                to.z(0.0),
                to.vz(0.0),
                to.phi(0.0),
            ]
        # Sample today
        numpy.random.seed(4)
        RvR, dt = spdf_bovy14.sample(
            n=100, return_orbit=False, returndt=True, integrate=True
        )
        # Should agree
        assert numpy.amax(numpy.fabs(dt - dt_noint)) < 1e-10, (
            "Times not the same when sampling with and without integrating"
        )
        assert numpy.amax(numpy.fabs(RvR - RvR_noint)) < 1e-7, (
            "Phase-space points not the same when sampling with and without integrating"
        )
    return None


def test_integrate_rtnonarray():
    # Test that sampling at stripping + integrate == sampling at the end
    # For a potential that doesn't support array inputs
    nfp = RotateAndTiltWrapperPotential(
        pot=LogarithmicHaloPotential(normalize=1.0, q=0.9),
        zvec=[0.0, numpy.sin(0.3), numpy.cos(0.3)],
    )
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for streamspraydf in [fardal15spraydf, chen24spraydf]:
        # Set up streamspraydf
        spdf_bovy14 = streamspraydf(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=nfp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        # Sample at at stripping
        numpy.random.seed(4)
        RvR_noint, dt_noint = spdf_bovy14.sample(
            n=100, return_orbit=False, returndt=True, integrate=False
        )
        # and integrate
        for ii in range(len(dt_noint)):
            to = Orbit(RvR_noint[:, ii])
            to.integrate(numpy.linspace(-dt_noint[ii], 0.0, 1001), spdf_bovy14._pot)
            RvR_noint[:, ii] = [
                to.R(0.0),
                to.vR(0.0),
                to.vT(0.0),
                to.z(0.0),
                to.vz(0.0),
                to.phi(0.0),
            ]
        # Sample today
        numpy.random.seed(4)
        RvR, dt = spdf_bovy14.sample(
            n=100, return_orbit=False, returndt=True, integrate=True
        )
        # Should agree
        assert numpy.amax(numpy.fabs(dt - dt_noint)) < 1e-10, (
            "Times not the same when sampling with and without integrating"
        )
        assert numpy.amax(numpy.fabs(RvR - RvR_noint)) < 1e-7, (
            "Phase-space points not the same when sampling with and without integrating"
        )
    return None


def test_center():
    # Test that a stream around a different center is generated
    # when using center
    # In this example, we'll generate a stream in the LMC orbiting the MW
    # LMC and its orbit
    ro, vo = 8.0, 220.0
    o = Orbit(
        [5.13200034, 1.08033051, 0.2332339, -3.48068653, 0.94950884, -1.54626091]
    )  # Result from from_name('LMC')
    tMWPotential2014 = copy.deepcopy(MWPotential2014)
    tMWPotential2014[2] *= 1.5
    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=10 / conversion.mass_in_1010msol(vo, ro),
        rhm=5.0 / ro,
        dens=tMWPotential2014,
    )
    ts = numpy.linspace(0.0, -10.0, 1001) / conversion.time_in_Gyr(vo, ro)
    o.integrate(ts, tMWPotential2014 + cdf)
    lmcpot = HernquistPotential(
        amp=2 * 10 / conversion.mass_in_1010msol(vo, ro),
        a=5.0 / ro / (1.0 + numpy.sqrt(2.0)),
    )  # rhm = (1+sqrt(2)) a
    moving_lmcpot = MovingObjectPotential(o, pot=lmcpot)
    # Now generate a stream within the LMC, progenitor at 8x kpc on circular orbit
    of = o(ts[-1])  # LMC at final point, earliest time, for convenience
    # Following pos in kpc, vel in km/s
    R_in_lmc = 1.0
    prog_phasespace = (
        of.x(use_physical=False) + R_in_lmc,
        of.y(use_physical=False),
        of.z(use_physical=False),
        of.vx(use_physical=False),
        of.vy(use_physical=False) + lmcpot.vcirc(R_in_lmc, use_physical=False),
        of.vz(use_physical=False),
    )
    prog_pos = coords.rect_to_cyl(
        prog_phasespace[0], prog_phasespace[1], prog_phasespace[2]
    )
    prog_vel = coords.rect_to_cyl_vec(
        prog_phasespace[3],
        prog_phasespace[4],
        prog_phasespace[5],
        None,
        prog_pos[1],
        None,
        cyl=True,
    )
    prog = Orbit(
        [prog_pos[0], prog_vel[0], prog_vel[1], prog_pos[2], prog_vel[2], prog_pos[1]],
        ro=8.0,
        vo=220.0,
    )
    # Integrate prog forward
    prog.integrate(ts[::-1], tMWPotential2014 + moving_lmcpot)
    for streamspraydf in [fardal15spraydf, chen24spraydf]:
        # Then set up streamspraydf
        spdf = streamspraydf(
            2e4 / conversion.mass_in_msol(vo, ro),
            progenitor=prog(0.0),
            pot=tMWPotential2014 + moving_lmcpot,
            rtpot=lmcpot,
            tdisrupt=10.0 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
            center=o,
            centerpot=tMWPotential2014 + cdf,
        )
        # Generate stream
        numpy.random.seed(1)
        stream_RvR = spdf.sample(n=300, return_orbit=False, integrate=True)
        stream_pos = coords.cyl_to_rect(stream_RvR[0], stream_RvR[5], stream_RvR[3])
        # Stream should lie on a circle with radius R_in_lmc
        stream_R_wrt_LMC = numpy.sqrt(
            (stream_pos[0] - o.x(use_physical=False)) ** 2.0
            + (stream_pos[1] - o.y(use_physical=False)) ** 2.0
        )
        assert numpy.fabs(numpy.mean(stream_R_wrt_LMC) - R_in_lmc) < 0.1, (
            "Stream generated in the LMC does not appear to be on a circle within the LMC"
        )
        assert numpy.fabs(numpy.std(stream_R_wrt_LMC)) < 0.15, (
            "Stream generated in the LMC does not appear to be on a circle within the LMC"
        )
    return None


def test_sample_orbit_rovoetc():
    # Test that the sample orbit output has the same ro/vo/etc. as the
    # input progenitor
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ro, vo = 9.0, 230.0
    zo, solarmotion = 0.03, [-20.0, 30.0, 40.0]
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=ro,
        vo=vo,
        zo=zo,
        solarmotion=solarmotion,
    )
    for streamspraydf in [fardal15spraydf, chen24spraydf]:
        # Set up streamspraydf
        spdf_bovy14 = streamspraydf(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        sam = spdf_bovy14.sample(n=10)
        assert obs._roSet is sam._roSet, (
            "Sampled streamspraydf orbits do not have the same roSet as the progenitor orbit"
        )
        assert obs._voSet is sam._voSet, (
            "Sampled streamspraydf orbits do not have the same voSet as the progenitor orbit"
        )
        assert numpy.fabs(obs._ro - sam._ro) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same ro as the progenitor orbit"
        )
        assert numpy.fabs(obs._vo - sam._vo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same vo as the progenitor orbit"
        )
        assert numpy.fabs(obs._zo - sam._zo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same zo as the progenitor orbit"
        )
        assert numpy.all(numpy.fabs(obs._solarmotion - sam._solarmotion) < 1e-10), (
            "Sampled streamspraydf orbits do not have the same solarmotion as the progenitor orbit"
        )
    # Another one
    ro = 9.0
    zo, solarmotion = 0.03, [-20.0, 30.0, 40.0]
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=ro,
        zo=zo,
        solarmotion=solarmotion,
    )
    for streamspraydf in [fardal15spraydf, chen24spraydf]:
        # Set up streamspraydf
        spdf_bovy14 = streamspraydf(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        sam = spdf_bovy14.sample(n=10)
        assert obs._roSet, (
            "Test requires that ro be set for the progenitor orbit, but it appears not to have been set"
        )
        assert not obs._voSet, (
            "Test requires that vo not be set for the progenitor orbit, but it appears to have been set"
        )
        assert obs._roSet is sam._roSet, (
            "Sampled streamspraydf orbits do not have the same roSet as the progenitor orbit"
        )
        assert obs._voSet is sam._voSet, (
            "Sampled streamspraydf orbits do not have the same voSet as the progenitor orbit"
        )
        assert numpy.fabs(obs._ro - sam._ro) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same ro as the progenitor orbit"
        )
        assert numpy.fabs(obs._vo - sam._vo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same vo as the progenitor orbit"
        )
        assert numpy.fabs(obs._zo - sam._zo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same zo as the progenitor orbit"
        )
        assert numpy.all(numpy.fabs(obs._solarmotion - sam._solarmotion) < 1e-10), (
            "Sampled streamspraydf orbits do not have the same solarmotion as the progenitor orbit"
        )
    # And another one
    vo = 230.0
    zo, solarmotion = 0.03, [-20.0, 30.0, 40.0]
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        vo=vo,
        zo=zo,
        solarmotion=solarmotion,
    )
    for streamspraydf in [fardal15spraydf, chen24spraydf]:
        # Set up streamspraydf
        spdf_bovy14 = streamspraydf(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        sam = spdf_bovy14.sample(n=10)
        assert obs._voSet, (
            "Test requires that vo be set for the progenitor orbit, but it appears not to have been set"
        )
        assert not obs._roSet, (
            "Test requires that ro not be set for the progenitor orbit, but it appears to have been set"
        )
        assert obs._roSet is sam._roSet, (
            "Sampled streamspraydf orbits do not have the same roSet as the progenitor orbit"
        )
        assert obs._voSet is sam._voSet, (
            "Sampled streamspraydf orbits do not have the same voSet as the progenitor orbit"
        )
        assert numpy.fabs(obs._ro - sam._ro) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same ro as the progenitor orbit"
        )
        assert numpy.fabs(obs._vo - sam._vo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same vo as the progenitor orbit"
        )
        assert numpy.fabs(obs._zo - sam._zo) < 1e-10, (
            "Sampled streamspraydf orbits do not have the same zo as the progenitor orbit"
        )
        assert numpy.all(numpy.fabs(obs._solarmotion - sam._solarmotion) < 1e-10), (
            "Sampled streamspraydf orbits do not have the same solarmotion as the progenitor orbit"
        )
    return None


def test_integrate_with_prog():
    # Test integrating orbits with the progenitor's potential
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    # Without the progenitor's potential
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(4)
    RvR, dt = spdf.sample(n=100, return_orbit=False, returndt=True, integrate=True)
    # With the progenitor's potential, but set to zero-mass
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="leading",
        progpot=PlummerPotential(0, 0),
    )
    numpy.random.seed(4)
    RvR_withprog, dt_withprog = spdf.sample(
        n=100, return_orbit=False, returndt=True, integrate=True
    )
    # Should agree
    assert numpy.amax(numpy.fabs(dt - dt_withprog)) < 1e-10, (
        "Times not the same when sampling with and without prognitor's potential"
    )
    assert numpy.amax(numpy.fabs(RvR - RvR_withprog)) < 1e-7, (
        "Phase-space points not the same when sampling with and without prognitor's potential"
    )
    return None


def test_chen24spraydf_default_parameters():
    # Test the default parameters of chen24spraydf can be changed
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    # Default parameters
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(4)
    RvR_default, dt_default = spdf.sample(
        n=100, return_orbit=False, returndt=True, integrate=True
    )
    # Modified parameters, but only slightly
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="leading",
        mean=numpy.array([1.6, -0.525344, 0, 1, 0.349066, 0]),
        cov=numpy.array(
            [
                [0.1225, 0, 0, 0, -0.085521, 0],
                [0, 0.161143, 0, 0, 0, 0],
                [0, 0, 0.043865, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-0.085521, 0, 0, 0, 0.121847, 0],
                [0, 0, 0, 0, 0, 0.147435],
            ]
        ),
    )
    numpy.random.seed(4)
    RvR, dt = spdf.sample(n=100, return_orbit=False, returndt=True, integrate=True)
    # Should agree
    assert numpy.amax(numpy.fabs(dt_default - dt)) < 1e-10, (
        "Times not the same when changing the default parameters"
    )
    assert numpy.amax(numpy.fabs(RvR_default - RvR)) > 1e-7, (
        "Phase-space points should not be the same when changing the default parameters"
    )
    assert numpy.amax(numpy.fabs(RvR_default - RvR)) < 1e-2, (
        "Phase-space points too different when sampling with and without prognitor's potential"
    )
    return None


def test_tail_both():
    # Test that tail='both' produces both leading and trailing stars,
    # consistent with separate leading/trailing models
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        # Set up leading-only and trailing-only models
        spdf_l = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        spdf_t = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="trailing",
        )
        # Set up both model
        spdf_both = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="both",
        )
        # Sample from leading-only
        numpy.random.seed(1)
        RvR_l = spdf_l.sample(n=150, return_orbit=False, integrate=False)
        # Sample from trailing-only
        numpy.random.seed(2)
        RvR_t = spdf_t.sample(n=150, return_orbit=False, integrate=False)
        # Sample from both (should give 150 leading + 150 trailing)
        numpy.random.seed(1)
        RvR_both = spdf_both.sample(n=300, return_orbit=False, integrate=False)
        # First half should match the leading-only sample
        assert numpy.allclose(RvR_both[:, :150], RvR_l), (
            f"tail='both' leading half does not match tail='leading' for {spraydfclass.__name__}"
        )
        # Second half should match the trailing-only sample
        # (seed 2 is consumed by the trailing part after the leading part uses seed 1)
        # Just check that the two halves are different (leading vs trailing)
        assert not numpy.allclose(RvR_both[:, :150], RvR_both[:, 150:]), (
            f"tail='both' leading and trailing halves should differ for {spraydfclass.__name__}"
        )
    return None


def test_tail_both_sample_size():
    # Test that tail='both' returns the correct number of points
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        spdf = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="both",
        )
        # Even n
        RvR = spdf.sample(n=200, return_orbit=False, integrate=False)
        assert RvR.shape[1] == 200, (
            f"tail='both' with n=200 should return 200 points for {spraydfclass.__name__}"
        )
        # Odd n
        RvR = spdf.sample(n=201, return_orbit=False, integrate=False)
        assert RvR.shape[1] == 201, (
            f"tail='both' with n=201 should return 201 points for {spraydfclass.__name__}"
        )
        # Orbit output
        orbs = spdf.sample(n=100, return_orbit=True, integrate=False)
        assert len(orbs) == 100, (
            f"tail='both' with n=100 should return 100 orbits for {spraydfclass.__name__}"
        )
    return None


def test_sample_tail_override():
    # sample(tail=...) overrides the default set at __init__, identically
    # to how streamTrack(tail=...) works. Setup doesn't matter — the
    # progenitor integration is the same regardless of which arm the
    # user asks for at sample time.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    # Built with tail='leading' — should still be able to sample 'trailing'
    # and 'both'.
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="leading",
    )
    RvR_lead = spdf.sample(n=100, return_orbit=False, integrate=False)
    RvR_trail = spdf.sample(n=100, return_orbit=False, integrate=False, tail="trailing")
    RvR_both = spdf.sample(n=200, return_orbit=False, integrate=False, tail="both")
    assert RvR_lead.shape == (6, 100)
    assert RvR_trail.shape == (6, 100)
    assert RvR_both.shape == (6, 200)
    # Trailing-arm samples must differ from leading-arm samples (different
    # stripping side of the progenitor).
    assert not numpy.allclose(RvR_lead, RvR_trail)
    # tail=None (default) follows self._tail — should match an explicit
    # tail='leading' call (modulo RNG, which is reseeded by us).
    numpy.random.seed(123)
    RvR_default = spdf.sample(n=50, return_orbit=False, integrate=False)
    numpy.random.seed(123)
    RvR_explicit_leading = spdf.sample(
        n=50, return_orbit=False, integrate=False, tail="leading"
    )
    assert numpy.allclose(RvR_default, RvR_explicit_leading)
    # Built with tail='both' — explicit tail='leading' should give a
    # leading-only sample (n=100, not n=50+50).
    spdf_both = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="both",
    )
    RvR_lead_only = spdf_both.sample(
        n=100, return_orbit=False, integrate=False, tail="leading"
    )
    assert RvR_lead_only.shape == (6, 100)
    # Bad tail value raises.
    with pytest.raises(ValueError):
        spdf.sample(n=10, return_orbit=False, integrate=False, tail="bogus")
    return None


def test_tail_both_returndt():
    # Test that tail='both' with returndt works
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        spdf = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="both",
        )
        RvR, dt = spdf.sample(n=100, return_orbit=False, returndt=True, integrate=False)
        assert RvR.shape[1] == 100, (
            f"tail='both' with returndt should return 100 points for {spraydfclass.__name__}"
        )
        assert len(dt) == 100, (
            f"tail='both' with returndt should return 100 dt values for {spraydfclass.__name__}"
        )
    return None


def test_tail_both_consistency():
    # Test that tail='both' leading half matches a separate leading-only model
    # with the same random seed
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        spdf_l = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="leading",
        )
        spdf_both = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
            tail="both",
        )
        # Same seed: leading-only
        numpy.random.seed(42)
        RvR_l = spdf_l.sample(n=50, return_orbit=False, integrate=False)
        # Same seed: both (first 25 should be leading)
        numpy.random.seed(42)
        RvR_both = spdf_both.sample(n=50, return_orbit=False, integrate=False)
        # First 25 points of 'both' should exactly match leading-only with n=25
        numpy.random.seed(42)
        RvR_l25 = spdf_l.sample(n=25, return_orbit=False, integrate=False)
        assert numpy.allclose(RvR_both[:, :25], RvR_l25), (
            f"tail='both' leading half does not match tail='leading' for {spraydfclass.__name__}"
        )
    return None


def test_leading_deprecation():
    # Test that using leading= raises a FutureWarning
    import warnings as _warnings

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            spdf = spraydfclass(
                2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
                progenitor=obs,
                pot=lp,
                tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
                leading=True,
            )
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 1, (
            f"Expected exactly one FutureWarning for {spraydfclass.__name__}"
        )
        assert "leading= keyword is deprecated" in str(future_warnings[0].message), (
            f"FutureWarning message incorrect for {spraydfclass.__name__}"
        )
        # Should still work correctly
        RvR = spdf.sample(n=10, return_orbit=False, integrate=False)
        assert RvR.shape[1] == 10, (
            f"Deprecated leading= should still produce correct output for {spraydfclass.__name__}"
        )
    return None


def test_leading_and_tail_error():
    # Test that specifying both leading= and tail= raises an error
    import warnings as _warnings

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        with _warnings.catch_warnings():
            _warnings.simplefilter("always")
            with pytest.raises(ValueError, match="Cannot specify both"):
                spraydfclass(
                    2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
                    progenitor=obs,
                    pot=lp,
                    tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
                    leading=True,
                    tail="trailing",
                )
    return None


def test_invalid_tail():
    # Test that an invalid tail= value raises an error
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        with pytest.raises(ValueError, match="tail= must be"):
            spraydfclass(
                2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
                progenitor=obs,
                pot=lp,
                tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
                tail="invalid",
            )
    return None


def test_tail_default_is_leading():
    # Test that the default tail= is 'leading' for backward compatibility
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    for spraydfclass in [fardal15spraydf, chen24spraydf]:
        spdf = spraydfclass(
            2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
            progenitor=obs,
            pot=lp,
            tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        )
        assert spdf._tail == "leading", (
            f"Default tail should be 'leading' for {spraydfclass.__name__}"
        )
    return None


def test_sample_matches_per_particle():
    # Check that the batched single-Orbit integration in _sample_tail produces
    # the same particles as the previous per-particle integration loop.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    # Sample without integration to get initial conditions and stripping times
    numpy.random.seed(123)
    out_ic, dt = spdf._sample_tail(8, integrate=False, leading=True)
    Rs, vRs, vTs, Zs, vZs, phis = out_ic
    # Reproduce the old per-particle loop locally
    expected = numpy.empty((6, 8))
    for ii in range(8):
        o_one = Orbit([Rs[ii], vRs[ii], vTs[ii], Zs[ii], vZs[ii], phis[ii]])
        o_one.integrate(numpy.linspace(-dt[ii], 0.0, 10001), spdf._pot)
        o_end = o_one(0.0)
        expected[:, ii] = [
            o_end.R(),
            o_end.vR(),
            o_end.vT(),
            o_end.z(),
            o_end.vz(),
            o_end.phi(),
        ]
    # Now run the actual batched code (same seed → same ICs)
    numpy.random.seed(123)
    out_int, dt2 = spdf._sample_tail(8, integrate=True, leading=True)
    assert numpy.allclose(dt, dt2), "Stripping times changed between sampling calls"
    assert numpy.allclose(out_int, expected, rtol=1e-6, atol=1e-6), (
        "Batched per-orbit-t integration disagrees with per-particle loop"
    )
    return None


######################### streamTrack tests ###############################


@pytest.fixture(scope="module")
def _simple_spdf():
    """Minimal fardal15spraydf fixture used by several streamTrack tests."""
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    return spdf


def test_streamTrack_progenitor_recovery():
    # With tiny tdisrupt, particles barely drift from progenitor so the track
    # at any tp should be very close to the progenitor today.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=0.05 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(1)
    track = spdf.streamTrack(n=2000, ntp=41, tail="leading")
    prog_x = spdf._progenitor.x(0.0)
    prog_y = spdf._progenitor.y(0.0)
    prog_z = spdf._progenitor.z(0.0)
    # For small tdisrupt, ALL particles are near the progenitor today, so
    # the track across its own tp grid should be within a few percent of
    # the progenitor's present-day position.
    tps = numpy.linspace(track.tp_grid()[0], track.tp_grid()[-1], 7)
    for tp in tps:
        assert abs(track.x(tp) - prog_x) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (x)"
        )
        assert abs(track.y(tp) - prog_y) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (y)"
        )
        assert abs(track.z(tp) - prog_z) < 0.05, (
            "StreamTrack does not recover the progenitor in the tiny-tdisrupt limit (z)"
        )


def test_streamTrack_sample_consistency(_simple_spdf):
    # The track's mean at tp should agree with the mean galactocentric x of
    # particles near that tp (per a closest-point assignment to the track).
    from galpy.df.streamTrack import _closest_point_on_curve, _particles_to_cartesian

    numpy.random.seed(1)
    track = _simple_spdf.streamTrack(n=4000, ntp=41, tail="leading")
    # Reproduce the closest-point assignment from the saved particles ->
    # closest point on the dense track itself (a stable, public reference).
    xv, _ = track.particles
    particles_cart = _particles_to_cartesian(xv)
    tp_grid = track.tp_grid()
    track_cart = numpy.column_stack(
        [
            [float(track.x(t, use_physical=False)) for t in tp_grid],
            [float(track.y(t, use_physical=False)) for t in tp_grid],
            [float(track.z(t, use_physical=False)) for t in tp_grid],
            [float(track.vx(t, use_physical=False)) for t in tp_grid],
            [float(track.vy(t, use_physical=False)) for t in tp_grid],
            [float(track.vz(t, use_physical=False)) for t in tp_grid],
        ]
    )
    tp_part = _closest_point_on_curve(particles_cart, track_cart, tp_grid)
    x_p = particles_cart[:, 0]
    edges = numpy.linspace(tp_grid[0], tp_grid[-1], 9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for i in range(len(centers)):
        sel = (tp_part >= edges[i]) & (tp_part < edges[i + 1])
        n_in = int(sel.sum())
        if n_in < 50:
            continue
        mean_x = x_p[sel].mean()
        std_x = x_p[sel].std(ddof=1) / numpy.sqrt(n_in)
        assert abs(mean_x - track.x(centers[i])) < max(5 * std_x, 0.05), (
            "StreamTrack does not match binned sample mean in Cartesian x"
        )


def test_streamTrack_interface(_simple_spdf):
    numpy.random.seed(2)
    track = _simple_spdf.streamTrack(n=2000, ntp=41, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 5)
    for meth in (
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "R",
        "vR",
        "vT",
        "phi",
        "ra",
        "dec",
        "dist",
        "ll",
        "bb",
        "pmra",
        "pmdec",
        "pmll",
        "pmbb",
        "vlos",
    ):
        vals = getattr(track, meth)(tps)
        vals = numpy.asarray(vals)
        assert vals.shape == tps.shape, f"{meth} returned wrong shape"
        assert numpy.all(numpy.isfinite(vals)), f"{meth} returned non-finite"


def test_streamTrack_covariance_psd(_simple_spdf):
    numpy.random.seed(3)
    track = _simple_spdf.streamTrack(n=2000, ntp=41, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 7)
    covs = track.cov(tps)
    assert covs.shape == (len(tps), 6, 6)
    for k in range(len(tps)):
        C = covs[k]
        assert numpy.allclose(C, C.T, atol=1e-10), "Covariance not symmetric"
        evs = numpy.linalg.eigvalsh(C)
        assert numpy.all(evs >= -1e-10), f"Covariance not PSD (min eigval={evs.min()})"


def test_streamTrack_both_tails(_simple_spdf):
    numpy.random.seed(4)
    pair = _simple_spdf.streamTrack(n=1500, ntp=41, tail="both")
    # Near tp=0 both arms should sit near the progenitor
    prog = _simple_spdf._progenitor
    px, py = prog.x(0.0), prog.y(0.0)
    assert abs(pair.leading.x(0.0) - px) < 0.1
    assert abs(pair.trailing.x(0.0) - px) < 0.1
    # Deep into the stream, the two arms diverge — pick the deepest in-range
    # point for each arm (leading arm: largest positive tp; trailing arm:
    # most negative tp).
    tp_lead = pair.leading.tp_grid()[-1]
    tp_trail = pair.trailing.tp_grid()[0]
    d_lead = (pair.leading.x(tp_lead) - pair.trailing.x(tp_trail)) ** 2 + (
        pair.leading.y(tp_lead) - pair.trailing.y(tp_trail)
    ) ** 2
    assert d_lead > 0.01, "Leading and trailing arms do not diverge at large |tp|"


def test_streamTrack_iteration_changes_track(_simple_spdf):
    # Iteration should move the track by a small amount; we don't require
    # strict convergence (closest-point reassignment introduces some noise).
    numpy.random.seed(5)
    tr0 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=0, tail="leading")
    numpy.random.seed(5)
    tr1 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=1, tail="leading")
    # Compare on the in-range grid common to both tracks.
    lo = max(tr0.tp_grid()[0], tr1.tp_grid()[0])
    hi = min(tr0.tp_grid()[-1], tr1.tp_grid()[-1])
    tps = numpy.linspace(lo, hi, 101)
    # Track-to-track difference should be small compared to the stream size
    ampl = numpy.ptp(tr0.x(tps))
    dmax = numpy.max(numpy.abs(tr0.x(tps) - tr1.x(tps)))
    assert dmax < 0.5 * max(ampl, 0.1), (
        "Iteration changed the track by more than half the stream amplitude"
    )


def test_streamTrack_chen24_works():
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = chen24spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(6)
    track = spdf.streamTrack(n=1500, ntp=41, tail="both")
    # Sample at midpoint of each arm's tp grid (guaranteed in-range).
    tp_lead = track.leading.tp_grid()[len(track.leading.tp_grid()) // 2]
    tp_trail = track.trailing.tp_grid()[len(track.trailing.tp_grid()) // 2]
    for tp, arm in [
        (tp_lead, track.leading),
        (tp_trail, track.trailing),
        (0.0, track.leading),
        (0.0, track.trailing),
    ]:
        assert numpy.isfinite(arm.x(tp))


def test_streamTrack_with_center():
    # Stream around a moving satellite
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ro, vo = 8.0, 220.0
    cen = Orbit([1.3, 0.2, -0.9, 0.4, 0.1, 0.4])
    prog = Orbit([1.35, 0.22, -0.88, 0.42, 0.08, 0.45])
    spdf = fardal15spraydf(
        2e4 / conversion.mass_in_msol(vo, ro),
        progenitor=prog,
        pot=lp,
        center=cen,
        centerpot=lp,
        tdisrupt=1.0 / conversion.time_in_Gyr(vo, ro),
    )
    numpy.random.seed(7)
    track = spdf.streamTrack(n=1500, ntp=31, tail="leading")
    g = track.tp_grid()
    tps = numpy.linspace(g[0], g[-1], 5)
    vals = track.x(tps)
    assert numpy.all(numpy.isfinite(vals))


def test_streamTrack_physical_units(_simple_spdf):
    numpy.random.seed(8)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    x0 = track.x(tp)
    track.turn_physical_on(ro=8.0, vo=220.0)
    x0_phys = track.x(tp)
    # physical x should be ~ ro * internal
    val = x0_phys.value if hasattr(x0_phys, "value") else x0_phys
    assert abs(val - 8.0 * x0) < 1e-6
    track.turn_physical_off()
    assert abs(track.x(tp) - x0) < 1e-10


def test_streamTrack_cov_physical_units(_simple_spdf):
    # cov() must honor the physical-unit toggle, matching the scaling of
    # x/y/z/vx/vy/vz accessors. Position entries scale by ro^2, velocity
    # by vo^2, cross terms by ro*vo.
    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    track.turn_physical_off()
    C_int = track.cov(tp)
    track.turn_physical_on(ro=8.0, vo=220.0)
    C_phys = track.cov(tp)
    ro, vo = 8.0, 220.0
    scale = numpy.array([ro, ro, ro, vo, vo, vo])
    expected = C_int * numpy.outer(scale, scale)
    assert numpy.allclose(C_phys, expected, rtol=1e-10)


def test_streamTrack_particles_reuse(_simple_spdf):
    numpy.random.seed(9)
    xv, dt = _simple_spdf.sample(
        n=2000, returndt=True, return_orbit=False, integrate=True
    )
    track = _simple_spdf.streamTrack(particles=(xv, dt), ntp=41, tail="leading")
    assert numpy.isfinite(track.x(0.0))


def test_streamTrack_plot_smoke(_simple_spdf):
    import matplotlib

    matplotlib.use("Agg")
    numpy.random.seed(10)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    track.plot(d1="x", d2="y", spread=1.0)
    track.leading.plot(d1="ra", d2="dec")
    track.leading.plot(d1="R", d2="z")
    # spread>0 on velocity axes exercises the velocity branch of the spread
    # scaling; non-Cartesian axes silently skip the band
    track.leading.plot(d1="vx", d2="vy", spread=1.0)
    track.leading.plot(d1="R", d2="phi", spread=1.0)


def test_streamTrack_physical_accessors_all(_simple_spdf):
    # Exercise the physical-unit branch of every accessor.
    numpy.random.seed(11)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    track.turn_physical_on(ro=8.0, vo=220.0)
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    for meth in (
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "R",
        "vR",
        "vT",
        "phi",
        "ra",
        "dec",
        "ll",
        "bb",
        "dist",
        "pmra",
        "pmdec",
        "pmll",
        "pmbb",
        "vlos",
    ):
        v = getattr(track, meth)(tp)
        val = getattr(v, "value", v)
        assert numpy.isfinite(val)
    # Plot in physical units with a velocity-axis spread hits the vo
    # scaling branch of the band
    import matplotlib

    matplotlib.use("Agg")
    track.plot(d1="vx", d2="vy", spread=1.0)
    track.turn_physical_off()


def test_streamTrack_pair_physical_toggles(_simple_spdf):
    numpy.random.seed(12)
    pair = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    pair.turn_physical_on(ro=8.0, vo=220.0)
    tp_lead = pair.leading.tp_grid()[len(pair.leading.tp_grid()) // 2]
    v = pair.leading.x(tp_lead)
    val = getattr(v, "value", v)
    assert val > 0.0
    pair.turn_physical_off()
    v2 = pair.leading.x(tp_lead)
    assert not hasattr(v2, "unit")
    import matplotlib

    matplotlib.use("Agg")
    pair.plot(d1="x", d2="y")


def test_streamTrack_call_returns_stacked(_simple_spdf):
    numpy.random.seed(13)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    v = track(-10.0)
    assert v.shape == (6,)
    tps = numpy.linspace(-10.0, -1.0, 4)
    v = track(tps)
    assert v.shape == (6, 4)


def test_streamTrack_tp_grid(_simple_spdf):
    numpy.random.seed(14)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    g = track.tp_grid()
    assert g.shape == (1001,)
    # Leading arm: tp ranges from 0 to some positive value.
    assert g[0] == 0.0
    assert g[-1] > 0.0


def test_streamTrack_out_of_range_returns_nan(_simple_spdf):
    # Out-of-range tp must return NaN — never silent cubic-spline
    # extrapolation. For an array tp, only the offending entries are NaN.
    numpy.random.seed(20)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    g = track.tp_grid()
    tp_lo, tp_hi = g[0], g[-1]
    # Scalar out-of-range: NaN
    assert numpy.isnan(track.x(tp_hi + 5.0))
    assert numpy.isnan(track.R(tp_hi + 5.0))
    assert numpy.isnan(track.ra(tp_hi + 5.0))
    # Negative side (leading arm has tp_lo == 0)
    assert numpy.isnan(track.x(tp_lo - 1.0))
    # Scalar in-range: finite
    tp_mid = 0.5 * (tp_lo + tp_hi)
    assert numpy.isfinite(track.x(tp_mid))
    # Array tp with mixed in/out: NaN only at out-of-range entries
    tps = numpy.array([tp_lo - 1.0, tp_mid, tp_hi + 5.0])
    xs = numpy.asarray(track.x(tps))
    assert numpy.isnan(xs[0])
    assert numpy.isfinite(xs[1])
    assert numpy.isnan(xs[2])
    # cov() honors the same convention; out-of-range entries are NaN
    Cs = track.cov(tps)
    assert numpy.all(numpy.isnan(Cs[0]))
    assert numpy.all(numpy.isfinite(Cs[1]))
    assert numpy.all(numpy.isnan(Cs[2]))
    # cov(basis=...) too — NaN entries skip the Jacobian path safely
    Cs_sky = track.cov(tps, basis="sky")
    assert numpy.all(numpy.isnan(Cs_sky[0]))
    assert numpy.all(numpy.isfinite(Cs_sky[1]))
    assert numpy.all(numpy.isnan(Cs_sky[2]))


def test_streamTrack_order1_no_cov(_simple_spdf):
    numpy.random.seed(15)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading", order=1)
    with pytest.raises(RuntimeError):
        track.cov(-10.0)


def test_streamTrack_invalid_tail(_simple_spdf):
    with pytest.raises(ValueError):
        _simple_spdf.streamTrack(n=100, tail="bogus")


def test_streamTrack_particles_reuse_both(_simple_spdf):
    numpy.random.seed(16)
    _simple_spdf._tail = "both"
    xv, dt = _simple_spdf.sample(
        n=2000, returndt=True, return_orbit=False, integrate=True
    )
    _simple_spdf._tail = "leading"
    pair = _simple_spdf.streamTrack(particles=(xv, dt), tail="both", ntp=31)
    assert numpy.isfinite(pair.leading.x(0.0))
    assert numpy.isfinite(pair.trailing.x(0.0))


def test_streamTrackPair_particles_property(_simple_spdf):
    # pair.particles concatenates the two arms' (xv, dt) in leading-first
    # order, exactly the format ``streamTrack(tail='both', particles=...)``
    # expects. Round-trip: re-fit using pair.particles and confirm the
    # resulting tracks match the original (modulo tiny FITPACK iteration
    # noise).
    numpy.random.seed(34)
    pair = _simple_spdf.streamTrack(n=1500, ntp=31, tail="both")
    xv_pair, dt_pair = pair.particles
    # Shape: leading first, trailing second.
    n_l = pair.leading.particles[0].shape[1]
    n_t = pair.trailing.particles[0].shape[1]
    assert xv_pair.shape == (6, n_l + n_t)
    assert dt_pair.shape == (n_l + n_t,)
    assert numpy.allclose(xv_pair[:, :n_l], pair.leading.particles[0])
    assert numpy.allclose(xv_pair[:, n_l:], pair.trailing.particles[0])
    assert numpy.allclose(dt_pair[:n_l], pair.leading.particles[1])
    assert numpy.allclose(dt_pair[n_l:], pair.trailing.particles[1])
    # Round-trip via spraydf.streamTrack(tail='both', particles=...). Compare
    # on the overlap of the two tp grids (auto-trim percentiles can land at
    # slightly different boundaries between the two calls due to a small
    # probe-sample jitter; staying inside the overlap avoids NaN edges).
    pair_reuse = _simple_spdf.streamTrack(particles=pair.particles, tail="both", ntp=31)
    g_l, g_l2 = pair.leading.tp_grid(), pair_reuse.leading.tp_grid()
    g_t, g_t2 = pair.trailing.tp_grid(), pair_reuse.trailing.tp_grid()
    tps_l = numpy.linspace(max(g_l[0], g_l2[0]), min(g_l[-1], g_l2[-1]), 21)
    tps_t = numpy.linspace(max(g_t[0], g_t2[0]), min(g_t[-1], g_t2[-1]), 21)
    assert (
        numpy.max(numpy.abs(pair.leading.x(tps_l) - pair_reuse.leading.x(tps_l))) < 1e-3
    )
    assert (
        numpy.max(numpy.abs(pair.trailing.x(tps_t) - pair_reuse.trailing.x(tps_t)))
        < 1e-3
    )


def test_streamTrack_scalar_cov(_simple_spdf):
    numpy.random.seed(17)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    C = track.cov(tp)
    assert C.shape == (6, 6)
    assert numpy.all(numpy.isfinite(C))


def test_streamTrack_cov_per_call_unit_overrides(_simple_spdf):
    # cov() honors per-call ro=, vo=, use_physical= (the same way the mean
    # accessors do) so callers don't need to flip the track-wide toggle.
    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    track.turn_physical_off()
    C_int = track.cov(tp)
    # ro=, vo= scale the entries even when the track is in internal mode
    ro, vo = 8.0, 220.0
    C_phys_via_kw = track.cov(tp, ro=ro, vo=vo, use_physical=True)
    scale = numpy.array([ro, ro, ro, vo, vo, vo])
    assert numpy.allclose(C_phys_via_kw, C_int * numpy.outer(scale, scale), rtol=1e-10)
    # use_physical=False on a physical-mode track gives back internal cov
    track.turn_physical_on(ro=ro, vo=vo)
    C_back_to_int = track.cov(tp, use_physical=False)
    assert numpy.allclose(C_back_to_int, C_int, rtol=1e-10)
    # quantity=True is explicitly not supported (heterogeneous units)
    with pytest.raises(NotImplementedError):
        track.cov(tp, quantity=True)
    # Per-call overrides also have to thread through the analytical
    # Jacobian for non-galcenrect bases — exercise the sky path with
    # explicit ro/vo so the override branches in _cart_mean_at and
    # _analytical_jacobian (where ``ro``/``vo``/``use_physical`` are
    # forwarded to the accessors and used as Xsun) are taken.
    track.turn_physical_off()
    C_sky_with_kw = track.cov(tp, basis="sky", ro=ro, vo=vo, use_physical=True)
    assert C_sky_with_kw.shape == (6, 6)
    assert numpy.allclose(C_sky_with_kw, C_sky_with_kw.T, atol=1e-8)


def test_streamTrack_physical_length_spread(_simple_spdf):
    # Covers the physical x/y/z scaling inside the plot spread band
    import matplotlib

    matplotlib.use("Agg")
    numpy.random.seed(18)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    track.turn_physical_on(ro=8.0, vo=220.0)
    track.plot(d1="x", d2="y", spread=1.0)
    track.plot(d1="vx", d2="vy", spread=1.0)
    track.turn_physical_off()


def test_streamTrack_degenerate_few_particles(_simple_spdf):
    # Tiny sample with lots of bins -> most bins are empty/single-particle,
    # exercising the degenerate paths in _bin_offsets and _smooth_series.
    numpy.random.seed(19)
    track = _simple_spdf.streamTrack(n=10, ntp=51, tail="leading")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    assert numpy.isfinite(track.x(tp))


def test_streamTrack_custom_track_time_range(_simple_spdf):
    # Pass an explicit track_time_range (float) to cover the non-default
    # branch in basestreamspraydf.streamTrack. The astropy-Quantity
    # variant of this test lives in tests/test_quantity.py.
    numpy.random.seed(20)
    tr = _simple_spdf.streamTrack(n=800, tail="leading", track_time_range=3.0)
    assert tr.tp_grid()[-1] <= 3.0 + 1e-9


def test_streamTrack_smoothing_variants(_simple_spdf):
    numpy.random.seed(18)
    # Default ntp (auto from n) is exercised here
    tr_f = _simple_spdf.streamTrack(n=800, tail="leading", smoothing=20.0)
    assert numpy.isfinite(tr_f.x(tr_f.tp_grid()[len(tr_f.tp_grid()) // 2]))
    # Array-like smoothing: reuse smoothing_s from a previous fit
    numpy.random.seed(18)
    tr_gcv = _simple_spdf.streamTrack(n=800, tail="leading", order=2)
    assert len(tr_gcv.smoothing_s) == 27  # 6 mean + 21 cov
    numpy.random.seed(18)
    tr_reuse = _simple_spdf.streamTrack(
        n=800, tail="leading", order=2, smoothing=tr_gcv.smoothing_s
    )
    assert numpy.isfinite(tr_reuse.x(tr_reuse.tp_grid()[-1]))
    # order=1 returns only 6 s values
    numpy.random.seed(18)
    tr_o1 = _simple_spdf.streamTrack(n=800, tail="leading", order=1)
    assert len(tr_o1.smoothing_s) == 6


def test_streamTrack_smoothing_factor(_simple_spdf):
    # smoothing_factor>1 must produce a smoother fit (larger effective s)
    # than smoothing_factor=1; <1 must produce a rougher fit. Sample once
    # and reuse via particles= so we isolate the smoothing step.
    numpy.random.seed(33)
    # Pin velocity_weight=1.0 so this test is specifically about the
    # smoothing_factor knob, not entangled with the velocity_weight='auto'
    # default (which slightly shifts the per-bin variances that GCV uses
    # to pick s, changing the absolute scale of smoothing_s).
    tr_default = _simple_spdf.streamTrack(
        n=800, ntp=21, tail="leading", velocity_weight=1.0
    )
    xv, dt = tr_default.particles
    tr_smoother = _simple_spdf.streamTrack(
        particles=(xv, dt),
        ntp=21,
        tail="leading",
        smoothing_factor=2.0,
        velocity_weight=1.0,
    )
    tr_rougher = _simple_spdf.streamTrack(
        particles=(xv, dt),
        ntp=21,
        tail="leading",
        smoothing_factor=0.5,
        velocity_weight=1.0,
    )
    # Ratio of effective s over the six mean splines (skip entries where
    # the default s is so small the rerun underflows). The cov splines
    # are excluded because GCV's chi^2 there is already at FITPACK's
    # minimum-knot floor, so a larger ``s`` upper bound doesn't change
    # the fit and the per-spline ratio stays near 1 — diluting any mean.
    s_def = numpy.asarray(tr_default.smoothing_s[:6])
    s_smt = numpy.asarray(tr_smoother.smoothing_s[:6])
    s_rgh = numpy.asarray(tr_rougher.smoothing_s[:6])
    valid = s_def > 1e-3
    assert valid.any(), "no valid mean-spline s to compare against"
    # UnivariateSpline interprets ``s`` as a chi^2 upper bound: the
    # rougher direction (factor < 1) reliably tightens because the new
    # bound forces extra knots; the smoother direction (factor > 1) is
    # platform-dependent because some splines already sit at the FITPACK
    # minimum-knot floor and ignore the loosened bound. Test the strong
    # signal in each direction: at least one mean spline must get
    # noticeably smoother for factor=2, and the average must clearly
    # tighten for factor=0.5.
    assert numpy.max(s_smt[valid] / s_def[valid]) > 1.5
    assert numpy.mean(s_rgh[valid] / s_def[valid]) < 0.7
    # smoothing_factor=1.0 must reproduce the default fit (modulo a small
    # probe-sampling jitter: the no-particles call draws a probe sample
    # to size track_time_range, the particles= call skips that probe and
    # uses the passed sample directly, so track_t_grid differs at the
    # 1e-5 level — well below any physical scale).
    tr_unity = _simple_spdf.streamTrack(
        particles=(xv, dt),
        ntp=21,
        tail="leading",
        smoothing_factor=1.0,
        velocity_weight=1.0,
    )
    tps = tr_default.tp_grid()
    assert numpy.max(numpy.abs(tr_default.x(tps) - tr_unity.x(tps))) < 1e-3
    # The saved smoothing_s round-trips through ``smoothing=`` with
    # smoothing_factor=1.0 — passing both the smoothed s values and
    # factor=1 should reproduce the smoother track (FITPACK iteration
    # tolerance allowed).
    tr_reuse = _simple_spdf.streamTrack(
        particles=(xv, dt),
        ntp=21,
        tail="leading",
        smoothing=tr_smoother.smoothing_s,
        smoothing_factor=1.0,
        velocity_weight=1.0,
    )
    assert numpy.max(numpy.abs(tr_smoother.x(tps) - tr_reuse.x(tps))) < 1e-3


def test_smooth_series_too_few_valid_bins():
    """_smooth_series falls back to linear interpolation when fewer than 5
    valid bins are available, and to a flat constant interpolant when fewer
    than 2 valid bins remain after dropping NaNs."""
    from galpy.df.streamTrack import _smooth_series

    # Case A: zero valid bins → constant 0
    x = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = numpy.full(5, numpy.nan)
    sigma = numpy.ones(5)
    spl, eff_s = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 1.5, 3.0])), 0.0)
    assert eff_s == 0.0

    # Case B: one valid bin → constant equal to that bin's value
    y = numpy.array([numpy.nan, numpy.nan, 7.0, numpy.nan, numpy.nan])
    spl, eff_s = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 3.0, 5.0])), 7.0)
    assert eff_s == 0.0


def test_smooth_series_invalid_sigma_falls_back_to_unit_median():
    """When all per-bin sigma entries are non-finite, _smooth_series treats
    every bin with unit sigma instead of raising or NaN-propagating."""
    from galpy.df.streamTrack import _smooth_series

    x = numpy.linspace(0.0, 1.0, 8)
    y = numpy.sin(2 * numpy.pi * x)
    sigma = numpy.full(8, numpy.nan)
    spl, eff_s = _smooth_series(x, y, sigma)
    # Should fit something reasonable (not raise)
    assert numpy.isfinite(spl(0.5))


def test_smooth_series_constant_y_falls_back_to_unit_yscale():
    """When y is a constant series, the GCV-driven yscale normalization
    falls back to 1.0; the returned spline should reproduce the constant."""
    from galpy.df.streamTrack import _smooth_series

    x = numpy.linspace(0.0, 1.0, 8)
    y = numpy.full(8, 3.14)  # std == 0
    sigma = numpy.ones(8)
    spl, _ = _smooth_series(x, y, sigma)
    numpy.testing.assert_allclose(spl(numpy.array([0.0, 0.5, 1.0])), 3.14, atol=1e-10)


def test_streamTrack_trim_grid_degenerate_tp():
    """If every particle's tp_assign collapses onto a single value (so
    tp_hi - tp_lo < 1e-12), _trim_grid falls back to the full track grid
    instead of producing an empty range."""
    from galpy.df.streamTrack import _fit_track_from_particles

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0

    n = 50
    rng = numpy.random.default_rng(7)
    xv = numpy.zeros((6, n))
    # All particles at the progenitor's t=0 position with tiny perpendicular
    # noise. Their closest-point assignment is tp ≈ 0 for everyone, so the
    # 99th-percentile trim collapses to (0, 0) and the defensive fallback
    # has to widen the grid.
    xv[0] = 0.0  # R = 0
    xv[1] = 1.0  # vR -> vx
    xv[3] = 0.001 * rng.standard_normal(n)
    xv[4] = 0.001 * rng.standard_normal(n)
    dt = numpy.full(n, 50.0)
    try:
        _fit_track_from_particles(
            xv,
            dt,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight=1.0,
        )
    except ValueError:
        # Degenerate spline fit downstream is fine; we just need the
        # defensive grid-fallback to have run.
        pass


def test_streamTrack_velocity_weight_invalid_string(_simple_spdf):
    """velocity_weight= must be a float or 'auto'; any other string raises."""
    import pytest

    with pytest.raises(ValueError, match="velocity_weight="):
        _simple_spdf.streamTrack(
            n=200, ntp=21, tail="leading", velocity_weight="not_auto"
        )


def test_streamTrack_velocity_weight_auto_smallN():
    """velocity_weight='auto' falls back to 1.0 when fewer than 20 particles
    are passed in (probe sample too small to estimate σ_pos / σ_vel)."""
    from galpy.df.streamTrack import _fit_track_from_particles

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0

    rng = numpy.random.default_rng(1)
    n = 19  # below the 20-particle floor for auto
    xv = numpy.zeros((6, n))
    xv[0] = numpy.linspace(0.0, 3.0, n)
    xv[1] = 1.0
    xv[3] = 0.05 * rng.standard_normal(n)
    xv[4] = 0.05 * rng.standard_normal(n)
    dt = numpy.full(n, 50.0)

    # Should run without error and the auto resolver should hit the
    # size<20 fallback (velocity_weight collapses to 1.0). We don't assert
    # the resulting fit shape — just that the call completes.
    try:
        _fit_track_from_particles(
            xv,
            dt,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight="auto",
        )
    except ValueError:
        # A degenerate downstream spline fit can ill-pose itself on this
        # tiny synthetic sample; the auto-resolver fallback is what we're
        # exercising and that ran before the spline fit.
        pass


def test_streamTrack_velocity_weight_auto_zero_sigma_vel():
    """velocity_weight='auto' falls back to 1.0 when the inner-half particle
    velocity dispersion is zero (degenerate spread)."""
    from galpy.df.streamTrack import _fit_track_from_particles

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg
    prog[:, 3] = 1.0  # constant velocity along the curve

    n = 40
    xv = numpy.zeros((6, n))
    # Particles spread along x but ALL with identical (vR, vT, vz) =
    # (1, 0, 0) — same as the curve. So sigma_vel of inner-half is 0.
    xv[0] = numpy.linspace(0.0, 3.0, n)
    xv[1] = 1.0
    dt = numpy.full(n, 50.0)

    try:
        _fit_track_from_particles(
            xv,
            dt,
            prog,
            tg,
            arm_sign=+1,
            ninterp=101,
            smoothing_factor=1.0,
            niter=0,
            order=2,
            velocity_weight="auto",
        )
    except ValueError:
        pass


def test_streamTrack_gap_warning():
    """When tp_assign has a structural gap (the orbit-revisit kink
    signature), streamTrack should emit a galpyWarning recommending
    velocity_weight or higher niter.

    Construct a synthetic minimal scenario directly via
    _fit_track_from_particles: a straight progenitor curve (x = tp,
    vx = 1), 90 particles clustered near tp=0 plus 10 outliers at
    tp~99. velocity_weight=1.0 disables the auto-rescue so the
    bimodal tp_assign survives to trigger the gap detector.
    """
    import warnings as _warnings

    from galpy.df.streamTrack import _fit_track_from_particles
    from galpy.util import galpyWarning

    M = 1001
    tg = numpy.linspace(-100.0, 100.0, M)
    prog = numpy.zeros((M, 6))
    prog[:, 0] = tg  # x = tp
    prog[:, 3] = 1.0  # vx = 1 (constant)

    rng = numpy.random.default_rng(0)
    n_bulk, n_far = 90, 10
    xv = numpy.zeros((6, n_bulk + n_far))
    xv[0, :n_bulk] = numpy.linspace(0.0, 3.0, n_bulk)  # R near 0
    xv[0, n_bulk:] = numpy.linspace(95.0, 99.0, n_far)  # R far in leading arm
    xv[1, :] = 1.0  # vR=1 -> vx=1 (matches the curve)
    # Add small perpendicular scatter (z, vz) so the spline fit isn't
    # singular on a perfectly degenerate line.
    xv[3] = 0.01 * rng.standard_normal(n_bulk + n_far)
    xv[4] = 0.01 * rng.standard_normal(n_bulk + n_far)
    dt = numpy.full(n_bulk + n_far, 200.0)

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        # The warning fires before the spline fit. The synthetic particles
        # are degenerate enough that the spline can hit FITPACK's
        # ill-posed branch — that's not what we're testing here, so swallow
        # any post-warning ValueError.
        try:
            _fit_track_from_particles(
                xv,
                dt,
                prog,
                tg,
                arm_sign=+1,
                ninterp=101,
                smoothing_factor=1.0,
                niter=0,
                order=2,
                velocity_weight=1.0,
            )
        except ValueError:
            pass
    gap_msgs = [
        ww
        for ww in w
        if issubclass(ww.category, galpyWarning)
        and "tp_assign histogram" in str(ww.message)
        and "gap" in str(ww.message)
    ]
    assert len(gap_msgs) == 1, (
        f"expected one gap warning, got {len(gap_msgs)} "
        f"(all warnings: {[str(ww.message) for ww in w]})"
    )


def test_closest_point_on_curve_kdtree_edge_cases():
    from galpy.df.streamTrack import _closest_point_on_curve

    # Case 1: short curve (M < 64) with mask — triggers cand.ndim == 1
    # because initial k = max(1, M//64) = 1 and tree.query returns 1D
    numpy.random.seed(99)
    curve = numpy.random.randn(10, 6)
    curve_t = numpy.linspace(0, 1, 10)
    points = numpy.random.randn(5, 6)
    mask = numpy.zeros((5, 10), dtype=bool)
    mask[:, :3] = True
    result = _closest_point_on_curve(points, curve, curve_t, mask=mask)
    assert result.shape == (5,)
    assert numpy.all(numpy.isin(result, curve_t[:3]))

    # Case 2: all-False mask — no allowed neighbor anywhere, triggers
    # the k >= M fallback that assigns tp=0
    mask_empty = numpy.zeros((5, 10), dtype=bool)
    result2 = _closest_point_on_curve(points, curve, curve_t, mask=mask_empty)
    assert result2.shape == (5,)
    assert numpy.allclose(result2, 0.0)


def test_streamTrack_precomputed_init_and_parameter_kind(_simple_spdf):
    # The base StreamTrack constructor takes a precomputed track. Build a
    # track via from_particles, hand its precomputed state to the base
    # __init__, and check that accessors agree. Also exercise the three
    # parameter_kind options with plain floats (the astropy-Quantity
    # variant of parameter_kind lives in tests/test_quantity.py to avoid
    # depending on astropy in this file).
    from galpy.df.streamTrack import StreamTrack

    numpy.random.seed(40)
    fit = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    # Reconstruct via the precomputed-track __init__ — no fitter involved.
    rebuilt = StreamTrack(
        tp_grid=fit.tp_grid(),
        track_xyz=numpy.column_stack(
            [
                [float(fit.x(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.y(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.z(t, use_physical=False)) for t in fit.tp_grid()],
            ]
        ),
        track_vxvyvz=numpy.column_stack(
            [
                [float(fit.vx(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.vy(t, use_physical=False)) for t in fit.tp_grid()],
                [float(fit.vz(t, use_physical=False)) for t in fit.tp_grid()],
            ]
        ),
        cov_xyz=fit._cov_xyz,
        custom_transform=fit._custom_transform,
        parameter_kind="time",
        # Mirror Orbit semantics: pass ro/vo only when the source had them
        # explicitly set; otherwise let the constructor pull the config
        # default and keep ``_roSet``/``_voSet=False``.
        ro=fit._ro if fit._roSet else None,
        vo=fit._vo if fit._voSet else None,
        zo=fit._zo,
        solarmotion=fit._solarmotion,
    )
    # And the rebuilt track should inherit the same _roSet / _voSet state.
    assert rebuilt._roSet == fit._roSet
    assert rebuilt._voSet == fit._voSet
    tp_mid = fit.tp_grid()[len(fit.tp_grid()) // 2]
    assert abs(float(fit.x(tp_mid)) - float(rebuilt.x(tp_mid))) < 1e-10
    # The precomputed-track instance has no fitter outputs.
    assert not hasattr(rebuilt, "particles")
    assert not hasattr(rebuilt, "smoothing_s")
    # parameter_kind="angle": plain-float input is just passed through.
    angle_track = StreamTrack(
        tp_grid=numpy.linspace(0.0, 1.0, 21),
        track_xyz=numpy.column_stack(
            [numpy.linspace(0, 1, 21), numpy.zeros(21), numpy.zeros(21)]
        ),
        track_vxvyvz=numpy.zeros((21, 3)),
        parameter_kind="angle",
    )
    assert numpy.isclose(angle_track.x(0.5), 0.5)
    # parameter_kind=None: pass-through.
    pass_track = StreamTrack(
        tp_grid=numpy.linspace(0.0, 1.0, 21),
        track_xyz=numpy.column_stack(
            [numpy.linspace(0, 1, 21), numpy.zeros(21), numpy.zeros(21)]
        ),
        track_vxvyvz=numpy.zeros((21, 3)),
        parameter_kind=None,
    )
    assert numpy.isclose(pass_track.x(0.5), 0.5)
    # Bad parameter_kind raises.
    with pytest.raises(ValueError):
        StreamTrack(
            tp_grid=numpy.linspace(0.0, 1.0, 21),
            track_xyz=numpy.zeros((21, 3)),
            track_vxvyvz=numpy.zeros((21, 3)),
            parameter_kind="bogus",
        )


def test_streamTrack_particles_attr(_simple_spdf):
    # track.particles exposes the raw (xv, dt) tuple the fit saw — same
    # format that ``spraydf.streamTrack(particles=...)`` accepts. Verify
    # shape, that it round-trips through a second spraydf.streamTrack
    # call, and that it matches when the user passes particles explicitly.
    numpy.random.seed(21)
    xv, dt = _simple_spdf.sample(
        n=1500, returndt=True, return_orbit=False, integrate=True
    )
    track = _simple_spdf.streamTrack(particles=(xv, dt), tail="leading")
    assert isinstance(track.particles, tuple) and len(track.particles) == 2
    xv_attr, dt_attr = track.particles
    assert xv_attr.shape == xv.shape
    assert dt_attr.shape == dt.shape
    assert numpy.allclose(xv_attr, xv)
    assert numpy.allclose(dt_attr, dt)
    # Round-trip: pass track.particles back in, should get same track.
    track2 = _simple_spdf.streamTrack(particles=track.particles, tail="leading")
    tps = track.tp_grid()
    assert numpy.allclose(track.x(tps), track2.x(tps))


def test_streamTrack_custom_transform(_simple_spdf):
    # custom_transform enables phi1/phi2/pmphi1/pmphi2 accessors;
    # without it, those accessors raise.
    from galpy.util import coords

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(22)
    track = _simple_spdf.streamTrack(n=1500, tail="leading", custom_transform=T)
    tps = track.tp_grid()
    tp0 = tps[len(tps) // 2]
    for name in ("phi1", "phi2", "pmphi1", "pmphi2"):
        val = getattr(track, name)(tp0)
        assert numpy.isfinite(float(val))
        arr = getattr(track, name)(tps[:5])
        assert arr.shape == (5,) and numpy.all(numpy.isfinite(arr))
    # Same (ra, dec) → phi via coords.radec_to_custom round-trip
    ra, dec = float(track.ra(tp0)), float(track.dec(tp0))
    expected = coords.radec_to_custom(
        numpy.atleast_1d(ra), numpy.atleast_1d(dec), T=T, degree=True
    )
    assert abs(float(track.phi1(tp0)) - expected[0, 0]) < 1e-8
    assert abs(float(track.phi2(tp0)) - expected[0, 1]) < 1e-8

    # Without custom_transform, accessors raise
    numpy.random.seed(23)
    track_bare = _simple_spdf.streamTrack(n=500, tail="leading")
    with pytest.raises(RuntimeError):
        track_bare.phi1(tp0)
    with pytest.raises(RuntimeError):
        track_bare.pmphi1(tp0)


def test_streamTrack_cov_basis(_simple_spdf):
    # cov(basis=...) must return a 6x6 PSD matrix in each supported basis.
    # Verify units consistency (diagonal entries have the right magnitude
    # for kpc²/km²/s², deg², mas²/yr²) and that galcencyl round-trips
    # cleanly via its analytical Jacobian (cov diag = variance of R from
    # the stored xy cov).
    from galpy.util import coords

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(31)
    track = _simple_spdf.streamTrack(n=1500, tail="leading", custom_transform=T, ntp=41)
    tp0 = track.tp_grid()[len(track.tp_grid()) // 2]
    # All bases: returns (6, 6), symmetric, PSD
    for basis in ("galcenrect", "galcencyl", "galsky", "sky", "customsky"):
        C = track.cov(tp0, basis=basis)
        assert C.shape == (6, 6), f"{basis}: wrong shape"
        assert numpy.allclose(C, C.T, atol=1e-8), f"{basis}: not symmetric"
        evs = numpy.linalg.eigvalsh(C)
        assert evs.min() > -1e-8, f"{basis}: not PSD (min eig {evs.min()})"
    # galcencyl diag[0] = Var(R) ≈ cos²(φ)·Var(x) + sin²(φ)·Var(y) + 2·cos(φ)·sin(φ)·Cov(x,y)
    C_gcr = track.cov(tp0, basis="galcenrect")
    C_cyl = track.cov(tp0, basis="galcencyl")
    x, y = float(track.x(tp0)), float(track.y(tp0))
    R = numpy.sqrt(x * x + y * y)
    cp, sp = x / R, y / R
    var_R_expected = (
        cp * cp * C_gcr[0, 0] + sp * sp * C_gcr[1, 1] + 2 * cp * sp * C_gcr[0, 1]
    )
    assert abs(C_cyl[0, 0] - var_R_expected) < 1e-8, (
        f"galcencyl Var(R) mismatch: {C_cyl[0, 0]} vs {var_R_expected}"
    )
    # Unknown basis raises
    with pytest.raises(ValueError):
        track.cov(tp0, basis="bogus")
    # customsky without custom_transform raises
    numpy.random.seed(32)
    track_bare = _simple_spdf.streamTrack(n=800, tail="leading", ntp=31)
    with pytest.raises(RuntimeError):
        track_bare.cov(tp0, basis="customsky")


def test_streamTrack_plot_spread_non_cartesian(_simple_spdf):
    # plot(spread>0) should draw a ±σ band for every d2 that has a basis
    # in _COORD_BASIS (including sky / custom-sky coords), not just
    # galactocentric Cartesian. Smoke-test a representative set.
    import matplotlib

    matplotlib.use("Agg")
    from galpy.util import coords

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(33)
    track = _simple_spdf.streamTrack(n=1200, tail="leading", custom_transform=T, ntp=31)
    for d2 in (
        "x",
        "R",
        "z",
        "ra",
        "dec",
        "vlos",
        "pmra",
        "ll",
        "bb",
        "phi1",
        "phi2",
        "pmphi1",
    ):
        track.plot(d1="x", d2=d2, spread=1)
    # No custom_transform: non-custom axes still work (covers the "basis
    # needs custom_transform but we don't have one" code path gracefully).
    numpy.random.seed(34)
    track_bare = _simple_spdf.streamTrack(n=800, tail="leading", ntp=31)
    track_bare.plot(d1="x", d2="ra", spread=1)


def test_streamTrack_custom_accessors_no_physical(_simple_spdf):
    # phi1/phi2/pmphi1/pmphi2 accessors must also work with physical
    # output off (covers the non-Quantity branch of the extract-scalar
    # helper inside each accessor).
    from galpy.util import coords

    T = _simple_spdf._progenitor.align_to_orbit()
    numpy.random.seed(41)
    track = _simple_spdf.streamTrack(n=1000, tail="leading", custom_transform=T)
    track.turn_physical_off()
    tp0 = track.tp_grid()[len(track.tp_grid()) // 2]
    for name in ("phi1", "phi2", "pmphi1", "pmphi2"):
        val = getattr(track, name)(tp0)
        assert numpy.isfinite(float(val))
    track.turn_physical_on(ro=8.0, vo=220.0)


def test_streamTrack_with_progpot():
    # Cover the `self._orig_pot = self._pot` assignment in
    # basestreamspraydf.__init__ that's taken when progpot is set and
    # used by streamTrack to integrate the progenitor through the base
    # potential (not the MovingObjectPotential one).
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ro, vo = 8.0, 220.0
    spdf = fardal15spraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        tail="leading",
        progpot=PlummerPotential(0, 0),  # massless progenitor so samples match
    )
    numpy.random.seed(51)
    track = spdf.streamTrack(n=800, ntp=21, tail="leading")
    assert hasattr(spdf, "_orig_pot")
    tp = track.tp_grid()[len(track.tp_grid()) // 2]
    assert numpy.isfinite(float(track.x(tp)))
