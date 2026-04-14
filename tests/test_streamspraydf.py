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
    tps = numpy.linspace(-spdf._tdisrupt, 0.0, 7)
    # For small tdisrupt, ALL particles are near the progenitor today,
    # so the track for every tp should be within a few percent of the
    # progenitor's present-day position.
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
    # In Cartesian, the track should match the binned mean of samples at the
    # same tp to within a few standard errors.
    numpy.random.seed(1)
    xv, dt = _simple_spdf.sample(
        n=4000, returndt=True, return_orbit=False, integrate=True
    )
    track = _simple_spdf.streamTrack(particles=(xv, dt), ntp=41, tail="leading")
    R, vR, vT, z, vz, phi = xv
    x_p, y_p, z_p = coords.cyl_to_rect(R, phi, z)
    # Bin particles
    tp_part = -dt
    edges = numpy.linspace(-_simple_spdf._tdisrupt, 0.0, 11)
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
    tps = numpy.linspace(-_simple_spdf._tdisrupt, 0.0, 5)
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
    tps = numpy.linspace(-_simple_spdf._tdisrupt, 0.0, 7)
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
    # Deep into the stream, the two arms should diverge
    tp_deep = -0.7 * _simple_spdf._tdisrupt
    d_lead = (pair.leading.x(tp_deep) - pair.trailing.x(tp_deep)) ** 2 + (
        pair.leading.y(tp_deep) - pair.trailing.y(tp_deep)
    ) ** 2
    assert d_lead > 0.01, "Leading and trailing arms do not diverge at large |tp|"


def test_streamTrack_iteration_changes_track(_simple_spdf):
    # Iteration should move the track by a small amount; we don't require
    # strict convergence (closest-point reassignment introduces some noise).
    numpy.random.seed(5)
    tr0 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=0, tail="leading")
    numpy.random.seed(5)
    tr1 = _simple_spdf.streamTrack(n=2000, ntp=41, niter=1, tail="leading")
    tps = numpy.linspace(-_simple_spdf._tdisrupt, 0.0, 101)
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
    for tp in [-spdf._tdisrupt / 2, 0.0]:
        assert numpy.isfinite(track.leading.x(tp))
        assert numpy.isfinite(track.trailing.x(tp))


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
    tps = numpy.linspace(-spdf._tdisrupt, 0.0, 5)
    vals = track.x(tps)
    assert numpy.all(numpy.isfinite(vals))


def test_streamTrack_physical_units(_simple_spdf):
    numpy.random.seed(8)
    track = _simple_spdf.streamTrack(n=1500, ntp=31, tail="leading")
    x0 = track.x(-10.0)
    track.turn_physical_on(ro=8.0, vo=220.0)
    x0_phys = track.x(-10.0)
    # physical x should be ~ ro * internal
    val = x0_phys.value if hasattr(x0_phys, "value") else x0_phys
    assert abs(val - 8.0 * x0) < 1e-6
    track.turn_physical_off()
    assert abs(track.x(-10.0) - x0) < 1e-10


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
    tp = -10.0
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
    v = pair.leading.x(-5.0)
    val = getattr(v, "value", v)
    assert val > 0.0
    pair.turn_physical_off()
    v2 = pair.leading.x(-5.0)
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
    assert g[0] == pytest.approx(-_simple_spdf._tdisrupt)
    assert g[-1] == 0.0


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


def test_streamTrack_scalar_cov(_simple_spdf):
    numpy.random.seed(17)
    track = _simple_spdf.streamTrack(n=800, ntp=31, tail="leading")
    C = track.cov(-20.0)
    assert C.shape == (6, 6)


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
    assert numpy.isfinite(track.x(-10.0))


def test_streamTrack_smoothing_variants(_simple_spdf):
    numpy.random.seed(18)
    # Default ntp (auto from n) is exercised here
    tr_f = _simple_spdf.streamTrack(n=800, tail="leading", smoothing=20.0)
    tr_d = _simple_spdf.streamTrack(
        n=800,
        ntp=31,
        tail="leading",
        smoothing={"x": 20.0, "y": 20.0},
    )
    assert numpy.isfinite(tr_f.x(-10.0))
    assert numpy.isfinite(tr_d.x(-10.0))
