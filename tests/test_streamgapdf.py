# Tests of streamgapdf implementation, impulse tests moved to
# test_streamgapdf_impulse.py
import numpy
import pytest
from scipy import integrate

sdf_sanders15 = None  # so we can set this up and then use in other tests
sdf_sanders15_unp = None  # so we can set this up and then use in other tests
sdfl_sanders15 = None  # so we can set this up and then use in other tests
sdfl_sanders15_unp = None  # so we can set this up and then use in other tests


@pytest.fixture(scope="module")
def setup_sanders15_trailing():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf, streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)  # km/s
    sdf_sanders15 = streamgapdf(
        sigv / V0,
        progenitor=prog_unp_peri,
        pot=lp,
        aA=aAI,
        leading=False,
        nTrackChunks=26,
        nTrackIterations=1,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        Vnorm=V0,
        Rnorm=R0,
        impactb=0.0,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
        timpact=0.88 / conversion.time_in_Gyr(V0, R0),
        impact_angle=-2.34,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
        rs=0.625 / R0,
    )
    # Also setup the unperturbed model
    sdf_sanders15_unp = streamdf(
        sigv / V0,
        progenitor=prog_unp_peri,
        pot=lp,
        aA=aAI,
        leading=False,
        nTrackChunks=26,
        nTrackIterations=1,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        Vnorm=V0,
        Rnorm=R0,
    )
    return sdf_sanders15, sdf_sanders15_unp


@pytest.fixture(scope="module")
def setup_sanders15_leading():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf, streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, PlummerPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)  # km/s
    # Use a Potential object for the impact
    pp = PlummerPotential(
        amp=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0), b=0.625 / R0
    )
    import warnings

    from galpy.util import galpyWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdfl_sanders15 = streamgapdf(
            sigv / V0,
            progenitor=prog_unp_peri,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=26,
            nTrackChunksImpact=29,
            nTrackIterations=1,
            sigMeanOffset=4.5,
            tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
            Vnorm=V0,
            Rnorm=R0,
            impactb=0.0,
            subhalovel=numpy.array([49.447319, 116.179436, 155.104156]) / V0,
            timpact=0.88 / conversion.time_in_Gyr(V0, R0),
            impact_angle=2.09,
            subhalopot=pp,
            nKickPoints=290,
            deltaAngleTrackImpact=4.5,
            multi=True,
        )  # test multi
        # Should raise warning bc of deltaAngleTrackImpact, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "WARNING: deltaAngleTrackImpact angle range large compared to plausible value"
            )
            if raisedWarning:
                break
        assert (
            raisedWarning
        ), "deltaAngleTrackImpact warning not raised when it should have been"
    # Also setup the unperturbed model
    sdfl_sanders15_unp = streamdf(
        sigv / V0,
        progenitor=prog_unp_peri,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=26,
        nTrackIterations=1,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        Vnorm=V0,
        Rnorm=R0,
    )
    return sdfl_sanders15, sdfl_sanders15_unp


# Put seed in first function, so the seed gets set even if other test files
# were run first
def test_setupimpact_error():
    numpy.random.seed(1)
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)  # km/s
    with pytest.raises(IOError) as excinfo:
        dumb = streamgapdf(
            sigv / V0,
            progenitor=prog_unp_peri,
            pot=lp,
            aA=aAI,
            leading=False,
            nTrackChunks=26,
            nTrackIterations=1,
            sigMeanOffset=4.5,
            tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
            Vnorm=V0,
            Rnorm=R0,
            impactb=0.0,
            subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
            timpact=0.88 / conversion.time_in_Gyr(V0, R0),
            impact_angle=-2.34,
        )
    # Should be including these:
    #                 GM=10.**-2.\
    #                     /conversion.mass_in_1010msol(V0,R0),
    #                 rs=0.625/R0)
    return None


def test_leadingwtrailingimpact_error():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)  # km/s
    with pytest.raises(ValueError) as excinfo:
        dumb = streamgapdf(
            sigv / V0,
            progenitor=prog_unp_peri,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=26,
            nTrackIterations=1,
            sigMeanOffset=4.5,
            tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
            Vnorm=V0,
            Rnorm=R0,
            impactb=0.0,
            subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
            timpact=0.88 / conversion.time_in_Gyr(V0, R0),
            impact_angle=-2.34,
            GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
            rs=0.625 / R0,
        )
    return None


def test_trailingwleadingimpact_error():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)  # km/s
    with pytest.raises(ValueError) as excinfo:
        dumb = streamgapdf(
            sigv / V0,
            progenitor=prog_unp_peri,
            pot=lp,
            aA=aAI,
            leading=False,
            nTrackChunks=26,
            nTrackIterations=1,
            sigMeanOffset=4.5,
            tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
            Vnorm=V0,
            Rnorm=R0,
            impactb=0.0,
            subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
            timpact=0.88 / conversion.time_in_Gyr(V0, R0),
            impact_angle=2.34,
            GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
            rs=0.625 / R0,
        )
    return None


# Exact setup from Section 5 of Sanders, Bovy, and Erkal (2015); should reproduce those results (which have been checked against a simulation)
def test_sanders15_setup(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    assert not sdf_sanders15 is None, "sanders15 streamgapdf setup did not work"
    assert (
        not sdf_sanders15_unp is None
    ), "sanders15 unperturbed streamdf setup did not work"
    return None


def test_sanders15_leading_setup(setup_sanders15_leading):
    # Load the streamgapdf objects
    sdfl_sanders15, sdfl_sanders15_unp = setup_sanders15_leading
    assert not sdfl_sanders15 is None, "sanders15 trailing streamdf setup did not work"
    assert (
        not sdfl_sanders15_unp is None
    ), "sanders15 unperturbed streamdf setup did not work"
    return None


# Some very basic tests
def test_nTrackIterations(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    assert sdf_sanders15.nTrackIterations == 1, "nTrackIterations should have been 1"
    return None


def test_nTrackChunks(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    assert sdf_sanders15._nTrackChunks == 26, "nTrackChunks should have been 26"
    return None


def test_deltaAngleTrackImpact(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    assert (
        numpy.fabs(sdf_sanders15._deltaAngleTrackImpact - 4.31) < 0.01
    ), "deltaAngleTrackImpact should have been ~4.31"
    return None


# Tests of the track near the impact
def test_trackNearImpact(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Sanity checks against numbers taken from plots of the simulation
    # Make sure we're near 14.5
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[14, 0] * sdf_sanders15._ro - 14.5) < 0.2
    ), "14th point along track near the impact is not near 14.5 kpc"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[14, 1] * sdf_sanders15._vo - 80) < 3.0
    ), "Point along the track near impact near R=14.5 does not have the correct radial velocity"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[14, 2] * sdf_sanders15._vo - 220.0) < 3.0
    ), "Point along the track near impact near R=14.5 does not have the correct tangential velocity"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[14, 3] * sdf_sanders15._ro - 0.0) < 1.0
    ), "Point along the track near impact near R=14.5 does not have the correct vertical height"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[14, 4] * sdf_sanders15._vo - 200.0) < 5.0
    ), "Point along the track near impact near R=14.5 does not have the correct vertical velocity"
    # Another one!
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 0] * sdf_sanders15._ro - 16.25) < 0.2
    ), "27th point along track near the impact is not near 16.25 kpc"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 1] * sdf_sanders15._vo + 130) < 3.0
    ), "Point along the track near impact near R=16.25 does not have the correct radial velocity"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 2] * sdf_sanders15._vo - 200.0) < 3.0
    ), "Point along the track near impact near R=16.25 does not have the correct tangential velocity"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 3] * sdf_sanders15._ro + 12.0) < 1.0
    ), "Point along the track near impact near R=16.25 does not have the correct vertical height"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 4] * sdf_sanders15._vo - 25.0) < 5.0
    ), "Point along the track near impact near R=16.25 does not have the correct vertical velocity"
    assert (
        numpy.fabs(sdf_sanders15._gap_ObsTrack[27, 5] - 1.2) < 0.2
    ), "Point along the track near impact near R=16.25 does not have the correct azimuth"
    return None


def test_interpolatedTrackNearImpact(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Sanity checks against numbers taken from plots of the simulation
    # Make sure we're near X=-10.9
    theta = 2.7
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackX(theta) * sdf_sanders15._ro + 10.9)
        < 0.2
    ), "Point along track near the impact at theta=2.7 is not near X=-10.9 kpc"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackY(theta) * sdf_sanders15._ro - 6.0)
        < 0.5
    ), "Point along track near the impact at theta=2.7 is not near Y=6. kpc"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackZ(theta) * sdf_sanders15._ro + 5.0)
        < 0.5
    ), "Point along track near the impact at theta=2.7 is not near Z=5. kpc"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackvX(theta) * sdf_sanders15._vo + 180.0)
        < 5
    ), "Point along track near the impact at theta=2.7 is not near vX=-180 km/s"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackvY(theta) * sdf_sanders15._vo + 190.0)
        < 5.0
    ), "Point along track near the impact at theta=2.7 is not near vY=190 km/s"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpTrackvZ(theta) * sdf_sanders15._vo - 170.0)
        < 5.0
    ), "Point along track near the impact at theta=2.7 is not near vZ=170 km/s"
    return None


# Test the calculation of the kicks in dv
def test_kickdv(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Closest one to the impact point, should be close to zero
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack - sdf_sanders15._impact_angle
        )
    )
    assert numpy.all(
        numpy.fabs(sdf_sanders15._kick_deltav[tIndx] * sdf_sanders15._vo) < 0.3
    ), "Kick near the impact point not close to zero"
    # The peak, size and location
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:, 0] * sdf_sanders15._vo))
            - 0.35
        )
        < 0.06
    ), "Peak dvx incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_deltav[:, 0] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        < 0.0
    ), "Location of peak dvx incorrect"
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:, 1] * sdf_sanders15._vo))
            - 0.35
        )
        < 0.06
    ), "Peak dvy incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_deltav[:, 1] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        > 0.0
    ), "Location of peak dvy incorrect"
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(sdf_sanders15._kick_deltav[:, 2] * sdf_sanders15._vo))
            - 1.8
        )
        < 0.06
    ), "Peak dvz incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_deltav[:, 2] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        > 0.0
    ), "Location of peak dvz incorrect"
    # Close to zero far from impact point
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack
            - sdf_sanders15._impact_angle
            - 1.5
        )
    )
    assert numpy.all(
        numpy.fabs(sdf_sanders15._kick_deltav[tIndx] * sdf_sanders15._vo) < 0.3
    ), "Kick far the impact point not close to zero"
    return None


# Test the calculation of the kicks in dO
def test_kickdO(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    from galpy.util import conversion

    # Closest one to the impact point, should be close to zero
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack - sdf_sanders15._impact_angle
        )
    )
    assert numpy.all(
        numpy.fabs(
            sdf_sanders15._kick_dOap[tIndx, :3]
            * conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
        )
        < 0.03
    ), "Kick near the impact point not close to zero"
    # The peak, size and location
    assert (
        numpy.fabs(
            numpy.amax(
                numpy.fabs(
                    sdf_sanders15._kick_dOap[:, 0]
                    * conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
                )
            )
            - 0.085
        )
        < 0.01
    ), "Peak dOR incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_dOap[:, 0])
        ]
        - sdf_sanders15._impact_angle
        < 0.0
    ), "Location of peak dOR incorrect"
    assert (
        numpy.fabs(
            numpy.amax(
                numpy.fabs(
                    sdf_sanders15._kick_dOap[:, 1]
                    * conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
                )
            )
            - 0.07
        )
        < 0.01
    ), "Peak dOp incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_dOap[:, 1])
        ]
        - sdf_sanders15._impact_angle
        < 0.0
    ), "Location of peak dvy incorrect"
    assert (
        numpy.fabs(
            numpy.amax(
                numpy.fabs(
                    sdf_sanders15._kick_dOap[:, 2]
                    * conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
                )
            )
            - 0.075
        )
        < 0.01
    ), "Peak dOz incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(sdf_sanders15._kick_dOap[:, 2])
        ]
        - sdf_sanders15._impact_angle
        < 0.0
    ), "Location of peak dOz incorrect"
    # Close to zero far from impact point
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack
            - sdf_sanders15._impact_angle
            - 1.5
        )
    )
    assert numpy.all(
        numpy.fabs(
            sdf_sanders15._kick_dOap[tIndx, :3]
            * conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
        )
        < 0.03
    ), "Kick far the impact point not close to zero"
    return None


def test_kickda(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # All angle kicks should be small, just test that they are smaller than dO/O close to the impact
    nIndx = (
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack - sdf_sanders15._impact_angle
        )
        < 0.75
    )
    assert numpy.all(
        numpy.fabs(sdf_sanders15._kick_dOap[nIndx, 3:])
        < 2.0
        * (
            numpy.fabs(
                sdf_sanders15._kick_dOap[nIndx, :3] / sdf_sanders15._progenitor_Omega
            )
        )
    ), "angle kicks not smaller than the frequency kicks"
    return None


# Test the interpolation of the kicks
def test_interpKickdO(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    from galpy.util import conversion

    freqConv = conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
    # Bunch of spot checks at some interesting angles
    # Impact angle
    theta = sdf_sanders15._impact_angle
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOpar(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp0(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp1(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOr(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOp(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOz(theta) * freqConv) < 10.0**-4.0
    ), "Frequency kick at the impact point is not zero"
    # random one
    theta = sdf_sanders15._impact_angle - 0.25
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOpar(theta) * freqConv + 0.07) < 0.002
    ), "Frequency kick near the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp0(theta) * freqConv) < 0.002
    ), "Frequency kick near the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp1(theta) * freqConv) < 0.003
    ), "Frequency kick near the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOr(theta) * freqConv - 0.05) < 0.01
    ), "Frequency kick near the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOp(theta) * freqConv - 0.035) < 0.01
    ), "Frequency kick near the impact point is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOz(theta) * freqConv - 0.04) < 0.01
    ), "Frequency kick near the impact point is not zero"
    # One beyond ._deltaAngleTrackImpact
    theta = sdf_sanders15._deltaAngleTrackImpact + 0.1
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOpar(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp0(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOperp1(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOr(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOp(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdOz(theta) * freqConv) < 10.0**-16.0
    ), "Frequency kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdar(theta)) < 10.0**-16.0
    ), "Angle kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdap(theta)) < 10.0**-16.0
    ), "Angle kick beyond ._deltaAngleTrackImpact is not zero"
    assert (
        numpy.fabs(sdf_sanders15._kick_interpdaz(theta)) < 10.0**-16.0
    ), "Angle kick beyond ._deltaAngleTrackImpact is not zero"
    return None


def test_interpKickda(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    thetas = numpy.linspace(-0.75, 0.75, 10) + sdf_sanders15._impact_angle
    assert numpy.all(
        numpy.fabs(sdf_sanders15._kick_interpdar(thetas))
        < 2.0
        * numpy.fabs(
            sdf_sanders15._kick_interpdOr(thetas) / sdf_sanders15._progenitor_Omegar
        )
    ), "Interpolated angle kick not everywhere smaller than the frequency kick after one period"
    return None


# Test the sampling of present-day perturbed points based on the model
def test_sample(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Sample stars from the model and compare them to the stream
    xv_mock_per = sdf_sanders15.sample(n=100000, xy=True).T
    # Rough gap-density check
    ingap = numpy.sum(
        (xv_mock_per[:, 0] * sdf_sanders15._ro > 4.0)
        * (xv_mock_per[:, 0] * sdf_sanders15._ro < 5.0)
    )
    edgegap = numpy.sum(
        (xv_mock_per[:, 0] * sdf_sanders15._ro > 1.0)
        * (xv_mock_per[:, 0] * sdf_sanders15._ro < 2.0)
    )
    outgap = numpy.sum(
        (xv_mock_per[:, 0] * sdf_sanders15._ro > -2.5)
        * (xv_mock_per[:, 0] * sdf_sanders15._ro < -1.5)
    )
    assert (
        numpy.fabs(ingap / float(edgegap) - 0.015 / 0.05) < 0.05
    ), "gap density versus edge of the gap is incorrect"
    assert (
        numpy.fabs(ingap / float(outgap) - 0.015 / 0.02) < 0.2
    ), "gap density versus outside of the gap is incorrect"
    # Test track of the stream
    tIndx = (
        (xv_mock_per[:, 0] * sdf_sanders15._ro > 4.0)
        * (xv_mock_per[:, 0] * sdf_sanders15._ro < 5.0)
        * (xv_mock_per[:, 1] * sdf_sanders15._ro < 5.0)
    )
    assert (
        numpy.fabs(numpy.median(xv_mock_per[tIndx, 1]) * sdf_sanders15._ro + 12.25)
        < 0.1
    ), "Location of mock track is incorrect near the gap"
    assert (
        numpy.fabs(numpy.median(xv_mock_per[tIndx, 2]) * sdf_sanders15._ro - 3.8) < 0.1
    ), "Location of mock track is incorrect near the gap"
    assert (
        numpy.fabs(numpy.median(xv_mock_per[tIndx, 3]) * sdf_sanders15._vo - 255.0)
        < 2.0
    ), "Location of mock track is incorrect near the gap"
    assert (
        numpy.fabs(numpy.median(xv_mock_per[tIndx, 4]) * sdf_sanders15._vo - 20.0) < 2.0
    ), "Location of mock track is incorrect near the gap"
    assert (
        numpy.fabs(numpy.median(xv_mock_per[tIndx, 5]) * sdf_sanders15._vo + 185.0)
        < 2.0
    ), "Location of mock track is incorrect near the gap"
    return None


# Test the sampling of present-day perturbed-unperturbed points
# (like in the paper)
def test_sample_offset(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Sample stars from the model and compare them to the stream
    numpy.random.seed(1)
    xv_mock_per = sdf_sanders15.sample(n=100000, xy=True).T
    numpy.random.seed(1)  # should give same points
    xv_mock_unp = sdf_sanders15_unp.sample(n=100000, xy=True).T
    # Test perturbation as a function of unperturbed X
    tIndx = (
        (xv_mock_unp[:, 0] * sdf_sanders15._ro > 0.0)
        * (xv_mock_unp[:, 0] * sdf_sanders15._ro < 1.0)
        * (xv_mock_unp[:, 1] * sdf_sanders15._ro < 5.0)
    )
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 0] - xv_mock_unp[tIndx, 0])
            * sdf_sanders15._ro
            + 0.65
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 1] - xv_mock_unp[tIndx, 1])
            * sdf_sanders15._ro
            - 0.1
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 2] - xv_mock_unp[tIndx, 2])
            * sdf_sanders15._ro
            - 0.4
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 3] - xv_mock_unp[tIndx, 3])
            * sdf_sanders15._vo
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 4] - xv_mock_unp[tIndx, 4])
            * sdf_sanders15._vo
            + 7.0
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 5] - xv_mock_unp[tIndx, 5])
            * sdf_sanders15._vo
            - 4.0
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    return None


# Test the sampling of present-day perturbed-unperturbed points
# (like in the paper, but for the leading stream impact)
def test_sample_offset_leading(setup_sanders15_leading):
    # Load the streamgapdf objects
    sdfl_sanders15, sdfl_sanders15_unp = setup_sanders15_leading
    # Sample stars from the model and compare them to the stream
    numpy.random.seed(1)
    xv_mock_per = sdfl_sanders15.sample(n=100000, xy=True).T
    numpy.random.seed(1)  # should give same points
    xv_mock_unp = sdfl_sanders15_unp.sample(n=100000, xy=True).T
    # Test perturbation as a function of unperturbed X
    tIndx = (
        (xv_mock_unp[:, 0] * sdfl_sanders15._ro > 13.0)
        * (xv_mock_unp[:, 0] * sdfl_sanders15._ro < 14.0)
        * (xv_mock_unp[:, 1] * sdfl_sanders15._ro > 5.0)
    )
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 0] - xv_mock_unp[tIndx, 0])
            * sdfl_sanders15._ro
            + 0.5
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 1] - xv_mock_unp[tIndx, 1])
            * sdfl_sanders15._ro
            - 0.3
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 2] - xv_mock_unp[tIndx, 2])
            * sdfl_sanders15._ro
            - 0.45
        )
        < 0.1
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 3] - xv_mock_unp[tIndx, 3])
            * sdfl_sanders15._vo
            + 2.0
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 4] - xv_mock_unp[tIndx, 4])
            * sdfl_sanders15._vo
            + 7.0
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    assert (
        numpy.fabs(
            numpy.median(xv_mock_per[tIndx, 5] - xv_mock_unp[tIndx, 5])
            * sdfl_sanders15._vo
            - 6.0
        )
        < 0.5
    ), "Location of perturbed mock track is incorrect near the gap"
    return None


# Tests of the density and meanOmega functions


def test_pOparapar(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that integrating pOparapar gives density_par
    dens_frompOpar_close = integrate.quad(
        lambda x: sdf_sanders15.pOparapar(x, 0.3),
        sdf_sanders15._meandO - 10.0 * numpy.sqrt(sdf_sanders15._sortedSigOEig[2]),
        sdf_sanders15._meandO + 10.0 * numpy.sqrt(sdf_sanders15._sortedSigOEig[2]),
    )[0]
    # This is actually in the gap!
    dens_fromOpar_half = integrate.quad(
        lambda x: sdf_sanders15.pOparapar(x, 2.6),
        sdf_sanders15._meandO - 10.0 * numpy.sqrt(sdf_sanders15._sortedSigOEig[2]),
        sdf_sanders15._meandO + 10.0 * numpy.sqrt(sdf_sanders15._sortedSigOEig[2]),
    )[0]
    assert (
        numpy.fabs(
            dens_fromOpar_half / dens_frompOpar_close
            - sdf_sanders15.density_par(2.6) / sdf_sanders15.density_par(0.3)
        )
        < 10.0**-4.0
    ), "density from integrating pOparapar not equal to that from density_par for Sanders15 stream"
    return None


def test_density_apar_approx(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that the approximate density agrees with the direct integration
    # Need to do this relatively to another density, because there is an
    # overall offset
    apar = 2.6
    assert (
        numpy.fabs(
            sdf_sanders15.density_par(apar, approx=False)
            / sdf_sanders15.density_par(apar, approx=True)
            / sdf_sanders15.density_par(0.3, approx=False)
            * sdf_sanders15.density_par(0.3, approx=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate density does not agree with direct integration"
    apar = 2.3
    assert (
        numpy.fabs(
            sdf_sanders15.density_par(apar, approx=False)
            / sdf_sanders15.density_par(apar, approx=True)
            / sdf_sanders15.density_par(0.3, approx=False)
            * sdf_sanders15.density_par(0.3, approx=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate density does not agree with direct integration"
    return None


def test_density_apar_approx_higherorder(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that the approximate density agrees with the direct integration
    # Need to do this relatively to another density, because there is an
    # overall offset
    apar = 2.6
    assert (
        numpy.fabs(
            sdf_sanders15.density_par(apar, approx=False)
            / sdf_sanders15.density_par(apar, approx=True, higherorder=True)
            / sdf_sanders15.density_par(0.3, approx=False)
            * sdf_sanders15.density_par(0.3, approx=True, higherorder=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate density does not agree with direct integration"
    apar = 2.3
    assert (
        numpy.fabs(
            sdf_sanders15.density_par(apar, approx=False)
            / sdf_sanders15.density_par(apar, approx=True, higherorder=True)
            / sdf_sanders15.density_par(0.3, approx=False)
            * sdf_sanders15.density_par(0.3, approx=True, higherorder=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate density does not agree with direct integration"
    return None


def test_minOpar(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that for Opar < minOpar, p(Opar,apar) is in fact zero!
    apar = 0.3
    dO = 10.0**-4.0
    assert (
        numpy.fabs(sdf_sanders15.pOparapar(sdf_sanders15.minOpar(apar) - dO, apar))
        < 10.0**-16.0
    ), "Probability for Opar < minOpar is not zero"
    apar = 2.6
    dO = 10.0**-4.0
    assert (
        numpy.fabs(sdf_sanders15.pOparapar(sdf_sanders15.minOpar(apar) - dO, apar))
        < 10.0**-16.0
    ), "Probability for Opar < minOpar is not zero"
    return None


def test_meanOmega_approx(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that the approximate meanOmega agrees with the direct integration
    # Need to do this relatively to another density, because there is an
    # overall offset
    apar = 2.6
    assert (
        numpy.fabs(
            sdf_sanders15.meanOmega(apar, approx=False, oned=True)
            / sdf_sanders15.meanOmega(apar, approx=True, oned=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate meanOmega does not agree with direct integration"
    apar = 2.3
    assert (
        numpy.fabs(
            sdf_sanders15.meanOmega(apar, approx=False, oned=True)
            / sdf_sanders15.meanOmega(apar, approx=True, oned=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate meanOmega does not agree with direct integration"
    return None


def test_meanOmega_approx_higherorder(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that the approximate meanOmega agrees with the direct integration
    # Need to do this relatively to another density, because there is an
    # overall offset
    apar = 2.6
    assert (
        numpy.fabs(
            sdf_sanders15.meanOmega(apar, approx=False, oned=True)
            / sdf_sanders15.meanOmega(apar, approx=True, higherorder=True, oned=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate meanOmega does not agree with direct integration"
    apar = 2.3
    assert (
        numpy.fabs(
            sdf_sanders15.meanOmega(apar, approx=False, oned=True)
            / sdf_sanders15.meanOmega(apar, approx=True, higherorder=True, oned=True)
            - 1.0
        )
        < 10.0**-3.0
    ), "Approximate meanOmega does not agree with direct integration"
    return None


def test_hernquist(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that Hernquist kicks are similar to Plummer kicks, but are
    # different in understood ways (...)
    from galpy.util import conversion

    # Switch to Hernquist
    V0, R0 = 220.0, 8.0
    impactb = 0.0
    subhalovel = numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0
    impact_angle = -2.34
    GM = 10.0**-2.0 / conversion.mass_in_1010msol(V0, R0)
    rs = 0.625 / R0
    sdf_sanders15._determine_deltav_kick(
        impact_angle, impactb, subhalovel, GM, rs, None, 3, True
    )
    hernquist_kicks = sdf_sanders15._kick_deltav
    # Back to Plummer
    sdf_sanders15._determine_deltav_kick(
        impact_angle, impactb, subhalovel, GM, rs, None, 3, False
    )
    # Repeat some of the deltav tests from above
    # Closest one to the impact point, should be close to zero
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack - sdf_sanders15._impact_angle
        )
    )
    assert numpy.all(
        numpy.fabs(hernquist_kicks[tIndx] * sdf_sanders15._vo) < 0.4
    ), "Kick near the impact point not close to zero for Hernquist"
    # The peak, size and location
    # Peak should be slightly less (guessed these correct!)
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(hernquist_kicks[:, 0] * sdf_sanders15._vo)) - 0.25
        )
        < 0.06
    ), "Peak dvx incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(hernquist_kicks[:, 0] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        < 0.0
    ), "Location of peak dvx incorrect"
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(hernquist_kicks[:, 1] * sdf_sanders15._vo)) - 0.25
        )
        < 0.06
    ), "Peak dvy incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(hernquist_kicks[:, 1] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        > 0.0
    ), "Location of peak dvy incorrect"
    assert (
        numpy.fabs(
            numpy.amax(numpy.fabs(hernquist_kicks[:, 2] * sdf_sanders15._vo)) - 1.3
        )
        < 0.06
    ), "Peak dvz incorrect"
    assert (
        sdf_sanders15._kick_interpolatedThetasTrack[
            numpy.argmax(hernquist_kicks[:, 2] * sdf_sanders15._vo)
        ]
        - sdf_sanders15._impact_angle
        > 0.0
    ), "Location of peak dvz incorrect"
    # Close to zero far from impact point
    tIndx = numpy.argmin(
        numpy.fabs(
            sdf_sanders15._kick_interpolatedThetasTrack
            - sdf_sanders15._impact_angle
            - 1.5
        )
    )
    assert numpy.all(
        numpy.fabs(hernquist_kicks[tIndx] * sdf_sanders15._vo) < 0.3
    ), "Kick far the impact point not close to zero"
    return None


def test_determine_deltav_valueerrort(setup_sanders15_trailing):
    # Load the streamgapdf objects
    sdf_sanders15, sdf_sanders15_unp = setup_sanders15_trailing
    # Test that modeling leading (trailing) impact for trailing (leading) arm
    # raises a ValueError when using _determine_deltav_kick
    from galpy.util import conversion

    # Switch to Hernquist
    V0, R0 = 220.0, 8.0
    impactb = 0.0
    subhalovel = numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0
    impact_angle = -2.34
    GM = 10.0**-2.0 / conversion.mass_in_1010msol(V0, R0)
    rs = 0.625 / R0
    # Can't do minus impact angle!
    with pytest.raises(ValueError) as excinfo:
        sdf_sanders15._determine_deltav_kick(
            -impact_angle, impactb, subhalovel, GM, rs, None, 3, True
        )
    return None


# Test the routine that rotates vectors to an arbitrary vector
def test_rotate_to_arbitrary_vector():
    from galpy.df.streamgapdf import _rotate_to_arbitrary_vector

    tol = -10.0
    v = numpy.array([[1.0, 0.0, 0.0]])
    # Rotate to 90 deg off
    ma = _rotate_to_arbitrary_vector(v, [0, 1.0, 0])
    assert (
        numpy.fabs(ma[0, 0, 1] + 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 0] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 2] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    # Rotate to 90 deg off
    ma = _rotate_to_arbitrary_vector(v, [0, 0, 1.0])
    assert (
        numpy.fabs(ma[0, 0, 2] + 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 0] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 1] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    # Rotate to same should be unit matrix
    ma = _rotate_to_arbitrary_vector(v, v[0])
    assert numpy.all(
        numpy.fabs(numpy.diag(ma[0]) - 1.0) < 10.0**tol
    ), "Rotation matrix to same vector is not unity"
    assert (
        numpy.fabs(numpy.sum(ma**2.0) - 3.0) < 10.0**tol
    ), "Rotation matrix to same vector is not unity"
    # Rotate to -same should be -unit matrix
    ma = _rotate_to_arbitrary_vector(v, -v[0])
    assert numpy.all(
        numpy.fabs(numpy.diag(ma[0]) + 1.0) < 10.0**tol
    ), "Rotation matrix to minus same vector is not minus unity"
    assert (
        numpy.fabs(numpy.sum(ma**2.0) - 3.0) < 10.0**tol
    ), "Rotation matrix to minus same vector is not minus unity"
    return None


# Test that the rotation routine works for multiple vectors
def test_rotate_to_arbitrary_vector_multi():
    from galpy.df.streamgapdf import _rotate_to_arbitrary_vector

    tol = -10.0
    v = numpy.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Rotate to 90 deg off
    ma = _rotate_to_arbitrary_vector(v, [0, 0, 1.0])
    assert (
        numpy.fabs(ma[0, 0, 2] + 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 0] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 1] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    # 2nd
    assert (
        numpy.fabs(ma[1, 1, 2] + 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 2, 1] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 0, 0] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 0, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 0, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 1, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 1, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 2, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[1, 2, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    return None


# Test the inverse of the routine that rotates vectors to an arbitrary vector
def test_rotate_to_arbitrary_vector_inverse():
    from galpy.df.streamgapdf import _rotate_to_arbitrary_vector

    tol = -10.0
    v = numpy.array([[1.0, 0.0, 0.0]])
    # Rotate to random vector and back
    a = numpy.random.uniform(size=3)
    a /= numpy.sqrt(numpy.sum(a**2.0))
    ma = _rotate_to_arbitrary_vector(v, a)
    ma_inv = _rotate_to_arbitrary_vector(v, a, inv=True)
    ma = numpy.dot(ma[0], ma_inv[0])
    assert numpy.all(
        numpy.fabs(ma - numpy.eye(3)) < 10.0**tol
    ), "Inverse rotation matrix incorrect"
    return None


# Test that rotating to vy in particular works as expected
def test_rotation_vy():
    from galpy.df.streamgapdf import _rotation_vy

    tol = -10.0
    v = numpy.array([[1.0, 0.0, 0.0]])
    # Rotate to 90 deg off
    ma = _rotation_vy(v)
    assert (
        numpy.fabs(ma[0, 0, 1] + 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 0] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 2] - 1.0) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 0, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 1, 2]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 0]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
    assert (
        numpy.fabs(ma[0, 2, 1]) < 10.0**tol
    ), "Rotation matrix to 90 deg off incorrect"
