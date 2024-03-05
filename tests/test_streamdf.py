import platform

WIN32 = platform.system() == "Windows"
import numpy
import pytest
from scipy import integrate, interpolate

from galpy.util import coords


# Exact setup from Bovy (2014); should reproduce those results (which have been
# sanity checked
@pytest.fixture(scope="module")
def bovy14_setup():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    # For custom_transform
    theta, dec_ngp, ra_ngp = coords.get_epoch_angles(2000.0)
    T = numpy.dot(
        numpy.array(
            [
                [numpy.cos(ra_ngp), -numpy.sin(ra_ngp), 0.0],
                [numpy.sin(ra_ngp), numpy.cos(ra_ngp), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        numpy.dot(
            numpy.array(
                [
                    [-numpy.sin(dec_ngp), 0.0, numpy.cos(dec_ngp)],
                    [0.0, 1.0, 0.0],
                    [numpy.cos(dec_ngp), 0.0, numpy.sin(dec_ngp)],
                ]
            ),
            numpy.array(
                [
                    [numpy.cos(theta), numpy.sin(theta), 0.0],
                    [numpy.sin(theta), -numpy.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    ).T
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        custom_transform=T,
    )
    return sdf_bovy14


# Trailing setup
@pytest.fixture(scope="module")
def bovy14_trailing_setup():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    lp_false = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    # This is the trailing of the stream that is going the opposite direction
    obs = Orbit(
        [1.56148083, -0.35081535, 1.15481504, 0.88719443, 0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    sdft_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        multi=True,  # test multi
        leading=False,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nTrackIterations=0,
        sigangle=0.657,
    )
    return sdft_bovy14


def test_progenitor_coordtransformparams():
    # Test related to #189: test that the streamdf setup throws a warning when the given coordinate transformation parameters differ from those of the given progenitor orbit
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions
    from galpy.util import galpyWarning

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    # odeint to make sure that the C integration warning isn't thrown
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8, integrate_method="odeint")
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.5,
        vo=235.0,
        zo=0.1,
        solarmotion=[0.0, -10.0, 0.0],
    )
    sigv = 0.365  # km/s
    # Turn warnings into errors to test for them
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        # Test w/ diff Rnorm
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            nosetup=True,  # won't look at track
            Rnorm=10.0,
        )
        # Should raise warning bc of Rnorm, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Warning: progenitor's ro does not agree with streamdf's ro and R0; this may have unexpected consequences when projecting into observables"
            )
            if raisedWarning:
                break
        assert raisedWarning, "streamdf setup does not raise warning when progenitor's  ro is different from ro"
    # Test w/ diff R0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            nosetup=True,  # won't look at track
            R0=10.0,
        )
        # Should raise warning bc of R0, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Warning: progenitor's ro does not agree with streamdf's ro and R0; this may have unexpected consequences when projecting into observables"
            )
            if raisedWarning:
                break
        assert raisedWarning, "streamdf setup does not raise warning when progenitor's  ro is different from R0"
    # Test w/ diff Vnorm
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            nosetup=True,  # won't look at track
            Rnorm=8.5,
            R0=8.5,
            Vnorm=220.0,
        )
        # Should raise warning bc of Vnorm, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Warning: progenitor's vo does not agree with streamdf's vo; this may have unexpected consequences when projecting into observables"
            )
            if raisedWarning:
                break
        assert raisedWarning, "streamdf setup does not raise warning when progenitor's  vo is different from vo"
    # Test w/ diff zo
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            nosetup=True,  # won't look at track
            Rnorm=8.5,
            R0=8.5,
            Vnorm=235.0,
            Zsun=0.025,
        )
        # Should raise warning bc of zo, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Warning: progenitor's zo does not agree with streamdf's Zsun; this may have unexpected consequences when projecting into observables"
            )
            if raisedWarning:
                break
        assert raisedWarning, "streamdf setup does not raise warning when progenitor's  zo is different from Zsun"
    # Test w/ diff vsun
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            nosetup=True,  # won't look at track
            Rnorm=8.5,
            R0=8.5,
            Vnorm=235.0,
            Zsun=0.1,
            vsun=[0.0, 220.0, 0.0],
        )
        # Should raise warning bc of vsun, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Warning: progenitor's solarmotion does not agree with streamdf's vsun (after accounting for vo); this may have unexpected consequences when projecting into observables"
            )
            if raisedWarning:
                break
        assert raisedWarning, "streamdf setup does not raise warning when progenitor's  solarmotion is different from vsun"
    return None


# Exact setup from Bovy (2014); should reproduce those results (which have been
# sanity checked
def test_bovy14_setup(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    assert not sdf_bovy14 is None, "bovy14 streamdf setup did not work"
    return None


def test_bovy14_freqratio(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the frequency ratio
    assert (
        (sdf_bovy14.freqEigvalRatio() - 30.0) ** 2.0 < 10.0** 0.0
    ), "streamdf model from Bovy (2014) does not give a frequency ratio of about 30"
    assert (
        (sdf_bovy14.freqEigvalRatio(isotropic=True) - 34.0) ** 2.0 < 10.0** 0.0
    ), "streamdf model from Bovy (2014) does not give an isotropic frequency ratio of about 34"
    return None


def test_bovy14_misalignment(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the misalignment
    assert (
        (sdf_bovy14.misalignment() / numpy.pi * 180.0 + 0.5) ** 2.0 < 10.0** -2.0
    ), "streamdf model from Bovy (2014) does not give a misalighment of about -0.5 degree"
    assert (
        (sdf_bovy14.misalignment(isotropic=True) / numpy.pi * 180.0 - 1.3) ** 2.0
        < 10.0** -2.0
    ), "streamdf model from Bovy (2014) does not give an isotropic misalighment of about 1.3 degree"
    return None


def test_bovy14_track_prog_diff(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the stream and the progenitor are close together, for both leading and trailing
    check_track_prog_diff(sdf_bovy14, "R", "Z", 0.1)
    check_track_prog_diff(sdf_bovy14, "R", "Z", 0.8, phys=True)  # do 1 with phys
    check_track_prog_diff(sdf_bovy14, "R", "X", 0.1)
    check_track_prog_diff(sdf_bovy14, "R", "Y", 0.1)
    check_track_prog_diff(sdf_bovy14, "R", "vZ", 0.03)
    check_track_prog_diff(sdf_bovy14, "R", "vZ", 6.6, phys=True)  # do 1 with phys
    check_track_prog_diff(sdf_bovy14, "R", "vX", 0.05)
    check_track_prog_diff(sdf_bovy14, "R", "vY", 0.05)
    check_track_prog_diff(sdf_bovy14, "R", "vT", 0.05)
    check_track_prog_diff(sdf_bovy14, "R", "vR", 0.05)
    check_track_prog_diff(sdf_bovy14, "ll", "bb", 0.3)
    check_track_prog_diff(sdf_bovy14, "ll", "dist", 0.5)
    check_track_prog_diff(sdf_bovy14, "ll", "vlos", 4.0)
    check_track_prog_diff(sdf_bovy14, "ll", "pmll", 0.3)
    check_track_prog_diff(sdf_bovy14, "ll", "pmbb", 0.25)
    return None


def test_bovy14_track_spread(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the spreads are small
    check_track_spread(sdf_bovy14, "R", "Z", 0.01, 0.005)
    check_track_spread(sdf_bovy14, "R", "Z", 0.08, 0.04, phys=True)  # do 1 with phys
    check_track_spread(
        sdf_bovy14, "R", "Z", 0.01, 0.005, interp=False
    )  # do 1 with interp
    check_track_spread(sdf_bovy14, "X", "Y", 0.01, 0.005)
    check_track_spread(sdf_bovy14, "X", "Y", 0.08, 0.04, phys=True)  # do 1 with phys
    check_track_spread(sdf_bovy14, "R", "phi", 0.01, 0.005)
    check_track_spread(sdf_bovy14, "vR", "vT", 0.005, 0.005)
    check_track_spread(sdf_bovy14, "vR", "vT", 1.1, 1.1, phys=True)  # do 1 with phys
    check_track_spread(sdf_bovy14, "vR", "vZ", 0.005, 0.005)
    check_track_spread(sdf_bovy14, "vX", "vY", 0.005, 0.005)
    delattr(sdf_bovy14, "_allErrCovs")  # to test that this is re-generated
    check_track_spread(sdf_bovy14, "ll", "bb", 0.5, 0.5)
    check_track_spread(sdf_bovy14, "dist", "vlos", 0.5, 5.0)
    check_track_spread(sdf_bovy14, "pmll", "pmbb", 0.5, 0.5)
    # These should all exist, so return None
    assert (
        sdf_bovy14._interpolate_stream_track() is None
    ), "_interpolate_stream_track does not return None, even though it should be set up"
    assert (
        sdf_bovy14._interpolate_stream_track_aA() is None
    ), "_interpolate_stream_track_aA does not return None, even though it should be set up"
    delattr(sdf_bovy14, "_interpolatedObsTrackAA")
    delattr(sdf_bovy14, "_interpolatedThetasTrack")
    # Re-build
    assert (
        sdf_bovy14._interpolate_stream_track_aA() is None
    ), "Re-building interpolated AA track does not return None"
    return None


def test_closest_trackpoint(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Check that we can find the closest trackpoint properly
    check_closest_trackpoint(sdf_bovy14, 50)
    check_closest_trackpoint(sdf_bovy14, 230, usev=True)
    check_closest_trackpoint(sdf_bovy14, 330, usev=True, xy=False)
    check_closest_trackpoint(sdf_bovy14, 40, xy=False)
    check_closest_trackpoint(sdf_bovy14, 4, interp=False)
    check_closest_trackpoint(sdf_bovy14, 6, interp=False, usev=True, xy=False)
    return None


def test_closest_trackpointLB(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Check that we can find the closest trackpoint properly in LB
    check_closest_trackpointLB(sdf_bovy14, 50)
    check_closest_trackpointLB(sdf_bovy14, 230, usev=True)
    check_closest_trackpointLB(sdf_bovy14, 4, interp=False)
    check_closest_trackpointLB(sdf_bovy14, 8, interp=False, usev=True)
    check_closest_trackpointLB(sdf_bovy14, -1, interp=False, usev=False)
    check_closest_trackpointLB(sdf_bovy14, -2, interp=False, usev=True)
    check_closest_trackpointLB(sdf_bovy14, -3, interp=False, usev=True)
    return None


def test_closest_trackpointaA(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Check that we can find the closest trackpoint properly in AA
    check_closest_trackpointaA(sdf_bovy14, 50)
    check_closest_trackpointaA(sdf_bovy14, 4, interp=False)
    return None


def test_pOparapar(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that integrating pOparapar gives density_par
    dens_frompOpar_close = integrate.quad(
        lambda x: sdf_bovy14.pOparapar(x, 0.1),
        sdf_bovy14._meandO - 4.0 * numpy.sqrt(sdf_bovy14._sortedSigOEig[2]),
        sdf_bovy14._meandO + 4.0 * numpy.sqrt(sdf_bovy14._sortedSigOEig[2]),
    )[0]
    dens_fromOpar_half = integrate.quad(
        lambda x: sdf_bovy14.pOparapar(x, 1.1),
        sdf_bovy14._meandO - 4.0 * numpy.sqrt(sdf_bovy14._sortedSigOEig[2]),
        sdf_bovy14._meandO + 4.0 * numpy.sqrt(sdf_bovy14._sortedSigOEig[2]),
    )[0]
    assert (
        numpy.fabs(
            dens_fromOpar_half / dens_frompOpar_close - sdf_bovy14.density_par(1.1)
        )
        < 10.0**-4.0
    ), "density from integrating pOparapar not equal to that from density_par for Bovy14 stream"
    return None


def test_density_par_valueerror(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the code throws a ValueError if coord is not understood
    with pytest.raises(ValueError) as excinfo:
        sdf_bovy14.density_par(0.1, coord="xi")
    return None


def test_density_par(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the density is close to 1 close to the progenitor and close to zero far from the progenitor
    assert (
        numpy.fabs(sdf_bovy14.density_par(0.1) - 1.0) < 10.0**-2.0
    ), "density near progenitor not close to 1 for Bovy14 stream"
    assert (
        numpy.fabs(sdf_bovy14.density_par(0.5) - 1.0) < 10.0**-2.0
    ), "density near progenitor not close to 1 for Bovy14 stream"
    assert (
        numpy.fabs(sdf_bovy14.density_par(1.8) - 0.0) < 10.0**-2.0
    ), "density far progenitor not close to 0 for Bovy14 stream"
    return None


def test_density_phi(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that the density in phi is correctly computed, by doing this by hand
    def dens_phi(apar):
        dapar = 10.0**-9.0
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar),
            sdf_bovy14._interpTrackY(apar),
            sdf_bovy14._interpTrackZ(apar),
        )
        R, phi, z = coords.rect_to_cyl(X, Y, Z)
        dX, dY, dZ = (
            sdf_bovy14._interpTrackX(apar + dapar),
            sdf_bovy14._interpTrackY(apar + dapar),
            sdf_bovy14._interpTrackZ(apar + dapar),
        )
        dR, dphi, dz = coords.rect_to_cyl(dX, dY, dZ)
        jac = numpy.fabs((dphi - phi) / dapar)
        return sdf_bovy14.density_par(apar) / jac

    apar = 0.1
    assert (
        numpy.fabs(dens_phi(apar) / sdf_bovy14.density_par(apar, coord="phi") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in phi is incorrect"
    apar = 0.5
    assert (
        numpy.fabs(dens_phi(apar) / sdf_bovy14.density_par(apar, coord="phi") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in phi is incorrect"
    apar = 1.8
    assert (
        numpy.fabs(dens_phi(apar) / sdf_bovy14.density_par(apar, coord="phi") - 1.0)
        < 10.0**-2.0
    ), "density far from progenitor in phi is incorrect"
    return None


def test_density_ll_and_customra(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that the density in ll is correctly computed, by doing this by hand
    # custom should be the same for this setup (see above)
    def dens_ll(apar):
        dapar = 10.0**-9.0
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar) * sdf_bovy14._ro,
        )
        X, Y, Z = coords.galcenrect_to_XYZ(
            X, Y, Z, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        l, b, d = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        dX, dY, dZ = (
            sdf_bovy14._interpTrackX(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar + dapar) * sdf_bovy14._ro,
        )
        dX, dY, dZ = coords.galcenrect_to_XYZ(
            dX, dY, dZ, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        dl, db, dd = coords.XYZ_to_lbd(dX, dY, dZ, degree=True)
        jac = numpy.fabs((dl - l) / dapar)
        return sdf_bovy14.density_par(apar) / jac

    apar = 0.1
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="ll") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ll is incorrect"
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="customra") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ll is incorrect"
    apar = 0.5
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="ll") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ll is incorrect"
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="customra") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ll is incorrect"
    apar = 1.8
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="ll") - 1.0)
        < 10.0**-2.0
    ), "density far from progenitor in ll is incorrect"
    assert (
        numpy.fabs(dens_ll(apar) / sdf_bovy14.density_par(apar, coord="customra") - 1.0)
        < 10.0**-2.0
    ), "density far from progenitor in ll is incorrect"
    return None


def test_density_ra(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that the density in ra is correctly computed, by doing this by hand
    def dens_ra(apar):
        dapar = 10.0**-9.0
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar) * sdf_bovy14._ro,
        )
        X, Y, Z = coords.galcenrect_to_XYZ(
            X, Y, Z, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        l, b, d = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        ra, dec = coords.lb_to_radec(l, b, degree=True)
        dX, dY, dZ = (
            sdf_bovy14._interpTrackX(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar + dapar) * sdf_bovy14._ro,
        )
        dX, dY, dZ = coords.galcenrect_to_XYZ(
            dX, dY, dZ, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        dl, db, dd = coords.XYZ_to_lbd(dX, dY, dZ, degree=True)
        dra, ddec = coords.lb_to_radec(dl, db, degree=True)
        jac = numpy.fabs((dra - ra) / dapar)
        return sdf_bovy14.density_par(apar) / jac

    apar = 0.1
    assert (
        numpy.fabs(dens_ra(apar) / sdf_bovy14.density_par(apar, coord="ra") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ra is incorrect"
    apar = 0.5
    assert (
        numpy.fabs(dens_ra(apar) / sdf_bovy14.density_par(apar, coord="ra") - 1.0)
        < 10.0**-2.0
    ), "density near progenitor in ra is incorrect"
    apar = 1.8
    assert (
        numpy.fabs(dens_ra(apar) / sdf_bovy14.density_par(apar, coord="ra") - 1.0)
        < 10.0**-2.0
    ), "density far from progenitor in ra is incorrect"
    return None


def test_density_ll_wsampling(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the density computed using density_par is correct using a
    # random sample
    numpy.random.seed(1)

    def ll(apar):
        """Quick function that returns l for a given apar"""
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar) * sdf_bovy14._ro,
        )
        X, Y, Z = coords.galcenrect_to_XYZ(
            X, Y, Z, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        l, b, d = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        return l

    LB = sdf_bovy14.sample(n=10000, lb=True)
    apar1, apar2 = 0.1, 0.6
    dens1 = float(numpy.sum((LB[0] > ll(apar1)) * (LB[0] < ll(apar1) + 2.0)))
    dens2 = float(numpy.sum((LB[0] > ll(apar2)) * (LB[0] < ll(apar2) + 2.0)))
    dens1_calc = sdf_bovy14.density_par(apar1, coord="ll")
    dens2_calc = sdf_bovy14.density_par(apar2, coord="ll")
    assert (
        numpy.fabs(dens1 / dens2 - dens1_calc / dens2_calc) < 0.1
    ), "density in ll computed using density_par does not agree with density from random sample"
    return None


def test_length(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the length is correct according to its definition
    thresh = 0.2
    assert (
        numpy.fabs(
            sdf_bovy14.density_par(sdf_bovy14.length(threshold=thresh))
            / sdf_bovy14.density_par(0.1)
            - thresh
        )
        < 10.0**-3.0
    ), "Stream length does not conform to its definition"
    thresh = 0.05
    assert (
        numpy.fabs(
            sdf_bovy14.density_par(sdf_bovy14.length(threshold=thresh))
            / sdf_bovy14.density_par(0.1)
            - thresh
        )
        < 10.0**-3.0
    ), "Stream length does not conform to its definition"
    return None


def test_length_valueerror(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    thresh = 0.00001
    with pytest.raises(ValueError) as excinfo:
        assert (
            numpy.fabs(
                sdf_bovy14.density_par(sdf_bovy14.length(threshold=thresh))
                / sdf_bovy14.density_par(0.1)
                - thresh
            )
            < 10.0**-3.0
        ), "Stream length does not conform to its definition"
    return None


def test_length_ang(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that this is roughly correct
    def dphidapar(apar):
        dapar = 10.0**-9.0
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar) * sdf_bovy14._ro,
        )
        X, Y, Z = coords.galcenrect_to_XYZ(
            X, Y, Z, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        l, b, d = coords.XYZ_to_lbd(X, Y, Z, degree=True)
        dX, dY, dZ = (
            sdf_bovy14._interpTrackX(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackY(apar + dapar) * sdf_bovy14._ro,
            sdf_bovy14._interpTrackZ(apar + dapar) * sdf_bovy14._ro,
        )
        dX, dY, dZ = coords.galcenrect_to_XYZ(
            dX, dY, dZ, Xsun=sdf_bovy14._R0, Zsun=sdf_bovy14._Zsun
        )
        dl, db, dd = coords.XYZ_to_lbd(dX, dY, dZ, degree=True)
        jac = numpy.fabs((dl - l) / dapar)
        return jac

    thresh = 0.2
    assert (
        numpy.fabs(
            sdf_bovy14.length(threshold=thresh) * dphidapar(0.3)
            - sdf_bovy14.length(threshold=thresh, ang=True)
        )
        < 10.0
    ), "Length in angular coordinates does not conform to rough expectation"
    # Dangerous hack to test case where l decreases along the stream
    sdf_bovy14._interpolatedObsTrackLB[:, :2] *= -1.0
    thresh = 0.2
    assert (
        numpy.fabs(
            sdf_bovy14.length(threshold=thresh) * dphidapar(0.3)
            - sdf_bovy14.length(threshold=thresh, ang=True)
        )
        < 10.0
    ), "Length in angular coordinates does not conform to rough expectation"
    # Go back
    sdf_bovy14._interpolatedObsTrackLB[:, :2] *= -1.0
    return None


def test_length_phys(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that this is roughly correct
    def dxdapar(apar):
        dapar = 10.0**-9.0
        X, Y, Z = (
            sdf_bovy14._interpTrackX(apar),
            sdf_bovy14._interpTrackY(apar),
            sdf_bovy14._interpTrackZ(apar),
        )
        dX, dY, dZ = (
            sdf_bovy14._interpTrackX(apar + dapar),
            sdf_bovy14._interpTrackY(apar + dapar),
            sdf_bovy14._interpTrackZ(apar + dapar),
        )
        jac = numpy.sqrt(
            ((dX - X) / dapar) ** 2.0
            + ((dY - Y) / dapar) ** 2.0
            + ((dZ - Z) / dapar) ** 2.0
        )
        return jac * sdf_bovy14._ro

    thresh = 0.2
    assert (
        numpy.fabs(
            sdf_bovy14.length(threshold=thresh) * dxdapar(0.3)
            - sdf_bovy14.length(threshold=thresh, phys=True)
        )
        < 1.0
    ), "Length in physical coordinates does not conform to rough expectation"
    return None


def test_meanOmega(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that meanOmega is close to constant and the mean Omega close to the progenitor
    assert numpy.all(
        numpy.fabs(sdf_bovy14.meanOmega(0.1) - sdf_bovy14._progenitor_Omega)
        < 10.0**-2.0
    ), "meanOmega near progenitor not close to mean Omega for Bovy14 stream"
    assert numpy.all(
        numpy.fabs(sdf_bovy14.meanOmega(0.5) - sdf_bovy14._progenitor_Omega)
        < 10.0**-2.0
    ), "meanOmega near progenitor not close to mean Omega for Bovy14 stream"
    return None


def test_meanOmega_oned(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that meanOmega is close to constant and the mean Omega close to the progenitor
    assert (
        numpy.fabs(sdf_bovy14.meanOmega(0.1, oned=True)) < 10.0**-2.0
    ), "One-dimensional meanOmega near progenitor not close to zero for Bovy14 stream"
    assert (
        numpy.fabs(sdf_bovy14.meanOmega(0.5, oned=True)) < 10.0**-2.0
    ), "Oned-dimensional meanOmega near progenitor not close to zero for Bovy14 stream"
    return None


def test_sigOmega_constant(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that sigOmega is close to constant close to the progenitor
    assert (
        numpy.fabs(sdf_bovy14.sigOmega(0.1) - sdf_bovy14.sigOmega(0.5)) < 10.0**-4.0
    ), "sigOmega near progenitor not close to constant for Bovy14 stream"
    return None


def test_sigOmega_small(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that sigOmega is smaller than the total spread
    assert sdf_bovy14.sigOmega(0.1) < numpy.sqrt(
        sdf_bovy14._sortedSigOEig[2]
    ), "sigOmega near progenitor not smaller than the total Omega spread"
    assert sdf_bovy14.sigOmega(0.5) < numpy.sqrt(
        sdf_bovy14._sortedSigOEig[2]
    ), "sigOmega near progenitor not smaller than the total Omega spread"
    assert sdf_bovy14.sigOmega(1.2) < numpy.sqrt(
        sdf_bovy14._sortedSigOEig[2]
    ), "sigOmega near progenitor not smaller than the total Omega spread"
    return None


def test_meantdAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the mean td for a given angle is close to what's expected
    assert (
        numpy.fabs(
            (sdf_bovy14.meantdAngle(0.1) - 0.1 / sdf_bovy14._meandO)
            / sdf_bovy14.meantdAngle(0.1)
        )
        < 10.0**-1.5
    ), "mean td close to the progenitor is not dangle/dO"
    assert (
        numpy.fabs(
            (sdf_bovy14.meantdAngle(0.4) - 0.4 / sdf_bovy14._meandO)
            / sdf_bovy14.meantdAngle(0.1)
        )
        < 10.0**-0.9
    ), "mean td close to the progenitor is not dangle/dO"
    assert (
        numpy.fabs(sdf_bovy14.meantdAngle(0.0) - 0.0) < 10.0**-0.9
    ), "mean td at the progenitor is not 0"
    assert (
        numpy.fabs(sdf_bovy14.meantdAngle(10.0) - sdf_bovy14._tdisrupt) < 10.0**-0.9
    ), "mean td far from the progenitor is not tdisrupt"
    return None


def test_sigtdAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the sigma of td for a given angle is small
    assert (
        sdf_bovy14.sigtdAngle(0.1) < 0.2 * 0.1 / sdf_bovy14._meandO
    ), "sigma of td close to the progenitor is not small"
    assert (
        sdf_bovy14.sigtdAngle(0.5) > 0.2 * 0.1 / sdf_bovy14._meandO
    ), "sigma of td in the middle of the stream is not large"
    # Spread at the progenitor should be zero
    assert (
        sdf_bovy14.sigtdAngle(0.0) < 1e-5
    ), "sigma of td at the progenitor is not zero"
    return None


def test_ptdAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the probability distribution for p(td|angle) is reasonable
    # at 0.1
    da = 0.1
    expected_max = da / sdf_bovy14._meandO
    assert sdf_bovy14.ptdAngle(expected_max, da) > sdf_bovy14.ptdAngle(
        2.0 * expected_max, da
    ), "ptdAngle does not peak close to where it is expected to peak"
    assert sdf_bovy14.ptdAngle(expected_max, da) > sdf_bovy14.ptdAngle(
        0.5 * expected_max, da
    ), "ptdAngle does not peak close to where it is expected to peak"
    # at 0.6
    da = 0.6
    expected_max = da / sdf_bovy14._meandO
    assert sdf_bovy14.ptdAngle(expected_max, da) > sdf_bovy14.ptdAngle(
        2.0 * expected_max, da
    ), "ptdAngle does not peak close to where it is expected to peak"
    assert sdf_bovy14.ptdAngle(expected_max, da) > sdf_bovy14.ptdAngle(
        0.5 * expected_max, da
    ), "ptdAngle does not peak close to where it is expected to peak"
    # Now test that the mean and sigma calculated with a simple Riemann sum agrees with meantdAngle
    da = 0.2
    ts = numpy.linspace(0.0, 100.0, 1001)
    pts = sdf_bovy14.ptdAngle(ts, da)
    assert (
        numpy.fabs(
            (numpy.sum(ts * pts) / numpy.sum(pts) - sdf_bovy14.meantdAngle(da))
            / sdf_bovy14.meantdAngle(da)
        )
        < 10.0**-2.0
    ), "mean td at angle 0.2 calculated with Riemann sum does not agree with that calculated by meantdAngle"
    assert (
        numpy.fabs(
            (
                numpy.sqrt(
                    numpy.sum(ts**2.0 * pts) / numpy.sum(pts)
                    - (numpy.sum(ts * pts) / numpy.sum(pts)) ** 2.0
                )
                - sdf_bovy14.sigtdAngle(da)
            )
            / sdf_bovy14.sigtdAngle(da)
        )
        < 10.0**-1.5
    ), "sig td at angle 0.2 calculated with Riemann sum does not agree with that calculated by meantdAngle"
    return None


def test_meanangledAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the mean perpendicular angle at a given angle is zero
    da = 0.1
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=False)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=True)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    da = 1.1
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=False)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=True)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    da = 0.0
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=False)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    assert (
        numpy.fabs(sdf_bovy14.meanangledAngle(da, smallest=True)) < 10.0**-2
    ), "mean perpendicular angle not zero"
    return None


def test_sigangledAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the spread in perpendicular angle is much smaller than 1 (the typical spread in the parallel angle)
    da = 0.1
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=True, smallest=False, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=True, smallest=True, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    da = 1.1
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=True, smallest=False, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=True, smallest=True, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    # w/o assuming zeroMean
    da = 0.1
    assert (
        sdf_bovy14.sigangledAngle(
            da, assumeZeroMean=False, smallest=False, simple=False
        )
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=False, smallest=True, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    # w/o assuming zeroMean, at da=0
    da = 0.0
    assert (
        sdf_bovy14.sigangledAngle(
            da, assumeZeroMean=False, smallest=False, simple=False
        )
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=False, smallest=True, simple=False)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    # simple estimate
    da = 0.1
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=False, smallest=False, simple=True)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    assert (
        sdf_bovy14.sigangledAngle(da, assumeZeroMean=False, smallest=True, simple=True)
        < 1.0 / sdf_bovy14.freqEigvalRatio()
    ), "spread in perpendicular angle is not small"
    return None


def test_pangledAngle(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Sanity check pangledAngle, does it peak near zero? Does the mean agree with meandAngle, does the sigma agree with sigdAngle?
    da = 0.1
    assert sdf_bovy14.pangledAngle(0.0, da, smallest=False) > sdf_bovy14.pangledAngle(
        0.1, da, smallest=False
    ), "pangledAngle does not peak near zero"
    assert sdf_bovy14.pangledAngle(0.0, da, smallest=False) > sdf_bovy14.pangledAngle(
        -0.1, da, smallest=False
    ), "pangledAngle does not peak near zero"
    # also for smallest
    assert sdf_bovy14.pangledAngle(0.0, da, smallest=True) > sdf_bovy14.pangledAngle(
        0.1, da, smallest=False
    ), "pangledAngle does not peak near zero"
    assert sdf_bovy14.pangledAngle(0.0, da, smallest=True) > sdf_bovy14.pangledAngle(
        -0.1, da, smallest=False
    ), "pangledAngle does not peak near zero"
    dangles = numpy.linspace(-0.01, 0.01, 201)
    pdangles = (
        numpy.array(
            [sdf_bovy14.pangledAngle(pda, da, smallest=False) for pda in dangles]
        )
    ).flatten()
    assert (
        numpy.fabs(numpy.sum(dangles * pdangles) / numpy.sum(pdangles)) < 10.0**-2.0
    ), "mean calculated using Riemann sum of pangledAngle does not agree with actual mean"
    acsig = sdf_bovy14.sigangledAngle(
        da, assumeZeroMean=True, smallest=False, simple=False
    )
    assert (
        numpy.fabs(
            (
                numpy.sqrt(numpy.sum(dangles**2.0 * pdangles) / numpy.sum(pdangles))
                - acsig
            )
            / acsig
        )
        < 10.0**-2.0
    ), "sigma calculated using Riemann sum of pangledAngle does not agree with actual sigma"
    # also for smallest
    pdangles = (
        numpy.array(
            [sdf_bovy14.pangledAngle(pda, da, smallest=True) for pda in dangles]
        )
    ).flatten()
    assert (
        numpy.fabs(numpy.sum(dangles * pdangles) / numpy.sum(pdangles)) < 10.0**-2.0
    ), "mean calculated using Riemann sum of pangledAngle does not agree with actual mean"
    acsig = sdf_bovy14.sigangledAngle(
        da, assumeZeroMean=True, smallest=True, simple=False
    )
    assert (
        numpy.fabs(
            (
                numpy.sqrt(numpy.sum(dangles**2.0 * pdangles) / numpy.sum(pdangles))
                - acsig
            )
            / acsig
        )
        < 10.0**-1.95
    ), "sigma calculated using Riemann sum of pangledAngle does not agree with actual sigma"
    return None


def test_bovy14_approxaA_inv(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the approximate action-angle conversion near the track works, ie, that the inverse gives the initial point
    # Point on track, interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[22, :]
    check_approxaA_inv(
        sdf_bovy14, -5.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=True
    )
    # Point on track, not interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[152, :]
    check_approxaA_inv(
        sdf_bovy14, -3.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=False
    )
    # Point near track, interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[22, :] * (1.0 + 10.0**-2.0)
    check_approxaA_inv(
        sdf_bovy14, -2.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=True
    )
    # Point near track, not interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[152, :] * (1.0 + 10.0**-2.0)
    check_approxaA_inv(
        sdf_bovy14, -2.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=False
    )
    # Point near end of track, interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[-23, :]
    check_approxaA_inv(
        sdf_bovy14, -2.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=True
    )
    # Point near end of track, not interpolated
    RvR = sdf_bovy14._interpolatedObsTrack[-23, :]
    check_approxaA_inv(
        sdf_bovy14, -2.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=False
    )
    # Now find some trackpoints close to where angles wrap, to test that wrapping is covered properly everywhere
    dphi = (
        numpy.roll(sdf_bovy14._interpolatedObsTrack[:, 5], -1)
        - sdf_bovy14._interpolatedObsTrack[:, 5]
    )
    indx = dphi < 0.0
    RvR = sdf_bovy14._interpolatedObsTrack[indx, :][0, :] * (1.0 + 10.0**-2.0)
    check_approxaA_inv(
        sdf_bovy14, -2.0, RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=False
    )
    return None


def test_bovy14_gaussApprox_onemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # First, test near an interpolated point, without using interpolation (non-trivial)
    tol = -3.0
    trackp = 110
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    # X
    XvX[0] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for X"
    # Y
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[1] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for Y"
    # Z
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[2] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for Z"
    # vX
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[3] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vX"
    # vY
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[4] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for vY"
    # vZ
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for vZ"
    return None


def test_bovy14_gaussApprox_threemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # First, test near an interpolated point, without using interpolation (non-trivial)
    tol = -3.0
    trackp = 110
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    # X,vX,vY
    XvX[0] = None
    XvX[3] = None
    XvX[4] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for X"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vX"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackXY[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for vY"
    # Y,Z,vZ
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[1] = None
    XvX[2] = None
    XvX[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for Y"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for Z"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackXY[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for vZ"
    return None


def test_bovy14_gaussApprox_fivemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # Test near an interpolation point
    tol = -3.0
    trackp = 110
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    # X,Z,vX,vY,vZ
    XvX[0] = None
    XvX[2] = None
    XvX[3] = None
    XvX[4] = None
    XvX[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False, cindx=1)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for X"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for Z"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackXY[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vX"
    assert (
        numpy.fabs(meanp[3] - sdf_bovy14._interpolatedObsTrackXY[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for vY"
    assert (
        numpy.fabs(meanp[4] - sdf_bovy14._interpolatedObsTrackXY[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for vZ"
    # Y,Z,vX,vY,vZ
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[1] = None
    XvX[2] = None
    XvX[3] = None
    XvX[4] = None
    XvX[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=False, cindx=1)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for Y"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for Z"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackXY[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vX"
    assert (
        numpy.fabs(meanp[3] - sdf_bovy14._interpolatedObsTrackXY[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for vY"
    assert (
        numpy.fabs(meanp[4] - sdf_bovy14._interpolatedObsTrackXY[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for vZ"
    return None


def test_bovy14_gaussApprox_interp(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Tests of Gaussian approximation when using interpolation
    tol = -10.0
    trackp = 234
    XvX = list(sdf_bovy14._interpolatedObsTrackXY[trackp, :].flatten())
    XvX[1] = None
    XvX[2] = None
    meanp, varp = sdf_bovy14.gaussApprox(XvX, interp=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 1]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for Y"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for Y"
    # also w/ default (which should be interp=True)
    meanp, varp = sdf_bovy14.gaussApprox(XvX)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackXY[trackp, 1]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for Y"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackXY[trackp, 2]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for Y"
    return None


def test_bovy14_gaussApproxLB_onemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # First, test near an interpolated point, without using interpolation (non-trivial)
    tol = -2.0
    trackp = 102
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    # l
    LB[0] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for l"
    # b
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[1] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for b"
    # d
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[2] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for d"
    # vlos
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[3] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vlos"
    # pmll
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[4] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for pmll"
    # pmbb
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for pmbb"
    return None


def test_bovy14_gaussApproxLB_threemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # First, test near an interpolated point, without using interpolation (non-trivial)
    tol = -1.8
    trackp = 102
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    # l,vlos,pmll
    LB[0] = None
    LB[3] = None
    LB[4] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for l"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackLB[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vlos"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackLB[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for pmll"
    # b,d,pmbb
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[1] = None
    LB[2] = None
    LB[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for b"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackLB[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for d"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackLB[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for pmbb"
    return None


def test_bovy14_gaussApproxLB_fivemissing(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test the Gaussian approximation
    # Test near an interpolation point
    tol = -1.98  # vlos just doesn't make -2.
    trackp = 102
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    # X,Z,vX,vY,vZ
    LB[0] = None
    LB[2] = None
    LB[3] = None
    LB[4] = None
    LB[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, cindx=1, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 0]) < 10.0**tol
    ), "gaussApprox along track does not work for l"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackLB[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for d"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackLB[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vlos"
    assert (
        numpy.fabs(meanp[3] - sdf_bovy14._interpolatedObsTrackLB[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for pmll"
    assert (
        numpy.fabs(meanp[4] - sdf_bovy14._interpolatedObsTrackLB[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for pmbb"
    # b,d,vlos,pmll,pmbb
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[1] = None
    LB[2] = None
    LB[3] = None
    LB[4] = None
    LB[5] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=False, cindx=1, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 1]) < 10.0**tol
    ), "gaussApprox along track does not work for b"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackLB[trackp, 2]) < 10.0**tol
    ), "gaussApprox along track does not work for d"
    assert (
        numpy.fabs(meanp[2] - sdf_bovy14._interpolatedObsTrackLB[trackp, 3]) < 10.0**tol
    ), "gaussApprox along track does not work for vlos"
    assert (
        numpy.fabs(meanp[3] - sdf_bovy14._interpolatedObsTrackLB[trackp, 4]) < 10.0**tol
    ), "gaussApprox along track does not work for pmll"
    assert (
        numpy.fabs(meanp[4] - sdf_bovy14._interpolatedObsTrackLB[trackp, 5]) < 10.0**tol
    ), "gaussApprox along track does not work for pmbb"
    return None


def test_bovy14_gaussApproxLB_interp(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Tests of Gaussian approximation when using interpolation
    tol = -10.0
    trackp = 234
    LB = list(sdf_bovy14._interpolatedObsTrackLB[trackp, :].flatten())
    LB[1] = None
    LB[2] = None
    meanp, varp = sdf_bovy14.gaussApprox(LB, interp=True, lb=True)
    assert (
        numpy.fabs(meanp[0] - sdf_bovy14._interpolatedObsTrackLB[trackp, 1]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for b"
    assert (
        numpy.fabs(meanp[1] - sdf_bovy14._interpolatedObsTrackLB[trackp, 2]) < 10.0**tol
    ), "Gaussian approximation when using interpolation does not work as expected for d"
    return None


def test_bovy14_callMargXZ(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Example from the tutorial and paper
    meanp, varp = sdf_bovy14.gaussApprox([None, None, 2.0 / 8.0, None, None, None])
    xs = (
        numpy.linspace(-3.0 * numpy.sqrt(varp[0, 0]), 3.0 * numpy.sqrt(varp[0, 0]), 11)
        + meanp[0]
    )
    logps = numpy.array(
        [sdf_bovy14.callMarg([x, None, 2.0 / 8.0, None, None, None]) for x in xs]
    )
    ps = numpy.exp(logps)
    ps /= numpy.sum(ps) * (xs[1] - xs[0]) * 8.0
    # Test that the mean is close to the approximation
    assert (
        numpy.fabs(numpy.sum(xs * ps) / numpy.sum(ps) - meanp[0]) < 10.0**-2.0
    ), "mean of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(X|Z)"
    assert (
        numpy.fabs(
            numpy.sqrt(
                numpy.sum(xs**2.0 * ps) / numpy.sum(ps)
                - (numpy.sum(xs * ps) / numpy.sum(ps)) ** 2.0
            )
            - numpy.sqrt(varp[0, 0])
        )
        < 10.0**-2.0
    ), "sigma of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(X|Z)"
    # Test that the mean is close to the approximation, when explicitly setting sigma and ngl
    logps = numpy.array(
        [
            sdf_bovy14.callMarg(
                [x, None, 2.0 / 8.0, None, None, None], ngl=6, nsigma=3.1
            )
            for x in xs
        ]
    )
    ps = numpy.exp(logps)
    ps /= numpy.sum(ps) * (xs[1] - xs[0]) * 8.0
    assert (
        numpy.fabs(numpy.sum(xs * ps) / numpy.sum(ps) - meanp[0]) < 10.0**-2.0
    ), "mean of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(X|Z)"
    assert (
        numpy.fabs(
            numpy.sqrt(
                numpy.sum(xs**2.0 * ps) / numpy.sum(ps)
                - (numpy.sum(xs * ps) / numpy.sum(ps)) ** 2.0
            )
            - numpy.sqrt(varp[0, 0])
        )
        < 10.0**-2.0
    ), "sigma of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(X|Z)"
    return None


def test_bovy14_callMargDPMLL(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # p(D|pmll)
    meanp, varp = sdf_bovy14.gaussApprox([None, None, None, None, 8.0, None], lb=True)
    xs = (
        numpy.linspace(-3.0 * numpy.sqrt(varp[1, 1]), 3.0 * numpy.sqrt(varp[1, 1]), 11)
        + meanp[1]
    )
    logps = numpy.array(
        [sdf_bovy14.callMarg([None, x, None, None, 8.0, None], lb=True) for x in xs]
    )
    ps = numpy.exp(logps)
    ps /= numpy.sum(ps) * (xs[1] - xs[0])
    # Test that the mean is close to the approximation
    assert (
        numpy.fabs(numpy.sum(xs * ps) / numpy.sum(ps) - meanp[1]) < 10.0**-2.0
    ), "mean of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(D|pmll)"
    assert (
        numpy.fabs(
            numpy.sqrt(
                numpy.sum(xs**2.0 * ps) / numpy.sum(ps)
                - (numpy.sum(xs * ps) / numpy.sum(ps)) ** 2.0
            )
            - numpy.sqrt(varp[1, 1])
        )
        < 10.0**-1.0
    ), "sigma of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(D|pmll)"
    # Test options
    assert (
        numpy.fabs(
            sdf_bovy14.callMarg([None, meanp[1], None, None, 8.0, None], lb=True)
            - sdf_bovy14.callMarg(
                [None, meanp[1], None, None, 8.0, None],
                lb=True,
                ro=sdf_bovy14._ro,
                vo=sdf_bovy14._vo,
                R0=sdf_bovy14._R0,
                Zsun=sdf_bovy14._Zsun,
                vsun=sdf_bovy14._vsun,
            )
        )
        < 10.0**-10.0
    ), "callMarg with ro, etc. options set to default does not agree with default"
    cindx = sdf_bovy14.find_closest_trackpointLB(
        None, meanp[1], None, None, 8.0, None, usev=True
    )
    assert (
        numpy.fabs(
            sdf_bovy14.callMarg([None, meanp[1], None, None, 8.0, None], lb=True)
            - sdf_bovy14.callMarg(
                [None, meanp[1], None, None, 8.0, None],
                lb=True,
                cindx=cindx,
                interp=True,
            )
        )
        < 10.0**10.0
    ), "callMarg with cindx set does not agree with it set to default"
    if cindx % 100 > 50:
        cindx = cindx // 100 + 1
    else:
        cindx = cindx // 100
    assert (
        numpy.fabs(
            sdf_bovy14.callMarg(
                [None, meanp[1], None, None, 8.0, None], lb=True, interp=False
            )
            - sdf_bovy14.callMarg(
                [None, meanp[1], None, None, 8.0, None],
                lb=True,
                interp=False,
                cindx=cindx,
            )
        )
        < 10.0**10.0
    ), "callMarg with cindx set does not agree with it set to default"
    # Same w/o interpolation
    return None


def test_bovy14_callMargVLOSPMBB(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # p(vlos|pmbb)
    meanp, varp = sdf_bovy14.gaussApprox([None, None, None, None, None, 5.0], lb=True)
    xs = (
        numpy.linspace(-3.0 * numpy.sqrt(varp[3, 3]), 3.0 * numpy.sqrt(varp[3, 3]), 11)
        + meanp[3]
    )
    logps = numpy.array(
        [sdf_bovy14.callMarg([None, None, None, x, None, 5.0], lb=True) for x in xs]
    )
    ps = numpy.exp(logps - numpy.amax(logps))
    ps /= numpy.sum(ps) * (xs[1] - xs[0])
    # Test that the mean is close to the approximation
    assert (
        numpy.fabs(numpy.sum(xs * ps) / numpy.sum(ps) - meanp[3]) < 5.0
    ), "mean of full PDF calculation does not agree with Gaussian approximation to the level at which this is expected for p(D|pmll)"
    return None


def test_callArgs(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Tests of _parse_call_args
    from galpy.orbit import Orbit

    # Just checking that different types of inputs give the same result
    trackp = 191
    RvR = sdf_bovy14._interpolatedObsTrack[trackp, :].flatten()
    OA = sdf_bovy14._interpolatedObsTrackAA[trackp, :].flatten()
    # RvR vs. array of OA
    s = numpy.ones(2)
    assert numpy.all(
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - sdf_bovy14(
                OA[0] * s,
                OA[1] * s,
                OA[2] * s,
                OA[3] * s,
                OA[4] * s,
                OA[5] * s,
                aAInput=True,
            )
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... and equivalent O,theta,... does not give the same result"
    # RvR vs. OA
    assert (
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - sdf_bovy14(OA[0], OA[1], OA[2], OA[3], OA[4], OA[5], aAInput=True)
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... and equivalent O,theta,... does not give the same result"
    # RvR vs. orbit
    assert (
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - sdf_bovy14(Orbit([RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5]]))
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... and equivalent orbit does not give the same result"
    # RvR vs. list of orbit
    assert (
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - sdf_bovy14([Orbit([RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5]])])
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... and equivalent list of orbit does not give the same result"
    # RvR w/ and w/o log
    assert (
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - numpy.log(
                sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], log=False)
            )
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... log and not log does not give the same result"
    # RvR w/ explicit interp
    assert (
        numpy.fabs(
            sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5])
            - sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4], RvR[5], interp=True)
        )
        < 10.0**-8.0
    ), "__call__ w/ R,vR,... w/ explicit interp does not give the same result as w/o"
    # RvR w/o phi should raise error
    try:
        sdf_bovy14(RvR[0], RvR[1], RvR[2], RvR[3], RvR[4])
    except OSError:
        pass
    else:
        raise AssertionError("__call__ w/o phi does not raise IOError")
    return None


def test_bovy14_sample(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    numpy.random.seed(1)
    RvR = sdf_bovy14.sample(n=1000)
    # Sanity checks
    # Range in Z
    indx = (RvR[3] > 4.0 / 8.0) * (RvR[3] < 5.0 / 8.0)
    meanp, varp = sdf_bovy14.gaussApprox([None, None, 4.5 / 8.0, None, None, None])
    # mean
    assert (
        numpy.fabs(
            numpy.sqrt(meanp[0] ** 2.0 + meanp[1] ** 2.0) - numpy.mean(RvR[0][indx])
        )
        < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    assert (
        numpy.fabs(meanp[4] - numpy.mean(RvR[4][indx])) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    # variance, use smaller range
    RvR = sdf_bovy14.sample(n=10000)
    indx = (RvR[3] > 4.4 / 8.0) * (RvR[3] < 4.6 / 8.0)
    assert (
        numpy.fabs(numpy.sqrt(varp[4, 4]) / numpy.std(RvR[4][indx]) - 1.0) < 10.0**0.0
    ), "Sample spread not similar to track spread"
    # Test that t is returned
    RvRdt = sdf_bovy14.sample(n=10, returndt=True)
    assert len(RvRdt) == 7, "dt not returned with returndt in sample"
    return None


def test_bovy14_sampleXY(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    XvX = sdf_bovy14.sample(n=1000, xy=True)
    # Sanity checks
    # Range in Z
    indx = (XvX[2] > 4.0 / 8.0) * (XvX[2] < 5.0 / 8.0)
    meanp, varp = sdf_bovy14.gaussApprox([None, None, 4.5 / 8.0, None, None, None])
    # mean
    assert (
        numpy.fabs(meanp[0] - numpy.mean(XvX[0][indx])) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    assert (
        numpy.fabs(meanp[1] - numpy.mean(XvX[1][indx])) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    assert (
        numpy.fabs(meanp[3] - numpy.mean(XvX[4][indx])) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    # variance, use smaller range
    XvX = sdf_bovy14.sample(n=10000)
    indx = (XvX[3] > 4.4 / 8.0) * (XvX[3] < 4.6 / 8.0)
    assert (
        numpy.fabs(numpy.sqrt(varp[0, 0]) / numpy.std(XvX[0][indx]) - 1.0) < 10.0**0.0
    ), "Sample spread not similar to track spread"
    # Test that t is returned
    XvXdt = sdf_bovy14.sample(n=10, returndt=True, xy=True)
    assert len(XvXdt) == 7, "dt not returned with returndt in sample"
    return None


def test_bovy14_sampleLB(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    LB = sdf_bovy14.sample(n=1000, lb=True)
    # Sanity checks
    # Range in l
    indx = (LB[0] > 212.5) * (LB[0] < 217.5)
    meanp, varp = sdf_bovy14.gaussApprox([215, None, None, None, None, None], lb=True)
    # mean
    assert (
        numpy.fabs((meanp[0] - numpy.mean(LB[1][indx])) / meanp[0]) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    assert (
        numpy.fabs((meanp[1] - numpy.mean(LB[2][indx])) / meanp[1]) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    assert (
        numpy.fabs((meanp[3] - numpy.mean(LB[4][indx])) / meanp[3]) < 10.0**-2.0
    ), "Sample track does not lie in the same location as the track"
    # variance, use smaller range
    LB = sdf_bovy14.sample(n=10000, lb=True)
    indx = (LB[0] > 214.0) * (LB[0] < 216.0)
    assert (
        numpy.fabs(numpy.sqrt(varp[0, 0]) / numpy.std(LB[1][indx]) - 1.0) < 10.0**0.0
    ), "Sample spread not similar to track spread"
    # Test that t is returned
    LBdt = sdf_bovy14.sample(n=10, returndt=True, lb=True)
    assert len(LBdt) == 7, "dt not returned with returndt in sample"
    return None


def test_bovy14_sampleA(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    AA = sdf_bovy14.sample(n=1000, returnaAdt=True)
    # Sanity checks
    indx = (AA[0][0] > 0.5625) * (AA[0][0] < 0.563)
    assert (
        numpy.fabs(numpy.mean(AA[0][2][indx]) - 0.42525) < 10.0**-1.0
    ), "Sample's vertical frequency at given radial frequency is not as expected"
    # Sanity check w/ time
    AA = sdf_bovy14.sample(n=10000, returnaAdt=True)
    daperp = numpy.sqrt(
        numpy.sum(
            (AA[1] - numpy.tile(sdf_bovy14._progenitor_angle, (10000, 1)).T) ** 2.0,
            axis=0,
        )
    )
    indx = (daperp > 0.24) * (daperp < 0.26)
    assert (
        numpy.fabs(
            (numpy.mean(AA[2][indx]) - sdf_bovy14.meantdAngle(0.25))
            / numpy.mean(AA[2][indx])
        )
        < 10.0**-2.0
    ), "mean stripping time along sample not as expected"
    return None


def test_subhalo_encounters(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that subhalo_encounters acts as expected
    # linear in sigma
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(sigma=300.0 / 220.0)
            / sdf_bovy14.subhalo_encounters(sigma=100.0 / 220.0)
            - 3
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with sigma"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(sigma=200.0 / 220.0)
            / sdf_bovy14.subhalo_encounters(sigma=100.0 / 220.0)
            - 2
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with sigma"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(sigma=50.0 / 220.0, yoon=True)
            / sdf_bovy14.subhalo_encounters(sigma=100.0 / 220.0, yoon=True)
            - 0.5
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with sigma"
    # linear in bmax
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(bmax=1.5)
            / sdf_bovy14.subhalo_encounters(bmax=0.5)
            - 3
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with bmax"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(bmax=1.0)
            / sdf_bovy14.subhalo_encounters(bmax=0.5)
            - 2
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with bmax"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(bmax=0.25, yoon=True)
            / sdf_bovy14.subhalo_encounters(bmax=0.5, yoon=True)
            - 0.5
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with bmax"
    # except when bmax is tiny, then it shouldn't matter
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(bmax=10.0**-7.0)
            / sdf_bovy14.subhalo_encounters(bmax=10.0**-6.0)
            - 1.0
        )
        < 10.0**-5.0
    ), "subhalo_encounters does not become insensitive to bmax at small bmax"
    # linear in nsubhalo
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(nsubhalo=1.5)
            / sdf_bovy14.subhalo_encounters(nsubhalo=0.5)
            - 3
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with nsubhalo"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(nsubhalo=1.0)
            / sdf_bovy14.subhalo_encounters(nsubhalo=0.5)
            - 2
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with nsubhalo"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(nsubhalo=0.25, yoon=True)
            / sdf_bovy14.subhalo_encounters(nsubhalo=0.5, yoon=True)
            - 0.5
        )
        < 10.0**-8.0
    ), "subhalo_encounters does not act linearly with nsubhalo"
    # For nsubhalo = 0.3 should have O(10) impacts (wow, that actually worked!)
    assert (
        numpy.fabs(
            numpy.log10(
                sdf_bovy14.subhalo_encounters(
                    nsubhalo=0.3, bmax=1.5 / 8.0, sigma=120.0 / 220.0
                )
            )
            - 1.0
        )
        < 0.5
    ), "Number of subhalo impacts does not behave as expected for reasonable inputs"
    # Except if you're Yoon et al., then it's more like 30
    assert (
        numpy.fabs(
            numpy.log10(
                sdf_bovy14.subhalo_encounters(
                    nsubhalo=0.3, bmax=1.5 / 8.0, sigma=120.0 / 220.0, yoon=True
                )
            )
            - 1.5
        )
        < 0.5
    ), "Number of subhalo impacts does not behave as expected for reasonable inputs"
    return None


def test_subhalo_encounters_venc(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that the dependence on venc of subhalo_encounters is correct
    def expected_venc(venc, sigma):
        return 1.0 - numpy.exp(-(venc**2.0) / 2.0 / sigma**2.0)

    sigma = 150.0 / 220.0
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=100.0 / 220.0, sigma=sigma)
            / sdf_bovy14.subhalo_encounters(sigma=sigma)
            / expected_venc(100.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=200.0 / 220.0, sigma=sigma)
            / sdf_bovy14.subhalo_encounters(sigma=sigma)
            / expected_venc(200.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=300.0 / 220.0, sigma=sigma)
            / sdf_bovy14.subhalo_encounters(sigma=sigma)
            / expected_venc(300.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    # Should go to 1
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=30000.0 / 220.0, sigma=sigma)
            / sdf_bovy14.subhalo_encounters(sigma=sigma)
            - 1.0
        )
        < 10.0**-4.0
    ), "subhalo_encounters venc behavior is not correct"
    return None


def test_subhalo_encounters_venc_yoon(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup

    # Test that the dependence on venc of subhalo_encounters is correct
    # in the Yoon et al. case
    def expected_venc(venc, sigma):
        return 1.0 - (1.0 + venc**2.0 / 4.0 / sigma**2.0) * numpy.exp(
            -(venc**2.0) / 4.0 / sigma**2.0
        )

    sigma = 150.0 / 220.0
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=100.0 / 220.0, sigma=sigma, yoon=True)
            / sdf_bovy14.subhalo_encounters(sigma=sigma, yoon=True)
            / expected_venc(100.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=200.0 / 220.0, sigma=sigma, yoon=True)
            / sdf_bovy14.subhalo_encounters(sigma=sigma, yoon=True)
            / expected_venc(200.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=300.0 / 220.0, sigma=sigma, yoon=True)
            / sdf_bovy14.subhalo_encounters(sigma=sigma, yoon=True)
            / expected_venc(300.0 / 220.0, sigma)
            - 1.0
        )
        < 10.0**-8.0
    ), "subhalo_encounters venc behavior is not correct"
    # Should go to 1
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(venc=30000.0 / 220.0, sigma=sigma, yoon=True)
            / sdf_bovy14.subhalo_encounters(sigma=sigma, yoon=True)
            - 1.0
        )
        < 10.0**-4.0
    ), "subhalo_encounters venc behavior is not correct"
    return None


def test_bovy14_oppositetrailing_setup_errors():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    lp_false = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    # This is the trailing of the stream that is going the opposite direction
    obs = Orbit(
        [1.56148083, -0.35081535, 1.15481504, 0.88719443, 0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    # Provoke some errors
    try:
        sdft_bovy14 = streamdf(
            sigv / 220.0, progenitor=obs, pot=lp_false, aA=aAI, leading=False
        )  # expl set iterations
    except OSError:
        pass
    else:
        raise AssertionError(
            "streamdf setup w/ potential neq actionAngle-potential did not raise IOError"
        )
    # Warning when deltaAngleTrack is too large (turn warning into error for testing; not using catch_warnings, bc we need this to actually fail [setup doesn't work for such a large deltaAngleTrack])
    import warnings

    warnings.simplefilter("error")
    try:
        sdft_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=False,
            deltaAngleTrack=100.0,
        )  # much too big deltaAngle
    except:
        pass
    else:
        raise AssertionError(
            "streamdf setup w/ deltaAngleTrack too large did not raise warning"
        )
    warnings.simplefilter("default")
    return None


def test_bovy14_oppositetrailing_setup_errors(bovy14_trailing_setup):
    sdft_bovy14 = bovy14_trailing_setup
    assert not sdft_bovy14 is None, "bovy14 streamdf setup did not work"
    return None


def test_calcaAJac():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df.streamdf import calcaAJac
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    R, vR, vT, z, vz, phi = (
        1.56148083,
        0.35081535,
        -1.15481504,
        0.88719443,
        -0.47713334,
        0.12019596,
    )
    jac = calcaAJac([R, vR, vT, z, vz, phi], aAI, dxv=10**-8.0 * numpy.ones(6))
    assert (
        numpy.fabs((numpy.fabs(numpy.linalg.det(jac)) - R) / R) < 10.0**-2.0
    ), "Determinant of (x,v) -> (J,theta) transformation is not equal to 1"
    # Now w/ frequencies
    jac = calcaAJac(
        [R, vR, vT, z, vz, phi],
        aAI,
        dxv=10**-8.0 * numpy.ones(6),
        actionsFreqsAngles=True,
    )
    # extract (J,theta)
    Jajac = jac[
        numpy.array(
            [True, True, True, False, False, False, True, True, True], dtype="bool"
        ),
        :,
    ]
    assert (
        numpy.fabs((numpy.fabs(numpy.linalg.det(Jajac)) - R) / R) < 10.0**-2.0
    ), "Determinant of (x,v) -> (J,theta) transformation is not equal to 1, when calculated with actionsFreqsAngles"
    # extract (O,theta)
    Oajac = jac[
        numpy.array(
            [False, False, False, True, True, True, True, True, True], dtype="bool"
        ),
        :,
    ]
    OJjac = calcaAJac(
        [R, vR, vT, z, vz, phi], aAI, dxv=10**-8.0 * numpy.ones(6), dOdJ=True
    )
    assert (
        numpy.fabs(
            (
                numpy.fabs(numpy.linalg.det(Oajac))
                - R * numpy.fabs(numpy.linalg.det(OJjac))
            )
            / R
            / numpy.fabs(numpy.linalg.det(OJjac))
        )
        < 10.0**-2.0
    ), "Determinant of (x,v) -> (O,theta) is not equal to that of dOdJ"
    OJjac = calcaAJac(
        [R, vR, vT, z, vz, phi], aAI, dxv=10**-8.0 * numpy.ones(6), freqs=True
    )
    assert (
        numpy.fabs(
            (numpy.fabs(numpy.linalg.det(Oajac)) - numpy.fabs(numpy.linalg.det(OJjac)))
            / numpy.fabs(numpy.linalg.det(OJjac))
        )
        < 10.0**-2.0
    ), "Determinant of (x,v) -> (O,theta) is not equal to that calculated w/ actionsFreqsAngles"
    return None


def test_calcaAJacLB():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df.streamdf import calcaAJac
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    R, vR, vT, z, vz, phi = (
        1.56148083,
        0.35081535,
        -1.15481504,
        0.88719443,
        -0.47713334,
        0.12019596,
    )
    # First convert these to l,b,d,vlos,pmll,pmbb
    XYZ = coords.galcencyl_to_XYZ(R * 8.0, phi, z * 8.0, Xsun=8.0, Zsun=0.02)
    l, b, d = coords.XYZ_to_lbd(XYZ[0], XYZ[1], XYZ[2], degree=True)
    vXYZ = coords.galcencyl_to_vxvyvz(
        vR * 220.0, vT * 220.0, vz * 220.0, phi=phi, vsun=[10.0, 240.0, -10.0]
    )
    vlos, pmll, pmbb = coords.vxvyvz_to_vrpmllpmbb(
        vXYZ[0], vXYZ[1], vXYZ[2], l, b, d, degree=True
    )
    jac = calcaAJac(
        [
            l,
            b,
            d,
            vlos,
            pmll,
            pmbb,
        ],
        aAI,
        dxv=10**-8.0 * numpy.ones(6),
        lb=True,
        R0=8.0,
        Zsun=0.02,
        vsun=[10.0, 240.0, -10.0],
        ro=8.0,
        vo=220.0,
    )
    lbdjac = numpy.fabs(
        numpy.linalg.det(coords.lbd_to_XYZ_jac(l, b, d, vlos, pmll, pmbb, degree=True))
    )
    assert (
        numpy.fabs(
            (numpy.fabs(numpy.linalg.det(jac)) * 8.0**3.0 * 220.0**3.0 - lbdjac)
            / lbdjac
        )
        < 10.0**-2.0
    ), "Determinant of (x,v) -> (J,theta) transformation is not equal to 1"
    return None


def test_estimateTdisrupt(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    from galpy.util import conversion

    td = numpy.log10(
        sdf_bovy14.estimateTdisrupt(1.0) * conversion.time_in_Gyr(220.0, 8.0)
    )
    assert (td > 0.0) * (td < 1.0), "estimate of disruption time is not a few Gyr"
    return None


def test_plotting(bovy14_setup, bovy14_trailing_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    sdft_bovy14 = bovy14_trailing_setup
    # Check plotting routines
    check_track_plotting(sdf_bovy14, "R", "Z")
    check_track_plotting(sdf_bovy14, "R", "Z", phys=True)  # do 1 with phys
    check_track_plotting(sdf_bovy14, "R", "Z", interp=False)  # do 1 w/o interp
    check_track_plotting(sdf_bovy14, "R", "X", spread=0)
    check_track_plotting(sdf_bovy14, "R", "Y", spread=0)
    check_track_plotting(sdf_bovy14, "R", "phi")
    check_track_plotting(sdf_bovy14, "R", "vZ")
    check_track_plotting(sdf_bovy14, "R", "vZ", phys=True)  # do 1 with phys
    check_track_plotting(sdf_bovy14, "R", "vZ", interp=False)  # do 1 w/o interp
    check_track_plotting(sdf_bovy14, "R", "vX", spread=0)
    check_track_plotting(sdf_bovy14, "R", "vY", spread=0)
    check_track_plotting(sdf_bovy14, "R", "vT")
    check_track_plotting(sdf_bovy14, "R", "vR")
    check_track_plotting(sdf_bovy14, "ll", "bb")
    check_track_plotting(sdf_bovy14, "ll", "bb", interp=False)  # do 1 w/o interp
    check_track_plotting(sdf_bovy14, "ll", "dist")
    check_track_plotting(sdf_bovy14, "ll", "vlos")
    check_track_plotting(sdf_bovy14, "ll", "pmll")
    delattr(sdf_bovy14, "_ObsTrackLB")  # rm, to test that this gets recalculated
    check_track_plotting(sdf_bovy14, "ll", "pmbb")
    # Also test plotCompareTrackAAModel
    sdf_bovy14.plotCompareTrackAAModel()
    sdft_bovy14.plotCompareTrackAAModel()  # has multi
    return None


def test_2ndsetup():
    # Test related to #195: when we re-do the setup with the same progenitor, we should get the same
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nosetup=True,
    )  # won't look at track
    rsdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nosetup=True,
    )  # won't look at track
    assert (
        numpy.fabs(sdf_bovy14.misalignment() - rsdf_bovy14.misalignment()) < 0.01
    ), "misalignment not the same when setting up the same streamdf w/ a previously used progenitor"
    assert (
        numpy.fabs(sdf_bovy14.freqEigvalRatio() - rsdf_bovy14.freqEigvalRatio()) < 0.01
    ), "freqEigvalRatio not the same when setting up the same streamdf w/ a previously used progenitor"
    return None


def test_bovy14_trackaa(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    # Test that the explicitly-calculated frequencies along the track are close to those that the track is based on (Fardal test, #194)
    from galpy.orbit import Orbit

    aastream = sdf_bovy14._ObsTrackAA  # freqs and angles that the track is based on
    RvR = sdf_bovy14._ObsTrack  # the track in R,vR,...
    aastream_expl = numpy.reshape(
        numpy.array(
            [sdf_bovy14._aA.actionsFreqsAngles(Orbit(trvr))[3:] for trvr in RvR]
        ),
        aastream.shape,
    )
    # frequencies, compare to offset between track and progenitor (spread in freq ~ 1/6 that diff, so as long as we're smaller than that we're fine)
    assert numpy.all(
        numpy.fabs(
            (aastream[:, :3] - aastream_expl[:, :3])
            / (aastream[0, :3] - sdf_bovy14._progenitor_Omega)
        )
        < 0.05
    ), "Explicitly calculated frequencies along the track do not agree with the frequencies on which the track is based for bovy14 setup"
    # angles
    assert numpy.all(
        numpy.fabs((aastream[:, 3:] - aastream_expl[:, 3:]) / 2.0 / numpy.pi) < 0.001
    ), "Explicitly calculated angles along the track do not agree with the angles on which the track is based for bovy14 setup"
    return None


def test_fardalpot_trackaa():
    # Test that the explicitly-calculated frequencies along the track are close to those that the track is based on (Fardal test, #194); used to fail for the potential suggested by Fardal
    # First setup this specific streamdf instance
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import FlattenedPowerPotential, IsochronePotential
    from galpy.util import conversion  # for unit conversions

    # test nested list of potentials
    pot = [
        IsochronePotential(b=0.8, normalize=0.8),
        [FlattenedPowerPotential(alpha=-0.7, q=0.6, normalize=0.2)],
    ]
    aAI = actionAngleIsochroneApprox(pot=pot, b=0.9)
    obs = Orbit([1.10, 0.32, -1.15, 1.10, 0.31, 3.0])
    sigv = 1.3  # km/s
    sdf_fardal = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=pot,
        aA=aAI,
        leading=True,
        nTrackChunks=21,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
    )
    # First test that the misalignment is indeed large
    assert (
        numpy.fabs(sdf_fardal.misalignment() / numpy.pi * 180.0) > 4.0
    ), "misalignment in Fardal test is not large"
    # Now run the test
    aastream = sdf_fardal._ObsTrackAA  # freqs and angles that the track is based on
    RvR = sdf_fardal._ObsTrack  # the track in R,vR,...
    aastream_expl = numpy.reshape(
        numpy.array(
            [sdf_fardal._aA.actionsFreqsAngles(Orbit(trvr))[3:] for trvr in RvR]
        ),
        aastream.shape,
    )
    # frequencies, compare to offset between track and progenitor (spread in freq ~ 1/6 that diff, so as long as we're smaller than that we're fine)
    # print numpy.fabs((aastream[:,:3]-aastream_expl[:,:3])/(aastream[0,:3]-sdf_fardal._progenitor_Omega))
    # print numpy.fabs((aastream[:,3:]-aastream_expl[:,3:])/2./numpy.pi)
    assert numpy.all(
        numpy.fabs(
            (aastream[:, :3] - aastream_expl[:, :3])
            / (aastream[0, :3] - sdf_fardal._progenitor_Omega)
        )
        < 0.05
    ), "Explicitly calculated frequencies along the track do not agree with the frequencies on which the track is based for Fardal setup"
    # angles
    assert numpy.all(
        numpy.fabs((aastream[:, 3:] - aastream_expl[:, 3:]) / 2.0 / numpy.pi) < 0.001
    ), "Explicitly calculated angles along the track do not agree with the angles on which the track is based for Fardal setup"
    return None


def test_fardalwmwpot_trackaa():
    # Test that the explicitly-calculated frequencies along the track are close to those that the track is based on (Fardal test, #194)
    # First setup this specific streamdf instance
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion  # for unit conversions

    aAI = actionAngleIsochroneApprox(pot=MWPotential2014, b=0.6)
    obs = Orbit([1.10, 0.32, -1.15, 1.10, 0.31, 3.0])
    sigv = 1.3  # km/s
    sdf_fardal = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=MWPotential2014,
        aA=aAI,
        leading=True,
        multi=True,
        nTrackChunks=21,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
    )
    # First test that the misalignment is indeed large
    assert (
        numpy.fabs(sdf_fardal.misalignment() / numpy.pi * 180.0) > 1.0
    ), "misalignment in Fardal test is not large enough"
    # Now run the test
    aastream = sdf_fardal._ObsTrackAA  # freqs and angles that the track is based on
    RvR = sdf_fardal._ObsTrack  # the track in R,vR,...
    aastream_expl = numpy.reshape(
        numpy.array(
            [sdf_fardal._aA.actionsFreqsAngles(Orbit(trvr))[3:] for trvr in RvR]
        ),
        aastream.shape,
    )
    # frequencies, compare to offset between track and progenitor (spread in freq ~ 1/6 that diff, so as long as we're smaller than that we're fine)
    # print numpy.fabs((aastream[:,:3]-aastream_expl[:,:3])/(aastream[0,:3]-sdf_fardal._progenitor_Omega))
    # print numpy.fabs((aastream[:,3:]-aastream_expl[:,3:])/2./numpy.pi)
    assert numpy.all(
        numpy.fabs(
            (aastream[:, :3] - aastream_expl[:, :3])
            / (aastream[0, :3] - sdf_fardal._progenitor_Omega)
        )
        < 0.05
    ), "Explicitly calculated frequencies along the track do not agree with the frequencies on which the track is based for Fardal setup"
    # angles
    assert numpy.all(
        numpy.fabs((aastream[:, 3:] - aastream_expl[:, 3:]) / 2.0 / numpy.pi) < 0.001
    ), "Explicitly calculated angles along the track do not agree with the angles on which the track is based for Fardal setup"
    return None


def test_setup_progIsTrack():
    # Test that setting up with progIsTrack=True gives a track that is very close to the given progenitor, such that it works as it should
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    sigv = 0.365  # km/s
    sdfp = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        progIsTrack=True,
    )
    assert numpy.all(
        numpy.fabs(obs.vxvv[0] - sdfp._ObsTrack[0, :]) < 10.0**-3.0
    ), "streamdf setup with progIsTrack does not return a track that is close to the given orbit at the start"
    # Integrate the orbit a little bit and test at a further point
    obs.integrate(numpy.linspace(0.0, 2.0, 10001), lp)
    indx = numpy.argmin(numpy.fabs(sdfp._interpolatedObsTrack[:, 0] - 1.75))
    oindx = numpy.argmin(numpy.fabs(obs.orbit[0, :, 0] - 1.75))
    assert numpy.all(
        numpy.fabs(sdfp._interpolatedObsTrack[indx, :5] - obs.orbit[0, oindx, :5])
        < 10.0**-2.0
    ), "streamdf setup with progIsTrack does not return a track that is close to the given orbit somewhat further from the start"
    return None


def test_bovy14_useTM_poterror():
    if WIN32:
        return None  # skip on Windows, because no TM
    # Test that setting up the stream model with useTM, but a different
    # actionAngleTorus potential raises a IOError
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    elp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    aAT = actionAngleTorus(pot=elp, tol=0.001)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    with pytest.raises(IOError) as excinfo:
        sdftm = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            useTM=aAT,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        )
    return None


def test_bovy14_useTM(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    if WIN32:
        return None  # skip on Windows, because no TM
    # Test that setting up with useTM is very close to the Bovy (2014) setup
    # Imports
    from scipy import interpolate

    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    aAT = actionAngleTorus(pot=lp, tol=0.001)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    sdftm = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        useTM=aAT,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
    )
    sindx = numpy.argsort(sdftm._interpolatedObsTrackLB[:, 0])
    interpb = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 1],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpb(sdf_bovy14._interpolatedObsTrackLB[:, 0])
            - sdf_bovy14._interpolatedObsTrackLB[:, 1]
        )
        < 0.1
    ), "stream track computed with useTM not close to that without in b"
    interpD = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 2],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpD(sdf_bovy14._interpolatedObsTrackLB[:, 0])
            - sdf_bovy14._interpolatedObsTrackLB[:, 2]
        )
        < 0.04
    ), "stream track computed with useTM not close to that without in distance"
    interpV = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 3],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpV(sdf_bovy14._interpolatedObsTrackLB[:, 0])
            - sdf_bovy14._interpolatedObsTrackLB[:, 3]
        )
        < 0.6
    ), "stream track computed with useTM not close to that without in line-of-sight velocity"
    return None


def test_bovy14_useTM_useTMHessian(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    if WIN32:
        return None  # skip on Windows, because no TM
    # Test that setting up with useTM is very close to the Bovy (2014) setup
    # Imports
    from scipy import interpolate

    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    aAT = actionAngleTorus(pot=lp, tol=0.001)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    sdftm = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        useTM=aAT,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        useTMHessian=True,
        multi=2,
    )
    sindx = numpy.argsort(sdftm._interpolatedObsTrackLB[:, 0])
    interpb = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 1],
        k=3,
    )
    # Only compare part closest to the progenitor, where this works
    cindx = sdf_bovy14._interpolatedObsTrackLB[:, 0] < 215.0
    assert numpy.all(
        numpy.fabs(
            interpb(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 1]
        )
        < 0.75
    ), (
        "stream track computed with useTM and useTMHessian not close to that without in b by %g"
        % (
            numpy.amax(
                numpy.fabs(
                    interpb(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
                    - sdf_bovy14._interpolatedObsTrackLB[cindx, 1]
                )
            )
        )
    )
    interpD = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 2],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpD(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 2]
        )
        < 0.2
    ), "stream track computed with useTM and useTMHessian not close to that without in distance"
    interpV = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 3],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpV(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 3]
        )
        < 4.0
    ), "stream track computed with useTM and useTMHessian not close to that without in line-of-sight velocity"
    return None


def test_bovy14_useTM_approxConstTrackFreq(bovy14_setup):
    # Load the streamdf object
    sdf_bovy14 = bovy14_setup
    if WIN32:
        return None  # skip on Windows, because no TM
    # Test that setting up with useTM is very close to the Bovy (2014) setup
    # Imports
    from scipy import interpolate

    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    aAT = actionAngleTorus(pot=lp, tol=0.001)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    sdftm = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        useTM=aAT,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        approxConstTrackFreq=True,
    )
    sindx = numpy.argsort(sdftm._interpolatedObsTrackLB[:, 0])
    interpb = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 1],
        k=3,
    )
    # Only compare part closest to the progenitor, where this should be a good approx.
    cindx = sdf_bovy14._interpolatedObsTrackLB[:, 0] < 215.0
    assert numpy.all(
        numpy.fabs(
            interpb(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 1]
        )
        < 0.1
    ), "stream track computed with useTM and approxConstTrackFreq not close to that without in b"
    interpD = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 2],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpD(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 2]
        )
        < 0.04
    ), "stream track computed with useTM and approxConstTrackFreq not close to that without in distance"
    interpV = interpolate.InterpolatedUnivariateSpline(
        sdftm._interpolatedObsTrackLB[sindx, 0],
        sdftm._interpolatedObsTrackLB[sindx, 3],
        k=3,
    )
    assert numpy.all(
        numpy.fabs(
            interpV(sdf_bovy14._interpolatedObsTrackLB[cindx, 0])
            - sdf_bovy14._interpolatedObsTrackLB[cindx, 3]
        )
        < 0.6
    ), "stream track computed with useTM and approxConstTrackFreq not close to that without in line-of-sight velocity"
    return None


def check_track_prog_diff(sdf, d1, d2, tol, phys=False):
    observe = [sdf._R0, 0.0, sdf._Zsun]
    observe.extend(sdf._vsun)
    # Test that the stream and the progenitor are close together in Z
    trackR = sdf._parse_track_dim(
        d1, interp=True, phys=phys
    )  # bit hacky to use private function
    trackZ = sdf._parse_track_dim(
        d2, interp=True, phys=phys
    )  # bit hacky to use private function
    ts = sdf._progenitor.t[sdf._progenitor.t < sdf._trackts[-1]]
    progR = sdf._parse_progenitor_dim(
        d1, ts, ro=sdf._ro, vo=sdf._vo, obs=observe, phys=phys
    )
    progZ = sdf._parse_progenitor_dim(
        d2, ts, ro=sdf._ro, vo=sdf._vo, obs=observe, phys=phys
    )
    # Interpolate progenitor, st we can put it on the same grid as the stream
    interpProgZ = interpolate.InterpolatedUnivariateSpline(progR, progZ, k=3)
    maxdevZ = numpy.amax(numpy.fabs(interpProgZ(trackR) - trackZ))
    assert (
        maxdevZ < tol
    ), f"Stream track deviates more from progenitor track in {d2} vs. {d1} than expected; max. deviation = {maxdevZ:f}"
    return None


def check_track_spread(sdf, d1, d2, tol1, tol2, phys=False, interp=True):
    # Check that the spread around the track is small
    addx, addy = sdf._parse_track_spread(d1, d2, interp=interp, phys=phys)
    assert (
        numpy.amax(addx) < tol1
    ), f"Stream track spread is larger in {d1} than expected; max. deviation = {numpy.amax(addx):f}"
    assert (
        numpy.amax(addy) < tol2
    ), f"Stream track spread is larger in {d2} than expected; max. deviation = {numpy.amax(addy):f}"
    return None


def check_track_plotting(sdf, d1, d2, phys=False, interp=True, spread=2, ls="-"):
    # Test that we can plot the stream track
    if not phys and d1 == "R" and d2 == "Z":  # one w/ default
        sdf.plotTrack(d1=d1, d2=d2, interp=interp, spread=spread)
        sdf.plotProgenitor(d1=d1, d2=d2)
    else:
        sdf.plotTrack(
            d1=d1,
            d2=d2,
            interp=interp,
            spread=spread,
            scaleToPhysical=phys,
            ls="none",
            color="k",
            lw=2.0,
            marker=".",
        )
        sdf.plotProgenitor(d1=d1, d2=d2, scaleToPhysical=phys)
    return None


def check_closest_trackpoint(sdf, trackp, usev=False, xy=True, interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if interp:
        if xy:
            RvR = sdf._interpolatedObsTrackXY[trackp, :]
        else:
            RvR = sdf._interpolatedObsTrack[trackp, :]
    else:
        if xy:
            RvR = sdf._ObsTrackXY[trackp, :]
        else:
            RvR = sdf._ObsTrack[trackp, :]
    R = RvR[0]
    vR = RvR[1]
    vT = RvR[2]
    z = RvR[3]
    vz = RvR[4]
    phi = RvR[5]
    indx = sdf.find_closest_trackpoint(
        R, vR, vT, z, vz, phi, interp=interp, xy=xy, usev=usev
    )
    assert indx == trackp, "Closest trackpoint to a trackpoint is not that trackpoint"
    # Same test for a slight offset
    doff = 10.0**-5.0
    indx = sdf.find_closest_trackpoint(
        R + doff,
        vR + doff,
        vT + doff,
        z + doff,
        vz + doff,
        phi + doff,
        interp=interp,
        xy=xy,
        usev=usev,
    )
    assert indx == trackp, (
        "Closest trackpoint to close to a trackpoint is not that trackpoint (%i,%i)"
        % (
            indx,
            trackp,
        )
    )
    return None


def check_closest_trackpointLB(sdf, trackp, usev=False, interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if trackp == -1:  # in this case, rm some coordinates
        trackp = 1
        if interp:
            RvR = sdf._interpolatedObsTrackLB[trackp, :]
        else:
            RvR = sdf._ObsTrackLB[trackp, :]
        R = RvR[0]
        vR = None
        vT = RvR[2]
        z = None
        vz = RvR[4]
        phi = None
    elif trackp == -2:  # in this case, rm some coordinates
        trackp = 1
        if interp:
            RvR = sdf._interpolatedObsTrackLB[trackp, :]
        else:
            RvR = sdf._ObsTrackLB[trackp, :]
        R = None
        vR = RvR[1]
        vT = None
        z = RvR[3]
        vz = None
        phi = RvR[5]
    elif trackp == -3:  # in this case, rm some coordinates
        trackp = 1
        if interp:
            RvR = sdf._interpolatedObsTrackLB[trackp, :]
        else:
            RvR = sdf._ObsTrackLB[trackp, :]
        R = RvR[0]
        vR = RvR[1]
        vT = None
        z = None
        vz = None
        phi = None
    else:
        if interp:
            RvR = sdf._interpolatedObsTrackLB[trackp, :]
        else:
            RvR = sdf._ObsTrackLB[trackp, :]
        R = RvR[0]
        vR = RvR[1]
        vT = RvR[2]
        z = RvR[3]
        vz = RvR[4]
        phi = RvR[5]
    indx = sdf.find_closest_trackpointLB(
        R, vR, vT, z, vz, phi, interp=interp, usev=usev
    )
    assert (
        indx == trackp
    ), "Closest trackpoint to a trackpoint is not that trackpoint in LB"
    # Same test for a slight offset
    doff = 10.0**-5.0
    if not R is None:
        R = R + doff
    if not vR is None:
        vR = vR + doff
    if not vT is None:
        vT = vT + doff
    if not z is None:
        z = z + doff
    if not vz is None:
        vz = vz + doff
    if not phi is None:
        phi = phi + doff
    indx = sdf.find_closest_trackpointLB(
        R, vR, vT, z, vz, phi, interp=interp, usev=usev
    )
    assert indx == trackp, (
        "Closest trackpoint to close to a trackpoint is not that trackpoint in LB (%i,%i)"
        % (indx, trackp)
    )
    return None


def check_closest_trackpointaA(sdf, trackp, interp=True):
    # Check that the closest trackpoint (close )to a trackpoint is the trackpoint
    if interp:
        RvR = sdf._interpolatedObsTrackAA[trackp, :]
    else:
        RvR = sdf._ObsTrackAA[trackp, :]
    # These aren't R,vR etc., but frequencies and angles
    R = RvR[0]
    vR = RvR[1]
    vT = RvR[2]
    z = RvR[3]
    vz = RvR[4]
    phi = RvR[5]
    indx = sdf._find_closest_trackpointaA(R, vR, vT, z, vz, phi, interp=interp)
    assert (
        indx == trackp
    ), "Closest trackpoint to a trackpoint is not that trackpoint for AA"
    # Same test for a slight offset
    doff = 10.0**-5.0
    indx = sdf._find_closest_trackpointaA(
        R + doff, vR + doff, vT + doff, z + doff, vz + doff, phi + doff, interp=interp
    )
    assert indx == trackp, (
        "Closest trackpoint to close to a trackpoint is not that trackpoint for AA (%i,%i)"
        % (indx, trackp)
    )
    return None


def check_approxaA_inv(sdf, tol, R, vR, vT, z, vz, phi, interp=True):
    # Routine to test that approxaA works
    # Calculate frequency-angle coords
    Oa = sdf._approxaA(R, vR, vT, z, vz, phi, interp=interp)
    # Now go back to real space
    RvR = sdf._approxaAInv(
        Oa[0, 0], Oa[1, 0], Oa[2, 0], Oa[3, 0], Oa[4, 0], Oa[5, 0], interp=interp
    ).flatten()
    if phi > numpy.pi:
        phi -= 2.0 * numpy.pi
    if phi < -numpy.pi:
        phi += 2.0 * numpy.pi
    # print numpy.fabs((RvR[0]-R)/R), numpy.fabs((RvR[1]-vR)/vR), numpy.fabs((RvR[2]-vT)/vT), numpy.fabs((RvR[3]-z)/z), numpy.fabs((RvR[4]-vz)/vz), numpy.fabs((RvR[5]-phi)/phi)
    assert numpy.fabs((RvR[0] - R) / R) < 10.0**tol, (
        "R after _approxaA and _approxaAInv does not agree with initial R; relative difference = %g"
        % (numpy.fabs((RvR[0] - R) / R))
    )
    assert (
        numpy.fabs((RvR[1] - vR) / vR) < 10.0**tol
    ), "vR after _approxaA and _approxaAInv does not agree with initial vR"
    assert (
        numpy.fabs((RvR[2] - vT) / vT) < 10.0**tol
    ), "vT after _approxaA and _approxaAInv does not agree with initial vT"
    assert (
        numpy.fabs((RvR[3] - z) / z) < 10.0**tol
    ), "z after _approxaA and _approxaAInv does not agree with initial z"
    assert (
        numpy.fabs((RvR[4] - vz) / vz) < 10.0**tol
    ), "vz after _approxaA and _approxaAInv does not agree with initial vz"
    assert numpy.fabs((RvR[5] - phi) / numpy.pi) < 10.0**tol, (
        "phi after _approxaA and _approxaAInv does not agree with initial phi; relative difference = %g"
        % (numpy.fabs((RvR[5] - phi) / phi))
    )
    return None
