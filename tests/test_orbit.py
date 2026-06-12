##############################TESTS ON ORBITS##################################
import os
import os.path
import platform
import sys
import warnings

WIN32 = platform.system() == "Windows"
MACOS = platform.system() == "Darwin"
import signal
import subprocess
import time

import astropy
import numpy
import pytest

PY2 = sys.version < "3"
_APY3 = astropy.__version__ > "3"
from test_actionAngle import reset_warning_registry
from test_potential import (
    BurkertPotentialNoC,
    KuzminKutuzovOblateStaeckelWrapperPotential,
    MWP14CylindricallySeparablePotentialWrapper,
    NFWTwoPowerTriaxialPotential,
    altExpwholeDiskSCFPotential,
    expwholeDiskMultipoleExpansionPotential,
    expwholeDiskSCFPotential,
    fullyRotatedTriaxialNFWPotential,
    mockAdiabaticContractionMWP14WrapperPotential,
    mockCombLinearPotential,
    mockFlatCorotatingRotationSpiralArmsPotential,
    mockFlatCosmphiDiskPotential,
    mockFlatCosmphiDiskwBreakPotential,
    mockFlatDehnenBarPotential,
    mockFlatDehnenSmoothBarPotential,
    mockFlatEllipticalDiskPotential,
    mockFlatGaussianAmplitudeBarPotential,
    mockFlatLopsidedDiskPotential,
    mockFlatSoftenedNeedleBarPotential,
    mockFlatSolidBodyRotationMultipoleExpansionPotential,
    mockFlatSolidBodyRotationPlanarSpiralArmsPotential,
    mockFlatSolidBodyRotationSpiralArmsPotential,
    mockFlatSpiralArmsPotential,
    mockFlatSteadyLogSpiralPotential,
    mockFlatTransientLogSpiralPotential,
    mockFlatTrulyCorotatingRotationSpiralArmsPotential,
    mockFlatTrulyGaussianAmplitudeBarPotential,
    mockFlatWeaklyTDMultipoleExpansionPotential,
    mockFlatWeaklyTDNonaxiM3MultipoleExpansionPotential,
    mockInterpSphericalPotential,
    mockKuzminLikeWrapperPotential,
    mockMovingObjectLongIntPotential,
    mockMultipoleExpansionAxiPotential,
    mockMultipoleExpansionLimitedGridPotential,
    mockMultipoleExpansionPotential,
    mockMultipoleExpansionSphericalPotential,
    mockRotatedAndTiltedMWP14WrapperPotential,
    mockRotatingFlatSpiralArmsPotential,
    mockSCFAxiDensity1Potential,
    mockSCFAxiDensity2Potential,
    mockSCFDensityPotential,
    mockSCFNFWPotential,
    mockSCFZeeuwPotential,
    mockSimpleLinearPotential,
    mockSlowFlatDecayingDehnenSmoothBarPotential,
    mockSlowFlatDehnenBarPotential,
    mockSlowFlatDehnenSmoothBarPotential,
    mockSlowFlatEllipticalDiskPotential,
    mockSlowFlatSteadyLogSpiralPotential,
    mockSpecialRotatingFlatSpiralArmsPotential,
    mockTDMultipoleExpansionLimitedGridPotential,
    nestedListPotential,
    oblateHernquistPotential,
    oblateNFWPotential,
    prolateJaffePotential,
    prolateNFWPotential,
    sech2DiskMultipoleExpansionPotential,
    sech2DiskSCFPotential,
    specialFlattenedPowerPotential,
    specialMiyamotoNagaiPotential,
    testlinearMWPotential,
    testMWPotential,
    testNullPotential,
    testorbitHenonHeilesPotential,
    testplanarMWPotential,
    triaxialLogarithmicHaloPotential,
    triaxialNFWPotential,
)

from galpy import potential
from galpy.potential.Potential import _check_c
from galpy.util import galpyWarning
from galpy.util.coords import _K

_GHACTIONS = bool(os.getenv("GITHUB_ACTIONS"))
if not _GHACTIONS:
    _QUICKTEST = True  # Run a more limited set of tests
else:
    _QUICKTEST = True  # Also do this for GH Actions, bc otherwise it takes too long
_NOLONGINTEGRATIONS = False
# Don't show all warnings, to reduce log output
warnings.simplefilter("always", galpyWarning)


# Test whether the energy of simple orbits is conserved for different
# integrators; tests 3D and 2D orbits, 1D orbits done separately below
# Parametrized list of tests generated programmatically in conftest.py
def test_energy_jacobi_conservation(pot, ttol, tjactol, firstTest):
    if _NOLONGINTEGRATIONS:
        return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 210.0, 5001)  # ~7.5 Gyr at the Solar circle
    growtimes = numpy.linspace(0.0, 280.0, 5001)  # for pots that grow slowly
    fasttimes = numpy.linspace(0.0, 14.0, 501)  # ~0.5 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    try:
        tclass = getattr(potential, pot)
    except AttributeError:
        tclass = getattr(sys.modules[__name__], pot)
    # if not pot == 'NFWPotential' and not pot == 'mockSlowFlatDecayingDehnenSmoothBarPotential': continue
    tp = tclass()
    if not hasattr(tp, "normalize"):
        return None  # skip these
    tp.normalize(1.0)
    if hasattr(tp, "toPlanar"):
        ptp = tp.toPlanar()
    else:
        ptp = None
    for integrator in integrators:
        if integrator == "dopr54_c" and (
            "Spiral" in pot or "Lopsided" in pot or "Dehnen" in pot or "Cosmphi" in pot
        ):
            ttimes = growtimes
        elif (
            integrator == "dopr54_c"
            and not "MovingObject" in pot
            and not pot == "FerrersPotential"
        ):
            ttimes = times
        else:
            ttimes = fasttimes
        # First track azimuth
        o = setup_orbit_energy(tp, axi=False, henon="Henon" in pot)
        if isinstance(tp, testMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        elif isinstance(tp, testplanarMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        else:
            o.integrate(ttimes, tp, method=integrator)
        tEs = o.E(ttimes)
        # print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
        if (
            not "Bar" in pot
            and not "Spiral" in pot
            and not "MovingObject" in pot
            and not "Slow" in pot
            and not "SolidBodyRotationMultipole" in pot
            and not "WeaklyTDMultipole" in pot
        ):
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g"
                % (pot, integrator, (numpy.std(tEs) / numpy.mean(tEs)))
            )
        # Jacobi
        if (
            "Elliptical" in pot
            or "Lopsided" in pot
            or "DehnenSmoothBar" in pot
            or "SolidBodyRotation" in pot
            or "CorotatingRotation" in pot
            or "GaussianAmplitudeBar" in pot
            or "WeaklyTDMultipole" in pot
            or "SteadyLogSpiralPotential" in pot
            or pot == "mockMovingObjectLongIntPotential"
            or "Cosmphi" in pot
            or "triaxialLog" in pot
            or "RotatedAndTilted" in pot
            or "Henon" in pot
        ):
            tJacobis = o.Jacobi(ttimes, pot=tp)
        else:
            tJacobis = o.Jacobi(ttimes)
        #            print(p, (numpy.std(tJacobis)/numpy.mean(tJacobis))**2.)
        assert (numpy.std(tJacobis) / numpy.mean(tJacobis)) ** 2.0 < 10.0**tjactol, (
            "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s at the %g level"
            % (pot, integrator, (numpy.std(tJacobis) / numpy.mean(tJacobis)) ** 2.0)
        )
        if firstTest or "testMWPotential" in pot:
            # Some basic checking of the energy and Jacobi functions
            assert (o.E(pot=None) - o.E(pot=tp)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
            )
            assert (o.E() - o.E(0.0)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with o.E() and o.E(0.) do not agree"
            )
            assert (o.Jacobi(OmegaP=None) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
            )
            assert (o.Jacobi(pot=None) - o.Jacobi(pot=tp)) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
            )
            assert (
                o.Jacobi(pot=None) - o.Jacobi(pot=potential.CompositePotential([tp]))
            ) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=[the Potential the orbit was integrated with] do not agree"
            )
            if not tp.isNonAxi:
                assert (o.Jacobi(OmegaP=1.0) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                )
                assert (
                    o.Jacobi(OmegaP=[0.0, 0.0, 1.0]) - o.Jacobi(OmegaP=1.0)
                ) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=[0,0,1] for axisymmetric potential is not equal to o.Jacobi with OmegaP=1"
                )
                assert (
                    o.Jacobi(OmegaP=numpy.array([0.0, 0.0, 1.0])) - o.Jacobi(OmegaP=1.0)
                ) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=[0,0,1] for axisymmetric potential is not equal to o.Jacobi with OmegaP=1"
                )
            o = setup_orbit_energy(tp, axi=False, henon="Henon" in pot)
            try:
                o.E()
            except AttributeError:
                pass
            else:
                raise AssertionError(
                    "o.E() before the orbit was integrated did not throw an AttributeError"
                )
            if not isinstance(tp, potential.linearPotential):
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.Jacobi() before the orbit was integrated did not throw an AttributeError"
                    )
        if "MovingObject" in pot:
            if _QUICKTEST and not (
                ("NFW" in pot and not tp.isNonAxi and "SCF" not in pot)
                or "linearMWPotential" in pot
                or ("Burkert" in pot and not tp.hasC)
            ):
                break
            else:
                continue
        # AnyAxisymmetricRazorThinDiskPot has bad behavior for odeint and isn't really meant for orbit integration
        if "AnyAxisymmetricRazorThinDiskPotential" in pot:
            break
        # Now do axisymmetric
        if not tp.isNonAxi:
            o = setup_orbit_energy(tp, axi=True, henon="Henon" in pot)
            if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
                o.integrate(ttimes, tp._potlist, method=integrator)
            else:
                o.integrate(ttimes, tp, method=integrator)
            tEs = o.E(ttimes)
            #            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g"
                % (pot, integrator, (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0)
            )
            # Jacobi
            tJacobis = o.Jacobi(ttimes)
            assert (
                numpy.std(tJacobis) / numpy.mean(tJacobis)
            ) ** 2.0 < 10.0**tjactol, (
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s"
                % (pot, integrator)
            )
            if firstTest or "MWPotential" in pot:
                # Some basic checking of the energy function
                assert (o.E(pot=None) - o.E(pot=tp)) ** 2.0 < 10.0**ttol, (
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
                )
                assert (o.E() - o.E(0.0)) ** 2.0 < 10.0**ttol, (
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                )
                assert (o.Jacobi(OmegaP=None) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                )
                assert (o.Jacobi(pot=None) - o.Jacobi(pot=tp)) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                )
                assert (
                    (
                        o.Jacobi(pot=None)
                        - o.Jacobi(
                            pot=potential.NullPotential(amp=0.0) + tp
                        )  # get around not knowing whether we need a CompositePotential or a planarCompositePotential
                    )
                    ** 2.0
                    < 10.0** ttol
                ), (
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                )
                assert (o.Jacobi(OmegaP=1.0) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                )
                o = setup_orbit_energy(tp, axi=True, henon="Henon" in pot)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.E() before the orbit was integrated did not throw an AttributeError"
                    )
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.Jacobi() before the orbit was integrated did not throw an AttributeError"
                    )
        if ptp is None:
            if _QUICKTEST and not (
                ("NFW" in pot and not tp.isNonAxi and "SCF" not in pot)
                or ("Burkert" in pot and not tp.hasC)
            ):
                break
            else:
                continue
        # Same for a planarPotential
        if pot == "mockRotatedAndTiltedMWP14WrapperPotential":
            break
        #            print integrator
        if not ptp is None and not ptp.isNonAxi:
            o = setup_orbit_energy(ptp, axi=True)
            if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
                o.integrate(
                    ttimes, potential.toPlanarPotential(tp._potlist), method=integrator
                )
            else:
                o.integrate(ttimes, ptp, method=integrator)
            tEs = o.E(ttimes)
            #                print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s"
                % (pot, integrator)
            )
            # Jacobi
            tJacobis = o.Jacobi(ttimes)
            assert (
                numpy.std(tJacobis) / numpy.mean(tJacobis)
            ) ** 2.0 < 10.0**tjactol, (
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s"
                % (pot, integrator)
            )
            if firstTest or "MWPotential" in pot:
                # Some basic checking of the energy function
                assert (o.E(pot=None) - o.E(pot=ptp)) ** 2.0 < 10.0**ttol, (
                    "Energy calculated with pot=None and pot=the planarPotential the orbit was integrated with do not agree for planarPotential"
                )
                assert (o.E(pot=None) - o.E(pot=tp)) ** 2.0 < 10.0**ttol, (
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree for planarPotential"
                )
                assert (o.E() - o.E(0.0)) ** 2.0 < 10.0**ttol, (
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                )
                assert (o.Jacobi(OmegaP=None) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                )
                assert (o.Jacobi(pot=None) - o.Jacobi(pot=tp)) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                )
                assert (o.Jacobi(OmegaP=1.0) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                    "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                )
                o = setup_orbit_energy(ptp, axi=True)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.E() before the orbit was integrated did not throw an AttributeError"
                    )
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.Jacobi() before the orbit was integrated did not throw an AttributeError"
                    )
        # Same for a planarPotential, track azimuth
        o = setup_orbit_energy(ptp, axi=False)
        if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
            o.integrate(
                ttimes, potential.toPlanarPotential(tp._potlist), method=integrator
            )
        else:
            o.integrate(ttimes, ptp, method=integrator)
        tEs = o.E(ttimes)
        # print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
        if (
            not "Bar" in pot
            and not "Spiral" in pot
            and not "SolidBodyRotationMultipole" in pot
            and not "WeaklyTDMultipole" in pot
        ):
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g"
                % (pot, integrator, (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0)
            )
        # Jacobi
        if (
            "DehnenSmoothBar" in pot
            or "DehnenBar" in pot
            or "SolidBodyRotation" in pot
            or "CorotatingRotation" in pot
            or "GaussianAmplitudeBar" in pot
            or "SpiralArmsPotential" in pot
            or "nestedListPotential" in pot
            or "WeaklyTDMultipole" in pot
        ):
            tJacobis = o.Jacobi(ttimes, pot=tp)
        else:
            tJacobis = o.Jacobi(ttimes)
        assert (numpy.std(tJacobis) / numpy.mean(tJacobis)) ** 2.0 < 10.0**tjactol, (
            "Jacobi integral conservation during the orbit integration fails by %g for potential %s and integrator %s"
            % ((numpy.std(tJacobis) / numpy.mean(tJacobis)) ** 2.0, pot, integrator)
        )
        if firstTest or "MWPotential" in pot:
            # Some basic checking of the energy function
            assert (o.E(pot=None) - o.E(pot=ptp)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with pot=None and pot=the planarPotential the orbit was integrated with do not agree for planarPotential"
            )
            assert (o.E(pot=None) - o.E(pot=tp)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree for planarPotential"
            )
            assert (o.E() - o.E(0.0)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with o.E() and o.E(0.) do not agree"
            )
            assert (o.Jacobi(OmegaP=None) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
            )
            assert (o.Jacobi(pot=None) - o.Jacobi(pot=tp)) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
            )
            assert (o.Jacobi(OmegaP=1.0) - o.Jacobi()) ** 2.0 < 10.0**ttol, (
                "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
            )
            o = setup_orbit_energy(ptp, axi=False)
            try:
                o.E()
            except AttributeError:
                pass
            else:
                raise AssertionError(
                    "o.E() before the orbit was integrated did not throw an AttributeError"
                )
            try:
                o.Jacobi()
            except AttributeError:
                pass
            else:
                raise AssertionError(
                    "o.Jacobi() before the orbit was integrated did not throw an AttributeError"
                )
            firstTest = False
        # Same for a planarPotential, but integrating w/ the potential directly, rather than the toPlanar instance; this tests that those potential attributes are passed to C correctly
        #            print integrator
        if not ptp is None and not ptp.isNonAxi:
            o = setup_orbit_energy(ptp, axi=True)
            if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
                o.integrate(ttimes, tp._potlist, method=integrator)
            else:
                o.integrate(ttimes, tp, method=integrator)
            tEs = o.E(ttimes)
            # print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g"
                % (pot, integrator, (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0)
            )
            # Jacobi
            if (
                "DehnenSmoothBar" in pot
                or "SolidBodyRotation" in pot
                or "CorotatingRotation" in pot
                or "GaussianAmplitudeBar" in pot
            ):
                tJacobis = o.Jacobi(ttimes, pot=tp)
            else:
                tJacobis = o.Jacobi(ttimes)
            assert (
                numpy.std(tJacobis) / numpy.mean(tJacobis)
            ) ** 2.0 < 10.0**tjactol, (
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s"
                % (pot, integrator)
            )
        # Same for a planarPotential, track azimuth
        o = setup_orbit_energy(ptp, axi=False)
        if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        else:
            o.integrate(ttimes, tp, method=integrator)
        tEs = o.E(ttimes)
        #            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
        if (
            not "Bar" in pot
            and not "Spiral" in pot
            and not "SolidBodyRotationMultipole" in pot
            and not "WeaklyTDMultipole" in pot
        ):
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s"
                % (pot, integrator)
            )
        # Jacobi
        if (
            "DehnenSmoothBar" in pot
            or "DehnenBar" in pot
            or "SolidBodyRotation" in pot
            or "CorotatingRotation" in pot
            or "GaussianAmplitudeBar" in pot
            or "SpiralArmsPotential" in pot
            or "nestedListPotential" in pot
            or "WeaklyTDMultipole" in pot
        ):
            tJacobis = o.Jacobi(ttimes, pot=tp)
        else:
            tJacobis = o.Jacobi(ttimes)
        assert (numpy.std(tJacobis) / numpy.mean(tJacobis)) ** 2.0 < 10.0**tjactol, (
            "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s"
            % (pot, integrator)
        )
        if _QUICKTEST and not (
            ("NFW" in pot and not tp.isNonAxi and "SCF" not in pot)
            or ("Burkert" in pot and not tp.hasC)
        ):
            break
    # raise AssertionError
    return None


# Test whether the energy of 1D orbits is conserved for different integrators
# Parametrized list of tests generated programmatically in conftest.py
def test_energy_conservation_linear(pot, ttol, firstTest):
    if _NOLONGINTEGRATIONS:
        return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 210.0, 5001)  # ~7.5 Gyr at the Solar circle
    growtimes = numpy.linspace(0.0, 280.0, 5001)  # for pots that grow slowly
    fasttimes = numpy.linspace(0.0, 14.0, 501)  # ~0.5 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Setup instance of potential
    try:
        tclass = getattr(potential, pot)
    except AttributeError:
        tclass = getattr(sys.modules[__name__], pot)
    # if not p == 'NFWPotential' and not p == 'mockFlatGaussianAmplitudeBarPotential': continue
    tp = tclass()
    if hasattr(tp, "toVertical"):
        if not hasattr(tp, "normalize"):
            return None  # skip these
        tp.normalize(1.0)
        tp = tp.toVertical(1.2, phi=0.3)
    elif isinstance(tp, potential.linearPotential):
        pass
    else:  # not 3D --> 1D or 1D, so skip
        return None
    for integrator in integrators:
        if integrator == "dopr54_c" and (
            "Spiral" in pot or "Lopsided" in pot or "Dehnen" in pot or "Cosmphi" in pot
        ):
            ttimes = growtimes
        elif (
            integrator == "dopr54_c"
            and not "MovingObject" in pot
            and not pot == "FerrersPotential"
            and not pot == "AnyAxisymmetricRazorThinDiskPotential"
        ):
            ttimes = times
        else:
            ttimes = fasttimes
        # First track azimuth
        o = setup_orbit_energy(tp)
        if isinstance(tp, testMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        elif isinstance(tp, testplanarMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        elif isinstance(tp, testlinearMWPotential):
            o.integrate(ttimes, tp._potlist, method=integrator)
        else:
            o.integrate(ttimes, tp, method=integrator)
        tEs = o.E(ttimes)
        # print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
        if (
            not "Bar" in pot
            and not "Spiral" in pot
            and not "MovingObject" in pot
            and not "Slow" in pot
            and not "SolidBodyRotationMultipole" in pot
            and not "WeaklyTDMultipole" in pot
        ):
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g"
                % (pot, integrator, (numpy.std(tEs) / numpy.mean(tEs)))
            )
        if firstTest or "testMWPotential" in pot or "linearMWPotential" in pot:
            # Some basic checking of the energy function
            assert (o.E(pot=None) - o.E(pot=tp)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
            )
            assert (o.E() - o.E(0.0)) ** 2.0 < 10.0**ttol, (
                "Energy calculated with o.E() and o.E(0.) do not agree"
            )
            o = setup_orbit_energy(tp, axi=False, henon="Henon" in pot)
            try:
                o.E()
            except AttributeError:
                pass
            else:
                raise AssertionError(
                    "o.E() before the orbit was integrated did not throw an AttributeError"
                )
        if _QUICKTEST and not (
            pot == "NFWPotential" or ("Burkert" in pot and not tp.hasC)
        ):
            break
    return None


# Test some long-term integrations for the symplectic integrators
def test_energy_symplec_longterm():
    if _NOLONGINTEGRATIONS:
        return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 10000.0, 100001)  # ~360 Gyr at the Solar circle
    integrators = [
        "leapfrog_c",  # don't do leapfrog, because it takes too long
        "symplec4_c",
        "symplec6_c",
    ]
    # Only use KeplerPotential
    pots = ["KeplerPotential"]
    # tolerances in log10
    tol = {}
    tol["default"] = -20.0
    tol["leapfrog_c"] = -16.0
    tol["leapfrog"] = -16.0
    for p in pots:
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        for integrator in integrators:
            if integrator in list(tol.keys()):
                ttol = tol[integrator]
            else:
                ttol = tol["default"]
            o = setup_orbit_energy(tp)
            o.integrate(times, tp, method=integrator)
            tEs = o.E(times)
            #            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            #            print p, ((numpy.mean(o.E(times[0:20]))-numpy.mean(o.E(times[-20:-1])))/numpy.mean(tEs))**2.
            assert (numpy.std(tEs) / numpy.mean(tEs)) ** 2.0 < 10.0**ttol, (
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %.20f"
                % (p, integrator, (numpy.std(tEs) / numpy.mean(tEs)) ** 2)
            )
            # Check whether there is a trend
            linfit = numpy.polyfit(times, tEs, 1)
            #            print p
            assert linfit[0] ** 2.0 < 10.0**ttol, (
                "Absence of secular trend in energy conservation fails for potential %s and symplectic integrator %s"
                % (p, integrator)
            )
    # raise AssertionError
    return None


def _cart_accel_3d(pot, rect, t=0.0):
    """Cartesian acceleration (ax,ay,az) at rectangular position rect=(x,y,z).

    The time t is forwarded to the force evaluators so this is also correct for
    explicitly time-dependent potentials (e.g. a rotating bar): the tangent
    vector along the flow is dx/dt = f(x(t),t), which is an exact solution of the
    (time-dependent) variational equation only when f is evaluated at the same t.
    """
    from galpy.potential import (
        evaluatephitorques,
        evaluateRforces,
        evaluatezforces,
    )

    x, y, z = rect[0], rect[1], rect[2]
    R = numpy.sqrt(x**2.0 + y**2.0)
    phi = numpy.arctan2(y, x)
    cp, sp = numpy.cos(phi), numpy.sin(phi)
    Rforce = evaluateRforces(pot, R, z, phi=phi, t=t)
    phitorque = evaluatephitorques(pot, R, z, phi=phi, t=t)
    zforce = evaluatezforces(pot, R, z, phi=phi, t=t)
    ax = cp * Rforce - sp / R * phitorque
    ay = sp * Rforce + cp / R * phitorque
    az = zforce
    return numpy.array([ax, ay, az])


def _skip_flowdir_identity(pot):
    """Whether to skip ONLY the flow-direction identity check (check (3) in
    test_liouville_3d) for this potential. This does NOT gate Liouville
    (det M = 1), symplecticity (M^T Omega M = Omega), or the FD-of-flow check --
    those are run for every potential, time-dependent or not, exactly as in the
    2D test_liouville_planar.

    The flow-direction identity M.f(x0) = f(x(t)) relies on the phase-space
    velocity f(x) being itself a solution of the variational equation, which only
    holds for an AUTONOMOUS system. For an explicitly time-dependent potential
    (e.g. a rotating bar) d/dt f(x(t),t) = J.f + df/dt picks up the extra df/dt
    term, so dx/dt no longer solves the (df/dt-free) variational equation and the
    identity fails -- a property of this particular check, not of the Hessian,
    which remains pinned by checks (2) and (4) and by test_dxdv_3d_c_vs_python.

    Detect time dependence by comparing the Cartesian acceleration at the test IC
    at two different times; returns True if the force depends on t."""
    rect = numpy.array([0.9, 0.18, 0.05])  # generic off-plane, off-axis point
    a0 = _cart_accel_3d(pot, rect, t=0.0)
    a1 = _cart_accel_3d(pot, rect, t=1.3)
    return numpy.amax(numpy.fabs(a1 - a0)) > 1e-12


def _orbit_rect_3d(o, ts):
    """Stack the integrated 6D orbit in rectangular phase space (nt,6)."""
    x = o.x(ts)
    y = o.y(ts)
    z = o.z(ts)
    vx = o.vx(ts)
    vy = o.vy(ts)
    vz = o.vz(ts)
    return numpy.array([x, y, z, vx, vy, vz]).T


def _integrate_stm_3d(pot, ic, times, integrator):
    """Integrate the full 6x6 state-transition matrix M(t) in CARTESIAN
    phase-space coordinates (x,y,z,vx,vy,vz): propagate the 6 canonical basis
    deviation vectors with rectIn=rectOut=True and stack them as the columns
    of M. Returns an (nt,6,6) array with M(times[0])=identity."""
    from galpy.orbit import Orbit

    canonical = numpy.eye(6)
    Mcols = []
    for ii in range(6):
        o = Orbit(ic)
        o.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method=integrator,
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        Mcols.append(o.getOrbit_dxdv())
    return numpy.array(Mcols).transpose(1, 2, 0)  # (nt,6,6), columns = e_i images


# 3D variational equations: Liouville (det M=1), symplecticity, and -- the
# parts that actually pin down the Cartesian Hessian K -- the flow-direction,
# finite-difference, and 2D-reduction bridge checks. See the docstring of each
# block below for which property is being validated.
def test_liouville_3d(pot):
    from galpy.orbit import Orbit

    integrators = [
        "dopr54_c",
        "dop853_c",
        "rk4_c",
        "rk6_c",
        "dop853",
        "odeint",
    ]
    # Generic, fully 3D initial condition (R,vR,vT,z,vz,phi)
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    # Omega in (x,y,z,vx,vy,vz) order for the symplecticity check
    Omega = numpy.zeros((6, 6))
    Omega[:3, 3:] = numpy.eye(3)
    Omega[3:, :3] = -numpy.eye(3)
    canonical = numpy.eye(6)
    pname = pot.__class__.__name__
    for integrator in integrators:
        if "_c" in integrator:
            rtol, atol = 1e-12, 1e-12
        else:
            rtol, atol = 1e-12, 1e-12
        # ---- Build the 6x6 monodromy/STM M: integrate the 6 canonical
        # basis deviation vectors with rectIn=rectOut=True ----
        Mcols = []
        for ii in range(6):
            o = Orbit(ic)
            o.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method=integrator,
                rectIn=True,
                rectOut=True,
                rtol=rtol,
                atol=atol,
            )
            Mcols.append(o.getOrbit_dxdv()[-1, :])
        M = numpy.array(Mcols).T  # columns are the propagated basis vectors
        # (1) Liouville: det M = 1 [necessary check]. The fixed-step,
        # lower-order rk4_c/rk6_c integrators are intrinsically a little less
        # accurate, so use a slightly looser (but still meaningful) tolerance.
        det_tol = 1e-7 if integrator in ("rk4_c", "rk6_c") else 1e-8
        detM = numpy.linalg.det(M)
        assert numpy.fabs(detM - 1.0) < det_tol, (
            f"3D Liouville det(M)={detM:g} differs from 1 for {pname}, "
            f"integrator {integrator}"
        )
        # (2) Symplecticity: M^T Omega M = Omega [necessary]
        symperr = numpy.amax(numpy.fabs(numpy.dot(M.T, numpy.dot(Omega, M)) - Omega))
        assert symperr < 1e-7, (
            f"3D symplecticity ||M^T Omega M - Omega||={symperr:g} too large "
            f"for {pname}, integrator {integrator}"
        )
        # (3) Flow-direction (validates K, free): for an AUTONOMOUS (time-
        # independent) system the phase-space velocity f(x) is an exact solution
        # of the variational equation, so the integrated deviation seeded with
        # f(x0) must equal f(x(t)) along the orbit. This identity does NOT hold
        # for an explicitly time-dependent potential: there d/dt f(x(t),t) =
        # J.f + df/dt picks up the extra df/dt term, so dx/dt no longer solves
        # the (df/dt-free) variational equation. Skip this single check for
        # time-dependent potentials -- their Hessian is still pinned by the
        # symplecticity check (2) above, the FD-of-flow check (4) below, and the
        # C-vs-Python check in test_dxdv_3d_c_vs_python.
        o = Orbit(ic)
        # The BASE orbit here supplies the ground truth f(x(t)) of check (3);
        # at scipy's default tolerances the pure-Python odeint base orbit
        # carries ~1e-6 of position error for some entries (measured 1.0e-6
        # for the composite MWPotential2014, just over the bound below, while
        # every other integrator sits at <2e-9) -- integrator noise in the
        # ground truth, not a Hessian error. Tighten ONLY the odeint base
        # orbit (the deviation integrations are already run at rtol=atol=1e-12;
        # MWPotential2014's odeint flow check measures 9.9e-11 with this);
        # every other integrator path is left byte-identical.
        base_kw = {"rtol": 1e-12, "atol": 1e-12} if integrator == "odeint" else {}
        o.integrate(times, pot, method=integrator, **base_kw)
        rect_orbit = _orbit_rect_3d(o, times)
        if not _skip_flowdir_identity(pot):  # gates ONLY check (3) below
            f0 = numpy.empty(6)
            f0[:3] = rect_orbit[0, 3:]
            f0[3:] = _cart_accel_3d(pot, rect_orbit[0, :3], t=times[0])
            o2 = Orbit(ic)
            o2.integrate_dxdv(
                f0,
                times,
                pot,
                method=integrator,
                rectIn=True,
                rectOut=True,
                rtol=rtol,
                atol=atol,
            )
            dev = o2.getOrbit_dxdv()  # (nt,6) rectangular deviation
            ftrue = numpy.empty((len(times), 6))
            ftrue[:, :3] = rect_orbit[:, 3:]
            for jj in range(len(times)):
                ftrue[jj, 3:] = _cart_accel_3d(pot, rect_orbit[jj, :3], t=times[jj])
            flowerr = numpy.amax(numpy.fabs(dev - ftrue))
            assert flowerr < 1e-6, (
                f"3D flow-direction deviation differs from f(x(t)) by {flowerr:g} "
                f"for {pname}, integrator {integrator}"
            )
        # (4) Finite-difference of the flow (validates K): integrate a base
        # orbit and orbits perturbed by eps*e_i and compare the dxdv column
        # to the FD of the integrated flow. Do a couple of i.
        eps = 1e-7
        obase = Orbit(ic)
        obase.integrate(times, pot, method=integrator)
        base_rect = _orbit_rect_3d(obase, times)
        for ii in [0, 2, 4]:  # an x, a z, and a vy perturbation
            pert_ic_rect = base_rect[0].copy()
            pert_ic_rect[ii] += eps
            # build a cylindrical IC from the perturbed rect IC
            from galpy.util import coords

            Rp, phip, Zp = coords.rect_to_cyl(
                pert_ic_rect[0], pert_ic_rect[1], pert_ic_rect[2]
            )
            vRp, vTp, vzp = coords.rect_to_cyl_vec(
                pert_ic_rect[3],
                pert_ic_rect[4],
                pert_ic_rect[5],
                pert_ic_rect[0],
                pert_ic_rect[1],
                pert_ic_rect[2],
            )
            opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
            opert.integrate(times, pot, method=integrator)
            pert_rect = _orbit_rect_3d(opert, times)
            fd = (pert_rect - base_rect) / eps
            # dxdv column for e_i
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method=integrator,
                rectIn=True,
                rectOut=True,
                rtol=rtol,
                atol=atol,
            )
            col = odx.getOrbit_dxdv()
            fderr = numpy.amax(numpy.fabs(fd - col))
            assert fderr < 1e-4, (
                f"3D finite-difference of the flow for e_{ii} differs from "
                f"the dxdv column by {fderr:g} for {pname}, integrator {integrator}"
            )
    return None


# Consolidated C-vs-Python 3D variational (dxdv) check, parametrized over the FULL
# categorized registry of potentials with a complete 3D C Hessian (see
# conftest.py). The C 3D variational integrator (dopr54_c) must match the trusted
# pure-Python analytic-2nd-derivative reference (dop853) to <1e-6 for UNIT-magnitude
# deviations along the canonical e_x, e_z, e_vy directions. That C-vs-Python
# comparison is what pins the Hessian VALUES (det(M)=1, validated separately in
# test_liouville_3d, is necessary but not sufficient: it holds for any symmetric K).
# For the non-axisymmetric category we additionally require that the zphideriv term
# (d2Phi/dz/dphi) is genuinely nonzero along the orbit, so the C zphideriv coupling
# is really exercised rather than multiplied by 0.
def test_dxdv_3d_c_vs_python(pot, pot_category):
    from galpy.orbit import Orbit
    from galpy.potential import evaluatephizderivs

    pname = pot.__class__.__name__
    # The full 3D C Hessian must be advertised so the C 3D path is actually taken.
    assert pot.hasC_dxdv3d, f"{pname} should advertise hasC_dxdv3d"
    # Generic, fully 3D initial condition (R,vR,vT,z,vz,phi)
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    if pot_category == "nonaxisymmetric":
        # Guard against a vacuous test: the z-phi coupling must be nonzero along the
        # orbit, otherwise the C zphideriv term is multiplied by 0.
        assert pot.isNonAxi, f"{pname} flagged nonaxisymmetric but isNonAxi is False"
        obase = Orbit(ic)
        obase.integrate(times, pot, method="dop853_c")
        base_rect = _orbit_rect_3d(obase, times)
        zphi_vals = numpy.array(
            [
                evaluatephizderivs(
                    pot,
                    numpy.sqrt(base_rect[jj, 0] ** 2 + base_rect[jj, 1] ** 2),
                    base_rect[jj, 2],
                    phi=numpy.arctan2(base_rect[jj, 1], base_rect[jj, 0]),
                )
                for jj in range(len(times))
            ]
        )
        assert numpy.amax(numpy.fabs(zphi_vals)) > 1e-3, (
            f"{pname} d2Phi/dz/dphi must be nonzero along the orbit to exercise "
            f"the C zphideriv term"
        )
    # UNIT-magnitude deviations: a ~1e-4 relative error shows as ~1e-4 absolute
    # (a tiny deviation would scale it down and hide bugs). Cover e_x, e_z, e_vy.
    maxdiff = 0.0
    for ii in [0, 2, 4]:
        dev = canonical[ii]
        oc = Orbit(ic)
        oc.integrate_dxdv(
            dev,
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        op = Orbit(ic)
        op.integrate_dxdv(
            dev,
            times,
            pot,
            method="dop853",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    assert maxdiff < 1e-6, (
        f"3D C variational integration for {pname} differs from the pure-Python "
        f"reference by {maxdiff:g} (unit deviation)"
    )
    return None


# ---- Closed-form STM ground truth (1/2): exact 3D isotropic harmonic
# oscillator. Inside its radius R, HomogeneousSpherePotential is
# Phi(r) = amp (r^2 - 3 R^2), an EXACTLY harmonic potential with
# omega^2 = 2 amp in every Cartesian coordinate. The STM of the harmonic flow
# is closed form: per-axis 2x2 blocks in (x_i, v_i)
#   [[ cos(w t), sin(w t)/w ], [ -w sin(w t), cos(w t) ]],
# i.e. M(t) = [[c I3, (s/w) I3], [-w s I3, c I3]] in the (x,y,z,vx,vy,vz)
# ordering. Unlike det(M)=1/symplecticity (necessary conditions only) and the
# C-vs-Python/FD-of-flow checks (cross-checks between integrators), this pins
# the integrated STM VALUES against an analytic ground truth that is fully
# independent of galpy's variational machinery.
def test_dxdv_3d_closed_form_stm_harmonic():
    from galpy.orbit import Orbit
    from galpy.potential import HomogeneousSpherePotential

    pot = HomogeneousSpherePotential(amp=1.0, R=3.0, normalize=True)
    assert pot.hasC_dxdv3d, "HomogeneousSphere should advertise hasC_dxdv3d"
    # omega^2 = 2 amp = d2Phi/dR2 at ANY interior point: read it off the
    # potential and guard the harmonic premise (R2deriv = z2deriv = constant,
    # Rzderiv = 0) at two distinct interior points
    omega2 = pot.R2deriv(0.7, 0.2)
    assert omega2 > 0.0
    for RR, zz in [(0.7, 0.2), (1.3, -0.4)]:
        assert numpy.fabs(pot.R2deriv(RR, zz) - omega2) < 1e-14
        assert numpy.fabs(pot.z2deriv(RR, zz) - omega2) < 1e-14
        assert numpy.fabs(pot.Rzderiv(RR, zz)) < 1e-14
    w = numpy.sqrt(omega2)
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 11)
    # non-vacuity (1): the orbit must stay INSIDE the sphere, where the
    # potential is exactly harmonic (outside it is Keplerian)
    o = Orbit(ic)
    o.integrate(times, pot, method="dop853_c")
    r = numpy.sqrt(o.x(times) ** 2.0 + o.y(times) ** 2.0 + o.z(times) ** 2.0)
    assert numpy.amax(r) < 0.9 * pot.R, (
        "test precondition: the orbit must stay well inside the homogeneous "
        "sphere for the potential to be exactly harmonic along it"
    )

    def analytic_M(tk):
        c, s = numpy.cos(w * tk), numpy.sin(w * tk)
        Mref = numpy.zeros((6, 6))
        Mref[:3, :3] = c * numpy.eye(3)
        Mref[:3, 3:] = s / w * numpy.eye(3)
        Mref[3:, :3] = -w * s * numpy.eye(3)
        Mref[3:, 3:] = c * numpy.eye(3)
        return Mref

    # non-vacuity (2): the deviations genuinely evolve over the test interval
    assert numpy.amax(numpy.fabs(analytic_M(times[-1]) - numpy.eye(6))) > 0.5
    for integrator in ["dopr54_c", "dop853_c", "dop853"]:
        Mt = _integrate_stm_3d(pot, ic, times, integrator)
        maxdiff = numpy.amax(
            numpy.fabs(Mt - numpy.array([analytic_M(tk) for tk in times]))
        )
        # measured ~4e-12 for all three integrators; 1e-8 leaves a wide margin
        assert maxdiff < 1e-8, (
            f"integrated 3D STM differs from the closed-form harmonic-oscillator "
            f"STM by {maxdiff:g} for integrator {integrator}"
        )
    return None


# ---- Closed-form STM ground truth (2/2): Kepler, via the Lagrange f,g
# solution. For mu = amp = 1 the two-body propagator in the
# change-of-eccentric-anomaly (dE = E - E0) formulation (Battin 1999, secs.
# 4.3-4.4; Vallado, "f and g functions in terms of the eccentric anomaly") is
#   a      = 1/(2/r0 - |v0|^2)                       (vis-viva; a>0: elliptic)
#   sigma0 = r0vec . v0vec                           (= r0 dr0/dt, mu=1)
#   Kepler: t = a^{3/2} [dE + (sigma0/sqrt(a))(1 - cos dE)
#                           - (1 - r0/a) sin dE]
#   r(t)   = a + (r0 - a) cos dE + sigma0 sqrt(a) sin dE
#   f      = 1 - (a/r0)(1 - cos dE)        g    = t - a^{3/2} (dE - sin dE)
#   fdot   = -(sqrt(a)/(r r0)) sin dE      gdot = 1 - (a/r)(1 - cos dE)
#   x(t) = f x0 + g v0 ;  v(t) = fdot x0 + gdot v0.
# The STM is the total derivative of (x(t),v(t)) wrt (x0,v0) at FIXED t:
# every scalar above depends on the initial state through (r0, a, sigma0), and
# dE depends on it IMPLICITLY through Kepler's equation F(dE; r0,a,sigma0,t)=0,
# so  d dE/ds0 = -(dF/ds0)/(dF/ddE)  with  dF/ddE = r(t)/a > 0 for an elliptic
# orbit (which also makes F monotonic in dE: the Newton solve below is safe).
# The closed-form partials assembled in _kepler_fg_stm below were derived by
# hand from these expressions and verified to machine precision (~2e-13)
# against an independent sympy implicit differentiation of the same f,g
# solution; the reference is fully independent of galpy's integrators.
def _kepler_fg_stm(s0, dE, t):
    """Analytic Kepler (mu=1) STM M = d(x,v)_t/d(x,v)_0 for an elliptic orbit,
    from the Lagrange f,g functions in the dE formulation; dE must solve
    Kepler's equation for (s0, t). s0 = (x0,y0,z0,vx0,vy0,vz0)."""
    rvec0 = numpy.array(s0[:3])
    vvec0 = numpy.array(s0[3:])
    r0 = numpy.sqrt(numpy.sum(rvec0**2.0))
    a = 1.0 / (2.0 / r0 - numpy.sum(vvec0**2.0))
    sqa = numpy.sqrt(a)
    sigma0 = numpy.sum(rvec0 * vvec0)
    cE, sE = numpy.cos(dE), numpy.sin(dE)
    r1 = a + (r0 - a) * cE + sigma0 * sqa * sE
    f = 1.0 - a / r0 * (1.0 - cE)
    g = t - a * sqa * (dE - sE)
    fd = -sqa / (r1 * r0) * sE
    gd = 1.0 - a / r1 * (1.0 - cE)
    # gradients of the basic scalars wrt s0; a = 1/(2/r0 - |v0|^2) chain-rules
    # to da/dr0vec = (2 a^2/r0^2) u0, da/dv0vec = 2 a^2 v0vec
    u0 = rvec0 / r0
    grad_r0 = numpy.concatenate([u0, numpy.zeros(3)])
    grad_a = numpy.concatenate([2.0 * a**2.0 / r0**2.0 * u0, 2.0 * a**2.0 * vvec0])
    grad_sigma0 = numpy.concatenate([vvec0, rvec0])
    # implicit dE through Kepler's equation
    # F = dE + (sigma0/sqa)(1-cE) - (1-r0/a) sE - t/a^{3/2}
    dFddE = r1 / a  # = 1 + (sigma0/sqa) sE - (1-r0/a) cE
    dFdsigma0 = (1.0 - cE) / sqa
    dFdr0 = sE / a
    dFda = (
        -sigma0 * (1.0 - cE) / (2.0 * a * sqa)
        - r0 * sE / a**2.0
        + 1.5 * t / (a**2.0 * sqa)
    )
    grad_dE = -(dFdsigma0 * grad_sigma0 + dFdr0 * grad_r0 + dFda * grad_a) / dFddE
    # r1 = a + (r0-a) cE + sigma0 sqa sE
    grad_r1 = (
        (1.0 - cE + sigma0 * sE / (2.0 * sqa)) * grad_a
        + cE * grad_r0
        + sqa * sE * grad_sigma0
        + (-(r0 - a) * sE + sigma0 * sqa * cE) * grad_dE
    )
    # f = 1 - (a/r0)(1-cE)
    grad_f = (
        -(1.0 - cE) / r0 * grad_a
        + a * (1.0 - cE) / r0**2.0 * grad_r0
        - a * sE / r0 * grad_dE
    )
    # g = t - a^{3/2} (dE - sE)
    grad_g = -1.5 * sqa * (dE - sE) * grad_a - a * sqa * (1.0 - cE) * grad_dE
    # fd = -(sqa sE)/(r1 r0)
    grad_fd = (
        -sE / (2.0 * sqa * r1 * r0) * grad_a
        + sqa * sE / (r1**2.0 * r0) * grad_r1
        + sqa * sE / (r1 * r0**2.0) * grad_r0
        - sqa * cE / (r1 * r0) * grad_dE
    )
    # gd = 1 - (a/r1)(1-cE)
    grad_gd = (
        -(1.0 - cE) / r1 * grad_a
        + a * (1.0 - cE) / r1**2.0 * grad_r1
        - a * sE / r1 * grad_dE
    )
    M = numpy.zeros((6, 6))
    M[:3, :] = numpy.outer(rvec0, grad_f) + numpy.outer(vvec0, grad_g)
    M[:3, :3] += f * numpy.eye(3)
    M[:3, 3:] += g * numpy.eye(3)
    M[3:, :] = numpy.outer(rvec0, grad_fd) + numpy.outer(vvec0, grad_gd)
    M[3:, :3] += fd * numpy.eye(3)
    M[3:, 3:] += gd * numpy.eye(3)
    return M


def _kepler_fg_state(s0, dE, t):
    """Analytic Kepler (mu=1) f,g propagation of the state s0 to time t; dE
    must solve Kepler's equation for (s0, t)."""
    rvec0 = numpy.array(s0[:3])
    vvec0 = numpy.array(s0[3:])
    r0 = numpy.sqrt(numpy.sum(rvec0**2.0))
    a = 1.0 / (2.0 / r0 - numpy.sum(vvec0**2.0))
    sqa = numpy.sqrt(a)
    sigma0 = numpy.sum(rvec0 * vvec0)
    cE, sE = numpy.cos(dE), numpy.sin(dE)
    r1 = a + (r0 - a) * cE + sigma0 * sqa * sE
    f = 1.0 - a / r0 * (1.0 - cE)
    g = t - a * sqa * (dE - sE)
    fd = -sqa / (r1 * r0) * sE
    gd = 1.0 - a / r1 * (1.0 - cE)
    return numpy.concatenate([f * rvec0 + g * vvec0, fd * rvec0 + gd * vvec0])


def _kepler_solve_dE(s0, t):
    """Solve Kepler's equation (dE formulation, mu=1) for dE at time t by
    Newton iteration from the mean-anomaly guess; dF/ddE = r(t)/a > 0 for an
    elliptic orbit, so F is strictly monotonic in dE and Newton is safe."""
    rvec0 = numpy.array(s0[:3])
    vvec0 = numpy.array(s0[3:])
    r0 = numpy.sqrt(numpy.sum(rvec0**2.0))
    a = 1.0 / (2.0 / r0 - numpy.sum(vvec0**2.0))
    sqa = numpy.sqrt(a)
    sigma0 = numpy.sum(rvec0 * vvec0)

    def F(dE):
        return (
            dE
            + sigma0 / sqa * (1.0 - numpy.cos(dE))
            - (1.0 - r0 / a) * numpy.sin(dE)
            - t / (a * sqa)
        )

    def dF(dE):
        return (a + (r0 - a) * numpy.cos(dE) + sigma0 * sqa * numpy.sin(dE)) / a

    dE = t / (a * sqa)  # mean-anomaly initial guess
    for _ in range(100):
        Fval = F(dE)
        if numpy.fabs(Fval) < 1e-14:
            break
        dE = dE - Fval / dF(dE)
    assert numpy.fabs(F(dE)) < 1e-12, "Kepler-equation Newton solve did not converge"
    return dE


def test_dxdv_3d_closed_form_stm_kepler():
    from galpy.orbit import Orbit
    from galpy.potential import KeplerPotential

    pot = KeplerPotential(amp=1.0)  # mu = 1, matching the reference above
    assert pot.hasC_dxdv3d, "KeplerPotential should advertise hasC_dxdv3d"
    # moderately eccentric, genuinely 3D (inclined) orbit
    ic = [1.0, 0.2, 0.8, 0.2, 0.1, 0.3]
    R, vR, vT, z, vz, phi = ic
    # the same cylindrical -> Cartesian map galpy uses (x along phi=0)
    s0 = numpy.array(
        [
            R * numpy.cos(phi),
            R * numpy.sin(phi),
            z,
            vR * numpy.cos(phi) - vT * numpy.sin(phi),
            vR * numpy.sin(phi) + vT * numpy.cos(phi),
            vz,
        ]
    )
    r0 = numpy.sqrt(numpy.sum(s0[:3] ** 2.0))
    a = 1.0 / (2.0 / r0 - numpy.sum(s0[3:] ** 2.0))
    assert a > 0.0, "test precondition: the orbit must be elliptic"
    # eccentricity from e^2 = 1 + 2 E L^2 = 1 - |L|^2/a (mu=1)
    Lvec = numpy.cross(s0[:3], s0[3:])
    ecc = numpy.sqrt(1.0 - numpy.sum(Lvec**2.0) / a)
    assert 0.2 < ecc < 0.8, (
        f"test precondition: moderately eccentric orbit expected, got e={ecc:g}"
    )
    # ~1.8 radial periods (T = 2 pi a^{3/2} ~ 4.4)
    times = numpy.linspace(0.0, 8.0, 17)
    dEs = [_kepler_solve_dE(s0, tk) for tk in times]
    # anchor the analytic reference: the f,g propagator must follow the same
    # orbit as a plain galpy integration (~2e-10 measured)
    o = Orbit(ic)
    o.integrate(times, pot, method="dop853_c")
    rect_orbit = _orbit_rect_3d(o, times)
    state_diff = numpy.amax(
        numpy.fabs(
            numpy.array(
                [_kepler_fg_state(s0, dEs[kk], tk) for kk, tk in enumerate(times)]
            )
            - rect_orbit
        )
    )
    assert state_diff < 1e-8, (
        f"analytic Kepler f,g propagation differs from the integrated orbit by "
        f"{state_diff:g}"
    )
    Mref = numpy.array([_kepler_fg_stm(s0, dEs[kk], tk) for kk, tk in enumerate(times)])
    # non-vacuity: the deviations genuinely evolve (Kepler shear grows secularly)
    assert numpy.amax(numpy.fabs(Mref[-1] - numpy.eye(6))) > 0.5
    for integrator in ["dopr54_c", "dop853_c"]:
        Mt = _integrate_stm_3d(pot, ic, times, integrator)
        maxdiff = numpy.amax(numpy.fabs(Mt - Mref))
        # measured ~1.5e-8 (dopr54_c) / ~1.4e-9 (dop853_c)
        assert maxdiff < 1e-7, (
            f"integrated 3D STM differs from the closed-form Kepler f,g STM by "
            f"{maxdiff:g} for integrator {integrator}"
        )
    return None


# ---- Conserved-integral left-relations. If C(x) is conserved along every
# orbit, C(x(t; x_0)) = C(x_0) identically in x_0; differentiating wrt x_0
# gives the exact STM relation
#   g(x(t))^T M(t) = g(x_0)^T ,   g = dC/dx (gradient wrt the CARTESIAN
# phase-space coordinates (x,y,z,vx,vy,vz)),
# i.e. the gradient of a conserved quantity along the orbit is a left
# "eigen-covector" of the STM. For an axisymmetric, time-independent potential
# both the energy and the z-angular momentum are conserved, with
#   E    = |v|^2/2 + Phi      -> g_E  = (dPhi/dx, dPhi/dy, dPhi/dz, vx, vy, vz)
#   L_z  = x vy - y vx        -> g_Lz = (vy, -vx, 0, -y, x, 0)
# (dL_z/dx = vy, dL_z/dy = -vx, dL_z/dvx = -y, dL_z/dvy = +x; dPhi/dx_i is
# MINUS the Cartesian acceleration). This constrains the STM ROWS against
# orbit-level quantities, complementary to det(M)=1/symplecticity (satisfied
# by any symmetric Hessian) and to the column-wise FD-of-flow check.
def test_dxdv_3d_conserved_integral_left_relations():
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1, normalize=True)
    assert pot.hasC_dxdv3d, "MiyamotoNagai should advertise hasC_dxdv3d"
    assert not pot.isNonAxi  # premise for L_z conservation
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 51)
    for integrator in ["dopr54_c", "dop853_c"]:
        Mt = _integrate_stm_3d(pot, ic, times, integrator)
        # non-vacuity: the deviations genuinely evolve (measured ~11)
        assert numpy.amax(numpy.fabs(Mt[-1] - numpy.eye(6))) > 1.0
        o = Orbit(ic)
        o.integrate(times, pot, method=integrator)
        rect = _orbit_rect_3d(o, times)

        def grad_E(kk):
            out = numpy.empty(6)
            # dPhi/d(x,y,z) = -acceleration
            out[:3] = -_cart_accel_3d(pot, rect[kk, :3], t=times[kk])
            out[3:] = rect[kk, 3:]
            return out

        def grad_Lz(kk):
            x, y = rect[kk, 0], rect[kk, 1]
            vx, vy = rect[kk, 3], rect[kk, 4]
            return numpy.array([vy, -vx, 0.0, -y, x, 0.0])

        for cname, gfun in [("E", grad_E), ("Lz", grad_Lz)]:
            g0 = gfun(0)
            maxerr = numpy.amax(
                numpy.fabs(
                    numpy.array(
                        [numpy.dot(gfun(kk), Mt[kk]) for kk in range(len(times))]
                    )
                    - g0
                )
            )
            # measured ~2e-10 (dopr54_c) / ~1e-11 (dop853_c)
            assert maxerr < 1e-8, (
                f"conserved-integral left-relation g^T(x(t)) M(t) = g^T(x_0) "
                f"violated at {maxerr:g} for {cname}, integrator {integrator}"
            )
    return None


def test_kuzmindisk_dxdv_3d_c_vs_python_offplane():
    # KuzminDiskPotential has a verified-correct full 3D C Hessian (hasC_dxdv3d=True)
    # but is deliberately excluded from the strict liouville3d_registry: its potential
    # ~ -amp/sqrt(R^2+(a+|z|)^2) is only C^0 across the disk plane, so d2Phi/dz2 and
    # d2Phi/dRdz are discontinuous at z=0. An orbit crossing z=0 makes the two adaptive
    # integrators legitimately diverge at the kink (NOT a Hessian error). This test
    # instead exercises the C 3D Hessian on an orbit that stays OFF the disk plane
    # (z>0 throughout), where the potential is smooth and the C 3D dxdv path must match
    # the pure-Python reference to high precision.
    from galpy.orbit import Orbit
    from galpy.potential import KuzminDiskPotential

    pot = KuzminDiskPotential(normalize=1.0, a=1.0)
    assert pot.hasC_dxdv3d, "KuzminDisk should advertise hasC_dxdv3d"
    # large z0, moving away from the plane -> the orbit never approaches z=0
    ic = [1.0, 0.1, 1.1, 2.0, 0.3, 0.2]
    times = numpy.linspace(0.0, 2.0, 101)
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dop853_c")
    assert numpy.amin(numpy.fabs(obase.z(times))) > 1.0, (
        "test precondition: the IC must keep the orbit well off the disk plane "
        "(away from the z=0 kink in d2Phi/dz2)"
    )
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dop853",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # off-plane the C-vs-Python dxdv agree to ~1e-11; 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"off-plane 3D C variational integration for KuzminDisk differs from the "
        f"pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_dehnenbar_dxdv_inside_rb_c_vs_python():
    # DehnenBarPotential's 3D C Hessian has a separate branch for r <= rb (the bar
    # break radius); the liouville3d_registry DehnenBar entry uses the shared IC at
    # R~1 (r > rb ~ 0.42), exercising only the r > rb branch. This test integrates a
    # deviation along an orbit that spends a substantial fraction of its time INSIDE
    # rb, exercising (and validating, against the pure-Python reference) the r <= rb
    # branch of each second derivative -- otherwise both untested.
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential

    pot = DehnenBarPotential(alpha=0.05)
    assert pot.hasC_dxdv3d, "DehnenBar should advertise hasC_dxdv3d"
    ic = [0.2, 0.05, 0.1, 0.08, 0.03, 0.2]  # r ~ 0.22 < rb -> starts inside the bar
    times = numpy.linspace(0.0, 2.0, 101)
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dop853_c")
    r = numpy.sqrt(obase.R(times) ** 2 + obase.z(times) ** 2)
    assert numpy.mean(r < pot._rb) > 0.1, (
        "test precondition: the orbit must spend time inside rb to exercise the "
        "r <= rb branch of the C Hessian"
    )
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, pot, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, pot, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # the bar orbit is mildly chaotic, so C-vs-Python agree to ~1e-6; 1e-5 is safe
    assert maxdiff < 1e-5, (
        f"inside-rb 3D C variational integration for DehnenBar differs from the "
        f"pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_spherical_dxdv_3d_c_vs_python_extra():
    # PseudoIsothermal, Einasto, and interpSpherical have verified-correct full 3D C
    # Hessians (hasC_dxdv3d=True) but are deliberately excluded from the strict
    # liouville3d_registry (see the note in tests/conftest.py): the spline-interpolated
    # potentials (Einasto, interpSpherical) and PseudoIsothermal's (1/r^2)*atan profile
    # hit accuracy floors in the registry's pure-Python odeint finite-difference-of-flow
    # check / its 1e-9 unit-deviation bridge tolerance -- not Hessian errors. This test
    # validates their C 3D variational Hessian directly: every C 3D-dxdv unit-deviation
    # column must match the pure-Python (dop853) variational integrator to high
    # precision (the genuine correctness criterion the registry would otherwise apply).
    import numpy

    from galpy.orbit import Orbit
    from galpy.potential import (
        EinastoPotential,
        HernquistPotential,
        PseudoIsothermalPotential,
        interpSphericalPotential,
    )

    pots = [
        PseudoIsothermalPotential(amp=1.0, a=1.1, normalize=True),
        EinastoPotential(amp=1.0, h=1.5, n=2.0, normalize=True),
        interpSphericalPotential(
            rforce=HernquistPotential(amp=1.0, a=1.3, normalize=True),
            rgrid=numpy.geomspace(0.01, 20.0, 401),
        ),
    ]
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    for pot in pots:
        pname = pot.__class__.__name__
        assert pot.hasC_dxdv3d, f"{pname} should advertise hasC_dxdv3d"
        maxdiff = 0.0
        for ii in range(6):
            oc = Orbit(ic)
            oc.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method="dopr54_c",
                rectIn=True,
                rectOut=True,
                rtol=1e-12,
                atol=1e-12,
            )
            op = Orbit(ic)
            op.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method="dop853",
                rectIn=True,
                rectOut=True,
                rtol=1e-12,
                atol=1e-12,
            )
            diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
            maxdiff = max(maxdiff, diff)
        # C-vs-Python 3D variational integration agrees to ~1e-6 or better for these
        # (spline-interpolated potentials are the loosest); 1e-5 leaves a safe margin.
        assert maxdiff < 1e-5, (
            f"3D C variational integration for {pname} differs from the pure-Python "
            f"reference by {maxdiff:g} (unit deviation)"
        )
    return None


def test_doubleexp_dxdv_3d_c_vs_python():
    # DoubleExponentialDiskPotential has a verified-correct full 3D C Hessian
    # (hasC_dxdv3d=True): its C R2deriv/z2deriv/Rzderiv use the same Ogata/Hankel
    # Bessel quadrature (J0/J1 nodes) as the C forces and the Python 2nd derivatives.
    # It is deliberately excluded from the strict liouville3d_registry because the
    # finite absolute accuracy of that quadrature puts the registry's
    # finite-difference-of-the-flow check (eps=1e-7 differencing of full nonlinear
    # orbits) right at its ~1.2e-4 floor, just over the 1e-4 bound -- NOT a Hessian
    # error. This test exercises the C 3D Hessian directly by comparing the C 3D
    # variational integration against the pure-Python reference, which the C path must
    # match to high precision (the two share no integrator code, only the analytic
    # Hessian, so agreement validates the C Hessian).
    from galpy.orbit import Orbit
    from galpy.potential import DoubleExponentialDiskPotential

    pot = DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.3, normalize=True)
    assert pot.hasC_dxdv3d, "DoubleExponentialDisk should advertise hasC_dxdv3d"
    # Fully 3D initial condition (R,vR,vT,z,vz,phi)
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dop853",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # the C-vs-Python dxdv agree to ~1e-9; 1e-6 leaves a wide margin
    assert maxdiff < 1e-6, (
        f"3D C variational integration for DoubleExponentialDisk differs from the "
        f"pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_doubleexp_dxdv_planar_c_vs_python():
    # DoubleExponentialDiskPotential also wires the PLANAR (2D, z=0) R2deriv in C
    # (hasC_dxdv=True): it is just the 3D R2deriv at z=0, so for an in-plane orbit
    # the C planar variational integration must match the pure-Python reference
    # (the two share only the analytic Hessian, not integrator code).
    from galpy.orbit import Orbit
    from galpy.potential import DoubleExponentialDiskPotential

    pot = DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.3, normalize=True)
    assert pot.hasC_dxdv, "DoubleExponentialDisk should advertise hasC_dxdv (planar)"
    ic = [1.0, 0.1, 1.1, 0.0]  # planar (R, vR, vT, phi)
    times = numpy.linspace(0.0, 5.0, 251)
    maxdiff = 0.0
    for dev in (numpy.array([1e-4, 0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 1e-4, 0.0])):
        oc = Orbit(ic)
        oc.integrate_dxdv(dev, times, pot, method="dopr54_c", rtol=1e-12, atol=1e-12)
        op = Orbit(ic)
        op.integrate_dxdv(dev, times, pot, method="dop853", rtol=1e-12, atol=1e-12)
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # planar C-vs-Python dxdv agree to ~1e-14; 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"planar C variational integration for DoubleExponentialDisk differs from the "
        f"pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_interprz_dxdv_3d():
    # interpRZPotential has a full 3D C Hessian (hasC_dxdv3d=True) when the
    # potential, both forces, AND the three 2nd derivatives are interpolated
    # with enable_c: like the forces, R2deriv/z2deriv/Rzderiv are each a
    # precomputed grid of exact (original-potential) values interpolated with a
    # 2D cubic B-spline in C (phi derivatives are identically zero --
    # axisymmetric). It is deliberately NOT in the strict liouville3d_registry
    # (the interpSpherical/Multipole precedent): every check involving it is
    # interpolation-limited, and its pure-Python reference path with enable_c
    # re-packs the full grids into C per evaluation, which is far too slow for
    # the registry sweep. This test validates the C 3D Hessian directly:
    # (1) C 3D variational integration on the interp potential must match the
    #     trusted pure-Python analytic-Hessian dxdv of the UNDERLYING
    #     MWPotential2014 to interpolation accuracy (~1e-4 measured; the two
    #     share NO code -- different potential representation, integrator, and
    #     Hessian -- so agreement pins the C Hessian values);
    # (2) Liouville det(M)=1 and symplecticity MtOmegaM=Omega of the 6x6 STM
    #     (necessary; holds at integrator accuracy for any symmetric K);
    # (3) finite-difference-of-the-flow: the dxdv column must match the FD of
    #     the integrated C flow (uses only the trusted C forces). dop853_c is
    #     excluded from this check only: the C2-continuous spline RHS breaks
    #     the order-8 error estimator, making the eps=1e-7 orbit differencing
    #     noisy (~1e-2) -- an integrator/FD-accuracy effect, not a Hessian
    #     error (dop853_c passes check (1) at 1.0e-4, identical to dopr54_c).
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, interpRZPotential
    from galpy.util import coords

    rzpot = interpRZPotential(
        RZPot=MWPotential2014,
        rgrid=(numpy.log(0.3), numpy.log(3.0), 151),
        zgrid=(0.0, 0.35, 151),
        logR=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        use_c=True,
        enable_c=True,
        zsym=True,
    )
    assert rzpot.hasC_dxdv3d, "interpRZPotential should advertise hasC_dxdv3d"
    # Negative control: without the interpolated 2nd derivatives there is no
    # 3D C Hessian (integrate_dxdv then falls back to Python, see
    # test_integrate_dxdv_3d_c_requires_full_hessian for the gate itself)
    rzpot_no2nd = interpRZPotential(
        RZPot=MWPotential2014,
        rgrid=(numpy.log(0.3), numpy.log(3.0), 11),
        zgrid=(0.0, 0.35, 11),
        logR=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        use_c=True,
        enable_c=True,
        zsym=True,
    )
    assert not rzpot_no2nd.hasC_dxdv3d, (
        "interpRZPotential w/o interpolated 2nd derivatives should not advertise hasC_dxdv3d"
    )
    # Generic, fully 3D initial condition (R,vR,vT,z,vz,phi); the orbit stays
    # within R in [0.98,1.31], |z| <= 0.077 -- well inside the grid
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    # (1) C dxdv on the interp potential vs the trusted pure-Python
    # analytic-Hessian dxdv on the underlying MWPotential2014
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, MWPotential2014, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        ref = op.getOrbit_dxdv()
        for integrator in ("dopr54_c", "dop853_c", "rk6_c"):
            oc = Orbit(ic)
            oc.integrate_dxdv(
                canonical[ii], times, rzpot, method=integrator,
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
            diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - ref))
            # measured: <= 1.0e-4 for each integrator (interpolation-limited);
            # 1e-3 leaves a safe margin while pinning any sign/factor error
            # in the C Hessian (those give O(1) differences)
            assert diff < 1e-3, (
                f"3D C variational integration for interpRZPotential differs from "
                f"the pure-Python reference on the underlying MWPotential2014 by "
                f"{diff:g} (unit deviation e_{ii}, integrator {integrator})"
            )
    # (2)+(3) det(M)=1, symplecticity, and FD-of-flow within the C integrators
    times = numpy.linspace(0.0, 3.0, 151)
    Omega = numpy.zeros((6, 6))
    Omega[:3, 3:] = numpy.eye(3)
    Omega[3:, :3] = -numpy.eye(3)
    for integrator in ("dopr54_c", "dop853_c", "rk6_c"):
        Mcols = []
        for ii in range(6):
            o = Orbit(ic)
            o.integrate_dxdv(
                canonical[ii], times, rzpot, method=integrator,
                rectIn=True, rectOut=True, rtol=1e-10, atol=1e-10,
            )  # fmt: skip
            Mcols.append(o.getOrbit_dxdv()[-1, :])
        M = numpy.array(Mcols).T
        detM = numpy.linalg.det(M)
        # measured: ~1.5e-7 (adaptive) / ~4e-12 (fixed-step rk6_c)
        assert numpy.fabs(detM - 1.0) < 1e-6, (
            f"3D Liouville det(M)={detM:g} differs from 1 for interpRZPotential, "
            f"integrator {integrator}"
        )
        symperr = numpy.amax(numpy.fabs(M.T @ Omega @ M - Omega))
        assert symperr < 1e-6, (
            f"3D symplecticity ||M^T Omega M - Omega||={symperr:g} too large for "
            f"interpRZPotential, integrator {integrator}"
        )
        if integrator == "dop853_c":
            continue  # FD-of-flow excluded for dop853_c, see docstring
        # (3) finite-difference of the flow (uses only the trusted C forces)
        eps = 1e-7
        obase = Orbit(ic)
        obase.integrate(times, rzpot, method=integrator, rtol=1e-12, atol=1e-12)
        base = _orbit_rect_3d(obase, times)
        for ii in [0, 2, 4]:  # x, z, vy perturbations
            p = base[0].copy()
            p[ii] += eps
            Rp, phip, Zp = coords.rect_to_cyl(p[0], p[1], p[2])
            vRp, vTp, vzp = coords.rect_to_cyl_vec(p[3], p[4], p[5], p[0], p[1], p[2])
            opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
            opert.integrate(times, rzpot, method=integrator, rtol=1e-12, atol=1e-12)
            fd = (_orbit_rect_3d(opert, times) - base) / eps
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii], times, rzpot, method=integrator,
                rectIn=True, rectOut=True, rtol=1e-10, atol=1e-10,
            )  # fmt: skip
            fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
            # measured: <= 6.5e-4 (interpolation-limited: the interpolated
            # exact-Hessian K differs from the exact Jacobian of the
            # interpolated forces at spline accuracy)
            assert fderr < 2e-3, (
                f"3D FD-of-flow for e_{ii} differs from the dxdv column by "
                f"{fderr:g} for interpRZPotential, integrator {integrator}"
            )
    return None


def test_kuzminlike_dxdv_planar_c_vs_python():
    # Regression test for the KuzminLikeWrapperPotential C planar dxdv path:
    # its d2xi/dR2 helper used pow(R^2+(a+sqrt(z^2+b^2))^2, 3.0) = xi^6 where
    # the correct denominator is xi^3 (exponent 1.5), making the C planar
    # variational integration wrong by O(1) for unit deviations (maxdiff ~0.8
    # over the orbit below) while leaving the forces -- and hence ordinary
    # orbit integration -- untouched. Fixed together with the 3D Hessian; the
    # C planar dxdv must now match the trusted pure-Python reference (the two
    # share only the analytic chain-rule Hessian, not integrator code).
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential, KuzminLikeWrapperPotential

    pot = KuzminLikeWrapperPotential(
        pot=HernquistPotential(amp=1.0, a=1.3, normalize=True), a=1.1, b=0.3
    )
    assert pot.hasC_dxdv, "KuzminLikeWrapper should advertise hasC_dxdv (planar)"
    ic = [1.0, 0.1, 1.1, 0.0]  # planar (R, vR, vT, phi)
    times = numpy.linspace(0.0, 5.0, 251)
    ptp = pot.toPlanar()
    canonical = numpy.eye(4)
    maxdiff = 0.0
    for ii in [0, 2]:  # e_x and e_vx unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, ptp, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, ptp, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # planar C-vs-Python dxdv agree to ~1e-11; 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"planar C variational integration for KuzminLikeWrapper differs from the "
        f"pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_kuzminlike_dxdv_3d_c_vs_miyamotonagai():
    # Physics-law cross-check of the KuzminLikeWrapper 3D C Hessian against a
    # completely INDEPENDENT C implementation: applying the Kuzmin-like
    # substitution to a KeplerPotential gives exactly the MiyamotoNagaiPotential
    # (for b != 0), whose full 3D C Hessian is implemented and validated
    # separately. The two C variational integrations share no Hessian code (the
    # wrapper chain-rules calcRforce/calcR2deriv of Kepler through xi; MN uses
    # its own closed-form second derivatives), so machine-precision agreement
    # (~1e-15) pins the wrapper's chain-rule Hessian values absolutely.
    from galpy.orbit import Orbit
    from galpy.potential import (
        KeplerPotential,
        KuzminLikeWrapperPotential,
        MiyamotoNagaiPotential,
    )

    kp = KeplerPotential(normalize=1.0)
    kwp = KuzminLikeWrapperPotential(pot=kp, a=1.3, b=0.2)
    mn = MiyamotoNagaiPotential(amp=kp._amp, a=1.3, b=0.2)
    assert kwp.hasC_dxdv3d, "KuzminLikeWrapper should advertise hasC_dxdv3d"
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        o1 = Orbit(ic)
        o1.integrate_dxdv(
            canonical[ii], times, kwp, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        o2 = Orbit(ic)
        o2.integrate_dxdv(
            canonical[ii], times, mn, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(o1.getOrbit_dxdv() - o2.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # identical flows evaluated by two independent C Hessians: ~1e-15; 1e-10
    # leaves room for floating-point evaluation-order differences only
    assert maxdiff < 1e-10, (
        f"3D C variational integration for KuzminLikeWrapper(Kepler) differs from "
        f"the independent MiyamotoNagai C Hessian by {maxdiff:g} (unit deviation)"
    )
    return None


# ---- "Staeckel-approximation" wrappers (OblateStaeckelWrapperPotential and
# CylindricallySeparablePotentialWrapper): dedicated C-vs-Python parity tests
# for the newly enabled planar (hasC_dxdv) and 3D (hasC_dxdv3d) C variational
# paths, complementing the full liouville3d-registry battery (det(M),
# symplecticity, flow-direction, FD-of-flow, 2D bridge, and the registry-wide
# C-vs-Python check at its global 1e-6 tolerance) with tight tolerances. The C
# Hessians are direct transcriptions of the trusted Python
# _R2deriv/_z2deriv/_Rzderiv, so C and Python share only the analytic
# formulas, not integrator code.
def test_oblatestaeckelwrapper_dxdv_planar_c_vs_python():
    # First-time enablement of the planar C variational path for
    # OblateStaeckelWrapperPotential: the C planar R2deriv (the v=pi/2
    # simplification of the full chain-rule R2deriv, with the wrapped
    # potential entering through its in-plane planarRforce/planarR2deriv only)
    # must match the trusted pure-Python reference, which evaluates the full
    # _R2deriv at z=0.
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential, OblateStaeckelWrapperPotential

    pot = OblateStaeckelWrapperPotential(
        pot=MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.3, normalize=True),
        delta=0.45,
        u0=1.15,
    )
    assert pot.hasC_dxdv, "OblateStaeckelWrapper should advertise hasC_dxdv (planar)"
    ic = [1.0, 0.1, 1.1, 0.0]  # planar (R, vR, vT, phi)
    times = numpy.linspace(0.0, 5.0, 251)
    ptp = pot.toPlanar()
    canonical = numpy.eye(4)
    maxdiff = 0.0
    for ii in [0, 2]:  # e_x and e_vx unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, ptp, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, ptp, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # measured: ~3e-11 (unit deviations); 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"planar C variational integration for OblateStaeckelWrapper differs from "
        f"the pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_oblatestaeckelwrapper_dxdv_3d_c_vs_kuzminkutuzov():
    # Physics-law cross-check of the OblateStaeckelWrapper 3D C Hessian against
    # a completely INDEPENDENT C implementation: a potential that is already an
    # oblate Staeckel potential with the same focal length is reconstructed
    # EXACTLY by the wrapper (for ANY u0): with Phi = (Ut(u)-Vt(v))/prefac,
    # U(u) = cosh^2 u Phi(u,pi/2) = Ut(u) - Vt(pi/2) and V(v) = refpot
    # - prefac(u0,v) Phi(u0,v) = Vt(v) - Vt(pi/2), so (U-V)/prefac = Phi.
    # KuzminKutuzovStaeckelPotential (Delta=1) has its own closed-form C
    # Hessian, validated separately in the liouville3d registry; the two C
    # variational integrations share no Hessian code (the wrapper chain-rules
    # the wrapped forces/2nd derivatives along the reference curves through
    # the (u,v) transform), so tight agreement pins the wrapper's chain-rule
    # Hessian values absolutely.
    from galpy.orbit import Orbit
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        OblateStaeckelWrapperPotential,
    )

    kk = KuzminKutuzovStaeckelPotential(amp=1.0, ac=5.0, Delta=1.0, normalize=True)
    wkk = OblateStaeckelWrapperPotential(pot=kk, delta=1.0, u0=1.3)
    assert wkk.hasC_dxdv3d, "OblateStaeckelWrapper should advertise hasC_dxdv3d"
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        o1 = Orbit(ic)
        o1.integrate_dxdv(
            canonical[ii], times, wkk, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        o2 = Orbit(ic)
        o2.integrate_dxdv(
            canonical[ii], times, kk, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(o1.getOrbit_dxdv() - o2.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # identical flows evaluated by two independent C Hessians; measured ~2e-11
    # (the wrapper's U/V go through acosh/uv round trips, so the two forces
    # differ at machine precision and the variational difference grows to
    # ~1e-11 over the orbit); 1e-9 leaves an order of magnitude of margin
    assert maxdiff < 1e-9, (
        f"3D C variational integration for OblateStaeckelWrapper(KuzminKutuzov) "
        f"differs from the independent KuzminKutuzov C Hessian by {maxdiff:g} "
        f"(unit deviation)"
    )
    return None


def test_cylsepwrapper_dxdv_planar_c_vs_python():
    # First-time enablement of the planar C variational path for
    # CylindricallySeparablePotentialWrapper: the C planar R2deriv simply
    # aggregates the wrapped potential's in-plane planarR2deriv (separability:
    # Phi_RR(R,z) = Phi_w,RR(R,0)) and must match the trusted pure-Python
    # reference.
    from galpy.orbit import Orbit
    from galpy.potential import (
        CylindricallySeparablePotentialWrapper,
        MiyamotoNagaiPotential,
    )

    pot = CylindricallySeparablePotentialWrapper(
        pot=MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.3, normalize=True), Rp=1.0
    )
    assert pot.hasC_dxdv, (
        "CylindricallySeparableWrapper should advertise hasC_dxdv (planar)"
    )
    ic = [1.0, 0.1, 1.1, 0.0]  # planar (R, vR, vT, phi)
    times = numpy.linspace(0.0, 5.0, 251)
    ptp = pot.toPlanar()
    canonical = numpy.eye(4)
    maxdiff = 0.0
    for ii in [0, 2]:  # e_x and e_vx unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, ptp, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, ptp, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # measured: ~3e-11 (unit deviations); 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"planar C variational integration for CylindricallySeparableWrapper "
        f"differs from the pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_cylsepwrapper_dxdv_3d_c_vs_python_tight():
    # Tight-tolerance 3D C-vs-Python parity for
    # CylindricallySeparablePotentialWrapper (the registry-wide
    # test_dxdv_3d_c_vs_python runs the same comparison at its global 1e-6
    # tolerance): the C R2deriv/z2deriv are direct transcriptions of the
    # Python _R2deriv/_z2deriv (the wrapped potential's own second derivatives
    # at (R,0) and (Rp,z)) with Rzderiv = 0 identically, so the two
    # integrations differ only by integrator implementation.
    from galpy.orbit import Orbit
    from galpy.potential import (
        CylindricallySeparablePotentialWrapper,
        MiyamotoNagaiPotential,
    )

    pot = CylindricallySeparablePotentialWrapper(
        pot=MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.3, normalize=True), Rp=1.0
    )
    assert pot.hasC_dxdv3d, "CylindricallySeparableWrapper should advertise hasC_dxdv3d"
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, pot, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, pot, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    # measured: ~5e-11 (unit deviations); 1e-9 leaves an order of magnitude
    assert maxdiff < 1e-9, (
        f"3D C variational integration for CylindricallySeparableWrapper differs "
        f"from the pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_integrate_dxdv_3d_staeckelwrappers_require_wrapped_hessian():
    # Both "Staeckel-approximation" wrappers advertise hasC_dxdv3d=True
    # unconditionally, but their C Hessians chain-rule the WRAPPED potential's
    # C forces/second derivatives, so when the wrapped potential lacks the 3D
    # C Hessian the C 3D variational path would silently aggregate 0 for the
    # unset derivatives (NULL-safe aggregators) and propagate a wrong
    # deviation. _check_c must recurse into the wrapped potential
    # (parentWrapperPotential branch) and integrate_dxdv must warn and fall
    # back to the pure-Python integrator. As in
    # test_integrate_dxdv_3d_c_requires_full_hessian, the no-3D-C-Hessian
    # wrapped potential is synthesized by forcing hasC_dxdv3d=False.
    from galpy.orbit import Orbit
    from galpy.potential import (
        CylindricallySeparablePotentialWrapper,
        MiyamotoNagaiPotential,
        OblateStaeckelWrapperPotential,
    )

    for wrap in [
        lambda p: OblateStaeckelWrapperPotential(pot=p, delta=0.45, u0=1.15),
        lambda p: CylindricallySeparablePotentialWrapper(pot=p, Rp=1.0),
    ]:
        mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.3)
        mn.hasC_dxdv3d = False  # force the wrapped-pot-without-3D-C-Hessian case
        pot = wrap(mn)
        pname = pot.__class__.__name__
        assert pot.hasC_dxdv3d, (
            f"test precondition: {pname} itself advertises hasC_dxdv3d"
        )
        assert not _check_c(pot, dxdv3d=True), (
            f"_check_c(dxdv3d) must recurse into the wrapped potential of "
            f"{pname} and report False"
        )
        ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
        times = numpy.linspace(0.0, 2.0, 101)
        dev = [1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]
        o_c = Orbit(ic)
        with pytest.warns(galpyWarning):
            o_c.integrate_dxdv(
                dev, times, pot, method="dopr54_c",
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
        o_py = Orbit(ic)
        o_py.integrate_dxdv(
            dev, times, pot, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
        dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
        assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-9, (
            f"3D integrate_dxdv did not fall back to the correct integrator for "
            f"{pname} when its wrapped potential lacks the full 3D C Hessian"
        )
    return None


def _check_dxdv_3d_c(
    pot,
    name,
    integrators=("dop853_c", "dopr54_c", "rk6_c"),
    det_tol=1e-7,
    symp_tol=1e-6,
    fd_tol=1e-4,
):
    # Shared 3D variational (dxdv) validation for the harmonic-expansion
    # potentials (SCF / MultipoleExpansion), exercised through the C
    # integrators only (their pure-Python reference is impractically slow --
    # SCF's numerical 2nd derivatives need many Python force evaluations per
    # RHS step). Checks, per integrator: (1) Liouville det(M)=1, (2)
    # symplecticity MᵀΩM=Ω in canonical Cartesian variables (necessary; pins
    # K's symmetry), and -- the part that actually pins the K VALUES against
    # the C forces -- (3) finite-difference-of-the-flow. (det_tol/symp_tol are
    # looser for the spline-interpolated MultipoleExpansion than for the
    # analytic-radial-basis SCF.)
    from galpy.orbit import Orbit
    from galpy.util import coords

    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 3.0, 151)
    canonical = numpy.eye(6)
    Omega = numpy.zeros((6, 6))
    Omega[:3, 3:] = numpy.eye(3)
    Omega[3:, :3] = -numpy.eye(3)
    for integrator in integrators:
        Mcols = []
        for ii in range(6):
            o = Orbit(ic)
            o.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method=integrator,
                rectIn=True,
                rectOut=True,
                rtol=1e-10,
                atol=1e-10,
            )
            Mcols.append(o.getOrbit_dxdv()[-1, :])
        M = numpy.array(Mcols).T
        this_det_tol = max(det_tol, 1e-6) if integrator == "rk6_c" else det_tol
        detM = numpy.linalg.det(M)
        assert numpy.fabs(detM - 1.0) < this_det_tol, (
            f"3D Liouville det(M)={detM:g} differs from 1 for {name}, {integrator}"
        )
        symperr = numpy.amax(numpy.fabs(M.T @ Omega @ M - Omega))
        assert symperr < symp_tol, (
            f"3D symplecticity err={symperr:g} too large for {name}, {integrator}"
        )
        # finite-difference of the flow (validates the K values vs the C forces)
        eps = 1e-7
        obase = Orbit(ic)
        obase.integrate(times, pot, method=integrator)
        base = _orbit_rect_3d(obase, times)
        for ii in [0, 2, 4]:  # x, z, vy perturbations
            p = base[0].copy()
            p[ii] += eps
            Rp, phip, Zp = coords.rect_to_cyl(p[0], p[1], p[2])
            vRp, vTp, vzp = coords.rect_to_cyl_vec(p[3], p[4], p[5], p[0], p[1], p[2])
            opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
            opert.integrate(times, pot, method=integrator)
            fd = (_orbit_rect_3d(opert, times) - base) / eps
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method=integrator,
                rectIn=True,
                rectOut=True,
                rtol=1e-10,
                atol=1e-10,
            )
            fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
            assert fderr < fd_tol, (
                f"3D FD-of-flow for e_{ii} differs from the dxdv column by "
                f"{fderr:g} for {name}, {integrator}"
            )
    return None


def test_scf_dxdv_3d():
    # SCFPotential has a full 3D C Hessian (hasC_dxdv3d=True) via the
    # Hernquist-Ostriker spherical-harmonic expansion. Validate the 3D
    # variational integration for an axisymmetric and a genuinely
    # non-axisymmetric (m=2 -> nonzero zphideriv) instance. SCF is NOT in the
    # strict liouville3d_registry because its pure-Python integrators are far
    # too slow for the per-potential registry sweep; this dedicated test uses
    # the C integrators only.
    from galpy.potential import SCFPotential, scf_compute_coeffs_spherical

    def _hern(R, z=0.0, phi=0.0):
        r = numpy.sqrt(R**2 + z**2)
        return 1.0 / (2.0 * numpy.pi) / (r * (1.0 + r) ** 3)

    Acos_s, _ = scf_compute_coeffs_spherical(_hern, 5, a=1.0)
    N = Acos_s.shape[0]
    # axisymmetric: spherical monopole only
    scf_axi = SCFPotential(Acos=Acos_s, a=1.0, normalize=True)
    assert scf_axi.hasC_dxdv3d, "SCFPotential should advertise hasC_dxdv3d"
    # non-axisymmetric: inject an l=2,m=2 term (exercises zphideriv)
    Acos = numpy.zeros((N, 3, 3))
    Asin = numpy.zeros((N, 3, 3))
    Acos[:, 0, 0] = Acos_s[:, 0, 0]
    Acos[0, 2, 2] = 0.05 * Acos_s[0, 0, 0]
    Asin[0, 2, 2] = 0.03 * Acos_s[0, 0, 0]
    scf_tri = SCFPotential(Acos=Acos, Asin=Asin, a=1.0, normalize=True)
    assert scf_tri.isNonAxi, "injected-m2 SCF should be non-axisymmetric"
    _check_dxdv_3d_c(scf_axi, "SCFPotential (axi)")
    _check_dxdv_3d_c(scf_tri, "SCFPotential (m=2)")
    return None


def test_multipole_dxdv_3d():
    # MultipoleExpansionPotential has a full 3D C Hessian (hasC_dxdv3d=True) via
    # the same spherical-harmonic machinery as SCF (shared transform). Validate
    # a spherical and a genuinely triaxial (nonzero zphideriv) instance. Like
    # SCF it is excluded from the strict registry (slow Python integrators; its
    # radial functions are spline-interpolated, so the FD-of-flow sits at
    # spline accuracy, not analytic accuracy).
    from galpy.potential import (
        HernquistPotential,
        MultipoleExpansionPotential,
        TriaxialNFWPotential,
    )

    rgrid = numpy.geomspace(1e-2, 30.0, 201)
    mep_sph = MultipoleExpansionPotential.from_density(
        dens=HernquistPotential(amp=2.0, a=1.0), symmetry="spherical", rgrid=rgrid
    )
    assert mep_sph.hasC_dxdv3d, "MultipoleExpansion should advertise hasC_dxdv3d"
    mep_tri = MultipoleExpansionPotential.from_density(
        dens=TriaxialNFWPotential(amp=3.0, a=2.0, b=0.8, c=0.6),
        symmetry="triaxial",
        L=6,
        rgrid=rgrid,
    )
    assert mep_tri.isNonAxi, "triaxial MultipoleExpansion should be non-axisymmetric"
    # spline-interpolated radial functions -> looser det/symp; use the adaptive
    # C integrators (fixed-step rk*_c is dominated by the spline error here).
    _check_dxdv_3d_c(
        mep_sph,
        "MultipoleExpansion (spherical)",
        integrators=("dop853_c", "dopr54_c"),
        det_tol=5e-6,
        symp_tol=5e-6,
    )
    _check_dxdv_3d_c(
        mep_tri,
        "MultipoleExpansion (triaxial)",
        integrators=("dop853_c", "dopr54_c"),
        det_tol=5e-6,
        symp_tol=5e-6,
    )
    # Time-dependent (rotating, weakly perturbed) multipole: exercises the
    # time-dependent radial-coefficient path of the C 3D Hessian (the Nt>0
    # branch of compute_multipole_hessian_cyl). Still a Hamiltonian flow, so
    # det(M)=1 / symplecticity hold.
    omega = 0.8
    hp = HernquistPotential(amp=2.0, a=1.0)
    mep_tdep = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            hp.dens(R, z, use_physical=False)
            * (1.0 + 0.05 * numpy.cos(2.0 * (phi - omega * t)))
        ),
        L=4,
        rgrid=rgrid,
        tgrid=numpy.linspace(0.0, 5.0, 41),
    )
    assert mep_tdep.isNonAxi, "time-dependent multipole should be non-axisymmetric"
    # det(M)=1 / symplecticity still hold to ~1e-8 (the Hessian is correct); the
    # FD-of-flow is looser because the coarse (rgrid x tgrid) interpolation of the
    # time-dependent radial coefficients is less accurate than the static splines.
    _check_dxdv_3d_c(
        mep_tdep,
        "MultipoleExpansion (time-dependent)",
        integrators=("dop853_c", "dopr54_c"),
        det_tol=5e-6,
        symp_tol=5e-6,
        fd_tol=1e-3,
    )
    return None


def test_disk_composite_dxdv_3d():
    # The DiskSCF / DiskMultipole (Kuijken-Dubinski) composites have a full 3D C
    # Hessian (hasC_dxdv3d=True) once BOTH the analytic [Sigma_i(r), Hz_i(z)]
    # disk pairs (their R2/z2/Rz deriv are in C; phi-derivs are identically
    # zero) AND the expansion sub-potential (SCF / MultipoleExpansion) have the
    # Hessian in C. A SMOOTH sech2 vertical profile is used: the exp |z| profile
    # is only C^0 across the disk plane (like KuzminDisk), so its FD-of-flow
    # sits at the z=0 kink, not at analytic accuracy (det(M)=1/symplecticity
    # still pass for it -- the Hessian itself is correct).
    from galpy.potential import DiskMultipoleExpansionPotential, DiskSCFPotential

    def _sphdens(R, z):
        return numpy.exp(-numpy.sqrt(R**2 + z**2)) / (4.0 * numpy.pi)

    Sigma = {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0}
    hz = {"type": "sech2", "h": 1.0 / 27.0}
    dscf = DiskSCFPotential(
        dens=_sphdens, Sigma=Sigma, hz=hz, a=1.0, N=4, L=4, normalize=True
    )
    assert dscf.hasC_dxdv3d, "DiskSCFPotential should advertise hasC_dxdv3d"
    # analytic SCF + analytic disk pairs -> tight
    _check_dxdv_3d_c(dscf, "DiskSCFPotential", det_tol=1e-7, symp_tol=1e-6)
    dmep = DiskMultipoleExpansionPotential(
        dens=_sphdens,
        Sigma=Sigma,
        hz=hz,
        L=4,
        rgrid=numpy.geomspace(1e-2, 30.0, 201),
        normalize=True,
    )
    assert dmep.hasC_dxdv3d, "DiskMultipoleExpansion should advertise hasC_dxdv3d"
    # spline-interpolated multipole part -> looser det/symp, adaptive integrators
    _check_dxdv_3d_c(
        dmep,
        "DiskMultipoleExpansionPotential",
        integrators=("dop853_c", "dopr54_c"),
        det_tol=5e-6,
        symp_tol=5e-6,
    )
    return None


############ 3D variational equations with DISSIPATIVE forces ################
# For a velocity-dependent force the variational Jacobian is the general
# J = [[0,I],[K + dF/dx, dF/dv]]: the dissipative position block is NOT the
# symmetric -grad grad Phi and there is a nonzero velocity block. The flow is
# non-conservative -- det M(t) = exp(int tr(dF/dv) dt') != 1 and symplecticity
# fails BY CONSTRUCTION -- so dissipative forces are deliberately NOT in the
# conftest liouville3d_registry (det(M)=1/symplecticity battery; verified by
# test_dissipative_excluded_from_liouville3d_registry below) and are instead
# validated at the orbit level only (galpy's convention: C code is tested
# through regular galpy orbit usage) by (a) the finite-difference-of-the-flow
# STM test (uses only the trusted forces) and (b) the quantitative
# phase-volume law det M(t) = exp(int tr(dF/dv) dt'), with the trace computed
# by central finite differences of the pure-Python forces -- an independent
# code path from the C-integrated STM it is compared against.


def _python_fd_trace_dFdv(dissip, base_rect, times, h=1e-6):
    """tr(dF/dv) of the dissipative force along the orbit, by central finite
    differences of the PYTHON force evaluators (evaluateRforces /
    evaluatephitorques / evaluatezforces with v=..., converted to Cartesian):
    a code path fully independent of the C variational machinery whose STM
    the phase-volume-law tests validate. Conservative forces do not depend on
    v, so only the dissipative force needs to be differentiated."""
    from galpy.potential import (
        evaluatephitorques,
        evaluateRforces,
        evaluatezforces,
    )

    def cart_force(q, t):
        x, y, z, vx, vy, vz = q
        R = numpy.sqrt(x**2 + y**2)
        cosphi, sinphi = x / R, y / R
        phi = numpy.arctan2(y, x)
        vR = vx * cosphi + vy * sinphi
        vT = -vx * sinphi + vy * cosphi
        FR = evaluateRforces(dissip, R, z, phi=phi, t=t, v=[vR, vT, vz])
        Fp = evaluatephitorques(dissip, R, z, phi=phi, t=t, v=[vR, vT, vz])
        Fz = evaluatezforces(dissip, R, z, phi=phi, t=t, v=[vR, vT, vz])
        return numpy.array(
            [cosphi * FR - sinphi / R * Fp, sinphi * FR + cosphi / R * Fp, Fz]
        )

    tr = numpy.empty(len(times))
    for kk in range(len(times)):
        s = 0.0
        for ii in range(3):
            qp = base_rect[kk].copy()
            qp[3 + ii] += h
            qm = base_rect[kk].copy()
            qm[3 + ii] -= h
            s += (cart_force(qp, times[kk])[ii] - cart_force(qm, times[kk])[ii]) / (
                2.0 * h
            )
        tr[kk] = s
    return tr


def _chandrasekhar_dxdv_setup(nt=501):
    """Shared setup for the dissipative orbit-level variational tests: a
    decaying orbit (r: ~1.0 -> ~0.75 over t=0..5) in MWPotential2014 +
    Chandrasekhar friction with rhm=0, i.e. the v-dependent Coulomb-log branch
    is exercised along the entire orbit."""
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        MWPotential2014,
    )

    cdf = ChandrasekharDynamicalFrictionForce(
        GMs=0.008, rhm=0.0, dens=MWPotential2014, maxr=10.0
    )
    pot = MWPotential2014 + cdf
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, nt)
    return cdf, pot, ic, times


def _orbit_rect_columns_3d(o, ts):
    return numpy.array([o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)]).T


def test_chandrasekhar_dxdv_fd_of_flow():
    # FD-of-the-flow STM test for the dissipative 3D variational equations:
    # every column i of the C-integrated deviation (seeded with the canonical
    # e_i) must match (x(t; x0+eps e_i) - x(t; x0))/eps built from plain orbit
    # integrations, which use only the separately-validated forces -- the same
    # check (and tolerance) as the conservative test_liouville_3d battery, for
    # a decaying orbit in MWPotential2014 + Chandrasekhar friction.
    from galpy.orbit import Orbit
    from galpy.util import coords

    cdf, pot, ic, times = _chandrasekhar_dxdv_setup()
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    # friction must be doing real work: the orbit decays
    rstart = numpy.sqrt(numpy.sum(base_rect[0, :3] ** 2))
    rend = numpy.sqrt(numpy.sum(base_rect[-1, :3] ** 2))
    assert rend < 0.85 * rstart, (
        f"Chandrasekhar test orbit does not decay (r: {rstart:g} -> {rend:g}); "
        "the dissipative variational test would be vacuous"
    )
    canonical = numpy.eye(6)
    eps = 1e-7
    for ii in range(6):
        pert = base_rect[0].copy()
        pert[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method="dopr54_c")
        fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
        assert fderr < 1e-4, (
            f"Dissipative 3D FD-of-flow for e_{ii} differs from the dxdv "
            f"column by {fderr:g} (MWPotential2014 + Chandrasekhar friction)"
        )
    return None


def test_chandrasekhar_dxdv_phase_volume_law():
    # Quantitative non-conservative test: for the general variational Jacobian
    # J = [[0,I],[K + dF/dx, dF/dv]], Abel/Jacobi-Liouville gives
    #   det M(t) = exp( int_0^t tr J dt' ) = exp( int_0^t tr(dF/dv) dt' )
    # (only the velocity block contributes to the trace). Build the full 6x6
    # STM M(t) from the 6 canonical deviation integrations and compare det M(t)
    # against the trace integrated along the orbit (trapezoidal rule on the
    # fine output grid), with tr(dF/dv) computed by central finite differences
    # of the pure-Python forces -- a code path independent of the C variational
    # machinery that produced the STM. Friction contracts phase volume, so
    # additionally det M < 1.
    from galpy.orbit import Orbit

    # the fine grid keeps the trapezoidal error of the trace integral (the
    # quadrature is the limiting factor, O(h^2)) below the 1e-5 tolerance
    cdf, pot, ic, times = _chandrasekhar_dxdv_setup(nt=1001)
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    canonical = numpy.eye(6)
    Mcols = []
    for ii in range(6):
        o = Orbit(ic)
        o.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        Mcols.append(o.getOrbit_dxdv())
    Mt = numpy.array(Mcols).transpose(1, 2, 0)  # (nt,6,6), columns = e_i images
    detM = numpy.array([numpy.linalg.det(Mt[kk]) for kk in range(len(times))])
    # tr(dF/dv) along the orbit by central FD of the PYTHON forces (the
    # friction is the only dissipative component; conservative forces do not
    # depend on v and contribute exactly 0)
    trJv = _python_fd_trace_dFdv(cdf, base_rect, times)
    integral = numpy.concatenate(
        ([0.0], numpy.cumsum(0.5 * (trJv[1:] + trJv[:-1]) * numpy.diff(times)))
    )
    pred = numpy.exp(integral)
    relerr = numpy.amax(numpy.fabs(detM - pred) / pred)
    assert relerr < 1e-5, (
        f"Dissipative phase-volume law det M(t) = exp(int tr(dF/dv) dt') "
        f"violated at relative level {relerr:g}"
    )
    # friction contracts phase volume: det M decays monotonically below 1
    assert detM[-1] < 1.0, f"det M(t_end)={detM[-1]:g} should be < 1 with friction"
    assert detM[-1] < 0.9, (
        f"det M(t_end)={detM[-1]:g}: phase-space contraction too weak for the "
        "phase-volume-law test to be meaningful"
    )
    return None


# NonInertialFrameForce 3D variational equations: the frame force
# F = -2 Omega x (v+v0) - Omega x (Omega x [r+x0]) - Omegadot x [r+x0] - a0(t)
# is LINEAR in (r, v), so its rectangular Jacobian blocks are exact closed
# forms: dF/dv = -2 [Omega]_x (antisymmetric -> tr(dF/dv)=0: phase-volume
# PRESERVING, det M(t)=1, unlike friction) and
# dF/dx = |Omega|^2 I - Omega Omega^T - [Omegadot]_x; translation terms
# contribute zero. Validated below at the orbit level only (galpy's
# convention: C code is tested through regular galpy orbit usage) by (a) the
# FD-of-the-flow STM test, (b) the det M(t)=1 phase-volume law (which needs
# no trace computation: tr(dF/dv)=0 exactly), and (c) an exact cross-check
# of the rotating-frame STM against the inertial-frame STM transformed by
# the frame map (Plummer sphere, constant vector Omega).


# The C rectangular friction Jacobian
# (ChandrasekharDynamicalFrictionForce.c::...RectDissipativeForceJacobian)
# mirrors the branch structure of the force it differentiates: three Coulomb-
# logarithm configurations (constant lnLambda; the rhm-based variable branch
# GMs/v^2 < rhm; the GMvs-based variable branch), the r < minr zero gate, and
# the clamped sigma_r spline outside the interpolation grid. The main
# FD-of-flow test above runs the GMvs-based branch (rhm=0) inside the grid;
# the configurations here select each of the other branches through the
# regular constructor options and validate with the same finite-difference-of-
# the-flow check (ground truth built from plain orbit integrations, which use
# only the separately-validated forces), each with a non-vacuity guard that
# the intended branch condition genuinely holds along the orbit.
@pytest.mark.parametrize(
    "config", ["const_lnLambda", "rhm_coulomb_log", "minr_gate", "sigma_clamp"]
)
def test_chandrasekhar_dxdv_fd_of_flow_branches(config):
    from galpy.orbit import Orbit
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        MWPotential2014,
        evaluateRforces,
    )
    from galpy.util import coords

    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 3.0, 301)
    if config == "const_lnLambda":
        # constant Coulomb logarithm: the dlnLambda/dx = dlnLambda/dv = 0
        # branch of the Jacobian
        cdf = ChandrasekharDynamicalFrictionForce(
            GMs=0.02, rhm=0.0, const_lnLambda=7.0, dens=MWPotential2014, maxr=10.0
        )
    elif config == "rhm_coulomb_log":
        # GMs/v^2 < rhm along the whole orbit (asserted below) -> the
        # rhm-based Coulomb logarithm lnLambda = 0.5 ln(1+r^2/(gamma^2 rhm^2)),
        # which depends on r but not on v (dlnLambda/dv = 0)
        cdf = ChandrasekharDynamicalFrictionForce(
            GMs=0.02, rhm=0.2, dens=MWPotential2014, maxr=10.0
        )
    elif config == "minr_gate":
        # minr above the whole orbit: the force and its Jacobian are
        # identically zero along the orbit (the r < minr gate). NOTE: the
        # force is discontinuous across r=minr, so the flow derivative of an
        # orbit that CROSSES minr picks up a jump (saltation) contribution
        # that the variational equations deliberately omit (the gate zeroes
        # the Jacobian on the inside) -- FD-of-flow and the dxdv integration
        # then genuinely disagree (measured ~0.7 for an orbit decaying through
        # minr=1). The consistent regime is an orbit entirely inside minr,
        # where the friction force vanishes identically and the gate supplies
        # the matching zero Jacobian at every step.
        cdf = ChandrasekharDynamicalFrictionForce(
            GMs=0.05, rhm=0.0, minr=1.5, dens=MWPotential2014, maxr=10.0
        )
    elif config == "sigma_clamp":
        # sigmar interpolation grid ends at maxr=0.5 < r along the whole
        # orbit: the C force clamps the spline argument (constant sigma_r
        # beyond the grid) and the Jacobian consistently uses sigma_r' = 0
        cdf = ChandrasekharDynamicalFrictionForce(
            GMs=0.02, rhm=0.0, dens=MWPotential2014, maxr=0.5
        )
    pot = MWPotential2014 + cdf
    obase = Orbit(ic)
    if config == "minr_gate":
        # galpy itself flags the r < minr regime (non-vacuity: the gate is on)
        with pytest.warns(galpyWarning, match="r < minr"):
            obase.integrate(times, pot, method="dopr54_c")
    else:
        obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    r = numpy.sqrt(numpy.sum(base_rect[:, :3] ** 2, axis=1))
    v = numpy.sqrt(numpy.sum(base_rect[:, 3:] ** 2, axis=1))
    # Non-vacuity guards: the intended branch condition genuinely holds along
    # the orbit (via the PYTHON-side quantities, independent of the C code)
    if config == "const_lnLambda":
        assert cdf._lnLambda == 7.0  # the constant-lnLambda branch is selected
        assert r[-1] < 0.85 * r[0], (
            f"const-lnLambda test orbit does not decay (r: {r[0]:g} -> {r[-1]:g}); "
            "the dissipative variational test would be vacuous"
        )
    elif config == "rhm_coulomb_log":
        assert not cdf._lnLambda  # variable Coulomb logarithm
        GMvs = cdf._ms / v**2.0
        assert numpy.all(GMvs < cdf._rhm), (
            f"GMs/v^2 (max {numpy.amax(GMvs):g}) does not stay below rhm="
            f"{cdf._rhm:g} along the orbit; the rhm-based Coulomb-log branch "
            "would not be selected"
        )
    elif config == "minr_gate":
        assert numpy.all(r < cdf._minr), (
            f"test orbit (max r {numpy.amax(r):g}) is not entirely inside "
            f"minr={cdf._minr:g}; the r < minr zero gate would not be exercised"
        )
        # ... but the friction configuration is not trivial: outside minr the
        # force is nonzero
        assert (
            numpy.fabs(evaluateRforces(cdf, 2.0, 0.0, phi=0.0, v=[0.1, 1.0, 0.1])) > 0.0
        )
    elif config == "sigma_clamp":
        assert numpy.all(r > cdf._maxr), (
            f"test orbit (min r {numpy.amin(r):g}) does not stay beyond the "
            f"sigmar interpolation grid (maxr={cdf._maxr:g}); the spline-clamp "
            "branch would not be exercised"
        )
    canonical = numpy.eye(6)
    eps = 1e-7
    for ii in range(6):
        pert = base_rect[0].copy()
        pert[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method="dopr54_c")
        fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
        assert fderr < 1e-4, (
            f"Dissipative 3D FD-of-flow for e_{ii} differs from the dxdv "
            f"column by {fderr:g} (MWPotential2014 + Chandrasekhar friction, "
            f"{config} configuration)"
        )
    return None


def _fdm_c_regime(fdf, q):
    """Classify which branch of the C FDM friction coefficient
    (FDMDynamicalFrictionForce.c) the force is on at the rectangular
    phase-space point q, replicating the C regime logic in Python:
    kr_C = 2 mhbar v r vs the Mach number M_sigma = v/sigma_r(r) (with the
    same clamped sigma_r spline as C) selects the zero-velocity
    (kr_C < M_sigma) / dispersion (kr_C > 4 M_sigma) / intermediate FDM
    regime, and the classical cutoff applies when Cfdm/Ccdm >= 1; a constant
    FDM factor short-circuits everything."""
    import scipy.special as sp

    if fdf._const_FDMfactor:
        return "const"
    r = numpy.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    v = numpy.sqrt(q[3] ** 2 + q[4] ** 2 + q[5] ** 2)
    krC = 2.0 * fdf._mhbar * v * r
    rcl = numpy.clip(r, fdf._sigmar_rs_4interp[0], fdf._sigmar_rs_4interp[-1])
    sr = fdf.sigmar(rcl)
    M = v / sr
    X = v / numpy.sqrt(2.0) / sr
    Xf = sp.erf(X) - 2.0 * X / numpy.sqrt(numpy.pi) * numpy.exp(-(X**2))
    if krC < M:
        Cfdm = (
            numpy.euler_gamma
            + numpy.log(krC)
            - sp.sici(krC)[1]
            + numpy.sin(krC) / krC
            - 1.0
        )
        regime = "zerovel"
    elif krC > 4.0 * M:
        Cfdm = numpy.log(krC / M) * Xf
        regime = "dispersion"
    else:
        Czv = numpy.euler_gamma + numpy.log(M) - sp.sici(M)[1] + numpy.sin(M) / M - 1.0
        mu = (2.0 * M - 0.5 * krC) / (1.5 * M)
        Cfdm = mu * Czv + (1.0 - mu) * numpy.log(4.0) * Xf
        regime = "intermediate"
    if Cfdm / (fdf.lnLambda(r, v) * Xf) >= 1.0:
        regime += "+cdmcutoff"
    return regime


def _fdm_dxdv_setup(nt=501):
    """Shared setup for the FDM dissipative orbit-level variational tests: a
    decaying orbit (r: ~1.0 -> ~0.80 over t=0..5) in MWPotential2014 + FDM
    friction with rhm=0 and mhbar=10, i.e. along the entire orbit the force
    is in the FDM dispersion regime (kr_C/M_sigma ~ 10-16 > 4) with the
    quantum-pressure suppression active (Cfdm/Ccdm ~ 0.5-0.65 < 1), so the
    FDM-specific branch of the C Jacobian -- not the classical cutoff -- is
    what these tests exercise (asserted in the tests)."""
    from galpy.potential import (
        FDMDynamicalFrictionForce,
        MWPotential2014,
    )

    fdf = FDMDynamicalFrictionForce(GMs=0.012, rhm=0.0, dens=MWPotential2014, maxr=10.0)
    fdf._mhbar = 10.0  # the exact quantity the C parser ships (p._mhbar)
    pot = MWPotential2014 + fdf
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, nt)
    return fdf, pot, ic, times


def _assert_fdm_suppression_active(fdf, rect_orbit):
    """Assert that the FDM force is on the FDM dispersion branch (not the
    classical cutoff, not another regime) all along the orbit, so the
    orbit-level tests genuinely exercise the FDM-specific Jacobian."""
    regimes = {_fdm_c_regime(fdf, rect_orbit[kk]) for kk in range(len(rect_orbit))}
    assert regimes == {"dispersion"}, (
        f"FDM test orbit wanders off the FDM dispersion branch (regimes "
        f"found: {regimes}); the FDM orbit-level variational tests would "
        "not exercise the FDM-specific Jacobian"
    )


def test_fdm_dxdv_fd_of_flow():
    # FD-of-the-flow STM test for the dissipative 3D variational equations
    # with FDMDynamicalFrictionForce (mirroring the Chandrasekhar test above):
    # every column i of the C-integrated deviation (seeded with the canonical
    # e_i) must match (x(t; x0+eps e_i) - x(t; x0))/eps built from plain orbit
    # integrations, which use only the separately-validated forces, for a
    # decaying orbit in MWPotential2014 + FDM friction on the FDM dispersion
    # branch (suppression active along the entire orbit).
    from galpy.orbit import Orbit
    from galpy.util import coords

    fdf, pot, ic, times = _fdm_dxdv_setup()
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    # friction must be doing real work: the orbit decays
    rstart = numpy.sqrt(numpy.sum(base_rect[0, :3] ** 2))
    rend = numpy.sqrt(numpy.sum(base_rect[-1, :3] ** 2))
    assert rend < 0.85 * rstart, (
        f"FDM test orbit does not decay (r: {rstart:g} -> {rend:g}); "
        "the dissipative variational test would be vacuous"
    )
    # ... and the FDM suppression must be active (not the classical cutoff)
    _assert_fdm_suppression_active(fdf, base_rect)
    canonical = numpy.eye(6)
    eps = 1e-7
    for ii in range(6):
        pert = base_rect[0].copy()
        pert[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method="dopr54_c")
        fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
        assert fderr < 1e-4, (
            f"Dissipative 3D FD-of-flow for e_{ii} differs from the dxdv "
            f"column by {fderr:g} (MWPotential2014 + FDM friction)"
        )
    return None


def test_fdm_dxdv_phase_volume_law():
    # Quantitative non-conservative test for FDMDynamicalFrictionForce
    # (mirroring the Chandrasekhar test above): Abel/Jacobi-Liouville gives
    #   det M(t) = exp( int_0^t tr(dF/dv) dt' )
    # with tr(dF/dv) computed by central finite differences of the
    # pure-Python FDM force along the orbit (trapezoidal rule on the fine
    # output grid) -- a code path independent of the C variational machinery
    # that produced the STM. The FDM suppression weakens the friction
    # relative to pure Chandrasekhar, but the phase volume still
    # contracts: det M < 1.
    from galpy.orbit import Orbit

    # the fine grid keeps the trapezoidal error of the trace integral (the
    # quadrature is the limiting factor, O(h^2)) below the 1e-5 tolerance
    fdf, pot, ic, times = _fdm_dxdv_setup(nt=1001)
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    # the FDM suppression must be active (not the classical cutoff), so the
    # FDM-specific branch of the Jacobian is what this test exercises
    _assert_fdm_suppression_active(fdf, base_rect)
    canonical = numpy.eye(6)
    Mcols = []
    for ii in range(6):
        o = Orbit(ic)
        o.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        Mcols.append(o.getOrbit_dxdv())
    Mt = numpy.array(Mcols).transpose(1, 2, 0)  # (nt,6,6), columns = e_i images
    detM = numpy.array([numpy.linalg.det(Mt[kk]) for kk in range(len(times))])
    # tr(dF/dv) along the orbit by central FD of the PYTHON forces (the
    # friction is the only dissipative component; conservative forces do not
    # depend on v and contribute exactly 0)
    trJv = _python_fd_trace_dFdv(fdf, base_rect, times)
    integral = numpy.concatenate(
        ([0.0], numpy.cumsum(0.5 * (trJv[1:] + trJv[:-1]) * numpy.diff(times)))
    )
    pred = numpy.exp(integral)
    relerr = numpy.amax(numpy.fabs(detM - pred) / pred)
    assert relerr < 1e-5, (
        f"Dissipative phase-volume law det M(t) = exp(int tr(dF/dv) dt') "
        f"violated at relative level {relerr:g} (FDM friction)"
    )
    # friction contracts phase volume: det M decays below 1
    assert detM[-1] < 1.0, f"det M(t_end)={detM[-1]:g} should be < 1 with friction"
    assert detM[-1] < 0.9, (
        f"det M(t_end)={detM[-1]:g}: phase-space contraction too weak for the "
        "phase-volume-law test to be meaningful"
    )
    return None


# The C rectangular FDM friction Jacobian
# (FDMDynamicalFrictionForce.c::...RectDissipativeForceJacobian) mirrors the
# branch structure of the force it differentiates: on top of the branches
# shared with Chandrasekhar friction (constant lnLambda; the rhm-based
# Coulomb-log branch GMs/v^2 < rhm; the r < minr zero gate; the clamped
# sigma_r spline outside the interpolation grid), the effective friction
# coefficient itself has the three FDM kr-regimes (zero-velocity kr < Msig /
# dispersion kr > 4 Msig / intermediate mu-interpolation in between, with
# kr = 2 mhbar v r and Msig = v/sigma_r), the classical Chandrasekhar cutoff
# when Cfdm/Ccdm >= 1, and the constant-FDM-factor short-circuit. The main
# FD-of-flow test above runs the dispersion regime with the suppression
# active (GMvs-based Coulomb log, inside the sigma grid); the configurations
# here select each of the other branches through the regular constructor
# options (+ the _mhbar boson-mass quantity the C parser ships, to place
# kr/Msig in the desired regime) and validate with the same
# finite-difference-of-the-flow check (ground truth built from plain orbit
# integrations, which use only the separately-validated forces), each with a
# non-vacuity guard -- via the _fdm_c_regime classifier, an independent
# Python replication of the C regime logic -- that the intended branch is
# genuinely the one the C code is on along the orbit.
@pytest.mark.parametrize(
    "config",
    [
        "minr_gate",
        "sigma_clamp",
        "rhm_coulomb_log_cdm_cutoff",
        "const_lnLambda_cdm_cutoff",
        "zerovel",
        "intermediate",
        "const_FDMfactor",
    ],
)
def test_fdm_dxdv_fd_of_flow_branches(config):
    from galpy.orbit import Orbit
    from galpy.potential import (
        FDMDynamicalFrictionForce,
        MWPotential2014,
        evaluateRforces,
    )
    from galpy.util import coords

    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 3.0, 301)
    if config == "minr_gate":
        # minr above the whole orbit: the force and its Jacobian are
        # identically zero along the orbit (the r < minr gate supplies the
        # matching zero Jacobian at every step; see the Chandrasekhar
        # minr_gate configuration above for why an orbit CROSSING minr is
        # deliberately not tested: the C^0 force jump there makes FD-of-flow
        # ill-posed)
        fdf = FDMDynamicalFrictionForce(
            GMs=0.05, rhm=0.0, minr=1.5, dens=MWPotential2014, maxr=10.0
        )
    elif config == "sigma_clamp":
        # sigmar interpolation grid ends at maxr=0.5 < r along the whole
        # orbit: the C force clamps the spline argument (constant sigma_r
        # beyond the grid) and the Jacobian consistently uses sigma_r' = 0;
        # mhbar=10 keeps the coefficient on the FDM dispersion branch with
        # the suppression active (as in the main FDM test)
        fdf = FDMDynamicalFrictionForce(
            GMs=0.012, rhm=0.0, dens=MWPotential2014, maxr=0.5
        )
        fdf._mhbar = 10.0
    elif config == "rhm_coulomb_log_cdm_cutoff":
        # GMs/v^2 < rhm along the whole orbit -> the rhm-based Coulomb
        # logarithm lnLambda = 0.5 ln(1+r^2/(gamma^2 rhm^2)) (r-dependent,
        # v-independent), and mhbar=100 pushes kr/Msig ~ 130-155 so that
        # Cfdm = ln(kr/Msig) Xf > Ccdm = lnLambda Xf: the classical-cutoff
        # branch Ceff = Ccdm of the Jacobian, which consumes the rhm-based
        # dlnLambda/dr
        fdf = FDMDynamicalFrictionForce(
            GMs=0.02, rhm=0.2, dens=MWPotential2014, maxr=10.0
        )
        fdf._mhbar = 100.0
    elif config == "const_lnLambda_cdm_cutoff":
        # constant Coulomb logarithm (dlnLambda/dx = dlnLambda/dv = 0 branch)
        # with mhbar=100 -> Cfdm = ln(kr/Msig) Xf ~ 4.9 Xf > Ccdm = 2 Xf: the
        # classical cutoff is active and consumes the zero lnLambda derivatives
        fdf = FDMDynamicalFrictionForce(
            GMs=0.02, rhm=0.0, const_lnLambda=2.0, dens=MWPotential2014, maxr=10.0
        )
        fdf._mhbar = 100.0
    elif config == "zerovel":
        # mhbar=0.6 places kr/Msig = 2 mhbar r sigma_r in [0.78,0.94] < 1
        # along the whole orbit: the zero-velocity (Cin-series) regime of the
        # FDM coefficient, with the suppression active (Cfdm ~ 0.15 << Ccdm)
        fdf = FDMDynamicalFrictionForce(
            GMs=0.08, rhm=0.0, dens=MWPotential2014, maxr=10.0
        )
        fdf._mhbar = 0.6
    elif config == "intermediate":
        # mhbar=2 places kr/Msig in [2.6,3.1], inside (1,4) along the whole
        # orbit: the mu-interpolated intermediate regime between the
        # zero-velocity and dispersion coefficients
        fdf = FDMDynamicalFrictionForce(
            GMs=0.05, rhm=0.0, dens=MWPotential2014, maxr=10.0
        )
        fdf._mhbar = 2.0
    elif config == "const_FDMfactor":
        # constant FDM factor: the force always applies it and the Jacobian
        # short-circuits to Ceff = const, dCeff/dr = dCeff/dv = 0
        fdf = FDMDynamicalFrictionForce(
            GMs=0.02, rhm=0.0, const_FDMfactor=0.5, dens=MWPotential2014, maxr=10.0
        )
    pot = MWPotential2014 + fdf
    obase = Orbit(ic)
    if config == "minr_gate":
        # galpy itself flags the r < minr regime (non-vacuity: the gate is on)
        with pytest.warns(galpyWarning, match="r < minr"):
            obase.integrate(times, pot, method="dopr54_c")
    else:
        obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    r = numpy.sqrt(numpy.sum(base_rect[:, :3] ** 2, axis=1))
    v = numpy.sqrt(numpy.sum(base_rect[:, 3:] ** 2, axis=1))
    # Non-vacuity guards: the intended branch is genuinely the one the C code
    # is on along the orbit (via the PYTHON-side _fdm_c_regime replication of
    # the C regime logic and constructor attributes, independent of the C code)
    regimes = {_fdm_c_regime(fdf, base_rect[kk]) for kk in range(len(base_rect))}
    if config == "minr_gate":
        assert numpy.all(r < fdf._minr), (
            f"test orbit (max r {numpy.amax(r):g}) is not entirely inside "
            f"minr={fdf._minr:g}; the r < minr zero gate would not be exercised"
        )
        # ... but the friction configuration is not trivial: outside minr the
        # force is nonzero
        assert (
            numpy.fabs(evaluateRforces(fdf, 2.0, 0.0, phi=0.0, v=[0.1, 1.0, 0.1])) > 0.0
        )
    elif config == "sigma_clamp":
        assert numpy.all(r > fdf._maxr), (
            f"test orbit (min r {numpy.amin(r):g}) does not stay beyond the "
            f"sigmar interpolation grid (maxr={fdf._maxr:g}); the spline-clamp "
            "branch would not be exercised"
        )
        assert regimes == {"dispersion"}, (
            f"sigma-clamp test orbit wanders off the FDM dispersion branch "
            f"(regimes found: {regimes})"
        )
    elif config == "rhm_coulomb_log_cdm_cutoff":
        assert not fdf._lnLambda  # variable Coulomb logarithm
        GMvs = fdf._ms / v**2.0
        assert numpy.all(GMvs < fdf._rhm), (
            f"GMs/v^2 (max {numpy.amax(GMvs):g}) does not stay below rhm="
            f"{fdf._rhm:g} along the orbit; the rhm-based Coulomb-log branch "
            "would not be selected"
        )
        assert regimes == {"dispersion+cdmcutoff"}, (
            f"rhm/cutoff test orbit is not on the classical-cutoff branch "
            f"along the whole orbit (regimes found: {regimes})"
        )
    elif config == "const_lnLambda_cdm_cutoff":
        assert fdf._lnLambda == 2.0  # the constant-lnLambda branch is selected
        assert regimes == {"dispersion+cdmcutoff"}, (
            f"const-lnLambda/cutoff test orbit is not on the classical-cutoff "
            f"branch along the whole orbit (regimes found: {regimes})"
        )
    elif config == "zerovel":
        assert regimes == {"zerovel"}, (
            f"zero-velocity test orbit wanders off the FDM zero-velocity "
            f"branch (regimes found: {regimes})"
        )
    elif config == "intermediate":
        assert regimes == {"intermediate"}, (
            f"intermediate-regime test orbit wanders off the FDM intermediate "
            f"branch (regimes found: {regimes})"
        )
    elif config == "const_FDMfactor":
        assert fdf._const_FDMfactor == 0.5  # the constant-factor short-circuit
        assert regimes == {"const"}
    canonical = numpy.eye(6)
    eps = 1e-7
    for ii in range(6):
        pert = base_rect[0].copy()
        pert[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method="dopr54_c")
        fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
        assert fderr < 1e-4, (
            f"Dissipative 3D FD-of-flow for e_{ii} differs from the dxdv "
            f"column by {fderr:g} (MWPotential2014 + FDM friction, "
            f"{config} configuration)"
        )
    return None


def _noninertial_dxdv_flow_setup():
    """Shared setup for the NonInertialFrameForce orbit-level variational
    tests: MWPotential2014 viewed from (1) a spinning-up frame (scalar Omega
    with constant Omegadot; pure fixed-args C path, pot_type 39) and (2) a
    frame with vector Omega(t) functions plus a translating origin
    (x0/v0/a0), evaluated through the cinterp C splines (pot_type 45)."""
    from galpy.potential import MWPotential2014, NonInertialFrameForce

    configs = {
        "omegaz_omegadot_args": MWPotential2014
        + NonInertialFrameForce(Omega=1.1, Omegadot=0.07),
        "vecfunc_linacc_cinterp": MWPotential2014
        + NonInertialFrameForce(
            Omega=[
                lambda t: 0.15 + 0.03 * numpy.sin(t),
                lambda t: 0.1 - 0.02 * t,
                lambda t: 1.0 + 0.05 * numpy.cos(t),
            ],
            Omegadot=[
                lambda t: 0.03 * numpy.cos(t),
                lambda t: -0.02 + 0.0 * t,
                lambda t: -0.05 * numpy.sin(t),
            ],
            x0=[lambda t: 0.05 * t**2, lambda t: -0.03 * t**2, lambda t: 0.01 * t**2],
            v0=[lambda t: 0.1 * t, lambda t: -0.06 * t, lambda t: 0.02 * t],
            a0=[
                lambda t: 0.1 + 0.0 * t,
                lambda t: -0.06 + 0.0 * t,
                lambda t: 0.02 + 0.0 * t,
            ],
            cinterp=True,
        ),
    }
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    return configs, ic, times


def _stm_from_dxdv(pot, ic, times):
    """Full 6x6 STM M(t) (shape (nt,6,6)) from the 6 canonical rectangular
    deviation integrations with the C dxdv integrator."""
    from galpy.orbit import Orbit

    canonical = numpy.eye(6)
    Mcols = []
    for ii in range(6):
        o = Orbit(ic)
        o.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        Mcols.append(o.getOrbit_dxdv())
    return numpy.array(Mcols).transpose(1, 2, 0)


def test_noninertial_dxdv_fd_of_flow():
    # FD-of-the-flow STM test for the NonInertialFrameForce variational
    # equations: every column i of the C-integrated deviation (seeded with the
    # canonical e_i) must match (x(t; x0+eps e_i) - x(t; x0))/eps built from
    # plain orbit integrations, which use only the separately-validated
    # forces -- the same check (and tolerance) as the conservative
    # test_liouville_3d battery and the Chandrasekhar FD-of-flow test, for
    # MWPotential2014 in a spinning-up frame (fixed-args path) and in a
    # vector-Omega(t)+translation frame through the cinterp splines.
    from galpy.orbit import Orbit
    from galpy.util import coords

    configs, ic, times = _noninertial_dxdv_flow_setup()
    canonical = numpy.eye(6)
    eps = 1e-7
    for cname, pot in configs.items():
        obase = Orbit(ic)
        obase.integrate(times, pot, method="dopr54_c")
        base_rect = _orbit_rect_columns_3d(obase, times)
        for ii in range(6):
            pert = base_rect[0].copy()
            pert[ii] += eps
            Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
            vRp, vTp, vzp = coords.rect_to_cyl_vec(
                pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
            )
            opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
            opert.integrate(times, pot, method="dopr54_c")
            fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii],
                times,
                pot,
                method="dopr54_c",
                rectIn=True,
                rectOut=True,
                rtol=1e-12,
                atol=1e-12,
            )
            fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
            assert fderr < 1e-4, (
                f"NonInertialFrameForce 3D FD-of-flow for e_{ii} differs from "
                f"the dxdv column by {fderr:g} ({cname})"
            )
    return None


def test_noninertial_dxdv_phase_volume_preserved():
    # KEY physics test: dF/dv = -2 [Omega]_x is antisymmetric, so
    # tr(dF/dv) = 0 and Abel/Jacobi-Liouville gives det M(t) =
    # exp(int tr(dF/dv) dt') = 1 EXACTLY: a rotating (even arbitrarily
    # time-dependent, translating) frame is phase-volume PRESERVING, despite
    # the force being velocity-dependent. Unlike friction (det M < 1, see
    # test_chandrasekhar_dxdv_phase_volume_law), this exercises the velocity
    # block of the variational Jacobian nontrivially while leaving the volume
    # invariant -- det M(t) = 1 must hold to ~1e-8 along the whole orbit for
    # both the fixed-args and the cinterp-spline C configurations.
    configs, ic, times = _noninertial_dxdv_flow_setup()
    for cname, pot in configs.items():
        Mt = _stm_from_dxdv(pot, ic, times)
        detM = numpy.array([numpy.linalg.det(Mt[kk]) for kk in range(len(times))])
        maxdev = numpy.amax(numpy.fabs(detM - 1.0))
        assert maxdev < 1e-8, (
            f"NonInertialFrameForce phase-volume preservation det M(t)=1 "
            f"violated at level {maxdev:g} ({cname})"
        )
    return None


def test_noninertial_dxdv_rotating_vs_inertial_stm():
    # Exact cross-check of the rotating-frame variational equations against
    # the trusted inertial-frame ones: for a steady spherical potential (a
    # Plummer sphere) and a constant vector Omega, the inertial orbit x(t)
    # and the rotating-frame orbit r(t) are related by x = R(t) r with
    # R(t) = expm([Omega]_x t) (galpy convention: Omega is the frequency of
    # the rotating frame as seen from the inertial frame), so the deviations
    # map as w_rot = T(t) w_in with T = [[R^T, 0], [-[Omega]_x R^T, R^T]]
    # (from dr = R^T dx, dr' = R^T dv + dR^T/dt dx, dR^T/dt = -[Omega]_x R^T)
    # and the STMs must satisfy M_rot(t) = T(t) M_in(t) T(0)^{-1}. This
    # validates the full Jacobian (Coriolis dF/dv AND centrifugal dF/dx)
    # including all signs/factors at integrator precision (~1e-11; tolerance
    # 1e-9), far beyond the 1e-4 FD-of-flow check.
    from scipy.linalg import expm

    from galpy.orbit import Orbit
    from galpy.potential import NonInertialFrameForce, PlummerPotential
    from galpy.util import coords

    pp = PlummerPotential(amp=1.0, b=0.7, normalize=True)
    Om = numpy.array([0.25, 0.15, 0.6])
    Omx = numpy.array(
        [[0.0, -Om[2], Om[1]], [Om[2], 0.0, -Om[0]], [-Om[1], Om[0], 0.0]]
    )
    pot_rot = pp + NonInertialFrameForce(Omega=Om)
    # inertial IC and the corresponding rotating-frame IC: at t=0, R(0)=I so
    # r(0) = x(0) and r'(0) = v(0) - Omega x x(0)
    x0 = numpy.array([0.9, 0.3, 0.2])
    v0 = numpy.array([-0.1, 0.95, 0.15])
    r0 = x0.copy()
    rd0 = v0 - numpy.cross(Om, x0)
    times = numpy.linspace(0.0, 5.0, 101)

    def cyl_ic(x, v):
        R, phi, Z = coords.rect_to_cyl(*x)
        vR, vT, vz = coords.rect_to_cyl_vec(*v, *x)
        return [R, vR, vT, Z, vz, phi]

    ic_in = cyl_ic(x0, v0)
    ic_rot = cyl_ic(r0, rd0)
    # the base orbits must agree under the frame map: r(t) = R(t)^T x(t)
    o_in = Orbit(ic_in)
    o_in.integrate(times, pp, method="dop853_c")
    o_rot = Orbit(ic_rot)
    o_rot.integrate(times, pot_rot, method="dop853_c")
    xin = _orbit_rect_columns_3d(o_in, times)
    xrot = _orbit_rect_columns_3d(o_rot, times)
    err_orb = numpy.amax(
        numpy.fabs(
            numpy.array(
                [
                    expm(Omx * times[kk]).T @ xin[kk, :3] - xrot[kk, :3]
                    for kk in range(len(times))
                ]
            )
        )
    )
    assert err_orb < 1e-10, (
        f"Rotating-frame orbit r(t) differs from R(t)^T x(t) by {err_orb:g}: "
        "the frame map underlying the STM cross-check does not hold"
    )
    # STM cross-check at several times along the orbit
    M_in = _stm_from_dxdv(pp, ic_in, times)
    M_rot = _stm_from_dxdv(pot_rot, ic_rot, times)

    def Tmat(t):
        Rt = expm(Omx * t)
        T = numpy.zeros((6, 6))
        T[:3, :3] = Rt.T
        T[3:, 3:] = Rt.T
        T[3:, :3] = -Omx @ Rt.T
        return T

    T0inv = numpy.linalg.inv(Tmat(0.0))
    for kk in [25, 50, 100]:
        pred = Tmat(times[kk]) @ M_in[kk] @ T0inv
        err_stm = numpy.amax(numpy.fabs(pred - M_rot[kk]))
        assert err_stm < 1e-9, (
            f"Rotating-frame STM differs from the transformed inertial STM by "
            f"{err_stm:g} at t={times[kk]:g}"
        )
    # and the rotating-frame STM preserves phase volume exactly
    detM = numpy.array([numpy.linalg.det(M_rot[kk]) for kk in range(len(times))])
    assert numpy.amax(numpy.fabs(detM - 1.0)) < 1e-9, (
        "Rotating-frame STM det M(t) != 1 in the rotating-vs-inertial cross-check"
    )
    return None


# The C rectangular frame-force Jacobian
# (NonInertialFrameForce.c::...RectDissipativeForceJacobian) mirrors the
# branch structure of the force it differentiates (see
# test_chandrasekhar_dxdv_fd_of_flow_branches for the friction analogue):
# the translation-only early exit (no rotation: F = -a0(t) does not depend
# on (x,v), so the Jacobian is identically zero), the scalar
# Omega_z(t)-as-function branch (Omega_z and Omegadot_z through the tdep
# helpers), and the constant-vector Omega + constant-vector Omegadot branch
# (Omega(t) = Omega + Omegadot t componentwise). The main FD-of-flow test
# above runs the constant-scalar-Omega+Omegadot fixed-args branch and the
# vector-Omega(t)-functions branch; the configurations here select each of
# the remaining branches through the regular constructor options and
# validate with the same finite-difference-of-the-flow check (ground truth
# built from plain orbit integrations, which use only the separately-
# validated forces), each with a non-vacuity guard (via the PYTHON-side
# configuration, independent of the C code) that the intended branch is
# genuinely selected and dynamically active.
@pytest.mark.parametrize("config", ["linacc_only", "omegaz_func", "vec_omegadot_args"])
def test_noninertial_dxdv_fd_of_flow_branches(config):
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, NonInertialFrameForce
    from galpy.util import coords

    if config == "linacc_only":
        # translation-only frame (a0 only, no rotation): F = -a0(t) is
        # independent of (x, v), so the frame-force Jacobian is identically
        # zero (the rot_acc early exit in C); the deviation still feels a0
        # through the displaced base orbit along which the potential Hessian
        # is evaluated, so FD-of-flow remains a nontrivial check
        nif = NonInertialFrameForce(a0=[0.06, -0.04, 0.03])
    elif config == "omegaz_func":
        # scalar Omega_z as a function of time (with its derivative provided
        # as a function too): the omegaz_only Omega-as-function evaluation of
        # Omega_z(t) and Omegadot_z(t) in the C Jacobian (through the cinterp
        # splines, the default for C integration)
        nif = NonInertialFrameForce(
            Omega=lambda t: 1.0 + 0.08 * numpy.sin(t),
            Omegadot=lambda t: 0.08 * numpy.cos(t),
        )
    elif config == "vec_omegadot_args":
        # constant vector Omega with constant vector Omegadot:
        # Omega(t) = Omega + Omegadot t with all six components nonzero, so
        # every componentwise term of this C branch is dynamically active
        nif = NonInertialFrameForce(
            Omega=numpy.array([0.05, 0.08, 1.0]),
            Omegadot=numpy.array([0.02, -0.015, 0.03]),
        )
    pot = MWPotential2014 + nif
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    # Non-vacuity guards: the intended branch is genuinely selected and its
    # quantities are nonzero, so a corrupted Jacobian term could not hide
    if config == "linacc_only":
        assert not nif._rot_acc and nif._lin_acc, (
            "linacc_only configuration does not select the translation-only branch"
        )
        assert numpy.all(numpy.fabs(nif._a0_py(2.5)) > 0.0), (
            "linacc_only test acceleration has zero components; the "
            "translation-only branch test would be vacuous"
        )
    elif config == "omegaz_func":
        assert nif._omegaz_only and nif._Omega_as_func and not nif._const_freq, (
            "omegaz_func configuration does not select the scalar "
            "Omega-as-function branch"
        )
        assert nif._cinterp  # evaluated through the C splines (pot_type 45)
        assert numpy.fabs(nif._Omega_py(1.5) - nif._Omega_py(0.0)) > 0.0, (
            "omegaz_func test frequency does not vary in time"
        )
        assert numpy.fabs(nif._Omegadot_py(0.0)) > 0.0, (
            "omegaz_func test frequency derivative vanishes; the Euler term "
            "of the Jacobian would be untested"
        )
    elif config == "vec_omegadot_args":
        assert (
            not nif._omegaz_only and not nif._Omega_as_func and not nif._const_freq
        ), (
            "vec_omegadot_args configuration does not select the "
            "constant-vector Omega + Omegadot branch"
        )
        assert numpy.all(numpy.fabs(nif._Omega) > 0.0) and numpy.all(
            numpy.fabs(nif._Omegadot) > 0.0
        ), (
            "vec_omegadot_args test frequencies have zero components; parts "
            "of the componentwise C branch would be untested"
        )
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dopr54_c")
    base_rect = _orbit_rect_columns_3d(obase, times)
    if config == "linacc_only":
        # ... and a0 genuinely acts: the base orbit differs substantially
        # from the inertial no-frame orbit
        onoframe = Orbit(ic)
        onoframe.integrate(times, MWPotential2014, method="dopr54_c")
        assert (
            numpy.amax(numpy.fabs(_orbit_rect_columns_3d(onoframe, times) - base_rect))
            > 0.1
        ), (
            "linacc_only test orbit is barely displaced by a0; the "
            "translation-only branch test would be vacuous"
        )
    canonical = numpy.eye(6)
    eps = 1e-7
    for ii in range(6):
        pert = base_rect[0].copy()
        pert[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(pert[0], pert[1], pert[2])
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert[3], pert[4], pert[5], pert[0], pert[1], pert[2]
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method="dopr54_c")
        fd = (_orbit_rect_columns_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
        assert fderr < 1e-4, (
            f"NonInertialFrameForce 3D FD-of-flow for e_{ii} differs from "
            f"the dxdv column by {fderr:g} ({config} configuration)"
        )
    return None


def test_noninertial_dxdv_flags_and_gate():
    # NonInertialFrameForce advertises its exact C rectangular Jacobian for
    # EVERY configuration (the force is linear in (x,v), so no configuration
    # is unwireable): hasC_dxdv3d=True on the class, aggregated through
    # CompositePotential. The pure-Python integrate_dxdv methods refuse
    # dissipative forces loudly, and a forced hasC_dxdv3d=False (the gate a
    # genuinely unwireable configuration would use) makes the C methods fall
    # back to odeint with a warning, which then also raises -- never a silent
    # wrong answer.
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, NonInertialFrameForce

    nif = NonInertialFrameForce(Omega=1.1, Omegadot=0.07)
    assert nif.hasC_dxdv3d, "NonInertialFrameForce should advertise hasC_dxdv3d"
    pot = MWPotential2014 + nif
    assert pot.hasC_dxdv3d, (
        "CompositePotential should aggregate hasC_dxdv3d=True for "
        "MWPotential2014 + NonInertialFrameForce"
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 3)
    for method in ["odeint", "dop853"]:
        o = Orbit(ic)
        with pytest.raises(NotImplementedError) as excinfo:
            o.integrate_dxdv(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                times,
                pot,
                method=method,
                rectIn=True,
                rectOut=True,
            )
        assert "dissipative" in str(excinfo.value)
    # forced-flag gate: a NonInertialFrameForce WITHOUT the C Jacobian must
    # not silently produce a conservative-only deviation
    nif_noc = NonInertialFrameForce(Omega=1.1, Omegadot=0.07)
    nif_noc.hasC_dxdv3d = False
    pot_noc = MWPotential2014 + nif_noc
    assert not pot_noc.hasC_dxdv3d
    o = Orbit(ic)
    with pytest.warns(galpyWarning, match="Using odeint"):
        with pytest.raises(NotImplementedError) as excinfo:
            o.integrate_dxdv(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                times,
                pot_noc,
                method="dopr54_c",
                rectIn=True,
                rectOut=True,
            )
    assert "dissipative" in str(excinfo.value)
    return None


def test_dissipative_dxdv_python_raises():
    # The pure-Python 3D variational RHS (_EOM_dxdv) only implements the
    # conservative system, so integrate_dxdv with a dissipative force must
    # fail loudly (NotImplementedError) for the Python-based methods rather
    # than silently produce a wrong deviation -- both when a Python method is
    # requested directly and when a C method falls back to odeint because a
    # dissipative force does not have its rectangular Jacobian in C
    # (hasC_dxdv3d=False).
    from galpy.orbit import Orbit
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        MWPotential2014,
    )
    from galpy.potential.DissipativeForce import DissipativeForce

    cdf, pot, ic, times = _chandrasekhar_dxdv_setup()
    # default flags: the base DissipativeForce does not have the C Jacobian;
    # the wired ChandrasekharDynamicalFrictionForce and
    # FDMDynamicalFrictionForce do (incl. through a CompositePotential, which
    # aggregates hasC_dxdv3d)
    from galpy.potential import FDMDynamicalFrictionForce

    assert not DissipativeForce(amp=1.0).hasC_dxdv3d
    assert cdf.hasC_dxdv3d
    assert pot.hasC_dxdv3d  # CompositePotential aggregation
    assert FDMDynamicalFrictionForce(GMs=0.01).hasC_dxdv3d
    times = times[:3]
    for method in ["odeint", "dop853"]:
        o = Orbit(ic)
        with pytest.raises(NotImplementedError) as excinfo:
            o.integrate_dxdv(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                times,
                pot,
                method=method,
                rectIn=True,
                rectOut=True,
            )
        assert "dissipative" in str(excinfo.value)
    # a dissipative force WITHOUT the C Jacobian (forced flag, so this stays
    # valid even once more dissipative forces gain hasC_dxdv3d=True): the C
    # method falls back to odeint with a warning, which then raises
    cdf_noc = ChandrasekharDynamicalFrictionForce(
        GMs=0.008, rhm=0.0, dens=MWPotential2014, maxr=10.0
    )
    cdf_noc.hasC_dxdv3d = False
    pot_noc = MWPotential2014 + cdf_noc
    assert not pot_noc.hasC_dxdv3d  # CompositePotential aggregation
    o = Orbit(ic)
    with pytest.warns(galpyWarning, match="Using odeint"):
        with pytest.raises(NotImplementedError) as excinfo:
            o.integrate_dxdv(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                times,
                pot_noc,
                method="dopr54_c",
                rectIn=True,
                rectOut=True,
            )
    assert "dissipative" in str(excinfo.value)
    return None


def test_dissipative_excluded_from_liouville3d_registry():
    # The det(M)=1/symplecticity battery (test_liouville_3d and the 2D bridge,
    # parametrized over the conftest liouville3d_registry) must NOT contain
    # dissipative forces: for them det M = exp(int tr(dF/dv) dt') != 1 and
    # symplecticity fails BY CONSTRUCTION (see the phase-volume-law test
    # above, which asserts det M < 1). The registry is a hand-curated literal
    # in tests/conftest.py; guard against someone accidentally adding a
    # dissipative force to it.
    import os
    import re

    conftest_file = os.path.join(os.path.dirname(__file__), "conftest.py")
    with open(conftest_file) as f:
        src = f.read()
    m = re.search(r"liouville3d_registry = \[(.*?)\n        \]", src, re.DOTALL)
    assert m is not None, "could not locate the liouville3d_registry in conftest.py"
    registry_src = m.group(1)
    for forbidden in ("DynamicalFriction", "NonInertialFrameForce", "Dissipative"):
        assert forbidden not in registry_src, (
            f"{forbidden} found in the liouville3d_registry: dissipative / "
            "velocity-dependent forces do not satisfy det(M)=1/symplecticity "
            "and must be validated by the dedicated dissipative dxdv tests"
        )
    return None


def _movingobject_setup(planar=False):
    # Host + softened moving object on an orbit integrated in the host: the
    # physically meaningful configuration (satellite + host) and a genuinely
    # non-axisymmetric, explicitly time-dependent potential. The object-track
    # window extends well beyond the test integration interval so both the C
    # (GSL spline) and the Python (Orbit interpolation) tracks are evaluated
    # well inside their interpolation ranges.
    from galpy.orbit import Orbit
    from galpy.potential import (
        LogarithmicHaloPotential,
        MovingObjectPotential,
        PlummerPotential,
    )

    host = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ts_obj = numpy.linspace(-1.0, 7.0, 1601)
    if planar:
        obj = Orbit([1.1, 0.1, 1.1, 1.0])  # in-plane object track
    else:
        obj = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 1.0])  # off-plane object track
    obj.integrate(ts_obj, host, method="dop853_c")
    # massive, well-softened kernel: significant forces but a smooth Hessian
    mop = MovingObjectPotential(obj, pot=PlummerPotential(amp=0.3, b=0.3), amp=1.2)
    return host + mop, mop


def test_movingobject_dxdv_3d_c_vs_python():
    # MovingObjectPotential wires the full 3D Hessian in C (hasC_dxdv3d, gated
    # on the kernel's own 3D C Hessian exactly like hasC): the kernel's Hessian
    # at the shifted point x-x_obj(t) -- the moving-object shift is a pure
    # translation of the evaluation point, so the time-dependence enters only
    # through that point, with no extra terms. The C 3D variational integration
    # must match the pure-Python analytic-2nd-derivative reference (dop853) for
    # UNIT-magnitude deviations; that C-vs-Python comparison is what pins the
    # Hessian VALUES (det(M)=1/symplecticity hold for any symmetric K).
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import evaluatephizderivs

    pot, mop = _movingobject_setup()
    assert mop.hasC_dxdv3d, (
        "MovingObjectPotential with a Plummer kernel should advertise hasC_dxdv3d"
    )
    # the host+object composite must propagate the capability (regression test
    # for CompositePotential.hasC_dxdv3d, without which the C 3D variational
    # path silently falls back to odeint for ANY composite)
    assert pot.hasC_dxdv3d, "host+object composite should advertise hasC_dxdv3d"
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 5.0, 251)
    # Guard against a vacuous test: the off-center, off-plane object must give a
    # genuinely nonzero z-phi coupling along the orbit, otherwise the C
    # zphideriv term is multiplied by 0.
    obase = Orbit(ic)
    obase.integrate(times, pot, method="dop853_c")
    zphi_vals = numpy.array(
        [
            evaluatephizderivs(
                mop,
                obase.R(tt),
                obase.z(tt),
                phi=obase.phi(tt),
                t=tt,
                use_physical=False,
            )
            for tt in times
        ]
    )
    assert numpy.amax(numpy.fabs(zphi_vals)) > 1e-3, (
        "d2Phi/dz/dphi must be nonzero along the orbit to exercise the C zphideriv term"
    )
    canonical = numpy.eye(6)
    maxdiff = 0.0
    for integrator in ("dopr54_c", "dop853_c"):
        for ii in [0, 2, 4]:  # e_x, e_z, e_vy unit deviations
            oc = Orbit(ic)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                oc.integrate_dxdv(
                    canonical[ii], times, pot, method=integrator,
                    rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
                )  # fmt: skip
                # the C path must actually be taken (no odeint fallback)
                assert not any("odeint" in str(ww.message) for ww in w), (
                    "C 3D variational integration fell back to odeint for the "
                    "host+moving-object potential"
                )
            op = Orbit(ic)
            op.integrate_dxdv(
                canonical[ii], times, pot, method="dop853",
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
            diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
            maxdiff = max(maxdiff, diff)
    # C-vs-Python agree to ~6e-11 (the dense object track makes the GSL-vs-Orbit
    # track-interpolation difference negligible); 1e-8 leaves a wide margin
    assert maxdiff < 1e-8, (
        f"3D C variational integration for the host+moving-object potential "
        f"differs from the pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    return None


def test_movingobject_dxdv_3d():
    # det(M)=1 / symplecticity / finite-difference-of-the-flow for the
    # host+moving-object potential through the C integrators (the FD-of-flow
    # is the check that pins the C Hessian VALUES against the trusted C
    # forces, independently of the Python reference).
    pot, _ = _movingobject_setup()
    _check_dxdv_3d_c(pot, "host+MovingObjectPotential")
    return None


def test_movingobject_dxdv_planar():
    # PLANAR variational equations for an in-plane object track: the C planar
    # Hessian (PlanarR2deriv/Planarphi2deriv/PlanarRphideriv -- genuinely
    # non-axisymmetric, since the object is off-center) is the kernel's PLANAR
    # Hessian at the shifted point, mirroring MovingObjectPotentialPlanarRforce.
    # Checks: (1) C-vs-Python dxdv for unit deviations (pins the Hessian
    # values), (2) Liouville det(M)=1 and FD-of-flow through the C integrators.
    from galpy.orbit import Orbit
    from galpy.util import coords

    pot, mop = _movingobject_setup(planar=True)
    assert mop.hasC_dxdv, (
        "MovingObjectPotential with a Plummer kernel should advertise hasC_dxdv"
    )
    assert pot.hasC_dxdv, "host+object composite should advertise hasC_dxdv"
    ic = [1.0, 0.1, 1.1, 0.2]  # planar (R, vR, vT, phi)
    times = numpy.linspace(0.0, 5.0, 251)
    canonical = numpy.eye(4)
    # (1) C vs pure-Python analytic reference, unit deviations e_x, e_vx
    maxdiff = 0.0
    for ii in [0, 2]:
        oc = Orbit(ic)
        oc.integrate_dxdv(
            canonical[ii], times, pot, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        op = Orbit(ic)
        op.integrate_dxdv(
            canonical[ii], times, pot, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(oc.getOrbit_dxdv() - op.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    assert maxdiff < 1e-8, (
        f"planar C variational integration for the host+moving-object potential "
        f"differs from the pure-Python reference by {maxdiff:g} (unit deviation)"
    )
    # (2) Liouville + FD-of-flow through the C integrators
    for integrator in ("dopr54_c", "dop853_c"):
        Mcols = []
        for ii in range(4):
            o = Orbit(ic)
            o.integrate_dxdv(
                canonical[ii], times, pot, method=integrator,
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
            Mcols.append(o.getOrbit_dxdv()[-1, :])
        M = numpy.array(Mcols).T
        detM = numpy.linalg.det(M)
        assert numpy.fabs(detM - 1.0) < 1e-8, (
            f"planar Liouville det(M)={detM:g} differs from 1 for the "
            f"host+moving-object potential, integrator {integrator}"
        )
        # FD-of-flow: column i = (x(t;x0+eps e_i)-x(t;x0))/eps vs the dxdv column
        eps = 1e-7
        obase = Orbit(ic)
        obase.integrate(times, pot, method=integrator)
        base = numpy.array(
            [obase.x(times), obase.y(times), obase.vx(times), obase.vy(times)]
        ).T
        for ii in [0, 3]:  # an x and a vy perturbation
            p = base[0].copy()
            p[ii] += eps
            Rp, phip, _ = coords.rect_to_cyl(p[0], p[1], 0.0)
            vRp, vTp, _ = coords.rect_to_cyl_vec(p[2], p[3], 0.0, p[0], p[1], 0.0)
            opert = Orbit([Rp, vRp, vTp, phip])
            opert.integrate(times, pot, method=integrator)
            pert = numpy.array(
                [opert.x(times), opert.y(times), opert.vx(times), opert.vy(times)]
            ).T
            fd = (pert - base) / eps
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii], times, pot, method=integrator,
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
            fderr = numpy.amax(numpy.fabs(fd - odx.getOrbit_dxdv()))
            assert fderr < 1e-4, (
                f"planar FD-of-flow for e_{ii} differs from the dxdv column by "
                f"{fderr:g} for the host+moving-object potential, integrator "
                f"{integrator}"
            )
    return None


def _planar_invariant(pot):
    """Whether the z=0 plane is invariant under the flow (F_z(z=0)=0
    everywhere), the premise of the 3D->2D bridge check below. A tilted or
    z-offset potential (e.g. the RotateAndTiltWrapperPotential registry
    entries) breaks the z -> -z symmetry: a planar IC then immediately leaves
    the z=0 plane, so the bridge identity does not apply (its Hessian is still
    pinned by test_dxdv_3d_c_vs_python and the FD-of-flow check in
    test_liouville_3d)."""
    from galpy.potential import evaluatezforces

    return all(
        numpy.fabs(
            evaluatezforces(pot, 1.0, 0.0, phi=testphi, t=0.0, use_physical=False)
        )
        < 1e-12
        for testphi in (0.0, 0.7, 2.1)
    )


# 2D-reduction bridge (validates the (x,y) block of K): for a planar IC with
# dz=dvz=0 and an in-plane deviation, the (x,y,vx,vy) sub-STM from the 3D
# integrate_dxdv must match the trusted planar integrate_dxdv result.
def test_liouville_3d_2d_bridge(pot):
    from galpy.orbit import Orbit

    # 2D-reduction bridge: for a planar IC (z=vz=0) with an in-plane deviation, the
    # deviation stays planar for any z-symmetric potential (Rzderiv and zphideriv both
    # vanish at z=0), so the (x,y,vx,vy) block of the 3D STM must match the trusted
    # planar integrate_dxdv -- a strong cross-check of the in-plane Cartesian Hessian.
    if not _planar_invariant(pot):
        pytest.skip(
            "the z=0 plane is not invariant for this potential (no z -> -z "
            "symmetry), so the 3D->2D bridge identity does not apply"
        )
    times = numpy.linspace(0.0, 5.0, 251)
    # Planar IC (z=0, vz=0): (R,vR,vT,phi) in 2D and (R,vR,vT,z=0,vz=0,phi) in 3D
    R, vR, vT, phi = 1.0, 0.1, 1.1, 0.2
    # In-plane rectangular deviations to compare (x, y, vx, vy basis)
    # 3D rect order: (x,y,z,vx,vy,vz); planar rect order: (x,y,vx,vy)
    dev3d_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dx
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # dy
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # dvx
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # dvy
    ]
    dev2d_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # 3D->planar index map: (x,y,vx,vy) live at 3D indices (0,1,3,4)
    planar_in_3d = [0, 1, 3, 4]
    pname = pot.__class__.__name__
    for dev3d, dev2d in zip(dev3d_list, dev2d_list):
        # 3D
        o3 = Orbit([R, vR, vT, 0.0, 0.0, phi])
        o3.integrate_dxdv(
            dev3d,
            times,
            pot,
            method="dop853_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        d3 = o3.getOrbit_dxdv()[-1]  # rect (dx,dy,dz,dvx,dvy,dvz)
        # planar (trusted)
        o2 = Orbit([R, vR, vT, phi])
        o2.integrate_dxdv(
            dev2d,
            times,
            pot.toPlanar(),
            method="dop853_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        d2 = o2.getOrbit_dxdv()[-1]  # rect (dx,dy,dvx,dvy)
        # z, vz deviations must stay zero in 3D
        assert numpy.fabs(d3[2]) < 1e-9 and numpy.fabs(d3[5]) < 1e-9, (
            f"3D in-plane deviation leaked into (dz,dvz) for {pname}"
        )
        bridge_err = numpy.amax(numpy.fabs(d3[planar_in_3d] - d2))
        assert bridge_err < 1e-9, (
            f"3D->2D bridge: (x,y,vx,vy) sub-STM differs from planar by "
            f"{bridge_err:g} for {pname}"
        )
    # Same bridge in the cylindrical frame with the DEFAULT rectIn=rectOut=False:
    # exercises and validates the cylindrical<->rectangular deviation transforms
    # inside integrateFullOrbit_dxdv (the default integrate_dxdv path), against the
    # trusted planar integrate_dxdv. Cyl deviation order: (dR,dvR,dvT,dz,dvz,dphi);
    # (dR,dvR,dvT,dphi) live at 3D cyl indices (0,1,2,5).
    dev3d_cyl_list = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dR
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # dvR
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # dvT
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # dphi
    ]
    dev2d_cyl_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    cyl_in_3d = [0, 1, 2, 5]
    for dev3d, dev2d in zip(dev3d_cyl_list, dev2d_cyl_list):
        o3 = Orbit([R, vR, vT, 0.0, 0.0, phi])
        o3.integrate_dxdv(dev3d, times, pot, method="dop853_c", rtol=1e-12, atol=1e-12)
        d3 = o3.getOrbit_dxdv()[-1]  # cyl (dR,dvR,dvT,dz,dvz,dphi)
        o2 = Orbit([R, vR, vT, phi])
        o2.integrate_dxdv(
            dev2d, times, pot.toPlanar(), method="dop853_c", rtol=1e-12, atol=1e-12
        )
        d2 = o2.getOrbit_dxdv()[-1]  # cyl (dR,dvR,dvT,dphi)
        assert numpy.fabs(d3[3]) < 1e-9 and numpy.fabs(d3[4]) < 1e-9, (
            f"3D in-plane cyl deviation leaked into (dz,dvz) for {pname}"
        )
        bridge_err = numpy.amax(numpy.fabs(d3[cyl_in_3d] - d2))
        assert bridge_err < 1e-9, (
            f"3D->2D cyl bridge differs from planar by {bridge_err:g} for {pname}"
        )
    return None


def test_integrate_dxdv_3d_base_orbit_integrity():
    # Regression: integrateFullOrbit_dxdv's returned BASE orbit must equal a
    # plain orbit integration in every phase-space coordinate. Previously
    # coords.rect_to_cyl/rect_to_cyl_vec passed Z/vz through by reference, so
    # the in-place column assignments clobbered them and the returned base
    # orbit had z replaced by vT -- invisible to every deviation-based test
    # (the rectOut deviation columns were correct) but corrupting anything
    # that restarts from the dxdv orbit (e.g. lyapunov renormalization
    # segments)
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    ts = numpy.linspace(0.0, 5.0, 51)
    ic = [1.0, 0.1, 1.1, 0.2, 0.15, 0.3]
    for method in ["dop853_c", "dop853"]:
        odx = Orbit(ic)
        odx.integrate_dxdv(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ts,
            MWPotential2014,
            method=method,
            rectIn=True,
            rectOut=True,
        )
        oref = Orbit(ic)
        oref.integrate(ts, MWPotential2014, method=method)
        diff = odx.getOrbit() - oref.getOrbit()
        # phi wrapping conventions differ between the two code paths
        diff[:, 5] = numpy.fabs(
            numpy.mod(diff[:, 5] + numpy.pi, 2.0 * numpy.pi) - numpy.pi
        )
        maxdiff = numpy.amax(numpy.fabs(diff), axis=0)
        for ii, name in enumerate(["R", "vR", "vT", "z", "vz", "phi"]):
            assert maxdiff[ii] < 1e-6, (
                f"dxdv base orbit {name} differs from plain integration by "
                f"{maxdiff[ii]:g} for method {method}"
            )
    return None


def test_integrate_dxdv_3d_multiobj_and_default_tol():
    # Cover and validate the multi-object (parallel_map) and pure-Python dop853
    # default-tolerance paths of the 3D integrate_dxdv.
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1, normalize=True)
    times = numpy.linspace(0.0, 2.0, 51)
    ic1 = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    ic2 = [1.2, -0.05, 0.9, -0.1, 0.03, 1.0]
    dev = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # multi-object: two orbits' dxdv at once must match the single-object runs
    omulti = Orbit([ic1, ic2])
    omulti.integrate_dxdv(
        [dev, dev], times, pot, method="dop853_c", rectIn=True, rectOut=True
    )
    gmulti = numpy.asarray(omulti.getOrbit_dxdv())  # (2,nt,6)
    for jj, ic in enumerate([ic1, ic2]):
        osingle = Orbit(ic)
        osingle.integrate_dxdv(
            dev, times, pot, method="dop853_c", rectIn=True, rectOut=True
        )
        assert numpy.amax(numpy.fabs(gmulti[jj] - osingle.getOrbit_dxdv())) < 1e-10, (
            "multi-object 3D dxdv differs from the single-object result"
        )
    # pure-Python dop853 with default (None) tolerances vs explicit tolerances
    odef = Orbit(ic1)
    odef.integrate_dxdv(dev, times, pot, method="dop853", rectIn=True, rectOut=True)
    oexp = Orbit(ic1)
    oexp.integrate_dxdv(
        dev,
        times,
        pot,
        method="dop853",
        rectIn=True,
        rectOut=True,
        rtol=1e-12,
        atol=1e-12,
    )
    assert numpy.amax(numpy.fabs(odef.getOrbit_dxdv() - oexp.getOrbit_dxdv())) < 1e-8, (
        "3D dxdv default-tolerance dop853 differs from the explicit-tolerance run"
    )
    return None


def test_integrate_dxdv_3d_c_requires_full_hessian():
    # A 3D potential with only a *planar* C dxdv implementation (hasC_dxdv=True)
    # but no full 3D C Hessian (hasC_dxdv3d=False) must NOT silently take the C 3D
    # variational path: that path would hit the NULL-safe aggregators (which return
    # 0 for the unset z2deriv/Rzderiv/...) and propagate a wrong, zero-curvature
    # deviation with no error. integrate_dxdv must instead fall back to the correct
    # pure-Python integrator. We assert (a) the flag state, (b) that a galpyWarning
    # is issued, and (c) that the C-method result matches the pure-Python result
    # (i.e. it really fell back, rather than returning the wrong C aggregate).
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    # As the Pvar-pot 3D-Hessian fan-out progresses, essentially every 3D
    # potential with a planar C dxdv path (hasC_dxdv=True) also gains the full
    # 3D C Hessian (hasC_dxdv3d=True) -- LogarithmicHalo got it in #907, MN3 and
    # NullPotential in this PR -- so there is no longer a stable *real* example
    # of "planar-but-not-3D". We therefore synthesize the scenario by forcing
    # hasC_dxdv3d=False on an instance. This directly exercises the
    # integrate_dxdv gate, which must warn and fall back to the pure-Python
    # integrator rather than silently taking the C 3D path (whose NULL-safe
    # aggregators would return a wrong, zero-curvature deviation for a genuinely
    # incomplete potential). The pytest.warns below is the primary assertion;
    # the value match confirms the fallback produced the correct result.
    pot = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    assert pot.hasC_dxdv, (
        "test precondition: MiyamotoNagai should have planar hasC_dxdv"
    )
    pot.hasC_dxdv3d = False  # force the planar-but-not-3D scenario
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 2.0, 101)
    dev = [1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]
    o_c = Orbit(ic)
    with pytest.warns(galpyWarning):
        o_c.integrate_dxdv(
            dev,
            times,
            pot,
            method="dopr54_c",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
    o_py = Orbit(ic)
    o_py.integrate_dxdv(
        dev,
        times,
        pot,
        method="dop853",
        rectIn=True,
        rectOut=True,
        rtol=1e-12,
        atol=1e-12,
    )
    dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
    dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
    # Without the gate, the un-gated C result (hitting the NULL-safe aggregators that
    # return 0 for the unset z2deriv/Rzderiv) differs from the correct value by a
    # sizeable fraction of the deviation, far above the 1e-9 tol below; a tight match
    # instead confirms that integrate_dxdv fell back to the Python integrator.
    assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-9, (
        "3D integrate_dxdv did not fall back to the correct integrator for a "
        "potential lacking the full 3D C Hessian (got a silently-wrong C result)"
    )
    return None


def test_integrate_dxdv_3d_wrapper_requires_wrapped_hessian():
    # The 3D-only wrappers (KuzminLikeWrapperPotential,
    # RotateAndTiltWrapperPotential) subclass WrapperPotential DIRECTLY rather
    # than the parentWrapperPotential delegator, so _check_c's wrapper branch
    # must match them too: a KuzminLike wrapper advertises hasC_dxdv3d=True
    # unconditionally, but its C Hessian chain-rules the WRAPPED potential's C
    # R2deriv/Rforce, so when the wrapped potential lacks the 3D C Hessian the
    # C 3D variational path would silently aggregate 0 for the unset R2deriv
    # (NULL-safe aggregators) and propagate a wrong deviation. _check_c must
    # therefore recurse into the wrapped potential and integrate_dxdv must warn
    # and fall back to the pure-Python integrator. As in
    # test_integrate_dxdv_3d_c_requires_full_hessian, the no-3D-C-Hessian
    # wrapped potential is synthesized by forcing hasC_dxdv3d=False.
    from galpy.orbit import Orbit
    from galpy.potential import KuzminLikeWrapperPotential, MiyamotoNagaiPotential

    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    mn.hasC_dxdv3d = False  # force the wrapped-potential-without-3D-C-Hessian case
    pot = KuzminLikeWrapperPotential(pot=mn, a=1.1, b=0.3)
    assert pot.hasC_dxdv3d, (
        "test precondition: the wrapper itself advertises hasC_dxdv3d"
    )
    assert not _check_c(pot, dxdv3d=True), (
        "_check_c(dxdv3d) must recurse into the wrapped potential of a "
        "direct-WrapperPotential subclass and report False"
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
    times = numpy.linspace(0.0, 2.0, 101)
    dev = [1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]
    o_c = Orbit(ic)
    with pytest.warns(galpyWarning):
        o_c.integrate_dxdv(
            dev, times, pot, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
    o_py = Orbit(ic)
    o_py.integrate_dxdv(
        dev, times, pot, method="dop853",
        rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
    )  # fmt: skip
    dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
    dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
    assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-9, (
        "3D integrate_dxdv did not fall back to the correct integrator for a "
        "wrapper whose wrapped potential lacks the full 3D C Hessian"
    )
    return None


# Tests of Orbit.lyapunov: largest Lyapunov exponent from the variational
# equations (Benettin et al. 1980 renormalization on top of integrate_dxdv)
def test_lyapunov_integrable():
    # In an integrable potential all orbits are regular, so the running
    # estimate lambda(t) of the largest Lyapunov exponent must decay ~ln(t)/t
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, LogarithmicHaloPotential

    tend = 1000.0
    ts = numpy.linspace(0.0, tend, 10001)
    # 3D orbit in the integrable isochrone potential
    ip = IsochronePotential(normalize=1.0, b=0.8)
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    lam = o.lyapunov(ts, pot=ip, method="dop853_c")
    assert numpy.isnan(lam[0]), (
        "lyapunov running estimate at ts[0] should be NaN (no elapsed time)"
    )
    assert lam[-1] < 3.0 * numpy.log(tend) / tend, (
        "lyapunov estimate for a regular 3D isochrone orbit does not decay "
        f"to ~ln(t)/t: lambda(t_end)={lam[-1]:g}"
    )
    assert lam[-1] < lam[len(ts) // 2], (
        "lyapunov estimate for a regular 3D isochrone orbit is not decreasing "
        "between the half-time and the end-time"
    )
    # Planar orbit in an axisymmetric potential: also regular (E,Lz integrals)
    lp = LogarithmicHaloPotential(normalize=1.0)
    op = Orbit([1.0, 0.1, 1.1, 0.0])
    lamp = op.lyapunov(ts, pot=lp, method="dop853_c")
    assert lamp[-1] < 3.0 * numpy.log(tend) / tend, (
        "lyapunov estimate for a regular planar axisymmetric orbit does not "
        f"decay to ~ln(t)/t: lambda(t_end)={lamp[-1]:g}"
    )
    assert lamp[-1] < lamp[len(ts) // 2], (
        "lyapunov estimate for a regular planar axisymmetric orbit is not "
        "decreasing between the half-time and the end-time"
    )
    return None


def test_lyapunov_henonheiles_chaotic():
    # Regression test of the largest Lyapunov exponent of a chaotic orbit of
    # the Henon-Heiles system at E=1/8, against documented literature values.
    # We use the well-documented chaotic orbit 'F' of Skokos et al. (2002,
    # arXiv:nlin/0210053, Sect. 3.2): Cartesian (x,y,px,py)=(0,-0.016,0.49974,0)
    # at H=1/8, whose finite-time Lyapunov estimate fluctuates around
    # 10^{-1.2}-10^{-1} for t=10^3-10^5 (their Fig. 4b). The classic
    # computation of Benettin, Galgani & Strelcyn (1976, Phys. Rev. A 14,
    # 2338; their Fig. 4, reproduced as Fig. 2 of Skokos 2010, Lect. Notes
    # Phys. 790, 63) finds the chaotic orbits at E=0.125 saturating at
    # k_n ~ 5-8 x 10^-2. So the documented largest Lyapunov exponent of the
    # chaotic sea at E=1/8 is ~0.04-0.08; here the estimate converges to
    # ~0.04-0.05, asserted below with a loose factor ~2 band, and is clearly
    # separated from the regular orbit 'E' of Skokos et al. (2002)
    # [(x,y,px,py)=(0,0.55,0.2417,0)] at the same energy.
    from galpy.orbit import Orbit
    from galpy.potential import HenonHeilesPotential

    hh = HenonHeilesPotential(amp=1.0)
    ts = numpy.linspace(0.0, 50000.0, 25001)
    # galpy planar coords: x=0,y<0 -> (R,phi)=(|y|,-pi/2), (vx,vy)=(vT,vR)
    oF = Orbit([0.016, 0.0, 0.49974, -numpy.pi / 2.0])
    assert numpy.fabs(oF.E(pot=hh) - 0.125) < 1e-4, (
        "Henon-Heiles chaotic test orbit is not at E=1/8"
    )
    lamF = oF.lyapunov(ts, pot=hh, method="dop853_c")
    # x=0,y>0 -> (R,phi)=(y,pi/2), (vx,vy)=(-vT,vR)
    oE = Orbit([0.55, 0.0, -0.2417, numpy.pi / 2.0])
    assert numpy.fabs(oE.E(pot=hh) - 0.125) < 1e-4, (
        "Henon-Heiles regular test orbit is not at E=1/8"
    )
    lamE = oE.lyapunov(ts, pot=hh, method="dop853_c")
    # Chaotic: converged into the (loose) literature band at both the
    # half-time and the end-time
    for lam_chaotic, tlabel in [(lamF[len(ts) // 2], "t=25000"), (lamF[-1], "t=50000")]:
        assert 0.02 < lam_chaotic < 0.14, (
            "Largest Lyapunov exponent of the documented chaotic Henon-Heiles "
            f"orbit at E=1/8 is {lam_chaotic:g} at {tlabel}, outside the "
            "documented range ~0.04-0.08 (with factor ~2 tolerance)"
        )
    # Regular orbit at the same energy: decaying to ~ln(t)/t
    assert lamE[-1] < 3.0 * numpy.log(ts[-1]) / ts[-1], (
        "lyapunov estimate for the regular Henon-Heiles orbit does not decay "
        f"to ~ln(t)/t: lambda(t_end)={lamE[-1]:g}"
    )
    # and the chaotic exponent is much larger than the regular one
    assert lamF[-1] > 30.0 * lamE[-1], (
        "Chaotic Henon-Heiles lyapunov estimate is not clearly separated from "
        f"the regular one: chaotic {lamF[-1]:g} vs regular {lamE[-1]:g}"
    )
    return None


def test_lyapunov_c_vs_python():
    # The C (dop853_c) and pure-Python (dop853) lyapunov estimates must agree
    # for the same orbit/potential/times; t is short enough (lambda*t ~ 13)
    # that the tightly-toleranced trajectories still shadow each other.
    # Also check that the running estimate is independent of the
    # renormalization interval (validating the Benettin et al. bookkeeping)
    from galpy.orbit import Orbit
    from galpy.potential import HenonHeilesPotential

    hh = HenonHeilesPotential(amp=1.0)
    ic = [0.016, 0.0, 0.49974, -numpy.pi / 2.0]
    ts = numpy.linspace(0.0, 300.0, 3001)
    oc = Orbit(ic)
    lam_c = oc.lyapunov(ts, pot=hh, method="dop853_c", rtol=1e-12, atol=1e-12)
    op = Orbit(ic)
    lam_py = op.lyapunov(ts, pot=hh, method="dop853", rtol=1e-12, atol=1e-12)
    for idx, tlabel in [(len(ts) // 2, "t=150"), (-1, "t=300")]:
        rel = numpy.fabs(lam_c[idx] - lam_py[idx]) / numpy.fabs(lam_c[idx])
        assert rel < 0.02, (
            f"C vs Python lyapunov estimates differ by {rel:g} at {tlabel} "
            f"(C: {lam_c[idx]:g}, Python: {lam_py[idx]:g})"
        )
    # Independence of the renormalization interval
    o37 = Orbit(ic)
    lam_c37 = o37.lyapunov(
        ts, pot=hh, method="dop853_c", renorm_every=37, rtol=1e-12, atol=1e-12
    )
    maxdiff = numpy.nanmax(numpy.fabs(lam_c37 - lam_c))
    assert maxdiff < 1e-4, (
        "lyapunov running estimate depends on the renormalization interval: "
        f"max |renorm_every=37 - renorm_every=10| = {maxdiff:g}"
    )
    return None


def test_lyapunov_api():
    # API behavior of Orbit.lyapunov: output shapes for multiple orbits,
    # per-orbit initial deviations, invariance under the deviation-vector
    # normalization (the variational equations are linear), the pot=None
    # default, physical-output conversion, and Quantity inputs for ts and dt
    from astropy import units

    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential
    from galpy.util import conversion

    ip = IsochronePotential(normalize=1.0, b=0.8)
    ts = numpy.linspace(0.0, 10.0, 101)
    # Quantity ts and dt inputs parse to the same natural-units result
    oq = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0], ro=8.0, vo=220.0)
    tnat_to_Gyr = conversion.time_in_Gyr(220.0, 8.0)
    ts_q = ts * tnat_to_Gyr * units.Gyr
    lam_q = oq.lyapunov(
        ts_q,
        pot=ip,
        method="rk4_c",
        dt=(ts[1] - ts[0]) / 2.0 * tnat_to_Gyr * units.Gyr,
        use_physical=False,
    )
    onat = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0], ro=8.0, vo=220.0)
    lam_nat = onat.lyapunov(
        ts, pot=ip, method="rk4_c", dt=(ts[1] - ts[0]) / 2.0, use_physical=False
    )
    assert numpy.amax(numpy.fabs(lam_q[1:] - lam_nat[1:])) < 1e-10, (
        "lyapunov with Quantity ts/dt does not match the natural-units call"
    )
    # Multiple orbits: shape (2,) -> output (2,nt)
    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.1, 0.0], [1.1, -0.1, 0.9, 0.0, 0.05, 1.0]])
    lam = o.lyapunov(ts, pot=ip, method="dop853_c")
    assert lam.shape == (2, len(ts)), (
        "lyapunov output shape incorrect for multiple orbits"
    )
    # Per-orbit initial deviations: shape (*input_shape,phasedim)
    lam2 = o.lyapunov(ts, pot=ip, method="dop853_c", dxdv0=numpy.ones((2, 6)))
    assert lam2.shape == (2, len(ts)), (
        "lyapunov output shape incorrect for per-orbit dxdv0"
    )
    # The variational equations are linear, so the normalization of dxdv0
    # is irrelevant
    o1 = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    dxdv0 = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lam_a = o1.lyapunov(ts, pot=ip, method="dop853_c", dxdv0=dxdv0)
    lam_b = o1.lyapunov(ts, pot=ip, method="dop853_c", dxdv0=1e-7 * dxdv0)
    assert numpy.allclose(lam_a[1:], lam_b[1:]), (
        "lyapunov estimate depends on the normalization of the initial "
        "deviation vector, but the variational equations are linear"
    )
    # pot=None uses the potential of the last orbit integration
    o1.integrate(ts, ip, method="dop853_c")
    lamd = o1.lyapunov(ts, method="dop853_c", dxdv0=dxdv0)
    assert numpy.allclose(lamd[1:], lam_a[1:]), (
        "lyapunov with pot=None does not use the potential of the last "
        "orbit integration"
    )
    # and the stored orbit integration is not clobbered by lyapunov
    assert numpy.allclose(o1.getOrbit()[-1], o1(ts[-1]).vxvv[0]), (
        "lyapunov clobbered the previously integrated orbit"
    )
    # Physical output: frequency conversion to 1/Gyr
    ro, vo = 8.0, 220.0
    ophys = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0], ro=ro, vo=vo)
    lphys = ophys.lyapunov(ts, pot=ip, method="dop853_c", quantity=False)
    lnat = ophys.lyapunov(ts, pot=ip, method="dop853_c", use_physical=False)
    assert numpy.allclose(lphys[1:], lnat[1:] * conversion.freq_in_Gyr(vo, ro)), (
        "lyapunov physical output is not the natural output converted to 1/Gyr"
    )
    return None


def test_lyapunov_python_fallback_warning():
    # A potential lacking the required C variational implementation must fall
    # back to the Python odeint integrator with a SINGLE galpyWarning (not one
    # warning per renormalization segment)
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.util import galpyWarning

    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    mn.hasC_dxdv3d = False  # force the no-3D-C-Hessian code path
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    ts = numpy.linspace(0.0, 5.0, 51)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lam = o.lyapunov(ts, pot=mn, method="dop853_c", renorm_every=10)
        n_fallback = sum(
            issubclass(wi.category, galpyWarning) and "Using odeint" in str(wi.message)
            for wi in w
        )
    assert n_fallback == 1, (
        "lyapunov should emit exactly one fallback warning when the potential "
        f"lacks adequate C implementations, got {n_fallback}"
    )
    assert lam.shape == (len(ts),), "lyapunov fallback output shape incorrect"
    return None


def test_lyapunov_errors():
    # Input validation of Orbit.lyapunov
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0.8)
    ts = numpy.linspace(0.0, 1.0, 11)
    # Only implemented for phase-space dimensions 4 and 6
    o5 = Orbit([1.0, 0.1, 1.1, 0.1, 0.1])
    with pytest.raises(AttributeError):
        o5.lyapunov(ts, pot=ip)
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    # No potential given and orbit not integrated
    with pytest.raises(AttributeError):
        o.lyapunov(ts)
    # Fewer than two times
    with pytest.raises(ValueError):
        o.lyapunov(numpy.array([0.0]), pot=ip)
    # Invalid renormalization interval
    with pytest.raises(ValueError):
        o.lyapunov(ts, pot=ip, renorm_every=0)
    # Zero-norm initial deviation
    with pytest.raises(ValueError):
        o.lyapunov(ts, pot=ip, dxdv0=numpy.zeros(6))
    # Wrong-dimensionality initial deviation
    with pytest.raises(ValueError):
        o.lyapunov(ts, pot=ip, dxdv0=numpy.ones(4))
    return None


# Tests of Orbit.lyapunov with spectrum=True: the full Lyapunov spectrum from
# QR re-orthonormalization of a full set of deviation vectors (Benettin et al.
# 1980, part 2; Shimada & Nagashima 1979)
def test_lyapunov_spectrum_integrable():
    # In an integrable potential all orbits are regular, so ALL running
    # estimates lambda_i(t) of the Lyapunov spectrum must decay ~ln(t)/t;
    # the spectrum must also obey the sum rule sum_i lambda_i = 0
    # (phase-space volume conservation of the Hamiltonian flow)
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    tend = 500.0
    ts = numpy.linspace(0.0, tend, 5001)
    ip = IsochronePotential(normalize=1.0, b=0.8)
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    lam = o.lyapunov(ts, pot=ip, method="dop853_c", spectrum=True)
    assert lam.shape == (6, len(ts)), (
        "lyapunov spectrum output shape incorrect for a single 3D orbit"
    )
    assert numpy.all(numpy.isnan(lam[:, 0])), (
        "lyapunov spectrum running estimates at ts[0] should be NaN (no elapsed time)"
    )
    maxend = numpy.max(numpy.fabs(lam[:, -1]))
    assert maxend < 3.0 * numpy.log(tend) / tend, (
        "Lyapunov spectrum of a regular 3D isochrone orbit does not decay "
        f"to ~ln(t)/t: max_i |lambda_i(t_end)|={maxend:g}"
    )
    assert maxend < numpy.max(numpy.fabs(lam[:, len(ts) // 2])), (
        "Lyapunov spectrum of a regular 3D isochrone orbit is not decreasing "
        "between the half-time and the end-time"
    )
    # Sum rule at all output times: the flow is symplectic, so the sum of the
    # running estimates is the (zero) log determinant of the deviation-matrix
    # propagator divided by the elapsed time
    maxsum = numpy.nanmax(numpy.fabs(numpy.nansum(lam, axis=0)))
    assert maxsum < 1e-8, (
        "Lyapunov spectrum of a 3D isochrone orbit violates the sum rule "
        f"sum_i lambda_i = 0: max_t |sum_i lambda_i(t)|={maxsum:g}"
    )
    return None


def test_lyapunov_spectrum_henonheiles_chaotic():
    # Full Lyapunov spectrum of the documented chaotic Henon-Heiles orbit 'F'
    # at E=1/8 (see test_lyapunov_henonheiles_chaotic for the provenance of
    # the initial condition and of the documented band of the largest
    # exponent): the largest exponent must lie in the documented band and be
    # consistent with the largest-exponent-only estimate, the two middle
    # exponents must tend to 0 (the flow direction and its symplectic pair),
    # and the spectrum must obey the sum rule sum_i lambda_i = 0 and the
    # symplectic pairing lambda_i = -lambda_{phasedim+1-i}
    from galpy.orbit import Orbit
    from galpy.potential import HenonHeilesPotential

    hh = HenonHeilesPotential(amp=1.0)
    tend = 25000.0
    ts = numpy.linspace(0.0, tend, 12501)
    oF = Orbit([0.016, 0.0, 0.49974, -numpy.pi / 2.0])
    lam = oF.lyapunov(ts, pot=hh, method="dop853_c", spectrum=True)
    assert lam.shape == (4, len(ts)), (
        "lyapunov spectrum output shape incorrect for a single planar orbit"
    )
    # Largest exponent in the documented band of the chaotic sea at E=1/8
    assert 0.02 < lam[0, -1] < 0.14, (
        "Largest exponent of the Lyapunov spectrum of the chaotic "
        f"Henon-Heiles orbit at E=1/8 is {lam[0, -1]:g} at t={tend:g}, "
        "outside the documented range ~0.04-0.08 (with factor ~2 tolerance)"
    )
    # and consistent with the largest-exponent-only (spectrum=False) estimate.
    # Finite-time estimates of a chaotic orbit fluctuate (different initial
    # deviation vectors have different alignment transients, and the deviation
    # vectors enter the adaptive step-size control), so the default-input
    # long-time comparison is loose (both estimates individually sit in the
    # documented band); the TIGHT consistency check seeds the spectrum with
    # the SAME initial deviation vector as the largest-exponent-only run, for
    # which |R_11| accumulation reduces mathematically to single-vector
    # Benettin on the same vector -- agreement at integrator precision
    oF2 = Orbit([0.016, 0.0, 0.49974, -numpy.pi / 2.0])
    lam_max = oF2.lyapunov(ts, pot=hh, method="dop853_c")
    rel = numpy.fabs(lam[0, -1] - lam_max[-1]) / lam_max[-1]
    assert rel < 0.3, (
        "Largest exponent of the Lyapunov spectrum of the chaotic "
        "Henon-Heiles orbit differs from the largest-exponent-only estimate "
        f"by {rel:g} (spectrum: {lam[0, -1]:g}, largest-only: {lam_max[-1]:g})"
    )
    ts_short = numpy.linspace(0.0, 300.0, 301)
    d0 = numpy.ones(4) / 2.0
    oF3 = Orbit([0.016, 0.0, 0.49974, -numpy.pi / 2.0])
    lam_s = oF3.lyapunov(ts_short, pot=hh, method="dop853_c", spectrum=True, dxdv0=d0)
    oF4 = Orbit([0.016, 0.0, 0.49974, -numpy.pi / 2.0])
    lam_max_s = oF4.lyapunov(ts_short, pot=hh, method="dop853_c", dxdv0=d0)
    rel_s = numpy.fabs(lam_s[0, -1] - lam_max_s[-1]) / lam_max_s[-1]
    assert rel_s < 1e-6, (
        "Largest exponent of the Lyapunov spectrum with the same initial "
        "deviation vector differs from the largest-exponent-only estimate "
        f"by {rel_s:g} (spectrum: {lam_s[0, -1]:g}, largest-only: {lam_max_s[-1]:g})"
    )
    # Middle exponents tend to zero like the regular-orbit ~ln(t)/t decay
    midmax = numpy.max(numpy.fabs(lam[1:3, -1]))
    assert midmax < 3.0 * numpy.log(tend) / tend, (
        "Middle exponents of the Lyapunov spectrum of the chaotic "
        f"Henon-Heiles orbit do not tend to zero: max={midmax:g}"
    )
    # Symplectic pairing: lambda_1 = -lambda_4 and lambda_2 = -lambda_3
    pair14 = numpy.fabs(lam[0, -1] + lam[3, -1])
    assert pair14 < 0.02 * lam[0, -1], (
        "Lyapunov spectrum of the chaotic Henon-Heiles orbit violates the "
        f"symplectic pairing lambda_1=-lambda_4: |lambda_1+lambda_4|={pair14:g} "
        f"vs lambda_1={lam[0, -1]:g}"
    )
    pair23 = numpy.fabs(lam[1, -1] + lam[2, -1])
    assert pair23 < 0.02 * lam[0, -1], (
        "Lyapunov spectrum of the chaotic Henon-Heiles orbit violates the "
        f"symplectic pairing lambda_2=-lambda_3: |lambda_2+lambda_3|={pair23:g}"
    )
    # Sum rule at all output times
    maxsum = numpy.nanmax(numpy.fabs(numpy.nansum(lam, axis=0)))
    assert maxsum < 1e-8, (
        "Lyapunov spectrum of the chaotic Henon-Heiles orbit violates the "
        f"sum rule sum_i lambda_i = 0: max_t |sum_i lambda_i(t)|={maxsum:g}"
    )
    # The Benettin/Shimada-Nagashima procedure orders the exponents from
    # largest to smallest by construction (the nearly-degenerate middle pair
    # can transiently swap, so only check across the large gaps)
    assert lam[0, -1] > lam[1, -1] and lam[2, -1] > lam[3, -1], (
        f"Lyapunov spectrum is not ordered from largest to smallest: {lam[:, -1]}"
    )
    return None


def test_lyapunov_spectrum_sumrule_mw():
    # The sum rule sum_i lambda_i = 0 (phase-space volume conservation) for a
    # 3D orbit in MWPotential2014, which exercises the full 3D C Hessian of a
    # composite (bulge+disk+halo) potential
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    ts = numpy.linspace(0.0, 100.0, 1001)
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    lam = o.lyapunov(ts, pot=MWPotential2014, method="dop853_c", spectrum=True)
    assert lam.shape == (6, len(ts)), (
        "lyapunov spectrum output shape incorrect for a 3D MWPotential2014 orbit"
    )
    maxsum = numpy.nanmax(numpy.fabs(numpy.nansum(lam, axis=0)))
    assert maxsum < 1e-8, (
        "Lyapunov spectrum of a 3D MWPotential2014 orbit violates the sum "
        f"rule sum_i lambda_i = 0: max_t |sum_i lambda_i(t)|={maxsum:g}"
    )
    relsum = numpy.fabs(numpy.sum(lam[:, -1])) / numpy.max(numpy.fabs(lam[:, -1]))
    assert relsum < 1e-6, (
        "Lyapunov spectrum of a 3D MWPotential2014 orbit violates the sum "
        f"rule sum_i lambda_i = 0: |sum_i lambda_i|/max_i |lambda_i|={relsum:g} "
        "at the end time"
    )
    return None


def test_lyapunov_spectrum_consistency():
    # When dxdv0= is supplied together with spectrum=True, it is used as the
    # FIRST deviation vector (completed to an orthonormal basis); because the
    # QR orthonormalization never changes the direction of the first vector,
    # the running estimate of the largest exponent is then mathematically
    # identical to the largest-exponent-only (spectrum=False) estimate with
    # the same dxdv0. Also check that the C (dop853_c) and pure-Python
    # (dop853) spectra agree
    from galpy.orbit import Orbit
    from galpy.potential import HenonHeilesPotential

    hh = HenonHeilesPotential(amp=1.0)
    ic = [0.016, 0.0, 0.49974, -numpy.pi / 2.0]
    ts = numpy.linspace(0.0, 300.0, 3001)
    dxdv0 = numpy.array([1.0, 0.0, 0.0, 0.0])
    oa = Orbit(ic)
    lam_max = oa.lyapunov(ts, pot=hh, method="dop853_c", dxdv0=dxdv0)
    ob = Orbit(ic)
    lam_spec = ob.lyapunov(ts, pot=hh, method="dop853_c", dxdv0=dxdv0, spectrum=True)
    maxdiff = numpy.nanmax(numpy.fabs(lam_spec[0] - lam_max))
    assert maxdiff < 1e-6, (
        "Largest exponent of the Lyapunov spectrum with dxdv0= as the first "
        "deviation vector differs from the largest-exponent-only estimate "
        f"with the same dxdv0: max_t |diff|={maxdiff:g}"
    )
    # C vs Python parity of the full spectrum
    ts2 = numpy.linspace(0.0, 100.0, 1001)
    oc = Orbit(ic)
    lam_c = oc.lyapunov(
        ts2, pot=hh, method="dop853_c", spectrum=True, rtol=1e-12, atol=1e-12
    )
    op = Orbit(ic)
    lam_py = op.lyapunov(
        ts2, pot=hh, method="dop853", spectrum=True, rtol=1e-12, atol=1e-12
    )
    maxdiff_cpy = numpy.max(numpy.fabs(lam_c[:, -1] - lam_py[:, -1]))
    assert maxdiff_cpy < 1e-8, (
        "C vs Python Lyapunov spectra differ at the end time: "
        f"max_i |diff|={maxdiff_cpy:g}"
    )
    return None


def test_lyapunov_spectrum_api():
    # API behavior of Orbit.lyapunov with spectrum=True: output shapes for
    # multiple orbits and per-orbit dxdv0, physical-output conversion of the
    # full spectrum, the single-warning Python fallback, and input validation
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, MiyamotoNagaiPotential
    from galpy.util import conversion, galpyWarning

    ip = IsochronePotential(normalize=1.0, b=0.8)
    ts = numpy.linspace(0.0, 10.0, 101)
    # Multiple orbits: shape (2,) -> output (2,phasedim,nt)
    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.1, 0.0], [1.1, -0.1, 0.9, 0.0, 0.05, 1.0]])
    lam = o.lyapunov(ts, pot=ip, method="dop853_c", spectrum=True)
    assert lam.shape == (2, 6, len(ts)), (
        "lyapunov spectrum output shape incorrect for multiple orbits"
    )
    assert numpy.all(numpy.isnan(lam[:, :, 0])), (
        "lyapunov spectrum running estimates at ts[0] should be NaN"
    )
    # Per-orbit initial deviations: shape (*input_shape,phasedim)
    lam2 = o.lyapunov(
        ts, pot=ip, method="dop853_c", spectrum=True, dxdv0=numpy.ones((2, 6))
    )
    assert lam2.shape == (2, 6, len(ts)), (
        "lyapunov spectrum output shape incorrect for per-orbit dxdv0"
    )
    # Physical output: each exponent is a frequency, converted to 1/Gyr
    ro, vo = 8.0, 220.0
    ophys = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0], ro=ro, vo=vo)
    lphys = ophys.lyapunov(ts, pot=ip, method="dop853_c", spectrum=True, quantity=False)
    lnat = ophys.lyapunov(
        ts, pot=ip, method="dop853_c", spectrum=True, use_physical=False
    )
    assert lphys.shape == (6, len(ts)), (
        "lyapunov spectrum physical output shape incorrect"
    )
    assert numpy.allclose(lphys[:, 1:], lnat[:, 1:] * conversion.freq_in_Gyr(vo, ro)), (
        "lyapunov spectrum physical output is not the natural output converted to 1/Gyr"
    )
    # Python fallback: a potential lacking the required 3D C Hessian must
    # fall back to odeint with a SINGLE galpyWarning, also for spectrum=True
    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    mn.hasC_dxdv3d = False  # force the no-3D-C-Hessian code path
    o1 = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    ts2 = numpy.linspace(0.0, 5.0, 51)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lamf = o1.lyapunov(
            ts2, pot=mn, method="dop853_c", renorm_every=10, spectrum=True
        )
        n_fallback = sum(
            issubclass(wi.category, galpyWarning) and "Using odeint" in str(wi.message)
            for wi in w
        )
    assert n_fallback == 1, (
        "lyapunov spectrum should emit exactly one fallback warning when the "
        f"potential lacks adequate C implementations, got {n_fallback}"
    )
    assert lamf.shape == (6, len(ts2)), (
        "lyapunov spectrum fallback output shape incorrect"
    )
    # Input validation: zero-norm dxdv0 also raises with spectrum=True
    with pytest.raises(ValueError):
        o1.lyapunov(ts2, pot=ip, spectrum=True, dxdv0=numpy.zeros(6))
    return None


def test_liouville_3d_nonaxi_flow():
    # Validate the NON-axisymmetric Hessian terms (phi2deriv/Rphideriv/zphideriv)
    # of the 3D variational RHS, which the axisymmetric MiyamotoNagai/Plummer
    # parametrization of test_liouville_3d never exercises (those have
    # phi2deriv==Rphideriv==zphideriv==0 identically). We use a genuinely triaxial
    # potential and the pure-Python path (no non-axisymmetric potential has a
    # complete 3D *C* Hessian yet), and check the dxdv columns against a
    # finite-difference of the flow -- which pins the signs/scales of the new
    # z-phi coupling terms (a wrong sign there would break the FD agreement even
    # though it leaves det(M)=1 and symplecticity intact).
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, evaluatephizderivs
    from galpy.util import coords

    pot = LogarithmicHaloPotential(normalize=1.0, b=0.7, q=0.8)
    ic = [1.0, 0.1, 1.1, 0.05, 0.08, 0.3]
    times = numpy.linspace(0.0, 3.0, 151)
    # Guard against a vacuous test: the non-axisymmetric coupling must be nonzero
    # along this orbit, otherwise the new terms are multiplied by 0 as before.
    assert numpy.fabs(evaluatephizderivs(pot, 1.0, 0.05, phi=0.3)) > 1e-3, (
        "test potential must have a nonzero d2Phi/dz/dphi to exercise zphideriv"
    )
    method = "dop853"
    rtol = atol = 1e-12
    eps = 1e-7
    canonical = numpy.eye(6)
    obase = Orbit(ic)
    obase.integrate(times, pot, method=method)
    base_rect = _orbit_rect_3d(obase, times)
    for ii in range(6):  # all six columns -> the full Cartesian Hessian K
        pert_ic_rect = base_rect[0].copy()
        pert_ic_rect[ii] += eps
        Rp, phip, Zp = coords.rect_to_cyl(
            pert_ic_rect[0], pert_ic_rect[1], pert_ic_rect[2]
        )
        vRp, vTp, vzp = coords.rect_to_cyl_vec(
            pert_ic_rect[3],
            pert_ic_rect[4],
            pert_ic_rect[5],
            pert_ic_rect[0],
            pert_ic_rect[1],
            pert_ic_rect[2],
        )
        opert = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
        opert.integrate(times, pot, method=method)
        fd = (_orbit_rect_3d(opert, times) - base_rect) / eps
        odx = Orbit(ic)
        odx.integrate_dxdv(
            canonical[ii],
            times,
            pot,
            method=method,
            rectIn=True,
            rectOut=True,
            rtol=rtol,
            atol=atol,
        )
        col = numpy.asarray(odx.getOrbit_dxdv())
        fderr = numpy.amax(numpy.fabs(fd - col))
        assert fderr < 1e-4, (
            f"non-axisymmetric 3D finite-difference of the flow for e_{ii} differs "
            f"from the dxdv column by {fderr:g} (checks phi2deriv/Rphideriv/zphideriv)"
        )
    return None


def test_liouville_planar():
    if _NOLONGINTEGRATIONS:
        return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 28.0, 1001)  # ~1 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "dop853_c",
        "dop853",
        "odeint",  # direct python solver
        "rk4_c",
        "rk6_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("mockFlatEllipticalDiskPotential")
    pots.append("mockFlatLopsidedDiskPotential")
    pots.append("mockFlatCosmphiDiskPotential")
    pots.append("mockFlatCosmphiDiskwBreakPotential")
    pots.append("mockSlowFlatEllipticalDiskPotential")
    pots.append("mockFlatDehnenBarPotential")
    pots.append("mockFlatDehnenBarPotential")
    pots.append("mockSlowFlatDehnenBarPotential")
    pots.append("mockFlatSoftenedNeedleBarPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("BurkertPotentialNoC")
    pots.append("NFWTwoPowerTriaxialPotential")  # for planar-from-full
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("mockFlatSpiralArmsPotential")
    pots.append("mockRotatingFlatSpiralArmsPotential")
    pots.append("mockSpecialRotatingFlatSpiralArmsPotential")
    # pots.append('mockFlatSteadyLogSpiralPotential')
    # pots.append('mockFlatTransientLogSpiralPotential')
    pots.append("mockFlatDehnenSmoothBarPotential")
    pots.append("mockSlowFlatDehnenSmoothBarPotential")
    pots.append("mockSlowFlatDecayingDehnenSmoothBarPotential")
    pots.append("mockFlatSolidBodyRotationSpiralArmsPotential")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("testorbitHenonHeilesPotential")
    pots.append("mockFlatTrulyCorotatingRotationSpiralArmsPotential")
    pots.append("mockFlatTrulyGaussianAmplitudeBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("testNullPotential")
    pots.append("mockKuzminLikeWrapperPotential")
    pots.append("mockMultipoleExpansionSphericalPotential")
    pots.append("mockMultipoleExpansionAxiPotential")
    pots.append("mockMultipoleExpansionPotential")
    pots.append("mockMultipoleExpansionLimitedGridPotential")
    pots.append("mockTDMultipoleExpansionLimitedGridPotential")
    pots.append("mockFlatWeaklyTDNonaxiM3MultipoleExpansionPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    # rmpots.append('BurkertPotential')
    # Don't have C implementations of the relevant 2nd derivatives
    # (DoubleExponentialDiskPotential now wires the planar R2deriv in C)
    rmpots.append("RazorThinExponentialDiskPotential")
    # Doesn't have C at all
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    # rmpots.append('PowerSphericalPotentialwCutoff')
    # SoftenedNeedleBarPotential now HAS the full analytic Hessian (Python and C;
    # the planar variational equations are exercised here through the realistic
    # mockFlatSoftenedNeedleBarPotential halo+bar configuration appended above, and
    # the Hessian values are pinned by the dedicated
    # test_softenedneedlebar_planar_dxdv_* tests and the liouville3d_registry). The
    # BARE normalized potential -- the entire flat rotation curve generated by a
    # fast-rotating (Omega_b=1.8) needle -- makes the fixed IC here strongly
    # chaotic (||STM|| ~ 6e4 over the 28-time-unit horizon, Lyapunov time ~2.5), so
    # |det M - 1| saturates at the double-precision cancellation floor (~0.1 for
    # the adaptive C integrators, ~4e4 for default-tolerance odeint) REGARDLESS of
    # Hessian correctness; no meaningful det-tolerance exists, hence it stays out.
    rmpots.append("SoftenedNeedleBarPotential")
    # Doesn't have the R2deriv
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -8.0
    tol["KeplerPotential"] = -6.5  # more difficult
    tol["MN3ExponentialDiskPotential"] = -7.0  # more difficult
    tol["NFWPotential"] = -6.0  # more difficult for rk4_c, only one that does this
    tol["TriaxialNFWPotential"] = -4.0  # more difficult
    tol["triaxialLogarithmicHaloPotential"] = -7.0  # more difficult
    tol["FerrersPotential"] = -2.0
    # numerical Ogata/Hankel-quadrature forces -> the adaptive C integrators reach
    # ~1.5e-8 over ~1 Gyr (a quadrature-accuracy effect, not a Hessian error)
    tol["DoubleExponentialDiskPotential"] = -7.0
    tol["HomogeneousSpherePotential"] = -4.0
    tol["KingPotential"] = -6.0
    tol["mockInterpSphericalPotential"] = -4.0  # == HomogeneousSpherePotential
    tol["mockFlatCosmphiDiskwBreakPotential"] = -7.0  # more difficult
    # rotating halo+bar: the fixed-step rk4_c reaches ~3e-7 over the ~1 Gyr horizon
    tol["mockFlatSoftenedNeedleBarPotential"] = -6.0
    tol["mockFlatTrulyCorotatingRotationSpiralArmsPotential"] = -5.0  # more difficult
    tol["mockMultipoleExpansionPotential"] = -6.5
    tol["mockMultipoleExpansionLimitedGridPotential"] = -5.0
    tol["mockTDMultipoleExpansionLimitedGridPotential"] = -4.0
    tol["mockFlatWeaklyTDNonaxiM3MultipoleExpansionPotential"] = -4.0
    # grid-spline-interpolated embedded Multipole part -> grid-level accuracy
    tol["DiskMultipoleExpansionPotential"] = -5.5
    tol["DiskSCFPotential"] = -7.0  # more difficult
    firstTest = True
    for p in pots:
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        # if not p == 'NFWPotential' and not p == 'mockSlowFlatDecayingDehnenSmoothBarPotential': continue
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        for integrator in integrators:
            if p in list(tol.keys()):
                ttol = tol[p]
            else:
                ttol = tol["default"]
            if isinstance(tp, testMWPotential) or isinstance(tp, testplanarMWPotential):
                thasC = _check_c(tp._potlist, dxdv=True)
            else:
                thasC = _check_c(tp, dxdv=True)
            if (
                (integrator == "odeint" or not thasC)
                and not p == "FerrersPotential"
                and not p == "MultipoleExpansionPotential"
                and not p == "DoubleExponentialDiskPotential"
            ):
                ttol = -4.0
            elif (
                integrator == "odeint" or not thasC
            ) and p == "MultipoleExpansionPotential":
                ttol = -3.0
            elif (
                integrator == "odeint" or not thasC
            ) and p == "DoubleExponentialDiskPotential":
                # pure-Python odeint variational integration of the numerical
                # Hankel-quadrature forces drifts to ~9e-4 over ~1 Gyr (integrator/
                # quadrature accuracy, not a Hessian error; the C integrators reach
                # ~1.5e-8, see tol[] above)
                ttol = -2.5
            if True:
                ttimes = times
            o = setup_orbit_liouville(ptp, axi=False, henon="Henon" in p)
            # Calculate the Jacobian d x / d x
            if hasattr(tp, "_potlist"):
                if isinstance(tp, testMWPotential):
                    plist = potential.toPlanarPotential(tp._potlist)
                else:
                    plist = tp._potlist
                o.integrate_dxdv(
                    [1.0, 0.0, 0.0, 0.0],
                    ttimes,
                    plist,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dx = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 1.0, 0.0, 0.0],
                    ttimes,
                    plist,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dy = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 0.0, 1.0, 0.0],
                    ttimes,
                    plist,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dvx = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 0.0, 0.0, 1.0],
                    ttimes,
                    plist,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dvy = o.getOrbit_dxdv()[-1, :]
            else:
                o.integrate_dxdv(
                    [1.0, 0.0, 0.0, 0.0],
                    ttimes,
                    ptp,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dx = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 1.0, 0.0, 0.0],
                    ttimes,
                    ptp,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dy = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 0.0, 1.0, 0.0],
                    ttimes,
                    ptp,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dvx = o.getOrbit_dxdv()[-1, :]
                o.integrate_dxdv(
                    [0.0, 0.0, 0.0, 1.0],
                    ttimes,
                    ptp,
                    method=integrator,
                    rectIn=True,
                    rectOut=True,
                )
                dvy = o.getOrbit_dxdv()[-1, :]
            tjac = numpy.linalg.det(numpy.array([dx, dy, dvx, dvy]))
            # print(p, integrator, numpy.fabs(tjac-1.),ttol)
            assert numpy.fabs(tjac - 1.0) < 10.0**ttol, (
                f"Liouville theorem jacobian differs from one by {numpy.fabs(tjac - 1.0):g} for {p} and integrator {integrator}"
            )
            if firstTest or ("Burkert" in p and not ptp.hasC):
                # Some one time tests
                # Test non-rectangular in- and output
                try:
                    o.integrate_dxdv(
                        [0.0, 0.0, 0.0, 1.0],
                        ttimes,
                        ptp,
                        method="leapfrog",
                        rectIn=True,
                        rectOut=True,
                    )
                except ValueError:
                    pass
                else:
                    raise AssertionError(
                        "integrate_dxdv with symplectic integrator should have raised ValueError, but didn't"
                    )
                firstTest = False
            if _QUICKTEST and not (
                ("NFW" in p and not ptp.isNonAxi and "SCF" not in p)
                or ("Burkert" in p and not ptp.hasC)
            ):
                break
    return None


def test_spiralarms_planar_dxdv_c_vs_python():
    # Regression test for a bug in the C SpiralArmsPotentialPlanarR2deriv (the planar
    # d^2Phi/dR^2 used by the C planar integrate_dxdv variational RHS): a stray extra
    # 1/Kn factor made the in-plane tidal tensor wrong by ~3e-4. It was invisible to
    # test_liouville_planar because det(M)=1 and symplecticity hold for ANY symmetric
    # K; only comparing the C dxdv path against the (correct) pure-Python path exposes
    # it. The C (dopr54_c) and Python (dop853) planar dxdv integrations must agree.
    from galpy.orbit import Orbit
    from galpy.potential import SpiralArmsPotential

    pot = SpiralArmsPotential()
    times = numpy.linspace(0.0, 4.0, 401)
    ic = [1.0, 0.1, 1.1, 0.3]  # planar [R, vR, vT, phi]
    # Use a UNIT deviation: the variational equation is linear in the deviation, so a
    # unit dx makes the STM column O(1) and the ~2.6e-4 *relative* PlanarR2deriv error
    # show up as an O(2.6e-4) absolute discrepancy (a tiny dx would scale it below tol).
    dev = [1.0, 0.0, 0.0, 0.0]
    o_c = Orbit(ic)
    o_c.integrate_dxdv(
        dev,
        times,
        pot,
        method="dopr54_c",
        rectIn=True,
        rectOut=True,
        rtol=1e-12,
        atol=1e-12,
    )
    o_py = Orbit(ic)
    o_py.integrate_dxdv(
        dev,
        times,
        pot,
        method="dop853",
        rectIn=True,
        rectOut=True,
        rtol=1e-12,
        atol=1e-12,
    )
    dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
    dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
    # Pre-fix the C path disagreed with the (correct) Python path by ~2.6e-4.
    assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-6, (
        "C planar dxdv (SpiralArms) disagrees with the pure-Python result: "
        f"max diff {numpy.amax(numpy.fabs(dev_c - dev_py)):g} (PlanarR2deriv bug?)"
    )
    return None


def test_twopower_planar_dxdv_c_vs_python():
    # Regression test for a bug in the C TwoPowerSphericalPotentialPlanarR2deriv (the
    # planar d^2Phi/dR^2 used by the C planar integrate_dxdv variational RHS): two terms
    # had the wrong power of R (R^(-alpha-1) instead of R^(-alpha), and R^(-alpha)
    # instead of R^(1-alpha)), making the in-plane tidal tensor wrong by ~7% (the error
    # vanishes at R=1, where the wrong powers coincide). The pure-Python _R2deriv is
    # correct (matches a finite-difference of -Rforce to ~1e-10). Only the BARE
    # TwoPowerSphericalPotential is affected; its Dehnen/DehnenCore/Hernquist/NFW/Jaffe
    # special-case subclasses use their own (correct) PlanarR2deriv. Comparing the C
    # (dopr54_c / dop853_c) planar dxdv path against the (correct) pure-Python (dop853)
    # path exposes it.
    from galpy.orbit import Orbit
    from galpy.potential import TwoPowerSphericalPotential

    # Non-degenerate (alpha, beta): must be the bare TwoPowerSphericalPotential, NOT a
    # special-case subclass dispatch (which would use a different, correct C R2deriv).
    pot = TwoPowerSphericalPotential(
        amp=1.0, a=1.5, alpha=1.0, beta=4.5, normalize=True
    )
    assert pot._specialSelf is None, (
        "test must use the bare TwoPowerSphericalPotential, not a subclass dispatch"
    )
    times = numpy.linspace(0.0, 4.0, 401)
    ic = [1.0, 0.1, 1.1, 0.3]  # planar [R, vR, vT, phi]
    # Use a UNIT deviation: the variational equation is linear in the deviation, so a
    # unit dx makes the STM column O(1) and the ~7% *relative* PlanarR2deriv error show
    # up as an O(0.07) absolute discrepancy (a tiny dx would scale it below tol).
    dev = [1.0, 0.0, 0.0, 0.0]
    for method in ["dopr54_c", "dop853_c"]:
        o_c = Orbit(ic)
        o_c.integrate_dxdv(
            dev,
            times,
            pot,
            method=method,
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        o_py = Orbit(ic)
        o_py.integrate_dxdv(
            dev,
            times,
            pot,
            method="dop853",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
        dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
        # Pre-fix the C path disagreed with the (correct) Python path by ~0.86.
        assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-6, (
            "C planar dxdv (TwoPowerSpherical) disagrees with the pure-Python result "
            f"(method={method}): max diff "
            f"{numpy.amax(numpy.fabs(dev_c - dev_py)):g} (PlanarR2deriv bug?)"
        )
    return None


def test_flattenedpower_planar_dxdv_c_vs_python():
    # Regression test for a bug in the C FlattenedPowerPotentialPlanarR2deriv (the
    # planar d^2Phi/dR^2 used by the C planar integrate_dxdv variational RHS): it used
    # (alpha + 1.) where the correct coefficient is (alpha + 2.) -- the pure-Python
    # _R2deriv is correct (matches a finite-difference of -Rforce to ~1e-10). The error
    # vanishes at R such that the two coincide, but is O(1) relative elsewhere, making
    # the in-plane tidal tensor (and hence the planar STM) wrong. test_liouville_planar
    # CANNOT catch this: det M = 1 holds for ANY symmetric Hessian (tr J = 0 regardless
    # of the R2deriv value), so it is insensitive to a wrong-but-symmetric R2deriv.
    # Comparing the C (dopr54_c / dop853_c) planar dxdv path against the (correct)
    # pure-Python (dop853) path exposes it.
    from galpy.orbit import Orbit
    from galpy.potential import FlattenedPowerPotential

    pot = FlattenedPowerPotential(amp=1.0, alpha=0.5, core=0.8, q=0.9, normalize=True)
    times = numpy.linspace(0.0, 4.0, 401)
    ic = [1.2, 0.1, 1.1, 0.3]  # planar [R, vR, vT, phi]; R != 1 so the bug is active
    # Use a UNIT deviation: the variational equation is linear in the deviation, so a
    # unit dx makes the STM column O(1) and the relative PlanarR2deriv error shows up as
    # an O(1) absolute discrepancy (a tiny dx would scale it below tol).
    dev = [1.0, 0.0, 0.0, 0.0]
    for method in ["dopr54_c", "dop853_c"]:
        o_c = Orbit(ic)
        o_c.integrate_dxdv(
            dev,
            times,
            pot,
            method=method,
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        o_py = Orbit(ic)
        o_py.integrate_dxdv(
            dev,
            times,
            pot,
            method="dop853",
            rectIn=True,
            rectOut=True,
            rtol=1e-12,
            atol=1e-12,
        )
        dev_c = numpy.asarray(o_c.getOrbit_dxdv())[-1]
        dev_py = numpy.asarray(o_py.getOrbit_dxdv())[-1]
        # Pre-fix the C path disagreed with the (correct) Python path by ~1.4.
        assert numpy.amax(numpy.fabs(dev_c - dev_py)) < 1e-6, (
            "C planar dxdv (FlattenedPower) disagrees with the pure-Python result "
            f"(method={method}): max diff "
            f"{numpy.amax(numpy.fabs(dev_c - dev_py)):g} (PlanarR2deriv bug?)"
        )
    return None


def test_softenedneedlebar_planar_dxdv_c_vs_python():
    # SoftenedNeedleBarPotential's planar Hessian (PlanarR2deriv/Planarphi2deriv/
    # PlanarRphideriv) in C must reproduce the pure-Python analytic reference for
    # UNIT-magnitude deviations (det(M)=1 in test_liouville_planar is necessary but
    # not sufficient: it holds for ANY symmetric in-plane Hessian, so only this
    # C-vs-Python comparison pins the planar Hessian VALUES). The rotating bar
    # (omegab != 0) also exercises the time-dependent phi - pa - omegab t angular
    # dependence of the C planar Hessian.
    from galpy.orbit import Orbit
    from galpy.potential import SoftenedNeedleBarPotential

    pot = SoftenedNeedleBarPotential(
        a=4.0, b=0.5, c=1.0, pa=0.4, omegab=1.8, normalize=True
    )
    assert pot.hasC_dxdv, "SoftenedNeedleBar should advertise hasC_dxdv"
    times = numpy.linspace(0.0, 4.0, 401)
    ic = [1.0, 0.1, 1.1, 0.3]  # planar [R, vR, vT, phi]
    canonical = numpy.eye(4)
    maxdiff = 0.0
    for ii in range(4):  # all four columns -> the full planar STM
        o_c = Orbit(ic)
        o_c.integrate_dxdv(
            canonical[ii], times, pot, method="dopr54_c",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        o_py = Orbit(ic)
        o_py.integrate_dxdv(
            canonical[ii], times, pot, method="dop853",
            rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
        )  # fmt: skip
        diff = numpy.amax(numpy.fabs(o_c.getOrbit_dxdv() - o_py.getOrbit_dxdv()))
        maxdiff = max(maxdiff, diff)
    assert maxdiff < 1e-6, (
        "C planar dxdv (SoftenedNeedleBar) disagrees with the pure-Python result: "
        f"max diff {maxdiff:g} (Planar Hessian bug?)"
    )
    return None


def test_softenedneedlebar_planar_dxdv_fd_of_flow():
    # Planar finite-difference-of-the-flow STM check for SoftenedNeedleBarPotential:
    # column i of the STM must equal (x(t; x0 + eps e_i) - x(t; x0))/eps, where the
    # perturbed orbits use ONLY the (long-trusted) forces -- so this validates the
    # new planar Hessian against the forces through the actual dynamics, in both the
    # C (dopr54_c) and pure-Python (dop853) variational integrators.
    from galpy.orbit import Orbit
    from galpy.potential import SoftenedNeedleBarPotential
    from galpy.util import coords

    pot = SoftenedNeedleBarPotential(
        a=4.0, b=0.5, c=1.0, pa=0.4, omegab=1.8, normalize=True
    )
    times = numpy.linspace(0.0, 4.0, 201)
    ic = [1.0, 0.1, 1.1, 0.3]  # planar [R, vR, vT, phi]
    eps = 1e-7
    canonical = numpy.eye(4)
    for method in ["dopr54_c", "dop853"]:
        obase = Orbit(ic)
        obase.integrate(times, pot, method=method)
        base_rect = numpy.array(
            [obase.x(times), obase.y(times), obase.vx(times), obase.vy(times)]
        ).T
        for ii in range(4):
            pert = base_rect[0].copy()
            pert[ii] += eps
            Rp, phip, _ = coords.rect_to_cyl(pert[0], pert[1], 0.0)
            vRp, vTp, _ = coords.rect_to_cyl_vec(
                pert[2], pert[3], 0.0, pert[0], pert[1], 0.0
            )
            opert = Orbit([Rp, vRp, vTp, phip])
            opert.integrate(times, pot, method=method)
            pert_rect = numpy.array(
                [opert.x(times), opert.y(times), opert.vx(times), opert.vy(times)]
            ).T
            fd = (pert_rect - base_rect) / eps
            odx = Orbit(ic)
            odx.integrate_dxdv(
                canonical[ii], times, pot, method=method,
                rectIn=True, rectOut=True, rtol=1e-12, atol=1e-12,
            )  # fmt: skip
            col = numpy.asarray(odx.getOrbit_dxdv())
            fderr = numpy.amax(numpy.fabs(fd - col))
            assert fderr < 1e-4, (
                f"planar finite-difference of the flow for e_{ii} differs from the "
                f"dxdv column by {fderr:g} for SoftenedNeedleBar, method {method}"
            )
    return None


# Test that integrating an orbit in MWPotential2014 using integrate_SOS conserves energy
def test_integrate_SOS_3D():
    pot = potential.MWPotential2014
    o = setup_orbit_energy(pot, axi=True)
    psis = numpy.linspace(0.0, 20.0 * numpy.pi, 1001)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.integrate_SOS(psis, pot, method=method)
        Es = o.E(o.t)
        assert (numpy.std(Es) / numpy.mean(Es)) ** 2.0 < 10.0**-10, (
            f"Energy is not conserved by integrate_sos for method={method}"
        )
    return None


# Test that the 3D SOS function returns points with z=0, vz > 0
def test_SOS_3D():
    pot = potential.MWPotential2014
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
        )
        zs = o.z(o.t)
        vzs = o.vz(o.t)
        assert (numpy.fabs(zs) < 10.0**-7.0).all(), (
            f"z on SOS is not zero for integrate_sos for method={method}"
        )
        assert (vzs > 0.0).all(), (
            f"vz on SOS is not positive for integrate_sos for method={method}"
        )
    return None


# Test that the 3D bruteSOS function returns points with z=0, vz > 0
def test_bruteSOS_3D():
    pot = potential.MWPotential2014
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
        )
        zs = o.z(o.t)
        vzs = o.vz(o.t)
        assert (numpy.fabs(zs) < 10.0**-3.0).all(), (
            f"z on SOS is not zero for bruteSOS for method={method}"
        )
        assert (vzs > 0.0).all(), (
            f"vz on SOS is not zero for bruteSOS for method={method}"
        )
    return None


# Test that Orbit.integrate accepts per-orbit time arrays (3D)
def test_integrate_indiv_t_3D():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    # Three different starting positions and three different time windows
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
            [1.2, -0.05, 0.9, -0.1, 0.1, 0.5],
            [0.8, 0.0, 1.0, 0.2, -0.05, 1.0],
        ]
    )
    nt = 401
    ts_indiv = numpy.array(
        [
            numpy.linspace(0.0, 5.0, nt),
            numpy.linspace(0.0, 7.0, nt),
            numpy.linspace(0.0, 9.0, nt),
        ]
    )
    for method in ["dop853_c", "dopr54_c", "rk4_c", "symplec4_c", "dop853", "odeint"]:
        # Batched per-orbit-t integration
        o_batch = Orbit(vxvvs)
        o_batch.integrate(ts_indiv, pot, method=method)
        # Reference: integrate each orbit separately
        for ii in range(len(vxvvs)):
            o_one = Orbit(vxvvs[ii])
            o_one.integrate(ts_indiv[ii], pot, method=method)
            # The batched per-orbit-t and single-orbit code paths feed
            # bit-for-bit identical inputs into the same integrator, so the
            # outputs should match exactly. (o_one stores its single orbit
            # with a leading size-1 axis, so compare to o_one.orbit[0].)
            assert numpy.array_equal(o_batch.orbit[ii], o_one.orbit[0]), (
                f"Per-orbit integration disagrees with single-orbit "
                f"integration for orbit {ii}, method={method}"
            )
    return None


# Test that Orbit.integrate accepts per-orbit time arrays (2D)
def test_integrate_indiv_t_2D():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.0],
            [1.2, -0.05, 0.9, 0.5],
            [0.8, 0.0, 1.0, 1.0],
        ]
    )
    nt = 401
    ts_indiv = numpy.array(
        [
            numpy.linspace(0.0, 5.0, nt),
            numpy.linspace(0.0, 7.0, nt),
            numpy.linspace(0.0, 9.0, nt),
        ]
    )
    for method in ["dop853_c", "dopr54_c", "rk4_c", "symplec4_c", "dop853", "odeint"]:
        o_batch = Orbit(vxvvs)
        o_batch.integrate(ts_indiv, pot, method=method)
        for ii in range(len(vxvvs)):
            o_one = Orbit(vxvvs[ii])
            o_one.integrate(ts_indiv[ii], pot, method=method)
            # The batched per-orbit-t and single-orbit code paths feed
            # bit-for-bit identical inputs into the same integrator, so the
            # outputs should match exactly. (o_one stores its single orbit
            # with a leading size-1 axis, so compare to o_one.orbit[0].)
            assert numpy.array_equal(o_batch.orbit[ii], o_one.orbit[0]), (
                f"Per-orbit integration disagrees with single-orbit "
                f"integration for orbit {ii}, method={method}"
            )
    return None


# Test that Orbit.integrate accepts per-orbit time arrays (1D)
def test_integrate_indiv_t_1D():
    from galpy.orbit import Orbit

    pot = potential.IsothermalDiskPotential(amp=1.0, sigma=1.0)
    vxvvs = numpy.array([[0.5, 0.1], [-0.3, 0.2], [0.8, -0.1]])
    nt = 401
    ts_indiv = numpy.array(
        [
            numpy.linspace(0.0, 5.0, nt),
            numpy.linspace(0.0, 7.0, nt),
            numpy.linspace(0.0, 9.0, nt),
        ]
    )
    for method in ["dop853_c", "dopr54_c", "rk4_c", "symplec4_c", "dop853", "odeint"]:
        o_batch = Orbit(vxvvs)
        o_batch.integrate(ts_indiv, pot, method=method)
        for ii in range(len(vxvvs)):
            o_one = Orbit(vxvvs[ii])
            o_one.integrate(ts_indiv[ii], pot, method=method)
            # The batched per-orbit-t and single-orbit code paths feed
            # bit-for-bit identical inputs into the same integrator, so the
            # outputs should match exactly. (o_one stores its single orbit
            # with a leading size-1 axis, so compare to o_one.orbit[0].)
            assert numpy.array_equal(o_batch.orbit[ii], o_one.orbit[0]), (
                f"Per-orbit integration disagrees with single-orbit "
                f"integration for orbit {ii}, method={method}"
            )
    return None


# Test per-orbit time arrays work for an Orbit with non-trivial leading shape
def test_integrate_indiv_t_3D_2Dshape():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    # Orbit with shape (2, 2)
    vxvvs = numpy.array(
        [
            [[1.0, 0.1, 1.1, 0.1, 0.05, 0.0], [1.05, 0.0, 1.0, 0.05, 0.05, 0.3]],
            [[1.2, -0.05, 0.9, -0.1, 0.1, 0.5], [0.95, 0.05, 1.05, 0.0, -0.05, 0.7]],
        ]
    )
    nt = 201
    # ts shape (2, 2, nt)
    ts_indiv = numpy.empty((2, 2, nt))
    for i in range(2):
        for j in range(2):
            ts_indiv[i, j] = numpy.linspace(0.0, 3.0 + 0.5 * (2 * i + j), nt)
    o_batch = Orbit(vxvvs)
    assert o_batch.shape == (2, 2)
    o_batch.integrate(ts_indiv, pot, method="dop853_c")
    # Compare to per-orbit single integrations
    for i in range(2):
        for j in range(2):
            o_one = Orbit(vxvvs[i, j])
            o_one.integrate(ts_indiv[i, j], pot, method="dop853_c")
            # batched orbit storage is flat: index 2*i+j; same code path as
            # the single-orbit run so the result is bit-for-bit identical.
            assert numpy.array_equal(o_batch.orbit[2 * i + j], o_one.orbit[0]), (
                f"Per-orbit integration with 2D Orbit shape disagrees at ({i},{j})"
            )
    return None


# Test that o.time(), o.x(), o.R(), etc. work after a per-orbit integration
def test_integrate_indiv_t_access():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
            [1.2, -0.05, 0.9, -0.1, 0.1, 0.5],
            [0.8, 0.0, 1.0, 0.2, -0.05, 1.0],
        ]
    )
    nt = 51
    ts = numpy.array(
        [
            numpy.linspace(0.0, 5.0, nt),
            numpy.linspace(0.0, 7.0, nt),
            numpy.linspace(0.0, 9.0, nt),
        ]
    )
    o = Orbit(vxvvs)
    o.integrate(ts, pot, method="dop853_c")

    # time() returns shape (*orbit.shape, nt)
    assert o.time().shape == (3, nt)
    assert numpy.allclose(o.time(), ts)

    # Scalar t: same time for every orbit, output shape == orbit.shape
    assert o.x(0.0).shape == (3,)
    # One time per orbit (shape == orbit.shape)
    assert o.x(numpy.array([0.0, 1.0, 2.0])).shape == (3,)
    # nt_q times per orbit (shape == orbit.shape + (nt_q,))
    assert o.x(ts).shape == (3, nt)
    # R() with on-grid query
    assert o.R(o.time()).shape == (3, nt)

    # Single-orbit slice exposes the orbit's own integration grid: time(),
    # the stored orbit, and quantity-method queries should all be bit-for-bit
    # equal to the corresponding fresh single-orbit integration.
    o_one = Orbit(vxvvs[1])
    o_one.integrate(ts[1], pot, method="dop853_c")
    assert o[1].shape == ()
    assert numpy.array_equal(o[1].time(), ts[1])  # reshaped self.t for scalar slice
    assert numpy.array_equal(o[1].time(), o_one.time())
    assert numpy.array_equal(o[1].orbit, o_one.orbit)
    assert numpy.array_equal(o[1].x(ts[1]), o_one.x(ts[1]))  # fast path
    qq = numpy.linspace(ts[1][0], ts[1][-1], 13)
    assert numpy.allclose(o[1].x(qq), o_one.x(qq), atol=1e-12, rtol=1e-12)

    # Per-orbit batched query equals stacked per-orbit single integrations
    expected = numpy.empty((3, nt))
    for ii in range(3):
        oi = Orbit(vxvvs[ii])
        oi.integrate(ts[ii], pot, method="dop853_c")
        expected[ii] = oi.x(ts[ii])
    assert numpy.array_equal(o.x(ts), expected)

    # Out-of-bounds query raises
    with pytest.raises(ValueError, match="not in the integration time domain"):
        o.x(numpy.array([100.0, 100.0, 100.0]))
    return None


# Test that the pure-Python integrator paths and force_map=True handle per-orbit t
def test_integrate_indiv_t_python_paths():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
            [1.2, -0.05, 0.9, -0.1, 0.1, 0.5],
        ]
    )
    nt = 51
    ts = numpy.array([numpy.linspace(0.0, 5.0, nt), numpy.linspace(0.0, 7.0, nt)])
    # Pure-Python "leapfrog" path
    o_lp = Orbit(vxvvs)
    o_lp.integrate(ts, pot, method="leapfrog")
    assert o_lp.orbit.shape == (2, nt, 6)
    # force_map=True with a C method routes through parallel_map
    o_fm = Orbit(vxvvs)
    o_fm.integrate(ts, pot, method="dop853_c", force_map=True)
    assert o_fm.orbit.shape == (2, nt, 6)

    # Axisymmetric (phasedim=5): pure-Python dop853/odeint branch with len(yo[0])==5
    vxvvs_axi = numpy.array([[1.0, 0.1, 1.1, 0.1, 0.05], [1.2, -0.05, 0.9, -0.1, 0.1]])
    o_axi = Orbit(vxvvs_axi)
    o_axi.integrate(ts, pot, method="dop853")
    assert o_axi.orbit.shape == (2, nt, 5)
    o_axi2 = Orbit(vxvvs_axi)
    o_axi2.integrate(ts, pot, method="odeint")
    assert o_axi2.orbit.shape == (2, nt, 5)

    # 2D (planar) leapfrog Python path
    pot_p = pot[0].toPlanar()
    vxvvs_p = numpy.array([[1.0, 0.1, 1.1, 0.0], [1.2, -0.05, 0.9, 0.5]])
    o_p = Orbit(vxvvs_p)
    o_p.integrate(ts, pot_p, method="leapfrog")
    assert o_p.orbit.shape == (2, nt, 4)

    # 1D leapfrog and odeint Python paths
    pot_l = potential.IsothermalDiskPotential(amp=1.0, sigma=1.0)
    vxvvs_l = numpy.array([[0.5, 0.1], [-0.3, 0.2]])
    o_l = Orbit(vxvvs_l)
    o_l.integrate(ts, pot_l, method="leapfrog")
    assert o_l.orbit.shape == (2, nt, 2)
    o_l2 = Orbit(vxvvs_l)
    o_l2.integrate(ts, pot_l, method="odeint")
    assert o_l2.orbit.shape == (2, nt, 2)

    # 1D forced parallel C path
    o_l3 = Orbit(vxvvs_l)
    o_l3.integrate(ts, pot_l, method="dop853_c", force_map=True)
    assert o_l3.orbit.shape == (2, nt, 2)

    # Single-orbit (size 1) per-orbit-t cases — these exercise the in-process
    # serial branch of parallel_map so the closure bodies get coverage credit.
    one_t = numpy.array([numpy.linspace(0.0, 5.0, nt)])  # shape (1, nt)
    one_full = Orbit(numpy.array([vxvvs[0]]))  # shape (1,) 3D
    one_full.integrate(one_t, pot, method="leapfrog")
    assert one_full.orbit.shape == (1, nt, 6)
    one_axi = Orbit(numpy.array([vxvvs_axi[0]]))  # shape (1,) 3D-axi (phasedim=5)
    one_axi.integrate(one_t, pot, method="dop853")
    assert one_axi.orbit.shape == (1, nt, 5)
    one_axi2 = Orbit(numpy.array([vxvvs_axi[0]]))
    one_axi2.integrate(one_t, pot, method="odeint")
    assert one_axi2.orbit.shape == (1, nt, 5)
    one_axi3 = Orbit(numpy.array([vxvvs_axi[0]]))
    one_axi3.integrate(one_t, pot, method="dop853_c", force_map=True)
    assert one_axi3.orbit.shape == (1, nt, 5)
    one_p = Orbit(numpy.array([vxvvs_p[0]]))  # shape (1,) planar
    one_p.integrate(one_t, pot_p, method="leapfrog")
    assert one_p.orbit.shape == (1, nt, 4)
    one_pa = Orbit(numpy.array([vxvvs_p[0][:3]]))  # shape (1,) axi-planar (phasedim=3)
    one_pa.integrate(one_t, pot_p, method="dop853")
    assert one_pa.orbit.shape == (1, nt, 3)
    one_pa2 = Orbit(numpy.array([vxvvs_p[0][:3]]))
    one_pa2.integrate(one_t, pot_p, method="odeint")
    assert one_pa2.orbit.shape == (1, nt, 3)
    one_l = Orbit(numpy.array([vxvvs_l[0]]))  # shape (1,) linear
    one_l.integrate(one_t, pot_l, method="leapfrog")
    assert one_l.orbit.shape == (1, nt, 2)
    # Also exercise the per-orbit interp for non-3D phasedim by querying off-grid
    qq = numpy.linspace(
        0.5, 4.5, 7
    )  # 1D shared shape (nt_q,) doesn't match self.shape=(1,)
    # Use shape (1, 7) instead (per-orbit-shaped)
    qq_per = numpy.array([qq])
    assert one_l.x(qq_per).shape == (1, 7)
    assert one_axi.R(qq_per).shape == (1, 7)

    # (Per-orbit Quantity-valued t is exercised in tests/test_quantity.py via
    # test_orbits_integrate_perOrbitTimeAsQuantity.)

    # Backward per-orbit integration (decreasing t per orbit) → triggers the
    # per-orbit sort branch in _setupOrbitInterp when an off-grid query
    # forces the interpolators to be built.
    ts_back = numpy.linspace(numpy.zeros(2), -3.0 * numpy.ones(2), nt, axis=-1)
    o_back = Orbit(vxvvs)
    o_back.integrate(ts_back, pot, method="dop853_c")
    qq_back = numpy.array(
        [numpy.linspace(-2.5, -0.5, 7), numpy.linspace(-2.5, -0.5, 7)]
    )
    assert o_back.x(qq_back).shape == (2, 7)

    # NaN-padded per-orbit self.t (produced by bruteSOS) with off-grid query
    # → triggers the NaN-drop branch in _setupOrbitInterp.
    o_nan = Orbit(
        numpy.array(
            [
                [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
                [1.05, 0.0, 1.0, 0.05, 0.05, 0.3],
            ]
        )
    )
    o_nan.bruteSOS(numpy.linspace(0.0, 50.0, 2001), pot, method="dop853_c")
    t_grid = numpy.asarray(o_nan.t)
    # Query at slightly off-grid points within each orbit's window
    qq_nan = numpy.zeros((2, 1))
    for ii in range(2):
        valid = ~numpy.isnan(t_grid[ii])
        valid_t = t_grid[ii, valid]
        # midpoint between first two valid times — guaranteed off-grid
        qq_nan[ii, 0] = 0.5 * (valid_t[0] + valid_t[1])
    _ = o_nan.x(qq_nan)

    # Flat (size,)-shaped one-time-per-orbit query when self.shape != (size,)
    vxvvs_grid = numpy.array(
        [
            [[1.0, 0.1, 1.1, 0.1, 0.05, 0.0], [1.05, 0.0, 1.0, 0.05, 0.05, 0.3]],
            [[1.2, -0.05, 0.9, -0.1, 0.1, 0.5], [0.95, 0.05, 1.05, 0.0, -0.05, 0.7]],
        ]
    )
    nt_g = 21
    ts_g = numpy.empty((2, 2, nt_g))
    for i in range(2):
        for j in range(2):
            ts_g[i, j] = numpy.linspace(0.0, 3.0, nt_g)
    o_g = Orbit(vxvvs_grid)
    o_g.integrate(ts_g, pot, method="dop853_c")
    # Pass flat shape (4,) — self.size=4, self.shape=(2,2)
    flat_t = numpy.array([0.0, 0.5, 1.0, 1.5])
    assert o_g.x(flat_t).shape == (2, 2)
    return None


# Test the access semantics for higher-dim Orbit shape (2,2)
def test_integrate_indiv_t_access_2Dshape():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [[1.0, 0.1, 1.1, 0.1, 0.05, 0.0], [1.05, 0.0, 1.0, 0.05, 0.05, 0.3]],
            [[1.2, -0.05, 0.9, -0.1, 0.1, 0.5], [0.95, 0.05, 1.05, 0.0, -0.05, 0.7]],
        ]
    )
    nt = 41
    ts = numpy.empty((2, 2, nt))
    for i in range(2):
        for j in range(2):
            ts[i, j] = numpy.linspace(0.0, 3.0 + 0.5 * (2 * i + j), nt)
    o = Orbit(vxvvs)
    o.integrate(ts, pot, method="dop853_c")

    assert o.shape == (2, 2)
    assert o.time().shape == (2, 2, nt)
    assert numpy.array_equal(o.time(), ts)

    # Scalar applies to all orbits, output (2,2)
    assert o.x(0.0).shape == (2, 2)
    # One time per orbit (shape (2,2)) → output (2,2)
    assert o.x(numpy.zeros((2, 2))).shape == (2, 2)
    # nt_q times per orbit (shape (2,2,nt)) → output (2,2,nt)
    assert o.x(ts).shape == (2, 2, nt)

    # 1D shape that doesn't match orbit shape rejected
    with pytest.raises(ValueError, match="incompatible"):
        o.x(numpy.linspace(0.0, 3.0, 11))

    # --- Slicing checks for non-trivial shape ---
    # (a) Single-element slice → shape (), time() shape (nt,)
    o_one_10 = Orbit(vxvvs[1, 0])
    o_one_10.integrate(ts[1, 0], pot, method="dop853_c")
    assert o[1, 0].shape == ()
    assert o[1, 0].size == 1
    assert numpy.array_equal(o[1, 0].time(), ts[1, 0])
    assert numpy.array_equal(o[1, 0].time(), o_one_10.time())
    assert numpy.array_equal(o[1, 0].orbit, o_one_10.orbit)
    assert numpy.array_equal(o[1, 0].x(ts[1, 0]), o_one_10.x(ts[1, 0]))

    # (b) Row slice (single index on first axis) → shape (2,), time() shape (2,nt)
    sub = o[1]
    assert sub.shape == (2,)
    assert sub.size == 2
    assert sub.time().shape == (2, nt)
    assert numpy.array_equal(sub.time(), ts[1])
    # Each element of the row slice is a fresh single integration
    for j in range(2):
        ref = Orbit(vxvvs[1, j])
        ref.integrate(ts[1, j], pot, method="dop853_c")
        assert numpy.array_equal(sub[j].time(), ts[1, j])
        assert numpy.array_equal(sub[j].orbit, ref.orbit)
    # Per-orbit query at ts[1] returns the stored grid
    assert numpy.array_equal(sub.x(ts[1]), o.x(ts)[1])

    # (c) Slice on first axis (slice(None)) → shape unchanged
    sub_all = o[:]
    assert sub_all.shape == (2, 2)
    assert numpy.array_equal(sub_all.time(), ts)
    assert numpy.array_equal(sub_all.orbit, o.orbit)

    # (d) Boolean / fancy indexing on first axis → shape (n_selected, 2)
    sub_bool = o[numpy.array([True, False])]
    assert sub_bool.shape == (1, 2)
    assert numpy.array_equal(sub_bool.time(), ts[:1])
    assert numpy.array_equal(
        sub_bool.orbit, o.orbit.reshape(2, 2, nt, 6)[:1].reshape(-1, nt, 6)
    )
    return None


# Validation tests: shape mismatches and non-evenly-spaced time arrays
def test_integrate_indiv_t_input_validation():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
            [1.2, -0.05, 0.9, -0.1, 0.1, 0.5],
        ]
    )
    o = Orbit(vxvvs)
    # Wrong leading shape (3 time arrays for 2 orbits)
    bad_t = numpy.array(
        [
            numpy.linspace(0.0, 5.0, 101),
            numpy.linspace(0.0, 7.0, 101),
            numpy.linspace(0.0, 9.0, 101),
        ]
    )
    with pytest.raises(ValueError, match="does not match Orbit shape"):
        o.integrate(bad_t, pot, method="dop853_c")
    # Non-evenly-spaced time array for a method that requires equispacing
    o2 = Orbit(vxvvs)
    bad_spacing = numpy.array(
        [
            numpy.linspace(0.0, 5.0, 101),
            numpy.concatenate(
                [numpy.linspace(0.0, 3.0, 50), numpy.linspace(3.5, 7.0, 51)]
            ),
        ]
    )
    with pytest.raises(ValueError, match="must be equally spaced"):
        o2.integrate(bad_spacing, pot, method="symplec4_c")
    return None


# Test that integrating an orbit in MWPotential2014 using integrate_SOS conserves energy
def test_integrate_SOS_2D():
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot, axi=True)
    psis = numpy.linspace(0.0, 20.0 * numpy.pi, 1001)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        for surface in ["x", "y"]:
            o.integrate_SOS(psis, pot, method=method)  # default is surface='x'
            Es = o.E(o.t)
            assert (numpy.std(Es) / numpy.mean(Es)) ** 2.0 < 10.0**-10, (
                f"Energy is not conserved by integrate_sos for method={method} and surface={surface}"
            )
    return None


# Test that the 2D SOS function returns points with x=0, vx > 0 when surface='x'
def test_SOS_2Dx():
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
            surface="x",
        )
        xs = o.x(o.t)
        vxs = o.vx(o.t)
        assert (numpy.fabs(xs) < 10.0**-6.0).all(), (
            f"x on SOS is not zero for integrate_sos for method={method}"
        )
        assert (vxs > 0.0).all(), (
            f"vx on SOS is not positive for integrate_sos for method={method}"
        )
    return None


# Test that the 2D SOS function returns points with y=0, vy > 0 when surface='y'
def test_SOS_2Dy():
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
            surface="y",
        )
        ys = o.y(o.t)
        vys = o.vy(o.t)
        assert (numpy.fabs(ys) < 10.0**-7.0).all(), (
            f"y on SOS is not zero for integrate_sos for method={method}"
        )
        assert (vys > 0.0).all(), (
            f"vy on SOS is not positive for integrate_sos for method={method}"
        )
    return None


# Test that the 2D SOS function returns points with x=0, vx > 0 when surface='x'
def test_bruteSOS_2Dx():
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
            surface="x",
        )
        xs = o.x(o.t)
        vxs = o.vx(o.t)
        assert (numpy.fabs(xs) < 10.0**-3.0).all(), (
            f"x on SOS is not zero for bruteSOS for method={method}"
        )
        assert (vxs > 0.0).all(), (
            f"vx on SOS is not zero for bruteSOS for method={method}"
        )
    return None


# Test that the 2D SOS function returns points with y=0, vy > 0 when surface='y'
def test_bruteSOS_2Dy():
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        o.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
            surface="y",
        )
        ys = o.y(o.t)
        vys = o.vy(o.t)
        assert (numpy.fabs(ys) < 10.0**-3.0).all(), (
            f"y on SOS is not zero for bruteSOS for method={method}"
        )
        assert (vys > 0.0).all(), f"vy SOS is not zero for bruteSOS for method={method}"
    return None


# Test that the SOS integration returns an error
# when the orbit does not leave the surface
def test_SOS_onsurfaceerror_3D():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.0, 0.0])
    with pytest.raises(
        RuntimeError,
        match="An orbit appears to be within the SOS surface. Refusing to perform specialized SOS integration, please use normal integration instead",
    ):
        o.SOS(potential.MWPotential2014)
    return None


# Test that the SOS integration returns an error
# when the orbit does not leave the surface
def test_SOS_onsurfaceerror_2D():
    from galpy.orbit import Orbit

    # An orbit considered in the book
    def orbit_xvxE(x, vx, E, pot=None):
        """Returns Orbit at (x,vx,y=0) with given E"""
        return Orbit(
            [
                x,
                vx,
                numpy.sqrt(
                    2.0
                    * (
                        E
                        - potential.evaluatePotentials(pot, x, 0.0, phi=0.0)
                        - vx**2.0 / 2
                    )
                ),
                0.0,
            ]
        )

    # Need the 2d zvc here (maybe should add to galpy?)
    def zvc(x, E, pot=None):
        """Returns the maximum v_x at this x and this
        energy: the zero-velocity curve"""
        return numpy.sqrt(
            2.0 * (E - potential.evaluatePotentials(pot, x, 0.0, phi=0.0))
        )

    lp = potential.LogarithmicHaloPotential(normalize=True, b=0.9, core=0.2)
    E = -0.87
    x = 0.204
    # This orbit remains in the y=0 plane and psi therefore
    # remains zero, thus not increasing
    maxvx = zvc(x, E, pot=lp)
    o = orbit_xvxE(x, maxvx, E, pot=lp)
    with pytest.raises(
        RuntimeError,
        match="An orbit appears to be within the SOS surface. Refusing to perform specialized SOS integration, please use normal integration instead",
    ):
        o.SOS(lp, surface="y")
    return None


# Test that the eccentricity of circular orbits is zero
def test_eccentricity():
    # return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 7.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "ias15_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("mockInterpSphericalPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -16.0
    tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["NFWPotential"] = -12.0  # these are more difficult
    tol["TriaxialNFWPotential"] = -12.0  # these are more difficult
    tol["MultipoleExpansionPotential"] = -15.0  # slightly more difficult
    tol["DiskMultipoleExpansionPotential"] = -6.0  # these are more difficult
    firstTest = True
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "isNonAxi") and tp.isNonAxi:
            continue  # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        else:
            ptp = None
        for integrator in integrators:
            # First do axi
            o = setup_orbit_eccentricity(tp, axi=True)
            if firstTest:
                try:
                    o.e()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.e() before the orbit was integrated did not throw an AttributeError"
                    )
            if isinstance(tp, testplanarMWPotential) or isinstance(tp, testMWPotential):
                o.integrate(times, tp._potlist, method=integrator)
            else:
                o.integrate(times, tp, method=integrator)
            tecc = o.e()
            #            print p, integrator, tecc
            assert tecc**2.0 < 10.0**ttol, (
                "Eccentricity of a circular orbit is not equal to zero by %g for potential %s and integrator %s"
                % (tecc**2.0, p, integrator)
            )
            # add tracking azimuth
            o = setup_orbit_eccentricity(tp, axi=False)
            if firstTest:
                try:
                    o.e()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.e() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, tp, method=integrator)
            tecc = o.e()
            #            print p, integrator, tecc
            assert tecc**2.0 < 10.0**ttol, (
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s"
                % (p, integrator)
            )
            if ptp is None:
                if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                    break
            # Same for a planarPotential
            #            print integrator
            o = setup_orbit_eccentricity(ptp, axi=True)
            if firstTest:
                try:
                    o.e()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.e() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, ptp, method=integrator)
            tecc = o.e()
            #            print p, integrator, tecc
            assert tecc**2.0 < 10.0**ttol, (
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s"
                % (p, integrator)
            )
            # Same for a planarPotential, track azimuth
            o = setup_orbit_eccentricity(ptp, axi=False)
            if firstTest:
                try:
                    o.e()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.e() before the orbit was integrated did not throw an AttributeError"
                    )
                firstTest = True
            o.integrate(times, ptp, method=integrator)
            tecc = o.e()
            #            print p, integrator, tecc
            assert tecc**2.0 < 10.0**ttol, (
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s"
                % (p, integrator)
            )
            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


# Test that the pericenter of orbits launched with vR=0 and vT > vc is the starting radius
def test_pericenter():
    # return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 7.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("mockInterpSphericalPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -16.0
    #    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    #    tol['NFWPotential']= -12. #these are more difficult
    firstTest = True
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "isNonAxi") and tp.isNonAxi:
            continue  # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        else:
            ptp = None
        for integrator in integrators:
            # First do axi
            o = setup_orbit_pericenter(tp, axi=True)
            if firstTest:
                try:
                    o.rperi()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rperi() before the orbit was integrated did not throw an AttributeError"
                    )
            if isinstance(tp, testplanarMWPotential) or isinstance(tp, testMWPotential):
                o.integrate(times, tp._potlist, method=integrator)
            else:
                o.integrate(times, tp, method=integrator)
            tperi = o.rperi()
            #               print p, integrator, tperi
            assert (tperi - o.R()) ** 2.0 < 10.0**ttol, (
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            # add tracking azimuth
            o = setup_orbit_pericenter(tp, axi=False)
            if firstTest:
                try:
                    o.rperi()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rperi() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, tp, method=integrator)
            tperi = o.rperi()
            #            print p, integrator, tperi
            assert (tperi - o.R()) ** 2.0 < 10.0**ttol, (
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            if ptp is None:
                if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                    break
            # Same for a planarPotential
            #            print integrator
            o = setup_orbit_pericenter(ptp, axi=True)
            if firstTest:
                try:
                    o.rperi()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rperi() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, ptp, method=integrator)
            tperi = o.rperi()
            #            print p, integrator, tperi
            assert (tperi - o.R()) ** 2.0 < 10.0**ttol, (
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            # Same for a planarPotential, track azimuth
            o = setup_orbit_pericenter(ptp, axi=False)
            if firstTest:
                try:
                    o.rperi()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rperi() before the orbit was integrated did not throw an AttributeError"
                    )
                firstTest = False
            o.integrate(times, ptp, method=integrator)
            tperi = o.rperi()
            #            print p, integrator, tperi
            assert (tperi - o.R()) ** 2.0 < 10.0**ttol, (
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


# Test that the apocenter of orbits launched with vR=0 and vT < vc is the starting radius
def test_apocenter():
    # return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 7.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("mockInterpSphericalPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -16.0
    tol["FlattenedPowerPotential"] = -14.0  # these are more difficult
    #    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    #    tol['NFWPotential']= -12. #these are more difficult
    firstTest = True
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "isNonAxi") and tp.isNonAxi:
            continue  # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        else:
            ptp = None
        for integrator in integrators:
            # First do axi
            o = setup_orbit_apocenter(tp, axi=True)
            if firstTest:
                try:
                    o.rap()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rap() before the orbit was integrated did not throw an AttributeError"
                    )
            if isinstance(tp, testplanarMWPotential) or isinstance(tp, testMWPotential):
                o.integrate(times, tp._potlist, method=integrator)
            else:
                o.integrate(times, tp, method=integrator)
            tapo = o.rap()
            # print p, integrator, tapo, (tapo-o.R())**2.
            assert (tapo - o.R()) ** 2.0 < 10.0**ttol, (
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            # add tracking azimuth
            o = setup_orbit_apocenter(tp, axi=False)
            if firstTest:
                try:
                    o.rap()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rap() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, tp, method=integrator)
            tapo = o.rap()
            #            print p, integrator, tapo
            assert (tapo - o.R()) ** 2.0 < 10.0**ttol, (
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            if ptp is None:
                if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                    break
            # Same for a planarPotential
            #            print integrator
            o = setup_orbit_apocenter(ptp, axi=True)
            if firstTest:
                try:
                    o.rap()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rap() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, ptp, method=integrator)
            tapo = o.rap()
            #            print p, integrator, tapo
            assert (tapo - o.R()) ** 2.0 < 10.0**ttol, (
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            # Same for a planarPotential, track azimuth
            o = setup_orbit_apocenter(ptp, axi=False)
            if firstTest:
                try:
                    o.rap()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.rap() before the orbit was integrated did not throw an AttributeError"
                    )
                firstTest = False
            o.integrate(times, ptp, method=integrator)
            tapo = o.rap()
            #            print p, integrator, tapo
            assert (tapo - o.R()) ** 2.0 < 10.0**ttol, (
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s"
                % (p, integrator)
            )
            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


# Test that the zmax of orbits launched with vz=0 is the starting height
def test_zmax():
    # return None
    # Basic parameters for the test
    times = numpy.linspace(0.0, 7.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    pots.append("mockInterpSphericalPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    # No C and therefore annoying
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -16.0
    tol["RazorThinExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["KuzminDiskPotential"] = -6.0  # these are more difficult
    #    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    firstTest = True
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "isNonAxi") and tp.isNonAxi:
            continue  # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp, "normalize"):
            continue  # skip these
        tp.normalize(1.0)
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        else:
            ptp = None
        for integrator in integrators:
            # First do axi
            o = setup_orbit_zmax(tp, axi=True)
            if firstTest:
                try:
                    o.zmax()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.zmax() before the orbit was integrated did not throw an AttributeError"
                    )
            if isinstance(tp, testMWPotential):
                o.integrate(times, tp._potlist, method=integrator)
            else:
                o.integrate(times, tp, method=integrator)
            tzmax = o.zmax()
            #            print p, integrator, tzmax
            assert (tzmax - o.z()) ** 2.0 < 10.0**ttol, (
                "Zmax for an orbit launched with vR=0 and vT > Vc is not equal to the initial height for potential %s and integrator %s"
                % (p, integrator)
            )
            # add tracking azimuth
            o = setup_orbit_zmax(tp, axi=False)
            if firstTest:
                try:
                    o.zmax()  # This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.zmax() before the orbit was integrated did not throw an AttributeError"
                    )
            o.integrate(times, tp, method=integrator)
            tzmax = o.zmax()
            #            print p, integrator, tzmax
            assert (tzmax - o.z()) ** 2.0 < 10.0**ttol, (
                "Zmax for an orbit launched with vR=0 and vT > Vc is not equal to the initial height for potential %s and integrator %s"
                % (p, integrator)
            )
            if firstTest:
                ptp = tp.toPlanar()
                o = setup_orbit_energy(ptp, axi=False)
                try:
                    o.zmax()  # This should throw an AttributeError, bc there is no zmax
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.zmax() for a planarOrbit did not throw an AttributeError"
                    )
                o = setup_orbit_energy(ptp, axi=True)
                try:
                    o.zmax()  # This should throw an AttributeError, bc there is no zmax
                except AttributeError:
                    pass
                else:
                    raise AssertionError(
                        "o.zmax() for a planarROrbit did not throw an AttributeError"
                    )
            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


# Test that vR of circular orbits is always zero

# Test the vT of circular orbits is always vc


# Test that the eccentricity, apo-, and pericenters of orbits calculated analytically agrees with the numerical calculation
def test_analytic_ecc_rperi_rap():
    # Basic parameters for the test
    times = numpy.linspace(0.0, 20.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    rmpots.append(
        "HomogeneousSpherePotential"
    )  # fails currently, because delta estimation gives a NaN due to a 0/0; delta should just be zero, but don't want to special-case
    # No C and therefore annoying
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -10.0
    tol["NFWPotential"] = -9.0  # these are more difficult
    tol["PlummerPotential"] = -9.0  # these are more difficult
    tol["EinastoPotential"] = -9.0  # these are more difficult
    tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["RazorThinExponentialDiskPotential"] = -8.0  # these are more difficult
    tol["IsochronePotential"] = -6.0  # these are more difficult
    tol["DehnenSphericalPotential"] = -8.0  # these are more difficult
    tol["DehnenCoreSphericalPotential"] = -8.0  # these are more difficult
    tol["JaffePotential"] = -6.0  # these are more difficult
    tol["TriaxialHernquistPotential"] = -8.0  # these are more difficult
    tol["TriaxialJaffePotential"] = -8.0  # these are more difficult
    tol["TriaxialNFWPotential"] = -8.0  # these are more difficult
    tol["PowerSphericalPotential"] = -8.0  # these are more difficult
    tol["PowerSphericalPotentialwCutoff"] = -8.0  # these are more difficult
    tol["FlattenedPowerPotential"] = -8.0  # these are more difficult
    tol["KeplerPotential"] = -8.0  # these are more difficult
    tol["PseudoIsothermalPotential"] = -7.0  # these are more difficult
    tol["KuzminDiskPotential"] = -8.0  # these are more difficult
    tol["DiskSCFPotential"] = -8.0  # these are more difficult
    tol["DiskMultipoleExpansionPotential"] = -8.0  # these are more difficult
    tol["PowerTriaxialPotential"] = -8.0  # these are more difficult
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        if p == "MWPotential":
            tp = potential.MWPotential
            ptp = [ttp.toPlanar() for ttp in tp]
        else:
            try:
                tclass = getattr(potential, p)
            except AttributeError:
                tclass = getattr(sys.modules[__name__], p)
            tp = tclass()
            if hasattr(tp, "isNonAxi") and tp.isNonAxi:
                continue  # skip, bc eccentricity of circ. =/= 0
            if not hasattr(tp, "normalize"):
                continue  # skip these
            tp.normalize(1.0)
            if hasattr(tp, "toPlanar"):
                ptp = tp.toPlanar()
            else:
                ptp = None
        for integrator in integrators:
            for ii in range(4):
                if ii == 0:  # axi, full
                    # First do axi
                    o = setup_orbit_analytic(tp, axi=True)
                    if isinstance(tp, testplanarMWPotential) or isinstance(
                        tp, testMWPotential
                    ):
                        o.integrate(times, tp._potlist, method=integrator)
                    else:
                        o.integrate(times, tp, method=integrator)
                elif ii == 1:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_analytic(tp, axi=False)
                    if isinstance(tp, testplanarMWPotential) or isinstance(
                        tp, testMWPotential
                    ):
                        o.integrate(times, tp._potlist, method=integrator)
                    else:
                        o.integrate(times, tp, method=integrator)
                elif ii == 2:  # axi, planar
                    if ptp is None:
                        continue
                    # First do axi
                    o = setup_orbit_analytic(ptp, axi=True)
                    if isinstance(ptp, testplanarMWPotential) or isinstance(
                        ptp, testMWPotential
                    ):
                        o.integrate(times, ptp._potlist, method=integrator)
                    else:
                        o.integrate(times, ptp, method=integrator)
                elif ii == 3:  # track azimuth, full
                    if ptp is None:
                        continue
                    # First do axi
                    o = setup_orbit_analytic(ptp, axi=False)
                    if isinstance(ptp, testplanarMWPotential) or isinstance(
                        ptp, testMWPotential
                    ):
                        o.integrate(times, ptp._potlist, method=integrator)
                    else:
                        o.integrate(times, ptp, method=integrator)
                # Eccentricity
                tecc = o.e()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    tecc_analytic = o.e(analytic=True, type="adiabatic")
                else:
                    tecc_analytic = o.e(analytic=True)
                # print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc - tecc_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed eccentricity does not agree with numerical estimate for potential %s and integrator %s, by %g"
                    % (p, integrator, (tecc - tecc_analytic) ** 2.0)
                )
                # Pericenter radius
                trperi = o.rperi()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trperi_analytic = o.rperi(analytic=True, type="adiabatic")
                else:
                    trperi_analytic = o.rperi(analytic=True)
                # print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s"
                    % (p, integrator)
                )
                assert (o.rperi(ro=8.0) / 8.0 - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Apocenter radius
                trap = o.rap()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trap_analytic = o.rap(analytic=True, type="adiabatic")
                else:
                    trap_analytic = o.rap(analytic=True)
                # print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s by %g"
                    % (p, integrator, (trap - trap_analytic) ** 2.0)
                )
                assert (o.rap(ro=8.0) / 8.0 - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Do this also for an orbit starting at pericenter
                if ii == 0:  # axi, full
                    # First do axi
                    o = setup_orbit_pericenter(tp, axi=True)
                    o.integrate(times, tp, method=integrator)
                elif ii == 1:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_pericenter(tp, axi=False)
                    o.integrate(times, tp, method=integrator)
                elif ii == 2:  # axi, planar
                    # First do axi
                    o = setup_orbit_pericenter(ptp, axi=True)
                    o.integrate(times, ptp, method=integrator)
                elif ii == 3:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_pericenter(ptp, axi=False)
                    o.integrate(times, ptp, method=integrator)
                # Eccentricity
                tecc = o.e()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    tecc_analytic = o.e(analytic=True, type="adiabatic")
                else:
                    tecc_analytic = o.e(analytic=True)
                # print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc - tecc_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed eccentricity does not agree with numerical estimate for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Pericenter radius
                trperi = o.rperi()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trperi_analytic = o.rperi(analytic=True, type="adiabatic")
                else:
                    trperi_analytic = o.rperi(analytic=True)
                # print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s"
                    % (p, integrator)
                )
                assert (o.rperi(ro=8.0) / 8.0 - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Apocenter radius
                trap = o.rap()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trap_analytic = o.rap(analytic=True, type="adiabatic")
                else:
                    trap_analytic = o.rap(analytic=True)
                # print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s by %g"
                    % (p, integrator, (trap - trap_analytic))
                )
                assert (o.rap(ro=8.0) / 8.0 - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Do this also for an orbit starting at apocenter
                if ii == 0:  # axi, full
                    # First do axi
                    o = setup_orbit_apocenter(tp, axi=True)
                    o.integrate(times, tp, method=integrator)
                elif ii == 1:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_apocenter(tp, axi=False)
                    o.integrate(times, tp, method=integrator)
                elif ii == 2:  # axi, planar
                    # First do axi
                    o = setup_orbit_apocenter(ptp, axi=True)
                    o.integrate(times, ptp, method=integrator)
                elif ii == 3:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_apocenter(ptp, axi=False)
                    o.integrate(times, ptp, method=integrator)
                # Eccentricity
                tecc = o.e()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    tecc_analytic = o.e(analytic=True, type="adiabatic")
                else:
                    tecc_analytic = o.e(analytic=True)
                # print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc - tecc_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed eccentricity does not agree with numerical estimate by %g for potential %s and integrator %s"
                    % ((tecc - tecc_analytic) ** 2.0, p, integrator)
                )
                # Pericenter radius
                trperi = o.rperi()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trperi_analytic = o.rperi(analytic=True, type="adiabatic")
                else:
                    trperi_analytic = o.rperi(analytic=True)
                # print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s"
                    % (p, integrator)
                )
                assert (o.rperi(ro=8.0) / 8.0 - trperi_analytic) ** 2.0 < 10.0**ttol, (
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
                # Apocenter radius
                trap = o.rap()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    trap_analytic = o.rap(analytic=True, type="adiabatic")
                else:
                    trap_analytic = o.rap(analytic=True)
                # print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s"
                    % (p, integrator)
                )
                assert (o.rap(ro=8.0) / 8.0 - trap_analytic) ** 2.0 < 10.0**ttol, (
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )

            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


def test_orbit_rguiding():
    from galpy.orbit import Orbit
    from galpy.potential import (
        LogarithmicHaloPotential,
        MWPotential2014,
        TriaxialNFWPotential,
        rl,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    assert numpy.fabs(o.rguiding(pot=lp) - rl(lp, Lz)) < 1e-10, (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    assert (
        numpy.fabs(o.rguiding(pot=MWPotential2014) - rl(MWPotential2014, Lz)) < 1e-10
    ), (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl"
    )
    # For an orbit integrated in a non-axisymmetric potential, such that Lz varies
    np = TriaxialNFWPotential(amp=20.0, c=0.8, b=0.7)
    npaxi = TriaxialNFWPotential(amp=20.0, c=0.8)
    R, Lz = 1.2, 2.4
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, np)
    assert (
        numpy.amax(
            numpy.fabs(
                o.rguiding(ts, pot=npaxi)
                - numpy.array([rl(npaxi, o.Lz(t)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl for integrated orbit"
    )
    return None


def test_orbit_rguiding_planar():
    from galpy.orbit import Orbit
    from galpy.potential import (
        LogarithmicHaloPotential,
        MWPotential2014,
        TriaxialNFWPotential,
        rl,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    assert numpy.fabs(o.rguiding(pot=lp) - rl(lp, Lz)) < 1e-10, (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0])
    assert (
        numpy.fabs(o.rguiding(pot=MWPotential2014) - rl(MWPotential2014, Lz)) < 1e-10
    ), (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl"
    )
    # For an orbit integrated in a non-axisymmetric potential, such that Lz varies
    np = TriaxialNFWPotential(amp=20.0, c=0.8, b=0.7)
    npaxi = TriaxialNFWPotential(amp=20.0, c=0.8)
    R, Lz = 1.2, 2.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, np)
    assert (
        numpy.amax(
            numpy.fabs(
                o.rguiding(ts, pot=npaxi)
                - numpy.array([rl(npaxi, o.Lz(t)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl for integrated orbit"
    )
    return None


def test_orbit_rE():
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        LogarithmicHaloPotential,
        MWPotential2014,
        rE,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=lp)
    assert numpy.fabs(o.rE(pot=lp) - rE(lp, E)) < 1e-10, (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=MWPotential2014)
    assert numpy.fabs(o.rE(pot=MWPotential2014) - rE(MWPotential2014, E)) < 1e-10, (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE"
    )
    # For an orbit integrated in a time-dependent potential, such that E varies
    dp = DehnenBarPotential()
    R, Lz = 0.9, 1
    o = Orbit([R, 0.0, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=lp)
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, lp + dp)
    assert (
        numpy.amax(
            numpy.fabs(
                o.rE(ts, pot=lp) - numpy.array([rE(lp, o.E(t, pot=lp)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE for integrated orbit"
    )
    return None


def test_orbit_rE_planar():
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        LogarithmicHaloPotential,
        MWPotential2014,
        rE,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    E = o.E(pot=lp)
    assert numpy.fabs(o.rE(pot=lp) - rE(lp, E)) < 1e-10, (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0])
    E = o.E(pot=MWPotential2014)
    assert numpy.fabs(o.rE(pot=MWPotential2014) - rE(MWPotential2014, E)) < 1e-10, (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE"
    )
    # For an orbit integrated in a time-dependent potential, such that E varies
    dp = DehnenBarPotential()
    R, Lz = 0.9, 1
    o = Orbit(
        [
            R,
            0.0,
            Lz / R,
            0.0,
        ]
    )
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, lp + dp)
    assert (
        numpy.amax(
            numpy.fabs(
                o.rE(ts, pot=lp) - numpy.array([rE(lp, o.E(t, pot=lp)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "rE returned by Orbit interface rE is different from that returned by potential interface rE for integrated orbit"
    )
    return None


def test_orbit_LcE():
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        LcE,
        LogarithmicHaloPotential,
        MWPotential2014,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=lp)
    assert numpy.fabs(o.LcE(pot=lp) - LcE(lp, E)) < 1e-10, (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=MWPotential2014)
    assert numpy.fabs(o.LcE(pot=MWPotential2014) - LcE(MWPotential2014, E)) < 1e-10, (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE"
    )
    # For an orbit integrated in a time-dependent potential, such that E varies
    dp = DehnenBarPotential()
    R, Lz = 0.9, 1
    o = Orbit([R, 0.0, Lz / R, 0.0, 0.1, 0.0])
    E = o.E(pot=lp)
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, lp + dp)
    assert (
        numpy.amax(
            numpy.fabs(
                o.LcE(ts, pot=lp) - numpy.array([LcE(lp, o.E(t, pot=lp)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE for integrated orbit"
    )
    return None


def test_orbit_LcE_planar():
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        LcE,
        LogarithmicHaloPotential,
        MWPotential2014,
    )

    # For a single potential
    lp = LogarithmicHaloPotential(normalize=1.0)
    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    E = o.E(pot=lp)
    assert numpy.fabs(o.LcE(pot=lp) - LcE(lp, E)) < 1e-10, (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE"
    )
    # For a list of potentials
    R, Lz = 1.4, 0.9
    o = Orbit([R, 0.4, Lz / R, 0.0])
    E = o.E(pot=MWPotential2014)
    assert numpy.fabs(o.LcE(pot=MWPotential2014) - LcE(MWPotential2014, E)) < 1e-10, (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE"
    )
    # For an orbit integrated in a time-dependent potential, such that E varies
    dp = DehnenBarPotential()
    R, Lz = 0.9, 1
    o = Orbit(
        [
            R,
            0.0,
            Lz / R,
            0.0,
        ]
    )
    ts = numpy.linspace(0.0, 10.0, 101)
    o.integrate(ts, lp + dp)
    assert (
        numpy.amax(
            numpy.fabs(
                o.LcE(ts, pot=lp) - numpy.array([LcE(lp, o.E(t, pot=lp)) for t in ts])
            )
        )
        < 1e-10
    ), (
        "LcE returned by Orbit interface LcE is different from that returned by potential interface LcE for integrated orbit"
    )
    return None


# Check that zmax calculated analytically agrees with numerical calculation
def test_analytic_zmax():
    # Basic parameters for the test
    times = numpy.linspace(0.0, 20.0, 251)  # ~10 Gyr at the Solar circle
    integrators = [
        "dopr54_c",  # first, because we do it for all potentials
        "odeint",  # direct python solver
        "dop853",
        "dop853_c",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    # Grab all of the potentials
    pots = [
        p
        for p in dir(potential)
        if (
            "Potential" in p
            and not "plot" in p
            and not "RZTo" in p
            and not "FullTo" in p
            and not "toPlanar" in p
            and not "evaluate" in p
            and not "Wrapper" in p
            and not "toVertical" in p
        )
    ]
    pots.append("testMWPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "MovingObjectPotential",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "SnapshotRZPotential",
        "InterpSnapshotRZPotential",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalHarmonicPotentialMixin",
        "SphericalPotential",
        "interpSphericalPotential",
        "CompositePotential",
        "planarCompositePotential",
        "baseCompositePotential",
        "KuijkenDubinskiDiskExpansionPotential",
    ]
    rmpots.append("SphericalShellPotential")
    rmpots.append("RingPotential")
    rmpots.append(
        "HomogeneousSpherePotential"
    )  # fails currently, because delta estimation gives a NaN due to a 0/0; delta should just be zero, but don't want to special-case
    # No C and therefore annoying
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    if False:  # _GHACTIONS:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -9.0
    tol["IsochronePotential"] = -4.0  # these are more difficult
    tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["RazorThinExponentialDiskPotential"] = -4.0  # these are more difficult
    tol["KuzminKutuzovStaeckelPotential"] = -4.0  # these are more difficult
    tol["PlummerPotential"] = -4.0  # these are more difficult
    tol["PseudoIsothermalPotential"] = -4.0  # these are more difficult
    tol["DehnenSphericalPotential"] = -8.0  # these are more difficult
    tol["DehnenCoreSphericalPotential"] = -8.0  # these are more difficult
    tol["HernquistPotential"] = -8.0  # these are more difficult
    tol["TriaxialHernquistPotential"] = -8.0  # these are more difficult
    tol["JaffePotential"] = -8.0  # these are more difficult
    tol["TriaxialJaffePotential"] = -8.0  # these are more difficult
    tol["TriaxialNFWPotential"] = -8.0  # these are more difficult
    tol["MiyamotoNagaiPotential"] = -7.0  # these are more difficult
    tol["MN3ExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["LogarithmicHaloPotential"] = -7.0  # these are more difficult
    tol["KeplerPotential"] = -7.0  # these are more difficult
    tol["PowerSphericalPotentialwCutoff"] = -8.0  # these are more difficult
    tol["FlattenedPowerPotential"] = -8.0  # these are more difficult
    tol["testMWPotential"] = -6.0  # these are more difficult
    tol["KuzminDiskPotential"] = -4  # these are more difficult
    tol["SCFPotential"] = -8.0  # these are more difficult
    tol["DiskSCFPotential"] = -6.0  # these are more difficult
    tol["MultipoleExpansionPotential"] = -8.0
    tol["DiskMultipoleExpansionPotential"] = -6.0  # these are more difficult
    for p in pots:
        # Setup instance of potential
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        if p == "MWPotential":
            tp = potential.MWPotential
        else:
            try:
                tclass = getattr(potential, p)
            except AttributeError:
                tclass = getattr(sys.modules[__name__], p)
            tp = tclass()
            if hasattr(tp, "isNonAxi") and tp.isNonAxi:
                continue  # skip, bc eccentricity of circ. =/= 0
            if not hasattr(tp, "normalize"):
                continue  # skip these
            tp.normalize(1.0)
        for integrator in integrators:
            for ii in range(2):
                if ii == 0:  # axi, full
                    # First do axi
                    o = setup_orbit_analytic_zmax(tp, axi=True)
                elif ii == 1:  # track azimuth, full
                    # First do axi
                    o = setup_orbit_analytic_zmax(tp, axi=False)
                if isinstance(tp, testMWPotential):
                    o.integrate(times, tp._potlist, method=integrator)
                else:
                    o.integrate(times, tp, method=integrator)
                tzmax = o.zmax()
                if ii < 2 and (
                    p == "BurkertPotential"
                    or "SCFPotential" in p
                    or "MultipoleExpansion" in p
                    or "FlattenedPower" in p
                    or "RazorThinExponential" in p
                    or "TwoPowerSpherical" in p
                ):  # no Rzderiv currently
                    tzmax_analytic = o.zmax(analytic=True, type="adiabatic")
                else:
                    tzmax_analytic = o.zmax(analytic=True)
                # print(p, integrator, tzmax, tzmax_analytic, (tzmax-tzmax_analytic)**2.)
                assert (tzmax - tzmax_analytic) ** 2.0 < 10.0**ttol, (
                    "Analytically computed zmax does not agree by %g with numerical estimate for potential %s and integrator %s"
                    % (numpy.fabs(tzmax - tzmax_analytic), p, integrator)
                )
                assert (o.zmax(ro=8.0) / 8.0 - tzmax_analytic) ** 2.0 < 10.0**ttol, (
                    "Zmax in physical coordinates does not agree with physical-scale times zmax in normalized coordinates for potential %s and integrator %s"
                    % (p, integrator)
                )
            if _QUICKTEST and (not "NFW" in p or tp.isNonAxi):
                break
    # raise AssertionError
    return None


# Test the error for when explicit stepsize does not divide the output stepsize
def test_check_integrate_dt():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([1.0, 0.1, 1.2, 0.3, 0.2, 2.0])
    times = numpy.linspace(0.0, 7.0, 251)
    # This shouldn't work
    try:
        o.integrate(times, lp, dt=(times[1] - times[0]) / 4.0 * 1.1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "dt that is not an integer divisor of the output step size does not raise a ValueError"
        )
    # This should
    try:
        o.integrate(times, lp, dt=(times[1] - times[0]) / 4.0)
    except ValueError:
        raise AssertionError(
            "dt that is an integer divisor of the output step size raises a ValueError"
        )
    return None


# Test that fixing the stepsize works, issue #207
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_fixedstepsize():
    if WIN32:
        return None  # skip on windows, because fails for reason that I can't figure out (runtimes[0] == 0.) and not that important
    import time

    from galpy.potential import LogarithmicHaloPotential

    # Integrators for which it should work
    integrators = ["leapfrog_c", "rk4_c", "rk6_c", "symplec4_c", "symplec6_c"]
    # Somewhat long time
    times = numpy.linspace(0.0, 100.0, 30001)
    # Test the following multiples
    mults = [1.0, 10.0]
    # Just do this for LogarithmicHaloPotential
    pot = LogarithmicHaloPotential(normalize=1.0)
    planarpot = pot.toPlanar()
    types = ["full", "rz", "planar", "r"]
    # Loop through integrators and different types of orbits
    for integrator in integrators:
        for type in types:
            if type == "full":
                o = setup_orbit_energy(pot, axi=False)
            elif type == "rz":
                o = setup_orbit_energy(pot, axi=True)
            elif type == "planar":
                o = setup_orbit_energy(planarpot, axi=False)
            elif type == "r":
                o = setup_orbit_energy(planarpot, axi=True)
            runtimes = numpy.empty(len(mults))
            for ii, mult in enumerate(mults):
                start = time.time()
                o.integrate(
                    times, pot, dt=(times[1] - times[0]) / mult, method=integrator
                )
                runtimes[ii] = time.time() - start
            for ii, mult in enumerate(mults):
                if ii == 0:
                    continue
                # Pretty loose test, because hard to get exactly right with overhead
                assert (
                    numpy.fabs(runtimes[ii] / runtimes[0] / mults[ii] * mults[0] - 1.0)
                    < 0.85
                ), (
                    "Runtime of integration with fixed stepsize for integrator %s, type or orbit %s, stepsize reduction %i is not %i times less (residual is %g, times %g and %g)"
                    % (
                        integrator,
                        type,
                        mults[ii],
                        mults[ii],
                        numpy.fabs(
                            runtimes[ii] / runtimes[0] / mults[ii] * mults[0] - 1.0
                        ),
                        mults[ii] / mults[0],
                        runtimes[ii] / runtimes[0],
                    )
                )
    return None


# Test that fixing the stepsize works for integrate_dxdv
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_fixedstepsize_dxdv():
    if WIN32:
        return None  # skip on windows, because test_fixedstepsize fails for reason that I can't figure out (runtimes[0] == 0.) and not that important
    import time

    from galpy.potential import LogarithmicHaloPotential

    # Integrators for which it should work
    integrators = ["rk4_c", "rk6_c"]
    # Somewhat long time
    from astropy import units

    times = numpy.linspace(0.0, 100.0, 90001) / 280.0 * units.Gyr
    # Test the following multiples
    mults = [1.0, 10.0]
    # Just do this for LogarithmicHaloPotential
    pot = LogarithmicHaloPotential(normalize=1.0)
    planarpot = pot.toPlanar()
    # Loop through integrators and different types of orbits
    for integrator in integrators:
        o = setup_orbit_energy(planarpot, axi=False)
        runtimes = numpy.empty(len(mults))
        for ii, mult in enumerate(mults):
            start = time.time()
            o.integrate_dxdv(
                1e-2 * numpy.ones(4),
                times,
                planarpot,
                dt=(times[1] - times[0]) / mult,
                method=integrator,
            )
            runtimes[ii] = time.time() - start
        for ii, mult in enumerate(mults):
            if ii == 0:
                continue
            # Pretty loose test, because hard to get exactly right with overhead
            assert (
                numpy.fabs(runtimes[ii] / runtimes[0] / mults[ii] * mults[0] - 1.0)
                < 0.85
            ), (
                "Runtime of integration with fixed stepsize for integrator %s, type or orbit %s, stepsize reduction %i is not %i times less (residual is %g, times %g and %g)"
                % (
                    integrator,
                    type,
                    mults[ii],
                    mults[ii],
                    numpy.fabs(runtimes[ii] / runtimes[0] / mults[ii] * mults[0] - 1.0),
                    mults[ii] / mults[0],
                    runtimes[ii] / runtimes[0],
                )
            )
    return None


# Check that adding a linear orbit to a planar orbit gives a FullOrbit
@pytest.mark.skip(reason="Not implemented for Orbits currently")
def test_add_linear_planar_orbit():
    from galpy.orbit import FullOrbit, RZOrbit

    kg = potential.KGPotential()
    ol = setup_orbit_energy(kg)
    # w/ azimuth
    plp = potential.NFWPotential().toPlanar()
    op = setup_orbit_energy(plp)
    of = ol + op
    assert isinstance(of._orb, FullOrbit.FullOrbit), (
        "Sum of linearOrbit and planarOrbit does not give a FullOrbit"
    )
    of = op + ol
    assert isinstance(of._orb, FullOrbit.FullOrbit), (
        "Sum of linearOrbit and planarOrbit does not give a FullOrbit"
    )
    # w/o azimuth
    op = setup_orbit_energy(plp, axi=True)
    of = ol + op
    assert isinstance(of._orb, RZOrbit.RZOrbit), (
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    )
    of = op + ol
    assert isinstance(of._orb, RZOrbit.RZOrbit), (
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    )
    # op + op shouldn't work
    try:
        of = op + op
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "Adding a planarOrbit to a planarOrbit did not raise AttributeError"
        )
    # w/ physical scale and coordinate-transformation parameters
    ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "dehnen"
    op = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=True)
    of = op + ol
    assert isinstance(of._orb, RZOrbit.RZOrbit), (
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    )
    assert numpy.fabs(op._orb._ro - of._orb._ro) < 10.0**-15.0, (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    assert numpy.fabs(op._orb._vo - of._orb._vo) < 10.0**-15.0, (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    assert numpy.fabs(op._orb._zo - of._orb._zo) < 10.0**-15.0, (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    assert numpy.all(
        numpy.fabs(op._orb._solarmotion - of._orb._solarmotion) < 10.0**-15.0
    ), (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    assert op._orb._roSet == of._orb._roSet, (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    assert op._orb._voSet == of._orb._voSet, (
        "Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters"
    )
    return None


# Check that pickling orbits works
def test_pickle():
    import pickle

    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0])
    po = pickle.dumps(o)
    upo = pickle.loads(po)
    assert o.R() == upo.R(), (
        "Pickled/unpickled orbit does not agree with original orbut for R"
    )
    assert o.vR() == upo.vR(), (
        "Pickled/unpickled orbit does not agree with original orbut for vR"
    )
    assert o.vT() == upo.vT(), (
        "Pickled/unpickled orbit does not agree with original orbut for vT"
    )
    assert o.z() == upo.z(), (
        "Pickled/unpickled orbit does not agree with original orbut for z"
    )
    assert o.vz() == upo.vz(), (
        "Pickled/unpickled orbit does not agree with original orbut for vz"
    )
    assert o.phi() == upo.phi(), (
        "Pickled/unpickled orbit does not agree with original orbut for phi"
    )
    assert (True ^ o._roSet) * (True ^ upo._roSet), (
        "Pickled/unpickled orbit does not agree with original orbut for roSet"
    )
    assert (True ^ o._voSet) * (True ^ upo._voSet), (
        "Pickled/unpickled orbit does not agree with original orbut for voSet"
    )
    # w/ physical scales etc.
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0], ro=10.0, vo=300.0)
    po = pickle.dumps(o)
    upo = pickle.loads(po)
    assert o.R() == upo.R(), (
        "Pickled/unpickled orbit does not agree with original orbut for R"
    )
    assert o.vR() == upo.vR(), (
        "Pickled/unpickled orbit does not agree with original orbut for vR"
    )
    assert o.vT() == upo.vT(), (
        "Pickled/unpickled orbit does not agree with original orbut for vT"
    )
    assert o.z() == upo.z(), (
        "Pickled/unpickled orbit does not agree with original orbut for z"
    )
    assert o.vz() == upo.vz(), (
        "Pickled/unpickled orbit does not agree with original orbut for vz"
    )
    assert o.phi() == upo.phi(), (
        "Pickled/unpickled orbit does not agree with original orbut for phi"
    )
    assert o._ro == upo._ro, (
        "Pickled/unpickled orbit does not agree with original orbut for ro"
    )
    assert o._vo == upo._vo, (
        "Pickled/unpickled orbit does not agree with original orbut for vo"
    )
    assert o._zo == upo._zo, (
        "Pickled/unpickled orbit does not agree with original orbut for zo"
    )
    assert numpy.all(o._solarmotion == upo._solarmotion), (
        "Pickled/unpickled orbit does not agree with original orbut for solarmotion"
    )
    assert (o._roSet) * (upo._roSet), (
        "Pickled/unpickled orbit does not agree with original orbut for roSet"
    )
    assert (o._voSet) * (upo._voSet), (
        "Pickled/unpickled orbit does not agree with original orbut for voSet"
    )
    return None


# Basic checks of the angular momentum function
def test_angularmomentum():
    from galpy.orbit import Orbit

    # Shouldn't work for a 1D orbit
    o = Orbit([1.0, 0.1])
    try:
        o.L()
    except AttributeError:
        pass
    else:
        raise AssertionError("Orbit.L() for linearOrbit did not raise AttributeError")
    # Also shouldn't work for an RZOrbit
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2])
    try:
        o.L()
    except AttributeError:
        pass
    else:
        raise AssertionError("Orbit.L() for RZOrbit did not raise AttributeError")
    # For a planarROrbit, should return Lz
    o = Orbit([1.0, 0.1, 1.1])
    assert numpy.ndim(o.L()) == 0, "planarOrbit's angular momentum isn't 1D"
    assert o.L() == 1.1, "planarOrbit's angular momentum isn't correct"
    if False:
        # JB 5/23/2019 isn'tn sure why he ever implemented the Omega
        # keyword for L, so decided not to support this in new Orbits
        # If Omega is given, then it should be subtracted
        times = numpy.linspace(0.0, 2.0, 51)
        from galpy.potential import MWPotential

        o.integrate(times, MWPotential)
        assert numpy.fabs(o.L(t=1.0, Omega=1.0) - 0.1) < 10.0**-16.0, (
            "o.L() w/ Omega does not work"
        )
    # For a FullOrbit, angular momentum should be 3D
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.0, numpy.pi / 2.0])
    assert o.L().shape[0] == 3, "FullOrbit's angular momentum is not 3D"
    assert numpy.fabs(o.L()[2] - 1.1) < 10.0**-16.0, "FullOrbit's Lz is not correct"
    assert numpy.fabs(o.L()[0] + 0.01) < 10.0**-16.0, "FullOrbit's Lx is not correct"
    assert numpy.fabs(o.L()[1] + 0.11) < 10.0**-16.0, "FullOrbit's Ly is not correct"
    return None


# Check that ER + Ez = E and that ER and EZ are separately conserved for orbits that stay close to the plane for the MWPotential
def test_ER_EZ():
    from galpy.potential import MWPotential

    ona = setup_orbit_analytic_EREz(MWPotential, axi=False)
    oa = setup_orbit_analytic_EREz(MWPotential, axi=True)
    os = [ona, oa]
    for o in os:
        times = numpy.linspace(0.0, 7.0, 251)  # ~10 Gyr at the Solar circle
        o.integrate(times, MWPotential)
        ERs = o.ER(times)
        Ezs = o.Ez(times)
        ERdiff = numpy.fabs(numpy.std(ERs - numpy.mean(ERs)) / numpy.mean(ERs))
        assert ERdiff < 10.0**-4.0, (
            "ER conservation for orbits close to the plane in MWPotential fails at %g%%"
            % (100.0 * ERdiff)
        )
        Ezdiff = numpy.fabs(numpy.std(Ezs - numpy.mean(Ezs)) / numpy.mean(Ezs))
        assert Ezdiff < 10.0**-1.7, (
            "Ez conservation for orbits close to the plane in MWPotential fails at %g%%"
            % (100.0 * Ezdiff)
        )
        # Some basic checking
        assert numpy.fabs(o.ER() - o.ER(pot=MWPotential)) < 10.0**-16.0, (
            "o.ER() not equal to o.ER(pot=)"
        )
        assert numpy.fabs(o.Ez() - o.Ez(pot=MWPotential)) < 10.0**-16.0, (
            "o.ER() not equal to o.Ez(pot=)"
        )
        assert numpy.fabs(o.ER(pot=None) - o.ER(pot=MWPotential)) < 10.0**-16.0, (
            "o.ER() not equal to o.ER(pot=)"
        )
        assert numpy.fabs(o.Ez(pot=None) - o.Ez(pot=MWPotential)) < 10.0**-16.0, (
            "o.ER() not equal to o.Ez(pot=)"
        )
    o = setup_orbit_analytic_EREz(MWPotential, axi=False)
    try:
        o.Ez()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.Ez() w/o potential before the orbit was integrated did not raise AttributeError"
        )
    try:
        o.ER()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.ER() w/o potential before the orbit was integrated did not raise AttributeError"
        )
    o = setup_orbit_analytic_EREz(MWPotential, axi=True)
    try:
        o.Ez()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.Ez() w/o potential before the orbit was integrated did not raise AttributeError"
        )
    try:
        o.ER()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.ER() w/o potential before the orbit was integrated did not raise AttributeError"
        )
    return None


# Check that the different setups work
def test_orbit_setup_linear():
    from galpy.orbit import Orbit

    # linearOrbit
    o = Orbit([1.0, 0.1])
    assert o.dim() == 1, "linearOrbit does not have dim == 1"
    assert numpy.fabs(o.x() - 1.0) < 10.0**-16.0, (
        "linearOrbit x setup does not agree with o.x()"
    )
    assert numpy.fabs(o.vx() - 0.1) < 10.0**-16.0, (
        "linearOrbit vx setup does not agree with o.vx()"
    )
    assert numpy.fabs(o.vr() - 0.1) < 10.0**-16.0, (
        "linearOrbit vx setup does not agree with o.vr()"
    )
    if False:
        # setphi was deprecated when moving to Orbits
        try:
            o.setphi(3.0)
        except AttributeError:
            pass
        else:
            raise AssertionError(
                "setphi applied to linearOrbit did not raise AttributeError"
            )
    return None


def test_orbit_setup_planar():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1])
    assert o.dim() == 2, "planarROrbit does not have dim == 2"
    assert numpy.fabs(o.R() - 1.0) < 10.0**-16.0, (
        "planarOrbit R setup does not agree with o.R()"
    )
    assert numpy.fabs(o.vR() - 0.1) < 10.0**-16.0, (
        "planarOrbit vR setup does not agree with o.vR()"
    )
    assert numpy.fabs(o.vT() - 1.1) < 10.0**-16.0, (
        "planarOrbit vT setup does not agree with o.vT()"
    )
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.0)
        assert numpy.fabs(o.phi() - 3.0) < 10.0**-16.0, (
            "Orbit setphi does not agree with o.phi()"
        )
        # planarROrbit no longer exists after moving to Orbits
        # assert not isinstance(o._orb,planarROrbit), 'After applying setphi, planarROrbit did not become planarOrbit'
    o = Orbit([1.0, 0.1, 1.1, 2.0])
    assert o.dim() == 2, "planarOrbit does not have dim == 2"
    assert numpy.fabs(o.R() - 1.0) < 10.0**-16.0, (
        "planarOrbit R setup does not agree with o.R()"
    )
    assert numpy.fabs(o.vR() - 0.1) < 10.0**-16.0, (
        "planarOrbit vR setup does not agree with o.vR()"
    )
    assert numpy.fabs(o.vT() - 1.1) < 10.0**-16.0, (
        "planarOrbit vT setup does not agree with o.vT()"
    )
    assert numpy.fabs(o.phi() - 2.0) < 10.0**-16.0, (
        "planarOrbit phi setup does not agree with o.phi()"
    )
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.0)
        assert numpy.fabs(o.phi() - 3.0) < 10.0**-16.0, (
            "Orbit setphi does not agree with o.phi()"
        )
    # lb, plane w/ default
    o = Orbit([120.0, 2.0, 0.5, 30.0], lb=True, zo=0.0, solarmotion=[-10.0, 10.0, 0.0])
    obs = [8.0, 0.0]
    assert numpy.fabs(o.ll(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    obs = [8.0, 0.0, -10.0, 230.0]
    assert numpy.fabs(o.pmll(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.pmbb(obs=obs) - 0.0) < 10.0**-5.5, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # also check that the ro,vo,solarmotion values are stored and used properly (issue #158 solution)
    o = Orbit(
        [120.0, 2.0, 0.5, 30.0], lb=True, zo=0.0, solarmotion=[-10.0, 10.0, 0.0], ro=7.5
    )
    assert numpy.fabs(o.ll() - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb() - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    obs = [8.5, 0.0, -10.0, 245.0]
    assert numpy.fabs(o.pmll() - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.pmbb() - 0.0) < 10.0**-5.5, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb in plane and obs=Orbit
    o = Orbit([120.0, 2.0, 0.5, 30.0], lb=True, zo=0.0, solarmotion=[-10.1, 4.0, 0.0])
    obs = Orbit([1.0, -10.1 / 220.0, 224.0 / 220, 0.0], solarmotion="hogg")
    assert numpy.fabs(o.ll(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmll(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmll()"
    )
    assert numpy.fabs(o.pmbb(obs=obs) - 0.0) < 10.0**-5.5, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb in plane and obs=Orbit in the plane
    o = Orbit([120.0, 2.0, 0.5, 30.0], lb=True, zo=0.0, solarmotion=[-10.1, 4.0, 0.0])
    obs = Orbit([1.0, -10.1 / 220.0, 224.0 / 220, 0.0, 0.0, 0.0], solarmotion="hogg")
    assert numpy.fabs(o.ll(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmll(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmll()"
    )
    assert numpy.fabs(o.pmbb(obs=obs) - 0.0) < 10.0**-5.5, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    return None


def test_orbit_setup():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.3])
    assert o.dim() == 3, "RZOrbitOrbit does not have dim == 3"
    assert numpy.fabs(o.R() - 1.0) < 10.0**-16.0, (
        "Orbit R setup does not agree with o.R()"
    )
    assert numpy.fabs(o.vR() - 0.1) < 10.0**-16.0, (
        "Orbit vR setup does not agree with o.vR()"
    )
    assert numpy.fabs(o.vT() - 1.1) < 10.0**-16.0, (
        "Orbit vT setup does not agree with o.vT()"
    )
    assert numpy.fabs(o.vphi() - 1.1) < 10.0**-16.0, (
        "Orbit vT setup does not agree with o.vphi()"
    )
    assert numpy.fabs(o.z() - 0.2) < 10.0**-16.0, (
        "Orbit z setup does not agree with o.z()"
    )
    assert numpy.fabs(o.vz() - 0.3) < 10.0**-16.0, (
        "Orbit vz setup does not agree with o.vz()"
    )
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.0)
        assert numpy.fabs(o.phi() - 3.0) < 10.0**-16.0, (
            "Orbit setphi does not agree with o.phi()"
        )
        # FullOrbit no longer exists after switch to Orbits
        # assert isinstance(o._orb,FullOrbit), 'After applying setphi, RZOrbit did not become FullOrbit'
    o = Orbit((1.0, 0.1, 1.1, 0.2, 0.3, 2.0))  # also testing tuple input
    assert o.dim() == 3, "FullOrbit does not have dim == 3"
    assert numpy.fabs(o.R() - 1.0) < 10.0**-16.0, (
        "Orbit R setup does not agree with o.R()"
    )
    assert numpy.fabs(o.vR() - 0.1) < 10.0**-16.0, (
        "Orbit vR setup does not agree with o.vR()"
    )
    assert numpy.fabs(o.vT() - 1.1) < 10.0**-16.0, (
        "Orbit vT setup does not agree with o.vT()"
    )
    assert numpy.fabs(o.z() - 0.2) < 10.0**-16.0, (
        "Orbit z setup does not agree with o.z()"
    )
    assert numpy.fabs(o.vz() - 0.3) < 10.0**-16.0, (
        "Orbit vz setup does not agree with o.vz()"
    )
    assert numpy.fabs(o.phi() - 2.0) < 10.0**-16.0, (
        "Orbit phi setup does not agree with o.phi()"
    )
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.0)
        assert numpy.fabs(o.phi() - 3.0) < 10.0**-16.0, (
            "Orbit setphi does not agree with o.phi()"
        )
    # Radec w/ default
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], radec=True)
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-12.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # Radec w/ hogg
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], radec=True, solarmotion="hogg")
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-12.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # Radec w/ dehnen and diff ro,vo
    o = Orbit(
        [120.0, 60.0, 2.0, 0.5, 0.4, 30.0],
        radec=True,
        solarmotion="dehnen",
        vo=240.0,
        ro=7.5,
        zo=0.01,
    )
    obs = [7.5, 0.0, 0.01, -10.0, 245.25, 7.17]
    assert numpy.fabs(o.ra(obs=obs, ro=7.5) - 120.0) < 10.0**-13.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec(obs=obs, ro=7.5) - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist(obs=obs, ro=7.5) - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra(obs=obs, ro=7.5, vo=240.0) - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec(obs=obs, ro=7.5, vo=240.0) - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos(obs=obs, ro=7.5, vo=240.0) - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # also check that the ro,vo,solarmotion values are stored and used properly (issue #158 solution)
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-13.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # Radec w/ schoenrich and diff ro,vo
    o = Orbit(
        [120.0, 60.0, 2.0, 0.5, 0.4, 30.0],
        radec=True,
        solarmotion="schoenrich",
        vo=240.0,
        ro=7.5,
        zo=0.035,
    )
    obs = [7.5, 0.0, 0.035, -11.1, 252.24, 7.25]
    assert numpy.fabs(o.ra(obs=obs, ro=7.5) - 120.0) < 10.0**-13.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec(obs=obs, ro=7.5) - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist(obs=obs, ro=7.5) - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra(obs=obs, ro=7.5, vo=240.0) - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec(obs=obs, ro=7.5, vo=240.0) - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos(obs=obs, ro=7.5, vo=240.0) - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # Radec w/ custom solarmotion and diff ro,vo
    o = Orbit(
        [120.0, 60.0, 2.0, 0.5, 0.4, 30.0],
        radec=True,
        solarmotion=[10.0, 20.0, 15.0],
        vo=240.0,
        ro=7.5,
        zo=0.035,
    )
    obs = [7.5, 0.0, 0.035, 10.0, 260.0, 15.0]
    assert numpy.fabs(o.ra(obs=obs, ro=7.5) - 120.0) < 10.0**-13.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec(obs=obs, ro=7.5) - 60.0) < 10.0**-13.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist(obs=obs, ro=7.5) - 2.0) < 10.0**-13.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra(obs=obs, ro=7.5, vo=240.0) - 0.5) < 10.0**-13.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec(obs=obs, ro=7.5, vo=240.0) - 0.4) < 10.0**-13.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos(obs=obs, ro=7.5, vo=240.0) - 30.0) < 10.0**-13.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb w/ default
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], lb=True)
    assert numpy.fabs(o.ll() - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb() - 60.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmll() - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vll() - _K) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.vll()"
    )
    assert numpy.fabs(o.pmbb() - 0.4) < 10.0**-10.0, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vbb() - 0.8 * _K) < 10.0**-10.0, (
        "Orbit pmbb setup does not agree with o.vbb()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb w/ default at the Sun
    o = Orbit([120.0, 60.0, 0.0, 10.0, 20.0, 30.0], uvw=True, lb=True, zo=0.0)
    assert numpy.fabs(o.dist() - 0.0) < 10.0**-2.0, (
        "Orbit dist setup does not agree with o.dist()"
    )  # because of tweak in the code to deal with at the Sun
    assert (
        o.U() ** 2.0 + o.V() ** 2.0 + o.W() ** 2.0 - 10.0**2.0 - 20.0**2.0 - 30.0**2.0
    ) < 10.0**-10.0, (
        "Velocity wrt the Sun when looking at Orbit at the Sun does not agree"
    )
    assert (o.vlos() ** 2.0 - 10.0**2.0 - 20.0**2.0 - 30.0**2.0) < 10.0**-10.0, (
        "Velocity wrt the Sun when looking at Orbit at the Sun does not agree"
    )
    # lb w/ default and UVW
    o = Orbit([120.0, 60.0, 2.0, -10.0, 20.0, -25.0], lb=True, uvw=True)
    assert numpy.fabs(o.ll() - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb() - 60.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.U() + 10.0) < 10.0**-10.0, (
        "Orbit U setup does not agree with o.U()"
    )
    assert numpy.fabs(o.V() - 20.0) < 10.0**-10.0, (
        "Orbit V setup does not agree with o.V()"
    )
    assert numpy.fabs(o.W() + 25.0) < 10.0**-10.0, (
        "Orbit W setup does not agree with o.W()"
    )
    # lb w/ default and UVW, test wrt helioXYZ
    o = Orbit([180.0, 0.0, 2.0, -10.0, 20.0, -25.0], lb=True, uvw=True)
    assert numpy.fabs(o.helioX() + 2.0) < 10.0**-10.0, (
        "Orbit helioX setup does not agree with o.helioX()"
    )
    assert numpy.fabs(o.helioY() - 0.0) < 10.0**-10.0, (
        "Orbit helioY setup does not agree with o.helioY()"
    )
    assert numpy.fabs(o.helioZ() - 0.0) < 10.0**-10.0, (
        "Orbit helioZ setup does not agree with o.helioZ()"
    )
    assert numpy.fabs(o.U() + 10.0) < 10.0**-10.0, (
        "Orbit U setup does not agree with o.U()"
    )
    assert numpy.fabs(o.V() - 20.0) < 10.0**-10.0, (
        "Orbit V setup does not agree with o.V()"
    )
    assert numpy.fabs(o.W() + 25.0) < 10.0**-10.0, (
        "Orbit W setup does not agree with o.W()"
    )
    # Radec w/ hogg and obs=Orbit
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], radec=True, solarmotion="hogg")
    obs = Orbit(
        [1.0, -10.1 / 220.0, 224.0 / 220, 0.0208 / 8.0, 6.7 / 220.0, 0.0],
        solarmotion="hogg",
    )
    assert numpy.fabs(o.ra(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec(obs=obs) - 60.0) < 10.0**-10.0, (
        "Orbit dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.vra(obs=obs) - _K) < 10.0**-10.0, (
        "Orbit pmra setup does not agree with o.vra()"
    )
    assert numpy.fabs(o.pmdec(obs=obs) - 0.4) < 10.0**-10.0, (
        "Orbit pmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vdec(obs=obs) - 0.8 * _K) < 10.0**-10.0, (
        "Orbit pmdec setup does not agree with o.vdec()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb, plane w/ default
    o = Orbit(
        [120.0, 0.0, 2.0, 0.5, 0.0, 30.0],
        lb=True,
        zo=0.0,
        solarmotion=[-10.0, 10.0, 0.0],
    )
    obs = [8.0, 0.0]
    assert numpy.fabs(o.ll(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    obs = [8.0, 0.0, -10.0, 230.0]
    assert numpy.fabs(o.pmll(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmll()"
    )
    assert numpy.fabs(o.pmbb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # lb in plane and obs=Orbit
    o = Orbit(
        [120.0, 0.0, 2.0, 0.5, 0.0, 30.0],
        lb=True,
        zo=0.0,
        solarmotion=[-10.1, 4.0, 0.0],
    )
    obs = Orbit([1.0, -10.1 / 220.0, 224.0 / 220, 0.0], solarmotion="hogg")
    assert numpy.fabs(o.ll(obs=obs) - 120.0) < 10.0**-10.0, (
        "Orbit ll setup does not agree with o.ll()"
    )
    assert numpy.fabs(o.bb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit bb setup does not agree with o.bb()"
    )
    assert numpy.fabs(o.dist(obs=obs) - 2.0) < 10.0**-10.0, (
        "Orbit dist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmll(obs=obs) - 0.5) < 10.0**-10.0, (
        "Orbit pmll setup does not agree with o.pmll()"
    )
    assert numpy.fabs(o.pmbb(obs=obs) - 0.0) < 10.0**-10.0, (
        "Orbit pmbb setup does not agree with o.pmbb()"
    )
    assert numpy.fabs(o.vlos(obs=obs) - 30.0) < 10.0**-10.0, (
        "Orbit vlos setup does not agree with o.vlos()"
    )
    # test galactocentric spherical coordinates
    o = Orbit([2.0, 2.0**0.5, 1.0, 2.0, 2.0**0.5, 0.5])
    assert numpy.fabs(o.vr() - 2.0) < 10.0**-10.0, (
        "Orbit galactocentric spherical coordinates are not correct"
    )
    assert numpy.fabs(o.vtheta() - 0.0) < 10.0**-10.0, (
        "Orbit galactocentric spherical coordinates are not correct"
    )
    assert numpy.fabs(o.theta() - numpy.pi / 4) < 10.0**-10.0, (
        "Orbit galactocentric spherical coordinates are not correct"
    )
    return None


def test_orbit_setup_SkyCoord():
    # Only run this for astropy>3
    if not _APY3:
        return None
    import astropy.coordinates as apycoords
    import astropy.units as u

    from galpy.orbit import Orbit

    ra = 120.0 * u.deg
    dec = 60.0 * u.deg
    distance = 2.0 * u.kpc
    pmra = 0.5 * u.mas / u.yr
    pmdec = 0.4 * u.mas / u.yr
    vlos = 30.0 * u.km / u.s
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
    )
    # w/ default
    o = Orbit(c)
    # galpy's sky is not exactly aligned with astropy's sky, so celestials are slightly off
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-8.0, (
        "Orbit SkyCoord ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-8.0, (
        "Orbit SkyCoord dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-14.0, (
        "Orbit SkyCoorddist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-8.0, (
        "Orbit SkyCoordpmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-8.0, (
        "Orbit SkyCoordpmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit SkyCoordvlos setup does not agree with o.vlos()"
    )
    # Radec w/ hogg
    o = Orbit(c, solarmotion="hogg")
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-8.0, (
        "Orbit SkyCoordra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-8.0, (
        "Orbit SkyCoorddec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-13.0, (
        "Orbit SkyCoorddist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-8.0, (
        "Orbit SkyCoordpmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-8.0, (
        "Orbit SkyCoordpmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit SkyCoordvlos setup does not agree with o.vlos()"
    )
    # Radec w/ dehnen and diff ro,vo
    o = Orbit(c, solarmotion="dehnen", vo=240.0, ro=7.5, zo=0.01)
    obs = [7.5, 0.0, 0.01, -10.0, 245.25, 7.17]
    assert numpy.fabs(o.ra(obs=obs, ro=7.5) - 120.0) < 10.0**-8.0, (
        "Orbit SkyCoordra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec(obs=obs, ro=7.5) - 60.0) < 10.0**-8.0, (
        "Orbit SkyCoorddec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist(obs=obs, ro=7.5) - 2.0) < 10.0**-13.0, (
        "Orbit SkyCoorddist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra(obs=obs, ro=7.5, vo=240.0) - 0.5) < 10.0**-8.0, (
        "Orbit SkyCoordpmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec(obs=obs, ro=7.5, vo=240.0) - 0.4) < 10.0**-8.0, (
        "Orbit SkyCoordpmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos(obs=obs, ro=7.5, vo=240.0) - 30.0) < 10.0**-13.0, (
        "Orbit SkyCoordvlos setup does not agree with o.vlos()"
    )
    # Now specifying the coordinate conversion parameters in the SkyCoord
    v_sun = apycoords.CartesianDifferential([-11.1, 215.0, 3.25] * u.km / u.s)
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        z_sun=1.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    o = Orbit(c)
    assert numpy.fabs(o.ra() - 120.0) < 10.0**-8.0, (
        "Orbit SkyCoord ra setup does not agree with o.ra()"
    )
    assert numpy.fabs(o.dec() - 60.0) < 10.0**-8.0, (
        "Orbit SkyCoord dec setup does not agree with o.dec()"
    )
    assert numpy.fabs(o.dist() - 2.0) < 10.0**-14.0, (
        "Orbit SkyCoorddist setup does not agree with o.dist()"
    )
    assert numpy.fabs(o.pmra() - 0.5) < 10.0**-8.0, (
        "Orbit SkyCoordpmra setup does not agree with o.pmra()"
    )
    assert numpy.fabs(o.pmdec() - 0.4) < 10.0**-8.0, (
        "Orbit SkyCoordpmdec setup does not agree with o.pmdec()"
    )
    assert numpy.fabs(o.vlos() - 30.0) < 10.0**-13.0, (
        "Orbit SkyCoordvlos setup does not agree with o.vlos()"
    )
    # Also test that the coordinate-transformation parameters are properly read
    assert numpy.fabs(o._ro - numpy.sqrt(10.0**2.0 - 1.0**2.0)) < 1e-12, (
        "Orbit SkyCoord setup does not properly store ro"
    )
    assert numpy.fabs(o._ro - numpy.sqrt(10.0**2.0 - 1.0**2.0)) < 1e-12, (
        "Orbit SkyCoord setup does not properly store ro"
    )
    assert numpy.fabs(o._zo - 1.0) < 1e-12, (
        "Orbit SkyCoord setup does not properly store zo"
    )
    assert numpy.all(
        numpy.fabs(o._solarmotion - numpy.array([[11.1, -5.0, 3.25]])) < 1e-12
    ), "Orbit SkyCoord setup does not properly store solarmotion"
    # If we only specify galcen_distance, but not z_sun, zo --> 0
    # Now specifying the coordinate conversion parameters in the SkyCoord
    v_sun = apycoords.CartesianDifferential([-11.1, 215.0, 3.25] * u.km / u.s)
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    o = Orbit(c)
    assert numpy.fabs(o._zo - 0.0) < 1e-12, (
        "Orbit SkyCoord setup does not properly store zo"
    )
    # If we specify both z_sun and zo, they need to be consistent
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        z_sun=1.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    with pytest.raises(ValueError) as excinfo:
        o = Orbit(c, zo=0.025)
    # If ro and galcen_distance are both specified, warn if they are not consistent
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        z_sun=1.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(c, ro=10.0)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Orbit's initialization normalization ro and zo are incompatible with SkyCoord's galcen_distance (should have galcen_distance^2 = ro^2 + zo^2)"
        )
    assert raisedWarning, (
        "Orbit initialization with SkyCoord with galcen_distance incompatible with ro should have raised a warning, but didn't"
    )
    # If ro and galcen_distance are both specified, don't warn if they *are* consistent (issue #370)
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        z_sun=1.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    with warnings.catch_warnings(record=True) as w:
        o = Orbit(c, ro=numpy.sqrt(10.0**2.0 - 1.0**2.0))
        for wi in w:
            assert not issubclass(wi.category, galpyWarning), (
                "Orbit initialization with SkyCoord with galcen_distance compatible with ro shouldn't have raised a warning, but did"
            )
    # If we specify both v_sun and solarmotion, they need to be consistent
    v_sun = apycoords.CartesianDifferential([-11.1, 215.0, 3.25] * u.km / u.s)
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        pm_ra_cosdec=pmra,
        pm_dec=pmdec,
        radial_velocity=vlos,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    # This should be fine
    o = Orbit(c, solarmotion=[11.1, -5.0, 3.25])
    # This shouldn't be
    with pytest.raises(ValueError) as excinfo:
        o = Orbit(c, solarmotion=[11.0, -4.0, 2.25])
    # Should get error if we give a SkyCoord without velocities
    c = apycoords.SkyCoord(
        ra=ra,
        dec=dec,
        distance=distance,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    with pytest.raises(TypeError) as excinfo:
        o = Orbit(c)
    return None


# Check that toPlanar works
def test_toPlanar():
    from galpy.orbit import Orbit

    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0, 2.0])
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for FullOrbit"
    assert obsp.R() == obs.R(), (
        "Planar orbit generated w/ toPlanar does not have the correct R"
    )
    assert obsp.vR() == obs.vR(), (
        "Planar orbit generated w/ toPlanar does not have the correct vR"
    )
    assert obsp.vT() == obs.vT(), (
        "Planar orbit generated w/ toPlanar does not have the correct vT"
    )
    assert obsp.phi() == obs.phi(), (
        "Planar orbit generated w/ toPlanar does not have the correct phi"
    )
    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0])
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for RZOrbit"
    assert obsp.R() == obs.R(), (
        "Planar orbit generated w/ toPlanar does not have the correct R"
    )
    assert obsp.vR() == obs.vR(), (
        "Planar orbit generated w/ toPlanar does not have the correct vR"
    )
    assert obsp.vT() == obs.vT(), (
        "Planar orbit generated w/ toPlanar does not have the correct vT"
    )
    ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for RZOrbit"
    assert obsp.R() == obs.R(), (
        "Planar orbit generated w/ toPlanar does not have the correct R"
    )
    assert obsp.vR() == obs.vR(), (
        "Planar orbit generated w/ toPlanar does not have the correct vR"
    )
    assert obsp.vT() == obs.vT(), (
        "Planar orbit generated w/ toPlanar does not have the correct vT"
    )
    assert numpy.fabs(obs._ro - obsp._ro) < 10.0**-15.0, (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert numpy.fabs(obs._vo - obsp._vo) < 10.0**-15.0, (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert numpy.fabs(obs._zo - obsp._zo) < 10.0**-15.0, (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert numpy.all(numpy.fabs(obs._solarmotion - obsp._solarmotion) < 10.0**-15.0), (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obs._roSet == obsp._roSet, (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obs._voSet == obsp._voSet, (
        "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    obs = Orbit([1.0, 0.1, 1.1, 2.0])
    try:
        obs.toPlanar()
    except AttributeError:
        pass
    else:
        raise AttributeError(
            "toPlanar() applied to a planar Orbit did not raise an AttributeError"
        )
    return None


# Check that toLinear works
def test_toLinear():
    from galpy.orbit import Orbit

    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0, 2.0])
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinear does not generate an Orbit w/ dim=1 for FullOrbit"
    assert obsl.x() == obs.z(), (
        "Linear orbit generated w/ toLinear does not have the correct z"
    )
    assert obsl.vx() == obs.vz(), (
        "Linear orbit generated w/ toLinear does not have the correct vx"
    )
    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0])
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinear does not generate an Orbit w/ dim=1 for FullOrbit"
    assert obsl.x() == obs.z(), (
        "Linear orbit generated w/ toLinear does not have the correct z"
    )
    assert obsl.vx() == obs.vz(), (
        "Linear orbit generated w/ toLinear does not have the correct vx"
    )
    obs = Orbit([1.0, 0.1, 1.1, 2.0])
    try:
        obs.toLinear()
    except AttributeError:
        pass
    else:
        raise AttributeError(
            "toLinear() applied to a planar Orbit did not raise an AttributeError"
        )
    # w/ scales
    ro, vo = 10.0, 300.0
    obs = Orbit([1.0, 0.1, 1.1, 0.3, 0.0, 2.0], ro=ro, vo=vo)
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinear does not generate an Orbit w/ dim=1 for FullOrbit"
    assert obsl.x() == obs.z(), (
        "Linear orbit generated w/ toLinear does not have the correct z"
    )
    assert obsl.vx() == obs.vz(), (
        "Linear orbit generated w/ toLinear does not have the correct vx"
    )
    assert numpy.fabs(obs._ro - obsl._ro) < 10.0**-15.0, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert numpy.fabs(obs._vo - obsl._vo) < 10.0**-15.0, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obsl._zo is None, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obsl._solarmotion is None, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obs._roSet == obsl._roSet, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    assert obs._voSet == obsl._voSet, (
        "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    )
    return None


# Check that some relevant errors are being raised
def test_attributeerrors():
    from galpy.orbit import Orbit

    # Vertical functions for planarOrbits
    o = Orbit([1.0, 0.1, 1.0, 0.1])
    try:
        o.z()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.z() for planarOrbit should have raised AttributeError, but did not"
        )
    try:
        o.vz()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vz() for planarOrbit should have raised AttributeError, but did not"
        )
    try:
        o.theta()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.theta() for planarROrbit should have raise AttributeError, but did not"
        )
    try:
        o.vtheta()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vtheta() for planarROrbit should have raise AttributeError, but did not"
        )
    # phi, x, y, vx, vy for Orbits that don't track phi
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2])
    try:
        o.phi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.phi() for RZOrbit should have raised AttributeError, but did not"
        )
    try:
        o.x()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.x() for RZOrbit should have raised AttributeError, but did not"
        )
    try:
        o.y()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.y() for RZOrbit should have raised AttributeError, but did not"
        )
    try:
        o.vx()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vx() for RZOrbit should have raised AttributeError, but did not"
        )
    try:
        o.vy()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vy() for RZOrbit should have raised AttributeError, but did not"
        )
    o = Orbit([1.0, 0.1, 1.1])
    try:
        o.phi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.phi() for planarROrbit should have raised AttributeError, but did not"
        )
    try:
        o.x()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.x() for planarROrbit should have raised AttributeError, but did not"
        )
    try:
        o.y()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.y() for planarROrbit should have raised AttributeError, but did not"
        )
    try:
        o.vx()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vx() for planarROrbit should have raised AttributeError, but did not"
        )
    try:
        o.vy()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vy() for planarROrbit should have raised AttributeError, but did not"
        )
    try:
        o.theta()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.theta() for planarROrbit should have raise AttributeError, but did not"
        )
    try:
        o.vtheta()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.vtheta() for planarROrbit should have raise AttributeError, but did not"
        )
    return None


# Test reversing an orbit
def test_flip():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    for ii in range(5):
        # Scales to test that these are properly propagated to the new Orbit
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbit_flip(llp, ro, vo, zo, solarmotion, axi=False)
        of = o.flip()
        # First check that the scales have been propagated properly
        assert numpy.fabs(o._ro - of._ro) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert numpy.fabs(o._vo - of._vo) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert (o._zo is None) * (of._zo is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert (o._solarmotion is None) * (of._solarmotion is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        else:
            assert numpy.fabs(o._zo - of._zo) < 10.0**-15.0, (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        assert o._roSet == of._roSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert o._voSet == of._voSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert numpy.abs(o.x() - of.x()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vx() + of.vx()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        else:
            assert numpy.abs(o.R() - of.R()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vR() + of.vR()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vT() + of.vT()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii % 2 == 1:
            assert numpy.abs(o.phi() - of.phi()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii < 2:
            assert numpy.abs(o.z() - of.z()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vz() + of.vz()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
    return None


# Test reversing an orbit inplace
def test_flip_inplace():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbit_flip(llp, ro, vo, zo, solarmotion, axi=False)
        of = o()
        of.flip(inplace=True)
        # First check that the scales have been propagated properly
        assert numpy.fabs(o._ro - of._ro) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert numpy.fabs(o._vo - of._vo) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert (o._zo is None) * (of._zo is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert (o._solarmotion is None) * (of._solarmotion is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        else:
            assert numpy.fabs(o._zo - of._zo) < 10.0**-15.0, (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        assert o._roSet == of._roSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert o._voSet == of._voSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert numpy.abs(o.x() - of.x()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vx() + of.vx()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        else:
            assert numpy.abs(o.R() - of.R()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vR() + of.vR()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vT() + of.vT()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii % 2 == 1:
            assert numpy.abs(o.phi() - of.phi()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii < 2:
            assert numpy.abs(o.z() - of.z()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vz() + of.vz()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
    return None


# Test reversing an orbit inplace after orbit integration
def test_flip_inplace_integrated():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    ts = numpy.linspace(0.0, 1.0, 11)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbit_flip(llp, ro, vo, zo, solarmotion, axi=False)
        of = o()
        if ii < 2 or ii == 3:
            o.integrate(ts, lp)
            of.integrate(ts, lp)
        elif ii == 2:
            o.integrate(ts, plp)
            of.integrate(ts, plp)
        else:
            o.integrate(ts, llp)
            of.integrate(ts, llp)
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o = o(0.5)
        of = of(0.5)
        # First check that the scales have been propagated properly
        assert numpy.fabs(o._ro - of._ro) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert numpy.fabs(o._vo - of._vo) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert (o._zo is None) * (of._zo is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert (o._solarmotion is None) * (of._solarmotion is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        else:
            assert numpy.fabs(o._zo - of._zo) < 10.0**-15.0, (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        assert o._roSet == of._roSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert o._voSet == of._voSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert numpy.abs(o.x() - of.x()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vx() + of.vx()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        else:
            assert numpy.abs(o.R() - of.R()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vR() + of.vR()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vT() + of.vT()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii % 2 == 1:
            assert numpy.abs(o.phi() - of.phi()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii < 2:
            assert numpy.abs(o.z() - of.z()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vz() + of.vz()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
    return None


# Test reversing an orbit inplace after orbit integration, and after having
# once evaluated the orbit before flipping inplace (#345)
# only difference wrt previous test is a line that evaluates of before
# flipping
def test_flip_inplace_integrated_evaluated():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    ts = numpy.linspace(0.0, 1.0, 11)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbit_flip(llp, ro, vo, zo, solarmotion, axi=False)
        of = o()
        if ii < 2 or ii == 3:
            o.integrate(ts, lp)
            of.integrate(ts, lp)
        elif ii == 2:
            o.integrate(ts, plp)
            of.integrate(ts, plp)
        else:
            o.integrate(ts, llp)
            of.integrate(ts, llp)
        # Evaluate, make sure it is at an interpolated time!
        dumb = of.R(0.52)
        # Now flip
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o = o(0.52)
        of = of(0.52)
        # First check that the scales have been propagated properly
        assert numpy.fabs(o._ro - of._ro) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert numpy.fabs(o._vo - of._vo) < 10.0**-15.0, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert (o._zo is None) * (of._zo is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert (o._solarmotion is None) * (of._solarmotion is None), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        else:
            assert numpy.fabs(o._zo - of._zo) < 10.0**-15.0, (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), (
                "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            )
        assert o._roSet == of._roSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        assert o._voSet == of._voSet, (
            "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        )
        if ii == 4:
            assert numpy.abs(o.x() - of.x()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vx() + of.vx()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        else:
            assert numpy.abs(o.R() - of.R()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vR() + of.vR()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vT() + of.vT()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii % 2 == 1:
            assert numpy.abs(o.phi() - of.phi()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
        if ii < 2:
            assert numpy.abs(o.z() - of.z()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
            assert numpy.abs(o.vz() + of.vz()) < 10.0**-10.0, (
                "o.flip() did not work as expected"
            )
    return None


# test getOrbit
def test_getOrbit():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([1.0, 0.1, 1.2, 0.3, 0.2, 2.0])
    times = numpy.linspace(0.0, 7.0, 251)
    o.integrate(times, lp)
    Rs = o.R(times)
    vRs = o.vR(times)
    vTs = o.vT(times)
    zs = o.z(times)
    vzs = o.vz(times)
    phis = o.phi(times)
    orbarray = o.getOrbit()
    assert numpy.all(numpy.fabs(Rs - orbarray[:, 0])) < 10.0**-16.0, (
        "getOrbit does not work as expected for R"
    )
    assert numpy.all(numpy.fabs(vRs - orbarray[:, 1])) < 10.0**-16.0, (
        "getOrbit does not work as expected for vR"
    )
    assert numpy.all(numpy.fabs(vTs - orbarray[:, 2])) < 10.0**-16.0, (
        "getOrbit does not work as expected for vT"
    )
    assert numpy.all(numpy.fabs(zs - orbarray[:, 3])) < 10.0**-16.0, (
        "getOrbit does not work as expected for z"
    )
    assert numpy.all(numpy.fabs(vzs - orbarray[:, 4])) < 10.0**-16.0, (
        "getOrbit does not work as expected for vz"
    )
    assert numpy.all(numpy.fabs(phis - orbarray[:, 5])) < 10.0**-16.0, (
        "getOrbit does not work as expected for phi"
    )
    return None


# Test new orbits formed from __call__
def test_newOrbit():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 0.0])
    ts = numpy.linspace(0.0, 1.0, 21)  # v. quick orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o.integrate(ts, lp)
    no = o(ts[-1])  # new orbit
    assert no.R() == o.R(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert no.vR() == o.vR(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert no.vT() == o.vT(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert no.z() == o.z(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert no.vz() == o.vz(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert no.phi() == o.phi(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    # Also test this for multiple time outputs
    nos = o(ts[-2:])  # new orbits
    # First t
    assert numpy.fabs(nos[0].R() - o.R(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert numpy.fabs(nos[0].vR() - o.vR(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert numpy.fabs(nos[0].vT() - o.vT(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert numpy.fabs(nos[0].z() - o.z(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert numpy.fabs(nos[0].vz() - o.vz(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert numpy.fabs(nos[0].phi() - o.phi(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not nos[0]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    # Second t
    assert numpy.fabs(nos[1].R() - o.R(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert numpy.fabs(nos[1].vR() - o.vR(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert numpy.fabs(nos[1].vT() - o.vT(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert numpy.fabs(nos[1].z() - o.z(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert numpy.fabs(nos[1].vz() - o.vz(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert numpy.fabs(nos[1].phi() - o.phi(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not nos[1]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    return None


# Test new orbits formed from __call__, before integration
def test_newOrbit_b4integration():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 0.0])
    no = o(0.0)  # New orbit formed before integration
    assert numpy.fabs(no.R() - o.R()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert numpy.fabs(no.vR() - o.vR()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert numpy.fabs(no.vT() - o.vT()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert numpy.fabs(no.z() - o.z()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert numpy.fabs(no.vz() - o.vz()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert numpy.fabs(no.phi() - o.phi()) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    return None


# Test that we can still get outputs when there aren't enough points for an actual interpolation
def test_newOrbit_badinterpolation():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 0.0])
    ts = numpy.linspace(
        0.0, 1.0, 3
    )  # v. quick orbit integration, w/ not enough points for interpolation
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o.integrate(ts, lp)
    no = o(ts[-1])  # new orbit
    print("Done")
    assert no.R() == o.R(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert no.vR() == o.vR(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert no.vT() == o.vT(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert no.z() == o.z(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert no.vz() == o.vz(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert no.phi() == o.phi(ts[-1]), (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not no._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    # Also test this for multiple time outputs
    nos = o(ts[-2:])  # new orbits
    # First t
    assert numpy.fabs(nos[0].R() - o.R(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert numpy.fabs(nos[0].vR() - o.vR(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert numpy.fabs(nos[0].vT() - o.vT(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert numpy.fabs(nos[0].z() - o.z(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert numpy.fabs(nos[0].vz() - o.vz(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert numpy.fabs(nos[0].phi() - o.phi(ts[-2])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not nos[0]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[0]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    # Second t
    assert numpy.fabs(nos[1].R() - o.R(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct R"
    )
    assert numpy.fabs(nos[1].vR() - o.vR(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vR"
    )
    assert numpy.fabs(nos[1].vT() - o.vT(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vT"
    )
    assert numpy.fabs(nos[1].z() - o.z(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct z"
    )
    assert numpy.fabs(nos[1].vz() - o.vz(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct vz"
    )
    assert numpy.fabs(nos[1].phi() - o.phi(ts[-1])) < 10.0**-10.0, (
        "New orbit formed from calling an old orbit does not have the correct phi"
    )
    assert not nos[1]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._roSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    assert not nos[1]._voSet, (
        "New orbit formed from calling an old orbit does not have the correct roSet"
    )
    # Try point in between, shouldn't work
    try:
        no = o(0.6)
    except LookupError:
        pass
    else:
        raise AssertionError(
            "Orbit interpolation with not enough points to interpolate should raise LookUpError, but did not"
        )
    return None


# Check the routines that should return physical coordinates
def test_physical_output():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0)
    plp = lp.toPlanar()
    for ii in range(4):
        ro, vo = 7.0, 200.0
        if ii == 0:  # axi, full
            o = setup_orbit_physical(lp, axi=True, ro=ro, vo=vo)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_physical(lp, axi=False, ro=ro, vo=vo)
        elif ii == 2:  # axi, planar
            o = setup_orbit_physical(plp, axi=True, ro=ro, vo=vo)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_physical(plp, axi=False, ro=ro, vo=vo)
        # Test positions
        assert numpy.fabs(o.R() / ro - o.R(use_physical=False)) < 10.0**-10.0, (
            "o.R() output for Orbit setup with ro= does not work as expected"
        )
        if ii % 2 == 1:
            assert numpy.fabs(o.x() / ro - o.x(use_physical=False)) < 10.0**-10.0, (
                "o.x() output for Orbit setup with ro= does not work as expected"
            )
            assert numpy.fabs(o.y() / ro - o.y(use_physical=False)) < 10.0**-10.0, (
                "o.y() output for Orbit setup with ro= does not work as expected"
            )
        if ii < 2:
            assert numpy.fabs(o.r() / ro - o.r(use_physical=False)) < 10.0**-10.0, (
                "o.r() output for Orbit setup with ro= does not work as expected"
            )
            assert numpy.fabs(o.z() / ro - o.z(use_physical=False)) < 10.0**-10.0, (
                "o.z() output for Orbit setup with ro= does not work as expected"
            )
        # Test velocities
        assert numpy.fabs(o.vR() / vo - o.vR(use_physical=False)) < 10.0**-10.0, (
            "o.vR() output for Orbit setup with vo= does not work as expected"
        )
        assert numpy.fabs(o.vT() / vo - o.vT(use_physical=False)) < 10.0**-10.0, (
            "o.vT() output for Orbit setup with vo= does not work as expected"
        )
        assert (
            numpy.fabs(o.vphi() * ro / vo - o.vphi(use_physical=False)) < 10.0**-10.0
        ), "o.vphi() output for Orbit setup with vo= does not work as expected"
        if ii % 2 == 1:
            assert numpy.fabs(o.vx() / vo - o.vx(use_physical=False)) < 10.0**-10.0, (
                "o.vx() output for Orbit setup with vo= does not work as expected"
            )
            assert numpy.fabs(o.vy() / vo - o.vy(use_physical=False)) < 10.0**-10.0, (
                "o.vy() output for Orbit setup with vo= does not work as expected"
            )
        if ii < 2:
            assert numpy.fabs(o.vz() / vo - o.vz(use_physical=False)) < 10.0**-10.0, (
                "o.vz() output for Orbit setup with vo= does not work as expected"
            )
        # Test energies
        assert (
            numpy.fabs(o.E(pot=lp) / vo**2.0 - o.E(pot=lp, use_physical=False))
            < 10.0**-10.0
        ), "o.E() output for Orbit setup with vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Jacobi(pot=lp) / vo**2.0 - o.Jacobi(pot=lp, use_physical=False)
            )
            < 10.0**-10.0
        ), "o.E() output for Orbit setup with vo= does not work as expected"
        if ii < 2:
            assert (
                numpy.fabs(o.ER(pot=lp) / vo**2.0 - o.ER(pot=lp, use_physical=False))
                < 10.0**-10.0
            ), "o.ER() output for Orbit setup with vo= does not work as expected"
            assert (
                numpy.fabs(o.Ez(pot=lp) / vo**2.0 - o.Ez(pot=lp, use_physical=False))
                < 10.0**-10.0
            ), "o.Ez() output for Orbit setup with vo= does not work as expected"
        # Test angular momentun
        if ii > 0:
            assert numpy.all(
                numpy.fabs(o.L() / vo / ro - o.L(use_physical=False)) < 10.0**-10.0
            ), "o.L() output for Orbit setup with ro=,vo= does not work as expected"
        # Test action-angle functions
        if ii == 1:
            assert (
                numpy.fabs(
                    o.jr(pot=lp, type="staeckel", delta=0.5) / vo / ro
                    - o.jr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jp(pot=lp, type="staeckel", delta=0.5) / vo / ro
                    - o.jp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jz(pot=lp, type="staeckel", delta=0.5) / vo / ro
                    - o.jz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tr(pot=lp, type="staeckel", delta=0.5)
                    / conversion.time_in_Gyr(vo, ro)
                    - o.Tr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tp(pot=lp, type="staeckel", delta=0.5)
                    / conversion.time_in_Gyr(vo, ro)
                    - o.Tp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tz(pot=lp, type="staeckel", delta=0.5)
                    / conversion.time_in_Gyr(vo, ro)
                    - o.Tz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Or(pot=lp, type="staeckel", delta=0.5)
                    / conversion.freq_in_Gyr(vo, ro)
                    - o.Or(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Or() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Op(pot=lp, type="staeckel", delta=0.5)
                    / conversion.freq_in_Gyr(vo, ro)
                    - o.Op(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Op() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Oz(pot=lp, type="staeckel", delta=0.5)
                    / conversion.freq_in_Gyr(vo, ro)
                    - o.Oz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Oz() output for Orbit setup with ro=,vo= does not work as expected"
    # Also test the times
    assert numpy.fabs(o.time(1.0) - ro / vo / 1.0227121655399913) < 10.0**-10.0, (
        "o.time() in physical coordinates does not work as expected"
    )
    assert (
        numpy.fabs(o.time(1.0, ro=ro, vo=vo) - ro / vo / 1.0227121655399913)
        < 10.0**-10.0
    ), "o.time() in physical coordinates does not work as expected"
    assert numpy.fabs(o.time(1.0, use_physical=False) - 1.0) < 10.0**-10.0, (
        "o.time() in physical coordinates does not work as expected"
    )
    return None


# Check that the routines that should return physical coordinates are turned off by turn_physical_off
def test_physical_output_off():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    plp = lp.toPlanar()
    for ii in range(4):
        ro, vo = 7.0, 200.0
        if ii == 0:  # axi, full
            o = setup_orbit_physical(lp, axi=True, ro=ro, vo=vo)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_physical(lp, axi=False, ro=ro, vo=vo)
        elif ii == 2:  # axi, planar
            o = setup_orbit_physical(plp, axi=True, ro=ro, vo=vo)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_physical(plp, axi=False, ro=ro, vo=vo)
        # turn off
        o.turn_physical_off()
        # Test positions
        assert numpy.fabs(o.R() - o.R(use_physical=False)) < 10.0**-10.0, (
            "o.R() output for Orbit setup with ro= does not work as expected when turned off"
        )
        if ii % 2 == 1:
            assert numpy.fabs(o.x() - o.x(use_physical=False)) < 10.0**-10.0, (
                "o.x() output for Orbit setup with ro= does not work as expected when turned off"
            )
            assert numpy.fabs(o.y() - o.y(use_physical=False)) < 10.0**-10.0, (
                "o.y() output for Orbit setup with ro= does not work as expected when turned off"
            )
        if ii < 2:
            assert numpy.fabs(o.z() - o.z(use_physical=False)) < 10.0**-10.0, (
                "o.z() output for Orbit setup with ro= does not work as expected when turned off"
            )
            assert numpy.fabs(o.r() - o.r(use_physical=False)) < 10.0**-10.0, (
                "o.r() output for Orbit setup with ro= does not work as expected when turned off"
            )
        # Test velocities
        assert numpy.fabs(o.vR() - o.vR(use_physical=False)) < 10.0**-10.0, (
            "o.vR() output for Orbit setup with vo= does not work as expected when turned off"
        )
        assert numpy.fabs(o.vT() - o.vT(use_physical=False)) < 10.0**-10.0, (
            "o.vT() output for Orbit setup with vo= does not work as expected"
        )
        assert numpy.fabs(o.vphi() - o.vphi(use_physical=False)) < 10.0**-10.0, (
            "o.vphi() output for Orbit setup with vo= does not work as expected when turned off"
        )
        if ii % 2 == 1:
            assert numpy.fabs(o.vx() - o.vx(use_physical=False)) < 10.0**-10.0, (
                "o.vx() output for Orbit setup with vo= does not work as expected when turned off"
            )
            assert numpy.fabs(o.vy() - o.vy(use_physical=False)) < 10.0**-10.0, (
                "o.vy() output for Orbit setup with vo= does not work as expected when turned off"
            )
        if ii < 2:
            assert numpy.fabs(o.vz() - o.vz(use_physical=False)) < 10.0**-10.0, (
                "o.vz() output for Orbit setup with vo= does not work as expected when turned off"
            )
        # Test energies
        assert (
            numpy.fabs(o.E(pot=lp) - o.E(pot=lp, use_physical=False)) < 10.0**-10.0
        ), (
            "o.E() output for Orbit setup with vo= does not work as expected when turned off"
        )
        assert (
            numpy.fabs(o.Jacobi(pot=lp) - o.Jacobi(pot=lp, use_physical=False))
            < 10.0**-10.0
        ), (
            "o.E() output for Orbit setup with vo= does not work as expected when turned off"
        )
        if ii < 2:
            assert (
                numpy.fabs(o.ER(pot=lp) - o.ER(pot=lp, use_physical=False))
                < 10.0**-10.0
            ), (
                "o.ER() output for Orbit setup with vo= does not work as expected when turned off"
            )
            assert (
                numpy.fabs(o.Ez(pot=lp) - o.Ez(pot=lp, use_physical=False))
                < 10.0**-10.0
            ), (
                "o.Ez() output for Orbit setup with vo= does not work as expected when turned off"
            )
        # Test angular momentun
        if ii > 0:
            assert numpy.all(
                numpy.fabs(o.L() - o.L(use_physical=False)) < 10.0**-10.0
            ), (
                "o.L() output for Orbit setup with ro=,vo= does not work as expected when turned off"
            )
        # Test action-angle functions
        if ii == 1:
            assert (
                numpy.fabs(
                    o.jr(pot=lp, type="staeckel", delta=0.5)
                    - o.jr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jp(pot=lp, type="staeckel", delta=0.5)
                    - o.jp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jz(pot=lp, type="staeckel", delta=0.5)
                    - o.jz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.jz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tr(pot=lp, type="staeckel", delta=0.5)
                    - o.Tr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tp(pot=lp, type="staeckel", delta=0.5)
                    - o.Tp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tz(pot=lp, type="staeckel", delta=0.5)
                    - o.Tz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Tz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Or(pot=lp, type="staeckel", delta=0.5)
                    - o.Or(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Or() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Op(pot=lp, type="staeckel", delta=0.5)
                    - o.Op(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Op() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Oz(pot=lp, type="staeckel", delta=0.5)
                    - o.Oz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
                )
                < 10.0**-10.0
            ), "o.Oz() output for Orbit setup with ro=,vo= does not work as expected"
    # Also test the times
    assert numpy.fabs(o.time(1.0) - 1.0) < 10.0**-10.0, (
        "o.time() in physical coordinates does not work as expected when turned off"
    )
    assert (
        numpy.fabs(o.time(1.0, ro=ro, vo=vo) - ro / vo / 1.0227121655399913)
        < 10.0**-10.0
    ), "o.time() in physical coordinates does not work as expected when turned off"
    return None


# Check that the routines that should return physical coordinates are turned
# back on by turn_physical_on
def test_physical_output_on():
    from astropy import units

    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    plp = lp.toPlanar()
    for ii in range(4):
        ro, vo = 7.0, 200.0
        if ii == 0:  # axi, full
            o = setup_orbit_physical(lp, axi=True, ro=ro, vo=vo)
        elif ii == 1:  # track azimuth, full
            o = setup_orbit_physical(lp, axi=False, ro=ro, vo=vo)
        elif ii == 2:  # axi, planar
            o = setup_orbit_physical(plp, axi=True, ro=ro, vo=vo)
        elif ii == 3:  # track azimuth, full
            o = setup_orbit_physical(plp, axi=False, ro=ro, vo=vo)
        o_orig = o()
        # turn off and on
        o.turn_physical_off()
        if ii == 0:
            o.turn_physical_on(ro=ro, vo=vo)
        elif ii == 1:
            o.turn_physical_on(ro=ro * units.kpc, vo=vo * units.km / units.s)
        else:
            o.turn_physical_on()
        # Test positions
        assert numpy.fabs(o.R() - o_orig.R(use_physical=True)) < 10.0**-10.0, (
            "o.R() output for Orbit setup with ro= does not work as expected when turned back on"
        )
        if ii % 2 == 1:
            assert numpy.fabs(o.x() - o_orig.x(use_physical=True)) < 10.0**-10.0, (
                "o.x() output for Orbit setup with ro= does not work as expected when turned back on"
            )
            assert numpy.fabs(o.y() - o_orig.y(use_physical=True)) < 10.0**-10.0, (
                "o.y() output for Orbit setup with ro= does not work as expected when turned back on"
            )
        if ii < 2:
            assert numpy.fabs(o.z() - o_orig.z(use_physical=True)) < 10.0**-10.0, (
                "o.z() output for Orbit setup with ro= does not work as expected when turned back on"
            )
        # Test velocities
        assert numpy.fabs(o.vR() - o_orig.vR(use_physical=True)) < 10.0**-10.0, (
            "o.vR() output for Orbit setup with vo= does not work as expected when turned back on"
        )
        assert numpy.fabs(o.vT() - o_orig.vT(use_physical=True)) < 10.0**-10.0, (
            "o.vT() output for Orbit setup with vo= does not work as expected"
        )
        assert numpy.fabs(o.vphi() - o_orig.vphi(use_physical=True)) < 10.0**-10.0, (
            "o.vphi() output for Orbit setup with vo= does not work as expected when turned back on"
        )
        if ii % 2 == 1:
            assert numpy.fabs(o.vx() - o_orig.vx(use_physical=True)) < 10.0**-10.0, (
                "o.vx() output for Orbit setup with vo= does not work as expected when turned back on"
            )
            assert numpy.fabs(o.vy() - o_orig.vy(use_physical=True)) < 10.0**-10.0, (
                "o.vy() output for Orbit setup with vo= does not work as expected when turned back on"
            )
        if ii < 2:
            assert numpy.fabs(o.vz() - o_orig.vz(use_physical=True)) < 10.0**-10.0, (
                "o.vz() output for Orbit setup with vo= does not work as expected when turned back on"
            )
        # Test energies
        assert (
            numpy.fabs(o.E(pot=lp) - o_orig.E(pot=lp, use_physical=True)) < 10.0**-10.0
        ), (
            "o.E() output for Orbit setup with vo= does not work as expected when turned back on"
        )
        assert (
            numpy.fabs(o.Jacobi(pot=lp) - o_orig.Jacobi(pot=lp, use_physical=True))
            < 10.0**-10.0
        ), (
            "o.E() output for Orbit setup with vo= does not work as expected when turned back on"
        )
        if ii < 2:
            assert (
                numpy.fabs(o.ER(pot=lp) - o_orig.ER(pot=lp, use_physical=True))
                < 10.0**-10.0
            ), (
                "o.ER() output for Orbit setup with vo= does not work as expected when turned back on"
            )
            assert (
                numpy.fabs(o.Ez(pot=lp) - o_orig.Ez(pot=lp, use_physical=True))
                < 10.0**-10.0
            ), (
                "o.Ez() output for Orbit setup with vo= does not work as expected when turned back on"
            )
        # Test angular momentun
        if ii > 0:
            assert numpy.all(
                numpy.fabs(o.L() - o_orig.L(use_physical=True)) < 10.0**-10.0
            ), (
                "o.L() output for Orbit setup with ro=,vo= does not work as expected when turned back on"
            )
        # Test action-angle functions
        if ii == 1:
            assert (
                numpy.fabs(
                    o.jr(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.jr(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.jr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jp(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.jp(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.jp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.jz(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.jz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.jz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tr(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Tr(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Tr() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tp(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Tp(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Tp() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Tz(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Tz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Tz() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Or(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Or(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Or() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Op(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Op(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Op() output for Orbit setup with ro=,vo= does not work as expected"
            assert (
                numpy.fabs(
                    o.Oz(pot=lp, type="staeckel", delta=0.5)
                    - o_orig.Oz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
                )
                < 10.0**-10.0
            ), "o.Oz() output for Orbit setup with ro=,vo= does not work as expected"
    # Also test the times
    assert (
        numpy.fabs(o.time(1.0) - o_orig.time(1.0, use_physical=True)) < 10.0**-10.0
    ), (
        "o_orig.time() in physical coordinates does not work as expected when turned back on"
    )
    return None


# Test that physical scales are propagated correctly when a new orbit is formed by calling an old orbit
def test_physical_newOrbit():
    from galpy.orbit import Orbit

    o = Orbit(
        [1.0, 0.1, 1.1, 0.1, 0.0, 0.0],
        ro=9.0,
        vo=230.0,
        zo=0.02,
        solarmotion=[-5.0, 15.0, 25.0],
    )
    ts = numpy.linspace(0.0, 1.0, 21)  # v. quick orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o.integrate(ts, lp)
    no = o(ts[-1])  # new orbit
    assert no._ro == 9.0, (
        "New orbit formed from calling old orbit's ro is not that of the old orbit"
    )
    assert no._vo == 230.0, (
        "New orbit formed from calling old orbit's vo is not that of the old orbit"
    )
    assert no._ro == 9.0, (
        "New orbit formed from calling old orbit's ro is not that of the old orbit"
    )
    assert no._vo == 230.0, (
        "New orbit formed from calling old orbit's vo is not that of the old orbit"
    )
    assert no._roSet, (
        "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    )
    assert no._voSet, (
        "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    )
    assert no._roSet, (
        "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    )
    assert no._voSet, (
        "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    )
    assert no._zo == 0.02, (
        "New orbit formed from calling old orbit's zo is not that of the old orbit"
    )
    assert no._solarmotion[0] == -5.0, (
        "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    )
    assert no._solarmotion[1] == 15.0, (
        "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    )
    assert no._solarmotion[2] == 25.0, (
        "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    )
    return None


# Test the orbit interpolation
def test_interpolation_issue187():
    # Test that fails because of issue 187 reported by Mark Fardal
    from scipy import interpolate

    from galpy.orbit import Orbit

    pot = potential.IsochronePotential(b=1.0 / 7.0, normalize=True)
    R, vR, vT, z, vz, phi = 1.0, 0.0, 0.8, 0.0, 0.0, 0.0
    orb = Orbit(vxvv=[R, vR, vT, z, vz, phi])
    ts = numpy.linspace(0.0, 10.0, 1000)
    orb.integrate(ts, pot)
    orbpts = orb.getOrbit()
    # detect phase wrap
    pdiff = orbpts[:, 5] - numpy.roll(orbpts[:, 5], 1)
    phaseWrapIndx = numpy.where(pdiff < -6.0)[0][0]
    tsPreWrap = numpy.linspace(
        ts[phaseWrapIndx] - 5.0e-2, ts[phaseWrapIndx] - 0.002, 100
    )
    tsPostWrap = numpy.linspace(
        ts[phaseWrapIndx] + 0.002, ts[phaseWrapIndx] + 5.0e-2, 100
    )
    # Interpolate just before and after the phase-wrap
    preWrapInterpolate = interpolate.InterpolatedUnivariateSpline(
        ts[phaseWrapIndx - 11 : phaseWrapIndx - 1],
        orbpts[phaseWrapIndx - 11 : phaseWrapIndx - 1, 5],
    )
    postWrapInterpolate = interpolate.InterpolatedUnivariateSpline(
        ts[phaseWrapIndx : phaseWrapIndx + 10],
        orbpts[phaseWrapIndx : phaseWrapIndx + 10, 5],
    )
    assert numpy.all(
        numpy.fabs(
            ((preWrapInterpolate(tsPreWrap) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
            - orb.phi(tsPreWrap)
        )
        < 10.0**-5.0
    ), "phase interpolation near a phase-wrap does not work"
    assert numpy.all(
        numpy.fabs(
            ((postWrapInterpolate(tsPostWrap) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
            - orb.phi(tsPostWrap)
        )
        < 10.0**-5.0
    ), "phase interpolation near a phase-wrap does not work"
    return None


# Test that fitting an orbit works
def test_orbitfit():
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0])
    ts = numpy.linspace(0.0, 1.0, 1001)
    o.integrate(ts, lp)
    # Create orbit points from this integrated orbit, each 100th point
    vxvv = o.getOrbit()[::100, :]
    # now fit
    of = Orbit.from_fit(o.vxvv[0], vxvv, pot=lp, tintJ=1.5)
    assert numpy.all(
        comp_orbfit(of, vxvv, numpy.linspace(0.0, 2.0, 1001), lp) < 10.0**-7.0
    ), "Orbit fit in configuration space does not work"
    return None


def test_orbitfit_potinput():
    from galpy.orbit import Orbit
    from galpy.potential import PotentialError

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0])
    ts = numpy.linspace(0.0, 1.0, 1001)
    o.integrate(ts, lp)
    # Create orbit points from this integrated orbit, each 100th point
    vxvv = o.getOrbit()[::100, :]
    # now fit, using another orbit instance, without potential, should error
    of = o()
    try:
        Orbit.from_fit(o.vxvv[0], vxvv, pot=None, tintJ=1.5)
    except PotentialError:
        pass
    else:
        raise AssertionError("Orbit fit w/o potential does not raise PotentialError")
    # Now give a potential to of
    of = Orbit.from_fit(o.vxvv[0], vxvv, pot=lp, tintJ=1.5)
    assert numpy.all(
        comp_orbfit(of, vxvv, numpy.linspace(0.0, 2.0, 1001), lp) < 10.0**-7.0
    ), "Orbit fit in configuration space does not work"
    return None


# Test orbit fit in observed Galactic coordinates
def test_orbitfit_lb():
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0])
    ts = numpy.linspace(0.0, 1.0, 1001)
    o.integrate(ts, lp)
    # Create orbit points from this integrated orbit, each 100th point
    vxvv = []
    for ii in range(10):
        vxvv.append(
            [
                o.ll(ii / 10.0),
                o.bb(ii / 10.0),
                o.dist(ii / 10.0),
                o.pmll(ii / 10.0),
                o.pmbb(ii / 10.0),
                o.vlos(ii / 10.0),
            ]
        )
    vxvv = numpy.array(vxvv)
    # now fit
    of = Orbit.from_fit(
        [o.ll(), o.bb(), o.dist(), o.pmll(), o.pmbb(), o.vlos()],
        vxvv,
        pot=lp,
        tintJ=1.5,
        lb=True,
        vxvv_err=0.01 * numpy.ones_like(vxvv),
    )
    compf = comp_orbfit(of, vxvv, numpy.linspace(0.0, 2.0, 1001), lp, lb=True)
    assert numpy.all(compf < 10.0**-4.0), "Orbit fit in lb space does not work"
    return None


# Test orbit fit in observed equatorial coordinates
def test_orbitfit_radec():
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ro, vo = 9.0, 230.0
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0], ro=ro, vo=vo)
    ts = numpy.linspace(0.0, 1.0, 1001)
    o.integrate(ts, lp)
    # Create orbit points from this integrated orbit, each 100th point
    vxvv = []
    for ii in range(10):
        vxvv.append(
            [
                o.ra(ii / 10.0),
                o.dec(ii / 10.0),
                o.dist(ii / 10.0),
                o.pmra(ii / 10.0),
                o.pmdec(ii / 10.0),
                o.vlos(ii / 10.0),
            ]
        )
    vxvv = numpy.array(vxvv)
    # now fit
    of = Orbit.from_fit(
        [o.ra(), o.dec(), o.dist(), o.pmra(), o.pmdec(), o.vlos()],
        vxvv,
        pot=lp,
        tintJ=1.5,
        radec=True,
        ro=ro,
        vo=vo,
    )
    compf = comp_orbfit(
        of, vxvv, numpy.linspace(0.0, 2.0, 1001), lp, lb=False, radec=True, ro=ro, vo=vo
    )
    assert numpy.all(compf < 10.0**-4.0), "Orbit fit in radec space does not work"
    return None


# Test orbit fit in custom coordinates (using Equatorial for testing)
def test_orbitfit_custom():
    from galpy.orbit import Orbit
    from galpy.util import coords

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ro, vo = 9.0, 230.0
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0], ro=ro, vo=vo)
    ts = numpy.linspace(0.0, 1.0, 1001)
    o.integrate(ts, lp)
    # Create orbit points from this integrated orbit, each 100th point
    vxvv = []
    for ii in range(10):
        vxvv.append(
            [
                o.ra(ii / 10.0),
                o.dec(ii / 10.0),
                o.dist(ii / 10.0),
                o.pmra(ii / 10.0),
                o.pmdec(ii / 10.0),
                o.vlos(ii / 10.0),
            ]
        )
    vxvv = numpy.array(vxvv)
    # now fit
    # First test the exception
    try:
        Orbit.from_fit(
            [o.ra(), o.dec(), o.dist(), o.pmra(), o.pmdec(), o.vlos()],
            vxvv,
            pot=lp,
            tintJ=1.5,
            customsky=True,
            ro=ro,
            vo=vo,
        )
    except OSError:
        pass
    else:
        raise AssertionError(
            "Orbit fit with custom sky coordinates but without the necessary coordinate-transformation functions did not raise an exception"
        )
    of = Orbit.from_fit(
        [o.ra(), o.dec(), o.dist(), o.pmra(), o.pmdec(), o.vlos()],
        vxvv,
        pot=lp,
        tintJ=1.5,
        customsky=True,
        lb_to_customsky=coords.lb_to_radec,
        pmllpmbb_to_customsky=coords.pmllpmbb_to_pmrapmdec,
        ro=ro,
        vo=vo,
    )
    compf = comp_orbfit(
        of, vxvv, numpy.linspace(0.0, 2.0, 1001), lp, lb=False, radec=True, ro=ro, vo=vo
    )
    assert numpy.all(compf < 10.0**-4.0), "Orbit fit in radec space does not work"
    return None


def comp_orbfit(of, vxvv, ts, pot, lb=False, radec=False, ro=None, vo=None):
    """Compare the output of the orbit fit properly, ro and vo only implemented for radec"""
    from galpy.util import coords

    coords._APY_COORDS_ORIG = coords._APY_COORDS
    coords._APY_COORDS = False  # too slow otherwise
    of.integrate(ts, pot)
    off = of.flip()
    off.integrate(ts, pot)
    # Flip velocities again
    off.vxvv[..., 1] *= -1.0
    off.vxvv[..., 2] *= -1.0
    off.vxvv[..., 4] *= -1.0
    if lb:
        allvxvv = []
        for ii in range(len(ts)):
            allvxvv.append(
                [
                    of.ll(ts[ii]),
                    of.bb(ts[ii]),
                    of.dist(ts[ii]),
                    of.pmll(ts[ii]),
                    of.pmbb(ts[ii]),
                    of.vlos(ts[ii]),
                ]
            )
            allvxvv.append(
                [
                    off.ll(ts[ii]),
                    off.bb(ts[ii]),
                    off.dist(ts[ii]),
                    off.pmll(ts[ii]),
                    off.pmbb(ts[ii]),
                    off.vlos(ts[ii]),
                ]
            )
        allvxvv = numpy.array(allvxvv)
    elif radec:
        allvxvv = []
        for ii in range(len(ts)):
            allvxvv.append(
                [
                    of.ra(ts[ii], ro=ro, vo=vo),
                    of.dec(ts[ii], ro=ro, vo=vo),
                    of.dist(ts[ii], ro=ro, vo=vo),
                    of.pmra(ts[ii], ro=ro, vo=vo),
                    of.pmdec(ts[ii], ro=ro, vo=vo),
                    of.vlos(ts[ii], ro=ro, vo=vo),
                ]
            )
            allvxvv.append(
                [
                    off.ra(ts[ii]),
                    off.dec(ts[ii], ro=ro, vo=vo),
                    off.dist(ts[ii], ro=ro, vo=vo),
                    off.pmra(ts[ii], ro=ro, vo=vo),
                    off.pmdec(ts[ii], ro=ro, vo=vo),
                    off.vlos(ts[ii], ro=ro, vo=vo),
                ]
            )
        allvxvv = numpy.array(allvxvv)
    else:
        allvxvv = numpy.concatenate((of.getOrbit(), off.getOrbit()), axis=0)
    out = []
    for ii in range(vxvv.shape[0]):
        out.append(numpy.amin(numpy.sum((allvxvv - vxvv[ii]) ** 2.0, axis=1)))
    coords._APY_COORDS = coords._APY_COORDS_ORIG
    return numpy.array(out)


def test_MWPotential_warning():
    # Test that using MWPotential throws a warning, see #229
    ts = numpy.linspace(0.0, 100.0, 1001)
    psis = numpy.linspace(0.0, 20.0 * numpy.pi, 1001)
    o = setup_orbit_energy(potential.MWPotential, axi=False)
    with pytest.warns(galpyWarning) as record:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        o.integrate(ts, potential.MWPotential)
        # Should raise warning bc of MWPotential, might raise others
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
        )
    assert raisedWarning, (
        "Orbit integration with MWPotential should have thrown a warning, but didn't"
    )
    # Also test for SOS integration
    with pytest.warns(galpyWarning) as record:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        o.integrate_SOS(psis, potential.MWPotential)
        # Should raise warning bc of MWPotential, might raise others
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
        )
    assert raisedWarning, (
        "Orbit integration with MWPotential should have thrown a warning, but didn't"
    )
    return None


# Test the new Orbit.time function
def test_time():
    # Setup orbit
    o = setup_orbit_energy(potential.MWPotential, axi=False)
    # Prior to integration, should return zero
    assert numpy.fabs(o.time() - 0.0) < 10.0**-10.0, (
        "Orbit.time before integration does not return zero"
    )
    # Then integrate
    times = numpy.linspace(0.0, 10.0, 1001)
    o.integrate(times, potential.MWPotential)
    assert numpy.all((o.time() - times) < 10.0**-8.0), (
        "Orbit.time after integration does not return the integration times"
    )
    return None


# Test interpolation with backwards orbit integration
def test_backinterpolation_issue204():
    # Setup orbit and its flipped version
    o = setup_orbit_energy(potential.MWPotential, axi=False)
    of = o.flip()
    # Times to integrate backward and forward of flipped (should agree)
    ntimes = numpy.linspace(0.0, -10.0, 1001)
    ptimes = -ntimes
    # Integrate the orbits
    o.integrate(ntimes, potential.MWPotential)
    of.integrate(ptimes, potential.MWPotential)
    # Test that interpolation works and gives the same result
    nitimes = numpy.linspace(0.0, -10.0, 2501)
    pitimes = -nitimes
    assert numpy.all((o.R(nitimes) - of.R(pitimes)) < 10.0**-8.0), (
        "Forward and backward integration with interpolation do not agree"
    )
    assert numpy.all((o.z(nitimes) - of.z(pitimes)) < 10.0**-8.0), (
        "Forward and backward integration with interpolation do not agree"
    )
    # Velocities should be flipped
    assert numpy.all((o.vR(nitimes) + of.vR(pitimes)) < 10.0**-8.0), (
        "Forward and backward integration with interpolation do not agree"
    )
    assert numpy.all((o.vT(nitimes) + of.vT(pitimes)) < 10.0**-8.0), (
        "Forward and backward integration with interpolation do not agree"
    )
    assert numpy.all((o.vT(nitimes) + of.vT(pitimes)) < 10.0**-8.0), (
        "Forward and backward integration with interpolation do not agree"
    )
    return None


# Test that Orbit.x .y .vx and .vy return a scalar for scalar time input
def test_scalarxyvzvz_issue247():
    # Setup an orbit
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o = setup_orbit_energy(lp, axi=False)
    assert isinstance(o.x(), float), "Orbit.x() does not return a scalar"
    assert isinstance(o.y(), float), "Orbit.y() does not return a scalar"
    assert isinstance(o.vx(), float), "Orbit.vx() does not return a scalar"
    assert isinstance(o.vy(), float), "Orbit.vy() does not return a scalar"
    # Also integrate and then test
    times = numpy.linspace(0.0, 10.0, 1001)
    o.integrate(times, lp)
    assert isinstance(o.x(5.0), float), "Orbit.x() does not return a scalar"
    assert isinstance(o.y(5.0), float), "Orbit.y() does not return a scalar"
    assert isinstance(o.vx(5.0), float), "Orbit.vx() does not return a scalar"
    assert isinstance(o.vy(5.0), float), "Orbit.vy() does not return a scalar"
    return None


# Test that all Orbit methods return a scalar for scalar time input (mentioned
# in #294)
def test_scalar_all():
    # Setup an orbit
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o = setup_orbit_energy(lp, axi=False)
    assert isinstance(o.R(), float), "Orbit.R() does not return a scalar"
    assert isinstance(o.vR(), float), "Orbit.vR() does not return a scalar"
    assert isinstance(o.vT(), float), "Orbit.vT() does not return a scalar"
    assert isinstance(o.z(), float), "Orbit.z() does not return a scalar"
    assert isinstance(o.vz(), float), "Orbit.vz() does not return a scalar"
    assert isinstance(o.phi(), float), "Orbit.phi() does not return a scalar"
    assert isinstance(o.r(), float), "Orbit.r() does not return a scalar"
    assert isinstance(o.x(), float), "Orbit.x() does not return a scalar"
    assert isinstance(o.y(), float), "Orbit.y() does not return a scalar"
    assert isinstance(o.vx(), float), "Orbit.vx() does not return a scalar"
    assert isinstance(o.vy(), float), "Orbit.vy() does not return a scalar"
    assert isinstance(o.theta(), float), "Orbit.theta() does not return a scalar"
    assert isinstance(o.vtheta(), float), "Orbit.vtheta() does not return a scalar"
    assert isinstance(o.vr(), float), "Orbit.vr() does not return a scalar"
    assert isinstance(o.ra(), float), "Orbit.ra() does not return a scalar"
    assert isinstance(o.dec(), float), "Orbit.dec() does not return a scalar"
    assert isinstance(o.ll(), float), "Orbit.ll() does not return a scalar"
    assert isinstance(o.bb(), float), "Orbit.bb() does not return a scalar"
    assert isinstance(o.dist(), float), "Orbit.dist() does not return a scalar"
    assert isinstance(o.pmra(), float), "Orbit.pmra() does not return a scalar"
    assert isinstance(o.pmdec(), float), "Orbit.pmdec() does not return a scalar"
    assert isinstance(o.pmll(), float), "Orbit.pmll() does not return a scalar"
    assert isinstance(o.pmbb(), float), "Orbit.pmbb() does not return a scalar"
    assert isinstance(o.vra(), float), "Orbit.vra() does not return a scalar"
    assert isinstance(o.vdec(), float), "Orbit.vdec() does not return a scalar"
    assert isinstance(o.vll(), float), "Orbit.vll() does not return a scalar"
    assert isinstance(o.vbb(), float), "Orbit.vbb() does not return a scalar"
    assert isinstance(o.vlos(), float), "Orbit.vlos() does not return a scalar"
    assert isinstance(o.helioX(), float), "Orbit.helioX() does not return a scalar"
    assert isinstance(o.helioY(), float), "Orbit.helioY() does not return a scalar"
    assert isinstance(o.helioZ(), float), "Orbit.helioZ() does not return a scalar"
    assert isinstance(o.U(), float), "Orbit.U() does not return a scalar"
    assert isinstance(o.V(), float), "Orbit.V() does not return a scalar"
    assert isinstance(o.W(), float), "Orbit.W() does not return a scalar"
    assert isinstance(o.E(pot=lp), float), "Orbit.E() does not return a scalar"
    assert isinstance(o.Jacobi(pot=lp), float), (
        "Orbit.Jacobi() does not return a scalar"
    )
    assert isinstance(o.ER(pot=lp), float), "Orbit.ER() does not return a scalar"
    assert isinstance(o.Ez(pot=lp), float), "Orbit.Ez() does not return a scalar"
    # Also integrate and then test
    times = numpy.linspace(0.0, 10.0, 1001)
    o.integrate(times, lp)
    assert isinstance(o.R(5.0), float), "Orbit.R() does not return a scalar"
    assert isinstance(o.vR(5.0), float), "Orbit.vR() does not return a scalar"
    assert isinstance(o.vT(5.0), float), "Orbit.vT() does not return a scalar"
    assert isinstance(o.z(5.0), float), "Orbit.z() does not return a scalar"
    assert isinstance(o.vz(5.0), float), "Orbit.vz() does not return a scalar"
    assert isinstance(o.phi(5.0), float), "Orbit.phi() does not return a scalar"
    assert isinstance(o.r(5.0), float), "Orbit.r() does not return a scalar"
    assert isinstance(o.x(5.0), float), "Orbit.x() does not return a scalar"
    assert isinstance(o.y(5.0), float), "Orbit.y() does not return a scalar"
    assert isinstance(o.vx(5.0), float), "Orbit.vx() does not return a scalar"
    assert isinstance(o.vy(5.0), float), "Orbit.vy() does not return a scalar"
    assert isinstance(o.theta(5.0), float), "Orbit.theta() does not return a scalar"
    assert isinstance(o.vtheta(5.0), float), "Orbit.vtheta() does not return a scalar"
    assert isinstance(o.vr(5.0), float), "Orbit.vr() does not return a scalar"
    assert isinstance(o.ra(5.0), float), "Orbit.ra() does not return a scalar"
    assert isinstance(o.dec(5.0), float), "Orbit.dec() does not return a scalar"
    assert isinstance(o.ll(5.0), float), "Orbit.ll() does not return a scalar"
    assert isinstance(o.bb(5.0), float), "Orbit.bb() does not return a scalar"
    assert isinstance(o.dist(5.0), float), "Orbit.dist() does not return a scalar"
    assert isinstance(o.pmra(5.0), float), "Orbit.pmra() does not return a scalar"
    assert isinstance(o.pmdec(5.0), float), "Orbit.pmdec() does not return a scalar"
    assert isinstance(o.pmll(5.0), float), "Orbit.pmll() does not return a scalar"
    assert isinstance(o.pmbb(5.0), float), "Orbit.pmbb() does not return a scalar"
    assert isinstance(o.vra(5.0), float), "Orbit.vra() does not return a scalar"
    assert isinstance(o.vdec(5.0), float), "Orbit.vdec() does not return a scalar"
    assert isinstance(o.vll(5.0), float), "Orbit.vll() does not return a scalar"
    assert isinstance(o.vbb(5.0), float), "Orbit.vbb() does not return a scalar"
    assert isinstance(o.vlos(5.0), float), "Orbit.vlos() does not return a scalar"
    assert isinstance(o.helioX(5.0), float), "Orbit.helioX() does not return a scalar"
    assert isinstance(o.helioY(5.0), float), "Orbit.helioY() does not return a scalar"
    assert isinstance(o.helioZ(5.0), float), "Orbit.helioZ() does not return a scalar"
    assert isinstance(o.U(5.0), float), "Orbit.U() does not return a scalar"
    assert isinstance(o.V(5.0), float), "Orbit.V() does not return a scalar"
    assert isinstance(o.W(5.0), float), "Orbit.W() does not return a scalar"
    assert isinstance(o.E(5.0), float), "Orbit.E() does not return a scalar"
    assert isinstance(o.Jacobi(5.0), float), "Orbit.Jacobi() does not return a scalar"
    assert isinstance(o.ER(5.0), float), "Orbit.ER() does not return a scalar"
    assert isinstance(o.Ez(5.0), float), "Orbit.Ez() does not return a scalar"
    return None


def test_call_issue256():
    # Reported by Semyeong Oh: non-integrated orbit with t=/=0 should return error
    from galpy.orbit import Orbit

    o = Orbit(vxvv=[5.0, -1.0, 0.8, 3, -0.1, 0])
    # no integration of the orbit
    with pytest.raises(ValueError) as excinfo:
        o.R(30)
    return None


# Test whether the output from the SkyCoord function is correct
def test_SkyCoord():
    from astropy import units

    from galpy.orbit import Orbit

    # In ra, dec
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], radec=True)
    assert numpy.fabs(o.SkyCoord().ra.degree - o.ra()) < 10.0**-13.0, (
        "Orbit SkyCoord ra and direct ra do not agree"
    )
    assert numpy.fabs(o.SkyCoord().dec.degree - o.dec()) < 10.0**-13.0, (
        "Orbit SkyCoord dec and direct dec do not agree"
    )
    assert numpy.fabs(o.SkyCoord().distance.kpc - o.dist()) < 10.0**-13.0, (
        "Orbit SkyCoord distance and direct distance do not agree"
    )
    # For a list
    o = Orbit([120.0, 60.0, 2.0, 0.5, 0.4, 30.0], radec=True)
    times = numpy.linspace(0.0, 2.0, 51)
    from galpy.potential import MWPotential

    o.integrate(times, MWPotential)
    ras = numpy.array([s.ra.degree for s in o.SkyCoord(times)])
    decs = numpy.array([s.dec.degree for s in o.SkyCoord(times)])
    dists = numpy.array([s.distance.kpc for s in o.SkyCoord(times)])
    assert numpy.all(numpy.fabs(ras - o.ra(times)) < 10.0**-13.0), (
        "Orbit SkyCoord ra and direct ra do not agree"
    )
    assert numpy.all(numpy.fabs(decs - o.dec(times)) < 10.0**-13.0), (
        "Orbit SkyCoord dec and direct dec do not agree"
    )
    assert numpy.all(numpy.fabs(dists - o.dist(times)) < 10.0**-13.0), (
        "Orbit SkyCoord distance and direct distance do not agree"
    )
    # Check that the GC frame parameters are correctly propagated
    if not _APY3:
        return None  # not done in python 2
    o = Orbit(
        [120.0, 60.0, 2.0, 0.5, 0.4, 30.0],
        radec=True,
        ro=10.0,
        zo=1.0,
        solarmotion=[-10.0, 34.0, 12.0],
    )
    assert (
        numpy.fabs(
            o.SkyCoord().galcen_distance.to(units.kpc).value
            - numpy.sqrt(10.0**2.0 + 1.0**2.0)
        )
        < 10.0**-13.0
    ), "Orbit SkyCoord GC frame attributes are incorrect"
    assert numpy.fabs(o.SkyCoord().z_sun.to(units.kpc).value - 1.0) < 10.0**-13.0, (
        "Orbit SkyCoord GC frame attributes are incorrect"
    )
    assert numpy.all(
        numpy.fabs(
            o.SkyCoord().galcen_v_sun.d_xyz.to(units.km / units.s).value
            - numpy.array([10.0, 220.0 + 34.0, 12.0])
        )
        < 10.0**-13.0
    ), "Orbit SkyCoord GC frame attributes are incorrect"
    return None


def test_orbit_obs_list_issue322():
    # Further tests of obs= list parameter for orbit output, mainly in relation
    # to issue #322
    from galpy.orbit import Orbit

    # The basic case, for a planar orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0])
    assert numpy.fabs(o.helioX(obs=[1.0, 0.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[1.0, 0.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0])
    assert numpy.fabs(o.helioX(obs=[0.0, 1.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[0.0, 1.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 3.0 * numpy.pi / 2.0])
    assert numpy.fabs(o.helioX(obs=[0.0, -1.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[0.0, -1.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Full orbit
    # The basic case, for a full orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 0.0])
    assert numpy.fabs(o.helioX(obs=[1.0, 0.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[1.0, 0.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0])
    assert numpy.fabs(o.helioX(obs=[0.0, 1.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[0.0, 1.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 3.0 * numpy.pi / 2.0])
    assert numpy.fabs(o.helioX(obs=[0.0, -1.0, 0.0], ro=1.0) - 0.1) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=[0.0, -1.0, 0.0], ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    return None


def test_orbit_obs_Orbit_issue322():
    # Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322
    from galpy.orbit import Orbit

    # The basic case, for a planar orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0])
    assert (
        numpy.fabs(o.helioX(obs=Orbit([1.0, 0.0, 0.0, 0.0]), ro=1.0) - 0.1) < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert numpy.fabs(o.helioY(obs=Orbit([1.0, 0.0, 0.0, 0.0]), ro=1.0)) < 10.0**-7.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0])
    assert (
        numpy.fabs(o.helioX(obs=Orbit([1.0, 0.0, 0.0, numpy.pi / 2.0]), ro=1.0) - 0.1)
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.helioY(obs=Orbit([1.0, 0.0, 0.0, numpy.pi / 2.0]), ro=1.0))
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 3.0 * numpy.pi / 2.0])
    assert (
        numpy.fabs(
            o.helioX(obs=Orbit([1.0, 0.0, 0.0, 3.0 * numpy.pi / 2.0]), ro=1.0) - 0.1
        )
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.helioY(obs=Orbit([1.0, 0.0, 0.0, 3.0 * numpy.pi / 2.0]), ro=1.0))
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Full orbit
    # The basic case, for a full orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 0.0])
    assert (
        numpy.fabs(o.helioX(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ro=1.0) - 0.1)
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.helioY(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), ro=1.0))
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0])
    assert (
        numpy.fabs(
            o.helioX(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, numpy.pi / 2.0]), ro=1.0) - 0.1
        )
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(
            o.helioY(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, numpy.pi / 2.0]), ro=1.0)
        )
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 3.0 * numpy.pi / 2.0])
    assert (
        numpy.fabs(
            o.helioX(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 3.0 * numpy.pi / 2.0]), ro=1.0)
            - 0.1
        )
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(
            o.helioY(obs=Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 3.0 * numpy.pi / 2.0]), ro=1.0)
        )
        < 10.0**-7.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    return None


def test_orbit_obs_Orbits_issue322():
    # Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322; specific case where the orbit gets evaluated at multiple
    # times
    from galpy.orbit import Orbit

    # Do non-zero Ysun case for planarOrbit
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.helioX(times, obs=obs, ro=1.0)[ii]
                - o.helioX(
                    times[ii], obs=[obs.x(times[ii]), obs.y(times[ii]), 0.0], ro=1.0
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.helioY(times, obs=obs, ro=1.0)[ii]
                - o.helioY(
                    times[ii], obs=[obs.x(times[ii]), obs.y(times[ii]), 0.0], ro=1.0
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    # Do non-zero Ysun case for planarOrbit, but giving FullOrbit for obs
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.0, 0.0, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.helioX(times, obs=obs, ro=1.0)[ii]
                - o.helioX(
                    times[ii],
                    obs=[obs.x(times[ii]), obs.y(times[ii]), obs.z(times[ii])],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.helioY(times, obs=obs, ro=1.0)[ii]
                - o.helioY(
                    times[ii],
                    obs=[obs.x(times[ii]), obs.y(times[ii]), obs.z(times[ii])],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    # Do non-zero Ysun case for FullOrbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.0, 0.0, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.helioX(times, obs=obs, ro=1.0)[ii]
                - o.helioX(
                    times[ii],
                    obs=[obs.x(times[ii]), obs.y(times[ii]), obs.z(times[ii])],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.helioY(times, obs=obs, ro=1.0)[ii]
                - o.helioY(
                    times[ii],
                    obs=[obs.x(times[ii]), obs.y(times[ii]), obs.z(times[ii])],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    return None


def test_orbit_obsvel_list_issue322():
    # Further tests of obs= list parameter for orbit output, incl. velocity
    # mainly in relation to issue #322
    from galpy.orbit import Orbit

    # The basic case, for a planar orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0])
    assert (
        numpy.fabs(o.U(obs=[1.0, 0.0, 0.0, 0.0, 0.7, 0.0], ro=1.0, vo=1.0) + 0.1)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[1.0, 0.0, 0.0, 0.0, 0.7, 0.0], ro=1.0, vo=1.0) - 0.5)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0])
    assert (
        numpy.fabs(o.U(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 0.7)
        < 10.0**-5.7
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 1.8)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 3.0 * numpy.pi / 2.0])
    assert (
        numpy.fabs(o.U(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 0.9)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) + 0.6)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Full orbit
    # The basic case, for a full orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 0.0])
    assert (
        numpy.fabs(o.U(obs=[1.0, 0.0, 0.0, 0.0, 0.7, 0.0], ro=1.0, vo=1.0) + 0.1)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[1.0, 0.0, 0.0, 0.0, 0.7, 0.0], ro=1.0, vo=1.0) - 0.5)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0])
    assert (
        numpy.fabs(o.U(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 0.7)
        < 10.0**-5.5
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 1.8)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 3.0 * numpy.pi / 2.0])
    assert (
        numpy.fabs(o.U(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) - 0.9)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"
    assert (
        numpy.fabs(o.V(obs=[0.0, 1.0, 0.0, 0.6, 0.8, 0.0], ro=1.0, vo=1.0) + 0.6)
        < 10.0**-6.0
    ), "Relative position wrt the Sun from using obs= keyword does not work as expected"

    return None


def test_orbit_obsvel_Orbit_issue322():
    # Further tests of obs= Orbit parameter for orbit output, incl. velocity
    # mainly in relation to issue #322
    from galpy.orbit import Orbit

    # The basic case, for a planar orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, 0.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.U(obs=obs, ro=1.0, vo=1.0) + 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.V(obs=obs, ro=1.0, vo=1.0) - 0.5) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, numpy.pi / 2.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.helioX(obs=obs, ro=1.0) - 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=obs, ro=1.0)) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 3.0 * numpy.pi / 2.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, 3.0 * numpy.pi / 2.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.helioX(obs=obs, ro=1.0) - 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=obs, ro=1.0)) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Full orbit
    # The basic case, for a full orbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 0.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, 0.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.U(obs=obs, ro=1.0, vo=1.0) + 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.V(obs=obs, ro=1.0, vo=1.0) - 0.5) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, numpy.pi / 2.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.helioX(obs=obs, ro=1.0) - 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=obs, ro=1.0)) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    # Now use non-zero Ysun
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, 3.0 * numpy.pi / 2.0])
    obs = Orbit([1.0, 0.0, 0.7, 0.0, 0.0, 3.0 * numpy.pi / 2.0], ro=1.0, vo=1.0)
    assert numpy.fabs(o.helioX(obs=obs, ro=1.0) - 0.1) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    assert numpy.fabs(o.helioY(obs=obs, ro=1.0)) < 10.0**-6.0, (
        "Relative position wrt the Sun from using obs= keyword does not work as expected"
    )
    return None


def test_orbit_obsvel_Orbits_issue322():
    # Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322; specific case where the orbit gets evaluated at multiple
    # times; for velocity
    from galpy.orbit import Orbit

    # Do non-zero Ysun case for planarOrbit
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.5, 1.3, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.U(times, obs=obs, ro=1.0)[ii]
                - o.U(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        0.0,
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        0.0,
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.V(times, obs=obs, ro=1.0)[ii]
                - o.V(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        0.0,
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        0.0,
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    # Do non-zero Ysun case for planarOrbit, but giving FullOrbit for obs
    o = Orbit([0.9, 0.1, 1.2, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.5, 1.3, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.U(times, obs=obs, ro=1.0)[ii]
                - o.U(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        obs.z(times[ii]),
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        obs.vz(times[ii]),
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.V(times, obs=obs, ro=1.0)[ii]
                - o.V(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        obs.z(times[ii]),
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        obs.vz(times[ii]),
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    # Do non-zero Ysun case for FullOrbit
    o = Orbit([0.9, 0.1, 1.2, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    obs = Orbit([1.0, 0.5, 1.3, 0.0, 0.0, numpy.pi / 2.0], ro=1.0)
    times = numpy.linspace(0.0, 2.0, 2)
    from galpy.potential import MWPotential2014

    o.integrate(times, MWPotential2014)
    obs.integrate(times, MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert (
            numpy.fabs(
                o.U(times, obs=obs, ro=1.0)[ii]
                - o.U(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        obs.z(times[ii]),
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        obs.vz(times[ii]),
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
        assert (
            numpy.fabs(
                o.V(times, obs=obs, ro=1.0)[ii]
                - o.V(
                    times[ii],
                    obs=[
                        obs.x(times[ii]),
                        obs.y(times[ii]),
                        obs.z(times[ii]),
                        obs.vx(times[ii]),
                        obs.vy(times[ii]),
                        obs.vz(times[ii]),
                    ],
                    ro=1.0,
                )
            )
            < 10.0**-10.0
        ), (
            "Relative position wrt the Sun from using obs= keyword does not work as expected"
        )
    return None


def test_orbit_dim_2dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit (using ~Plevne's example)
    from galpy.orbit import Orbit
    from galpy.util import conversion

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    ell_p = potential.EllipticalDiskPotential()
    pota = b_p + ell_p
    o = Orbit(vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0)
    ts = numpy.linspace(
        0.0, 3.5 / conversion.time_in_Gyr(vo=220.0, ro=8.0), 1000, endpoint=True
    )
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="odeint")
    return None


def test_orbit_dim_1dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit
    from galpy.util import conversion

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    pota = potential.RZToverticalPotential(b_p, 1.1)
    o = Orbit(vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0)
    ts = numpy.linspace(
        0.0, 3.5 / conversion.time_in_Gyr(vo=220.0, ro=8.0), 1000, endpoint=True
    )
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="odeint")
    return None


def test_orbit_dim_1dPot_2dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    pota = potential.linearCompositePotential([b_p.toVertical(1.1)])
    o = Orbit(vxvv=[1.1, 0.1, 1.1, 0.1])
    ts = numpy.linspace(0.0, 10.0, 1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="dop853")
    return None


def test_orbit_dim_3dPot_1dOrb():
    # Test that orbit integration throws an error when using a 3D potential
    # for a 1D orbit
    from galpy.orbit import Orbit

    pota = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    o = Orbit(vxvv=[1.1, 0.1])
    ts = numpy.linspace(0.0, 10.0, 1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="dop853")
    return None


def test_orbit_dim_2dPot_1dOrb():
    # Test that orbit integration throws an error when using a 2D potential
    # for a 1D orbit
    from galpy.orbit import Orbit

    pota = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    ).toPlanar()
    o = Orbit(vxvv=[1.1, 0.1])
    ts = numpy.linspace(0.0, 10.0, 1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="dop853")
    return None


# Test whether ro warning is sounded when calling ra etc.
def test_orbit_radecetc_roWarning():
    from galpy.orbit import Orbit

    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 0.2])
    check_radecetc_roWarning(o, "ra")
    check_radecetc_roWarning(o, "dec")
    check_radecetc_roWarning(o, "ll")
    check_radecetc_roWarning(o, "bb")
    check_radecetc_roWarning(o, "dist")
    check_radecetc_roWarning(o, "pmra")
    check_radecetc_roWarning(o, "pmdec")
    check_radecetc_roWarning(o, "pmll")
    check_radecetc_roWarning(o, "pmbb")
    check_radecetc_roWarning(o, "vra")
    check_radecetc_roWarning(o, "vdec")
    check_radecetc_roWarning(o, "vll")
    check_radecetc_roWarning(o, "vbb")
    check_radecetc_roWarning(o, "helioX")
    check_radecetc_roWarning(o, "helioY")
    check_radecetc_roWarning(o, "helioZ")
    check_radecetc_roWarning(o, "U")
    check_radecetc_roWarning(o, "V")
    check_radecetc_roWarning(o, "W")
    return None


# Test whether vo warning is sounded when calling pmra etc.
def test_orbit_radecetc_voWarning():
    from galpy.orbit import Orbit

    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 0.2])
    check_radecetc_voWarning(o, "pmra")
    check_radecetc_voWarning(o, "pmdec")
    check_radecetc_voWarning(o, "pmll")
    check_radecetc_voWarning(o, "pmbb")
    check_radecetc_voWarning(o, "vra")
    check_radecetc_voWarning(o, "vdec")
    check_radecetc_voWarning(o, "vll")
    check_radecetc_voWarning(o, "vbb")
    check_radecetc_voWarning(o, "U")
    check_radecetc_voWarning(o, "V")
    check_radecetc_voWarning(o, "W")
    return None


# Test whether orbit evaluation methods sound warning when called with
# unitless time when orbit is integrated with unitfull times
def test_orbit_method_integrate_t_asQuantity_warning():
    from astropy import units

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Setup and integrate orbit
    ts = numpy.linspace(0.0, 10.0, 1001) * units.Gyr
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 0.2])
    o.integrate(ts, MWPotential2014)
    # Now check
    check_integrate_t_asQuantity_warning(o, "R")
    check_integrate_t_asQuantity_warning(o, "vR")
    check_integrate_t_asQuantity_warning(o, "vT")
    check_integrate_t_asQuantity_warning(o, "z")
    check_integrate_t_asQuantity_warning(o, "vz")
    check_integrate_t_asQuantity_warning(o, "phi")
    check_integrate_t_asQuantity_warning(o, "r")
    check_integrate_t_asQuantity_warning(o, "x")
    check_integrate_t_asQuantity_warning(o, "y")
    check_integrate_t_asQuantity_warning(o, "vx")
    check_integrate_t_asQuantity_warning(o, "vy")
    check_integrate_t_asQuantity_warning(o, "theta")
    check_integrate_t_asQuantity_warning(o, "vtheta")
    check_integrate_t_asQuantity_warning(o, "vr")
    check_integrate_t_asQuantity_warning(o, "ra")
    check_integrate_t_asQuantity_warning(o, "dec")
    check_integrate_t_asQuantity_warning(o, "ll")
    check_integrate_t_asQuantity_warning(o, "bb")
    check_integrate_t_asQuantity_warning(o, "dist")
    check_integrate_t_asQuantity_warning(o, "pmra")
    check_integrate_t_asQuantity_warning(o, "pmdec")
    check_integrate_t_asQuantity_warning(o, "pmll")
    check_integrate_t_asQuantity_warning(o, "pmbb")
    check_integrate_t_asQuantity_warning(o, "vra")
    check_integrate_t_asQuantity_warning(o, "vdec")
    check_integrate_t_asQuantity_warning(o, "vll")
    check_integrate_t_asQuantity_warning(o, "vbb")
    check_integrate_t_asQuantity_warning(o, "vlos")
    check_integrate_t_asQuantity_warning(o, "helioX")
    check_integrate_t_asQuantity_warning(o, "helioY")
    check_integrate_t_asQuantity_warning(o, "helioZ")
    check_integrate_t_asQuantity_warning(o, "U")
    check_integrate_t_asQuantity_warning(o, "V")
    check_integrate_t_asQuantity_warning(o, "W")
    check_integrate_t_asQuantity_warning(o, "E")
    check_integrate_t_asQuantity_warning(o, "L")
    check_integrate_t_asQuantity_warning(o, "Jacobi")
    check_integrate_t_asQuantity_warning(o, "ER")
    check_integrate_t_asQuantity_warning(o, "Ez")
    return None


# Test whether ro in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputro_quantity():
    from astropy import units

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.1, 0.1, 1.1, 0.2, 0.3, 0.3])
    ro = 11.0
    assert (
        numpy.fabs(
            o.E(pot=MWPotential2014, ro=ro * units.kpc)
            - o.E(pot=MWPotential2014, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.ER(pot=MWPotential2014, ro=ro * units.kpc)
            - o.ER(pot=MWPotential2014, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Ez(pot=MWPotential2014, ro=ro * units.kpc)
            - o.Ez(pot=MWPotential2014, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014, ro=ro * units.kpc)
            - o.Jacobi(pot=MWPotential2014, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value when input ro is Quantity"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014, ro=ro * units.kpc)
            - o.L(pot=MWPotential2014, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True, ro=ro * units.kpc)
            - o.rap(pot=MWPotential2014, analytic=True, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True, ro=ro * units.kpc)
            - o.rperi(pot=MWPotential2014, analytic=True, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True, ro=ro * units.kpc)
            - o.zmax(pot=MWPotential2014, analytic=True, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.jr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.jr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.jp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.jp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.jz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.jz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.wr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.wr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.wp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.wp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.wz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.wz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Or(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Or(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Op(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Op(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value when input ro is Quantity"
    assert (
        numpy.fabs(
            o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro * units.kpc)
            - o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5, ro=ro)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value when input ro is Quantity"
    assert numpy.fabs(o.time(ro=ro * units.kpc) - o.time(ro=ro)) < 10.0**-8.0, (
        "Orbit method time does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.R(ro=ro * units.kpc) - o.R(ro=ro)) < 10.0**-8.0, (
        "Orbit method R does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vR(ro=ro * units.kpc) - o.vR(ro=ro)) < 10.0**-8.0, (
        "Orbit method vR does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vT(ro=ro * units.kpc) - o.vT(ro=ro)) < 10.0**-8.0, (
        "Orbit method vT does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.z(ro=ro * units.kpc) - o.z(ro=ro)) < 10.0**-8.0, (
        "Orbit method z does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vz(ro=ro * units.kpc) - o.vz(ro=ro)) < 10.0**-8.0, (
        "Orbit method vz does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.phi(ro=ro * units.kpc) - o.phi(ro=ro)) < 10.0**-8.0, (
        "Orbit method phi does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vphi(ro=ro * units.kpc) - o.vphi(ro=ro)) < 10.0**-8.0, (
        "Orbit method vphi does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.x(ro=ro * units.kpc) - o.x(ro=ro)) < 10.0**-8.0, (
        "Orbit method x does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.y(ro=ro * units.kpc) - o.y(ro=ro)) < 10.0**-8.0, (
        "Orbit method y does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vx(ro=ro * units.kpc) - o.vx(ro=ro)) < 10.0**-8.0, (
        "Orbit method vx does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vy(ro=ro * units.kpc) - o.vy(ro=ro)) < 10.0**-8.0, (
        "Orbit method vy does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.theta(ro=ro * units.kpc) - o.theta(ro=ro)) < 10.0**-8.0, (
        "Orbit method theta does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vtheta(ro=ro * units.kpc) - o.vtheta(ro=ro)) < 10.0**-8.0, (
        "Orbit method vtheta does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vr(ro=ro * units.kpc) - o.vr(ro=ro)) < 10.0**-8.0, (
        "Orbit method vr does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.ra(ro=ro * units.kpc) - o.ra(ro=ro)) < 10.0**-8.0, (
        "Orbit method ra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.dec(ro=ro * units.kpc) - o.dec(ro=ro)) < 10.0**-8.0, (
        "Orbit method dec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.ll(ro=ro * units.kpc) - o.ll(ro=ro)) < 10.0**-8.0, (
        "Orbit method ll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.bb(ro=ro * units.kpc) - o.bb(ro=ro)) < 10.0**-8.0, (
        "Orbit method bb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.dist(ro=ro * units.kpc) - o.dist(ro=ro)) < 10.0**-8.0, (
        "Orbit method dist does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmra(ro=ro * units.kpc) - o.pmra(ro=ro)) < 10.0**-8.0, (
        "Orbit method pmra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmdec(ro=ro * units.kpc) - o.pmdec(ro=ro)) < 10.0**-8.0, (
        "Orbit method pmdec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmll(ro=ro * units.kpc) - o.pmll(ro=ro)) < 10.0**-8.0, (
        "Orbit method pmll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmbb(ro=ro * units.kpc) - o.pmbb(ro=ro)) < 10.0**-8.0, (
        "Orbit method pmbb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vlos(ro=ro * units.kpc) - o.vlos(ro=ro)) < 10.0**-8.0, (
        "Orbit method vlos does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vra(ro=ro * units.kpc) - o.vra(ro=ro)) < 10.0**-8.0, (
        "Orbit method vra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vdec(ro=ro * units.kpc) - o.vdec(ro=ro)) < 10.0**-8.0, (
        "Orbit method vdec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vll(ro=ro * units.kpc) - o.vll(ro=ro)) < 10.0**-8.0, (
        "Orbit method vll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vbb(ro=ro * units.kpc) - o.vbb(ro=ro)) < 10.0**-8.0, (
        "Orbit method vbb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioX(ro=ro * units.kpc) - o.helioX(ro=ro)) < 10.0**-8.0, (
        "Orbit method helioX does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioY(ro=ro * units.kpc) - o.helioY(ro=ro)) < 10.0**-8.0, (
        "Orbit method helioY does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioZ(ro=ro * units.kpc) - o.helioZ(ro=ro)) < 10.0**-8.0, (
        "Orbit method helioZ does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.U(ro=ro * units.kpc) - o.U(ro=ro)) < 10.0**-8.0, (
        "Orbit method U does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.V(ro=ro * units.kpc) - o.V(ro=ro)) < 10.0**-8.0, (
        "Orbit method V does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.W(ro=ro * units.kpc) - o.W(ro=ro)) < 10.0**-8.0, (
        "Orbit method W does not return the correct value when input ro is Quantity"
    )
    return None


# Test whether vo in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputvo_quantity():
    from astropy import units

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.1, 0.1, 1.1, 0.2, 0.3, 0.3])
    vo = 222.0
    assert (
        numpy.fabs(
            o.E(pot=MWPotential2014, vo=vo * units.km / units.s)
            - o.E(pot=MWPotential2014, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.ER(pot=MWPotential2014, vo=vo * units.km / units.s)
            - o.ER(pot=MWPotential2014, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Ez(pot=MWPotential2014, vo=vo * units.km / units.s)
            - o.Ez(pot=MWPotential2014, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014, vo=vo * units.km / units.s)
            - o.Jacobi(pot=MWPotential2014, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value when input vo is Quantity"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014, vo=vo * units.km / units.s)
            - o.L(pot=MWPotential2014, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True, vo=vo * units.km / units.s)
            - o.rap(pot=MWPotential2014, analytic=True, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True, vo=vo * units.km / units.s)
            - o.rperi(pot=MWPotential2014, analytic=True, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True, vo=vo * units.km / units.s)
            - o.zmax(pot=MWPotential2014, analytic=True, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.jr(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.jr(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.jp(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.jp(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.jz(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.jz(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.wr(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.wr(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.wp(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.wp(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.wz(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.wz(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Tr(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Tp(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Tz(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Or(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Or(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Op(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Op(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(
            o.Oz(
                pot=MWPotential2014,
                type="staeckel",
                delta=0.5,
                vo=vo * units.km / units.s,
            )
            - o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5, vo=vo)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.time(vo=vo * units.km / units.s) - o.time(vo=vo)) < 10.0**-8.0
    ), "Orbit method time does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.R(vo=vo * units.km / units.s) - o.R(vo=vo)) < 10.0**-8.0, (
        "Orbit method R does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vR(vo=vo * units.km / units.s) - o.vR(vo=vo)) < 10.0**-8.0, (
        "Orbit method vR does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vT(vo=vo * units.km / units.s) - o.vT(vo=vo)) < 10.0**-8.0, (
        "Orbit method vT does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.z(vo=vo * units.km / units.s) - o.z(vo=vo)) < 10.0**-8.0, (
        "Orbit method z does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vz(vo=vo * units.km / units.s) - o.vz(vo=vo)) < 10.0**-8.0, (
        "Orbit method vz does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.phi(vo=vo * units.km / units.s) - o.phi(vo=vo)) < 10.0**-8.0, (
        "Orbit method phi does not return the correct value when input vo is Quantity"
    )
    assert (
        numpy.fabs(o.vphi(vo=vo * units.km / units.s) - o.vphi(vo=vo)) < 10.0**-8.0
    ), "Orbit method vphi does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.x(vo=vo * units.km / units.s) - o.x(vo=vo)) < 10.0**-8.0, (
        "Orbit method x does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.y(vo=vo * units.km / units.s) - o.y(vo=vo)) < 10.0**-8.0, (
        "Orbit method y does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vx(vo=vo * units.km / units.s) - o.vx(vo=vo)) < 10.0**-8.0, (
        "Orbit method vx does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vy(vo=vo * units.km / units.s) - o.vy(vo=vo)) < 10.0**-8.0, (
        "Orbit method vy does not return the correct value when input vo is Quantity"
    )
    assert (
        numpy.fabs(o.theta(vo=vo * units.km / units.s) - o.theta(vo=vo)) < 10.0**-8.0
    ), "Orbit method theta does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.vtheta(vo=vo * units.km / units.s) - o.vtheta(vo=vo)) < 10.0**-8.0
    ), "Orbit method vtheta does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.vr(vo=vo * units.km / units.s) - o.vr(vo=vo)) < 10.0**-8.0, (
        "Orbit method vr does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.ra(vo=vo * units.km / units.s) - o.ra(vo=vo)) < 10.0**-8.0, (
        "Orbit method ra does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.dec(vo=vo * units.km / units.s) - o.dec(vo=vo)) < 10.0**-8.0, (
        "Orbit method dec does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.ll(vo=vo * units.km / units.s) - o.ll(vo=vo)) < 10.0**-8.0, (
        "Orbit method ll does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.bb(vo=vo * units.km / units.s) - o.bb(vo=vo)) < 10.0**-8.0, (
        "Orbit method bb does not return the correct value when input vo is Quantity"
    )
    assert (
        numpy.fabs(o.dist(vo=vo * units.km / units.s) - o.dist(vo=vo)) < 10.0**-8.0
    ), "Orbit method dist does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.pmra(vo=vo * units.km / units.s) - o.pmra(vo=vo)) < 10.0**-8.0
    ), "Orbit method pmra does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.pmdec(vo=vo * units.km / units.s) - o.pmdec(vo=vo)) < 10.0**-8.0
    ), "Orbit method pmdec does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.pmll(vo=vo * units.km / units.s) - o.pmll(vo=vo)) < 10.0**-8.0
    ), "Orbit method pmll does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.pmbb(vo=vo * units.km / units.s) - o.pmbb(vo=vo)) < 10.0**-8.0
    ), "Orbit method pmbb does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.vlos(vo=vo * units.km / units.s) - o.vlos(vo=vo)) < 10.0**-8.0
    ), "Orbit method vlos does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.vra(vo=vo * units.km / units.s) - o.vra(vo=vo)) < 10.0**-8.0, (
        "Orbit method vra does not return the correct value when input vo is Quantity"
    )
    assert (
        numpy.fabs(o.vdec(vo=vo * units.km / units.s) - o.vdec(vo=vo)) < 10.0**-8.0
    ), "Orbit method vdec does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.vll(vo=vo * units.km / units.s) - o.vll(vo=vo)) < 10.0**-8.0, (
        "Orbit method vll does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.vbb(vo=vo * units.km / units.s) - o.vbb(vo=vo)) < 10.0**-8.0, (
        "Orbit method vbb does not return the correct value when input vo is Quantity"
    )
    assert (
        numpy.fabs(o.helioX(vo=vo * units.km / units.s) - o.helioX(vo=vo)) < 10.0**-8.0
    ), "Orbit method helioX does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.helioY(vo=vo * units.km / units.s) - o.helioY(vo=vo)) < 10.0**-8.0
    ), "Orbit method helioY does not return the correct value when input vo is Quantity"
    assert (
        numpy.fabs(o.helioZ(vo=vo * units.km / units.s) - o.helioZ(vo=vo)) < 10.0**-8.0
    ), "Orbit method helioZ does not return the correct value when input vo is Quantity"
    assert numpy.fabs(o.U(vo=vo * units.km / units.s) - o.U(vo=vo)) < 10.0**-8.0, (
        "Orbit method U does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.V(vo=vo * units.km / units.s) - o.V(vo=vo)) < 10.0**-8.0, (
        "Orbit method V does not return the correct value when input vo is Quantity"
    )
    assert numpy.fabs(o.W(vo=vo * units.km / units.s) - o.W(vo=vo)) < 10.0**-8.0, (
        "Orbit method W does not return the correct value when input vo is Quantity"
    )
    return None


# Test whether obs in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputobs_quantity():
    from astropy import units

    from galpy.orbit import Orbit

    o = Orbit([1.1, 0.1, 1.1, 0.2, 0.3, 0.3])
    obs = [11.0, 0.1, 0.2, -10.0, 245.0, 7.0]
    obs_units = [
        11.0 * units.kpc,
        0.1 * units.kpc,
        0.2 * units.kpc,
        -10.0 * units.km / units.s,
        245.0 * units.km / units.s,
        7.0 * units.km / units.s,
    ]
    assert numpy.fabs(o.ra(obs=obs_units) - o.ra(obs=obs)) < 10.0**-8.0, (
        "Orbit method ra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.dec(obs=obs_units) - o.dec(obs=obs)) < 10.0**-8.0, (
        "Orbit method dec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.ll(obs=obs_units) - o.ll(obs=obs)) < 10.0**-8.0, (
        "Orbit method ll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.bb(obs=obs_units) - o.bb(obs=obs)) < 10.0**-8.0, (
        "Orbit method bb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.dist(obs=obs_units) - o.dist(obs=obs)) < 10.0**-8.0, (
        "Orbit method dist does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmra(obs=obs_units) - o.pmra(obs=obs)) < 10.0**-8.0, (
        "Orbit method pmra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmdec(obs=obs_units) - o.pmdec(obs=obs)) < 10.0**-8.0, (
        "Orbit method pmdec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmll(obs=obs_units) - o.pmll(obs=obs)) < 10.0**-8.0, (
        "Orbit method pmll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.pmbb(obs=obs_units) - o.pmbb(obs=obs)) < 10.0**-8.0, (
        "Orbit method pmbb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vlos(obs=obs_units) - o.vlos(obs=obs)) < 10.0**-8.0, (
        "Orbit method vlos does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vra(obs=obs_units) - o.vra(obs=obs)) < 10.0**-8.0, (
        "Orbit method vra does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vdec(obs=obs_units) - o.vdec(obs=obs)) < 10.0**-8.0, (
        "Orbit method vdec does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vll(obs=obs_units) - o.vll(obs=obs)) < 10.0**-8.0, (
        "Orbit method vll does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.vbb(obs=obs_units) - o.vbb(obs=obs)) < 10.0**-8.0, (
        "Orbit method vbb does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioX(obs=obs_units) - o.helioX(obs=obs)) < 10.0**-8.0, (
        "Orbit method helioX does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioY(obs=obs_units) - o.helioY(obs=obs)) < 10.0**-8.0, (
        "Orbit method helioY does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.helioZ(obs=obs_units) - o.helioZ(obs=obs)) < 10.0**-8.0, (
        "Orbit method helioZ does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.U(obs=obs_units) - o.U(obs=obs)) < 10.0**-8.0, (
        "Orbit method U does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.V(obs=obs_units) - o.V(obs=obs)) < 10.0**-8.0, (
        "Orbit method V does not return the correct value when input ro is Quantity"
    )
    assert numpy.fabs(o.W(obs=obs_units) - o.W(obs=obs)) < 10.0**-8.0, (
        "Orbit method W does not return the correct value when input ro is Quantity"
    )
    return None


# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_full():
    integrators = [
        "dopr54_c",
        "leapfrog_c",
        "dop853_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    scriptpath = "orbitint4sigint.py"
    if not "tests" in os.getcwd():
        scriptpath = os.path.join("tests", scriptpath)
    ntries = 10
    for integrator in integrators:
        p = subprocess.Popen(
            ["python", scriptpath, integrator, "full"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for line in iter(p.stdout.readline, b""):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid, signal.SIGINT)
        time.sleep(1)
        cnt = 0
        while p.poll() is None and cnt < ntries:  # wait a little longer
            time.sleep(4)
            cnt += 1

        if p.poll() == 2 and WIN32:
            break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None:
                msg = -100
            else:
                msg = p.poll()
            raise AssertionError(
                "Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i"
                % (integrator, msg)
            )
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None


# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_planar():
    integrators = [
        "dopr54_c",
        "leapfrog_c",
        "dop853_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "ias15_c",
    ]
    scriptpath = "orbitint4sigint.py"
    if not "tests" in os.getcwd():
        scriptpath = os.path.join("tests", scriptpath)
    ntries = 10
    for integrator in integrators:
        p = subprocess.Popen(
            ["python", scriptpath, integrator, "planar"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for line in iter(p.stdout.readline, b""):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid, signal.SIGINT)
        time.sleep(1)
        cnt = 0
        while p.poll() is None and cnt < ntries:  # wait a little longer
            time.sleep(4)
            cnt += 1

        if p.poll() == 2 and WIN32:
            break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None:
                msg = -100
            else:
                msg = p.poll()
            raise AssertionError(
                "Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i"
                % (integrator, msg)
            )
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None


# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_planardxdv():
    integrators = ["dopr54_c", "rk4_c", "rk6_c", "dop853_c"]
    scriptpath = "orbitint4sigint.py"
    if not "tests" in os.getcwd():
        scriptpath = os.path.join("tests", scriptpath)
    ntries = 10
    for integrator in integrators:
        p = subprocess.Popen(
            ["python", scriptpath, integrator, "planardxdv"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for line in iter(p.stdout.readline, b""):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid, signal.SIGINT)
        time.sleep(1)
        cnt = 0
        while p.poll() is None and cnt < ntries:  # wait a little longer
            time.sleep(4)
            cnt += 1

        if p.poll() == 2 and WIN32:
            break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None:
                msg = -100
            else:
                msg = p.poll()
            raise AssertionError(
                "Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i"
                % (integrator, msg)
            )
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None


def test_orbitint_pythonfallback():
    # Check if a warning is raised when the potential has no C integrator
    from galpy.orbit import Orbit

    bp = (
        BurkertPotentialNoC()
    )  # BurkertPotentialNoC is already imported at the top of test_orbit.py
    bp.normalize(1.0)
    ts = numpy.linspace(0.0, 1.0, 101)
    for orb in [
        Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 1.0]),
        Orbit([1.0, 0.1, 1.1, 0.1, 0.0]),
        Orbit([1.0, 0.1, 1.1, 1.0]),
        Orbit([1.0, 0.1, 1.1]),
    ]:
        with pytest.warns(galpyWarning) as record:
            if PY2:
                reset_warning_registry("galpy")
            warnings.simplefilter("always", galpyWarning)
            # Test w/ dopr54_c
            orb.integrate(ts, bp, method="dopr54_c")
        raisedWarning = False
        for rec in record:
            # check that the message matches
            print(rec.message.args[0])
            raisedWarning += (
                str(rec.message.args[0])
                == "Cannot use C integration because some of the potentials are not implemented in C (using odeint instead)"
            )
        assert raisedWarning, "Orbit integration did not raise fallback warning"
    return None


def test_orbitint_dissipativefallback():
    # Check if a warning is raised when one tries to integrate an orbit
    # in a dissipative force law with a symplectic integrator
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, dens=lp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    ts = numpy.linspace(0.0, 1.0, 101)
    for orb in [Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 1.0])]:
        with pytest.warns(galpyWarning) as record:
            orb.integrate(ts, lp + cdf, method="leapfrog")
        raisedWarning = False
        for rec in record:
            # check that the message matches
            raisedWarning += (
                str(rec.message.args[0])
                == "Cannot use symplectic integration because some of the included forces are dissipative (using non-symplectic integrator odeint instead)"
            )
        assert raisedWarning, (
            "Orbit integration with symplectic integrator for dissipative force did not raise fallback warning"
        )
    return None


# Test that the functions that supposedly *always* return output in physical
# units actually do so; see issue #294
def test_intrinsic_physical_output():
    from galpy.orbit import Orbit
    from galpy.util import coords

    o = Orbit(
        [0.9, 0.0, 1.0, 0.0, 0.2, 0.0],
        ro=8.0,
        vo=220.0,
        zo=0.0,
        solarmotion=[-20.0, 30.0, 40.0],
    )
    # 04/2018: not quite anylonger w/ astropy def. of plane, but close
    l_true = 0.0
    b_true = 0.0
    ra_true, dec_true = coords.lb_to_radec(l_true, b_true, degree=True, epoch=None)
    assert numpy.fabs(o.ra() - ra_true) < 10.0**-3.8, (
        "Orbit.ra does not return correct ra in degree"
    )
    assert numpy.fabs(o.dec() - dec_true) < 10.0**-3.8, (
        "Orbit.dec does not return correct dec in degree"
    )
    assert numpy.fabs(o.ll() - l_true) < 10.0**-4.0, (
        "Orbit.ll does not return correct l in degree"
    )
    assert numpy.fabs(o.bb() - b_true) < 10.0**-4.0, (
        "Orbit.bb does not return correct b in degree"
    )
    assert numpy.fabs(o.dist() - 0.8) < 10.0**-8.0, (
        "Orbit.dist does not return correct dist in kpc"
    )
    pmll_true = -30.0 / 0.8 / _K
    pmbb_true = 4.0 / 0.8 / _K
    pmra_true, pmdec_true = coords.pmllpmbb_to_pmrapmdec(
        pmll_true, pmbb_true, l_true, b_true, degree=True, epoch=None
    )
    assert numpy.fabs(o.pmra() - pmra_true) < 10.0**-5.0, (
        "Orbit.pmra does not return correct pmra in mas/yr"
    )
    assert numpy.fabs(o.pmdec() - pmdec_true) < 10.0**-5.0, (
        "Orbit.pmdec does not return correct pmdec in mas/yr"
    )
    assert numpy.fabs(o.pmll() - pmll_true) < 10.0**-5.0, (
        "Orbit.pmll does not return correct pmll in mas/yr"
    )
    assert numpy.fabs(o.pmbb() - pmbb_true) < 10.0**-4.7, (
        "Orbit.pmbb does not return correct pmbb in mas/yr"
    )
    assert numpy.fabs(o.vra() - pmra_true * 0.8 * _K) < 10.0**-4.8, (
        "Orbit.vra does not return correct vra in km/s"
    )
    assert numpy.fabs(o.vdec() - pmdec_true * 0.8 * _K) < 10.0**-4.6, (
        "Orbit.vdec does not return correct vdec in km/s"
    )
    assert numpy.fabs(o.vll() - pmll_true * 0.8 * _K) < 10.0**-5.0, (
        "Orbit.vll does not return correct vll in km/s"
    )
    assert numpy.fabs(o.vbb() - pmbb_true * 0.8 * _K) < 10.0**-4.0, (
        "Orbit.vbb does not return correct vbb in km/s"
    )
    assert numpy.fabs(o.vlos() + 20.0) < 10.0**-8.0, (
        "Orbit.vlos does not return correct vlos in km/s"
    )
    assert numpy.fabs(o.U() + 20.0) < 10.0**-4.0, (
        "Orbit.U does not return correct U in km/s"
    )
    assert numpy.fabs(o.V() - pmll_true * 0.8 * _K) < 10.0**-4.8, (
        "Orbit.V does not return correct V in km/s"
    )
    assert numpy.fabs(o.W() - pmbb_true * 0.8 * _K) < 10.0**-4.0, (
        "Orbit.W does not return correct W in km/s"
    )
    assert numpy.fabs(o.helioX() - 0.8) < 10.0**-8.0, (
        "Orbit.helioX does not return correct helioX in kpc"
    )
    # For non-trivial helioY and helioZ tests
    o = Orbit(
        [1.0 / numpy.sqrt(2.0), 0.0, 1.0, 0.0, 0.2, numpy.pi / 4.0],
        ro=8.0,
        vo=220.0,
        zo=0.0,
        solarmotion=[-20.0, 30.0, 40.0],
    )
    assert numpy.fabs(o.helioY() - 4.0) < 10.0**-5.0, (
        "Orbit.helioY does not return correct helioY in kpc"
    )
    o = Orbit(
        [0.9, 0.0, 1.0, 0.3, 0.2, numpy.pi / 4.0],
        ro=8.0,
        vo=220.0,
        zo=0.0,
        solarmotion=[-20.0, 30.0, 40.0],
    )
    assert numpy.fabs(o.helioZ() - 0.3 * 8.0) < 10.0**-4.8, (
        "Orbit.helioZ does not return correct helioZ in kpc"
    )
    return None


def test_doublewrapper_2d():
    # Test that a doubly-wrapped potential gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
    )

    # potential= flat vc + doubly-wrapped bar
    pot = LogarithmicHaloPotential(normalize=1.0) + SolidBodyRotationWrapperPotential(
        pot=DehnenSmoothWrapperPotential(
            pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
            tform=5.0,
            tsteady=15.0,
        ),
        omega=1.3,
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_doublewrapper_3d():
    # Test that a doubly-wrapped potential gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
    )

    # potential= flat vc + doubly-wrapped bar
    pot = LogarithmicHaloPotential(normalize=1.0) + SolidBodyRotationWrapperPotential(
        pot=DehnenSmoothWrapperPotential(
            pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
            tform=5.0,
            tsteady=15.0,
        ),
        omega=1.3,
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.z() - oc.z()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vz() - oc.vz()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_wrapper_followedbyanotherpotential_2d():
    # Test that a wrapped potential that gets followed by another potential
    # gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
        SpiralArmsPotential,
    )

    # potential= flat vc + doubly-wrapped bar
    pot = (
        LogarithmicHaloPotential(normalize=1.0)
        + SolidBodyRotationWrapperPotential(
            pot=DehnenSmoothWrapperPotential(
                pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
                tform=5.0,
                tsteady=15.0,
            ),
            omega=1.3,
        )
        + SpiralArmsPotential(N=4, omega=0.79, amp=0.9)
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_wrapper_followedbyanotherpotential_3d():
    # Test that a wrapped potential that gets followed by another potential
    # gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
        SpiralArmsPotential,
    )

    # potential= flat vc + doubly-wrapped bar
    pot = (
        LogarithmicHaloPotential(normalize=1.0)
        + SolidBodyRotationWrapperPotential(
            pot=DehnenSmoothWrapperPotential(
                pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
                tform=5.0,
                tsteady=15.0,
            ),
            omega=1.3,
        )
        + SpiralArmsPotential(N=4, omega=0.79, amp=0.9)
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.z() - oc.z()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vz() - oc.vz()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_wrapper_complicatedsequence_2d():
    # Test that a complicated combination of potentials and wrapped potentials
    # gets passed to C correctly, by comparing orbit integrated in C to that
    # in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
        SpiralArmsPotential,
    )

    # potential= flat vc + doubly-wrapped bar + spiral-arms
    pot = (
        LogarithmicHaloPotential(normalize=0.2)
        + SolidBodyRotationWrapperPotential(
            pot=DehnenSmoothWrapperPotential(
                pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
                tform=5.0,
                tsteady=15.0,
            ),
            omega=1.3,
        )
        + DehnenSmoothWrapperPotential(
            pot=SpiralArmsPotential(N=4, omega=0.79, amp=0.9), tform=5.0, tsteady=15.0
        )
        + LogarithmicHaloPotential(normalize=0.8)
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_wrapper_complicatedsequence_3d():
    # Test that a complicated combination of potentials and wrapped potentials
    # gets passed to C correctly, by comparing orbit integrated in C to that
    # in python
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
        SolidBodyRotationWrapperPotential,
        SpiralArmsPotential,
    )

    # potential= flat vc + doubly-wrapped bar + spiral-arms
    pot = (
        LogarithmicHaloPotential(normalize=0.2)
        + SolidBodyRotationWrapperPotential(
            pot=DehnenSmoothWrapperPotential(
                pot=DehnenBarPotential(omegab=1.0, rb=5.0 / 8.0, Af=1.0 / 100.0),
                tform=5.0,
                tsteady=15.0,
            ),
            omega=1.3,
        )
        + DehnenSmoothWrapperPotential(
            pot=SpiralArmsPotential(N=4, omega=0.79, amp=0.9), tform=5.0, tsteady=15.0
        )
        + LogarithmicHaloPotential(normalize=0.8)
    )
    # Integrate orbit in C and python
    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    oc = o()
    ts = numpy.linspace(0.0, 20.0, 1001)
    o.integrate(ts, pot, method="leapfrog")
    oc.integrate(ts, pot, method="leapfrog_c")
    # Check that they end up in the same point
    o = o(ts[-1])
    oc = oc(ts[-1])
    assert numpy.fabs(o.x() - oc.x()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.y() - oc.y()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.z() - oc.z()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vx() - oc.vx()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vy() - oc.vy()) < 10.0**-4.0, (
        "Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large"
    )
    assert numpy.fabs(o.vz() - oc.vz()) < 10.0**-4.0, (
        "Final orbit position between C and Python integration of a doubly-wrapped orbit is too large"
    )
    return None


def test_orbit_sun_setup():
    # Test that setting up an Orbit with no vxvv returns the Orbit of the Sun
    from galpy.orbit import Orbit

    o = Orbit()
    assert numpy.fabs(o.dist()) < 1e-10, (
        "Orbit with no vxvv does not produce an orbit with zero distance"
    )
    assert numpy.fabs(o.vll()) < 1e-10, (
        "Orbit with no vxvv does not produce an orbit with zero velocity in the Galactic longitude direction"
    )
    assert numpy.fabs(o.vbb()) < 1e-10, (
        "Orbit with no vxvv does not produce an orbit with zero velocity in the Galactic latitude direction"
    )
    assert numpy.fabs(o.vlos()) < 1e-10, (
        "Orbit with no vxvv does not produce an orbit with zero line-of-sight velocity"
    )


def test_integrate_dxdv_errors():
    from galpy.orbit import Orbit

    ts = numpy.linspace(0.0, 10.0, 1001)
    # Test that attempting to use integrate_dxdv with a non-phasedim==4/6 orbit
    # raises error (1D and 3D-without-phi orbits)
    o = Orbit([1.0, 0.1])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None, ts, potential.toVertical(potential.MWPotential, 1.0))
    o = Orbit([1.0, 0.1, 1.0])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None, ts, potential.MWPotential)
    o = Orbit([1.0, 0.1, 1.0, 0.1, 0.1])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None, ts, potential.MWPotential)
    # Test that a random string as the integrator doesn't work (4D)
    o = Orbit([1.0, 0.1, 1.0, 3.0])
    with pytest.raises(ValueError) as excinfo:
        o.integrate_dxdv(
            None, ts, potential.MWPotential, method="some non-existent integrator"
        )
    # Test that a random string as the integrator doesn't work (6D)
    o = Orbit([1.0, 0.1, 1.0, 0.1, 0.1, 3.0])
    with pytest.raises(ValueError) as excinfo:
        o.integrate_dxdv(
            None, ts, potential.MWPotential, method="some non-existent integrator"
        )
    # Test that a symplectic integrator raises for 6D orbits
    o = Orbit([1.0, 0.1, 1.0, 0.1, 0.1, 3.0])
    with pytest.raises(ValueError) as excinfo:
        o.integrate_dxdv(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ts,
            potential.MWPotential,
            method="symplec4_c",
        )
    return None


# Test that the internal interpolator is reset when the orbit is re-integrated
def test_orbinterp_reset_integrate():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    op = o()
    ts = numpy.linspace(0.0, 100.0, 10001)
    o.integrate(ts, MWPotential)
    o.R(numpy.linspace(0.0, o.t[-1], 1001))
    o.integrate(ts, MWPotential2014)
    op.integrate(ts, MWPotential2014)
    # If things are reset correctly, o and op should now agree on everything
    assert numpy.all(numpy.fabs(o.R(ts) - op.R(ts)) < 10.0**-10.0), (
        "Orbit R not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vR(ts) - op.vR(ts)) < 10.0**-10.0), (
        "Orbit vR not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vT(ts) - op.vT(ts)) < 10.0**-10.0), (
        "Orbit vT not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.z(ts) - op.z(ts)) < 10.0**-10.0), (
        "Orbit z not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vz(ts) - op.vz(ts)) < 10.0**-10.0), (
        "Orbit vz not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.phi(ts) - op.phi(ts)) < 10.0**-10.0), (
        "Orbit phi not reset correctly"
    )
    assert numpy.fabs(o.rperi() - op.rperi()) < 10.0**-10.0, (
        "Orbit rperi not reset correctly"
    )
    assert numpy.fabs(o.rap() - op.rap()) < 10.0**-10.0, "Orbit rap not reset correctly"
    assert numpy.fabs(o.e() - op.e()) < 10.0**-10.0, "Orbit e not reset correctly"
    return None


# Test that the internal interpolator is reset when the orbit is re-integrated
# with integrate_SOS
def test_orbinterp_reset_integrateSOS():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    op = o()
    ts = numpy.linspace(0.0, 100.0, 10001)
    psis = numpy.linspace(0.0, 100.0, 10001)
    o.integrate(ts, MWPotential)
    o.R(numpy.linspace(0.0, o.t[-1], 1001))
    o.integrate_SOS(psis, MWPotential2014)
    op.integrate_SOS(psis, MWPotential2014)
    ts = o.t
    # If things are reset correctly, o and op should now agree on everything
    assert numpy.all(numpy.fabs(o.R(ts) - op.R(ts)) < 10.0**-10.0), (
        "Orbit R not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vR(ts) - op.vR(ts)) < 10.0**-10.0), (
        "Orbit vR not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vT(ts) - op.vT(ts)) < 10.0**-10.0), (
        "Orbit vT not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.z(ts) - op.z(ts)) < 10.0**-10.0), (
        "Orbit z not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vz(ts) - op.vz(ts)) < 10.0**-10.0), (
        "Orbit vz not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.phi(ts) - op.phi(ts)) < 10.0**-10.0), (
        "Orbit phi not reset correctly"
    )
    assert numpy.fabs(o.rperi() - op.rperi()) < 10.0**-10.0, (
        "Orbit rperi not reset correctly"
    )
    assert numpy.fabs(o.rap() - op.rap()) < 10.0**-10.0, "Orbit rap not reset correctly"
    assert numpy.fabs(o.e() - op.e()) < 10.0**-10.0, "Orbit e not reset correctly"
    return None


# Test that an off-grid query against a per-orbit Orbit whose stored grid
# contains an all-NaN row (e.g. an orbit that never crossed the bruteSOS
# surface) raises a clear, actionable ValueError rather than scipy's opaque
# "fpcurf0:m=0" error from the spline build.
def test_indiv_t_query_short_row_raises_clearly():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Two orbits: one normal, one with z=0, vz=0 — the latter never crosses
    # the SOS surface, so bruteSOS leaves it with an all-NaN row.
    o = Orbit(
        numpy.array(
            [
                [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
                [1.0, 0.1, 1.1, 0.0, 0.0, 0.0],
            ]
        )
    )
    o.bruteSOS(numpy.linspace(0.0, 50.0, 1001), MWPotential2014, method="dop853_c")
    t_grid = numpy.asarray(o.t)
    # Sanity: orbit 1 is the all-NaN one
    valid_per_orbit = numpy.sum(~numpy.isnan(t_grid), axis=-1)
    assert valid_per_orbit[0] >= 2, "first orbit should have crossings"
    assert valid_per_orbit[1] == 0, "second orbit (z=0,vz=0) should have no crossings"
    # Off-grid query → must raise with a clear message that names the orbit
    qq = numpy.array([[0.5, 1.5], [10.0, 20.0]])
    with pytest.raises(ValueError, match=r"orbit\(s\) \[1\]"):
        o.x(qq)
    # On-grid (fast-path) query is allowed and returns NaN for the empty row
    on_grid = o.x(t_grid)
    assert on_grid.shape == (2, t_grid.shape[1])
    assert numpy.isnan(on_grid[1]).all(), (
        "on-grid query of an all-NaN row should propagate NaN"
    )
    return None


# Test that integrating with a per-orbit time array on top of a previously
# integrated Orbit (or with a shared 1D t after a previous per-orbit
# integrate) does NOT trigger continuation: the per-orbit-t early-out in
# _should_continue_integration must fire and the new integrate must restart
# from the original initial conditions.
def test_indiv_t_disables_continuation():
    from galpy.orbit import Orbit

    pot = potential.MWPotential2014
    vxvvs = numpy.array(
        [
            [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
            [1.2, -0.05, 0.9, -0.1, 0.1, 0.5],
        ]
    )

    # 1) 1D-t integrate first, then per-orbit-t re-integrate.
    o = Orbit(vxvvs)
    o.integrate(numpy.linspace(0.0, 5.0, 501), pot, method="dop853_c")
    assert numpy.asarray(o.t).ndim == 1
    ts_pe = numpy.array([numpy.linspace(0.0, 3.0, 301), numpy.linspace(0.0, 4.0, 301)])
    o.integrate(ts_pe, pot, method="dop853_c")
    # If continuation had wrongly fired, self.t would be merged from old + new.
    # The per-orbit-t early-out in _should_continue_integration must restart
    # the integration from the original initial conditions instead.
    o_ref = Orbit(vxvvs)
    o_ref.integrate(ts_pe, pot, method="dop853_c")
    assert o.t.shape == o_ref.t.shape
    assert numpy.allclose(o.t, o_ref.t)
    assert numpy.allclose(o.orbit, o_ref.orbit, atol=1e-12, rtol=1e-12)

    # 2) Per-orbit-t integrate first, then 1D-t re-integrate.
    o2 = Orbit(vxvvs)
    o2.integrate(ts_pe, pot, method="dop853_c")
    assert numpy.asarray(o2.t).ndim == 2
    ts_shared = numpy.linspace(0.0, 4.0, 401)
    o2.integrate(ts_shared, pot, method="dop853_c")
    # Same expectation: must not continue from the prior per-orbit run.
    o2_ref = Orbit(vxvvs)
    o2_ref.integrate(ts_shared, pot, method="dop853_c")
    assert numpy.asarray(o2.t).ndim == 1
    assert numpy.allclose(o2.t, o2_ref.t)
    assert numpy.allclose(o2.orbit, o2_ref.orbit, atol=1e-12, rtol=1e-12)
    return None


# Test that integrating after bruteSOS does NOT continue from the previous
# integration (bruteSOS rewrites self.t into a NaN-padded crossings grid and
# sets _cannot_continue_integration; a subsequent integrate() must therefore
# go through the from-scratch path in _should_continue_integration).
def test_integrate_after_bruteSOS_does_not_continue():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    ic = [1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi]
    o = Orbit(ic)
    o.bruteSOS(numpy.linspace(0.0, 100.0, 5001), MWPotential2014, method="dop853_c")
    # After bruteSOS: self.t is 2D NaN-padded and _cannot_continue_integration
    # is True. Re-integrating must restart from the original initial
    # conditions, not continue from any previous state.
    assert getattr(o, "_cannot_continue_integration", False), (
        "bruteSOS should set _cannot_continue_integration"
    )
    new_ts = numpy.linspace(0.0, 10.0, 1001)
    o.integrate(new_ts, MWPotential2014, method="dop853_c")
    # Reference: a fresh Orbit integrated on the same grid — bit-for-bit
    # identical if the post-bruteSOS integrate restarted from scratch.
    o_ref = Orbit(ic)
    o_ref.integrate(new_ts, MWPotential2014, method="dop853_c")
    assert numpy.allclose(o.t, new_ts) and o.t.ndim == 1, (
        "self.t after re-integrate should be the new 1D time grid"
    )
    assert o.orbit.shape == o_ref.orbit.shape, (
        "post-bruteSOS integrate should produce the same orbit shape as a fresh one"
    )
    assert numpy.allclose(o.orbit, o_ref.orbit, atol=1e-12, rtol=1e-12), (
        "post-bruteSOS integrate must restart from the original ICs (no continuation)"
    )
    # And every quantity-method query at the new grid agrees with the fresh run.
    for method_name in ("R", "vR", "vT", "z", "vz", "phi", "x", "y"):
        a = getattr(o, method_name)(new_ts)
        b = getattr(o_ref, method_name)(new_ts)
        assert numpy.allclose(a, b, atol=1e-10, rtol=1e-10), (
            f"o.{method_name}(new_ts) disagrees with fresh-orbit reference"
        )
    return None


# Test that the internal interpolator is reset when the orbit is re-integrated
# with bruteSOS
def test_orbinterp_reset_bruteSOS():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.1, -0.03, numpy.pi])
    op = o()
    ts = numpy.linspace(0.0, 100.0, 10001)
    ts2 = numpy.linspace(0.0, 99.0, 10001)
    o.integrate(ts, MWPotential)
    o.R(numpy.linspace(0.0, o.t[-1], 1001))
    o.bruteSOS(ts2, MWPotential2014)
    op.bruteSOS(ts2, MWPotential2014)
    ts = o.t
    # If things are reset correctly, o and op should now agree on everything
    assert numpy.all(numpy.fabs(o.R(ts) - op.R(ts)) < 10.0**-10.0), (
        "Orbit R not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vR(ts) - op.vR(ts)) < 10.0**-10.0), (
        "Orbit vR not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vT(ts) - op.vT(ts)) < 10.0**-10.0), (
        "Orbit vT not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.z(ts) - op.z(ts)) < 10.0**-10.0), (
        "Orbit z not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vz(ts) - op.vz(ts)) < 10.0**-10.0), (
        "Orbit vz not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.phi(ts) - op.phi(ts)) < 10.0**-10.0), (
        "Orbit phi not reset correctly"
    )
    assert numpy.fabs(o.rperi() - op.rperi()) < 10.0**-10.0, (
        "Orbit rperi not reset correctly"
    )
    assert numpy.fabs(o.rap() - op.rap()) < 10.0**-10.0, "Orbit rap not reset correctly"
    assert numpy.fabs(o.e() - op.e()) < 10.0**-10.0, "Orbit e not reset correctly"
    return None


# Test that the internal interpolator is reset when the orbit is re-integrated
# with integratedxdv
def test_orbinterp_reset_integratedxdv():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, numpy.pi])
    op = o()
    ts = numpy.linspace(0.0, 100.0, 10001)
    o.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, MWPotential)
    o.R(numpy.linspace(0.0, o.t[-1], 1001))
    o.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, MWPotential2014)
    op.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, MWPotential2014)
    # If things are reset correctly, o and op should now agree on everything
    assert numpy.all(numpy.fabs(o.R(ts) - op.R(ts)) < 10.0**-10.0), (
        "Orbit R not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vR(ts) - op.vR(ts)) < 10.0**-10.0), (
        "Orbit vR not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.vT(ts) - op.vT(ts)) < 10.0**-10.0), (
        "Orbit vT not reset correctly"
    )
    assert numpy.all(numpy.fabs(o.phi(ts) - op.phi(ts)) < 10.0**-10.0), (
        "Orbit phi not reset correctly"
    )
    assert numpy.fabs(o.rperi() - op.rperi()) < 10.0**-10.0, (
        "Orbit rperi not reset correctly"
    )
    assert numpy.fabs(o.rap() - op.rap()) < 10.0**-10.0, "Orbit rap not reset correctly"
    assert numpy.fabs(o.e() - op.e()) < 10.0**-10.0, "Orbit e not reset correctly"
    return None


# Test that an error is raised when integration time array is not equally spaced (see #700)
def test_integrate_notevenlyspaced_issue700():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    times = numpy.concatenate(
        [numpy.linspace(0, 10, 21), numpy.linspace(12.0, 50.0, 20)]
    )
    orb = Orbit()
    # Test that the correct error is raised when the time array is not equally spaced
    with pytest.raises(ValueError) as excinfo:
        orb.integrate(times, MWPotential2014, method="symplec6_c")
    assert (
        str(excinfo.value)
        == "Input time array must be equally spaced for method symplec6_c, use method='dop853_c', method='dop853', or method='odeint' instead for non-equispaced time arrays"
    ), "Input time array must be equally spaced error not raised"
    # Also test backwards integration
    with pytest.raises(ValueError) as excinfo:
        orb.integrate(-times, MWPotential2014, method="symplec6_c")
    assert (
        str(excinfo.value)
        == "Input time array must be equally spaced for method symplec6_c, use method='dop853_c', method='dop853', or method='odeint' instead for non-equispaced time arrays"
    ), "Input time array must be equally spaced error not raised"
    # Also test integrate_dxdv
    with pytest.raises(ValueError) as excinfo:
        orb.toPlanar().integrate_dxdv(None, times, MWPotential2014, method="dopr54_c")
    assert (
        str(excinfo.value)
        == "Input time array must be equally spaced for method dopr54_c, use method='dop853_c', method='dop853', or method='odeint' instead for non-equispaced time arrays"
    ), "Input time array must be equally spaced error not raised"
    # Also test integrateSOS, just use times for psi...
    with pytest.raises(ValueError) as excinfo:
        orb.integrate_SOS(times, MWPotential2014, method="dopr54_c")
    assert (
        str(excinfo.value)
        == "Input psi array must be equally spaced for method dopr54_c, use method='dop853_c', method='dop853', or method='odeint' instead for non-equispaced psi arrays"
    ), "Input time array must be equally spaced error not raised"
    return None


# Test that integrators that should be fine with unevenly-spaced times are fine with it
def test_integrate_notevenlyspaced_ok():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    time_1 = numpy.linspace(0, 50.0, 1001)
    time_2 = numpy.concatenate(
        [numpy.linspace(0, 20.0, 1001), numpy.linspace(20.02, 50.0, 1001)]
    )
    for integrator in ["odeint", "dop853", "dop853_c"]:
        o_1 = Orbit()
        o_1.integrate(time_1, MWPotential2014, method=integrator)
        o_2 = Orbit()
        o_2.integrate(time_2, MWPotential2014, method=integrator)
        assert numpy.all(numpy.fabs(o_1.R(time_1) - o_2.R(time_1)) < 10.0**-5.0), (
            "Integration with unevenly-spaced times does not work"
        )
        assert numpy.all(numpy.fabs(o_1.vR(time_1) - o_2.vR(time_1)) < 10.0**-4.0), (
            "Integration with unevenly-spaced times does not work"
        )
        assert numpy.all(numpy.fabs(o_1.vT(time_1) - o_2.vT(time_1)) < 10.0**-4.0), (
            "Integration with unevenly-spaced times does not work"
        )
    for integrator in [
        "leapfrog",
        "leapfrog_c",
        "symplec4_c",
        "symplec6_c",
        "rk4_c",
        "rk6_c",
        "dopr54_c",
        "ias15_c",
    ]:
        o_1 = Orbit()
        o_1.integrate(time_1, MWPotential2014, method=integrator)
        o_2 = Orbit()
        with pytest.raises(ValueError) as excinfo:
            o_2.integrate(time_2, MWPotential2014, method=integrator)
        assert (
            str(excinfo.value)
            == f"Input time array must be equally spaced for method {integrator}, use method='dop853_c', method='dop853', or method='odeint' instead for non-equispaced time arrays"
        ), f"Input time array must be equally spaced for method{integrator}"
    return None


def test_linear_plotting():
    from galpy.orbit import Orbit
    from galpy.potential.verticalPotential import RZToverticalPotential

    o = Orbit([1.0, 1.0])
    times = numpy.linspace(0.0, 7.0, 251)
    from galpy.potential import LogarithmicHaloPotential

    lp = RZToverticalPotential(LogarithmicHaloPotential(normalize=1.0, q=0.8), 1.0)
    try:
        o.plotE()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    # Integrate
    o.integrate(times, lp)
    # Energy
    o.plotE()
    o.plotE(pot=lp, d1="x", xlabel=r"$xlabel$")
    o.plotE(pot=lp, d1="vx", ylabel=r"$ylabel$")
    # Plot the orbit itself, defaults
    o.plot()
    o.plot(ro=8.0)
    # Plot the orbit itself in 3D
    try:
        o.plot3d()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.plot3d for linearOrbit did not raise Exception")
    return None


# Check plotting routines
def test_planar_plotting():
    from galpy.orbit import Orbit
    from galpy.potential.planarPotential import RZToplanarPotential

    o = Orbit([1.0, 0.1, 1.1, 2.0])
    oa = Orbit([1.0, 0.1, 1.1])
    times = numpy.linspace(0.0, 7.0, 251)
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    try:
        o.plotE()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    try:
        o.plotJacobi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    try:
        oa.plotE()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotE() before the orbit was integrated did not raise AttributeError for planarROrbit"
        )
    try:
        oa.plotJacobi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarROrbit"
        )
    # Integrate
    o.integrate(times, lp)
    oa.integrate(times, lp)
    # Energy
    o.plotE()
    o.plotE(pot=lp, d1="R")
    o.plotE(pot=lp, d1="vR")
    o.plotE(pot=lp, d1="phi")
    o.plotE(pot=lp + RZToplanarPotential(lp), d1="vT")
    oa.plotE()
    oa.plotE(pot=lp, d1="R")
    oa.plotE(pot=lp, d1="vR")
    oa.plotE(pot=lp + RZToplanarPotential(lp), d1="vT")
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    o.plotJacobi(pot=lp, d1="vR")
    o.plotJacobi(pot=lp, d1="phi")
    o.plotJacobi(pot=lp + RZToplanarPotential(lp), d1="vT")
    oa.plotJacobi()
    oa.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    oa.plotJacobi(pot=lp, d1="vR")
    oa.plotJacobi(pot=lp + RZToplanarPotential(lp), d1="vT")
    # Plot the orbit itself, defaults
    o.plot()
    o.plot(ro=8.0)
    oa.plot()
    o.plotx(d1="vx")
    o.plotvx(d1="y")
    o.ploty(d1="vy")
    o.plotvy(d1="x")
    # Plot the orbit itself in 3D, defaults
    o.plot3d()
    o.plot3d(ro=8.0)
    oa.plot3d()
    o.plot3d(d1="x", d2="vx", d3="y")
    o.plot3d(d1="vx", d2="y", d3="vy")
    o.plot3d(d1="y", d2="vy", d3="x")
    o.plot3d(d1="vy", d2="x", d3="vx")
    return None


# Check plotting routines
def test_full_plotting():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0])
    oa = Orbit([1.0, 0.1, 1.1, 0.1, 0.2])
    times = numpy.linspace(0.0, 7.0, 251)
    from galpy.potential import LogarithmicHaloPotential

    if True:  # not _GHACTIONS:
        from galpy.potential import DoubleExponentialDiskPotential

        dp = DoubleExponentialDiskPotential(normalize=1.0)
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    try:
        o.plotE()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    try:
        o.plotEz()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotEz() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    try:
        o.plotJacobi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarOrbit"
        )
    try:
        oa.plotE()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotE() before the orbit was integrated did not raise AttributeError for planarROrbit"
        )
    try:
        oa.plotEz()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotEz() before the orbit was integrated did not raise AttributeError for planarROrbit"
        )
    try:
        oa.plotJacobi()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarROrbit"
        )
    # Integrate
    o.integrate(times, lp)
    oa.integrate(times, lp)
    # Energy
    o.plotE()
    o.plotE(normed=True)
    o.plotE(pot=lp, d1="R")
    o.plotE(pot=lp, d1="vR")
    o.plotE(pot=lp, d1="vT")
    o.plotE(pot=lp, d1="z")
    o.plotE(pot=lp, d1="vz")
    o.plotE(pot=lp, d1="phi")
    if True:  # not _GHACTIONS:
        o.plotE(pot=dp, d1="phi")
    oa.plotE()
    oa.plotE(pot=lp, d1="R")
    oa.plotE(pot=lp, d1="vR")
    oa.plotE(pot=lp, d1="vT")
    oa.plotE(pot=lp, d1="z")
    oa.plotE(pot=lp, d1="vz")
    # Vertical energy
    o.plotEz()
    o.plotEz(normed=True)
    o.plotEz(pot=lp, d1="R")
    o.plotEz(pot=lp, d1="vR")
    o.plotEz(pot=lp, d1="vT")
    o.plotEz(pot=lp, d1="z")
    o.plotEz(pot=lp, d1="vz")
    o.plotEz(pot=lp, d1="phi")
    if True:  # not _GHACTIONS:
        o.plotEz(pot=dp, d1="phi")
    oa.plotEz()
    oa.plotEz(normed=True)
    oa.plotEz(pot=lp, d1="R")
    oa.plotEz(pot=lp, d1="vR")
    oa.plotEz(pot=lp, d1="vT")
    oa.plotEz(pot=lp, d1="z")
    oa.plotEz(pot=lp, d1="vz")
    # Radial energy
    o.plotER()
    o.plotER(normed=True)
    # Radial energy
    oa.plotER()
    oa.plotER(normed=True)
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(normed=True)
    o.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    o.plotJacobi(pot=lp, d1="vR")
    o.plotJacobi(pot=lp, d1="vT")
    o.plotJacobi(pot=lp, d1="z")
    o.plotJacobi(pot=lp, d1="vz")
    o.plotJacobi(pot=lp, d1="phi")
    oa.plotJacobi()
    oa.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    oa.plotJacobi(pot=lp, d1="vR")
    oa.plotJacobi(pot=lp, d1="vT")
    oa.plotJacobi(pot=lp, d1="z")
    oa.plotJacobi(pot=lp, d1="vz")
    # Plot the orbit itself
    o.plot()  # defaults
    oa.plot()
    o.plot(d1="vR", label="vR")
    o.plot(d2="vR", label=["vR"])
    o.plotR()
    o.plotvR(d1="vT")
    o.plotvT(d1="z")
    o.plotz(d1="vz")
    o.plotvz(d1="phi")
    o.plotphi(d1="vR")
    o.plotx(d1="vx")
    o.plotvx(d1="y")
    o.ploty(d1="vy")
    o.plotvy(d1="x")
    # Remaining attributes
    o.plot(d1="ra", d2="dec")
    o.plot(d2="ra", d1="dec")
    o.plot(d1="pmra", d2="pmdec")
    o.plot(d2="pmra", d1="pmdec")
    o.plot(d1="ll", d2="bb")
    o.plot(d2="ll", d1="bb")
    o.plot(d1="pmll", d2="pmbb")
    o.plot(d2="pmll", d1="pmbb")
    o.plot(d1="vlos", d2="dist")
    o.plot(d2="vlos", d1="dist")
    o.plot(d1="helioX", d2="U")
    o.plot(d2="helioX", d1="U")
    o.plot(d1="helioY", d2="V")
    o.plot(d2="helioY", d1="V")
    o.plot(d1="helioZ", d2="W")
    o.plot(d2="helioZ", d1="W")
    o.plot(d2="r", d1="R")
    o.plot(d2="R", d1="r")
    # Some more energies etc.
    o.plot(d1="E", d2="R")
    o.plot(d1="Enorm", d2="R")
    o.plot(d1="Ez", d2="R")
    o.plot(d1="Eznorm", d2="R")
    o.plot(d1="ER", d2="R")
    o.plot(d1="ERnorm", d2="R")
    o.plot(d1="Jacobi", d2="R")
    o.plot(d1="Jacobinorm", d2="R")
    # callables
    o.plot(d1=lambda t: t, d2=lambda t: o.R(t))
    # Expressions
    o.plot(d1="t", d2="r*R/vR")
    o.plot(
        d1=f"R*cos(phi-{o.Op(quantity=False) - o.Or(quantity=False) / 2:f}*t)",
        d2=f"R*sin(phi-{o.Op(quantity=False) - o.Or(quantity=False) / 2:f}*t)",
    )
    with pytest.raises(TypeError) as excinfo:
        # Unparsable expression gives TypeError
        o.plot(d1="t", d2="r@2")
    # Test AttributeErrors
    try:
        oa.plotx()
    except AttributeError:
        pass
    else:
        raise AssertionError("plotx() applied to RZOrbit did not raise AttributeError")
    try:
        oa.plotvx()
    except AttributeError:
        pass
    else:
        raise AssertionError("plotvx() applied to RZOrbit did not raise AttributeError")
    try:
        oa.ploty()
    except AttributeError:
        pass
    else:
        raise AssertionError("ploty() applied to RZOrbit did not raise AttributeError")
    try:
        oa.plotvy()
    except AttributeError:
        pass
    else:
        raise AssertionError("plotvy() applied to RZOrbit did not raise AttributeError")
    try:
        oa.plot(d1="x")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot(d1='x') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot(d1="vx")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot(d1='vx') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot(d1="y")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot(d1='y') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot(d1="vy")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot(d1='vy') applied to RZOrbit did not raise AttributeError"
        )
    # Plot the orbit itself in 3D
    o.plot3d()  # defaults
    oa.plot3d()
    o.plot3d(d1="t", d2="z", d3="R")
    o.plot3d(d1="r", d2="t", d3="phi")
    o.plot3d(d1="vT", d2="vR", d3="t")
    o.plot3d(d1="z", d2="vT", d3="vz")
    o.plot3d(d1="vz", d2="z", d3="phi")
    o.plot3d(d1="phi", d2="vz", d3="R")
    o.plot3d(d1="vR", d2="phi", d3="vR")
    o.plot3d(d1="vx", d2="x", d3="y")
    o.plot3d(d1="y", d2="vx", d3="vy")
    o.plot3d(d1="vy", d2="y", d3="x")
    o.plot3d(d1="x", d2="vy", d3="vx")
    o.plot3d(d1="x", d2="r", d3="vx")
    o.plot3d(d1="x", d2="vy", d3="r")
    # Remaining attributes
    o.plot3d(d1="ra", d2="dec", d3="pmra")
    o.plot3d(d2="ra", d1="dec", d3="pmdec")
    o.plot3d(d1="pmra", d2="pmdec", d3="ra")
    o.plot3d(d2="pmra", d1="pmdec", d3="dec")
    o.plot3d(d1="ll", d2="bb", d3="pmll")
    o.plot3d(d2="ll", d1="bb", d3="pmbb")
    o.plot3d(d1="pmll", d2="pmbb", d3="ll")
    o.plot3d(d2="pmll", d1="pmbb", d3="bb")
    o.plot3d(d1="vlos", d2="dist", d3="vlos")
    o.plot3d(d2="vlos", d1="dist", d3="dist")
    o.plot3d(d1="helioX", d2="U", d3="V")
    o.plot3d(d2="helioX", d1="U", d3="helioY")
    o.plot3d(d1="helioY", d2="V", d3="W")
    o.plot3d(d2="helioY", d1="V", d3="helioZ")
    o.plot3d(d1="helioZ", d2="W", d3="U")
    o.plot3d(d2="helioZ", d1="W", d3="helioX")
    # callables don't work
    o.plot3d(d1=lambda t: t, d2=lambda t: o.R(t), d3=lambda t: o.z(t))
    # Test AttributeErrors
    try:
        o.plot3d(d1="R")  # shouldn't work, bc there is no default
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d with just d1= set should have raised AttributeError, but did not"
        )
    try:
        oa.plot3d(d2="x", d1="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d2='x') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d2="vx", d1="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d2='vx') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d2="y", d1="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d2='y') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot(d2="vy", d1="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d2='vy') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d1="x", d2="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d1='x') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d1="vx", d2="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d1='vx') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d1="y", d2="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d1='y') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d1="vy", d2="R", d3="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d1='vy') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d3="x", d2="R", d1="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d3='x') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d3="vx", d2="R", d1="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d3='vx') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d3="y", d2="R", d1="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d3='y') applied to RZOrbit did not raise AttributeError"
        )
    try:
        oa.plot3d(d3="vy", d2="R", d1="t")
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "plot3d(d3='vy') applied to RZOrbit did not raise AttributeError"
        )
    return None


def test_plotSOS():
    # 3D
    pot = potential.MWPotential2014
    o = setup_orbit_energy(pot)
    o.plotSOS(pot)
    o.plotSOS(pot, use_physical=True)
    # 2D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    o.plotSOS(pot, label="test")
    o.plotSOS(pot, use_physical=True, label=["test"])
    o.plotSOS(pot, surface="y")
    o.plotSOS(pot, surface="y", use_physical=True)
    return None


def test_plotBruteSOS():
    # 3D
    pot = potential.MWPotential2014
    o = setup_orbit_energy(pot)
    o.plotBruteSOS(numpy.linspace(0.0, 100.0, 100001), pot)
    o.plotBruteSOS(numpy.linspace(0.0, 100.0, 100001), pot, use_physical=True)
    # 2D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    o = setup_orbit_energy(pot)
    o.plotBruteSOS(numpy.linspace(0.0, 100.0, 100001), pot, label="test")
    o.plotBruteSOS(
        numpy.linspace(0.0, 100.0, 100001), pot, use_physical=True, label=["test"]
    )
    o.plotBruteSOS(numpy.linspace(0.0, 100.0, 100001), pot, surface="y")
    o.plotBruteSOS(
        numpy.linspace(0.0, 100.0, 100001), pot, surface="y", use_physical=True
    )
    return None


def test_from_name_values():
    from galpy.orbit import Orbit

    # test Vega
    o = Orbit.from_name("Vega")
    assert numpy.isclose(o.ra(), 279.23473479), "RA of Vega does not match SIMBAD value"
    assert numpy.isclose(o.dec(), 38.78368896), (
        "DEC of Vega does not match SIMBAD value"
    )
    assert numpy.isclose(o.dist(), 1 / 130.23), (
        "Parallax of Vega does not match SIMBAD value"
    )
    assert numpy.isclose(o.pmra(), 200.94), "PMRA of Vega does not match SIMBAD value"
    assert numpy.isclose(o.pmdec(), 286.23), "PMDec of Vega does not match SIMBAD value"
    assert numpy.isclose(o.vlos(), -13.50), (
        "radial velocity of Vega does not match SIMBAD value"
    )

    # test Lacaille 8760
    o = Orbit.from_name("Lacaille 8760")
    assert numpy.isclose(o.ra(), 319.31362024), (
        "RA of Lacaille 8760 does not match SIMBAD value"
    )
    assert numpy.isclose(o.dec(), -38.86736390), (
        "DEC of Lacaille 8760 does not match SIMBAD value"
    )
    assert numpy.isclose(o.dist(), 1 / 251.9124), (
        "Parallax of Lacaille 8760 does not match SIMBAD value"
    )
    assert numpy.isclose(o.pmra(), -3258.996), (
        "PMRA of Lacaille 8760 does not match SIMBAD value"
    )
    assert numpy.isclose(o.pmdec(), -1145.862), (
        "PMDec of Lacaille 8760 does not match SIMBAD value"
    )
    assert numpy.isclose(o.vlos(), 20.56), (
        "radial velocity of Lacaille 8760 does not match SIMBAD value"
    )

    # test LMC
    o = Orbit.from_name("LMC")
    assert numpy.isclose(o.ra(), 78.77), "RA of LMC does not match value on file"
    assert numpy.isclose(o.dec(), -69.01), "DEC of LMC does not match value on file"
    # Remove distance for now, because SIMBAD has the wrong distance (100 Mpc)
    assert numpy.isclose(o.dist(), 50.1), "Parallax of LMC does not match value on file"
    assert numpy.isclose(o.pmra(), 1.850), "PMRA of LMC does not match value on file"
    assert numpy.isclose(o.pmdec(), 0.234), "PMDec of LMC does not match value on file"
    assert numpy.isclose(o.vlos(), 262.2), (
        "radial velocity of LMC does not match value on file"
    )

    # test a distant hypervelocity star
    o = Orbit.from_name("[BGK2006] HV 5")
    assert numpy.isclose(o.ra(), 139.498), (
        "RA of [BGK2006] HV 5 does not match value on file"
    )
    assert numpy.isclose(o.dec(), 67.377), (
        "DEC of [BGK2006] HV 5 does not match SIMBAD value"
    )
    assert numpy.isclose(o.dist(), 55.0), (
        "Parallax of [BGK2006] HV 5 does not match SIMBAD value"
    )
    assert numpy.isclose(o.pmra(), 0.001), (
        "PMRA of [BGK2006] HV 5 does not match SIMBAD value"
    )
    assert numpy.isclose(o.pmdec(), -0.989), (
        "PMDec of [BGK2006] HV 5 does not match SIMBAD value"
    )
    assert numpy.isclose(o.vlos(), 553.0), (
        "radial velocity of [BGK2006] HV 5 does not match SIMBAD value"
    )

    # Sagittarius dwarf, which has its distance in Mpc in SIMBAD
    o = Orbit.from_name("SDG")
    assert numpy.isclose(o.dist(), 20.0), "Distance of SDG does not match value on file"

    return None


def test_from_name_errors():
    from galpy.orbit import Orbit

    # test GJ 440
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name("GJ 440")
    msg = "failed to find some coordinates for GJ 440 in SIMBAD"
    assert str(excinfo.value) == msg, (
        f"expected message '{msg}' but got '{str(excinfo.value)}' instead"
    )

    # test with a fake object
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name("abc123")
    msg = "failed to find abc123 in SIMBAD"
    assert str(excinfo.value) == msg, (
        f"expected message '{msg}' but got '{str(excinfo.value)}' instead"
    )

    # test GRB 090423
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name("GRB 090423")
    msg = "failed to find some coordinates for GRB 090423 in SIMBAD"
    assert str(excinfo.value) == msg, (
        f"expected message '{msg}' but got '{str(excinfo.value)}' instead"
    )


def test_from_name_named():
    # Test that the values from the JSON file are correctly transferred
    import json

    # Read the JSON file
    import os

    import galpy.orbit
    from galpy.orbit import Orbit

    named_objects_file = os.path.join(
        os.path.dirname(os.path.realpath(galpy.orbit.__file__)), "named_objects.json"
    )
    with open(named_objects_file) as json_file:
        named_data = json.load(json_file)
    del named_data["_collections"]
    del named_data["_synonyms"]
    for obj in named_data:
        o = Orbit.from_name(obj)
        for attr in named_data[obj]:
            if "source" in attr or "dr2" in attr:
                continue
            # Skip entries with missing vlos for now
            if numpy.isnan(named_data[obj]["vlos"]):
                continue
            # Skip errors until we use them
            if attr == "pmcorr" or "_e" in attr:
                continue
            if attr == "ro" or attr == "vo" or attr == "zo" or attr == "solarmotion":
                assert numpy.all(
                    numpy.isclose(getattr(o, f"_{attr:s}"), named_data[obj][attr])
                )
            elif attr == "distance":
                assert numpy.isclose(o.dist(), named_data[obj][attr])
            else:
                assert numpy.isclose(getattr(o, f"{attr:s}")(), named_data[obj][attr])
    return None


def test_from_name_collections():
    # Test that the values from the JSON file are correctly transferred,
    # for collections of objects
    import json

    # Read the JSON file
    import os

    import galpy.orbit
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import _known_objects_collections_original_keys

    named_objects_file = os.path.join(
        os.path.dirname(os.path.realpath(galpy.orbit.__file__)), "named_objects.json"
    )
    with open(named_objects_file) as json_file:
        named_data = json.load(json_file)
    for obj in _known_objects_collections_original_keys:
        o = Orbit.from_name(obj)
        for ii, individual_obj in enumerate(named_data["_collections"][obj]):
            for attr in named_data[individual_obj]:
                if "source" in attr or "dr2 in attr":
                    continue
                if (
                    attr == "ro"
                    or attr == "vo"
                    or attr == "zo"
                    or attr == "solarmotion"
                ):
                    continue  # don't test these here
                elif attr == "distance":
                    assert numpy.isclose(o.dist()[ii], named_data[individual_obj][attr])
                else:
                    assert numpy.isclose(
                        getattr(o, f"{attr:s}")()[ii], named_data[individual_obj][attr]
                    )
    return None


def test_from_name_solarsystem():
    # Test that the solar system matches Bovy et al. (2010)'s input data
    from astropy import units

    from galpy.orbit import Orbit

    correct_xyz = numpy.array(
        [
            [
                0.324190175,
                0.090955208,
                -0.022920510,
                -4.627851589,
                10.390063716,
                1.273504997,
            ],
            [
                -0.701534590,
                -0.168809218,
                0.037947785,
                1.725066954,
                -7.205747212,
                -0.198268558,
            ],
            [
                -0.982564148,
                -0.191145980,
                -0.000014724,
                1.126784520,
                -6.187988860,
                0.000330572,
            ],
            [
                1.104185888,
                -0.826097003,
                -0.044595990,
                3.260215854,
                4.524583075,
                0.014760239,
            ],
            [
                3.266443877,
                -3.888055863,
                -0.057015321,
                2.076140727,
                1.904040630,
                -0.054374153,
            ],
            [
                -9.218802228,
                1.788299816,
                0.335737817,
                -0.496457364,
                -2.005021061,
                0.054667082,
            ],
            [
                19.930781147,
                -2.555241579,
                -0.267710968,
                0.172224285,
                1.357933443,
                0.002836325,
            ],
            [
                24.323085642,
                -17.606227355,
                -0.197974999,
                0.664855006,
                0.935497207,
                -0.034716967,
            ],
        ]
    )
    os = Orbit.from_name("solar system")
    for ii, o in enumerate(os):
        assert (
            numpy.fabs((o.x() * units.kpc).to(units.AU).value - correct_xyz[ii, 0])
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert (
            numpy.fabs((o.y() * units.kpc).to(units.AU).value - correct_xyz[ii, 1])
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert (
            numpy.fabs((o.z() * units.kpc).to(units.AU).value - correct_xyz[ii, 2])
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert (
            numpy.fabs(
                (o.vx() * units.km / units.s).to(units.AU / units.yr).value
                - correct_xyz[ii, 3]
            )
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert (
            numpy.fabs(
                (o.vy() * units.km / units.s).to(units.AU / units.yr).value
                - correct_xyz[ii, 4]
            )
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert (
            numpy.fabs(
                (o.vz() * units.km / units.s).to(units.AU / units.yr).value
                - correct_xyz[ii, 5]
            )
            < 1e-8
        ), "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
    return None


def test_rguiding_errors():
    from galpy.orbit import Orbit
    from galpy.potential import TriaxialNFWPotential

    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    # No potential raises error
    with pytest.raises(RuntimeError) as excinfo:
        o.rguiding()
    # non-axi potential raises error
    np = TriaxialNFWPotential(amp=20.0, c=0.8, b=0.7)
    with pytest.raises(RuntimeError) as excinfo:
        o.rguiding(pot=np)
    return None


def test_rE_errors():
    from galpy.orbit import Orbit
    from galpy.potential import TriaxialNFWPotential

    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    # No potential raises error
    with pytest.raises(RuntimeError) as excinfo:
        o.rE()
    # non-axi potential raises error
    np = TriaxialNFWPotential(amp=20.0, c=0.8, b=0.7)
    with pytest.raises(RuntimeError) as excinfo:
        o.rE(pot=np)
    return None


def test_LcE_errors():
    from galpy.orbit import Orbit
    from galpy.potential import TriaxialNFWPotential

    R, Lz = 1.0, 1.4
    o = Orbit([R, 0.4, Lz / R, 0.0])
    # No potential raises error
    with pytest.raises(RuntimeError) as excinfo:
        o.LcE()
    # non-axi potential raises error
    np = TriaxialNFWPotential(amp=20.0, c=0.8, b=0.7)
    with pytest.raises(RuntimeError) as excinfo:
        o.LcE(pot=np)
    return None


def test_phi_range():
    # Test that the range returned by Orbit.phi is [-pi,pi],
    # example from Jeremy Webb
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit()
    ts = numpy.linspace(0.0, -30, 5000)
    o.integrate(ts, MWPotential2014)
    assert numpy.all(o.phi(ts) <= numpy.pi), "o.phi does not return values <= pi"
    assert numpy.all(o.phi(ts) >= -numpy.pi), "o.phi does not return values >= pi"
    assert numpy.all(o.phi(ts[::-1]) <= numpy.pi), "o.phi does not return values <= pi"
    assert numpy.all(o.phi(ts[::-1]) >= -numpy.pi), "o.phi does not return values >= pi"
    # Also really interpolated
    its = numpy.linspace(0.0, -30, 5001)
    assert numpy.all(o.phi(its) <= numpy.pi), "o.phi does not return values <= pi"
    assert numpy.all(o.phi(its) >= -numpy.pi), "o.phi does not return values >= pi"
    return None


def test_orbit_time():
    # Test that Orbit.time returns the time correctly, with units when that's
    # required
    from astropy import units as u

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    ts = numpy.linspace(0.0, 1.0, 1001) * u.Gyr
    o = Orbit()
    o.integrate(ts, MWPotential2014)
    # No argument, in this case should be times in Gyr
    assert numpy.all(numpy.fabs((ts - o.time(quantity=True)) / ts[-1]).value < 1e-10), (
        "Orbit.time does not return the correct times"
    )
    # with argument, should be time in Gyr
    assert numpy.all(
        numpy.fabs((ts - o.time(ts, quantity=True)) / ts[-1]).value < 1e-10
    ), "Orbit.time does not return the correct times"
    assert (
        numpy.fabs((ts[-1] - o.time(ts[-1], quantity=True)) / ts[-1]).value < 1e-10
    ), "Orbit.time does not return the correct times"
    # with argument without units --> units
    assert numpy.all(
        numpy.fabs(
            ts
            - o.time(
                ts.to(u.Gyr).value
                / conversion.time_in_Gyr(
                    MWPotential2014[0]._vo, MWPotential2014[0]._ro
                ),
                quantity=True,
            )
        ).value
        < 1e-10
    ), "Orbit.time does not return the correct times"
    assert (
        numpy.fabs(
            ts[-1]
            - o.time(
                ts[-1].to(u.Gyr).value
                / conversion.time_in_Gyr(
                    MWPotential2014[0]._vo, MWPotential2014[0]._ro
                ),
                quantity=True,
            )
        ).value
        < 1e-10
    ), "Orbit.time does not return the correct times"
    # Now should get without units
    o.turn_physical_off()
    assert numpy.all(
        numpy.fabs(
            ts.to(u.Gyr).value
            / conversion.time_in_Gyr(MWPotential2014[0]._vo, MWPotential2014[0]._ro)
            - o.time(
                ts.to(u.Gyr).value
                / conversion.time_in_Gyr(MWPotential2014[0]._vo, MWPotential2014[0]._ro)
            )
        )
        < 1e-10
    ), "Orbit.time does not return the correct times"
    assert (
        numpy.fabs(
            ts[-1].to(u.Gyr).value
            / conversion.time_in_Gyr(MWPotential2014[0]._vo, MWPotential2014[0]._ro)
            - o.time(
                ts[-1].to(u.Gyr).value
                / conversion.time_in_Gyr(MWPotential2014[0]._vo, MWPotential2014[0]._ro)
            )
        )
        < 1e-10
    ), "Orbit.time does not return the correct times"
    return None


# Test that issue 402 is resolved: initialization with a SkyCoord when radec=True should work fine
def test_SkyCoord_init_with_radecisTrue():
    if not _APY3:
        return None  # not done in python 2
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from galpy.orbit import Orbit

    # Example is for NGC5466 from @jjensen4571
    rv_ngc5466 = 106.93
    pmra_ngc5466 = -5.41
    pmdec_ngc5466 = -0.79
    ra_ngc5466 = 211.363708
    dec_ngc5466 = 28.534445
    rhel_ngc5466 = 16.0
    mean_ngc5466 = SkyCoord(
        ra=ra_ngc5466 * u.deg,
        dec=dec_ngc5466 * u.deg,
        distance=rhel_ngc5466 * u.kpc,
        pm_ra_cosdec=pmra_ngc5466 * u.mas / u.yr,
        pm_dec=pmdec_ngc5466 * u.mas / u.yr,
        radial_velocity=rv_ngc5466 * u.km / u.s,
    )
    o_sky = Orbit(mean_ngc5466, radec=True, ro=8.1, vo=229.0, solarmotion="schoenrich")
    o_radec = Orbit(
        [
            ra_ngc5466,
            dec_ngc5466,
            rhel_ngc5466,
            pmra_ngc5466,
            pmdec_ngc5466,
            rv_ngc5466,
        ],
        radec=True,
        ro=8.1,
        vo=229.0,
        solarmotion="schoenrich",
    )
    assert numpy.fabs(o_sky.ra() - o_radec.ra()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    assert numpy.fabs(o_sky.dec() - o_radec.dec()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    assert numpy.fabs(o_sky.dist() - o_radec.dist()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    assert numpy.fabs(o_sky.pmra() - o_radec.pmra()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    assert numpy.fabs(o_sky.pmdec() - o_radec.pmdec()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    assert numpy.fabs(o_sky.vlos() - o_radec.vlos()) < 1e-8, (
        "Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec"
    )
    # Let's also test lb=True for good measure
    o_sky = Orbit(mean_ngc5466, lb=True, ro=8.1, vo=229.0, solarmotion="schoenrich")
    assert numpy.fabs(o_sky.ra() - o_radec.ra()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    assert numpy.fabs(o_sky.dec() - o_radec.dec()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    assert numpy.fabs(o_sky.dist() - o_radec.dist()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    assert numpy.fabs(o_sky.pmra() - o_radec.pmra()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    assert numpy.fabs(o_sky.pmdec() - o_radec.pmdec()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    assert numpy.fabs(o_sky.vlos() - o_radec.vlos()) < 1e-8, (
        "Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb"
    )
    return None


# Test related to issue #415: calling an Orbit with a single int time does not
# work properly
# Test from @jamesmlane
def test_orbit_call_single_time_as_int():
    from galpy import orbit, potential

    pot = potential.MWPotential2014
    o = orbit.Orbit()
    times = numpy.array([0, 1, 2])
    o.integrate(times, pot)
    # Make sure this does not raise TypeErrpr
    try:
        o.x(times[0])
    except TypeError:
        raise
    # Test that the value makes sense
    assert numpy.fabs(o.x(times[0]) - o.x()) < 1e-10
    return None


# Test related to issue #415: calling an Orbit with a single Quantity time
# does not work properly
# Test from @jamesmlane
def test_orbit_call_single_time_as_Quantity():
    from astropy import units as u

    from galpy import orbit, potential

    pot = potential.MWPotential2014
    o = orbit.Orbit()
    times = numpy.array([0, 1, 2]) * u.Gyr
    o.integrate(times, pot)
    # Make sure this does not raise TypeErrpr
    try:
        o.x(times[0])
    except TypeError:
        raise
    # Test that the value makes sense
    assert numpy.fabs(o.x(times[0]) - o.x()) < 1e-10
    return None


# Setup the orbit for the energy test
def setup_orbit_energy(tp, axi=False, henon=False):
    # Need to treat Henon sep. here, bc cannot be scaled to be reasonable
    from galpy.orbit import Orbit

    if isinstance(tp, potential.linearPotential):
        o = Orbit([1.0, 1.0])
    elif isinstance(tp, potential.planarPotential):
        if henon:
            if axi:
                o = Orbit(
                    [
                        0.1,
                        0.3,
                        0.0,
                    ]
                )
            else:
                o = Orbit([0.1, 0.3, 0.0, numpy.pi])
        else:
            if axi:
                o = Orbit([1.0, 1.1, 1.1])
            else:
                o = Orbit([1.0, 1.1, 1.1, numpy.pi / 2.0])
    else:
        if axi:
            o = Orbit([1.0, 1.1, 1.1, 0.1, 0.1])
        else:
            o = Orbit([1.0, 1.1, 1.1, 0.1, 0.1, 0.0])
    return o


# Setup the orbit for the Liouville test
def setup_orbit_liouville(tp, axi=False, henon=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.linearPotential):
        o = Orbit([1.0, 1.0])
    elif isinstance(tp, potential.planarPotential):
        if henon:
            if axi:
                o = Orbit(
                    [
                        0.1,
                        0.3,
                        0.0,
                    ]
                )
            else:
                o = Orbit([0.1, 0.3, 0.0, numpy.pi])
        else:
            if axi:
                o = Orbit([1.0, 0.1, 1.1])
            else:
                o = Orbit([1.0, 0.1, 1.1, 0.0])
    else:
        if axi:
            o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1])
        else:
            o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.0])
    return o


# Setup the orbit for the eccentricity test
def setup_orbit_eccentricity(tp, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 0.0, 1.0])
        else:
            o = Orbit([1.0, 0.0, 1.0, 0.0])
    else:
        if axi:
            o = Orbit([1.0, 0.0, 1.0, 0.0, 0.0])
        else:
            o = Orbit([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return o


# Setup the orbit for the pericenter test
def setup_orbit_pericenter(tp, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 0.0, 1.1])
        else:
            o = Orbit([1.0, 0.0, 1.1, 0.0])
    else:
        if axi:
            o = Orbit([1.0, 0.0, 1.1, 0.0, 0.0])
        else:
            o = Orbit([1.0, 0.0, 1.1, 0.0, 0.0, 0.0])
    return o


# Setup the orbit for the apocenter test
def setup_orbit_apocenter(tp, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 0.0, 0.9])
        else:
            o = Orbit([1.0, 0.0, 0.9, 0.0])
    else:
        if axi:
            o = Orbit([1.0, 0.0, 0.9, 0.0, 0.0])
        else:
            o = Orbit([1.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    return o


# Setup the orbit for the zmax test
def setup_orbit_zmax(tp, axi=False):
    from galpy.orbit import Orbit

    if axi:
        o = Orbit([1.0, 0.0, 0.98, 0.05, 0.0])
    else:
        o = Orbit([1.0, 0.0, 0.98, 0.05, 0.0, 0.0])
    return o


# Setup the orbit for the apocenter test
def setup_orbit_analytic(tp, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 0.1, 0.9])
        else:
            o = Orbit([1.0, 0.1, 0.9, 0.0])
    else:
        if axi:
            o = Orbit([1.0, 0.1, 0.9, 0.0, 0.0])
        else:
            o = Orbit([1.0, 0.1, 0.9, 0.0, 0.0, 0.0])
    return o


# Setup the orbit for the zmax test
def setup_orbit_analytic_zmax(tp, axi=False):
    from galpy.orbit import Orbit

    if axi:
        o = Orbit([1.0, 0.0, 1.0, 0.05, 0.03])
    else:
        o = Orbit([1.0, 0.0, 1.0, 0.05, 0.03, 0.0])
    return o


# Setup the orbit for the ER, EZ test
def setup_orbit_analytic_EREz(tp, axi=False):
    from galpy.orbit import Orbit

    if axi:
        o = Orbit([1.0, 0.03, 1.0, 0.05, 0.03])
    else:
        o = Orbit([1.0, 0.03, 1.0, 0.05, 0.03, 0.0])
    return o


# Setup the orbit for the physical-coordinates test
def setup_orbit_physical(tp, axi=False, ro=None, vo=None):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 1.1, 1.1], ro=ro, vo=vo)
        else:
            o = Orbit([1.0, 1.1, 1.1, 0.0], ro=ro, vo=vo)
    else:
        if axi:
            o = Orbit([1.0, 1.1, 1.1, 0.1, 0.1], ro=ro, vo=vo)
        else:
            o = Orbit([1.0, 1.1, 1.1, 0.1, 0.1, 0.0], ro=ro, vo=vo)
    return o


# Setup the orbit for the energy test
def setup_orbit_flip(tp, ro, vo, zo, solarmotion, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.linearPotential):
        o = Orbit([1.0, 1.0], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
    elif isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit([1.0, 1.1, 1.1], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
        else:
            o = Orbit(
                [1.0, 1.1, 1.1, 0.0], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion
            )
    else:
        if axi:
            o = Orbit(
                [1.0, 1.1, 1.1, 0.1, 0.1], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion
            )
        else:
            o = Orbit(
                [1.0, 1.1, 1.1, 0.1, 0.1, 0.0],
                ro=ro,
                vo=vo,
                zo=zo,
                solarmotion=solarmotion,
            )
    return o


def check_radecetc_roWarning(o, funcName):
    # Convenience function to check whether the ro-needs-to-be-specified
    # warning is sounded
    with pytest.warns(galpyWarning) as record:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        getattr(o, funcName)()
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == f"Method {funcName}(.) requires ro to be given at Orbit initialization or at method evaluation; using default ro which is {8.0:f} kpc"
        )
    assert raisedWarning, (
        "Orbit method %s without ro specified should have thrown a warning, but didn't"
        % funcName
    )
    return None


def check_radecetc_voWarning(o, funcName):
    # Convenience function to check whether the vo-needs-to-be-specified
    # warning is sounded
    with pytest.warns(galpyWarning) as record:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        getattr(o, funcName)()
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == f"Method {funcName}(.) requires vo to be given at Orbit initialization or at method evaluation; using default vo which is {220.0:f} km/s"
        )
    assert raisedWarning, (
        "Orbit method %s without vo specified should have thrown a warning, but didn't"
        % funcName
    )
    return None


def check_integrate_t_asQuantity_warning(o, funcName):
    with pytest.warns(galpyWarning) as record:
        getattr(o, funcName)(1.0)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "You specified integration times as a Quantity, but are evaluating at times not specified as a Quantity; assuming that time given is in natural (internal) units (multiply time by unit to get output at physical time)"
        )
    assert raisedWarning, (
        "Orbit method %s with unitless time after integrating with unitful time should have thrown a warning, but didn't"
        % funcName
    )
    return None


def test_integrate_method_warning():
    """Test Orbit.integrate raises an error if method is invalid"""
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0])
    t = numpy.arange(0.0, 10.0, 0.001)
    with pytest.raises(ValueError):
        o.integrate(t, MWPotential2014, method="rk4")


def test_MovingObjectPotential_orbit():
    # Test integration of an object with a MovingObjectPotential
    # Test that orbits integrated by C and Python are the same
    from galpy.orbit import Orbit
    from galpy.potential import (
        HernquistPotential,
        MovingObjectPotential,
        MWPotential2014,
    )

    tmax = 5.0
    times = numpy.linspace(0, tmax, 101)

    orbit_pot = HernquistPotential(amp=1e-1, a=1e-1, ro=8, vo=220)
    o = Orbit([1.0, 0.03, 1.0, 0.05, 0.03, 0.0])
    o.integrate(times, MWPotential2014)
    orbit_potential = MovingObjectPotential(o, pot=orbit_pot)

    total_potential = MWPotential2014 + orbit_potential

    oc = Orbit([0.5, 0.5, 0.5, 0.05, 0.03, 0.0])
    op = Orbit([0.5, 0.5, 0.5, 0.05, 0.03, 0.0])
    oc.integrate(times, total_potential, method="leapfrog_c")
    op.integrate(times, total_potential, method="leapfrog")
    assert numpy.fabs(oc.x(tmax) - op.x(tmax)) < 10.0**-3.0, (
        "Final orbit position between C and Python integration in a MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.y(tmax) - op.y(tmax)) < 10.0**-3.0, (
        "Final orbit position between C and Python integration in a MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.z(tmax) - op.z(tmax)) < 10.0**-3.0, (
        "Final orbit position between C and Python integration in a MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.vx(tmax) - op.vx(tmax)) < 10.0**-3.0, (
        "Final orbit velocity between C and Python integration in a MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.vy(tmax) - op.vy(tmax)) < 10.0**-3.0, (
        "Final orbit velocity between C and Python integration in a MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.vz(tmax) - op.vz(tmax)) < 10.0**-3.0, (
        "Final orbit velocity between C and Python integration in a MovingObjectPotential is too large"
    )
    return None


def test_MovingObjectPotential_planar_orbit():
    # Test integration of an object with a MovingObjectPotential
    # Test that orbits integrated by C and Python are the same
    from galpy.orbit import Orbit
    from galpy.potential import (
        HernquistPotential,
        MovingObjectPotential,
        MWPotential2014,
    )

    tmax = 5.0
    times = numpy.linspace(0, tmax, 101)

    orbit_pot = HernquistPotential(amp=0.1, a=0.1, ro=8.0, vo=220.0)
    o = Orbit([0.4, 0.1, 0.6, 0.0])
    o.integrate(times, MWPotential2014)
    orbit_potential = MovingObjectPotential(o, pot=orbit_pot)

    total_potential = MWPotential2014 + orbit_potential

    oc = Orbit([0.5, -0.1, 0.5, 1.0])
    op = Orbit([0.5, -0.1, 0.5, 1.0])
    oc.integrate(times, total_potential, method="leapfrog_c")
    op.integrate(times, total_potential, method="leapfrog")

    assert numpy.fabs(oc.x(tmax) - op.x(tmax)) < 10.0**-3.0, (
        "Final orbit position between C and Python integration in a planar MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.y(tmax) - op.y(tmax)) < 10.0**-3.0, (
        "Final orbit position between C and Python integration in a planar MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.vx(tmax) - op.vx(tmax)) < 10.0**-3.0, (
        "Final orbit velocity between C and Python integration in a planar MovingObjectPotential is too large"
    )
    assert numpy.fabs(oc.vy(tmax) - op.vy(tmax)) < 10.0**-3.0, (
        "Final orbit velocity between C and Python integration in a planar MovingObjectPotential is too large"
    )
    return None


# Test that all integrators can start from a negative time
def test_integrate_negative_time():
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    dp = DehnenBarPotential()
    methods = [
        "odeint",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "dopr54_c",
        "dop853_c",
        "ias15_c",
    ]
    # negative time to negative time
    times = numpy.linspace(-70.0, -30.0, 1001)
    for method in methods:
        o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.1])
        o.integrate(times, MWPotential2014 + dp, method=method)
        assert (
            numpy.std(o.Jacobi(times)) / numpy.fabs(numpy.mean(o.Jacobi(times))) < 1e-7
        ), (
            f"Orbit integration with method {method} does not conserve energy when integrating from a negative time to a negative time"
        )
    # negative time to positive time
    times = numpy.linspace(-30.0, 10.0, 1001)
    for method in methods:
        o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.1])
        o.integrate(times, MWPotential2014 + dp, method=method)
        assert (
            numpy.std(o.Jacobi(times)) / numpy.fabs(numpy.mean(o.Jacobi(times))) < 1e-4
        ), (
            f"Orbit integration with method {method} does not conserve energy when integrating from a negative time to a positive time"
        )
    return None


# Test that all integrators can integrate backwards in time
def test_integrate_backwards():
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    dp = DehnenBarPotential()
    methods = [
        "odeint",
        "leapfrog",
        "leapfrog_c",
        "rk4_c",
        "rk6_c",
        "symplec4_c",
        "symplec6_c",
        "dopr54_c",
        "dop853_c",
        "ias15_c",
    ]
    # negative time to negative time
    times = numpy.linspace(-30.0, -70.0, 1001)
    for method in methods:
        o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.1])
        o.integrate(times, MWPotential2014 + dp, method=method)
        assert (
            numpy.std(o.Jacobi(times)) / numpy.fabs(numpy.mean(o.Jacobi(times))) < 1e-7
        ), (
            f"Orbit integration with method {method} does not conserve energy when integrating from a negative time to a negative time"
        )
    # positive time to negative time
    times = numpy.linspace(30.0, -10.0, 1001)
    for method in methods:
        o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.1])
        o.integrate(times, MWPotential2014 + dp, method=method)
        assert (
            numpy.std(o.Jacobi(times)) / numpy.fabs(numpy.mean(o.Jacobi(times))) < 1e-4
        ), (
            f"Orbit integration with method {method} does not conserve energy when integrating from a negative time to a positive time"
        )
    # positive time to positive time
    times = numpy.linspace(70.0, 30.0, 1001)
    for method in methods:
        o = Orbit([1.0, 0.1, 1.1, 0.1, 0.1, 0.1])
        o.integrate(times, MWPotential2014 + dp, method=method)
        assert (
            numpy.std(o.Jacobi(times)) / numpy.fabs(numpy.mean(o.Jacobi(times))) < 1e-4
        ), (
            f"Orbit integration with method {method} does not conserve energy when integrating from a negative time to a positive time"
        )
    return None


# Test that Orbit._call_internal(t0) and Orbit._call_internal(t=t0) return the same results
def test_call_internal_kwargs():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([1.0, 0.1, 1.2, 0.3, 0.2, 2.0])
    times = numpy.array([0.0, 10.0])
    o.integrate(times, lp)
    assert numpy.array_equal(o._call_internal(10.0), o._call_internal(t=10.0)), (
        "Orbit._call_internal(t0) and Orbit._call_internal(t=t0) return different results"
    )
    return None


def test_apy_sunkeywords_not_supplied():
    # Test for issues #709: print warning when a SkyCoord is used to initialize an
    # Orbit object, but the Sun's position and velocity are not specified through
    # galcen_distance, galcen_v_sun, and z_sun
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from galpy.orbit import Orbit

    # Just missing galcen_distance
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        # galcen_distance=8.3 * u.kpc,
        z_sun=0.025 * u.kpc,
        galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (galcen_distance) and this was not explicitly set in the Orbit initialization using the keywords (ro); this is required for Orbit initialization; proceeding with default value"
        )
    assert raisedWarning, (
        "Orbit initialization without galcen_distance should have thrown a warning, but didn't"
    )
    # Just missing z_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        galcen_distance=8.3 * u.kpc,
        # z_sun=0.025 * u.kpc,
        galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (z_sun) and this was not explicitly set in the Orbit initialization using the keywords (zo); this is required for Orbit initialization; proceeding with default value"
        )
    assert raisedWarning, (
        "Orbit initialization without z_sun should have thrown a warning, but didn't"
    )
    # Just missing galcen_v_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        galcen_distance=8.3 * u.kpc,
        z_sun=0.025 * u.kpc,
        # galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (galcen_v_sun) and this was not explicitly set in the Orbit initialization using the keywords (vo, solarmotion); this is required for Orbit initialization; proceeding with default value"
        )
    assert raisedWarning, (
        "Orbit initialization without galcen_v_sun should have thrown a warning, but didn't"
    )
    # Missing galcen_distance and z_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        # galcen_distance=8.3 * u.kpc,
        # z_sun=0.025 * u.kpc,
        galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (galcen_distance, z_sun) and these were not explicitly set in the Orbit initialization using the keywords (ro, zo); these are required for Orbit initialization; proceeding with default values"
        )
    assert raisedWarning, (
        "Orbit initialization without galcen_distance and z_sun should have thrown a warning, but didn't"
    )
    # Missing galcen_distance and galcen_v_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        # galcen_distance=8.3 * u.kpc,
        z_sun=0.025 * u.kpc,
        # galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (galcen_distance, galcen_v_sun) and these were not explicitly set in the Orbit initialization using the keywords (ro, vo, solarmotion); these are required for Orbit initialization; proceeding with default values"
        )
    assert raisedWarning, (
        "Orbit initialization without galcen_distance and galcen_v_sun should have thrown a warning, but didn't"
    )
    # Missing z_sun and galcen_v_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        galcen_distance=8.3 * u.kpc,
        # z_sun=0.025 * u.kpc,
        # galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (z_sun, galcen_v_sun) and these were not explicitly set in the Orbit initialization using the keywords (zo, vo, solarmotion); these are required for Orbit initialization; proceeding with default values"
        )
    assert raisedWarning, (
        "Orbit initialization without z_sun and galcen_v_sun should have thrown a warning, but didn't"
    )
    # Missing all: galcen_distance, z_sun, galcen_v_sun
    vxvv = SkyCoord(
        ra=1 * u.deg,
        dec=1 * u.deg,
        distance=20.8 * u.pc,
        pm_ra_cosdec=0.0 * u.mas / u.yr,
        pm_dec=0.0 * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        # galcen_distance=8.3 * u.kpc,
        # z_sun=0.025 * u.kpc,
        # galcen_v_sun=[-11.1, 220.0, 7.25] * u.km / u.s,
    )
    with pytest.warns(galpyWarning) as record:
        o = Orbit(vxvv)
    raisedWarning = False
    for rec in record:
        # check that the message matches
        raisedWarning += (
            str(rec.message.args[0])
            == "Supplied SkyCoord does not contain (galcen_distance, z_sun, galcen_v_sun) and these were not explicitly set in the Orbit initialization using the keywords (ro, zo, vo, solarmotion); these are required for Orbit initialization; proceeding with default values"
        )
    assert raisedWarning, (
        "Orbit initialization without galcen_distance, z_sun, and galcen_v_sun should have thrown a warning, but didn't"
    )

    return None


# Test runs for two different rtol/atol values for 1d, 2d and 3d orbit integration on a Kepler pot. for all algorithms available
# test is passed if the difference in the orbital reconstruction is not exactly zero for all sampling points and
# if the orbital energy loss is smaller for the more precise rtol/atol orbit reconstruction
def test_1d_tol_integration():
    from galpy import orbit

    ttol_vec = [1e-12, 1e-6]
    times = numpy.linspace(
        0.0, 10.0, 250
    )  # with this time stepping, rk6_c and symplec6_c results will not be affected by changes in rtol/atol
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
    pot = potential.KeplerPotential(amp=1.0, normalize=True).toVertical(1.0)
    for integrator in integrators:
        o_list = []
        for cnt_tol in numpy.arange(len(ttol_vec)):
            # initialise a test orbit with few rounds, integrate trajectory, append to list of orbits
            o = orbit.Orbit([1.0, 0.8])  # Orbit([R, vR])
            o.integrate(
                times,
                pot,
                method=integrator,
                rtol=ttol_vec[cnt_tol],
                atol=ttol_vec[cnt_tol],
            )
            o_list.append(o)

        # make test for differing reconstruction precision and energy loss along the orbits
        Delta_r = numpy.sum(numpy.abs(o_list[0].r(times) - o_list[1].r(times)))
        Delta_E = numpy.sum(numpy.abs(o_list[0].E(times) - o_list[1].E(times)))

        # if special integrators yield same reconstructions
        if integrator == "rk6_c" or integrator == "symplec6_c":
            assert Delta_r == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - position difference"
            )
            assert Delta_E == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - energy difference"
            )
        else:  # for all other integration routines check that differences are moderate as expected
            assert Delta_r > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - position difference"
            )
            assert Delta_r < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - position difference"
            )
            assert Delta_E > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - energy difference"
            )
            assert Delta_E < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - energy difference"
            )

    return None


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
    for integrator in integrators:
        o_list = []
        for cnt_tol in numpy.arange(len(ttol_vec)):
            # initialise a test orbit with few rounds, integrate trajectory, append to list of orbits
            o = orbit.Orbit([1.0, 0.1, 0.8])  # Orbit([R, vR, vT])
            o.integrate(
                times,
                pot,
                method=integrator,
                rtol=ttol_vec[cnt_tol],
                atol=ttol_vec[cnt_tol],
            )
            o_list.append(o)

        # make test for differing reconstruction precision and energy loss along the orbits
        Delta_r = numpy.sum(numpy.abs(o_list[0].r(times) - o_list[1].r(times)))
        Delta_E = numpy.sum(numpy.abs(o_list[0].E(times) - o_list[1].E(times)))

        # if special integrators yield same reconstructions
        if integrator == "rk6_c":
            assert Delta_r == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - position difference"
            )
            assert Delta_E == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - energy difference"
            )
        else:  # for all other integration routines check that differences are moderate as expected
            assert Delta_r > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - position difference"
            )
            assert Delta_r < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - position difference"
            )
            assert Delta_E > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - energy difference"
            )
            assert Delta_E < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - energy difference"
            )

    return None


def test_3d_tol_integration():
    from galpy import orbit

    ttol_vec = [1e-12, 1e-6]
    times = numpy.linspace(
        0.0, 2.1, 250
    )  # with this time stepping, rk6_c and symplec6_c results will not be affected by changes in rtol/atol
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
    for integrator in integrators:
        o_list = []
        for cnt_tol in numpy.arange(len(ttol_vec)):
            # initialise a test orbit with few rounds, integrate trajectory, append to list of orbits
            o = orbit.Orbit(
                [1.0, 0.8, 0.1, 0.03, 0.17, 0.0]
            )  # Orbit([R, vR, vT, z, vZ, phi])
            o.integrate(
                times,
                pot,
                method=integrator,
                rtol=ttol_vec[cnt_tol],
                atol=ttol_vec[cnt_tol],
            )
            o_list.append(o)

        # make test for differing reconstruction precision and energy loss along the orbits
        Delta_r = numpy.sum(numpy.abs(o_list[0].r(times) - o_list[1].r(times)))
        Delta_E = numpy.sum(numpy.abs(o_list[0].E(times) - o_list[1].E(times)))

        # if special integrators yield same reconstructions
        if integrator == "rk6_c" or integrator == "symplec6_c":
            assert Delta_r == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - position difference"
            )
            assert Delta_E == 0.0, (
                f"{integrator} orbit integration is unexpectedly sensitive to rtol/atol - energy difference"
            )
        else:  # for all other integration routines check that differences are moderate as expected
            assert Delta_r > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - position difference"
            )
            assert Delta_r < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - position difference"
            )
            assert Delta_E > 0.0, (
                f"{integrator} orbit integration unexpectedly not sensitive to rtol/atol - energy difference"
            )
            assert Delta_E < 0.1, (
                f"{integrator} orbit integration has worse than expected reconstruction precision - energy difference"
            )

    return None


# Test orbit continuation feature
def test_orbit_continuation_forward():
    # Test forward continuation of orbit integration
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)
    r1_at_10 = o.r(10.0)

    # Second integration continuing from first
    t2 = numpy.linspace(10.0, 20.0, 101)
    o.integrate(t2, MWPotential2014)

    # Check that time array was merged
    assert len(o.time()) == 201, "Time array should have 201 points after continuation"
    assert numpy.isclose(o.time()[0], 0.0), "First time should be 0"
    assert numpy.isclose(o.time()[-1], 20.0), "Last time should be 20"
    assert numpy.isclose(o.time()[100], 10.0), "Middle time should be 10"

    # Check that orbit was merged
    assert o.orbit.shape[1] == 201, "Orbit should have 201 time points"

    # Check that r at junction is continuous
    assert numpy.isclose(o.r(10.0), r1_at_10), "r should be continuous at junction"

    # Check that methods work across the full range
    r_all = o.r(o.time())
    assert r_all.shape == (201,), "r should work for all times"

    return None


def test_orbit_continuation_backward():
    # Test backward continuation of orbit integration
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration (forward)
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)
    r1_at_0 = o.r(0.0)

    # Second integration going backward
    t2 = numpy.linspace(0.0, -10.0, 101)
    o.integrate(t2, MWPotential2014)

    # Check that time array was merged correctly
    assert len(o.time()) == 201, "Time array should have 201 points after continuation"
    assert numpy.isclose(o.time()[0], -10.0), "First time should be -10"
    assert numpy.isclose(o.time()[-1], 10.0), "Last time should be 10"
    assert numpy.isclose(o.time()[100], 0.0), "Middle time should be 0"

    # Check that orbit was merged
    assert o.orbit.shape[1] == 201, "Orbit should have 201 time points"

    # Check that r at junction is continuous
    assert numpy.isclose(o.r(0.0), r1_at_0), "r should be continuous at junction"

    # Check that methods work across the full range
    r_all = o.r(o.time())
    assert r_all.shape == (201,), "r should work for all times"

    return None


def test_orbit_continuation_different_spacing():
    # Test continuation with different time spacing
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration with 101 points
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Second integration with 51 points (different spacing)
    t2 = numpy.linspace(10.0, 20.0, 51)
    o.integrate(t2, MWPotential2014)

    # Check that time array was merged
    assert len(o.time()) == 151, "Time array should have 151 points (101 + 51 - 1)"
    assert numpy.isclose(o.time()[0], 0.0), "First time should be 0"
    assert numpy.isclose(o.time()[-1], 20.0), "Last time should be 20"

    # Check that orbit was merged
    assert o.orbit.shape[1] == 151, "Orbit should have 151 time points"

    return None


def test_orbit_continuation_methods():
    # Test that orbit methods work correctly after continuation
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Store values at t=10
    r_at_10_before = o.r(10.0)
    vR_at_10_before = o.vR(10.0)
    vT_at_10_before = o.vT(10.0)
    z_at_10_before = o.z(10.0)
    vz_at_10_before = o.vz(10.0)
    E_at_10_before = o.E(10.0)

    # Second integration continuing
    t2 = numpy.linspace(10.0, 20.0, 101)
    o.integrate(t2, MWPotential2014)

    # Check that values at t=10 are the same
    assert numpy.isclose(o.r(10.0), r_at_10_before), "r(10) should be continuous"
    assert numpy.isclose(o.vR(10.0), vR_at_10_before), "vR(10) should be continuous"
    assert numpy.isclose(o.vT(10.0), vT_at_10_before), "vT(10) should be continuous"
    assert numpy.isclose(o.z(10.0), z_at_10_before), "z(10) should be continuous"
    assert numpy.isclose(o.vz(10.0), vz_at_10_before), "vz(10) should be continuous"
    assert numpy.isclose(o.E(10.0), E_at_10_before), "E(10) should be continuous"

    # Check that methods work for the full time range
    r_all = o.r(o.time())
    assert r_all.shape == (201,), "r should work for all times"

    E_all = o.E(o.t)
    assert E_all.shape == (201,), "E should work for all times"

    return None


def test_orbit_continuation_no_duplicate_time():
    # Test that the duplicate time point at the junction is not included twice
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Second integration
    t2 = numpy.linspace(10.0, 20.0, 101)
    o.integrate(t2, MWPotential2014)

    # Check that time 10 appears only once
    t_10_count = numpy.sum(numpy.isclose(o.t, 10.0))
    assert t_10_count == 1, "Time 10 should appear exactly once in merged array"

    return None


def test_orbit_continuation_vs_noncontinued_forward():
    # Test that continued integration matches non-continued approach
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Continued integration
    o_cont = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o_cont.integrate(t1, MWPotential2014)
    t2 = numpy.linspace(10.0, 20.0, 101)
    o_cont.integrate(t2, MWPotential2014)

    # Non-continued integration (old approach)
    o_full = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t_full = numpy.linspace(0.0, 20.0, 201)
    o_full.integrate(t_full, MWPotential2014)

    # Compare r values across the full range
    r_cont = o_cont.r(o_cont.time())
    r_full = o_full.r(o_full.time())

    assert numpy.allclose(r_cont, r_full, rtol=1e-10), (
        "Continued integration r values should match non-continued integration"
    )

    # Compare at specific times
    test_times = [0.0, 5.0, 10.0, 15.0, 20.0]
    for t in test_times:
        assert numpy.isclose(o_cont.r(t), o_full.r(t), rtol=1e-10), (
            f"r should match at t={t}"
        )

    return None


def test_orbit_continuation_vs_noncontinued_reinit():
    # Test continuation by re-initializing at junction point
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # First integration
    o1 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o1.integrate(t1, MWPotential2014)

    # Continue from the end point
    o_cont = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    o_cont.integrate(t1, MWPotential2014)
    t2 = numpy.linspace(10.0, 20.0, 101)
    o_cont.integrate(t2, MWPotential2014)

    # Non-continued: re-initialize at end of t1 and integrate t2
    # Get state at t=10
    o_reinit = o1(10.0)
    o_reinit.integrate(t2, MWPotential2014)

    # Compare second half of continued orbit to re-initialized orbit
    # The re-initialized orbit starts at t=10, so we compare from that point
    for i, t in enumerate(t2):
        r_cont = o_cont.r(t)
        r_reinit = o_reinit.r(t)
        assert numpy.isclose(r_cont, r_reinit, rtol=1e-10), (
            f"r should match at t={t} (continued vs re-initialized)"
        )

    return None


def test_orbit_continuation_vs_noncontinued_backward():
    # Test that backward continued integration works correctly
    # Backward continuation integrates from state at t=0 backward in time
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Continued integration (forward then backward)
    o_cont = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o_cont.integrate(t1, MWPotential2014)

    t2 = numpy.linspace(0.0, -10.0, 101)
    o_cont.integrate(t2, MWPotential2014)

    # Compare to re-initializing from initial condition and integrating backward
    o_reinit = o_cont()
    o_reinit.integrate(t2, MWPotential2014)

    # The backward part should match for the full time range
    test_times = numpy.linspace(-10.0, 0.0, 11)
    for t in test_times:
        r_cont = o_cont.r(t)
        r_reinit = o_reinit.r(t)
        assert numpy.isclose(r_cont, r_reinit, rtol=1e-10), (
            f"Backward continuation should match re-initialized integration at t={t}"
        )

    return None


def test_orbit_continuation_different_potential_warning():
    # Test that warning is issued when continuing with different potential
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Second integration with different potential should issue warning
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, LogarithmicHaloPotential())

        # Check that a warning was issued
        assert len(w) > 0, (
            "Warning should be issued when continuing with different potential"
        )
        warning_messages = [str(warning.message) for warning in w]
        assert any("different potential" in msg for msg in warning_messages), (
            "Warning should mention different potential"
        )

    # Check that continuation still happened
    assert len(o.time()) == 201, "Integration should still be continued"

    return None


def test_orbit_continuation_potential_comparison_planar():
    # Test potential comparison for planar potentials (2D orbits)
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    # Test with 2D orbit (planar)
    o = Orbit([1.0, 0.1, 1.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Continue with same potential - should NOT warn
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, MWPotential2014)
        # Filter for the specific warning we care about
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) == 0, (
            f"Should not warn when continuing with same potential, but received warnings: {[str(w.message) for w in pot_warnings]}"
        )

    # New orbit, continue with different potential - should warn
    o2 = Orbit([1.0, 0.1, 1.1])
    o2.integrate(t1, MWPotential2014)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o2.integrate(t2, LogarithmicHaloPotential())
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, (
            "Should warn when continuing with different potential"
        )

    return None


def test_orbit_continuation_potential_comparison_linear():
    # Test potential comparison for linear potentials (1D orbits)
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    vertPot = MWPotential2014[0].toVertical(1.0)
    # Test with 1D orbit (linear)
    o = Orbit([1.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, vertPot)
    # Continue with same potential - should NOT warn
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, vertPot)
        # Filter for the specific warning we care about
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) == 0, (
            f"Should not warn when continuing with same potential, but received warnings: {[str(w.message) for w in pot_warnings]}"
        )
    # New orbit, continue with different potential - should warn
    o2 = Orbit([1.0, 0.1])
    o2.integrate(t1, vertPot)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o2.integrate(t2, LogarithmicHaloPotential().toVertical(1.0))
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, (
            "Should warn when continuing with different potential"
        )
    return None


def test_orbit_continuation_potential_comparison_single_vs_list():
    # Test potential comparison between single potential and list
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    # MWPotential2014 is a list of potentials
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)

    # Continue with single potential - should warn (different)
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, LogarithmicHaloPotential())
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, "Should warn when list vs single potential"

    return None


def test_orbit_continuation_potential_comparison_nested_list():
    # Test potential comparison with nested lists
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014, flatten

    # Use a list of potentials
    pot1 = MWPotential2014
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, pot1)

    # Continue with same list - should NOT warn
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, pot1)
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) == 0, "Should not warn with same list potential"

    # Continue with different list - should warn
    pot2 = [LogarithmicHaloPotential(), LogarithmicHaloPotential(normalize=0.9)]
    o2 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    o2.integrate(t1, pot1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o2.integrate(t2, pot2)
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, "Should warn with different list potential"

    # Test with actual nested list
    pot3 = LogarithmicHaloPotential() + [LogarithmicHaloPotential(normalize=0.9)]
    o3 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    o3.integrate(t1, pot3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o3.integrate(t2, pot3)
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) == 0, "Should not warn with same nested list potential"

    return None


def test_orbit_continuation_potential_comparison_planar_wrapper():
    # Test that planar potential wrappers are compared correctly
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    # 2D orbit uses toPlanarPotential internally
    o = Orbit([1.0, 0.1, 1.1])
    t1 = numpy.linspace(0.0, 10.0, 101)

    # First integration with MWPotential2014
    o.integrate(t1, MWPotential2014)

    # Continue with same potential (MWPotential2014) - should NOT warn
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, MWPotential2014)
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) == 0, "Planar wrappers of same potential should match"

    # New orbit, continue with different potential - should warn
    o2 = Orbit([1.0, 0.1, 1.1])
    o2.integrate(t1, MWPotential2014)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o2.integrate(t2, LogarithmicHaloPotential())
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, (
            "Planar wrappers of different potentials should not match"
        )

    return None


def test_orbit_continuation_potential_comparison_list_length():
    # Test potential comparison when list lengths differ
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014

    # Start with a list of potentials
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, MWPotential2014)  # This is a list of 3 potentials

    # Continue with a list of different length - should warn
    pot2 = [LogarithmicHaloPotential()]  # Only 1 potential
    t2 = numpy.linspace(10.0, 20.0, 101)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o.integrate(t2, pot2)
        pot_warnings = [
            warning for warning in w if "different potential" in str(warning.message)
        ]
        assert len(pot_warnings) > 0, "Should warn when list lengths differ"

    return None


def test_orbit_continuation_1d_forward():
    # Test forward continuation for 1D orbit
    from galpy.orbit import Orbit
    from galpy.potential import KGPotential

    pot = KGPotential(amp=1.0, K=1.0)
    o = Orbit([1.0, 0.1])  # 1D orbit: x, vx

    # First integration
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, pot)
    x_at_10 = o.x(10.0)

    # Second integration continuing from first
    t2 = numpy.linspace(10.0, 20.0, 101)
    o.integrate(t2, pot)

    # Check that time array was merged
    times = o.time()
    assert len(times) == 201, "Time array should have 201 points after continuation"
    assert numpy.isclose(times[0], 0.0), "First time should be 0"
    assert numpy.isclose(times[-1], 20.0), "Last time should be 20"

    # Check continuity at junction
    assert numpy.isclose(o.x(10.0), x_at_10), "x should be continuous at junction"

    # Compare to full integration
    o_full = Orbit([1.0, 0.1])
    t_full = numpy.linspace(0.0, 20.0, 201)
    o_full.integrate(t_full, pot)

    # Should match at various points
    for t in [0.0, 5.0, 10.0, 15.0, 20.0]:
        assert numpy.isclose(o.x(t), o_full.x(t), rtol=1e-10), (
            f"x should match at t={t}"
        )

    return None


def test_orbit_continuation_1d_backward():
    # Test backward continuation for 1D orbit
    from galpy.orbit import Orbit
    from galpy.potential import KGPotential

    pot = KGPotential(amp=1.0, K=1.0)
    o = Orbit([1.0, 0.1])

    # First integration forward
    t1 = numpy.linspace(0.0, 10.0, 101)
    o.integrate(t1, pot)

    # Second integration backward
    t2 = numpy.linspace(0.0, -10.0, 101)
    o.integrate(t2, pot)

    # Check that time array was merged
    times = o.time()
    assert len(times) == 201, "Time array should have 201 points"
    assert numpy.isclose(times[0], -10.0), "First time should be -10"
    assert numpy.isclose(times[-1], 10.0), "Last time should be 10"

    # Compare to re-initialization approach
    o_reinit = o()
    o_reinit.integrate(t2, pot)

    test_times = numpy.linspace(-10.0, 0.0, 11)
    for t in test_times:
        assert numpy.isclose(o.x(t), o_reinit.x(t), rtol=1e-10), (
            f"Backward continuation should match re-initialized integration at t={t}"
        )

    return None


def test_orbit_continuation_chained():
    # Test that continuations can be chained (continuing from an already continued orbit)
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])

    # First integration
    t1 = numpy.linspace(0.0, 5.0, 51)
    o.integrate(t1, MWPotential2014)

    # First continuation
    t2 = numpy.linspace(5.0, 10.0, 51)
    o.integrate(t2, MWPotential2014)

    # Second continuation (chaining)
    t3 = numpy.linspace(10.0, 15.0, 51)
    o.integrate(t3, MWPotential2014)

    # Check that all three integrations were merged
    times = o.time()
    assert len(times) == 151, (
        "Time array should have 151 points after two continuations"
    )
    assert numpy.isclose(times[0], 0.0), "First time should be 0"
    assert numpy.isclose(times[-1], 15.0), "Last time should be 15"

    # Compare to full integration over entire time range
    o_full = Orbit([1.0, 0.1, 1.1, 0.0, 0.1])
    t_full = numpy.linspace(0.0, 15.0, 151)
    o_full.integrate(t_full, MWPotential2014)

    # Check that the entire trajectory matches
    times_check = numpy.linspace(0.0, 15.0, 31)  # Sample at various points
    for t in times_check:
        assert numpy.isclose(o.r(t), o_full.r(t), rtol=1e-10), (
            f"Chained continuation should match full integration at t={t}"
        )
        assert numpy.isclose(o.E(t), o_full.E(t), rtol=1e-10), (
            f"Energy should match at t={t}"
        )

    return None


# Tests for automatic time determination in orbit integration
def test_integrate_auto_default_3D():
    # Test auto-time integration with default (10 tdyn) for 3D orbit
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o.integrate(MWPotential2014)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0, "Orbit should be integrated"
    # Check array length: 101 points/tdyn × 10 tdyn + 1 = 1011
    assert len(o.t) == 1011, f"Expected 1011 time points, got {len(o.t)}"
    # Check time starts at 0
    assert numpy.abs(o.t[0]) < 1e-10, "Time should start at 0"
    # Check time is positive
    assert o.t[-1] > 0, "Final time should be positive"
    return None


def test_integrate_auto_default_2D():
    # Test auto-time integration with default (10 tdyn) for 2D orbit
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0])
    o.integrate(MWPotential2014)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0, "Orbit should be integrated"
    # Check array length: 101 points/tdyn × 10 tdyn + 1 = 1011
    assert len(o.t) == 1011, f"Expected 1011 time points, got {len(o.t)}"
    # Check time starts at 0
    assert numpy.abs(o.t[0]) < 1e-10, "Time should start at 0"
    # Check time is positive
    assert o.t[-1] > 0, "Final time should be positive"
    return None


def test_integrate_auto_1D_raises():
    # Test that 1D orbit raises ValueError
    from galpy.orbit import Orbit
    from galpy.potential import KeplerPotential

    o = Orbit([1.0, 0.1])
    kp = KeplerPotential(normalize=1.0)
    with pytest.raises(ValueError, match="not supported for 1D orbits"):
        o.integrate(kp)
    return None


def test_integrate_auto_composite_pot():
    # Test auto-time integration with CompositePotential
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential, NFWPotential

    hp = HernquistPotential(amp=2.0, a=1.3)
    nfw = NFWPotential(amp=1.0, a=2.0)
    pot = hp + nfw  # CompositePotential

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o.integrate(pot)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0, "Orbit should be integrated"
    # Check array length
    assert len(o.t) == 1011, f"Expected 1011 time points, got {len(o.t)}"
    return None


def test_integrate_auto_planar_pot():
    # Test auto-time integration with planar potential
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0])
    o.integrate(MWPotential2014)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0, "Orbit should be integrated"
    assert len(o.t) == 1011, f"Expected 1011 time points, got {len(o.t)}"
    return None


def test_integrate_auto_r_zero_raises():
    # Test that orbit at r=0 raises ValueError
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Orbit at r=0 (R=0, z=0)
    o = Orbit([0.0, 0.1, 0.1, 0.0, 0.1, 0.0])
    with pytest.raises(ValueError, match="r ≈ 0"):
        o.integrate(MWPotential2014)
    return None


def test_integrate_auto_backward_compat():
    # Test that explicit time array still works (backward compatibility)
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o1 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o2 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])

    t = numpy.linspace(0, 10, 100)
    o1.integrate(t, MWPotential2014)
    o2.integrate(t, MWPotential2014)

    # Both should give same results
    assert len(o1.t) == 100, "Explicit time array should have 100 points"
    assert len(o2.t) == 100, "Explicit time array should have 100 points"
    assert numpy.allclose(o1.t, o2.t), "Times should match"
    return None


def test_integrate_auto_energy_conservation():
    # Test that energy is conserved over auto-generated time
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o.integrate(MWPotential2014)

    # Energy should be conserved
    E_initial = o.E(o.t[0])
    E_final = o.E(o.t[-1])
    assert numpy.abs((E_final - E_initial) / E_initial) < 1e-6, (
        "Energy should be conserved to better than 1e-6"
    )
    return None


def test_integrate_auto_multiple_orbits():
    # Test auto-time integration with multiple orbits
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Multiple orbits at different radii
    o = Orbit([[1.0, 0.1, 1.1, 0.0, 0.1, 0.0], [2.0, 0.2, 1.2, 0.1, 0.2, 0.1]])
    o.integrate(MWPotential2014)

    # Should use max(r) for time determination
    assert hasattr(o, "t") and len(o.t) > 0, "Orbit should be integrated"
    assert len(o.t) == 1011, f"Expected 1011 time points, got {len(o.t)}"
    return None


def test_integrate_auto_continuation():
    # Test that continuation behavior works with auto-time
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])

    # First integration with explicit time
    t1 = numpy.linspace(0, 5, 100)
    o.integrate(t1, MWPotential2014)

    # Second integration continuing from first (also explicit time)
    t2 = numpy.linspace(5, 10, 51)
    o.integrate(t2, MWPotential2014)

    # Should have merged
    assert len(o.t) == 150, (
        f"Expected 150 time points after continuation, got {len(o.t)}"
    )
    assert numpy.isclose(o.t[0], 0), "Time should start at 0"
    assert numpy.isclose(o.t[-1], 10), "Time should end at 10"
    return None


def test_integrate_auto_composite_with_bar_3D():
    # Test auto-time with composite potential where some components fail tdyn
    # DehnenBarPotential doesn't support tdyn, so should filter to working components
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    # Add DehnenBarPotential which doesn't support tdyn
    pot = MWPotential2014 + DehnenBarPotential()

    # Should work by filtering to MWPotential2014 components
    o.integrate(pot)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0
    # Should have 1011 points (101 × 10 + 1)
    assert len(o.t) == 1011
    assert numpy.abs(o.t[0]) < 1e-10
    assert o.t[-1] > 0
    return None


def test_integrate_auto_composite_with_bar_2D():
    # Test auto-time with 2D composite where some components fail vcirc
    # DehnenBarPotential.toPlanar() doesn't support vcirc, should filter to working components
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    o = Orbit([1.0, 0.1, 1.1, 0.0])
    # Add planar bar potential which doesn't support vcirc
    pot = MWPotential2014[0].toPlanar() + DehnenBarPotential().toPlanar()

    # Should work by filtering to working planar potentials and using vcirc fallback
    o.integrate(pot)

    # Check integration occurred
    assert hasattr(o, "t") and len(o.t) > 0
    # Should have 1011 points (101 × 10 + 1)
    assert len(o.t) == 1011
    assert numpy.abs(o.t[0]) < 1e-10
    assert o.t[-1] > 0
    return None


def test_integrate_auto_no_tdyn_no_vcirc_raises():
    # Test that ValueError is raised when potential supports neither tdyn nor vcirc
    # DehnenBarPotential.toPlanar() doesn't support either method
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential

    o = Orbit([1.0, 0.1, 1.1, 0.0])
    pot = DehnenBarPotential().toPlanar()

    # Should raise ValueError since neither tdyn nor vcirc work
    with pytest.raises(ValueError, match="Cannot calculate dynamical time"):
        o.integrate(pot)
    return None


def test_integrate_auto_deprecated_list():
    # Test deprecated list of potentials interface (still needs to work for backward compatibility)
    # Need to suppress DeprecationWarning since tests run with warnings as errors
    import warnings

    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential, NFWPotential

    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    # Create list of potentials (deprecated syntax)
    pot_list = [NFWPotential(amp=1.0, a=2.0), HernquistPotential(amp=2.0, a=1.3)]

    # Suppress the deprecation warning for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        o.integrate(pot_list)

    # Check integration occurred with default 10 tdyn
    assert hasattr(o, "t") and len(o.t) > 0
    assert len(o.t) == 1011  # 101 × 10 + 1
    return None


def test_integrate_auto_tdyn_filtering_consistency_3D():
    # Test that dynamical time is the same whether using MWPotential2014 alone
    # or MWPotential2014 + DehnenBarPotential (which doesn't support tdyn)
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    # Create two orbits at the same position
    o1 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    o2 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])

    # Integrate with MWPotential2014 alone
    o1.integrate(MWPotential2014)

    # Integrate with MWPotential2014 + DehnenBarPotential
    # DehnenBarPotential doesn't support tdyn, so should be filtered out
    pot_composite = MWPotential2014 + DehnenBarPotential()
    o2.integrate(pot_composite)

    # The integration times should be identical since they should use the same tdyn
    # (calculated from MWPotential2014 only in both cases)
    assert numpy.allclose(o1.t, o2.t), (
        f"Integration times differ: {o1.t[-1]} vs {o2.t[-1]}"
    )
    return None


def test_integrate_auto_vcirc_filtering_consistency_2D():
    # Test that dynamical time is the same whether using planar MWPotential2014 alone
    # or planar MWPotential2014 + planar DehnenBarPotential (which doesn't support vcirc)
    from galpy.orbit import Orbit
    from galpy.potential import DehnenBarPotential, MWPotential2014

    # Create two orbits at the same position
    o1 = Orbit([1.0, 0.1, 1.1, 0.0])
    o2 = Orbit([1.0, 0.1, 1.1, 0.0])

    # Integrate with planar MWPotential2014 alone
    pot_planar = MWPotential2014.toPlanar()
    o1.integrate(pot_planar)

    # Integrate with planar MWPotential2014 + planar DehnenBarPotential
    # Planar DehnenBarPotential doesn't support vcirc, so should be filtered out
    pot_composite = MWPotential2014.toPlanar() + DehnenBarPotential().toPlanar()
    o2.integrate(pot_composite)

    # The integration times should be identical since they should use the same vcirc
    # (calculated from planar MWPotential2014 only in both cases)
    assert numpy.allclose(o1.t, o2.t), (
        f"Integration times differ: {o1.t[-1]} vs {o2.t[-1]}"
    )
    return None


def test_orbit_align_to_orbit():
    # Orbit.align_to_orbit() is a thin method wrapper around
    # galpy.util.coords.align_to_orbit — must forward this orbit's
    # galactocentric Cartesian kinematics plus Xsun/Zsun and produce
    # the same rotation matrix as the coords function.
    from galpy.orbit import Orbit
    from galpy.util import coords as gcoords

    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    T_method = prog.align_to_orbit()
    T_func = gcoords.align_to_orbit(
        float(prog.x(use_physical=False)),
        float(prog.y(use_physical=False)),
        float(prog.z(use_physical=False)),
        float(prog.vx(use_physical=False)),
        float(prog.vy(use_physical=False)),
        float(prog.vz(use_physical=False)),
        Xsun=1.0,
        Zsun=prog._zo / prog._ro,
    )
    assert T_method.shape == (3, 3)
    assert numpy.allclose(T_method, T_func)
    # center_phi1 kwarg threads through
    T0_method = prog.align_to_orbit(center_phi1=0.0)
    T0_func = gcoords.align_to_orbit(
        float(prog.x(use_physical=False)),
        float(prog.y(use_physical=False)),
        float(prog.z(use_physical=False)),
        float(prog.vx(use_physical=False)),
        float(prog.vy(use_physical=False)),
        float(prog.vz(use_physical=False)),
        Xsun=1.0,
        Zsun=prog._zo / prog._ro,
        center_phi1=0.0,
    )
    assert numpy.allclose(T0_method, T0_func)


def test_orbit_phi1phi2_after_align_to_orbit():
    # After Orbit.align_to_orbit() stashes a custom_transform, the
    # phi1/phi2/pmphi1/pmphi2 accessors should work without further setup
    # and reproduce coords.radec_to_custom / pmrapmdec_to_custom directly.
    from galpy.orbit import Orbit
    from galpy.util import coords as gcoords

    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    T = prog.align_to_orbit()
    # The orbit's own phi1/phi2 should land at (180, ~0) by the
    # default center_phi1=180 alignment, matching the analytical
    # round-trip via radec_to_custom.
    p12 = gcoords.radec_to_custom(
        numpy.atleast_1d(prog.ra()),
        numpy.atleast_1d(prog.dec()),
        T=T,
        degree=True,
    )
    assert abs(float(prog.phi1()) - p12[0, 0]) < 1e-8
    assert abs(float(prog.phi2()) - p12[0, 1]) < 1e-8
    pm12 = gcoords.pmrapmdec_to_custom(
        numpy.atleast_1d(prog.pmra()),
        numpy.atleast_1d(prog.pmdec()),
        numpy.atleast_1d(prog.ra()),
        numpy.atleast_1d(prog.dec()),
        T=T,
        degree=True,
    )
    assert abs(float(prog.pmphi1()) - pm12[0, 0]) < 1e-8
    assert abs(float(prog.pmphi2()) - pm12[0, 1]) < 1e-8


def test_orbit_phi1phi2_T_kwarg_override():
    # Without calling align_to_orbit, an explicit T= still works (and
    # overrides any stashed matrix).
    from galpy.orbit import Orbit

    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    # Identity T means phi1=ra, phi2=dec (modulo wrap on the equator)
    val = float(prog.phi1(T=numpy.eye(3)))
    assert abs(val - float(prog.ra())) < 1e-8


def test_orbit_phi1_no_transform_raises():
    # Without align_to_orbit and without an explicit T=, the accessors
    # raise so the user knows to set up the rotation matrix.
    from galpy.orbit import Orbit

    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    import pytest

    for name in ("phi1", "phi2", "pmphi1", "pmphi2"):
        with pytest.raises(RuntimeError):
            getattr(prog, name)()


def test_orbit_plot_phi1phi2():
    # Orbit.plot dispatches d1='phi1', d2='phi2' through the new
    # accessors; smoke-test that the call returns successfully and
    # picks up the labeldict_radec entries.
    import matplotlib

    matplotlib.use("Agg")
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    prog = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
        ro=8.0,
        vo=220.0,
    )
    prog.align_to_orbit()
    prog.integrate(numpy.linspace(0, 1, 11), MWPotential2014)
    line = prog.plot(d1="phi1", d2="phi2")
    assert line and len(line) >= 1
    line2 = prog.plot(d1="phi1", d2="pmphi2")
    assert line2 and len(line2) >= 1


def test_orbit_phi1phi2_multi_shape_and_time():
    # phi1/phi2/pmphi1/pmphi2 must broadcast correctly over (N orbits,
    # M times): shape (N, M) on the input vxvv shape (N,), and each
    # row should match the per-orbit single-Orbit accessor result.
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    vxvv = numpy.array(
        [
            [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596],
            [1.20, 0.10, -0.95, 0.50, -0.20, 0.30],
            [1.40, 0.20, -1.00, 0.70, -0.30, 0.20],
        ]
    )
    multi = Orbit(vxvv, ro=8.0, vo=220.0)
    assert multi.shape == (3,)
    ts = numpy.linspace(0, 1, 4)
    multi.integrate(ts, MWPotential2014)
    # Build a custom transform from the first orbit, share across all
    T = multi[0].align_to_orbit()
    multi._custom_transform = T
    out = multi.phi1(ts)
    assert out.shape == (3, 4), f"phi1 shape {out.shape} (expected (3, 4))"
    # Each row should agree with a per-orbit call (which we have to set
    # up the transform on separately because Orbit getitem yields a
    # fresh instance without the parent's _custom_transform)
    for i in range(3):
        single = multi[i]
        single._custom_transform = T
        single_out = single.phi1(ts)
        assert single_out.shape == (4,), f"single[{i}] phi1 shape {single_out.shape}"
        assert numpy.allclose(out[i], single_out), (
            f"orbit {i} phi1 differs between multi and single: {out[i]} vs {single_out}"
        )
        assert numpy.allclose(multi.phi2(ts)[i], single.phi2(ts))
        assert numpy.allclose(multi.pmphi1(ts)[i], single.pmphi1(ts))
        assert numpy.allclose(multi.pmphi2(ts)[i], single.pmphi2(ts))
