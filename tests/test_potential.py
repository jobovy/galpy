############################TESTS ON POTENTIALS################################
import os
import sys

PY3 = sys.version > "3"
import numpy
import pytest
from scipy import optimize

try:
    import pynbody

    _PYNBODY_LOADED = True
except ImportError:
    _PYNBODY_LOADED = False
from galpy import orbit, potential
from galpy.util import _rotate_to_arbitrary_vector, coords


# Test whether the normalization of the potential works
def test_normalize_potential():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "isNonAxi") and tp.isNonAxi:
            continue  # skip, bc vcirc not well defined
        if not hasattr(tp, "normalize"):
            continue
        tp.normalize(1.0)
        assert (tp.Rforce(1.0, 0.0) + 1.0) ** 2.0 < 10.0**-16.0, (
            "Normalization of %s potential fails" % p
        )
        assert (tp.vcirc(1.0) ** 2.0 - 1.0) ** 2.0 < 10.0**-16.0, (
            "Normalization of %s potential fails" % p
        )
        tp.normalize(0.5)
        if hasattr(tp, "toPlanar"):
            ptp = tp.toPlanar()
        else:
            ptp = tp
        assert (ptp.Rforce(1.0, 0.0) + 0.5) ** 2.0 < 10.0**-16.0, (
            "Normalization of %s potential fails" % p
        )
        assert (ptp.vcirc(1.0) ** 2.0 - 0.5) ** 2.0 < 10.0**-16.0, (
            "Normalization of %s potential fails" % p
        )
    # Also test SphericalShell and RingPotential's setup, bc not done elsewhere
    tp = potential.SphericalShellPotential(normalize=1.0)
    assert (tp.Rforce(1.0, 0.0) + 1.0) ** 2.0 < 10.0**-16.0, (
        "Normalization of %s potential fails" % p
    )
    assert (tp.vcirc(1.0) ** 2.0 - 1.0) ** 2.0 < 10.0**-16.0, (
        "Normalization of %s potential fails" % p
    )
    tp = potential.RingPotential(normalize=0.5)
    assert (tp.Rforce(1.0, 0.0) + 0.5) ** 2.0 < 10.0**-16.0, (
        "Normalization of %s potential fails" % p
    )
    assert (tp.vcirc(1.0) ** 2.0 - 0.5) ** 2.0 < 10.0**-16.0, (
        "Normalization of %s potential fails" % p
    )
    return None


# Test whether the derivative of the potential is minus the force
def test_forceAsDeriv_potential():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("testlinearMWPotential")
    pots.append("mockInterpRZPotential")
    if _PYNBODY_LOADED:
        pots.append("mockSnapshotRZPotential")
        pots.append("mockInterpSnapshotRZPotential")
    pots.append("mockCosmphiDiskPotentialnegcp")
    pots.append("mockCosmphiDiskPotentialnegp")
    pots.append("mockDehnenBarPotentialT1")
    pots.append("mockDehnenBarPotentialTm1")
    pots.append("mockDehnenBarPotentialTm1Omega0")
    pots.append("mockDehnenBarPotentialTm5")
    pots.append("mockEllipticalDiskPotentialT1")
    pots.append("mockEllipticalDiskPotentialTm1")
    pots.append("mockEllipticalDiskPotentialTm5")
    pots.append("mockSteadyLogSpiralPotentialT1")
    pots.append("mockSteadyLogSpiralPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm1Omega0")
    pots.append("mockSteadyLogSpiralPotentialTm5")
    pots.append("mockTransientLogSpiralPotential")
    pots.append("mockFlatEllipticalDiskPotential")  # for evaluate w/ nonaxi lists
    pots.append("mockMovingObjectPotential")
    pots.append("mockMovingObjectPotentialExplPlummer")
    pots.append("oblateHernquistPotential")
    pots.append("oblateNFWPotential")
    pots.append("oblatenoGLNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("zRotatedTriaxialNFWPotential")
    pots.append("yRotatedTriaxialNFWPotential")
    pots.append("fullyRotatedTriaxialNFWPotential")
    pots.append("fullyRotatednoGLTriaxialNFWPotential")
    pots.append("HernquistTwoPowerTriaxialPotential")
    pots.append("NFWTwoPowerTriaxialPotential")
    pots.append("JaffeTwoPowerTriaxialPotential")
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("mockAxisymmetricFerrersPotential")
    pots.append("sech2DiskSCFPotential")
    pots.append("expwholeDiskSCFPotential")
    pots.append("nonaxiDiskSCFPotential")
    pots.append("rotatingSpiralArmsPotential")
    pots.append("specialSpiralArmsPotential")
    pots.append("DehnenSmoothDehnenBarPotential")
    pots.append("mockDehnenSmoothBarPotentialT1")
    pots.append("mockDehnenSmoothBarPotentialTm1")
    pots.append("mockDehnenSmoothBarPotentialTm5")
    pots.append("mockDehnenSmoothBarPotentialDecay")
    pots.append("SolidBodyRotationSpiralArmsPotential")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("CorotatingRotationSpiralArmsPotential")
    pots.append("GaussianAmplitudeDehnenBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockTimeDependentAmplitudeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.0, 0.125, -0.125, 0.25, -0.25])
    phis = numpy.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
    )
    # tolerances in log10
    tol = {}
    tol["default"] = -8.0
    tol["DoubleExponentialDiskPotential"] = -6.0  # these are more difficult
    tol["RazorThinExponentialDiskPotential"] = -6.0
    tol["AnyAxisymmetricRazorThinDiskPotential"] = -4.9
    tol["mockInterpRZPotential"] = -4.0
    tol["FerrersPotential"] = -7.0
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        # Set tolerance
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        # Radial force
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                dr = 10.0**-8.0
                newR = Rs[ii] + dr
                dr = newR - Rs[ii]  # Representable number
                if isinstance(tp, potential.linearPotential):
                    mpotderivR = (
                        potential.evaluatelinearPotentials(tp, Rs[ii])
                        - potential.evaluatelinearPotentials(tp, Rs[ii] + dr)
                    ) / dr
                    tRforce = potential.evaluatelinearForces(tp, Rs[ii])
                elif isinstance(tp, potential.planarPotential):
                    mpotderivR = (
                        potential.evaluateplanarPotentials(tp, Rs[ii], phi=Zs[jj])
                        - potential.evaluateplanarPotentials(
                            tp, Rs[ii] + dr, phi=Zs[jj]
                        )
                    ) / dr
                    tRforce = potential.evaluateplanarRforces(tp, Rs[ii], phi=Zs[jj])
                else:
                    mpotderivR = (
                        potential.evaluatePotentials(tp, Rs[ii], Zs[jj], phi=1.0)
                        - potential.evaluatePotentials(tp, Rs[ii] + dr, Zs[jj], phi=1.0)
                    ) / dr
                    tRforce = potential.evaluateRforces(tp, Rs[ii], Zs[jj], phi=1.0)
                if tRforce**2.0 < 10.0**ttol:
                    assert (
                        mpotderivR**2.0 < 10.0**ttol
                    ), f"Calculation of the Radial force as the Radial derivative of the {p} potential fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tRforce-mpotderivR):e}, rel. diff = {numpy.fabs((tRforce-mpotderivR)/tRforce):e}"
                else:
                    assert (
                        (tRforce - mpotderivR) ** 2.0 / tRforce** 2.0 < 10.0** ttol
                    ), f"Calculation of the Radial force as the Radial derivative of the {p} potential fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tRforce-mpotderivR):e}, rel. diff = {numpy.fabs((tRforce-mpotderivR)/tRforce):e}"
        # azimuthal torque, if it exists
        if isinstance(tp, potential.linearPotential):
            continue
        for ii in range(len(Rs)):
            for jj in range(len(phis)):
                dphi = 10.0**-8.0
                newphi = phis[jj] + dphi
                dphi = newphi - phis[jj]  # Representable number
                if isinstance(tp, potential.planarPotential):
                    mpotderivphi = (
                        tp(Rs[ii], phi=phis[jj]) - tp(Rs[ii], phi=phis[jj] + dphi)
                    ) / dphi
                    tphitorque = potential.evaluateplanarphitorques(
                        tp, Rs[ii], phi=phis[jj]
                    )
                else:
                    mpotderivphi = (
                        tp(Rs[ii], 0.05, phi=phis[jj])
                        - tp(Rs[ii], 0.05, phi=phis[jj] + dphi)
                    ) / dphi
                    tphitorque = potential.evaluatephitorques(
                        tp, Rs[ii], 0.05, phi=phis[jj]
                    )
                try:
                    if tphitorque**2.0 < 10.0**ttol:
                        assert mpotderivphi**2.0 < 10.0**ttol
                    else:
                        assert (
                            tphitorque - mpotderivphi
                        ) ** 2.0 / tphitorque**2.0 < 10.0**ttol
                except AssertionError:
                    if isinstance(tp, potential.planarPotential):
                        raise AssertionError(
                            f"Calculation of the azimuthal torque as the azimuthal derivative of the {p} potential fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tphitorque-mpotderivphi):e}, rel. diff = {numpy.fabs((tphitorque-mpotderivphi)/tphitorque):e}"
                        )
                    else:
                        raise AssertionError(
                            f"Calculation of the azimuthal torque as the azimuthal derivative of the {p} potential fails at (R,Z,phi) = ({Rs[ii]:.3f},0.05,{phis[jj]:.3f}); diff = {numpy.fabs(tphitorque-mpotderivphi):e}, rel. diff = {numpy.fabs((tphitorque-mpotderivphi)/tphitorque):e}"
                        )
        # Vertical force, if it exists
        if isinstance(tp, potential.planarPotential) or isinstance(
            tp, potential.linearPotential
        ):
            continue

        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                ##Excluding KuzminDiskPotential when z = 0
                if Zs[jj] == 0 and isinstance(tp, potential.KuzminDiskPotential):
                    continue
                dz = 10.0**-8.0
                newZ = Zs[jj] + dz
                dz = newZ - Zs[jj]  # Representable number
                mpotderivz = (
                    tp(Rs[ii], Zs[jj], phi=1.0) - tp(Rs[ii], Zs[jj] + dz, phi=1.0)
                ) / dz
                tzforce = potential.evaluatezforces(tp, Rs[ii], Zs[jj], phi=1.0)
                if tzforce**2.0 < 10.0**ttol:
                    assert (
                        mpotderivz**2.0 < 10.0**ttol
                    ), f"Calculation of the vertical force as the vertical derivative of the {p} potential fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(mpotderivz):e}, rel. diff = {numpy.fabs((tzforce-mpotderivz)/tzforce):e}"
                else:
                    assert (
                        (tzforce - mpotderivz) ** 2.0 / tzforce** 2.0 < 10.0** ttol
                    ), f"Calculation of the vertical force as the vertical derivative of the {p} potential fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(mpotderivz):e}, rel. diff = {numpy.fabs((tzforce-mpotderivz)/tzforce):e}"


# Test whether the second derivative of the potential is minus the derivative of the force
def test_2ndDeriv_potential():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("testlinearMWPotential")
    pots.append("mockInterpRZPotential")
    pots.append("mockCosmphiDiskPotentialnegcp")
    pots.append("mockCosmphiDiskPotentialnegp")
    pots.append("mockDehnenBarPotentialT1")
    pots.append("mockDehnenBarPotentialTm1")
    pots.append("mockDehnenBarPotentialTm1Omega0")
    pots.append("mockDehnenBarPotentialTm5")
    pots.append("mockEllipticalDiskPotentialT1")
    pots.append("mockEllipticalDiskPotentialTm1")
    pots.append("mockEllipticalDiskPotentialTm5")
    pots.append("mockSteadyLogSpiralPotentialT1")
    pots.append("mockSteadyLogSpiralPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm1Omega0")
    pots.append("mockSteadyLogSpiralPotentialTm5")
    pots.append("mockTransientLogSpiralPotential")
    pots.append("mockFlatEllipticalDiskPotential")  # for evaluate w/ nonaxi lists
    pots.append("oblateHernquistPotential")  # in case these are ever implemented
    pots.append("oblateNFWPotential")
    pots.append("oblatenoGLNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("HernquistTwoPowerTriaxialPotential")
    pots.append("NFWTwoPowerTriaxialPotential")
    pots.append("JaffeTwoPowerTriaxialPotential")
    pots.append("mockAxisymmetricFerrersPotential")
    pots.append("rotatingSpiralArmsPotential")
    pots.append("specialSpiralArmsPotential")
    pots.append("DehnenSmoothDehnenBarPotential")
    pots.append("mockDehnenSmoothBarPotentialT1")
    pots.append("mockDehnenSmoothBarPotentialTm1")
    pots.append("mockDehnenSmoothBarPotentialTm5")
    pots.append("mockDehnenSmoothBarPotentialDecay")
    pots.append("SolidBodyRotationSpiralArmsPotential")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("CorotatingRotationSpiralArmsPotential")
    pots.append("GaussianAmplitudeDehnenBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockTimeDependentAmplitudeWrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.0, 0.125, -0.125, 0.25, -0.25])
    phis = numpy.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
    )
    # tolerances in log10
    tol = {}
    tol["default"] = -8.0
    tol["DoubleExponentialDiskPotential"] = -3.0  # these are more difficult
    tol["RazorThinExponentialDiskPotential"] = -6.0
    tol["AnyAxisymmetricRazorThinDiskPotential"] = -4.5
    tol["mockInterpRZPotential"] = -4.0
    tol["DehnenBarPotential"] = -7.0
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        # Set tolerance
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        # 2nd radial
        if hasattr(tp, "_R2deriv"):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
                    if (
                        p == "RazorThinExponentialDiskPotential"
                        and numpy.fabs(Zs[jj]) > 0.0
                    ):
                        continue  # Not implemented
                    dr = 10.0**-8.0
                    newR = Rs[ii] + dr
                    dr = newR - Rs[ii]  # Representable number
                    if isinstance(tp, potential.linearPotential):
                        mRforcederivR = (
                            tp.Rforce(Rs[ii]) - tp.Rforce(Rs[ii] + dr)
                        ) / dr
                        tR2deriv = tp.R2deriv(Rs[ii])
                    elif isinstance(tp, potential.planarPotential):
                        mRforcederivR = (
                            tp.Rforce(Rs[ii], Zs[jj]) - tp.Rforce(Rs[ii] + dr, Zs[jj])
                        ) / dr
                        tR2deriv = potential.evaluateplanarR2derivs(
                            tp, Rs[ii], phi=Zs[jj]
                        )
                    else:
                        mRforcederivR = (
                            tp.Rforce(Rs[ii], Zs[jj], phi=1.0)
                            - tp.Rforce(Rs[ii] + dr, Zs[jj], phi=1.0)
                        ) / dr
                        tR2deriv = potential.evaluateR2derivs(
                            tp, Rs[ii], Zs[jj], phi=1.0
                        )
                    if tR2deriv**2.0 < 10.0**ttol:
                        assert (
                            mRforcederivR**2.0 < 10.0**ttol
                        ), f"Calculation of the second Radial derivative of the potential as the Radial derivative of the {p} Radial force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tR2deriv-mRforcederivR):e}, rel. diff = {numpy.fabs((tR2deriv-mRforcederivR)/tR2deriv):e}"
                    else:
                        assert (
                            (tR2deriv - mRforcederivR) ** 2.0 / tR2deriv** 2.0
                            < 10.0** ttol
                        ), f"Calculation of the second Radial derivative of the potential as the Radial derivative of the {p} Radial force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tR2deriv-mRforcederivR):e}, rel. diff = {numpy.fabs((tR2deriv-mRforcederivR)/tR2deriv):e}"
        # 2nd azimuthal
        if not isinstance(tp, potential.linearPotential) and hasattr(tp, "_phi2deriv"):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    dphi = 10.0**-8.0
                    newphi = phis[jj] + dphi
                    dphi = newphi - phis[jj]  # Representable number
                    if isinstance(tp, potential.planarPotential):
                        mphitorquederivphi = (
                            tp.phitorque(Rs[ii], phi=phis[jj])
                            - tp.phitorque(Rs[ii], phi=phis[jj] + dphi)
                        ) / dphi
                        tphi2deriv = tp.phi2deriv(Rs[ii], phi=phis[jj])
                    else:
                        mphitorquederivphi = (
                            tp.phitorque(Rs[ii], 0.05, phi=phis[jj])
                            - tp.phitorque(Rs[ii], 0.05, phi=phis[jj] + dphi)
                        ) / dphi
                        tphi2deriv = potential.evaluatephi2derivs(
                            tp, Rs[ii], 0.05, phi=phis[jj]
                        )
                    try:
                        if tphi2deriv**2.0 < 10.0**ttol:
                            assert mphitorquederivphi**2.0 < 10.0**ttol
                        else:
                            assert (
                                tphi2deriv - mphitorquederivphi
                            ) ** 2.0 / tphi2deriv**2.0 < 10.0**ttol
                    except AssertionError:
                        if isinstance(tp, potential.planarPotential):
                            raise AssertionError(
                                f"Calculation of the second azimuthal derivative of the potential as the azimuthal derivative of the {p} azimuthal torque fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tphi2deriv-mphitorquederivphi):e}, rel. diff = {numpy.fabs((tphi2deriv-mphitorquederivphi)/tphi2deriv):e}"
                            )
                        else:
                            raise AssertionError(
                                f"Calculation of the second azimuthal derivative of the potential as the azimuthal derivative of the {p} azimuthal torque fails at (R,Z,phi) = ({Rs[ii]:.3f},0.05,{phis[jj]:.3f}); diff = {numpy.fabs(tphi2deriv-mphitorquederivphi):e}, rel. diff = {numpy.fabs((tphi2deriv-mphitorquederivphi)/tphi2deriv):e}"
                            )
        # mixed radial azimuthal: Isn't this the same as what's below??
        if not isinstance(tp, potential.linearPotential) and hasattr(tp, "_Rphideriv"):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    dphi = 10.0**-8.0
                    newphi = phis[jj] + dphi
                    dphi = newphi - phis[jj]  # Representable number
                    if isinstance(tp, potential.planarPotential):
                        mRforcederivphi = (
                            tp.Rforce(Rs[ii], phi=phis[jj])
                            - tp.Rforce(Rs[ii], phi=phis[jj] + dphi)
                        ) / dphi
                        tRphideriv = tp.Rphideriv(Rs[ii], phi=phis[jj])
                    else:
                        mRforcederivphi = (
                            tp.Rforce(Rs[ii], 0.05, phi=phis[jj])
                            - tp.Rforce(Rs[ii], 0.05, phi=phis[jj] + dphi)
                        ) / dphi
                        tRphideriv = potential.evaluateRphiderivs(
                            tp, Rs[ii], 0.05, phi=phis[jj]
                        )
                    try:
                        if tRphideriv**2.0 < 10.0**ttol:
                            assert mRforcederivphi**2.0 < 10.0**ttol
                        else:
                            assert (
                                tRphideriv - mRforcederivphi
                            ) ** 2.0 / tRphideriv**2.0 < 10.0**ttol
                    except AssertionError:
                        if isinstance(tp, potential.planarPotential):
                            raise AssertionError(
                                f"Calculation of the mixed radial, azimuthal derivative of the potential as the azimuthal derivative of the {p} Radial force fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tRphideriv-mRforcederivphi):e}, rel. diff = {numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv):e}"
                            )
                        else:
                            raise AssertionError(
                                f"Calculation of the mixed radial, azimuthal derivative of the potential as the azimuthal derivative of the {p} azimuthal torque fails at (R,Z,phi) = ({Rs[ii]:.3f},0.05,{phis[jj]:.3f}); diff = {numpy.fabs(tRphideriv-mRforcederivphi):e}, rel. diff = {numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv):e}"
                            )
        # 2nd vertical
        if (
            not isinstance(tp, potential.planarPotential)
            and not isinstance(tp, potential.linearPotential)
            and hasattr(tp, "_z2deriv")
        ):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
                    if p == "RazorThinExponentialDiskPotential":
                        continue  # Not implemented, or badly defined
                    if p == "TwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "specialTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "DehnenTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "DehnenCoreTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "HernquistTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "JaffeTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    if p == "NFWTwoPowerSphericalPotential":
                        continue  # Not implemented, or badly defined
                    # Excluding KuzminDiskPotential at z = 0
                    if p == "KuzminDiskPotential" and Zs[jj] == 0:
                        continue
                    dz = 10.0**-8.0
                    newz = Zs[jj] + dz
                    dz = newz - Zs[jj]  # Representable number
                    mzforcederivz = (
                        tp.zforce(Rs[ii], Zs[jj], phi=1.0)
                        - tp.zforce(Rs[ii], Zs[jj] + dz, phi=1.0)
                    ) / dz
                    tz2deriv = potential.evaluatez2derivs(tp, Rs[ii], Zs[jj], phi=1.0)
                    if tz2deriv**2.0 < 10.0**ttol:
                        assert (
                            mzforcederivz**2.0 < 10.0**ttol
                        ), f"Calculation of the second vertical derivative of the potential as the vertical derivative of the {p} vertical force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tz2deriv-mzforcederivz):e}, rel. diff = {numpy.fabs((tz2deriv-mzforcederivz)/tz2deriv):e}"
                    else:
                        assert (
                            (tz2deriv - mzforcederivz) ** 2.0 / tz2deriv** 2.0
                            < 10.0** ttol
                        ), f"Calculation of the second vertical derivative of the potential as the vertical derivative of the {p} vertical force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tz2deriv-mzforcederivz):e}, rel. diff = {numpy.fabs((tz2deriv-mzforcederivz)/tz2deriv):e}"
        # mixed radial vertical
        if (
            not isinstance(tp, potential.planarPotential)
            and not isinstance(tp, potential.linearPotential)
            and hasattr(tp, "_Rzderiv")
        ):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
                    # Excluding KuzminDiskPotential at z = 0
                    if p == "KuzminDiskPotential" and Zs[jj] == 0:
                        continue
                    #                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    dz = 10.0**-8.0
                    newz = Zs[jj] + dz
                    dz = newz - Zs[jj]  # Representable number
                    mRforcederivz = (
                        tp.Rforce(Rs[ii], Zs[jj], phi=1.0)
                        - tp.Rforce(Rs[ii], Zs[jj] + dz, phi=1.0)
                    ) / dz
                    tRzderiv = potential.evaluateRzderivs(tp, Rs[ii], Zs[jj], phi=1.0)
                    if tRzderiv**2.0 < 10.0**ttol:
                        assert (
                            mRforcederivz**2.0 < 10.0**ttol
                        ), f"Calculation of the mixed radial vertical derivative of the potential as the vertical derivative of the {p} radial force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tRzderiv-mRforcederivz):e}, rel. diff = {numpy.fabs((tRzderiv-mRforcederivz)/tRzderiv):e}"
                    else:
                        assert (
                            (tRzderiv - mRforcederivz) ** 2.0 / tRzderiv** 2.0
                            < 10.0** ttol
                        ), f"Calculation of the mixed radial vertical derivative of the potential as the vertical derivative of the {p} radial force fails at (R,Z) = ({Rs[ii]:.3f},{Zs[jj]:.3f}); diff = {numpy.fabs(tRzderiv-mRforcederivz):e}, rel. diff = {numpy.fabs((tRzderiv-mRforcederivz)/tRzderiv):e}"
        # mixed radial, azimuthal
        if not isinstance(tp, potential.linearPotential) and hasattr(tp, "_Rphideriv"):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    #                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    dphi = 10.0**-8.0
                    newphi = phis[jj] + dphi
                    dphi = newphi - phis[jj]  # Representable number
                    if isinstance(tp, potential.planarPotential):
                        mRforcederivphi = (
                            tp.Rforce(Rs[ii], phi=phis[jj])
                            - tp.Rforce(Rs[ii], phi=phis[jj] + dphi)
                        ) / dphi
                        tRphideriv = potential.evaluateplanarPotentials(
                            tp, Rs[ii], phi=phis[jj], dR=1, dphi=1
                        )
                    else:
                        mRforcederivphi = (
                            tp.Rforce(Rs[ii], 0.1, phi=phis[jj])
                            - tp.Rforce(Rs[ii], 0.1, phi=phis[jj] + dphi)
                        ) / dphi
                        tRphideriv = potential.evaluatePotentials(
                            tp, Rs[ii], 0.1, phi=phis[jj], dR=1, dphi=1
                        )
                    if tRphideriv**2.0 < 10.0**ttol:
                        assert (
                            mRforcederivphi**2.0 < 10.0**ttol
                        ), f"Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the {p} radial force fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tRphideriv-mRforcederivphi):e}, rel. diff = {numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv):e}"
                    else:
                        assert (
                            (tRphideriv - mRforcederivphi) ** 2.0 / tRphideriv** 2.0
                            < 10.0** ttol
                        ), f"Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the {p} radial force fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tRphideriv-mRforcederivphi):e}, rel. diff = {numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv):e}"
        # mixed azimuthal, vertical
        if (
            not isinstance(tp, potential.planarPotential)
            and not isinstance(tp, potential.linearPotential)
            and hasattr(tp, "_phizderiv")
        ):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    #                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    dphi = 10.0**-8.0
                    newphi = phis[jj] + dphi
                    dphi = newphi - phis[jj]  # Representable number
                    mzforcederivphi = (
                        tp.zforce(Rs[ii], 0.1, phi=phis[jj])
                        - tp.zforce(Rs[ii], 0.1, phi=phis[jj] + dphi)
                    ) / dphi
                    tphizderiv = potential.evaluatephizderivs(
                        tp, Rs[ii], 0.1, phi=phis[jj]
                    )
                    if tphizderiv**2.0 < 10.0**ttol:
                        assert (
                            mzforcederivphi**2.0 < 10.0**ttol
                        ), f"Calculation of the mixed azimuthal vertical derivative of the potential as the azimuthal derivative of the {p} vertical force fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tphizderiv-mzforcederivphi):e}, rel. diff = {numpy.fabs((tphizderiv-mzforcederivphi)/tphizderiv):e}"
                    else:
                        assert (
                            (tphizderiv - mzforcederivphi) ** 2.0 / tphizderiv** 2.0
                            < 10.0** ttol
                        ), f"Calculation of the mixed azimuthal vertical derivative of the potential as the azimuthal derivative of the {p} vertical force fails at (R,phi) = ({Rs[ii]:.3f},{phis[jj]:.3f}); diff = {numpy.fabs(tphizderiv-mzforcederivphi):e}, rel. diff = {numpy.fabs((tphizderiv-mzforcederivphi)/tphizderiv):e}"


# Test whether the Poisson equation is satisfied if _dens and the relevant second derivatives are implemented
def test_poisson_potential():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialFlattenedPowerPotential")
    pots.append("specialPowerSphericalPotential")
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("testlinearMWPotential")
    pots.append("oblateHernquistPotential")  # in cae these are ever implemented
    pots.append("oblateNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("HernquistTwoPowerTriaxialPotential")
    pots.append("NFWTwoPowerTriaxialPotential")
    pots.append("JaffeTwoPowerTriaxialPotential")
    pots.append("rotatingSpiralArmsPotential")
    pots.append("specialSpiralArmsPotential")
    pots.append("DehnenSmoothDehnenBarPotential")
    pots.append("SolidBodyRotationSpiralArmsPotential")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("CorotatingRotationSpiralArmsPotential")
    pots.append("GaussianAmplitudeDehnenBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockTimeDependentAmplitudeWrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.0, 0.125, -0.125, 0.25, -0.25])
    phis = numpy.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
    )
    # tolerances in log10
    tol = {}
    tol["default"] = -8.0
    tol["DoubleExponentialDiskPotential"] = -3.0  # these are more difficult
    tol["SpiralArmsPotential"] = -3  # these are more difficult
    tol["rotatingSpiralArmsPotential"] = -3
    tol["specialSpiralArmsPotential"] = -4
    tol["SolidBodyRotationSpiralArmsPotential"] = -2.9  # these are more difficult
    tol["nestedListPotential"] = -3  # these are more difficult
    # tol['RazorThinExponentialDiskPotential']= -6.
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # if 'Isochrone' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        # Set tolerance
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        # 2nd radial
        if (
            not hasattr(tp, "_dens")
            or not hasattr(tp, "_R2deriv")
            or not hasattr(tp, "_Rforce")
            or not hasattr(tp, "phi2deriv")
            or not hasattr(tp, "_z2deriv")
        ):
            continue
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                for kk in range(len(phis)):
                    tpoissondens = tp.dens(
                        Rs[ii], Zs[jj], phi=phis[kk], forcepoisson=True
                    )
                    tdens = potential.evaluateDensities(
                        tp, Rs[ii], Zs[jj], phi=phis[kk], forcepoisson=False
                    )
                    if tdens**2.0 < 10.0**ttol:
                        assert (
                            tpoissondens**2.0 < 10.0**ttol
                        ), f"Poisson equation relation between the derivatives of the potential and the implemented density is not satisfied for the {p} potential at (R,Z,phi) = ({Rs[ii]:.3f},{Zs[jj]:.3f},{phis[kk]:.3f}); diff = {numpy.fabs(tdens-tpoissondens):e}, rel. diff = {numpy.fabs((tdens-tpoissondens)/tdens):e}"
                    else:
                        assert (
                            (tpoissondens - tdens) ** 2.0 / tdens** 2.0 < 10.0** ttol
                        ), f"Poisson equation relation between the derivatives of the potential and the implemented density is not satisfied for the {p} potential at (R,Z,phi) = ({Rs[ii]:.3f},{Zs[jj]:.3f},{phis[kk]:.3f}); diff = {numpy.fabs(tdens-tpoissondens):e}, rel. diff = {numpy.fabs((tdens-tpoissondens)/tdens):e}"
    return None


# Test whether the (integrated) Poisson equation is satisfied if _surfdens and the relevant second derivatives are implemented
def test_poisson_surfdens_potential():
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
    """
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('DehnenTwoPowerSphericalPotential')
    pots.append('DehnenCoreTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerSphericalPotential')
    pots.append('JaffeTwoPowerSphericalPotential')
    pots.append('NFWTwoPowerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialMN3ExponentialDiskPotentialPD')
    pots.append('specialMN3ExponentialDiskPotentialSECH')
    pots.append('specialFlattenedPowerPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('testplanarMWPotential')
    pots.append('testlinearMWPotential')
    pots.append('oblateHernquistPotential') # in cae these are ever implemented
    pots.append('oblateNFWPotential')
    pots.append('oblateJaffePotential')
    pots.append('prolateHernquistPotential')
    pots.append('prolateNFWPotential')
    pots.append('prolateJaffePotential')
    pots.append('triaxialHernquistPotential')
    pots.append('triaxialNFWPotential')
    pots.append('triaxialJaffePotential')
    pots.append('HernquistTwoPowerTriaxialPotential')
    pots.append('NFWTwoPowerTriaxialPotential')
    pots.append('JaffeTwoPowerTriaxialPotential')
    pots.append('rotatingSpiralArmsPotential')
    pots.append('specialSpiralArmsPotential')
    pots.append('DehnenSmoothDehnenBarPotential')
    pots.append('SolidBodyRotationSpiralArmsPotential')
    pots.append('triaxialLogarithmicHaloPotential')
    pots.append('CorotatingRotationSpiralArmsPotential')
    pots.append('GaussianAmplitudeDehnenBarPotential')
    pots.append('nestedListPotential')
    """
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    rmpots.append(
        "RazorThinExponentialDiskPotential"
    )  # R2deriv not implemented for |Z| > 0
    for p in rmpots:
        pots.remove(p)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.125, 0.25, 1.0, 10.0])
    phis = numpy.array(
        [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
    )
    # tolerances in log10
    tol = {}
    tol["default"] = -8.0
    tol["DoubleExponentialDiskPotential"] = -3.0  # these are more difficult
    tol[
        "SphericalShellPotential"
    ] = -0  # Direct integration fails to deal with delta function!
    # tol['SpiralArmsPotential']= -3 #these are more difficult
    # tol['rotatingSpiralArmsPotential']= -3
    # tol['specialSpiralArmsPotential']= -4
    # tol['SolidBodyRotationSpiralArmsPotential']= -2.9 #these are more difficult
    # tol['nestedListPotential']= -3 #these are more difficult
    # tol['RazorThinExponentialDiskPotential']= -6.
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # if 'Isochrone' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        # Set tolerance
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        # 2nd radial
        if (
            not hasattr(tp, "_surfdens")
            or not hasattr(tp, "_R2deriv")
            or not hasattr(tp, "_Rforce")
            or not hasattr(tp, "phi2deriv")
            or not hasattr(tp, "_zforce")
            or (
                tclass._surfdens == potential.Potential._surfdens
                and not p == "FlattenedPowerPotential"
            )
        ):  # make sure _surfdens is explicitly implemented
            continue
        for ii in range(len(Rs)):
            for kk in range(len(phis)):
                for jj in range(len(Zs)):
                    tpoissondens = tp.surfdens(
                        Rs[ii], Zs[jj], phi=phis[kk], forcepoisson=True
                    )
                    tdens = potential.evaluateSurfaceDensities(
                        tp, Rs[ii], Zs[jj], phi=phis[kk], forcepoisson=False
                    )
                    if tdens**2.0 < 10.0**ttol:
                        assert (
                            tpoissondens**2.0 < 10.0**ttol
                        ), f"Poisson equation relation between the derivatives of the potential and the implemented surface density is not satisfied for the {p} potential at (R,Z,phi) = ({Rs[ii]:.3f},{Zs[jj]:.3f},{phis[kk]:.3f}); diff = {numpy.fabs(tdens-tpoissondens):e}, rel. diff = {numpy.fabs((tdens-tpoissondens)/tdens):e}"
                    else:
                        assert (
                            (tpoissondens - tdens) ** 2.0 / tdens** 2.0 < 10.0** ttol
                        ), f"Poisson equation relation between the derivatives of the potential and the implemented surface density is not satisfied for the {p} potential at (R,Z,phi) = ({Rs[ii]:.3f},{Zs[jj]:.3f},{phis[kk]:.3f}); diff = {numpy.fabs(tdens-tpoissondens):e}, rel. diff = {numpy.fabs((tdens-tpoissondens)/tdens):e}"
                if p == "mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination":
                    continue  # takes a long time otherwise... skip after all z at one (R,phi)
    return None


# Test whether the _evaluate function is correctly implemented in specifying derivatives
def test_evaluateAndDerivs_potential():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialFlattenedPowerPotential")
    pots.append("specialPowerSphericalPotential")
    pots.append("mockCosmphiDiskPotentialnegcp")
    pots.append("mockCosmphiDiskPotentialnegp")
    pots.append("mockDehnenBarPotentialT1")
    pots.append("mockDehnenBarPotentialTm1")
    pots.append("mockDehnenBarPotentialTm1Omega0")
    pots.append("mockDehnenBarPotentialTm5")
    pots.append("mockEllipticalDiskPotentialT1")
    pots.append("mockEllipticalDiskPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm1Omega0")
    pots.append("mockEllipticalDiskPotentialTm5")
    pots.append("mockSteadyLogSpiralPotentialT1")
    pots.append("mockSteadyLogSpiralPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm5")
    pots.append("mockTransientLogSpiralPotential")
    pots.append("mockMovingObjectPotential")
    pots.append("oblateHernquistPotential")  # in cae these are ever implemented
    pots.append("oblateNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("sech2DiskSCFPotential")
    pots.append("expwholeDiskSCFPotential")
    pots.append("nonaxiDiskSCFPotential")
    pots.append("rotatingSpiralArmsPotential")
    pots.append("specialSpiralArmsPotential")
    pots.append("SolidBodyRotationSpiralArmsPotential")
    pots.append("DehnenSmoothDehnenBarPotential")
    pots.append("mockDehnenSmoothBarPotentialT1")
    pots.append("mockDehnenSmoothBarPotentialTm1")
    pots.append("mockDehnenSmoothBarPotentialTm5")
    pots.append("mockDehnenSmoothBarPotentialDecay")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("CorotatingRotationSpiralArmsPotential")
    pots.append("GaussianAmplitudeDehnenBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockTimeDependentAmplitudeWrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    # tolerances in log10
    tol = {}
    tol["default"] = -12.0
    # tol['DoubleExponentialDiskPotential']= -3. #these are more difficult
    # tol['RazorThinExponentialDiskPotential']= -6.
    for p in pots:
        # if 'Isochrone' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        # Set tolerance
        if p in list(tol.keys()):
            ttol = tol[p]
        else:
            ttol = tol["default"]
        # 1st radial
        if isinstance(tp, potential.linearPotential):
            continue
        elif isinstance(tp, potential.planarPotential):
            tevaldr = tp(1.2, phi=0.1, dR=1)
            trforce = tp.Rforce(1.2, phi=0.1)
        else:
            tevaldr = tp(1.2, 0.1, phi=0.1, dR=1)
            trforce = tp.Rforce(1.2, 0.1, phi=0.1)
        if not tevaldr is None:
            if tevaldr**2.0 < 10.0**ttol:
                assert trforce**2.0 < 10.0**ttol, (
                    "Calculation of radial derivative through _evaluate and Rforce inconsistent for the %s potential"
                    % p
                )
            else:
                assert (tevaldr + trforce) ** 2.0 / tevaldr**2.0 < 10.0**ttol, (
                    "Calculation of radial derivative through _evaluate and Rforce inconsistent for the %s potential"
                    % p
                )
        # 2nd radial
        hasR2 = True
        from galpy.potential import PotentialError

        if "RazorThin" in p:
            R2z = 0.0
        else:
            R2z = 0.1
        try:
            if isinstance(tp, potential.planarPotential):
                tp.R2deriv(1.2)
            else:
                tp.R2deriv(1.2, R2z)
        except PotentialError:
            hasR2 = False
        if hasR2:
            if isinstance(tp, potential.planarPotential):
                tevaldr2 = tp(1.2, phi=0.1, dR=2)
                tr2deriv = tp.R2deriv(1.2, phi=0.1)
            else:
                tevaldr2 = tp(1.2, R2z, phi=0.1, dR=2)
                tr2deriv = tp.R2deriv(1.2, R2z, phi=0.1)
            if not tevaldr2 is None:
                if tevaldr2**2.0 < 10.0**ttol:
                    assert tr2deriv * 2.0 < 10.0**ttol, (
                        "Calculation of 2nd radial derivative through _evaluate and R2deriv inconsistent for the %s potential"
                        % p
                    )
                else:
                    assert (tevaldr2 - tr2deriv) ** 2.0 / tevaldr2**2.0 < 10.0**ttol, (
                        "Calculation of 2nd radial derivative through _evaluate and R2deriv inconsistent for the %s potential"
                        % p
                    )
        # 1st phi
        if isinstance(tp, potential.planarPotential):
            tevaldphi = tp(1.2, phi=0.1, dphi=1)
            tphitorque = tp.phitorque(1.2, phi=0.1)
        else:
            tevaldphi = tp(1.2, 0.1, phi=0.1, dphi=1)
            tphitorque = tp.phitorque(1.2, 0.1, phi=0.1)
        if not tevaldphi is None:
            if tevaldphi**2.0 < 10.0**ttol:
                assert tphitorque**2.0 < 10.0**ttol, (
                    "Calculation of azimuthal derivative through _evaluate and phitorque inconsistent for the %s potential"
                    % p
                )
            else:
                assert (tevaldphi + tphitorque) ** 2.0 / tevaldphi**2.0 < 10.0**ttol, (
                    "Calculation of azimuthal derivative through _evaluate and phitorque inconsistent for the %s potential"
                    % p
                )
        # 2nd phi
        hasphi2 = True
        try:
            if isinstance(tp, potential.planarPotential):
                tp.phi2deriv(1.2, phi=0.1)
            else:
                tp.phi2deriv(1.2, 0.1, phi=0.1)
        except (PotentialError, AttributeError):
            hasphi2 = False
        if hasphi2 and hasattr(tp, "_phi2deriv"):
            if isinstance(tp, potential.planarPotential):
                tevaldphi2 = tp(1.2, phi=0.1, dphi=2)
                tphi2deriv = tp.phi2deriv(1.2, phi=0.1)
            else:
                tevaldphi2 = tp(1.2, 0.1, phi=0.1, dphi=2)
                tphi2deriv = tp.phi2deriv(1.2, 0.1, phi=0.1)
            if not tevaldphi2 is None:
                if tevaldphi2**2.0 < 10.0**ttol:
                    assert tphi2deriv * 2.0 < 10.0**ttol, (
                        "Calculation of 2nd azimuthal derivative through _evaluate and phi2deriv inconsistent for the %s potential"
                        % p
                    )
                else:
                    assert (
                        tevaldphi2 - tphi2deriv
                    ) ** 2.0 / tevaldphi2**2.0 < 10.0**ttol, (
                        "Calculation of 2nd azimuthal derivative through _evaluate and phi2deriv inconsistent for the %s potential"
                        % p
                    )
        # Test that much higher derivatives are not implemented
        try:
            tp(1.2, 0.1, dR=4, dphi=10)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(
                "Higher-order derivative request in potential __call__ does not raise NotImplementedError for %s"
                % p
            )
        continue
        # mixed radial,vertical
        if isinstance(tp, potential.planarPotential):
            tevaldrz = tp(1.2, 0.1, phi=0.1, dR=1, dz=1)
            trzderiv = tp.Rzderiv(1.2, 0.1, phi=0.1)
        else:
            tevaldrz = tp(1.2, 0.1, phi=0.1, dR=1, dz=1)
            trzderiv = tp.Rzderiv(1.2, 0.1, phi=0.1)
        if not tevaldrz is None:
            if tevaldrz**2.0 < 10.0**ttol:
                assert trzderiv * 2.0 < 10.0**ttol, (
                    "Calculation of mixed radial,vertical derivative through _evaluate and z2deriv inconsistent for the %s potential"
                    % p
                )
            else:
                assert (tevaldrz - trzderiv) ** 2.0 / tevaldrz**2.0 < 10.0**ttol, (
                    "Calculation of mixed radial,vertical derivative through _evaluate and z2deriv inconsistent for the %s potential"
                    % p
                )
    return None


# Test that potentials can be multiplied or divided by a number
def test_amp_mult_divide():
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
    pots.append("specialTwoPowerSphericalPotential")
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    pots.append("NFWTwoPowerSphericalPotential")
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("testMWPotential")
    pots.append("testplanarMWPotential")
    pots.append("testlinearMWPotential")
    pots.append("mockInterpRZPotential")
    if _PYNBODY_LOADED:
        pots.append("mockSnapshotRZPotential")
        pots.append("mockInterpSnapshotRZPotential")
    pots.append("mockCosmphiDiskPotentialnegcp")
    pots.append("mockCosmphiDiskPotentialnegp")
    pots.append("mockDehnenBarPotentialT1")
    pots.append("mockDehnenBarPotentialTm1")
    pots.append("mockDehnenBarPotentialTm1Omega0")
    pots.append("mockDehnenBarPotentialTm5")
    pots.append("mockEllipticalDiskPotentialT1")
    pots.append("mockEllipticalDiskPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm1Omega0")
    pots.append("mockEllipticalDiskPotentialTm5")
    pots.append("mockSteadyLogSpiralPotentialT1")
    pots.append("mockSteadyLogSpiralPotentialTm1")
    pots.append("mockSteadyLogSpiralPotentialTm5")
    pots.append("mockTransientLogSpiralPotential")
    pots.append("mockFlatEllipticalDiskPotential")  # for evaluate w/ nonaxi lists
    pots.append("mockMovingObjectPotential")
    pots.append("mockMovingObjectPotentialExplPlummer")
    pots.append("oblateHernquistPotential")
    pots.append("oblateNFWPotential")
    pots.append("oblatenoGLNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("zRotatedTriaxialNFWPotential")
    pots.append("yRotatedTriaxialNFWPotential")
    pots.append("fullyRotatedTriaxialNFWPotential")
    pots.append("fullyRotatednoGLTriaxialNFWPotential")
    pots.append("HernquistTwoPowerTriaxialPotential")
    pots.append("NFWTwoPowerTriaxialPotential")
    pots.append("JaffeTwoPowerTriaxialPotential")
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("mockAxisymmetricFerrersPotential")
    pots.append("sech2DiskSCFPotential")
    pots.append("expwholeDiskSCFPotential")
    pots.append("nonaxiDiskSCFPotential")
    pots.append("rotatingSpiralArmsPotential")
    pots.append("specialSpiralArmsPotential")
    pots.append("DehnenSmoothDehnenBarPotential")
    pots.append("mockDehnenSmoothBarPotentialT1")
    pots.append("mockDehnenSmoothBarPotentialTm1")
    pots.append("mockDehnenSmoothBarPotentialTm5")
    pots.append("mockDehnenSmoothBarPotentialDecay")
    pots.append("SolidBodyRotationSpiralArmsPotential")
    pots.append("triaxialLogarithmicHaloPotential")
    pots.append("CorotatingRotationSpiralArmsPotential")
    pots.append("GaussianAmplitudeDehnenBarPotential")
    pots.append("nestedListPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
    R, Z, phi = 0.75, 0.2, 1.76
    nums = numpy.random.uniform(size=len(pots))  # random set of amp changes
    for num, p in zip(nums, pots):
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        if isinstance(tp, potential.linearPotential):
            assert (
                numpy.fabs(tp(R) * num - (num * tp)(R)) < 1e-10
            ), "Multiplying a linearPotential with a number does not behave as expected"
            # Other way...
            assert (
                numpy.fabs(tp(R) * num - (tp * num)(R)) < 1e-10
            ), "Multiplying a linearPotential with a number does not behave as expected"
            assert (
                numpy.fabs(tp(R) / num - (tp / num)(R)) < 1e-10
            ), "Dividing a linearPotential with a number does not behave as expected"
        elif isinstance(tp, potential.planarPotential):
            assert (
                numpy.fabs(tp(R, phi=phi) * num - (num * tp)(R, phi=phi)) < 1e-10
            ), "Multiplying a planarPotential with a number does not behave as expected"
            # Other way...
            assert (
                numpy.fabs(tp(R, phi=phi) * num - (tp * num)(R, phi=phi)) < 1e-10
            ), "Multiplying a planarPotential with a number does not behave as expected"
            assert (
                numpy.fabs(tp(R, phi=phi) / num - (tp / num)(R, phi=phi)) < 1e-10
            ), "Dividing a planarPotential with a number does not behave as expected"
        else:
            assert (
                numpy.fabs(tp(R, Z, phi=phi) * num - (num * tp)(R, Z, phi=phi)) < 1e-10
            ), "Multiplying a Potential with a number does not behave as expected"
            # Other way...
            assert (
                numpy.fabs(tp(R, Z, phi=phi) * num - (tp * num)(R, Z, phi=phi)) < 1e-10
            ), "Multiplying a Potential with a number does not behave as expected"
            assert (
                numpy.fabs(tp(R, Z, phi=phi) / num - (tp / num)(R, Z, phi=phi)) < 1e-10
            ), "Dividing a Potential with a number does not behave as expected"
    return None


# Test whether potentials that support array input do so correctly
def test_potential_array_input():
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
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    rmpots.append("FerrersPotential")
    rmpots.append("PerfectEllipsoidPotential")
    rmpots.append("TriaxialHernquistPotential")
    rmpots.append("TriaxialJaffePotential")
    rmpots.append("TriaxialNFWPotential")
    rmpots.append("TwoPowerTriaxialPotential")
    rmpots.append("DoubleExponentialDiskPotential")
    rmpots.append("RazorThinExponentialDiskPotential")
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    rmpots.append("AnySphericalPotential")
    rmpots.append("SphericalShellPotential")
    rmpots.append("HomogeneousSpherePotential")
    rmpots.append("TriaxialGaussianPotential")
    rmpots.append("PowerTriaxialPotential")
    # These cannot be setup without arguments
    rmpots.append("MovingObjectPotential")
    rmpots.append("SnapshotRZPotential")
    rmpots.append("InterpSnapshotRZPotential")
    # 2D ones that cannot use this test
    rmpots.append("CosmphiDiskPotential")
    rmpots.append("EllipticalDiskPotential")
    rmpots.append("LopsidedDiskPotential")
    rmpots.append("HenonHeilesPotential")
    rmpots.append("TransientLogSpiralPotential")
    rmpots.append("SteadyLogSpiralPotential")
    # 1D ones that cannot use this test
    rmpots.append("IsothermalDiskPotential")
    rmpots.append("KGPotential")
    for p in rmpots:
        pots.remove(p)
    rs = numpy.linspace(0.1, 2.0, 11)
    zs = numpy.linspace(-2.0, 2.0, 11)
    phis = numpy.linspace(0.0, numpy.pi, 11)
    ts = numpy.linspace(0.0, 10.0, 11)
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        # Potential itself
        tpevals = numpy.array(
            [tp(r, z, phi=phi, t=t) for (r, z, phi, t) in zip(rs, zs, phis, ts)]
        )
        assert numpy.all(
            numpy.fabs(tp(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} evaluation does not work as expected for array inputs"
        # Rforce
        tpevals = numpy.array(
            [tp.Rforce(r, z, phi=phi, t=t) for (r, z, phi, t) in zip(rs, zs, phis, ts)]
        )
        assert numpy.all(
            numpy.fabs(tp.Rforce(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} Rforce evaluation does not work as expected for array inputs"
        # zforce
        tpevals = numpy.array(
            [tp.zforce(r, z, phi=phi, t=t) for (r, z, phi, t) in zip(rs, zs, phis, ts)]
        )
        assert numpy.all(
            numpy.fabs(tp.zforce(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} zforce evaluation does not work as expected for array inputs"
        # phitorque
        tpevals = numpy.array(
            [
                tp.phitorque(r, z, phi=phi, t=t)
                for (r, z, phi, t) in zip(rs, zs, phis, ts)
            ]
        )
        assert numpy.all(
            numpy.fabs(tp.phitorque(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} zforce evaluation does not work as expected for array inputs"
        # R2deriv
        if hasattr(tp, "_R2deriv"):
            tpevals = numpy.array(
                [
                    tp.R2deriv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.R2deriv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} R2deriv evaluation does not work as expected for array inputs"
        # z2deriv
        if (
            hasattr(tp, "_z2deriv") and not p == "TwoPowerSphericalPotential"
        ):  # latter bc done through R2deriv
            tpevals = numpy.array(
                [
                    tp.z2deriv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.z2deriv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} z2deriv evaluation does not work as expected for array inputs"
        # phi2deriv
        if hasattr(tp, "_R2deriv"):
            tpevals = numpy.array(
                [
                    tp.phi2deriv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.phi2deriv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} phi2deriv evaluation does not work as expected for array inputs"
        # Rzderiv
        if hasattr(tp, "_Rzderiv"):
            tpevals = numpy.array(
                [
                    tp.Rzderiv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.Rzderiv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} Rzderiv evaluation does not work as expected for array inputs"
        # Rphideriv
        if hasattr(tp, "_Rphideriv"):
            tpevals = numpy.array(
                [
                    tp.Rphideriv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.Rphideriv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} Rphideriv evaluation does not work as expected for array inputs"
        # phizderiv
        if hasattr(tp, "_phizderiv"):
            tpevals = numpy.array(
                [
                    tp.phizderiv(r, z, phi=phi, t=t)
                    for (r, z, phi, t) in zip(rs, zs, phis, ts)
                ]
            )
            assert numpy.all(
                numpy.fabs(tp.phizderiv(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
            ), f"{p} phizderiv evaluation does not work as expected for array inputs"
        # dens
        tpevals = numpy.array(
            [tp.dens(r, z, phi=phi, t=t) for (r, z, phi, t) in zip(rs, zs, phis, ts)]
        )
        assert numpy.all(
            numpy.fabs(tp.dens(rs, zs, phi=phis, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} dens evaluation does not work as expected for array inputs"
    return None


# Test that 1D potentials created using toVertical can handle array input if
# their 3D versions can
def test_toVertical_array():
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
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    rmpots = [
        "Potential",
        "MWPotential",
        "MWPotential2014",
        "interpRZPotential",
        "linearPotential",
        "planarAxiPotential",
        "planarPotential",
        "verticalPotential",
        "PotentialError",
        "EllipsoidalPotential",
        "NumericalPotentialDerivativesMixin",
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    rmpots.append("FerrersPotential")
    rmpots.append("PerfectEllipsoidPotential")
    rmpots.append("TriaxialHernquistPotential")
    rmpots.append("TriaxialJaffePotential")
    rmpots.append("TriaxialNFWPotential")
    rmpots.append("TwoPowerTriaxialPotential")
    rmpots.append("DoubleExponentialDiskPotential")
    rmpots.append("RazorThinExponentialDiskPotential")
    rmpots.append("AnyAxisymmetricRazorThinDiskPotential")
    rmpots.append("AnySphericalPotential")
    rmpots.append("SphericalShellPotential")
    rmpots.append("HomogeneousSpherePotential")
    rmpots.append("TriaxialGaussianPotential")
    rmpots.append("PowerTriaxialPotential")
    # These cannot be setup without arguments
    rmpots.append("MovingObjectPotential")
    rmpots.append("SnapshotRZPotential")
    rmpots.append("InterpSnapshotRZPotential")
    for p in rmpots:
        pots.remove(p)
    xs = numpy.linspace(-2.0, 2.0, 11)
    ts = numpy.linspace(0.0, 10.0, 11)
    for p in pots:
        # if not 'NFW' in p: continue #For testing the test
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        # Only do 3D --> 1D potentials
        if not isinstance(tp, potential.Potential):
            continue
        tp = potential.toVerticalPotential(tp, 0.8, phi=0.2)
        # Potential itself
        tpevals = numpy.array([tp(x, t=t) for (x, t) in zip(xs, ts)])
        assert numpy.all(
            numpy.fabs(tp(xs, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} evaluation does not work as expected for array inputs for toVerticalPotential potentials"
        # force
        tpevals = numpy.array([tp.force(x, t=t) for (x, t) in zip(xs, ts)])
        assert numpy.all(
            numpy.fabs(tp.force(xs, t=ts) - tpevals) < 10.0**-10.0
        ), f"{p} force evaluation does not work as expected for array inputs for toVerticalPotential"
    # Also test Morgan's example
    pot = potential.toVerticalPotential(potential.MWPotential2014, 1.0)
    # Potential itself
    tpevals = numpy.array(
        [potential.evaluatelinearPotentials(pot, x, t=t) for (x, t) in zip(xs, ts)]
    )
    assert numpy.all(
        numpy.fabs(potential.evaluatelinearPotentials(pot, xs, t=ts) - tpevals)
        < 10.0**-10.0
    ), f"{p} evaluation does not work as expected for array inputs for toVerticalPotential potentials"
    # Rforce
    tpevals = numpy.array(
        [potential.evaluatelinearForces(pot, x, t=t) for (x, t) in zip(xs, ts)]
    )
    assert numpy.all(
        numpy.fabs(potential.evaluatelinearForces(pot, xs, t=ts) - tpevals)
        < 10.0**-10.0
    ), f"{p} force evaluation does not work as expected for array inputs for toVerticalPotential"
    return None


# Test that all potentials can be evaluated at zero
def test_potential_at_zero():
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
    # pots.append('specialTwoPowerSphericalPotential')
    # pots.append('DehnenTwoPowerSphericalPotential')
    # pots.append('DehnenCoreTwoPowerSphericalPotential')
    # pots.append('HernquistTwoPowerSphericalPotential')
    # pots.append('JaffeTwoPowerSphericalPotential')
    # pots.append('NFWTwoPowerSphericalPotential') # Difficult, and who cares?
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("testMWPotential")
    pots.append("mockInterpRZPotential")
    if _PYNBODY_LOADED:
        pots.append("mockSnapshotRZPotential")
        pots.append("mockInterpSnapshotRZPotential")
    pots.append("oblateHernquistPotential")
    pots.append("oblateNFWPotential")
    pots.append("oblatenoGLNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    pots.append("zRotatedTriaxialNFWPotential")  # Difficult bc of rotation
    pots.append("yRotatedTriaxialNFWPotential")  # Difficult bc of rotation
    pots.append("fullyRotatedTriaxialNFWPotential")  # Difficult bc of rotation
    pots.append("fullyRotatednoGLTriaxialNFWPotential")  # Difficult bc of rotation
    pots.append("HernquistTwoPowerTriaxialPotential")
    pots.append("NFWTwoPowerTriaxialPotential")
    # pots.append('JaffeTwoPowerTriaxialPotential') # not finite
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("sech2DiskSCFPotential")
    pots.append("expwholeDiskSCFPotential")
    pots.append("nonaxiDiskSCFPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    # Remove some more potentials that we don't support for now TO DO
    rmpots.append("BurkertPotential")  # Need to figure out...
    # rmpots.append('FerrersPotential') # Need to figure out...
    # rmpots.append('KuzminKutuzovStaeckelPotential') # Need to figure out...
    rmpots.append("RazorThinExponentialDiskPotential")  # Need to figure out...
    rmpots.append("RingPotential")  # Easy, but who cares?
    # rmpots.append('SoftenedNeedleBarPotential') # Not that hard, but haven't done it
    rmpots.append("SpiralArmsPotential")
    rmpots.append("TwoPowerSphericalPotential")  # Need to figure out
    # rmpots.append('TwoPowerTriaxialPotential') # Need to figure out
    # 2D ones that cannot use this test
    rmpots.append("CosmphiDiskPotential")
    rmpots.append("EllipticalDiskPotential")
    rmpots.append("LopsidedDiskPotential")
    rmpots.append("HenonHeilesPotential")
    rmpots.append("TransientLogSpiralPotential")
    rmpots.append("SteadyLogSpiralPotential")
    # 1D ones that cannot use this test
    rmpots.append("IsothermalDiskPotential")
    rmpots.append("KGPotential")
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        assert not numpy.isnan(
            potential.evaluatePotentials(tp, 0, 0, phi=0.0, t=0.0)
        ), f"Potential {p} evaluated at zero gave NaN"
        # Also for arrays
        if (
            p == "FerrersPotential"
            or p == "HomogeneousSpherePotential"
            or p == "PerfectEllipsoidPotential"
            or p == "SphericalShellPotential"
            or p == "AnyAxisymmetricRazorThinDiskPotential"
            or p == "AnySphericalPotential"
            or p == "mockRotatedAndTiltedMWP14WrapperPotential"
            or p == "mockRotatedAndTiltedMWP14WrapperPotentialwInclination"
            or p == "mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination"
            or p == "mockRotatedTiltedOffsetMWP14WrapperPotential"
            or p == "mockOffsetMWP14WrapperPotential"
            or "riaxial" in p
            or "oblate" in p
            or "prolate" in p
        ):
            continue
        assert not numpy.any(
            numpy.isnan(
                potential.evaluatePotentials(
                    tp, numpy.zeros(4), numpy.zeros(4), phi=0.0, t=0.0
                )
            )
        ), f"Potential {p} evaluated at zero gave NaN"
    return None


# Test that all potentials can be evaluated with large numbers and with infinity
def test_potential_at_infinity():
    # One of the main reasons for this test is the implementation of vesc,
    # which uses the potential at infinity. Import what vesc uses for infinity
    from galpy.potential.plotEscapecurve import _INF

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
    # pots.append('specialTwoPowerSphericalPotential')
    pots.append("DehnenTwoPowerSphericalPotential")
    pots.append("DehnenCoreTwoPowerSphericalPotential")
    pots.append("HernquistTwoPowerSphericalPotential")
    pots.append("JaffeTwoPowerSphericalPotential")
    # pots.append('NFWTwoPowerSphericalPotential') # Difficult, and who cares?
    pots.append("specialMiyamotoNagaiPotential")
    pots.append("specialMN3ExponentialDiskPotentialPD")
    pots.append("specialMN3ExponentialDiskPotentialSECH")
    pots.append("specialPowerSphericalPotential")
    pots.append("specialFlattenedPowerPotential")
    pots.append("testMWPotential")
    pots.append("mockInterpRZPotential")
    # if _PYNBODY_LOADED:
    #    pots.append('mockSnapshotRZPotential')
    #    pots.append('mockInterpSnapshotRZPotential')
    pots.append("oblateHernquistPotential")
    pots.append("oblateNFWPotential")
    pots.append("oblatenoGLNFWPotential")
    pots.append("oblateJaffePotential")
    pots.append("prolateHernquistPotential")
    pots.append("prolateNFWPotential")
    pots.append("prolateJaffePotential")
    pots.append("triaxialHernquistPotential")
    pots.append("triaxialNFWPotential")
    pots.append("triaxialJaffePotential")
    # pots.append('zRotatedTriaxialNFWPotential') # Difficult bc of rotation
    # pots.append('yRotatedTriaxialNFWPotential') # Difficult bc of rotation
    # pots.append('fullyRotatedTriaxialNFWPotential') # Difficult bc of rotation
    # pots.append('fullyRotatednoGLTriaxialNFWPotential') # Difficult bc of rotation
    # pots.append('HernquistTwoPowerTriaxialPotential')
    # pots.append('NFWTwoPowerTriaxialPotential')
    # pots.append('JaffeTwoPowerTriaxialPotential')
    pots.append("mockSCFZeeuwPotential")
    pots.append("mockSCFNFWPotential")
    pots.append("mockSCFAxiDensity1Potential")
    pots.append("mockSCFAxiDensity2Potential")
    pots.append("mockSCFDensityPotential")
    pots.append("sech2DiskSCFPotential")
    pots.append("expwholeDiskSCFPotential")
    pots.append("nonaxiDiskSCFPotential")
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
    pots.append("mockAdiabaticContractionMWP14WrapperPotential")
    pots.append("mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotential")
    pots.append("mockRotatedAndTiltedMWP14WrapperPotentialwInclination")
    pots.append("mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination")
    pots.append("mockRotatedTiltedOffsetMWP14WrapperPotential")
    pots.append("mockOffsetMWP14WrapperPotential")
    pots.append("mockKuzminLikeWrapperPotential")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    # Remove some more potentials that we don't support for now TO DO
    rmpots.append("FerrersPotential")  # Need to figure out...
    rmpots.append("KuzminKutuzovStaeckelPotential")  # Need to figure out...
    rmpots.append("RazorThinExponentialDiskPotential")  # Need to figure out...
    rmpots.append("SoftenedNeedleBarPotential")  # Not that hard, but haven't done it
    rmpots.append("SpiralArmsPotential")  # Need to have 0 x cos = 0
    rmpots.append("TwoPowerTriaxialPotential")  # Need to figure out
    # 2D ones that cannot use this test
    rmpots.append("CosmphiDiskPotential")
    rmpots.append("EllipticalDiskPotential")
    rmpots.append("LopsidedDiskPotential")
    rmpots.append("HenonHeilesPotential")
    rmpots.append("TransientLogSpiralPotential")
    rmpots.append("SteadyLogSpiralPotential")
    # 1D ones that cannot use this test
    rmpots.append("IsothermalDiskPotential")
    rmpots.append("KGPotential")
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        # Setup instance of potential
        try:
            tclass = getattr(potential, p)
        except AttributeError:
            tclass = getattr(sys.modules[__name__], p)
        tp = tclass()
        if hasattr(tp, "normalize"):
            tp.normalize(1.0)
        assert not numpy.isnan(
            potential.evaluatePotentials(tp, numpy.inf, 0, phi=0.0, t=0.0)
        ), f"Potential {p} evaluated at infinity gave NaN"
        assert not numpy.isnan(
            potential.evaluatePotentials(tp, _INF, 0, phi=0.0, t=0.0)
        ), f"Potential {p} evaluated at vesc _INF gave NaN"
        # Also for arrays
        if (
            p == "HomogeneousSpherePotential"
            or p == "PerfectEllipsoidPotential"
            or p == "SphericalShellPotential"
            or p == "AnyAxisymmetricRazorThinDiskPotential"
            or p == "AnySphericalPotential"
            or p == "mockRotatedAndTiltedMWP14WrapperPotential"
            or p == "mockRotatedAndTiltedMWP14WrapperPotentialwInclination"
            or p == "mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination"
            or p == "mockRotatedTiltedOffsetMWP14WrapperPotential"
            or p == "mockOffsetMWP14WrapperPotential"
            or "riaxial" in p
            or "oblate" in p
            or "prolate" in p
        ):
            continue
        assert not numpy.any(
            numpy.isnan(
                potential.evaluatePotentials(
                    tp, numpy.inf * numpy.ones(4), numpy.zeros(4), phi=0.0, t=0.0
                )
            )
        ), f"Potential {p} evaluated at infinity gave NaN"
        assert not numpy.any(
            numpy.isnan(
                potential.evaluatePotentials(
                    tp, _INF * numpy.ones(4), numpy.zeros(4), phi=0.0, t=0.0
                )
            )
        ), f"Potential {p} evaluated at vesc _INF gave NaN"
    return None


# Test that the amplitude for potentials with a finite mass and amp=mass is
# correct through the relation -r^2 F_r =~ GM at large r
def test_finitemass_amp():
    r_large = 10000.0
    # KeplerPotential
    mass = 3.0
    kp = potential.KeplerPotential(amp=mass)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * kp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of KeplerPotential does not not equal total mass"
    # IsochronePotential
    r_large = 1000000000.0
    mass = 3.0
    ip = potential.IsochronePotential(amp=mass, b=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * ip.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of IsochronePotential does not not equal total mass"
    # PlummerPotential
    r_large = 10000.0
    mass = 3.0
    pp = potential.PlummerPotential(amp=mass, b=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * pp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of PlummerPotential does not not equal total mass"
    # SphericalShellPotential
    mass = 3.0
    sp = potential.SphericalShellPotential(amp=mass, a=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * sp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of SphericalShellPotential does not not equal total mass"
    # RingPotential
    mass = 3.0
    rp = potential.RingPotential(amp=mass, a=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * rp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of RingPotential does not not equal total mass"
    # KuzminDiskPotential
    r_large = 1000000000.0
    mass = 3.0
    kp = potential.KuzminDiskPotential(amp=mass, a=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * kp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of KuzminDiskPotential does not not equal total mass"
    # MiyamotoNagaiPotential
    r_large = 1000000000.0
    mass = 3.0
    mp = potential.MiyamotoNagaiPotential(amp=mass, a=0.4)
    assert (
        numpy.fabs(
            mass
            + r_large**2.0
            * mp.rforce(
                r_large / numpy.sqrt(2.0),
                r_large / numpy.sqrt(2.0),
            )
        )
        < 1e-8
    ), "Mass amp parameter of MiyamotoNagaiPotential does not not equal total mass"
    return None


# Test that the spherically radial force is correct
def test_rforce():
    # Spherical potentials: Rforce = rforce x R / r; zforce = rforce x z /r
    pp = potential.PlummerPotential(amp=2.0, b=2.0)
    R, z = 1.3, 0.4
    r = numpy.sqrt(R * R + z * z)
    assert (
        numpy.fabs(pp.Rforce(R, z) * r / R - pp.rforce(R, z)) < 10.0**-10.0
    ), "rforce does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            potential.evaluateRforces(pp, R, z) * r / R
            - potential.evaluaterforces(pp, R, z)
        )
        < 10.0**-10.0
    ), "evaluaterforces does not behave as expected for spherical potentials"
    return None


def test_rforce_dissipative():
    # Use dynamical friction along a radial orbit at z=0 --> spherical
    pp = potential.PlummerPotential(amp=1.12, b=2.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=pp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    R, z, phi = 1.3, 0.0, 1.1
    v = [0.1, 0.0, 0.0]
    r = numpy.sqrt(R * R + z * z)
    assert (
        numpy.fabs(
            cdfc.Rforce(R, z, phi=phi, v=v) * r / R - cdfc.rforce(R, z, phi=phi, v=v)
        )
        < 10.0**-10.0
    ), "rforce does not behave as expected for spherical potentials for dissipative forces"
    assert (
        numpy.fabs(
            potential.evaluateRforces([pp, cdfc], R, z, phi=phi, v=v) * r / R
            - potential.evaluaterforces([pp, cdfc], R, z, phi=phi, v=v)
        )
        < 10.0**-10.0
    ), "evaluaterforces does not behave as expected for spherical potentials for dissipative forces"
    assert (
        numpy.fabs(
            potential.evaluateRforces(cdfc, R, z, phi=phi, v=v) * r / R
            - potential.evaluaterforces(cdfc, R, z, phi=phi, v=v)
        )
        < 10.0**-10.0
    ), "evaluaterforces does not behave as expected for spherical potentials for dissipative forces"
    return None


# Test that the spherically second radial derivative is correct
def test_r2deriv():
    # Spherical potentials: Rforce = rforce x R / r; zforce = rforce x z /r
    # and R2deriv = r2deriv x (R/r)^2 - rforce x z^2/r^3
    # and z2deriv = z2deriv x (z/r)^2 - rforce x R^2/R^3
    # and Rzderiv = r2deriv x Rz/r^2 + rforce x Rz/r^3
    pp = potential.PlummerPotential(amp=2.0, b=2.0)
    R, z = 1.3, 0.4
    r = numpy.sqrt(R * R + z * z)
    assert (
        numpy.fabs(
            pp.R2deriv(R, z)
            - pp.r2deriv(R, z) * (R / r) ** 2.0
            + pp.rforce(R, z) * z**2.0 / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            pp.z2deriv(R, z)
            - pp.r2deriv(R, z) * (z / r) ** 2.0
            + pp.rforce(R, z) * R**2.0 / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            pp.Rzderiv(R, z)
            - pp.r2deriv(R, z) * R * z / r**2.0
            - pp.rforce(R, z) * R * z / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            potential.evaluateR2derivs([pp], R, z)
            - potential.evaluater2derivs([pp], R, z) * (R / r) ** 2.0
            + potential.evaluaterforces([pp], R, z) * z**2.0 / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            potential.evaluatez2derivs([pp], R, z)
            - potential.evaluater2derivs([pp], R, z) * (z / r) ** 2.0
            + potential.evaluaterforces([pp], R, z) * R**2.0 / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    assert (
        numpy.fabs(
            potential.evaluateRzderivs([pp], R, z)
            - potential.evaluater2derivs([pp], R, z) * R * z / r**2.0
            - potential.evaluaterforces([pp], R, z) * R * z / r**3.0
        )
        < 10.0**-10.0
    ), "r2deriv does not behave as expected for spherical potentials"
    return None


# Check that the masses are calculated correctly for spherical potentials
def test_mass_spher():
    # PowerPotential close to Kepler should be very steep
    pp = potential.PowerSphericalPotential(amp=2.0, alpha=2.999)
    kp = potential.KeplerPotential(amp=2.0)
    assert (
        numpy.fabs(
            ((3.0 - 2.999) / (4.0 * numpy.pi) * pp.mass(10.0) - kp.mass(10.0))
            / kp.mass(10.0)
        )
        < 10.0**-2.0
    ), "Mass for PowerSphericalPotential close to KeplerPotential is not close to KeplerPotential's mass"
    pp = potential.PowerSphericalPotential(amp=2.0)
    # mass = amp x r^(3-alpha)
    tR = 1.0
    assert (
        numpy.fabs(
            potential.mass(pp, tR, forceint=True) - pp._amp * tR ** (3.0 - pp.alpha)
        )
        < 10.0**-10.0
    ), "Mass for PowerSphericalPotential not as expected"
    tR = 2.0
    assert (
        numpy.fabs(
            potential.mass([pp], tR, forceint=True) - pp._amp * tR ** (3.0 - pp.alpha)
        )
        < 10.0**-10.0
    ), "Mass for PowerSphericalPotential not as expected"
    tR = 20.0
    assert (
        numpy.fabs(pp.mass(tR, forceint=True) - pp._amp * tR ** (3.0 - pp.alpha))
        < 10.0**-9.0
    ), "Mass for PowerSphericalPotential not as expected"
    # Test that for a cut-off potential, the mass far beyond the cut-off is
    # 2pi rc^(3-alpha) gamma(1.5-alpha/2)
    pp = potential.PowerSphericalPotentialwCutoff(amp=2.0)
    from scipy import special

    expecMass = (
        2.0
        * pp._amp
        * numpy.pi
        * pp.rc ** (3.0 - pp.alpha)
        * special.gamma(1.5 - pp.alpha / 2.0)
    )
    tR = 5.0
    assert (
        numpy.fabs((pp.mass(tR, forceint=True) - expecMass) / expecMass) < 10.0**-6.0
    ), "Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected"
    tR = 15.0
    assert (
        numpy.fabs((pp.mass(tR, forceint=True) - expecMass) / expecMass) < 10.0**-6.0
    ), "Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected"
    tR = 50.0
    assert (
        numpy.fabs((pp.mass(tR, forceint=True) - expecMass) / expecMass) < 10.0**-6.0
    ), "Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected"
    # Jaffe and Hernquist both have finite masses, NFW diverges logarithmically
    jp = potential.JaffePotential(amp=2.0, a=0.1)
    hp = potential.HernquistPotential(amp=2.0, a=0.1)
    np = potential.NFWPotential(amp=2.0, a=0.1)
    tR = 10.0
    # Limiting behavior
    jaffemass = jp._amp * (1.0 - jp.a / tR)
    hernmass = hp._amp / 2.0 * (1.0 - 2.0 * hp.a / tR)
    nfwmass = np._amp * (numpy.log(tR / np.a) - 1.0 + np.a / tR)
    assert (
        numpy.fabs((jp.mass(tR, forceint=True) - jaffemass) / jaffemass) < 10.0**-3.0
    ), "Limit mass for Jaffe potential not as expected"
    assert (
        numpy.fabs((hp.mass(tR, forceint=True) - hernmass) / hernmass) < 10.0**-3.0
    ), "Limit mass for Hernquist potential not as expected"
    assert (
        numpy.fabs((np.mass(tR, forceint=True) - nfwmass) / nfwmass) < 10.0**-2.0
    ), "Limit mass for NFW potential not as expected"
    tR = 200.0
    # Limiting behavior, add z, to test that too
    jaffemass = jp._amp * (1.0 - jp.a / tR)
    hernmass = hp._amp / 2.0 * (1.0 - 2.0 * hp.a / tR)
    nfwmass = np._amp * (numpy.log(tR / np.a) - 1.0 + np.a / tR)
    assert (
        numpy.fabs((jp.mass(tR, forceint=True) - jaffemass) / jaffemass) < 10.0**-6.0
    ), "Limit mass for Jaffe potential not as expected"
    assert (
        numpy.fabs((hp.mass(tR, forceint=True) - hernmass) / hernmass) < 10.0**-6.0
    ), "Limit mass for Jaffe potential not as expected"
    assert (
        numpy.fabs((np.mass(tR, forceint=True) - nfwmass) / nfwmass) < 10.0**-4.0
    ), "Limit mass for NFW potential not as expected"
    # Burkert as an example of a SphericalPotential
    bp = potential.BurkertPotential(amp=2.0, a=3.0)
    assert (
        numpy.fabs(bp.mass(4.2, forceint=True) - bp.mass(4.2)) < 1e-6
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"
    return None


# Check that the masses are implemented correctly for spherical potentials
def test_mass_spher_analytic():
    # TwoPowerSphericalPotentials all have explicitly implemented masses
    dcp = potential.DehnenCoreSphericalPotential(amp=2.0)
    jp = potential.JaffePotential(amp=2.0)
    hp = potential.HernquistPotential(amp=2.0)
    np = potential.NFWPotential(amp=2.0)
    tp = potential.TwoPowerSphericalPotential(amp=2.0)
    dp = potential.DehnenSphericalPotential(amp=2.0)
    pp = potential.PlummerPotential(amp=2.0, b=1.3)
    tR = 2.0
    assert (
        numpy.fabs(dcp.mass(tR, forceint=True) - dcp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for Dehnen Core potential"
    assert (
        numpy.fabs(jp.mass(tR, forceint=True) - jp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for Jaffe potential"
    assert (
        numpy.fabs(hp.mass(tR, forceint=True) - hp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for Hernquist potential"
    assert (
        numpy.fabs(np.mass(tR, forceint=True) - np.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for NFW potential"
    assert (
        numpy.fabs(tp.mass(tR, forceint=True) - tp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for TwoPowerSpherical potential"
    assert (
        numpy.fabs(dp.mass(tR, forceint=True) - dp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for DehnenSphericalPotential potential, for not z is None"
    assert (
        numpy.fabs(pp.mass(tR, forceint=True) - pp.mass(tR)) < 10.0**-10.0
    ), "Explicit mass does not agree with integral of the density for Plummer potential"
    return None


# Check that the masses within (0,R) and (-z,z) are calculated correctly for spherical potentials
def test_mass_spher_z():
    from scipy import integrate

    def sphermass(pot, R, z):
        return (
            4.0
            * numpy.pi
            * integrate.dblquad(
                lambda rp, zp: rp * potential.evaluateDensities(pot, rp, 0.0, phi=0.0),
                0.0,
                z,
                lambda z: z,
                lambda z: numpy.sqrt(R**2.0 + z**2.0),
            )[0]
        )

    # TwoPowerSphericalPotential
    tp = potential.TwoPowerSphericalPotential(amp=2.0)
    assert (
        numpy.fabs(tp.mass(4.2, 1.3) - sphermass(tp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # DehnenSphericalPotential
    dp = potential.DehnenSphericalPotential(amp=2.0)
    assert (
        numpy.fabs(dp.mass(4.2, 1.3) - sphermass(dp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # PlummerPotential
    pp = potential.PlummerPotential(amp=2.0, b=1.3)
    assert (
        numpy.fabs(pp.mass(4.2, 1.3) - sphermass(pp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # DehnenCoreSphericalPotential
    dcp = potential.DehnenCoreSphericalPotential(amp=2.0)
    assert (
        numpy.fabs(dcp.mass(4.2, 1.3) - sphermass(dcp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # JaffePotential
    jp = potential.JaffePotential(amp=2.0)
    assert (
        numpy.fabs(jp.mass(4.2, 1.3) - sphermass(jp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # HernquistPotential
    hp = potential.HernquistPotential(amp=2.0)
    assert (
        numpy.fabs(hp.mass(4.2, 1.3) - sphermass(hp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # NFWPotential
    np = potential.NFWPotential(amp=2.0)
    assert (
        numpy.fabs(np.mass(4.2, 1.3) - sphermass(np, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # SCF version of HernquistPotential
    hp = potential.SCFPotential.from_density(
        potential.HernquistPotential(amp=2.0), 1, 0, 1.0, symmetry="spherical"
    )
    assert (
        numpy.fabs(hp.mass(4.2, 1.3) - sphermass(hp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # AnySphericalPotential version of HernquistPotential
    hp = potential.AnySphericalPotential(
        dens=lambda r: potential.HernquistPotential(amp=2.0).dens(r, 0.0),
    )
    assert (
        numpy.fabs(hp.mass(4.2, 1.3) - sphermass(hp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # TwoPowerTriaxialPotential that is actually spherical
    ttp = potential.TwoPowerTriaxialPotential(amp=2.0, a=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(ttp.mass(4.2, 1.3) - sphermass(ttp, 4.2, 1.3)) < 1e-5
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # TriaxialHernquistPotential that is actually spherical
    thp = potential.TriaxialHernquistPotential(amp=2.0, a=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(thp.mass(4.2, 1.3) - sphermass(thp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # TriaxialJaffe potential that is actually spherical
    tjp = potential.TriaxialJaffePotential(amp=2.0, a=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(tjp.mass(4.2, 1.3) - sphermass(tjp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # TriaxialNFW potential that is actually spherical
    tnp = potential.TriaxialNFWPotential(amp=2.0, a=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(tnp.mass(4.2, 1.3) - sphermass(tnp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # PerfectEllipsoidPotential that is actually spherical
    pep = potential.PerfectEllipsoidPotential(amp=2.0, a=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(pep.mass(4.2, 1.3) - sphermass(pep, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # TriaxialGaussian potential that is actually spherical
    tgp = potential.TriaxialGaussianPotential(amp=2.0, sigma=1.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(tgp.mass(4.2, 1.3) - sphermass(tgp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    # Dummy EllipsoidalPotential for testing the general approach
    from galpy.potential.EllipsoidalPotential import EllipsoidalPotential

    class dummy(EllipsoidalPotential):
        def __init__(
            self,
            amp=1.0,
            b=1.0,
            c=1.0,
            zvec=None,
            pa=None,
            glorder=50,
            normalize=False,
            ro=None,
            vo=None,
        ):
            EllipsoidalPotential.__init__(
                self, amp=amp, b=b, c=c, zvec=zvec, pa=pa, glorder=glorder, ro=ro, vo=vo
            )
            return None

        def _mdens(self, m):
            return m**-2.0

    dp = dummy(amp=2.0, b=1.0, c=1.0)
    r = 1.9
    assert (
        numpy.fabs(dp.mass(4.2, 1.3) - sphermass(dp, 4.2, 1.3)) < 1e-10
    ), "2D Mass computed with Potentials's general implementation incorrect for spherical potential"

    return None


# Check that the masses are calculated correctly for axisymmetric potentials
def test_mass_axi():
    # For Miyamoto-Nagai, we know that mass integrated over everything should be equal to amp, so
    mp = potential.MiyamotoNagaiPotential(amp=1.0)
    assert (
        numpy.fabs(mp.mass(200.0, 20.0) - 1.0) < 0.01
    ), "Total mass of Miyamoto-Nagai potential w/ amp=1 is not equal to 1"
    # Also spherical
    assert (
        numpy.fabs(mp.mass(200.0) - 1.0) < 0.01
    ), "Total mass of Miyamoto-Nagai potential w/ amp=1 is not equal to 1"
    # For a double-exponential disk potential, the
    # mass(R,z) = amp x hR^2 x hz x (1-(1+R/hR)xe^(-R/hR)) x (1-e^(-Z/hz)
    dp = potential.DoubleExponentialDiskPotential(amp=2.0)

    def dblexpmass(r, z, dp):
        return (
            4.0
            * numpy.pi
            * dp._amp
            * dp._hr**2.0
            * dp._hz
            * (1.0 - (1.0 + r / dp._hr) * numpy.exp(-r / dp._hr))
            * (1.0 - numpy.exp(-z / dp._hz))
        )

    tR, tz = 0.01, 0.01
    assert (
        numpy.fabs(dp.mass(tR, tz, forceint=True) - dblexpmass(tR, tz, dp)) < 5e-8
    ), "Mass for DoubleExponentialDiskPotential incorrect"
    tR, tz = 0.1, 0.05
    assert (
        numpy.fabs(dp.mass(tR, tz, forceint=True) - dblexpmass(tR, tz, dp)) < 3e-7
    ), "Mass for DoubleExponentialDiskPotential incorrect"
    tR, tz = 1.0, 0.1
    assert (
        numpy.fabs(dp.mass(tR, tz, forceint=True) - dblexpmass(tR, tz, dp)) < 1e-6
    ), "Mass for DoubleExponentialDiskPotential incorrect"
    tR, tz = 5.0, 0.1
    assert (
        numpy.fabs(
            (dp.mass(tR, tz, forceint=True) - dblexpmass(tR, tz, dp))
            / dblexpmass(tR, tz, dp)
        )
        < 10.0**-5.0
    ), "Mass for DoubleExponentialDiskPotential incorrect"
    tR, tz = 5.0, 1.0
    assert (
        numpy.fabs(
            (dp.mass(tR, tz, forceint=True) - dblexpmass(tR, tz, dp))
            / dblexpmass(tR, tz, dp)
        )
        < 10.0**-5.0
    ), "Mass for DoubleExponentialDiskPotential incorrect"
    # Razor thin disk
    rp = potential.RazorThinExponentialDiskPotential(amp=2.0)

    def razexpmass(r, z, dp):
        return (
            2.0
            * numpy.pi
            * rp._amp
            * rp._hr**2.0
            * (1.0 - (1.0 + r / rp._hr) * numpy.exp(-r / rp._hr))
        )

    tR, tz = 0.01, 0.01
    assert (
        numpy.fabs((rp.mass(tR, tz) - razexpmass(tR, tz, rp)) / razexpmass(tR, tz, rp))
        < 10.0**-10.0
    ), "Mass for RazorThinExponentialDiskPotential incorrect"
    tR, tz = 0.1, 0.05
    assert (
        numpy.fabs((rp.mass(tR, tz) - razexpmass(tR, tz, rp)) / razexpmass(tR, tz, rp))
        < 10.0**-10.0
    ), "Mass for RazorThinExponentialDiskPotential incorrect"
    tR, tz = 1.0, 0.1
    assert (
        numpy.fabs((rp.mass(tR, tz) - razexpmass(tR, tz, rp)) / razexpmass(tR, tz, rp))
        < 10.0**-10.0
    ), "Mass for RazorThinExponentialDiskPotential incorrect"
    tR, tz = 5.0, 0.1
    assert (
        numpy.fabs((rp.mass(tR, tz) - razexpmass(tR, tz, rp)) / razexpmass(tR, tz, rp))
        < 10.0**-10.0
    ), "Mass for RazorThinExponentialDiskPotential incorrect"
    tR, tz = 5.0, 1.0
    assert (
        numpy.fabs((rp.mass(tR, tz) - razexpmass(tR, tz, rp)) / razexpmass(tR, tz, rp))
        < 10.0**-10.0
    ), "Mass for RazorThinExponentialDiskPotential incorrect"
    # Kuzmin disk, amp = mass
    kp = potential.KuzminDiskPotential(amp=2.0, a=3.0)
    assert (
        numpy.fabs(kp.mass(1000.0, 20.0) - 2.0) < 1e-2
    ), "Mass for KuzminDiskPotential incorrect"
    assert (
        numpy.fabs(kp.mass(1000.0) - 2.0) < 1e-2
    ), "Mass for KuzminDiskPotential incorrect"
    # Test that nonAxi raises error
    from galpy.orbit import Orbit

    mop = potential.MovingObjectPotential(Orbit([1.0, 0.1, 1.1, 0.1, 0.0, 0.0]))
    with pytest.raises(NotImplementedError) as excinfo:
        mop.mass(1.0, 0.0)
    # also for lists
    with pytest.raises(NotImplementedError) as excinfo:
        potential.mass(mop, 1.0, 0.0)
    with pytest.raises(NotImplementedError) as excinfo:
        potential.mass([mop], 1.0, 0.0)
    return None


# Test that axisymmetric masses are correctly returned for negative z (issue #555)
# they should just be the same as those for positive z
def test_mass_axi_negz():
    # Example from @sferrone in issue #555
    R = 1.6303380979868902
    z = 1.2732319411637634
    assert (
        numpy.fabs(
            potential.mass(potential.MWPotential2014, R, z)
            - potential.mass(potential.MWPotential2014, R, -z)
        )
        < 1e-10
    ), "Axisymmetric mass for negative z is not the same as for positive z"
    return None


# Check that the masses are calculated correctly for spheroidal potentials
def test_mass_spheroidal():
    # PerfectEllipsoidPotential: total mass is amp, no matter what the axis ratio
    pep = potential.PerfectEllipsoidPotential(amp=2.0, a=3.0, b=1.3, c=1.9)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for PerfectEllipsoidPotential is incorrect"
    pep = potential.PerfectEllipsoidPotential(amp=2.0, a=3.0, b=1.0, c=1.9)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for PerfectEllipsoidPotential is incorrect"
    pep = potential.PerfectEllipsoidPotential(amp=2.0, a=3.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for PerfectEllipsoidPotential is incorrect"
    pep = potential.PerfectEllipsoidPotential(amp=2.0, a=3.0, b=0.7, c=0.5)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for PerfectEllipsoidPotential is incorrect"
    # For TwoPowerTriaxial, the masses should be bxc times that for the spherical version
    b = 0.7
    c = 0.5
    tpp = potential.TriaxialJaffePotential(amp=2.0, a=3.0, b=b, c=c)
    sp = potential.JaffePotential(amp=2.0, a=3.0)
    assert (
        numpy.fabs(tpp.mass(1.3) / b / c - sp.mass(1.3)) < 1e-6
    ), "TwoPowerTriaxialPotential mass incorrect"
    tpp = potential.TriaxialHernquistPotential(amp=2.0, a=3.0, b=b, c=c)
    sp = potential.HernquistPotential(amp=2.0, a=3.0)
    assert (
        numpy.fabs(tpp.mass(1.3) / b / c - sp.mass(1.3)) < 1e-6
    ), "TwoPowerTriaxialPotential mass incorrect"
    tpp = potential.TriaxialNFWPotential(amp=2.0, a=3.0, b=b, c=c)
    sp = potential.NFWPotential(amp=2.0, a=3.0)
    assert (
        numpy.fabs(tpp.mass(1.3) / b / c - sp.mass(1.3)) < 1e-6
    ), "TwoPowerTriaxialPotential mass incorrect"
    tpp = potential.TwoPowerTriaxialPotential(
        amp=2.0, a=3.0, b=b, c=c, alpha=1.1, beta=4.1
    )
    sp = potential.TwoPowerSphericalPotential(amp=2.0, a=3.0, alpha=1.1, beta=4.1)
    assert (
        numpy.fabs(tpp.mass(1.3) / b / c - sp.mass(1.3)) < 1e-6
    ), "TwoPowerTriaxialPotential mass incorrect"
    # For TriaxialGaussianPotential, total mass is amp, no matter b/c
    pep = potential.TriaxialGaussianPotential(amp=2.0, sigma=3.0, b=1.3, c=1.9)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for TriaxialGaussianPotential is incorrect"
    pep = potential.TriaxialGaussianPotential(amp=2.0, sigma=3.0, b=1.0, c=1.9)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for TriaxialGaussianPotential is incorrect"
    pep = potential.TriaxialGaussianPotential(amp=2.0, sigma=3.0, b=1.0, c=1.0)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for TriaxialGaussianPotential is incorrect"
    pep = potential.TriaxialGaussianPotential(amp=2.0, sigma=3.0, b=0.7, c=0.5)
    assert (
        numpy.fabs(pep.mass(1000.0) - 2.0) < 1e-2
    ), "Total mass for TriaxialGaussianPotential is incorrect"

    # Dummy EllipsoidalPotential for testing the general approach
    from galpy.potential.EllipsoidalPotential import EllipsoidalPotential

    class dummy(EllipsoidalPotential):
        def __init__(
            self,
            amp=1.0,
            b=1.0,
            c=1.0,
            zvec=None,
            pa=None,
            glorder=50,
            normalize=False,
            ro=None,
            vo=None,
        ):
            EllipsoidalPotential.__init__(
                self, amp=amp, b=b, c=c, zvec=zvec, pa=pa, glorder=glorder, ro=ro, vo=vo
            )
            return None

        def _mdens(self, m):
            return m**-2.0

    b = 1.2
    c = 1.7
    dp = dummy(amp=2.0, b=b, c=c)
    r = 1.9
    assert (
        numpy.fabs(dp.mass(r) / b / c - 4.0 * numpy.pi * 2.0 * r) < 1e-6
    ), "General potential.EllipsoidalPotential mass incorrect"
    r = 3.9
    assert (
        numpy.fabs(dp.mass(r) / b / c - 4.0 * numpy.pi * 2.0 * r) < 1e-6
    ), "General potential.EllipsoidalPotential mass incorrect"
    return None


# Check that toVertical and toPlanar work
def test_toVertical_toPlanar():
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
    pots.append("mockInterpSphericalPotential")
    pots.append("mockInterpSphericalPotentialwForce")
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
        "SphericalPotential",
        "interpSphericalPotential",
    ]
    if False:
        rmpots.append("DoubleExponentialDiskPotential")
        rmpots.append("RazorThinExponentialDiskPotential")
    for p in rmpots:
        pots.remove(p)
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
        if isinstance(tp, potential.linearPotential) or isinstance(
            tp, potential.planarPotential
        ):
            continue
        tpp = tp.toPlanar()
        assert isinstance(tpp, potential.planarPotential), (
            "Conversion into planar potential of potential %s fails" % p
        )
        tlp = tp.toVertical(1.0, phi=2.0)
        assert isinstance(tlp, potential.linearPotential), (
            "Conversion into linear potential of potential %s fails" % p
        )


def test_RZToplanarPotential():
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    plp = potential.RZToplanarPotential(lp)
    assert isinstance(
        plp, potential.planarPotential
    ), "Running an RZPotential through RZToplanarPotential does not produce a planarPotential"
    # Check that a planarPotential through RZToplanarPotential is still planar
    pplp = potential.RZToplanarPotential(plp)
    assert isinstance(
        pplp, potential.planarPotential
    ), "Running a planarPotential through RZToplanarPotential does not produce a planarPotential"
    # Check that a list with a mix of planar and 3D potentials produces list of planar
    ppplp = potential.RZToplanarPotential([lp, plp])
    for p in ppplp:
        assert isinstance(
            p, potential.planarPotential
        ), "Running a list with a mix of planar and 3D potentials through RZToPlanarPotential does not produce a list of planar potentials"
    # Check that giving an object that is not a list or Potential instance produces an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToplanarPotential("something else")
    # Check that given a list of objects that are not a Potential instances gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToplanarPotential([3, 4, 45])
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToplanarPotential([lp, 3, 4, 45])
    # Check that using a non-axisymmetric potential gives an error
    lpna = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, b=0.8)
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToplanarPotential(lpna)
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToplanarPotential([lpna])
    # Check that giving potential.ChandrasekharDynamicalFrictionForce
    # gives an error
    pp = potential.PlummerPotential(amp=1.12, b=2.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=pp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.RZToplanarPotential([pp, cdfc])
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.RZToplanarPotential(cdfc)
    return None


def test_toPlanarPotential():
    tnp = potential.TriaxialNFWPotential(normalize=1.0, b=0.5)
    ptnp = potential.toPlanarPotential(tnp)
    assert isinstance(
        ptnp, potential.planarPotential
    ), "Running a non-axisymmetric Potential through toPlanarPotential does not produce a planarPotential"
    # Also for list
    ptnp = potential.toPlanarPotential([tnp])
    assert isinstance(
        ptnp[0], potential.planarPotential
    ), "Running a non-axisymmetric Potential through toPlanarPotential does not produce a planarPotential"
    # Check that a planarPotential through toPlanarPotential is still planar
    pptnp = potential.toPlanarPotential(tnp)
    assert isinstance(
        pptnp, potential.planarPotential
    ), "Running a planarPotential through toPlanarPotential does not produce a planarPotential"
    # Check that running potential.NonInertialFrameforce through works
    nip = potential.NonInertialFrameForce(Omega=numpy.array([0.0, 1.0, 0.0]))
    assert isinstance(
        potential.toPlanarPotential(nip), potential.planarForce
    ), "Running a potential.NonInertialFrameForce through toPlanarPotential does not produce a planarDissipativeForce"

    try:
        ptnp = potential.toPlanarPotential("something else")
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "Using toPlanarPotential with a string rather than an Potential or a planarPotential did not raise PotentialError"
        )
    # Check that list of objects that are not potentials gives error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.toPlanarPotential([3, 4, 45])
    return None


def test_RZToverticalPotential():
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    plp = potential.RZToverticalPotential(lp, 1.2)
    assert isinstance(
        plp, potential.linearPotential
    ), "Running an RZPotential through RZToverticalPotential does not produce a linearPotential"
    # Check that a verticalPotential through RZToverticalPotential is still vertical
    pplp = potential.RZToverticalPotential(plp, 1.2)
    assert isinstance(
        pplp, potential.linearPotential
    ), "Running a linearPotential through RZToverticalPotential does not produce a linearPotential"
    # Also for list
    pplp = potential.RZToverticalPotential([plp], 1.2)
    assert isinstance(
        pplp[0], potential.linearPotential
    ), "Running a linearPotential through RZToverticalPotential does not produce a linearPotential"
    # Check that giving an object that is not a list or Potential instance produces an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential("something else", 1.2)
    # Check that given a list of objects that are not a Potential instances gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential([3, 4, 45], 1.2)
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential([lp, 3, 4, 45], 1.2)
    # Check that giving a planarPotential gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential(lp.toPlanar(), 1.2)
    # Check that giving a list of planarPotential gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential([lp.toPlanar()], 1.2)
    # Check that using a non-axisymmetric potential gives an error
    lpna = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, b=0.8)
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential(lpna, 1.2)
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.RZToverticalPotential([lpna], 1.2)
    # Check that giving potential.ChandrasekharDynamicalFrictionForce
    # gives an error
    pp = potential.PlummerPotential(amp=1.12, b=2.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=pp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.RZToverticalPotential([pp, cdfc], 1.2)
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.RZToverticalPotential(cdfc, 1.2)
    return None


def test_toVerticalPotential():
    tnp = potential.TriaxialNFWPotential(normalize=1.0, b=0.5)
    ptnp = potential.toVerticalPotential(tnp, 1.2, phi=0.8)
    assert isinstance(
        ptnp, potential.linearPotential
    ), "Running a non-axisymmetric Potential through toVerticalPotential does not produce a linearPotential"
    # Also for list
    ptnp = potential.toVerticalPotential([tnp], 1.2, phi=0.8)
    assert isinstance(
        ptnp[0], potential.linearPotential
    ), "Running a non-axisymmetric Potential through toVerticalPotential does not produce a linearPotential"
    # Check that a linearPotential through toVerticalPotential is still vertical
    ptnp = potential.toVerticalPotential(tnp, 1.2, phi=0.8)
    pptnp = potential.toVerticalPotential(ptnp, 1.2, phi=0.8)
    assert isinstance(
        pptnp, potential.linearPotential
    ), "Running a linearPotential through toVerticalPotential does not produce a linearPotential"
    # also for list
    pptnp = potential.toVerticalPotential([ptnp], 1.2, phi=0.8)
    assert isinstance(
        pptnp[0], potential.linearPotential
    ), "Running a linearPotential through toVerticalPotential does not produce a linearPotential"
    try:
        ptnp = potential.toVerticalPotential("something else", 1.2, phi=0.8)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "Using toVerticalPotential with a string rather than an Potential or a linearPotential did not raise PotentialError"
        )
    # Check that giving a planarPotential gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.toVerticalPotential(tnp.toPlanar(), 1.2, phi=0.8)
    # Check that giving a list of planarPotential gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.toVerticalPotential([tnp.toPlanar()], 1.2, phi=0.8)
    # Check that giving a list of non-potentials gives error
    with pytest.raises(potential.PotentialError) as excinfo:
        plp = potential.toVerticalPotential([3, 4, 45], 1.2)
    # Check that giving potential.ChandrasekharDynamicalFrictionForce
    # gives an error
    pp = potential.PlummerPotential(amp=1.12, b=2.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=pp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.toVerticalPotential([pp, cdfc], 1.2, phi=0.8)
    with pytest.raises(NotImplementedError) as excinfo:
        plp = potential.toVerticalPotential(cdfc, 1.2, phi=0.8)
    # Check that running a non-axisymmetric potential through toVertical w/o
    # phi gives an error
    with pytest.raises(potential.PotentialError) as excinfo:
        ptnp = potential.toVerticalPotential(tnp, 1.2)
    return None


# Sanity check the derivative of the rotation curve and the frequencies in the plane
def test_dvcircdR_omegac_epifreq_rl_vesc():
    # Derivative of rotation curve
    # LogarithmicHaloPotential: rotation everywhere flat
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    assert (
        lp.dvcircdR(1.0) ** 2.0 < 10.0**-16.0
    ), "LogarithmicHaloPotential's rotation curve is not flat at R=1"
    assert (
        lp.dvcircdR(0.5) ** 2.0 < 10.0**-16.0
    ), "LogarithmicHaloPotential's rotation curve is not flat at R=0.5"
    assert (
        lp.dvcircdR(2.0) ** 2.0 < 10.0**-16.0
    ), "LogarithmicHaloPotential's rotation curve is not flat at R=2"
    # Kepler potential, vc = vc_0(R/R0)^-0.5 -> dvcdR= -0.5 vc_0 (R/R0)**-1.5
    kp = potential.KeplerPotential(normalize=1.0)
    assert (
        (kp.dvcircdR(1.0) + 0.5) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's rotation curve is not what it should be at R=1"
    assert (
        (kp.dvcircdR(0.5) + 0.5**-0.5) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's rotation curve is not what it should be at R=0.5"
    assert (
        (kp.dvcircdR(2.0) + 0.5**2.5) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's rotation curve is not what it should be at R=2"
    # Rotational frequency
    assert (
        lp.omegac(1.0) - 1.0
    ) ** 2.0 < 10.0**-16.0, "LogarithmicHalo's rotational frequency is off at R=1"
    assert (
        lp.omegac(0.5) - 2.0
    ) ** 2.0 < 10.0**-16.0, "LogarithmicHalo's rotational frequency is off at R=0.5"
    assert (
        lp.omegac(2.0) - 0.5
    ) ** 2.0 < 10.0**-16.0, "LogarithmicHalo's rotational frequency is off at R=2"
    assert (
        (lp.toPlanar().omegac(2.0) - 0.5) ** 2.0 < 10.0** -16.0
    ), "LogarithmicHalo's rotational frequency is off at R=2 through planarPotential"
    # Epicycle frequency, flat rotation curve
    assert (
        (lp.epifreq(1.0) - numpy.sqrt(2.0) * lp.omegac(1.0)) ** 2.0 < 10.0** -16.0
    ), "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=1"
    assert (
        (lp.epifreq(0.5) - numpy.sqrt(2.0) * lp.omegac(0.5)) ** 2.0 < 10.0** -16.0
    ), "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=0.5"
    assert (
        (lp.epifreq(2.0) - numpy.sqrt(2.0) * lp.omegac(2.0)) ** 2.0 < 10.0** -16.0
    ), "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=2"
    assert (
        (lp.toPlanar().epifreq(2.0) - numpy.sqrt(2.0) * lp.omegac(2.0)) ** 2.0
        < 10.0** -16.0
    ), "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=, through planar2"
    # Epicycle frequency, Kepler
    assert (
        (kp.epifreq(1.0) - kp.omegac(1.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=1"
    assert (
        (kp.epifreq(0.5) - kp.omegac(0.5)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=0.5"
    assert (
        (kp.epifreq(2.0) - kp.omegac(2.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=2"
    # Check radius of circular orbit, Kepler
    assert (
        (kp.rl(1.0) - 1.0) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's radius of a circular orbit is wrong at Lz=1."
    assert (
        (kp.rl(0.5) - 1.0 / 4.0) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's radius of a circular orbit is wrong at Lz=0.5"
    assert (
        (kp.rl(2.0) - 4.0) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's radius of a circular orbit is wrong at Lz=2."
    # Check radius of circular orbit, PowerSphericalPotential with close-to-flat rotation curve
    pp = potential.PowerSphericalPotential(alpha=1.8, normalize=1.0)
    assert (
        (pp.rl(1.0) - 1.0) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=1."
    assert (
        (pp.rl(0.5) - 0.5 ** (10.0 / 11.0)) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=0.5"
    assert (
        (pp.rl(2.0) - 2.0 ** (10.0 / 11.0)) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=2."
    # Check radius of circular orbit, PowerSphericalPotential with steeper rotation curve
    pp = potential.PowerSphericalPotential(alpha=0.5, normalize=1.0)
    assert (
        (pp.rl(1.0) - 1.0) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=1."
    assert (
        (pp.rl(0.0625) - 0.0625 ** (4.0 / 7.0)) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=0.0625"
    assert (
        (pp.rl(16.0) - 16.0 ** (4.0 / 7.0)) ** 2.0 < 10.0** -16.0
    ), "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=16."
    # Check radius in MWPotential2014 at very small lz, to test small lz behavior
    lz = 0.000001
    assert (
        numpy.fabs(
            potential.vcirc(
                potential.MWPotential2014, potential.rl(potential.MWPotential2014, lz)
            )
            * potential.rl(potential.MWPotential2014, lz)
            - lz
        )
        < 1e-12
    ), "Radius of circular orbit at small Lz in MWPotential2014 does not work as expected"
    # Escape velocity of Kepler potential
    assert (
        kp.vesc(1.0) ** 2.0 - 2.0
    ) ** 2.0 < 10.0**-16.0, "KeplerPotential's escape velocity is wrong at R=1"
    assert (
        kp.vesc(0.5) ** 2.0 - 2.0 * kp.vcirc(0.5) ** 2.0
    ) ** 2.0 < 10.0**-16.0, "KeplerPotential's escape velocity is wrong at R=0.5"
    assert (
        kp.vesc(2.0) ** 2.0 - 2.0 * kp.vcirc(2.0) ** 2.0
    ) ** 2.0 < 10.0**-16.0, "KeplerPotential's escape velocity is wrong at R=2"
    assert (
        (kp.toPlanar().vesc(2.0) ** 2.0 - 2.0 * kp.vcirc(2.0) ** 2.0) ** 2.0
        < 10.0** -16.0
    ), "KeplerPotential's escape velocity is wrong at R=2, through planar"
    # W/ different interface
    assert (
        (kp.vcirc(1.0) - potential.vcirc(kp, 1.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's circular velocity does not agree between kp.vcirc and vcirc(kp)"
    assert (
        (kp.vcirc(1.0) - potential.vcirc(kp.toPlanar(), 1.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's circular velocity does not agree between kp.vcirc and vcirc(kp.toPlanar)"
    assert (
        (kp.vesc(1.0) - potential.vesc(kp, 1.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's escape velocity does not agree between kp.vesc and vesc(kp)"
    assert (
        (kp.vesc(1.0) - potential.vesc(kp.toPlanar(), 1.0)) ** 2.0 < 10.0** -16.0
    ), "KeplerPotential's escape velocity does not agree between kp.vesc and vesc(kp.toPlanar)"
    return None


def test_vcirc_phi_axi():
    # Test that giving phi to vcirc for an axisymmetric potential doesn't
    # affect the answer
    kp = potential.KeplerPotential(normalize=1.0)
    phis = numpy.linspace(0.0, numpy.pi, 101)
    vcs = numpy.array([kp.vcirc(1.0, phi) for phi in phis])
    assert numpy.all(
        numpy.fabs(vcs - 1.0) < 10.0**-8.0
    ), "Setting phi= in vcirc for axisymmetric potential gives different answers for different phi"
    # One at a different radius
    R = 0.5
    vcs = numpy.array([kp.vcirc(R, phi) for phi in phis])
    assert numpy.all(
        numpy.fabs(vcs - kp.vcirc(R)) < 10.0**-8.0
    ), "Setting phi= in vcirc for axisymmetric potential gives different answers for different phi"
    return None


def test_vcirc_phi_nonaxi():
    # Test that giving phi to vcirc for a non-axisymmetric potential does
    # affect the answer
    tnp = potential.TriaxialNFWPotential(b=0.4, normalize=1.0)
    # limited phi range
    phis = numpy.linspace(numpy.pi / 5.0, numpy.pi / 2.0, 5)
    vcs = numpy.array([tnp.vcirc(1.0, phi) for phi in phis])
    assert numpy.all(
        numpy.fabs(vcs - 1.0) > 0.01
    ), "Setting phi= in vcirc for axisymmetric potential does not give different answers for different phi"
    # One at a different radius
    R = 0.5
    vcs = numpy.array([tnp.vcirc(R, phi) for phi in phis])
    assert numpy.all(
        numpy.fabs(vcs - tnp.vcirc(R, phi=0.0)) > 0.01
    ), "Setting phi= in vcirc for axisymmetric potential does not give different answers for different phi"
    return None


def test_vcirc_vesc_special():
    # Test some special cases of vcirc and vesc
    dp = potential.EllipticalDiskPotential()
    try:
        potential.plotRotcurve([dp])
    except (AttributeError, potential.PotentialError):  # should be raised
        pass
    else:
        raise AssertionError(
            "plotRotcurve for non-axisymmetric potential should have raised AttributeError, but didn't"
        )
    try:
        potential.plotEscapecurve([dp])
    except AttributeError:  # should be raised
        pass
    else:
        raise AssertionError(
            "plotEscapecurve for non-axisymmetric potential should have raised AttributeError, but didn't"
        )
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    assert (
        numpy.fabs(potential.calcRotcurve(lp, 0.8) - lp.vcirc(0.8)) < 10.0**-16.0
    ), "Circular velocity calculated with calcRotcurve not the same as that calculated with vcirc"
    assert (
        numpy.fabs(potential.calcEscapecurve(lp, 0.8) - lp.vesc(0.8)) < 10.0**-16.0
    ), "Escape velocity calculated with calcEscapecurve not the same as that calculated with vcirc"
    return None


def test_lindbladR():
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    assert (
        numpy.fabs(lp.lindbladR(0.5, "corotation") - 2.0) < 10.0**-10.0
    ), "Location of co-rotation resonance is wrong for LogarithmicHaloPotential"
    assert (
        numpy.fabs(
            lp.omegac(lp.lindbladR(0.5, 2)) - 2.0 / (2.0 - numpy.sqrt(2.0)) * 0.5
        )
        < 10.0**-14.0
    ), "Location of m=2 resonance is wrong for LogarithmicHaloPotential"
    assert (
        numpy.fabs(
            lp.omegac(lp.lindbladR(0.5, -2)) + 2.0 / (-2.0 - numpy.sqrt(2.0)) * 0.5
        )
        < 10.0**-14.0
    ), "Location of m=-2 resonance is wrong for LogarithmicHaloPotential"
    # Also through general interface
    assert (
        numpy.fabs(
            lp.omegac(potential.lindbladR(lp, 0.5, -2))
            + 2.0 / (-2.0 - numpy.sqrt(2.0)) * 0.5
        )
        < 10.0**-14.0
    ), "Location of m=-2 resonance is wrong for LogarithmicHaloPotential"
    # Also for planar
    assert (
        numpy.fabs(
            lp.omegac(lp.toPlanar().lindbladR(0.5, -2))
            + 2.0 / (-2.0 - numpy.sqrt(2.0)) * 0.5
        )
        < 10.0**-14.0
    ), "Location of m=-2 resonance is wrong for LogarithmicHaloPotential"
    # Test non-existent ones
    mp = potential.MiyamotoNagaiPotential(normalize=1.0, a=0.3)
    assert (
        mp.lindbladR(3.0, 2) is None
    ), "MiyamotoNagai w/ OmegaP=3 should not have a inner m=2 LindbladR"
    assert (
        mp.lindbladR(6.0, "corotation") is None
    ), "MiyamotoNagai w/ OmegaP=6 should not have a inner m=2 LindbladR"
    # Test error
    try:
        lp.lindbladR(0.5, "wrong resonance")
    except OSError:
        pass
    else:
        raise AssertionError(
            "lindbladR w/ wrong m input should have raised IOError, but didn't"
        )
    return None


def test_rE_flatvc():
    # Test the rE function for the case of a flat rotation curve
    # Expected rE when vc(1)=1 is exp(E-1/2) (e.g., Dehnen 1999 epicycle)
    lp = potential.LogarithmicHaloPotential(normalize=1.0)

    def expected_rE(E):
        return numpy.exp(E - 0.5)

    Es = numpy.linspace(-10.0, 20.0, 101)
    rEs = numpy.array([lp.rE(E) for E in Es])
    assert (
        numpy.amax(numpy.fabs(rEs - expected_rE(Es))) < 1e-6
    ), "rE method does not give the expected result for a flat rotation curve"
    # Also as function
    rEs = numpy.array([potential.rE(lp, E) for E in Es])
    assert (
        numpy.amax(numpy.fabs(rEs - expected_rE(Es))) < 1e-6
    ), "rE method does not give the expected result for a flat rotation curve"
    return None


def test_rE_powervc():
    # Test the rE function for the case of a power-law rotation curve: v = r^beta
    # Expected rE when vc(1)=1 is (2 beta E / [1+beta])**(1./[2beta])
    # (e.g., Dehnen 1999 epicycle)
    betas = [-0.45, -0.2, 0.6, 0.9]

    def expected_rE(E, beta):
        return (2.0 * beta * E / (1.0 + beta)) ** (1.0 / 2.0 / beta)

    for beta in betas:
        pp = PowerSphericalPotential(alpha=2.0 - 2.0 * beta, normalize=1.0)
        rmin, rmax = 1e-8, 1e5
        Emin = pp.vcirc(rmin) ** 2.0 / 2.0 + pp(rmin, 0.0)
        Emax = pp.vcirc(rmax) ** 2.0 / 2.0 + pp(rmax, 0.0)
        Es = numpy.linspace(Emin, Emax, 101)
        # Test both method and function
        if beta < 0.0:
            rEs = numpy.array([pp.rE(E) for E in Es])
        else:
            rEs = numpy.array([potential.rE(pp, E) for E in Es])
        assert (
            numpy.amax(numpy.fabs(rEs - expected_rE(Es, beta))) < 1e-8
        ), "rE method does not give the expected result for a power-law rotation curve"
    return None


def test_rE_MWPotential2014():
    # Test the rE function for MWPotential2014
    # No closed-form true answer, so just check that the expected relation holds
    def Ec(r):
        return potential.vcirc(
            potential.MWPotential2014, r
        ) ** 2.0 / 2.0 + potential.evaluatePotentials(potential.MWPotential2014, r, 0.0)

    rmin, rmax = 1e-8, 1e5
    Emin = Ec(rmin)
    Emax = Ec(rmax)
    Es = numpy.linspace(Emin, Emax, 101)
    rEs = numpy.array([potential.rE(potential.MWPotential2014, E) for E in Es])
    Ecs = numpy.array([Ec(rE) for rE in rEs])
    assert (
        numpy.amax(numpy.fabs(Ecs - Es)) < 1e-8
    ), "rE method does not give the expected result for MWPotential2014"
    return None


def test_LcE_flatvc():
    # Test the LcE function for the case of a flat rotation curve
    # Expected LcE when vc(1)=1 is exp(E-1/2) (e.g., Dehnen 1999 epicycle)
    lp = potential.LogarithmicHaloPotential(normalize=1.0)

    def expected_LcE(E):
        return numpy.exp(E - 0.5)

    Es = numpy.linspace(-10.0, 20.0, 101)
    LcEs = numpy.array([lp.LcE(E) for E in Es])
    assert (
        numpy.amax(numpy.fabs(LcEs - expected_LcE(Es))) < 1e-6
    ), "LcE method does not give the expected result for a flat rotation curve"
    # Also as function
    LcEs = numpy.array([potential.LcE(lp, E) for E in Es])
    assert (
        numpy.amax(numpy.fabs(LcEs - expected_LcE(Es))) < 1e-6
    ), "LcE method does not give the expected result for a flat rotation curve"
    return None


def test_LcE_powervc():
    # Test the LcE function for the case of a power-law rotation curve: v = r^beta
    # Expected LcE when vc(1)=1 is (2 beta E / [1+beta])**([1.+beta]/[2beta])
    # (e.g., Dehnen 1999 epicycle)
    betas = [-0.45, -0.2, 0.6, 0.9]

    def expected_LcE(E, beta):
        return (2.0 * beta * E / (1.0 + beta)) ** ((1.0 + beta) / 2.0 / beta)

    for beta in betas:
        pp = PowerSphericalPotential(alpha=2.0 - 2.0 * beta, normalize=1.0)
        rmin, rmax = 1e-8, 1e5
        Emin = pp.vcirc(rmin) ** 2.0 / 2.0 + pp(rmin, 0.0)
        Emax = pp.vcirc(rmax) ** 2.0 / 2.0 + pp(rmax, 0.0)
        Es = numpy.linspace(Emin, Emax, 101)
        # Test both method and function
        if beta < 0.0:
            LcEs = numpy.array([pp.LcE(E) for E in Es])
        else:
            LcEs = numpy.array([potential.LcE(pp, E) for E in Es])
        assert (
            numpy.amax(numpy.fabs(LcEs - expected_LcE(Es, beta))) < 1e-5
        ), "LcE method does not give the expected result for a power-law rotation curve"
    return None


def test_vterm():
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    assert (
        numpy.fabs(lp.vterm(30.0, deg=True) - 0.5 * (lp.omegac(0.5) - 1.0))
        < 10.0**-10.0
    ), "vterm for LogarithmicHaloPotential at l=30 is incorrect"
    assert (
        numpy.fabs(
            lp.vterm(numpy.pi / 3.0, deg=False)
            - numpy.sqrt(3.0) / 2.0 * (lp.omegac(numpy.sqrt(3.0) / 2.0) - 1.0)
        )
        < 10.0**-10.0
    ), "vterm for LogarithmicHaloPotential at l=60 in rad is incorrect"
    # Also using general interface
    assert (
        numpy.fabs(potential.vterm(lp, 30.0, deg=True) - 0.5 * (lp.omegac(0.5) - 1.0))
        < 10.0**-10.0
    ), "vterm for LogarithmicHaloPotential at l=30 is incorrect"
    assert (
        numpy.fabs(
            potential.vterm(lp, numpy.pi / 3.0, deg=False)
            - numpy.sqrt(3.0) / 2.0 * (lp.omegac(numpy.sqrt(3.0) / 2.0) - 1.0)
        )
        < 10.0**-10.0
    ), "vterm for LogarithmicHaloPotential at l=60 in rad is incorrect"
    return None


def test_flattening():
    # Simple tests: LogarithmicHalo
    qs = [0.75, 1.0, 1.25]
    for q in qs:
        lp = potential.LogarithmicHaloPotential(normalize=1.0, q=q)
        assert (lp.flattening(1.0, 0.001) - q) ** 2.0 < 10.0**-16.0, (
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (1.,0.001)"
            % q
        )
        assert (lp.flattening(1.0, 0.1) - q) ** 2.0 < 10.0**-16.0, (
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (1.,0.1)"
            % q
        )
        assert (lp.flattening(0.5, 0.001) - q) ** 2.0 < 10.0**-16.0, (
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.001)"
            % q
        )
        assert (lp.flattening(0.5, 0.1) - q) ** 2.0 < 10.0**-16.0, (
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.1)"
            % q
        )
        # One test with the general interface
        assert (potential.flattening(lp, 0.5, 0.1) - q) ** 2.0 < 10.0**-16.0, (
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.1), through potential.flattening"
            % q
        )
    # Check some spherical potentials
    kp = potential.KeplerPotential(normalize=1.0)
    assert (
        (kp.flattening(1.0, 0.02) - 1.0) ** 2.0 < 10.0** -16.0
    ), "Flattening of KeplerPotential is not equal to 1 at (R,z) = (1.,0.02)"
    np = potential.NFWPotential(normalize=1.0, a=5.0)
    assert (
        (np.flattening(1.0, 0.02) - 1.0) ** 2.0 < 10.0** -16.0
    ), "Flattening of NFWPotential is not equal to 1 at (R,z) = (1.,0.02)"
    hp = potential.HernquistPotential(normalize=1.0, a=5.0)
    assert (
        (hp.flattening(1.0, 0.02) - 1.0) ** 2.0 < 10.0** -16.0
    ), "Flattening of HernquistPotential is not equal to 1 at (R,z) = (1.,0.02)"
    # Disk potentials should be oblate everywhere
    mp = potential.MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    assert (
        mp.flattening(1.0, 0.1) <= 1.0
    ), "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (1.,0.1)"
    assert (
        mp.flattening(1.0, 2.0) <= 1.0
    ), "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (1.,2.)"
    assert (
        mp.flattening(3.0, 3.0) <= 1.0
    ), "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (3.,3.)"
    return None


def test_verticalfreq():
    # For spherical potentials, vertical freq should be equal to rotational freq
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    kp = potential.KeplerPotential(normalize=1.0)
    np = potential.NFWPotential(normalize=1.0)
    bp = potential.BurkertPotential(normalize=1.0)
    rs = numpy.linspace(0.2, 2.0, 21)
    for r in rs:
        assert (
            numpy.fabs(lp.verticalfreq(r) - lp.omegac(r)) < 10.0**-10.0
        ), "Verticalfreq for spherical potential does not equal rotational freq"
        assert (
            numpy.fabs(kp.verticalfreq(r) - kp.omegac(r)) < 10.0**-10.0
        ), "Verticalfreq for spherical potential does not equal rotational freq"
        # Through general interface
        assert (
            numpy.fabs(potential.verticalfreq(np, r) - np.omegac(r)) < 10.0**-10.0
        ), "Verticalfreq for spherical potential does not equal rotational freq"
        assert (
            numpy.fabs(potential.verticalfreq([bp], r) - bp.omegac(r)) < 10.0**-10.0
        ), "Verticalfreq for spherical potential does not equal rotational freq"
    # For Double-exponential disk potential, epi^2+vert^2-2*rot^2 =~ 0 at very large distances (no longer explicitly, because we don't use a Kepler potential anylonger)
    if True:
        dp = potential.DoubleExponentialDiskPotential(normalize=1.0, hr=0.05, hz=0.01)
        assert (
            numpy.fabs(
                dp.epifreq(1.0) ** 2.0
                + dp.verticalfreq(1.0) ** 2.0
                - 2.0 * dp.omegac(1.0) ** 2.0
            )
            < 10.0**-4.0
        ), "epi^2+vert^2-2*rot^2 !=~ 0 for dblexp potential, very far from center"
        # Closer to the center, this becomes the Poisson eqn.
        assert (
            numpy.fabs(
                dp.epifreq(0.125) ** 2.0
                + dp.verticalfreq(0.125) ** 2.0
                - 2.0 * dp.omegac(0.125) ** 2.0
                - 4.0 * numpy.pi * dp.dens(0.125, 0.0)
            )
            / 4.0
            / numpy.pi
            / dp.dens(0.125, 0.0)
            < 10.0**-3.0
        ), "epi^2+vert^2-2*rot^2 !=~ dens for dblexp potential"
    return None


def test_planar_nonaxi():
    dp = potential.EllipticalDiskPotential()
    try:
        potential.evaluateplanarPotentials(dp, 1.0)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "evaluateplanarPotentials for non-axisymmetric potential w/o specifying phi did not raise PotentialError"
        )
    try:
        potential.evaluateplanarRforces(dp, 1.0)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "evaluateplanarRforces for non-axisymmetric potential w/o specifying phi did not raise PotentialError"
        )
    try:
        potential.evaluateplanarphitorques(dp, 1.0)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "evaluateplanarphitorques for non-axisymmetric potential w/o specifying phi did not raise PotentialError"
        )
    try:
        potential.evaluateplanarR2derivs(dp, 1.0)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "evaluateplanarR2derivs for non-axisymmetric potential w/o specifying phi did not raise PotentialError"
        )
    return None


def test_ExpDisk_special():
    # Test some special cases for the ExponentialDisk potentials
    # Test that array input works
    dp = potential.DoubleExponentialDiskPotential(normalize=1.0)
    rs = numpy.linspace(0.1, 2.11)
    zs = numpy.ones_like(rs) * 0.1
    # Potential itself
    dpevals = numpy.array([dp(r, z) for (r, z) in zip(rs, zs)])
    assert numpy.all(
        numpy.fabs(dp(rs, zs) - dpevals) < 10.0**-10.0
    ), "DoubleExppnentialDiskPotential evaluation does not work as expected for array inputs"
    # Rforce
    # dpevals= numpy.array([dp.Rforce(r,z) for (r,z) in zip(rs,zs)])
    # assert numpy.all(numpy.fabs(dp.Rforce(rs,zs)-dpevals) < 10.**-10.), \
    #    'DoubleExppnentialDiskPotential Rforce evaluation does not work as expected for array inputs'
    # zforce
    # dpevals= numpy.array([dp.zforce(r,z) for (r,z) in zip(rs,zs)])
    # assert numpy.all(numpy.fabs(dp.zforce(rs,zs)-dpevals) < 10.**-10.), \
    #    'DoubleExppnentialDiskPotential zforce evaluation does not work as expected for array inputs'
    # R2deriv
    # dpevals= numpy.array([dp.R2deriv(r,z) for (r,z) in zip(rs,zs)])
    # assert numpy.all(numpy.fabs(dp.R2deriv(rs,zs)-dpevals) < 10.**-10.), \
    #    'DoubleExppnentialDiskPotential R2deriv evaluation does not work as expected for array inputs'
    # z2deriv
    # dpevals= numpy.array([dp.z2deriv(r,z) for (r,z) in zip(rs,zs)])
    # assert numpy.all(numpy.fabs(dp.z2deriv(rs,zs)-dpevals) < 10.**-10.), \
    #    'DoubleExppnentialDiskPotential z2deriv evaluation does not work as expected for array inputs'
    # Rzderiv
    # dpevals= numpy.array([dp.Rzderiv(r,z) for (r,z) in zip(rs,zs)])
    # assert numpy.all(numpy.fabs(dp.Rzderiv(rs,zs)-dpevals) < 10.**-10.), \
    #    'DoubleExppnentialDiskPotential Rzderiv evaluation does not work as expected for array inputs'
    # Check the PotentialError for z=/=0 evaluation of R2deriv of RazorThinDiskPotential
    rp = potential.RazorThinExponentialDiskPotential(normalize=1.0)
    try:
        rp.R2deriv(1.0, 0.1)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "RazorThinExponentialDiskPotential's R2deriv did not raise AttributeError for z=/= 0 input"
        )
    return None


def test_DehnenBar_special():
    # Test some special cases for the DehnenBar potentials
    # Test that array input works
    dp = potential.DehnenBarPotential()
    # Test from rs < rb through to rs > rb
    rs = numpy.linspace(0.1 * dp._rb, 2.11 * dp._rb)
    zs = numpy.ones_like(rs) * 0.1
    phis = numpy.ones_like(rs) * 0.1
    # Potential itself
    dpevals = numpy.array([dp(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential evaluation does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential evaluation does not work as expected for array inputs"
    # Rforce
    dpevals = numpy.array([dp.Rforce(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rforce(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rforce evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.Rforce(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rforce(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rforce does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.Rforce(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rforce(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rforce does not work as expected for array inputs"
    # zforce
    dpevals = numpy.array([dp.zforce(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.zforce(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential zforce evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.zforce(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.zforce(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential zforce does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.zforce(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.zforce(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential zforce does not work as expected for array inputs"
    # phitorque
    dpevals = numpy.array(
        [dp.phitorque(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phitorque(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential zforce evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.phitorque(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phitorque(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phitorque does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.phitorque(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phitorque(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phitorque does not work as expected for array inputs"
    # R2deriv
    dpevals = numpy.array([dp.R2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.R2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential R2deriv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.R2deriv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.R2deriv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential R2deriv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.R2deriv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.R2deriv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential R2deriv does not work as expected for array inputs"
    # z2deriv
    dpevals = numpy.array([dp.z2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.z2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential z2deriv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.z2deriv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.z2deriv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential z2deriv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.z2deriv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.z2deriv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential z2deriv does not work as expected for array inputs"
    # phi2deriv
    dpevals = numpy.array(
        [dp.phi2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phi2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential z2deriv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.phi2deriv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phi2deriv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phi2deriv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.phi2deriv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phi2deriv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phi2deriv does not work as expected for array inputs"
    # Rzderiv
    dpevals = numpy.array([dp.Rzderiv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rzderiv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rzderiv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.Rzderiv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rzderiv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rzderiv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.Rzderiv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rzderiv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rzderiv does not work as expected for array inputs"
    # Rphideriv
    dpevals = numpy.array(
        [dp.Rphideriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.Rphideriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rphideriv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.Rphideriv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rphideriv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rphideriv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.Rphideriv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rphideriv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential Rphideriv does not work as expected for array inputs"
    # phizderiv
    dpevals = numpy.array(
        [dp.phizderiv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phizderiv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phizderiv evaluation does not work as expected for array inputs"
    # R array, z not an array
    dpevals = numpy.array([dp.phizderiv(r, zs[0], phi) for (r, phi) in zip(rs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phizderiv(rs, zs[0], phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phizderiv does not work as expected for array inputs"
    # z array, R not an array
    dpevals = numpy.array([dp.phizderiv(rs[0], z, phi) for (z, phi) in zip(zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.phizderiv(rs[0], zs, phis) - dpevals) < 10.0**-10.0
    ), "DehnenBarPotential phizderiv does not work as expected for array inputs"
    return None


def test_SpiralArm_special():
    # Test some special cases for the DehnenBar potentials
    # Test that array input works
    dp = potential.SpiralArmsPotential()
    rs = numpy.linspace(0.1, 2.0, 11)
    zs = numpy.ones_like(rs) * 0.1
    phis = numpy.ones_like(rs) * 0.1
    # Potential itself
    dpevals = numpy.array([dp(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential evaluation does not work as expected for array inputs"
    # Rforce
    dpevals = numpy.array([dp.Rforce(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rforce(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential Rforce evaluation does not work as expected for array inputs"
    # zforce
    dpevals = numpy.array([dp.zforce(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.zforce(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential zforce evaluation does not work as expected for array inputs"
    # phitorque
    dpevals = numpy.array(
        [dp.phitorque(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phitorque(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential zforce evaluation does not work as expected for array inputs"
    # R2deriv
    dpevals = numpy.array([dp.R2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.R2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential R2deriv evaluation does not work as expected for array inputs"
    # z2deriv
    dpevals = numpy.array([dp.z2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.z2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential z2deriv evaluation does not work as expected for array inputs"
    # phi2deriv
    dpevals = numpy.array(
        [dp.phi2deriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phi2deriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential z2deriv evaluation does not work as expected for array inputs"
    # Rzderiv
    dpevals = numpy.array([dp.Rzderiv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.Rzderiv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential Rzderiv evaluation does not work as expected for array inputs"
    # Rphideriv
    dpevals = numpy.array(
        [dp.Rphideriv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.Rphideriv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential Rzderiv evaluation does not work as expected for array inputs"
    # phizderiv
    dpevals = numpy.array(
        [dp.phizderiv(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)]
    )
    assert numpy.all(
        numpy.fabs(dp.phizderiv(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential Rzderiv evaluation does not work as expected for array inputs"
    # dens
    dpevals = numpy.array([dp.dens(r, z, phi) for (r, z, phi) in zip(rs, zs, phis)])
    assert numpy.all(
        numpy.fabs(dp.dens(rs, zs, phis) - dpevals) < 10.0**-10.0
    ), "SpiralArmsPotential Rzderiv evaluation does not work as expected for array inputs"
    return None


def test_MovingObject_density():
    mp = mockMovingObjectPotential()
    # Just test that the density far away from the object is close to zero
    assert (
        numpy.fabs(mp.dens(5.0, 0.0)) < 10.0**-8.0
    ), "Density far away from MovingObject is not close to zero"
    return None


# test specialSelf for TwoPowerSphericalPotential
def test_TwoPowerSphericalPotentialSpecialSelf():
    # TODO replace manual additions with an automatic method
    # that checks the signatures all methods in all potentials
    kw = dict(amp=1.0, a=1.0, normalize=False, ro=None, vo=None)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.0, 0.125, -0.125])

    pot = potential.TwoPowerSphericalPotential(alpha=0, beta=4, **kw)
    comp = potential.DehnenCoreSphericalPotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    pot = potential.TwoPowerSphericalPotential(alpha=1, beta=4, **kw)
    comp = potential.HernquistPotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    pot = potential.TwoPowerSphericalPotential(alpha=2, beta=4, **kw)
    comp = potential.JaffePotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    pot = potential.TwoPowerSphericalPotential(alpha=1, beta=3, **kw)
    comp = potential.NFWPotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    return None


def test_DehnenSphericalPotentialSpecialSelf():
    # TODO replace manual additions with an automatic method
    # that checks the signatures all methods in all potentials
    kw = dict(amp=1.0, a=1.0, normalize=False, ro=None, vo=None)
    Rs = numpy.array([0.5, 1.0, 2.0])
    Zs = numpy.array([0.0, 0.125, -0.125])

    pot = potential.DehnenSphericalPotential(alpha=0, **kw)
    comp = potential.DehnenCoreSphericalPotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))
    assert all(pot._R2deriv(Rs, Zs) == comp._R2deriv(Rs, Zs))
    assert all(pot._Rzderiv(Rs, Zs) == comp._Rzderiv(Rs, Zs))

    pot = potential.DehnenSphericalPotential(alpha=1, **kw)
    comp = potential.HernquistPotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    pot = potential.DehnenSphericalPotential(alpha=2, **kw)
    comp = potential.JaffePotential(**kw)
    assert all(pot._evaluate(Rs, Zs) == comp._evaluate(Rs, Zs))
    assert all(pot._Rforce(Rs, Zs) == comp._Rforce(Rs, Zs))
    assert all(pot._zforce(Rs, Zs) == comp._zforce(Rs, Zs))

    return None


# Test that MWPotential is what it's supposed to be
def test_MWPotential2014():
    pot = potential.MWPotential2014
    V0, R0 = 220.0, 8.0
    # Check the parameters of the bulge
    assert pot[0].rc == 1.9 / R0, "MWPotential2014's bulge cut-off radius is incorrect"
    assert (
        pot[0].alpha == 1.8
    ), "MWPotential2014's bulge power-law exponent is incorrect"
    assert (
        numpy.fabs(pot[0].Rforce(1.0, 0.0) + 0.05) < 10.0**-14.0
    ), "MWPotential2014's bulge amplitude is incorrect"
    # Check the parameters of the disk
    assert (
        numpy.fabs(pot[1]._a - 3.0 / R0) < 10.0**-14.0
    ), "MWPotential2014's disk scale length is incorrect"
    assert (
        numpy.fabs(pot[1]._b - 0.28 / R0) < 10.0**-14.0
    ), "MWPotential2014's disk scale height is incorrect"
    assert (
        numpy.fabs(pot[1].Rforce(1.0, 0.0) + 0.60) < 10.0**-14.0
    ), "MWPotential2014's disk amplitude is incorrect"
    # Check the parameters of the halo
    assert (
        numpy.fabs(pot[2].a - 16.0 / R0) < 10.0**-14.0
    ), "MWPotential2014's halo scale radius is incorrect"
    assert (
        numpy.fabs(pot[2].Rforce(1.0, 0.0) + 0.35) < 10.0**-14.0
    ), "MWPotential2014's halo amplitude is incorrect"
    return None


# Test that the McMillan17 potential is what it's supposed to be
def test_McMillan17():
    from galpy.potential.mwpotentials import McMillan17
    from galpy.util import conversion

    ro, vo = McMillan17[0]._ro, McMillan17[0]._vo
    # Check some numbers from Table 3 of McMillan17: vertical force at the Sun
    assert (
        numpy.fabs(
            -potential.evaluatezforces(McMillan17, 1.0, 1.1 / 8.21, use_physical=False)
            * conversion.force_in_2piGmsolpc2(vo, ro)
            - 73.9
        )
        < 0.2
    ), "Vertical force at the Sun in McMillan17 does not agree with what it should be"
    # Halo density at the Sun
    assert (
        numpy.fabs(
            potential.evaluateDensities(McMillan17[1], 1.0, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 0.0101
        )
        < 1e-4
    ), "Halo density at the Sun in McMillan17 does not agree with what it should be"
    # Halo concentration
    assert (
        numpy.fabs(McMillan17[1].conc(overdens=94.0, wrtcrit=True, H=70.4) - 15.4)
        < 1e-1
    ), "Halo concentration in McMillan17 does not agree with what it is supposed to be"
    # Let's compute the mass of the NFWPotenial and add the paper's number for the mass in stars and gas. The following is the total mass in units of $10^11\,M_\odot$:
    assert (
        numpy.fabs(
            (McMillan17[1].mass(50.0 / 8.21, quantity=False)) / 10.0**11.0
            + 0.543
            + 0.122
            - 5.1
        )
        < 1e-1
    ), "Mass within 50 kpc in McMillan17 does not agree with what it is supposed to be"
    # Mass of the bulge is slightly off
    assert (
        numpy.fabs((McMillan17[2].mass(50.0 / 8.21, quantity=False)) / 10.0**9.0 - 9.23)
        < 4e-1
    ), "Bulge mass in McMillan17 does not agree with what it is supposed to be"
    # Mass in stars, compute bulge+disk and subtract what's supposed to be gas
    assert (
        numpy.fabs(
            (
                McMillan17[0].mass(50.0 / 8.21, quantity=False)
                + McMillan17[2].mass(50.0 / 8.21, quantity=False)
            )
            / 10.0**10.0
            - 1.22
            - 5.43
        )
        < 1e-1
    ), "Stellar massi n McMillan17 does not agree with what it is supposed to be"
    return None


# Test that the Cautun20 potential is what it's supposed to be
def test_Cautun20():
    from galpy.potential.mwpotentials import Cautun20
    from galpy.util import conversion

    ro, vo = Cautun20[0]._ro, Cautun20[0]._vo
    # Check the rotation velocity at a few distances
    # at the Sun
    assert (
        numpy.fabs(potential.vcirc(Cautun20, 1.0, quantity=False) - 230.1) < 1e-1
    ), "Total circular velocity at the Sun in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[0], 1.0, quantity=False) - 157.6) < 1e-1
    ), "Halo circular velocity at the Sun in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[1], 1.0, quantity=False) - 151.2) < 1e-1
    ), "Disc circular velocity at the Sun in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[2], 1.0, quantity=False) - 70.8) < 1e-1
    ), "Bulge circular velocity at the Sun in Cautun20 does not agree with what it should be"
    # at 50 kpc
    assert (
        numpy.fabs(potential.vcirc(Cautun20, 50.0 / ro, quantity=False) - 184.3) < 1e-1
    ), "Total circular velocity at 50 kpc in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[0], 50.0 / ro, quantity=False) - 166.9)
        < 1e-1
    ), "Halo circular velocity at 50 kpc in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[1], 50.0 / ro, quantity=False) - 68.9)
        < 1e-1
    ), "Disc circular velocity at 50 kpc in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(potential.vcirc(Cautun20[2], 50.0 / ro, quantity=False) - 28.3)
        < 1e-1
    ), "Bulge circular velocity at 50 kpc in Cautun20 does not agree with what it should be"
    # check the enclosed halo mass
    assert (
        numpy.fabs((Cautun20[0].mass(50.0 / ro, quantity=False)) / 10.0**11 - 3.23)
        < 1e-2
    ), "DM halo mass within 50 kpc in Cautun20 does not agree with what it is supposed to be"
    assert (
        numpy.fabs((Cautun20[0].mass(200.0 / ro, quantity=False)) / 10.0**11 - 9.03)
        < 1e-2
    ), "DM halo mass within 50 kpc in Cautun20 does not agree with what it is supposed to be"
    # check the CGM density
    assert (
        numpy.fabs(
            potential.evaluateDensities(Cautun20[3], 1.0, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            * 1.0e5
            - 9.34
        )
        < 1e-2
    ), "CGM density at the Sun in Cautun20 does not agree with what it should be"
    assert (
        numpy.fabs(
            potential.evaluateDensities(Cautun20[3], 50.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            * 1.0e6
            - 6.49
        )
        < 1e-2
    ), "CGM density at 50 kpc in Cautun20 does not agree with what it should be"
    # Halo density at the Sun
    assert (
        numpy.fabs(
            potential.evaluateDensities(Cautun20[0], 1.0, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            * 1.0e3
            - 8.8
        )
        < 5e-2
    ), "Halo density at the Sun in Cautun20 does not agree with what it should be"
    return None


# Test that the Irrgang13 potentials are what they are supposed to be
def test_Irrgang13():
    from galpy.potential.mwpotentials import Irrgang13I, Irrgang13II, Irrgang13III

    # Model I
    ro, vo = Irrgang13I[0]._ro, Irrgang13I[0]._vo
    # Check some numbers from Table 1 of Irrgang13: circular velocity at the Sun
    assert (
        numpy.fabs(potential.vcirc(Irrgang13I, 1.0, quantity=False) - 242.0) < 1e-2
    ), "Circular velocity at the Sun in Irrgang13I does not agree with what it should be"
    # Mass of the bulge
    assert (
        numpy.fabs(Irrgang13I[0].mass(100.0, quantity=False) / 1e9 - 9.5) < 1e-2
    ), "Mass of the bulge in Irrgang13I does not agree with what it should be"
    # Mass of the disk
    assert (
        numpy.fabs(Irrgang13I[1].mass(100.0, 10.0, quantity=False) / 1e10 - 6.6) < 1e-2
    ), "Mass of the disk in Irrgang13I does not agree with what it should be"
    # Mass of the halo (go to edge in Irrgang13I)
    assert (
        numpy.fabs(Irrgang13I[2].mass(200.0 / ro, quantity=False) / 1e12 - 1.8) < 1e-1
    ), "Mass of the halo in Irrgang13I does not agree with what it should be"
    # Escape velocity at the Sun
    assert (
        numpy.fabs(potential.vesc(Irrgang13I, 1.0, quantity=False) - 616.4) < 1e0
    ), "Escape velocity at the Sun in Irrgang13I does not agree with what it should be"
    # Oort A
    assert (
        numpy.fabs(
            0.5
            * (
                potential.vcirc(Irrgang13I, 1.0, use_physical=False)
                - potential.dvcircdR(Irrgang13I, 1.0, use_physical=False)
            )
            * vo
            / ro
            - 15.06
        )
        < 1e-1
    ), "Oort A in Irrgang13I does not agree with what it should be"
    # Oort B
    assert (
        numpy.fabs(
            -0.5
            * (
                potential.vcirc(Irrgang13I, 1.0, use_physical=False)
                + potential.dvcircdR(Irrgang13I, 1.0, use_physical=False)
            )
            * vo
            / ro
            + 13.74
        )
        < 1e-1
    ), "Oort B in Irrgang13I does not agree with what it should be"

    # Model II
    ro, vo = Irrgang13II[0]._ro, Irrgang13II[0]._vo
    # Check some numbers from Table 2 of Irrgang13: circular velocity at the Sun
    assert (
        numpy.fabs(potential.vcirc(Irrgang13II, 1.0, quantity=False) - 240.4) < 3e-2
    ), "Circular velocity at the Sun in Irrgang13II does not agree with what it should be"
    # Mass of the bulge
    assert (
        numpy.fabs(Irrgang13II[0].mass(100.0, quantity=False) / 1e9 - 4.1) < 1e-1
    ), "Mass of the bulge in Irrgang13II does not agree with what it should be"
    # Mass of the disk
    assert (
        numpy.fabs(Irrgang13II[1].mass(100.0, 10.0, quantity=False) / 1e10 - 6.6) < 1e-1
    ), "Mass of the disk in Irrgang13II does not agree with what it should be"
    # Mass of the halo (go to edge in Irrgang13II)
    assert (
        numpy.fabs(Irrgang13II[2].mass(100.0, quantity=False) / 1e12 - 1.6) < 1e-1
    ), "Mass of the halo in Irrgang13II does not agree with what it should be"
    # Escape velocity at the Sun
    assert (
        numpy.fabs(potential.vesc(Irrgang13II, 1.0, quantity=False) - 575.9) < 1e0
    ), "Escape velocity at the Sun in Irrgang13II does not agree with what it should be"
    # Oort A
    assert (
        numpy.fabs(
            0.5
            * (
                potential.vcirc(Irrgang13II, 1.0, use_physical=False)
                - potential.dvcircdR(Irrgang13II, 1.0, use_physical=False)
            )
            * vo
            / ro
            - 15.11
        )
        < 1e-1
    ), "Oort A in Irrgang13II does not agree with what it should be"
    # Oort B
    assert (
        numpy.fabs(
            -0.5
            * (
                potential.vcirc(Irrgang13II, 1.0, use_physical=False)
                + potential.dvcircdR(Irrgang13II, 1.0, use_physical=False)
            )
            * vo
            / ro
            + 13.68
        )
        < 1e-1
    ), "Oort B in Irrgang13II does not agree with what it should be"

    # Model III
    ro, vo = Irrgang13III[0]._ro, Irrgang13III[0]._vo
    # Check some numbers from Table 3 of Irrgang13: circular velocity at the Sun
    assert (
        numpy.fabs(potential.vcirc(Irrgang13III, 1.0, quantity=False) - 239.7) < 3e-2
    ), "Circular velocity at the Sun in Irrgang13III does not agree with what it should be"
    # Mass of the bulge
    assert (
        numpy.fabs(Irrgang13III[0].mass(100.0, quantity=False) / 1e9 - 10.2) < 1e-1
    ), "Mass of the bulge in Irrgang13III does not agree with what it should be"
    # Mass of the disk
    assert (
        numpy.fabs(Irrgang13III[1].mass(100.0, 10.0, quantity=False) / 1e10 - 7.2)
        < 1e-1
    ), "Mass of the disk in Irrgang13III does not agree with what it should be"
    # Escape velocity at the Sun
    assert (
        numpy.fabs(potential.vesc(Irrgang13III, 1.0, quantity=False) - 811.5) < 1e0
    ), "Escape velocity at the Sun in Irrgang13III does not agree with what it should be"
    # Oort A
    assert (
        numpy.fabs(
            0.5
            * (
                potential.vcirc(Irrgang13III, 1.0, use_physical=False)
                - potential.dvcircdR(Irrgang13III, 1.0, use_physical=False)
            )
            * vo
            / ro
            - 14.70
        )
        < 1e-1
    ), "Oort A in Irrgang13III does not agree with what it should be"
    # Oort B
    assert (
        numpy.fabs(
            -0.5
            * (
                potential.vcirc(Irrgang13III, 1.0, use_physical=False)
                + potential.dvcircdR(Irrgang13III, 1.0, use_physical=False)
            )
            * vo
            / ro
            + 14.08
        )
        < 1e-1
    ), "Oort B in Irrgang13III does not agree with what it should be"
    return None


# Test that the Dehnen & Binney (1998) models are what they are supposed to be
def test_DehnenBinney98():
    from galpy.potential.mwpotentials import (
        DehnenBinney98I,
        DehnenBinney98II,
        DehnenBinney98III,
        DehnenBinney98IV,
    )

    check_DehnenBinney98_model(DehnenBinney98I, model="model 1")
    check_DehnenBinney98_model(DehnenBinney98II, model="model 2")
    check_DehnenBinney98_model(DehnenBinney98III, model="model 3")
    check_DehnenBinney98_model(DehnenBinney98IV, model="model 4")
    return None


def check_DehnenBinney98_model(pot, model="model 1"):
    from galpy.util import conversion

    truth = {
        "model 1": {"SigmaR0": 43.3, "vc": 222.0, "Fz": 68.0, "A": 14.4, "B": -13.3},
        "model 2": {"SigmaR0": 52.1, "vc": 217.0, "Fz": 72.2, "A": 14.3, "B": -12.9},
        "model 3": {"SigmaR0": 52.7, "vc": 217.0, "Fz": 72.5, "A": 14.1, "B": -13.1},
        "model 4": {"SigmaR0": 50.7, "vc": 220.0, "Fz": 72.1, "A": 13.8, "B": -13.6},
    }
    phys_kwargs = conversion.get_physical(pot)
    ro = phys_kwargs.get("ro")
    vo = phys_kwargs.get("vo")
    assert (
        numpy.fabs(pot[1].surfdens(1.0, 10.0 / ro) - truth[model]["SigmaR0"]) < 0.2
    ), f"Surface density at R0 in Dehnen & Binney (1998) {model} does not agree with paper value"
    assert (
        numpy.fabs(potential.vcirc(pot, 1.0) - truth[model]["vc"]) < 0.5
    ), f"Circular velocity at R0 in Dehnen & Binney (1998) {model} does not agree with paper value"
    assert (
        numpy.fabs(
            -potential.evaluatezforces(pot, 1.0, 1.1 / ro, use_physical=False)
            * conversion.force_in_2piGmsolpc2(vo, ro)
            - truth[model]["Fz"]
        )
        < 0.2
    ), f"Vertical force at R0 in Dehnen & Binney (1998) {model} does not agree with paper value"
    assert (
        numpy.fabs(
            0.5
            * (
                potential.vcirc(pot, 1.0, use_physical=False)
                - potential.dvcircdR(pot, 1.0, use_physical=False)
            )
            * vo
            / ro
            - truth[model]["A"]
        )
        < 0.05
    ), f"Oort A in Dehnen & Binney (1998) {model} does not agree with paper value"
    assert (
        numpy.fabs(
            -0.5
            * (
                potential.vcirc(pot, 1.0, use_physical=False)
                + potential.dvcircdR(pot, 1.0, use_physical=False)
            )
            * vo
            / ro
            - truth[model]["B"]
        )
        < 0.05
    ), f"Oort A in Dehnen & Binney (1998) {model} does not agree with paper value"
    return None


# Test that the virial setup of NFW works
def test_NFW_virialsetup_wrtmeanmatter():
    H, Om, overdens, wrtcrit = 71.0, 0.32, 201.0, False
    ro, vo = 220.0, 8.0
    conc, mvir = 12.0, 1.1
    np = potential.NFWPotential(
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    assert (
        numpy.fabs(conc - np.conc(H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit))
        < 10.0**-6.0
    ), "NFWPotential virial setup's concentration does not work"
    assert (
        numpy.fabs(
            mvir - np.mvir(H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit) / 10.0**12.0
        )
        < 10.0**-6.0
    ), "NFWPotential virial setup's virial mass does not work"
    return None


def test_NFW_virialsetup_wrtcrit():
    H, Om, overdens, wrtcrit = 71.0, 0.32, 201.0, True
    ro, vo = 220.0, 8.0
    conc, mvir = 12.0, 1.1
    np = potential.NFWPotential(
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    assert (
        numpy.fabs(conc - np.conc(H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit))
        < 10.0**-6.0
    ), "NFWPotential virial setup's concentration does not work"
    assert (
        numpy.fabs(
            mvir - np.mvir(H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit) / 10.0**12.0
        )
        < 10.0**-6.0
    ), "NFWPotential virial setup's virial mass does not work"
    return None


def test_TriaxialNFW_virialsetup_wrtmeanmatter():
    H, Om, overdens, wrtcrit = 71.0, 0.32, 201.0, False
    ro, vo = 220.0, 8.0
    conc, mvir = 12.0, 1.1
    np = potential.NFWPotential(
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    tnp = potential.TriaxialNFWPotential(
        b=0.3,
        c=0.7,
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    assert (
        numpy.fabs(np.a - tnp.a) < 10.0**-10.0
    ), "TriaxialNFWPotential virial setup's concentration does not work"
    assert (
        numpy.fabs(np._amp - tnp._amp * 4.0 * numpy.pi * tnp.a**3) < 10.0**-6.0
    ), "TriaxialNFWPotential virial setup's virial mass does not work"
    return None


def test_TriaxialNFW_virialsetup_wrtcrit():
    H, Om, overdens, wrtcrit = 71.0, 0.32, 201.0, True
    ro, vo = 220.0, 8.0
    conc, mvir = 12.0, 1.1
    np = potential.NFWPotential(
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    tnp = potential.TriaxialNFWPotential(
        b=0.3,
        c=0.7,
        conc=conc,
        mvir=mvir,
        vo=vo,
        ro=ro,
        H=H,
        Om=Om,
        overdens=overdens,
        wrtcrit=wrtcrit,
    )
    assert (
        numpy.fabs(np.a - tnp.a) < 10.0**-10.0
    ), "TriaxialNFWPotential virial setup's concentration does not work"
    assert (
        numpy.fabs(np._amp - tnp._amp * 4.0 * numpy.pi * tnp.a**3) < 10.0**-6.0
    ), "TriaxialNFWPotential virial setup's virial mass does not work"
    return None


# Test that setting up an NFW potential with rmax,vmax works as expected
def test_NFW_rmaxvmaxsetup():
    rmax, vmax = 1.2, 3.23
    np = potential.NFWPotential(rmax=rmax, vmax=vmax)
    assert (
        numpy.fabs(np.rmax() - rmax) < 10.0**-10.0
    ), "NFWPotential setup with rmax,vmax does not work as expected"
    assert (
        numpy.fabs(np.vmax() - vmax) < 10.0**-10.0
    ), "NFWPotential setup with rmax,vmax does not work as expected"
    return None


def test_conc_attributeerror():
    pp = potential.PowerSphericalPotential(normalize=1.0)
    # This potential doesn't have a scale, so we cannot calculate the concentration
    try:
        pp.conc(220.0, 8.0)
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "conc function for potential w/o scale did not raise AttributeError"
        )
    return None


def test_mvir_attributeerror():
    mp = potential.MiyamotoNagaiPotential(normalize=1.0)
    # Don't think I will ever implement the virial radius for this
    try:
        mp.mvir(220.0, 8.0)
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "mvir function for potential w/o rvir did not raise AttributeError"
        )
    return None


# Test that virial quantities are correctly computed when specifying a different (ro,vo) pair from Potential setup (see issue #290)
def test_NFW_virialquantities_diffrovo():
    from galpy.util import conversion

    H, Om, overdens, wrtcrit = 71.0, 0.32, 201.0, False
    ro_setup, vo_setup = 220.0, 8.0
    ros = [7.0, 8.0, 9.0]
    vos = [220.0, 230.0, 240.0]
    for ro, vo in zip(ros, vos):
        np = potential.NFWPotential(amp=2.0, a=3.0, ro=ro_setup, vo=vo_setup)
        # Computing the overdensity in physical units
        od = (
            np.mvir(ro=ro, vo=vo, H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit)
            / 4.0
            / numpy.pi
            * 3.0
            / np.rvir(ro=ro, vo=vo, H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit)
            ** 3.0
        ) * (10.0**6.0 / H**2.0 * 8.0 * numpy.pi / 3.0 / Om * (4.302 * 10.0**-6.0))
        assert (
            numpy.fabs(od - overdens) < 0.1
        ), "NFWPotential's virial quantities computed in physical units with different (ro,vo) from setup are incorrect"
        od = (
            np.mvir(
                ro=ro,
                vo=vo,
                H=H,
                Om=Om,
                overdens=overdens,
                wrtcrit=wrtcrit,
                use_physical=False,
            )
            / 4.0
            / numpy.pi
            * 3.0
            / np.rvir(
                ro=ro,
                vo=vo,
                H=H,
                Om=Om,
                overdens=overdens,
                wrtcrit=wrtcrit,
                use_physical=False,
            )
            ** 3.0
        ) * conversion.dens_in_meanmatterdens(vo, ro, H=H, Om=Om)
        assert (
            numpy.fabs(od - overdens) < 0.01
        ), "NFWPotential's virial quantities computed in internal units with different (ro,vo) from setup are incorrect"
        # Also test concentration
        assert (
            numpy.fabs(
                np.conc(ro=ro, vo=vo, H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit)
                - np.rvir(ro=ro, vo=vo, H=H, Om=Om, overdens=overdens, wrtcrit=wrtcrit)
                / np._scale
                / ro
            )
            < 0.01
        ), "NFWPotential's concentration computed for different (ro,vo) from setup is incorrect"
    return None


# Test that rmax and vmax are correctly determined for an NFW potential
def test_NFW_rmaxvmax():
    # Setup with rmax,vmax
    rmax, vmax = 1.2, 3.23
    np = potential.NFWPotential(rmax=rmax, vmax=vmax)
    # Now determine rmax and vmax numerically
    rmax_opt = optimize.minimize_scalar(lambda r: -np.vcirc(r), bracket=[0.01, 100.0])[
        "x"
    ]
    assert (
        numpy.fabs(rmax_opt - rmax) < 10.0**-7.0
    ), "NFW rmax() function does not behave as expected"
    assert (
        numpy.fabs(np.vcirc(rmax_opt) - vmax) < 10.0**-8.0
    ), "NFW rmax() function does not behave as expected"
    assert (
        numpy.fabs(np.vcirc(rmax_opt) - np.vmax()) < 10.0**-8.0
    ), "NFW vmax() function does not behave as expected"
    return None


def test_LinShuReductionFactor():
    # Test that the LinShuReductionFactor is implemented correctly, by comparing to figure 1 in Lin & Shu (1966)
    from galpy.potential import (
        LinShuReductionFactor,
        LogarithmicHaloPotential,
        epifreq,
        omegac,
    )

    lp = LogarithmicHaloPotential(normalize=1.0)  # work in flat rotation curve
    # nu^2 = 0.2, x=4 for m=2,sigmar=0.1
    # w/ nu = m(OmegaP-omegac)/epifreq, x=sr^2*k^2/epifreq^2
    R, m, sr = 0.9, 2.0, 0.1
    tepi, tomegac = epifreq(lp, R), omegac(lp, R)
    OmegaP = tepi * numpy.sqrt(0.2) / m + tomegac  # leads to nu^2 = 0.2
    k = numpy.sqrt(4.0) * tepi / sr
    assert (
        numpy.fabs(LinShuReductionFactor(lp, R, sr, m=m, k=k, OmegaP=OmegaP) - 0.18)
        < 0.01
    ), "LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)"
    # nu^2 = 0.8, x=10
    OmegaP = tepi * numpy.sqrt(0.8) / m + tomegac  # leads to nu^2 = 0.8
    k = numpy.sqrt(10.0) * tepi / sr
    assert (
        numpy.fabs(LinShuReductionFactor(lp, R, sr, m=m, k=k, OmegaP=OmegaP) - 0.04)
        < 0.01
    ), "LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)"
    # Similar test, but using a nonaxiPot= input
    from galpy.potential import SteadyLogSpiralPotential

    sp = SteadyLogSpiralPotential(m=2.0, omegas=OmegaP, alpha=k * R)
    assert (
        numpy.fabs(LinShuReductionFactor(lp, R, sr, nonaxiPot=sp) - 0.04) < 0.01
    ), "LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)"
    # Test exception
    try:
        LinShuReductionFactor(lp, R, sr)
    except OSError:
        pass
    else:
        raise AssertionError(
            "LinShuReductionFactor w/o nonaxiPot set or k=,m=,OmegaP= set did not raise IOError"
        )
    return None


def test_nemoaccname():
    # There is no real good way to test this (I think), so I'm just testing to
    # what I think is the correct output now to make sure this isn't
    # accidentally changed
    # Log
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    assert lp.nemo_accname() == "LogPot", "Logarithmic potential's NEMO name incorrect"
    # NFW
    np = potential.NFWPotential(normalize=1.0)
    assert np.nemo_accname() == "NFW", "NFW's NEMO name incorrect"
    # Miyamoto-Nagai
    mp = potential.MiyamotoNagaiPotential(normalize=1.0)
    assert mp.nemo_accname() == "MiyamotoNagai", "MiyamotoNagai's NEMO name incorrect"
    # Power-spherical w/ cut-off
    pp = potential.PowerSphericalPotentialwCutoff(normalize=1.0)
    assert (
        pp.nemo_accname() == "PowSphwCut"
    ), "Power-spherical potential w/ cuto-ff's NEMO name incorrect"
    # MN3ExponentialDiskPotential
    mp = potential.MN3ExponentialDiskPotential(normalize=1.0)
    assert (
        mp.nemo_accname() == "MiyamotoNagai+MiyamotoNagai+MiyamotoNagai"
    ), "MN3ExponentialDiskPotential's NEMO name incorrect"
    # Plummer
    pp = potential.PlummerPotential(normalize=1.0)
    assert pp.nemo_accname() == "Plummer", "PlummerPotential's NEMO name incorrect"
    # Hernquist
    hp = potential.HernquistPotential(normalize=1.0)
    assert hp.nemo_accname() == "Dehnen", "HernquistPotential's NEMO name incorrect"
    return None


def test_nemoaccnamepars_attributeerror():
    # Use BurkertPotential (unlikely that I would implement that one in NEMO soon)
    bp = potential.BurkertPotential(normalize=1.0)
    try:
        bp.nemo_accname()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "nemo_accname for potential w/o accname does not raise AttributeError"
        )
    try:
        bp.nemo_accpars(220.0, 8.0)
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "nemo_accpars for potential w/o accname does not raise AttributeError"
        )
    return None


def test_nemoaccnames():
    # Just test MWPotential2014 and a single potential
    # MWPotential2014
    assert (
        potential.nemo_accname(potential.MWPotential2014)
        == "PowSphwCut+MiyamotoNagai+NFW"
    ), "MWPotential2014's NEMO name is incorrect"
    # Power-spherical w/ cut-off
    pp = potential.PowerSphericalPotentialwCutoff(normalize=1.0)
    assert (
        potential.nemo_accname(pp) == "PowSphwCut"
    ), "Power-spherical potential w/ cut-off's NEMO name incorrect"
    return None


def test_nemoaccpars():
    # Log
    lp = potential.LogarithmicHaloPotential(
        amp=2.0, core=3.0, q=27.0
    )  # completely ridiculous, but tests scalings
    vo, ro = 2.0, 3.0
    vo /= 1.0227121655399913
    ap = lp.nemo_accpars(vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Logarithmic potential's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 8.0) < 10.0**-8.0
    ), "Logarithmic potential's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 729.0) < 10.0**-8.0
    ), "Logarithmic potential's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 1.0) < 10.0**-8.0
    ), "Logarithmic potential's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[4]) - 27.0) < 10.0**-8.0
    ), "Logarithmic potential's NEMO accpars incorrect"
    # Miyamoto-Nagai
    mp = potential.MiyamotoNagaiPotential(amp=3.0, a=2.0, b=5.0)
    vo, ro = 7.0, 9.0
    vo /= 1.0227121655399913
    ap = mp.nemo_accpars(vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 1323.0) < 10.0**-5.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 18.0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    # Power-spherical w/ cut-off
    pp = potential.PowerSphericalPotentialwCutoff(amp=3.0, alpha=4.0, rc=5.0)
    vo, ro = 7.0, 9.0
    vo /= 1.0227121655399913
    ap = pp.nemo_accpars(vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 11907.0) < 10.0**-4.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 4.0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    # NFW
    np = potential.NFWPotential(amp=1.0 / 0.2162165954, a=1.0 / 16)
    vo, ro = 3.0, 4.0
    vo /= 1.0227121655399913
    ap = np.nemo_accpars(vo, ro).split(",")
    assert numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0, "NFW's NEMO accpars incorrect"
    assert numpy.fabs(float(ap[1]) - 0.25) < 10.0**-8.0, "NFW's NEMO accpars incorrect"
    assert numpy.fabs(float(ap[2]) - 12.0) < 10.0**-8.0, "NFW's NEMO accpars incorrect"
    # MN3ExponentialDiskPotential
    mn = potential.MN3ExponentialDiskPotential(normalize=1.0, hr=2.0, hz=0.5)
    vo, ro = 3.0, 4.0
    ap = mn.nemo_accpars(vo, ro).replace("#", ",").split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[4]) - 0) < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[8]) - 0) < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    # Test ratios
    assert (
        numpy.fabs(float(ap[1]) / float(ap[5]) - mn._mn3[0]._amp / mn._mn3[1]._amp)
        < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) / float(ap[9]) - mn._mn3[0]._amp / mn._mn3[2]._amp)
        < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) / float(ap[6]) - mn._mn3[0]._a / mn._mn3[1]._a)
        < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) / float(ap[10]) - mn._mn3[0]._a / mn._mn3[2]._a)
        < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) / float(ap[7]) - 1.0) < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) / float(ap[11]) - 1.0) < 10.0**-8.0
    ), "MN3ExponentialDiskPotential 's NEMO accpars incorrect"
    # Plummer
    pp = potential.PlummerPotential(amp=3.0, b=5.0)
    vo, ro = 7.0, 9.0
    vo /= 1.0227121655399913
    ap = pp.nemo_accpars(vo, ro).split(",")
    assert numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0, "Plummer's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 1323.0) < 10.0**-5.0
    ), "Plummer's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 45.0) < 10.0**-8.0
    ), "Plummer's NEMO accpars incorrect"
    # Hernquist
    hp = potential.HernquistPotential(amp=2.0, a=1.0 / 4.0)
    vo, ro = 3.0, 4.0
    vo /= 1.0227121655399913
    ap = hp.nemo_accpars(vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Hernquist's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 1.0) < 10.0**-8.0
    ), "Hernquist's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 9.0 * 4) < 10.0**-7.0
    ), "Hernquist's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 1.0) < 10.0**-8.0
    ), "Hernquist's NEMO accpars incorrect"
    return None


def test_nemoaccparss():
    # Just combine a few of the above ones
    # Miyamoto + PowerSpherwCut
    mp = potential.MiyamotoNagaiPotential(amp=3.0, a=2.0, b=5.0)
    pp = potential.PowerSphericalPotentialwCutoff(amp=3.0, alpha=4.0, rc=5.0)
    vo, ro = 7.0, 9.0
    vo /= 1.0227121655399913
    ap = potential.nemo_accpars(mp, vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 1323.0) < 10.0**-5.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 18.0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "MiyamotoNagai's NEMO accpars incorrect"
    # PowSpherwCut
    ap = potential.nemo_accpars(pp, vo, ro).split(",")
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 11907.0) < 10.0**-4.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 4.0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    # Combined
    apc = potential.nemo_accpars([mp, pp], vo, ro).split("#")
    ap = apc[0].split(",")  # should be MN
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 1323.0) < 10.0**-5.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 18.0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    ap = apc[1].split(",")  # should be PP
    assert (
        numpy.fabs(float(ap[0]) - 0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[1]) - 11907.0) < 10.0**-4.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[2]) - 4.0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    assert (
        numpy.fabs(float(ap[3]) - 45.0) < 10.0**-8.0
    ), "Miyamoto+Power-spherical potential w/ cut-off's NEMO accpars incorrect"
    return None


def test_MN3ExponentialDiskPotential_inputs():
    # Test the inputs of the MN3ExponentialDiskPotential
    # IOError for hz so large that b is negative
    try:
        mn = potential.MN3ExponentialDiskPotential(amp=1.0, hz=50.0)
    except OSError:
        pass
    else:
        raise AssertionError(
            "MN3ExponentialDiskPotential with ridiculous hz should have given IOError, but didn't"
        )
    # Warning when b/Rd > 3 or (b/Rd > 1.35 and posdens)
    # Turn warnings into errors to test for them
    import warnings

    from galpy.util import galpyWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        mn = MN3ExponentialDiskPotential(normalize=1.0, hz=1.438, hr=1.0)
        # Should raise warning bc of MN3ExponentialDiskPotential,
        # might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = "MN3ExponentialDiskPotential" in str(wa.message)
            if raisedWarning:
                break
        assert raisedWarning, "MN3ExponentialDiskPotential w/o posdens, but with b/Rd > 3 did not raise galpyWarning"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        mn = MN3ExponentialDiskPotential(normalize=1.0, hr=1.0, hz=0.7727, posdens=True)
        raisedWarning = False
        for wa in w:
            raisedWarning = "MN3ExponentialDiskPotential" in str(wa.message)
            if raisedWarning:
                break
        assert raisedWarning, "MN3ExponentialDiskPotential w/o posdens, but with b/Rd > 1.35 did not raise galpyWarning"
    return None


def test_MN3ExponentialDiskPotential_hz():
    # Test that we correctly convert from hz/Rd to b/Rd
    # exp
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=1.0, sech=False)
    assert (
        numpy.fabs(mn._brd - 1.875) < 0.05
    ), "b/Rd not computed correctly for exponential profile"
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=2.0, hz=1.0, sech=False)
    assert (
        numpy.fabs(mn._brd - 0.75) < 0.05
    ), "b/Rd not computed correctly for exponential profile"
    # sech
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=2.0, sech=True)
    assert (
        numpy.fabs(mn._brd - 2.1) < 0.05
    ), "b/Rd not computed correctly for sech^2 profile"
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=2.0, hz=2.0, sech=True)
    assert (
        numpy.fabs(mn._brd - 0.9) < 0.05
    ), "b/Rd not computed correctly for sech^2 profile"
    return None


def test_MN3ExponentialDiskPotential_approx():
    # Test that the 3MN approximation works to the advertised level
    # Zero thickness
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001, sech=False)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001)
    dpmass = dp.mass(4.0, 5.0 * 0.001)
    assert (
        numpy.fabs(mn.mass(4.0, 5.0 * 0.001) - dpmass) / dpmass < 0.005
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62, sech=False)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(4.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(4.0, 10.0 * 0.6) - dpmass) / dpmass < 0.01
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness w/ sech
    mn = potential.MN3ExponentialDiskPotential(amp=0.5, hr=1.0, hz=1.24, sech=True)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(4.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(4.0, 20.0 * 0.6) - dpmass) / dpmass < 0.01
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # At 10 Rd
    # Zero thickness
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001, sech=False)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001)
    dpmass = dp.mass(10.0, 5.0 * 0.001)
    assert (
        numpy.fabs(mn.mass(10.0, 5.0 * 0.001) - dpmass) / dpmass < 0.04
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness
    mn = potential.MN3ExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62, sech=False)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(10.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(10.0, 10.0 * 0.6) - dpmass) / dpmass < 0.04
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness w/ sech
    mn = potential.MN3ExponentialDiskPotential(amp=0.5, hr=1.0, hz=1.24, sech=True)
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(10.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(10.0, 20.0 * 0.6) - dpmass) / dpmass < 0.04
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # For posdens the deviations are larger
    # Zero thickness
    mn = potential.MN3ExponentialDiskPotential(
        amp=1.0, hr=1.0, hz=0.001, sech=False, posdens=True
    )
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001)
    dpmass = dp.mass(4.0, 5.0 * 0.001)
    assert (
        numpy.fabs(mn.mass(4.0, 5.0 * 0.001) - dpmass) / dpmass < 0.015
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness
    mn = potential.MN3ExponentialDiskPotential(
        amp=1.0, hr=1.0, hz=0.62, sech=False, posdens=True
    )
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(4.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(4.0, 10.0 * 0.6) - dpmass) / dpmass < 0.015
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # At 10 Rd
    # Zero thickness
    mn = potential.MN3ExponentialDiskPotential(
        amp=1.0, hr=1.0, hz=0.001, sech=False, posdens=True
    )
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.001)
    dpmass = dp.mass(10.0, 5.0 * 0.001)
    assert (
        numpy.fabs(mn.mass(10.0, 5.0 * 0.001) - dpmass) / dpmass > 0.04
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    assert (
        numpy.fabs(mn.mass(10.0, 5.0 * 0.001) - dpmass) / dpmass < 0.07
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    # Finite thickness
    mn = potential.MN3ExponentialDiskPotential(
        amp=1.0, hr=1.0, hz=0.62, sech=False, posdens=True
    )
    dp = potential.DoubleExponentialDiskPotential(amp=1.0, hr=1.0, hz=0.62)
    dpmass = dp.mass(10.0, 5.0 * 0.6)
    assert (
        numpy.fabs(mn.mass(10.0, 10.0 * 0.6) - dpmass) / dpmass < 0.08
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    assert (
        numpy.fabs(mn.mass(10.0, 10.0 * 0.6) - dpmass) / dpmass > 0.03
    ), "MN3ExponentialDiskPotential does not approximate the enclosed mass as advertised"
    return None


def test_TwoPowerTriaxialPotential_vs_TwoPowerSphericalPotential():
    # Test that TwoPowerTriaxialPotential with spherical parameters is the same
    # as TwoPowerSphericalPotential
    tol = -4.0  # tough general case
    rs = numpy.linspace(0.001, 25.0, 1001)
    tnp = potential.TwoPowerTriaxialPotential(
        normalize=1.0, b=1.0, c=1.0, a=1.5, alpha=1.5, beta=3.5
    )
    np = potential.TwoPowerSphericalPotential(normalize=1.0, a=1.5, alpha=1.5, beta=3.5)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [numpy.sqrt(tnp.Rforce(r, 0.0) / np.Rforce(r, 0.0)) for r in rs]
            )
            - 1.0
        )
        < 10.0**tol
    ), "Vcirc not the same for TwoPowerSphericalPotential and spherical version of TwoPowerTriaxialPotential"
    # Also do specific cases
    tol = -8.0  # much better
    # Hernquist
    tnp = potential.TriaxialHernquistPotential(normalize=1.0, b=1.0, c=1.0, a=1.5)
    np = potential.HernquistPotential(normalize=1.0, a=1.5)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [numpy.sqrt(tnp.Rforce(r, 0.0) / np.Rforce(r, 0.0)) for r in rs]
            )
            - 1.0
        )
        < 10.0**tol
    ), "Vcirc not the same for Hernquist and spherical version of TriaxialHernquist"
    # NFW
    tnp = potential.TriaxialNFWPotential(normalize=1.0, b=1.0, c=1.0, a=1.5)
    np = potential.NFWPotential(normalize=1.0, a=1.5)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [numpy.sqrt(tnp.Rforce(r, 0.0) / np.Rforce(r, 0.0)) for r in rs]
            )
            - 1.0
        )
        < 10.0**tol
    ), "Vcirc not the same for NFW and spherical version of TriaxialNFW"
    # Jaffe
    tnp = potential.TriaxialJaffePotential(normalize=1.0, b=1.0, c=1.0, a=1.5)
    np = potential.JaffePotential(normalize=1.0, a=1.5)
    assert numpy.all(
        numpy.fabs(
            numpy.array(
                [numpy.sqrt(tnp.Rforce(r, 0.0) / np.Rforce(r, 0.0)) for r in rs]
            )
            - 1.0
        )
        < 10.0**tol
    ), "Vcirc not the same for Jaffe and spherical version of TriaxialJaffe"
    return None


# Test that TwoPowerTriaxial setup raises an error for bad values of alpha
# and beta
def test_TwoPowerTriaxialPotential_alphahigherror():
    with pytest.raises(IOError) as excinfo:
        dummy = potential.TwoPowerTriaxialPotential(alpha=3.5)
    return None


def test_TwoPowerTriaxialPotential_betalowerror():
    with pytest.raises(IOError) as excinfo:
        dummy = potential.TwoPowerTriaxialPotential(beta=1.0)
    return None


# Test that DehnenSphericalPotential setup raises an error for bad values of alpha
def test_DehnenSphericalPotential_alphalowhigherror():
    with pytest.raises(IOError) as excinfo:
        dummy = potential.DehnenSphericalPotential(alpha=-0.5)
    with pytest.raises(IOError) as excinfo:
        dummy = potential.DehnenSphericalPotential(alpha=3.5)
    return None


# Test that FerrersPotential raises a value error for n < 0
def test_FerrersPotential_nNegative():
    with pytest.raises(ValueError) as excinfo:
        dummy = potential.FerrersPotential(n=-1.0)
    return None


# Test that SphericalShellPotential raises a value error for normalize=True and a > 1
def test_SphericalShellPotential_normalizer0():
    with pytest.raises(ValueError) as excinfo:
        dummy = potential.SphericalShellPotential(normalize=1.0, a=2.0)
    return None


# Test that RingPotential raises a value error for normalize=True and a > 1
def test_RingPotential_normalizer0():
    with pytest.raises(ValueError) as excinfo:
        dummy = potential.RingPotential(normalize=1.0, a=2.0)
    return None


def test_planeRotatedNFWPotential():
    # Test that the rotation according to pa works as expected
    tnp = potential.TriaxialNFWPotential(
        normalize=1.0, a=1.5, b=0.5, pa=30.0 / 180.0 * numpy.pi
    )
    # Compute the potential at a fixed radius, minimum should be at pa!
    Rs = 0.8
    phis = numpy.linspace(0.0, numpy.pi, 1001)
    pot = numpy.array([tnp(Rs, 0.0, phi=phi) for phi in phis])
    minphi = numpy.argmin(pot)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 30.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential rotated around the z axis does not behave as expected"
    # Same for density, but max instead
    dens = numpy.array([tnp.dens(Rs, 0.0, phi=phi) for phi in phis])
    minphi = numpy.argmax(dens)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 30.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential rotated around the z axis does not behave as expected"
    # Also do a negative angle
    tnp = potential.TriaxialNFWPotential(
        normalize=1.0, a=1.5, b=0.5, pa=-60.0 / 180.0 * numpy.pi
    )
    # Compute the potential at a fixed radius, minimum should be at pa!
    Rs = 0.8
    phis = numpy.linspace(0.0, numpy.pi, 1001)
    pot = numpy.array([tnp(Rs, 0.0, phi=phi) for phi in phis])
    minphi = numpy.argmin(pot)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 120.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential rotated around the z axis does not behave as expected"
    # Same for density, but max instead
    dens = numpy.array([tnp.dens(Rs, 0.0, phi=phi) for phi in phis])
    minphi = numpy.argmax(dens)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 120.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential rotated around the z axis does not behave as expected"
    return None


def test_zaxisRotatedNFWPotential():
    from galpy.util import coords

    # Test that the rotation according to zvec works as expected
    pa = 30.0 / 180.0 * numpy.pi
    tnp = potential.TriaxialNFWPotential(
        normalize=1.0, a=1.5, c=0.5, zvec=[0.0, -numpy.sin(pa), numpy.cos(pa)]
    )
    # Compute the potential at a fixed radius in the y/z plane,
    # minimum should be at pa!
    Rs = 0.8
    phis = numpy.linspace(0.0, numpy.pi, 1001)
    xs = numpy.zeros_like(phis)
    ys = Rs * numpy.cos(phis)
    zs = Rs * numpy.sin(phis)
    tR, tphi, tz = coords.rect_to_cyl(xs, ys, zs)
    pot = numpy.array([tnp(r, z, phi=phi) for r, z, phi in zip(tR, tz, tphi)])
    minphi = numpy.argmin(pot)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 30.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential with rotated z axis does not behave as expected"
    # Same for density, but max instead
    dens = numpy.array([tnp.dens(r, z, phi=phi) for r, z, phi in zip(tR, tz, tphi)])
    minphi = numpy.argmax(dens)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 30.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential with rotated z axis does not behave as expected"
    # Another one
    pa = -60.0 / 180.0 * numpy.pi
    tnp = potential.TriaxialNFWPotential(
        normalize=1.0, a=1.5, c=0.5, zvec=[-numpy.sin(pa), 0.0, numpy.cos(pa)]
    )
    # Compute the potential at a fixed radius in the z/z plane,
    # minimum should be at pa!
    Rs = 0.8
    phis = numpy.linspace(0.0, numpy.pi, 1001)
    xs = Rs * numpy.cos(phis)
    ys = numpy.zeros_like(phis)
    zs = Rs * numpy.sin(phis)
    tR, tphi, tz = coords.rect_to_cyl(xs, ys, zs)
    pot = numpy.array([tnp(r, z, phi=phi) for r, z, phi in zip(tR, tz, tphi)])
    minphi = numpy.argmin(pot)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 120.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential with rotated z axis does not behave as expected"
    # Same for density, but max instead
    dens = numpy.array([tnp.dens(r, z, phi=phi) for r, z, phi in zip(tR, tz, tphi)])
    minphi = numpy.argmax(dens)
    minphi_pred = numpy.argmin(numpy.fabs(phis - 120.0 / 180.0 * numpy.pi))
    assert (
        minphi == minphi_pred
    ), "Flattened NFW potential with rotated z axis does not behave as expected"
    return None


def test_nonaxierror_function():
    # Test that the code throws an exception when calling a non-axisymmetric
    # potential without phi
    tnp = potential.TriaxialNFWPotential(amp=1.0, b=0.7, c=0.9)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatePotentials(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateDensities(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateRforces(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatezforces(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatephitorques(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateR2derivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatez2derivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateRzderivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatephi2derivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateRphiderivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluatephizderivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluaterforces(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluater2derivs(tnp, 1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        potential.evaluateSurfaceDensities(tnp, 1.0, 0.1)
    return None


def test_SoftenedNeedleBarPotential_density():
    # Some simple tests of the density of the SoftenedNeedleBarPotential
    # For a spherical softening kernel, density should be symmetric to y/z
    sbp = potential.SoftenedNeedleBarPotential(
        normalize=1.0, a=1.0, c=0.1, b=0.0, pa=0.0
    )
    assert (
        numpy.fabs(
            sbp.dens(2.0, 0.0, phi=numpy.pi / 4.0)
            - sbp.dens(numpy.sqrt(2.0), numpy.sqrt(2.0), phi=0.0)
        )
        < 10.0**-13.0
    ), "SoftenedNeedleBarPotential with spherical softening kernel does not appear to have a spherically symmetric density"
    # Another one
    assert (
        numpy.fabs(
            sbp.dens(4.0, 0.0, phi=numpy.pi / 4.0)
            - sbp.dens(2.0 * numpy.sqrt(2.0), 2.0 * numpy.sqrt(2.0), phi=0.0)
        )
        < 10.0**-13.0
    ), "SoftenedNeedleBarPotential with spherical softening kernel does not appear to have a spherically symmetric density"
    # For a flattened softening kernel, the density at (y,z) should be higher than at (z,y)
    sbp = potential.SoftenedNeedleBarPotential(
        normalize=1.0, a=1.0, c=0.1, b=0.3, pa=0.0
    )
    assert (
        sbp.dens(2.0, 0.0, phi=numpy.pi / 4.0)
        > sbp.dens(numpy.sqrt(2.0), numpy.sqrt(2.0), phi=0.0)
    ), "SoftenedNeedleBarPotential with flattened softening kernel does not appear to have a consistent"
    # Another one
    assert (
        sbp.dens(4.0, 0.0, phi=numpy.pi / 4.0)
        > sbp.dens(2.0 * numpy.sqrt(2.0), 2.0 * numpy.sqrt(2.0), phi=0.0)
    ), "SoftenedNeedleBarPotential with flattened softening kernel does not appear to have a consistent"
    return None


def test_DiskSCFPotential_SigmaDerivs():
    # Test that the derivatives of Sigma are correctly implemented in DiskSCF
    # Very rough finite difference checks
    dscfp = potential.DiskSCFPotential(
        dens=lambda R, z: 1.0,  # doesn't matter
        Sigma=[
            {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
            {"type": "expwhole", "h": 1.0 / 3.0, "amp": 1.0, "Rhole": 0.5},
        ],
        hz=[{"type": "exp", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 27.0}],
        a=1.0,
        N=2,
        L=2,
    )
    # Sigma exp
    testRs = numpy.linspace(0.3, 1.5, 101)
    dR = 10.0**-8.0
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._Sigma[0](testRs + dR) - dscfp._Sigma[0](testRs)) / dR
                - dscfp._dSigmadR[0](testRs)
            )
            / dscfp._dSigmadR[0](testRs)
        )
        < 10.0**-7.0
    ), "Derivative dSigmadR does not agree with finite-difference derivative of Sigma for exponential profile in DiskSCFPotential"
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._dSigmadR[0](testRs + dR) - dscfp._dSigmadR[0](testRs)) / dR
                - dscfp._d2SigmadR2[0](testRs)
            )
            / dscfp._d2SigmadR2[0](testRs)
        )
        < 10.0**-7.0
    ), "Derivative d2SigmadR2 does not agree with finite-difference derivative of dSigmadR for exponential profile in DiskSCFPotential"
    # Sigma expwhole
    dR = 10.0**-8.0
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._Sigma[1](testRs + dR) - dscfp._Sigma[1](testRs)) / dR
                - dscfp._dSigmadR[1](testRs)
            )
            / dscfp._dSigmadR[1](testRs)
        )
        < 10.0**-4.0
    ), "Derivative dSigmadR does not agree with finite-difference derivative of Sigma for exponential-with-hole profile in DiskSCFPotential"
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._dSigmadR[1](testRs + dR) - dscfp._dSigmadR[1](testRs)) / dR
                - dscfp._d2SigmadR2[1](testRs)
            )
            / dscfp._d2SigmadR2[1](testRs)
        )
        < 10.0**-4.0
    ), "Derivative d2SigmadR2 does not agree with finite-difference derivative of dSigmadR for exponential-with-hole profile in DiskSCFPotential"
    return None


def test_DiskSCFPotential_verticalDerivs():
    # Test that the derivatives of Sigma are correctly implemented in DiskSCF
    # Very rough finite difference checks
    dscfp = potential.DiskSCFPotential(
        dens=lambda R, z: 1.0,  # doesn't matter
        Sigma=[
            {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
            {"type": "expwhole", "h": 1.0 / 3.0, "amp": 1.0, "Rhole": 0.5},
        ],
        hz=[{"type": "exp", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 27.0}],
        a=1.0,
        N=2,
        L=2,
    )
    # Vertical exp
    testzs = numpy.linspace(0.1 / 27.0, 3.0 / 27, 101)
    dz = 10.0**-8.0
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._Hz[0](testzs + dz) - dscfp._Hz[0](testzs)) / dz
                - dscfp._dHzdz[0](testzs)
            )
            / dscfp._dHzdz[0](testzs)
        )
        < 10.0**-5.5
    ), "Derivative dHzdz does not agree with finite-difference derivative of Hz for exponential profile in DiskSCFPotential"
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._dHzdz[0](testzs + dz) - dscfp._dHzdz[0](testzs)) / dz
                - dscfp._hz[0](testzs)
            )
            / dscfp._hz[0](testzs)
        )
        < 10.0**-6.0
    ), "Derivative hz does not agree with finite-difference derivative of dHzdz for exponential profile in DiskSCFPotential"
    # Vertical sech^2
    dz = 10.0**-8.0
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._Hz[1](testzs + dz) - dscfp._Hz[1](testzs)) / dz
                - dscfp._dHzdz[1](testzs)
            )
            / dscfp._dHzdz[1](testzs)
        )
        < 10.0**-5.5
    ), "Derivative dSigmadz does not agree with finite-difference derivative of Sigma for sech2 profile in DiskSCFPotential"
    assert numpy.all(
        numpy.fabs(
            (
                (dscfp._dHzdz[1](testzs + dz) - dscfp._dHzdz[1](testzs)) / dz
                - dscfp._hz[1](testzs)
            )
            / dscfp._hz[1](testzs)
        )
        < 10.0**-6.0
    ), "Derivative hz does not agree with finite-difference derivative of dHzdz for sech2 profile in DiskSCFPotential"
    return None


def test_DiskSCFPotential_nhzNeqnsigmaError():
    with pytest.raises(ValueError) as excinfo:
        dummy = potential.DiskSCFPotential(
            dens=lambda R, z: numpy.exp(-3.0 * R)
            * 1.0
            / numpy.cosh(z / 2.0 * 27.0) ** 2.0
            / 4.0
            * 27.0,
            Sigma={"h": 1.0 / 3.0, "type": "exp", "amp": 1.0},
            hz=[{"type": "sech2", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 27.0}],
            a=1.0,
            N=5,
            L=5,
        )
    return None


def test_DiskSCFPotential_againstDoubleExp():
    # Test that the DiskSCFPotential approx. of a dbl-exp disk agrees with
    # DoubleExponentialDiskPotential
    dp = potential.DoubleExponentialDiskPotential(amp=13.5, hr=1.0 / 3.0, hz=1.0 / 27.0)
    dscfp = potential.DiskSCFPotential(
        dens=lambda R, z: dp.dens(R, z),
        Sigma_amp=1.0,
        Sigma=lambda R: numpy.exp(-3.0 * R),
        dSigmadR=lambda R: -3.0 * numpy.exp(-3.0 * R),
        d2SigmadR2=lambda R: 9.0 * numpy.exp(-3.0 * R),
        hz={"type": "exp", "h": 1.0 / 27.0},
        a=1.0,
        N=10,
        L=10,
    )
    testRs = numpy.linspace(0.3, 1.5, 101)
    testzs = numpy.linspace(0.1 / 27.0, 3.0 / 27, 101)
    testR = 0.9 * numpy.ones_like(testzs)
    testz = 1.5 / 27.0 * numpy.ones_like(testRs)
    # Test potential
    assert numpy.all(
        numpy.fabs((dp(testRs, testz) - dscfp(testRs, testz)) / dscfp(testRs, testz))
        < 10.0**-2.5
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    assert numpy.all(
        numpy.fabs((dp(testR, testzs) - dscfp(testR, testzs)) / dscfp(testRs, testz))
        < 10.0**-2.5
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    # Rforce
    assert numpy.all(
        numpy.fabs(
            (
                numpy.array([dp.Rforce(r, z) for (r, z) in zip(testRs, testz)])
                - dscfp.Rforce(testRs, testz)
            )
            / dscfp.Rforce(testRs, testz)
        )
        < 10.0**-2.0
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    assert numpy.all(
        numpy.fabs(
            (
                numpy.array([dp.Rforce(r, z) for (r, z) in zip(testR, testzs)])
                - dscfp.Rforce(testR, testzs)
            )
            / dscfp.Rforce(testRs, testz)
        )
        < 10.0**-2.0
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    # zforce
    assert numpy.all(
        numpy.fabs(
            (
                numpy.array([dp.zforce(r, z) for (r, z) in zip(testRs, testz)])
                - dscfp.zforce(testRs, testz)
            )
            / dscfp.zforce(testRs, testz)
        )
        < 10.0**-1.5
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    # Following has rel. large difference at high z
    assert numpy.all(
        numpy.fabs(
            (
                numpy.array([dp.zforce(r, z) for (r, z) in zip(testR, testzs)])
                - dscfp.zforce(testR, testzs)
            )
            / dscfp.zforce(testRs, testz)
        )
        < 10.0**-1.0
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    return None


def test_DiskSCFPotential_againstDoubleExp_dens():
    # Test that the DiskSCFPotential approx. of a dbl-exp disk agrees with
    # DoubleExponentialDiskPotential
    dp = potential.DoubleExponentialDiskPotential(amp=13.5, hr=1.0 / 3.0, hz=1.0 / 27.0)
    dscfp = potential.DiskSCFPotential(
        dens=lambda R, z: dp.dens(R, z),
        Sigma={"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
        hz={"type": "exp", "h": 1.0 / 27.0},
        a=1.0,
        N=10,
        L=10,
    )
    testRs = numpy.linspace(0.3, 1.5, 101)
    testzs = numpy.linspace(0.1 / 27.0, 3.0 / 27, 101)
    testR = 0.9 * numpy.ones_like(testzs)
    testz = 1.5 / 27.0 * numpy.ones_like(testRs)
    # Test density
    assert numpy.all(
        numpy.fabs(
            (dp.dens(testRs, testz) - dscfp.dens(testRs, testz))
            / dscfp.dens(testRs, testz)
        )
        < 10.0**-1.25
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    # difficult at high z
    assert numpy.all(
        numpy.fabs(
            (dp.dens(testR, testzs) - dscfp.dens(testR, testzs))
            / dscfp.dens(testRs, testz)
        )
        < 10.0**-1.0
    ), "DiskSCFPotential for double-exponential disk does not agree with DoubleExponentialDiskPotential"
    return None


def test_WrapperPotential_dims():
    # Test that WrapperPotentials get assigned to Potential/planarPotential
    # correctly, based on input pot=
    from galpy.potential.WrapperPotential import (
        WrapperPotential,
        parentWrapperPotential,
        planarWrapperPotential,
    )

    dp = potential.DehnenBarPotential()
    # 3D pot should be Potential, Wrapper, parentWrapper, not planarX
    dwp = potential.DehnenSmoothWrapperPotential(pot=dp)
    assert isinstance(
        dwp, potential.Potential
    ), "WrapperPotential for 3D pot= is not an instance of Potential"
    assert not isinstance(
        dwp, potential.planarPotential
    ), "WrapperPotential for 3D pot= is an instance of planarPotential"
    assert isinstance(
        dwp, parentWrapperPotential
    ), "WrapperPotential for 3D pot= is not an instance of parentWrapperPotential"
    assert isinstance(
        dwp, WrapperPotential
    ), "WrapperPotential for 3D pot= is not an instance of WrapperPotential"
    assert not isinstance(
        dwp, planarWrapperPotential
    ), "WrapperPotential for 3D pot= is an instance of planarWrapperPotential"
    # 2D pot should be Potential, Wrapper, parentWrapper, not planarX
    dwp = potential.DehnenSmoothWrapperPotential(pot=dp.toPlanar())
    assert isinstance(
        dwp, potential.planarPotential
    ), "WrapperPotential for 3D pot= is not an instance of planarPotential"
    assert not isinstance(
        dwp, potential.Potential
    ), "WrapperPotential for 3D pot= is an instance of Potential"
    assert isinstance(
        dwp, parentWrapperPotential
    ), "WrapperPotential for 3D pot= is not an instance of parentWrapperPotential"
    assert isinstance(
        dwp, planarWrapperPotential
    ), "WrapperPotential for 3D pot= is not an instance of planarWrapperPotential"
    assert not isinstance(
        dwp, WrapperPotential
    ), "WrapperPotential for 3D pot= is an instance of WrapperPotential"
    return None


def test_Wrapper_potinputerror():
    # Test that setting up a WrapperPotential with anything other than a
    # (list of) planar/Potentials raises an error
    with pytest.raises(ValueError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=1)
    return None


def test_Wrapper_incompatibleunitserror():
    # Test that setting up a WrapperPotential with a potential with
    # incompatible units to the wrapper itself raises an error
    # 3D
    ro, vo = 8.0, 220.0
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro, vo=vo)
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=1.1 * ro, vo=vo)
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=ro, vo=vo * 1.1)
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=1.1 * ro, vo=vo * 1.1)
    # 2D
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro, vo=vo).toPlanar()
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=1.1 * ro, vo=vo)
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=ro, vo=vo * 1.1)
    with pytest.raises(AssertionError) as excinfo:
        potential.DehnenSmoothWrapperPotential(pot=hp, ro=1.1 * ro, vo=vo * 1.1)
    return None


def test_Wrapper_Force_error():
    # Test that applying a wrapper to a DissipativeForce does not currently work
    def M(t):
        return 1.0

    # Initialize potentials and time-varying potentials
    df = potential.ChandrasekharDynamicalFrictionForce(GMs=1.0)
    with pytest.raises(RuntimeError) as excinfo:
        df_wrap = potential.TimeDependentAmplitudeWrapperPotential(A=M, amp=1, pot=df)
    assert (
        "WrapperPotential cannot currently wrap non-Potential Force objects"
        == excinfo.value.args[0]
    )
    # Also test for list
    with pytest.raises(RuntimeError) as excinfo:
        df_wrap = potential.TimeDependentAmplitudeWrapperPotential(
            A=M, amp=1, pot=potential.MWPotential2014 + df
        )
    assert (
        "WrapperPotential cannot currently wrap non-Potential Force objects"
        == excinfo.value.args[0]
    )
    return None


def test_WrapperPotential_unittransfer_3d():
    # Test that units are properly transferred between a potential and its
    # wrapper
    from galpy.util import conversion

    ro, vo = 9.0, 230.0
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro, vo=vo)
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["ro"] - ro) < 1e-10
    ), "ro not properly transferred to wrapper when wrapping a potential with ro set"
    assert (
        numpy.fabs(hpw_phys["vo"] - vo) < 1e-10
    ), "vo not properly transferred to wrapper when wrapping a potential with vo set"
    # Just set ro
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro)
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert not hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["ro"] - ro) < 1e-10
    ), "ro not properly transferred to wrapper when wrapping a potential with ro set"
    # Just set vo
    hp = potential.HernquistPotential(amp=0.55, a=1.3, vo=vo)
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert not hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["vo"] - vo) < 1e-10
    ), "vo not properly transferred to wrapper when wrapping a potential with vo set"
    return None


def test_WrapperPotential_unittransfer_2d():
    # Test that units are properly transferred between a potential and its
    # wrapper
    from galpy.util import conversion

    ro, vo = 9.0, 230.0
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro, vo=vo).toPlanar()
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["ro"] - ro) < 1e-10
    ), "ro not properly transferred to wrapper when wrapping a potential with ro set"
    assert (
        numpy.fabs(hpw_phys["vo"] - vo) < 1e-10
    ), "vo not properly transferred to wrapper when wrapping a potential with vo set"
    # Just set ro
    hp = potential.HernquistPotential(amp=0.55, a=1.3, ro=ro).toPlanar()
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert not hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["ro"] - ro) < 1e-10
    ), "ro not properly transferred to wrapper when wrapping a potential with ro set"
    # Just set vo
    hp = potential.HernquistPotential(amp=0.55, a=1.3, vo=vo).toPlanar()
    hpw = potential.DehnenSmoothWrapperPotential(pot=hp)
    hpw_phys = conversion.get_physical(hpw, include_set=True)
    assert not hpw_phys["roSet"], "ro not set when wrapping a potential with ro set"
    assert hpw_phys["voSet"], "vo not set when wrapping a potential with vo set"
    assert (
        numpy.fabs(hpw_phys["vo"] - vo) < 1e-10
    ), "vo not properly transferred to wrapper when wrapping a potential with vo set"
    return None


def test_WrapperPotential_serialization():
    import pickle

    from galpy.potential.WrapperPotential import WrapperPotential

    dp = potential.DehnenBarPotential()
    dwp = potential.DehnenSmoothWrapperPotential(pot=dp)
    pickled_dwp = pickle.dumps(dwp)
    unpickled_dwp = pickle.loads(pickled_dwp)
    assert isinstance(
        unpickled_dwp, WrapperPotential
    ), "Deserialized WrapperPotential is not an instance of WrapperPotential"
    testRs = numpy.linspace(0.1, 1, 100)
    testzs = numpy.linspace(-1, 1, 100)
    testphis = numpy.linspace(0, 2 * numpy.pi, 100)
    testts = numpy.linspace(0, 1, 100)
    for R, z, phi, t in zip(testRs, testzs, testphis, testts):
        assert dwp(R, z, phi, t) == unpickled_dwp(
            R, z, phi, t
        ), "Deserialized WrapperPotential does not agree with original WrapperPotential"


def test_WrapperPotential_print():
    dp = potential.DehnenBarPotential()
    dwp = potential.DehnenSmoothWrapperPotential(pot=dp)
    assert print(dwp) is None, "Printing a 3D wrapper potential fails"
    dp = potential.DehnenBarPotential().toPlanar()
    dwp = potential.DehnenSmoothWrapperPotential(pot=dp)
    assert print(dwp) is None, "Printing a 2D wrapper potential fails"
    return None


def test_dissipative_ignoreInPotentialDensity2ndDerivs():
    # Test that dissipative forces are ignored when they are included in lists
    # given to evaluatePotentials, evaluateDensities, and evaluate2ndDerivs
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, b=0.8)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=lp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    R, z = 2.0, 0.4
    assert (
        numpy.fabs(
            potential.evaluatePotentials([lp, cdfc], R, z, phi=1.0)
            - potential.evaluatePotentials([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluatePotentials"
    assert (
        numpy.fabs(
            potential.evaluateDensities([lp, cdfc], R, z, phi=1.0)
            - potential.evaluateDensities([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluateDensities"
    assert (
        numpy.fabs(
            potential.evaluateR2derivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluateR2derivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluateR2derivs"
    assert (
        numpy.fabs(
            potential.evaluatez2derivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluatez2derivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluatez2derivs"
    assert (
        numpy.fabs(
            potential.evaluateRzderivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluateRzderivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluateRzderivs"
    assert (
        numpy.fabs(
            potential.evaluatephi2derivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluatephi2derivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluatephi2derivs"
    assert (
        numpy.fabs(
            potential.evaluateRphiderivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluateRphiderivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluateRphiderivs"
    assert (
        numpy.fabs(
            potential.evaluatephizderivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluatephizderivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluatephizderivs"
    assert (
        numpy.fabs(
            potential.evaluater2derivs([lp, cdfc], R, z, phi=1.0)
            - potential.evaluater2derivs([lp, cdfc], R, z, phi=1.0)
        )
        < 1e-10
    ), "Dissipative forces not ignored in evaluater2derivs"
    return None


def test_dissipative_noVelocityError():
    # Test that calling evaluateXforces for a dissipative potential
    # without including velocity produces an error
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, b=0.8)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=lp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    R, z, phi = 2.0, 0.4, 1.1
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluateRforces([lp, cdfc], R, z, phi=phi)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluatephitorques([lp, cdfc], R, z, phi=phi)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluatezforces([lp, cdfc], R, z, phi=phi)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluaterforces([lp, cdfc], R, z, phi=phi)
    return None


def test_dissipative_noVelocityError_2d():
    # Test that calling evaluateXforces for a dissipative potential
    # without including velocity produces an error in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, b=0.8).toPlanar()
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, const_lnLambda=8.0, dens=lp, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    ).toPlanar()
    R, z, phi = 2.0, 0.4, 1.1
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluateplanarRforces(lp + cdfc, R, phi=phi)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.evaluateplanarphitorques(lp + cdfc, R, phi=phi)
    return None


def test_NonInertialFrameForce_2d():
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    nip = potential.NonInertialFrameForce(Omega=0.5)
    # Total radial force on circular orbit at R=2. in the non-inertial frame should be zero
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(
                potential.toPlanarPotential(lp + nip),
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel radial force on circular orbit"
    # Also splitting them up
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(
                potential.toPlanarPotential(lp), 2.0, phi=0.0
            )
            + potential.evaluateplanarRforces(
                potential.toPlanarPotential(nip),
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel radial force on circular orbit"
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(
                potential.toPlanarPotential(lp), 2.0, phi=0.0
            )
            + potential.toPlanarPotential(nip).Rforce(
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel radial force on circular orbit"
    # also the total azimuthal force should be zero
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(
                potential.toPlanarPotential(lp + nip),
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel phi torque on circular orbit"
    # Also splitting them up
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(
                potential.toPlanarPotential(lp), 2.0, phi=0.0
            )
            + potential.evaluateplanarphitorques(
                potential.toPlanarPotential(nip),
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel phi torque on circular orbit"
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(
                potential.toPlanarPotential(lp), 2.0, phi=0.0
            )
            + potential.toPlanarPotential(nip).phitorque(
                2.0,
                phi=0.0,
                v=[0.0, 0.0],
            )
        )
        < 1e-10
    ), "Non-inertial frame force does not cancel phi torque on circular orbit"
    return None


def test_RingPotential_correctPotentialIntegral():
    # Test that the RingPotential's potential is correct, by comparing it to a
    # direct integral solution of the Poisson equation
    from scipy import integrate, special

    # Direct solution
    def pot(R, z, amp=1.0, a=0.75):
        return (
            -amp
            * integrate.quad(
                lambda k: special.jv(0, k * R)
                * special.jv(0, k * a)
                * numpy.exp(-k * numpy.fabs(z)),
                0.0,
                numpy.inf,
            )[0]
        )

    rp = potential.RingPotential(amp=3.0, a=0.75)
    # Just check a bunch of (R,z)s; z=0 the direct integration doesn't work well, so we don't check that
    Rs, zs = [1.2, 1.2, 0.2, 0.2], [0.1, -1.1, -0.1, 1.1]
    for R, z in zip(Rs, zs):
        assert (
            numpy.fabs(pot(R, z, amp=3.0) - rp(R, z)) < 1e-8
        ), f"RingPotential potential evaluation does not agree with direct integration at (R,z) = ({R},{z})"
    return None


def test_DehnenSmoothWrapper_decay():
    # Test that DehnenSmoothWrapperPotential with decay=True is the opposite
    # of decay=False
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    pot_grow = potential.DehnenSmoothWrapperPotential(pot=lp, tform=4.0, tsteady=3.0)
    pot_decay = potential.DehnenSmoothWrapperPotential(
        pot=lp, tform=4.0, tsteady=3.0, decay=True
    )
    ts = numpy.linspace(0.0, 10.0, 1001)
    assert (
        numpy.amax(
            numpy.fabs(
                lp(2.0, 0.0, ts)
                - [pot_grow(2.0, 0.0, t=t) + pot_decay(2.0, 0.0, t=t) for t in ts]
            )
        )
        < 1e-10
    ), "DehnenSmoothWrapper with decay=True is not the opposite of the same with decay=False"
    assert (
        numpy.amax(
            numpy.fabs(
                lp.Rforce(2.0, 0.0, ts)
                - [
                    pot_grow.Rforce(2.0, 0.0, t=t) + pot_decay.Rforce(2.0, 0.0, t=t)
                    for t in ts
                ]
            )
        )
        < 1e-10
    ), "DehnenSmoothWrapper with decay=True is not the opposite of the same with decay=False"
    return None


def test_AdiabaticContractionWrapper():
    # Some basic tests of adiabatic contraction
    dm1 = AdiabaticContractionWrapperPotential(
        pot=potential.MWPotential2014[2],
        baryonpot=potential.MWPotential2014[:2],
        f_bar=None,
        method="cautun",
    )
    dm2 = AdiabaticContractionWrapperPotential(
        pot=potential.MWPotential2014[2],
        baryonpot=potential.MWPotential2014[:2],
        f_bar=0.157,
        method="cautun",
    )
    dm3 = AdiabaticContractionWrapperPotential(
        pot=potential.MWPotential2014[2],
        baryonpot=potential.MWPotential2014[:2],
        f_bar=0.157,
        method="blumenthal",
    )
    dm4 = AdiabaticContractionWrapperPotential(
        pot=potential.MWPotential2014[2],
        baryonpot=potential.MWPotential2014[:2],
        f_bar=0.157,
        method="gnedin",
    )
    # at large r, the contraction should be almost negligible (1% for Cautun)
    r = 50.0
    assert (
        numpy.fabs(dm1.vcirc(r) / potential.MWPotential2014[2].vcirc(r) - 1.02) < 1e-2
    ), '"cautun" adiabatic contraction at large distances'
    assert (
        numpy.fabs(dm2.vcirc(r) / potential.MWPotential2014[2].vcirc(r) - 0.97) < 1e-2
    ), '"cautun" adiabatic contraction at large distances'
    assert (
        numpy.fabs(dm3.vcirc(r) / potential.MWPotential2014[2].vcirc(r) - 0.98) < 1e-2
    ), '"blumenthal" adiabatic contraction at large distances'
    assert (
        numpy.fabs(dm4.vcirc(r) / potential.MWPotential2014[2].vcirc(r) - 0.98) < 1e-2
    ), '"gnedin" adiabatic contraction at large distances'
    # For MWPotential2014, contraction at 1 kpc should be about 4 in mass for
    # Cautun (their Fig. 2; Mstar ~ 7e10 Msun)
    r = 1.0 / dm1._ro
    assert (
        numpy.fabs(dm1.mass(r) / potential.MWPotential2014[2].mass(r) - 3.40) < 1e-2
    ), '"cautun" adiabatic contraction does not agree at R ~ 1 kpc'
    assert (
        numpy.fabs(dm2.mass(r) / potential.MWPotential2014[2].mass(r) - 3.18) < 1e-2
    ), '"cautun" adiabatic contraction does not agree at R ~ 1 kpc'
    assert (
        numpy.fabs(dm3.mass(r) / potential.MWPotential2014[2].mass(r) - 4.22) < 1e-2
    ), '"blumenthal" adiabatic contraction does not agree at R ~ 1 kpc'
    assert (
        numpy.fabs(dm4.mass(r) / potential.MWPotential2014[2].mass(r) - 4.04) < 1e-2
    ), '"gnedin" adiabatic contraction does not agree at R ~ 1 kpc'
    # At 10 kpc, it should be more like 2
    r = 10.0 / dm1._ro
    assert (
        numpy.fabs(dm1.mass(r) / potential.MWPotential2014[2].mass(r) - 1.78) < 1e-2
    ), '"cautun" adiabatic contraction does not agree at R ~ 10 kpc'
    assert (
        numpy.fabs(dm2.mass(r) / potential.MWPotential2014[2].mass(r) - 1.64) < 1e-2
    ), '"cautun" adiabatic contraction does not agree at R ~ 10 kpc'
    assert (
        numpy.fabs(dm3.mass(r) / potential.MWPotential2014[2].mass(r) - 1.67) < 1e-2
    ), '"blumenthal" adiabatic contraction does not agree at R ~ 10 kpc'
    assert (
        numpy.fabs(dm4.mass(r) / potential.MWPotential2014[2].mass(r) - 1.43) < 1e-2
    ), '"gnedin" adiabatic contraction does not agree at R ~ 10 kpc'
    return None


def test_RotateAndTiltWrapper():
    # some tests of the rotate and tilt wrapper
    zvec = numpy.array([numpy.sqrt(1 / 3.0), numpy.sqrt(1 / 3.0), numpy.sqrt(1 / 3.0)])
    zvec /= numpy.sqrt(numpy.sum(zvec**2))
    rot = _rotate_to_arbitrary_vector(numpy.array([[0.0, 0.0, 1.0]]), zvec, inv=True)[0]
    galaxy_pa = 0.3
    pa_rot = numpy.array(
        [
            [numpy.cos(galaxy_pa), numpy.sin(galaxy_pa), 0.0],
            [-numpy.sin(galaxy_pa), numpy.cos(galaxy_pa), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = numpy.dot(pa_rot, rot)
    xyz_test = numpy.array([0.5, 0.5, 0.5])
    Rphiz_test = coords.rect_to_cyl(xyz_test[0], xyz_test[1], xyz_test[2])
    txyz_test = numpy.dot(rot, xyz_test)
    tRphiz_test = coords.rect_to_cyl(txyz_test[0], txyz_test[1], txyz_test[2])
    testpot = potential.RotateAndTiltWrapperPotential(
        zvec=zvec, galaxy_pa=galaxy_pa, pot=potential.MWPotential2014
    )
    # test against the transformed potential and a MWPotential evaluated at the transformed coords
    assert (
        (
            evaluatePotentials(testpot, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1])
            - evaluatePotentials(
                potential.MWPotential2014,
                tRphiz_test[0],
                tRphiz_test[2],
                phi=tRphiz_test[1],
            )
        )
        < 1e-6
    ), "Evaluating potential at same relative position in a Rotated and tilted MWPotential2014 and non-Rotated does not give same result"
    # Also a triaxial NFW
    NFW_wrapped = potential.RotateAndTiltWrapperPotential(
        zvec=zvec,
        galaxy_pa=galaxy_pa,
        pot=potential.TriaxialNFWPotential(amp=1.0, b=0.7, c=0.5),
    )
    NFW_rot = potential.TriaxialNFWPotential(
        amp=1.0, zvec=zvec, pa=galaxy_pa, b=0.7, c=0.5
    )
    assert (
        (
            evaluatePotentials(
                NFW_wrapped, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
            - evaluatePotentials(
                NFW_rot, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
        )
        < 1e-6
    ), "Wrapped and Internally rotated NFW potentials do not match when evaluated at the same point"
    # Try not specifying galaxy_pa, shouldn be =0
    NFW_wrapped = potential.RotateAndTiltWrapperPotential(
        zvec=zvec, pot=potential.TriaxialNFWPotential(amp=1.0, b=0.7, c=0.5)
    )
    NFW_rot = potential.TriaxialNFWPotential(amp=1.0, zvec=zvec, pa=0.0, b=0.7, c=0.5)
    assert (
        (
            evaluatePotentials(
                NFW_wrapped, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
            - evaluatePotentials(
                NFW_rot, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
        )
        < 1e-6
    ), "Wrapped and Internally rotated NFW potentials do not match when evaluated at the same point"
    # Try not specifying zvec, should be =[0,0,1]
    NFW_wrapped = potential.RotateAndTiltWrapperPotential(
        galaxy_pa=galaxy_pa, pot=potential.TriaxialNFWPotential(amp=1.0, b=0.7, c=0.5)
    )
    NFW_rot = potential.TriaxialNFWPotential(
        amp=1.0, zvec=[0.0, 0.0, 1.0], pa=galaxy_pa, b=0.7, c=0.5
    )
    assert (
        (
            evaluatePotentials(
                NFW_wrapped, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
            - evaluatePotentials(
                NFW_rot, Rphiz_test[0], Rphiz_test[2], phi=Rphiz_test[1]
            )
        )
        < 1e-6
    ), "Wrapped and Internally rotated NFW potentials do not match when evaluated at the same point"
    # make sure the offset works as intended
    # triaxial NFW at x,y,z = [20.,0.,3.]
    NFW_wrapped = potential.RotateAndTiltWrapperPotential(
        zvec=zvec,
        galaxy_pa=galaxy_pa,
        offset=[20.0, 0.0, 3.0],
        pot=potential.TriaxialNFWPotential(amp=1.0, b=0.7, c=0.5),
    )
    NFW_rot = potential.TriaxialNFWPotential(
        amp=1.0, zvec=zvec, pa=galaxy_pa, b=0.7, c=0.5
    )
    assert (
        (
            evaluatePotentials(NFW_wrapped, 0.0, 0.0, phi=0.0)
            - evaluatePotentials(NFW_rot, 20.0, -3.0, phi=numpy.pi)
        )
        < 1e-6
    ), "Wrapped + Offset and Internally rotated NFW potentials do not match when evaluated at the same point"


def test_RotateAndTiltWrapper_pa_inclination_rotation_matrix():
    # Test that the formula for the rotation matrix given in the documentation agrees with the code
    def rotation_matrix_docs(galaxy_pa, inclination, sky_pa):
        return numpy.dot(
            numpy.array(
                [
                    [numpy.cos(galaxy_pa), numpy.sin(galaxy_pa), 0.0],
                    [-numpy.sin(galaxy_pa), numpy.cos(galaxy_pa), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            numpy.dot(
                numpy.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, numpy.cos(inclination), numpy.sin(inclination)],
                        [0.0, -numpy.sin(inclination), numpy.cos(inclination)],
                    ]
                ),
                numpy.array(
                    [
                        [numpy.sin(sky_pa), -numpy.cos(sky_pa), 0.0],
                        [numpy.cos(sky_pa), numpy.sin(sky_pa), 0.0],
                        [0.0, 0.0, 1],
                    ]
                ),
            ),
        )

    galaxy_pa, inclination, sky_pa = 0.3, -0.2, 0.1
    rtwp = potential.RotateAndTiltWrapperPotential(
        pot=potential.MWPotential2014,
        galaxy_pa=galaxy_pa,
        inclination=inclination,
        sky_pa=sky_pa,
    )
    assert numpy.all(
        numpy.fabs(rtwp._rot - rotation_matrix_docs(galaxy_pa, inclination, sky_pa))
        < 1e-10
    ), "Rotation matrix in RotateAndTiltWrapperPotential does not agree with the formula in the documentation"

    galaxy_pa, inclination, sky_pa = -0.3, 1.2, 2.1
    rtwp = potential.RotateAndTiltWrapperPotential(
        pot=potential.MWPotential2014,
        galaxy_pa=galaxy_pa,
        inclination=inclination,
        sky_pa=sky_pa,
    )
    assert numpy.all(
        numpy.fabs(rtwp._rot - rotation_matrix_docs(galaxy_pa, inclination, sky_pa))
        < 1e-10
    ), "Rotation matrix in RotateAndTiltWrapperPotential does not agree with the formula in the documentation"
    return None


def test_integration_RotateAndTiltWrapper():
    ## test a quick orbit integration to hit the C code (also test pure python)
    # two potentials, one offset
    offset = [3.0, 2.0, 1.0]
    mwpot = potential.MWPotential2014
    mwpot_wrapped = potential.RotateAndTiltWrapperPotential(
        pot=potential.MWPotential2014, offset=offset
    )
    # initialise orbit
    ro = 8.0
    orb = orbit.Orbit(ro=ro)
    # another, offset by the same as the potential
    init = orb.vxvv[0]
    R, vR, vT, z, vz, phi = init
    x, y, z = coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    tx, ty, tz = x - offset[0], y - offset[1], z - offset[2]
    tR, tphi, tz = coords.rect_to_cyl(tx, ty, tz)
    tvR, tvT, tvz = coords.rect_to_cyl_vec(vx, vy, vz, tR, tphi, tz, cyl=True)
    orb_t = orbit.Orbit([tR, tvR, tvT, tz, tvz, tphi], ro=ro)
    # integrate
    ts = numpy.linspace(0.0, 1.0, 1000)
    orb.integrate(ts, pot=mwpot, method="dop853")
    orb_t.integrate(ts, pot=mwpot_wrapped, method="dop853")
    # translate other orbit to match first one:
    orb_vxvv = orb_t.getOrbit()
    R, vR, vT, z, vz, phi = (
        orb_vxvv[:, 0],
        orb_vxvv[:, 1],
        orb_vxvv[:, 2],
        orb_vxvv[:, 3],
        orb_vxvv[:, 4],
        orb_vxvv[:, 5],
    )
    x, y, z = coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    tx, ty, tz = x + offset[0], y + offset[1], z + offset[2]
    tR, tphi, tz = coords.rect_to_cyl(tx, ty, tz)
    # check equal
    Rphi = numpy.dstack([orb.R(ts), orb.z(ts)])[0]
    Rphi_t = numpy.dstack([tR * ro, tz * ro])[0]
    assert numpy.all(
        numpy.fabs(Rphi - Rphi_t) < 10.0**-10
    ), "Pure python orbit integration in an offset potential does not work as expected"
    # reinitialise orbits, just to be sure
    orb = orbit.Orbit(ro=ro)
    init = orb.vxvv[0]
    R, vR, vT, z, vz, phi = init
    offset = [3.0, 2.0, 1.0]
    x, y, z = coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    tx, ty, tz = x - offset[0], y - offset[1], z - offset[2]
    tR, tphi, tz = coords.rect_to_cyl(tx, ty, tz)
    tvR, tvT, tvz = coords.rect_to_cyl_vec(vx, vy, vz, tR, tphi, tz, cyl=True)
    orb_t = orbit.Orbit([tR, tvR, tvT, tz, tvz, tphi], ro=ro)
    # integrate, use C
    orb.integrate(ts, pot=mwpot, method="dop853_c")
    orb_t.integrate(ts, pot=mwpot_wrapped, method="dop853_c")
    orb_vxvv = orb_t.getOrbit()
    R, vR, vT, z, vz, phi = (
        orb_vxvv[:, 0],
        orb_vxvv[:, 1],
        orb_vxvv[:, 2],
        orb_vxvv[:, 3],
        orb_vxvv[:, 4],
        orb_vxvv[:, 5],
    )
    x, y, z = coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = coords.cyl_to_rect_vec(vR, vT, vz, phi)
    tx, ty, tz = x + offset[0], y + offset[1], z + offset[2]
    tR, tphi, tz = coords.rect_to_cyl(tx, ty, tz)
    # check equal
    Rphi = numpy.dstack([orb.R(ts), orb.z(ts)])[0]
    Rphi_t = numpy.dstack([tR * ro, tz * ro])[0]
    assert numpy.all(
        numpy.fabs(Rphi - Rphi_t) < 10.0**-10
    ), "C orbit integration in an offset potential does not work as expected"
    return None


def test_vtermnegl_issue314():
    # Test related to issue 314: vterm for negative l
    rp = potential.RazorThinExponentialDiskPotential(normalize=1.0, hr=3.0 / 8.0)
    assert (
        numpy.fabs(rp.vterm(0.5) + rp.vterm(-0.5)) < 10.0**-8.0
    ), "vterm for negative l does not behave as expected"
    return None


def test_Ferrers_Rzderiv_issue319():
    # Test that the Rz derivative works for the FerrersPotential (issue 319)
    fp = potential.FerrersPotential(normalize=1.0)
    from test_SpiralArmsPotential import deriv as derivative

    rzderiv = fp.Rzderiv(0.5, 0.2, phi=1.0)
    rzderiv_finitediff = derivative(
        lambda x: -fp.zforce(x, 0.2, phi=1.0), 0.5, dx=10.0**-8.0
    )
    assert (
        numpy.fabs(rzderiv - rzderiv_finitediff) < 10.0**-7.0
    ), "Rzderiv for FerrersPotential does not agree with finite-difference calculation"
    return None


def test_rtide():
    # Test that rtide is being calculated properly in select potentials
    lp = potential.LogarithmicHaloPotential()
    assert (
        abs(1.0 - lp.rtide(1.0, 0.0, M=1.0) / 0.793700525984) < 10.0**-12.0
    ), "Calculation of rtide in logarithmic potential fails"
    pmass = potential.PlummerPotential(b=0.0)
    assert (
        abs(1.0 - pmass.rtide(1.0, 0.0, M=1.0) / 0.693361274351) < 10.0**-12.0
    ), "Calculation of rtide in point-mass potential fails"
    # Also test function interface
    assert (
        abs(1.0 - potential.rtide([lp], 1.0, 0.0, M=1.0) / 0.793700525984) < 10.0**-12.0
    ), "Calculation of rtide in logarithmic potential fails"
    pmass = potential.PlummerPotential(b=0.0)
    assert (
        abs(1.0 - potential.rtide([pmass], 1.0, 0.0, M=1.0) / 0.693361274351)
        < 10.0**-12.0
    ), "Calculation of rtide in point-mass potential fails"
    return None


def test_rtide_noMError():
    # Test the running rtide without M= input raises error
    lp = potential.LogarithmicHaloPotential()
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = lp.rtide(1.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.rtide([lp], 1.0, 0.0)
    return None


def test_ttensor():
    pmass = potential.KeplerPotential(normalize=1.0)
    tij = pmass.ttensor(1.0, 0.0, 0.0)
    # Full tidal tensor here should be diag(2,-1,-1)
    assert numpy.all(
        numpy.fabs(tij - numpy.diag([2, -1, -1])) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    # Also test eigenvalues
    tij = pmass.ttensor(1.0, 0.0, 0.0, eigenval=True)
    assert numpy.all(
        numpy.fabs(tij - numpy.array([2, -1, -1])) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    # Also test function interface
    tij = potential.ttensor([pmass], 1.0, 0.0, 0.0)
    # Full tidal tensor here should be diag(2,-1,-1)
    assert numpy.all(
        numpy.fabs(tij - numpy.diag([2, -1, -1])) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    # Also test eigenvalues
    tij = potential.ttensor([pmass], 1.0, 0.0, 0.0, eigenval=True)
    assert numpy.all(
        numpy.fabs(tij - numpy.array([2, -1, -1])) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    # Also Test symmetry when y!=0 and z!=0
    tij = potential.ttensor([pmass], 1.0, 1.0, 1.0)
    assert numpy.all(
        numpy.fabs(tij[0][1] - tij[1][0]) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    assert numpy.all(
        numpy.fabs(tij[0][2] - tij[2][0]) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    assert numpy.all(
        numpy.fabs(tij[1][2] - tij[2][1]) < 1e-10
    ), "Calculation of tidal tensor in point-mass potential fails"
    return None


def test_ttensor_trace():
    # Test that the trace of the tidal tensor == -4piG density for a bunch of
    # potentials
    pots = [
        potential.KeplerPotential(normalize=1.0),
        potential.LogarithmicHaloPotential(normalize=3.0, q=0.8),
        potential.MiyamotoNagaiPotential(normalize=0.5, a=3.0, b=0.5),
    ]
    R, z, phi = 1.3, -0.2, 2.0
    for pot in pots:
        assert (
            numpy.fabs(
                numpy.trace(pot.ttensor(R, z, phi=phi))
                + 4.0 * numpy.pi * pot.dens(R, z, phi=phi)
            )
            < 1e-10
        ), "Trace of the tidal tensor not equal 4piG density"
    # Also test a list
    assert (
        numpy.fabs(
            numpy.trace(potential.ttensor(potential.MWPotential2014, R, z, phi=phi))
            + 4.0
            * numpy.pi
            * potential.evaluateDensities(potential.MWPotential2014, R, z, phi=phi)
        )
        < 1e-10
    ), "Trace of the tidal tensor not equal 4piG density"
    return None


def test_ttensor_nonaxi():
    # Test that computing the tidal tensor for a non-axi potential raises error
    lp = potential.LogarithmicHaloPotential(normalize=1.0, b=0.8, q=0.7)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = lp.ttensor(1.0, 0.0, 0.0)
    with pytest.raises(potential.PotentialError) as excinfo:
        dummy = potential.ttensor(lp, 1.0, 0.0, 0.0)
    return None


# Test that zvc_range returns the range over which the zvc is defined for a
# given E,Lz
def test_zvc_range():
    E, Lz = -1.25, 0.6
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rmin, 0.0)
            + Lz**2.0 / 2.0 / Rmin**2.0
            - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rmax, 0.0)
            + Lz**2.0 / 2.0 / Rmax**2.0
            - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    R_a_little_less = Rmin - 1e-4
    assert (
        potential.evaluatePotentials(potential.MWPotential2014, R_a_little_less, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_less**2.0
        > E
    ), "zvc_range does not give the minimum R for which Phi_eff(R,0) < E"
    R_a_little_more = Rmax + 1e-4
    assert (
        potential.evaluatePotentials(potential.MWPotential2014, R_a_little_more, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_more**2.0
        > E
    ), "zvc_range does not give the maximum R for which Phi_eff(R,0) < E"
    # Another one for good measure
    E, Lz = -2.25, 0.2
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rmin, 0.0)
            + Lz**2.0 / 2.0 / Rmin**2.0
            - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rmax, 0.0)
            + Lz**2.0 / 2.0 / Rmax**2.0
            - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    R_a_little_less = Rmin - 1e-4
    assert (
        potential.evaluatePotentials(potential.MWPotential2014, R_a_little_less, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_less**2.0
        > E
    ), "zvc_range does not give the minimum R for which Phi_eff(R,0) < E"
    R_a_little_more = Rmax + 1e-4
    assert (
        potential.evaluatePotentials(potential.MWPotential2014, R_a_little_more, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_more**2.0
        > E
    ), "zvc_range does not give the maximum R for which Phi_eff(R,0) < E"
    # Also one for a single potential
    pot = potential.PlummerPotential(normalize=True)
    E, Lz = -1.9, 0.2
    Rmin, Rmax = pot.zvc_range(E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, Rmin, 0.0) + Lz**2.0 / 2.0 / Rmin**2.0 - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, Rmax, 0.0) + Lz**2.0 / 2.0 / Rmax**2.0 - E
        )
        < 1e-8
    ), "zvc_range does not return radius at which Phi_eff(R,0) = E"
    R_a_little_less = Rmin - 1e-4
    assert (
        potential.evaluatePotentials(pot, R_a_little_less, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_less**2.0
        > E
    ), "zvc_range does not give the minimum R for which Phi_eff(R,0) < E"
    R_a_little_more = Rmax + 1e-4
    assert (
        potential.evaluatePotentials(pot, R_a_little_more, 0.0)
        + Lz**2.0 / 2.0 / R_a_little_more**2.0
        > E
    ), "zvc_range does not give the maximum R for which Phi_eff(R,0) < E"
    return None


# Test that we get [NaN,NaN] when there are no orbits for this combination of E and Lz
def test_zvc_range_undefined():
    # Set up circular orbit at Rc, then ask for Lz > Lzmax(E)
    Rc = 0.6653
    E = (
        potential.evaluatePotentials(potential.MWPotential2014, Rc, 0.0)
        + potential.vcirc(potential.MWPotential2014, Rc) ** 2.0 / 2.0
    )
    Lzmax = Rc * potential.vcirc(potential.MWPotential2014, Rc)
    assert numpy.all(
        numpy.isnan(potential.zvc_range(potential.MWPotential2014, E, Lzmax + 1e-4))
    ), "zvc_range does not return [NaN,NaN] when no orbits exist at this combination of (E,Lz)"
    return None


def test_zvc_at_rminmax():
    E, Lz = -1.25, 0.6
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert (
        numpy.fabs(potential.zvc(potential.MWPotential2014, Rmin, E, Lz)) < 1e-8
    ), "zvc at minimum from zvc_range is not at zero height"
    assert (
        numpy.fabs(potential.zvc(potential.MWPotential2014, Rmax, E, Lz)) < 1e-8
    ), "zvc at maximum from zvc_range is not at zero height"
    # Another one for good measure
    E, Lz = -2.25, 0.2
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert (
        numpy.fabs(potential.zvc(potential.MWPotential2014, Rmin, E, Lz)) < 1e-8
    ), "zvc at minimum from zvc_range is not at zero height"
    assert (
        numpy.fabs(potential.zvc(potential.MWPotential2014, Rmax, E, Lz)) < 1e-8
    ), "zvc at maximum from zvc_range is not at zero height"
    # Also for a single potential
    pot = potential.PlummerPotential(normalize=True)
    E, Lz = -1.9, 0.2
    Rmin, Rmax = pot.zvc_range(E, Lz)
    assert (
        numpy.fabs(pot.zvc(Rmin, E, Lz)) < 1e-8
    ), "zvc at minimum from zvc_range is not at zero height"
    assert (
        numpy.fabs(pot.zvc(Rmax, E, Lz)) < 1e-8
    ), "zvc at maximum from zvc_range is not at zero height"
    return None


def test_zvc():
    E, Lz = -1.25, 0.6
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    Rtrial = 0.5 * (Rmin + Rmax)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.25 * (Rmax - Rmin)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.75 * (Rmax - Rmin)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    # Another one for good measure
    E, Lz = -2.25, 0.2
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    Rtrial = 0.5 * (Rmin + Rmax)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.25 * (Rmax - Rmin)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.75 * (Rmax - Rmin)
    ztrial = potential.zvc(potential.MWPotential2014, Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(potential.MWPotential2014, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    # Also for a single potential
    pot = potential.PlummerPotential(normalize=True)
    E, Lz = -1.9, 0.2
    Rmin, Rmax = pot.zvc_range(E, Lz)
    Rtrial = 0.5 * (Rmin + Rmax)
    ztrial = pot.zvc(Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.25 * (Rmax - Rmin)
    ztrial = pot.zvc(Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    Rtrial = Rmin + 0.75 * (Rmax - Rmin)
    ztrial = pot.zvc(Rtrial, E, Lz)
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, Rtrial, ztrial)
            + Lz**2.0 / 2.0 / Rtrial**2.0
            - E
        )
        < 1e-8
    ), "zvc does not return the height at which Phi_eff(R,z) = E"
    return None


# Test that zvc outside of zvc_range is NaN
def test_zvc_undefined():
    E, Lz = -1.25, 0.6
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert numpy.isnan(
        potential.zvc(potential.MWPotential2014, Rmin - 1e-4, E, Lz)
    ), "zvc at R < Rmin is not NaN"
    assert numpy.isnan(
        potential.zvc(potential.MWPotential2014, Rmax + 1e-4, E, Lz)
    ), "zvc at R > Rmax is not NaN"
    # Another one for good measure
    E, Lz = -2.25, 0.2
    Rmin, Rmax = potential.zvc_range(potential.MWPotential2014, E, Lz)
    assert numpy.isnan(
        potential.zvc(potential.MWPotential2014, Rmin - 1e-4, E, Lz)
    ), "zvc at R < Rmin is not NaN"
    assert numpy.isnan(
        potential.zvc(potential.MWPotential2014, Rmax + 1e-4, E, Lz)
    ), "zvc at R > Rmax is not NaN"
    return None


# Check that we get the correct ValueError if no solution can be found
def test_zvc_valueerror():
    E, Lz = -1.25 + 100, 0.6
    with pytest.raises(ValueError) as excinfo:
        potential.zvc(potential.MWPotential2014, 0.7, E + 100, Lz)
    return None


def test_rhalf():
    # Test some known cases
    a = numpy.pi
    # Hernquist, r12= (1+sqrt(2))a
    hp = potential.HernquistPotential(amp=1.0, a=a)
    assert (
        numpy.fabs(hp.rhalf() - (1.0 + numpy.sqrt(2.0)) * a) < 1e-10
    ), "Half-mass radius of the Hernquist potential incorrect"
    # DehnenSpherical, r12= a/(2^(1/(3-alpha)-1)
    alpha = 1.34
    hp = potential.DehnenSphericalPotential(amp=1.0, a=a, alpha=alpha)
    assert (
        numpy.fabs(hp.rhalf() - a / (2 ** (1.0 / (3.0 - alpha)) - 1.0)) < 1e-10
    ), "Half-mass radius of the DehnenSpherical potential incorrect"
    # Plummer, r12= b/sqrt(1/0.5^(2/3)-1)
    pp = potential.PlummerPotential(amp=1.0, b=a)
    assert (
        numpy.fabs(potential.rhalf(pp) - a / numpy.sqrt(0.5 ** (-2.0 / 3.0) - 1.0))
        < 1e-10
    ), "Half-mass radius of the Plummer potential incorrect"
    return None


def test_tdyn():
    # Spherical: tdyn = 2piR/vc
    a = numpy.pi
    # Hernquist
    hp = potential.HernquistPotential(amp=1.0, a=a)
    R = 1.4
    assert (
        numpy.fabs(hp.tdyn(R) - 2.0 * numpy.pi * R / hp.vcirc(R)) < 1e-10
    ), "Dynamical time of the Hernquist potential incorrect"
    # DehnenSpherical
    alpha = 1.34
    hp = potential.DehnenSphericalPotential(amp=1.0, a=a, alpha=alpha)
    assert (
        numpy.fabs(potential.tdyn(hp, R) - 2.0 * numpy.pi * R / hp.vcirc(R)) < 1e-10
    ), "Dynamical time of the DehnenSpherical potential incorrect"
    # Axi, this approx. holds
    hp = potential.MiyamotoNagaiPotential(amp=1.0, a=a, b=a / 5.0)
    R = 3.4
    assert (
        numpy.fabs(hp.tdyn(R) / (2.0 * numpy.pi * R / hp.vcirc(R)) - 1.0) < 0.03
    ), "Dynamical time of the Miyamoto-Nagai potential incorrect"
    return None


def test_NumericalPotentialDerivativesMixin():
    # Test that the NumericalPotentialDerivativesMixin works as expected
    def get_mixin_first_instance(cls, *args, **kwargs):
        # Function to return instance of a class for Potential cls where
        # the NumericalPotentialDerivativesMixin comes first, so all derivs
        # are numerical (should otherwise always be used second!)
        class NumericalPot(potential.NumericalPotentialDerivativesMixin, cls):
            def __init__(self, *args, **kwargs):
                potential.NumericalPotentialDerivativesMixin.__init__(self, kwargs)
                cls.__init__(self, *args, **kwargs)

        return NumericalPot(*args, **kwargs)

    # Function to check all numerical derivatives
    def check_numerical_derivs(Pot, NumPot, tol=1e-6, tol2=1e-5):
        # tol: tolerance for forces, tol2: tolerance for 2nd derivatives
        # Check wide range of R,z,phi
        Rs = numpy.array([0.5, 1.0, 2.0])
        Zs = numpy.array([0.0, 0.125, -0.125, 0.25, -0.25])
        phis = numpy.array(
            [0.0, 0.5, -0.5, 1.0, -1.0, numpy.pi, 0.5 + numpy.pi, 1.0 + numpy.pi]
        )
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                for kk in range(len(phis)):
                    # Forces
                    assert (
                        numpy.fabs(
                            (
                                Pot.Rforce(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.Rforce(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.Rforce(Rs[ii], Zs[jj], phi=phis[kk])
                        )
                        < tol
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct Rforce"
                    assert (
                        numpy.fabs(
                            (
                                Pot.zforce(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.zforce(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.zforce(Rs[ii], Zs[jj], phi=phis[kk]) ** (Zs[jj] > 0.0)
                        )
                        < tol
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct zforce"
                    assert (
                        numpy.fabs(
                            (
                                Pot.phitorque(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.phitorque(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.phitorque(Rs[ii], Zs[jj], phi=phis[kk])
                            ** Pot.isNonAxi
                        )
                        < tol
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct phitorque"
                    # Second derivatives
                    assert (
                        numpy.fabs(
                            (
                                Pot.R2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.R2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.R2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct R2deriv"
                    assert (
                        numpy.fabs(
                            (
                                Pot.z2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.z2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.z2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct z2deriv"
                    assert (
                        numpy.fabs(
                            (
                                Pot.phi2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.phi2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.phi2deriv(Rs[ii], Zs[jj], phi=phis[kk])
                            ** Pot.isNonAxi
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct phi2deriv"
                    assert (
                        numpy.fabs(
                            (
                                Pot.Rzderiv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.Rzderiv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.Rzderiv(Rs[ii], Zs[jj], phi=phis[kk])
                            ** (Zs[jj] > 0.0)
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct Rzderiv"
                    assert (
                        numpy.fabs(
                            (
                                Pot.Rphideriv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.Rphideriv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.Rphideriv(Rs[ii], Zs[jj], phi=phis[kk])
                            ** Pot.isNonAxi
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct Rphideriv"
                    assert (
                        numpy.fabs(
                            (
                                Pot.phizderiv(Rs[ii], Zs[jj], phi=phis[kk])
                                - NumPot.phizderiv(Rs[ii], Zs[jj], phi=phis[kk])
                            )
                            / Pot.phizderiv(Rs[ii], Zs[jj], phi=phis[kk])
                            ** (Pot.isNonAxi * (Zs[jj] != 0.0))
                        )
                        < tol2
                    ), f"NumericalPotentialDerivativesMixin applied to {Pot.__class__.__name__} Potential does not give the correct phizderiv"
        return None

    # Now check some potentials
    # potential.MiyamotoNagaiPotential
    mp = potential.MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.05)
    num_mp = get_mixin_first_instance(
        potential.MiyamotoNagaiPotential, amp=1.0, a=0.5, b=0.05
    )
    check_numerical_derivs(mp, num_mp)
    # potential.DehnenBarPotential
    dp = potential.DehnenBarPotential()
    num_dp = get_mixin_first_instance(potential.DehnenBarPotential)
    check_numerical_derivs(dp, num_dp)
    return None


# Test that we don't get the "FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated" numpy warning for the SCF potential; issue #347
def test_scf_tupleindexwarning():
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", FutureWarning)
        p = mockSCFZeeuwPotential()
        p.Rforce(1.0, 0.0)
    # another one reported by Nil, now problem is with array input
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", FutureWarning)
        p = mockSCFZeeuwPotential()
        p.Rforce(numpy.atleast_1d(1.0), numpy.atleast_1d(0.0))
    return None


# Test that conversion between xi and R works as expected
def test_scf_xiToR():
    from galpy.potential.SCFPotential import _RToxi, _xiToR

    a = numpy.pi
    r = 1.4
    assert (
        numpy.fabs(_xiToR(_RToxi(r, a=a), a=a) - r) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    xi = 1.3
    assert (
        numpy.fabs(_RToxi(_xiToR(xi, a=a), a=a) - xi) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    # Also for arrays
    r = numpy.linspace(0.1, 5.3, 21)
    assert numpy.all(
        numpy.fabs(_xiToR(_RToxi(r, a=a), a=a) - r) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    xi = numpy.linspace(-0.9, 0.9, 21)
    assert numpy.all(
        numpy.fabs(_RToxi(_xiToR(xi, a=a), a=a) - xi) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    # Check 0 and inf
    r = 0
    assert (
        numpy.fabs(_RToxi(r, a=a) + 1) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    xi = -1.0
    assert (
        numpy.fabs(_xiToR(xi, a=a)) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    r = numpy.inf
    assert (
        numpy.fabs(_RToxi(r, a=a) - 1) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    xi = 1.0
    assert numpy.isinf(
        _xiToR(xi, a=a)
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    # Also for arrays with zero and inf
    r = numpy.concatenate((numpy.linspace(0.0, 5.3, 21), [numpy.inf]))
    assert numpy.all(
        numpy.fabs(_xiToR(_RToxi(r, a=a), a=a)[:-1] - r[:-1]) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    assert (
        numpy.fabs(_RToxi(r, a=a)[-1] - 1.0) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    xi = numpy.linspace(-1, 1, 21)
    assert numpy.all(
        numpy.fabs(_RToxi(_xiToR(xi, a=a), a=a) - xi) < 1e-10
    ), "_RToxi and _xiToR aren't each other's inverse in  r <-> xi conversion used in SCF potential"
    return None


# Test that attempting to multiply or divide a potential by something other than a number raises an error
def test_mult_divide_error():
    # 3D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    with pytest.raises(TypeError) as excinfo:
        pot * [1.0, 2.0]
    with pytest.raises(TypeError) as excinfo:
        [1.0, 2.0] * pot
    with pytest.raises(TypeError) as excinfo:
        pot / [1.0, 2.0]
    # 2D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    with pytest.raises(TypeError) as excinfo:
        pot * [1.0, 2.0]
    with pytest.raises(TypeError) as excinfo:
        [1.0, 2.0] * pot
    with pytest.raises(TypeError) as excinfo:
        pot / [1.0, 2.0]
    # 1D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toVertical(1.1)
    with pytest.raises(TypeError) as excinfo:
        pot * [1.0, 2.0]
    with pytest.raises(TypeError) as excinfo:
        [1.0, 2.0] * pot
    with pytest.raises(TypeError) as excinfo:
        pot / [1.0, 2.0]
    return None


# Test that arithmetically adding potentials returns lists of potentials
def test_add_potentials():
    assert (
        potential.MWPotential2014
        == potential.MWPotential2014[0]
        + potential.MWPotential2014[1]
        + potential.MWPotential2014[2]
    ), "Potential addition of components of MWPotential2014 does not give MWPotential2014"
    # 3D
    pot1 = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    pot2 = potential.MiyamotoNagaiPotential(normalize=0.2, a=0.4, b=0.1)
    pot3 = potential.HernquistPotential(normalize=0.4, a=0.1)
    assert pot1 + pot2 == [pot1, pot2]
    assert pot1 + pot2 + pot3 == [pot1, pot2, pot3]
    assert (pot1 + pot2) + pot3 == [pot1, pot2, pot3]
    assert pot1 + (pot2 + pot3) == [pot1, pot2, pot3]
    # 2D
    pot1 = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    pot2 = potential.MiyamotoNagaiPotential(normalize=0.2, a=0.4, b=0.1).toPlanar()
    pot3 = potential.HernquistPotential(normalize=0.4, a=0.1).toPlanar()
    assert pot1 + pot2 == [pot1, pot2]
    assert pot1 + pot2 + pot3 == [pot1, pot2, pot3]
    assert (pot1 + pot2) + pot3 == [pot1, pot2, pot3]
    assert pot1 + (pot2 + pot3) == [pot1, pot2, pot3]
    # 1D
    pot1 = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toVertical(1.1)
    pot2 = potential.MiyamotoNagaiPotential(normalize=0.2, a=0.4, b=0.1).toVertical(1.1)
    pot3 = potential.HernquistPotential(normalize=0.4, a=0.1).toVertical(1.1)
    assert pot1 + pot2 == [pot1, pot2]
    assert pot1 + pot2 + pot3 == [pot1, pot2, pot3]
    assert (pot1 + pot2) + pot3 == [pot1, pot2, pot3]
    assert pot1 + (pot2 + pot3) == [pot1, pot2, pot3]
    return None


# Test that attempting to multiply or divide a potential by something other
# than a number raises a TypeError (test both left and right)
def test_add_potentials_error():
    # 3D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    with pytest.raises(TypeError) as excinfo:
        3 + pot
    with pytest.raises(TypeError) as excinfo:
        pot + 3
    # 2D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    with pytest.raises(TypeError) as excinfo:
        3 + pot
    with pytest.raises(TypeError) as excinfo:
        pot + 3
    # 1D
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toVertical(1.1)
    with pytest.raises(TypeError) as excinfo:
        3 + pot
    with pytest.raises(TypeError) as excinfo:
        pot + 3
    return None


# Test that adding potentials with incompatible unit systems raises an error
def test_add_potentials_unitserror():
    # 3D
    ro, vo = 8.0, 220.0
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, ro=ro, vo=vo)
    potro = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo)
    potvo = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9, ro=ro, vo=vo * 1.1)
    potrovo = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo * 1.1
    )
    with pytest.raises(AssertionError) as excinfo:
        pot + potro
    with pytest.raises(AssertionError) as excinfo:
        pot + potvo
    with pytest.raises(AssertionError) as excinfo:
        pot + potrovo
    with pytest.raises(AssertionError) as excinfo:
        potro + pot
    with pytest.raises(AssertionError) as excinfo:
        potvo + pot
    with pytest.raises(AssertionError) as excinfo:
        potrovo + pot
    # 2D
    pot = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro, vo=vo
    ).toPlanar()
    potro = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo
    ).toPlanar()
    potvo = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro, vo=vo * 1.1
    ).toPlanar()
    potrovo = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo * 1.1
    ).toPlanar()
    with pytest.raises(AssertionError) as excinfo:
        pot + potro
    with pytest.raises(AssertionError) as excinfo:
        pot + potvo
    with pytest.raises(AssertionError) as excinfo:
        pot + potrovo
    with pytest.raises(AssertionError) as excinfo:
        potro + pot
    with pytest.raises(AssertionError) as excinfo:
        potvo + pot
    with pytest.raises(AssertionError) as excinfo:
        potrovo + pot
    # 1D
    pot = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro, vo=vo
    ).toVertical(1.1)
    potro = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo
    ).toVertical(1.1)
    potvo = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro, vo=vo * 1.1
    ).toVertical(1.1)
    potrovo = potential.LogarithmicHaloPotential(
        normalize=1.0, q=0.9, ro=ro * 1.1, vo=vo * 1.1
    ).toVertical(1.1)
    with pytest.raises(AssertionError) as excinfo:
        pot + potro
    with pytest.raises(AssertionError) as excinfo:
        pot + potvo
    with pytest.raises(AssertionError) as excinfo:
        pot + potrovo
    with pytest.raises(AssertionError) as excinfo:
        potro + pot
    with pytest.raises(AssertionError) as excinfo:
        potvo + pot
    with pytest.raises(AssertionError) as excinfo:
        potrovo + pot
    return None


# Test unit handling of interpolated Spherical potentials
def test_interSphericalPotential_unithandling():
    pot = potential.HernquistPotential(amp=1.0, a=2.0, ro=8.3, vo=230.0)
    # Test that setting up the interpolated potential with inconsistent units
    # raises a RuntimeError
    with pytest.raises(RuntimeError):
        ipot = potential.interpSphericalPotential(
            rforce=pot, rgrid=numpy.geomspace(0.01, 5.0, 201), ro=7.5
        )
    with pytest.raises(RuntimeError):
        ipot = potential.interpSphericalPotential(
            rforce=pot, rgrid=numpy.geomspace(0.01, 5.0, 201), vo=210.0
        )
    # Check that units are properly transferred
    ipot = potential.interpSphericalPotential(
        rforce=pot, rgrid=numpy.geomspace(0.01, 5.0, 201)
    )
    assert (
        ipot._roSet
    ), "ro of interpSphericalPotential not set, even though that of parent was set"
    assert (
        ipot._ro == pot._ro
    ), "ro of interpSphericalPotential does not agree with that of the parent potential"
    assert (
        ipot._voSet
    ), "vo of interpSphericalPotential not set, even though that of parent was set"
    assert (
        ipot._vo == pot._vo
    ), "vo of interpSphericalPotential does not agree with that of the parent potential"
    return None


# Test that the amplitude of the isothermal disk potential is set correctly (issue #400)
def test_isodisk_amplitude_issue400():
    # Morgan's example
    z = numpy.linspace(-0.1, 0.1, 10001)
    pot = potential.IsothermalDiskPotential(amp=0.1, sigma=20.5 / 220.0)
    # Density at z=0 should be 0.1, no density or 2nd deriv for 1D at this
    # point, so manually compute
    z = numpy.linspace(-2e-4, 2e-4, 5)
    dens_at_0 = 1.0 / (numpy.pi * 4) * numpy.gradient(numpy.gradient(pot(z), z), z)[2]
    assert (
        numpy.fabs(dens_at_0 - 0.1) < 1e-7
    ), "Density at z=0 for IsothermalDiskPotential is not correct"
    return None


def test_TimeDependentAmplitudeWrapperPotential_against_DehnenSmooth():
    # Test that TimeDependentAmplitudeWrapperPotential acts the same as DehnenSmooth
    # Test = LogPot + DehnenBar grown smoothly
    # Both using the DehnenSmoothWrapper and the new TimeDependentAmplitudeWrapperPotential
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential()
    dbp = potential.DehnenBarPotential(tform=-100000.0, tsteady=1.0)
    dp = potential.DehnenSmoothWrapperPotential(pot=dbp)
    tp = potential.TimeDependentAmplitudeWrapperPotential(pot=dbp, A=dp._smooth)
    # Orbit of the Sun
    o = Orbit()
    ts = numpy.linspace(0.0, -20.0, 1001)
    o.integrate(ts, lp + dp)
    ott = o()
    ott.integrate(ts, lp + tp)
    tol = 1e-10
    assert (
        numpy.amax(numpy.fabs(o.x(ts) - ott.x(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.y(ts) - ott.y(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.z(ts) - ott.z(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.vx(ts) - ott.vx(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.vy(ts) - ott.vy(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.vz(ts) - ott.vz(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    return None


def test_TimeDependentAmplitudeWrapperPotential_against_DehnenSmooth_2d():
    # Test that TimeDependentAmplitudeWrapperPotential acts the same as DehnenSmooth
    # Test = LogPot + DehnenBar grown smoothly
    # Both using the DehnenSmoothWrapper and the new TimeDependentAmplitudeWrapperPotential
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential()
    dbp = potential.DehnenBarPotential(tform=-100000.0, tsteady=1.0)
    dp = potential.DehnenSmoothWrapperPotential(pot=dbp)
    tp = potential.TimeDependentAmplitudeWrapperPotential(pot=dbp, A=dp._smooth)
    # Orbit of the Sun
    o = Orbit().toPlanar()
    ts = numpy.linspace(0.0, -20.0, 1001)
    o.integrate(ts, lp + dp)
    ott = o()
    ott.integrate(ts, lp + tp)
    tol = 1e-10
    assert (
        numpy.amax(numpy.fabs(o.x(ts) - ott.x(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.y(ts) - ott.y(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.vx(ts) - ott.vx(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    assert (
        numpy.amax(numpy.fabs(o.vy(ts) - ott.vy(ts))) < tol
    ), "Integrating an orbit in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    return None


def test_TimeDependentAmplitudeWrapperPotential_against_DehnenSmooth_2d_dxdv():
    # Test that TimeDependentAmplitudeWrapperPotential acts the same as DehnenSmooth
    # Test = LogPot + DehnenBar grown smoothly
    # Both using the DehnenSmoothWrapper and the new TimeDependentAmplitudeWrapperPotential
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential()
    dbp = potential.DehnenBarPotential(tform=-100000.0, tsteady=1.0)
    dp = potential.DehnenSmoothWrapperPotential(pot=dbp)
    tp = potential.TimeDependentAmplitudeWrapperPotential(pot=dbp, A=dp._smooth)
    # Orbit of the Sun
    o = Orbit().toPlanar()
    ts = numpy.linspace(0.0, -20.0, 1001)
    o.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, lp + dp, rectIn=True, rectOut=True)
    ott = o()
    ott.integrate_dxdv([1.0, 0.0, 0.0, 0.0], ts, lp + tp, rectIn=True, rectOut=True)
    tol = 1e-10
    assert (
        numpy.amax(numpy.fabs(o.getOrbit_dxdv() - ott.getOrbit_dxdv())) < tol
    ), "Integrating an orbit with dxdv in a growing DehnenSmoothWrapper does not agree between DehnenSmooth and TimeDependentWrapper"
    return None


def test_TimeDependentAmplitudeWrapperPotential_inputerrors():
    # TypeError when A not supplied
    lp = potential.LogarithmicHaloPotential()
    with pytest.raises(
        TypeError,
        match="A= input to TimeDependentAmplitudeWrapperPotential should be a function",
    ):
        tp = TimeDependentAmplitudeWrapperPotential(pot=lp)
    # TypeError when supplying a function with no argument
    with pytest.raises(
        TypeError,
        match="A= input to TimeDependentAmplitudeWrapperPotential should be a function that can be called with a single parameter",
    ):
        tp = TimeDependentAmplitudeWrapperPotential(pot=lp, A=lambda: 1.0)
    # TypeError when supplying a function with more than 1 argument
    with pytest.raises(
        TypeError,
        match="A= input to TimeDependentAmplitudeWrapperPotential should be a function that can be called with a single parameter",
    ):
        tp = TimeDependentAmplitudeWrapperPotential(pot=lp, A=lambda x, y: x + y)
    # But having additional arguments have defaults should be allowed
    tp = TimeDependentAmplitudeWrapperPotential(pot=lp, A=lambda x, y=1.0: x + y)
    # Return value should be a number
    with pytest.raises(
        TypeError,
        match=r"A= function needs to return a number \(specifically, a numbers.Number\)",
    ):
        tp = TimeDependentAmplitudeWrapperPotential(pot=lp, A=lambda t: (t, t + 1))
    with pytest.raises(
        TypeError,
        match=r"A= function needs to return a number \(specifically, a numbers.Number\)",
    ):
        tp = TimeDependentAmplitudeWrapperPotential(
            pot=lp, A=lambda t: numpy.array([t])
        )
    return None


def test_KuzminLikeWrapperPotential_against_KuzminDisk():
    # Test that the KuzminLikeWrapperPotential applied to a KeplerPotential gives the
    # same potential as the KuzminDiskPotential
    from galpy.potential import (
        KeplerPotential,
        KuzminDiskPotential,
        KuzminLikeWrapperPotential,
    )

    kp = KeplerPotential(amp=1.0)
    kdp = KuzminDiskPotential(amp=1.0, a=1.3)
    kwp = KuzminLikeWrapperPotential(pot=kp, a=1.3)
    # Check that the potentials are the same in all ways
    R, z = 1.1, 0.2
    tol = 1e-10
    assert (
        numpy.fabs(kdp(R, z) - kwp(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same potential as KuzminDiskPotential"
    assert (
        numpy.fabs(kdp.Rforce(R, z) - kwp.Rforce(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same Rforce as KuzminDiskPotential"
    assert (
        numpy.fabs(kdp.zforce(R, z) - kwp.zforce(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same zforce as KuzminDiskPotential"
    assert (
        numpy.fabs(kdp.R2deriv(R, z) - kwp.R2deriv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same R2deriv as KuzminDiskPotential"
    assert (
        numpy.fabs(kdp.z2deriv(R, z) - kwp.z2deriv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same z2deriv as KuzminDiskPotential"
    assert (
        numpy.fabs(kdp.Rzderiv(R, z) - kwp.Rzderiv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same Rzderiv as KuzminDiskPotential"
    return None


def test_KuzminLikeWrapperPotential_against_MiyamotoNagai():
    # Test that the KuzminLikeWrapperPotential applied to a KeplerPotential with non-zero
    # b gives the same potential as the MiyamotoNagaiPotential
    from galpy.potential import (
        KeplerPotential,
        KuzminLikeWrapperPotential,
        MiyamotoNagaiPotential,
    )

    kp = KeplerPotential(amp=1.0)
    mnp = MiyamotoNagaiPotential(amp=1.0, a=1.3, b=0.2)
    kwp = KuzminLikeWrapperPotential(pot=kp, a=1.3, b=0.2)
    # Check that the potentials are the same in all ways
    R, z = 1.1, 0.2
    tol = 1e-10
    assert (
        numpy.fabs(mnp(R, z) - kwp(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same potential as MiyamotoNagaiPotential"
    assert (
        numpy.fabs(mnp.Rforce(R, z) - kwp.Rforce(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same Rforce as MiyamotoNagaiPotential"
    assert (
        numpy.fabs(mnp.zforce(R, z) - kwp.zforce(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same zforce as MiyamotoNagaiPotential"
    assert (
        numpy.fabs(mnp.R2deriv(R, z) - kwp.R2deriv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same R2deriv as MiyamotoNagaiPotential"
    assert (
        numpy.fabs(mnp.z2deriv(R, z) - kwp.z2deriv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same z2deriv as MiyamotoNagaiPotential"
    assert (
        numpy.fabs(mnp.Rzderiv(R, z) - kwp.Rzderiv(R, z)) < tol
    ), "KuzminLikeWrapperPotential does not give the same Rzderiv as MiyamotoNagaiPotential"
    return None


def test_KuzminLikeWrapperPotential_NonAxiError():
    # Test that the KuzminLikeWrapperPotential applied to a non-axisymmetric potential
    # raises a RuntimeError
    from galpy.potential import KuzminLikeWrapperPotential, LogarithmicHaloPotential

    with pytest.raises(RuntimeError) as excinfo:
        kwp = KuzminLikeWrapperPotential(
            pot=LogarithmicHaloPotential(q=0.8, b=0.9), a=1.3
        )
    assert (
        "KuzminLikeWrapperPotential only works for spherical or axisymmetric potentials"
        == excinfo.value.args[0]
    )
    return None


def test_phiforce_deprecation():
    # Test that phiforce is being deprecated correctly for phitorque
    import warnings

    # Check that we've removed phiforce in the correct version
    from packaging.version import parse as parse_version

    deprecation_version = parse_version("1.8.3")
    from galpy import __version__ as galpy_version

    galpy_version = parse_version(galpy_version)
    should_be_removed = galpy_version > deprecation_version
    # Now test
    lp = potential.LogarithmicHaloPotential()
    # Method
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        try:
            lp.phiforce(1.0, 0.1)
        except AttributeError:
            if not should_be_removed:
                raise AssertionError(
                    "phiforce stopped working before it is supposed to have been removed"
                )
        else:
            if should_be_removed:
                raise AssertionError(
                    "phiforce not removed when it was supposed to be removed"
                )
            raisedWarning = False
            for wa in w:
                raisedWarning = (
                    str(wa.message)
                    == "phiforce has been renamed phitorque, because it has always really been a torque (per unit mass); please switch to the new method name, because the old name will be removed in v1.9 and may be re-used for the actual phi force component"
                )
                if raisedWarning:
                    break
            assert (
                raisedWarning
            ), "phiforce deprecation did not raise the expected warning"
    # Function
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        try:
            potential.evaluatephiforces(lp, 1.0, 0.1)
        except AttributeError:
            if not should_be_removed:
                raise AssertionError(
                    "phiforce stopped working before it is supposed to have been removed"
                )
        else:
            if should_be_removed:
                raise AssertionError(
                    "phiforce not removed when it was supposed to be removed"
                )
            raisedWarning = False
            for wa in w:
                raisedWarning = (
                    str(wa.message)
                    == "evaluatephiforces has been renamed evaluatephitorques, because it has always really been a torque (per unit mass); please switch to the new method name, because the old name will be removed in v1.9 and may be re-used for the actual phi force component"
                )
                if raisedWarning:
                    break
            assert (
                raisedWarning
            ), "phiforce deprecation did not raise the expected warning"


def test_phiforce_deprecation_2d():
    # Test that phiforce is being deprecated correctly for phitorque
    import warnings

    # Check that we've removed phiforce in the correct version
    from packaging.version import parse as parse_version

    deprecation_version = parse_version("1.8.3")
    from galpy import __version__ as galpy_version

    galpy_version = parse_version(galpy_version)
    should_be_removed = galpy_version > deprecation_version
    # Now test
    lp = potential.LogarithmicHaloPotential().toPlanar()
    # Method
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        try:
            lp.phiforce(1.0, 0.1)
        except AttributeError:
            if not should_be_removed:
                raise AssertionError(
                    "phiforce stopped working before it is supposed to have been removed"
                )
        else:
            if should_be_removed:
                raise AssertionError(
                    "phiforce not removed when it was supposed to be removed"
                )
            raisedWarning = False
            for wa in w:
                raisedWarning = (
                    str(wa.message)
                    == "phiforce has been renamed phitorque, because it has always really been a torque (per unit mass); please switch to the new method name, because the old name will be removed in v1.9 and may be re-used for the actual phi force component"
                )
                if raisedWarning:
                    break
            assert (
                raisedWarning
            ), "phiforce deprecation did not raise the expected warning"
    # Function
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FutureWarning)
        try:
            potential.evaluateplanarphiforces(lp, 1.0, 0.1)
        except AttributeError:
            if not should_be_removed:
                raise AssertionError(
                    "phiforce stopped working before it is supposed to have been removed"
                )
        else:
            if should_be_removed:
                raise AssertionError(
                    "phiforce not removed when it was supposed to be removed"
                )
            raisedWarning = False
            for wa in w:
                raisedWarning = (
                    str(wa.message)
                    == "evaluateplanarphiforces has been renamed evaluateplanarphitorques, because it has always really been a torque (per unit mass); please switch to the new method name, because the old name will be removed in v1.9 and may be re-used for the actual phi force component"
                )
                if raisedWarning:
                    break
            assert (
                raisedWarning
            ), "phiforce deprecation did not raise the expected warning"


# Test that Pot is required to be a positional argument for Potential functions
def test_potential_Pot_is_positional():
    from galpy import potential
    from galpy.potential import MWPotential2014

    for func in [
        potential.evaluatePotentials,
        potential.evaluateRforces,
        potential.evaluatezforces,
        potential.evaluateR2derivs,
        potential.evaluatez2derivs,
        potential.evaluateRzderivs,
        potential.evaluaterforces,
        potential.evaluatephitorques,
        potential.evaluateDensities,
        potential.evaluateSurfaceDensities,
        potential.flattening,
        potential.rtide,
        potential.ttensor,
    ]:
        with pytest.raises(TypeError) as excinfo:
            func(Pot=MWPotential2014, R=1.0, z=0.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
    for func in [
        potential.omegac,
        potential.epifreq,
        potential.verticalfreq,
        potential.rhalf,
        potential.tdyn,
    ]:
        with pytest.raises(TypeError) as excinfo:
            func(Pot=MWPotential2014, R=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        # Special cases
        with pytest.raises(TypeError) as excinfo:
            potential.lindbladR(Pot=MWPotential2014, OmegaP=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.rl(Pot=MWPotential2014, lz=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.rE(Pot=MWPotential2014, E=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.LcE(Pot=MWPotential2014, E=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.vterm(Pot=MWPotential2014, l=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.zvc_range(Pot=MWPotential2014, E=1.0, Lz=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.zvc(Pot=MWPotential2014, R=1.0, E=1.0, Lz=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
        with pytest.raises(TypeError) as excinfo:
            potential.rhalf(Pot=MWPotential2014)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
    return None


# Test that Pot is required to be a positional argument for Potential functions
def test_potential_Pot_is_positional_planar():
    from galpy import potential
    from galpy.potential import MWPotential2014

    for func in [
        potential.evaluateplanarPotentials,
        potential.evaluateplanarRforces,
        potential.evaluateplanarR2derivs,
        potential.evaluateplanarphitorques,
    ]:
        with pytest.raises(TypeError) as excinfo:
            func(Pot=potential.toPlanarPotential(MWPotential2014), R=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
    return None


# Test that Pot is required to be a positional argument for Potential functions
def test_potential_Pot_is_positional_linear():
    from galpy import potential
    from galpy.potential import MWPotential2014

    for func in [potential.evaluatelinearPotentials, potential.evaluatelinearForces]:
        with pytest.raises(TypeError) as excinfo:
            func(Pot=potential.toVerticalPotential(MWPotential2014, 1.0), x=1.0)
        assert "required positional argument: 'Pot'" in excinfo.value.args[0]
    return None


# Issue #495
def test_diskscf_overflow():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential.mwpotentials import McMillan17
    from galpy.util import conversion

    ro17 = conversion.get_physical(McMillan17)["ro"]
    vo17 = conversion.get_physical(McMillan17)["vo"]
    o17 = Orbit([209.3, 26.8, 46.5, -1.16, -0.88, 189.11], radec=True, ro=ro17, vo=vo17)
    delta = estimateDeltaStaeckel(
        McMillan17, o17.R(use_physical=False), o17.z(use_physical=False)
    )
    assert not numpy.isnan(
        delta
    ), "estimateDeltaStaeckel returns NaN due to overflow in DiskSCFPotential"


def test_InterpSnapshotRZPotential_pickling():
    # Test that InterpSnapshotRZPotential can be pickled (see #507, #509)
    if not _PYNBODY_LOADED:
        pytest.skip()
    import pickle

    import pynbody

    from galpy.potential import InterpSnapshotRZPotential

    # Set up simple snapshot: 1 star!
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    spi = InterpSnapshotRZPotential(s)
    test = pickle.dumps(spi)
    newspi = pickle.loads(test)
    # Inside the grid
    assert (
        numpy.fabs(newspi(1.0, 0.0) - spi(1.0, 0.0)) < 1e-10
    ), "Unpickled InterpSnapshotRZPotential does not return the same potential as original instance"
    # Outside the grid, needs _origPot
    assert (
        numpy.fabs(newspi(1.0, 10.0) - spi(1.0, 10.0)) < 1e-10
    ), "Unpickled InterpSnapshotRZPotential does not return the same potential as original instance"
    return None


# Test that trying to plot a potential with xy=True and effective=True raises a RuntimeError
def test_plotting_xy_effective_error():
    # First a single potential
    kp = potential.KeplerPotential(normalize=1.0)
    with pytest.raises(RuntimeError) as excinfo:
        kp.plot(xy=True, effective=True)
    assert "xy and effective cannot be True at the same time" in excinfo.value.args[0]
    # Then a list of potentials
    with pytest.raises(RuntimeError) as excinfo:
        potential.plotPotentials(potential.MWPotential2014, xy=True, effective=True)
    assert "xy and effective cannot be True at the same time" in excinfo.value.args[0]
    return None


def test_plotting():
    import tempfile

    # Some tests of the plotting routines, to make sure they don't fail
    kp = potential.KeplerPotential(normalize=1.0)
    # Plot the rotation curve
    kp.plotRotcurve()
    kp.toPlanar().plotRotcurve()  # through planar interface
    kp.plotRotcurve(Rrange=[0.01, 10.0], grid=101, savefilename=None)
    potential.plotRotcurve([kp])
    potential.plotRotcurve([kp], Rrange=[0.01, 10.0], grid=101, savefilename=None)
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        kp.plotRotcurve(Rrange=[0.01, 10.0], grid=101, savefilename=tmp_savefilename)
        # Then plot using the saved file
        kp.plotRotcurve(Rrange=[0.01, 10.0], grid=101, savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    # Plot the escape-velocity curve
    kp.plotEscapecurve()
    kp.toPlanar().plotEscapecurve()  # Through planar interface
    kp.plotEscapecurve(Rrange=[0.01, 10.0], grid=101, savefilename=None)
    potential.plotEscapecurve([kp])
    potential.plotEscapecurve([kp], Rrange=[0.01, 10.0], grid=101, savefilename=None)
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        kp.plotEscapecurve(Rrange=[0.01, 10.0], grid=101, savefilename=tmp_savefilename)
        # Then plot using the saved file
        kp.plotEscapecurve(Rrange=[0.01, 10.0], grid=101, savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    # Plot the potential itself
    kp.plot()
    kp.plot(
        t=1.0,
        rmin=0.01,
        rmax=1.8,
        nrs=11,
        zmin=-0.55,
        zmax=0.55,
        nzs=11,
        effective=False,
        Lz=None,
        xy=True,
        xrange=[0.01, 1.8],
        yrange=[-0.55, 0.55],
        justcontours=True,
        ncontours=11,
        savefilename=None,
    )
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        kp.plot(
            t=1.0,
            rmin=0.01,
            rmax=1.8,
            nrs=11,
            zmin=-0.55,
            zmax=0.55,
            nzs=11,
            effective=False,
            Lz=None,
            xrange=[0.01, 1.8],
            yrange=[-0.55, 0.55],
            ncontours=11,
            savefilename=tmp_savefilename,
        )
        # Then plot using the saved file
        kp.plot(
            t=1.0,
            rmin=0.01,
            rmax=1.8,
            nrs=11,
            zmin=-0.55,
            zmax=0.55,
            nzs=11,
            effective=False,
            Lz=None,
            xrange=[0.01, 1.8],
            yrange=[-0.55, 0.55],
            ncontours=11,
            savefilename=tmp_savefilename,
        )
    finally:
        os.remove(tmp_savefilename)
    potential.plotPotentials([kp])
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        potential.plotPotentials(
            [kp],
            rmin=0.01,
            rmax=1.8,
            nrs=11,
            zmin=-0.55,
            zmax=0.55,
            nzs=11,
            justcontours=True,
            xy=True,
            ncontours=11,
            savefilename=tmp_savefilename,
        )
        # Then plot using the saved file
        potential.plotPotentials(
            [kp],
            t=1.0,
            rmin=0.01,
            rmax=1.8,
            nrs=11,
            zmin=-0.55,
            zmax=0.55,
            nzs=11,
            ncontours=11,
            savefilename=tmp_savefilename,
        )
    finally:
        os.remove(tmp_savefilename)
    # Plot the effective potential
    kp.plot()
    kp.plot(effective=True, Lz=1.0)
    try:
        kp.plot(effective=True, Lz=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Potential.plot with effective=True, but Lz=None did not return a RuntimeError"
        )
    potential.plotPotentials([kp], effective=True, Lz=1.0)
    try:
        potential.plotPotentials([kp], effective=True, Lz=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Potential.plot with effective=True, but Lz=None did not return a RuntimeError"
        )
    # Plot the density of a LogarithmicHaloPotential
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    lp.plotDensity()
    lp.plotDensity(
        t=1.0,
        rmin=0.05,
        rmax=1.8,
        nrs=11,
        zmin=-0.55,
        zmax=0.55,
        nzs=11,
        aspect=1.0,
        log=True,
        justcontours=True,
        xy=True,
        ncontours=11,
        savefilename=None,
    )
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        lp.plotDensity(savefilename=tmp_savefilename)
        # Then plot using the saved file
        lp.plotDensity(savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    potential.plotDensities([lp])
    potential.plotDensities(
        [lp],
        t=1.0,
        rmin=0.05,
        rmax=1.8,
        nrs=11,
        zmin=-0.55,
        zmax=0.55,
        nzs=11,
        aspect=1.0,
        log=True,
        xy=True,
        justcontours=True,
        ncontours=11,
        savefilename=None,
    )
    # Plot the surface density of a LogarithmicHaloPotential
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    lp.plotSurfaceDensity()
    lp.plotSurfaceDensity(
        t=1.0,
        z=2.0,
        xmin=0.05,
        xmax=1.8,
        nxs=11,
        ymin=-0.55,
        ymax=0.55,
        nys=11,
        aspect=1.0,
        log=True,
        justcontours=True,
        ncontours=11,
        savefilename=None,
    )
    # Also while saving the result
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        lp.plotSurfaceDensity(savefilename=tmp_savefilename)
        # Then plot using the saved file
        lp.plotSurfaceDensity(savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    potential.plotSurfaceDensities([lp])
    potential.plotSurfaceDensities(
        [lp],
        t=1.0,
        z=2.0,
        xmin=0.05,
        xmax=1.8,
        nxs=11,
        ymin=-0.55,
        ymax=0.55,
        nys=11,
        aspect=1.0,
        log=True,
        justcontours=True,
        ncontours=11,
        savefilename=None,
    )
    # Plot the potential itself for a 2D potential
    kp.toPlanar().plot()
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        kp.toPlanar().plot(Rrange=[0.01, 1.8], grid=11, savefilename=tmp_savefilename)
        # Then plot using the saved file
        kp.toPlanar().plot(Rrange=[0.01, 1.8], grid=11, savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    dp = potential.EllipticalDiskPotential()
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        dp.plot(
            xrange=[0.01, 1.8],
            yrange=[0.01, 1.8],
            gridx=11,
            gridy=11,
            ncontours=11,
            savefilename=tmp_savefilename,
        )
        # Then plot using the saved file
        dp.plot(
            xrange=[0.01, 1.8],
            yrange=[0.01, 1.8],
            gridx=11,
            gridy=11,
            ncontours=11,
            savefilename=tmp_savefilename,
        )
    finally:
        os.remove(tmp_savefilename)
    potential.plotplanarPotentials([dp], gridx=11, gridy=11)
    # Tests of linearPotential plotting
    lip = potential.RZToverticalPotential(
        potential.MiyamotoNagaiPotential(normalize=1.0), 1.0
    )
    lip.plot()
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        lip.plot(t=0.0, min=-15.0, max=15, ns=21, savefilename=tmp_savefilename)
        # Then plot using the saved file
        lip.plot(t=0.0, min=-15.0, max=15, ns=21, savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    savefile, tmp_savefilename = tempfile.mkstemp()
    try:
        os.close(savefile)  # Easier this way
        os.remove(tmp_savefilename)
        # First save
        potential.plotlinearPotentials(
            lip, t=0.0, min=-15.0, max=15, ns=21, savefilename=tmp_savefilename
        )
        # Then plot using the saved file
        potential.plotlinearPotentials(
            lip, t=0.0, min=-15.0, max=15, ns=21, savefilename=tmp_savefilename
        )
    finally:
        os.remove(tmp_savefilename)
    return None


# Classes for testing Integer TwoSphericalPotential and for testing special
# cases of some other potentials
from galpy.potential import (
    BurkertPotential,
    DiskSCFPotential,
    FerrersPotential,
    FlattenedPowerPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    MN3ExponentialDiskPotential,
    MWPotential,
    NullPotential,
    PowerSphericalPotential,
    SoftenedNeedleBarPotential,
    SpiralArmsPotential,
    TriaxialHernquistPotential,
    TriaxialJaffePotential,
    TriaxialNFWPotential,
    TwoPowerSphericalPotential,
    TwoPowerTriaxialPotential,
    interpRZPotential,
)


class mockSphericalSoftenedNeedleBarPotential(SoftenedNeedleBarPotential):
    def __init__(self):
        SoftenedNeedleBarPotential.__init__(
            self, amp=1.0, a=0.000001, b=0.0, c=10.0, omegab=0.0, pa=0.0
        )
        self.normalize(1.0)
        self.isNonAxi = False
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if phi is None:
            phi = 0.0
        x, y, z = self._compute_xyz(R, phi, z, t)
        Tp, Tm = self._compute_TpTm(x, y, z)
        return numpy.log((x - self._a + Tm) / (x + self._a + Tp)) / 2.0 / self._a


class specialTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=1.5, beta=3.0)
        return None


class DehnenTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=1.5, beta=4.0)
        return None


class DehnenCoreTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=0, beta=4.0)
        return None


class HernquistTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=1.0, beta=4.0)
        return None


class JaffeTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=2.0, beta=4.0)
        return None


class NFWTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self, amp=1.0, a=5.0, alpha=1.0, beta=3.0)
        return None


class specialPowerSphericalPotential(PowerSphericalPotential):
    def __init__(self):
        PowerSphericalPotential.__init__(self, amp=1.0, alpha=2.0)
        return None


class specialMiyamotoNagaiPotential(MiyamotoNagaiPotential):
    def __init__(self):
        MiyamotoNagaiPotential.__init__(self, amp=1.0, a=0.0, b=0.1)
        return None


class specialFlattenedPowerPotential(FlattenedPowerPotential):
    def __init__(self):
        FlattenedPowerPotential.__init__(self, alpha=0.0)
        return None


class specialMN3ExponentialDiskPotentialPD(MN3ExponentialDiskPotential):
    def __init__(self):
        MN3ExponentialDiskPotential.__init__(self, normalize=1.0, posdens=True)
        return None


class specialMN3ExponentialDiskPotentialSECH(MN3ExponentialDiskPotential):
    def __init__(self):
        MN3ExponentialDiskPotential.__init__(self, normalize=1.0, sech=True)
        return None


class BurkertPotentialNoC(BurkertPotential):
    def __init__(self):
        # Just to force not using C
        BurkertPotential.__init__(self)
        self.hasC = False
        self.hasC_dxdv = False
        return None


class oblateHernquistPotential(TriaxialHernquistPotential):
    def __init__(self):
        TriaxialHernquistPotential.__init__(self, normalize=1.0, b=1.0, c=0.2)
        return None


class oblateNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(self, normalize=1.0, b=1.0, c=0.2)
        return None


class oblatenoGLNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(self, normalize=1.0, b=1.0, c=0.2, glorder=None)
        return None


class oblateJaffePotential(TriaxialJaffePotential):
    def __init__(self):
        TriaxialJaffePotential.__init__(self, normalize=1.0, b=1.0, c=0.2)
        return None


class prolateHernquistPotential(TriaxialHernquistPotential):
    def __init__(self):
        TriaxialHernquistPotential.__init__(self, normalize=1.0, b=1.0, c=1.8)
        return None


class prolateNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(self, normalize=1.0, b=1.0, c=1.8)
        return None


class prolateJaffePotential(TriaxialJaffePotential):
    def __init__(self):
        TriaxialJaffePotential.__init__(self, normalize=1.0, b=1.0, c=1.8)
        return None


class rotatingSpiralArmsPotential(SpiralArmsPotential):
    def __init__(self):
        SpiralArmsPotential.__init__(self, omega=1.1)


class specialSpiralArmsPotential(SpiralArmsPotential):
    def __init__(self):
        SpiralArmsPotential.__init__(
            self,
            omega=1.3,
            N=4.0,
            Cs=[8.0 / 3.0 / numpy.pi, 1.0 / 2.0, 8.0 / 15.0 / numpy.pi],
        )


class triaxialHernquistPotential(TriaxialHernquistPotential):
    def __init__(self):
        TriaxialHernquistPotential.__init__(self, normalize=1.0, b=1.4, c=0.6)
        return None


class triaxialNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(self, normalize=1.0, b=0.2, c=1.8)
        return None


class triaxialJaffePotential(TriaxialJaffePotential):
    def __init__(self):
        TriaxialJaffePotential.__init__(self, normalize=1.0, b=0.4, c=0.7)
        return None


class zRotatedTriaxialNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(
            self,
            normalize=1.0,
            b=1.5,
            c=0.2,
            zvec=[numpy.sin(0.5), 0.0, numpy.cos(0.5)],
        )
        return None


class yRotatedTriaxialNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(self, normalize=1.0, b=1.5, c=0.2, pa=0.2)
        return None


class fullyRotatedTriaxialNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(
            self,
            normalize=1.0,
            b=1.5,
            c=0.2,
            zvec=[numpy.sin(0.5), 0.0, numpy.cos(0.5)],
            pa=0.2,
        )
        return None


class fullyRotatednoGLTriaxialNFWPotential(TriaxialNFWPotential):
    def __init__(self):
        TriaxialNFWPotential.__init__(
            self,
            normalize=1.0,
            b=1.5,
            c=0.2,
            zvec=[numpy.sin(0.5), 0.0, numpy.cos(0.5)],
            pa=0.2,
            glorder=None,
        )
        return None


class triaxialLogarithmicHaloPotential(LogarithmicHaloPotential):
    def __init__(self):
        LogarithmicHaloPotential.__init__(self, normalize=1.0, b=0.7, q=0.9, core=0.5)
        return None

    def OmegaP(self):
        return 0.0


# Implementations through TwoPowerTriaxialPotential
class HernquistTwoPowerTriaxialPotential(TwoPowerTriaxialPotential):
    def __init__(self):
        TwoPowerTriaxialPotential.__init__(
            self, amp=1.0, a=5.0, alpha=1.0, beta=4.0, b=0.3, c=1.8
        )
        return None


class NFWTwoPowerTriaxialPotential(TwoPowerTriaxialPotential):
    def __init__(self):
        TwoPowerTriaxialPotential.__init__(
            self, amp=1.0, a=2.0, alpha=1.0, beta=3.0, b=1.3, c=0.8
        )
        self.isNonAxi = True  # to test planar-from-full
        return None


class JaffeTwoPowerTriaxialPotential(TwoPowerTriaxialPotential):
    def __init__(self):
        TwoPowerTriaxialPotential.__init__(
            self, amp=1.0, a=5.0, alpha=2.0, beta=4.0, b=1.3, c=1.8
        )
        return None


class testNullPotential(NullPotential):
    def normalize(self, norm):
        pass


# Other DiskSCFPotentials
class sech2DiskSCFPotential(DiskSCFPotential):
    def __init__(self):
        DiskSCFPotential.__init__(
            self,
            dens=lambda R, z: numpy.exp(-3.0 * R)
            * 1.0
            / numpy.cosh(z / 2.0 * 27.0) ** 2.0
            / 4.0
            * 27.0,
            Sigma={"h": 1.0 / 3.0, "type": "exp", "amp": 1.0},
            hz={"type": "sech2", "h": 1.0 / 27.0},
            a=1.0,
            N=5,
            L=5,
        )
        return None


class expwholeDiskSCFPotential(DiskSCFPotential):
    def __init__(self):
        # Add a Hernquist potential because otherwise the density near the
        # center is zero
        from galpy.potential import HernquistPotential

        hp = HernquistPotential(normalize=0.5)
        DiskSCFPotential.__init__(
            self,
            dens=lambda R, z: 13.5
            * numpy.exp(-0.5 / (R + 10.0**-10.0) - 3.0 * R - numpy.fabs(z) * 27.0)
            + hp.dens(R, z),
            Sigma={"h": 1.0 / 3.0, "type": "expwhole", "amp": 1.0, "Rhole": 0.5},
            hz={"type": "exp", "h": 1.0 / 27.0},
            a=1.0,
            N=5,
            L=5,
        )
        return None


# Same as above, but specify type as 'exp' and give Rhole, to make sure that
# case is handled correctly
class altExpwholeDiskSCFPotential(DiskSCFPotential):
    def __init__(self):
        # Add a Hernquist potential because otherwise the density near the
        # center is zero
        from galpy.potential import HernquistPotential

        hp = HernquistPotential(normalize=0.5)
        DiskSCFPotential.__init__(
            self,
            dens=lambda R, z: 13.5
            * numpy.exp(-0.5 / (R + 10.0**-10.0) - 3.0 * R - numpy.fabs(z) * 27.0)
            + hp.dens(R, z),
            Sigma={"h": 1.0 / 3.0, "type": "exp", "amp": 1.0, "Rhole": 0.5},
            hz={"type": "exp", "h": 1.0 / 27.0},
            a=1.0,
            N=5,
            L=5,
        )
        return None


class nonaxiDiskSCFPotential(DiskSCFPotential):
    def __init__(self):
        thp = triaxialHernquistPotential()
        DiskSCFPotential.__init__(
            self,
            dens=lambda R, z, phi: 13.5
            * numpy.exp(-3.0 * R)
            * numpy.exp(-27.0 * numpy.fabs(z))
            + thp.dens(R, z, phi=phi),
            Sigma_amp=[0.5, 0.5],
            Sigma=[lambda R: numpy.exp(-3.0 * R), lambda R: numpy.exp(-3.0 * R)],
            dSigmadR=[
                lambda R: -3.0 * numpy.exp(-3.0 * R),
                lambda R: -3.0 * numpy.exp(-3.0 * R),
            ],
            d2SigmadR2=[
                lambda R: 9.0 * numpy.exp(-3.0 * R),
                lambda R: 9.0 * numpy.exp(-3.0 * R),
            ],
            hz=lambda z: 13.5 * numpy.exp(-27.0 * numpy.fabs(z)),
            Hz=lambda z: (numpy.exp(-27.0 * numpy.fabs(z)) - 1.0 + 27.0 * numpy.fabs(z))
            / 54.0,
            dHzdz=lambda z: 0.5
            * numpy.sign(z)
            * (1.0 - numpy.exp(-27.0 * numpy.fabs(z))),
            N=5,
            L=5,
        )
        return None


# An axisymmetric FerrersPotential
class mockAxisymmetricFerrersPotential(FerrersPotential):
    def __init__(self):
        FerrersPotential.__init__(self, normalize=1.0, b=1.0, c=0.2)
        return None


class mockInterpRZPotential(interpRZPotential):
    def __init__(self):
        interpRZPotential.__init__(
            self,
            RZPot=MWPotential,
            rgrid=(0.01, 2.1, 101),
            zgrid=(0.0, 0.26, 101),
            logR=True,
            interpPot=True,
            interpRforce=True,
            interpzforce=True,
            interpDens=True,
        )


class mockSnapshotRZPotential(potential.SnapshotRZPotential):
    def __init__(self):
        # Test w/ equivalent of KeplerPotential: one mass
        kp = potential.KeplerPotential(amp=1.0)
        s = pynbody.new(star=1)
        s["mass"] = 1.0 / numpy.fabs(kp.Rforce(1.0, 0.0))  # forces vc(1,0)=1
        s["eps"] = 0.0
        potential.SnapshotRZPotential.__init__(self, s)


class mockInterpSnapshotRZPotential(potential.InterpSnapshotRZPotential):
    def __init__(self):
        # Test w/ equivalent of KeplerPotential: one mass
        kp = potential.KeplerPotential(amp=1.0)
        s = pynbody.new(star=1)
        s["mass"] = 1.0 / numpy.fabs(kp.Rforce(1.0, 0.0))  # forces vc(1,0)=1
        s["eps"] = 0.0
        potential.InterpSnapshotRZPotential.__init__(
            self,
            s,
            rgrid=(0.01, 2.0, 101),
            zgrid=(0.0, 0.3, 101),
            logR=False,
            interpPot=True,
            zsym=True,
        )


# Some special cases of 2D, non-axisymmetric potentials, to make sure they
# are covered; need 3 to capture all of the transient behavior
from galpy.potential import (
    CosmphiDiskPotential,
    DehnenBarPotential,
    EllipticalDiskPotential,
    HenonHeilesPotential,
    SteadyLogSpiralPotential,
    TransientLogSpiralPotential,
)


class mockDehnenBarPotentialT1(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(
            self,
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            tform=0.5,
            tsteady=0.5,
            alpha=0.01,
            Af=0.04,
        )


class mockDehnenBarPotentialTm1(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(
            self,
            omegab=1.9,
            rb=0.6,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            tform=-1.0,
            tsteady=1.01,
            alpha=0.01,
            Af=0.04,
        )


# Also one with omegab=0. to test that that works
class mockDehnenBarPotentialTm1Omega0(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(
            self,
            omegab=0.0,
            rb=0.6,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            tform=-1.0,
            tsteady=1.01,
            alpha=0.01,
            Af=0.04,
        )


class mockDehnenBarPotentialTm5(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(
            self,
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            tform=-5.0,
            tsteady=2.0,
            alpha=0.01,
            Af=0.04,
        )


class mockCosmphiDiskPotentialnegcp(CosmphiDiskPotential):
    def __init__(self):
        CosmphiDiskPotential.__init__(
            self,
            amp=1.0,
            phib=25.0 * numpy.pi / 180.0,
            p=1.0,
            phio=0.01,
            m=1.0,
            rb=0.9,
            cp=-0.05,
            sp=0.05,
        )


class mockCosmphiDiskPotentialnegp(CosmphiDiskPotential):
    def __init__(self):
        CosmphiDiskPotential.__init__(
            self,
            amp=1.0,
            phib=25.0 * numpy.pi / 180.0,
            p=-1.0,
            phio=0.01,
            m=1.0,
            cp=-0.05,
            sp=0.05,
        )


class mockEllipticalDiskPotentialT1(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(
            self,
            amp=1.0,
            phib=25.0 * numpy.pi / 180.0,
            p=1.0,
            twophio=0.02,
            tform=0.5,
            tsteady=1.0,
            cp=0.05,
            sp=0.05,
        )


class mockEllipticalDiskPotentialTm1(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(
            self,
            amp=1.0,
            phib=25.0 * numpy.pi / 180.0,
            p=1.0,
            twophio=0.02,
            tform=-1.0,
            tsteady=None,
            cp=-0.05,
            sp=0.05,
        )


class mockEllipticalDiskPotentialTm5(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(
            self,
            amp=1.0,
            phib=25.0 * numpy.pi / 180.0,
            p=1.0,
            twophio=0.02,
            tform=-5.0,
            tsteady=-1.0,
            cp=-0.05,
            sp=0.05,
        )


class mockSteadyLogSpiralPotentialT1(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(
            self,
            amp=1.0,
            omegas=0.65,
            A=-0.035,
            m=2,
            gamma=numpy.pi / 4.0,
            p=-0.3,
            tform=0.5,
            tsteady=1.0,
        )


class mockSteadyLogSpiralPotentialTm1(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(
            self,
            amp=1.0,
            omegas=0.65,
            A=-0.035,
            m=2,
            gamma=numpy.pi / 4.0,
            p=-0.3,
            tform=-1.0,
            tsteady=None,
        )


# Also one with omegab=0. to test that that works
class mockSteadyLogSpiralPotentialTm1Omega0(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(
            self,
            amp=1.0,
            omegas=0.0,
            A=-0.035,
            m=2,
            gamma=numpy.pi / 4.0,
            p=-0.3,
            tform=-1.0,
            tsteady=None,
        )


class mockSteadyLogSpiralPotentialTm5(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(
            self,
            amp=1.0,
            omegas=0.65,
            A=-0.035,
            m=2,
            gamma=numpy.pi / 4.0,
            p=-0.3,
            tform=-1.0,
            tsteady=-5.0,
        )


class mockTransientLogSpiralPotential(TransientLogSpiralPotential):
    def __init__(self):
        TransientLogSpiralPotential.__init__(
            self, amp=1.0, omegas=0.65, A=-0.035, m=2, gamma=numpy.pi / 4.0, p=-0.3
        )


##Potentials used for mock SCF
def rho_Zeeuw(R, z=0.0, phi=0.0, a=1.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    return 3.0 / (4 * numpy.pi) * numpy.power((a + r), -4.0) * a


def axi_density1(R, z=0, phi=0.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    h = potential.HernquistPotential()
    return h.dens(R, z, phi) * (1 + numpy.cos(theta) + numpy.cos(theta) ** 2.0)


def axi_density2(R, z=0, phi=0.0):
    r, theta, phi = coords.cyl_to_spher(R, z, phi)
    return rho_Zeeuw(R, z, phi) * (1 + numpy.cos(theta) + numpy.cos(theta) ** 2)


def scf_density(R, z=0, phi=0.0):
    eps = 0.1
    return axi_density2(R, z, phi) * (1 + eps * (numpy.cos(phi) + numpy.sin(phi)))


##Mock SCF class
class mockSCFZeeuwPotential(potential.SCFPotential):
    def __init__(self):
        Acos, Asin = potential.scf_compute_coeffs_spherical(rho_Zeeuw, 2)
        potential.SCFPotential.__init__(self, amp=1.0, Acos=Acos, Asin=Asin)


class mockSCFNFWPotential(potential.SCFPotential):
    def __init__(self):
        nfw = potential.NFWPotential()
        Acos, Asin = potential.scf_compute_coeffs_spherical(nfw.dens, 10)
        potential.SCFPotential.__init__(self, amp=1.0, Acos=Acos, Asin=Asin)


class mockSCFAxiDensity1Potential(potential.SCFPotential):
    def __init__(self):
        Acos, Asin = potential.scf_compute_coeffs_axi(axi_density1, 10, 2)
        potential.SCFPotential.__init__(self, amp=1.0, Acos=Acos, Asin=Asin)


class mockSCFAxiDensity2Potential(potential.SCFPotential):
    def __init__(self):
        Acos, Asin = potential.scf_compute_coeffs_axi(axi_density2, 10, 2)
        potential.SCFPotential.__init__(self, amp=1.0, Acos=Acos, Asin=Asin)


class mockSCFDensityPotential(potential.SCFPotential):
    def __init__(self):
        Acos, Asin = potential.scf_compute_coeffs(scf_density, 10, 10, phi_order=30)
        potential.SCFPotential.__init__(self, amp=1.0, Acos=Acos, Asin=Asin)


# Test interpSphericalPotential
class mockInterpSphericalPotential(potential.interpSphericalPotential):
    def __init__(self):
        hp = potential.HomogeneousSpherePotential(normalize=1.0, R=1.1)
        potential.interpSphericalPotential.__init__(
            self, rforce=hp, rgrid=numpy.linspace(0.0, 1.1, 201)
        )


class mockInterpSphericalPotentialwForce(potential.interpSphericalPotential):
    def __init__(self):
        hp = potential.HomogeneousSpherePotential(normalize=1.0, R=1.1)
        potential.interpSphericalPotential.__init__(
            self,
            rforce=lambda r: hp.Rforce(r, 0.0),
            Phi0=hp(0.0, 0.0),
            rgrid=numpy.linspace(0.0, 1.1, 201),
        )


# Class to test potentials given as lists, st we can use their methods as class.
from galpy.potential import (
    Potential,
    _isNonAxi,
    evaluateDensities,
    evaluatephitorques,
    evaluatephizderivs,
    evaluateplanarphitorques,
    evaluateplanarPotentials,
    evaluateplanarR2derivs,
    evaluateplanarRforces,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluateRzderivs,
    evaluateSurfaceDensities,
    evaluatez2derivs,
    evaluatezforces,
    planarPotential,
)


class testMWPotential(Potential):
    """Initialize with potential in natural units"""

    def __init__(self, potlist=MWPotential):
        self._potlist = potlist
        Potential.__init__(self, amp=1.0)
        self.isNonAxi = _isNonAxi(self._potlist)
        return None

    def _evaluate(self, R, z, phi=0, t=0, dR=0, dphi=0):
        return evaluatePotentials(self._potlist, R, z, phi=phi, t=t, dR=dR, dphi=dphi)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return evaluateRforces(self._potlist, R, z, phi=phi, t=t)

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        return evaluatephitorques(self._potlist, R, z, phi=phi, t=t)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return evaluatezforces(self._potlist, R, z, phi=phi, t=t)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        return evaluateR2derivs(self._potlist, R, z, phi=phi, t=t)

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        return evaluatez2derivs(self._potlist, R, z, phi=phi, t=t)

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        return evaluateRzderivs(self._potlist, R, z, phi=phi, t=t)

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        return evaluatePotentials(self._potlist, R, z, phi=phi, t=t, dphi=2)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        return evaluatePotentials(self._potlist, R, z, phi=phi, t=t, dR=1, dphi=1)

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        return evaluatephizderivs(self._potlist, R, z, phi=phi, t=t)

    def _dens(self, R, z, phi=0.0, t=0.0, forcepoisson=False):
        return evaluateDensities(
            self._potlist, R, z, phi=phi, t=t, forcepoisson=forcepoisson
        )

    def _surfdens(self, R, z, phi=0.0, t=0.0, forcepoisson=False):
        return evaluateSurfaceDensities(
            self._potlist, R, z, phi=phi, t=t, forcepoisson=forcepoisson
        )

    def vcirc(self, R):
        return potential.vcirc(self._potlist, R)

    def normalize(self, norm, t=0.0):
        self._amp = norm

    def OmegaP(self):
        return 1.0


# Class to test lists of planarPotentials
class testplanarMWPotential(planarPotential):
    """Initialize with potential in natural units"""

    def __init__(self, potlist=MWPotential):
        self._potlist = [p.toPlanar() for p in potlist if isinstance(p, Potential)]
        self._potlist.extend([p for p in potlist if isinstance(p, planarPotential)])
        planarPotential.__init__(self, amp=1.0)
        self.isNonAxi = _isNonAxi(self._potlist)
        return None

    def _evaluate(self, R, phi=0, t=0, dR=0, dphi=0):
        return evaluateplanarPotentials(self._potlist, R, phi=phi, t=t)

    def _Rforce(self, R, phi=0.0, t=0.0):
        return evaluateplanarRforces(self._potlist, R, phi=phi, t=t)

    def _phitorque(self, R, phi=0.0, t=0.0):
        return evaluateplanarphitorques(self._potlist, R, phi=phi, t=t)

    def _R2deriv(self, R, phi=0.0, t=0.0):
        return evaluateplanarR2derivs(self._potlist, R, phi=phi, t=t)

    def _phi2deriv(self, R, phi=0.0, t=0.0):
        return evaluateplanarPotentials(self._potlist, R, phi=phi, t=t, dphi=2)

    def _Rphideriv(self, R, phi=0.0, t=0.0):
        return evaluateplanarPotentials(self._potlist, R, phi=phi, t=t, dR=1, dphi=1)

    def vcirc(self, R):
        return potential.vcirc(self._potlist, R)

    def normalize(self, norm, t=0.0):
        self._amp = norm

    def OmegaP(self):
        return 1.0


class mockFlatEllipticalDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.EllipticalDiskPotential(
                    phib=numpy.pi / 2.0,
                    p=0.0,
                    tform=None,
                    tsteady=None,
                    twophio=14.0 / 220.0,
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockSlowFlatEllipticalDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.EllipticalDiskPotential(
                    phib=numpy.pi / 2.0,
                    p=0.0,
                    twophio=14.0 / 220.0,
                    tform=1.0,
                    tsteady=250.0,
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockFlatLopsidedDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.LopsidedDiskPotential(
                    phib=numpy.pi / 2.0, p=0.0, phio=10.0 / 220.0
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockFlatCosmphiDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.CosmphiDiskPotential(
                    phib=numpy.pi / 2.0, p=0.0, phio=10.0 / 220.0
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockFlatCosmphiDiskwBreakPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.CosmphiDiskPotential(
                    phib=numpy.pi / 2.0, p=0.0, phio=10.0 / 220.0, rb=0.99, m=6
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockFlatDehnenBarPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.DehnenBarPotential(),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockSlowFlatDehnenBarPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.DehnenBarPotential(tform=1.0, tsteady=250.0, rolr=2.5),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockFlatSteadyLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.SteadyLogSpiralPotential(),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockSlowFlatSteadyLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.SteadyLogSpiralPotential(tform=0.1, tsteady=25.0),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockFlatTransientLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.TransientLogSpiralPotential(to=-10.0),
            ],
        )  # this way, it's basically a steady spiral

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockFlatSpiralArmsPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.SpiralArmsPotential(),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockRotatingFlatSpiralArmsPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.SpiralArmsPotential(omega=1.3),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockSpecialRotatingFlatSpiralArmsPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                potential.SpiralArmsPotential(
                    omega=1.3, N=4, Cs=[8 / 3 / numpy.pi, 1 / 2, 8 / 15 / numpy.pi]
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


# Class to test lists of linearPotentials
from galpy.potential import (
    RZToverticalPotential,
    evaluatelinearForces,
    evaluatelinearPotentials,
    linearPotential,
)


class testlinearMWPotential(linearPotential):
    """Initialize with potential in natural units"""

    def __init__(self, potlist=MWPotential):
        self._potlist = RZToverticalPotential(potlist, 1.0)
        linearPotential.__init__(self, amp=1.0)
        return None

    def _evaluate(self, R, phi=0, t=0, dR=0, dphi=0):
        return evaluatelinearPotentials(self._potlist, R, t=t)

    def _force(self, R, t=0.0):
        return evaluatelinearForces(self._potlist, R, t=t)

    def normalize(self, norm, t=0.0):
        self._amp = norm


class mockCombLinearPotential(testlinearMWPotential):
    def __init__(self):
        testlinearMWPotential.__init__(
            self,
            potlist=[
                potential.MWPotential[0],
                potential.MWPotential[1].toVertical(1.0),
                potential.MWPotential[2].toVertical(1.0),
            ],
        )


class mockSimpleLinearPotential(testlinearMWPotential):
    def __init__(self):
        testlinearMWPotential.__init__(
            self,
            potlist=potential.MiyamotoNagaiPotential(normalize=1.0).toVertical(1.0),
        )


from galpy.potential import PlummerPotential


class mockMovingObjectPotential(testMWPotential):
    def __init__(self, rc=0.75, maxt=1.0, nt=50):
        from galpy.orbit import Orbit

        self._rc = rc
        o1 = Orbit([self._rc, 0.0, 1.0, 0.0, 0.0, 0.0])
        o2 = Orbit([self._rc, 0.0, 1.0, 0.0, 0.0, numpy.pi])
        lp = potential.LogarithmicHaloPotential(normalize=1.0)
        times = numpy.linspace(0.0, maxt, nt)
        o1.integrate(times, lp, method="dopr54_c")
        o2.integrate(times, lp, method="dopr54_c")
        self._o1p = potential.MovingObjectPotential(o1)
        self._o2p = potential.MovingObjectPotential(o2)
        testMWPotential.__init__(self, [self._o1p, self._o2p])
        self.isNonAxi = True
        return None

    def phi2deriv(self, R, z, phi=0.0, t=0.0):
        raise AttributeError

    def OmegaP(self):
        return 1.0 / self._rc


class mockMovingObjectPotentialExplPlummer(testMWPotential):
    def __init__(self, rc=0.75, maxt=1.0, nt=50):
        from galpy.orbit import Orbit

        self._rc = rc
        o1 = Orbit([self._rc, 0.0, 1.0, 0.0, 0.0, 0.0])
        o2 = Orbit([self._rc, 0.0, 1.0, 0.0, 0.0, numpy.pi])
        lp = potential.LogarithmicHaloPotential(normalize=1.0)
        times = numpy.linspace(0.0, maxt, nt)
        o1.integrate(times, lp, method="dopr54_c")
        o2.integrate(times, lp, method="dopr54_c")
        oplum = potential.PlummerPotential(amp=0.06, b=0.01)
        self._o1p = potential.MovingObjectPotential(o1, pot=oplum)
        self._o2p = potential.MovingObjectPotential(o2, pot=oplum)
        testMWPotential.__init__(self, [self._o1p, self._o2p])
        self.isNonAxi = True
        return None

    def phi2deriv(self, R, z, phi=0.0, t=0.0):
        raise AttributeError

    def OmegaP(self):
        return 1.0 / self._rc


class mockMovingObjectLongIntPotential(mockMovingObjectPotential):
    def __init__(self, rc=0.75):
        mockMovingObjectPotential.__init__(self, rc=rc, maxt=15.0, nt=3001)
        return None


# Classes to test wrappers
from galpy.potential import (
    AdiabaticContractionWrapperPotential,
    CorotatingRotationWrapperPotential,
    DehnenSmoothWrapperPotential,
    GaussianAmplitudeWrapperPotential,
    KuzminLikeWrapperPotential,
    RotateAndTiltWrapperPotential,
    SolidBodyRotationWrapperPotential,
    TimeDependentAmplitudeWrapperPotential,
)
from galpy.potential.WrapperPotential import parentWrapperPotential


class DehnenSmoothDehnenBarPotential(DehnenSmoothWrapperPotential):
    # This wrapped potential should be the same as the default DehnenBar
    # for t > -99
    #
    # Need to use __new__ because new Wrappers are created using __new__
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(tform=-100.0, tsteady=1.0)  # on after t=-99
        return DehnenSmoothWrapperPotential.__new__(
            cls, amp=1.0, pot=dpn, tform=-4.0 * 2.0 * numpy.pi / dpn.OmegaP()
        )


# Additional DehnenSmooth instances to catch all smoothing cases
class mockDehnenSmoothBarPotentialT1(DehnenSmoothWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        return DehnenSmoothWrapperPotential.__new__(
            cls,
            amp=1.0,
            pot=dpn,  #                               tform=-4.*2.*numpy.pi/dpn.OmegaP())
            tform=0.5,
            tsteady=0.5,
        )


class mockDehnenSmoothBarPotentialTm1(DehnenSmoothWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        return DehnenSmoothWrapperPotential.__new__(
            cls, amp=1.0, pot=dpn, tform=-1.0, tsteady=1.01
        )


class mockDehnenSmoothBarPotentialTm5(DehnenSmoothWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        return DehnenSmoothWrapperPotential.__new__(
            cls, amp=1.0, pot=dpn, tform=-5.0, tsteady=2.0
        )


class mockDehnenSmoothBarPotentialDecay(DehnenSmoothWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        return DehnenSmoothWrapperPotential.__new__(
            cls,
            amp=1.0,
            pot=dpn,  #                               tform=-4.*2.*numpy.pi/dpn.OmegaP())
            tform=-0.5,
            tsteady=1.0,
            decay=True,
        )


class mockFlatDehnenSmoothBarPotential(testMWPotential):
    def __init__(self):
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                DehnenSmoothWrapperPotential(
                    amp=1.0,
                    pot=dpn,
                    tform=-4.0 * 2.0 * numpy.pi / dpn.OmegaP(),
                    tsteady=2.0 * 2 * numpy.pi / dpn.OmegaP(),
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1]._pot.OmegaP()


class mockSlowFlatDehnenSmoothBarPotential(testMWPotential):
    def __init__(self):
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                DehnenSmoothWrapperPotential(
                    amp=1.0, pot=dpn, tform=0.1, tsteady=500.0
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1]._pot.OmegaP()


class mockSlowFlatDecayingDehnenSmoothBarPotential(testMWPotential):
    def __init__(self):
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                DehnenSmoothWrapperPotential(
                    amp=1.0, pot=dpn, tform=-250.0, tsteady=500.0, decay=True
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1]._pot.OmegaP()


# A DehnenSmoothWrappered version of LogarithmicHaloPotential for simple aAtest
class mockSmoothedLogarithmicHaloPotential(DehnenSmoothWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        return DehnenSmoothWrapperPotential.__new__(
            cls,
            amp=1.0,
            pot=potential.LogarithmicHaloPotential(normalize=1.0),
            tform=-1.0,
            tsteady=0.5,
        )


# SolidBodyWrapperPotential
class SolidBodyRotationSpiralArmsPotential(SolidBodyRotationWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        spn = potential.SpiralArmsPotential(omega=0.0, phi_ref=0.0)
        return SolidBodyRotationWrapperPotential.__new__(
            cls, amp=1.0, pot=spn.toPlanar(), omega=1.1, pa=0.4
        )


class mockFlatSolidBodyRotationSpiralArmsPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                SolidBodyRotationWrapperPotential(
                    amp=1.0, pot=potential.SpiralArmsPotential(), omega=1.3
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


# Special case to test handling of pure planarWrapper, not necessary for new wrappers
class mockFlatSolidBodyRotationPlanarSpiralArmsPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0).toPlanar(),
                SolidBodyRotationWrapperPotential(
                    amp=1.0, pot=potential.SpiralArmsPotential().toPlanar(), omega=1.3
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class testorbitHenonHeilesPotential(testplanarMWPotential):
    # Need this class, bc orbit tests skip potentials that do not have
    # .normalize, and HenonHeiles as a non-axi planarPotential instance
    # does not
    def __init__(self):
        testplanarMWPotential.__init__(self, potlist=[HenonHeilesPotential(amp=1.0)])

    def OmegaP(self):
        # Non-axi, so need to set this to zero for Jacobi
        return 0.0


# CorotatingWrapperPotential
class CorotatingRotationSpiralArmsPotential(CorotatingRotationWrapperPotential):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        spn = potential.SpiralArmsPotential(omega=0.0, phi_ref=0.0)
        return CorotatingRotationWrapperPotential.__new__(
            cls, amp=1.0, pot=spn.toPlanar(), vpo=1.1, beta=-0.2, pa=0.4, to=3.0
        )


class mockFlatCorotatingRotationSpiralArmsPotential(testMWPotential):
    # With beta=1 this has a fixed pattern speed --> Jacobi conserved
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                CorotatingRotationWrapperPotential(
                    amp=1.0,
                    pot=potential.SpiralArmsPotential(),
                    vpo=1.3,
                    beta=1.0,
                    pa=0.3,
                    to=3.0,
                ),
            ],
        )

    def OmegaP(self):
        return 1.3


# beta =/= 1 --> Liouville should still hold!
class mockFlatTrulyCorotatingRotationSpiralArmsPotential(testMWPotential):
    # With beta=1 this has a fixed pattern speed --> Jacobi conserved
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                CorotatingRotationWrapperPotential(
                    amp=1.0,
                    pot=potential.SpiralArmsPotential(),
                    vpo=1.3,
                    beta=0.1,
                    pa=-0.3,
                    to=-3.0,
                ),
            ],
        )

    def OmegaP(self):
        return 1.3


# GaussianAmplitudeWrapperPotential
class GaussianAmplitudeDehnenBarPotential(GaussianAmplitudeWrapperPotential):
    # Need to use __new__ because new Wrappers are created using __new__
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(tform=-100.0, tsteady=1.0)  # on after t=-99
        return GaussianAmplitudeWrapperPotential.__new__(
            cls, amp=1.0, pot=dpn, to=0.0, sigma=1.0
        )


# Basically constant
class mockFlatGaussianAmplitudeBarPotential(testMWPotential):
    def __init__(self):
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                GaussianAmplitudeWrapperPotential(
                    amp=1.0, pot=dpn, to=10, sigma=1000000.0
                ),
            ],
        )

    def OmegaP(self):
        return self._potlist[1]._pot.OmegaP()


# For Liouville
class mockFlatTrulyGaussianAmplitudeBarPotential(testMWPotential):
    def __init__(self):
        dpn = DehnenBarPotential(
            omegab=1.9,
            rb=0.4,
            barphi=25.0 * numpy.pi / 180.0,
            beta=0.0,
            alpha=0.01,
            Af=0.04,
            tform=-99.0,
            tsteady=1.0,
        )
        testMWPotential.__init__(
            self,
            potlist=[
                potential.LogarithmicHaloPotential(normalize=1.0),
                GaussianAmplitudeWrapperPotential(amp=1.0, pot=dpn, to=10, sigma=1.0),
            ],
        )

    def OmegaP(self):
        return self._potlist[1]._pot.OmegaP()


# A GaussianAmplitudeWrappered version of LogarithmicHaloPotential for simple aAtest
class mockGaussianAmplitudeSmoothedLogarithmicHaloPotential(
    GaussianAmplitudeWrapperPotential
):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        return GaussianAmplitudeWrapperPotential.__new__(
            cls,
            amp=1.0,
            pot=potential.LogarithmicHaloPotential(normalize=1.0),
            to=0.0,
            sigma=100000000000000.0,
        )


class nestedListPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self, potlist=[potential.MWPotential2014, potential.SpiralArmsPotential()]
        )

    def OmegaP(self):
        return self._potlist[1].OmegaP()


class mockAdiabaticContractionMWP14WrapperPotential(
    AdiabaticContractionWrapperPotential
):
    def __init__(self):
        AdiabaticContractionWrapperPotential.__init__(
            self,
            pot=potential.MWPotential2014[2],
            baryonpot=potential.MWPotential2014[:2],
            f_bar=None,
        )


class mockAdiabaticContractionMWP14ExplicitfbarWrapperPotential(
    AdiabaticContractionWrapperPotential
):
    def __init__(self):
        AdiabaticContractionWrapperPotential.__init__(
            self,
            pot=potential.MWPotential2014[2],
            baryonpot=potential.MWPotential2014[:2],
            f_bar=0.1,
        )

    def normalize(self, norm):
        self._amp *= norm / numpy.fabs(self.Rforce(1.0, 0.0, use_physical=False))


class mockRotatedAndTiltedMWP14WrapperPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                RotateAndTiltWrapperPotential(
                    pot=potential.MWPotential2014,
                    zvec=[
                        numpy.sqrt(1 / 3.0),
                        numpy.sqrt(1 / 3.0),
                        numpy.sqrt(1 / 3.0),
                    ],
                    galaxy_pa=0.4,
                )
            ],
        )

    def OmegaP(self):
        return 0.0


class mockRotatedAndTiltedMWP14WrapperPotentialwInclination(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                RotateAndTiltWrapperPotential(
                    pot=potential.MWPotential2014,
                    inclination=2.0,
                    galaxy_pa=0.3,
                    sky_pa=None,
                )
            ],
        )

    def OmegaP(self):
        return 0.0


class mockRotatedAndTiltedTriaxialLogHaloPotentialwInclination(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                RotateAndTiltWrapperPotential(
                    pot=potential.LogarithmicHaloPotential(normalize=1.0, b=0.7, q=0.5),
                    inclination=2.0,
                    galaxy_pa=0.3,
                    sky_pa=None,
                )
            ],
        )

    def OmegaP(self):
        return 0.0


class mockRotatedTiltedOffsetMWP14WrapperPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                RotateAndTiltWrapperPotential(
                    pot=potential.MWPotential2014,
                    zvec=[
                        numpy.sqrt(1 / 3.0),
                        numpy.sqrt(1 / 3.0),
                        numpy.sqrt(1 / 3.0),
                    ],
                    galaxy_pa=0.4,
                    offset=[1.0, 1.0, 1.0],
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


class mockOffsetMWP14WrapperPotential(testMWPotential):
    def __init__(self):
        testMWPotential.__init__(
            self,
            potlist=[
                RotateAndTiltWrapperPotential(
                    pot=potential.MWPotential2014,
                    zvec=None,
                    galaxy_pa=None,
                    offset=[1.0, 1.0, 1.0],
                ),
            ],
        )

    def OmegaP(self):
        return 0.0


# TimeDependentAmplitudeWrapperPotential
class mockTimeDependentAmplitudeWrapperPotential(
    TimeDependentAmplitudeWrapperPotential
):
    # Need to use __new__ because new Wrappers are created using __new__
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dpn = DehnenBarPotential(tform=-100.0, tsteady=1.0)  # on after t=-99
        dps = DehnenSmoothWrapperPotential(
            pot=dpn, tform=-4.0 * 2.0 * numpy.pi / dpn.OmegaP()
        )
        return DehnenSmoothWrapperPotential.__new__(
            cls, amp=1.0, pot=dpn, A=dps._smooth
        )


# A TimeDependentAmplitudeWrapperPotential version of LogarithmicHaloPotential for simple aAtest
class mockSmoothedLogarithmicHaloPotentialwTimeDependentAmplitudeWrapperPotential(
    TimeDependentAmplitudeWrapperPotential
):
    def __new__(cls, *args, **kwargs):
        if kwargs.get("_init", False):
            return parentWrapperPotential.__new__(cls, *args, **kwargs)
        dps = DehnenSmoothWrapperPotential(
            pot=potential.LogarithmicHaloPotential(normalize=1.0),
            tform=-1.0,
            tsteady=0.5,
        )
        return TimeDependentAmplitudeWrapperPotential.__new__(
            cls,
            amp=1.0,
            pot=potential.LogarithmicHaloPotential(normalize=1.0),
            A=dps._smooth,
        )


class mockKuzminLikeWrapperPotential(KuzminLikeWrapperPotential):
    def __init__(self):
        KuzminLikeWrapperPotential.__init__(
            self,
            pot=LogarithmicHaloPotential(normalize=1.0),
            a=1.0,
            b=0.1,
        )
