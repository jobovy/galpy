import sys
import warnings

import numpy
import pytest
from test_actionAngle import reset_warning_registry

from galpy.util import galpyWarning

PY2 = sys.version < "3"
# Print all galpyWarnings always for tests of warnings
warnings.simplefilter("always", galpyWarning)


# Basic sanity checking: circular orbit should have constant R, zero vR, vT=vc
def test_actionAngleTorus_basic():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import (
        FlattenedPowerPotential,
        MWPotential,
        PlummerPotential,
        rl,
        vcirc,
    )

    tol = -4.0
    jr = 10.0**-10.0
    jz = 10.0**-10.0
    aAT = actionAngleTorus(pot=MWPotential)
    # at R=1, Lz=1
    jphi = 1.0
    angler = numpy.linspace(0.0, 2.0 * numpy.pi, 101)
    anglephi = numpy.linspace(0.0, 2.0 * numpy.pi, 101) + 1.0
    anglez = numpy.linspace(0.0, 2.0 * numpy.pi, 101) + 2.0
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    assert numpy.all(
        numpy.fabs(RvR[0] - rl(MWPotential, jphi)) < 10.0**tol
    ), "circular orbit does not have constant radius for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[1]) < 10.0**tol
    ), "circular orbit does not have zero radial velocity for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[2] - vcirc(MWPotential, rl(MWPotential, jphi))) < 10.0**tol
    ), "circular orbit does not have constant vT=vc for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[3]) < 10.0**tol
    ), "circular orbit does not have zero vertical height for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[4]) < 10.0**tol
    ), "circular orbit does not have zero vertical velocity for actionAngleTorus"
    # at Lz=1.5, using Plummer
    tol = -3.25
    pp = PlummerPotential(normalize=1.0)
    aAT = actionAngleTorus(pot=pp)
    jphi = 1.5
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    assert numpy.all(
        numpy.fabs(RvR[0] - rl(pp, jphi)) < 10.0**tol
    ), "circular orbit does not have constant radius for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[1]) < 10.0**tol
    ), "circular orbit does not have zero radial velocity for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[2] - vcirc(pp, rl(pp, jphi))) < 10.0**tol
    ), "circular orbit does not have constant vT=vc for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[3]) < 10.0**tol
    ), "circular orbit does not have zero vertical height for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[4]) < 10.0**tol
    ), "circular orbit does not have zero vertical velocity for actionAngleTorus"
    # at Lz=0.5, using FlattenedPowerPotential
    tol = -4.0
    fp = FlattenedPowerPotential(normalize=1.0)
    aAT = actionAngleTorus(pot=fp)
    jphi = 0.5
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    assert numpy.all(
        numpy.fabs(RvR[0] - rl(fp, jphi)) < 10.0**tol
    ), "circular orbit does not have constant radius for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[1]) < 10.0**tol
    ), "circular orbit does not have zero radial velocity for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[2] - vcirc(fp, rl(fp, jphi))) < 10.0**tol
    ), "circular orbit does not have constant vT=vc for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[3]) < 10.0**tol
    ), "circular orbit does not have zero vertical height for actionAngleTorus"
    assert numpy.all(
        numpy.fabs(RvR[4]) < 10.0**tol
    ), "circular orbit does not have zero vertical velocity for actionAngleTorus"
    return None


# Basic sanity checking: close-to-circular orbit should have freq. = epicycle freq.
def test_actionAngleTorus_basic_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import (
        HernquistPotential,
        JaffePotential,
        PowerSphericalPotential,
        epifreq,
        omegac,
        rl,
        verticalfreq,
    )

    tol = -3.0
    jr = 10.0**-6.0
    jz = 10.0**-6.0
    jp = JaffePotential(normalize=1.0)
    aAT = actionAngleTorus(pot=jp)
    # at Lz=1
    jphi = 1.0
    om = aAT.Freqs(jr, jphi, jz)
    assert (
        numpy.fabs((om[0] - epifreq(jp, rl(jp, jphi))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - omegac(jp, rl(jp, jphi))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - verticalfreq(jp, rl(jp, jphi))) / om[2]) < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    # at Lz=1.5, w/ different potential
    pp = PowerSphericalPotential(normalize=1.0)
    aAT = actionAngleTorus(pot=pp)
    jphi = 1.5
    om = aAT.Freqs(jr, jphi, jz)
    assert (
        numpy.fabs((om[0] - epifreq(pp, rl(pp, jphi))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - omegac(pp, rl(pp, jphi))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - verticalfreq(pp, rl(pp, jphi))) / om[2]) < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    # at Lz=0.5, w/ different potential
    tol = -2.5  # appears more difficult
    hp = HernquistPotential(normalize=1.0)
    aAT = actionAngleTorus(pot=hp)
    jphi = 0.5
    om = aAT.Freqs(jr, jphi, jz)
    assert (
        numpy.fabs((om[0] - epifreq(hp, rl(hp, jphi))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - omegac(hp, rl(hp, jphi))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - verticalfreq(hp, rl(hp, jphi))) / om[2]) < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    return None


# Test that orbit from actionAngleTorus is the same as an integrated orbit
def test_actionAngleTorus_orbit():
    from galpy.actionAngle import actionAngleTorus
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Set up instance
    aAT = actionAngleTorus(pot=MWPotential2014, tol=10.0**-5.0)
    jr, jphi, jz = 0.05, 1.1, 0.025
    # First calculate frequencies and the initial RvR
    RvRom = aAT.xvFreqs(
        jr, jphi, jz, numpy.array([0.0]), numpy.array([1.0]), numpy.array([2.0])
    )
    om = RvRom[1:]
    # Angles along an orbit
    ts = numpy.linspace(0.0, 100.0, 1001)
    angler = ts * om[0]
    anglephi = 1.0 + ts * om[1]
    anglez = 2.0 + ts * om[2]
    # Calculate the orbit using actionAngleTorus
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    # Calculate the orbit using orbit integration
    orb = Orbit(
        [
            RvRom[0][0, 0],
            RvRom[0][0, 1],
            RvRom[0][0, 2],
            RvRom[0][0, 3],
            RvRom[0][0, 4],
            RvRom[0][0, 5],
        ]
    )
    orb.integrate(ts, MWPotential2014)
    # Compare
    tol = -3.0
    assert numpy.all(
        numpy.fabs(orb.R(ts) - RvR[0]) < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in R"
    assert numpy.all(
        numpy.fabs(orb.vR(ts) - RvR[1]) < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in vR"
    assert numpy.all(
        numpy.fabs(orb.vT(ts) - RvR[2]) < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in vT"
    assert numpy.all(
        numpy.fabs(orb.z(ts) - RvR[3]) < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in z"
    assert numpy.all(
        numpy.fabs(orb.vz(ts) - RvR[4]) < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in vz"
    assert numpy.all(
        numpy.fabs((orb.phi(ts) - RvR[5] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in phi"
    return None


# Test that actionAngleTorus w/ interp pot gives same freqs as regular pot
# Doesn't work well: TM aborts because our interpolated forces aren't
# consistent enough with the potential for TM's taste, but we test that it at
# at least works somewhat
def test_actionAngleTorus_interppot_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import LogarithmicHaloPotential, interpRZPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ip = interpRZPotential(
        RZPot=lp,
        interpPot=True,
        interpDens=True,
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
    )
    aAT = actionAngleTorus(pot=lp)
    aATi = actionAngleTorus(pot=ip)
    jr, jphi, jz = 0.05, 1.1, 0.02
    om = aAT.Freqs(jr, jphi, jz)
    omi = aATi.Freqs(jr, jphi, jz)
    assert (
        numpy.fabs((om[0] - omi[0]) / om[0]) < 0.2
    ), "Radial frequency computed using the torus machine does not agree between potential and interpolated potential"
    assert (
        numpy.fabs((om[1] - omi[1]) / om[1]) < 0.2
    ), "Azimuthal frequency computed using the torus machine does not agree between potential and interpolated potential"
    assert (
        numpy.fabs((om[2] - omi[2]) / om[2]) < 0.8
    ), "Vertical frequency computed using the torus machine does not agree between potential and interpolated potential"
    return None


# Test the actionAngleTorus against an isochrone potential: actions
def test_actionAngleTorus_Isochrone_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleTorus
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    tol = -6.0
    aAT = actionAngleTorus(pot=ip, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.0])
    anglephi = numpy.array([numpy.pi])
    anglez = numpy.array([numpy.pi / 2.0])
    # Calculate position from aAT
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    # Calculate actions from aAI
    ji = aAI(*RvR)
    djr = numpy.fabs((ji[0] - jr) / jr)
    dlz = numpy.fabs((ji[1] - jphi) / jphi)
    djz = numpy.fabs((ji[2] - jz) / jz)
    assert djr < 10.0**tol, (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%"
        % (djr * 100.0)
    )
    assert dlz < 10.0**tol, (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**tol, (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Jr at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleTorus against an isochrone potential: frequencies and angles
def test_actionAngleTorus_Isochrone_freqsAngles():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleTorus
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    tol = -6.0
    aAT = actionAngleTorus(pot=ip, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.1]) + numpy.linspace(0.0, numpy.pi, 101)
    angler = angler % (2.0 * numpy.pi)
    anglephi = numpy.array([numpy.pi]) + numpy.linspace(0.0, numpy.pi, 101)
    anglephi = anglephi % (2.0 * numpy.pi)
    anglez = numpy.array([numpy.pi / 2.0]) + numpy.linspace(0.0, numpy.pi, 101)
    anglez = anglez % (2.0 * numpy.pi)
    # Calculate position from aAT
    RvRom = aAT.xvFreqs(jr, jphi, jz, angler, anglephi, anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws = aAI.actionsFreqsAngles(*RvRom[0].T)
    dOr = numpy.fabs(ws[3] - RvRom[1])
    dOp = numpy.fabs(ws[4] - RvRom[2])
    dOz = numpy.fabs(ws[5] - RvRom[3])
    dar = numpy.fabs(ws[6] - angler)
    dap = numpy.fabs(ws[7] - anglephi)
    daz = numpy.fabs(ws[8] - anglez)
    dar[dar > numpy.pi] -= 2.0 * numpy.pi
    dar[dar < -numpy.pi] += 2.0 * numpy.pi
    dap[dap > numpy.pi] -= 2.0 * numpy.pi
    dap[dap < -numpy.pi] += 2.0 * numpy.pi
    daz[daz > numpy.pi] -= 2.0 * numpy.pi
    daz[daz < -numpy.pi] += 2.0 * numpy.pi
    assert numpy.all(dOr < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Or at %f%%"
        % (numpy.nanmax(dOr) * 100.0)
    )
    assert numpy.all(dOp < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Ophi at %f%%"
        % (numpy.nanmax(dOp) * 100.0)
    )
    assert numpy.all(dOz < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for Oz at %f%%"
        % (numpy.nanmax(dOz) * 100.0)
    )
    assert numpy.all(dar < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for ar at %f"
        % (numpy.nanmax(dar))
    )
    assert numpy.all(dap < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for aphi at %f"
        % (numpy.nanmax(dap))
    )
    assert numpy.all(daz < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochrone applied to isochrone potential disagree for az at %f"
        % (numpy.nanmax(daz))
    )
    return None


# Test the actionAngleTorus against a Staeckel potential: actions
def test_actionAngleTorus_Staeckel_actions():
    from galpy.actionAngle import actionAngleStaeckel, actionAngleTorus
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.2
    kp = KuzminKutuzovStaeckelPotential(normalize=1.0, Delta=delta)
    aAS = actionAngleStaeckel(pot=kp, delta=delta, c=True)
    tol = -3.0
    aAT = actionAngleTorus(pot=kp, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.0])
    anglephi = numpy.array([numpy.pi])
    anglez = numpy.array([numpy.pi / 2.0])
    # Calculate position from aAT
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    # Calculate actions from aAI
    ji = aAS(*RvR)
    djr = numpy.fabs((ji[0] - jr) / jr)
    dlz = numpy.fabs((ji[1] - jphi) / jphi)
    djz = numpy.fabs((ji[2] - jz) / jz)
    assert djr < 10.0**tol, (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%"
        % (djr * 100.0)
    )
    assert dlz < 10.0**tol, (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**tol, (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Jr at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleTorus against an isochrone potential: frequencies and angles
def test_actionAngleTorus_Staeckel_freqsAngles():
    from galpy.actionAngle import actionAngleStaeckel, actionAngleTorus
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.2
    kp = KuzminKutuzovStaeckelPotential(normalize=1.0, Delta=delta)
    aAS = actionAngleStaeckel(pot=kp, delta=delta, c=True)
    tol = -3.0
    aAT = actionAngleTorus(pot=kp, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.1]) + numpy.linspace(0.0, numpy.pi, 101)
    angler = angler % (2.0 * numpy.pi)
    anglephi = numpy.array([numpy.pi]) + numpy.linspace(0.0, numpy.pi, 101)
    anglephi = anglephi % (2.0 * numpy.pi)
    anglez = numpy.array([numpy.pi / 2.0]) + numpy.linspace(0.0, numpy.pi, 101)
    anglez = anglez % (2.0 * numpy.pi)
    # Calculate position from aAT
    RvRom = aAT.xvFreqs(jr, jphi, jz, angler, anglephi, anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws = aAS.actionsFreqsAngles(*RvRom[0].T)
    dOr = numpy.fabs(ws[3] - RvRom[1])
    dOp = numpy.fabs(ws[4] - RvRom[2])
    dOz = numpy.fabs(ws[5] - RvRom[3])
    dar = numpy.fabs(ws[6] - angler)
    dap = numpy.fabs(ws[7] - anglephi)
    daz = numpy.fabs(ws[8] - anglez)
    dar[dar > numpy.pi] -= 2.0 * numpy.pi
    dar[dar < -numpy.pi] += 2.0 * numpy.pi
    dap[dap > numpy.pi] -= 2.0 * numpy.pi
    dap[dap < -numpy.pi] += 2.0 * numpy.pi
    daz[daz > numpy.pi] -= 2.0 * numpy.pi
    daz[daz < -numpy.pi] += 2.0 * numpy.pi
    assert numpy.all(dOr < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Or at %f%%"
        % (numpy.nanmax(dOr) * 100.0)
    )
    assert numpy.all(dOp < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Ophi at %f%%"
        % (numpy.nanmax(dOp) * 100.0)
    )
    assert numpy.all(dOz < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for Oz at %f%%"
        % (numpy.nanmax(dOz) * 100.0)
    )
    assert numpy.all(dar < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for ar at %f"
        % (numpy.nanmax(dar))
    )
    assert numpy.all(dap < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for aphi at %f"
        % (numpy.nanmax(dap))
    )
    assert numpy.all(daz < 10.0**tol), (
        "actionAngleTorus and actionAngleStaeckel applied to Staeckel potential disagree for az at %f"
        % (numpy.nanmax(daz))
    )
    return None


# Test the actionAngleTorus against a general potential w/ actionAngleIsochroneApprox: actions
def test_actionAngleTorus_isochroneApprox_actions():
    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.potential import MWPotential2014

    aAIA = actionAngleIsochroneApprox(pot=MWPotential2014, b=0.8)
    tol = -2.5
    aAT = actionAngleTorus(pot=MWPotential2014, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.0])
    anglephi = numpy.array([numpy.pi])
    anglez = numpy.array([numpy.pi / 2.0])
    # Calculate position from aAT
    RvR = aAT(jr, jphi, jz, angler, anglephi, anglez).T
    # Calculate actions from aAIA
    ji = aAIA(*RvR)
    djr = numpy.fabs((ji[0] - jr) / jr)
    dlz = numpy.fabs((ji[1] - jphi) / jphi)
    djz = numpy.fabs((ji[2] - jz) / jz)
    assert djr < 10.0**tol, (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Jr at %f%%"
        % (djr * 100.0)
    )
    assert dlz < 10.0**tol, (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Jr at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**tol, (
        "actionAngleTorus and actionAngleMWPotential2014 applied to MWPotential2014 potential disagree for Jr at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleTorus against a general potential w/ actionAngleIsochrone: frequencies and angles
def test_actionAngleTorus_isochroneApprox_freqsAngles():
    from galpy.actionAngle import actionAngleIsochroneApprox, actionAngleTorus
    from galpy.potential import MWPotential2014

    aAIA = actionAngleIsochroneApprox(pot=MWPotential2014, b=0.8)
    tol = -3.5
    aAT = actionAngleTorus(pot=MWPotential2014, tol=tol)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.1]) + numpy.linspace(0.0, numpy.pi, 21)
    angler = angler % (2.0 * numpy.pi)
    anglephi = numpy.array([numpy.pi]) + numpy.linspace(0.0, numpy.pi, 21)
    anglephi = anglephi % (2.0 * numpy.pi)
    anglez = numpy.array([numpy.pi / 2.0]) + numpy.linspace(0.0, numpy.pi, 21)
    anglez = anglez % (2.0 * numpy.pi)
    # Calculate position from aAT
    RvRom = aAT.xvFreqs(jr, jphi, jz, angler, anglephi, anglez)
    # Calculate actions, frequencies, and angles from aAI
    ws = aAIA.actionsFreqsAngles(*RvRom[0].T)
    dOr = numpy.fabs(ws[3] - RvRom[1])
    dOp = numpy.fabs(ws[4] - RvRom[2])
    dOz = numpy.fabs(ws[5] - RvRom[3])
    dar = numpy.fabs(ws[6] - angler)
    dap = numpy.fabs(ws[7] - anglephi)
    daz = numpy.fabs(ws[8] - anglez)
    dar[dar > numpy.pi] -= 2.0 * numpy.pi
    dar[dar < -numpy.pi] += 2.0 * numpy.pi
    dap[dap > numpy.pi] -= 2.0 * numpy.pi
    dap[dap < -numpy.pi] += 2.0 * numpy.pi
    daz[daz > numpy.pi] -= 2.0 * numpy.pi
    daz[daz < -numpy.pi] += 2.0 * numpy.pi
    assert numpy.all(dOr < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Or at %f%%"
        % (numpy.nanmax(dOr) * 100.0)
    )
    assert numpy.all(dOp < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Ophi at %f%%"
        % (numpy.nanmax(dOp) * 100.0)
    )
    assert numpy.all(dOz < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for Oz at %f%%"
        % (numpy.nanmax(dOz) * 100.0)
    )
    assert numpy.all(dar < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for ar at %f"
        % (numpy.nanmax(dar))
    )
    assert numpy.all(dap < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for aphi at %f"
        % (numpy.nanmax(dap))
    )
    assert numpy.all(daz < 10.0**tol), (
        "actionAngleTorus and actionAngleIsochroneApprox applied to MWPotential2014 potential disagree for az at %f"
        % (numpy.nanmax(daz))
    )
    return None


# Test that the frequencies returned by hessianFreqs are the same as those returned by Freqs
def test_actionAngleTorus_hessian_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    freqO = aAT.Freqs(jr, jphi, jz)[:3]
    hessO = aAT.hessianFreqs(jr, jphi, jz)[1:4]
    assert numpy.all(
        numpy.fabs(numpy.array(freqO) - numpy.array(hessO)) < 10.0**-8.0
    ), "actionAngleTorus methods Freqs and hessianFreqs return different frequencies"
    return None


# Test that the Hessian is approximately symmetric
def test_actionAngleTorus_hessian_symm():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    h = aAT.hessianFreqs(jr, jphi, jz, tol=0.0001, nosym=True)[0]
    assert numpy.all(
        numpy.fabs((h - h.T) / h) < 0.03
    ), "actionAngleTorus Hessian is not symmetric"
    return None


# Test that the Hessian is approximately correct
def test_actionAngleTorus_hessian_linear():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    h = aAT.hessianFreqs(jr, jphi, jz, tol=0.0001, nosym=True)[0]
    dj = numpy.array([0.02, 0.005, -0.01])
    do_fromhessian = numpy.dot(h, dj)
    O = numpy.array(aAT.Freqs(jr, jphi, jz)[:3])
    do = numpy.array(aAT.Freqs(jr + dj[0], jphi + dj[1], jz + dj[2])[:3]) - O
    assert numpy.all(
        numpy.fabs((do_fromhessian - do) / O) < 0.001
    ), "actionAngleTorus Hessian does not return good approximation to dO/dJ"
    return None


# Test that the frequencies returned by xvJacobianFreqs are the same as those returned by Freqs
def test_actionAngleTorus_jacobian_freqs():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    freqO = aAT.Freqs(jr, jphi, jz)[:3]
    hessO = aAT.xvJacobianFreqs(
        jr, jphi, jz, numpy.array([0.0]), numpy.array([1.0]), numpy.array([2.0])
    )[3:6]
    assert numpy.all(
        numpy.fabs(numpy.array(freqO) - numpy.array(hessO)) < 10.0**-8.0
    ), "actionAngleTorus methods Freqs and xvJacobianFreqs return different frequencies"
    return None


# Test that the Hessian returned by xvJacobianFreqs are the same as those returned by hessianFreqs
def test_actionAngleTorus_jacobian_hessian():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    freqO = aAT.hessianFreqs(jr, jphi, jz)[0]
    hessO = aAT.xvJacobianFreqs(
        jr, jphi, jz, numpy.array([0.0]), numpy.array([1.0]), numpy.array([2.0])
    )[2]
    assert numpy.all(
        numpy.fabs(numpy.array(freqO) - numpy.array(hessO)) < 10.0**-8.0
    ), "actionAngleTorus methods hessianFreqs and xvJacobianFreqs return different Hessians"
    return None


# Test that the xv returned by xvJacobianFreqs are the same as those returned by __call__
def test_actionAngleTorus_jacobian_xv():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.0, 1.0])
    anglephi = numpy.array([1.0, 2.0])
    anglez = numpy.array([2.0, 3.0])
    freqO = aAT(jr, jphi, jz, angler, anglephi, anglez)
    hessO = aAT.xvJacobianFreqs(jr, jphi, jz, angler, anglephi, anglez)[0]
    assert numpy.all(
        numpy.fabs(numpy.array(freqO) - numpy.array(hessO)) < 10.0**-8.0
    ), "actionAngleTorus methods __call__ and xvJacobianFreqs return different xv"
    return None


# Test that the determinant of the Jacobian returned by xvJacobianFreqs is close to 1/R (should be 1 for rectangular coordinates, 1/R for cylindrical
def test_actionAngleTorus_jacobian_detone():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014, dJ=0.0001)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.0, 1.0])
    anglephi = numpy.array([1.0, 2.0])
    anglez = numpy.array([2.0, 3.0])
    jf = aAT.xvJacobianFreqs(jr, jphi, jz, angler, anglephi, anglez)
    assert (
        numpy.fabs(jf[0][0, 0] * numpy.fabs(numpy.linalg.det(jf[1][0])) - 1) < 0.01
    ), "Jacobian returned by actionAngleTorus method xvJacobianFreqs does not have the expected determinant"
    assert (
        numpy.fabs(jf[0][1, 0] * numpy.fabs(numpy.linalg.det(jf[1][1])) - 1) < 0.01
    ), "Jacobian returned by actionAngleTorus method xvJacobianFreqs does not have the expected determinant"
    return None


# Test that Jacobian returned by xvJacobianFreqs is approximately correct
def test_actionAngleTorus_jacobian_linear():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential2014

    aAT = actionAngleTorus(pot=MWPotential2014)
    jr, jphi, jz = 0.075, 1.1, 0.05
    angler = numpy.array([0.5])
    anglephi = numpy.array([1.0])
    anglez = numpy.array([2.0])
    jf = aAT.xvJacobianFreqs(jr, jphi, jz, angler, anglephi, anglez)
    xv = aAT(jr, jphi, jz, angler, anglephi, anglez)
    dja = 2.0 * numpy.array([0.001, 0.002, 0.003, -0.002, 0.004, 0.002])
    xv_direct = aAT(
        jr + dja[0],
        jphi + dja[1],
        jz + dja[2],
        angler + dja[3],
        anglephi + dja[4],
        anglez + dja[5],
    )
    xv_fromjac = xv + numpy.dot(jf[1], dja)
    assert numpy.all(
        numpy.fabs((xv_fromjac - xv_direct) / xv_direct) < 0.01
    ), "Jacobian returned by actionAngleTorus method xvJacobianFreqs does not appear to be correct"
    return None


# Test error when potential is not implemented in C
def test_actionAngleTorus_nocerr():
    from test_potential import BurkertPotentialNoC

    from galpy.actionAngle import actionAngleTorus

    bp = BurkertPotentialNoC()
    try:
        aAT = actionAngleTorus(pot=bp)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "actionAngleTorus initialization with potential w/o C should have given a RuntimeError, but didn't"
        )
    return None


# Test error when potential is not axisymmetric
def test_actionAngleTorus_nonaxierr():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import TriaxialNFWPotential

    np = TriaxialNFWPotential(normalize=1.0, b=0.9)
    try:
        aAT = actionAngleTorus(pot=np)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "actionAngleTorus initialization with non-axisymmetric potential should have given a RuntimeError, but didn't"
        )
    return None


# Test the Autofit torus warnings
def test_actionAngleTorus_AutoFitWarning():
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAT = actionAngleTorus(pot=lp, tol=10.0**-8.0)
    # These should give warnings
    jr, jp, jz = 0.27209033, 1.80253892, 0.6078445
    ar, ap, az = (
        numpy.array([1.95732492]),
        numpy.array([6.16753224]),
        numpy.array([4.08233059]),
    )
    # Turn warnings into errors to test for them
    import warnings

    with warnings.catch_warnings(record=True) as w:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        aAT(jr, jp, jz, ar, ap, az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAT.xvFreqs(jr, jp, jz, ar, ap, az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAT.Freqs(jr, jp, jz)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAT.hessianFreqs(jr, jp, jz)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAT.xvJacobianFreqs(jr, jp, jz, ar, ap, az)
        # Should raise warning bc of Autofit, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "actionAngleTorus' AutoFit exited with non-zero return status -3: Fit failed the goal by more than 2"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleTorus with flattened LogarithmicHaloPotential and a particular orbit should have thrown a warning, but didn't"
    return None


def test_MWPotential_warning_torus():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleTorus
    from galpy.potential import MWPotential

    if PY2:
        reset_warning_registry("galpy")
    warnings.simplefilter("error", galpyWarning)
    try:
        aAA = actionAngleTorus(pot=MWPotential)
    except:
        pass
    else:
        raise AssertionError(
            "actionAngleTorus with MWPotential should have thrown a warning, but didn't"
        )
    # Turn warnings back into warnings
    warnings.simplefilter("always", galpyWarning)
    return None


def test_load_library():
    # Test that loading the library again gives the same library as the first
    # time
    from galpy.util._load_extension_libs import load_libgalpy_actionAngleTorus

    first_lib = load_libgalpy_actionAngleTorus()[0]
    second_lib = load_libgalpy_actionAngleTorus()[0]
    assert (
        first_lib == second_lib
    ), "libgalpy_actionAngleTorus loaded second time is not the same as first time"
    return None
