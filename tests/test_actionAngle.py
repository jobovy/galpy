import sys
import warnings

import numpy
import pytest

from galpy.util import galpyWarning

PY2 = sys.version < "3"
# Print all galpyWarnings always for tests of warnings
warnings.simplefilter("always", galpyWarning)


# Test the actions of an actionAngleHarmonic
def test_actionAngleHarmonic_conserved_actions():
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleHarmonic
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    ipz = ip.toVertical(1.2)
    # Omega = sqrt(4piG density / 3)
    aAH = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    obs = Orbit([0.1, -0.3])
    ntimes = 1001
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, ipz)
    js = aAH(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(
        numpy.fabs(js - numpy.tile(numpy.mean(js), (len(times), 1)).T)
    ) / numpy.mean(js)
    assert maxdj < 10.0**-4.0, "Action conservation fails at %g%%" % (100.0 * maxdj)
    return None


# Test that the angles of an actionAngleHarmonic increase linearly
def test_actionAngleHarmonic_linear_angles():
    from galpy.actionAngle import actionAngleHarmonic, dePeriod
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    ipz = ip.toVertical(1.2)
    # Omega = sqrt(4piG density / 3)
    aAH = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    obs = Orbit([0.1, -0.3])
    ntimes = 1001
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, ipz)
    acfs_init = aAH.actionsFreqsAngles(obs.x(), obs.vx())  # to check the init. angles
    acfs = aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))
    angle = dePeriod(numpy.reshape(acfs[2], (1, len(times)))).flatten()
    # Do linear fit to the angle, check that deviations are small, check
    # that the slope is the frequency
    linfit = numpy.polyfit(times, angle, 1)
    assert numpy.fabs((linfit[1] - acfs_init[2]) / acfs_init[2]) < 10.0**-5.0, (
        "Angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (100.0 * numpy.fabs((linfit[1] - acfs_init[2]) / acfs_init[2]))
    )
    assert numpy.fabs(linfit[0] - acfs_init[1]) < 10.0**-5.0, (
        "Frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (100.0 * numpy.fabs((linfit[0] - acfs_init[1]) / acfs_init[1]))
    )
    devs = angle - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**-6.0, (
        "Maximum deviation from linear trend in the angles is %g" % maxdev
    )
    # Finally test that the frequency returned by actionsFreqs == that from actionsFreqsAngles
    assert (
        numpy.all(
            numpy.fabs(
                aAH.actionsFreqs(obs.x(times), obs.vx(times))[1]
                - aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))[1]
            )
        )
        < 1e-100
    ), (
        "Frequency returned by actionsFreqs not equal to that returned by actionsFreqsAngles"
    )
    return None


# Test physical output for actionAngleHarmonic
def test_physical_harmonic():
    from galpy.actionAngle import actionAngleHarmonic
    from galpy.potential import IsochronePotential
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aAH = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=ro, vo=vo
    )
    aAHnu = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    # __call__
    assert numpy.fabs(aAH(-0.1, 0.1) - aAHnu(-0.1, 0.1) * ro * vo) < 10.0**-8.0, (
        "actionAngle function __call__ does not return Quantity with the right value for actionAngleHarmonic"
    )
    # actionsFreqs
    assert (
        numpy.fabs(
            aAH.actionsFreqs(0.2, 0.1)[0] - aAHnu.actionsFreqs(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleHarmonic"
    )
    assert (
        numpy.fabs(
            aAH.actionsFreqs(0.2, 0.1)[1]
            - aAHnu.actionsFreqs(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleHarmonic"
    )
    # actionsFreqsAngles
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[0]
            - aAHnu.actionsFreqsAngles(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
    )
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[1]
            - aAHnu.actionsFreqsAngles(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
    )
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[2] - aAHnu.actionsFreqsAngles(0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
    )
    return None


# Test the actions of an actionAngleVertical
def test_actionAngleVertical_conserved_actions():
    # Use an isothermal disk potential
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    obs = Orbit([0.1, -0.3])
    ntimes = 1001
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, isopot)
    js = aAV(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(
        numpy.fabs(
            (js - numpy.tile(numpy.mean(js), (len(times), 1)).T) / numpy.mean(js)
        )
    )
    assert maxdj < 10.0**-4.0, "Action conservation fails at %g%%" % (100.0 * maxdj)
    return None


# Test the frequencies of an actionAngleVertical
def test_actionAngleVertical_conserved_freqs():
    # Use an isothermal disk potential
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    obs = Orbit([0.1, -0.3])
    ntimes = 1001
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, isopot)
    js, os = aAV.actionsFreqs(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(
        numpy.fabs(
            (js - numpy.tile(numpy.mean(js), (len(times), 1)).T) / numpy.mean(js)
        )
    )
    assert maxdj < 10.0**-4.0, "Action conservation fails at %g%%" % (100.0 * maxdj)
    maxdo = numpy.amax(
        numpy.fabs(
            (os - numpy.tile(numpy.mean(os), (len(times), 1)).T) / numpy.mean(os)
        )
    )
    assert maxdo < 10.0**-4.0, "Frequency conservation fails at %g%%" % (100.0 * maxdo)
    return None


# Test that the angles of an actionAngleVertical increase linearly
def test_actionAngleVertical_linear_angles():
    from galpy.actionAngle import actionAngleVertical, dePeriod
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    obs = Orbit([0.1, -0.3])
    ntimes = 1001
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, isopot)
    acfs_init = aAV.actionsFreqsAngles(obs.x(), obs.vx())  # to check the init. angles
    acfs = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    angle = dePeriod(numpy.reshape(acfs[2], (1, len(times)))).flatten()
    # Do linear fit to the angle, check that deviations are small, check
    # that the slope is the frequency
    linfit = numpy.polyfit(times, angle, 1)
    assert numpy.fabs((linfit[1] - acfs_init[2]) / acfs_init[2]) < 10.0**-5.0, (
        "Angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (100.0 * numpy.fabs((linfit[1] - acfs_init[2]) / acfs_init[2]))
    )
    assert numpy.fabs(linfit[0] - acfs_init[1]) < 10.0**-5.0, (
        "Frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (100.0 * numpy.fabs((linfit[0] - acfs_init[1]) / acfs_init[1]))
    )
    devs = angle - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**-6.0, (
        "Maximum deviation from linear trend in the angles is %g" % maxdev
    )
    # Finally test that the frequency returned by actionsFreqs == that from actionsFreqsAngles
    assert (
        numpy.all(
            numpy.fabs(
                aAV.actionsFreqs(obs.x(times), obs.vx(times))[1]
                - aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))[1]
            )
        )
        < 1e-100
    ), (
        "Frequency returned by actionsFreqs not equal to that returned by actionsFreqsAngles"
    )
    return None


# Test that unbound orbits are handled properly
def test_actionAngleVertical_unbound():
    from galpy.actionAngle import actionAngleVertical
    from galpy.potential import (
        MWPotential2014,
        evaluatelinearPotentials,
        toVerticalPotential,
    )

    mwp14_v = toVerticalPotential(MWPotential2014, 1.0)
    aAV = actionAngleVertical(pot=mwp14_v)
    vesc = numpy.sqrt(
        2.0
        * (
            evaluatelinearPotentials(mwp14_v, numpy.inf)
            - evaluatelinearPotentials(mwp14_v, 0.0)
        )
    )
    assert numpy.fabs(aAV(0.0, vesc + 1e-4) - 9999.99) < 10.0**-8.0, (
        "actionAngleVertical does not return J=9999.99 for unbound orbits"
    )
    assert numpy.fabs(aAV.actionsFreqs(0.0, vesc + 1e-4)[0] - 9999.99) < 10.0**-8.0, (
        "actionAngleVertical does not return J=9999.99 for unbound orbits"
    )
    assert numpy.fabs(aAV.actionsFreqs(0.0, vesc + 1e-4)[1] - 9999.99) < 10.0**-8.0, (
        "actionAngleVertical does not return O=9999.99 for unbound orbits"
    )
    assert (
        numpy.fabs(aAV.actionsFreqsAngles(0.0, vesc + 1e-4)[0] - 9999.99) < 10.0**-8.0
    ), "actionAngleVertical does not return J=9999.99 for unbound orbits"
    assert (
        numpy.fabs(aAV.actionsFreqsAngles(0.0, vesc + 1e-4)[1] - 9999.99) < 10.0**-8.0
    ), "actionAngleVertical does not return O=9999.99 for unbound orbits"
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.0, vesc + 1e-4)[2]
            - ((9999.99 * 9999.99) % (2 * numpy.pi))
        )
        < 10.0**-8.0
    ), "actionAngleVertical does not return O=9999.99 for unbound orbits"
    return None


# Test actionAngleVertical against actionAngleHarmonic for HO
def test_actionAngleVertical_Harmonic_actions():
    from galpy.actionAngle import actionAngleHarmonic, actionAngleVertical
    from galpy.orbit import Orbit
    from galpy.potential import linearPotential

    # Stop-gap until we implement a proper 1D (or 3D) HO potential,
    # limit of taking Isochrone leads to 1e-7 fluctuations in the potential
    # that mess up this test
    class HO(linearPotential):
        def __init__(self, omega):
            linearPotential.__init__(self, amp=1.0)
            self._omega = omega

        def _evaluate(self, x, t=0.0):
            return self._omega**2.0 * x**2.0 / 2.0

        def _force(self, x, t=0.0):
            return -(self._omega**2.0) * x

    ipz = HO(omega=2.23)
    aAH = actionAngleHarmonic(omega=ipz._omega)
    aAV = actionAngleVertical(pot=ipz)
    obs = Orbit([0.1, -0.3])
    ntimes = 101
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, ipz)
    js = aAH(obs.x(times), obs.vx(times))
    jsv = aAV(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(numpy.fabs((js - jsv) / js))
    assert maxdj < 10.0**-10.0, (
        "Actions of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdj)
    )
    return None


def test_actionAngleVertical_Harmonic_actionsFreqs():
    from galpy.actionAngle import actionAngleHarmonic, actionAngleVertical
    from galpy.orbit import Orbit
    from galpy.potential import linearPotential

    # Stop-gap until we implement a proper 1D (or 3D) HO potential,
    # limit of taking Isochrone leads to 1e-7 fluctuations in the potential
    # that mess up this test
    class HO(linearPotential):
        def __init__(self, omega):
            linearPotential.__init__(self, amp=1.0)
            self._omega = omega

        def _evaluate(self, x, t=0.0):
            return self._omega**2.0 * x**2.0 / 2.0

        def _force(self, x, t=0.0):
            return -(self._omega**2.0) * x

    ipz = HO(omega=2.23)
    aAH = actionAngleHarmonic(omega=ipz._omega)
    aAV = actionAngleVertical(pot=ipz)
    obs = Orbit([0.1, -0.3])
    ntimes = 101
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, ipz)
    js, os = aAH.actionsFreqs(obs.x(times), obs.vx(times))
    jsv, osv = aAV.actionsFreqs(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(numpy.fabs((js - jsv) / js))
    assert maxdj < 10.0**-10.0, (
        "Actions of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdj)
    )
    maxdo = numpy.amax(numpy.fabs((os - osv) / os))
    assert maxdo < 10.0**-10.0, (
        "Frequencies of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdo)
    )
    return None


def test_actionAngleVertical_Harmonic_actionsFreqsAngles():
    from galpy.actionAngle import actionAngleHarmonic, actionAngleVertical
    from galpy.orbit import Orbit
    from galpy.potential import linearPotential

    # Stop-gap until we implement a proper 1D (or 3D) HO potential,
    # limit of taking Isochrone leads to 1e-7 fluctuations in the potential
    # that mess up this test
    class HO(linearPotential):
        def __init__(self, omega):
            linearPotential.__init__(self, amp=1.0)
            self._omega = omega

        def _evaluate(self, x, t=0.0):
            return self._omega**2.0 * x**2.0 / 2.0

        def _force(self, x, t=0.0):
            return -(self._omega**2.0) * x

    ipz = HO(omega=2.236)
    aAH = actionAngleHarmonic(omega=ipz._omega)
    aAV = actionAngleVertical(pot=ipz)
    obs = Orbit([0.1, -0.3])
    ntimes = 101
    times = numpy.linspace(0.0, 20.0, ntimes)
    obs.integrate(times, ipz)
    js, os, anss = aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))
    jsv, osv, anssv = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    maxdj = numpy.amax(numpy.fabs((js - jsv) / js))
    assert maxdj < 10.0**-10.0, (
        "Actions of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdj)
    )
    maxdo = numpy.amax(numpy.fabs((os - osv) / os))
    assert maxdo < 10.0**-10.0, (
        "Frequencies of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdo)
    )
    maxda = numpy.amax(
        numpy.fabs(((anss - anssv) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
    )
    assert maxda < 10.0**-10.0, (
        "Angles of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxda)
    )
    return None


# Test physical output for actionAngleVertical
def test_physical_vertical():
    from galpy.actionAngle import actionAngleVertical
    from galpy.potential import IsothermalDiskPotential
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    # Omega = sqrt(4piG density / 3)
    aAV = actionAngleVertical(pot=isopot, ro=ro, vo=vo)
    aAVnu = actionAngleVertical(pot=isopot)
    # __call__
    assert numpy.fabs(aAV(-0.1, 0.1) - aAVnu(-0.1, 0.1) * ro * vo) < 10.0**-8.0, (
        "actionAngle function __call__ does not return Quantity with the right value for actionAngleVertical"
    )
    # actionsFreqs
    assert (
        numpy.fabs(
            aAV.actionsFreqs(0.2, 0.1)[0] - aAVnu.actionsFreqs(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleVertical"
    )
    assert (
        numpy.fabs(
            aAV.actionsFreqs(0.2, 0.1)[1]
            - aAVnu.actionsFreqs(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleVertical"
    )
    # actionsFreqsAngles
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[0]
            - aAVnu.actionsFreqsAngles(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    )
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[1]
            - aAVnu.actionsFreqsAngles(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    )
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[2] - aAVnu.actionsFreqsAngles(0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), (
        "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    )
    return None


# Basic sanity checking of the actionAngleIsochrone actions
def test_actionAngleIsochrone_basic_actions():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit

    aAI = actionAngleIsochrone(b=1.2)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAI(R, vR, vT, z, vz)
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the isochrone potential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the isochrone potential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAI(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the isochrone potential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-4.0, (
        "Close-to-circular orbit in the isochrone potential does not have small Jz"
    )
    # Close-to-circular orbit, called with time
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAI(Orbit([R, vR, vT, z, vz]), 0.0)
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the isochrone potential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-4.0, (
        "Close-to-circular orbit in the isochrone potential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleIsochrone actions
def test_actionAngleIsochrone_basic_freqs():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    jos = aAI.actionsFreqs(R, vR, vT, z, vz)
    assert numpy.fabs((jos[3] - ip.epifreq(1.0)) / ip.epifreq(1.0)) < 10.0**-12.0, (
        "Circular orbit in the isochrone potential does not have Or=kappa at %g%%"
        % (100.0 * numpy.fabs((jos[3] - ip.epifreq(1.0)) / ip.epifreq(1.0)))
    )
    assert numpy.fabs((jos[4] - ip.omegac(1.0)) / ip.omegac(1.0)) < 10.0**-12.0, (
        "Circular orbit in the isochrone potential does not have Op=Omega at %g%%"
        % (100.0 * numpy.fabs((jos[4] - ip.omegac(1.0)) / ip.omegac(1.0)))
    )
    assert (
        numpy.fabs((jos[5] - ip.verticalfreq(1.0)) / ip.verticalfreq(1.0)) < 10.0**-12.0
    ), "Circular orbit in the isochrone potential does not have Oz=nu at %g%%" % (
        100.0 * numpy.fabs((jos[5] - ip.verticalfreq(1.0)) / ip.verticalfreq(1.0))
    )
    # close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.01, 1.01, 0.01, 0.01
    jos = aAI.actionsFreqs(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs((jos[3] - ip.epifreq(1.0)) / ip.epifreq(1.0)) < 10.0**-2.0, (
        "Close-to-circular orbit in the isochrone potential does not have Or=kappa at %g%%"
        % (100.0 * numpy.fabs((jos[3] - ip.epifreq(1.0)) / ip.epifreq(1.0)))
    )
    assert numpy.fabs((jos[4] - ip.omegac(1.0)) / ip.omegac(1.0)) < 10.0**-2.0, (
        "Close-to-circular orbit in the isochrone potential does not have Op=Omega at %g%%"
        % (100.0 * numpy.fabs((jos[4] - ip.omegac(1.0)) / ip.omegac(1.0)))
    )
    assert (
        numpy.fabs((jos[5] - ip.verticalfreq(1.0)) / ip.verticalfreq(1.0)) < 10.0**-2.0
    ), (
        "Close-to-circular orbit in the isochrone potential does not have Oz=nu at %g%%"
        % (100.0 * numpy.fabs((jos[5] - ip.verticalfreq(1.0)) / ip.verticalfreq(1.0)))
    )
    return None


# Test that EccZmaxRperiRap for an IsochronePotential are correctly computed
# by comparing to a numerical orbit integration
def test_actionAngleIsochrone_EccZmaxRperiRap_againstOrbit():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.03, 0.0])
    ecc, zmax, rperi, rap = aAI.EccZmaxRperiRap(o)
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, ip)
    assert numpy.fabs(ecc - o.e()) < 1e-10, (
        "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(zmax - o.zmax()) < 1e-5, (
        "Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(rperi - o.rperi()) < 1e-10, (
        "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(rap - o.rap()) < 1e-10, (
        "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
    )
    # Another one
    o = Orbit([1.0, 0.1, 1.1, 0.2, -0.3, 0.0])
    ecc, zmax, rperi, rap = aAI.EccZmaxRperiRap(
        o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()
    )
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, ip)
    assert numpy.fabs(ecc - o.e()) < 1e-10, (
        "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(zmax - o.zmax()) < 1e-3, (
        "Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(rperi - o.rperi()) < 1e-10, (
        "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(rap - o.rap()) < 1e-10, (
        "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
    )
    return None


# Test that EccZmaxRperiRap for an IsochronePotential are correctly computed
# by comparing to a numerical orbit integration for a Kepler potential
def test_actionAngleIsochrone_EccZmaxRperiRap_againstOrbit_kepler():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0)
    aAI = actionAngleIsochrone(ip=ip)
    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.03, 0.0])
    ecc, zmax, rperi, rap = aAI.EccZmaxRperiRap(o.R(), o.vR(), o.vT(), o.z(), o.vz())
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, ip)
    assert numpy.fabs(ecc - o.e()) < 1e-10, (
        "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    )
    # Don't do zmax, because zmax for Kepler is approximate
    assert numpy.fabs(rperi - o.rperi()) < 1e-10, (
        "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    )
    assert numpy.fabs(rap - o.rap()) < 1e-10, (
        "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
    )
    return None


# Test the actions of an actionAngleIsochrone
def test_actionAngleIsochrone_conserved_actions():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAI, obs, ip, -5.0, -5.0, -5.0)
    else:
        check_actionAngle_conserved_actions(aAI, obs, ip, -8.0, -8.0, -8.0)
    return None


# Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleIsochrone_linear_angles():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAI, obs, ip, -5.0, -5.0, -5.0, -6.0, -6.0, -6.0, -5.0, -5.0, -5.0
        )
    else:
        check_actionAngle_linear_angles(
            aAI, obs, ip, -6.0, -6.0, -6.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0
        )
    return None


# Test that the angles of an actionAngleIsochrone increase linearly for an
# orbit in the mid-plane (non-inclined; has potential issues, because the
# the ascending node is not well defined)
def test_actionAngleIsochrone_noninclinedorbit_linear_angles():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.0, 0.0, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAI, obs, ip, -5.0, -5.0, -5.0, -6.0, -6.0, -6.0, -5.0, -5.0, -5.0
        )
    else:
        check_actionAngle_linear_angles(
            aAI, obs, ip, -6.0, -6.0, -6.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0
        )
    return None


def test_actionAngleIsochrone_almostnoninclinedorbit_linear_angles():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    eps = 1e-10
    obs = Orbit([1.1, 0.3, 1.2, 0.0, eps, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAI, obs, ip, -5.0, -5.0, -5.0, -6.0, -6.0, -6.0, -5.0, -5.0, -5.0
        )
    else:
        check_actionAngle_linear_angles(
            aAI, obs, ip, -6.0, -6.0, -6.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0
        )
    return None


# Test that the Kelperian limit of the isochrone actions/angles works
def test_actionAngleIsochrone_kepler_actions():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0.0)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    times = numpy.linspace(0.0, 100.0, 101)
    obs.integrate(times, ip, method="dopr54_c")
    jrs, jps, jzs = aAI(
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    jc = ip._amp / numpy.sqrt(-2.0 * obs.E())
    L = numpy.sqrt(numpy.sum(obs.L() ** 2.0))
    # Jr = Jc-L
    assert numpy.all(numpy.fabs(jrs - (jc - L)) < 10.0**-5.0), (
        "Radial action for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(jps - obs.R() * obs.vT()) < 10.0**-10.0), (
        "Azimuthal action for the Kepler potential not correct"
    )
    assert numpy.all(
        numpy.fabs(jzs - (L - numpy.fabs(obs.R() * obs.vT()))) < 10.0**-10.0
    ), "Vertical action for the Kepler potential not correct"
    return None


def test_actionAngleIsochrone_kepler_freqs():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0.0)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    times = numpy.linspace(0.0, 100.0, 101)
    obs.integrate(times, ip, method="dopr54_c")
    _, _, _, ors, ops, ozs = aAI.actionsFreqs(
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    jc = ip._amp / numpy.sqrt(-2.0 * obs.E())
    oc = ip._amp**2.0 / jc**3.0  # (BT08 eqn. E4)
    assert numpy.all(numpy.fabs(ors - oc) < 10.0**-10.0), (
        "Radial frequency for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(ops - oc) < 10.0**-10.0), (
        "Azimuthal frequency for the Kepler potential not correct"
    )
    assert numpy.all(
        numpy.fabs(ozs - numpy.sign(obs.R() * obs.vT()) * oc) < 10.0**-10.0
    ), "Vertical frequency for the Kepler potential not correct"
    return None


def test_actionAngleIsochrone_kepler_angles():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0.0)
    aAI = actionAngleIsochrone(ip=ip)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    times = numpy.linspace(0.0, 100.0, 101)
    obs.integrate(times, ip, method="dopr54_c")
    _, _, _, _, _, _, ars, aps, azs = aAI.actionsFreqsAngles(
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    jc = ip._amp / numpy.sqrt(-2.0 * obs.E())
    oc = ip._amp**2.0 / jc**3.0  # (BT08 eqn. E4)
    # theta_r = Or x times + theta_r,0
    assert numpy.all(numpy.fabs(ars - oc * times - ars[0]) < 10.0**-10.0), (
        "Radial angle for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(aps - oc * times - aps[0]) < 10.0**-10.0), (
        "Azimuthal angle for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(azs - oc * times - azs[0]) < 10.0**-10.0), (
        "Vertical angle for the Kepler potential not correct"
    )
    return None


# Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_actions():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(Orbit([R, vR, vT]))
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the spherical LogarithmicHaloPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the spherical LogarithmicHaloPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-4.0, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_freqs():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import CompositePotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=CompositePotential([lp]))
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    jos = aAS.actionsFreqs(R, vR, vT, z, vz)
    assert numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)) < 10.0**-12.0, (
        "Circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%"
        % (100.0 * numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)))
    )
    assert numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)) < 10.0**-12.0, (
        "Circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%"
        % (100.0 * numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)))
    )
    assert (
        numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)) < 10.0**-12.0
    ), (
        "Circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%"
        % (100.0 * numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)))
    )
    # close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.01, 1.01, 0.01, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)) < 10.0**-1.9, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%"
        % (100.0 * numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)))
    )
    assert numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)) < 10.0**-1.9, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%"
        % (100.0 * numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)))
    )
    assert (
        numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)) < 10.0**-1.9
    ), (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%"
        % (100.0 * numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)))
    )


# Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_freqsAngles():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    # v. close-to-circular orbit using actionsFreqsAngles
    R, vR, vT, z, vz = 1.0, 10.0**-8.0, 1.0, 10.0**-8.0, 0.0
    jos = aAS.actionsFreqsAngles(R, vR, vT, z, vz, 0.0)
    assert numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)) < 10.0**-1.9, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Or=kappa at %g%%"
        % (100.0 * numpy.fabs((jos[3] - lp.epifreq(1.0)) / lp.epifreq(1.0)))
    )
    assert numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)) < 10.0**-1.9, (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Op=Omega at %g%%"
        % (100.0 * numpy.fabs((jos[4] - lp.omegac(1.0)) / lp.omegac(1.0)))
    )
    assert (
        numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)) < 10.0**-1.9
    ), (
        "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have Oz=nu at %g%%"
        % (100.0 * numpy.fabs((jos[5] - lp.verticalfreq(1.0)) / lp.verticalfreq(1.0)))
    )
    return None


# Test that EccZmaxRperiRap for a spherical potential are correctly computed
# by comparing to a numerical orbit integration
def test_actionAngleSpherical_EccZmaxRperiRap_againstOrbit():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.03, 0.0])
    ecc, zmax, rperi, rap = aAS.EccZmaxRperiRap(o)
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, lp)
    assert numpy.fabs(ecc - o.e()) < 1e-9, (
        "Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(zmax - o.zmax()) < 1e-4, (
        "Analytically calculated zmax does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(rperi - o.rperi()) < 1e-8, (
        "Analytically calculated rperi does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(rap - o.rap()) < 1e-8, (
        "Analytically calculated rap does not agree with numerically calculated one for a spherical potential"
    )
    # Another one
    o = Orbit([1.0, 0.1, 1.1, 0.2, -0.3, 0.0])
    ecc, zmax, rperi, rap = aAS.EccZmaxRperiRap(o.R(), o.vR(), o.vT(), o.z(), o.vz())
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, lp)
    assert numpy.fabs(ecc - o.e()) < 1e-9, (
        "Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(zmax - o.zmax()) < 1e-3, (
        "Analytically calculated zmax does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(rperi - o.rperi()) < 1e-8, (
        "Analytically calculated rperi does not agree with numerically calculated one for a spherical potential"
    )
    assert numpy.fabs(rap - o.rap()) < 1e-8, (
        "Analytically calculated rap does not agree with numerically calculated one for a spherical potential"
    )
    return None


# Test the actions of an actionAngleSpherical
def test_actionAngleSpherical_conserved_actions():
    from galpy import potential
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(aAS, obs, lp, -5.0, -5.0, -5.0, ntimes=101)
    else:
        check_actionAngle_conserved_actions(aAS, obs, lp, -8.0, -8.0, -8.0, ntimes=101)
    return None


# Test the actions of an actionAngleSpherical
def test_actionAngleSpherical_conserved_actions_fixed_quad():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(
            aAS, obs, lp, -5.0, -5.0, -5.0, ntimes=101, fixed_quad=True
        )
    else:
        check_actionAngle_conserved_actions(
            aAS, obs, lp, -8.0, -8.0, -8.0, ntimes=101, fixed_quad=True
        )
    return None


# Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleSpherical_linear_angles():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            ntimes=501,
        )  # need fine sampling for de-period
    else:
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -6.0,
            -6.0,
            -6.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            ntimes=501,
        )  # need fine sampling for de-period
    return None


# Test that the angles of an actionAngleIsochrone increase linearly
def test_actionAngleSpherical_linear_angles_fixed_quad():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            ntimes=501,  # need fine sampling for de-period
            fixed_quad=True,
        )
    else:
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -6.0,
            -6.0,
            -6.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            ntimes=501,  # need fine sampling for de-period
            fixed_quad=True,
        )
    return None


# Test that the angles of an actionAngleSpherical increase linearly for an
# orbit in the mid-plane (non-inclined; has potential issues, because the
# the ascending node is not well defined)
def test_actionAngleSpherical_noninclinedorbit_linear_angles():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    obs = Orbit([1.1, 0.3, 1.2, 0.0, 0.0, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            ntimes=501,
        )  # need fine sampling for de-period
    else:
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -6.0,
            -6.0,
            -6.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            ntimes=501,
        )  # need fine sampling for de-period
    return None


def test_actionAngleSpherical_almostnoninclinedorbit_linear_angles():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=lp)
    eps = 1e-10
    obs = Orbit([1.1, 0.3, 1.2, 0.0, eps, 2.0])
    from galpy.orbit.Orbits import ext_loaded

    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            -4.0,
            ntimes=501,
        )  # need fine sampling for de-period
    else:
        check_actionAngle_linear_angles(
            aAS,
            obs,
            lp,
            -6.0,
            -6.0,
            -6.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            -8.0,
            ntimes=501,
        )  # need fine sampling for de-period
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleSpherical
def test_actionAngleSpherical_conserved_EccZmaxRperiRap_ecc():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import NFWPotential

    np = NFWPotential(normalize=1.0, a=2.0)
    aAS = actionAngleSpherical(pot=np)
    obs = Orbit([1.1, 0.2, 1.3, 0.1, 0.0, 2.0])
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAS, obs, np, -1.1, -0.4, -1.8, -1.8, ntimes=101, inclphi=True
    )
    return None


# Test the actionAngleSpherical against an isochrone potential: actions
def test_actionAngleSpherical_otherIsochrone_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleSpherical(pot=ip)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAS(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-10.0, (
        "actionAngleSpherical applied to isochrone potential fails for Jr at %g%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleSpherical applied to isochrone potential fails for Lz at %g%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-10.0, (
        "actionAngleSpherical applied to isochrone potential fails for Jz at %g%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleSpherical against an isochrone potential: frequencies
def test_actionAngleSpherical_otherIsochrone_freqs():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleSpherical(pot=ip)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqs(R, vR, vT, z, vz, phi)
    jiaO = aAS.actionsFreqs(R, vR, vT, z, vz, phi)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Or at %g%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Op at %g%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Oz at %g%%"
        % (dOz * 100.0)
    )
    return None


# Test the actionAngleSpherical against an isochrone potential: frequencies
def test_actionAngleSpherical_otherIsochrone_freqs_fixed_quad():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleSpherical(pot=ip)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqs(R, vR, vT, z, vz, phi)
    jiaO = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, phi]), fixed_quad=True)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Or at %g%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Op at %g%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for Oz at %g%%"
        % (dOz * 100.0)
    )
    return None


# Test the actionAngleSpherical against an isochrone potential: angles
def test_actionAngleSpherical_otherIsochrone_angles():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleSpherical(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    jiaO = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for ar at %g%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for ap at %g%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-6.0, (
        "actionAngleSpherical applied to isochrone potential fails for az at %g%%"
        % (daz * 100.0)
    )
    return None


# Test that actionAngleSpherical works at small r
# Test that the adiabatic approximation, which sends the spherical code a
# planar point with an angular momentum increased by gamma J_z, handles a
# circular planar orbit with vertical motion: the point is a turning point of
# the modified radial problem but not a circular orbit of it
def test_actionAngleAdiabatic_circular_planar_gamma():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import HernquistPotential

    hp = HernquistPotential(normalize=1.0)
    aAA = actionAngleAdiabatic(pot=hp, gamma=1.0, c=False)
    ecc, zmax, rperi, rap = aAA.EccZmaxRperiRap(1.0, 0.0, 1.0, 0.05, 0.03)
    assert numpy.all(numpy.isfinite([ecc, zmax, rperi, rap])), (
        "Adiabatic EccZmaxRperiRap of a circular planar orbit with vertical motion is not finite"
    )
    assert rperi <= 1.0 <= rap, (
        "Adiabatic EccZmaxRperiRap of a circular planar orbit with vertical motion does not bracket its radius"
    )
    assert rap > rperi + 1e-4, (
        "The adiabatic approximation's modified radial problem for a circular planar orbit with vertical motion is not a libration"
    )
    jr = aAA(1.0, 0.0, 1.0, 0.05, 0.03)[0]
    assert numpy.isfinite(jr) and jr > 0.0, (
        "Adiabatic radial action of a circular planar orbit with vertical motion is not finite and positive"
    )
    return None


# Test that actionAngleSpherical's near-circular path, which solves the radial
# problem relative to the circular orbit, keeps the exact isochrone's accuracy
# at small radii and small amplitudes, where the plain energy difference has
# no digits left: frequencies and angles to 1e-7 down to the hand-over to the
# epicycle, the exactly circular orbit at a tiny radius included
def test_actionAngleSpherical_near_circular_small_radius():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential, epifreq, vcirc

    def wrap(x):
        return numpy.fabs(((x + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi)

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAS = actionAngleSpherical(pot=ip)
    aAI = actionAngleIsochrone(ip=ip)
    for rc in (0.1, 1e-2, 1e-4):
        kappa = epifreq(ip, rc, use_physical=False)
        vc = vcirc(ip, rc, use_physical=False)
        for wrc in (1e-2, 1e-3, 1.01e-4, 1e-5):
            w = wrc * rc
            # mid-flight and a third of the way to apocentre
            for r, vr in (
                (rc, w * kappa),
                (
                    rc + w * numpy.cos(numpy.pi / 3.0),
                    -w * kappa * numpy.sin(numpy.pi / 3.0),
                ),
            ):
                args = (r, vr, rc * vc / r, 0.0, 0.0, 0.7)
                f = aAS.actionsFreqsAngles(*args)
                g = aAI.actionsFreqsAngles(*args)
                assert numpy.all(numpy.isfinite(numpy.array([x[0] for x in f]))), (
                    "actionAngleSpherical is not finite at rc={}, w/rc={}".format(
                        rc, wrc
                    )
                )
                assert numpy.fabs(f[3][0] / g[3][0] - 1.0) < 1e-7, (
                    "Omega_r at rc={}, w/rc={} is off by {:g}".format(
                        rc, wrc, f[3][0] / g[3][0] - 1.0
                    )
                )
                assert numpy.fabs(f[4][0] / g[4][0] - 1.0) < 1e-7, (
                    "Omega_phi at rc={}, w/rc={} is off by {:g}".format(
                        rc, wrc, f[4][0] / g[4][0] - 1.0
                    )
                )
                # the isochrone's own action and angles lose their digits at
                # small radii; compare where they have them
                if g[0][0] > 1e-9:
                    assert numpy.fabs(f[0][0] / g[0][0] - 1.0) < 1e-6, (
                        "J_r at rc={}, w/rc={} is off by {:g}".format(
                            rc, wrc, f[0][0] / g[0][0] - 1.0
                        )
                    )
                    assert wrap(f[6][0] - g[6][0]) < 1e-7, (
                        "theta_r at rc={}, w/rc={} is off by {:g}".format(
                            rc, wrc, wrap(f[6][0] - g[6][0])
                        )
                    )
                    assert wrap(f[8][0] - g[8][0]) < 1e-7, (
                        "theta_z at rc={}, w/rc={} is off by {:g}".format(
                            rc, wrc, wrap(f[8][0] - g[8][0])
                        )
                    )
    # the same through the planar potential
    from galpy.potential import toPlanarPotential

    aAP = actionAngleSpherical(pot=toPlanarPotential(ip))
    rc = 0.1
    kappa = epifreq(ip, rc, use_physical=False)
    vc = vcirc(ip, rc, use_physical=False)
    args = (rc, 1e-3 * rc * kappa, vc, 0.0, 0.0, 0.7)
    f = aAS.actionsFreqsAngles(*args)
    g = aAP.actionsFreqsAngles(*args)
    for ii in (0, 3, 4, 6):
        assert numpy.fabs(f[ii][0] - g[ii][0]) < 1e-12, (
            "The near-circular path through a planar potential differs from the three-dimensional one"
        )
    # the exactly circular orbit at a tiny radius, where the energy above the
    # circular orbit's is pure round-off
    r = 1e-6
    f = aAS.actionsFreqsAngles(r, 0.0, vcirc(ip, r, use_physical=False), 0.0, 0.0, 0.7)
    g = aAI.actionsFreqs(r, 0.0, vcirc(ip, r, use_physical=False), 0.0, 0.0)
    assert f[0][0] == 0.0, "J_r of a circular orbit at r=1e-6 is not zero"
    assert numpy.all(numpy.isfinite(numpy.array([x[0] for x in f]))), (
        "actionAngleSpherical is not finite for a circular orbit at r=1e-6"
    )
    assert (
        numpy.fabs(f[3][0] / g[3][0] - 1.0) < 1e-10
        and numpy.fabs(f[4][0] / g[4][0] - 1.0) < 1e-10
    ), "The frequencies of a circular orbit at r=1e-6 are not the epicycle's"
    return None


# Test that the circular-orbit machinery is only reached by orbits close to
# circular: a nearly radial orbit in a potential whose circular speed is not
# finite near the centre, and an eccentric orbit in a potential that has forces
# but no second derivatives, both evaluate as before; and a near-circular orbit
# in the latter gets its epicycle frequency from the force's finite difference
def test_actionAngleSpherical_screen():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import (
        HernquistPotential,
        NFWPotential,
        Potential,
        epifreq,
        vcirc,
    )

    # nearly radial in an NFW potential: the circular radius of L=1e-12 has a
    # NaN circular speed, which must never be looked up
    aAS = actionAngleSpherical(pot=NFWPotential(normalize=1.0))
    jr = aAS(1.0, 0.2, 1e-12, 0.0, 0.0)[0]
    jr0 = aAS(1.0, 0.2, 0.0, 0.0, 0.0)[0]
    assert numpy.isfinite(jr) and numpy.fabs(jr - jr0) < 1e-8, (
        "A nearly radial NFW orbit's J_r is not finite or not the radial orbit's"
    )
    assert numpy.all(numpy.isfinite(aAS.EccZmaxRperiRap(1.0, 0.2, 1e-12, 0.0, 0.0))), (
        "A nearly radial NFW orbit's turning points are not finite"
    )

    class ForceOnlyHernquist(Potential):
        def _evaluate(self, R, z, phi=0.0, t=0.0):
            return -0.5 / (1.0 + numpy.hypot(R, z))

        def _Rforce(self, R, z, phi=0.0, t=0.0):
            r = numpy.hypot(R, z)
            return -0.5 * R / r / (1.0 + r) ** 2

        def _zforce(self, R, z, phi=0.0, t=0.0):
            r = numpy.hypot(R, z)
            return -0.5 * z / r / (1.0 + r) ** 2

    fpot = ForceOnlyHernquist()
    hp = HernquistPotential(amp=1.0, a=1.0)
    aAF = actionAngleSpherical(pot=fpot)
    aAH = actionAngleSpherical(pot=hp)
    # eccentric: as with the full potential, no second derivative needed
    assert (
        numpy.fabs(aAF(1.0, 0.1, 0.2, 0.0, 0.0)[0] - aAH(1.0, 0.1, 0.2, 0.0, 0.0)[0])
        < 1e-10
    ), (
        "An eccentric orbit's J_r in a force-only potential differs from the full potential's"
    )
    # near-circular: the epicycle frequency from the force's finite difference
    vc = vcirc(hp, 1.0, use_physical=False)
    ff = aAF.actionsFreqs(1.0, 1e-5 * vc, vc, 0.0, 0.0)
    fh = aAH.actionsFreqs(1.0, 1e-5 * vc, vc, 0.0, 0.0)
    assert numpy.fabs(ff[3][0] / epifreq(hp, 1.0, use_physical=False) - 1.0) < 1e-7, (
        "A near-circular orbit's Omega_r in a force-only potential is not the epicycle frequency"
    )
    assert numpy.fabs(ff[0][0] / fh[0][0] - 1.0) < 1e-6, (
        "A near-circular orbit's J_r in a force-only potential differs from the full potential's"
    )
    return None


def _near_circular_point(pot, rc, wrc, phase):
    """A point of the epicycle of relative half-width wrc around the circular
    orbit at rc, at the given radial phase: (r, v_r, v_t)"""
    from galpy.potential import epifreq, vcirc

    w = wrc * rc
    r = rc - w * numpy.cos(phase)
    return (
        r,
        w * epifreq(pot, rc, use_physical=False) * numpy.sin(phase),
        rc * vcirc(pot, rc, use_physical=False) / r,
    )


# Test actionAngleSpherical across the hand-over to the epicycle at a harmonic
# half-width of 1e-5 of the circular radius, at several radial phases and radii
# and in several potentials:
# everything finite, the frequencies within the epicycle's own accuracy of the
# epicycle and circular frequencies, the radial angle within the epicycle's own
# accuracy of the phase, and no jump across either hand-over
def test_actionAngleSpherical_epicycle_handover():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import (
        BurkertPotential,
        HernquistPotential,
        IsochronePotential,
        NFWPotential,
        epifreq,
        omegac,
    )

    def wrap(x):
        return numpy.fabs(((x + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi)

    pots = [
        IsochronePotential(normalize=1.0, b=1.2),
        HernquistPotential(normalize=1.0),
        NFWPotential(normalize=1.0),
        BurkertPotential(normalize=1.0),
    ]
    phases = [0.1, 0.5 * numpy.pi, numpy.pi - 0.16, 1.9 * numpy.pi]
    for pot in pots:
        aAS = actionAngleSpherical(pot=pot)
        for rc in (1.0, 0.1):
            kappa = epifreq(pot, rc, use_physical=False)
            Omc = omegac(pot, rc, use_physical=False)
            for wrc in (3e-5, 1.01e-5, 0.99e-5, 3e-6, 3e-7):
                for phase in phases:
                    r, vr, vt = _near_circular_point(pot, rc, wrc, phase)
                    f = aAS.actionsFreqsAngles(r, vr, vt, 0.0, 0.0, 0.7)
                    assert numpy.all(numpy.isfinite(numpy.array([x[0] for x in f]))), (
                        "actionAngleSpherical is not finite for {} at rc={}, w/rc={}, phase={}".format(
                            type(pot).__name__, rc, wrc, phase
                        )
                    )
                    # the quadratures' accuracy is bounded by the potential's
                    # own round-off in the force, amplified next to the turning
                    # points as the libration shrinks (the Burkert force loses
                    # four digits at a tenth of its scale radius); the
                    # epicycle's by (w/rc)^2
                    noisy = isinstance(pot, BurkertPotential)
                    tol = 1e-4 if noisy else 1e-7
                    assert numpy.fabs(f[3][0] / kappa - 1.0) < tol, (
                        "Omega_r for {} at rc={}, w/rc={}, phase={} is off from kappa by {:g}".format(
                            type(pot).__name__, rc, wrc, phase, f[3][0] / kappa - 1.0
                        )
                    )
                    assert numpy.fabs(f[4][0] / Omc - 1.0) < tol, (
                        "Omega_phi for {} at rc={}, w/rc={}, phase={} is off from Omega_c by {:g}".format(
                            type(pot).__name__, rc, wrc, phase, f[4][0] / Omc - 1.0
                        )
                    )
                    assert wrap(f[6][0] - phase) < 10.0 * wrc + 3e-4 * noisy, (
                        "theta_r for {} at rc={}, w/rc={}, phase={} is off from the phase by {:g}".format(
                            type(pot).__name__, rc, wrc, phase, wrap(f[6][0] - phase)
                        )
                    )
            # no jump across the hand-over: the two sides of the threshold
            # agree to what the epicycle's approximation allows
            for wlo, whi in ((0.99e-5, 1.01e-5),):
                for phase in phases:
                    flo = aAS.actionsFreqsAngles(
                        *_near_circular_point(pot, rc, wlo, phase), 0.0, 0.0, 0.7
                    )
                    fhi = aAS.actionsFreqsAngles(
                        *_near_circular_point(pot, rc, whi, phase), 0.0, 0.0, 0.7
                    )
                    noisy = isinstance(pot, BurkertPotential)
                    tol = 1e-4 if noisy else 1e-7
                    assert (
                        numpy.fabs(flo[0][0] / fhi[0][0] - (wlo / whi) ** 2)
                        < 0.1 * (whi - wlo) / whi + tol
                    ), "J_r jumps across the hand-over at w/rc={} for {}".format(
                        whi, type(pot).__name__
                    )
                    assert (
                        numpy.fabs(flo[3][0] / fhi[3][0] - 1.0) < tol
                        and numpy.fabs(flo[4][0] / fhi[4][0] - 1.0) < tol
                    ), (
                        "The frequencies jump across the hand-over at w/rc={} for {}".format(
                            whi, type(pot).__name__
                        )
                    )
                    assert (
                        wrap(flo[6][0] - fhi[6][0]) < 10.0 * whi + 3e-4 * noisy
                        and wrap(flo[8][0] - fhi[8][0]) < 10.0 * whi + 3e-4 * noisy
                    ), "The angles jump across the hand-over at w/rc={} for {}".format(
                        whi, type(pot).__name__
                    )
    return None


# Test that the adiabatic approximation, which sends the spherical code a
# planar point with an angular momentum raised by gamma J_z, handles
# near-circular orbits at small radii, where a fixed probe of the radial
# equation's sign next to the turning point used to overshoot the whole
# libration and declare the orbit unbound
def test_actionAngleAdiabatic_near_circular_small_radius():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import (
        HernquistPotential,
        IsochronePotential,
        MiyamotoNagaiPotential,
        vcirc,
    )

    for pot in (
        IsochronePotential(normalize=1.0),
        HernquistPotential(normalize=1.0),
        MiyamotoNagaiPotential(normalize=1.0),
    ):
        aAA = actionAngleAdiabatic(pot=pot, c=False)
        for r in (1e-4, 1e-6):
            vc = vcirc(pot, r, use_physical=False)
            for vz in (0.0, 1e-3 * vc):
                ecc, zmax, rperi, rap = aAA.EccZmaxRperiRap(
                    r, 0.0, 1.00001 * vc, 0.0, vz
                )
                assert numpy.all(numpy.isfinite([ecc, zmax, rperi, rap])), (
                    "Adiabatic EccZmaxRperiRap is not finite for a near-circular orbit at r={} in {}".format(
                        r, type(pot).__name__
                    )
                )
                assert rperi <= r <= rap and rap > rperi, (
                    "Adiabatic turning points do not bracket a near-circular orbit at r={} in {}".format(
                        r, type(pot).__name__
                    )
                )
                assert numpy.all(numpy.isfinite(aAA(r, 0.0, 1.00001 * vc, 0.0, vz))), (
                    "Adiabatic actions are not finite for a near-circular orbit at r={} in {}".format(
                        r, type(pot).__name__
                    )
                )
    return None


# Test that small orbits far from circular, whose energy above the potential's
# is a tiny fraction of the potential's, keep the exact isochrone's accuracy:
# the radial frequency of radial orbits down to r=1e-6 against its closed form
# (-2E)^(3/2) / GM, and small eccentric orbits, which are solved relative to
# the point itself; also a small eccentric Kepler orbit near its apocentre,
# where the turning-point brackets must stay inside the allowed interval
def test_actionAngleSpherical_small_orbits():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import IsochronePotential, KeplerPotential, vcirc

    def wrap(x):
        return numpy.fabs(((x + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi)

    GM, b = 2.0, 1.2
    ip = IsochronePotential(amp=GM, b=b)
    aAS = actionAngleSpherical(pot=ip)
    aAI = actionAngleIsochrone(ip=ip)
    for r in (1e-6, 1e-4, 1e-2):
        for vr in (0.0, 0.3 * numpy.sqrt(-2.0 * ip(r, 0.0))):
            E = 0.5 * vr**2.0 + ip(r, 0.0)
            Or_true = (-2.0 * E) ** 1.5 / GM
            f = aAS.actionsFreqs(r, vr, 0.0, 0.0, 0.0)
            assert numpy.fabs(f[3][0] / Or_true - 1.0) < 1e-8, (
                "Omega_r of a radial isochrone orbit at r={}, vr={} is off by {:g}".format(
                    r, vr, f[3][0] / Or_true - 1.0
                )
            )
    for r, vrf, vtf in (
        (1e-3, 0.3, 0.7),
        (1e-4, 0.5, 0.5),
        (1e-2, 0.2, 0.9),
        (1e-6, 0.0, 0.5),
    ):
        vc = vcirc(ip, r, use_physical=False)
        args = (r, vrf * vc, vtf * vc, 0.0, 0.0, 0.7)
        f = aAS.actionsFreqsAngles(*args)
        g = aAI.actionsFreqsAngles(*args)
        assert (
            numpy.fabs(f[3][0] / g[3][0] - 1.0) < 1e-8
            and numpy.fabs(f[4][0] / g[4][0] - 1.0) < 1e-8
        ), (
            "The frequencies of a small eccentric isochrone orbit at r={} are off".format(
                r
            )
        )
        # the isochrone's own action and angles lose their digits at the
        # smallest radii
        if g[0][0] > 1e-9:
            assert numpy.fabs(f[0][0] / g[0][0] - 1.0) < 1e-6, (
                "J_r of a small eccentric isochrone orbit at r={} is off by {:g}".format(
                    r, f[0][0] / g[0][0] - 1.0
                )
            )
            assert wrap(f[6][0] - g[6][0]) < 1e-6 and wrap(f[8][0] - g[8][0]) < 1e-6, (
                "The angles of a small eccentric isochrone orbit at r={} are off".format(
                    r
                )
            )
    # nearly radial: the azimuthal frequency's quadrature must converge (it
    # was capped short of convergence once), and the pericentre of a tiny
    # angular momentum must be found rather than replaced by the centre; in
    # a harmonic potential every orbit has (Omega_r, Omega_phi) = (2, 1)
    from galpy.potential import PowerSphericalPotential

    ipn = IsochronePotential(normalize=1.0, b=1.2)
    aASn = actionAngleSpherical(pot=ipn)
    aAIn = actionAngleIsochrone(ip=ipn)
    args = (1e-3, 0.0, 3e-4 * vcirc(ipn, 1e-3, use_physical=False), 0.0, 0.0)
    assert (
        numpy.fabs(
            aASn.actionsFreqs(*args)[4][0] / aAIn.actionsFreqs(*args)[4][0] - 1.0
        )
        < 1e-7
    ), "Omega_phi of a small nearly radial isochrone orbit has not converged"
    hp = PowerSphericalPotential(alpha=0.0, normalize=1.0)
    fh = actionAngleSpherical(pot=hp).actionsFreqs(1e-6, 0.0, 1e-12, 0.0, 0.0)
    assert (
        numpy.fabs(fh[3][0] / 2.0 - 1.0) < 1e-7
        and numpy.fabs(fh[4][0] - 1.0) < 1e-7
        and numpy.fabs(fh[5][0] - 1.0) < 1e-7
    ), "The frequencies of a tiny nearly radial harmonic orbit are not (2, 1, 1)"
    # a small Kepler orbit near apocentre: J_r = sqrt(GM / -2E) - L
    kp = KeplerPotential(amp=1.0)
    aAK = actionAngleSpherical(pot=kp)
    R, vR, vT = 1e-6, 1e-5, 300.0
    jr = aAK(R, vR, vT, 0.0, 0.0)[0]
    E = 0.5 * (vR**2.0 + vT**2.0) - 1.0 / R
    assert numpy.fabs(jr / (1.0 / numpy.sqrt(-2.0 * E) - R * vT) - 1.0) < 1e-8, (
        "J_r of a small eccentric Kepler orbit near apocentre is off"
    )
    return None


# Test actionAngleSpherical's near-circular path outside the isochrone, against
# an integrated orbit: along the orbit the actions are constant, the frequencies
# are constant, and the angles advance linearly at the frequencies
def test_actionAngleSpherical_near_circular_integrated_orbit():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential, epifreq

    def wrap(x):
        return ((x + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi

    hp = HernquistPotential(normalize=1.0)
    aAS = actionAngleSpherical(pot=hp)
    rc = 0.7
    ts = numpy.linspace(
        0.0, 3.0 * 2.0 * numpy.pi / epifreq(hp, rc, use_physical=False), 61
    )
    for wrc, tolJ, tolO, tolA in ((1e-3, 1e-6, 1e-7, 1e-6), (3e-6, 1e-3, 1e-7, 3e-5)):
        r, vr, vt = _near_circular_point(hp, rc, wrc, 0.3)
        o = Orbit([r, vr, vt, 0.0, 0.0, 0.7])
        o.integrate(ts, hp, method="dop853_c", rtol=1e-14, atol=1e-14)
        f = aAS.actionsFreqsAngles(
            o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts)
        )
        assert numpy.all(numpy.isfinite(numpy.array(f))), (
            "Angles along an integrated near-circular orbit are not finite at w/rc={}".format(
                wrc
            )
        )
        assert numpy.std(f[0]) / numpy.mean(f[0]) < tolJ, (
            "J_r is not constant along an integrated near-circular orbit at w/rc={}: {:g}".format(
                wrc, numpy.std(f[0]) / numpy.mean(f[0])
            )
        )
        for ii in (3, 4):
            assert numpy.std(f[ii]) / numpy.mean(f[ii]) < tolO, (
                "The frequencies are not constant along an integrated near-circular orbit at w/rc={}".format(
                    wrc
                )
            )
        for ia, io in ((6, 3), (7, 4)):
            resid = wrap(f[ia] - f[ia][0] - numpy.mean(f[io]) * ts)
            assert numpy.max(numpy.fabs(resid)) < tolA, (
                "Angle {} does not advance linearly along an integrated near-circular orbit at w/rc={}: {:g}".format(
                    ia, wrc, numpy.max(numpy.fabs(resid))
                )
            )
    return None


# Test that actionAngleSpherical handles exactly radial orbits (L = 0), which
# have no circular orbit to be an epicycle around, against the isochrone's
# closed forms J_r = GM / sqrt(-2E) - sqrt(GM b) and Omega_r = (-2E)^(3/2) / GM,
# mid-flight and at apocentre, where the pericentre is the centre
def test_actionAngleSpherical_radial():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import IsochronePotential

    GM, b = 2.0, 1.0
    ip = IsochronePotential(amp=GM, b=b)
    aAS = actionAngleSpherical(pot=ip)
    for R, vR in [(0.5, 0.3), (0.5, 0.0), (1.3, 0.0)]:
        E = 0.5 * vR**2.0 + ip(R, 0.0)
        jr_true = GM / numpy.sqrt(-2.0 * E) - numpy.sqrt(GM * b)
        Or_true = (-2.0 * E) ** 1.5 / GM
        jr, Lz, jz = aAS(R, vR, 0.0, 0.0, 0.0)
        assert numpy.fabs(jr - jr_true) < 1e-10, (
            "actionAngleSpherical J_r of a radial isochrone orbit is off"
        )
        assert Lz == 0.0 and jz == 0.0, (
            "actionAngleSpherical L_z, J_z of a radial orbit are not zero"
        )
        jr, _, _, Or, _, _ = aAS.actionsFreqs(R, vR, 0.0, 0.0, 0.0)
        assert numpy.fabs(jr - jr_true) < 1e-10, (
            "actionAngleSpherical J_r of a radial isochrone orbit is off"
        )
        assert numpy.fabs(Or - Or_true) < 1e-8, (
            "actionAngleSpherical Omega_r of a radial isochrone orbit is off"
        )
    # the angles of a radial orbit at the same radius on the way out and on
    # the way in: finite, and symmetric about the apocentre (the radial
    # angle advances from the pericentre, the centre, where the azimuthal
    # integral has nothing to sweep)
    vr = 0.9 * numpy.sqrt(-2.0 * ip(0.1, 0.0))
    fout = aAS.actionsFreqsAngles(0.1, vr, 0.0, 0.0, 0.0, 0.7)
    fin = aAS.actionsFreqsAngles(0.1, -vr, 0.0, 0.0, 0.0, 0.7)
    assert numpy.isfinite(fout[6][0]) and numpy.isfinite(fin[6][0]), (
        "actionAngleSpherical radial angle of a radial orbit is not finite"
    )
    assert 0.0 < fout[6][0] < numpy.pi < fin[6][0] < 2.0 * numpy.pi, (
        "actionAngleSpherical radial angle of a radial orbit is not on the right side of the apocentre"
    )
    assert numpy.fabs(fout[6][0] + fin[6][0] - 2.0 * numpy.pi) < 1e-10, (
        "actionAngleSpherical radial angles of a radial orbit are not symmetric about the apocentre"
    )
    # at apocentre, the pericentre is the centre
    ecc, zmax, rperi, rap = aAS.EccZmaxRperiRap(0.5, 0.0, 0.0, 0.0, 0.0)
    assert ecc == 1.0 and rperi == 0.0 and numpy.fabs(rap - 0.5) < 1e-14, (
        "actionAngleSpherical turning points of a radial orbit at apocentre are off"
    )
    # continuity with a nearly radial orbit at apocentre
    jr0 = aAS(0.5, 0.0, 0.0, 0.0, 0.0)[0]
    jr1 = aAS(0.5, 0.0, 1e-12, 0.0, 0.0)[0]
    assert numpy.fabs(jr0 - jr1) < 1e-10, (
        "actionAngleSpherical J_r is discontinuous between a radial and a nearly radial orbit"
    )
    return None


def test_actionAngleSpherical_far_apocentre():
    # A bound orbit whose apocentre lies far beyond its current radius (here
    # e = 0.993 with the apocentre at 245, in a Hernquist potential with
    # a = 0.5): the search for the apocentre used to give up at a fixed
    # radius of 100 and declare the orbit unbound. Whether an orbit is bound
    # is now decided by the radial equation at infinity (or at a large
    # radius, for a potential that cannot be evaluated at infinity)
    from scipy import integrate, optimize

    from galpy.actionAngle import UnboundError, actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential, evaluatePotentials, rl

    def check(pot):
        aAS = actionAngleSpherical(pot=pot)
        o = Orbit([1.0, 1.265, 0.9, 0.9, 0.1, 0.0])
        E, L = o.E(pot=pot), numpy.sqrt(numpy.sum(o.L() ** 2.0))
        assert E < 0.0, "The test orbit is not bound"
        jr, jphi, jz, Or, Op, Oz, ar, ap, az = aAS.actionsFreqsAngles(
            o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()
        )

        # the turning points and the radial action by direct quadrature
        def pr2(r):
            return 2.0 * (E - evaluatePotentials(pot, r, 0.0)) - L**2.0 / r**2.0

        rc = rl(pot, L)
        rperi = optimize.brentq(pr2, 1e-6, rc, xtol=1e-14)
        rap = optimize.brentq(pr2, rc, 1e6, xtol=1e-12)
        assert rap > 200.0, "The test orbit's apocentre is not far"
        jr_quad = (
            integrate.quad(
                lambda r: numpy.sqrt(numpy.fabs(pr2(r))), rperi, rap, limit=200
            )[0]
            / numpy.pi
        )
        assert numpy.fabs(jr[0] / jr_quad - 1.0) < 1e-8, (
            "The radial action of a bound orbit with a far apocentre is not the quadrature's: %g vs %g"
            % (jr[0], jr_quad)
        )
        assert numpy.fabs(o.rap(pot=pot, analytic=True) / rap - 1.0) < 1e-8, (
            "The analytic apocentre of a bound orbit with a far apocentre is not the root of the radial equation"
        )
        # conserved along the orbit integrated over one full radial period
        ts = numpy.linspace(0.0, 2.0 * numpy.pi / Or[0], 101)
        o.integrate(ts, pot)
        j = aAS.actionsFreqsAngles(
            o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts)
        )
        assert numpy.ptp(j[0]) / numpy.mean(j[0]) < 1e-9, (
            "The radial action is not conserved along a bound orbit with a far apocentre"
        )
        assert numpy.ptp(j[2]) / numpy.mean(j[2]) < 1e-9, (
            "The vertical action is not conserved along a bound orbit with a far apocentre"
        )
        # an unbound orbit is still recognized as such
        ou = Orbit([1.0, 0.3, 3.0, 0.5, 0.1, 0.0])
        assert ou.E(pot=pot) > 0.0
        with pytest.raises(UnboundError):
            aAS.actionsFreqsAngles(ou.R(), ou.vR(), ou.vT(), ou.z(), ou.vz(), ou.phi())
        return None

    check(HernquistPotential(normalize=1.0, a=0.5))

    # the same for a potential that cannot be evaluated at infinity
    class _NaNAtInfinityPotential(HernquistPotential):
        def _evaluate(self, R, z, phi=0.0, t=0.0):
            if numpy.any(numpy.isinf(R)):
                return numpy.nan
            return super()._evaluate(R, z, phi=phi, t=t)

    check(_NaNAtInfinityPotential(normalize=1.0, a=0.5))

    # and for one that raises there
    class _RaisesAtInfinityPotential(HernquistPotential):
        def _evaluate(self, R, z, phi=0.0, t=0.0):
            if numpy.any(numpy.isinf(R)):
                raise ValueError("cannot be evaluated at infinity")
            return super()._evaluate(R, z, phi=phi, t=t)

    check(_RaisesAtInfinityPotential(normalize=1.0, a=0.5))
    return None


def test_actionAngleSpherical_smallr():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential()
    # Circular orbit at very small r: rperi is r itself (an epicycle of zero
    # amplitude around the circular orbit, whose radius is found to the root
    # tolerance of the circular-orbit condition)
    o = Orbit([0.000000001, 0.0, ip.vcirc(0.000000001), 0.0, 0.0, 0.0])
    assert (
        numpy.fabs(o.rperi(analytic=True, pot=ip, type="spherical") - 0.0) < 10.0**-8.0
    ), "rperi is not tiny for a circular orbit at very small r"
    # Orbit just outside rperi, very small r; the Orbit interface lifts a
    # planar orbit to a height of 1e-10, which gives this one an angular
    # momentum of that height times its radial velocity and a pericentre of
    # 1e-10, resolved now that the search no longer gives up at 1e-9
    o = Orbit([0.000000001, 0.0001, ip.vcirc(0.000000001), 0.0, 0.0, 0.0])
    assert numpy.fabs(o.rperi(analytic=True, pot=ip, type="spherical") - 0.0) < 2e-10, (
        "rperi is not ~0 for very small r"
    )
    return None


# Test actionAngleSpherical for circular and near-circular orbits, against
# the isochrone's exact transformation: the general quadratures lose precision
# as the epicyclic amplitude shrinks (and the turning-point search its bracket),
# so below an amplitude of 1e-4 of the circular radius the actions, frequencies
# and angles are the epicycle's; both regimes and the circular orbit itself
def test_actionAngleSpherical_near_circular():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleSpherical
    from galpy.potential import (
        IsochronePotential,
        LogarithmicHaloPotential,
        epifreq,
        omegac,
        vcirc,
    )

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAS = actionAngleSpherical(pot=ip)
    aAI = actionAngleIsochrone(ip=ip)
    wrap = lambda d: numpy.fabs((d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
    R, z, phi = 0.9, 0.3, 0.7
    r = numpy.sqrt(R**2 + z**2)
    vc = vcirc(ip, r, use_physical=False)
    rhat, that = numpy.array([R, z]) / r, numpy.array([-z, R]) / r
    # an inclined orbit at its guiding radius with a radial kick v_r, from
    # an amplitude of ~1e-2 of the radius down to the circular orbit
    for vr in (1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-9, 0.0):
        vRz = vr * rhat + 0.6 * vc * that
        args = (R, vRz[0], 0.8 * vc, z, vRz[1], phi)
        f = aAS.actionsFreqsAngles(*args)
        g = aAI.actionsFreqsAngles(*args)
        assert numpy.fabs(f[0][0] - g[0][0]) < 3e-5 * g[0][0] + 1e-15, (
            "J_r of a near-circular orbit is wrong at v_r = %g: %g vs %g"
            % (vr, f[0][0], g[0][0])
        )
        assert numpy.fabs(f[3][0] / g[3][0] - 1.0) < 3e-5, (
            "Omega_r of a near-circular orbit is wrong at v_r = %g" % vr
        )
        assert numpy.fabs(f[5][0] / g[5][0] - 1.0) < 1e-7, (
            "Omega_z of a near-circular orbit is wrong at v_r = %g" % vr
        )
        # the isochrone's own angles are not finite for the tiniest orbits
        if numpy.isfinite(g[6][0]) and g[0][0] > 1e-14:
            assert wrap(f[6][0] - g[6][0]) < 3e-4, (
                "theta_r of a near-circular orbit is wrong at v_r = %g" % vr
            )
        if numpy.isfinite(g[8][0]):
            assert wrap(f[8][0] - g[8][0]) < 1e-7, (
                "theta_z of a near-circular orbit is wrong at v_r = %g" % vr
            )
        assert numpy.all(numpy.isfinite(numpy.array([x[0] for x in f]))), (
            "The forward transformation is not finite at v_r = %g" % vr
        )
        # the actions-only and actions-and-frequencies paths agree with it
        assert numpy.fabs(aAS(*args)[0][0] - f[0][0]) < 1e-15
        assert numpy.fabs(aAS.actionsFreqs(*args)[3][0] - f[3][0]) < 1e-15
    # the cases that used to fail or return NaN, in a logarithmic halo
    lp = LogarithmicHaloPotential(normalize=1.0)
    aAS = actionAngleSpherical(pot=lp)
    r = 0.99
    vc = vcirc(lp, r, use_physical=False)
    kappa = epifreq(lp, r, use_physical=False)
    for args, jr_expect in (
        ((r, 0.0, vc, 0.0, 0.0, 0.3), 0.0),  # exactly circular
        ((r, 1e-8, vc, 0.0, 0.0, 0.3), 0.5e-16 / kappa),  # a tiny radial kick
        (
            (r * (1.0 + 1e-7), 0.0, vc / (1.0 + 1e-7), 0.0, 0.0, 0.3),
            0.5 * kappa * (r * 1e-7) ** 2,
        ),  # at apocentre
        (
            (0.7, 5e-17 * 0.7, 0.8 * vc, 0.7, 5e-17 * 0.7, 0.3),
            None,
        ),  # an eccentric orbit at its apocentre to round-off
    ):
        f = aAS.actionsFreqsAngles(*args)
        assert numpy.all(numpy.isfinite(numpy.array([x[0] for x in f]))), (
            f"The forward transformation is not finite at {args}"
        )
        if jr_expect is not None:
            assert numpy.fabs(f[0][0] - jr_expect) < 1e-3 * jr_expect + 1e-16, (
                "J_r at {} is not the epicycle's: {:g} vs {:g}".format(
                    args, f[0][0], jr_expect
                )
            )
            assert numpy.fabs(f[3][0] / kappa - 1.0) < 1e-6
            assert numpy.fabs(f[5][0] / omegac(lp, r, use_physical=False) - 1.0) < 1e-6
    # the epicycle's turning points are also what EccZmaxRperiRap returns
    e, zmax, rperi, rap = aAS.EccZmaxRperiRap(r, 1e-6, vc, 0.0, 0.0, 0.3)
    w = 1e-6 / kappa
    # (r_c is found to the root tolerance of the circular-orbit condition)
    assert numpy.fabs(rperi - (r - w)) < 1e-9 and numpy.fabs(rap - (r + w)) < 1e-9, (
        "The turning points of an epicycle are not r_c -/+ w"
    )
    return None


# Test that actionAngleSpherical's angler works when at pericenter
def test_actionAngleSpherical_angler_at_pericenter():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential()
    o = Orbit([1.0, 0.0, ip.vcirc(1.0) * 2.1, 0.0, 0.0, 0.0])
    # Radial angle wr should be zero
    assert numpy.fabs(o.wr(analytic=True, pot=ip, type="spherical")) < 10.0**-10.0, (
        "angler is not 0 at pericenter"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, gamma=1.0)
    # circular orbit
    R, vR, vT, phi = 1.0, 0.0, 1.0, 2.0
    js = aAA(Orbit([R, vR, vT, phi]))
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.01, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_gamma0():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential[0] + MWPotential[1:], gamma=0.0)
    # circular orbit
    R, vR, vT, phi = 1.0, 0.0, 1.0, 2.0
    js = aAA(Orbit([R, vR, vT, phi]))
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.01, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    # test nested list of potentials
    aAA = actionAngleAdiabatic(pot=MWPotential[0] + MWPotential[1:], c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_unboundz_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=True, gamma=0.0)
    # Unbound in z, so jz should be very large
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 10.0
    js = aAA(R, vR, vT, z, vz)
    assert js[2] > 1000.0, (
        "Unbound orbit in z in the MWPotential does not have large Jz"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_zerolz_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=True, gamma=0.0)
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, 0.0, 0.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, 0.0, 0.0000001, 0.0, 0.0
    js2 = aAA(R, vR, vT, z, vz)
    assert numpy.fabs(js[0] - js2[0]) < 10.0**-6.0, (
        "Orbit with zero angular momentum does not have the correct Jr"
    )
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, -0.5, 0.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, -0.5, 0.0000001, 0.0, 0.0
    js2 = aAA(R, vR, vT, z, vz)
    assert numpy.fabs(js[0] - js2[0]) < 10.0**-6.0, (
        "Orbit with zero angular momentum does not have the correct Jr"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic frequencies
def test_actionAngleAdiabatic_basic_freqs():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq

    aAS = actionAngleAdiabatic(pot=MWPotential, delta=0.71, c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    jos = aAS.actionsFreqs(R, vR, vT, z, vz)
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.01, 1.01, 0.01, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.03, 1.02, 0.03, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, 2.0]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-0.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, -0.03, 1.02, 0.03, 0.01
    jos = aAS.actionsFreqs(R, vR, vT, z, vz, 2.0)
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-0.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_freqsAngles():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq

    aAS = actionAngleAdiabatic(pot=MWPotential, delta=0.71, c=True)
    # v. close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 10.0**-4.0, 1.0, 10.0**-4.0, 0.0
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, 2.0]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic angles
def test_actionAngleAdiabatic_circular_angles_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAS = actionAngleAdiabatic(pot=MWPotential, delta=0.71, c=True)
    # Circular orbits, have zero/pi r and z angles in our implementation
    R, vR, vT, z, vz, phi = 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
    js = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    assert (
        numpy.fabs(js[6]) < 10.0**-8.0 or numpy.fabs(js[6] - numpy.pi) < 10.0**-8.0
    ), "Circular orbit does not have zero/pi r angles"
    assert (
        numpy.fabs(js[8]) < 10.0**-8.0 or numpy.fabs(js[8] - numpy.pi) < 10.0**-8.0
    ), "Circular orbit does not have zero/pi z angles"
    return None


# Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, gamma=1.0)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap_gamma0():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MiyamotoNagaiPotential

    mp = MiyamotoNagaiPotential(normalize=1.0, a=1.5, b=0.3)
    aAA = actionAngleAdiabatic(pot=mp, gamma=0.0, c=False)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap_gamma_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, gamma=1.0, c=True)
    # circular orbit
    R, vR, vT, z, vz, phi = 1.0, 0.0, 1.0, 0.0, 0.0, 2.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(Orbit([R, vR, vT, z, vz, phi]))
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz, phi = 1.01, 0.01, 1.0, 0.01, 0.01, 2.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz, phi)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_actions():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=False)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.2, -8.0, -1.7, ntimes=101
    )
    return None


# The C adiabatic actions integrate sqrt(F) over an interval where F has a sqrt zero
# at the endpoint(s) -- z=zmax for Jz, BOTH rperi and rap for JR. Plain Gauss-Legendre
# is only algebraically convergent there (O(n^-3)), which cost ~4 digits at the
# order-10 default (gh#1354). The C now substitutes z=zmax*sin(phi) / R=cc-rr*cos(theta)
# to make the integrand analytic at those ends. Guard it against the pure-Python path,
# which uses adaptive quadrature and is exact to ~1e-14: before the fix this grid gave
# max rel err 7.9e-4 (jr) and 1.5e-4 (jz); after, 6.8e-10 and 9.3e-10 at order 20
# (1.9e-6 / 1.5e-6 with the substitution but order still 10, which this bar also catches).
def test_actionAngleAdiabatic_c_matches_python_quadrature():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential2014

    aAC = actionAngleAdiabatic(pot=MWPotential2014, gamma=1.0, c=True)
    aAP = actionAngleAdiabatic(pot=MWPotential2014, gamma=1.0, c=False)
    # a spread of eccentricities and vertical amplitudes, off any symmetry line
    R = numpy.array([0.6, 0.9, 1.0, 1.3, 1.8, 0.75, 1.15, 2.0])
    vR = numpy.array([0.05, -0.12, 0.2, -0.05, 0.1, 0.18, -0.2, 0.08])
    vT = numpy.array([1.0, 0.9, 0.8, 1.05, 0.7, 1.1, 0.85, 0.6])
    z = numpy.array([0.05, 0.12, -0.2, 0.08, 0.3, -0.1, 0.25, 0.15])
    vz = numpy.array([0.08, -0.15, 0.1, 0.2, -0.05, 0.12, 0.18, -0.1])
    jrC, _, jzC = aAC(R, vR, vT, z, vz)
    jrP, _, jzP = aAP(R, vR, vT, z, vz)
    jrC, jzC = numpy.asarray(jrC, dtype=float), numpy.asarray(jzC, dtype=float)
    jrP, jzP = numpy.asarray(jrP, dtype=float), numpy.asarray(jzP, dtype=float)
    djr = numpy.amax(numpy.fabs(jrC - jrP) / numpy.fabs(jrP))
    djz = numpy.amax(numpy.fabs(jzC - jzP) / numpy.fabs(jzP))
    assert djr < 1e-8, (
        f"C jr disagrees with the exact python quadrature by {djr:.3e} (>1e-8): the "
        "endpoint substitution or the quadrature order in calcJRAdiabatic may have "
        "been lost"
    )
    assert djz < 1e-8, (
        f"C jz disagrees with the exact python quadrature by {djz:.3e} (>1e-8): the "
        "endpoint substitution or the quadrature order in calcJzAdiabatic may have "
        "been lost"
    )
    return None


# Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import CylindricallySeparablePotentialWrapper, MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabatic(pot=MWPotential, c=True)
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.4, -8.0, -1.7, ntimes=101
    )

    # Applying actionAngleAdiabatic to a separable potential should give very good
    # conservation of actions
    cyl_pot = CylindricallySeparablePotentialWrapper(pot=MWPotential, Rp=1.1)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabatic(pot=cyl_pot, c=True, gamma=0.0)
    check_actionAngle_conserved_actions(aAA, obs, cyl_pot, -8.0, -8.0, -8.0, ntimes=101)
    return None


# Test the actions of an actionAngleAdiabatic, single pot
def test_actionAngleAdiabatic_conserved_actions_singlepot():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    mp = MiyamotoNagaiPotential(normalize=1.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleAdiabatic(pot=mp, c=False)
    check_actionAngle_conserved_actions(
        aAA, obs, mp, -1.5, -8.0, -2.0, ntimes=101, inclphi=True
    )
    return None


# Test the actions of an actionAngleAdiabatic, single pot, C
def test_actionAngleAdiabatic_conserved_actions_singlepot_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    mp = MiyamotoNagaiPotential(normalize=1.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleAdiabatic(pot=mp, c=True)
    check_actionAngle_conserved_actions(
        aAA, obs, mp, -1.5, -8.0, -2.0, ntimes=101, inclphi=True
    )
    return None


# Test the actions of an actionAngleAdiabatic, interpolated pot
def test_actionAngleAdiabatic_conserved_actions_interppot_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
    )
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleAdiabatic(pot=ip, c=True)
    check_actionAngle_conserved_actions(aAA, obs, ip, -1.4, -8.0, -1.7, ntimes=101)
    return None


# Test that the actions for a cylindrically-separable potential are very well conserved
def test_actionAngleAdiabatic_conserved_actions_cylsep():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import CylindricallySeparablePotentialWrapper, MWPotential2014

    cyl_pot = CylindricallySeparablePotentialWrapper(pot=MWPotential2014, Rp=1.1)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabatic(pot=cyl_pot, c=False, gamma=0.0)
    check_actionAngle_conserved_actions(aAA, obs, cyl_pot, -8.0, -8.0, -8.0, ntimes=101)
    return None


# Test that the actions for a cylindrically-separable potential are very well conserved
def test_actionAngleAdiabatic_conserved_actions_cylsep_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import CylindricallySeparablePotentialWrapper, MWPotential2014

    cyl_pot = CylindricallySeparablePotentialWrapper(pot=MWPotential2014, Rp=1.1)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabatic(pot=cyl_pot, c=True, gamma=0.0)
    check_actionAngle_conserved_actions(aAA, obs, cyl_pot, -8.0, -8.0, -8.0, ntimes=101)
    return None


# Test the frequencies of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_frequencies():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=False)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    times = numpy.linspace(0.0, 100.0, 101)
    obs.integrate(times, MWPotential, method="dopr54_c")
    os = aAA.actionsFreqs(obs(times))[3:]
    maxdo = numpy.amax(
        numpy.fabs(os - numpy.tile(numpy.mean(os, axis=1), (len(times), 1)).T), axis=1
    ) / numpy.mean(os, axis=1)
    assert maxdo[0] < 10.0**-2.0, "Or conservation fails at %g%%" % (100.0 * maxdo[0])
    assert maxdo[1] < 10.0**-2.0, "Oz conservation fails at %g%%" % (100.0 * maxdo[1])
    assert maxdo[2] < 10.0**-1.0, "Oz conservation fails at %g%%" % (100.0 * maxdo[2])
    return None
    return None


# Test that the angles of an actionAngleAdiabatic increase linearly
def test_actionAngleAdiabatic_linear_angles():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=False)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 0.0])
    check_actionAngle_linear_angles(
        aAA,
        obs,
        MWPotential,
        -1.5,
        -4.0,
        -1.5,
        -2.5,
        -2.5,
        -0.5,
        -1.5,
        -3.0,
        -0.5,
        ntimes=1001,
    )  # need fine sampling for de-period
    return None


# Test that the angles of an actionAngleAdiabatic for a cylindrically-separable potential
# increase linearly to very good approximation
def test_actionAngleAdiabatic_linear_angles_cylsep():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import CylindricallySeparablePotentialWrapper, MWPotential2014

    pot = CylindricallySeparablePotentialWrapper(pot=MWPotential2014, Rp=1.1)
    aAA = actionAngleAdiabatic(pot=pot, c=False, gamma=0.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 0.0])
    check_actionAngle_linear_angles(
        aAA,
        obs,
        pot,
        -8.0,
        -8.0,
        -7.5,
        -8.0,
        -8.0,
        -8.0,
        -7.0,
        -7.0,
        -7.0,
        ntimes=1001,
    )  # need fine sampling for de-period
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=False, gamma=1.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 0.0])
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, MWPotential, -1.7, -1.4, -2.0, -2.0, ntimes=101
    )
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap_ecc():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=False, gamma=1.0)
    obs = Orbit([1.1, 0.2, 1.3, 0.1, 0.0, 2.0])
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, MWPotential, -1.1, -0.4, -1.8, -1.8, ntimes=101, inclphi=True
    )
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRap_singlepot_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    mp = MiyamotoNagaiPotential(normalize=1.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleAdiabatic(pot=mp, c=True)
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, mp, -1.7, -1.4, -2.0, -2.0, ntimes=101
    )
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_EccZmaxRperiRa_interppot_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
    )
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleAdiabatic(pot=ip, c=True)
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, ip, -1.7, -1.4, -2.0, -2.0, ntimes=101
    )
    return None


# Test the actionAngleAdiabatic against an isochrone potential: actions
def test_actionAngleAdiabatic_Isochrone_actions():
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleIsochrone
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleAdiabatic(pot=ip, c=True)
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-1.2, (
        "actionAngleAdiabatic applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleAdiabatic applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-1.2, (
        "actionAngleAdiabatic applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions (incl. conserved, bc takes a lot of time)
def test_actionAngleAdiabaticGrid_basicAndConserved_actions():
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabaticGrid(pot=MWPotential, gamma=1.0, c=False)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz, 0.0)
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(aAA.Jz(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # setup w/ multi
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, gamma=1.0, c=False, numcores=2)
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )
    # Check that actions are conserved along the orbit
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.2, -8.0, -1.7, ntimes=101
    )
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabaticGrid_basic_actions_c():
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabaticGrid(pot=MWPotential, c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )


# actionAngleAdiabaticGrid actions outside the grid
def test_actionAngleAdiabaticGrid_outsidegrid_multiple_python():
    # The pure-Python (c=False) grid raised
    #   TypeError: 'float' object is not subscriptable
    # whenever TWO OR MORE points fell outside the grid: actionAngleAdiabatic's
    # len(R) > 1 branch loops over points calling its own scalar branch and then
    # does ojr[ii] = tjr[0], but the scalar _justjr return was a bare float
    # while its _justjz and general siblings both wrapped in numpy.atleast_1d.
    #
    # One off-grid point never caught it (that call takes the scalar path and
    # never reaches the loop), and the only other off-grid test uses c=True,
    # which returns from the C branch before the loop -- so both arms of the
    # existing coverage were blind to it.
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleAdiabaticGrid
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=False)
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, c=False, Rmax=2.0, zmax=0.2)
    for n in (1, 2, 3):  # 1 is the case that always worked; 2+ is the bug
        R = numpy.array([3.0 + 0.5 * ii for ii in range(n)])
        o = numpy.ones(n)
        js = aA(R, 0.1 * o, 1.0 * o, 0.1 * o, 0.1 * o)
        jsa = aAA(R, 0.1 * o, 1.0 * o, 0.1 * o, 0.1 * o)
        assert numpy.all(numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0), (
            f"actionAngleAdiabaticGrid c=False jr wrong for {n} off-grid points"
        )
        assert numpy.all(numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0), (
            f"actionAngleAdiabaticGrid c=False jz wrong for {n} off-grid points"
        )


def test_actionAngleAdiabaticGrid_outsidegrid_c():
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleAdiabaticGrid
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, c=True, Rmax=2.0, zmax=0.2)
    R, vR, vT, z, vz, phi = 3.0, 0.1, 1.0, 0.1, 0.1, 2.0
    js = aA(R, vR, vT, z, vz, phi)
    jsa = aAA(R, vR, vT, z, vz, phi)
    assert numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    assert numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    assert numpy.fabs(js[2] - aAA.Jz(R, vR, vT, z, vz, phi)) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    # Also for array
    s = numpy.ones(2)
    js = aA(R, vR, vT, z, vz, phi)
    jsa = aAA(R * s, vR * s, vT * s, z * s, vz * s, phi * s)
    assert numpy.all(numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0), (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    assert numpy.all(numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0), (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    return None


# Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabaticGrid_conserved_actions_c():
    from galpy.actionAngle import actionAngleAdiabaticGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, c=True)
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.4, -8.0, -1.7, ntimes=101
    )
    return None


# Test the actionAngleAdiabatic against an isochrone potential: actions
def test_actionAngleAdiabaticGrid_Isochrone_actions():
    from galpy.actionAngle import actionAngleAdiabaticGrid, actionAngleIsochrone
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleAdiabaticGrid(pot=ip, c=True)
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-1.2, (
        "actionAngleAdiabatic applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleAdiabatic applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-1.2, (
        "actionAngleAdiabatic applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import CompositePotential, MWPotential

    # test nested list of potentials
    aAS = actionAngleStaeckel(
        pot=CompositePotential([MWPotential[0], MWPotential[1:]]),
        delta=0.71,
        c=False,
        useu0=True,
    )
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import CompositePotential, MWPotential

    # test nested list of potentials
    aAS = actionAngleStaeckel(
        pot=CompositePotential([MWPotential[0], MWPotential[1:]]),
        delta=0.71,
        c=True,
        useu0=True,
    )
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]), u0=1.15)
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions, w/ u0, and interppot
def test_actionAngleStaeckel_basic_actions_u0_interppot_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
    )
    aAS = actionAngleStaeckel(pot=ip, delta=0.71, c=True, useu0=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0][0]) < 10.0**-12.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2][0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 2.0 * 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Unbound orbit, shouldn't fail
    R, vR, vT, z, vz = 1.0, 0.0, 10.0, 0.1, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert js[0] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Jr"
    )
    # Another unbound orbit, shouldn't fail
    R, vR, vT, z, vz = 1.0, 0.1, 10.0, 0.1, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert js[0] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Jr"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_zerolz_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=True, delta=0.71)
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, 0.0, 0.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, 0.0, 0.0000001, 0.0, 0.0
    js2 = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0] - js2[0]) < 10.0**-6.0, (
        "Orbit with zero angular momentum does not have the correct Jr"
    )
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, -0.5, 0.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, -0.5, 0.0000001, 0.0, 0.0
    js2 = aAS(R, vR, vT, z, vz)
    assert numpy.fabs(js[0] - js2[0]) < 10.0**-6.0, (
        "Orbit with zero angular momentum does not have the correct Jr"
    )
    return None


# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_c_angles_freqs_near_turning_point():
    # c=True frequencies and angles used to go wrong for points near a
    # turning point: when |p^2| < 1e-7 there, the C code adopted the
    # evaluation point ITSELF as the turning point instead of solving for
    # it. That O(eps) endpoint error enters the actions only at
    # O(eps^1.5) (invisible, ~1e-12) but the 1/sqrt(W)-divergent
    # frequency and angle integrands at O(sqrt(eps)) -- theta_z errors up
    # to ~1e-4 -- and, through ~1e-12-absolute root tolerances, also left
    # an order-growing angle error on generic points. Frequencies are
    # torus constants and the Python path is exact here, so both provide
    # sharp regression checks.
    import numpy

    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    o = Orbit([1.1, 0.35, 1.1, 0.3, 0.25, 0.0])
    ts = numpy.linspace(0.0, 8.0, 17)
    o.integrate(ts, kkp)
    R, vR, vT, z, vz, phi = (
        numpy.array([float(f(t)) for t in ts])
        for f in (o.R, o.vR, o.vT, o.z, o.vz, o.phi)
    )
    # a point of this orbit within ~1e-8 of its upper vertical turning
    # point (p_v^2 ~ 6e-8), given as a literal so the near-turning
    # regime is hit deterministically on every platform
    R[-1], vR[-1], vT[-1], z[-1], vz[-1], phi[-1] = (
        1.0374878950211397,
        0.26453976273320806,
        1.166278667740467,
        0.3280135549475248,
        0.0331712140016195,
        -2.550334704756344,
    )
    aAC = actionAngleStaeckel(pot=kkp, delta=1.3, c=True, order=100)
    aAP = actionAngleStaeckel(pot=kkp, delta=1.3, c=False)
    C = aAC.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    # frequencies are torus constants: they may not vary along the orbit
    # (the near-turning point used to be off by ~2e-4 in Omega_z)
    for k, name in ((3, "Omega_R"), (5, "Omega_z")):
        spread = numpy.ptp(numpy.array(C[k])) / numpy.fabs(
            numpy.median(numpy.array(C[k]))
        )
        assert spread < 1e-8, (
            "c=True %s varies along an orbit by %g near a turning point"
            % (name, spread)
        )
    # angles agree with the (exact) Python path pointwise, including at
    # the near-turning sample (used to be off by ~4e-4 in theta_z)
    P = aAP.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    for k, name in ((6, "theta_R"), (8, "theta_z")):
        d = (
            numpy.remainder(
                numpy.array(C[k]) - numpy.array(P[k]) + numpy.pi, 2.0 * numpy.pi
            )
            - numpy.pi
        )
        assert numpy.max(numpy.fabs(d)) < 1e-6, (
            "c=True %s disagrees with c=False by %g near a turning point"
            % (name, numpy.max(numpy.fabs(d)))
        )
    return None


def test_actionAngleStaeckel_actions_order():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._delta, c=False)
    # The chi-anomaly composite quadrature is machine-converged at any order,
    # so low and high order must both match a very-high-order reference at
    # machine precision (the old fixed-order rule converged only slowly here)
    jrt, jpt, jzt = aAS(o, order=10000, fixed_quad=True)
    jr1, jp1, jz1 = aAS(o, order=5, fixed_quad=True)
    jr2, jp2, jz2 = aAS(o, order=50, fixed_quad=True)
    assert numpy.fabs(jr1 - jrt) < 1e-14, (
        "actionAngleStaeckel low-order actions do not match the high-order "
        "reference at machine precision"
    )
    assert numpy.fabs(jr2 - jrt) < 1e-14, (
        "actionAngleStaeckel medium-order actions do not match the high-order "
        "reference at machine precision"
    )
    assert numpy.fabs(jz1 - jzt) < 1e-14, (
        "actionAngleStaeckel low-order actions do not match the high-order "
        "reference at machine precision"
    )
    assert numpy.fabs(jz2 - jzt) < 1e-14, (
        "actionAngleStaeckel medium-order actions do not match the high-order "
        "reference at machine precision"
    )
    return None


def test_actionAngleStaeckel_actions_order_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt = aAS(o, order=10000)
    jr1, jp1, jz1 = aAS(o, order=5)
    jr2, jp2, jz2 = aAS(o, order=50)
    assert numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel frequencies
def test_actionAngleStaeckel_basic_freqs_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    jos = aAS.actionsFreqs(R, vR, vT, z, vz)
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.01, 1.01, 0.01, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.03, 1.02, 0.03, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, 2.0]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-0.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_freqsAngles():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # v. close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 10.0**-4.0, 1.0, 10.0**-4.0, 0.0
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, 2.0]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleStaeckel frequencies
def test_actionAngleStaeckel_basic_freqs_c_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, epifreq, omegac, verticalfreq

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True, useu0=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    jos = aAS.actionsFreqs(R, vR, vT, z, vz)
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    # close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.01, 1.01, 0.01, 0.01
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz]), u0=1.15)
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.5
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_freqs_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import (
        MWPotential,
        epifreq,
        interpRZPotential,
        omegac,
        verticalfreq,
    )

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
    )
    aAS = actionAngleStaeckel(pot=ip, delta=0.71, c=True, useu0=True)
    # v. close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 10.0**-4.0, 1.0, 10.0**-4.0, 0.0
    jos = aAS.actionsFreqs(Orbit([R, vR, vT, z, vz, 2.0]))
    assert (
        numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Or=kappa at %g%%" % (
        100.0
        * numpy.fabs((jos[3] - epifreq(MWPotential, 1.0)) / epifreq(MWPotential, 1.0))
    )
    assert (
        numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Op=Omega at %g%%" % (
        100.0
        * numpy.fabs((jos[4] - omegac(MWPotential, 1.0)) / omegac(MWPotential, 1.0))
    )
    assert (
        numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
        < 10.0**-1.9
    ), "Close-to-circular orbit in the MWPotential does not have Oz=nu at %g%%" % (
        100.0
        * numpy.fabs(
            (jos[5] - verticalfreq(MWPotential, 1.0)) / verticalfreq(MWPotential, 1.0)
        )
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_freqs_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Unbound orbit, shouldn't fail
    R, vR, vT, z, vz = 1.0, 0.1, 10.0, 0.1, 0.0
    js = aAS.actionsFreqs(R, vR, vT, z, vz)
    assert js[0] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Jr"
    )
    assert js[3] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Or"
    )
    assert js[4] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Op"
    )
    assert js[5] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Oz"
    )
    return None


# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_freqs_order_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt, ort, opt, ozt = aAS.actionsFreqs(o, order=10000)
    jr1, jp1, jz1, or1, op1, oz1 = aAS.actionsFreqs(o, order=5)
    jr2, jp2, jz2, or2, op2, oz2 = aAS.actionsFreqs(o, order=50)
    assert numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(or1 - ort) > numpy.fabs(or2 - ort), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(op1 - opt) > numpy.fabs(op2 - opt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(oz1 - ozt) > numpy.fabs(oz2 - ozt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Unbound orbit, shouldn't fail
    R, vR, vT, z, vz, phi = 1.0, 0.1, 10.0, 0.1, 0.0, 0.0
    js = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    assert js[0] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large Jr"
    )
    assert js[6] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large ar"
    )
    assert js[7] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large ap"
    )
    assert js[8] > 1000.0, (
        "Unbound in R orbit in the MWPotential does not have large az"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_circular_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Circular orbits, have zero r and z angles in our implementation
    R, vR, vT, z, vz, phi = 1.0, 0.0, 1.0, 0.0, 0.0, 1.0
    js = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    assert numpy.fabs(js[6]) < 10.0**-8.0, "Circular orbit does not have zero angles"
    assert numpy.fabs(js[8]) < 10.0**-8.0, "Circular orbit does not have zero angles"
    return None


# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_angles_order_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt, ort, opt, ozt, art, apt, azt = aAS.actionsFreqsAngles(o, order=10000)
    jr1, jp1, jz1, or1, op1, oz1, ar1, ap1, az1 = aAS.actionsFreqsAngles(o, order=5)
    jr2, jp2, jz2, or2, op2, oz2, ar2, ap2, az2 = aAS.actionsFreqsAngles(o, order=50)
    assert numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(or1 - ort) > numpy.fabs(or2 - ort), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(op1 - opt) > numpy.fabs(op2 - opt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(oz1 - ozt) > numpy.fabs(oz2 - ozt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(ar1 - art) > numpy.fabs(ar2 - art), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(ap1 - apt) > numpy.fabs(ap2 - apt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    assert numpy.fabs(az1 - azt) > numpy.fabs(az2 - azt), (
        "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    )
    return None


# Test that the pure-Python (c=False) actionAngleStaeckel frequencies and angles
# agree with the C implementation over a grid of ICs hitting every branch.
def test_actionAngleStaeckel_single_action_cache():
    # actionAngleStaeckelSingle caches JR/Jz per (fixed_quad, order): a repeat
    # call with the same settings returns the cached value, while changing the
    # order recomputes (the cache used to ignore order, which silently
    # returned the first result for any subsequent order)
    from galpy.actionAngle.actionAngleStaeckel import actionAngleStaeckelSingle
    from galpy.potential import MWPotential2014

    aA = actionAngleStaeckelSingle(
        1.1, 0.05, 0.9, 0.15, 0.12, pot=MWPotential2014, delta=0.45
    )
    jr1 = numpy.atleast_1d(aA.JR(fixed_quad=True, order=10))[0]
    jr2 = numpy.atleast_1d(aA.JR(fixed_quad=True, order=10))[0]  # cache hit
    assert jr1 == jr2, (
        "Repeated actionAngleStaeckelSingle.JR call with identical settings "
        "does not return the cached value"
    )
    jz1 = numpy.atleast_1d(aA.Jz(fixed_quad=True, order=10))[0]
    jz2 = numpy.atleast_1d(aA.Jz(fixed_quad=True, order=10))[0]  # cache hit
    assert jz1 == jz2, (
        "Repeated actionAngleStaeckelSingle.Jz call with identical settings "
        "does not return the cached value"
    )
    # Changing the order must recompute, not return the cached value; both
    # are converged, so they agree to machine precision without being
    # bit-identical in general
    jr3 = numpy.atleast_1d(aA.JR(fixed_quad=True, order=40))[0]
    jz3 = numpy.atleast_1d(aA.Jz(fixed_quad=True, order=40))[0]
    assert numpy.fabs(jr3 - jr1) < 1e-12, (
        "actionAngleStaeckelSingle.JR at a different order does not agree "
        "with the default order"
    )
    assert numpy.fabs(jz3 - jz1) < 1e-12, (
        "actionAngleStaeckelSingle.Jz at a different order does not agree "
        "with the default order"
    )
    return None


def test_actionAngleStaeckel_chi_quadrature_convergence():
    # The chi-anomaly composite quadratures behind the pure-Python path are
    # machine-converged at the default order: frequencies and angles must
    # match a much finer chi mesh at machine precision, including the
    # partial-oscillation (angle) integrals on both sides of the turning
    # points and the midplane
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    aAS = actionAngleStaeckel(pot=kksp, delta=1.4, c=False)
    # The tolerances are set by the evaluation noise of the fudge-form
    # momentum function S (a difference of O(1) potential terms), not by the
    # quadrature rule: a few 1e-12 for generic orbits, and looser for a
    # nearly planar orbit whose tiny v oscillation has S far below the
    # cancellation scale (the old fixed-order rule erred at 4.6e-4 here)
    for ic, tol in (
        ([1.0, 0.5, 1.1, 0.2, -0.3, 0.4], 3e-11),
        ([1.0, -0.2, 1.1, -0.2, 0.25, 2.1], 3e-11),  # z<0, vR<0: other branches
        ([1.1, 0.02, 0.9, 0.002, 0.02, 1.0], 1e-8),  # nearly planar
    ):
        lo = aAS.actionsFreqsAngles(*ic, fixed_quad=True, order=10)
        hi = aAS.actionsFreqsAngles(*ic, fixed_quad=True, order=200)
        for ii in range(9):
            assert numpy.fabs(lo[ii][0] - hi[ii][0]) < tol, (
                "Pure-Python actionAngleStaeckel chi-quadrature output %i at "
                "the default order does not match a much finer chi mesh "
                "(diff %g)" % (ii, numpy.fabs(lo[ii][0] - hi[ii][0]))
            )
    return None


def test_actionAngleStaeckel_actions_c_convergence():
    # C path: the t^2-substituted action integrals are converged at the
    # default order; the previous plain Gauss-Legendre rule against the
    # sqrt branch points at the turning points erred systematically at
    # ~4.6e-4 (J_R) / 1.4e-4 (J_z) and converged only as order^-3
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    aAc = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    for ic in (
        (1.1, 0.05, 0.9, 0.15, 0.12, 0.3),
        (0.7, -0.15, 1.05, 0.05, -0.2, 1.1),
        (1.6, 0.2, 0.7, 0.3, 0.1, 2.5),
    ):
        lo = aAc(*ic, order=10)
        hi = aAc(*ic, order=1280)
        assert numpy.fabs(lo[0][0] - hi[0][0]) < 1e-10, (
            "C actionAngleStaeckel J_R at the default order does not match a "
            "very-high-order reference (diff %g)" % numpy.fabs(lo[0][0] - hi[0][0])
        )
        assert numpy.fabs(lo[2][0] - hi[2][0]) < 1e-10, (
            "C actionAngleStaeckel J_z at the default order does not match a "
            "very-high-order reference (diff %g)" % numpy.fabs(lo[2][0] - hi[2][0])
        )
    return None


def test_actionAngleStaeckel_python_c_freqsAngles():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import LogarithmicHaloPotential

    # Flattened logarithmic halo: genuinely close to Staeckel-separable
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    # The Python path's chi-anomaly quadratures are machine-converged, so the
    # C-vs-Python difference is dominated by the C path's errors: its
    # fixed-order truncation in the frequency/angle integrals (1e-4 at the
    # default order=10 on this grid), removed by running C at order=200, and
    # below that its turning-point root-finding tolerance, which enters the
    # 1/sqrt(S) integrals amplified as sqrt(delta) and is platform-dependent
    # (~1e-8 on Linux, ~1e-6 on Windows through libm differences in the
    # roots). The 1e-5 tolerance sits above that floor while still catching
    # any branch/convention disagreement, which produces O(1) errors.
    aAc = actionAngleStaeckel(pot=lp, delta=0.5, c=True, order=200)
    aAp = actionAngleStaeckel(pot=lp, delta=0.5, c=False)

    def wrapdiff(a, b):
        d = (a - b) % (2.0 * numpy.pi)
        return numpy.minimum(d, 2.0 * numpy.pi - d)

    # Grid hitting all branches: small/large Jr, small/large Jz (near-planar),
    # near-circular, eccentric, z>0 and z<0 (vx</>pi/2), vR>0/<0, vz>0/<0
    # (pux/pvx signs), prograde and retrograde (vT<0). The vR=vz=0,z<0 corner
    # hits a pre-existing pure-Python calcVmin limitation (the actions path
    # raises there too), so we keep |vz|>0 when z<0. The grid steps stay clear
    # of the |pvx|<1e-3 annulus right at a vertical turning point, where the
    # tiny partial-integral bound sqrt(vx-vmin) amplifies the C-vs-scipy brentq
    # vmin tolerance (~1e-9) -- a measure-zero root-find floor, the Staeckel
    # analog of the Spherical at-peri/apo edge (neither path is ground truth).
    maxfreqdiff = 0.0
    maxangdiff = 0.0
    n = 0
    for R in [0.7, 1.0, 1.3]:
        for vR in [-0.25, 0.0, 0.25]:
            for vT in [-0.6, 0.4, 0.9]:  # retrograde + prograde + near-circular
                for z in [-0.2, 0.0, 0.2]:
                    for vz in [-0.25, 0.05, 0.25]:
                        for phi in [0.4, 2.7]:
                            if z < 0.0 and vR == 0.0 and vz == 0.0:
                                continue
                            fc = aAc.actionsFreqs(R, vR, vT, z, vz)
                            fp = aAp.actionsFreqs(R, vR, vT, z, vz)
                            ac = aAc.actionsFreqsAngles(R, vR, vT, z, vz, phi)
                            ap = aAp.actionsFreqsAngles(R, vR, vT, z, vz, phi)
                            n += 1
                            # jr,Lz,jz,Omegar,Omegaphi,Omegaz
                            for ii in range(6):
                                if numpy.isnan(fc[ii][0]) or numpy.isnan(fp[ii][0]):
                                    continue
                                maxfreqdiff = max(
                                    maxfreqdiff, numpy.fabs(fc[ii][0] - fp[ii][0])
                                )
                            # angler, anglephi, anglez (wrap-aware)
                            for ii in (6, 7, 8):
                                if numpy.isnan(ac[ii][0]) or numpy.isnan(ap[ii][0]):
                                    continue
                                maxangdiff = max(
                                    maxangdiff, wrapdiff(ac[ii][0], ap[ii][0])
                                )
    assert n > 100, "Staeckel c vs Python parity grid did not evaluate enough points"
    assert maxfreqdiff < 1e-5, (
        "Pure-Python actionAngleStaeckel frequencies do not agree with C "
        "implementation; max diff = %g" % maxfreqdiff
    )
    assert maxangdiff < 1e-5, (
        "Pure-Python actionAngleStaeckel angles do not agree with C "
        "implementation; max diff = %g" % maxangdiff
    )
    # Exactly-circular orbits (vR=vz=z=0, vT=vcirc): detA=0, so the C path gets
    # IEEE 0/0=NaN and substitutes epifreq/omegac/verticalfreq while the angles
    # are 0. The pure-Python path must reproduce this (and not raise on the
    # scalar 0/0). useu0 True and False both exercised.
    from galpy.potential import vcirc

    for usu in (False, True):
        aApc = actionAngleStaeckel(pot=lp, delta=0.5, c=False, useu0=usu)
        for R in [0.7, 1.0, 1.3]:
            vc = vcirc(lp, R, use_physical=False)
            fc = aAc.actionsFreqsAngles(R, 0.0, vc, 0.0, 0.0, 0.4)
            fp = aApc.actionsFreqsAngles(R, 0.0, vc, 0.0, 0.0, 0.4)
            for ii in range(9):
                d = (
                    numpy.fabs(fc[ii][0] - fp[ii][0])
                    if ii < 6
                    else wrapdiff(fc[ii][0], fp[ii][0])
                )
                assert d < 1e-6, (
                    "Staeckel circular c vs Python mismatch (useu0=%s) at "
                    "component %d: %g" % (usu, ii, d)
                )
    return None


# Test that the pure-Python (c=False) actionAngleStaeckel angles increase
# linearly with frequency along an integrated orbit.
def test_actionAngleStaeckel_python_linear_angles():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    check_actionAngle_linear_angles(
        aAS,
        obs,
        MWPotential,
        -2.0,
        -4.0,
        -3.0,
        -3.0,
        -3.0,
        -2.0,
        -2.0,
        -3.5,
        -2.0,
        ntimes=1001,
    )  # need fine sampling for de-period
    return None


# Test that actionAngleAdiabatic with c=False can compute frequencies and angles
# (it delegates to pure-Python Spherical + Vertical, needing no Staeckel C) and
# that the radial/azimuthal part matches the underlying Spherical actionsFreqs
# for an in-plane (z=vz=0) orbit, where the Adiabatic reduces exactly to it.
def test_actionAngleAdiabatic_python_freqsAngles():
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleSpherical
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    aAA = actionAngleAdiabatic(pot=lp, c=False)
    R, vR, vT, z, vz, phi = 1.0, 0.1, 0.9, 0.05, 0.1, 1.0
    # actionsFreqs and actionsFreqsAngles run without C and agree on the actions
    jr, lz, jz, Or, Op, Oz = aAA.actionsFreqs(R, vR, vT, z, vz)
    (
        jra,
        lza,
        jza,
        Ora,
        Opa,
        Oza,
        ar,
        ap,
        az,
    ) = aAA.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    assert numpy.fabs(jr - jra) < 1e-10, (
        "actionAngleAdiabatic actionsFreqs and actionsFreqsAngles disagree on jr"
    )
    assert numpy.fabs(jz - jza) < 1e-10, (
        "actionAngleAdiabatic actionsFreqs and actionsFreqsAngles disagree on jz"
    )
    assert numpy.fabs(Or - Ora) < 1e-10, (
        "actionAngleAdiabatic actionsFreqs and actionsFreqsAngles disagree on Or"
    )
    assert numpy.all(numpy.isfinite([jr, lz, jz, Or, Op, Oz])), (
        "actionAngleAdiabatic c=False actionsFreqs returned non-finite values"
    )
    assert numpy.all(numpy.isfinite([ar, ap, az])), (
        "actionAngleAdiabatic c=False actionsFreqsAngles returned non-finite angles"
    )
    # For an in-plane orbit (z=vz=0), the Adiabatic radial part is exactly the
    # Spherical actionsFreqs.
    aASph = actionAngleSpherical(pot=lp)
    jr0, lz0, jz0, Or0, Op0, Oz0 = aAA.actionsFreqs(R, vR, vT, 0.0, 0.0)
    sjr, slz, sjz, sOr, sOp, sOz = aASph.actionsFreqs(R, vR, vT, 0.0, 0.0)
    assert numpy.fabs(jr0 - sjr) < 1e-10, (
        "actionAngleAdiabatic in-plane radial action does not match Spherical"
    )
    assert numpy.fabs(Or0 - sOr) < 1e-10, (
        "actionAngleAdiabatic in-plane radial frequency does not match Spherical"
    )
    assert numpy.fabs(Op0 - sOp) < 1e-10, (
        "actionAngleAdiabatic in-plane azimuthal frequency does not match Spherical"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False, useu0=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap_u0_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True, useu0=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz, u0=1.15)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Test that using different delta for different phase-space points works
def test_actionAngleStaeckel_indivdelta_actions():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # actions with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=False)
    jr0, jp0, jz0 = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=False)
    jr1, jp1, jz1 = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with individual delta
    jri, jpi, jzi = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2]), delta=deltas
    )
    # Check that they agree as expected
    assert numpy.fabs(jr0[0] - jri[0]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jr1[1] - jri[1]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz0[0] - jzi[0]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz1[1] - jzi[1]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


# Test that no_median option for estimateDeltaStaeckel returns the same results as when
# individual values are calculated separately
def test_estimateDeltaStaeckel_no_median():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.001, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    # generate no_median deltas
    nomed = estimateDeltaStaeckel(
        MWPotential2014, o.R(ts[:10]), o.z(ts[:10]), no_median=True
    )
    # and the individual ones
    indiv = numpy.array(
        [
            estimateDeltaStaeckel(MWPotential2014, o.R(ts[i]), o.z(ts[i]))
            for i in range(10)
        ]
    )
    # check that values agree
    assert (numpy.fabs(nomed - indiv) < 1e-10).all(), (
        "no_median option returns different values to individual Delta estimation"
    )
    return None


# Test that the replacement of z=0 with a small value works
def test_estimateDeltaStaeckel_z_is_0():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.potential import MWPotential2014

    # Test that z=0 works for a single value
    n = 11
    rs = numpy.linspace(0.1, 10.0, n)
    for r in rs:
        delta0 = estimateDeltaStaeckel(MWPotential2014, r, 0.0)
        deltasmall = estimateDeltaStaeckel(MWPotential2014, r, 5e-4)
        assert numpy.fabs(delta0 - deltasmall) < 1e-3, (
            "Delta computed with z=0 does not agree with that computed for small z"
        )
    # And an array
    delta0 = estimateDeltaStaeckel(MWPotential2014, rs, numpy.zeros(n))
    deltasmall = estimateDeltaStaeckel(MWPotential2014, rs, 5e-4 * numpy.ones(n))
    assert numpy.all(numpy.fabs(delta0 - deltasmall) < 1e-3), (
        "Delta computed with array of z=0 does not agree with that computed for array of small z"
    )


def test_actionAngleStaeckel_indivdelta_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # actions with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=True)
    jr0, jp0, jz0 = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=True)
    jr1, jp1, jz1 = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with individual delta
    jri, jpi, jzi = aAS(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2]), delta=deltas
    )
    # Check that they agree as expected
    assert numpy.fabs(jr0[0] - jri[0]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jr1[1] - jri[1]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz0[0] - jzi[0]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz1[1] - jzi[1]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


def test_actionAngleStaeckel_indivdelta_freqs_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # actions with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=True)
    jr0, jp0, jz0, or0, op0, oz0 = aAS.actionsFreqs(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=True)
    jr1, jp1, jz1, or1, op1, oz1 = aAS.actionsFreqs(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
    )
    # actions with individual delta
    jri, jpi, jzi, ori, opi, ozi = aAS.actionsFreqs(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
        delta=deltas,
    )
    # Check that they agree as expected
    assert numpy.fabs(jr0[0] - jri[0]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jr1[1] - jri[1]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz0[0] - jzi[0]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz1[1] - jzi[1]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(or0[0] - ori[0]) < 1e-10, (
        "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(or1[1] - ori[1]) < 1e-10, (
        "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(op0[0] - opi[0]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(op1[1] - opi[1]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(oz0[0] - ozi[0]) < 1e-10, (
        "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(oz1[1] - ozi[1]) < 1e-10, (
        "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


def test_actionAngleStaeckel_indivdelta_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # actions with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=True)
    jr0, jp0, jz0, or0, op0, oz0, ar0, ap0, az0 = aAS.actionsFreqsAngles(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=True)
    jr1, jp1, jz1, or1, op1, oz1, ar1, ap1, az1 = aAS.actionsFreqsAngles(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
    )
    # actions with individual delta
    jri, jpi, jzi, ori, opi, ozi, ari, api, azi = aAS.actionsFreqsAngles(
        o.R(ts[:2]),
        o.vR(ts[:2]),
        o.vT(ts[:2]),
        o.z(ts[:2]),
        o.vz(ts[:2]),
        o.phi(ts[:2]),
        delta=deltas,
    )
    # Check that they agree as expected
    assert numpy.fabs(jr0[0] - jri[0]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jr1[1] - jri[1]) < 1e-10, (
        "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz0[0] - jzi[0]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(jz1[1] - jzi[1]) < 1e-10, (
        "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(or0[0] - ori[0]) < 1e-10, (
        "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(or1[1] - ori[1]) < 1e-10, (
        "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(op0[0] - opi[0]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(op1[1] - opi[1]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(oz0[0] - ozi[0]) < 1e-10, (
        "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(oz1[1] - ozi[1]) < 1e-10, (
        "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ar0[0] - ari[0]) < 1e-10, (
        "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ar1[1] - ari[1]) < 1e-10, (
        "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ap0[0] - api[0]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ap1[1] - api[1]) < 1e-10, (
        "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(az0[0] - azi[0]) < 1e-10, (
        "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(az1[1] - azi[1]) < 1e-10, (
        "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


def test_actionAngleStaeckel_indivdelta_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=False)
    e0, z0, rp0, ra0 = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=False)
    e1, z1, rp1, ra1 = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with individual delta
    ei, zi, rpi, rai = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2]), delta=deltas
    )
    # Check that they agree as expected
    assert numpy.fabs(e0[0] - ei[0]) < 1e-10, (
        "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(e1[1] - ei[1]) < 1e-10, (
        "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(z0[0] - zi[0]) < 1e-10, (
        "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(z1[1] - zi[1]) < 1e-10, (
        "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(rp0[0] - rpi[0]) < 1e-10, (
        "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(rp1[1] - rpi[1]) < 1e-10, (
        "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ra0[0] - rai[0]) < 1e-10, (
        "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ra1[1] - rai[1]) < 1e-10, (
        "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


def test_actionAngleStaeckel_indivdelta_EccZmaxRperiRap_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Briefly integrate orbit to get multiple points
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.25, 1.0])
    ts = numpy.linspace(0.0, 1.0, 101)
    o.integrate(ts, MWPotential2014)
    deltas = [0.2, 0.4]
    # with one delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[0], c=True)
    e0, z0, rp0, ra0 = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with another delta
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=deltas[1], c=True)
    e1, z1, rp1, ra1 = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2])
    )
    # actions with individual delta
    ei, zi, rpi, rai = aAS.EccZmaxRperiRap(
        o.R(ts[:2]), o.vR(ts[:2]), o.vT(ts[:2]), o.z(ts[:2]), o.vz(ts[:2]), delta=deltas
    )
    # Check that they agree as expected
    assert numpy.fabs(e0[0] - ei[0]) < 1e-10, (
        "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(e1[1] - ei[1]) < 1e-10, (
        "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(z0[0] - zi[0]) < 1e-10, (
        "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(z1[1] - zi[1]) < 1e-10, (
        "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(rp0[0] - rpi[0]) < 1e-10, (
        "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(rp1[1] - rpi[1]) < 1e-10, (
        "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ra0[0] - rai[0]) < 1e-10, (
        "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    assert numpy.fabs(ra1[1] - rai[1]) < 1e-10, (
        "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    )
    return None


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=False, delta=0.71)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    check_actionAngle_conserved_actions(
        aAS, obs, MWPotential, -2.0, -8.0, -2.0, ntimes=101
    )
    return None


# Test the actions of an actionAngleStaeckel, more eccentric orbit
def test_actionAngleStaeckel_conserved_actions_ecc():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=False, delta=0.71)
    obs = Orbit([1.1, 0.2, 1.3, 0.3, 0.0])
    # Jr tol -1.4 (was -1.5): the pure-Python path now uses the C v0=pi/2
    # convention, which conserves this eccentric orbit's Jr to 3.28% (identical
    # to c=True) rather than the v0=vx 3.16%.
    check_actionAngle_conserved_actions(
        aAS, obs, MWPotential, -1.4, -8.0, -1.4, ntimes=101
    )
    return None


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import (
        DiskSCFPotential,
        DoubleExponentialDiskPotential,
        FlattenedPowerPotential,
        KeplerPotential,
        KuzminDiskPotential,
        KuzminLikeWrapperPotential,
        MWPotential,
        OblateStaeckelWrapperPotential,
        PerfectEllipsoidPotential,
        PowerTriaxialPotential,
        SCFPotential,
        TriaxialGaussianPotential,
        TriaxialHernquistPotential,
        TriaxialJaffePotential,
        TriaxialNFWPotential,
        TwoPowerTriaxialPotential,
        interpRZPotential,
    )

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
    )
    pots = [
        MWPotential,
        DoubleExponentialDiskPotential(normalize=1.0),
        FlattenedPowerPotential(normalize=1.0),
        FlattenedPowerPotential(normalize=1.0, alpha=0.0),
        KuzminDiskPotential(normalize=1.0, a=1.0 / 8.0),
        TriaxialHernquistPotential(
            normalize=1.0, c=0.2, pa=1.1
        ),  # tests rot, but not well
        TriaxialNFWPotential(normalize=1.0, c=0.3, pa=1.1),
        TriaxialJaffePotential(normalize=1.0, c=0.4, pa=1.1),
        TwoPowerTriaxialPotential(normalize=1.0, alpha=1.5, beta=3.5, c=0.5, pa=1.1),
        TwoPowerTriaxialPotential(
            normalize=1.0, alpha=2.0, beta=3.5, c=0.5, pa=1.1
        ),  # tests special case alpha=2
        SCFPotential(normalize=1.0),
        DiskSCFPotential(normalize=1.0),
        ip,
        PerfectEllipsoidPotential(normalize=1.0, c=0.98),
        TriaxialGaussianPotential(normalize=1.0, c=0.98),
        PowerTriaxialPotential(normalize=1.0, c=0.98),
        OblateStaeckelWrapperPotential(pot=MWPotential, delta=0.71, u0=1.0),
        KuzminLikeWrapperPotential(pot=KeplerPotential(normalize=1.0), a=0.7, b=0.01),
    ]
    for pot in pots:
        aAS = actionAngleStaeckel(pot=pot, c=True, delta=0.71)
        obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
        if not ext_loaded:  # odeint is not as accurate as dopr54_c
            check_actionAngle_conserved_actions(
                aAS, obs, pot, -1.6, -6.0, -1.6, ntimes=101, inclphi=True
            )
        else:
            check_actionAngle_conserved_actions(
                aAS, obs, pot, -1.6, -8.0, -1.65, ntimes=101, inclphi=True
            )
    return None


# Regression test for the exact-mode evaluation cache of
# OblateStaeckelWrapperPotential in C: the cache is stateful per parsed
# potentialArg, so it relies on every actionAngle C entry point handing each
# OpenMP thread its own parsed copy; a shared copy shows up as nondeterministic,
# wrong results (torn cache doubles). Check that all six C entry points are
# deterministic under repetition with enough points to engage multiple threads,
# that the cached exact mode agrees with the cache-free tabulated mode, and
# that the actions agree with pure Python
def test_actionAngle_oblatestaeckelwrapper_cache_c():
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleStaeckel
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45)
    swptab = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45, ntab=3000)
    n = 64
    R = numpy.linspace(0.8, 1.2, n)
    vR = 0.15 * numpy.sin(9.0 * R)
    vT = 1.05 + 0.1 * numpy.cos(7.0 * R)
    z = 0.1 * numpy.sin(13.0 * R)
    vz = 0.05 * numpy.cos(11.0 * R)
    phi = numpy.linspace(0.0, 2.0 * numpy.pi, n)
    all_results = []
    for pot in (swp, swptab):
        aAS = actionAngleStaeckel(pot=pot, delta=0.45, c=True)
        aAA = actionAngleAdiabatic(pot=pot, c=True)
        results = []
        for _ in range(2):
            out = []
            out.extend(aAS(R, vR, vT, z, vz))
            out.extend(aAS.actionsFreqs(R, vR, vT, z, vz))
            out.extend(aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi))
            out.extend(aAS.EccZmaxRperiRap(R, vR, vT, z, vz))
            out.extend(aAA(R, vR, vT, z, vz))
            out.extend(aAA.EccZmaxRperiRap(R, vR, vT, z, vz))
            results.append(out)
        for first, second in zip(*results):
            assert numpy.all(first == second), (
                "OblateStaeckelWrapperPotential C actionAngle evaluation is not "
                "deterministic under repetition; the exact-mode cache is likely "
                "racing between OpenMP threads"
            )
        all_results.append(results[0])
    # residual difference is tabulation truncation (the frequencies lean on the
    # tabulated second derivatives); a racing cache errs at the percent level
    for exact, tab in zip(*all_results):
        assert numpy.amax(numpy.fabs(exact - tab)) < 1e-4, (
            "Cached exact-mode and cache-free tabulated-mode "
            "OblateStaeckelWrapperPotential disagree in C actionAngle evaluation"
        )
    aASpy = actionAngleStaeckel(pot=swp, delta=0.45, c=False)
    sub = slice(0, n, 16)
    jpy = aASpy(R[sub], vR[sub], vT[sub], z[sub], vz[sub])
    for jc, jp in zip(all_results[0][:3], jpy):
        assert numpy.amax(numpy.fabs(jc[sub] - jp)) < 1e-4, (
            "C and Python actions of the cached OblateStaeckelWrapperPotential disagree"
        )
    return None


# Test the actions of an actionAngleStaeckel, for a dblexp disk far away from the center
def test_actionAngleStaeckel_conserved_actions_c_specialdblexp():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import DoubleExponentialDiskPotential

    pot = DoubleExponentialDiskPotential(normalize=1.0)
    aAS = actionAngleStaeckel(pot=pot, c=True, delta=0.01)
    # Close to circular in the Keplerian regime
    obs = Orbit([7.05, 0.002, pot.vcirc(7.05), 0.003, 0.0, 2.0])
    check_actionAngle_conserved_actions(
        aAS, obs, pot, -2.0, -7.0, -2.0, ntimes=101, inclphi=True
    )
    return None


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_wSpherical_conserved_actions_c():
    from test_potential import (
        mockGaussianAmplitudeSmoothedLogarithmicHaloPotential,
        mockSCFZeeuwPotential,
        mockSmoothedLogarithmicHaloPotential,
        mockSmoothedLogarithmicHaloPotentialwTimeDependentAmplitudeWrapperPotential,
        mockSphericalSoftenedNeedleBarPotential,
    )

    from galpy import potential
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded

    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    lpb = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0, b=1.0)  # same |^
    hp = potential.HernquistPotential(normalize=1.0)
    jp = potential.JaffePotential(normalize=1.0)
    np = potential.NFWPotential(normalize=1.0)
    etnp = potential.ExpTruncNFWPotential(normalize=1.0)
    ip = potential.IsochronePotential(normalize=1.0, b=1.0)
    pp = potential.PowerSphericalPotential(normalize=1.0)
    lp2 = potential.PowerSphericalPotential(normalize=1.0, alpha=2.0)
    ppc = potential.PowerSphericalPotentialwCutoff(normalize=1.0)
    plp = potential.PlummerPotential(normalize=1.0)
    psp = potential.PseudoIsothermalPotential(normalize=1.0)
    bp = potential.BurkertPotential(normalize=1.0)
    scfp = potential.SCFPotential(normalize=1.0)
    scfzp = mockSCFZeeuwPotential()
    scfzp.normalize(1.0)
    msoftneedlep = mockSphericalSoftenedNeedleBarPotential()
    msmlp = mockSmoothedLogarithmicHaloPotential()
    mgasmlp = mockGaussianAmplitudeSmoothedLogarithmicHaloPotential()
    dp = potential.DehnenSphericalPotential(normalize=1.0)
    dcp = potential.DehnenCoreSphericalPotential(normalize=1.0)
    homp = potential.HomogeneousSpherePotential(normalize=1.0)
    ihomp = potential.interpSphericalPotential(
        rforce=potential.HomogeneousSpherePotential(normalize=1.0, R=1.1),
        rgrid=numpy.linspace(0.0, 1.1, 201),
    )
    ep = potential.EinastoPotential(normalize=1.0, h=2.2)
    tpsp = potential.TwoPowerSphericalPotential(normalize=1.0, alpha=1.5, beta=3.5)
    tpsp_beta3 = potential.TwoPowerSphericalPotential(
        normalize=1.0, alpha=1.5, beta=3.0
    )
    msmlpwtdp = (
        mockSmoothedLogarithmicHaloPotentialwTimeDependentAmplitudeWrapperPotential()
    )
    mep = potential.MultipoleExpansionPotential.from_density(
        dens=potential.HernquistPotential(normalize=1.0),
        L=6,
        symmetry="spherical",
        normalize=1.0,
    )
    mep_nonaxi = potential.MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi: (
            potential.HernquistPotential(normalize=1.0).dens(R, z, phi)
            * (1.0 + 1e-9 * numpy.cos(phi))
        ),
        L=2,
        symmetry=None,
        normalize=1.0,
    )
    mep_tdep_nonaxi_m3 = potential.MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            potential.HernquistPotential(normalize=1.0).dens(R, z, phi)
            * (1.0 + 1e-9 * numpy.cos(phi + 1.3 * t))
        ),
        L=3,
        symmetry=None,
        normalize=1.0,
        rgrid=numpy.geomspace(1e-3, 50, 51),
        tgrid=numpy.linspace(0, 300, 11),
    )
    scf_tdep_nonaxi_m3 = potential.SCFPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            potential.HernquistPotential(normalize=1.0).dens(R, z, phi)
            * (1.0 + 1e-9 * numpy.cos(phi + 1.3 * t))
        ),
        N=10,
        L=3,
        symmetry=None,
        tgrid=numpy.linspace(0, 300, 11),
    )
    scf_tdep_nonaxi_m3.normalize(1.0)
    pots = [
        lp,
        lpb,
        hp,
        jp,
        np,
        etnp,
        ip,
        pp,
        lp2,
        ppc,
        plp,
        psp,
        bp,
        scfp,
        scfzp,
        msoftneedlep,
        msmlp,
        mgasmlp,
        dp,
        dcp,
        homp,
        ihomp,
        msmlpwtdp,
        ep,
        tpsp,
        tpsp_beta3,
        mep,
        mep_nonaxi,
        mep_tdep_nonaxi_m3,
        scf_tdep_nonaxi_m3,
    ]
    for pot in pots:
        aAS = actionAngleStaeckel(pot=pot, c=True, delta=0.01)
        obs = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
        if not ext_loaded:  # odeint is not as accurate as dopr54_c
            check_actionAngle_conserved_actions(
                aAS, obs, pot, -2.0, -5.0, -2.0, ntimes=101, inclphi=True
            )
        else:
            check_actionAngle_conserved_actions(
                aAS, obs, pot, -2.0, -8.0, -2.0, ntimes=101, inclphi=True
            )
    return None


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_actions_fixed_quad():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=False, delta=0.71)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    if not ext_loaded:  # odeint is not as accurate as dopr54_c
        check_actionAngle_conserved_actions(
            aAS,
            obs,
            MWPotential,
            -2.0,
            -5.0,
            -2.0,
            ntimes=101,
            fixed_quad=True,
            inclphi=True,
        )
    else:
        check_actionAngle_conserved_actions(
            aAS,
            obs,
            MWPotential,
            -2.0,
            -8.0,
            -2.0,
            ntimes=101,
            fixed_quad=True,
            inclphi=True,
        )
    return None


# Test that the angles of an actionAngleStaeckel increase linearly
def test_actionAngleStaeckel_linear_angles():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    check_actionAngle_linear_angles(
        aAS,
        obs,
        MWPotential,
        -2.0,
        -4.0,
        -3.0,
        -3.0,
        -3.0,
        -2.0,
        -2.0,
        -3.5,
        -2.0,
        ntimes=1001,
    )  # need fine sampling for de-period
    return None


# Test that the angles of an actionAngleStaeckel increase linearly, interppot
def test_actionAngleStaeckel_linear_angles_interppot():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
    )
    aAS = actionAngleStaeckel(pot=ip, delta=0.71, c=True)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    check_actionAngle_linear_angles(
        aAS,
        obs,
        MWPotential,
        -2.0,
        -4.0,
        -3.0,
        -3.0,
        -3.0,
        -2.0,
        -2.0,
        -3.5,
        -2.0,
        ntimes=1001,
    )  # need fine sampling for de-period
    return None


# Test that the angles of an actionAngleStaeckel increase linearly
def test_actionAngleStaeckel_linear_angles_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True, useu0=True)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    check_actionAngle_linear_angles(
        aAS,
        obs,
        MWPotential,
        -2.0,
        -4.0,
        -3.0,
        -3.0,
        -3.0,
        -2.0,
        -2.0,
        -3.5,
        -2.0,
        ntimes=1001,
    )  # need fine sampling for de-period
    # specifying u0
    check_actionAngle_linear_angles(
        aAS,
        obs,
        MWPotential,
        -2.0,
        -4.0,
        -3.0,
        -3.0,
        -3.0,
        -2.0,
        -2.0,
        -3.5,
        -2.0,
        ntimes=1001,
        u0=1.23,
    )  # need fine sampling for de-period
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=False, delta=0.71)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 0.0])
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAS, obs, MWPotential, -2.0, -2.0, -2.0, -2.0, ntimes=101
    )
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap_ecc():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, c=False, delta=0.71)
    obs = Orbit([1.1, 0.2, 1.3, 0.3, 0.0, 2.0])
    # ecc/zmax tols loosened (ecc -1.8->-1.7, zmax -1.4->-1.3): the pure-Python
    # path now uses the C v0=pi/2 convention and conserves ecc/zmax to 1.58%/4.15%
    # (identical to c=True), vs the v0=vx values the old tols were set for.
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAS, obs, MWPotential, -1.7, -1.3, -1.8, -1.8, ntimes=101, inclphi=True
    )
    return None


# Test the conservation of ecc, zmax, rperi, rap of an actionAngleStaeckel
def test_actionAngleStaeckel_conserved_EccZmaxRperiRap_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import (
        DiskSCFPotential,
        DoubleExponentialDiskPotential,
        FlattenedPowerPotential,
        KeplerPotential,
        KuzminDiskPotential,
        KuzminLikeWrapperPotential,
        MWPotential,
        PerfectEllipsoidPotential,
        SCFPotential,
        TriaxialHernquistPotential,
        TriaxialJaffePotential,
        TriaxialNFWPotential,
        TwoPowerTriaxialPotential,
        interpRZPotential,
    )

    ip = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 101),
        zgrid=(0.0, 1.0, 101),
        logR=True,
        use_c=True,
        enable_c=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
    )
    pots = [
        MWPotential,
        DoubleExponentialDiskPotential(normalize=1.0),
        FlattenedPowerPotential(normalize=1.0),
        FlattenedPowerPotential(normalize=1.0, alpha=0.0),
        KuzminDiskPotential(normalize=1.0, a=1.0 / 8.0),
        TriaxialHernquistPotential(
            normalize=1.0, c=0.2, pa=1.1
        ),  # tests rot, but not well
        TriaxialNFWPotential(normalize=1.0, c=0.3, pa=1.1),
        TriaxialJaffePotential(normalize=1.0, c=0.4, pa=1.1),
        TwoPowerTriaxialPotential(normalize=1.0, alpha=1.5, beta=3.5, c=0.5, pa=1.1),
        TwoPowerTriaxialPotential(
            normalize=1.0, alpha=2.0, beta=3.5, c=0.5, pa=1.1
        ),  # tests special case alpha=2
        SCFPotential(normalize=1.0),
        DiskSCFPotential(normalize=1.0),
        ip,
        PerfectEllipsoidPotential(normalize=1.0, c=0.98),
        KuzminLikeWrapperPotential(pot=KeplerPotential(normalize=1.0), a=0.7, b=0.01),
    ]
    for pot in pots:
        aAS = actionAngleStaeckel(pot=pot, c=True, delta=0.71)
        obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
        check_actionAngle_conserved_EccZmaxRperiRap(
            aAS, obs, pot, -1.8, -1.3, -1.8, -1.8, ntimes=101
        )
    return None


# Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions():
    from galpy.actionAngle import (
        actionAngleIsochrone,
        actionAngleStaeckel,
        estimateDeltaStaeckel,
    )
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleStaeckel(pot=ip, c=False, delta=0.1)  # not ideal
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions_fixed_quad():
    from galpy.actionAngle import (
        actionAngleIsochrone,
        actionAngleStaeckel,
        estimateDeltaStaeckel,
    )
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleStaeckel(pot=ip, c=False, delta=0.1)  # not ideal
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi, fixed_quad=True)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])[0]
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])[0]
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])[0]
    assert djr < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckel_otherIsochrone_actions_c():
    from galpy.actionAngle import (
        actionAngleIsochrone,
        actionAngleStaeckel,
        estimateDeltaStaeckel,
    )
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleStaeckel(pot=ip, c=True, delta=0.1)  # not ideal
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-3.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleStaeckel against an isochrone potential: frequencies
def test_actionAngleStaeckel_otherIsochrone_freqs():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleStaeckel
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleStaeckel(pot=ip, delta=0.1, c=True)
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    jiO = aAI.actionsFreqs(R, vR, vT, z, vz, phi)
    jiaO = aAS.actionsFreqs(R, vR, vT, z, vz, phi)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-5.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Or at %g%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-5.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Op at %g%%"
        % (dOp * 100.0)
    )
    assert dOz < 1.5 * 10.0**-4.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Oz at %g%%"
        % (dOz * 100.0)
    )
    return None


# Test the actionAngleStaeckel against an isochrone potential: angles
def test_actionAngleStaeckel_otherIsochrone_angles():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleStaeckel
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAS = actionAngleStaeckel(pot=ip, delta=0.1, c=True)
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.03, -0.01, 2.0
    jiO = aAI.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    jiaO = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-4.0, (
        "actionAngleStaeckel applied to isochrone potential fails for ar at %g%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-6.0, (
        "actionAngleStaeckel applied to isochrone potential fails for ap at %g%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-4.0, (
        "actionAngleStaeckel applied to isochrone potential fails for az at %g%%"
        % (daz * 100.0)
    )
    return None


# Test that actionAngleStaeckel at very small u works okay
def test_actionAngleStaeckel_smallu():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, vcirc

    aAS = actionAngleStaeckel(pot=MWPotential2014, c=False, delta=0.45)

    rmin = 8e-9
    o = Orbit([rmin, 0.0, vcirc(MWPotential2014, rmin) / 20, 1e-8, 0.0])
    ezrpra = aAS.EccZmaxRperiRap(o)
    # Check that rperi is close to zero
    assert numpy.fabs(ezrpra[2]) < 1e-8, (
        "actionAngleStaeckel at very small u does not give rperi=0"
    )
    return None


# Basic sanity checking of the actionAngleStaeckelGrid actions (incl. conserved and ecc etc., bc takes a lot of time)
def test_actionAngleStaeckelGrid_basicAndConserved_actions():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleStaeckelGrid(
        pot=MWPotential, delta=0.71, c=False, nLz=20, interpecc=True
    )
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    assert numpy.fabs(aAA.JR(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(aAA.Jz(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jz"
    )
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Check that actions are conserved along the orbit
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.2, -8.0, -1.7, ntimes=101
    )
    # and the eccentricity etc.
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, MWPotential, -2.0, -2.0, -2.0, -2.0, ntimes=101
    )
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckelGrid_basic_actions_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    rzpot = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(0.0, 1.0, 101),
        interpPot=True,
        use_c=True,
        enable_c=True,
        zsym=True,
    )
    aAA = actionAngleStaeckelGrid(pot=rzpot, delta=0.71, c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    assert numpy.fabs(js[0]) < 10.0**-8.0, (
        "Circular orbit in the MWPotential does not have Jr=0"
    )
    assert numpy.fabs(js[2]) < 10.0**-8.0, (
        "Circular orbit in the MWPotential does not have Jz=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(js[0]) < 10.0**-4.0, (
        "Close-to-circular orbit in the MWPotential does not have small Jr"
    )
    assert numpy.fabs(js[2]) < 10.0**-3.0, (
        "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    )


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckelGrid_conserved_actions_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleStaeckelGrid(pot=MWPotential, delta=0.71, c=True)
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.4, -8.0, -1.7, ntimes=101
    )
    return None


# Test the setup of an actionAngleStaeckelGrid
def test_actionAngleStaeckelGrid_setuperrs():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.potential import MWPotential

    try:
        aAA = actionAngleStaeckelGrid()
    except OSError:
        pass
    else:
        raise AssertionError("actionAngleStaeckelGrid w/o pot does not give IOError")
    try:
        aAA = actionAngleStaeckelGrid(pot=MWPotential)
    except OSError:
        pass
    else:
        raise AssertionError("actionAngleStaeckelGrid w/o delta does not give IOError")
    return None


# Test the actionAngleStaeckel against an isochrone potential: actions
def test_actionAngleStaeckelGrid_Isochrone_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleStaeckelGrid
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAA = actionAngleStaeckelGrid(pot=ip, delta=0.1, c=True)
    R, vR, vT, z, vz, phi = 1.01, 0.05, 1.05, 0.05, 0.0, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-1.2, (
        "actionAngleStaeckel applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleStaeckel applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-1.2, (
        "actionAngleStaeckel applied to isochrone potential fails for Jz at %f%%"
        % (djz * 100.0)
    )
    return None


# Basic sanity checking of the actionAngleStaeckelGrid eccentricity etc.
def test_actionAngleStaeckelGrid_basic_EccZmaxRperiRap_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, interpRZPotential

    rzpot = interpRZPotential(
        RZPot=MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(0.0, 1.0, 101),
        interpPot=True,
        use_c=True,
        enable_c=True,
        zsym=True,
    )
    aAA = actionAngleStaeckelGrid(pot=rzpot, delta=0.71, c=True, interpecc=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-16.0, (
        "Circular orbit in the MWPotential does not have zmax=0"
    )
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(Orbit([R, vR, vT, z, vz]))
    assert numpy.fabs(te) < 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    )
    assert numpy.fabs(tzmax) < 2.0 * 10.0**-2.0, (
        "Close-to-circular orbit in the MWPotential does not have small zmax"
    )
    return None


# Test the actions of an actionAngleStaeckel
def test_actionAngleStaeckelGrid_conserved_EccZmaxRperiRap_c():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAA = actionAngleStaeckelGrid(pot=MWPotential, delta=0.71, c=True, interpecc=True)
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAA, obs, MWPotential, -2.0, -2.0, -2.0, -2.0, ntimes=101, inclphi=True
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: actions
def test_actionAngleIsochroneApprox_otherIsochrone_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAIA(R, vR, vT, z, vz, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-2.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    if not ext_loaded:  # odeint is less accurate than dopr54_c
        assert djz < 10.0**-6.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    else:
        assert djz < 10.0**-10.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: frequencies
def test_actionAngleIsochroneApprox_otherIsochrone_freqs():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqs(R, vR, vT, z, vz, phi)
    jiaO = aAIA.actionsFreqs(R, vR, vT, z, vz, phi)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%"
        % (dOz * 100.0)
    )
    # Same with _firstFlip, shouldn't be different bc doesn't do anything for R,vR,... input
    jiaO = aAIA.actionsFreqs(R, vR, vT, z, vz, phi, _firstFlip=True)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%"
        % (dOz * 100.0)
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: angles
def test_actionAngleIsochroneApprox_otherIsochrone_angles():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    jiaO = aAIA.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%"
        % (daz * 100.0)
    )
    # Same with _firstFlip, shouldn't be different bc doesn't do anything for R,vR,... input
    jiaO = aAIA.actionsFreqsAngles(R, vR, vT, z, vz, phi, _firstFlip=True)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%"
        % (daz * 100.0)
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: actions, cumul
def test_actionAngleIsochroneApprox_otherIsochrone_actions_cumul():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    jia = aAIA(R, vR, vT, z, vz, phi, cumul=True)
    djr = numpy.fabs((ji[0] - jia[0][0, -1]) / ji[0])
    djz = numpy.fabs((ji[2] - jia[2][0, -1]) / ji[2])
    assert djr < 10.0**-2.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    if not ext_loaded:  # odeint is less accurate than dopr54_c
        assert djz < 10.0**-6.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    else:
        assert djz < 10.0**-10.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: actions; planarOrbit
def test_actionAngleIsochroneApprox_otherIsochrone_planarOrbit_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, phi = 1.1, 0.3, 1.2, 2.0
    ji = aAI(R, vR, vT, 0.0, 0.0, phi)
    jia = aAIA(R, vR, vT, phi)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    assert djr < 10.0**-2.0, (
        "actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: actions; integrated planarOrbit
def test_actionAngleIsochroneApprox_otherIsochrone_planarOrbit_integratedOrbit_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, phi = 1.1, 0.3, 1.2, 2.0
    ji = aAI(R, vR, vT, 0.0, 0.0, phi)
    o = Orbit([R, vR, vT, phi])
    ts = numpy.linspace(0.0, 250.0, 25000)
    o.integrate(ts, ip)
    jia = aAIA(o)
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    assert djr < 10.0**-2.0, (
        "actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleIsochroneApprox applied to isochrone potential for planarOrbit fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: actions; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_actions():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import ext_loaded
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    ji = aAI(R, vR, vT, z, vz, phi)
    # Setup an orbit, and integrated it first
    o = Orbit([R, vR, vT, z, vz, phi])
    ts = numpy.linspace(0.0, 250.0, 25000)  # Integrate for a long time, not the default
    o.integrate(ts, ip)
    jia = aAIA(o)  # actions, with an integrated orbit
    djr = numpy.fabs((ji[0] - jia[0]) / ji[0])
    dlz = numpy.fabs((ji[1] - jia[1]) / ji[1])
    djz = numpy.fabs((ji[2] - jia[2]) / ji[2])
    assert djr < 10.0**-2.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Jr at %f%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because ip is a spherical potential
    assert dlz < 10.0**-10.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Lz at %f%%"
        % (dlz * 100.0)
    )
    if not ext_loaded:  # odeint is less accurate than dopr54_c
        assert djz < 10.0**-6.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    else:
        assert djz < 10.0**-10.0, (
            "actionAngleIsochroneApprox applied to isochrone potential fails for Jz at %f%%"
            % (djz * 100.0)
        )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: frequencies; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_freqs():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqs(R, vR, vT, z, vz, phi)
    # Setup an orbit, and integrated it first
    o = Orbit([R, vR, vT, z, vz, phi])
    ts = numpy.linspace(0.0, 250.0, 25000)  # Integrate for a long time, not the default
    o.integrate(ts, ip)
    jiaO = aAIA.actionsFreqs([o])  # for list
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%"
        % (dOz * 100.0)
    )
    # Same with specifying ts
    jiaO = aAIA.actionsFreqs(o, ts=ts)
    dOr = numpy.fabs((jiO[3] - jiaO[3]) / jiO[3])
    dOp = numpy.fabs((jiO[4] - jiaO[4]) / jiO[4])
    dOz = numpy.fabs((jiO[5] - jiaO[5]) / jiO[5])
    assert dOr < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Or at %f%%"
        % (dOr * 100.0)
    )
    assert dOp < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Op at %f%%"
        % (dOp * 100.0)
    )
    assert dOz < 10.0**-6.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for Oz at %f%%"
        % (dOz * 100.0)
    )
    return None


# Test the actionAngleIsochroneApprox against an isochrone potential: angles; for an integrated orbit
def test_actionAngleIsochroneApprox_otherIsochrone_integratedOrbit_angles():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    jiO = aAI.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    # Setup an orbit, and integrated it first
    o = Orbit([R, vR, vT, z, vz, phi])
    ts = numpy.linspace(0.0, 250.0, 25000)  # Integrate for a long time, not the default
    o.integrate(ts, ip)
    jiaO = aAIA.actionsFreqsAngles(o)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%"
        % (daz * 100.0)
    )
    # Same with specifying ts
    jiaO = aAIA.actionsFreqsAngles(o, ts=ts)
    dar = numpy.fabs((jiO[6] - jiaO[6]) / jiO[6])
    dap = numpy.fabs((jiO[7] - jiaO[7]) / jiO[7])
    daz = numpy.fabs((jiO[8] - jiaO[8]) / jiO[8])
    assert dar < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ar at %f%%"
        % (dar * 100.0)
    )
    assert dap < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for ap at %f%%"
        % (dap * 100.0)
    )
    assert daz < 10.0**-4.0, (
        "actionAngleIsochroneApprox applied to isochrone potential fails for az at %f%%"
        % (daz * 100.0)
    )
    return None


# Check that actionAngleIsochroneApprox gives the same answer for different setups
def test_actionAngleIsochroneApprox_diffsetups():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    # Different setups
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    aAIip = actionAngleIsochroneApprox(
        pot=lp, ip=IsochronePotential(normalize=1.0, b=0.8)
    )
    aAIaAIip = actionAngleIsochroneApprox(
        pot=lp, aAI=actionAngleIsochrone(ip=IsochronePotential(normalize=1.0, b=0.8))
    )
    aAIrk6 = actionAngleIsochroneApprox(pot=lp, b=0.8, integrate_method="rk6_c")
    aAIlong = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=200.0)
    aAImany = actionAngleIsochroneApprox(pot=lp, b=0.8, ntintJ=20000)
    # Orbit to test on
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    # Actions, frequencies, angles
    acfs = numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsip = numpy.array(list(aAIip.actionsFreqsAngles(obs()))).flatten()
    acfsaAIip = numpy.array(list(aAIaAIip.actionsFreqsAngles(obs()))).flatten()
    acfsrk6 = numpy.array(list(aAIrk6.actionsFreqsAngles(obs()))).flatten()
    acfslong = numpy.array(list(aAIlong.actionsFreqsAngles(obs()))).flatten()
    acfsmany = numpy.array(list(aAImany.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip = numpy.array(
        list(aAI.actionsFreqsAngles(obs(), _firstFlip=True))
    ).flatten()
    # Check that they are the same
    assert numpy.amax(numpy.fabs((acfs - acfsip) / acfs)) < 10.0**-15.0, (
        "actionAngleIsochroneApprox calculated w/ b= and ip= set to the equivalent IsochronePotential do not agree"
    )
    assert numpy.amax(numpy.fabs((acfs - acfsaAIip) / acfs)) < 10.0**-15.0, (
        "actionAngleIsochroneApprox calculated w/ b= and aAI= set to the equivalent IsochronePotential do not agree"
    )
    assert numpy.amax(numpy.fabs((acfs - acfsrk6) / acfs)) < 10.0**-8.0, (
        "actionAngleIsochroneApprox calculated w/ integrate_method=dopr54_c and rk6_c do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfsrk6) / acfs)))
    )
    assert numpy.amax(numpy.fabs((acfs - acfslong) / acfs)) < 10.0**-2.0, (
        "actionAngleIsochroneApprox calculated w/ tintJ=100 and 200 do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfslong) / acfs)))
    )
    assert numpy.amax(numpy.fabs((acfs - acfsmany) / acfs)) < 10.0**-4.0, (
        "actionAngleIsochroneApprox calculated w/ ntintJ=10000 and 20000 do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfsmany) / acfs)))
    )
    assert numpy.amax(numpy.fabs((acfs - acfsfirstFlip) / acfs)) < 10.0**-4.0, (
        "actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfsmany) / acfs)))
    )
    return None


# Check that actionAngleIsochroneApprox gives the same answer w/ and w/o firstFlip
def test_actionAngleIsochroneApprox_firstFlip():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    # Orbit to test on
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    # Actions, frequencies, angles
    acfs = numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip = numpy.array(
        list(aAI.actionsFreqsAngles(obs(), _firstFlip=True))
    ).flatten()
    # Check that they are the same
    assert numpy.amax(numpy.fabs((acfs - acfsfirstFlip) / acfs)) < 10.0**-4.0, (
        "actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfsfirstFlip) / acfs)))
    )
    # Also test that this still works when the orbit was already integrated
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    ts = numpy.linspace(0.0, 250.0, 25000)
    obs.integrate(ts, lp)
    acfs = numpy.array(list(aAI.actionsFreqsAngles(obs()))).flatten()
    acfsfirstFlip = numpy.array(
        list(aAI.actionsFreqsAngles(obs(), _firstFlip=True))
    ).flatten()
    # Check that they are the same
    assert numpy.amax(numpy.fabs((acfs - acfsfirstFlip) / acfs)) < 10.0**-4.0, (
        "actionAngleIsochroneApprox calculated w/ _firstFlip and w/o do not agree at %g%%"
        % (100.0 * numpy.amax(numpy.fabs((acfs - acfsfirstFlip) / acfs)))
    )
    return None


# Test the actionAngleIsochroneApprox used in Bovy (2014)
def test_actionAngleIsochroneApprox_bovy14():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    times = numpy.linspace(0.0, 100.0, 51)
    obs.integrate(times, lp, method="dopr54_c")
    js = aAI(
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    maxdj = numpy.amax(
        numpy.fabs(js - numpy.tile(numpy.mean(js, axis=1), (len(times), 1)).T), axis=1
    ) / numpy.mean(js, axis=1)
    assert maxdj[0] < 3.0 * 10.0**-2.0, (
        "Jr conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%"
        % (100.0 * maxdj[0])
    )
    assert maxdj[1] < 10.0**-2.0, (
        "Lz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%"
        % (100.0 * maxdj[1])
    )
    assert maxdj[2] < 2.0 * 10.0**-2.0, (
        "Jz conservation for the GD-1 like orbit of Bovy (2014) fails at %f%%"
        % (100.0 * maxdj[2])
    )
    return None


# Test the actionAngleIsochroneApprox for a triaxial potential
def test_actionAngleIsochroneApprox_triaxialnfw_conserved_actions():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import TriaxialNFWPotential

    tnp = TriaxialNFWPotential(b=0.9, c=0.8, normalize=1.0)
    aAI = actionAngleIsochroneApprox(pot=tnp, b=0.8, tintJ=200.0)
    obs = Orbit([1.0, 0.2, 1.1, 0.1, 0.1, 0.0])
    check_actionAngle_conserved_actions(
        aAI, obs, tnp, -1.7, -2.0, -1.7, ntimes=51, inclphi=True
    )
    return None


def test_actionAngleIsochroneApprox_triaxialnfw_linear_angles():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import TriaxialNFWPotential

    tnp = TriaxialNFWPotential(b=0.9, c=0.8, normalize=1.0)
    aAI = actionAngleIsochroneApprox(pot=tnp, b=0.8, tintJ=200.0)
    obs = Orbit([1.0, 0.2, 1.1, 0.1, 0.1, 0.0])
    check_actionAngle_linear_angles(
        aAI,
        obs,
        tnp,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -5.0,
        -4.0,
        -4.0,
        -4.0,
        separate_times=True,
        maxt=4.0,
        ntimes=51,
    )  # quick, essentially tests that nothing is grossly wrong
    return None


def test_actionAngleIsochroneApprox_plotting():
    from matplotlib import pyplot

    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    # Various plots that should be produced
    aAI.plot(obs)
    aAI.plot(obs, type="jr")
    aAI.plot(
        numpy.reshape(obs.R(obs.t), (1, len(obs.t))),
        numpy.reshape(obs.vR(obs.t), (1, len(obs.t))),
        numpy.reshape(obs.vT(obs.t), (1, len(obs.t))),
        numpy.reshape(obs.z(obs.t), (1, len(obs.t))),
        numpy.reshape(obs.vz(obs.t), (1, len(obs.t))),
        numpy.reshape(obs.phi(obs.t), (1, len(obs.t))),
        type="lz",
    )
    aAI.plot(obs, type="jz")
    aAI.plot(obs, type="jr", downsample=True)
    aAI.plot(obs, type="lz", downsample=True)
    aAI.plot(obs, type="jz", downsample=True)
    aAI.plot(obs, type="araz")
    aAI.plot(obs, type="araz", downsample=True)
    aAI.plot(obs, type="araz", deperiod=True)
    aAI.plot(obs, type="araphi", deperiod=True)
    aAI.plot(obs, type="azaphi", deperiod=True)
    aAI.plot(obs, type="araphi", deperiod=True, downsample=True)
    aAI.plot(obs, type="azaphi", deperiod=True, downsample=True)
    # With integrated orbit, just to make sure we're covering this
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    obs.integrate(numpy.linspace(0.0, 200.0, 20001), lp)
    aAI.plot(obs, type="jr")
    pyplot.close("all")
    return None


# Test the Orbit interface
def test_orbit_interface_spherical():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, NFWPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    obs = Orbit([1.0, 0.2, 1.5, 0.3, 0.1, 2.0])
    # resetaA has been deprecated
    # assert not obs.resetaA(), 'obs.resetaA() does not return False when called before having set up an actionAngle instance'
    aAS = actionAngleSpherical(pot=lp)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "spherical"
    try:
        obs.jr(type=type)
    except AttributeError:
        pass  # should raise this, as we have not specified a potential
    else:
        raise AssertionError(
            "obs.jr w/o pot= does not raise AttributeError before the orbit was integrated"
        )
    acfso = numpy.array(
        [
            obs.jr(pot=lp, type=type),
            obs.jp(pot=lp, type=type),
            obs.jz(pot=lp, type=type),
            obs.Or(pot=lp, type=type),
            obs.Op(pot=lp, type=type),
            obs.Oz(pot=lp, type=type),
            obs.wr(pot=lp, type=type),
            obs.wp(pot=lp, type=type),
            obs.wz(pot=lp, type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface"
    )
    assert (
        numpy.abs(obs.Tr(pot=lp, type=type) - 2.0 * numpy.pi / acfs[3]) < 10.0**-16.0
    ), "Orbit.Tr does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(obs.Tp(pot=lp, type=type) - 2.0 * numpy.pi / acfs[4]) < 10.0**-16.0
    ), "Orbit.Tp does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(obs.Tz(pot=lp, type=type) - 2.0 * numpy.pi / acfs[5]) < 10.0**-16.0
    ), "Orbit.Tz does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(obs.TrTp(pot=lp, type=type) - acfs[4] / acfs[3] * numpy.pi)
        < 10.0**-16.0
    ), "Orbit.TrTp does not agree with actionAngleSpherical frequency"
    # Different spherical potential
    np = NFWPotential(normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "spherical"
    # resetaA has been deprecated
    # assert obs.resetaA(pot=np), 'obs.resetaA() does not return True after having set up an actionAngle instance'
    obs.integrate(
        numpy.linspace(0.0, 1.0, 11), np
    )  # to test that not specifying the potential works
    acfso = numpy.array(
        [
            obs.jr(type=type),
            obs.jp(type=type),
            obs.jz(type=type),
            obs.Or(type=type),
            obs.Op(type=type),
            obs.Oz(type=type),
            obs.wr(type=type),
            obs.wp(type=type),
            obs.wz(type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface"
    )
    # Directly test _resetaA --> deprecated
    # assert obs._orb._resetaA(pot=lp), 'OrbitTop._resetaA does not return True when resetting the actionAngle instance'
    # Test that unit conversions to physical units are handled correctly
    ro, vo = 8.0, 220.0
    obs = Orbit([1.0, 0.2, 1.5, 0.3, 0.1, 2.0], ro=ro, vo=vo)
    aAS = actionAngleSpherical(pot=lp)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "spherical"
    acfso = numpy.array(
        [
            obs.jr(pot=lp, type=type) / ro / vo,
            obs.jp(pot=lp, type=type) / ro / vo,
            obs.jz(pot=lp, type=type) / ro / vo,
            obs.Or(pot=lp, type=type) / vo * ro / 1.0227121655399913,
            obs.Op(pot=lp, type=type) / vo * ro / 1.0227121655399913,
            obs.Oz(pot=lp, type=type) / vo * ro / 1.0227121655399913,
            obs.wr(pot=lp, type=type),
            obs.wp(pot=lp, type=type),
            obs.wz(pot=lp, type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-9.0, (
        "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface when using physical coordinates"
    )
    assert (
        numpy.abs(
            obs.Tr(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[3]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tr does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(
            obs.Tp(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[4]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tp does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(
            obs.Tz(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[5]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tz does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(obs.TrTp(pot=lp, type=type) - acfs[4] / acfs[3] * numpy.pi)
        < 10.0**-8.0
    ), (
        "Orbit.TrTp does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    # Test frequency in km/s/kpc
    assert (
        numpy.abs(obs.Or(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[3])
        < 10.0**-8.0
    ), (
        "Orbit.Or does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    )
    assert (
        numpy.abs(obs.Op(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[4])
        < 10.0**-8.0
    ), (
        "Orbit.Op does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    )
    assert (
        numpy.abs(obs.Oz(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[5])
        < 10.0**-8.0
    ), (
        "Orbit.Oz does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    )
    return None


# Test the Orbit interface for actionAngleStaeckel
def test_orbit_interface_staeckel():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "staeckel"
    acfso = numpy.array(
        [
            obs.jr(pot=MWPotential, type=type, delta=0.71),
            obs.jp(pot=MWPotential, type=type, delta=0.71),
            obs.jz(pot=MWPotential, type=type, delta=0.71),
            obs.Or(pot=MWPotential, type=type, delta=0.71),
            obs.Op(pot=MWPotential, type=type, delta=0.71),
            obs.Oz(pot=MWPotential, type=type, delta=0.71),
            obs.wr(pot=MWPotential, type=type, delta=0.71),
            obs.wp(pot=MWPotential, type=type, delta=0.71),
            obs.wz(pot=MWPotential, type=type, delta=0.71),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface"
    )
    return None


# Further tests of the Orbit interface for actionAngleStaeckel
def test_orbit_interface_staeckel_defaultdelta():
    from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    est_delta = estimateDeltaStaeckel(MWPotential2014, obs.R(), obs.z())
    # Just need to trigger delta estimation in orbit
    jr_orb = obs.jr(pot=MWPotential2014, type="staeckel")
    assert numpy.fabs(est_delta - obs._aA._delta) < 1e-10, (
        "Directly estimated delta does not agree with Orbit-interface-estimated delta"
    )
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=est_delta)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "staeckel"
    acfso = numpy.array(
        [
            obs.jr(pot=MWPotential2014, type=type),
            obs.jp(pot=MWPotential2014, type=type),
            obs.jz(pot=MWPotential2014, type=type),
            obs.Or(pot=MWPotential2014, type=type),
            obs.Op(pot=MWPotential2014, type=type),
            obs.Oz(pot=MWPotential2014, type=type),
            obs.wr(pot=MWPotential2014, type=type),
            obs.wp(pot=MWPotential2014, type=type),
            obs.wz(pot=MWPotential2014, type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface"
    )
    return None


def test_orbit_interface_staeckel_PotentialErrors():
    # staeckel approx. w/ automatic delta should fail if delta cannot be found
    from galpy.orbit import Orbit
    from galpy.potential import (
        PotentialError,
        SpiralArmsPotential,
        TwoPowerSphericalPotential,
    )

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])

    # Version of TwoPowerSphericalPotential that does not have R2deriv
    class TwoPowerSphericalPotentialNoR2deriv(TwoPowerSphericalPotential):
        _R2deriv = property()  # turns it off!

    tp = TwoPowerSphericalPotentialNoR2deriv(normalize=1.0, alpha=1.2, beta=2.5)
    # Check that this potential indeed does not have second derivs
    with pytest.raises(PotentialError) as excinfo:
        dummy = tp.R2deriv(1.0, 0.1)
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    # Now check that estimating delta fails
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=tp, type="staeckel")
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "second derivatives" in str(excinfo.value), (
        "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    )
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp, type="staeckel")
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "not axisymmetric" in str(excinfo.value), (
        "Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric"
    )
    return None


def test_orbits_interface_staeckel_PotentialErrors():
    # staeckel approx. w/ automatic delta should fail if delta cannot be found
    from galpy.orbit import Orbit
    from galpy.potential import (
        PotentialError,
        SpiralArmsPotential,
        TwoPowerSphericalPotential,
    )

    obs = Orbit(
        [[1.05, 0.02, 1.05, 0.03, 0.0, 2.0], [1.15, -0.02, 1.02, -0.03, 0.0, 2.0]]
    )

    # Version of TwoPowerSphericalPotential that does not have R2deriv
    class TwoPowerSphericalPotentialNoR2deriv(TwoPowerSphericalPotential):
        _R2deriv = property()  # turns it off!

    tp = TwoPowerSphericalPotentialNoR2deriv(normalize=1.0, alpha=1.2, beta=2.5)
    # Check that this potential indeed does not have second derivs
    with pytest.raises(PotentialError) as excinfo:
        dummy = tp.R2deriv(1.0, 0.1)
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    # Now check that estimating delta fails
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=tp, type="staeckel")
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "second derivatives" in str(excinfo.value), (
        "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    )
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp, type="staeckel")
        pytest.fail(
            "SpiralArms appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "not axisymmetric" in str(excinfo.value), (
        "Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric"
    )
    return None


# Test the Orbit interface for actionAngleAdiabatic
def test_orbit_interface_adiabatic():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAS = actionAngleAdiabatic(pot=MWPotential)
    acfs = numpy.array(list(aAS(obs))).reshape(3)
    type = "adiabatic"
    acfso = numpy.array(
        [
            obs.jr(pot=MWPotential, type=type),
            obs.jp(pot=MWPotential, type=type),
            obs.jz(pot=MWPotential, type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
    )
    return None


def test_orbit_interface_adiabatic_2d():
    # Test with 2D orbit
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 2.0])
    aAS = actionAngleAdiabatic(pot=MWPotential)
    acfs = numpy.array(list(aAS(obs))).reshape(3)
    type = "adiabatic"
    acfso = numpy.array(
        [
            obs.jr(pot=MWPotential, type=type),
            obs.jp(pot=MWPotential, type=type),
            obs.jz(pot=MWPotential, type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
    )
    return None


def test_orbit_interface_adiabatic_2d_2dpot():
    # Test with 2D orbit
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, toPlanarPotential

    obs = Orbit([1.05, 0.02, 1.05, 2.0])
    aAS = actionAngleAdiabatic(pot=toPlanarPotential(MWPotential))
    acfs = numpy.array(list(aAS(obs))).reshape(3)
    type = "adiabatic"
    acfso = numpy.array(
        [
            obs.jr(pot=toPlanarPotential(MWPotential), type=type),
            obs.jp(pot=toPlanarPotential(MWPotential), type=type),
            obs.jz(pot=toPlanarPotential(MWPotential), type=type),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
    )
    return None


def test_orbit_interface_actionAngleIsochroneApprox():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 2.0])
    aAS = actionAngleIsochroneApprox(pot=MWPotential, b=0.8)
    acfs = aAS.actionsFreqsAngles([obs()])
    acfs = numpy.array(acfs).reshape(9)
    type = "isochroneApprox"
    acfso = numpy.array(
        [
            obs.jr(pot=MWPotential, type=type, b=0.8),
            obs.jp(pot=MWPotential, type=type, b=0.8),
            obs.jz(pot=MWPotential, type=type, b=0.8),
            obs.Or(pot=MWPotential, type=type, b=0.8),
            obs.Op(pot=MWPotential, type=type, b=0.8),
            obs.Oz(pot=MWPotential, type=type, b=0.8),
            obs.wr(pot=MWPotential, type=type, b=0.8),
            obs.wp(pot=MWPotential, type=type, b=0.8),
            obs.wz(pot=MWPotential, type=type, b=0.8),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-13.0, (
        "Orbit interface for actionAngleIsochroneApprox does not return the same as actionAngle interface"
    )
    assert (
        numpy.abs(obs.Tr(pot=MWPotential, type=type, b=0.8) - 2.0 * numpy.pi / acfso[3])
        < 10.0**-13.0
    ), "Orbit.Tr does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(obs.Tp(pot=MWPotential, type=type, b=0.8) - 2.0 * numpy.pi / acfso[4])
        < 10.0**-13.0
    ), "Orbit.Tp does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(obs.Tz(pot=MWPotential, type=type, b=0.8) - 2.0 * numpy.pi / acfso[5])
        < 10.0**-13.0
    ), "Orbit.Tz does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(
            obs.TrTp(pot=MWPotential, type=type, b=0.8) - acfso[4] / acfso[3] * numpy.pi
        )
        < 10.0**-13.0
    ), "Orbit.TrTp does not agree with actionAngleIsochroneApprox frequency"
    return None


def test_orbit_interface_unbound_simple_adiabatic_noc():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        [[1.05, 0.02, 1.05, 0.03, 0.0, 2.0], [1.05, 0.02, 10.05, 0.03, 0.0, 2.0]]
    )
    aAAnoc = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    jr, jp, jz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="adiabatic", c=False),
        obs.jp(pot=MWPotential2014, type="adiabatic", c=False),
        obs.jz(pot=MWPotential2014, type="adiabatic", c=False),
        obs.e(pot=MWPotential2014, type="adiabatic", analytic=True, c=False),
        obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True, c=False),
        obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True, c=False),
        obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True, c=False),
    )
    assert numpy.fabs(jr[0] - aAAnoc(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aAAnoc(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - aAAnoc(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(e[0] - aAAnoc.EccZmaxRperiRap(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(zmax[0] - aAAnoc.EccZmaxRperiRap(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rperi[0] - aAAnoc.EccZmaxRperiRap(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rap[0] - aAAnoc.EccZmaxRperiRap(obs[0])[3]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(e[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(zmax[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rperi[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rap[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_simple_adiabatic_c():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        [[1.05, 0.02, 1.05, 0.03, 0.0, 2.0], [1.05, 0.02, 10.05, 0.03, 0.0, 2.0]]
    )
    aAAc = actionAngleAdiabatic(pot=MWPotential2014, c=True)
    jr, jp, jz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="adiabatic", c=True),
        obs.jp(pot=MWPotential2014, type="adiabatic", c=True),
        obs.jz(pot=MWPotential2014, type="adiabatic", c=True),
        obs.e(pot=MWPotential2014, type="adiabatic", analytic=True, c=True),
        obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True, c=True),
        obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True, c=True),
        obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True, c=True),
    )
    # Action tolerances currently 1e-5, because they use C implementations for the
    # direct evaluation, but Python for the Orbit interface
    assert numpy.fabs(jr[0] - aAAc(obs[0])[0]) < 10.0**-5.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aAAc(obs[0])[1]) < 10.0**-5.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - aAAc(obs[0])[2]) < 10.0**-5.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(e[0] - aAAc.EccZmaxRperiRap(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(zmax[0] - aAAc.EccZmaxRperiRap(obs[0])[1]) < 10.0**-5.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rperi[0] - aAAc.EccZmaxRperiRap(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rap[0] - aAAc.EccZmaxRperiRap(obs[0])[3]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(e[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(zmax[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rperi[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rap[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_simple_staeckel_noc():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        [[1.05, 0.02, 1.05, 0.03, 0.0, 2.0], [1.05, 0.02, 10.05, 0.03, 0.0, 2.0]]
    )
    aASnoc = actionAngleStaeckel(pot=MWPotential2014, delta=0.71, c=False)
    jr, jp, jz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71, c=False),
        obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71, c=False),
        obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71, c=False),
        obs.e(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False),
        obs.zmax(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
        ),
        obs.rperi(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
        ),
        obs.rap(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
        ),
    )
    # The Orbit jr/jp/jz interface computes actions through the (now-available)
    # c=False actionsFreqsAngles path, so compare against that same path.
    refjr, _, refjz = aASnoc.actionsFreqsAngles(obs[0])[:3]
    assert numpy.fabs(jr[0] - refjr) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aASnoc(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - refjz) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(e[0] - aASnoc.EccZmaxRperiRap(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(zmax[0] - aASnoc.EccZmaxRperiRap(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rperi[0] - aASnoc.EccZmaxRperiRap(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rap[0] - aASnoc.EccZmaxRperiRap(obs[0])[3]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(e[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(zmax[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rperi[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rap[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_simple_staeckel_c():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        [[1.05, 0.02, 1.05, 0.03, 0.0, 2.0], [1.05, 0.02, 10.05, 0.03, 0.0, 2.0]]
    )
    aASc = actionAngleStaeckel(pot=MWPotential2014, delta=0.71, c=True)
    jr, jp, jz, omr, omp, omz, wr, wp, wz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.Or(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.Op(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.Oz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.wr(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.wp(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.wz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True),
        obs.e(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True),
        obs.zmax(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
        ),
        obs.rperi(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
        ),
        obs.rap(
            pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
        ),
    )
    assert numpy.fabs(jr[0] - aASc(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aASc(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - aASc(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(omr[0] - aASc.actionsFreqs(obs[0])[3]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(omp[0] - aASc.actionsFreqs(obs[0])[4]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(omz[0] - aASc.actionsFreqs(obs[0])[5]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(wr[0] - aASc.actionsFreqsAngles(obs[0])[6]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(wp[0] - aASc.actionsFreqsAngles(obs[0])[7]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(wz[0] - aASc.actionsFreqsAngles(obs[0])[8]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(e[0] - aASc.EccZmaxRperiRap(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(zmax[0] - aASc.EccZmaxRperiRap(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rperi[0] - aASc.EccZmaxRperiRap(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(rap[0] - aASc.EccZmaxRperiRap(obs[0])[3]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(omr[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(omp[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(omz[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(wr[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(wp[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(wz[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_simple_2d_adiabatic():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit([[1.05, 0.02, 1.05, 2.0], [1.05, 0.02, 10.05, 2.0]])
    # in 2D, adiabatic and Staeckel are the same and the same as spherical
    aAS = actionAngleSpherical(pot=MWPotential2014)
    jr, jp, jz = (
        obs.jr(pot=MWPotential2014, type="adiabatic"),
        obs.jp(pot=MWPotential2014, type="adiabatic"),
        obs.jz(pot=MWPotential2014, type="adiabatic"),
    )
    assert numpy.fabs(jr[0] - aAS(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aAS(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - aAS(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_simple_2d_staeckel():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit([[1.05, 0.02, 1.05, 2.0], [1.05, 0.02, 10.05, 2.0]])
    # in 2D, adiabatic and Staeckel are the same and the same as spherical
    aAS = actionAngleSpherical(pot=MWPotential2014)
    jr, jp, jz = (
        obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71),
    )
    assert numpy.fabs(jr[0] - aAS(obs[0])[0]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jp[0] - aAS(obs[0])[1]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.fabs(jz[0] - aAS(obs[0])[2]) < 10.0**-10.0, (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_complexshape_adiabatic():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        numpy.array(
            [
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
            ]
        )
    )
    aAA = actionAngleAdiabatic(pot=MWPotential2014)
    jr, jp, jz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="adiabatic"),
        obs.jp(pot=MWPotential2014, type="adiabatic"),
        obs.jz(pot=MWPotential2014, type="adiabatic"),
        obs.e(pot=MWPotential2014, type="adiabatic", analytic=True),
        obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True),
        obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True),
        obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True),
    )
    assert numpy.all(numpy.fabs(jr[:, 0] - aAA(obs[:, 0])[0]) < 10.0**-10.0), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.fabs(jp[:, 0] - aAA(obs[:, 0])[1]) < 10.0**-10.0), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.fabs(jz[:, 0] - aAA(obs[:, 0])[2]) < 10.0**-10.0), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(e[:, 0] - aAA.EccZmaxRperiRap(obs[:, 0])[0]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(zmax[:, 0] - aAA.EccZmaxRperiRap(obs[:, 0])[1]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rperi[:, 0] - aAA.EccZmaxRperiRap(obs[:, 0])[2]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rap[:, 0] - aAA.EccZmaxRperiRap(obs[:, 0])[3]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(e[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(zmax[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rperi[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(rap[:, 1])), (
        "Orbit interface for actionAngleAdiabatic does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_complexshape_staeckel():
    # Test that an unbound orbit in a set of orbits is handled correctly
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        numpy.array(
            [
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [1.05, 0.02, 1.05, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
            ]
        )
    )
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.71)
    jr, jp, jz, omr, omp, omz, wr, wp, wz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.Or(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.Op(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.Oz(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.wr(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.wp(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.wz(pot=MWPotential2014, type="staeckel", delta=0.71),
        obs.e(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True),
        obs.zmax(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True),
        obs.rperi(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True),
        obs.rap(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True),
    )
    assert numpy.all(numpy.fabs(jr[:, 0] - aAS(obs[:, 0])[0]) < 10.0**-10.0), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.fabs(jp[:, 0] - aAS(obs[:, 0])[1]) < 10.0**-10.0), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.fabs(jz[:, 0] - aAS(obs[:, 0])[2]) < 10.0**-10.0), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omr[:, 0] - aAS.actionsFreqs(obs[:, 0])[3]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omp[:, 0] - aAS.actionsFreqs(obs[:, 0])[4]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omz[:, 0] - aAS.actionsFreqs(obs[:, 0])[5]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wr[:, 0] - aAS.actionsFreqsAngles(obs[:, 0])[6]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wp[:, 0] - aAS.actionsFreqsAngles(obs[:, 0])[7]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wz[:, 0] - aAS.actionsFreqsAngles(obs[:, 0])[8]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(e[:, 0] - aAS.EccZmaxRperiRap(obs[:, 0])[0]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(zmax[:, 0] - aAS.EccZmaxRperiRap(obs[:, 0])[1]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rperi[:, 0] - aAS.EccZmaxRperiRap(obs[:, 0])[2]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rap[:, 0] - aAS.EccZmaxRperiRap(obs[:, 0])[3]) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


def test_orbit_interface_unbound_staeckeldelta_handling():
    # Test that the automagically determined delta is handled correctly when there are unbound orbits
    # Use a complex shape
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    obs = Orbit(
        numpy.array(
            [
                [
                    [1.15, 0.02, 1.15, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [1.02, 0.02, 0.95, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
                [
                    [0.97, 0.02, 1.25, 0.03, 0.0, 2.0],
                    [1.05, 0.02, 10.05, 0.03, 0.0, 2.0],
                ],
            ]
        )
    )
    # Compute the actions with the automagically determined delta using the orbit interface
    jr, jp, jz, omr, omp, omz, wr, wp, wz, e, zmax, rperi, rap = (
        obs.jr(pot=MWPotential2014, type="staeckel"),
        obs.jp(pot=MWPotential2014, type="staeckel"),
        obs.jz(pot=MWPotential2014, type="staeckel"),
        obs.Or(pot=MWPotential2014, type="staeckel"),
        obs.Op(pot=MWPotential2014, type="staeckel"),
        obs.Oz(pot=MWPotential2014, type="staeckel"),
        obs.wr(pot=MWPotential2014, type="staeckel"),
        obs.wp(pot=MWPotential2014, type="staeckel"),
        obs.wz(pot=MWPotential2014, type="staeckel"),
        obs.e(pot=MWPotential2014, type="staeckel", analytic=True),
        obs.zmax(pot=MWPotential2014, type="staeckel", analytic=True),
        obs.rperi(pot=MWPotential2014, type="staeckel", analytic=True),
        obs.rap(pot=MWPotential2014, type="staeckel", analytic=True),
    )
    # Now do the same with the actionAngle interface
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.71)  # just a dummy delta
    bound_indx = numpy.array([True, False, True, False, True, False])
    assert numpy.all(
        numpy.fabs(jr[:, 0] - aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[0])
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jp[:, 0] - aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[1])
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jz[:, 0] - aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[2])
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omr[:, 0] - aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[3]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omp[:, 0] - aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[4]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omz[:, 0] - aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[5]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wr[:, 0]
            - aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[6]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wp[:, 0]
            - aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[7]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wz[:, 0]
            - aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[8]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            e[:, 0]
            - aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[0]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            zmax[:, 0]
            - aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[1]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            rperi[:, 0]
            - aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[2]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            rap[:, 0]
            - aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[3]
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jr[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jp[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(numpy.isnan(jz[:, 1])), (
        "Orbit interface for actionAngleStaeckel does not return NaN for unbound orbit in a collection with an unbound orbit"
    )
    return None


# Test physical output for actionAngleStaeckel
def test_physical_staeckel():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.71, ro=ro, vo=vo)
    aAnu = actionAngleStaeckel(pot=MWPotential, delta=0.71)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), (
            "actionAngle function actionsFreqs does not return Quantity with the right value"
        )
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), (
            "actionAngle function actionsFreqs does not return Quantity with the right value"
        )
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), (
            "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
        )
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), (
            "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
        )
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), (
            "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
        )
    return None


# Test the b estimation
def test_estimateBIsochrone():
    from galpy.actionAngle import estimateBIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    o = Orbit([1.1, 0.3, 1.2, 0.2, 0.5, 2.0])
    times = numpy.linspace(0.0, 100.0, 1001)
    o.integrate(times, ip)
    bmin, bmed, bmax = estimateBIsochrone(ip, o.R(times), o.z(times))
    assert numpy.fabs(bmed - 1.2) < 10.0**-15.0, (
        "Estimated scale parameter b when estimateBIsochrone is applied to an IsochronePotential is wrong"
    )
    return None


# Test the focal delta estimation
def test_estimateDeltaStaeckel():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    o = Orbit([1.1, 0.05, 1.1, 0.05, 0.0, 2.0])
    times = numpy.linspace(0.0, 100.0, 1001)
    o.integrate(times, MWPotential)
    delta = estimateDeltaStaeckel(MWPotential, o.R(times), o.z(times))
    assert numpy.fabs(delta - 0.71) < 10.0**-3.0, (
        "Estimated focal parameter delta when estimateDeltaStaeckel is applied to the MWPotential is wrong"
    )
    return None


# Test the focal delta estimation
def test_estimateDeltaStaeckel_spherical():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    o = Orbit([1.1, 0.05, 1.1, 0.05, 0.0, 2.0])
    times = numpy.linspace(0.0, 100.0, 1001)
    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    o.integrate(times, lp)
    # Need to set delta0=0 so spherical actualluy returns 0
    delta = estimateDeltaStaeckel(lp, o.R(), o.z(), delta0=0.0)
    assert numpy.fabs(delta) < 10.0**-6.0, (
        "Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong"
    )
    delta = estimateDeltaStaeckel(lp, o.R(times), o.z(times), delta0=0.0)
    assert numpy.fabs(delta) < 10.0**-16.0, (
        "Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong"
    )
    return None


# Test that setting up the non-spherical actionAngle routines raises a warning when using MWPotential, see #229
def test_MWPotential_warning_adiabatic():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleAdiabaticGrid
    from galpy.potential import MWPotential

    with warnings.catch_warnings(record=True) as w:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        aAA = actionAngleAdiabatic(pot=MWPotential, gamma=1.0)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleAdiabatic with MWPotential should have thrown a warning, but didn't"
        )
    # Grid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAA = actionAngleAdiabaticGrid(
            pot=MWPotential, gamma=1.0, nEz=5, nEr=5, nLz=5, nR=5
        )
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleAdiabaticGrid with MWPotential should have thrown a warning, but didn't"
        )
    return None


def test_MWPotential_warning_staeckel():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelGrid
    from galpy.potential import MWPotential

    with warnings.catch_warnings(record=True) as w:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        aAA = actionAngleStaeckel(pot=MWPotential, delta=0.5)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleStaeckel with MWPotential should have thrown a warning, but didn't"
        )
    # Grid
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        aAA = actionAngleStaeckelGrid(pot=MWPotential, delta=0.5, nE=5, npsi=5, nLz=5)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleStaeckelGrid with MWPotential should have thrown a warning, but didn't"
        )
    return None


def test_MWPotential_warning_isochroneapprox():
    # Test that using MWPotential throws a warning, see #229
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential

    with warnings.catch_warnings(record=True) as w:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        aAA = actionAngleIsochroneApprox(pot=MWPotential, b=1.0)
        # Should raise warning bc of MWPotential, might raise others
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleIsochroneApprox with MWPotential should have thrown a warning, but didn't"
        )
    return None


# Test of the fix to issue 361
def test_actionAngleAdiabatic_issue361():
    from galpy import actionAngle
    from galpy.potential import MWPotential2014

    aA_adi = actionAngle.actionAngleAdiabatic(pot=MWPotential2014, c=True)
    R = 8.7007 / 8.0
    vT = 188.5 / 220.0
    jr_good, _, _ = aA_adi(R, -0.1 / 220.0, vT, 0, 0)
    jr_bad, _, _ = aA_adi(R, -0.09 / 220.0, vT, 0, 0)
    assert numpy.fabs(jr_good - jr_bad) < 1e-6, (
        f"Nearby JR for orbit near apocenter disagree too much, likely because one completely fails: Jr_good = {jr_good}, Jr_bad = {jr_bad}"
    )
    return None


# Test that evaluating actionAngle with multi-dimensional orbit doesn't work
def test_actionAngle_orbitInput_multid_error():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    orbits = Orbit(
        numpy.array(
            [
                [[1.0, 0.1, 1.1, -0.1, -0.2, 0.0], [1.0, 0.2, 1.2, 0.0, -0.1, 1.0]],
                [[1.0, -0.2, 0.9, 0.2, 0.2, 2.0], [1.2, -0.4, 1.1, -0.1, 0.0, -2.0]],
                [[1.0, 0.2, 0.9, 0.3, -0.2, 0.1], [1.2, 0.4, 1.1, -0.2, 0.05, 4.0]],
            ]
        )
    )
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    with pytest.raises(RuntimeError) as excinfo:
        aAS(orbits)
        pytest.fail(
            "Evaluating actionAngle methods with Orbit instances with multi-dimensional shapes is not support"
        )
    return None


# Test that actionAngleHarmonicInverse is the inverse of actionAngleHarmonic
def test_actionAngleHarmonicInverse_wrtHarmonic():
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleHarmonic, actionAngleHarmonicInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    ipz = ip.toVertical(1.2)
    # Omega = sqrt(4piG density / 3)
    aAH = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    aAHI = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, ipz)
    j, _, a = aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))
    xi, vxi = aAHI(numpy.median(j), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleHarmonicInverse is not the inverse of actionAngleHarmonic for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleHarmonicInverse is not the inverse of actionAngleHarmonic for an example orbit"
    )
    return None


def test_actionAngleHarmonicInverse_freqs_wrtHarmonic():
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleHarmonic, actionAngleHarmonicInverse
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aAH = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    aAHI = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    tol = -10.0
    j = 0.1
    Om = aAHI.Freqs(j)
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAH.actionsFreqs(*aAHI(j, 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleHarmonicInverse does not agree with that computed by actionAngleHarmonic"
    )
    return None


# Test that orbit from actionAngleHarmonicInverse is the same as an integrated orbit
def test_actionAngleHarmonicInverse_orbit():
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleHarmonicInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    ipz = ip.toVertical(1.2)
    # Omega = sqrt(4piG density / 3)
    aAHI = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    j = 0.01
    # First calculate frequencies and the initial x,v
    xvom = aAHI.xvFreqs(j, numpy.array([0.1]))
    om = xvom[2:]
    # Angles along an orbit
    ts = numpy.linspace(0.0, 20.0, 1001)
    angle = 0.1 + ts * om[0]
    # Calculate the orbit using actionAngleHarmonicInverse
    xv = aAHI(j, angle)
    # Calculate the orbit using orbit integration
    orb = Orbit([xvom[0][0], xvom[1][0]])
    orb.integrate(ts, ipz, method="dopr54_c")
    # Compare
    tol = -7.0
    assert numpy.all(numpy.fabs(orb.x(ts) - xv[0]) < 10.0**tol), (
        "Integrated orbit does not agree with actionAngleHarmmonicInverse orbit in x"
    )
    assert numpy.all(numpy.fabs(orb.vx(ts) - xv[1]) < 10.0**tol), (
        "Integrated orbit does not agree with actionAngleHarmmonicInverse orbit in v"
    )
    return None


# Test physical output for actionAngleHarmonicInverse
def test_physical_actionAngleHarmonicInverse():
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleHarmonicInverse
    from galpy.potential import IsochronePotential
    from galpy.util import conversion

    ip = IsochronePotential(normalize=5.0, b=10000.0)
    ro, vo = 7.0, 230.0
    aAHI = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=ro, vo=vo
    )
    aAHInu = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    correct_fac = [ro, vo]
    for ii in range(2):
        assert (
            numpy.fabs(aAHI(0.1, -0.2)[ii] - aAHInu(0.1, -0.2)[ii] * correct_fac[ii])
            < 10.0**-8.0
        ), (
            "actionAngleInverse function __call__ does not return Quantity with the right value"
        )
    correct_fac = [ro, vo, conversion.freq_in_Gyr(vo, ro)]
    for ii in range(3):
        assert (
            numpy.fabs(
                aAHI.xvFreqs(0.1, -0.2)[ii]
                - aAHInu.xvFreqs(0.1, -0.2)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), (
            "actionAngleInverse function xvFreqs does not return Quantity with the right value"
        )
    assert (
        numpy.fabs(aAHI.Freqs(0.1) - aAHInu.Freqs(0.1) * conversion.freq_in_Gyr(vo, ro))
        < 10.0**-8.0
    ), "actionAngleInverse function Freqs does not return Quantity with the right value"
    return None


# Test that actionAngleIsochroneInverse is the inverse of actionAngleIsochrone
def test_actionAngleIsochroneInverse_wrtIsochrone():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=2.0, b=1.5)
    aAI = actionAngleIsochrone(ip=ip)
    aAII = actionAngleIsochroneInverse(ip=ip)
    # Check a few orbits
    tol = -7.0
    R, vR, vT, z, vz, phi = 1.1, 0.1, 1.1, 0.1, 0.2, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, 0.1, -1.1, 0.1, 0.2, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, -0.1, 1.1, 0.1, 0.2, 0.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, -0.1, 1.1, 0.1, -0.2, 0.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, -4.1, 1.1, 0.1, -0.2, 0.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    return None


# Test that actionAngleIsochroneInverse is the inverse of actionAngleIsochrone,
# for an orbit that is not inclined (at z=0); possibly problematic, because
# the longitude of the ascending node is ambiguous; set to zero by convention
# in actionAngleIsochrone
def test_actionAngleIsochroneInverse_wrtIsochrone_noninclinedorbit():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=2.0, b=1.5)
    aAI = actionAngleIsochrone(ip=ip)
    aAII = actionAngleIsochroneInverse(ip=ip)
    # Check a few orbits
    tol = -7.0
    R, vR, vT, z, vz, phi = 1.1, 0.1, 1.1, 0.0, 0.0, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, 0.1, -1.1, 0.0, 0.0, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    # also some almost non-inclined orbits
    eps = 1e-10
    R, vR, vT, z, vz, phi = 1.1, 0.1, 1.1, 0.0, eps, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    R, vR, vT, z, vz, phi = 1.1, 0.1, -1.1, 0.0, eps, 2.3
    o = Orbit([R, vR, vT, z, vz, phi])
    check_actionAngleIsochroneInverse_wrtIsochrone(ip, aAI, aAII, o, tol, ntimes=1001)
    return None


# Basic sanity checking: close-to-circular orbit should have freq. = epicycle freq.
def test_actionAngleIsochroneInverse_basic_freqs():
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential, epifreq, omegac, rl, verticalfreq

    jr = 10.0**-6.0
    jz = 10.0**-6.0
    ip = IsochronePotential(normalize=1.0)
    aAII = actionAngleIsochroneInverse(ip=ip)
    tol = -5.0
    # at Lz=1
    jphi = 1.0
    om = aAII.Freqs(jr, jphi, jz)
    assert numpy.fabs((om[0] - epifreq(ip, rl(ip, jphi))) / om[0]) < 10.0**tol, (
        "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    )
    assert numpy.fabs((om[1] - omegac(ip, rl(ip, jphi))) / om[1]) < 10.0**tol, (
        "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    )
    assert numpy.fabs((om[2] - verticalfreq(ip, rl(ip, jphi))) / om[2]) < 10.0**tol, (
        "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    )
    # at Lz=1.5, w/ different potential normalization
    ip = IsochronePotential(normalize=1.2)
    aAII = actionAngleIsochroneInverse(ip=ip)
    jphi = 1.5
    om = aAII.Freqs(jr, jphi, jz)
    assert numpy.fabs((om[0] - epifreq(ip, rl(ip, jphi))) / om[0]) < 10.0**tol, (
        "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    )
    assert numpy.fabs((om[1] - omegac(ip, rl(ip, jphi))) / om[1]) < 10.0**tol, (
        "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    )
    assert numpy.fabs((om[2] - verticalfreq(ip, rl(ip, jphi))) / om[2]) < 10.0**tol, (
        "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    )
    return None


def test_actionAngleIsochroneInverse_freqs_wrtIsochrone():
    from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    jr = 0.1
    jz = 0.2
    ip = IsochronePotential(normalize=1.04, b=1.2)
    aAI = actionAngleIsochrone(ip=ip)
    aAII = actionAngleIsochroneInverse(ip=ip)
    # at Lz=1
    tol = -10.0
    jphi = 1.0
    Or, Op, Oz = aAII.Freqs(jr, jphi, jz)
    # Compute frequency with actionAngleIsochrone
    _, _, _, Ori, Opi, Ozi = aAI.actionsFreqs(*aAII(jr, jphi, jz, 0.0, 1.0, 2.0)[:6])
    assert numpy.fabs((Or - Ori) / Or) < 10.0**tol, (
        "Radial frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    assert numpy.fabs((Op - Opi) / Op) < 10.0**tol, (
        "Azimuthal frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    assert numpy.fabs((Oz - Ozi) / Oz) < 10.0**tol, (
        "Vertical frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    # at Lz=1.5
    jphi = 1.51
    Or, Op, Oz = aAII.Freqs(jr, jphi, jz)
    # Compute frequency with actionAngleIsochrone
    _, _, _, Ori, Opi, Ozi = aAI.actionsFreqs(*aAII(jr, jphi, jz, 0.0, 1.0, 2.0)[:6])
    assert numpy.fabs((Or - Ori) / Or) < 10.0**tol, (
        "Radial frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    assert numpy.fabs((Op - Opi) / Op) < 10.0**tol, (
        "Azimuthal frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    assert numpy.fabs((Oz - Ozi) / Oz) < 10.0**tol, (
        "Vertical frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    )
    return None


# Test that orbit from actionAngleIsochroneInverse is the same as an integrated orbit
def test_actionAngleIsochroneInverse_orbit():
    from galpy.actionAngle.actionAngleIsochroneInverse import (
        actionAngleIsochroneInverse,
    )
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    # Set up instance
    ip = IsochronePotential(normalize=1.03, b=1.2)
    aAII = actionAngleIsochroneInverse(ip=ip)
    jr, jphi, jz = 0.05, 1.1, 0.025
    # First calculate frequencies and the initial RvR
    RvRom = aAII.xvFreqs(
        jr, jphi, jz, numpy.array([0.0]), numpy.array([1.0]), numpy.array([2.0])
    )
    om = RvRom[6:]
    # Angles along an orbit
    ts = numpy.linspace(0.0, 100.0, 1001)
    angler = ts * om[0]
    anglephi = 1.0 + ts * om[1]
    anglez = 2.0 + ts * om[2]
    # Calculate the orbit using actionAngleTorus
    RvR = aAII(jr, jphi, jz, angler, anglephi, anglez)
    # Calculate the orbit using orbit integration
    orb = Orbit(
        [RvRom[0][0], RvRom[1][0], RvRom[2][0], RvRom[3][0], RvRom[4][0], RvRom[5][0]]
    )
    orb.integrate(ts, ip)
    # Compare
    tol = -3.0
    assert numpy.all(numpy.fabs(orb.R(ts) - RvR[0]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in R"
    )
    assert numpy.all(numpy.fabs(orb.vR(ts) - RvR[1]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vR"
    )
    assert numpy.all(numpy.fabs(orb.vT(ts) - RvR[2]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vT"
    )
    assert numpy.all(numpy.fabs(orb.z(ts) - RvR[3]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in z"
    )
    assert numpy.all(numpy.fabs(orb.vz(ts) - RvR[4]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vz"
    )
    assert numpy.all(
        numpy.fabs((orb.phi(ts) - RvR[5] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        < 10.0**tol
    ), "Integrated orbit does not agree with torus orbit in phi"
    return None


# Test physical output for actionAngleIsochroneInverse
def test_physical_actionAngleIsochroneInverse():
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    ip = IsochronePotential(normalize=1.01, b=1.02)
    aAII = actionAngleIsochroneInverse(ip=ip, ro=ro, vo=vo)
    aAIInu = actionAngleIsochroneInverse(ip=ip)
    correct_fac = [ro, vo, vo, ro, vo, 1.0]
    for ii in range(6):
        assert (
            numpy.fabs(
                aAII(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii]
                - aAIInu(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), (
            "actionAngleInverse function __call__ does not return Quantity with the right value"
        )
    correct_fac = [
        ro,
        vo,
        vo,
        ro,
        vo,
        1.0,
        conversion.freq_in_Gyr(vo, ro),
        conversion.freq_in_Gyr(vo, ro),
        conversion.freq_in_Gyr(vo, ro),
    ]
    for ii in range(9):
        assert (
            numpy.fabs(
                aAII.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii]
                - aAIInu.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), (
            "actionAngleInverse function xvFreqs does not return Quantity with the right value"
        )
    for ii in range(3):
        assert (
            numpy.fabs(
                aAII.Freqs(0.1, 1.1, 0.1)[ii]
                - aAIInu.Freqs(0.1, 1.1, 0.1)[ii] * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), (
            "actionAngleInverse function Freqs does not return Quantity with the right value"
        )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
def test_actionAngleVerticalInverse_wrtVertical():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E()], use_pointtransform=False
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E(pot=isopot)], use_pointtransform=False
    )
    tol = -10.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated orbit
def test_actionAngleVerticalInverse_orbit():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=False
    )

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
# when using a point transformation
def test_actionAngleVerticalInverse_wrtVertical_pointtransform():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E()], use_pointtransform=True
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using a point transform"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using a point transform"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_pointtransform():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E(pot=isopot)], use_pointtransform=True
    )
    tol = -10.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using a point transform"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated
# orbit when using a point transformation
def test_actionAngleVerticalInverse_orbit_pointtransform():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=True
    )

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
# when using the exact (ODE-based) point transformation
def test_actionAngleVerticalInverse_wrtVertical_exactpointtransform():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E()], use_pointtransform="exact"
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_exactpointtransform():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E(pot=isopot)], use_pointtransform="exact"
    )
    # The accuracy of the exact point transformation is set by the tolerance
    # of its ODE solution (rtol=1e-12), which limits the frequency to ~1e-10
    # relative accuracy (unlike the polynomial point transformation, whose
    # imperfection is absorbed to machine precision by the S_n coefficients)
    tol = -9.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using the exact point transform"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated
# orbit when using the exact (ODE-based) point transformation
def test_actionAngleVerticalInverse_orbit_exactpointtransform():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform="exact"
    )

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using the exact point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using the exact point transform"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using the exact point transform"
    )
    return None


# Test that the exact (ODE-based) point transformation maps the torus exactly
# onto a harmonic-oscillator torus: all nSn coefficients should be zero to
# within the accuracy of the ODE solution / spline representation (~1e-10),
# unlike for the polynomial point transformation or no point transformation
def test_actionAngleVerticalInverse_coeffs_exactpointtransform():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform="exact"
    )
    assert numpy.nanmax(numpy.fabs(aAVI._nSn)) < 1e-9, (
        "nSn coefficients using the exact point transformation are not all close to zero"
    )
    # Compare against no point transformation, where the coefficients are O(0.01-1)
    aAVI_nopt = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[0.1, 1.0, 10.0],
        use_pointtransform=False,
        momentum_matched=False,
    )
    assert numpy.nanmax(numpy.fabs(aAVI._nSn)) < 1e-6 * numpy.nanmax(
        numpy.fabs(aAVI_nopt._nSn)
    ), (
        "nSn coefficients using the exact point transformation are not orders of magnitude smaller than without a point transformation"
    )
    # Also check the edge case of a grid consisting only of the E=0 torus,
    # for which the point transformation is the identity
    aAVI0 = actionAngleVerticalInverse(
        pot=isopot, nta=32, Es=[0.0], use_pointtransform="exact"
    )
    assert numpy.all(aAVI0._nSn == 0.0), (
        "nSn coefficients of the E=0 torus are not all zero when using the exact point transformation"
    )
    return None


# Test that evaluating with the point transformation only (skipping the
# generating-function mapping, which is the identity for the exact point
# transformation) agrees with the full machinery and conserves energy
def test_actionAngleVerticalInverse_orbit_exactpointtransform_ptonly():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[0.1, 1.0, 10.0],
        use_pointtransform="exact",
        pt_only=True,
    )
    aAVIfull = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform="exact"
    )
    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # pt_only agrees with the full machinery at the level of the accuracy of
    # the point transformation itself
    xf, vf = aAVIfull(aAVIfull.J(1.0), ta)
    assert numpy.amax(numpy.fabs(x - xf)) < 1e-8, (
        "pt_only evaluation does not agree with the full generating-function evaluation for the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(v - vf)) < 1e-8, (
        "pt_only evaluation does not agree with the full generating-function evaluation for the exact point transformation"
    )
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-9, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using pt_only evaluation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using pt_only evaluation"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using pt_only evaluation"
    )
    return None


# Test the pt_only diagnostics: a warning when the point transformation is not
# accurate enough and errors when pt_only is combined with a
# non-exact point transformation
def test_actionAngleVerticalInverse_ptonly_warnings_errors():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential
    from galpy.util import galpyWarning

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    # Coarse point-transformation mesh --> coefficients not small --> warning
    with pytest.warns(galpyWarning, match="not accurate enough"):
        actionAngleVerticalInverse(
            pot=isopot,
            nta=128,
            Es=[1.0],
            use_pointtransform="exact",
            pt_only=True,
            pt_nxa=7,
        )
    # pt_only requires the exact point transformation
    with pytest.raises(ValueError):
        actionAngleVerticalInverse(
            pot=isopot, nta=128, Es=[1.0], use_pointtransform=True, pt_only=True
        )
    with pytest.raises(ValueError):
        actionAngleVerticalInverse(
            pot=isopot, nta=128, Es=[1.0], use_pointtransform=False, pt_only=True
        )
    with pytest.raises(ValueError):
        actionAngleVerticalInverse(
            pot=isopot,
            nta=128,
            Es=[1.0],
            use_pointtransform=False,
            momentum_matched=False,
            pt_only=True,
        )
    return None


# Test that actionAngleVerticalInverse with the exact point transformation
# also works when using only bisection to solve equations
def test_actionAngleVerticalInverse_wrtVertical_exactpointtransform_bisect():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[obs.E()],
        use_pointtransform="exact",
        bisect=True,
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform and bisection"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform and bisection"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
# when using only bisection to solve equations
def test_actionAngleVerticalInverse_wrtVertical_bisect():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E()], use_pointtransform=False, bisect=True
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_bisect():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[obs.E(pot=isopot)],
        use_pointtransform=False,
        bisect=True,
    )
    tol = -10.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using bisection"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated orbit
def test_actionAngleVerticalInverse_orbit_bisect():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[0.1, 1.0, 10.0],
        use_pointtransform=False,
        bisect=True,
    )

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
# when using a point transformation
def test_actionAngleVerticalInverse_wrtVertical_pointtransform_bisect():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    # Set up actionAngleVerticalInverse for this energy
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[obs.E()], use_pointtransform=True, bisect=True
    )
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection and a point transformation"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection and a point transformation"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_pointtransform_bisect():
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[obs.E(pot=isopot)],
        use_pointtransform=True,
        bisect=True,
    )
    tol = -10.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using bisection and a point transformation"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated
# orbit when using a point transformation
def test_actionAngleVerticalInverse_orbit_pointtransform_bisect():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[0.1, 1.0, 10.0],
        use_pointtransform=True,
        bisect=True,
    )

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    x, v = aAVI(aAVI.J(1.0), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection and a point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection and a point transformation"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection and a point transformation"
    )
    return None


# Tests of interpolated actionAngleVerticalInverse need fixture to set up the
# interpolated actionAngleVerticalInverse
@pytest.fixture(scope="module")
def setup_actionAngleVerticalInverse_interpolated():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aA1Dinv = actionAngleVerticalInverse(
        pot=isopot,
        nta=2 * 128,
        Es=numpy.linspace(0.0, 4.0, 1001),
        setup_interp=True,
        use_pointtransform=False,
    )
    return aA1Dinv, isopot


@pytest.fixture(scope="module")
def setup_actionAngleVerticalInverse_interpolated_pointtransform():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aA1Dinv = actionAngleVerticalInverse(
        pot=isopot,
        nta=2 * 128,
        Es=numpy.linspace(0.0, 4.0, 1001),
        setup_interp=True,
        use_pointtransform=True,
        pt_deg=7,
    )
    return aA1Dinv, isopot


@pytest.fixture(scope="module")
def setup_actionAngleVerticalInverse_interpolated_exactpointtransform():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aA1Dinv = actionAngleVerticalInverse(
        pot=isopot,
        nta=2 * 128,
        Es=numpy.linspace(0.0, 4.0, 1001),
        setup_interp=True,
        use_pointtransform="exact",
    )
    return aA1Dinv, isopot


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
def test_actionAngleVerticalInverse_wrtVertical_interpolation(
    setup_actionAngleVerticalInverse_interpolated,
):
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_interpolation(
    setup_actionAngleVerticalInverse_interpolated,
):
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    # Freqs routes through the map's dE/dJ, which differentiates the
    # Hermite energy interpolant and is therefore ~4e-10 off the isolated
    # true frequency between grid nodes, where the frequency table would
    # be exact; the map's answer is preferred because it is exactly the
    # frequency of the (x, v) trajectories the map returns, and an answer
    # inconsistent with the returned orbits is the wrong kind of accurate
    tol = -9.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation"
    )
    # and the two public answers agree EXACTLY: Freqs is the frequency of
    # the trajectories _xvFreqs returns, which is the point of the routing
    j = float(aAVI.J(obs.E(pot=isopot)))
    assert numpy.fabs(float(aAVI.Freqs(j)) - float(aAVI._xvFreqs(j, 0.0)[2])) == 0.0, (
        "Freqs and _xvFreqs disagree on the frequency of the same torus"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated orbit
def test_actionAngleVerticalInverse_orbit_interpolation(
    setup_actionAngleVerticalInverse_interpolated,
):
    from galpy.orbit import Orbit
    from galpy.potential import evaluatelinearPotentials

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    Ei = 1.3132
    x, v = aAVI(aAVI.J(Ei), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(Ei), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
def test_actionAngleVerticalInverse_wrtVertical_interpolation_pointtransform(
    setup_actionAngleVerticalInverse_interpolated_pointtransform,
):
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_pointtransform
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and a point transformation"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and a point transformation"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_interpolation_pointtransform(
    setup_actionAngleVerticalInverse_interpolated_pointtransform,
):
    # Create harmonic oscillator potential as isochrone w/ large b --> 1D
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_pointtransform
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    tol = -7.5
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation and a point transformation"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated orbit
def test_actionAngleVerticalInverse_orbit_interpolation_pointtransform(
    setup_actionAngleVerticalInverse_interpolated_pointtransform,
):
    from galpy.orbit import Orbit
    from galpy.potential import evaluatelinearPotentials

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_pointtransform

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    Ei = 1.3132
    x, v = aAVI(aAVI.J(Ei), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-8, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation and a point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(Ei), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-7, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and a point transformation"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-7, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and a point transformation"
    )
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
def test_actionAngleVerticalInverse_wrtVertical_interpolation_exactpointtransform(
    setup_actionAngleVerticalInverse_interpolated_exactpointtransform,
):
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_exactpointtransform
    aAV = actionAngleVertical(pot=isopot)
    # Check a few orbits
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    times = numpy.linspace(0.0, 30.0, 1001)
    obs.integrate(times, isopot)
    j, _, a = aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    xi, vxi = aAVI(aAVI.J(obs.E()), a)
    assert numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and the exact point transformation"
    )
    return None


def test_actionAngleVerticalInverse_freqs_wrtVertical_interpolation_exactpointtransform(
    setup_actionAngleVerticalInverse_interpolated_exactpointtransform,
):
    from galpy.actionAngle import actionAngleVertical
    from galpy.orbit import Orbit

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_exactpointtransform
    aAV = actionAngleVertical(pot=isopot)
    x, vx = 0.1, -0.3
    obs = Orbit([x, vx])
    tol = -7.5
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation and the exact point transformation"
    )
    return None


# Test that orbit from actionAngleVerticalInverse is the same as an integrated orbit
def test_actionAngleVerticalInverse_orbit_interpolation_exactpointtransform(
    setup_actionAngleVerticalInverse_interpolated_exactpointtransform,
):
    from galpy.orbit import Orbit
    from galpy.potential import evaluatelinearPotentials

    aAVI, isopot = setup_actionAngleVerticalInverse_interpolated_exactpointtransform

    ta = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    Ei = 1.3132
    x, v = aAVI(aAVI.J(Ei), ta)
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(E) / numpy.mean(E) < 1e-8, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation and the exact point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(Ei), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-7, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-7, (
        "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and the exact point transformation"
    )
    return None


def test_actionAngleVerticalInverse_plotting():
    import matplotlib.pyplot as pyplot

    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=False
    )
    aAVIpt = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=True
    )
    aAVIept = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform="exact"
    )

    # the older evaluations' convergence plot, alone and overplotted
    gs = aAVIpt.plot_convergence(1.0, return_gridspec=True)
    aAVIept.plot_convergence(1.0, overplot=gs)
    pyplot.close()
    # the momentum-matched map's, alone, overplotted, and on the bottom torus
    gs = aAVI.plot_convergence(1.0, return_gridspec=True)
    aAVI.plot_convergence(0.1, overplot=gs)
    pyplot.close()
    aAVI.plot_convergence(1.0)
    pyplot.close()
    aAVI0 = actionAngleVerticalInverse(pot=isopot, nta=128, Es=[0.0, 1.0])
    aAVI0.plot_convergence(0.0)
    pyplot.close()
    gs = aAVI.plot_power(0.1, return_gridspec=True)
    gs = aAVI.plot_power([0.1, 1.0, 10.0], overplot=gs)
    gs = aAVIept.plot_power([0.1, 1.0, 10.0], overplot=gs)
    pyplot.close()
    gs = aAVIpt.plot_power(0.1, return_gridspec=True)
    pyplot.close()
    aAVI.plot_orbit(1.0)
    aAVIept.plot_orbit(1.0)
    pyplot.close()
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
@pytest.mark.parametrize(
    "fixture",
    [
        "setup_actionAngleVerticalInverse_interpolated",
        "setup_actionAngleVerticalInverse_interpolated_pointtransform",
        "setup_actionAngleVerticalInverse_interpolated_exactpointtransform",
    ],
)
def test_actionAngleVerticalInverse_interpolation_plotting(fixture, request):
    import matplotlib.pyplot as pyplot

    aAVI, _ = request.getfixturevalue(fixture)
    gs = aAVI.plot_convergence(3.7, return_gridspec=True)
    pyplot.close()
    aAVI.plot_power(numpy.linspace(0.0, 4.0, 1001))
    pyplot.close()
    aAVI.plot_orbit(3.706)
    pyplot.close()
    aAVI.plot_interp(3.706)
    pyplot.close()
    return None


def test_actionAngleVerticalInverse_convergence_warnings():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    # Setup warnings
    with warnings.catch_warnings(record=True) as w:
        if PY2:
            reset_warning_registry("galpy")
        warnings.simplefilter("always", galpyWarning)
        aAVI = actionAngleVerticalInverse(
            pot=isopot,
            nta=4 * 128,
            Es=[300.0],
            use_pointtransform=False,
            momentum_matched=False,
            maxiter=100,
        )
        # Should raise convergence warnings
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Torus mapping with Newton-Raphson did not converge in 100 iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleVerticalInverse for large energy should have raised convergence warning, but didn't"
        )
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Torus mapping with bisection did not converge in 100 iterations for energies: 300"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleVerticalInverse for large energy should have raised convergence warning, but didn't"
        )
    # The momentum-matched map warns when its truncated anomaly series does
    # not reconstruct the momentum, which for this potential's nearly
    # linear outskirts happens at high energy with the default number of
    # harmonics, and fails outright when the turning point cannot be found
    with pytest.warns(galpyWarning, match="not converged for energies: 30"):
        actionAngleVerticalInverse(pot=isopot, nta=128, Es=[1.0, 30.0])
    with pytest.raises(RuntimeError, match="turning point could not be found"):
        actionAngleVerticalInverse(pot=isopot, nta=128, Es=[300.0])
    return None


def test_actionAngleVerticalInverse_plotting_errors():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot,
        nta=4 * 128,
        Es=[0.1, 1.0, 10.0, 20.0, 30.0],
        use_pointtransform=False,
    )
    with pytest.raises(ValueError) as excinfo:
        gs = aAVI.plot_convergence(1.1, return_gridspec=True)
        pytest.fail(
            "Calling plot_convergence with an energy not given should have given a ValueError, but did not"
        )
    # and in the older evaluation
    aAVIold = actionAngleVerticalInverse(
        pot=isopot, nta=128, Es=[0.1, 1.0], momentum_matched=False
    )
    with pytest.raises(ValueError) as excinfo:
        aAVIold.plot_convergence(1.1)
        pytest.fail(
            "Calling plot_convergence with an energy not given should have given a ValueError, but did not"
        )
    with pytest.raises(ValueError) as excinfo:
        aAVI.plot_power(1.1)
        pytest.fail(
            "Calling plot_convergence with an energy not given should have given a ValueError, but did not"
        )
    with pytest.raises(RuntimeError) as excinfo:
        aAVI.plot_power(numpy.linspace(0.0, 4.0, 1001), overplot=True)
        pytest.fail(
            "Calling plot_power with overplot=True and many Es should have raised a RuntimeError, but didn't"
        )
    with pytest.raises(ValueError) as excinfo:
        aAVI.plot_orbit(1.1)
        pytest.fail(
            "Calling plot_convergence with an energy not given should have given a ValueError, but did not"
        )
    return None


def test_actionAngleVerticalInverse_interpolation_errors():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=True
    )
    # Interpolation not being set up should lead to a bunch of errors
    with pytest.raises(RuntimeError) as excinfo:
        aAVI.nSn(0.1)
        pytest.fail(
            "Calling nSn without interpolation should have raised a RuntimeError, but did not"
        )
    with pytest.raises(RuntimeError) as excinfo:
        aAVI.dSndJ(0.1)
        pytest.fail(
            "Calling dSndJ without interpolation should have raised a RuntimeError, but did not"
        )
    with pytest.raises(RuntimeError) as excinfo:
        aAVI.pt_coeffs(0.1)
        pytest.fail(
            "Calling pt_coeffs without interpolation should have raised a RuntimeError, but did not"
        )
    with pytest.raises(RuntimeError) as excinfo:
        aAVI.pt_deriv_coeffs(0.1)
        pytest.fail(
            "Calling pt_deriv_coeffs without interpolation should have raised a RuntimeError, but did not"
        )
    return None


# Test that evaluating various functions for an actionAngleVerticalInverse instance for an E not in the instantiation raises an error
def test_actionAngleVerticalInverse_notE_errors():
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    # Set up instance
    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=True
    )
    with pytest.raises(ValueError) as excinfo:
        aAVI.J(0.11)
        pytest.fail(
            "Calling J with an energy not given should have given a ValueError, but did not"
        )
    with pytest.raises(ValueError) as excinfo:
        # actually action input here, but this is fine
        aAVI.xvFreqs(0.11, 0.0)
        pytest.fail(
            "Calling xvFreqs with an energy not given should have given a ValueError, but did not"
        )
    with pytest.raises(ValueError) as excinfo:
        # actually action input here, but this is fine
        aAVI.Freqs(0.11)
        pytest.fail(
            "Calling Freqs with an energy not given should have given a ValueError, but did not"
        )
    return None


# Test that computing actionAngle coordinates in C for a NullPotential leads to an error
def test_nullpotential_error():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import NullPotential

    np = NullPotential()
    aAS = actionAngleStaeckel(pot=np, delta=1.0)
    with pytest.raises(NotImplementedError) as excinfo:
        aAS(1.0, 0.0, 1.0, 0.1, 0.0)
        pytest.fail(
            "Calculating actionAngle coordinates in C for a NullPotential should have given a NotImplementedError, but did not"
        )
    return None


def check_actionAngleIsochroneInverse_wrtIsochrone(
    pot, aAI, aAII, obs, tol, ntimes=1001
):
    times = numpy.linspace(0.0, 30.0, ntimes)
    obs.integrate(times, pot)
    jr, jp, jz, _, _, _, ar, ap, az = aAI.actionsFreqsAngles(
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    Ri, vRi, vTi, zi, vzi, phii = aAII(
        numpy.median(jr), numpy.median(jp), numpy.median(jz), ar, ap, az
    )
    assert numpy.amax(numpy.fabs(obs.R(times) - Ri)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert (
        numpy.amax(
            numpy.fabs((obs.phi(times) - phii + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        )
        < 10.0**tol
    ), (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.z(times) - zi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.vR(times) - vRi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.vT(times) - vTi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(obs.vz(times) - vzi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    return None


# Test that the actions are conserved along an orbit
def check_actionAngle_conserved_actions(
    aA, obs, pot, toljr, toljp, toljz, ntimes=1001, fixed_quad=False, inclphi=False
):
    times = numpy.linspace(0.0, 100.0, ntimes)
    obs.integrate(times, pot, method="dopr54_c")
    if fixed_quad and inclphi:
        js = aA(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
            fixed_quad=True,
        )
    elif fixed_quad and not inclphi:
        js = aA(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            fixed_quad=True,
        )
    elif inclphi:
        js = aA(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
        )
    else:
        # Test Orbit with multiple objects case, but calling
        js = aA(obs(times))
    maxdj = numpy.amax(
        numpy.fabs(js - numpy.tile(numpy.mean(js, axis=1), (len(times), 1)).T), axis=1
    ) / numpy.mean(js, axis=1)
    assert maxdj[0] < 10.0**toljr, "Jr conservation fails at %g%%" % (100.0 * maxdj[0])
    assert maxdj[1] < 10.0**toljp, "Lz conservation fails at %g%%" % (100.0 * maxdj[1])
    assert maxdj[2] < 10.0**toljz, "Jz conservation fails at %g%%" % (100.0 * maxdj[2])
    return None


# Test that the angles increase linearly
def check_actionAngle_linear_angles(
    aA,
    obs,
    pot,
    tolinitar,
    tolinitap,
    tolinitaz,
    tolor,
    tolop,
    toloz,
    toldar,
    toldap,
    toldaz,
    maxt=100.0,
    ntimes=1001,
    separate_times=False,
    fixed_quad=False,
    u0=None,
):
    from galpy.actionAngle import dePeriod

    times = numpy.linspace(0.0, maxt, ntimes)
    obs.integrate(times, pot, method="dopr54_c")
    if fixed_quad:
        acfs_init = aA.actionsFreqsAngles(
            obs, fixed_quad=True
        )  # to check the init. angles
        acfs = aA.actionsFreqsAngles(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
            fixed_quad=True,
        )
    elif not u0 is None:
        acfs_init = aA.actionsFreqsAngles(obs, u0=u0)  # to check the init. angles
        acfs = aA.actionsFreqsAngles(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
            u0=(u0 + times * 0.0),
        )  # array
    else:
        acfs_init = aA.actionsFreqsAngles(obs())  # to check the init. angles
        if separate_times:
            acfs = numpy.array(
                [
                    aA.actionsFreqsAngles(
                        obs.R(t), obs.vR(t), obs.vT(t), obs.z(t), obs.vz(t), obs.phi(t)
                    )
                    for t in times
                ]
            )[:, :, 0].T
            acfs = (
                acfs[0],
                acfs[1],
                acfs[2],
                acfs[3],
                acfs[4],
                acfs[5],
                acfs[6],
                acfs[7],
                acfs[8],
            )
        else:
            acfs = aA.actionsFreqsAngles(
                obs.R(times),
                obs.vR(times),
                obs.vT(times),
                obs.z(times),
                obs.vz(times),
                obs.phi(times),
            )
    ar = dePeriod(numpy.reshape(acfs[6], (1, len(times)))).flatten()
    ap = dePeriod(numpy.reshape(acfs[7], (1, len(times)))).flatten()
    az = dePeriod(numpy.reshape(acfs[8], (1, len(times)))).flatten()
    # Do linear fit to radial angle, check that deviations are small, check
    # that the slope is the frequency
    if acfs_init[6].ndim > 0:
        acfs_init_radial_angle = acfs_init[6][0]
    else:
        acfs_init_radial_angle = acfs_init[6]
    linfit = numpy.polyfit(times, ar, 1)
    assert (
        numpy.fabs((linfit[1] - acfs_init_radial_angle) / acfs_init_radial_angle)
        < 10.0**tolinitar
    ), (
        "Radial angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (
            100.0
            * numpy.fabs((linfit[1] - acfs_init_radial_angle) / acfs_init_radial_angle)
        )
    )
    if acfs_init[3].ndim > 0:
        acfs_init_radial_freq = acfs_init[3][0]
    else:
        acfs_init_radial_freq = acfs_init[3]
    assert numpy.fabs(linfit[0] - acfs_init_radial_freq) < 10.0**tolor, (
        "Radial frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (
            100.0
            * numpy.fabs((linfit[0] - acfs_init_radial_freq) / acfs_init_radial_freq)
        )
    )
    devs = ar - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**toldar, (
        "Maximum deviation from linear trend in the radial angles is %g" % maxdev
    )
    # Do linear fit to azimuthal angle, check that deviations are small, check
    # that the slope is the frequency
    if acfs_init[7].ndim > 0:
        acfs_init_azimuthal_angle = acfs_init[7][0]
    else:
        acfs_init_azimuthal_angle = acfs_init[7]
    linfit = numpy.polyfit(times, ap, 1)
    assert (
        numpy.fabs((linfit[1] - acfs_init_azimuthal_angle) / acfs_init_azimuthal_angle)
        < 10.0**tolinitap
    ), (
        "Azimuthal angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (
            100.0
            * numpy.fabs(
                (linfit[1] - acfs_init_azimuthal_angle) / acfs_init_azimuthal_angle
            )
        )
    )
    if acfs_init[4].ndim > 0:
        acfs_init_azimuthal_freq = acfs_init[4][0]
    else:
        acfs_init_azimuthal_freq = acfs_init[4]
    assert numpy.fabs(linfit[0] - acfs_init_azimuthal_freq) < 10.0**tolop, (
        "Azimuthal frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (
            100.0
            * numpy.fabs(
                (linfit[0] - acfs_init_azimuthal_freq) / acfs_init_azimuthal_freq
            )
        )
    )
    devs = ap - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**toldap, (
        "Maximum deviation from linear trend in the azimuthal angle is %g" % maxdev
    )
    # Do linear fit to vertical angle, check that deviations are small, check
    # that the slope is the frequency
    if acfs_init[8].ndim > 0:
        acfs_init_vertical_angle = acfs_init[8][0]
    else:
        acfs_init_vertical_angle = acfs_init[8]
    linfit = numpy.polyfit(times, az, 1)
    assert (
        numpy.fabs((linfit[1] - acfs_init_vertical_angle) / acfs_init_vertical_angle)
        < 10.0**tolinitaz
    ), (
        "Vertical angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (
            100.0
            * numpy.fabs(
                (linfit[1] - acfs_init_vertical_angle) / acfs_init_vertical_angle
            )
        )
    )
    if acfs_init[5].ndim > 0:
        acfs_init_vertical_freq = acfs_init[5][0]
    else:
        acfs_init_vertical_freq = acfs_init[5]
    assert numpy.fabs(linfit[0] - acfs_init_vertical_freq) < 10.0**toloz, (
        "Vertical frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (
            100.0
            * numpy.fabs(
                (linfit[0] - acfs_init_vertical_freq) / acfs_init_vertical_freq
            )
        )
    )
    devs = az - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**toldaz, (
        "Maximum deviation from linear trend in the vertical angles is %g" % maxdev
    )
    return None


# Test that the ecc, zmax, rperi, rap are conserved along an orbit
def check_actionAngle_conserved_EccZmaxRperiRap(
    aA, obs, pot, tole, tolzmax, tolrperi, tolrap, ntimes=1001, inclphi=False
):
    times = numpy.linspace(0.0, 100.0, ntimes)
    obs.integrate(times, pot, method="dopr54_c")
    if inclphi:
        es, zmaxs, rperis, raps = aA.EccZmaxRperiRap(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
        )
    else:
        es, zmaxs, rperis, raps = aA.EccZmaxRperiRap(
            obs.R(times), obs.vR(times), obs.vT(times), obs.z(times), obs.vz(times)
        )
    assert numpy.amax(numpy.fabs(es / numpy.mean(es) - 1)) < 10.0**tole, (
        "Eccentricity conservation fails at %g%%"
        % (100.0 * numpy.amax(numpy.fabs(es / numpy.mean(es) - 1)))
    )
    assert numpy.amax(numpy.fabs(zmaxs / numpy.mean(zmaxs) - 1)) < 10.0**tolzmax, (
        "Zmax conservation fails at %g%%"
        % (100.0 * numpy.amax(numpy.fabs(zmaxs / numpy.mean(zmaxs) - 1)))
    )
    assert numpy.amax(numpy.fabs(rperis / numpy.mean(rperis) - 1)) < 10.0**tolrperi, (
        "Rperi conservation fails at %g%%"
        % (100.0 * numpy.amax(numpy.fabs(rperis / numpy.mean(rperis) - 1)))
    )
    assert numpy.amax(numpy.fabs(raps / numpy.mean(raps) - 1)) < 10.0**tolrap, (
        "Rap conservation fails at %g%%"
        % (100.0 * numpy.amax(numpy.fabs(raps / numpy.mean(raps) - 1)))
    )
    return None


# Python 2 bug: setting simplefilter to 'always' still does not display
# warnings that were already displayed using 'once' or 'default', so some
# warnings tests fail; need to reset the registry
# Has become an issue at pytest 3.8.0, which seems to have changed the scope of
# filterwarnings (global one at the start is ignored)
def reset_warning_registry(pattern=".*"):
    "clear warning registry for all match modules"
    import re
    import sys

    key = "__warningregistry__"
    for mod in sys.modules.values():
        if hasattr(mod, key) and re.match(pattern, mod.__name__):
            getattr(mod, key).clear()


# Exercise the remaining branches of the pure-Python (c=False) Staeckel
# freqs/angles path: the _actionsFreqs (no-angles) input forms + useu0 +
# close-to-circular fallback, and the angle-wrap / S<=0-turning-point / plunging
# branches that the parity grid does not reach. Just runs them and asserts the
# outputs are finite (correctness is covered by the c-vs-Python parity test).
def test_actionAngleStaeckel_python_freqsAngles_branches():
    import warnings

    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, vcirc

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAS = actionAngleStaeckel(pot=lp, delta=0.5, c=False)
    aAS_u0 = actionAngleStaeckel(pot=lp, delta=0.5, c=False, useu0=True)
    vc = vcirc(lp, 1.0, use_physical=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # _actionsFreqs (no angles): 5-arg, 6-arg (with phi), and Orbit input
        for out in (
            aAS.actionsFreqs(1.0, 0.1, 0.9, 0.2, 0.1),
            aAS.actionsFreqs(1.0, 0.1, 0.9, 0.2, 0.1, 0.3),
            aAS.actionsFreqs(Orbit([1.0, 0.1, 0.9, 0.2, 0.1, 0.3])),
            aAS_u0.actionsFreqs(1.0, 0.1, 0.9, 0.2, 0.1),  # useu0 -> calcu0
            aAS.actionsFreqs(1.0, 0.0, vc, 0.0, 0.0),  # circular fallback
            aAS_u0.actionsFreqs(1.0, 0.0, vc, 0.0, 0.0),  # circular + useu0
        ):
            for o in out:
                assert numpy.all(numpy.isfinite(numpy.atleast_1d(o)))
        # Angle/turning-point branches: exact peri/apo (vr=0), near-radial
        # (plunging, sharp turning points -> S<=0 guards / umin->0), highly
        # inclined (vmin small), and a range of phi to hit the +/-2pi wraps.
        r2 = numpy.sqrt(1.0**2 + 0.3**2)
        vcr2 = vcirc(lp, r2, use_physical=False)
        branch_ics = [
            (0.9, 0.0, 1.4 * vc, 0.3, 0.0),  # vr=0 pericenter (z!=0)
            (0.9, 0.0, 0.6 * vc, 0.3, 0.0),  # vr=0 apocenter
            (1.0, 0.9 * vc, 0.02 * vc, 0.0, 0.05),  # near-radial / plunging
            (1.0, 0.6 * vc, 0.18 * vc, 0.0, 0.5 * vc),  # eccentric, very inclined
            (1.0, 0.0, 0.7 * vcr2, 0.0, 0.7 * vcr2),  # large vz, strong z-motion
        ]
        for ic in branch_ics:
            for phi in (0.0, 1.5, 3.0, 5.5, 6.0):
                out = aAS.actionsFreqsAngles(*ic, phi)
                for o in out:
                    assert numpy.all(numpy.isfinite(numpy.atleast_1d(o)))
    return None


# The four actionAngle entry points map a MISSING implementation method to
# NotImplementedError. They must not also swallow an AttributeError raised from
# INSIDE an implementation that does exist: that turned real failures (e.g. handing
# backend arrays to the C code) into a misleading "method not implemented".
_AA_ENTRY_POINTS = [
    ("__call__", "_evaluate"),
    ("actionsFreqs", "_actionsFreqs"),
    ("actionsFreqsAngles", "_actionsFreqsAngles"),
    ("EccZmaxRperiRap", "_EccZmaxRperiRap"),
]


@pytest.mark.parametrize("public,private", _AA_ENTRY_POINTS)
def test_actionAngle_missing_method_raises_notimplemented(public, private):
    from galpy.actionAngle import actionAngle

    aA = actionAngle()  # base class implements none of them
    assert not hasattr(aA, private), (
        f"test assumes the base class has no {private}; it now does"
    )
    with pytest.raises(NotImplementedError) as excinfo:
        getattr(aA, public)(1.0, 0.1, 1.1, 0.1, 0.1)
    assert public.strip("_") in str(excinfo.value), (
        f"NotImplementedError for {public} should name the method, got: {excinfo.value}"
    )
    return None


@pytest.mark.parametrize("public,private", _AA_ENTRY_POINTS)
def test_actionAngle_inner_attributeerror_is_not_masked(public, private):
    """An AttributeError from inside the implementation must propagate unchanged."""
    from galpy.actionAngle import actionAngle

    sentinel = "inner attribute failure, not a missing method"

    class _Boom(actionAngle):
        def __init__(self):
            actionAngle.__init__(self)

    def _raise(*args, **kwargs):
        raise AttributeError(sentinel)

    setattr(_Boom, private, _raise)
    aA = _Boom()
    with pytest.raises(AttributeError) as excinfo:
        getattr(aA, public)(1.0, 0.1, 1.1, 0.1, 0.1)
    assert sentinel in str(excinfo.value), (
        f"{public} masked the inner AttributeError; got: {excinfo.value}"
    )
    return None


def test_actionAngleIsochroneInverse_kepler_bracketing_fallback():
    # The Halley iteration solves Kepler's equation for every angle at once
    # and converges for any e < 1, so the bracketing fallback is not reached
    # in normal use. It still has to be right when it is, so force it: with
    # the acceptance residual set negative every angle is handed to brentq,
    # and the two solvers must agree
    import numpy

    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    aAII = actionAngleIsochroneInverse(ip=IsochronePotential(amp=1.2, b=0.6))
    angler = numpy.linspace(0.01, 2.0 * numpy.pi - 0.01, 32)
    ref = numpy.array(
        [
            numpy.atleast_1d(q)
            for q in aAII._xvFreqs(0.1, 0.7, 0.2, angler, angler * 0.7, angler * 1.3)[
                :6
            ]
        ]
    )
    # the package __init__ rebinds this name to the class, so reach the
    # module through sys.modules rather than by importing it
    import sys

    _m = sys.modules["galpy.actionAngle.actionAngleIsochroneInverse"]
    tol = _m._KEPLER_RESID_TOL
    try:
        _m._KEPLER_RESID_TOL = -1.0  # every angle goes to the bracketing solve
        got = numpy.array(
            [
                numpy.atleast_1d(q)
                for q in aAII._xvFreqs(
                    0.1, 0.7, 0.2, angler, angler * 0.7, angler * 1.3
                )[:6]
            ]
        )
    finally:
        _m._KEPLER_RESID_TOL = tol
    assert numpy.max(numpy.fabs(got - ref)) < 1e-12, (
        "The bracketing fallback disagrees with the Halley iteration: %g"
        % numpy.max(numpy.fabs(got - ref))
    )
    return None


def test_actionAngleStaeckel_nearaxis_c_python_parity():
    # main had no near-axis Staeckel coverage, which is how two defects shipped:
    # (a) an axis-reaching orbit (Lz == 0) drove the Lz^2 cosh(u)/sinh^3(u) term
    # to 0/0 at umin = 0 inside the chi-anomaly edge reconstruction, so c=True
    # returned NaN actions once the order was raised to ~50; and (b) the C chi
    # quadrature was a single `order`-point rule over the whole anomaly where the
    # pure-Python path uses max(2*order,20) panels of a 10-point rule, leaving C
    # ~4.5e-5 off near the axis and floored -- more nodes did not help, because
    # they crowded into the region where Q is a MODEL of S rather than S itself.
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    Rg = numpy.array([0.8, 1.0, 1.3])
    vRg = numpy.array([0.2, 0.45])
    vTg = numpy.array([0.0, 1e-4, 3e-4])  # Lz = 0, ~1e-4, ~3e-4
    zg = numpy.array([0.1, 0.28])
    G = numpy.meshgrid(Rg, vRg, vTg, zg, indexing="ij")
    R, vR, vT, z = (g.ravel() for g in G)
    vz = 0.1 * numpy.ones_like(R)
    aAF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    aAC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    jr_f, _, jz_f = aAF(R, vR, vT, z, vz)
    jr_c, _, jz_c = aAC(R, vR, vT, z, vz)
    jr_f, jz_f, jr_c, jz_c = (numpy.atleast_1d(x) for x in (jr_f, jz_f, jr_c, jz_c))
    assert numpy.all(numpy.isfinite(jr_c)), (
        "C actionAngleStaeckel jr is not finite for near-axis orbits"
    )
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7, atol=1e-10)
    numpy.testing.assert_allclose(jz_f, jz_c, rtol=1e-7, atol=1e-10)
    # the NaN regression: a high order puts nodes right at the endpoint, which is
    # where the 0/0 used to bite
    for order in (50, 200):
        jr_hi = numpy.atleast_1d(aAC(R, vR, vT, z, vz, order=order)[0])
        assert numpy.all(numpy.isfinite(jr_hi)), (
            "C actionAngleStaeckel jr returns NaN for axis-reaching orbits at "
            "order=%i" % order
        )
    # jr must not jump between an exactly-radial orbit and a nearly-radial one
    aAC2 = actionAngleStaeckel(pot=MWPotential2014, delta=0.71, c=True)
    j0 = numpy.atleast_1d(aAC2(1.0, 0.0, 0.0, 0.0, 0.0)[0])[0]
    j1 = numpy.atleast_1d(aAC2(1.0, 0.0, 1e-7, 0.0, 0.0)[0])[0]
    assert numpy.fabs(j0 - j1) < 1e-6, (
        "C actionAngleStaeckel jr is discontinuous at Lz = 0"
    )
    return None


def test_actionAngleStaeckel_delta_from_potential():
    # A potential that supplies its own focal length does not need delta=:
    # an exact Staeckel potential and an OblateStaeckelWrapperPotential give
    # the same actions, frequencies, and angles as with delta= given; a
    # potential without one still requires it
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        MWPotential2014,
        OblateStaeckelWrapperPotential,
    )

    R, vR, vT, z, vz, phi = 1.0, 0.1, 1.05, 0.1, 0.05, 2.0
    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    for pot, delta in (
        (kksp, 1.4),
        (OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45), 0.45),
        # a composite potential whose components share a focal length
        (kksp + KuzminKutuzovStaeckelPotential(amp=0.3, ac=2.0, Delta=1.4), 1.4),
    ):
        given = numpy.array(
            actionAngleStaeckel(pot=pot, delta=delta, c=False).actionsFreqsAngles(
                R, vR, vT, z, vz, phi
            )
        )
        own = numpy.array(
            actionAngleStaeckel(pot=pot, c=False).actionsFreqsAngles(
                R, vR, vT, z, vz, phi
            )
        )
        assert numpy.all(given == own), (
            "actionAngleStaeckel with the potential's own focal length does not agree with delta= given"
        )
    with pytest.raises(OSError, match="delta="):
        actionAngleStaeckel(pot=MWPotential2014)
    with pytest.raises(OSError, match="delta="):
        # components with different focal lengths do not supply one
        actionAngleStaeckel(
            pot=kksp + KuzminKutuzovStaeckelPotential(amp=0.3, ac=2.0, Delta=1.1)
        )
    return None


def test_actionAngleStaeckel_perfect_ellipsoid_delta():
    # An axisymmetric oblate perfect ellipsoid is a Staeckel potential in the
    # prolate spheroidal coordinates of focal length a sqrt(1 - c^2), which
    # it supplies to actionAngleStaeckel: the actions along an orbit are
    # then conserved to the quadrature's accuracy, and they are those with
    # the focal length given; a triaxial or a prolate perfect ellipsoid
    # supplies none
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import PerfectEllipsoidPotential

    a, c = 1.3, 0.6
    pot = PerfectEllipsoidPotential(amp=1.0, a=a, b=1.0, c=c, normalize=1.0)
    o = Orbit([1.0, 0.3, 0.9, 0.2, 0.25, 0.0])
    ts = numpy.linspace(0.0, 100.0, 1001)
    o.integrate(ts, pot, method="dop853_c")
    own = actionAngleStaeckel(pot=pot, c=True, order=100)
    jr, lz, jz = own(o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts))
    assert numpy.ptp(jr) < 1e-10 and numpy.ptp(jz) < 1e-10, (
        "The actions of an orbit in an oblate perfect ellipsoid are not conserved with the potential's own focal length: %g %g"
        % (numpy.ptp(jr), numpy.ptp(jz))
    )
    given = actionAngleStaeckel(
        pot=pot, delta=a * numpy.sqrt(1.0 - c**2), c=True, order=100
    )
    assert numpy.all(
        numpy.array(
            given(o.R(ts[:5]), o.vR(ts[:5]), o.vT(ts[:5]), o.z(ts[:5]), o.vz(ts[:5]))
        )
        == numpy.array(
            own(o.R(ts[:5]), o.vR(ts[:5]), o.vT(ts[:5]), o.z(ts[:5]), o.vz(ts[:5]))
        )
    ), "The perfect ellipsoid's own focal length does not agree with delta= given"
    for kw in (dict(b=0.8, c=0.6), dict(b=1.0, c=1.4)):
        with pytest.raises(OSError, match="delta="):
            actionAngleStaeckel(
                pot=PerfectEllipsoidPotential(amp=1.0, a=a, normalize=1.0, **kw)
            )
    return None


def test_actionAngleVerticalInverse_polynomial_pt_true_action():
    # With a polynomial point transformation, each torus's stored action used
    # to be the mean auxiliary action in the sheared gauge, which is off from
    # the torus's actual action by O(residual) (4e-4 at degree 7, growing
    # with energy): asking for a torus's actual action returned a neighbor,
    # and the top torus's actual action lay outside the stored range and
    # crashed the interpolated lookup. The offset is now computed in closed
    # form and subtracted, so the stored action is the actual one while the
    # Fourier structure keeps the point-transformed action as its base.
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    Es = numpy.linspace(0.0, 2.0, 9)
    angles = numpy.linspace(0.05, 6.2, 41)
    for setup_interp in (False, True):
        aAVI = actionAngleVerticalInverse(
            pot=pot, Es=Es, nta=128, use_pointtransform=True, setup_interp=setup_interp
        )
        # the stored action is the actual one; the point-transformed action
        # the Fourier structure is built on differs from it by the offset
        assert numpy.amax(numpy.fabs(aAVI._jaoffset) / aAVI._js[1:].min()) > 1e-5, (
            "The polynomial point transformation has no action offset to correct"
        )
        for ii in range(1, len(Es)):  # including the top torus
            jtrue = float(
                numpy.asarray(aAV(0.0, numpy.sqrt(2.0 * Es[ii]))[0]).ravel()[0]
            )
            assert numpy.fabs(aAVI.J(Es[ii]) / jtrue - 1.0) < 1e-8, (
                "J(E) does not return the torus's actual action"
            )
            x, v = aAVI(jtrue, angles)
            assert numpy.all(numpy.isfinite(x)) and numpy.all(numpy.isfinite(v)), (
                "Requesting the actual action of a torus fails (setup_interp=%s)"
                % setup_interp
            )
            assert numpy.amax(numpy.fabs(aAV(x, v)[0] - jtrue)) / jtrue < 1e-7, (
                "Requesting the actual action of a torus returns a different torus "
                "(setup_interp=%s)" % setup_interp
            )
            assert numpy.isfinite(float(aAVI.Freqs(jtrue))), (
                "Freqs fails at the actual action"
            )
    # without a point transformation, and with the exact one, there is no
    # offset: the mean auxiliary action IS the torus's action
    for kwargs in (dict(momentum_matched=False), dict(use_pointtransform="exact")):
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, **kwargs)
        assert numpy.all(aAVI._jaoffset == 0.0), (
            "An action offset appeared for a non-polynomial mode"
        )
    return None


def test_actionAngleVerticalInverse_polynomial_pt_offset_closed_form():
    # With a polynomial point transformation, the mean auxiliary action of
    # the sheared gauge v_a = v / pi' -- the point-transformed action J^A that
    # each torus's Fourier structure is built on -- differs from the torus's
    # actual action. Because (x_a, v_a) -> (j_a, theta_a) is the harmonic
    # action-angle map, 2 pi <j_a> is the loop integral of v_a dx_a =
    # v pi'^-2 dx, so the offset is the loop integral
    #     J^A - J = (1 / 2 pi) Int v(x) [ pi'(x_a)^-2 - 1 ] dx
    # along the orbit, from the potential and the fitted transformation
    # alone. The class computes it at construction and stores the actual
    # action J = <j_a> - offset, which must then agree with the forward
    # transformation's action, an independent quadrature, on every torus
    # and for every degree; and the offset is what it claims to be: the
    # difference between the mean auxiliary action and the stored action
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    Es = numpy.linspace(0.0, 2.0, 9)
    jforward = numpy.array(
        [
            float(numpy.asarray(aAV(0.0, numpy.sqrt(2.0 * E))[0]).ravel()[0])
            for E in Es[1:]
        ]
    )
    for pt_deg in (3, 7, 11):
        aAVI = actionAngleVerticalInverse(
            pot=pot, Es=Es, nta=128, use_pointtransform=True, pt_deg=pt_deg
        )
        assert numpy.amax(numpy.fabs(aAVI._jaoffset[1:]) / jforward) > 1e-8, (
            "The offset is not there to be corrected (degree %d)" % pt_deg
        )
        assert numpy.amax(numpy.fabs(aAVI._js[1:] - jforward) / jforward) < 1e-8, (
            "The stored action, mean auxiliary action minus the closed-form offset, "
            "does not agree with the forward transformation (degree %d)" % pt_deg
        )
        assert (
            numpy.amax(
                numpy.fabs(numpy.nanmean(aAVI._ja, axis=1) - aAVI._js - aAVI._jaoffset)
            )
            < 1e-15
        ), "The offset is not the difference it is defined as"
    # and the exact point transformation has no offset
    aAVI = actionAngleVerticalInverse(
        pot=pot, Es=Es, nta=128, use_pointtransform="exact"
    )
    assert numpy.all(aAVI._jaoffset == 0.0), (
        "The exact point transformation has an action offset"
    )
    assert numpy.amax(numpy.fabs(aAVI._js[1:] - jforward) / jforward) < 1e-8, (
        "The exact point transformation's stored action disagrees with the forward "
        "transformation"
    )
    return None


def test_actionAngleVerticalInverse_momentum_matched_reconstruction():
    # The momentum-matched map returns points that lie on the requested
    # torus: at every grid torus the energy of (x, v) is the torus's energy
    # and the forward transformation returns the requested action. The map
    # is exact up to the truncation of its anomaly series, so refining
    # mm_npt converges spectrally rather than at some fixed order, and the
    # anomaly samples mm_nta only need to resolve that series
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    Es = numpy.linspace(0.0, 2.0, 9)
    angles = numpy.linspace(0.05, 6.2, 41)

    def worst(**kwargs):
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, **kwargs)
        wE, wJ = 0.0, 0.0
        for E in Es[1:]:
            J = aAVI.J(E)
            x, v = aAVI(J, angles)
            H = 0.5 * v**2.0 + evaluatelinearPotentials(pot, x, use_physical=False)
            wE = max(wE, numpy.amax(numpy.fabs(H / E - 1.0)))
            wJ = max(wJ, numpy.amax(numpy.fabs(aAV(x, v)[0] - J)) / J)
        return wE, wJ

    wE8, _ = worst(mm_npt=8)
    wE12, _ = worst(mm_npt=12)
    wE28, wJ28 = worst(mm_npt=28)
    assert wE28 < 1e-9, "The reconstruction does not return the torus: %g" % wE28
    assert wJ28 < 1e-9, "The reconstruction does not return the action: %g" % wJ28
    assert wE28 < 1e-2 * wE12 and wE12 < 3e-1 * wE8, (
        "The reconstruction error is not limited by the anomaly-map "
        "truncation: %g, %g, %g" % (wE8, wE12, wE28)
    )
    # the samples resolve the series once there are more than four per
    # harmonic; beyond that the map is converged
    aAVI1 = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, mm_nta=192)
    aAVI2 = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, mm_nta=1024)
    for E in Es[1:]:
        x1, v1 = aAVI1(aAVI1.J(E), angles)
        x2, v2 = aAVI2(aAVI2.J(E), angles)
        assert numpy.amax(numpy.fabs(x1 - x2)) < 1e-9, (
            "The map depends on the number of anomaly samples beyond the resolved series"
        )
        assert numpy.amax(numpy.fabs(v1 - v2)) < 1e-9, (
            "The map depends on the number of anomaly samples beyond the resolved series"
        )
    with pytest.raises(ValueError) as excinfo:
        actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, mm_npt=20, mm_nta=64)
    assert "mm_nta" in str(excinfo.value)
    # and the default number of samples is safe whatever nta is
    aAVI3 = actionAngleVerticalInverse(pot=pot, Es=Es, nta=16)
    x3, v3 = aAVI3(aAVI3.J(1.0), angles)
    x2, v2 = aAVI2(aAVI2.J(1.0), angles)
    assert numpy.amax(numpy.fabs(x3 - x2)) < 1e-9, (
        "A small nta breaks the momentum-matched map"
    )
    return None


def test_actionAngleVerticalInverse_momentum_matched_angles():
    # End to end: enter at a requested action and angle, come out at a point,
    # and let the forward transformation say whether it is the right one.
    # The family carries the action derivatives of every torus (Hermite
    # constraints), so at a grid torus the angle is at the floor of the
    # map's truncation whatever the grid; the frequency likewise, because
    # E(J) is a Hermite spline through the exactly known dE/dJ
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    # off the turning points, where the FORWARD transformation cannot place
    # an angle: there p = 0 and the angle is 0 or pi by definition
    th = 2.0 * numpy.pi * (numpy.arange(400) + 0.37) / 400.0

    def errs(nE):
        Es = numpy.linspace(0.0, 2.0, nE)
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
        j = aAVI.J(Es[(nE - 1) // 2])
        x, v = aAVI(j, th)
        xf, vf, Om = aAVI.xvFreqs(j, th)
        assert numpy.amax(numpy.fabs(xf - x)) == 0.0, "__call__ and xvFreqs disagree"
        assert numpy.amax(numpy.fabs(vf - v)) == 0.0, "__call__ and xvFreqs disagree"
        assert Om == aAVI.Freqs(j), "xvFreqs and Freqs disagree"
        jf, Omf, thfwd = aAV.actionsFreqsAngles(x, v)
        dth = numpy.fabs((thfwd - th + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        return (
            numpy.amax(numpy.fabs(jf - j)),
            numpy.amax(dth),
            numpy.amax(numpy.fabs(Omf - Om)) / numpy.mean(Omf),
        )

    for nE in (9, 33):
        dj, dth, dom = errs(nE)
        assert dj < 1e-10, "The evaluation leaves the requested torus: %g" % dj
        assert dth < 1e-9, "The evaluated angle is wrong: %g" % dth
        assert dom < 1e-9, "The frequency is not exact at the nodes: %g" % dom
    return None


def test_actionAngleVerticalInverse_momentum_matched_turning_points():
    # The map's factors diverge and vanish together at the turning points
    # and are grouped so that the evaluation is finite there and next to
    # them: at the turning-point angles the velocity vanishes and the
    # potential alone carries the energy, and the orbit is smooth through
    # them. The zero-action torus is the point at the bottom, and the map
    # goes over to the harmonic oscillator there with corrections linear in
    # the action
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    Es = numpy.linspace(0.0, 2.0, 9)
    aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
    for E in Es[1:]:
        J = aAVI.J(E)
        x, v = aAVI(J, numpy.array([0.5 * numpy.pi, 1.5 * numpy.pi]))
        assert numpy.amax(numpy.fabs(v)) < 1e-14, (
            "The velocity does not vanish at the turning points: %g"
            % numpy.amax(numpy.fabs(v))
        )
        assert (
            numpy.amax(
                numpy.fabs(
                    evaluatelinearPotentials(pot, x, use_physical=False) / E - 1.0
                )
            )
            < 1e-12
        ), "The turning points do not sit at the energy"
        assert numpy.fabs(x[0] + x[1]) < 1e-14, "The turning points are not symmetric"
        xmax = x[0]
        # smooth through the turning points: x is stationary, v linear
        for delta in (1e-9, 1e-6, 1e-4):
            th = 0.5 * numpy.pi + numpy.array([-delta, 0.0, delta])
            for tth in (th, th + numpy.pi):
                x, v = aAVI(J, tth)
                assert numpy.all(numpy.isfinite(x)) and numpy.all(numpy.isfinite(v)), (
                    "The evaluation is not finite next to a turning point"
                )
                assert numpy.amax(numpy.fabs(x - x[1])) < xmax * delta**2.0, (
                    "The orbit is not stationary in x through a turning point"
                )
                assert numpy.amax(numpy.fabs(v)) < 4.0 * xmax * delta, (
                    "The velocity does not vanish linearly at a turning point"
                )
                assert (
                    numpy.fabs(v[0] + v[2]) < 1e-6 * numpy.fabs(v[0] - v[2]) + 1e-15
                ), "The velocity is not odd about a turning point"
    # the zero-action torus is a point
    x, v = aAVI(0.0, numpy.linspace(0.0, 2.0 * numpy.pi, 11))
    assert numpy.all(x == 0.0) and numpy.all(v == 0.0), (
        "The zero-action torus is not the point at the bottom"
    )
    with pytest.raises(ValueError) as excinfo:
        aAVI(-0.1, numpy.array([0.3]))
    assert "non-negative" in str(excinfo.value)
    # the harmonic limit: xmax^2 omega / (2 J) -> 1 and Omega -> omega, with
    # the anharmonic corrections vanishing linearly in J
    omega0 = numpy.sqrt(4.0 * numpy.pi)
    th = numpy.linspace(0.0, 2.0 * numpy.pi, 1001)
    J1 = aAVI.J(Es[1])
    prev = None
    for J in (1e-4 * J1, 1e-3 * J1, 1e-2 * J1):
        x, v = aAVI(J, th)
        dev = numpy.array(
            [
                numpy.amax(x) ** 2.0 * omega0 / (2.0 * J) - 1.0,
                numpy.amax(v) ** 2.0 / (2.0 * J * omega0) - 1.0,
                aAVI.Freqs(J) / omega0 - 1.0,
            ]
        )
        assert numpy.amax(numpy.fabs(dev)) < 30.0 * J / J1, (
            "The map does not go over to the harmonic oscillator at small action"
        )
        if prev is not None:
            ratio = numpy.fabs(dev) / numpy.fabs(prev)
            assert numpy.all(ratio > 5.0) and numpy.all(ratio < 20.0), (
                "The anharmonic corrections are not linear in the action"
            )
        prev = dev
    return None


def test_actionAngleVerticalInverse_momentum_matched_symplectic():
    # The point of the construction: the map (J, theta) -> (x, v) is
    # canonical whatever the tables contain, because the evaluation
    # differentiates the same interpolants it reads. So the Poisson bracket
    # {x, v} in (theta, J), by finite differences of the public map, is one
    # at the grid tori and between them -- also on a grid so coarse that
    # the interpolation error between tori is large. The older evaluation
    # keeps separate tables for the generating function and its action
    # derivative and has a symplectic defect even at its grid tori
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    angles = numpy.linspace(0.05, 6.2, 41)

    def bracket(aAVI, J, h=1e-5):
        hj = h * J
        xp, vp = aAVI(J, angles + h)
        xm, vm = aAVI(J, angles - h)
        xJp, vJp = aAVI(J + hj, angles)
        xJm, vJm = aAVI(J - hj, angles)
        return numpy.amax(
            numpy.fabs(
                (xp - xm) / (2.0 * h) * (vJp - vJm) / (2.0 * hj)
                - (xJp - xJm) / (2.0 * hj) * (vp - vm) / (2.0 * h)
                - 1.0
            )
        )

    for nE in (3, 9):
        Es = numpy.linspace(0.0, 2.0, nE)
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
        js = numpy.array([aAVI.J(E) for E in Es])
        # at a node, and in the middle of every interval, the first and the
        # last included
        for J in numpy.concatenate([js[1:], 0.5 * (js[:-1] + js[1:])]):
            assert bracket(aAVI, J) < 1e-6, (
                "The momentum-matched map is not canonical (%d energies, J = %g)"
                % (nE, J)
            )
        if nE == 3:
            # while on this grid the interpolation error between the tori
            # is large, so the canonicity is not the accuracy of the tables
            J = 0.5 * (js[1] + js[2])
            x, v = aAVI(J, angles)
            assert numpy.amax(numpy.fabs(aAV(x, v)[0] - J)) / J > 1e-5, (
                "The coarse grid is too accurate for this check to mean anything"
            )
    old = actionAngleVerticalInverse(
        pot=pot,
        Es=numpy.linspace(0.0, 2.0, 9),
        nta=128,
        momentum_matched=False,
        setup_interp=True,
    )
    assert bracket(old, float(old.J(1.0))) > 1e-4, (
        "The older evaluation has no symplectic defect to contrast with"
    )
    return None


def test_actionAngleVerticalInverse_momentum_matched_is_the_default():
    # The canonical map is the default, so that the inverse methods agree
    # with each other, for any number of energies down to one; the older
    # evaluation remains reachable, and its coefficient tables do not exist
    # under the canonical map
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    Es = numpy.linspace(0.0, 2.0, 9)
    angles = numpy.linspace(0.05, 6.2, 41)
    aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, setup_interp=True)
    for func in (aAVI.nSn, aAVI.dSndJ, aAVI.pt_coeffs, aAVI.pt_deriv_coeffs):
        with pytest.raises(RuntimeError) as excinfo:
            func(1.0)
        assert "momentum_matched" in str(excinfo.value), (
            "The older evaluation's tables did not raise under the canonical map"
        )
    # explicitly off, and off through the older point transformation, since
    # the momentum-matched map is itself a point transformation
    for kwargs in (
        {"momentum_matched": False},
        {"use_pointtransform": True, "pt_deg": 7},
    ):
        old = actionAngleVerticalInverse(
            pot=pot, Es=Es, nta=128, setup_interp=True, **kwargs
        )
        assert numpy.all(numpy.isfinite(old.nSn(1.0))), (
            "The older evaluation is no longer reachable"
        )
    # any number of energies will do, down to one, and each family returns
    # its tori
    for tEs in (Es[:3], Es[:2], Es[1:3], [Es[2]]):
        small = actionAngleVerticalInverse(pot=pot, Es=tEs, nta=128)
        for E in tEs:
            if E == 0.0:
                continue
            J = small.J(E)
            x, v = small(J, angles)
            jf, _, thf = aAV.actionsFreqsAngles(x, v)
            assert numpy.amax(numpy.fabs(jf - J)) / J < 1e-9, (
                "A %d-torus family does not return its torus" % len(tEs)
            )
            assert (
                numpy.amax(
                    numpy.fabs((thf - angles + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
                )
                < 1e-9
            ), "A %d-torus family does not return its torus's angles" % len(tEs)
    # the degenerate grid of only the harmonic bottom builds
    bottom = actionAngleVerticalInverse(pot=pot, Es=[0.0], nta=128)
    x, v = bottom(0.0, angles)
    assert numpy.all(x == 0.0) and numpy.all(v == 0.0), (
        "The bottom-only family does not build as the point at the bottom"
    )
    return None


def test_actionAngleVerticalInverse_momentum_matched_offnode_frequency():
    # The canonical map can evaluate between grid tori, so it can report a
    # frequency there too, and it is the map's own (the derivative of the
    # energy interpolant the map reads), so it agrees with what xvFreqs
    # reports; the tabulated frequencies of the older evaluation cannot,
    # and raise
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    Es = numpy.linspace(0.0, 2.0, 9)
    aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
    jm = 0.5 * (aAVI.J(Es[3]) + aAVI.J(Es[4]))
    Om = aAVI.Freqs(jm)
    assert numpy.isfinite(Om) and Om > 0.0, "No frequency between the grid tori"
    assert Om == aAVI.xvFreqs(jm, numpy.array([0.3]))[2], (
        "The off-node frequency is not the map's own"
    )
    # on a node it is the true frequency of the torus
    J = aAVI.J(Es[4])
    x, v = aAVI(J, numpy.array([0.3]))
    assert numpy.fabs(aAVI.Freqs(J) / aAV.actionsFreqs(x, v)[1][0] - 1.0) < 1e-9, (
        "The on-node frequency is not the torus's"
    )
    # and with the older evaluation the off-node case raises
    old = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, momentum_matched=False)
    with pytest.raises(ValueError) as excinfo:
        old.Freqs(jm)
    assert "not found" in str(excinfo.value)
    return None


def test_actionAngleVerticalInverse_momentum_matched_between_tori():
    # Between the grid tori the evaluation reads interpolated tables, and
    # two things used to spoil the end intervals while the interior was fine:
    # a mirror-symmetric spline extension that imposed a zero slope at both
    # ends of the grid (D grows linearly out of the harmonic bottom), and a
    # zero-energy node whose frequency was a placeholder copied from the
    # first torus, off by O(J_1). Both cost ~1e-2 in the action in the end
    # intervals on a nine-node grid, converging only at first order. The
    # tables are now Hermite splines in the action, which also need no
    # assumption about the spacing of the energy grid, so a non-uniform
    # grid must do as well as a uniform one.
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    angles = numpy.linspace(0.05, 6.2, 41)

    def worst_between(n, uniform):
        r = numpy.linspace(0.0, 1.0, n)
        Es = 2.0 * r if uniform else 2.0 * (0.55 * r + 0.45 * r**2.5)
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
        js = numpy.array([aAVI.J(E) for E in Es])
        # at the nodes the construction preserves the action whatever the
        # tables contain
        for J in js[1:]:
            x, v = aAVI(J, angles)
            assert numpy.amax(numpy.fabs(aAV(x, v)[0] - J)) < 1e-7, (
                "The evaluation leaves the requested torus at a grid node"
            )
        # between EVERY pair of nodes, including the first and the last
        # interval, which are the ones that used to be wrong
        out = 0.0
        for jm in 0.5 * (js[:-1] + js[1:]):
            x, v = aAVI(jm, angles)
            out = max(out, numpy.amax(numpy.fabs(aAV(x, v)[0] - jm)) / jm)
        return out

    for uniform in (True, False):
        w9, w33 = worst_between(9, uniform), worst_between(33, uniform)
        assert w9 < 2e-4, (
            "The evaluation between grid tori is inaccurate on a %s grid: %g"
            % ("uniform" if uniform else "non-uniform", w9)
        )
        assert w33 < 1e-6, (
            "The evaluation between grid tori is inaccurate on a %s grid: %g"
            % ("uniform" if uniform else "non-uniform", w33)
        )
        # and it converges with the grid at the interpolant's order rather
        # than at the first order the edge defects imposed
        assert w33 < 3e-2 * w9, (
            "The evaluation between grid tori does not converge with the grid"
        )
    return None


def test_actionAngleVerticalInverse_momentum_matched_bottom_interval():
    # The bottom node is the harmonic limit, where the map is the identity
    # and its action derivatives come from the polynomial through the exact
    # bottom value and the first two tori, so the bottom interval is as
    # accurate as the rest and converges with the grid; a family of one
    # torus agrees with the same torus inside a larger family, so the fit
    # does not depend on its starting point
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    angles = numpy.linspace(0.05, 6.2, 41)
    prev = None
    for nE in (9, 17, 33):
        Es = numpy.linspace(0.0, 2.0, nE)
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
        J1 = aAVI.J(Es[1])
        worst = 0.0
        for J in (0.1 * J1, 0.5 * J1, 0.9 * J1):
            x, v = aAVI(J, angles)
            worst = max(worst, numpy.amax(numpy.fabs(aAV(x, v)[0] - J)) / J)
        assert worst < 1e-5, (
            "The family is inaccurate in the bottom interval (%d energies): %g"
            % (nE, worst)
        )
        if prev is not None:
            assert worst < 0.3 * prev, (
                "The bottom interval does not converge with the grid"
            )
        prev = worst
    single = actionAngleVerticalInverse(pot=pot, Es=[1.0], nta=128)
    family = actionAngleVerticalInverse(
        pot=pot, Es=numpy.linspace(0.0, 2.0, 9), nta=128
    )
    xs, vs = single(single.J(1.0), angles)
    xf, vf = family(family.J(1.0), angles)
    assert numpy.amax(numpy.fabs(xs - xf)) < 1e-12, (
        "A torus depends on the family it is built in"
    )
    assert numpy.amax(numpy.fabs(vs - vf)) < 1e-12, (
        "A torus depends on the family it is built in"
    )
    return None


def test_actionAngleVerticalInverse_zero_energy_frequency():
    # The zero-energy torus is the harmonic oscillator at the midplane, so its
    # frequency is sqrt(Phi''(0)) -- for the isothermal disk with amp=1 that is
    # sqrt(4 pi) exactly, whatever sigma. It used to be a placeholder copied
    # from the first torus, off by O(J_1), which the momentum-matched family
    # then interpolated through; it now comes from the potential's second
    # derivative, so it is exact to round-off
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    omega0 = numpy.sqrt(4.0 * numpy.pi)
    for momentum_matched in (True, False):
        aAVI = actionAngleVerticalInverse(
            pot=pot,
            Es=numpy.linspace(0.0, 2.0, 9),
            nta=128,
            momentum_matched=momentum_matched,
        )
        assert numpy.fabs(float(aAVI.Freqs(0.0)) / omega0 - 1.0) < 1e-14, (
            "Freqs at zero action is not the midplane's harmonic frequency"
        )
    return None


def test_actionAngleVerticalInverse_momentum_matched_interpolation():
    # With setup_interp=True, J(E) and E(J) are available between the grid
    # tori, from the same Hermite energy interpolant whose derivative is the
    # frequency, so E(J(E)) = E to round-off, Freqs is the derivative of E,
    # and the map evaluated at J(E) returns a torus of energy E to the
    # family's interpolation accuracy -- for any number of tori down to one
    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAV = actionAngleVertical(pot=pot)
    angles = numpy.linspace(0.05, 6.2, 41)
    Et = numpy.array([0.9, 1.0, 1.3])
    for Es in ([1.0], [0.5, 1.5], [0.0, 1.0, 2.0], numpy.linspace(0.0, 2.0, 9)):
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, setup_interp=True)
        Jt = aAVI.J(Et)
        assert numpy.amax(numpy.fabs(aAVI.E(Jt) - Et)) < 1e-12, (
            "E(J(E)) does not return the energy"
        )
        assert (
            numpy.amax(numpy.fabs(Jt - numpy.array([aAVI.J(E) for E in Et]).flatten()))
            == 0.0
        ), "J(E) differs between array and scalar input"
        h = 1e-6
        for J in Jt:
            fd = (aAVI.E(J + h) - aAVI.E(J - h)) / (2.0 * h)
            assert numpy.fabs(aAVI.Freqs(J) / fd - 1.0) < 1e-6, (
                "Freqs is not the derivative of E(J)"
            )
        if len(Es) == 9:
            for E, J in zip(Et, Jt):
                x, v = aAVI(J, angles)
                H = 0.5 * v**2.0 + evaluatelinearPotentials(pot, x, use_physical=False)
                assert numpy.amax(numpy.fabs(H / E - 1.0)) < 1e-5, (
                    "The interpolated torus does not have the requested energy"
                )
                assert numpy.amax(numpy.fabs(aAV(x, v)[0] - J)) / J < 1e-5, (
                    "The interpolated torus does not have the requested action"
                )
            # at a grid energy the interpolant returns the grid torus
            assert numpy.fabs(aAVI.E(aAVI.J(1.0)) - 1.0) < 1e-14
            x, v = aAVI(aAVI.J(1.0), angles)
            assert numpy.amax(numpy.fabs(aAV(x, v)[0] - aAVI.J(1.0))) < 1e-10
    return None


# ---------- actionAngleSphericalInverse tests: the momentum-matched canonical
# ---------- map for spherical potentials (canonical.tex, the spherical case)


def _spherical_inverse_potential():
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(normalize=1.0)


@pytest.fixture(scope="module")
def spherical_inverse_interp():
    # a small (E, L) interpolation grid in the logarithmic halo
    from galpy.actionAngle import actionAngleSphericalInverse

    return actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.4,
        Rinf=6.0,
        nE=8,
        nL=8,
        mm_nta=128,
        mm_npt=24,
    )


@pytest.fixture(scope="module")
def spherical_inverse_explicit():
    # two explicit tori in the logarithmic halo, set up without the progress
    # bar (the interpolated fixture keeps it)
    from galpy.actionAngle import actionAngleSphericalInverse

    return actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        Es=[0.7, 1.1],
        Ls=[0.9, 0.7],
        progressbar=False,
    )


def _spherical_inverse_E_of_u(u, L, Rinf=6.0):
    # the energy at fraction u from the circular orbit's to the grid's top,
    # in the grid's quadratic spacing, for the logarithmic halo (v_c = 1)
    Ec = 0.5 + numpy.log(L)
    return Ec + (numpy.log(Rinf) - Ec) * u**2


def _spherical_inverse_forward_jr(E, L):
    # the forward transformation's own J_r of a point on the (E, L) torus
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import evaluatePotentials, rl

    pot = _spherical_inverse_potential()
    r0 = rl(pot, L, use_physical=False)
    vR = numpy.sqrt(2.0 * (E - evaluatePotentials(pot, r0, 0.0)) - L**2 / r0**2)
    return float(actionAngleSpherical(pot=pot)(r0, vR, L / r0, 0.0, 0.0, 0.0)[0][0])


def _spherical_inverse_roundtrip(aAI, jr, jphi, jz, angler, anglephi, anglez):
    # evaluate the public map and pass the result through the forward
    # transformation: (action, angle, frequency) errors and the energy spread
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import evaluatePotentials

    pot = _spherical_inverse_potential()
    R, vR, vT, z, vz, phi = aAI(jr, jphi, jz, angler, anglephi, anglez)
    f = actionAngleSpherical(pot=pot).actionsFreqsAngles(R, vR, vT, z, vz, phi)
    Om = aAI.Freqs(jr, jphi, jz)
    wrap = lambda d: numpy.amax(
        numpy.fabs((d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
    )
    dJ = numpy.amax(
        numpy.fabs(f[0] - jr) + numpy.fabs(f[1] - jphi) + numpy.fabs(f[2] - jz)
    ) / (jr + jz + numpy.fabs(jphi))
    dth = max(wrap(f[6] - angler), wrap(f[7] - anglephi), wrap(f[8] - anglez))
    dOm = max(
        numpy.amax(numpy.fabs(f[3] / Om[0] - 1.0)),
        numpy.amax(numpy.fabs(f[4] / Om[1] - 1.0)),
        numpy.amax(numpy.fabs(f[5] / Om[2] - 1.0)),
    )
    H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
    return dJ, dth, dOm, numpy.ptp(H) / numpy.fabs(numpy.mean(H))


def _spherical_inverse_symplectic_defect(xvmap, jr, jphi, jz, ar, ap, az, h=1e-6):
    # max |A^T Omega A - Omega| of the finite-difference 6x6 Jacobian of the
    # public map (theta, J) -> (q, p), q = (R, z, phi), p = (v_R, v_z, R v_phi)
    def qp(jr, jphi, jz, ar, ap, az):
        R, vR, vT, z, vz, phi = xvmap(jr, jphi, jz, ar, ap, az)
        return numpy.array([R[0], z[0], phi[0], vR[0], vz[0], R[0] * vT[0]])

    x = [ar, ap, az, jr, jphi, jz]
    A = numpy.empty((6, 6))
    for k in range(6):
        up, dn = list(x), list(x)
        up[k] += h
        dn[k] -= h
        A[:, k] = (qp(*up[3:], *up[:3]) - qp(*dn[3:], *dn[:3])) / (2.0 * h)
    Omega = numpy.zeros((6, 6))
    Omega[:3, 3:] = numpy.eye(3)
    Omega[3:, :3] = -numpy.eye(3)
    return numpy.amax(numpy.fabs(A.T @ Omega @ A - Omega))


def test_actionAngleSphericalInverse_nodes(spherical_inverse_explicit):
    # A discrete family returns its own tori: the action label is the forward
    # transformation's, and the map round-trips through it at the forward
    # code's floor, for inclined orbits and all three angles
    aAI = spherical_inverse_explicit
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    for E, L in ((0.7, 0.9), (1.1, 0.7)):
        jr = aAI.Jr(E, L)
        assert numpy.fabs(jr - _spherical_inverse_forward_jr(E, L)) < 1e-10, (
            "The family's action label is not the forward transformation's"
        )
        for jphi in (0.7 * L, -0.4 * L):
            jz = L - numpy.fabs(jphi)
            dJ, dth, dOm, dH = _spherical_inverse_roundtrip(
                aAI, jr, jphi, jz, angler, anglephi, anglez
            )
            assert dJ < 1e-10, "The map does not return the requested torus: %g" % dJ
            assert dth < 1e-7, "The map does not return the requested angles: %g" % dth
            assert dOm < 1e-8, "The frequencies are not the torus's: %g" % dOm
            assert dH < 1e-10, "The reconstructed loop is not at one energy: %g" % dH
    return None


def test_actionAngleSphericalInverse_orbit(spherical_inverse_explicit):
    # Traversing a torus at its frequencies is an orbit of the potential: at
    # a node the energy is constant along the torus to the map's truncation,
    # and the points agree with an integrated orbit started from the first
    # of them over ten radial periods
    from galpy.orbit import Orbit

    aAI = spherical_inverse_explicit
    pot = _spherical_inverse_potential()
    wrap = lambda d: (d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    for E, L in zip([0.7, 1.1], [0.9, 0.7]):
        jr, jphi, jz = aAI.Jr(E, L), 0.6 * L, 0.4 * L
        Om = aAI.Freqs(jr, jphi, jz)
        ts = numpy.linspace(0.0, 10.0 * 2.0 * numpy.pi / Om[0], 1001)
        R, vR, vT, z, vz, phi = aAI(
            jr, jphi, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + pot(R, z)
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-10, (
            "Energy is not conserved along an actionAngleSphericalInverse torus at a node: %g"
            % (numpy.std(H) / numpy.fabs(numpy.mean(H)))
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, orb.R(ts)),
            ("z", z, orb.z(ts)),
            ("vR", vR, orb.vR(ts)),
            ("vT", vT, orb.vT(ts)),
            ("vz", vz, orb.vz(ts)),
            ("phi", phi, phi + wrap(orb.phi(ts) - phi)),
        ):
            assert numpy.amax(numpy.fabs(orbit - torus)) < 1e-8, (
                "%s along an actionAngleSphericalInverse torus at a node does not agree with the integrated orbit: %g"
                % (name, numpy.amax(numpy.fabs(orbit - torus)))
            )
    return None


def test_actionAngleSphericalInverse_interpolation(spherical_inverse_interp):
    # The interpolated family: exact at a node of its own grid, accurate to
    # the family's interpolation error between nodes, its frequencies those
    # of the returned orbits, and J_r(E, L) the forward transformation's
    aAI = spherical_inverse_interp
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    Lgrid = numpy.linspace(0.7, 1.4, 8)
    # (u, L) = (3/7, Lgrid[4]) is a node of this grid (u runs from the
    # circular edge, 0, to 1 in eight steps), where the action, the angles,
    # and the frequencies are all at the forward transformation's floor,
    # because the family's first partials are exact there (its frequencies
    # and the analytic slopes of its tables are Hermite constraints);
    # (0.55, 0.99) is between nodes, at the family's interpolation error
    for u, L, tolJ, tolth, tolOm in (
        (3.0 / 7.0, Lgrid[4], 1e-10, 1e-7, 1e-8),
        (0.55, 0.99, 1e-4, 1e-4, 1e-4),
    ):
        E = _spherical_inverse_E_of_u(u, L)
        jrf = _spherical_inverse_forward_jr(E, L)
        assert numpy.fabs(aAI.Jr(E, L) / jrf - 1.0) < tolJ, (
            "J_r(E, L) of the family is not the forward transformation's"
        )
        jphi, jz = 0.6 * L, 0.4 * L
        dJ, dth, dOm, dH = _spherical_inverse_roundtrip(
            aAI, jrf, jphi, jz, angler, anglephi, anglez
        )
        assert dJ < tolJ, "The map does not return the requested torus: %g" % dJ
        assert dth < tolth, "The map does not return the requested angles: %g" % dth
        assert dOm < tolOm, "The frequencies are not the torus's: %g" % dOm
        assert dH < tolJ, "The reconstructed loop is not at one energy: %g" % dH
        # Freqs is exactly the frequency of the returned orbits
        assert numpy.all(
            numpy.array(aAI.Freqs(jrf, jphi, jz))
            == numpy.array(aAI.xvFreqs(jrf, jphi, jz, 0.3, 1.0, 2.0)[6:])
        ), "Freqs and xvFreqs disagree"
    return None


def test_actionAngleSphericalInverse_turning_point_derivatives():
    # The closed-form derivatives of the radial turning points with respect
    # to E at fixed L and to L at fixed E (the level-set rule on the
    # effective potential, which the family's Hermite constraints use) agree
    # with finite differences of the turning points themselves
    from galpy.actionAngle import actionAngleSphericalInverse

    aAI = actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(), Es=[0.9, 1.5], Ls=[0.9, 0.75]
    )
    h = 1e-6
    for E, L in ((0.9, 0.9), (1.5, 0.75)):
        for q in range(2):
            r = aAI._turning_points(E, L)[q]
            dE, dL = aAI._turning_point_derivs(r, E, L)
            fdE = (
                aAI._turning_points(E + h, L)[q] - aAI._turning_points(E - h, L)[q]
            ) / (2.0 * h)
            fdL = (
                aAI._turning_points(E, L + h)[q] - aAI._turning_points(E, L - h)[q]
            ) / (2.0 * h)
            assert numpy.fabs(dE - fdE) < 1e-7 * (1.0 + numpy.fabs(fdE)), (
                "The turning point's E-derivative is not the level-set rule's"
            )
            assert numpy.fabs(dL - fdL) < 1e-7 * (1.0 + numpy.fabs(fdL)), (
                "The turning point's L-derivative is not the level-set rule's"
            )
    return None


def test_actionAngleSphericalInverse_symplectic(spherical_inverse_interp):
    # Manifest canonicity: the symplectic defect of the public map is at the
    # finite-difference floor -- measured on the analytic isochrone inverse
    # with the same harness -- between the nodes of the family, and just as
    # much on a grid so coarse that its interpolation error is large
    from galpy.actionAngle import (
        actionAngleIsochroneInverse,
        actionAngleSphericalInverse,
    )
    from galpy.potential import IsochronePotential

    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=1.0, b=0.5)),
        0.2,
        0.6,
        0.3,
        0.7,
        1.0,
        2.0,
    )
    aAI = spherical_inverse_interp
    L = 0.99
    jr = _spherical_inverse_forward_jr(_spherical_inverse_E_of_u(0.55, L), L)
    defect = _spherical_inverse_symplectic_defect(
        aAI, jr, 0.6 * L, 0.4 * L, 0.7, 1.0, 2.0
    )
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect between the nodes is above the finite-difference "
        "floor: %g vs %g" % (defect, floor)
    )
    coarse = actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.4,
        Rinf=6.0,
        nE=4,
        nL=4,
        mm_nta=128,
        mm_npt=24,
    )
    angler = numpy.linspace(0.05, 6.2, 21)
    dJ = _spherical_inverse_roundtrip(
        coarse, jr, 0.6 * L, 0.4 * L, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )[0]
    assert dJ > 1e-5, "The coarse grid is too accurate for this check to mean anything"
    defect = _spherical_inverse_symplectic_defect(
        coarse, jr, 0.6 * L, 0.4 * L, 0.7, 1.0, 2.0
    )
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect on the coarse grid is above the floor: %g vs %g, "
        "so canonicity is contingent on the tables" % (defect, floor)
    )
    return None


def test_actionAngleSphericalInverse_small_action(spherical_inverse_interp):
    # A radial action down to round-off above zero resolves to a small u,
    # not to the circular edge itself: the frequencies stay finite and tend
    # to the epicycle and circular frequencies, and the point tends to the
    # circular orbit's (L = 1 is a row of the fixture's grid, where the
    # edge's slopes are exact)
    from galpy.potential import epifreq, rl

    aAI = spherical_inverse_interp
    pot = _spherical_inverse_potential()
    L, jphi, jz = 1.0, 0.6, 0.4
    rc, kappa, Omc = rl(pot, L), epifreq(pot, rl(pot, L)), 1.0 / rl(pot, L)
    circ = numpy.array(aAI(0.0, jphi, jz, 0.3, 1.0, 2.0)).flatten()
    for jr in (1e-16, 1e-12, 1e-8):
        Om = numpy.array(aAI.Freqs(jr, jphi, jz))
        out = numpy.array(aAI(jr, jphi, jz, 0.3, 1.0, 2.0)).flatten()
        assert numpy.all(numpy.isfinite(Om)) and numpy.all(numpy.isfinite(out)), (
            "A tiny radial action J_r = %g gives non-finite output" % jr
        )
        assert (
            numpy.fabs(Om[0] / kappa - 1.0) < 1e-6
            and numpy.fabs(Om[2] / Omc - 1.0) < 1e-6
        ), (
            "The frequencies at J_r = %g are not the epicycle limit's: %s vs (%g, %g)"
            % (jr, Om, kappa, Omc)
        )
        # the libration's half-width in the epicycle limit, sqrt(2 J_r / kappa)
        amp = numpy.sqrt(2.0 * jr / kappa)
        assert (
            numpy.fabs(numpy.sqrt(out[0] ** 2 + out[3] ** 2) - rc) < 2.0 * amp + 1e-9
        ), (
            "The point at J_r = %g is not within the epicycle amplitude of the circular radius"
            % jr
        )
        assert numpy.amax(numpy.fabs(out - circ)) < 3.0 * amp + 1e-9, (
            "The point at J_r = %g is not continuous with the circular orbit's" % jr
        )
    return None


def test_actionAngleSphericalInverse_symplectic_perturbed(spherical_inverse_interp):
    # Manifest canonicity: the map is symplectic for whatever the tables
    # contain, because every derivative it uses is the stored interpolant's
    # own. Perturb the stored values and slopes of a copy of the family by
    # amounts that spoil its accuracy, rebuild the interpolants, and check
    # that the symplectic defect is still at the finite-difference floor
    import copy

    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=1.0, b=0.5)),
        0.2,
        0.6,
        0.3,
        0.7,
        1.0,
        2.0,
    )
    aAP = copy.deepcopy(spherical_inverse_interp)
    rng = numpy.random.default_rng(3)
    nu, nL = aAP._jr_tab.shape
    # the action and its slopes (the edge row stays at J_r = 0)
    aAP._jr_tab[1:] *= 1.0 + 1e-3 * rng.uniform(-1.0, 1.0, (nu - 1, nL))
    aAP._jr_dx *= 1.0 + 1e-2 * rng.uniform(-1.0, 1.0, (nu, nL))
    aAP._jr_dL += 1e-3 * rng.uniform(-1.0, 1.0, (nu, nL))
    # the turning points, pericentres in and apocentres out, and their slopes
    aAP._sup_tab[1:, :, 0] -= 1e-3 * rng.uniform(0.0, 1.0, (nu - 1, nL))
    aAP._sup_tab[1:, :, 1] += 1e-3 * rng.uniform(0.0, 1.0, (nu - 1, nL))
    aAP._sup_du *= 1.0 + 1e-2 * rng.uniform(-1.0, 1.0, aAP._sup_du.shape)
    aAP._sup_dL += 1e-3 * rng.uniform(-1.0, 1.0, aAP._sup_dL.shape)
    # the map's coefficients and their slopes
    aAP._Dm_tab += 1e-3 * rng.uniform(-1.0, 1.0, aAP._Dm_tab.shape)
    aAP._Dm_du *= 1.0 + 0.1 * rng.uniform(-1.0, 1.0, aAP._Dm_du.shape)
    aAP._Dm_dL += 1e-3 * rng.uniform(-1.0, 1.0, aAP._Dm_dL.shape)
    aAP._rebuild_interp()
    L = 0.99
    jr = _spherical_inverse_forward_jr(_spherical_inverse_E_of_u(0.55, L), L)
    angler = numpy.linspace(0.05, 6.2, 21)
    dJ = _spherical_inverse_roundtrip(
        aAP, jr, 0.6 * L, 0.4 * L, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )[0]
    assert dJ > 1e-5, "The perturbation of the tables is too small to mean anything"
    defect = _spherical_inverse_symplectic_defect(
        aAP, jr, 0.6 * L, 0.4 * L, 0.7, 1.0, 2.0
    )
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect with perturbed tables is above the floor: %g vs %g, "
        "so canonicity is contingent on the tables" % (defect, floor)
    )
    return None


def test_actionAngleSphericalInverse_extremes():
    # A grid spanning a factor ~7 in angular momentum, with tori up to
    # eccentricity ~0.9: the defect stays at the floor and the round trip is
    # the (coarse) grid's interpolation error
    from galpy.actionAngle import actionAngleSphericalInverse

    aAI = actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        setup_interp=True,
        Rmin=0.15,
        Rmax=1.0,
        Rinf=3.0,
        nE=12,
        nL=12,
    )
    L = 0.57
    E = _spherical_inverse_E_of_u(0.55, L, Rinf=3.0)
    jr = _spherical_inverse_forward_jr(E, L)
    defect = _spherical_inverse_symplectic_defect(aAI, jr, 0.3, L - 0.3, 2.0, 1.0, 2.0)
    assert defect < 1e-7, (
        "The symplectic defect on the wide, eccentric grid is not at the "
        "finite-difference floor: %g" % defect
    )
    angler = numpy.linspace(0.4, 2.7, 5)
    dJ = _spherical_inverse_roundtrip(
        aAI, jr, 0.3, L - 0.3, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )[0]
    assert dJ < 1e-2, (
        "The wide-grid reconstruction's actions do not round-trip within the "
        "grid's interpolation error: %g" % dJ
    )
    return None


def test_actionAngleSphericalInverse_convergence_warnings():
    # An under-resolved map warns and names the torus; a torus that reaches
    # beyond the depth of the fitted auxiliary raises
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.util import galpyWarning

    pot = _spherical_inverse_potential()
    with pytest.warns(
        galpyWarning, match="not converged for the \\(E, L\\) tori: \\(1.53, 0.9\\)"
    ):
        actionAngleSphericalInverse(pot=pot, Es=[1.53], Ls=[0.9], mm_npt=2, mm_nta=16)
    with pytest.raises(RuntimeError, match="not bound in the fitted auxiliary"):
        actionAngleSphericalInverse(pot=pot, Es=[5.0], Ls=[0.8])
    return None


def test_actionAngleSphericalInverse_escape_grid():
    # In a potential with an escape energy the apocentre of every torus
    # diverges there, so the family's tables have a pole just beyond the
    # top of a grid that reaches far out; the grid's energies are therefore
    # uniform in the logarithm of the radial orbit's apocentre rather than
    # in the energy, which keeps the tables smooth up to the top row. Checked
    # on a Hernquist grid reaching to two hundred times its innermost circular
    # radius: the round trip through the forward transformation at the
    # midpoint of the top cell, which the linear spacing gets wrong by
    # order unity, and the grid's coverage (the top row at the potential at
    # Rinf, the circular orbits at the bottom)
    from scipy.optimize import brentq

    from galpy.actionAngle import actionAngleSpherical, actionAngleSphericalInverse
    from galpy.potential import HernquistPotential, evaluatePotentials, rl, vcirc

    pot = HernquistPotential(normalize=1.0, a=0.5)
    aAI = actionAngleSphericalInverse(
        pot=pot,
        setup_interp=True,
        Rmin=0.1,
        Rmax=3.0,
        Rinf=20.0,
        nE=16,
        nL=8,
        mm_npt=256,
        progressbar=False,
    )
    aAS = actionAngleSpherical(pot=pot)
    # a column of the grid, so that the interpolation is along the energy (the
    # L derivatives between energy nodes still lean on the neighbouring
    # columns, through the estimated cross derivatives)
    L = numpy.linspace(0.1 * vcirc(pot, 0.1), 3.0 * vcirc(pot, 3.0), 8)[4]
    Emax = evaluatePotentials(pot, 20.0, 0.0)
    rc = rl(pot, L)
    Ec = evaluatePotentials(pot, rc, 0.0) + L**2 / (2.0 * rc**2)
    assert aAI.Jr(Ec, L) == 0.0, "The circular orbit is not the bottom of the grid"
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Jr(Emax + 1e-6, L)
    # the top cell's midpoint in the grid's own variable: the apocentre of
    # the radial orbit at the geometric mean of those at the top two rows
    r0 = lambda E: brentq(lambda r: evaluatePotentials(pot, r, 0.0) - E, rc, 20.0)
    r0c = r0(Ec)
    x = 0.5 * ((14.0 / 15.0) ** 2 + 1.0)
    E = evaluatePotentials(pot, r0c * (20.0 / r0c) ** x, 0.0)
    jr = aAI.Jr(E, L)
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi, anglez = 0.3 + 1.7 * angler, 0.7 + 2.3 * angler
    R, vR, vT, z, vz, phi = aAI(jr, 0.6 * L, 0.4 * L, angler, anglephi, anglez)
    f = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    dth = max(
        numpy.amax(
            numpy.fabs((f[6] - angler + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        ),
        numpy.amax(
            numpy.fabs((f[7] - anglephi + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        ),
        numpy.amax(
            numpy.fabs((f[8] - anglez + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        ),
    )
    assert numpy.amax(numpy.fabs(f[0] - jr)) / jr < 1e-3, (
        "The top cell of a grid reaching toward the escape energy does not "
        "interpolate the action"
    )
    assert dth < 3e-3, (
        "The top cell of a grid reaching toward the escape energy does not "
        "interpolate the angles"
    )
    return None


def test_actionAngleSphericalInverse_errors(
    spherical_inverse_interp, spherical_inverse_explicit
):
    # every guarded misuse raises informatively
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import IsochronePotential

    pot = _spherical_inverse_potential()
    with pytest.raises(ValueError, match="unbound"):
        actionAngleSphericalInverse(
            pot=IsochronePotential(normalize=1.0), Es=[0.5], Ls=[0.9]
        )
    with pytest.raises(OSError, match="Must specify pot="):
        actionAngleSphericalInverse()
    with pytest.raises(ValueError, match="same length"):
        actionAngleSphericalInverse(pot=pot, Es=[0.7, 1.1], Ls=[0.9])
    with pytest.raises(ValueError, match="even"):
        actionAngleSphericalInverse(pot=pot, Es=[0.7], Ls=[0.9], mm_nta=127)
    with pytest.raises(ValueError, match="mm_nta must exceed"):
        actionAngleSphericalInverse(pot=pot, Es=[0.7], Ls=[0.9], mm_nta=64, mm_npt=99)
    with pytest.raises(ValueError, match="mm_nta must exceed"):
        # clears the samples' own Nyquist harmonic but not the map's highest
        actionAngleSphericalInverse(pot=pot, Es=[0.7], Ls=[0.9], mm_nta=64, mm_npt=20)
    with pytest.raises(ValueError, match="nE >= 4"):
        actionAngleSphericalInverse(pot=pot, setup_interp=True, nE=3)
    with pytest.raises(ValueError, match="below"):
        # E below the circular orbit's energy at this L
        actionAngleSphericalInverse(pot=pot, Es=[0.0], Ls=[1.0])
    with pytest.raises(ValueError, match="Rinf"):
        # Rinf below the grid's radial range
        actionAngleSphericalInverse(
            pot=pot, setup_interp=True, Rmin=0.7, Rmax=1.4, Rinf=1.0
        )
    aAI = spherical_inverse_interp
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Freqs(0.15, 5.0, 0.0)  # L outside the grid
    with pytest.raises(ValueError, match="outside the interpolated family"):
        aAI(50.0, 0.65, 0.35, 0.3, 1.0, 2.0)  # J_r outside the family
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Jr(1.0, 5.0)
    with pytest.raises(ValueError, match="outside the interpolation grid at L"):
        aAI.Jr(0.1, 1.0)  # below the circular orbit's energy
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Freqs(0.0, 0.3, 0.2)  # a circular request outside the grid's L range
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI(0.0, 0.3, 0.2, 0.3, 1.0, 2.0)
    aAD = spherical_inverse_explicit
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD.Freqs(0.123, 0.4, 0.2)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD(0.123, 0.4, 0.2, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD.Jr(0.8, 0.9)
    return None


def test_actionAngleSphericalInverse_circular(spherical_inverse_interp):
    # The circular orbit is the edge of the family and a torus in its own
    # right: the family's grid has the circular orbits as its bottom row,
    # with the epicycle limit's exact partials (J_r, the turning points) and
    # the map's slope extrapolated from the next rows, so J_r -> 0 is
    # reached continuously; the circular orbit itself (J_r = 0) evaluates
    # in closed form, with the epicycle and circular frequencies, in both
    # kinds of family. The forward transformation cannot take an exactly
    # circular orbit, so that one is checked through its invariants and its
    # continuity with tiny J_r
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import epifreq, evaluatePotentials, omegac, rl

    pot = _spherical_inverse_potential()
    aAI = spherical_inverse_interp
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)

    def invariants(aAI, L, jphi):
        rc = rl(pot, L, use_physical=False)
        R, vR, vT, z, vz, phi = aAI(
            0.0, jphi, L - numpy.fabs(jphi), angler, anglephi, anglez
        )
        r = numpy.sqrt(R**2 + z**2)
        Ltot = numpy.sqrt((R * vT) ** 2 + (z * vT) ** 2 + (R * vz - z * vR) ** 2)
        E = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
        Ec = evaluatePotentials(pot, rc, 0.0) + L**2 / (2.0 * rc**2)
        return max(
            numpy.amax(numpy.fabs(r / rc - 1.0)),
            numpy.amax(numpy.fabs((R * vR + z * vz) / rc)),
            numpy.amax(numpy.fabs(R * vT - jphi)),
            numpy.amax(numpy.fabs(Ltot - L)),
            numpy.amax(numpy.fabs(E - Ec)),
        )

    for L, jphi in ((0.99, 0.7 * 0.99), (0.99, -0.4 * 0.99), (1.3, 1.3)):
        rc = rl(pot, L, use_physical=False)
        Ec = evaluatePotentials(pot, rc, 0.0) + L**2 / (2.0 * rc**2)
        assert aAI.Jr(Ec, L) == 0.0, "The circular orbit's radial action is not zero"
        Om = aAI.Freqs(0.0, jphi, L - numpy.fabs(jphi))
        assert numpy.fabs(Om[0] / epifreq(pot, rc, use_physical=False) - 1.0) < 1e-12, (
            "The circular orbit's radial frequency is not the epicycle frequency"
        )
        assert numpy.fabs(Om[2] / omegac(pot, rc, use_physical=False) - 1.0) < 1e-12, (
            "The circular orbit's vertical frequency is not the circular frequency"
        )
        assert invariants(aAI, L, jphi) < 1e-13, (
            "The circular orbit is not at the circular radius with no radial "
            "motion and the requested angular momenta and energy"
        )
    # continuity: the map at tiny J_r approaches the circular orbit as its
    # amplitude, sqrt(2 J_r / kappa)
    L, jphi = 0.99, 0.7 * 0.99
    rc = rl(pot, L, use_physical=False)
    circ = numpy.array(aAI(0.0, jphi, L - jphi, angler, anglephi, anglez))
    for jr in (1e-6, 1e-8):
        near = numpy.array(aAI(jr, jphi, L - jphi, angler, anglephi, anglez))
        amp = numpy.sqrt(2.0 * jr / epifreq(pot, rc, use_physical=False)) / rc
        assert numpy.amax(numpy.fabs(near - circ)) < 5.0 * amp, (
            "The map does not approach the circular orbit as its amplitude"
        )
    # the first cell above the edge round-trips through the forward
    # transformation at the family's accuracy
    for u in (0.02, 0.08):
        E = _spherical_inverse_E_of_u(u, L)
        jr = _spherical_inverse_forward_jr(E, L)
        dJ, dth, dOm, dH = _spherical_inverse_roundtrip(
            aAI, jr, jphi, L - jphi, angler, anglephi, anglez
        )
        assert dJ < 1e-5 and dth < 1e-3 and dOm < 1e-5, (
            "The family is inaccurate just above the circular edge: %g %g %g"
            % (dJ, dth, dOm)
        )
    # discrete families: a circular torus among librating ones, and alone
    L = 0.9
    rc = rl(pot, L, use_physical=False)
    Ec = evaluatePotentials(pot, rc, 0.0) + L**2 / (2.0 * rc**2)
    for Es in ([Ec, 0.9], [Ec]):
        aAD = actionAngleSphericalInverse(pot=pot, Es=Es, Ls=[L] * len(Es))
        assert aAD.Jr(Ec, L) == 0.0
        assert invariants(aAD, L, 0.63) < 1e-13, (
            "A discrete family's circular torus is not the circular orbit"
        )
        assert (
            numpy.fabs(
                aAD.Freqs(0.0, 0.63, L - 0.63)[0] / epifreq(pot, rc, use_physical=False)
                - 1.0
            )
            < 1e-12
        )
    return None


def test_actionAngleSphericalInverse_auxiliary():
    # the fitted auxiliary is exposed; a given one is used instead of the
    # fit and gives the same tori to the forward transformation's floor
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import IsochronePotential

    pot = _spherical_inverse_potential()
    Es, Ls = [0.7, 1.1], [0.9, 0.7]
    aAF = actionAngleSphericalInverse(pot=pot, Es=Es, Ls=Ls)
    assert isinstance(aAF.auxiliary, IsochronePotential), (
        "The fitted auxiliary is not exposed as an IsochronePotential"
    )
    ip = IsochronePotential(amp=4.0, b=0.5)
    aAG = actionAngleSphericalInverse(pot=pot, Es=Es, Ls=Ls, auxiliary=ip)
    assert aAG.auxiliary is ip, "The given auxiliary is not the one used"
    assert numpy.fabs(aAG.auxiliary._amp - aAF.auxiliary._amp) > 1e-3, (
        "The given auxiliary coincides with the fitted one, so the test is void"
    )
    angler = numpy.linspace(0.0, 2.0 * numpy.pi, 17, endpoint=False)
    for E, L in zip(Es, Ls):
        for aAI in (aAF, aAG):
            dj, dth, dOm, dE = _spherical_inverse_roundtrip(
                aAI, aAI.Jr(E, L), 0.6 * L, 0.4 * L, angler, 0.3, 1.1
            )
            assert dj < 1e-9 and dth < 1e-7 and dOm < 1e-8 and dE < 1e-10, (
                "A torus does not round-trip through the forward transformation with the %s auxiliary: %g %g %g %g"
                % ("given" if aAI is aAG else "fitted", dj, dth, dOm, dE)
            )
    with pytest.raises(TypeError, match="IsochronePotential"):
        actionAngleSphericalInverse(pot=pot, Es=Es, Ls=Ls, auxiliary=pot)
    return None


def test_actionAngleSphericalInverse_maxiter(
    spherical_inverse_explicit, spherical_inverse_interp
):
    # maxiter governs the angle solve: with none, the solve falls back on
    # safeguarded root-finding and still returns the torus, for explicit
    # tori and for the interpolated family alike (the fixtures' tori, built
    # again with maxiter=0, against the fixtures)
    from galpy.actionAngle import actionAngleSphericalInverse

    pot = _spherical_inverse_potential()
    angler = numpy.linspace(0.05, 6.2, 11)
    aAD = actionAngleSphericalInverse(pot=pot, Es=[0.7, 1.1], Ls=[0.9, 0.7], maxiter=0)
    jrD = aAD.Jr(0.7, 0.9)
    fbD = numpy.array(aAD(jrD, 0.6, 0.3, angler, 1.0, 2.0))
    ntD = numpy.array(spherical_inverse_explicit(jrD, 0.6, 0.3, angler, 1.0, 2.0))
    assert numpy.amax(numpy.fabs(fbD - ntD)) < 1e-9, (
        "The safeguarded fallback of the angle solve does not agree with Newton for an explicit torus: %g"
        % numpy.amax(numpy.fabs(fbD - ntD))
    )
    aAI = actionAngleSphericalInverse(
        pot=pot,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.4,
        Rinf=6.0,
        nE=8,
        nL=8,
        mm_nta=128,
        mm_npt=24,
        maxiter=0,
    )
    L = 0.99
    jr = _spherical_inverse_forward_jr(_spherical_inverse_E_of_u(0.55, L), L)
    fb = numpy.array(
        aAI(jr, 0.6 * L, 0.4 * L, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0)
    )
    nt = numpy.array(
        spherical_inverse_interp(
            jr, 0.6 * L, 0.4 * L, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
        )
    )
    assert numpy.amax(numpy.fabs(fb - nt)) < 1e-9, (
        "The safeguarded fallback of the angle solve does not agree with Newton: %g"
        % numpy.amax(numpy.fabs(fb - nt))
    )
    return None


# ---------- the momentum-matched Staeckel inverse
def _staeckel_inverse_potential():
    from galpy.potential import KuzminKutuzovStaeckelPotential

    return KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)


_STAECKEL_INVERSE_DELTA = 1.3
_STAECKEL_INVERSE_ICS = {
    "benign": [1.1, 0.3, 0.9, 0.25, 0.2, 0.0],
    "eccentric": [1.1, 0.9, 0.35, 0.15, 0.1, 0.0],
    "near-shell": [1.1, 0.001, 0.8425895627614183, 0.15, 0.25, 0.0],
    "near-planar": [1.1, 0.4, 0.9, 0.002, 0.002, 0.0],
}


def _staeckel_inverse_labels(ic, pot=None, delta=_STAECKEL_INVERSE_DELTA):
    # (E, L_z, I_3) of a phase-space point, from the separated
    # Hamilton-Jacobi equation in the prolate spheroidal coordinates of the
    # potential's focal length, independently of the inverse's internals
    from galpy.orbit import Orbit
    from galpy.potential import evaluatePotentials

    pot = _staeckel_inverse_potential() if pot is None else pot
    o = Orbit(ic)
    E = float(o.E(pot=pot))
    Lz = float(o.R() * o.vT())
    R, z, vR, vz = float(o.R()), float(o.z()), float(o.vR()), float(o.vz())
    d1 = numpy.sqrt(R**2 + (z + delta) ** 2)
    d2 = numpy.sqrt(R**2 + (z - delta) ** 2)
    u = numpy.arccosh((d1 + d2) / 2.0 / delta)
    v = numpy.arccos((d1 - d2) / 2.0 / delta)
    pu = delta * (vR * numpy.cosh(u) * numpy.sin(v) + vz * numpy.sinh(u) * numpy.cos(v))
    Uu = evaluatePotentials(pot, delta * numpy.sinh(u), 0.0) * (
        numpy.sinh(u) ** 2 + 1.0
    )
    I3 = (
        E * numpy.sinh(u) ** 2
        - Uu
        - (pu**2 + Lz**2 / numpy.sinh(u) ** 2) / (2.0 * delta**2)
    )
    return E, Lz, float(I3)


def _staeckel_inverse_forward():
    from galpy.actionAngle import actionAngleStaeckel

    return actionAngleStaeckel(
        pot=_staeckel_inverse_potential(),
        delta=_STAECKEL_INVERSE_DELTA,
        c=True,
        order=200,
    )


@pytest.fixture(scope="module")
def staeckel_inverse_explicit():
    # two explicit tori, a benign and an eccentric one
    from galpy.actionAngle import actionAngleStaeckelInverse

    labels = [
        _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[k])
        for k in ("benign", "eccentric")
    ]
    return actionAngleStaeckelInverse(
        pot=_staeckel_inverse_potential(),
        Es=[l[0] for l in labels],
        Lzs=[l[1] for l in labels],
        I3s=[l[2] for l in labels],
    )


def _staeckel_inverse_roundtrip(
    aAI, jr, jphi, jz, angler, anglephi, anglez, aAS=None, pot=None
):
    # evaluate the public map and pass the result through the forward
    # transformation: (action, angle, frequency) errors and the energy spread
    from galpy.potential import evaluatePotentials

    pot = _staeckel_inverse_potential() if pot is None else pot
    aAS = _staeckel_inverse_forward() if aAS is None else aAS
    R, vR, vT, z, vz, phi = aAI(jr, jphi, jz, angler, anglephi, anglez)
    f = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    Om = aAI.Freqs(jr, jphi, jz)
    wrap = lambda d: numpy.amax(
        numpy.fabs((d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
    )
    dJ = numpy.amax(
        numpy.fabs(f[0] - jr) + numpy.fabs(f[1] - jphi) + numpy.fabs(f[2] - jz)
    ) / (jr + jz + numpy.fabs(jphi))
    dth = max(wrap(f[6] - angler), wrap(f[7] - anglephi), wrap(f[8] - anglez))
    dOm = max(
        numpy.amax(numpy.fabs(f[3] / Om[0] - 1.0)),
        numpy.amax(numpy.fabs(f[4] / Om[1] - 1.0)),
        numpy.amax(numpy.fabs(f[5] / Om[2] - 1.0)),
    )
    H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
    return dJ, dth, dOm, numpy.ptp(H) / numpy.fabs(numpy.mean(H))


def test_actionAngleStaeckelInverse_nodes(staeckel_inverse_explicit):
    # A discrete family returns its own tori: the action labels are the
    # forward transformation's, and the map round-trips through it at the
    # forward code's floor, for all three angles, for the benign and the
    # eccentric torus
    aAI = staeckel_inverse_explicit
    aAS = _staeckel_inverse_forward()
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    for key in ("benign", "eccentric"):
        ic = _STAECKEL_INVERSE_ICS[key]
        E, Lz, I3 = _staeckel_inverse_labels(ic)
        jrf, _, jzf = (float(numpy.atleast_1d(x)[0]) for x in aAS(*ic))
        jr, jz = aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)
        assert numpy.fabs(jr - jrf) < 1e-9 and numpy.fabs(jz - jzf) < 1e-9, (
            "The family's action labels are not the forward transformation's"
        )
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aAI, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS
        )
        assert dJ < 1e-9, "The map does not return the requested torus: %g" % dJ
        assert dth < 1e-7, "The map does not return the requested angles: %g" % dth
        assert dOm < 1e-8, "The frequencies are not the torus's: %g" % dOm
        assert dH < 1e-10, "The reconstructed torus is not at one energy: %g" % dH
    return None


def test_actionAngleStaeckelInverse_angleconventions(staeckel_inverse_explicit):
    # The forward actionAngleStaeckel's angles of a point, fed to the
    # inverse, return that point: the angle conventions match (theta_R = 0
    # at the inner u turning point, theta_z = 0 at the upward midplane
    # crossing at pericentre)
    aAI = staeckel_inverse_explicit
    aAS = _staeckel_inverse_forward()
    for key in ("benign", "eccentric"):
        ic = _STAECKEL_INVERSE_ICS[key]
        E, Lz, I3 = _staeckel_inverse_labels(ic)
        out = aAS.actionsFreqsAngles(*ic)
        ar, ap, az = (numpy.asarray(out[i]).ravel() for i in (6, 7, 8))
        rec = numpy.array(
            aAI(aAI.JR(E, Lz, I3), Lz, aAI.Jz(E, Lz, I3), ar, ap, az)
        ).flatten()
        diff = rec - numpy.array(ic)
        diff[5] = (diff[5] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(diff)) < 1e-9, (
            "Feeding the forward actionAngleStaeckel angles to "
            "actionAngleStaeckelInverse does not return the original point: %g"
            % numpy.amax(numpy.fabs(diff))
        )
    return None


def test_actionAngleStaeckelInverse_orbit(staeckel_inverse_explicit):
    # Traversing a torus at its frequencies is an orbit of the potential: at
    # a node the energy is constant along the torus to the maps' truncation,
    # and the points agree with an integrated orbit started from the first
    # of them over ten radial periods
    from galpy.orbit import Orbit
    from galpy.potential import evaluatePotentials

    aAI = staeckel_inverse_explicit
    pot = _staeckel_inverse_potential()
    wrap = lambda d: (d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    for key in ("benign", "eccentric"):
        E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[key])
        jr, jz = aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)
        Om = aAI.Freqs(jr, Lz, jz)
        ts = numpy.linspace(0.0, 10.0 * 2.0 * numpy.pi / Om[0], 1001)
        R, vR, vT, z, vz, phi = aAI(
            jr, Lz, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-11, (
            "Energy is not conserved along an actionAngleStaeckelInverse torus at a node: %g"
            % (numpy.std(H) / numpy.fabs(numpy.mean(H)))
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, orb.R(ts)),
            ("z", z, orb.z(ts)),
            ("vR", vR, orb.vR(ts)),
            ("vT", vT, orb.vT(ts)),
            ("vz", vz, orb.vz(ts)),
            ("phi", phi, phi + wrap(orb.phi(ts) - phi)),
        ):
            assert numpy.amax(numpy.fabs(orbit - torus)) < 1e-7, (
                "%s along an actionAngleStaeckelInverse torus at a node does not agree with the integrated orbit: %g"
                % (name, numpy.amax(numpy.fabs(orbit - torus)))
            )
    return None


def test_actionAngleStaeckelInverse_turning_point_derivatives(
    staeckel_inverse_explicit,
):
    # The closed-form derivatives of the turning points (the level-set rule
    # on the separated momenta, which the family's Hermite constraints use)
    # and of the shell u agree with finite differences of the turning points
    # themselves across neighbouring tori
    aAI = staeckel_inverse_explicit
    h = 1e-6
    for key in ("benign", "eccentric"):
        E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[key])
        uc, wu, wv, ush, _, _ = aAI._turning_points(E, Lz, I3)
        dinner = aAI._tp_derivs_u(uc - wu, E, Lz)
        douter = aAI._tp_derivs_u(uc + wu, E, Lz)
        dvm = aAI._tp_derivs_v(0.5 * numpy.pi - wv, E, Lz)
        dush = aAI._dushell(ush, E, Lz)
        for k, (dE, dI, dL) in enumerate(((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h))):
            up = aAI._turning_points(E + dE, Lz + dL, I3 + dI)
            dn = aAI._turning_points(E - dE, Lz - dL, I3 - dI)
            fd = [
                ((up[0] - up[1]) - (dn[0] - dn[1])) / (2.0 * h),
                ((up[0] + up[1]) - (dn[0] + dn[1])) / (2.0 * h),
                ((0.5 * numpy.pi - up[2]) - (0.5 * numpy.pi - dn[2])) / (2.0 * h),
            ]
            for name, ana, num in (
                ("inner u", dinner[k], fd[0]),
                ("outer u", douter[k], fd[1]),
                ("v", dvm[k], fd[2]),
            ):
                assert numpy.fabs(ana - num) < 1e-6 * (1.0 + numpy.fabs(num)), (
                    "The %s turning point's derivative is not the level-set rule's"
                    % name
                )
            if k != 1:
                assert numpy.fabs(dush[k // 2] - (up[3] - dn[3]) / (2.0 * h)) < 1e-6 * (
                    1.0 + numpy.fabs(dush[k // 2])
                ), "The shell u's derivative is not the level-set rule's"
    return None


def test_actionAngleStaeckelInverse_extremes():
    # Explicit tori at the edges of the (E, L_z, I_3) space: near-shell
    # (J_R -> 0) and near-planar (J_z -> 0) tori are the forward
    # transformation's tori and are orbits of the potential (the forward
    # code's angles and frequencies are themselves inaccurate that close to
    # a degenerate libration, so those are checked against an integrated
    # orbit), and a near-polar torus (L_z -> 0) works with the many
    # harmonics its v map needs, of the order of L / |L_z|
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.orbit import Orbit
    from galpy.potential import evaluatePotentials
    from galpy.util import galpyWarning

    pot = _staeckel_inverse_potential()
    aAS = _staeckel_inverse_forward()
    angler = numpy.linspace(0.05, 6.2, 21)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    wrap = lambda d: (d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    for key in ("near-shell", "near-planar"):
        E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[key])
        aAI = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
        jr, jz = aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)
        dJ = _staeckel_inverse_roundtrip(
            aAI, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS
        )[0]
        assert dJ < 1e-9, (
            "The %s torus's actions do not round-trip through the forward transformation: %g"
            % (key, dJ)
        )
        Om = aAI.Freqs(jr, Lz, jz)
        ts = numpy.linspace(0.0, 5.0 * 2.0 * numpy.pi / Om[0], 501)
        R, vR, vT, z, vz, phi = aAI(
            jr, Lz, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-11, (
            "Energy is not conserved along the %s torus" % key
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, orb.R(ts)),
            ("z", z, orb.z(ts)),
            ("vR", vR, orb.vR(ts)),
            ("vz", vz, orb.vz(ts)),
            ("phi", phi, phi + wrap(orb.phi(ts) - phi)),
        ):
            assert numpy.amax(numpy.fabs(orbit - torus)) < 1e-7, (
                "%s along the %s torus does not agree with the integrated orbit: %g"
                % (name, key, numpy.amax(numpy.fabs(orbit - torus)))
            )
    # near-polar: a torus with L / |L_z| ~ 8
    E, Lz, I3 = _staeckel_inverse_labels([1.0, 0.2, 0.03, 0.3, 0.6, 0.0])
    with pytest.warns(galpyWarning, match="v anomaly map is not converged"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=16)
    aAI = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=128)
    assert aAI.Jz(E, Lz, I3) / numpy.fabs(Lz) > 5.0, (
        "The near-polar torus is not near-polar"
    )
    dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
        aAI, aAI.JR(E, Lz, I3), Lz, aAI.Jz(E, Lz, I3), angler, anglephi, anglez, aAS=aAS
    )
    assert dJ < 1e-8 and dth < 1e-6 and dOm < 1e-7 and dH < 1e-10, (
        "The near-polar torus does not round-trip through the forward transformation: %g %g %g %g"
        % (dJ, dth, dOm, dH)
    )
    return None


def test_actionAngleStaeckelInverse_convergence_warnings():
    # An under-resolved map warns and names the torus; a torus that reaches
    # beyond the depth of the fitted auxiliary raises
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.util import galpyWarning

    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["eccentric"])
    with pytest.warns(
        galpyWarning,
        match="u anomaly map is not converged for the \\(E, L_z, I_3\\) tori: \\(%.6g, %.6g, %.6g\\)"
        % (E, Lz, I3),
    ):
        actionAngleStaeckelInverse(
            pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=2, mm_nta=16
        )
    # a deep radial orbit (its u libration spans a factor of two hundred in
    # radius), whose lift with two harmonics is unbound by a wide margin on
    # every platform (the lifted energy is -0.46 of the auxiliary torus's,
    # against the 0.05 at which the guard trips; four harmonics keep it bound,
    # at 0.19, and only warn)
    E, Lz, I3 = _staeckel_inverse_labels([1.0, 1.9, 0.15, 0.05, 0.05, 0.0])
    with pytest.raises(RuntimeError, match="not bound in the fitted auxiliary"):
        actionAngleStaeckelInverse(
            pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=2, mm_nta=16
        )
    return None


def test_actionAngleStaeckelInverse_errors(staeckel_inverse_explicit):
    # every guarded misuse raises informatively
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import (
        IsochronePotential,
        MWPotential2014,
        OblateStaeckelWrapperPotential,
    )

    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    with pytest.raises(OSError, match="Must specify pot="):
        actionAngleStaeckelInverse()
    with pytest.raises(OSError, match="supplies its focal length"):
        actionAngleStaeckelInverse(pot=MWPotential2014, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(TypeError, match="conflict"):
        actionAngleStaeckelInverse(pot=pot, delta=1.3, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(TypeError, match="conflict"):
        actionAngleStaeckelInverse(
            pot=OblateStaeckelWrapperPotential(pot=pot, delta=1.3),
            delta=1.3,
            Es=[E],
            Lzs=[Lz],
            I3s=[I3],
        )
    with pytest.raises(TypeError, match="u0= requires delta="):
        actionAngleStaeckelInverse(pot=pot, u0=1.0, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(TypeError, match="IsochronePotential"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], auxiliary=pot)
    with pytest.raises(ValueError, match="same length"):
        actionAngleStaeckelInverse(pot=pot, Es=[E, E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="L_z = 0"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[0.0], I3s=[I3])
    with pytest.raises(ValueError, match="even"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_nta=127)
    with pytest.raises(ValueError, match="mm_nta must exceed"):
        actionAngleStaeckelInverse(
            pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_nta=64, mm_npt=20
        )
    with pytest.raises(ValueError, match="mm_npt must be at least 2"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=1)
    with pytest.raises(ValueError, match="below the circular orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[-10.0], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="unbound"):
        actionAngleStaeckelInverse(pot=pot, Es=[0.1], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="above the shell orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3 + 10.0])
    with pytest.raises(ValueError, match="below the planar orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3 - 10.0])
    aAD = staeckel_inverse_explicit
    with pytest.raises(ValueError, match="non-negative"):
        aAD(-0.1, 0.4, 0.2, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="L_z = 0"):
        aAD(0.1, 0.0, 0.1, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD.Freqs(0.123, 0.4, 0.2)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD(0.123, 0.4, 0.2, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD.JR(E + 0.1, Lz, I3)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        aAD(0.0, 0.4, 0.0, 0.3, 1.0, 2.0)  # a circular request that is not a node
    return None


def test_actionAngleStaeckelInverse_circular():
    # The circular orbit is a torus in its own right: it evaluates in
    # closed form at the circular radius in the plane, at the requested
    # azimuthal angle, with the epicycle, circular, and vertical
    # frequencies, among librating tori or alone
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import epifreq, evaluatePotentials, rl

    pot = _staeckel_inverse_potential()
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)

    def invariants(aAI, Lz):
        Rc = rl(pot, numpy.fabs(Lz), use_physical=False)
        R, vR, vT, z, vz, phi = aAI(0.0, Lz, 0.0, angler, anglephi, anglez)
        return max(
            numpy.amax(numpy.fabs(R / Rc - 1.0)),
            numpy.amax(numpy.fabs(vR)),
            numpy.amax(numpy.fabs(vz)),
            numpy.amax(numpy.fabs(z)),
            numpy.amax(numpy.fabs(R * vT - Lz)),
            numpy.amax(
                numpy.fabs((phi - anglephi + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
            ),
        )

    Lz = 0.93
    Rc = rl(pot, Lz, use_physical=False)
    # discrete families: a circular torus among librating ones, and alone
    Ec = evaluatePotentials(pot, Rc, 0.0) + Lz**2 / (2.0 * Rc**2)
    Ipl = Lz**2 / (2.0 * _STAECKEL_INVERSE_DELTA**2) - Ec
    E1, Lz1, I31 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    for Es, Lzs, I3s in (([Ec, E1], [Lz, Lz1], [Ipl, I31]), ([Ec], [Lz], [Ipl])):
        aAD = actionAngleStaeckelInverse(pot=pot, Es=Es, Lzs=Lzs, I3s=I3s)
        assert aAD.JR(Ec, Lz, Ipl) == 0.0 and aAD.Jz(Ec, Lz, Ipl) == 0.0
        assert invariants(aAD, Lz) < 1e-11, (
            "A discrete family's circular torus is not the circular orbit"
        )
        assert (
            numpy.fabs(
                aAD.Freqs(0.0, Lz, 0.0)[0] / epifreq(pot, Rc, use_physical=False) - 1.0
            )
            < 1e-10
        )
    return None


def test_actionAngleStaeckelInverse_auxiliary(staeckel_inverse_explicit):
    # the fitted auxiliary is exposed; a given one is used instead of the
    # fit and gives the same tori to the forward transformation's floor
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import IsochronePotential

    pot = _staeckel_inverse_potential()
    aAF = staeckel_inverse_explicit
    assert isinstance(aAF.auxiliary, IsochronePotential), (
        "The fitted auxiliary is not exposed as an IsochronePotential"
    )
    ip = IsochronePotential(amp=4.0, b=0.5)
    aAG = actionAngleStaeckelInverse(
        pot=pot, Es=aAF._Es, Lzs=aAF._Lzs, I3s=aAF._I3s, auxiliary=ip
    )
    assert aAG.auxiliary is ip, "The given auxiliary is not the one used"
    assert numpy.fabs(aAG.auxiliary._amp - aAF.auxiliary._amp) > 1e-3, (
        "The given auxiliary coincides with the fitted one, so the test is void"
    )
    aAS = _staeckel_inverse_forward()
    angler = numpy.linspace(0.0, 2.0 * numpy.pi, 17, endpoint=False)
    for E, Lz, I3 in zip(aAF._Es, aAF._Lzs, aAF._I3s):
        for aAI in (aAF, aAG):
            dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
                aAI, aAI.JR(E, Lz, I3), Lz, aAI.Jz(E, Lz, I3), angler, 0.3, 1.1, aAS=aAS
            )
            assert dJ < 1e-9 and dth < 1e-7 and dOm < 1e-8 and dH < 1e-10, (
                "A torus does not round-trip through the forward transformation with the %s auxiliary: %g %g %g %g"
                % ("given" if aAI is aAG else "fitted", dJ, dth, dOm, dH)
            )
    return None


def test_actionAngleStaeckelInverse_potentials():
    # The potential's Staeckel form is taken from the potential itself, from
    # an OblateStaeckelWrapperPotential, or from delta= for a general
    # axisymmetric potential (which is then wrapped): the first two give the
    # same tori for a Kuzmin-Kutuzov potential, the last two the same tori
    # for a wrapped general potential; a retrograde torus is the prograde
    # one mirrored
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    angler = numpy.linspace(0.05, 6.2, 11)
    aAK = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
    aAW = actionAngleStaeckelInverse(
        pot=OblateStaeckelWrapperPotential(pot=pot, delta=1.3),
        Es=[E],
        Lzs=[Lz],
        I3s=[I3],
    )
    jr, jz = aAK.JR(E, Lz, I3), aAK.Jz(E, Lz, I3)
    assert (
        numpy.fabs(aAW.JR(E, Lz, I3) - jr) < 1e-12
        and numpy.fabs(aAW.Jz(E, Lz, I3) - jz) < 1e-12
    ), (
        "A wrapped Staeckel potential does not give the same tori as the potential itself"
    )
    xK = numpy.array(aAK(jr, Lz, jz, angler, 1.0, 2.0))
    xW = numpy.array(aAW(jr, Lz, jz, angler, 1.0, 2.0))
    assert numpy.amax(numpy.fabs(xK - xW)) < 1e-10, (
        "A wrapped Staeckel potential does not evaluate to the same points as the potential itself"
    )
    # retrograde: the mirror image, with the sign of the azimuthal frequency
    aAR = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[-Lz], I3s=[I3])
    assert (
        numpy.fabs(aAR.JR(E, -Lz, I3) - jr) < 1e-12
        and numpy.fabs(aAR.Jz(E, -Lz, I3) - jz) < 1e-12
    )
    OmP, OmR = numpy.array(aAK.Freqs(jr, Lz, jz)), numpy.array(aAR.Freqs(jr, -Lz, jz))
    assert numpy.amax(numpy.fabs(OmR - OmP * [1.0, -1.0, 1.0])) < 1e-12, (
        "The retrograde torus's frequencies are not the prograde one's mirrored"
    )
    dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(aAR, jr, -Lz, jz, angler, 1.0, 2.0)
    assert dJ < 1e-9 and dth < 1e-7 and dOm < 1e-8 and dH < 1e-10, (
        "The retrograde torus does not round-trip through the forward transformation"
    )
    # a general potential in the Staeckel approximation: delta= wraps it
    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45)
    E, Lz, I3 = _staeckel_inverse_labels(
        [1.0, 0.1, 1.05, 0.1, 0.1, 0.0], pot=swp, delta=0.45
    )
    aAG = actionAngleStaeckelInverse(
        pot=MWPotential2014, delta=0.45, Es=[E], Lzs=[Lz], I3s=[I3]
    )
    aAWG = actionAngleStaeckelInverse(pot=swp, Es=[E], Lzs=[Lz], I3s=[I3])
    jr, jz = aAG.JR(E, Lz, I3), aAG.Jz(E, Lz, I3)
    assert (
        numpy.fabs(aAWG.JR(E, Lz, I3) - jr) < 1e-12
        and numpy.fabs(aAWG.Jz(E, Lz, I3) - jz) < 1e-12
    ), "delta= does not give the same tori as the explicitly wrapped potential"
    # the torus lives in the wrapped model: the forward transformation of
    # that model recovers it
    R, vR, vT, z, vz, phi = aAG(jr, Lz, jz, angler, 1.0, 2.0)
    f = actionAngleStaeckel(pot=swp, delta=0.45, c=False, order=200).actionsFreqsAngles(
        R, vR, vT, z, vz, phi
    )
    assert (
        numpy.amax(numpy.fabs(f[0] - jr)) < 1e-8
        and numpy.amax(numpy.fabs(f[2] - jz)) < 1e-8
    ), (
        "The torus in the Staeckel approximation of a general potential is not that model's"
    )
    return None


def test_actionAngleStaeckelInverse_maxiter():
    # maxiter governs the angle solve: with none, the solve falls back on
    # safeguarded root-finding and still returns the torus
    from galpy.actionAngle import actionAngleStaeckelInverse

    pot = _staeckel_inverse_potential()
    angler = numpy.linspace(0.05, 6.2, 11)
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    aAN = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
    aAF = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], maxiter=0)
    jr, jz = aAN.JR(E, Lz, I3), aAN.Jz(E, Lz, I3)
    nt = numpy.array(aAN(jr, Lz, jz, angler, 1.0, 2.0))
    fb = numpy.array(aAF(jr, Lz, jz, angler, 1.0, 2.0))
    assert numpy.amax(numpy.fabs(fb - nt)) < 1e-9, (
        "The safeguarded fallback of the angle solve does not agree with Newton for an explicit torus: %g"
        % numpy.amax(numpy.fabs(fb - nt))
    )
    return None


def test_actionAngleStaeckelInverse_perfect_ellipsoid():
    # The oblate perfect ellipsoid is a Staeckel potential too, with focal
    # length a sqrt(1 - c^2) for semi-axes a and c a, which it supplies
    # itself, and galpy evaluates it by Gaussian quadrature of the
    # ellipsoidal integrals: explicit tori in it round-trip through the
    # forward transformation, judged in that potential, and keep their
    # energy along the reconstructed torus
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import PerfectEllipsoidPotential

    pot = PerfectEllipsoidPotential(amp=1.0, a=1.0, b=1.0, c=0.6, normalize=1.0)
    delta = 0.8
    aAS = actionAngleStaeckel(pot=pot, delta=delta, c=False, order=200)
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi, anglez = 0.3 + 1.7 * angler, 0.7 + 2.3 * angler
    for ic in ([1.0, 0.3, 1.1, 0.2, 0.25, 0.0], [1.2, 0.5, 0.8, 0.3, 0.4, 0.0]):
        E, Lz, I3 = _staeckel_inverse_labels(ic, pot=pot, delta=delta)
        aA = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
        jr, jz = aA.JR(E, Lz, I3), aA.Jz(E, Lz, I3)
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aA, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS, pot=pot
        )
        assert dJ < 1e-10 and dth < 1e-9 and dOm < 1e-9 and dH < 1e-9, (
            "A torus in the perfect ellipsoid does not round-trip through the "
            f"forward transformation: {dJ}, {dth}, {dOm}, {dH}"
        )
    return None


def test_actionAngleStaeckelInverse_degenerate_tori():
    # A planar orbit (J_z = 0) and a shell orbit (J_R = 0) are tori with one
    # degenerate libration: they are set up from their labels (the planar
    # orbit's third integral is that of any point in the plane, the shell
    # orbit's the one at which the u libration closes, taken from the
    # instance's own edge relation), evaluate in the plane and on the shell
    # with the other action the forward transformation's, and a torus that
    # reaches high latitudes brackets its v turning point beyond the
    # default scan
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import evaluatePotentials

    pot = _staeckel_inverse_potential()
    aAS = _staeckel_inverse_forward()
    angler = numpy.linspace(0.05, 6.2, 21)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    # planar
    E, Lz, I3 = _staeckel_inverse_labels([1.1, 0.4, 0.9, 0.0, 0.0, 0.0])
    aAP = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
    assert aAP.Jz(E, Lz, I3) == 0.0, "The planar orbit's vertical action is not zero"
    jr = aAP.JR(E, Lz, I3)
    R, vR, vT, z, vz, phi = aAP(jr, Lz, 0.0, angler, anglephi, anglez)
    assert numpy.amax(numpy.fabs(z)) < 1e-15 and numpy.amax(numpy.fabs(vz)) < 1e-15, (
        "The planar orbit does not stay in the plane"
    )
    H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
    assert numpy.ptp(H) / numpy.fabs(E) < 1e-12, "The planar orbit is not at one energy"
    f = aAS(R, vR, vT, z, vz, phi)
    assert (
        numpy.amax(numpy.fabs(f[0] - jr)) < 1e-9
        and numpy.amax(numpy.fabs(f[2])) < 1e-12
    ), "The planar orbit's radial action is not the forward transformation's"
    # shell
    E, Lz, _ = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    Ish = aAP._I3_shell(E, Lz, aAP._ushell(E, Lz))
    aASh = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[Ish])
    assert aASh.JR(E, Lz, Ish) == 0.0, "The shell orbit's radial action is not zero"
    jz = aASh.Jz(E, Lz, Ish)
    R, vR, vT, z, vz, phi = aASh(0.0, Lz, jz, angler, anglephi, anglez)
    H = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(pot, R, z)
    assert numpy.ptp(H) / numpy.fabs(E) < 1e-12, "The shell orbit is not at one energy"
    f = aAS(R, vR, vT, z, vz, phi)
    assert (
        numpy.amax(numpy.fabs(f[0])) < 1e-7 and numpy.amax(numpy.fabs(f[2] - jz)) < 1e-7
    ), (
        "The shell orbit's actions are not the forward transformation's: {:g} {:g}".format(
            numpy.amax(numpy.fabs(f[0])),
            numpy.amax(numpy.fabs(f[2] - jz)),
        )
    )
    assert numpy.all(numpy.fabs(numpy.array(aASh.Freqs(0.0, Lz, jz))) > 0.0), (
        "The shell orbit's frequencies are not all finite and non-zero"
    )
    # high latitudes
    E, Lz, I3 = _staeckel_inverse_labels([1.0, 0.1, 0.3, 0.0, 1.2, 0.0])
    aAH = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=64)
    dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
        aAH, aAH.JR(E, Lz, I3), Lz, aAH.Jz(E, Lz, I3), angler, anglephi, anglez, aAS=aAS
    )
    assert dJ < 1e-7 and dth < 1e-6 and dOm < 1e-7 and dH < 1e-8, (
        "The high-latitude torus does not round-trip through the forward transformation: %g %g %g %g"
        % (dJ, dth, dOm, dH)
    )
    return None
