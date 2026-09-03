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
def test_actionAngleSpherical_smallr():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ip = IsochronePotential()
    # Orbit at rperi, very small r
    o = Orbit([0.000000001, 0.0, ip.vcirc(0.000000001), 0.0, 0.0, 0.0])
    # Code should have rperi = 0
    assert (
        numpy.fabs(o.rperi(analytic=True, pot=ip, type="spherical") - 0.0) < 10.0**-10.0
    ), "rperi is not 0 for very small r"
    # Orbit just outside rperi, very small r
    o = Orbit([0.000000001, 0.0001, ip.vcirc(0.000000001), 0.0, 0.0, 0.0])
    assert (
        numpy.fabs(o.rperi(analytic=True, pot=ip, type="spherical") - 0.0) < 10.0**-10.0
    ), "rperi is not 0 for very small r"
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
        pot=isopot, nta=4 * 128, Es=[0.1, 1.0, 10.0], use_pointtransform=False
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
    tol = -10.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert numpy.fabs((Om - Omi) / Om) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation"
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

    gs = aAVI.plot_convergence(1.0, return_gridspec=True)
    aAVIpt.plot_convergence(1.0, overplot=gs)
    aAVIept.plot_convergence(1.0, overplot=gs)
    pyplot.close()
    gs = aAVI.plot_power(0.1, return_gridspec=True)
    gs = aAVI.plot_power([0.1, 1.0, 10.0], overplot=gs)
    gs = aAVIept.plot_power([0.1, 1.0, 10.0], overplot=gs)
    pyplot.close()
    aAVI.plot_orbit(1.0)
    aAVIept.plot_orbit(1.0)
    pyplot.close()
    return None


# Test that actionAngleVerticalInverse is the inverse of actionAngleVertical
def test_actionAngleVerticalInverse_interpolation_plotting(
    setup_actionAngleVerticalInverse_interpolated,
):
    import matplotlib.pyplot as pyplot

    aAVI, _ = setup_actionAngleVerticalInverse_interpolated
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
            pot=isopot, nta=4 * 128, Es=[300.0], use_pointtransform=False, maxiter=100
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


# ---------------- actionAngleStaeckelInverse tests ----------------
def _kk_torus_labels(kkp, delta, ic):
    """(E, Lz, I3) of an orbit IC in a Staeckel potential, computed
    independently of actionAngleStaeckelInverse's internals"""
    from galpy.orbit import Orbit
    from galpy.potential import evaluatePotentials

    o = Orbit(ic)
    E = float(o.E(pot=kkp))
    Lz = float(o.R() * o.vT())
    R, z, vR, vz = float(o.R()), float(o.z()), float(o.vR()), float(o.vz())
    d1 = numpy.sqrt(R**2.0 + (z + delta) ** 2.0)
    d2 = numpy.sqrt(R**2.0 + (z - delta) ** 2.0)
    u = numpy.arccosh((d1 + d2) / 2.0 / delta)
    v = numpy.arccos((d1 - d2) / 2.0 / delta)
    pu = delta * (vR * numpy.cosh(u) * numpy.sin(v) + vz * numpy.sinh(u) * numpy.cos(v))
    Uu = evaluatePotentials(kkp, delta * numpy.sinh(u), 0.0) * (
        numpy.sinh(u) ** 2.0 + 1.0
    )
    I3 = (
        E * numpy.sinh(u) ** 2.0
        - Uu
        - (pu**2.0 + Lz**2.0 / numpy.sinh(u) ** 2.0) / 2.0 / delta**2.0
    )
    return E, Lz, float(I3)


def test_actionAngleStaeckelInverse_actionsFreqs_vs_forward():
    # Actions and frequencies of the inverse setup agree with the forward
    # actionAngleStaeckel code run at high quadrature order; delta is
    # derived from the potential automatically
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    # No delta= given: derived from the KuzminKutuzov potential
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    assert numpy.fabs(aASI._delta - delta) < 1e-15, (
        "actionAngleStaeckelInverse did not derive delta from the potential"
    )
    aAS = actionAngleStaeckel(pot=kkp, delta=delta, c=True, order=200)
    jr, _, jz = (float(numpy.asarray(x).ravel()[0]) for x in aAS(*ic))
    assert numpy.fabs(aASI._jr[0] - jr) < 1e-6, (
        "actionAngleStaeckelInverse J_R does not agree with the forward actionAngleStaeckel"
    )
    assert numpy.fabs(aASI._jz[0] - jz) < 1e-6, (
        "actionAngleStaeckelInverse J_z does not agree with the forward actionAngleStaeckel"
    )
    out = aAS.actionsFreqs(*ic)
    OmR, Omphi, Omz = (float(numpy.asarray(out[i]).ravel()[0]) for i in (3, 4, 5))
    OmiR, Omiphi, Omiz = aASI.Freqs(aASI._jr[0], Lz, aASI._jz[0])
    assert numpy.fabs(OmiR - OmR) < 1e-8, (
        "actionAngleStaeckelInverse Omega_R does not agree with the forward code"
    )
    assert numpy.fabs(Omiz - Omz) < 1e-8, (
        "actionAngleStaeckelInverse Omega_z does not agree with the forward code"
    )
    assert numpy.fabs(Omiphi - Omphi) < 1e-8, (
        "actionAngleStaeckelInverse Omega_phi does not agree with the forward code"
    )
    return None


def test_actionAngleStaeckelInverse_wrapper():
    # An OblateStaeckelWrapperPotential can be passed directly, providing
    # delta and u0; specifying them as well is an error
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        OblateStaeckelWrapperPotential,
    )

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    swp = OblateStaeckelWrapperPotential(pot=kkp, delta=delta, u0=1.15)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    aASIw = actionAngleStaeckelInverse(pot=swp, Es=[E], Lzs=[Lz], I3s=[I3])
    assert numpy.fabs(aASI._jr[0] - aASIw._jr[0]) < 1e-12, (
        "actionAngleStaeckelInverse with a wrapped Staeckel potential does "
        "not agree with the unwrapped potential"
    )
    assert numpy.fabs(aASI._jz[0] - aASIw._jz[0]) < 1e-12, (
        "actionAngleStaeckelInverse with a wrapped Staeckel potential does "
        "not agree with the unwrapped potential"
    )
    with pytest.raises(TypeError):
        # delta and u0 are no longer accepted: the potential supplies them
        actionAngleStaeckelInverse(pot=swp, delta=1.3, Es=[E], Lzs=[Lz], I3s=[I3])
    return None


def test_actionAngleStaeckelInverse_angleconventions():
    # Round trip through both codes: the forward actionAngleStaeckel angles
    # of a point, fed to the inverse, return the same point (this requires
    # matching angle conventions)
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    aAS = actionAngleStaeckel(pot=kkp, delta=delta, c=True, order=200)
    out = aAS.actionsFreqsAngles(*ic)
    ar, ap, az = (numpy.asarray(out[i]).ravel() for i in (6, 7, 8))
    R, vR, vT, z, vz, phi = aASI(aASI._jr[0], Lz, aASI._jz[0], ar, ap, az)
    rec = numpy.array(
        [
            float(R[0]),
            float(vR[0]),
            float(vT[0]),
            float(z[0]),
            float(vz[0]),
            float(phi[0]),
        ]
    )
    diff = rec - numpy.array(ic)
    diff[5] = (diff[5] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    assert numpy.amax(numpy.fabs(diff)) < 1e-6, (
        "Feeding the forward actionAngleStaeckel angles to "
        "actionAngleStaeckelInverse does not return the original point; "
        "angle conventions do not match"
    )
    return None


def test_actionAngleStaeckelInverse_orbit():
    # Evaluating the torus along theta(t) = theta0 + Omega t reproduces the
    # integrated orbit over many periods and both branches, and the
    # Hamiltonian is constant along the reconstructed torus at machine
    # precision
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential, evaluatePotentials

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    aAS = actionAngleStaeckel(pot=kkp, delta=delta, c=True, order=200)
    for ic, postol in (
        ([1.1, 0.3, 0.9, 0.25, 0.2, 0.0], 1e-6),  # benign
        ([1.1, 0.9, 0.35, 0.15, 0.1, 0.0], 1e-6),  # eccentric
        ([1.1, 0.001, 0.8425895627614183, 0.15, 0.25, 0.0], 1e-6),  # near-shell J_R->0
        ([1.1, 0.4, 0.9, 0.002, 0.002, 0.0], 1e-5),  # near-planar J_z->0
    ):
        E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
        aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
        jr, jz = aASI._jr[0], aASI._jz[0]
        o = Orbit(ic)
        ts = numpy.linspace(0.0, 60.0, 501)
        o.integrate(ts, kkp, method="dop853_c")
        out = aAS.actionsFreqsAngles(*ic)
        th0 = [float(numpy.asarray(out[i]).ravel()[0]) for i in (6, 7, 8)]
        OmR, Omphi, Omz = aASI.Freqs(jr, Lz, jz)
        R, vR, vT, z, vz, phi = aASI(
            jr,
            Lz,
            jz,
            th0[0] + OmR * ts,
            th0[1] + Omphi * ts,
            th0[2] + Omz * ts,
        )
        assert numpy.amax(numpy.fabs(R - o.R(ts))) < postol, (
            "actionAngleStaeckelInverse orbit traversal does not agree with "
            "direct orbit integration in R"
        )
        assert numpy.amax(numpy.fabs(z - o.z(ts))) < postol, (
            "actionAngleStaeckelInverse orbit traversal does not agree with "
            "direct orbit integration in z"
        )
        dphi = numpy.fabs((phi - o.phi(ts) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        assert numpy.amax(dphi) < postol, (
            "actionAngleStaeckelInverse orbit traversal does not agree with "
            "direct orbit integration in phi"
        )
        H = 0.5 * (vR**2.0 + vz**2.0 + vT**2.0) + evaluatePotentials(kkp, R, z)
        assert numpy.amax(numpy.fabs(H - E)) / numpy.fabs(E) < 1e-13, (
            "Hamiltonian is not constant at machine precision along the "
            "reconstructed actionAngleStaeckelInverse torus"
        )
    return None


def test_actionAngleStaeckelInverse_xvFreqs_consistency():
    # xvFreqs returns the same (x,v) as __call__ plus the frequencies
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    jr, jz = aASI._jr[0], aASI._jz[0]
    ang = (numpy.array([0.1, 2.0]), numpy.array([0.3, 1.0]), numpy.array([0.2, 4.0]))
    out1 = aASI(jr, Lz, jz, *ang)
    out2 = aASI.xvFreqs(jr, Lz, jz, *ang)
    for o1, o2 in zip(out1, out2[:6]):
        assert numpy.all(numpy.fabs(numpy.array(o1) - numpy.array(o2)) < 1e-14), (
            "actionAngleStaeckelInverse xvFreqs phase-space output does not "
            "agree with __call__"
        )
    assert numpy.all(
        numpy.fabs(numpy.array(out2[6:]) - numpy.array(aASI.Freqs(jr, Lz, jz))) < 1e-14
    ), "actionAngleStaeckelInverse xvFreqs frequencies do not agree with Freqs"
    return None


def test_actionAngleStaeckelInverse_wrongactions_error():
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError) as excinfo:
        aASI(0.2, 0.5, 0.1, 0.0, 0.0, 0.0)
        pytest.fail(
            "Evaluating actionAngleStaeckelInverse with actions not set up "
            "should have raised a ValueError, but did not"
        )
    with pytest.raises(ValueError) as excinfo:
        aASI.Freqs(0.2, 0.5, 0.1)
        pytest.fail(
            "Freqs with actions not set up should have raised a ValueError, but did not"
        )
    return None


def test_actionAngleStaeckelInverse_periodmatrix_finitediff():
    # Self-consistency of the six complete profile integrals: the period
    # matrix d(J_R,J_z)/d(E,I3,Lz), which is what builds the frequencies and
    # the angle-profile coefficients, must agree with finite differences of
    # the actions across neighbouring tori. This is a test-suite check only:
    # computing it at setup would cost ~5x the setup of a torus, and the
    # construction has no fitted ingredient that could silently drift
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    # Rows of M from the stored complete profiles (see the class docstring)
    PEu, PIu, PLu = (p[0, -1] for p in aASI._Pu)
    PEv, PIv, PLv = (p[0, -1] for p in aASI._Pv)
    M = (
        numpy.array(
            [
                [PEu, -PIu, -PLu],
                [PEv, PIv, -PLv],
            ]
        )
        / numpy.pi
    )
    h = 1e-6
    for jj, (dE, dI3, dLz) in enumerate(((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h))):
        up = actionAngleStaeckelInverse(
            pot=kkp, Es=[E + dE], Lzs=[Lz + dLz], I3s=[I3 + dI3]
        )
        dw = actionAngleStaeckelInverse(
            pot=kkp, Es=[E - dE], Lzs=[Lz - dLz], I3s=[I3 - dI3]
        )
        fd = numpy.array(
            [
                (up._jr[0] - dw._jr[0]) / 2.0 / h,
                (up._jz[0] - dw._jz[0]) / 2.0 / h,
            ]
        )
        assert numpy.all(numpy.fabs((fd - M[:, jj]) / M[:, jj]) < 1e-8), (
            "Period matrix of actionAngleStaeckelInverse does not agree with "
            "finite differences of the actions across neighbouring tori "
            "(column %i): %s vs %s" % (jj, M[:, jj], fd)
        )
    return None


def test_actionAngleStaeckelInverse_thinshell():
    # A u oscillation narrower than the turning-point scan spacing is
    # recovered by refining the maximum of W_u, and its profiles are
    # computed stably through the turning-point limits of Q = W/[y(1-y)]
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential, evaluatePotentials

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)

    def Uofu(u):
        return evaluatePotentials(kkp, delta * numpy.sinh(u), 0.0) * (
            numpy.sinh(u) ** 2.0 + 1.0
        )

    # Labels of an exact shell torus (double root of W_u at ustar):
    # W_u'(ustar) = 0 and W_u(ustar) = 0 solved for (E, I3) at fixed Lz
    ustar, Lz = 1.0, 0.99
    du = 1e-6
    Up = (Uofu(ustar + du) - Uofu(ustar - du)) / (2.0 * du)
    sh, ch = numpy.sinh(ustar), numpy.cosh(ustar)
    Estar = (Up - Lz**2.0 * ch / (delta**2.0 * sh**3.0)) / numpy.sinh(2.0 * ustar)
    I3star = Estar * sh**2.0 - Uofu(ustar) - Lz**2.0 / (2.0 * delta**2.0 * sh**2.0)
    # Slightly above the shell energy: a genuine, ultra-thin oscillation
    aASI = actionAngleStaeckelInverse(
        pot=kkp, Es=[Estar + 1e-9], Lzs=[Lz], I3s=[I3star]
    )
    assert aASI._umaxs[0] - aASI._umins[0] < 1e-4, (
        "Ultra-thin shell torus was not recovered by the turning-point "
        "refinement (the u oscillation should be narrower than the scan "
        "spacing)"
    )
    assert 0.0 < aASI._jr[0] < 1e-8, (
        "Ultra-thin shell torus does not have the expected tiny radial action"
    )
    # The frequencies must connect continuously to those of a nearby,
    # normally-resolved torus
    aASI_ref = actionAngleStaeckelInverse(
        pot=kkp, Es=[Estar + 1e-4], Lzs=[Lz], I3s=[I3star]
    )
    assert numpy.fabs(aASI._OmegaR[0] - aASI_ref._OmegaR[0]) < 1e-3, (
        "Radial frequency of the ultra-thin shell torus does not connect "
        "to that of a nearby resolved torus"
    )
    assert numpy.fabs(aASI._Omegaz[0] - aASI_ref._Omegaz[0]) < 1e-4, (
        "Vertical frequency of the ultra-thin shell torus does not connect "
        "to that of a nearby resolved torus"
    )
    assert numpy.fabs(aASI._Omegaphi[0] - aASI_ref._Omegaphi[0]) < 1e-4, (
        "Azimuthal frequency of the ultra-thin shell torus does not connect "
        "to that of a nearby resolved torus"
    )
    return None


def test_actionAngleStaeckelInverse_ultrathin_shell_frequencies():
    # Very thin u oscillations: the direct evaluation of W_u near the turning
    # points is pure cancellation noise there, so Q must be reconstructed from
    # the analytic derivative wherever W is small RELATIVE to its size on the
    # torus -- a fixed threshold in the anomaly leaves nodes on the wrong side
    # of the noise floor for a thin torus, and a single such node used to send
    # 1/sqrt(Q) to ~1e161 and the frequencies to zero
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential, evaluatePotentials

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)

    def Uofu(u):
        return evaluatePotentials(kkp, delta * numpy.sinh(u), 0.0) * (
            numpy.sinh(u) ** 2.0 + 1.0
        )

    # Shell-orbit labels (double root of W_u at ustar), then approach them.
    # The tolerance is set by the construction, not by the assertion: the
    # frequencies hold to 1.2e-4 down to a u-width of 1.4e-4 and 4.5e-4 at
    # 1.4e-5, then degrade smoothly (6e-3 at 1.4e-6, 6e-2 at 1.4e-7) as the
    # turning-point solve runs out of digits, so the test stops where the
    # construction is still trustworthy
    ustar, Lz, du = 1.0, 0.99, 1e-6
    Up = (Uofu(ustar + du) - Uofu(ustar - du)) / (2.0 * du)
    sh, ch = numpy.sinh(ustar), numpy.cosh(ustar)
    Estar = (Up - Lz**2.0 * ch / (delta**2.0 * sh**3.0)) / numpy.sinh(2.0 * ustar)
    I3star = Estar * sh**2.0 - Uofu(ustar) - Lz**2.0 / (2.0 * delta**2.0 * sh**2.0)
    ref = None
    for dE in (1e-6, 1e-8, 1e-10):
        aASI = actionAngleStaeckelInverse(
            pot=kkp, Es=[Estar + dE], Lzs=[Lz], I3s=[I3star]
        )
        assert numpy.isfinite(aASI._OmegaR[0]) and aASI._OmegaR[0] > 0.0, (
            "Radial frequency of an ultra-thin shell torus is not finite and "
            "positive (dE = %g)" % dE
        )
        if ref is None:
            ref = aASI._OmegaR[0]
        assert numpy.fabs(aASI._OmegaR[0] / ref - 1.0) < 1e-3, (
            "Radial frequency of an ultra-thin shell torus does not stay at "
            "the shell-limit value as the torus is thinned (dE = %g)" % dE
        )
    return None


def test_actionAngleStaeckelInverse_unresolvable_shell_error():
    # One ulp below the shell edge the two u turning points are separated by
    # a few ulp: the oscillation cannot be resolved in double precision and
    # the frequencies built on it would be noise (they used to come out as a
    # plausible-looking zero), so setup must fail loudly instead
    from scipy import optimize

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential, evaluatePotentials

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    Lz = 0.99
    E = -1.5

    def Wu(u, I3):
        return (
            2.0
            * delta**2.0
            * (
                E * numpy.sinh(u) ** 2.0
                - evaluatePotentials(kkp, delta * numpy.sinh(u), 0.0)
                * (numpy.sinh(u) ** 2.0 + 1.0)
                - I3
            )
            - Lz**2.0 / numpy.sinh(u) ** 2.0
        )

    def maxWu(I3):
        return -optimize.minimize_scalar(
            lambda u: -Wu(u, I3),
            bounds=(1e-3, 12.0),
            method="bounded",
            options={"xatol": 1e-13},
        ).fun

    # I3 of the shell edge (double root of W_u), then one ulp inside it
    I3lo = Lz**2.0 / 2.0 / delta**2.0 - E  # the planar edge, where W_u is widest
    I3shell = optimize.brentq(maxWu, I3lo, I3lo + 200.0, xtol=1e-14)
    I3 = numpy.nextafter(I3shell, -numpy.inf)
    with pytest.raises(ValueError, match="unresolvable in double precision"):
        actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    return None


@pytest.fixture(scope="module")
def setup_actionAngleStaeckelInverse_interpolated():
    # A grid fine enough that the tests are limited by the construction
    # rather than by the interpolation; shared across the tests below,
    # because setting it up takes a while
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    aASI = actionAngleStaeckelInverse(
        pot=kkp,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=11,
        nE=11,
        nI3=11,
    )
    aAS = actionAngleStaeckel(pot=kkp, delta=1.3, c=True, order=200)
    return aASI, aAS, kkp


def test_actionAngleStaeckelInverse_interp_roundtrip(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # With setup_interp, arbitrary actions inside the grid are accepted, and
    # a round trip through the forward transformation returns the original
    # phase-space point
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    for ic in (
        [1.1, 0.3, 0.9, 0.25, 0.2, 0.0],
        [1.0, -0.2, 1.05, -0.15, 0.25, 1.2],  # z<0, vR<0
        [1.3, 0.5, 0.75, 0.1, 0.35, 2.1],
    ):
        jr, jphi, jz = (float(numpy.atleast_1d(x)[0]) for x in aAS(*ic))
        out = aAS.actionsFreqsAngles(*ic)
        angles = [float(numpy.atleast_1d(out[ii])[0]) for ii in (6, 7, 8)]
        Rvv = [
            float(numpy.atleast_1d(q)[0])
            for q in aASI(jr, jphi, jz, angles[0], angles[1], angles[2])
        ]
        for got, want, name in zip(Rvv[:5], ic[:5], ("R", "vR", "vT", "z", "vz")):
            # the canonical family's honest interpolation error at this
            # grid (the removed interpolated-direct path was ~2x tighter
            # here at the price of a symplectic defect at 1e-3)
            assert numpy.fabs(got - want) < 4e-4, (
                "Interpolated actionAngleStaeckelInverse does not invert the "
                "forward transformation for %s (%g vs %g)" % (name, got, want)
            )
    return None


def test_actionAngleStaeckelInverse_interp_convergence_and_freqs(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # The round-trip error is set by the interpolation, so it must fall
    # steeply as the grid is refined; the interpolated frequencies must match
    # those of the directly-constructed torus
    from galpy.actionAngle import actionAngleStaeckelInverse

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    jr, jphi, jz = (float(numpy.atleast_1d(x)[0]) for x in aAS(*ic))
    out = aAS.actionsFreqsAngles(*ic)
    angles = [float(numpy.atleast_1d(out[ii])[0]) for ii in (6, 7, 8)]
    errs = []
    for n in (5, 9):
        coarse = actionAngleStaeckelInverse(
            pot=kkp,
            setup_interp=True,
            Rmin=0.7,
            Rmax=1.6,
            Rinf=8.0,
            nLz=n,
            nE=n,
            nI3=n,
        )
        Rvv = [
            float(numpy.atleast_1d(q)[0])
            for q in coarse(jr, jphi, jz, angles[0], angles[1], angles[2])
        ]
        errs.append(numpy.fabs(Rvv[0] - ic[0]))
    assert errs[1] < 0.25 * errs[0], (
        "Interpolated actionAngleStaeckelInverse does not converge with the "
        "size of the grid (%g -> %g)" % (errs[0], errs[1])
    )
    freqs = aASI.Freqs(jr, jphi, jz)
    for got, want, name in zip(
        freqs,
        (float(numpy.atleast_1d(out[ii])[0]) for ii in (3, 4, 5)),
        ("Omega_R", "Omega_phi", "Omega_z"),
    ):
        # the frequencies are the stored energy table's own derivative
        # chains (the integrator contract: exactly consistent with the
        # family), and match the true frequencies to the chain accuracy
        assert numpy.fabs(got / want - 1.0) < 5e-4, (
            "Interpolated actionAngleStaeckelInverse frequency %s does not "
            "agree with the forward code (%g vs %g)" % (name, got, want)
        )
    return None


def _canon_integral_labels(aASI, jr, jphi, jz):
    # the (E, I3) labels of an interpolated canonical torus, from the same
    # closed rectified relations the integrals entry point inverts
    import numpy

    x = aASI._canon_coords(jr, jphi, jz)
    E = float(aASI._canon_table_eval(numpy.atleast_2d(x))[2, 0])
    wI = x[2] / (aASI._nI3 - 1)
    Ipl = aASI._I3_planar(E, jphi)
    Ish = aASI._I3_shell(E, jphi)
    I3 = Ipl + numpy.sin(numpy.pi * wI / 2.0) ** 2 * (Ish - Ipl)
    return E, I3


def test_actionAngleStaeckelInverse_interp_integrals_by_name(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # A torus can be labelled by its integrals of motion by name, which works
    # the same with and without units; the dimensions alone could not tell
    # (E, L_z, I3) from the actions when everything is in internal units
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    jr, jphi, jz = 0.06, 0.9, 0.03
    angles = [numpy.array([0.3, 2.7]) for _ in range(3)]
    E, I3 = _canon_integral_labels(aASI, jr, jphi, jz)
    by_name = numpy.array(aASI(*angles, E=E, Lz=jphi, I3=I3))
    by_act = numpy.array(aASI(jr, jphi, jz, *angles))
    # exact at nodes; between nodes the closed rectification and the
    # E-table's Lz-direction spline agree to spline error, not machine
    assert numpy.all(numpy.fabs(by_name - by_act) < 1e-4), (
        "Labelling a torus by its integrals by name disagrees with labelling "
        "it by its actions"
    )
    # the same through Freqs and xvFreqs
    assert numpy.all(
        numpy.fabs(
            numpy.array(aASI.Freqs(E=E, Lz=jphi, I3=I3))
            - numpy.array(aASI.Freqs(jr, jphi, jz))
        )
        < 1e-4
    ), "Freqs by integral name disagrees with Freqs by actions"
    assert len(aASI.xvFreqs(*angles, E=E, Lz=jphi, I3=I3)) == 9, (
        "xvFreqs by integral name does not return the expected quantities"
    )
    # all of the integrals have to be given
    with pytest.raises(ValueError) as excinfo:
        aASI(*angles, E=E, Lz=jphi)
    assert "I3" in str(excinfo.value), (
        "Leaving out one of the integrals of motion does not say which is missing"
    )


def test_actionAngleStaeckelInverse_interp_outside_grid_message(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # Actions outside the grid have to say which way they fall out and what
    # the grid does reach: rho and zeta are rectified coordinates, so a
    # near-circular torus and a too-energetic one are otherwise reported
    # identically, with nothing to act on
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    angles = [numpy.array([0.3]) for _ in range(3)]
    with pytest.raises(ValueError) as excinfo:
        aASI(1e-7, 0.9, 1e-7, *angles)
    assert "below" in str(excinfo.value) and "J_R+J_z" in str(excinfo.value), (
        "A near-circular torus outside the grid does not report that it falls "
        "below the covered total action"
    )
    with pytest.raises(ValueError) as excinfo:
        aASI(3.0, 0.9, 2.0, *angles)
    assert "above" in str(excinfo.value) and "J_R+J_z" in str(excinfo.value), (
        "A too-energetic torus outside the grid does not report that it falls "
        "above the covered total action"
    )
    # (the canonical grid covers the full oscillation-direction range,
    # so the third directional case of the old grid no longer exists)
    return None


def test_actionAngleStaeckelInverse_interp_integrals_roundtrip(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # Reading the integrals off an interpolated torus and asking for that
    # torus back has to return the same grid point: both directions go
    # through the same closed rectified relations. The energy along the
    # returned torus matches the label to the family's interpolation error
    # (machine-exact H was the removed interpolated-direct path's virtue,
    # bought with contingent canonicity; the canonical family trades it
    # for a symplectic defect at the finite-difference floor)
    from galpy.potential import evaluatePotentials

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    for jr, jphi, jz in [(0.06, 0.9, 0.03), (0.12, 1.1, 0.08), (0.02, 0.8, 0.10)]:
        x = aASI._canon_coords(jr, jphi, jz)
        E, I3 = _canon_integral_labels(aASI, jr, jphi, jz)
        back = aASI._canon_coords_integrals(E, jphi, I3)
        assert numpy.all(numpy.fabs(numpy.array(x) - numpy.array(back)) < 2e-4), (
            "Labelling an interpolated torus by its integrals and looking it "
            "back up does not return the same grid point"
        )
        angles = [numpy.array([0.3, 2.7]) for _ in range(3)]
        R, vR, vT, z, vz, phi = aASI(E, jphi, I3, *angles, integrals=True)
        H = 0.5 * (vR**2.0 + vT**2.0 + vz**2.0) + evaluatePotentials(
            kkp, R, z, use_physical=False
        )
        assert numpy.all(numpy.fabs(H - E) < 1e-4), (
            "The interpolated torus requested by its integrals does not have "
            "the requested energy within the interpolation error"
        )
    return None


def test_actionAngleStaeckelInverse_interp_by_integrals(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # Tori are often specified by their integrals rather than their actions,
    # and integrals=True takes them directly: the grid coordinates follow
    # from the circular/outer energies and the planar/shell edges with no
    # inversion, so this route is at least as accurate as the action one
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    jr, jphi, jz = (float(numpy.atleast_1d(x)[0]) for x in aAS(*ic))
    out = aAS.actionsFreqsAngles(*ic)
    angles = [float(numpy.atleast_1d(out[ii])[0]) for ii in (6, 7, 8)]
    E, Lz, I3 = _kk_torus_labels(kkp, 1.3, ic)
    by_int = [
        float(numpy.atleast_1d(q)[0]) for q in aASI(E, Lz, I3, *angles, integrals=True)
    ]
    by_act = [
        float(numpy.atleast_1d(q)[0])
        for q in aASI(jr, jphi, jz, angles[0], angles[1], angles[2])
    ]
    for aa, bb, name in zip(by_int, by_act, ("R", "vR", "vT", "z", "vz", "phi")):
        diff = numpy.fabs(aa - bb)
        if name == "phi":  # returned modulo 2 pi
            diff = numpy.fabs((aa - bb + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        assert diff < 1e-4, (
            "The integral and action entry points of the interpolated "
            "actionAngleStaeckelInverse disagree for %s (%g vs %g)" % (name, aa, bb)
        )
    # the label Newton converges far below the interpolation error that now
    # dominates both routes, so the inversion no longer costs anything
    # measurable and neither route is systematically ahead; what must hold
    # is that skipping it does not make the answer worse
    assert numpy.fabs(by_int[0] - ic[0]) <= 2.0 * numpy.fabs(by_act[0] - ic[0]), (
        "The integral entry point, which requires no inversion, is less "
        "accurate than the action entry point"
    )
    # frequencies by the same route
    freqs = aASI.Freqs(E, Lz, I3, integrals=True)
    for got, want in zip(
        freqs, (float(numpy.atleast_1d(out[ii])[0]) for ii in (3, 4, 5))
    ):
        assert numpy.fabs(got / want - 1.0) < 5e-4, (
            "Interpolated frequencies by integrals disagree with the forward code"
        )
    # two consecutive evaluations of one torus: the second hits the cache
    aASI(E, Lz, I3, *angles, integrals=True)
    again = [
        float(numpy.atleast_1d(q)[0]) for q in aASI(E, Lz, I3, *angles, integrals=True)
    ]
    assert numpy.all(numpy.array(again) == numpy.array(by_int)), (
        "Re-evaluating the same interpolated torus does not reproduce it"
    )
    # outside the grid, by either route and in either direction
    with pytest.raises(ValueError, match="outside the grid"):
        aASI(E, 5.0, I3, *angles, integrals=True)
    with pytest.raises(ValueError, match="outside the grid"):
        aASI(-1e3, Lz, I3, *angles, integrals=True)
    with pytest.raises(ValueError, match="outside the grid"):
        aASI(jr, 5.0, jz, *angles)  # L_z beyond the grid, action route
    with pytest.raises(ValueError, match="outside the grid"):
        aASI(50.0, Lz, 50.0, *angles)  # actions beyond the outer energy
    return None


def test_actionAngleStaeckelInverse_integrals_requires_interp():
    # integrals=True only makes sense for an interpolating instance
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, 1.3, ic)
    aASI = actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="requires an actionAngleStaeckelInverse"):
        aASI(E, Lz, I3, 0.1, 0.2, 0.3, integrals=True)
    with pytest.raises(ValueError, match="requires an actionAngleStaeckelInverse"):
        aASI.Freqs(E, Lz, I3, integrals=True)
    return None


def test_actionAngleStaeckelInverse_interp_degenerate_edges(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # Interpolation must work at the edges of action space, not just inside:
    # J_R = 0 is a shell orbit (u fixed on a spheroid) and J_z = 0 a planar
    # one. The construction degenerates exactly there, so those grid nodes
    # carry the analytic limits -- the degenerate oscillation is harmonic, so
    # its angle is its anomaly and its cross profiles vanish.
    from galpy.potential import evaluatePotentials
    from galpy.util import coords

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    jr, jphi, jz = (
        float(numpy.atleast_1d(x)[0]) for x in aAS(1.1, 0.3, 0.9, 0.25, 0.2)
    )
    th = numpy.linspace(0.05, 2.0 * numpy.pi - 0.05, 32)
    # J_z = 0: the orbit must lie in the midplane, with no vertical motion
    R, vR, vT, z, vz, phi = (
        numpy.atleast_1d(q) for q in aASI(jr, jphi, 0.0, th, th * 0.7, th * 1.3)
    )
    assert numpy.amax(numpy.fabs(z)) < 1e-12, (
        "A J_z = 0 torus is not confined to the midplane"
    )
    assert numpy.amax(numpy.fabs(vz)) < 1e-12, "A J_z = 0 torus has vertical motion"
    H = 0.5 * (vR**2.0 + vz**2.0 + vT**2.0) + evaluatePotentials(kkp, R, z)
    # the canonical family's H-constancy is its interpolation error
    # (machine-constant H was the removed path's p = sqrt(W) virtue,
    # bought with contingent canonicity)
    assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-4, (
        "The Hamiltonian is not constant along an interpolated J_z = 0 torus"
    )
    # J_R = 0: a shell orbit, confined to a spheroid of constant u
    R, vR, vT, z, vz, phi = (
        numpy.atleast_1d(q) for q in aASI(0.0, jphi, jz, th, th * 0.7, th * 1.3)
    )
    u, _ = coords.Rz_to_uv(R, z, delta=1.3)
    assert numpy.ptp(u) < 1e-6, (
        "A J_R = 0 torus is not confined to a spheroid of constant u"
    )
    H = 0.5 * (vR**2.0 + vz**2.0 + vT**2.0) + evaluatePotentials(kkp, R, z)
    assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-4, (
        "The Hamiltonian is not constant along an interpolated J_R = 0 torus"
    )
    return None


def test_actionAngleStaeckelInverse_notorus_errors():
    # Unbound / invalid torus labels raise errors
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    delta = 1.3
    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=delta)
    with pytest.raises(ValueError) as excinfo:
        # E > 0: the u oscillation is not enclosed (unbound)
        actionAngleStaeckelInverse(pot=kkp, Es=[10.0], Lzs=[0.99], I3s=[1.9])
        pytest.fail(
            "Setting up an unbound actionAngleStaeckelInverse torus should "
            "have raised a ValueError, but did not"
        )
    with pytest.raises(ValueError) as excinfo:
        # deeply negative E with large I3: W_u < 0 everywhere
        actionAngleStaeckelInverse(pot=kkp, Es=[-3.0], Lzs=[0.99], I3s=[5.0])
        pytest.fail(
            "Setting up an actionAngleStaeckelInverse torus with no bound u "
            "oscillation should have raised a ValueError, but did not"
        )
    # valid u oscillation, but the v oscillation does not reach the midplane
    ic = [1.1, 0.3, 0.9, 0.25, 0.2, 0.0]
    E, Lz, I3 = _kk_torus_labels(kkp, delta, ic)
    with pytest.raises(ValueError) as excinfo:
        actionAngleStaeckelInverse(pot=kkp, Es=[E], Lzs=[Lz], I3s=[I3 - 0.3])
        pytest.fail(
            "Setting up an actionAngleStaeckelInverse torus that does not "
            "reach the midplane should have raised a ValueError, but did not"
        )
    with pytest.raises(OSError) as excinfo:
        from galpy.potential import LogarithmicHaloPotential

        actionAngleStaeckelInverse(
            pot=LogarithmicHaloPotential(normalize=1.0),
            Es=[0.78],
            Lzs=[0.99],
            I3s=[0.1],
        )
        pytest.fail(
            "Setting up actionAngleStaeckelInverse without delta for a "
            "potential it cannot be derived from should have raised an "
            "OSError, but did not"
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


# ---------- the canonical (momentum-matched) Staeckel construction (T1;
# ---------- STAECKEL_CANONICAL_MATH.md section 10)
_aascanon_cache = {}


def _staeckel_canonical_tori():
    # four valid KK tori spanning mild / eccentric-in-u / near-shell /
    # near-planar, built through the direct construction's own edge helpers
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        OblateStaeckelWrapperPotential,
    )

    kk = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=3.0, Delta=1.25)
    probe = actionAngleStaeckelInverse.__new__(actionAngleStaeckelInverse)
    probe._pot = kk
    probe._staeckelwrap = OblateStaeckelWrapperPotential(pot=kk, delta=kk._delta)
    probe._delta = probe._staeckelwrap._delta
    Es, Lzs, I3s = [], [], []
    for Lz, dE, f in (
        (0.9, 0.10, 0.50),
        (0.45, 0.45, 0.50),
        (0.6, 0.15, 0.97),
        (0.9, 0.15, 0.03),
    ):
        Rc, Ec = probe._circular_orbit(Lz)
        E = Ec + dE
        lo, hi = probe._I3_planar(E, Lz), probe._I3_shell(E, Lz)
        Es.append(E)
        Lzs.append(Lz)
        I3s.append(lo + f * (hi - lo))
    return kk, Es, Lzs, I3s


def _staeckel_canonical_setup():
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    if "canon" not in _aascanon_cache:
        kk, Es, Lzs, I3s = _staeckel_canonical_tori()
        _aascanon_cache["pot"] = kk
        _aascanon_cache["canon"] = actionAngleStaeckelInverse(
            pot=kk, Es=Es, Lzs=Lzs, I3s=I3s, canonical=True, ncanon=128, npt=32
        )
        _aascanon_cache["direct"] = actionAngleStaeckelInverse(
            pot=kk, Es=Es, Lzs=Lzs, I3s=I3s
        )
    return _aascanon_cache["pot"], _aascanon_cache["canon"], _aascanon_cache["direct"]


def test_actionAngleStaeckelInverse_canonical_collapse():
    # the section-10 claim as an executable fact: the momentum-matched
    # product lift puts each Staeckel torus EXACTLY on its equal-action
    # isochrone torus (max|J^A - label| at machine noise), and the
    # zero-mode labels equal the direct quadrature actions (Stokes)
    _, aac, _ = _staeckel_canonical_setup()
    assert aac._can_maxdev < 1e-11, (
        "The canonical lift's action deviation is not at the machine floor: "
        "%g" % aac._can_maxdev
    )
    assert aac._can_stokes < 1e-13, (
        "The canonical zero-mode labels do not equal the direct quadrature "
        "actions: %g" % aac._can_stokes
    )
    return None


def test_actionAngleStaeckelInverse_canonical_vs_direct():
    # the through-the-toy evaluation (angle Newton -> correspondence
    # tables -> analytic isochrone inverse -> per-degree un-lift) equals
    # the exact direct reconstruction pointwise
    _, aac, aad = _staeckel_canonical_setup()
    angler = numpy.linspace(0.4, 5.9, 7)
    anglephi = numpy.linspace(0.0, 5.0, 7)
    anglez = numpy.linspace(0.3, 6.0, 7)
    for ii in range(len(aac._Es)):
        jr, Lz, jz = aac._jr[ii], aac._Lzs[ii], aac._jz[ii]
        xc = numpy.array(aac._xvFreqs(jr, Lz, jz, angler, anglephi, anglez)[:6])
        xd = numpy.array(aad._xvFreqs(jr, Lz, jz, angler, anglephi, anglez)[:6])
        assert numpy.max(numpy.fabs(xc - xd)) < 1e-9, (
            "Canonical and direct evaluation disagree on torus %d: %g"
            % (ii, numpy.max(numpy.fabs(xc - xd)))
        )
    return None


def test_actionAngleStaeckelInverse_canonical_roundtrip():
    # the forward Staeckel code recovers the requested actions on the
    # canonically reconstructed points, and the energy is conserved
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import evaluatePotentials

    kk, aac, _ = _staeckel_canonical_setup()
    aS = actionAngleStaeckel(pot=kk, delta=kk._delta, c=False)
    angler = numpy.linspace(0.4, 5.9, 5)
    anglephi = numpy.linspace(0.0, 5.0, 5)
    anglez = numpy.linspace(0.3, 6.0, 5)
    for ii in (0, 2):
        jr, Lz, jz = aac._jr[ii], aac._Lzs[ii], aac._jz[ii]
        R, vR, vT, z, vz, phi = aac._evaluate(jr, Lz, jz, angler, anglephi, anglez)
        E = 0.5 * (vR**2 + vT**2 + vz**2) + evaluatePotentials(
            kk, R, z, use_physical=False
        )
        assert numpy.max(numpy.fabs(E - aac._Es[ii])) < 1e-11, (
            "The canonical reconstruction does not conserve the torus "
            "energy: %g" % numpy.max(numpy.fabs(E - aac._Es[ii]))
        )
        ji = aS(R, vR, vT, z, vz, phi)
        assert numpy.max(numpy.fabs(ji[0] - jr)) < 1e-6
        assert numpy.max(numpy.fabs(ji[1] - Lz)) < 1e-10
        assert numpy.max(numpy.fabs(ji[2] - jz)) < 1e-6
    return None


def test_actionAngleStaeckelInverse_canonical_polar():
    # the required Lz -> 0 edge: the toy vertical loop opens toward
    # [0, pi] and the v-map's anomaly parametrization must stay clean
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    kk, _, _ = _staeckel_canonical_setup()
    from galpy.potential import OblateStaeckelWrapperPotential

    probe = actionAngleStaeckelInverse.__new__(actionAngleStaeckelInverse)
    probe._pot = kk
    probe._staeckelwrap = OblateStaeckelWrapperPotential(pot=kk, delta=kk._delta)
    probe._delta = probe._staeckelwrap._delta
    Es, Lzs, I3s = [], [], []
    for Lz in (1e-3, 0.05):
        Rc, Ec = probe._circular_orbit(Lz)
        E = Ec + 0.2
        lo, hi = probe._I3_planar(E, Lz), probe._I3_shell(E, Lz)
        Es.append(E)
        Lzs.append(Lz)
        I3s.append(lo + 0.5 * (hi - lo))
    # the pole boundary layer (width ~ Lz/L^A) needs map resolution; the
    # measured ladder: maxdev 5.0e-5 / 2.1e-7 / 7.2e-12 at (ncanon, npt) =
    # (128, 32) / (256, 64) / (512, 128) -- spectral, resolution-controlled
    aac = actionAngleStaeckelInverse(
        pot=kk, Es=Es, Lzs=Lzs, I3s=I3s, canonical=True, ncanon=512, npt=128
    )
    aad = actionAngleStaeckelInverse(pot=kk, Es=Es, Lzs=Lzs, I3s=I3s)
    assert aac._can_maxdev < 1e-10, (
        "The canonical lift fails at the polar edge: %g" % aac._can_maxdev
    )
    angler = numpy.linspace(0.4, 5.9, 5)
    for ii in range(2):
        jr, Lz, jz = aac._jr[ii], aac._Lzs[ii], aac._jz[ii]
        xc = numpy.array(
            aac._xvFreqs(jr, Lz, jz, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0)[:6]
        )
        xd = numpy.array(
            aad._xvFreqs(jr, Lz, jz, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0)[:6]
        )
        assert numpy.max(numpy.fabs(xc - xd)) < 1e-10, (
            "Canonical and direct evaluation disagree at the polar edge: %g"
            % numpy.max(numpy.fabs(xc - xd))
        )
    return None


def test_actionAngleStaeckelInverse_canonical_errors():
    # guarded misuse and the defensive raises, fired for real
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    kk, aac, _ = _staeckel_canonical_setup()
    _, Es, Lzs, I3s = _staeckel_canonical_tori()
    with pytest.raises(ValueError) as excinfo:
        actionAngleStaeckelInverse(
            pot=kk, Es=Es[:1], Lzs=Lzs[:1], I3s=I3s[:1], canonical=True, ncanon=127
        )
    assert "even" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        actionAngleStaeckelInverse(
            pot=kk,
            Es=Es[:1],
            Lzs=Lzs[:1],
            I3s=I3s[:1],
            canonical=True,
            ncanon=64,
            npt=99,
        )
    assert "npt" in str(excinfo.value)
    # the Newton else-raises via maxiter=0 (white-box)
    maxiter = aac._maxiter
    try:
        aac._maxiter = 0
        # A monotone map is still inverted, by bracketing, when Newton is
        # not allowed to run at all: sum_m m |D_m| < 1 makes eta(tau)
        # strictly increasing, so the root on [0, 2 pi] is unique.
        Dm = aac._can_Dmu[0]
        assert numpy.sum(numpy.arange(1, len(Dm) + 1) * numpy.fabs(Dm)) < 1.0
        tau = aac._tau_of_eta(numpy.array([2.0]), Dm)
        eta_back = tau + numpy.sum(
            numpy.sin(tau[:, None] * numpy.arange(1, len(Dm) + 1)[None, :]) * Dm,
            axis=1,
        )
        assert numpy.fabs(eta_back - 2.0)[0] < 1e-10
        # A non-finite map would be bracketed against NaN, so that raises
        with pytest.raises(RuntimeError) as excinfo:
            aac._tau_of_eta(numpy.array([2.0]), numpy.nan * Dm)
        assert "not finite" in str(excinfo.value)
        with pytest.raises(RuntimeError) as excinfo:
            aac._tau_of_eta(numpy.array([numpy.nan]), Dm)
        assert "not finite" in str(excinfo.value)
        # A folded map has no unique root, so that still raises
        folded = numpy.zeros_like(Dm)
        folded[0] = 1.0
        with pytest.raises(RuntimeError) as excinfo:
            aac._tau_of_eta(numpy.array([2.0]), folded)
        assert "map anomaly" in str(excinfo.value)
        assert "not monotone" in str(excinfo.value)
        with pytest.raises(RuntimeError) as excinfo:
            aac._canonical_torus_tables(0)
        assert "did not converge" in str(excinfo.value)
        with pytest.raises(RuntimeError) as excinfo:
            aac._xvFreqs_canonical(
                0, numpy.array([2.0]), numpy.array([1.0]), numpy.array([2.0])
            )
        assert "target angles" in str(excinfo.value)
    finally:
        aac._maxiter = maxiter
        aac._canonical_torus_tables(0)
    # an externally-inconsistent toy fails the correspondence informatively
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.potential import IsochronePotential

    aAIc = aac._aAIc
    try:
        aac._aAIc = actionAngleIsochrone(
            ip=IsochronePotential(amp=aac._GMc / 1000.0, b=aac._bc)
        )
        with pytest.raises(RuntimeError) as excinfo:
            aac._canonical_torus_tables(0)
        assert "unbound lifted samples" in str(excinfo.value)
    finally:
        aac._aAIc = aAIc
        aac._canonical_torus_tables(0)
    return None


def _staeckel_symplectic_defect(xvmapper, jr, jphi, jz, ar, ap, az, h=1e-6):
    # max |A^T Omega A - Omega| of the 6x6 Jacobian of
    # (theta, J) -> (q, p) with q = (R, z, phi), p = (v_R, v_z, R v_T)
    def xp(args):
        R, vR, vT, z, vz, phi = xvmapper(*args)[:6]
        return numpy.array([R[0], z[0], phi[0], vR[0], vz[0], R[0] * vT[0]])

    x0 = numpy.array([jr, jphi, jz, ar, ap, az], dtype="float")
    idx = [3, 4, 5, 0, 1, 2]
    A = numpy.empty((6, 6))
    for col, ii in enumerate(idx):
        xps = x0.copy()
        xps[ii] += h
        xms = x0.copy()
        xms[ii] -= h
        A[:, col] = (xp(xps) - xp(xms)) / (2.0 * h)
    Om = numpy.zeros((6, 6))
    Om[:3, 3:] = numpy.eye(3)
    Om[3:, :3] = -numpy.eye(3)
    return numpy.max(numpy.fabs(A.T @ Om @ A - Om))


def test_actionAngleStaeckelInverse_canonical_family_defect(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # the assembled interpolated (J, theta) -> (x, v) map is symplectic at
    # the finite-difference floor, calibrated by the analytic isochrone
    # inverse through the same harness -- for ANY stored tables (the
    # manifest test injects noise into all of them and re-checks)
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    ctl = actionAngleIsochroneInverse(ip=IsochronePotential(amp=aASI._GMc, b=aASI._bc))
    floor = max(
        _staeckel_symplectic_defect(ctl._xvFreqs, 0.1, 0.6, 0.1, ar, 1.0, 2.0)
        for ar in (0.7, 2.0, 4.5)
    )
    assert floor < 3e-8
    Lz = float(aASI._Lzgrid[5] + 0.3 * (aASI._Lzgrid[6] - aASI._Lzgrid[5]))
    for jr, jz in ((0.03, 0.05), (0.06, 0.10), (0.10, 0.03)):
        for ar in (0.7, 2.0, 4.5):
            defect = _staeckel_symplectic_defect(
                aASI._xvFreqs, jr, Lz, jz, ar, 1.0, 2.0
            )
            assert defect < 3e-8, (
                "The canonical Staeckel family's symplectic defect is not at "
                "the finite-difference floor: %g at (jr, jz, ar) = "
                "(%g, %g, %g)" % (defect, jr, jz, ar)
            )
    return None


def test_actionAngleStaeckelInverse_canonical_family_manifest():
    # canonicity is manifest: noise in every stored table moves the
    # evaluation but leaves the defect at the floor; a fresh instance,
    # because its tables get ruined
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    aASI = actionAngleStaeckelInverse(
        pot=kkp,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=6,
        nE=6,
        nI3=6,
    )
    Lz = float(aASI._Lzgrid[3])
    args = (0.06, Lz, 0.10, numpy.array([2.0]), numpy.array([1.0]), numpy.array([2.0]))
    x0 = numpy.array(aASI._xvFreqs(*args)[:6]).flatten()
    rng = numpy.random.default_rng(11)
    aASI._canon_tab_raw *= 1.0 + 1e-5 * rng.standard_normal(aASI._canon_tab_raw.shape)
    aASI._rebuild_canon_interp()
    x1 = numpy.array(aASI._xvFreqs(*args)[:6]).flatten()
    moved = numpy.max(numpy.fabs(x1 - x0))
    assert moved > 1e-7, (
        "The injected table noise did not reach the evaluation (moved %g)" % moved
    )
    defect = _staeckel_symplectic_defect(aASI._xvFreqs, 0.06, Lz, 0.10, 2.0, 1.0, 2.0)
    assert defect < 3e-8, (
        "The symplectic defect is not invariant under stored-table noise: %g" % defect
    )
    return None


def test_actionAngleStaeckelInverse_canonical_family_warning():
    # an under-resolved anomaly map is reported with actionable advice
    import warnings as warnings_module

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential
    from galpy.util import galpyWarning

    kkp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=3.0, Delta=1.25)
    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")
        actionAngleStaeckelInverse(
            pot=kkp,
            setup_interp=True,
            Rmin=0.5,
            Rmax=1.5,
            Rinf=6.0,
            nLz=4,
            nE=4,
            nI3=4,
            ncanon=128,
            npt=12,
        )
        assert any(
            issubclass(ww.category, galpyWarning)
            and "under-resolved" in str(ww.message)
            for ww in w
        ), "An under-resolved family does not warn with actionable advice"
    return None


def test_actionAngleStaeckelInverse_canonical_family_guards(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # the remaining defensive raises, fired for real: the toy-anomaly and
    # family-Newton non-convergence (white-box via maxiter/comp sabotage),
    # the interior label-match failure, and the above-grid energy
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    Lz = float(aASI._Lzgrid[5])
    maxiter = aASI._maxiter
    try:
        aASI._maxiter = 0
        with pytest.raises(RuntimeError) as excinfo:
            aASI._canon_toy_radial(0.06, 0.5, numpy.array([2.0]))
        assert "eccentric anomaly" in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            # an interior pair that the crippled label Newton cannot match:
            # inside the covered total-action band, so neither directional
            # diagnosis applies
            aASI._canon_coords(0.06, Lz, 0.10)
        assert "could not be matched" in str(excinfo.value)
    finally:
        aASI._maxiter = maxiter
    comp = aASI._canon_comp
    try:
        # A sabotage that under-relaxation CANNOT rescue.  The iteration's
        # multiplier is 1 - omega (1 + c'), so any c' > -1 converges for
        # small enough omega -- including c' = 2, which diverges undamped but
        # is recovered at omega = 1/2.  Only c' < -1 is hopeless: here
        # c' = -2 gives multiplier 1 + omega, above one for every omega.
        aASI._canon_comp = lambda tr, tz, jr, LA, Lzz, v, dq: (
            -2.0 * tr,
            -2.0 * tr,
            -2.0 * tz,
        )
        with pytest.raises(RuntimeError) as excinfo:
            aASI._xvFreqs_canonical_interp(
                0.06,
                Lz,
                0.10,
                numpy.array([2.0]),
                numpy.array([1.0]),
                numpy.array([2.0]),
            )
        assert "toy angles" in str(excinfo.value)
        # while c' = 2, which diverges at full step, is now recovered: this
        # is the limit-cycle/divergence fix acting on the real evaluation
        # path rather than on the isolated solver
        aASI._canon_comp = lambda tr, tz, jr, LA, Lzz, v, dq: (
            2.0 * tr,
            2.0 * tr,
            2.0 * tz,
        )
        out = aASI._xvFreqs_canonical_interp(
            0.06,
            Lz,
            0.10,
            numpy.array([2.0]),
            numpy.array([1.0]),
            numpy.array([2.0]),
        )
        assert numpy.all(numpy.isfinite(numpy.asarray(out[0], dtype=float))), (
            "The rescued solve did not produce a finite point"
        )
    finally:
        aASI._canon_comp = comp
    # E above the grid through the integrals entry point
    E, I3 = _canon_integral_labels(aASI, 0.06, Lz, 0.10)
    with pytest.raises(ValueError) as excinfo:
        aASI.Freqs(100.0, Lz, I3, integrals=True)
    assert "above the energies" in str(excinfo.value)
    # and the constructor's pot guard
    from galpy.actionAngle import actionAngleStaeckelInverse

    with pytest.raises(OSError) as excinfo:
        actionAngleStaeckelInverse()
    assert "Must specify pot=" in str(excinfo.value)
    return None


def test_actionAngleStaeckelInverse_canonical_node_derivatives():
    # the analytic d/dJ of every per-torus quantity the family stores --
    # the two u turning points, the v turning point, and both anomaly-map
    # degrees -- against finite differences of independently constructed
    # neighbouring tori. The toy is held fixed, as it is across the family
    # grid, so these are the derivatives the interpolation must reproduce
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    kk, aac, _ = _staeckel_canonical_setup()
    ii = 0
    p0 = [aac._Es[ii], aac._Lzs[ii], aac._I3s[ii]]

    def build(E, Lz, I3):
        # a single-torus instance forced onto the family's toy
        one = actionAngleStaeckelInverse(
            pot=kk,
            Es=[E],
            Lzs=[Lz],
            I3s=[I3],
            canonical=True,
            ncanon=aac._ncanon,
            npt=aac._npt,
        )
        one._GMc, one._bc = aac._GMc, aac._bc
        one._aAIc, one._aAIinvc = aac._aAIc, aac._aAIinvc
        one._canonical_torus_tables(0)
        return one

    dsupJ, dDmu, dDmv = aac._canon_node_dJ(ii)
    hs = [1e-5 * numpy.fabs(pp) for pp in p0]
    fsup = numpy.empty((3, 3))
    fu = numpy.empty((aac._npt, 3))
    fv = numpy.empty((aac._npt, 3))
    dJdalpha = numpy.empty((3, 3))
    for kk_ in range(3):
        pp, pm = list(p0), list(p0)
        pp[kk_] += hs[kk_]
        pm[kk_] -= hs[kk_]
        ap, am = build(*pp), build(*pm)
        fsup[:, kk_] = [
            (ap._umins[0] - am._umins[0]) / (2.0 * hs[kk_]),
            (ap._umaxs[0] - am._umaxs[0]) / (2.0 * hs[kk_]),
            (ap._vmins[0] - am._vmins[0]) / (2.0 * hs[kk_]),
        ]
        fu[:, kk_] = (ap._can_Dmu[0] - am._can_Dmu[0]) / (2.0 * hs[kk_])
        fv[:, kk_] = (ap._can_Dmv[0] - am._can_Dmv[0]) / (2.0 * hs[kk_])
        dJdalpha[:, kk_] = [
            (ap._jr[0] - am._jr[0]) / (2.0 * hs[kk_]),
            (pp[1] - pm[1]) / (2.0 * hs[kk_]),
            (ap._jz[0] - am._jz[0]) / (2.0 * hs[kk_]),
        ]
    # the finite differences are d/d(E, Lz, I3); convert to d/dJ
    dalphadJ = numpy.linalg.inv(dJdalpha)
    for name, ana, fd in (
        ("turning points", dsupJ, fsup @ dalphadJ),
        ("u-degree anomaly map", dDmu, fu @ dalphadJ),
        ("v-degree anomaly map", dDmv, fv @ dalphadJ),
    ):
        assert numpy.max(numpy.fabs(ana - fd)) < 1e-7, (
            "The analytic d/dJ of the %s disagrees with finite differences: "
            "%g" % (name, numpy.max(numpy.fabs(ana - fd)))
        )
    return None


def test_actionAngleStaeckelInverse_canonical_family_degenerate_accuracy(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # Tori with a small action sit where the corresponding oscillation's
    # half-width vanishes, and storing the turning points themselves loses
    # that half-width to cancellation: umax and umin agree to every digit
    # the grid resolves, so their difference is noise (50% wrong one cell
    # from the shell edge, and with it the angles). Storing the midpoint and
    # the squared half-width scaled by the action that drives it keeps these
    # tori as accurate as the interior ones; without it every case below is
    # an order of magnitude worse
    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    Lz = float(aASI._Lzgrid[5] + 0.3 * (aASI._Lzgrid[6] - aASI._Lzgrid[5]))
    angler = numpy.linspace(0.3, 6.0, 9)
    for jr, jz in (
        (0.002, 0.10),
        (0.0005, 0.10),
        (0.10, 0.002),
        (0.10, 0.0005),
        (0.005, 0.005),
    ):
        R, vR, vT, z, vz, phi = (
            numpy.atleast_1d(q)
            for q in aASI(jr, Lz, jz, angler, angler * 0.7, angler * 1.3)
        )
        oo = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
        dR = numpy.fabs(
            (numpy.atleast_1d(oo[6]) - angler + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        )
        dz = numpy.fabs(
            (numpy.atleast_1d(oo[8]) - angler * 1.3 + numpy.pi) % (2.0 * numpy.pi)
            - numpy.pi
        )
        assert numpy.max(numpy.maximum(dR, dz)) < 5e-3, (
            "The canonical family's angles degrade at the near-degenerate "
            "torus (J_R, J_z) = (%g, %g): %g"
            % (jr, jz, numpy.max(numpy.maximum(dR, dz)))
        )
        ji = aAS(R, vR, vT, z, vz, phi)
        assert numpy.max(numpy.fabs(ji[0] - jr)) < 2e-5
        assert numpy.max(numpy.fabs(ji[2] - jz)) < 2e-5
    return None


def test_actionAngleStaeckelInverse_toy_angle_limit_cycle():
    # The toy-angle solve is a damped Picard iteration, so its multiplier is
    # -c'. On a torus where c' reaches 1 it does not diverge -- it CYCLES with
    # period two, the residual flipping sign at constant amplitude, and the
    # step limiter never engages because it only caps steps above 0.5. This
    # was seen within ~5e-4 of the radial turning point during a long
    # integration. Under-relaxation turns the multiplier into 1 - 2 omega and
    # breaks the cycle.
    import numpy

    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    class _Cycler:
        # c(theta^A) = theta^A exactly, so theta^A + c = 2 theta^A and the
        # undamped update has multiplier -1: the worst case, not merely a
        # slow one
        _maxiter = 60
        _angle_tol = 1e-13

        def _canon_comp(self, thetaAr, thetaAz, jr, LA, Lz, v, dq):
            return thetaAr, numpy.zeros_like(thetaAr), thetaAz

    thR = numpy.array([0.7, 2.0])
    thz = numpy.array([1.3, 4.1])
    tAr, tAz, cphi = actionAngleStaeckelInverse._toy_angle_solve(
        _Cycler(), thR, thz, 0.02, 1.0, 0.9, None, None
    )

    # theta^A + theta^A = theta, and the residual is taken mod 2 pi, so any
    # branch of it is a solution
    def wrapped(a, b):
        return numpy.amax(numpy.fabs((a - b + numpy.pi) % (2.0 * numpy.pi) - numpy.pi))

    assert wrapped(2.0 * tAr, thR) < 1e-12, (
        "The radial toy angle did not escape the limit cycle"
    )
    assert wrapped(2.0 * tAz, thz) < 1e-12, (
        "The vertical toy angle did not escape the limit cycle"
    )

    # and a contracting problem is untouched: c' = 0 converges in one step
    class _Easy(_Cycler):
        def _canon_comp(self, thetaAr, thetaAz, jr, LA, Lz, v, dq):
            z = numpy.zeros_like(thetaAr)
            return z, z, z

    tAr, tAz, _ = actionAngleStaeckelInverse._toy_angle_solve(
        _Easy(), thR, thz, 0.02, 1.0, 0.9, None, None
    )
    assert wrapped(tAr, thR) < 1e-13, "A contracting solve was disturbed"


def test_actionAngleStaeckelInverse_canonical_label_inversion_vectorized(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # The label inversion is a two-dimensional Newton per torus, and at
    # these array sizes its Python-level call overhead dominates the
    # arithmetic, which makes an ensemble expensive. The vectorized form
    # solves all tori together and must agree with the scalar one exactly
    import numpy

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    rng = numpy.random.default_rng(42)
    n = 16
    Lz = rng.uniform(float(aASI._Lzgrid[2]), float(aASI._Lzgrid[-3]), n)
    jr = rng.uniform(0.02, 0.08, n)
    jz = rng.uniform(0.02, 0.08, n)
    ref = numpy.array(
        [numpy.atleast_2d(aASI._canon_coords(jr[i], Lz[i], jz[i]))[0] for i in range(n)]
    )
    got = aASI._canon_coords_vec(jr, Lz, jz)
    assert numpy.amax(numpy.fabs(got - ref)) < 1e-10, (
        "The vectorized label inversion disagrees with the scalar one: %g"
        % numpy.amax(numpy.fabs(got - ref))
    )
    # and it raises on an L_z outside the grid, as the scalar one does
    with pytest.raises(ValueError) as excinfo:
        aASI._canon_coords_vec(jr, Lz * 0.0 + 1e3, jz)
    assert "outside the grid" in str(excinfo.value)
    # non-convergence is reported with the offending torus, not silently
    maxiter = aASI._maxiter
    try:
        aASI._maxiter = 1
        with pytest.raises(ValueError) as excinfo:
            aASI._canon_coords_vec(jr, Lz, jz)
        assert "did not converge" in str(excinfo.value)
    finally:
        aASI._maxiter = maxiter
    return None


def test_actionAngleStaeckelInverse_canonical_chains_vectorized(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # The parameter chains for many tori at once, including the
    # reconstruction of the turning points from the stored midpoint-and-K
    # combinations, must agree with the scalar routine exactly
    import numpy

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    rng = numpy.random.default_rng(7)
    n = 12
    Lz = rng.uniform(float(aASI._Lzgrid[2]), float(aASI._Lzgrid[-3]), n)
    jr = rng.uniform(0.02, 0.08, n)
    jz = rng.uniform(0.02, 0.08, n)
    x = aASI._canon_coords_vec(jr, Lz, jz)
    v, dq = aASI._canon_family_chains_vec(x)
    for i in range(n):
        v1, dq1 = aASI._canon_family_chains(x[i : i + 1])
        assert numpy.amax(numpy.fabs(v[:, i] - v1)) < 1e-12, (
            "Vectorized family values disagree with the scalar ones"
        )
        assert numpy.amax(numpy.fabs(dq[:, i] - dq1)) < 1e-12, (
            "Vectorized family chains disagree with the scalar ones"
        )
    # J_R and J_z carry the identity rows of the label matrix by
    # construction, which is what lets the turning-point reconstruction use
    # the exact actions rather than interpolated ones
    assert numpy.amax(numpy.fabs(dq[0] - numpy.array([1.0, 0.0, 0.0]))) < 1e-8
    assert numpy.amax(numpy.fabs(dq[1] - numpy.array([0.0, 0.0, 1.0]))) < 1e-8
    return None


def test_actionAngleStaeckelInverse_xvFreqs_arrayJ(
    setup_actionAngleStaeckelInverse_interpolated,
):
    # The array-J evaluation -- per-point actions paired with per-point
    # angles in one vectorized pass -- must agree with the scalar path
    # looped over the same tori, for every output including the frequencies
    import numpy

    aASI, aAS, kkp = setup_actionAngleStaeckelInverse_interpolated
    rng = numpy.random.default_rng(3)
    n = 14
    Lz = rng.uniform(float(aASI._Lzgrid[2]), float(aASI._Lzgrid[-3]), n)
    jr = rng.uniform(0.02, 0.08, n)
    jz = rng.uniform(0.02, 0.08, n)
    thr = rng.uniform(0.0, 2.0 * numpy.pi, n)
    thp = rng.uniform(0.0, 2.0 * numpy.pi, n)
    thz = rng.uniform(0.0, 2.0 * numpy.pi, n)
    got = aASI._xvFreqs_arrayJ(jr, Lz, jz, thr, thp, thz)
    for i in range(n):
        ref = aASI._xvFreqs_canonical_interp(
            float(jr[i]),
            float(Lz[i]),
            float(jz[i]),
            numpy.array([thr[i]]),
            numpy.array([thp[i]]),
            numpy.array([thz[i]]),
        )
        for k in range(9):
            assert (
                numpy.fabs(numpy.atleast_1d(got[k])[i] - numpy.atleast_1d(ref[k])[0])
                < 1e-10
            ), "The array-J evaluation disagrees with the scalar path (output %i)" % k
    # the per-point anomaly-map inversion rescues unconverged points through
    # the scalar routine (which brackets when Newton cannot run), so the
    # array-J path survives maxiter = 0 on the map inversion alone; the
    # toy-angle iteration reports non-convergence instead of stalling
    x = aASI._canon_coords_vec(jr[:2], Lz[:2], jz[:2])
    v, dq = aASI._canon_family_chains_vec(x)
    npt = aASI._npt
    Dm = v[6 : 6 + npt]
    eta = numpy.array([2.0, 1.0])
    tau_ref = aASI._tau_of_eta_vec(eta, Dm)
    maxiter = aASI._maxiter
    try:
        aASI._maxiter = 0
        tau_rescued = aASI._tau_of_eta_vec(eta, Dm)
        assert numpy.amax(numpy.fabs(tau_rescued - tau_ref)) < 1e-8, (
            "The anomaly-map rescue disagrees with the converged inversion"
        )
        with pytest.raises(RuntimeError) as excinfo:
            aASI._toy_angle_solve_vec(
                thr[:2], thz[:2], jr[:2], jz[:2] + numpy.fabs(Lz[:2]), Lz[:2], v, dq
            )
        assert "did not converge" in str(excinfo.value)
    finally:
        aASI._maxiter = maxiter
    # a warm= dict carries the label coordinates and toy angles between
    # evaluations: the warm-started solve must agree with the cold one and
    # store its state for the next iterate
    warm = {}
    got_w0 = aASI._xvFreqs_arrayJ(jr, Lz, jz, thr, thp, thz, warm=warm)
    assert "x" in warm and "thetaA" in warm, (
        "the warm dict was not populated by the array-J evaluation"
    )
    got_w = aASI._xvFreqs_arrayJ(jr, Lz, jz, thr, thp, thz, warm=warm)
    for k in range(9):
        assert (
            numpy.amax(
                numpy.fabs(numpy.atleast_1d(got_w[k]) - numpy.atleast_1d(got[k]))
            )
            < 1e-10
        ), "warm-started evaluation disagrees with the cold one (output %i)" % k
    # the vectorized solve escapes the same period-two limit cycle as the
    # scalar one, through under-relaxation (see the scalar limit-cycle test)
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )

    class _CyclerV:
        _maxiter = 60
        _angle_tol = 1e-13

        def _canon_comp_vec(self, thetaAr, thetaAz, jr, LA, Lz, v, dq):
            return thetaAr, numpy.zeros_like(thetaAr), thetaAz

    thR2 = numpy.array([0.7, 2.0])
    thz2 = numpy.array([1.3, 4.1])
    tAr, tAz, _ = actionAngleStaeckelInverse._toy_angle_solve_vec(
        _CyclerV(), thR2, thz2, None, None, None, None, None
    )
    assert (
        numpy.amax(
            numpy.fabs((2.0 * tAr - thR2 + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        )
        < 1e-12
    ), "The vectorized toy-angle solve did not escape the limit cycle"
    return None


def test_actionAngleIsochroneInverse_kepler_bisection_fallback():
    # The batched Kepler solve falls back to per-point bisection for any
    # point Newton leaves unconverged; with Newton disabled entirely, the
    # whole solve runs through the fallback and must reproduce the normal
    # result (the bracket [0, 2 pi] always contains the root)
    import numpy

    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=1.2)
    aAII = actionAngleIsochroneInverse(ip=ip)
    jr = numpy.array([0.1, 0.02, 0.3])
    jphi = numpy.array([1.1, 0.9, 0.7])
    jz = numpy.array([0.05, 0.15, 0.02])
    ar = numpy.array([0.3, 2.9, 5.5])
    ap = numpy.array([1.0, 4.0, 0.2])
    az = numpy.array([2.0, 0.5, 3.3])
    ref = aAII._xvFreqs(jr, jphi, jz, ar, ap, az)
    try:
        aAII._kepler_maxiter = 0
        got = aAII._xvFreqs(jr, jphi, jz, ar, ap, az)
    finally:
        del aAII._kepler_maxiter
    for k in range(len(ref)):
        assert (
            numpy.amax(numpy.fabs(numpy.atleast_1d(got[k]) - numpy.atleast_1d(ref[k])))
            < 1e-8
        ), "The Kepler bisection fallback disagrees with Newton (output %i)" % k
    return None


def test_actionAngleStaeckelInverse_narrow_grid():
    # The grid is a box in (L_z, w_E, w_I). Spanning a sub-interval of each
    # axis localizes it on a target -- a stream, say -- and since the
    # interpolation error goes as the SPACING, a domain narrower by F is worth
    # as much as F times more nodes. Rmin/Rmax/Rinf set only the outer extent
    # and cannot express a narrow energy box, which is what this adds.
    import numpy

    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.4933)
    kw = dict(
        pot=swp, setup_interp=True, Rmin=0.75, Rmax=1.25, Rinf=1.5, nLz=5, nE=5, nI3=5
    )
    wide = actionAngleStaeckelInverse(**kw)
    narrow = actionAngleStaeckelInverse(
        Lzlim=(0.90, 0.94), wElim=(0.30, 0.40), wIlim=(0.40, 0.55), **kw
    )
    # the requested box is what gets built
    assert numpy.fabs(narrow._Lzgrid[0] - 0.90) < 1e-12, "L_z limit ignored"
    assert numpy.fabs(narrow._Lzgrid[-1] - 0.94) < 1e-12, "L_z limit ignored"
    assert numpy.fabs(narrow._wEgrid[0] - 0.30) < 1e-12, "w_E limit ignored"
    assert numpy.fabs(narrow._wIgrid[-1] - 0.55) < 1e-12, "w_I limit ignored"
    # and it is genuinely narrower in every axis
    for g in ("_Lzgrid", "_wEgrid", "_wIgrid"):
        assert numpy.ptp(getattr(narrow, g)) < numpy.ptp(getattr(wide, g)), (
            "%s was not narrowed" % g
        )
    # the default path is untouched: w_E still spans the padded unit interval
    # and w_I the whole of it, so the degenerate planar and shell edges are
    # still included there and excluded here
    assert wide._wIgrid[0] == 0.0 and wide._wIgrid[-1] == 1.0, (
        "the default w_I grid changed"
    )
    assert narrow._wIgrid[0] > 0.0 and narrow._wIgrid[-1] < 1.0, (
        "a narrow grid should not reach the degenerate edges"
    )
    # A None limit means that axis's own edge.  This matters because the
    # edges are degeneracies, not arbitrary boundaries: w_E = 0 is the
    # circular orbit, which the default pads away from, and w_I = 1 is the
    # shell orbit, whose handling tests for the grid REACHING it.  A box
    # meant to sit on one has to say so, not approach it with a number.
    edged = actionAngleStaeckelInverse(wElim=(None, 0.15), wIlim=(0.8, None), **kw)
    assert edged._wEgrid[0] == wide._wEgrid[0], (
        "None did not keep the padded circular edge"
    )
    assert numpy.fabs(edged._wEgrid[-1] - 0.15) < 1e-12, "the given w_E end moved"
    assert edged._wIgrid[-1] == 1.0, (
        "None did not land exactly on the shell edge, so its handling is skipped"
    )
    assert numpy.fabs(edged._wIgrid[0] - 0.8) < 1e-12, "the given w_I end moved"
    return None


def test_actionAngleStaeckelInverse_target_box():
    # target= localizes the grid on the tori of a set of phase-space points
    # without the user expressing the box in the grid's own (L_z, w_E, w_I)
    # coordinates: the constructor labels the points through the same
    # relations the node lattice uses and pads their range -- the local-grid
    # support for a target orbit or stream.
    import numpy

    from galpy.actionAngle import actionAngleStaeckel
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.4933)
    # a target orbit, sampled at a few times; its (E, L_z, I3) are conserved,
    # so all points share ONE torus and the box comes from target_minwidth
    o = Orbit([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    ts = numpy.linspace(0.0, 12.0, 8)
    o.integrate(ts, swp)
    kw = dict(
        pot=swp, setup_interp=True, Rmin=0.75, Rmax=1.25, Rinf=3.0, nLz=5, nE=5, nI3=5
    )
    wide = actionAngleStaeckelInverse(**kw)
    loc = actionAngleStaeckelInverse(target=o(ts), **kw)
    # the box is recorded and the grids honor it
    assert loc._targetbox is not None, "the target box was not recorded"
    for key, grid in (("Lzlim", "_Lzgrid"), ("wElim", "_wEgrid"), ("wIlim", "_wIgrid")):
        lo, hi = loc._targetbox[key]
        assert numpy.fabs(getattr(loc, grid)[0] - lo) < 1e-12, (
            "%s was not honored by the grid" % key
        )
        assert numpy.fabs(getattr(loc, grid)[-1] - hi) < 1e-12, (
            "%s was not honored by the grid" % key
        )
        assert numpy.ptp(getattr(loc, grid)) < numpy.ptp(getattr(wide, grid)), (
            "the target grid is not narrower in %s" % grid
        )
    # the target's torus sits INSIDE the local grid with room for the stencil
    # (labels from the forward transform with c=False: the Python angles are
    # exact along an orbit, while the C ones carry a converged ~2e-7 defect
    # for wrapper potentials)
    aAS = actionAngleStaeckel(pot=swp, delta=0.4933, c=False)
    R, vR, vT, z, vz = (
        numpy.atleast_1d(getattr(o(ts), name)(use_physical=False)).ravel()
        for name in ("R", "vR", "vT", "z", "vz")
    )
    dxw, dxl = [], []
    for ii in range(len(R)):
        jr, lz, jz, _, _, _, ar, ap, az = (
            float(numpy.atleast_1d(q)[0])
            for q in aAS.actionsFreqsAngles(R[ii], vR[ii], vT[ii], z[ii], vz[ii], 0.0)
        )
        x = loc._canon_coords(jr, lz, jz)
        assert numpy.min(numpy.array([numpy.min(x), 4.0 - numpy.max(x)])) > 1.0, (
            "a target point's torus sits within a cell of the box faces: %s" % x
        )
        for aa, dx in ((wide, dxw), (loc, dxl)):
            out = aa(
                jr,
                lz,
                jz,
                numpy.array([ar]),
                numpy.array([ap]),
                numpy.array([az]),
            )
            dx.append(
                numpy.hypot(
                    float(numpy.atleast_1d(out[0])[0]) - R[ii],
                    float(numpy.atleast_1d(out[3])[0]) - z[ii],
                )
            )
    # localization pays: the same 5^3 nodes reproduce the target orders of
    # magnitude better than the default box does (measured 7.7e-7 vs 2.7e-3)
    assert numpy.max(dxl) < 5e-6, (
        "the target-localized grid does not reproduce the target: %g" % numpy.max(dxl)
    )
    assert numpy.max(dxl) < numpy.max(dxw) / 100.0, (
        "the target-localized grid is not much better than the default box "
        "(%g vs %g)" % (numpy.max(dxl), numpy.max(dxw))
    )
    return None


def test_actionAngleStaeckelInverse_target_box_relations():
    # The box machinery itself, on a discrete instance (the label relations
    # exist before any grid does): padding, the minimum width, snapping to
    # the default edges, the circular-degeneracy guards, and the input
    # parser and its errors.
    import numpy
    import pytest
    from scipy import optimize

    from galpy.actionAngle.actionAngleStaeckelInverse import (
        _parse_target,
        actionAngleStaeckelInverse,
    )
    from galpy.orbit import Orbit
    from galpy.potential import (
        MWPotential2014,
        OblateStaeckelWrapperPotential,
        evaluatePotentials,
        rl,
        vcirc,
    )

    delta = 0.4933
    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=delta)
    # a valid discrete torus: circular energy plus a mid-range third integral
    Lz = 0.9
    Rc = rl(swp, Lz, use_physical=False)
    Ec = evaluatePotentials(swp, Rc, 0.0, use_physical=False) + Lz**2.0 / 2.0 / Rc**2.0
    Emax = (
        evaluatePotentials(swp, 3.0, 0.0, use_physical=False) + Lz**2.0 / 2.0 / 3.0**2.0
    )
    E = Ec + 0.4**2.0 * (Emax - Ec)
    Ipl = Lz**2.0 / 2.0 / delta**2.0 - E

    def maxWu(I3):
        return -optimize.minimize_scalar(
            lambda u: (
                -(
                    2.0 * delta**2.0 * (E * numpy.sinh(u) ** 2.0 - swp._U(u) - I3)
                    - Lz**2.0 / numpy.sinh(u) ** 2.0
                )
            ),
            bounds=(1e-3, 20.0),
            method="bounded",
            options={"xatol": 1e-13},
        ).fun

    Ish = optimize.brentq(maxWu, Ipl, Ipl + 5.0, xtol=1e-14)
    aA = actionAngleStaeckelInverse(
        pot=swp, Es=[E], Lzs=[Lz], I3s=[Ipl + 0.4 * (Ish - Ipl)]
    )
    # a spread-out target: a numeric box strictly inside the defaults
    aA._target = _parse_target(
        numpy.array(
            [
                [1.07, 0.13, 0.98, 0.13, 0.08],
                [1.0, 0.05, 1.02, 0.05, 0.02],
                [1.12, -0.1, 0.95, 0.2, -0.05],
            ]
        )
    )
    aA._target_pad, aA._target_minwidth = 1.5, 0.02
    box = aA._target_box(0.75, 1.25, 3.0, 0.02)
    for lim, w0, w1 in zip(box, (None, None, 0.0), (None, None, 1.0)):
        assert lim[0] is None or lim[1] is None or lim[0] < lim[1], (
            f"the box is not ordered: {box}"
        )
    assert box[2][0] is not None and box[2][1] is not None, (
        f"an interior w_I box should be numeric: {box}"
    )
    # huge padding: every degeneracy-bounded end snaps to its own edge
    # (None), while the L_z ends are anchors rather than degeneracies, so
    # the lower one is merely guarded against L_z <= 0
    aA._target_pad = 50.0
    box = aA._target_box(0.75, 1.25, 3.0, 0.02)
    assert box[1] == (None, None) and box[2] == (None, None), (
        f"a box beyond the default edges did not snap to them: {box}"
    )
    assert numpy.fabs(box[0][0] - 0.51) < 1e-12, (
        "the L_z lower end was not guarded at half the smallest target L_z"
    )
    assert box[0][1] > vcirc(swp, 1.25, use_physical=False) * 1.25, (
        "the L_z upper end should exceed its anchor (it is not a degeneracy)"
    )
    # a single point: zero spread on every axis, so the box is the minimum
    # width, centered on the point's torus
    aA._target = _parse_target([1.07, 0.13, 0.98, 0.13, 0.08])
    aA._target_pad = 1.5
    box = aA._target_box(0.75, 1.25, 3.0, 0.02)
    assert numpy.fabs(box[1][1] - box[1][0] - 0.02 * 0.96) < 1e-12, (
        f"the single-point w_E box is not the minimum width: {box}"
    )
    assert numpy.fabs(box[2][1] - box[2][0] - 0.02) < 1e-12, (
        f"the single-point w_I box is not the minimum width: {box}"
    )
    # the circular degeneracy: when the shell relation degenerates to the
    # planar one (or cannot bracket at all), every I3 label coincides and
    # the w_I direction is free, centered at 1/2
    aA._I3_shell = lambda E, Lz: aA._I3_planar(E, Lz)
    box = aA._target_box(0.75, 1.25, 3.0, 0.02)
    assert (
        numpy.fabs(box[2][0] - 0.49) < 1e-12 and numpy.fabs(box[2][1] - 0.51) < 1e-12
    ), f"a degenerate shell relation did not center the w_I box: {box}"

    def _nobracket(E, Lz):
        raise ValueError("f(a) and f(b) must have different signs")

    aA._I3_shell = _nobracket
    box = aA._target_box(0.75, 1.25, 3.0, 0.02)
    assert (
        numpy.fabs(box[2][0] - 0.49) < 1e-12 and numpy.fabs(box[2][1] - 0.51) < 1e-12
    ), f"an unbracketable shell relation did not center the w_I box: {box}"
    # the parser: an Orbit and its own rows are the same target, a sixth
    # (phi) column is ignored, and a bare point is one row
    vxvv = numpy.array(
        [[1.07, 0.13, 0.98, 0.13, 0.08, 0.3], [1.0, 0.05, 1.02, 0.05, 0.02, 5.2]]
    )
    assert numpy.array_equal(_parse_target(Orbit(vxvv)), _parse_target(vxvv)), (
        "an Orbit and its own rows parse differently"
    )
    assert numpy.array_equal(_parse_target(vxvv), _parse_target(vxvv[:, :5])), (
        "the phi column was not ignored"
    )
    assert _parse_target([1.07, 0.13, 0.98, 0.13, 0.08]).shape == (5, 1), (
        "a bare point is not one row"
    )
    with pytest.raises(ValueError, match="rows must be"):
        _parse_target([1.0, 0.1, 1.0, 0.1])
    with pytest.raises(ValueError, match="rows must be"):
        _parse_target(numpy.ones((2, 2, 5)))
    with pytest.raises(ValueError, match="R > 0"):
        _parse_target([-1.0, 0.1, 1.0, 0.1, 0.05])
    with pytest.raises(ValueError, match="R > 0"):
        _parse_target([1.0, 0.1, 0.0, 0.1, 0.05])
    # constructor guards: target= shapes the interpolation grid, so it
    # needs one, and it computes the box itself
    with pytest.raises(TypeError, match="requires setup_interp"):
        actionAngleStaeckelInverse(pot=swp, target=[1.0, 0.1, 1.0, 0.1, 0.05])
    with pytest.raises(TypeError, match="cannot be combined"):
        actionAngleStaeckelInverse(
            pot=swp,
            setup_interp=True,
            target=[1.0, 0.1, 1.0, 0.1, 0.05],
            wIlim=(0.3, 0.5),
        )
    # a target the grid cannot cover: energy above the outer energy
    with pytest.raises(ValueError, match="increase Rinf"):
        actionAngleStaeckelInverse(
            pot=swp,
            setup_interp=True,
            Rmin=0.75,
            Rmax=1.25,
            Rinf=1.3,
            target=[1.0, 0.9, 1.0, 0.3, 0.9],
        )
    return None


def test_actionAngleStaeckelInverse_target_box_adaptive():
    # target= composes with an adaptive family: the labelling then runs in
    # the LOCAL chart at each point's (E, L_z), the same smooth surfaces the
    # node lattice is built from.
    import numpy

    from galpy.actionAngle import actionAngleStaeckel
    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    o = Orbit([1.1, 0.15, 1.1, 0.1, 0.08, 0.0])
    ts = numpy.linspace(0.0, 10.0, 6)
    o.integrate(ts, kkp)
    loc = actionAngleStaeckelInverse(
        pot=kkp,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
        u0=lambda E, Lz: 1.05 + 0.10 * numpy.tanh(Lz),
        target=o(ts),
    )
    assert loc._targetbox is not None, "the target box was not recorded"
    # single delta: the family varies its reference curve, not its focal
    # length
    assert numpy.ptp(loc._canon_deltas) == 0.0, (
        "a u0-only family's focal length is not constant"
    )
    # the target box is NARROW (a single orbit), so the callable varies
    # only a little across it -- but it must vary
    assert (
        max(w._u0 for row_ in loc._canon_wraps for w in row_)
        - min(w._u0 for row_ in loc._canon_wraps for w in row_)
        > 1e-6
    ), "the u0-only family did not vary its reference curve"
    # a loose round trip: on an exactly Staeckel potential the wrapper at
    # the true focal length is exact for ANY u0, so only the coarse 4^3
    # grid limits this
    aAS = actionAngleStaeckel(pot=kkp, delta=1.3, c=False)
    jr, lz, jz, _, _, _, ar, ap, az = (
        float(numpy.atleast_1d(q)[0])
        for q in aAS.actionsFreqsAngles(1.1, 0.15, 1.1, 0.1, 0.08, 0.0)
    )
    out = loc(jr, lz, jz, numpy.array([ar]), numpy.array([ap]), numpy.array([az]))
    dx = numpy.hypot(
        float(numpy.atleast_1d(out[0])[0]) - 1.1,
        float(numpy.atleast_1d(out[3])[0]) - 0.1,
    )
    assert dx < 1e-3, (
        "the adaptive target-localized family does not reproduce the target "
        "point: %g" % dx
    )
    return None


def test_actionAngleStaeckelInverse_u0only_adaptive():
    # u0-only adaptation: fixed focal length, adaptive u0(E, L_z) reference
    # curve. Measured motivation on MWPotential2014: the |dPhi|-optimal
    # delta is nearly universal while u0 placement carries factors of a few
    # to ~30 of model quality; and u0 is construction-only GAUGE -- the
    # map's (R, z) <-> (u, v) uses delta alone -- so a fixed-delta family
    # needs no chart compensation at all: the stored delta row must come
    # out constant, and canonicity must hold with no new terms.
    import numpy
    import pytest

    from galpy.actionAngle.actionAngleStaeckelInverse import (
        _parse_target,
        actionAngleStaeckelInverse,
    )
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    aASI = actionAngleStaeckelInverse(
        pot=kkp,
        u0=lambda E, Lz: 1.05 + 0.10 * numpy.tanh(Lz),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    # the delta row is CONSTANT: u0 never enters the map, only the build
    row = 6 + 2 * aASI._npt
    assert numpy.ptp(aASI._canon_tab_raw[row]) == 0.0, (
        "a u0-only family's stored focal length is not constant"
    )
    # but the nodes genuinely differ in their reference curve
    u0s = [aASI._canon_wraps[a][b]._u0 for a in (0, 3) for b in (0, 3)]
    assert max(u0s) - min(u0s) > 0.01, (
        "the u0-only family did not vary its reference curve"
    )
    # exactly canonical with zero new compensation terms
    Lz = float(aASI._Lzgrid[2])
    defect = _staeckel_symplectic_defect(aASI._xvFreqs, 0.06, Lz, 0.10, 2.0, 1.0, 2.0)
    assert defect < 3e-8, "a u0-only family is not exactly canonical: %g" % defect
    # u0='fit' with a fixed focal length: the zero-velocity midpoint rule,
    # no |dPhi| survey involved
    aASIf = actionAngleStaeckelInverse(
        pot=kkp,
        u0="fit",
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=4,
        nE=4,
        nI3=4,
    )
    defect = _staeckel_symplectic_defect(aASIf._xvFreqs, 0.06, Lz, 0.10, 2.0, 1.0, 2.0)
    assert defect < 3e-8, "a u0='fit' family is not exactly canonical: %g" % defect
    # the midpoint rule is sane on the grid and falls back gracefully off it
    E22 = float(aASIf._canon_tab_raw[2][2, 2, 0])
    assert 0.3 < aASIf._u0_func(E22, Lz) < 3.0, "the midpoint u0 is not sane"
    assert numpy.isfinite(aASIf._u0_func(1e10, Lz)), (
        "the midpoint rule does not fall back for an unbracketable energy"
    )
    # the chart-local label relations run under u0-only adaptivity
    x = aASIf._canon_coords_integrals(E22, Lz, 0.1)
    assert numpy.all(numpy.isfinite(x)), (
        "chart-local integrals labels failed for a u0-only family"
    )
    # and so does the target box
    aASIf._target = _parse_target([1.1, 0.15, 1.1, 0.1, 0.08])
    aASIf._target_pad, aASIf._target_minwidth = 1.5, 0.02
    box = aASIf._target_box(0.7, 1.6, 8.0, 0.02)
    assert box[2][0] is None or 0.0 < box[2][0] < 1.0, (
        "the target box failed for a u0-only family"
    )
    # a WRAPPED potential with adaptive u0 must unwrap for its node charts:
    # rewrapping would Staeckelize the Staeckelization
    from galpy.potential import OblateStaeckelWrapperPotential

    wkk = OblateStaeckelWrapperPotential(pot=kkp, delta=1.3)
    aASIw = actionAngleStaeckelInverse(
        pot=wkk,
        u0=lambda E, Lz: 1.05 + 0.10 * numpy.tanh(Lz),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=3,
        nE=3,
        nI3=3,
    )
    assert aASIw._canon_wraps[0][0]._pot is kkp, (
        "the node charts wrap the wrapper instead of the raw potential"
    )
    # guards: adaptive u0 needs the interpolated family
    with pytest.raises(TypeError, match="requires setup_interp"):
        actionAngleStaeckelInverse(pot=kkp, u0=lambda E, Lz: 1.0, Es=[0.5])
    with pytest.raises(TypeError, match="requires setup_interp"):
        actionAngleStaeckelInverse(pot=kkp, u0="fit", Es=[0.5])
    return None


def test_actionAngleStaeckelInverse_u0only_rawpot():
    # the convenience spelling still works alongside an adaptive u0: a RAW
    # potential with scalar delta= and callable u0= builds its reference
    # wrapper at the DEFAULT u0 (each node gets its own), rather than
    # trying to call float() on the callable
    import numpy

    from galpy.actionAngle.actionAngleStaeckelInverse import (
        actionAngleStaeckelInverse,
    )
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aASI = actionAngleStaeckelInverse(
        pot=lp,
        delta=0.8,
        u0=lambda E, Lz: 1.1 + 0.05 * numpy.tanh(E),
        setup_interp=True,
        Rmin=0.8,
        Rmax=1.4,
        Rinf=5.0,
        nLz=3,
        nE=3,
        nI3=3,
    )
    assert numpy.fabs(aASI._staeckelwrap._u0 - numpy.arcsinh(1.0 / 0.8)) < 1e-12, (
        "the reference wrapper's u0 moved off its default"
    )
    assert (
        numpy.ptp([aASI._canon_wraps[a][b]._u0 for a in (0, 2) for b in (0, 2)]) > 0.0
    ), "the nodes did not get their own u0"
    return None


def test_actionAngleStaeckelInverse_adaptive_delta_canonical():
    # The focal length may vary across the family -- delta(E, L_z), the
    # adaptive-Staeckel design -- provided the compensation carries its
    # chain. Manifest canonicity then extends to the chart parameter: the
    # map must stay exactly symplectic for an ARBITRARY smooth delta table,
    # not merely for the constant one construction wrote.
    import numpy

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    aASI = actionAngleStaeckelInverse(
        pot=kkp,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=8.0,
        nLz=6,
        nE=6,
        nI3=6,
    )
    Lz = float(aASI._Lzgrid[3])
    args = (0.06, Lz, 0.10, numpy.array([2.0]), numpy.array([1.0]), numpy.array([2.0]))
    x0 = numpy.array(aASI._xvFreqs(*args)[:6]).flatten()
    # a LARGE smooth chart variation: 5% in delta across (L_z, E), constant
    # in I3 as the design prescribes
    row = 6 + 2 * aASI._npt
    nLz, nE, nI3 = aASI._canon_shape
    ii, jj = numpy.meshgrid(numpy.arange(nLz), numpy.arange(nE), indexing="ij")
    mod = 1.0 + 0.05 * numpy.sin(2.0 * ii / (nLz - 1) + 3.0 * jj / (nE - 1))
    aASI._canon_tab_raw[row] = aASI._delta * mod[:, :, None]
    aASI._rebuild_canon_interp()
    x1 = numpy.array(aASI._xvFreqs(*args)[:6]).flatten()
    moved = numpy.max(numpy.fabs(x1 - x0))
    assert moved > 1e-3, (
        "A 5%% delta variation did not reach the evaluation (moved %g)" % moved
    )
    defect = _staeckel_symplectic_defect(aASI._xvFreqs, 0.06, Lz, 0.10, 2.0, 1.0, 2.0)
    assert defect < 3e-8, (
        "The map is not symplectic under a varying focal length: %g" % defect
    )
    # and the delta-chain is load-bearing: severing it must break canonicity
    # by orders of magnitude, or the term above is decorative
    orig = aASI._canon_family_chains

    def severed(x):
        v, dq = orig(x)
        dq = dq.copy()
        dq[row] = 0.0
        return v, dq

    try:
        aASI._canon_family_chains = severed
        broken = _staeckel_symplectic_defect(
            aASI._xvFreqs, 0.06, Lz, 0.10, 2.0, 1.0, 2.0
        )
    finally:
        aASI._canon_family_chains = orig
    assert broken > 1e-2, (
        "Severing the delta-chain did not break canonicity (%g), so the "
        "compensation term is not what keeps the varying-delta map "
        "symplectic" % broken
    )
    return None


def test_actionAngleStaeckelInverse_single_delta():
    # The family uses a SINGLE focal length, like the forward
    # actionAngleStaeckel (which fixes delta and varies u0): the varying-
    # delta construction was measured to lose to a hand-tuned constant
    # (the global surface fit was the weak link) and was removed, while
    # the map keeps storing delta as a table row and carrying its chain,
    # so varying delta remains a documented possibility of the FORMAT.
    import numpy
    import pytest

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        MWPotential2014,
        OblateStaeckelWrapperPotential,
        evaluatePotentials,
        rl,
    )

    kkp = KuzminKutuzovStaeckelPotential(amp=4.0, ac=5.0, Delta=1.3)
    # every varying-delta spelling is rejected, discrete and interpolated
    # alike, with the pointer to what to use instead
    for bad in ("fit", lambda E, Lz: 1.3):
        with pytest.raises(TypeError, match="SINGLE delta"):
            actionAngleStaeckelInverse(pot=kkp, delta=bad, Es=[0.5])
        with pytest.raises(TypeError, match="SINGLE delta"):
            actionAngleStaeckelInverse(pot=kkp, delta=bad, setup_interp=True)
    # and a u0 string other than 'fit' stays rejected
    with pytest.raises(ValueError, match="only string value u0="):
        actionAngleStaeckelInverse(pot=kkp, u0="midplane", setup_interp=True)
    # the scalar convenience spelling survives unchanged
    with pytest.raises(TypeError) as excinfo:
        actionAngleStaeckelInverse(pot=MWPotential2014, u0=1.1, Es=[-1.2])
    assert "requires delta" in str(excinfo.value)
    Rc = rl(MWPotential2014, 0.9, use_physical=False)
    swp = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45, u0=1.1)
    Ec = evaluatePotentials(swp, Rc, 0.0, use_physical=False) + 0.9**2 / 2.0 / Rc**2
    E = Ec + 0.05 * numpy.fabs(Ec)
    I3 = 0.9**2 / (2.0 * 0.45**2) - E + 0.01
    scal = actionAngleStaeckelInverse(
        pot=MWPotential2014, delta=0.45, u0=1.1, Es=[E], Lzs=[0.9], I3s=[I3]
    )
    assert numpy.fabs(scal._delta - 0.45) < 1e-14, "scalar delta= not honoured"
    assert numpy.fabs(scal._staeckelwrap._u0 - 1.1) < 1e-14, "u0= not honoured"
    return None
