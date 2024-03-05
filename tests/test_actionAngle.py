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
    ), "Frequency returned by actionsFreqs not equal to that returned by actionsFreqsAngles"
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
    assert (
        numpy.fabs(aAH(-0.1, 0.1) - aAHnu(-0.1, 0.1) * ro * vo) < 10.0**-8.0
    ), "actionAngle function __call__ does not return Quantity with the right value for actionAngleHarmonic"
    # actionsFreqs
    assert (
        numpy.fabs(
            aAH.actionsFreqs(0.2, 0.1)[0] - aAHnu.actionsFreqs(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleHarmonic"
    assert (
        numpy.fabs(
            aAH.actionsFreqs(0.2, 0.1)[1]
            - aAHnu.actionsFreqs(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleHarmonic"
    # actionsFreqsAngles
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[0]
            - aAHnu.actionsFreqsAngles(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[1]
            - aAHnu.actionsFreqsAngles(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
    assert (
        numpy.fabs(
            aAH.actionsFreqsAngles(0.2, 0.1)[2] - aAHnu.actionsFreqsAngles(0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleHarmonic"
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
    ), "Frequency returned by actionsFreqs not equal to that returned by actionsFreqsAngles"
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
    assert (
        numpy.fabs(aAV(0.0, vesc + 1e-4) - 9999.99) < 10.0**-8.0
    ), "actionAngleVertical does not return J=9999.99 for unbound orbits"
    assert (
        numpy.fabs(aAV.actionsFreqs(0.0, vesc + 1e-4)[0] - 9999.99) < 10.0**-8.0
    ), "actionAngleVertical does not return J=9999.99 for unbound orbits"
    assert (
        numpy.fabs(aAV.actionsFreqs(0.0, vesc + 1e-4)[1] - 9999.99) < 10.0**-8.0
    ), "actionAngleVertical does not return O=9999.99 for unbound orbits"
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
    assert (
        numpy.fabs(aAV(-0.1, 0.1) - aAVnu(-0.1, 0.1) * ro * vo) < 10.0**-8.0
    ), "actionAngle function __call__ does not return Quantity with the right value for actionAngleVertical"
    # actionsFreqs
    assert (
        numpy.fabs(
            aAV.actionsFreqs(0.2, 0.1)[0] - aAVnu.actionsFreqs(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleVertical"
    assert (
        numpy.fabs(
            aAV.actionsFreqs(0.2, 0.1)[1]
            - aAVnu.actionsFreqs(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value for actionAngleVertical"
    # actionsFreqsAngles
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[0]
            - aAVnu.actionsFreqsAngles(0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[1]
            - aAVnu.actionsFreqsAngles(0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    assert (
        numpy.fabs(
            aAV.actionsFreqsAngles(0.2, 0.1)[2] - aAVnu.actionsFreqsAngles(0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value for actionAngleVertical"
    return None


# Basic sanity checking of the actionAngleIsochrone actions
def test_actionAngleIsochrone_basic_actions():
    from galpy.actionAngle import actionAngleIsochrone
    from galpy.orbit import Orbit

    aAI = actionAngleIsochrone(b=1.2)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAI(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the isochrone potential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the isochrone potential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAI(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the isochrone potential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-4.0
    ), "Close-to-circular orbit in the isochrone potential does not have small Jz"
    # Close-to-circular orbit, called with time
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAI(Orbit([R, vR, vT, z, vz]), 0.0)
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the isochrone potential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-4.0
    ), "Close-to-circular orbit in the isochrone potential does not have small Jz"
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
    assert (
        numpy.fabs(ecc - o.e()) < 1e-10
    ), "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(zmax - o.zmax()) < 1e-5
    ), "Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(rperi - o.rperi()) < 1e-10
    ), "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(rap - o.rap()) < 1e-10
    ), "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
    # Another one
    o = Orbit([1.0, 0.1, 1.1, 0.2, -0.3, 0.0])
    ecc, zmax, rperi, rap = aAI.EccZmaxRperiRap(
        o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()
    )
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, ip)
    assert (
        numpy.fabs(ecc - o.e()) < 1e-10
    ), "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(zmax - o.zmax()) < 1e-3
    ), "Analytically calculated zmax does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(rperi - o.rperi()) < 1e-10
    ), "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(rap - o.rap()) < 1e-10
    ), "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
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
    assert (
        numpy.fabs(ecc - o.e()) < 1e-10
    ), "Analytically calculated eccentricity does not agree with numerically calculated one for an IsochronePotential"
    # Don't do zmax, because zmax for Kepler is approximate
    assert (
        numpy.fabs(rperi - o.rperi()) < 1e-10
    ), "Analytically calculated rperi does not agree with numerically calculated one for an IsochronePotential"
    assert (
        numpy.fabs(rap - o.rap()) < 1e-10
    ), "Analytically calculated rap does not agree with numerically calculated one for an IsochronePotential"
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
    assert numpy.all(
        numpy.fabs(jrs - (jc - L)) < 10.0**-5.0
    ), "Radial action for the Kepler potential not correct"
    assert numpy.all(
        numpy.fabs(jps - obs.R() * obs.vT()) < 10.0**-10.0
    ), "Azimuthal action for the Kepler potential not correct"
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
    assert numpy.all(
        numpy.fabs(ors - oc) < 10.0**-10.0
    ), "Radial frequency for the Kepler potential not correct"
    assert numpy.all(
        numpy.fabs(ops - oc) < 10.0**-10.0
    ), "Azimuthal frequency for the Kepler potential not correct"
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
    assert numpy.all(
        numpy.fabs(ars - oc * times - ars[0]) < 10.0**-10.0
    ), "Radial angle for the Kepler potential not correct"
    assert numpy.all(
        numpy.fabs(aps - oc * times - aps[0]) < 10.0**-10.0
    ), "Azimuthal angle for the Kepler potential not correct"
    assert numpy.all(
        numpy.fabs(azs - oc * times - azs[0]) < 10.0**-10.0
    ), "Vertical angle for the Kepler potential not correct"
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
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the spherical LogarithmicHaloPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the spherical LogarithmicHaloPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-4.0
    ), "Close-to-circular orbit in the spherical LogarithmicHaloPotential does not have small Jz"
    return None


# Basic sanity checking of the actionAngleSpherical actions
def test_actionAngleSpherical_basic_freqs():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    aAS = actionAngleSpherical(pot=[lp])
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
    assert (
        numpy.fabs(ecc - o.e()) < 1e-9
    ), "Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(zmax - o.zmax()) < 1e-4
    ), "Analytically calculated zmax does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(rperi - o.rperi()) < 1e-8
    ), "Analytically calculated rperi does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(rap - o.rap()) < 1e-8
    ), "Analytically calculated rap does not agree with numerically calculated one for a spherical potential"
    # Another one
    o = Orbit([1.0, 0.1, 1.1, 0.2, -0.3, 0.0])
    ecc, zmax, rperi, rap = aAS.EccZmaxRperiRap(o.R(), o.vR(), o.vT(), o.z(), o.vz())
    ts = numpy.linspace(0.0, 100.0, 100001)
    o.integrate(ts, lp)
    assert (
        numpy.fabs(ecc - o.e()) < 1e-9
    ), "Analytically calculated eccentricity does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(zmax - o.zmax()) < 1e-3
    ), "Analytically calculated zmax does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(rperi - o.rperi()) < 1e-8
    ), "Analytically calculated rperi does not agree with numerically calculated one for a spherical potential"
    assert (
        numpy.fabs(rap - o.rap()) < 1e-8
    ), "Analytically calculated rap does not agree with numerically calculated one for a spherical potential"
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
    assert (
        numpy.fabs(o.wr(analytic=True, pot=ip, type="spherical")) < 10.0**-10.0
    ), "angler is not 0 at pericenter"
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
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.01, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_gamma0():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=[MWPotential[0], MWPotential[1:]], gamma=0.0)
    # circular orbit
    R, vR, vT, phi = 1.0, 0.0, 1.0, 2.0
    js = aAA(Orbit([R, vR, vT, phi]))
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.01, 0.0, 0.0
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
    return None


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_basic_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    # test nested list of potentials
    aAA = actionAngleAdiabatic(pot=[MWPotential[0], MWPotential[1:]], c=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"


# Basic sanity checking of the actionAngleAdiabatic actions
def test_actionAngleAdiabatic_unboundz_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, c=True, gamma=0.0)
    # Unbound in z, so jz should be very large
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 10.0
    js = aAA(R, vR, vT, z, vz)
    assert (
        js[2] > 1000.0
    ), "Unbound orbit in z in the MWPotential does not have large Jz"
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
    assert (
        numpy.fabs(js[0] - js2[0]) < 10.0**-6.0
    ), "Orbit with zero angular momentum does not have the correct Jr"
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, -0.5, 0.0, 0.0, 0.0
    js = aAA(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, -0.5, 0.0000001, 0.0, 0.0
    js2 = aAA(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0] - js2[0]) < 10.0**-6.0
    ), "Orbit with zero angular momentum does not have the correct Jr"
    return None


# Basic sanity checking of the actionAngleAdiabatic ecc, zmax, rperi, rap calc.
def test_actionAngleAdiabatic_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.potential import MWPotential

    aAA = actionAngleAdiabatic(pot=MWPotential, gamma=1.0)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz, phi = 1.01, 0.01, 1.0, 0.01, 0.01, 2.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz, phi)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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


# Test the actions of an actionAngleAdiabatic
def test_actionAngleAdiabatic_conserved_actions_c():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0])
    aAA = actionAngleAdiabatic(pot=MWPotential, c=True)
    check_actionAngle_conserved_actions(
        aAA, obs, MWPotential, -1.4, -8.0, -1.7, ntimes=101
    )
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
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(aAA.Jz(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # setup w/ multi
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, gamma=1.0, c=False, numcores=2)
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"
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
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"


# actionAngleAdiabaticGrid actions outside the grid
def test_actionAngleAdiabaticGrid_outsidegrid_c():
    from galpy.actionAngle import actionAngleAdiabatic, actionAngleAdiabaticGrid
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    aAA = actionAngleAdiabaticGrid(pot=MWPotential, c=True, Rmax=2.0, zmax=0.2)
    R, vR, vT, z, vz, phi = 3.0, 0.1, 1.0, 0.1, 0.1, 2.0
    js = aA(R, vR, vT, z, vz, phi)
    jsa = aAA(R, vR, vT, z, vz, phi)
    assert (
        numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0
    ), "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    assert (
        numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0
    ), "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    assert (
        numpy.fabs(js[2] - aAA.Jz(R, vR, vT, z, vz, phi)) < 10.0**-8.0
    ), "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    # Also for array
    s = numpy.ones(2)
    js = aA(R, vR, vT, z, vz, phi)
    jsa = aAA(R * s, vR * s, vT * s, z * s, vz * s, phi * s)
    assert numpy.all(
        numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0
    ), "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    assert numpy.all(
        numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0
    ), "actionAngleAdiabaticGrid evaluation outside of the grid fails"
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
    assert (
        numpy.fabs(js[0][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    # test nested list of potentials
    aAS = actionAngleStaeckel(
        pot=[MWPotential[0], MWPotential[1:]], delta=0.71, c=False, useu0=True
    )
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    return None


# Basic sanity checking of the actionAngleStaeckel actions
def test_actionAngleStaeckel_basic_actions_u0_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    # test nested list of potentials
    aAS = actionAngleStaeckel(
        pot=[MWPotential[0], MWPotential[1:]], delta=0.71, c=True, useu0=True
    )
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]), u0=1.15)
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
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
    assert (
        numpy.fabs(js[0][0]) < 10.0**-12.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2][0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
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
    assert (
        numpy.fabs(js[0]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAS(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 2.0 * 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_actions_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Unbound orbit, shouldn't fail
    R, vR, vT, z, vz = 1.0, 0.0, 10.0, 0.1, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert (
        js[0] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Jr"
    # Another unbound orbit, shouldn't fail
    R, vR, vT, z, vz = 1.0, 0.1, 10.0, 0.1, 0.0
    js = aAS(R, vR, vT, z, vz)
    assert (
        js[0] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Jr"
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
    assert (
        numpy.fabs(js[0] - js2[0]) < 10.0**-6.0
    ), "Orbit with zero angular momentum does not have the correct Jr"
    # Zero angular momentum, so rperi=0, but should have finite jr
    R, vR, vT, z, vz = 1.0, -0.5, 0.0, 0.0, 0.0
    js = aAS(R, vR, vT, z, vz)
    R, vR, vT, z, vz = 1.0, -0.5, 0.0000001, 0.0, 0.0
    js2 = aAS(R, vR, vT, z, vz)
    assert (
        numpy.fabs(js[0] - js2[0]) < 10.0**-6.0
    ), "Orbit with zero angular momentum does not have the correct Jr"
    return None


# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_actions_order():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._Delta, c=False)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt = aAS(o, order=10000, fixed_quad=True)
    jr1, jp1, jz1 = aAS(o, order=5, fixed_quad=True)
    jr2, jp2, jz2 = aAS(o, order=50, fixed_quad=True)
    assert (
        numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    return None


def test_actionAngleStaeckel_actions_order_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._Delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt = aAS(o, order=10000)
    jr1, jp1, jz1 = aAS(o, order=5)
    jr2, jp2, jz2 = aAS(o, order=50)
    assert (
        numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
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
    assert (
        js[0] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Jr"
    assert (
        js[3] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Or"
    assert (
        js[4] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Op"
    assert (
        js[5] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Oz"
    return None


# Check that precision increases with increasing Gauss-Legendre order
def test_actionAngleStaeckel_freqs_order_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import KuzminKutuzovStaeckelPotential

    kksp = KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4)
    o = Orbit([1.0, 0.5, 1.1, 0.2, -0.3, 0.4])
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._Delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt, ort, opt, ozt = aAS.actionsFreqs(o, order=10000)
    jr1, jp1, jz1, or1, op1, oz1 = aAS.actionsFreqs(o, order=5)
    jr2, jp2, jz2, or2, op2, oz2 = aAS.actionsFreqs(o, order=50)
    assert (
        numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(or1 - ort) > numpy.fabs(or2 - ort)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(op1 - opt) > numpy.fabs(op2 - opt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(oz1 - ozt) > numpy.fabs(oz2 - ozt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    return None


# Basic sanity checking of the actionAngleStaeckel actions, unbound
def test_actionAngleStaeckel_unboundr_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True)
    # Unbound orbit, shouldn't fail
    R, vR, vT, z, vz, phi = 1.0, 0.1, 10.0, 0.1, 0.0, 0.0
    js = aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    assert (
        js[0] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large Jr"
    assert (
        js[6] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large ar"
    assert (
        js[7] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large ap"
    assert (
        js[8] > 1000.0
    ), "Unbound in R orbit in the MWPotential does not have large az"
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
    aAS = actionAngleStaeckel(pot=kksp, delta=kksp._Delta, c=True)
    # We'll assume that order=10000 is the truth, so 50 should be better than 5
    jrt, jpt, jzt, ort, opt, ozt, art, apt, azt = aAS.actionsFreqsAngles(o, order=10000)
    jr1, jp1, jz1, or1, op1, oz1, ar1, ap1, az1 = aAS.actionsFreqsAngles(o, order=5)
    jr2, jp2, jz2, or2, op2, oz2, ar2, ap2, az2 = aAS.actionsFreqsAngles(o, order=50)
    assert (
        numpy.fabs(jr1 - jrt) > numpy.fabs(jr2 - jrt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(jz1 - jzt) > numpy.fabs(jz2 - jzt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(or1 - ort) > numpy.fabs(or2 - ort)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(op1 - opt) > numpy.fabs(op2 - opt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(oz1 - ozt) > numpy.fabs(oz2 - ozt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(ar1 - art) > numpy.fabs(ar2 - art)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(ap1 - apt) > numpy.fabs(ap2 - apt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    assert (
        numpy.fabs(az1 - azt) > numpy.fabs(az2 - azt)
    ), "Accuracy of actionAngleStaeckel does not increase with increasing order of integration"
    return None


# Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    return None


# Basic sanity checking of the actionAngleStaeckel ecc, zmax, rperi, rap calc.
def test_actionAngleStaeckel_basic_EccZmaxRperiRap_u0():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=False, useu0=True)
    # circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAS.EccZmaxRperiRap(R, vR, vT, z, vz, u0=1.15)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.fabs(jr0[0] - jri[0]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jr1[1] - jri[1]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz0[0] - jzi[0]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz1[1] - jzi[1]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    assert (
        numpy.fabs(nomed - indiv) < 1e-10
    ).all(), "no_median option returns different values to individual Delta estimation"
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
        assert (
            numpy.fabs(delta0 - deltasmall) < 1e-3
        ), "Delta computed with z=0 does not agree with that computed for small z"
    # And an array
    delta0 = estimateDeltaStaeckel(MWPotential2014, rs, numpy.zeros(n))
    deltasmall = estimateDeltaStaeckel(MWPotential2014, rs, 5e-4 * numpy.ones(n))
    assert numpy.all(
        numpy.fabs(delta0 - deltasmall) < 1e-3
    ), "Delta computed with array of z=0 does not agree with that computed for array of small z"


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
    assert (
        numpy.fabs(jr0[0] - jri[0]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jr1[1] - jri[1]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz0[0] - jzi[0]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz1[1] - jzi[1]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    assert (
        numpy.fabs(jr0[0] - jri[0]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jr1[1] - jri[1]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz0[0] - jzi[0]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz1[1] - jzi[1]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(or0[0] - ori[0]) < 1e-10
    ), "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(or1[1] - ori[1]) < 1e-10
    ), "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(op0[0] - opi[0]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(op1[1] - opi[1]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(oz0[0] - ozi[0]) < 1e-10
    ), "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(oz1[1] - ozi[1]) < 1e-10
    ), "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    assert (
        numpy.fabs(jr0[0] - jri[0]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jr1[1] - jri[1]) < 1e-10
    ), "Radial action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz0[0] - jzi[0]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(jz1[1] - jzi[1]) < 1e-10
    ), "Vertical action computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(or0[0] - ori[0]) < 1e-10
    ), "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(or1[1] - ori[1]) < 1e-10
    ), "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(op0[0] - opi[0]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(op1[1] - opi[1]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(oz0[0] - ozi[0]) < 1e-10
    ), "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(oz1[1] - ozi[1]) < 1e-10
    ), "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ar0[0] - ari[0]) < 1e-10
    ), "Radial frequencyaction computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ar1[1] - ari[1]) < 1e-10
    ), "Radial frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ap0[0] - api[0]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ap1[1] - api[1]) < 1e-10
    ), "Azimuthal computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(az0[0] - azi[0]) < 1e-10
    ), "Azimuthal frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(az1[1] - azi[1]) < 1e-10
    ), "Vertical frequency computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    assert (
        numpy.fabs(e0[0] - ei[0]) < 1e-10
    ), "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(e1[1] - ei[1]) < 1e-10
    ), "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(z0[0] - zi[0]) < 1e-10
    ), "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(z1[1] - zi[1]) < 1e-10
    ), "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(rp0[0] - rpi[0]) < 1e-10
    ), "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(rp1[1] - rpi[1]) < 1e-10
    ), "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ra0[0] - rai[0]) < 1e-10
    ), "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ra1[1] - rai[1]) < 1e-10
    ), "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    assert (
        numpy.fabs(e0[0] - ei[0]) < 1e-10
    ), "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(e1[1] - ei[1]) < 1e-10
    ), "Eccentricity computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(z0[0] - zi[0]) < 1e-10
    ), "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(z1[1] - zi[1]) < 1e-10
    ), "Zmax computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(rp0[0] - rpi[0]) < 1e-10
    ), "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(rp1[1] - rpi[1]) < 1e-10
    ), "Pericenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ra0[0] - rai[0]) < 1e-10
    ), "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
    assert (
        numpy.fabs(ra1[1] - rai[1]) < 1e-10
    ), "Apocenter computed with individual delta does not agree with that computed using the fixed orbit-wide default"
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
    check_actionAngle_conserved_actions(
        aAS, obs, MWPotential, -1.5, -8.0, -1.4, ntimes=101
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
        PerfectEllipsoidPotential,
        PowerTriaxialPotential,
        SCFPotential,
        TriaxialGaussianPotential,
        TriaxialHernquistPotential,
        TriaxialJaffePotential,
        TriaxialNFWPotential,
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
        SCFPotential(normalize=1.0),
        DiskSCFPotential(normalize=1.0),
        ip,
        PerfectEllipsoidPotential(normalize=1.0, c=0.98),
        TriaxialGaussianPotential(normalize=1.0, c=0.98),
        PowerTriaxialPotential(normalize=1.0, c=0.98),
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
    msmlpwtdp = (
        mockSmoothedLogarithmicHaloPotentialwTimeDependentAmplitudeWrapperPotential()
    )
    pots = [
        lp,
        lpb,
        hp,
        jp,
        np,
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
    check_actionAngle_conserved_EccZmaxRperiRap(
        aAS, obs, MWPotential, -1.8, -1.4, -1.8, -1.8, ntimes=101, inclphi=True
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
    assert (
        numpy.fabs(ezrpra[2]) < 1e-8
    ), "actionAngleStaeckel at very small u does not give rperi=0"
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
    assert (
        numpy.fabs(aAA.JR(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(aAA.Jz(R, vR, vT, z, vz, 0.0)) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jz"
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.fabs(js[0]) < 10.0**-8.0
    ), "Circular orbit in the MWPotential does not have Jr=0"
    assert (
        numpy.fabs(js[2]) < 10.0**-8.0
    ), "Circular orbit in the MWPotential does not have Jz=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    js = aAA(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(js[0]) < 10.0**-4.0
    ), "Close-to-circular orbit in the MWPotential does not have small Jr"
    assert (
        numpy.fabs(js[2]) < 10.0**-3.0
    ), "Close-to-circular orbit in the MWPotentialspherical LogarithmicHalo does not have small Jz"


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
    assert (
        numpy.fabs(te) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have e=0"
    assert (
        numpy.fabs(tzmax) < 10.0**-16.0
    ), "Circular orbit in the MWPotential does not have zmax=0"
    # Close-to-circular orbit
    R, vR, vT, z, vz = 1.01, 0.01, 1.0, 0.01, 0.01
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 0.99, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
    # Another close-to-circular orbit
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.01, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(Orbit([R, vR, vT, z, vz]))
    assert (
        numpy.fabs(te) < 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small eccentricity"
    assert (
        numpy.fabs(tzmax) < 2.0 * 10.0**-2.0
    ), "Close-to-circular orbit in the MWPotential does not have small zmax"
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
    assert (
        numpy.amax(numpy.fabs((acfs - acfsip) / acfs)) < 10.0**-15.0
    ), "actionAngleIsochroneApprox calculated w/ b= and ip= set to the equivalent IsochronePotential do not agree"
    assert (
        numpy.amax(numpy.fabs((acfs - acfsaAIip) / acfs)) < 10.0**-15.0
    ), "actionAngleIsochroneApprox calculated w/ b= and aAI= set to the equivalent IsochronePotential do not agree"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface"
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
    assert (
        maxdev < 10.0**-9.0
    ), "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface when using physical coordinates"
    assert (
        numpy.abs(
            obs.Tr(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[3]
        )
        < 10.0**-8.0
    ), "Orbit.Tr does not agree with actionAngleSpherical frequency when using physical coordinates"
    assert (
        numpy.abs(
            obs.Tp(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[4]
        )
        < 10.0**-8.0
    ), "Orbit.Tp does not agree with actionAngleSpherical frequency when using physical coordinates"
    assert (
        numpy.abs(
            obs.Tz(pot=lp, type=type) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[5]
        )
        < 10.0**-8.0
    ), "Orbit.Tz does not agree with actionAngleSpherical frequency when using physical coordinates"
    assert (
        numpy.abs(obs.TrTp(pot=lp, type=type) - acfs[4] / acfs[3] * numpy.pi)
        < 10.0**-8.0
    ), "Orbit.TrTp does not agree with actionAngleSpherical frequency when using physical coordinates"
    # Test frequency in km/s/kpc
    assert (
        numpy.abs(obs.Or(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[3])
        < 10.0**-8.0
    ), "Orbit.Or does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    assert (
        numpy.abs(obs.Op(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[4])
        < 10.0**-8.0
    ), "Orbit.Op does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    assert (
        numpy.abs(obs.Oz(pot=lp, type=type, kmskpc=True) / vo * ro - acfs[5])
        < 10.0**-8.0
    ), "Orbit.Oz does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface"
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
    assert (
        numpy.fabs(est_delta - obs._aA._delta) < 1e-10
    ), "Directly estimated delta does not agree with Orbit-interface-estimated delta"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface"
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
    assert (
        "second derivatives" in str(excinfo.value)
    ), "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp, type="staeckel")
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert (
        "not axisymmetric" in str(excinfo.value)
    ), "Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric"
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
    assert (
        "second derivatives" in str(excinfo.value)
    ), "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        obs.jr(pot=sp, type="staeckel")
        pytest.fail(
            "SpiralArms appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert (
        "not axisymmetric" in str(excinfo.value)
    ), "Estimating delta for a non-axi potential should have failed with a message about the fact that the potential is non-axisymmetric"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
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
    assert (
        maxdev < 10.0**-16.0
    ), "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface"
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
    assert (
        maxdev < 10.0**-13.0
    ), "Orbit interface for actionAngleIsochroneApprox does not return the same as actionAngle interface"
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
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
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
    assert (
        numpy.fabs(bmed - 1.2) < 10.0**-15.0
    ), "Estimated scale parameter b when estimateBIsochrone is applied to an IsochronePotential is wrong"
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
    assert (
        numpy.fabs(delta - 0.71) < 10.0**-3.0
    ), "Estimated focal parameter delta when estimateDeltaStaeckel is applied to the MWPotential is wrong"
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
    assert (
        numpy.fabs(delta) < 10.0**-6.0
    ), "Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong"
    delta = estimateDeltaStaeckel(lp, o.R(times), o.z(times), delta0=0.0)
    assert (
        numpy.fabs(delta) < 10.0**-16.0
    ), "Estimated focal parameter delta when estimateDeltaStaeckel is applied to a spherical potential is wrong"
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
        assert raisedWarning, "actionAngleAdiabatic with MWPotential should have thrown a warning, but didn't"
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
        assert raisedWarning, "actionAngleAdiabaticGrid with MWPotential should have thrown a warning, but didn't"
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
        assert raisedWarning, "actionAngleStaeckel with MWPotential should have thrown a warning, but didn't"
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
        assert raisedWarning, "actionAngleStaeckelGrid with MWPotential should have thrown a warning, but didn't"
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
        assert raisedWarning, "actionAngleIsochroneApprox with MWPotential should have thrown a warning, but didn't"
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
    assert (
        numpy.fabs(jr_good - jr_bad) < 1e-6
    ), f"Nearby JR for orbit near apocenter disagree too much, likely because one completely fails: Jr_good = {jr_good}, Jr_bad = {jr_bad}"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleHarmonicInverse is not the inverse of actionAngleHarmonic for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleHarmonicInverse is not the inverse of actionAngleHarmonic for an example orbit"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleHarmonicInverse does not agree with that computed by actionAngleHarmonic"
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
    assert numpy.all(
        numpy.fabs(orb.x(ts) - xv[0]) < 10.0**tol
    ), "Integrated orbit does not agree with actionAngleHarmmonicInverse orbit in x"
    assert numpy.all(
        numpy.fabs(orb.vx(ts) - xv[1]) < 10.0**tol
    ), "Integrated orbit does not agree with actionAngleHarmmonicInverse orbit in v"
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
        ), "actionAngleInverse function __call__ does not return Quantity with the right value"
    correct_fac = [ro, vo, conversion.freq_in_Gyr(vo, ro)]
    for ii in range(3):
        assert (
            numpy.fabs(
                aAHI.xvFreqs(0.1, -0.2)[ii]
                - aAHInu.xvFreqs(0.1, -0.2)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse function xvFreqs does not return Quantity with the right value"
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
    assert (
        numpy.fabs((om[0] - epifreq(ip, rl(ip, jphi))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - omegac(ip, rl(ip, jphi))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - verticalfreq(ip, rl(ip, jphi))) / om[2]) < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    # at Lz=1.5, w/ different potential normalization
    ip = IsochronePotential(normalize=1.2)
    aAII = actionAngleIsochroneInverse(ip=ip)
    jphi = 1.5
    om = aAII.Freqs(jr, jphi, jz)
    assert (
        numpy.fabs((om[0] - epifreq(ip, rl(ip, jphi))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - omegac(ip, rl(ip, jphi))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - verticalfreq(ip, rl(ip, jphi))) / om[2]) < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
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
    assert (
        numpy.fabs((Or - Ori) / Or) < 10.0**tol
    ), "Radial frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    assert (
        numpy.fabs((Op - Opi) / Op) < 10.0**tol
    ), "Azimuthal frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    assert (
        numpy.fabs((Oz - Ozi) / Oz) < 10.0**tol
    ), "Vertical frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    # at Lz=1.5
    jphi = 1.51
    Or, Op, Oz = aAII.Freqs(jr, jphi, jz)
    # Compute frequency with actionAngleIsochrone
    _, _, _, Ori, Opi, Ozi = aAI.actionsFreqs(*aAII(jr, jphi, jz, 0.0, 1.0, 2.0)[:6])
    assert (
        numpy.fabs((Or - Ori) / Or) < 10.0**tol
    ), "Radial frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    assert (
        numpy.fabs((Op - Opi) / Op) < 10.0**tol
    ), "Azimuthal frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
    assert (
        numpy.fabs((Oz - Ozi) / Oz) < 10.0**tol
    ), "Vertical frequency computed using actionAngleIsochroneInverse does not agree with that computed by actionAngleIsochrone"
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
        ), "actionAngleInverse function __call__ does not return Quantity with the right value"
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
        ), "actionAngleInverse function xvFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aAII.Freqs(0.1, 1.1, 0.1)[ii]
                - aAIInu.Freqs(0.1, 1.1, 0.1)[ii] * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngleInverse function Freqs does not return Quantity with the right value"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-10
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using a point transform"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using a point transform"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using a point transform"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-10
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using bisection"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-10
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection and a point transformation"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection and a point transformation"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using bisection and a point transformation"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-10
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection and a point transformation"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(1.0), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection and a point transformation"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection and a point transformation"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation"
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
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-10
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(Ei), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-8
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-8
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation"
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
    assert (
        numpy.amax(numpy.fabs(obs.x(times) - xi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and a point transformation"
    assert (
        numpy.amax(numpy.fabs(obs.vx(times) - vxi)) < 10.0**-6.0
    ), "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and a point transformation"
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
    tol = -8.0
    Om = aAVI.Freqs(aAVI.J(obs.E(pot=isopot)))
    # Compute frequency with actionAngleHarmonic
    _, Omi = aAV.actionsFreqs(*aAVI(aAVI.J(obs.E(pot=isopot)), 0.0))
    assert (
        numpy.fabs((Om - Omi) / Om) < 10.0**tol
    ), "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation and a point transformation"
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
    assert (
        numpy.std(E) / numpy.mean(E) < 1e-8
    ), "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation and a point transformation"
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / Om, 1001)
    x, v = aAVI(aAVI.J(Ei), Om * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert (
        numpy.amax(numpy.fabs(orb.x(ts) - x)) < 1e-7
    ), "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and a point transformation"
    assert (
        numpy.amax(numpy.fabs(orb.vx(ts) - v)) < 1e-7
    ), "Velocity does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and a point transformation"
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

    gs = aAVI.plot_convergence(1.0, return_gridspec=True)
    aAVIpt.plot_convergence(1.0, overplot=gs)
    pyplot.close()
    gs = aAVI.plot_power(0.1, return_gridspec=True)
    gs = aAVI.plot_power([0.1, 1.0, 10.0], overplot=gs)
    pyplot.close()
    aAVI.plot_orbit(1.0)
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
        assert raisedWarning, "actionAngleVerticalInverse for large energy should have raised convergence warning, but didn't"
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Torus mapping with bisection did not converge in 100 iterations for energies: 300"
            )
            if raisedWarning:
                break
        assert raisedWarning, "actionAngleVerticalInverse for large energy should have raised convergence warning, but didn't"
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
    assert (
        numpy.amax(numpy.fabs(obs.R(times) - Ri)) < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    assert (
        numpy.amax(
            numpy.fabs((obs.phi(times) - phii + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
        )
        < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.z(times) - zi)) < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.vR(times) - vRi)) < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.vT(times) - vTi)) < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    assert (
        numpy.amax(numpy.fabs(obs.vz(times) - vzi)) < 10.0**tol
    ), "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
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
    linfit = numpy.polyfit(times, ar, 1)
    assert numpy.fabs((linfit[1] - acfs_init[6]) / acfs_init[6]) < 10.0**tolinitar, (
        "Radial angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (100.0 * numpy.fabs((linfit[1] - acfs_init[6]) / acfs_init[6]))
    )
    assert numpy.fabs(linfit[0] - acfs_init[3]) < 10.0**tolor, (
        "Radial frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (100.0 * numpy.fabs((linfit[0] - acfs_init[3]) / acfs_init[3]))
    )
    devs = ar - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**toldar, (
        "Maximum deviation from linear trend in the radial angles is %g" % maxdev
    )
    # Do linear fit to azimuthal angle, check that deviations are small, check
    # that the slope is the frequency
    linfit = numpy.polyfit(times, ap, 1)
    assert numpy.fabs((linfit[1] - acfs_init[7]) / acfs_init[7]) < 10.0**tolinitap, (
        "Azimuthal angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (100.0 * numpy.fabs((linfit[1] - acfs_init[7]) / acfs_init[7]))
    )
    assert numpy.fabs(linfit[0] - acfs_init[4]) < 10.0**tolop, (
        "Azimuthal frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (100.0 * numpy.fabs((linfit[0] - acfs_init[4]) / acfs_init[4]))
    )
    devs = ap - linfit[0] * times - linfit[1]
    maxdev = numpy.amax(numpy.fabs(devs))
    assert maxdev < 10.0**toldap, (
        "Maximum deviation from linear trend in the azimuthal angle is %g" % maxdev
    )
    # Do linear fit to vertical angle, check that deviations are small, check
    # that the slope is the frequency
    linfit = numpy.polyfit(times, az, 1)
    assert numpy.fabs((linfit[1] - acfs_init[8]) / acfs_init[8]) < 10.0**tolinitaz, (
        "Vertical angle obtained by fitting linear trend to the orbit does not agree with the initially-calculated angle by %g%%"
        % (100.0 * numpy.fabs((linfit[1] - acfs_init[8]) / acfs_init[8]))
    )
    assert numpy.fabs(linfit[0] - acfs_init[5]) < 10.0**toloz, (
        "Vertical frequency obtained by fitting linear trend to the orbit does not agree with the initially-calculated frequency by %g%%"
        % (100.0 * numpy.fabs((linfit[0] - acfs_init[5]) / acfs_init[5]))
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
