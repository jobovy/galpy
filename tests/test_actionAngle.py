import sys
import warnings

import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array
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
    js = as_numpy(aAH(obs.x(times), obs.vx(times)))
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
    acfs_init = tuple(
        as_numpy(a) for a in aAH.actionsFreqsAngles(obs.x(), obs.vx())
    )  # to check the init. angles
    acfs = tuple(
        as_numpy(a) for a in aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))
    )
    angle = as_numpy(dePeriod(numpy.reshape(acfs[2], (1, len(times))))).flatten()
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
    assert numpy.all(
        numpy.fabs(
            as_numpy(aAH.actionsFreqs(obs.x(times), obs.vx(times))[1])
            - as_numpy(aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))[1])
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
    # Under a forced jax/torch backend obs.x()/vx() are backend arrays, so aAV
    # returns one too; the numpy reductions below need numpy (detach grad).
    js = as_numpy(aAV(obs.x(times), obs.vx(times)))
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
    js = as_numpy(js)  # backend array -> numpy for the reductions below
    os = as_numpy(os)
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
    angle = as_numpy(dePeriod(numpy.reshape(acfs[2], (1, len(times))))).flatten()
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
    assert numpy.all(
        numpy.fabs(
            as_numpy(aAV.actionsFreqs(obs.x(times), obs.vx(times))[1])
            - as_numpy(aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))[1])
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
    from galpy.backend import use
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
    # setup, on the host: a test-local potential has only the python
    # integrators, which cannot read a forced backend's CUDA forces
    with use("numpy", force=True):
        obs.integrate(times, ipz)
    js = as_numpy(aAH(obs.x(times), obs.vx(times)))
    jsv = as_numpy(aAV(obs.x(times), obs.vx(times)))
    maxdj = numpy.amax(numpy.fabs((js - jsv) / js))
    assert maxdj < 10.0**-10.0, (
        "Actions of harmonic oscillator computed using actionAngleVertical do not agree with those computed using actionAngleHarmonic at %g%%"
        % (100.0 * maxdj)
    )
    return None


def test_actionAngleVertical_Harmonic_actionsFreqs():
    from galpy.actionAngle import actionAngleHarmonic, actionAngleVertical
    from galpy.backend import use
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
    # setup, on the host: a test-local potential has only the python
    # integrators, which cannot read a forced backend's CUDA forces
    with use("numpy", force=True):
        obs.integrate(times, ipz)
    js, os = (as_numpy(a) for a in aAH.actionsFreqs(obs.x(times), obs.vx(times)))
    jsv, osv = (as_numpy(a) for a in aAV.actionsFreqs(obs.x(times), obs.vx(times)))
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
    from galpy.backend import use
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
    # setup, on the host: a test-local potential has only the python
    # integrators, which cannot read a forced backend's CUDA forces
    with use("numpy", force=True):
        obs.integrate(times, ipz)
    js, os, anss = (
        as_numpy(a) for a in aAH.actionsFreqsAngles(obs.x(times), obs.vx(times))
    )
    jsv, osv, anssv = (
        as_numpy(a) for a in aAV.actionsFreqsAngles(obs.x(times), obs.vx(times))
    )
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


def test_physical_1d_call_container_matches_natural():
    # A 1D __call__ returns ONE quantity -- the action J -- not a sequence of
    # them, so turning physical output on must not change the container. It used
    # to: physical_conversion_actionAngle scaled element-by-element against a
    # three-long factor list, which silently tuple-ified the result below length
    # four and raised IndexError at or above it.
    from galpy.actionAngle import actionAngleHarmonic, actionAngleVertical
    from galpy.potential import IsothermalDiskPotential

    ro, vo = 7.0, 230.0
    pairs = (
        (
            actionAngleVertical(
                pot=IsothermalDiskPotential(amp=1.0, sigma=0.5), ro=ro, vo=vo
            ),
            actionAngleVertical(pot=IsothermalDiskPotential(amp=1.0, sigma=0.5)),
        ),
        (
            actionAngleHarmonic(omega=1.1, ro=ro, vo=vo),
            actionAngleHarmonic(omega=1.1),
        ),
    )
    for aA, aAnu in pairs:
        name = aA.__class__.__name__
        # 4 and 9 are past the old three-element ceiling, 1-3 below it
        for n in (1, 2, 3, 4, 9):
            x = numpy.linspace(-0.15, 0.2, n)
            vx = numpy.linspace(0.05, 0.2, n)
            phys = aA(x, vx)
            nat = as_numpy(aAnu(x, vx))
            assert not isinstance(phys, tuple), (
                f"{name}: physical __call__ returned a tuple for n={n}, but the "
                "natural call returns a single array of actions"
            )
            phys = as_numpy(phys)
            assert phys.shape == nat.shape, (
                f"{name}: physical __call__ gave shape {phys.shape} for n={n}, "
                f"natural gave {nat.shape}"
            )
            # The decorator's only arithmetic is one multiply by ro*vo, so this
            # holds exactly -- no tolerance needed or wanted.
            assert numpy.all(phys == nat * (ro * vo)), (
                f"{name}: physical __call__ for n={n} is not the natural value "
                f"times ro*vo\n  physical: {phys!r}\n  natural*ro*vo: "
                f"{nat * (ro * vo)!r}"
            )
    return None


def test_physical_3d_call_stays_a_tuple():
    # The counterpart to the test above: 3D methods genuinely return several
    # separate quantities, and that must keep being a tuple whose entries are
    # scaled by their own factors. Pins the discriminator in
    # physical_conversion_actionAngle from being "simplified" back to len().
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, ro=ro, vo=vo)
    aAnu = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    o = (1.0, 0.1, 1.1, 0.1, 0.03, 0.2)
    for meth, nout in (("__call__", 3), ("actionsFreqs", 6), ("EccZmaxRperiRap", 4)):
        phys = getattr(aA, meth)(*o[:5])
        assert isinstance(phys, tuple) and len(phys) == nout, (
            f"actionAngleStaeckel.{meth} physical output should be a {nout}-tuple, "
            f"got {type(phys).__name__} of length {numpy.size(phys)}"
        )
    # and the per-entry factors still differ: actions scale by ro*vo, the
    # frequencies by freq_in_Gyr, which a single shared factor would break
    aFphys = aA.actionsFreqs(*o[:5])
    aFnat = aAnu.actionsFreqs(*o[:5])
    for ii in range(3):
        assert numpy.all(as_numpy(aFphys[ii]) == as_numpy(aFnat[ii]) * (ro * vo)), (
            f"actionsFreqs entry {ii} (an action) is not scaled by ro*vo"
        )
    freqfac = conversion.freq_in_Gyr(vo, ro)
    for ii in range(3, 6):
        assert numpy.all(as_numpy(aFphys[ii]) == as_numpy(aFnat[ii]) * freqfac), (
            f"actionsFreqs entry {ii} (a frequency) is not scaled by freq_in_Gyr"
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
    jrs, jps, jzs = (
        as_numpy(a)
        for a in aAI(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
        )
    )
    jc = as_numpy(ip._amp) / numpy.sqrt(-2.0 * as_numpy(obs.E()))
    L = numpy.sqrt(numpy.sum(as_numpy(obs.L()) ** 2.0))
    # Jr = Jc-L
    assert numpy.all(numpy.fabs(jrs - (jc - L)) < 10.0**-5.0), (
        "Radial action for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(jps - as_numpy(obs.R() * obs.vT())) < 10.0**-10.0), (
        "Azimuthal action for the Kepler potential not correct"
    )
    assert numpy.all(
        numpy.fabs(jzs - (L - numpy.fabs(as_numpy(obs.R() * obs.vT())))) < 10.0**-10.0
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
    _, _, _, ors, ops, ozs = (
        as_numpy(a)
        for a in aAI.actionsFreqs(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
        )
    )
    jc = as_numpy(ip._amp) / numpy.sqrt(-2.0 * as_numpy(obs.E()))
    oc = as_numpy(ip._amp) ** 2.0 / jc**3.0  # (BT08 eqn. E4)
    assert numpy.all(numpy.fabs(ors - oc) < 10.0**-10.0), (
        "Radial frequency for the Kepler potential not correct"
    )
    assert numpy.all(numpy.fabs(ops - oc) < 10.0**-10.0), (
        "Azimuthal frequency for the Kepler potential not correct"
    )
    assert numpy.all(
        numpy.fabs(ozs - numpy.sign(as_numpy(obs.R() * obs.vT())) * oc) < 10.0**-10.0
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
    _, _, _, _, _, _, ars, aps, azs = (
        as_numpy(a)
        for a in aAI.actionsFreqsAngles(
            obs.R(times),
            obs.vR(times),
            obs.vT(times),
            obs.z(times),
            obs.vz(times),
            obs.phi(times),
        )
    )
    jc = as_numpy(ip._amp) / numpy.sqrt(-2.0 * as_numpy(obs.E()))
    oc = as_numpy(ip._amp) ** 2.0 / jc**3.0  # (BT08 eqn. E4)
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

    from galpy.backend import get_namespace

    class ForceOnlyHernquist(Potential):  # namespace ops: runs on the backends
        def _evaluate(self, R, z, phi=0.0, t=0.0):
            return -0.5 / (1.0 + get_namespace(R, z).sqrt(R**2.0 + z**2.0))

        def _Rforce(self, R, z, phi=0.0, t=0.0):
            r = get_namespace(R, z).sqrt(R**2.0 + z**2.0)
            return -0.5 * R / r / (1.0 + r) ** 2

        def _zforce(self, R, z, phi=0.0, t=0.0):
            r = get_namespace(R, z).sqrt(R**2.0 + z**2.0)
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
        0.0,
        3.0 * 2.0 * numpy.pi / float(as_numpy(epifreq(hp, rc, use_physical=False))),
        61,
    )
    for wrc, tolJ, tolO, tolA in ((1e-3, 1e-6, 1e-7, 1e-6), (3e-6, 1e-3, 1e-7, 3e-5)):
        r, vr, vt = _near_circular_point(hp, rc, wrc, 0.3)
        o = Orbit([r, vr, vt, 0.0, 0.0, 0.7])
        o.integrate(ts, hp, method="dop853_c", rtol=1e-14, atol=1e-14)
        f = [
            as_numpy(x)
            for x in aAS.actionsFreqsAngles(
                o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts)
            )
        ]
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
        E = float(as_numpy(o.E(pot=pot)))
        L = float(numpy.sqrt(numpy.sum(as_numpy(o.L()) ** 2.0)))
        assert E < 0.0, "The test orbit is not bound"
        jr, jphi, jz, Or, Op, Oz, ar, ap, az = (
            as_numpy(x)
            for x in aAS.actionsFreqsAngles(
                o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()
            )
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
        j = [
            as_numpy(x)
            for x in aAS.actionsFreqsAngles(
                o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts)
            )
        ]
        assert numpy.ptp(j[0]) / numpy.mean(j[0]) < 1e-9, (
            "The radial action is not conserved along a bound orbit with a far apocentre"
        )
        assert numpy.ptp(j[2]) / numpy.mean(j[2]) < 1e-9, (
            "The vertical action is not conserved along a bound orbit with a far apocentre"
        )
        # an unbound orbit is still recognized as such
        ou = Orbit([1.0, 0.3, 3.0, 0.5, 0.1, 0.0])
        assert ou.E(pot=pot) > 0.0
        from galpy.backend import get_namespace

        if get_namespace(numpy.zeros(1)) is numpy:
            with pytest.raises(UnboundError):
                aAS.actionsFreqsAngles(
                    ou.R(), ou.vR(), ou.vT(), ou.z(), ou.vz(), ou.phi()
                )
        else:  # the backends return NaN instead (traceable)
            fu = aAS.actionsFreqsAngles(
                ou.R(), ou.vR(), ou.vT(), ou.z(), ou.vz(), ou.phi()
            )
            assert all(numpy.all(numpy.isnan(as_numpy(x))) for x in fu[3:])
            assert numpy.isnan(as_numpy(fu[0])[0])
        return None

    check(HernquistPotential(normalize=1.0, a=0.5))
    from galpy.backend import get_namespace

    if get_namespace(numpy.zeros(1)) is not numpy:
        # numpy's apocentre search asks the potential at infinity; the backend
        # bracket never does (and these Python branches on R do not trace)
        return None

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
    vc = float(as_numpy(vcirc(ip, r, use_physical=False)))
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
    assert (
        numpy.fabs(as_numpy(o.wr(analytic=True, pot=ip, type="spherical")))
        < 10.0**-10.0
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

    # test a nested combined potential
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
    from galpy.backend import name_of_namespace, resolve_namespace
    from galpy.orbit import Orbit
    from galpy.potential import CylindricallySeparablePotentialWrapper, MWPotential2014

    pot = CylindricallySeparablePotentialWrapper(pot=MWPotential2014, Rp=1.1)
    aAA = actionAngleAdiabatic(pot=pot, c=False, gamma=0.0)
    obs = Orbit([1.05, 0.02, 1.05, 0.03, 0.0, 0.0])
    # The three linear-trend bars are backend-aware. numpy keeps -7.0 exactly;
    # jax/torch reassociate the accumulated angle sum differently and land just
    # over it -- measured 1.67707e-07 (jax) and 1.77912e-07 (torch) against
    # 1e-7, i.e. a factor 1.7-1.8, not a wrong answer. -6.5 (3.16e-7) clears
    # both with ~1.8x margin and still fails anything genuinely broken.
    lin = -7.0 if name_of_namespace(resolve_namespace()) == "numpy" else -6.5
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
        lin,
        lin,
        lin,
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
        # as_numpy: under a forced backend these come back as tensors and
        # numpy.all/numpy.fabs on a Tensor raises.
        js = [as_numpy(a) for a in aA(R, 0.1 * o, 1.0 * o, 0.1 * o, 0.1 * o)]
        jsa = [as_numpy(a) for a in aAA(R, 0.1 * o, 1.0 * o, 0.1 * o, 0.1 * o)]
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
    # as_numpy at the assertion sites: under a forced backend these are
    # tensors, and numpy.fabs/numpy.all on a Tensor raises.
    js = [as_numpy(a) for a in aA(R, vR, vT, z, vz, phi)]
    jsa = [as_numpy(a) for a in aAA(R, vR, vT, z, vz, phi)]
    assert numpy.fabs(js[0] - jsa[0]) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    assert numpy.fabs(js[2] - jsa[2]) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    assert numpy.fabs(js[2] - as_numpy(aAA.Jz(R, vR, vT, z, vz, phi))) < 10.0**-8.0, (
        "actionAngleAdiabaticGrid evaluation outside of the grid fails"
    )
    # Also for array
    s = numpy.ones(2)
    js = [as_numpy(a) for a in aA(R, vR, vT, z, vz, phi)]
    jsa = [as_numpy(a) for a in aAA(R * s, vR * s, vT * s, z * s, vz * s, phi * s)]
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
@pytest.mark.backend_managed  # numpy-only: the Single (scipy.quad) is not backend-capable
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

    # test a nested combined potential
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

    # test a nested combined potential
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


# Test that the C frequencies and angles agree with the Python ones along an
# orbit that stays very close to the plane: the momentum function of the v
# libration is then small everywhere, which C took for a point at the
# vertical turning point
def test_actionAngleStaeckel_smalljz_freqs_angles_c():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([1.1, 0.14907, 1.0679, 5.8111e-4, -7.0220e-5, 2.5817])
    ts = numpy.linspace(0.0, 20.0, 401)
    o.integrate(ts, MWPotential2014)
    out = [
        numpy.array(
            actionAngleStaeckel(
                pot=MWPotential2014, delta=0.45, c=c
            ).actionsFreqsAngles(
                o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts), o.phi(ts)
            )
        )
        for c in (True, False)
    ]
    diff = numpy.fabs(out[0] - out[1])
    diff[6:] = numpy.fabs((diff[6:] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)
    assert numpy.all(diff[:6] < 5e-3) and numpy.all(diff[6:] < 5e-3), (
        "C and Python actionAngleStaeckel frequencies and angles differ along "
        "an orbit very close to the plane"
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


def test_actionAngleStaeckel_python_scalar_only_potential():
    # A potential whose forces take scalar coordinates only is evaluated point
    # by point on the vectorized pure-Python (numpy) path: its actions,
    # frequencies, and angles agree with C's; it has no backend evaluation
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.backend import use
    from galpy.potential import DoubleExponentialDiskPotential, HernquistPotential

    xv = (
        numpy.array([1.0, 0.9]),
        numpy.array([0.05, -0.1]),
        numpy.array([0.95, 1.05]),
        numpy.array([0.05, -0.02]),
        numpy.array([0.03, 0.06]),
        numpy.array([0.3, 2.0]),
    )
    # The potential is set up on numpy too: normalize= under a forced backend
    # makes its amplitude a backend array
    with use("numpy", force=True):
        pot = DoubleExponentialDiskPotential(
            normalize=0.6, hr=1.0 / 3.0, hz=1.0 / 27.0
        ) + HernquistPotential(normalize=0.4, a=2.0)
        aAp = actionAngleStaeckel(pot=pot, delta=0.3, c=False)
        aAc = actionAngleStaeckel(pot=pot, delta=0.3, c=True)
        for p, c in zip(aAp.actionsFreqsAngles(*xv), aAc.actionsFreqsAngles(*xv)):
            numpy.testing.assert_allclose(p, c, rtol=1e-8, atol=1e-10)
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
    # quadrature rule: a few 1e-11 for generic orbits, and looser for a
    # nearly planar orbit whose tiny v oscillation has S far below the
    # cancellation scale (the old fixed-order rule erred at 4.6e-4 here).
    # That cancellation scale is platform-dependent, so the generic bars sit at
    # 1e-10 rather than at the ~2e-11 measured here: torch on a CI runner has
    # produced 3.5e-11 on the same inputs (numpy 1.1e-11 / 2.2e-11, torch
    # 9.5e-12 / 2.0e-11, jax 4.5e-12 / 1.8e-11 locally). 1e-10 still leaves six
    # orders of margin against the O(1) errors a branch/convention disagreement
    # produces, which is what this test is for.
    for ic, tol in (
        ([1.0, 0.5, 1.1, 0.2, -0.3, 0.4], 1e-10),
        ([1.0, -0.2, 1.1, -0.2, 0.25, 2.1], 1e-10),  # z<0, vR<0: other branches
        ([1.1, 0.02, 0.9, 0.002, 0.02, 1.0], 1e-8),  # nearly planar
    ):
        lo = aAS.actionsFreqsAngles(*ic, order=10)
        hi = aAS.actionsFreqsAngles(*ic, order=200)
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
    # Evaluate the whole parity grid in ONE vectorised call per method rather
    # than a Python scalar loop: the Staeckel action/freq/angle quadratures are
    # per-point independent, so a batched array call is point-for-point identical
    # to the scalar loop but amortises the per-call backend-graph dispatch (486
    # scalar calls -> one array call is ~300x faster under jax, ~30x under numpy),
    # which keeps this test off the slow-skip ledger.
    Rg, vRg, vTg, zg, vzg, phig = [], [], [], [], [], []
    for R in [0.7, 1.0, 1.3]:
        for vR in [-0.25, 0.0, 0.25]:
            for vT in [-0.6, 0.4, 0.9]:  # retrograde + prograde + near-circular
                for z in [-0.2, 0.0, 0.2]:
                    for vz in [-0.25, 0.05, 0.25]:
                        for phi in [0.4, 2.7]:
                            if z < 0.0 and vR == 0.0 and vz == 0.0:
                                continue
                            Rg.append(R)
                            vRg.append(vR)
                            vTg.append(vT)
                            zg.append(z)
                            vzg.append(vz)
                            phig.append(phi)
    Rg, vRg, vTg, zg, vzg, phig = (
        numpy.array(Rg),
        numpy.array(vRg),
        numpy.array(vTg),
        numpy.array(zg),
        numpy.array(vzg),
        numpy.array(phig),
    )
    n = len(Rg)
    fc = aAc.actionsFreqs(Rg, vRg, vTg, zg, vzg)
    fp = aAp.actionsFreqs(Rg, vRg, vTg, zg, vzg)
    ac = aAc.actionsFreqsAngles(Rg, vRg, vTg, zg, vzg, phig)
    ap = aAp.actionsFreqsAngles(Rg, vRg, vTg, zg, vzg, phig)
    maxfreqdiff = 0.0
    maxangdiff = 0.0
    # jr,Lz,jz,Omegar,Omegaphi,Omegaz (skip any point either path returns NaN)
    for ii in range(6):
        a, b = as_numpy(fc[ii]), as_numpy(fp[ii])
        good = ~(numpy.isnan(a) | numpy.isnan(b))
        if good.any():
            maxfreqdiff = max(maxfreqdiff, numpy.max(numpy.fabs(a[good] - b[good])))
    # angler, anglephi, anglez (wrap-aware)
    for ii in (6, 7, 8):
        a, b = as_numpy(ac[ii]), as_numpy(ap[ii])
        good = ~(numpy.isnan(a) | numpy.isnan(b))
        if good.any():
            maxangdiff = max(maxangdiff, numpy.max(wrapdiff(a[good], b[good])))
    assert n > 100, "Staeckel c vs Python parity grid did not evaluate enough points"
    assert maxfreqdiff < 1e-5, (
        "Pure-Python actionAngleStaeckel frequencies do not agree with C "
        "implementation; max diff = %g" % maxfreqdiff
    )
    assert maxangdiff < 1e-5, (
        "Pure-Python actionAngleStaeckel angles do not agree with C "
        "implementation; max diff = %g" % maxangdiff
    )
    # A few points are ALSO evaluated with scalar (float) inputs so the
    # scalar-input float->array promotion in the C and Python actionsFreqs
    # branches stays exercised (the batched call above passes arrays and skips
    # it), asserting scalar python-vs-C parity on those points too.
    for R, vR, vT, z, vz, phi in [
        (0.7, -0.25, -0.6, -0.2, -0.25, 0.4),
        (1.0, 0.0, 0.4, 0.0, 0.05, 2.7),
        (1.3, 0.25, 0.9, 0.2, 0.25, 0.4),
    ]:
        fcs = aAc.actionsFreqs(R, vR, vT, z, vz)
        fps = aAp.actionsFreqs(R, vR, vT, z, vz)
        acs = aAc.actionsFreqsAngles(R, vR, vT, z, vz, phi)
        aps = aAp.actionsFreqsAngles(R, vR, vT, z, vz, phi)
        for ii in range(6):
            a, b = as_numpy(fcs[ii])[0], as_numpy(fps[ii])[0]
            if not (numpy.isnan(a) or numpy.isnan(b)):
                assert numpy.fabs(a - b) < 1e-5
        for ii in (6, 7, 8):
            a, b = as_numpy(acs[ii])[0], as_numpy(aps[ii])[0]
            if not (numpy.isnan(a) or numpy.isnan(b)):
                assert wrapdiff(a, b) < 1e-5
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


# Regression: estimateDeltaStaeckel with array (R,z) must work for potentials
# whose methods only accept scalar inputs (the numpy path evaluates per-element).
def test_estimateDeltaStaeckel_array_scalaronly_potential():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.potential import DoubleExponentialDiskPotential

    pot = DoubleExponentialDiskPotential(normalize=1.0)
    R = numpy.array([1.0, 1.1, 0.9, 1.05])
    z = numpy.array([0.1, 0.2, 0.05, 0.15])
    # array call (no_median) must not crash and must equal the per-element
    # scalar estimates (the numpy path evaluates the potential per element)
    arr = estimateDeltaStaeckel(pot, R, z, no_median=True)
    indiv = numpy.array([estimateDeltaStaeckel(pot, R[i], z[i]) for i in range(len(R))])
    assert (numpy.fabs(arr - indiv) < 1e-10).all(), (
        "estimateDeltaStaeckel array call disagrees with per-element estimates"
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
        # as_numpy: estimateDeltaStaeckel returns a backend array under a forced
        # backend (it now runs on jax/torch), so coerce before the numpy assertion
        delta0 = as_numpy(estimateDeltaStaeckel(MWPotential2014, r, 0.0))
        deltasmall = as_numpy(estimateDeltaStaeckel(MWPotential2014, r, 5e-4))
        assert numpy.fabs(delta0 - deltasmall) < 1e-3, (
            "Delta computed with z=0 does not agree with that computed for small z"
        )
    # And an array
    delta0 = as_numpy(estimateDeltaStaeckel(MWPotential2014, rs, numpy.zeros(n)))
    deltasmall = as_numpy(
        estimateDeltaStaeckel(MWPotential2014, rs, 5e-4 * numpy.ones(n))
    )
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
            # under a forced backend these come back as backend arrays; the
            # reductions below are numpy's
            results.append([as_numpy(a) for a in out])
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
    jpy = [as_numpy(a) for a in aASpy(R[sub], vR[sub], vT[sub], z[sub], vz[sub])]
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
def test_actionAngleStaeckel_conserved_actions_phi():
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
    # e = 0 exactly on every backend: the backend turning-point solve snaps a
    # pair it cannot resolve from f's rounding noise to circular (it used to
    # leave e ~ 1e-10 .. 3e-8 in the grid's circular row, CPU-dependent)
    ecc_tol = 10.0**-16.0
    assert numpy.fabs(te) < ecc_tol, (
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
    # circular orbit: e and zmax are 0. numpy hits machine-eps; a forced backend
    # does not, so the bars are relaxed from 1e-16 -- e to 1e-14 (it only reorders
    # the (rap-rperi)/(rap+rperi) cancellation, ~1.6e-16), zmax to 1e-9 (the
    # interpolated-grid path evaluates the vertical action to ~2.7e-11 here).
    R, vR, vT, z, vz = 1.0, 0.0, 1.0, 0.0, 0.0
    te, tzmax, _, _ = aAA.EccZmaxRperiRap(R, vR, vT, z, vz)
    assert numpy.fabs(te) < 10.0**-14.0, (
        "Circular orbit in the MWPotential does not have e=0"
    )
    assert numpy.fabs(tzmax) < 10.0**-9.0, (
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
        numpy.reshape(as_numpy(obs.R(obs.t)), (1, len(obs.t))),
        numpy.reshape(as_numpy(obs.vR(obs.t)), (1, len(obs.t))),
        numpy.reshape(as_numpy(obs.vT(obs.t)), (1, len(obs.t))),
        numpy.reshape(as_numpy(obs.z(obs.t)), (1, len(obs.t))),
        numpy.reshape(as_numpy(obs.vz(obs.t)), (1, len(obs.t))),
        numpy.reshape(as_numpy(obs.phi(obs.t)), (1, len(obs.t))),
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


# Setup for the tests of actionAngleAdiabaticApprox against actionAngleSpherical:
# an NFW potential transformed to an isochrone, points along one orbit, and the
# actions, frequencies, and angles of these points from actionAngleSpherical
def setup_actionAngleAdiabaticApprox_nfw(npoints=2):
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleIsochrone,
        actionAngleSpherical,
    )
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, NFWPotential

    nfw = NFWPotential(a=1.0, normalize=1.0)
    aAA = actionAngleAdiabaticApprox(
        pot=nfw,
        aA=actionAngleIsochrone(ip=IsochronePotential(b=0.5, normalize=1.0)),
        tintJ=200.0,
        npoints=npoints,
    )
    obs = Orbit([1.0, 0.9, 1.1, 0.1, 0.3, 0.0])
    times = numpy.linspace(0.0, 10.0, 6)
    obs.integrate(times, nfw, method="dop853_c")
    args = (
        obs.R(times),
        obs.vR(times),
        obs.vT(times),
        obs.z(times),
        obs.vz(times),
        obs.phi(times),
    )
    return (
        aAA,
        args,
        times,
        tuple(
            as_numpy(a) for a in actionAngleSpherical(pot=nfw).actionsFreqsAngles(*args)
        ),
    )


# Test the actionAngleAdiabaticApprox against actionAngleSpherical: actions
def test_actionAngleAdiabaticApprox_nfw_actions():
    aAA, args, times, jfa = setup_actionAngleAdiabaticApprox_nfw()
    js = [as_numpy(j) for j in aAA(*args)]
    djr = numpy.amax(numpy.fabs(js[0] / jfa[0] - 1.0))
    dlz = numpy.amax(numpy.fabs(js[1] / jfa[1] - 1.0))
    djz = numpy.amax(numpy.fabs(js[2] / jfa[2] - 1.0))
    assert djr < 10.0**-4.5, (
        "actionAngleAdiabaticApprox applied to NFW potential fails for Jr at %g%%"
        % (djr * 100.0)
    )
    # Lz and Jz are easy, because both potentials are spherical
    assert dlz < 10.0**-9.0, (
        "actionAngleAdiabaticApprox applied to NFW potential fails for Lz at %g%%"
        % (dlz * 100.0)
    )
    assert djz < 10.0**-9.0, (
        "actionAngleAdiabaticApprox applied to NFW potential fails for Jz at %g%%"
        % (djz * 100.0)
    )
    return None


# Test the actionAngleAdiabaticApprox against actionAngleSpherical: frequencies
@pytest.mark.parametrize("npoints", [2, 4])
def test_actionAngleAdiabaticApprox_nfw_freqs(npoints):
    aAA, args, times, jfa = setup_actionAngleAdiabaticApprox_nfw(npoints=npoints)
    jfs = [as_numpy(j) for j in aAA.actionsFreqs(*args)]
    assert len(jfs) == 6, "actionAngleAdiabaticApprox.actionsFreqs does not return six"
    djr = numpy.amax(numpy.fabs(jfs[0] / jfa[0] - 1.0))
    assert djr < 10.0**-4.5, (
        "actionAngleAdiabaticApprox.actionsFreqs applied to NFW potential fails for Jr at %g%%"
        % (djr * 100.0)
    )
    for ii, name in zip(range(3, 6), ["Or", "Op", "Oz"]):
        dO = numpy.amax(numpy.fabs(jfs[ii] / jfa[ii] - 1.0))
        assert dO < 10.0**-3.5, (
            "actionAngleAdiabaticApprox applied to NFW potential fails for %s at %g%%"
            % (name, dO * 100.0)
        )
    return None


# Test the actionAngleAdiabaticApprox against actionAngleSpherical: the angles
# are those of the torus up to an offset that is constant along the orbit (the
# phase accumulated during the transformation), so they increase at the true
# frequencies
def test_actionAngleAdiabaticApprox_nfw_angles():
    aAA, args, times, jfa = setup_actionAngleAdiabaticApprox_nfw()
    jfas = [as_numpy(j) for j in aAA.actionsFreqsAngles(*args)]
    for ii, name in zip(range(6, 9), ["ar", "ap", "az"]):
        offset = jfas[ii] - jfa[ii]
        doffset = (offset - offset[0] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(doffset)) < 10.0**-3.3, (
            "actionAngleAdiabaticApprox applied to NFW potential has an offset from the true %s that is not constant along the orbit, but varies by %g"
            % (name, numpy.amax(numpy.fabs(doffset)))
        )
        dangle = (jfas[ii] - jfas[ii][0] - jfa[ii - 3] * times + numpy.pi) % (
            2.0 * numpy.pi
        ) - numpy.pi
        assert numpy.amax(numpy.fabs(dangle)) < 10.0**-3.3, (
            "actionAngleAdiabaticApprox applied to NFW potential has %s that does not increase at the true frequency, off by %g"
            % (name, numpy.amax(numpy.fabs(dangle)))
        )
    return None


# Test that the actions are conserved along an orbit in MWPotential2014 when
# transforming to a Staeckel potential, and better so than the Staeckel
# approximation in MWPotential2014 itself
def test_actionAngleAdiabaticApprox_staeckel_conserved_actions():
    from galpy.actionAngle import actionAngleAdiabaticApprox, actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    delta = 0.4
    aAA = actionAngleAdiabaticApprox(
        pot=MWPotential2014,
        aA=actionAngleStaeckel(
            pot=OblateStaeckelWrapperPotential(
                pot=MWPotential2014, delta=delta, u0=1.0
            ),
            delta=delta,
        ),
        ntintJ=101,
    )
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=delta)
    obs = Orbit([1.0, 0.2, 1.1, 0.1, 0.3, 0.0])
    times = numpy.linspace(0.0, 20.0, 11)
    obs.integrate(times, MWPotential2014, method="dop853_c")
    js = numpy.array(aAA(obs(times)))
    jss = numpy.array(aAS(obs(times)))
    maxdj = numpy.amax(numpy.fabs(js.T / numpy.mean(js, axis=1) - 1.0), axis=0)
    maxdjs = numpy.amax(numpy.fabs(jss.T / numpy.mean(jss, axis=1) - 1.0), axis=0)
    assert maxdj[0] < 10.0**-3.0, "Jr conservation fails at %g%%" % (100.0 * maxdj[0])
    assert maxdj[1] < 10.0**-9.0, "Lz conservation fails at %g%%" % (100.0 * maxdj[1])
    assert maxdj[2] < 10.0**-3.0, "Jz conservation fails at %g%%" % (100.0 * maxdj[2])
    assert maxdj[0] < maxdjs[0] / 10.0, (
        "Jr is not conserved better by actionAngleAdiabaticApprox than by actionAngleStaeckel"
    )
    assert maxdj[2] < maxdjs[2] / 2.0, (
        "Jz is not conserved better by actionAngleAdiabaticApprox than by actionAngleStaeckel"
    )
    # and the actions are close to those of the Staeckel approximation
    assert numpy.all(
        numpy.fabs(numpy.mean(js, axis=1) / numpy.mean(jss, axis=1) - 1.0) < 0.01
    ), "actionAngleAdiabaticApprox actions are not close to actionAngleStaeckel's"
    return None


# Test the other action-angle methods that one can transform to:
# actionAngleSpherical and actionAngleAdiabatic
def test_actionAngleAdiabaticApprox_otheraA():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleAdiabaticApprox,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import (
        HernquistPotential,
        MiyamotoNagaiPotential,
        MWPotential2014,
        NFWPotential,
    )

    # actionAngleSpherical: NFW --> Hernquist
    nfw = NFWPotential(a=1.0, normalize=1.0)
    aAA = actionAngleAdiabaticApprox(
        pot=nfw,
        aA=actionAngleSpherical(pot=HernquistPotential(a=2.0, normalize=1.0)),
        ntintJ=101,
    )
    R, vR, vT, z, vz, phi = 1.0, 0.9, 1.1, 0.1, 0.3, 0.0
    jfs = aAA.actionsFreqs(R, vR, vT, z, vz, phi)
    jfa = actionAngleSpherical(pot=nfw).actionsFreqs(R, vR, vT, z, vz, phi)
    for ii, tol in enumerate([-6.0, -9.0, -9.0, -4.0, -4.0, -4.0]):
        assert numpy.fabs(jfs[ii] / jfa[ii] - 1.0) < 10.0**tol, (
            "actionAngleAdiabaticApprox with actionAngleSpherical fails for output %i at %g%%"
            % (ii, 100.0 * numpy.fabs(jfs[ii] / jfa[ii] - 1.0))
        )
    # actionAngleAdiabatic: MWPotential2014 --> Miyamoto-Nagai, for a disk
    # orbit; compare to the Staeckel approximation
    aAA = actionAngleAdiabaticApprox(
        pot=MWPotential2014,
        aA=actionAngleAdiabatic(
            pot=MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0), gamma=1.0
        ),
        tintJ=20.0,
        ntintJ=101,
    )
    R, vR, vT, z, vz, phi = 1.0, 0.1, 1.1, 0.03, 0.05, 0.0
    jfas = aAA.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    jfa = actionAngleStaeckel(pot=MWPotential2014, delta=0.4).actionsFreqsAngles(
        R, vR, vT, z, vz, phi
    )
    assert len(jfas) == 9, (
        "actionAngleAdiabaticApprox.actionsFreqsAngles does not return nine"
    )
    for ii, tol in enumerate([0.05, 1e-9, 0.05, 0.05, 0.05]):
        assert numpy.fabs(jfas[ii] / jfa[ii] - 1.0) < tol, (
            "actionAngleAdiabaticApprox with actionAngleAdiabatic fails for output %i at %g%%"
            % (ii, 100.0 * numpy.fabs(jfas[ii] / jfa[ii] - 1.0))
        )
    return None


# Test the different forms of input
def test_actionAngleAdiabaticApprox_inputs():
    from galpy.actionAngle import actionAngleAdiabaticApprox, actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, NFWPotential

    nfw = NFWPotential(a=1.0, normalize=1.0)
    aAI = actionAngleIsochrone(ip=IsochronePotential(b=0.5, normalize=1.0))
    aAA = actionAngleAdiabaticApprox(pot=nfw, aA=aAI, tintJ=20.0, ntintJ=101)
    vxvv = numpy.array(
        [[1.0, 0.9, 1.1, 0.1, 0.3, 0.0], [1.1, 0.3, 0.9, -0.2, 0.1, 2.0]]
    )
    # arrays are the reference
    jfa = numpy.array(aAA.actionsFreqsAngles(*vxvv.T))
    assert jfa.shape == (9, 2), (
        "actionAngleAdiabaticApprox output for array input does not have the correct shape"
    )
    # floats
    for ii in range(2):
        assert numpy.all(
            numpy.fabs(
                numpy.array(aAA.actionsFreqsAngles(*vxvv[ii]))[:, 0] - jfa[:, ii]
            )
            < 1e-10
        ), "actionAngleAdiabaticApprox for float input does not agree with array input"
    # Orbit, single and multiple
    assert numpy.all(
        numpy.fabs(numpy.array(aAA.actionsFreqsAngles(Orbit(vxvv))) - jfa) < 1e-10
    ), "actionAngleAdiabaticApprox for Orbit input does not agree with array input"
    assert numpy.all(
        numpy.fabs(numpy.array(aAA(Orbit(vxvv[0])))[:, 0] - jfa[:3, 0]) < 1e-10
    ), "actionAngleAdiabaticApprox for Orbit input does not agree with array input"
    # planar: R,vR,vT,phi as floats, arrays, and Orbits against z=vz=0
    vxvv[:, 3:5] = 0.0
    jfa = numpy.array(aAA.actionsFreqsAngles(*vxvv.T))
    assert numpy.all(numpy.fabs(jfa[2]) < 1e-10), (
        "actionAngleAdiabaticApprox for a planar orbit does not give zero Jz"
    )
    planar = vxvv[:, [0, 1, 2, 5]]
    assert numpy.all(
        numpy.fabs(numpy.array(aAA.actionsFreqsAngles(*planar.T)) - jfa) < 1e-10
    ), "actionAngleAdiabaticApprox for planar array input does not agree with z=vz=0"
    assert numpy.all(
        numpy.fabs(numpy.array(aAA.actionsFreqs(*planar[0]))[:, 0] - jfa[:6, 0]) < 1e-10
    ), "actionAngleAdiabaticApprox for planar float input does not agree with z=vz=0"
    assert numpy.all(
        numpy.fabs(numpy.array(aAA.actionsFreqsAngles(Orbit(planar))) - jfa) < 1e-10
    ), "actionAngleAdiabaticApprox for planar Orbit input does not agree with z=vz=0"
    return None


# Test that a negative npoints raises an error
def test_actionAngleAdiabaticApprox_npoints_error():
    from galpy.actionAngle import actionAngleAdiabaticApprox, actionAngleIsochrone
    from galpy.potential import NFWPotential

    with pytest.raises(OSError, match="npoints= must be at least 0"):
        actionAngleAdiabaticApprox(
            pot=NFWPotential(a=1.0, normalize=1.0),
            aA=actionAngleIsochrone(b=0.5),
            npoints=-2,
        )
    with pytest.raises(ValueError, match="npoints= must be even"):
        actionAngleAdiabaticApprox(
            pot=NFWPotential(a=1.0, normalize=1.0),
            aA=actionAngleIsochrone(b=0.5),
            npoints=3,
        )
    return None


# Test npoints=0: each point is transformed on its own, without frequencies
def test_actionAngleAdiabaticApprox_singlepoint():
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleIsochrone,
        actionAngleSpherical,
    )
    from galpy.orbit import Orbit
    from galpy.potential import NFWPotential

    np = NFWPotential(a=1.0, normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    aAA = actionAngleAdiabaticApprox(
        pot=np, aA=actionAngleIsochrone(b=0.5), tintJ=200.0, npoints=0
    )
    o = Orbit([1.0, 0.9, 1.1, 0.1, 0.3, 0.0])
    ts = numpy.linspace(0.0, 10.0, 6)
    o.integrate(ts, np)
    true = numpy.array([as_numpy(x) for x in aAS.actionsFreqsAngles(o(ts))])
    raw = aAA.actionsFreqsAngles(o(ts))
    out = numpy.array([as_numpy(x) for x in raw])
    assert numpy.all(numpy.fabs(out[:3] / true[:3] - 1.0) < 10.0**-4.5), (
        "actionAngleAdiabaticApprox with npoints=0 does not return the actions"
    )
    if is_backend_array(raw[3]):
        # (on a backend: dH/dJ by automatic differentiation)
        assert numpy.all(numpy.fabs(out[3:6] / true[3:6] - 1.0) < 10.0**-3.5), (
            "actionAngleAdiabaticApprox with npoints=0 does not return the frequencies"
        )
    else:
        assert numpy.all(numpy.isnan(out[3:6])), (
            "actionAngleAdiabaticApprox with npoints=0 returns frequencies"
        )
    assert numpy.all((out[6:] >= 0.0) & (out[6:] < 2.0 * numpy.pi)), (
        "actionAngleAdiabaticApprox with npoints=0 does not return angles in [0, 2 pi)"
    )
    # the angles are those of the torus up to a constant offset
    dang = (out[6:] - true[6:] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    assert numpy.all(numpy.fabs(dang - dang[:, :1]) < 10.0**-3.0), (
        "actionAngleAdiabaticApprox with npoints=0 does not return angles that "
        "differ from the true angles by a constant on a torus"
    )
    # __call__ and actionsFreqs
    assert numpy.allclose(numpy.array(aAA(o(ts))), out[:3], rtol=0.0, atol=1e-10)
    assert numpy.allclose(
        numpy.array(aAA.actionsFreqs(o(ts)))[:3], out[:3], rtol=0.0, atol=1e-10
    )
    return None


# Test the angles' range, integer input, and an Orbit without phi
def test_actionAngleAdiabaticApprox_inputs_and_range():
    from galpy.actionAngle import actionAngleAdiabaticApprox, actionAngleIsochrone
    from galpy.orbit import Orbit
    from galpy.potential import NFWPotential

    aAA = actionAngleAdiabaticApprox(
        pot=NFWPotential(a=1.0, normalize=1.0),
        aA=actionAngleIsochrone(b=0.5),
        tintJ=20.0,
    )
    o = Orbit([1.0, 0.9, 1.1, 0.1, 0.3, 0.0])
    ts = numpy.linspace(0.0, 30.0, 31)
    o.integrate(ts, NFWPotential(a=1.0, normalize=1.0))
    ang = numpy.array(aAA.actionsFreqsAngles(o(ts)))[6:]
    assert numpy.all((ang >= 0.0) & (ang < 2.0 * numpy.pi)), (
        "actionAngleAdiabaticApprox does not return angles in [0, 2 pi)"
    )
    # integers
    assert numpy.allclose(
        numpy.array(aAA(1, 0.9, 1.1, 0.1, 0.3, 0)),
        numpy.array(aAA(1.0, 0.9, 1.1, 0.1, 0.3, 0.0)),
        rtol=0.0,
        atol=1e-12,
    ), "actionAngleAdiabaticApprox does not accept integer input"
    # an Orbit without phi, also after a call with phi
    with pytest.raises(ValueError, match="Must specify phi"):
        aAA(Orbit([1.0, 0.9, 1.1, 0.1, 0.3]))
    return None


# Test that actionAngleAdiabaticApproxInverse is the inverse of
# actionAngleAdiabaticApprox for any duration of the transformation, and
# returns points with the requested actions for a long one
def test_actionAngleAdiabaticApproxInverse_inverse():
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleAdiabaticApproxInverse,
        actionAngleIsochrone,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
    )
    from galpy.potential import NFWPotential

    np = NFWPotential(a=1.0, normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    rng = numpy.random.default_rng(0)
    jr, lz, jz = (
        rng.uniform(0.05, 0.5, 6),
        rng.uniform(0.5, 1.5, 6),
        rng.uniform(0.01, 0.3, 6),
    )
    acts = numpy.array([jr, lz, jz])
    ang = rng.uniform(0.0, 2.0 * numpy.pi, (3, 6))
    for tintJ, tol in zip((20.0, 200.0), (10.0**-1.0, 10.0**-3.5)):
        aAAI = actionAngleAdiabaticApproxInverse(
            pot=np, aAI=actionAngleIsochroneInverse(b=0.5), tintJ=tintJ
        )
        aAA = actionAngleAdiabaticApprox(
            pot=np, aA=actionAngleIsochrone(b=0.5), tintJ=tintJ, npoints=0
        )
        xv = aAAI(jr, lz, jz, *ang)
        back = numpy.array(aAA.actionsFreqsAngles(*xv))
        dang = (back[6:] - ang + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.all(numpy.fabs(back[:3] - acts) < 1e-8) and numpy.all(
            numpy.fabs(dang) < 1e-7
        ), (
            "actionAngleAdiabaticApproxInverse is not the inverse of "
            f"actionAngleAdiabaticApprox for tintJ={tintJ}"
        )
        assert numpy.all(numpy.fabs(numpy.array(aAS(*xv)) / acts - 1.0) < tol), (
            "actionAngleAdiabaticApproxInverse does not return points with the "
            f"requested actions for tintJ={tintJ}"
        )
    return None


# Test the frequencies and the broadcasting of
# actionAngleAdiabaticApproxInverse
def test_actionAngleAdiabaticApproxInverse_freqs_shapes():
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleAdiabaticApproxInverse,
        actionAngleIsochrone,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
    )
    from galpy.actionAngle.actionAngleInverse import actionAngleInverse
    from galpy.potential import NFWPotential

    np = NFWPotential(a=1.0, normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    aAAI = actionAngleAdiabaticApproxInverse(
        pot=np,
        aAI=actionAngleIsochroneInverse(b=0.5),
        aA=actionAngleIsochrone(b=0.5),
        tintJ=200.0,
    )
    jr, lz, jz = (
        numpy.array([0.1, 0.3]),
        numpy.array([1.0, 0.8]),
        numpy.array([0.05, 0.2]),
    )
    angs = numpy.array([0.5, 2.0, 4.0])
    # every angle on every torus
    out = numpy.array(
        aAAI.xvFreqs(jr[:, None], lz[:, None], jz[:, None], angs, 1.0, 2.0)
    )
    assert out.shape == (9, 2, 3), (
        "actionAngleAdiabaticApproxInverse does not broadcast actions against angles"
    )
    true = numpy.array(aAS.actionsFreqs(*(x.ravel() for x in out[:6])))
    assert numpy.all(
        numpy.fabs(out[6:].reshape((3, 6)) / true[3:] - 1.0) < 10.0**-3.0
    ), "actionAngleAdiabaticApproxInverse.xvFreqs does not return the frequencies"
    freqs = numpy.array(aAAI.Freqs(jr, lz, jz))
    assert freqs.shape == (3, 2) and numpy.all(
        numpy.fabs(freqs / out[6:, :, 0] - 1.0) < 10.0**-3.0
    ), "actionAngleAdiabaticApproxInverse.Freqs does not return the frequencies"
    # one torus, floats
    one = numpy.array(aAAI(0.1, 1.0, 0.05, 0.5, 1.0, 2.0))
    assert numpy.allclose(one[:, 0], out[:6, 0, 0], rtol=0.0, atol=1e-10)
    # without aA=, the frequencies are those of the forward transformation
    # that is set up from aAI=
    auto = actionAngleAdiabaticApproxInverse(
        pot=np, aAI=actionAngleIsochroneInverse(b=0.5), tintJ=200.0
    )
    assert numpy.allclose(
        numpy.array(auto.Freqs(jr, lz, jz)), freqs, rtol=0.0, atol=1e-10
    ), "actionAngleAdiabaticApproxInverse without aA= does not return the frequencies"

    # ... unless galpy cannot set that up from aAI=
    class otherInverse(actionAngleInverse):
        def __init__(self, aAI):
            actionAngleInverse.__init__(self)
            self._aAI, self._pot = aAI, aAI._pot

        def _evaluate(self, *args, **kwargs):
            return self._aAI(*args, **kwargs)

    other = otherInverse(actionAngleIsochroneInverse(b=0.5))
    nofreq = actionAngleAdiabaticApproxInverse(pot=np, aAI=other, tintJ=200.0)
    assert numpy.allclose(
        numpy.array(nofreq(0.1, 1.0, 0.05, 0.5, 1.0, 2.0)), one, rtol=0.0, atol=1e-10
    )
    with pytest.raises(RuntimeError, match="require aA= at initialization"):
        nofreq.Freqs(0.1, 1.0, 0.05)
    with pytest.raises(ValueError, match="aA= is necessary with aAAI="):
        actionAngleAdiabaticApprox(aAAI=nofreq)
    return None


# Test the frequencies of actionAngleAdiabaticApproxInverse against the exact
# frequencies of the returned points in a spherical potential: they converge
# with the duration of the transformation
def test_actionAngleAdiabaticApproxInverse_freqs_converge():
    from galpy.actionAngle import (
        actionAngleAdiabaticApproxInverse,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
    )
    from galpy.potential import NFWPotential

    np = NFWPotential(a=2.0, normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    jr, lz, jz = (
        numpy.array([0.05, 0.2]),
        numpy.array([1.0, 0.6]),
        numpy.array([0.1, 0.3]),
    )
    angs = (numpy.array([1.0, 2.0]), numpy.array([2.0, 3.0]), numpy.array([3.0, 4.0]))
    err = []
    for tintJ, tol in zip((20.0, 100.0, 400.0), (1e-3, 3e-5, 3e-6)):
        aAAI = actionAngleAdiabaticApproxInverse(
            pot=np, aAI=actionAngleIsochroneInverse(b=1.0), tintJ=tintJ
        )
        out = numpy.array(aAAI.xvFreqs(jr, lz, jz, *angs))
        true = numpy.array(aAS.actionsFreqs(*out[:5]))[3:]
        err.append(numpy.amax(numpy.fabs(out[6:] / true - 1.0)))
        assert err[-1] < tol, (
            "actionAngleAdiabaticApproxInverse.xvFreqs does not return the "
            f"frequencies of the returned points for tintJ={tintJ}"
        )
        # those of the torus are evaluated at one of its points
        assert (
            numpy.amax(numpy.fabs(numpy.array(aAAI.Freqs(jr, lz, jz)) / true - 1.0))
            < 3.0 * tol
        ), (
            "actionAngleAdiabaticApproxInverse.Freqs does not return the "
            f"frequencies of the tori for tintJ={tintJ}"
        )
    assert err[2] < err[1] / 5.0 < err[0] / 25.0, (
        "The frequencies of actionAngleAdiabaticApproxInverse do not converge "
        "with tintJ"
    )
    return None


# Test that each of actionAngleAdiabaticApprox and
# actionAngleAdiabaticApproxInverse can be set up from an instance of the other
def test_actionAngleAdiabaticApprox_from_each_other():
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleAdiabaticApproxInverse,
        actionAngleIsochrone,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleSphericalInverse,
        actionAngleStaeckel,
        actionAngleStaeckelInverse,
    )
    from galpy.orbit import Orbit
    from galpy.potential import (
        HernquistPotential,
        MWPotential2014,
        NFWPotential,
        OblateStaeckelWrapperPotential,
    )

    np = NFWPotential(a=1.0, normalize=1.0)
    x = (1.0, 0.3, 1.1, 0.1, 0.2, 0.5)
    # isochrone: either from the other, nothing else given
    aAA = actionAngleAdiabaticApprox(
        pot=np, aA=actionAngleIsochrone(b=0.5), tintJ=30.0, npoints=0, dt=0.01
    )
    aAAI = actionAngleAdiabaticApproxInverse(aAA=aAA)
    jfa = numpy.array(aAA.actionsFreqsAngles(*x))[:, 0]
    assert numpy.allclose(
        numpy.array(aAAI(*jfa[:3], *jfa[6:]))[:, 0], x, rtol=0.0, atol=1e-8
    ), "actionAngleAdiabaticApproxInverse(aAA=) is not the inverse of aAA"
    assert numpy.allclose(
        numpy.array(
            actionAngleAdiabaticApprox(aAAI=aAAI, npoints=0).actionsFreqsAngles(*x)
        )[[0, 1, 2, 6, 7, 8], 0],
        jfa[[0, 1, 2, 6, 7, 8]],
        rtol=0.0,
        atol=1e-10,
    ), "actionAngleAdiabaticApprox(aAAI=) is not the transformation aAAI inverts"
    # also when the inverse was set up without aA=, or with aA= given here
    bare = actionAngleAdiabaticApproxInverse(
        pot=np, aAI=actionAngleIsochroneInverse(b=0.5), tintJ=30.0, dt=0.01
    )
    for kw in ({}, dict(aA=actionAngleIsochrone(b=0.5))):
        assert numpy.allclose(
            numpy.array(actionAngleAdiabaticApprox(aAAI=bare, npoints=0, **kw)(*x))[
                :, 0
            ],
            jfa[:3],
            rtol=0.0,
            atol=1e-10,
        ), "actionAngleAdiabaticApprox(aAAI=) differs from the direct set-up"
    # the frequencies with points on either side of a point are fit about the
    # advance of the angles at the auxiliary's frequencies, such that
    # npoints_dt does not have to resolve half a period
    aAS = actionAngleSpherical(pot=np)
    wide = numpy.array(
        actionAngleAdiabaticApprox(aAAI=aAAI, npoints_dt=4.0).actionsFreqs(*x)
    )[3:, 0]
    assert numpy.all(
        numpy.fabs(wide / numpy.array(aAS.actionsFreqs(*x[:5]))[3:, 0] - 1.0) < 1e-3
    ), "actionAngleAdiabaticApprox frequencies fail for a large npoints_dt"
    # a Staeckel auxiliary: the forward transformation from its inverse
    aux = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45, u0=1.0)
    aAAI = actionAngleAdiabaticApproxInverse(
        pot=MWPotential2014,
        aAI=actionAngleStaeckelInverse(
            pot=aux, delta=0.45, jrs=[0.05], Lzs=[1.0], jzs=[0.03], progressbar=False
        ),
        tintJ=30.0,
    )
    xv = numpy.array(aAAI(0.05, 1.0, 0.03, 1.0, 2.0, 3.0))[:, 0]
    back = numpy.array(
        actionAngleAdiabaticApprox(aAAI=aAAI, npoints=0).actionsFreqsAngles(*xv)
    )[:, 0]
    assert numpy.allclose(
        back[[0, 1, 2, 6, 7, 8]], [0.05, 1.0, 0.03, 1.0, 2.0, 3.0], rtol=0.0, atol=1e-5
    ), "actionAngleAdiabaticApprox(aAAI=) with a Staeckel auxiliary is not the inverse"
    assert numpy.all(numpy.isfinite(numpy.array(aAAI.Freqs(0.05, 1.0, 0.03))))
    # a Staeckel inverse set up with delta= from the potential itself: the
    # auxiliary potential is the Staeckel potential that it approximates the
    # potential with, not the potential
    direct = actionAngleAdiabaticApproxInverse(
        pot=MWPotential2014,
        aAI=actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=0.45,
            u0=1.0,
            jrs=[0.05],
            Lzs=[1.0],
            jzs=[0.03],
            progressbar=False,
        ),
        tintJ=30.0,
    )
    assert numpy.allclose(
        numpy.array(direct(0.05, 1.0, 0.03, 1.0, 2.0, 3.0))[:, 0],
        xv,
        rtol=0.0,
        atol=1e-9,
    ), (
        "actionAngleAdiabaticApproxInverse does not transform from the "
        "Staeckel potential of an actionAngleStaeckelInverse with delta="
    )
    assert numpy.allclose(
        numpy.array(
            actionAngleAdiabaticApprox(aAAI=direct, npoints=0).actionsFreqsAngles(*xv)
        )[[0, 1, 2, 6, 7, 8], 0],
        back[[0, 1, 2, 6, 7, 8]],
        rtol=0.0,
        atol=1e-9,
    ), (
        "actionAngleAdiabaticApprox does not transform to the Staeckel "
        "potential of an actionAngleStaeckelInverse with delta="
    )
    # an actionAngleStaeckel for a potential that is not of Staeckel form
    # (its Staeckel approximation) stands for the Staeckel potential with
    # its focal length
    wrapped = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45)
    for kw in ({}, dict(ro=8.0, vo=220.0)):
        assert numpy.allclose(
            numpy.array(
                actionAngleAdiabaticApprox(
                    pot=MWPotential2014,
                    aA=actionAngleStaeckel(pot=MWPotential2014, delta=0.45, **kw),
                    tintJ=30.0,
                ).actionsFreqsAngles(*xv)
            ),
            numpy.array(
                actionAngleAdiabaticApprox(
                    pot=MWPotential2014,
                    aA=actionAngleStaeckel(pot=wrapped, delta=0.45, **kw),
                    tintJ=30.0,
                ).actionsFreqsAngles(*xv)
            ),
            rtol=1e-10,
            atol=1e-10,
        ), (
            "actionAngleAdiabaticApprox with an actionAngleStaeckel aA= for a "
            "potential that is not of Staeckel form does not transform to the "
            "Staeckel potential of its focal length"
        )
    # one whose Staeckel potential follows its tori has no single auxiliary
    # potential
    follows = actionAngleStaeckelInverse(
        pot=MWPotential2014,
        delta=0.45,
        u0="torus",
        jrs=[0.05],
        Lzs=[1.0],
        jzs=[0.03],
        progressbar=False,
    )
    with pytest.raises(NotImplementedError, match="different Staeckel potential"):
        actionAngleAdiabaticApproxInverse(pot=MWPotential2014, aAI=follows)
    with pytest.raises(NotImplementedError, match="different Staeckel potential"):
        actionAngleAdiabaticApprox(
            pot=MWPotential2014, aA=actionAngleStaeckel(inverse=follows)
        )
    # a spherical auxiliary other than the isochrone: likewise
    hp = HernquistPotential(a=1.5, normalize=1.0)
    jr, lz, jz = (float(j[0]) for j in actionAngleSpherical(pot=hp)(*x[:5]))
    aAAI = actionAngleAdiabaticApproxInverse(
        pot=np,
        aAI=actionAngleSphericalInverse(
            pot=hp,
            Es=[Orbit(list(x)).E(pot=hp)],
            Ls=[numpy.fabs(lz) + jz],
            progressbar=False,
        ),
        tintJ=30.0,
    )
    xv = numpy.array(aAAI(jr, lz, jz, 1.0, 2.0, 3.0))[:, 0]
    back = numpy.array(
        actionAngleAdiabaticApprox(aAAI=aAAI, npoints=0).actionsFreqsAngles(*xv)
    )[:, 0]
    assert numpy.allclose(
        back[[0, 1, 2, 6, 7, 8]], [jr, lz, jz, 1.0, 2.0, 3.0], rtol=0.0, atol=1e-5
    ), "actionAngleAdiabaticApprox(aAAI=) with a spherical auxiliary is not the inverse"
    assert numpy.all(numpy.isfinite(numpy.array(aAAI.Freqs(jr, lz, jz))))
    # the inverse of a transformation to a potential without a known inverse
    with pytest.raises(ValueError, match="aAI= is necessary with aAA="):
        actionAngleAdiabaticApproxInverse(
            aAA=actionAngleAdiabaticApprox(pot=np, aA=actionAngleSpherical(pot=np))
        )
    return None


# Test the calibrated angles of actionAngleAdiabaticApprox and its inverse
# against the angles of a spherical potential
def test_actionAngleAdiabaticApprox_calibrate_angles():
    from galpy.actionAngle import (
        actionAngleAdiabaticApprox,
        actionAngleAdiabaticApproxInverse,
        actionAngleIsochrone,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
    )
    from galpy.actionAngle.actionAngleInverse import actionAngleInverse
    from galpy.potential import IsochronePotential, NFWPotential

    def wrap(x):
        return (x + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

    np = NFWPotential(a=1.0, normalize=1.0)
    aAS = actionAngleSpherical(pot=np)
    rng = numpy.random.default_rng(1)
    n = 40
    x = (
        rng.uniform(0.5, 1.5, n),
        rng.normal(0.0, 0.3, n),
        rng.choice([-1.0, 1.0], n, p=[0.2, 0.8]) * rng.uniform(0.5, 1.2, n),
        rng.uniform(-0.5, 0.5, n),
        rng.normal(0.0, 0.3, n),
        rng.uniform(0.0, 2.0 * numpy.pi, n),
    )
    true = numpy.array(aAS.actionsFreqsAngles(*x))[6:]
    # forward: the auxiliary's inverse set up automatically for an isochrone
    for npoints in (0, 2):
        aAA = actionAngleAdiabaticApprox(
            pot=np,
            aA=actionAngleIsochrone(b=0.5),
            tintJ=300.0,
            npoints=npoints,
            calibrate_angles=True,
        )
        ang = numpy.array(aAA.actionsFreqsAngles(*x))[6:]
        assert numpy.all(numpy.fabs(wrap(ang - true)) < 1e-3), (
            "actionAngleAdiabaticApprox with calibrate_angles=True does not "
            f"return the angles of the usual convention for npoints={npoints}"
        )
    # they differ from the uncalibrated ones
    raw = numpy.array(
        actionAngleAdiabaticApprox(
            pot=np, aA=actionAngleIsochrone(b=0.5), tintJ=300.0
        ).actionsFreqsAngles(*x)
    )[6:]
    assert numpy.amax(numpy.fabs(wrap(raw - true))) > 1.0
    # without the auxiliary's inverse: the half turns from the angles that
    # the auxiliary assigns to the points themselves
    ip = IsochronePotential(b=0.5, normalize=1.0)
    aAA = actionAngleAdiabaticApprox(
        pot=np,
        aA=actionAngleSpherical(pot=ip),
        tintJ=300.0,
        npoints=0,
        calibrate_angles=True,
    )
    assert numpy.all(
        numpy.fabs(wrap(numpy.array(aAA.actionsFreqsAngles(*x))[6:] - true)) < 1e-3
    ), (
        "actionAngleAdiabaticApprox with calibrate_angles=True does not return "
        "the angles of the usual convention without an inverse of aA="
    )
    # a point that is unbound in the auxiliary potential before the
    # transformation: no such angles, with a warning
    light = IsochronePotential(b=0.5, normalize=0.3)
    aAA = actionAngleAdiabaticApprox(
        pot=np,
        aA=actionAngleSpherical(pot=light),
        tintJ=300.0,
        npoints=0,
        calibrate_angles=True,
    )
    fast = (1.0, 0.3, 1.1, 0.1, 0.2, 0.5)
    with pytest.warns(galpyWarning, match="may be half a turn off"):
        ang = numpy.array(aAA.actionsFreqsAngles(*fast))[6:, 0]
    # (correct up to the half turns, at the precision of this poor auxiliary)
    assert numpy.all(
        numpy.fabs(numpy.sin(ang - numpy.array(aAS.actionsFreqsAngles(*fast))[6:, 0]))
        < 0.1
    ), "calibrated angles of a point that is unbound in the auxiliary are off"
    # inverse
    jr, lz, jz = (
        rng.uniform(0.02, 0.4, n),
        rng.choice([-1.0, 1.0], n, p=[0.2, 0.8]) * rng.uniform(0.4, 1.4, n),
        rng.uniform(0.0, 0.4, n),
    )
    ang = rng.uniform(0.0, 2.0 * numpy.pi, (3, n))
    aAAI = actionAngleAdiabaticApproxInverse(
        pot=np,
        aAI=actionAngleIsochroneInverse(b=0.5),
        tintJ=300.0,
        calibrate_angles=True,
    )
    xv = aAAI(jr, lz, jz, *ang)
    # (these tori include more eccentric ones, with larger errors at this tintJ)
    assert numpy.all(
        numpy.fabs(wrap(numpy.array(aAS.actionsFreqsAngles(*xv))[6:] - ang)) < 3e-3
    ), (
        "actionAngleAdiabaticApproxInverse with calibrate_angles=True does not "
        "return points with the requested angles of the usual convention"
    )
    # and the two calibrated transformations invert each other to the
    # precision of the orbit integration: the offset is the same function of
    # the actions in both
    back = numpy.array(
        actionAngleAdiabaticApprox(
            aAAI=aAAI, npoints=0, calibrate_angles=True
        ).actionsFreqsAngles(*xv)
    )
    assert numpy.all(
        numpy.fabs(back[:3] - numpy.array([jr, lz, jz])) < 1e-8
    ) and numpy.all(numpy.fabs(wrap(back[6:] - ang)) < 1e-6), (
        "actionAngleAdiabaticApprox and its inverse with calibrate_angles=True "
        "do not invert each other"
    )
    aAA = actionAngleAdiabaticApprox(
        pot=np,
        aA=actionAngleIsochrone(b=0.5),
        tintJ=300.0,
        npoints=0,
        calibrate_angles=True,
    )
    jfa = numpy.array(aAA.actionsFreqsAngles(*x))
    xb = numpy.array(
        actionAngleAdiabaticApproxInverse(aAA=aAA, calibrate_angles=True)(
            *jfa[:3], *jfa[6:]
        )
    )
    assert numpy.all(numpy.fabs(xb[:5] - numpy.array(x)[:5]) < 1e-6) and numpy.all(
        numpy.fabs(wrap(xb[5] - x[5])) < 1e-6
    ), (
        "actionAngleAdiabaticApproxInverse with calibrate_angles=True does not "
        "return the points that actionAngleAdiabaticApprox was given"
    )

    # calibration needs the forward transformation
    class otherInverse(actionAngleInverse):
        def __init__(self, aAI):
            actionAngleInverse.__init__(self)
            self._aAI, self._pot = aAI, aAI._pot

        def _evaluate(self, *args, **kwargs):  # pragma: no cover
            return self._aAI(*args, **kwargs)

    with pytest.raises(ValueError, match="calibrate_angles=True requires aA="):
        actionAngleAdiabaticApproxInverse(
            pot=np,
            aAI=otherInverse(actionAngleIsochroneInverse(b=0.5)),
            calibrate_angles=True,
        )
    return None


# Test that using MWPotential throws a warning
def test_actionAngleAdiabaticApprox_MWPotential_warning():
    from galpy.actionAngle import actionAngleAdiabaticApprox, actionAngleIsochrone
    from galpy.potential import MWPotential

    with pytest.warns(galpyWarning, match="Use of MWPotential as a Milky-Way-like"):
        actionAngleAdiabaticApprox(pot=MWPotential, aA=actionAngleIsochrone(b=0.8))
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
        as_numpy(obs.jr(type=type))
    except AttributeError:
        pass  # should raise this, as we have not specified a potential
    else:
        raise AssertionError(
            "obs.jr w/o pot= does not raise AttributeError before the orbit was integrated"
        )
    acfso = numpy.array(
        [
            as_numpy(obs.jr(pot=lp, type=type)),
            as_numpy(obs.jp(pot=lp, type=type)),
            as_numpy(obs.jz(pot=lp, type=type)),
            as_numpy(obs.Or(pot=lp, type=type)),
            as_numpy(obs.Op(pot=lp, type=type)),
            as_numpy(obs.Oz(pot=lp, type=type)),
            as_numpy(obs.wr(pot=lp, type=type)),
            as_numpy(obs.wp(pot=lp, type=type)),
            as_numpy(obs.wz(pot=lp, type=type)),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-16.0, (
        "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface"
    )
    assert (
        numpy.abs(as_numpy(obs.Tr(pot=lp, type=type)) - 2.0 * numpy.pi / acfs[3])
        < 10.0**-16.0
    ), "Orbit.Tr does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(as_numpy(obs.Tp(pot=lp, type=type)) - 2.0 * numpy.pi / acfs[4])
        < 10.0**-16.0
    ), "Orbit.Tp does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(as_numpy(obs.Tz(pot=lp, type=type)) - 2.0 * numpy.pi / acfs[5])
        < 10.0**-16.0
    ), "Orbit.Tz does not agree with actionAngleSpherical frequency"
    assert (
        numpy.abs(as_numpy(obs.TrTp(pot=lp, type=type)) - acfs[4] / acfs[3] * numpy.pi)
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
            as_numpy(obs.jr(type=type)),
            as_numpy(obs.jp(type=type)),
            as_numpy(obs.jz(type=type)),
            as_numpy(obs.Or(type=type)),
            as_numpy(obs.Op(type=type)),
            as_numpy(obs.Oz(type=type)),
            as_numpy(obs.wr(type=type)),
            as_numpy(obs.wp(type=type)),
            as_numpy(obs.wz(type=type)),
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
            as_numpy(obs.jr(pot=lp, type=type)) / ro / vo,
            as_numpy(obs.jp(pot=lp, type=type)) / ro / vo,
            as_numpy(obs.jz(pot=lp, type=type)) / ro / vo,
            as_numpy(obs.Or(pot=lp, type=type)) / vo * ro / 1.0227121655399913,
            as_numpy(obs.Op(pot=lp, type=type)) / vo * ro / 1.0227121655399913,
            as_numpy(obs.Oz(pot=lp, type=type)) / vo * ro / 1.0227121655399913,
            as_numpy(obs.wr(pot=lp, type=type)),
            as_numpy(obs.wp(pot=lp, type=type)),
            as_numpy(obs.wz(pot=lp, type=type)),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-9.0, (
        "Orbit interface for actionAngleSpherical does not return the same as actionAngle interface when using physical coordinates"
    )
    assert (
        numpy.abs(
            as_numpy(obs.Tr(pot=lp, type=type)) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[3]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tr does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(
            as_numpy(obs.Tp(pot=lp, type=type)) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[4]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tp does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(
            as_numpy(obs.Tz(pot=lp, type=type)) / ro * vo * 1.0227121655399913
            - 2.0 * numpy.pi / acfs[5]
        )
        < 10.0**-8.0
    ), (
        "Orbit.Tz does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    assert (
        numpy.abs(as_numpy(obs.TrTp(pot=lp, type=type)) - acfs[4] / acfs[3] * numpy.pi)
        < 10.0**-8.0
    ), (
        "Orbit.TrTp does not agree with actionAngleSpherical frequency when using physical coordinates"
    )
    # Test frequency in km/s/kpc
    assert (
        numpy.abs(as_numpy(obs.Or(pot=lp, type=type, kmskpc=True)) / vo * ro - acfs[3])
        < 10.0**-8.0
    ), (
        "Orbit.Or does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    )
    assert (
        numpy.abs(as_numpy(obs.Op(pot=lp, type=type, kmskpc=True)) / vo * ro - acfs[4])
        < 10.0**-8.0
    ), (
        "Orbit.Op does not agree with actionAngleSpherical frequency when using physical coordinates with km/s/kpc"
    )
    assert (
        numpy.abs(as_numpy(obs.Oz(pot=lp, type=type, kmskpc=True)) / vo * ro - acfs[5])
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
            as_numpy(obs.jr(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.jp(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.jz(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.Or(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.Op(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.Oz(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.wr(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.wp(pot=MWPotential, type=type, delta=0.71)),
            as_numpy(obs.wz(pot=MWPotential, type=type, delta=0.71)),
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
    jr_orb = as_numpy(obs.jr(pot=MWPotential2014, type="staeckel"))
    assert numpy.fabs(est_delta - obs._aA._delta) < 1e-10, (
        "Directly estimated delta does not agree with Orbit-interface-estimated delta"
    )
    # Compare the interfaces at the delta the Orbit actually used: under --jit the
    # boundary-jitted estimateDeltaStaeckel and the Orbit's internal (eager)
    # estimate can differ by an ulp (CPU-dependent XLA codegen), which the
    # actions amplify to ~5e-11 -- not an interface difference
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=obs._aA._delta)
    acfs = numpy.array(list(aAS.actionsFreqsAngles(obs))).reshape(9)
    type = "staeckel"
    acfso = numpy.array(
        [
            as_numpy(obs.jr(pot=MWPotential2014, type=type)),
            as_numpy(obs.jp(pot=MWPotential2014, type=type)),
            as_numpy(obs.jz(pot=MWPotential2014, type=type)),
            as_numpy(obs.Or(pot=MWPotential2014, type=type)),
            as_numpy(obs.Op(pot=MWPotential2014, type=type)),
            as_numpy(obs.Oz(pot=MWPotential2014, type=type)),
            as_numpy(obs.wr(pot=MWPotential2014, type=type)),
            as_numpy(obs.wp(pot=MWPotential2014, type=type)),
            as_numpy(obs.wz(pot=MWPotential2014, type=type)),
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
        as_numpy(obs.jr(pot=tp, type="staeckel"))
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "second derivatives" in str(excinfo.value), (
        "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    )
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        as_numpy(obs.jr(pot=sp, type="staeckel"))
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
        as_numpy(obs.jr(pot=tp, type="staeckel"))
        pytest.fail(
            "TwoPowerSphericalPotentialNoR2deriv appears to now have second derivatives, means that it cannot be used to test exceptions based on not having the second derivatives any longer"
        )
    assert "second derivatives" in str(excinfo.value), (
        "Estimating delta for potential lacking second derivatives should have failed with a message about the lack of second derivatives"
    )
    # Generic non-axi
    sp = SpiralArmsPotential()
    with pytest.raises(PotentialError) as excinfo:
        as_numpy(obs.jr(pot=sp, type="staeckel"))
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
            as_numpy(obs.jr(pot=MWPotential, type=type)),
            as_numpy(obs.jp(pot=MWPotential, type=type)),
            as_numpy(obs.jz(pot=MWPotential, type=type)),
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
            as_numpy(obs.jr(pot=MWPotential, type=type)),
            as_numpy(obs.jp(pot=MWPotential, type=type)),
            as_numpy(obs.jz(pot=MWPotential, type=type)),
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
            as_numpy(obs.jr(pot=toPlanarPotential(MWPotential), type=type)),
            as_numpy(obs.jp(pot=toPlanarPotential(MWPotential), type=type)),
            as_numpy(obs.jz(pot=toPlanarPotential(MWPotential), type=type)),
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
            as_numpy(obs.jr(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.jp(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.jz(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.Or(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.Op(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.Oz(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.wr(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.wp(pot=MWPotential, type=type, b=0.8)),
            as_numpy(obs.wz(pot=MWPotential, type=type, b=0.8)),
        ]
    )
    maxdev = numpy.amax(numpy.abs(acfs - acfso))
    assert maxdev < 10.0**-13.0, (
        "Orbit interface for actionAngleIsochroneApprox does not return the same as actionAngle interface"
    )
    assert (
        numpy.abs(
            as_numpy(obs.Tr(pot=MWPotential, type=type, b=0.8))
            - 2.0 * numpy.pi / acfso[3]
        )
        < 10.0**-13.0
    ), "Orbit.Tr does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(
            as_numpy(obs.Tp(pot=MWPotential, type=type, b=0.8))
            - 2.0 * numpy.pi / acfso[4]
        )
        < 10.0**-13.0
    ), "Orbit.Tp does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(
            as_numpy(obs.Tz(pot=MWPotential, type=type, b=0.8))
            - 2.0 * numpy.pi / acfso[5]
        )
        < 10.0**-13.0
    ), "Orbit.Tz does not agree with actionAngleIsochroneApprox frequency"
    assert (
        numpy.abs(
            as_numpy(obs.TrTp(pot=MWPotential, type=type, b=0.8))
            - acfso[4] / acfso[3] * numpy.pi
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
        as_numpy(obs.jr(pot=MWPotential2014, type="adiabatic", c=False)),
        as_numpy(obs.jp(pot=MWPotential2014, type="adiabatic", c=False)),
        as_numpy(obs.jz(pot=MWPotential2014, type="adiabatic", c=False)),
        as_numpy(obs.e(pot=MWPotential2014, type="adiabatic", analytic=True, c=False)),
        as_numpy(
            obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True, c=False)
        ),
        as_numpy(
            obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True, c=False)
        ),
        as_numpy(
            obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True, c=False)
        ),
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
        as_numpy(obs.jr(pot=MWPotential2014, type="adiabatic", c=True)),
        as_numpy(obs.jp(pot=MWPotential2014, type="adiabatic", c=True)),
        as_numpy(obs.jz(pot=MWPotential2014, type="adiabatic", c=True)),
        as_numpy(obs.e(pot=MWPotential2014, type="adiabatic", analytic=True, c=True)),
        as_numpy(
            obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True, c=True)
        ),
        as_numpy(
            obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True, c=True)
        ),
        as_numpy(obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True, c=True)),
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
        as_numpy(obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71, c=False)),
        as_numpy(obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71, c=False)),
        as_numpy(obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71, c=False)),
        as_numpy(
            obs.e(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
            )
        ),
        as_numpy(
            obs.zmax(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
            )
        ),
        as_numpy(
            obs.rperi(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
            )
        ),
        as_numpy(
            obs.rap(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=False
            )
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
        as_numpy(obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.Or(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.Op(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.Oz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.wr(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.wp(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(obs.wz(pot=MWPotential2014, type="staeckel", delta=0.71, c=True)),
        as_numpy(
            obs.e(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
            )
        ),
        as_numpy(
            obs.zmax(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
            )
        ),
        as_numpy(
            obs.rperi(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
            )
        ),
        as_numpy(
            obs.rap(
                pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True, c=True
            )
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
        as_numpy(obs.jr(pot=MWPotential2014, type="adiabatic")),
        as_numpy(obs.jp(pot=MWPotential2014, type="adiabatic")),
        as_numpy(obs.jz(pot=MWPotential2014, type="adiabatic")),
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
        as_numpy(obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71)),
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
        as_numpy(obs.jr(pot=MWPotential2014, type="adiabatic")),
        as_numpy(obs.jp(pot=MWPotential2014, type="adiabatic")),
        as_numpy(obs.jz(pot=MWPotential2014, type="adiabatic")),
        as_numpy(obs.e(pot=MWPotential2014, type="adiabatic", analytic=True)),
        as_numpy(obs.zmax(pot=MWPotential2014, type="adiabatic", analytic=True)),
        as_numpy(obs.rperi(pot=MWPotential2014, type="adiabatic", analytic=True)),
        as_numpy(obs.rap(pot=MWPotential2014, type="adiabatic", analytic=True)),
    )
    assert numpy.all(
        numpy.fabs(jr[:, 0] - as_numpy(aAA(obs[:, 0])[0])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jp[:, 0] - as_numpy(aAA(obs[:, 0])[1])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jz[:, 0] - as_numpy(aAA(obs[:, 0])[2])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(e[:, 0] - as_numpy(aAA.EccZmaxRperiRap(obs[:, 0])[0])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(zmax[:, 0] - as_numpy(aAA.EccZmaxRperiRap(obs[:, 0])[1]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rperi[:, 0] - as_numpy(aAA.EccZmaxRperiRap(obs[:, 0])[2]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleAdiabatic does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rap[:, 0] - as_numpy(aAA.EccZmaxRperiRap(obs[:, 0])[3]))
        < 10.0**-10.0
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
        as_numpy(obs.jr(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.jp(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.jz(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.Or(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.Op(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.Oz(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.wr(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.wp(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(obs.wz(pot=MWPotential2014, type="staeckel", delta=0.71)),
        as_numpy(
            obs.e(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True)
        ),
        as_numpy(
            obs.zmax(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True)
        ),
        as_numpy(
            obs.rperi(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True)
        ),
        as_numpy(
            obs.rap(pot=MWPotential2014, type="staeckel", delta=0.71, analytic=True)
        ),
    )
    assert numpy.all(
        numpy.fabs(jr[:, 0] - as_numpy(aAS(obs[:, 0])[0])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jp[:, 0] - as_numpy(aAS(obs[:, 0])[1])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(jz[:, 0] - as_numpy(aAS(obs[:, 0])[2])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omr[:, 0] - as_numpy(aAS.actionsFreqs(obs[:, 0])[3])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omp[:, 0] - as_numpy(aAS.actionsFreqs(obs[:, 0])[4])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(omz[:, 0] - as_numpy(aAS.actionsFreqs(obs[:, 0])[5])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wr[:, 0] - as_numpy(aAS.actionsFreqsAngles(obs[:, 0])[6]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wp[:, 0] - as_numpy(aAS.actionsFreqsAngles(obs[:, 0])[7]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(wz[:, 0] - as_numpy(aAS.actionsFreqsAngles(obs[:, 0])[8]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(e[:, 0] - as_numpy(aAS.EccZmaxRperiRap(obs[:, 0])[0])) < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(zmax[:, 0] - as_numpy(aAS.EccZmaxRperiRap(obs[:, 0])[1]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rperi[:, 0] - as_numpy(aAS.EccZmaxRperiRap(obs[:, 0])[2]))
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(rap[:, 0] - as_numpy(aAS.EccZmaxRperiRap(obs[:, 0])[3]))
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
        as_numpy(obs.jr(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.jp(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.jz(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.Or(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.Op(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.Oz(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.wr(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.wp(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.wz(pot=MWPotential2014, type="staeckel")),
        as_numpy(obs.e(pot=MWPotential2014, type="staeckel", analytic=True)),
        as_numpy(obs.zmax(pot=MWPotential2014, type="staeckel", analytic=True)),
        as_numpy(obs.rperi(pot=MWPotential2014, type="staeckel", analytic=True)),
        as_numpy(obs.rap(pot=MWPotential2014, type="staeckel", analytic=True)),
    )
    # Now do the same with the actionAngle interface
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.71)  # just a dummy delta
    bound_indx = numpy.array([True, False, True, False, True, False])
    assert numpy.all(
        numpy.fabs(
            jr[:, 0] - as_numpy(aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[0])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            jp[:, 0] - as_numpy(aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[1])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            jz[:, 0] - as_numpy(aAS(obs[:, 0], delta=obs._aA._delta[bound_indx])[2])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omr[:, 0]
            - as_numpy(aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[3])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omp[:, 0]
            - as_numpy(aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[4])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            omz[:, 0]
            - as_numpy(aAS.actionsFreqs(obs[:, 0], delta=obs._aA._delta[bound_indx])[5])
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wr[:, 0]
            - as_numpy(
                aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[6]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wp[:, 0]
            - as_numpy(
                aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[7]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            wz[:, 0]
            - as_numpy(
                aAS.actionsFreqsAngles(obs[:, 0], delta=obs._aA._delta[bound_indx])[8]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            e[:, 0]
            - as_numpy(
                aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[0]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            zmax[:, 0]
            - as_numpy(
                aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[1]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            rperi[:, 0]
            - as_numpy(
                aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[2]
            )
        )
        < 10.0**-10.0
    ), (
        "Orbit interface for actionAngleStaeckel does not return the same as actionAngle interface for bound orbit in a collection with an unbound orbit"
    )
    assert numpy.all(
        numpy.fabs(
            rap[:, 0]
            - as_numpy(
                aAS.EccZmaxRperiRap(obs[:, 0], delta=obs._aA._delta[bound_indx])[3]
            )
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
    # Backend-agnostic reductions (identity on numpy): the orbit accessors and
    # inverse outputs are jax/torch arrays under a forced backend.
    ox, ovx, xi, vxi = (
        as_numpy(obs.x(times)),
        as_numpy(obs.vx(times)),
        as_numpy(xi),
        as_numpy(vxi),
    )
    assert numpy.amax(numpy.fabs(ox - xi)) < 10.0**-6.0, (
        "actionAngleHarmonicInverse is not the inverse of actionAngleHarmonic for an example orbit"
    )
    assert numpy.amax(numpy.fabs(ovx - vxi)) < 10.0**-6.0, (
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
    # First calculate frequencies and the initial x,v. as_numpy (identity on
    # numpy) keeps the orbit-integration plumbing below on numpy under a forced
    # backend; aAHI(j, angle) further down still runs the backend for real.
    xvom = [as_numpy(q) for q in aAHI.xvFreqs(j, numpy.array([0.1]))]
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
    # Backend-agnostic reductions (identity on numpy): orbit accessors + inverse
    # outputs are jax/torch arrays under a forced backend.
    ox, ovx, xv0, xv1 = (
        as_numpy(orb.x(ts)),
        as_numpy(orb.vx(ts)),
        as_numpy(xv[0]),
        as_numpy(xv[1]),
    )
    assert numpy.all(numpy.fabs(ox - xv0) < 10.0**tol), (
        "Integrated orbit does not agree with actionAngleHarmmonicInverse orbit in x"
    )
    assert numpy.all(numpy.fabs(ovx - xv1) < 10.0**tol), (
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
    om = [as_numpy(o) for o in aAII.Freqs(jr, jphi, jz)]
    assert (
        numpy.fabs((om[0] - as_numpy(epifreq(ip, rl(ip, jphi)))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - as_numpy(omegac(ip, rl(ip, jphi)))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - as_numpy(verticalfreq(ip, rl(ip, jphi)))) / om[2])
        < 10.0**tol
    ), "Close-to-circular orbit does not have Oz=nu for actionAngleTorus"
    # at Lz=1.5, w/ different potential normalization
    ip = IsochronePotential(normalize=1.2)
    aAII = actionAngleIsochroneInverse(ip=ip)
    jphi = 1.5
    om = [as_numpy(o) for o in aAII.Freqs(jr, jphi, jz)]
    assert (
        numpy.fabs((om[0] - as_numpy(epifreq(ip, rl(ip, jphi)))) / om[0]) < 10.0**tol
    ), "Close-to-circular orbit does not have Or=kappa for actionAngleTorus"
    assert (
        numpy.fabs((om[1] - as_numpy(omegac(ip, rl(ip, jphi)))) / om[1]) < 10.0**tol
    ), "Close-to-circular orbit does not have Ophi=omega for actionAngleTorus"
    assert (
        numpy.fabs((om[2] - as_numpy(verticalfreq(ip, rl(ip, jphi)))) / om[2])
        < 10.0**tol
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
    Or, Op, Oz = (as_numpy(o) for o in aAII.Freqs(jr, jphi, jz))
    # Compute frequency with actionAngleIsochrone
    _, _, _, Ori, Opi, Ozi = (
        as_numpy(o) for o in aAI.actionsFreqs(*aAII(jr, jphi, jz, 0.0, 1.0, 2.0)[:6])
    )
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
    Or, Op, Oz = (as_numpy(o) for o in aAII.Freqs(jr, jphi, jz))
    # Compute frequency with actionAngleIsochrone
    _, _, _, Ori, Opi, Ozi = (
        as_numpy(o) for o in aAI.actionsFreqs(*aAII(jr, jphi, jz, 0.0, 1.0, 2.0)[:6])
    )
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
    # First calculate frequencies and the initial RvR. as_numpy (identity on
    # numpy) keeps the orbit-integration plumbing below on numpy under a forced
    # backend; aAII(...) further down still runs the backend for real.
    RvRom = [
        as_numpy(q)
        for q in aAII.xvFreqs(
            jr, jphi, jz, numpy.array([0.0]), numpy.array([1.0]), numpy.array([2.0])
        )
    ]
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
    # Backend-agnostic reductions (identity on numpy): orbit accessors + inverse
    # outputs are jax/torch arrays under a forced backend.
    oR, ovR, ovT, oz, ovz, ophi = (
        as_numpy(orb.R(ts)),
        as_numpy(orb.vR(ts)),
        as_numpy(orb.vT(ts)),
        as_numpy(orb.z(ts)),
        as_numpy(orb.vz(ts)),
        as_numpy(orb.phi(ts)),
    )
    RvR = [as_numpy(q) for q in RvR]
    assert numpy.all(numpy.fabs(oR - RvR[0]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in R"
    )
    assert numpy.all(numpy.fabs(ovR - RvR[1]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vR"
    )
    assert numpy.all(numpy.fabs(ovT - RvR[2]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vT"
    )
    assert numpy.all(numpy.fabs(oz - RvR[3]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in z"
    )
    assert numpy.all(numpy.fabs(ovz - RvR[4]) < 10.0**tol), (
        "Integrated orbit does not agree with torus orbit in vz"
    )
    assert numpy.all(
        numpy.fabs((ophi - RvR[5] + numpy.pi) % (2.0 * numpy.pi) - numpy.pi) < 10.0**tol
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
                as_numpy(aAII(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii])
                - as_numpy(aAIInu(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii]) * correct_fac[ii]
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
                as_numpy(aAII.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii])
                - as_numpy(aAIInu.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii])
                * correct_fac[ii]
            )
            < 10.0**-8.0
        ), (
            "actionAngleInverse function xvFreqs does not return Quantity with the right value"
        )
    for ii in range(3):
        assert (
            numpy.fabs(
                as_numpy(aAII.Freqs(0.1, 1.1, 0.1)[ii])
                - as_numpy(aAIInu.Freqs(0.1, 1.1, 0.1)[ii])
                * conversion.freq_in_Gyr(vo, ro)
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using a point transform"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using a point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using a point transform"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using the exact point transform"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using the exact point transform"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.nanmax(numpy.fabs(as_numpy(aAVI._nSn))) < 1e-9, (
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
    assert numpy.nanmax(numpy.fabs(as_numpy(aAVI._nSn))) < 1e-6 * numpy.nanmax(
        numpy.fabs(as_numpy(aAVI_nopt._nSn))
    ), (
        "nSn coefficients using the exact point transformation are not orders of magnitude smaller than without a point transformation"
    )
    # Also check the edge case of a grid consisting only of the E=0 torus,
    # for which the point transformation is the identity
    aAVI0 = actionAngleVerticalInverse(
        pot=isopot, nta=32, Es=[0.0], use_pointtransform="exact"
    )
    assert numpy.all(as_numpy(aAVI0._nSn == 0.0)), (
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
    assert numpy.amax(numpy.fabs(as_numpy(x - xf))) < 1e-8, (
        "pt_only evaluation does not agree with the full generating-function evaluation for the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(v - vf))) < 1e-8, (
        "pt_only evaluation does not agree with the full generating-function evaluation for the exact point transformation"
    )
    # Compute energy and check whether it's conserved
    E = evaluatelinearPotentials(isopot, x) + v**2.0 / 2.0
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-9, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using pt_only evaluation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using pt_only evaluation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using the exact point transform and bisection"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using bisection and a point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using bisection and a point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(1.0))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(1.0), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using bisection and a point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
        "Frequency computed using actionAngleVerticalInverse does not agree with that computed by actionAngleVertical when using interpolation"
    )
    # and the two public answers agree EXACTLY: Freqs is the frequency of
    # the trajectories _xvFreqs returns, which is the point of the routing
    j = float(aAVI.J(obs.E(pot=isopot)))
    assert (
        numpy.fabs(as_numpy(float(aAVI.Freqs(j)) - float(aAVI._xvFreqs(j, 0.0)[2])))
        == 0.0
    ), "Freqs and _xvFreqs disagree on the frequency of the same torus"
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-10, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(Ei), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-8, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-8, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and a point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-8, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation and a point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(Ei), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-7, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and a point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-7, (
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
    assert numpy.amax(numpy.fabs(as_numpy(obs.x(times) - xi))) < 10.0**-6.0, (
        "actionAngleVerticalInverse is not the inverse of actionAngleVertical for an example orbit when using interpolation and the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(obs.vx(times) - vxi))) < 10.0**-6.0, (
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
    assert numpy.fabs(as_numpy((Om - Omi) / Om)) < 10.0**tol, (
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
    assert numpy.std(as_numpy(E)) / numpy.mean(as_numpy(E)) < 1e-8, (
        "Energy is not conserved along the actionAngleVerticalInverse torus for the IsothermalDiskPotential when using interpolation and the exact point transformation"
    )
    # Now traverse the orbit at the frequency rate and check against orbit integration
    Om = aAVI.Freqs(aAVI.J(Ei))
    ts = numpy.linspace(0.0, 2.0 * numpy.pi / as_numpy(Om), 1001)
    x, v = aAVI(aAVI.J(Ei), as_numpy(Om) * ts)
    orb = Orbit([x[0], v[0]])
    orb.integrate(ts, isopot)
    assert numpy.amax(numpy.fabs(as_numpy(orb.x(ts) - x))) < 1e-7, (
        "Position does not agree with that of the integrated orbit along the torus of the IsothermalDiskPotential when using interpolation and the exact point transformation"
    )
    assert numpy.amax(numpy.fabs(as_numpy(orb.vx(ts) - v))) < 1e-7, (
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
        # with the forward transformation's accurate actions, frequencies
        # and turning points the torus-mapping iteration converges even at
        # this energy, so the warnings are exercised with a small maxiter
        aAVI = actionAngleVerticalInverse(
            pot=isopot,
            nta=4 * 128,
            Es=[300.0],
            use_pointtransform=False,
            momentum_matched=False,
            maxiter=5,
        )
        # Should raise convergence warnings
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Torus mapping with Newton-Raphson did not converge in 5 iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)"
            )
            if raisedWarning:
                break
        assert raisedWarning, (
            "actionAngleVerticalInverse for large energy should have raised convergence warning, but didn't"
        )
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Torus mapping with bisection did not converge in 5 iterations for energies: 300"
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
    # and fails outright when the turning point cannot be found: an energy
    # above the escape energy of a potential that has one
    from galpy.potential import MWPotential2014, toVerticalPotential

    with pytest.raises(RuntimeError, match="turning point could not be found"):
        actionAngleVerticalInverse(
            pot=toVerticalPotential(MWPotential2014, 1.0), nta=128, Es=[0.1, 5.0]
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
        numpy.median(as_numpy(jr)),
        numpy.median(as_numpy(jp)),
        numpy.median(as_numpy(jz)),
        ar,
        ap,
        az,
    )
    # Backend-agnostic: under a forced backend the orbit accessors and the
    # inverse outputs are jax/torch arrays; bring them to numpy for the
    # numpy.amax/fabs reductions below (identity on numpy).
    oR, ovR, ovT, oz, ovz, ophi = (
        as_numpy(obs.R(times)),
        as_numpy(obs.vR(times)),
        as_numpy(obs.vT(times)),
        as_numpy(obs.z(times)),
        as_numpy(obs.vz(times)),
        as_numpy(obs.phi(times)),
    )
    Ri, vRi, vTi, zi, vzi, phii = (
        as_numpy(Ri),
        as_numpy(vRi),
        as_numpy(vTi),
        as_numpy(zi),
        as_numpy(vzi),
        as_numpy(phii),
    )
    assert numpy.amax(numpy.fabs(oR - Ri)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert (
        numpy.amax(numpy.fabs((ophi - phii + numpy.pi) % (2.0 * numpy.pi) - numpy.pi))
        < 10.0**tol
    ), (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(oz - zi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(ovR - vRi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(ovT - vTi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    assert numpy.amax(numpy.fabs(ovz - vzi)) < 10.0**tol, (
        "actionAngleIsochroneInverse is not the inverse of actionAngleIsochrone for an example orbit"
    )
    return None


# Test that the actions are conserved along an orbit
def check_actionAngle_conserved_actions(
    aA, obs, pot, toljr, toljp, toljz, ntimes=1001, fixed_quad=False, inclphi=False
):
    times = numpy.linspace(0.0, 100.0, ntimes)
    obs.integrate(times, pot, method="dopr54_c")
    if fixed_quad or inclphi:
        args = [obs.R(times), obs.vR(times), obs.vT(times), obs.z(times), obs.vz(times)]
        if inclphi:
            args.append(obs.phi(times))
        js = aA(*args, **({"fixed_quad": True} if fixed_quad else {}))
    else:
        # Test Orbit with multiple objects case, but calling
        js = aA(obs(times))
    # Backend-agnostic: the actions are jax/torch arrays under a forced backend;
    # bring them to a numpy 2d array for the reductions (identity on numpy).
    js = numpy.array([as_numpy(j) for j in js])
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
    # Backend-agnostic: actionsFreqsAngles returns jax/torch arrays under a forced
    # backend; bring them to numpy for the reductions below (identity on numpy).
    acfs = tuple(as_numpy(a) for a in acfs)
    acfs_init = tuple(as_numpy(a) for a in acfs_init)
    ar = as_numpy(dePeriod(numpy.reshape(acfs[6], (1, len(times))))).flatten()
    ap = as_numpy(dePeriod(numpy.reshape(acfs[7], (1, len(times))))).flatten()
    az = as_numpy(dePeriod(numpy.reshape(acfs[8], (1, len(times))))).flatten()
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
    # Backend-agnostic: the actionAngle outputs are jax/torch arrays under a
    # forced backend; bring them to numpy for the reductions below (identity on
    # numpy, so byte-identical there).
    es, zmaxs, rperis, raps = (as_numpy(v) for v in (es, zmaxs, rperis, raps))
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
            numpy.atleast_1d(as_numpy(q))
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
                numpy.atleast_1d(as_numpy(q))
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


def test_actionAngleIsochroneInverse_array_actions():
    # Arrays of actions broadcast against the angles by numpy's rules, as for
    # the other inverses: actions of shape (n,1) with angles of shape (m,)
    # are every angle on every torus, an array of actions at a scalar angle
    # is that angle on every torus, and arrays of one shape are paired; each
    # reproduces the single torus to round-off, frequencies included, also when
    # Kepler's equation goes through the bracketing solve
    import sys

    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    aAII = actionAngleIsochroneInverse(ip=IsochronePotential(amp=1.2, b=0.6))

    def _np(out):
        return numpy.array([as_numpy(q) for q in out])

    rng = numpy.random.default_rng(3)
    n, m = 6, 5
    jr = rng.uniform(0.01, 0.3, n)
    jphi = rng.uniform(0.5, 1.2, n) * rng.choice([-1.0, 1.0], n)
    jz = rng.uniform(0.0, 0.3, n)
    ang = numpy.linspace(0.2, 6.0, m)
    _m = sys.modules["galpy.actionAngle.actionAngleIsochroneInverse"]
    tol = _m._KEPLER_RESID_TOL
    for resid_tol in (tol, -1.0):  # the Halley iteration, the bracketing solve
        try:
            _m._KEPLER_RESID_TOL = resid_tol
            grid = _np(
                aAII.xvFreqs(
                    jr[:, None], jphi[:, None], jz[:, None], ang, 2.0 * ang, ang[::-1]
                )
            )
            fixed = _np(aAII(jr, jphi, jz, ang[1], 2.0 * ang[1], ang[3]))
            paired = _np(aAII(jr[:m], jphi[:m], jz[:m], ang, 2.0 * ang, ang))
            ones = [
                [
                    as_numpy(q)
                    for q in aAII.xvFreqs(
                        jr[ii], jphi[ii], jz[ii], ang, 2.0 * ang, ang[::-1]
                    )
                ]
                for ii in range(n)
            ]
            diag = [
                _np(aAII(jr[ii], jphi[ii], jz[ii], ang[ii], 2.0 * ang[ii], ang[ii]))
                for ii in range(m)
            ]
        finally:
            _m._KEPLER_RESID_TOL = tol
        assert grid.shape == (9, n, m), "(n,1) actions with (m,) angles are not (n,m)"
        assert fixed.shape == (6, n), "Arrays of actions at a scalar angle are not (n,)"
        for ii in range(n):
            assert (
                numpy.amax(numpy.fabs(grid[:6, ii] - numpy.array(ones[ii][:6]))) < 1e-12
            ), "Every angle on every torus differs from the single torus"
            assert (
                numpy.amax(numpy.fabs(grid[6:, ii, 0] - numpy.array(ones[ii][6:])))
                < 1e-12
            ), "The frequencies of an array of tori differ from the single torus's"
        assert numpy.amax(numpy.fabs(fixed - grid[:6, :, 1])) < 1e-12, (
            "Arrays of actions at a scalar angle differ from the same points of the grid"
        )
        assert (
            numpy.amax(
                numpy.fabs(paired[:, :, None] - numpy.array(diag).transpose(1, 0, 2))
            )
            < 1e-12
        ), "Paired arrays of actions and angles differ from the single tori"
    assert _np(aAII.Freqs(jr[:, None], jphi[:, None], jz[:, None])).shape == (
        3,
        n,
        1,
    ), "The frequencies of an array of tori do not keep its shape"
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


def test_actionAngleStaeckel_python_smalllz_belowplane():
    # The Python code's vertical action for orbits with very little angular
    # momentum: the lower turning point in v, close to the symmetry axis, has
    # to be found from points far from the plane on either side of it (it was
    # searched for below the point's own v over a limited range, which missed
    # it for points below the plane and for the smallest angular momenta, and
    # the vertical action then came out as zero or too small)
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    Lz = numpy.array([1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 1e-3])
    z = numpy.array([1.2, 1.2, 1.2, -1.2, -1.2, -1.2])
    R, vR = 0.083 * numpy.ones(6), 0.73 * numpy.ones(6)
    vz = -0.37 * numpy.sign(z)
    aAF = actionAngleStaeckel(pot=MWPotential2014, delta=0.33, c=False)
    aAC = actionAngleStaeckel(pot=MWPotential2014, delta=0.33, c=True)
    jr_f, _, jz_f = (as_numpy(a) for a in aAF(R, vR, Lz / R, z, vz))
    jr_c, _, jz_c = (as_numpy(a) for a in aAC(R, vR, Lz / R, z, vz))
    numpy.testing.assert_allclose(jr_f, jr_c, rtol=1e-7)
    # (the C code's quadrature is the less precise one this close to the axis)
    numpy.testing.assert_allclose(jz_f, jz_c, rtol=1e-4)
    # the mirror image of a point in the plane has the same actions
    numpy.testing.assert_allclose(jz_f[:3], jz_f[:2:-1], rtol=1e-10)
    # and the actions are continuous down to an orbit that reaches the axis
    jr_0, _, jz_0 = (
        as_numpy(a) for a in aAF(R[2:4], vR[2:4], numpy.zeros(2), z[2:4], vz[2:4])
    )
    numpy.testing.assert_allclose(jr_0, jr_f[2:4], rtol=1e-6)
    numpy.testing.assert_allclose(jz_0, jz_f[2:4], rtol=1e-4)
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
        assert (
            numpy.amax(numpy.fabs(as_numpy(aAVI._jaoffset)) / aAVI._js[1:].min()) > 1e-5
        ), "The polynomial point transformation has no action offset to correct"
        for ii in range(1, len(Es)):  # including the top torus
            jtrue = float(as_numpy(aAV(0.0, numpy.sqrt(2.0 * Es[ii]))[0]).ravel()[0])
            assert numpy.fabs(as_numpy(aAVI.J(Es[ii]) / jtrue - 1.0)) < 1e-8, (
                "J(E) does not return the torus's actual action"
            )
            x, v = aAVI(jtrue, angles)
            assert numpy.all(numpy.isfinite(as_numpy(x))) and numpy.all(
                numpy.isfinite(as_numpy(v))
            ), (
                "Requesting the actual action of a torus fails (setup_interp=%s)"
                % setup_interp
            )
            assert (
                numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - jtrue))) / jtrue < 1e-7
            ), (
                "Requesting the actual action of a torus returns a different torus "
                "(setup_interp=%s)" % setup_interp
            )
            assert numpy.isfinite(as_numpy(float(aAVI.Freqs(jtrue)))), (
                "Freqs fails at the actual action"
            )
    # without a point transformation, and with the exact one, there is no
    # offset: the mean auxiliary action IS the torus's action
    for kwargs in (dict(momentum_matched=False), dict(use_pointtransform="exact")):
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, **kwargs)
        assert numpy.all(as_numpy(aAVI._jaoffset == 0.0)), (
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
        [float(as_numpy(aAV(0.0, numpy.sqrt(2.0 * E))[0]).ravel()[0]) for E in Es[1:]]
    )
    for pt_deg in (3, 7, 11):
        aAVI = actionAngleVerticalInverse(
            pot=pot, Es=Es, nta=128, use_pointtransform=True, pt_deg=pt_deg
        )
        assert numpy.amax(numpy.fabs(as_numpy(aAVI._jaoffset[1:])) / jforward) > 1e-8, (
            "The offset is not there to be corrected (degree %d)" % pt_deg
        )
        assert (
            numpy.amax(numpy.fabs(as_numpy(aAVI._js[1:] - jforward)) / jforward) < 1e-8
        ), (
            "The stored action, mean auxiliary action minus the closed-form offset, "
            "does not agree with the forward transformation (degree %d)" % pt_deg
        )
        assert (
            numpy.amax(
                as_numpy(
                    numpy.fabs(
                        as_numpy(
                            numpy.nanmean(aAVI._ja, axis=1) - aAVI._js - aAVI._jaoffset
                        )
                    )
                )
            )
            < 1e-15
        ), "The offset is not the difference it is defined as"
    # and the exact point transformation has no offset
    aAVI = actionAngleVerticalInverse(
        pot=pot, Es=Es, nta=128, use_pointtransform="exact"
    )
    assert numpy.all(as_numpy(aAVI._jaoffset == 0.0)), (
        "The exact point transformation has an action offset"
    )
    assert (
        numpy.amax(numpy.fabs(as_numpy(aAVI._js[1:] - jforward)) / jforward) < 1e-8
    ), (
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
            wE = max(wE, numpy.amax(numpy.fabs(as_numpy(H / E - 1.0))))
            wJ = max(wJ, numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - J))) / J)
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
        assert numpy.amax(numpy.fabs(as_numpy(x1 - x2))) < 1e-9, (
            "The map depends on the number of anomaly samples beyond the resolved series"
        )
        assert numpy.amax(numpy.fabs(as_numpy(v1 - v2))) < 1e-9, (
            "The map depends on the number of anomaly samples beyond the resolved series"
        )
    with pytest.raises(ValueError) as excinfo:
        actionAngleVerticalInverse(pot=pot, Es=Es, nta=128, mm_npt=20, mm_nta=64)
    assert "mm_nta" in str(excinfo.value)
    # and the default number of samples is safe whatever nta is
    aAVI3 = actionAngleVerticalInverse(pot=pot, Es=Es, nta=16)
    x3, v3 = aAVI3(aAVI3.J(1.0), angles)
    x2, v2 = aAVI2(aAVI2.J(1.0), angles)
    assert numpy.amax(numpy.fabs(as_numpy(x3 - x2))) < 1e-9, (
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
        assert numpy.amax(numpy.fabs(as_numpy(xf - x))) == 0.0, (
            "__call__ and xvFreqs disagree"
        )
        assert numpy.amax(numpy.fabs(as_numpy(vf - v))) == 0.0, (
            "__call__ and xvFreqs disagree"
        )
        assert Om == aAVI.Freqs(j), "xvFreqs and Freqs disagree"
        jf, Omf, thfwd = aAV.actionsFreqsAngles(x, v)
        dth = numpy.fabs(
            (as_numpy(thfwd) - as_numpy(th) + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        )
        return (
            numpy.amax(numpy.fabs(as_numpy(jf - j))),
            numpy.amax(as_numpy(dth)),
            numpy.amax(numpy.fabs(as_numpy(Omf - Om))) / numpy.mean(as_numpy(Omf)),
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
        assert numpy.amax(numpy.fabs(as_numpy(v))) < 1e-14, (
            "The velocity does not vanish at the turning points: %g"
            % numpy.amax(numpy.fabs(as_numpy(v)))
        )
        assert (
            numpy.amax(
                as_numpy(
                    numpy.fabs(
                        as_numpy(
                            evaluatelinearPotentials(pot, x, use_physical=False) / E
                            - 1.0
                        )
                    )
                )
            )
            < 1e-12
        ), "The turning points do not sit at the energy"
        assert numpy.fabs(as_numpy(x[0] + x[1])) < 1e-14, (
            "The turning points are not symmetric"
        )
        xmax = x[0]
        # smooth through the turning points: x is stationary, v linear
        for delta in (1e-9, 1e-6, 1e-4):
            th = 0.5 * numpy.pi + numpy.array([-delta, 0.0, delta])
            for tth in (th, th + numpy.pi):
                x, v = aAVI(J, tth)
                assert numpy.all(numpy.isfinite(as_numpy(x))) and numpy.all(
                    numpy.isfinite(as_numpy(v))
                ), "The evaluation is not finite next to a turning point"
                assert numpy.amax(numpy.fabs(as_numpy(x - x[1]))) < xmax * delta**2.0, (
                    "The orbit is not stationary in x through a turning point"
                )
                assert numpy.amax(numpy.fabs(as_numpy(v))) < 4.0 * xmax * delta, (
                    "The velocity does not vanish linearly at a turning point"
                )
                assert (
                    numpy.fabs(as_numpy(v[0] + v[2]))
                    < 1e-6 * numpy.fabs(as_numpy(v[0] - v[2])) + 1e-15
                ), "The velocity is not odd about a turning point"
    # the zero-action torus is a point
    x, v = aAVI(0.0, numpy.linspace(0.0, 2.0 * numpy.pi, 11))
    assert numpy.all(as_numpy(x) == 0.0) and numpy.all(as_numpy(v) == 0.0), (
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
        Jn = float(as_numpy(J))
        dev = numpy.array(
            [
                numpy.amax(as_numpy(x)) ** 2.0 * omega0 / (2.0 * Jn) - 1.0,
                numpy.amax(as_numpy(v)) ** 2.0 / (2.0 * Jn * omega0) - 1.0,
                float(as_numpy(aAVI.Freqs(J))) / omega0 - 1.0,
            ]
        )
        assert numpy.amax(numpy.fabs(as_numpy(dev))) < 30.0 * J / J1, (
            "The map does not go over to the harmonic oscillator at small action"
        )
        if prev is not None:
            ratio = numpy.fabs(as_numpy(dev)) / numpy.fabs(as_numpy(prev))
            assert numpy.all(as_numpy(ratio > 5.0)) and numpy.all(
                as_numpy(ratio < 20.0)
            ), "The anharmonic corrections are not linear in the action"
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
            as_numpy(
                numpy.fabs(
                    as_numpy(
                        (xp - xm) / (2.0 * h) * (vJp - vJm) / (2.0 * hj)
                        - (xJp - xJm) / (2.0 * hj) * (vp - vm) / (2.0 * h)
                        - 1.0
                    )
                )
            )
        )

    for nE in (3, 9):
        Es = numpy.linspace(0.0, 2.0, nE)
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, nta=128)
        js = numpy.array([float(as_numpy(aAVI.J(E))) for E in Es])
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
            assert numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - J))) / J > 1e-5, (
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
        assert numpy.all(numpy.isfinite(as_numpy(old.nSn(1.0)))), (
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
            assert numpy.amax(numpy.fabs(as_numpy(jf - J))) / J < 1e-9, (
                "A %d-torus family does not return its torus" % len(tEs)
            )
            assert (
                numpy.amax(
                    as_numpy(
                        numpy.fabs(
                            as_numpy(
                                (as_numpy(thf) - angles + numpy.pi) % (2.0 * numpy.pi)
                                - numpy.pi
                            )
                        )
                    )
                )
                < 1e-9
            ), "A %d-torus family does not return its torus's angles" % len(tEs)
    # the degenerate grid of only the harmonic bottom builds
    bottom = actionAngleVerticalInverse(pot=pot, Es=[0.0], nta=128)
    x, v = bottom(0.0, angles)
    assert numpy.all(as_numpy(x == 0.0)) and numpy.all(as_numpy(v == 0.0)), (
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
    assert numpy.isfinite(as_numpy(Om)) and Om > 0.0, (
        "No frequency between the grid tori"
    )
    assert Om == aAVI.xvFreqs(jm, numpy.array([0.3]))[2], (
        "The off-node frequency is not the map's own"
    )
    # on a node it is the true frequency of the torus
    J = aAVI.J(Es[4])
    x, v = aAVI(J, numpy.array([0.3]))
    assert (
        numpy.fabs(as_numpy(aAVI.Freqs(J) / aAV.actionsFreqs(x, v)[1][0] - 1.0)) < 1e-9
    ), "The on-node frequency is not the torus's"
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
        js = numpy.array([float(as_numpy(aAVI.J(E))) for E in Es])
        # at the nodes the construction preserves the action whatever the
        # tables contain
        for J in js[1:]:
            x, v = aAVI(J, angles)
            assert numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - J))) < 1e-7, (
                "The evaluation leaves the requested torus at a grid node"
            )
        # between EVERY pair of nodes, including the first and the last
        # interval, which are the ones that used to be wrong
        out = 0.0
        for jm in 0.5 * (js[:-1] + js[1:]):
            x, v = aAVI(jm, angles)
            out = max(out, numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - jm))) / jm)
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
            worst = max(worst, numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - J))) / J)
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
    assert numpy.amax(numpy.fabs(as_numpy(xs - xf))) < 1e-12, (
        "A torus depends on the family it is built in"
    )
    assert numpy.amax(numpy.fabs(as_numpy(vs - vf))) < 1e-12, (
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
        assert numpy.fabs(as_numpy(float(aAVI.Freqs(0.0)) / omega0 - 1.0)) < 1e-14, (
            "Freqs at zero action is not the midplane's harmonic frequency"
        )
    return None


def test_actionAngleVerticalInverse_zmax():
    # zmax= sets up nE tori between rest and the torus that reaches zmax,
    # uniform in the speed with which they cross the midplane: the same
    # object as the corresponding energies given by hand, in
    # the momentum-matched and in the mapping modes, whose largest torus
    # reaches zmax, and which cannot be combined with Es=
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential, evaluatelinearPotentials

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    zmax, nE = 1.3, 17
    Es = evaluatelinearPotentials(pot, zmax) * numpy.linspace(0.0, 1.0, nE) ** 2
    tol = 1e-13 if is_backend_array(Es) else 0.0
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 21)
    for momentum_matched in (True, False):
        aAVI = actionAngleVerticalInverse(
            pot=pot,
            zmax=zmax,
            nE=nE,
            setup_interp=True,
            momentum_matched=momentum_matched,
        )
        aAVI_Es = actionAngleVerticalInverse(
            pot=pot, Es=Es, setup_interp=True, momentum_matched=momentum_matched
        )
        Jmax = float(aAVI.J(Es[-1]))
        for J in (0.3 * Jmax, Jmax):
            # exactly on numpy; under a forced backend to round-off, since the
            # mapping mode builds on numpy while Es above came from the backend
            numpy.testing.assert_allclose(
                numpy.array([as_numpy(o) for o in aAVI(J, angles)]),
                numpy.array([as_numpy(o) for o in aAVI_Es(J, angles)]),
                rtol=0.0,
                atol=tol,
                err_msg="zmax= does not set up the tori of the corresponding energies",
            )
        # (the mapping mode's torus is accurate to its generating function)
        assert numpy.fabs(numpy.amax(as_numpy(aAVI(Jmax, angles)[0])) / zmax - 1.0) < (
            1e-12 if momentum_matched else 1e-6
        ), "The largest torus set up with zmax= does not reach zmax"
    with pytest.raises(ValueError) as excinfo:
        actionAngleVerticalInverse(pot=pot, zmax=zmax, Es=Es)
    assert "not both" in str(excinfo.value)
    # without either, the two default tori
    aAVI = actionAngleVerticalInverse(pot=pot)
    aAVI_Es = actionAngleVerticalInverse(pot=pot, Es=[0.1, 0.3])
    for E in (0.1, 0.3):
        assert abs(float(as_numpy(aAVI.J(E))) - float(as_numpy(aAVI_Es.J(E)))) <= tol, (
            "The default tori are not those at E = 0.1 and 0.3"
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
        assert numpy.amax(numpy.fabs(as_numpy(aAVI.E(Jt)) - Et)) < 1e-12, (
            "E(J(E)) does not return the energy"
        )
        assert (
            numpy.amax(
                as_numpy(
                    numpy.fabs(
                        as_numpy(Jt)
                        - numpy.array([float(as_numpy(aAVI.J(E))) for E in Et])
                    )
                )
            )
            == 0.0
        ), "J(E) differs between array and scalar input"
        h = 1e-6
        for J in Jt:
            fd = (aAVI.E(J + h) - aAVI.E(J - h)) / (2.0 * h)
            assert numpy.fabs(as_numpy(aAVI.Freqs(J) / fd - 1.0)) < 1e-6, (
                "Freqs is not the derivative of E(J)"
            )
        if len(Es) == 9:
            for E, J in zip(Et, Jt):
                x, v = aAVI(J, angles)
                H = 0.5 * v**2.0 + evaluatelinearPotentials(pot, x, use_physical=False)
                assert numpy.amax(numpy.fabs(as_numpy(H / E - 1.0))) < 1e-5, (
                    "The interpolated torus does not have the requested energy"
                )
                assert numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - J))) / J < 1e-5, (
                    "The interpolated torus does not have the requested action"
                )
            # at a grid energy the interpolant returns the grid torus
            assert numpy.fabs(as_numpy(aAVI.E(aAVI.J(1.0)) - 1.0)) < 1e-14
            x, v = aAVI(aAVI.J(1.0), angles)
            assert numpy.amax(numpy.fabs(as_numpy(aAV(x, v)[0] - aAVI.J(1.0)))) < 1e-10
    return None


# Midplane orbits with no vertical momentum stay in the plane: zmax = 0 and
# jz = 0 exactly. The C vmin solve used to land on a ~1e-8 root of W's
# rounding noise at the double root v = pi/2 (zmax 1.9e-8 at R = 1 + 1e-15),
# so whether a circular orbit got zmax = 0 depended on the last bits.
@pytest.mark.parametrize("R", [0.9, 0.99, 1.0, 1.0 + 1e-15, 1.0 + 1e-12, 1.01, 1.1])
def test_actionAngleStaeckel_c_midplane_zmax_zero(R):
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential, vcirc

    aAS = actionAngleStaeckel(pot=MWPotential, delta=0.71, c=True, useu0=True)
    o = Orbit([R, 0.0, vcirc(MWPotential, R), 0.0, 0.0])
    _, tzmax, _, _ = aAS.EccZmaxRperiRap(o)
    assert numpy.fabs(tzmax) < 1e-16, f"zmax = {tzmax} for a midplane orbit"
    assert numpy.fabs(aAS(o)[2]) < 1e-16, "jz != 0 for a midplane orbit"


# ---------- actionAngleSphericalInverse tests: the momentum-matched canonical
# ---------- map for spherical potentials (canonical.tex, the spherical case)


def _spherical_inverse_potential():
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(normalize=1.0)


def test_actionAngleVerticalInverse_array_actions():
    # Arrays of actions and angles broadcast against each other, so that a
    # population of stars, each on its own torus, is mapped in one call; the
    # result agrees with evaluating every torus on its own, including the
    # zero-action torus (the bottom of the potential) and repeated tori, and
    # a single action keeps its scalar frequency
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.backend import is_backend_array
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    aAVI = actionAngleVerticalInverse(
        pot=pot, Es=numpy.linspace(0.0, 2.0, 17), nta=128, setup_interp=True
    )
    rng = numpy.random.default_rng(3)
    Jmax = float(aAVI.J(2.0))
    J = rng.uniform(0.0, Jmax, 40)
    J[::7] = 0.0
    J[3] = J[10]
    angles = rng.uniform(0.0, 2.0 * numpy.pi, 40)
    x, v, Om = (as_numpy(t) for t in aAVI.xvFreqs(J, angles))
    assert x.shape == (40,) and v.shape == (40,) and Om.shape == (40,), (
        "Array actions do not return arrays of the actions' shape"
    )
    for ii in range(40):
        xi, vi, Omi = (as_numpy(t) for t in aAVI.xvFreqs(J[ii], angles[ii]))
        assert numpy.fabs(x[ii] - xi[0]) < 1e-12, (
            "x of an array of actions differs from the torus's own"
        )
        assert numpy.fabs(v[ii] - vi[0]) < 1e-12, (
            "v of an array of actions differs from the torus's own"
        )
        assert numpy.fabs(Om[ii] - Omi) < 1e-12, (
            "The frequency of an array of actions differs from the torus's own"
        )
    x2, v2 = (as_numpy(t) for t in aAVI(J, angles))
    assert numpy.all(x2 == x) and numpy.all(v2 == v), (
        "__call__ with array actions differs from xvFreqs"
    )
    assert numpy.all(as_numpy(aAVI.Freqs(J)) == Om), (
        "Freqs with array actions differs from xvFreqs"
    )
    # (5, 1) actions against (7,) angles
    x3, v3, Om3 = (as_numpy(t) for t in aAVI.xvFreqs(J[:5, None], angles[None, :7]))
    assert x3.shape == (5, 7) and Om3.shape == (5, 7), (
        "Actions and angles do not broadcast against each other"
    )
    for ii in range(5):
        xi, vi, Omi = (as_numpy(t) for t in aAVI.xvFreqs(J[ii], angles[:7]))
        assert numpy.amax(numpy.fabs(x3[ii] - xi)) < 1e-12, (
            "Broadcast actions and angles differ from the torus's own"
        )
        assert numpy.amax(numpy.fabs(v3[ii] - vi)) < 1e-12, (
            "Broadcast actions and angles differ from the torus's own"
        )
        assert numpy.all(Om3[ii] == Omi), (
            "Broadcast actions and angles differ from the torus's own"
        )
    # a single action with many angles keeps its float frequency
    # (a 0-d array under a forced backend, whose results stay on the backend)
    x4, v4, Om4 = aAVI.xvFreqs(J[1], angles)
    assert (
        isinstance(Om4, float) or (is_backend_array(Om4) and Om4.ndim == 0)
    ) and x4.shape == (40,), "A single action does not return a float frequency"
    # many actions at one angle
    x5, v5, Om5 = (as_numpy(t) for t in aAVI.xvFreqs(J[:4], 1.0))
    assert x5.shape == (4,) and Om5.shape == (4,), (
        "Array actions at one angle do not return arrays of the actions' shape"
    )
    for ii in range(4):
        assert numpy.fabs(x5[ii] - as_numpy(aAVI.xvFreqs(J[ii], 1.0)[0])[0]) < 1e-12, (
            "Array actions at one angle differ from the torus's own"
        )
    # a negative action raises
    with pytest.raises(ValueError):
        aAVI(numpy.array([0.1, -0.1]), angles[:2])
    # a call over many points is evaluated in chunks: the same answer
    import galpy.actionAngle.actionAngleInverseUtil as util

    chunk = util._EVAL_CHUNK
    util._EVAL_CHUNK = 7
    try:
        x7, v7, Om7 = (as_numpy(t) for t in aAVI.xvFreqs(J, angles))
    finally:
        util._EVAL_CHUNK = chunk
    assert numpy.all(x7 == x) and numpy.all(v7 == v) and numpy.all(Om7 == Om), (
        "Evaluation in chunks differs from the evaluation in one piece"
    )
    # a family of a single torus takes arrays too
    aAVI1 = actionAngleVerticalInverse(pot=pot, Es=[1.0], nta=128, setup_interp=True)
    J1 = float(aAVI1.J(1.0)) * numpy.array([0.9, 1.0, 1.1])
    x6, v6, Om6 = (as_numpy(t) for t in aAVI1.xvFreqs(J1, angles[:3]))
    for ii in range(3):
        xi, vi, Omi = (as_numpy(t) for t in aAVI1.xvFreqs(J1[ii], angles[ii]))
        assert (
            numpy.fabs(x6[ii] - xi[0]) < 1e-12 and numpy.fabs(v6[ii] - vi[0]) < 1e-12
        ), "Array actions on a one-torus family differ from the torus's own"
        assert numpy.fabs(Om6[ii] - Omi) < 1e-12, (
            "Array actions on a one-torus family differ from the torus's own"
        )
    return None


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


def _float_or_0d(x):
    # a scalar result: a float on numpy, a 0-d array under a forced backend
    return isinstance(x, float) or (is_backend_array(x) and x.ndim == 0)


def _spherical_inverse_npt(out):
    # an evaluation's outputs as numpy (GPU tensors under a forced backend)
    return tuple(as_numpy(t) for t in out)


def _spherical_inverse_np(t):
    # a galpy value as a numpy float (a GPU tensor under a forced backend)
    return float(as_numpy(t))


def _spherical_inverse_xv(aAI, *args):
    # an evaluation as one numpy array
    return numpy.array([as_numpy(t) for t in aAI(*args)])


def _spherical_inverse_forward_jr(E, L):
    # the forward transformation's own J_r of a point on the (E, L) torus
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import evaluatePotentials, rl

    pot = _spherical_inverse_potential()
    r0 = float(as_numpy(rl(pot, L, use_physical=False)))
    Phi0 = float(as_numpy(evaluatePotentials(pot, r0, 0.0)))
    vR = numpy.sqrt(2.0 * (E - Phi0) - L**2 / r0**2)
    return float(
        as_numpy(actionAngleSpherical(pot=pot)(r0, vR, L / r0, 0.0, 0.0, 0.0)[0])[0]
    )


def _spherical_inverse_roundtrip(aAI, jr, jphi, jz, angler, anglephi, anglez):
    # evaluate the public map and pass the result through the forward
    # transformation: (action, angle, frequency) errors and the energy spread
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import evaluatePotentials

    pot = _spherical_inverse_potential()
    jr = float(as_numpy(jr))
    R, vR, vT, z, vz, phi = (
        as_numpy(t) for t in aAI(jr, jphi, jz, angler, anglephi, anglez)
    )
    f = [
        as_numpy(t)
        for t in actionAngleSpherical(pot=pot).actionsFreqsAngles(R, vR, vT, z, vz, phi)
    ]
    Om = [float(as_numpy(o)) for o in aAI.Freqs(jr, jphi, jz)]
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
    H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
    return dJ, dth, dOm, numpy.ptp(H) / numpy.fabs(numpy.mean(H))


def _spherical_inverse_symplectic_defect(xvmap, jr, jphi, jz, ar, ap, az, h=1e-6):
    # max |A^T Omega A - Omega| of the finite-difference 6x6 Jacobian of the
    # public map (theta, J) -> (q, p), q = (R, z, phi), p = (v_R, v_z, R v_phi)
    def qp(jr, jphi, jz, ar, ap, az):
        R, vR, vT, z, vz, phi = (as_numpy(t) for t in xvmap(jr, jphi, jz, ar, ap, az))
        return numpy.array([R[0], z[0], phi[0], vR[0], vz[0], R[0] * vT[0]])

    # as floats: a backend scalar's += in the steps below would be in place,
    # moving the point shared by every step
    x = [float(as_numpy(v)) for v in (ar, ap, az, jr, jphi, jz)]
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
        jr = float(as_numpy(aAI.Jr(E, L)))
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
        jr, jphi, jz = float(as_numpy(aAI.Jr(E, L))), 0.6 * L, 0.4 * L
        Om = [float(as_numpy(o)) for o in aAI.Freqs(jr, jphi, jz)]
        ts = numpy.linspace(0.0, 10.0 * 2.0 * numpy.pi / Om[0], 1001)
        R, vR, vT, z, vz, phi = (
            as_numpy(t)
            for t in aAI(
                jr, jphi, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
            )
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(pot(R, z))
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-10, (
            "Energy is not conserved along an actionAngleSphericalInverse torus at a node: %g"
            % (numpy.std(H) / numpy.fabs(numpy.mean(H)))
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, as_numpy(orb.R(ts))),
            ("z", z, as_numpy(orb.z(ts))),
            ("vR", vR, as_numpy(orb.vR(ts))),
            ("vT", vT, as_numpy(orb.vT(ts))),
            ("vz", vz, as_numpy(orb.vz(ts))),
            ("phi", phi, phi + wrap(as_numpy(orb.phi(ts)) - phi)),
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
    Lgrid = numpy.geomspace(0.7, 1.4, 8)
    # (u, L) = (3/7, Lgrid[4]) is a node of this grid (u runs from the
    # circular edge, 0, to 1 in eight steps, and L is spaced in the
    # logarithm by default), where the action, the angles,
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
        assert numpy.fabs(float(as_numpy(aAI.Jr(E, L))) / jrf - 1.0) < tolJ, (
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
            numpy.array([as_numpy(o) for o in aAI.Freqs(jrf, jphi, jz)])
            == numpy.array(
                [as_numpy(o) for o in aAI.xvFreqs(jrf, jphi, jz, 0.3, 1.0, 2.0)[6:]]
            )
        ), "Freqs and xvFreqs disagree"
    return None


def test_actionAngleSphericalInverse_turning_point_derivatives():
    # The closed-form derivatives of the radial turning points with respect
    # to E at fixed L and to L at fixed E (the level-set rule on the
    # effective potential, which the family's Hermite constraints use) agree
    # with finite differences of the turning points themselves
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.backend import use

    # the numpy closed forms (the backend's are tested with the backends)
    with use("numpy", force=True):
        aAI = actionAngleSphericalInverse(
            pot=_spherical_inverse_potential(), Es=[0.9, 1.5], Ls=[0.9, 0.75]
        )
    h = 1e-6
    for E, L in ((0.9, 0.9), (1.5, 0.75)):
        for q in range(2):

            def tp(E, L):
                with use("numpy", force=True):
                    return aAI._turning_points(E, L)[q]

            r = tp(E, L)
            with use("numpy", force=True):
                dE, dL = aAI._turning_point_derivs(r, E, L)
            fdE = (tp(E + h, L) - tp(E - h, L)) / (2.0 * h)
            fdL = (tp(E, L + h) - tp(E, L - h)) / (2.0 * h)
            assert numpy.fabs(dE - fdE) < 1e-7 * (1.0 + numpy.fabs(fdE)), (
                "The turning point's E-derivative is not the level-set rule's"
            )
            assert numpy.fabs(dL - fdL) < 1e-7 * (1.0 + numpy.fabs(fdL)), (
                "The turning point's L-derivative is not the level-set rule's"
            )
    return None


def test_actionAngleSphericalInverse_second_order_tables():
    # An explicit torus's one-node family is quadratic: the second
    # derivatives of its turning points and anomaly-map coefficients in
    # (E, L), and the derivatives of its frequencies, all computed from the
    # torus alone, agree with finite differences of the first-order tables
    # across explicit tori built at E +- h and L +- h with the auxiliary
    # held fixed (a one-node family fits its own isochrone, which would
    # otherwise move between the tori)
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import IsochronePotential

    pot = _spherical_inverse_potential()
    h = 1e-4
    for E, L in ((0.9, 0.9), (1.1, 0.7)):
        base = actionAngleSphericalInverse(
            pot=pot, Es=[E], Ls=[L], mm_npt=48, progressbar=False
        )
        aux = IsochronePotential(amp=base._GM, b=base._b)

        def tables(dE, dL):
            inst = actionAngleSphericalInverse(
                pot=pot,
                Es=[E + dE],
                Ls=[L + dL],
                mm_npt=48,
                progressbar=False,
                auxiliary=aux,
            )
            return numpy.concatenate(
                [
                    [
                        as_numpy(d)
                        for d in inst._turning_point_derivs(
                            inst._rps[0], E + dE, L + dL
                        )
                    ],
                    [
                        as_numpy(d)
                        for d in inst._turning_point_derivs(
                            inst._ras[0], E + dE, L + dL
                        )
                    ],
                    [as_numpy(inst._OmRs[0]), as_numpy(inst._Ompsis[0])],
                    as_numpy(inst._dDm_dEs[0]),
                    as_numpy(inst._dDm_dLs[0]),
                ]
            )

        fdE = (tables(h, 0.0) - tables(-h, 0.0)) / (2.0 * h)
        fdL = (tables(0.0, h) - tables(0.0, -h)) / (2.0 * h)
        d2rp, d2ra, D2 = (
            as_numpy(t[0]) for t in (base._d2rps, base._d2ras, base._d2Dms)
        )
        dOmR, dOmpsi = as_numpy(base._dOmRs[0]), as_numpy(base._dOmpsis[0])
        anaE = numpy.concatenate(
            [
                [d2rp[0], d2rp[1], d2ra[0], d2ra[1], dOmR[0], dOmpsi[0]],
                D2[:, 0],
                D2[:, 1],
            ]
        )
        anaL = numpy.concatenate(
            [
                [d2rp[1], d2rp[2], d2ra[1], d2ra[2], dOmR[1], dOmpsi[1]],
                D2[:, 1],
                D2[:, 2],
            ]
        )
        for fd, ana, name in ((fdE, anaE, "E"), (fdL, anaL, "L")):
            scale = numpy.maximum(numpy.fabs(ana), numpy.amax(numpy.fabs(ana)) * 1e-2)
            assert numpy.amax(numpy.fabs(fd - ana) / scale) < 1e-5, (
                "The second-order tables' %s-derivatives are not the finite "
                "differences of the first-order ones: %g"
                % (name, numpy.amax(numpy.fabs(fd - ana) / scale))
            )
        # the chain to the actions is consistent with the frequencies
        q = base._quadratic_node(0)
        OmR0 = float(as_numpy(base._OmRs[0]))
        assert (
            numpy.fabs(float(as_numpy(q["Om"][0])) - OmR0) < 1e-14
            and numpy.fabs(float(as_numpy(q["d2E"][0])) - dOmR[0] * OmR0) < 1e-14
        ), "The quadratic one-node family's chain to the actions is inconsistent"
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
    rc = float(as_numpy(rl(pot, L)))
    kappa, Omc = float(as_numpy(epifreq(pot, rc))), 1.0 / rc
    circ = numpy.array(
        [as_numpy(t) for t in aAI(0.0, jphi, jz, 0.3, 1.0, 2.0)]
    ).flatten()
    for jr in (1e-16, 1e-12, 1e-8):
        Om = numpy.array([as_numpy(o) for o in aAI.Freqs(jr, jphi, jz)])
        out = numpy.array(
            [as_numpy(t) for t in aAI(jr, jphi, jz, 0.3, 1.0, 2.0)]
        ).flatten()
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
    # the message states how many harmonics the torus needs: about six times
    # the square root of its ratio of apocentre to pericentre
    with pytest.raises(
        RuntimeError,
        match="not bound in the fitted auxiliary.*about six times the square "
        "root of that ratio in harmonics",
    ):
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
    f_ = _spherical_inverse_np
    L = numpy.geomspace(0.1 * f_(vcirc(pot, 0.1)), 3.0 * f_(vcirc(pot, 3.0)), 8)[4]
    Emax = f_(evaluatePotentials(pot, 20.0, 0.0))
    rc = f_(rl(pot, L))
    Ec = f_(evaluatePotentials(pot, rc, 0.0)) + L**2 / (2.0 * rc**2)
    jr_c = aAI.Jr(Ec, L)
    # (exactly 0 on numpy; jax's interpolation weights at the logarithmic node
    # leave ~1e-48)
    assert numpy.fabs(f_(jr_c)) <= (1e-30 if is_backend_array(jr_c) else 0.0), (
        "The circular orbit is not the bottom of the grid"
    )
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Jr(Emax + 1e-6, L)
    # the top cell's midpoint in the grid's own variable: the apocentre of
    # the radial orbit at the geometric mean of those at the top two rows
    r0 = lambda E: brentq(lambda r: f_(evaluatePotentials(pot, r, 0.0)) - E, rc, 20.0)
    r0c = r0(Ec)
    x = 0.5 * ((14.0 / 15.0) ** 2 + 1.0)
    E = f_(evaluatePotentials(pot, r0c * (20.0 / r0c) ** x, 0.0))
    jr = float(as_numpy(aAI.Jr(E, L)))
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi, anglez = 0.3 + 1.7 * angler, 0.7 + 2.3 * angler
    R, vR, vT, z, vz, phi = aAI(jr, 0.6 * L, 0.4 * L, angler, anglephi, anglez)
    f = [as_numpy(t) for t in aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)]
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
        R, vR, vT, z, vz, phi = (
            as_numpy(t)
            for t in aAI(0.0, jphi, L - numpy.fabs(jphi), angler, anglephi, anglez)
        )
        rc = float(as_numpy(rc))
        r = numpy.sqrt(R**2 + z**2)
        Ltot = numpy.sqrt((R * vT) ** 2 + (z * vT) ** 2 + (R * vz - z * vR) ** 2)
        E = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
        Ec = float(as_numpy(evaluatePotentials(pot, rc, 0.0))) + L**2 / (2.0 * rc**2)
        return max(
            numpy.amax(numpy.fabs(r / rc - 1.0)),
            numpy.amax(numpy.fabs((R * vR + z * vz) / rc)),
            numpy.amax(numpy.fabs(R * vT - jphi)),
            numpy.amax(numpy.fabs(Ltot - L)),
            numpy.amax(numpy.fabs(E - Ec)),
        )

    f_ = _spherical_inverse_np
    for L, jphi in ((0.99, 0.7 * 0.99), (0.99, -0.4 * 0.99), (1.3, 1.3)):
        rc = f_(rl(pot, L, use_physical=False))
        Ec = f_(evaluatePotentials(pot, rc, 0.0)) + L**2 / (2.0 * rc**2)
        assert f_(aAI.Jr(Ec, L)) == 0.0, (
            "The circular orbit's radial action is not zero"
        )
        Om = [f_(o) for o in aAI.Freqs(0.0, jphi, L - numpy.fabs(jphi))]
        assert (
            numpy.fabs(Om[0] / f_(epifreq(pot, rc, use_physical=False)) - 1.0) < 1e-12
        ), "The circular orbit's radial frequency is not the epicycle frequency"
        assert (
            numpy.fabs(Om[2] / f_(omegac(pot, rc, use_physical=False)) - 1.0) < 1e-12
        ), "The circular orbit's vertical frequency is not the circular frequency"
        assert invariants(aAI, L, jphi) < 1e-13, (
            "The circular orbit is not at the circular radius with no radial "
            "motion and the requested angular momenta and energy"
        )
    # continuity: the map at tiny J_r approaches the circular orbit as its
    # amplitude, sqrt(2 J_r / kappa)
    L, jphi = 0.99, 0.7 * 0.99
    rc = f_(rl(pot, L, use_physical=False))
    circ = _spherical_inverse_xv(aAI, 0.0, jphi, L - jphi, angler, anglephi, anglez)
    for jr in (1e-6, 1e-8):
        near = _spherical_inverse_xv(aAI, jr, jphi, L - jphi, angler, anglephi, anglez)
        amp = numpy.sqrt(2.0 * jr / f_(epifreq(pot, rc, use_physical=False))) / rc
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
    rc = f_(rl(pot, L, use_physical=False))
    Ec = f_(evaluatePotentials(pot, rc, 0.0)) + L**2 / (2.0 * rc**2)
    for Es in ([Ec, 0.9], [Ec]):
        aAD = actionAngleSphericalInverse(pot=pot, Es=Es, Ls=[L] * len(Es))
        assert f_(aAD.Jr(Ec, L)) == 0.0
        assert invariants(aAD, L, 0.63) < 1e-13, (
            "A discrete family's circular torus is not the circular orbit"
        )
        assert (
            numpy.fabs(
                f_(aAD.Freqs(0.0, 0.63, L - 0.63)[0])
                / f_(epifreq(pot, rc, use_physical=False))
                - 1.0
            )
            < 1e-12
        )
    return None


def test_actionAngleSphericalInverse_array_actions(spherical_inverse_interp):
    # Arrays of actions and angles broadcast against each other: a population
    # of stars on distinct tori, circular orbits among them, is mapped in one
    # call and agrees with evaluating each torus on its own; a single torus
    # keeps its scalar frequencies; the first action outside the family is
    # named
    aAI = spherical_inverse_interp
    rng = numpy.random.default_rng(4)
    n = 30
    L = rng.uniform(aAI._Lgrid[0], aAI._Lgrid[-1], n)
    jrmax = numpy.array([float(as_numpy(aAI.Jr(aAI._Emax, tL))) for tL in L])
    jr = rng.uniform(0.0, 1.0, n) * jrmax
    jr[::8] = 0.0
    jphi = rng.uniform(-1.0, 1.0, n) * L
    jz = L - numpy.fabs(jphi)
    angler, anglephi, anglez = rng.uniform(0.0, 2.0 * numpy.pi, (3, n))
    out = _spherical_inverse_npt(aAI.xvFreqs(jr, jphi, jz, angler, anglephi, anglez))
    assert len(out) == 9 and all(o.shape == (n,) for o in out), (
        "Array actions do not return nine arrays of the actions' shape"
    )
    for ii in range(n):
        outi = _spherical_inverse_npt(
            aAI.xvFreqs(jr[ii], jphi[ii], jz[ii], angler[ii], anglephi[ii], anglez[ii])
        )
        for kk in range(6):
            assert numpy.fabs(out[kk][ii] - outi[kk][0]) < 1e-9 * (
                1.0 + numpy.fabs(outi[kk][0])
            ), "Array actions differ from the torus's own evaluation"
        for kk in range(6, 9):
            assert numpy.fabs(out[kk][ii] - outi[kk]) < 1e-9 * (
                1.0 + numpy.fabs(outi[kk])
            ), "Frequencies of array actions differ from the torus's own"
    out2 = _spherical_inverse_npt(aAI(jr, jphi, jz, angler, anglephi, anglez))
    assert all(numpy.all(out2[kk] == out[kk]) for kk in range(6)), (
        "__call__ with array actions differs from xvFreqs"
    )
    fr = _spherical_inverse_npt(aAI.Freqs(jr, jphi, jz))
    assert all(numpy.all(fr[kk] == out[6 + kk]) for kk in range(3)), (
        "Freqs with array actions differs from xvFreqs"
    )
    # (4, 1) actions against (5,) angles
    out3 = _spherical_inverse_npt(
        aAI.xvFreqs(
            jr[:4, None],
            jphi[:4, None],
            jz[:4, None],
            angler[None, :5],
            anglephi[None, :5],
            anglez[None, :5],
        )
    )
    assert out3[0].shape == (4, 5) and out3[6].shape == (4, 5), (
        "Actions and angles do not broadcast against each other"
    )
    for ii in range(4):
        outi = _spherical_inverse_npt(
            aAI.xvFreqs(jr[ii], jphi[ii], jz[ii], angler[:5], anglephi[:5], anglez[:5])
        )
        for kk in range(6):
            assert numpy.amax(numpy.fabs(out3[kk][ii] - outi[kk])) < 1e-9 * (
                1.0 + numpy.amax(numpy.fabs(outi[kk]))
            ), "Broadcast actions and angles differ from the torus's own"
        for kk in range(6, 9):
            assert numpy.all(
                numpy.fabs(out3[kk][ii] - outi[kk])
                < 1e-9 * (1.0 + numpy.fabs(outi[kk]))
            ), "Broadcast actions and angles differ from the torus's own"
    # a call over many points is evaluated in chunks: the same answer, to
    # the round-off of the circular-radius solve, whose route depends on the
    # number of angular momenta it is given
    import galpy.actionAngle.actionAngleInverseUtil as util

    chunk = util._EVAL_CHUNK
    util._EVAL_CHUNK = 7
    try:
        out7 = _spherical_inverse_npt(
            aAI.xvFreqs(jr, jphi, jz, angler, anglephi, anglez)
        )
    finally:
        util._EVAL_CHUNK = chunk
    assert all(
        numpy.all(numpy.fabs(out7[kk] - out[kk]) < 1e-10 * (1.0 + numpy.fabs(out[kk])))
        for kk in range(9)
    ), "Evaluation in chunks differs from the evaluation in one piece"
    # a single torus at many angles keeps its float frequencies
    outs_raw = aAI.xvFreqs(jr[1], jphi[1], jz[1], angler, anglephi, anglez)
    outs = _spherical_inverse_npt(outs_raw)
    # (0-d arrays under a forced backend, whose results stay on the backend)
    assert all(
        isinstance(outs_raw[kk], float)
        or (is_backend_array(outs_raw[kk]) and outs_raw[kk].ndim == 0)
        for kk in range(6, 9)
    ) and outs[0].shape == (n,), "A single torus does not return float frequencies"
    frs_raw = aAI.Freqs(jr[1], jphi[1], jz[1])
    frs = _spherical_inverse_npt(frs_raw)
    assert all(
        isinstance(frs_raw[kk], float)
        or (is_backend_array(frs_raw[kk]) and frs_raw[kk].ndim == 0)
        for kk in range(3)
    ), "A single torus does not return float frequencies"
    # the first action outside the family is named
    Lbad = L.copy()
    Lbad[5] = 10.0 * aAI._Lgrid[-1]
    with pytest.raises(
        ValueError, match="L = %s outside the interpolation grid" % Lbad[5]
    ):
        aAI.Freqs(jr, 0.0 * jphi, Lbad)
    jrbad = jr.copy()
    jrbad[7] = 3.0 * jrmax[7]
    with pytest.raises(ValueError, match="outside the interpolated family's range"):
        aAI.xvFreqs(jrbad, jphi, jz, angler, anglephi, anglez)
    return None


def test_actionAngleSphericalInverse_array_actions_explicit(spherical_inverse_explicit):
    # Arrays of actions on explicit tori: the set-up tori, repeated, at their
    # own angles, agree with evaluating each on its own; an action that is
    # not one of the tori is named
    aAI = spherical_inverse_explicit
    rng = numpy.random.default_rng(5)
    jr = numpy.tile(aAI._jrs, 3)
    L = numpy.tile(aAI._Ls, 3)
    jphi = rng.uniform(-1.0, 1.0, len(L)) * L
    jz = L - numpy.fabs(jphi)
    angler, anglephi, anglez = rng.uniform(0.0, 2.0 * numpy.pi, (3, len(L)))
    out = _spherical_inverse_npt(aAI.xvFreqs(jr, jphi, jz, angler, anglephi, anglez))
    for ii in range(len(L)):
        outi = _spherical_inverse_npt(
            aAI.xvFreqs(jr[ii], jphi[ii], jz[ii], angler[ii], anglephi[ii], anglez[ii])
        )
        for kk in range(6):
            assert numpy.fabs(out[kk][ii] - outi[kk][0]) < 1e-10 * (
                1.0 + numpy.fabs(outi[kk][0])
            ), "Array actions on explicit tori differ from the torus's own evaluation"
        for kk in range(6, 9):
            assert out[kk][ii] == outi[kk], (
                "Frequencies of array actions on explicit tori differ from the torus's own"
            )
    fr = _spherical_inverse_npt(aAI.Freqs(jr, jphi, jz))
    assert all(numpy.all(fr[kk] == out[6 + kk]) for kk in range(3)), (
        "Freqs with array actions differs from xvFreqs"
    )
    with pytest.raises(ValueError, match="is not one of the set-up tori"):
        aAI.Freqs(1.1 * jr, jphi, jz)
    with pytest.raises(ValueError, match="is not one of the set-up tori"):
        aAI.xvFreqs(1.1 * jr, jphi, jz, angler, anglephi, anglez)
    return None


def test_actionAngleSphericalInverse_tau_solve_fallback(spherical_inverse_explicit):
    # With a single Newton step allowed, the anomaly solve falls back to a
    # bracketed root find for every angle that has not converged, and
    # returns the same phase-space points
    from galpy.actionAngle import actionAngleSphericalInverse

    aAI = spherical_inverse_explicit
    aAI1 = actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        Es=[0.7],
        Ls=[0.9],
        progressbar=False,
        maxiter=1,
    )
    jr, L = aAI1._jrs[0], 0.9
    angler = numpy.linspace(0.1, 6.1, 7)
    out = _spherical_inverse_npt(
        aAI.xvFreqs(jr, 0.5, L - 0.5, angler, 0.3 + 0.0 * angler, 1.2 + 0.0 * angler)
    )
    out1 = _spherical_inverse_npt(
        aAI1.xvFreqs(jr, 0.5, L - 0.5, angler, 0.3 + 0.0 * angler, 1.2 + 0.0 * angler)
    )
    for kk in range(6):
        assert numpy.amax(numpy.fabs(out[kk] - out1[kk])) < 1e-10, (
            "The bracketed fallback of the anomaly solve differs from the Newton solve"
        )
    return None


def test_actionAngleSphericalInverse_auxiliary_kepler():
    # A Kepler target has no isochrone scale: the auxiliary's fit runs to its
    # floor in b and stops there, and the torus is still exact, the Kepler
    # torus being the isochrone's at b -> 0
    from galpy.actionAngle import actionAngleSpherical, actionAngleSphericalInverse
    from galpy.potential import KeplerPotential

    kp = KeplerPotential(normalize=1.0)
    aAI = actionAngleSphericalInverse(pot=kp, Es=[-0.4], Ls=[0.9], progressbar=False)
    assert aAI.auxiliary.b < 1e-7, (
        "The auxiliary's scale radius did not run to its floor for a Kepler target"
    )
    assert numpy.fabs(aAI.auxiliary._amp - 1.0) < 1e-6, (
        "The auxiliary's mass is not the Kepler potential's"
    )
    jr = aAI._jrs[0]
    angles = numpy.linspace(0.1, 6.0, 7)
    R, vR, vT, z, vz, phi = aAI(
        jr, 0.5, 0.4, angles, 0.3 + 0.0 * angles, 1.1 + 0.0 * angles
    )
    jrb, jpb, jzb = _spherical_inverse_npt(
        actionAngleSpherical(pot=kp)(R, vR, vT, z, vz, phi)
    )
    assert numpy.amax(numpy.fabs(jrb - jr)) < 1e-10, (
        "The Kepler torus does not round-trip through the forward transformation"
    )
    assert numpy.amax(numpy.fabs(jzb - 0.4)) < 1e-10, (
        "The Kepler torus does not round-trip through the forward transformation"
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
    fbD = _spherical_inverse_xv(aAD, jrD, 0.6, 0.3, angler, 1.0, 2.0)
    ntD = _spherical_inverse_xv(
        spherical_inverse_explicit, jrD, 0.6, 0.3, angler, 1.0, 2.0
    )
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
    fb = _spherical_inverse_xv(
        aAI, jr, 0.6 * L, 0.4 * L, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )
    nt = _spherical_inverse_xv(
        spherical_inverse_interp,
        jr,
        0.6 * L,
        0.4 * L,
        angler,
        0.0 * angler + 1.0,
        0.0 * angler + 2.0,
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


def _staeckel_inverse_xv(aAI, *args):
    # an evaluation as one numpy array (GPU tensors under a forced backend)
    return numpy.array([as_numpy(t) for t in aAI(*args)])


def _staeckel_inverse_forward():
    from galpy.actionAngle import actionAngleStaeckel

    return actionAngleStaeckel(
        pot=_staeckel_inverse_potential(),
        delta=_STAECKEL_INVERSE_DELTA,
        c=True,
        order=200,
    )


@pytest.fixture(scope="module")
def staeckel_inverse_interp():
    # a small (L_z, E, I_3) interpolation grid in a Kuzmin-Kutuzov potential
    from galpy.actionAngle import actionAngleStaeckelInverse

    return actionAngleStaeckelInverse(
        pot=_staeckel_inverse_potential(),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=3.5,
        nE=8,
        nLz=8,
        nI3=8,
        mm_npt=48,
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
    jr, jphi, jz = (float(as_numpy(t)) for t in (jr, jphi, jz))
    R, vR, vT, z, vz, phi = (
        as_numpy(t) for t in aAI(jr, jphi, jz, angler, anglephi, anglez)
    )
    f = [as_numpy(t) for t in aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)]
    Om = [float(as_numpy(o)) for o in aAI.Freqs(jr, jphi, jz)]
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
    H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
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
        jrf, _, jzf = (float(numpy.atleast_1d(as_numpy(x))[0]) for x in aAS(*ic))
        jr, jz = (float(as_numpy(t)) for t in (aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)))
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
        ar, ap, az = (numpy.asarray(as_numpy(out[i])).ravel() for i in (6, 7, 8))
        rec = _staeckel_inverse_xv(
            aAI, aAI.JR(E, Lz, I3), Lz, aAI.Jz(E, Lz, I3), ar, ap, az
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
        jr, jz = (float(as_numpy(a(E, Lz, I3))) for a in (aAI.JR, aAI.Jz))
        Om = [float(as_numpy(o)) for o in aAI.Freqs(jr, Lz, jz)]
        ts = numpy.linspace(0.0, 10.0 * 2.0 * numpy.pi / Om[0], 1001)
        R, vR, vT, z, vz, phi = (
            as_numpy(t)
            for t in aAI(
                jr, Lz, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
            )
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-11, (
            "Energy is not conserved along an actionAngleStaeckelInverse torus at a node: %g"
            % (numpy.std(H) / numpy.fabs(numpy.mean(H)))
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, as_numpy(orb.R(ts))),
            ("z", z, as_numpy(orb.z(ts))),
            ("vR", vR, as_numpy(orb.vR(ts))),
            ("vT", vT, as_numpy(orb.vT(ts))),
            ("vz", vz, as_numpy(orb.vz(ts))),
            ("phi", phi, phi + wrap(as_numpy(orb.phi(ts)) - phi)),
        ):
            assert numpy.amax(numpy.fabs(orbit - torus)) < 1e-7, (
                "%s along an actionAngleStaeckelInverse torus at a node does not agree with the integrated orbit: %g"
                % (name, numpy.amax(numpy.fabs(orbit - torus)))
            )
    return None


def test_actionAngleStaeckelInverse_interpolation(staeckel_inverse_interp):
    # The interpolated family: exact at a node of its own grid, accurate to
    # the family's interpolation error between nodes, its frequencies those
    # of the returned orbits, and (J_R, J_z)(E, L_z, I_3) the forward
    # transformation's
    aAI = staeckel_inverse_interp
    aAS = _staeckel_inverse_forward()
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    # a node of the grid (the fourth energy, third third-integral, and
    # fifth angular momentum), where the actions, the angles, and the
    # frequencies are all at the forward transformation's floor, because
    # the family's first partials are exact there; and a torus between
    # nodes, at the family's interpolation error
    nI, nL = 8, 8
    ii = (4 * nI + 3) * nL + 5
    for (E, Lz, I3), tolJ, tolth, tolOm in (
        ((aAI._Es[ii], aAI._Lzs[ii], aAI._I3s[ii]), 1e-9, 1e-7, 1e-8),
        (_staeckel_inverse_labels([1.05, 0.25, 0.95, 0.2, 0.3, 0.0]), 1e-5, 1e-4, 1e-4),
    ):
        jr, jz = (float(as_numpy(t)) for t in (aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)))
        # the forward transformation's actions of a point on this torus:
        # the inverse's own point, at the family's accuracy
        R, vR, vT, z, vz, phi = (
            numpy.atleast_1d(as_numpy(q))[0] for q in aAI(jr, Lz, jz, 0.7, 1.0, 2.0)
        )
        jrf, _, jzf = (
            float(numpy.atleast_1d(as_numpy(x))[0]) for x in aAS(R, vR, vT, z, vz, phi)
        )
        assert numpy.fabs(jr - jrf) + numpy.fabs(jz - jzf) < tolJ * (jr + jz), (
            "(J_R, J_z)(E, L_z, I_3) of the family is not the forward transformation's"
        )
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aAI, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS
        )
        assert dJ < tolJ, "The map does not return the requested torus: %g" % dJ
        assert dth < tolth, "The map does not return the requested angles: %g" % dth
        assert dOm < tolOm, "The frequencies are not the torus's: %g" % dOm
        assert dH < tolJ, "The reconstructed torus is not at one energy: %g" % dH
        assert numpy.all(
            numpy.array([as_numpy(o) for o in aAI.Freqs(jr, Lz, jz)])
            == numpy.array(
                [as_numpy(o) for o in aAI.xvFreqs(jr, Lz, jz, 0.3, 1.0, 2.0)[6:]]
            )
        ), "Freqs and xvFreqs disagree"
    return None


def test_actionAngleStaeckelInverse_turning_point_derivatives(
    staeckel_inverse_explicit,
):
    # The closed-form derivatives of the turning points (the level-set rule
    # on the separated momenta, which the family's Hermite constraints use)
    # and of the shell u agree with finite differences of the turning points
    # themselves across neighbouring tori
    from galpy.backend import use

    aAI = staeckel_inverse_explicit
    h = 1e-6
    # the closed forms are the numpy path's (a family built on a backend has
    # its own, checked in test_backend_actionAngleStaeckelInverse.py)
    with use("numpy", force=True):
        for key in ("benign", "eccentric"):
            E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[key])
            uc, wu, wv, ush, _, _ = aAI._turning_points(E, Lz, I3)
            dinner = aAI._tp_derivs_u(uc - wu, E, Lz)
            douter = aAI._tp_derivs_u(uc + wu, E, Lz)
            dvm = aAI._tp_derivs_v(0.5 * numpy.pi - wv, E, Lz)
            dush = aAI._dushell(ush, E, Lz)
            for k, (dE, dI, dL) in enumerate(
                ((h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h))
            ):
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
                    assert numpy.fabs(
                        dush[k // 2] - (up[3] - dn[3]) / (2.0 * h)
                    ) < 1e-6 * (1.0 + numpy.fabs(dush[k // 2])), (
                        "The shell u's derivative is not the level-set rule's"
                    )
    return None


def test_actionAngleStaeckelInverse_second_order_tables():
    # An explicit torus's one-node family is quadratic: the second
    # derivatives of its turning points, of its two actions (the period
    # matrix's derivatives) and of both maps' coefficients in (E, I_3, L_z),
    # all computed from the torus alone, agree with finite differences of the
    # first-order tables across explicit tori built at each integral +- h
    # with the auxiliary held fixed (a one-node family fits its own
    # isochrone, which would otherwise move between the tori), and the
    # frequencies' derivatives in the actions, chained through the inverse
    # period matrix, agree with finite differences of the frequencies
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import IsochronePotential

    pot = _staeckel_inverse_potential()
    h = 1e-3
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    for key in ("benign", "eccentric"):
        E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[key])
        base = actionAngleStaeckelInverse(
            pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], progressbar=False
        )
        aux = IsochronePotential(amp=base._GM, b=base._b)

        def tables(da):
            inst = actionAngleStaeckelInverse(
                pot=pot,
                Es=[E + da[0]],
                Lzs=[Lz + da[2]],
                I3s=[I3 + da[1]],
                progressbar=False,
                auxiliary=aux,
            )
            return {
                "sup": inst._dsups[0],
                "M": numpy.linalg.inv(inst._Ns[0])[:2],
                "Om": inst._Ns[0][0],
                "Dmu": inst._dDmus[0],
                "Dmv": inst._dDmvs[0],
            }

        # Richardson-extrapolated central differences, which leave the
        # first-order tables' own round-off as the floor
        fd = []
        for b in range(3):
            da = numpy.zeros(3)
            da[b] = h
            t1, t2 = (tables(f * da) for f in (1.0, 2.0))
            m1, m2 = (tables(-f * da) for f in (1.0, 2.0))
            fd.append(
                {
                    k: (4.0 * (t1[k] - m1[k]) / (2.0 * h) - (t2[k] - m2[k]) / (4.0 * h))
                    / 3.0
                    for k in t1
                }
            )
        for name, ana, tol in (
            ("sup", base._d2sups[0], 1e-5),
            ("M", base._d2Ms[0], 1e-5),
            ("Dmu", base._d2Dmus[0], 1e-5),
            # the v slopes carry the round-off of the thin libration's
            # matching, which the differences amplify
            ("Dmv", base._d2Dmvs[0], 2e-4),
        ):
            scale = numpy.maximum(numpy.fabs(ana), 1e-2 * numpy.amax(numpy.fabs(ana)))
            worst = max(
                numpy.amax(numpy.fabs(fd[j][name][:, i] - ana[:, kk]) / scale[:, kk])
                for kk, (i, j) in enumerate(pairs)
            )
            assert worst < tol, (
                "The second-order %s tables are not the finite differences of "
                "the first-order ones on the %s torus: %g" % (name, key, worst)
            )
        # the frequencies' derivatives in the actions, through the chain:
        # dOmega_i/dalpha_b = sum_j E_ij dJ_j/dalpha_b
        q = base._quadratic_node(0)
        pred = q["d2E"] @ numpy.linalg.inv(base._Ns[0])
        num = numpy.array([fd[b]["Om"] for b in range(3)]).T
        scale = numpy.maximum(numpy.fabs(pred), 1e-2 * numpy.amax(numpy.fabs(pred)))
        assert numpy.amax(numpy.fabs(num - pred) / scale) < 1e-5, (
            "The quadratic one-node family's frequency derivatives are not the "
            "finite differences of the frequencies on the %s torus: %g"
            % (key, numpy.amax(numpy.fabs(num - pred) / scale))
        )
        assert numpy.amax(
            numpy.fabs(q["d2E"] - q["d2E"].T)
        ) < 1e-12 and numpy.array_equal(q["Om"], base._Ns[0][0]), (
            "The quadratic one-node family's chain to the actions is inconsistent"
        )
    # a torus with a degenerate libration keeps zero second-order tables and
    # chains to vanishing derivatives of the degenerate half-width
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    aAI = actionAngleStaeckelInverse(
        pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], progressbar=False
    )
    with aAI._in_chart(E, Lz):
        Ipl = aAI._I3_planar(E, Lz)
    aAP = actionAngleStaeckelInverse(
        pot=pot, Es=[E], Lzs=[Lz], I3s=[Ipl], progressbar=False
    )
    q = aAP._quadratic_node(0)
    assert (
        not numpy.any(aAP._d2Dmvs[0])
        and not numpy.any(q["dwv"])
        and not numpy.any(q["d2wv"])
    ), "A planar orbit's one-node family has derivatives of its v half-width"
    return None


def test_actionAngleStaeckelInverse_symplectic(staeckel_inverse_interp):
    # Manifest canonicity: the symplectic defect of the public map is at the
    # finite-difference floor -- measured on the analytic isochrone inverse
    # with the same harness -- between the nodes of the family, and just as
    # much on a grid so coarse that its interpolation error is large
    from galpy.actionAngle import (
        actionAngleIsochroneInverse,
        actionAngleStaeckelInverse,
    )
    from galpy.potential import IsochronePotential

    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=4.0, b=0.5)),
        0.06,
        0.9,
        0.1,
        0.7,
        1.0,
        2.0,
    )
    aAI = staeckel_inverse_interp
    for jr, Lz, jz in ((0.06, 0.93, 0.10), (0.15, 1.21, 0.03), (0.02, 0.8, 0.2)):
        defect = _spherical_inverse_symplectic_defect(aAI, jr, Lz, jz, 0.7, 1.0, 2.0)
        assert defect < 20.0 * floor + 1e-9, (
            "The symplectic defect between the nodes is above the finite-difference "
            "floor: %g vs %g" % (defect, floor)
        )
    coarse = actionAngleStaeckelInverse(
        pot=_staeckel_inverse_potential(),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=3.5,
        nE=4,
        nLz=4,
        nI3=4,
        mm_npt=48,
    )
    angler = numpy.linspace(0.05, 6.2, 21)
    dJ = _staeckel_inverse_roundtrip(
        coarse, 0.06, 0.93, 0.10, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )[0]
    assert dJ > 1e-5, "The coarse grid is too accurate for this check to mean anything"
    defect = _spherical_inverse_symplectic_defect(
        coarse, 0.06, 0.93, 0.10, 0.7, 1.0, 2.0
    )
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect on the coarse grid is above the floor: %g vs %g, "
        "so canonicity is contingent on the tables" % (defect, floor)
    )
    return None


def test_actionAngleStaeckelInverse_symplectic_perturbed(staeckel_inverse_interp):
    # Manifest canonicity: the map is symplectic for whatever the tables
    # contain, because every derivative it uses is the stored interpolant's
    # own. Perturb the stored values and slopes of a copy of the family by
    # amounts that spoil its accuracy, rebuild the interpolants, and check
    # that the symplectic defect is still at the finite-difference floor
    import copy

    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=4.0, b=0.5)),
        0.06,
        0.9,
        0.1,
        0.7,
        1.0,
        2.0,
    )
    aAP = copy.deepcopy(staeckel_inverse_interp)
    rng = numpy.random.default_rng(3)
    # every table and every slope, where they are not identically zero (the
    # degenerate faces stay degenerate)
    for name in (
        "_A_tab",
        "_A_dx",
        "_A_dy",
        "_A_dL",
        "_U_tab",
        "_U_dwE",
        "_U_dsu",
        "_U_dL",
        "_V_tab",
        "_V_dwE",
        "_V_dsv",
        "_V_dL",
    ):
        tab = getattr(aAP, name)
        mask = tab != 0.0
        tab[mask] *= 1.0 + 1e-3 * rng.uniform(-1.0, 1.0, tab.shape)[mask]
    aAP._rebuild_interp()
    jr, Lz, jz = 0.06, 0.93, 0.10
    angler = numpy.linspace(0.05, 6.2, 21)
    dJ = _staeckel_inverse_roundtrip(
        aAP, jr, Lz, jz, angler, 0.0 * angler + 1.0, 0.0 * angler + 2.0
    )[0]
    assert dJ > 1e-5, "The perturbation of the tables is too small to mean anything"
    defect = _spherical_inverse_symplectic_defect(aAP, jr, Lz, jz, 0.7, 1.0, 2.0)
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect with perturbed tables is above the floor: %g vs %g, "
        "so canonicity is contingent on the tables" % (defect, floor)
    )
    return None


def test_actionAngleStaeckelInverse_small_action(staeckel_inverse_interp):
    # Actions down to round-off above zero resolve to the edges' immediate
    # neighbourhood, not to the edges themselves: the frequencies stay
    # finite and tend to the epicycle, circular, and vertical frequencies,
    # and the point tends to the circular orbit's within the librations'
    # amplitudes; a vanishing action alone puts the torus on the shell or
    # the planar face, where the other action round-trips through the
    # forward transformation
    from galpy.potential import epifreq, rl, verticalfreq

    aAI = staeckel_inverse_interp
    aAS = _staeckel_inverse_forward()
    pot = _staeckel_inverse_potential()
    # a row of the fixture's grid in angular momentum, where the edge's
    # values and slopes are exact
    Lz = float(aAI._Lzgrid[3])
    Rc = float(as_numpy(rl(pot, Lz)))
    kappa, Omc, nu = (
        float(as_numpy(epifreq(pot, Rc))),
        Lz / Rc**2,
        float(as_numpy(verticalfreq(pot, Rc))),
    )
    circ = _staeckel_inverse_xv(aAI, 0.0, Lz, 0.0, 0.3, 1.0, 2.0).flatten()
    for jr, jz in (
        (1e-16, 1e-16),
        (1e-12, 1e-12),
        (1e-8, 0.0),
        (0.0, 1e-8),
        (1e-8, 1e-8),
    ):
        Om = numpy.array([as_numpy(o) for o in aAI.Freqs(jr, Lz, jz)])
        out = _staeckel_inverse_xv(aAI, jr, Lz, jz, 0.3, 1.0, 2.0).flatten()
        assert numpy.all(numpy.isfinite(Om)) and numpy.all(numpy.isfinite(out)), (
            "Tiny actions (J_R, J_z) = ({:g}, {:g}) give non-finite output".format(
                jr, jz
            )
        )
        assert numpy.amax(numpy.fabs(Om / [kappa, Omc, nu] - 1.0)) < 1e-6, (
            "The frequencies at (J_R, J_z) = (%g, %g) are not the epicycle limit's: %s"
            % (jr, jz, Om)
        )
        # the librations' half-widths in the epicycle limit
        amp = numpy.sqrt(2.0 * jr / kappa) + numpy.sqrt(2.0 * jz / nu)
        assert numpy.amax(numpy.fabs(out - circ)) < 3.0 * amp + 1e-9, (
            "The point at (J_R, J_z) = (%g, %g) is not continuous with the circular orbit's"
            % (jr, jz)
        )
    angler = numpy.linspace(0.05, 6.2, 11)
    for jr, jz in ((0.0, 0.05), (0.05, 0.0)):
        R, vR, vT, z, vz, phi = (as_numpy(t) for t in aAI(jr, Lz, jz, angler, 1.0, 2.0))
        f = [as_numpy(t) for t in aAS(R, vR, vT, z, vz, phi)]
        assert (
            numpy.amax(numpy.fabs(f[0] - jr)) < 1e-5
            and numpy.amax(numpy.fabs(f[2] - jz)) < 1e-5
        ), (
            "The torus on the %s face does not round-trip through the forward transformation"
            % ("shell" if jr == 0.0 else "planar")
        )
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
        jr, jz = (float(as_numpy(t)) for t in (aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)))
        dJ = _staeckel_inverse_roundtrip(
            aAI, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS
        )[0]
        assert dJ < 1e-9, (
            "The %s torus's actions do not round-trip through the forward transformation: %g"
            % (key, dJ)
        )
        Om = [float(as_numpy(o)) for o in aAI.Freqs(jr, Lz, jz)]
        ts = numpy.linspace(0.0, 5.0 * 2.0 * numpy.pi / Om[0], 501)
        R, vR, vT, z, vz, phi = (
            as_numpy(t)
            for t in aAI(
                jr, Lz, jz, 0.4 + Om[0] * ts, 1.1 + Om[1] * ts, 2.3 + Om[2] * ts
            )
        )
        H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
        assert numpy.std(H) / numpy.fabs(numpy.mean(H)) < 1e-11, (
            "Energy is not conserved along the %s torus" % key
        )
        orb = Orbit([R[0], vR[0], vT[0], z[0], vz[0], phi[0]])
        orb.integrate(ts, pot, method="dop853_c")
        for name, torus, orbit in (
            ("R", R, as_numpy(orb.R(ts))),
            ("z", z, as_numpy(orb.z(ts))),
            ("vR", vR, as_numpy(orb.vR(ts))),
            ("vz", vz, as_numpy(orb.vz(ts))),
            ("phi", phi, phi + wrap(as_numpy(orb.phi(ts)) - phi)),
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
    with pytest.raises(
        RuntimeError,
        match="not bound in the fitted auxiliary.*at most about six times its "
        "square root",
    ):
        actionAngleStaeckelInverse(
            pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=2, mm_nta=16
        )
    return None


def test_actionAngleStaeckelInverse_nonfinite_tables():
    # A family with tori whose tables are not finite cannot be interpolated,
    # and its set-up says which tori: here the potential's forces are not
    # finite above |z| = 1.2, which the most inclined tori of the grid reach
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import get_namespace
    from galpy.potential import LogarithmicHaloPotential, Potential

    class Holed(Potential):
        def __init__(self):
            Potential.__init__(self, amp=1.0)
            self._lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)

        def _hole(self, z):
            # (in the namespace of z: the family is built on a forced backend)
            xp = get_namespace(z)
            z = xp.asarray(z)
            return xp.where(xp.abs(z) > 1.2, numpy.nan, 1.0)

        def _evaluate(self, R, z, phi=0.0, t=0.0):
            return self._lp(R, z, use_physical=False)

        def _Rforce(self, R, z, phi=0.0, t=0.0):
            return self._lp.Rforce(R, z, use_physical=False) * self._hole(z)

        def _zforce(self, R, z, phi=0.0, t=0.0):
            return self._lp.zforce(R, z, use_physical=False) * self._hole(z)

        def _R2deriv(self, R, z, phi=0.0, t=0.0):
            return self._lp.R2deriv(R, z, use_physical=False) * self._hole(z)

        def _z2deriv(self, R, z, phi=0.0, t=0.0):
            return self._lp.z2deriv(R, z, use_physical=False) * self._hole(z)

        def _Rzderiv(self, R, z, phi=0.0, t=0.0):
            return self._lp.Rzderiv(R, z, use_physical=False) * self._hole(z)

    with pytest.raises(
        ValueError,
        match="The tables of the \\(E, L_z, I_3\\) tori \\(.* are not finite",
    ):
        actionAngleStaeckelInverse(
            pot=Holed(),
            delta=0.8,
            setup_interp=True,
            Rmin=0.7,
            Rmax=1.5,
            Rinf=4.0,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=16,
            progressbar=False,
        )
    return None


def test_actionAngleStaeckelInverse_errors(
    staeckel_inverse_interp, staeckel_inverse_explicit
):
    # every guarded misuse raises informatively
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import (
        IsochronePotential,
        MiyamotoNagaiPotential,
        MWPotential2014,
        OblateStaeckelWrapperPotential,
    )

    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    with pytest.raises(OSError, match="Must specify pot="):
        actionAngleStaeckelInverse()
    with pytest.raises(OSError, match="supplies its focal length"):
        actionAngleStaeckelInverse(pot=MWPotential2014, Es=[E], Lzs=[Lz], I3s=[I3])
    # a potential that supplies its focal length rejects a delta= that
    # disagrees with it and a u0=, and accepts a delta= that agrees
    with pytest.raises(TypeError, match="disagrees"):
        actionAngleStaeckelInverse(pot=pot, delta=1.4, Es=[E], Lzs=[Lz], I3s=[I3])
    with pytest.raises(TypeError, match="disagrees"):
        actionAngleStaeckelInverse(
            pot=OblateStaeckelWrapperPotential(pot=pot, delta=1.3),
            delta=1.4,
            Es=[E],
            Lzs=[Lz],
            I3s=[I3],
        )
    with pytest.raises(TypeError, match="u0= cannot be given"):
        actionAngleStaeckelInverse(
            pot=pot, delta=1.3, u0=1.0, Es=[E], Lzs=[Lz], I3s=[I3]
        )
    assert actionAngleStaeckelInverse(
        pot=pot, delta=1.3, Es=[E], Lzs=[Lz], I3s=[I3]
    ).JR(E, Lz, I3) == actionAngleStaeckelInverse(
        pot=pot, Es=[E], Lzs=[Lz], I3s=[I3]
    ).JR(E, Lz, I3), (
        "A delta= that agrees with the potential's own focal length changes the torus"
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
    # an Rinf that exceeds Rmax but not the apocentre of the planar orbit
    # without angular momentum at the circular energy of the grid's largest
    # L_z, which the energy variable's bottom edge needs inside Rinf
    with pytest.raises(ValueError, match="Rinf is too small"):
        actionAngleStaeckelInverse(
            pot=MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.3),
            delta=0.4,
            setup_interp=True,
            Rmin=0.8,
            Rmax=1.5,
            Rinf=2.5,
            nE=4,
            nLz=4,
            nI3=4,
            progressbar=False,
        )
    with pytest.raises(ValueError, match="mm_npt must be at least 2"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=1)
    with pytest.raises(ValueError, match=">= 4"):
        actionAngleStaeckelInverse(pot=pot, setup_interp=True, nE=3)
    with pytest.raises(ValueError, match="below the circular orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[-10.0], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="unbound"):
        actionAngleStaeckelInverse(pot=pot, Es=[0.1], Lzs=[Lz], I3s=[I3])
    with pytest.raises(ValueError, match="above the shell orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3 + 10.0])
    with pytest.raises(ValueError, match="below the planar orbit's"):
        actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3 - 10.0])
    with pytest.raises(ValueError, match="Rinf"):
        actionAngleStaeckelInverse(
            pot=pot, setup_interp=True, Rmin=0.7, Rmax=1.6, Rinf=1.0
        )
    aAI = staeckel_inverse_interp
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Freqs(0.05, 5.0, 0.05)  # L_z outside the grid
    with pytest.raises(ValueError, match="above the covered"):
        aAI(50.0, 0.93, 0.1, 0.3, 1.0, 2.0)  # J_R outside the family
    with pytest.raises(ValueError, match="above the covered"):
        aAI(0.0, 0.93, 50.0, 0.3, 1.0, 2.0)  # J_z outside the family, on the shell face
    with pytest.raises(ValueError, match="non-negative"):
        aAI(-0.1, 0.93, 0.1, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="L_z = 0"):
        aAI(0.1, 0.0, 0.1, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.JR(E, 5.0, I3)
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.JR(-10.0, 0.93, I3)  # below the circular orbit's energy
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Jz(E, 0.93, I3 + 10.0)  # above the shell orbit's third integral
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI.Freqs(0.0, 5.0, 0.0)  # a circular request outside the grid's L_z range
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


def test_actionAngleStaeckelInverse_circular(staeckel_inverse_interp):
    # The circular orbits are the bottom face of the family's grid, with
    # the epicycle limit's exact partials, so both actions reach zero
    # continuously; the circular orbit itself evaluates in closed form at
    # the circular radius in the plane, at the requested azimuthal angle,
    # with the epicycle, circular, and vertical frequencies, in a grid or
    # as an explicit torus
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import epifreq, evaluatePotentials, omegac, rl, verticalfreq

    pot = _staeckel_inverse_potential()
    aAI = staeckel_inverse_interp
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)

    def invariants(aAI, Lz):
        Rc = float(as_numpy(rl(pot, numpy.fabs(Lz), use_physical=False)))
        R, vR, vT, z, vz, phi = (
            as_numpy(t) for t in aAI(0.0, Lz, 0.0, angler, anglephi, anglez)
        )
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

    for Lz in (0.93, -0.93, 1.3):
        Rc = rl(pot, numpy.fabs(Lz), use_physical=False)
        Ec = evaluatePotentials(pot, Rc, 0.0) + Lz**2 / (2.0 * Rc**2)
        Ipl = Lz**2 / (2.0 * _STAECKEL_INVERSE_DELTA**2) - Ec
        assert aAI.JR(Ec, Lz, Ipl) == 0.0 and aAI.Jz(Ec, Lz, Ipl) == 0.0, (
            "The circular orbit's actions are not zero"
        )
        Om = [float(as_numpy(o)) for o in aAI.Freqs(0.0, Lz, 0.0)]
        assert (
            numpy.fabs(
                Om[0] / float(as_numpy(epifreq(pot, Rc, use_physical=False))) - 1.0
            )
            < 1e-10
        ), "The circular orbit's radial frequency is not the epicycle frequency"
        assert (
            numpy.fabs(
                Om[1]
                / (
                    numpy.sign(Lz)
                    * float(as_numpy(omegac(pot, Rc, use_physical=False)))
                )
                - 1.0
            )
            < 1e-10
        ), "The circular orbit's azimuthal frequency is not the circular frequency"
        assert (
            numpy.fabs(
                Om[2] / float(as_numpy(verticalfreq(pot, Rc, use_physical=False))) - 1.0
            )
            < 1e-10
        ), "The circular orbit's vertical frequency is not the vertical frequency"
        assert invariants(aAI, Lz) < 1e-11, (
            "The circular orbit is not at the circular radius in the plane with no radial or vertical motion"
        )
    # continuity: the map at tiny actions approaches the circular orbit as
    # the librations' amplitudes
    Lz = 0.93
    Rc = float(as_numpy(rl(pot, Lz, use_physical=False)))
    circ = _staeckel_inverse_xv(aAI, 0.0, Lz, 0.0, angler, anglephi, anglez)
    for jr, jz in ((1e-6, 1e-6), (1e-8, 1e-8)):
        near = _staeckel_inverse_xv(aAI, jr, Lz, jz, angler, anglephi, anglez)
        amp = numpy.sqrt(
            2.0 * jr / float(as_numpy(epifreq(pot, Rc, use_physical=False)))
        ) + numpy.sqrt(
            2.0 * jz / float(as_numpy(verticalfreq(pot, Rc, use_physical=False)))
        )
        assert numpy.amax(numpy.fabs(near - circ)) < 5.0 * amp, (
            "The map does not approach the circular orbit as the librations' amplitudes"
        )
    # the first cell above the edge round-trips through the forward
    # transformation at the family's accuracy
    aAS = _staeckel_inverse_forward()
    for jr, jz in ((0.002, 0.003), (0.008, 0.006)):
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aAI, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS
        )
        assert dJ < 1e-4 and dth < 1e-3 and dOm < 1e-4, (
            "The family is inaccurate just above the circular edge: %g %g %g"
            % (dJ, dth, dOm)
        )
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
                float(as_numpy(aAD.Freqs(0.0, Lz, 0.0)[0]))
                / float(as_numpy(epifreq(pot, Rc, use_physical=False)))
                - 1.0
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
    jr, jz = (float(as_numpy(t)) for t in (aAK.JR(E, Lz, I3), aAK.Jz(E, Lz, I3)))
    assert (
        numpy.fabs(float(as_numpy(aAW.JR(E, Lz, I3))) - jr) < 1e-12
        and numpy.fabs(float(as_numpy(aAW.Jz(E, Lz, I3))) - jz) < 1e-12
    ), (
        "A wrapped Staeckel potential does not give the same tori as the potential itself"
    )
    xK = _staeckel_inverse_xv(aAK, jr, Lz, jz, angler, 1.0, 2.0)
    xW = _staeckel_inverse_xv(aAW, jr, Lz, jz, angler, 1.0, 2.0)
    assert numpy.amax(numpy.fabs(xK - xW)) < 1e-10, (
        "A wrapped Staeckel potential does not evaluate to the same points as the potential itself"
    )
    # retrograde: the mirror image, with the sign of the azimuthal frequency
    aAR = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[-Lz], I3s=[I3])
    assert (
        numpy.fabs(float(as_numpy(aAR.JR(E, -Lz, I3))) - jr) < 1e-12
        and numpy.fabs(float(as_numpy(aAR.Jz(E, -Lz, I3))) - jz) < 1e-12
    )
    OmP, OmR = (
        numpy.array([as_numpy(o) for o in a.Freqs(jr, l, jz)])
        for a, l in ((aAK, Lz), (aAR, -Lz))
    )
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
    jr, jz = (float(as_numpy(a(E, Lz, I3))) for a in (aAG.JR, aAG.Jz))
    assert (
        numpy.fabs(float(as_numpy(aAWG.JR(E, Lz, I3))) - jr) < 1e-12
        and numpy.fabs(float(as_numpy(aAWG.Jz(E, Lz, I3))) - jz) < 1e-12
    ), "delta= does not give the same tori as the explicitly wrapped potential"
    # the torus lives in the wrapped model: the forward transformation of
    # that model recovers it
    R, vR, vT, z, vz, phi = (as_numpy(t) for t in aAG(jr, Lz, jz, angler, 1.0, 2.0))
    f = [
        as_numpy(t)
        for t in actionAngleStaeckel(
            pot=swp, delta=0.45, c=False, order=200
        ).actionsFreqsAngles(R, vR, vT, z, vz, phi)
    ]
    assert (
        numpy.amax(numpy.fabs(f[0] - jr)) < 1e-8
        and numpy.amax(numpy.fabs(f[2] - jz)) < 1e-8
    ), (
        "The torus in the Staeckel approximation of a general potential is not that model's"
    )
    return None


def test_actionAngleStaeckelInverse_maxiter():
    # maxiter governs the angle solve: with none, the solve falls back on
    # safeguarded root-finding and still returns the torus, for explicit
    # tori and for an interpolated family alike
    from galpy.actionAngle import actionAngleStaeckelInverse

    pot = _staeckel_inverse_potential()
    angler = numpy.linspace(0.05, 6.2, 11)
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    aAN = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3])
    aAF = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], maxiter=0)
    jr, jz = (float(as_numpy(t)) for t in (aAN.JR(E, Lz, I3), aAN.Jz(E, Lz, I3)))
    nt = _staeckel_inverse_xv(aAN, jr, Lz, jz, angler, 1.0, 2.0)
    fb = _staeckel_inverse_xv(aAF, jr, Lz, jz, angler, 1.0, 2.0)
    assert numpy.amax(numpy.fabs(fb - nt)) < 1e-9, (
        "The safeguarded fallback of the angle solve does not agree with Newton for an explicit torus: %g"
        % numpy.amax(numpy.fabs(fb - nt))
    )
    kw = dict(
        pot=pot,
        setup_interp=True,
        Rmin=0.8,
        Rmax=1.4,
        Rinf=3.0,
        nE=5,
        nLz=5,
        nI3=5,
        mm_npt=32,
    )
    aAN = actionAngleStaeckelInverse(**kw)
    aAF = actionAngleStaeckelInverse(maxiter=0, **kw)
    nt = _staeckel_inverse_xv(aAN, 0.05, 1.0, 0.04, angler, 1.0, 2.0)
    fb = _staeckel_inverse_xv(aAF, 0.05, 1.0, 0.04, angler, 1.0, 2.0)
    assert numpy.amax(numpy.fabs(fb - nt)) < 1e-9, (
        "The safeguarded fallback of the angle solve does not agree with Newton: %g"
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
        jr, jz = (float(as_numpy(t)) for t in (aA.JR(E, Lz, I3), aA.Jz(E, Lz, I3)))
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aA, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS, pot=pot
        )
        assert dJ < 1e-10 and dth < 1e-9 and dOm < 1e-9 and dH < 1e-9, (
            "A torus in the perfect ellipsoid does not round-trip through the "
            f"forward transformation: {dJ}, {dth}, {dOm}, {dH}"
        )
    return None


def test_actionAngleStaeckelInverse_perfect_ellipsoid_family():
    # A small family in the oblate perfect ellipsoid, whose escape energy
    # compresses the grid's energies toward the top: tori between its nodes
    # round-trip through the forward transformation
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import PerfectEllipsoidPotential

    pot = PerfectEllipsoidPotential(amp=1.0, a=1.0, b=1.0, c=0.6, normalize=1.0)
    delta = 0.8
    aAS = actionAngleStaeckel(pot=pot, delta=delta, c=False, order=200)
    aAF = actionAngleStaeckelInverse(
        pot=pot,
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=4.0,
        nE=5,
        nLz=5,
        nI3=5,
    )
    angler = numpy.linspace(0.05, 6.2, 41)
    anglephi, anglez = 0.3 + 1.7 * angler, 0.7 + 2.3 * angler
    for ic in (
        [1.0, 0.3, 1.1, 0.2, 0.25, 0.0],
        [1.2, 0.5, 0.8, 0.3, 0.4, 0.0],
        [0.9, 0.2, 1.0, 0.4, 0.1, 0.0],
    ):
        E, Lz, I3 = _staeckel_inverse_labels(ic, pot=pot, delta=delta)
        jr, jz = (float(as_numpy(t)) for t in (aAF.JR(E, Lz, I3), aAF.Jz(E, Lz, I3)))
        dJ, dth, dOm, dH = _staeckel_inverse_roundtrip(
            aAF, jr, Lz, jz, angler, anglephi, anglez, aAS=aAS, pot=pot
        )
        assert dJ < 5e-5 and dth < 2e-3 and dOm < 2e-3 and dH < 5e-5, (
            "A torus between the nodes of a family in the perfect ellipsoid does "
            f"not round-trip through the forward transformation: {dJ}, {dth}, {dOm}, {dH}"
        )
    return None


def test_actionAngleStaeckelInverse_u0_adaptive():
    # An adaptive reference curve (a callable u0(E, L_z) or u0='planar') gives
    # every (E, L_z) of a family its own Staeckel model. The curve is
    # construction-only gauge: on an exact Staeckel potential, whose
    # approximation is itself whatever the curve, the family reproduces the
    # one with the fixed curve, and on a general potential, where the charts
    # genuinely differ across the family, the map stays exactly canonical
    from galpy.actionAngle import (
        actionAngleIsochroneInverse,
        actionAngleStaeckelInverse,
    )
    from galpy.potential import (
        IsochronePotential,
        LogarithmicHaloPotential,
        OblateStaeckelWrapperPotential,
    )

    pot = _staeckel_inverse_potential()
    grid = dict(
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=3.5,
        nE=5,
        nLz=5,
        nI3=5,
        mm_npt=48,
        progressbar=False,
    )
    base = actionAngleStaeckelInverse(pot=pot, **grid)
    angler = numpy.linspace(0.05, 6.2, 11)
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    for u0 in (lambda E, Lz: 1.0 + 0.2 * numpy.tanh(Lz) + 0.1 * E, "planar"):
        aAI = actionAngleStaeckelInverse(pot=pot, u0=u0, **grid)
        for jr, Lzq, jz in ((0.06, 0.93, 0.10), (0.15, 1.21, 0.03), (0.02, 0.8, 0.2)):
            x = _staeckel_inverse_xv(aAI, jr, Lzq, jz, angler, 1.0, 2.0)
            xb = _staeckel_inverse_xv(base, jr, Lzq, jz, angler, 1.0, 2.0)
            assert numpy.amax(numpy.fabs(x - xb)) < 1e-9, (
                "An adaptive reference curve changes the family of an exact "
                "Staeckel potential"
            )
            Om = numpy.array([as_numpy(o) for o in aAI.Freqs(jr, Lzq, jz)])
            Omb = numpy.array([as_numpy(o) for o in base.Freqs(jr, Lzq, jz)])
            assert numpy.amax(numpy.fabs(Om / Omb - 1.0)) < 1e-10, (
                "An adaptive reference curve changes the frequencies of an exact "
                "Staeckel potential"
            )
        assert (
            numpy.fabs(
                float(as_numpy(aAI.JR(E, Lz, I3))) - float(as_numpy(base.JR(E, Lz, I3)))
            )
            < 1e-10
            and numpy.fabs(
                float(as_numpy(aAI.Jz(E, Lz, I3))) - float(as_numpy(base.Jz(E, Lz, I3)))
            )
            < 1e-10
        ), "The integrals entry point does not run in the local chart"
    # the charts of a wrapped potential wrap the raw one: the same family
    aAW = actionAngleStaeckelInverse(
        pot=OblateStaeckelWrapperPotential(pot=pot, delta=1.3), u0="planar", **grid
    )
    assert (
        numpy.amax(
            numpy.fabs(
                _staeckel_inverse_xv(aAW, 0.06, 0.93, 0.10, angler, 1.0, 2.0)
                - _staeckel_inverse_xv(aAI, 0.06, 0.93, 0.10, angler, 1.0, 2.0)
            )
        )
        < 1e-12
    ), "The adaptive charts of a wrapped potential differ from the raw potential's"
    # a general potential on a wide grid: the reference curve moves the
    # family (the charts differ), and the map is exactly canonical all the
    # same, the curve being construction-only gauge
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    wide = dict(
        setup_interp=True,
        Rmin=0.5,
        Rmax=2.0,
        Rinf=6.0,
        nE=5,
        nLz=5,
        nI3=5,
        mm_npt=48,
        progressbar=False,
    )
    aAF = actionAngleStaeckelInverse(pot=lp, delta=0.8, u0="planar", **wide)
    aAD = actionAngleStaeckelInverse(pot=lp, delta=0.8, **wide)
    xF = _staeckel_inverse_xv(aAF, 0.3, 1.9, 0.1, angler, 1.0, 2.0)
    xD = _staeckel_inverse_xv(aAD, 0.3, 1.9, 0.1, angler, 1.0, 2.0)
    assert numpy.amax(numpy.fabs(xF - xD)) > 1e-4, (
        "The adaptive reference curve did not move the family of a general potential"
    )
    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=4.0, b=0.5)),
        0.06,
        0.9,
        0.1,
        0.7,
        1.0,
        2.0,
    )
    defect = _spherical_inverse_symplectic_defect(aAF, 0.3, 1.9, 0.1, 0.7, 1.0, 2.0)
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect of a family with an adaptive reference curve is "
        "above the floor: %g vs %g" % (defect, floor)
    )
    # guard: the strings are the rules
    with pytest.raises(ValueError, match="string values u0= accepts"):
        actionAngleStaeckelInverse(pot=pot, u0="midplane", **grid)
    # and a number is one reference curve, shared by all tori
    with pytest.raises(TypeError, match="one reference curve shared by all tori"):
        actionAngleStaeckelInverse(
            pot=lp,
            delta=0.8,
            u0=[1.0, 1.2],
            Es=[-0.5, -0.4],
            Lzs=[0.5, 0.6],
            I3s=[0.1, 0.1],
        )
    return None


def _staeckel_inverse_general_labels(pot, delta, Rinf, Lz, fE, y):
    # (E, L_z, I_3) of a torus of a general potential's Staeckel models with
    # focal length delta, at a fraction fE of the way from the circular
    # orbit's energy to that of the planar orbit with apocentre Rinf and y of
    # the way from the planar to the shell orbit's third integral, neither of
    # which depends on the reference curve; from the potential alone
    from scipy import optimize

    from galpy.potential import evaluatePotentials, rl

    Rc = rl(pot, Lz)
    Ec = evaluatePotentials(pot, Rc, 0.0) + Lz**2 / 2.0 / Rc**2
    E = Ec + fE * (evaluatePotentials(pot, Rinf, 0.0) + Lz**2 / 2.0 / Rinf**2 - Ec)
    Ipl = Lz**2 / 2.0 / delta**2 - E
    Ish = -optimize.minimize_scalar(
        lambda u: (
            -(
                E * numpy.sinh(u) ** 2
                - numpy.cosh(u) ** 2
                * evaluatePotentials(pot, delta * numpy.sinh(u), 0.0)
                - Lz**2 / 2.0 / delta**2 / numpy.sinh(u) ** 2
            )
        ),
        bounds=(0.1, 4.0),
        method="bounded",
        options=dict(xatol=1e-12),
    ).fun
    return E, Lz, Ipl + y * (Ish - Ipl)


def _staeckel_inverse_rule_frequencies(actions, E, Lz, I3):
    # dE/dJ along a family of tori that each live in their own Staeckel model:
    # central differences of actions(E, Lz, I3) -> (J_R, J_z) over neighbouring
    # tori, inverted; returns (Omega_R, Omega_phi, Omega_z)
    M = numpy.zeros((3, 3))
    M[2, 2] = 1.0
    x0 = numpy.array([E, I3, Lz])
    for k in range(3):
        xm, xp = x0.copy(), x0.copy()
        xm[k] -= 1e-4 * numpy.fabs(x0[k])
        xp[k] += 1e-4 * numpy.fabs(x0[k])
        M[:2, k] = (
            numpy.array(actions(xp[0], xp[2], xp[1]))
            - numpy.array(actions(xm[0], xm[2], xm[1]))
        ) / (xp[k] - xm[k])
    N = numpy.linalg.inv(M)
    return numpy.array([N[0, 0], N[0, 2], N[0, 1]])


def test_actionAngleStaeckelInverse_u0_adaptive_tables():
    # With an adaptive reference curve every (E, L_z) of a family has its own
    # Staeckel model, and the variation of the model from node to node is
    # part of the family's derivative tables: the family then converges
    # between its nodes to the tori of the models that the curve assigns
    # (without it the actions converge slowly and the frequencies not at
    # all), and its frequencies are the derivatives of the energy with
    # respect to the actions along that family of models, which are not those
    # of each torus's own model
    import pickle

    from scipy import optimize

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import LogarithmicHaloPotential, evaluatePotentials, rl

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    delta, Rinf = 0.8, 4.0

    def u0fit(E, Lz):
        # the curve of u0='planar', independently: the harmonic mean of the
        # pericentre and apocentre of the planar orbit of (E, L_z)
        phieff = lambda R: evaluatePotentials(lp, R, 0.0) + Lz**2 / 2.0 / R**2
        Rc = rl(lp, Lz)
        if E <= phieff(Rc) + 1e-12:
            return numpy.arcsinh(Rc / delta)
        Rp = optimize.brentq(lambda R: phieff(R) - E, 1e-3 * Rc, Rc, xtol=1e-14)
        Ra = optimize.brentq(lambda R: phieff(R) - E, Rc, 100.0 * Rc, xtol=1e-14)
        return numpy.arcsinh(2.0 / (1.0 / Rp + 1.0 / Ra) / delta)

    def explicit(E, Lz, I3):
        # the torus of (E, L_z, I_3) in the model that the curve assigns to it
        aAI = actionAngleStaeckelInverse(
            pot=lp,
            delta=delta,
            u0=u0fit(E, Lz),
            Es=[E],
            Lzs=[Lz],
            I3s=[I3],
            mm_npt=32,
        )
        return aAI, aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)

    tests = [
        _staeckel_inverse_general_labels(lp, delta, Rinf, *q)
        for q in ((1.05, 0.45, 0.55), (0.9, 0.3, 0.8), (1.2, 0.6, 0.3))
    ]
    refs = []
    for E, Lz, I3 in tests:
        aAI, jr, jz = explicit(E, Lz, I3)
        Om = _staeckel_inverse_rule_frequencies(
            lambda *args: explicit(*args)[1:], E, Lz, I3
        )
        assert (
            numpy.amax(numpy.fabs(Om / numpy.array(aAI.Freqs(jr, Lz, jz)) - 1.0)) > 5e-3
        ), (
            "The frequencies along the family of models do not differ from "
            "those of a torus's own model in a general potential"
        )
        refs.append((jr, jz, Om))
    # (the grid's third integrals spaced in the third integral alone, in
    # which the three test tori sit between the nodes of both grids)
    wide = dict(
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.5,
        Rinf=Rinf,
        mm_npt=32,
        axis_weight=0.0,
        progressbar=False,
    )
    fams, errs = [], []
    for n in (5, 9):
        fams.append(
            actionAngleStaeckelInverse(
                pot=lp, delta=delta, u0="planar", nE=n, nLz=n, nI3=n, **wide
            )
        )
        errs.append(
            [
                (
                    numpy.fabs(fams[-1].Jz(E, Lz, I3) - jz) / (jr + jz + Lz),
                    numpy.amax(
                        numpy.fabs(numpy.array(fams[-1].Freqs(jr, Lz, jz)) / Om - 1.0)
                    ),
                )
                for (E, Lz, I3), (jr, jz, Om) in zip(tests, refs)
            ]
        )
    errs = numpy.array(errs)
    assert numpy.all(errs[1, :, 0] < errs[0, :, 0] / 5.0) and numpy.all(
        errs[1, :, 0] < 2e-6
    ), "The actions of a family with u0='planar' do not converge between nodes"
    assert numpy.all(errs[1, :, 1] < errs[0, :, 1] / 5.0) and numpy.all(
        errs[1, :, 1] < 1e-4
    ), (
        "The frequencies of a family with u0='planar' do not converge between "
        "nodes to dE/dJ along the family of models"
    )
    # a callable that is the same curve gives the same family, its own
    # derivatives taken by differences
    famc = actionAngleStaeckelInverse(
        pot=lp, delta=delta, u0=u0fit, nE=5, nLz=5, nI3=5, **wide
    )
    angler = numpy.linspace(0.05, 6.2, 11)
    for (E, Lz, I3), (jr, jz, Om) in zip(tests, refs):
        assert (
            numpy.amax(
                numpy.fabs(
                    numpy.array(famc.Freqs(jr, Lz, jz))
                    / numpy.array(fams[0].Freqs(jr, Lz, jz))
                    - 1.0
                )
            )
            < 1e-9
        ), "A callable u0 that is the curve of u0='planar' changes the frequencies"
        assert (
            numpy.amax(
                numpy.fabs(
                    numpy.array(famc(jr, Lz, jz, angler, 1.0, angler[::-1]))
                    - numpy.array(fams[0](jr, Lz, jz, angler, 1.0, angler[::-1]))
                )
            )
            < 1e-9
        ), "A callable u0 that is the curve of u0='planar' changes the family"
    # the family pickles and evaluates identically afterwards
    pts = [
        numpy.array([refs[0][0], refs[2][0]]),
        numpy.array([tests[0][1], tests[2][1]]),
        numpy.array([refs[0][1], refs[2][1]]),
        numpy.array([0.3, 2.0]),
        numpy.array([1.0, 4.0]),
        numpy.array([2.0, 5.5]),
    ]
    again = pickle.loads(pickle.dumps(fams[0]))
    assert numpy.all(
        numpy.array(fams[0].xvFreqs(*pts)) == numpy.array(again.xvFreqs(*pts))
    ), "A pickled family with u0='planar' does not evaluate identically"
    return None


def test_actionAngleStaeckelInverse_u0_rules():
    # A rule for the reference curve that follows each torus along the third
    # integral: u0='torus' places it at the harmonic mean of the radii at
    # which the torus's own inner and outer turning spheroids cross the
    # midplane ('planar' and 'shell' are that rule on the planar and on the
    # shell orbit of the torus's (E, L_z)), and a callable may take I_3 as a
    # third argument. The u libration does not depend on the reference curve,
    # so the rule is explicit in a torus's labels
    import pickle

    from scipy import optimize

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, evaluatePotentials

    angler = numpy.linspace(0.05, 6.2, 11)
    # an exact Staeckel potential is its own approximation whatever the
    # curve: no rule changes its family. Under a forced backend, what checks
    # the rules' numerics (this, the references and the callables below) runs
    # on numpy, which test_backend_actionAngleStaeckelInverse compares the
    # backends' rules with: eager jax compiles each family's construction
    # anew, and the rules' families on the backend are those of u0='torus'
    # and the target's
    with use("numpy", force=True):
        pot = _staeckel_inverse_potential()
        grid = dict(
            setup_interp=True,
            Rmin=0.7,
            Rmax=1.6,
            Rinf=3.5,
            nE=5,
            nLz=5,
            nI3=5,
            mm_npt=48,
            progressbar=False,
        )
        base = actionAngleStaeckelInverse(pot=pot, **grid)
        xb = numpy.array(base(0.06, 0.93, 0.10, angler, 1.0, 2.0))
        Omb = numpy.array(base.Freqs(0.06, 0.93, 0.10))
        for u0 in (
            "torus",
            "shell",
            lambda E, Lz, I3: 1.0 + 0.1 * numpy.tanh(E) + 0.2 * numpy.tanh(Lz + I3),
            lambda *args: 1.0 + 0.1 * numpy.tanh(args[0]) + 0.2 * numpy.tanh(args[2]),
        ):
            aAI = actionAngleStaeckelInverse(pot=pot, u0=u0, **grid)
            assert (
                numpy.amax(
                    numpy.fabs(
                        numpy.array(aAI(0.06, 0.93, 0.10, angler, 1.0, 2.0)) - xb
                    )
                )
                < 1e-9
            ), (
                "A rule for the reference curve changes the family of a Staeckel potential"
            )
            assert (
                numpy.amax(
                    numpy.fabs(numpy.array(aAI.Freqs(0.06, 0.93, 0.10)) / Omb - 1.0)
                )
                < 1e-9
            ), (
                "A rule for the reference curve changes a Staeckel potential's frequencies"
            )
        # a general potential, in which the models genuinely differ (lpn for the
        # references, built on numpy; lp for the families, on a forced backend)
        lpn = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    delta, Rinf = 0.8, 4.0

    def turning(E, Lz, I3):
        # the u libration of (E, L_z, I_3), from the potential's midplane: the
        # shell u (where p_u^2 is largest) and the two turning points
        W = lambda u: (
            2.0
            * delta**2
            * (
                E * numpy.sinh(u) ** 2
                - numpy.cosh(u) ** 2
                * evaluatePotentials(lpn, delta * numpy.sinh(u), 0.0)
                - I3
            )
            - Lz**2 / numpy.sinh(u) ** 2
        )
        ush = optimize.minimize_scalar(
            lambda u: -W(u),
            bounds=(0.1, 4.0),
            method="bounded",
            options=dict(xatol=1e-12),
        ).x
        if W(ush) <= 1e-9:
            return ush, ush, ush
        return (
            ush,
            optimize.brentq(W, 0.05, ush, xtol=1e-14),
            optimize.brentq(W, ush, 6.0, xtol=1e-14),
        )

    def u0torus(E, Lz, I3):
        _, um, up = turning(E, Lz, I3)
        Rm, Rp = delta * numpy.sinh(um), delta * numpy.sinh(up)
        return numpy.arcsinh(2.0 / (1.0 / Rm + 1.0 / Rp) / delta)

    def actions(E, Lz, I3):
        # (J_R, J_z) of (E, L_z, I_3) in the model that the rule assigns to it
        aAI = actionAngleStaeckelInverse(
            pot=lpn,
            delta=delta,
            u0=u0torus(E, Lz, I3),
            Es=[E],
            Lzs=[Lz],
            I3s=[I3],
            mm_npt=32,
        )
        return aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)

    # the reference: 21 explicit tori
    with use("numpy", force=True):
        tests = [
            _staeckel_inverse_general_labels(lpn, delta, Rinf, *q)
            for q in ((1.05, 0.45, 0.55), (0.9, 0.3, 0.97), (1.2, 0.6, 0.3))
        ]
        refs = [
            (*actions(*t), _staeckel_inverse_rule_frequencies(actions, *t))
            for t in tests
        ]
    wide = dict(
        setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=Rinf, mm_npt=32, progressbar=False
    )
    fams, errs = [], []
    for n in (5, 9):
        fams.append(
            actionAngleStaeckelInverse(
                pot=lp, delta=delta, u0="torus", nE=n, nLz=n, nI3=n, **wide
            )
        )
        errs.append(
            [
                (
                    numpy.fabs(fams[-1].Jz(E, Lz, I3) - jz) / (jr + jz + Lz),
                    numpy.amax(
                        numpy.fabs(numpy.array(fams[-1].Freqs(jr, Lz, jz)) / Om - 1.0)
                    ),
                )
                for (E, Lz, I3), (jr, jz, Om) in zip(tests, refs)
            ]
        )
    errs = numpy.array(errs)
    assert numpy.all(errs[1, :, 0] < errs[0, :, 0] / 5.0) and numpy.all(
        errs[1, :, 0] < 2e-6
    ), "The actions of a family with u0='torus' do not converge between nodes"
    assert numpy.all(errs[1, :, 1] < errs[0, :, 1] / 3.0) and numpy.all(
        errs[1, :, 1] < 1e-4
    ), (
        "The frequencies of a family with u0='torus' do not converge between "
        "nodes to dE/dJ along the family of models"
    )
    # callables that are the built-in rules give the same families, with
    # their derivatives taken by differences, up to the shell face (a third
    # parameter with a default does not make a callable one of I_3)
    for rule, same in (
        ("torus", u0torus),
        ("shell", lambda E, Lz, d=delta: turning(E, Lz, Lz**2 / 2.0 / d**2 - E)[0]),
    ):
        with use("numpy", force=True):
            fam = (
                fams[0]
                if rule == "torus"
                else actionAngleStaeckelInverse(
                    pot=lpn, delta=delta, u0=rule, nE=5, nLz=5, nI3=5, **wide
                )
            )
            famc = actionAngleStaeckelInverse(
                pot=lpn, delta=delta, u0=same, nE=5, nLz=5, nI3=5, **wide
            )
        for (E, Lz, I3), (jr, jz, Om) in zip(tests, refs):
            x = numpy.array(fam(jr, Lz, jz, angler, 1.0, angler[::-1]))
            xc = numpy.array(famc(jr, Lz, jz, angler, 1.0, angler[::-1]))
            dOm = numpy.array(famc.Freqs(jr, Lz, jz)) / fam.Freqs(jr, Lz, jz) - 1.0
            assert (
                numpy.amax(numpy.fabs(x[:5] - xc[:5])) < 3e-6
                and numpy.amax(numpy.fabs(dOm)) < 2e-5
            ), f"A callable u0 that is the rule u0='{rule}' changes the family"
    # a family localized on a target places each target point's chart with
    # the point's own third integral, and holds the target's torus
    (E, Lz, I3), (jr, jz, Om) = tests[0], refs[0]
    box = actionAngleStaeckelInverse(
        pot=lp,
        delta=delta,
        u0="torus",
        target=Orbit(numpy.array(fams[1](jr, Lz, jz, angler, 1.0, angler[::-1])).T),
        nE=4,
        nLz=4,
        nI3=4,
        **wide,
    )
    assert (
        numpy.amax(numpy.fabs(numpy.array(box.Freqs(jr, Lz, jz)) / Om - 1.0)) < 1e-4
    ), "A target family with u0='torus' does not hold its target's torus"
    # the family pickles and evaluates identically afterwards
    pts = [numpy.array([r[k] for r in (refs[0], refs[2])]) for k in (0, 1)]
    pts = [pts[0], numpy.array([tests[0][1], tests[2][1]]), pts[1]]
    pts += [numpy.array(a) for a in ([0.3, 2.0], [1.0, 4.0], [2.0, 5.5])]
    again = pickle.loads(pickle.dumps(fams[0]))
    assert numpy.all(
        numpy.array(fams[0].xvFreqs(*pts)) == numpy.array(again.xvFreqs(*pts))
    ), "A pickled family with u0='torus' does not evaluate identically"
    return None


def test_actionAngleStaeckelInverse_u0_rules_explicit():
    # Explicit tori accept a rule for the reference curve: such a torus is
    # the torus that the rule's family has at its integrals, with that
    # family's frequencies (dE/dJ along the rule) and angles, so that a
    # family converges between its nodes to the explicit tori of its rule
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import LogarithmicHaloPotential

    angler = numpy.linspace(0.05, 6.2, 11)
    # an exact Staeckel potential: no rule changes an explicit torus
    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    one = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=48)
    jr, jz = one.JR(E, Lz, I3), one.Jz(E, Lz, I3)
    for u0 in (
        "torus",
        "planar",
        "shell",
        lambda E, Lz, I3: 1.0 + 0.1 * numpy.tanh(I3),
    ):
        aAI = actionAngleStaeckelInverse(
            pot=pot, u0=u0, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=48
        )
        assert (
            numpy.amax(
                numpy.fabs(
                    numpy.array(aAI(jr, Lz, jz, angler, 1.0, 2.0))
                    - numpy.array(one(jr, Lz, jz, angler, 1.0, 2.0))
                )
            )
            < 1e-9
            and numpy.amax(
                numpy.fabs(
                    numpy.array(aAI.Freqs(jr, Lz, jz)) / one.Freqs(jr, Lz, jz) - 1.0
                )
            )
            < 1e-9
        ), "A rule for the reference curve changes a torus of a Staeckel potential"
    # a general potential
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    delta, Rinf = 0.8, 4.0

    def explicit(E, Lz, I3, u0="torus"):
        aAI = actionAngleStaeckelInverse(
            pot=lp, delta=delta, u0=u0, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=32
        )
        return aAI, aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)

    # on the planar orbit 'torus' is 'planar'; for an inclined torus the three
    # built-in rules give three different models (the radial action is the
    # same in all: U(u) does not depend on the reference curve)
    E, Lz, I3 = _staeckel_inverse_general_labels(lp, delta, Rinf, 1.05, 0.45, 0.0)
    aT, jr, _ = explicit(E, Lz, I3)
    aF = explicit(E, Lz, I3, u0="planar")[0]
    assert (
        numpy.amax(
            numpy.fabs(
                numpy.array(aT(jr, Lz, 0.0, angler, 1.0, 2.0))
                - numpy.array(aF(jr, Lz, 0.0, angler, 1.0, 2.0))
            )
        )
        < 1e-12
    ), "u0='torus' is not u0='planar' for a planar orbit"
    tests = [
        _staeckel_inverse_general_labels(lp, delta, Rinf, *q)
        for q in ((1.05, 0.45, 0.55), (0.9, 0.3, 0.8), (1.2, 0.6, 0.3))
    ]
    (_, jrT, jzT), (_, jrF, jzF), (_, jrS, jzS) = (
        explicit(*tests[1], u0=u0) for u0 in ("torus", "planar", "shell")
    )
    assert numpy.fabs(jrT - jrF) < 1e-13 and numpy.fabs(jrT - jrS) < 1e-13, (
        "The radial action depends on the reference curve"
    )
    assert jzF < jzT - 1e-4 and jzT < jzS - 1e-4, (
        "The built-in rules do not give an inclined torus different models"
    )
    fam = actionAngleStaeckelInverse(
        pot=lp,
        delta=delta,
        u0="torus",
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.5,
        Rinf=Rinf,
        nE=9,
        nLz=9,
        nI3=9,
        mm_npt=32,
        progressbar=False,
    )
    for E, Lz, I3 in tests:
        aAI, jr, jz = explicit(E, Lz, I3)
        Om = numpy.array(aAI.Freqs(jr, Lz, jz))
        # the frequencies are dE/dJ along the rule
        Omfd = _staeckel_inverse_rule_frequencies(
            lambda *args: explicit(*args)[1:], E, Lz, I3
        )
        assert numpy.amax(numpy.fabs(Om / Omfd - 1.0)) < 1e-7, (
            "The frequencies of a torus set up with a rule are not dE/dJ along the rule"
        )
        # the mirror torus
        aAm = explicit(E, -Lz, I3)[0]
        assert (
            numpy.amax(
                numpy.fabs(numpy.array(aAm.Freqs(jr, -Lz, jz)) / Om - [1.0, -1.0, 1.0])
            )
            < 1e-12
        ), "The frequencies of a torus set up with a rule do not mirror with L_z"
        # the family of the same rule holds this torus between its nodes
        assert (
            numpy.amax(numpy.fabs(numpy.array(fam.Freqs(jr, Lz, jz)) / Om - 1.0)) < 1e-4
            and numpy.amax(
                numpy.fabs(
                    numpy.array(fam(jr, Lz, jz, angler, 1.0, angler[::-1]))
                    - numpy.array(aAI(jr, Lz, jz, angler, 1.0, angler[::-1]))
                )[:5]
            )
            < 3e-4
        ), "A family with u0='torus' does not converge to the rule's explicit tori"
    # the second-order tables of a torus set up with a rule are not available
    # (they would need the model's second derivatives along the rule)
    with pytest.raises(NotImplementedError, match="second-order tables"):
        aAI._quadratic_node(0)
    return None


def test_actionAngleStaeckelInverse_u0_rules_points():
    # With a rule for the reference curve, the map of the u libration
    # depends on the curve too, although the libration itself does not:
    # the curve changes J_z, and with it the auxiliary torus onto which the
    # u libration is lifted, whose angular momentum is J_z + |L_z|. With
    # that part of the derivative tables, the points of a family converge
    # between its nodes to those of the tori that the rule sets up
    # explicitly, at the same actions and angles (without it they stall)
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    kw = dict(pot=lp, delta=0.8, u0="torus", mm_npt=32, progressbar=False)
    angles = numpy.random.default_rng(5).uniform(0.0, 2.0 * numpy.pi, (3, 12))
    refs = []
    for Lz, fE, y in ((1.05, 0.45, 0.55), (0.95, 0.3, 0.8), (1.15, 0.6, 0.3)):
        E, Lz, I3 = _staeckel_inverse_general_labels(lp, 0.8, 2.8, Lz, fE, y)
        ex = actionAngleStaeckelInverse(Es=[E], Lzs=[Lz], I3s=[I3], **kw)
        acts = (ex.JR(E, Lz, I3), Lz, ex.Jz(E, Lz, I3))
        refs.append((acts, numpy.array(ex(*acts, *angles))[:5]))
    errs = []
    for n in (5, 9):
        fam = actionAngleStaeckelInverse(
            setup_interp=True, Rmin=0.8, Rmax=1.3, Rinf=2.8, nE=n, nLz=n, nI3=n, **kw
        )
        errs.append(
            [
                numpy.amax(numpy.fabs(numpy.array(fam(*acts, *angles))[:5] - x))
                for acts, x in refs
            ]
        )
    errs = numpy.array(errs)
    assert numpy.all(errs[1] < errs[0] / 5.0) and numpy.all(errs[1] < 2e-5), (
        "The points of a family with a rule for u0 do not converge between its "
        "nodes to those of the tori that the rule sets up explicitly"
    )
    return None


def test_actionAngleStaeckelInverse_delta_callable_tables():
    # A callable delta(E, Lz, y) lets the focal length follow the tori of a
    # family: it is a fourth table, and the derivative tables carry the
    # variation of the model along it (the frequencies are dE/dJ along it)
    import pickle

    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import LogarithmicHaloPotential, evaluatePotentials, rl

    pot = _staeckel_inverse_potential()
    grid = dict(setup_interp=True, Rmin=0.7, Rmax=1.6, Rinf=3.5, progressbar=False)
    grid.update(nE=4, nLz=4, nI3=4, mm_npt=32)
    calls = []

    def exact(E, Lz, y):
        calls.append(numpy.shape(E) + numpy.shape(Lz) + numpy.shape(y))
        return _STAECKEL_INVERSE_DELTA

    base = actionAngleStaeckelInverse(pot=pot, **grid)
    const = actionAngleStaeckelInverse(pot=pot, delta=exact, **grid)
    assert all(c == (4, 4, 4) * 3 for c in calls), (
        "A callable delta= is not called with the arrays of all the grid's nodes"
    )
    tori = [(0.06, 0.93, 0.10), (0.15, 1.21, 0.03), (0.05, 1.0, 0.0), (0.0, 1.0, 0.05)]
    for jr, Lz, jz in tori:
        assert (
            numpy.amax(
                numpy.fabs(
                    numpy.array(const.Freqs(jr, Lz, jz)) / base.Freqs(jr, Lz, jz) - 1.0
                )
            )
            < 1e-9
        ), "A rule that returns a Staeckel potential's focal length changes its family"
    acts = numpy.array(tori[:2]).T
    assert numpy.all(const.in_domain(*acts)) and not numpy.any(
        const.in_domain(5.0 + acts[0], acts[1], acts[2])
    ), "The domain of a family with a callable delta= is not found"
    assert numpy.all(
        numpy.array(pickle.loads(pickle.dumps(const)).Freqs(*acts))
        == numpy.array(const.Freqs(*acts))
    ), "A pickled family with a callable delta= does not evaluate identically"
    # a general potential: with y the fraction of the way from the planar
    # to the shell orbit's I_3 (axis_weight = 0) and a rule that depends on
    # it, and with the default y and a rule that does not, so that the same
    # tori can be set up explicitly, each in its own model
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    Rinf = 4.0
    wide = dict(setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=Rinf, progressbar=False)

    def edges(Lz):
        Rc = rl(lp, Lz)
        Ec = evaluatePotentials(lp, Rc, 0.0) + Lz**2 / 2.0 / Rc**2
        return Ec, evaluatePotentials(lp, Rinf, 0.0) + Lz**2 / 2.0 / Rinf**2 - Ec

    for rule, axis_weight in (
        (lambda E, Lz, f: 0.8 * (1.0 - 0.25 * f + 0.1 * (Lz - 1.0)), 0.0),
        (lambda E, Lz, f: 0.8 * (1.0 + 0.1 * (Lz - 1.0) + 0.3 * E), 0.25),
    ):
        knobs = dict(axis_weight=axis_weight, mm_npt=32)

        def actions(E, Lz, y):
            # (J_R, J_z) of the torus (E, L_z, y), set up in its own model
            Ec, dE = edges(Lz)
            delta = rule(E, Lz, y)
            lab = _staeckel_inverse_general_labels(
                lp, delta, Rinf, Lz, (E - Ec) / dE, y
            )
            kw = dict(Es=[lab[0]], Lzs=[Lz], I3s=[lab[2]], mm_npt=32, progressbar=False)
            aAI = actionAngleStaeckelInverse(pot=lp, delta=delta, u0="torus", **kw)
            return aAI.JR(*lab), aAI.Jz(*lab)

        refs = []
        for Lz, fE, y in ((1.05, 0.45, 0.55), (0.9, 0.3, 0.8)):
            E = edges(Lz)[0] + fE * edges(Lz)[1]
            jr, jz = actions(E, Lz, y)
            refs.append(
                (jr, Lz, jz, _staeckel_inverse_rule_frequencies(actions, E, Lz, y))
            )
        errs = numpy.array(
            [
                [
                    numpy.amax(
                        numpy.fabs(numpy.array(fam.Freqs(jr, Lz, jz)) / Om - 1.0)
                    )
                    for jr, Lz, jz, Om in refs
                ]
                for fam in (
                    actionAngleStaeckelInverse(
                        pot=lp, delta=rule, nE=n, nLz=n, nI3=n, **wide, **knobs
                    )
                    for n in (4, 7)
                )
            ]
        )
        assert numpy.all(errs[1] < errs[0] / 5.0) and numpy.all(errs[1] < 3e-4), (
            "The frequencies of a family whose focal length follows its tori "
            "do not converge between nodes to dE/dJ along the family of models"
        )
    for err, match, kw in (
        (
            TypeError,
            "requires setup_interp=True; explicit tori .* delta=family.delta",
            dict(delta="fit", Es=[-1.5], Lzs=[1.0], I3s=[1.0]),
        ),
        (TypeError, "built-in rule for the reference curve", dict(u0=1.0, **wide)),
        (
            ValueError,
            "positive focal length",
            dict(delta=lambda E, L, y: 0.5 - y, **wide),
        ),
    ):
        with pytest.raises(err, match=match):
            actionAngleStaeckelInverse(**{**dict(pot=lp, delta=rule), **kw})
    with pytest.raises(NotImplementedError, match="do not label the tori"):
        const.JR(-1.5, 1.0, 1.0)
    return None


def test_actionAngleStaeckelInverse_delta_callable_points():
    # The points of a family whose focal length follows its tori: the angles
    # compensate the dilation of the prolate spheroidal coordinates from
    # torus to torus, -(R p_R + z p_z) dln(delta)/dJ, which couples the two
    # librations, so that the map stays exactly canonical
    import pickle

    from galpy.actionAngle import (
        actionAngleIsochroneInverse,
        actionAngleStaeckelInverse,
    )
    from galpy.potential import IsochronePotential, LogarithmicHaloPotential

    iso = actionAngleIsochroneInverse(ip=IsochronePotential(amp=4.0, b=0.5))
    floor = _spherical_inverse_symplectic_defect(iso, 0.06, 0.9, 0.1, 0.7, 1.0, 2.0)
    # an exact Staeckel potential: a rule that returns its focal length
    # changes nothing; one that varies (also at the circular energy and
    # along the third integral) builds the tori of Staeckel approximations
    # of it, a different family (except in the plane, where every Staeckel
    # approximation is exact), and the map is canonical all the same
    pot = _staeckel_inverse_potential()
    grid = dict(setup_interp=True, Rmin=0.7, Rmax=1.6, Rinf=3.5, progressbar=False)
    grid.update(nE=4, nLz=4, nI3=4, mm_npt=32)

    def rule(E, Lz, f):
        return _STAECKEL_INVERSE_DELTA * (
            1.0 + 0.15 * numpy.sin(2.0 * Lz) + 0.2 * (f - 0.5) + 0.05 * E * f
        )

    base = actionAngleStaeckelInverse(pot=pot, **grid)
    const = actionAngleStaeckelInverse(
        pot=pot, delta=lambda E, Lz, f: _STAECKEL_INVERSE_DELTA, **grid
    )
    vary = actionAngleStaeckelInverse(pot=pot, delta=rule, **grid)
    angler = numpy.linspace(0.05, 6.2, 11)
    tori = [(0.06, 0.93, 0.10), (0.15, 1.21, 0.03), (0.02, 0.8, 0.2)]
    # (the last two on the faces on which one libration is degenerate)
    for jr, Lz, jz in tori + [(0.05, 1.0, 0.0), (0.0, 1.0, 0.05)]:
        xb = numpy.array(base(jr, Lz, jz, angler, 1.0, 2.0))
        assert (
            numpy.amax(
                numpy.fabs(numpy.array(const(jr, Lz, jz, angler, 1.0, 2.0)) - xb)
            )
            < 1e-9
        ), "A rule that returns a Staeckel potential's focal length changes its family"
        assert (
            numpy.amax(numpy.fabs(numpy.array(vary(jr, Lz, jz, angler, 1.0, 2.0)) - xb))
            > 1e-3
        ) == (jz > 0.0), (
            "A focal length that follows the tori does not change the tori out "
            "of the plane, and only those"
        )
    for jr, Lz, jz in tori:
        defect = _spherical_inverse_symplectic_defect(vary, jr, Lz, jz, 0.7, 1.0, 2.0)
        assert defect < 20.0 * floor + 1e-9, (
            "The symplectic defect of a family whose focal length follows its "
            "tori is above the floor: %g vs %g" % (defect, floor)
        )
    # many tori at once, a pickled family, and the angle solve's fallback
    # (which a Newton iteration that is cut short leaves to it)
    acts = numpy.array(tori).T
    angs = numpy.array([[0.3, 2.0, 4.1], [1.0, 4.0, 0.2], [2.0, 5.5, 3.3]])
    many = numpy.array(vary.xvFreqs(*acts, *angs))
    one = numpy.array(
        [
            [numpy.ravel(q)[0] for q in vary.xvFreqs(*acts[:, k], *angs[:, k : k + 1])]
            for k in range(3)
        ]
    ).T
    assert numpy.amax(numpy.fabs(many - one)) < 1e-10, (
        "Tori of a family whose focal length follows them do not evaluate "
        "together as they do one by one"
    )
    assert numpy.all(
        numpy.array(pickle.loads(pickle.dumps(vary)).xvFreqs(*acts, *angs)) == many
    ), "A pickled family with a callable delta= does not evaluate identically"
    short = actionAngleStaeckelInverse(pot=pot, delta=rule, maxiter=2, **grid)
    assert (
        numpy.amax(
            numpy.fabs(
                numpy.array(short(*tori[0], angler[:3], 1.0, 2.0))
                - numpy.array(vary(*tori[0], angler[:3], 1.0, 2.0))
            )
        )
        < 1e-8
    ), "The angle solve's fallback does not agree with its Newton iteration"
    # a general potential: the points between the nodes of a coarse family
    # are those of a finer one (without the variation of the model along the
    # focal length in the maps' derivative tables they differ by 0.08)
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    wide = dict(setup_interp=True, Rmin=0.8, Rmax=1.3, Rinf=2.8, progressbar=False)
    acts = numpy.array([[0.05, 1.0, 0.04], [0.12, 0.95, 0.08], [0.03, 1.15, 0.15]]).T
    coarse, fine = (
        numpy.array(
            actionAngleStaeckelInverse(
                pot=lp,
                delta=lambda E, Lz, f: 0.8 * (1.0 - 0.25 * f + 0.1 * Lz + 0.3 * E),
                nE=n,
                nLz=n,
                nI3=n,
                mm_npt=32,
                **wide,
            )(*acts, *angs)
        )
        for n in (4, 7)
    )
    assert numpy.amax(numpy.fabs(coarse - fine)[:5]) < 5e-3, (
        "The points of a family whose focal length follows its tori do not "
        "converge between its nodes"
    )
    return None


def test_actionAngleStaeckelInverse_delta_fit():
    # delta='fit' gives every torus of a family the focal length that
    # minimizes the spread of the true Hamiltonian over the torus of its
    # Staeckel approximation, fitted smoothly over the family; the delta
    # method returns a family's focal lengths and is itself a rule
    import pickle

    from galpy.actionAngle import (
        actionAngleIsochroneInverse,
        actionAngleStaeckelInverse,
    )
    from galpy.potential import (
        IsochronePotential,
        LogarithmicHaloPotential,
        evaluatePotentials,
    )

    # an exact Staeckel potential: the fit finds its focal length, at which
    # the spread vanishes, and with it the potential's own family
    pot = _staeckel_inverse_potential()
    grid = dict(setup_interp=True, Rmin=0.7, Rmax=1.6, Rinf=3.5, progressbar=False)
    grid.update(nE=4, nLz=4, nI3=4, mm_npt=32)
    base = actionAngleStaeckelInverse(pot=pot, **grid)
    fit = actionAngleStaeckelInverse(pot=pot, delta="fit", delta_deg=(1, 1, 1), **grid)
    E, Lz, y = (
        numpy.array([-1.6, -1.3]),
        numpy.array([0.8, 1.2]),
        numpy.array([0.2, 0.7]),
    )
    assert (
        numpy.amax(numpy.fabs(fit.delta(E, Lz, y) / _STAECKEL_INVERSE_DELTA - 1.0))
        < 1e-4
    ), "delta='fit' does not find the focal length of a Staeckel potential"
    assert base.delta(E[0], Lz[0], y[0]) == _STAECKEL_INVERSE_DELTA, (
        "The delta method does not return the focal length of a family that has one"
    )
    angs = numpy.array([[0.3, 2.0, 4.1], [1.0, 4.0, 0.2], [2.0, 5.5, 3.3]])
    acts = numpy.array([[0.06, 0.93, 0.10], [0.15, 1.21, 0.03], [0.02, 0.8, 0.2]]).T
    assert (
        numpy.amax(
            numpy.fabs(
                numpy.array(fit.xvFreqs(*acts, *angs))
                - numpy.array(base.xvFreqs(*acts, *angs))
            )
        )
        < 1e-4
    ), "delta='fit' does not find the family of a Staeckel potential"
    # a family with a callable: the method returns the family's own table
    rule = lambda E, Lz, f: 1.3 * (1.0 + 0.2 * (f - 0.5) + 0.05 * E * f)  # noqa: E731
    vary = actionAngleStaeckelInverse(pot=pot, delta=rule, **grid)
    assert numpy.all(vary.delta(E, Lz, y) == rule(E, Lz, y)), (
        "The delta method does not return the focal lengths of a family with "
        "a callable delta="
    )
    # the callable is not pickled with the family
    with pytest.raises(RuntimeError, match="was not pickled"):
        pickle.loads(pickle.dumps(vary)).delta(E, Lz, y)
    # a general potential: the fitted family is canonical, the true energy
    # varies less over its tori than over those of single focal lengths,
    # and its delta method gives another family the same focal lengths
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    wide = dict(setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=4.0, progressbar=False)
    wide.update(nE=4, nLz=4, nI3=4, mm_npt=32)
    fit = actionAngleStaeckelInverse(pot=lp, delta="fit", delta_deg=(1, 1, 1), **wide)
    iso = actionAngleIsochroneInverse(ip=IsochronePotential(amp=4.0, b=0.5))
    floor = _spherical_inverse_symplectic_defect(iso, 0.06, 0.9, 0.1, 0.7, 1.0, 2.0)
    defect = _spherical_inverse_symplectic_defect(fit, 0.08, 1.05, 0.06, 0.7, 1.0, 2.0)
    assert defect < 20.0 * floor + 1e-9, (
        "The symplectic defect of a family with delta='fit' is above the "
        "floor: %g vs %g" % (defect, floor)
    )
    th = 2.0 * numpy.pi * (numpy.arange(12) + 0.5) / 12.0
    thR, thz = (t.ravel() for t in numpy.meshgrid(th, th))

    def spread(fam, jr, Lz, jz):
        R, vR, vT, z, vz, _ = (
            as_numpy(q) for q in fam(jr, Lz, jz, thR, 0.0 * thR, thz)
        )
        return numpy.std(
            0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(lp, R, z))
        )

    single = [
        actionAngleStaeckelInverse(pot=lp, delta=d, u0="torus", **wide)
        for d in (0.5, 0.8, 1.2)
    ]
    acts = numpy.array([[0.08, 1.05, 0.06], [0.03, 0.9, 0.12], [0.15, 1.2, 0.02]])
    for tor in acts:
        assert spread(fit, *tor) < 0.8 * min(spread(fam, *tor) for fam in single), (
            "The true energy does not vary less over a torus of a family with "
            "delta='fit' than over those of single focal lengths"
        )
    again = actionAngleStaeckelInverse(pot=lp, delta=fit.delta, **wide)
    out = numpy.array(fit.xvFreqs(*acts.T, *angs))
    assert numpy.amax(numpy.fabs(numpy.array(again.xvFreqs(*acts.T, *angs)) - out)) < (
        1e-9
    ), "A family given another's delta method does not have its focal lengths"
    # the fitted focal length is as smooth as the polynomial and gives the
    # tables nothing more to resolve: between its nodes the family's
    # points are those of a finer family with the same focal lengths to
    # 3e-4 (single focal lengths from 0.5 to 1.2 in this box: 5e-4 to 1e-3)
    finer = actionAngleStaeckelInverse(
        pot=lp, delta=fit.delta, **{**wide, "nE": 7, "nLz": 7, "nI3": 7}
    )
    err = numpy.amax(numpy.fabs(numpy.array(finer.xvFreqs(*acts.T, *angs)) - out)[:5])
    assert err < 5e-4, (
        "The points of a family with delta='fit' do not converge between its "
        "nodes: %g" % err
    )
    assert numpy.all(
        numpy.array(pickle.loads(pickle.dumps(fit)).xvFreqs(*acts.T, *angs)) == out
    ), "A pickled family with delta='fit' does not evaluate identically"
    with pytest.raises(ValueError, match="only string value delta= accepts"):
        actionAngleStaeckelInverse(pot=lp, delta="best", **wide)
    return None


def test_actionAngleStaeckelInverse_delta_explicit():
    # Explicit tori accept a rule for the focal length, a callable or the
    # delta method of a family: such a torus is the torus that the rule's
    # family has at its actions, with that family's angles (which carry the
    # variation of the focal length from torus to torus) and frequencies,
    # also where the family's grid does not reach
    import pickle

    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import name_of_namespace, resolve_namespace, use
    from galpy.potential import LogarithmicHaloPotential

    # (the families and the explicit tori set up with numpy, as their
    # construction on a backend is test_backend_actionAngleStaeckelInverse's;
    # on a backend, C and Python are the same backend code, run once)
    on_numpy = name_of_namespace(resolve_namespace()) == "numpy"
    wide = dict(setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=4.0, progressbar=False)
    wide.update(nE=4, nLz=4, nI3=4, mm_npt=32)
    rule = lambda E, Lz, f: 0.8 * (1.0 + 0.2 * (f - 0.5) + 0.05 * E * f + 0.1 * Lz)  # noqa: E731
    with use("numpy", force=True):
        lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
        fit = actionAngleStaeckelInverse(
            pot=lp, delta="fit", delta_deg=(1, 1, 1), **wide
        )
        vary = actionAngleStaeckelInverse(pot=lp, delta=rule, u0="torus", **wide)
    angs = numpy.array(
        [[0.3, 2.0, 4.1, 5.5], [1.0, 4.0, 0.2, 3.0], [2.0, 5.5, 3.3, 1.0]]
    )
    wrap = lambda d: (d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi  # noqa: E731

    def explicit(fam, delta, J, **kw):
        J = numpy.atleast_2d(J)
        if "Es" not in kw:
            kw.update(jrs=J[:, 0], jzs=J[:, 2])
        kw.update(mm_npt=32, auxiliary=fam.auxiliary, progressbar=False)
        with use("numpy", force=True):
            return actionAngleStaeckelInverse(pot=lp, delta=delta, Lzs=J[:, 1], **kw)

    def points(aA, J):
        # (the frequencies of one torus are floats)
        return numpy.array([o + 0.0 * angs[0] for o in aA.xvFreqs(*J, *angs)])

    # at the nodes of a family (one of them a shell and one a planar orbit),
    # where the family is not interpolated, the explicit tori of its rule
    # are the family's, set up from their actions or from their integrals
    for fam, delta in ((fit, fit.delta), (vary, rule), (vary, vary.delta)):
        nodes = ((1, 1, 1), (2, 2, 0), (3, 1, 2), (2, 3, 1), (2, 0, 3))
        J = numpy.array(
            [
                [fam._A_tab[n][0], fam._Lzgrid[n[2]], fam._A_tab[n][1]]
                for n in nodes[: 5 if delta is not vary.delta else 1]
            ]
        )
        aAI = explicit(fam, delta, J)
        # (from the integrals: the tori with both librations, whose third
        # integral does not have to be that of a face to round-off)
        aAE = explicit(fam, delta, J[:3], Es=aAI._Es[:3], I3s=aAI._I3s[:3])
        for k, (jr, Lz, jz, E, I3) in enumerate(zip(*J.T, aAI._Es, aAI._I3s)):
            ref = points(fam, (jr, Lz, jz))
            for aA in (aAI, aAE)[: 1 + (k < len(aAE._Es))]:
                dev = numpy.fabs(points(aA, (jr, Lz, jz)) - ref)
                dev[5] = numpy.fabs(wrap(dev[5]))
                assert numpy.amax(dev) < 1e-9, (
                    "An explicit torus with a family's rule for the focal length "
                    "is not the family's torus at a node: %g" % numpy.amax(dev)
                )
                assert (
                    numpy.fabs(aA.JR(E, Lz, I3) - jr) < 1e-10
                    and numpy.fabs(aA.Jz(E, Lz, I3) - jz) < 1e-10
                ), "A torus set up with a rule does not have the requested actions"
        assert numpy.all(
            aAI.delta(aAI._Es, J[:, 1], 0.3) == delta(aAI._Es, J[:, 1], 0.3)
        )
    # where the family's grid does not reach (small |L_z|, of either sign),
    # the forward transformation in the rule's models returns the actions,
    # the frequencies and the angles of the explicit tori (the first torus
    # lies on the grid: to 1e-9): that of the family, and that of the
    # explicit tori themselves, with the rule of a family or a callable, in
    # C and in Python (to the same accuracy: the reference curve of each
    # point's model is that of the focal length that the iteration ends
    # at); a pickled instance evaluates identically, and is not inverted
    # without its rule
    J = numpy.array([(0.08, 1.05, 0.06), (0.2, 0.3, 0.6), (0.05, -0.3, 0.8)])
    aAI, aAU = explicit(fit, fit.delta, J), explicit(vary, rule, J)
    again = pickle.loads(pickle.dumps(aAI))
    # (the points of each set's tori, all transformed at once)
    pts = {id(aA): [points(aA, j) for j in J] for aA in (aAI, aAU)}
    assert all(numpy.all(points(again, j) == p) for j, p in zip(J, pts[id(aAI)])), (
        "A pickled set of explicit tori with a rule does not evaluate identically"
    )
    for aA, inverse, tol, c in (
        (aA, inverse, tol, c)
        for aA, inverse, tol in ((aAI, fit, 1e-8), (aAI, aAI, 1e-8), (aAU, aAU, 1e-6))
        for c in ((True, False) if on_numpy else (True,))
    ):
        aAF = actionAngleStaeckel(inverse=inverse, c=c)
        x = numpy.concatenate(pts[id(aA)], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", galpyWarning)
            out = numpy.array(aAF.actionsFreqsAngles(*x[:6]))
            assert numpy.allclose(
                numpy.array(aAF.actionsFreqs(*x[:5])), out[:6], atol=1e-10
            ) and numpy.allclose(numpy.array(aAF(*x[:5])), out[:3], atol=1e-10)
        for k, ((jr, Lz, jz), p) in enumerate(zip(J, pts[id(aA)])):
            o = out[:, 4 * k : 4 * k + 4]
            dev = numpy.fabs(
                o - numpy.array([jr, Lz, jz, *p[6:, 0], 0.0, 0.0, 0.0])[:, None]
            )
            dev[6:] = numpy.fabs(wrap(o[6:] - angs))
            assert numpy.amax(dev) < (1e-9 if k == 0 else tol), (
                "The forward transformation in the rule's models does not return "
                "the actions and angles of an explicit torus of the rule (c=%s): %g"
                % (c, numpy.amax(dev))
            )
    with pytest.raises(RuntimeError, match="was not pickled"):
        again.delta(0.5, 1.0, 0.5)
    with pytest.raises(RuntimeError, match="was not pickled with them"):
        actionAngleStaeckel(inverse=again)

    # the map of the explicit tori of a rule is canonical (to the central
    # differences across neighbouring tori); it is not without the terms of
    # the rule, on the same tori with their focal lengths as numbers
    def xv(jr, Lz, jz, *angles, terms=True):
        aA = explicit(fit, fit.delta, (jr, Lz, jz))
        if not terms:
            aA = explicit(fit, float(aA._deltas[0]), (jr, Lz, jz), u0="torus")
        return aA(jr, Lz, jz, *([a] for a in angles))

    args = (0.1, 0.9, 0.3, 0.7, 1.0, 2.0)
    defect = _spherical_inverse_symplectic_defect(xv, *args, h=1e-4)
    assert defect < 1e-6, (
        "The map of explicit tori with a rule for the focal length is not "
        "canonical: %g" % defect
    )
    assert (
        _spherical_inverse_symplectic_defect(
            lambda *a: xv(*a, terms=False), *args, h=1e-4
        )
        > 1e-2
    ), "Tori of different focal lengths are canonical without the rule's terms"
    # the circular orbit; actions that the rule's model has no torus for;
    # and a rule without a focal length that is its own at the torus
    aAC = explicit(fit, fit.delta, (0.0, 0.9, 0.0))
    assert numpy.fabs(aAC(0.0, 0.9, 0.0, 1.0, 2.0, 3.0)[0] - lp.rl(0.9)) < 1e-10, (
        "The circular orbit set up with a rule is not at the circular radius"
    )
    # actions for which the model of the focal length that the iteration
    # starts from has no torus (at this small |L_z| the u libration splits
    # for delta > 0.6 and the actions jump over these), while that of the
    # rule's own focal length at the torus does: found from the rule at the
    # shell orbit, and from a focal length between the rule's extremes;
    # none when no start has a torus
    near = (0.05, 0.15, 0.8)
    for own in (
        lambda E, L, f: 0.9 - 0.42 * f + 0.0 * E,
        lambda E, L, f: 0.9 - 0.45 * numpy.exp(-(((f - 0.85) / 0.15) ** 2)) + 0.0 * E,
    ):
        aAN = explicit(fit, own, near)
        assert (
            numpy.fabs(aAN.JR(aAN._Es[0], 0.15, aAN._I3s[0]) - 0.05) < 1e-10
            and numpy.fabs(aAN.Jz(aAN._Es[0], 0.15, aAN._I3s[0]) - 0.8) < 1e-10
            and aAN.delta(0.0, 0.15, 0.5) > 0.65
            and aAN._deltas[0] < 0.6
        ), "A torus that only the rule's own model holds is not set up"
    with pytest.raises(ValueError, match="No torus with the actions .* rule delta="):
        explicit(fit, lambda E, L, f: 0.7 + 0.0 * f, near)
    with pytest.raises(ValueError, match="positive focal lengths"):
        explicit(fit, lambda E, L, f: 0.5 - f, near)
    with pytest.raises(RuntimeError, match="did not converge"):
        explicit(
            fit,
            lambda E, L, f: numpy.where(f < 0.4, 0.9, 0.5),
            (0.0, 0.9, 0.0),
            Es=[0.84],
            I3s=[0.9],
        )
    return None


def test_actionAngleStaeckelInverse_delta_explicit_many():
    # The focal lengths of explicit tori that are set up from their actions
    # with a rule are found for all of the tori at once: the tori are those
    # that are found one at a time, for every rule for the reference curve
    # (also for the torus on the shell face and the rule u0='shell', which
    # are found one by one), and a rule that is not positive is refused for
    # a torus on the shell face as it is for any other
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    rule = lambda E, Lz, f: 0.8 * (1.0 + 0.2 * (f - 0.5) + 0.05 * E * f + 0.1 * Lz)  # noqa: E731
    jr = numpy.array([0.0, 0.03, 0.25, 0.1, 0.3])
    Lz = numpy.array([0.9, 1.1, -0.6, 0.8, 1.2])
    jz = numpy.array([0.1, 0.2, 0.05, 0.3, 0.02])
    angs = numpy.array([0.3, 2.0, 4.1])

    def setup(sel, u0, delta=rule):
        return actionAngleStaeckelInverse(
            pot=lp,
            delta=delta,
            u0=u0,
            jrs=jr[sel],
            Lzs=Lz[sel],
            jzs=jz[sel],
            mm_npt=32,
            progressbar=False,
        )

    for u0 in ("torus", "planar", "shell"):
        many = setup(slice(None), u0)
        for ii in range(len(jr)):
            one = setup([ii], u0)
            assert numpy.fabs(one._deltas[0] / many._deltas[ii] - 1.0) < 1e-9, (
                f"The focal length of a torus under a rule with u0={u0} differs "
                "between many tori and one"
            )
            assert numpy.all(
                numpy.fabs(
                    numpy.array(many(jr[ii], Lz[ii], jz[ii], *angs))
                    - numpy.array(one(jr[ii], Lz[ii], jz[ii], *angs))
                )
                < 1e-8
            ), f"A torus under a rule with u0={u0} differs between many tori and one"
    with pytest.raises(ValueError, match="positive focal lengths"):
        setup([0], "torus", delta=lambda E, L, f: 0.5 - f)
    return None


def test_actionAngleStaeckelInverse_delta_target():
    # target= with a focal length that follows the tori: the box of the
    # target is found in the rule's models, and a localized family that is
    # given another family's delta method has that family's tori
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    o = Orbit([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    ts = numpy.linspace(0.0, 30.0, 61)
    o.integrate(ts, lp)
    acts = [float(numpy.ravel(q)[0]) for q in actionAngleStaeckel(pot=lp, delta=0.8)(o)]
    kw = dict(pot=lp, setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=4.0, mm_npt=32)
    kw.update(progressbar=False, delta_deg=(1, 1, 1))
    wide = actionAngleStaeckelInverse(delta="fit", nE=7, nLz=7, nI3=7, **kw)
    calls = []

    def rule(E, Lz, f):
        calls.append(numpy.shape(E))
        return wide.delta(E, Lz, f)

    small = dict(target=o(ts), nE=4, nLz=4, nI3=4, **kw)
    loc = actionAngleStaeckelInverse(delta=rule, **small)
    assert calls[0] == (61,) and all(c == (4, 4, 4) for c in calls[1:]), (
        "A callable delta= is not called with the arrays of a target's points "
        "and of the grid's nodes"
    )
    fit = actionAngleStaeckelInverse(delta="fit", **small)
    angs = numpy.array([[0.3, 2.0, 4.1], [1.0, 4.0, 0.2], [2.0, 5.5, 3.3]])
    acts = [a + 0.0 * angs[0] for a in acts]
    assert numpy.all(loc.in_domain(*acts)) and numpy.all(fit.in_domain(*acts)), (
        "A localized family whose focal length follows its tori does not "
        "hold its target"
    )
    ref = numpy.array(wide.xvFreqs(*acts, *angs))
    dloc, dfit = (
        numpy.amax(numpy.fabs(numpy.array(fam.xvFreqs(*acts, *angs)) - ref)[:5])
        for fam in (loc, fit)
    )
    assert dloc < 1e-5, (
        "A localized family given another family's delta method does not have "
        "that family's tori: %g" % dloc
    )
    assert dloc < dfit < 1e-3, (
        "A localized family with delta='fit' is not fitted on its own box: %g" % dfit
    )
    return None


def test_actionAngleStaeckelInverse_target():
    # target= localizes the family's grid on the tori of a set of phase-space
    # points, through the same label relations the node lattice uses: the
    # same number of nodes then reproduces the target far better than the
    # default box does, and tori outside the box are refused
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use
    from galpy.orbit import Orbit

    pot = _staeckel_inverse_potential()
    aAS = _staeckel_inverse_forward()
    o = Orbit([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    ts = numpy.linspace(0.0, 12.0, 8)
    o.integrate(ts, pot)
    kw = dict(
        pot=pot,
        setup_interp=True,
        Rmin=0.75,
        Rmax=1.25,
        Rinf=3.0,
        nE=5,
        nLz=5,
        nI3=5,
        mm_npt=48,
        progressbar=False,
    )
    wide = actionAngleStaeckelInverse(**kw)
    loc = actionAngleStaeckelInverse(target=o(ts), **kw)
    rows = numpy.array(
        [
            as_numpy(o(ts).R()),
            as_numpy(o(ts).vR()),
            as_numpy(o(ts).vT()),
            as_numpy(o(ts).z()),
            as_numpy(o(ts).vz()),
        ]
    ).T
    labels = [aAS.actionsFreqsAngles(*row, 0.0) for row in rows]

    def miss(aAI, row, lab):
        jr, lz, jz = (float(lab[k][0]) for k in (0, 1, 2))
        ar, ap, az = (numpy.atleast_1d(float(lab[k][0])) for k in (6, 7, 8))
        out = aAI(jr, lz, jz, ar, ap, az)
        return numpy.hypot(float(out[0][0]) - row[0], float(out[3][0]) - row[3])

    dxw = max(miss(wide, row, lab) for row, lab in zip(rows, labels))
    dxl = max(miss(loc, row, lab) for row, lab in zip(rows, labels))
    assert dxl < 1e-6, "The localized grid does not reproduce its target: %g" % dxl
    assert dxl < dxw / 100.0, (
        "The localized grid is not much better than the default box on its target: "
        "%g vs %g" % (dxl, dxw)
    )
    # the box excludes the degenerate faces and the circular orbit, which
    # the target's tori are nowhere near, and energies outside it
    jr, lz, jz = (float(labels[0][k][0]) for k in (0, 1, 2))
    with pytest.raises(ValueError, match="does not reach"):
        loc(0.0, lz, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="does not reach"):
        loc(jr, lz, 0.0, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="circular face"):
        loc(0.0, lz, 0.0, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="circular face"):
        loc.Freqs(0.0, lz, 0.0)
    # out-of-range errors point at target_pad, not at Rinf
    hint = r"\(the grid is localized on a target: increase target_pad"
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc(jr, 0.5, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match=r"above the covered [\d.]+ " + hint):
        loc(10.0 * jr, lz, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match=f"at \\|L_z\\| = {lz} " + hint):
        loc(0.3 * jr, lz, jz, 0.3, 1.0, 2.0)
    E, Lz, I3 = _staeckel_inverse_labels([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc.JR(E + 0.1, Lz, I3)
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc.JR(E, Lz, I3 + 0.5)
    with pytest.raises(ValueError, match="circular face"):
        with use("numpy", force=True):  # the numpy closed form
            Ec_wide = float(wide._circular(Lz)["Ec"])
        loc.JR(Ec_wide, Lz, I3)
    assert (
        numpy.fabs(float(as_numpy(loc.JR(E, Lz, I3))) - jr) < 1e-7
        and numpy.fabs(float(as_numpy(loc.Jz(E, Lz, I3))) - jz) < 1e-7
    ), "The localized family's actions from the integrals are off"
    # a single point: the box has the minimum width and still holds it; an
    # Orbit and its own rows are the same target, with a phi column ignored
    one = actionAngleStaeckelInverse(target=rows[0], **kw)
    assert miss(one, rows[0], labels[0]) < 1e-6, (
        "The grid localized on a single point does not reproduce it"
    )
    # a circular orbit as the target: its box reaches the circular face
    from galpy.potential import vcirc

    circ = actionAngleStaeckelInverse(
        target=[1.0, 0.0, vcirc(pot, 1.0), 0.0, 0.0], **kw
    )
    assert numpy.all(
        numpy.isfinite([as_numpy(o) for o in circ.Freqs(0.0, vcirc(pot, 1.0), 0.0)])
    ), "The grid localized on a circular orbit does not reach the circular face"
    six = actionAngleStaeckelInverse(
        target=numpy.hstack((rows, 0.3 * numpy.ones((len(rows), 1)))), **kw
    )
    # the same target and so the same box (exactly), and the same family (to
    # round-off: two builds under jax differ in the last bits, because XLA's
    # multithreaded FFT on the CPU is not reproducible from call to call)
    assert numpy.array_equal(six._target, loc._target) and six._box == loc._box, (
        "Rows with a phi column are not the same target as the Orbit"
    )
    numpy.testing.assert_allclose(
        numpy.array([as_numpy(t) for t in six(jr, lz, jz, 0.3, 1.0, 2.0)]),
        numpy.array([as_numpy(t) for t in loc(jr, lz, jz, 0.3, 1.0, 2.0)]),
        rtol=0.0,
        atol=1e-13,
        err_msg="Rows with a phi column do not give the Orbit's family",
    )
    # a padding so generous that the box reaches every face: the faces are
    # then in the family, like the default box's (a target of two distinct
    # tori, since a single torus has no spread for the padding to scale)
    o2 = Orbit([0.9, -0.1, 1.15, -0.05, 0.1, 0.0])
    o2.integrate(ts, pot)
    rows2 = numpy.array(
        [
            as_numpy(o2(ts).R()),
            as_numpy(o2(ts).vR()),
            as_numpy(o2(ts).vT()),
            as_numpy(o2(ts).z()),
            as_numpy(o2(ts).vz()),
        ]
    ).T
    big = actionAngleStaeckelInverse(
        target=numpy.vstack((rows, rows2)), target_pad=5.0, **kw
    )
    with pytest.raises(ValueError, match="beyond Rinf"):
        actionAngleStaeckelInverse(
            target=numpy.vstack((rows, rows2)), target_pad=50.0, **kw
        )
    # a padding under which the circular orbits of the box stay inside Rinf
    # but the bottom of its energies does not: the message names the padding
    with pytest.raises(ValueError, match="Rinf is too small.*or reduce target_pad"):
        actionAngleStaeckelInverse(
            target=numpy.vstack((rows, rows2)), target_pad=12.0, **kw
        )
    assert numpy.all(
        numpy.isfinite(as_numpy(big(0.0, lz, 0.0, 0.3, 1.0, 2.0)[0]))
    ) and (numpy.all(numpy.isfinite(as_numpy(big(0.0, lz, jz, 0.3, 1.0, 2.0)[0])))), (
        "A generously padded target box does not reach the faces"
    )
    with pytest.raises(
        ValueError, match=r"above the covered actions \[0, [\d.]+\] " + hint
    ):
        big(0.0, lz, 50.0, 0.3, 1.0, 2.0)
    # composes with an adaptive reference curve
    fit = actionAngleStaeckelInverse(target=o(ts), u0="planar", **kw)
    assert max(miss(fit, row, lab) for row, lab in zip(rows, labels)) < 1e-6, (
        "The localized family with an adaptive reference curve misses its target"
    )
    # guards: the target shapes the interpolation grid, must be coverable,
    # and must parse
    with pytest.raises(TypeError, match="requires setup_interp"):
        actionAngleStaeckelInverse(
            pot=pot, target=rows[0], Es=[E], Lzs=[Lz], I3s=[I3], progressbar=False
        )
    with pytest.raises(ValueError, match="increase Rinf"):
        actionAngleStaeckelInverse(
            target=[1.0, 0.9, 1.0, 0.3, 0.9], **{**kw, "Rinf": 1.3}
        )
    with pytest.raises(ValueError, match="rows must be"):
        actionAngleStaeckelInverse(target=[1.0, 0.1, 1.0, 0.1], **kw)
    with pytest.raises(ValueError, match="rows must be"):
        actionAngleStaeckelInverse(target=numpy.ones((2, 2, 5)), **kw)
    with pytest.raises(ValueError, match="R > 0"):
        actionAngleStaeckelInverse(target=[-1.0, 0.1, 1.0, 0.1, 0.05], **kw)
    with pytest.raises(ValueError, match="R > 0"):
        actionAngleStaeckelInverse(target=[1.0, 0.1, 0.0, 0.1, 0.05], **kw)
    return None


def test_actionAngleStaeckelInverse_array_actions(staeckel_inverse_interp):
    # Arrays of actions broadcast against the angles: many distinct tori are
    # evaluated in one vectorized pass that reproduces the one-torus path
    # looped over the same tori to round-off, frequencies included, with tori
    # on the degenerate faces and the circular orbit mixed in, and whatever
    # the shape of the arrays (under a forced backend the outputs stay on it,
    # a single torus's frequencies as 0-d arrays)
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use

    def npa(out):
        return numpy.array(_spherical_inverse_npt(out))

    def ref(f, *args):
        # the one-torus path's values, on numpy also under a forced backend
        # (whose one-torus path agrees with numpy's to 1e-12, see
        # test_backend_actionAngleStaeckelInverse.py): the reference
        # evaluations would otherwise dominate an eager backend's run
        with use("numpy", force=True):
            return npa(f(*args))

    aAI = staeckel_inverse_interp
    rng = numpy.random.default_rng(4)
    n = 40
    jr = rng.uniform(0.01, 0.15, n)
    jz = rng.uniform(0.01, 0.15, n)
    Lz = rng.uniform(0.85, 1.2, n) * rng.choice([-1.0, 1.0], n)
    jr[3], jz[7], jr[11] = 0.0, 0.0, 0.0
    jz[11] = 0.0
    th = rng.uniform(0.0, 2.0 * numpy.pi, (3, n))
    out = npa(aAI(jr, Lz, jz, th[0], th[1], th[2]))
    Om = npa(aAI.Freqs(jr, Lz, jz))
    for ii in range(n):
        one = ref(aAI, jr[ii], Lz[ii], jz[ii], th[0, ii], th[1, ii], th[2, ii])
        assert numpy.amax(numpy.fabs(out[:, ii] - one[:, 0])) < 1e-11, (
            "The array evaluation differs from the one-torus path at torus %d: %g"
            % (ii, numpy.amax(numpy.fabs(out[:, ii] - one[:, 0])))
        )
        # the array path finds the energy variable's edges on the raw
        # potential's midplane, which the Staeckel wrapper reproduces to its
        # own arithmetic (1e-12), where the one-torus path uses the wrapper
        assert (
            numpy.amax(numpy.fabs(Om[:, ii] - ref(aAI.Freqs, jr[ii], Lz[ii], jz[ii])))
            < 1e-10
        ), "The array frequencies differ from the one-torus path"
    # tiny actions, whose label inversion the one-torus path rescues by a
    # restart, agree too
    tiny = numpy.array([1e-12, 1e-8, 1e-8, 0.05])
    tinyz = numpy.array([1e-12, 1e-8, 0.05, 1e-8])
    Lt = numpy.ones(4)
    outt = npa(aAI(tiny, Lt, tinyz, 0.3, 1.0, 2.0))
    for ii in range(4):
        one = ref(aAI, tiny[ii], Lt[ii], tinyz[ii], 0.3, 1.0, 2.0)
        assert numpy.amax(numpy.fabs(outt[:, ii] - one[:, 0])) < 1e-11, (
            "The array evaluation differs from the one-torus path at a tiny action"
        )
    # a family whose Newton cannot converge: every torus takes the one-torus
    # path's bracketed fallback, and the answers agree all the same (on numpy
    # also under a forced backend: an eager backend's safeguarded steps take
    # minutes; test_backend_actionAngleStaeckelInverse.py's
    # test_array_actions_fallbacks covers the backends' fallback)
    with use("numpy", force=True):
        slow = actionAngleStaeckelInverse(
            pot=_staeckel_inverse_potential(),
            setup_interp=True,
            Rmin=0.7,
            Rmax=1.6,
            Rinf=3.5,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=48,
            maxiter=1,
            progressbar=False,
        )
    outs = ref(slow, jr[:5], Lz[:5], jz[:5], th[0, :5], th[1, :5], th[2, :5])
    for ii in range(5):
        one = ref(slow, jr[ii], Lz[ii], jz[ii], th[0, ii], th[1, ii], th[2, ii])
        assert numpy.amax(numpy.fabs(outs[:, ii] - one[:, 0])) < 1e-9, (
            "The array evaluation's fallback differs from the one-torus path"
        )
    # explicit tori: the arrays are evaluated torus by torus (the tori set up
    # on numpy, as the fixture's family: their evaluation is what is tested)
    E, Lzb, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    with use("numpy", force=True):
        aAE = actionAngleStaeckelInverse(
            pot=_staeckel_inverse_potential(),
            Es=[E],
            Lzs=[Lzb],
            I3s=[I3],
            progressbar=False,
        )
    jrb, jzb = (
        float(as_numpy(aAE.JR(E, Lzb, I3))),
        float(as_numpy(aAE.Jz(E, Lzb, I3))),
    )
    outE = npa(
        aAE(
            numpy.array([jrb, jrb]),
            numpy.array([Lzb, Lzb]),
            numpy.array([jzb, jzb]),
            numpy.array([0.3, 1.1]),
            numpy.array([1.0, 2.0]),
            numpy.array([2.0, 0.5]),
        )
    )
    for ii, (ar, ap, az) in enumerate(((0.3, 1.0, 2.0), (1.1, 2.0, 0.5))):
        one = ref(aAE, jrb, Lzb, jzb, ar, ap, az)
        assert numpy.amax(numpy.fabs(outE[:, ii] - one[:, 0])) < 1e-12
    assert (
        numpy.amax(
            numpy.fabs(
                npa(
                    aAE.Freqs(
                        numpy.array([jrb, jrb]),
                        numpy.array([Lzb, Lzb]),
                        numpy.array([jzb, jzb]),
                    )
                )[:, 0]
                - ref(aAE.Freqs, jrb, Lzb, jzb)
            )
        )
        < 1e-12
    )
    # guards, as for one torus
    with pytest.raises(ValueError, match="non-negative"):
        aAI(numpy.array([0.05, -0.1]), 0.9, 0.05, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="L_z = 0"):
        aAI(0.05, numpy.array([0.9, 0.0]), 0.05, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        aAI(0.05, numpy.array([0.9, 5.0]), 0.05, 1.0, 1.0, 1.0)
    # a torus beyond the family's actions: the batched label inversion hands
    # it to the one-torus inversion, which says so
    with pytest.raises(ValueError, match="could not be matched"):
        aAI(numpy.array([0.05, 0.47]), 0.9, numpy.array([0.05, 0.34]), 1.0, 1.0, 1.0)
    return None


def test_actionAngleStaeckelInverse_array_actions_shapes(staeckel_inverse_interp):
    # The shapes of an array evaluation (test_actionAngleStaeckelInverse_
    # array_actions's broadcasting checks, split from it so that each part
    # runs in the time an eager backend allows a test: an eager jax pass
    # compiles its operations anew for every new shape)

    def npa(out):
        return numpy.array(_spherical_inverse_npt(out))

    def is_scalar(o):
        return isinstance(o, float) or (is_backend_array(o) and o.ndim == 0)

    aAI = staeckel_inverse_interp
    rng = numpy.random.default_rng(4)
    n = 40
    jr = rng.uniform(0.01, 0.15, n)
    jz = rng.uniform(0.01, 0.15, n)
    Lz = rng.uniform(0.85, 1.2, n) * rng.choice([-1.0, 1.0], n)
    jr[3], jz[7], jr[11] = 0.0, 0.0, 0.0
    jz[11] = 0.0
    th = rng.uniform(0.0, 2.0 * numpy.pi, (3, n))
    # the arrays' shape is kept, and a scalar action broadcasts
    out2 = npa(
        aAI(
            jr[:6].reshape(2, 3),
            Lz[:6].reshape(2, 3),
            0.05,
            th[0, :6].reshape(2, 3),
            1.0,
            2.0,
        )
    )
    assert (
        out2[0].shape == (2, 3)
        and numpy.amax(
            numpy.fabs(
                out2[0]
                - npa(aAI(jr[:6], Lz[:6], 0.05, th[0, :6], 1.0, 2.0))[0].reshape(2, 3)
            )
        )
        < 1e-12
    ), "The array evaluation does not keep the arrays' shape"
    # the broadcasting rules: an array of actions at a scalar angle is that
    # angle on every torus, actions of shape (n,1) with angles of shape (m,)
    # are every angle on every torus, and a scalar torus takes angles of any
    # shape and returns its frequencies as floats
    ang = numpy.linspace(0.2, 6.0, 5)
    grid = npa(
        aAI.xvFreqs(jr[:4, None], Lz[:4, None], jz[:4, None], ang, 2.0 * ang, ang[::-1])
    )
    assert grid.shape == (9, 4, 5), "(n,1) actions with (m,) angles are not (n,m)"
    fixed = npa(aAI.xvFreqs(jr[:4], Lz[:4], jz[:4], ang[1], 2.0 * ang[1], ang[3]))
    assert fixed.shape == (9, 4), "Arrays of actions at a scalar angle are not (n,)"
    assert numpy.amax(numpy.fabs(fixed - grid[:, :, 1])) < 1e-11, (
        "Arrays of actions at a scalar angle differ from the same points of the grid"
    )
    for ii in range(4):
        one = aAI.xvFreqs(jr[ii], Lz[ii], jz[ii], ang, 2.0 * ang, ang[::-1])
        assert all(is_scalar(o) for o in one[6:]), (
            "The frequencies of a single torus are not floats"
        )
        one = _spherical_inverse_npt(one)
        assert numpy.amax(numpy.fabs(grid[:6, ii] - numpy.array(one[:6]))) < 1e-11, (
            "Every angle on every torus differs from the one-torus path"
        )
        assert numpy.amax(numpy.fabs(grid[6:, ii, 0] - numpy.array(one[6:]))) < 1e-10
    two = _spherical_inverse_npt(
        aAI(jr[1], Lz[1], jz[1], ang[:4].reshape(2, 2), 1.0, 2.0)
    )
    assert two[0].shape == (2, 2) and numpy.allclose(
        two[0].ravel(),
        as_numpy(aAI(jr[1], Lz[1], jz[1], ang[:4], 1.0, 2.0)[0]),
        rtol=1e-13,
    ), "A single torus does not keep the shape of its angles"
    assert npa(aAI.Freqs(jr[:4, None], Lz[:4, None], jz[:4, None])).shape == (
        3,
        4,
        1,
    ), "The frequencies of an array of tori do not keep its shape"
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
    from galpy.backend import use
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
    jr = float(as_numpy(aAP.JR(E, Lz, I3)))
    R, vR, vT, z, vz, phi = (
        as_numpy(t) for t in aAP(jr, Lz, 0.0, angler, anglephi, anglez)
    )
    assert numpy.amax(numpy.fabs(z)) < 1e-15 and numpy.amax(numpy.fabs(vz)) < 1e-15, (
        "The planar orbit does not stay in the plane"
    )
    H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
    assert numpy.ptp(H) / numpy.fabs(E) < 1e-12, "The planar orbit is not at one energy"
    f = [as_numpy(t) for t in aAS(R, vR, vT, z, vz, phi)]
    assert (
        numpy.amax(numpy.fabs(f[0] - jr)) < 1e-9
        and numpy.amax(numpy.fabs(f[2])) < 1e-12
    ), "The planar orbit's radial action is not the forward transformation's"
    # shell
    E, Lz, _ = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    with use("numpy", force=True):  # the numpy path's edge relation
        Ish = float(aAP._I3_shell(E, Lz, aAP._ushell(E, Lz)))
    aASh = actionAngleStaeckelInverse(pot=pot, Es=[E], Lzs=[Lz], I3s=[Ish])
    assert aASh.JR(E, Lz, Ish) == 0.0, "The shell orbit's radial action is not zero"
    jz = float(as_numpy(aASh.Jz(E, Lz, Ish)))
    R, vR, vT, z, vz, phi = (
        as_numpy(t) for t in aASh(0.0, Lz, jz, angler, anglephi, anglez)
    )
    H = 0.5 * (vR**2 + vT**2 + vz**2) + as_numpy(evaluatePotentials(pot, R, z))
    assert numpy.ptp(H) / numpy.fabs(E) < 1e-12, "The shell orbit is not at one energy"
    f = [as_numpy(t) for t in aAS(R, vR, vT, z, vz, phi)]
    assert (
        numpy.amax(numpy.fabs(f[0])) < 1e-7 and numpy.amax(numpy.fabs(f[2] - jz)) < 1e-7
    ), (
        "The shell orbit's actions are not the forward transformation's: {:g} {:g}".format(
            numpy.amax(numpy.fabs(f[0])),
            numpy.amax(numpy.fabs(f[2] - jz)),
        )
    )
    assert numpy.all(
        numpy.fabs(numpy.array([as_numpy(o) for o in aASh.Freqs(0.0, Lz, jz)])) > 0.0
    ), "The shell orbit's frequencies are not all finite and non-zero"
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


def test_actionAngleSphericalInverse_auxiliary_circular():
    # A family of one circular torus samples the rotation curve at a single
    # radius; every isochrone scale radius fits it, and the fit must return a
    # finite auxiliary (the isochrone with that radius as its scale radius and
    # the target's circular speed there) rather than NaN
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import HernquistPotential, evaluatePotentials, vcirc

    hp = HernquistPotential(amp=2.0, a=1.0)
    R = 1.3
    # (as_numpy: under a forced backend the results stay on the backend)
    vc = float(as_numpy(vcirc(hp, R)))
    Ec = 0.5 * vc**2.0 + float(as_numpy(evaluatePotentials(hp, R, 0.0)))
    aAI = actionAngleSphericalInverse(
        pot=hp, Es=[Ec], Ls=[R * vc], mm_npt=24, progressbar=False
    )
    aux = aAI.auxiliary
    assert numpy.isfinite(aux._amp) and numpy.isfinite(aux.b), (
        "The auxiliary of a lone circular torus is not finite"
    )
    assert numpy.fabs(aux.b - R) < 1e-10, (
        "The auxiliary of a lone circular torus does not keep its scale radius at that radius"
    )
    assert numpy.fabs(float(as_numpy(vcirc(aux, R))) / vc - 1.0) < 1e-10, (
        "The auxiliary of a lone circular torus does not match the target's circular speed there"
    )
    R, vR, vT, z, vz, phi = _spherical_inverse_npt(
        aAI(0.0, 0.6 * R * vc, 0.4 * R * vc, 0.3, 0.2, 1.1)
    )
    assert numpy.all(numpy.isfinite([R, vR, vT, z, vz, phi])), (
        "A lone circular torus does not evaluate"
    )
    return None


def test_actionAngleStaeckelInverse_auxiliary_circular():
    # A family of one planar circular torus samples the rotation curve at a
    # single radius, which every isochrone scale radius fits: the auxiliary
    # must be finite (it used to come out as NaN) and the torus evaluates to
    # the circular orbit
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import evaluatePotentials, rl, vcirc

    pot = _staeckel_inverse_potential()
    Lz = 0.93
    # (as_numpy: under a forced backend the results stay on the backend)
    Rc = float(as_numpy(rl(pot, Lz, use_physical=False)))
    Ec = float(as_numpy(evaluatePotentials(pot, Rc, 0.0))) + Lz**2 / (2.0 * Rc**2)
    Ipl = Lz**2 / (2.0 * _STAECKEL_INVERSE_DELTA**2) - Ec
    aAI = actionAngleStaeckelInverse(
        pot=pot, Es=[Ec], Lzs=[Lz], I3s=[Ipl], progressbar=False
    )
    aux = aAI.auxiliary
    assert numpy.isfinite(aux._amp) and numpy.isfinite(aux.b), (
        "The auxiliary of a lone circular torus is not finite"
    )
    assert numpy.fabs(as_numpy(vcirc(aux, Rc) / vcirc(pot, Rc)) - 1.0) < 1e-10, (
        "The auxiliary of a lone circular torus does not match the target's circular speed at its radius"
    )
    angler = numpy.linspace(0.05, 6.2, 11)
    R, vR, vT, z, vz, phi = _spherical_inverse_npt(
        aAI(0.0, Lz, 0.0, angler, 0.3 + angler, 0.7 + angler)
    )
    assert numpy.all(numpy.isfinite([R, vR, vT, z, vz, phi])), (
        "A lone circular torus does not evaluate"
    )
    assert (
        numpy.amax(numpy.fabs(R / Rc - 1.0)) < 1e-10
        and numpy.amax(numpy.fabs(vR)) < 1e-10
    ), "A lone circular torus does not sit at the circular radius"
    return None


def test_actionAngleSphericalInverse_E():
    # The public E(jr, L) is the inverse of Jr(E, L): on an interpolated family
    # it returns, to round-off, the energy whose radial action at that L is
    # jr, for scalars and for arrays that broadcast; actions outside the grid
    # raise; an instance set up with explicit tori returns their energies and
    # refuses other actions
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import HernquistPotential, evaluatePotentials, vcirc

    hp = HernquistPotential(amp=2.0, a=1.0)
    Rmin, Rmax, Rinf = 0.5, 2.0, 10.0
    aAI = actionAngleSphericalInverse(
        pot=hp,
        setup_interp=True,
        Rmin=Rmin,
        Rmax=Rmax,
        Rinf=Rinf,
        nE=8,
        nL=8,
        mm_npt=32,
        progressbar=False,
    )
    # (as_numpy: under a forced backend the results stay on the backend)
    Einf = float(as_numpy(evaluatePotentials(hp, Rinf, 0.0)))
    Rs = numpy.array([0.6, 1.0, 1.7])
    Ls = Rs * as_numpy(vcirc(hp, Rs))
    for L, R in zip(Ls, Rs):
        Ec = 0.5 * float(as_numpy(vcirc(hp, R))) ** 2.0 + float(
            as_numpy(evaluatePotentials(hp, R, 0.0))
        )
        Es = Ec + (Einf - Ec) * numpy.array([0.05, 0.3, 0.7, 0.95])
        jrs = numpy.array([float(as_numpy(aAI.Jr(E, L))) for E in Es])
        Eback = as_numpy(aAI.E(jrs, L))
        assert Eback.shape == jrs.shape, "E(jr, L) does not keep the shape of jr"
        assert numpy.all(numpy.fabs(Eback - Es) < 1e-10 * (1.0 + numpy.fabs(Es))), (
            "E(Jr(E, L), L) does not return E"
        )
        Es1 = aAI.E(float(jrs[1]), L)
        assert _float_or_0d(Es1) and numpy.fabs(as_numpy(Es1) - Es[1]) < 1e-10, (
            "E(jr, L) for scalars does not return the float energy"
        )
    # broadcasting: (n, 1) actions against (m,) angular momenta
    jr_grid = numpy.array([[0.02], [0.05]])
    Eb = as_numpy(aAI.E(jr_grid, Ls))
    assert Eb.shape == (2, 3), "E(jr, L) does not broadcast jr against L"
    for i in range(2):
        for j in range(3):
            assert (
                numpy.fabs(as_numpy(aAI.Jr(Eb[i, j], Ls[j])) - jr_grid[i, 0]) < 1e-10
            ), "E(jr, L) with broadcast inputs is not the inverse of Jr(E, L)"
    with pytest.raises(ValueError):
        aAI.E(10.0, Ls[0])
    # explicit tori
    Es_exp, Ls_exp = [-0.35, -0.3], [0.5, 0.6]
    aAIe = actionAngleSphericalInverse(
        pot=hp, Es=Es_exp, Ls=Ls_exp, mm_npt=32, progressbar=False
    )
    jr0, jr1 = (float(as_numpy(aAIe.Jr(E, L))) for E, L in zip(Es_exp, Ls_exp))
    assert as_numpy(aAIe.E(jr1, Ls_exp[1])) == Es_exp[1], (
        "E(jr, L) of an explicit torus is not its energy"
    )
    assert numpy.all(
        as_numpy(aAIe.E(numpy.array([jr0, jr1]), numpy.array(Ls_exp)))
        == numpy.array(Es_exp)
    ), "E(jr, L) of explicit tori does not return their energies for arrays"
    with pytest.raises(ValueError):
        aAIe.E(jr1 * 1.1, Ls_exp[1])
    return None


def test_actionAngleVerticalInverse_E():
    # The public E(j) is the inverse of J(E): on a momentum-matched family it
    # is the family's Hermite E(J), exact at the nodes with the frequency as
    # its slope between them, for scalars and arrays; actions outside the
    # family raise; a torus-mapping instance returns the energies of its
    # set-up tori and refuses other actions
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import MWPotential2014, toVerticalPotential

    pot = toVerticalPotential(MWPotential2014, 1.0)
    Es = numpy.linspace(0.0, 0.08, 9)[1:]
    aAV = actionAngleVerticalInverse(pot=pot, Es=Es, setup_interp=True)
    # (as_numpy: under a forced backend the results stay on the backend)
    js = numpy.array([float(as_numpy(aAV.J(E))) for E in Es])
    assert numpy.all(numpy.fabs(as_numpy(aAV.E(js)) - Es) < 1e-12), (
        "E(J(E)) does not return E at the nodes"
    )
    assert _float_or_0d(aAV.E(float(js[2]))), (
        "E(j) for a scalar does not return a float"
    )
    jmid = 0.5 * (js[2] + js[3])
    h = 1e-5 * jmid
    dEdj = float(as_numpy(aAV.E(jmid + h) - aAV.E(jmid - h))) / (2.0 * h)
    assert numpy.fabs(dEdj / float(as_numpy(aAV.Freqs(jmid))) - 1.0) < 1e-6, (
        "The slope of E(j) between the nodes is not the family's frequency"
    )
    with pytest.raises(ValueError):
        aAV.E(2.0 * js[-1])
    from galpy.backend import use

    # the torus-mapping construction is numpy-only (its lookups run on any
    # backend)
    with use("numpy", force=True):
        aAVtm = actionAngleVerticalInverse(
            pot=toVerticalPotential(MWPotential2014, 1.0),
            Es=Es[:3],
            momentum_matched=False,
        )
    jtm = aAVtm.J(Es[1])
    assert as_numpy(aAVtm.E(jtm)) == Es[1], (
        "E(j) of a torus-mapping instance is not the set-up energy"
    )
    with pytest.raises(ValueError):
        aAVtm.E(1.5 * jtm)


def test_actionAngleVertical_arrays_and_small_tori():
    # The vectorized forward transformation: arrays of points in one call give
    # the same actions, frequencies and angles as the points one by one, floats
    # give one-element arrays, bound and unbound points mix with the 9999.99
    # sentinel, a point at its turning point or at the midplane is fine, and
    # the smallest tori of a vertical potential measured from the midplane,
    # whose energy above the minimum is within the potential's round-off, have
    # the harmonic frequency with the anharmonic correction linear in the
    # action down to actions of 1e-13
    from galpy.actionAngle import actionAngleVertical
    from galpy.backend import use
    from galpy.potential import MWPotential2014, toVerticalPotential, verticalfreq

    pot = toVerticalPotential(MWPotential2014, 1.0)
    aAV = actionAngleVertical(pot=pot)
    # (as_numpy: under a forced backend the results stay on the backend)
    npt = _spherical_inverse_npt
    x = numpy.array([0.05, -0.2, 0.3, 0.0, 0.01, -0.15, 0.1])
    vx = numpy.array([0.1, 0.3, 0.0, 0.5, 0.02, -0.25, 10.0])
    J, Om, ang = npt(aAV.actionsFreqsAngles(x, vx))
    assert J.shape == x.shape and Om.shape == x.shape and ang.shape == x.shape
    for ii in range(len(x) - 1):
        Ji, Omi, angi = npt(aAV.actionsFreqsAngles(float(x[ii]), float(vx[ii])))
        assert Ji.shape == (1,), "A float input does not give a one-element array"
        assert numpy.fabs(Ji[0] - J[ii]) < 1e-14 * (1.0 + J[ii]), (
            "Array and single-point actions differ"
        )
        assert numpy.fabs(Omi[0] - Om[ii]) < 1e-13 * Om[ii], (
            "Array and single-point frequencies differ"
        )
        assert numpy.fabs(angi[0] - ang[ii]) < 1e-12, (
            "Array and single-point angles differ"
        )
        assert numpy.fabs(
            as_numpy(aAV(float(x[ii]), float(vx[ii])))[0] - J[ii]
        ) < 1e-14 * (1.0 + J[ii])
        Jf, Omf = npt(aAV.actionsFreqs(float(x[ii]), float(vx[ii])))
        assert (
            numpy.fabs(Jf[0] - J[ii]) < 1e-14 * (1.0 + J[ii])
            and numpy.fabs(Omf[0] - Om[ii]) < 1e-13 * Om[ii]
        )
    assert J[-1] == 9999.99 and Om[-1] == 9999.99, (
        "An unbound point does not return the sentinel"
    )
    assert numpy.all(J[:-1] < 9999.0), "A bound point returns the sentinel"
    # at the turning point the angle is pi/2, at the midplane with vx > 0 it is 0
    assert numpy.fabs(ang[2] - 0.5 * numpy.pi) < 1e-10, (
        "The angle at the turning point is not pi/2"
    )
    assert numpy.fabs(ang[3]) < 1e-12, "The angle at the midplane is not 0"
    # the harmonic limit: (Omega / nu0 - 1) / vz^2 tends to a constant
    nu0 = float(as_numpy(verticalfreq(MWPotential2014, 1.0)))
    vzs = numpy.array([1e-3, 1e-4, 1e-5, 1e-6])
    Js, Oms, angs = npt(aAV.actionsFreqsAngles(0.3 * vzs / nu0, 0.95 * vzs))
    coeff = (Oms / nu0 - 1.0) / vzs**2.0
    assert numpy.all(numpy.fabs(coeff / coeff[0] - 1.0) < 1e-3), (
        "The frequencies of the smallest tori do not follow the anharmonic correction: round-off"
    )
    assert numpy.all(numpy.fabs(Js / (vzs**2.0 / 2.0 / nu0) / 0.9925 - 1.0) < 1e-3), (
        "The actions of the smallest tori are not the harmonic ones"
    )
    assert numpy.all(numpy.fabs(angs - angs[0]) < 1e-5), (
        "The angles of the smallest tori are not self-similar"
    )
    # every point at its turning point
    Jt, Omt, angt = npt(
        aAV.actionsFreqsAngles(numpy.array([0.1, -0.2]), numpy.array([0.0, 0.0]))
    )
    assert numpy.all(numpy.fabs(angt - numpy.array([0.5, 1.5]) * numpy.pi) < 1e-10), (
        "Points at their turning points do not have angle pi/2 (above) and 3 pi/2 (below the midplane)"
    )
    assert numpy.fabs(Jt[0] - as_numpy(aAV(0.1, 0.0))[0]) < 1e-14, (
        "A point at its turning point differs between array and float calls"
    )
    # the single-point turning-point method agrees with the vectorized one
    # (both numpy internals)
    with use("numpy", force=True):
        xm = aAV.calcxmax(0.05, 0.1)
        assert numpy.fabs(xm - aAV._points(0.05, 0.1)[3][0]) < 1e-12, (
            "calcxmax disagrees with the vectorized turning point"
        )
        assert aAV.calcxmax(0.3, 0.0) == 0.3, "calcxmax at the turning point is not |x|"
        assert (
            numpy.fabs(aAV.calcxmax(0.0, 0.5) - aAV._points(0.0, 0.5)[3][0]) < 1e-12
        ), "calcxmax from the midplane disagrees"
    return None


def test_actionAngleVerticalInverse_small_tori():
    # The smallest tori of a vertical potential measured from the midplane:
    # the construction samples the momentum through the forward
    # transformation's radicand, which uses the force where the potential's
    # own round-off would spoil E - Phi, so the tori build without the
    # convergence warning, round-trip through the forward transformation, and
    # the bottom torus has the harmonic frequency
    import warnings

    from galpy.actionAngle import actionAngleVertical, actionAngleVerticalInverse
    from galpy.potential import (
        MWPotential2014,
        evaluatelinearPotentials,
        toVerticalPotential,
        verticalfreq,
    )

    pot = toVerticalPotential(MWPotential2014, 1.0)
    # (as_numpy: under a forced backend the results stay on the backend)
    Phi0 = float(as_numpy(evaluatelinearPotentials(pot, 0.0, use_physical=False)))
    Es = Phi0 + numpy.concatenate([[0.0], numpy.geomspace(1e-9, 1e-3, 13)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aAVI = actionAngleVerticalInverse(pot=pot, Es=Es, setup_interp=True)
    assert not any("converged" in str(x.message) for x in w), (
        "The smallest tori trigger the convergence warning: round-off in E - Phi"
    )
    nu0 = float(as_numpy(verticalfreq(MWPotential2014, 1.0)))
    assert numpy.fabs(float(as_numpy(aAVI.Freqs(0.0))) / nu0 - 1.0) < 1e-9, (
        "The bottom torus does not have the harmonic frequency"
    )
    aAV = actionAngleVertical(pot=pot)
    angles = numpy.linspace(0.1, 6.1, 25)
    for E in Es[1:6]:
        J = float(as_numpy(aAVI.J(E)))
        x, v = aAVI(J, angles)
        Jf, Of, af = _spherical_inverse_npt(aAV.actionsFreqsAngles(x, v))
        assert numpy.all(numpy.fabs(Jf / J - 1.0) < 1e-8), (
            "The smallest tori do not round-trip in the action"
        )
        d = (af - angles + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.all(numpy.fabs(d) < 1e-7), (
            "The smallest tori do not round-trip in the angle"
        )
        assert numpy.all(
            numpy.fabs(Of / float(as_numpy(aAVI.Freqs(J))) - 1.0) < 1e-8
        ), "The smallest tori do not round-trip in the frequency"
    return None


def test_actionAngleSphericalInverse_Lspacing():
    # The angular-momentum grid's spacing: 'log' (the default), 'sqrt' and
    # 'linear' all build and evaluate; a grid that reaches a small Rmin has a
    # linear innermost cell that spans a large factor in L and holds its
    # most eccentric tori, where the 'log' spacing is far more accurate, as
    # the family's own J_r(E, L) against the exact action of a torus set up
    # on its own shows; an unknown spacing raises
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import HernquistPotential, evaluatePotentials, vcirc

    hp = HernquistPotential(amp=2.0, a=1.0)
    Rmin, Rmax, Rinf = 0.1, 5.0, 20.0
    kw = dict(
        pot=hp,
        setup_interp=True,
        Rmin=Rmin,
        Rmax=Rmax,
        Rinf=Rinf,
        nE=8,
        nL=8,
        mm_npt=160,
        progressbar=False,
    )
    fams = {
        sp: actionAngleSphericalInverse(Lspacing=sp, **kw)
        for sp in ("linear", "sqrt", "log")
    }

    # (as floats: under a forced backend the results stay on the backend)
    def vc(R):
        return float(as_numpy(vcirc(hp, R)))

    def Phi(R):
        return float(as_numpy(evaluatePotentials(hp, R, 0.0)))

    Lmin = Rmin * vc(Rmin)
    Ltest = numpy.sqrt(
        Lmin * (Lmin + (Rmax * vc(Rmax) - Lmin) / 7.0)
    )  # inside the linear grid's first cell
    Rc = 0.25
    Ec = 0.5 * vc(Rc) ** 2.0 + Phi(Rc)
    E = Ec + 0.5 * (Phi(Rinf) - Ec)
    jr_exact = actionAngleSphericalInverse(
        pot=hp, Es=[E], Ls=[Ltest], mm_npt=160, progressbar=False
    ).Jr(E, Ltest)
    err = {
        sp: numpy.fabs(float(as_numpy(f.Jr(E, Ltest) / jr_exact)) - 1.0)
        for sp, f in fams.items()
    }
    assert err["log"] < 1e-2 * err["linear"], (
        "The 'log' spacing does not improve the innermost cell's action over the 'linear' one"
    )
    assert err["sqrt"] < err["linear"], (
        "The 'sqrt' spacing does not improve the innermost cell's action"
    )
    for f in fams.values():
        R, vR, vT, z, vz, phi = _spherical_inverse_npt(
            f(0.05, 0.6 * Ltest, 0.4 * Ltest, 0.3, 0.2, 1.1)
        )
        assert numpy.all(numpy.isfinite([R, vR, vT, z, vz, phi])), (
            "A spaced family does not evaluate"
        )
    # the default is the 'log' spacing: the family built without Lspacing= is
    # the 'log' one, and not the 'linear' one
    default = actionAngleSphericalInverse(**kw)
    point = (0.05, 0.6 * Ltest, 0.4 * Ltest, 0.3, 0.2, 1.1)
    jr_default = default.Jr(E, Ltest)
    # (to round-off under a forced backend: jax's CPU FFT is not bitwise
    # reproducible from one build to the next)
    tol = 1e-13 if is_backend_array(jr_default) else 0.0
    jr_default = _spherical_inverse_np(jr_default)
    assert numpy.fabs(
        jr_default - _spherical_inverse_np(fams["log"].Jr(E, Ltest))
    ) <= tol and numpy.all(
        numpy.fabs(
            _spherical_inverse_xv(default, *point)
            - _spherical_inverse_xv(fams["log"], *point)
        )
        <= tol
    ), "The default spacing of the angular-momentum grid is not 'log'"
    assert jr_default != _spherical_inverse_np(fams["linear"].Jr(E, Ltest)), (
        "The default spacing of the angular-momentum grid is still 'linear'"
    )
    with pytest.raises(ValueError):
        actionAngleSphericalInverse(
            pot=hp,
            setup_interp=True,
            Rmin=0.5,
            Rmax=2.0,
            Rinf=10.0,
            nE=6,
            nL=6,
            Lspacing="quadratic",
            progressbar=False,
        )
    return None


@pytest.fixture(scope="module")
def spherical_inverse_eccentric():
    # an (E, L) grid in the logarithmic halo that reaches eccentric tori
    # (apocentres up to fifty times the pericentre), set up with more
    # harmonics than any of its tori keeps
    from galpy.actionAngle import actionAngleSphericalInverse

    return actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        setup_interp=True,
        Rmin=0.2,
        Rmax=1.0,
        Rinf=4.0,
        nE=8,
        nL=8,
        mm_npt=64,
        progressbar=False,
    )


def _spherical_inverse_eccentric_jr(aAI, u, L):
    # the radial action at the grid coordinates (u, L) of the eccentric grid
    return aAI.Jr(_spherical_inverse_E_of_u(u, L, Rinf=4.0), L)


def _spherical_inverse_eccentric_points(aAI):
    # one point inside each of the 7 x 7 cells of the eccentric grid (uniform
    # in u, logarithmic in L), with random inclinations and angles
    us, Ls = numpy.linspace(0.0, 1.0, 8), numpy.geomspace(0.2, 1.0, 8)
    a, b = numpy.divmod(numpy.arange(49), 7)
    u = us[a] + 0.37 * (us[a + 1] - us[a])
    L = Ls[b] + 0.61 * (Ls[b + 1] - Ls[b])
    jr = numpy.array(
        [_spherical_inverse_eccentric_jr(aAI, tu, tL) for tu, tL in zip(u, L)]
    )
    rng = numpy.random.default_rng(7)
    jphi = rng.uniform(-1.0, 1.0, 49) * L
    return (jr, jphi, L - numpy.fabs(jphi), *rng.uniform(0.0, 2.0 * numpy.pi, (3, 49)))


def test_actionAngleSphericalInverse_adaptive_harmonics(spherical_inverse_eccentric):
    # Every torus of an interpolated family keeps the harmonics of its map
    # down to its round-off floor (those up to the last whose coefficient
    # exceeds 1e-11 and as many more as three further decades take at the
    # rate of the coefficients' fall), and every point is evaluated with the
    # harmonics of its cell: the largest number that the four tori at its
    # corners keep, rounded up to 16, 32, ... or mm_npt. On a grid whose
    # cells use 16, 32 and all 64 harmonics: (a) an array of points spread
    # over all cells, shuffled and as a 2D array, agrees with the points
    # evaluated one at a time; (b) the points round-trip through the forward
    # transformation within the grid's interpolation error; (c) the map is
    # canonical in a cell of each kind; (d) it is continuous across the
    # edges between cells that use different numbers; (e) the grid's own
    # tori are as accurate as with all harmonics
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    aAI = spherical_inverse_eccentric
    us, Ls = numpy.linspace(0.0, 1.0, 8), numpy.geomspace(0.2, 1.0, 8)
    # (a) the same points one at a time, in another order, and as a 2D array
    pts = _spherical_inverse_eccentric_points(aAI)
    out = numpy.array(aAI.xvFreqs(*pts))
    one = numpy.array(
        [
            [numpy.ravel(o)[0] for o in aAI.xvFreqs(*(t[k] for t in pts))]
            for k in range(49)
        ]
    ).T
    perm = numpy.random.default_rng(8).permutation(49)
    shuffled = numpy.empty((9, 49))
    shuffled[:, perm] = aAI.xvFreqs(*(t[perm] for t in pts))
    grid = numpy.array(aAI.xvFreqs(*(t.reshape(7, 7) for t in pts)))
    assert grid.shape == (9, 7, 7), "A 2D array of points does not keep its shape"
    for other, what in (
        (one, "one at a time"),
        (shuffled, "in another order"),
        (grid.reshape(9, 49), "as a 2D array"),
    ):
        assert numpy.all(numpy.fabs(out - other) < 1e-11 * (1.0 + numpy.fabs(out))), (
            "Points grouped by their numbers of harmonics differ from the same "
            "points evaluated %s" % what
        )
    # (b) the round trip in a cell with 16 harmonics, one with 32 and one with
    # 64: within the tolerance of the between-node test of the small grid
    # where this grid's interpolation error allows, and within twice the
    # error of the same family evaluated with all harmonics in the eccentric
    # cell (7e-4 in the angles, 1.5e-4 in the frequencies)
    angler = numpy.linspace(0.05, 6.2, 21)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    cells = ((1, 3), (3, 4), (6, 5))
    for (a, b), tols in zip(
        cells, ((1e-4,) * 4, (1e-4,) * 4, (1e-4, 1.5e-3, 3e-4, 1e-4))
    ):
        L = Ls[b] + 0.61 * (Ls[b + 1] - Ls[b])
        jr = _spherical_inverse_eccentric_jr(aAI, us[a] + 0.37 * (us[a + 1] - us[a]), L)
        errs = _spherical_inverse_roundtrip(
            aAI, jr, 0.6 * L, 0.4 * L, angler, anglephi, anglez
        )
        assert all(err < tol for err, tol in zip(errs, tols)), (
            "The round trip (actions, angles, frequencies, energy) in the cell "
            "(%i, %i) is not within the grid's interpolation error: %s" % (a, b, errs)
        )
    # (c) the symplectic defect at the finite-difference floor, with the
    # bound of the small grid's test (the defect is 3e-10 to 2e-9 at these
    # points, also when all harmonics are read), in a cell with 16
    # harmonics, one with 32, and one whose tori keep different numbers
    # (34 and 35 against 43 and 45), all away from the circular edge and
    # from the cells' edges, where the finite differences themselves fail
    floor = _spherical_inverse_symplectic_defect(
        actionAngleIsochroneInverse(ip=IsochronePotential(amp=1.0, b=0.5)),
        0.2,
        0.6,
        0.3,
        0.7,
        1.0,
        2.0,
    )
    for a, b in ((1, 3), (3, 4), (5, 0)):
        L = Ls[b] + 0.61 * (Ls[b + 1] - Ls[b])
        jr = _spherical_inverse_eccentric_jr(aAI, us[a] + 0.37 * (us[a + 1] - us[a]), L)
        defect = _spherical_inverse_symplectic_defect(
            aAI, jr, 0.6 * L, 0.4 * L, 0.7, 1.0, 2.0
        )
        assert defect < 20.0 * floor + 1e-9, (
            "The symplectic defect in the cell (%i, %i) is above the "
            "finite-difference floor: %g vs %g" % (a, b, defect, floor)
        )
    # (d) points at a distance d either side of an edge between cells that
    # use different numbers of harmonics differ in proportion to d (the map
    # changes by up to 13 d there): across two edges in u and one in L
    edges = (
        ((1, 2), (2, 2), lambda s: (us[2] + s, 0.5 * (Ls[2] + Ls[3]))),
        ((3, 0), (4, 0), lambda s: (us[4] + s, 0.5 * (Ls[0] + Ls[1]))),
        ((4, 4), (4, 5), lambda s: (0.5 * (us[4] + us[5]), Ls[5] + s)),
    )
    for lo, hi, at in edges:
        for d in (1e-6, 1e-9):
            sides = []
            for s in (-d, d):
                u, L = at(s)
                sides.append(
                    numpy.ravel(
                        aAI(
                            _spherical_inverse_eccentric_jr(aAI, u, L),
                            0.6 * L,
                            0.4 * L,
                            1.3,
                            0.7,
                            2.1,
                        )
                    )
                )
            assert numpy.amax(numpy.fabs(sides[1] - sides[0])) < 30.0 * d, (
                "The map jumps across the edge between the cells %s and %s: %g "
                "at distance %g"
                % (lo, hi, numpy.amax(numpy.fabs(sides[1] - sides[0])), d)
            )
    # (e) what a torus of the grid drops is round-off: along every librating
    # torus of the grid, the energy of the returned points is the torus's to
    # 1e-10 of the squared speed (1.5e-11 at most and 1.5e-13 in the median;
    # keeping only the harmonics above 1e-11 leaves 5e-10 and 1e-11)
    from galpy.potential import evaluatePotentials

    u, L = (x.ravel() for x in numpy.meshgrid(us[1:], Ls, indexing="ij"))
    E = _spherical_inverse_E_of_u(u, L, Rinf=4.0)
    jr = numpy.array([float(as_numpy(aAI.Jr(tE, tL))) for tE, tL in zip(E, L)])
    R, vR, vT, z, vz, _ = (
        as_numpy(x)
        for x in aAI(
            jr,
            0.6 * L,
            0.4 * L,
            *numpy.random.default_rng(4).uniform(0.0, 6.28, (3, 12, 56)),
        )
    )
    v2 = vR**2 + vT**2 + vz**2
    dE = numpy.fabs(
        as_numpy(
            evaluatePotentials(
                _spherical_inverse_potential(), numpy.sqrt(R**2 + z**2), 0.0 * R
            )
        )
        + 0.5 * v2
        - E
    )
    assert numpy.all(dE < 1e-10 * v2), (
        "The grid's own tori do not conserve the energy to round-off: %g"
        % numpy.amax(dE / v2)
    )
    return None


def test_actionAngleSphericalInverse_harmonics_beyond_need(spherical_inverse_eccentric):
    # A torus's harmonics beyond those that it keeps are zero, in the
    # coefficients and in their slopes, which are solved for in the kept
    # harmonics alone: the same grid set up with half as many harmonics
    # again, still more than any torus keeps, is the same family, with the
    # same numbers of harmonics torus by torus (they are decided by
    # coefficients well above the round-off floor) and the same points to
    # the round-off of the two set-ups (8e-11; solving the slopes in all
    # mm_npt harmonics instead makes the two differ by 2e-9)
    from galpy.actionAngle import actionAngleSphericalInverse

    aAI = spherical_inverse_eccentric
    more = actionAngleSphericalInverse(
        pot=_spherical_inverse_potential(),
        setup_interp=True,
        Rmin=0.2,
        Rmax=1.0,
        Rinf=4.0,
        nE=8,
        nL=8,
        mm_npt=96,
        progressbar=False,
    )
    pts = _spherical_inverse_eccentric_points(aAI)
    out, out_more = numpy.array(aAI.xvFreqs(*pts)), numpy.array(more.xvFreqs(*pts))
    assert numpy.all(numpy.fabs(out - out_more) < 1e-9 * (1.0 + numpy.fabs(out))), (
        "The family depends on the number of harmonics beyond those its tori keep: %g"
        % (numpy.amax(numpy.fabs(out - out_more) / (1.0 + numpy.fabs(out))))
    )
    return None


def test_actionAngleSphericalInverse_frequencies_isochrone():
    # The frequencies of a torus are a quadrature in the anomaly between its
    # turning points, which are solved to round-off: for an isochrone
    # target, whose radial action and frequencies are closed forms, explicit
    # tori from nearly circular to eccentric return the radial action to
    # 1e-13 (3e-15) and the frequencies to 5e-12 (8e-13; turning points
    # solved to an absolute 1e-12 leave 1e-10 in the frequencies)
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import IsochronePotential

    GM, b = 1.3, 0.8
    Ls = numpy.array([0.05, 0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0])
    sq = numpy.sqrt(Ls**2 + 4.0 * GM * b)
    # fractions of the circular orbit's energy -2 GM^2 / (L + sq)^2
    Es = (
        -2.0
        * GM**2
        / (Ls + sq) ** 2
        * numpy.array([0.9, 0.95, 0.5, 0.8, 0.3, 0.99, 0.6, 0.4])
    )
    aAI = actionAngleSphericalInverse(
        pot=IsochronePotential(amp=GM, b=b), Es=Es, Ls=Ls, mm_npt=64, progressbar=False
    )
    jr = numpy.array([float(as_numpy(aAI.Jr(E, L))) for E, L in zip(Es, Ls)])
    OmR, Omphi, Omz = (as_numpy(o) for o in aAI.Freqs(jr, 0.7 * Ls, 0.3 * Ls))
    OmR_true = (-2.0 * Es) ** 1.5 / GM
    Ompsi_true = 0.5 * OmR_true * (1.0 + Ls / sq)
    assert numpy.all(
        numpy.fabs(jr / (GM / numpy.sqrt(-2.0 * Es) - 0.5 * (Ls + sq)) - 1.0) < 1e-13
    ), "The radial actions of isochrone tori are not the closed form's"
    for Om, Om_true, what in (
        (OmR, OmR_true, "radial"),
        (Omphi, Ompsi_true, "azimuthal"),
        (Omz, Ompsi_true, "vertical"),
    ):
        assert numpy.all(numpy.fabs(Om / Om_true - 1.0) < 5e-12), (
            "The %s frequencies of isochrone tori are not the closed form's: %g"
            % (what, numpy.amax(numpy.fabs(Om / Om_true - 1.0)))
        )
    return None


def test_actionAngleStaeckelInverse_Lzspacing():
    # The Staeckel family takes the same angular-momentum spacing option as the
    # spherical one: a family spaced in the logarithm (the default), in the
    # square root of L_z or in L_z itself builds and evaluates, the family
    # built without Lzspacing= is the 'log' one, and an unknown spacing raises
    from galpy.actionAngle import actionAngleStaeckelInverse

    kw = dict(
        pot=_staeckel_inverse_potential(),
        setup_interp=True,
        Rmin=0.7,
        Rmax=1.6,
        Rinf=3.5,
        nE=4,
        nLz=4,
        nI3=4,
        mm_npt=48,
    )
    out = {}
    for sp in ("log", "sqrt", "linear"):
        aAS = actionAngleStaeckelInverse(Lzspacing=sp, **kw)
        out[sp] = numpy.array(_spherical_inverse_npt(aAS.Freqs(0.02, 0.9, 0.02)))
        assert numpy.all(numpy.isfinite(out[sp])), (
            f"A Staeckel family with Lzspacing={sp!r} does not evaluate"
        )
    default = actionAngleStaeckelInverse(**kw).Freqs(0.02, 0.9, 0.02)
    # (to round-off under a forced backend: jax's CPU FFT is not bitwise
    # reproducible from one build to the next)
    tol = 1e-13 if is_backend_array(default[0]) else 0.0
    default = numpy.array(_spherical_inverse_npt(default))
    assert numpy.all(numpy.fabs(default - out["log"]) <= tol), (
        "The default spacing of the angular-momentum grid is not 'log'"
    )
    assert numpy.any(default != out["linear"]), (
        "The default spacing of the angular-momentum grid is still 'linear'"
    )
    with pytest.raises(ValueError):
        actionAngleStaeckelInverse(Lzspacing="cubic", **kw)
    return None


def _staeckel_inverse_node_actions(aAI, n):
    # (J_R, L_z, J_z) of the nodes of a family's n^3 grid, as (n, n, n, 3)
    # along (E, I_3, L_z): the set-up's own actions, which JR and Jz return
    # at the nodes to round-off (1024 calls of those take minutes under an
    # eager backend)
    A = as_numpy(aAI._A_tab)
    return numpy.stack(
        [A[..., 0], as_numpy(aAI._Lzs).reshape(n, n, n), A[..., 1]], axis=-1
    )


def _staeckel_inverse_between(J, axes=(0, 1, 2)):
    # the mean actions of the nodes that are neighbours along the given axes
    # of the grid: along all three, a torus inside each cell
    for axis in axes:
        m = J.shape[axis]
        J = 0.5 * (J.take(range(m - 1), axis) + J.take(range(1, m), axis))
    return J


def test_actionAngleStaeckelInverse_adaptive_harmonics(staeckel_inverse_interp):
    # Every torus of an interpolated family keeps the coefficients of each of
    # its two maps down to the map's round-off floor, and every point is
    # evaluated with the harmonics of its cell: in each map the largest
    # number that the eight tori at the cell's corners keep, rounded up to
    # the harmonics 16, 32, ... or mm_npt, the points of a call being grouped
    # by the two numbers. On the small grid, whose cells read (16, 8), (16,
    # 16), (16, 24), (32, 16) and (32, 24) coefficients of the (u, v) maps:
    # (a) an array of points spread over all cells agrees with the same
    # points in another order, as a 2D array, in a call of few points (whose
    # groups are evaluated together) and one at a time (with all harmonics);
    # (b) a torus inside a cell of each group round-trips through the
    # forward transformation within the tolerances of the between-node test;
    # (c) the grid's own tori are as accurate as with all harmonics; (d) the
    # map is continuous across faces between cells of different groups
    from galpy.potential import evaluatePotentials

    aAI = staeckel_inverse_interp
    nodes = _staeckel_inverse_node_actions(aAI, 8)
    cells = _staeckel_inverse_between(nodes)
    # (a) four points on a torus inside each of the 7^3 cells: 184 to 516
    # points per group; they differ by 3e-13 at most from one at a time
    rng = numpy.random.default_rng(7)
    pts = [numpy.repeat(q, 4) for q in cells.reshape(-1, 3).T]
    pts += list(rng.uniform(0.0, 2.0 * numpy.pi, (3, 1372)))
    out = numpy.array(aAI.xvFreqs(*pts))
    perm, sub = rng.permutation(1372), slice(None, None, 29)
    shuffled = numpy.empty((9, 1372))
    shuffled[:, perm] = aAI.xvFreqs(*(t[perm] for t in pts))
    grid = numpy.array(aAI.xvFreqs(*(t.reshape(343, 4) for t in pts)))
    assert grid.shape == (9, 343, 4), "A 2D array of points does not keep its shape"
    few = numpy.array(aAI.xvFreqs(*(t[sub] for t in pts)))
    one = numpy.array(
        [
            [numpy.ravel(o)[0] for o in aAI.xvFreqs(*(t[k] for t in pts))]
            for k in range(0, 1372, 29)
        ]
    ).T
    for ref, other, what in (
        (out, shuffled, "in another order"),
        (out, grid.reshape(9, 1372), "as a 2D array"),
        (out[:, sub], few, "in a call of few points"),
        (out[:, sub], one, "one at a time"),
    ):
        assert numpy.all(numpy.fabs(ref - other) < 1e-11 * (1.0 + numpy.fabs(ref))), (
            "Points grouped by their numbers of harmonics differ from the same "
            "points evaluated %s" % what
        )
    assert numpy.all(numpy.array(aAI.Freqs(*pts[:3])) == out[6:]), (
        "Freqs and xvFreqs disagree for arrays of tori"
    )
    # (b) the round trip in a cell of each of the five groups, and in a cell
    # on the edge where the circular face meets the shell face, at the
    # grid's interpolation error there (1.8e-4 in the angles; 2.4e-4 if the
    # nodes on that edge, all of whose tables vanish, kept no harmonics)
    angler = numpy.linspace(0.05, 6.2, 21)
    anglephi = (0.3 + 1.7 * angler) % (2.0 * numpy.pi)
    anglez = (0.7 + 2.3 * angler) % (2.0 * numpy.pi)
    for cell in ((1, 0, 3), (2, 3, 3), (3, 4, 4), (5, 1, 5), (5, 2, 5), (0, 6, 3)):
        jr, Lz, jz = cells[cell]
        errs = _staeckel_inverse_roundtrip(aAI, jr, Lz, jz, angler, anglephi, anglez)
        tols = (1e-5, 2.1e-4 if cell[0] == 0 else 1e-4, 1e-4, 1e-5)
        assert all(err < tol for err, tol in zip(errs, tols)), (
            "The round trip (actions, angles, frequencies, energy) in the cell %s "
            "is not within the grid's interpolation error: %s" % (cell, errs)
        )
    # (c) what a torus of the grid drops is round-off: along every torus of
    # the grid with both librations, the energy of the returned points is
    # the torus's, relative to the squared speed, to 5e-14 in the median
    # (2e-13 when all harmonics are read) and to 4e-7 at most, on the tori
    # whose v map this grid's mm_npt does not resolve, which keep all of it
    jr, Lz, jz = nodes[1:, 1:-1].reshape(-1, 3).T
    R, vR, vT, z, vz, _ = (
        as_numpy(a)
        for a in aAI(
            jr, Lz, jz, *numpy.random.default_rng(4).uniform(0.0, 6.28, (3, 12, 336))
        )
    )
    v2 = vR**2 + vT**2 + vz**2
    dE = numpy.amax(
        numpy.fabs(
            as_numpy(evaluatePotentials(_staeckel_inverse_potential(), R, z))
            + 0.5 * v2
            - aAI._Es.reshape(8, 8, 8)[1:, 1:-1].ravel()
        )
        / v2,
        axis=0,
    )
    assert numpy.median(dE) < 1e-12 and numpy.all(dE < 1e-6), (
        "The grid's own tori do not conserve the energy as with all harmonics: "
        "median %g, maximum %g" % (numpy.median(dE), numpy.amax(dE))
    )
    # (d) points at a distance d either side of a face between cells of
    # different groups differ in proportion to d (by 2.2 d at most on this
    # grid): across three nodes of L_z at the mean actions of the face's
    # four nodes, where the number of the v map, of the u map and of both
    # changes
    faces = _staeckel_inverse_between(nodes, axes=(0, 1))
    for face in ((2, 0, 5), (3, 0, 2), (5, 2, 6)):
        jr, Lz, jz = faces[face]
        for d in (1e-6, 1e-9):
            lo, hi = (numpy.ravel(aAI(jr, Lz + s, jz, 1.3, 0.7, 2.1)) for s in (-d, d))
            assert numpy.amax(numpy.fabs(hi - lo)) < 10.0 * d, (
                "The map jumps across the face %s between cells of different "
                "groups: %g at distance %g" % (face, numpy.amax(numpy.fabs(hi - lo)), d)
            )
    return None


def test_actionAngleStaeckelInverse_harmonics_beyond_need():
    # A torus's coefficients beyond those that it keeps are zero, in both
    # maps and in their slopes, which are solved for in the kept harmonics
    # alone: a grid whose tori all keep fewer coefficients than mm_npt=72
    # allows (18 of 72 in the u map and 31 of 36 in the v map at most), set
    # up with mm_npt=96 instead, is the same family, with the same points at
    # a torus inside each cell to the round-off of the two set-ups (1e-11)
    from galpy.actionAngle import actionAngleStaeckelInverse

    fams = [
        actionAngleStaeckelInverse(
            pot=_staeckel_inverse_potential(),
            setup_interp=True,
            Rmin=0.9,
            Rmax=1.2,
            Rinf=2.2,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=mm_npt,
            progressbar=False,
        )
        for mm_npt in (72, 96)
    ]
    cells = _staeckel_inverse_between(_staeckel_inverse_node_actions(fams[0], 4))
    pts = [q[:, None] for q in cells.reshape(-1, 3).T]
    pts += list(numpy.random.default_rng(7).uniform(0.0, 2.0 * numpy.pi, (3, 27, 8)))
    out, more = (numpy.array(aAI.xvFreqs(*pts)) for aAI in fams)
    assert numpy.all(numpy.fabs(out - more) < 1e-10 * (1.0 + numpy.fabs(out))), (
        "The family depends on the number of harmonics beyond those its tori keep: %g"
        % (numpy.amax(numpy.fabs(out - more) / (1.0 + numpy.fabs(out))))
    )


def test_actionAngleInverse_families_pickle():
    # Interpolated families of the inverse transformations pickle (on v2 their
    # interpolants hold their array namespace, a module, which pickle stores
    # by name) and evaluate identically after a round trip
    import pickle

    from galpy.actionAngle import (
        actionAngleSphericalInverse,
        actionAngleStaeckelInverse,
        actionAngleVerticalInverse,
    )
    from galpy.potential import (
        HernquistPotential,
        KuzminKutuzovStaeckelPotential,
        MWPotential2014,
        evaluatelinearPotentials,
        toVerticalPotential,
    )

    hp = HernquistPotential(amp=2.0, a=1.0)
    vp = toVerticalPotential(MWPotential2014, 1.0)
    E0 = float(evaluatelinearPotentials(vp, 0.0))
    cases = (
        (
            actionAngleSphericalInverse(
                pot=hp, setup_interp=True, nE=6, nL=6, progressbar=False
            ),
            ([0.05, 0.1], [0.5, -0.4], [0.2, 0.3], [0.3, 2.0], [1.0, 4.0], [2.0, 5.5]),
        ),
        (
            actionAngleStaeckelInverse(
                pot=KuzminKutuzovStaeckelPotential(normalize=1.0, ac=4.0, Delta=1.4),
                setup_interp=True,
                Rmin=0.5,
                Rmax=1.5,
                Rinf=3.5,
                nE=4,
                nLz=4,
                nI3=4,
                progressbar=False,
            ),
            (
                [0.03, 0.05],
                [1.0, 1.1],
                [0.03, 0.04],
                [0.3, 2.0],
                [1.0, 4.0],
                [2.0, 5.5],
            ),
        ),
        (
            actionAngleVerticalInverse(
                pot=vp, Es=numpy.linspace(E0 + 0.01, E0 + 0.3, 9), setup_interp=True
            ),
            ([0.01, 0.02], [0.3, 2.0]),
        ),
    )
    for aA, pts in cases:
        pts = [numpy.array(p) for p in pts]
        aB = pickle.loads(pickle.dumps(aA))
        assert numpy.all(
            numpy.array([as_numpy(o) for o in aA.xvFreqs(*pts)])
            == numpy.array([as_numpy(o) for o in aB.xvFreqs(*pts)])
        ), f"A pickled {type(aA).__name__} family does not evaluate identically"
    return None


def test_actionAngleStaeckelInverse_split_tori():
    # In the Staeckel approximation of a cusped potential the foci attract
    # (V(0) > U(0)), and near the axis the momentum function of the v
    # libration can be positive again beyond the turning point of the
    # libration about the midplane, on a torus of its own with the same
    # integrals: the torus of given integrals is the one about the midplane
    # (its actions continue those of the tori below it in I_3, up to the
    # shell orbit), and where no torus continues those about the midplane
    # an interpolation grid fails with a message that says so
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import NFWPotential, TriaxialNFWPotential

    halo = TriaxialNFWPotential(normalize=1.0, a=1.0, b=1.0, c=0.8)
    E, Lz = -4.952347997451089, 0.0045331251128010955
    Ipl, Ish = 4.952454913285056, 5.017135158552722
    jzs = []
    for y in (0.999, 1.0):
        I3 = Ipl + y * (Ish - Ipl)
        aAI = actionAngleStaeckelInverse(
            pot=halo, delta=0.31, Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=32
        )
        jzs.append(aAI.Jz(E, Lz, I3))
        assert numpy.all(
            numpy.isfinite(aAI(aAI.JR(E, Lz, I3), Lz, jzs[-1], 0.5, 1.0, 2.0))
        ), "A torus about the midplane of a cusped halo does not evaluate"
    assert numpy.fabs(jzs[1] / jzs[0] - 1.0) < 1e-2, (
        "The shell orbit of a cusped halo is not the limit of the tori about "
        "the midplane"
    )
    with pytest.raises(ValueError, match="does not fall monotonically"):
        actionAngleStaeckelInverse(
            pot=NFWPotential(normalize=1.0, a=1.0),
            delta=0.31,
            setup_interp=True,
            Rmin=0.05,
            Rmax=3.0,
            Rinf=8.0,
            nE=6,
            nLz=6,
            nI3=6,
            mm_npt=32,
            progressbar=False,
        )
    # likewise the u libration: at small |L_z| in MWPotential2014, whose
    # bulge is cusped, its momentum function has a second maximum close to
    # the centre at some energies, and a grid that reaches those tori fails
    from galpy.potential import MWPotential2014

    with pytest.raises(ValueError, match="more than one maximum"):
        actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=0.3,
            setup_interp=True,
            Rmin=0.04,
            Rmax=2.0,
            Rinf=5.0,
            nE=16,
            nLz=4,
            nI3=4,
            mm_npt=32,
            progressbar=False,
        )
    return None


def test_actionAngleStaeckelInverse_polar_torus():
    # A torus that passes within a degree of the axis: its v turning point
    # lies beyond the scan that brackets the first root below the midplane,
    # from whose end the bracket is extended toward the axis; its actions
    # are the forward transformation's
    from galpy.actionAngle import actionAngleStaeckelInverse

    ic = [0.05, 0.6, 0.4, 2.5, 0.0, 0.0]
    E, Lz, I3 = _staeckel_inverse_labels(ic)
    aAI = actionAngleStaeckelInverse(
        pot=_staeckel_inverse_potential(), Es=[E], Lzs=[Lz], I3s=[I3], mm_npt=64
    )
    jr, jz = aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)
    jrf, _, jzf = _staeckel_inverse_forward()(*ic[:5])
    assert (
        numpy.fabs(jr / jrf[0] - 1.0) < 1e-8 and numpy.fabs(jz / jzf[0] - 1.0) < 1e-8
    ), "The actions of a nearly polar torus are not the forward transformation's"
    R, _, _, z, _, _ = (
        as_numpy(t) for t in aAI(jr, Lz, jz, 1.0, 1.0, numpy.linspace(0.0, 6.28, 201))
    )
    assert numpy.all(numpy.isfinite(R)) and numpy.amin(R / numpy.hypot(R, z)) < 0.1, (
        "A nearly polar torus does not come close to the axis"
    )
    return None


def test_actionAngleStaeckelInverse_in_domain(
    staeckel_inverse_interp, staeckel_inverse_explicit
):
    # in_domain tells which tori an instance evaluates: for a family those
    # with |L_z| inside the grid and (J_R, J_z) covered there, the circular
    # orbits and the tori on the degenerate faces included; for explicit
    # tori those tori. An array evaluation with tori outside raises once,
    # counting them all
    aAI = staeckel_inverse_interp
    pot = _staeckel_inverse_potential()
    rng = numpy.random.default_rng(4)
    n = 60
    jr, jz = 10.0 ** rng.uniform(-3, 0.3, (2, n))
    Lz = rng.uniform(0.6, 2.2, n) * rng.choice([-1.0, 1.0], n)
    inside = aAI.in_domain(jr, Lz, jz)
    assert inside.shape == (n,) and 5 < numpy.sum(inside) < n - 5, (
        "The test tori are not on both sides of the family's edge"
    )
    out = numpy.array(aAI(jr[inside], Lz[inside], jz[inside], 0.3, 1.0, 2.0))
    assert numpy.all(numpy.isfinite(out)), (
        "A torus that in_domain places inside the family does not evaluate"
    )
    for ii in numpy.flatnonzero(~inside):
        with pytest.raises(ValueError, match="outside the interpol|not be matched"):
            aAI(jr[ii], Lz[ii], jz[ii], 0.3, 1.0, 2.0)
    assert aAI.in_domain(jr[:, None], Lz[0], jz[None, :4]).shape == (n, 4), (
        "in_domain does not broadcast its actions"
    )
    # the grid spans the circular angular momenta of Rmin = 0.7 and Rmax = 1.6
    offgrid = (numpy.fabs(Lz) < 0.7 * float(as_numpy(pot.vcirc(0.7)))) | (
        numpy.fabs(Lz) > 1.6 * float(as_numpy(pot.vcirc(1.6)))
    )
    msg = (
        f"{numpy.sum(~inside)} of the {n} tori lie outside the interpolated "
        f"family: {numpy.sum(offgrid)} with .* and "
        f"{numpy.sum(~inside & ~offgrid)} whose .* could not be matched"
    )
    with pytest.raises(ValueError, match=msg):
        aAI(jr, Lz, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match=msg):
        aAI.Freqs(jr, Lz, jz)
    # single tori: inside, the circular orbit, the degenerate faces, and
    # what an evaluation rejects
    for actions, expected in (
        ((0.05, 0.9, 0.05), True),
        ((0.0, 0.9, 0.0), True),
        ((0.05, -0.9, 0.0), True),
        ((0.0, 0.9, 0.05), True),
        ((0.0, 5.0, 0.0), False),
        ((50.0, 0.9, 0.0), False),
        ((0.05, 5.0, 0.05), False),
        ((-0.05, 0.9, 0.05), False),
        ((0.05, 0.0, 0.05), False),
    ):
        assert aAI.in_domain(*actions) is expected, (
            f"in_domain{actions} of a family is not {expected}"
        )
    # explicit tori
    aAE = staeckel_inverse_explicit
    labels = [
        _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS[k])
        for k in ("benign", "eccentric")
    ]
    jrs = numpy.array([aAE.JR(*l) for l in labels])
    jzs = numpy.array([aAE.Jz(*l) for l in labels])
    Lzs = numpy.array([l[1] for l in labels])
    assert numpy.all(aAE.in_domain(jrs, Lzs, jzs)), (
        "in_domain does not hold the explicit tori of an instance"
    )
    assert not numpy.any(aAE.in_domain(1.1 * jrs, Lzs, jzs)), (
        "in_domain holds tori that an instance of explicit tori does not have"
    )
    return None


def test_actionAngleStaeckelInverse_label_inversion(staeckel_inverse_interp):
    # The actions of a family at integrals inside its grid are inside it by
    # construction, also where an action rises steeply out of a face, as J_z
    # does out of the planar face for eccentric tori: the label inversion's
    # Newton steps are halved until they reduce the residual, so that it
    # finds every such torus, one at a time and in an array
    from galpy.potential import evaluatePotentials

    aAI = staeckel_inverse_interp
    pot = _staeckel_inverse_potential()
    actions = []
    for Lz in 0.7 * pot.vcirc(0.7) * numpy.array([1.01, 1.3, 1.8]):
        for R0 in (2.2, 2.8, 3.4):
            # the energy of the planar orbit without angular momentum that
            # reaches R0, and third integrals close above the planar orbit's
            E = evaluatePotentials(pot, R0, 0.0)
            Ipl = Lz**2 / 2.0 / _STAECKEL_INVERSE_DELTA**2 - E
            for dI in (1e-4, 1e-3, 1e-2, 5e-2):
                jr, jz = aAI.JR(E, Lz, Ipl + dI), aAI.Jz(E, Lz, Ipl + dI)
                actions.append((jr, Lz, jz))
                assert aAI.in_domain(jr, Lz, jz), (
                    "The actions of a family at integrals inside its grid are "
                    "not inside its domain"
                )
                assert numpy.all(numpy.isfinite(aAI(jr, Lz, jz, 0.1, 0.2, 0.3)))
    jr, Lz, jz = numpy.array(actions).T
    assert numpy.all(aAI.in_domain(jr, Lz, jz)) and numpy.all(
        numpy.isfinite(aAI(jr, Lz, jz, 0.1, 0.2, 0.3))
    ), "An array of tori inside a family's grid is not located"
    return None


_STAECKEL_INVERSE_AXIS = dict(
    setup_interp=True, Rmin=0.1, Rmax=1.5, Rinf=4.0, nE=12, nLz=5, nI3=9, mm_npt=96
)


def _staeckel_inverse_axis_potential():
    # a Kuzmin-Kutuzov potential whose focal length is comparable to its
    # tori, so that tori of small |L_z| come in both kinds: stopped off the
    # axis by the centrifugal barrier in u (those that would pass between
    # the foci without angular momentum) or in v (those that would loop
    # around them)
    from galpy.potential import KuzminKutuzovStaeckelPotential

    return KuzminKutuzovStaeckelPotential(normalize=1.0, ac=2.0, Delta=0.5)


@pytest.fixture(scope="module")
def staeckel_inverse_axis():
    # a small interpolation grid that reaches tori close to the axis
    from galpy.actionAngle import actionAngleStaeckelInverse

    return actionAngleStaeckelInverse(
        pot=_staeckel_inverse_axis_potential(),
        progressbar=False,
        **_STAECKEL_INVERSE_AXIS,
    )


def _staeckel_inverse_axis_errors(aAI, Lz):
    # the spread of the energy over tori of small |L_z| relative to their
    # mean kinetic energy, and the error of their actions (the forward
    # transformation's at the returned points) relative to their total
    # action, for tori of decreasing |L_z| / L of both kinds
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import evaluatePotentials

    pot = _staeckel_inverse_axis_potential()
    aAF = actionAngleStaeckel(pot=pot, delta=0.5, c=True)
    th = numpy.linspace(0.1, 2.0 * numpy.pi + 0.1, 12, endpoint=False)
    thR, thz = (t.ravel() for t in numpy.meshgrid(th, th + 0.05, indexing="ij"))
    Lz = float(as_numpy(Lz))
    spread, dJ = [], []
    for L in (0.15, 0.3, 0.6, 1.0):
        for jr in (0.05, 0.3):
            jz = L - Lz
            R, vR, vT, z, vz, _ = (as_numpy(t) for t in aAI(jr, Lz, jz, thR, 0.0, thz))
            assert numpy.all(R > 0.0), "A torus of small |L_z| reaches R <= 0"
            K = 0.5 * (vR**2 + vT**2 + vz**2)
            Phi = as_numpy(evaluatePotentials(pot, R, z))
            spread.append(numpy.std(K + Phi) / numpy.mean(K))
            jrf, _, jzf = (as_numpy(t) for t in aAF(R, vR, vT, z, vz))
            dJ.append(numpy.amax(numpy.hypot(jrf - jr, jzf - jz)) / (jr + L))
    return numpy.array(spread), numpy.array(dJ)


def test_actionAngleStaeckelInverse_near_axis(staeckel_inverse_axis):
    # At small |L_z| the centrifugal barrier stops either the u or the v
    # libration just off the axis, on either side of the third integral of
    # the L_z = 0 orbits through the foci, around which the turning points
    # and every other table of a torus change over a range of the third
    # integral proportional to |L_z|. A grid spaced in the third integral
    # itself does not resolve that; the grid's third-integral variable
    # blends in the logarithm of the turning points' distances to the axis
    # (axis_weight), in which the tables are smooth, and the half-widths
    # are stored in variables that keep the inner turning points off the
    # axis: the tori of small |L_z| / L of an exact Staeckel potential are
    # then accurate, as they are not with a weight of zero
    import pickle

    from galpy.actionAngle import actionAngleStaeckelInverse

    aAI = staeckel_inverse_axis
    pot = _staeckel_inverse_axis_potential()
    Lz = 1.3 * 0.1 * pot.vcirc(0.1)
    spread, dJ = _staeckel_inverse_axis_errors(aAI, Lz)
    assert numpy.amax(spread) < 1e-2, (
        "The energy varies over a family's tori of small |L_z| in a Staeckel potential"
    )
    assert numpy.amax(dJ) < 1.5e-2, (
        "A family's tori of small |L_z| do not have the requested actions"
    )
    aA0 = actionAngleStaeckelInverse(
        pot=pot, progressbar=False, axis_weight=0.0, **_STAECKEL_INVERSE_AXIS
    )
    spread0, dJ0 = _staeckel_inverse_axis_errors(aA0, Lz)
    assert numpy.amax(spread0) > 10.0 * numpy.amax(spread), (
        "The third-integral variable's weight of the axis does not make the "
        "tori of small |L_z| more accurate"
    )
    assert numpy.amax(dJ0) > 10.0 * numpy.amax(dJ), (
        "The third-integral variable's weight of the axis does not make the "
        "actions of the tori of small |L_z| more accurate"
    )
    # the family's actions at given integrals, whose third integral is
    # located in the same variable: those of the torus through a point
    jr, jz = 0.3, 0.6 - Lz
    R, vR, vT, z, vz, phi = aAI(jr, Lz, jz, 1.0, 2.0, 3.0)
    E, _, I3 = _staeckel_inverse_labels(
        [R[0], vR[0], vT[0], z[0], vz[0], phi[0]], pot=pot, delta=0.5
    )
    for fam in (aAI, aA0):
        assert numpy.fabs(fam.JR(E, Lz, I3) - jr) < 2e-2 * (jr + jz), (
            "JR at the integrals of a torus of small |L_z| is not its radial action"
        )
        assert numpy.fabs(fam.Jz(E, Lz, I3) - jz) < 2e-2 * (jr + jz), (
            "Jz at the integrals of a torus of small |L_z| is not its vertical action"
        )
    # the family pickles, with its grid's third-integral variable, and
    # evaluates identically afterwards
    again = pickle.loads(pickle.dumps(aAI))
    pts = (jr, Lz, jz, 1.0, 2.0, 3.0)
    assert numpy.all(numpy.array(aAI(*pts)) == numpy.array(again(*pts))), (
        "A pickled family does not evaluate identically"
    )
    assert again.JR(E, Lz, I3) == aAI.JR(E, Lz, I3), (
        "A pickled family does not locate a third integral identically"
    )
    return None


def test_actionAngleStaeckelInverse_pickle_format(
    staeckel_inverse_interp, staeckel_inverse_explicit
):
    # An interpolated family is pickled with the format of its tables and
    # evaluates identically after a round trip; a pickled family without the
    # format (as before the formats were numbered) or with an older one,
    # whose tables would be decoded wrongly, is refused with a message that
    # says to rebuild it; explicit tori carry no family tables
    import pickle

    from galpy.actionAngle import actionAngleStaeckelInverse

    aAI = staeckel_inverse_interp
    pts = (0.05, 0.9, 0.05, 0.3, 1.0, 2.0)
    again = pickle.loads(pickle.dumps(aAI))
    assert numpy.all(numpy.array(aAI(*pts)) == numpy.array(again(*pts))), (
        "A pickled family does not evaluate identically"
    )
    for change in (
        lambda st: st.pop("_family_format"),
        lambda st: st.update(_family_format=1),
    ):
        state = dict(aAI.__dict__)
        change(state)
        with pytest.raises(RuntimeError, match="rebuild the family"):
            actionAngleStaeckelInverse.__new__(actionAngleStaeckelInverse).__setstate__(
                state
            )
    aAE = pickle.loads(pickle.dumps(staeckel_inverse_explicit))
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    assert aAE.JR(E, Lz, I3) == staeckel_inverse_explicit.JR(E, Lz, I3), (
        "Pickled explicit tori do not return their actions identically"
    )
    return None


def test_actionAngleStaeckelInverse_from_actions():
    # Explicit tori are set up from their actions with jrs=, Lzs=, jzs=: the
    # integrals are solved for, on the faces (a shell, a planar and a
    # circular orbit) as elsewhere, and in the Staeckel model that a rule
    # for the reference curve gives the torus; the tori have the requested
    # actions (the forward transformation's at their points in a Staeckel
    # potential), and actions that no torus has are refused
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    pot = _staeckel_inverse_potential()
    aAF = _staeckel_inverse_forward()
    J = numpy.array(
        [
            (0.05, 0.9, 0.05),
            (0.3, -0.6, 0.2),
            (0.0, 0.9, 0.1),
            (0.2, 1.1, 0.0),
            (0.0, 0.8, 0.0),
            (1.5, 0.3, 0.02),
        ]
    )
    aAI = actionAngleStaeckelInverse(pot=pot, jrs=J[:, 0], Lzs=J[:, 1], jzs=J[:, 2])
    assert numpy.all(aAI.in_domain(*J.T)), (
        "Tori set up from their actions are not the instance's tori"
    )
    for jr, Lz, jz in J:
        R, vR, vT, z, vz, _ = (
            as_numpy(o) for o in aAI(jr, Lz, jz, [0.3, 2.0, 4.5], 1.0, [1.0, 3.0, 5.5])
        )
        jrf, Lzf, jzf = (as_numpy(o) for o in aAF(R, vR, vT, z, vz))
        assert (
            numpy.amax(numpy.fabs(jrf - jr)) < 1e-8
            and numpy.amax(numpy.fabs(jzf - jz)) < 1e-8
            and numpy.amax(numpy.fabs(Lzf - Lz)) < 1e-10
        ), "A torus set up from its actions does not have them"
    # a general potential, with a fixed reference curve and with one that
    # follows the torus: the tori are those of the requested actions
    for kw in ({}, {"u0": "torus"}):
        aAM = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=0.45,
            jrs=[0.05, 0.4],
            Lzs=[1.0, 0.3],
            jzs=[0.02, 0.3],
            **kw,
        )
        assert numpy.all(aAM.in_domain([0.05, 0.4], [1.0, 0.3], [0.02, 0.3]))
        assert numpy.all(
            numpy.isfinite(
                [
                    as_numpy(o)
                    for o in aAM([0.05, 0.4], [1.0, 0.3], [0.02, 0.3], 1.0, 2.0, 3.0)
                ]
            )
        )
    # guards
    with pytest.raises(ValueError, match="both jrs= and jzs="):
        actionAngleStaeckelInverse(pot=pot, jrs=[0.1], Lzs=[0.9])
    with pytest.raises(ValueError, match="same length"):
        actionAngleStaeckelInverse(pot=pot, jrs=[0.1, 0.2], Lzs=[0.9], jzs=[0.1])
    with pytest.raises(ValueError, match="non-negative"):
        actionAngleStaeckelInverse(pot=pot, jrs=[-0.1], Lzs=[0.9], jzs=[0.1])
    # a torus that reaches beyond every radius at which a bound libration
    # is looked for
    with pytest.raises(ValueError, match="No torus with the actions"):
        actionAngleStaeckelInverse(pot=pot, jrs=[1e9], Lzs=[0.9], jzs=[0.1])
    # actions that are tiny but not zero are those of the face that they
    # are next to: the shell, the planar and the circular orbit
    tiny = actionAngleStaeckelInverse(
        pot=pot, jrs=[1e-13, 0.2, 1e-13], Lzs=[0.9, 1.1, 0.8], jzs=[0.1, 1e-13, 1e-13]
    )
    for (jr, Lz, jz), face in zip(
        ((1e-13, 0.9, 0.1), (0.2, 1.1, 1e-13), (1e-13, 0.8, 1e-13)),
        ((0.0, 0.9, 0.1), (0.2, 1.1, 0.0), (0.0, 0.8, 0.0)),
    ):
        assert (
            numpy.amax(
                numpy.fabs(
                    numpy.array([as_numpy(o) for o in tiny(jr, Lz, jz, 1.0, 2.0, 3.0)])
                    - numpy.array([as_numpy(o) for o in aAI(*face, 1.0, 2.0, 3.0)])
                )
            )
            < 1e-9
        ), "A torus of a tiny action is not the torus of the face next to it"
    # in the Staeckel approximation of a cusped potential the u libration
    # splits in two at small |L_z|: the torus is the one around the shell
    # orbit, whose actions jump where it merges with the inner one, and
    # actions inside the jump are no torus's
    kw = dict(pot=MWPotential2014, delta=0.3, u0="torus", mm_npt=64)
    with pytest.raises(ValueError, match="No torus with the actions"):
        actionAngleStaeckelInverse(jrs=[0.11032], Lzs=[0.00614], jzs=[0.24778], **kw)
    outer = actionAngleStaeckelInverse(jrs=[0.1068], Lzs=[0.00614], jzs=[0.36], **kw)
    R, _, _, z, _, _ = (
        as_numpy(o)
        for o in outer(0.1068, 0.00614, 0.36, numpy.linspace(0.0, 6.0, 13), 1.0, 2.0)
    )
    assert numpy.all(numpy.isfinite(R)) and numpy.amin(numpy.hypot(R, z)) > 0.09, (
        "The torus around the shell orbit of a split u libration reaches inside "
        "the barrier"
    )
    # and likewise when the torus is set up from its integrals
    E, Lz, I3 = -1.36267, 0.00614, 3.37947
    ints = actionAngleStaeckelInverse(Es=[E], Lzs=[Lz], I3s=[I3], **kw)
    R, _, _, z, _, _ = (
        as_numpy(o)
        for o in ints(
            ints.JR(E, Lz, I3),
            Lz,
            ints.Jz(E, Lz, I3),
            numpy.linspace(0.0, 6.0, 13),
            1.0,
            2.0,
        )
    )
    assert numpy.all(numpy.isfinite(R)) and numpy.amin(numpy.hypot(R, z)) > 0.09, (
        "The torus around the shell orbit of a split u libration, set up from "
        "its integrals, reaches inside the barrier"
    )
    return None


def test_actionAngleStaeckelInverse_setup_not_kept():
    # What the set-up computes along the way (the samples of every torus on
    # mm_nta anomalies and their lifts onto the auxiliary) is not kept: an
    # instance holds its tables (and a family the interpolants built from
    # them), whose size is set by mm_npt, so that a pickled family or set of
    # explicit tori has the same size whatever mm_nta is, of order ten kB
    # per torus, and evaluates identically
    import pickle

    from galpy.actionAngle import actionAngleStaeckelInverse

    pot = _staeckel_inverse_potential()
    E, Lz, I3 = _staeckel_inverse_labels(_STAECKEL_INVERSE_ICS["benign"])
    setups = (
        (dict(setup_interp=True, Rmin=0.8, Rmax=1.4, Rinf=3.0, nE=4, nLz=4, nI3=4), 64),
        (dict(Es=[E], Lzs=[Lz], I3s=[I3]), 1),
    )
    for kwargs, ntori in setups:
        sizes = []
        for nta in (384, 1536):
            aAI = actionAngleStaeckelInverse(pot=pot, mm_npt=48, mm_nta=nta, **kwargs)
            pickled = pickle.dumps(aAI)
            sizes.append(len(pickled))
            jr, jz = aAI.JR(E, Lz, I3), aAI.Jz(E, Lz, I3)
            pts = (jr, Lz, jz, 0.3, 1.0, 2.0)
            assert numpy.all(
                numpy.array([as_numpy(o) for o in aAI(*pts)])
                == numpy.array([as_numpy(o) for o in pickle.loads(pickled)(*pts)])
            ), "A pickled instance does not evaluate identically"
        assert numpy.fabs(sizes[1] / sizes[0] - 1.0) < 0.01, (
            "The size of a pickled actionAngleStaeckelInverse instance depends on "
            "mm_nta: samples of the set-up are kept"
        )
        assert sizes[1] < 2**16 + ntori * 2**14, (
            "A pickled actionAngleStaeckelInverse instance holds more than "
            "about ten kB per torus"
        )
    return None


def test_actionAngleStaeckelInverse_interp_shell_face_large_I3():
    # The grid's shell face is the shell orbits exactly, however large the
    # third integral (a focal length far below the tori's size makes it
    # large, and the grid's third-integral variable then rounds at the face):
    # a node placed a rounding below the face is a torus whose u libration
    # all but vanishes, and its tables are not finite. The family below was
    # refused for that reason; its shell tori have a single radius, up to the
    # square of the ratio of the focal length to that radius
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    aAI = actionAngleStaeckelInverse(
        pot=MWPotential2014,
        delta=0.006,
        u0="torus",
        setup_interp=True,
        Rmin=0.0031,
        Rmax=0.0062,
        Rinf=16.0,
        nE=4,
        nLz=4,
        nI3=4,
        mm_npt=64,
        progressbar=False,
    )
    angles = numpy.linspace(0.1, 6.0, 5)
    for jz in (0.1, 1.0):
        assert aAI.in_domain(0.0, 0.0015, jz), (
            "A shell torus of the family with a large third integral is not in its domain"
        )
        R, vR, vT, z, vz, phi = aAI(
            numpy.zeros(5),
            0.0015 * numpy.ones(5),
            jz * numpy.ones(5),
            angles,
            0.7 * angles,
            1.3 * angles,
        )
        r = numpy.sqrt(R**2.0 + z**2.0)
        assert numpy.all(numpy.isfinite([R, vR, vT, z, vz, phi])), (
            "A shell torus of a family with a large third integral is not finite"
        )
        assert numpy.all(numpy.fabs(r / numpy.mean(r) - 1.0) < 1e-3), (
            "A shell torus of a family with a focal length far below its size does not have a single radius"
        )
    return None


def test_actionAngleStaeckelInverse_from_actions_many():
    # The solve for the integrals of tori from their actions is vectorized
    # over the tori: many tori at once have the requested actions, on the
    # anomaly grid of the solve and on a finer one of their own, and
    # whatever the rule for the reference curve (in a Staeckel potential
    # every rule gives the same model, so the forward transformation checks
    # the built-in rules, which are vectorized, and a callable, which is
    # evaluated torus by torus); the tori without a solution are counted in
    # one error
    import pickle

    from galpy.actionAngle import actionAngleStaeckelInverse

    pot = _staeckel_inverse_potential()
    aAF = _staeckel_inverse_forward()
    rng = numpy.random.default_rng(4)
    n = 12
    jr, jz = rng.uniform(0.01, 0.5, n), rng.uniform(0.01, 0.3, n)
    Lz = rng.uniform(0.3, 1.2, n) * rng.choice([-1.0, 1.0], n)
    for kw in (
        {},
        {"mm_npt": 384},
        {"u0": "torus", "mm_npt": 48},
        {"u0": "planar"},
        {"u0": "shell"},
        {"u0": lambda E, Lz, I3: 1.0 + 0.1 * E + 0.01 * I3},
    ):
        aAI = actionAngleStaeckelInverse(pot=pot, jrs=jr, Lzs=Lz, jzs=jz, **kw)
        R, vR, vT, z, vz, _ = (as_numpy(o) for o in aAI(jr, Lz, jz, 1.0, 2.0, 3.0))
        jrf, Lzf, jzf = (as_numpy(o) for o in aAF(R, vR, vT, z, vz))
        assert (
            numpy.amax(numpy.fabs(jrf - jr)) < 1e-8
            and numpy.amax(numpy.fabs(jzf - jz)) < 1e-8
            and numpy.amax(numpy.fabs(Lzf - Lz)) < 1e-10
        ), f"Tori set up together from their actions do not have them ({kw})"
        # what the solve tabulates is not kept: a pickle holds the tori's
        # tables, a few kB per torus
        assert kw or len(pickle.dumps(aAI)) < 2**16 + n * 2**13
    with pytest.raises(ValueError, match="of 2 of the 3 tori"):
        actionAngleStaeckelInverse(
            pot=pot, jrs=[1e9, 0.1, 2e9], Lzs=[0.9, 0.9, 0.9], jzs=[0.1, 0.1, 0.1]
        )
    with pytest.raises(ValueError, match="of 1 of the 1 tori"):
        actionAngleStaeckelInverse(pot=pot, jrs=[1e9], Lzs=[0.9], jzs=[0.1], mm_npt=48)
    # the circular orbit, with a rule for the reference curve
    circ = actionAngleStaeckelInverse(
        pot=pot, u0="torus", jrs=[0.0], Lzs=[0.8], jzs=[0.0]
    )
    assert circ.in_domain(0.0, 0.8, 0.0)
    return None


def test_actionAngleStaeckelInverse_from_actions_sheets():
    # In the Staeckel approximation of a cusped potential the momentum
    # function of the u libration has two maxima at small |L_z|, and the
    # tori form three sheets: one libration across both wells below the
    # level of the barrier between them, and above it the libration about
    # the outer maximum and that about the inner one, with the actions
    # jumping across the barrier's level. A torus is found on whichever
    # sheet has its actions (about the outer maximum also where the inner
    # one is the higher of the two), and the tori whose actions lie in a
    # jump are counted in one error
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.3, mm_npt=64)
    th = numpy.linspace(0.0, 2.0 * numpy.pi, 12, endpoint=False)
    thR, thz = (t.ravel() for t in numpy.meshgrid(th, th))

    def reach(jr, Lz, jz):
        aAI = actionAngleStaeckelInverse(jrs=[jr], Lzs=[Lz], jzs=[jz], **kw)
        R, _, _, z, _, _ = (as_numpy(o) for o in aAI(jr, Lz, jz, thR, 0.0, thz))
        assert numpy.all(numpy.isfinite(R)) and numpy.all(R > 0.0)
        return numpy.amin(numpy.hypot(R, z)), numpy.amax(R)

    # about the outer maximum, at an energy at which the inner one is higher
    assert reach(0.015098, 0.002183, 0.232254)[0] > 0.15, (
        "The torus about the outer maximum of a split u libration reaches "
        "inside the barrier"
    )
    # across both wells
    assert reach(0.322184, 0.009836, 0.110735)[0] < 0.02, (
        "The torus across both wells of a split u libration does not reach "
        "the inner one"
    )
    # only the libration about the inner maximum has these actions
    assert reach(0.009532, 0.008259, 0.182493)[1] < 0.12, (
        "The torus about the inner maximum of a split u libration reaches "
        "beyond the barrier"
    )
    # in the jump of J_R across the barrier's level, and in the jump of J_z
    # where the v libration merges with the lobes of its momentum function
    # next to the axis
    with pytest.raises(ValueError, match="of 2 of the 3 tori"):
        actionAngleStaeckelInverse(
            jrs=[0.041499, 0.3, 0.00131],
            Lzs=[0.001016, 0.5, 0.00212],
            jzs=[0.225525, 0.2, 0.114926],
            **kw,
        )
    return None


def test_actionAngleStaeckel_inverse_single_model():
    # actionAngleStaeckel(inverse=) of tori in a single Staeckel model is
    # the transformation of that model
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    jr, Lz, jz = (
        numpy.array([0.08, 0.02, 0.2]),
        numpy.array([0.9, 1.2, -0.7]),
        numpy.array([0.06, 0.15, 0.02]),
    )
    kw = dict(pot=MWPotential2014, jrs=jr, Lzs=Lz, jzs=jz, mm_npt=128)
    aAI = actionAngleStaeckelInverse(delta=0.45, u0=1.2, progressbar=False, **kw)
    angs = numpy.random.default_rng(1).uniform(0.0, 2.0 * numpy.pi, (3, len(jr)))
    for c in (True, False):
        aAS = actionAngleStaeckel(inverse=aAI, c=c, order=20)
        out = numpy.array(aAS.actionsFreqsAngles(*aAI(jr, Lz, jz, *angs)))
        dang = (out[6:] - angs + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(out[:3] - numpy.array([jr, Lz, jz]))) < 1e-8, (
            f"actionAngleStaeckel(inverse=) does not return the actions of tori "
            f"in a single Staeckel model (c={c})"
        )
        assert (
            numpy.amax(numpy.fabs(out[3:6] - numpy.array(aAI.Freqs(jr, Lz, jz)))) < 1e-7
        ), (
            f"actionAngleStaeckel(inverse=) does not return the frequencies of "
            f"tori in a single Staeckel model (c={c})"
        )
        assert numpy.amax(numpy.fabs(dang)) < 1e-6, (
            f"actionAngleStaeckel(inverse=) does not return the angles of tori "
            f"in a single Staeckel model (c={c})"
        )
    # the units of the inverse carry over
    phys = actionAngleStaeckelInverse(delta=0.45, ro=8.0, vo=220.0, **kw)
    fwd = actionAngleStaeckel(inverse=phys)
    assert fwd._roSet and fwd._voSet and fwd._ro == 8.0 and fwd._vo == 220.0, (
        "actionAngleStaeckel(inverse=) does not take over the inverse's units"
    )
    with pytest.raises(TypeError, match="cannot be given with inverse="):
        actionAngleStaeckel(inverse=aAI, delta=0.4)
    return None


def _inverse_u0_of_ELz(E, Lz):
    return numpy.arcsinh((0.5 + 0.5 * Lz + 0.1 * E) / 0.45)


def _inverse_u0_of_ELzI3(E, Lz, I3):
    return _inverse_u0_of_ELz(E, Lz) + 0.02 * numpy.tanh(I3)


@pytest.mark.parametrize(
    "u0", ["torus", "planar", "shell", _inverse_u0_of_ELz, _inverse_u0_of_ELzI3]
)
def test_actionAngleStaeckel_inverse_u0torus(u0):
    # actionAngleStaeckel(inverse=) of tori that each have their own Staeckel
    # model (every rule for the reference curve: the built-in ones and
    # callables of two and of three integrals): every point is transformed in
    # the model of the torus through it, with the energy of that model
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    jr, Lz, jz = (
        numpy.array([0.08, 0.2, 0.02, 0.05, 0.3]),
        numpy.array([0.9, -0.7, 1.1, 1.3, 0.6]),
        numpy.array([0.06, 0.02, 0.1, 0.004, 0.3]),
    )
    kw = dict(pot=MWPotential2014, jrs=jr, Lzs=Lz, jzs=jz, mm_npt=256)
    aAI = actionAngleStaeckelInverse(delta=0.45, u0=u0, progressbar=False, **kw)
    aAS = actionAngleStaeckel(inverse=aAI, order=20)
    sel = numpy.tile(numpy.arange(len(jr)), 8)
    angs = numpy.random.default_rng(1).uniform(0.0, 2.0 * numpy.pi, (3, len(sel)))
    x = aAI(jr[sel], Lz[sel], jz[sel], *angs)
    out = numpy.array(aAS(*x))
    # (the most eccentric and inclined torus with the shell orbit's curve,
    # far from it, is itself resolved to a few 1e-9 at this mm_npt)
    assert numpy.amax(numpy.fabs(out - numpy.array([jr[sel], Lz[sel], jz[sel]]))) < (
        1e-8 if u0 == "shell" else 1e-9
    ), (
        "actionAngleStaeckel(inverse=) does not return the actions of tori "
        "that each have their own Staeckel model"
    )
    # the Staeckel approximation in the potential itself does not
    rough = numpy.array(actionAngleStaeckel(pot=MWPotential2014, delta=0.45)(*x))
    assert numpy.amax(numpy.fabs(rough[0] - jr[sel])) > 1e-4
    # the turning points are those of the same model
    rperi, rap = (numpy.asarray(q) for q in aAS.EccZmaxRperiRap(*x)[2:])
    assert numpy.all((rperi < rap) & (rperi > 0.0))
    # the frequencies and angles are the family's: the model's, corrected
    # for the dependence of the reference curve on the torus
    freqs = numpy.array(aAI.Freqs(jr, Lz, jz))[:, sel]
    for c in (True, False):
        full = numpy.array(aAS.actionsFreqsAngles(*x, c=c))
        dang = (full[6:] - angs + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(full[3:6] - freqs)) < 1e-7, (
            f"actionAngleStaeckel(inverse=) does not return the frequencies of "
            f"tori that each have their own Staeckel model (c={c})"
        )
        assert numpy.amax(numpy.fabs(dang)) < 1e-6, (
            f"actionAngleStaeckel(inverse=) does not return the angles of tori "
            f"that each have their own Staeckel model (c={c})"
        )
        assert numpy.allclose(
            numpy.array(aAS.actionsFreqs(*x, c=c)), full[:6], rtol=0.0, atol=1e-12
        )
    # the model's own frequencies are not the family's
    own = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    assert numpy.amax(numpy.fabs(numpy.array(own.actionsFreqs(*x))[3:] - freqs)) > 1e-4
    # the pure-Python implementation is the same transformation
    assert numpy.amax(numpy.fabs(numpy.array(aAS(*x, c=False)) - out)) < 1e-8, (
        "actionAngleStaeckel(inverse=) in Python does not agree with C for "
        "tori that each have their own Staeckel model"
    )
    assert numpy.allclose(
        aAS.EccZmaxRperiRap(*x, c=False)[2:], (rperi, rap), rtol=1e-6, atol=0.0
    )
    return None


def test_actionAngleStaeckel_inverse_deltafit():
    # actionAngleStaeckel(inverse=) of a family whose focal length follows
    # its tori (delta='fit'): every point is transformed in the model of the
    # torus through it, whose focal length is the family's at that torus's
    # actions; at the family's grid nodes, where its map is not interpolated,
    # the actions, frequencies, and angles are those of its tori
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import name_of_namespace, resolve_namespace, use
    from galpy.potential import MWPotential2014

    # (built on numpy, as such families are; evaluated under the backend)
    with use("numpy", force=True):
        aAI = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta="fit",
            delta_deg=(1, 1, 1),
            setup_interp=True,
            Rmin=0.85,
            Rmax=1.15,
            Rinf=2.5,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=128,
            progressbar=False,
        )
    aAS = actionAngleStaeckel(inverse=aAI, order=20)
    nodes = [(i, j, k) for i in (1, 2) for j in (1, 2) for k in (1, 2)] * 3
    jr, jz = numpy.array([aAI._A_tab[n][:2] for n in nodes]).T
    Lz = numpy.array([aAI._Lzgrid[n[2]] for n in nodes]) * numpy.tile(
        [1.0, -1.0], len(nodes) // 2
    )
    angs = numpy.random.default_rng(1).uniform(0.0, 2.0 * numpy.pi, (3, len(jr)))
    x = aAI(jr, Lz, jz, *angs)
    freqs = numpy.array(aAI.Freqs(jr, Lz, jz))
    # (on a backend, c=True runs the backend code that c=False runs)
    on_numpy = name_of_namespace(resolve_namespace()) == "numpy"
    for c in (True, False) if on_numpy else (True,):
        out = numpy.array(aAS.actionsFreqsAngles(*x, c=c))
        dang = (out[6:] - angs + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(out[:3] - numpy.array([jr, Lz, jz]))) < 1e-7, (
            f"actionAngleStaeckel(inverse=) does not return the actions of a "
            f"family whose focal length follows its tori (c={c})"
        )
        assert numpy.amax(numpy.fabs(out[3:6] - freqs)) < 1e-7, (
            f"actionAngleStaeckel(inverse=) does not return the frequencies of "
            f"a family whose focal length follows its tori (c={c})"
        )
        assert numpy.amax(numpy.fabs(dang)) < 1e-7, (
            f"actionAngleStaeckel(inverse=) does not return the angles of a "
            f"family whose focal length follows its tori (c={c})"
        )
        assert numpy.allclose(
            numpy.array(aAS(*x[:5], c=c)), out[:3], rtol=0.0, atol=1e-8
        )
        assert numpy.allclose(
            numpy.array(aAS.actionsFreqs(*x[:5], c=c)), out[:6], rtol=0.0, atol=1e-8
        )
    # a single focal length does not give the family's frequencies
    own = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    assert numpy.amax(numpy.fabs(numpy.array(own.actionsFreqs(*x))[3:] - freqs)) > 1e-3
    # a point outside the grid's range of energy or of L_z is transformed
    # like any other, with the focal length of the rule at the edge of that
    # range, and with a warning (the first point is on the grid)
    one = numpy.ones(3)
    far = (one, 0.1 * one, numpy.array([1.0, 1.0, 1.6]), 0.1 * one)
    far += (numpy.array([0.01, 0.9, 0.01]),)
    with pytest.warns(galpyWarning, match="2 of the 3 points lie outside"):
        out = numpy.array(aAS.actionsFreqsAngles(*far, 0.0 * one))
    py = out  # (on a backend, c=False runs the same backend code)
    if on_numpy:
        with pytest.warns(galpyWarning, match="2 of the 3 points lie outside"):
            py = numpy.array(aAS.actionsFreqsAngles(*far, 0.0 * one, c=False))
    assert numpy.all(numpy.isfinite(out)) and numpy.allclose(
        py, out, rtol=0.0, atol=1e-7
    ), "actionAngleStaeckel(inverse=) does not transform a point off a family's grid"
    with pytest.warns(galpyWarning, match="lie outside"):
        assert numpy.allclose(numpy.array(aAS(*far)), out[:3], atol=1e-8)
    # the family's delta method is that rule: held at the edge of the range
    # beyond it, so that another family and explicit tori that are given
    # it have the models of the forward transformation off the grid
    Lmin, top = aAI._Lzgrid[0], aAI.delta(5.0, 1.0, 0.3)
    assert (
        aAI.delta(-1.0, 0.5 * Lmin, 0.3) == aAI.delta(-1.0, Lmin, 0.3)
        and aAI.delta(50.0, 1.0, 0.3) == top
        and numpy.fabs(aAI.delta(-1.19, 1.0, 0.3) - top) > 1e-3
    ), "The delta method of a fitted family is not held beyond the grid's range"
    with use("numpy", force=True):
        wider = actionAngleStaeckelInverse(
            pot=MWPotential2014, delta=aAI.delta, setup_interp=True, progressbar=False
        )
    two = numpy.array(
        actionAngleStaeckel(inverse=wider, order=20).actionsFreqsAngles(*far, 0.0 * one)
    )
    assert numpy.allclose(two, out, rtol=0.0, atol=1e-7), (
        "actionAngleStaeckel(inverse=) of a family that was given another's delta "
        "method does not agree with that of the other off its grid"
    )
    with use("numpy", force=True):
        tori = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=aAI.delta,
            jrs=out[0],
            Lzs=out[1],
            jzs=out[2],
            mm_npt=128,
            progressbar=False,
        )
    back = numpy.array(tori(*out[:3], *out[6:]))[:5]
    # (the second point, far above the grid's energies, to the accuracy of
    # the forward transformation there)
    dev = numpy.amax(numpy.fabs(back - numpy.array(far)), axis=0)
    assert numpy.all(dev < [1e-8, 2e-6, 1e-8]), (
        "Explicit tori that were given a family's delta method do not return "
        "the points off its grid that actionAngleStaeckel(inverse=) transforms"
    )
    # a family with a callable rule: the rule itself where the family has
    # it, and the family's table of it for a pickled family, which does not
    import pickle

    kw = dict(pot=MWPotential2014, setup_interp=True, Rmin=0.85, Rmax=1.15, Rinf=2.5)
    kw.update(nE=4, nLz=4, nI3=4, mm_npt=128, progressbar=False)
    with use("numpy", force=True):
        cbI = actionAngleStaeckelInverse(
            delta=lambda E, Lz, f: 0.45 * (1.0 + 0.1 * (f - 0.5) + 0.05 * Lz), **kw
        )
    jr, jz = numpy.array([cbI._A_tab[n][:2] for n in nodes]).T
    x = cbI(jr, Lz, jz, *angs)
    freqs = numpy.array(cbI.Freqs(jr, Lz, jz))
    for fam in (cbI, pickle.loads(pickle.dumps(cbI))):
        out = numpy.array(
            actionAngleStaeckel(inverse=fam, order=20).actionsFreqsAngles(*x)
        )
        dang = (out[6:] - angs + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
        assert numpy.amax(numpy.fabs(out[:3] - numpy.array([jr, Lz, jz]))) < 1e-7, (
            "actionAngleStaeckel(inverse=) does not return the actions of a "
            "family with a callable delta="
        )
        assert (
            numpy.amax(numpy.fabs(out[3:6] - freqs)) < 1e-7
            and numpy.amax(numpy.fabs(dang)) < 1e-7
        ), (
            "actionAngleStaeckel(inverse=) does not return the frequencies and "
            "angles of a family with a callable delta="
        )
        assert not on_numpy or numpy.allclose(
            numpy.array(actionAngleStaeckel(inverse=fam, order=20)(*x[:5], c=False)),
            out[:3],
            rtol=0.0,
            atol=1e-8,
        ), (
            "actionAngleStaeckel(inverse=) in Python does not return the actions "
            "of the C code for a family with a callable delta="
        )
    return None


_POLISH_RULES = [{}, {"u0": "torus"}, {"delta": "fit", "delta_deg": (1, 1, 1)}]


def _polish_setup(rule):
    # a family, its forward transformation with the polish, and points of
    # its interpolated map
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import use
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.45, setup_interp=True, progressbar=False)
    kw.update(Rmin=0.7, Rmax=1.5, Rinf=4.0, nE=4, nLz=4, nI3=4, mm_npt=32)
    if rule.get("delta") == "fit":
        # (built on numpy, as such families are; evaluated under the backend)
        with use("numpy", force=True):
            aAI = actionAngleStaeckelInverse(**{**kw, **rule})
    else:
        aAI = actionAngleStaeckelInverse(**{**kw, **rule})
    aAS = actionAngleStaeckel(inverse=aAI, polish=True)
    rng = numpy.random.default_rng(2)
    jr, Lz, jz = (
        rng.uniform(0.0, 0.2, 400),
        rng.uniform(0.75, 1.45, 400) * rng.choice([-1.0, 1.0], 400),
        rng.uniform(0.0, 0.2, 400),
    )
    ok = aAI.in_domain(jr, Lz, jz)
    jr, Lz, jz = jr[ok][:60], Lz[ok][:60], jz[ok][:60]
    angs = rng.uniform(0.0, 2.0 * numpy.pi, (3, len(jr)))
    return aAI, aAS, (jr, Lz, jz), angs, numpy.array(aAI(jr, Lz, jz, *angs))


@pytest.mark.parametrize("rule", _POLISH_RULES)
def test_actionAngleStaeckel_inverse_polish(rule):
    # actionAngleStaeckel(inverse=, polish=True) of a family returns the
    # actions and angles that the family's interpolated map sends to each
    # point, whether the family has one Staeckel model or one per torus
    # (main's test, split in two for the backends' time per test; the
    # other half is test_actionAngleStaeckel_inverse_polish_few_points)
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    aAI, aAS, (jr, Lz, jz), angs, x = _polish_setup(rule)
    out = numpy.array(aAS.actionsFreqsAngles(*x))
    dang = (out[6:] - angs + numpy.pi) % (2.0 * numpy.pi) - numpy.pi
    assert numpy.amax(numpy.fabs(out[:3] - numpy.array([jr, Lz, jz]))) < 1e-10, (
        "actionAngleStaeckel(inverse=, polish=True) does not invert a family's actions"
    )
    assert (
        numpy.amax(numpy.fabs(out[3:6] - numpy.array(aAI.Freqs(jr, Lz, jz)))) < 1e-9
    ), (
        "actionAngleStaeckel(inverse=, polish=True) does not return a family's frequencies"
    )
    assert numpy.amax(numpy.fabs(dang)) < 1e-9, (
        "actionAngleStaeckel(inverse=, polish=True) does not invert a family's angles"
    )
    # without the polish, the family's map is inverted to its interpolation
    # error only
    rough = numpy.array(actionAngleStaeckel(inverse=aAI)(*x[:5]))
    assert 1e-8 < numpy.amax(numpy.fabs(rough[0] - jr)) < 1e-2
    # the polish is of a family's map
    if not rule:
        with pytest.raises(ValueError, match="polish= requires"):
            actionAngleStaeckel(pot=MWPotential2014, delta=0.45, polish=True)
    return None


@pytest.mark.parametrize("rule", _POLISH_RULES)
def test_actionAngleStaeckel_inverse_polish_few_points(rule):
    # actionAngleStaeckel(inverse=, polish=True) of a few points: tori
    # without one or both of the librations, the other methods and inputs,
    # and points that no torus of the family passes through
    from galpy.orbit import Orbit

    aAI, aAS, _, _, x = _polish_setup(rule)
    # tori without one or both of the librations (the angle of a vanishing
    # action is arbitrary)
    fjr, fLz, fjz = (
        numpy.array([0.05, 0.0, 0.0, 0.03]),
        numpy.array([1.0, 1.2, 0.9, -1.1]),
        numpy.array([0.0, 0.04, 0.0, 0.0]),
    )
    flat = numpy.array(aAS.actionsFreqsAngles(*aAI(fjr, fLz, fjz, 1.0, 2.0, 3.0)))
    assert numpy.amax(numpy.fabs(flat[:3] - numpy.array([fjr, fLz, fjz]))) < 1e-10, (
        "actionAngleStaeckel(inverse=, polish=True) does not invert a family's "
        "tori with a vanishing action"
    )
    # the other methods and inputs: actions alone without phi, an Orbit
    out = numpy.array(aAS.actionsFreqsAngles(*x[:, :3]))
    assert numpy.allclose(numpy.array(aAS(*x[:5, :3])), out[:3], rtol=0.0, atol=1e-11)
    assert numpy.allclose(
        numpy.array(aAS.actionsFreqs(Orbit(x[:, :3].T))),
        out[:6],
        rtol=0.0,
        atol=1e-11,
    )
    # points that no torus of the family passes through
    with pytest.warns(galpyWarning, match="not on a torus of inverse="):
        off = numpy.array(
            aAS.actionsFreqsAngles(
                numpy.array([1.0, x[0, 0]]),
                numpy.array([0.9, x[1, 0]]),
                numpy.array([1.0, x[2, 0]]),
                numpy.array([0.0, x[3, 0]]),
                numpy.array([2.0, x[4, 0]]),
                numpy.array([0.0, x[5, 0]]),
            )
        )
    assert numpy.all(numpy.isnan(off[[0, 2, 3, 4, 5, 6, 7, 8], 0])) and numpy.all(
        numpy.isfinite(off[:, 1])
    ), "actionAngleStaeckel(inverse=, polish=True) does not return nan off the family"
    return None


# actionAngleStaeckel(inverse=) close to the symmetry axis, where the
# momentum function of the u libration has two lobes and the iteration for
# the reference curve of a point's torus does not converge on its own
def test_actionAngleStaeckel_inverse_nearaxis():
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import name_of_namespace, resolve_namespace, use
    from galpy.potential import MWPotential2014
    from galpy.util import galpyWarning

    # (the family set up with numpy; on a backend, C and Python are the same
    # backend code, run once)
    on_numpy = name_of_namespace(resolve_namespace()) == "numpy"
    with use("numpy", force=True):
        fam = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=0.3,
            u0="torus",
            setup_interp=True,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=16,
            progressbar=False,
        )
    # points of explicit tori (mm_npt=2048) of the same delta and u0, with
    # their actions and angles: the iteration starts in a model in which the
    # point is unbound, or cycles between two models
    x = numpy.array(
        [
            [1.3267761370656899, -0.2848181700458333, 0.008451579340850553,
             3.1377831825578557, -0.44892690709616395, 4.988159473847943],
            [0.021903441859716655, -1.8349049193596925, -1.146280448585597,
             0.3001546680366327, 0.3538612503538491, 5.419170176063995],
            [1.853481323788396, 0.04553761502142584, 0.0006463128272536712,
             -2.433172469481341, -0.28105638136566685, 3.1151682303146635],
            [0.003737462681295734, -2.115570785838872, -0.26604529254019094,
             -0.31242862791340587, 0.09782480253575135, 4.856416063842641],
        ]
    ).T  # fmt: skip
    jt = numpy.array(
        [
            [1.182418040998694, 0.011213353789957886, 0.4662783034154542,
             4.688436699052436, 5.075298574657615, 0.9254495061799101],
            [1.378793945119122, -0.02510748716052455, 0.44298870111665367,
             6.282911676292473, 4.403371120106943, 1.4750133216604375],
            [0.8745522399516648, 0.0011979287546395553, 0.4691823389139185,
             2.4377512621089314, 3.41759195384933, 3.4475543979237586],
            [0.47000582306714855, -0.0009943343524033699, 0.43787154295535813,
             6.262769210330358, 3.264869388376143, 4.854572168552439],
        ]
    ).T  # fmt: skip
    for c in (True, False) if on_numpy else (True,):
        aAS = actionAngleStaeckel(inverse=fam, c=c)
        out = numpy.array(aAS.actionsFreqsAngles(*x))
        assert numpy.allclose(out[:3], jt[:3], rtol=0.0, atol=3e-6), (
            f"actionAngleStaeckel(inverse=, c={c}) does not return the actions "
            "of tori close to the symmetry axis"
        )
        assert numpy.allclose(out[6:], jt[3:], rtol=0.0, atol=1e-3), (
            f"actionAngleStaeckel(inverse=, c={c}) does not return the angles "
            "of tori close to the symmetry axis"
        )
        assert numpy.allclose(
            numpy.array(aAS(*x[:5])), out[:3], rtol=0.0, atol=1e-10
        ) and numpy.allclose(
            numpy.array(aAS.actionsFreqs(*x[:5])), out[:6], rtol=0.0, atol=1e-10
        ), f"actionAngleStaeckel(inverse=, c={c}) methods differ close to the axis"
        # points that no torus passes through: in a gap between the tori
        # (where the inner turning point jumps between the lobes), unbound
        gap = [0.09240174859677441, -0.33436781352537776, 0.01642332461550298,
               -0.858922697831848, 0.9767430270265254, 0.0]  # fmt: skip
        unbound = [1.0, 3.0, 1.0, 0.1, 0.2, 0.0]
        pts = numpy.array([gap, unbound, x[:, 0]]).T
        for method, n in (
            (aAS.actionsFreqsAngles, 6),
            (aAS.actionsFreqs, 5),
            (aAS, 5),
        ):
            with pytest.warns(galpyWarning, match="2 of the 3 points are not on"):
                off = numpy.array(method(*pts[:n]))
            assert (
                numpy.all(numpy.isnan(numpy.delete(off, 1, axis=0)[:, :2]))
                and numpy.all(numpy.isfinite(off[:, 2]))
                and numpy.allclose(off[1], pts[0] * pts[2])
            ), f"actionAngleStaeckel(inverse=, c={c}) is not nan off the tori"
        with pytest.warns(galpyWarning, match="1 of the 1 points are not on"):
            assert numpy.isnan(aAS(*unbound[:5])[0][0])
    # a focal length that follows the tori as well: C against Python
    if not on_numpy:
        # (C and Python are the same backend code there: the fitted rule
        # close to the axis on a backend is test_..._inverse_nearaxis_fit's)
        return None
    fam = actionAngleStaeckelInverse(
        pot=MWPotential2014,
        delta="fit",
        u0="torus",
        setup_interp=True,
        nE=6,
        nLz=6,
        nI3=6,
        mm_npt=64,
        progressbar=False,
    )
    x = numpy.array(
        [
            [0.45228688699702013, -0.13050461587844964, 0.01796834670575424,
             -2.5741563918786396, 0.04652282147245728],
            [0.8003757901056533, -0.0290118399962716, 0.0174757261263844,
             -2.45092834381873, -0.02797011556256042],
            [0.09240174859677441, -0.33436781352537776, 0.01642332461550298,
             -0.858922697831848, 0.9767430270265254],
        ]
    ).T  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", galpyWarning)
        out = [
            numpy.array(actionAngleStaeckel(inverse=fam, c=c)(*x))
            for c in (True, False)
        ]
    assert numpy.all(numpy.isfinite(out[0][:, :2])) and numpy.allclose(
        out[0][:, :2], out[1][:, :2], rtol=0.0, atol=1e-6
    ), "actionAngleStaeckel(inverse=) with delta='fit' differs between C and Python"
    return None


def test_actionAngleStaeckel_inverse_nearaxis_fit():
    # For a family whose focal length follows its tori by the fitted rule,
    # the focal length of the torus through a point is a fixed point of the
    # rule. Close to the symmetry axis the rule is not defined at every
    # focal length (the point is on no torus of some), and the iteration
    # can start at, or step to, one where it is not although the fixed
    # point exists: these points are solved for by bracketing, and C and
    # Python agree on them and on the points that are on no torus
    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import name_of_namespace, resolve_namespace, use
    from galpy.potential import MWPotential2014
    from galpy.util import galpyWarning

    # (the family set up with numpy; on a backend, C and Python are the same
    # backend code, run once)
    on_numpy = name_of_namespace(resolve_namespace()) == "numpy"
    with use("numpy", force=True):
        fam = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta="fit",
            u0="torus",
            setup_interp=True,
            nE=5,
            nLz=5,
            nI3=5,
            mm_npt=32,
            progressbar=False,
        )
    # four points whose focal length the iteration does not find (the first
    # from C, the others from Python), and three that are on no torus of the
    # family's rule
    x = numpy.array(
        [
            [0.48338706674192317, -0.7225153255750134, -0.03088495958464515, 0.4849125307190425, 0.3466863229537384],
            [2.263004194937672, 0.3885950256128122, -0.004413752377344823, 1.1396172230949002, -0.12977835378893876],
            [0.5715959802707503, -0.2807778867586726, 0.004480860692506641, 1.1325103872165077, 0.1059506621009235],
            [0.34135411290346473, -0.7068648898571247, 0.0347781501349293, -0.8069430106440976, 0.48158637694732803],
            [0.12838270122596657, 0.09187311237546948, -0.0009217365275638495, 0.5327978161696283, -0.8121233557162888],
            [0.6357503174271786, -0.01975186792013046, 0.017750945701678594, 1.747728793897359, -0.47981606272293625],
            [1.545268791580364, 0.004669803341139917, -0.0068426999559747, 0.8500650097427931, 0.38041633315831735],
        ]
    ).T  # fmt: skip
    # (on a backend, one of the points that are found and one that is on no
    # torus: an eager backend pays for each step of the bracketing that all
    # of them take, on arrays the more points the larger)
    nfound, x = (4, x) if on_numpy else (1, x[:, [0, 4]])
    out = {}
    for c in (True, False) if on_numpy else (True,):
        aAS = actionAngleStaeckel(inverse=fam, c=c)
        with pytest.warns(
            galpyWarning,
            match=f"{x.shape[1] - nfound} of the {x.shape[1]} points are not on a torus",
        ):
            out[c] = numpy.array(aAS.actionsFreqsAngles(*x, 0.0 * x[0]))
        # (the angular momentum is that of the point itself)
        assert numpy.all(numpy.isfinite(out[c][:, :nfound])) and numpy.all(
            numpy.isnan(numpy.delete(out[c][:, nfound:], 1, axis=0))
        ), (
            f"actionAngleStaeckel(inverse=, c={c}) with a fitted focal length "
            "does not find the tori of points close to the symmetry axis, or "
            "returns values for points that are on none"
        )
    assert numpy.allclose(
        out[True][:3, :nfound], out[not on_numpy][:3, :nfound], rtol=0.0, atol=1e-6
    ), (
        "actionAngleStaeckel(inverse=) with a fitted focal length: the actions "
        "from C and Python differ close to the symmetry axis"
    )
    # and the first of them is where its explicit torus puts it
    jr, lz, jz, _, _, _, ar, ap, az = out[True][:, 0]
    with use("numpy", force=True):
        tor = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta=fam.delta,
            u0="torus",
            jrs=[jr],
            Lzs=[lz],
            jzs=[jz],
            mm_npt=512,
            progressbar=False,
        )
    xt = numpy.array(tor(jr, lz, jz, ar, ap, az)).ravel()
    assert numpy.allclose(xt[:5], x[:, 0], rtol=0.0, atol=1e-3), (
        "actionAngleStaeckel(inverse=) with a fitted focal length: a point "
        "close to the symmetry axis is not on the torus that it is assigned"
    )
    return None
