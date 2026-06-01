# Tests of integrating orbits in non-inertial frames
import numpy
import pytest

from galpy import potential
from galpy.orbit import Orbit
from galpy.util import coords


def test_lsrframe_scalaromegaz():
    # Test that integrating an orbit in the LSR frame is equivalent to
    # normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    framepot = potential.NonInertialFrameForce(cinterp=False, Omega=omega)
    dp_frame = potential.DehnenBarPotential(omegab=1.8 - omega, rb=0.5, Af=0.03)
    diskframepot = lp + dp_frame + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for integration method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for integration method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_lsrframe_scalaromegaz_2d():
    # Test that integrating an orbit in the LSR frame is equivalent to
    # normal orbit integration in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    framepot = potential.NonInertialFrameForce(cinterp=False, Omega=omega)
    dp_frame = potential.DehnenBarPotential(omegab=1.8 - omega, rb=0.5, Af=0.03)
    diskframepot = lp + dp_frame + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for integration method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for integration method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_lsrframe_vecomegaz():
    # Test that integrating an orbit in the LSR frame is equivalent to
    # normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=numpy.array([0.0, 0.0, omega])
    )
    dp_frame = potential.DehnenBarPotential(omegab=1.8 - omega, rb=0.5, Af=0.03)
    diskframepot = lp + dp_frame + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_lsrframe_vecomegaz_2d():
    # Test that integrating an orbit in the LSR frame is equivalent to
    # normal orbit integration in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=numpy.array([0.0, 0.0, omega])
    )
    dp_frame = potential.DehnenBarPotential(omegab=1.8 - omega, rb=0.5, Af=0.03)
    diskframepot = lp + dp_frame + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_scalaromegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega, Omegadot=omegadot
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_scalaromegaz_2d():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega, Omegadot=omegadot
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_vecomegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False,
        Omega=numpy.array([0.0, 0.0, omega]),
        Omegadot=numpy.array([0.0, 0.0, omegadot]),
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_vecomegaz_2d():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False,
        Omega=numpy.array([0.0, 0.0, omega]),
        Omegadot=numpy.array([0.0, 0.0, omegadot]),
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_funcomegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    omega_func = lambda t: lp.omegac(1.0) + 0.02 * t
    omegadot_func = lambda t: 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_funcomegaz_2d():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    omega_func = lambda t: lp.omegac(1.0) + 0.02 * t
    omegadot_func = lambda t: 0.02
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_vecfuncomegaz():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    omega_func = [lambda t: 0.0, lambda t: 0.0, lambda t: lp.omegac(1.0) + 0.02 * t]
    omegadot_func = [lambda t: 0.0, lambda t: 0.0, lambda t: 0.02]
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_accellsrframe_vecfuncomegaz_2D():
    # Test that integrating an orbit in an LSR frame that is accelerating
    # is equivalent to normal orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = lp.omegac(1.0)
    omegadot = 0.02
    omega_func = [lambda t: 0.0, lambda t: 0.0, lambda t: lp.omegac(1.0) + 0.02 * t]
    omegadot_func = [lambda t: 0.0, lambda t: 0.0, lambda t: 0.02]
    diskpot = lp
    framepot = potential.NonInertialFrameForce(
        cinterp=False, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = lp + framepot

    # Now integrate the orbit of the Sun in both the inertial and the lsr frame
    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.R(ts) * numpy.cos(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        o_ys = o.R(ts) * numpy.sin(o.phi(ts) - omega * ts - omegadot * ts**2.0 / 2.0)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in the acceleratingly-rotating LSR frame does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-6)
    check_orbit(method="dop853_c", tol=1e-9)
    return None


def test_arbitraryaxisrotation_nullpotential():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works
    # Start with a test where there is no potential, so a static
    # object should remain static
    np = potential.NullPotential()

    def check_orbit(zvec=[1.0, 0.0, 1.0], omega=1.3, method="odeint", tol=1e-9):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=np, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, np, method=method)
        # Non-inertial frame
        # First compute initial condition
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(pot=np, rot=rot, omega=omega)
            + potential.NonInertialFrameForce(
                cinterp=False, Omega=numpy.array(derive_noninert_omega(omega, rot=rot))
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t), op.y(t), phi=op.z(t), t=t, rot=rot, omega=omega, rect=True
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[0.0, 1.0, 1.0], omega=1.3, method="odeint", tol=1e-5)
    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-4)
    check_orbit(
        zvec=[2.0, 3.0, 1.0], omega=0.9, method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_arbitraryaxisrotation():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(zvec=[1.0, 0.0, 1.0], omega=1.3, method="odeint", tol=1e-9):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        # First compute initial condition
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(pot=diskpot, rot=rot, omega=omega)
            + potential.NonInertialFrameForce(
                cinterp=False, Omega=numpy.array(derive_noninert_omega(omega, rot=rot))
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t), op.y(t), phi=op.z(t), t=t, rot=rot, omega=omega, rect=True
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[0.0, 1.0, 1.0], omega=1.3, method="odeint", tol=1e-5)
    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-4)
    check_orbit(
        zvec=[2.0, 3.0, 1.0], omega=0.9, method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_arbitraryaxisrotation_omegadot_nullpotential():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works, where the frame rotation is changing in time
    # Start with a test where there is no potential, so a static
    # object should remain static
    np = potential.NullPotential()

    def check_orbit(
        zvec=[1.0, 0.0, 1.0], omega=1.3, omegadot=0.1, method="odeint", tol=1e-9
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=np, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, np, method=method)
        # Non-inertial frame
        # First compute initial condition, no need to specify omegadot, bc t=0
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        # Omegadot is just a scaled version of Omega
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(
                pot=np, rot=rot, omega=omega, omegadot=omegadot
            )
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=numpy.array(derive_noninert_omega(omega, rot=rot)),
                Omegadot=numpy.array(derive_noninert_omega(omega, rot=rot))
                * omegadot
                / omega,
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t),
                op.y(t),
                phi=op.z(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                rect=True,
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[0.0, 1.0, 1.0], omega=1.3, method="odeint", tol=1e-5)
    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-4)
    check_orbit(
        zvec=[2.0, 3.0, 1.0], omega=0.9, method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_arbitraryaxisrotation_omegadot():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works, where the frame rotation is changing in time
    # Start with a test where there is no potential, so a static
    # object should remain static
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(
        zvec=[1.0, 0.0, 1.0], omega=1.3, omegadot=0.03, method="odeint", tol=1e-9
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        # First compute initial condition, no need to specify omegadot, bc t=0
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        # Omegadot is just a scaled version of Omega
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(
                pot=diskpot, rot=rot, omega=omega, omegadot=omegadot
            )
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=numpy.array(derive_noninert_omega(omega, rot=rot)),
                Omegadot=numpy.array(derive_noninert_omega(omega, rot=rot))
                * omegadot
                / omega,
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t),
                op.y(t),
                phi=op.z(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                rect=True,
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-5)
    return None


def test_arbitraryaxisrotation_omegafunc_nullpotential():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works, where the frame rotation is changing in time
    # Start with a test where there is no potential, so a static
    # object should remain static
    np = potential.NullPotential()

    def check_orbit(
        zvec=[1.0, 0.0, 1.0],
        omega=1.3,
        omegadot=0.1,
        omegadotdot=0.01,
        method="odeint",
        tol=1e-9,
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=np, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, np, method=method)
        # Non-inertial frame
        # First compute initial condition, no need to specify omegadot, bc t=0
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        # Omegadot is just a scaled version of Omega
        Omega = numpy.array(derive_noninert_omega(omega, rot=rot))
        Omegadot = numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadot / omega
        Omegadotdot = (
            numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadotdot / omega
        )
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(
                pot=np, rot=rot, omega=omega, omegadot=omegadot, omegadotdot=omegadotdot
            )
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=[
                    lambda t: (
                        Omega[0] + Omegadot[0] * t + Omegadotdot[0] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[1] + Omegadot[1] * t + Omegadotdot[1] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[2] + Omegadot[2] * t + Omegadotdot[2] * t**2.0 / 2.0
                    ),
                ],
                Omegadot=[
                    lambda t: Omegadot[0] + Omegadotdot[0] * t,
                    lambda t: Omegadot[1] + Omegadotdot[1] * t,
                    lambda t: Omegadot[2] + Omegadotdot[2] * t,
                ],
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t),
                op.y(t),
                phi=op.z(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                omegadotdot=omegadotdot,
                rect=True,
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                omegadotdot=omegadotdot,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[0.0, 1.0, 1.0], omega=1.3, method="odeint", tol=1e-5)
    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-4)
    check_orbit(zvec=[2.0, 3.0, 1.0], omega=0.9, method="dop853_c", tol=1e-4)
    return None


def test_arbitraryaxisrotation_omegafunc():
    # Test that integrating an orbit in a frame rotating around an
    # arbitrary axis works, where the frame rotation is changing in time
    # Start with a test where there is no potential, so a static
    # object should remain static
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(
        zvec=[1.0, 0.0, 1.0],
        omega=1.3,
        omegadot=0.03,
        omegadotdot=0.01,
        method="odeint",
        tol=1e-9,
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the inertial frame
        # and then as seen by the rotating observer
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        # First compute initial condition, no need to specify omegadot, bc t=0
        Rp, phip, zp = rotate_and_omega(
            o.R(), o.z(), phi=o.phi(), t=0.0, rot=rot, omega=-omega
        )
        vRp, vTp, vzp = rotate_and_omega_vec(
            o.vR(),
            o.vT(),
            o.vz(),
            o.R(),
            o.z(),
            phi=o.phi(),
            t=0.0,
            rot=rot,
            omega=-omega,
        )
        op = Orbit([Rp, vRp, vTp, zp, vzp, phip])
        # Omegadot is just a scaled version of Omega
        Omega = numpy.array(derive_noninert_omega(omega, rot=rot))
        Omegadot = numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadot / omega
        Omegadotdot = (
            numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadotdot / omega
        )
        op.integrate(
            ts,
            RotatingPotentialWrapperPotential(
                pot=diskpot,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                omegadotdot=omegadotdot,
            )
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=[
                    lambda t: (
                        Omega[0] + Omegadot[0] * t + Omegadotdot[0] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[1] + Omegadot[1] * t + Omegadotdot[1] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[2] + Omegadot[2] * t + Omegadotdot[2] * t**2.0 / 2.0
                    ),
                ],
                Omegadot=[
                    lambda t: Omegadot[0] + Omegadotdot[0] * t,
                    lambda t: Omegadot[1] + Omegadotdot[1] * t,
                    lambda t: Omegadot[2] + Omegadotdot[2] * t,
                ],
            ),
            method=method,
        )
        # Compare
        # Orbit in the inertial frame
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vRs = o.vR(ts)
        o_vTs = o.vT(ts)
        o_vzs = o.vz(ts)
        # and that computed in the non-inertial frame converted back to inertial
        op_xs, op_ys, op_zs = [], [], []
        op_vRs, op_vTs, op_vzs = [], [], []
        for ii, t in enumerate(ts):
            xyz = rotate_and_omega(
                op.x(t),
                op.y(t),
                phi=op.z(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                omegadotdot=omegadotdot,
                rect=True,
            )
            op_xs.append(xyz[0])
            op_ys.append(xyz[1])
            op_zs.append(xyz[2])
            vRTz = rotate_and_omega_vec(
                op.vR(t),
                op.vT(t),
                op.vz(t),
                op.R(t),
                op.z(t),
                phi=op.phi(t),
                t=t,
                rot=rot,
                omega=omega,
                omegadot=omegadot,
                omegadotdot=omegadotdot,
            )
            op_vRs.append(vRTz[0])
            op_vTs.append(vRTz[1])
            op_vzs.append(vRTz[2])
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vRs - op_vRs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vTs - op_vTs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a rotating frame around an arbitrary axis does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(zvec=[2.0, 0.0, 1.0], omega=-1.1, method="dop853", tol=1e-5)
    return None


def test_linacc_constantacc_z():
    # Test that a linearly-accelerating frame along the z direction works
    # with a constant acceleration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    az = 0.02
    # Trick to use a non-numba version, by causing numba compilation
    # to fail. Fails because scipy special is not supported in base
    # numba
    from scipy import special

    intaz = lambda t: (
        0.02 * t**2.0 / 2.0 * (special.erf(t) + 2.0) / (special.erf(t) + 2.0)
    )
    framepot = potential.NonInertialFrameForce(cinterp=False, a0=[0.0, 0.0, az])
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot, x0=[lambda t: 0.0, lambda t: 0.0, intaz]
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = o()
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        op_zs = op.z(ts) + intaz(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-9)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_constantacc_x_2d():
    # Test that a linearly-accelerating frame along the x direction works
    # with a constant acceleration in 2D
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    ax = 0.02
    intax = lambda t: ax * t**2.0 / 2.0
    framepot = potential.NonInertialFrameForce(cinterp=False, a0=[ax, 0.0, 0.0])
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot, x0=[intax, lambda t: 0.0, lambda t: 0.0]
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = o()
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        op_xs = op.x(ts) + intax(ts)
        op_ys = op.y(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_constantacc_xyz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a constant acceleration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    ax, ay, az = -0.03, 0.04, 0.02
    inta = [
        lambda t: -0.03 * t**2.0 / 2.0,
        lambda t: 0.04 * t**2.0 / 2.0,
        lambda t: 0.02 * t**2.0 / 2.0,
    ]
    framepot = potential.NonInertialFrameForce(cinterp=False, a0=[ax, ay, az])
    diskframepot = (
        AcceleratingPotentialWrapperPotential(pot=diskpot, x0=inta) + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = o()
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        op_xs = op.x(ts) + inta[0](ts)
        op_ys = op.y(ts) + inta[1](ts)
        op_zs = op.z(ts) + inta[2](ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_z():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    az = lambda t: 0.02 + 0.03 * t / 20.0
    intaz = lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0
    framepot = potential.NonInertialFrameForce(
        cinterp=False, a0=[lambda t: 0.0, lambda t: 0.0, az]
    )
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot, x0=[lambda t: 0.0, lambda t: 0.0, intaz]
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = o()
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        op_xs = op.x(ts)
        op_ys = op.y(ts)
        op_zs = op.z(ts) + intaz(ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-9)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_xyz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    ax, ay, az = (
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    )
    inta = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    framepot = potential.NonInertialFrameForce(cinterp=False, a0=[ax, ay, az])
    diskframepot = (
        AcceleratingPotentialWrapperPotential(pot=diskpot, x0=inta) + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = o()
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        op_xs = op.x(ts) + inta[0](ts)
        op_ys = op.y(ts) + inta[1](ts)
        op_zs = op.z(ts) + inta[2](ts)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_xyz_accellsrframe_scalaromegaz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration, also combining it with changing
    # rotation around the z axis
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.02
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega, Omegadot=omegadot
    )
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot, x0=x0, omegaz=omega, omegazdot=omegadot
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vxs = o.vx(ts)
        o_vys = o.vy(ts)
        o_vzs = o.vz(ts)
        op_xs = op.x(ts) + x0[0](ts)
        op_ys = op.y(ts) + x0[1](ts)
        op_zs = op.z(ts) + x0[2](ts)
        Rp, phip, _ = coords.rect_to_cyl(op_xs, op_ys, op_zs)
        phip += omega * ts + omegadot * ts**2.0 / 2.0
        op_xs, op_ys, _ = coords.cyl_to_rect(Rp, phip, op_zs)
        op_vxs = op.vx(ts) + v0[0](ts)
        op_vys = op.vy(ts) + v0[1](ts)
        op_vzs = op.vz(ts) + v0[2](ts)
        vRp, vTp, _ = coords.rect_to_cyl_vec(
            op_vxs,
            op_vys,
            op_vzs,
            op.x(ts) + x0[0](ts),
            op.y(ts) + x0[1](ts),
            op.z(ts) + x0[2](ts),
        )
        vTp += omega * Rp + omegadot * ts * Rp
        op_vxs, op_vys, _ = coords.cyl_to_rect_vec(vRp, vTp, op_vzs, phi=phip)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vxs - op_vxs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vys - op_vys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_xyz_accellsrframe_vecomegaz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration, also combining it with changing
    # rotation around the z axis
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.02
    framepot = potential.NonInertialFrameForce(
        cinterp=False,
        x0=x0,
        v0=v0,
        a0=a0,
        Omega=numpy.array([0.0, 0.0, omega]),
        Omegadot=numpy.array([0.0, 0.0, omegadot]),
    )
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot, x0=x0, omegaz=omega, omegazdot=omegadot
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vxs = o.vx(ts)
        o_vys = o.vy(ts)
        o_vzs = o.vz(ts)
        op_xs = op.x(ts) + x0[0](ts)
        op_ys = op.y(ts) + x0[1](ts)
        op_zs = op.z(ts) + x0[2](ts)
        Rp, phip, _ = coords.rect_to_cyl(op_xs, op_ys, op_zs)
        phip += omega * ts + omegadot * ts**2.0 / 2.0
        op_xs, op_ys, _ = coords.cyl_to_rect(Rp, phip, op_zs)
        op_vxs = op.vx(ts) + v0[0](ts)
        op_vys = op.vy(ts) + v0[1](ts)
        op_vzs = op.vz(ts) + v0[2](ts)
        vRp, vTp, _ = coords.rect_to_cyl_vec(
            op_vxs,
            op_vys,
            op_vzs,
            op.x(ts) + x0[0](ts),
            op.y(ts) + x0[1](ts),
            op.z(ts) + x0[2](ts),
        )
        vTp += omega * Rp + omegadot * ts * Rp
        op_vxs, op_vys, _ = coords.cyl_to_rect_vec(vRp, vTp, op_vzs, phi=phip)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vxs - op_vxs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vys - op_vys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_xyz_accellsrframe_scalarfuncomegaz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration, also combining it with changing
    # rotation around the z axis
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.1
    omegadotdot = 0.01
    omega_func = lambda t: omega + omegadot * t + omegadotdot * t**2.0 / 2.0
    omegadot_func = lambda t: omegadot + omegadotdot * t
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot,
            x0=x0,
            omegaz=omega,
            omegazdot=omegadot,
            omegazdotdot=omegadotdot,
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vxs = o.vx(ts)
        o_vys = o.vy(ts)
        o_vzs = o.vz(ts)
        op_xs = op.x(ts) + x0[0](ts)
        op_ys = op.y(ts) + x0[1](ts)
        op_zs = op.z(ts) + x0[2](ts)
        Rp, phip, _ = coords.rect_to_cyl(op_xs, op_ys, op_zs)
        phip += omega * ts + omegadot * ts**2.0 / 2.0 + omegadotdot * ts**3.0 / 6.0
        op_xs, op_ys, _ = coords.cyl_to_rect(Rp, phip, op_zs)
        op_vxs = op.vx(ts) + v0[0](ts)
        op_vys = op.vy(ts) + v0[1](ts)
        op_vzs = op.vz(ts) + v0[2](ts)
        vRp, vTp, _ = coords.rect_to_cyl_vec(
            op_vxs,
            op_vys,
            op_vzs,
            op.x(ts) + x0[0](ts),
            op.y(ts) + x0[1](ts),
            op.z(ts) + x0[2](ts),
        )
        vTp += omega * Rp + omegadot * ts * Rp + omegadotdot * ts**2.0 / 2.0 * Rp
        op_vxs, op_vys, _ = coords.cyl_to_rect_vec(vRp, vTp, op_vzs, phi=phip)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vxs - op_vxs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vys - op_vys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_linacc_changingacc_xyz_accellsrframe_funcomegaz():
    # Test that a linearly-accelerating frame along the z direction works
    # with a changing acceleration, also combining it with changing
    # rotation around the z axis
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.1
    omegadotdot = 0.01
    omega_func = [
        lambda t: 0.0,
        lambda t: 0.0,
        lambda t: omega + omegadot * t + omegadotdot * t**2.0 / 2.0,
    ]
    omegadot_func = [lambda t: 0.0, lambda t: 0.0, lambda t: omegadot + omegadotdot * t]
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega_func, Omegadot=omegadot_func
    )
    diskframepot = (
        AcceleratingPotentialWrapperPotential(
            pot=diskpot,
            x0=x0,
            omegaz=omega,
            omegazdot=omegadot,
            omegazdotdot=omegadotdot,
        )
        + framepot
    )

    def check_orbit(method="odeint", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # Inertial frame
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot, method=method)
        # Non-inertial frame
        op = Orbit([o.R(), o.vR(), o.vT() - omega * o.R(), o.z(), o.vz(), o.phi()])
        op.integrate(ts, diskframepot, method=method)
        # Compare
        o_xs = o.x(ts)
        o_ys = o.y(ts)
        o_zs = o.z(ts)
        o_vxs = o.vx(ts)
        o_vys = o.vy(ts)
        o_vzs = o.vz(ts)
        op_xs = op.x(ts) + x0[0](ts)
        op_ys = op.y(ts) + x0[1](ts)
        op_zs = op.z(ts) + x0[2](ts)
        Rp, phip, _ = coords.rect_to_cyl(op_xs, op_ys, op_zs)
        phip += omega * ts + omegadot * ts**2.0 / 2.0 + omegadotdot * ts**3.0 / 6.0
        op_xs, op_ys, _ = coords.cyl_to_rect(Rp, phip, op_zs)
        op_vxs = op.vx(ts) + v0[0](ts)
        op_vys = op.vy(ts) + v0[1](ts)
        op_vzs = op.vz(ts) + v0[2](ts)
        vRp, vTp, _ = coords.rect_to_cyl_vec(
            op_vxs,
            op_vys,
            op_vzs,
            op.x(ts) + x0[0](ts),
            op.y(ts) + x0[1](ts),
            op.z(ts) + x0[2](ts),
        )
        vTp += omega * Rp + omegadot * ts * Rp + omegadotdot * ts**2.0 / 2.0 * Rp
        op_vxs, op_vys, _ = coords.cyl_to_rect_vec(vRp, vTp, op_vzs, phi=phip)
        assert numpy.amax(numpy.fabs(o_xs - op_xs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_ys - op_ys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_zs - op_zs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vxs - op_vxs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vys - op_vys)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )
        assert numpy.amax(numpy.fabs(o_vzs - op_vzs)) < tol, (
            f"Integrating an orbit in a linearly-accelerating, acceleratingly-rotating frame with constant acceleration does not agree with the equivalent orbit in the inertial frame for method {method}"
        )

    check_orbit(method="odeint", tol=1e-5)
    check_orbit(method="dop853", tol=1e-9)
    check_orbit(
        method="dop853_c", tol=1e-5
    )  # Lower tol, because diff integrators for inertial and non-inertial, bc wrapper not implemented in C
    return None


def test_python_vs_c_arbitraryaxisrotation():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(
        zvec=[1.0, 0.0, 1.0],
        omega=1.3,
        py_method="dop853",
        c_method="dop853_c",
        tol=1e-9,
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False, Omega=numpy.array(derive_noninert_omega(omega, rot=rot))
            ),
            method=py_method,
        )
        # In C
        op = o()
        op.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False, Omega=numpy.array(derive_noninert_omega(omega, rot=rot))
            ),
            method=c_method,
        )
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


def test_python_vs_c_arbitraryaxisrotation_omegadot():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(
        zvec=[1.0, 0.0, 1.0],
        omega=1.3,
        omegadot=0.03,
        py_method="dop853",
        c_method="dop853_c",
        tol=1e-9,
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=numpy.array(derive_noninert_omega(omega, rot=rot)),
                Omegadot=numpy.array(derive_noninert_omega(omega, rot=rot))
                * omegadot
                / omega,
            ),
            method=py_method,
        )
        # In C
        op = o()
        op.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=numpy.array(derive_noninert_omega(omega, rot=rot)),
                Omegadot=numpy.array(derive_noninert_omega(omega, rot=rot))
                * omegadot
                / omega,
            ),
            method=c_method,
        )
        # Compare
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit(tol=1e-8)
    return None


def test_python_vs_c_arbitraryaxisrotation_funcomega():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp

    def check_orbit(
        zvec=[1.0, 0.0, 1.0],
        omega=1.3,
        omegadot=0.03,
        omegadotdot=0.01,
        py_method="dop853",
        c_method="dop853_c",
        tol=1e-9,
    ):
        # Set up the rotating frame's rotation matrix
        rotpot = potential.RotateAndTiltWrapperPotential(pot=diskpot, zvec=zvec)
        rot = rotpot._rot
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        Omega = numpy.array(derive_noninert_omega(omega, rot=rot))
        Omegadot = numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadot / omega
        Omegadotdot = (
            numpy.array(derive_noninert_omega(omega, rot=rot)) * omegadotdot / omega
        )
        o.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=[
                    lambda t: (
                        Omega[0] + Omegadot[0] * t + Omegadotdot[0] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[1] + Omegadot[1] * t + Omegadotdot[1] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[2] + Omegadot[2] * t + Omegadotdot[2] * t**2.0 / 2.0
                    ),
                ],
                Omegadot=[
                    lambda t: Omegadot[0] + Omegadotdot[0] * t,
                    lambda t: Omegadot[1] + Omegadotdot[1] * t,
                    lambda t: Omegadot[2] + Omegadotdot[2] * t,
                ],
            ),
            method=py_method,
        )
        # In C
        op = o()
        op.integrate(
            ts,
            diskpot
            + potential.NonInertialFrameForce(
                cinterp=False,
                Omega=[
                    lambda t: (
                        Omega[0] + Omegadot[0] * t + Omegadotdot[0] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[1] + Omegadot[1] * t + Omegadotdot[1] * t**2.0 / 2.0
                    ),
                    lambda t: (
                        Omega[2] + Omegadot[2] * t + Omegadotdot[2] * t**2.0 / 2.0
                    ),
                ],
                Omegadot=[
                    lambda t: Omegadot[0] + Omegadotdot[0] * t,
                    lambda t: Omegadot[1] + Omegadotdot[1] * t,
                    lambda t: Omegadot[2] + Omegadotdot[2] * t,
                ],
            ),
            method=c_method,
        )
        # Compare
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit(tol=3e-8)
    return None


def test_python_vs_c_linacc_changingacc_xyz():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    ax, ay, az = (
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    )
    inta = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    framepot = potential.NonInertialFrameForce(cinterp=False, a0=[ax, ay, az])

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-9):
        o = Orbit()
        o.turn_physical_off()
        # In Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        # Compare
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit(tol=1e-10)
    return None


def test_python_vs_c_linacc_changingacc_xyz_accellsrframe_scalaromegaz():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.02
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega, Omegadot=omegadot
    )

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-9):
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


def test_python_vs_c_linacc_changingacc_xyz_accellsrframe_scalaromegaz_2d():
    # Integrate an orbit in both Python and C to check that they match in 2D
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.02
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega, Omegadot=omegadot
    )

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-8):
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit().toPlanar()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


def test_python_vs_c_linacc_changingacc_xyz_accellsrframe_vecomegaz():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.02
    framepot = potential.NonInertialFrameForce(
        cinterp=False,
        x0=x0,
        v0=v0,
        a0=a0,
        Omega=numpy.array([0.0, 0.0, omega]),
        Omegadot=numpy.array([0.0, 0.0, omegadot]),
    )

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-9):
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


def test_python_vs_c_linacc_changingacc_xyz_accellsrframe_scalarfuncomegaz():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.1
    omegadotdot = 0.01
    omega_func = lambda t: omega + omegadot * t + omegadotdot * t**2.0 / 2.0
    omegadot_func = lambda t: omegadot + omegadotdot * t
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega_func, Omegadot=omegadot_func
    )

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-8):
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


def test_python_vs_c_linacc_changingacc_xyz_accellsrframe_vecomegaz():
    # Integrate an orbit in both Python and C to check that they match
    # We don't need to known the true answer here
    lp = potential.MiyamotoNagaiPotential(normalize=1.0, a=1.0, b=0.2)
    dp = potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03)
    diskpot = lp + dp
    x0 = [
        lambda t: -0.03 * t**2.0 / 2.0 - 0.03 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.04 * t**2.0 / 2.0 + 0.08 * t**3.0 / 6.0 / 20.0,
        lambda t: 0.02 * t**2.0 / 2.0 + 0.03 * t**3.0 / 6.0 / 20.0,
    ]
    v0 = [
        lambda t: -0.03 * t - 0.03 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.04 * t + 0.08 * t**2.0 / 2.0 / 20.0,
        lambda t: 0.02 * t + 0.03 * t**2.0 / 2.0 / 20.0,
    ]
    a0 = [
        lambda t: -0.03 - 0.03 * t / 20.0,
        lambda t: 0.04 + 0.08 * t / 20.0,
        lambda t: 0.02 + 0.03 * t / 20.0,
    ]
    omega = lp.omegac(1.0)
    omegadot = 0.1
    omegadotdot = 0.01
    omega_func = [
        lambda t: 0.0,
        lambda t: 0.0,
        lambda t: omega + omegadot * t + omegadotdot * t**2.0 / 2.0,
    ]
    omegadot_func = [lambda t: 0.0, lambda t: 0.0, lambda t: omegadot + omegadotdot * t]
    framepot = potential.NonInertialFrameForce(
        cinterp=False, x0=x0, v0=v0, a0=a0, Omega=omega_func, Omegadot=omegadot_func
    )

    def check_orbit(py_method="dop853", c_method="dop853_c", tol=1e-8):
        # Now integrate an orbit in the rotating frame in Python
        o = Orbit()
        o.turn_physical_off()
        # Rotating frame in Python
        ts = numpy.linspace(0.0, 20.0, 1001)
        o.integrate(ts, diskpot + framepot, method=py_method)
        # In C
        op = o()
        op.integrate(ts, diskpot + framepot, method=c_method)
        assert numpy.amax(numpy.fabs(o.x(ts) - op.x(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.y(ts) - op.y(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.z(ts) - op.z(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vx(ts) - op.vx(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vy(ts) - op.vy(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        assert numpy.amax(numpy.fabs(o.vz(ts) - op.vz(ts))) < tol, (
            f"Integrating an orbit in a rotating frame in Python does not agree with integrating the same orbit in C; using methods {py_method} and {c_method}"
        )
        return None

    check_orbit()
    return None


from galpy.potential.Potential import (
    _evaluatephitorques,
    _evaluateRforces,
    _evaluatezforces,
)

# Utility wrappers and other functions
from galpy.potential.WrapperPotential import parentWrapperPotential


class AcceleratingPotentialWrapperPotential(parentWrapperPotential):
    def __init__(
        self,
        amp=1.0,
        pot=None,
        x0=[lambda t: 0.0, lambda t: 0.0, lambda t: 0.0],
        omegaz=None,
        omegazdot=None,
        omegazdotdot=None,
        ro=None,
        vo=None,
    ):
        # x0 = accelerated origin
        self._x0 = x0
        # we also allow for rotation around the z axis:
        # Omega(t) = omegaz + omegazdot * t + omegazdotdot * t^2 / 2
        self._omegaz = omegaz
        self._omegazdot = omegazdot
        self._omegazdotdot = omegazdotdot

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return numpy.cos(phi) * Fxyz[0] + numpy.sin(phi) * Fxyz[1]

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return R * (-numpy.sin(phi) * Fxyz[0] + numpy.cos(phi) * Fxyz[1])

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return self._force_xyz(R, z, phi=phi, t=t)[2]

    def _force_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular forces in the transformed frame"""
        x, y, _ = coords.cyl_to_rect(R, phi, z)
        xp = x + self._x0[0](t)
        yp = y + self._x0[1](t)
        zp = z + self._x0[2](t)
        Rp, phip, zp = coords.rect_to_cyl(xp, yp, zp)
        if not self._omegaz is None:
            phip += self._omegaz * t
            if not self._omegazdot is None:
                phip += self._omegazdot * t**2.0 / 2.0
            if not self._omegazdotdot is None:
                phip += self._omegazdotdot * t**3.0 / 6.0
        Rforcep = _evaluateRforces(self._pot, Rp, zp, phi=phip, t=t)
        phitorquep = _evaluatephitorques(self._pot, Rp, zp, phi=phip, t=t)
        zforcep = _evaluatezforces(self._pot, Rp, zp, phi=phip, t=t)
        xforcep = numpy.cos(phip) * Rforcep - numpy.sin(phip) * phitorquep / Rp
        yforcep = numpy.sin(phip) * Rforcep + numpy.cos(phip) * phitorquep / Rp
        if not self._omegaz is None:
            rotphi = self._omegaz * t
            if not self._omegazdot is None:
                rotphi += self._omegazdot * t**2.0 / 2.0
            if not self._omegazdotdot is None:
                rotphi += self._omegazdotdot * t**3.0 / 6.0
            return numpy.dot(
                numpy.array(
                    [
                        [numpy.cos(rotphi), numpy.sin(rotphi), 0.0],
                        [-numpy.sin(rotphi), numpy.cos(rotphi), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                numpy.array([xforcep, yforcep, zforcep]),
            )
        else:
            return numpy.array([xforcep, yforcep, zforcep])


# Functions and wrappers for rotation around an arbitrary axis
# Rotation happens around an axis that is rotated by rot
# So transformation from rotating to inertial is
# rot.T x omega-rotation x rot
def rotate_and_omega(
    R,
    z,
    phi=0.0,
    t=0.0,
    rot=None,
    omega=None,
    omegadot=None,
    omegadotdot=None,
    rect=False,
):
    # From the rotating frame to the inertial frame
    if rect:
        x, y, z = R, z, phi
    else:
        x, y, z = coords.cyl_to_rect(R, phi, z)
    xyzp = numpy.dot(rot, numpy.array([x, y, z]))
    Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
    phip += omega * t
    if not omegadot is None:
        phip += omegadot * t**2.0 / 2.0
    if not omegadotdot is None:
        phip += omegadotdot * t**3.0 / 6.0
    xp, yp, zp = coords.cyl_to_rect(Rp, phip, zp)
    xyz = numpy.dot(rot.T, numpy.array([xp, yp, zp]))
    if rect:
        R, phi, z = xyz[0], xyz[1], xyz[2]
    else:
        R, phi, z = coords.rect_to_cyl(xyz[0], xyz[1], xyz[2])
    return R, phi, z


def rotate_and_omega_vec(
    vR,
    vT,
    vz,
    R,
    z,
    phi=0.0,
    t=0.0,
    rot=None,
    omega=None,
    omegadot=None,
    omegadotdot=None,
):
    # From the rotating frame to the inertial frame, for vectors
    x, y, z = coords.cyl_to_rect(R, phi, z)
    vx, vy, vz = coords.cyl_to_rect_vec(vR, vT, vz, phi=phi)
    xyzp = numpy.dot(rot, numpy.array([x, y, z]))
    Rp, phip, zp = coords.rect_to_cyl(xyzp[0], xyzp[1], xyzp[2])
    vxyzp = numpy.dot(rot, numpy.array([vx, vy, vz]))
    vRp, vTp, vzp = coords.rect_to_cyl_vec(
        vxyzp[0], vxyzp[1], vxyzp[2], xyzp[0], xyzp[1], xyzp[2]
    )
    phip += omega * t
    vTp += omega * Rp
    if not omegadot is None:
        phip += omegadot * t**2.0 / 2.0
        vTp += omegadot * t * Rp
    if not omegadotdot is None:
        phip += omegadotdot * t**3.0 / 6.0
        vTp += omegadotdot * t**2.0 / 2.0 * Rp
    xp, yp, zp = coords.cyl_to_rect(Rp, phip, zp)
    vxp, vyp, vzp = coords.cyl_to_rect_vec(vRp, vTp, vzp, phi=phip)
    xyz = numpy.dot(rot.T, numpy.array([xp, yp, zp]))
    vxyz = numpy.dot(rot.T, numpy.array([vxp, vyp, vzp]))
    vR, vT, vz = coords.rect_to_cyl_vec(
        vxyz[0], vxyz[1], vxyz[2], xyz[0], xyz[1], xyz[2]
    )
    return vR, vT, vz


def derive_noninert_omega(
    omega, rot=None, x=1.0, y=2.0, z=3.0, x2=-1.0, y2=2.0, z2=-5.0, t=0.0
):
    # Numerically compute Omega of the non-inertial frame
    # Need to use the rotation of two arbitrary (non-parallel) vectors
    # To fully describe the rotation
    # Then can solve for it
    # See https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Conversion_formulae_for_derivatives
    # Let's begin
    # Compute d r / d t, numerically
    eps = 1e-6
    r11 = rotate_and_omega(x, y, phi=z, t=t, rot=rot, omega=omega, rect=True)
    r12 = rotate_and_omega(x, y, phi=z, t=t + eps, rot=rot, omega=omega, rect=True)
    dr1dt = (numpy.array(r12) - numpy.array(r11)) / eps
    r21 = rotate_and_omega(x2, y2, phi=z2, t=t, rot=rot, omega=omega, rect=True)
    r22 = rotate_and_omega(x2, y2, phi=z2, t=t + eps, rot=rot, omega=omega, rect=True)
    dr2dt = (numpy.array(r22) - numpy.array(r21)) / eps
    # Solve
    # dxdt= -omegaz * y + omegay * z
    # dydt=  omegaz * x - omegax * z
    # dzdt= -omegay * x + omegax * y
    # with the two given points. Use
    # dy1dt= omegaz * x1 - omegax * z1
    # dy2dt= omegaz * x2 - omegax * z2
    # dy1dt * x2 - dy2dt * x1 = -omegax * ( z1 * x2 - z2 * x1)
    # and
    # dx1dt= -omegaz * y1 + omegay * z1
    # dx2dt= -omegaz * y2 + omegay * z2
    # dx1dt * y2 - dx2dt * y1 =  omegay * ( z1 * y2 - z2 * y1 )
    # dx1dt * z2 - dx2dt * z1 = -omegaz * ( y1 * z2 - y2 * z1 )
    omegax = -(dr1dt[1] * r21[0] - dr2dt[1] * r11[0]) / (
        r11[2] * r21[0] - r21[2] * r11[0]
    )
    omegay = (dr1dt[0] * r21[1] - dr2dt[0] * r11[1]) / (
        r11[2] * r21[1] - r21[2] * r11[1]
    )
    omegaz = -(dr1dt[0] * r21[2] - dr2dt[0] * r11[2]) / (
        r11[1] * r21[2] - r21[1] * r11[2]
    )
    return (omegax, omegay, omegaz)


class RotatingPotentialWrapperPotential(parentWrapperPotential):
    def __init__(
        self,
        amp=1.0,
        pot=None,
        rot=None,
        omega=None,
        omegadot=None,
        omegadotdot=None,
        ro=None,
        vo=None,
    ):
        # Frame rotates as rot.T x omega-rotation x rot, see transformations above
        self._rot = rot
        self._omega = omega
        self._omegadot = omegadot
        self._omegadotdot = omegadotdot

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return numpy.cos(phi) * Fxyz[0] + numpy.sin(phi) * Fxyz[1]

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        Fxyz = self._force_xyz(R, z, phi=phi, t=t)
        return R * (-numpy.sin(phi) * Fxyz[0] + numpy.cos(phi) * Fxyz[1])

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return self._force_xyz(R, z, phi=phi, t=t)[2]

    def _force_xyz(self, R, z, phi=0.0, t=0.0):
        """Get the rectangular forces in the transformed frame"""
        Rp, phip, zp = rotate_and_omega(
            R,
            z,
            phi=phi,
            t=t,
            rot=self._rot,
            omega=self._omega,
            omegadot=self._omegadot,
            omegadotdot=self._omegadotdot,
        )
        Rforcep = _evaluateRforces(self._pot, Rp, zp, phi=phip, t=t)
        phitorquep = _evaluatephitorques(self._pot, Rp, zp, phi=phip, t=t)
        zforcep = _evaluatezforces(self._pot, Rp, zp, phi=phip, t=t)
        xforcep = numpy.cos(phip) * Rforcep - numpy.sin(phip) * phitorquep / Rp
        yforcep = numpy.sin(phip) * Rforcep + numpy.cos(phip) * phitorquep / Rp
        # Now figure out the inverse rotation matrix to rotate the forces
        # The way this is written, we effectively compute the transpose of the
        # rotation matrix, which is its inverse
        inv_rot = numpy.array(
            [
                list(
                    rotate_and_omega(
                        1.0,
                        0.0,
                        phi=0.0,
                        t=t,
                        rot=self._rot,
                        omega=self._omega,
                        omegadot=self._omegadot,
                        omegadotdot=self._omegadotdot,
                        rect=True,
                    )
                ),
                list(
                    rotate_and_omega(
                        0.0,
                        1.0,
                        phi=0.0,
                        t=t,
                        rot=self._rot,
                        omega=self._omega,
                        omegadot=self._omegadot,
                        omegadotdot=self._omegadotdot,
                        rect=True,
                    )
                ),
                list(
                    rotate_and_omega(
                        0.0,
                        0.0,
                        phi=1.0,
                        t=t,
                        rot=self._rot,
                        omega=self._omega,
                        omegadot=self._omegadot,
                        omegadotdot=self._omegadotdot,
                        rect=True,
                    )
                ),
            ]
        )
        return numpy.dot(inv_rot, numpy.array([xforcep, yforcep, zforcep]))


# ----------------------------------------------------------------------------
# Tests of the cinterp=True option: on-the-fly C cubic-spline interpolation of
# the time-dependent inputs (a0, x0, v0, Omega) over the integration time
# range, with Omegadot obtained as the spline derivative of Omega (C pot_type
# 45). These check agreement with the exact cinterp=False tfunc integration
# (pot_type 39) and exercise the 2D/3D, Omegadot-derivative, multi-orbit,
# wrapper, SOS, and constant-input code paths.
# ----------------------------------------------------------------------------


def _cinterp_TvsF_maxdiff(pot_builder, ic, ts, fns, method="dop853_c"):
    # Integrate the same setup with cinterp False and True and return the max
    # absolute difference in the requested coordinate functions. The only
    # difference between the two is how the time-dependent inputs are evaluated
    # in C (spline vs function callback), so the difference is the interpolation
    # error.
    o_false = Orbit(ic)
    o_false.turn_physical_off()
    o_false.integrate(ts, pot_builder(False), method=method)
    o_true = Orbit(ic)
    o_true.turn_physical_off()
    o_true.integrate(ts, pot_builder(True), method=method)
    return max(
        numpy.amax(numpy.fabs(getattr(o_false, f)(ts) - getattr(o_true, f)(ts)))
        for f in fns
    )


def test_cinterp_func_a0_3d():
    # cinterp=True agrees with cinterp=False for a 3D smooth function a0
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.02 * numpy.cos(0.7 * t),
        lambda t: 0.015 * numpy.sin(0.5 * t),
        lambda t: 0.01 * numpy.cos(0.3 * t),
    ]
    pot_builder = lambda cinterp: (
        lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z", "vx", "vy", "vz"])
    assert md < 1e-8, (
        f"cinterp=True does not agree with cinterp=False for a 3D function a0 (max diff {md})"
    )
    return None


def test_cinterp_func_a0_2d():
    # cinterp=True agrees with cinterp=False for a 2D (planar) smooth function a0
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.02 * numpy.cos(0.7 * t),
        lambda t: 0.015 * numpy.sin(0.5 * t),
        lambda t: 0.0,
    ]
    pot_builder = lambda cinterp: (
        lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.2]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "vx", "vy"])
    assert md < 1e-8, (
        f"cinterp=True does not agree with cinterp=False for a 2D function a0 (max diff {md})"
    )
    return None


def test_cinterp_linear_a0_exact():
    # For a linear (degree<=1) a0 the natural cubic spline is exact, so cinterp
    # should agree with the exact integration to ~machine precision.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.02 + 0.003 * t,
        lambda t: -0.01 + 0.002 * t,
        lambda t: 0.005 - 0.001 * t,
    ]
    pot_builder = lambda cinterp: (
        lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-9, (
        f"cinterp=True does not agree with the exact integration for a linear a0 (max diff {md})"
    )
    return None


def test_cinterp_funcomegaz():
    # cinterp=True agrees with cinterp=False for a scalar-Omega_z function (and
    # its Omegadot), in both 3D and 2D. Exercises the Omegadot spline-derivative
    # path for the omegaz_only case.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega_fun = lambda t: 1.0 + 0.05 * numpy.sin(0.2 * t)
    omegadot_fun = lambda t: 0.05 * 0.2 * numpy.cos(0.2 * t)
    pot_builder = lambda cinterp: (
        lp
        + potential.NonInertialFrameForce(
            Omega=omega_fun, Omegadot=omegadot_fun, cinterp=cinterp
        )
    )
    ts = numpy.linspace(0.0, 10.0, 201)
    md3 = _cinterp_TvsF_maxdiff(
        pot_builder, [1.0, 0.1, 1.1, 0.05, 0.1, 0.3], ts, ["x", "y", "z"]
    )
    md2 = _cinterp_TvsF_maxdiff(pot_builder, [1.0, 0.1, 1.1, 0.2], ts, ["x", "y"])
    assert md3 < 1e-8, f"cinterp Omega_z(t) 3D max diff {md3}"
    assert md2 < 1e-8, f"cinterp Omega_z(t) 2D max diff {md2}"
    return None


def test_cinterp_vecfuncomega():
    # cinterp=True agrees with cinterp=False for a 3D vector-Omega function (and
    # its Omegadot). Exercises the Omegadot spline-derivative path for the
    # full-3D-Omega case.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = [
        lambda t: 0.02 * numpy.cos(0.1 * t),
        lambda t: 0.03 + 0.0 * t,
        lambda t: 1.0 + 0.04 * numpy.sin(0.15 * t),
    ]
    omegadot = [
        lambda t: -0.02 * 0.1 * numpy.sin(0.1 * t),
        lambda t: 0.0 * t,
        lambda t: 0.04 * 0.15 * numpy.cos(0.15 * t),
    ]
    pot_builder = lambda cinterp: (
        lp
        + potential.NonInertialFrameForce(
            Omega=omega, Omegadot=omegadot, cinterp=cinterp
        )
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-8, f"cinterp vector Omega(t) 3D max diff {md}"
    return None


def test_cinterp_constfreq_lin():
    # cinterp=True with a constant Omega and no Omegadot (const_freq=True), but a
    # moving/accelerating origin (function a0, x0, v0): exercises the spline
    # a0/x0/v0 path with the const_freq branch (Euler term omitted). A function
    # Omega without a matching Omegadot is not a supported configuration (it has
    # no derivative to use), so const_freq is tested here with a constant Omega.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = 1.0  # constant scalar Omega_z -> not interpolated, const_freq=True
    x0 = [
        lambda t: 0.1 * numpy.sin(0.2 * t),
        lambda t: 0.05 * numpy.cos(0.15 * t),
        lambda t: 0.02 * t,
    ]
    v0 = [
        lambda t: 0.1 * 0.2 * numpy.cos(0.2 * t),
        lambda t: -0.05 * 0.15 * numpy.sin(0.15 * t),
        lambda t: 0.02 + 0.0 * t,
    ]
    a0 = [
        lambda t: 0.01 * numpy.cos(t),
        lambda t: 0.01 * numpy.sin(t),
        lambda t: 0.005 + 0.0 * t,
    ]
    pot_builder = lambda cinterp: (
        lp
        + potential.NonInertialFrameForce(
            Omega=omega, x0=x0, v0=v0, a0=a0, cinterp=cinterp
        )
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-8, f"cinterp const_freq with moving origin max diff {md}"
    return None


def test_cinterp_combined_rotlin():
    # cinterp=True agrees with cinterp=False for the full combined case: a 3D
    # function Omega (+Omegadot) AND an accelerating origin (a0, x0, v0). This
    # exercises the a0/x0/v0 splines and the Omegadot spline-derivative path
    # together.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    omega = [
        lambda t: 0.02 * numpy.cos(0.1 * t),
        lambda t: 0.03 + 0.0 * t,
        lambda t: 1.0 + 0.04 * t,
    ]
    omegadot = [
        lambda t: -0.02 * 0.1 * numpy.sin(0.1 * t),
        lambda t: 0.0 * t,
        lambda t: 0.04 + 0.0 * t,
    ]
    x0 = [
        lambda t: 0.1 * numpy.sin(0.2 * t),
        lambda t: 0.05 * numpy.cos(0.15 * t),
        lambda t: 0.02 * t,
    ]
    v0 = [
        lambda t: 0.1 * 0.2 * numpy.cos(0.2 * t),
        lambda t: -0.05 * 0.15 * numpy.sin(0.15 * t),
        lambda t: 0.02 + 0.0 * t,
    ]
    a0 = [
        lambda t: 0.01 * numpy.cos(t),
        lambda t: 0.01 * numpy.sin(t),
        lambda t: 0.005 + 0.0 * t,
    ]
    pot_builder = lambda cinterp: (
        lp
        + potential.NonInertialFrameForce(
            Omega=omega, Omegadot=omegadot, x0=x0, v0=v0, a0=a0, cinterp=cinterp
        )
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-8, f"cinterp combined rot+lin max diff {md}"
    return None


def test_cinterp_multiple_orbits_indiv_t():
    # cinterp=True works for multiple orbits integrated with per-orbit (2D) time
    # arrays: the interpolation table is built over the overall [tmin,tmax].
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.02 * numpy.cos(0.5 * t),
        lambda t: 0.01 * numpy.sin(0.4 * t),
        lambda t: 0.005 * numpy.cos(0.2 * t),
    ]
    ics = [
        [1.0, 0.1, 1.1, 0.05, 0.1, 0.3],
        [1.2, 0.0, 0.9, 0.1, 0.05, 1.0],
    ]
    # Per-orbit time arrays with different ranges -> table over overall min/max
    ts = numpy.array([numpy.linspace(0.0, 8.0, 151), numpy.linspace(0.0, 12.0, 151)])

    def integ(cinterp):
        o = Orbit(ics)
        o.turn_physical_off()
        o.integrate(
            ts,
            lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp),
            method="dop853_c",
        )
        return o

    o_false = integ(False)
    o_true = integ(True)
    md = numpy.amax(numpy.fabs(o_false.getOrbit() - o_true.getOrbit()))
    assert md < 1e-8, (
        f"cinterp=True does not agree with cinterp=False for per-orbit time arrays (max diff {md})"
    )
    return None


def test_cinterp_wrapper_and_movingobject_combo():
    # cinterp=True works and agrees with cinterp=False when the
    # NonInertialFrameForce is combined with a wrapper potential and another
    # spline-using force (MovingObjectPotential) in the same potential, stressing
    # the pot_args / pot_tfuncs / spline-block ordering through the C parser.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    ts = numpy.linspace(0.0, 10.0, 201)
    # A moving object: its own integrated orbit -> MovingObjectPotential (a
    # spline force, pot_type -6)
    o_obj = Orbit([1.5, 0.0, 0.9, 0.0, 0.05, 0.0])
    o_obj.turn_physical_off()
    o_obj.integrate(ts, lp)
    mop = potential.MovingObjectPotential(
        o_obj, pot=potential.HernquistPotential(amp=0.02, a=0.3)
    )
    # A time-dependent amplitude wrapper around a bar (parsed as a wrapper)
    dsw = potential.DehnenSmoothWrapperPotential(
        pot=potential.DehnenBarPotential(omegab=1.8, rb=0.5, Af=0.03),
        tform=-5.0,
        tsteady=2.0,
    )
    a0 = [
        lambda t: 0.01 * numpy.cos(0.5 * t),
        lambda t: 0.008 * numpy.sin(0.3 * t),
        lambda t: 0.005 + 0.0 * t,
    ]
    pot_builder = lambda cinterp: (
        lp + dsw + mop + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-7, (
        f"cinterp=True does not agree with cinterp=False in a wrapper+MovingObject combo (max diff {md})"
    )
    return None


def test_cinterp_n_resolution():
    # A finer cinterp_n gives smaller interpolation error than a coarse one for
    # a non-polynomial input.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.05 * numpy.cos(2.0 * t),
        lambda t: 0.05 * numpy.sin(1.7 * t),
        lambda t: 0.03 * numpy.cos(1.3 * t),
    ]
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    coarse = _cinterp_TvsF_maxdiff(
        lambda cinterp: (
            lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp, cinterp_n=50)
        ),
        ic,
        ts,
        ["x", "y", "z"],
    )
    fine = _cinterp_TvsF_maxdiff(
        lambda cinterp: (
            lp
            + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp, cinterp_n=20000)
        ),
        ic,
        ts,
        ["x", "y", "z"],
    )
    assert fine < coarse, (
        f"Finer cinterp_n did not reduce the interpolation error (coarse {coarse}, fine {fine})"
    )
    assert fine < 1e-8, f"cinterp with fine cinterp_n max diff {fine}"
    return None


def test_cinterp_sos_raises():
    # Surface-of-section integration with cinterp=True (and genuine function
    # inputs) is rejected with a clear error, since the time range is not known
    # in advance; cinterp=False works.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [
        lambda t: 0.02 * numpy.cos(t),
        lambda t: 0.01 * numpy.sin(t),
        lambda t: 0.0,
    ]
    o = Orbit([1.0, 0.1, 1.1, 0.05, 0.1, 0.3])
    o.turn_physical_off()
    with pytest.raises(NotImplementedError):
        o.SOS(lp + potential.NonInertialFrameForce(a0=a0, cinterp=True))
    # cinterp=False must work
    o2 = Orbit([1.0, 0.1, 1.1, 0.05, 0.1, 0.3])
    o2.turn_physical_off()
    o2.SOS(lp + potential.NonInertialFrameForce(a0=a0, cinterp=False))
    return None


def test_cinterp_constant_a0_noop():
    # A constant a0 with cinterp=True is a no-op (nothing to interpolate -> the
    # tfunc path, pot_type 39), so it is byte-for-byte identical to cinterp=False
    # and is allowed for SOS integration.
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    a0 = [0.02, -0.01, 0.005]
    pot_builder = lambda cinterp: (
        lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-12, (
        f"Constant-a0 cinterp=True is not identical to cinterp=False (max diff {md})"
    )
    # SOS with a constant a0 and cinterp=True must NOT raise (no interpolation)
    o = Orbit([1.0, 0.1, 1.1, 0.05, 0.1, 0.3])
    o.turn_physical_off()
    o.SOS(lp + potential.NonInertialFrameForce(a0=a0, cinterp=True))
    return None


def test_cinterp_scalar_only_and_constant_funcs():
    # The cinterp table is built by evaluating the supplied functions on a grid,
    # vectorized when possible. This exercises the scalar-loop fallback (a
    # scalar-only callable that raises on array input) and the scalar-broadcast
    # path (a callable returning a bare constant), alongside a vectorizable one.
    import math

    lp = potential.LogarithmicHaloPotential(normalize=1.0)

    def a0x_scalar_only(t):  # math.cos raises on array input -> scalar-loop path
        return 0.02 * math.cos(0.5 * t)

    a0 = [
        a0x_scalar_only,
        lambda t: 0.01,  # bare constant -> scalar-broadcast path
        lambda t: 0.005 * numpy.sin(0.3 * t),  # vectorizable
    ]
    pot_builder = lambda cinterp: (
        lp + potential.NonInertialFrameForce(a0=a0, cinterp=cinterp)
    )
    ic = [1.0, 0.1, 1.1, 0.05, 0.1, 0.3]
    ts = numpy.linspace(0.0, 10.0, 201)
    md = _cinterp_TvsF_maxdiff(pot_builder, ic, ts, ["x", "y", "z"])
    assert md < 1e-8, (
        f"cinterp with scalar-only/constant a0 functions does not agree with cinterp=False (max diff {md})"
    )
    return None


def test_noninertialframeforce_requires_x0v0_when_rot_and_lin():
    # When the frame both rotates (Omega) and accelerates (a0), x0 and v0 are
    # required (they describe the moving origin); omitting them must raise a
    # clear error at construction rather than a confusing TypeError later during
    # integration.
    with pytest.raises(ValueError):
        potential.NonInertialFrameForce(a0=[0.0, 0.0, -1.0], Omega=1.0)
    with pytest.raises(ValueError):
        potential.NonInertialFrameForce(
            a0=[lambda t: 0.01, lambda t: 0.0, lambda t: 0.0],
            Omega=lambda t: 1.0,
        )
    return None
