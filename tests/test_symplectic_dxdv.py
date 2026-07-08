##############################################################################
# test_symplectic_dxdv.py: tests of the phase-space-volume (dxdv) variational
#   integration for the C SYMPLECTIC integrators (leapfrog_c/symplec4_c/
#   symplec6_c). These carry a phase-space deviation alongside the base orbit
#   through the exact, closed-form drift/kick tangent maps of the discrete
#   symplectic step (drift M_D=[[I,hI],[0,I]], kick M_K=[[I,0],[hK,I]] with K
#   the conservative Cartesian Hessian), rather than integrating the RK
#   variational RHS. Because each elementary map is exactly symplectic for a
#   conservative system, the propagated 6x6 STM M is symplectic to machine
#   precision (a sharper property than the RK dxdv path). Only the 6D (3D)
#   orbit case is wired; the planar (4D) symplectic dxdv path is not.
##############################################################################
import numpy
import pytest

SYMPLECTIC = ["leapfrog_c", "symplec4_c", "symplec6_c"]
ORDER = {"leapfrog_c": 2, "symplec4_c": 4, "symplec6_c": 6}

# Omega in (x,y,z,vx,vy,vz) order for the symplecticity check M^T Omega M = Omega
_Omega = numpy.zeros((6, 6))
_Omega[:3, 3:] = numpy.eye(3)
_Omega[3:, :3] = -numpy.eye(3)
_CANON = numpy.eye(6)

# A generic, fully 3D initial condition (R,vR,vT,z,vz,phi)
_IC = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]


def _rect(o, ts):
    """Stack the integrated 6D orbit in rectangular phase space."""
    return numpy.array([o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)]).T


def _build_M(ic, times, pot, method, dt=None, rtol=1e-12, atol=1e-12):
    """The 6x6 STM at the final time: integrate the six canonical basis
    deviation vectors (rectIn=rectOut=True) and stack them as columns."""
    from galpy.orbit import Orbit

    cols = []
    for ii in range(6):
        o = Orbit(ic)
        o.integrate_dxdv(
            _CANON[ii],
            times,
            pot,
            method=method,
            dt=dt,
            rectIn=True,
            rectOut=True,
            rtol=rtol,
            atol=atol,
        )
        cols.append(o.getOrbit_dxdv()[-1, :])
    return numpy.array(cols).T


def _fd_of_flow_M(ic, times, pot, method, dt, eps=1e-6):
    """The 6x6 STM by central finite-difference of the base flow: integrate the
    unperturbed base orbit and orbits perturbed by +/- eps along each Cartesian
    phase-space axis and difference the endpoints."""
    from galpy.orbit import Orbit
    from galpy.util import coords

    obase = Orbit(ic)
    obase.integrate(times, pot, method=method, dt=dt)
    y0 = _rect(obase, times[0])
    Mfd = numpy.empty((6, 6))
    for ii in range(6):
        cols = []
        for sgn in (+1.0, -1.0):
            y = y0.copy()
            y[ii] += sgn * eps
            Rp, phip, Zp = coords.rect_to_cyl(y[0], y[1], y[2])
            vRp, vTp, vzp = coords.rect_to_cyl_vec(y[3], y[4], y[5], y[0], y[1], y[2])
            op = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
            op.integrate(times, pot, method=method, dt=dt)
            cols.append(_rect(op, times[-1]))
        Mfd[:, ii] = (cols[0] - cols[1]) / (2.0 * eps)
    return Mfd


# ----------------------------------------------------------------------------
def test_symplectic_dxdv_base_bit_identity():
    """The base orbit carried alongside the deviation must be BIT-IDENTICAL to a
    plain (no-dxdv) integration with the same method/dt/IC: the symplectic dxdv
    step estimates its stepsize from the base block only and steps the base via
    the same drift/kick sequence, so the deviation machinery cannot perturb the
    base integration."""
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    times = numpy.linspace(0.0, 5.0, 251)
    for method in SYMPLECTIC:
        o1 = Orbit(_IC)
        o1.integrate_dxdv(
            _CANON[0],
            times,
            pot,
            method=method,
            dt=0.01,
            rectIn=True,
            rectOut=True,
        )
        base_dxdv = numpy.asarray(o1.getOrbit())
        o2 = Orbit(_IC)
        o2.integrate(times, pot, method=method, dt=0.01)
        base_plain = numpy.asarray(o2.getOrbit())
        assert numpy.array_equal(base_dxdv, base_plain), (
            f"symplectic dxdv base orbit not bit-identical to plain integrate "
            f"for {method}: max|diff|="
            f"{numpy.amax(numpy.fabs(base_dxdv - base_plain)):g}"
        )
    return None


def test_symplectic_dxdv_liouville_symplecticity():
    """Exact per-step symplecticity of the conservative symplectic STM: the 6x6
    M must satisfy det(M)=1 and M^T Omega M = Omega to machine precision (much
    tighter than the RK dxdv path, whose STM is only approximately symplectic).
    Tested on an axisymmetric (MiyamotoNagai) and a composite (MWPotential2014)
    potential."""
    from galpy.potential import MiyamotoNagaiPotential, MWPotential2014

    times = numpy.linspace(0.0, 5.0, 251)
    pots = [
        ("MiyamotoNagai", MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)),
        ("MWPotential2014", MWPotential2014),
    ]
    for pname, pot in pots:
        for method in SYMPLECTIC:
            M = _build_M(_IC, times, pot, method, dt=0.01)
            detm1 = numpy.fabs(numpy.linalg.det(M) - 1.0)
            symperr = numpy.amax(numpy.fabs(M.T @ _Omega @ M - _Omega))
            assert detm1 < 1e-10, (
                f"|det(M)-1|={detm1:g} exceeds 1e-10 for {pname}, {method}"
            )
            assert symperr < 1e-10, (
                f"||M^T Omega M - Omega||={symperr:g} exceeds 1e-10 for "
                f"{pname}, {method}"
            )
    return None


def test_symplectic_dxdv_closed_form_harmonic():
    """3D isotropic harmonic oscillator (interior of a homogeneous sphere,
    a=-omega^2 r, constant K=-omega^2 I): the propagated STM M must match the
    analytic flow exp(t A), A=[[0,I],[-omega^2 I,0]], to the method order (2/4/6)
    and be exactly symplectic. The order-appropriate absolute tolerances alone
    distinguish the three methods; a dt-halving ratio confirms the order for the
    two lower-order methods (symplec6 reaches the roundoff floor)."""
    from scipy.linalg import expm

    from galpy.orbit import Orbit
    from galpy.potential import HomogeneousSpherePotential

    pot = HomogeneousSpherePotential(amp=1.0, R=3.0, normalize=True)
    assert pot.hasC_dxdv3d
    # omega^2 = R2deriv = z2deriv (constant, Rzderiv=0) everywhere inside R
    omega2 = pot.R2deriv(0.7, 0.2)
    assert omega2 > 0.0
    for RR, zz in [(0.7, 0.2), (1.3, -0.4)]:
        assert numpy.fabs(pot.R2deriv(RR, zz) - omega2) < 1e-14
        assert numpy.fabs(pot.z2deriv(RR, zz) - omega2) < 1e-14
        assert numpy.fabs(pot.Rzderiv(RR, zz)) < 1e-14
    A = numpy.zeros((6, 6))
    A[:3, 3:] = numpy.eye(3)
    A[3:, :3] = -omega2 * numpy.eye(3)
    T = 2.0
    # precondition: the orbit stays well inside the sphere (harmonic core)
    o = Orbit(_IC)
    times_pre = numpy.linspace(0.0, T, 51)
    o.integrate(times_pre, pot, method="dop853_c")
    r = numpy.sqrt(o.x(times_pre) ** 2 + o.y(times_pre) ** 2 + o.z(times_pre) ** 2)
    assert numpy.amax(r) < 0.9 * pot.R
    # order-appropriate absolute tolerances at dt=0.02 (a lower-order method
    # cannot reach a higher-order method's bound)
    abstol = {"leapfrog_c": 1e-3, "symplec4_c": 1e-6, "symplec6_c": 1e-9}
    for method in SYMPLECTIC:
        dt = 0.02
        tt = numpy.linspace(0.0, T, int(round(T / dt)) + 1)
        M = _build_M(_IC, tt, pot, method, dt=dt)
        Mana = expm(T * A)
        err = numpy.amax(numpy.fabs(M - Mana))
        assert err < abstol[method], (
            f"harmonic STM error {err:g} exceeds {abstol[method]:g} for {method}"
        )
        assert numpy.fabs(numpy.linalg.det(M) - 1.0) < 1e-10
    # convergence order for the two lower-order methods (symplec6 hits the
    # double-precision roundoff floor of the analytic reference, so its dt->dt/2
    # ratio is not measurable and is not asserted)
    for method in ["leapfrog_c", "symplec4_c"]:
        errs = []
        for dt in [0.02, 0.01]:
            tt = numpy.linspace(0.0, T, int(round(T / dt)) + 1)
            M = _build_M(_IC, tt, pot, method, dt=dt)
            errs.append(numpy.amax(numpy.fabs(M - expm(T * A))))
        ratio = errs[0] / errs[1]
        expected = 2.0 ** ORDER[method]
        assert 0.7 * expected < ratio < 1.3 * expected, (
            f"harmonic convergence ratio {ratio:.2f} not ~{expected:.0f} "
            f"(order {ORDER[method]}) for {method}"
        )
    return None


def test_symplectic_dxdv_kepler_fd_of_flow():
    """Kepler (nonlinear): the symplectic STM M must match the
    finite-difference-of-the-flow STM (which certifies M is the Jacobian of the
    actual discrete map, independent of the symplecticity argument) and remain
    exactly symplectic."""
    from galpy.potential import KeplerPotential

    pot = KeplerPotential(normalize=1.0)
    ic = [1.0, 0.0, 1.0, 0.1, 0.05, 0.4]
    times = numpy.linspace(0.0, 2.0, 101)
    for method in SYMPLECTIC:
        M = _build_M(ic, times, pot, method, dt=0.01)
        Mfd = _fd_of_flow_M(ic, times, pot, method, dt=0.01)
        err = numpy.amax(numpy.fabs(M - Mfd))
        assert err < 1e-6, (
            f"Kepler symplectic STM differs from FD-of-flow by {err:g} for {method}"
        )
        assert numpy.fabs(numpy.linalg.det(M) - 1.0) < 1e-10
    return None


def test_symplectic_dxdv_fd_of_flow_per_column():
    """Per-column finite-difference-of-the-flow on MiyamotoNagai: each column of
    the symplectic STM must equal the central difference of the integrated flow
    along the corresponding Cartesian phase-space axis, to the double-precision
    central-difference floor."""
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    times = numpy.linspace(0.0, 2.0, 101)
    for method in SYMPLECTIC:
        M = _build_M(_IC, times, pot, method, dt=0.01)
        Mfd = _fd_of_flow_M(_IC, times, pot, method, dt=0.01)
        err = numpy.amax(numpy.fabs(M - Mfd))
        assert err < 1e-6, (
            f"per-column FD-of-flow differs from the symplectic STM by {err:g} "
            f"for {method}"
        )
    return None


def test_symplectic_dxdv_dop853_agreement():
    """The symplectic STM must agree with the (RK) dop853_c dxdv STM: both
    converge to the same continuous STM, so at finite dt they agree to the
    looser of the two accuracies and the difference shrinks at the method order
    as dt->0 (verified for leapfrog_c/symplec4_c; symplec6_c converges below the
    dop853_c reference floor, so only its absolute agreement is checked)."""
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    times = numpy.linspace(0.0, 2.0, 101)
    Mref = _build_M(_IC, times, pot, "dop853_c", rtol=1e-13, atol=1e-13)
    # leapfrog_c/symplec4_c: check convergence order across dt
    for method in ["leapfrog_c", "symplec4_c"]:
        errs = []
        for dt in [0.02, 0.01, 0.005]:
            M = _build_M(_IC, times, pot, method, dt=dt)
            errs.append(numpy.amax(numpy.fabs(M - Mref)))
        for kk, dt in enumerate([0.02, 0.01, 0.005]):
            assert errs[kk] < 1e-2, (
                f"{method} STM differs from dop853_c by {errs[kk]:g} at dt={dt}"
            )
        expected = 2.0 ** ORDER[method]
        for ratio in (errs[0] / errs[1], errs[1] / errs[2]):
            assert 0.7 * expected < ratio < 1.3 * expected, (
                f"{method} vs dop853_c convergence ratio {ratio:.2f} not "
                f"~{expected:.0f}"
            )
    # symplec6_c: converges below the dop853_c reference floor; require tight
    # absolute agreement only
    M = _build_M(_IC, times, pot, "symplec6_c", dt=0.01)
    err = numpy.amax(numpy.fabs(M - Mref))
    assert err < 1e-9, f"symplec6_c STM differs from dop853_c by {err:g}"
    return None


def test_symplectic_dxdv_planar_rejected():
    """The C symplectic integrators are only wired for the 6D (3D) dxdv path;
    the planar (4D) symplectic dxdv path is not implemented, so requesting one
    for a planar orbit must raise (as before)."""
    from galpy.orbit import Orbit
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    times = numpy.linspace(0.0, 2.0, 51)
    for method in SYMPLECTIC:
        o = Orbit([1.0, 0.1, 1.1, 0.2])
        with pytest.raises(ValueError):
            o.integrate_dxdv(
                [1.0, 0.0, 0.0, 0.0],
                times,
                pot,
                method=method,
                rectIn=True,
                rectOut=True,
            )
    return None
