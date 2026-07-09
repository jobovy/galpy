##############################################################################
# test_linear_dxdv.py: tests of the phase-space-volume (dxdv) variational
#   integration for 1D (linear) orbits, across the RK integrators
#   (rk4_c/rk6_c/dopr54_c/dop853_c/dop853/odeint) AND the symplectic ones
#   (leapfrog_c/symplec4_c/symplec6_c). The 1D variational RHS carries the 2D
#   deviation [dx,dv] with d(dx)/dt=dv, d(dv)/dt=K*dx and K=dF/dx=-d^2Phi/dx^2
#   (linear2deriv). The RK methods integrate the 4D state [x,v,dx,dv]; the
#   symplectic ones carry the deviation through the exact drift/kick tangent
#   maps of the discrete step (K the 1x1 conservative Hessian). There is no
#   cyl<->rect transform in 1D, so the deviation is the raw [dx,dv] 2-vector.
##############################################################################
import numpy
import pytest

SYMPLECTIC = ["leapfrog_c", "symplec4_c", "symplec6_c"]
ORDER = {"leapfrog_c": 2, "symplec4_c": 4, "symplec6_c": 6}
# Fixed-step methods whose 1D dxdv base is byte-identical to plain integrate
FIXED_STEP = ["leapfrog_c", "symplec4_c", "symplec6_c", "rk4_c", "rk6_c"]
# Adaptive/step-controlled: base differs from plain by the step-control tweak
ADAPTIVE = ["dopr54_c", "dop853_c", "dop853", "odeint"]
# All RK/pure-Python (non-symplectic) methods with a 1D dxdv path
RK = ["rk4_c", "rk6_c", "dopr54_c", "dop853_c", "dop853", "odeint"]
ALL = SYMPLECTIC + RK

# Omega in (x,v) order for the symplecticity check M^T Omega M = Omega
_Omega = numpy.array([[0.0, 1.0], [-1.0, 0.0]])
_CANON = numpy.eye(2)


def _dt_for(method, dt):
    """Fixed stepsize only for the C integrators; None for pure-Python."""
    return dt if "_c" in method else None


def _harmonic_STM(omega, t):
    """Analytic 2x2 STM of the 1D harmonic oscillator a=-omega^2 x."""
    return numpy.array(
        [
            [numpy.cos(omega * t), numpy.sin(omega * t) / omega],
            [-omega * numpy.sin(omega * t), numpy.cos(omega * t)],
        ]
    )


def _build_M(ic, times, pot, method, dt=0.005, rtol=1e-12, atol=1e-12):
    """The 2x2 STM at the final time: integrate the two canonical basis
    deviation vectors and stack their [dx,dv] endpoints as columns."""
    from galpy.orbit import Orbit

    cols = []
    for ii in range(2):
        o = Orbit(ic)
        o.integrate_dxdv(
            _CANON[ii],
            times,
            pot,
            method=method,
            dt=_dt_for(method, dt),
            rtol=rtol,
            atol=atol,
        )
        cols.append(o.getOrbit_dxdv()[-1, :])
    return numpy.array(cols).T


def _fd_of_flow_M(ic, times, pot, method, dt=0.005, eps=1e-6):
    """The 2x2 STM by central finite-difference of the base flow: integrate the
    unperturbed base orbit and orbits perturbed by +/- eps along x and v, and
    difference the [x,v] endpoints."""
    from galpy.orbit import Orbit

    obase = Orbit(ic)
    obase.integrate(times, pot, method=method, dt=_dt_for(method, dt))
    y0 = numpy.array([obase.x(times[0]), obase.vx(times[0])])
    Mfd = numpy.empty((2, 2))
    for ii in range(2):
        cols = []
        for sgn in (+1.0, -1.0):
            y = y0.copy()
            y[ii] += sgn * eps
            op = Orbit([y[0], y[1]])
            op.integrate(times, pot, method=method, dt=_dt_for(method, dt))
            cols.append(numpy.array([op.x(times[-1]), op.vx(times[-1])]))
        Mfd[:, ii] = (cols[0] - cols[1]) / (2.0 * eps)
    return Mfd


# ----------------------------------------------------------------------------
def test_linear_dxdv_base_bit_identity():
    """The 1D base orbit carried alongside the deviation (getOrbit, columns 0:2)
    must match a plain (no-dxdv) integration with the same method/dt/IC. For the
    fixed-step methods (all symplectic + rk4_c/rk6_c) it is BIT-IDENTICAL (there
    is no cyl<->rect reconstruction in 1D and the base block is stepped by the
    same arithmetic); the adaptive integrators tweak their step control from the
    augmented state, so they match to a loose tolerance only."""
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, KGPotential

    times = numpy.linspace(0.0, 5.0, 251)
    cases = [
        ("KG", KGPotential(amp=1.0, K=0.3, F=0.2, D=1.5), [0.3, 0.1]),
        ("IsoDisk", IsothermalDiskPotential(amp=1.0, sigma=0.2), [0.4, -0.1]),
    ]
    for pname, pot, ic in cases:
        for method in ALL:
            o1 = Orbit(ic)
            o1.integrate_dxdv(
                _CANON[0], times, pot, method=method, dt=_dt_for(method, 0.01)
            )
            base_dxdv = numpy.asarray(o1.getOrbit())
            o2 = Orbit(ic)
            o2.integrate(times, pot, method=method, dt=_dt_for(method, 0.01))
            base_plain = numpy.asarray(o2.getOrbit())
            if method in FIXED_STEP:
                assert numpy.array_equal(base_dxdv, base_plain), (
                    f"1D dxdv base not bit-identical to plain integrate for "
                    f"{pname}, {method}: max|diff|="
                    f"{numpy.amax(numpy.fabs(base_dxdv - base_plain)):g}"
                )
            else:
                err = numpy.amax(numpy.fabs(base_dxdv - base_plain))
                assert err < 1e-6, (
                    f"1D dxdv base differs from plain integrate by {err:g} for "
                    f"{pname}, {method}"
                )
    return None


def test_linear_dxdv_closed_form_harmonic():
    """1D harmonic oscillator a=-omega^2 x (constant K=-omega^2), realized two
    ways: KGPotential with K=0 (Phi=amp F x^2, omega^2=2 amp F, exact for all x)
    and a verticalPotential of a HomogeneousSpherePotential (constant z2deriv in
    the harmonic core). The propagated 2x2 STM must match the analytic flow
    [[cos wt, sin wt/w],[-w sin wt, cos wt]] -- tightly for the RK/pure-Python
    methods, and to the method order (2/4/6) for the symplectic ones. This test
    also fixes the SIGN of K (a wrong sign would give sinh/cosh, not sin/cos)."""
    from galpy.orbit import Orbit
    from galpy.potential import (
        HomogeneousSpherePotential,
        KGPotential,
        toVerticalPotential,
    )

    T = 2.0
    # (a) KGPotential(K=0): exact harmonic, omega^2 = 2 amp F
    F = 0.5
    kg = KGPotential(amp=1.0, K=0.0, F=F, D=1.0)
    assert kg.hasC_dxdv
    # (b) verticalPotential(HomogeneousSphere): harmonic core, omega^2 = z2deriv
    hs = HomogeneousSpherePotential(amp=1.0, R=3.0, normalize=True)
    vp = toVerticalPotential(hs, R=1.0)
    assert vp.hasC_dxdv
    cases = [
        ("KG_K0", kg, numpy.sqrt(2.0 * F), [0.3, 0.0]),
        ("vp_HS", vp, numpy.sqrt(hs.z2deriv(1.0, 0.0)), [0.2, 0.0]),
    ]
    # order-appropriate absolute tolerances at dt=0.02 for the symplectic methods
    sabstol = {"leapfrog_c": 1e-3, "symplec4_c": 1e-6, "symplec6_c": 1e-9}
    for pname, pot, omega, ic in cases:
        Mana = _harmonic_STM(omega, T)
        if pname == "vp_HS":
            # precondition: orbit stays inside the sphere (harmonic core)
            o = Orbit(ic)
            tpre = numpy.linspace(0.0, T, 51)
            o.integrate(tpre, pot, method="dop853_c")
            assert numpy.amax(numpy.fabs(o.x(tpre))) < 0.9 * hs.R
        # RK / pure-Python: high-order/adaptive -> tight
        for method in RK:
            dt = 0.005
            tt = numpy.linspace(0.0, T, int(round(T / dt)) + 1)
            M = _build_M(ic, tt, pot, method, dt=dt)
            err = numpy.amax(numpy.fabs(M - Mana))
            assert err < 1e-7, (
                f"1D harmonic STM error {err:g} exceeds 1e-7 for {pname}, {method}"
            )
            assert numpy.fabs(numpy.linalg.det(M) - 1.0) < 1e-8
        # symplectic: match to the method order at dt=0.02
        for method in SYMPLECTIC:
            dt = 0.02
            tt = numpy.linspace(0.0, T, int(round(T / dt)) + 1)
            M = _build_M(ic, tt, pot, method, dt=dt)
            err = numpy.amax(numpy.fabs(M - Mana))
            assert err < sabstol[method], (
                f"1D harmonic STM error {err:g} exceeds {sabstol[method]:g} for "
                f"{pname}, {method}"
            )
            assert numpy.fabs(numpy.linalg.det(M) - 1.0) < 1e-10
    # convergence order for the two lower-order symplectic methods (KG case)
    for method in ["leapfrog_c", "symplec4_c"]:
        errs = []
        for dt in [0.02, 0.01]:
            tt = numpy.linspace(0.0, T, int(round(T / dt)) + 1)
            M = _build_M([0.3, 0.0], tt, kg, method, dt=dt)
            errs.append(
                numpy.amax(numpy.fabs(M - _harmonic_STM(numpy.sqrt(2.0 * F), T)))
            )
        ratio = errs[0] / errs[1]
        expected = 2.0 ** ORDER[method]
        assert 0.7 * expected < ratio < 1.3 * expected, (
            f"1D harmonic convergence ratio {ratio:.2f} not ~{expected:.0f} "
            f"(order {ORDER[method]}) for {method}"
        )
    return None


def test_linear_dxdv_liouville_symplecticity():
    """Liouville det(M)=1 and symplecticity M^T Omega M = Omega (Omega=[[0,1],
    [-1,0]]) of the 2x2 STM. (For a 2x2 matrix these are equivalent, since
    M^T Omega M = det(M) Omega identically; both are asserted to match the task
    spec.) Tested on nonlinear (IsothermalDisk, KGPotential) and vertical
    (verticalPotential of MiyamotoNagai) 1D potentials; symplectic methods reach
    machine precision, the RK/pure-Python methods a looser bound."""
    from galpy.potential import (
        IsothermalDiskPotential,
        KGPotential,
        MiyamotoNagaiPotential,
        toVerticalPotential,
    )

    times = numpy.linspace(0.0, 4.0, 201)
    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    cases = [
        ("IsoDisk", IsothermalDiskPotential(amp=1.0, sigma=0.2), [0.3, 0.1]),
        ("KG", KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8), [0.4, -0.15]),
        ("vp_MN", toVerticalPotential(mn, R=1.0), [0.15, 0.08]),
    ]
    for pname, pot, ic in cases:
        for method in ALL:
            M = _build_M(ic, times, pot, method, dt=0.01)
            detm1 = numpy.fabs(numpy.linalg.det(M) - 1.0)
            symperr = numpy.amax(numpy.fabs(M.T @ _Omega @ M - _Omega))
            tol = 1e-10 if method in SYMPLECTIC else 1e-8
            assert detm1 < tol, (
                f"|det(M)-1|={detm1:g} exceeds {tol:g} for {pname}, {method}"
            )
            assert symperr < tol, (
                f"||M^T Omega M - Omega||={symperr:g} exceeds {tol:g} for {pname}, {method}"
            )
    return None


def test_linear_dxdv_fd_of_flow_per_column():
    """Per-column finite-difference-of-the-flow: each column of the 1D STM must
    equal the central difference of the integrated base flow along x and v. This
    certifies M is the Jacobian of the actual discrete map (independent of the
    symplecticity/Liouville argument) and, crucially, catches sign/factor errors
    in K=-linear2deriv that det(M)=1 alone cannot. Tested on nonlinear
    (IsothermalDisk, KGPotential) and vertical (verticalPotential of
    MiyamotoNagai) potentials."""
    from galpy.potential import (
        IsothermalDiskPotential,
        KGPotential,
        MiyamotoNagaiPotential,
        toVerticalPotential,
    )

    times = numpy.linspace(0.0, 2.0, 101)
    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1)
    cases = [
        ("IsoDisk", IsothermalDiskPotential(amp=1.0, sigma=0.2), [0.3, 0.1]),
        ("KG", KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8), [0.4, -0.15]),
        ("vp_MN", toVerticalPotential(mn, R=1.0), [0.15, 0.08]),
    ]
    for pname, pot, ic in cases:
        for method in ALL:
            M = _build_M(ic, times, pot, method, dt=0.01)
            Mfd = _fd_of_flow_M(ic, times, pot, method, dt=0.01)
            err = numpy.amax(numpy.fabs(M - Mfd))
            # odeint's default adaptive tolerance sets a looser FD-of-flow floor
            tol = 1e-5 if method == "odeint" else 1e-6
            assert err < tol, (
                f"1D per-column FD-of-flow differs from the STM by {err:g} for "
                f"{pname}, {method}"
            )
    return None


def test_linear_dxdv_method_agreement():
    """All 1D methods converge to the same continuous STM: at matched (small) dt
    the symplectic and RK STMs must agree with a tight dop853_c reference, and
    the symplectic STMs converge to it at their method order as dt->0."""
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.2)
    ic = [0.3, 0.1]
    times = numpy.linspace(0.0, 2.0, 101)
    Mref = _build_M(ic, times, pot, "dop853_c", dt=0.005, rtol=1e-13, atol=1e-13)
    # all methods agree with the reference at a small stepsize
    for method in ALL:
        M = _build_M(ic, times, pot, method, dt=0.005)
        err = numpy.amax(numpy.fabs(M - Mref))
        assert err < 1e-3, f"{method} STM differs from dop853_c by {err:g}"
    # method-order convergence of the two lower-order symplectic methods
    for method in ["leapfrog_c", "symplec4_c"]:
        errs = []
        for dt in [0.02, 0.01, 0.005]:
            M = _build_M(ic, times, pot, method, dt=dt)
            errs.append(numpy.amax(numpy.fabs(M - Mref)))
        expected = 2.0 ** ORDER[method]
        for ratio in (errs[0] / errs[1], errs[1] / errs[2]):
            assert 0.7 * expected < ratio < 1.3 * expected, (
                f"{method} vs dop853_c convergence ratio {ratio:.2f} not ~{expected:.0f}"
            )
    return None


def test_linear_dxdv_pure_python_rejected():
    """The pure-Python 'leapfrog' and 'ias15_c' have no dxdv path and must raise
    for a 1D orbit (only the C symplectic and RK/pure-Python integrators are
    wired)."""
    from galpy.orbit import Orbit
    from galpy.potential import KGPotential

    pot = KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8)
    times = numpy.linspace(0.0, 2.0, 51)
    for method in ("leapfrog", "ias15_c"):
        o = Orbit([0.3, 0.1])
        with pytest.raises(ValueError):
            o.integrate_dxdv([1.0, 0.0], times, pot, method=method)
    return None


def test_linear_dxdv_dt_none_autostep():
    # integrate_dxdv without an explicit dt (dt=None -> the -9999.99 auto-step
    # sentinel in integrateLinearOrbit_dxdv_c).
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    times = numpy.linspace(0.0, 5.0, 101)
    o = Orbit([0.2, 0.05])
    o.integrate_dxdv([1.0, 0.0], times, pot, method="dopr54_c")  # no dt
    M = numpy.asarray(o.getOrbit_dxdv())
    assert M.shape == (101, 2)
    assert numpy.all(numpy.isfinite(M))
    return None


def test_linear_dxdv_multiple_orbits():
    # multi-orbit 1D dxdv (the parallel_map branch of integrateLinearOrbit_dxdv);
    # each orbit's STM must match a single-orbit solve.
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential

    pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    times = numpy.linspace(0.0, 5.0, 101)
    ics = [[0.2, 0.05], [0.3, -0.02]]
    o = Orbit(ics)
    o.integrate_dxdv([[1.0, 0.0], [1.0, 0.0]], times, pot, method="rk4_c", dt=0.01)
    Mmulti = numpy.asarray(o.getOrbit_dxdv())
    assert Mmulti.shape == (2, 101, 2)
    for ii, ic in enumerate(ics):
        os_ = Orbit(ic)
        os_.integrate_dxdv([1.0, 0.0], times, pot, method="rk4_c", dt=0.01)
        numpy.testing.assert_allclose(
            Mmulti[ii], numpy.asarray(os_.getOrbit_dxdv()), rtol=1e-10, atol=1e-12
        )
    return None


def test_linear_dxdv_composite_potential():
    # A composite (sum of) linear potentials: the pure-Python integrator's
    # variational RHS calls the composite _force2deriv (sum of the components),
    # which the C integrators bypass. Check it agrees with the C integrator (which
    # sums the components' C linear2deriv).
    from galpy.orbit import Orbit
    from galpy.potential import IsothermalDiskPotential, KGPotential

    comp = [
        KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8),
        IsothermalDiskPotential(amp=0.5, sigma=0.4),
    ]
    times = numpy.linspace(0.0, 5.0, 101)
    o = Orbit([0.2, 0.05])
    o.integrate_dxdv([1.0, 0.0], times, comp, method="dop853")  # Python -> _force2deriv
    Mpy = numpy.asarray(o.getOrbit_dxdv())
    assert Mpy.shape == (101, 2)
    assert numpy.all(numpy.isfinite(Mpy))
    oc = Orbit([0.2, 0.05])
    oc.integrate_dxdv([1.0, 0.0], times, comp, method="rk4_c", dt=0.005)
    numpy.testing.assert_allclose(
        Mpy[-1], numpy.asarray(oc.getOrbit_dxdv())[-1], rtol=1e-4
    )
    return None
