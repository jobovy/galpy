###############################################################################
# test_backend_inbackend_ode.py: the in-backend differentiable orbit integrator
# (galpy.backend._reference.integrate_orbit) -- diffrax (jax) / torchdiffeq
# (torch) integration of the backend-agnostic forces.
#
# Proves, for the reference migrated potentials:
#   1. the in-backend trajectory matches galpy's C integrator (modulo phi 2pi),
#   2. autodiff through the ODE solve gives correct gradients of the final state
#      w.r.t. initial conditions AND potential parameters (vs finite difference),
#   3. the 6x6 state-transition matrix d x(t)/d x0 via jax jacrev matches a
#      column-by-column finite-difference of the flow,
#   4. the torch.autograd orbit gradient matches the jax.grad one (cross-backend).
#
# Self-skips unless the runtime ODE extra (diffrax / torchdiffeq) is installed.
###############################################################################
import numpy
import pytest

from galpy.orbit import Orbit
from galpy.potential import (
    BurkertPotential,
    DehnenBarPotential,
    DehnenCoreSphericalPotential,
    DehnenSphericalPotential,
    DoubleExponentialDiskPotential,
    EllipticalDiskPotential,
    FlattenedPowerPotential,
    HernquistPotential,
    IsochronePotential,
    IsothermalDiskPotential,
    JaffePotential,
    KeplerPotential,
    KGPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    MN3ExponentialDiskPotential,
    NFWPotential,
    PerfectEllipsoidPotential,
    PlummerPotential,
    PowerSphericalPotential,
    PowerSphericalPotentialwCutoff,
    RZToverticalPotential,
    SCFPotential,
    SoftenedNeedleBarPotential,
    SpiralArmsPotential,
    SteadyLogSpiralPotential,
    TriaxialHernquistPotential,
    TriaxialNFWPotential,
    TwoPowerSphericalPotential,
)

pytestmark = pytest.mark.backend_managed

HAVE_JAX = False
HAVE_TORCH = False
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import diffrax  # noqa: F401
    import jax.numpy as jnp

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    torch.set_default_dtype(torch.float64)
    import torchdiffeq  # noqa: F401

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    pass

from galpy.backend._reference import integrate_orbit  # noqa: E402

_IC = [1.0, 0.1, 0.9, 0.2, 0.05, 0.3]  # R, vR, vT, z, vz, phi
_TS = numpy.linspace(0.0, 6.0, 120)
# Broad potential sweep: every family the in-backend (diffrax/torchdiffeq) path
# supports, validated trajectory-vs-C + grad-vs-FD across all of them (the sweep
# in PR-pillar2; KuzminDisk omitted -- its |z| kink makes the *gradient* undefined
# at a z=0 plane crossing, not a migration gap; CosmphiDisk is planar-only).
_POTS = [
    ("Plummer", PlummerPotential(amp=1.0, b=0.6)),
    ("Isochrone", IsochronePotential(amp=1.0, b=0.8)),
    ("Hernquist", HernquistPotential(amp=1.0, a=0.7)),
    ("NFW", NFWPotential(amp=1.0, a=1.5)),
    ("Jaffe", JaffePotential(amp=1.0, a=0.7)),
    ("DehnenSpherical", DehnenSphericalPotential(amp=1.0, a=1.0, alpha=1.5)),
    ("DehnenCoreSpherical", DehnenCoreSphericalPotential(amp=1.0, a=1.0)),
    ("Kepler", KeplerPotential(amp=1.0)),
    ("PowerSpherical", PowerSphericalPotential(amp=1.0, alpha=2.0)),
    (
        "PowerSphericalwCutoff",
        PowerSphericalPotentialwCutoff(amp=1.0, alpha=1.0, rc=1.0),
    ),
    ("Burkert", BurkertPotential(amp=1.0, a=1.0)),
    (
        "TwoPowerSpherical",
        TwoPowerSphericalPotential(amp=1.0, a=1.0, alpha=1.0, beta=3.0),
    ),
    ("MiyamotoNagai", MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1)),
    (
        "DoubleExponentialDisk",
        DoubleExponentialDiskPotential(amp=1.0, hr=1.0 / 3.0, hz=1.0 / 16.0),
    ),
    (
        "MN3ExponentialDisk",
        MN3ExponentialDiskPotential(amp=1.0, hr=1.0 / 3.0, hz=1.0 / 16.0),
    ),
    ("LogHalo", LogarithmicHaloPotential(amp=1.0, q=0.8)),
    ("FlattenedPower", FlattenedPowerPotential(amp=1.0, alpha=0.5, q=0.9)),
    ("TriaxialNFW", TriaxialNFWPotential(amp=1.0, a=1.0, b=0.8, c=0.6)),
    ("PerfectEllipsoid", PerfectEllipsoidPotential(amp=1.0, a=1.0, b=0.9, c=0.7)),
    ("TriaxialHernquist", TriaxialHernquistPotential(amp=1.0, a=1.0, b=0.8, c=0.6)),
    ("DehnenBar", DehnenBarPotential()),
    ("SoftenedNeedleBar", SoftenedNeedleBarPotential(amp=1.0, a=1.0, b=0.1, c=0.5)),
    ("SpiralArms", SpiralArmsPotential()),
    ("SCF", SCFPotential(amp=1.0)),
]


def _wrap_phi(a):
    a = numpy.array(a, dtype=float)
    a[..., 5] = (a[..., 5] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return a


def _c_reference(pot):
    o = Orbit(_IC)
    o.integrate(_TS, pot, method="dop853_c")
    return numpy.array(
        [[o.R(t), o.vR(t), o.vT(t), o.z(t), o.vz(t), o.phi(t)] for t in _TS]
    )


def test_inbackend_numpy_raises():
    # The in-backend ODE integrator is for jax/torch only; a numpy IC must raise
    # (numpy orbits use galpy's C/scipy integrators via Orbit.integrate).
    with pytest.raises(NotImplementedError):
        integrate_orbit(PlummerPotential(amp=1.0, b=0.6), numpy.asarray(_IC), _TS)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("name,pot", _POTS, ids=[p[0] for p in _POTS])
def test_inbackend_matches_c_jax(name, pot):
    ref = _c_reference(pot)
    got = numpy.asarray(integrate_orbit(pot, jnp.asarray(_IC), jnp.asarray(_TS)))
    numpy.testing.assert_allclose(_wrap_phi(got), _wrap_phi(ref), rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("name,pot", _POTS, ids=[p[0] for p in _POTS])
def test_inbackend_matches_c_torch(name, pot):
    ref = _c_reference(pot)
    got = (
        integrate_orbit(pot, torch.as_tensor(_IC), torch.as_tensor(_TS))
        .detach()
        .numpy()
    )
    numpy.testing.assert_allclose(_wrap_phi(got), _wrap_phi(ref), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_grad_ic_vs_fd():
    # d(final R)/d(vR0) through the ODE solve, autodiff vs central finite-difference
    p = PlummerPotential(amp=1.0, b=0.6)
    ts = jnp.asarray(_TS)

    def final_R(vR0):
        ic = jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3])
        return integrate_orbit(p, ic, ts)[-1][0]

    ad = float(jax.grad(final_R)(jnp.asarray(0.1)))
    eps = 1e-6
    fd = (float(final_R(0.1 + eps)) - float(final_R(0.1 - eps))) / (2 * eps)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_grad_param_vs_fd():
    # d(final R)/d(Plummer b) -- parameter gradient backpropagated through the solve
    ts = jnp.asarray(_TS)

    def final_R(b):
        return integrate_orbit(PlummerPotential(amp=1.0, b=b), jnp.asarray(_IC), ts)[
            -1
        ][0]

    ad = float(jax.grad(final_R)(jnp.asarray(0.6)))
    eps = 1e-6
    fd = (float(final_R(0.6 + eps)) - float(final_R(0.6 - eps))) / (2 * eps)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_stm_vs_fd():
    # the full 6x6 state-transition matrix M = d x(t_f)/d x0 via reverse-mode
    # autodiff (jacrev) must match a column-by-column finite-difference of the flow.
    # (diffrax's diffeqsolve is a custom_vjp -> reverse-mode only; FD of the flow is
    # the independent ground-truth check, as in the C variational test battery.)
    p = PlummerPotential(amp=1.0, b=0.6)
    ts = jnp.asarray(_TS)

    def final_state(y0):
        return integrate_orbit(p, y0, ts)[-1]

    y0 = numpy.asarray(_IC, dtype=float)
    M = numpy.asarray(jax.jacrev(final_state)(jnp.asarray(y0)))
    eps = 1e-6
    M_fd = numpy.zeros((6, 6))
    for j in range(6):
        yp, ym = y0.copy(), y0.copy()
        yp[j] += eps
        ym[j] -= eps
        M_fd[:, j] = (
            numpy.asarray(final_state(jnp.asarray(yp)))
            - numpy.asarray(final_state(jnp.asarray(ym)))
        ) / (2 * eps)
    numpy.testing.assert_allclose(M, M_fd, rtol=1e-5, atol=1e-6)
    assert numpy.max(numpy.abs(M)) > 1e-3  # non-trivial STM


@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_inbackend_grad_torch_matches_jax():
    # d(final R)/d(vR0) through the ODE solve, via torch.autograd vs jax.grad. Both
    # are exact through their solvers, so this cross-validates the two backend
    # integrators and is FD-independent. (We deliberately avoid an adaptive-solver
    # finite-difference reference here: torchdiffeq's adaptive dopri8 chooses
    # slightly different step sequences for the +/-eps solves, so a torch-dopri8 FD
    # jitters by ~1e-3 even though the *autodiff* gradient is correct -- confirmed by
    # dopri5/rk4 and jax all agreeing.)
    p = PlummerPotential(amp=1.0, b=0.6)

    def fR_jax(vR0):
        ic = jnp.array([1.0, vR0, 0.9, 0.2, 0.05, 0.3])
        return integrate_orbit(p, ic, jnp.asarray(_TS))[-1][0]

    g_jax = float(jax.grad(fR_jax)(jnp.asarray(0.1)))

    ic = torch.tensor(_IC, dtype=torch.float64, requires_grad=True)
    integrate_orbit(p, ic, torch.as_tensor(_TS))[-1][0].backward()
    g_torch = float(ic.grad[1])

    numpy.testing.assert_allclose(g_torch, g_jax, rtol=1e-6, atol=1e-8)


# --------------------- native-planar / composite potentials ---------------------
# The in-backend RHS used to call the 3D force layer on the potential, so a
# native-planar potential (SteadyLogSpiral/EllipticalDisk -- no z-argument) or a
# planarCompositePotential crashed. It now dispatches to the planar force layer
# for a planarForce (planar orbit integrated as 3D with z=vz=0, no z-force) and to
# the 3D layer otherwise. Covers native-planar, .toPlanar(), the modern
# planarCompositePotential (pot1+pot2), and a 3D CompositePotential -- all vs
# galpy's C integrator. (Legacy lists are converted to a composite by
# Orbit.integrate before reaching the integrator, so they need no special path.)
_IC_PLANAR = [1.0, 0.1, 0.9, 0.2]  # R, vR, vT, phi
_SPIRAL = SteadyLogSpiralPotential(amp=1.0, omegas=0.65, A=-0.035)
_EDISK = EllipticalDiskPotential(
    twophio=0.05, phib=25.0 / 180.0 * numpy.pi, p=0.0, tform=-150.0, tsteady=125.0
)
_PLANAR_POTS = [
    ("native-SteadyLogSpiral", _SPIRAL),
    ("native-EllipticalDisk", _EDISK),
    ("toPlanar-LogHalo", LogarithmicHaloPotential(amp=1.0, q=0.8).toPlanar()),
    ("composite-planar", LogarithmicHaloPotential(amp=1.0, q=0.8).toPlanar() + _SPIRAL),
    (
        "composite-3D",
        MiyamotoNagaiPotential(amp=0.8, a=0.5, b=0.1)
        + LogarithmicHaloPotential(amp=0.2, q=0.8),
    ),
]


def _wrap_phi_planar(a):
    a = numpy.array(a, dtype=float)
    a[..., 3] = (a[..., 3] + numpy.pi) % (2 * numpy.pi) - numpy.pi
    return a


def _c_reference_planar(pot, ic=None):
    o = Orbit(_IC_PLANAR if ic is None else list(ic))
    o.integrate(_TS, pot, method="dop853_c")
    return numpy.array([[o.R(t), o.vR(t), o.vT(t), o.phi(t)] for t in _TS])


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("name,pot", _PLANAR_POTS, ids=[p[0] for p in _PLANAR_POTS])
def test_inbackend_planar_matches_c_jax(name, pot):
    ref = _c_reference_planar(pot)
    got = numpy.asarray(integrate_orbit(pot, jnp.asarray(_IC_PLANAR), jnp.asarray(_TS)))
    numpy.testing.assert_allclose(
        _wrap_phi_planar(got), _wrap_phi_planar(ref), rtol=1e-6, atol=1e-7
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("name,pot", _PLANAR_POTS, ids=[p[0] for p in _PLANAR_POTS])
def test_inbackend_planar_matches_c_torch(name, pot):
    ref = _c_reference_planar(pot)
    got = (
        integrate_orbit(pot, torch.as_tensor(_IC_PLANAR), torch.as_tensor(_TS))
        .detach()
        .numpy()
    )
    numpy.testing.assert_allclose(
        _wrap_phi_planar(got), _wrap_phi_planar(ref), rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_planar_composite_grad_vs_fd_jax():
    # d(final R)/d(vR0) through a composite-planar orbit solve, autodiff vs FD --
    # the differentiable-evolution path an evolveddiskdf moment rides on.
    pot = LogarithmicHaloPotential(amp=1.0, q=0.8).toPlanar() + _SPIRAL
    ts = jnp.asarray(_TS)

    def final_R(vR0):
        ic = jnp.array([1.0, vR0, 0.9, 0.2])
        return integrate_orbit(pot, ic, ts)[-1][0]

    ad = float(jax.grad(final_R)(jnp.asarray(0.1)))
    eps = 1e-6
    fd = (float(final_R(0.1 + eps)) - float(final_R(0.1 - eps))) / (2 * eps)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_planar_batch_jax():
    # a batch of planar orbits (N,4) with a composite-planar potential in one solve
    pot = LogarithmicHaloPotential(amp=1.0, q=0.8).toPlanar() + _SPIRAL
    ics = numpy.array([_IC_PLANAR, [0.9, 0.0, 1.0, 0.1], [1.1, -0.05, 0.85, 0.4]])
    ref = numpy.array([_c_reference_planar(pot, ic)[-1] for ic in ics])
    out = numpy.asarray(integrate_orbit(pot, jnp.asarray(ics), jnp.asarray(_TS)))
    assert out.shape == (len(_TS), 3, 4)
    numpy.testing.assert_allclose(
        _wrap_phi_planar(out[-1]), _wrap_phi_planar(ref), rtol=1e-6, atol=1e-7
    )


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_orbit_integrate_composite_planar_end_to_end_jax():
    # end-to-end via Orbit.integrate with a jax IC + composite-planar potential and
    # a C-method NAME (rk6_c): a backend IC routes a C dxdv method to the
    # differentiable in-backend integrator, and the composed potential is forwarded
    # (not the raw list). Planar, so it takes the in-backend path (not 6D C-STM) and
    # the final coordinate is differentiable w.r.t. the IC.
    pot = LogarithmicHaloPotential(amp=1.0, q=0.8).toPlanar() + _SPIRAL

    def final_R(vR0):
        o = Orbit(jnp.array([1.0, vR0, 0.9, 0.2]))
        o.integrate(jnp.asarray(_TS), pot, method="rk6_c")
        return o.getOrbit().reshape(-1, 4)[-1, 0]

    ref = _c_reference_planar(pot)[-1, 0]
    numpy.testing.assert_allclose(
        float(final_R(jnp.asarray(0.1))), ref, rtol=1e-6, atol=1e-7
    )
    ad = float(jax.grad(final_R)(jnp.asarray(0.1)))
    eps = 1e-6
    fd = (
        float(final_R(jnp.asarray(0.1 + eps))) - float(final_R(jnp.asarray(0.1 - eps)))
    ) / (2 * eps)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


# ----------------------- in-backend solver/adjoint/max_steps knobs (#102) -------
# integrate_orbit (and Orbit.integrate via inbackend_kwargs) forwards a 'solver',
# 'adjoint' (jax), and 'max_steps' to the underlying diffrax/torchdiffeq call. The
# headline use is jax SECOND derivatives: the default RecursiveCheckpointAdjoint is
# reverse-mode first-order only, so adjoint='direct' (diffrax.DirectAdjoint) is what
# makes jax.hessian work through the solve (torchdiffeq double-backprops as-is).
@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_solver_adjoint_options_jax():
    import diffrax

    pot = PlummerPotential(amp=1.0, b=0.6)
    ic, ts = jnp.asarray(_IC), jnp.asarray(_TS)
    base = numpy.asarray(integrate_orbit(pot, ic, ts))
    # explicit 'recursive' adjoint + 'dopri8' solver NAME reproduces the defaults
    same = numpy.asarray(
        integrate_orbit(pot, ic, ts, solver="dopri8", adjoint="recursive")
    )
    numpy.testing.assert_allclose(same, base, rtol=1e-10, atol=1e-10)
    # a diffrax solver INSTANCE passes through; another solver still matches to tol
    alt = numpy.asarray(integrate_orbit(pot, ic, ts, solver=diffrax.Tsit5()))
    numpy.testing.assert_allclose(alt, base, rtol=1e-6, atol=1e-7)
    # a diffrax adjoint INSTANCE also passes through unchanged
    inst = numpy.asarray(
        integrate_orbit(pot, ic, ts, adjoint=diffrax.RecursiveCheckpointAdjoint())
    )
    numpy.testing.assert_allclose(inst, base, rtol=1e-10, atol=1e-10)
    # unknown names raise a clear ValueError
    with pytest.raises(ValueError):
        integrate_orbit(pot, ic, ts, solver="nope")
    with pytest.raises(ValueError):
        integrate_orbit(pot, ic, ts, adjoint="nope")


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_hessian_direct_adjoint_jax():
    # second derivative d2 R(T) / d vR0^2 through the diffrax solve. adjoint='direct'
    # (DirectAdjoint) makes it differentiable twice; the default adjoint cannot.
    pot = PlummerPotential(amp=1.0, b=0.6)
    ts = jnp.asarray(_TS)

    def final_R(vR):
        ic = jnp.asarray(_IC).at[1].set(vR)
        return integrate_orbit(pot, ic, ts, adjoint="direct", max_steps=2048)[-1][0]

    h = float(jax.hessian(final_R)(jnp.asarray(0.1)))
    g = jax.grad(final_R)
    fd2 = float((g(jnp.asarray(0.1 + 1e-3)) - g(jnp.asarray(0.1 - 1e-3))) / 2e-3)
    assert numpy.isfinite(h)
    numpy.testing.assert_allclose(h, fd2, rtol=1e-3, atol=1e-6)

    # the DEFAULT (recursive) adjoint cannot be differentiated twice -> errors
    def final_R_default(vR):
        ic = jnp.asarray(_IC).at[1].set(vR)
        return integrate_orbit(pot, ic, ts)[-1][0]

    with pytest.raises(Exception):
        jax.hessian(final_R_default)(jnp.asarray(0.1))


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_orbit_integrate_inbackend_kwargs_jax():
    # Orbit.integrate threads inbackend_kwargs down to the in-backend solver: a
    # Hessian of a final coordinate through Orbit(jax IC).integrate(method='diffrax').
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_R(vR):
        ic = jnp.asarray(_IC).at[1].set(vR)
        o = Orbit(ic)
        o.integrate(
            jnp.asarray(_TS),
            pot,
            method="diffrax",
            inbackend_kwargs={"adjoint": "direct", "max_steps": 2048},
        )
        return o.getOrbit().reshape(-1, len(_IC))[-1, 0]  # final R

    h = float(jax.hessian(final_R)(jnp.asarray(0.1)))
    # Not just "is finite": the kwargs only pick solver knobs, so the AD Hessian
    # must MATCH a central finite-difference of jax.grad, h-converged. A wrong
    # but finite Hessian (e.g. adjoint silently not threading) passed before.
    g = jax.grad(final_R)

    def fd(step):
        return float(
            (g(jnp.asarray(0.1 + step)) - g(jnp.asarray(0.1 - step))) / (2.0 * step)
        )

    fd_coarse, fd_fine = fd(1e-4), fd(5e-5)
    assert abs(fd_coarse - fd_fine) <= 1e-3 * (abs(fd_fine) + 1e-8), (
        f"FD not converged: {fd_coarse:.8g} vs {fd_fine:.8g}"
    )
    assert abs(h - fd_fine) <= 1e-5 * (abs(fd_fine) + 1e-8), (
        f"AD Hessian {h:.8g} != FD-of-grad {fd_fine:.8g}"
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_inbackend_solver_maxsteps_torch():
    # torchdiffeq path: 'solver' selects the method and 'max_steps' caps the step
    # count (torchdiffeq max_num_steps); the defaults reproduce the plain call.
    pot = PlummerPotential(amp=1.0, b=0.6)
    ic = torch.tensor(_IC, dtype=torch.float64)
    ts = torch.as_tensor(_TS)
    base = integrate_orbit(pot, ic, ts).detach().numpy()
    same = (
        integrate_orbit(pot, ic, ts, solver="dopri5", max_steps=100000).detach().numpy()
    )
    numpy.testing.assert_allclose(same, base, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_per_orbit_times_with_knobs_jax():
    # PER-ORBIT time grids (shape (N, nt)) route through jax.vmap; the solver/
    # max_steps knobs must thread into each per-orbit solve.
    pot = PlummerPotential(amp=1.0, b=0.6)
    ics = jnp.stack([jnp.asarray(_IC), jnp.asarray(_IC).at[1].set(0.2)])
    ts2 = jnp.stack([jnp.asarray(_TS), jnp.asarray(_TS) * 0.5])
    out = integrate_orbit(pot, ics, ts2, solver="dopri8", max_steps=50000)
    assert out.shape == (len(_TS), 2, 6)
    assert bool(numpy.isfinite(numpy.asarray(out)).all())


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_inbackend_per_orbit_times_with_knobs_torch():
    # PER-ORBIT grids on torch use a per-orbit odeint loop; solver/max_steps thread in.
    pot = PlummerPotential(amp=1.0, b=0.6)
    ic0 = torch.tensor(_IC, dtype=torch.float64)
    ics = torch.stack([ic0, ic0.clone()])
    ts0 = torch.as_tensor(_TS)
    ts2 = torch.stack([ts0, ts0 * 0.5])
    out = integrate_orbit(pot, ics, ts2, solver="dopri5", max_steps=50000)
    assert tuple(out.shape) == (len(_TS), 2, 6)
    assert bool(torch.isfinite(out).all())


###############################################################################
# 1D linear potentials: the in-backend ODE's dim==2 branch ([x, vx]). Same
# diffrax/torchdiffeq path as the 3D/planar cases, exercised for the
# linearPotential force layer (_evaluatelinearForces) -- the vertical-orbit
# integration actionAngleVertical rides on. Covers native-linear, the
# toVertical wrapper of a 3D potential, and a linear composite.
###############################################################################
_IC_1D = [0.15, 0.1]  # x, vx (1D phase space)
_LINEAR_POTS = [
    ("native-IsothermalDisk", IsothermalDiskPotential(amp=1.0, sigma=0.3)),
    ("native-KG", KGPotential(K=1.15, F=0.03, D=1.8)),
    # toVertical: the 1D vertical potential of a 3D disk at R=1 (wraps _zforce)
    ("toVertical-MN", RZToverticalPotential(MiyamotoNagaiPotential(a=0.5, b=0.1), 1.0)),
    (
        "composite-linear",
        IsothermalDiskPotential(amp=1.0, sigma=0.3)
        + KGPotential(K=1.15, F=0.03, D=1.8),
    ),
]


def _c_reference_1d(pot, ic=None):
    o = Orbit(_IC_1D if ic is None else list(ic))
    o.integrate(_TS, pot, method="dop853_c")
    return numpy.array([[o.x(t), o.vx(t)] for t in _TS])


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
@pytest.mark.parametrize("name,pot", _LINEAR_POTS, ids=[p[0] for p in _LINEAR_POTS])
def test_inbackend_1d_matches_c_jax(name, pot):
    ref = _c_reference_1d(pot)
    got = numpy.asarray(integrate_orbit(pot, jnp.asarray(_IC_1D), jnp.asarray(_TS)))
    assert got.shape == (len(_TS), 2)
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
@pytest.mark.parametrize("name,pot", _LINEAR_POTS, ids=[p[0] for p in _LINEAR_POTS])
def test_inbackend_1d_matches_c_torch(name, pot):
    ref = _c_reference_1d(pot)
    got = (
        integrate_orbit(pot, torch.as_tensor(_IC_1D), torch.as_tensor(_TS))
        .detach()
        .numpy()
    )
    assert got.shape == (len(_TS), 2)
    numpy.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_1d_grad_vs_fd_jax():
    # d(final x)/d(vx0) through the 1D solve, autodiff vs central FD.
    pot = IsothermalDiskPotential(amp=1.0, sigma=0.3)
    ts = jnp.asarray(_TS)

    def final_x(vx0):
        return integrate_orbit(pot, jnp.array([0.15, vx0]), ts)[-1][0]

    ad = float(jax.grad(final_x)(jnp.asarray(0.1)))
    eps = 1e-6
    fd = (float(final_x(0.1 + eps)) - float(final_x(0.1 - eps))) / (2 * eps)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="need jax and torch")
def test_inbackend_1d_grad_torch_matches_jax():
    # cross-backend: torch.autograd d(final x)/d(vx0) == jax.grad, 1D path.
    pot = KGPotential(K=1.15, F=0.03, D=1.8)
    jg = float(
        jax.grad(
            lambda v: integrate_orbit(pot, jnp.array([0.15, v]), jnp.asarray(_TS))[-1][
                0
            ]
        )(jnp.asarray(0.1))
    )
    v = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    integrate_orbit(
        pot,
        torch.stack([torch.tensor(0.15, dtype=torch.float64), v]),
        torch.as_tensor(_TS),
    )[-1][0].backward()
    numpy.testing.assert_allclose(float(v.grad), jg, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_1d_batch_jax():
    # a batch of 1D orbits (N,2) integrated in ONE solve vs per-orbit C.
    pot = IsothermalDiskPotential(amp=1.0, sigma=0.3) + KGPotential(
        K=1.15, F=0.03, D=1.8
    )
    ics = numpy.array([_IC_1D, [0.3, -0.05], [-0.2, 0.15]])
    ref = numpy.array([_c_reference_1d(pot, ic)[-1] for ic in ics])
    out = numpy.asarray(integrate_orbit(pot, jnp.asarray(ics), jnp.asarray(_TS)))
    assert out.shape == (len(_TS), 3, 2)
    numpy.testing.assert_allclose(out[-1], ref, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_orbit_integrate_inbackend_kwargs_tolerance_jax():
    # rtol/atol are reachable through inbackend_kwargs, not just as integrate()
    # arguments. They used to collide with the explicitly-forwarded pair and raise
    # "got multiple values for keyword argument 'rtol'" -- which made the tolerance
    # UNREACHABLE for callers that expose only the dict (actionAngleIsochroneApprox's
    # integrate_kwargs is the motivating one: it has no rtol/atol parameters).
    pot = PlummerPotential(amp=1.0, b=0.6)

    def final_state(**kw):
        o = Orbit(jnp.asarray(_IC))
        o.integrate(jnp.asarray(_TS), pot, method="diffrax", **kw)
        return numpy.asarray(o.getOrbit()).reshape(-1, len(_IC))[-1]

    tight = final_state()  # default rtol=atol=1e-12
    loose = final_state(inbackend_kwargs={"rtol": 1e-5, "atol": 1e-5})
    d_loose = numpy.max(numpy.fabs(loose - tight))

    # Two-sided: the knob must actually REACH the solver (a silently-dropped
    # rtol would give a bit-identical trajectory), and it must behave like a
    # tolerance rather than garbage.
    assert d_loose > 1e-9, f"loose tolerance changed nothing: max|d|={d_loose:.3e}"
    assert d_loose < 1e-2, f"loose tolerance diverged: max|d|={d_loose:.3e}"

    # Passing the SAME loose tolerance the documented way must agree closely with
    # passing it through the dict -- same knob, two spellings. Both are adaptive
    # solves with the same controller, so require them to track to 1e-12.
    via_args = final_state(rtol=1e-5, atol=1e-5)
    numpy.testing.assert_allclose(loose, via_args, rtol=1e-12, atol=1e-12)

    # Precedence is documented as "inbackend_kwargs overrides": with a tight
    # explicit pair AND a loose dict, the loose one must win. This is the
    # discriminating assertion -- if precedence were reversed the result would
    # equal `tight` instead.
    override = final_state(
        rtol=1e-12, atol=1e-12, inbackend_kwargs={"rtol": 1e-5, "atol": 1e-5}
    )
    numpy.testing.assert_allclose(override, loose, rtol=1e-12, atol=1e-12)
    assert numpy.max(numpy.fabs(override - tight)) == pytest.approx(d_loose, rel=1e-9)

    # The dict must be REUSABLE: callers that store it and pass it on every call
    # (actionAngleIsochroneApprox keeps self._integrate_kwargs) would otherwise get
    # the tolerance honoured once and silently dropped afterwards. This is why the
    # implementation pops from a COPY; popping from the caller's dict passes every
    # assertion above and still breaks here on the second integration.
    shared = {"rtol": 1e-5, "atol": 1e-5}
    first = final_state(inbackend_kwargs=shared)
    second = final_state(inbackend_kwargs=shared)
    assert shared == {"rtol": 1e-5, "atol": 1e-5}, f"caller dict mutated: {shared}"
    numpy.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(second, loose, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not HAVE_JAX, reason="jax/diffrax not installed")
def test_inbackend_nsteps_makes_the_batched_reverse_pass_exact_jax():
    """``nsteps`` (constant stepping) fixes reverse-mode AD under ``jax.vmap``.

    diffrax's ``DirectAdjoint`` -- the one needed for second-order AD, e.g. an
    outer d/d(parameter) through an inner ``jacrev``, as the stream track does --
    differentiates the solver's OWN operations. With the default adaptive
    controller, ``jax.vmap`` batch elements choose different step sequences, so
    the batched reverse pass disagrees with the unbatched one (measured here at
    1e-3, and 2-13% through a full action-angle map, growing with integration
    time) even though the FORWARD values agree to ~1e-10. That asymmetry is what
    made it look like a jacrev-batching bug for a long time; it is not, and
    plain ``vmap(jacrev)`` is exact in JAX.

    Constant steps remove the step-size dependence entirely. Asserted two ways:
    the batched Jacobian must match the unbatched one to near machine precision,
    AND the constant-step trajectory must still agree with the adaptive one, so
    the fix cannot be "pass by integrating badly".
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.potential import LogarithmicHaloPotential

    # an ARRAY-valued parameter is the differentiable configuration
    lp = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(0.9))
    ts = jnp.linspace(0.0, 8.0, 60)
    V = jnp.asarray(
        [
            [1.561, 0.351, -1.155, 0.887, -0.477, 0.120],
            [1.500, 0.300, -1.100, 0.850, -0.450, 0.200],
        ]
    )

    def _final(v, **extra):
        return integrate_orbit(
            lp,
            v,
            ts,
            rtol=1e-10,
            atol=1e-10,
            max_steps=20000,
            adjoint="direct",
            **extra,
        )[-1]

    J = jax.jacrev(lambda v: _final(v, nsteps=2000))
    seq = numpy.stack([numpy.asarray(J(V[i]), dtype=float) for i in range(V.shape[0])])
    vm = numpy.asarray(jax.vmap(J)(V), dtype=float)
    scale = numpy.max(numpy.fabs(seq))
    assert numpy.all(numpy.fabs(seq - vm) < 1e-11 * scale), (
        "constant-step batched Jacobian != unbatched: "
        f"max rel {numpy.max(numpy.fabs(seq - vm)) / scale:.3e}"
    )

    # ... and the constant-step solve is still the right trajectory
    ad = numpy.asarray(_final(V[0]), dtype=float)
    cs = numpy.asarray(_final(V[0], nsteps=2000), dtype=float)
    assert numpy.all(numpy.fabs(ad - cs) < 1e-8 * numpy.max(numpy.fabs(ad))), (
        f"constant-step trajectory differs from adaptive: "
        f"max rel {numpy.max(numpy.fabs(ad - cs)) / numpy.max(numpy.fabs(ad)):.3e}"
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/torchdiffeq not installed")
def test_inbackend_nsteps_is_rejected_on_torch():
    """``nsteps`` is a jax/diffrax option; torchdiffeq picks its own steps.

    Passing it on torch must fail loudly rather than being silently ignored --
    a silently-dropped nsteps would hand back an adaptively-stepped solve while
    the caller believes the steps are fixed, which is exactly the confusion this
    option exists to remove.
    """
    import torch

    pot = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    vxvv = torch.tensor([1.0, 0.1, 1.1, 0.1, 0.2, 0.0], dtype=torch.float64)
    ts = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="jax/diffrax option"):
        integrate_orbit(pot, vxvv, ts, nsteps=100)
