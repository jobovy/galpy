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
    FlattenedPowerPotential,
    HernquistPotential,
    IsochronePotential,
    JaffePotential,
    KeplerPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    MN3ExponentialDiskPotential,
    NFWPotential,
    PerfectEllipsoidPotential,
    PlummerPotential,
    PowerSphericalPotential,
    PowerSphericalPotentialwCutoff,
    SCFPotential,
    SoftenedNeedleBarPotential,
    SpiralArmsPotential,
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
    assert numpy.isfinite(h)


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
