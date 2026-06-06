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
from galpy.potential import IsochronePotential, PlummerPotential

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
_POTS = [
    ("Plummer", PlummerPotential(amp=1.0, b=0.6)),
    ("Isochrone", IsochronePotential(amp=1.0, b=0.8)),
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
