###############################################################################
# test_backend_symplecticode.py: the backend-native, differentiable leapfrog
# integrator (galpy.util.symplecticode.leapfrog).
#
# leapfrog is the Python symplectic integrator used for no-C potentials
# (Orbit.integrate(..., method='leapfrog')); Orbit hands it numpy ICs but,
# under a forced/default jax|torch backend, the force func returns a backend
# array, which flips leapfrog onto its namespace-generic path. These tests hit
# that path directly with a simple-harmonic force (force = -omega**2 q), which
# is cheap enough to run under eager jax/torch and has a closed-form solution
# (q(t) = q0 cos(omega t) + (p0/omega) sin(omega t)). They prove:
#   1. the backend result is byte-identical to the numpy path (same steps),
#   2. it matches the analytic SHM trajectory,
#   3. autodiff through leapfrog gives correct gradients of the final state
#      w.r.t. the initial condition AND a force parameter (vs FD / analytic),
#   4. the numpy path is unchanged.
#
# Self-skips the backend cases unless jax / torch is installed.
###############################################################################
import numpy
import pytest

from galpy.util import symplecticode

pytestmark = pytest.mark.backend_managed

HAVE_JAX = False
HAVE_TORCH = False
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    torch.set_default_dtype(torch.float64)

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    pass

_TS = numpy.linspace(0.0, 6.0, 80)
_YO = numpy.array([1.0, 0.0])  # SHM initial condition [q0, p0]
_RTOL = 1e-8
_ATOL = 1e-8


def _analytic_shm(ts, q0, p0, omega):
    # exact solution of q'' = -omega**2 q
    return q0 * numpy.cos(omega * ts) + (p0 / omega) * numpy.sin(omega * ts)


# ------------------------------------------------------------- numpy path unchanged
def test_leapfrog_numpy_path_unchanged():
    # the historical numpy path (item-assign into a preallocated array) still
    # integrates SHM correctly -- guards byte-identity of the numpy branch
    out = symplecticode.leapfrog(lambda q, t=0.0: -q, _YO, _TS, rtol=_RTOL, atol=_ATOL)
    assert isinstance(out, numpy.ndarray)
    assert out.shape == (len(_TS), 2)
    numpy.testing.assert_allclose(
        out[:, 0], _analytic_shm(_TS, 1.0, 0.0, 1.0), rtol=1e-5, atol=1e-5
    )


# ------------------------------------------------------- backend value parity vs numpy
@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
def test_leapfrog_jax_value_parity():
    # a numpy IC with a jax-returning force flips leapfrog onto the backend path
    # (the IC is coerced onto the namespace); the trajectory is a jax array that
    # is bit-for-bit identical to the numpy path and matches analytic SHM.
    ref = symplecticode.leapfrog(lambda q, t=0.0: -q, _YO, _TS, rtol=_RTOL, atol=_ATOL)
    got = symplecticode.leapfrog(
        lambda q, t=0.0: -jnp.asarray(q), _YO, _TS, rtol=_RTOL, atol=_ATOL
    )
    assert "jax" in type(got).__module__
    numpy.testing.assert_array_equal(numpy.asarray(got), ref)  # byte-identical
    numpy.testing.assert_allclose(
        numpy.asarray(got)[:, 0],
        _analytic_shm(_TS, 1.0, 0.0, 1.0),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_leapfrog_torch_value_parity():
    ref = symplecticode.leapfrog(lambda q, t=0.0: -q, _YO, _TS, rtol=_RTOL, atol=_ATOL)
    got = symplecticode.leapfrog(
        lambda q, t=0.0: -torch.as_tensor(numpy.asarray(q)),
        _YO,
        _TS,
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert "torch" in type(got).__module__
    numpy.testing.assert_array_equal(got.detach().cpu().numpy(), ref)  # byte-identical
    numpy.testing.assert_allclose(
        got.detach().cpu().numpy()[:, 0],
        _analytic_shm(_TS, 1.0, 0.0, 1.0),
        rtol=1e-5,
        atol=1e-5,
    )


# --------------------------------------------------- gradient w.r.t. IC (vs FD/analytic)
@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
def test_leapfrog_jax_grad_ic_vs_fd():
    # d(final q)/d(q0): a backend IC (grad-tracked) stays on the backend (no
    # coerce); the gradient matches FD and the analytic value cos(omega T).
    def final_q(q0):
        y = jnp.stack([q0, jnp.asarray(0.0)])
        r = symplecticode.leapfrog(
            lambda q, t=0.0: -q, y, jnp.asarray(_TS), rtol=_RTOL, atol=_ATOL
        )
        return r[-1, 0]

    g = float(jax.grad(final_q)(1.0))
    eps = 1e-6
    fd = float((final_q(1.0 + eps) - final_q(1.0 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)
    numpy.testing.assert_allclose(g, numpy.cos(_TS[-1]), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_leapfrog_torch_grad_ic_vs_fd():
    q0 = torch.tensor(1.0, requires_grad=True)
    y = torch.stack([q0, torch.tensor(0.0)])
    r = symplecticode.leapfrog(
        lambda q, t=0.0: -q, y, torch.as_tensor(_TS), rtol=_RTOL, atol=_ATOL
    )
    r[-1, 0].backward()
    numpy.testing.assert_allclose(
        float(q0.grad), numpy.cos(_TS[-1]), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------- gradient w.r.t. a force parameter (jax)
@pytest.mark.skipif(not HAVE_JAX, reason="jax not installed")
def test_leapfrog_jax_grad_param_vs_fd():
    # d(final q)/d(omega) for force = -omega**2 q: gradients flow through the
    # parameter carried by func (the potential-parameter analogue), matching FD.
    def final_q(omega):
        r = symplecticode.leapfrog(
            lambda q, t=0.0: -(omega**2) * q,
            jnp.asarray(_YO),
            jnp.asarray(_TS),
            rtol=_RTOL,
            atol=_ATOL,
        )
        return r[-1, 0]

    g = float(jax.grad(final_q)(1.0))
    eps = 1e-6
    fd = float((final_q(1.0 + eps) - final_q(1.0 - eps)) / (2 * eps))
    numpy.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


# --------------------------------------------- torch force-parameter gradient matches jax
@pytest.mark.skipif(not (HAVE_JAX and HAVE_TORCH), reason="needs both jax and torch")
def test_leapfrog_torch_grad_param_matches_jax():
    def final_q_jax(omega):
        r = symplecticode.leapfrog(
            lambda q, t=0.0: -(omega**2) * q,
            jnp.asarray(_YO),
            jnp.asarray(_TS),
            rtol=_RTOL,
            atol=_ATOL,
        )
        return r[-1, 0]

    g_jax = float(jax.grad(final_q_jax)(1.0))
    omega = torch.tensor(1.0, requires_grad=True)
    r = symplecticode.leapfrog(
        lambda q, t=0.0: -(omega**2) * q,
        torch.as_tensor(_YO),
        torch.as_tensor(_TS),
        rtol=_RTOL,
        atol=_ATOL,
    )
    r[-1, 0].backward()
    numpy.testing.assert_allclose(float(omega.grad), g_jax, rtol=1e-6, atol=1e-8)
