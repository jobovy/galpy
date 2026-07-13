###############################################################################
# test_backend_movingobject.py: backend-agnostic tests for
# MovingObjectPotential. The class wraps an (already-migrated) kernel potential
# shifted by an integrated orbit; only its own geometry internals (_cylR,
# _cyldiff, and the Cartesian<->cylindrical cos/sin in the force / Hessian
# methods) carried raw numpy. Those internals worked eagerly on a backend but
# crashed under a jax trace (raw numpy.cos/sin/sqrt on a tracer). This module
# proves:
#   1. numpy / jax / torch value parity for every migrated method, over a 3D and
#      a 2D (z_obj = 0) object track;
#   2. the exact gap: jax.jacfwd / jax.jit over evaluateRforces (wrt R and wrt
#      phi) return finite (they previously raised TracerArrayConversionError);
#   3. the force gradient wrt R and a phi-quadratic loss h-converge to central
#      finite differences (stringent grad-vs-FD, not finite-and-nonzero);
#   4. eager jax + eager torch return backend arrays (a numpy orbit query mixed
#      with backend field coords promotes through the per-side namespaces of
#      _cyldiff -- torch rejects xp.cos on a bare numpy scalar).
#
# The object orbit is numpy-integrated (a traced/backend orbit query is the
# caller's concern); the migrated internals only need the field coords to carry
# the backend, which is what a jacfwd wrt R / phi supplies.
###############################################################################
import numpy
import pytest

from galpy import potential
from galpy.orbit import Orbit
from galpy.potential import evaluatephitorques, evaluateRforces

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# --- object tracks (numpy-integrated) + kernel ---------------------------------
_LP = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
_TS = numpy.linspace(-1.0, 5.0, 601)
_O3 = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 1.0])  # 3D track (z_obj != 0)
_O3.integrate(_TS, _LP, method="dop853_c")
_O2 = Orbit([1.1, 0.1, 1.1, 1.0])  # planar track (z_obj = 0 branch)
_O2.integrate(_TS, _LP, method="dop853_c")
_KERNEL = potential.PlummerPotential(amp=0.3, b=0.3)

_MOP3 = potential.MovingObjectPotential(_O3, pot=_KERNEL, amp=1.2)
_MOP2 = potential.MovingObjectPotential(_O2, pot=_KERNEL, amp=1.2)
_MOPS = [("3D", _MOP3), ("2D", _MOP2)]

_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_dens",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
]

# (R, z, phi, t) incl. t != 0 (object has moved), phi != 0, z = 0.
_POINTS = [
    (1.0, 0.05, 0.2, 0.0),
    (0.8, -0.1, 2.3, 1.7),
    (1.3, 0.2, -1.0, 3.1),
    (0.9, 0.0, 4.0, 2.2),
]


def _toscalar(backend_name, x):
    if backend_name == "numpy":
        return x
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


# --- value parity across backends ----------------------------------------------
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("mop_id,mop", _MOPS, ids=[m[0] for m in _MOPS])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, mop_id, mop, method):
    for R, z, phi, t in _POINTS:
        ref = float(getattr(mop, method)(R, z, phi=phi, t=t))
        got = float(
            getattr(mop, method)(
                _toscalar(backend_name, R),
                _toscalar(backend_name, z),
                phi=_toscalar(backend_name, phi),
                t=t,
            )
        )
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- eager backend arrays -------------------------------------------------------
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_returns_backend_array(backend_name):
    R, z, phi, t = 1.0, 0.05, 0.2, 1.7
    ref = evaluateRforces(_MOP3, R, z, phi=phi, t=t)
    got = evaluateRforces(
        _MOP3,
        _toscalar(backend_name, R),
        _toscalar(backend_name, z),
        phi=_toscalar(backend_name, phi),
        t=t,
    )
    if backend_name == "jax":
        assert isinstance(got, jax.Array)
    else:
        assert isinstance(got, torch.Tensor)
    numpy.testing.assert_allclose(float(got), ref, rtol=1e-12, atol=1e-14)


# --- the exact gap: traced jacfwd / jit over evaluateRforces --------------------
@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_jacfwd_jit_finite():
    R0, z0, phi0, t0 = 1.0, 0.05, 0.2, 1.7
    ref = evaluateRforces(_MOP3, R0, z0, phi=phi0, t=t0)
    # jit over evaluateRforces
    jitted = float(
        jax.jit(lambda R, z, p: evaluateRforces(_MOP3, R, z, phi=p, t=t0))(
            jnp.asarray(R0), jnp.asarray(z0), jnp.asarray(phi0)
        )
    )
    numpy.testing.assert_allclose(jitted, ref, rtol=1e-12, atol=1e-14)
    # jacfwd wrt R and wrt phi return finite (previously TracerArrayConversion)
    gR = jax.jacfwd(
        lambda R: evaluateRforces(
            _MOP3, R, jnp.asarray(z0), phi=jnp.asarray(phi0), t=t0
        )
    )(jnp.asarray(R0))
    gphi = jax.jacfwd(
        lambda p: evaluateRforces(_MOP3, jnp.asarray(R0), jnp.asarray(z0), phi=p, t=t0)
    )(jnp.asarray(phi0))
    assert numpy.isfinite(float(gR))
    assert numpy.isfinite(float(gphi))


# --- stringent grad-vs-FD (h-convergence) ---------------------------------------
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_vs_fd_R(backend_name):
    # d(Rforce)/dR h-converges to a central finite difference.
    R0, z0, phi0, t0 = 1.0, 0.05, 0.2, 1.7

    def rforce_np(R):
        return float(evaluateRforces(_MOP3, R, z0, phi=phi0, t=t0))

    if backend_name == "jax":
        ad = float(
            jax.grad(
                lambda R: evaluateRforces(
                    _MOP3, R, jnp.asarray(z0), phi=jnp.asarray(phi0), t=t0
                )
            )(jnp.asarray(R0))
        )
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        evaluateRforces(
            _MOP3, Rt, torch.tensor(z0), phi=torch.tensor(phi0), t=t0
        ).backward()
        ad = float(Rt.grad)
    best = min(
        abs(ad - (rforce_np(R0 + h) - rforce_np(R0 - h)) / (2 * h))
        for h in (1e-3, 1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-7, best


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_vs_fd_phi(backend_name):
    # A phi-dependent quadratic loss (exercises the cos/sin(phi) field paths)
    # h-converges to a central finite difference.
    R0, z0, phi0, t0 = 1.0, 0.05, 0.2, 1.7

    def loss_np(phi):
        fr = float(evaluateRforces(_MOP3, R0, z0, phi=phi, t=t0))
        fp = float(evaluatephitorques(_MOP3, R0, z0, phi=phi, t=t0))
        return fr**2 + 0.5 * fp**2

    if backend_name == "jax":

        def loss(phi):
            fr = evaluateRforces(_MOP3, jnp.asarray(R0), jnp.asarray(z0), phi=phi, t=t0)
            fp = evaluatephitorques(
                _MOP3, jnp.asarray(R0), jnp.asarray(z0), phi=phi, t=t0
            )
            return fr**2 + 0.5 * fp**2

        ad = float(jax.grad(loss)(jnp.asarray(phi0)))
    else:
        pt = torch.tensor(phi0, requires_grad=True)
        fr = evaluateRforces(_MOP3, torch.tensor(R0), torch.tensor(z0), phi=pt, t=t0)
        fp = evaluatephitorques(_MOP3, torch.tensor(R0), torch.tensor(z0), phi=pt, t=t0)
        (fr**2 + 0.5 * fp**2).backward()
        ad = float(pt.grad)
    best = min(
        abs(ad - (loss_np(phi0 + h) - loss_np(phi0 - h)) / (2 * h))
        for h in (1e-3, 1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-7, best
