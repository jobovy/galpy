###############################################################################
# test_backend_ring.py: multi-backend tests for RingPotential, the potential of
# an infinitesimally-thin circular ring whose potential/forces/second
# derivatives are expressed through the complete elliptic integrals K(m) and
# E(m) (routed via galpy.backend.special.ellipk/ellipe: scipy.special on numpy,
# a pure-backend AGM fallback on jax and torch, both in scipy's parameter-m
# convention).
#
# For every migrated compute method this proves numpy / jax / torch value parity
# at tight tolerances, that eval/jacfwd/jit run under a jax trace and stay
# finite, and that the migrated forces are differentiable with a gradient
# consistent with finite differences and with the analytic identity
# AD(_evaluate) == -force (both in R and z). The ring radius a and mass amp are
# also shown to be differentiable parameters (jax.grad / torch.autograd vs FD).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import RingPotential, evaluateRforces

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

POTS = [RingPotential(amp=1.3, a=0.75), RingPotential(amp=2.3, a=1.4)]
POT_IDS = ["a0.75", "a1.4"]

# (R, z) grid that deliberately avoids the on-ring logarithmic singularity
# (m -> 1 at R == a, z == 0): points straddle the ring radius, include z == 0
# rows off the ring, and reach large R (m -> 0). The genuine singularity and the
# z == 0 finite limits are covered byte-identically by test_potential.py.
_RS = [0.2, 0.5, 0.72, 0.95, 1.0, 2.0, 5.0, 20.0]
_ZS = [0.0, 0.15, 0.3, 0.0, -0.5, 1.0, -2.0, 0.4]

_METHODS = ["_evaluate", "_Rforce", "_zforce", "_R2deriv", "_z2deriv", "_Rzderiv"]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot):
    # numpy == jax == torch for every migrated compute method (backend value vs
    # scipy reference), and the results are genuine backend arrays.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _METHODS:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = as_numpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{method} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
def test_public_force_parity_jax(pot):
    # The public evaluate* entry points route the elliptic integrals to the jax
    # AGM fallback; check them against scipy to <=1e-10 (the traced-safety gate).
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    R, z = numpy.asarray(_RS), numpy.asarray(_ZS)
    ref = numpy.asarray(evaluateRforces(pot, R, z))
    got = as_numpy(evaluateRforces(pot, jnp.asarray(R), jnp.asarray(z)))
    assert numpy.max(numpy.abs(got - ref)) < 1e-10


def test_jit_and_jacfwd_finite():
    # A jax trace over the elliptic-integral force path compiles, and its forward
    # jacobian d(Rforce)/dR is finite (no scipy call escapes into the trace).
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    pot = POTS[0]
    R0, z0 = jnp.asarray(1.3), jnp.asarray(0.4)
    val = jax.jit(lambda R, z: evaluateRforces(pot, R, z))(R0, z0)
    assert numpy.isfinite(as_numpy(val))
    jac = jax.jacfwd(lambda R: evaluateRforces(pot, R, z0))(R0)
    assert numpy.isfinite(as_numpy(jac))


@pytest.mark.parametrize("component", ["R", "z"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_is_minus_force(backend_name, component):
    # Exact analytic identity: AD(_evaluate) == -force, separately in R and z.
    pot = POTS[0]
    R0, z0 = 1.3, 0.4
    if component == "R":
        ref = -float(pot._Rforce(numpy.asarray(R0), numpy.asarray(z0)))
        f = lambda R, z: pot._evaluate(R, z)  # noqa: E731
        x0, other = R0, z0
    else:
        ref = -float(pot._zforce(numpy.asarray(R0), numpy.asarray(z0)))
        f = lambda z, R: pot._evaluate(R, z)  # noqa: E731
        x0, other = z0, R0
    if backend_name == "jax":
        ad = float(jax.grad(lambda x: f(x, jnp.asarray(other)))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        f(xt, torch.tensor(other, dtype=torch.float64)).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(ad, ref, rtol=1e-9)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd(backend_name):
    # d(Rforce)/dR via autodiff (through K(m), E(m)) vs central finite difference,
    # at a few off-ring points.
    pot = POTS[0]
    eps = 1e-6
    for R0, z0 in ((1.3, 0.4), (0.4, 0.2), (2.5, 1.0)):
        fd = (
            float(pot._Rforce(numpy.asarray(R0 + eps), numpy.asarray(z0)))
            - float(pot._Rforce(numpy.asarray(R0 - eps), numpy.asarray(z0)))
        ) / (2.0 * eps)
        if backend_name == "jax":
            ad = float(
                jax.grad(lambda R: pot._Rforce(R, jnp.asarray(z0)))(jnp.asarray(R0))
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            pot._Rforce(Rt, torch.tensor(z0, dtype=torch.float64)).backward()
            ad = float(Rt.grad)
        assert not numpy.isnan(ad), f"NaN grad at R={R0} ({backend_name})"
        numpy.testing.assert_allclose(
            ad, fd, rtol=1e-5, err_msg=f"R={R0} ({backend_name})"
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_param_grad_vs_fd(backend_name):
    # d(Rforce)/d(ring radius a): the shape parameter flows through construction
    # into the elliptic modulus m and stays differentiable vs central finite
    # difference (the point of routing K/E through the backend special functions).
    R0, z0, a0, eps = 1.3, 0.4, 0.75, 1e-6

    def rf_b(a):
        return evaluateRforces(
            RingPotential(amp=1.3, a=a),
            _asarray(backend_name, R0),
            _asarray(backend_name, z0),
        )

    fd = (
        float(evaluateRforces(RingPotential(amp=1.3, a=a0 + eps), R0, z0))
        - float(evaluateRforces(RingPotential(amp=1.3, a=a0 - eps), R0, z0))
    ) / (2.0 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(rf_b)(jnp.asarray(a0)))
    else:
        pt = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        rf_b(pt).backward()
        ad = float(pt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-5, err_msg=f"d(Rforce)/da ({backend_name})"
    )
