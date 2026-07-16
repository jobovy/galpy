###############################################################################
# test_backend_anyaxisymdisk.py: multi-backend tests for
# AnyAxisymmetricRazorThinDiskPotential.
#
# This potential integrates an arbitrary surface density Sigma(R) against the
# razor-thin-disk Green's function (complete elliptic integrals K/E). The numpy
# path keeps scipy's adaptive quad + scipy.special.ellipk/ellipe and is
# byte-identical; a jax/torch input routes to the backend split Gauss-Legendre
# quadrature (galpy.backend.quadrature) with the backend elliptic fallback
# (galpy.backend.special.ellipk/ellipe), so the FORCES are jit/grad-safe.
#
# The eager-only gap this closes: before migration the forces returned a bare
# python float under a forced backend, so downstream ``xp.sqrt`` (e.g. vcirc)
# crashed ("sqrt(): argument must be Tensor, not float") and jax.jacfwd could
# not trace them.
#
# Design note: a PLAIN concrete backend scalar reuses scipy's accurate value
# (wrapped as a backend array); native GL runs only when a gradient is actually
# taken (a tracer, or a grad-tracking torch tensor). GL cannot resolve the
# small-z sheet structure the derivative FD-probes hit, so the concrete path
# stays scipy-accurate while the differentiable path stays native.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import get_namespace, is_backend_array
from galpy.potential import (
    AnyAxisymmetricRazorThinDiskPotential,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluateRzderivs,
    evaluatez2derivs,
    evaluatezforces,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture. The -W error marks are the
# coverage-shard trap guard: a numpy.<ufunc> on a torch tensor emits a
# DeprecationWarning, so the namespace-agnostic surface density below must never
# fall back to bare numpy on the backend path.
pytestmark = [
    pytest.mark.backend_managed,
    pytest.mark.filterwarnings("error::DeprecationWarning"),
    pytest.mark.filterwarnings("error::FutureWarning"),
]

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


def _surfdens(R):
    # namespace-agnostic (numpy / jax / torch): differentiable on every backend
    return 1.5 * get_namespace(R).exp(-3.0 * R)


# One instance, shared: __init__ integrates surfdens once (scipy) for _pot_zero.
_POT = AnyAxisymmetricRazorThinDiskPotential(surfdens=_surfdens)

# Physical (R, z) grid with z != 0 for the differentiable checks; z == 0 is
# covered by the value-parity / scalar tests (its force is exact by symmetry).
_RS = [0.5, 0.9, 1.3, 2.1]
_ZS = [0.15, 0.3, 0.25, 0.4]


def _scalar(backend_name, x):
    if backend_name == "numpy":
        return numpy.float64(x)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_force_returns_backend_array_matching_numpy(backend_name):
    # Eager backend force is a backend array (the eager-only-gap fix) whose value
    # matches the numpy/scipy reference, INCLUDING the z == 0 plane (the vcirc /
    # normalize crash site).
    for R in _RS + [1.0]:
        for z in _ZS + [0.0]:
            refR = float(evaluateRforces(_POT, numpy.float64(R), numpy.float64(z)))
            refz = float(evaluatezforces(_POT, numpy.float64(R), numpy.float64(z)))
            Rb, zb = _scalar(backend_name, R), _scalar(backend_name, z)
            gotR = evaluateRforces(_POT, Rb, zb)
            gotz = evaluatezforces(_POT, Rb, zb)
            assert is_backend_array(gotR) and is_backend_array(gotz)
            numpy.testing.assert_allclose(float(gotR), refR, rtol=1e-10, atol=1e-12)
            numpy.testing.assert_allclose(float(gotz), refz, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_traced_potential_and_2ndderivs_backend(backend_name):
    # The backend GL path for the POTENTIAL and the three SECOND derivatives
    # (_evaluate_gl / _R2deriv_gl / _z2deriv_gl / _Rzderiv_gl) is taken only for a TRACER
    # (jax) or a requires_grad tensor (torch) -- a CONCRETE backend scalar reuses scipy
    # via _bk_dispatch. #1121 landed these ~36 GL lines with NO traced backend test (only
    # the forces were traced), so feat/backends coverage jumped 15->51. Trace over each
    # (jit for jax, requires_grad for torch) so the GL branch runs, and check the value
    # matches the numpy/scipy reference (z != 0: the 2nd derivs are singular in the plane).
    R0, z0 = 1.3, 0.25
    for fn in (
        evaluatePotentials,
        evaluateR2derivs,
        evaluatez2derivs,
        evaluateRzderivs,
    ):
        ref = float(fn(_POT, numpy.float64(R0), numpy.float64(z0)))
        if backend_name == "jax":
            got = jax.jit(lambda R, fn=fn: fn(_POT, R, jnp.asarray(z0)))(
                jnp.asarray(R0)
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            got = fn(_POT, Rt, torch.tensor(z0, dtype=torch.float64))
        assert is_backend_array(got), (fn.__name__, backend_name)
        numpy.testing.assert_allclose(
            float(got), ref, rtol=1e-6, atol=1e-8, err_msg=fn.__name__
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_traced_force_finite():
    # The exact failure that defined this gap: jax.jacfwd / jax.jit over the
    # forces must trace and return finite (they previously returned a bare float).
    R0, z0 = jnp.asarray(1.2), jnp.asarray(0.3)
    rf = lambda R: evaluateRforces(_POT, R, z0)
    zf = lambda z: evaluatezforces(_POT, R0, z)
    gR = jax.jacfwd(rf)(R0)
    jR = jax.jit(rf)(R0)
    gz = jax.jacfwd(zf)(z0)
    assert jnp.isfinite(gR) and jnp.isfinite(jR) and jnp.isfinite(gz)
    # jit value matches the eager scipy reference (z>0 -> GL is machine-accurate)
    numpy.testing.assert_allclose(
        float(jR), float(evaluateRforces(_POT, numpy.float64(1.2), numpy.float64(0.3)))
    )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_default_surfdens_jit_safe():
    # The DEFAULT surfdens is now backend-agnostic (was a bare ``numpy.exp`` that calls
    # __array__ on a jit tracer), so a DEFAULT-constructed disk -- how the potential-zoo
    # differentiability sweep builds it -- is jit/jacfwd-safe too. numpy path unchanged
    # (is_backend_array guard: numpy / python-float / Quantity R keep numpy.exp).
    pot = AnyAxisymmetricRazorThinDiskPotential()  # default surfdens
    R0, z0 = 1.1, 0.3
    ref = float(evaluateRforces(pot, numpy.float64(R0), numpy.float64(z0)))
    rf = lambda R: evaluateRforces(pot, R, jnp.asarray(z0))
    jR = float(jax.jit(rf)(jnp.asarray(R0)))
    gR = float(jax.jacfwd(rf)(jnp.asarray(R0)))
    assert numpy.isfinite(jR) and numpy.isfinite(gR)
    numpy.testing.assert_allclose(jR, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd(backend_name):
    # d(Rforce)/dR via autodiff vs central finite differences, h-converging (not a
    # finite-and-nonzero check), at physical z > 0.
    for R0, z0 in zip(_RS, _ZS):
        znum = numpy.float64(z0)

        def fd(h):
            fp = float(evaluateRforces(_POT, numpy.float64(R0 + h), znum))
            fm = float(evaluateRforces(_POT, numpy.float64(R0 - h), znum))
            return (fp - fm) / (2.0 * h)

        if backend_name == "jax":
            ad = float(
                jax.grad(lambda R: evaluateRforces(_POT, R, jnp.asarray(z0)))(
                    jnp.asarray(R0)
                )
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            evaluateRforces(_POT, Rt, torch.tensor(z0, dtype=torch.float64)).backward()
            ad = float(Rt.grad)
        # central FD converges O(h^2); the two finest h bracket the AD tightly
        rels = [abs((ad - fd(h)) / ad) for h in (1e-4, 1e-5)]
        assert min(rels) < 1e-7, f"R={R0} z={z0} ({backend_name}): rels={rels}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_zforce_grad_vs_fd(backend_name):
    # d(zforce)/dz via autodiff vs central FD, h-converging, at physical z > 0.
    for R0, z0 in zip(_RS, _ZS):
        Rnum = numpy.float64(R0)

        def fd(h):
            fp = float(evaluatezforces(_POT, Rnum, numpy.float64(z0 + h)))
            fm = float(evaluatezforces(_POT, Rnum, numpy.float64(z0 - h)))
            return (fp - fm) / (2.0 * h)

        if backend_name == "jax":
            ad = float(
                jax.grad(lambda z: evaluatezforces(_POT, jnp.asarray(R0), z))(
                    jnp.asarray(z0)
                )
            )
        else:
            zt = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
            evaluatezforces(_POT, torch.tensor(R0, dtype=torch.float64), zt).backward()
            ad = float(zt.grad)
        rels = [abs((ad - fd(h)) / ad) for h in (1e-4, 1e-5)]
        assert min(rels) < 1e-6, f"R={R0} z={z0} ({backend_name}): rels={rels}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_vcirc_no_longer_crashes(backend_name):
    # Regression for the reported eager crash: vcirc = xp.sqrt(R * -Rforce) fed a
    # bare python float under a forced backend. With the migrated (backend-array)
    # force it evaluates and matches the numpy value.
    pot = AnyAxisymmetricRazorThinDiskPotential(surfdens=_surfdens)
    pot.normalize(1.0)
    ref = float(pot.vcirc(1.3, use_physical=False))
    got = float(pot.vcirc(_scalar(backend_name, 1.3), use_physical=False))
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)
