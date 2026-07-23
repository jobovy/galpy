###############################################################################
# test_backend_anyspherical.py: multi-backend tests for AnySphericalPotential,
# whose enclosed-mass (_rawmass) and outer-potential (_revaluate tail) integrals
# of the USER-SUPPLIED density profile were numpy-only (scipy.integrate.quad
# with a scalar limit). The class now follows the data: a numpy coord keeps the
# scipy path (byte-identical), a jax/torch coord routes to in-backend fixed-order
# Gauss-Legendre (galpy.backend.quadrature), so the force / potential / 2nd
# derivative become jit- and grad-safe under a trace.
#
# This proves: eager jax/torch return backend arrays matching numpy; jax.jacfwd
# and jax.jit over evaluateRforces are finite (THE gap that defined this
# migration); and the force gradient w.r.t. R and w.r.t. the amp parameter
# h-converges to a central finite difference (a stringent grad-vs-FD check, not
# finite-and-nonzero). Backends that are not installed self-skip, so this is
# green on numpy alone.
###############################################################################
import warnings

import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import (
    AnySphericalPotential,
    evaluatePotentials,
    evaluater2derivs,
    evaluateRforces,
)
from galpy.util._optional_deps import _APY_LOADED

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


# A unitless density that broadcasts elementwise (so the backend Gauss-Legendre
# path can call it on the whole node array) -- this is the backend contract the
# user opts into by passing backend coords. Two amplitudes exercise the linear
# amp scaling of the force.
def _dens(r):
    return 0.64 / r / (1.0 + r) ** 3


POTS = [
    AnySphericalPotential(amp=1.3, dens=_dens),
    AnySphericalPotential(amp=0.7, dens=_dens),
]
POT_IDS = ["amp1.3", "amp0.7"]

# Scalar radii (the numpy force path is scalar-only: an array r collapses to
# r[0]) away from r == 0.
_RZ = [(0.3, 0.0), (0.7, 0.2), (1.1, 0.3), (2.5, 0.5), (5.0, 0.1)]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_backend_value_parity(backend_name, pot):
    # Eager jax/torch return a backend array whose value matches the numpy
    # (scipy) path to Gauss-Legendre accuracy on this smooth density.
    for R0, z0 in _RZ:
        R = _asarray(backend_name, R0)
        z = _asarray(backend_name, z0)
        for fn in (evaluateRforces, evaluatePotentials, evaluater2derivs):
            got = fn(pot, R, z)
            assert backend_name in type(got).__module__, (fn.__name__, type(got))
            ref = fn(pot, R0, z0)
            numpy.testing.assert_allclose(
                as_numpy(got),
                ref,
                rtol=1e-8,
                atol=1e-9,
                err_msg=f"{fn.__name__} R={R0} ({backend_name})",
            )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
def test_jacfwd_and_jit_rforces_finite(pot):
    # THE gap: jax.jacfwd / jax.jit over evaluateRforces must be finite (the old
    # numpy.atleast_1d + scipy.integrate.quad internals crashed under a trace).
    for R0, z0 in _RZ:
        z = jnp.asarray(z0)

        def rf(R):
            return evaluateRforces(pot, R, z)

        g = jax.jacfwd(rf)(jnp.asarray(R0))
        assert bool(jnp.isfinite(g)), f"jacfwd non-finite at R={R0}"
        jv = jax.jit(rf)(jnp.asarray(R0))
        assert bool(jnp.isfinite(jv)), f"jit non-finite at R={R0}"
        numpy.testing.assert_allclose(
            float(jv), evaluateRforces(pot, R0, z0), rtol=1e-8
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd_R(backend_name):
    # d(Rforce)/dR via autodiff vs central FD (h-convergence, not finite-nonzero).
    pot = POTS[0]
    z0 = 0.3
    for R0 in (0.7, 1.1, 2.5):
        if backend_name == "jax":
            ad = float(
                jax.jacfwd(lambda R: evaluateRforces(pot, R, jnp.asarray(z0)))(
                    jnp.asarray(R0)
                )
            )
        else:
            Rt = torch.tensor(R0, requires_grad=True)
            f = evaluateRforces(pot, Rt, torch.tensor(z0))
            (g,) = torch.autograd.grad(f, Rt)
            ad = float(g)
        assert not numpy.isnan(ad)
        h = 1e-5
        fp = float(evaluateRforces(pot, R0 + h, z0))
        fm = float(evaluateRforces(pot, R0 - h, z0))
        numpy.testing.assert_allclose(
            ad,
            (fp - fm) / (2.0 * h),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"R={R0} ({backend_name})",
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd_amp(backend_name):
    # d(Rforce)/d(amp) via autodiff vs central FD -- the density integral is
    # differentiable through the amp parameter.
    R0, z0, amp0 = 1.1, 0.3, 1.3
    if backend_name == "jax":
        ad = float(
            jax.jacfwd(
                lambda a: evaluateRforces(
                    AnySphericalPotential(amp=a, dens=_dens),
                    jnp.asarray(R0),
                    jnp.asarray(z0),
                )
            )(jnp.asarray(amp0))
        )
    else:
        at = torch.tensor(amp0, requires_grad=True)
        f = evaluateRforces(
            AnySphericalPotential(amp=at, dens=_dens),
            torch.tensor(R0),
            torch.tensor(z0),
        )
        (g,) = torch.autograd.grad(f, at)
        ad = float(g)
    h = 1e-5
    fp = float(evaluateRforces(AnySphericalPotential(amp=amp0 + h, dens=_dens), R0, z0))
    fm = float(evaluateRforces(AnySphericalPotential(amp=amp0 - h, dens=_dens), R0, z0))
    numpy.testing.assert_allclose(ad, (fp - fm) / (2.0 * h), rtol=1e-6, atol=1e-8)


# ==================== units-based density on the backend ================== #
# A units-based density runs through astropy Quantity arithmetic, which strips a
# jax/torch quadrature node to numpy (and emits a numpy-2 __array__ deprecation).
# The backend integrands then did `numpy * Tensor`, which RAISES. Such a density
# is inherently non-differentiable, so it is now evaluated on the numpy node and
# the result anchored back on the backend (AnySphericalPotential._backend_dens).
# Assert: no raise, a backend array, numpy value parity, and -- crucially, since
# build.yml runs test_backend*.py under numpy with `-W error` -- NO Deprecation/
# FutureWarning escapes the units evaluation.
@pytest.mark.skipif(not _APY_LOADED, reason="astropy not installed")
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_units_density_backend_value_parity(backend_name):
    from astropy import units

    from galpy.util import conversion

    ro, vo = 8.0, 220.0

    def dens_units(r):
        return (
            0.64
            / r
            / (1.0 + r) ** 3
            * conversion.dens_in_msolpc3(vo, ro)
            * units.Msun
            / units.pc**3
        )

    pot = AnySphericalPotential(dens=dens_units, ro=ro, vo=vo)
    assert pot._dens_needs_numpy
    for R0, z0 in _RZ:
        R = _asarray(backend_name, R0)
        z = _asarray(backend_name, z0)
        for fn in (evaluatePotentials, evaluateRforces, evaluater2derivs):
            ref = fn(pot, R0, z0, use_physical=False)
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                warnings.simplefilter("error", FutureWarning)
                got = fn(pot, R, z, use_physical=False)
            assert backend_name in type(got).__module__, (fn.__name__, type(got))
            numpy.testing.assert_allclose(
                as_numpy(got),
                ref,
                rtol=1e-8,
                atol=1e-9,
                err_msg=f"{fn.__name__} R={R0} ({backend_name})",
            )
