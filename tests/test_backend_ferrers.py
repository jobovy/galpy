###############################################################################
# test_backend_ferrers.py: backend tests for FerrersPotential.
#
# Ferrers' potential / force / 2nd-derivative methods evaluate a
# scipy.integrate.quad integral over concrete inputs, so they produce identical
# VALUES under numpy / jax / torch (eager evaluation with concrete arrays) but
# are NOT jax-traceable through the quadrature (that is the Pspecial-deferred
# path). This module therefore:
#   1. checks numpy / jax / torch value parity for every migrated compute method
#      (this drives the traced-path branch of the per-instance md5 force cache,
#      `if xp is not numpy: return self._xyzforces(...)`),
#   2. checks the fully-arithmetic _dens (the only autodiff-friendly method):
#      value parity AND that its eager-both-branch xp.where guard gives a finite
#      gradient on the m2 >= 1 (outside-ellipsoid) dead branch instead of NaN.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import FerrersPotential

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
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

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# Triaxial, rotated (pa) and rotating (omegab) so the cos/sin de-rotation and the
# phi-dependent Hessian branches are all exercised.
_FE = FerrersPotential(amp=1.3, a=1.5, n=2, b=0.9, c=0.7, pa=0.3, omegab=1.0)

# Every migrated compute method (all scalar in their quadrature, so probed at a
# scalar point).
_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
    "_dens",
]
_R0, _Z0, _PHI0, _T0 = 1.2, 0.3, 0.4, 0.0


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, method):
    # numpy / jax / torch agree at a scalar (R, z, phi, t), all on the active
    # backend (the rotated cos/sin de-rotation needs a same-namespace phi/t under
    # torch). The traced backends take the `if xp is not numpy` direct-compute
    # path of _cached_xyzforces (no md5 cache) for the force methods.
    ref = float(
        getattr(_FE, method)(
            numpy.asarray(_R0), numpy.asarray(_Z0), numpy.asarray(_PHI0), _T0
        )
    )
    got = float(
        _tonumpy(
            getattr(_FE, method)(
                _asarray(backend_name, _R0),
                _asarray(backend_name, _Z0),
                _asarray(backend_name, _PHI0),
                _asarray(backend_name, _T0),
            )
        )
    )
    numpy.testing.assert_allclose(
        got, ref, rtol=1e-11, atol=1e-13, err_msg=f"Ferrers.{method} ({backend_name})"
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_dens_inside_outside_value_parity(backend_name):
    # _dens is branch-free under jax/torch (eager xp.where on m2 < 1). Check both
    # an inside-ellipsoid (m2 < 1) and an outside (m2 >= 1, value 0) point.
    for R0, z0, expect_zero in [(0.3, 0.05, False), (3.0, 0.5, True)]:
        ref = float(_FE._dens(numpy.asarray(R0), numpy.asarray(z0), _PHI0, _T0))
        got = float(
            _tonumpy(
                _FE._dens(
                    _asarray(backend_name, R0), _asarray(backend_name, z0), _PHI0, _T0
                )
            )
        )
        if expect_zero:
            assert ref == 0.0
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dens_grad_inside_vs_finite_difference(backend_name):
    # _dens is the one fully-arithmetic (autodiff-friendly) method. Inside the
    # ellipsoid AD(d_dens/dR) matches central FD.
    R0, z0 = 0.3, 0.05
    eps = 1e-6
    fd = (
        float(_FE._dens(numpy.asarray(R0 + eps), numpy.asarray(z0), _PHI0, _T0))
        - float(_FE._dens(numpy.asarray(R0 - eps), numpy.asarray(z0), _PHI0, _T0))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: _FE._dens(R, jnp.asarray(z0), jnp.asarray(_PHI0), _T0))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        _FE._dens(
            R, torch.tensor(z0, dtype=torch.float64), torch.tensor(_PHI0), _T0
        ).backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dens_grad_outside_is_finite(backend_name):
    # On the m2 >= 1 (outside) dead branch the guarded base (1 - m2/a**2 -> 1)
    # must keep the reverse-mode gradient finite (0), not NaN, for the
    # non-integer-safe power. The where selects the 0.0 value.
    R0, z0 = 3.0, 0.5  # outside the ellipsoid
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: _FE._dens(R, jnp.asarray(z0), jnp.asarray(_PHI0), _T0))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = _FE._dens(
            R, torch.tensor(z0, dtype=torch.float64), torch.tensor(_PHI0), _T0
        )
        y.backward()
        ad = 0.0 if R.grad is None else float(R.grad)
    assert numpy.isfinite(ad), f"Ferrers _dens outside grad not finite ({backend_name})"
    assert ad == 0.0
