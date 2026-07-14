###############################################################################
# test_backend_interprz.py: multi-backend tests for interpRZPotential.
#
# interpRZPotential interpolates precomputed R-z grids of the potential, the two
# forces, the three interpolated 2nd derivatives, and the density with scipy
# ``RectBivariateSpline``. The numpy code path keeps calling the scipy splines'
# ``.ev`` exactly as before (byte-identical); the jax/torch path evaluates the
# SAME tensor-product piecewise polynomial (via ``rect_bivariate_to_ppoly`` +
# ``eval_rect_ppoly``: searchsorted + 2D Horner in namespace ops), so values
# agree with scipy to ~1 ulp and the potential is exactly autodifferentiable.
#
# For every backend this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-9) for the seven
#      interpolated 2D methods, on an interior grid that includes negative z
#      (the zsym odd-force branch) and both logR conventions;
#   2. jit + jacfwd over the public evaluatePotentials/evaluateRforces on backend
#      (R,z) return finite (traced safety);
#   3. autodiff of the interpolated force/potential matches central finite
#      differences computed on the numpy/scipy-spline path (grad-vs-FD).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import (
    MiyamotoNagaiPotential,
    evaluatePotentials,
    evaluateRforces,
    interpRZPotential,
)

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

_INTERP_KW = dict(
    interpPot=True,
    interpRforce=True,
    interpzforce=True,
    interpR2deriv=True,
    interpz2deriv=True,
    interpRzderiv=True,
    interpDens=True,
    zsym=True,
)


def _build(logR):
    base = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1)
    rgrid = (numpy.log(0.05), numpy.log(16.0), 21) if logR else (0.05, 16.0, 21)
    return interpRZPotential(
        RZPot=base, rgrid=rgrid, zgrid=(0.0, 1.0, 21), logR=logR, **_INTERP_KW
    )


# Built once (grid construction is the expensive part).
CASES = [_build(True), _build(False)]
CASE_IDS = ["logR", "linR"]

_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]

# Interior query grid; negative z exercises the zsym odd (sign-flipped) branch.
_RS = numpy.array([0.3, 0.8, 1.0, 1.3, 2.5, 8.0])
_ZS = numpy.array([-0.4, -0.1, 0.05, 0.15, 0.3, 0.6])


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _METHODS:
        ref = numpy.asarray(getattr(pot, method)(_RS, _ZS))
        got = as_numpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-9, atol=1e-11, err_msg=f"{CASE_IDS}.{method}"
        )


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot):
    # The public evaluate* (through the unit decorators and _amp) agree too.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for fn in (evaluatePotentials, evaluateRforces):
        ref = numpy.asarray(fn(pot, _RS, _ZS))
        got = as_numpy(fn(pot, R, z))
        numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
def test_jax_traced_finite(pot):
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    Rj = jnp.asarray(_RS)
    zj = jnp.asarray(_ZS)
    for fn in (evaluatePotentials, evaluateRforces):
        jitted = jax.jit(lambda R_, z_: fn(pot, R_, z_))(Rj, zj)
        assert numpy.all(numpy.isfinite(as_numpy(jitted)))
        jac = jax.jacfwd(lambda R_: fn(pot, R_, zj))(Rj)
        assert numpy.all(numpy.isfinite(as_numpy(jac)))


# One interior (R,z) point; grad w.r.t. R (Rforce) and z (potential).
_FD_R0, _FD_Z0 = 1.15, 0.22


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_rforce_vs_fd(backend_name, pot):
    # d(Rforce)/dR: AD vs central FD on the numpy/scipy-spline path.
    eps = 1e-5

    def f_np(Rv):
        return float(evaluateRforces(pot, numpy.asarray(Rv), numpy.asarray(_FD_Z0)))

    fd = (f_np(_FD_R0 + eps) - f_np(_FD_R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda Rv: evaluateRforces(pot, Rv, jnp.asarray(_FD_Z0)))(
                jnp.asarray(_FD_R0)
            )
        )
    else:
        R = torch.tensor(_FD_R0, dtype=torch.float64, requires_grad=True)
        y = evaluateRforces(pot, R, torch.tensor(_FD_Z0, dtype=torch.float64))
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_pot_wrt_z_vs_fd(backend_name, pot):
    # d(Pot)/dz: AD vs central FD; -d(Pot)/dz should also track zforce.
    eps = 1e-5

    def f_np(zv):
        return float(evaluatePotentials(pot, numpy.asarray(_FD_R0), numpy.asarray(zv)))

    fd = (f_np(_FD_Z0 + eps) - f_np(_FD_Z0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda zv: evaluatePotentials(pot, jnp.asarray(_FD_R0), zv))(
                jnp.asarray(_FD_Z0)
            )
        )
    else:
        zt = torch.tensor(_FD_Z0, dtype=torch.float64, requires_grad=True)
        y = evaluatePotentials(pot, torch.tensor(_FD_R0, dtype=torch.float64), zt)
        y.backward()
        ad = float(zt.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)
