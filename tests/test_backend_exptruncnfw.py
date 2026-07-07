###############################################################################
# test_backend_exptruncnfw.py: multi-backend tests for ExpTruncNFWPotential,
# the exponentially-truncated NFW profile whose closed-form enclosed mass and
# outer-potential integral go through the exponential integral E_1 (routed via
# galpy.backend.special.exp1: scipy on numpy, -expi(-x) on jax, a pure-backend
# Lentz/series fallback on torch).
#
# For every migrated compute method this proves numpy / jax / torch value parity
# at the existing tolerances, and that the migrated radial force (which passes
# through exp1) is differentiable with a gradient consistent with the potential
# (jax.grad / torch.autograd vs finite differences and vs the analytic identity
# AD(_evaluate) == -_Rforce). The small-r Taylor branch and the closed-form
# branch of _F are both exercised (via a grid that straddles _small_r_thresh).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import ExpTruncNFWPotential

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

# amp= and mass= constructions exercise the two _amp conventions (mass= also
# bakes _amp into _ddensdr/_d2densdr2). a=1.5, rc=8 => _small_r_thresh=1.5e-3.
POTS = [
    ExpTruncNFWPotential(amp=1.3, a=1.5, rc=8.0),
    ExpTruncNFWPotential(mass=5.0, a=1.3, rc=6.0),
]
POT_IDS = ["amp", "mass"]

# Radial grid straddling _small_r_thresh (~1.5e-3): the first two points hit the
# Taylor-series branch of _F, the rest the closed (E_1) form. r == 0 is excluded
# (its finite-limit substitution is checked in test_potential.py).
_RS = [5.0e-4, 1.0e-3, 0.3, 0.8, 1.5, 4.0, 12.0]
_ZS = [0.2, 0.35, 0.15, 0.4, 0.25, 0.3, 0.5]

_THREE_D = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]
_ONE_D = ["_revaluate", "_rforce", "_r2deriv", "_rdens", "_F", "_G", "_mass"]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_threed_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _THREE_D:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = as_numpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{method} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_oned_value_parity(backend_name, pot):
    r = _asarray(backend_name, _RS)
    rnp = numpy.asarray(_RS)
    for method in _ONE_D:
        ref = numpy.asarray(getattr(pot, method)(rnp))
        got = as_numpy(getattr(pot, method)(r))
        # rtol 1e-11 (not 1e-12): _r2deriv = 4*pi*rho - 2*F/r^3 subtracts two large
        # terms at the smallest grid radius (r=5e-4), a cancellation that amplifies
        # the last-ULP difference between scipy.exp1 (numpy) and the native/fallback
        # exp1 (jax/torch) to ~7e-13 -- hardware-dependent around a 1e-12 cut.
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-11, atol=1e-13, err_msg=f"{method} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_density_derivs_and_mass_at_infinity(backend_name, pot):
    r = _asarray(backend_name, _RS)
    rnp = numpy.asarray(_RS)
    for method in ("_ddensdr", "_d2densdr2"):
        ref = numpy.asarray(getattr(pot, method)(rnp))
        got = as_numpy(getattr(pot, method)(r))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{method} ({backend_name})"
        )
    # _ddenstwobetadr dispatches on the array's OWN namespace (data-first)
    for beta in (0.0, 0.5, -0.5, 1.0):
        ref = numpy.asarray(pot._ddenstwobetadr(rnp, beta=beta))
        got = as_numpy(pot._ddenstwobetadr(r, beta=beta))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"_ddenstwobetadr beta={beta}"
        )
    # closed-form total mass M(inf) = amp * F(inf) stays finite on every backend
    inf = _asarray(backend_name, float("inf"))
    numpy.testing.assert_allclose(
        as_numpy(pot._mass(inf)), float(pot._mass(numpy.inf)), rtol=1e-12
    )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_rforce_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = as_numpy(pot.Rforce(R, z))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd(backend_name, pot):
    # d/dr[_rforce] via autodiff (through exp1 in _F_closed) vs central FD, at a
    # closed-form point and near the series/closed seam. r == 0 is avoided.
    eps = 1e-6
    for r0 in (0.3, 0.8, 4.0):
        fd = (
            float(pot._rforce(numpy.asarray(r0 + eps)))
            - float(pot._rforce(numpy.asarray(r0 - eps)))
        ) / (2.0 * eps)
        if backend_name == "jax":
            ad = float(jax.grad(lambda x: pot._rforce(x))(jnp.asarray(r0)))
        else:
            xt = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
            pot._rforce(xt).backward()
            ad = float(xt.grad)
        assert not numpy.isnan(ad), f"NaN grad at r={r0} ({backend_name})"
        numpy.testing.assert_allclose(
            ad, fd, rtol=1e-5, err_msg=f"r={r0} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_is_minus_rforce(backend_name, pot):
    # Exact analytic identity AD(_evaluate) == -_Rforce (spherical: dPhi/dR at
    # z=0 equals the full radial force through R/r == 1).
    R0, z0 = 1.3, 0.4
    ref = -float(pot._Rforce(numpy.asarray(R0), numpy.asarray(z0)))
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(R, torch.tensor(z0, dtype=torch.float64)).backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, ref, rtol=1e-9)
