###############################################################################
# test_backend_disk.py: backend-agnostic tests for the analytic disk / simple
# potential family (P2.2).
#
# For each migrated potential and each migrated compute method, this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14),
#   2. autodiff (jax.grad / torch.autograd) of _evaluate (or _force, for the
#      one-dimensional linearPotentials) matches central finite differences.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    FlattenedPowerPotential,
    IsothermalDiskPotential,
    KGPotential,
    KuzminDiskPotential,
    MiyamotoNagaiPotential,
    RazorThinExponentialDiskPotential,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
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

# --- 3D potentials: (R,z) signature ------------------------------------------
# Each entry: (instance, [migrated methods]).
_POTS_3D = [
    (
        MiyamotoNagaiPotential(amp=1.3, a=0.8, b=0.3),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_dens",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
        ],
    ),
    # a == 0 exercises the special-case branch in MiyamotoNagaiPotential.
    (
        MiyamotoNagaiPotential(amp=0.9, a=0.0, b=0.6),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_dens",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
        ],
    ),
    (
        KuzminDiskPotential(amp=1.2, a=0.7),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_surfdens",
        ],
    ),
    (
        FlattenedPowerPotential(amp=1.1, alpha=0.5, q=0.9),
        ["_evaluate", "_Rforce", "_zforce", "_R2deriv", "_z2deriv", "_dens"],
    ),
    # alpha == 0 exercises the LogarithmicHalo-like branch (uses xp.log).
    (
        FlattenedPowerPotential(amp=1.1, alpha=0.0, q=0.85),
        ["_evaluate", "_Rforce", "_zforce", "_R2deriv", "_z2deriv", "_dens"],
    ),
    # Only _surfdens is analytically clean (the rest use scipy.special).
    (RazorThinExponentialDiskPotential(amp=1.0, hr=0.4), ["_surfdens"]),
]
_POT3D_IDS = [f"{type(p).__name__}-{i}" for i, (p, _) in enumerate(_POTS_3D)]

_RS = [0.5, 1.0, 2.0]
_ZS = [0.1, 0.2, 0.3]

# --- 1D linearPotentials: (x) signature --------------------------------------
_POTS_1D = [
    IsothermalDiskPotential(amp=1.0, sigma=0.2),
    KGPotential(amp=1.0, K=1.15, F=0.03, D=1.8),
]
_POT1D_IDS = [type(p).__name__ for p in _POTS_1D]
_XS = [0.3, 0.7, 1.4]


def _asarray(backend_name, x, requires_grad=False):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


# --- value parity ------------------------------------------------------------
def _iter_3d_cases():
    for (pot, methods), pid in zip(_POTS_3D, _POT3D_IDS):
        for method in methods:
            yield pytest.param(pot, method, id=f"{pid}-{method}")


@pytest.mark.parametrize("pot,method", list(_iter_3d_cases()))
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_3d(backend_name, pot, method):
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, _RS), _asarray(backend_name, _ZS))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", _POTS_1D, ids=_POT1D_IDS)
@pytest.mark.parametrize("method", ["_evaluate", "_force"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_1d(backend_name, pot, method):
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(_XS)))
    got = _tonumpy(getattr(pot, method)(_asarray(backend_name, _XS)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- autodiff vs finite difference -------------------------------------------
# Only potentials whose _evaluate is migrated (not RazorThin, whose _evaluate
# still uses scipy.special).
_GRAD_POTS_3D = [(p, ms) for (p, ms) in _POTS_3D if "_evaluate" in ms]
_GRAD3D_IDS = [f"{type(p).__name__}-{i}" for i, (p, _) in enumerate(_GRAD_POTS_3D)]


@pytest.mark.parametrize("pot", [p for p, _ in _GRAD_POTS_3D], ids=_GRAD3D_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference_3d(backend_name, pot):
    R0, z0 = 1.3, 0.4
    eps = 1e-6

    def phi_np(R):
        return float(pot._evaluate(numpy.asarray(R), numpy.asarray(z0)))

    fd = (phi_np(R0 + eps) - phi_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._evaluate(R, torch.tensor(z0, dtype=torch.float64))
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("pot", _POTS_1D, ids=_POT1D_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference_1d(backend_name, pot):
    x0 = 0.9
    eps = 1e-6

    def phi_np(x):
        return float(pot._evaluate(numpy.asarray(x)))

    fd = (phi_np(x0 + eps) - phi_np(x0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(lambda x: pot._evaluate(x))(jnp.asarray(x0)))
    else:
        x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(x).backward()
        ad = float(x.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


# --- force/Hessian identity under autodiff (extra confidence) ----------------
@pytest.mark.parametrize("pot", [p for p, _ in _GRAD_POTS_3D], ids=_GRAD3D_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_identity_3d(backend_name, pot):
    # d(_evaluate)/dR == -_Rforce
    R0, z0 = 1.3, 0.4
    ref_force = -float(pot._Rforce(R0, z0))
    if backend_name == "jax":
        g1 = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(R, torch.tensor(z0, dtype=torch.float64)).backward()
        g1 = float(R.grad)
    numpy.testing.assert_allclose(g1, ref_force, rtol=1e-9)
