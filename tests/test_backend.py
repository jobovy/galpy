###############################################################################
# test_backend.py: core tests for the galpy.backend multi-backend layer (the
# resolver) and the reference migrated potentials (Plummer, Isochrone).
# Per-family potential tests live in tests/test_backend_<family>.py; autodiff/
# orbit/action-angle backend tests will get their own test_backend_*.py modules.
#
# Proves end-to-end, for the two reference potentials (Plummer, Isochrone):
#   1. numpy / jax / torch produce identical values (at the existing tolerances),
#   2. autodiff (jax.grad / torch.autograd) matches finite differences,
#   3. the physics identities d(Phi)/dR == -Rforce and d(-Rforce)/dR == R2deriv
#      hold under autodiff (the gradient of a migrated potential is consistent
#      with its hand-coded force/Hessian),
#   4. device & dtype are preserved,
#   5. the resolver precedence (xp= > data > context > numpy) behaves.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.potential import IsochronePotential, PlummerPotential

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

POTS = [PlummerPotential(amp=1.3, b=0.7), IsochronePotential(amp=2.0, b=1.1)]
POT_IDS = [type(p).__name__ for p in POTS]
METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
    "_surfdens",
]

_RS = [0.5, 1.0, 2.0]
_ZS = [0.1, 0.2, 0.3]


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


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, method):
    # Reference is always numpy
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, _RS), _asarray(backend_name, _ZS))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot):
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


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_and_hessian_identities(backend_name, pot):
    # d(_evaluate)/dR == -_Rforce  and  d(-_Rforce)/dR == _R2deriv
    R0, z0 = 1.3, 0.4
    ref_force = -float(pot._Rforce(R0, z0))
    ref_R2 = float(pot._R2deriv(R0, z0))
    if backend_name == "jax":
        g1 = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
        g2 = float(
            jax.grad(lambda R: -pot._Rforce(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(R, torch.tensor(z0, dtype=torch.float64)).backward()
        g1 = float(R.grad)
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        (-pot._Rforce(R, torch.tensor(z0, dtype=torch.float64))).backward()
        g2 = float(R.grad)
    numpy.testing.assert_allclose(g1, ref_force, rtol=1e-9)
    numpy.testing.assert_allclose(g2, ref_R2, rtol=1e-9)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
def test_jax_vmap_and_jit(pot):
    R = jnp.asarray(_RS)
    z = jnp.asarray(_ZS)
    ref = numpy.asarray(pot._Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    jitted = numpy.asarray(jax.jit(pot._Rforce)(R, z))
    numpy.testing.assert_allclose(jitted, ref, rtol=1e-12)
    vmapped = numpy.asarray(jax.vmap(pot._Rforce)(R, z))
    numpy.testing.assert_allclose(vmapped, ref, rtol=1e-12)


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
def test_torch_dtype_preserved(pot):
    for dt in (torch.float64, torch.float32):
        out = pot._Rforce(torch.tensor(_RS, dtype=dt), torch.tensor(_ZS, dtype=dt))
        assert out.dtype == dt


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_method_parity(backend_name, pot):
    # Decorator pass-through: the public Rforce (through the unit decorators and
    # _amp) must give identical values across backends.
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(pot.Rforce(_asarray(backend_name, _RS), _asarray(backend_name, _ZS)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- resolver behavior ---------------------------------------------------------
def test_resolver_default_is_numpy():
    assert backend.get_namespace() is numpy
    assert backend.get_namespace(1.0, 2.0) is numpy
    assert backend.backend() == "numpy"


def test_resolver_data_first():
    assert backend.get_namespace(numpy.ones(3)) is numpy
    if "jax" in BACKENDS:
        assert "jax" in backend.get_namespace(jnp.ones(3)).__name__
    if "torch" in BACKENDS:
        assert "torch" in backend.get_namespace(torch.ones(3)).__name__


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_resolver_precedence():
    # explicit xp= wins over everything
    assert backend.get_namespace(numpy.ones(3), xp="jax").__name__.startswith("jax")
    with backend.use("jax"):
        assert "jax" in backend.get_namespace().__name__  # context default
        assert backend.get_namespace(numpy.ones(3)) is numpy  # data still wins
    assert backend.get_namespace() is numpy  # restored


def test_resolver_unknown_backend_raises():
    with pytest.raises(ValueError):
        backend.get_namespace(xp="tensorflow")


def test_resolver_xp_module_and_names():
    # xp= can be a namespace module (passed through) or a name string.
    assert backend.get_namespace(xp=numpy) is numpy
    assert backend.get_namespace(xp="numpy") is numpy
    if "jax" in BACKENDS:
        assert backend.get_namespace(xp="jax").__name__.startswith("jax")
    if "torch" in BACKENDS:
        assert "torch" in backend.get_namespace(xp="torch").__name__


def test_set_default_backend():
    try:
        backend.set_default_backend("numpy")
        assert backend.backend() == "numpy"
        if "jax" in BACKENDS:
            backend.set_default_backend("jax", force=True)
            assert backend.backend() == "jax"
            assert backend.get_namespace(numpy.ones(3)).__name__.startswith("jax")
    finally:
        backend.set_default_backend("numpy")


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_force_mode_overrides_data():
    # force=True makes the backend win even over numpy array inputs (the
    # whole-suite test mode); without force, data wins.
    with backend.use("jax", force=False):
        assert backend.get_namespace(numpy.ones(3)) is numpy
    with backend.use("jax", force=True):
        assert backend.get_namespace(numpy.ones(3)).__name__.startswith("jax")
        assert backend.backend() == "jax"
    assert backend.get_namespace(numpy.ones(3)) is numpy  # restored


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_force_mode_runs_numpy_inputs_in_backend():
    # Under a forced backend, a potential called with numpy inputs computes in the
    # backend (numpy inputs are pulled in at the first xp.<fn> call).
    pot = PlummerPotential(amp=1.3, b=0.7)
    with backend.use("jax", force=True):
        out = pot._evaluate(numpy.asarray(_RS), numpy.asarray(_ZS))
    assert "jax" in type(out).__module__
    ref = numpy.asarray(pot._evaluate(numpy.asarray(_RS), numpy.asarray(_ZS)))
    numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-12)
