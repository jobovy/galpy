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
    # Independent FD cross-check of the migrated _evaluate gradient. The exact
    # analytic identity AD(_evaluate)==-_Rforce is asserted (far more tightly) in
    # test_force_hessian_identities_3d; this FD test additionally catches a latent
    # bug shared by BOTH the gradient and the hand-coded _Rforce (which the
    # analytic identity alone could not detect).
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


# --- _mass parity + grad (R-only signature) ----------------------------------
# _mass(R, z=None) is migrated to xp for these potentials.
_MASS_POTS = [
    KuzminDiskPotential(amp=1.2, a=0.7),
    RazorThinExponentialDiskPotential(amp=1.0, hr=0.4),
]
_MASS_IDS = [type(p).__name__ for p in _MASS_POTS]
_MASS_RS = [0.3, 1.0, 2.5]


@pytest.mark.parametrize("pot", _MASS_POTS, ids=_MASS_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_mass_value_parity(backend_name, pot):
    ref = numpy.asarray(pot._mass(numpy.asarray(_MASS_RS)))
    got = _tonumpy(pot._mass(_asarray(backend_name, _MASS_RS)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", _MASS_POTS, ids=_MASS_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_mass_grad_vs_finite_difference(backend_name, pot):
    R0 = 1.3
    eps = 1e-6
    fd = (
        float(pot._mass(numpy.asarray(R0 + eps)))
        - float(pot._mass(numpy.asarray(R0 - eps)))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(lambda R: pot._mass(R))(jnp.asarray(R0)))
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._mass(R).backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


# --- singular-point parity for rewritten branches ----------------------------
# KuzminDisk's |z| kink (z == 0): _zforce / _Rzderiv / _z2deriv use xp.sign /
# xp.abs and must agree across backends exactly at z == 0.
@pytest.mark.parametrize("method", ["_zforce", "_Rzderiv", "_z2deriv"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_kuzmin_zkink_at_zero(backend_name, method):
    pot = KuzminDiskPotential(amp=1.2, a=0.7)
    R = [0.5, 1.0, 2.0]
    z = [0.0, 0.0, 0.0]
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(R), numpy.asarray(z)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, R), _asarray(backend_name, z))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# MiyamotoNagai a == 0 (and the a == 0, b == 0, z == 0 triple-singular point):
# the a == 0 config branch must be finite and identical across backends, even
# at the b == 0, z == 0 point that 0/0-poisons the general formula.
@pytest.mark.parametrize(
    "method", ["_zforce", "_dens", "_z2deriv", "_Rzderiv", "_Rforce", "_evaluate"]
)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_miyamoto_a0_branch_parity(backend_name, method):
    pot = MiyamotoNagaiPotential(amp=0.9, a=0.0, b=0.6)
    R = [0.5, 1.0, 2.0]
    z = [0.0, 0.1, 0.3]
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(R), numpy.asarray(z)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, R), _asarray(backend_name, z))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("method", ["_zforce", "_Rzderiv"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_miyamoto_a0_b0_z0_finite(backend_name, method):
    # a == 0, b == 0, z == 0: the simplified branch avoids the general formula's
    # 0/0 (asqrtbz / sqrtbz) and must be finite and identical across backends.
    pot = MiyamotoNagaiPotential(amp=0.9, a=0.0, b=0.0)
    R = [0.5, 1.0, 2.0]
    z = [0.0, 0.0, 0.0]
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(R), numpy.asarray(z)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, R), _asarray(backend_name, z))
    )
    assert numpy.all(numpy.isfinite(ref))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


###############################################################################
# Analytic-identity autodiff checks. galpy's sign conventions are
#   Rforce = -dPhi/dR,  zforce = -dPhi/dz;  R2deriv = d^2Phi/dR^2, etc.
# so under autodiff (exact to ~1e-9, unlike finite differences ~1e-5):
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_Rforce  wrt R) == -_R2deriv      AD(_Rforce  wrt z) == -_Rzderiv
#   AD(_zforce  wrt z) == -_z2deriv
# This cross-validates the hand-coded forces / 2nd-derivatives, not just the AD
# plumbing. These disk potentials are axisymmetric, so phi-direction pairs are
# absent from their method lists. A pair is checked only when BOTH of its
# methods are migrated for that potential: FlattenedPowerPotential's _Rzderiv is
# the base-class numerical derivative (not migrated / not listed), so its
# (_Rforce wrt z) pair is correctly skipped rather than checked.
###############################################################################
_R, _Z = 0, 1
_ID_PAIRS_2D = [
    ("_evaluate", _R, "_Rforce", "R"),
    ("_evaluate", _Z, "_zforce", "z"),
    ("_Rforce", _R, "_R2deriv", "R"),
    ("_Rforce", _Z, "_Rzderiv", "z"),
    ("_zforce", _Z, "_z2deriv", "z"),
]


def _grad_wrt(backend_name, fn, *args, argnum=0):
    # AD of scalar-valued fn(*args) wrt args[argnum]. Mirrors the pilot:
    # jax.grad for jax; a fresh leaf tensor + scalar backward() for torch.
    if backend_name == "jax":
        jargs = [jnp.asarray(a) for a in args]

        def f(x):
            full = list(jargs)
            full[argnum] = x
            return fn(*full)

        return float(jax.grad(f)(jargs[argnum]))
    targs = [torch.tensor(a, dtype=torch.float64) for a in args]
    leaf = torch.tensor(args[argnum], dtype=torch.float64, requires_grad=True)
    targs[argnum] = leaf
    out = fn(*targs)
    out.backward()  # backward() needs a scalar output; all args here are scalars
    return float(leaf.grad)


# Iterate over every 3D potential with its migrated-method list; gate each pair
# on both methods being present. KuzminDisk has a |z| kink at z == 0, so the
# smooth identity point keeps z0 != 0 (as the existing tests do).
_ID_POTS_3D = [(p, ms) for (p, ms) in _POTS_3D]
_ID3D_IDS = [f"{type(p).__name__}-{i}" for i, (p, _) in enumerate(_ID_POTS_3D)]


@pytest.mark.parametrize("pot,methods", _ID_POTS_3D, ids=_ID3D_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities_3d(backend_name, pot, methods):
    # For EVERY identity pair where both methods are migrated for this potential,
    # AD(lower wrt var) == -higher, exact to rtol=1e-9.
    if not (set(methods) & {"_evaluate", "_Rforce", "_zforce"}):
        pytest.skip("no migrated force/evaluate methods to form an identity pair")
    R0, z0 = 1.3, 0.4
    n_checked = 0
    for lower, argnum, higher, vn in _ID_PAIRS_2D:
        if lower not in methods or higher not in methods:
            continue
        ad = _grad_wrt(
            backend_name,
            lambda R, z, _l=lower: getattr(pot, _l)(R, z),
            R0,
            z0,
            argnum=argnum,
        )
        ref = -float(getattr(pot, higher)(R0, z0))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            err_msg=f"{type(pot).__name__}: AD({lower}/{vn})==-{higher}",
        )
        n_checked += 1
    assert n_checked > 0, f"no identity pairs exercised for {type(pot).__name__}"


# --- 1D linearPotential force identity ----------------------------------------
# AD(_evaluate wrt x) == -_force for the 1D IsothermalDisk / KG potentials.
@pytest.mark.parametrize("pot", _POTS_1D, ids=_POT1D_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_identity_1d(backend_name, pot):
    x0 = 0.9
    ad = _grad_wrt(backend_name, lambda x: pot._evaluate(x), x0)
    ref = -float(pot._force(x0))
    numpy.testing.assert_allclose(ad, ref, rtol=1e-9)
