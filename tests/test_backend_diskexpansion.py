###############################################################################
# test_backend_diskexpansion.py: backend-agnostic tests for the
# KuijkenDubinskiDiskExpansionPotential layer (P2.5) -- the analytic
# Sigma(r) * Hz(z) correction terms shared by DiskSCFPotential and
# DiskMultipoleExpansionPotential.
#
# The expansion sub-potential self._me (SCFPotential /
# MultipoleExpansionPotential) is a separate migration item, so these tests
# isolate the migrated Kuijken-Dubinski layer by attaching an
# already-backend-clean PlummerPotential as self._me: every KD method
# (_evaluate, forces, second derivatives, _dens) then runs its full code path
# -- the built-in dict Sigma/hz profiles (exp, expwhole, exp/sech2 hz,
# multi-component) plus the _me delegation -- under numpy, jax, and torch.
#
# For each configuration and method this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14),
#      including at the z == 0 kink of the exp vertical profile,
#   2. autodiff (jax.grad / torch.autograd) of _evaluate matches central
#      finite differences,
#   3. the analytic identities AD(_evaluate) == -force, AD(force) == -2nd-deriv
#      hold to rtol=1e-9,
#   4. the phiME effective-density helper is backend-clean when the user
#      density callable is (the documented requirement).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import get_namespace
from galpy.potential import PlummerPotential
from galpy.potential.KuijkenDubinskiDiskExpansionPotential import (
    KuijkenDubinskiDiskExpansionPotential,
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


def _asarray(backend_name, x, requires_grad=False):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


def _dens_xp(R, z):
    # backend-clean stand-in for the default (numpy-lambda) disk density
    xp = get_namespace(R, z)
    return 13.5 * xp.exp(-3.0 * R) * xp.exp(-27.0 * xp.abs(z))


def _kd_pot(Sigma, hz):
    """A KuijkenDubinski-layer potential with a backend-clean _me (Plummer),
    so the migrated correction-term math is exercised end-to-end on every
    backend without depending on the SCF/Multipole migrations."""
    pot = KuijkenDubinskiDiskExpansionPotential(
        amp=1.0, dens=_dens_xp, Sigma=Sigma, hz=hz
    )
    pot._me = PlummerPotential(amp=1.1, b=0.8)
    pot._finish_init(False)
    return pot


# --- CASES: every built-in dict profile type + multi-component ---------------
_POTS = [
    # (id, potential)
    (
        "exp-exp",
        _kd_pot(
            {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
            {"type": "exp", "h": 1.0 / 27.0},
        ),
    ),
    (
        "expwhole-sech2",
        _kd_pot(
            {"type": "expwhole", "h": 1.0 / 3.0, "amp": 1.0, "Rhole": 0.5},
            {"type": "sech2", "h": 1.0 / 27.0},
        ),
    ),
    # two Sigma components with per-component vertical profiles
    (
        "two-component",
        _kd_pot(
            [
                {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
                {"type": "expwhole", "h": 0.5, "amp": 0.5, "Rhole": 0.4},
            ],
            [{"type": "exp", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 20.0}],
        ),
    ),
]
_POT_IDS = [pid for pid, _ in _POTS]

_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]

_RS = [0.4, 1.0, 2.1]
_ZS = [-0.3, 0.05, 0.2]


# --- value parity -------------------------------------------------------------
@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("pot", [p for _, p in _POTS], ids=_POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, method):
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, _RS), _asarray(backend_name, _ZS))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# The exp vertical profile has a |z| kink (xp.sign / xp.abs): values at z == 0
# must agree exactly across backends (cf. the KuzminDisk z-kink test).
@pytest.mark.parametrize("method", ["_zforce", "_Rzderiv", "_z2deriv", "_dens"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_zkink_at_zero(backend_name, method):
    pot = _POTS[0][1]  # exp-exp
    R = [0.5, 1.0, 2.0]
    z = [0.0, 0.0, 0.0]
    ref = numpy.asarray(getattr(pot, method)(numpy.asarray(R), numpy.asarray(z)))
    got = _tonumpy(
        getattr(pot, method)(_asarray(backend_name, R), _asarray(backend_name, z))
    )
    assert numpy.all(numpy.isfinite(ref))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# sech2's logsumexp overflow guard: large |z| must stay finite on every backend.
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sech2_large_z_no_overflow(backend_name):
    pot = _POTS[1][1]  # expwhole-sech2
    R = [1.0, 1.0]
    z = [-40.0, 40.0]
    ref = numpy.asarray(pot._evaluate(numpy.asarray(R), numpy.asarray(z)))
    assert numpy.all(numpy.isfinite(ref))
    got = _tonumpy(pot._evaluate(_asarray(backend_name, R), _asarray(backend_name, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- autodiff vs finite difference ---------------------------------------------
@pytest.mark.parametrize("pot", [p for _, p in _POTS], ids=_POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot):
    # Independent FD cross-check of the migrated _evaluate gradient (the exact
    # analytic identities are asserted, far more tightly, below). z0 != 0 keeps
    # away from the exp-profile |z| kink.
    R0, z0 = 1.3, 0.4
    eps = 1e-6

    def phi_np(R, z):
        return float(pot._evaluate(numpy.asarray(R), numpy.asarray(z)))

    fdR = (phi_np(R0 + eps, z0) - phi_np(R0 - eps, z0)) / (2 * eps)
    fdz = (phi_np(R0, z0 + eps) - phi_np(R0, z0 - eps)) / (2 * eps)
    if backend_name == "jax":
        adR = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
        adz = float(
            jax.grad(lambda z: pot._evaluate(jnp.asarray(R0), z))(jnp.asarray(z0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(R, z).backward()
        adR, adz = float(R.grad), float(z.grad)
    numpy.testing.assert_allclose(adR, fdR, rtol=1e-5)
    numpy.testing.assert_allclose(adz, fdz, rtol=1e-5)


###############################################################################
# Analytic-identity autodiff checks (galpy sign conventions):
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_Rforce  wrt R) == -_R2deriv      AD(_Rforce  wrt z) == -_Rzderiv
#   AD(_zforce  wrt z) == -_z2deriv
# This cross-validates the hand-coded correction-term forces / 2nd derivatives.
###############################################################################
_R, _Z = 0, 1
_ID_PAIRS = [
    ("_evaluate", _R, "_Rforce"),
    ("_evaluate", _Z, "_zforce"),
    ("_Rforce", _R, "_R2deriv"),
    ("_Rforce", _Z, "_Rzderiv"),
    ("_zforce", _Z, "_z2deriv"),
]


def _grad_wrt(backend_name, fn, *args, argnum=0):
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
    fn(*targs).backward()
    return float(leaf.grad)


@pytest.mark.parametrize("pot", [p for _, p in _POTS], ids=_POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities(backend_name, pot):
    R0, z0 = 1.3, 0.4
    for lower, argnum, higher in _ID_PAIRS:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, _l=lower: getattr(pot, _l)(R, z),
            R0,
            z0,
            argnum=argnum,
        )
        ref = -float(getattr(pot, higher)(R0, z0))
        numpy.testing.assert_allclose(
            ad, ref, rtol=1e-9, err_msg=f"AD({lower}) == -{higher} ({backend_name})"
        )


# --- phiME effective density (coefficient-computation helper) ------------------
# Backend-clean iff the user dens callable is (the documented requirement);
# this is the right-hand side fed to SCF/Multipole from_density.
@pytest.mark.parametrize("pot", [p for _, p in _POTS], ids=_POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_phiME_dens_value_parity(backend_name, pot):
    ref = numpy.asarray(pot._phiME_dens_func(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(
        pot._phiME_dens_func(_asarray(backend_name, _RS), _asarray(backend_name, _ZS))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
