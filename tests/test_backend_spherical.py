###############################################################################
# test_backend_spherical.py: multi-backend tests for the spherical-potential
# family migrated to the galpy.backend namespace layer (P2.1).
#
# For every migrated potential and every migrated compute method this proves:
#   1. numpy / jax / torch produce identical values at the existing tolerances,
#   2. autodiff (jax.grad / torch.autograd) of _evaluate matches finite
#      differences (so the migrated compute path is differentiable and its
#      gradient is consistent with the hand-coded force).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
# Methods that genuinely require scipy.special with no backend-agnostic
# replacement yet (e.g. the base TwoPower._evaluate/_Rforce via hyp2f1,
# PowerSpherical._surfdens via hyp2f1, PowerSphericalPotentialwCutoff._mass via
# hyp1f1) remain numpy-only and are NOT exercised here. The previously deferred
# closed-form cases -- the xlogy-based NFW._evaluate and Burkert._revaluate, and
# the complex-arithmetic _surfdens of Burkert/Hernquist/Jaffe/NFW -- are now
# migrated and ARE covered below (incl. their R == a removable singularity).
###############################################################################
import numpy
import pytest

from galpy.potential import (
    BurkertPotential,
    DehnenCoreSphericalPotential,
    DehnenSphericalPotential,
    HernquistPotential,
    HomogeneousSpherePotential,
    JaffePotential,
    KeplerPotential,
    NFWPotential,
    PowerSphericalPotential,
    PowerSphericalPotentialwCutoff,
    SphericalShellPotential,
    TwoPowerSphericalPotential,
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

# Each entry: (potential instance, list of migrated methods to check, list of
# methods supporting the AD-vs-FD _evaluate check [empty if _evaluate deferred]).
# Only methods whose compute path is fully namespace-swapped are listed.
_THREE_D = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]

CASES = [
    # Dehnen family: fully analytic, all 3D methods migrated.
    (DehnenSphericalPotential(amp=1.3, a=1.1, alpha=1.5), _THREE_D, True),
    (DehnenCoreSphericalPotential(amp=1.3, a=1.1), _THREE_D, True),
    # Hernquist / Jaffe: _evaluate/_Rforce/_zforce/_R2deriv/_Rzderiv/_dens
    # migrated; _z2deriv delegates to _R2deriv; _surfdens deferred (complex).
    (
        HernquistPotential(amp=1.3, a=1.1),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_dens",
        ],
        True,
    ),
    (
        JaffePotential(amp=1.3, a=1.1),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_dens",
        ],
        True,
    ),
    # NFW: forces/2nd-derivs/dens migrated; _evaluate now migrated (the xlogy
    # was rewritten as the equivalent NaN-safe -(1/r) log(1+r/a)).
    (
        NFWPotential(amp=1.3, a=1.1),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_dens",
        ],
        True,
    ),
    # PowerSpherical / Kepler: all listed compute methods migrated (_surfdens
    # uses hyp2f1, deferred).
    (PowerSphericalPotential(amp=1.3, alpha=1.5), _THREE_D, True),
    (KeplerPotential(amp=1.3), _THREE_D, True),
    # TwoPower base (non-special): only _dens is scipy-free.
    (
        TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=3.5),
        ["_dens"],
        False,
    ),
    # HomogeneousSphere: piecewise, all methods migrated via xp.where.
    (HomogeneousSpherePotential(amp=1.3, R=1.1), _THREE_D, True),
    # Burkert: _rforce/_r2deriv/_rdens migrated => 3D forces/derivs/dens work;
    # _revaluate now migrated (the xlogy was rewritten as the equivalent
    # NaN-safe (2/x) log(1+x^2)), so _evaluate is included.
    (
        BurkertPotential(amp=1.3, a=1.1),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_dens",
        ],
        True,
    ),
    # SphericalShell: all 1D methods migrated via xp.where (+ surfdens).
    (
        SphericalShellPotential(amp=1.3, a=0.7),
        [
            "_evaluate",
            "_Rforce",
            "_zforce",
            "_R2deriv",
            "_z2deriv",
            "_Rzderiv",
            "_dens",
        ],
        True,
    ),
]

CASE_IDS = [type(p).__name__ for p, _, _ in CASES]

# Grid avoids r == a exact points (shell singularities) and r == 0.
_RS = [0.5, 1.0, 2.0]
_ZS = [0.1, 0.2, 0.3]


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


@pytest.mark.parametrize("pot,methods,_ad", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, methods, _ad):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in methods:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = _tonumpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{type(pot).__name__}.{method}"
        )


@pytest.mark.parametrize("pot,methods,_ad", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot, methods, _ad):
    # The public Rforce (through the unit decorators and _amp) must give
    # identical values across backends. Only meaningful where _Rforce is
    # actually migrated (the base TwoPower _Rforce is scipy-deferred).
    if "_Rforce" not in methods:
        pytest.skip("_Rforce deferred (scipy.special) for this potential")
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(pot.Rforce(R, z))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot,methods,ad_ok", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot, methods, ad_ok):
    if not ad_ok:
        pytest.skip("_evaluate deferred (scipy.special) for this potential")
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


@pytest.mark.parametrize("pot,methods,ad_ok", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_matches_rforce(backend_name, pot, methods, ad_ok):
    # d(_evaluate)/dR == -_Rforce (consistency of the migrated gradient with the
    # hand-coded radial force). Only where _evaluate is migrated.
    if not ad_ok:
        pytest.skip("_evaluate deferred (scipy.special) for this potential")
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


###############################################################################
# _surfdens for the cuspy / complex-arithmetic profiles.
#
# These were migrated from the numpy `sqrt(neg + 0j)` + `if Rma == 0` scalar
# branch to a backend-agnostic xp form: a complex sqrt built from a NaN-safe
# real argument, the generic-branch result via xp.real, and the removable
# singularity at R == a handled with a NaN-safe xp.where. The grids below
# deliberately *include* R == a so the safe-`where` guards are exercised.
###############################################################################
# (potential, scale-radius attribute)
_SURFDENS_CASES = [
    BurkertPotential(amp=1.3, a=1.1),
    HernquistPotential(amp=1.3, a=1.1),
    JaffePotential(amp=1.3, a=1.1),
    NFWPotential(amp=1.3, a=1.1),
]
_SURFDENS_IDS = [type(p).__name__ for p in _SURFDENS_CASES]


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_surfdens_value_parity(backend_name, pot):
    a = pot.a
    # Grid spanning R < a, R == a (the removable singularity), and R > a.
    Rs = [0.5, 0.9 * a, a, 1.3 * a, 2.0]
    zs = [0.3, 0.2, 0.4, 0.2, 0.3]
    ref = numpy.asarray(pot._surfdens(numpy.asarray(Rs), numpy.asarray(zs)))
    got = _tonumpy(
        pot._surfdens(_asarray(backend_name, Rs), _asarray(backend_name, zs))
    )
    numpy.testing.assert_allclose(
        got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{type(pot).__name__}._surfdens"
    )


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_surfdens_edge_finite(backend_name, pot):
    # The R == a removable singularity must give the same finite value across
    # backends (this is the branch the safe-`where` guards protect).
    a = pot.a
    ref = float(numpy.asarray(pot._surfdens(numpy.asarray(a), numpy.asarray(0.3))))
    got = float(
        _tonumpy(pot._surfdens(_asarray(backend_name, a), _asarray(backend_name, 0.3)))
    )
    assert numpy.isfinite(ref)
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def _surfdens_grad(backend_name, pot, R0, z0):
    if backend_name == "jax":
        return float(
            jax.grad(lambda R: pot._surfdens(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
    pot._surfdens(R, torch.tensor(z0, dtype=torch.float64)).backward()
    return float(R.grad)


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_surfdens_grad_vs_finite_difference(backend_name, pot):
    # Off-edge (R != a), the migrated generic branch must be differentiable and
    # its gradient must match finite differences.
    z0 = 0.3
    R0 = 0.6
    eps = 1e-6
    fd = (
        float(pot._surfdens(numpy.asarray(R0 + eps), numpy.asarray(z0)))
        - float(pot._surfdens(numpy.asarray(R0 - eps), numpy.asarray(z0)))
    ) / (2 * eps)
    ad = _surfdens_grad(backend_name, pot, R0, z0)
    assert numpy.isfinite(ad), f"{type(pot).__name__} grad NaN at R={R0}"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_surfdens_grad_finite_at_edge(backend_name, pot):
    # At the exact removable singularity R == a the dead generic branch's
    # 1/Rma / 1/(R^2-a^2) terms would, without the safe-`where` guards, inject
    # NaNs into reverse-mode autodiff. The guards must keep the gradient FINITE.
    # (Its *value* at the measure-zero edge point is the R-independent edge
    # branch's derivative, i.e. 0; we only require finiteness, not FD-match, at
    # this single kink point.)
    a = pot.a
    ad = _surfdens_grad(backend_name, pot, float(a), 0.3)
    assert numpy.isfinite(ad), f"{type(pot).__name__} grad NaN at edge R==a"


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_powerspherical_wcutoff_dens_parity(backend_name):
    # PowerSphericalPotentialwCutoff._dens / _ddensdr / _d2densdr2 are the
    # scipy-free (closed-form) methods of that potential and were migrated to xp.
    pot = PowerSphericalPotentialwCutoff(amp=1.3, alpha=1.3, rc=1.2)
    Rs = [0.5, 1.0, 2.0]
    zs = [0.1, 0.2, 0.3]
    ref = numpy.asarray(pot._dens(numpy.asarray(Rs), numpy.asarray(zs)))
    got = _tonumpy(pot._dens(_asarray(backend_name, Rs), _asarray(backend_name, zs)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    for method in ["_ddensdr", "_d2densdr2"]:
        ref = numpy.asarray(getattr(pot, method)(numpy.asarray(Rs)))
        got = _tonumpy(getattr(pot, method)(_asarray(backend_name, Rs)))
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, err_msg=method)
