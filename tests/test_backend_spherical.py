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
# replacement yet (e.g. PowerSpherical._surfdens via hyp2f1,
# PowerSphericalPotentialwCutoff._mass via hyp1f1) remain numpy-only and are
# NOT exercised here. The previously deferred closed-form cases -- the
# xlogy-based NFW._evaluate and Burkert._revaluate, and the complex-arithmetic
# _surfdens of Burkert/Hernquist/Jaffe/NFW -- are now migrated and ARE covered
# below (incl. their R == a removable singularity), as is the generic-
# (alpha, beta) TwoPower base, whose incomplete-beta series are static in
# (alpha, beta) (see the dedicated block at the end).
###############################################################################
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager

from galpy.backend import as_numpy, is_backend_array
from galpy.potential import (
    BurkertPotential,
    DehnenCoreSphericalPotential,
    DehnenSphericalPotential,
    EinastoPotential,
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

    # Mirrors the jax_enable_x64 above: galpy's tolerances assume float64, and
    # the identity checks here are tight enough (rtol=1e-9) to fail at float32.
    # Not inherited -- the conftest fixture only sets this under --backend torch,
    # so without it a single-file run of THIS file silently drops to float32.
    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# Each entry: (potential instance, list of migrated 2D (R,z) methods to check,
# list of migrated 1D radial methods [_revaluate/_rforce/_r2deriv, empty if the
# potential is not implemented via the SphericalPotential 1D interface], bool
# for whether _evaluate is migrated [supports the AD-vs-FD _evaluate check]).
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
# Potentials built on the SphericalPotential 1D interface expose these three
# radial methods (_R*/_z* are derived from them analytically in the base class).
_ONE_D = ["_revaluate", "_rforce", "_r2deriv"]

CASES = [
    # Dehnen family: fully analytic, all 3D methods migrated.
    (DehnenSphericalPotential(amp=1.3, a=1.1, alpha=1.5), _THREE_D, [], True),
    (DehnenCoreSphericalPotential(amp=1.3, a=1.1), _THREE_D, [], True),
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
        [],
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
        [],
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
        [],
        True,
    ),
    # PowerSpherical / Kepler: all listed compute methods migrated (_surfdens
    # uses hyp2f1, deferred).
    (PowerSphericalPotential(amp=1.3, alpha=1.5), _THREE_D, [], True),
    (KeplerPotential(amp=1.3), _THREE_D, [], True),
    # TwoPower base (non-special alpha/beta): hyp2f1/gamma routed through the
    # galpy.backend.special router, so all 3D methods are migrated. (The
    # dedicated block at the end additionally pins numpy byte-parity, the
    # beta == 3 branch, force grads vs FD, and jax.jit survival.)
    (
        TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=3.5),
        _THREE_D,
        [],
        True,
    ),
    # HomogeneousSphere: piecewise, all methods migrated via xp.where.
    (HomogeneousSpherePotential(amp=1.3, R=1.1), _THREE_D, [], True),
    # Burkert: _revaluate/_rforce/_r2deriv/_rdens migrated => the derived 3D
    # forces/derivs/dens work; the xlogy in _revaluate was rewritten as the
    # equivalent NaN-safe (2/x) log(1+x^2), so _evaluate is included. Built on
    # the SphericalPotential 1D interface, so the 1D radial methods are checked.
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
        _ONE_D,
        True,
    ),
    # Einasto: _revaluate/_rforce/_r2deriv migrated through the special-function
    # router (gamma/gammaincc; native on numpy/jax/torch) and the r == 0 core
    # branch through a guarded xp.where; _rdens is closed-form. The derived 3D
    # methods come from the SphericalPotential base. n=1.5 exercises the
    # non-trivial s**(1/n) exponent.
    (
        EinastoPotential(amp=1.3, h=1.1, n=1.5),
        _THREE_D,
        _ONE_D,
        True,
    ),
    # SphericalShell: all 1D methods migrated via xp.where (+ surfdens). The 1D
    # radial methods have a kink at r == a (= 0.7); the identity point r0 = 1.3
    # is well away from it.
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
        _ONE_D,
        True,
    ),
]

CASE_IDS = [type(p).__name__ for p, _, _, _ in CASES]

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


@pytest.mark.parametrize("pot,methods,_methods1d,_ad", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, methods, _methods1d, _ad):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in methods:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = as_numpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{type(pot).__name__}.{method}"
        )


@pytest.mark.parametrize("pot,methods,_methods1d,_ad", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot, methods, _methods1d, _ad):
    # The public Rforce (through the unit decorators and _amp) must give
    # identical values across backends. Only meaningful where _Rforce is
    # actually migrated.
    if "_Rforce" not in methods:
        pytest.skip("_Rforce deferred (scipy.special) for this potential")
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = as_numpy(pot.Rforce(R, z))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot,methods,_methods1d,ad_ok", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(
    backend_name, pot, methods, _methods1d, ad_ok
):
    # Independent (finite-difference) cross-check that the migrated _evaluate is
    # differentiable end-to-end. The exact analytic identity AD(_evaluate)==-_Rforce
    # is asserted (much more tightly) in test_force_hessian_identities below; this
    # FD test additionally guards against both the gradient AND the hand-coded
    # _Rforce sharing a latent bug (the analytic identity would pass in that case,
    # the FD check would not).
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


###############################################################################
# Analytic-identity autodiff checks. galpy's sign conventions are
#   Rforce = -dPhi/dR,  zforce = -dPhi/dz;  R2deriv = d^2Phi/dR^2, etc.
# so under autodiff (exact to ~1e-9, unlike finite differences ~1e-5):
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_Rforce  wrt R) == -_R2deriv      AD(_Rforce  wrt z) == -_Rzderiv
#   AD(_zforce  wrt z) == -_z2deriv
# and for the 1D radial interface:
#   AD(_revaluate wrt r) == -_rforce     AD(_rforce wrt r) == -_r2deriv
# This cross-validates the hand-coded forces / 2nd-derivatives, not just the AD
# plumbing. Spherical potentials are axisymmetric, so phi-direction pairs are
# absent from their method lists and are (correctly) never exercised here.
###############################################################################
# (lower method differentiated, derivative variable index in (R,z), higher
# method, var-name). The grid point R0,z0 is smooth (away from any kink).
_R, _Z = 0, 1
_ID_PAIRS_2D = [
    ("_evaluate", _R, "_Rforce", "R"),
    ("_evaluate", _Z, "_zforce", "z"),
    ("_Rforce", _R, "_R2deriv", "R"),
    ("_Rforce", _Z, "_Rzderiv", "z"),
    ("_zforce", _Z, "_z2deriv", "z"),
]
_ID_PAIRS_1D = [
    ("_revaluate", "_rforce"),
    ("_rforce", "_r2deriv"),
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


@pytest.mark.parametrize("pot,methods,methods1d,_ad", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities(backend_name, pot, methods, methods1d, _ad):
    # For EVERY identity pair where both methods are migrated for this potential,
    # AD(lower wrt var) == -higher, exact to rtol=1e-9. Gate on list membership so
    # the (absent) phi pairs of these axisymmetric potentials are simply skipped.
    # Potentials with no migrated force/eval pairs (only _dens) have nothing to
    # cross-validate here.
    if not (set(methods) & {"_evaluate", "_Rforce", "_zforce"}) and not methods1d:
        pytest.skip("no migrated force/evaluate methods to form an identity pair")
    R0, z0 = 1.3, 0.4
    n_checked = 0
    for lower, argnum, higher, _vn in _ID_PAIRS_2D:
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
            err_msg=f"{type(pot).__name__}: AD({lower}/{_vn})==-{higher}",
        )
        n_checked += 1
    # 1D radial interface (r0 smooth, away from any r == a shell kink).
    r0 = 1.3
    for lower, higher in _ID_PAIRS_1D:
        if lower not in methods1d or higher not in methods1d:
            continue
        ad = _grad_wrt(backend_name, lambda r, _l=lower: getattr(pot, _l)(r), r0)
        ref = -float(getattr(pot, higher)(r0))
        numpy.testing.assert_allclose(
            ad, ref, rtol=1e-9, err_msg=f"{type(pot).__name__}: AD({lower})==-{higher}"
        )
        n_checked += 1
    assert n_checked > 0, f"no identity pairs exercised for {type(pot).__name__}"


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
    got = as_numpy(
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
        as_numpy(pot._surfdens(_asarray(backend_name, a), _asarray(backend_name, 0.3)))
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
    got = as_numpy(pot._dens(_asarray(backend_name, Rs), _asarray(backend_name, zs)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    for method in ["_ddensdr", "_d2densdr2"]:
        ref = numpy.asarray(getattr(pot, method)(numpy.asarray(Rs)))
        got = as_numpy(getattr(pot, method)(_asarray(backend_name, Rs)))
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, err_msg=method)


def _surfdens_grad_z(backend_name, pot, R0, z0):
    """d(surfdens)/dz via autodiff at fixed R."""
    if backend_name == "jax":
        return float(
            jax.grad(lambda z: pot._surfdens(jnp.asarray(R0), z))(jnp.asarray(z0))
        )
    z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
    pot._surfdens(torch.tensor(R0, dtype=torch.float64), z).backward()
    return float(z.grad)


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_surfdens_z0_value_parity(backend_name, pot):
    # The R == a edge branch of _surfdens carries a 1/z**n (n in {3, 5}) that is
    # singular at z == 0. Under the eager xp.where BOTH branches are evaluated,
    # so in the generic region (R != a) at z == 0 that dead edge branch must be
    # z-guarded. The VALUE there is the generic branch's 0 and must be parity-
    # identical (and equal to numpy's 0) across backends, with no scalar crash.
    R0 = 0.6  # != a (== 1.1 for all cases)
    val = as_numpy(
        pot._surfdens(_asarray(backend_name, R0), _asarray(backend_name, 0.0))
    )
    ref = numpy.asarray(pot._surfdens(numpy.asarray(R0), numpy.asarray(0.0)))
    numpy.testing.assert_allclose(val, ref, rtol=1e-12, atol=0.0)
    assert numpy.isfinite(float(val)), (
        f"{type(pot).__name__} surfdens(R!=a,z=0) not finite"
    )


###############################################################################
# Einasto r == 0 core branch. _revaluate has a separate finite value at r == 0
# (the generic branch's (1-Q)/s term is 0/0 there) handled by a guarded
# xp.where: the dead generic branch is evaluated at the safe s == 1 so that,
# under eager backends, neither the 0/0 nor the (n > 1) infinite d(s**(1/n))/ds
# can NaN-poison values or reverse-mode gradients at the core.
###############################################################################
_EINASTO_NS = [1, 1.5, 3]


@pytest.mark.parametrize("n", _EINASTO_NS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_einasto_revaluate_core_value_parity(backend_name, n):
    pot = EinastoPotential(amp=1.3, h=1.1, n=n)
    # grid including r == 0 (the guarded core branch) and generic points
    rs = [0.0, 0.5, 1.3]
    ref = numpy.asarray(pot._revaluate(numpy.asarray(rs)))
    got = as_numpy(pot._revaluate(_asarray(backend_name, rs)))
    assert numpy.all(numpy.isfinite(ref))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n", _EINASTO_NS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_einasto_revaluate_grad_finite_at_core(backend_name, n):
    # At r == 0 the radial force vanishes (the Einasto density is cored), so
    # AD(_revaluate) there must be 0 and in particular FINITE: without the
    # ssafe guard the dead generic branch NaN-poisons the gradient.
    pot = EinastoPotential(amp=1.3, h=1.1, n=n)
    ad = _grad_wrt(backend_name, lambda r: pot._revaluate(r), 0.0)
    assert numpy.isfinite(ad), f"Einasto(n={n}) d_revaluate/dr NaN at r=0"
    numpy.testing.assert_allclose(ad, 0.0, atol=1e-12)


@pytest.mark.parametrize("pot", _SURFDENS_CASES, ids=_SURFDENS_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_surfdens_grad_z_finite_off_edge(backend_name, pot):
    # In the generic region (R != a), z == 0 is a valid finite-value input
    # (surfdens == 0). Without z-guarding the dead R == a edge branch, its
    # 1/z**n NaN-poisons reverse-mode d(surfdens)/dz at z == 0. The guard must
    # keep this gradient FINITE (matching the analytic _dens at the midplane).
    R0 = 0.6
    ad = _surfdens_grad_z(backend_name, pot, R0, 0.0)
    assert numpy.isfinite(ad), f"{type(pot).__name__} dSurf/dz NaN at z=0 (R!=a)"
    # surfdens integrates rho over [-z, z], so d(surfdens)/dz|_{z=0} = 2*rho(R,0);
    # cross-check the gradient against the analytic density (the true slope).
    rho = float(numpy.asarray(pot._dens(numpy.asarray(R0), numpy.asarray(0.0))))
    numpy.testing.assert_allclose(ad, 2.0 * rho, rtol=1e-7, atol=1e-10)


###############################################################################
# Generic-(alpha, beta) TwoPowerSphericalPotential: regression pins for its
# backend paths. Before they existed, these methods called scipy.special on
# the backend arrays directly:
# torch tensors with requires_grad RAISED (RuntimeError in Tensor.__array__),
# no-grad torch outputs silently round-tripped through numpy (graph never
# built), eager jax outputs degraded to plain numpy.ndarray, and jax.grad /
# jax.jit died with TracerArrayConversionError. These tests pin:
#   1. numpy BYTE-parity of _evaluate/_Rforce with the module-level numpy
#      formulas (numpy inputs never reach a backend path),
#   2. jax/torch grads of Phi, the forces and second derivatives wrt R vs
#      central finite differences of the numpy path (the detach/raise pin),
#   3. jax.jit survival (and jit == numpy values).
# _evaluate is the incomplete-beta form for every beta (3.0 included), its
# backend series static in (alpha, beta); _Rforce is M/r^2 from the same
# incomplete beta (_tp_radial), no longer hyp2f1(-r/a).
# alpha=1.5 keeps _specialSelf unset (a true generic case). Cross-backend VALUE
# parity on all methods is covered by the CASES entry above.
###############################################################################
_TWOPOWER_GENERIC = [
    TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=3.5),
    TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=4.5),
    TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=1.5, beta=3.0),
]
_TWOPOWER_GENERIC_IDS = ["beta3.5", "beta4.5", "beta3.0"]
_TWOPOWER_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
]


def _twopower_scipy_reference(pot, method, R, z):
    """The numpy formulas for _evaluate/_Rforce, reproduced verbatim (numpy +
    scipy.special) as the byte-parity reference."""
    a, alpha, beta = pot.a, pot.alpha, pot.beta
    if method == "_Rforce":
        from galpy.potential.TwoPowerSphericalPotential import _tp_radial

        r = numpy.sqrt(R**2.0 + z**2.0)
        x = r / a
        f1 = _tp_radial(alpha, beta, x / (1.0 + x), 1.0 / (1.0 + x), False)[0]
        return -R * (f1 / a**3.0)
    # Phi = -(M(x)/x + O(x))/a, the two incomplete betas through scipy
    from galpy.util.special import incomplete_beta

    x = numpy.sqrt(R**2.0 + z**2.0) / a
    w, s = x / (1.0 + x), 1.0 / (1.0 + x)
    return (
        -(
            incomplete_beta(3.0 - alpha, beta - 3.0, w, s) / x
            + incomplete_beta(beta - 2.0, 2.0 - alpha, s, w)
        )
        / a
    )


@pytest.mark.parametrize("pot", _TWOPOWER_GENERIC, ids=_TWOPOWER_GENERIC_IDS)
def test_twopower_generic_numpy_byte_parity(pot):
    # On a dense grid, the routed numpy path must be BYTE-identical to the
    # pre-routing direct scipy.special formulas (.tobytes() equality).
    Rg, zg = numpy.meshgrid(
        numpy.linspace(0.1, 10.0, 40), numpy.linspace(-3.0, 3.0, 30)
    )
    Rg, zg = Rg.ravel(), zg.ravel()
    for method in ["_evaluate", "_Rforce"]:
        got = getattr(pot, method)(Rg, zg)
        ref = _twopower_scipy_reference(pot, method, Rg, zg)
        assert isinstance(got, numpy.ndarray), f"{method} left the numpy namespace"
        assert got.tobytes() == ref.tobytes(), (
            f"numpy {method} not byte-identical to the direct-scipy formula "
            f"(alpha={pot.alpha}, beta={pot.beta})"
        )


@pytest.mark.parametrize("method", _TWOPOWER_METHODS)
@pytest.mark.parametrize("pot", _TWOPOWER_GENERIC, ids=_TWOPOWER_GENERIC_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_twopower_generic_grad_vs_finite_difference(backend_name, pot, method):
    # jax.grad / torch.autograd.grad of the routed methods wrt R must match a
    # central finite difference of the numpy path. Pre-routing, torch RAISED
    # ("Can't call numpy() on Tensor that requires grad") and jax raised
    # TracerArrayConversionError -- this is the detach/raise regression pin.
    f = getattr(pot, method)
    R0, z0, eps = 1.3, 0.4, 1e-6
    fd = (
        float(f(numpy.asarray(R0 + eps), numpy.asarray(z0)))
        - float(f(numpy.asarray(R0 - eps), numpy.asarray(z0)))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(lambda R: f(R, jnp.asarray(z0)))(jnp.asarray(R0)))
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        (ad,) = torch.autograd.grad(f(R, torch.tensor(z0, dtype=torch.float64)), R)
        ad = float(ad)
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-6, err_msg=f"{backend_name} grad of {method}"
    )


# d Phi/dr = M(x)/(x a)^2 (50-digit references, a = 1.3): the backend gradient
# of the incomplete-beta Phi where the old closed forms failed -- beta -> 3
# (Gamma(beta-3) poles), alpha -> 2 / alpha >= 2, alpha = beta, beta = 180, and
# the |q log(s2/s1)| > 1 branch; hyp2f1 _Rforce is 1e-4 off at beta = 3 +- 1e-12
_TWOPOWER_EDGE_DPHIDR = [  # alpha, beta, x, dPhi/dr
    (1.5, 3.000000000001, 0.4999, 3.8410361076102557e-1),
    (1.5, 3.000000000001, 5.0, 2.9898888674384627e-2),
    (1.5, 2.999999999999, 0.5001, 3.8397857915822237e-1),
    (1.999999999999, 100.0, 0.499999, 2.4400756420590309e-2),
    (2.0, 3.5, 0.3, 1.6165940919522785),
    (2.5, 2.5, 1.0, 1.1834319526627218),
    (1.5, 180.0, 0.5, 8.8887346478789128e-4),
    (1.96, 3.02, 1e-06, 3.2740078878820132e5),
]


@pytest.mark.parametrize("alpha,beta,x,dphidr", _TWOPOWER_EDGE_DPHIDR)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_twopower_phi_grad_edges(backend_name, alpha, beta, x, dphidr):
    pot = TwoPowerSphericalPotential(amp=1.0, a=1.3, alpha=alpha, beta=beta)
    R0 = x * 1.3
    if backend_name == "jax":
        f = lambda R: pot._evaluate(R, jnp.zeros_like(R))  # noqa: E731
        ad = float(jax.grad(f)(jnp.asarray(R0)))
        # and under jit: a static-length series
        ad_jit = float(jax.jit(jax.grad(f))(jnp.asarray(R0)))
        assert abs(ad_jit / dphidr - 1.0) < 1e-12, ad_jit
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        (ad,) = torch.autograd.grad(pot._evaluate(R, torch.zeros_like(R)), R)
        ad = float(ad)
    assert abs(ad / dphidr - 1.0) < 1e-12, f"{backend_name}: {ad} vs {dphidr}"


# Forces, second derivatives and mass from the same incomplete beta M:
# dPhi/dr = M/r^2, Phi'' = 4 pi rho - 2 M/r^3, Phi'' - Phi'/r = 4 pi rho - 3 M/r^3
# (the alpha ~ 1 / alpha ~ 0 cancellations of the last two taken out in closed
# form). 50-digit values and d/da at fixed r = x a (a = 1.2, amp = 2);
# Rzderiv at (R, z) = (0.6, 0.8) r, the others at (r, 0). The hyp2f1(-r/a)
# forms were up to 4e-3 off at beta = 3 -+ 1e-12, NaN at r/a = 1e4 for
# beta = 180, and lost Phi'' at alpha = 1 and small r.
_TWOPOWER_DERIV_CASES = [  # alpha, beta, quantity, x, value, d/da
    (1.5, 3.000000000001, "Rforce", 0.4999, -0.90157653081407408, 0.89115086186945043),
    (1.5, 3.000000000001, "R2deriv", 5.0, -0.016349354802538274, 0.010272145872264871),
    (1.5, 2.999999999999, "Rzderiv", 0.4999, -1.3085405546703867, 1.4260790635900432),
    (1.5, 2.999999999999, "mass", 5.0, 2.5264560929906962, -1.2678762905234898),
    (1.0, 3.5, "R2deriv", 1e-08, -0.9645061475212197, 2.4112653477044776),
    (1.0, 3.5, "Rforce", 1e-08, -0.69444443287037057, 1.1574073784722229),
    (0.0, 5.5, "Rzderiv", 1e-08, -7.6388884916666802e-9, 2.5462961307870437e-8),
    (1.5, 180.0, "Rforce", 10000.0, -5.2159588732345013e-12, 0.0),
    (1.5, 180.0, "R2deriv", 10000.0, -8.6932647887241688e-16, 0.0),
    (0.0, 180.0, "Rforce", 0.03, -4.9019920225253475e-4, 1.6978802841276199e-4),
    (0.0, 180.0, "Rzderiv", 0.03, -0.016891359635497198, 0.011868677714290157),
    (0.5, 2.0001, "Rforce", 10.0, -0.094279530319555269, 0.10029818879539434),
    (0.5, 2.0001, "R2deriv", 10.0, -5.6834361737197777e-3, 7.2191900889775899e-3),
    (1.999999999999, 4.0, "Rzderiv", 0.1, -105.60146923742811, 83.479422322162556),
    (2.5, 2.5, "R2deriv", 1.0, -3.4722222222222226, 1.4467592592592595),
    (2.9, 2.1, "Rforce", 10.0, -0.27779204975531618, 0.099220270259507471),
    # alpha > beta (p + q < 0) above the split with |beta - 3| < 0.05: the
    # reflected series' coefficients once took log1p(-2)
    (2.99, 2.98, "Rforce", 2.0, -34.968714956363698, 0.29458306649690608),
    (2.99, 2.98, "mass", 2.0, 201.41979814865490, -1.6967984630221790),
    (2.999, 2.97, "R2deriv", 50.0, -0.018585857782860795, 1.7108229514431519e-5),
    (2.97, 2.96, "Rzderiv", 10000.0, -7.3340176868906600e-11, 1.9807904837937884e-12),
]


def _twopower_deriv_quantity(pot, q, r):
    if q == "mass":
        return pot._amp * pot._mass(r)
    if q == "Rzderiv":
        return pot._amp * pot._Rzderiv(0.6 * r, 0.8 * r)
    return pot._amp * getattr(pot, "_" + q)(r, 0.0 * r)


@pytest.mark.parametrize(
    "alpha,beta,q,x,ref,dref",
    _TWOPOWER_DERIV_CASES,
    ids=[f"{c[2]}-a{c[0]}-b{c[1]}-x{c[3]}" for c in _TWOPOWER_DERIV_CASES],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_twopower_derivs_value_and_parameter_gradient(
    backend_name, alpha, beta, q, x, ref, dref
):
    a0, amp0 = 1.2, 2.0
    r = x * a0
    # d/da is ~1e-700 at r/a = 1e4, beta = 180: its terms are ~value/a, which
    # AD cancels to rounding
    datol = 1e-15 * abs(ref) / a0

    def f(a, amp):
        pot = TwoPowerSphericalPotential(amp=amp, a=a, alpha=alpha, beta=beta)
        return _twopower_deriv_quantity(pot, q, r * (a * 0.0 + 1.0))

    if backend_name == "jax":
        args = (jnp.asarray(a0), jnp.asarray(amp0))
        val = float(f(*args))
        da, damp = (float(g) for g in jax.grad(f, argnums=(0, 1))(*args))
        # and under jit: static-length series
        jval = float(jax.jit(f)(*args))
        jda = float(jax.jit(jax.grad(f))(*args))
        assert abs(jval - val) <= 1e-15 * abs(val), (jval, val)
        assert abs(jda - da) <= 1e-13 * abs(dref) + datol, (jda, da)
    else:
        a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        amp = torch.tensor(amp0, dtype=torch.float64, requires_grad=True)
        out = f(a, amp)
        da, damp = (float(g) for g in torch.autograd.grad(out, (a, amp)))
        val = float(out.detach())
    assert abs(val / ref - 1.0) < 5e-14, f"{backend_name}: {val} vs {ref}"
    assert abs(damp / (ref / amp0) - 1.0) < 5e-14, f"d/damp {damp}"
    assert abs(da - dref) <= 1e-12 * abs(dref) + datol, f"d/da {da} vs {dref}"


@pytest.mark.parametrize("method", _TWOPOWER_METHODS)
@pytest.mark.parametrize("pot", _TWOPOWER_GENERIC, ids=_TWOPOWER_GENERIC_IDS)
def test_twopower_generic_jax_jit(pot, method):
    # The routed methods must survive jax.jit (pre-routing: scipy.special on a
    # tracer raised TracerArrayConversionError) and match the numpy values.
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    f = getattr(pot, method)
    R0, z0 = 1.3, 0.4
    ref = float(f(numpy.asarray(R0), numpy.asarray(z0)))
    assert_jit_matches_eager(
        lambda R: f(R, jnp.asarray(z0)),
        jnp.asarray(R0),
        rtol=1e-12,
        atol=0.0,
        ref=ref,
        err_msg=method,
    )


###############################################################################
# EinastoPotential._calculate_dn: the shape parameter d_n is the root of
# 2*Gamma_upper(3n, x) - Gamma(3n) = 0, solved at construction time. It used
# scipy.optimize.fsolve on a scipy.special residual, so d(d_n)/dn did not exist
# on any backend. It now goes through galpy's brentq (implicit function theorem)
# for backend inputs, while the numpy path keeps fsolve verbatim (routing numpy
# through a bracketing solver would move its last bits).
###############################################################################

_EIN_N = 4.0
_EIN_DN = 11.668363153044764  # numpy fsolve value, pinned for byte-identity
# Backend-vs-numpy agreement floor for the d_n solve, set from the measured
# accuracy of each backend's OWN gammaincc rather than picked round: brentq
# converges to the root of the function it is handed, so the root error is
# |Q_backend - Q_scipy| / pdf. jax tracks scipy to <=9e-15 for every a here.
# torch switches igammac algorithm near a=3n~20 and loses ~6 digits above it
# (|dQ| <= 2.2e-16 for a<=12, but 2.4e-10..4.0e-10 for a>=21), which maps to a
# 1.6e-10 root shift at n=10. Upstream torch, not the solver: the residual at
# the returned root sits at torch's own noise floor.
_EIN_TOL_FWD = {"jax": 1e-12, "torch": 1e-9}


def _ein_dn(n, guess_at=_EIN_N):
    """d_n for index ``n``; ``guess_at`` sets the numpy fsolve starting guess.

    The guess feeds scipy, so it stays a plain float and is kept separate from
    ``n``, which may be a backend array. The backend path ignores it (it
    brackets instead of guessing).
    """
    from galpy.potential import EinastoPotential

    pot = EinastoPotential(amp=1.0, n=guess_at, rs=1.0)
    return pot._calculate_dn(n, pot._estimate_dn(guess_at))


def test_einasto_dn_numpy_is_byte_identical_and_solves_its_equation():
    """numpy path untouched: same bits as before, and it really is the root."""
    from scipy import special

    dn = _ein_dn(_EIN_N)
    assert dn == _EIN_DN, f"numpy d_n moved: {dn!r} != {_EIN_DN!r}"
    g = special.gamma(3 * _EIN_N)
    resid = 2 * (special.gammaincc(3 * _EIN_N, dn) * g) - g
    assert abs(resid) < 1e-9 * g, f"d_n is not the root: residual {resid!r}"


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0, 10.0])
@pytest.mark.parametrize("backend_name", [b for b in BACKENDS if b != "numpy"])
def test_einasto_dn_forward_matches_numpy(backend_name, n):
    """The backend brentq lands on the same root as the numpy fsolve.

    Swept over the index range so the ``[1e-12, 3n]`` bracket is exercised
    rather than one lucky n: d_n is the MEDIAN of Gamma(3n), which is below the
    mean 3n for every n>0, so 3n brackets the root from above throughout.

    Covers torch as well as jax: torch's FORWARD solve works, only its gradient
    is blocked upstream (see the next test's docstring).
    """
    ref = _ein_dn(n, guess_at=n)  # the value galpy ships today, via scipy
    # this module passes dtype explicitly rather than setting a global default
    # (see the float64 tensors above); a bare torch.tensor(n) would be float32
    # and land ~2e-6 off -- float32 precision, not a solver error.
    xp_n = (
        jnp.asarray(n)
        if backend_name == "jax"
        else torch.tensor(n, dtype=torch.float64)
    )
    raw = _ein_dn(xp_n, guess_at=n)
    # the value alone cannot see a numpy fallback: fsolve accepts a backend
    # array and returns the RIGHT np.float64, so check we stayed on-backend.
    assert is_backend_array(raw), f"{backend_name} fell back to numpy: {raw!r}"
    got = float(as_numpy(raw))
    tol = _EIN_TOL_FWD[backend_name]
    assert abs(got - ref) < tol * ref, (
        f"{backend_name} n={n}: {got!r} vs numpy {ref!r} "
        f"(rel {abs(got - ref) / ref:.3e}, tol {tol:.0e})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_einasto_dn_gradient_vs_finite_difference(backend_name):
    """d(d_n)/dn by the implicit function theorem vs a central difference.

    Both backends. torch gets here through galpy's own ``_IncGamma`` autograd
    rule in ``backend.special._fallback.gammainc``: ``torch.special.gammainc``
    taken directly still raises ``NotImplementedError`` for the derivative
    w.r.t. its ORDER, which is precisely the derivative the implicit-function
    theorem needs. This test was jax-only until that fallback existed.

    The FD reference runs on the NUMPY path, so this compares the backend
    derivative against an independent evaluation rather than against itself.
    The h=1e-5 truncation error is the floor: both backends land 1.5e-10 from
    it (and far closer to each other than either is to it), so the 1e-8 bar
    measures the AD rather than the finite difference.
    """
    h = 1e-5
    fd = (_ein_dn(_EIN_N + h) - _ein_dn(_EIN_N - h)) / (2.0 * h)
    got = _grad_wrt(backend_name, _ein_dn, _EIN_N)
    assert abs(got - fd) < 1e-8 * abs(fd), (
        f"{backend_name}: d(d_n)/dn={got!r} vs finite difference {fd!r} "
        f"(rel err {abs(got - fd) / abs(fd):.3e})"
    )


@pytest.mark.parametrize("backend_name", [b for b in BACKENDS if b != "numpy"])
def test_einasto_dn_solves_on_the_backend_inside_a_forced_context(backend_name):
    """Inside ``use(backend, force=True)`` the solve must run ON the backend.

    Deliberately NOT a "stays on numpy" assertion. A forced backend means
    everything runs there, setup included, so the contract this pins is that
    the brentq path is reachable under a force -- not that scipy survives it.
    Coercing the potential's own scalar params so the CONSTRUCTOR reaches this
    path is the follow-up; this test pins the half that is true today.
    """
    from galpy import backend as _backend

    xp_n = (
        jnp.asarray(_EIN_N)
        if backend_name == "jax"
        else torch.tensor(_EIN_N, dtype=torch.float64)
    )
    with _backend.use(backend_name, force=True):
        raw = _ein_dn(xp_n)
    assert is_backend_array(raw), f"{backend_name} forced: fell to numpy: {raw!r}"
    got = float(as_numpy(raw))
    tol = _EIN_TOL_FWD[backend_name]
    assert abs(got - _EIN_DN) < tol * _EIN_DN, f"{backend_name} forced: {got!r}"


# Small r: the backend incomplete-beta series against numpy's scipy hyp2f1 (the
# old closed form needed 2F1 at -a/r ~ -1e5 and cancelled ~a/r-fold there).
@pytest.mark.parametrize("backend_name", [b for b in BACKENDS if b != "numpy"])
@pytest.mark.parametrize("alpha,beta", [(1.5, 3.5), (1.5, 4.5), (0.5, 4.0), (1.0, 5.0)])
def test_twopower_generic_potential_at_small_r(backend_name, alpha, beta):
    pot = TwoPowerSphericalPotential(amp=1.3, a=1.1, alpha=alpha, beta=beta)
    assert pot._specialSelf is None  # the generic hyp2f1 branch
    R = 1.1 * numpy.array([1e-5, 3e-5, 1e-4, 1e-3])
    z = numpy.zeros_like(R)
    got = as_numpy(pot._evaluate(_asarray(backend_name, R), _asarray(backend_name, z)))
    numpy.testing.assert_allclose(got, pot._evaluate(R, z), rtol=1e-13, atol=0.0)
