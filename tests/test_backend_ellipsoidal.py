###############################################################################
# test_backend_ellipsoidal.py: per-family backend tests for the ellipsoidal /
# triaxial potentials (EllipsoidalPotential base + subclasses).
#
# Proves, for the migrated compute methods of each potential:
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14),
#   2. autodiff (jax.grad / torch.autograd) on a migrated scalar potential
#      (_evaluate or, where _evaluate is deferred, _Rforce) matches central
#      finite differences (rtol=1e-5),
#   3. the per-instance numpy quadrature cache is never touched by the traced
#      (jax/torch) path (so autodiff is pure and reentrant).
#
# Scope notes (see the module docstrings / the PR's deferred list):
#   * The Gauss-Legendre quadrature path (glorder set, the default) is migrated;
#     the scipy.integrate fallback (glorder=None) is deferred (Pspecial PR).
#   * TwoPowerTriaxialPotential._evaluate uses scipy.special.hyp2f1 in _psi and
#     is therefore NOT migrated (its forces/2nd-derivs/dens, which only use the
#     pure-arithmetic _mdens, ARE migrated).
#   * The rotated (zvec/pa) compute path -- _rotate_to_aligned /
#     _rotate_force_back applied to forces, density, and the potential -- is
#     backend-agnostic and is exercised here with explicit rotated instances.
#   * _mass is migrated for the CLOSED-FORM subclasses (PerfectEllipsoid,
#     TriaxialHernquist, TriaxialJaffe, TriaxialNFW); it remains Pspecial-blocked
#     for TwoPowerTriaxial (scipy.special.hyp2f1), TriaxialGaussian
#     (scipy.special.erf), and the generic EllipsoidalPotential / PowerTriaxial
#     base (scipy.integrate.quad), so those are not parametrized below.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    PerfectEllipsoidPotential,
    PowerTriaxialPotential,
    TriaxialGaussianPotential,
    TriaxialHernquistPotential,
    TriaxialJaffePotential,
    TriaxialNFWPotential,
    TwoPowerTriaxialPotential,
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

# Compute methods migrated for every potential in this family (forces, 2nd
# derivatives, density). All use only the pure-arithmetic _mdens/_mdens_deriv.
COMMON_METHODS = [
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
# _evaluate is migrated for every subclass whose _psi is namespace-clean; it is
# deferred for TwoPowerTriaxialPotential (psi uses scipy.special.hyp2f1).
EVAL = ["_evaluate"]

# (name, instance, methods); aligned (default) instances exercise the migrated
# Gauss-Legendre quadrature path.
_CASES = [
    ("Perfect", PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7), EVAL),
    ("Gauss", TriaxialGaussianPotential(amp=1.3, sigma=1.5, b=0.9, c=0.7), EVAL),
    ("Power", PowerTriaxialPotential(amp=1.3, alpha=1.2, b=0.9, c=0.7), EVAL),
    ("Hernquist", TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7), EVAL),
    ("Jaffe", TriaxialJaffePotential(amp=1.3, a=1.5, b=0.9, c=0.7), EVAL),
    ("NFW", TriaxialNFWPotential(amp=1.3, a=1.5, b=0.9, c=0.7), EVAL),
    # TwoPower: _evaluate deferred (hyp2f1), but forces/2nd-derivs/dens migrated.
    (
        "TwoPower",
        TwoPowerTriaxialPotential(amp=1.3, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.7),
        [],
    ),
]

# Rotated (zvec + pa) instances. The rotated compute path (_rotate_to_aligned /
# _rotate_force_back) is backend-agnostic; a prior review flagged it had no
# coverage. Only the forces, density, and (where migrated) potential are defined
# for rotated frames -- the 2nd derivatives raise NotImplementedError -- so the
# rotated cases use a reduced method list. TwoPower's _evaluate stays deferred.
_ROT_KW = dict(zvec=[0.0, 1.0, 1.0], pa=0.3)
_ROT_METHODS = ["_Rforce", "_zforce", "_phitorque", "_dens"]
_ROT_CASES = [
    (
        "Perfect-rot",
        PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "Gauss-rot",
        TriaxialGaussianPotential(amp=1.3, sigma=1.5, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "Power-rot",
        PowerTriaxialPotential(amp=1.3, alpha=1.2, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "Hernquist-rot",
        TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "Jaffe-rot",
        TriaxialJaffePotential(amp=1.3, a=1.5, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "NFW-rot",
        TriaxialNFWPotential(amp=1.3, a=1.5, b=0.9, c=0.7, **_ROT_KW),
        EVAL,
    ),
    (
        "TwoPower-rot",
        TwoPowerTriaxialPotential(
            amp=1.3, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.7, **_ROT_KW
        ),
        [],
    ),
]

# Potentials whose closed-form _mass is migrated to the backend namespace
# (PerfectEllipsoid: atan; TriaxialNFW: log; Hernquist/Jaffe: pure arithmetic).
# The others keep a scipy.special / scipy.integrate _mass (Pspecial-blocked).
_MASS_POTS = [
    pytest.param(PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Perfect"),
    pytest.param(
        TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Hernquist"
    ),
    pytest.param(TriaxialJaffePotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Jaffe"),
    pytest.param(TriaxialNFWPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="NFW"),
]

# Flatten to (case_id, pot, method) for value-parity parametrization.
_VALUE_PARAMS = []
for _name, _pot, _eval in _CASES:
    for _m in _eval + COMMON_METHODS:
        _VALUE_PARAMS.append(pytest.param(_pot, _m, id=f"{_name}-{_m}"))
# Rotated value-parity params (forces / dens / migrated potential only).
for _name, _pot, _eval in _ROT_CASES:
    for _m in _eval + _ROT_METHODS:
        _VALUE_PARAMS.append(pytest.param(_pot, _m, id=f"{_name}-{_m}"))

# Rotated potentials whose _evaluate is migrated (rotated autodiff check).
_ROT_EVAL_POTS = [
    pytest.param(pot, id=name) for (name, pot, ev) in _ROT_CASES if ev == EVAL
]

_RS = [0.5, 1.0, 2.0]
_ZS = [0.1, 0.2, 0.3]
_PHIS = [0.3, 0.6, 0.9]


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


@pytest.mark.parametrize("pot,method", _VALUE_PARAMS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, method):
    # Reference is always numpy.
    ref = numpy.asarray(
        getattr(pot, method)(
            numpy.asarray(_RS), numpy.asarray(_ZS), numpy.asarray(_PHIS)
        )
    )
    got = _tonumpy(
        getattr(pot, method)(
            _asarray(backend_name, _RS),
            _asarray(backend_name, _ZS),
            _asarray(backend_name, _PHIS),
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- exact analytic-identity gradient checks ----------------------------------
# galpy sign conventions: Rforce=-dPhi/dR, zforce=-dPhi/dz, phitorque=-dPhi/dphi;
# R2deriv=d2Phi/dR2, etc. Under autodiff this gives, for each (lower, var, higher)
# triple below, AD(lower wrt var) == -higher EXACTLY (not just to FD precision):
#
#   AD(_evaluate   wrt R)   == -_Rforce       AD(_Rforce wrt R)   == -_R2deriv
#   AD(_evaluate   wrt z)   == -_zforce       AD(_Rforce wrt z)   == -_Rzderiv
#   AD(_evaluate   wrt phi) == -_phitorque    AD(_Rforce wrt phi) == -_Rphideriv
#   AD(_zforce     wrt z)   == -_z2deriv      AD(_zforce wrt phi) == -_phizderiv
#   AD(_phitorque  wrt phi) == -_phi2deriv
#
# This cross-validates the hand-coded analytic forces and the (phi-dependent)
# triaxial Hessian against autodiff, far more stringently than finite differences
# (these identities replace the now-subsumed FD _evaluate / _Rforce checks).
# Variable name -> positional index into (R, z, phi).
_VAR_IDX = {"R": 0, "z": 1, "phi": 2}

# (lower_method, var, higher_method): AD(lower wrt var) == -higher.
_IDENTITY_PAIRS = [
    ("_evaluate", "R", "_Rforce"),
    ("_evaluate", "z", "_zforce"),
    ("_evaluate", "phi", "_phitorque"),
    ("_Rforce", "R", "_R2deriv"),
    ("_Rforce", "z", "_Rzderiv"),
    ("_Rforce", "phi", "_Rphideriv"),
    ("_zforce", "z", "_z2deriv"),
    ("_zforce", "phi", "_phizderiv"),
    ("_phitorque", "phi", "_phi2deriv"),
]

# Off-axis, off-plane, non-zero-phi smooth point so every derivative (including
# the phi-direction ones, which vanish on axis) is exercised and nonzero.
_R0, _Z0, _PHI0 = 1.3, 0.4, 0.5


def _ad_grad(backend_name, method, var, point):
    """AD gradient of ``method`` (a scalar-returning bound potential method) with
    respect to one of (R, z, phi) at ``point=(R0, z0, phi0)``, as a python float.

    Mirrors the canonical jax.grad / torch.autograd pattern: a fresh leaf tensor
    per backward, scalar output."""
    idx = _VAR_IDX[var]
    if backend_name == "jax":

        def f(v):
            args = [jnp.asarray(point[0]), jnp.asarray(point[1]), jnp.asarray(point[2])]
            args[idx] = v
            return method(*args)

        return float(jax.grad(f)(jnp.asarray(point[idx])))
    # torch: a fresh leaf tensor that requires grad for the chosen variable.
    args = [
        torch.tensor(point[0], dtype=torch.float64),
        torch.tensor(point[1], dtype=torch.float64),
        torch.tensor(point[2], dtype=torch.float64),
    ]
    leaf = torch.tensor(point[idx], dtype=torch.float64, requires_grad=True)
    args[idx] = leaf
    method(*args).backward()
    return float(leaf.grad)


def _method_migrated(name, eval_migrated):
    """Whether ``<name>`` is namespace-migrated (callable on a traced backend).

    Forces, 2nd derivatives, and density depend only on the pure-arithmetic
    _mdens/_mdens_deriv and are migrated for every potential here; _evaluate is
    deferred for potentials whose _psi uses scipy.special (TwoPower)."""
    if name == "_evaluate":
        return eval_migrated
    return name in COMMON_METHODS


# Build (pot, lower, var, higher) params for every identity pair whose BOTH
# methods are migrated for that (aligned) potential. TwoPower keeps the six
# force/2nd-deriv pairs but drops the three _evaluate pairs (psi uses hyp2f1).
_IDENTITY_PARAMS = []
for _name, _pot, _eval in _CASES:
    _eval_migrated = _eval == EVAL
    for _lower, _var, _higher in _IDENTITY_PAIRS:
        if _method_migrated(_lower, _eval_migrated) and _method_migrated(
            _higher, _eval_migrated
        ):
            _IDENTITY_PARAMS.append(
                pytest.param(
                    _pot, _lower, _var, _higher, id=f"{_name}-{_lower}-d{_var}"
                )
            )


@pytest.mark.parametrize("pot,lower,var,higher", _IDENTITY_PARAMS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_and_hessian_identities(backend_name, pot, lower, var, higher):
    # AD(lower wrt var) == -higher, exactly (rtol=1e-9). Both methods share the
    # same Gauss-Legendre quadrature nodes, so autodiff of the lower method
    # reproduces the analytic higher method to round-off, not just FD precision.
    point = (_R0, _Z0, _PHI0)
    ad = _ad_grad(backend_name, getattr(pot, lower), var, point)
    ref = -float(
        getattr(pot, higher)(
            numpy.asarray(_R0), numpy.asarray(_Z0), numpy.asarray(_PHI0)
        )
    )
    numpy.testing.assert_allclose(ad, ref, rtol=1e-9)


@pytest.mark.parametrize("pot", _MASS_POTS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_mass_value_parity(backend_name, pot):
    # The closed-form _mass (via the public mass()) is migrated for these
    # subclasses; numpy / jax / torch must agree.
    Rs = numpy.asarray([0.5, 1.0, 2.0])
    ref = numpy.asarray(pot.mass(Rs))
    got = _tonumpy(pot.mass(_asarray(backend_name, [0.5, 1.0, 2.0])))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", _MASS_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_mass_grad_vs_finite_difference(backend_name, pot):
    # The migrated closed-form _mass is differentiable; check d mass / dR.
    R0, eps = 1.7, 1e-6

    def m_np(R):
        return float(pot._mass(numpy.asarray(R)))

    fd = (m_np(R0 + eps) - m_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(lambda R: pot._mass(R))(jnp.asarray(R0)))
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._mass(R)
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("pot", _ROT_EVAL_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_rotated_vs_finite_difference(backend_name, pot):
    # Autodiff through the rotated (zvec/pa) potential path vs central FD.
    R0, z0, phi0 = 1.3, 0.4, 0.5
    eps = 1e-6

    def phi_np(R):
        return float(
            pot._evaluate(numpy.asarray(R), numpy.asarray(z0), numpy.asarray(phi0))
        )

    fd = (phi_np(R0 + eps) - phi_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0), jnp.asarray(phi0)))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._evaluate(
            R,
            torch.tensor(z0, dtype=torch.float64),
            torch.tensor(phi0, dtype=torch.float64),
        )
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_traced_path_does_not_touch_cache(backend_name):
    # The refactored quadrature cache (_force_hash / _cached_F*) must be written
    # ONLY by the numpy path; the traced path must leave self-state untouched so
    # autodiff is pure and reentrant.
    pot = PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7)
    assert pot._force_hash is None
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    pot._Rforce(R, z, phi)
    pot._R2deriv(R, z, phi)
    assert pot._force_hash is None
    assert pot._2ndderiv_hash is None


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_numpy_cache_unaffected_by_traced_call(backend_name):
    # A numpy evaluation fills the cache; a subsequent traced call at the same
    # point must not corrupt it, and a later numpy call must reuse the (correct)
    # cached force.
    pot = PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7)
    Rn, zn, phin = numpy.asarray(_RS), numpy.asarray(_ZS), numpy.asarray(_PHIS)
    ref_z = numpy.asarray(pot._zforce(Rn, zn, phin))  # fills cache via zforce
    h = pot._force_hash
    # Traced call at the same point.
    pot._Rforce(
        _asarray(backend_name, _RS),
        _asarray(backend_name, _ZS),
        _asarray(backend_name, _PHIS),
    )
    assert pot._force_hash == h  # numpy cache untouched by traced call
    got_z = numpy.asarray(pot._zforce(Rn, zn, phin))  # reuse cache
    numpy.testing.assert_allclose(got_z, ref_z, rtol=1e-14, atol=0.0)


def test_evaluate_xyz_namespace_fallback():
    # _evaluate_xyz infers the backend from its (x,y,z) arguments when called
    # without an explicit ``xp`` (the public _evaluate always passes one, so this
    # exercises the defensive get_namespace fallback). It must match _evaluate.
    pot = PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7)
    R, z, phi = 0.7, 0.3, 0.0  # aligned, axisymmetric instance: x=R, y=0, z=z
    x = numpy.asarray(R)
    y = numpy.asarray(0.0)
    zz = numpy.asarray(z)
    got = numpy.asarray(pot._evaluate_xyz(x, y, zz))  # no xp -> get_namespace
    ref = numpy.asarray(
        pot._evaluate(numpy.asarray(R), numpy.asarray(z), numpy.asarray(phi))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-14, atol=0.0)


# ---------------------------------------------------------------------------
# Degenerate-regime audit (m=0 center, on-axis / in-plane, R=inf).
#
# The ellipsoidal family has no xp.where dead-branch hazards in its compute
# methods (the only xp.where, ``y = where(isinf(R), 0., y)`` in _evaluate, has
# a finite live branch everywhere). The genuine singularities are at the
# ellipsoidal radius m -> 0 (center / on the symmetry axes), where the cuspy
# subclasses' _mdens / _mdens_deriv / _psi and the 2nd-derivative integrand's
# densDeriv(m)/m diverge. These tests guard that:
#   * the singular point itself never *raises* for scalar input in any backend
#     (a genuine inf / nan return is acceptable; a Python exception is a bug),
#   * numpy / jax / torch agree at VALID points just off the singular regimes,
#   * reverse-mode AD (jax.grad / torch.autograd) is NaN-free at those valid
#     near-singular points (a singular dead term would NaN-poison 0*inf=nan),
#   * the 2nd-derivative quadrature stays byte-identical to pre-migration numpy
#     (the fused accumulation must reassociate exactly like the original ``+=``).
# ---------------------------------------------------------------------------

# Aligned instances spanning the cuspy (m=0) and smooth-core density families.
_DEGEN_POTS = [
    pytest.param(PerfectEllipsoidPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Perfect"),
    pytest.param(
        TriaxialGaussianPotential(amp=1.3, sigma=1.5, b=0.9, c=0.7), id="Gauss"
    ),
    pytest.param(PowerTriaxialPotential(amp=1.3, alpha=1.2, b=0.9, c=0.7), id="Power"),
    pytest.param(
        TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Hernquist"
    ),
    pytest.param(TriaxialJaffePotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="Jaffe"),
    pytest.param(TriaxialNFWPotential(amp=1.3, a=1.5, b=0.9, c=0.7), id="NFW"),
    pytest.param(
        TwoPowerTriaxialPotential(amp=1.3, a=1.5, alpha=1.5, beta=3.5, b=0.9, c=0.7),
        id="TwoPower",
    ),
]

# Methods migrated to the backend namespace for every subclass (forces +
# 2nd-derivatives; _evaluate is deferred for TwoPower so excluded here).
_DEGEN_METHODS = [
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_R2deriv",
    "_z2deriv",
    "_phi2deriv",
    "_Rzderiv",
    "_Rphideriv",
    "_phizderiv",
]

# VALID points just off the singular regimes (never AT them): near the center,
# near the z-axis (R->0), near the disk plane (z->0).
_NEAR_DEGEN = [
    pytest.param((1e-5, 1e-5, 0.3), id="near-center"),
    pytest.param((1e-6, 0.8, 0.2), id="near-zaxis"),
    pytest.param((1.0, 1e-7, 0.4), id="near-plane"),
]

# Exact singular regimes (m=0 along the axes, the origin, and R=inf).
_AT_DEGEN = [
    pytest.param((0.0, 0.0, 0.0), id="center"),
    pytest.param((0.0, 0.9, 0.0), id="on-zaxis"),
    pytest.param((1.2, 0.0, 0.3), id="in-plane"),
    pytest.param((numpy.inf, 0.5, 0.2), id="R-inf"),
]


@pytest.mark.parametrize("point", _AT_DEGEN)
@pytest.mark.parametrize("method", _DEGEN_METHODS)
@pytest.mark.parametrize("pot", _DEGEN_POTS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_degenerate_regime_no_scalar_raise(backend_name, pot, method, point):
    # At a genuine singularity a finite/inf/nan scalar result is fine; a Python
    # exception is not. (NotImplementedError for rotated 2nd-derivs is N/A here:
    # all _DEGEN_POTS are aligned.)
    R = _asarray(backend_name, point[0])
    z = _asarray(backend_name, point[1])
    phi = _asarray(backend_name, point[2])
    # Must not raise for any backend at the singular point.
    getattr(pot, method)(R, z, phi)


@pytest.mark.parametrize("point", _NEAR_DEGEN)
@pytest.mark.parametrize("method", _DEGEN_METHODS)
@pytest.mark.parametrize("pot", _DEGEN_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_degenerate_regime_value_parity(backend_name, pot, method, point):
    # numpy reference vs jax/torch at VALID points just off the singularities.
    ref = numpy.asarray(
        getattr(pot, method)(
            numpy.asarray(point[0]), numpy.asarray(point[1]), numpy.asarray(point[2])
        )
    )
    got = _tonumpy(
        getattr(pot, method)(
            _asarray(backend_name, point[0]),
            _asarray(backend_name, point[1]),
            _asarray(backend_name, point[2]),
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("var", [0, 1])  # grad wrt R (0) and z (1)
@pytest.mark.parametrize("point", _NEAR_DEGEN)
@pytest.mark.parametrize("method", _DEGEN_METHODS)
@pytest.mark.parametrize("pot", _DEGEN_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_degenerate_regime_ad_is_finite(backend_name, pot, method, point, var):
    # Reverse-mode AD must be NaN-free at VALID near-singular points (no dead
    # singular term poisoning the gradient via 0*inf=nan).
    if backend_name == "jax":

        def f(v):
            args = [jnp.asarray(point[0]), jnp.asarray(point[1]), jnp.asarray(point[2])]
            args[var] = v
            return getattr(pot, method)(*args)

        g = float(jax.grad(f)(jnp.asarray(point[var])))
    else:  # torch
        args = [
            torch.tensor(point[0], dtype=torch.float64),
            torch.tensor(point[1], dtype=torch.float64),
            torch.tensor(point[2], dtype=torch.float64),
        ]
        leaf = torch.tensor(point[var], dtype=torch.float64, requires_grad=True)
        args[var] = leaf
        getattr(pot, method)(*args).backward()
        g = float(leaf.grad)
    assert numpy.isfinite(g), f"non-finite AD gradient {g} for {method} at {point}"


def test_2ndderiv_quadrature_byte_identical_accumulation():
    # The migrated _2ndDerivInt_all groups the two added terms of each diagonal
    # entry as ``acc + (a + b)`` so the floating-point accumulation reassociates
    # EXACTLY like the pre-migration ``acc += a + b``. This guards against a
    # silent ULP-level drift in numpy 2nd-derivative output. We reconstruct the
    # reference with the explicit right-grouped accumulation and require a
    # byte-identical match.
    from galpy.potential.EllipsoidalPotential import _2ndDerivInt_all

    pot = TriaxialHernquistPotential(amp=1.3, a=1.5, b=0.9, c=0.7)
    x, y, z = 1.1973792922037507, 0.5062438450012458, 0.7
    got = _2ndDerivInt_all(
        x,
        y,
        z,
        pot._mdens,
        pot._mdens_deriv,
        pot._b2,
        pot._c2,
        numpy,
        glx=pot._glx,
        glw=pot._glw,
    )

    # Reference: explicit ``+=`` (right-grouped) accumulation.
    xx = xy = xz = yy = yz = zz = 0.0
    x2, y2, z2 = x**2, y**2, z**2
    for k in range(len(pot._glx)):
        s, w = pot._glx[k], pot._glw[k]
        t = 1.0 / s**2 - 1.0
        inv1t, invb2t, invc2t = (
            1.0 / (1.0 + t),
            1.0 / (pot._b2 + t),
            1.0 / (pot._c2 + t),
        )
        m = numpy.sqrt(x2 * inv1t + y2 * invb2t + z2 * invc2t)
        denom = numpy.sqrt(
            (1.0 + (pot._b2 - 1.0) * s**2) * (1.0 + (pot._c2 - 1.0) * s**2)
        )
        w_over_denom = w / denom
        dens_val = pot._mdens(m)
        dderiv_over_m = pot._mdens_deriv(m) / m
        xi, yi, zi = x * inv1t, y * invb2t, z * invc2t
        dd_xi = w_over_denom * dderiv_over_m * xi
        dd_yi = w_over_denom * dderiv_over_m * yi
        dd_zi = w_over_denom * dderiv_over_m * zi
        xx += dd_xi * xi + w_over_denom * dens_val * inv1t
        xy += dd_xi * yi
        xz += dd_xi * zi
        yy += dd_yi * yi + w_over_denom * dens_val * invb2t
        yz += dd_yi * zi
        zz += dd_zi * zi + w_over_denom * dens_val * invc2t
    ref = (xx, xy, xz, yy, yz, zz)

    for g, r in zip(got, ref):
        assert g.tobytes() == numpy.asarray(r).tobytes()
