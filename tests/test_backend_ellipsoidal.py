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

# Potentials whose _evaluate is migrated (used for the autodiff check).
_EVAL_POTS = [pytest.param(pot, id=name) for (name, pot, ev) in _CASES if ev == EVAL]
# Every potential supports a migrated _Rforce, used for the autodiff check on
# potentials whose _evaluate is deferred.
_ALL_POTS = [pytest.param(pot, id=name) for (name, pot, _ev) in _CASES]

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


@pytest.mark.parametrize("pot", _EVAL_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot):
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


@pytest.mark.parametrize("pot", _ALL_POTS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_rforce_vs_finite_difference(backend_name, pot):
    # _Rforce is migrated for every potential (depends only on _mdens); check its
    # gradient wrt R against central finite differences. This also covers the
    # autodiff path for potentials whose _evaluate is deferred (TwoPower).
    R0, z0, phi0 = 1.3, 0.4, 0.5
    eps = 1e-6

    def f_np(R):
        return float(
            pot._Rforce(numpy.asarray(R), numpy.asarray(z0), numpy.asarray(phi0))
        )

    fd = (f_np(R0 + eps) - f_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._Rforce(R, jnp.asarray(z0), jnp.asarray(phi0)))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._Rforce(
            R,
            torch.tensor(z0, dtype=torch.float64),
            torch.tensor(phi0, dtype=torch.float64),
        )
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


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
