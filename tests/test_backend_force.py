###############################################################################
# test_backend_force.py: multi-backend tests for the base-class force plumbing
# migrated to the galpy.backend namespace layer.
#
# Specifically this covers the bare-numpy compute path that was swept to xp:
#   * Force.rforce  (Force.py)  -- builds r = sqrt(R^2 + z^2) and projects the
#     cylindrical Rforce/zforce onto the spherical radial direction. This is the
#     SAME public method used by a DissipativeForce (it inherits Force.rforce),
#     so a ChandrasekharDynamicalFrictionForce exercises it too.
#   * SphericalPotential._mass (SphericalPotential.py) -- already on xp; pinned
#     here for the SphericalPotential subclass (HernquistPotential) per the
#     stage's "test a SphericalPotential subclass + a dissipative force" ask.
#
# For each it proves:
#   1. eager jax returns a jax array and eager torch returns a torch tensor
#      (a bare-numpy path would silently DETACH on jax / pass eager torch),
#   2. jax.grad through the method matches a central finite difference,
#   3. numpy returns the exact same value as before (byte / round-trip parity).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
# The DissipativeForce._isDissipative numpy.prod path is pure dispatch logic on
# a list of python bools (no coordinate array flows through it), so it is left
# on numpy and not exercised here.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    HernquistPotential,
    LogarithmicHaloPotential,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
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


def _module_of(x):
    return type(x).__module__


# A spherical-potential subclass (its public rforce goes through Force.rforce)
# and a dissipative force (which INHERITS Force.rforce). The dynamical-friction
# force needs a velocity and a background density.
_HERN = HernquistPotential(amp=1.3, a=1.1, normalize=False)
_LP = LogarithmicHaloPotential(normalize=1.0)
_CDF = ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.1, dens=_LP)

_R0, _Z0 = 1.0, 0.5
_V0 = [0.1, 0.2, 0.05]  # cylindrical velocity for the dissipative force


def _rforce_np(obj, R, z, v=None):
    if v is None:
        return float(obj.rforce(numpy.asarray(R), numpy.asarray(z)))
    return float(obj.rforce(numpy.asarray(R), numpy.asarray(z), v=numpy.asarray(v)))


def _rforce_backend(backend_name, obj, R, z, v=None):
    Rb, zb = _asarray(backend_name, R), _asarray(backend_name, z)
    if v is None:
        return obj.rforce(Rb, zb)
    return obj.rforce(Rb, zb, v=_asarray(backend_name, v))


# (label, force-like object, needs-velocity, backend_ok). backend_ok marks
# whether the object's OWN _Rforce/_zforce compute path is backend-clean: the
# HernquistPotential is (fully analytic), but ChandrasekharDynamicalFrictionForce
# still evaluates scipy.special.erf + numpy.exp/log internally, which on a
# jax/torch array both COERCE the result back to numpy AND emit a numpy-2.0
# __array_wrap__ DeprecationWarning (promoted to an error under the CI
# -W error::DeprecationWarning). So for the dissipative force only the numpy path
# is exercised here; its jax/torch parity, backend-array return, and AD are all
# deferred to the dissipative-force migration (a separate concern from the swept
# base-class Force.rforce plumbing, which the HernquistPotential case covers).
_CASES = [
    ("HernquistPotential", _HERN, False, True),
    ("ChandrasekharDynamicalFrictionForce", _CDF, True, False),
]
_CASE_IDS = [c[0] for c in _CASES]


@pytest.mark.parametrize("label,obj,needs_v,backend_ok", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_rforce_value_parity(backend_name, label, obj, needs_v, backend_ok):
    # Force.rforce (the swept compute path) must give identical values across
    # backends, including when reached through a DissipativeForce. For a force
    # whose OWN _Rforce is not backend-clean yet (Chandrasekhar: scipy.special.erf
    # + numpy.exp on the backend array, which both coerce to numpy AND emit a
    # numpy-2.0 __array_wrap__ DeprecationWarning that CI promotes to an error),
    # only the numpy path is meaningful -- the jax/torch cases are deferred with
    # the rest of the dissipative-force migration (see _CASES note).
    if not backend_ok and backend_name != "numpy":
        pytest.skip("force's own _Rforce not backend-clean yet (dissipative)")
    v = _V0 if needs_v else None
    ref = _rforce_np(obj, _R0, _Z0, v=v)
    got = float(_tonumpy(_rforce_backend(backend_name, obj, _R0, _Z0, v=v)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14, err_msg=label)


@pytest.mark.parametrize("label,obj,needs_v,backend_ok", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_returns_backend_array(backend_name, label, obj, needs_v, backend_ok):
    # A bare-numpy r = numpy.sqrt(...) inside Force.rforce would silently DETACH
    # the jax output to a plain numpy.ndarray and pass eager torch unnoticed;
    # the swept xp path must return the native backend array type. Only checkable
    # when the force's OWN _Rforce is backend-clean: Chandrasekhar's scipy.erf
    # coerces the result back to numpy, so it cannot return a backend array until
    # that subclass is migrated (deferred; see _CASES note).
    if not backend_ok:
        pytest.skip("force's own _Rforce not backend-clean yet (dissipative)")
    v = _V0 if needs_v else None
    out = _rforce_backend(backend_name, obj, _R0, _Z0, v=v)
    assert backend_name in _module_of(out), (
        f"{label}: rforce left the {backend_name} namespace ({_module_of(out)})"
    )


@pytest.mark.parametrize("label,obj,needs_v,backend_ok", _CASES, ids=_CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_finite_difference(
    backend_name, label, obj, needs_v, backend_ok
):
    # jax.grad / torch.autograd of Force.rforce wrt R must match a central finite
    # difference of the numpy path. Pre-sweep, jax.grad would have died with a
    # TracerArrayConversionError on the bare numpy.sqrt. Only meaningful when the
    # object's OWN _Rforce/_zforce are backend-differentiable (see _CASES note).
    if not backend_ok:
        pytest.skip("force's own _Rforce/_zforce not backend-differentiable yet")
    v = _V0 if needs_v else None
    eps = 1e-6
    fd = (
        _rforce_np(obj, _R0 + eps, _Z0, v=v) - _rforce_np(obj, _R0 - eps, _Z0, v=v)
    ) / (2 * eps)
    if backend_name == "jax":
        vj = None if v is None else jnp.asarray(v)
        if vj is None:
            ad = float(
                jax.grad(lambda R: obj.rforce(R, jnp.asarray(_Z0)))(jnp.asarray(_R0))
            )
        else:
            ad = float(
                jax.grad(lambda R: obj.rforce(R, jnp.asarray(_Z0), v=vj))(
                    jnp.asarray(_R0)
                )
            )
    else:
        R = torch.tensor(_R0, dtype=torch.float64, requires_grad=True)
        zt = torch.tensor(_Z0, dtype=torch.float64)
        if v is None:
            out = obj.rforce(R, zt)
        else:
            out = obj.rforce(R, zt, v=torch.tensor(v, dtype=torch.float64))
        (g,) = torch.autograd.grad(out, R)
        ad = float(g)
    assert numpy.isfinite(ad), f"{label}: rforce grad not finite"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, err_msg=f"{backend_name} {label}")


###############################################################################
# SphericalPotential._mass: -R^2 * _rforce(R), already on xp. Pin parity, the
# backend-array return type, and jax.grad-vs-FD for the HernquistPotential.
###############################################################################
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_spherical_mass_value_parity(backend_name):
    ref = float(_HERN._mass(numpy.asarray(_R0)))
    got = float(_tonumpy(_HERN._mass(_asarray(backend_name, _R0))))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_spherical_mass_returns_backend_array(backend_name):
    out = _HERN._mass(_asarray(backend_name, _R0))
    assert backend_name in _module_of(out), (
        f"_mass left the {backend_name} namespace ({_module_of(out)})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_spherical_mass_grad_vs_finite_difference(backend_name):
    eps = 1e-6
    fd = (
        float(_HERN._mass(numpy.asarray(_R0 + eps)))
        - float(_HERN._mass(numpy.asarray(_R0 - eps)))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(lambda R: _HERN._mass(R))(jnp.asarray(_R0)))
    else:
        R = torch.tensor(_R0, dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(_HERN._mass(R), R)
        ad = float(g)
    assert numpy.isfinite(ad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)
