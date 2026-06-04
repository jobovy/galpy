###############################################################################
# test_backend_dissipative.py: per-family backend tests for the dissipative /
# velocity-dependent forces (f = f(x, v)).
#
# FAMILY: DissipativeForce, planarDissipativeForce,
#         ChandrasekharDynamicalFrictionForce, FDMDynamicalFrictionForce.
#
# Status of this family under the backend migration (P2.7):
#   * The *base* classes (DissipativeForce, planarDissipativeForce) have no
#     numpy-calling private compute methods: the public Rforce/zforce/phitorque
#     are backend-agnostic delegators and planarDissipativeForceFromFull-
#     DissipativeForce just forwards to the wrapped 3D force. Nothing to swap.
#   * The actual force compute path of Chandrasekhar/FDM dynamical friction
#     (_Rforce/_zforce/_phitorque -> _calc_force / frictionFactor) is NOT yet
#     backend-agnostic and is DEFERRED, because it irreducibly depends on:
#       - scipy.special.erf (and scipy.special.sici for FDM) -- the Chandrasekhar
#         X-function -- which is out of scope until the later "Pspecial" backend
#         router lands;
#       - a scipy.interpolate spline (self.sigmar) for the velocity dispersion;
#       - a mutable per-instance input-hash cache (_force_hash / _cached_force,
#         built with hashlib.md5(numpy.array(...))), which coerces tracers to
#         numpy and is the exact cache hazard called out by the convention;
#       - data-dependent scalar branching (if r < minr / if r > maxr /
#         if kr > 2*M_sigma ...).
#
# So this module does not assert numpy/jax/torch *value parity* of migrated
# compute methods (there are none yet for this family). Instead it:
#   1. pins the numpy path of the velocity-dependent forces (regression guard so
#      a future migration cannot silently change numpy values), and codifies the
#      velocity calling convention v=[vR, vT, vz];
#   2. documents, via strict xfail, that jax/torch differentiability of the
#      dissipative compute path is currently blocked -- when the scipy.special
#      router + cache/branching refactor land, these xfails will start passing
#      and flag that the migration of this family is complete.
#
# Mirrors tests/test_backend_pilot.py for structure; self-skips on backends that
# are not installed.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    FDMDynamicalFrictionForce,
)

# This module manages backends explicitly, so it is exempt from the global
# --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends (numpy always present).
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
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


# --- factories ----------------------------------------------------------------
# Deterministic, scipy-light configurations: const_lnLambda removes the
# Coulomb-log branching and the default LogarithmicHaloPotential density gives an
# analytic sigmar = 1/sqrt(2), so the numpy reference values are stable.
def _make_cdf():
    return ChandrasekharDynamicalFrictionForce(
        GMs=0.05, const_lnLambda=10.0, minr=0.5, maxr=25.0
    )


def _make_fdm():
    return FDMDynamicalFrictionForce(
        GMs=0.05, const_lnLambda=10.0, const_FDMfactor=0.7, minr=0.5, maxr=25.0
    )


POT_FACTORIES = [_make_cdf, _make_fdm]
POT_IDS = ["ChandrasekharDynamicalFrictionForce", "FDMDynamicalFrictionForce"]

_R, _Z, _PHI = 1.5, 0.4, 0.3
_V = [0.3, 0.4, 0.1]  # (vR, vT, vz)

# Locked-in numpy reference values (regression guard); recomputed below so they
# stay in sync, but kept explicit here as documentation of the expected physics.
_NUMPY_REF = {
    "ChandrasekharDynamicalFrictionForce": {
        "Rforce": -0.040151523108560246,
        "zforce": -0.013383841036186752,
        "phitorque": -0.08030304621712052,
    },
    "FDMDynamicalFrictionForce": {
        "Rforce": -0.032863380449264804,
        "zforce": -0.010954460149754937,
        "phitorque": -0.06572676089852962,
    },
}


# --- 1. numpy-path regression + velocity convention ---------------------------
@pytest.mark.parametrize("factory, potid", zip(POT_FACTORIES, POT_IDS), ids=POT_IDS)
def test_numpy_value_regression(factory, potid):
    # Pin the numpy values of the velocity-dependent public forces so a future
    # backend migration of this family cannot silently change the numpy path.
    pot = factory()
    for method in ("Rforce", "zforce", "phitorque"):
        got = float(getattr(pot, method)(_R, _Z, v=_V))
        numpy.testing.assert_allclose(
            got, _NUMPY_REF[potid][method], rtol=1e-12, atol=1e-14
        )


@pytest.mark.parametrize("factory, potid", zip(POT_FACTORIES, POT_IDS), ids=POT_IDS)
def test_velocity_calling_convention(factory, potid):
    # Establish the velocity-dependent force convention: forces are called with
    # v = [vR, vT, vz], and the private compute methods carry the same signature
    # and factor cleanly into _cached_force * {v[0], v[1]*R, v[2]}.
    pot = factory()
    fR = pot._Rforce(_R, _Z, phi=_PHI, t=0.0, v=_V)
    fz = pot._zforce(_R, _Z, phi=_PHI, t=0.0, v=_V)
    tphi = pot._phitorque(_R, _Z, phi=_PHI, t=0.0, v=_V)
    # _cached_force is the velocity-independent prefactor shared by all three.
    base = pot._cached_force
    numpy.testing.assert_allclose(fR, base * _V[0], rtol=1e-12)
    numpy.testing.assert_allclose(fz, base * _V[2], rtol=1e-12)
    numpy.testing.assert_allclose(tphi, base * _V[1] * _R, rtol=1e-12)
    # Public force == _amp * private compute (decorator applies _amp).
    numpy.testing.assert_allclose(
        float(pot.Rforce(_R, _Z, v=_V)), pot._amp * fR, rtol=1e-12
    )


@pytest.mark.parametrize("factory, potid", zip(POT_FACTORIES, POT_IDS), ids=POT_IDS)
def test_below_minr_is_zero(factory, potid):
    # Inside minr the friction force is identically zero (data-dependent branch
    # that the deferred migration must reproduce with xp.where).
    pot = factory()
    assert float(pot.Rforce(0.2, 0.0, v=_V)) == 0.0
    assert float(pot.zforce(0.2, 0.0, v=_V)) == 0.0
    assert float(pot.phitorque(0.2, 0.0, v=_V)) == 0.0


# --- 2. deferred-backend documentation ---------------------------------------
# The dissipative compute path is not yet backend-agnostic; jax.grad on it raises
# TracerArrayConversionError (tracers are coerced to numpy by the hash cache and
# scipy.special). We assert that this is *currently* the behavior with a strict
# xfail so the day the Pspecial router + cache refactor land, these flip to
# PASS and signal that the family's migration is ready to be wired up here.
@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("factory, potid", zip(POT_FACTORIES, POT_IDS), ids=POT_IDS)
@pytest.mark.xfail(
    strict=True,
    reason="dissipative force compute path deferred: depends on scipy.special.erf"
    " / scipy interpolate sigmar / mutable input-hash cache (Pspecial PR)",
)
def test_jax_grad_evaluate_blocked(factory, potid):
    pot = factory()

    def f(R):
        return pot._Rforce(
            R,
            jnp.asarray(_Z),
            phi=jnp.asarray(_PHI),
            t=0.0,
            v=[jnp.asarray(_V[0]), jnp.asarray(_V[1]), jnp.asarray(_V[2])],
        )

    # If this stops raising (i.e. the path is migrated), the strict xfail fails,
    # prompting us to replace it with a real grad-vs-finite-difference check.
    g = float(jax.grad(f)(jnp.asarray(_R)))
    assert numpy.isfinite(g)
