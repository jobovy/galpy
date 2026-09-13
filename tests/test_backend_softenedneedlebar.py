###############################################################################
# test_backend_softenedneedlebar.py: backend tests for SoftenedNeedleBarPotential.
#
# The rotating "softened needle" bar evaluates a closed-form log potential in a
# bar-aligned frame (phid = phi - pa - omegab t). numpy inputs are byte-identical
# to before; jax/torch inputs route through the backend namespace so the whole
# force chain is jit/grad-safe. This module checks:
#   1. numpy / jax / torch value parity for the migrated first-order methods.
#   2. the rotating-bar t-anchoring gap: a scalar Python-float t under a FORCED
#      backend must not crash (torch.cos rejects a plain float) and must match
#      numpy -- the concrete-t de-rotation coefficient falls back to numpy.
#   3. the force gradient w.r.t. R h-converges to a central finite difference.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, use
from galpy.potential import (
    SoftenedNeedleBarPotential,
    evaluateRforces,
    evaluatezforces,
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

    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# Rotated (pa) and rotating (omegab) so the cos/sin de-rotation is exercised.
_SN = SoftenedNeedleBarPotential(amp=1.2, a=1.0, c=0.5, pa=0.3, omegab=1.4)
# Triaxial (b != 0) so the full closed-form Hessian (incl. the b-dependent pzz
# term) is exercised.
_SNB = SoftenedNeedleBarPotential(amp=1.2, a=1.0, b=0.3, c=0.5, pa=0.3, omegab=1.4)
_METHODS = ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_dens"]
_HESS_METHODS = [
    "_R2deriv",
    "_z2deriv",
    "_phi2deriv",
    "_Rzderiv",
    "_Rphideriv",
    "_phizderiv",
]
_R0, _Z0, _PHI0, _T0 = 1.2, 0.3, 0.4, 0.0


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, method):
    # numpy / jax / torch agree at a scalar (R, z, phi, t), all on the active
    # backend; the numpy path is byte-identical.
    ref = float(
        getattr(_SN, method)(
            numpy.asarray(_R0), numpy.asarray(_Z0), numpy.asarray(_PHI0), _T0
        )
    )
    got = float(
        as_numpy(
            getattr(_SN, method)(
                _asarray(backend_name, _R0),
                _asarray(backend_name, _Z0),
                _asarray(backend_name, _PHI0),
                _asarray(backend_name, _T0),
            )
        )
    )
    rtol, atol = (0.0, 0.0) if backend_name == "numpy" else (1e-11, 1e-13)
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=rtol,
        atol=atol,
        err_msg=f"SoftenedNeedle.{method} ({backend_name})",
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotating_scalar_t_under_forced_backend(backend_name):
    # The orbit integrator hands the force a scalar Python-float t while the
    # coordinates are backend arrays. Under a FORCED backend get_namespace(t) would
    # resolve to that backend and torch.cos(python_float) raises (the rotating-bar
    # t-anchoring gap); the concrete-t de-rotation coefficient must fall back to
    # numpy (byte-identical, broadcasts). Checked at a scalar float t != 0.
    R0, z0, phi0, t0 = 1.1, 0.15, 0.35, 0.7  # scalar Python-float t
    methods = ("_Rforce", "_zforce", "_phitorque")
    ref = numpy.array(
        [
            float(
                getattr(_SN, m)(
                    numpy.asarray(R0), numpy.asarray(z0), numpy.asarray(phi0), t0
                )
            )
            for m in methods
        ]
    )
    with use(backend_name, force=True):
        got = numpy.array(
            [
                float(
                    as_numpy(
                        getattr(_SN, m)(
                            _asarray(backend_name, R0),
                            _asarray(backend_name, z0),
                            _asarray(backend_name, phi0),
                            t0,
                        )
                    )
                )
                for m in methods
            ]
        )
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-11,
        atol=1e-13,
        err_msg=f"SoftenedNeedle rotating scalar-t ({backend_name})",
    )


@pytest.mark.parametrize("force", ["R", "z"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_grad_vs_finite_difference(backend_name, force):
    # Stringent grad-vs-FD: the force gradient w.r.t. R h-converges to a central
    # finite difference (central-FD error ~ h^2).
    fn = {"R": evaluateRforces, "z": evaluatezforces}[force]
    z0, phi0, t0 = 0.3, 0.4, 0.1
    R0 = 1.3

    def num_force(R):
        return float(
            fn(_SN, numpy.asarray(R), numpy.asarray(z0), phi=numpy.asarray(phi0), t=t0)
        )

    if backend_name == "jax":
        ad = float(
            jax.jacfwd(
                lambda R: fn(
                    _SN, R, jnp.asarray(z0), phi=jnp.asarray(phi0), t=jnp.asarray(t0)
                )
            )(jnp.asarray(R0))
        )
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        fn(
            _SN, Rt, torch.tensor(z0), phi=torch.tensor(phi0), t=torch.tensor(t0)
        ).backward()
        ad = float(Rt.grad)
    prev = None
    for h in (1e-3, 1e-4, 1e-5):
        fd = (num_force(R0 + h) - num_force(R0 - h)) / (2 * h)
        rel = abs(ad - fd) / (abs(fd) + 1e-30)
        if prev is not None:
            assert rel <= prev + 1e-12
        prev = rel
    assert rel < 1e-6, (
        f"SoftenedNeedle {force}-force grad-vs-FD rel={rel:.2e} ({backend_name})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_returns_backend_array(backend_name):
    # A backend coordinate returns a backend array (the migrated path is taken).
    from galpy.backend import is_backend_array

    out = _SN._Rforce(
        _asarray(backend_name, _R0),
        _asarray(backend_name, _Z0),
        _asarray(backend_name, _PHI0),
        _asarray(backend_name, 0.1),
    )
    assert is_backend_array(out)


# --- numpy-scalar coords under a FORCED backend (the dxdv_3d_c-vs-python gap) ---
# Non-axisymmetric phi and t != 0 so the rotating bar frame is fully exercised.
_R1, _Z1, _PHI1, _T1 = 1.3, 0.35, 0.9, 0.6


def _fresh_triax():
    # A fresh (cold-cache) triaxial instance. The numpy Hessian md5 cache is
    # per-instance, so a shared instance whose cache was warmed by a numpy call
    # would mask the pre-fix backend gap (a cache hit returns the numpy value
    # without ever running the backend code); fresh instances keep every
    # forced-backend check an honest regression.
    return SoftenedNeedleBarPotential(amp=1.2, a=1.0, b=0.3, c=0.5, pa=0.3, omegab=1.4)


# numpy reference values, on a dedicated instance never used for a forced call.
_REF_POT = _fresh_triax()


def _numpy_ref(method):
    return float(
        getattr(_REF_POT, method)(
            numpy.asarray(_R1), numpy.asarray(_Z1), numpy.asarray(_PHI1), _T1
        )
    )


@pytest.mark.parametrize("method", _METHODS + _HESS_METHODS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_numpy_scalar_coords_under_forced_backend(backend_name, method):
    # The python orbit integrator (test_dxdv_3d_c_vs_python) hands the raw force
    # and second-derivative methods NUMPY-SCALAR coords while the backend is
    # FORCED. get_namespace then resolves to the backend but the coords stay
    # numpy, and torch.cos/sqrt(numpy.float64) raises -- both in the forces and
    # in the previously-unmigrated Hessian (numpy.cos/sin/sqrt + an md5 cache).
    # The fix coerces coords at the boundary and gives the Hessian a backend
    # branch; every method must match the numpy reference.
    ref = _numpy_ref(method)
    pot = _fresh_triax()  # cold cache: the forced call must run backend code
    with use(backend_name, force=True):
        got = float(
            as_numpy(
                getattr(pot, method)(
                    numpy.float64(_R1),
                    numpy.float64(_Z1),
                    numpy.float64(_PHI1),
                    numpy.float64(_T1),
                )
            )
        )
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-11,
        atol=1e-13,
        err_msg=f"SoftenedNeedle.{method} numpy-scalar coords, forced {backend_name}",
    )


# Each second derivative equals -d(migrated force)/d(coord); cross-terms are
# checked against the force whose FD is cheapest to reason about.
_HESS_FD = [
    ("_R2deriv", "_Rforce", "R"),
    ("_z2deriv", "_zforce", "z"),
    ("_phi2deriv", "_phitorque", "phi"),
    ("_Rzderiv", "_Rforce", "z"),
    ("_Rphideriv", "_phitorque", "R"),
    ("_phizderiv", "_phitorque", "z"),
]


@pytest.mark.parametrize("deriv,force_method,wrt", _HESS_FD)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_hessian_grad_vs_finite_difference(backend_name, deriv, force_method, wrt):
    # Stringent grad-vs-FD: each migrated second derivative equals
    # -d(migrated force)/d(coord) by central finite difference, h-converging as
    # h^2. This catches sign/factor transcription errors in the backend Hessian
    # branch that the numpy A/B byte-identity check cannot (the numpy path is
    # unchanged, so a backend-only typo would not show up there).
    idx = {"R": 0, "z": 1, "phi": 2}[wrt]
    pot = _fresh_triax()

    def force_at(coords):
        with use(backend_name, force=True):
            return float(
                as_numpy(
                    getattr(pot, force_method)(
                        numpy.float64(coords[0]),
                        numpy.float64(coords[1]),
                        numpy.float64(coords[2]),
                        numpy.float64(_T1),
                    )
                )
            )

    with use(backend_name, force=True):
        analytic = float(
            as_numpy(
                getattr(pot, deriv)(
                    numpy.float64(_R1),
                    numpy.float64(_Z1),
                    numpy.float64(_PHI1),
                    numpy.float64(_T1),
                )
            )
        )
    prev = None
    for h in (1e-3, 1e-4, 1e-5):
        cp = [_R1, _Z1, _PHI1]
        cm = [_R1, _Z1, _PHI1]
        cp[idx] += h
        cm[idx] -= h
        fd = -(force_at(cp) - force_at(cm)) / (2.0 * h)
        rel = abs(analytic - fd) / (abs(fd) + 1e-30)
        if prev is not None:  # central FD error ~ h^2: shrinks as h shrinks
            assert rel <= prev + 1e-11
        prev = rel
    assert rel < 1e-6, (
        f"SoftenedNeedle {deriv} grad-vs-FD rel={rel:.2e} ({backend_name})"
    )


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("method", _HESS_METHODS)
def test_hessian_jit_matches_numpy(method):
    # The Hessian backend branch must be trace-safe: no per-instance md5 cache
    # (hashing a tracer is illegal) and no numpy.cos/sin/sqrt (which would strip a
    # tracer to numpy / raise a TracerArrayConversionError under jit). jax.jit
    # compiles each second derivative and it matches the numpy reference.
    ref = _numpy_ref(method)
    pot = _fresh_triax()
    f = jax.jit(lambda R, z, phi, t: getattr(pot, method)(R, z, phi, t))
    got = float(
        f(
            jnp.asarray(_R1),
            jnp.asarray(_Z1),
            jnp.asarray(_PHI1),
            jnp.asarray(_T1),
        )
    )
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-11,
        atol=1e-13,
        err_msg=f"SoftenedNeedle.{method} jax.jit",
    )
