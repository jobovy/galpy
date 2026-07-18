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
_METHODS = ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_dens"]
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
