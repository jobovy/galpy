###############################################################################
# test_backend_cosmphidisk.py: forced-backend scalar-coercion fixes for
# CosmphiDiskPotential (and the Potential.vterm rotation-curve method).
#
# Under a forced backend the pure-Python integrator / a scalar API call hands the
# force numpy/scalar coordinates while get_namespace resolves to the backend, so
# xp.ones_like(scalar) / xp.where(numpy) (CosmphiDisk) and xp.sin(python_float)
# (vterm) raised. coerce_coords (CosmphiDisk) and the data-guard (vterm) fix it;
# the numpy path stays byte-identical.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, use
from galpy.potential import CosmphiDiskPotential, LogarithmicHaloPotential

pytestmark = pytest.mark.backend_managed

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

_CP = CosmphiDiskPotential(amp=1.0, phib=0.4, m=2, rb=0.9)
_METHODS = [
    "_evaluate",
    "_Rforce",
    "_phitorque",
    "_R2deriv",
    "_phi2deriv",
    "_Rphideriv",
]
_R0, _PHI0 = 1.1, 0.35


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_cosmphi_value_parity(backend_name, method):
    ref = float(getattr(_CP, method)(numpy.asarray(_R0), phi=numpy.asarray(_PHI0)))
    got = float(
        as_numpy(
            getattr(_CP, method)(
                _asarray(backend_name, _R0), phi=_asarray(backend_name, _PHI0)
            )
        )
    )
    rtol, atol = (0.0, 0.0) if backend_name == "numpy" else (1e-12, 1e-14)
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=atol, err_msg=f"CosmphiDisk.{method} ({backend_name})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cosmphi_numpy_scalar_under_forced_backend(backend_name):
    # The integrator scenario: numpy-scalar R/phi under a FORCED backend, where
    # xp.ones_like(scalar) / xp.where(numpy) raised before coerce_coords. Both an
    # inside-rb (R < rb) and an outside point, since the where-branches differ.
    methods = ("_evaluate", "_Rforce", "_phitorque")
    for R in (0.6, 1.4):  # inside and outside rb=0.9
        ref = numpy.array([float(getattr(_CP, m)(R, phi=_PHI0)) for m in methods])
        with use(backend_name, force=True):
            got = numpy.array(
                [float(as_numpy(getattr(_CP, m)(R, phi=_PHI0))) for m in methods]
            )
        numpy.testing.assert_allclose(
            got,
            ref,
            rtol=1e-11,
            atol=1e-13,
            err_msg=f"CosmphiDisk forced R={R} ({backend_name})",
        )


@pytest.mark.parametrize("force", ["R", "phi"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cosmphi_grad_vs_finite_difference(backend_name, force):
    # CosmphiDisk is a planar potential -> call its _Rforce/_phitorque(R, phi) directly.
    fn = {"R": _CP._Rforce, "phi": _CP._phitorque}[force]
    R0 = 1.3

    def num(R):
        return float(fn(numpy.asarray(R), phi=numpy.asarray(_PHI0)))

    if backend_name == "jax":
        ad = float(jax.jacfwd(lambda R: fn(R, phi=jnp.asarray(_PHI0)))(jnp.asarray(R0)))
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        fn(Rt, phi=torch.tensor(_PHI0)).backward()
        ad = float(Rt.grad)
    # best over a few h (the gradient is machine-exact, so at tiny h the FD is
    # roundoff-limited -- take the closest match rather than requiring monotone rel).
    best = min(
        abs(ad - (num(R0 + h) - num(R0 - h)) / (2 * h)) / (abs(num(R0 + h)) + 1e-30)
        for h in (1e-3, 1e-4, 1e-5)
    )
    assert best < 1e-6, (
        f"CosmphiDisk {force} grad-vs-FD best={best:.2e} ({backend_name})"
    )


# --- Potential.vterm under a forced backend on a numpy/scalar longitude ---------
_LP = LogarithmicHaloPotential(normalize=1.0, q=0.9)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("deg", [True, False])
def test_vterm_numpy_scalar_under_forced_backend(backend_name, deg):
    # vterm's `xp = get_namespace(l)` resolved to the backend for a scalar Python /
    # numpy longitude, so xp.sin(python_float) raised. The data-guard keeps a
    # non-backend l on numpy. Checked against numpy.
    ell = 45.0 if deg else 0.8
    ref = float(_LP.vterm(ell, deg=deg, use_physical=False))
    with use(backend_name, force=True):
        got = float(as_numpy(_LP.vterm(ell, deg=deg, use_physical=False)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
