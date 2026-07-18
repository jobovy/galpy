###############################################################################
# test_backend_ferrers.py: backend tests for FerrersPotential.
#
# Ferrers' potential / force / 2nd-derivative methods evaluate an ellipsoidal
# integral from a confocal lower limit to infinity. numpy inputs keep the
# scipy.integrate.quad adaptive path (byte-identical to before); jax/torch
# inputs route to a fixed-order Gauss-Legendre semi-infinite quadrature
# (galpy.backend.quadrature) with the lower limit found by the backend brentq
# (galpy.backend.optimize), so the whole force/2nd-deriv chain is now
# jit/grad-safe under a trace (not just eager). This module checks:
#   1. numpy / jax / torch value parity for every migrated compute method. The
#      backend GL path is MORE accurate than scipy's adaptive quadrature, so the
#      2nd-derivative methods differ from the numpy reference by scipy's own
#      adaptive-tolerance floor (~1e-6), not to machine precision.
#   2. the migrated force integral is TRACEABLE: jax.jacfwd / jax.jit and torch
#      autograd over evaluateRforces/evaluatezforces return finite values, and
#      eager jax/torch return backend arrays.
#   3. the force gradient w.r.t. R h-converges to a central finite difference
#      (jax and torch), the stringent grad-vs-FD check.
#   4. the fully-arithmetic _dens: value parity AND a finite (0) gradient on the
#      m2 >= 1 (outside-ellipsoid) dead branch instead of NaN.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, use
from galpy.potential import (
    FerrersPotential,
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

# Triaxial, rotated (pa) and rotating (omegab) so the cos/sin de-rotation and the
# phi-dependent Hessian branches are all exercised.
_FE = FerrersPotential(amp=1.3, a=1.5, n=2, b=0.9, c=0.7, pa=0.3, omegab=1.0)

# Every migrated compute method (all scalar in their quadrature, so probed at a
# scalar point). The 2nd-derivative methods only match the scipy reference to
# its adaptive-tolerance floor (the GL path is the more accurate one).
_FIRST_ORDER = ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_dens"]
_SECOND_ORDER = [
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
]
_METHODS = _FIRST_ORDER + _SECOND_ORDER
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
    # backend. numpy vs numpy is exact (the numpy path is byte-identical); the
    # backend GL quadrature matches the scipy reference to ~1e-9 for the
    # forces/potential and to scipy's adaptive floor (~1e-6) for the 2nd
    # derivatives (GL being the more accurate of the two).
    ref = float(
        getattr(_FE, method)(
            numpy.asarray(_R0), numpy.asarray(_Z0), numpy.asarray(_PHI0), _T0
        )
    )
    got = float(
        as_numpy(
            getattr(_FE, method)(
                _asarray(backend_name, _R0),
                _asarray(backend_name, _Z0),
                _asarray(backend_name, _PHI0),
                _asarray(backend_name, _T0),
            )
        )
    )
    if backend_name == "numpy":
        rtol, atol = 0.0, 0.0  # numpy path is byte-identical
    elif method in _SECOND_ORDER:
        rtol, atol = 1e-6, 1e-8
    else:
        rtol, atol = 1e-9, 1e-11
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=atol, err_msg=f"Ferrers.{method} ({backend_name})"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_traced_force_finite(backend_name):
    # The migrated force integral is now traceable: jax.jit / jax.jacfwd and
    # torch autograd over the public force evaluators return finite values (the
    # exact failure -- numpy.roots on a traced lower limit -- that defined this
    # gap). Probed at both an inside- and an outside-ellipsoid point.
    for R0 in (0.6, 2.5):
        z0, phi0, t0 = 0.3, 0.4, 0.1
        if backend_name == "jax":
            args = (jnp.asarray(z0), jnp.asarray(phi0), jnp.asarray(t0))
            val = jax.jit(
                lambda R: evaluateRforces(_FE, R, args[0], phi=args[1], t=args[2])
            )(jnp.asarray(R0))
            jac = jax.jacfwd(
                lambda R: evaluateRforces(_FE, R, args[0], phi=args[1], t=args[2])
            )(jnp.asarray(R0))
            assert numpy.isfinite(float(val))
            assert numpy.isfinite(float(jac))
        else:
            Rt = torch.tensor(R0, requires_grad=True)
            val = evaluateRforces(
                _FE, Rt, torch.tensor(z0), phi=torch.tensor(phi0), t=torch.tensor(t0)
            )
            val.backward()
            assert numpy.isfinite(float(val.detach()))
            assert numpy.isfinite(float(Rt.grad))


@pytest.mark.parametrize("force", ["R", "z"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_grad_vs_finite_difference(backend_name, force):
    # Stringent grad-vs-FD: the force gradient w.r.t. R h-converges to a central
    # finite difference (a central-FD error ~ h^2), at both an inside and an
    # outside point. This exercises the differentiable confocal lower limit
    # (brentq implicit-function gradient) and the GL quadrature together.
    fn = {"R": evaluateRforces, "z": evaluatezforces}[force]
    z0, phi0, t0 = 0.3, 0.4, 0.1

    def num_force(R):
        return float(
            fn(_FE, numpy.asarray(R), numpy.asarray(z0), phi=numpy.asarray(phi0), t=t0)
        )

    for R0 in (0.6, 2.5):
        if backend_name == "jax":
            ad = float(
                jax.jacfwd(
                    lambda R: fn(
                        _FE,
                        R,
                        jnp.asarray(z0),
                        phi=jnp.asarray(phi0),
                        t=jnp.asarray(t0),
                    )
                )(jnp.asarray(R0))
            )
        else:
            Rt = torch.tensor(R0, requires_grad=True)
            fn(
                _FE, Rt, torch.tensor(z0), phi=torch.tensor(phi0), t=torch.tensor(t0)
            ).backward()
            ad = float(Rt.grad)
        # central FD, shrink h and require the relative error to shrink with it.
        prev = None
        for h in (1e-3, 1e-4, 1e-5):
            fd = (num_force(R0 + h) - num_force(R0 - h)) / (2 * h)
            rel = abs(ad - fd) / (abs(fd) + 1e-30)
            if prev is not None:
                # central FD is O(h^2): each 10x smaller h shrinks the error ~100x
                assert rel < prev, f"{force}@{R0} not h-converging: {prev} -> {rel}"
            prev = rel
        assert rel < 1e-8, f"{force}@{R0} grad-vs-FD not converged: {rel}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_returns_backend_array(backend_name):
    # Eager (concrete backend arrays) still returns a backend array (not a numpy
    # scalar) from the migrated force path.
    val = evaluateRforces(
        _FE,
        _asarray(backend_name, _R0),
        _asarray(backend_name, _Z0),
        phi=_asarray(backend_name, _PHI0),
        t=_asarray(backend_name, _T0),
    )
    if backend_name == "jax":
        assert isinstance(val, jax.Array)
    else:
        assert torch.is_tensor(val)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_dens_inside_outside_value_parity(backend_name):
    # _dens is branch-free under jax/torch (eager xp.where on m2 < 1). Check both
    # an inside-ellipsoid (m2 < 1) and an outside (m2 >= 1, value 0) point.
    for R0, z0, expect_zero in [(0.3, 0.05, False), (3.0, 0.5, True)]:
        ref = float(_FE._dens(numpy.asarray(R0), numpy.asarray(z0), _PHI0, _T0))
        got = float(
            as_numpy(
                _FE._dens(
                    _asarray(backend_name, R0), _asarray(backend_name, z0), _PHI0, _T0
                )
            )
        )
        if expect_zero:
            assert ref == 0.0
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dens_grad_inside_vs_finite_difference(backend_name):
    # _dens is a fully-arithmetic (autodiff-friendly) method. Inside the
    # ellipsoid AD(d_dens/dR) matches central FD.
    R0, z0 = 0.3, 0.05
    eps = 1e-6
    fd = (
        float(_FE._dens(numpy.asarray(R0 + eps), numpy.asarray(z0), _PHI0, _T0))
        - float(_FE._dens(numpy.asarray(R0 - eps), numpy.asarray(z0), _PHI0, _T0))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: _FE._dens(R, jnp.asarray(z0), jnp.asarray(_PHI0), _T0))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        _FE._dens(
            R, torch.tensor(z0, dtype=torch.float64), torch.tensor(_PHI0), _T0
        ).backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dens_grad_outside_is_finite(backend_name):
    # On the m2 >= 1 (outside) dead branch the guarded base (1 - m2/a**2 -> 1)
    # must keep the reverse-mode gradient finite (0), not NaN, for the
    # non-integer-safe power. The where selects the 0.0 value.
    R0, z0 = 3.0, 0.5  # outside the ellipsoid
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: _FE._dens(R, jnp.asarray(z0), jnp.asarray(_PHI0), _T0))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = _FE._dens(
            R, torch.tensor(z0, dtype=torch.float64), torch.tensor(_PHI0), _T0
        )
        y.backward()
        ad = 0.0 if R.grad is None else float(R.grad)
    assert numpy.isfinite(ad), f"Ferrers _dens outside grad not finite ({backend_name})"
    assert ad == 0.0


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotating_scalar_t_under_forced_backend(backend_name):
    # The orbit integrator hands the force a scalar Python-float t while the
    # coordinates are backend arrays. Under a FORCED backend get_namespace(t) would
    # resolve to that backend and torch.cos(python_float) raises (the rotating-bar
    # t-anchoring gap); the concrete-t rotation coefficient must fall back to numpy
    # (byte-identical, broadcasts into the backend force). Checked at a scalar float
    # t != 0 so the omegab*t de-rotation is actually exercised, against numpy.
    R0, z0, phi0, t0 = 1.1, 0.15, 0.35, 0.7  # scalar Python-float t
    methods = ("_Rforce", "_zforce", "_phitorque")
    ref = numpy.array(
        [
            float(
                getattr(_FE, m)(
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
                        getattr(_FE, m)(
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
        rtol=1e-9,
        atol=1e-11,
        err_msg=f"Ferrers rotating scalar-t ({backend_name})",
    )
