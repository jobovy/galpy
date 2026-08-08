###############################################################################
# test_backend_planarpotential.py: multi-backend tests for the convenience
# methods of galpy.potential.planarPotential (and the module-level functional
# interface) that were swept from bare numpy onto the galpy.backend namespace.
#
# Swept (class) methods, each with its own bare-numpy outer sqrt:
#   planarPotential.epifreq, .vcirc, .omegac and planarAxiPotential.vesc.
# These live on the *axisymmetric* planar potential class, so they are tested
# on single axisymmetric planar potentials (planarPotentialFromRZPotential).
#
# Module-level functional interface that delegates to the (already migrated)
# per-potential _evaluate/_Rforce/_R2deriv and must flow jax/torch through and
# differentiate:
#   evaluateplanarPotentials / evaluateplanarRforces / evaluateplanarR2derivs
# tested on single potentials AND on toPlanarPotential(MWPotential2014) (a
# planarCompositePotential).
#
# evaluateplanarphitorques is also exercised but, for an *axisymmetric*
# potential, planarAxiPotential._phitorque returns the literal constant 0.0
# (independent of R). That constant is intentionally NOT anchored on the
# namespace (anchoring would change the byte-identical numpy return from a
# Python float 0.0 to a 0-d array), so for axi potentials it is checked for
# numpy parity and that jax.grad yields the correct zero, not for array-ness.
#
# For the array-returning entries this proves the four discriminating
# properties:
#   (a) eager jax returns a jax array,
#   (b) jax.grad through it works and matches a finite difference,
#   (c) eager torch returns a torch tensor,
#   (d) numpy returns the SAME value as the bare-numpy reference.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
#
# NOTE: planarPotential.LinShuReductionFactor wraps scipy.integrate.quad
# (scipy quadrature) and is intentionally NOT migrated / not tested for AD.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import count_boundary_crossings

from galpy.backend import use
from galpy.potential import (
    HenonHeilesPotential,
    HernquistPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
    MWPotential2014,
    evaluateplanarphitorques,
    evaluateplanarPotentials,
    evaluateplanarR2derivs,
    evaluateplanarRforces,
    toPlanarPotential,
)

# This module manages backends explicitly; exempt from the global --backend
# force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None
    _HAS_JAX = False
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _single_axi_pots():
    # Single axisymmetric planar potentials (planarPotentialFromRZPotential):
    # carry the full set of swept methods, including vesc.
    return [
        toPlanarPotential(LogarithmicHaloPotential(normalize=1.0)),
        toPlanarPotential(MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0)),
        toPlanarPotential(HernquistPotential(normalize=1.0)),
    ]


def _functional_targets():
    # The functional interface accepts single potentials and the composite.
    return _single_axi_pots() + [toPlanarPotential(MWPotential2014)]


# --- swept class methods (axisymmetric only) ------------------------------- #
def _class_methods():
    return [
        ("epifreq", lambda p, R: p.epifreq(R, use_physical=False), [(1.0,), (1.3,)]),
        ("vcirc", lambda p, R: p.vcirc(R, use_physical=False), [(1.0,), (1.3,)]),
        ("omegac", lambda p, R: p.omegac(R, use_physical=False), [(1.0,), (1.3,)]),
        ("vesc", lambda p, R: p.vesc(R, use_physical=False), [(1.0,), (1.3,)]),
    ]


# --- R-dependent functional interface (returns arrays) --------------------- #
def _functional_methods():
    return [
        (
            "evaluateplanarPotentials",
            lambda p, R: evaluateplanarPotentials(p, R, use_physical=False),
            [(1.0,), (1.3,)],
        ),
        (
            "evaluateplanarRforces",
            lambda p, R: evaluateplanarRforces(p, R, use_physical=False),
            [(1.0,), (1.3,)],
        ),
        (
            "evaluateplanarR2derivs",
            lambda p, R: evaluateplanarR2derivs(p, R, use_physical=False),
            [(1.0,), (1.3,)],
        ),
    ]


def _ids(methods):
    out = []
    for name, _fn, pts in methods:
        for pt in pts:
            out.append(f"{name}-{'_'.join(str(x) for x in pt)}")
    return out


def _flat(methods):
    out = []
    for name, fn, pts in methods:
        for pt in pts:
            out.append((name, fn, pt))
    return out


# class-method entries: (name, fn, pt, pot)
def _flat_with_pots(methods, pots):
    out, ids = [], []
    for name, fn, pts in methods:
        for pt in pts:
            for p in pots:
                out.append((name, fn, pt, p))
                ids.append(f"{name}-{'_'.join(str(x) for x in pt)}-{type(p).__name__}")
    return out, ids


CLASS_FLAT, CLASS_IDS = _flat_with_pots(_class_methods(), _single_axi_pots())
FUNC_FLAT, FUNC_IDS = _flat_with_pots(_functional_methods(), _functional_targets())


def _jax_grad_fd(fn, p, pt):
    """jax.grad of sum(fn) vs central finite difference, per coordinate."""
    for i in range(len(pt)):

        def g(xi, i=i):
            args = [jnp.asarray(pt[j]) if j != i else xi for j in range(len(pt))]
            return jnp.sum(fn(p, *args))

        x0 = jnp.asarray(pt[i])
        grad = float(jax.grad(g)(x0))
        eps = 1e-6 * max(1.0, abs(float(x0)))
        fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
        assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (i, grad, fd)


def _torch_grad_fd(fn, p, pt):
    for i in range(len(pt)):
        targs = [torch.as_tensor(pt[j], dtype=torch.float64) for j in range(len(pt))]
        targs[i].requires_grad_(True)
        out = fn(p, *targs)
        out = out.sum() if out.ndim > 0 else out
        out.backward()
        grad = float(targs[i].grad)
        eps = 1e-6 * max(1.0, abs(pt[i]))
        xs = list(pt)
        xs[i] = pt[i] + eps
        fp = numpy.asarray(fn(p, *xs)).sum()
        xs[i] = pt[i] - eps
        fm = numpy.asarray(fn(p, *xs)).sum()
        fd = float((fp - fm) / (2.0 * eps))
        assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (i, grad, fd)


# ===================== class methods (axisymmetric) ======================== #
@pytest.mark.parametrize("name,fn,pt,p", CLASS_FLAT, ids=CLASS_IDS)
def test_class_numpy_finite(name, fn, pt, p):
    out = numpy.asarray(fn(p, *pt))
    assert numpy.all(numpy.isfinite(out)), (name, type(p).__name__, out)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", CLASS_FLAT, ids=CLASS_IDS)
def test_class_jax_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[jnp.asarray(x) for x in pt])
    assert "jax" in type(out).__module__, (name, type(out))
    numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", CLASS_FLAT, ids=CLASS_IDS)
def test_class_jax_grad_matches_fd(name, fn, pt, p):
    _jax_grad_fd(fn, p, pt)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", CLASS_FLAT, ids=CLASS_IDS)
def test_class_torch_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[torch.as_tensor(x, dtype=torch.float64) for x in pt])
    assert torch.is_tensor(out), (name, type(out))
    numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", CLASS_FLAT, ids=CLASS_IDS)
def test_class_torch_autograd_matches_fd(name, fn, pt, p):
    _torch_grad_fd(fn, p, pt)


# ============== R-dependent functional interface (array out) =============== #
@pytest.mark.parametrize("name,fn,pt,p", FUNC_FLAT, ids=FUNC_IDS)
def test_func_numpy_finite(name, fn, pt, p):
    out = numpy.asarray(fn(p, *pt))
    assert numpy.all(numpy.isfinite(out)), (name, type(p).__name__, out)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", FUNC_FLAT, ids=FUNC_IDS)
def test_func_jax_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[jnp.asarray(x) for x in pt])
    assert "jax" in type(out).__module__, (name, type(out))
    numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", FUNC_FLAT, ids=FUNC_IDS)
def test_func_jax_grad_matches_fd(name, fn, pt, p):
    _jax_grad_fd(fn, p, pt)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", FUNC_FLAT, ids=FUNC_IDS)
def test_func_torch_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[torch.as_tensor(x, dtype=torch.float64) for x in pt])
    assert torch.is_tensor(out), (name, type(out))
    numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", FUNC_FLAT, ids=FUNC_IDS)
def test_func_torch_autograd_matches_fd(name, fn, pt, p):
    _torch_grad_fd(fn, p, pt)


# ============ evaluateplanarphitorques on axisymmetric pots ================ #
# Axisymmetric: _phitorque == constant 0.0 (deliberately not anchored on the
# namespace to keep the numpy return byte-identical). Verify numpy parity and
# that jax.grad yields the correct zero (no NaN poisoning).
@pytest.mark.parametrize("p", _functional_targets(), ids=lambda p: type(p).__name__)
def test_phitorque_axi_numpy_zero(p):
    val = evaluateplanarphitorques(p, 1.2, use_physical=False)
    numpy.testing.assert_allclose(numpy.asarray(val), 0.0, atol=0.0)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("p", _functional_targets(), ids=lambda p: type(p).__name__)
def test_phitorque_axi_jax_grad_zero(p):
    val = evaluateplanarphitorques(p, jnp.asarray(1.2), use_physical=False)
    numpy.testing.assert_allclose(numpy.asarray(val), 0.0, atol=0.0)
    grad = float(
        jax.grad(lambda R: jnp.sum(evaluateplanarphitorques(p, R, use_physical=False)))(
            jnp.asarray(1.2)
        )
    )
    assert grad == 0.0, grad


# ============ HenonHeiles raw methods under a FORCED backend =============== #
# Regression (scalar-under-forced-backend): the python orbit integrator calls
# HenonHeilesPotential._evaluate/_Rforce/_phitorque (and the 2nd derivatives)
# DIRECTLY with numpy-scalar (R, phi) -- the integrator RHS bypasses the
# @physical_input coercion gate. Under a forced backend get_namespace(R, phi)
# resolves to jax/torch while phi stays a numpy scalar, so xp.sin/cos(3*phi)
# raised (torch rejects numpy.float64). Assert no raise, a backend array, and
# numpy value parity. (HenonHeiles is the potential behind test_lyapunov_*.)
@pytest.mark.parametrize("backend_name", ["jax", "torch"])
def test_henonheiles_raw_methods_numpy_scalar_forced_backend(backend_name):
    if backend_name == "jax" and not _HAS_JAX:  # pragma: no cover
        pytest.skip("jax not installed")
    if backend_name == "torch" and not _HAS_TORCH:  # pragma: no cover
        pytest.skip("torch not installed")
    p = HenonHeilesPotential(amp=1.3)
    methods = (
        "_evaluate",
        "_Rforce",
        "_phitorque",
        "_R2deriv",
        "_phi2deriv",
        "_Rphideriv",
    )
    for R0, phi0 in [(1.2, 0.7), (0.5, -2.1), (2.3, 3.14159)]:
        R = numpy.float64(R0)
        phi = numpy.float64(phi0)
        ref = {m: float(getattr(p, m)(R, phi)) for m in methods}
        with use(backend_name, force=True):
            for m in methods:
                got = getattr(p, m)(R, phi)
                assert backend_name in type(got).__module__, (m, type(got))
                numpy.testing.assert_allclose(
                    float(got),
                    ref[m],
                    rtol=1e-12,
                    atol=1e-14,
                    err_msg=f"{m} R={R0} phi={phi0}",
                )


###############################################################################
# Boundary-crossing guard for the planar->3D adapter forwards.
#
# The adapters used to forward through the DECORATED public API
# (self._Pot(...), .Rforce, .phitorque). Under a forced backend each of those
# re-enters @backend_input and pays a coercion, so a caller that evaluates a
# planar potential in a loop pays it per iteration: actionAngleSpherical's
# rperi/rap bisection crossed 981 times per actions() call because of this.
#
# Values stay correct when that regresses -- the only symptom is time -- so no
# ordinary assertion catches it. This watches the coercion itself.
###############################################################################


def _adapter_targets(mk):
    """(name, thunk) for the five migrated forwards, plus the R2deriv control."""
    from galpy.potential import DehnenBarPotential

    axi = toPlanarPotential(MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0))
    bar = toPlanarPotential(DehnenBarPotential(omegab=1.8, rb=0.6, Af=0.03))
    R, phi = mk([1.0]), mk([0.3])
    migrated = [
        ("axi._evaluate", lambda: axi._evaluate(R, t=0.0)),
        ("axi._Rforce", lambda: axi._Rforce(R, t=0.0)),
        ("bar._evaluate", lambda: bar._evaluate(R, phi=phi, t=0.0)),
        ("bar._Rforce", lambda: bar._Rforce(R, phi=phi, t=0.0)),
        ("bar._phitorque", lambda: bar._phitorque(R, phi=phi, t=0.0)),
    ]
    return migrated, ("axi._R2deriv", lambda: axi._R2deriv(R, t=0.0))


def _assert_adapters_do_not_recross(backend, mk):
    migrated, control = _adapter_targets(mk)
    for name, fn in migrated:
        assert count_boundary_crossings(fn, backend) == 0, (
            f"{name} re-crossed the @backend_input boundary; it should forward "
            "to the undecorated inner evaluator"
        )
    # Negative control: _R2deriv is deliberately NOT migrated (no
    # _evaluateR2derivs exists), so it still crosses. Without this, a spy that
    # silently stopped working would make every assertion above pass vacuously.
    cname, cfn = control
    assert count_boundary_crossings(cfn, backend) > 0, (
        f"control failed: {cname} should still cross, so a zero here means the "
        "counter is broken, not that the code improved"
    )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_planar_adapter_forwards_do_not_recross_boundary_jax():
    _assert_adapters_do_not_recross("jax", jnp.asarray)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_planar_adapter_forwards_do_not_recross_boundary_torch():
    _assert_adapters_do_not_recross(
        "torch", lambda v: torch.tensor(v, dtype=torch.float64)
    )
