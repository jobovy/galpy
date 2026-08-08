###############################################################################
# test_backend_linearpotential.py: multi-backend tests for the compute paths
# of galpy.potential.linearPotential -- the class methods (__call__/force and
# their _nodecorator helpers) and the module-level functional interface
# (evaluatelinearPotentials / evaluatelinearForces).
#
# Unlike Potential.py / planarPotential.py, linearPotential.py carries NO bare
# numpy on any AD compute path: _call_nodecorator / _force_nodecorator and the
# functional _evaluatelinearPotentials / _evaluatelinearForces are pure
# passthroughs (self._amp * self._evaluate(...) / self._force(...)) that inherit
# their array namespace from the (already migrated) per-potential
# _evaluate / _force. So there was nothing to sweep in the file itself; these
# tests pin that the whole linear dispatch chain nonetheless flows jax/torch
# through and differentiates, and that numpy stays byte-identical.
#
# Tested on single linear potentials (IsothermalDiskPotential, KGPotential), on
# a linearCompositePotential (their sum) to cover the composite dispatch, and on
# verticalPotential -- the 3D->1D wrapper -- built from an axisymmetric
# (MiyamotoNagaiPotential) and a non-axisymmetric (LogarithmicHaloPotential at a
# fixed phi) 3D potential, so the tR/tphi broadcast (xp.ones_like) and the
# delegation to the parent's evaluate/zforce flow backend arrays through.
#
# Proves the four discriminating properties for every entry:
#   (a) eager jax returns a jax array,
#   (b) jax.grad through it works and matches a finite difference,
#   (c) eager torch returns a torch tensor (and torch autograd matches FD),
#   (d) numpy returns the SAME value as the bare-numpy reference.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
#
# NOTE: linearPotential.plot / plotlinearPotentials build numpy grids
# (linspace/zeros) for plotting only -- not an AD compute path -- and are
# intentionally NOT migrated / not tested for AD (same call as plotplanarPotentials).
###############################################################################
import numpy
import pytest

from galpy.backend import use
from galpy.potential import (
    IsothermalDiskPotential,
    KGPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
)
from galpy.potential.linearPotential import (
    evaluatelinearForces,
    evaluatelinearPotentials,
)
from galpy.potential.verticalPotential import toVerticalPotential, verticalPotential

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


def _single_pots():
    return [
        IsothermalDiskPotential(amp=1.0, sigma=0.5),
        KGPotential(),
    ]


def _composite_pot():
    # linearCompositePotential via __add__: exercises the composite dispatch.
    return IsothermalDiskPotential(amp=1.0, sigma=0.5) + KGPotential()


def _vertical_pots():
    # verticalPotential wrappers over 3D parents: exercise the xp.ones_like
    # tR/tphi broadcast and the delegation to the parent evaluate/zforce.
    # Axisymmetric parent (no phi) and non-axisymmetric parent (fixed phi);
    # toVerticalPotential of a 2-component list yields a linearCompositePotential
    # of vertical potentials (composite-of-wrappers dispatch).
    return [
        verticalPotential(MiyamotoNagaiPotential(normalize=1.0), R=1.1),
        verticalPotential(
            LogarithmicHaloPotential(normalize=1.0, b=0.8, q=0.9), R=1.1, phi=0.7
        ),
        toVerticalPotential(
            MiyamotoNagaiPotential(normalize=1.0)
            + LogarithmicHaloPotential(normalize=0.5),
            1.1,
        ),
    ]


def _all_targets():
    return _single_pots() + [_composite_pot()] + _vertical_pots()


# --- compute methods + functional interface (all return arrays) ------------ #
def _methods():
    # x in the domain of both potentials; avoid x=0 (KG sqrt cusp / cosh log
    # are smooth there but pick a generic interior point).
    pts = [(0.7,), (1.3,)]
    return [
        ("call", lambda p, x: p(x, use_physical=False), pts),
        ("force", lambda p, x: p.force(x, use_physical=False), pts),
        (
            "evaluatelinearPotentials",
            lambda p, x: evaluatelinearPotentials(p, x, use_physical=False),
            pts,
        ),
        (
            "evaluatelinearForces",
            lambda p, x: evaluatelinearForces(p, x, use_physical=False),
            pts,
        ),
    ]


def _flat_with_pots(methods, pots):
    out, ids = [], []
    for name, fn, pts in methods:
        for pt in pts:
            for p in pots:
                out.append((name, fn, pt, p))
                ids.append(f"{name}-{'_'.join(str(x) for x in pt)}-{type(p).__name__}")
    return out, ids


FLAT, IDS = _flat_with_pots(_methods(), _all_targets())


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


# ============================ numpy parity ================================= #
@pytest.mark.parametrize("name,fn,pt,p", FLAT, ids=IDS)
def test_numpy_finite(name, fn, pt, p):
    out = numpy.asarray(fn(p, *pt))
    assert numpy.all(numpy.isfinite(out)), (name, type(p).__name__, out)


# ============================ jax ========================================== #
@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", FLAT, ids=IDS)
def test_jax_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[jnp.asarray(x) for x in pt])
    assert "jax" in type(out).__module__, (name, type(out))
    numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt,p", FLAT, ids=IDS)
def test_jax_grad_matches_fd(name, fn, pt, p):
    _jax_grad_fd(fn, p, pt)


# ============================ torch ======================================== #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", FLAT, ids=IDS)
def test_torch_eager_and_value(name, fn, pt, p):
    ref = numpy.asarray(fn(p, *pt))
    out = fn(p, *[torch.as_tensor(x, dtype=torch.float64) for x in pt])
    assert torch.is_tensor(out), (name, type(out))
    numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt,p", FLAT, ids=IDS)
def test_torch_autograd_matches_fd(name, fn, pt, p):
    _torch_grad_fd(fn, p, pt)


# ==================== amp as a backend array =============================== #
# IsothermalDiskPotential folds amp into the scale height H via a backend-aware
# sqrt (the is_backend_array(self._amp) branch in __init__). A jax/torch amp
# must therefore differentiate through H -- exercise d(force)/d(amp) and pin it
# to a central finite difference. The numpy FD reference builds with a plain
# python-float amp, so it also guards that path staying byte-identical.
def _isodisk_force_of_amp(amp, x):
    return IsothermalDiskPotential(amp=amp, sigma=0.5).force(x, use_physical=False)


def _amp_fd(a, x0, h=1e-6):
    fp = float(_isodisk_force_of_amp(a + h, x0))
    fm = float(_isodisk_force_of_amp(a - h, x0))
    return (fp - fm) / (2.0 * h)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_jax_amp_grad_matches_fd():
    a0, x0 = 1.3, 0.7
    grad = float(
        jax.grad(lambda a: _isodisk_force_of_amp(a, jnp.asarray(x0)))(jnp.asarray(a0))
    )
    fd = _amp_fd(a0, x0)
    assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-6), (grad, fd)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_torch_amp_grad_matches_fd():
    a0, x0 = 1.3, 0.7
    at = torch.as_tensor(a0, dtype=torch.float64).requires_grad_(True)
    _isodisk_force_of_amp(at, torch.as_tensor(x0, dtype=torch.float64)).backward()
    grad = float(at.grad)
    fd = _amp_fd(a0, x0)
    assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-6), (grad, fd)


# ============ raw compute methods under a FORCED backend ================== #
# Regression (scalar-under-forced-backend): the python orbit integrator calls
# IsothermalDiskPotential._evaluate/_force/_force2deriv DIRECTLY with a
# numpy-scalar x (the integrator RHS bypasses the @physical_input coercion gate,
# unlike the p(x)/p.force(x)/evaluatelinear* public entries which coerce). Under
# a forced backend get_namespace(x) then resolves to jax/torch while x stays a
# numpy scalar, so xp.tanh/xp.cosh(x) raised (torch rejects numpy.float64).
# Assert no raise, a backend array, and numpy value parity.
@pytest.mark.parametrize("backend_name", ["jax", "torch"])
def test_isodisk_raw_methods_numpy_scalar_forced_backend(backend_name):
    if backend_name == "jax" and not _HAS_JAX:  # pragma: no cover
        pytest.skip("jax not installed")
    if backend_name == "torch" and not _HAS_TORCH:  # pragma: no cover
        pytest.skip("torch not installed")
    p = IsothermalDiskPotential(amp=1.0, sigma=1.0)
    methods = ("_evaluate", "_force", "_force2deriv")
    for x0 in (0.5, -0.3, 1.7):
        x = numpy.float64(x0)
        ref = {m: float(getattr(p, m)(x)) for m in methods}
        with use(backend_name, force=True):
            for m in methods:
                got = getattr(p, m)(x)
                assert backend_name in type(got).__module__, (m, type(got))
                numpy.testing.assert_allclose(
                    float(got), ref[m], rtol=1e-12, atol=1e-14, err_msg=f"{m} x={x0}"
                )


###############################################################################
# Boundary-crossing guard for the vertical->3D adapter forwards.
#
# verticalPotential wraps a 3D potential at fixed (R,phi). Those forwards used
# to go through the DECORATED public API (self._Pot(...), .zforce), so under a
# forced backend every call re-entered @backend_input and paid a coercion --
# actionAngleVertical.Jz crossed 1708 times per call because of it.
#
# The values stay correct when that regresses; only time changes. So this
# watches the coercion, not the result.
###############################################################################


def _count_vertical_coercions(fn, backend):
    """Run fn() under a forced backend; count non-numpy coerce_coords calls."""
    import galpy.backend._input as _bi

    real = _bi.coerce_coords
    n = [0]

    def spy(xp, *coords):
        if xp is not numpy:
            n[0] += 1
        return real(xp, *coords)

    _bi.coerce_coords = spy
    try:
        with use(backend, force=True):
            fn()
    finally:
        _bi.coerce_coords = real
    return n[0]


def _assert_vertical_adapters_do_not_recross(backend, mk):
    from galpy.potential import MiyamotoNagaiPotential

    vp = toVerticalPotential(MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0), 1.0)
    z = mk([0.1])
    for name, fn in [
        ("verticalPotential._evaluate", lambda: vp._evaluate(z, t=0.0)),
        ("verticalPotential._force", lambda: vp._force(z, t=0.0)),
    ]:
        assert _count_vertical_coercions(fn, backend) == 0, (
            f"{name} re-crossed the @backend_input boundary; it should forward "
            "to the undecorated inner evaluator"
        )
    # Negative control: _force2deriv is deliberately NOT migrated (there is no
    # _evaluatez2derivs), so it must still cross. Without this, a spy that
    # stopped working would make the assertions above pass vacuously.
    assert _count_vertical_coercions(lambda: vp._force2deriv(z, t=0.0), backend) > 0, (
        "control failed: _force2deriv should still cross, so a zero here means "
        "the counter is broken, not that the code improved"
    )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_vertical_adapter_forwards_do_not_recross_boundary_jax():
    _assert_vertical_adapters_do_not_recross("jax", jnp.asarray)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_vertical_adapter_forwards_do_not_recross_boundary_torch():
    _assert_vertical_adapters_do_not_recross(
        "torch", lambda v: torch.tensor(v, dtype=torch.float64)
    )
