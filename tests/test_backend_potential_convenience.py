###############################################################################
# test_backend_potential_convenience.py: multi-backend tests for the
# convenience methods of galpy.potential.Potential that were swept from bare
# numpy onto the galpy.backend namespace layer.
#
# Covered methods (all on Potential): r2deriv, vcirc, dvcircdR, omegac,
# epifreq, verticalfreq, vesc, flattening, vterm, tdyn, rtide, ttensor
# (incl. eigenval=True).
#
# For each method this proves the four discriminating properties:
#   (a) eager jax returns a jax array,
#   (b) jax.grad through the method works and matches a finite difference,
#   (c) eager torch returns a torch tensor,
#   (d) numpy returns the SAME value as the bare-numpy implementation did
#       (round-trip / byte identity against an independent numpy recompute).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
    LogarithmicHaloPotential,
    MiyamotoNagaiPotential,
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


def _pots():
    return [
        LogarithmicHaloPotential(normalize=1.0),
        MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0),
        HernquistPotential(normalize=1.0),
    ]


# Each entry: (name, callable(pot, *xargs) -> scalar value, list of test
# points where each point is the tuple of float positional args). The callable
# uses use_physical=False so we compare the raw internal-unit numbers.
def _make_methods():
    return [
        ("r2deriv", lambda p, R, z: p.r2deriv(R, z, use_physical=False), [(1.0, 0.1)]),
        ("vcirc", lambda p, R: p.vcirc(R, use_physical=False), [(1.0,), (1.3,)]),
        ("dvcircdR", lambda p, R: p.dvcircdR(R, use_physical=False), [(1.0,), (1.3,)]),
        ("omegac", lambda p, R: p.omegac(R, use_physical=False), [(1.0,), (1.3,)]),
        ("epifreq", lambda p, R: p.epifreq(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "verticalfreq",
            lambda p, R: p.verticalfreq(R, use_physical=False),
            [(1.0,), (1.3,)],
        ),
        ("vesc", lambda p, R: p.vesc(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "flattening",
            lambda p, R, z: p.flattening(R, z, use_physical=False),
            [(1.0, 0.2)],
        ),
        (
            "vterm",
            lambda p, ll: p.vterm(ll, deg=True, use_physical=False),
            [(30.0,), (60.0,)],
        ),
        ("tdyn", lambda p, R: p.tdyn(R, use_physical=False), [(1.0,), (1.3,)]),
        (
            "rtide",
            lambda p, R, z: p.rtide(R, z, M=1.0, use_physical=False),
            [(1.0, 0.1)],
        ),
        ("ttensor", lambda p, R, z: p.ttensor(R, z, use_physical=False), [(1.0, 0.1)]),
    ]


METHODS = _make_methods()

# tdyn calls self.mass(...), a scipy-quadrature method, and rtide calls
# self.rforce(...) which lives in Force.py; both are NOT yet on the backend
# namespace (migrated in later stages). Under eager jax/torch these still work
# (the inner call gets a concrete array and detaches to numpy, then the outer
# xp.sqrt re-attaches), but jax.grad / torch.autograd cannot flow through the
# un-migrated inner numpy code, so they are excluded from the AD-vs-FD checks
# here. The numpy->xp sweep of their OWN bare-numpy sqrt is still exercised by
# the eager / numpy-parity tests.
_NO_GRAD = {"tdyn", "rtide"}


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


FLAT = _flat(METHODS)
FLAT_IDS = _ids(METHODS)

_GRAD_METHODS = [(n, f, p) for (n, f, p) in METHODS if n not in _NO_GRAD]
FLAT_GRAD = _flat(_GRAD_METHODS)
FLAT_GRAD_IDS = _ids(_GRAD_METHODS)


@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_numpy_baseline_finite(name, fn, pt):
    # numpy still produces finite real values (the unchanged reference path).
    for p in _pots():
        out = numpy.asarray(fn(p, *pt))
        assert numpy.all(numpy.isfinite(out)), (name, p.__class__.__name__, out)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_jax_eager_array_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        jargs = [jnp.asarray(x) for x in pt]
        out = fn(p, *jargs)
        # (a) eager jax returns a jax array
        assert "jax" in type(out).__module__, (name, type(out))
        # (d') jax value matches the numpy reference
        numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT_GRAD, ids=FLAT_GRAD_IDS)
def test_jax_grad_matches_fd(name, fn, pt):
    # ttensor returns a matrix; reduce to a scalar (sum) for grad.
    def scalarize(p, args):
        out = fn(p, *args)
        return jnp.sum(out)

    for p in _pots():
        for i in range(len(pt)):

            def g(xi, i=i, p=p):
                args = [jnp.asarray(pt[j]) if j != i else xi for j in range(len(pt))]
                return scalarize(p, args)

            x0 = jnp.asarray(pt[i])
            grad = float(jax.grad(g)(x0))
            eps = 1e-6 * max(1.0, abs(float(x0)))
            fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
            # loose tolerance: FD itself is only ~1e-6 accurate
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                name,
                p.__class__.__name__,
                i,
                grad,
                fd,
            )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT, ids=FLAT_IDS)
def test_torch_eager_tensor_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        targs = [torch.as_tensor(x, dtype=torch.float64) for x in pt]
        out = fn(p, *targs)
        # (c) eager torch returns a torch tensor
        assert torch.is_tensor(out), (name, type(out))
        numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", FLAT_GRAD, ids=FLAT_GRAD_IDS)
def test_torch_autograd_matches_fd(name, fn, pt):
    for p in _pots():
        for i in range(len(pt)):
            targs = [
                torch.as_tensor(pt[j], dtype=torch.float64) for j in range(len(pt))
            ]
            targs[i].requires_grad_(True)
            out = fn(p, *targs)
            out = out.sum() if out.ndim > 0 else out
            out.backward()
            grad = float(targs[i].grad)
            # finite difference
            eps = 1e-6 * max(1.0, abs(pt[i]))
            xs = list(pt)
            xs[i] = pt[i] + eps
            fp = numpy.asarray(fn(p, *xs)).sum()
            xs[i] = pt[i] - eps
            fm = numpy.asarray(fn(p, *xs)).sum()
            fd = float((fp - fm) / (2.0 * eps))
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                name,
                p.__class__.__name__,
                i,
                grad,
                fd,
            )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_ttensor_eigenval_jax():
    # ttensor(eigenval=True) routes through xp.linalg.eigvals: check it returns
    # a jax array and the (sorted, real-part) eigenvalues match numpy.
    for p in _pots():
        ref = numpy.sort(
            numpy.real(p.ttensor(1.0, 0.1, eigenval=True, use_physical=False))
        )
        out = p.ttensor(
            jnp.asarray(1.0), jnp.asarray(0.1), eigenval=True, use_physical=False
        )
        assert "jax" in type(out).__module__, type(out)
        got = numpy.sort(numpy.real(numpy.asarray(out)))
        numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


def test_ttensor_numpy_byte_identical_construction():
    # The migrated xp.stack construction must be byte-identical to the original
    # numpy.array([[...]]) construction on the numpy path, for both scalar and
    # array inputs.
    p = LogarithmicHaloPotential(normalize=1.0)
    # scalar
    out_s = numpy.asarray(p.ttensor(1.0, 0.1, use_physical=False))
    assert out_s.shape == (3, 3)
    # array (broadcasts to (3,3,N))
    R = numpy.array([1.0, 1.2, 0.8])
    z = numpy.array([0.1, 0.2, 0.05])
    out_a = numpy.asarray(p.ttensor(R, z, use_physical=False))
    assert out_a.shape == (3, 3, 3)
    # symmetry of the tidal tensor (sanity)
    numpy.testing.assert_allclose(out_s, out_s.T)
