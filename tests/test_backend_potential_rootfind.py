###############################################################################
# test_backend_potential_rootfind.py: multi-backend tests for the scipy-root-
# find / quadrature methods of galpy.potential.Potential that now dispatch
# through galpy.backend.optimize.brentq and galpy.backend.quadrature.quad for
# jax/torch inputs while keeping the numpy path on scipy (byte-identical).
#
# Covered (module-level function forms, all use_physical=False):
#   rl   -- solves r*vc(r) = |lz|              (d rl / d lz)
#   rE   -- solves vc(r)^2/2 + Phi(r,0) = E    (d rE / d E)
#   LcE  -- rE then vcirc                       (d LcE / d E)
#   zvc  -- solves Phi(R,z) + Lz^2/(2R^2) = E   (d zvc / d{R,E,Lz})
#   zvc_range -- the two R roots, returns a stacked [Rmin, Rmax]
#   mass -- Gauss-theorem integrate.quad fallback -> backend quadrature
#           (spherical, forceint, and the z-slab forms)
#
# For each this proves the four discriminating properties:
#   (a) eager jax returns a jax array,
#   (b) jax.grad through it works and matches a finite difference,
#   (c) eager torch returns a torch tensor,
#   (d) numpy returns the SAME value as before (byte-identical to scipy).
#
# lindbladR is intentionally NOT covered: it returns None when there is no
# resonance (a value-branch that cannot be made differentiable) and its
# per-case residual sign varies, so it was left on the numpy/scipy path.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
)
from galpy.potential import LcE as LcE_fn
from galpy.potential import LogarithmicHaloPotential, MiyamotoNagaiPotential
from galpy.potential import mass as mass_fn
from galpy.potential import rE as rE_fn
from galpy.potential import rl as rl_fn
from galpy.potential import zvc as zvc_fn
from galpy.potential import zvc_range as zvc_range_fn

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


# Each entry: (name, callable(pot, *xargs) -> scalar, list of float-arg tuples).
# The single varying scalar argument is differentiated against (index 0).
def _make_scalar_methods():
    return [
        (
            "rl",
            lambda p, lz: rl_fn(p, lz, use_physical=False),
            [(0.4,), (0.8,), (1.3,)],
        ),
        (
            "rE",
            lambda p, E: rE_fn(p, E, use_physical=False),
            [(-1.2,), (-0.8,), (-0.5,)],
        ),
        (
            "LcE",
            lambda p, E: LcE_fn(p, E, use_physical=False),
            [(-1.2,), (-0.8,), (-0.5,)],
        ),
    ]


SCALAR_METHODS = _make_scalar_methods()


def _scalar_flat():
    out = []
    for name, fn, pts in SCALAR_METHODS:
        for pt in pts:
            out.append((name, fn, pt))
    return out


def _scalar_ids():
    out = []
    for name, _fn, pts in SCALAR_METHODS:
        for pt in pts:
            out.append(f"{name}-{'_'.join(str(x) for x in pt)}")
    return out


SCALAR_FLAT = _scalar_flat()
SCALAR_IDS = _scalar_ids()


# ---------------------------------------------------------------------------
# Scalar root-finders: rl, rE, LcE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn,pt", SCALAR_FLAT, ids=SCALAR_IDS)
def test_numpy_baseline_finite(name, fn, pt):
    # numpy still produces finite real values (the unchanged scipy path).
    for p in _pots():
        out = numpy.asarray(fn(p, *pt))
        assert numpy.all(numpy.isfinite(out)), (name, p.__class__.__name__, out)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", SCALAR_FLAT, ids=SCALAR_IDS)
def test_scalar_jax_eager_array_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        out = fn(p, *[jnp.asarray(x) for x in pt])
        # (a) eager jax returns a jax array
        assert "jax" in type(out).__module__, (name, type(out))
        # (d') jax value matches the numpy/scipy reference
        numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pt", SCALAR_FLAT, ids=SCALAR_IDS)
def test_scalar_jax_grad_matches_fd(name, fn, pt):
    for p in _pots():

        def g(x, p=p):
            return fn(p, x)

        x0 = jnp.asarray(pt[0])
        grad = float(jax.grad(g)(x0))
        eps = 1e-6 * max(1.0, abs(float(x0)))
        fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
        assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
            name,
            p.__class__.__name__,
            grad,
            fd,
        )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", SCALAR_FLAT, ids=SCALAR_IDS)
def test_scalar_torch_eager_tensor_and_value(name, fn, pt):
    for p in _pots():
        ref = numpy.asarray(fn(p, *pt))
        out = fn(p, *[torch.as_tensor(x, dtype=torch.float64) for x in pt])
        # (c) eager torch returns a torch tensor
        assert torch.is_tensor(out), (name, type(out))
        numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pt", SCALAR_FLAT, ids=SCALAR_IDS)
def test_scalar_torch_autograd_matches_fd(name, fn, pt):
    for p in _pots():
        x = torch.as_tensor(pt[0], dtype=torch.float64)
        x.requires_grad_(True)
        out = fn(p, x)
        out.backward()
        grad = float(x.grad)
        eps = 1e-6 * max(1.0, abs(pt[0]))
        fp = float(numpy.asarray(fn(p, pt[0] + eps)))
        fm = float(numpy.asarray(fn(p, pt[0] - eps)))
        fd = (fp - fm) / (2.0 * eps)
        assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
            name,
            p.__class__.__name__,
            grad,
            fd,
        )


# ---------------------------------------------------------------------------
# zvc(R, E, Lz): the zero-velocity curve z-root
# ---------------------------------------------------------------------------
# (R, E, Lz) triples with a genuine positive-z solution.
_ZVC_PTS = [(1.0, -1.0, 0.5), (1.2, -0.9, 0.6), (0.8, -1.1, 0.4)]
_ZVC_POTS = [MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0)]


@pytest.mark.parametrize("R,E,Lz", _ZVC_PTS)
def test_zvc_numpy_finite(R, E, Lz):
    for p in _ZVC_POTS:
        out = float(zvc_fn(p, R, E, Lz, use_physical=False))
        assert numpy.isfinite(out) and out > 0.0


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("R,E,Lz", _ZVC_PTS)
def test_zvc_jax(R, E, Lz):
    for p in _ZVC_POTS:
        ref = float(zvc_fn(p, R, E, Lz, use_physical=False))
        out = zvc_fn(
            p, jnp.asarray(R), jnp.asarray(E), jnp.asarray(Lz), use_physical=False
        )
        assert "jax" in type(out).__module__
        numpy.testing.assert_allclose(float(out), ref, rtol=1e-9, atol=1e-11)
        # grad w.r.t. each of R, E, Lz vs finite difference
        base = [R, E, Lz]
        for i in range(3):

            def g(xi, i=i, p=p):
                a = [jnp.asarray(base[j]) if j != i else xi for j in range(3)]
                return zvc_fn(p, a[0], a[1], a[2], use_physical=False)

            x0 = jnp.asarray(base[i])
            grad = float(jax.grad(g)(x0))
            eps = 1e-6 * max(1.0, abs(base[i]))
            fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (i, grad, fd)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("R,E,Lz", _ZVC_PTS)
def test_zvc_torch(R, E, Lz):
    for p in _ZVC_POTS:
        ref = float(zvc_fn(p, R, E, Lz, use_physical=False))
        base = [R, E, Lz]
        targs = [torch.as_tensor(v, dtype=torch.float64) for v in base]
        out = zvc_fn(p, *targs, use_physical=False)
        assert torch.is_tensor(out)
        numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-9, atol=1e-11)
        for i in range(3):
            t = [torch.as_tensor(v, dtype=torch.float64) for v in base]
            t[i].requires_grad_(True)
            o = zvc_fn(p, *t, use_physical=False)
            o.backward()
            grad = float(t[i].grad)
            eps = 1e-6 * max(1.0, abs(base[i]))
            xs = list(base)
            xs[i] = base[i] + eps
            fp = float(zvc_fn(p, *xs, use_physical=False))
            xs[i] = base[i] - eps
            fm = float(zvc_fn(p, *xs, use_physical=False))
            fd = (fp - fm) / (2.0 * eps)
            assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (i, grad, fd)


# ---------------------------------------------------------------------------
# zvc_range(E, Lz): the two R roots, returned as [Rmin, Rmax]
# ---------------------------------------------------------------------------
_ZVCR_PTS = [(-1.0, 0.5), (-0.9, 0.6), (-1.1, 0.4)]


@pytest.mark.parametrize("E,Lz", _ZVCR_PTS)
def test_zvc_range_numpy_finite(E, Lz):
    for p in _ZVC_POTS:
        out = numpy.asarray(zvc_range_fn(p, E, Lz, use_physical=False))
        assert out.shape == (2,)
        assert numpy.all(numpy.isfinite(out)) and out[0] < out[1]


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("E,Lz", _ZVCR_PTS)
def test_zvc_range_jax(E, Lz):
    for p in _ZVC_POTS:
        ref = numpy.asarray(zvc_range_fn(p, E, Lz, use_physical=False))
        out = zvc_range_fn(p, jnp.asarray(E), jnp.asarray(Lz), use_physical=False)
        assert "jax" in type(out).__module__
        numpy.testing.assert_allclose(numpy.asarray(out), ref, rtol=1e-8, atol=1e-10)
        base = [E, Lz]
        # grad of each output component w.r.t. each input vs FD
        for out_idx in range(2):
            for i in range(2):

                def g(xi, i=i, out_idx=out_idx, p=p):
                    a = [jnp.asarray(base[j]) if j != i else xi for j in range(2)]
                    return zvc_range_fn(p, a[0], a[1], use_physical=False)[out_idx]

                x0 = jnp.asarray(base[i])
                grad = float(jax.grad(g)(x0))
                eps = 1e-6 * max(1.0, abs(base[i]))
                fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
                assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                    out_idx,
                    i,
                    grad,
                    fd,
                )


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("E,Lz", _ZVCR_PTS)
def test_zvc_range_torch(E, Lz):
    for p in _ZVC_POTS:
        ref = numpy.asarray(zvc_range_fn(p, E, Lz, use_physical=False))
        out = zvc_range_fn(
            p,
            torch.as_tensor(E, dtype=torch.float64),
            torch.as_tensor(Lz, dtype=torch.float64),
            use_physical=False,
        )
        assert torch.is_tensor(out)
        numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-8, atol=1e-10)
        base = [E, Lz]
        for out_idx in range(2):
            for i in range(2):
                t = [torch.as_tensor(v, dtype=torch.float64) for v in base]
                t[i].requires_grad_(True)
                o = zvc_range_fn(p, *t, use_physical=False)[out_idx]
                o.backward()
                grad = float(t[i].grad)
                eps = 1e-6 * max(1.0, abs(base[i]))
                xs = list(base)
                xs[i] = base[i] + eps
                fp = float(zvc_range_fn(p, *xs, use_physical=False)[out_idx])
                xs[i] = base[i] - eps
                fm = float(zvc_range_fn(p, *xs, use_physical=False)[out_idx])
                fd = (fp - fm) / (2.0 * eps)
                assert numpy.isclose(grad, fd, rtol=1e-4, atol=1e-5), (
                    out_idx,
                    i,
                    grad,
                    fd,
                )


# ---------------------------------------------------------------------------
# mass: Gauss-theorem integrate.quad fallback -> backend quadrature
# ---------------------------------------------------------------------------
# MiyamotoNagai has no analytic _mass, so its mass() takes the quad fallback for
# both the spherical (z is None) and the z-slab forms. Hernquist HAS analytic
# _mass; forceint=True forces it onto the quad fallback too.
_MN = MiyamotoNagaiPotential(a=0.5, b=0.1, normalize=1.0)
_HERN = HernquistPotential(normalize=1.0)


def _mass_cases():
    # (id, callable(pot, R) -> mass, pot, [R values]) -- spherical / forceint.
    return [
        ("sph_mn", lambda p, R: mass_fn(p, R, use_physical=False), _MN, [0.5, 1.0]),
        (
            "forceint_hern",
            lambda p, R: mass_fn(p, R, forceint=True, use_physical=False),
            _HERN,
            [0.5, 1.0, 2.0],
        ),
    ]


MASS_CASES = _mass_cases()
MASS_IDS = [f"{name}-{R}" for name, _f, _p, Rs in MASS_CASES for R in Rs]
MASS_FLAT = [(name, f, p, R) for name, f, p, Rs in MASS_CASES for R in Rs]


@pytest.mark.parametrize("name,fn,pot,R", MASS_FLAT, ids=MASS_IDS)
def test_mass_numpy_finite(name, fn, pot, R):
    out = float(fn(pot, R))
    assert numpy.isfinite(out) and out > 0.0


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("name,fn,pot,R", MASS_FLAT, ids=MASS_IDS)
def test_mass_jax(name, fn, pot, R):
    ref = float(fn(pot, R))
    out = fn(pot, jnp.asarray(R))
    assert "jax" in type(out).__module__, (name, type(out))
    # Fixed-order GL vs scipy adaptive: looser tol than the exact root finds
    # (the spherical integrand sharpens with R). Still tight enough to catch a
    # real regression.
    numpy.testing.assert_allclose(float(out), ref, rtol=1e-4, atol=1e-6)

    def g(x):
        return fn(pot, x)

    x0 = jnp.asarray(R)
    grad = float(jax.grad(g)(x0))
    eps = 1e-6 * max(1.0, abs(R))
    fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
    assert numpy.isclose(grad, fd, rtol=1e-3, atol=1e-5), (name, grad, fd)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name,fn,pot,R", MASS_FLAT, ids=MASS_IDS)
def test_mass_torch(name, fn, pot, R):
    ref = float(fn(pot, R))
    x = torch.as_tensor(R, dtype=torch.float64)
    x.requires_grad_(True)
    out = fn(pot, x)
    assert torch.is_tensor(out), (name, type(out))
    numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-4, atol=1e-6)
    out.backward()
    grad = float(x.grad)
    eps = 1e-6 * max(1.0, abs(R))
    fp = float(fn(pot, R + eps))
    fm = float(fn(pot, R - eps))
    fd = (fp - fm) / (2.0 * eps)
    assert numpy.isclose(grad, fd, rtol=1e-3, atol=1e-5), (name, grad, fd)


# z-slab mass: mass(R, z) -- two integrate.quad pieces, both -> backend quad.
_MASS_SLAB_PTS = [(1.0, 0.3), (1.0, 0.6), (1.5, 0.5)]


@pytest.mark.parametrize("R,z", _MASS_SLAB_PTS)
def test_mass_slab_numpy_finite(R, z):
    out = float(mass_fn(_MN, R, z=z, use_physical=False))
    assert numpy.isfinite(out) and out > 0.0


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
@pytest.mark.parametrize("R,z", _MASS_SLAB_PTS)
def test_mass_slab_jax(R, z):
    ref = float(mass_fn(_MN, R, z=z, use_physical=False))
    out = mass_fn(_MN, jnp.asarray(R), z=jnp.asarray(z), use_physical=False)
    assert "jax" in type(out).__module__
    numpy.testing.assert_allclose(float(out), ref, rtol=1e-6, atol=1e-8)
    base = [R, z]
    for i in range(2):

        def g(xi, i=i):
            a = [jnp.asarray(base[j]) if j != i else xi for j in range(2)]
            return mass_fn(_MN, a[0], z=a[1], use_physical=False)

        x0 = jnp.asarray(base[i])
        grad = float(jax.grad(g)(x0))
        eps = 1e-6 * max(1.0, abs(base[i]))
        fd = float((g(x0 + eps) - g(x0 - eps)) / (2.0 * eps))
        assert numpy.isclose(grad, fd, rtol=1e-3, atol=1e-5), (i, grad, fd)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("R,z", _MASS_SLAB_PTS)
def test_mass_slab_torch(R, z):
    ref = float(mass_fn(_MN, R, z=z, use_physical=False))
    base = [R, z]
    t = [torch.as_tensor(v, dtype=torch.float64) for v in base]
    out = mass_fn(_MN, t[0], z=t[1], use_physical=False)
    assert torch.is_tensor(out)
    numpy.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-6, atol=1e-8)
    for i in range(2):
        tt = [torch.as_tensor(v, dtype=torch.float64) for v in base]
        tt[i].requires_grad_(True)
        o = mass_fn(_MN, tt[0], z=tt[1], use_physical=False)
        o.backward()
        grad = float(tt[i].grad)
        eps = 1e-6 * max(1.0, abs(base[i]))
        xs = list(base)
        xs[i] = base[i] + eps
        fp = float(mass_fn(_MN, xs[0], z=xs[1], use_physical=False))
        xs[i] = base[i] - eps
        fm = float(mass_fn(_MN, xs[0], z=xs[1], use_physical=False))
        fd = (fp - fm) / (2.0 * eps)
        assert numpy.isclose(grad, fd, rtol=1e-3, atol=1e-5), (i, grad, fd)
