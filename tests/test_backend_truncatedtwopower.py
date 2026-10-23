###############################################################################
# test_backend_truncatedtwopower.py: multi-backend tests for TruncatedNFWPotential
# and TruncatedHernquistPotential (the Baltz, Marshall & Oguri 2009 truncation),
# static and with a time-dependent truncation radius.
#
# numpy / jax / torch value parity in every regime (the small-r Taylor series,
# the closed form, the outer integral's asymptotic series); derivatives with
# respect to a, rt and the tabulated rt(t) values against finite differences,
# jit-clean (jitted == eager, finite on both sides of every switch); and the
# gradient of an orbit through a MovingObjectPotential driven by a
# time-dependent kernel with respect to its rt table.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import TruncatedHernquistPotential, TruncatedNFWPotential

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
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
CLASSES = [TruncatedNFWPotential, TruncatedHernquistPotential]

# a = 0.6, rt = 1.7: the series below r = 0.06 and the outer series beyond
# r = 17, so these points cover all three regimes
_PTS = [(1e-6, 1e-6), (0.03, 0.02), (0.7, 0.3), (3.0, -1.0), (80.0, 20.0)]
_TGRID = numpy.linspace(0.0, 4.0, 9)
_RTS = 2.0 + numpy.sin(_TGRID)
_METHODS = ["__call__", "Rforce", "zforce", "R2deriv", "z2deriv", "Rzderiv", "dens"]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


def _kwargs(tdep, backend_name="numpy"):
    if tdep:
        return dict(rt=_asarray(backend_name, _RTS), n=1, tgrid=_TGRID)
    return dict(rt=1.7, n=2)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("tdep", [False, True])
@pytest.mark.parametrize("cls", CLASSES)
def test_value_parity(backend_name, tdep, cls):
    # backend inputs, and backend parameters, give the numpy values
    pn = cls(amp=1.2, a=0.6, **_kwargs(tdep))
    pb = cls(amp=1.2, a=_asarray(backend_name, 0.6), **_kwargs(tdep, backend_name))
    for R, z in _PTS:
        for m in _METHODS:
            want = getattr(pn, m)(R, z, t=1.3)
            for p in (pn, pb):
                got = as_numpy(
                    getattr(p, m)(
                        _asarray(backend_name, R), _asarray(backend_name, z), t=1.3
                    )
                )
                assert numpy.fabs(got / want - 1.0) < 1e-13, (
                    f"{cls.__name__} {m} at (R,z)=({R},{z}) on {backend_name}"
                )
        r = numpy.sqrt(R * R + z * z)
        got = as_numpy(pb.mass(_asarray(backend_name, r), t=1.3))
        assert numpy.fabs(got / pn.mass(r, t=1.3) - 1.0) < 1e-13
    return None


def _grad(backend_name, f, *args):
    """Gradient of the scalar f with respect to each of args"""
    if backend_name == "jax":
        return [
            as_numpy(g)
            for g in jax.grad(f, argnums=tuple(range(len(args))))(
                *[jnp.asarray(a) for a in args]
            )
        ]
    targs = [torch.tensor(a, requires_grad=True) for a in args]
    f(*targs).backward()
    return [as_numpy(a.grad) for a in targs]


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("n", [1, 2])
def test_param_grad_vs_fd(backend_name, cls, n):
    # d/da and d/drt of the potential and the radial force, in every regime,
    # against central differences
    h = 1e-6
    for R, z in _PTS:
        for m in ("__call__", "Rforce", "R2deriv"):
            f = lambda a, rt: getattr(cls(amp=1.2, a=a, rt=rt, n=n), m)(R, z)
            ga, grt = _grad(backend_name, f, 0.6, 1.7)
            fd_a = (f(0.6 + h, 1.7) - f(0.6 - h, 1.7)) / (2 * h)
            fd_rt = (f(0.6, 1.7 + h) - f(0.6, 1.7 - h)) / (2 * h)
            scale = max(numpy.fabs(fd_a), numpy.fabs(fd_rt))
            assert numpy.fabs(ga - fd_a) < 1e-7 * scale, (
                f"{cls.__name__} n={n} d{m}/da at R={R}: AD {ga} FD {fd_a}"
            )
            assert numpy.fabs(grt - fd_rt) < 1e-7 * scale, (
                f"{cls.__name__} n={n} d{m}/drt at R={R}: AD {grt} FD {fd_rt}"
            )
    return None


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("cls", CLASSES)
def test_jit_grad_matches_eager(cls):
    # jitted gradients match the eager ones and are finite on both sides of the
    # small-r and large-r switches (each branch is evaluated at a safe point
    # where it is not used, so neither can NaN-poison the other's gradient)
    from galpy.potential.TruncatedTwoPowerSphericalPotential import (
        _LARGE_X,
        _SMALL_X,
    )

    a, rt = 0.6, 1.7
    xs = [_SMALL_X * a * (1 - 1e-9), _SMALL_X * a * (1 + 1e-9)]
    xs += [_LARGE_X * rt * (1 - 1e-9), _LARGE_X * rt * (1 + 1e-9), 1e-9, 1e6]
    for r in xs:
        f = lambda a, rt: (
            cls(amp=1.2, a=a, rt=rt, n=2).Rforce(r, 0.0)
            + cls(amp=1.2, a=a, rt=rt, n=2)(r, 0.0)
        )
        eager = jax.grad(f, argnums=(0, 1))(jnp.float64(a), jnp.float64(rt))
        jitted = jax.jit(jax.grad(f, argnums=(0, 1)))(jnp.float64(a), jnp.float64(rt))
        for e, j in zip(eager, jitted):
            assert numpy.isfinite(float(e))
            # XLA reorders the arithmetic under jit: tens of ulps
            assert numpy.fabs(float(j) - float(e)) <= 1e-13 * numpy.fabs(float(e))
    return None


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("cls", CLASSES)
def test_timedep_table_grad_vs_fd(backend_name, cls):
    # the derivative with respect to the tabulated rt values (through the
    # backend's not-a-knot spline solve) against central differences
    h = 1e-6
    for t in (0.0, 1.3, 3.9, 6.0):
        f = lambda rv: cls(amp=1.2, a=0.6, rt=rv, n=1, tgrid=_TGRID).Rforce(
            0.7, 0.3, t=t
        )
        (g,) = _grad(backend_name, f, _RTS)
        fd = numpy.array(
            [
                (f(_RTS + h * e) - f(_RTS - h * e)) / (2 * h)
                for e in numpy.eye(len(_RTS))
            ]
        )
        assert numpy.amax(numpy.fabs(g - fd)) < 1e-7 * numpy.amax(numpy.fabs(fd)), (
            f"{cls.__name__}: d Rforce / d rt-table at t={t} on {backend_name}"
        )
        if t > _TGRID[-1]:  # held at the last value: only it matters (the
            # last interval's polynomial reproduces it to round-off)
            assert numpy.amax(numpy.fabs(g[:-1])) < 1e-14 * numpy.fabs(g[-1])
    return None


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_movingobject_timedep_orbit_grad():
    # jax.grad of an orbit's final position through a MovingObjectPotential whose
    # kernel has a time-dependent rt, with respect to the rt table and to a,
    # against central differences of C orbits
    from galpy import potential
    from galpy.orbit import Orbit

    host = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    sat = Orbit([1.5, 0.0, 0.9, 0.2, 0.1, 0.0])
    sat.integrate(numpy.linspace(-0.5, 1.5, 401), host, method="dop853_c")
    tg = numpy.linspace(0.0, 1.0, 4)
    rts = 0.6 - 0.1 * tg
    ts = numpy.linspace(0.0, 1.0, 11)
    ic = numpy.array([1.2, 0.1, 1.0, 0.1, 0.0, 0.3])

    def final_x(rv, a=0.1, method="diffrax"):
        kern = TruncatedHernquistPotential(amp=0.05, a=a, rt=rv, n=1, tgrid=tg)
        o = Orbit(jnp.asarray(ic) if method == "diffrax" else ic)
        o.integrate(
            ts,
            host + potential.MovingObjectPotential(sat, pot=kern),
            method=method,
            rtol=1e-10,
            atol=1e-10,
        )
        return o.x(ts[-1])

    g_rt, g_a = jax.grad(final_x, argnums=(0, 1))(jnp.asarray(rts), jnp.float64(0.1))
    h = 1e-5
    fd_rt = numpy.array(
        [
            (
                final_x(rts + h * e, method="dop853_c")
                - final_x(rts - h * e, method="dop853_c")
            )
            / (2 * h)
            for e in numpy.eye(len(rts))
        ]
    )
    fd_a = (
        final_x(rts, a=0.1 + h, method="dop853_c")
        - final_x(rts, a=0.1 - h, method="dop853_c")
    ) / (2 * h)
    scale = max(numpy.amax(numpy.fabs(fd_rt)), numpy.fabs(fd_a))
    assert numpy.amax(numpy.fabs(as_numpy(g_rt) - fd_rt)) < 1e-5 * scale
    assert numpy.fabs(float(g_a) - fd_a) < 1e-5 * scale
    return None
