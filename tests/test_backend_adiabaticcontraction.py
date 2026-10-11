###############################################################################
# test_backend_adiabaticcontraction.py: AdiabaticContractionWrapperPotential
# built on the backend of a differentiated parameter (of the halo, the baryons,
# or f_bar): enclosed masses, the fixed-point contraction, Rvir (Gnedin) and the
# interpolated force all in that namespace. Asserts, for all three methods:
#   * values vs the numpy wrapper (forces, density, potential, mass, vcirc),
#   * d/d(halo a, disk amp, bulge a, f_bar) vs h-converged finite differences,
#   * jax eager / jit / vmap and torch backward / torch.compile.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    AdiabaticContractionWrapperPotential,
    HernquistPotential,
    MiyamotoNagaiPotential,
    NFWPotential,
    PlummerPotential,
    evaluateDensities,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluatezforces,
    mass,
    vcirc,
)

pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_METHODS = ["cautun", "blumenthal", "gnedin"]
# halo a, disk (or Plummer) amp, bulge a, f_bar
_P0 = numpy.array([3.0, 1.0, 0.3, 0.157])
# (R, z) straddling the grid's rmax=50 (Kepler extrapolation beyond it)
_R = numpy.array([0.3, 2.0, 8.0, 60.0])
_Z = numpy.array([0.1, -0.5, 1.5, 3.0])
_RM = numpy.array([0.5, 5.0])  # mass / vcirc radii


def _build(p, method, disk):
    # disk=True: a MiyamotoNagai disk, whose mass is a backend quadrature;
    # disk=False: all enclosed masses analytic
    second = (
        MiyamotoNagaiPotential(amp=p[1], a=3.0, b=0.2)
        if disk
        else PlummerPotential(amp=p[1], b=1.5)
    )
    return AdiabaticContractionWrapperPotential(
        pot=NFWPotential(amp=2.0, a=p[0]),
        baryonpot=HernquistPotential(amp=0.5, a=p[2]) + second,
        method=method,
        f_bar=p[3],
    )


def _outputs(ac, cat):
    # pot first: its len(_R) entries carry the numpy Phi0 correction below
    return cat(
        [
            evaluatePotentials(ac, _R, _Z),
            evaluateRforces(ac, _R, _Z),
            evaluatezforces(ac, _R, _Z),
            evaluateDensities(ac, _R, _Z),
            evaluateR2derivs(ac, _R, _Z),
            mass(ac, _RM),
            vcirc(ac, _RM),
        ]
    )


def _F(backend, method, disk):
    """p -> all outputs, with the wrapper built on p's backend"""
    if backend == "jax":
        return lambda p: _outputs(_build(p, method, disk), jnp.concatenate)
    return lambda p: _outputs(
        _build(p, method, disk), lambda o: torch.cat([torch.as_tensor(x) for x in o])
    )


def _gold_mass(Pot, R, use_physical=False):
    # numpy's mass() of a disk uses scipy's default quad, 3e-8 off and noisy in
    # R (the spline's second derivative amplifies that to 6e-7 in the density);
    # this one is converged, for a reference at the backend's precision
    from scipy import integrate

    from galpy.potential.Potential import mass as _mass

    def one(p):
        if not isinstance(p, MiyamotoNagaiPotential):
            return _mass(p, R, use_physical=False)
        f = lambda th: p.rforce(R * numpy.sin(th), R * numpy.cos(th)) * numpy.sin(th)
        q = integrate.quad(
            f,
            0.0,
            numpy.pi,
            points=[numpy.pi / 2.0],
            epsabs=0.0,
            epsrel=1e-13,
            limit=200,
        )[0]
        return -(R**2.0) * q / 2.0

    return sum(one(p) for p in getattr(Pot, "_potlist", [Pot]))


@pytest.fixture(scope="module")
def numpy_ref():
    """The numpy wrapper's outputs, with converged enclosed masses and Phi0.

    numpy integrates the piecewise-linear force for Phi0 with scipy's quad,
    which hits its 50-subinterval limit (1e-6 off); the backend sums the exact
    trapezoid, so the potential is shifted by that constant."""
    from unittest import mock

    cache = {}

    def ref(method, disk):
        if (method, disk) not in cache:
            with mock.patch("galpy.potential.mass", _gold_mass):
                ac = _build(_P0, method, disk)
            out = _outputs(ac, numpy.concatenate)
            exact = (
                numpy.trapezoid(ac._rforce_grid, ac._rgrid)
                + ac._rforce_grid[-1] * ac._rgrid[-1]
            )
            out[: len(_R)] += exact - ac._Phi0
            cache[(method, disk)] = out
        return cache[(method, disk)]

    return ref


_RTOL = 1e-12  # measured <= 4e-14


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("method", _METHODS)
def test_jax_jit_values_vs_numpy(method, disk, numpy_ref):
    got = jax.jit(_F("jax", method, disk))(jnp.asarray(_P0))
    numpy.testing.assert_allclose(
        numpy.asarray(got), numpy_ref(method, disk), rtol=_RTOL, atol=1e-14
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("method", _METHODS)
def test_torch_values_vs_numpy(method, disk, numpy_ref):
    p = torch.tensor(_P0, dtype=torch.float64, requires_grad=True)
    got = _F("torch", method, disk)(p)
    assert got.requires_grad
    numpy.testing.assert_allclose(
        got.detach().numpy(), numpy_ref(method, disk), rtol=_RTOL, atol=1e-14
    )


def _fd(F, method):
    """d(outputs)/dp by central differences of the backend F, h-converged.

    Cautun is smooth in p: Richardson over h = (2, 1, 0.5)e-3 p. Blumenthal and
    Gnedin interpolate on the contracted radii, so they are smooth only between
    knot crossings: one tiny step (1e-6 p) at which no output crosses one, with
    h-convergence asserted against 2e-6 p."""
    cols = []
    for i in range(len(_P0)):

        def cd(h):
            e = numpy.zeros_like(_P0)
            e[i] = h * _P0[i]
            return (F(_P0 + e) - F(_P0 - e)) / (2.0 * e[i])

        if method == "cautun":
            c = [cd(h) for h in (2e-3, 1e-3, 5e-4)]
            d = [(4.0 * c[k + 1] - c[k]) / 3.0 for k in (0, 1)]
            tol = 1e-8
        else:
            d = [cd(2e-6), cd(1e-6)]
            tol = 1e-6
        numpy.testing.assert_allclose(
            d[0], d[1], rtol=tol, atol=tol * float(numpy.max(numpy.abs(d[1])))
        )
        cols.append(d[1])
    return numpy.stack(cols, axis=1)


def _assert_grad(jac, fd, method):
    # measured AD vs FD: <= 1e-10 (Cautun); <= 3e-8 at the 1e-6 step's
    # round-off (the others, densities worst); atol per parameter column
    rtol = 1e-9 if method == "cautun" else 1e-7
    tol = rtol * (numpy.abs(fd) + numpy.max(numpy.abs(fd), axis=0))
    bad = numpy.abs(jac - fd) > tol
    assert not bad.any(), f"AD vs FD at {numpy.argwhere(bad)}: {jac[bad]} vs {fd[bad]}"


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("method", _METHODS)
def test_jax_jit_grad_vs_finite_difference(method):
    F = _F("jax", method, True)
    Fj = jax.jit(F)
    jac = jax.jit(jax.jacfwd(F))(jnp.asarray(_P0))
    fd = _fd(lambda p: numpy.asarray(Fj(jnp.asarray(p))), method)
    _assert_grad(numpy.asarray(jac), fd, method)
    # reverse mode, through the same jit
    g = jax.jit(jax.grad(lambda p: F(p)[5]))(jnp.asarray(_P0))
    numpy.testing.assert_allclose(numpy.asarray(g), numpy.asarray(jac)[5], rtol=1e-12)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_eager_grad_and_vmap():
    # eager (no jit): the fixed point's Python-loop iteration
    F = _F("jax", "gnedin", True)
    k = len(_R) + 1  # Rforce at R=2
    g = jax.grad(lambda p: F(p)[k])(jnp.asarray(_P0))
    jac = jax.jit(jax.jacfwd(F))(jnp.asarray(_P0))
    numpy.testing.assert_allclose(numpy.asarray(g), numpy.asarray(jac)[k], rtol=1e-12)
    # vmap over two halo scale lengths: each as if built alone
    P = jnp.asarray(numpy.stack([_P0, _P0 * [1.2, 1.0, 1.0, 1.0]]))
    gv = jax.vmap(jax.grad(lambda p: F(p)[k]))(P)
    numpy.testing.assert_allclose(numpy.asarray(gv[0]), numpy.asarray(g), rtol=1e-12)
    g1 = jax.jit(jax.grad(lambda p: F(p)[k]))(P[1])
    numpy.testing.assert_allclose(numpy.asarray(gv[1]), numpy.asarray(g1), rtol=1e-12)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("method", _METHODS)
def test_torch_backward_vs_finite_difference(method):
    F = _F("torch", method, True)

    def Fnp(p):  # a grad-carrying leaf keeps the backend construction
        return F(torch.tensor(p, requires_grad=True)).detach().numpy()

    p = torch.tensor(_P0, dtype=torch.float64, requires_grad=True)
    out = F(p)
    jac = numpy.stack(
        [
            torch.autograd.grad(out[k], p, retain_graph=True)[0].numpy()
            for k in range(out.shape[0])
        ]
    )
    _assert_grad(jac, _fd(Fnp, method), method)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_compile_evaluation():
    # a differentiated wrapper evaluated under torch.compile: values and
    # d/d(parameters) as eager
    p = torch.tensor(_P0, dtype=torch.float64, requires_grad=True)
    ac = _build(p, "cautun", True)
    R = torch.tensor(_R)
    Z = torch.tensor(_Z)
    f = lambda R, z: evaluateRforces(ac, R, z) + evaluatePotentials(ac, R, z)
    eager = f(R, Z)
    got = torch.compile(f, backend="aot_eager")(R, Z)
    numpy.testing.assert_allclose(
        got.detach().numpy(), eager.detach().numpy(), rtol=1e-13
    )
    (ge,) = torch.autograd.grad(eager.sum(), p, retain_graph=True)
    (gc,) = torch.autograd.grad(got.sum(), p)
    numpy.testing.assert_allclose(gc.numpy(), ge.numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    "backend", [b for b, m in (("jax", jax), ("torch", torch)) if m is not None]
)
def test_numpy_built_wrapper_on_backend_coordinates(backend):
    # a float-parameter wrapper is the numpy/scipy one; it now opts in to
    # backend coordinates (coerced at the boundary) and evaluates on them
    from galpy.backend import as_numpy, is_backend_array, is_backend_compatible

    ac = _build(_P0, "blumenthal", True)
    assert is_backend_compatible(ac)
    xp = jnp if backend == "jax" else torch
    for fn in (evaluatePotentials, evaluateRforces, evaluateDensities):
        got = fn(ac, xp.asarray(_R), xp.asarray(_Z))
        assert is_backend_array(got)
        numpy.testing.assert_allclose(as_numpy(got), fn(ac, _R, _Z), rtol=1e-12)


@pytest.mark.parametrize(
    "backend", [b for b, m in (("jax", jax), ("torch", torch)) if m is not None]
)
@pytest.mark.parametrize("wrtcrit", [True, False])
def test_nfw_rvir_differentiable(backend, wrtcrit):
    # Gnedin's orbit-averaged radius needs the halo's Rvir: a root in the
    # halo's parameters, now on their backend (float64 bracket for a float64 a)
    a0, kw = 3.0, dict(overdens=180.0, wrtcrit=wrtcrit)
    rv = lambda a: NFWPotential(amp=2.0, a=a).rvir(**kw)
    if backend == "jax":
        got = rv(jnp.asarray(a0))  # untraced: the scipy root
        g = jax.grad(rv)(a0)
        got_t = jax.jit(rv)(a0)
    else:
        at = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        got_t = rv(at)
        (g,) = torch.autograd.grad(got_t, at)
        got = got_t
    numpy.testing.assert_allclose(float(got), rv(a0), rtol=1e-14)
    numpy.testing.assert_allclose(float(got_t), rv(a0), rtol=1e-14)
    c = [(rv(a0 + h) - rv(a0 - h)) / (2.0 * h) for h in (1e-2, 5e-3)]
    fd = (4.0 * c[1] - c[0]) / 3.0  # Richardson
    numpy.testing.assert_allclose(float(g), fd, rtol=1e-9)


@pytest.mark.parametrize(
    "backend", [b for b, m in (("jax", jax), ("torch", torch)) if m is not None]
)
def test_amp_honoured_on_the_backend(backend):
    # Phi, forces, density, mass scale with amp (the numpy path ignores amp)
    def F(amp, p):
        ac = AdiabaticContractionWrapperPotential(
            amp=amp,
            pot=NFWPotential(amp=2.0, a=p),
            baryonpot=HernquistPotential(amp=0.5, a=0.3),
        )
        return evaluateRforces(ac, 2.0, 0.3), mass(ac, 5.0)

    if backend == "jax":
        g = jax.jacrev(lambda amp: jnp.stack(F(amp, jnp.asarray(3.0))))(1.7)
        v = jnp.stack(F(1.0, jnp.asarray(3.0)))
    else:
        amp = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
        out = F(amp, torch.tensor(3.0, dtype=torch.float64))
        g = torch.stack([torch.autograd.grad(o, amp)[0] for o in out])
        v = torch.stack(
            F(1.0, torch.tensor(3.0, dtype=torch.float64, requires_grad=True))
        )
        v, g = v.detach(), g.detach()
    numpy.testing.assert_allclose(numpy.asarray(g), numpy.asarray(v), rtol=1e-14)


@pytest.mark.parametrize(
    "backend", [b for b, m in (("jax", jax), ("torch", torch)) if m is not None]
)
def test_forced_mode_float_coordinates(backend):
    # under use(..., force=True), plain-float coordinates and parameters built
    # inside the differentiated function: d Rforce(2, 0.3) / d(halo a) vs
    # Richardson FD of the numpy wrapper (all-analytic masses)
    from galpy.backend import use

    def F(a):
        ac = AdiabaticContractionWrapperPotential(
            pot=NFWPotential(amp=2.0, a=a),
            baryonpot=HernquistPotential(amp=0.5, a=0.3)
            + PlummerPotential(amp=1.0, b=1.5),
            method="cautun",
        )
        return evaluateRforces(ac, 2.0, 0.3)

    a0 = 3.0
    c = [(F(a0 + h) - F(a0 - h)) / (2.0 * h) for h in (6e-3, 3e-3)]
    fd = (4.0 * c[1] - c[0]) / 3.0
    with use(backend, force=True):
        if backend == "jax":
            g = jax.grad(F)(a0)
            gj = jax.jit(jax.grad(F))(a0)
            numpy.testing.assert_allclose(float(gj), float(g), rtol=1e-12)
        else:
            a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
            (g,) = torch.autograd.grad(F(a), a)
    numpy.testing.assert_allclose(float(g), fd, rtol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_numpy_built_wrapper_under_jit_in_the_coordinates():
    # the float-parameter (scipy) wrapper, jitted and differentiated in R
    ac = _build(_P0, "cautun", True)
    f = lambda R: evaluateRforces(ac, R, 0.3)
    numpy.testing.assert_allclose(float(jax.jit(f)(2.0)), f(2.0), rtol=1e-13)
    numpy.testing.assert_allclose(
        float(jax.jit(jax.grad(f))(2.0)),
        evaluateR2derivs(ac, 2.0, 0.3) * -1.0,
        rtol=1e-12,
    )
