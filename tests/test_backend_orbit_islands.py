###############################################################################
# test_backend_orbit_islands.py: Orbit methods that computed on numpy under a
# forced backend now run on the backend -- rguiding/rE/LcE (one vectorized root
# over all points), E's fallback for potentials that refuse 2D arrays, Jacobi's
# Omega.L, a numpy orbit read at a 1-D backend time, and a backend Orbit's
# align_to_orbit. Values match numpy;
# derivatives match finite differences or closed forms.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array, use
from galpy.orbit import Orbit
from galpy.potential import (
    DehnenBarPotential,
    LogarithmicHaloPotential,
    MWPotential2014,
    Potential,
    SpiralArmsPotential,
)

pytestmark = pytest.mark.backend_managed

BACKENDS = []
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

_IC = [[1.0, 0.1, 1.1, 0.1, 0.05, 0.3], [0.8, -0.2, 0.9, -0.05, 0.1, 1.2]]
_TS = numpy.linspace(0.0, 5.0, 101)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _orbit(ic=_IC, pot=MWPotential2014, ts=_TS):
    o = Orbit(ic)
    o.integrate(ts, pot)
    return o


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["rguiding", "rE", "LcE"])
def test_root_characteristic_forced_matches_numpy(backend, fn):
    # forced: one vectorized backend root over every (orbit, time), numpy's value
    o = _orbit()
    tq = _TS[::10]
    ref = getattr(o, fn)(tq, pot=MWPotential2014, use_physical=False)
    with use(backend, force=True):
        got = getattr(o, fn)(tq, pot=MWPotential2014, use_physical=False)
    assert is_backend_array(got)
    assert tuple(got.shape) == ref.shape
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["rguiding", "rE", "LcE"])
def test_root_characteristic_ic_grad_vs_fd(backend, fn):
    # d fn(t=0) / d vT through the backend root; rguiding = rl(R vT) and
    # rE/LcE through E = Phi + v^2/2 -- against a central finite difference
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    ic = numpy.array([1.0, 0.1, 1.1, 0.1, 0.05, 0.3])
    h = 1e-6

    def numpy_at(vT):
        o = Orbit(list(ic[:2]) + [vT] + list(ic[3:]))
        return float(getattr(o, fn)(pot=lp, use_physical=False))

    fd = (numpy_at(ic[2] + h) - numpy_at(ic[2] - h)) / (2.0 * h)
    if backend == "jax":

        def f(vT):
            x = jnp.asarray(ic).at[2].set(vT)
            return getattr(Orbit(x), fn)(pot=lp, use_physical=False).reshape(())

        grad = jax.grad(f)(jnp.asarray(ic[2]))
    else:
        vT = torch.tensor(ic[2], requires_grad=True)
        x = torch.cat([torch.tensor(ic[:2]), vT[None], torch.tensor(ic[3:])])
        val = getattr(Orbit(x), fn)(pot=lp, use_physical=False).reshape(())
        (grad,) = torch.autograd.grad(val, vT)
    numpy.testing.assert_allclose(float(as_numpy(grad)), fd, rtol=1e-7)


class _ScalarOnlyPotential(Potential):
    """Harmonic potential whose _evaluate rejects arrays (forces at any shape)"""

    def __init__(self):
        Potential.__init__(self, amp=1.0)
        self.isNonAxi = False

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        for x in (R, z, t):
            if numpy.ndim(x) > 0:
                raise ValueError("scalar-only")
        return 0.5 * (R**2.0 + z**2.0)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        return -R

    def _zforce(self, R, z, phi=0.0, t=0.0):
        return -z


@pytest.mark.parametrize("backend", BACKENDS)
def test_E_fallback_flat_on_backend(backend):
    # SpiralArmsPotential refuses arrays of more than one dimension, so E(ts)
    # takes the fallback: one flattened backend call, numpy's per-point values
    pot = MWPotential2014 + SpiralArmsPotential()
    o = _orbit(pot=pot, ts=_TS[:21])
    ref = o.E(_TS[:21], pot=pot, use_physical=False)
    with use(backend, force=True):
        got = o.E(_TS[:21], pot=pot, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [6, 5, 4, 3])
def test_E_fallback_scalar_only_on_backend(backend, dim):
    # a scalar-only potential: the flattened call raises too, so E is stacked
    # per point on the backend; checked against the closed-form energy
    pot = _ScalarOnlyPotential()
    ic = {
        6: [[1.0, 0.1, 1.1, 0.1, 0.05, 0.3], [1.4, -0.2, 0.7, 0.0, 0.1, 1.0]],
        5: [[1.0, 0.1, 1.1, 0.1, 0.05], [1.4, -0.2, 0.7, 0.0, 0.1]],
        4: [[1.0, 0.1, 1.1, 0.3], [1.4, -0.2, 0.7, 1.0]],
        3: [[1.0, 0.1, 1.1], [1.4, -0.2, 0.7]],
    }[dim]
    o = Orbit(ic)
    ts = numpy.linspace(0.0, 0.3, 4)
    o.integrate(ts, pot if dim > 4 else pot.toPlanar())
    with use(backend, force=True):
        got = o.E(ts, use_physical=False)
    assert is_backend_array(got)
    R, vR, vT = (
        as_numpy(getattr(o, a)(ts, use_physical=False)) for a in "R vR vT".split()
    )
    ref = 0.5 * R**2.0 + 0.5 * vR**2.0 + 0.5 * vT**2.0
    if dim > 4:
        z, vz = (as_numpy(getattr(o, a)(ts, use_physical=False)) for a in ("z", "vz"))
        ref = ref + 0.5 * z**2.0 + 0.5 * vz**2.0
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_jacobi_vector_omegap_on_backend(backend):
    # a 3-vector pattern speed: Omega . L on the backend, numpy's value
    dp = DehnenBarPotential()
    pot = MWPotential2014 + dp
    o = _orbit(pot=pot, ts=_TS[:21])
    OmegaP = numpy.array([0.0, 0.0, dp.OmegaP()])
    for t in (None, _TS[:21]):
        args = () if t is None else (t,)
        ref = o.Jacobi(*args, pot=pot, OmegaP=OmegaP, use_physical=False)
        with use(backend, force=True):
            got = o.Jacobi(*args, pot=pot, OmegaP=OmegaP, use_physical=False)
        assert is_backend_array(got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_numpy_orbit_at_backend_times(backend):
    # a 1-D backend time on a numpy orbit: the in-backend spline, numpy's values,
    # and dR/dt = vR (the interpolant's slope against the orbit's own vR)
    o = _orbit(ic=_IC[0])
    tq = numpy.array([0.123, 1.77, 3.3333])
    ref = o.R(tq)
    got = o.R(_arr(backend, tq))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-14)
    if backend == "jax":
        dRdt = jax.grad(lambda t: jnp.sum(o.R(t)))(jnp.asarray(tq))
    else:
        t = torch.tensor(tq, requires_grad=True)
        (dRdt,) = torch.autograd.grad(o.R(t).sum(), t)
    numpy.testing.assert_allclose(as_numpy(dRdt), o.vR(tq), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("as_orbit", [True, False])
def test_isochroneapprox_forced_on_backend(backend, as_orbit):
    # forced: the orbit-assembly and angle fit on the backend (the orbits
    # themselves integrate in C: nothing is differentiated), numpy's values
    from galpy.actionAngle import actionAngleIsochroneApprox

    aAI = actionAngleIsochroneApprox(pot=MWPotential2014, b=0.8)
    ic = [1.0, 0.1, 1.1, 0.1, 0.05, 0.3]

    def call():
        if as_orbit:
            return aAI.actionsFreqsAngles(Orbit(ic))
        return aAI.actionsFreqsAngles(*[numpy.array([v]) for v in ic])

    ref = call()
    with use(backend, force=True):
        got = call()
    for r, g in zip(ref, got):
        assert is_backend_array(g)
        numpy.testing.assert_allclose(as_numpy(g), r, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["rguiding", "rE", "LcE"])
def test_root_characteristic_numpy_orbit_backend_potential(backend, fn):
    # a numpy orbit in a potential with a backend parameter: one backend root
    # over all points, numpy's values, and d/dq against a central difference
    o = _orbit(pot=LogarithmicHaloPotential(normalize=1.0, q=0.9))
    tq = _TS[::25]

    def f(q):
        return getattr(o, fn)(
            tq, pot=LogarithmicHaloPotential(normalize=1.0, q=q), use_physical=False
        )

    ref = f(0.9)
    got = f(_arr(backend, 0.9))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10)
    h = 1e-6
    fd = (f(0.9 + h) - f(0.9 - h)) / (2.0 * h)
    if backend == "jax":
        g = jax.jacfwd(lambda q: f(q))(jnp.asarray(0.9))
    else:
        g = torch.autograd.functional.jacobian(f, torch.tensor(0.9))
    numpy.testing.assert_allclose(as_numpy(g), fd, rtol=1e-6, atol=1e-10)


def _scalar_only_linear():
    from galpy.potential import linearPotential

    class _ScalarOnlyLinear(linearPotential):
        def __init__(self):
            linearPotential.__init__(self, amp=1.0)

        def _evaluate(self, x, t=0.0):
            if numpy.ndim(x) > 0 or numpy.ndim(t) > 0:
                raise ValueError("scalar-only")
            return 0.5 * x**2.0

        def _force(self, x, t=0.0):
            return -x

    return _ScalarOnlyLinear()


@pytest.mark.parametrize("backend", ["numpy"] + BACKENDS)
def test_E_fallback_scalar_only_linear(backend):
    # a 1D orbit in a scalar-only potential: E per point (forced: on the backend)
    o = Orbit([[1.0, 0.1], [0.5, -0.3]])
    ts = numpy.linspace(0.0, 0.3, 4)
    o.integrate(ts, _scalar_only_linear())
    with use(backend, force=True):
        E = o.E(ts, use_physical=False)
    assert is_backend_array(E) == (backend != "numpy")
    x, vx = o.x(ts, use_physical=False), o.vx(ts, use_physical=False)
    numpy.testing.assert_allclose(as_numpy(E), 0.5 * x**2.0 + 0.5 * vx**2.0, rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["rguiding", "rE", "LcE"])
def test_root_characteristic_forced_single_point_on_backend(backend, fn):
    # one point of a numpy orbit under a forced backend: the backend root too
    o = Orbit(_IC[0])
    ref = getattr(o, fn)(pot=MWPotential2014, use_physical=False)
    with use(backend, force=True):
        got = getattr(o, fn)(pot=MWPotential2014, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11)


_ALIGN = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_align_to_orbit_backend_orbit(backend):
    # a backend Orbit keeps its rotation (and phi1/phi2) on the backend
    from galpy.util import coords

    o = Orbit(_ALIGN)
    ob = Orbit(_arr(backend, _ALIGN))
    ref = o.align_to_orbit()
    got = ob.align_to_orbit()
    assert is_backend_array(got) and is_backend_array(ob.phi1())
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0.0, atol=3e-15)
    numpy.testing.assert_allclose(as_numpy(ob.phi1()), o.phi1(), rtol=1e-13)
    # radec_to_custom: a backend T with numpy (ra, dec) is on the backend
    ra, dec = numpy.array([10.0, 200.0]), numpy.array([-30.0, 45.0])
    out = coords.radec_to_custom(ra, dec, T=got, degree=True)
    assert is_backend_array(out)
    numpy.testing.assert_allclose(
        as_numpy(out), coords.radec_to_custom(ra, dec, T=ref, degree=True), rtol=1e-13
    )
