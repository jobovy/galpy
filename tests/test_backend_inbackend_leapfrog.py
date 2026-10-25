###############################################################################
# test_backend_inbackend_leapfrog.py: the in-backend kick-drift-kick leapfrog
# (Orbit.integrate(method="leapfrog_jax"/"leapfrog_torch"),
# galpy.backend._reference.integrate_orbit_leapfrog).
#
# Proves, on jax and on torch:
#   1. final states converge to dop853_c at second order in the step, for a few
#      thousand orbits in MWPotential2014 + a MovingObjectPotential + a
#      NonInertialFrameForce, and for every phase-space dimension (torch and
#      jax also agree to round-off),
#   2. energy is conserved to second order with no secular drift over 1e4 steps
#      (a non-symplectic second-order method drifts on the same orbits; jax),
#   3. reverse-mode gradients w.r.t. a potential parameter, an initial
#      condition and the final time match central differences of the same
#      leapfrog, for nblock=1 and the default,
#   4. the reverse pass's memory grows as sqrt(nsteps) by default (as nsteps
#      for nblock=1),
#   5. float32 inputs integrate in float32 end to end and agree with float64
#      to the truncation error down to the round-off floor,
#   6. the Orbit route adds (almost) nothing to the traced computation (jax),
#      and the error paths are clear.
###############################################################################
import math
from functools import partial

import numpy
import pytest

from galpy.backend import as_numpy, get_namespace
from galpy.backend._reference import integrate_orbit_leapfrog
from galpy.backend._reference.inbackend_ode import (
    _default_nblock,
    _velocity_dependent,
)
from galpy.orbit import Orbit
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    HernquistPotential,
    IsothermalDiskPotential,
    MiyamotoNagaiPotential,
    MovingObjectPotential,
    MWPotential2014,
    NFWPotential,
    NonInertialFrameForce,
    toPlanarPotential,
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

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

needs_jax = pytest.mark.skipif(jax is None, reason="jax not installed")
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


def _arr(backend, x, dtype="float64"):
    if backend == "jax":
        return jnp.asarray(x, dtype=getattr(jnp, dtype))
    return torch.tensor(numpy.asarray(x), dtype=getattr(torch, dtype))


def _integrate(backend, ic, ts, pot, **kw):
    o = Orbit(ic)
    o.integrate(ts, pot, method=f"leapfrog_{backend}", **kw)
    return o


def _mnh(amp_halo=None):
    """MiyamotoNagai + NFW + Hernquist (normalized: numpy-scalar amplitudes)."""
    halo = (
        NFWPotential(normalize=0.35, a=2.0)
        if amp_halo is None
        else NFWPotential(amp=amp_halo, a=2.0)
    )
    return (
        MiyamotoNagaiPotential(normalize=0.6, a=0.375, b=0.035)
        + halo
        + HernquistPotential(normalize=0.05, a=0.07)
    )


def _frame_force():
    # a time-dependent frame acceleration (velocity-independent: no rotation)
    return NonInertialFrameForce(
        a0=[
            lambda t: 0.02 * get_namespace(t).cos(0.4 * t),
            lambda t: -0.015 * get_namespace(t).sin(0.3 * t + 0.2),
            lambda t: 0.01 * get_namespace(t).cos(0.5 * t + 1.0),
        ]
    )


def _ics(n, seed=1):
    rng = numpy.random.default_rng(seed)
    return numpy.column_stack(
        [
            rng.uniform(0.6, 2.0, n),
            rng.normal(0.0, 0.15, n),
            rng.uniform(0.7, 1.05, n),
            rng.normal(0.0, 0.1, n),
            rng.normal(0.0, 0.08, n),
            rng.uniform(0.0, 2.0 * numpy.pi, n),
        ]
    )


def _cart(y):
    """Orbit-order (..., 6) -> Cartesian (..., 6), so phi wrapping is moot."""
    y = numpy.asarray(as_numpy(y), dtype=float)
    R, vR, vT, z, vz, phi = (y[..., i] for i in range(6))
    c, s = numpy.cos(phi), numpy.sin(phi)
    return numpy.stack([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz], -1)


def _final(backend, pot, ts, nsteps, nblock=None):
    """Orbit route (jitted on jax): (N, 6) initial conditions -> (N, 6) final
    states"""

    def f(ic):
        o = _integrate(
            backend, ic, ts, pot, inbackend_kwargs={"nsteps": nsteps, "nblock": nblock}
        )
        return o.getOrbit()[:, -1]

    return jax.jit(f) if backend == "jax" else f


# --- 1. second-order convergence to dop853_c ---------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_matches_dop853c_second_order(backend):
    # the object's orbit spans the integration; few knots keep the compile short
    oobj = Orbit([4.0, -0.3, 0.6, 1.0, 0.1, 0.5])
    oobj.integrate(numpy.linspace(-1.0, 6.0, 141), MWPotential2014, method="dop853_c")
    mop = MovingObjectPotential(oobj, pot=HernquistPotential(amp=0.05, a=0.1))
    pot = MWPotential2014[0] + MWPotential2014[1] + MWPotential2014[2] + mop
    pot = pot + _frame_force()
    ic = _ics(2000)
    ts = numpy.array([0.0, 5.0])
    oc = Orbit(ic)
    oc.integrate(ts, pot, method="dop853_c")
    xc = _cart(oc.getOrbit()[:, -1])
    errs = []
    for nsteps in (250, 500, 1000):
        x = _cart(_final(backend, pot, ts, nsteps)(_arr(backend, ic)))
        errs.append(numpy.max(numpy.fabs(x - xc), axis=1))
    errs = numpy.array(errs)
    for coarse, fine in zip(errs[:-1], errs[1:]):
        order = math.log2(numpy.sqrt(numpy.mean(coarse**2) / numpy.mean(fine**2)))
        assert abs(order - 2.0) < 1e-3, f"order {order}, expected 2"
        ratio = coarse / fine
        assert numpy.all((ratio > 3.95) & (ratio < 4.05)), (
            f"per-orbit error ratios for halving dt span {ratio.min()}-{ratio.max()}"
        )
    assert numpy.max(errs[-1]) < 1e-4  # absolute accuracy at the finest step


_PHASE_ICS = {
    2: numpy.array([[0.1, 0.2], [0.3, -0.1]]),
    3: _ics(3)[:, :3],
    4: _ics(3)[:, [0, 1, 2, 5]],
    5: _ics(3)[:, :5],
    6: _ics(3),
}


def _phase_pot(phasedim):
    return IsothermalDiskPotential(amp=1.0, sigma=0.3) if phasedim == 2 else _mnh()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("phasedim", [2, 3, 4, 5, 6])
def test_leapfrog_phasedims_second_order(backend, phasedim):
    pot = _phase_pot(phasedim)
    ic = _PHASE_ICS[phasedim]
    ts = numpy.linspace(0.0, 6.0, 4)
    oc = Orbit(ic)
    oc.integrate(ts, pot, method="dop853_c")
    yc = numpy.asarray(oc.getOrbit())

    def comparable(y):  # phi -> (x, y) where present
        y = numpy.asarray(as_numpy(y), dtype=float)
        if phasedim == 4:
            return numpy.concatenate(
                [
                    y[..., :3],
                    (y[..., 0] * numpy.cos(y[..., 3]))[..., None],
                    (y[..., 0] * numpy.sin(y[..., 3]))[..., None],
                ],
                -1,
            )
        return _cart(y) if phasedim == 6 else y

    errs = []
    for nsteps in (600, 1200):
        o = _integrate(
            backend, _arr(backend, ic), ts, pot, inbackend_kwargs={"nsteps": nsteps}
        )
        y = o.getOrbit()
        assert tuple(y.shape) == yc.shape
        errs.append(numpy.max(numpy.fabs(comparable(y) - comparable(yc))[:, 1:]))
    assert errs[1] < 1e-4
    assert 3.9 < errs[0] / errs[1] < 4.1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nblock", [None, 1, 2, 10, 40])
def test_leapfrog_output_grid(backend, nblock):
    # 10 output intervals of 40 steps: blocks holding several intervals (2),
    # exactly one (10) and several blocks per interval (40) give the same states
    # as integrating to each output time separately with the same step
    pot = _mnh()
    ic = _arr(backend, _ics(4))
    ts = numpy.linspace(0.0, 4.0, 11)
    o = _integrate(
        backend, ic, ts, pot, inbackend_kwargs={"nsteps": 400, "nblock": nblock}
    )
    y = as_numpy(o.getOrbit())
    numpy.testing.assert_allclose(_cart(y[:, 0]), _cart(ic), rtol=0.0, atol=1e-15)
    for i in (1, 5, 10):
        yi = integrate_orbit_leapfrog(pot, ic, ts[[0, i]], nsteps=40 * i)[-1]
        numpy.testing.assert_allclose(_cart(y[:, i]), _cart(yi), rtol=0.0, atol=1e-13)
    # accessors at the integration times index the stored states
    numpy.testing.assert_array_equal(as_numpy(o.R(ts)), y[..., 0])


@needs_jax
@needs_torch
@pytest.mark.parametrize("phasedim", [2, 3, 4, 5, 6])
def test_leapfrog_torch_matches_jax(phasedim):
    # the same steps in the same order: the two agree to round-off
    pot = _phase_pot(phasedim)
    if phasedim > 4:  # (planar orbits drop the frame force's z component)
        pot = pot + _frame_force()
    ic = _PHASE_ICS[phasedim]
    ts = numpy.linspace(0.0, 6.0, 4)
    for nblock in (None, 1):
        kw = {"inbackend_kwargs": {"nsteps": 600, "nblock": nblock}}
        yj = _integrate("jax", _arr("jax", ic), ts, pot, **kw).getOrbit()
        yt = _integrate("torch", _arr("torch", ic), ts, pot, **kw).getOrbit()
        numpy.testing.assert_allclose(as_numpy(yt), as_numpy(yj), rtol=0.0, atol=1e-13)


# --- 2. energy ------------------------------------------------------------------
def _energy_errors(method, nsteps, **kw):
    pot = _mnh()
    ic = jnp.asarray(
        [
            [1.0, 0.2, 0.9, 0.05, 0.1, 0.3],
            [1.5, -0.1, 1.1, 0.0, 0.2, 1.0],
            [0.7, 0.3, 0.7, 0.1, -0.05, 2.0],
        ]
    )
    ts = numpy.linspace(0.0, 100.0, 1001)
    o = Orbit(ic)
    o.integrate(ts, pot, method=method, inbackend_kwargs=dict(kw, nsteps=nsteps))
    E = numpy.asarray(o.E(ts, pot=pot, use_physical=False))
    dE = (E - E[:, :1]) / numpy.fabs(E[:, :1])
    # secular drift: the linear trend over the integration, relative to the error
    trend = numpy.polyfit(ts, dE.T, 1)[0] * ts[-1]
    return numpy.max(numpy.fabs(dE), axis=1), numpy.fabs(trend)


@needs_jax
def test_leapfrog_energy_second_order_no_drift():
    dEmax, trend = _energy_errors("leapfrog_jax", 10000)
    dEmax2, trend2 = _energy_errors("leapfrog_jax", 20000)
    assert numpy.all(dEmax < 1e-4)
    numpy.testing.assert_allclose(dEmax / dEmax2, 4.0, rtol=0.01)
    assert numpy.all(trend < 0.05 * dEmax) and numpy.all(trend2 < 0.05 * dEmax2)
    # control: the drift measure sees a non-symplectic second-order method drift
    diffrax = pytest.importorskip("diffrax")
    dEmaxh, trendh = _energy_errors("diffrax", 10000, solver=diffrax.Heun())
    assert numpy.max(trendh / dEmaxh) > 0.5


# --- 3. gradients -----------------------------------------------------------------
_GIC = _ics(20, seed=3)
_GW = numpy.random.default_rng(4).normal(size=(20, 3))
_GONE = numpy.zeros_like(_GIC)
_GONE[0, 0] = 1.0


def _gloss(backend, nblock, theta, R0, t1):
    pot = _mnh(amp_halo=theta) + _frame_force()
    ic = _arr(backend, _GIC) + (R0 - _GIC[0, 0]) * _arr(backend, _GONE)
    xp = get_namespace(ic)
    o = _integrate(
        backend,
        ic,
        xp.stack([xp.zeros_like(t1), t1])
        if hasattr(t1, "shape")
        else numpy.array([0.0, t1]),
        pot,
        inbackend_kwargs={"nsteps": 400, "nblock": nblock},
    )
    y = o.getOrbit()[:, -1]
    R, vT, z, phi = y[:, 0], y[:, 2], y[:, 3], y[:, 5]
    x = xp.stack([R * xp.cos(phi), R * xp.sin(phi), z], axis=-1)
    return xp.sum(_arr(backend, _GW) * x)


_GARGS = (4.0, float(_GIC[0, 0]), 5.0)


def _grad(backend, nblock, args=_GARGS):
    f = partial(_gloss, backend, nblock)
    if backend == "jax":
        return [float(g) for g in jax.jit(jax.grad(f, argnums=(0, 1, 2)))(*args)]
    targs = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in args]
    return [float(g) for g in torch.autograd.grad(f(*targs), targs)]


def _value(backend, nblock, args):
    f = partial(_gloss, backend, nblock)
    if backend == "jax":
        return float(jax.jit(f)(*args))
    with torch.no_grad():
        return float(f(*[torch.tensor(a, dtype=torch.float64) for a in args]))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nblock", [1, None])
def test_leapfrog_grad_vs_central_difference(backend, nblock):
    grad = _grad(backend, nblock)

    def fd(i, h):
        up, dn = list(_GARGS), list(_GARGS)
        up[i] += h
        dn[i] -= h
        return (_value(backend, nblock, up) - _value(backend, nblock, dn)) / (2.0 * h)

    for i in range(3):
        h = 3e-4
        # Richardson: removes the central difference's O(h^2) truncation
        rich = (4.0 * fd(i, h / 2) - fd(i, h)) / 3.0
        numpy.testing.assert_allclose(grad[i], rich, rtol=1e-9)
        assert abs(grad[i] - fd(i, h)) > abs(grad[i] - rich)


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_grad_nblock_independent(backend):
    numpy.testing.assert_allclose(_grad(backend, 1), _grad(backend, None), rtol=1e-12)


@needs_torch
def test_leapfrog_torch_func_and_no_grad():
    # torch.func transforms reject checkpointing, so the steps are stored there:
    # the same gradient as autograd's checkpointed one (not w.r.t. the times,
    # which the Orbit route reads concretely)
    g = torch.func.grad(partial(_gloss, "torch", None), argnums=(0, 1))(
        *[torch.tensor(a, dtype=torch.float64) for a in _GARGS[:2]], _GARGS[2]
    )
    numpy.testing.assert_allclose(
        [float(x) for x in g], _grad("torch", None)[:2], rtol=1e-12
    )
    # without autograd recording nothing is checkpointed: the same values
    ic = _arr("torch", _ics(3))
    ts = numpy.linspace(0.0, 2.0, 5)
    with torch.no_grad():
        y0 = integrate_orbit_leapfrog(_mnh(), ic, ts, nsteps=200)
    y1 = integrate_orbit_leapfrog(_mnh(), ic.requires_grad_(), ts, nsteps=200)
    assert y1.requires_grad and not y0.requires_grad
    numpy.testing.assert_array_equal(as_numpy(y0), as_numpy(y1))


@needs_torch
def test_leapfrog_torch_grad_recomputes_at_the_forward_default_dtype():
    # Orbit.integrate runs galpy's internals at a float64 default dtype, but
    # backward (outside it) recomputes the checkpointed steps: MWPotential2014's
    # bulge makes default-dtype tensors, which then came out float32 and made
    # torch reject the recomputation
    ts = numpy.array([0.0, 1.0])
    ic0 = _ics(5)

    def f(x):
        o = _integrate(
            "torch", x, ts, MWPotential2014, inbackend_kwargs={"nsteps": 100}
        )
        return torch.mean(o.getOrbit()[:, -1, 0])

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # torch's default (other tests set it)
    try:
        x = torch.tensor(ic0, requires_grad=True)
        (g,) = torch.autograd.grad(f(x), x)
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(prev)
    h = 1e-5
    for i, j in ((0, 0), (3, 2)):
        e = numpy.zeros_like(ic0)
        e[i, j] = h
        with torch.no_grad():
            fd = (f(torch.tensor(ic0 + e)) - f(torch.tensor(ic0 - e))) / (2.0 * h)
        numpy.testing.assert_allclose(float(g[i, j]), float(fd), rtol=1e-7)


# --- 4. memory ---------------------------------------------------------------------
def _reverse_temp_bytes(nsteps, nblock):
    pot = HernquistPotential(amp=1.0, a=0.5)
    ic = jnp.asarray(_ics(100))

    def loss(ic):
        ys = integrate_orbit_leapfrog(
            pot, ic, numpy.array([0.0, 5.0]), nsteps=nsteps, nblock=nblock
        )
        return jnp.sum(ys[-1] ** 2)

    compiled = jax.jit(jax.grad(loss)).lower(ic).compile()
    return compiled.memory_analysis().temp_size_in_bytes


@needs_jax
def test_leapfrog_reverse_memory_scaling():
    # default nblock = sqrt(nsteps) for these: memory a + b sqrt(nsteps), so
    # quadrupling nsteps doubles the increment; one block: a + b nsteps
    for nblock, expected in ((None, 2.0), (1, 4.0)):
        m = [_reverse_temp_bytes(n, nblock) for n in (400, 1600, 6400)]
        ratio = (m[2] - m[1]) / (m[1] - m[0])
        assert abs(ratio - expected) < 0.05 * expected, (nblock, m, ratio)
    # and sqrt(nsteps) blocks need far less than one at 6400 steps
    assert _reverse_temp_bytes(6400, None) < _reverse_temp_bytes(6400, 1) / 10


def _torch_peak_bytes(nsteps, nblock):
    """Peak bytes held by the tensors that the forward and backward passes
    allocate (torch has no CPU allocator statistics): each op's new storages
    are tracked until freed, the newest after every op (temporaries die
    quickly), all of them every 256 ops"""
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class Peak(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.live, self.young, self.cur, self.peak, self.nops = {}, [], 0, 0, 0

        def _drop(self, keys):
            for k in keys:
                if k in self.live and self.live[k][0].expired():
                    self.cur -= self.live.pop(k)[1]

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            self._drop(self.young)
            self.young = [k for k in self.young if k in self.live][-32:]
            for t in tree_leaves(out):
                if isinstance(t, torch.Tensor):
                    s = t.untyped_storage()
                    if s.data_ptr() not in self.live:
                        self.live[s.data_ptr()] = (StorageWeakRef(s), s.nbytes())
                        self.cur += s.nbytes()
                        self.young.append(s.data_ptr())
            self.nops += 1
            if self.nops % 256 == 0:
                self._drop(list(self.live))
            self.peak = max(self.peak, self.cur)
            return out

    pot = HernquistPotential(amp=1.0, a=0.5)
    ic = torch.tensor(_ics(100), requires_grad=True)
    with Peak() as m:
        ys = integrate_orbit_leapfrog(
            pot, ic, numpy.array([0.0, 5.0]), nsteps=nsteps, nblock=nblock
        )
        (ys[-1] ** 2).sum().backward()
    return m.peak


@needs_torch
def test_leapfrog_torch_reverse_memory_scaling():
    # as for jax: a + b sqrt(nsteps) by default, a + b nsteps for one block
    for nblock, expected in ((None, 2.0), (1, 4.0)):
        m = [_torch_peak_bytes(n, nblock) for n in (25, 100, 400)]
        ratio = (m[2] - m[1]) / (m[1] - m[0])
        assert abs(ratio - expected) < 0.05 * expected, (nblock, m, ratio)
    assert _torch_peak_bytes(400, None) < _torch_peak_bytes(400, 1) / 3


def test_default_nblock():
    assert _default_nblock(8000, 1) == 80
    assert _default_nblock(6400, 1) == 80
    assert _default_nblock(7919, 1) == 1  # prime: one block
    assert _default_nblock(400, 10) == 20  # multiple of the 10 intervals
    assert _default_nblock(400, 4) == 20
    assert _default_nblock(8000, 1000) == 1000  # many outputs: one per interval
    assert _default_nblock(10000, 1000) == 1000
    assert _default_nblock(1200, 3) == 30


# --- 5. float32 ---------------------------------------------------------------------
def _orbit_route_float32(backend):
    # through Orbit, with float32 times too
    pot = _mnh()
    ts = numpy.linspace(0.0, 1.0, 3)
    o = _integrate(
        backend,
        _arr(backend, _ics(2), "float32"),
        _arr(backend, ts, "float32"),
        pot,
        inbackend_kwargs={"nsteps": 10},
    )
    f32 = jnp.float32 if backend == "jax" else torch.float32
    assert o.getOrbit().dtype == f32
    assert o.x(1.0, use_physical=False).dtype == f32
    assert o.E(ts, use_physical=False).dtype == f32


@needs_jax
def test_leapfrog_float32_end_to_end():
    pot = _mnh()
    ts = numpy.linspace(0.0, 1.0, 3)

    def f(ic):
        return integrate_orbit_leapfrog(pot, ic, ts, nsteps=10)

    for dtype, has_f64 in ((jnp.float32, False), (jnp.float64, True)):
        ic = jnp.asarray(_ics(2), dtype=dtype)
        jaxpr = str(jax.make_jaxpr(f)(ic))
        assert ("f64" in jaxpr) is has_f64, dtype
        assert f(ic).dtype == dtype
    _orbit_route_float32("jax")


@needs_torch
def test_leapfrog_torch_float32_end_to_end():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    class F64(TorchDispatchMode):  # the ops with a float64 output
        def __init__(self):
            super().__init__()
            self.ops = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if any(
                torch.is_tensor(t) and t.dtype == torch.float64
                for t in tree_leaves(out)
            ):
                self.ops.add(str(func))
            return out

    # torch.asarray(numpy float64 amp, dtype=float32) lifts the scalar in
    # float64 before casting it: no float64 arithmetic
    lift = {"aten.lift_fresh.default", "aten.clone.default"}

    pot = _mnh()
    ts = numpy.linspace(0.0, 1.0, 3)
    for dtype in (torch.float32, torch.float64):
        ic = torch.tensor(_ics(2), dtype=dtype, requires_grad=True)
        with F64() as d:
            y = integrate_orbit_leapfrog(pot, ic, ts, nsteps=10)
            y.sum().backward()
        assert y.dtype == dtype and ic.grad.dtype == dtype
        assert (d.ops <= lift) is (dtype == torch.float32), d.ops
    _orbit_route_float32("torch")


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_float32_matches_float64(backend):
    pot = _mnh()
    ic = _ics(300, seed=2)
    ts = numpy.array([0.0, 20.0])
    oc = Orbit(ic)
    oc.integrate(ts, pot, method="dop853_c")
    xc = _cart(oc.getOrbit()[:, -1])[:, :3]

    def final(nsteps, dtype):
        y = integrate_orbit_leapfrog(pot, _arr(backend, ic, dtype), ts, nsteps=nsteps)
        assert str(y.dtype).endswith(dtype)
        return _cart(y[-1])[:, :3]

    # eager torch: a shorter integration for the round-off floor
    for nsteps in (400, 6400 if backend == "jax" else 1600):
        x64, x32 = final(nsteps, "float64"), final(nsteps, "float32")
        trunc = numpy.median(numpy.max(numpy.fabs(x64 - xc), axis=1))
        diff = numpy.max(numpy.fabs(x32 - x64), axis=1)
        # round-off floor: random-walk accumulation of float32 rounding
        assert numpy.median(diff) < 1e-6 * math.sqrt(nsteps)
        assert numpy.percentile(diff, 90) < 3e-6 * math.sqrt(nsteps)
        if nsteps == 400:  # truncation-dominated: float32 is as accurate
            assert numpy.median(diff) < 0.02 * trunc
            e32 = numpy.median(numpy.max(numpy.fabs(x32 - xc), axis=1))
            assert abs(e32 / trunc - 1.0) < 0.01


# --- 6. Orbit route, traced times, continuation, errors -----------------------------
@needs_jax
def test_leapfrog_orbit_route_overhead():
    pot = _mnh()
    ts = numpy.array([0.0, 5.0])
    ic = jnp.asarray(_ics(10))

    def bare(ic):
        return integrate_orbit_leapfrog(pot, ic, ts, nsteps=100)[-1]

    def orbit(ic):
        return _integrate(
            "jax", ic, ts, pot, inbackend_kwargs={"nsteps": 100}
        ).getOrbit()[:, -1]

    def neqns(f):
        return len(str(jax.make_jaxpr(f)(ic)).splitlines())

    assert neqns(orbit) <= neqns(bare) + 10
    numpy.testing.assert_array_equal(jax.jit(orbit)(ic), jax.jit(bare)(ic))


@needs_jax
def test_leapfrog_traced_times():
    pot = _mnh()
    ic = jnp.asarray(_ics(3))

    def final(t1):
        o = _integrate(
            "jax",
            ic,
            jnp.stack([jnp.zeros_like(t1), t1]),
            pot,
            inbackend_kwargs={"nsteps": 500},
        )
        return o.getOrbit()[:, -1]

    numpy.testing.assert_allclose(
        jax.jit(final)(5.0),
        integrate_orbit_leapfrog(pot, ic, numpy.array([0.0, 5.0]), nsteps=500)[-1],
        rtol=0.0,
        atol=1e-13,
    )
    with pytest.raises(ValueError, match="dt needs concrete integration times"):
        jax.jit(
            lambda t1: integrate_orbit_leapfrog(
                pot, ic, jnp.stack([jnp.zeros_like(t1), t1]), dt=0.01
            )
        )(5.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_dt(backend):
    # dt= is the step: the same as nsteps
    pot = _mnh()
    ic = _arr(backend, _ics(3))
    o = _integrate(backend, ic, numpy.linspace(0.0, 5.0, 6), pot, dt=0.01)
    numpy.testing.assert_allclose(
        as_numpy(o.getOrbit()[:, -1]),
        as_numpy(
            integrate_orbit_leapfrog(pot, ic, numpy.array([0.0, 5.0]), nsteps=500)[-1]
        ),
        rtol=0.0,
        atol=1e-13,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_single_orbit(backend):
    pot = _mnh()
    ic = _ics(3)
    ts = numpy.linspace(0.0, 2.0, 5)
    kw = {"inbackend_kwargs": {"nsteps": 200}}
    ob = _integrate(backend, _arr(backend, ic), ts, pot, **kw)
    o = _integrate(backend, _arr(backend, ic[1]), ts, pot, **kw)
    assert tuple(o.getOrbit().shape) == (5, 6)
    numpy.testing.assert_allclose(
        _cart(o.getOrbit()), _cart(ob.getOrbit()[1]), rtol=0.0, atol=1e-14
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_continuation(backend):
    pot = _mnh()
    ic = _arr(backend, _ics(3))
    o = Orbit(ic)
    kw = {"method": f"leapfrog_{backend}", "inbackend_kwargs": {"nsteps": 200}}
    o.integrate(numpy.linspace(0.0, 2.0, 3), pot, **kw)
    o.integrate(numpy.linspace(2.0, 4.0, 3), pot, **kw)
    o1 = _integrate(backend, ic, numpy.linspace(0.0, 4.0, 5), pot, dt=0.01)
    numpy.testing.assert_allclose(
        as_numpy(o.getOrbit()), as_numpy(o1.getOrbit()), rtol=0.0, atol=1e-13
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_velocity_dependent_raises(backend):
    cdf = ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.1, dens=_mnh()[1])
    rot = NonInertialFrameForce(Omega=0.5)
    for force in (cdf, rot):
        with pytest.raises(ValueError, match="velocity-independent forces"):
            _integrate(
                backend,
                _arr(backend, _ics(2)),
                numpy.array([0.0, 1.0]),
                _mnh() + force,
                inbackend_kwargs={"nsteps": 10},
            )
    # planar versions of 3D forces are judged by the wrapped force
    assert _velocity_dependent(toPlanarPotential(cdf))
    assert not _velocity_dependent(toPlanarPotential(_frame_force()))
    assert not _velocity_dependent([_mnh(), _frame_force()])


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_input_errors(backend):
    pot = _mnh()
    ic = _arr(backend, _ics(2))
    ts = numpy.linspace(0.0, 1.0, 3)

    def lf(ts=ts, ic=ic, **kw):
        _integrate(backend, ic, ts, pot, **kw)

    with pytest.raises(ValueError, match="requires a jax/torch initial condition"):
        lf(ic=_ics(2), inbackend_kwargs={"nsteps": 10})
    with pytest.raises(ValueError, match="set their number with"):
        lf()
    with pytest.raises(ValueError, match="takes inbackend_kwargs 'nsteps'"):
        lf(inbackend_kwargs={"nsteps": 10, "solver": "dopri5"})
    with pytest.raises(ValueError, match="multiple of the number of output"):
        lf(inbackend_kwargs={"nsteps": 11})
    with pytest.raises(TypeError, match="nsteps must be an integer"):
        lf(inbackend_kwargs={"nsteps": 10.0})
    with pytest.raises(TypeError, match="nsteps must be an integer"):
        lf(inbackend_kwargs={"nsteps": True})
    with pytest.raises(ValueError, match="equally-spaced output times"):
        lf(ts=numpy.array([0.0, 0.4, 1.0]), inbackend_kwargs={"nsteps": 10})
    for nblock in (5, 3, 0, True, 2.0):  # non-divisor, misaligned, ...
        with pytest.raises(ValueError, match="nblock="):
            lf(inbackend_kwargs={"nsteps": 12, "nblock": nblock})
    with pytest.raises(ValueError, match="either dt or nsteps"):
        lf(dt=0.1, inbackend_kwargs={"nsteps": 10})
    with pytest.raises(ValueError, match="does not divide the integration"):
        lf(dt=0.03)
    with pytest.raises(ValueError, match="at least two times"):
        integrate_orbit_leapfrog(pot, ic, numpy.array([1.0]), nsteps=10)
    with pytest.raises(NotImplementedError, match="one time array shared"):
        lf(ts=numpy.tile(ts, (2, 1)), inbackend_kwargs={"nsteps": 10})
    with pytest.raises(NotImplementedError, match="jax or torch initial condition"):
        integrate_orbit_leapfrog(pot, _ics(2), ts, nsteps=10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_leapfrog_method_needs_its_backend(backend):
    other = {"jax": "torch", "torch": "jax"}[backend]
    if other not in BACKENDS:
        pytest.skip(f"{other} not installed")
    with pytest.raises(ValueError, match=f"'leapfrog_{other}' requires a {other}"):
        Orbit(_arr(backend, _ics(2))).integrate(
            numpy.array([0.0, 1.0]),
            _mnh(),
            method=f"leapfrog_{other}",
            inbackend_kwargs={"nsteps": 10},
        )
