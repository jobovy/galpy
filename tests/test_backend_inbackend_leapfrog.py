###############################################################################
# test_backend_inbackend_leapfrog.py: the in-backend kick-drift-kick leapfrog
# (Orbit.integrate(method="leapfrog_inbackend"),
# galpy.backend._reference.integrate_orbit_leapfrog; jax).
#
# Proves:
#   1. final states converge to dop853_c at second order in the step, for a few
#      thousand orbits in MWPotential2014 + a MovingObjectPotential + a
#      NonInertialFrameForce, and for every phase-space dimension,
#   2. energy is conserved to second order with no secular drift over 1e4 steps
#      (a non-symplectic second-order method drifts on the same orbits),
#   3. reverse-mode gradients w.r.t. a potential parameter, an initial
#      condition and the final time match central differences of the same
#      leapfrog, for nblock=1 and the default,
#   4. the reverse pass's compiled temporary memory grows as sqrt(nsteps) by
#      default (as nsteps for nblock=1),
#   5. float32 inputs integrate in float32 end to end and agree with float64
#      to the truncation error down to the round-off floor,
#   6. the Orbit route adds (almost) nothing to the traced computation, and the
#      error paths are clear.
###############################################################################
import math

import numpy
import pytest

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

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from galpy.backend import get_namespace  # noqa: E402
from galpy.backend._reference import integrate_orbit_leapfrog  # noqa: E402
from galpy.backend._reference.inbackend_ode import (  # noqa: E402
    _default_nblock,
    _velocity_dependent,
)


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
    y = numpy.asarray(y, dtype=float)
    R, vR, vT, z, vz, phi = (y[..., i] for i in range(6))
    c, s = numpy.cos(phi), numpy.sin(phi)
    return numpy.stack([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz], -1)


def _leapfrog_final(pot, ts, nsteps, nblock=None):
    """jitted Orbit route: (N, 6) initial conditions -> (N, 6) final states"""

    def f(ic):
        o = Orbit(ic)
        o.integrate(
            ts,
            pot,
            method="leapfrog_inbackend",
            inbackend_kwargs={"nsteps": nsteps, "nblock": nblock},
        )
        return o.getOrbit()[:, -1]

    return jax.jit(f)


# --- 1. second-order convergence to dop853_c ---------------------------------
def test_leapfrog_matches_dop853c_second_order():
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
        x = _cart(_leapfrog_final(pot, ts, nsteps)(jnp.asarray(ic)))
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


@pytest.mark.parametrize("phasedim", [2, 3, 4, 5, 6])
def test_leapfrog_phasedims_second_order(phasedim):
    pot = IsothermalDiskPotential(amp=1.0, sigma=0.3) if phasedim == 2 else _mnh()
    ic = {
        2: numpy.array([[0.1, 0.2], [0.3, -0.1]]),
        3: _ics(3)[:, :3],
        4: _ics(3)[:, [0, 1, 2, 5]],
        5: _ics(3)[:, :5],
        6: _ics(3),
    }[phasedim]
    ts = numpy.linspace(0.0, 6.0, 4)
    oc = Orbit(ic)
    oc.integrate(ts, pot, method="dop853_c")
    yc = numpy.asarray(oc.getOrbit())

    def comparable(y):  # phi -> (x, y) where present
        y = numpy.asarray(y, dtype=float)
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
        o = Orbit(jnp.asarray(ic))
        o.integrate(
            ts,
            pot,
            method="leapfrog_inbackend",
            inbackend_kwargs={"nsteps": nsteps},
        )
        y = o.getOrbit()
        assert y.shape == yc.shape
        errs.append(numpy.max(numpy.fabs(comparable(y) - comparable(yc))[:, 1:]))
    assert errs[1] < 1e-4
    assert 3.9 < errs[0] / errs[1] < 4.1


@pytest.mark.parametrize("nblock", [None, 1, 2, 10, 40])
def test_leapfrog_output_grid(nblock):
    # 10 output intervals of 40 steps: blocks holding several intervals (2),
    # exactly one (10) and several blocks per interval (40) give the same states
    # as integrating to each output time separately with the same step
    pot = _mnh()
    ic = jnp.asarray(_ics(4))
    ts = numpy.linspace(0.0, 4.0, 11)
    o = Orbit(ic)
    o.integrate(
        ts,
        pot,
        method="leapfrog_inbackend",
        inbackend_kwargs={"nsteps": 400, "nblock": nblock},
    )
    y = numpy.asarray(o.getOrbit())
    numpy.testing.assert_allclose(_cart(y[:, 0]), _cart(ic), rtol=0.0, atol=1e-15)
    for i in (1, 5, 10):
        yi = integrate_orbit_leapfrog(pot, ic, ts[[0, i]], nsteps=40 * i)[-1]
        numpy.testing.assert_allclose(_cart(y[:, i]), _cart(yi), rtol=0.0, atol=1e-13)
    # accessors at the integration times index the stored states
    numpy.testing.assert_array_equal(numpy.asarray(o.R(ts)), y[..., 0])


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


def test_leapfrog_energy_second_order_no_drift():
    dEmax, trend = _energy_errors("leapfrog_inbackend", 10000)
    dEmax2, trend2 = _energy_errors("leapfrog_inbackend", 20000)
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


def _gloss(theta, R0, t1, nblock):
    pot = _mnh(amp_halo=theta) + _frame_force()
    ic = jnp.asarray(_GIC).at[0, 0].set(R0)
    o = Orbit(ic)
    o.integrate(
        jnp.stack([jnp.zeros_like(t1), t1]),
        pot,
        method="leapfrog_inbackend",
        inbackend_kwargs={"nsteps": 400, "nblock": nblock},
    )
    x = jnp.asarray(_cart_jnp(o.getOrbit()[:, -1]))[:, :3]
    return jnp.sum(jnp.asarray(_GW) * x)


def _cart_jnp(y):
    R, vR, vT, z, vz, phi = (y[..., i] for i in range(6))
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz], -1)


@pytest.mark.parametrize("nblock", [1, None])
def test_leapfrog_grad_vs_central_difference(nblock):
    args = (4.0, float(_GIC[0, 0]), 5.0)
    grad = jax.jit(jax.grad(_gloss, argnums=(0, 1, 2)), static_argnums=3)(*args, nblock)
    f = jax.jit(_gloss, static_argnums=3)

    def fd(i, h):
        up, dn = list(args), list(args)
        up[i] += h
        dn[i] -= h
        return (f(*up, nblock) - f(*dn, nblock)) / (2.0 * h)

    for i in range(3):
        h = 3e-4
        # Richardson: removes the central difference's O(h^2) truncation
        rich = (4.0 * fd(i, h / 2) - fd(i, h)) / 3.0
        numpy.testing.assert_allclose(float(grad[i]), float(rich), rtol=1e-9)
        assert abs(float(grad[i]) - float(fd(i, h))) > abs(float(grad[i]) - float(rich))


def test_leapfrog_grad_nblock_independent():
    args = (4.0, float(_GIC[0, 0]), 5.0)
    g = jax.jit(jax.grad(_gloss, argnums=(0, 1, 2)), static_argnums=3)
    g1, gd = g(*args, 1), g(*args, None)
    for a, b in zip(g1, gd):
        numpy.testing.assert_allclose(float(a), float(b), rtol=1e-12)


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


def test_leapfrog_reverse_memory_scaling():
    # default nblock = sqrt(nsteps) for these: memory a + b sqrt(nsteps), so
    # quadrupling nsteps doubles the increment; one block: a + b nsteps
    for nblock, expected in ((None, 2.0), (1, 4.0)):
        m = [_reverse_temp_bytes(n, nblock) for n in (400, 1600, 6400)]
        ratio = (m[2] - m[1]) / (m[1] - m[0])
        assert abs(ratio - expected) < 0.05 * expected, (nblock, m, ratio)
    # and sqrt(nsteps) blocks need far less than one at 6400 steps
    assert _reverse_temp_bytes(6400, None) < _reverse_temp_bytes(6400, 1) / 10


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
    # the Orbit route, with float32 times too
    ic = jnp.asarray(_ics(2), dtype=jnp.float32)
    o = Orbit(ic)
    o.integrate(
        jnp.asarray(ts, dtype=jnp.float32),
        pot,
        method="leapfrog_inbackend",
        inbackend_kwargs={"nsteps": 10},
    )
    assert o.getOrbit().dtype == jnp.float32
    assert o.x(1.0, use_physical=False).dtype == jnp.float32
    assert o.E(ts, use_physical=False).dtype == jnp.float32


def test_leapfrog_float32_matches_float64():
    pot = _mnh()
    ic = _ics(300, seed=2)
    ts = numpy.array([0.0, 20.0])
    oc = Orbit(ic)
    oc.integrate(ts, pot, method="dop853_c")
    xc = _cart(oc.getOrbit()[:, -1])[:, :3]

    def final(nsteps, dtype):
        y = integrate_orbit_leapfrog(
            pot, jnp.asarray(ic, dtype=dtype), ts, nsteps=nsteps
        )[-1]
        assert y.dtype == dtype
        return _cart(y)[:, :3]

    for nsteps in (400, 6400):
        x64, x32 = final(nsteps, jnp.float64), final(nsteps, jnp.float32)
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
def test_leapfrog_orbit_route_overhead():
    pot = _mnh()
    ts = numpy.array([0.0, 5.0])
    ic = jnp.asarray(_ics(10))

    def bare(ic):
        return integrate_orbit_leapfrog(pot, ic, ts, nsteps=100)[-1]

    def orbit(ic):
        o = Orbit(ic)
        o.integrate(
            ts, pot, method="leapfrog_inbackend", inbackend_kwargs={"nsteps": 100}
        )
        return o.getOrbit()[:, -1]

    def neqns(f):
        return len(str(jax.make_jaxpr(f)(ic)).splitlines())

    assert neqns(orbit) <= neqns(bare) + 10
    numpy.testing.assert_array_equal(jax.jit(orbit)(ic), jax.jit(bare)(ic))


def test_leapfrog_traced_times_and_dt():
    pot = _mnh()
    ic = jnp.asarray(_ics(3))

    def final(t1):
        o = Orbit(ic)
        o.integrate(
            jnp.stack([jnp.zeros_like(t1), t1]),
            pot,
            method="leapfrog_inbackend",
            inbackend_kwargs={"nsteps": 500},
        )
        return o.getOrbit()[:, -1]

    yt = jax.jit(final)(5.0)
    numpy.testing.assert_allclose(
        yt,
        integrate_orbit_leapfrog(pot, ic, numpy.array([0.0, 5.0]), nsteps=500)[-1],
        rtol=0.0,
        atol=1e-13,
    )
    # dt= is the step: the same as nsteps
    o = Orbit(ic)
    o.integrate(numpy.linspace(0.0, 5.0, 6), pot, method="leapfrog_inbackend", dt=0.01)
    numpy.testing.assert_allclose(o.getOrbit()[:, -1], yt, rtol=0.0, atol=1e-13)
    with pytest.raises(ValueError, match="dt needs concrete integration times"):
        jax.jit(
            lambda t1: integrate_orbit_leapfrog(
                pot, ic, jnp.stack([jnp.zeros_like(t1), t1]), dt=0.01
            )
        )(5.0)


def test_leapfrog_single_orbit():
    pot = _mnh()
    ic = _ics(3)
    ts = numpy.linspace(0.0, 2.0, 5)
    kw = {"method": "leapfrog_inbackend", "inbackend_kwargs": {"nsteps": 200}}
    ob = Orbit(jnp.asarray(ic))
    ob.integrate(ts, pot, **kw)
    o = Orbit(jnp.asarray(ic[1]))
    o.integrate(ts, pot, **kw)
    assert o.getOrbit().shape == (5, 6)
    numpy.testing.assert_allclose(
        _cart(o.getOrbit()), _cart(ob.getOrbit()[1]), rtol=0.0, atol=1e-14
    )


def test_leapfrog_continuation():
    pot = _mnh()
    ic = jnp.asarray(_ics(3))
    o = Orbit(ic)
    kw = {"method": "leapfrog_inbackend", "inbackend_kwargs": {"nsteps": 200}}
    o.integrate(numpy.linspace(0.0, 2.0, 3), pot, **kw)
    o.integrate(numpy.linspace(2.0, 4.0, 3), pot, **kw)
    o1 = Orbit(ic)
    o1.integrate(numpy.linspace(0.0, 4.0, 5), pot, method=kw["method"], dt=0.01)
    numpy.testing.assert_allclose(o.getOrbit(), o1.getOrbit(), rtol=0.0, atol=1e-13)


def test_leapfrog_velocity_dependent_raises():
    cdf = ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.1, dens=_mnh()[1])
    rot = NonInertialFrameForce(Omega=0.5)
    o = Orbit(jnp.asarray(_ics(2)))
    for force in (cdf, rot):
        with pytest.raises(ValueError, match="velocity-independent forces"):
            o.integrate(
                numpy.array([0.0, 1.0]),
                _mnh() + force,
                method="leapfrog_inbackend",
                inbackend_kwargs={"nsteps": 10},
            )
    # planar versions of 3D forces are judged by the wrapped force
    assert _velocity_dependent(toPlanarPotential(cdf))
    assert not _velocity_dependent(toPlanarPotential(_frame_force()))
    assert not _velocity_dependent([_mnh(), _frame_force()])


def test_leapfrog_input_errors():
    pot = _mnh()
    ic = jnp.asarray(_ics(2))
    ts = numpy.linspace(0.0, 1.0, 3)

    def lf(ts=ts, ic=ic, **kw):
        o = Orbit(ic)
        o.integrate(ts, pot, method="leapfrog_inbackend", **kw)

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


def test_leapfrog_torch_not_implemented():
    torch = pytest.importorskip("torch")
    o = Orbit(torch.tensor(_ics(2), dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="requires a jax initial condition"):
        o.integrate(
            numpy.array([0.0, 1.0]),
            _mnh(),
            method="leapfrog_inbackend",
            inbackend_kwargs={"nsteps": 10},
        )
