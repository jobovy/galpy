###############################################################################
# test_backend_inbackend_symplec.py: the in-backend fourth- and sixth-order
# symplectic integrators (Orbit.integrate(method="symplec4_jax"/"symplec4_torch"/
# "symplec6_jax"/"symplec6_torch"), galpy.backend._reference.integrate_orbit_symplectic
# with order=4/6), the higher-order siblings of the in-backend leapfrog
# (test_backend_inbackend_leapfrog.py).
#
# Proves, on jax and on torch:
#   1. the coefficient table equals the one in galpy/util/bovy_symplecticode.c,
#      and a batch integrated with the same constant step agrees with the C
#      integrators symplec4_c/symplec6_c to round-off,
#   2. the final states converge to dop853_c at fourth and sixth order in the
#      step, for a batch of orbits in MWPotential2014 and for every phase-space
#      dimension (torch and jax agree to round-off),
#   3. the energy error oscillates without a secular drift (jax),
#   4. reverse-mode gradients w.r.t. a potential parameter, an initial
#      condition and the final time match central differences of the same
#      integrator, for nblock=1 and the default, under jax.jit as well,
#   5. the error paths are clear (a non-conservative force, the wrong backend,
#      an invalid order).
###############################################################################
import os
import re

import numpy
import pytest

from galpy.backend import as_numpy, get_namespace
from galpy.backend._reference import integrate_orbit_symplectic
from galpy.backend._reference.symplectic_coeffs import (
    COEFFICIENTS,
    FORCE_EVALUATIONS,
    SYMPLEC4_CS,
    SYMPLEC4_DS,
    SYMPLEC6_CS,
    SYMPLEC6_DS,
)
from galpy.orbit import Orbit
from galpy.potential import (
    ChandrasekharDynamicalFrictionForce,
    HernquistPotential,
    IsothermalDiskPotential,
    MiyamotoNagaiPotential,
    MWPotential2014,
    NFWPotential,
    NonInertialFrameForce,
    PlummerPotential,
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
ORDERS = [4, 6]


def _arr(backend, x, dtype="float64"):
    if backend == "jax":
        return jnp.asarray(x, dtype=getattr(jnp, dtype))
    return torch.tensor(numpy.asarray(x), dtype=getattr(torch, dtype))


def _integrate(backend, order, ic, ts, pot, **kw):
    o = Orbit(ic)
    o.integrate(ts, pot, method=f"symplec{order}_{backend}", **kw)
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
    # a time-dependent frame acceleration (no rotation: derives from a potential)
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
    """Orbit-order (..., phasedim) -> coordinates in which phi wrapping is moot:
    Cartesian for the 6-D and 4-D (planar with phi) orbits, as given otherwise."""
    y = numpy.asarray(as_numpy(y), dtype=float)
    if y.shape[-1] == 6:
        R, vR, vT, z, vz, phi = (y[..., i] for i in range(6))
        c, s = numpy.cos(phi), numpy.sin(phi)
        return numpy.stack([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz], -1)
    if y.shape[-1] == 4:
        R, vR, vT, phi = (y[..., i] for i in range(4))
        c, s = numpy.cos(phi), numpy.sin(phi)
        return numpy.stack([R * c, R * s, vR * c - vT * s, vR * s + vT * c], -1)
    return y


_TS = numpy.linspace(0.0, 56.2, 11)  # 2 Gyr in internal units (ro=8, vo=220)


# --- 1. the C integrators: same coefficients, same orbits ----------------------
def test_coefficients_match_the_c_source():
    # the tables are copied from bovy_symplecticode.c; keep the two in step
    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "galpy",
        "util",
        "bovy_symplecticode.c",
    )
    if not os.path.exists(src):  # an installed galpy without its C sources
        pytest.skip("C source not available")
    with open(src) as fh:
        text = fh.read()

    def table(name):
        m = re.search(rf"{name}\[\d+\]=\s*\{{([^}}]*)\}}", text)
        return [float(x) for x in m.group(1).replace("\n", " ").split(",")]

    for name, ours in (
        ("symplec4_cs", SYMPLEC4_CS),
        ("symplec4_ds", SYMPLEC4_DS),
        ("symplec6_cs", SYMPLEC6_CS),
        ("symplec6_ds", SYMPLEC6_DS),
    ):
        numpy.testing.assert_array_equal(table(name), ours)
    for order in ORDERS:
        cs, ds = COEFFICIENTS[order]
        assert len(cs) == len(ds) + 1 == FORCE_EVALUATIONS[order] + 1
        # the drifts and the kicks each span one step
        numpy.testing.assert_allclose(sum(cs), 1.0, rtol=1e-14)
        numpy.testing.assert_allclose(sum(ds), 1.0, rtol=1e-14)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_the_c_integrator_at_the_same_step(backend, order):
    # the same drift/kick composition as symplec4_c/symplec6_c: with the same
    # constant step the two differ only by the order of floating-point
    # operations (the C code merges the last drift of a step with the first of
    # the next), i.e., to round-off
    ic = _ics(16)
    ts = numpy.linspace(0.0, 20.0, 41)
    nsteps = 400  # dt = 0.05, an integer divisor of the output interval 0.5
    oc = Orbit(ic)
    oc.integrate(ts, MWPotential2014, method=f"symplec{order}_c", dt=0.05)
    o = _integrate(
        backend,
        order,
        _arr(backend, ic),
        ts,
        MWPotential2014,
        inbackend_kwargs={"nsteps": nsteps},
    )
    numpy.testing.assert_allclose(
        _cart(o.getOrbit()), _cart(oc.getOrbit()), rtol=0.0, atol=1e-11
    )


# --- 2. convergence to dop853_c -----------------------------------------------
def _dop853(ic, ts, pot):
    o = Orbit(ic)
    o.integrate(ts, pot, method="dop853_c", rtol=1e-13, atol=1e-13)
    return o.getOrbit()


def _errors(backend, order, ic, ts, pot, nsteps_list):
    ref = _cart(_dop853(ic, ts, pot)[:, -1])
    out = []
    for n in nsteps_list:
        o = _integrate(
            backend, order, _arr(backend, ic), ts, pot, inbackend_kwargs={"nsteps": n}
        )
        out.append(numpy.max(numpy.abs(_cart(o.getOrbit()[:, -1]) - ref)))
    return numpy.array(out)


# where each method is in its asymptotic regime for these orbits (error well
# above round-off, well below the orbit scale), with the expected ratio per
# halving of the step: 2^4 = 16 and 2^6 = 64
_CONV = {
    4: ([1600, 3200, 6400], 16.0, (13.0, 19.0)),
    6: ([1600, 3200], 64.0, (40.0, 110.0)),
}


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_converges_to_dop853c_at_its_order(backend, order):
    nsteps, _, (lo, hi) = _CONV[order]
    errs = _errors(backend, order, _ics(20), _TS, MWPotential2014, nsteps)
    ratio = errs[:-1] / errs[1:]
    assert numpy.all((ratio > lo) & (ratio < hi)), (order, errs, ratio)
    assert errs[-1] < {4: 3e-6, 6: 3e-8}[order], errs


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_moving_satellite_and_frame_force(backend, order):
    # a time-dependent potential (a satellite on an orbit) and the acceleration
    # of a non-inertial frame (no rotation: conservative), the leapfrog test's
    # setup: the same convergence over a shorter span
    from galpy.potential import MovingObjectPotential

    oobj = Orbit([4.0, -0.3, 0.6, 1.0, 0.1, 0.5])
    oobj.integrate(numpy.linspace(-1.0, 6.0, 141), MWPotential2014, method="dop853_c")
    mop = MovingObjectPotential(oobj, pot=HernquistPotential(amp=0.05, a=0.1))
    pot = MWPotential2014[0] + MWPotential2014[1] + MWPotential2014[2] + mop
    pot = pot + _frame_force()
    ts = numpy.array([0.0, 5.0])
    nsteps = {4: [100, 200, 400], 6: [100, 200]}[order]
    lo, hi = _CONV[order][2]
    errs = _errors(backend, order, _ics(50, seed=3), ts, pot, nsteps)
    ratio = errs[:-1] / errs[1:]
    assert numpy.all((ratio > lo) & (ratio < hi)), (order, errs, ratio)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("phasedim", [2, 3, 4, 5, 6])
def test_phasedims(backend, order, phasedim):
    ts = numpy.linspace(0.0, 20.0, 11)
    if phasedim == 2:
        pot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
        ic = numpy.array([[0.3, 0.1], [-0.2, 0.3], [0.5, -0.4]])
    else:
        pot = _mnh() if phasedim > 4 else toPlanarPotential(_mnh())
        ic = _ics(3, seed=5)[:, : phasedim if phasedim != 3 else 3]
        if phasedim == 4:
            ic = numpy.column_stack([ic[:, :3], _ics(3, seed=5)[:, 5]])
    nsteps = [1600, 3200]
    errs = _errors(backend, order, ic, ts, pot, nsteps)
    assert errs[-1] < 1e-7, errs
    lo, hi = _CONV[order][2]
    # the round-off floor is close for order 6 on these short integrations:
    # only require the right order while the error is still well above it
    if errs[-1] > 1e-10:
        assert lo < errs[0] / errs[1] < hi, (order, errs)


@needs_jax
@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("order", ORDERS)
def test_torch_matches_jax(order):
    ic = _ics(6, seed=2)
    kw = {"inbackend_kwargs": {"nsteps": 800}}
    yj = _integrate("jax", order, _arr("jax", ic), _TS, _mnh(), **kw).getOrbit()
    yt = _integrate("torch", order, _arr("torch", ic), _TS, _mnh(), **kw).getOrbit()
    numpy.testing.assert_allclose(_cart(yt), _cart(yj), rtol=0.0, atol=1e-12)


# --- 3. energy ---------------------------------------------------------------------
@needs_jax
@pytest.mark.parametrize("order", ORDERS)
def test_energy_no_secular_drift(order):
    # a static potential: the energy error of a symplectic method is a bounded
    # oscillation, so its envelope at the end of a long integration is no larger
    # than at the start (a non-symplectic method's grows linearly), at a step
    # coarse enough that the error is well above round-off
    pot = _mnh()
    ic = jnp.asarray(_ics(3, seed=4))
    ts = numpy.linspace(0.0, 400.0, 401)  # ~40 to 80 orbital periods
    nsteps = {4: 8000, 6: 4000}[order]
    o = _integrate("jax", order, ic, ts, pot, inbackend_kwargs={"nsteps": nsteps})
    E = numpy.asarray(o.E(ts, pot=pot, use_physical=False))
    dE = numpy.fabs(E - E[:, :1]) / numpy.fabs(E[:, :1])
    dEmax = dE.max(axis=1)
    assert numpy.all(dEmax > 1e-12) and numpy.all(dEmax < 1e-5), dEmax
    first = dE[:, :40].max(axis=1)
    last = dE[:, -40:].max(axis=1)
    assert numpy.all(last < 2.0 * first + 1e-14), (first, last)
    # the control: a non-symplectic method of the same order drifts
    diffrax = pytest.importorskip("diffrax")
    oc = Orbit(ic)
    oc.integrate(
        ts,
        pot,
        method="diffrax",
        inbackend_kwargs={"nsteps": nsteps, "solver": diffrax.Heun()},
    )
    Ec = numpy.asarray(oc.E(ts, pot=pot, use_physical=False))
    dEc = numpy.fabs(Ec - Ec[:, :1]) / numpy.fabs(Ec[:, :1])
    assert numpy.all(dEc[:, -40:].max(axis=1) > 5.0 * dEc[:, :40].max(axis=1))


# --- 4. gradients ------------------------------------------------------------------
def _grad_setup():
    ic = numpy.array([1.1, 0.12, 1.05, 0.08, 0.05, 0.3])
    ts = numpy.linspace(0.0, 25.0, 6)
    return ic, ts


def _value(backend, order, nblock, amp, ic, tf, jit=False):
    ic0, ts = _grad_setup()

    def f(amp, ic, tf):
        o = Orbit(ic)
        # (torch.linspace does not carry the gradient of its end point)
        xp = get_namespace(ic)
        tt = tf * (
            xp.linspace(0.0, 1.0, 6)
            if backend == "jax"
            else torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        )
        o.integrate(
            tt,
            _mnh(amp_halo=amp),
            method=f"symplec{order}_{backend}",
            inbackend_kwargs={"nsteps": 500, "nblock": nblock},
        )
        y = o.getOrbit()[-1]
        return y[0] * y[3] + y[1] * y[4]  # a smooth scalar of the final state

    if backend == "jax" and jit:
        f = jax.jit(f)
    return f(amp, ic, tf)


def _richardson(backend, order, nblock, which):
    ic0, ts = _grad_setup()
    x0 = {"amp": 0.35, "ic": ic0[1], "tf": ts[-1]}[which]
    h = {"amp": 1e-3, "ic": 1e-4, "tf": 1e-3}[which]

    def g(x):
        amp, ic, tf = 0.35, ic0.copy(), ts[-1]
        if which == "amp":
            amp = x
        elif which == "ic":
            ic[1] = x
        else:
            tf = x
        return float(
            as_numpy(
                _value(
                    backend,
                    order,
                    nblock,
                    _arr(backend, amp),
                    _arr(backend, ic),
                    _arr(backend, tf),
                )
            )
        )

    d1 = (g(x0 + h) - g(x0 - h)) / (2 * h)
    d2 = (g(x0 + h / 2) - g(x0 - h / 2)) / h
    return (4 * d2 - d1) / 3


@pytest.mark.parametrize("which", ["amp", "ic", "tf"])
@pytest.mark.parametrize("nblock", [1, None])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_grad_vs_central_difference(backend, order, nblock, which):
    ic0, ts = _grad_setup()
    if backend == "jax":
        args = [jnp.asarray(0.35), jnp.asarray(ic0), jnp.asarray(ts[-1])]
        k = ["amp", "ic", "tf"].index(which)
        grad = jax.grad(lambda *a: _value(backend, order, nblock, *a), argnums=k)(*args)
        got = float(grad[1]) if which == "ic" else float(grad)
    else:
        amp = torch.tensor(0.35, dtype=torch.float64, requires_grad=which == "amp")
        ic = torch.tensor(ic0, dtype=torch.float64, requires_grad=which == "ic")
        tf = torch.tensor(ts[-1], dtype=torch.float64, requires_grad=which == "tf")
        _value(backend, order, nblock, amp, ic, tf).backward()
        got = float(
            {
                "amp": amp.grad,
                "ic": ic.grad[1] if ic.grad is not None else None,
                "tf": tf.grad,
            }[which]
        )
    want = _richardson(backend, order, nblock, which)
    numpy.testing.assert_allclose(got, want, rtol=1e-7)


@needs_jax
@pytest.mark.parametrize("order", ORDERS)
def test_grad_under_jit_matches_eager(order):
    ic0, ts = _grad_setup()
    args = [jnp.asarray(0.35), jnp.asarray(ic0), jnp.asarray(ts[-1])]
    g = jax.grad(lambda *a: _value("jax", order, None, *a), argnums=0)(*args)
    gj = jax.grad(lambda *a: _value("jax", order, None, *a, jit=True), argnums=0)(*args)
    numpy.testing.assert_allclose(float(gj), float(g), rtol=1e-12)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("order", ORDERS)
def test_torch_no_grad_matches_grad_enabled(order):
    # without autograd the integrator neither checkpoints nor stores the steps:
    # the same orbit as with autograd enabled
    ic0, ts = _grad_setup()
    pot = _mnh()
    kw = dict(inbackend_kwargs={"nsteps": 500})
    want = _integrate("torch", order, _arr("torch", ic0), ts, pot, **kw).getOrbit()
    with torch.no_grad():
        got = _integrate("torch", order, _arr("torch", ic0), ts, pot, **kw).getOrbit()
    numpy.testing.assert_array_equal(as_numpy(got), as_numpy(want))


# --- 5. error paths -----------------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_nonconservative_force_raises(backend, order):
    pot = _mnh() + ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.05, dens=_mnh())
    with pytest.raises(ValueError, match="derive from a potential"):
        _integrate(
            backend,
            order,
            _arr(backend, _ics(2)),
            _TS,
            pot,
            inbackend_kwargs={"nsteps": 100},
        )


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_wrong_backend_raises(backend, order):
    other = "torch" if backend == "jax" else "jax"
    o = Orbit(_arr(backend, _ics(2)))
    with pytest.raises(ValueError, match=f"requires a {other} initial condition"):
        o.integrate(
            _TS,
            _mnh(),
            method=f"symplec{order}_{other}",
            inbackend_kwargs={"nsteps": 100},
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_order_raises(backend):
    with pytest.raises(ValueError, match="order must be 2, 4 or 6"):
        integrate_orbit_symplectic(
            PlummerPotential(amp=1.0, b=0.5),
            _arr(backend, _ics(1)[0]),
            _TS,
            order=3,
            nsteps=100,
        )
