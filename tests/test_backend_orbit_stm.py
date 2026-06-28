###############################################################################
# test_backend_orbit_stm.py: the fast C state-transition-matrix (STM) orbit
# autodiff path (galpy.backend._{jax,torch}.orbit_stm + _reference.inbackend_stm).
# Forward = galpy's compiled C variational integrator; gradient w.r.t. the
# initial conditions = sum_t M(t)^T cotangent[t]. Validates: forward == the C
# orbit, grad == finite-difference, jacrev == the directly-assembled STM, torch
# gradcheck, and agreement with the independent in-backend ODE path.
###############################################################################
import numpy
import pytest

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

from galpy.potential import (
    DehnenBarPotential,
    MiyamotoNagaiPotential,
    MWPotential2014,
)

_METHODS = ["rk4_c", "rk6_c", "dop853_c"]
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])  # R,vR,vT,z,vz,phi
_TS = numpy.linspace(0.0, 2.0, 9)


def _pots():
    return {
        "MiyamotoNagai": MiyamotoNagaiPotential(normalize=1.0),
        "MWPotential2014": MWPotential2014,
        "DehnenBar": DehnenBarPotential(),  # non-axisymmetric
    }


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _integ(backend, pot, vxvv, ts, method):
    if backend == "jax":
        from galpy.backend._jax.orbit_stm import integrate
    else:
        from galpy.backend._torch.orbit_stm import integrate
    return integrate(pot, vxvv, ts, method=method)


# ---------------------------------------------------------------- forward parity
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", _METHODS)
def test_forward_matches_c(backend, method):
    # the wrapper's forward IS galpy's C integrator -> must match Orbit.integrate
    from galpy.orbit import Orbit

    for name, pot in _pots().items():
        o = Orbit(list(_IC))
        o.integrate(_TS, pot, method=method)
        ref = numpy.array(
            [o.R(_TS), o.vR(_TS), o.vT(_TS), o.z(_TS), o.vz(_TS), o.phi(_TS)]
        ).T
        got = _np(_integ(backend, pot, _arr(backend, _IC), _TS, method))
        # the wrapper's forward is the dxdv variant of the C integrator; for the
        # fixed-step methods its shared 12-D step sequence differs from base-only
        # Orbit.integrate at the integrator level (~1e-8), 1e-12 for dop853_c.
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-6, atol=1e-7, err_msg=f"{name} {method} {backend}"
        )


# ------------------------------------------------------------- grad vs finite-diff
def _fd_grad_final_R(pot, method, eps=1e-6):
    from galpy.orbit import Orbit

    def fR(ic):
        o = Orbit(list(ic))
        o.integrate(_TS, pot, method=method)
        return o.R(_TS[-1])

    base = fR(_IC)
    return numpy.array(
        [(fR(_IC + eps * numpy.eye(6)[j]) - base) / eps for j in range(6)]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_grad_final_R_vs_fd(backend):
    pot = MiyamotoNagaiPotential(normalize=1.0)
    gfd = _fd_grad_final_R(pot, "dop853_c")
    if backend == "jax":
        g = jax.grad(lambda v: _integ("jax", pot, v, _TS, "dop853_c")[-1, 0])(
            jnp.asarray(_IC)
        )
        g = _np(g)
    else:
        v = torch.tensor(_IC, requires_grad=True)
        _integ("torch", pot, v, _TS, "dop853_c")[-1, 0].backward()
        g = _np(v.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-4, atol=1e-5)


# --------------------------------------------------- jacrev == assembled STM (jax)
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
@pytest.mark.parametrize("method", _METHODS)
def test_jacrev_equals_stm(method):
    from galpy.backend._reference.inbackend_stm import c_stm_forward

    pot = MiyamotoNagaiPotential(normalize=1.0)
    _, M = c_stm_forward(pot, _IC, _TS, method, 1e-10, 1e-10)
    Jall = jax.jacrev(lambda v: _integ("jax", pot, v, _TS, method))(jnp.asarray(_IC))
    numpy.testing.assert_allclose(_np(Jall), M, rtol=1e-10, atol=1e-10)


# ----------------------------------------------------------------- torch gradcheck
@pytest.mark.skipif("torch" not in BACKENDS, reason="needs torch")
@pytest.mark.parametrize("method", ["rk4_c", "rk6_c"])
def test_torch_gradcheck(method):
    pot = MiyamotoNagaiPotential(normalize=1.0)
    ts = numpy.linspace(0.0, 1.0, 4)
    v = torch.tensor(_IC, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda vv: _integ("torch", pot, vv, ts, method), (v,), eps=1e-6, atol=1e-4
    )


# ------------------------------------- C-STM vs in-backend ODE (independent check)
@pytest.mark.parametrize("backend", BACKENDS)
def test_cstm_grad_matches_inbackend_ode(backend):
    # the fast C-STM IC gradient must match the diffrax/torchdiffeq AD gradient
    from galpy.backend._reference.inbackend_ode import integrate_orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    if backend == "jax":
        g_stm = jax.grad(lambda v: _integ("jax", pot, v, _TS, "dop853_c")[-1, 0])(
            jnp.asarray(_IC)
        )
        g_ode = jax.grad(lambda v: integrate_orbit(pot, v, jnp.asarray(_TS))[-1, 0])(
            jnp.asarray(_IC)
        )
    else:
        v1 = torch.tensor(_IC, requires_grad=True)
        _integ("torch", pot, v1, _TS, "dop853_c")[-1, 0].backward()
        g_stm = v1.grad
        v2 = torch.tensor(_IC, requires_grad=True)
        integrate_orbit(pot, v2, torch.tensor(_TS))[-1, 0].backward()
        g_ode = v2.grad
    numpy.testing.assert_allclose(_np(g_stm), _np(g_ode), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------- cross-backend agreement
@pytest.mark.skipif(
    "jax" not in BACKENDS or "torch" not in BACKENDS, reason="needs both"
)
def test_torch_grad_matches_jax():
    pot = MiyamotoNagaiPotential(normalize=1.0)
    g_jax = _np(
        jax.grad(lambda v: _integ("jax", pot, v, _TS, "dop853_c")[-1, 0])(
            jnp.asarray(_IC)
        )
    )
    v = torch.tensor(_IC, requires_grad=True)
    _integ("torch", pot, v, _TS, "dop853_c")[-1, 0].backward()
    numpy.testing.assert_allclose(_np(v.grad), g_jax, rtol=1e-8, atol=1e-10)


# -------------------------------------------------------------- batch / vmap (jax)
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_batch_and_vmap():
    pot = MiyamotoNagaiPotential(normalize=1.0)
    ics = jnp.asarray(numpy.stack([_IC, _IC * 1.01, _IC * 0.99]))
    batch = _integ("jax", pot, ics, _TS, "dop853_c")
    assert batch.shape == (3, len(_TS), 6)
    vm = jax.vmap(lambda v: _integ("jax", pot, v, _TS, "dop853_c"))(ics)
    numpy.testing.assert_allclose(_np(batch), _np(vm), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------- numpy IC raises
def test_numpy_ic_raises():
    from galpy.backend._reference.inbackend_stm import integrate_stm

    pot = MiyamotoNagaiPotential(normalize=1.0)
    with pytest.raises(NotImplementedError):
        integrate_stm(pot, _IC, _TS, method="dop853_c")


# ------------------------------------------------------------- bad method raises
def test_integrate_stm_bad_method_raises():
    # integrate_stm only supports the dxdv-capable C integrators
    from galpy.backend._reference.inbackend_stm import integrate_stm

    pot = MiyamotoNagaiPotential(normalize=1.0)
    with pytest.raises(ValueError):
        integrate_stm(pot, _IC, _TS, method="odeint")


# ----------------------------------------------------- public dispatcher routing
@pytest.mark.parametrize("backend", BACKENDS)
def test_integrate_stm_dispatch(backend):
    # the public dispatcher routes a jax/torch IC to the matching backend wrapper
    # -> identical result to calling that wrapper directly
    from galpy.backend._reference.inbackend_stm import integrate_stm

    pot = MiyamotoNagaiPotential(normalize=1.0)
    v = _arr(backend, _IC)
    got = _np(integrate_stm(pot, v, _TS, method="dop853_c"))
    ref = _np(_integ(backend, pot, v, _TS, "dop853_c"))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


# -------------------------------------------- batch gradient (batch M^T einsum)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_gradient_matches_single(backend):
    # a gradient through a BATCHED integrate exercises the batched STM contraction
    # ("ntab,nta->nb"); each row must equal the single-orbit IC gradient
    pot = MiyamotoNagaiPotential(normalize=1.0)
    ics = numpy.stack([_IC, _IC * 1.01, _IC * 0.99])
    if backend == "jax":
        gb = _np(
            jax.grad(
                lambda vv: _integ("jax", pot, vv, _TS, "dop853_c")[:, -1, 0].sum()
            )(jnp.asarray(ics))
        )
    else:
        v = torch.tensor(ics, requires_grad=True)
        _integ("torch", pot, v, _TS, "dop853_c")[:, -1, 0].sum().backward()
        gb = _np(v.grad)
    for j, ic in enumerate(ics):
        if backend == "jax":
            gj = _np(
                jax.grad(lambda v: _integ("jax", pot, v, _TS, "dop853_c")[-1, 0])(
                    jnp.asarray(ic)
                )
            )
        else:
            vv = torch.tensor(ic, requires_grad=True)
            _integ("torch", pot, vv, _TS, "dop853_c")[-1, 0].backward()
            gj = _np(vv.grad)
        numpy.testing.assert_allclose(gb[j], gj, rtol=1e-8, atol=1e-10)


# ----------------------------------------- Orbit.integrate(...) C-STM routing (#57)
# Orbit.integrate(method=<dxdv C method>) built from a jax/torch IC routes through
# the C-STM, so getOrbit()/the accessors are backend arrays differentiable w.r.t.
# the IC -- no method='diffrax'/'torchdiffeq' needed. A numpy IC is untouched.
def _is_backend(backend, x):
    return "jax" in type(x).__module__ if backend == "jax" else torch.is_tensor(x)


# the four dxdv-capable C integrators the routing accepts (incl. dopr54_c)
_ROUTE_METHODS = ["rk4_c", "rk6_c", "dopr54_c", "dop853_c"]
# every per-time scalar accessor that should survive on the backend + match C
_ROUTE_ACCESSORS = ["R", "vR", "vT", "z", "vz", "x", "y", "vx", "vy", "r", "vr", "E"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", _ROUTE_METHODS)
def test_orbit_integrate_cstm_forward_parity(backend, method):
    # the routed C-STM trajectory matches the numpy/C orbit across potentials and
    # EVERY phase-space accessor, and every accessor stays a backend array.
    from galpy.orbit import Orbit

    ts = _arr(backend, _TS)
    for name, pot in _pots().items():
        o = Orbit(_arr(backend, _IC))
        o.integrate(ts, pot, method=method)
        assert _is_backend(backend, o.getOrbit()), f"{name} getOrbit left {backend}"
        onp = Orbit(list(_IC))
        onp.integrate(_TS, pot, method=method)
        for acc in _ROUTE_ACCESSORS:
            val = getattr(o, acc)(ts, use_physical=False)
            assert _is_backend(backend, val), f"{name}.{acc} left {backend}"
            numpy.testing.assert_allclose(
                _np(val),
                numpy.asarray(getattr(onp, acc)(_TS, use_physical=False)),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{name} {acc} {method} {backend}",
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_full_ic_jacobian_vs_fd(backend):
    # the full 6x6 d(final state)/d(IC) -- the STM at t[-1] -- via reverse-mode
    # autodiff of the ROUTED Orbit.integrate, vs central finite difference.
    from galpy.orbit import Orbit

    pot = MWPotential2014

    def final_np(ic):
        o = Orbit(list(ic))
        o.integrate(_TS, pot, method="dop853_c")
        return numpy.asarray(o.getOrbit()[-1])  # (6,) Orbit order

    eps = 1e-6
    jfd = numpy.array(
        [
            (
                final_np(_IC + eps * numpy.eye(6)[j])
                - final_np(_IC - eps * numpy.eye(6)[j])
            )
            / (2.0 * eps)
            for j in range(6)
        ]
    ).T  # (6 out, 6 in)

    def final_b(v):
        o = Orbit(v)
        o.integrate(_arr(backend, _TS), pot, method="dop853_c")
        return o.getOrbit()[-1]

    if backend == "jax":
        jac = _np(jax.jacrev(final_b)(jnp.asarray(_IC)))
    else:
        jac = _np(torch.autograd.functional.jacobian(final_b, torch.tensor(_IC)))
    numpy.testing.assert_allclose(jac, jfd, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("acc", ["e", "rap", "rperi", "zmax"])
def test_orbit_integrate_cstm_grad_accessor_vs_fd(backend, acc):
    # gradient through a derived orbit characteristic (e/rap/rperi/zmax) over the
    # C integration matches finite difference, for an axisymmetric and a
    # non-trivial (MWPotential2014) potential.
    from galpy.orbit import Orbit

    for pot in (MiyamotoNagaiPotential(normalize=1.0), MWPotential2014):

        def f_np(ic):
            o = Orbit(list(ic))
            o.integrate(_TS, pot, method="dop853_c")
            return float(getattr(o, acc)(use_physical=False))

        eps = 1e-6
        gfd = numpy.array(
            [
                (f_np(_IC + eps * numpy.eye(6)[j]) - f_np(_IC - eps * numpy.eye(6)[j]))
                / (2.0 * eps)
                for j in range(6)
            ]
        )

        def f_b(v):
            o = Orbit(v)
            o.integrate(_arr(backend, _TS), pot, method="dop853_c")
            return getattr(o, acc)(use_physical=False)

        if backend == "jax":
            g = _np(jax.grad(f_b)(jnp.asarray(_IC)))
        else:
            v = torch.tensor(_IC, requires_grad=True)
            f_b(v).backward()
            g = _np(v.grad)
        numpy.testing.assert_allclose(
            g, gfd, rtol=1e-4, atol=1e-5, err_msg=f"{acc} {backend}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_matches_functional(backend):
    # the routed Orbit.integrate uses the same orbit_stm wrapper as the functional
    # interface -> the trajectory is identical to orbit_stm.integrate(...).
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    o = Orbit(_arr(backend, _IC))
    o.integrate(_arr(backend, _TS), pot, method="dop853_c")
    func = _np(_integ(backend, pot, _arr(backend, _IC), _TS, "dop853_c"))  # (nt,6)
    # identical wrapper -> identical to ~roundoff (backend-vs-numpy ts in the
    # internal numpy.asarray gives a sub-1e-11 delta; a different path would
    # differ by >>1e-9).
    numpy.testing.assert_allclose(_np(o.getOrbit()), func, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_fallback_inbackend(backend):
    # a non-6D (here 5D) orbit is not C-STM-eligible -> in-backend ODE fallback:
    # the trajectory matches the numpy 5D orbit AND stays a differentiable backend
    # array.
    pytest.importorskip("diffrax" if backend == "jax" else "torchdiffeq")
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    ic5 = numpy.array(_IC[:5])  # R,vR,vT,z,vz (no phi)
    o5 = Orbit(_arr(backend, ic5))
    o5.integrate(_arr(backend, _TS), pot, method="dop853_c")
    assert _is_backend(backend, o5.getOrbit())
    onp = Orbit(list(ic5))
    onp.integrate(_TS, pot, method="dop853_c")
    numpy.testing.assert_allclose(
        _np(o5.R(_arr(backend, _TS), use_physical=False)),
        onp.R(_TS, use_physical=False),
        rtol=1e-6,
        atol=1e-6,
    )

    def fR(v):
        o = Orbit(v)
        o.integrate(_arr(backend, _TS), pot, method="dop853_c")
        return o.R(_arr(backend, _TS), use_physical=False)[-1]

    if backend == "jax":
        g = _np(jax.grad(fR)(jnp.asarray(ic5)))[1]
    else:
        v = torch.tensor(ic5, requires_grad=True)
        fR(v).backward()
        g = _np(v.grad)[1]
    assert numpy.isfinite(g)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_multiorbit_matches_loop(backend):
    # A multi-orbit backend IC integrates in ONE stacked, differentiable C-STM
    # solve; getOrbit() is (N, nt, 6) and matches the per-orbit single-orbit loop.
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    ics = numpy.array([_IC, _IC + 0.03, _IC - 0.02])  # (3, 6)
    ts = _arr(backend, _TS)
    om = Orbit(_arr(backend, ics))
    om.integrate(ts, pot, method="dop853_c")
    multi = _np(om.getOrbit())
    assert multi.shape == (len(ics), len(_TS), 6)
    for k in range(len(ics)):
        ok = Orbit(_arr(backend, ics[k]))
        ok.integrate(ts, pot, method="dop853_c")
        numpy.testing.assert_allclose(
            multi[k], _np(ok.getOrbit()), rtol=1e-12, atol=1e-12
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_orbit_integrate_cstm_multiorbit_grad():
    # The batched C-STM is gradient-exact: each orbit's IC jacobian equals the
    # single-orbit jacobian (diagonal blocks) and the orbits are independent
    # (the off-diagonal blocks of jacrev are exactly zero).
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    ics = jnp.asarray(numpy.array([_IC, _IC + 0.03, _IC - 0.02]))
    N = ics.shape[0]

    def final_batch(v):  # (N,6) -> (N,6) final state
        o = Orbit(v)
        o.integrate(jnp.asarray(_TS), pot, method="dop853_c")
        return o.getOrbit()[:, -1, :]

    Jb = _np(jax.jacrev(final_batch)(ics))  # (N, 6, N, 6)

    def final_single(v6):
        o = Orbit(v6)
        o.integrate(jnp.asarray(_TS), pot, method="dop853_c")
        return o.getOrbit()[-1, :]

    for i in range(N):
        for j in range(N):
            if i != j:
                assert numpy.max(numpy.abs(Jb[i, :, j, :])) == 0.0  # independent
        Js = _np(jax.jacrev(final_single)(ics[i]))  # (6, 6)
        numpy.testing.assert_allclose(Jb[i, :, i, :], Js, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", _ROUTE_METHODS)
def test_orbit_integrate_numpy_ic_unaffected(method):
    # a numpy/list IC + a C method stays the byte-identical numpy path (no routing).
    from galpy.orbit import Orbit

    for pot in (MiyamotoNagaiPotential(normalize=1.0), MWPotential2014):
        o = Orbit(list(_IC))
        o.integrate(_TS, pot, method=method)
        assert isinstance(o.getOrbit(), numpy.ndarray)
        assert isinstance(o.R(_TS, use_physical=False), numpy.ndarray)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_numpy_times(backend):
    # a numpy/list time array (not a backend array) is coerced onto the IC's
    # namespace before the C-STM solve (the is_backend_array(t) else-branch of
    # _integrate_cstm); the result is still a differentiable backend array.
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    o = Orbit(_arr(backend, _IC))
    o.integrate(_TS, pot, method="dop853_c")  # _TS is a plain numpy array
    assert _is_backend(backend, o.getOrbit())
    onp = Orbit(list(_IC))
    onp.integrate(_TS, pot, method="dop853_c")
    numpy.testing.assert_allclose(
        _np(o.getOrbit()), onp.getOrbit(), rtol=1e-6, atol=1e-7
    )
