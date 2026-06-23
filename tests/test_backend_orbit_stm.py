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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", _METHODS)
def test_orbit_integrate_routes_cstm(backend, method):
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    ts = _arr(backend, _TS)
    o = Orbit(_arr(backend, _IC))
    o.integrate(ts, pot, method=method)
    assert _is_backend(backend, o.getOrbit())
    assert _is_backend(backend, o.R(ts, use_physical=False))
    onp = Orbit(list(_IC))
    onp.integrate(_TS, pot, method=method)
    numpy.testing.assert_allclose(
        _np(o.R(ts, use_physical=False)),
        onp.R(_TS, use_physical=False),
        rtol=1e-6,
        atol=1e-7,
        err_msg=f"{method} {backend}",
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_grad_ecc_vs_fd(backend):
    # the headline: d(eccentricity)/d(IC) over a *C* integration, via the STM.
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)

    def ecc_np(ic):
        o = Orbit(list(ic))
        o.integrate(_TS, pot, method="dop853_c")
        return float(o.e(use_physical=False))

    eps = 1e-6
    gfd = numpy.array(
        [
            (ecc_np(_IC + eps * numpy.eye(6)[j]) - ecc_np(_IC - eps * numpy.eye(6)[j]))
            / (2 * eps)
            for j in range(6)
        ]
    )

    def ecc_b(v):
        o = Orbit(v)
        o.integrate(_arr(backend, _TS), pot, method="dop853_c")
        return o.e(use_physical=False)

    if backend == "jax":
        g = _np(jax.grad(ecc_b)(jnp.asarray(_IC)))
    else:
        v = torch.tensor(_IC, requires_grad=True)
        ecc_b(v).backward()
        g = _np(v.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_integrate_cstm_fallback_inbackend(backend):
    # a non-6D orbit is not C-STM-eligible -> falls back to the in-backend ODE
    # solver, still returning a differentiable backend array.
    pytest.importorskip("diffrax" if backend == "jax" else "torchdiffeq")
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    o5 = Orbit(_arr(backend, _IC[:5]))  # 5D (no phi) -> in-backend fallback
    o5.integrate(_arr(backend, _TS), pot, method="dop853_c")
    assert _is_backend(backend, o5.getOrbit())


def test_orbit_integrate_numpy_ic_unaffected():
    # a numpy/list IC + a C method stays the byte-identical numpy path (no routing).
    from galpy.orbit import Orbit

    pot = MiyamotoNagaiPotential(normalize=1.0)
    o = Orbit(list(_IC))
    o.integrate(_TS, pot, method="dop853_c")
    assert isinstance(o.getOrbit(), numpy.ndarray)
