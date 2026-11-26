###############################################################################
# test_backend_actionAngleAdiabaticApprox.py: the adiabatic-transformation
# action-angle classes with backend arrays and under a forced backend. For
# concrete inputs they run on the host (and return on the namespace of the
# inputs); traced or differentiated inputs, differentiated parameters of the
# potentials, and in-backend integrators run in the backend
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.actionAngle import (
    actionAngleAdiabaticApprox,
    actionAngleIsochrone,
    actionAngleIsochroneInverse,
    actionAngleSpherical,
    actionAngleStaeckel,
)
from galpy.backend import as_numpy, is_backend_array
from galpy.backend._namespaces import namespace_for_name
from galpy.potential import IsochronePotential, NFWPotential
from galpy.util import galpyWarning

pytestmark = pytest.mark.backend_managed

BACKENDS = []
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

_X = numpy.array(
    [
        [1.0, 0.1, 1.1, 0.1, 0.2, 0.3],
        [0.9, -0.2, 0.9, -0.2, 0.1, 2.0],
    ]
).T


def _setup(a=2.0, **kwargs):
    return actionAngleAdiabaticApprox(
        **{
            **dict(
                pot=NFWPotential(a=a, amp=3.0),
                aA=actionAngleIsochrone(b=0.5),
                tintJ=20.0,
            ),
            **kwargs,
        }
    )


def _derivative(backend_name, func, x, ii):
    """The derivative of the sum over the points of output ii of func, with
    respect to the array x of inputs [k,M]: the partials of each point's
    output with respect to its own inputs"""
    if backend_name == "jax":
        return numpy.asarray(
            jax.grad(lambda x: jax.numpy.sum(func(*x)[ii]))(jax.numpy.asarray(x))
        )
    x = torch.tensor(x, requires_grad=True)
    (grad,) = torch.autograd.grad(torch.sum(func(*x)[ii]), x)
    return as_numpy(grad)


def _finite_difference(func, x, ii, wrap=False, h=1e-6):
    out = numpy.empty_like(x)
    for jj in range(len(x)):
        dx = numpy.zeros_like(x)
        dx[jj] = h
        d = as_numpy(func(*(x + dx))[ii]) - as_numpy(func(*(x - dx))[ii])
        out[jj] = ((d + numpy.pi) % (2.0 * numpy.pi) - numpy.pi if wrap else d) / (
            2.0 * h
        )
    return out


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_backend(backend_name):
    aAA = _setup()
    ref = numpy.array(aAA.actionsFreqsAngles(*_X))
    xp = namespace_for_name(backend_name)
    # backend arrays in: backend arrays out, with numpy's values
    for method in ("__call__", "actionsFreqs", "actionsFreqsAngles"):
        got = getattr(aAA, method)(*(xp.asarray(x) for x in _X))
        assert all(is_backend_array(g) for g in got)
        assert numpy.allclose(
            numpy.array([as_numpy(g) for g in got]),
            ref[: len(got)],
            rtol=1e-10,
            atol=1e-10,
        )
    # a forced backend with numpy input, also when the instance is set up
    # under it
    with backend.use(backend_name, force=True):
        got = _setup().actionsFreqsAngles(*_X)
    assert all(is_backend_array(g) for g in got)
    assert numpy.allclose(
        numpy.array([as_numpy(g) for g in got]), ref, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize(
    "kwargs",
    [dict(npoints=2, calibrate_angles=True), dict(npoints=0)],
    ids=["calibrated", "singlepoint"],
)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_derivatives(backend_name, kwargs):
    # Differentiated inputs: the values are numpy's, and the derivatives with
    # respect to the phase-space point those of finite differences
    aAA = _setup(**kwargs)
    ref = numpy.array(aAA.actionsFreqsAngles(*_X))
    if backend_name == "torch":
        got = aAA.actionsFreqsAngles(*torch.tensor(_X, requires_grad=True))
        # (for npoints=0, numpy has no frequencies)
        keep = numpy.all(numpy.isfinite(ref), axis=1)
        assert numpy.allclose(
            numpy.array([as_numpy(g.detach()) for g in got])[keep],
            ref[keep],
            rtol=1e-8,
            atol=1e-8,
        )
    for ii in (0, 6) + (4,) * (kwargs["npoints"] > 0):
        assert numpy.allclose(
            _derivative(backend_name, aAA.actionsFreqsAngles, _X, ii),
            _finite_difference(aAA.actionsFreqsAngles, _X, ii, wrap=ii > 5),
            rtol=1e-5,
            atol=1e-5,
        ), f"derivative of output {ii} of actionAngleAdiabaticApprox"
    # (actions and actions with frequencies alone, and planar input)
    assert numpy.allclose(
        _derivative(backend_name, aAA, _X, 0),
        _derivative(backend_name, aAA.actionsFreqs, _X, 0),
        rtol=1e-10,
        atol=1e-10,
    )
    planar = _X[[0, 1, 2, 5]]
    assert numpy.allclose(
        _derivative(backend_name, aAA, planar, 0),
        _finite_difference(aAA, planar, 0),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_inbackend_and_parameters(backend_name):
    # An in-backend integrator gives the C integrator's values for backend
    # arrays, and derivatives with respect to a parameter of the potential,
    # which the default integrator is rerouted for
    xp = namespace_for_name(backend_name)
    method = {"jax": "diffrax", "torch": "torchode"}[backend_name]
    ref = numpy.array(_setup(npoints=0)(*_X))
    got = _setup(npoints=0, integrate_method=method)(*(xp.asarray(x) for x in _X))
    assert numpy.allclose(
        numpy.array([as_numpy(g) for g in got]), ref, rtol=1e-8, atol=1e-8
    )

    def jr(a, method="dop853_c"):
        return _setup(a=a, npoints=0, integrate_method=method)(*_X[:, 0])[0][0]

    fd = (float(jr(2.0 + 1e-5)) - float(jr(2.0 - 1e-5))) / 2e-5
    for method in ("dop853_c", method):
        if backend_name == "jax":
            grad = float(jax.grad(lambda a: jr(a, method))(2.0))
        else:
            a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
            (grad,) = torch.autograd.grad(jr(a, method), a)
        assert abs(float(grad) / fd - 1.0) < 1e-6


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_calibrate_without_inverse_backend(backend_name):
    # calibrate_angles without an inverse of aA=: the offset is that of each
    # point's own image, also in the backend
    kwargs = dict(
        aA=actionAngleSpherical(pot=IsochronePotential(b=0.5, normalize=1.0)),
        npoints=0,
        calibrate_angles=True,
    )
    aAA = _setup(**kwargs)
    assert numpy.allclose(
        _derivative(backend_name, aAA.actionsFreqsAngles, _X, 6),
        _finite_difference(aAA.actionsFreqsAngles, _X, 6, wrap=True),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "backend_name,c",
    [(b, True) for b in BACKENDS] + ([("torch", False)] if torch else []),
)
def test_actionAngleAdiabaticApprox_calibrate_unbound_in_auxiliary_backend(
    backend_name, c
):
    # a point that is unbound in the potential of aA=: no half turns from
    # aA=, with a warning, also in the backend (with c=False, aA= raises; on
    # torch only, as eager jax takes minutes for it)
    aAA = _setup(
        aA=actionAngleStaeckel(
            pot=IsochronePotential(b=0.5, normalize=0.3), delta=0.1, c=c
        ),
        npoints=0,
        calibrate_angles=True,
    )
    fast = numpy.array([[1.0, 0.3, 1.1, 0.1, 0.2, 0.5]]).T
    with pytest.warns(galpyWarning, match="may be half a turn off"):
        grad = _derivative(backend_name, aAA.actionsFreqsAngles, fast, 6)
    with pytest.warns(galpyWarning, match="may be half a turn off"):
        fd = _finite_difference(aAA.actionsFreqsAngles, fast, 6, wrap=True)
    assert numpy.allclose(grad, fd, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApproxInverse_backend(backend_name):
    from galpy.actionAngle import actionAngleAdiabaticApproxInverse

    aAA = _setup()

    def setup_inverse(**kwargs):
        return actionAngleAdiabaticApproxInverse(
            pot=NFWPotential(a=2.0, amp=3.0),
            aAI=actionAngleIsochroneInverse(b=0.5),
            aA=actionAngleIsochrone(b=0.5),
            tintJ=20.0,
            **kwargs,
        )

    aAAI = setup_inverse()
    ja = numpy.array(aAA.actionsFreqsAngles(*_X))[[0, 1, 2, 6, 7, 8]]
    ref = numpy.array(aAAI.xvFreqs(*ja))
    reffreqs = numpy.array(aAAI.Freqs(*ja[:3]))
    xp = namespace_for_name(backend_name)
    bja = [xp.asarray(x) for x in ja]
    for got, want in (
        (aAAI(*bja), ref[:6]),
        (aAAI.xvFreqs(*bja), ref),
        (aAAI.Freqs(*bja[:3]), reffreqs),
    ):
        assert all(is_backend_array(g) for g in got)
        assert numpy.allclose(
            numpy.array([as_numpy(g) for g in got]), want, rtol=1e-10, atol=1e-10
        )
    with backend.use(backend_name, force=True):
        got = setup_inverse().xvFreqs(*ja)
    assert all(is_backend_array(g) for g in got)
    assert numpy.allclose(
        numpy.array([as_numpy(g) for g in got]), ref, rtol=1e-8, atol=1e-8
    )
    # differentiated inputs: derivatives with respect to actions and angles
    for calibrate in (False, True):
        aAAI = setup_inverse(calibrate_angles=calibrate)
        for ii in (0, 7):
            assert numpy.allclose(
                _derivative(backend_name, aAAI.xvFreqs, ja, ii),
                _finite_difference(aAAI.xvFreqs, ja, ii),
                rtol=1e-4,
                atol=1e-4,
            ), f"derivative of output {ii} of actionAngleAdiabaticApproxInverse"
    assert numpy.allclose(
        _derivative(backend_name, aAAI.Freqs, ja[:3], 1),
        _finite_difference(aAAI.Freqs, ja[:3], 1),
        rtol=1e-4,
        atol=1e-4,
    )
    assert numpy.allclose(
        _derivative(backend_name, aAAI, ja, 3),
        _finite_difference(aAAI, ja, 3),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_frequencies_npoints0(backend_name):
    # With npoints=0, the frequencies of backend inputs are dH/dJ at fixed
    # angles, by differentiating the transformation along each point's orbit
    # in an in-backend integration (also for the default C method). Between
    # two isochrones, they approach the exact frequencies as tintJ grows; on
    # numpy there are none
    true = IsochronePotential(b=0.8, normalize=1.0)
    aAA = actionAngleAdiabaticApprox(
        pot=true,
        aA=actionAngleIsochrone(ip=IsochronePotential(b=0.5, normalize=1.0)),
        tintJ=100.0,
        npoints=0,
    )
    exact = numpy.array(actionAngleIsochrone(ip=true).actionsFreqs(*_X))
    xp = namespace_for_name(backend_name)
    got = aAA.actionsFreqsAngles(*(xp.asarray(x) for x in _X))
    assert all(is_backend_array(g) for g in got)
    got = numpy.array([as_numpy(g) for g in got])
    assert numpy.allclose(got[:6], exact, rtol=1e-4, atol=1e-6)
    assert numpy.allclose(
        numpy.array([as_numpy(g) for g in aAA.actionsFreqs(*map(xp.asarray, _X))]),
        got[:6],
        rtol=1e-12,
        atol=1e-12,
    )
    assert numpy.all(numpy.isnan(aAA.actionsFreqs(*_X)[3:]))
    # more than 100 points (integrated with diffrax on jax): the same
    many = numpy.tile(_X, 51)
    got = aAA.actionsFreqs(*(xp.asarray(x) for x in many))
    assert numpy.allclose(
        numpy.array([as_numpy(g) for g in got])[:, :2], exact, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApprox_frequency_derivatives_npoints0(backend_name):
    # The frequencies for npoints=0 are differentiable with respect to the
    # potential's parameters and, with an in-backend integrator, the
    # phase-space point (a second derivative: through the C integrators it
    # raises, while the actions remain differentiable)
    x = _X[:, 0]

    def out(a, x, ii=3, **kwargs):
        return _setup(a=a, npoints=0, tintJ=5.0, **kwargs).actionsFreqs(*x)[ii][0]

    xp = namespace_for_name(backend_name)
    fd_a = (
        float(out(2.0 + 1e-5, xp.asarray(x))) - float(out(2.0 - 1e-5, xp.asarray(x)))
    ) / 2e-5
    fd_x = [
        _finite_difference(
            lambda *x: [out(2.0, xp.asarray(numpy.array(x)), ii)], x, 0, h=1e-5
        )
        for ii in (3, 0)
    ]
    if backend_name == "jax":
        grad_a = jax.grad(lambda a: out(a, jax.numpy.asarray(x)))(2.0)
        with pytest.raises(NotImplementedError, match="second derivatives"):
            jax.grad(lambda x: out(2.0, x))(jax.numpy.asarray(x))
        jr_x = jax.grad(lambda x: out(2.0, x, 0))(jax.numpy.asarray(x))
        grad_x = jax.grad(lambda x: out(2.0, x, integrate_method="diffrax"))(
            jax.numpy.asarray(x)
        )
    else:
        a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        (grad_a,) = torch.autograd.grad(out(a, torch.tensor(x)), a)
        tx = torch.tensor(x, requires_grad=True)
        with pytest.raises(RuntimeError, match="second derivatives"):
            torch.autograd.grad(out(2.0, tx), tx)
        (jr_x,) = torch.autograd.grad(out(2.0, tx, 0), tx)
        (grad_x,) = torch.autograd.grad(out(2.0, tx, integrate_method="torchode"), tx)
        # (an aA= whose derivatives are C's, to first order, raises too)
        aA = actionAngleStaeckel(
            pot=IsochronePotential(b=0.5, normalize=1.0), delta=0.1, c=True
        )
        with pytest.raises(RuntimeError, match="through the C actions"):
            torch.autograd.grad(out(2.0, tx, aA=aA, integrate_method="torchode"), tx)
    assert abs(float(grad_a) / fd_a - 1.0) < 1e-6
    assert numpy.allclose(as_numpy(grad_x), fd_x[0], rtol=1e-6, atol=1e-7)
    assert numpy.allclose(as_numpy(jr_x), fd_x[1], rtol=1e-6, atol=1e-7)


def test_actionAngleAdiabaticApprox_integrator_rule():
    # which integrations of a C integrate_method run in the backend's own
    # solver (_inbackend): on jax, on any device, for many points and under
    # jax.jit or jax.vmap, except for the derivatives of the tangents (the
    # frequencies'); for derivatives with respect to the potential's
    # parameters; and never for another integrate_method or on PyTorch
    import importlib

    mod = importlib.import_module("galpy.actionAngle.actionAngleAdiabaticApprox")
    aAA = _setup(npoints=0)
    if jax:
        jnp = jax.numpy
        few, many, most, more = (jnp.ones((n, 6)) for n in (10, 200, 600, 1001))
        assert not mod._inbackend(aAA, few) and not mod._inbackend(aAA, most)
        assert mod._inbackend(aAA, more)
        assert not mod._inbackend(aAA, few, tangent=True)
        assert mod._inbackend(aAA, many, tangent=True)
        seen = {}

        def probe(tag, **kwargs):
            def f(x):
                seen[tag] = mod._inbackend(aAA, x, **kwargs)
                return jnp.sum(x)

            return f

        jax.jit(probe("jit", tangent=True))(few)
        jax.vmap(probe("vmap"))(jnp.ones((2, 1, 6)))
        jax.grad(probe("grad tangent", tangent=True))(many)
        jax.jit(jax.grad(probe("jit grad tangent", tangent=True)))(many)
        jax.grad(probe("grad"))(most)
        jax.grad(probe("grad many"))(many)
        jax.jit(jax.grad(probe("jit grad")))(few)
        jax.grad(
            lambda a: (
                seen.update(param=mod._inbackend(_setup(a=a, npoints=0), few)) or a
            )
        )(2.0)
        assert seen == {
            "jit": True,
            "vmap": True,
            "grad tangent": False,
            "jit grad tangent": False,
            "grad": True,
            "grad many": False,
            "jit grad": True,
            "param": True,
        }
        assert not mod._inbackend(_setup(npoints=0, integrate_method="diffrax"), more)
    if torch:
        assert not mod._inbackend(aAA, torch.ones((2000, 6), dtype=torch.float64))


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_columnwise_jvp(backend_name):
    # galpy.backend.optimize.columnwise_jvp: f(p) and the derivative of each
    # column of f(p) in the direction of that column of t, with or without
    # an aux output, and differentiable again with create_graph
    from galpy.backend.optimize import columnwise_jvp

    xp = namespace_for_name(backend_name)
    p = xp.asarray(numpy.array([[0.3, -0.7], [1.1, 0.4]]))
    t = xp.asarray(numpy.array([[1.0, 2.0], [0.5, -1.0]]))

    def f(p):
        return xp.stack([p[0] * p[1], p[0] ** 2.0])

    for reverse in (False, True, None):  # (reverse: backward passes on jax too)
        val, rate = columnwise_jvp(f, p, t, reverse=reverse)
        assert numpy.allclose(as_numpy(val), [[0.33, -0.28], [0.09, 0.49]])
        assert numpy.allclose(as_numpy(rate), [[1.25, 1.5], [0.6, -2.8]])
    for aux in (lambda p: 2.0 * p, lambda p: (2.0 * p, 3.0 * p)):
        val, rate, got = columnwise_jvp(
            lambda p: (f(p), aux(p)), p, t, has_aux=True, reverse=True
        )
        assert numpy.allclose(as_numpy(rate), [[1.25, 1.5], [0.6, -2.8]])
        want = aux(as_numpy(p))
        for g, w in zip(
            *((got, want) if isinstance(want, tuple) else ((got,), (want,)))
        ):
            assert numpy.allclose(as_numpy(g), w)
    if backend_name == "jax":
        # reverse=None: backward passes for a function that only has those
        @jax.custom_vjp
        def g(p):
            return f(p)

        g.defvjp(lambda p: (f(p), p), lambda p, ct: (jax.vjp(f, p)[1](ct)[0],))
        assert numpy.allclose(
            as_numpy(columnwise_jvp(g, p, t, reverse=None)[1]),
            [[1.25, 1.5], [0.6, -2.8]],
        )
        # traced: the values differentiable, the rate not (also under jit,
        # and for a custom_vjp within a jitted function)
        for wrap, h in ((lambda u: u, g), (jax.jit, g), (lambda u: u, jax.jit(g))):
            assert numpy.allclose(
                as_numpy(
                    wrap(
                        jax.grad(
                            lambda p: columnwise_jvp(h, p, t, reverse=None)[0][1, 0]
                        )
                    )(p)
                ),
                [[0.6, 0.0], [0.0, 0.0]],
            )
            with pytest.raises(NotImplementedError, match="second derivatives"):
                wrap(
                    jax.grad(lambda p: columnwise_jvp(h, p, t, reverse=None)[1][1, 0])
                )(p)
        assert numpy.allclose(
            as_numpy(jax.jit(lambda p: columnwise_jvp(g, p, t, reverse=None)[1])(p)),
            [[1.25, 1.5], [0.6, -2.8]],
        )

        def bad(p):
            raise TypeError("other")

        with pytest.raises(TypeError, match="other"):
            columnwise_jvp(bad, p, t, reverse=None)
    # d(rate[1, 0])/d(p[0, 0]) = 2 t[0, 0]
    if backend_name == "jax":
        d = jax.grad(lambda p: columnwise_jvp(f, p, t)[1][1, 0])(p)
        assert numpy.allclose(
            as_numpy(
                jax.grad(lambda p: columnwise_jvp(f, p, t, reverse=None)[1][1, 0])(p)
            ),
            as_numpy(d),
        )
    else:
        p = torch.tensor(as_numpy(p), requires_grad=True)
        (d,) = torch.autograd.grad(
            columnwise_jvp(f, p, t, create_graph=True)[1][1, 0], p
        )
    assert numpy.allclose(as_numpy(d), [[2.0, 0.0], [0.0, 0.0]])
