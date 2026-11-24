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
)
from galpy.backend import as_numpy, is_backend_array
from galpy.backend._namespaces import namespace_for_name
from galpy.potential import IsochronePotential, NFWPotential

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
    return grad.numpy()


def _finite_difference(func, x, ii, wrap=False, h=1e-6):
    out = numpy.empty_like(x)
    for jj in range(len(x)):
        dx = numpy.zeros_like(x)
        dx[jj] = h
        d = numpy.array(func(*(x + dx))[ii]) - numpy.array(func(*(x - dx))[ii])
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
        assert numpy.allclose(
            numpy.array([as_numpy(g.detach()) for g in got]),
            ref,
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
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
