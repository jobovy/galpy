###############################################################################
# test_backend_actionAngleAdiabaticApprox.py: the adiabatic-transformation
# action-angle classes with backend arrays and under a forced backend. They
# integrate orbits in C, so they run on the host for concrete inputs (and
# return on the namespace of the inputs) and refuse to be differentiated
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.actionAngle import (
    actionAngleAdiabaticApprox,
    actionAngleIsochrone,
    actionAngleIsochroneInverse,
)
from galpy.backend import as_numpy, is_backend_array
from galpy.backend._namespaces import namespace_for_name
from galpy.potential import NFWPotential

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


def _setup():
    return actionAngleAdiabaticApprox(
        pot=NFWPotential(a=2.0, normalize=1.0),
        aA=actionAngleIsochrone(b=0.5),
        tintJ=20.0,
    )


def _differentiate(backend_name, func, x):
    """The derivative of func's first output with respect to x[0]"""
    if backend_name == "jax":
        return jax.grad(lambda R: func(R, *x[1:])[0][0])(jax.numpy.asarray(x[0]))
    R = torch.tensor(x[0], requires_grad=True)
    return func(R, *(torch.tensor(a) for a in x[1:]))


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
    # the orbit integration in C cannot be differentiated
    with pytest.raises(NotImplementedError, match="cannot be traced or differentiated"):
        _differentiate(backend_name, aAA, _X)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_actionAngleAdiabaticApproxInverse_backend(backend_name):
    from galpy.actionAngle import actionAngleAdiabaticApproxInverse

    aAA = _setup()

    def setup_inverse():
        return actionAngleAdiabaticApproxInverse(
            pot=NFWPotential(a=2.0, normalize=1.0),
            aAI=actionAngleIsochroneInverse(b=0.5),
            aA=actionAngleIsochrone(b=0.5),
            tintJ=20.0,
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
    with pytest.raises(NotImplementedError, match="cannot be traced or differentiated"):
        _differentiate(backend_name, aAAI, ja)
