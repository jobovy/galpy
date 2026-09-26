###############################################################################
# test_backend_autodiff.py: galpy.backend.autodiff.graft_derivative -- a numpy
# value with a given first derivative, higher orders from a backend function.
# A deliberately INCONSISTENT toy tells the three sources apart: value 10.0,
# first derivative 2.0, higher(x) = x**3 (whose own value/derivative differ).
###############################################################################
import numpy
import pytest

from galpy.backend.autodiff import graft_derivative

pytestmark = pytest.mark.backend_managed

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

_X = 1.5
_VALUE = numpy.array([10.0, 20.0])
_DERIV = numpy.array([2.0, 4.0])


def _higher(x):
    return (
        x**3 * torch.tensor([1.0, 2.0])
        if torch.is_tensor(x)
        else x**3 * jax.numpy.array([1.0, 2.0])
    )


def _f_jax(x):
    return graft_derivative(x, _VALUE, _DERIV, _higher)


def test_graft_derivative_jax():
    numpy.testing.assert_array_equal(numpy.asarray(_f_jax(_X)), _VALUE)
    # first order: the given derivative, forward and reverse
    numpy.testing.assert_array_equal(numpy.asarray(jax.jacfwd(_f_jax)(_X)), _DERIV)
    numpy.testing.assert_array_equal(numpy.asarray(jax.jacrev(_f_jax)(_X)), _DERIV)
    # second and third order: those of higher, x^3 -> 6x, 6
    for d2 in (jax.jacfwd(jax.jacrev(_f_jax)), jax.jacrev(jax.jacrev(_f_jax))):
        numpy.testing.assert_allclose(
            numpy.asarray(d2(_X)), 6.0 * _X * numpy.array([1.0, 2.0]), rtol=1e-15
        )
    d3 = jax.jacrev(jax.jacrev(jax.jacrev(_f_jax)))
    numpy.testing.assert_allclose(numpy.asarray(d3(_X)), [6.0, 12.0], rtol=1e-15)


def test_graft_derivative_torch():
    x = torch.tensor(_X, requires_grad=True)
    y = graft_derivative(x, _VALUE, _DERIV, _higher)
    numpy.testing.assert_array_equal(y.detach().numpy(), _VALUE)
    w = torch.tensor([1.0, 0.5])
    (g,) = torch.autograd.grad((y * w).sum(), x)
    assert float(g) == 2.0 * 1.0 + 4.0 * 0.5
    # create_graph: the backward itself comes from higher, so its derivative does
    (g,) = torch.autograd.grad(
        (graft_derivative(x, _VALUE, _DERIV, _higher) * w).sum(), x, create_graph=True
    )
    numpy.testing.assert_allclose(float(g), 3.0 * _X**2 * 2.0, rtol=1e-15)
    (h,) = torch.autograd.grad(g, x, create_graph=True)
    numpy.testing.assert_allclose(float(h), 6.0 * _X * 2.0, rtol=1e-15)
    (h3,) = torch.autograd.grad(h, x)
    numpy.testing.assert_allclose(float(h3), 12.0, rtol=1e-15)
    # a leaf that does not itself require grad still gets higher's derivative
    xd = torch.tensor(_X)
    ctx_y = graft_derivative(xd.requires_grad_(), _VALUE, _DERIV, _higher)
    assert ctx_y.grad_fn is not None
