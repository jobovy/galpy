###############################################################################
# test_backend_util.py: multi-backend tests for galpy.util helpers that the
# stream DFs / rotated potentials sit on -- currently _rotate_to_arbitrary_vector,
# the batched "rotate v onto unit vector a" matrix builder used by
# streamspraydf._setup_rot, streamgapdf, EllipsoidalPotential, and
# RotateAndTiltWrapperPotential.
#
# It used numpy.tile / numpy.cross / a preallocated numpy.empty with row-wise
# in-place assignment / boolean masked assignment, all of which reject torch
# tensors. The migration keeps the numpy path byte-identical (a verbatim branch)
# and adds an out-of-place, differentiable backend branch whose rotaxis-norm
# denominator is guarded so a v parallel to a does not NaN-poison gradients.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.util import _rotate_to_arbitrary_vector

pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
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

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

_rng = numpy.random.default_rng(314159)
# generic rows + one nearly-aligned and one nearly-anti-aligned with a to hit
# the |costheta -+ 1| < 1e-10 masked branches.
_V = numpy.vstack([_rng.normal(size=(5, 3)), [1e-13, 1.0, 1e-13], [1e-13, -1.0, 1e-13]])
_A = numpy.array([0.0, 1.0, 0.0])


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("inv", [False, True])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_rotate_value_parity(backend_name, inv):
    ref = _rotate_to_arbitrary_vector(numpy.asarray(_V), numpy.asarray(_A), inv=inv)
    got = _rotate_to_arbitrary_vector(
        _asarray(backend_name, _V), _asarray(backend_name, _A), inv=inv
    )
    numpy.testing.assert_allclose(
        as_numpy(got),
        ref,
        rtol=1e-12,
        atol=1e-13,
        err_msg=f"inv={inv} ({backend_name})",
    )
    # the two degenerate rows must be exactly +/- I (masked branch)
    numpy.testing.assert_allclose(as_numpy(got)[-2], numpy.eye(3), atol=1e-12)
    numpy.testing.assert_allclose(as_numpy(got)[-1], -numpy.eye(3), atol=1e-12)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_rotate_dontcutsmall_parity(backend_name):
    # _dontcutsmall=True path (used for the module-level galcen rotations); pass
    # only non-degenerate rows so both paths are finite.
    v = _V[:5]
    ref = _rotate_to_arbitrary_vector(
        numpy.asarray(v), numpy.asarray(_A), _dontcutsmall=True
    )
    got = _rotate_to_arbitrary_vector(
        _asarray(backend_name, v), _asarray(backend_name, _A), _dontcutsmall=True
    )
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_rotate_is_a_rotation(backend_name):
    # R must actually rotate each v onto |v|*a: R . v_hat == a.
    v = _V[:5]
    R = _rotate_to_arbitrary_vector(
        _asarray(backend_name, v), _asarray(backend_name, _A)
    )
    R = as_numpy(R)
    for i in range(len(v)):
        vhat = v[i] / numpy.linalg.norm(v[i])
        numpy.testing.assert_allclose(R[i] @ vhat, _A, atol=1e-10)
        # orthogonal: R R^T == I
        numpy.testing.assert_allclose(R[i] @ R[i].T, numpy.eye(3), atol=1e-10)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotate_grad_through(backend_name):
    # d out[0,0,0] / d v[0,0] must be finite (the guarded denominator prevents
    # NaN poisoning even with degenerate rows present in the batch) and match FD.
    eps = 1e-6
    vp = _V.copy()
    vp[0, 0] += eps
    vm = _V.copy()
    vm[0, 0] -= eps
    fd = (
        _rotate_to_arbitrary_vector(vp, _A)[0, 0, 0]
        - _rotate_to_arbitrary_vector(vm, _A)[0, 0, 0]
    ) / (2 * eps)
    if backend_name == "jax":

        def f(x):
            vv = jnp.asarray(_V).at[0, 0].set(x)
            return _rotate_to_arbitrary_vector(vv, jnp.asarray(_A))[0, 0, 0]

        g = float(jax.grad(f)(jnp.asarray(_V[0, 0])))
    else:
        vt = torch.tensor(_V, dtype=torch.float64, requires_grad=True)
        _rotate_to_arbitrary_vector(vt, torch.tensor(_A, dtype=torch.float64))[
            0, 0, 0
        ].backward()
        g = float(vt.grad[0, 0])
    assert not numpy.isnan(g)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotate_numpy_data_under_forced_backend(backend_name):
    # Dispatch is DATA-first: a genuine numpy v must take the byte-identical numpy
    # branch even under a forced non-numpy default (the sampler leaves feed numpy
    # arrays through here while the forced default is torch/jax). Regression for the
    # as_backend_constant(numpy dtype -> torch.asarray) crash.
    from galpy.backend import use

    ref = _rotate_to_arbitrary_vector(numpy.asarray(_V), numpy.asarray(_A))
    with use(backend_name, force=True):
        got = _rotate_to_arbitrary_vector(numpy.asarray(_V), numpy.asarray(_A))
    assert isinstance(got, numpy.ndarray)
    numpy.testing.assert_array_equal(got, ref)
