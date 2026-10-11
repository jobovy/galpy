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

from galpy.backend import as_numpy, is_backend_array
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


# A backend TARGET AXIS a (a differentiated zvec) with a numpy v: the callers
# (EllipsoidalPotential, RotateAndTiltWrapperPotential) rotate the numpy
# [[0,0,1]] onto a parameter-built axis. The axis is parametrized by an angle,
# a = (sin t cos p, sin t sin p, cos t), so the gradient is w.r.t. the angle.
_T0, _P0 = 0.7, 0.4
_VZ = numpy.array([[0.0, 0.0, 1.0]])


def _axis(xp, t, as_list=False):
    a = [
        xp.sin(t) * numpy.cos(_P0),
        xp.sin(t) * numpy.sin(_P0),
        xp.cos(t),
    ]
    return a if as_list else xp.stack(a)


def _rot_of_angle_fd(inv):
    def f(t):
        return _rotate_to_arbitrary_vector(_VZ, _axis(numpy, t), inv=inv)[0]

    # Richardson-extrapolated central difference: O(h^4)
    h = 1e-3
    d1 = (f(_T0 + h) - f(_T0 - h)) / (2 * h)
    d2 = (f(_T0 + h / 2) - f(_T0 - h / 2)) / h
    return (4 * d2 - d1) / 3


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("inv", [False, True])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotate_backend_axis_value_parity(backend_name, inv, as_list):
    xp = jnp if backend_name == "jax" else torch
    ref = _rotate_to_arbitrary_vector(_VZ, _axis(numpy, _T0), inv=inv)
    got = _rotate_to_arbitrary_vector(
        _VZ, _axis(xp, xp.asarray(_T0), as_list=as_list), inv=inv
    )
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("inv", [False, True])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotate_grad_wrt_axis_angle(backend_name, inv, as_list):
    # the Jacobian of all 9 matrix entries w.r.t. the axis angle
    fd = _rot_of_angle_fd(inv)
    if backend_name == "jax":

        def f(t):
            return _rotate_to_arbitrary_vector(
                _VZ, _axis(jnp, t, as_list=as_list), inv=inv
            )[0]

        t0 = jnp.asarray(_T0)
        for g in (jax.jacfwd(f)(t0), jax.jit(jax.jacrev(f))(t0)):
            numpy.testing.assert_allclose(numpy.asarray(g), fd, rtol=1e-9, atol=1e-11)
    else:
        t = torch.tensor(_T0, requires_grad=True)
        out = _rotate_to_arbitrary_vector(
            _VZ, _axis(torch, t, as_list=as_list), inv=inv
        )[0]
        g = numpy.array(
            [
                float(torch.autograd.grad(out.reshape(-1)[i], t, retain_graph=True)[0])
                for i in range(9)
            ]
        ).reshape(3, 3)
        numpy.testing.assert_allclose(g, fd, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(jax is None, reason="needs jax")
def test_rotate_axis_angle_vmap_jit():
    # a batch of axis angles under vmap(jit): what a fit over orientations uses
    ts = numpy.array([0.3, _T0, 1.2])
    f = jax.jit(jax.vmap(lambda t: _rotate_to_arbitrary_vector(_VZ, _axis(jnp, t))[0]))
    got = numpy.asarray(f(jnp.asarray(ts)))
    for t, g in zip(ts, got):
        ref = _rotate_to_arbitrary_vector(_VZ, _axis(numpy, t))[0]
        numpy.testing.assert_allclose(g, ref, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rotate_backend_axis_under_forced_backend(backend_name):
    # FORCED mode: a Python-float angle under use(..., force=True) builds a
    # forced-backend axis; the rotation follows it onto the backend
    from galpy.backend import get_namespace, use

    ref = _rotate_to_arbitrary_vector(_VZ, _axis(numpy, _T0))
    with use(backend_name, force=True):
        xp = get_namespace()
        got = _rotate_to_arbitrary_vector(_VZ, _axis(xp, xp.asarray(_T0)))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-14, atol=1e-15)


@pytest.mark.skipif(jax is None, reason="needs jax")
@pytest.mark.parametrize("kick", ["plummer", "hernquist"])
def test_impulse_kicks_under_jit(kick):
    # the closed-form impulse kicks rotate onto a backend y-axis built inside
    # the trace; host-converting that axis made them un-jittable
    import importlib

    # galpy.df.streamgapdf is the CLASS; the kicks live in the submodule
    fn = getattr(
        importlib.import_module("galpy.df.streamgapdf"), f"impulse_deltav_{kick}"
    )
    v = numpy.array([[3.4, 1.9, 0.0], [1.0, 0.2, -0.3]])
    y = numpy.array([0.1, -0.4])
    w = numpy.array([0.0, 1.0, 0.0])
    ref = fn(v, y, 0.08, w, 0.003, 0.078)
    jf = jax.jit(lambda v, y, b: fn(v, y, b, jnp.asarray(w), 0.003, 0.078))
    got = numpy.asarray(jf(jnp.asarray(v), jnp.asarray(y), 0.08))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-16)

    # and d(sum dv^2)/db under jit vs Richardson FD of the numpy path
    def s(b):
        return numpy.sum(fn(v, y, b, w, 0.003, 0.078) ** 2)

    h = 1e-4
    d1 = (s(0.08 + h) - s(0.08 - h)) / (2 * h)
    d2 = (s(0.08 + h / 2) - s(0.08 - h / 2)) / h
    fd = (4 * d2 - d1) / 3
    g = jax.jit(
        jax.grad(
            lambda b: jnp.sum(
                fn(jnp.asarray(v), jnp.asarray(y), b, jnp.asarray(w), 0.003, 0.078) ** 2
            )
        )
    )(0.08)
    numpy.testing.assert_allclose(float(g), fd, rtol=1e-8)
