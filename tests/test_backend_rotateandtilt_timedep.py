###############################################################################
# test_backend_rotateandtilt_timedep.py: the time-dependent center= and
# orientation= of RotateAndTiltWrapperPotential (and the non-axisymmetric
# MovingObjectPotential built on them) on the jax/torch backends:
#   1. numpy / jax / torch value parity for every method, with numpy tables;
#   2. tables given as backend arrays (center path, quaternions, rotation
#      matrices) give the numpy tables' values;
#   3. under jax.jit, the center's and the orientation's tables and the wrapped
#      SCF potential's coefficients may all be traced (lmc-perturbation's
#      handoff 20: a deformed LMC carried on a traced track), and gradients
#      through them agree with finite differences; the same for torch autograd.
###############################################################################
import numpy
import pytest

from galpy import potential
from galpy.backend import as_numpy
from galpy.orbit import Orbit
from galpy.potential import (
    evaluatephitorques,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluateRphiderivs,
    evaluateRzderivs,
    evaluatez2derivs,
    evaluatezforces,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
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

_FUNCS = (
    evaluatePotentials,
    evaluateRforces,
    evaluatezforces,
    evaluatephitorques,
    evaluateR2derivs,
    evaluatez2derivs,
    evaluateRzderivs,
    evaluateRphiderivs,
)
_POINTS = ((0.9, 0.1, 0.3, 0.37), (1.4, -0.2, 2.1, 1.21), (0.6, 0.3, -1.0, -0.6))

_A = numpy.zeros((2, 3, 3))
_A[0, 0, 0] = 1.0
_A[0, 2, 2] = 0.1
_A[0, 2, 1] = 0.03
_A[1, 2, 0] = 0.02
_TT = numpy.linspace(-1.0, 2.0, 31)
_XYZ = numpy.array([0.3 * numpy.cos(_TT), 0.3 * numpy.sin(_TT), 0.1 * _TT]).T
_ANG = 0.9 * _TT + 0.2 * _TT**2
_AXIS = numpy.array([1.0, 2.0, 2.0]) / 3.0
_Q = numpy.array([numpy.cos(_ANG / 2.0)] + [numpy.sin(_ANG / 2.0) * a for a in _AXIS]).T


def _rotmats(q):
    w, x, y, z = q.T
    return numpy.moveaxis(
        numpy.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        ),
        -1,
        0,
    )


def _wrapper(A=_A, xyz=_XYZ, q=_Q):
    return potential.RotateAndTiltWrapperPotential(
        pot=potential.SCFPotential(Acos=A, Asin=0.5 * A, a=1.0),
        center=(_TT, xyz),
        orientation=(_TT, q),
        offset=[0.05, -0.02, 0.01],
    )


_O3 = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 1.0])
_O3.integrate(numpy.linspace(-1.0, 2.0, 301), potential.MWPotential2014)
_MOP = potential.MovingObjectPotential(
    _O3, pot=potential.SCFPotential(amp=0.1, Acos=_A, Asin=0.5 * _A, a=0.2)
)


def _arr(backend, x):
    if backend == "jax":
        return jnp.asarray(x)
    if backend == "torch":
        return torch.as_tensor(x, dtype=torch.float64)
    return x


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which", ["wrapper", "movingobject"])
def test_value_parity(backend, which):
    # numpy tables evaluated at backend coordinates: the numpy values
    pot = _wrapper() if which == "wrapper" else _MOP
    for R, z, phi, t in _POINTS:
        for f in _FUNCS:
            want = f(pot, R, z, phi=phi, t=t)
            got = f(
                pot,
                *(_arr(backend, v) for v in (R, z)),
                phi=_arr(backend, phi),
                t=_arr(backend, t),
            )
            assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12 * max(
                1.0, numpy.fabs(want)
            ), f"{which} {f.__name__} on {backend} does not agree with numpy"
    return None


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("form", ["quaternions", "matrices"])
def test_backend_tables(backend, form):
    # tables given as backend arrays (splines built in the backend) give the
    # numpy tables' values
    pot_np = _wrapper()
    q = _Q if form == "quaternions" else _rotmats(_Q)
    pot_bk = _wrapper(xyz=_arr(backend, _XYZ), q=_arr(backend, q))
    for R, z, phi, t in _POINTS:
        for f in _FUNCS:
            want = f(pot_np, R, z, phi=phi, t=t)
            got = f(
                pot_bk,
                *(_arr(backend, v) for v in (R, z)),
                phi=_arr(backend, phi),
                t=_arr(backend, t),
            )
            assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12 * max(
                1.0, numpy.fabs(want)
            ), (
                f"{f.__name__} with {form} backend tables on {backend} does not agree with numpy tables"
            )
    return None


def _fd_grad(fun, x, h=1e-6, valid=None):
    # central finite-difference gradient of a scalar numpy function (zero
    # outside of the valid entries)
    g = numpy.zeros_like(x)
    for idx in numpy.ndindex(x.shape):
        if valid is not None and not valid(idx):
            continue
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        g[idx] = (fun(xp) - fun(xm)) / (2.0 * h)
    return g


def _loss(A, xyz, q, rforce=False):
    # the potential (or Rforce) of a moving, turning SCF at two points and
    # times
    pot = _wrapper(A=A, xyz=xyz, q=q)
    f = evaluateRforces if rforce else evaluatePotentials
    return f(pot, 0.9, 0.1, phi=0.3, t=0.37) + f(pot, 1.4, -0.2, phi=2.1, t=1.21)


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("rforce", [False, True])
def test_jax_jit_traced_tables_and_coefficients_grad(rforce):
    # all of the center's and the orientation's tables and the SCF
    # coefficients traced inside jax.jit; the value is numpy's and the
    # gradients agree with finite differences
    want = _loss(_A, _XYZ, _Q, rforce=rforce)

    def jloss(A, xyz, q):
        return _loss(A, xyz, q, rforce=rforce)

    args = (jnp.asarray(_A), jnp.asarray(_XYZ), jnp.asarray(_Q))
    got = jax.jit(jloss)(*args)
    assert numpy.fabs(float(got) - want) < 1e-12, (
        "jax.jit with traced tables and coefficients does not agree with numpy"
    )
    grads = jax.jit(jax.grad(jloss, argnums=(0, 1, 2)))(*args)
    for ii, (g, x) in enumerate(zip(grads, (_A, _XYZ, _Q))):

        def fun(v, ii=ii):
            a = [_A, _XYZ, _Q]
            a[ii] = v
            return _loss(*a, rforce=rforce)

        # the SCF coefficients: only m <= l enter the expansion
        valid = (lambda idx: idx[2] <= idx[1]) if ii == 0 else None
        fd = _fd_grad(fun, x, valid=valid)
        g = numpy.asarray(g)
        if valid is not None:
            g = numpy.array(
                [g[idx] if valid(idx) else 0.0 for idx in numpy.ndindex(g.shape)]
            ).reshape(g.shape)
        assert numpy.amax(numpy.fabs(g - fd)) < 1e-7 * max(
            1.0, numpy.amax(numpy.fabs(fd))
        ), f"jax gradient wrt argument {ii} does not agree with finite differences"
    return None


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_traced_rotation_matrices():
    # the orientation as traced rotation matrices (converted to quaternions in
    # the backend)
    mats = _rotmats(_Q)

    def jloss(m):
        return _loss(_A, _XYZ, m)

    assert (
        numpy.fabs(float(jax.jit(jloss)(jnp.asarray(mats))) - _loss(_A, _XYZ, _Q))
        < 1e-12
    ), "jax.jit with traced rotation matrices does not agree with quaternions"
    g = jax.grad(jloss)(jnp.asarray(mats))
    assert numpy.all(numpy.isfinite(numpy.asarray(g)))
    return None


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_autograd_tables():
    # torch autograd through the center's and the orientation's tables agrees
    # with finite differences
    xyz = torch.tensor(_XYZ, requires_grad=True)
    q = torch.tensor(_Q, requires_grad=True)
    loss = _loss(_A, xyz, q)
    assert numpy.fabs(float(loss) - _loss(_A, _XYZ, _Q)) < 1e-12
    loss.backward()
    for g, x, ii in ((xyz.grad, _XYZ, 1), (q.grad, _Q, 2)):

        def fun(v, ii=ii):
            a = [_A, _XYZ, _Q]
            a[ii] = v
            return _loss(*a)

        fd = _fd_grad(fun, x)
        assert numpy.amax(numpy.fabs(g.numpy() - fd)) < 1e-7 * max(
            1.0, numpy.amax(numpy.fabs(fd))
        ), "torch gradient wrt a table does not agree with finite differences"
    return None


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_constant_backend_center(backend):
    # a constant center given as a backend array (or a list holding one)
    want = potential.RotateAndTiltWrapperPotential(
        pot=potential.SCFPotential(Acos=_A, Asin=0.5 * _A, a=1.0),
        center=[0.3, -0.1, 0.2],
    )
    for center in (
        _arr(backend, numpy.array([0.3, -0.1, 0.2])),
        [_arr(backend, 0.3), -0.1, 0.2],
    ):
        got = potential.RotateAndTiltWrapperPotential(
            pot=potential.SCFPotential(Acos=_A, Asin=0.5 * _A, a=1.0),
            center=center,
        )
        for R, z, phi, t in _POINTS:
            assert (
                numpy.fabs(
                    float(
                        as_numpy(
                            evaluateRforces(
                                got,
                                _arr(backend, R),
                                _arr(backend, z),
                                phi=_arr(backend, phi),
                                t=t,
                            )
                        )
                    )
                    - evaluateRforces(want, R, z, phi=phi, t=t)
                )
                < 1e-12
            ), "A backend constant center does not agree with a numpy one"
    return None


@pytest.mark.parametrize("backend", BACKENDS)
def test_orbit_center_held_outside(backend):
    # an Orbit center (_MOP's) is held at its end positions outside of its
    # integration times, as in C, on every backend and at a traced jax time
    for t_out, t_end in ((-1.5, -1.0), (2.6, 2.0)):
        want = evaluateRforces(_MOP, 1.1, 0.2, phi=0.3, t=t_end)

        def rforce(t):
            return evaluateRforces(
                _MOP,
                *(_arr(backend, v) for v in (1.1, 0.2)),
                phi=_arr(backend, 0.3),
                t=t,
            )

        gots = [rforce(_arr(backend, t_out))]
        if backend == "jax":
            gots.append(jax.jit(rforce)(jnp.asarray(t_out)))
        for got in gots:
            assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12, (
                f"An Orbit center on {backend} is not held outside of its integration times"
            )
    return None


def _ns(v):
    # the namespace of v (numpy for numpy and Python inputs)
    from galpy.backend import get_namespace, is_backend_array

    return get_namespace(v) if is_backend_array(v) else numpy


def _center_func(t):
    xp = _ns(t)
    return xp.stack(
        [0.3 * xp.cos(0.7 * t), 0.2 * xp.sin(0.7 * t), 0.05 * t + 0.0 * xp.cos(t)]
    )


def _orientation_func(t):
    # a (3,3) rotation by 0.9 t about z followed by (0.3 + 0.2 t) about x
    xp = _ns(t)
    a, b = 0.9 * t, 0.3 + 0.2 * t
    ca, sa, cb, sb = xp.cos(a), xp.sin(a), xp.cos(b), xp.sin(b)
    z0, one = 0.0 * ca, 0.0 * ca + 1.0
    rz = xp.stack(
        [xp.stack([ca, sa, z0]), xp.stack([-sa, ca, z0]), xp.stack([z0, z0, one])]
    )
    rx = xp.stack(
        [xp.stack([one, z0, z0]), xp.stack([z0, cb, sb]), xp.stack([z0, -sb, cb])]
    )
    return rx @ rz


def _function_wrappers(omega=1.3):
    scf = potential.SCFPotential(amp=0.1, Acos=_A, Asin=0.5 * _A, a=0.2)
    return (
        potential.RotateAndTiltWrapperPotential(
            pot=scf, center=_center_func, orientation=_orientation_func
        ),
        potential.RotateAndTiltWrapperPotential(
            pot=scf, inclination=0.4, galaxy_pa=lambda t: omega * t + 0.2
        ),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_functions_of_time_parity(backend):
    # a center and an orientation, and an angle, given as functions of time:
    # the backends evaluate them in the backend and give numpy's values
    for pot in _function_wrappers():
        for R, z, phi, t in _POINTS:
            for f in _FUNCS:
                want = f(pot, R, z, phi=phi, t=t)
                got = f(
                    pot,
                    *(_arr(backend, v) for v in (R, z)),
                    phi=_arr(backend, phi),
                    t=_arr(backend, t),
                )
                assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12 * max(
                    1.0, numpy.fabs(want)
                ), (
                    f"{f.__name__} of a wrapper with functions of time on {backend} does not agree with numpy"
                )
    return None


def _pattern_speed_loss(omega):
    # the radial force of an inclined SCF turning with pattern speed omega,
    # at two points and times
    pot = _function_wrappers(omega=omega)[1]
    return evaluateRforces(pot, 0.9, 0.1, phi=0.3, t=0.37) + evaluateRforces(
        pot, 1.4, -0.2, phi=2.1, t=1.21
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_functions_of_time_grad(backend):
    # a parameter inside an angle that is a function of time (a pattern
    # speed) is differentiable, also traced under jax.jit, and its gradient
    # agrees with finite differences
    fd = (_pattern_speed_loss(1.3 + 1e-6) - _pattern_speed_loss(1.3 - 1e-6)) / 2e-6
    want = _pattern_speed_loss(1.3)
    if backend == "jax":
        got = jax.jit(_pattern_speed_loss)(jnp.asarray(1.3))
        grad = jax.jit(jax.grad(_pattern_speed_loss))(jnp.asarray(1.3))
    else:
        om = torch.tensor(1.3, requires_grad=True)
        got = _pattern_speed_loss(om)
        (grad,) = torch.autograd.grad(got, om)
    assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12, (
        f"A pattern speed on {backend} does not give numpy's value"
    )
    assert numpy.fabs(float(as_numpy(grad)) - fd) < 1e-7 * max(1.0, numpy.fabs(fd)), (
        f"The gradient wrt a pattern speed on {backend} does not agree with finite differences"
    )
    return None
