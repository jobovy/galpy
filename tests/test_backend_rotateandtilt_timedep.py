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


def _wrapper(A=_A, xyz=_XYZ, q=_Q, tt=_TT):
    return potential.RotateAndTiltWrapperPotential(
        pot=potential.SCFPotential(Acos=A, Asin=0.5 * A, a=1.0),
        center=(tt, xyz),
        orientation=(tt, q),
        offset=[0.05, -0.02, 0.01],
    )


_O3 = Orbit([1.1, 0.1, 1.1, 0.1, 0.1, 1.0])
_O3.integrate(numpy.linspace(-1.0, 2.0, 301), potential.MWPotential2014)
_MOP = potential.MovingObjectPotential(
    _O3, pot=potential.SCFPotential(amp=0.1, Acos=_A, Asin=0.5 * _A, a=0.2)
)
_O2 = Orbit([1.1, 0.1, 1.1, 0.1])
_O2.integrate(numpy.linspace(-1.0, 2.0, 301), potential.MWPotential2014)
_PLANAR_CENTER = potential.RotateAndTiltWrapperPotential(
    pot=potential.SCFPotential(amp=0.1, Acos=_A, Asin=0.5 * _A, a=0.2),
    center=_O2,
    inclination=0.3,
    galaxy_pa=0.2,
)


def _arr(backend, x):
    if backend == "jax":
        return jnp.asarray(x)
    if backend == "torch":
        return torch.as_tensor(x, dtype=torch.float64)
    return x


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("which", ["wrapper", "movingobject", "planar_orbit_center"])
def test_value_parity(backend, which):
    # numpy tables (and a planar Orbit center) evaluated at backend
    # coordinates: the numpy values
    pot = {"wrapper": _wrapper(), "movingobject": _MOP}.get(which, _PLANAR_CENTER)
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
@pytest.mark.parametrize("form", ["quaternions", "matrices", "times"])
def test_backend_tables(backend, form):
    # tables given as backend arrays (splines built in the backend), or numpy
    # tables at backend times, give the numpy tables' values
    pot_np = _wrapper()
    if form == "times":
        pot_bk = _wrapper(tt=_arr(backend, _TT))
    else:
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


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_backend_constant_orientation_and_decreasing_tables(backend):
    # backend tables with decreasing times (reversed in the backend) and a
    # constant orientation= given as a backend rotation matrix or quaternion
    # give the numpy values
    from galpy.potential.RotateAndTiltWrapperPotential import _matrix_to_quaternion

    scf = potential.SCFPotential(Acos=_A, Asin=0.5 * _A, a=1.0)
    angles = potential.RotateAndTiltWrapperPotential(
        pot=scf, inclination=0.4, galaxy_pa=0.3, sky_pa=-0.7
    )
    rot = angles._rot
    pairs = (
        (
            _wrapper(),
            _wrapper(
                xyz=_arr(backend, _XYZ[::-1].copy()),
                q=_arr(backend, _Q[::-1].copy()),
                tt=_arr(backend, _TT[::-1].copy()),
            ),
        ),
        (
            angles,
            potential.RotateAndTiltWrapperPotential(
                pot=scf, orientation=_arr(backend, rot)
            ),
        ),
        (
            angles,
            potential.RotateAndTiltWrapperPotential(
                pot=scf, orientation=_arr(backend, _matrix_to_quaternion(rot[None])[0])
            ),
        ),
    )
    for want_pot, got_pot in pairs:
        for R, z, phi, t in _POINTS:
            for f in _FUNCS:
                want = f(want_pot, R, z, phi=phi, t=t)
                got = f(
                    got_pot,
                    *(_arr(backend, v) for v in (R, z)),
                    phi=_arr(backend, phi),
                    t=_arr(backend, t),
                )
                assert numpy.fabs(float(as_numpy(got)) - want) < 1e-12 * max(
                    1.0, numpy.fabs(want)
                ), (
                    f"{f.__name__} with a decreasing backend table or a constant backend orientation= on {backend} does not agree with numpy"
                )
    return None


def _per_point_cases():
    # a moving and turning bar from tables, a bar turning through angles that
    # are functions of time, and a non-axisymmetric MovingObjectPotential
    bar = potential.SoftenedNeedleBarPotential(
        amp=0.05, a=0.4, b=0.0, c=0.1, pa=0.0, omegab=0.0
    )
    return {
        "tables": potential.RotateAndTiltWrapperPotential(
            pot=bar, center=(_TT, _XYZ), orientation=(_TT, _Q)
        ),
        "functions": potential.RotateAndTiltWrapperPotential(
            pot=bar, galaxy_pa=lambda t: 1.3 * t, inclination=0.2
        ),
        "movingobject": potential.MovingObjectPotential(
            _O3,
            pot=potential.SoftenedNeedleBarPotential(
                amp=0.02, a=0.05, b=0.0, c=0.01, omegab=4.0
            ),
        ),
    }


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_per_point_times(backend):
    # One time per point (as torchode, which steps every orbit at its own
    # time, evaluates the forces): each point with the center and orientation
    # at its own time, as a loop over the points does; equal times reduce to
    # one
    R, z, phi = (
        numpy.array([0.9, 1.2, 1.6]),
        numpy.array([0.1, -0.2, 0.05]),
        numpy.array([0.3, 1.1, -0.7]),
    )
    for tt in (numpy.array([0.2, 0.9, 1.7]), numpy.array([0.9, 0.9, 0.9])):
        for name, pot in _per_point_cases().items():
            for f in _FUNCS + (potential.evaluateDensities,):
                want = numpy.array(
                    [f(pot, R[i], z[i], phi=phi[i], t=tt[i]) for i in range(3)]
                )
                got = as_numpy(
                    f(
                        pot,
                        _arr(backend, R),
                        _arr(backend, z),
                        phi=_arr(backend, phi),
                        t=_arr(backend, tt),
                    )
                )
                assert numpy.amax(numpy.fabs(got - want)) < 1e-12 * max(
                    1.0, numpy.amax(numpy.fabs(want))
                ), (
                    f"{f.__name__} of {name} at one time per point on {backend} does not agree with a loop over the points"
                )
    return None


_ICS = [
    [1.1, 0.1, 1.05, 0.05, 0.05, 0.3],
    [1.0, 0.0, 1.0, 0.1, 0.0, 1.0],
    [1.3, -0.1, 0.9, 0.0, 0.1, 2.0],
]
_TS = numpy.linspace(0.0, 1.0, 11)


def _c_orbits(pot):
    o = Orbit(_ICS)
    o.integrate(_TS, potential.MWPotential2014 + pot, method="dop853_c")
    return numpy.array([o.x(_TS), o.y(_TS), o.z(_TS)])


def _c_fd_center_scale(h=1e-4):
    # d x(t_end) / d (scale of the center's table) of the first orbit, by
    # central finite differences of C orbits (which agree with the in-backend
    # integrators to ~1e-11, far below what the step needs)
    def x(scale):
        pot = potential.RotateAndTiltWrapperPotential(
            pot=_per_point_cases()["tables"]._pot,
            center=(_TT, scale * _XYZ),
            orientation=(_TT, _Q),
        )
        o = Orbit(_ICS[0])
        o.integrate(_TS, potential.MWPotential2014 + pot, method="dop853_c")
        return o.x(_TS[-1])

    return (x(1.0 + h) - x(1.0 - h)) / (2.0 * h)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torchode_moving_turning():
    # torchode integrates orbits through moving and turning potentials as C
    # does, each orbit at its own time, and differentiates through them with
    # respect to the center's table
    def T(x):
        return torch.as_tensor(numpy.asarray(x), dtype=torch.float64)

    for name, pot in _per_point_cases().items():
        o = Orbit(T(_ICS))
        o.integrate(T(_TS), potential.MWPotential2014 + pot, method="torchode")
        got = numpy.array([as_numpy(f(T(_TS))) for f in (o.x, o.y, o.z)])
        assert numpy.amax(numpy.fabs(got - _c_orbits(pot))) < 1e-9, (
            f"torchode orbits in {name} do not agree with C"
        )

    def final_x(scale):
        pot = potential.RotateAndTiltWrapperPotential(
            pot=_per_point_cases()["tables"]._pot,
            center=(_TT, scale * T(_XYZ)),
            orientation=(_TT, T(_Q)),
        )
        o = Orbit(T(_ICS[0]))
        o.integrate(T(_TS), potential.MWPotential2014 + pot, method="torchode")
        return o.x(T(_TS[-1]))

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    final_x(scale).backward()
    fd = _c_fd_center_scale()
    assert numpy.fabs(float(scale.grad) - fd) < 1e-6 * numpy.fabs(fd), (
        "The torchode gradient through a moving potential does not agree with finite differences"
    )
    return None


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_diffrax_moving_turning():
    # diffrax integrates orbits through a moving and turning potential as C
    # does, and differentiates through them with respect to the center's table
    pot = _per_point_cases()["tables"]
    o = Orbit(jnp.asarray(_ICS))
    o.integrate(jnp.asarray(_TS), potential.MWPotential2014 + pot, method="diffrax")
    got = numpy.array([as_numpy(f(jnp.asarray(_TS))) for f in (o.x, o.y, o.z)])
    assert numpy.amax(numpy.fabs(got - _c_orbits(pot))) < 1e-9, (
        "diffrax orbits in a moving and turning potential do not agree with C"
    )

    def final_x(scale):
        p = potential.RotateAndTiltWrapperPotential(
            pot=pot._pot,
            center=(_TT, scale * jnp.asarray(_XYZ)),
            orientation=(_TT, jnp.asarray(_Q)),
        )
        oo = Orbit(jnp.asarray(_ICS[0]))
        oo.integrate(jnp.asarray(_TS), potential.MWPotential2014 + p, method="diffrax")
        return oo.x(jnp.asarray(_TS[-1]))

    g = float(jax.grad(final_x)(1.0))
    fd = _c_fd_center_scale()
    assert numpy.fabs(g - fd) < 1e-6 * numpy.fabs(fd), (
        "The diffrax gradient through a moving potential does not agree with finite differences"
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


def _center_shape_wrappers():
    # centers returned as a (3,1) array (as [o.x(t), o.y(t), o.z(t)] of an
    # Orbit) and as a list of scalars (backend scalars at a backend t)
    scf = potential.SCFPotential(amp=0.1, Acos=_A, Asin=0.5 * _A, a=0.2)
    return (
        potential.RotateAndTiltWrapperPotential(
            pot=scf, center=lambda t: _center_func(t)[:, None]
        ),
        potential.RotateAndTiltWrapperPotential(
            pot=scf,
            center=lambda t: [
                0.3 * _ns(t).cos(0.7 * t),
                0.2 * _ns(t).sin(0.7 * t),
                0.05,
            ],
        ),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_functions_of_time_parity(backend):
    # a center and an orientation, and an angle, given as functions of time:
    # the backends evaluate them in the backend and give numpy's values
    for pot in _function_wrappers() + _center_shape_wrappers():
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


# rotation by -90 deg about x: the wrapped mid-plane contains the vertical line
_EDGE_ON = numpy.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_SPLIT_CASES = {
    "center": dict(center=_center_func),
    "center+offset": dict(center=_center_func, offset=[0.05, -0.02, 0.01]),
    "center+tilt": dict(center=_center_func, zvec=[0.36, 0.48, 0.8], galaxy_pa=0.4),
    "orientation": dict(orientation=_orientation_func),
    "tables": dict(
        center=(_TT, _XYZ), orientation=(_TT, _Q), offset=[0.05, -0.02, 0.01]
    ),
    "edge-on": dict(orientation=lambda t: _EDGE_ON),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(_SPLIT_CASES))
def test_vertical_quad_split_time_dependent(backend, case):
    # the vertical quadratures' split is where the wrapped potential's own z'
    # vanishes at time t (0 edge-on), on numpy and the backends
    pot = potential.RotateAndTiltWrapperPotential(
        pot=potential.MiyamotoNagaiPotential(a=0.5, b=0.05), **_SPLIT_CASES[case]
    )
    for R, _, phi, t in _POINTS:
        want = float(pot._vertical_quad_split(R, phi=phi, t=t))
        got = float(
            as_numpy(
                pot._vertical_quad_split(
                    _arr(backend, R), phi=_arr(backend, phi), t=_arr(backend, t)
                )
            )
        )
        assert numpy.fabs(got - want) < 1e-13, (
            f"The vertical split of a time-dependent wrapper on {backend} does not agree with numpy"
        )
        if case == "edge-on":
            assert want == 0.0
        else:
            zp = float(pot._rect_transformed(numpy, R, want, phi, t=t)[2])
            assert numpy.fabs(zp) < 1e-12, f"z' = {zp:e} at the split {want:e}"
    return None
