###############################################################################
# test_backend_aa_grad_probes.py: gradient-accuracy probes for the AA backend
# (jax/torch) paths beyond actionAngleStaeckel's grafted actions (PR D of the
# derivative fix; see test_backend_staeckel_grad.py for the Staeckel harness).
#
# 1) CONFIRMATIONS: Spherical dJr/d(R,vR,vT), Vertical dJ/d(x,vx) and
#    Adiabatic d(jr,jz)/d(coords) first gradients under backend AD match numpy
#    central FD (measured <= ~2e-8 relative) and are quadrature-converged.
# 2) FIXED HERE: the Adiabatic jax path used to error under jax.grad w.r.t. R
#    (_BatchedVerticalPotential concretized the batch R at construction);
#    the under_jax_trace fallback in verticalPotential.py removes it.
# 3) PHASE-2 FLAGS (strict=False xfails that flip when the derivative
#    integrands land): Staeckel FREQUENCY gradients and jax.hessian through
#    the grafted ACTIONS are today finite but wrong (divergent 1/t^2-type
#    quadratures under one extra differentiation); measured baselines in the
#    reasons below.
###############################################################################
import importlib

import numpy
import pytest

pytestmark = [
    pytest.mark.backend_managed,
    # array_api_compat's asarray-requires-grad notice (torch tensor construction).
    pytest.mark.filterwarnings("ignore:torch.asarray. unspecified requires_grad"),
]

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

from galpy.actionAngle import (
    actionAngleAdiabatic,
    actionAngleSpherical,
    actionAngleVertical,
)
from galpy.actionAngle.actionAngleStaeckel import (
    _staeckel_actions,
    _staeckel_actions_freqs,
)
from galpy.potential import (
    HernquistPotential,
    MiyamotoNagaiPotential,
    toVerticalPotential,
)

_HP = HernquistPotential(normalize=1.0, a=1.1)
_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.0375)
_DELTA, _ORDER = 0.45, 10
_ORBIT = (1.0, 0.2, 1.1, 0.1, 0.15)  # generic Staeckel/Adiabatic probe orbit

_aAS_sph = actionAngleSpherical(pot=_HP)
_aAV = actionAngleVertical(pot=toVerticalPotential(_MP, 1.0))
_aAA = actionAngleAdiabatic(pot=_MP, c=False)


def _fd(func, coords, i, eps=1e-6):
    # central finite difference of a scalar numpy-path function -- the gold
    # reference the backend AD gradients must reproduce.
    up = list(coords)
    dn = list(coords)
    up[i] += eps
    dn[i] -= eps
    return (func(*up) - func(*dn)) / (2.0 * eps)


# --------------------------------------------------------------- confirmations
_SPH_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2),
    "inward": (0.9, -0.1, 1.05, -0.15, 0.1),
}


def _sph_np_jr(*coords):
    return float(_aAS_sph(*[numpy.array([c]) for c in coords])[0][0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_SPH_ORBITS))
def test_spherical_action_grad_vs_fd(backend, orbit):
    # dJr/d(R,vR,vT) via backend AD vs numpy FD (measured <= 2e-8 relative).
    coords = _SPH_ORBITS[orbit]
    fd = [_fd(_sph_np_jr, coords, i) for i in range(3)]
    if backend == "jax":
        args = [jnp.asarray([c]) for c in coords]

        def f(i, x):
            a = list(args)
            a[i] = x
            return jnp.sum(_aAS_sph(*a)[0])

        ad = [float(jax.grad(lambda x, i=i: f(i, x))(args[i])[0]) for i in range(3)]
    else:
        args = [torch.tensor([c], requires_grad=True) for c in coords]
        g = torch.autograd.grad(_aAS_sph(*args)[0].sum(), args[:3])
        ad = [float(gg[0]) for gg in g]
    numpy.testing.assert_allclose(ad, fd, rtol=1e-3)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
def test_spherical_action_grad_convergence(backend, monkeypatch):
    # dJr/dR at the default backend GL order vs 3x the order: measured
    # relative difference 8e-15, i.e. the gradient quadrature is converged.
    coords = _SPH_ORBITS["generic"]
    sphmod = importlib.import_module("galpy.actionAngle.actionAngleSpherical")
    args = [jnp.asarray([c]) for c in coords]

    def djr_dR():
        def f(R):
            return jnp.sum(_aAS_sph(R, *args[1:])[0])

        return float(jax.grad(f)(args[0])[0])

    g_default = djr_dR()
    monkeypatch.setattr(sphmod, "_BACKEND_GL_ORDER", 3 * sphmod._BACKEND_GL_ORDER)
    g_high = djr_dR()
    numpy.testing.assert_allclose(g_default, g_high, rtol=1e-4)


_VERT_ORBITS = {"generic": (0.1, 0.15), "high": (0.25, -0.2)}


def _vert_np_J(x, vx):
    return float(_aAV(numpy.array([x]), numpy.array([vx]))[0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_VERT_ORBITS))
def test_vertical_action_grad_vs_fd(backend, orbit):
    # dJ/d(x,vx) via backend AD vs numpy FD (measured <= 8e-11 relative).
    coords = _VERT_ORBITS[orbit]
    fd = [_fd(_vert_np_J, coords, i) for i in range(2)]
    if backend == "jax":
        args = [jnp.asarray([c]) for c in coords]

        def f(i, x):
            a = list(args)
            a[i] = x
            return jnp.sum(_aAV(*a)[0])

        ad = [float(jax.grad(lambda x, i=i: f(i, x))(args[i])[0]) for i in range(2)]
    else:
        args = [torch.tensor([c], requires_grad=True) for c in coords]
        g = torch.autograd.grad(_aAV(*args).sum(), args)
        ad = [float(gg[0]) for gg in g]
    numpy.testing.assert_allclose(ad, fd, rtol=1e-3)


def _adia_np(*coords):
    out = _aAA(*[numpy.array([c]) for c in coords])
    return float(out[0][0]), float(out[2][0])


def _adia_fd(coords):
    fd_jr = [_fd(lambda *o: _adia_np(*o)[0], coords, i) for i in range(5)]
    fd_jz = [_fd(lambda *o: _adia_np(*o)[1], coords, i) for i in range(5)]
    return fd_jr, fd_jz


@pytest.mark.parametrize("backend", BACKENDS)
def test_adiabatic_action_grad_vs_fd(backend):
    # d(jr,jz)/d(coords) via backend AD vs numpy FD (measured <= 2e-8
    # relative). torch covers all five coordinates; jax covers R separately
    # (the once-erroring _BatchedVerticalPotential construction, below).
    coords = _ORBIT
    fd_jr, fd_jz = _adia_fd(coords)
    if backend == "jax":
        args = [jnp.asarray([c]) for c in coords]
        for idx, fd in ((0, fd_jr), (2, fd_jz)):
            for i in range(1, 5):

                def f(x, i=i, idx=idx):
                    a = list(args)
                    a[i] = x
                    return jnp.sum(_aAA(*a)[idx])

                ad = float(jax.grad(f)(args[i])[0])
                numpy.testing.assert_allclose(ad, fd[i], rtol=1e-3, atol=1e-8)
    else:
        args = [torch.tensor([c], requires_grad=True) for c in coords]
        out = _aAA(*args)
        # jz is exactly independent of (vR, vT): allow_unused -> 0
        gjr = torch.autograd.grad(out[0].sum(), args, retain_graph=True)
        gjz = torch.autograd.grad(out[2].sum(), args, allow_unused=True)
        gjr = [float(g[0]) for g in gjr]
        gjz = [0.0 if g is None else float(g[0]) for g in gjz]
        numpy.testing.assert_allclose(gjr, fd_jr, rtol=1e-3, atol=1e-8)
        numpy.testing.assert_allclose(gjz, fd_jz, rtol=1e-3, atol=1e-8)


# jax d/dR used to raise ConcretizationTypeError in
# _BatchedVerticalPotential.__init__ (float() on the traced batch R for the
# scalar-parent bookkeeping); fixed by the under_jax_trace fallback there.
@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
def test_adiabatic_action_grad_dR_jax(backend):
    coords = _ORBIT
    fd_jr, fd_jz = _adia_fd(coords)
    args = [jnp.asarray([c]) for c in coords]
    djr = float(jax.grad(lambda R: jnp.sum(_aAA(R, *args[1:])[0]))(args[0])[0])
    djz = float(jax.grad(lambda R: jnp.sum(_aAA(R, *args[1:])[2]))(args[0])[0])
    numpy.testing.assert_allclose([djr, djz], [fd_jr[0], fd_jz[0]], rtol=1e-3)


def test_batched_vertical_potential_numpy_R0_branch():
    # covers the numpy branch of _BatchedVerticalPotential.__init__ (R0=float(R),
    # taken for a non-backend-array R -- the forced-numpy all-backend path); the
    # backend-array branch (R0=1.0) is covered by the adiabatic grad tests above.
    from galpy.potential import MiyamotoNagaiPotential, evaluatePotentials
    from galpy.potential.verticalPotential import _BatchedVerticalPotential

    mn = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.05)
    R, z = numpy.array([0.8, 1.0, 1.3]), numpy.array([0.1, 0.15, 0.05])
    got = _BatchedVerticalPotential(mn, R)(z)
    want = [
        evaluatePotentials(mn, R[i], z[i], use_physical=False)
        - evaluatePotentials(mn, R[i], 0.0, use_physical=False)
        for i in range(3)
    ]
    numpy.testing.assert_allclose(got, want, rtol=1e-12)


# ---------------------------------------------------------- Phase-2 xfail flags
def _np_staeckel_Or(*coords):
    out = _staeckel_actions_freqs(
        numpy, *[numpy.array([c]) for c in coords], _MP, _DELTA, _ORDER
    )
    return float(out[3][0])


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
@pytest.mark.xfail(
    strict=False,
    reason="Phase 2: Staeckel FREQUENCY gradients. Omega comes from the "
    "1/sqrt(S) Jacobian panels (_staeckel_jacobian); AD differentiates the "
    "already ~1/t-singular integrand once more (S^(-3/2) ~ 1/t^2), a "
    "divergent quadrature whose GL sum is finite but wrong: measured "
    "dOmega_r/dR = 104.89 via jax.grad vs -1.9327 by FD (factor ~-54). "
    "Flips when the Phase-2 derivative integrands land.",
)
def test_staeckel_freq_grad_dR_phase2(backend):
    coords = _ORBIT
    fd = _fd(_np_staeckel_Or, coords, 0, eps=1e-5)
    args = [jnp.asarray([c]) for c in coords]

    def f(R):
        a = [R] + args[1:]
        return jnp.sum(_staeckel_actions_freqs(jnp, *a, _MP, _DELTA, _ORDER)[3])

    ad = float(jax.grad(f)(args[0])[0])
    numpy.testing.assert_allclose(ad, fd, rtol=1e-3)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b == "jax"])
@pytest.mark.xfail(
    strict=False,
    reason="Phase 2: SECOND derivatives through the grafted actions. The "
    "donor t^2 gradient integrand is itself ~1/t-singular under one more "
    "differentiation, so jacfwd(grad(Jr)) w.r.t. (R,z) is finite but wrong: "
    "measured [[-278.7, -151.9], [-151.9, -96.2]] vs the "
    "FD-of-the-grafted-gradient reference [[0.592, 0.305], [0.305, 0.684]] "
    "(up to ~500x off). The FD-of-grafted-grad values are the Phase-2 "
    "baseline this must match.",
)
def test_staeckel_action_hessian_phase2(backend):
    coords = _ORBIT
    args = [jnp.asarray([c]) for c in coords]

    def jr_of_Rz(Rz):
        a = [Rz[0:1], args[1], args[2], Rz[1:2], args[4]]
        return jnp.sum(_staeckel_actions(jnp, *a, _MP, _DELTA, _ORDER)[0])

    grad_fn = jax.grad(jr_of_Rz)
    Rz0 = jnp.asarray([coords[0], coords[3]])
    H_ad = numpy.asarray(jax.jacfwd(grad_fn)(Rz0))
    # FD of the (accurate, grafted) first gradient = the Phase-2 baseline
    eps = 1e-5
    H_fd = numpy.zeros((2, 2))
    for i in range(2):
        up = numpy.array([coords[0], coords[3]])
        dn = up.copy()
        up[i] += eps
        dn[i] -= eps
        H_fd[:, i] = (
            numpy.asarray(grad_fn(jnp.asarray(up)))
            - numpy.asarray(grad_fn(jnp.asarray(dn)))
        ) / (2.0 * eps)
    numpy.testing.assert_allclose(H_ad, H_fd, rtol=1e-2)
