###############################################################################
# test_backend_staeckel_angle_grad.py: c=True C-native Staeckel ANGLE gradients
# (#131 PR-B). actionsFreqsAngles with a jax/torch input now returns
# differentiable angle{r,phi,z} via the fused C angle Jacobian (N,3,5): the
# angle rows compose PR-A's action Hessians (dOmega/dcoord, d(dI3dJ)/dcoord)
# through the same dP/dcoord chain, PLUS the angle-specific current-position
# boundary term [f(ux)/sqrt(S(ux))]*dux/dcoord over the partial integrals.
# phi enters analytically (d(angle_phi)/dphi == 1). First-order only. numpy path
# byte-identical. Mirrors test_backend_staeckel_freq_grad.py (the freq grads).
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

from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MiyamotoNagaiPotential

_MP = MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.0375)
_DELTA, _ORDER = 0.45, 10
_AAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True)
# (R,vR,vT,z,vz,phi). Interior orbits are used for grad-vs-FD; the vR0/vz0/z0
# edges (near a confocal turning point) are checked for finiteness/guard only
# (there S(ux)->0 makes the boundary factor large; the c=False AD reference can
# itself be NaN at z=0, so those are not FD-comparable -- guarded to 0 in C).
_ORBITS = {
    "generic": (1.0, 0.2, 1.1, 0.1, 0.15, 0.3),
    "eccentric": (1.2, 0.35, 0.85, 0.25, -0.2, -0.7),
}
_EDGE_ORBITS = {
    "edge_vR0": (1.0, 0.0, 1.1, 0.1, 0.15, 0.3),
    "edge_vz0": (1.0, 0.2, 1.1, 0.1, 0.0, 0.3),
    "edge_z0": (1.0, 0.2, 1.1, 0.0, 0.15, 0.3),
}


def _wrap(d):
    # unwrap a mod-2pi angle DIFFERENCE onto (-pi,pi] so central FD across the
    # [0,2pi) seam is correct (the wrap has unit derivative a.e., so AD == this).
    while d > numpy.pi:
        d -= 2.0 * numpy.pi
    while d <= -numpy.pi:
        d += 2.0 * numpy.pi
    return d


def _np_angles(orbit):
    out = _AAS.actionsFreqsAngles(*[numpy.array([c]) for c in orbit])
    return float(out[6][0]), float(out[7][0]), float(out[8][0])


_FD_CACHE = {}


def _fd_angle_grad(orbit, eps=1e-6):
    # central FD of the c=True numpy angles over the 6 coords (R,vR,vT,z,vz,phi).
    if orbit in _FD_CACHE:
        return _FD_CACHE[orbit]
    g = [[], [], []]  # d(angle_r,angle_phi,angle_z) over the 6 coords
    for i in range(6):
        up = list(orbit)
        dn = list(orbit)
        up[i] += eps
        dn[i] -= eps
        fu = _np_angles(tuple(up))
        fd = _np_angles(tuple(dn))
        for k in range(3):
            g[k].append(_wrap(fu[k] - fd[k]) / (2.0 * eps))
    _FD_CACHE[orbit] = g
    return g


def _backend_angle_grads(backend, orbit, useu0=False):
    aAS = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True, useu0=useu0)
    if backend == "jax":
        args = [jnp.asarray([x]) for x in orbit]
        return [
            [
                float(g[0])
                for g in jax.grad(
                    lambda *a: jnp.sum(aAS.actionsFreqsAngles(*a)[6 + k]),
                    argnums=(0, 1, 2, 3, 4, 5),
                )(*args)
            ]
            for k in range(3)
        ]
    args = [torch.tensor([x], requires_grad=True) for x in orbit]
    out = aAS.actionsFreqsAngles(*args)
    # allow_unused: angle_r/angle_z do not depend on phi (the 6th input), so its
    # grad is legitimately None -> 0 (jax returns 0 for such; torch needs this).
    return [
        [
            float(g[0]) if g is not None else 0.0
            for g in torch.autograd.grad(
                out[6 + k].sum(), args, retain_graph=True, allow_unused=True
            )
        ]
        for k in range(3)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_ORBITS))
def test_staeckel_angle_grad_vs_fd(backend, orbit):
    # d(angle{r,phi,z})/d(R,vR,vT,z,vz,phi) via c=True backend AD vs numpy central
    # FD, on interior orbits. Residual is the GL-vs-fixed_quad quadrature + FD
    # floor; pre-#131-PR-B the angles were stop_gradient'd (NO gradient).
    coords = _ORBITS[orbit]
    fd = _fd_angle_grad(coords)
    g = _backend_angle_grads(backend, coords)
    for k in range(3):
        numpy.testing.assert_allclose(g[k], fd[k], rtol=3e-3, atol=3e-5)
        assert numpy.all(numpy.isfinite(g[k]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_angle_value_c_true_matches_numpy(backend):
    # the c=True backend angle VALUES equal the numpy c=True values (same C path)
    arr = jnp.asarray if backend == "jax" else (lambda x: torch.tensor(x))
    for orbit in list(_ORBITS.values()) + list(_EDGE_ORBITS.values()):
        ref = _np_angles(orbit)
        out = _AAS.actionsFreqsAngles(*[arr([x]) for x in orbit])
        got = [float(numpy.asarray(out[6 + k][0])) for k in range(3)]
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_angle_phi_dphi_is_one(backend):
    # angle_phi = raw(R..vz) + phi (mod 2pi) -> d(angle_phi)/dphi == 1 exactly,
    # and no other angle depends on phi.
    for orbit in _ORBITS.values():
        g = _backend_angle_grads(backend, orbit)
        # g[1] = d(angle_phi)/d(coords); phi is the 6th coord (index 5)
        numpy.testing.assert_allclose(g[1][5], 1.0, rtol=1e-10, atol=1e-10)
        # angle_r, angle_z do not depend on phi
        numpy.testing.assert_allclose(g[0][5], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(g[2][5], 0.0, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("orbit", list(_EDGE_ORBITS))
def test_staeckel_angle_grad_finite_edges(backend, orbit):
    # near-turning-point edge orbits (vR=0/vz=0/z=0): the angle grad must be
    # FINITE (guarded, not NaN/inf) -- the C zeros the angle rows where S(pos)->0
    # per the AA turning-point-edge convention.
    g = _backend_angle_grads(backend, _EDGE_ORBITS[orbit])
    for k in range(3):
        assert numpy.all(numpy.isfinite(g[k])), f"non-finite angle grad row {k}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_staeckel_angle_grad_useu0(backend):
    # useu0=True: the reference u0=calcu0(E,Lz) is coordinate-dependent, so the C
    # Jacobian adds the du0/dx term. The FD reference must use the SAME useu0
    # numpy path (angles are gauge-invariant only up to the u0 quadrature offset).
    coords = _ORBITS["generic"]
    eps = 1e-6
    aASU = actionAngleStaeckel(pot=_MP, delta=_DELTA, c=True, useu0=True)
    fd = [[], [], []]
    for i in range(6):
        up = list(coords)
        dn = list(coords)
        up[i] += eps
        dn[i] -= eps
        fu = aASU.actionsFreqsAngles(*[numpy.array([c]) for c in up])
        fdn = aASU.actionsFreqsAngles(*[numpy.array([c]) for c in dn])
        for k in range(3):
            fd[k].append(
                _wrap(float(fu[6 + k][0]) - float(fdn[6 + k][0])) / (2.0 * eps)
            )
    g = _backend_angle_grads(backend, coords, useu0=True)
    for k in range(3):
        numpy.testing.assert_allclose(g[k], fd[k], rtol=3e-3, atol=3e-5)
