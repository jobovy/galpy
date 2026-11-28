###############################################################################
# test_backend_actionAngleStaeckelInverse.py: actionAngleStaeckelInverse on
# jax/torch -- values against the numpy path (a numpy-built family on backend
# inputs, and a family built on a backend), the derivatives in the actions and
# the angles against finite differences of the numpy path, the canonicity of
# the autodiff Jacobian (for perturbed tables and explicit tori too), the
# explicit tori's quadratic one-node family, the derivatives in the
# potential's parameters, the degenerate tori (circular, shell, planar), the
# adaptive reference curve and the target box, jit/vmap/compile, float32, and
# the construction's and evaluation's checks
###############################################################################
import contextlib
import copy
import importlib

import numpy
import pytest

from galpy.backend import as_numpy

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

# the symplectic form on (theta_R, theta_phi, theta_z, J_R, J_phi, J_z) ->
# (R, z, phi, v_R, v_z, R v_T)
_OMEGA = numpy.block(
    [[numpy.zeros((3, 3)), numpy.eye(3)], [-numpy.eye(3), numpy.zeros((3, 3))]]
)
_ANGLES = numpy.array([0.0, 0.4, 1.5, numpy.pi, 4.0, 6.0])
# in the plane (a circular orbit, the target's box) a backend evaluates the
# Staeckel model through the potential it approximates, which it equals there;
# numpy through the model's own force, which since the exact dR/du of the
# wrapper's U derivatives (#1836) agrees to round-off: measured <= 3.4e-13
_PLANE_ATOL = 1e-12
# (E, L_z, I_3) of the benign and the eccentric torus of test_actionAngle.py's
# Staeckel inverse tests (from the phase-space points [1.1, 0.3, 0.9, 0.25,
# 0.2, 0] and [1.1, 0.9, 0.35, 0.15, 0.1, 0] by the separated Hamilton-Jacobi
# equation)
_LABELS = [
    (-1.4741137603872914, 0.9900000000000001, 1.9081353408558568),
    (-1.5137008868502326, 0.385, 1.6040112451580693),
]
# tori of the interpolated family: generic, retrograde, on the shell face, on
# the planar face, and the circular orbit
_TORI = [
    (0.06, 0.93, 0.10),
    (0.15, -1.21, 0.03),
    (0.0, 0.8, 0.2),
    (0.02, 1.0, 0.0),
    (0.0, 1.1, 0.0),
]


def _arr(backend, x, grad=False):
    if backend == "jax":
        return jnp.asarray(x)
    return torch.tensor(x, requires_grad=grad)


def _is_backend(backend, x):
    return type(x).__module__.split(".")[0] == (
        "jaxlib" if backend == "jax" else "torch"
    )


def _pot(amp=4.0):
    from galpy.potential import KuzminKutuzovStaeckelPotential

    return KuzminKutuzovStaeckelPotential(amp=amp, ac=5.0, Delta=1.3)


def _interp(pot=None, **kw):
    from galpy.actionAngle import actionAngleStaeckelInverse

    kw = {
        "setup_interp": True,
        "Rmin": 0.7,
        "Rmax": 1.6,
        "Rinf": 3.5,
        "nE": 5,
        "nLz": 5,
        "nI3": 5,
        "mm_npt": 48,
        "progressbar": False,
        **kw,
    }
    return actionAngleStaeckelInverse(pot=_pot() if pot is None else pot, **kw)


def _explicit(pot=None, labels=_LABELS, **kw):
    from galpy.actionAngle import actionAngleStaeckelInverse

    kw = {
        "Es": [l[0] for l in labels],
        "Lzs": [l[1] for l in labels],
        "I3s": [l[2] for l in labels],
        "mm_npt": 48,
        "progressbar": False,
        **kw,
    }
    return actionAngleStaeckelInverse(pot=_pot() if pot is None else pot, **kw)


@pytest.fixture(scope="module")
def interp():
    return _interp()


@pytest.fixture(scope="module")
def explicit():
    return _explicit()


def _flat(out):
    return numpy.concatenate([numpy.ravel(as_numpy(o)) for o in out])


def _qp(xv):
    R, vR, vT, z, vz, phi = xv[:6]
    return [R[0], z[0], phi[0], vR[0], vz[0], R[0] * vT[0]]


def _jacobian(backend, aA, x0):
    """The 6x6 autodiff Jacobian of (theta, J) -> (q, p) at x0 = (angles,
    actions)"""
    if backend == "jax":

        def f(x):
            return jnp.stack(_qp(aA(x[3], x[4], x[5], x[0:1], x[1:2], x[2:3])))

        return numpy.asarray(jax.jacfwd(f)(jnp.asarray(x0)))

    def g(x):
        return torch.stack(_qp(aA(x[3], x[4], x[5], x[0:1], x[1:2], x[2:3])))

    return as_numpy(torch.autograd.functional.jacobian(g, torch.tensor(x0)))


def _jac_freqs(backend, aA, j0):
    """The autodiff Jacobian of the frequencies in the actions"""
    if backend == "jax":
        return numpy.asarray(
            jax.jacfwd(lambda j: jnp.stack(aA.Freqs(j[0], j[1], j[2])))(jnp.asarray(j0))
        )
    return as_numpy(
        torch.autograd.functional.jacobian(
            lambda j: torch.stack(aA.Freqs(j[0], j[1], j[2])), torch.tensor(j0)
        )
    )


def _defect(A):
    return numpy.amax(numpy.fabs(A.T @ _OMEGA @ A - _OMEGA))


def _fd(f, x0, h):
    return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["explicit", "interp"])
def test_values_match_numpy(backend, mode, explicit, interp):
    # A numpy-built family evaluated on backend inputs gives the numpy path's
    # (x, v, Omega) and (J_R, J_z)(E, L_z, I_3) to round-off, on its degenerate
    # faces too; a family BUILT on a backend (forced) agrees to ~1e-11, its
    # turning points being polished roots where numpy's brentq stops at its
    # xtol, which the map's slopes amplify
    from galpy import backend as gb

    aN = explicit if mode == "explicit" else interp
    with gb.use(backend, force=True):
        aB = _explicit() if mode == "explicit" else _interp()
    assert aB._bk is not None and aN._bk is None
    if mode == "explicit":
        tori = [(aN._jrs[k], aN._Lzs[k], aN._jzs[k]) for k in range(2)]
        labels = _LABELS
    else:
        tori = _TORI
        labels = _LABELS[:1]
    for jr, jphi, jz in tori:
        a = (_ANGLES, 0.2 + _ANGLES, 0.7 - _ANGLES)
        ref = _flat(aN.xvFreqs(jr, jphi, jz, *a))
        ins = [_arr(backend, v) for v in (jr, jphi, jz, *a)]
        on_bk = aN.xvFreqs(*ins)
        assert all(_is_backend(backend, o) for o in on_bk)
        numpy.testing.assert_allclose(_flat(on_bk), ref, rtol=0.0, atol=_PLANE_ATOL)
        numpy.testing.assert_allclose(
            _flat(aB.xvFreqs(*ins)), ref, rtol=0.0, atol=3e-11
        )
        # numpy inputs to a backend-built family evaluate on its backend
        assert _is_backend(backend, aB(jr, jphi, jz, 0.3, 1.0, 2.0)[0])
        reff = _flat(aN.Freqs(jr, jphi, jz))
        numpy.testing.assert_allclose(
            _flat(aN.Freqs(*ins[:3])), reff, rtol=0.0, atol=_PLANE_ATOL
        )
        numpy.testing.assert_allclose(
            _flat(aB.Freqs(*ins[:3])), reff, rtol=0.0, atol=3e-11
        )
    for E, Lz, I3 in labels:
        ins = [_arr(backend, v) for v in (E, Lz, I3)]
        for f in ("JR", "Jz"):
            ref = getattr(aN, f)(E, Lz, I3)
            assert numpy.fabs(float(as_numpy(getattr(aN, f)(*ins))) - ref) < 1e-12
            assert numpy.fabs(float(as_numpy(getattr(aB, f)(*ins))) - ref) < 1e-11


@pytest.mark.parametrize("backend", BACKENDS)
def test_derivatives_vs_finite_difference(backend, interp):
    # d(q, p)/d(theta, J) by autodiff against a converged 5-point finite
    # difference of the numpy path, between the grid's nodes, for a prograde
    # and a retrograde torus; and the frequencies' derivatives in the actions
    for x0 in (
        numpy.array([0.4, 1.0, 2.0, 0.06, 0.93, 0.10]),
        numpy.array([5.0, 3.0, 0.7, 0.15, -1.21, 0.03]),
    ):

        def qp_np(x):
            return numpy.array(_qp(interp(x[3], x[4], x[5], *x[:3])))

        A = numpy.empty((6, 6))
        for k in range(6):
            e = numpy.zeros(6)
            e[k] = 1.0
            A[:, k] = _fd(lambda h: qp_np(x0 + h * e), 0.0, 1e-4)
        numpy.testing.assert_allclose(
            _jacobian(backend, interp, x0), A, rtol=0.0, atol=2e-10
        )
        dO = _jac_freqs(backend, interp, x0[3:])
        for k in range(3):
            e = numpy.zeros(3)
            e[k] = 1.0
            numpy.testing.assert_allclose(
                dO[:, k],
                _fd(lambda h: numpy.array(interp.Freqs(*(x0[3:] + h * e))), 0.0, 1e-4),
                rtol=0.0,
                atol=1e-9,
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_canonical(backend, interp, explicit):
    # Manifest canonicity, with exact derivatives: the autodiff Jacobian of
    # the map is symplectic to round-off between the family's nodes (and on
    # its shell and planar faces), for tables perturbed so much that the map
    # is no longer accurate (every derivative the map uses is its stored
    # interpolants' own), and for an explicit torus, a quadratic one-node
    # family; as the analytic isochrone inverse is
    from galpy.actionAngle import actionAngleIsochroneInverse

    x0 = numpy.array([0.4, 1.0, 2.0, 0.06, 0.93, 0.10])
    floor = _defect(
        _jacobian(backend, actionAngleIsochroneInverse(ip=interp.auxiliary), x0)
    )
    assert floor < 1e-12
    assert _defect(_jacobian(backend, interp, x0)) < 1e-12
    aP = copy.deepcopy(interp)
    rng = numpy.random.default_rng(3)
    for tab in (aP._A_tab, aP._U_tab, aP._V_tab):
        tab[1:] *= 1.0 + 1e-3 * rng.uniform(-1.0, 1.0, tab[1:].shape)
    for tab in (aP._A_dx, aP._U_dwE, aP._V_dsv):
        tab *= 1.0 + 1e-2 * rng.uniform(-1.0, 1.0, tab.shape)
    aP._rebuild_interp()
    assert (
        numpy.amax(
            numpy.fabs(_flat(aP(*x0[3:], *x0[:3])) - _flat(interp(*x0[3:], *x0[:3])))
        )
        > 1e-5
    )
    assert _defect(_jacobian(backend, aP, x0)) < 1e-12
    xe = numpy.array([0.4, 1.0, 2.0, explicit._jrs[0], 0.99, explicit._jzs[0]])
    assert _defect(_jacobian(backend, explicit, xe)) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_action_derivatives(backend):
    # An explicit torus is a quadratic one-node family: from its second-order
    # tables (computed from the torus alone), the derivatives of the map and
    # of its frequencies in the actions at the node are the true family's,
    # against finite differences across explicit tori built at E +- h,
    # I_3 +- h and L_z +- h (with the auxiliary held fixed; the physical point
    # does not depend on it once the maps are resolved), and so are the
    # gradient and Hessian of (J_R, J_z)(E, I_3, L_z)
    from galpy.potential import IsochronePotential

    E0, L0, I0 = _LABELS[0]
    h = 1e-4
    base = _explicit(labels=[_LABELS[0]])
    aux = IsochronePotential(amp=base._GM, b=base._b)

    def tor(E, L, I):
        return _explicit(labels=[(E, L, I)], auxiliary=aux)

    x0 = numpy.array([0.4, 1.0, 2.0, base._jrs[0], L0, base._jzs[0]])
    J = _jacobian(backend, base, x0)[:, [3, 5, 4]]  # in (J_R, J_z, J_phi)
    dO = _jac_freqs(backend, base, x0[3:])[:, [0, 2, 1]]
    dJda = numpy.linalg.inv(base._Ns[0])  # d(J_R, J_z, J_phi)/d(E, I_3, L_z)

    def qpO(aA):
        out = aA.xvFreqs(aA._jrs[0], aA._Lzs[0], aA._jzs[0], 0.4, 1.0, 2.0)
        return numpy.array(_qp(out) + [float(o) for o in out[6:]])

    tori = []
    for k in range(3):
        d = numpy.zeros(3)
        d[k] = h
        up = tor(E0 + d[0], L0 + d[2], I0 + d[1])
        dn = tor(E0 - d[0], L0 - d[2], I0 - d[1])
        tori.append((up, dn))
        fd = (qpO(up) - qpO(dn)) / (2.0 * h)
        numpy.testing.assert_allclose(J @ dJda[:, k], fd[:6], rtol=0.0, atol=3e-7)
        numpy.testing.assert_allclose(dO @ dJda[:, k], fd[6:], rtol=0.0, atol=3e-7)
    # (J_R, J_z)(E, I_3, L_z): the gradient against the neighbouring tori's
    # actions, the Hessian against their period matrices
    for q, name in ((0, "JR"), (1, "Jz")):

        def f(v, name=name):
            return getattr(base, name)(v[0], v[2], v[1])

        if backend == "jax":
            g = numpy.asarray(jax.grad(f)(jnp.asarray([E0, I0, L0])))
            H = numpy.asarray(jax.hessian(f)(jnp.asarray([E0, I0, L0])))
        else:
            v = torch.tensor([E0, I0, L0], requires_grad=True)
            g = as_numpy(torch.autograd.grad(f(v), v)[0])
            H = as_numpy(torch.autograd.functional.hessian(f, v.detach()))
        acts = (lambda a: a._jrs[0]) if q == 0 else (lambda a: a._jzs[0])
        numpy.testing.assert_allclose(
            g,
            [(acts(up) - acts(dn)) / (2.0 * h) for up, dn in tori],
            rtol=0.0,
            atol=1e-7,
        )
        M = lambda a: numpy.linalg.inv(a._Ns[0])[q]  # noqa: E731
        numpy.testing.assert_allclose(
            H,
            numpy.array([(M(up) - M(dn)) / (2.0 * h) for up, dn in tori]),
            rtol=0.0,
            atol=3e-7,
        )


# ---------- derivatives in the potential's parameters
def _param_family(p, mode, fitted=False):
    from galpy.potential import IsochronePotential

    # a given auxiliary, so that the finite differences below are of the
    # same map (the fitted one is frozen on a backend: no derivative flows
    # through the fit, and refitting moves a family at its interpolation error)
    aux = None if fitted else IsochronePotential(amp=6.0, b=1.1)
    if mode == "interp":
        return _interp(pot=_pot(amp=p), auxiliary=aux)
    return _explicit(pot=_pot(amp=p), auxiliary=aux)


def _param_out(aA, mode):
    # interpolated: the torus of fixed actions; explicit: the first torus at
    # its own (E, L_z, I_3), its actions moving with the parameter
    xp = aA._bk["xp"]
    if mode == "interp":
        J = (0.06, 0.93, 0.10)
    else:
        J = (aA._bk["jrs"][0], 0.99, aA._bk["jzs"][0])
    a = [xp.asarray(v) for v in (_ANGLES[:4], _ANGLES[:4] + 0.2, 0.7 - _ANGLES[:4])]
    return xp.concat([xp.reshape(o, (-1,)) for o in aA.xvFreqs(*J, *a)])


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_derivatives_wrt_potential_parameters(mode):
    # The construction is differentiable in the potential's parameters: the
    # (x, v, Omega) of a torus against a 5-point finite difference of the
    # backend construction itself (its roots are polished; numpy's stop at
    # brentq's xtol), for the amplitude of the Kuzmin-Kutuzov potential, on
    # an interpolated family and on explicit tori (at the torus's own (E, L_z,
    # I_3)). The explicit tori's finite differences have their floor, ~3e-9,
    # at h = 3e-4: their stencil error grows as h^4 above it and their noise
    # (~2e-12 in the values) as 1/h below; the interpolated family's stencil
    # error stays below that noise up to h = 1e-2, where they agree to 1e-10
    # (at 3e-4 the noise alone put them up to 1e-8 off)
    p0 = 4.0
    h, atol = (1e-2, 1e-9) if mode == "interp" else (3e-4, 5e-9)
    pt = torch.tensor(p0, requires_grad=True)
    y = _param_out(_param_family(pt, mode), mode)
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], pt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )

    def num(p):
        return as_numpy(_param_out(_param_family(torch.tensor(p), mode), mode).detach())

    numpy.testing.assert_allclose(g, _fd(num, p0, h), rtol=0.0, atol=atol)


@pytest.mark.skipif(jax is None or torch is None, reason="jax and torch needed")
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_jax_jit_construction(mode):
    # The whole pipeline -- the construction (its auxiliary fitted on the
    # device) and the evaluation -- traces under jax.jit, and
    # agrees with torch's eager construction
    def f(p):
        return _param_out(_param_family(p, mode, fitted=True), mode)

    y = _param_out(_param_family(torch.tensor(4.0), mode, fitted=True), mode)
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(f)(jnp.asarray(4.0))),
        as_numpy(y),
        rtol=0.0,
        atol=1e-11,
    )


@pytest.mark.skipif(jax is None or torch is None, reason="jax and torch needed")
def test_jax_parameter_derivatives_match_torch():
    # jax's derivatives in the potential's parameters through the construction
    # of explicit tori, jitted reverse mode, equal torch's
    pt = torch.tensor(4.0, requires_grad=True)
    y = _param_out(_param_family(pt, "explicit"), "explicit")
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], pt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )
    gj = jax.jit(
        jax.jacrev(lambda p: _param_out(_param_family(p, "explicit"), "explicit"))
    )(jnp.asarray(4.0))
    # two backends' round-off through the construction: 1.8e-11 on CI
    numpy.testing.assert_allclose(numpy.asarray(gj), g, rtol=0.0, atol=5e-11)


# ---------- jit, vmap, compile, float32
@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_and_vmap(interp, explicit):
    # the evaluation traces under jax.jit, one trace per family for all its
    # tori (a degenerate torus's branches are all traced, and its flags
    # select), and vmaps over tori
    cases = [
        (interp, _TORI),
        (
            explicit,
            [(explicit._jrs[k], explicit._Lzs[k], explicit._jzs[k]) for k in range(2)],
        ),
    ]
    E, Lz, I3 = _LABELS[0]
    for aA, tori in cases:

        @jax.jit
        def f(j, a, aA=aA):
            return jnp.concatenate(
                [jnp.ravel(o) for o in aA.xvFreqs(j[0], j[1], j[2], a, 1.0, 2.0)]
            )

        @jax.jit
        def g(j, aA=aA):
            return jnp.stack(aA.Freqs(j[0], j[1], j[2]))

        @jax.jit
        def h(e, l, i, aA=aA):
            return jnp.stack([aA.JR(e, l, i), aA.Jz(e, l, i)])

        for J in tori:
            numpy.testing.assert_allclose(
                numpy.asarray(f(jnp.asarray(J), jnp.asarray(_ANGLES))),
                _flat(aA.xvFreqs(*J, _ANGLES, 1.0, 2.0)),
                rtol=0.0,
                atol=_PLANE_ATOL,
            )
            numpy.testing.assert_allclose(
                numpy.asarray(g(jnp.asarray(J))),
                _flat(aA.Freqs(*J)),
                rtol=0.0,
                atol=_PLANE_ATOL,
            )
        numpy.testing.assert_allclose(
            numpy.asarray(h(*(jnp.asarray(v) for v in (E, Lz, I3)))),
            [aA.JR(E, Lz, I3), aA.Jz(E, Lz, I3)],
            rtol=0.0,
            atol=1e-12,
        )
    # vmap over tori, a shell, a planar and a circular one among them
    jrs = numpy.array([0.06, 0.0, 0.02, 0.0])
    jzs = numpy.array([0.10, 0.2, 0.0, 0.0])
    out = jax.vmap(
        lambda jr, jz: interp.xvFreqs(jr, 1.0, jz, jnp.asarray(_ANGLES), 1.0, 2.0)[0]
    )(jnp.asarray(jrs), jnp.asarray(jzs))
    for k in range(4):
        numpy.testing.assert_allclose(
            numpy.asarray(out[k]),
            interp.xvFreqs(jrs[k], 1.0, jzs[k], _ANGLES, 1.0, 2.0)[0],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_compile(explicit):
    # the evaluation runs under torch.compile (an explicit torus)
    jr, Lz, jz = explicit._jrs[1], explicit._Lzs[1], explicit._jzs[1]
    ref = _flat(explicit.xvFreqs(jr, Lz, jz, _ANGLES, 1.0, 2.0))
    f = torch.compile(
        lambda j, a: torch.cat(
            [
                torch.atleast_1d(o).ravel()
                for o in explicit.xvFreqs(j, Lz, jz, a, 1.0, 2.0)
            ]
        ),
        backend="eager",
    )
    got = f(torch.tensor(jr), torch.tensor(_ANGLES))
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_in_float32_out(backend, interp):
    # computed in float64, returned in the dtype of float32 inputs
    if backend == "jax":
        f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731
    else:
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    ref = interp.xvFreqs(0.06, 0.93, 0.10, _ANGLES, 1.0, 2.0)
    got = interp.xvFreqs(
        f32(0.06), f32(0.93), f32(0.10), f32(_ANGLES), f32(1.0), f32(2.0)
    )
    for g, r in zip(got, ref):
        assert str(g.dtype).endswith("float32")
        numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=3e-6)
    E, Lz, I3 = _LABELS[0]
    for out in (
        *interp.Freqs(f32(0.06), f32(0.93), f32(0.10)),
        interp.JR(f32(E), f32(Lz), f32(I3)),
        interp.Jz(f32(E), f32(Lz), f32(I3)),
    ):
        assert str(out.dtype).endswith("float32")


# ---------- degenerate tori
def _degenerate_labels():
    # (E, L_z, I_3) of a circular, a shell and a planar torus at the benign
    # torus's L_z, from a numpy family's own edge relations
    aN = _explicit(labels=[_LABELS[0]])
    E, Lz, _ = _LABELS[0]
    Ec = aN._circular(Lz)["Ec"]
    return [
        (Ec, Lz, aN._I3_planar(Ec, Lz)),
        (E, Lz, aN._I3_shell(E, Lz, aN._ushell(E, Lz))),
        (E, Lz, aN._I3_planar(E, Lz)),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spacing", ["linear", "sqrt", "log"])
def test_Lzspacing(backend, spacing):
    # The spacing of the L_z grid on a backend: a family built there (forced)
    # has numpy's nodes, to round-off, and numpy's tori, to its
    # construction's ~1e-10; an unknown spacing raises
    from galpy import backend as gb

    aN = _interp(Lzspacing=spacing)
    with gb.use(backend, force=True):
        aB = _interp(Lzspacing=spacing)
    numpy.testing.assert_allclose(aB._Lzgrid, aN._Lzgrid, rtol=1e-14, atol=0.0)
    J = _TORI[0]
    ref = _flat(aN.xvFreqs(*J, _ANGLES, 1.0, 2.0))
    got = aB.xvFreqs(*(_arr(backend, v) for v in J), _arr(backend, _ANGLES), 1.0, 2.0)
    numpy.testing.assert_allclose(_flat(got), ref, rtol=0.0, atol=2e-10)
    with gb.use(backend, force=True):
        with pytest.raises(ValueError, match="spacing 'cubic' not understood"):
            _interp(Lzspacing="cubic")


@pytest.mark.parametrize("backend", BACKENDS)
def test_lone_circular_torus_auxiliary(backend):
    # A family of one planar circular torus spans a single radius, which
    # every isochrone scale fits: the auxiliary is the isochrone through the
    # rotation curve there with b that radius, on numpy and on a backend (a
    # NaN auxiliary, as a fit through its Newton step gave, made the
    # backend's Galerkin solve singular on CUDA)
    from galpy import backend as gb
    from galpy.potential import vcirc

    labels = _degenerate_labels()[:1]
    aN = _explicit(labels=labels)
    with gb.use(backend, force=True):
        aB = _explicit(labels=labels)
    rc = float(aN._delta * numpy.sinh(aN._ucs[0]))
    vc2 = float(as_numpy(vcirc(aN._staeckelwrap, rc, use_physical=False))) ** 2
    for aA in (aN, aB):
        GM, b = (float(as_numpy(t)) for t in (aA._GM, aA._b))
        s = numpy.sqrt(b**2 + rc**2)
        assert abs(b / rc - 1.0) < 1e-14
        assert abs(GM * rc**2 / (s * (b + s) ** 2) / vc2 - 1.0) < 1e-13


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate_explicit_tori(backend):
    # Explicit circular, shell and planar tori -- both librations degenerate,
    # or one -- built on a backend agree with the numpy ones, in their tables
    # and their evaluation, with symplectic autodiff Jacobians; the degeneracy
    # of a torus is decided on its concrete labels, and under a trace both
    # branches run and the flags select
    from galpy import backend as gb

    labels = _degenerate_labels()
    aN = _explicit(labels=labels)
    with gb.use(backend, force=True):
        aB = _explicit(labels=labels)
    numpy.testing.assert_array_equal(aN._jrs == 0.0, [True, True, False])
    numpy.testing.assert_array_equal(aN._jzs == 0.0, [True, False, True])
    if backend == "jax":

        @jax.jit
        def traced(j):
            return jnp.concatenate(
                [
                    jnp.ravel(o)
                    for o in aN.xvFreqs(
                        j[0], j[1], j[2], jnp.asarray(_ANGLES), 1.0, 2.0
                    )
                ]
            )

    for k in ("jrs", "jzs", "wus", "wvs", "Dmus", "Dmvs"):
        numpy.testing.assert_allclose(
            getattr(aB, "_" + k), getattr(aN, "_" + k), rtol=0.0, atol=1e-12
        )
    for k in range(3):
        J = (aN._jrs[k], aN._Lzs[k], aN._jzs[k])
        ref = _flat(aN.xvFreqs(*J, _ANGLES, 1.0, 2.0))
        for aA in (aN, aB):
            got = aA.xvFreqs(
                *(_arr(backend, v) for v in J), _arr(backend, _ANGLES), 1.0, 2.0
            )
            numpy.testing.assert_allclose(_flat(got), ref, rtol=0.0, atol=1e-11)
        if k > 0:
            # the point does not depend on the degenerate angle: the map is a
            # symplectic embedding of the live (theta, J) pairs, (theta_phi,
            # theta_z, J_phi, J_z) on the shell and (theta_R, theta_phi, J_R,
            # J_phi) in the plane, so the form pulls back to theirs
            Jac = _jacobian(backend, aN, numpy.array([0.4, 1.0, 2.0, *J]))
            live = [1, 2, 4, 5] if k == 1 else [0, 1, 3, 4]
            A = Jac[:, live]
            assert numpy.all(numpy.isfinite(Jac))
            assert (
                numpy.amax(numpy.fabs(A.T @ _OMEGA @ A - _OMEGA[numpy.ix_(live, live)]))
                < 1e-12
            )
        if backend == "jax":
            numpy.testing.assert_allclose(
                numpy.asarray(traced(jnp.asarray(J))), ref, rtol=0.0, atol=_PLANE_ATOL
            )


# ---------- the adaptive reference curve and the target box
@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_adaptive_and_target():
    # u0='planar' (natively, frozen), a callable u0 (on the host) and a target's
    # box (its labels natively, frozen) give the numpy families on a backend;
    # the box's faces are a static choice that a traced construction cannot
    # make
    from galpy import backend as gb
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    angler = numpy.linspace(0.05, 6.2, 11)
    o = Orbit([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    ts = numpy.linspace(0.0, 12.0, 8)
    o.integrate(ts, _pot())
    cases = [
        (
            dict(
                pot=LogarithmicHaloPotential(normalize=1.0, q=0.9),
                delta=0.8,
                u0="planar",
                Rmin=0.5,
                Rmax=2.0,
                Rinf=6.0,
            ),
            [(0.3, 1.9, 0.1), (0.06, 0.93, 0.10)],
        ),
        (
            dict(u0=lambda E, Lz: 1.0 + 0.2 * numpy.tanh(Lz) + 0.1 * E),
            [(0.06, 0.93, 0.10), (0.15, 1.21, 0.03), (0.0, 1.1, 0.0)],
        ),
        (
            dict(target=o(ts), Rmin=0.75, Rmax=1.25, Rinf=3.0),
            [(0.0385, 1.1025, 0.01266)],
        ),
        (
            dict(target=o(ts), u0="planar", Rmin=0.75, Rmax=1.25, Rinf=3.0),
            [(0.0385, 1.1025, 0.01266)],
        ),
        # a rule that uses each point's third integral to place its chart
        (
            dict(target=o(ts), u0="torus", Rmin=0.75, Rmax=1.25, Rinf=3.0),
            [(0.0385, 1.1025, 0.01266)],
        ),
    ]
    fams = []
    for kw, tori in cases:
        aN = _interp(**kw)
        with gb.use("torch", force=True):
            aB = _interp(**kw)
        numpy.testing.assert_allclose(
            numpy.array(aB._box, dtype=float),
            numpy.array(aN._box, dtype=float),
            rtol=0.0,
            atol=_PLANE_ATOL,
        )
        for tab in ("A_tab", "U_tab", "V_tab"):
            numpy.testing.assert_allclose(
                getattr(aB, "_" + tab), getattr(aN, "_" + tab), rtol=0.0, atol=1e-12
            )
        for J in tori:
            numpy.testing.assert_allclose(
                _flat(aB(*J, angler, 1.0, 2.0)),
                _flat(aN(*J, angler, 1.0, 2.0)),
                rtol=0.0,
                atol=3e-11,
            )
        fams.append(aB)
    # the target's checks: its box excludes the faces and the circular orbit
    loc, (jr, lz, jz) = fams[2], (0.0385, 1.1025, 0.01266)
    with pytest.raises(ValueError, match="does not reach"):
        loc(0.0, lz, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="does not reach"):
        loc(jr, lz, 0.0, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="circular face"):
        loc.Freqs(0.0, lz, 0.0)
    # the target orbit's (E, L_z, I_3), with E inside [E_c, E_max] but outside
    # the box
    E, Lz, I3 = -1.465095655583749, 1.1025, 1.8591431356904407
    # out-of-range errors point at target_pad, not at Rinf
    hint = r"\(the grid is localized on a target: increase target_pad"
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc.JR(E + 0.1, Lz, I3)
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc.JR(E, Lz, I3 + 0.5)
    with pytest.raises(ValueError, match="outside the interpolation grid.*" + hint):
        loc(jr, 0.5, jz, 0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match=r"above the covered [\d.]+ " + hint):
        loc(10.0 * jr, lz, jz, 0.3, 1.0, 2.0)
    numpy.testing.assert_allclose(
        float(as_numpy(loc.JR(E, Lz, I3))), jr, rtol=0.0, atol=1e-4
    )
    # a padding that reaches beyond Rinf (two tori: one has no spread to pad)
    o2 = Orbit([0.9, -0.1, 1.15, -0.05, 0.1, 0.0])
    o2.integrate(ts, _pot())
    rows = numpy.vstack(
        [
            numpy.array([p(ts).R(), p(ts).vR(), p(ts).vT(), p(ts).z(), p(ts).vz()]).T
            for p in (o, o2)
        ]
    )
    with gb.use("torch", force=True):
        with pytest.raises(ValueError, match="beyond Rinf"):
            _interp(target=rows, target_pad=50.0, Rmin=0.75, Rmax=1.25, Rinf=3.0)
        with pytest.raises(ValueError, match="increase Rinf"):
            _interp(target=[1.0, 0.9, 1.0, 0.3, 0.9], Rmin=0.75, Rmax=1.25, Rinf=1.3)
    if jax is not None:

        def traced(p):
            return _interp(
                pot=_pot(amp=p), target=o(ts), Rmin=0.75, Rmax=1.25, Rinf=3.0
            )._bk["Lzgrid"]

        with pytest.raises(NotImplementedError, match="static"):
            jax.jit(traced)(jnp.asarray(4.0))


# ---------- rules for the reference curve: every torus in its own Staeckel model
# two interior nodes of a 5^3 family in _rule_pot (Rmin=0.8, Rmax=1.3, Rinf=2.8;
# the labels do not depend on the rule), the second at L_z < 0
_RULE_LABELS = [(0.66369425438, 1.0198039027, 0.3498601593), (0.907152284, -1.1514100371, 0.2670275931)]  # fmt: skip


def _rule_pot(amp=1.0):
    # not of Staeckel form, so that the model varies along the rule
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(amp=amp, q=0.8)


def _rule_callable(E, Lz, I3):
    return 1.0 + 0.1 * numpy.tanh(I3) + 0.05 * E


def _rule_tori(u0, pot=None, **kw):
    from galpy.actionAngle import actionAngleStaeckelInverse

    return actionAngleStaeckelInverse(
        pot=_rule_pot() if pot is None else pot,
        delta=0.8,
        u0=u0,
        Es=[l[0] for l in _RULE_LABELS],
        Lzs=[l[1] for l in _RULE_LABELS],
        I3s=[l[2] for l in _RULE_LABELS],
        mm_npt=16,
        progressbar=False,
        **kw,
    )


@pytest.mark.parametrize(
    "backend,rule",
    # the rules' code is backend-agnostic: jax (eager, which compiles per
    # shape) runs one of them, under jax.jit the next test
    [
        (b, r)
        for b in BACKENDS
        for r in (("torus",) if b == "jax" else ("torus", "shell", "callable"))
    ],
)
def test_rules_for_the_reference_curve(backend, rule):
    # u0='torus' and 'shell' natively, a callable of (E, L_z, I_3) on the
    # host, all frozen (a gauge), with the model's variation along the rule
    # in the tables: a family and explicit tori built on a backend equal
    # numpy's
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse

    angler = numpy.linspace(0.05, 6.2, 7)
    for u0 in (_rule_callable if rule == "callable" else rule,):
        fkw = dict(
            pot=_rule_pot(),
            delta=0.8,
            u0=u0,
            setup_interp=True,
            Rmin=0.8,
            Rmax=1.3,
            Rinf=2.8,
            nE=5,
            nLz=5,
            nI3=5,
            mm_npt=16,
            progressbar=False,
        )
        aN = actionAngleStaeckelInverse(**fkw)
        eN = _rule_tori(u0)
        with gb.use(backend, force=True):
            aB = actionAngleStaeckelInverse(**fkw)
            eB = _rule_tori(u0)
        for tab in ("A_tab", "U_tab", "V_tab"):
            numpy.testing.assert_allclose(
                getattr(aB, "_" + tab), getattr(aN, "_" + tab), rtol=0.0, atol=1e-12
            )
        # the circular orbit's frequencies, in its own chart
        numpy.testing.assert_allclose(
            _flat(aB.Freqs(0.0, 1.0, 0.0)), _flat(aN.Freqs(0.0, 1.0, 0.0)), atol=1e-12
        )
        tori = [(0.05, 1.0, 0.02), (0.1, 1.1, 0.05)]
        for aA, a0 in ((aB, aN), (eB, eN)):
            if aA is eB:
                tori = [(eN._jrs[k], eN._Lzs[k], eN._jzs[k]) for k in range(2)]
            for J in tori:
                numpy.testing.assert_allclose(
                    _flat(aA(*J, angler, 1.0, 2.0)),
                    _flat(a0(*J, angler, 1.0, 2.0)),
                    rtol=0.0,
                    atol=3e-11,
                )


def _rule_out(aA):
    # the first torus at its own (E, L_z, I_3), its actions moving with the
    # parameter
    xp = aA._bk["xp"]
    J = (aA._bk["jrs"][0], _RULE_LABELS[0][1], aA._bk["jzs"][0])
    a = [xp.asarray(v) for v in (_ANGLES[:4], _ANGLES[:4] + 0.2, 0.7 - _ANGLES[:4])]
    return xp.concat([xp.reshape(o, (-1,)) for o in aA.xvFreqs(*J, *a)])


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_rule_derivatives_wrt_potential_parameters():
    # Through the model's variation along the rule (J_z's, the v half-width's
    # and both maps' along u0): explicit tori with a callable rule, whose u0
    # and partials at the tori's fixed labels do not move with the parameter,
    # against a 5-point finite difference of the backend construction
    from galpy.potential import IsochronePotential

    aux = IsochronePotential(amp=3.44, b=0.565)

    def fam(p):
        return _rule_tori(_rule_callable, pot=_rule_pot(amp=p), auxiliary=aux)

    p0, h = 1.0, 3e-4
    pt = torch.tensor(p0, requires_grad=True)
    y = _rule_out(fam(pt))
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], pt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )
    numpy.testing.assert_allclose(
        g,
        _fd(lambda p: as_numpy(_rule_out(fam(torch.tensor(p))).detach()), p0, h),
        rtol=0.0,
        atol=5e-9,
    )


@pytest.mark.skipif(jax is None or torch is None, reason="jax and torch needed")
def test_rule_jax_jit_construction():
    # u0='torus' natively under jax.jit (the rule's turning points and
    # partials on the device): the traced construction of explicit tori and
    # its jitted derivative equal torch's eager ones
    def f(p):
        return _rule_out(_rule_tori("torus", pot=_rule_pot(amp=p)))

    pt = torch.tensor(1.0, requires_grad=True)
    y = f(pt)
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], pt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(f)(jnp.asarray(1.0))),
        as_numpy(y.detach()),
        rtol=0.0,
        atol=1e-11,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(jax.jacrev(f))(jnp.asarray(1.0))),
        g,
        rtol=0.0,
        atol=5e-11,
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_split_tori_family_backend():
    # a family built on a backend whose grid reaches split librations
    # raises numpy's errors: a v libration whose momentum function rises
    # again beyond its turning point (a cusped halo near the axis) and a u
    # libration whose momentum function has two maxima (MWPotential2014's
    # cusped bulge at small |L_z|)
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014, NFWPotential

    kw = dict(setup_interp=True, mm_npt=32, progressbar=False)
    with gb.use("torch", force=True):
        with pytest.raises(ValueError, match="does not fall monotonically"):
            actionAngleStaeckelInverse(
                pot=NFWPotential(normalize=1.0, a=1.0),
                delta=0.31,
                Rmin=0.05,
                Rmax=3.0,
                Rinf=8.0,
                nE=6,
                nLz=6,
                nI3=6,
                **kw,
            )
        with pytest.raises(ValueError, match="more than one maximum"):
            actionAngleStaeckelInverse(
                pot=MWPotential2014,
                delta=0.3,
                Rmin=0.04,
                Rmax=2.0,
                Rinf=5.0,
                nE=16,
                nLz=4,
                nI3=4,
                **kw,
            )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_in_domain_backend_family():
    # in_domain and the counted error of an array evaluation for a family
    # built on a backend: the single tori (degenerate faces) are looked up
    # on the backend, the many through numpy's label inversion
    from galpy import backend as gb

    with gb.use("torch", force=True):
        aB = _interp()
    assert aB._bk is not None
    for actions, expected in (
        ((0.05, 0.9, 0.05), True),
        ((0.0, 0.9, 0.05), True),
        ((0.05, -0.9, 0.0), True),
        ((0.05, 5.0, 0.0), False),
        ((50.0, 0.9, 0.0), False),
    ):
        assert aB.in_domain(*actions) is expected, actions
    jr = numpy.array([0.05, 0.06, 50.0])
    Lz = numpy.array([0.9, 5.0, 0.9])
    jz = numpy.array([0.05, 0.05, 0.05])
    assert list(aB.in_domain(jr, Lz, jz)) == [True, False, False]
    with gb.use("torch", force=True):
        with pytest.raises(
            ValueError, match="2 of the 3 tori lie outside the interpolated family"
        ):
            aB(jr, Lz, jz, 0.3, 1.0, 2.0)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_backend():
    # tori set up from their actions on a backend: the integrals are solved
    # on a host copy of the model, and the tori built on the backend equal
    # numpy's; a differentiated parameter would not reach those integrals
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse

    J = dict(jrs=[0.05, 0.3], Lzs=[0.9, -0.6], jzs=[0.05, 0.2])

    aN = actionAngleStaeckelInverse(pot=_pot(), mm_npt=48, progressbar=False, **J)
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(pot=_pot(), mm_npt=48, progressbar=False, **J)
    assert aB._bk is not None
    numpy.testing.assert_allclose(aB._Es, aN._Es, rtol=1e-13, atol=0.0)
    numpy.testing.assert_allclose(aB._I3s, aN._I3s, rtol=1e-13, atol=1e-15)
    for jr, Lz, jz in zip(J["jrs"], J["Lzs"], J["jzs"]):
        numpy.testing.assert_allclose(
            _flat(aB(jr, Lz, jz, _ANGLES[:3], 1.0, 2.0)),
            _flat(aN(jr, Lz, jz, _ANGLES[:3], 1.0, 2.0)),
            rtol=0.0,
            atol=3e-12,
        )
    with pytest.raises(NotImplementedError, match="differentiated jax/torch"):
        actionAngleStaeckelInverse(
            pot=_pot(amp=torch.tensor(4.0, requires_grad=True)), **J
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_split_u_libration_explicit(backend):
    # an explicit torus whose u libration is split in the Staeckel
    # approximation of a cusped potential (p_u^2 has a second maximum close
    # to the centre): the backend brackets its turning points from the
    # barrier next to the shell u, as numpy does, and builds numpy's torus
    # (from the shell u outward, the bracket steps over the barrier here and
    # finds the libration across both wells)
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(
        pot=MWPotential2014,
        delta=0.3,
        mm_npt=64,
        progressbar=False,
        Es=[-1.5],
        Lzs=[0.004],
        I3s=[3.3629344522623326],
    )
    aN = actionAngleStaeckelInverse(**kw)
    with gb.use(backend, force=True):
        aB = actionAngleStaeckelInverse(**kw)
    numpy.testing.assert_allclose(aB._jrs, aN._jrs, rtol=1e-12, atol=0.0)
    numpy.testing.assert_allclose(aB._jzs, aN._jzs, rtol=1e-12, atol=0.0)
    # (this close to the axis the points agree to 1e-10)
    J = (aN._jrs[0], 0.004, aN._jzs[0])
    numpy.testing.assert_allclose(
        _flat(aB(*J, _ANGLES[:3], 1.0, 2.0)),
        _flat(aN(*J, _ANGLES[:3], 1.0, 2.0)),
        rtol=0.0,
        atol=3e-10,
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_sheets_backend():
    # tori set up from their actions on each sheet of a split u libration:
    # the backend builds each torus on the libration that the solve for its
    # integrals found it on (about the outer maximum where the inner one is
    # the higher, across both wells, about the inner maximum), and under the
    # 'shell' rule takes the rule's partials at the shell orbit's u
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    J = dict(
        jrs=[0.015098, 0.322184, 0.009532],
        Lzs=[0.002183, 0.009836, 0.008259],
        jzs=[0.232254, 0.110735, 0.182493],
    )
    for rule in ({}, {"u0": "shell"}):
        kw = dict(pot=MWPotential2014, delta=0.3, mm_npt=64, progressbar=False)
        aN = actionAngleStaeckelInverse(**J, **kw, **rule)
        with gb.use("torch", force=True):
            aB = actionAngleStaeckelInverse(**J, **kw, **rule)
        numpy.testing.assert_allclose(aB._Es, aN._Es, rtol=1e-13, atol=0.0)
        numpy.testing.assert_allclose(aB._I3s, aN._I3s, rtol=1e-13, atol=0.0)
        for jr, Lz, jz in zip(*J.values()):
            numpy.testing.assert_allclose(
                _flat(aB(jr, Lz, jz, _ANGLES[:3], 1.0, 2.0)),
                _flat(aN(jr, Lz, jz, _ANGLES[:3], 1.0, 2.0)),
                rtol=0.0,
                atol=3e-10,
            )


# tori set up from their actions natively (traced or differentiated actions,
# or on a GPU): a generic torus and one on each face, in MWPotential2014
_NATIVE_J = dict(jrs=[0.05, 0.0, 0.02], Lzs=[1.0, 0.8, -1.2], jzs=[0.03, 0.04, 0.0])
# actions of no torus of MWPotential2014's model of delta=0.27, u0='torus' (the
# actions jump close to the axis)
_NO_TORUS = (0.01662122421268204, 0.017912561242134813, 0.20920350851878325)


def _native_kw(rule, u0="torus"):
    from galpy.potential import MWPotential2014

    if rule:
        delta = lambda E, Lz, f: 0.45 * (1.0 + 0.1 * (f - 0.5) + 0.05 * Lz)  # noqa: E731
    else:
        delta = 0.45
    if u0 == "callable":
        u0 = lambda E, Lz: 1.0 + 0.1 * Lz  # noqa: E731
    return dict(pot=MWPotential2014, delta=delta, u0=u0, mm_npt=32, progressbar=False)


def _native_points(aA, J):
    return numpy.array(
        [
            _flat(aA(jr, Lz, jz, _ANGLES[:3], 1.0, 2.0))
            for jr, Lz, jz in zip(J["jrs"], J["Lzs"], J["jzs"])
        ]
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    "rule,u0",
    [
        (False, "torus"),
        (True, "torus"),
        (False, "shell"),
        (True, "shell"),
        (False, "callable"),
    ],
)
def test_tori_from_actions_native(rule, u0):
    # tori set up from differentiated actions are solved for natively, all at
    # once (as on a GPU and under jax.jit): their integrals (and focal
    # lengths under a rule) are numpy's, and their points numpy's to the
    # precision of the turning points that the solve finds (which numpy's
    # tori are set up on)
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse

    kw = _native_kw(rule, u0)
    aN = actionAngleStaeckelInverse(**_NATIVE_J, **kw)
    J = {k: torch.tensor(v, requires_grad=True) for k, v in _NATIVE_J.items()}
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(**J, **kw)
        pB = _native_points(aB, _NATIVE_J)
    # (under a rule, to the 1e-12 to which its focal lengths converge)
    rtol = 1e-12 if rule else 1e-13
    numpy.testing.assert_allclose(as_numpy(aB._Es), aN._Es, rtol=rtol, atol=0.0)
    numpy.testing.assert_allclose(as_numpy(aB._I3s), aN._I3s, rtol=rtol, atol=1e-15)
    if rule:
        numpy.testing.assert_allclose(
            as_numpy(aB._bk["deltas"]), aN._deltas, rtol=rtol, atol=0.0
        )
    numpy.testing.assert_allclose(
        pB, _native_points(aN, _NATIVE_J), rtol=0.0, atol=1e-9
    )


def _unconverged_tori():
    # two near-axis tori of a fitted rule at angles whose vectorized Newton
    # iteration for the anomalies does not converge: the keywords that set
    # them up, the angles, and numpy's points
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    with gb.use("numpy", force=True):
        fam = actionAngleStaeckelInverse(
            pot=MWPotential2014,
            delta="fit",
            u0="torus",
            setup_interp=True,
            Rmin=0.08,
            Rmax=14.5,
            Rinf=30.0,
            nE=4,
            nLz=4,
            nI3=4,
            mm_npt=32,
            progressbar=False,
        )
    kw = dict(
        pot=MWPotential2014,
        delta=fam.delta,
        u0="torus",
        mm_npt=128,
        progressbar=False,
        jrs=[1.445704293904461, 0.02952728077189904],
        Lzs=[0.0017666654511333476, 0.022944891605053343],
        jzs=[0.0056624704253855455, 0.791342968271593],
    )
    angles = ([6.154419250568444, 2.508819429227821], [0.0, 0.0])
    angles = (*angles, [2.314008829991801, 4.505178120075055])
    xn = numpy.array(
        actionAngleStaeckelInverse(**kw).xvFreqs(
            kw["jrs"], kw["Lzs"], kw["jzs"], *angles
        )
    )
    return kw, angles, xn


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_unconverged_angles_vectorized(monkeypatch):
    # points whose vectorized Newton iteration for the anomalies does not
    # converge (near-axis tori of a fitted rule): solved all at once from
    # their requested angles within brackets as wide as their angle
    # relations can need, as numpy, not torus by torus
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse

    mod = importlib.import_module("galpy.actionAngle.actionAngleStaeckelInverse")
    kw, angles, xn = _unconverged_tori()
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(**kw)
    steps = []
    brentq = mod._backend_brentq

    def spy(*args, **kwargs):
        steps.append(kwargs.get("newton_steps"))
        return brentq(*args, **kwargs)

    def per_torus(*args, **kwargs):
        raise AssertionError("a point went torus by torus")

    monkeypatch.setattr(mod, "_backend_brentq", spy)
    monkeypatch.setattr(mod.actionAngleStaeckelInverse, "_bk_per_torus", per_torus)
    xb = numpy.array(
        [
            as_numpy(t)
            for t in aB.xvFreqs(
                *(torch.tensor(t) for t in (kw["jrs"], kw["Lzs"], kw["jzs"], *angles))
            )
        ]
    )
    assert 24 in steps, "the points were not solved from their requested angles"
    assert numpy.allclose(xb[:6], xn[:6], rtol=0.0, atol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_tori_unconverged_angles_jit():
    # under jax.jit, the points that the Newton iteration for the anomalies
    # leaves far from their roots are solved again from their requested
    # angles within wide brackets, a few at a time: numpy's points
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse

    kw, angles, xn = _unconverged_tori()
    with gb.use("jax", force=True):
        aB = actionAngleStaeckelInverse(**kw)
    xb = numpy.asarray(
        jax.jit(lambda *a: jnp.stack(aB.xvFreqs(*a)))(
            *(jnp.asarray(t) for t in (kw["jrs"], kw["Lzs"], kw["jzs"], *angles))
        )
    )
    assert numpy.allclose(xb[:6], xn[:6], rtol=0.0, atol=1e-9)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_native_fine_anomalies():
    # many tori on a fine anomaly grid: the native solve first on a coarser
    # grid, then on the tori's own, numpy's integrals
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    rng = numpy.random.default_rng(3)
    n = 130
    J = dict(
        jrs=10 ** rng.uniform(-2.0, -1.0, n),
        Lzs=rng.uniform(0.5, 1.5, n),
        jzs=10 ** rng.uniform(-2.3, -1.3, n),
    )
    kw = dict(pot=MWPotential2014, delta=0.45, mm_npt=32, mm_nta=264, progressbar=False)
    aN = actionAngleStaeckelInverse(**J, **kw)
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(
            **{k: torch.tensor(v, requires_grad=True) for k, v in J.items()}, **kw
        )
    numpy.testing.assert_allclose(as_numpy(aB._Es), aN._Es, rtol=1e-13, atol=0.0)
    numpy.testing.assert_allclose(as_numpy(aB._I3s), aN._I3s, rtol=1e-13, atol=1e-14)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_native_inputs():
    # set up natively from differentiated actions: numpy actions among them
    # (on the device of the set-up), the frequencies of one torus, and the
    # errors for unequal lengths, L_z = 0, negative actions, a rule with
    # negative focal lengths, and actions that are not one of the tori
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.45, mm_npt=32, progressbar=False)
    Lz, jz = numpy.array([1.0, 0.8]), numpy.array([0.03, 0.01])
    aN = actionAngleStaeckelInverse(jrs=[0.05, 0.02], Lzs=Lz, jzs=jz, **kw)
    jr = torch.tensor([0.05, 0.02], requires_grad=True)
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(jrs=jr, Lzs=Lz, jzs=jz, **kw)
        numpy.testing.assert_allclose(
            [
                float(as_numpy(t))
                for t in aB.Freqs(*(torch.tensor(v) for v in (0.05, 1.0, 0.03)))
            ],
            [float(numpy.ravel(t)[0]) for t in aN.Freqs(0.05, 1.0, 0.03)],
            rtol=1e-9,
        )
        with pytest.raises(ValueError, match="not one of the set-up tori"):
            aB(
                torch.tensor([0.05, 0.07]),
                torch.tensor(Lz),
                torch.tensor(jz),
                0.1,
                0.2,
                0.3,
            )
        for bad, match in (
            (dict(jrs=jr, Lzs=numpy.array([1.0]), jzs=jz), "same length"),
            (dict(jrs=jr, Lzs=numpy.array([1.0, 0.0]), jzs=jz), "L_z = 0"),
            (dict(jrs=-jr, Lzs=Lz, jzs=jz), "non-negative"),
            (
                dict(jrs=jr, Lzs=Lz, jzs=jz, delta=lambda E, L, f: 0.0 * E - 0.45),
                "positive focal lengths",
            ),
        ):
            with pytest.raises(ValueError, match=match):
                actionAngleStaeckelInverse(**{**kw, **bad})


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_native_derivatives():
    # the derivatives of a torus's points in its actions through its set-up
    # from them equal finite differences of numpy's tori set up at the
    # displaced actions (a single model: the reference curve and focal length
    # held, as a family's derivatives hold them)
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.45, mm_npt=32, progressbar=False)
    j0 = numpy.array([0.05, 1.0, 0.03])

    def numpy_R(j):
        aN = actionAngleStaeckelInverse(jrs=[j[0]], Lzs=[j[1]], jzs=[j[2]], **kw)
        return numpy.array([numpy.ravel(t)[0] for t in aN(*j, 0.4, 1.0, 2.0)[:2]])

    x = torch.tensor(j0, requires_grad=True)
    with gb.use("torch", force=True):
        aB = actionAngleStaeckelInverse(jrs=x[0:1], Lzs=x[1:2], jzs=x[2:3], **kw)
        out = aB(x[0], x[1], x[2], 0.4, 1.0, 2.0)[:2]
    for k in range(2):
        g = as_numpy(torch.autograd.grad(out[k], x, retain_graph=True)[0])
        fd = numpy.array(
            [_fd(lambda h: numpy_R(j0 + h * e)[k], 0.0, 1e-4) for e in numpy.eye(3)]
        )
        numpy.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tori_from_actions_follow_the_data():
    # differentiated actions set up their tori natively without a forced
    # backend, whatever the potential; a scalar is one torus, and lists and
    # floats among the actions are numpy's
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.45, mm_npt=32, progressbar=False)

    def numpy_R(jz):
        aN = actionAngleStaeckelInverse(jrs=0.05, Lzs=[1.0], jzs=jz, **kw)
        return numpy.ravel(aN(0.05, 1.0, jz, 0.4, 1.0, 2.0)[0])[0]

    jz = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    aB = actionAngleStaeckelInverse(jrs=0.05, Lzs=[1.0], jzs=jz, **kw)
    assert aB._bk["Es"].shape == (1,) and aB._bk["Es"].requires_grad
    R = aB(0.05, 1.0, jz, 0.4, 1.0, 2.0)[0]
    numpy.testing.assert_allclose(
        numpy.ravel(as_numpy(R))[0], numpy_R(0.03), rtol=0.0, atol=1e-9
    )
    (g,) = torch.autograd.grad(R.sum(), jz)
    numpy.testing.assert_allclose(
        float(g), _fd(numpy_R, 0.03, 1e-4), rtol=1e-6, atol=1e-8
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("case", ["fixed", "rule", "models", "forced"])
def test_tori_from_actions_chunked(case):
    # explicit tori set up natively in chunks (chunk=), each chunk on the
    # auxiliary fitted to all of them: the tables, points and derivatives of
    # the set-up in one pass, with a fixed focal length, a rule, per-torus
    # models (as the adiabatic family's inverse gives them), and integrals
    # solved on numpy under a forced backend; a last chunk shorter than the
    # others
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    if case == "models":
        kw = dict(
            pot=MWPotential2014,
            mm_npt=32,
            progressbar=False,
            _models=(torch.full((3,), 0.45), torch.full((3,), 1.5)),
        )
    else:
        kw = _native_kw(case == "rule")
    ang = (0.4, 1.0, 2.0)

    def build(chunk):
        jr = torch.tensor(_NATIVE_J["jrs"], requires_grad=case != "forced")
        Lz, jz = (torch.tensor(_NATIVE_J[k]) for k in ("Lzs", "jzs"))
        with gb.use("torch", force=case == "forced"):
            aB = actionAngleStaeckelInverse(jrs=jr, Lzs=Lz, jzs=jz, chunk=chunk, **kw)
            x = torch.stack(aB(jr, Lz, jz, *(a + 0.0 * jr for a in ang)))
        g = torch.autograd.grad(x.sum(), jr)[0] if jr.requires_grad else 0.0 * jr
        return aB, as_numpy(x), as_numpy(g)

    a1, x1, g1 = build(None)
    a2, x2, g2 = build(2)
    assert a2._bk["Es"].shape == (3,) and float(a2._GM) == pytest.approx(
        float(a1._GM), rel=1e-14
    )
    numpy.testing.assert_allclose(x2, x1, rtol=0.0, atol=1e-12)
    numpy.testing.assert_allclose(g2, g1, rtol=1e-9, atol=1e-12)
    for k in ("_Es", "_jrs", "_Ns", "_Dmus"):
        numpy.testing.assert_allclose(getattr(a2, k), getattr(a1, k), atol=1e-10)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="CUDA torch not available"
)
def test_tori_from_cuda_actions_cpu_default_device():
    # differentiated actions on the GPU with the CPU as the default device
    # (and a potential with numpy parameters): the tori are set up and
    # evaluated on the GPU
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.45, mm_npt=32, progressbar=False)
    jz = torch.tensor(0.03, dtype=torch.float64, device="cuda", requires_grad=True)
    aB = actionAngleStaeckelInverse(jrs=0.05, Lzs=[1.0], jzs=jz, **kw)
    R = aB(0.05, 1.0, jz, 0.4, 1.0, 2.0)[0]
    assert aB._bk["Es"].device.type == "cuda" and R.device.type == "cuda"
    aN = actionAngleStaeckelInverse(jrs=[0.05], Lzs=[1.0], jzs=[0.03], **kw)
    numpy.testing.assert_allclose(
        numpy.ravel(as_numpy(R))[0],
        numpy.ravel(aN(0.05, 1.0, 0.03, 0.4, 1.0, 2.0)[0])[0],
        rtol=0.0,
        atol=1e-9,
    )


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_construction_from_actions():
    # the whole set-up of tori from their actions, and their evaluation,
    # traces under jax.jit; actions of no torus return nan under the trace
    # (the others unchanged) and raise for concrete actions
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.27, u0="torus", mm_npt=32, progressbar=False)
    J = {k: list(v) for k, v in _NATIVE_J.items()}
    for k, v in zip(("jrs", "Lzs", "jzs"), _NO_TORUS):
        J[k].append(v)
    J = {k: numpy.array(v) for k, v in J.items()}

    # (each torus at its own angles, in one call: one trace of the evaluation)
    angs = (_ANGLES[:4], 1.0, 2.0)

    def f(jr, Lz, jz):
        with gb.use("jax", force=True):
            aB = actionAngleStaeckelInverse(jrs=jr, Lzs=Lz, jzs=jz, **kw)
            return jnp.stack(aB(jr, Lz, jz, *angs))

    out = numpy.asarray(jax.jit(f)(*(jnp.asarray(J[k]) for k in ("jrs", "Lzs", "jzs"))))
    good = {k: v[:3] for k, v in J.items()}
    aN = actionAngleStaeckelInverse(**good, **kw)
    ref = numpy.array(aN(good["jrs"], good["Lzs"], good["jzs"], _ANGLES[:3], 1.0, 2.0))
    numpy.testing.assert_allclose(out[:, :3], ref, rtol=0.0, atol=1e-9)
    assert numpy.all(numpy.isnan(out[:, 3])), (
        "actions of no torus do not return nan under jax.jit"
    )
    with pytest.raises(ValueError, match="No torus with the actions"):
        with gb.use("torch", force=True):
            actionAngleStaeckelInverse(
                **{k: torch.tensor(v, requires_grad=True) for k, v in J.items()}, **kw
            )


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("forced", [False, True])
def test_jax_jit_construction_from_actions_single_curve(forced):
    # tori on the model's single reference curve set up from their actions
    # under jax.jit: without a forced backend, the traced actions pick it;
    # under a forced one, the model, set up in the trace, gives V itself
    # (one torus per chunk: a jax.lax.map over the construction)
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.27, mm_npt=32, progressbar=False)
    J = {k: numpy.array(v[:2]) for k, v in _NATIVE_J.items()}

    def f(jr, Lz, jz):
        with gb.use("jax", force=True) if forced else contextlib.nullcontext():
            aB = actionAngleStaeckelInverse(jrs=jr, Lzs=Lz, jzs=jz, chunk=1, **kw)
            return jnp.stack(aB(jr, Lz, jz, _ANGLES[:2], 1.0, 2.0))

    out = numpy.asarray(jax.jit(f)(*(jnp.asarray(J[k]) for k in ("jrs", "Lzs", "jzs"))))
    ref = numpy.array(
        actionAngleStaeckelInverse(**J, **kw)(
            J["jrs"], J["Lzs"], J["jzs"], _ANGLES[:2], 1.0, 2.0
        )
    )
    numpy.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_bracket_steps_roll_under_jax_trace():
    # the bisections and golden-section searches of the turning points and
    # of the focal-length fit are one loop under a jax trace (their steps
    # are not unrolled: a traced delta='fit' family compiles), with numpy's
    # roots and minima
    from galpy.actionAngle.actionAngleStaeckelInverse import _bisect, _golden

    lo, hi = numpy.array([0.1, 0.5, 1.0]), numpy.array([2.0, 3.0, 1.6])
    for helper, fn in (
        (_bisect, lambda x: x**2 - 1.5),
        (_golden, lambda x: (x - 1.3) ** 2),
    ):

        def run(a, b, n):
            return helper(fn, a, b, n=n)

        sizes = [
            len(jax.make_jaxpr(lambda a, b: run(a, b, n))(lo, hi).jaxpr.eqns)
            for n in (10, 40)
        ]
        assert sizes[0] == sizes[1]
        out = jax.jit(lambda a, b: run(a, b, 40))(jnp.asarray(lo), jnp.asarray(hi))
        numpy.testing.assert_allclose(
            numpy.asarray(out), run(lo, hi, 40), rtol=0.0, atol=1e-12
        )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_shell_face_large_I3_backend():
    # a family set up on a backend: its nodes on the shell face stay on it
    # however large the third integral (as numpy's since main #2046: the
    # backend's node solve stops at a face), and their tori are finite and
    # of a single radius
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.potential import MWPotential2014

    kw = dict(pot=MWPotential2014, delta=0.006, u0="torus", setup_interp=True)
    kw.update(Rmin=0.0031, Rmax=0.0062, Rinf=16.0, nE=4, nLz=4, nI3=4)
    with gb.use("torch", force=True):
        aAI = actionAngleStaeckelInverse(**kw, mm_npt=64, progressbar=False)
        angles = numpy.linspace(0.1, 6.0, 5)
        for jz in (0.1, 1.0):
            R, vR, vT, z, vz, phi = numpy.array(
                [
                    as_numpy(t)
                    for t in aAI(
                        numpy.zeros(5),
                        0.0015 * numpy.ones(5),
                        jz * numpy.ones(5),
                        angles,
                        0.7 * angles,
                        1.3 * angles,
                    )
                ]
            )
            r = numpy.sqrt(R**2.0 + z**2.0)
            assert numpy.all(numpy.isfinite([R, vR, vT, z, vz, phi])), (
                "A shell torus of a backend family with a large third integral is not finite"
            )
            assert numpy.all(numpy.fabs(r / numpy.mean(r) - 1.0) < 1e-3), (
                "A shell torus of a backend family does not have a single radius"
            )


# ---------- the construction's contract and checks
@pytest.mark.parametrize("backend", BACKENDS)
def test_construction_contract(backend):
    # Under a forced backend, or for a potential with backend parameters, the
    # family is built natively and evaluates there, numpy inputs included
    from galpy import backend as gb

    aA = _explicit(pot=_pot(amp=_arr(backend, 4.0)))
    assert aA._bk is not None
    jr, jz = float(as_numpy(aA._bk["jrs"][0])), float(as_numpy(aA._bk["jzs"][0]))
    assert _is_backend(backend, aA(jr, 0.99, jz, 0.3, 1.0, 2.0)[0])
    assert _is_backend(backend, aA.Freqs(jr, 0.99, jz)[0])
    assert _is_backend(backend, aA.JR(*_LABELS[0]))
    with gb.use(backend, force=True):
        aA = _explicit()
    assert aA._bk is not None and _is_backend(backend, aA._bk["jrs"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_construction_checks(backend):
    # the construction's checks and warnings on a backend, as on numpy
    from galpy import backend as gb
    from galpy.potential import MiyamotoNagaiPotential
    from galpy.util import galpyWarning

    E, Lz, I3 = _LABELS[0]
    with gb.use(backend, force=True):
        with pytest.raises(ValueError, match="below the circular orbit's"):
            _explicit(labels=[(E - 1.0, Lz, I3)])
        with pytest.raises(ValueError, match="above the shell"):
            _explicit(labels=[(E, Lz, I3 + 1.0)])
        with pytest.raises(ValueError, match="below the planar"):
            _explicit(labels=[(E, Lz, I3 - 1.0)])
        with pytest.raises(ValueError, match="unbound"):
            _explicit(labels=[(0.5, Lz, I3)])
        with pytest.raises(ValueError, match="nE, nLz, and nI3 >= 4"):
            _interp(nE=3)
        with pytest.raises(ValueError, match="Rinf has to exceed Rmax"):
            _interp(Rinf=1.0)
        # as test_actionAngle.py's: an Rinf beyond Rmax but inside the zero-L_z
        # apocentre at the top circular energy, where the backend bracket of
        # that apocentre would not raise
        with pytest.raises(ValueError, match="Rinf is too small"):
            _interp(
                pot=MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.3),
                delta=0.4,
                Rmin=0.8,
                Rmax=1.5,
                Rinf=2.5,
                nE=4,
                nLz=4,
                nI3=4,
            )
        with pytest.raises(TypeError, match="IsochronePotential"):
            _explicit(auxiliary=_pot())
        with pytest.warns(galpyWarning, match="not converged for the"):
            _explicit(mm_npt=4, mm_nta=40)
        # test_actionAngle.py's deep radial orbit, whose lift with two
        # harmonics is unbound
        deep = (-0.25500291609372505, 0.15, 0.2685635457298501)
        with pytest.raises(
            RuntimeError,
            match="not bound in the fitted auxiliary.*at most about six times its "
            "square root",
        ):
            _explicit(labels=[deep], mm_npt=2, mm_nta=16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_evaluation_checks(backend, interp, explicit):
    # the evaluation's checks on backend inputs, as on numpy
    def a(*xs):
        return tuple(_arr(backend, x) for x in xs)

    E, Lz, I3 = _LABELS[0]
    ang = a(0.3, 1.0, 2.0)
    with pytest.raises(ValueError, match="could not be matched.*increase Rinf"):
        interp(*a(50.0, 0.93, 0.1), *ang)  # J_R outside the family
    with pytest.raises(ValueError, match="above the covered"):
        interp(*a(0.0, 0.93, 50.0), *ang)  # J_z outside, on the shell face
    with pytest.raises(ValueError, match="non-negative"):
        interp(*a(-0.1, 0.93, 0.1), *ang)
    with pytest.raises(ValueError, match="L_z = 0"):
        interp(*a(0.1, 0.0, 0.1), *ang)
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        interp.JR(*a(-10.0, 0.93, I3))  # below the circular orbit's energy
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        interp.Jz(*a(E, 0.93, I3 + 10.0))  # above the shell orbit's I_3
    Ec = interp._circular(0.93)["Ec"]
    assert float(as_numpy(interp.JR(*a(Ec, 0.93, I3)))) == 0.0
    assert float(as_numpy(interp.Jz(*a(Ec, 0.93, I3)))) == 0.0
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        explicit.Freqs(*a(0.123, 0.4, 0.2))
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        explicit(*a(0.123, 0.4, 0.2), *ang)
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        explicit.Freqs(*a(0.0, 0.4, 0.0))  # a circular request that is not a node
    with pytest.raises(ValueError, match="not one of the set-up tori"):
        explicit.JR(*a(E + 0.1, Lz, I3))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_maxiter_fallback(interp, explicit):
    # with maxiter=0 the angle solve takes the safeguarded steps and agrees
    # with Newton (torch: under eager jax these steps take minutes; the
    # forced-jax run of test_actionAngle.py's maxiter test covers jax)
    angler = torch.tensor(numpy.linspace(0.05, 6.2, 11))
    jr, jz = (float(as_numpy(t)) for t in (explicit._jrs[0], explicit._jzs[0]))
    for aA, J in ((interp, (0.06, 0.93, 0.10)), (explicit, (jr, _LABELS[0][1], jz))):
        fb = copy.copy(aA)
        fb._maxiter = 0
        J = tuple(torch.tensor(j) for j in J)
        numpy.testing.assert_allclose(
            _flat(fb(*J, angler, 1.0, 2.0)),
            _flat(aA(*J, angler, 1.0, 2.0)),
            rtol=0.0,
            atol=1e-9,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_hermite_family_3d(backend):
    # HermiteFamily3D built on a backend agrees with the numpy one, called
    # like ip(x, y, z, dx=, dy=, dz=), and so does the numpy one at backend
    # points
    from galpy.util._hermite import HermiteFamily3D

    x, y, z = (
        numpy.linspace(0.0, 1.0, 6) ** 2,
        numpy.linspace(0.5, 2.0, 5),
        numpy.linspace(-1.0, 1.0, 5),
    )
    X, Y, Z = numpy.meshgrid(x, y, z, indexing="ij")
    g, e = numpy.sin(3.0 * X) * numpy.cos(Y), numpy.exp(0.5 * Z)
    tabs = [
        (t * e)[..., None] * numpy.array([1.0, 2.0])
        for t in (
            g,
            3.0 * numpy.cos(3.0 * X) * numpy.cos(Y),
            -numpy.sin(3.0 * X) * numpy.sin(Y),
            0.5 * g,
        )
    ]
    ipn = HermiteFamily3D(x, y, z, *tabs)
    ipb = HermiteFamily3D(x, _arr(backend, y), z, *(_arr(backend, t) for t in tabs))
    for pt in ((0.3, 0.7, 0.2), (x[2], y[3], z[1]), (0.999, 1.99, -0.95)):
        for d in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
            kw = dict(zip(("dx", "dy", "dz"), d))
            ref = ipn(*pt, **kw)[0, 0]
            for ip in (ipb, ipn):
                got = ip(*(_arr(backend, v) for v in pt), **kw)[0, 0]
                assert _is_backend(backend, got)
                numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0.0, atol=1e-12)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_tabulated_staeckel_model():
    # a tabulated Staeckel model (ntab=; ~2e-7 off the potential in the plane
    # here) is its own in the plane too: a family built on a backend agrees
    # with numpy's, at the circular orbit as elsewhere
    from galpy import backend as gb
    from galpy.potential import MWPotential2014, OblateStaeckelWrapperPotential

    pot = OblateStaeckelWrapperPotential(pot=MWPotential2014, delta=0.45, ntab=100)
    aN = _interp(pot=pot)
    with gb.use("torch", force=True):
        aB = _interp(pot=pot)
    for J in ((0.06, 0.93, 0.10), (0.0, 1.1, 0.0)):
        got = aB.xvFreqs(*(_arr("torch", v) for v in (*J, _ANGLES)), 1.0, 2.0)
        numpy.testing.assert_allclose(
            _flat(got), _flat(aN.xvFreqs(*J, _ANGLES, 1.0, 2.0)), rtol=0.0, atol=1e-11
        )


# arrays of actions: the interpolated family's tori as rows (generic,
# retrograde, the shell and planar faces, the circular orbit) and the first
# repeated, each at its own angles
_ARR_TORI = numpy.array(_TORI + [_TORI[0]])
_ARR_ANG = numpy.array(
    [
        [0.4, 1.0, 2.0, 5.0, 3.0, 0.4],
        [1.0, 2.5, 0.3, 3.0, 1.0, 1.0],
        [2.0, 0.7, 4.0, 0.7, 6.0, 2.0],
    ]
)


def _arr_tori(mode, explicit):
    if mode == "interp":
        return _ARR_TORI
    return numpy.array(
        [(explicit._jrs[k], explicit._Lzs[k], explicit._jzs[k]) for k in (0, 1, 0)]
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["explicit", "interp"])
def test_array_actions_match_numpy(backend, mode, explicit, interp):
    # arrays of actions broadcast against the angles as on numpy and agree
    # with it point by point: paired, every angle on every torus, an array of
    # actions at scalar angles, and the frequencies, with the distinct tori
    # evaluated once each (concrete actions); and paired when every point is
    # its own torus (actions that carry a derivative: torch's requires_grad,
    # jax's JVP tracers)
    aA = explicit if mode == "explicit" else interp
    J = _arr_tori(mode, explicit)
    n = len(J)
    ang = _ARR_ANG[:, :n]
    cases = [
        ((J[:, 0], J[:, 1], J[:, 2]), tuple(ang)),
        ((J[:, 0:1], J[:, 1:2], J[:, 2:3]), (_ANGLES[:4], 0.2 + _ANGLES[:4], 1.0)),
        ((J[:, 0], J[:, 1], J[:, 2]), (0.3, 1.0, 2.0)),
    ]

    def check(ref, got):
        for r, g in zip(ref, got):
            assert _is_backend(backend, g)
            assert tuple(g.shape) == numpy.shape(r)
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=_PLANE_ATOL)

    for acts, angs in cases:
        check(
            aA.xvFreqs(*acts, *angs),
            aA.xvFreqs(*(_arr(backend, numpy.array(a)) for a in acts), *angs),
        )
    acts, angs = cases[0]
    bacts = [_arr(backend, numpy.array(a), grad=True) for a in acts]
    if backend == "jax":
        got = jax.jvp(
            lambda *a: aA.xvFreqs(*a, *angs),
            tuple(bacts),
            tuple(jnp.ones_like(b) for b in bacts),
        )[0]
    else:
        got = aA.xvFreqs(*bacts, *angs)
    check(aA.xvFreqs(*acts, *angs), got)
    ref = aA.Freqs(J[:, 0:1], J[:, 1:2], J[:, 2:3])
    got = aA.Freqs(*(_arr(backend, numpy.array(J[:, k : k + 1])) for k in range(3)))
    for r, g in zip(ref, got):
        assert tuple(g.shape) == (n, 1)
        numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=_PLANE_ATOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_array_actions_derivatives(backend, interp):
    # each point's (x, v) depends on its own actions only, also when two
    # points share a torus: the autodiff Jacobian in an array of radial
    # actions is diagonal, its diagonal the finite difference of the numpy
    # path (tori off J_R = 0, so that the stencil stays inside the domain)
    J = _ARR_TORI[[0, 1, 3, 5]]
    ang = _ARR_ANG[:, [0, 1, 3, 5]]
    jphi, jz = J[:, 1], J[:, 2]

    def R(jr):
        return interp.xvFreqs(jr, jphi, jz, *ang)[0]

    if backend == "jax":
        jac = numpy.asarray(jax.jacfwd(lambda jr: R(jr))(jnp.asarray(J[:, 0])))
    else:
        jac = as_numpy(
            torch.autograd.functional.jacobian(lambda jr: R(jr), torch.tensor(J[:, 0]))
        )
    assert numpy.all(jac[~numpy.eye(len(J), dtype=bool)] == 0.0)
    for k in range(len(J)):
        e = numpy.zeros(len(J))
        e[k] = 1.0
        fd = _fd(lambda h: numpy.asarray(R(J[:, 0] + h * e))[k], 0.0, 1e-4)
        assert numpy.fabs(jac[k, k] - fd) < 2e-10


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_array_actions_derivatives_wrt_potential_parameters():
    # the distinct tori of an array, each evaluated once, carry the
    # derivatives in the potential's parameters: those of the array are the
    # single tori's
    pt = torch.tensor(4.0, requires_grad=True)
    aA = _param_family(pt, "interp")
    J = _ARR_TORI[[0, 1, 5]]
    ang = _ARR_ANG[:, [0, 1, 5]]
    out = aA.xvFreqs(*(J[:, k] for k in range(3)), *ang)
    # (every evaluation shares the family's construction graph)
    g = torch.autograd.grad(sum(o.sum() for o in out[:6]), pt, retain_graph=True)[0]
    g1 = sum(
        torch.autograd.grad(
            sum(o.sum() for o in aA.xvFreqs(*J[k], *ang[:, k : k + 1])[:6]),
            pt,
            retain_graph=True,
        )[0]
        for k in range(len(J))
    )
    assert numpy.fabs(float(g) - float(g1)) < 1e-10 * (1.0 + numpy.fabs(float(g1)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_hermite_ev4_rows(backend):
    # ev4_rows evaluates many points at once as ev4 does one at a time and as
    # the numpy batch_all does, for a family built on a backend and on numpy
    from galpy.util._hermite import HermiteFamily3D

    x, y, z = (
        numpy.linspace(0.0, 1.0, 6) ** 2,
        numpy.linspace(0.5, 2.0, 5),
        numpy.linspace(-1.0, 1.0, 5),
    )
    X, Y, Z = numpy.meshgrid(x, y, z, indexing="ij")
    g, e = numpy.sin(3.0 * X) * numpy.cos(Y), numpy.exp(0.5 * Z)
    tabs = [
        (t * e)[..., None] * numpy.array([1.0, 2.0])
        for t in (
            g,
            3.0 * numpy.cos(3.0 * X) * numpy.cos(Y),
            -numpy.sin(3.0 * X) * numpy.sin(Y),
            0.5 * g,
        )
    ]
    ipn = HermiteFamily3D(x, y, z, *tabs)
    ipb = HermiteFamily3D(x, _arr(backend, y), z, *(_arr(backend, t) for t in tabs))
    pts = numpy.array(
        [(0.3, 0.7, 0.2), (x[2], y[3], z[1]), (0.999, 1.99, -0.95), (0.0, 0.5, -1.0)]
    )
    ref = ipn.batch_all(pts[:, 0], pts[:, 1], pts[:, 2])
    for ip in (ipb, ipn):
        got = ip.ev4_rows(*(_arr(backend, pts[:, k]) for k in range(3)))
        for r, gt in zip(ref, got):
            assert _is_backend(backend, gt) and tuple(gt.shape) == r.shape
            numpy.testing.assert_allclose(as_numpy(gt), r, rtol=0.0, atol=1e-12)
        for k in range(len(pts)):
            one = ip.ev4(*(_arr(backend, v) for v in pts[k]))
            for o, gt in zip(one, got):
                numpy.testing.assert_allclose(
                    as_numpy(gt)[k], as_numpy(o), rtol=0.0, atol=1e-14
                )
    # more points than a chunk of the read holds (2604 for two tables), also
    # under jax.jit
    many = numpy.random.default_rng(2).uniform(
        [0.0, 0.5, -1.0], [1.0, 2.0, 1.0], (3000, 3)
    )
    ref = ipn.batch_all(*many.T)
    ev = ipb.ev4_rows
    if backend == "jax":
        import jax

        ev = jax.jit(ipb.ev4_rows)
    for r, gt in zip(ref, ev(*(_arr(backend, c) for c in many.T))):
        numpy.testing.assert_allclose(as_numpy(gt), r, rtol=0.0, atol=1e-12)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_array_actions(interp):
    # arrays of tori trace once under jax.jit, their faces and circular
    # orbits selected by where(), and agree with numpy: paired, every angle
    # on every torus, and the frequencies
    J, ang = _ARR_TORI, _ARR_ANG
    f = jax.jit(lambda a, b, c: jnp.stack(interp.xvFreqs(a, b, c, *ang)))
    numpy.testing.assert_allclose(
        numpy.asarray(f(*(jnp.asarray(J[:, k]) for k in range(3)))),
        numpy.array(interp.xvFreqs(J[:, 0], J[:, 1], J[:, 2], *ang)),
        rtol=0.0,
        atol=_PLANE_ATOL,
    )
    a4 = _ANGLES[:4]
    h = jax.jit(lambda a, b, c: jnp.stack(interp.xvFreqs(a, b, c, a4, 1.0, 2.0)))
    numpy.testing.assert_allclose(
        numpy.asarray(h(*(jnp.asarray(J[:, k : k + 1]) for k in range(3)))),
        numpy.array(interp.xvFreqs(J[:, 0:1], J[:, 1:2], J[:, 2:3], a4, 1.0, 2.0)),
        rtol=0.0,
        atol=_PLANE_ATOL,
    )
    g = jax.jit(lambda a, b, c: jnp.stack(interp.Freqs(a, b, c)))
    numpy.testing.assert_allclose(
        numpy.asarray(g(*(jnp.asarray(J[:, k]) for k in range(3)))),
        numpy.array(interp.Freqs(J[:, 0], J[:, 1], J[:, 2])),
        rtol=0.0,
        atol=_PLANE_ATOL,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunked_evaluation(backend, interp):
    # chunk= evaluates arrays of actions in equal chunks of at most that many
    # points: the values of one pass, eagerly and under jax.jit, where the
    # compiled program is one chunk's (it does not grow with the points)
    assert interp.chunk is None
    J, ang = _ARR_TORI, _ARR_ANG
    acts = [_arr(backend, J[:, k]) for k in range(3)]
    angs = [_arr(backend, a) for a in ang]
    one = _flat(interp.xvFreqs(*acts, *angs)), _flat(interp.Freqs(*acts))
    try:
        interp.chunk = 4
        got = _flat(interp.xvFreqs(*acts, *angs)), _flat(interp.Freqs(*acts))
        for g, o in zip(got, one):
            numpy.testing.assert_allclose(g, o, rtol=0.0, atol=1e-13)
        if backend == "jax":

            def f(ang):
                return lambda a, b, c: jnp.stack(interp.xvFreqs(a, b, c, *ang))

            J2, ang2 = numpy.concatenate([J, J]), numpy.concatenate([ang, ang], axis=1)
            jaxprs = [
                jax.make_jaxpr(f(an))(*(jnp.asarray(JJ[:, k]) for k in range(3))).jaxpr
                for JJ, an in ((J, ang), (J2, ang2))
            ]
            assert len(jaxprs[0].eqns) == len(jaxprs[1].eqns), (
                "the chunked program grows with the points"
            )
            # (the chunks are one jax.lax.map)
            assert any(e.primitive.name == "scan" for e in jaxprs[1].eqns)
            numpy.testing.assert_allclose(
                numpy.asarray(jax.jit(f(ang))(*acts)).ravel(),
                one[0],
                rtol=0.0,
                atol=1e-11,
            )
    finally:
        interp.chunk = None


def _delta_rule_family():
    # a family whose focal length follows its tori (built on numpy)
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use

    def rule(E, Lz, y):
        return (
            1.3
            * (1.0 + 0.15 * numpy.tanh(Lz - 1.0) + 0.2 * (y - 0.5))
            * (1.0 + 0.05 * E)
        )

    with use("numpy", force=True):
        return actionAngleStaeckelInverse(
            pot=_pot(), delta=rule, setup_interp=True, Rmin=0.7, Rmax=1.6,
            Rinf=3.5, nE=4, nLz=4, nI3=4, mm_npt=32, progressbar=False,
        )  # fmt: skip


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("kind", ["torus", "callable", "delta"])
def test_jax_jit_array_actions_adaptive(kind):
    # a family whose reference curve (or focal length) is per torus evaluates
    # traced arrays of tori in the one vectorized pass, each row with its own
    # chart (and focal length): the traced program does not grow with the
    # number of tori (it was one unrolled torus per point), it agrees with
    # numpy, faces and the circular orbit too, and its derivatives in the
    # actions match numpy's finite differences
    if kind == "delta":
        aA = _delta_rule_family()
    else:
        u0 = (
            "torus"
            if kind == "torus"
            else lambda E, Lz: 1.0 + 0.2 * numpy.tanh(Lz) + 0.1 * E
        )  # noqa: E731
        aA = _interp(u0=u0)

    def f(ang):
        return lambda a, b, c: jnp.stack(aA.xvFreqs(a, b, c, *ang))

    def acts(J):
        return [jnp.asarray(J[:, k]) for k in range(3)]

    neqns = [
        len(jax.make_jaxpr(f(ang))(*acts(J)).jaxpr.eqns)
        for J, ang in ((_ARR_TORI[:3], _ARR_ANG[:, :3]), (_ARR_TORI, _ARR_ANG))
    ]
    assert neqns[0] == neqns[1], "the traced pass grows with the number of tori"
    J, ang = _ARR_TORI, _ARR_ANG
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(f(ang))(*acts(J))),
        numpy.array(aA.xvFreqs(J[:, 0], J[:, 1], J[:, 2], *ang)),
        rtol=0.0,
        atol=1e-11,
    )
    g = jax.jit(lambda a, b, c: jnp.stack(aA.Freqs(a, b, c)))
    numpy.testing.assert_allclose(
        numpy.asarray(g(*acts(J))),
        numpy.array(aA.Freqs(J[:, 0], J[:, 1], J[:, 2])),
        rtol=0.0,
        atol=1e-11,
    )
    # d(x, v, Omega)/dJ_R of the generic and the retrograde torus
    J2, ang2 = _ARR_TORI[:2], _ARR_ANG[:, :2]
    jac = numpy.asarray(
        jax.jit(jax.jacfwd(lambda a: f(ang2)(a, *acts(J2)[1:])))(acts(J2)[0])
    )
    h = 1e-5
    fd = (
        numpy.array(aA.xvFreqs(J2[:, 0] + h, J2[:, 1], J2[:, 2], *ang2))
        - numpy.array(aA.xvFreqs(J2[:, 0] - h, J2[:, 1], J2[:, 2], *ang2))
    ) / (2.0 * h)
    numpy.testing.assert_allclose(
        numpy.einsum("ijj->ij", jac),
        fd,
        rtol=0.0,
        atol=1e-6 * numpy.amax(numpy.fabs(fd)),
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_array_actions_fallbacks(interp, monkeypatch):
    # points whose vectorized Newton iteration is cut at one step are solved
    # again all at once, from their requested angles; tiny actions; points
    # whose label inversion does not converge in the backend (here: flagged)
    # take the one-torus path; a torus beyond the family and an |L_z| beyond
    # the grid raise as for one torus (torch: under eager jax the one-torus
    # path's safeguarded steps take minutes)
    J, ang = _ARR_TORI[[0, 1, 5, 0]], _ARR_ANG[:, [0, 1, 5, 2]]
    J[3] = (0.09, 1.05, 0.05)
    fb = copy.copy(interp)
    fb._maxiter = 1
    tt = [torch.tensor(J[:, k]) for k in range(3)]
    numpy.testing.assert_allclose(
        _flat(fb.xvFreqs(*tt, *ang)),
        _flat(interp.xvFreqs(J[:, 0], J[:, 1], J[:, 2], *ang)),
        rtol=0.0,
        atol=1e-9,
    )
    tiny = numpy.array([1e-12, 1e-8, 1e-8, 0.05])
    tinyz = numpy.array([1e-12, 1e-8, 0.05, 1e-8])
    Lt = numpy.ones(4)
    numpy.testing.assert_allclose(
        _flat(interp(*(torch.tensor(v) for v in (tiny, Lt, tinyz)), 0.3, 1.0, 2.0)),
        _flat(interp(tiny, Lt, tinyz, 0.3, 1.0, 2.0)),
        rtol=0.0,
        atol=1e-11,
    )
    coords = type(interp)._bk_interp_coords_rows

    def flagged(self, *args, **kwargs):
        x, y, bad = coords(self, *args, **kwargs)
        if bad is not None:
            bad = numpy.array(bad)
            bad[0] = True
        return x, y, bad

    with monkeypatch.context() as m:
        m.setattr(type(interp), "_bk_interp_coords_rows", flagged)
        numpy.testing.assert_allclose(
            _flat(interp.xvFreqs(*tt, *ang)),
            _flat(interp.xvFreqs(J[:, 0], J[:, 1], J[:, 2], *ang)),
            rtol=0.0,
            atol=1e-9,
        )
    # an array of one distinct torus takes the one-torus path
    numpy.testing.assert_allclose(
        _flat(interp(*(torch.tensor([v, v]) for v in J[0]), ang[0, :2], 1.0, 2.0)),
        _flat(interp(*(numpy.array([v, v]) for v in J[0]), ang[0, :2], 1.0, 2.0)),
        rtol=0.0,
        atol=_PLANE_ATOL,
    )
    with pytest.raises(ValueError, match="could not be matched"):
        interp(
            torch.tensor([0.05, 50.0]),
            torch.tensor([0.93, 0.93]),
            torch.tensor([0.05, 0.1]),
            1.0,
            1.0,
            1.0,
        )
    with pytest.raises(ValueError, match="outside the interpolation grid"):
        interp(
            torch.tensor([0.05, 0.06]), torch.tensor([0.93, 5.0]), 0.05, 1.0, 1.0, 1.0
        )


# ---------- a focal length that follows the tori
@pytest.mark.parametrize("backend", BACKENDS)
def test_delta_family_on_backends(backend):
    # a family whose focal length follows its tori is built natively under a
    # forced backend, with numpy's tables (to the callable's finite
    # differences) and points, and a numpy-built one evaluates natively with
    # backend inputs, in the vectorized pass, each point in the coordinates
    # of its torus's own focal length: numpy's values, and derivatives in the
    # angles and actions that match a finite difference and keep the map
    # canonical
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use

    grid = dict(setup_interp=True, Rmin=0.7, Rmax=1.6, Rinf=3.5, progressbar=False)
    grid.update(nE=4, nLz=4, nI3=4, mm_npt=32)

    def rule(E, Lz, y):
        return (
            1.3
            * (1.0 + 0.15 * numpy.tanh(Lz - 1.0) + 0.2 * (y - 0.5))
            * (1.0 + 0.05 * E)
        )

    with use(backend, force=True):
        bk = actionAngleStaeckelInverse(pot=_pot(), delta=rule, **grid)
    with use("numpy", force=True):
        aA = actionAngleStaeckelInverse(pot=_pot(), delta=rule, **grid)
    for k in ("_A_tab", "_A_dx", "_A_dy", "_A_dL", "_U_tab", "_V_tab"):
        numpy.testing.assert_allclose(
            as_numpy(getattr(bk, k)), getattr(aA, k), rtol=0.0, atol=2e-9
        )
    acts = numpy.array([[0.06, 0.93, 0.10], [0.15, -1.21, 0.03], [0.02, 0.8, 0.2]]).T
    angs = numpy.array([[0.3, 2.0, 4.1], [1.0, 4.0, 0.2], [2.0, 5.5, 3.3]])
    ref = aA.xvFreqs(*acts, *angs)
    with use(backend, force=True):
        numpy.testing.assert_allclose(
            _flat(bk.xvFreqs(*acts, *angs)), _flat(ref), rtol=0.0, atol=1e-10
        )
    got = aA.xvFreqs(
        *(_arr(backend, q) for q in acts), *(_arr(backend, q) for q in angs)
    )
    assert all(_is_backend(backend, g) for g in got)
    numpy.testing.assert_allclose(_flat(got), _flat(ref), rtol=0.0, atol=1e-12)
    x0 = numpy.array([0.4, 1.0, 2.0, 0.06, 0.93, 0.10])

    def qp_np(x):
        return numpy.array(_qp(aA(x[3], x[4], x[5], *x[:3])))

    A = numpy.empty((6, 6))
    for k in range(6):
        e = numpy.zeros(6)
        e[k] = 1.0
        A[:, k] = _fd(lambda h: qp_np(x0 + h * e), 0.0, 1e-4)
    J = _jacobian(backend, aA, x0)
    numpy.testing.assert_allclose(J, A, rtol=0.0, atol=2e-9)
    assert _defect(J) < 1e-8
    # (E, L_z, I_3) do not label its tori, as on numpy; its delta method
    # reads backend inputs on the host
    E, y = _arr(backend, -1.6), _arr(backend, 0.5)
    with pytest.raises(NotImplementedError, match="do not label"):
        aA.JR(E, _arr(backend, 0.93), _arr(backend, 2.0))
    assert aA.delta(E, _arr(backend, 0.93), y) == aA.delta(-1.6, 0.93, 0.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_torus_focal_length_broadcast(backend):
    # the per-torus focal length of a construction broadcasts along the tori's
    # axis, and refuses an array whose leading axis does not run over them
    from galpy.actionAngle.actionAngleStaeckelInverse import _PerTorus

    D = _PerTorus(_arr(backend, numpy.array([1.0, 2.0, 3.0])))
    x = _arr(backend, numpy.ones((3, 2)))
    numpy.testing.assert_allclose(as_numpy(2.0 * D * x)[:, 1], [2.0, 4.0, 6.0])
    numpy.testing.assert_allclose(as_numpy(1.0 / (-D) + x)[:, 0], [0.0, 0.5, 2.0 / 3.0])
    with pytest.raises(ValueError, match="does not run over the tori"):
        D * _arr(backend, numpy.ones((2, 3)))


def _log_halo(q):
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(normalize=1.0, q=q)


_LOG_GRID = dict(
    setup_interp=True,
    Rmin=0.7,
    Rmax=1.5,
    Rinf=4.0,
    progressbar=False,
    nE=4,
    nLz=4,
    nI3=4,
    mm_npt=32,
    delta_deg=(1, 1, 1),
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_delta_fit_family_native(backend):
    # delta='fit' natively: the fit's optima and polynomial on the device
    # are numpy's, and so are the family's tables and points
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use

    with use("numpy", force=True):
        aA = actionAngleStaeckelInverse(pot=_log_halo(0.8), delta="fit", **_LOG_GRID)
    with use(backend, force=True):
        bk = actionAngleStaeckelInverse(pot=_log_halo(0.8), delta="fit", **_LOG_GRID)
    numpy.testing.assert_allclose(bk._delta_fit[0], aA._delta_fit[0], atol=1e-10)
    numpy.testing.assert_allclose(bk._delta_fit[1:], aA._delta_fit[1:], atol=1e-12)
    for k in ("_A_tab", "_A_dx", "_A_dy", "_A_dL"):
        numpy.testing.assert_allclose(
            as_numpy(getattr(bk, k)), getattr(aA, k), rtol=0.0, atol=1e-10
        )
    acts = numpy.array([[0.05, 1.0, 0.04], [0.12, 0.95, 0.08], [0.03, 1.15, 0.15]]).T
    angs = numpy.array([[0.3, 2.0, 4.1], [1.0, 4.0, 0.2], [2.0, 5.5, 3.3]])
    with use(backend, force=True):
        got = bk.xvFreqs(*acts, *angs)
    numpy.testing.assert_allclose(
        _flat(got), _flat(aA.xvFreqs(*acts, *angs)), rtol=0.0, atol=1e-11
    )
    # its delta method is numpy's
    E, Lz, y = (
        numpy.array([0.6, 0.9]),
        numpy.array([0.8, 1.2]),
        numpy.array([0.2, 0.7]),
    )
    numpy.testing.assert_allclose(bk.delta(E, Lz, y), aA.delta(E, Lz, y), atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rule", ["callable", "fit"])
def test_delta_family_target_native(backend, rule):
    # target= with a focal length that follows the tori, natively: each
    # point labelled in the model of its own focal length (the rule's, or
    # the fit's optimum at the point), numpy's box and tables; a rule that
    # is not positive raises as on numpy
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use
    from galpy.orbit import Orbit

    lp = _log_halo(0.8)
    o = Orbit([1.05, 0.12, 1.05, 0.12, 0.06, 0.0])
    ts = numpy.linspace(0.0, 30.0, 31)
    o.integrate(ts, lp)

    def callable_rule(E, Lz, f):
        return 0.8 * (1.0 - 0.25 * f + 0.1 * (Lz - 1.0) + 0.05 * E)

    kw = dict(_LOG_GRID, target=o(ts))
    delta = callable_rule if rule == "callable" else "fit"
    with use("numpy", force=True):
        aA = actionAngleStaeckelInverse(pot=lp, delta=delta, **kw)
    with use(backend, force=True):
        bk = actionAngleStaeckelInverse(pot=lp, delta=delta, **kw)
    numpy.testing.assert_allclose(
        numpy.array(bk._box), numpy.array(aA._box), rtol=0.0, atol=1e-10
    )
    numpy.testing.assert_allclose(as_numpy(bk._A_tab), aA._A_tab, rtol=0.0, atol=1e-10)
    if rule == "callable":
        with use(backend, force=True):
            with pytest.raises(ValueError, match="positive focal length"):
                actionAngleStaeckelInverse(
                    pot=lp, delta=lambda E, Lz, f: 0.5 - f, **_LOG_GRID
                )


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("rule", ["callable", "fit"])
def test_delta_family_derivatives_wrt_potential_parameters(rule):
    # A family whose focal length follows its tori, differentiated in the
    # potential's parameters with its focal lengths held (a gauge, as its
    # reference curves): the flattening of a logarithmic halo, which leaves
    # the midplane, the grid's energies and, with axis_weight=0, its nodes'
    # fractions of the third integral as they are, so that every node keeps
    # its focal length; against a 5-point finite difference of the
    # construction with the rule (the fitted one: the family's delta method)
    from galpy.actionAngle import actionAngleStaeckelInverse

    grid = dict(_LOG_GRID, axis_weight=0.0)

    def callable_rule(E, Lz, f):
        return 0.8 * (1.0 - 0.25 * f + 0.1 * (Lz - 1.0) + 0.05 * E)

    acts = [torch.tensor(v) for v in ([0.05, 0.12], [1.0, 0.95], [0.04, 0.08])]
    angs = [torch.tensor(v) for v in ([0.3, 2.0], [1.0, 4.0], [2.0, 5.5])]

    def out(fam):
        return torch.cat([torch.reshape(o, (-1,)) for o in fam.xvFreqs(*acts, *angs)])

    def fam(q, delta):
        return actionAngleStaeckelInverse(pot=_log_halo(q), delta=delta, **grid)

    q0 = 0.8
    qt = torch.tensor(q0, requires_grad=True)
    f0 = fam(qt, callable_rule if rule == "callable" else "fit")
    y = out(f0)
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], qt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )
    held = callable_rule if rule == "callable" else f0.delta
    numpy.testing.assert_allclose(
        g,
        _fd(lambda q: as_numpy(out(fam(torch.tensor(q), held)).detach()), q0, 1e-3),
        rtol=0.0,
        atol=2e-9,
    )


def _rule_tori_args():
    # the families and explicit tori of test_actionAngleStaeckelInverse_
    # delta_explicit: a fitted rule and a callable, on and off the grid
    lp = _log_halo(0.8)
    wide = dict(setup_interp=True, Rmin=0.7, Rmax=1.5, Rinf=4.0, progressbar=False)
    wide.update(nE=4, nLz=4, nI3=4, mm_npt=32)
    J = numpy.array([(0.08, 1.05, 0.06), (0.2, 0.3, 0.6)])
    angs = numpy.array([[0.3, 2.0], [1.0, 4.0], [2.0, 5.5]])
    return lp, wide, J, angs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rule", ["callable", "family", "backend_family"])
def test_rule_tori_on_backends(backend, rule):
    # Explicit tori whose focal lengths follow a rule (delta= a callable, or
    # the delta method of a fitted family set up with numpy or on the
    # backend), set up on a backend from their actions (the integrals on the
    # host, the tables natively) and from their integrals (the rule's fixed
    # point natively, on every torus at once): numpy's tori, their focal
    # lengths and their map; a rule that is not positive raises, and one
    # without a fixed point does not converge, as on numpy
    from galpy.actionAngle import actionAngleStaeckelInverse
    from galpy.backend import use

    if rule == "backend_family" and backend != "torch":
        pytest.skip("a fitted family built on jax: test_delta_fit_family_native")
    lp, wide, J, angs = _rule_tori_args()
    with use("numpy", force=True):
        fit = actionAngleStaeckelInverse(
            pot=lp, delta="fit", delta_deg=(1, 1, 1), **wide
        )

    def callable_rule(E, Lz, f):
        return 0.8 * (1.0 + 0.2 * (f - 0.5) + 0.05 * E * f + 0.1 * Lz)

    def explicit(delta, **kw):
        if "Es" not in kw:
            kw.update(jrs=J[:, 0], jzs=J[:, 2])
        kw.update(mm_npt=32, auxiliary=fit.auxiliary, progressbar=False)
        return actionAngleStaeckelInverse(pot=lp, delta=delta, Lzs=J[:, 1], **kw)

    def points(aA, k):
        out = aA.xvFreqs(*J[k], *angs)
        return numpy.array([as_numpy(o) + 0.0 * angs[0] for o in out])

    if rule == "backend_family":
        with use(backend, force=True):
            fam = actionAngleStaeckelInverse(
                pot=lp, delta="fit", delta_deg=(1, 1, 1), **wide
            )
        delta = fam.delta
    else:
        delta = callable_rule if rule == "callable" else fit.delta
    with use("numpy", force=True):
        ref = explicit(callable_rule if rule == "callable" else fit.delta)
        refs = [points(ref, k) for k in range(len(J))]
    with use(backend, force=True):
        aA = explicit(delta)
        aE = explicit(delta, Es=ref._Es, I3s=ref._I3s)
    for k in range(len(J)):
        for bk in (aA, aE):
            numpy.testing.assert_allclose(points(bk, k), refs[k], rtol=0.0, atol=1e-8)
    with use(backend, force=True):
        with pytest.raises(ValueError, match="positive focal lengths"):
            explicit(lambda E, L, f: 0.5 - f + 0.0 * E, Es=ref._Es, I3s=ref._I3s)
        with pytest.raises(RuntimeError, match="did not converge"):
            actionAngleStaeckelInverse(
                pot=lp,
                delta=lambda E, L, f: numpy.where(f < 0.4, 0.9, 0.5),
                Es=[0.84],
                Lzs=[0.9],
                I3s=[0.9],
                mm_npt=32,
                auxiliary=fit.auxiliary,
                progressbar=False,
            )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_bracket_compaction_on_backends():
    # actionAngleStaeckel.from_inverse() of many points, of which a few need the
    # reference curve bracketed (close to the symmetry axis, where the
    # iteration does not converge on its own) and one is on no torus
    # (unbound): eagerly, the brackets run on the points in need alone, and
    # give numpy's tori
    import warnings

    from galpy.actionAngle import actionAngleStaeckel, actionAngleStaeckelInverse
    from galpy.backend import use
    from galpy.potential import MWPotential2014
    from galpy.util import galpyWarning

    with use("numpy", force=True):
        fam = actionAngleStaeckelInverse(
            pot=MWPotential2014, delta=0.3, u0="torus", setup_interp=True,
            nE=4, nLz=4, nI3=4, mm_npt=16, progressbar=False,
        )  # fmt: skip
    rng = numpy.random.default_rng(4)
    n = 29
    # (the points close to the axis of test_actionAngleStaeckel_inverse_nearaxis)
    near = numpy.array(
        [
            [1.3267761370656899, -0.2848181700458333, 0.008451579340850553,
             3.1377831825578557, -0.44892690709616395],
            [0.021903441859716655, -1.8349049193596925, -1.146280448585597,
             0.3001546680366327, 0.3538612503538491],
            [1.853481323788396, 0.04553761502142584, 0.0006463128272536712,
             -2.433172469481341, -0.28105638136566685],
            [0.003737462681295734, -2.115570785838872, -0.26604529254019094,
             -0.31242862791340587, 0.09782480253575135],
            [1.0, 3.0, 1.0, 0.1, 0.2],  # (unbound)
        ]
    )  # fmt: skip
    x = [
        numpy.concatenate([t, near[:, k]])
        for k, t in enumerate(
            (
                rng.uniform(0.8, 1.2, n),
                rng.normal(0.0, 0.05, n),
                rng.uniform(0.9, 1.1, n),
                rng.normal(0.0, 0.05, n),
                rng.normal(0.0, 0.05, n),
            )
        )
    ]
    aAS = actionAngleStaeckel.from_inverse(fam, c=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", galpyWarning)
        ref = numpy.array(aAS(*x))
        out = numpy.array([as_numpy(o) for o in aAS(*[torch.tensor(a) for a in x])])
    assert numpy.all(numpy.isnan(out[[0, 2], -1])) and numpy.all(
        numpy.isfinite(out[:, :-1])
    )
    numpy.testing.assert_allclose(out, ref, rtol=0.0, atol=1e-8)
