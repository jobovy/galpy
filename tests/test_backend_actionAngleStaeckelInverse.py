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
import copy

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
@pytest.mark.parametrize("spacing", ["sqrt", "log"])
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
    rc = float(aN._delta * numpy.sinh(aN._samples[0]["uc"]))
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
    # u0='fit' (natively, frozen), a callable u0 (on the host) and a target's
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
                u0="fit",
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
            dict(target=o(ts), u0="fit", Rmin=0.75, Rmax=1.25, Rinf=3.0),
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
        with pytest.raises(RuntimeError, match="not bound in the fitted auxiliary"):
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


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_array_actions_adaptive():
    # a family whose reference curve is per torus evaluates traced arrays of
    # tori one torus at a time
    aA = _interp(u0=lambda E, Lz: 1.0 + 0.2 * numpy.tanh(Lz) + 0.1 * E)
    J, ang = _ARR_TORI[[0, 1]], _ARR_ANG[:, :2]
    f = jax.jit(lambda a, b, c: jnp.stack(aA.xvFreqs(a, b, c, *ang)))
    numpy.testing.assert_allclose(
        numpy.asarray(f(*(jnp.asarray(J[:, k]) for k in range(3)))),
        numpy.array(aA.xvFreqs(J[:, 0], J[:, 1], J[:, 2], *ang)),
        rtol=0.0,
        atol=_PLANE_ATOL,
    )


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_array_actions_fallbacks(interp):
    # points whose vectorized solve does not converge take the one-torus
    # path: an angle solve cut at one Newton step, and the label inversion of
    # tiny actions; a torus beyond the family and an |L_z| beyond the grid
    # raise as for one torus (torch: under eager jax the one-torus path's
    # safeguarded steps take minutes)
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
