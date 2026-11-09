###############################################################################
# test_backend_actionAngleSphericalInverse.py: actionAngleSphericalInverse
# on jax/torch -- values against the numpy path (a numpy-built family on
# backend inputs, and a family built on a backend), the derivatives in the
# actions and the angles against finite differences of the numpy path, the
# canonicity of the autodiff Jacobian (for perturbed tables and explicit
# tori too), the derivatives in the potential's parameters, the circular
# orbit, jit/vmap/compile, float32, and the construction's checks
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

# the symplectic form on (theta_r, theta_phi, theta_z, J_r, J_phi, J_z) ->
# (R, z, phi, v_R, v_z, R v_T)
_OMEGA = numpy.block(
    [[numpy.zeros((3, 3)), numpy.eye(3)], [-numpy.eye(3), numpy.zeros((3, 3))]]
)
_ANGLES = numpy.array([0.0, 0.4, 1.5, numpy.pi, 4.0, 6.0])


def _arr(backend, x, grad=False):
    if backend == "jax":
        return jnp.asarray(x)
    return torch.tensor(x, requires_grad=grad)


def _is_backend(backend, x):
    return type(x).__module__.split(".")[0] == (
        "jaxlib" if backend == "jax" else "torch"
    )


def _pot(amp=1.0):
    from galpy.potential import LogarithmicHaloPotential

    return LogarithmicHaloPotential(amp=amp)


def _interp(pot=None, **kw):
    from galpy.actionAngle import actionAngleSphericalInverse

    kw = {
        "setup_interp": True,
        "Rmin": 0.7,
        "Rmax": 1.4,
        "Rinf": 6.0,
        "nE": 8,
        "nL": 8,
        "mm_nta": 128,
        "mm_npt": 24,
        "progressbar": False,
        **kw,
    }
    return actionAngleSphericalInverse(pot=_pot() if pot is None else pot, **kw)


def _explicit(pot=None, **kw):
    from galpy.actionAngle import actionAngleSphericalInverse

    kw = {"Es": [0.7, 1.1], "Ls": [0.9, 0.7], "progressbar": False, **kw}
    return actionAngleSphericalInverse(pot=_pot() if pot is None else pot, **kw)


@pytest.fixture(scope="module")
def interp():
    return _interp()


_FORCED = {}


def _forced(backend, mode):
    """The test family built on a backend (forced), once per module"""
    from galpy import backend as gb

    if (backend, mode) not in _FORCED:
        with gb.use(backend, force=True):
            _FORCED[(backend, mode)] = _explicit() if mode == "explicit" else _interp()
    return _FORCED[(backend, mode)]


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


def _defect(A):
    return numpy.amax(numpy.fabs(A.T @ _OMEGA @ A - _OMEGA))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["explicit", "interp"])
def test_values_match_numpy(backend, mode, explicit, interp):
    # A numpy-built family evaluated on backend inputs gives the numpy path's
    # (x, v, Omega) and J_r(E, L) to round-off; a family BUILT on a backend
    # (forced) agrees to ~1e-10, because its turning points are polished roots
    # where numpy's brentq stops at xtol = 1e-12, which the frequencies and
    # the map's slopes amplify through the samples next to them (numpy with
    # xtol = 1e-15 agrees with the backend build to ~6e-11)
    from galpy import backend as gb

    aN = explicit if mode == "explicit" else interp
    aB = _forced(backend, mode)
    assert aB._bk is not None and aN._bk is None
    if mode == "explicit":
        tori = [(aN._jrs[0], 0.5, 0.4), (aN._jrs[1], -0.4, 0.3)]
        labels = [(0.7, 0.9), (1.1, 0.7)]
    else:
        tori = [(0.05, 0.6, 0.4), (0.3, -0.5, 0.6), (0.0, 0.7, 0.3)]
        labels = [(0.8, 1.0), (0.5, 1.0), (1.2, 0.8)]
    for jr, jphi, jz in tori:
        a = (_ANGLES, 0.2 + _ANGLES, 0.7 - _ANGLES)
        ref = _flat(aN.xvFreqs(jr, jphi, jz, *a))
        ins = [_arr(backend, v) for v in (jr, jphi, jz, *a)]
        on_bk = aN.xvFreqs(*ins)
        assert all(_is_backend(backend, o) for o in on_bk)
        numpy.testing.assert_allclose(_flat(on_bk), ref, rtol=0.0, atol=1e-12)
        numpy.testing.assert_allclose(
            _flat(aB.xvFreqs(*ins)),
            ref,
            rtol=0.0,
            atol=3e-10 if mode == "explicit" else 2e-11,
        )
        # numpy inputs to a backend-built family evaluate on its backend
        assert _is_backend(backend, aB(jr, jphi, jz, 0.3, 1.0, 2.0)[0])
        reff = _flat(aN.Freqs(jr, jphi, jz))
        numpy.testing.assert_allclose(
            _flat(aN.Freqs(*ins[:3])), reff, rtol=0.0, atol=1e-12
        )
        numpy.testing.assert_allclose(
            _flat(aB.Freqs(*ins[:3])), reff, rtol=0.0, atol=1e-10
        )
    for E, L in labels:
        Ea, La = _arr(backend, E), _arr(backend, L)
        ref = aN.Jr(E, L)
        assert numpy.fabs(float(as_numpy(aN.Jr(Ea, La))) - ref) < 1e-12
        assert numpy.fabs(float(as_numpy(aB.Jr(Ea, La))) - ref) < 1e-10


def _fd(f, x0, h):
    return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)


@pytest.mark.parametrize("backend", BACKENDS)
def test_derivatives_vs_finite_difference(backend, interp):
    # d(q, p)/d(theta, J) by autodiff against a converged 5-point finite
    # difference of the numpy path, between the grid's nodes, for a prograde
    # and a retrograde torus; and the frequencies' derivatives in the actions
    for x0 in (
        numpy.array([0.4, 1.0, 2.0, 0.137, 0.55, 0.3]),
        numpy.array([5.0, 3.0, 0.7, 0.21, -0.45, 0.4]),
    ):

        def qp_np(x):
            return numpy.array(_qp(interp(x[3], x[4], x[5], *x[:3])))

        A = numpy.empty((6, 6))
        for k in range(6):
            e = numpy.zeros(6)
            e[k] = 1.0
            A[:, k] = _fd(lambda h: qp_np(x0 + h * e), 0.0, 1e-4)
        J = _jacobian(backend, interp, x0)
        numpy.testing.assert_allclose(J, A, rtol=0.0, atol=1e-10)

        def om_np(j):
            return numpy.array(interp.Freqs(*j))

        if backend == "jax":
            dO = numpy.asarray(
                jax.jacfwd(lambda j: jnp.stack(interp.Freqs(j[0], j[1], j[2])))(
                    jnp.asarray(x0[3:])
                )
            )
        else:
            dO = as_numpy(
                torch.autograd.functional.jacobian(
                    lambda j: torch.stack(interp.Freqs(j[0], j[1], j[2])),
                    torch.tensor(x0[3:]),
                )
            )
        for k in range(3):
            e = numpy.zeros(3)
            e[k] = 1.0
            numpy.testing.assert_allclose(
                dO[:, k],
                _fd(lambda h: om_np(x0[3:] + h * e), 0.0, 1e-4),
                rtol=0.0,
                atol=1e-9,
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_canonical(backend, interp, explicit):
    # Manifest canonicity, with exact derivatives: the autodiff Jacobian of
    # the map is symplectic to round-off between the family's nodes, for
    # tables perturbed so much that the map is no longer accurate (every
    # derivative the map uses is its stored interpolant's own), and for an
    # explicit torus (a one-node family, linear in (J_r, L) with its slopes);
    # as the analytic isochrone inverse is
    from galpy.actionAngle import actionAngleIsochroneInverse

    x0 = numpy.array([0.4, 1.0, 2.0, 0.137, 0.55, 0.3])
    floor = _defect(
        _jacobian(backend, actionAngleIsochroneInverse(ip=interp.auxiliary), x0)
    )
    assert floor < 1e-12
    assert _defect(_jacobian(backend, interp, x0)) < 1e-12
    aP = copy.deepcopy(interp)
    rng = numpy.random.default_rng(3)
    nu, nL = aP._jr_tab.shape
    aP._jr_tab[1:] *= 1.0 + 1e-3 * rng.uniform(-1.0, 1.0, (nu - 1, nL))
    aP._jr_dx *= 1.0 + 1e-2 * rng.uniform(-1.0, 1.0, (nu, nL))
    aP._sup_tab[1:, :, 0] -= 1e-3 * rng.uniform(0.0, 1.0, (nu - 1, nL))
    aP._sup_tab[1:, :, 1] += 1e-3 * rng.uniform(0.0, 1.0, (nu - 1, nL))
    aP._Dm_tab += 1e-3 * rng.uniform(-1.0, 1.0, aP._Dm_tab.shape)
    aP._Dm_du *= 1.0 + 0.1 * rng.uniform(-1.0, 1.0, aP._Dm_du.shape)
    aP._rebuild_interp()
    assert (
        numpy.amax(
            numpy.fabs(_flat(aP(*x0[3:], *x0[:3])) - _flat(interp(*x0[3:], *x0[:3])))
        )
        > 1e-5
    )
    assert _defect(_jacobian(backend, aP, x0)) < 1e-12
    xe = numpy.array([0.4, 1.0, 2.0, explicit._jrs[0], 0.5, 0.4])
    assert _defect(_jacobian(backend, explicit, xe)) < 1e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_action_derivatives(backend):
    # An explicit torus is a quadratic one-node family: from its second-order
    # tables (computed from the torus alone), the derivatives of the map and
    # of its frequencies in the actions at the node are the true family's,
    # against finite differences across explicit tori built at E +- h and
    # L +- h (with the auxiliary held fixed; the physical point does not
    # depend on it once the maps are resolved), and so are those of J_r(E, L)
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import IsochronePotential

    pot = _pot()
    E0, L0, h = 0.7, 0.9, 1e-4
    base = _explicit(Es=[E0], Ls=[L0])
    aux = IsochronePotential(amp=base._GM, b=base._b)

    def tor(E, L):
        return actionAngleSphericalInverse(
            pot=pot, Es=[E], Ls=[L], auxiliary=aux, progressbar=False
        )

    jphi = 0.5
    x0 = numpy.array([0.4, 1.0, 2.0, base._jrs[0], jphi, L0 - jphi])
    J = _jacobian(backend, base, x0)

    def qpO(aA, jz):
        out = aA.xvFreqs(aA._jrs[0], jphi, jz, 0.4, 1.0, 2.0)
        return numpy.array(_qp(out) + [float(o) for o in out[6:]])

    up, dn = tor(E0 + h, L0), tor(E0 - h, L0)
    fdE = (qpO(up, L0 - jphi) - qpO(dn, L0 - jphi)) / (2.0 * h)
    upL, dnL = tor(E0, L0 + h), tor(E0, L0 - h)
    fdL = (qpO(upL, L0 - jphi + h) - qpO(dnL, L0 - jphi - h)) / (2.0 * h)
    OmR, Ompsi = base._OmRs[0], base._Ompsis[0]
    # d/dE|L = d/dJ_r / Omega_r; d/dL|E = d/dJ_z|J_r - d/dJ_r Omega_psi / Omega_r
    numpy.testing.assert_allclose(J[:, 3] / OmR, fdE[:6], rtol=0.0, atol=1e-7)
    numpy.testing.assert_allclose(
        J[:, 5] - J[:, 3] * Ompsi / OmR, fdL[:6], rtol=0.0, atol=1e-7
    )
    # the frequencies' derivatives in the actions
    if backend == "jax":
        dO = numpy.asarray(
            jax.jacfwd(lambda j: jnp.stack(base.Freqs(j[0], j[1], j[2])))(
                jnp.asarray(x0[3:])
            )
        )
    else:
        dO = as_numpy(
            torch.autograd.functional.jacobian(
                lambda j: torch.stack(base.Freqs(j[0], j[1], j[2])),
                torch.tensor(x0[3:]),
            )
        )
    numpy.testing.assert_allclose(dO[:, 0] / OmR, fdE[6:], rtol=0.0, atol=1e-7)
    numpy.testing.assert_allclose(
        dO[:, 2] - dO[:, 0] * Ompsi / OmR, fdL[6:], rtol=0.0, atol=1e-7
    )
    # J_r(E, L): its gradient and Hessian against the neighbouring tori's
    # actions and frequencies
    if backend == "jax":
        f = lambda v: base.Jr(v[0], v[1])  # noqa: E731
        g = numpy.asarray(jax.grad(f)(jnp.asarray([E0, L0])))
        H = numpy.asarray(jax.hessian(f)(jnp.asarray([E0, L0])))
    else:
        v = torch.tensor([E0, L0], requires_grad=True)
        g = as_numpy(torch.autograd.grad(base.Jr(v[0], v[1]), v)[0])
        H = as_numpy(
            torch.autograd.functional.hessian(lambda v: base.Jr(v[0], v[1]), v.detach())
        )
    numpy.testing.assert_allclose(
        g,
        [(up._jrs[0] - dn._jrs[0]) / (2 * h), (upL._jrs[0] - dnL._jrs[0]) / (2 * h)],
        rtol=0.0,
        atol=1e-7,
    )
    dJE = lambda a: 1.0 / a._OmRs[0]  # noqa: E731
    dJL = lambda a: -a._Ompsis[0] / a._OmRs[0]  # noqa: E731
    numpy.testing.assert_allclose(
        H,
        [
            [(dJE(up) - dJE(dn)) / (2 * h), (dJE(upL) - dJE(dnL)) / (2 * h)],
            [(dJL(up) - dJL(dn)) / (2 * h), (dJL(upL) - dJL(dnL)) / (2 * h)],
        ],
        rtol=0.0,
        atol=1e-7,
    )


# ---------- derivatives in the potential's parameters
_KW_PARAM = {
    "setup_interp": True,
    "Rmin": 0.7,
    "Rmax": 1.4,
    "Rinf": 6.0,
    "nE": 4,
    "nL": 4,
    "mm_nta": 128,
    "mm_npt": 24,
    "progressbar": False,
}
_ANG4 = numpy.array([0.0, 0.4, 1.5, 4.0])


def _param_family(p, mode, fitted=False):
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.potential import HernquistPotential, IsochronePotential

    # a given auxiliary, so that the finite differences below are of the
    # same map (the fitted one is frozen on a backend: no derivative flows
    # through the fit, and refitting moves an interpolated family at its
    # interpolation error)
    aux = None if fitted else IsochronePotential(amp=2.0, b=0.6)
    if mode == "interp":
        return actionAngleSphericalInverse(pot=_pot(amp=p), auxiliary=aux, **_KW_PARAM)
    return actionAngleSphericalInverse(
        pot=HernquistPotential(amp=2.0, a=p),
        Es=[-0.35, -0.3],
        Ls=[0.5, 0.6],
        auxiliary=aux,
        progressbar=False,
    )


def _param_out(aA, mode):
    # interpolated: the torus of fixed actions; explicit: the first torus at
    # its own (E, L), its action moving with the parameter
    xp = aA._bk["xp"]
    jr = 0.1 if mode == "interp" else aA._bk["jrs"][0]
    a = [xp.asarray(v) for v in (_ANG4, _ANG4 + 0.2, 0.7 - _ANG4)]
    jphi = 0.55 if mode == "interp" else 0.3
    jz = 0.3 if mode == "interp" else 0.2
    return xp.concat([xp.reshape(o, (-1,)) for o in aA.xvFreqs(jr, jphi, jz, *a)])


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_derivatives_wrt_potential_parameters(mode):
    # The construction is differentiable in the potential's parameters: the
    # (x, v, Omega) of the torus against a 5-point finite difference of the
    # backend construction itself (its roots are polished; numpy's stop at
    # brentq's xtol), for the logarithmic halo's amplitude on an interpolated
    # family and the Hernquist scale on explicit tori (at the torus's own
    # (E, L), which is the derivative of the one-node family there with no
    # action term). jax's derivative under jit is checked against this one
    # in test_jax_jit_construction.
    p0, h = (1.0, 1e-3) if mode == "interp" else (0.8, 1e-3)
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

    fd = _fd(num, p0, h)
    numpy.testing.assert_allclose(g, fd, rtol=0.0, atol=1e-9)


@pytest.mark.skipif(jax is None or torch is None, reason="jax and torch needed")
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_jax_jit_construction(mode):
    # The whole pipeline -- the construction (its auxiliary fitted on the
    # device) and the evaluation -- traces under jax.jit, and so
    # does its derivative in a potential parameter; both agree with torch's
    # eager construction and autodiff
    p0 = 1.0 if mode == "interp" else 0.8

    def f(p):
        return _param_out(_param_family(p, mode, fitted=True), mode)

    pt = torch.tensor(p0, requires_grad=True)
    y = _param_out(_param_family(pt, mode, fitted=True), mode)
    g = numpy.array(
        [
            float(torch.autograd.grad(y[k], pt, retain_graph=True)[0])
            for k in range(len(y))
        ]
    )
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(f)(jnp.asarray(p0))),
        as_numpy(y.detach()),
        rtol=0.0,
        atol=1e-12,
    )
    if mode == "interp":
        numpy.testing.assert_allclose(
            numpy.asarray(jax.jit(jax.jacfwd(f))(jnp.asarray(p0))),
            g,
            rtol=0.0,
            atol=1e-10,
        )


# ---------- jit, vmap, compile, float32
@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_and_vmap(interp, explicit):
    # the evaluation traces under jax.jit (the circular and librating branches
    # both traced and selected), and vmaps over tori
    for aA, (jr, jphi, jz) in (
        (interp, (0.137, 0.55, 0.3)),
        (interp, (0.0, 0.55, 0.3)),
        (explicit, (explicit._jrs[1], -0.4, 0.3)),
    ):
        ref = _flat(aA.xvFreqs(jr, jphi, jz, _ANGLES, 1.0, 2.0))

        def f(j, a):
            return jnp.concatenate(
                [jnp.ravel(o) for o in aA.xvFreqs(j, jphi, jz, a, 1.0, 2.0)]
            )

        got = jax.jit(f)(jnp.asarray(jr), jnp.asarray(_ANGLES))
        numpy.testing.assert_allclose(numpy.asarray(got), ref, rtol=0.0, atol=1e-12)
        freqs_jit = jax.jit(lambda j: jnp.stack(aA.Freqs(j, jphi, jz)))
        numpy.testing.assert_allclose(
            numpy.asarray(freqs_jit(jnp.asarray(jr))),
            _flat(aA.Freqs(jr, jphi, jz)),
            rtol=0.0,
            atol=1e-12,
        )
    # J_r(E, L) under jit, including the circular orbit
    for aA, (E, L) in (
        (interp, (0.8, 1.0)),
        (interp, (0.5, 1.0)),
        (explicit, (0.7, 0.9)),
    ):
        got = jax.jit(lambda e, l: aA.Jr(e, l))(jnp.asarray(E), jnp.asarray(L))
        assert numpy.fabs(float(got) - aA.Jr(E, L)) < 1e-12
    # vmap over the radial action
    jrs = numpy.array([0.0, 0.05, 0.137, 0.3])
    out = jax.vmap(
        lambda j: interp.xvFreqs(j, 0.55, 0.3, jnp.asarray(_ANGLES), 1.0, 2.0)[0]
    )(jnp.asarray(jrs))
    for k, j in enumerate(jrs):
        numpy.testing.assert_allclose(
            numpy.asarray(out[k]),
            interp.xvFreqs(j, 0.55, 0.3, _ANGLES, 1.0, 2.0)[0],
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_label_and_circular_radius_solves(interp):
    # Under jax.jit the label u of J_r and the circular radius of L come from
    # a few Newton steps started at guesses interpolated between the grid's
    # nodes, where the eager solves section the bracket (a traced bisection
    # would take ~50 sequential halvings); at and next to the family's edges
    # both give the same frequencies to round-off, single and vmapped
    jphi, jz = 0.55, 0.3
    L = jphi + jz
    jlo, jhi = (float(interp._jr_ip(x, L)[0, 0]) for x in (0.0, 1.0))
    jrs = numpy.array(
        [jlo + 1e-14, jlo + 1e-9, jlo + 1e-4, 0.5 * (jlo + jhi), jhi - 1e-9, jhi]
    )
    eager = numpy.array([_flat(interp.Freqs(j, jphi, jz)) for j in jrs])
    f = jax.jit(lambda j: jnp.stack(interp.Freqs(j, jphi, jz)))
    got = numpy.array([numpy.asarray(f(jnp.asarray(j))) for j in jrs])
    numpy.testing.assert_allclose(got, eager, rtol=1e-13, atol=0.0)
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(jax.vmap(f))(jnp.asarray(jrs))),
        eager,
        rtol=1e-13,
        atol=0.0,
    )
    # the circular orbit at a traced L, to the grid's L edges
    Lg = numpy.linspace(0.7, 1.4, 8)
    for Lc in (Lg[0] + 1e-12, 0.97, Lg[-1] - 1e-12):
        jz_c = Lc - jphi
        ref = _flat(interp.xvFreqs(0.0, jphi, jz_c, _ANGLES, 1.0, 2.0))
        got = jax.jit(
            lambda jp: jnp.concatenate(
                [
                    jnp.ravel(o)
                    for o in interp.xvFreqs(0.0, jp, Lc - jp, _ANGLES, 1.0, 2.0)
                ]
            )
        )(jnp.asarray(jphi))
        numpy.testing.assert_allclose(numpy.asarray(got), ref, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_array_actions(backend, mode, interp, explicit):
    # Arrays of actions on a backend, broadcast against the angles, each
    # point its own torus (circular ones among them): a numpy-built family on
    # backend inputs gives the numpy array evaluation to round-off and a
    # family built on a backend (forced) to its construction's ~1e-10 (as a
    # single torus); the outputs take the broadcast shape ((n, 1) actions
    # against (m,) angles too) and the frequencies are arrays; each point's
    # derivatives in its actions and its angle are the single torus's
    from galpy import backend as gb

    aN = interp if mode == "interp" else explicit
    if mode == "interp":
        jr = numpy.array([0.0, 0.02, 0.137, 0.3, 0.02, 0.137])
        jphi = numpy.array([0.55, -0.8, 0.9, 0.55, 0.55, -0.4])
        jz = numpy.array([0.3, 0.05, 0.0, 0.3, 0.3, 0.5])
    else:
        idx = numpy.array([0, 1, 0, 1, 0, 1])
        jr = aN._jrs[idx]
        jphi = numpy.array([0.6, -0.4, 0.3, 0.2, -0.6, 0.5])
        jz = aN._Ls[idx] - numpy.fabs(jphi)
    ang = _ANGLES
    ref = aN.xvFreqs(jr, jphi, jz, ang, ang + 0.3, ang + 0.7)
    ref2 = aN(jr[:3, None], jphi[:3, None], jz[:3, None], ang[None, :], 1.0, 2.0)
    aB = _forced(backend, mode)
    for fam, tol in ((aN, 2e-12), (aB, 2e-10)):
        got = fam.xvFreqs(
            *(_arr(backend, t) for t in (jr, jphi, jz, ang, ang + 0.3, ang + 0.7))
        )
        got2 = fam(
            *(_arr(backend, t)[:3, None] for t in (jr, jphi, jz)),
            _arr(backend, ang)[None, :],
            1.0,
            2.0,
        )
        for g, r in zip(got + got2, ref + ref2):
            assert _is_backend(backend, g) and tuple(g.shape) == r.shape
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=tol)
        Om = fam.Freqs(*(_arr(backend, t) for t in (jr, jphi, jz)))
        for g, r in zip(Om, aN.Freqs(jr, jphi, jz)):
            assert _is_backend(backend, g) and tuple(g.shape) == jr.shape
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=tol)
        if mode == "interp":  # circular orbits only
            j0 = 0.0 * jr
            got = fam.xvFreqs(
                *(_arr(backend, t) for t in (j0, jphi, jz, ang, ang, ang))
            )
            got += fam.Freqs(*(_arr(backend, t) for t in (j0, jphi, jz)))
            ref0 = aN.xvFreqs(j0, jphi, jz, ang, ang, ang) + aN.Freqs(j0, jphi, jz)
            for g, r in zip(got, ref0):
                numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=tol)
    # per-point derivatives against the single torus's, at the librating,
    # inclined points (at J_z = 0 the inclination's derivative is singular)
    lib = (jr > 0.0) & (jz > 0.0)
    J = numpy.stack([jr[lib], jphi[lib], jz[lib]])
    A = numpy.stack([ang[lib], ang[lib] + 0.3, ang[lib] + 0.7])
    n = J.shape[1]
    if backend == "jax":

        def f(J, A):
            return jnp.stack(aN.xvFreqs(*J, *A)[:6])

        gJ, gA = (
            numpy.asarray(d)
            for d in jax.jit(jax.jacfwd(f, argnums=(0, 1)))(
                jnp.asarray(J), jnp.asarray(A)
            )
        )  # (6, n, 3, n): diagonal in the points
        off = ~numpy.eye(n, dtype=bool)
        for g in (gJ, gA):
            assert numpy.all(numpy.moveaxis(g, 2, 1)[..., off] == 0.0)
        one = jax.jit(jax.jacfwd(lambda J, A: f(J, A)[:, 0], argnums=(0, 1)))
        for k in range(n):
            sJ, sA = one(jnp.asarray(J[:, k]), jnp.asarray(A[:, k : k + 1]))
            numpy.testing.assert_allclose(
                gJ[:, k, :, k], numpy.asarray(sJ), rtol=1e-10, atol=1e-12
            )
            numpy.testing.assert_allclose(
                gA[:, k, :, k], numpy.asarray(sA)[..., 0], rtol=1e-10, atol=1e-12
            )
    else:
        Jt, At = _arr(backend, J, grad=True), _arr(backend, A, grad=True)
        out = aN.xvFreqs(*Jt, *At)[:6]
        for q in range(6):
            gJ, gA = torch.autograd.grad(out[q].sum(), (Jt, At), retain_graph=True)
            for k in range(n):
                Jk = _arr(backend, J[:, k], grad=True)
                Ak = _arr(backend, A[:, k : k + 1], grad=True)
                sJ, sA = torch.autograd.grad(aN.xvFreqs(*Jk, *Ak)[q][0], (Jk, Ak))
                numpy.testing.assert_allclose(
                    as_numpy(gJ[:, k]), as_numpy(sJ), rtol=1e-10, atol=1e-12
                )
                numpy.testing.assert_allclose(
                    as_numpy(gA[:, k]), as_numpy(sA)[:, 0], rtol=1e-10, atol=1e-12
                )


def _grad2(backend, f, a, b):
    """The derivatives of the scalar f(a, b) in a and b"""
    if backend == "jax":
        return tuple(
            float(g)
            for g in jax.grad(f, argnums=(0, 1))(jnp.asarray(a), jnp.asarray(b))
        )
    at, bt = _arr(backend, a, grad=True), _arr(backend, b, grad=True)
    return tuple(float(g) for g in torch.autograd.grad(f(at, bt), (at, bt)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["interp", "explicit"])
def test_public_energy(backend, mode, interp, explicit):
    # E(jr, L) on a backend: a numpy-built family on backend inputs gives the
    # numpy E to round-off, and a family built on a backend (forced) to its
    # construction's ~1e-10, in the broadcast shape of (jr, L) (a 0-d array
    # for scalars); its derivatives in (J_r, L) are the torus's frequencies
    # (Omega_r, Omega_psi); jax.jit traces it; an action outside the family
    # (interpolated) or not among the set-up tori (explicit) raises
    aN = interp if mode == "interp" else explicit
    if mode == "interp":
        jr, L = numpy.array([[0.0], [0.02], [0.137]]), numpy.array([0.75, 0.8, 0.9])
        j1, L1 = 0.05, 0.8
    else:
        jr, L = aN._jrs[:, None], aN._Ls[None, :] + 0.0 * aN._jrs[:, None]
        jr, L = numpy.diag(jr + 0.0 * L), numpy.diag(L)
        j1, L1 = aN._jrs[0], aN._Ls[0]
    ref = aN.E(jr, L)
    for fam, tol in ((aN, 1e-13), (_forced(backend, mode), 2e-10)):
        got = fam.E(_arr(backend, jr), _arr(backend, L))
        assert _is_backend(backend, got) and tuple(got.shape) == ref.shape
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0.0, atol=tol)
        got1 = fam.E(_arr(backend, j1), _arr(backend, L1))
        assert _is_backend(backend, got1) and got1.ndim == 0
        assert abs(float(as_numpy(got1)) - aN.E(j1, L1)) < tol
    OmR, OmL, _ = aN.Freqs(j1, L1, 0.0)
    dJ, dL = _grad2(backend, aN.E, j1, L1)
    assert abs(dJ - OmR) < 1e-12 * OmR and abs(dL - OmL) < 1e-12 * abs(OmL)
    if backend == "jax":
        f = jax.jit(aN.E)
        assert float(f(jnp.asarray(j1), jnp.asarray(L1))) == float(
            aN.E(jnp.asarray(j1), jnp.asarray(L1))
        )
    if mode == "interp":
        with pytest.raises(ValueError, match="outside the interpolated family"):
            aN.E(_arr(backend, 10.0), _arr(backend, L1))
    else:
        with pytest.raises(ValueError, match="is not one of the set-up tori"):
            aN.E(_arr(backend, 1.1 * j1), _arr(backend, L1))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spacing", ["sqrt", "log"])
def test_Lspacing(backend, spacing):
    # The spacing of the angular-momentum grid on a backend: a family built
    # there (forced) has numpy's nodes, to round-off, and numpy's tori, to its
    # construction's ~1e-10; an unknown spacing raises
    from galpy import backend as gb

    aN = _interp(Lspacing=spacing)
    with gb.use(backend, force=True):
        aB = _interp(Lspacing=spacing)
    numpy.testing.assert_allclose(
        as_numpy(aB._Lgrid_bk), aN._Lgrid, rtol=1e-14, atol=0.0
    )
    for J in ((0.05, 0.6, 0.3), (0.137, -0.9, 0.1)):
        ref = aN.xvFreqs(*J, _ANGLES, 1.0, 2.0)
        got = aB.xvFreqs(
            *(_arr(backend, v) for v in J), _arr(backend, _ANGLES), 1.0, 2.0
        )
        for g, r in zip(got, ref):
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=2e-10)
    with gb.use(backend, force=True):
        with pytest.raises(ValueError, match="spacing 'cubic' not understood"):
            _interp(Lspacing="cubic")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_compile(explicit):
    # the evaluation runs under torch.compile (an explicit torus: the
    # interpolated family's root-finds -- r_c, r_0, the label -- each unroll
    # their bisections into dynamo's trace, minutes of compile)
    jr = explicit._jrs[1]
    ref = _flat(explicit.xvFreqs(jr, -0.4, 0.3, _ANGLES, 1.0, 2.0))
    f = torch.compile(
        lambda j, a: torch.cat(
            [
                torch.atleast_1d(o).ravel()
                for o in explicit.xvFreqs(j, -0.4, 0.3, a, 1.0, 2.0)
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
    ref = interp.xvFreqs(0.137, 0.55, 0.3, _ANGLES, 1.0, 2.0)
    got = interp.xvFreqs(
        f32(0.137), f32(0.55), f32(0.3), f32(_ANGLES), f32(1.0), f32(2.0)
    )
    for g, r in zip(got, ref):
        assert str(g.dtype).endswith("float32")
        numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=2e-6)
    for out in (
        *interp.Freqs(f32(0.137), f32(0.55), f32(0.3)),
        interp.Jr(f32(0.8), f32(1.0)),
    ):
        assert str(out.dtype).endswith("float32")


# ---------- the circular orbit
@pytest.mark.parametrize("backend", BACKENDS)
def test_circular_orbit(backend, interp):
    # J_r = 0 is the circular orbit, in both kinds of family and in a family
    # built on the backend with only circular tori (its auxiliary the closed
    # form one), with finite derivatives in the other actions and the angles
    from galpy import backend as gb
    from galpy.potential import evaluatePotentials, rl

    pot = _pot()
    L = 0.9
    rc = rl(pot, L)
    Ec = evaluatePotentials(pot, rc, 0.0) + L**2 / (2.0 * rc**2)
    aN = _explicit(Es=[Ec, 0.9], Ls=[L, L])
    with gb.use(backend, force=True):
        aB = _explicit(Es=[Ec], Ls=[L])
    assert numpy.fabs(aB._GM - aN._GM) > 1e-3  # the closed-form auxiliary
    for aA, ref in ((interp, interp), (aN, aN), (aB, aN)):
        x = _flat(
            aA.xvFreqs(
                *(_arr(backend, v) for v in (0.0, 0.63, L - 0.63, _ANGLES, 1.0, 2.0))
            )
        )
        numpy.testing.assert_allclose(
            x,
            _flat(ref.xvFreqs(0.0, 0.63, L - 0.63, _ANGLES, 1.0, 2.0)),
            rtol=0.0,
            atol=1e-12,
        )
        numpy.testing.assert_allclose(
            _flat(aA.Freqs(*(_arr(backend, v) for v in (0.0, 0.63, L - 0.63)))),
            _flat(ref.Freqs(0.0, 0.63, L - 0.63)),
            rtol=0.0,
            atol=1e-12,
        )
    assert float(as_numpy(aB.Jr(_arr(backend, Ec), _arr(backend, L)))) == 0.0
    J = _jacobian(backend, interp, numpy.array([0.4, 1.0, 2.0, 0.0, 0.55, 0.3]))
    assert numpy.all(numpy.isfinite(J[:, [0, 1, 2, 4, 5]]))
    # J_r(E, L) of the interpolated family is 0 at the circular orbit, with
    # the one-sided derivative dJ_r/dE = 1/kappa there (L = 1 is a grid row)
    from galpy.potential import epifreq

    L1 = 1.0
    rc1 = rl(pot, L1)
    Ec1 = evaluatePotentials(pot, rc1, 0.0) + L1**2 / (2.0 * rc1**2)
    if backend == "jax":
        Jr, dJr = jax.value_and_grad(lambda E: interp.Jr(E, L1))(jnp.asarray(Ec1))
    else:
        Et = torch.tensor(Ec1, requires_grad=True)
        Jr = interp.Jr(Et, L1)
        (dJr,) = torch.autograd.grad(Jr, Et)
    assert float(as_numpy(Jr)) == 0.0
    numpy.testing.assert_allclose(
        float(as_numpy(dJr)), 1.0 / epifreq(pot, rc1), rtol=1e-8, atol=0.0
    )


# ---------- the construction's contract and checks
@pytest.mark.parametrize("backend", BACKENDS)
def test_construction_contract(backend):
    # Under a forced backend, or for a potential with backend parameters, the
    # family is built natively and evaluates there, numpy inputs included
    from galpy import backend as gb

    gpot = _pot(amp=_arr(backend, 1.0))
    fams = [_explicit(pot=gpot)]
    if backend == "torch":  # jax's interpolated construction: test_jax_jit_construction
        fams.append(_interp(pot=gpot, nE=4, nL=4))
    for aA in fams:
        assert aA._bk is not None
        jr = float(as_numpy(aA.Jr(0.7, 0.9) if not aA._interp else 0.1))
        assert _is_backend(backend, aA(jr, 0.5, 0.4, 0.3, 1.0, 2.0)[0])
        assert _is_backend(backend, aA.Freqs(jr, 0.5, 0.4)[0])
    with gb.use(backend, force=True):
        aA = _explicit()
    assert aA._bk is not None and _is_backend(backend, aA._bk["jrs"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_construction_checks(backend):
    # the construction's checks and warnings on a backend, as on numpy
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleSphericalInverse
    from galpy.util import galpyWarning

    pot = _pot()
    with gb.use(backend, force=True):
        with pytest.raises(ValueError, match="below the circular orbit's"):
            _explicit(Es=[0.0], Ls=[0.9])
        with pytest.raises(ValueError, match="nE >= 4"):
            _interp(nE=3)
        with pytest.raises(ValueError, match="Rinf is too small"):
            _interp(Rinf=1.0)
        with pytest.raises(TypeError, match="IsochronePotential"):
            _explicit(auxiliary=pot)
        with pytest.warns(galpyWarning, match="not converged for the"):
            actionAngleSphericalInverse(
                pot=pot, Es=[1.53], Ls=[0.9], mm_npt=2, mm_nta=16, progressbar=False
            )
        with pytest.raises(RuntimeError, match="not bound in the fitted auxiliary"):
            actionAngleSphericalInverse(pot=pot, Es=[5.0], Ls=[0.8], progressbar=False)
    from galpy.potential import HernquistPotential

    with gb.use(backend, force=True):
        with pytest.raises(ValueError, match="unbound"):
            actionAngleSphericalInverse(
                pot=HernquistPotential(amp=2.0, a=0.8),
                Es=[0.1],
                Ls=[0.5],
                progressbar=False,
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_evaluation_checks(backend, interp, explicit):
    # the evaluation's checks on backend inputs, as on numpy
    from galpy import backend as gb

    if backend == "torch":  # the same code on jax: its construction is slow
        with gb.use(backend, force=True):
            aI, aE = _interp(nE=4, nL=4), _explicit()
    else:
        aI, aE = interp, explicit
    for aA in (interp, aI):
        with pytest.raises(ValueError, match="outside the interpolation grid"):
            aA.Freqs(_arr(backend, 0.15), 5.0, 0.0)
        with pytest.raises(ValueError, match="outside the interpolated family"):
            aA(_arr(backend, 50.0), 0.65, 0.35, 0.3, 1.0, 2.0)
        with pytest.raises(ValueError, match="outside the interpolation grid"):
            aA(_arr(backend, 0.0), 0.3, 0.2, 0.3, 1.0, 2.0)
        with pytest.raises(ValueError, match="outside the interpolation grid at L"):
            aA.Jr(_arr(backend, 0.1), 1.0)
    for aA in (explicit, aE):
        for jr in (0.123, 0.0):
            with pytest.raises(ValueError, match="not one of the set-up tori"):
                aA(_arr(backend, jr), 0.4, 0.2, 0.3, 1.0, 2.0)
            with pytest.raises(ValueError, match="not one of the set-up tori"):
                aA.Freqs(_arr(backend, jr), 0.4, 0.2)
        with pytest.raises(ValueError, match="not one of the set-up tori"):
            aA.Jr(_arr(backend, 0.8), 0.9)


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="CUDA torch not available"
)
def test_cuda_potential_parameters_cpu_default_device():
    # potential parameters on the GPU with the CPU as the default device: the
    # family is built and evaluated on the GPU
    gpot = _pot(amp=torch.tensor(1.0, device="cuda"))
    aA = _explicit(pot=gpot)
    out = aA(float(aA._jrs[0]), 0.5, 0.4, numpy.array([0.3, 1.0]), 1.0, 2.0)
    assert out[0].device.type == "cuda"
    numpy.testing.assert_allclose(
        as_numpy(out[0]),
        _explicit()(aA._jrs[0], 0.5, 0.4, numpy.array([0.3, 1.0]), 1.0, 2.0)[0],
        rtol=0.0,
        atol=1e-10,
    )


# ---------- the building blocks
@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_isochrone_vc2(backend):
    # The auxiliary's least-squares fit on a backend, as a traced construction
    # runs it on the device: the numpy fit to round-off, eager and (jax)
    # under jit and vmap over rotation curves, for a logarithmic halo, an
    # isochrone (which it recovers) and a Kepler potential (whose fit runs to
    # the floor in b and stops there), and at a single radius
    from galpy.actionAngle.actionAngleInverseUtil import _fit_isochrone_vc2
    from galpy.potential import IsochronePotential, KeplerPotential, vcirc

    rf = numpy.geomspace(0.3, 3.0, 25)
    curves = numpy.array(
        [
            numpy.log(vcirc(p, rf) ** 2)
            for p in (
                _pot(),
                IsochronePotential(amp=2.3, b=0.6),
                KeplerPotential(normalize=1.0),
            )
        ]
    )
    ref = numpy.array([_fit_isochrone_vc2(rf, c) for c in curves])
    numpy.testing.assert_allclose(ref[1], [2.3, 0.6], rtol=1e-14, atol=0.0)
    numpy.testing.assert_allclose(ref[2], [1.0, 1e-8], rtol=1e-7, atol=0.0)
    for c, r in zip(curves, ref):
        got = _fit_isochrone_vc2(_arr(backend, rf), _arr(backend, c))
        assert all(_is_backend(backend, g) and g.ndim == 0 for g in got)
        numpy.testing.assert_allclose(
            [float(as_numpy(g)) for g in got], r, rtol=1e-14, atol=0.0
        )
    # a single radius (a lone circular torus's range): the start, exact there
    r1 = numpy.full(25, 1.3)
    c1 = numpy.full(25, numpy.log(vcirc(_pot(), 1.3) ** 2))
    for GM, b in (
        _fit_isochrone_vc2(r1, c1),
        (
            float(as_numpy(t))
            for t in _fit_isochrone_vc2(_arr(backend, r1), _arr(backend, c1))
        ),
    ):
        s1 = numpy.sqrt(b**2 + 1.3**2)
        assert abs(b / 1.3 - 1.0) < 1e-15
        assert abs(numpy.log(GM * 1.3**2 / (s1 * (b + s1) ** 2)) - c1[0]) < 1e-14
    if backend == "jax":
        f = jax.jit(
            jax.vmap(lambda c: jnp.stack(_fit_isochrone_vc2(jnp.asarray(rf), c)))
        )
        numpy.testing.assert_allclose(
            numpy.asarray(f(jnp.asarray(curves))), ref, rtol=1e-14, atol=0.0
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_hermite_family_2d(backend):
    # HermiteFamily2D built on a backend (differentiable in its tables and its
    # grid) agrees with the numpy one, called like a RectBivariateSpline and
    # through its vectorized ev, for one table and for a stack
    from galpy.util._hermite import HermiteFamily2D

    x, y = numpy.linspace(0.0, 1.0, 7) ** 2, numpy.linspace(0.5, 2.0, 6)
    X, Y = numpy.meshgrid(x, y, indexing="ij")
    scale = numpy.array([1.0, 2.0])
    f = (numpy.sin(3.0 * X) * numpy.cos(Y))[..., None] * scale
    fx = (3.0 * numpy.cos(3.0 * X) * numpy.cos(Y))[..., None] * scale
    fy = (-numpy.sin(3.0 * X) * numpy.sin(Y))[..., None] * scale
    for tabs in ((f, fx, fy), (f[..., 0], fx[..., 0], fy[..., 0])):
        ipn = HermiteFamily2D(x, y, *tabs)
        ipb = HermiteFamily2D(x, _arr(backend, y), *(_arr(backend, t) for t in tabs))
        for a, b in ((0.3, 0.7), (x[2], y[3]), (0.999, 1.99)):
            for dx, dy in ((0, 0), (1, 0), (0, 1)):
                ref = ipn.points(a, b, dx=dx, dy=dy)[0]
                for ip in (
                    ipb,
                    ipn,
                ):  # backend-built, and numpy-built on backend points
                    if dx == dy == 0:  # the value through __call__ too
                        got = ip(_arr(backend, a), _arr(backend, b))[0, 0]
                        assert _is_backend(backend, got)
                        numpy.testing.assert_allclose(
                            as_numpy(got), ref, rtol=0.0, atol=1e-12
                        )
                    got = ip.ev(_arr(backend, a), _arr(backend, b), dx=dx, dy=dy)
                    assert _is_backend(backend, got)
                    numpy.testing.assert_allclose(
                        as_numpy(got), ref, rtol=0.0, atol=1e-12
                    )
        got = ipb.ev(
            _arr(backend, numpy.array([0.3, 0.5])),
            _arr(backend, numpy.array([0.7, 1.5])),
        )
        numpy.testing.assert_allclose(
            as_numpy(got)[1], ipn(0.5, 1.5)[0, 0], rtol=0.0, atol=1e-12
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_unconverged_anomaly_solve(backend, explicit):
    # with maxiter too small for the anomaly's Newton iteration to converge,
    # the backend root-finder's safeguarded steps in the full bracket finish
    # the solve: the same torus as with the default maxiter
    aA = _explicit(maxiter=1)
    jr = explicit._jrs[0]
    ins = [
        _arr(backend, v)
        for v in (jr, 0.5, 0.4, _ANGLES, 1.0 + 0.0 * _ANGLES, 2.0 + 0.0 * _ANGLES)
    ]
    numpy.testing.assert_allclose(
        _flat(aA.xvFreqs(*ins)), _flat(explicit.xvFreqs(*ins)), rtol=0.0, atol=1e-10
    )
