###############################################################################
# test_backend_actionAngleVerticalInverse.py: the momentum-matched map of
# actionAngleVerticalInverse evaluated natively under jax/torch -- values
# against the numpy path, derivatives in the action and the angle against
# converged finite differences of the numpy path, canonicity, the J(E)/E(J)
# inverses, jit/compile, and the construction/legacy-mode contract.
###############################################################################
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


def _arr(backend, x, grad=False):
    if backend == "jax":
        return jnp.asarray(x)
    return torch.tensor(x, requires_grad=grad)


def _is_backend(backend, x):
    return type(x).__module__.split(".")[0] == (
        "jaxlib" if backend == "jax" else "torch"
    )


def _aAVI(**kw):
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import KGPotential

    kw.setdefault("Es", numpy.linspace(0.0, 0.6, 9))
    return actionAngleVerticalInverse(pot=KGPotential(K=1.0, F=0.5, D=0.5), **kw)


# angles through both turning points (pi/2, 3pi/2) and the midplane crossings
_ANGLES = numpy.array([0.0, 0.3, 0.5 * numpy.pi, 2.0, numpy.pi, 1.5 * numpy.pi, 5.9])


def _fd(f, x0, h):
    return (-f(x0 + 2 * h) + 8 * f(x0 + h) - 8 * f(x0 - h) + f(x0 - 2 * h)) / (12 * h)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "kw",
    [
        {"Es": [0.1, 0.3]},
        {"setup_interp": True},
        {"Es": [0.2]},
    ],
    ids=["two_tori", "interp", "one_torus"],
)
def test_values_match_numpy(backend, kw):
    # a numpy-built family evaluated on a backend gives the numpy path's
    # (x, v, Omega) to round-off, on and between the grid tori and at J = 0. A
    # family BUILT on a backend (forced) agrees to ~1e-11: its turning points are
    # polished roots (E - Phi(xmax) ~ 1e-17, against ~2e-14 for numpy's
    # calcxmax), which the frequency amplifies through the samples next to them
    from galpy import backend as gb

    aA = _aAVI(**kw)
    js = list(aA._js) + [0.137]
    for j in js:
        ref = aA.xvFreqs(j, _ANGLES)
        got = aA.xvFreqs(_arr(backend, j), _arr(backend, _ANGLES))
        with gb.use(backend, force=True):
            gotf = _aAVI(**kw).xvFreqs(j, _ANGLES)
        for g, gf, r in zip(got, gotf, ref):
            assert _is_backend(backend, g) and _is_backend(backend, gf)
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=2e-14)
            numpy.testing.assert_allclose(as_numpy(gf), r, rtol=0.0, atol=1e-11)
        Om = aA.Freqs(_arr(backend, j))
        assert _is_backend(backend, Om)
        assert abs(float(as_numpy(Om)) - aA.Freqs(j)) < 1e-14


def _jac_backend(backend, aA, j0, angles):
    # d(x, v)/dJ, d(x, v)/dtheta (per angle), and dOmega/dJ by autodiff
    if backend == "jax":

        def f(j, a):
            x, v, O = aA.xvFreqs(j, a)
            return jnp.concatenate([x, v, jnp.atleast_1d(O)])

        gJ = numpy.asarray(
            jax.jacfwd(f, argnums=0)(jnp.asarray(j0), jnp.asarray(angles))
        )
        gA = numpy.asarray(
            jax.jacfwd(f, argnums=1)(jnp.asarray(j0), jnp.asarray(angles))
        )
        n = len(angles)
        return (
            gJ[:n],
            gJ[n : 2 * n],
            numpy.diag(gA[:n]),
            numpy.diag(gA[n : 2 * n]),
            gJ[-1],
        )
    jt, at = _arr(backend, j0, grad=True), _arr(backend, angles, grad=True)
    x, v, O = aA.xvFreqs(jt, at)
    n = len(angles)
    dxdJ, dvdJ = numpy.empty(n), numpy.empty(n)
    for k in range(n):  # per-angle d/dJ (x[k] depends on J only through itself)
        (dxdJ[k],) = torch.autograd.grad(x[k], jt, retain_graph=True)
        (dvdJ[k],) = torch.autograd.grad(v[k], jt, retain_graph=True)
    # x[k] depends on angle k alone, so the gradient of the sum is the diagonal
    (dxda,) = torch.autograd.grad(x.sum(), at, retain_graph=True)
    (dvda,) = torch.autograd.grad(v.sum(), at, retain_graph=True)
    (dOdJ,) = torch.autograd.grad(O, jt)
    return dxdJ, dvdJ, dxda.numpy(), dvda.numpy(), float(dOdJ)


@pytest.mark.parametrize("backend", BACKENDS)
def test_derivatives_vs_finite_difference(backend):
    # d(x, v)/dJ, d(x, v)/dtheta and dOmega/dJ against a converged 5-point FD of
    # the numpy path, through both turning points (where the momentum's
    # sin(eta)/sin(tau) form is regularized) and between the grid tori
    aA = _aAVI(setup_interp=True)
    j0, h = 0.137, 3e-4
    n = len(_ANGLES)

    def npxv(j, a):
        x, v, O = aA.xvFreqs(j, numpy.atleast_1d(a))
        return numpy.concatenate([x, v, [O]])

    dJ = _fd(lambda j: npxv(j, _ANGLES), j0, h)
    dA = numpy.array([_fd(lambda a: npxv(j0, a)[:2], a, h) for a in _ANGLES])
    dxdJ, dvdJ, dxda, dvda, dOdJ = _jac_backend(backend, aA, j0, _ANGLES)
    numpy.testing.assert_allclose(dxdJ, dJ[:n], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dvdJ, dJ[n : 2 * n], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dOdJ, dJ[-1], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dxda, dA[:, 0], rtol=0.0, atol=1e-11)
    numpy.testing.assert_allclose(dvda, dA[:, 1], rtol=0.0, atol=1e-11)
    # the map is canonical: {x, v} = dx/dtheta dv/dJ - dx/dJ dv/dtheta = 1
    numpy.testing.assert_allclose(dxda * dvdJ - dxdJ * dvda, 1.0, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_action_torus(backend):
    # J = 0 is the point at the bottom, with finite derivatives; J < 0 raises
    aA = _aAVI()
    x, v, O = aA.xvFreqs(_arr(backend, 0.0), _arr(backend, _ANGLES))
    assert numpy.all(as_numpy(x) == 0.0) and numpy.all(as_numpy(v) == 0.0)
    assert abs(float(as_numpy(O)) - aA.Freqs(0.0)) < 1e-14
    dxdJ, dvdJ, dxda, dvda, dOdJ = _jac_backend(backend, aA, 0.0, _ANGLES)
    assert all(numpy.all(numpy.isfinite(d)) for d in (dxdJ, dvdJ, dxda, dvda, dOdJ))
    with pytest.raises(ValueError, match="non-negative"):
        aA(_arr(backend, -0.1), _arr(backend, _ANGLES))


@pytest.mark.parametrize("backend", BACKENDS)
def test_J_and_E_inverses(backend):
    # setup_interp=True: E(J) and J(E) invert each other (also off the grid), and
    # dJ/dE = 1 / Omega; without interpolation J(E) looks up a grid torus, with
    # dJ/dE = 1 / Omega of that torus
    aAi = _aAVI(setup_interp=True)
    for E in (0.05, 0.27, 0.5):
        Et = _arr(backend, E, grad=backend == "torch")
        J = aAi.J(Et)
        assert _is_backend(backend, J)
        assert abs(float(as_numpy(J)) - float(aAi.J(E))) < 1e-13
        assert abs(float(as_numpy(aAi.E(J))) - E) < 1e-13
        if backend == "jax":
            dJdE = float(jax.grad(aAi.J)(jnp.asarray(E)))
        else:
            (dJdE,) = torch.autograd.grad(J, Et)
        assert abs(float(dJdE) * aAi.Freqs(float(aAi.J(E))) - 1.0) < 1e-10
    # a one-torus family: E(J) is the node's tangent line, inverted in closed form
    aA1 = _aAVI(Es=[0.3], setup_interp=True)
    for E in (0.25, 0.3, 0.35):
        J = aA1.J(_arr(backend, E))
        assert abs(float(as_numpy(J)) - float(aA1.J(E))) < 1e-14
        assert abs(float(as_numpy(aA1.E(J))) - E) < 1e-14
    aA = _aAVI(Es=[0.1, 0.3])
    Et = _arr(backend, 0.3, grad=backend == "torch")
    J = aA.J(Et)
    assert float(as_numpy(J)) == aA._js[1]
    if backend == "jax":
        dJdE = float(jax.grad(aA.J)(jnp.asarray(0.3)))
    else:
        (dJdE,) = torch.autograd.grad(J, Et)
    assert abs(float(dJdE) - 1.0 / aA._Omegas[1]) < 1e-14
    with pytest.raises(ValueError, match="Given energy not found"):
        aA.J(_arr(backend, 0.2))


@pytest.mark.parametrize("backend", BACKENDS)
def test_float32_in_float32_out(backend):
    # computed in float64, returned in the dtype of a float32 input
    aA = _aAVI(setup_interp=True)
    if backend == "jax":
        f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731
    else:
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    ref = aA.xvFreqs(0.137, _ANGLES)
    got = aA.xvFreqs(f32(0.137), f32(_ANGLES))
    for g, r in zip(got, ref):
        assert str(g.dtype).endswith("float32")
        numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=2e-6)
    for out in (aA.Freqs(f32(0.137)), aA.E(f32(0.137)), aA.J(f32(0.27))):
        assert str(out.dtype).endswith("float32")


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit():
    # the evaluation (bracketed root find included) traces under jax.jit
    aA = _aAVI(setup_interp=True)
    ref = numpy.concatenate([numpy.ravel(v) for v in aA.xvFreqs(0.137, _ANGLES)])

    def f(j, a):
        return jnp.concatenate([jnp.ravel(v) for v in aA.xvFreqs(j, a)])

    got = jax.jit(f)(jnp.asarray(0.137), jnp.asarray(_ANGLES))
    numpy.testing.assert_allclose(numpy.asarray(got), ref, rtol=0.0, atol=2e-14)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_compile():
    aA = _aAVI(setup_interp=True)
    ref = numpy.concatenate([numpy.ravel(v) for v in aA.xvFreqs(0.137, _ANGLES)])
    f = torch.compile(
        lambda j, a: torch.cat([torch.atleast_1d(v).ravel() for v in aA.xvFreqs(j, a)]),
        backend="eager",
    )
    got = f(torch.tensor(0.137), torch.tensor(_ANGLES))
    numpy.testing.assert_allclose(got.numpy(), ref, rtol=0.0, atol=2e-14)


def test_construction_and_legacy_contract():
    # Under a forced backend, or for a potential with backend parameters, the
    # momentum-matched family is built natively and evaluates there (numpy
    # inputs included). The legacy modes (momentum_matched=False / a point
    # transformation) construct on numpy -- under a forced backend too -- and
    # evaluate natively for backend inputs or under a forced backend; a legacy
    # construction cannot take a potential with backend parameters.
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    isopot = IsothermalDiskPotential(amp=1.0, sigma=0.5)
    legacy = actionAngleVerticalInverse(pot=isopot, Es=[0.3], momentum_matched=False)
    j0 = legacy._js[0]
    for bk in BACKENDS:
        assert _is_backend(bk, legacy(_arr(bk, j0), _arr(bk, numpy.array([0.1])))[0])
        assert _is_backend(bk, legacy.Freqs(_arr(bk, j0)))
        assert _is_backend(bk, legacy.J(_arr(bk, 0.3)))
        with gb.use(bk, force=True):
            aA = actionAngleVerticalInverse(pot=isopot, Es=[0.3])
            assert _is_backend(bk, aA(aA._js[0], numpy.array([0.1]))[0])
            for kw in ({"momentum_matched": False}, {"use_pointtransform": True}):
                aL = actionAngleVerticalInverse(pot=isopot, Es=[0.3], **kw)
                assert _is_backend(bk, aL(aL._js[0], numpy.array([0.1]))[0])
        gpot = IsothermalDiskPotential(amp=_arr(bk, 1.0), sigma=0.5)
        aA = actionAngleVerticalInverse(pot=gpot, Es=[0.3])
        assert _is_backend(bk, aA(float(aA._js[0]), numpy.array([0.1]))[0])
        assert _is_backend(bk, aA.Freqs(float(aA._js[0])))
        assert _is_backend(bk, aA.J(0.3))
        with pytest.raises(NotImplementedError, match="legacy construction"):
            actionAngleVerticalInverse(pot=gpot, Es=[0.3], momentum_matched=False)
        with gb.use(bk, force=True):
            with pytest.raises(NotImplementedError, match="legacy construction"):
                actionAngleVerticalInverse(pot=gpot, Es=[0.3], use_pointtransform=True)


def _param_case(name, backend, p):
    from galpy.potential import IsothermalDiskPotential, KGPotential

    if name == "iso_sigma":
        return IsothermalDiskPotential(amp=1.0, sigma=p)
    return KGPotential(K=p, F=0.5, D=0.5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,p0", [("iso_sigma", 0.5), ("kg_K", 1.0)])
@pytest.mark.parametrize(
    "kw", [{"setup_interp": True}, {"Es": [0.1, 0.3]}], ids=["interp", "two_tori"]
)
def test_derivatives_wrt_potential_parameters(backend, name, p0, kw):
    # the construction is differentiable in the potential's parameters: the
    # (x, v, Omega) of a torus of fixed action at fixed angles, against a
    # 5-point FD of the backend construction itself (numpy's calcxmax roots are
    # too noisy for a sharper reference than ~1e-8)
    from galpy.actionAngle import actionAngleVerticalInverse

    J0 = 0.11

    def out(p):
        aA = actionAngleVerticalInverse(pot=_param_case(name, backend, p), **kw)
        x, v, O = aA.xvFreqs(J0, _ANGLES)
        return x, v, O

    def num(p):
        x, v, O = out(_arr(backend, p))
        return numpy.concatenate([as_numpy(x), as_numpy(v), [float(as_numpy(O))]])

    h = 3e-4
    fd = (-num(p0 + 2 * h) + 8 * num(p0 + h) - 8 * num(p0 - h) + num(p0 - 2 * h)) / (
        12 * h
    )
    if backend == "jax":
        g = numpy.asarray(
            jax.jacfwd(
                lambda p: jnp.concatenate(
                    [jnp.ravel(o) for o in out(p)[:2]] + [jnp.atleast_1d(out(p)[2])]
                )
            )(jnp.asarray(p0))
        )
    else:
        pt = torch.tensor(p0, requires_grad=True)
        x, v, O = out(pt)
        flat = torch.cat([x, v, O.reshape(1)])
        g = numpy.array(
            [
                float(torch.autograd.grad(flat[k], pt, retain_graph=True)[0])
                for k in range(len(flat))
            ]
        )
    numpy.testing.assert_allclose(g, fd, rtol=0.0, atol=3e-10)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_jax_jit_construction():
    # the whole pipeline -- construction and evaluation -- traces under jax.jit,
    # and its derivative in a potential parameter does too
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    def f(s):
        aA = actionAngleVerticalInverse(
            pot=IsothermalDiskPotential(amp=1.0, sigma=s),
            Es=numpy.linspace(0.0, 0.6, 9),
            setup_interp=True,
        )
        x, v, O = aA.xvFreqs(0.11, jnp.asarray(_ANGLES))
        return jnp.concatenate([x, v, jnp.atleast_1d(O)])

    s0 = jnp.asarray(0.5)
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(f)(s0)), numpy.asarray(f(s0)), rtol=0.0, atol=1e-14
    )
    numpy.testing.assert_allclose(
        numpy.asarray(jax.jit(jax.jacfwd(f))(s0)),
        numpy.asarray(jax.jacfwd(f)(s0)),
        rtol=0.0,
        atol=1e-11,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_construction_warns_unconverged(backend):
    # the backend construction checks the map's convergence like numpy does
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import galpyWarning

    pot = LogarithmicHaloPotential(normalize=1.0).toVertical(1.0)
    with gb.use(backend, force=True):
        with pytest.warns(galpyWarning, match="not converged"):
            actionAngleVerticalInverse(pot=pot, Es=[0.5, 4.0], mm_npt=6)


def _offset_oscillator():
    # a harmonic oscillator whose potential is 1, not 0, at the midplane
    from galpy.potential.linearPotential import linearPotential

    class _Offset(linearPotential):
        def __init__(self):
            linearPotential.__init__(self, amp=1.0)

        def _evaluate(self, x, t=0.0):
            return 1.0 + 0.5 * x**2.0

        def _force(self, x, t=0.0):
            return -x

        def _x2deriv(self, x, t=0.0):
            return 1.0 + 0.0 * x

    return _Offset()


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_construction_edge_grids(backend):
    # the backend construction handles the same edge grids as numpy: a bottom
    # torus at E = Phi(0) != 0, a family of bottom tori only, and an energy
    # without a turning point (above Phi at infinity), which raises
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import KGPotential, PlummerPotential

    # Phi(0) = 1: an exact harmonic oscillator, whose bottom torus has the
    # frequency sqrt(Phi''(0)) = 1 exactly (numpy approximates it there by the
    # torus at E + 1e-5, to ~5e-9)
    with gb.use(backend, force=True):
        aA = actionAngleVerticalInverse(pot=_offset_oscillator(), Es=[1.0, 1.5, 2.0])
        for j in (0.0, 0.5, 1.0):
            x, v, O = aA.xvFreqs(j, _ANGLES)
            amp = numpy.sqrt(2.0 * j)
            numpy.testing.assert_allclose(
                as_numpy(x), amp * numpy.sin(_ANGLES), rtol=0.0, atol=1e-13
            )
            numpy.testing.assert_allclose(
                as_numpy(v), amp * numpy.cos(_ANGLES), rtol=0.0, atol=1e-13
            )
            assert abs(float(as_numpy(O)) - 1.0) < 1e-13
    # a family of the bottom torus only
    kg = KGPotential(K=1.0, F=0.5, D=0.5)
    ref = actionAngleVerticalInverse(pot=kg, Es=[0.0])
    with gb.use(backend, force=True):
        aA = actionAngleVerticalInverse(pot=kg, Es=[0.0])
        for g, r in zip(aA.xvFreqs(0.0, _ANGLES), ref.xvFreqs(0.0, _ANGLES)):
            assert _is_backend(backend, g)
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=1e-14)
    plummer = PlummerPotential(normalize=1.0).toVertical(1.0)
    with pytest.raises(RuntimeError, match="turning point could not be found"):
        actionAngleVerticalInverse(pot=plummer, Es=[0.1, 3.28])
    with gb.use(backend, force=True):
        with pytest.raises(RuntimeError, match="turning point could not be found"):
            actionAngleVerticalInverse(pot=plummer, Es=[0.1, 3.28])
        actionAngleVerticalInverse(pot=plummer, Es=[0.1, 1.0])  # bound: no raise


_LEGACY_MODES = {
    "no_mm": {"momentum_matched": False},
    "poly_pt": {"use_pointtransform": True, "pt_deg": 7},
    "exact_pt": {"use_pointtransform": "exact"},
    "exact_pt_only": {"use_pointtransform": "exact", "pt_only": True},
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", list(_LEGACY_MODES))
@pytest.mark.parametrize("interp", [False, True], ids=["grid", "interp"])
def test_legacy_values_match_numpy(backend, mode, interp):
    # the legacy evaluation (built on numpy) gives the numpy path's
    # (x, v, Omega) natively, for backend inputs and under a forced backend, on
    # the grid tori and (interpolated) between them
    from galpy import backend as gb
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import KGPotential

    kw = dict(
        pot=KGPotential(K=1.0, F=0.5, D=0.5),
        Es=numpy.linspace(0.0, 0.6, 31) if interp else [0.1, 0.3],
        nta=64,
        setup_interp=interp,
        **_LEGACY_MODES[mode],
    )
    aA = actionAngleVerticalInverse(**kw)
    with gb.use(backend, force=True):
        aAf = actionAngleVerticalInverse(**kw)
    js = list(aA._js[1:3]) + ([float(aA.J(0.27))] if interp else [])
    for j in js:
        ref = aA.xvFreqs(j, _ANGLES)
        got = aA.xvFreqs(_arr(backend, j), _arr(backend, _ANGLES))
        with gb.use(backend, force=True):
            gotf = aAf.xvFreqs(j, _ANGLES)
        for g, gf, r in zip(got, gotf, ref):
            assert _is_backend(backend, g) and _is_backend(backend, gf)
            numpy.testing.assert_allclose(as_numpy(g), r, rtol=0.0, atol=1e-13)
            numpy.testing.assert_allclose(as_numpy(gf), r, rtol=0.0, atol=1e-13)
        assert abs(float(as_numpy(aA.Freqs(_arr(backend, j)))) - aA.Freqs(j)) < 1e-14
    if interp:
        # the interpolated coefficient tables, NaN off the grid
        for meth in ("nSn", "dSndJ", "pt_coeffs", "pt_deriv_coeffs"):
            E = numpy.array([0.05, 0.27, 0.7])
            got, ref = getattr(aA, meth)(_arr(backend, E)), getattr(aA, meth)(E)
            assert _is_backend(backend, got)
            numpy.testing.assert_allclose(
                as_numpy(got), ref, rtol=0.0, atol=1e-14, equal_nan=True
            )
            assert numpy.all(numpy.isnan(as_numpy(got)[-1]))
        assert abs(float(as_numpy(aA.E(_arr(backend, js[-1])))) - aA.E(js[-1])) < 1e-14
    else:
        with pytest.raises(ValueError, match="Given action/energy not found"):
            aA(_arr(backend, 0.123), _arr(backend, _ANGLES))
        with pytest.raises(ValueError, match="Given action/energy not found"):
            aA.Freqs(_arr(backend, 0.123))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["no_mm", "poly_pt", "exact_pt"])
def test_legacy_derivatives_vs_finite_difference(backend, mode):
    # d(x, v)/dJ and d(x, v)/dtheta of the (interpolated) legacy evaluation
    # against a converged 5-point FD of the numpy path, through the turning points
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import KGPotential

    aA = actionAngleVerticalInverse(
        pot=KGPotential(K=1.0, F=0.5, D=0.5),
        Es=numpy.linspace(0.0, 0.6, 31),
        nta=64,
        setup_interp=True,
        **_LEGACY_MODES[mode],
    )
    j0, h, n = float(aA.J(0.27)), 3e-4, len(_ANGLES)

    def npxv(j, a):
        x, v, O = aA.xvFreqs(j, numpy.atleast_1d(a))
        return numpy.concatenate([x, v, [O]])

    dJ = _fd(lambda j: npxv(j, _ANGLES), j0, h)
    dA = numpy.array([_fd(lambda a: npxv(j0, a)[:2], a, h) for a in _ANGLES])
    dxdJ, dvdJ, dxda, dvda, _ = _jac_backend(backend, aA, j0, _ANGLES)
    numpy.testing.assert_allclose(dxdJ, dJ[:n], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dvdJ, dJ[n : 2 * n], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dxda, dA[:, 0], rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(dvda, dA[:, 1], rtol=0.0, atol=1e-10)


# --- CUDA potential parameters with torch's default device left on the CPU ----
# Every table the native construction creates must follow the potential's
# device; --device cuda sets torch's DEFAULT device to cuda, which hides a
# device-less constructor, so keep the default on the CPU here.
@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs a CUDA GPU"
)
@pytest.mark.parametrize("momentum_matched", [True, False])
def test_cuda_potential_parameters_cpu_default_device(momentum_matched):
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.backend import use
    from galpy.potential import IsothermalDiskPotential

    Es = numpy.linspace(0.0, 2.0, 9)  # includes the bottom torus
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 17)
    ref = actionAngleVerticalInverse(
        pot=IsothermalDiskPotential(amp=1.0, sigma=0.5),
        Es=Es,
        setup_interp=True,
        momentum_matched=momentum_matched,
    )
    xr, vr, Or = ref.xvFreqs(0.3, angles)
    with torch.device("cpu"):
        cuda = torch.device("cuda")
        kw = dict(dtype=torch.float64, device=cuda)
        pot = IsothermalDiskPotential(
            amp=torch.tensor(1.0, **kw), sigma=torch.tensor(0.5, **kw)
        )
        with use("torch"):
            if momentum_matched:
                aA = actionAngleVerticalInverse(pot=pot, Es=Es, setup_interp=True)
            else:  # legacy: built on numpy, evaluated natively
                aA = actionAngleVerticalInverse(
                    pot=IsothermalDiskPotential(amp=1.0, sigma=0.5),
                    Es=Es,
                    setup_interp=True,
                    momentum_matched=False,
                )
        x, v, O = aA.xvFreqs(torch.tensor(0.3, **kw), torch.tensor(angles, **kw))
    assert x.device.type == "cuda" and v.device.type == "cuda"
    numpy.testing.assert_allclose(as_numpy(x), xr, rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(as_numpy(v), vr, rtol=0.0, atol=1e-10)
    numpy.testing.assert_allclose(float(as_numpy(O)), Or, rtol=1e-10)
