###############################################################################
# test_backend_streamgapdf.py: backend (jax/torch) coverage for the analytic
# impulse-approximation kernels of streamgapdf (Plummer / Hernquist, straight &
# curved-stream, HernquistX, _rotation_vy). The numpy path is byte-identical
# (test_streamgapdf_impulse unchanged); this exercises the resolved-namespace
# dispatch:
#   (a) value parity numpy<->jax<->torch of every kernel (incl. the wperp->0
#       degenerate perpendicular-impact branch and all three HernquistX
#       regimes), reusing the test_streamgapdf_impulse configs with FIXED seeds,
#   (b) grad-vs-FD of ||plummer_curvedstream||^2 w.r.t. b/GM/rs/w and of
#       HernquistX across regimes (jax.grad / torch.autograd vs central FD).
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

from galpy.backend import as_numpy, is_backend_array, use
from galpy.df.streamgapdf import (
    HernquistX,
    _rotation_vy,
    impulse_deltav_hernquist,
    impulse_deltav_hernquist_curvedstream,
    impulse_deltav_plummer,
    impulse_deltav_plummer_curvedstream,
)


def _to_backend(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.asarray(x)


# ------- Fixed-seed input configs (mirror test_streamgapdf_impulse) -------
def _make_cases():
    numpy.random.seed(12345)
    xpos = numpy.random.normal(size=20)
    vb = numpy.zeros((20, 3))
    vb[:, 0] = 3.4
    xposc = numpy.array([xpos, numpy.zeros(20), numpy.zeros(20)]).T
    wperp_nonzero = numpy.array([0.0, numpy.pi / 2.0, 0.0])
    # s spanning all three HernquistX regimes incl. near s=1
    sarr = numpy.concatenate(
        [
            numpy.linspace(1e-6, 0.999999, 30),
            numpy.array([1.0 - 1e-11, 1.0, 1.0 + 1e-11]),
            numpy.linspace(1.000001, numpy.sqrt(2.0), 30),
        ]
    )
    return {
        "plummer_bunch": (
            impulse_deltav_plummer,
            dict(v=vb.copy(), y=xpos.copy(), b=3.0, w=wperp_nonzero, GM=1.5, rs=4.0),
        ),
        # perpendicular impact -> wperp==0 degenerate (guarded) branch
        "plummer_perp": (
            impulse_deltav_plummer,
            dict(
                v=numpy.array([[0.0, numpy.pi, 0.0]]),
                y=numpy.array([0.0]),
                b=3.0,
                w=wperp_nonzero,
                GM=1.5,
                rs=4.0,
            ),
        ),
        "plummer_curved_bunch": (
            impulse_deltav_plummer_curvedstream,
            dict(
                v=vb.copy(),
                x=xposc.copy(),
                b=3.0,
                w=wperp_nonzero,
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.0, 0.0]),
                GM=numpy.pi,
                rs=numpy.exp(1.0),
            ),
        ),
        "plummer_curved_single": (
            impulse_deltav_plummer_curvedstream,
            dict(
                v=numpy.array([[3.4, 0.1, 0.2]]),
                x=numpy.array([[4.0, 0.1, 0.0]]),
                b=3.0,
                w=numpy.array([0.2, 1.1, 0.3]),
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.1, 0.2]),
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquist_bunch": (
            impulse_deltav_hernquist,
            dict(
                v=vb.copy(), y=xpos.copy(), b=3.0, w=wperp_nonzero, GM=numpy.pi, rs=2.0
            ),
        ),
        # perpendicular impact -> wperp==0 degenerate (guarded) branch
        "hernquist_perp": (
            impulse_deltav_hernquist,
            dict(
                v=numpy.array([[0.0, numpy.pi, 0.0]]),
                y=numpy.array([2.0]),
                b=3.0,
                w=wperp_nonzero,
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquist_curved_bunch": (
            impulse_deltav_hernquist_curvedstream,
            dict(
                v=vb.copy(),
                x=xposc.copy(),
                b=3.0,
                w=wperp_nonzero,
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.0, 0.0]),
                GM=numpy.pi,
                rs=numpy.exp(1.0),
            ),
        ),
        "hernquist_curved_single": (
            impulse_deltav_hernquist_curvedstream,
            dict(
                v=numpy.array([[3.4, 0.1, 0.2]]),
                x=numpy.array([[4.0, 0.1, 0.0]]),
                b=3.0,
                w=numpy.array([0.2, 1.1, 0.3]),
                x0=numpy.array([0.0, 0.0, 0.0]),
                v0=numpy.array([3.4, 0.1, 0.2]),
                GM=1.5,
                rs=4.0,
            ),
        ),
        "hernquistX": (HernquistX, dict(s=sarr)),
        "rotation_vy_fwd": (_rotation_vy, dict(v=vb.copy(), inv=False)),
        "rotation_vy_inv": (_rotation_vy, dict(v=vb.copy(), inv=True)),
    }


CASES = _make_cases()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(CASES))
def test_kernel_parity(backend, name):
    fn, kwargs = CASES[name]
    ref = numpy.asarray(fn(**kwargs))
    bkwargs = {
        k: (_to_backend(backend, v) if isinstance(v, numpy.ndarray) else v)
        for k, v in kwargs.items()
    }
    got = fn(**bkwargs)
    assert is_backend_array(got), f"{name} on {backend} did not return a backend array"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10, atol=1e-12)


# ------- grad-vs-FD: ||plummer_curvedstream||^2 w.r.t. b/GM/rs/w -------
_GRAD_CFG = dict(
    v=numpy.array([[3.4, 0.1, 0.2], [3.3, -0.1, 0.15]]),
    x=numpy.array([[4.0, 0.1, 0.0], [3.5, -0.2, 0.1]]),
    b=3.0,
    w=numpy.array([0.2, 1.1, 0.3]),
    x0=numpy.array([0.0, 0.0, 0.0]),
    v0=numpy.array([3.4, 0.1, 0.2]),
    GM=1.5,
    rs=4.0,
)


def _loss_np(b, GM, rs, w):
    c = _GRAD_CFG
    kick = impulse_deltav_plummer_curvedstream(
        c["v"], c["x"], b, w, c["x0"], c["v0"], GM, rs
    )
    return float(numpy.sum(kick**2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", ["b", "GM", "rs"])
def test_plummer_curved_grad_scalar_vs_fd(backend, param):
    c = _GRAD_CFG
    base = dict(b=c["b"], GM=c["GM"], rs=c["rs"], w=c["w"])
    h = 1e-6
    lo = dict(base)
    lo[param] = base[param] - h
    hi = dict(base)
    hi[param] = base[param] + h
    gfd = (_loss_np(**hi) - _loss_np(**lo)) / (2.0 * h)

    def loss_backend(bval, GMval, rsval, wval):
        kick = impulse_deltav_plummer_curvedstream(
            _to_backend(backend, c["v"]),
            _to_backend(backend, c["x"]),
            bval,
            wval,
            _to_backend(backend, c["x0"]),
            _to_backend(backend, c["v0"]),
            GMval,
            rsval,
        )
        return (kick**2).sum()

    if backend == "jax":
        args = dict(
            bval=jnp.asarray(c["b"]),
            GMval=jnp.asarray(c["GM"]),
            rsval=jnp.asarray(c["rs"]),
            wval=jnp.asarray(c["w"]),
        )
        key = {"b": "bval", "GM": "GMval", "rs": "rsval"}[param]
        g = float(jax.grad(lambda p: loss_backend(**{**args, key: p}))(args[key]))
    else:
        vals = {
            "bval": torch.tensor(c["b"], requires_grad=(param == "b")),
            "GMval": torch.tensor(c["GM"], requires_grad=(param == "GM")),
            "rsval": torch.tensor(c["rs"], requires_grad=(param == "rs")),
            "wval": torch.tensor(c["w"]),
        }
        key = {"b": "bval", "GM": "GMval", "rs": "rsval"}[param]
        loss_backend(**vals).backward()
        g = float(vals[key].grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_plummer_curved_grad_w_vs_fd(backend):
    c = _GRAD_CFG
    h = 1e-6
    gfd = numpy.empty(3)
    for i in range(3):
        wl = c["w"].copy()
        wl[i] -= h
        wh = c["w"].copy()
        wh[i] += h
        gfd[i] = (
            _loss_np(c["b"], c["GM"], c["rs"], wh)
            - _loss_np(c["b"], c["GM"], c["rs"], wl)
        ) / (2.0 * h)

    def loss_backend(wval):
        kick = impulse_deltav_plummer_curvedstream(
            _to_backend(backend, c["v"]),
            _to_backend(backend, c["x"]),
            c["b"],
            wval,
            _to_backend(backend, c["x0"]),
            _to_backend(backend, c["v0"]),
            c["GM"],
            c["rs"],
        )
        return (kick**2).sum()

    if backend == "jax":
        g = numpy.asarray(jax.grad(loss_backend)(jnp.asarray(c["w"])))
    else:
        wt = torch.tensor(c["w"], requires_grad=True)
        loss_backend(wt).backward()
        g = wt.grad.detach().cpu().numpy()
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("s0", [0.3, 0.7, 0.95, 1.05, 1.3])
def test_hernquistX_grad_vs_fd(backend, s0):
    h = 1e-7
    gfd = (float(HernquistX(s0 + h)) - float(HernquistX(s0 - h))) / (2.0 * h)
    if backend == "jax":
        g = float(jax.grad(lambda s: HernquistX(s))(jnp.asarray(s0)))
    else:
        st = torch.tensor(s0, requires_grad=True)
        HernquistX(st).backward()
        g = float(st.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------
# Gap-track Phase 2b: the backend twin of _determine_deltaOmegaTheta_kick --
# propagate the velocity kick deltav(angle) -> delta(Omega,theta)(angle) along
# the near-impact track and build the differentiable dO/da(angle) interpolants.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _gapdf_kick():
    # A real (numpy) Sanders15 trailing gap DF; capture the numpy reference of
    # the kick track, then each test swaps _kick_deltav to a backend array and
    # re-runs the (fast) kick propagation, restoring numpy state afterwards.
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0)
    sdf = streamgapdf(
        sigv / V0,
        progenitor=prog,
        pot=lp,
        aA=aAI,
        leading=False,
        nTrackChunks=26,
        nTrackIterations=1,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        vo=V0,
        ro=R0,
        impactb=0.0,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
        timpact=0.88 / conversion.time_in_Gyr(V0, R0),
        impact_angle=-2.34,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
        rs=0.625 / R0,
    )
    deltav_np = numpy.asarray(sdf._kick_deltav).copy()
    theta = numpy.linspace(1e-4, sdf._deltaAngleTrackImpact * 0.999, 40)
    evals = (
        "_kick_interpdOr",
        "_kick_interpdOp",
        "_kick_interpdOz",
        "_kick_interpdar",
        "_kick_interpdap",
        "_kick_interpdaz",
        "_kick_interpdOpar",
        "_kick_interpdOperp0",
        "_kick_interpdOperp1",
    )
    ref = {
        "dOap": sdf._kick_dOap.copy(),
        "evals": {e: getattr(sdf, e)(theta).copy() for e in evals},
    }
    return sdf, ref, deltav_np, theta, evals


def _reset_kick_numpy(sdf, deltav_np):
    sdf._kick_deltav = deltav_np
    sdf._determine_deltaOmegaTheta_kick(3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_kick_track_value_parity(_gapdf_kick, backend):
    # The backend twin reproduces the numpy kick track to the Spline1D-vs-scipy
    # floor, and dispatch actually fires (the outputs are backend arrays).
    sdf, ref, deltav_np, theta, evals = _gapdf_kick
    try:
        sdf._kick_deltav = _to_backend(backend, deltav_np)
        sdf._determine_deltaOmegaTheta_kick(3)
        assert is_backend_array(sdf._kick_dOap)
        numpy.testing.assert_allclose(
            as_numpy(sdf._kick_dOap), ref["dOap"], rtol=1e-9, atol=1e-11
        )
        thb = _to_backend(backend, theta)
        for name in evals:
            bv = as_numpy(getattr(sdf, name)(thb))
            numpy.testing.assert_allclose(
                bv, ref["evals"][name], rtol=1e-7, atol=1e-9, err_msg=name
            )
    finally:
        del sdf  # the copy; the shared fixture was never touched


@pytest.mark.parametrize("backend", BACKENDS)
def test_kick_track_grad_vs_fd(_gapdf_kick, backend):
    # d(sum w * _kick_interpdOpar(theta)) / d(deltav) is exact vs central FD --
    # the frequency/angle kick is differentiable in the velocity kick (composes
    # with the #1167 impulse's d(deltav)/d(perturber)).
    sdf, ref, deltav_np, theta, evals = _gapdf_kick
    rng = numpy.random.RandomState(3)
    w = rng.randn(len(theta))

    def loss(dv_backend, th_backend, w_backend):
        sdf._kick_deltav = dv_backend
        sdf._determine_deltaOmegaTheta_kick(3)
        return (w_backend * sdf._kick_interpdOpar(th_backend)).sum()

    try:
        thb = _to_backend(backend, theta)
        wb = _to_backend(backend, w)
        # direction for the FD check
        d = rng.randn(*deltav_np.shape)
        d /= numpy.linalg.norm(d)
        if backend == "jax":
            g = jax.grad(lambda dv: loss(dv, thb, wb))(jnp.asarray(deltav_np))
            ad_dir = float(numpy.sum(as_numpy(g) * d))
        else:
            dv = torch.tensor(deltav_np, requires_grad=True)
            loss(dv, thb, wb).backward()
            ad_dir = float(numpy.sum(as_numpy(dv.grad) * d))
        h = 1e-3
        lp = float(as_numpy(loss(_to_backend(backend, deltav_np + h * d), thb, wb)))
        lm = float(as_numpy(loss(_to_backend(backend, deltav_np - h * d), thb, wb)))
        fd = (lp - lm) / (2.0 * h)
        numpy.testing.assert_allclose(ad_dir, fd, rtol=1e-6, atol=1e-9)
    finally:
        _reset_kick_numpy(sdf, deltav_np)


# --------------------------------------------------------------------------
# Gap-track Phase 3: the backend gap DF-evaluation layer (pOparapar / minOpar /
# _density_par / meanOmega). Value parity on both backends; torch grad-vs-FD
# (density/meanOmega are torch-differentiable w.r.t. the velocity kick; jax
# differentiability through minOpar's argmin integration-limit is a follow-up).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_gapdf_eval_value_parity(_gapdf_kick, backend):
    sdf, _ref, deltav_np, _theta, _evals = _gapdf_kick
    dangles = [0.05, 0.1, 0.2, 0.3]
    Opar_arr = numpy.linspace(-0.5, 0.8, 25)
    ref_dens = {d: float(sdf._density_par(d)) for d in dangles}
    ref_mO = {
        d: float(sdf.meanOmega(d, oned=True, use_physical=False)) for d in dangles
    }
    # 3D (oned=False) meanOmega -> exercises the is_backend_array(dO1D) combine
    ref_mO3d = {
        d: numpy.asarray(sdf.meanOmega(d, use_physical=False)).copy() for d in dangles
    }
    ref_min = {d: float(sdf.minOpar(d)) for d in dangles}
    ref_pO = {d: sdf.pOparapar(Opar_arr.copy(), d).copy() for d in dangles}
    try:
        sdf._kick_deltav = _to_backend(backend, deltav_np)
        sdf._determine_deltaOmegaTheta_kick(3)
        assert is_backend_array(sdf._kick_interpdOpar_poly.c)
        for d in dangles:
            numpy.testing.assert_allclose(
                float(as_numpy(sdf._density_par(d))), ref_dens[d], rtol=1e-6
            )
            numpy.testing.assert_allclose(
                float(as_numpy(sdf.meanOmega(d, oned=True, use_physical=False))),
                ref_mO[d],
                rtol=1e-6,
            )
            numpy.testing.assert_allclose(
                as_numpy(sdf.meanOmega(d, use_physical=False)),
                ref_mO3d[d],
                rtol=1e-6,
                atol=1e-10,
            )
            numpy.testing.assert_allclose(
                float(as_numpy(sdf.minOpar(d))), ref_min[d], rtol=1e-6, atol=1e-12
            )
            numpy.testing.assert_allclose(
                as_numpy(sdf.pOparapar(_to_backend(backend, Opar_arr), d)),
                ref_pO[d],
                rtol=1e-6,
                atol=1e-10,
            )
    finally:
        _reset_kick_numpy(sdf, deltav_np)


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
@pytest.mark.parametrize("method", ["_density_par", "meanOmega"])
def test_gapdf_eval_grad_vs_fd_torch(_gapdf_kick, method):
    # d(density|meanOmega)/d(deltav) is exact vs central FD (h-converged) -- the
    # gap density/mean-frequency are differentiable w.r.t. the velocity kick,
    # hence w.r.t. the perturber via the #1167 impulse.
    sdf, _ref, deltav_np, _theta, _evals = _gapdf_kick
    dangle = 0.15
    rng = numpy.random.RandomState(4)
    d = rng.randn(*deltav_np.shape)
    d /= numpy.linalg.norm(d)

    def loss(x):
        sdf._kick_deltav = x
        sdf._determine_deltaOmegaTheta_kick(3)
        if method == "_density_par":
            return sdf._density_par(dangle)
        return sdf.meanOmega(dangle, oned=True, use_physical=False)

    try:
        tv = torch.tensor(deltav_np, requires_grad=True)
        loss(tv).backward()
        ad = float(numpy.sum(as_numpy(tv.grad) * d))
        h = 1e-4
        fp = float(as_numpy(loss(torch.as_tensor(deltav_np + h * d))))
        fm = float(as_numpy(loss(torch.as_tensor(deltav_np - h * d))))
        fd = (fp - fm) / (2.0 * h)
        numpy.testing.assert_allclose(ad, fd, rtol=1e-4, atol=1e-9)
    finally:
        _reset_kick_numpy(sdf, deltav_np)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gapdf_kick_spline_order_1(_gapdf_kick, backend):
    # Exercise the k=1 (piecewise-linear) backend poly build: _coeffs only exists
    # for the k=3 cubic, so k=1 synthesizes poly.c = [slope, y_left].
    sdf, _ref, deltav_np, _theta, _evals = _gapdf_kick
    try:
        sdf._kick_deltav = _to_backend(backend, deltav_np)
        sdf._determine_deltaOmegaTheta_kick(1)
        assert sdf._kick_spline_order == 1
        assert is_backend_array(sdf._kick_interpdOpar_poly.c)
        assert sdf._kick_interpdOpar_poly.c.shape[0] == 2  # [slope, y_left]
        # the DF-eval layer still evaluates on the k=1 pw-linear kick
        assert numpy.isfinite(float(as_numpy(sdf._density_par(0.1))))
    finally:
        _reset_kick_numpy(sdf, deltav_np)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gapdf_eval_broadcasts_over_dangle(_gapdf_kick, backend):
    # The backend density / meanOmega / minOpar broadcast over an array of
    # angles (one call on the whole grid where the track interpolation used to
    # loop): elementwise the same as the per-angle calls, and the per-angle
    # calls are the numpy values (test_gapdf_eval_value_parity)
    sdf, _ref, deltav_np, _theta, _evals = _gapdf_kick
    dangles = numpy.array([0.05, 0.1, 0.2, 0.3, 0.9])
    try:
        sdf._kick_deltav = _to_backend(backend, deltav_np)
        sdf._determine_deltaOmegaTheta_kick(3)
        db = _to_backend(backend, dangles)
        for fn in (
            lambda d: sdf._density_par(d),
            lambda d: sdf.meanOmega(d, oned=True, use_physical=False),
            lambda d: sdf.minOpar(d),
        ):
            batched = as_numpy(fn(db))
            assert batched.shape == dangles.shape
            single = numpy.array([float(as_numpy(fn(d))) for d in dangles])
            numpy.testing.assert_allclose(batched, single, rtol=1e-14, atol=1e-16)
    finally:
        _reset_kick_numpy(sdf, deltav_np)


@pytest.fixture(scope="module")
def _gapdf_backend_ct():
    """A streamgapdf whose impact coordtransform has been re-run on the backend.

    Built on numpy first and then re-run with a backend progenitor + diffrax aA
    (the idiom test_backend_streamdf uses), because a full backend __init__ is
    ~280 s -- too close to the per-test cap. Module-scoped so the several
    backend-parity assertions below share the one expensive setup.

    Returns ``(sdf, ref, kick_ref)``: the object with backend-side track
    quantities, the numpy reference for the coordtransform, and the numpy
    reference for the kick interpolation (captured before the switch, with the
    kick attributes then cleared so the backend run rebuilds them).
    """
    import numpy as _np

    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    V0, R0 = 220.0, 8.0
    ic = [
        2.6556151742081835,
        0.2183747276300308,
        0.67876510797240575,
        -2.0143395648974671,
        -0.3273737682604374,
        0.24218273922966019,
    ]
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    sdf = streamgapdf(
        0.365 * (10.0 / 2.0) ** (1.0 / 3.0) / V0,
        progenitor=Orbit(_np.array(ic)),
        pot=lp,
        aA=actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=20.0),
        leading=False,
        nTrackChunks=5,
        nTrackIterations=1,
        nTrackChunksImpact=5,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        impactb=0.1 / R0,
        subhalovel=_np.array([6.82200571, 132.7700529, 149.4174464]) / V0,
        timpact=0.88 / conversion.time_in_Gyr(V0, R0),
        impact_angle=-2.34,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
        rs=0.625 / R0,
    )
    ref = {
        k: _np.asarray(getattr(sdf, k), dtype=float)
        for k in (
            "_gap_thetasTrack",
            "_gap_ObsTrack",
            "_gap_ObsTrackAA",
            "_gap_ObsTrackXY",
            "_gap_detdOdJps",
            "_gap_alljacsTrack",
            "_gap_allinvjacsTrack",
        )
    }
    # numpy reference for the kick interpolation, then clear the cached
    # attributes: _interpolate_stream_track_kick early-returns when
    # _kick_interpolatedThetasTrack already exists, so the backend run would
    # otherwise never build anything.
    sdf._interpolate_stream_track_kick()
    sdf._interpolate_stream_track_kick_aA()
    kick_ref = {
        k: _np.asarray(getattr(sdf, k), dtype=float)
        for k in (
            "_kick_interpolatedThetasTrack",
            "_kick_interpolatedObsTrackXY",
            "_kick_interpolatedObsTrack",
            "_kick_interpolatedObsTrackAA",
            "_kick_ObsTrackXY_closest",
        )
    }
    for k in (
        "_kick_interpolatedThetasTrack",
        "_kick_interpolatedObsTrackXY",
        "_kick_interpolatedObsTrack",
        "_kick_interpolatedObsTrackAA",
    ):
        delattr(sdf, k)
    with use("jax", force=True):
        sdf._aA = actionAngleIsochroneApprox(
            pot=lp,
            b=0.8,
            tintJ=20.0,
            integrate_method="diffrax",
            integrate_kwargs={"max_steps": 2000000},
        )
        prog = Orbit(jnp.asarray(ic))
        prog.turn_physical_off()
        sdf._progenitor = prog
        # through the DISPATCH, not the private method: that also re-runs
        # _gap_progenitor_setup, which has to pick the backend integrator
        # NB the SIGNED impact angle: the object stores numpy.fabs(...), and
        # feeding that back flips the arm and trips the leading/trailing check
        sdf._determine_impact_coordtransform(
            sdf._deltaAngleTrackImpact,
            sdf._nTrackChunksImpact,
            sdf._timpact,
            -2.34,
        )
    return sdf, ref, kick_ref


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_impact_coordtransform_backend_matches_numpy(_gapdf_backend_ct):
    # streamgapdf's (x,v) <-> (O,theta) setup at the impact used streamdf's NUMPY
    # per-chunk helper directly, bypassing the backend dispatch; the backend twin
    # must reproduce it.
    import numpy as _np

    sdf, ref, _ = _gapdf_backend_ct
    # the Jacobian determinant and its inverse amplify, as in the streamdf track
    tols = {"_gap_detdOdJps": 1e-3, "_gap_allinvjacsTrack": 1e-3}
    for k, r in ref.items():
        got = as_numpy(getattr(sdf, k))
        # the chunk-map OUTPUTS must stay on the backend; _gap_thetasTrack
        # follows its extent, which is numpy for an object built on numpy
        if k != "_gap_thetasTrack":
            assert is_backend_array(getattr(sdf, k)), f"{k} must stay on the backend"
        rel = _np.max(_np.abs(_np.asarray(got, dtype=float) - r)) / max(
            _np.max(_np.abs(r)), 1e-30
        )
        assert rel < tols.get(k, 1e-4), f"{k} backend-vs-numpy {rel:.3e}"


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_kick_interpolation_backend_matches_numpy(_gapdf_backend_ct):
    # With the track on the backend, _interpolate_stream_track_kick and its
    # _aA twin take their backend branches: six Spline1D fits and a stack in
    # place of six scipy InterpolatedUnivariateSplines and numpy column
    # assignment, neither of which can hold a traced value. Both are driven
    # through the DISPATCH so the branch selection is covered too.
    import numpy as _np

    sdf, _, kick_ref = _gapdf_backend_ct
    with use("jax", force=True):
        sdf._interpolate_stream_track_kick()
        sdf._interpolate_stream_track_kick_aA()
    for k, r in kick_ref.items():
        got = getattr(sdf, k)
        # _kick_interpolatedThetasTrack is the interpolation GRID: _span_grid
        # keeps it numpy unless the knots are themselves traced, which they are
        # not for an object built on numpy (same exception the coordtransform
        # test makes for _gap_thetasTrack). Its VALUES are still checked below.
        if k != "_kick_interpolatedThetasTrack":
            assert is_backend_array(got), f"{k} must stay on the backend"
        rel = _np.max(_np.abs(_np.asarray(as_numpy(got), dtype=float) - r)) / max(
            _np.max(_np.abs(r)), 1e-30
        )
        assert rel < 1e-4, f"{k} backend-vs-numpy {rel:.3e}"
    # the six per-coordinate splines must be the backend Spline1D, and agree
    # with the interpolated track they were used to build
    itp = sdf._kick_interpolatedThetasTrack
    for ii, nm in enumerate(("X", "Y", "Z", "vX", "vY", "vZ")):
        spl = getattr(sdf, f"_kick_interpTrack{nm}")
        col = as_numpy(spl(itp))
        want = _np.asarray(
            as_numpy(sdf._kick_interpolatedObsTrackXY)[:, ii], dtype=float
        )
        _np.testing.assert_allclose(_np.asarray(col, dtype=float), want, rtol=1e-12)


# --------------------------------------------------------------------------
# End-to-end perturber chain: GM / rs / impactb -> deltav -> d(Omega,theta),
# differentiated in ONE pass through the assembled DF. The pieces were covered
# separately before (the impulse kernels vs d/d(deltav) of the DF observables),
# but nothing differentiated a perturber parameter all the way through, which
# is the gradient a subhalo fit actually needs.
# --------------------------------------------------------------------------
_CHAIN_V0, _CHAIN_R0 = 220.0, 8.0


def _chain_kick(sdf, param, val, spline_order=3):
    """Re-run the kick determination with one perturber parameter replaced."""
    from galpy.util import conversion

    # the SIGNED angle: the object stores numpy.fabs(impact_angle), and feeding
    # that back flips the arm and trips the leading/trailing guard
    signed_angle = sdf._impact_angle if sdf._leading else -sdf._impact_angle
    base = dict(
        impact_angle=signed_angle,
        impactb=0.1 / _CHAIN_R0,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / _CHAIN_V0,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(_CHAIN_V0, _CHAIN_R0),
        rs=0.625 / _CHAIN_R0,
    )
    base[param] = val
    sdf._determine_deltav_kick(
        base["impact_angle"],
        base["impactb"],
        base["subhalovel"],
        base["GM"],
        base["rs"],
        None,
        spline_order,
        False,
    )
    sdf._determine_deltaOmegaTheta_kick(spline_order)
    return sdf._kick_dOap


@pytest.mark.parametrize("param", ["GM", "rs", "impactb"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gapdf_perturber_chain_grad_vs_fd(_gapdf_kick, backend, param, request):
    import copy

    from galpy.util import conversion

    # deepcopy: _determine_deltav_kick rebuilds _kick_ObsTrackXY_closest on
    # whichever backend is active, and the fixture is module-scoped, so mutating
    # it in place leaks a jax array into the torch run (and vice versa) and the
    # namespace probe then sees two backends. A copy costs far less than
    # rebuilding the DF and keeps each parametrization independent.
    sdf = copy.deepcopy(_gapdf_kick[0])
    deltav_np = _gapdf_kick[2]
    x0 = {
        "GM": 10.0**-2.0 / conversion.mass_in_1010msol(_CHAIN_V0, _CHAIN_R0),
        "rs": 0.625 / _CHAIN_R0,
        "impactb": 0.1 / _CHAIN_R0,
    }[param]

    def loss_np(v):
        return float(
            numpy.sum(numpy.asarray(as_numpy(_chain_kick(sdf, param, v))) ** 2.0)
        )

    try:
        h = 1e-6 * x0
        fd = (loss_np(x0 + h) - loss_np(x0 - h)) / (2.0 * h)
        with use(backend, force=True):
            if backend == "jax":
                ad = float(
                    jax.grad(
                        lambda v: jnp.sum(
                            jnp.asarray(_chain_kick(sdf, param, v)) ** 2.0
                        )
                    )(jnp.asarray(x0))
                )
            else:
                t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
                (torch.as_tensor(_chain_kick(sdf, param, t)) ** 2.0).sum().backward()
                ad = float(t.grad)
        # the chain is pure arithmetic on the impulse kernels, so it is exact --
        # the FD reference is the only error (rtol 1e-5 leaves ~1000x margin on
        # the observed ~1e-10 agreement)
        numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-12)
    finally:
        _reset_kick_numpy(sdf, deltav_np)


# --------------------------------------------------------------------------
# The whole constructor on the backend: a jax progenitor + a diffrax aA build
# every stage (offset setup, impact coordinate transform, kick, track) as jax
# arrays, eagerly and under jax.jit, and the model is differentiable in the
# subhalo parameters through it.
# --------------------------------------------------------------------------
_FULL_V0, _FULL_R0 = 220.0, 8.0
_FULL_IC = [
    2.6556151742081835,
    0.2183747276300308,
    0.67876510797240575,
    -2.0143395648974671,
    -0.3273737682604374,
    0.24218273922966019,
]
_FULL_DANGLES = numpy.array([0.3, 0.6, 0.9])


def _full_kwargs():
    from galpy.util import conversion

    V0, R0 = _FULL_V0, _FULL_R0
    return dict(
        leading=False,
        nTrackChunks=5,
        nTrackIterations=1,
        nTrackChunksImpact=5,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        impactb=0.1 / R0,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
        timpact=0.88 / conversion.time_in_Gyr(V0, R0),
        impact_angle=-2.34,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
        rs=0.625 / R0,
    )


def _full_build(backend, **overrides):
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    if backend == "numpy":
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=20.0, ntintJ=1000)
        prog = Orbit(numpy.array(_FULL_IC))
    else:
        aA = actionAngleIsochroneApprox(
            pot=lp, b=0.8, tintJ=20.0, ntintJ=1000, integrate_method="diffrax"
        )
        prog = Orbit(jnp.asarray(_FULL_IC))
    kw = _full_kwargs()
    kw.update(overrides)
    with use(backend, force=True):
        return streamgapdf(
            0.365 * (10.0 / 2.0) ** (1.0 / 3.0) / _FULL_V0,
            progenitor=prog,
            pot=lp,
            aA=aA,
            **kw,
        )


def _full_outputs(sdf):
    """density and mean frequency along the stream, the kicks, both tracks"""
    if is_backend_array(sdf._kick_dOap):  # the backend evaluators broadcast
        d = jnp.asarray(_FULL_DANGLES)
        dens = sdf._density_par(d)
        mO = sdf.meanOmega(d, oned=True, use_physical=False)
    else:
        dens = numpy.array([sdf._density_par(d) for d in _FULL_DANGLES])
        mO = numpy.array(
            [sdf.meanOmega(d, oned=True, use_physical=False) for d in _FULL_DANGLES]
        )
    return {
        "density": dens,
        "meanOmega": mO,
        "kick_dOap": sdf._kick_dOap,
        "gap_ObsTrack": sdf._gap_ObsTrack,
        "ObsTrack": sdf._ObsTrack,
        "interpolatedObsTrackXY": sdf._interpolatedObsTrackXY,
    }


@pytest.fixture(scope="module")
def _full_pair():
    """(numpy-built, jax-built) streamgapdf with the same configuration."""
    ref = _full_build("numpy")
    bk = _full_build("jax")
    return ref, bk


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_full_construction_on_jax_matches_numpy(_full_pair):
    # Every stage runs on jax (nothing laundered through numpy), and the model
    # agrees with the numpy one to the gap between numpy's finite-difference
    # action-angle Jacobians and the backend's exact ones. Measured (max abs
    # difference / max abs value): density 2.5e-6, meanOmega 1.5e-6, tracks
    # 3.3e-6 / 4.8e-6 / 9.4e-6, and the kicks 6.2e-5 -- the kicks map the
    # velocity kick through the near-impact AA Jacobians themselves, where the
    # finite-difference gap is largest (cf. the streamdf track Jacobians).
    tol = {"kick_dOap": 2e-4}
    ref, bk = _full_pair
    got, want = _full_outputs(bk), _full_outputs(ref)
    for k in got:
        assert is_backend_array(got[k]), f"{k} was laundered to numpy"
        g, w = numpy.asarray(as_numpy(got[k])), numpy.asarray(want[k])
        rel = numpy.max(numpy.abs(g - w)) / numpy.max(numpy.abs(w))
        assert rel < tol.get(k, 2e-5), f"{k}: jax vs numpy {rel:.2e}"
    for k in ("_gap_alljacsTrack", "_kick_deltav", "_kick_interpolatedObsTrackAA"):
        assert is_backend_array(getattr(bk, k)), f"{k} was laundered to numpy"


def _kick_tail(sdf, GM, rs, impactb, subhalovel):
    """Re-run the constructor from the kick on (a copy of) a built object.

    The subhalo parameters enter the constructor first in _determine_deltav_kick;
    everything before it (offset setup, impact coordinate transform) does not
    depend on them, so for these parameters this IS the full rebuild."""
    import copy

    s = copy.copy(sdf)
    with use("jax", force=True):
        s._determine_deltav_kick(-2.34, impactb, subhalovel, GM, rs, None, 3, False)
        s._determine_deltaOmegaTheta_kick(3)
        d = jnp.asarray(_FULL_DANGLES)
        return jnp.concatenate(
            [
                s._density_par(d),
                s.meanOmega(d, oned=True, use_physical=False),
                jnp.ravel(s._kick_dOap[::25]),
            ]
        )


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_full_construction_grad_subhalo_params_vs_fd(_full_pair):
    # d(density, meanOmega, kicks)/d(GM, rs, b, w) through the backend-built
    # model, vs Richardson-extrapolated central differences of rebuilds.
    _, bk = _full_pair
    kw = _full_kwargs()
    p0 = numpy.concatenate(
        [[kw["GM"], kw["rs"], kw["impactb"]], numpy.asarray(kw["subhalovel"])]
    )

    def f(p):
        return _kick_tail(bk, p[0], p[1], p[2], p[3:6])

    jac = numpy.asarray(jax.jacrev(f)(jnp.asarray(p0)))

    def cfd(i, h):
        dp = numpy.zeros_like(p0)
        dp[i] = h
        return (
            numpy.asarray(f(jnp.asarray(p0 + dp)))
            - numpy.asarray(f(jnp.asarray(p0 - dp)))
        ) / (2.0 * h)

    for i in range(p0.shape[0]):
        h = 1e-3 * abs(p0[i])
        fd = (4.0 * cfd(i, h / 2.0) - cfd(i, h)) / 3.0  # Richardson: O(h^4)
        scale = numpy.max(numpy.abs(fd))
        assert scale > 0.0
        err = numpy.max(numpy.abs(jac[:, i] - fd)) / scale
        # measured 4e-9 (GM) .. 4.9e-7 (the small x component of w)
        assert err < 2e-6, f"parameter {i}: AD vs FD {err:.2e}"


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_full_construction_under_jit(_full_pair):
    # The whole constructor traces: jax.jit of (GM -> model) runs, and agrees
    # with the eager construction (measured 1.9e-11). Under jit every stage is
    # staged out, including the impact transform and both tracks.
    _, bk = _full_pair
    GM0 = _full_kwargs()["GM"]

    def model(GM):
        s = _full_build("jax", GM=GM)
        out = _full_outputs(s)
        return jnp.concatenate([jnp.ravel(v) for v in out.values()])

    got = numpy.asarray(jax.jit(model)(jnp.asarray(GM0)))
    want = numpy.concatenate(
        [numpy.ravel(as_numpy(v)) for v in _full_outputs(bk).values()]
    )
    assert numpy.max(numpy.abs(got - want)) < 1e-9 * numpy.max(numpy.abs(want))


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_impact_arm_is_structural_under_a_trace():
    # The arm of the impact is the sign of the impact angle; traced, there is no
    # concrete sign and the modelled arm is used (a mismatch raises eagerly).
    from galpy.df.streamgapdf import _impact_is_leading

    assert _impact_is_leading(-2.34, True) is False
    assert _impact_is_leading(jnp.asarray(2.34), False) is True
    for leading in (True, False):
        out = jax.jit(lambda a: a * float(_impact_is_leading(a, leading)))(-2.34)
        assert float(out) == (-2.34 if leading else 0.0)


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_nTrackChunksImpact_must_be_given_when_traced():
    # A structural integer: the default is read off the impact's angle range,
    # which has no concrete value under a trace.
    from types import SimpleNamespace

    from galpy.df.streamgapdf import streamgapdf

    seen = {}

    def probe(dati):
        m = SimpleNamespace(
            _gap_progenitor_setup=lambda: None,
            _leading=False,
            _progenitor_Omega_along_dOmega=-0.5,
            _sigMeanSign=1.0,
            _deltaAngleTrackImpact=dati,
        )
        try:
            streamgapdf._determine_impact_coordtransform(m, dati, None, 1.0, -2.0)
        except ValueError as e:
            seen["msg"] = str(e)
        return dati

    jax.jit(probe)(1.3)
    assert "nTrackChunksImpact" in seen.get("msg", "")


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
@pytest.mark.parametrize("dati", [0.3, 1.3, 2.97])
def test_nTrackChunksImpact_default_from_concrete_backend_range(dati):
    # A CONCRETE backend angle range gives the numpy default, floor(r/0.15)+1
    # (at least 4), so an eager backend construction picks the same chunks.
    from galpy.df.streamgapdf import streamgapdf

    class _Stop(Exception):
        pass

    class _Mock:
        _leading = False
        _progenitor_Omega_along_dOmega = -0.5
        _sigMeanSign = 1.0

        def _gap_progenitor_setup(self):
            pass

        @property
        def _gap_progenitor(self):  # read right after the chunk count is set
            raise _Stop

    m = _Mock()
    m._deltaAngleTrackImpact = jnp.asarray(dati)
    with pytest.raises(_Stop):
        streamgapdf._determine_impact_coordtransform(m, dati, None, 1.0, -2.0)
    assert m._nTrackChunksImpact == max(int(numpy.floor(dati / 0.15)) + 1, 4)


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_backend_impact_angle_value(_gapdf_kick):
    # A backend impact angle gives the same kick as the float
    import copy

    sdf = copy.deepcopy(_gapdf_kick[0])
    ref = numpy.asarray(_chain_kick(sdf, "impact_angle", -2.34))
    with use("jax", force=True):
        got = _chain_kick(sdf, "impact_angle", jnp.asarray(-2.34))
    assert is_backend_array(sdf._impact_angle)
    got = numpy.asarray(as_numpy(got))
    if jax.default_backend() == "gpu":
        # the kick is dOap = Oap - track, ~1e-3 from O(1) frequencies/angles: the
        # GPU's last-ulp differences in Oap survive the cancellation (jax CPU is
        # bit-identical to numpy). Measured (TITAN Xp, H100) <= 2 ulp of the
        # column's largest |track|; 5e-11 relative at worst
        track = numpy.abs(numpy.asarray(as_numpy(sdf._kick_interpolatedObsTrackAA)))
        tol = 1e-12 * numpy.abs(ref) + 2.0 * numpy.spacing(track.max(axis=0))
        assert numpy.all(numpy.abs(got - ref) <= tol), numpy.abs(got - ref).max()
    else:
        numpy.testing.assert_allclose(got, ref, rtol=1e-12)


def _impact_tail(sdf, timpact, q=None):
    """Re-run the constructor from the impact on a copy of a built object.

    timpact first enters at the impact stage, so for it this IS the full model
    (bar the unperturbed track, which does not depend on it). For q it is the
    model with the unperturbed setup (progenitor frequencies, frequency
    covariance) held fixed: the impact-time gap track, its action-angle
    Jacobians (second derivatives of isochroneApprox) and the kicks."""
    import copy

    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import LogarithmicHaloPotential

    s = copy.copy(sdf)
    for cached in ("_kick_interpolatedThetasTrack", "_kick_interpolatedObsTrackAA"):
        s.__dict__.pop(cached, None)  # else the kick stage returns the cached ones
    if q is not None:
        s._pot = LogarithmicHaloPotential(normalize=1.0, q=q)
        s._aA = actionAngleIsochroneApprox(
            pot=s._pot, b=0.8, tintJ=20.0, ntintJ=1000, integrate_method="diffrax"
        )
    kw = _full_kwargs()
    with use("jax", force=True):
        s._determine_deltaAngleTrackImpact(None, timpact)
        s._determine_impact_coordtransform(
            s._deltaAngleTrackImpact,
            kw["nTrackChunksImpact"],
            timpact,
            kw["impact_angle"],
        )
        s._determine_deltav_kick(
            kw["impact_angle"],
            kw["impactb"],
            kw["subhalovel"],
            kw["GM"],
            kw["rs"],
            None,
            3,
            False,
        )
        s._determine_deltaOmegaTheta_kick(3)
        d = jnp.asarray(_FULL_DANGLES)
        return {
            "density": s._density_par(d),
            "meanOmega": s.meanOmega(d, oned=True, use_physical=False),
            "kick_dOap": s._kick_dOap,
            "gap_alljacsTrack": s._gap_alljacsTrack,
            "gap_ObsTrack": s._gap_ObsTrack,
        }


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
@pytest.mark.parametrize("param", ["timpact", "q"])
def test_impact_grad_timpact_q_vs_fd(_full_pair, param):
    # d/d(timpact) and d/dq through the impact stage vs a central FD of re-runs:
    # second derivatives of isochroneApprox (the gap track's AA Jacobians), which
    # the arccos/arcsin isochrone angles (before #1654) got wrong by up to 1.6e-4.
    # Measured vs Richardson-extrapolated FD: <= 5e-9 for every output; this
    # single step is good to ~2e-7.
    _, bk = _full_pair
    t0 = _full_kwargs()["timpact"]
    if param == "timpact":
        x0 = t0

        def f(x):
            return _impact_tail(bk, x)
    else:
        x0 = 0.9

        def f(x):
            return _impact_tail(bk, t0, q=x)

    ad = jax.jacfwd(f)(x0)
    h = 2e-5 * x0
    plus, minus = f(jnp.asarray(x0 + h)), f(jnp.asarray(x0 - h))
    for k in ad:
        fd = (numpy.asarray(plus[k]) - numpy.asarray(minus[k])) / (2.0 * h)
        scale = numpy.max(numpy.abs(fd))
        err = numpy.max(numpy.abs(numpy.asarray(ad[k]) - fd)) / scale
        assert err < 1e-6, f"d({k})/d({param}): AD vs FD {err:.2e}"
