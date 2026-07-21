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

from galpy.backend import as_numpy, is_backend_array
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
        _reset_kick_numpy(sdf, deltav_np)


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
