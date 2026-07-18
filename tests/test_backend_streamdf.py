###############################################################################
# test_backend_streamdf.py: multi-backend tests for streamdf's DF-evaluation
# methods (Phase A.1: the closed-form distributions + frequency moments that
# operate on an ASSEMBLED track).
#
# The stream track itself is assembled with numpy (Phase B); these methods
# evaluate the stream DF at given parallel-angle / frequency offsets using the
# precomputed track scalars (self._meandO, self._sortedSigOEig, ...). Migrated to
# the galpy.backend namespace layer: a jax/torch input routes to native
# erf/exp/sqrt (so d(DF)/d(offset) flows and jits), the numpy path keeps
# scipy.special (byte-identical). pOparapar / ptdAngle's numpy in-place masked
# write becomes xp.where (jit/grad-safe; the t=0 dO->inf dead branch is guarded).
#
# Proves per method: (a) backend value parity vs the numpy path (which is
# byte-identical -- the else-branch is the verbatim original), and (b) grad-vs-FD
# h-converges (stringent, not finite-and-nonzero). Backends not installed self-skip.
###############################################################################
import numpy
import pytest

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

from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.backend import as_numpy, get_namespace
from galpy.backend.jacobian import jacobian
from galpy.df.streamdf import (
    _determine_stream_spread_single,
    _determine_stream_track_single,
    _real_eig,
    _vmap_track_chunks,
    calcaAJac,
)
from galpy.orbit import Orbit
from galpy.potential import IsochronePotential, LogarithmicHaloPotential
from galpy.util import conversion


@pytest.fixture(scope="module")
def sdf():
    # The canonical Bovy (2014) GD-1-like stream (as in test_streamdf.py). Assembling
    # the track is slow, so build once per module.
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    return streamdf_ctor(lp, aAI, obs)


def streamdf_ctor(lp, aAI, obs):
    from galpy.df import streamdf

    return streamdf(
        0.365 / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
    )


def _arr(backend_name, x):
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


# scalar-valued (or first-element) DF evaluations at a parallel angle, and a
# stripping-time p(t|a) that is array-valued.
_DANGLE = 0.5
_METHODS = [
    ("density_par", lambda s, d: s._density_par(d), False),
    ("meanOmega1D", lambda s, d: s.meanOmega(d, oned=True, use_physical=False), False),
    ("sigOmega", lambda s, d: s.sigOmega(d, use_physical=False), False),
]


@pytest.mark.parametrize("name,fn,_arrarg", _METHODS, ids=[m[0] for m in _METHODS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_df_eval_value_parity(sdf, backend_name, name, fn, _arrarg):
    ref = float(fn(sdf, _DANGLE))
    got = float(fn(sdf, _arr(backend_name, _DANGLE)))
    numpy.testing.assert_allclose(
        got, ref, rtol=1e-11, atol=1e-13, err_msg=f"{name} {backend_name}"
    )


@pytest.mark.parametrize("name,fn,_arrarg", _METHODS, ids=[m[0] for m in _METHODS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_df_eval_grad_vs_fd(sdf, backend_name, name, fn, _arrarg):
    # d(DF)/d(dangle) AD must h-converge to a central FD of the numpy path.
    if backend_name == "jax":
        ad = float(jax.grad(lambda d: fn(sdf, d))(jnp.asarray(_DANGLE)))
    else:
        dt = torch.tensor(_DANGLE, dtype=torch.float64, requires_grad=True)
        fn(sdf, dt).backward()
        ad = float(dt.grad)
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (float(fn(sdf, _DANGLE + h)) - float(fn(sdf, _DANGLE - h))) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-5 * abs(ad) + 1e-7, (
        f"{name} {backend_name} grad-vs-FD best={best:.2e}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_pOparapar_where_and_grad(sdf, backend_name):
    # pOparapar's masked Gaussian: backend value parity + d/d(Opar) h-converges.
    Op0, ap0 = float(sdf._meandO) * 1.05, 0.3
    ref = float(numpy.atleast_1d(sdf.pOparapar(Op0, ap0))[0])
    got = float(
        numpy.atleast_1d(
            sdf.pOparapar(_arr(backend_name, Op0), _arr(backend_name, ap0))
        )[0]
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda O: sdf.pOparapar(O, jnp.asarray(ap0)).sum())(
                jnp.asarray(Op0)
            )
        )
    else:
        opar_t = torch.tensor(Op0, dtype=torch.float64, requires_grad=True)
        sdf.pOparapar(opar_t, torch.tensor(ap0, dtype=torch.float64)).sum().backward()
        ad = float(opar_t.grad)
    h = 1e-5
    fd = (
        float(sdf.pOparapar(Op0 + h, ap0).sum())
        - float(sdf.pOparapar(Op0 - h, ap0).sum())
    ) / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd) + 1e-6, (
        f"pOparapar grad {backend_name}: {ad} vs {fd}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_ptdAngle_where_and_grad(sdf, backend_name):
    # ptdAngle's masked p(t|a): backend value parity (incl. the t<=0 / t>=tdisrupt zero
    # region) + d/d(t) h-converges, with the dO=dangle/t dead branch guarded.
    ts = numpy.array([0.5, 1.5, 2.5]) * sdf._tdisrupt / 3.0
    ref = numpy.asarray(sdf.ptdAngle(ts, _DANGLE))
    got = numpy.asarray(
        sdf.ptdAngle(_arr(backend_name, ts), _arr(backend_name, _DANGLE))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)
    t0 = float(ts[1])
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda t: sdf.ptdAngle(t, jnp.asarray(_DANGLE)))(jnp.asarray(t0))
        )
    else:
        tt = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
        sdf.ptdAngle(tt, torch.tensor(_DANGLE, dtype=torch.float64)).backward()
        ad = float(tt.grad)
    h = 1e-5
    fd = (
        float(sdf.ptdAngle(t0 + h, _DANGLE)) - float(sdf.ptdAngle(t0 - h, _DANGLE))
    ) / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd) + 1e-6, (
        f"ptdAngle grad {backend_name}: {ad} vs {fd}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_meanOmega_3d_value_parity_and_grad(sdf, backend_name):
    # meanOmega's default (oned=False) returns the full 3-vector mean frequency
    # (progenitor_Omega + dO1D * dsigomeanProgDirection * sign); meanOmega1D only
    # exercises the oned=True return, so cover the 3-vector backend path here.
    ref = numpy.asarray(sdf.meanOmega(_DANGLE, use_physical=False))
    got = numpy.asarray(sdf.meanOmega(_arr(backend_name, _DANGLE), use_physical=False))
    assert ref.shape == (3,)
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)

    # grad of a quadratic loss on the 3-vector AD must h-converge to a central FD
    # of the numpy path (stringent, not finite-and-nonzero).
    def loss(d):
        return (sdf.meanOmega(d, use_physical=False) ** 2.0).sum()

    if backend_name == "jax":
        ad = float(jax.grad(loss)(jnp.asarray(_DANGLE)))
    else:
        dt = torch.tensor(_DANGLE, dtype=torch.float64, requires_grad=True)
        loss(dt).backward()
        ad = float(dt.grad)
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (float(loss(_DANGLE + h)) - float(loss(_DANGLE - h))) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-5 * abs(ad) + 1e-8, (
        f"meanOmega3D grad {backend_name}: best={best:.2e}"
    )


###############################################################################
# Phase A.2: the stripping-time moments meantdAngle / sigtdAngle.
#
# Both integrate the (Phase-A.1) p(t|dangle) over [Tlow, Thigh]. The numpy path
# keeps scipy's adaptive quad (byte-identical -- the else-branch is verbatim); a
# jax/torch dangle routes to in-backend fixed-order Gauss-Legendre (galpy.backend
# .quadrature.quad) with the upper limit clamped at tdisrupt, where p(t|dangle)
# jumps to 0 -- so the GL interval stays smooth and converges fast. The denom==0
# progenitor/far-field control flow becomes xp.where and num/denom (and the
# sqrt(var) in sigtdAngle) are dead-branch guarded so AD stays finite.
#
# Value parity floors at ~1e-6 in the clamped far field: that residual is scipy's
# OWN adaptive error at the t=tdisrupt jump (the clamped-GL integrand is smooth
# and more accurate), so this is the expected adaptive-vs-GL floor, not a backend
# error. Below dangle~0.43 (Thigh < tdisrupt, no clamp) parity is ~1e-15.
###############################################################################
_TDMOMENTS = [
    ("meantdAngle", lambda s, d: s.meantdAngle(d, use_physical=False)),
    ("sigtdAngle", lambda s, d: s.sigtdAngle(d, use_physical=False)),
]
# 0.3 = unclamped (Thigh < tdisrupt, GL ~machine-precise); 0.8 = clamped far field.
_TDMOMENT_DANGLES = [0.3, 0.8]


@pytest.mark.parametrize("dangle", _TDMOMENT_DANGLES)
@pytest.mark.parametrize("name,fn", _TDMOMENTS, ids=[m[0] for m in _TDMOMENTS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdmoment_value_parity(sdf, backend_name, name, fn, dangle):
    ref = float(fn(sdf, dangle))
    got = float(fn(sdf, _arr(backend_name, dangle)))
    # rtol 1e-5: fixed-GL vs scipy adaptive; the ~1e-6 clamped-field floor is
    # scipy's error at the t=tdisrupt jump, not the backend's.
    numpy.testing.assert_allclose(
        got, ref, rtol=1e-5, atol=1e-8, err_msg=f"{name} {backend_name} dangle={dangle}"
    )


@pytest.mark.parametrize("dangle", _TDMOMENT_DANGLES)
@pytest.mark.parametrize("name,fn", _TDMOMENTS, ids=[m[0] for m in _TDMOMENTS])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdmoment_grad_vs_fd(sdf, backend_name, name, fn, dangle):
    # d(moment)/d(dangle) AD must h-converge to a central FD of the SAME backend
    # (GL) path. The forward values feeding the FD are value-parity-validated
    # above, so this is a stringent check of the gradient (FD-of-scipy is noisier
    # than the AD near the jump, so it is not the reference here).
    if backend_name == "jax":
        ad = float(jax.grad(lambda d: fn(sdf, d))(jnp.asarray(dangle)))

        def fdfun(d):
            return float(fn(sdf, jnp.asarray(d)))
    else:
        dt = torch.tensor(dangle, dtype=torch.float64, requires_grad=True)
        fn(sdf, dt).backward()
        ad = float(dt.grad)

        def fdfun(d):
            return float(fn(sdf, torch.tensor(d, dtype=torch.float64)))

    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (fdfun(dangle + h) - fdfun(dangle - h)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-5 * abs(ad) + 1e-6, (
        f"{name} {backend_name} dangle={dangle} grad-vs-FD best={best:.2e}"
    )


###############################################################################
# Phase A.3: the nested-quad perpendicular-angle moments (pangledAngle,
# meanangledAngle, sigangledAngle) + the _pangledAnglet leaf.
#
# meanangledAngle/sigangledAngle are ratios of 2-D integrals (outer over
# angleperp, inner over t via the batched pangledAngle). numpy keeps scipy's
# adaptive quad (byte-identical); a jax/torch dangle routes to nested in-backend
# GL (fixed_quad), so d(moment)/d(dangle) flows and jits. meanangledAngle is 0 by
# odd symmetry (x*pangledAngle over [aplow,-aplow]) so it gets a reference-match
# to that analytic zero; the meaningful grad target is sigangledAngle (even x^2).
# Value parity is ~1e-13 (the inner [0,tdisrupt] integrand is smooth, no
# straddled jump); grad-vs-FD h-converges to ~1e-13.
###############################################################################
_ANGLED_DANGLES = [0.3, 0.6, 1.0]


@pytest.mark.parametrize("dangle", _ANGLED_DANGLES)
@pytest.mark.parametrize("simple", [False, True], ids=["full", "simple"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sigangled_value_parity(sdf, backend_name, simple, dangle):
    ref = float(sdf.sigangledAngle(dangle, simple=simple, use_physical=False))
    got = float(
        sdf.sigangledAngle(
            _arr(backend_name, dangle), simple=simple, use_physical=False
        )
    )
    # full nested GL matches scipy ~1e-13; the simple estimate rides on the A.2
    # meantdAngle (clamped-GL floor ~1e-6), so allow a looser rtol there.
    rtol = 1e-5 if simple else 1e-9
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=1e-10, err_msg=f"sigangled simple={simple} d={dangle}"
    )


@pytest.mark.parametrize("dangle", _ANGLED_DANGLES)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sigangled_grad_vs_fd(sdf, backend_name, dangle):
    def fn(s, d):
        return s.sigangledAngle(d, use_physical=False)

    if backend_name == "jax":
        ad = float(jax.grad(lambda d: fn(sdf, d))(jnp.asarray(dangle)))

        def fdfun(d):
            return float(fn(sdf, jnp.asarray(d)))
    else:
        dt = torch.tensor(dangle, dtype=torch.float64, requires_grad=True)
        fn(sdf, dt).backward()
        ad = float(dt.grad)

        def fdfun(d):
            return float(fn(sdf, torch.tensor(d, dtype=torch.float64)))

    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (fdfun(dangle + h) - fdfun(dangle - h)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-5 * abs(ad) + 1e-8, (
        f"sigangled grad {backend_name} d={dangle} best={best:.2e}"
    )


@pytest.mark.parametrize("dangle", _ANGLED_DANGLES)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_meanangled_is_zero(sdf, backend_name, dangle):
    # meanangledAngle == 0 by odd-integrand symmetry (numpy returns exactly 0);
    # the backend must match that analytic zero (not merely be finite).
    got = float(sdf.meanangledAngle(_arr(backend_name, dangle), use_physical=False))
    assert abs(got) < 1e-12, f"meanangled {backend_name} d={dangle} = {got}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_pangledAngle_array_parity_and_grad(sdf, backend_name):
    # p(angle_perp|dangle) over an array of angleperp: batched-inner-quad value
    # parity + d(sum)/d(dangle) h-converges.
    ap = numpy.array([0.0, 0.01, -0.01, 0.02])
    ref = numpy.asarray(sdf.pangledAngle(ap, 0.6))
    got = numpy.asarray(
        sdf.pangledAngle(_arr(backend_name, ap), _arr(backend_name, 0.6))
    )
    assert got.shape == ref.shape
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda d: sdf.pangledAngle(jnp.asarray(ap), d).sum())(
                jnp.asarray(0.6)
            )
        )
    else:
        dt = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        sdf.pangledAngle(torch.tensor(ap, dtype=torch.float64), dt).sum().backward()
        ad = float(dt.grad)
    h = 1e-5
    fd = (
        float(sdf.pangledAngle(ap, 0.6 + h).sum())
        - float(sdf.pangledAngle(ap, 0.6 - h).sum())
    ) / (2 * h)
    assert abs(ad - fd) < 1e-4 * abs(fd) + 1e-8, (
        f"pangledAngle grad {backend_name}: {ad} vs {fd}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sigangled_assumezeromean_false(sdf, backend_name):
    # assumeZeroMean=False computes the nummean integral explicitly (the odd
    # integrand ~0 by symmetry); covers that backend branch + matches numpy.
    d = 0.6
    ref = float(sdf.sigangledAngle(d, assumeZeroMean=False, use_physical=False))
    got = float(
        sdf.sigangledAngle(
            _arr(backend_name, d), assumeZeroMean=False, use_physical=False
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10)


###############################################################################
# Phase B.1: calcaAJac -- the AA-map Jacobian d(J,Omega,theta)/d(x,v), the
# foundation of the stream-track spine.
#
# numpy: verbatim finite differences (byte-identical; dispatched by is_backend_
# array, the numpy branch is untouched). jax/torch: the EXACT Jacobian by
# autodiff of aA.actionsFreqsAngles (galpy.backend.jacobian -> jax.jacrev /
# torch.autograd.functional.jacobian). jacfwd / vectorize=True are unavailable
# because the C-STM orbit integrator is a custom_vjp (jax) / custom autograd
# Function (torch). AD matches the numpy FD to its truncation floor (~1e-5) and
# is ITSELF differentiable: d(Jacobian)/d(potential q) flows (higher-order AD --
# the point of Phase B, a differentiable track).
#
# The default C-STM integrator carries IC gradients (the Jacobian itself, tested
# fast below); the higher-order d(Jac)/d(q) needs the in-backend ODE integrator
# (diffrax/torchdiffeq), exercised with a reduced tintJ so the second-order solve
# stays affordable. Its FD reference is the EXACT C-STM Jacobian at q +/- h (no
# inner finite differences), so the outer central FD h-converges cleanly.
###############################################################################
_XV = numpy.array([1.0, 0.2, 0.9, 0.1, 0.05, 0.0])


@pytest.fixture(scope="module")
def aA_iso():
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    return actionAngleIsochroneApprox(pot=lp, b=0.8)


@pytest.mark.parametrize(
    "flags,shape",
    [
        (dict(actionsFreqsAngles=True), (9, 6)),  # used per-chunk by the track
        (dict(dOdJ=True), (3, 3)),  # dOmega/dJ used in streamdf.__init__
        (dict(freqs=True), (6, 6)),  # freqs+angles rows
        (dict(), (6, 6)),  # actions+angles rows
    ],
    ids=["actionsFreqsAngles", "dOdJ", "freqs", "actions"],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_calcaAJac_ad_equals_fd(aA_iso, backend_name, flags, shape):
    # Backend AD Jacobian == numpy FD Jacobian to the numpy FD truncation floor.
    fd = calcaAJac(_XV.copy(), aA_iso, **flags)
    ad = as_numpy(calcaAJac(_arr(backend_name, _XV), aA_iso, **flags))
    assert fd.shape == shape and tuple(ad.shape) == shape
    err = numpy.max(numpy.abs(ad - fd))
    assert err < 5e-4, f"{backend_name} {flags}: max|AD-FD|={err:.2e}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_calcaAJac_backend_rejects_lb_coordfunc(aA_iso, backend_name):
    # lb / coordFunc are unsupported on the backend path; they raise (not misbehave).
    for kw in (dict(lb=True), dict(coordFunc=lambda x: x)):
        with pytest.raises(NotImplementedError):
            calcaAJac(_arr(backend_name, _XV), aA_iso, actionsFreqsAngles=True, **kw)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_calcaAJac_backend_accepts_scalar_list(aA_iso, backend_name):
    # A list of per-coordinate backend scalars is stacked into the (6,) vector
    # (covers the non-(6,)-array input branch); result matches the array form.
    xv_list = [_arr(backend_name, float(v)) for v in _XV]
    ad_list = as_numpy(calcaAJac(xv_list, aA_iso, actionsFreqsAngles=True))
    ad_vec = as_numpy(
        calcaAJac(_arr(backend_name, _XV), aA_iso, actionsFreqsAngles=True)
    )
    numpy.testing.assert_allclose(ad_list, ad_vec, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_calcaAJac_higher_order_grad_vs_fd(backend_name):
    # d(sum(W*Jac))/d(potential q) flows (higher-order AD). FD reference = exact
    # C-STM Jacobian at q +/- h (no inner FD), so the central difference is clean.
    tintJ, ntintJ = 8.0, 800
    Wn = numpy.random.default_rng(0).standard_normal((9, 6))

    def cstm_loss(qval):
        lp = LogarithmicHaloPotential(normalize=1.0, q=qval)
        aAq = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=tintJ, ntintJ=ntintJ)
        J = as_numpy(calcaAJac(_arr(backend_name, _XV), aAq, actionsFreqsAngles=True))
        return float((Wn * J).sum())

    if backend_name == "jax":
        W = jnp.asarray(Wn)

        def loss(qq):
            lp = LogarithmicHaloPotential(normalize=1.0, q=qq)
            aAq = actionAngleIsochroneApprox(
                pot=lp,
                b=0.8,
                tintJ=tintJ,
                ntintJ=ntintJ,
                integrate_method="diffrax",
                integrate_kwargs={"adjoint": "direct", "max_steps": 20000},
            )
            return jnp.sum(
                W * calcaAJac(jnp.asarray(_XV), aAq, actionsFreqsAngles=True)
            )

        ad = float(jax.grad(loss)(jnp.asarray(0.9)))
    else:
        W = torch.tensor(Wn, dtype=torch.float64)
        q = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        aAq = actionAngleIsochroneApprox(
            pot=lp, b=0.8, tintJ=tintJ, ntintJ=ntintJ, integrate_method="torchdiffeq"
        )
        L = (
            W
            * calcaAJac(
                torch.tensor(_XV, dtype=torch.float64), aAq, actionsFreqsAngles=True
            )
        ).sum()
        ad = float(torch.autograd.grad(L, q)[0])

    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (cstm_loss(0.9 + h) - cstm_loss(0.9 - h)) / (2 * h))
        for h in (1e-3, 1e-4, 1e-5)
    )
    # clean C-STM FD floors ~3e-5 at h=1e-4; a wrong grad would miss by O(|ad|).
    assert best < 1e-4 * abs(ad) + 1e-4, (
        f"{backend_name} higher-order grad-vs-FD best={best:.2e}"
    )


@pytest.mark.skipif(not AD_BACKENDS, reason="needs a float64-capable AD backend")
def test_calcaAJac_numpy_stays_fd(aA_iso):
    # A plain-numpy xv keeps the historical finite-difference Jacobian (byte-identical
    # to numpy-only installs) even when jax/torch is available -- it must NOT silently
    # route to a backend AD path. A backend xv gets the exact AD Jacobian instead; the
    # two agree only at the coarse one-sided-FD-vs-AD floor (some Jacobian entries differ
    # ~1%), which is exactly why the differentiable track is worth having.
    fd = calcaAJac(_XV.copy(), aA_iso, actionsFreqsAngles=True)
    assert type(fd) is numpy.ndarray and tuple(fd.shape) == (9, 6)
    for backend_name in AD_BACKENDS:
        ad = as_numpy(
            calcaAJac(_arr(backend_name, _XV), aA_iso, actionsFreqsAngles=True)
        )
        numpy.testing.assert_allclose(ad, fd, rtol=2e-2, atol=1e-4)


###############################################################################
# Phase B.2: _determine_stream_track_single -- the per-chunk track workhorse.
#
# Dual-path (dispatched by is_backend_array on the per-chunk phase-space point):
# the numpy branch is the verbatim original (byte-identical); a backend orbit
# routes to _determine_stream_track_single_backend -- a PURE function (no numpy
# item-assignment: xp.stack/concat build every array) so it is differentiable and
# map-ready (Phase B.3, jax.lax.map). Returns a plain tuple of backend arrays (numpy keeps its
# dtype=object array); the boolean-mask angle wrap becomes xp.where, numpy.mod
# becomes xp.remainder, the actions+angles row select uses static indices.
###############################################################################


class _OrbAtT:
    # Stand-in exposing .vxvv[0] = the backend phase-space at trackt (what B.3's
    # backend-preserving per-chunk orbit-eval will feed the workhorse; today
    # Orbit.__call__ numpy-coerces the interpolated point).
    def __init__(self, xv):
        self.vxvv = xv[None, :]


def _track_args():
    d = numpy.array([1.0, 0.5, -0.3])
    dsig = d / numpy.linalg.norm(d)
    progenitor_angle = numpy.array([2.9, 0.5, 1.1])

    def meanOmega(x):
        return numpy.array([1.3, 0.85, 0.95]) * (1.0 + 0.01 * (x + 1.0))

    return (progenitor_angle, 1.0, dsig, meanOmega, 0.05)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_determine_stream_track_single_value_parity(aA_iso, backend_name):
    # Backend path == numpy path at the identical phase-space point (all 6 outputs).
    # The numpy path uses the finite-difference aA Jacobian and the backend path the
    # exact AD Jacobian, so out[1] (the freq/angle Jacobian) agrees only at the coarse
    # one-sided-FD floor (~5e-5 abs, ~100% rel on near-zero entries). Every other output
    # -- actions/freqs/angles, inverse Jacobian, ObsTrack, ObsTrackAA, detdOdJ -- matches
    # far below the 1e-5 integrator floor.
    args = _track_args()
    out_np = _determine_stream_track_single(
        aA_iso, lambda t: Orbit(_XV.copy()), 0.0, *args
    )
    xv_b = _arr(backend_name, _XV)
    out_b = _determine_stream_track_single(aA_iso, lambda t: _OrbAtT(xv_b), 0.0, *args)
    assert (
        isinstance(out_np, numpy.ndarray) and out_np.dtype == object
    )  # numpy unchanged
    assert isinstance(out_b, tuple) and len(out_b) == 6  # backend: plain tuple
    # Per-output atol at the honest FD-vs-AD floor: actions/freqs/angles (out[0]) and the
    # AA track (out[4]) are Jacobian-independent (~1e-12); the freq/angle Jacobian (out[1])
    # differs ~5e-5, its inverse (out[2]) ~2e-3 (matrix inversion amplifies the FD error),
    # which propagates ~1e-4 into ObsTrack (out[3]); detdOdJ (out[5]) stays ~1e-6. Tight
    # gradient correctness is covered separately by the mock-AA grad-vs-FD test.
    atols = [1e-9, 2e-4, 5e-3, 5e-4, 1e-9, 1e-5]
    for i in range(6):
        a = numpy.asarray(out_np[i], dtype=float)
        b = as_numpy(out_b[i]).astype(float)
        assert a.shape == b.shape
        numpy.testing.assert_allclose(
            b, a, rtol=2e-2, atol=atols[i], err_msg=f"out[{i}]"
        )


# A smooth NONLINEAR mock AA (jacrev/2nd-order AD well-defined everywhere): the
# real AAs cannot supply the 2nd derivative ObsTrack's gradient needs (isochrone
# Approx integrates its auxiliary orbit via the C-STM pure_callback = 1st-order
# only; the analytic isochrone AA's arccos/where angle jacrev NaN-poisons) -- both
# are AA-migration limits UPSTREAM of B.2. The mock isolates B.2's OWN assembly
# gradient (calcaAJac jacrev+inv, det, where-wrap, remainder, matmul, offset).
class _MockAA:
    def actionsFreqsAngles(self, R, vR, vT, z, vz, phi):
        xp = get_namespace(R)

        def r(e):
            return xp.reshape(e, (1,))

        return (
            r(R * R + 0.1 * vR * vR),
            r(R * vT),
            r(z * z + 0.1 * vz * vz + 0.5),
            r(1.0 + 0.2 * R * R + 0.1 * vT + 0.05 * z * vz),
            r(0.8 + 0.1 * R + 0.05 * vR * vR + 0.02 * R * vT),
            r(0.9 + 0.15 * z * z + 0.1 * vz + 0.03 * R),
            r(R + 0.3 * vR + phi + 0.1 * R * z + 0.05 * xp.sin(vT)),
            r(phi + 0.2 * vT + 0.1 * z + 0.05 * vR * vz),
            r(0.5 * z + 0.1 * vz + 0.2 + 0.05 * R + 0.02 * z * R),
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_determine_stream_track_single_grad_vs_fd(backend_name):
    # d(sum(W*ObsTrack))/d(R0) through a real in-backend (diffrax/torchdiffeq) orbit
    # + the backend assembly must h-converge to a central FD (stringent).
    ip = IsochronePotential(normalize=1.0, b=1.2)
    ts = numpy.linspace(0.0, 2.0, 41)
    ic = [1.0, 0.15, 0.85, 0.2, 0.18, 0.0]
    tt = 0.6
    args = _track_args()
    Wn = numpy.array([0.3, -0.7, 1.1, 0.5, -0.2, 0.9])
    method = "diffrax" if backend_name == "jax" else "torchdiffeq"

    def obstrack_np(R0):
        o = Orbit(_arr(backend_name, [R0] + ic[1:]))
        o.integrate(_arr(backend_name, ts), ip, method=method)
        xv = _arr(
            backend_name,
            [
                float(as_numpy(o.R(tt))),
                float(as_numpy(o.vR(tt))),
                float(as_numpy(o.vT(tt))),
                float(as_numpy(o.z(tt))),
                float(as_numpy(o.vz(tt))),
                float(as_numpy(o.phi(tt))),
            ],
        )
        out = _determine_stream_track_single(
            _MockAA(), lambda t: _OrbAtT(xv), tt, *args
        )
        return float(numpy.sum(Wn * as_numpy(out[3])))

    if backend_name == "jax":
        W = jnp.asarray(Wn)

        def loss(R0):
            o = Orbit(jnp.array([R0, ic[1], ic[2], ic[3], ic[4], ic[5]]))
            o.integrate(jnp.asarray(ts), ip, method="diffrax")
            xv = jnp.stack([o.R(tt), o.vR(tt), o.vT(tt), o.z(tt), o.vz(tt), o.phi(tt)])
            out = _determine_stream_track_single(
                _MockAA(), lambda t: _OrbAtT(xv), tt, *args
            )
            return jnp.sum(W * out[3])

        ad = float(jax.grad(loss)(jnp.asarray(1.0)))
    else:
        W = torch.tensor(Wn, dtype=torch.float64)
        R0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        o = Orbit(
            torch.stack([R0] + [torch.tensor(v, dtype=torch.float64) for v in ic[1:]])
        )
        o.integrate(torch.as_tensor(ts), ip, method="torchdiffeq")
        xv = torch.stack([o.R(tt), o.vR(tt), o.vT(tt), o.z(tt), o.vz(tt), o.phi(tt)])
        out = _determine_stream_track_single(
            _MockAA(), lambda t: _OrbAtT(xv), tt, *args
        )
        (W * out[3]).sum().backward()
        ad = float(R0.grad)

    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (obstrack_np(1.0 + h) - obstrack_np(1.0 - h)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-4 * abs(ad) + 1e-6, (
        f"{backend_name} ObsTrack grad-vs-FD best={best:.2e}"
    )


###############################################################################
# Phase B.3: _determine_stream_track -- the FULL stream track, backend-native.
#
# The numpy body (dispatched away when the progenitor is a backend orbit / the
# offsets are backend arrays) is BYTE-IDENTICAL -- verified by a git-stash A/B on
# a full numpy streamdf (_ObsTrack/_ObsTrackAA/_alljacsTrack/_allinvjacsTrack/
# _allAcfsTrack/_detdOdJps/_ObsTrackXY/_interpolatedObsTrackXY all bit-unchanged).
#
# The backend path (streamdf._determine_stream_track_backend) mirrors the numpy
# body but is PURE and jax.lax.map-mapped over the chunk grid -- FORK-FREE (no
# parallel_map), no numpy.empty/item-assignment. (NOT jax.vmap: vmap batches the
# per-chunk calcaAJac jax.jacrev -- no vmap batching rule over the diffrax/C-STM
# integration -- and silently leaks the outer d/d(param) gradient ~20%; lax.map
# runs the chunks sequentially so the analytic AD equals a finite-difference of
# the same track to ~1e-4.) The auxiliary orbit integrates on the backend
# (diffrax/torchdiffeq) and the per-chunk AA Jacobian is the exact AD calcaAJac,
# so the track is differentiable to the potential parameters (via the auxiliary
# integration + the AA) and the progenitor IC (via _ic_backend); the progenitor's
# own freqs/angles are recomputed from the backend progenitor so the offset
# (track AA - progenitor AA) carries the physical d(offset)/dparam cancellation.
# Full differentiability needs an integrate_method='diffrax'/'torchdiffeq' AA (the
# track uses the AA's 2nd derivative); the frequency-covariance moments
# (_meandO/_sortedSigOEig/_dsigomeanProgDirection -- an eigendecomposition) stay
# constant here (a ~10% residual vs a full numpy-rebuild FD), a Phase-C follow-up.
#
# These tests build a NUMPY streamdf, then re-run the track on the backend with a
# diffrax AA and check value parity + the differentiable pipeline. Slow (a backend
# track integrates many orbits under AD); jax only (torch: torchdiffeq + a
# list-comp fallback for the chunk loop -- lax.map is the jax priority).
###############################################################################
_STREAM_IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]


def _tdisrupt():
    return 4.5 / conversion.time_in_Gyr(220.0, 8.0)


def _build_numpy_sdf(
    qval, nTrackChunks, tintJ, deltaAngleTrack=None, nTrackIterations=0
):
    from galpy.df import streamdf as _streamdf

    lp = LogarithmicHaloPotential(normalize=1.0, q=qval)
    aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=tintJ)
    obs = Orbit(numpy.array(_STREAM_IC))
    return _streamdf(
        0.365 / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aA,
        leading=True,
        nTrackChunks=nTrackChunks,
        nTrackIterations=nTrackIterations,
        deltaAngleTrack=deltaAngleTrack,
        tdisrupt=_tdisrupt(),
        nospreadsetup=True,
        useInterp=False,
        interpTrack=False,
    )


def _run_backend_track(sdf, integrate_kwargs):
    # Swap in a diffrax AA + a backend progenitor and re-run _determine_stream_track
    # (dispatched to the backend path). Mutates sdf in place.
    aA = actionAngleIsochroneApprox(
        pot=sdf._pot,
        b=0.8,
        tintJ=sdf._aA._tintJ,
        integrate_method="diffrax",
        integrate_kwargs=integrate_kwargs,
    )
    prog_vxvv = numpy.asarray(sdf._progenitor.vxvv[0], dtype=float)
    progb = Orbit(jnp.asarray(prog_vxvv))
    progb.turn_physical_off()
    sdf._aA = aA
    sdf._progenitor = progb
    sdf._determine_stream_track(sdf._nTrackChunks)


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_determine_stream_track_value_parity():
    # The backend track (diffrax AA, exact AD Jacobian, jax.lax.map) reproduces the numpy
    # track (C-STM AA, finite-difference Jacobian). The physical track (_ObsTrack/_ObsTrackXY),
    # actions/angles and detdOdJ match far below the 1e-5 integrator floor; only the stored
    # freq/angle Jacobian _alljacsTrack (~5e-5) and its inverse _allinvjacsTrack (~9e-3, matrix
    # inversion amplifies the FD error) agree at the coarser one-sided-FD-vs-AD floor, since
    # numpy keeps FD there while the backend uses exact AD.
    # nTrackIterations=1 also exercises the backend refinement loop (both the numpy
    # reference and the backend re-run refine once, so parity is unaffected).
    sdf = _build_numpy_sdf(0.9, nTrackChunks=4, tintJ=15, nTrackIterations=1)
    names = (
        "_ObsTrack",
        "_ObsTrackAA",
        "_alljacsTrack",
        "_allinvjacsTrack",
        "_allAcfsTrack",
        "_detdOdJps",
        "_ObsTrackXY",
    )
    ref = {a: numpy.array(getattr(sdf, a), dtype=float) for a in names}
    _run_backend_track(sdf, {"max_steps": 1000000})
    assert get_namespace(sdf._ObsTrack) is not numpy  # stored as backend arrays
    # FD-vs-AD floor only for the stored Jacobian and its (inversion-amplified) inverse;
    # every physical/AA quantity stays at the tight integrator floor.
    fd_floor = {"_alljacsTrack": 1e-4, "_allinvjacsTrack": 2e-2}
    for a in names:
        got = as_numpy(getattr(sdf, a)).astype(float)
        err = numpy.max(numpy.abs(got - ref[a]))
        tol = fd_floor.get(a, 1e-5)
        assert err < tol, f"{a}: max|backend-numpy|={err:.2e}"


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_determine_stream_track_map_no_fork(monkeypatch):
    # The backend chunk loop is jax.lax.map (fork-free): parallel_map must NOT be
    # used. NOT jax.vmap -- vmap batches the per-chunk calcaAJac jax.jacrev (no vmap
    # batching rule over the diffrax/C-STM integration) and silently leaks the outer
    # d/d(param) gradient; lax.map runs the chunks sequentially so the AD is exact.
    sdf = _build_numpy_sdf(0.9, nTrackChunks=4, tintJ=15)
    from galpy.util import multi as _multi

    calls = {"laxmap": 0}
    orig = jax.lax.map

    def spy(*a, **k):
        calls["laxmap"] += 1
        return orig(*a, **k)

    def no_fork(*a, **k):
        raise AssertionError("backend stream track must not fork (parallel_map)")

    monkeypatch.setattr(jax.lax, "map", spy)
    monkeypatch.setattr(_multi, "parallel_map", no_fork)
    _run_backend_track(sdf, {"max_steps": 1000000})
    assert calls["laxmap"] > 0


def _track_loss_fn(sdf, prog_vxvv, tintJ, W):
    # Return loss(param) closures: q (potential) and R0 (progenitor IC), each with a
    # `direct` flag selecting the AA adjoint -- 'direct' (twice-diff, for jax.grad)
    # or the default recursive (smooth adaptive forward, for the finite difference).
    def _aA(lp, direct):
        ikw = (
            {"adjoint": "direct", "max_steps": 20000}
            if direct
            else {"max_steps": 200000}
        )
        return actionAngleIsochroneApprox(
            pot=lp, b=0.8, tintJ=tintJ, integrate_method="diffrax", integrate_kwargs=ikw
        )

    def loss_q(qq, direct):
        lp = LogarithmicHaloPotential(normalize=1.0, q=qq)
        sdf._pot = lp
        sdf._aA = _aA(lp, direct)
        progb = Orbit(jnp.stack([jnp.asarray(v) for v in prog_vxvv]))
        progb.turn_physical_off()
        sdf._progenitor = progb
        sdf._determine_stream_track_backend()
        return jnp.sum(W * sdf._ObsTrack)

    def loss_ic(r0, direct):
        lp = sdf._pot
        sdf._aA = _aA(lp, direct)
        progb = Orbit(jnp.stack([r0] + [jnp.asarray(v) for v in prog_vxvv[1:]]))
        progb.turn_physical_off()
        sdf._progenitor = progb
        sdf._determine_stream_track_backend()
        return jnp.sum(W * sdf._ObsTrack)

    return loss_q, loss_ic


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_determine_stream_track_differentiable_potential():
    # d(sum(W*ObsTrack))/d(LogHalo q) flows end-to-end through the backend track
    # (auxiliary diffrax integration + AD calcaAJac + lax.map assembly). The analytic
    # AD MUST equal a central FD of the SAME backend track -- here to ~1e-3 (the
    # stringent grad-vs-FD bar; a vmap chunk loop instead leaks ~20%). [The separate
    # ~10% gap to a full numpy-rebuild FD is the frozen frequency-covariance
    # eigendecomposition (_meandO/_dsigomeanProgDirection), a differentiable-__init__
    # (Phase C) follow-up -- not tested here.]
    nch, tintJ, delta, q0 = 3, 8, 0.5, 0.9
    Wn = numpy.random.default_rng(1).standard_normal((nch, 6))
    W = jnp.asarray(Wn)
    sdf = _build_numpy_sdf(q0, nTrackChunks=nch, tintJ=tintJ, deltaAngleTrack=delta)
    prog_vxvv = numpy.asarray(sdf._progenitor.vxvv[0], dtype=float)
    loss_q, _ = _track_loss_fn(sdf, prog_vxvv, tintJ, W)

    ad = float(jax.grad(lambda q: loss_q(q, True))(jnp.asarray(q0)))
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(
            ad
            - (
                float(loss_q(jnp.asarray(q0 + h), False))
                - float(loss_q(jnp.asarray(q0 - h), False))
            )
            / (2 * h)
        )
        for h in (1e-3, 1e-4)
    )
    assert best < 3e-3 * abs(ad), f"q AD={ad:.6e} best|AD-ownFD|={best:.2e}"


@pytest.mark.slow
@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_determine_stream_track_differentiable_progenitor():
    # d(sum(W*ObsTrack))/d(progenitor R) flows through the backend track (the IC
    # enters via _ic_backend -> the auxiliary integration + AD calcaAJac + lax.map
    # assembly); the analytic AD MUST equal a central FD of the SAME backend track,
    # here to ~1e-3 (same stringent grad-vs-FD bar as the potential gradient).
    nch, tintJ, delta = 3, 8, 0.5
    r0 = _STREAM_IC[0]
    Wn = numpy.random.default_rng(2).standard_normal((nch, 6))
    W = jnp.asarray(Wn)
    sdf = _build_numpy_sdf(0.9, nTrackChunks=nch, tintJ=tintJ, deltaAngleTrack=delta)
    prog_vxvv = numpy.asarray(sdf._progenitor.vxvv[0], dtype=float)
    _, loss_ic = _track_loss_fn(sdf, prog_vxvv, tintJ, W)

    ad = float(jax.grad(lambda r: loss_ic(r, True))(jnp.asarray(r0)))
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(
            ad
            - (
                float(loss_ic(jnp.asarray(r0 + h), False))
                - float(loss_ic(jnp.asarray(r0 - h), False))
            )
            / (2 * h)
        )
        for h in (1e-3, 1e-4)
    )
    assert best < 3e-3 * abs(ad), f"R0 AD={ad:.6e} best|AD-ownFD|={best:.2e}"


###############################################################################
# Coverage for the backend-dispatch branches calcaAJac / the jax track do not hit.
###############################################################################
@pytest.mark.skipif(not AD_BACKENDS, reason="needs a jax/torch backend")
def test_jacobian_infers_namespace_when_xp_none():
    # calcaAJac always passes xp= explicitly; the helper also infers the namespace
    # from x when xp is None (jacobian of v -> 2v is 2*I).
    be = AD_BACKENDS[0]
    x = _arr(be, numpy.array([0.3, -0.7, 1.1]))
    jac = as_numpy(jacobian(lambda v: 2.0 * v, x))
    numpy.testing.assert_allclose(jac, 2.0 * numpy.eye(3), atol=1e-12)


def test_jacobian_rejects_numpy_namespace():
    # numpy has no autodiff -> the helper raises rather than silently degrading.
    with pytest.raises(ValueError, match="jax or torch"):
        jacobian(lambda v: v, numpy.zeros(3), xp=numpy)


@pytest.mark.skipif("torch" not in BACKENDS, reason="needs torch")
def test_vmap_track_chunks_torch_stack():
    # The non-jax path of _vmap_track_chunks stacks a Python list of per-chunk
    # 6-tuples (jax uses lax.map; torch cannot trace torch.func.vmap over the
    # torchdiffeq custom-autograd orbit). Exercised with a trivial `single`.
    import torch

    xp = get_namespace(torch.zeros(1))
    xv0 = torch.arange(6.0).reshape(3, 2)  # 3 chunks
    thetas = torch.tensor([0.1, 0.2, 0.3])

    def single(xv0_i, theta_i):
        return tuple(xv0_i.sum() + theta_i + k for k in range(6))

    out = _vmap_track_chunks(xp, single, xv0, thetas)
    assert isinstance(out, tuple) and len(out) == 6
    for k in range(6):
        assert tuple(out[k].shape) == (3,)
        expected = numpy.array([float(xv0[i].sum() + thetas[i] + k) for i in range(3)])
        numpy.testing.assert_allclose(as_numpy(out[k]), expected, atol=1e-12)


###############################################################################
# Phase C: _determine_stream_spread_single -- the per-chunk covariance assembly.
# numpy path is byte-identical; a backend sigomatrixEig / invjac routes to a
# PURE/functional twin (item-assignment -> xp.where/xp.concat) so the 6x6 stream
# covariance differentiates w.r.t. the frequency covariance, the dispersions and
# the track Jacobian.
###############################################################################
def _spread_inputs():
    rng = numpy.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    sigo = A @ A.T + 3.0 * numpy.eye(3)  # SPD frequency covariance
    eig = _real_eig(sigo)  # (eigvals(3,), eigvecs(3,3))
    invjac = rng.standard_normal((6, 6))
    return eig, invjac


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_determine_stream_spread_single_value_parity(backend_name):
    eig, invjac = _spread_inputs()
    theta = 0.7
    sigOm = lambda t: 0.5 + 0.0 * t
    sigAn = lambda t: 0.3 + 0.0 * t
    f_np, l_np = _determine_stream_spread_single(
        eig, numpy.asarray(theta), sigOm, sigAn, invjac
    )
    eb = (_arr(backend_name, eig[0]), _arr(backend_name, eig[1]))
    f_b, l_b = _determine_stream_spread_single(
        eb, _arr(backend_name, theta), sigOm, sigAn, _arr(backend_name, invjac)
    )
    numpy.testing.assert_allclose(as_numpy(f_b), f_np, rtol=1e-11, atol=1e-13)
    numpy.testing.assert_allclose(as_numpy(l_b), l_np, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_determine_stream_spread_single_grad_vs_fd(backend_name):
    # d(sum full_cov + sum local_cov)/d(invjac[0,0]) AD h-converges to a central FD
    # of the numpy path (the covariance is now differentiable through the assembly).
    eig, invjac = _spread_inputs()
    theta = 0.7
    sigOm = lambda t: 0.5 + 0.0 * t
    sigAn = lambda t: 0.3 + 0.0 * t

    def loss_np(J):
        f, l = _determine_stream_spread_single(eig, theta, sigOm, sigAn, J)
        return float(f.sum() + l.sum())

    if backend_name == "jax":
        eb = (jnp.asarray(eig[0]), jnp.asarray(eig[1]))

        def loss(J):
            f, l = _determine_stream_spread_single(eb, theta, sigOm, sigAn, J)
            return f.sum() + l.sum()

        ad = float(jax.grad(loss)(jnp.asarray(invjac))[0, 0])
    else:
        Jt = torch.tensor(invjac, requires_grad=True)
        eb = (torch.tensor(eig[0]), torch.tensor(eig[1]))
        f, l = _determine_stream_spread_single(eb, theta, sigOm, sigAn, Jt)
        (f.sum() + l.sum()).backward()
        ad = float(Jt.grad[0, 0])
    best = float("inf")
    for h in (1e-4, 1e-5, 1e-6):
        Jp, Jm = invjac.copy(), invjac.copy()
        Jp[0, 0] += h
        Jm[0, 0] -= h
        fd = (loss_np(Jp) - loss_np(Jm)) / (2 * h)
        best = min(best, abs(ad - fd))
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, (
        f"spread grad-vs-FD {backend_name} best={best:.2e}"
    )


###############################################################################
# Phase C.2: _cart_and_interp_cov -- eigen-slerp interpolation of the 6x6 covs.
# numpy path byte-identical; backend twin: eigh (not eig) + functional sign-align
# + backend-native eigenvalue cubic spline + guarded slerp + reconstruction.
###############################################################################
from galpy.util import coords as _coords


def _mock_spread_sdf(nC, xp_mk):
    # Minimal object exposing the attributes _cart_and_interp_cov reads. Random
    # but reproducible ObsTrack + SPD chunk covariances + monotone theta grids.
    from galpy.df.streamdf import streamdf as _cls

    rng = numpy.random.default_rng(7)
    dtheta = 1.3
    thetas = numpy.linspace(0.0, dtheta, nC)
    interp = numpy.linspace(0.0, dtheta, 6 * nC)
    obs = rng.standard_normal((nC, 6)) * 0.3 + numpy.array(
        [1.1, 0.05, 1.0, 0.1, 0.02, 0.3]
    )
    covs = numpy.empty((nC, 6, 6))
    for ii in range(nC):
        A = rng.standard_normal((6, 6))
        covs[ii] = A @ A.T + 3.0 * numpy.eye(6)  # SPD

    class _M:
        pass

    m = _M()
    m._nTrackChunks = nC
    m._ObsTrack = xp_mk(obs)
    m._thetasTrack = thetas
    m._interpolatedThetasTrack = interp
    m._cart_and_interp_cov = _cls._cart_and_interp_cov.__get__(m)
    m._cart_and_interp_cov_backend = _cls._cart_and_interp_cov_backend.__get__(m)
    return m, covs


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cart_and_interp_cov_value_parity(backend_name):
    m_np, covs = _mock_spread_sdf(5, numpy.asarray)
    xy_np, interp_np = m_np._cart_and_interp_cov(covs)
    m_b, _ = _mock_spread_sdf(5, lambda a: _arr(backend_name, a))
    xy_b, interp_b = m_b._cart_and_interp_cov(_arr(backend_name, covs))
    # chunk-level Cartesian covariance == numpy to machine precision;
    # interpolated matches to the eigh-vs-eig + spline floor.
    numpy.testing.assert_allclose(as_numpy(xy_b), xy_np, rtol=1e-11, atol=1e-12)
    numpy.testing.assert_allclose(as_numpy(interp_b), interp_np, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cart_and_interp_cov_grad_vs_fd(backend_name):
    # d(sum allErrCovsXY)/d(chunk_covs[0,0,0]) AD h-converges to a central FD of numpy.
    m_np, covs = _mock_spread_sdf(5, numpy.asarray)

    def loss_np(c):
        return float(m_np._cart_and_interp_cov(c)[0].sum())

    if backend_name == "jax":
        m_b, _ = _mock_spread_sdf(5, jnp.asarray)

        def loss(c):
            return m_b._cart_and_interp_cov(c)[0].sum()

        ad = float(jax.grad(loss)(jnp.asarray(covs))[0, 0, 0])
    else:
        m_b, _ = _mock_spread_sdf(5, lambda a: torch.tensor(a))
        ct = torch.tensor(covs, requires_grad=True)
        m_b._cart_and_interp_cov(ct)[0].sum().backward()
        ad = float(ct.grad[0, 0, 0])
    best = float("inf")
    for h in (1e-4, 1e-5, 1e-6):
        cp, cm = covs.copy(), covs.copy()
        cp[0, 0, 0] += h
        cm[0, 0, 0] -= h
        best = min(best, abs(ad - (loss_np(cp) - loss_np(cm)) / (2 * h)))
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, (
        f"cart_and_interp grad {backend_name} {best:.2e}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cart_and_interp_cov_slerp_degeneracy_finite(backend_name):
    # Near-identical adjacent chunks drive the slerp Omega->0 (numpy NaNs there,
    # unguarded / sin(Omega)); the backend must stay finite in value AND gradient.
    m_b, covs = _mock_spread_sdf(5, lambda a: _arr(backend_name, a))
    covs[1] = covs[0]  # identical adjacent covariance -> parallel eigenvectors
    if backend_name == "jax":
        cb = jnp.asarray(covs)
        out = m_b._cart_and_interp_cov(cb)
        g = jax.grad(lambda c: m_b._cart_and_interp_cov(c)[1].sum())(cb)
        assert bool(jnp.all(jnp.isfinite(out[1]))) and bool(jnp.all(jnp.isfinite(g)))
    else:
        cb = torch.tensor(covs, requires_grad=True)
        out = m_b._cart_and_interp_cov(cb)
        out[1].sum().backward()
        assert bool(torch.all(torch.isfinite(out[1]))) and bool(
            torch.all(torch.isfinite(cb.grad))
        )


@pytest.mark.parametrize("nargs", [3, 6])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cyl_to_rect_jac_backend(backend_name, nargs):
    rng = numpy.random.default_rng(1)
    args = rng.standard_normal(nargs)
    ref = _coords.cyl_to_rect_jac(*[numpy.asarray(a) for a in args])
    got = as_numpy(_coords.cyl_to_rect_jac(*[_arr(backend_name, a) for a in args]))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
