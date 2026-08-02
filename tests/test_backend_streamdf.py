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
from galpy.backend import as_numpy, get_namespace, is_backend_array
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
def test_calcaAJac_backend_lb_coordfunc_falls_back_to_numpy(aA_iso, backend_name):
    # lb / coordFunc have no backend implementation. They used to be unreachable
    # with a backend xv, so the guard raised; now that the coords chain is
    # backend-native the stream track reaches here with lb=True, so the call
    # lands on numpy and takes the finite-difference path -- exactly what it did
    # before. Assert it MATCHES numpy rather than merely not raising: the
    # fallback silently gives up AD, so a wrong value would otherwise be
    # invisible (jax's as_numpy cast is read-only, and the FD path writes in
    # place -- that bug passed a "does not raise" check on torch).
    for kw in (dict(lb=True), dict(coordFunc=lambda x: x)):
        got = as_numpy(
            calcaAJac(_arr(backend_name, _XV), aA_iso, actionsFreqsAngles=True, **kw)
        )
        ref = calcaAJac(numpy.array(_XV), aA_iso, actionsFreqsAngles=True, **kw)
        numpy.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-15)


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


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_cart_and_interp_cov_backend_coerce_and_guard(backend_name):
    # The backend twin's defensive branches: the nC<4 guard (cubic spline needs
    # >=4 knots) and the numpy-input coercion that fires when it is reached via a
    # MIXED dispatch (one of chunk_covs/_ObsTrack backend, the other numpy). The
    # coercion must be value-preserving, so each mixed call matches the all-backend
    # reference (same seed -> same data).
    mk = lambda a: _arr(backend_name, a)
    # (a) nC < 4 -> ValueError
    m3, covs3 = _mock_spread_sdf(3, mk)
    with pytest.raises(ValueError, match="nTrackChunks"):
        m3._cart_and_interp_cov_backend(mk(covs3))
    # reference: fully-backend inputs
    m_all, covs = _mock_spread_sdf(5, mk)
    xy_ref = as_numpy(m_all._cart_and_interp_cov_backend(mk(covs))[0])
    # (b) numpy chunk_covs + backend _ObsTrack -> coerce chunk_covs
    xy_b = m_all._cart_and_interp_cov_backend(covs)[0]
    assert is_backend_array(xy_b)
    numpy.testing.assert_allclose(as_numpy(xy_b), xy_ref, rtol=1e-11, atol=1e-12)
    # (c) backend chunk_covs + numpy _ObsTrack/_thetasTrack -> coerce those
    m_nobs, covs2 = _mock_spread_sdf(5, numpy.asarray)
    xy_n = m_nobs._cart_and_interp_cov_backend(mk(covs2))[0]
    assert is_backend_array(xy_n)
    numpy.testing.assert_allclose(as_numpy(xy_n), xy_ref, rtol=1e-11, atol=1e-12)


###############################################################################
# Phase C.4: _determine_stream_spread pipeline dispatches to the backend when the
# track (_allinvjacsTrack) is a backend array -- assembles the per-chunk covs
# functionally + reuses the C.2 eigen-slerp interpolation.
###############################################################################
@pytest.mark.slow
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_determine_stream_spread_backend_pipeline(backend_name):
    import copy

    from galpy.df import streamdf as _cls

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=15)
    obs = Orbit(numpy.array(_STREAM_IC))
    sdf = _cls(
        0.365 / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aA,
        leading=True,
        nTrackChunks=5,
        nTrackIterations=0,
        tdisrupt=_tdisrupt(),
    )
    # Same invjacs/ObsTrack on the backend -> isolates the pipeline from the track
    # FD-vs-AD floor; the backend spread must match the numpy spread to ~machine.
    s = copy.copy(sdf)
    s._allinvjacsTrack = _arr(backend_name, numpy.array(sdf._allinvjacsTrack))
    s._ObsTrack = _arr(backend_name, numpy.array(sdf._ObsTrack))
    s._determine_stream_spread()
    for attr in (
        "_allErrCovsXY",
        "_interpolatedAllErrCovsXY",
        "_allErrCovsLocalXY",
        "_interpolatedAllErrCovsLocalXY",
    ):
        numpy.testing.assert_allclose(
            as_numpy(getattr(s, attr)),
            numpy.array(getattr(sdf, attr)),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{attr} ({backend_name})",
        )


###############################################################################
# Phase D: backend-native, differentiable inverse-CDF sampling (replaces ARS).
#
# _sample_aAt draws the frequency along the largest eigenvalue by piecewise-linear
# inversion of the closed-form tilted-Gaussian CDF (galpy.backend.sampling.
# linear_inverse_cdf_sample, matching sphericaldf #1181) instead of adaptive
# rejection (util.ars). The SAME
# algorithm runs on numpy/jax/torch, dispatching only the RNG source + namespace
# on the `key`. numpy is INTENTIONALLY not byte-identical to the old ARS (a
# different exact sampler on a shifted RNG stream, user-approved); the sampled
# DISTRIBUTION is preserved (KS-indistinguishable, moments match). The payoff:
# with a backend key the draw is a reproducible backend array, differentiable
# w.r.t. the distribution parameters (mO/sigma^2) and jit/GPU-able (no rejection
# loop, static shapes) -- proven below by grad-vs-FD h-convergence.
###############################################################################
import copy as _copy

from galpy.backend import random as grandom
from galpy.backend.sampling import linear_inverse_cdf_sample


def _xp_of(backend_name):
    return numpy if backend_name == "numpy" else get_namespace(_arr(backend_name, 1.0))


# --- (d) distributional parity vs the OLD ARS (numpy path) --------------------
def test_sample_aAt_distribution_vs_ars(sdf):
    # Draw the along-eigenvector frequency both ways and KS-test them. Fixed seeds
    # -> deterministic; at n=1e5 a same-distribution KS statistic is ~0.006, so a
    # comfortable threshold both proves parity and is non-flaky.
    from scipy import stats

    from galpy.util.ars import ars

    mO, s2 = sdf._meandO, sdf._sortedSigOEig[2]

    def h_ars(x, p):
        return -0.5 * (x - p[0]) ** 2.0 / p[1] + numpy.log(x)

    def hp_ars(x, p):
        return -(x - p[0]) / p[1] + 1.0 / x

    n = 100000
    numpy.random.seed(20)
    old = numpy.array(
        ars(
            [0.0, 0.0],
            [True, False],
            [mO - numpy.sqrt(s2), mO + numpy.sqrt(s2)],
            h_ars,
            hp_ars,
            nsamples=n,
            hxparams=(mO, s2),
            maxn=100,
        )
    )
    u = numpy.random.default_rng(20).uniform(size=n)
    og, cg = sdf._dOmega_inverse_cdf_grid(numpy, mO, s2, u)
    new = linear_inverse_cdf_sample(numpy, og, cg, u)
    ks = stats.ks_2samp(old, new)
    assert ks.statistic < 0.02, f"KS stat {ks.statistic:.4f} (p={ks.pvalue:.3f})"
    assert abs(new.mean() / old.mean() - 1.0) < 0.02, "dO1 mean differs from ARS"
    assert abs(new.std() / old.std() - 1.0) < 0.02, "dO1 std differs from ARS"


# --- (e) the streamdf-level sampler: cross-backend, key, grad, jit -------------
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_aAt_cross_backend_same_u(sdf, backend_name):
    # sdf's own grid builder + the sampler, same u across numpy and the backend.
    mO, s2 = sdf._meandO, sdf._sortedSigOEig[2]
    u = numpy.random.default_rng(11).uniform(size=2000)
    og_n, cg_n = sdf._dOmega_inverse_cdf_grid(numpy, mO, s2, u)
    ref = linear_inverse_cdf_sample(numpy, og_n, cg_n, u)
    xp = _xp_of(backend_name)
    og_b, cg_b = sdf._dOmega_inverse_cdf_grid(
        xp, xp.asarray(mO), xp.asarray(s2), xp.asarray(u)
    )
    got = linear_inverse_cdf_sample(xp, og_b, cg_b, xp.asarray(u))
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_aAt_backend_key_runs(sdf, backend_name):
    # A backend key drives _sample_aAt end-to-end (the returnaAdt path) and yields
    # backend arrays of the right shape with dt in [0, tdisrupt].
    k = grandom.key(5, backend=backend_name)
    Om, angle, dt = sdf._sample_aAt(1000, key=k)
    assert is_backend_array(Om) and Om.shape == (3, 1000)
    assert angle.shape == (3, 1000) and dt.shape == (1000,)
    dtn = as_numpy(dt)
    assert dtn.min() >= 0.0 and dtn.max() <= sdf._tdisrupt


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_aAt_grad_vs_fd(sdf, backend_name):
    # d/d(mO), d/d(s2) of a quadratic loss on _sample_aAt's (Om, angle) at a FIXED
    # backend key (fixed u) h-converges to a central FD (backend eager at +/- h).
    # The linear inverse-CDF is only C0 (the icdf slope jumps at each cdf knot), so
    # the AD grad is exact ALMOST EVERYWHERE but the central FD is knot-straddling
    # limited: use a finer h-sweep (to land the FD inside a single knot interval)
    # and a 1e-3 tolerance -- still a genuine grad-vs-FD check, just not machine
    # precision like the smooth cubic. This is the intended cost of linear #1181.
    mO0, s20 = sdf._meandO, sdf._sortedSigOEig[2]
    xp = _xp_of(backend_name)

    def loss(mO, s2):
        s = _copy.copy(sdf)
        s._meandO = mO
        s._sortedSigOEig = [sdf._sortedSigOEig[0], sdf._sortedSigOEig[1], s2]
        Om, angle, dt = s._sample_aAt(1500, key=grandom.key(9, backend=backend_name))
        return xp.mean(Om**2) + xp.mean(angle**2)

    def fd(mO, s2):
        return float(as_numpy(loss(xp.asarray(mO), xp.asarray(s2))))

    if backend_name == "jax":
        g = jax.grad(loss, argnums=(0, 1))(jnp.asarray(mO0), jnp.asarray(s20))
        ad_mO, ad_s2 = float(g[0]), float(g[1])
    else:
        mO_t = torch.tensor(mO0, dtype=torch.float64, requires_grad=True)
        s2t = torch.tensor(s20, dtype=torch.float64, requires_grad=True)
        loss(mO_t, s2t).backward()
        ad_mO, ad_s2 = float(mO_t.grad), float(s2t.grad)
    assert numpy.isfinite(ad_mO) and numpy.isfinite(ad_s2)
    best_mO = min(
        abs(ad_mO - (fd(mO0 + h, s20) - fd(mO0 - h, s20)) / (2 * h))
        for h in (1e-6, 1e-7, 1e-8)
    )
    best_s2 = min(
        abs(ad_s2 - (fd(mO0, s20 + h) - fd(mO0, s20 - h)) / (2 * h))
        for h in (1e-10, 1e-11, 1e-12)
    )
    assert best_mO < 1e-3 * abs(ad_mO) + 1e-8, f"{backend_name} d/dmO {best_mO:.2e}"
    assert best_s2 < 1e-3 * abs(ad_s2) + 1e-8, f"{backend_name} d/ds2 {best_s2:.2e}"


def test_sample_aAt_jit(sdf):
    # _sample_aAt at a fixed n runs under jax.jit (static shapes, no rejection).
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")

    def sample(mO, s2):
        s = _copy.copy(sdf)
        s._meandO = mO
        s._sortedSigOEig = [sdf._sortedSigOEig[0], sdf._sortedSigOEig[1], s2]
        Om, angle, dt = s._sample_aAt(400, key=grandom.key(2, backend="jax"))
        return jnp.mean(Om)

    mO, s2 = jnp.asarray(sdf._meandO), jnp.asarray(sdf._sortedSigOEig[2])
    numpy.testing.assert_allclose(
        float(jax.jit(sample)(mO, s2)), float(sample(mO, s2)), rtol=0, atol=1e-10
    )


def test_sample_t_numpy_byte_identical(sdf):
    # sample_t(key=None) is byte-identical to the historical numpy.random.uniform
    # draw times tdisrupt (the one part of the sampler that stays byte-identical).
    numpy.random.seed(77)
    got = sdf.sample_t(2000, key=None)
    numpy.random.seed(77)
    ref = numpy.random.uniform(size=2000) * sdf._tdisrupt
    numpy.testing.assert_array_equal(got, ref)


###############################################################################
# Phase E: _interpolate_stream_track / _interpolate_stream_track_aA -- the
# stream-track interpolation splines.
#
# numpy path byte-identical (scipy InterpolatedUnivariateSpline(k=3), the
# else-branch is the verbatim original -- verified by a git-stash A/B: the
# interpTrackX/Y/Z/vX/vY/vZ evaluations, _interpolatedObsTrackXY and
# _interpolatedObsTrack are bit-unchanged). Backend twins
# (_interpolate_stream_track_backend / _interpolate_stream_track_aA_backend),
# dispatched when the assembled track _ObsTrack is a backend array, build the six
# coordinate splines in-backend via Spline1D (cubic_spline_coeffs, bc='not-a-knot'
# == scipy IUS(k=3) to ~1e-13) and assemble the fine-grid track functionally, so
# the interpolated track (and the AA track) is backend-native and DIFFERENTIABLE
# w.r.t. the track (_ObsTrack) / the frequency-covariance scalars.
###############################################################################
def _mock_track_sdf(nC, nInterp, mk):
    # Minimal object exposing what _interpolate_stream_track reads: a backend
    # ObsTrack + a monotone theta grid. R (col 0) kept positive so rect<->cyl is
    # well-defined.
    from galpy.df.streamdf import streamdf as _cls

    rng = numpy.random.default_rng(11)
    dtheta = 1.3
    obs = rng.standard_normal((nC, 6)) * 0.15 + numpy.array(
        [1.1, 0.05, 1.0, 0.1, 0.02, 0.3]
    )

    class _M:
        pass

    m = _M()
    m._nTrackChunks = nC
    m._ObsTrack = mk(obs)
    m._thetasTrack = numpy.linspace(0.0, dtheta, nC)
    m._deltaAngleTrack = dtheta
    m.nInterpolatedTrackChunks = nInterp
    m._interpolate_stream_track = _cls._interpolate_stream_track.__get__(m)
    m._interpolate_stream_track_backend = (
        _cls._interpolate_stream_track_backend.__get__(m)
    )
    return m, obs


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_interpolate_stream_track_value_parity(backend_name):
    nC, nI = 8, 301
    m_np, _ = _mock_track_sdf(nC, nI, numpy.asarray)
    m_np._interpolate_stream_track()
    m_b, _ = _mock_track_sdf(nC, nI, lambda a: _arr(backend_name, a))
    m_b._interpolate_stream_track()
    assert is_backend_array(m_b._interpolatedObsTrackXY)
    assert is_backend_array(m_b._interpolatedObsTrack)
    q = numpy.linspace(-0.1, 1.4, 53)  # incl. out-of-range (ext=0 extrapolation)
    qb = _arr(backend_name, q)
    for n in ("X", "Y", "Z", "vX", "vY", "vZ"):
        numpy.testing.assert_allclose(
            as_numpy(getattr(m_b, "_interpTrack" + n)(qb)),
            getattr(m_np, "_interpTrack" + n)(q),
            rtol=1e-11,
            atol=1e-12,
            err_msg=f"interpTrack{n} ({backend_name})",
        )
    numpy.testing.assert_allclose(
        as_numpy(m_b._interpolatedObsTrackXY),
        m_np._interpolatedObsTrackXY,
        rtol=1e-11,
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        as_numpy(m_b._interpolatedObsTrack),
        m_np._interpolatedObsTrack,
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_interpolate_stream_track_derivative_parity(backend_name):
    # Spline1D.derivative() on the backend spline == scipy IUS.derivative() on the
    # numpy path (used by streamdf.length(phys=True)).
    nC, nI = 8, 51
    m_np, _ = _mock_track_sdf(nC, nI, numpy.asarray)
    m_np._interpolate_stream_track()
    m_b, _ = _mock_track_sdf(nC, nI, lambda a: _arr(backend_name, a))
    m_b._interpolate_stream_track()
    q = numpy.linspace(0.0, 1.3, 37)
    qb = _arr(backend_name, q)
    numpy.testing.assert_allclose(
        as_numpy(m_b._interpTrackX.derivative()(qb)),
        m_np._interpTrackX.derivative()(q),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_interpolate_stream_track_grad_vs_fd(backend_name):
    # d(sum interpTrackX(q))/d(_ObsTrack[0,0]) AD h-converges to a central FD of
    # the numpy path (stringent, not finite-and-nonzero).
    nC, nI = 8, 101
    q = numpy.linspace(0.0, 1.3, 41)

    def loss_np(o):
        m, _ = _mock_track_sdf(nC, nI, numpy.asarray)
        m._ObsTrack = o
        m._interpolate_stream_track()
        return float(numpy.sum(m._interpTrackX(q)))

    _, obs = _mock_track_sdf(nC, nI, numpy.asarray)
    if backend_name == "jax":

        def loss(o):
            m, _ = _mock_track_sdf(nC, nI, jnp.asarray)
            m._ObsTrack = o
            m._interpolate_stream_track()
            return jnp.sum(m._interpTrackX(jnp.asarray(q)))

        ad = float(jax.grad(loss)(jnp.asarray(obs))[0, 0])
    else:
        obs_t = torch.tensor(obs, requires_grad=True)
        m, _ = _mock_track_sdf(nC, nI, lambda a: a)
        m._ObsTrack = obs_t
        m._interpolate_stream_track()
        torch.sum(m._interpTrackX(torch.tensor(q))).backward()
        ad = float(obs_t.grad[0, 0])
    best = float("inf")
    for h in (1e-4, 1e-5, 1e-6):
        op, om = obs.copy(), obs.copy()
        op[0, 0] += h
        om[0, 0] -= h
        best = min(best, abs(ad - (loss_np(op) - loss_np(om)) / (2 * h)))
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, f"track grad {backend_name} {best:.2e}"


def _backendify_track(sdf, mk):
    # A shallow copy of a real (numpy) streamdf with the assembled track and the
    # AA scalars promoted to a backend array, and the cached interpolation
    # dropped, so _interpolate_stream_track[_aA] dispatch to their backend twins.
    s = _copy.copy(sdf)
    for a in (
        "_ObsTrack",
        "_progenitor_Omega",
        "_progenitor_angle",
        "_dsigomeanProgDirection",
        "_meandO",
        "_sortedSigOEig",
    ):
        setattr(s, a, mk(numpy.asarray(getattr(sdf, a))))
    for a in (
        "_interpolatedThetasTrack",
        "_interpolatedObsTrackXY",
        "_interpolatedObsTrack",
        "_interpolatedObsTrackAA",
    ):
        if hasattr(s, a):
            delattr(s, a)
    return s


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_interpolate_stream_track_real_sdf_parity(sdf, backend_name):
    # The real Bovy(2014) track promoted to the backend, rebuilt INSIDE a forced-
    # backend context (mirrors the --backend shard's namespace/coercion). Both the
    # physical track (_interpolatedObsTrackXY/_interpolatedObsTrack) and the AA
    # track (_interpolatedObsTrackAA) match the numpy setup to ~1e-11.
    from galpy import backend as _bk

    ref_XY = numpy.asarray(sdf._interpolatedObsTrackXY)
    ref_Track = numpy.asarray(sdf._interpolatedObsTrack)
    ref_AA = numpy.asarray(sdf._interpolatedObsTrackAA)
    mk = lambda a: _arr(backend_name, a)
    with _bk.use(backend_name, force=True):
        s = _backendify_track(sdf, mk)
        s._interpolate_stream_track()
        s._interpolate_stream_track_aA()
        assert is_backend_array(s._interpolatedObsTrackXY)
        assert is_backend_array(s._interpolatedObsTrack)
        assert is_backend_array(s._interpolatedObsTrackAA)
        got_XY = as_numpy(s._interpolatedObsTrackXY)
        got_Track = as_numpy(s._interpolatedObsTrack)
        got_AA = as_numpy(s._interpolatedObsTrackAA)
    numpy.testing.assert_allclose(got_XY, ref_XY, rtol=1e-11, atol=1e-12)
    numpy.testing.assert_allclose(got_Track, ref_Track, rtol=1e-11, atol=1e-12)
    numpy.testing.assert_allclose(got_AA, ref_AA, rtol=1e-11, atol=1e-12)


def _mock_aa_sdf(mk, meandO, nInterp=121):
    from galpy.df.streamdf import streamdf as _cls

    dsig = numpy.array([0.3, 0.8, 0.5])
    dsig = dsig / numpy.linalg.norm(dsig)

    class _M:
        pass

    m = _M()
    m._ObsTrack = mk(numpy.zeros((5, 6)))  # dispatch + namespace only
    m._interpolatedThetasTrack = numpy.linspace(0.0, 1.4, nInterp)
    m._meandO = meandO
    m._sortedSigOEig = mk(numpy.array([0.0, 0.0, 3.1e-4]))
    m._tdisrupt = 12.0
    m._sigMeanSign = 1.0
    m._progenitor_Omega = mk(numpy.array([0.31, 0.55, -0.42]))
    m._progenitor_angle = mk(numpy.array([1.1, 2.2, 0.7]))
    m._dsigomeanProgDirection = mk(dsig)
    m._interpolate_stream_track_aA_backend = (
        _cls._interpolate_stream_track_aA_backend.__get__(m)
    )
    m.meanOmega = _cls.meanOmega.__get__(m)
    return m


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_interpolate_stream_track_aA_grad_vs_fd(backend_name):
    # d(sum _interpolatedObsTrackAA)/d(_meandO) AD h-converges to a central FD of
    # the numpy build (the AA track is differentiable via meanOmega's dmOs).
    md0 = 0.021

    def loss_np(md):
        m = _mock_aa_sdf(numpy.asarray, float(md))
        m._interpolate_stream_track_aA_backend()
        return float(numpy.sum(m._interpolatedObsTrackAA))

    if backend_name == "jax":

        def loss(md):
            m = _mock_aa_sdf(jnp.asarray, md)
            m._interpolate_stream_track_aA_backend()
            return jnp.sum(m._interpolatedObsTrackAA)

        ad = float(jax.grad(loss)(jnp.asarray(md0)))
    else:
        mdt = torch.tensor(md0, requires_grad=True)
        m = _mock_aa_sdf(lambda a: torch.tensor(a), mdt)
        m._interpolate_stream_track_aA_backend()
        torch.sum(m._interpolatedObsTrackAA).backward()
        ad = float(mdt.grad)
    best = float("inf")
    for h in (1e-5, 1e-6, 1e-7):
        best = min(best, abs(ad - (loss_np(md0 + h) - loss_np(md0 - h)) / (2 * h)))
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-4 * abs(ad) + 1e-6, f"aA grad {backend_name} {best:.2e}"


###############################################################################
# _approxaAInv -- the linear track inverse (frequency-angle -> R,vR,vT,z,vz,phi)
# that sample(returnaAdt=False) applies to every drawn (Omega, angle). The numpy
# body (dispatched away when the query points OR the assembled track are backend
# arrays) is BYTE-IDENTICAL -- verified by a git-stash A/B on a real numpy sdf,
# out-of-band, for both interp=True/False (array_equal, maxdiff 0.0).
#
# The backend twin (_approxaAInv_backend) vectorises the per-point loop. Every
# discrete selection -- the 9**3 wrap (argmin over the cross-product norm), the
# closest interp/non-interp track point, the two Jacobian indices (the numpy
# data-dependent branch becomes clamped indices + xp.where) -- is a stop-gradient
# integer argmin that GATHERs the continuous track/Jacobian rows it points at
# (reparameterised nearest-neighbour). The gradient flows through dOa (the
# offset), the gathered _interpolatedObsTrack/_allinvjacsTrack, and the smoothing
# weight, NOT through the index. Proven below: value parity, grad-vs-FD (the
# query offset AND the gather-gradient) and end-to-end differentiable sampling.
###############################################################################
def _approxaainv_query(sdf, n=16):
    # realistic frequency/angle query points from the numpy sampler (seeded)
    numpy.random.seed(202)
    Om, angle, _ = sdf._sample_aAt(n, key=None)
    return numpy.vstack([Om, angle])  # (6,n): Or,Op,Oz, ar,ap,az


def _np_inv(sdf, Q, interp):
    return numpy.asarray(
        sdf._approxaAInv(Q[0], Q[1], Q[2], Q[3], Q[4], Q[5], interp=interp)
    )


@pytest.mark.parametrize("interp", [True, False], ids=["interp", "noninterp"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaAInv_value_parity(sdf, backend_name, interp):
    # backend _approxaAInv == the numpy path to ~1e-11 at the same query points,
    # with the query points built INSIDE the forced-backend context.
    from galpy import backend as _bk

    Q = _approxaainv_query(sdf)
    ref = _np_inv(sdf, Q, interp)
    with _bk.use(backend_name, force=True):
        Qb = [_arr(backend_name, Q[i]) for i in range(6)]
        out = sdf._approxaAInv(*Qb, interp=interp)
        assert is_backend_array(out)
        got = as_numpy(out)
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaAInv_grad_vs_fd_query(sdf, backend_name):
    # d(sum W*out)/d(query offset) AD h-converges to a central FD of the numpy
    # path (the gradient flows through dOa and the smoothing weight; the discrete
    # nearest-neighbour indices are stop-gradient).
    Q = _approxaainv_query(sdf)
    n = Q.shape[1]
    rng = numpy.random.RandomState(7)
    W, Dir = rng.randn(6, n), rng.randn(6, n)
    interp = True

    def loss_np(Qm):
        return float(numpy.sum(W * _np_inv(sdf, Qm, interp)))

    if backend_name == "jax":

        def loss(qf):
            Qm = qf.reshape(6, n)
            out = sdf._approxaAInv(*(Qm[i] for i in range(6)), interp=interp)
            return jnp.sum(jnp.asarray(W) * out)

        g = numpy.asarray(jax.grad(loss)(jnp.asarray(Q.ravel()))).reshape(6, n)
    else:
        qt = torch.tensor(Q, requires_grad=True)
        out = sdf._approxaAInv(*(qt[i] for i in range(6)), interp=interp)
        (torch.as_tensor(W) * out).sum().backward()
        g = qt.grad.numpy()
    ad = float(numpy.sum(g * Dir))
    best = min(
        abs(ad - (loss_np(Q + h * Dir) - loss_np(Q - h * Dir)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, f"{backend_name} query grad {best:.2e}"


@pytest.mark.parametrize(
    "attr,idx",
    [("_allinvjacsTrack", "jac"), ("_interpolatedObsTrack", "closest")],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaAInv_grad_vs_fd_table(sdf, backend_name, attr, idx):
    # the gather-gradient: d(sum W*out)/d(a gathered track/Jacobian entry) AD
    # h-converges to a central FD of the numpy path (grad flows through the
    # nearest-neighbour gather into the continuous table row).
    Q = _approxaainv_query(sdf)
    n = Q.shape[1]
    W = numpy.random.RandomState(3).randn(6, n)
    interp = True
    # an index actually gathered by query point 0
    ci = sdf._find_closest_trackpointaA(*[Q[i][0] for i in range(6)], interp=True)
    ji = sdf._find_closest_trackpointaA(*[Q[i][0] for i in range(6)], interp=False)
    entry = (int(ji), 2, 3) if idx == "jac" else (int(ci), 1)
    base = numpy.asarray(getattr(sdf, attr))

    def loss_np(delta):
        s = _copy.copy(sdf)
        T = base.copy()
        T[entry] += delta
        setattr(s, attr, T)
        return float(numpy.sum(W * _np_inv(s, Q, interp)))

    Qb = [_arr(backend_name, Q[i]) for i in range(6)]
    if backend_name == "jax":

        def loss(Tflat):
            s = _copy.copy(sdf)
            setattr(s, attr, Tflat.reshape(base.shape))
            out = s._approxaAInv_backend(*Qb, interp=interp)
            return jnp.sum(jnp.asarray(W) * out)

        ad = float(
            numpy.asarray(jax.grad(loss)(jnp.asarray(base.ravel()))).reshape(
                base.shape
            )[entry]
        )
    else:
        T = torch.tensor(base, requires_grad=True)
        s = _copy.copy(sdf)
        setattr(s, attr, T)
        out = s._approxaAInv_backend(*Qb, interp=interp)
        (torch.as_tensor(W) * out).sum().backward()
        ad = float(T.grad.numpy()[entry])
    best = min(
        abs(ad - (loss_np(h) - loss_np(-h)) / (2 * h)) for h in (1e-4, 1e-5, 1e-6)
    )
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, f"{backend_name} {attr} grad {best:.2e}"


@pytest.mark.parametrize("interp", [True, False], ids=["interp", "noninterp"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaAInv_sample_forced_backend(sdf, backend_name, interp):
    # sample(returnaAdt=False) composes _sample_aAt -> _approxaAInv_backend under a
    # forced backend: a backend array of shape (6,n) that matches the numpy path
    # fed the same (backend-key) draws.
    from galpy import backend as _bk

    n = 150
    k = grandom.key(3, backend=backend_name)
    Om, angle, _ = sdf._sample_aAt(n, key=k)
    ref = numpy.asarray(
        sdf._approxaAInv(
            *(as_numpy(Om[i]) for i in range(3)),
            *(as_numpy(angle[i]) for i in range(3)),
            interp=interp,
        )
    )
    with _bk.use(backend_name, force=True):
        RvR = sdf.sample(n, returnaAdt=False, interp=interp, key=k)
    assert is_backend_array(RvR) and tuple(RvR.shape) == (6, n)
    numpy.testing.assert_allclose(as_numpy(RvR), ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaAInv_sample_grad_vs_fd(sdf, backend_name):
    # end-to-end: d(sum z^2)/d(progenitor angle) through the WHOLE sample path.
    # _progenitor_angle feeds both the angle draw (_sample_aAt) and the wrap
    # disambiguation (_approxaAInv_backend); a fixed backend key gives common
    # random numbers so the central FD is valid.
    from galpy import backend as _bk

    n, comp, interp = 150, 0, True
    base = numpy.asarray(sdf._progenitor_angle)
    key = grandom.key(8, backend=backend_name)
    xp = _xp_of(backend_name)

    def loss(paval, differentiable=False):
        s = _copy.copy(sdf)
        if differentiable and backend_name == "jax":
            s._progenitor_angle = jnp.asarray(base).at[comp].set(paval)
        elif differentiable:
            pa = torch.tensor(base).clone()
            pa[comp] = paval
            s._progenitor_angle = pa
        else:
            pa = base.copy()
            pa[comp] = float(as_numpy(paval))
            s._progenitor_angle = _arr(backend_name, pa)
        with _bk.use(backend_name, force=True):
            return xp.sum(s.sample(n, returnaAdt=False, interp=interp, key=key)[3] ** 2)

    if backend_name == "jax":
        ad = float(jax.grad(lambda p: loss(p, True))(jnp.asarray(base[comp])))
    else:
        pv = torch.tensor(base[comp], requires_grad=True)
        loss(pv, True).backward()
        ad = float(pv.grad)
    best = min(
        abs(ad - float(as_numpy(loss(base[comp] + h) - loss(base[comp] - h))) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-4 * abs(ad) + 1e-6, f"{backend_name} sample grad {best:.2e}"


###############################################################################
# _approxaA -- the FORWARD linear track map ((R,vR,vT,z,vz,phi) -> (O,a)) that
# streamgapdf's _determine_deltaOmegaTheta_kick applies to every kicked point (a
# Phase-2a prerequisite of the streamgapdf gap-track work). The numpy body
# (dispatched away when the query points OR the assembled track are backend
# arrays) is BYTE-IDENTICAL -- verified by a git-stash A/B on a real numpy sdf,
# out-of-band, for interp=True/False AND the cindx=range identity path
# (array_equal, maxdiff 0.0).
#
# The backend twin (_approxaA_backend) vectorises the per-point loop. Every
# discrete selection -- the closest interp/non-interp track point and the two
# Jacobian indices (numpy's data-dependent branch -> clamped indices + xp.where)
# -- is a stop-gradient integer argmin over the Cartesian (X,Y,Z) distance that
# GATHERs the continuous track/Jacobian rows it points at (reparameterised
# nearest-neighbour). The gradient flows through dxv (the offset), the gathered
# _interpolatedObsTrack/_alljacsTrack/_interpolatedObsTrackAA rows and the
# smoothing weight, NOT the index. Proven below: value parity (incl. the
# cindx identity path), grad-vs-FD for the query offset AND the gather-gradient.
###############################################################################
def _approxaa_query(sdf, n=16):
    # realistic config-space query points: interpolated-track config points
    # perturbed off the track (near-but-not-ON a chunk -> smooth smoothing weight)
    numpy.random.seed(321)
    npts = sdf._interpolatedObsTrack.shape[0]
    sel = numpy.linspace(0, npts - 1, n).astype(int)
    base = numpy.asarray(sdf._interpolatedObsTrack)[sel]
    return (base + 0.01 * numpy.random.randn(n, 6)).T  # (6,n): R,vR,vT,z,vz,phi


def _np_fwd(sdf, Q, interp):
    return numpy.asarray(
        sdf._approxaA(Q[0], Q[1], Q[2], Q[3], Q[4], Q[5], interp=interp)
    )


@pytest.mark.parametrize("interp", [True, False], ids=["interp", "noninterp"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaA_value_parity(sdf, backend_name, interp):
    # backend _approxaA == the numpy path to ~1e-11 at the same query points,
    # with the query points built INSIDE the forced-backend context.
    from galpy import backend as _bk

    Q = _approxaa_query(sdf)
    ref = _np_fwd(sdf, Q, interp)
    with _bk.use(backend_name, force=True):
        Qb = [_arr(backend_name, Q[i]) for i in range(6)]
        out = sdf._approxaA(*Qb, interp=interp)
        assert is_backend_array(out)
        got = as_numpy(out)
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("interp", [True, False], ids=["interp", "noninterp"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaA_cindx_identity_parity(sdf, backend_name, interp):
    # the cindx path streamgapdf._determine_deltaOmegaTheta_kick uses: query the
    # track points themselves with cindx=range(N) (identity closest-point map).
    from galpy import backend as _bk

    track = sdf._interpolatedObsTrack if interp else sdf._ObsTrack
    Q = numpy.asarray(track).T  # (6,N)
    cindx = range(Q.shape[1])
    ref = numpy.asarray(
        sdf._approxaA(*(Q[i] for i in range(6)), interp=interp, cindx=cindx)
    )
    with _bk.use(backend_name, force=True):
        Qb = [_arr(backend_name, Q[i]) for i in range(6)]
        out = sdf._approxaA(*Qb, interp=interp, cindx=cindx)
        assert is_backend_array(out)
        got = as_numpy(out)
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaA_grad_vs_fd_query(sdf, backend_name):
    # d(sum W*out)/d(query point) AD h-converges to a central FD of the numpy path
    # (grad flows through dxv and the smoothing weight; the discrete
    # nearest-neighbour indices are stop-gradient).
    Q = _approxaa_query(sdf)
    n = Q.shape[1]
    rng = numpy.random.RandomState(7)
    W, Dir = rng.randn(6, n), rng.randn(6, n)
    interp = True

    def loss_np(Qm):
        return float(numpy.sum(W * _np_fwd(sdf, Qm, interp)))

    if backend_name == "jax":

        def loss(qf):
            Qm = qf.reshape(6, n)
            out = sdf._approxaA(*(Qm[i] for i in range(6)), interp=interp)
            return jnp.sum(jnp.asarray(W) * out)

        g = numpy.asarray(jax.grad(loss)(jnp.asarray(Q.ravel()))).reshape(6, n)
    else:
        qt = torch.tensor(Q, requires_grad=True)
        out = sdf._approxaA(*(qt[i] for i in range(6)), interp=interp)
        (torch.as_tensor(W) * out).sum().backward()
        g = qt.grad.numpy()
    ad = float(numpy.sum(g * Dir))
    best = min(
        abs(ad - (loss_np(Q + h * Dir) - loss_np(Q - h * Dir)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, f"{backend_name} query grad {best:.2e}"


@pytest.mark.parametrize(
    "attr,idx",
    [
        ("_alljacsTrack", "jac"),
        ("_interpolatedObsTrack", "closest"),
        ("_interpolatedObsTrackAA", "aa"),
    ],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_approxaA_grad_vs_fd_table(sdf, backend_name, attr, idx):
    # the gather-gradient: d(sum W*out)/d(a gathered track/Jacobian entry) AD
    # h-converges to a central FD of the numpy path (grad flows through the
    # Cartesian nearest-neighbour gather into the continuous table row).
    Q = _approxaa_query(sdf)
    n = Q.shape[1]
    W = numpy.random.RandomState(3).randn(6, n)
    interp = True
    # indices actually gathered by query point 0 (Cartesian nearest-neighbour)
    X0 = Q[0][0] * numpy.cos(Q[5][0])
    Y0 = Q[0][0] * numpy.sin(Q[5][0])
    Z0 = Q[3][0]
    ci = sdf._find_closest_trackpoint(
        X0, Y0, Z0, Q[3][0], Q[4][0], Q[5][0], interp=True, xy=True, usev=False
    )
    ji = sdf._find_closest_trackpoint(
        Q[0][0], Q[1][0], Q[2][0], Q[3][0], Q[4][0], Q[5][0], interp=False, xy=False
    )
    entry = (int(ji), 2, 3) if idx == "jac" else (int(ci), 1)
    base = numpy.asarray(getattr(sdf, attr))

    def loss_np(delta):
        s = _copy.copy(sdf)
        T = base.copy()
        T[entry] += delta
        setattr(s, attr, T)
        return float(numpy.sum(W * _np_fwd(s, Q, interp)))

    Qb = [_arr(backend_name, Q[i]) for i in range(6)]
    if backend_name == "jax":

        def loss(Tflat):
            s = _copy.copy(sdf)
            setattr(s, attr, Tflat.reshape(base.shape))
            out = s._approxaA_backend(*Qb, interp=interp)
            return jnp.sum(jnp.asarray(W) * out)

        ad = float(
            numpy.asarray(jax.grad(loss)(jnp.asarray(base.ravel()))).reshape(
                base.shape
            )[entry]
        )
    else:
        T = torch.tensor(base, requires_grad=True)
        s = _copy.copy(sdf)
        setattr(s, attr, T)
        out = s._approxaA_backend(*Qb, interp=interp)
        (torch.as_tensor(W) * out).sum().backward()
        ad = float(T.grad.numpy()[entry])
    best = min(
        abs(ad - (loss_np(h) - loss_np(-h)) / (2 * h)) for h in (1e-4, 1e-5, 1e-6)
    )
    assert numpy.isfinite(ad) and abs(ad) > 0
    assert best < 1e-5 * abs(ad) + 1e-6, f"{backend_name} {attr} grad {best:.2e}"


# --- callMarg / __call__ marginalized-PDF reduction on a backend --------------
# streamdf.callMarg feeds __call__ phase-space coords from the migrated
# coords.rect_to_cyl transforms; under a FORCED backend those become tensors, so
# __call__'s numpy.sum(...,axis=0) reduction (torch takes dim=, not axis=) raised
# "sum() got an unexpected keyword argument 'axis'" -- the crash behind
# test_streamdf.py::test_bovy14_callMargXZ. The dual-path migration (_call_backend,
# prepData4Call's xp.where wrap, the callMarg _bspecial.logsumexp reduction) runs
# the WHOLE marginalized PDF on the data's namespace and RETURNS a backend array
# (scipy.logsumexp would silently np.asarray it back to numpy). numpy is
# byte-identical (the else-branches are the verbatim original). Forced backend is
# required because callMarg builds its integration coords internally from the
# numpy gaussApprox/meshgrid, so only a forced context makes rect_to_cyl tensorise.
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_callMarg_reduction_backend_parity_and_type(sdf, backend_name):
    from galpy import backend

    meanp, _ = sdf.gaussApprox([None, None, 2.0 / 8.0, None, None, None])
    xy = [float(meanp[0]), None, 2.0 / 8.0, None, None, None]  # p(X|Z) at the peak
    ref = float(sdf.callMarg(xy))  # numpy path
    ref2 = float(sdf.callMarg(xy, ngl=6, nsigma=3.1))
    with backend.use(backend_name, force=True):
        got = sdf.callMarg(xy)
        got2 = sdf.callMarg(xy, ngl=6, nsigma=3.1)
        # the marginalization must stay on the backend, not silently exit to numpy
        assert is_backend_array(got), (
            f"{backend_name}: callMarg must return a backend array, got {type(got)}"
        )
        assert is_backend_array(got2)
    numpy.testing.assert_allclose(
        float(as_numpy(got)),
        ref,
        rtol=1e-10,
        atol=0.0,
        err_msg=f"callMarg {backend_name}",
    )
    numpy.testing.assert_allclose(
        float(as_numpy(got2)),
        ref2,
        rtol=1e-10,
        atol=0.0,
        err_msg=f"callMarg ngl6/nsigma3.1 {backend_name}",
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_call_backend_value_and_grad(sdf, backend_name):
    # __call__ (the migrated reduction kernel) at a well-within-stream track point:
    # value parity + d(logDF)/d(R) h-converges to a central FD of the numpy path.
    # For jax the kernel is additionally jax.jit'd -- it must TRACE the migrated
    # path (no ConcretizationError / no in-place-assignment error).
    tp = numpy.asarray(
        sdf._interpolatedObsTrack[500]
    )  # (R,vR,vT,z,vz,phi) on the track
    R, vR, vT, z, vz, phi = (numpy.array([v]) for v in tp)
    ref = float(numpy.asarray(sdf(R, vR, vT, z, vz, phi, log=True))[0])
    assert numpy.isfinite(ref)
    rest = [_arr(backend_name, a) for a in (vR, vT, z, vz, phi)]
    if backend_name == "jax":

        def kernel(Rv):
            return sdf(Rv, *rest, log=True)[0]

        got = float(jax.jit(kernel)(_arr(backend_name, R)))
        # grad matches the (1,) input shape -> index the single element
        ad = float(numpy.asarray(jax.jit(jax.grad(kernel))(_arr(backend_name, R)))[0])
    else:
        Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
        out = sdf(Rt, *rest, log=True)[0]
        got = float(out.detach())
        out.backward()
        ad = float(Rt.grad[0])
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=0.0)
    h = 1e-6
    fd = (
        float(numpy.asarray(sdf(R + h, vR, vT, z, vz, phi, log=True))[0])
        - float(numpy.asarray(sdf(R - h, vR, vT, z, vz, phi, log=True))[0])
    ) / (2 * h)
    assert numpy.isfinite(ad) and abs(ad - fd) < 1e-5 * abs(fd) + 1e-6, (
        f"{backend_name} __call__ grad {ad} vs FD {fd}"
    )
