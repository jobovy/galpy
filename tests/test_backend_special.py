###############################################################################
# test_backend_special.py: the native-preferring special-function router
# (galpy.backend.special). For each Tier-1 function this asserts:
#   1. value parity numpy/jax/torch vs scipy.special (numpy byte-identical;
#      jax/torch to rtol 1e-12) on galpy's argument ranges;
#   2. autodiff (jax.grad / torch.autograd) matches central finite differences;
#   3. the fallback table (_NEEDS_FALLBACK) matches the installed backends, and
#      every fallback agrees with scipy (so a fallback is deletable once the
#      backend ships the native function).
###############################################################################
import numpy
import pytest
import scipy.special as scipy_special
from conftest import torch_compiles

from galpy.backend import as_numpy
from galpy.backend import special as gsp
from galpy.backend import use
from galpy.backend.special._router import (
    _NATIVE_MISSING,
    _NATIVE_UNRELIABLE,
    _backend_special,
)

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
_TORCH_COMPILES = torch_compiles()


def _asarray(backend, x, requires_grad=False):
    if backend == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


# (router fn, scipy fn, n-args, sample points). Ranges cover what galpy uses.
_POS = numpy.array([0.1, 0.5, 0.9, 1.3, 2.0, 3.7, 5.0])
_REAL = numpy.array([-3.0, -1.2, -0.3, 0.0, 0.4, 1.1, 2.5])
_GAMMA_X = numpy.array([0.2, 0.5, 1.0, 2.5, 4.0, -0.5, -1.5, -2.5])  # incl. reflection
_A = numpy.array([0.5, 1.0, 2.0, 3.5])
_XG = numpy.array([0.05, 0.5, 1.5, 3.0, 6.0])

UNARY = [
    ("gammaln", gsp.gammaln, scipy_special.gammaln, _POS),
    ("gamma", gsp.gamma, scipy_special.gamma, _GAMMA_X),
    ("erf", gsp.erf, scipy_special.erf, _REAL),
    ("erfc", gsp.erfc, scipy_special.erfc, _REAL),
    ("i0", gsp.i0, scipy_special.i0, numpy.abs(_REAL)),
    ("i1", gsp.i1, scipy_special.i1, _REAL),
]


@pytest.mark.parametrize("name,fn,sp_fn,pts", UNARY, ids=[u[0] for u in UNARY])
@pytest.mark.parametrize("backend", BACKENDS)
def test_unary_value_parity(backend, name, fn, sp_fn, pts):
    ref = sp_fn(pts)
    got = as_numpy(fn(_asarray(backend, pts)))
    rtol = 0.0 if backend == "numpy" else 1e-12  # numpy must be byte-identical
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=1e-12, err_msg=f"{name} ({backend})"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_gammainc_value_parity(backend):
    for sp_fn, fn in [
        (scipy_special.gammainc, gsp.gammainc),
        (scipy_special.gammaincc, gsp.gammaincc),
    ]:
        for a in _A:
            ref = sp_fn(a, _XG)
            got = as_numpy(fn(_asarray(backend, a), _asarray(backend, _XG)))
            rtol = 0.0 if backend == "numpy" else 1e-12
            numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_xlogy_value_parity_incl_zero(backend):
    x = numpy.array([0.0, 0.0, 1.0, 2.5, 0.3])
    y = numpy.array([0.0, 5.0, 2.0, 0.7, 10.0])  # x=0 -> 0 even when y=0
    ref = scipy_special.xlogy(x, y)
    got = as_numpy(gsp.xlogy(_asarray(backend, x), _asarray(backend, y)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logsumexp_value_parity(backend):
    # incl. +-800 to exercise the overflow guard (naive exp would inf/0 out)
    a = numpy.array([[-1.2, 0.3, 800.0, -800.0], [0.5, 2.0, 799.0, 0.0]])
    for axis in (0, 1, None):
        ref = scipy_special.logsumexp(a, axis=axis)
        got = as_numpy(gsp.logsumexp(_asarray(backend, a), axis=axis))
        rtol = 0.0 if backend == "numpy" else 1e-12
        numpy.testing.assert_allclose(
            got, ref, rtol=rtol, atol=1e-12, err_msg=f"logsumexp axis={axis}"
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_logsumexp_grad_vs_fd(backend):
    rest = [-0.7, 1.1]
    x0 = 0.3
    eps = 1e-6

    def f_np(x):
        return float(scipy_special.logsumexp(numpy.array([x] + rest)))

    fd = (f_np(x0 + eps) - f_np(x0 - eps)) / (2 * eps)
    if backend == "jax":
        ad = float(
            jax.grad(lambda x: gsp.logsumexp(jnp.stack([x, *map(jnp.asarray, rest)])))(
                jnp.asarray(x0)
            )
        )
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.logsumexp(
            torch.stack([xt, *(torch.tensor(r, dtype=torch.float64) for r in rest)])
        ).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("name,fn,sp_fn,pts", UNARY, ids=[u[0] for u in UNARY])
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_unary_grad_vs_fd(backend, name, fn, sp_fn, pts):
    # differentiate at smooth interior points (avoid gamma's poles at <=0 ints)
    x0 = 1.3 if name in ("gammaln", "gamma", "i0", "i1") else 0.7
    eps = 1e-6
    fd = (float(sp_fn(x0 + eps)) - float(sp_fn(x0 - eps))) / (2 * eps)
    if backend == "jax":
        ad = float(jax.grad(lambda x: fn(x))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        fn(xt).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad), f"NaN grad for {name} ({backend})"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, err_msg=f"{name} grad ({backend})")


def test_fallback_table_matches_installed_backends():
    # _NATIVE_MISSING must list exactly the functions the backend's special
    # module lacks (hasattr); the UNRELIABLE set is the opposite (must be
    # present, else there is nothing to override).
    tier12 = [
        "gammaln", "gamma", "gammainc", "gammaincc", "erf", "erfc", "i0", "i1",
        "hyp2f1", "hyp1f1", "ellipk", "ellipe", "k0", "k1", "kn", "iv", "sici",
        "exp1",
    ]  # fmt: skip
    for backend in AD_BACKENDS:
        xp = _asarray(backend, 1.0)
        from galpy.backend import get_namespace

        _name, sp = _backend_special(get_namespace(xp))
        for fn in tier12:
            missing = fn in _NATIVE_MISSING.get(backend, frozenset())
            native = hasattr(sp, fn)
            assert missing == (not native), (
                f"{backend}: {fn} native={native} but listed-as-missing={missing}; "
                f"update _NATIVE_MISSING"
            )
        for fn in _NATIVE_UNRELIABLE.get(backend, frozenset()):
            assert hasattr(sp, fn), (
                f"{backend}: {fn} is listed UNRELIABLE but absent natively; it "
                f"belongs in _NATIVE_MISSING instead"
            )


# --- iv (modified Bessel I, integer order) and sici (sine/cosine integral) ----
@pytest.mark.parametrize("backend", BACKENDS)
def test_iv_value_parity(backend):
    # spans x=0, small x (series, no 2/x cancellation), the |x|=2 series/
    # recurrence seam, and large x (overflow-regime parity).
    pts = numpy.array([0.0, 1e-3, 1e-2, 0.3, 0.8, 1.5, 2.0, 2.5, 3.5, 5.0, 30.0])
    for n in (0, 1, 2):
        ref = scipy_special.iv(n, pts)
        got = as_numpy(gsp.iv(n, _asarray(backend, pts)))
        rtol = 0.0 if backend == "numpy" else 1e-10  # series/recurrence ~1e-15
        numpy.testing.assert_allclose(
            got, ref, rtol=rtol, atol=1e-12, err_msg=f"iv n={n} ({backend})"
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_iv_zero_value_and_grad_finite(backend):
    # I_n(0)=0 for n>=2 and I_n'(0)=0: the upward recurrence divides by x, so
    # x=0 must be handled (no NaN in value OR reverse-mode gradient).
    assert as_numpy(gsp.iv(2, _asarray(backend, 0.0))) == 0.0
    if backend == "jax":
        g = float(jax.grad(lambda x: gsp.iv(2, x))(jnp.asarray(0.0)))
    else:
        xt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        gsp.iv(2, xt).backward()
        g = float(xt.grad)
    assert g == 0.0, f"iv(2,0) grad not finite-zero: {g} ({backend})"


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_iv_grad_vs_analytic(backend):
    x0 = 2.0  # d/dx I2 = I1 - (2/x) I2
    ref = scipy_special.iv(1, x0) - 2.0 / x0 * scipy_special.iv(2, x0)
    if backend == "jax":
        ad = float(jax.grad(lambda x: gsp.iv(2, x))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.iv(2, xt).backward()
        ad = float(xt.grad)
    numpy.testing.assert_allclose(ad, ref, rtol=1e-6, err_msg=f"iv grad ({backend})")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sici_value_parity(backend):
    # span both regimes (series x<=6, Gauss-Laguerre auxiliary x>6)
    pts = numpy.array([0.2, 0.8, 2.0, 5.0, 7.0, 20.0, 80.0])
    rsi, rci = scipy_special.sici(pts)
    si, ci = gsp.sici(_asarray(backend, pts))
    si, ci = as_numpy(si), as_numpy(ci)
    rtol = 0.0 if backend == "numpy" else 1e-10
    numpy.testing.assert_allclose(
        si, rsi, rtol=rtol, atol=1e-11, err_msg=f"Si ({backend})"
    )
    numpy.testing.assert_allclose(
        ci, rci, rtol=rtol, atol=1e-11, err_msg=f"Ci ({backend})"
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_sici_grad_vs_analytic(backend):
    x0 = 3.0  # d/dx Si = sin(x)/x ; d/dx Ci = cos(x)/x
    refsi, refci = numpy.sin(x0) / x0, numpy.cos(x0) / x0
    if backend == "jax":
        gsi = float(jax.grad(lambda x: gsp.sici(x)[0])(jnp.asarray(x0)))
        gci = float(jax.grad(lambda x: gsp.sici(x)[1])(jnp.asarray(x0)))
    else:
        xs = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.sici(xs)[0].backward()
        gsi = float(xs.grad)
        xc = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.sici(xc)[1].backward()
        gci = float(xc.grad)
    numpy.testing.assert_allclose(gsi, refsi, rtol=1e-6, err_msg=f"Si grad ({backend})")
    numpy.testing.assert_allclose(gci, refci, rtol=1e-6, err_msg=f"Ci grad ({backend})")


# --- Tier 2: hyp2f1 / hyp1f1 / ellipk / ellipe --------------------------------
# galpy's 2F1 calls (forces use c=a+1; the beta!=3 eval uses c=a+2), z = -w <= 0.
_HYP2F1_CASES = [
    (2.0, 2.0, 3.0),  # NFW-like force (alpha=1, beta=3): 2F1(3-a, b-a, 4-a)
    (2.0, 3.0, 3.0),  # Hernquist-like (alpha=1, beta=4)
    (1.0, 2.0, 2.0),  # Jaffe-like (alpha=2, beta=4)
    (1.5, 2.0, 2.5),  # Dehnen alpha=1.5, beta=3.5
    (0.5, 1.0, 1.5),  # PowerSpherical _surfdens 2F1(0.5, alpha/2, 1.5)
    (1.0, 3.0, 3.0),  # beta!=3 eval, c=a+2 (beta=4): 2F1(beta-3, beta-alpha, beta-1)
]
# realistic radii r/a <~ 50 -- the fallback quadrature is ~1e-10 here
_HYP2F1_W = numpy.array([0.0, 1e-3, 0.05, 0.5, 0.9, 1.0, 1.7, 5.0, 12.0, 25.0, 50.0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("a,b,c", _HYP2F1_CASES, ids=[str(x) for x in _HYP2F1_CASES])
def test_hyp2f1_value_parity(backend, a, b, c):
    z = -_HYP2F1_W
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    # numpy routes to scipy itself (exact); the jax/torch fallback quadrature
    # measures 1.35e-14 worst-case over this grid, so 1e-12 keeps ~70x headroom
    # for libm differences across platforms while still pinning real accuracy.
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("a,b,c", _HYP2F1_CASES, ids=[str(x) for x in _HYP2F1_CASES])
def test_hyp2f1_extreme_z_bounded_error(backend, a, b, c):
    # Far beyond realistic radii (r/a up to 500) the fixed-order quadrature
    # degrades gracefully -- still ~1e-5, never diverges (unlike jax native).
    z = -numpy.array([100.0, 250.0, 500.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)


@pytest.mark.skipif("'torch' not in BACKENDS or not _TORCH_COMPILES")
@pytest.mark.parametrize("a,b,c", _HYP2F1_CASES, ids=[str(x) for x in _HYP2F1_CASES])
def test_hyp2f1_survives_inductor_fusion(a, b, c):
    # Regression: inductor's FUSED expm1 (unlike its standalone one, which is
    # exact) degenerates to exp(x)-1 for tiny arguments and returns 0 there. The
    # fallback's first quadrature node sits at XL ~ 1e-49, so T became 0, then
    # T**(B-1) with B-1 < 0 became inf, and the whole quadrature was inf -- at
    # EVERY z, not just extreme ones. That made TwoPowerSpherical/-Triaxial
    # compile to inf while eager was correct, i.e. SILENTLY wrong output rather
    # than an error. Compare compiled against scipy, not merely against eager,
    # so a regression that breaks both paths at once still fails.
    z = -_HYP2F1_W
    ref = scipy_special.hyp2f1(a, b, c, z)
    compiled = torch.compile(
        lambda zz: gsp.hyp2f1(a, b, c, zz), fullgraph=False, dynamic=False
    )
    got = as_numpy(compiled(_asarray("torch", z)))
    assert numpy.all(numpy.isfinite(got)), "inductor reintroduced the inf blow-up"
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hyp1f1_value_parity(backend):
    for alpha in [0.0, 1.0, 1.8, 2.5]:
        a, b = 1.5 - alpha / 2.0, 2.5 - alpha / 2.0
        X = numpy.array([0.0, 1e-3, 0.1, 1.0, 4.0, 16.0, 64.0, 256.0])
        ref = scipy_special.hyp1f1(a, b, -X)
        got = as_numpy(gsp.hyp1f1(a, b, _asarray(backend, -X)))
        rtol = 0.0 if backend == "numpy" else 1e-9  # b=a+1 -> exact via gammainc
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_elliptic_value_parity(backend):
    m = numpy.array([0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    for fn, sp_fn in [
        (gsp.ellipk, scipy_special.ellipk),
        (gsp.ellipe, scipy_special.ellipe),
    ]:
        ref = sp_fn(m)
        got = as_numpy(fn(_asarray(backend, m)))
        rtol = 0.0 if backend == "numpy" else 1e-12  # AGM is ~1e-15
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_tier2_grad_vs_fd(backend):
    # differentiate each Tier-2 fn at a smooth interior point vs central FD
    eps = 1e-6
    specs = [
        ("hyp2f1", lambda zz: gsp.hyp2f1(2.0, 2.0, 3.0, zz),
         lambda x: scipy_special.hyp2f1(2.0, 2.0, 3.0, x), -3.0),
        ("hyp1f1", lambda zz: gsp.hyp1f1(1.5, 2.5, zz),
         lambda x: scipy_special.hyp1f1(1.5, 2.5, x), -2.0),
        ("ellipk", lambda mm: gsp.ellipk(mm), scipy_special.ellipk, 0.4),
        ("ellipe", lambda mm: gsp.ellipe(mm), scipy_special.ellipe, 0.4),
    ]  # fmt: skip
    for name, fn, sp_fn, x0 in specs:
        fd = (float(sp_fn(x0 + eps)) - float(sp_fn(x0 - eps))) / (2 * eps)
        if backend == "jax":
            ad = float(jax.grad(lambda x: fn(x))(jnp.asarray(x0)))
        else:
            xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            fn(xt).backward()
            ad = float(xt.grad)
        assert not numpy.isnan(ad), f"NaN grad for {name} ({backend})"
        numpy.testing.assert_allclose(
            ad, fd, rtol=1e-5, err_msg=f"{name} grad ({backend})"
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_jax_native_hyp2f1_hyp1f1_are_unreliable():
    # Tripwire / justification for _NATIVE_UNRELIABLE: jax's native hyp2f1 and
    # hyp1f1 are catastrophically wrong for z < -1 (galpy's regime). If this
    # ever starts PASSING (jax fixed them), move them to native in the router.
    import jax.scipy.special as jsp

    z = jnp.asarray(-50.0)
    bad2 = float(jsp.hyp2f1(2.0, 2.0, 3.0, z))
    ref2 = float(scipy_special.hyp2f1(2.0, 2.0, 3.0, -50.0))
    bad1 = float(jsp.hyp1f1(1.5, 2.5, jnp.asarray(-64.0)))
    ref1 = float(scipy_special.hyp1f1(1.5, 2.5, -64.0))
    assert not numpy.isfinite(bad2) or abs(bad2 - ref2) > 1e-3 * abs(ref2), (
        "jax native hyp2f1 now accurate for z<-1; route it natively in the router"
    )
    assert abs(bad1 - ref1) > 1e-3 * abs(ref1), (
        "jax native hyp1f1 now accurate for z<-1; route it natively in the router"
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gamma_fallback_agrees_and_no_nan_grad(backend):
    # Exercises the pure-backend gamma fallback (torch); on jax it routes native,
    # but we still check parity + finite gradient across the reflection at x<0.
    pts = _GAMMA_X
    got = as_numpy(gsp.gamma(_asarray(backend, pts)))
    numpy.testing.assert_allclose(got, scipy_special.gamma(pts), rtol=1e-11, atol=1e-11)
    # gradient at a negative non-integer (reflection branch) must be finite + correct
    x0 = -1.5
    eps = 1e-6
    fd = (
        float(scipy_special.gamma(x0 + eps)) - float(scipy_special.gamma(x0 - eps))
    ) / (2 * eps)
    if backend == "jax":
        ad = float(jax.grad(lambda x: gsp.gamma(x))(jnp.asarray(x0)))
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.gamma(xt).backward()
        ad = float(xt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-4)


# --- Direct fallback-implementation tests (paths the router rarely reaches) ----
def _ns(backend):
    from galpy.backend import get_namespace

    return get_namespace(_asarray(backend, 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_xlogy_fallback_direct(backend):
    # xlogy is native on every backend, so the router never reaches the fallback;
    # exercise (and validate vs scipy) the pure-backend implementation directly,
    # including the 0*log(0)=0 convention.
    from galpy.backend.special._fallback.xlogy import xlogy_fallback

    x = numpy.array([0.0, 0.0, 1.0, 2.5, 0.3])
    y = numpy.array([0.0, 5.0, 2.0, 0.7, 10.0])
    ref = scipy_special.xlogy(x, y)
    got = as_numpy(
        xlogy_fallback(_ns(backend), _asarray(backend, x), _asarray(backend, y))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_fallback_alt_labeling(backend):
    # 2F1 whose Euler labeling has to come from b, i.e. the SECOND branch of
    # _euler_labeling. The case used to be (2.0, 0.8, 2.5), chosen when the
    # admissibility test was c - P >= 1: c-a was 0.5 so a was refused. The test
    # is now c - P > 0, which a satisfies, so that case silently moved to the
    # FIRST branch and stopped covering this one -- it kept passing, because
    # either labeling gives the right value. a < 0 is what disqualifies a for
    # good, whatever the bound on c - P.
    a, b, c = -1.0, 0.8, 2.5  # a < 0 -> labeling must pick b
    z = -numpy.array([0.0, 0.1, 1.0, 5.0, 30.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-14, atol=1e-15)


# Parameter sets the Euler integral cannot take directly. galpy's own anisotropic
# DFs request all three (measured by instrumenting the fallback over
# test_sphericaldf), and before the transformation/series routes existed each one
# raised NotImplementedError.
_HYP2F1_EULER_TRANSFORMED = [
    (-3.2, 4.4, 5.2),  # only positive parameter has c-b = 0.8 < 1
    (2.0, 2.0, 2.5),  # c-a = c-b = 0.5 < 1, both positive
]
_HYP2F1_BOTH_NONPOSITIVE = [
    (-0.98, -0.04, 2.98),  # |a-b| = 0.94
    (-0.51, -0.98, 2.51),  # |a-b| = 0.47
    (-0.5, -0.5, 2.51),  # a == b: |a-b| = 0, the series' worst conditioning
]


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize(
    "a,b,c", _HYP2F1_EULER_TRANSFORMED, ids=[str(x) for x in _HYP2F1_EULER_TRANSFORMED]
)
def test_hyp2f1_fallback_euler_transformed(backend, a, b, c):
    # Euler's transformation, 2F1(a,b;c;z) = (1-z)^(c-a-b) 2F1(c-a,c-b;c;z),
    # leaves z alone, so the quadrature's z<=0 machinery applies verbatim to the
    # transformed parameters. Accuracy is therefore the quadrature's own: this
    # measures 1e-14 across the whole grid, so 1e-12 is a real bound, not a
    # smoke check. Includes z=0 (where 2F1=1 exactly) and r/a=500.
    z = -numpy.array([0.0, 1e-3, 0.06, 0.617, 1.0, 5.0, 50.0, 500.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize(
    "a,b,c", _HYP2F1_BOTH_NONPOSITIVE, ids=[str(x) for x in _HYP2F1_BOTH_NONPOSITIVE]
)
def test_hyp2f1_fallback_both_parameters_nonpositive(backend, a, b, c):
    # With a and b both non-positive there is no admissible Euler labeling under
    # ANY of the four standard transformations, so this routes to the Pfaff
    # series. Over |z| <= 20 that is exact to double precision even for a == b
    # (measured worst case 9e-13), which is where galpy's DFs actually call it
    # (|z| < 1). The looser large-|z| behaviour is pinned separately below so
    # this bound stays tight enough to catch a real regression.
    z = -numpy.array([0.0, 1e-3, 0.06, 0.617, 1.0, 5.0, 20.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_series_route_degrades_but_stays_bounded(backend):
    # The counterpart: past |z| ~ 50 the series route falls off, because its
    # convergence is algebraic in the term count rather than spectral. Pin the
    # documented behaviour (2e-6 at |z|=50, see _SERIES_TERMS) so that neither a
    # silent accuracy loss nor a divergence goes unnoticed.
    a, b, c = -0.51, -0.98, 2.51
    z = -numpy.array([50.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    rel = numpy.max(numpy.fabs(got / ref - 1.0))
    assert rel < 1e-5, f"series route worse than documented at |z|=50: {rel:.2e}"
    assert rel > 1e-9, (
        "series route is now MORE accurate than documented at |z|=50 "
        f"({rel:.2e}); if _SERIES_TERMS grew, refresh the table in its comment"
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize(
    "a,b,c",
    _HYP2F1_EULER_TRANSFORMED + _HYP2F1_BOTH_NONPOSITIVE,
    ids=[str(x) for x in _HYP2F1_EULER_TRANSFORMED + _HYP2F1_BOTH_NONPOSITIVE],
)
def test_hyp2f1_new_routes_grad_vs_fd(backend, a, b, c):
    # Both new routes must differentiate, not merely evaluate: the DFs that need
    # them are consumed by gradient-based work. Central differences on scipy is
    # the reference, so this checks the derivative of the RIGHT function and not
    # just self-consistency of the backend graph.
    x0, eps = 0.617, 1e-6

    def f_np(w):
        return float(scipy_special.hyp2f1(a, b, c, -w))

    fd = (f_np(x0 + eps) - f_np(x0 - eps)) / (2 * eps)
    if backend == "jax":
        ad = float(jax.grad(lambda w: gsp.hyp2f1(a, b, c, -w))(jnp.asarray(x0)))
    else:
        wt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.hyp2f1(a, b, c, -wt).backward()
        ad = float(wt.grad)
    assert not numpy.isnan(ad), f"gradient is NaN for ({a}, {b}, {c})"
    numpy.testing.assert_allclose(ad, fd, rtol=1e-6)


def test_fallback_unsupported_regimes_raise():
    # The fallbacks raise (rather than silently return a low-accuracy value)
    # outside the regime galpy needs and they are accurate in.
    from galpy.backend.special._fallback.hyp1f1 import hyp1f1_fallback
    from galpy.backend.special._fallback.hyp2f1 import _euler_labeling

    # 2F1 outside the Euler integral's domain. The bound used to be
    # c - P >= 1 and (2.0, 2.0, 2.5) sat just under it; with the tanh-sinh rule
    # the requirement is only what the Beta integral needs, c - P > 0, and that
    # case is now perfectly well supported. Both ways of being outside are
    # checked, because they fail for different reasons:
    with pytest.raises(NotImplementedError):
        _euler_labeling(2.0, 3.0, 1.5)  # c - P <= 0 for both
    with pytest.raises(NotImplementedError):
        _euler_labeling(-1.0, -2.0, 3.0)  # no positive P at all
    # 1F1 only implements b = a + 1
    with pytest.raises(NotImplementedError):
        hyp1f1_fallback(numpy, 1.0, 3.0, numpy.array([-1.0]))


# --- Tier 3: modified Bessel functions of the second kind (k0, k1, kn) --------
# RazorThinExponentialDisk forces use k0/k1/kn(2,.) on real x > 0.
_BESSEL_X = numpy.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 7.0, 20.0, 80.0, 300.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_bessel_k_value_parity(backend):
    x = _BESSEL_X
    for name, fn, sp_fn in [
        ("k0", gsp.k0, scipy_special.k0),
        ("k1", gsp.k1, scipy_special.k1),
    ]:
        ref = sp_fn(x)
        got = as_numpy(fn(_asarray(backend, x)))
        rtol = 0.0 if backend == "numpy" else 1e-12  # series + scaled-trapezoid
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13, err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bessel_kn_value_parity(backend):
    # kn via the upward recurrence from k0, k1 (galpy uses kn(2, .)). n=0,1
    # exercise the recurrence base cases (kn_fallback short-circuits to K0/K1).
    x = _BESSEL_X
    for n in (0, 1, 2, 3, 5):
        ref = scipy_special.kn(n, x)
        got = as_numpy(gsp.kn(n, _asarray(backend, x)))
        rtol = 0.0 if backend == "numpy" else 1e-11  # recurrence amplifies a touch
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13, err_msg=f"kn{n}")


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_bessel_k_grad_vs_fd(backend):
    # K0'(x) = -K1(x); K1'(x) = -K0(x) - K1(x)/x. Check AD vs central FD on both
    # the series (x<2) and the scaled-trapezoid (x>2) branches.
    eps = 1e-6
    for name, fn, sp_fn in [
        ("k0", gsp.k0, scipy_special.k0),
        ("k1", gsp.k1, scipy_special.k1),
    ]:
        for x0 in (0.7, 1.5, 4.0, 25.0):
            fd = (float(sp_fn(x0 + eps)) - float(sp_fn(x0 - eps))) / (2 * eps)
            if backend == "jax":
                ad = float(jax.grad(lambda xx: fn(xx))(jnp.asarray(x0)))
            else:
                xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
                fn(xt).backward()
                ad = float(xt.grad)
            assert not numpy.isnan(ad), f"NaN grad for {name} at x={x0} ({backend})"
            numpy.testing.assert_allclose(
                ad, fd, rtol=1e-5, err_msg=f"{name} grad at x={x0} ({backend})"
            )


@pytest.mark.skipif("torch" not in BACKENDS, reason="needs torch")
def test_torch_native_bessel_k_is_nondifferentiable():
    # Tripwire / justification for using the fallback on torch: torch.special has
    # modified_bessel_k0/k1 (accurate) but they have no autograd backward.
    xt = torch.tensor(2.5, dtype=torch.float64, requires_grad=True)
    out = torch.special.modified_bessel_k0(xt)
    assert not out.requires_grad, (
        "torch.special.modified_bessel_k0 is now differentiable; it can be routed "
        "natively (with a k0/k1 name alias) instead of via the fallback"
    )


# --- Tier 3b: exponential integral E_1 (ExpTruncNFWPotential closed forms) -----
# numpy uses scipy; jax uses -expi(-x) (its native exp1 = expn is not twice-
# differentiable); torch uses the series/Lentz fallback. galpy evaluates E_1 on
# beta = (a+r)/rc > 0, spanning the small-x series and large-x CF regimes.
# spans the E1 domain reached by ExpTruncNFW: alpha=a/rc can be tiny (<1e-3 for
# a<<rc) and beta=(a+r)/rc grows large at large r -- test both tails plus the
# series<->continued-fraction crossover at x~1. E1(150)~5e-68 is still float64.
_EXP1_X = numpy.array(
    [1e-5, 1e-4, 1e-3, 0.05, 0.4, 0.9, 1.0, 1.1, 2.5, 8.0, 25.0, 60.0, 100.0, 150.0]
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exp1_value_parity(backend):
    ref = scipy_special.exp1(_EXP1_X)
    got = as_numpy(gsp.exp1(_asarray(backend, _EXP1_X)))
    rtol = 0.0 if backend == "numpy" else 1e-10  # jax native / torch Lentz+series
    numpy.testing.assert_allclose(
        got, ref, rtol=rtol, atol=1e-13, err_msg=f"exp1 ({backend})"
    )
    # E_1(inf) = 0 must hold on every backend (potential-at-infinity / total mass)
    assert float(as_numpy(gsp.exp1(_asarray(backend, numpy.inf)))) == 0.0


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_exp1_grad_vs_fd(backend):
    # E_1'(x) = -e^{-x}/x. Check AD (jax -expi / torch fallback) vs central FD on
    # both the series (x < 1) and continued-fraction (x > 1) branches.
    eps = 1e-6
    for x0 in (0.4, 0.9, 1.1, 2.5, 8.0):
        fd = (
            float(scipy_special.exp1(x0 + eps)) - float(scipy_special.exp1(x0 - eps))
        ) / (2 * eps)
        if backend == "jax":
            ad = float(jax.grad(lambda x: gsp.exp1(x))(jnp.asarray(x0)))
        else:
            xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            gsp.exp1(xt).backward()
            ad = float(xt.grad)
        assert not numpy.isnan(ad), f"NaN grad at x={x0} ({backend})"
        numpy.testing.assert_allclose(
            ad, fd, rtol=1e-5, err_msg=f"exp1' at x={x0} ({backend})"
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_exp1_nested_grad(backend):
    # The constantbetadf half-integer DFs take a SECOND derivative through E_1, so
    # the router's exp1 must be twice-differentiable. E_1''(x) = e^{-x}(1/x+1/x^2).
    for x0 in (0.4, 2.5, 8.0):
        exact = numpy.exp(-x0) * (1.0 / x0 + 1.0 / x0**2)
        if backend == "jax":
            ad2 = float(jax.grad(jax.grad(lambda x: gsp.exp1(x)))(jnp.asarray(x0)))
        else:
            xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            g = torch.autograd.grad(gsp.exp1(xt), xt, create_graph=True)[0]
            ad2 = float(torch.autograd.grad(g, xt)[0])
        numpy.testing.assert_allclose(
            ad2, exact, rtol=1e-6, err_msg=f"exp1'' at x={x0} ({backend})"
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="needs jax")
def test_jax_native_exp1_second_deriv_unreliable():
    # Tripwire / justification for routing jax's exp1 through -expi(-x): jax's
    # native exp1 is implemented as expn(1, x), whose SECOND derivative (needed by
    # the half-integer constant-beta DFs) fails to trace. If this ever stops
    # raising, jax's native exp1 could be routed directly in the router.
    import jax.scipy.special as jsp

    with pytest.raises(Exception):
        jax.grad(jax.grad(lambda x: jsp.exp1(x)))(jnp.asarray(2.5))


# --- Tier 4a: associated Legendre P_l^m (SCF / MultipoleExpansion) -------------
def _scipy_assoc_ref(L, M, x, deriv):
    arr = numpy.asarray(
        scipy_special.assoc_legendre_p_all(
            L - 1, M - 1, numpy.asarray(x, dtype=float), branch_cut=2, diff_n=deriv
        )
    )
    return numpy.moveaxis(arr[:, :, :M], (1, 2), (-2, -1))  # (deriv+1, *x.shape, L, M)


@pytest.mark.parametrize("backend", BACKENDS)
def test_assoc_legendre_value_parity(backend):
    L, M = 7, 5
    x = numpy.array([0.2, 0.5, -0.6, 0.9, -0.95])  # cos(theta), |x| < 1
    ref = _scipy_assoc_ref(L, M, x, 2)  # P, dP/dx, d2P/dx2
    P, dP, d2 = gsp.assoc_legendre(L, M, _asarray(backend, x), deriv=2)
    if backend == "numpy":  # numpy must be byte-identical to scipy
        assert numpy.array_equal(as_numpy(P), ref[0])
        assert numpy.array_equal(as_numpy(dP), ref[1])
        assert numpy.array_equal(as_numpy(d2), ref[2])
    else:
        for got, r, nm in [(P, ref[0], "P"), (dP, ref[1], "dP"), (d2, ref[2], "d2P")]:
            numpy.testing.assert_allclose(
                as_numpy(got), r, rtol=1e-11, atol=1e-11, err_msg=nm
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_assoc_legendre_value_only_and_shape(backend):
    L, M = 5, 3
    x = numpy.array([0.3, -0.4])
    P = gsp.assoc_legendre(L, M, _asarray(backend, x))  # deriv=0 -> just P
    assert tuple(as_numpy(P).shape) == (2, L, M)
    numpy.testing.assert_allclose(
        as_numpy(P), _scipy_assoc_ref(L, M, x, 0)[0], rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_assoc_legendre_autodiff_matches_analytic(backend):
    # d/dx of P_l^m via autodiff must match the analytically-returned dP/dx.
    L, M = 6, 4
    x0 = 0.4
    _, dP_an = gsp.assoc_legendre(L, M, _asarray(backend, x0), deriv=1)
    dP_an = as_numpy(dP_an)
    for ll, mm in [(3, 2), (5, 1), (4, 0), (5, 3)]:
        if backend == "jax":
            g = float(
                jax.grad(lambda xx: gsp.assoc_legendre(L, M, xx)[ll, mm])(
                    jnp.asarray(x0)
                )
            )
        else:
            xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
            gsp.assoc_legendre(L, M, xt)[ll, mm].backward()
            g = float(xt.grad)
        numpy.testing.assert_allclose(
            g, dP_an[ll, mm], rtol=1e-6, err_msg=f"P_{ll}^{mm}"
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="lax.scan Bonnet roll is jax-only")
def test_assoc_legendre_traced_bonnet_scan():
    # Under a jax trace the value Bonnet l-recurrence is rolled into ONE lax.scan
    # (_bonnet_scan_jax) and the derivative tables are computed by vectorized ops
    # over static (l, m) grids (_legendre_dP_jax/_legendre_d2_jax); eager
    # numpy/torch/jax keep the unrolled Python loops (byte-identical, verified by
    # the parity tests above + the origin byte-diff harness). This asserts: (a) the
    # value roll fires (scan primitive; small deriv=0 graph) and the derivative
    # tables are rolled (deriv=1/2 graphs an order of magnitude below the unrolled
    # loop), (b) the traced result matches the eager loop to the XLA fma floor for
    # every deriv level, (c) parity holds AT the poles x=+-1 (m==0 finite entries
    # exact, m>=1 NaN positions match), and (d) grad through the traced derivative
    # tables matches central FD.
    L, M = 10, 5
    x = jnp.asarray(numpy.linspace(-0.9, 0.9, 7), dtype=jnp.float64)

    # (a) rolled: scan primitive present; deriv=0 graph small (eager unrolls to
    #     ~200 eqns) and the deriv=1/2 derivative-table graphs are far below the
    #     eager-unrolled ~600/~1000 eqns.
    def _neqns(d):
        def f(xx):
            out = gsp.assoc_legendre(L, M, xx, deriv=d)
            return out if d == 0 else sum(o.sum() for o in out)

        return len(jax.make_jaxpr(f)(x).jaxpr.eqns)

    eqns0 = jax.make_jaxpr(lambda xx: gsp.assoc_legendre(L, M, xx))(x).jaxpr.eqns
    assert any(e.primitive.name == "scan" for e in eqns0), "lax.scan roll did not fire"
    assert len(eqns0) < 60, f"deriv=0 jaxpr not rolled ({len(eqns0)} eqns)"
    assert _neqns(1) < 150, "deriv=1 derivative table not rolled"
    assert _neqns(2) < 250, "deriv=2 derivative table not rolled"

    # (b) traced (jit) vs eager (Python loop) to the XLA fma floor, all deriv.
    for d in (0, 1, 2):
        eager = gsp.assoc_legendre(L, M, x, deriv=d)
        traced = jax.jit(lambda xx, d=d: gsp.assoc_legendre(L, M, xx, deriv=d))(x)
        eager = eager if isinstance(eager, tuple) else (eager,)
        traced = traced if isinstance(traced, tuple) else (traced,)
        for e, t in zip(eager, traced):
            numpy.testing.assert_allclose(
                as_numpy(t), as_numpy(e), rtol=1e-6, atol=1e-6
            )

    # (c) parity at the poles x=+-1: m==0 derivative columns are finite (exact
    #     closed form) and the m>=1 NaN pattern matches the eager loop.
    xp1 = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    for d in (1, 2):
        eager = gsp.assoc_legendre(L, M, xp1, deriv=d)
        traced = jax.jit(lambda xx, d=d: gsp.assoc_legendre(L, M, xx, deriv=d))(xp1)
        for e, t in zip(eager, traced):
            en, tn = as_numpy(e), as_numpy(t)
            assert numpy.array_equal(numpy.isnan(en), numpy.isnan(tn))  # NaN pattern
            fin = ~numpy.isnan(en)
            assert numpy.array_equal(tn[fin], en[fin])  # finite entries bit-exact

    # (d) grad-vs-FD through the traced derivative tables (random direction,
    #     quadratic loss mixing P, dP and d2), interior x (no poles).
    x0 = jnp.asarray(numpy.linspace(-0.75, 0.8, 7), dtype=jnp.float64)
    dirn = jax.random.normal(jax.random.PRNGKey(1), x0.shape, dtype=jnp.float64)
    w = jnp.asarray(numpy.arange(1, L * M + 1, dtype=float)).reshape(L, M)

    def loss(t):
        P, dP, d2 = gsp.assoc_legendre(L, M, x0 + t * dirn, deriv=2)
        return jnp.sum(w * dP**2) + jnp.sum(d2[..., 5, 1] ** 2) + jnp.sum(P[..., 4, 2])

    g = float(jax.grad(loss)(0.0))
    errs = [abs((float(loss(h)) - float(loss(-h))) / (2 * h) - g) for h in (1e-4, 1e-6)]
    assert errs[1] < errs[0], f"FD did not converge to the grad: {errs}"
    numpy.testing.assert_allclose(
        (float(loss(1e-6)) - float(loss(-1e-6))) / 2e-6, g, rtol=1e-5
    )


# --- Tier 4b: Gegenbauer C_n^alpha (SCF radial basis) -------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_gegenbauer_value_parity(backend):
    N = 8
    x = numpy.array([-0.9, -0.3, 0.0, 0.4, 0.95])
    for alpha in (1.5, 3.5, 2 * 3 + 1.5):  # SCF uses alpha = 2l + 3/2
        got = as_numpy(gsp.gegenbauer(N, alpha, _asarray(backend, x)))
        assert got.shape == x.shape + (N,)
        for n in range(N):
            ref = scipy_special.eval_gegenbauer(n, alpha, x)
            numpy.testing.assert_allclose(
                got[..., n], ref, rtol=1e-11, atol=1e-12, err_msg=f"C_{n}^{alpha}"
            )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gegenbauer_grad_vs_fd(backend):
    # d/dx C_n^alpha(x) = 2 alpha C_{n-1}^{alpha+1}(x); check AD vs central FD.
    N, alpha, x0 = 6, 2.5, 0.4
    eps = 1e-6
    n = 4
    fd = (
        float(scipy_special.eval_gegenbauer(n, alpha, x0 + eps))
        - float(scipy_special.eval_gegenbauer(n, alpha, x0 - eps))
    ) / (2 * eps)
    if backend == "jax":
        ad = float(
            jax.grad(lambda xx: gsp.gegenbauer(N, alpha, xx)[n])(jnp.asarray(x0))
        )
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.gegenbauer(N, alpha, xt)[n].backward()
        ad = float(xt.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.skipif(
    "jax" not in BACKENDS, reason="lax.scan Gegenbauer roll is jax-only"
)
def test_gegenbauer_traced_scan():
    # Under a jax trace the n-recurrence is rolled into ONE lax.scan
    # (_gegenbauer_scan_jax); eager numpy/torch/jax keep the unrolled Python loop
    # (byte-identical, covered by the parity tests above). Assert: (a) the scan
    # fires and the jaxpr is small AND N-independent (the eager loop unrolls O(N)),
    # (b) traced == eager to the XLA fma floor across N, (c) grad through the scan
    # matches the eager grad.
    alpha = 2 * 2 + 1.5  # SCF radial basis uses alpha = 2l + 3/2
    x = jnp.asarray(numpy.linspace(-0.9, 0.9, 7), dtype=jnp.float64)

    # (a) rolled: scan primitive present and the jaxpr is small + N-independent
    #     (the eager loop would grow ~O(N)).
    e10 = jax.make_jaxpr(lambda xx: gsp.gegenbauer(10, alpha, xx))(x).jaxpr.eqns
    e24 = jax.make_jaxpr(lambda xx: gsp.gegenbauer(24, alpha, xx))(x).jaxpr.eqns
    assert any(e.primitive.name == "scan" for e in e10), "lax.scan roll did not fire"
    assert len(e10) < 30, f"jaxpr not rolled ({len(e10)} eqns)"
    assert len(e10) == len(e24), "rolled jaxpr must be N-independent"

    # (b) traced (jit) vs eager (Python loop) to the XLA fma floor, several N.
    for N in (5, 10, 24):
        eager = as_numpy(gsp.gegenbauer(N, alpha, x))
        traced = as_numpy(jax.jit(lambda xx, N=N: gsp.gegenbauer(N, alpha, xx))(x))
        numpy.testing.assert_allclose(traced, eager, rtol=1e-6, atol=1e-6)

    # (c) grad through the traced scan matches the eager grad.
    x0 = jnp.asarray(0.3)
    g_eager = float(jax.grad(lambda xx: gsp.gegenbauer(12, alpha, xx).sum())(x0))
    g_jit = float(jax.jit(jax.grad(lambda xx: gsp.gegenbauer(12, alpha, xx).sum()))(x0))
    numpy.testing.assert_allclose(g_jit, g_eager, rtol=1e-6, atol=1e-9)


# --- edge-case hardening: poles, domain endpoints, AD plateaus, numpy params ---
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_assoc_legendre_pole_derivs(backend):
    # m=0 x-derivatives are finite at the symmetry-axis poles x=+-1 (the
    # (x^2-1) denominators must not 0/0-NaN); match the numpy/scipy reference.
    L, M = 6, 1
    for xv in (1.0, -1.0):
        _, rdP, rd2 = gsp.assoc_legendre(L, M, numpy.asarray(xv), deriv=2)
        _, gdP, gd2 = gsp.assoc_legendre(L, M, _asarray(backend, xv), deriv=2)
        gdP, gd2 = as_numpy(gdP)[..., 0], as_numpy(gd2)[..., 0]
        assert not numpy.any(numpy.isnan(gdP)) and not numpy.any(numpy.isnan(gd2))
        numpy.testing.assert_allclose(gdP, rdP[..., 0], rtol=1e-10, atol=1e-12)
        numpy.testing.assert_allclose(gd2, rd2[..., 0], rtol=1e-10, atol=1e-12)


def test_scf_spherical_zaxis_forces_finite():
    # end-to-end: spherical SCF forces / 2nd-derivs on the z-axis (cos theta=+-1)
    # are finite and backend-consistent (regression for the assoc_legendre poles).
    from galpy.potential import (
        HernquistPotential,
        SCFPotential,
        scf_compute_coeffs_spherical,
    )

    Acos, Asin = scf_compute_coeffs_spherical(
        HernquistPotential(amp=2.0, a=1.0).dens, 6
    )
    scf = SCFPotential(Acos=Acos, Asin=Asin)
    for meth in ("Rforce", "zforce", "R2deriv", "z2deriv"):
        for z in (1.5, -1.5):
            ref = float(getattr(scf, meth)(0.0, z))
            for backend in AD_BACKENDS:
                got = as_numpy(
                    getattr(scf, meth)(_asarray(backend, 0.0), _asarray(backend, z))
                )
                assert numpy.isfinite(got), f"{meth}(0,{z}) {backend} not finite"
                numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ellipk_ellipe_negative_m_and_unit(backend):
    # scipy supports all real m<1 (incl. m<0); m=1 gives K=inf, E=1.
    m = numpy.array([-5.0, -1.0, -0.5, 0.0, 0.3, 0.9, 0.999])
    rtol = 0.0 if backend == "numpy" else 1e-9
    for fn, sp_fn in (
        (gsp.ellipk, scipy_special.ellipk),
        (gsp.ellipe, scipy_special.ellipe),
    ):
        got = as_numpy(fn(_asarray(backend, m)))
        numpy.testing.assert_allclose(got, sp_fn(m), rtol=rtol, atol=1e-12)
    assert numpy.isinf(as_numpy(gsp.ellipk(_asarray(backend, 1.0))))
    numpy.testing.assert_allclose(as_numpy(gsp.ellipe(_asarray(backend, 1.0))), 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ellipkm1_stays_exact_where_ellipk_cannot(backend):
    # K(m) near m=1 can only be reached through the complement: reconstructing
    # the argument as 1-m1 collapses to exactly 1.0 once m1 drops below an ulp,
    # and ellipk(1.0) is inf. ellipkm1 takes m1 directly, so it stays finite and
    # accurate all the way down. (This is how the razor-thin-disk integrands hit
    # it: their 1-m = ((a-R)^2+z^2)/((a+R)^2+z^2) vanishes as z -> 0 at a = R.)
    m1 = numpy.array([0.9, 0.5, 1e-3, 1e-9, 1e-15, 1e-16, 1e-20, 1e-100, 1e-300])
    got = as_numpy(gsp.ellipkm1(_asarray(backend, m1)))
    numpy.testing.assert_allclose(got, scipy_special.ellipkm1(m1), rtol=1e-13)
    assert numpy.all(numpy.isfinite(got))
    # the route this replaces really does fail on the same inputs
    assert numpy.isinf(scipy_special.ellipk(1.0 - m1[-1]))
    # and agrees with ellipk wherever ellipk is still usable
    numpy.testing.assert_allclose(
        as_numpy(gsp.ellipkm1(_asarray(backend, m1[:3]))),
        as_numpy(gsp.ellipk(_asarray(backend, 1.0 - m1[:3]))),
        rtol=1e-13,
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_ellipkm1_grad_matches_closed_form(backend):
    # dK/dm1 = -dK/dm = -(E(m) - (1-m) K(m)) / (2 m (1-m)), evaluated at m = 1-m1.
    for x in (0.3, 1e-3, 1e-6):
        m = 1.0 - x
        ref = -(scipy_special.ellipe(m) - x * scipy_special.ellipk(m)) / (2.0 * m * x)
        if backend == "jax":
            got = float(jax.grad(lambda t: gsp.ellipkm1(t))(jnp.asarray(x)))
        else:
            t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            gsp.ellipkm1(t).backward()
            got = float(t.grad)
        numpy.testing.assert_allclose(got, ref, rtol=1e-11)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_ellipe_negative_m_grad_finite(backend):
    # dE/dm = (E-K)/(2m) is finite for m<0 (sqrt(m) must not enter the E series).
    ref = (scipy_special.ellipe(-1.0) - scipy_special.ellipk(-1.0)) / (2.0 * -1.0)
    if backend == "jax":
        g = float(jax.grad(lambda m: gsp.ellipe(m))(jnp.asarray(-1.0)))
    else:
        mt = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
        gsp.ellipe(mt).backward()
        g = float(mt.grad)
    assert numpy.isfinite(g)
    numpy.testing.assert_allclose(g, ref, rtol=1e-6)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gamma_negative_integer_poles_nan(backend):
    # Gamma at -1, -2, ... is nan (scipy), not a huge finite reflection value;
    # Gamma(0) is +inf (a distinct scipy convention).
    for x in (-1.0, -2.0, -3.0):
        assert numpy.isnan(as_numpy(gsp.gamma(_asarray(backend, x))))
    assert numpy.isinf(as_numpy(gsp.gamma(_asarray(backend, 0.0))))
    xs = numpy.array([-2.5, -1.5, -0.5, 0.5, 2.5, 6.0])  # off-pole unchanged
    numpy.testing.assert_allclose(
        as_numpy(gsp.gamma(_asarray(backend, xs))), scipy_special.gamma(xs), rtol=1e-10
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_grad_at_zero_is_abc(backend):
    # d/dz 2F1(a,b;c;z)|_0 = a b / c (no zero-gradient plateau near z=0).
    a, b, c = 2.0, 2.0, 3.0
    if backend == "jax":
        ad = float(jax.grad(lambda z: gsp.hyp2f1(a, b, c, z))(jnp.asarray(0.0)))
    else:
        zt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        gsp.hyp2f1(a, b, c, zt).backward()
        ad = float(zt.grad)
    numpy.testing.assert_allclose(ad, a * b / c, rtol=1e-5)


@pytest.mark.skipif("torch" not in BACKENDS, reason="needs torch")
def test_router_promotes_numpy_scalar_params_torch():
    # numpy.float64 parameters (not only python scalars) must be promoted for
    # torch.special, so a numpy-typed potential parameter works on the torch path.
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    ref = scipy_special.gammainc(0.5, numpy.array([0.5, 1.0, 2.0]))
    for a in (0.5, numpy.float64(0.5)):
        got = gsp.gammainc(a, x)
        assert torch.is_tensor(got)
        numpy.testing.assert_allclose(got.detach().numpy(), ref, rtol=1e-12)
    from galpy.potential import PowerSphericalPotentialwCutoff as PSPC

    r = torch.tensor([1.0, 2.0], dtype=torch.float64)
    p_py = PSPC(alpha=1.3, rc=2.0)
    p_np = PSPC(alpha=numpy.float64(1.3), rc=2.0)
    assert torch.allclose(p_py._rforce(r), p_np._rforce(r))


# 0 < z < 1 was UNTESTED until 2026-08-11: every grid above uses z = -_HYP2F1_W,
# i.e. z <= 0. The fallback is built for z <= 0 and, applied to positive z, was
# returning the first-order Taylor series 1 + (ab/c)z -- 5.1e-03 wrong at z=0.1
# rising to 6.4e-01 at z=0.95, silently. galpy itself only passes z <= 0
# (TwoPowerSphericalPotential spans -15.8 .. -0.06) so nothing in-tree was
# affected, which is exactly why no test caught it. It now enters via Pfaff,
# 2F1(a,b;c;z) = (1-z)^{-a} 2F1(a, c-b; c; z/(z-1)), landing on the z <= 0 domain.
_HYP2F1_ZPOS = numpy.array([1e-8, 0.05, 0.1, 0.4, 0.8, 0.95, 0.999])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("a,b,c", _HYP2F1_CASES, ids=[str(x) for x in _HYP2F1_CASES])
def test_hyp2f1_positive_z_matches_scipy(backend, a, b, c):
    z = _HYP2F1_ZPOS
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    # Same bar as the z <= 0 parity test: Pfaff reuses that machinery, so the
    # accuracy is the same modulo the (1-z)^{-a} prefactor. Measured worst case
    # over this grid is 9.9e-07, on the one parameter set whose z <= 0 accuracy
    # is itself ~4e-06 (a pre-existing floor, not introduced by the transform),
    # so 1e-5 pins the transform without re-litigating that floor.
    rtol = 0.0 if backend == "numpy" else 1e-5
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-10)
    # And pin the specific failure mode: the old code returned the 2-term series,
    # so assert we are NOT that. Without this the test would still pass if a
    # future change silently reverted to a low-order truncation at small z.
    if backend != "numpy":
        two_term = 1.0 + (a * b / c) * z
        assert numpy.max(numpy.abs(got - two_term)) > 1e-3, (
            "hyp2f1 looks like the first-order Taylor series again"
        )


# a-grid that CROSSES the a ~ 20 boundary. The pre-existing _A tops out at 3.5,
# which is why torch's ~6-digit loss above a ~ 21 went unnoticed for so long:
# the whole failing regime was outside the grid.
_A_LARGE = numpy.array([12.0, 21.0, 30.0, 60.0, 150.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_gammainc_large_order_value_parity(backend):
    # Crosses a = 20, where torch.special.gammaincc switches algorithm and loses
    # ~6 digits (|dQ| ~ 5e-10 for a >= 21 vs ~1e-16 below). galpy keeps the
    # native FORWARD on torch -- the accurate series/CF is ~485x slower on a
    # scalar and MWPotential2014 calls this constantly -- so this pins native's
    # own accuracy, not scipy's. Tightening it below 1e-8 would be asserting a
    # precision galpy does not currently buy on this path; see the module
    # docstring in _fallback/gammainc.py.
    for sp_fn, fn in [
        (scipy_special.gammainc, gsp.gammainc),
        (scipy_special.gammaincc, gsp.gammaincc),
    ]:
        for a in _A_LARGE:
            x = numpy.array([0.3 * a, 0.7 * a, a, 1.4 * a])
            ref = sp_fn(a, x)
            got = as_numpy(fn(_asarray(backend, a), _asarray(backend, x)))
            rtol = 0.0 if backend == "numpy" else 1e-8
            numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-300)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gammainc_grad_wrt_order(backend):
    # d/da is the capability the fallback adds: torch has no igamma/igammac
    # derivative w.r.t. the order at all ("the derivative for 'igamma: input'
    # is not implemented"), so this raises without it. Checked against a
    # central difference taken on the scipy path, at h chosen for the ~1e-10
    # truncation floor of a first-order FD.
    a0, x0, h = 3.0, 2.0, 1e-5
    for sp_fn, fn in [
        (scipy_special.gammainc, gsp.gammainc),
        (scipy_special.gammaincc, gsp.gammaincc),
    ]:
        fd = (sp_fn(a0 + h, x0) - sp_fn(a0 - h, x0)) / (2 * h)
        if backend == "jax":
            ad = float(jax.grad(lambda a: fn(a, jnp.asarray(x0)))(jnp.asarray(a0)))
        else:
            at = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
            fn(at, torch.tensor(x0, dtype=torch.float64)).backward()
            ad = float(at.grad)
        assert not numpy.isnan(ad)
        numpy.testing.assert_allclose(ad, fd, rtol=1e-8)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gammainc_grad_wrt_argument_still_works(backend):
    # Guard the other derivative: the fallback must not regress d/dx, which the
    # native torch path already had. dP/dx = x^(a-1) e^-x / Gamma(a), exactly.
    a0, x0 = 3.0, 2.0
    exact = x0 ** (a0 - 1.0) * numpy.exp(-x0) / scipy_special.gamma(a0)
    if backend == "jax":
        ad = float(
            jax.grad(lambda x: gsp.gammainc(jnp.asarray(a0), x))(jnp.asarray(x0))
        )
    else:
        xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        gsp.gammainc(torch.tensor(a0, dtype=torch.float64), xt).backward()
        ad = float(xt.grad)
    numpy.testing.assert_allclose(ad, exact, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gammainc_endpoints_zero_and_infinity(backend):
    # x = inf is a REAL argument: the potential at r = inf and the total mass
    # both reach it, and an unguarded Lentz recurrence returns NaN there
    # (b = inf -> d = 0, then h *= d*c = 0*inf). x = 0 is the other endpoint.
    # Exact values, so compare exactly rather than with a tolerance.
    for a in (0.5, 1.4, 3.0, 30.0):
        xs = numpy.array([0.0, numpy.inf])
        p = as_numpy(gsp.gammainc(_asarray(backend, a), _asarray(backend, xs)))
        q = as_numpy(gsp.gammaincc(_asarray(backend, a), _asarray(backend, xs)))
        assert not numpy.any(numpy.isnan(p)), f"gammainc NaN at an endpoint, a={a}"
        assert not numpy.any(numpy.isnan(q)), f"gammaincc NaN at an endpoint, a={a}"
        numpy.testing.assert_array_equal(p, [0.0, 1.0])
        numpy.testing.assert_array_equal(q, [1.0, 0.0])


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_gammainc_grad_wrt_argument_at_endpoints(backend):
    # x = 0 is a REAL evaluation point (the value test above pins P(a,0)=0), and
    # dP/dx = x^(a-1) e^-x / Gamma(a) is computed as prefix(a,x)/x -- which is
    # 0/0 there and returned NaN before this was guarded. The limit depends on a:
    #     a < 1 -> +inf,   a = 1 -> 1,   a > 1 -> 0.
    # NB jax's own native gammainc returns NaN for d/dx at a=1, x=0, so the
    # reference here is the analytic limit, not another library.
    for a0, want in ((0.5, numpy.inf), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0)):
        if backend == "jax":
            if a0 == 1.0:
                continue  # jax's native path is NaN here; nothing of ours to pin
            ad = float(
                jax.grad(lambda x: gsp.gammainc(jnp.asarray(a0), x))(jnp.asarray(0.0))
            )
        else:
            xt = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            gsp.gammainc(torch.tensor(a0, dtype=torch.float64), xt).backward()
            ad = float(xt.grad)
        assert not numpy.isnan(ad), f"d/dx gammainc(a={a0}) is NaN at x=0"
        if numpy.isinf(want):
            assert numpy.isinf(ad) and ad > 0, f"a={a0}: want +inf, got {ad}"
        else:
            numpy.testing.assert_allclose(ad, want, rtol=0, atol=1e-15)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_gammainc_grad_with_an_integer_order_dtype():
    # The x=0 endpoint limit must be built in the RESULT dtype, not with
    # *_like(a): callers legitimately pass an INTEGER order -- EinastoPotential
    # does, via its 3n/dn arguments -- and torch.full_like(<int tensor>, inf)
    # raises "value cannot be converted to type int64_t without overflow".
    # The first cut of the endpoint guard used full_like(a, inf) and broke
    # every Einasto torch gradient; the value tests never saw it because they
    # only ever passed a float order.
    for a_int, a_flt in ((3, 3.0), (1, 1.0), (2, 2.0)):
        ai = torch.tensor(a_int)  # int64 on purpose
        af = torch.tensor(a_flt, dtype=torch.float64)
        # away from the endpoint: integer and float order must agree exactly
        gi, gf = [], []
        for a in (ai, af):
            x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
            gsp.gammainc(a, x).backward()
            (gi if a is ai else gf).append(float(x.grad))
        numpy.testing.assert_allclose(gi[0], gf[0], rtol=0, atol=0)
        # ...and AT the endpoint, where the limit branch actually runs
        x0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        gsp.gammainc(ai, x0).backward()
        want = 1.0 if a_int == 1 else 0.0  # a >= 1 here, so no +inf case
        numpy.testing.assert_allclose(float(x0.grad), want, rtol=0, atol=1e-15)
    # a = 1 at x = 0.5 has a closed form: P(1,x) = 1 - e^-x, so dP/dx = e^-x.
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    gsp.gammainc(torch.tensor(1), x).backward()
    numpy.testing.assert_allclose(float(x.grad), numpy.exp(-0.5), rtol=1e-14)


# --- hyp2f1 with BACKEND (a, b, c): differentiable in the parameters ----------
# Until now the fallback pinned its parameters to Python floats: math.lgamma for
# the Gamma prefactor and float(math.ceil(...)) for the substitution exponent. On
# jax that raised (TracerBoolConversion / Concretization) as soon as an exponent
# was traced. On torch it did NOT raise -- __float__ ran and silently DETACHED
# the prefactor, so the gradient came back finite, with requires_grad=True and a
# grad_fn, and simply wrong: measured d/da 2F1(a,2;3;-5) = -8.98e-03 against a
# true -8.56e-02, and TwoPowerTriaxial dPhi/dalpha 0.834 against 0.548. Only
# grad-vs-finite-difference catches that, which is what these do.
#
# The bar is per route, because the routes do not differentiate equally well.
# The value is spectrally accurate only when c - B is an integer (then the
# (1-t)^{c-B-1} factor is a polynomial and Gauss-Legendre is exact); the
# DERIVATIVE integrand carries an extra log factor that nothing regularizes, so
# it converges algebraically. The numbers below are measured, not guessed.
_HYP2F1_PARAM_GRAD = [
    # (a, b, c, z, rtol, route)
    (-3.2, 4.4, 5.2, -5.0, 1e-8, "euler-transformed"),
    (-0.51, -0.98, 2.51, -5.0, 1e-8, "pfaff-series"),
    (1.7, 2.3, 3.4, -5.0, 1e-4, "euler, non-integer c-B"),
    (2.0, 2.0, 3.0, -5.0, 1e-3, "euler, c-a == 1 exactly (galpy's own force call)"),
    # tanh-sinh route (B < _TS_B_MAX). Without these the route had NO parameter-
    # gradient coverage at all, and it sizes its own node grid from B -- exactly
    # the shape that silently detaches if it reaches for float(B). It cannot:
    # under jax.grad the parameters have concrete truth values but are still
    # tracers, so the grid is chosen by comparisons only.
    (-1.010, 0.020, 3.010, -0.6, 1e-6, "tanh-sinh, the real constantbetadf B"),
    (-1.6, 0.2, 3.2, -2.0, 1e-7, "tanh-sinh, small B"),
]


def _hyp2f1_fd(a, b, c, z, idx, h=1e-5):
    """d/d(param idx) of SCIPY's 2F1 by central differences -- the reference."""
    up, dn = [a, b, c], [a, b, c]
    up[idx] += h
    dn[idx] -= h
    return (scipy_special.hyp2f1(*up, z) - scipy_special.hyp2f1(*dn, z)) / (2.0 * h)


@pytest.mark.parametrize(
    "a,b,c,z,rtol,route",
    _HYP2F1_PARAM_GRAD,
    ids=[x[-1] for x in _HYP2F1_PARAM_GRAD],
)
def test_hyp2f1_jax_parameter_gradient_matches_finite_difference(
    a, b, c, z, rtol, route
):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with use("jax", force=True):
        zb = jnp.asarray(z)
        for idx, name in ((0, "a"), (1, "b"), (2, "c")):
            g = float(
                jax.grad(lambda p: gsp.hyp2f1(p[0], p[1], p[2], zb))(
                    jnp.array([a, b, c])
                )[idx]
            )
            assert g == pytest.approx(_hyp2f1_fd(a, b, c, z, idx), rel=rtol), (
                f"d/d{name} on the {route} route"
            )


@pytest.mark.parametrize(
    "a,b,c,z,rtol,route",
    _HYP2F1_PARAM_GRAD,
    ids=[x[-1] for x in _HYP2F1_PARAM_GRAD],
)
def test_hyp2f1_torch_parameter_gradient_matches_finite_difference(
    a, b, c, z, rtol, route
):
    torch = pytest.importorskip("torch")

    with use("torch", force=True):
        zb = torch.tensor(z, dtype=torch.float64)
        for idx, name in ((0, "a"), (1, "b"), (2, "c")):
            p = torch.tensor([a, b, c], dtype=torch.float64, requires_grad=True)
            out = gsp.hyp2f1(p[0], p[1], p[2], zb)
            (grad,) = torch.autograd.grad(out, p)
            assert float(grad[idx]) == pytest.approx(
                _hyp2f1_fd(a, b, c, z, idx), rel=rtol
            ), f"d/d{name} on the {route} route"


@pytest.mark.parametrize(
    "a,b,c",
    _HYP2F1_CASES + _HYP2F1_EULER_TRANSFORMED + _HYP2F1_BOTH_NONPOSITIVE,
    ids=[
        str(x)
        for x in _HYP2F1_CASES + _HYP2F1_EULER_TRANSFORMED + _HYP2F1_BOTH_NONPOSITIVE
    ],
)
def test_hyp2f1_traced_parameters_reproduce_the_concrete_route(a, b, c):
    # With traced (a, b, c) the route can no longer be chosen with `if`, so all
    # of it -- regime test, Euler labelling, series labelling -- is selected with
    # where(). This asserts that selection lands on the same answer the Python
    # branches give, for every parameter set the concrete tests cover, i.e. all
    # three routes. The bar is the series' own truncation noise at |z| = 50,
    # where the two arms round differently over 512 cancelling terms.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    with use("jax", force=True):
        zb = jnp.asarray(-_HYP2F1_W)
        concrete = as_numpy(gsp.hyp2f1(a, b, c, zb))
        traced = as_numpy(
            jax.jit(lambda p: gsp.hyp2f1(p[0], p[1], p[2], zb))(jnp.array([a, b, c]))
        )
        numpy.testing.assert_allclose(traced, concrete, rtol=1e-5, atol=1e-13)
        # ... and both are the scipy answer to the concrete route's own accuracy
        ref = scipy_special.hyp2f1(a, b, c, -_HYP2F1_W)
        numpy.testing.assert_allclose(traced, ref, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp1f1_parameter_gradient_matches_finite_difference(backend):
    # Sibling of the 2F1 case above, same defect: math.gamma(a) ran through
    # __float__ on a backend a and silently detached the Gamma factor. torch
    # returned a gradient that was 42% wrong -- finite, requires_grad=True,
    # grad_fn set -- and jax raised ConcretizationTypeError. Routed, both agree
    # with scipy's finite difference to ~5e-10.
    a0, z0 = 1.7, -3.0
    h = 1e-6
    ref = (
        scipy_special.hyp1f1(a0 + h, a0 + h + 1.0, z0)
        - scipy_special.hyp1f1(a0 - h, a0 - h + 1.0, z0)
    ) / (2.0 * h)
    with use(backend, force=True):
        if backend == "jax":
            import jax
            import jax.numpy as jnp

            got = float(
                jax.grad(lambda av: gsp.hyp1f1(av, av + 1.0, jnp.asarray(z0)))(a0)
            )
        else:
            import torch

            a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
            out = gsp.hyp1f1(a, a + 1.0, torch.tensor(z0, dtype=torch.float64))
            (grad,) = torch.autograd.grad(out, a)
            got = float(grad)
    assert got == pytest.approx(ref, rel=1e-7)


# ---------------------------------------------------------------------------
# hyp2f1 small-B accuracy: the Euler integral's xi^k substitution needs
# k >= ~6/B to regularize the t^{B-1} endpoint, and k is capped at 12 because
# X = xi^k underflows above that. So for small B the endpoint is simply not
# regularized and plain Gauss-Legendre loses badly. tanh-sinh needs no
# substitution and is routed in below _TS_B_MAX.
#
# B is the parameter the Euler labelling PICKS: a < 0 disqualifies a, forcing
# B = b. That is not synthetic -- constantbetaHernquistdf(beta=-1.5) asks for
# (a, b, c) = (-1.010, 0.020, 3.010), where the shipping rule was 7.2e-02 off.
#
# Reference is scipy, itself checked against mpmath (dps=50) at these same
# parameters to 1.5e-16, so the arbiter is not the thing being tested.
# ---------------------------------------------------------------------------
_SMALL_B = [
    # (B, tolerance) -- tolerances are the MEASURED accuracy, not round numbers.
    # All four land at ~1e-15, so one bar covers them; they are kept as separate
    # cases because the SHIPPING rule degraded steeply across this range
    # (1.8e-11 -> 7.2e-02) and a single B would not show that it no longer does.
    (0.200, 1e-14),
    (0.100, 1e-14),
    (0.050, 1e-14),
    (0.020, 1e-14),  # was 7.2e-02 on the shipping rule
]


@pytest.mark.parametrize("B,tol", _SMALL_B, ids=[f"B={b}" for b, _ in _SMALL_B])
@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_small_euler_parameter_is_accurate(backend, B, tol):
    a, c = -1.010, B + 3.0  # a < 0 forces the labelling onto B = b
    for z in (-0.6, -2.0, -15.8):
        ref = scipy_special.hyp2f1(a, B, c, z)
        got = as_numpy(gsp.hyp2f1(a, B, c, _asarray(backend, z)))
        rel = abs(got / ref - 1.0)
        assert rel < tol, (
            f"{backend} hyp2f1({a},{B},{c},{z}): rel {rel:.3e} > {tol:.0e}"
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_the_real_constantbetadf_request(backend):
    # The exact parameter set constantbetaHernquistdf(beta=-1.5) requests,
    # recorded by instrumenting the fallback. Shipping rule: 7.18e-02 off.
    a, b, c, z = -1.010, 0.020, 3.010, -0.6
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    rel = abs(got / ref - 1.0)
    assert rel < 1e-14, f"{backend}: rel {rel:.3e} (was 7.18e-02 before the route)"


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_large_B_is_machine_precision(backend):
    # These are the large-B cases the Gauss-Legendre rule was kept for. They
    # used to need loose bars -- (-1.6, 1.2, 3.6) sat at 5.3e-08, which is why
    # this test once asserted 1e-08 and was named "..._route_is_unchanged".
    # With one tanh-sinh rule for every B they are at machine precision, so the
    # bars are tightened to the measured values: a regression here would mean
    # the single rule had given up something the old two-rule split had.
    for a, b, c, tol in (
        (-3.2, 4.4, 5.2, 1e-14),
        (2.0, 2.0, 2.5, 1e-14),
        (-1.6, 1.2, 3.6, 1e-14),
    ):
        for z in (-0.6, -2.0):
            ref = scipy_special.hyp2f1(a, b, c, z)
            got = as_numpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
            rel = abs(got / ref - 1.0)
            assert rel < tol, f"{backend} regressed at ({a},{b},{c},{z}): {rel:.3e}"
