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

from galpy.backend import special as gsp
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


def _asarray(backend, x, requires_grad=False):
    if backend == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


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
    got = _tonumpy(fn(_asarray(backend, pts)))
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
            got = _tonumpy(fn(_asarray(backend, a), _asarray(backend, _XG)))
            rtol = 0.0 if backend == "numpy" else 1e-12
            numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_xlogy_value_parity_incl_zero(backend):
    x = numpy.array([0.0, 0.0, 1.0, 2.5, 0.3])
    y = numpy.array([0.0, 5.0, 2.0, 0.7, 10.0])  # x=0 -> 0 even when y=0
    ref = scipy_special.xlogy(x, y)
    got = _tonumpy(gsp.xlogy(_asarray(backend, x), _asarray(backend, y)))
    rtol = 0.0 if backend == "numpy" else 1e-12
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)


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
        "hyp2f1", "hyp1f1", "ellipk", "ellipe", "k0", "k1", "kn",
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
    got = _tonumpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    rtol = 0.0 if backend == "numpy" else 1e-9  # fallback quadrature at r/a<~50
    numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
@pytest.mark.parametrize("a,b,c", _HYP2F1_CASES, ids=[str(x) for x in _HYP2F1_CASES])
def test_hyp2f1_extreme_z_bounded_error(backend, a, b, c):
    # Far beyond realistic radii (r/a up to 500) the fixed-order quadrature
    # degrades gracefully -- still ~1e-5, never diverges (unlike jax native).
    z = -numpy.array([100.0, 250.0, 500.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = _tonumpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hyp1f1_value_parity(backend):
    for alpha in [0.0, 1.0, 1.8, 2.5]:
        a, b = 1.5 - alpha / 2.0, 2.5 - alpha / 2.0
        X = numpy.array([0.0, 1e-3, 0.1, 1.0, 4.0, 16.0, 64.0, 256.0])
        ref = scipy_special.hyp1f1(a, b, -X)
        got = _tonumpy(gsp.hyp1f1(a, b, _asarray(backend, -X)))
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
        got = _tonumpy(fn(_asarray(backend, m)))
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
    got = _tonumpy(gsp.gamma(_asarray(backend, pts)))
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
    got = _tonumpy(
        xlogy_fallback(_ns(backend), _asarray(backend, x), _asarray(backend, y))
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_hyp2f1_fallback_alt_labeling(backend):
    # 2F1 case whose accurate Euler labeling comes from b (c-a < 1 <= c-b), i.e.
    # the second branch of _euler_labeling -- not exercised by galpy's calls
    # (which always satisfy c-a >= 1). jax/torch route through the fallback here.
    a, b, c = 2.0, 0.8, 2.5  # c-a=0.5 (<1), c-b=1.7 (>=1) -> labeling picks b
    z = -numpy.array([0.0, 0.1, 1.0, 5.0, 30.0])
    ref = scipy_special.hyp2f1(a, b, c, z)
    got = _tonumpy(gsp.hyp2f1(a, b, c, _asarray(backend, z)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_fallback_unsupported_regimes_raise():
    # The fallbacks raise (rather than silently return a low-accuracy value)
    # outside the regime galpy needs and they are accurate in.
    from galpy.backend.special._fallback.hyp1f1 import hyp1f1_fallback
    from galpy.backend.special._fallback.hyp2f1 import _euler_labeling

    # 2F1 with c - max(a,b) < 1 (both endpoints awkward): unsupported
    with pytest.raises(NotImplementedError):
        _euler_labeling(2.0, 2.0, 2.5)  # c-a=c-b=0.5
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
        got = _tonumpy(fn(_asarray(backend, x)))
        rtol = 0.0 if backend == "numpy" else 1e-12  # series + scaled-trapezoid
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13, err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_bessel_kn_value_parity(backend):
    # kn via the upward recurrence from k0, k1 (galpy uses kn(2, .)). n=0,1
    # exercise the recurrence base cases (kn_fallback short-circuits to K0/K1).
    x = _BESSEL_X
    for n in (0, 1, 2, 3, 5):
        ref = scipy_special.kn(n, x)
        got = _tonumpy(gsp.kn(n, _asarray(backend, x)))
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
        assert numpy.array_equal(_tonumpy(P), ref[0])
        assert numpy.array_equal(_tonumpy(dP), ref[1])
        assert numpy.array_equal(_tonumpy(d2), ref[2])
    else:
        for got, r, nm in [(P, ref[0], "P"), (dP, ref[1], "dP"), (d2, ref[2], "d2P")]:
            numpy.testing.assert_allclose(
                _tonumpy(got), r, rtol=1e-11, atol=1e-11, err_msg=nm
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_assoc_legendre_value_only_and_shape(backend):
    L, M = 5, 3
    x = numpy.array([0.3, -0.4])
    P = gsp.assoc_legendre(L, M, _asarray(backend, x))  # deriv=0 -> just P
    assert tuple(_tonumpy(P).shape) == (2, L, M)
    numpy.testing.assert_allclose(
        _tonumpy(P), _scipy_assoc_ref(L, M, x, 0)[0], rtol=1e-11, atol=1e-11
    )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_assoc_legendre_autodiff_matches_analytic(backend):
    # d/dx of P_l^m via autodiff must match the analytically-returned dP/dx.
    L, M = 6, 4
    x0 = 0.4
    _, dP_an = gsp.assoc_legendre(L, M, _asarray(backend, x0), deriv=1)
    dP_an = _tonumpy(dP_an)
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


# --- Tier 4b: Gegenbauer C_n^alpha (SCF radial basis) -------------------------
@pytest.mark.parametrize("backend", BACKENDS)
def test_gegenbauer_value_parity(backend):
    N = 8
    x = numpy.array([-0.9, -0.3, 0.0, 0.4, 0.95])
    for alpha in (1.5, 3.5, 2 * 3 + 1.5):  # SCF uses alpha = 2l + 3/2
        got = _tonumpy(gsp.gegenbauer(N, alpha, _asarray(backend, x)))
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
