###############################################################################
# test_backend_diskdf.py: Track F Pdf.3 PR-1 -- backend (jax/torch) coverage for
# the diskdf differentiable eval + moment path (dehnendf / shudf). The numpy path
# is byte-identical (test_diskdf unchanged); this exercises the resolved-namespace
# dispatch:
#   (a) value parity numpy<->jax<->torch of eval (via __call__) and of the moment
#       quadratures (surfacemass / sigma2surfacemass / sigmaR2 / meanvT / oortA),
#       which run the scipy.dblquad region as a fixed-order nested Gauss-Legendre
#       rule on the backend, and
#   (b) grad-vs-FD of a moment w.r.t. R (jax.grad / torch.autograd vs central FD).
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
from galpy.df import dehnendf, shudf
from galpy.df.diskdf import vRvTRToEL

_dehnen = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
_shu = shudf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2))
# beta != 0 exercises the non-flat-rotation-curve _eval_backend branches.
_dehnen_b = dehnendf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
_shu_b = shudf(beta=0.2, profileParams=(1.0 / 4.0, 1.0, 0.2))
_DFS = [
    ("dehnendf", _dehnen),
    ("shudf", _shu),
    ("dehnendf_beta", _dehnen_b),
    ("shudf_beta", _shu_b),
]

# (vR, vT, R) test points; prograde (L>0) so the shu DF is non-zero.
_ELPTS = [(0.1, 0.9, 0.9), (0.0, 1.0, 1.0), (-0.05, 0.95, 1.1), (0.2, 0.8, 1.2)]
_RPTS = [0.8, 1.0, 1.2]


def _scalar(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(float(x))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dfname,df", _DFS)
def test_eval_call_parity(backend, dfname, df):
    # eval via __call__(E, L) value byte-identity numpy<->backend.
    for vR, vT, R in _ELPTS:
        E, L = vRvTRToEL(vR, vT, R, df._beta, df._dftype)
        ref = float(df(E, L))
        with use(backend, force=True):
            got = df(_scalar(backend, E), _scalar(backend, L))
        assert is_backend_array(got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dfname,df", _DFS)
@pytest.mark.parametrize(
    "fn", ["surfacemass", "sigma2surfacemass", "sigmaR2", "meanvT", "oortA"]
)
def test_moment_parity(backend, dfname, df, fn):
    # moment quadrature (backend nested-GL) parity vs numpy scipy.dblquad.
    for R in _RPTS:
        ref = float(getattr(df, fn)(R, use_physical=False))
        with use(backend, force=True):
            got = getattr(df, fn)(_scalar(backend, R), use_physical=False)
        assert is_backend_array(got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn", ["surfacemass", "surfacemassDerivative", "sigma2", "sigma2Derivative"]
)
def test_ssp_plain_scalar_under_numpy_pin_stays_on_numpy(backend, fn, monkeypatch):
    # _ssp's hot numpy callers -- the scipy quadratures and samplers (_on_numpy),
    # evolveddiskdf's numpy grids, the DFcorrection quadrature -- call it with
    # PLAIN scalars inside a numpy pin within a forced backend, ~133k times per
    # scipy-quad oortA(). There a plain scalar must not be coerced onto the
    # backend: that round trip was an 8.7x slowdown (25.3 s vs 2.9 s for one
    # oortA) when it regressed. A value check cannot see the round trip, so
    # watch the coercion itself.
    import galpy.backend._input as _bi

    real, entered = _bi.coerce_coords, []

    def spy(xp, *coords, **kwargs):
        # **kwargs: coerce_coords grew a device= anchor; a positional-only spy
        # turns every call through it into a TypeError.
        entered.append(getattr(xp, "__name__", str(xp)))
        return real(xp, *coords, **kwargs)

    monkeypatch.setattr(_bi, "coerce_coords", spy)

    ref = getattr(_dehnen._surfaceSigmaProfile, fn)(0.8)
    with use(backend, force=True), use("numpy", force=True):
        got = _dehnen._ssp(fn, 0.8)
    on_backend = [n for n in entered if "numpy" != n]
    assert not on_backend, f"{fn}: plain scalar coerced onto {on_backend}"
    assert not is_backend_array(got), f"{fn}: plain scalar leaked onto {backend}"
    numpy.testing.assert_array_equal(got, ref, err_msg=f"{fn}: value changed")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn", ["surfacemass", "surfacemassDerivative", "sigma2", "sigma2Derivative"]
)
def test_ssp_plain_scalar_forced_on_backend(backend, fn):
    # outside those pins, a forced backend evaluates a plain scalar on it
    ref = getattr(_dehnen._surfaceSigmaProfile, fn)(0.8)
    with use(backend, force=True):
        got = _dehnen._ssp(fn, 0.8)
    assert is_backend_array(got), f"{fn}: plain scalar stayed on numpy"
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn", ["surfacemass", "surfacemassDerivative", "sigma2", "sigma2Derivative"]
)
def test_ssp_backend_array_uses_backend(backend, fn):
    # and a backend array gets the backend path (and stays differentiable there)
    ref = float(getattr(_dehnen._surfaceSigmaProfile, fn)(0.8))
    with use(backend, force=True):
        got = _dehnen._ssp(fn, _scalar(backend, 0.8))
    assert is_backend_array(got), f"{fn}: backend array fell back to numpy"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-14, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dfname,df", _DFS)
@pytest.mark.parametrize("fn", ["surfacemass", "sigmaR2"])
def test_moment_grad_vs_fd(backend, dfname, df, fn):
    # d(moment)/dR: jax.grad / torch.autograd vs central FD (FD floor -> rtol 1e-5).
    R0, h = 1.1, 1e-6

    def npval(R):
        return float(getattr(df, fn)(R, use_physical=False))

    gfd = (npval(R0 + h) - npval(R0 - h)) / (2.0 * h)
    if backend == "jax":
        with use("jax", force=True):
            g = float(
                jax.grad(lambda R: getattr(df, fn)(R, use_physical=False))(
                    jnp.asarray(R0)
                )
            )
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        with use("torch", force=True):
            getattr(df, fn)(Rt, use_physical=False).backward()
        g = float(Rt.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=1e-5, atol=1e-8)


# --------------------------------------------------------------------------
# d/d(profile parameter): the surfaceSigmaProfile's own scale lengths.
#
# surfacemass / sigma2surfacemass / _vmomentsurfacemass and _ssp all dispatched
# on is_backend_array(R) ALONE. A differentiated profile parameter makes the
# result traced whatever R is (surfacemass is exp(-R/params[0])), so with a
# plain float R these fell into the numpy branch and ran numpy.sqrt / numpy.exp
# on a tracer: jax raised, and torch DETACHED to a silent zero -- a wrong
# gradient with a right value, which no value-only test can catch. The backend
# quadrature twin already existed; only the dispatch was wrong.
# --------------------------------------------------------------------------
_HR0 = 1.0 / 3.0


def _dehnen_quantity(hr, fn, backend):
    from galpy.df import dehnendf

    with use(backend, force=True):
        df = dehnendf(beta=0.0, profileParams=(hr, 1.0, 0.2))
        v = getattr(df, fn)(0.9, use_physical=False)
        return (
            v.reshape(-1)[0]
            if is_backend_array(v)
            else numpy.atleast_1d(v).reshape(-1)[0]
        )


@pytest.mark.parametrize("fn", ["surfacemass", "sigmaR2", "meanvT"])
@pytest.mark.parametrize("ip", [0, 2])  # hr, sigma_R(R=1)
@pytest.mark.parametrize("backend", BACKENDS)
def test_dehnendf_profile_parameter_grad_without_backend_context(backend, ip, fn):
    # A differentiated profile parameter at a Python-float R, with no use()
    # context: the parameter is the data (the leaves and the moment quadrature
    # used to resolve numpy from R alone and meet the traced parameter)
    from galpy.df import dehnendf

    p0 = (1.0 / 3.0, 1.0, 0.2)

    def q(p):
        pp = list(p0)
        pp[ip] = p
        return getattr(dehnendf(beta=0.0, profileParams=tuple(pp)), fn)(
            0.9, use_physical=False
        )

    if backend == "jax":
        g = float(jax.grad(q)(p0[ip]))
    else:
        t = torch.tensor(p0[ip], requires_grad=True)
        (g,) = torch.autograd.grad(q(t), t)
        g = float(g)
    fd = [
        (float(q(p0[ip] + h)) - float(q(p0[ip] - h))) / (2.0 * h) for h in (1e-4, 1e-5)
    ]
    # one numpy quadrature (not the backend one): FD converged to <= 1.3e-6
    # between h=1e-4 and 1e-5, AD vs h=1e-5 measured <= 1.3e-8
    assert abs(fd[0] - fd[1]) < 2e-6 * abs(fd[1])
    numpy.testing.assert_allclose(g, fd[1], rtol=5e-8)


@pytest.mark.parametrize("fn", ["surfacemass", "sigma2"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_dehnendf_grad_wrt_profile_parameter(backend, fn):
    h = 1e-6
    gfd = (
        float(_dehnen_quantity(_HR0 + h, fn, "numpy"))
        - float(_dehnen_quantity(_HR0 - h, fn, "numpy"))
    ) / (2.0 * h)
    if backend == "jax":
        g = float(jax.grad(lambda t: _dehnen_quantity(t, fn, "jax"))(jnp.asarray(_HR0)))
    else:
        t = torch.tensor(_HR0, dtype=torch.float64, requires_grad=True)
        _dehnen_quantity(t, fn, "torch").backward()
        g = float(t.grad)
    assert numpy.isfinite(g), f"{fn}: gradient must not be nan/inf"
    # the silent-zero guard: torch used to return a finite 0 here
    assert abs(g) > 0.0, f"{fn}: gradient is identically zero (detached?)"
    numpy.testing.assert_allclose(g, gfd, rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------
# Array R: the moment quadrature batches over R (the GL grid takes the two
# trailing axes); used to raise a (k,) vs (1, n) broadcasting error.
# --------------------------------------------------------------------------
_RARR = [[0.8, 1.0], [1.2, 1.1]]


def _array(backend, x, requires_grad=False):
    if backend == "jax":
        return jnp.asarray(x)
    return torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dfname,df", _DFS)
@pytest.mark.parametrize(
    "fn",
    [
        "surfacemass",
        "sigma2surfacemass",
        "sigmaR2",
        "sigmaT2",
        "meanvT",
        "meanvR",
        "kurtosisvR",
        "oortA",
    ],
)
def test_moment_array_R_matches_scalar(backend, dfname, df, fn):
    # array R == a loop over scalar R (the scalar path is matched to numpy above);
    # no ro/vo set, so these return internal units (kurtosisvR takes no use_physical)
    with use(backend, force=True):
        got = getattr(df, fn)(_array(backend, _RARR))
        ref = [float(getattr(df, fn)(_scalar(backend, r))) for r in numpy.ravel(_RARR)]
    assert is_backend_array(got)
    assert tuple(got.shape) == numpy.shape(_RARR)
    # the batched GL sum reorders the reduction: sigmaT2/kurtosis cancel to ~1e-13
    numpy.testing.assert_allclose(as_numpy(got).ravel(), ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("deriv", [None, "R"])
def test_vmomentsurfacemass_array_R_matches_scalar(backend, deriv):
    with use(backend, force=True):
        got = _shu_b.vmomentsurfacemass(
            _array(backend, _RPTS), 0, 2, deriv=deriv, use_physical=False
        )
        ref = [
            float(
                _shu_b.vmomentsurfacemass(
                    _scalar(backend, r), 0, 2, deriv=deriv, use_physical=False
                )
            )
            for r in _RPTS
        ]
    assert tuple(got.shape) == (len(_RPTS),)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dfname,df", _DFS)
@pytest.mark.parametrize("fn", ["surfacemass", "meanvT"])
def test_moment_array_R_grad_vs_fd(backend, dfname, df, fn):
    # d(moment)/dR at an array R (jax.vmap(jax.grad) / torch backward of the sum)
    # vs a central FD of the same (smooth, fixed-GL) backend quadrature
    R0, h = numpy.array(_RPTS), 1e-5

    def f(R):
        return getattr(df, fn)(R, use_physical=False)

    with use(backend, force=True):
        gfd = (
            as_numpy(f(_array(backend, R0 + h))) - as_numpy(f(_array(backend, R0 - h)))
        ) / (2.0 * h)
        if backend == "jax":
            g = numpy.asarray(jax.vmap(jax.grad(f))(jnp.asarray(R0)))
        else:
            Rt = _array(backend, R0, requires_grad=True)
            f(Rt).sum().backward()
            g = as_numpy(Rt.grad)
    # FD truncation ~h^2 f''' and roundoff ~1e-16/h: measured <= 1e-9
    numpy.testing.assert_allclose(g, gfd, rtol=1e-8, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dehnendf_profile_parameter_grad_numpy_array_R(backend):
    # a differentiated profile parameter with a NUMPY array R takes the batched
    # backend quadrature (not the per-element numpy loop)
    R = numpy.array(_RPTS)

    def q(hr, R):
        return dehnendf(beta=0.0, profileParams=(hr, 1.0, 0.2)).surfacemass(
            R, use_physical=False
        )

    if backend == "jax":
        g = numpy.asarray(jax.jacfwd(q)(jnp.asarray(_HR0), R))
        ref = [float(jax.grad(q)(jnp.asarray(_HR0), float(r))) for r in R]
    else:
        g = as_numpy(
            torch.autograd.functional.jacobian(
                lambda t: q(t, R), torch.tensor(_HR0, dtype=torch.float64)
            )
        )
        ref = []
        for r in R:
            t = torch.tensor(_HR0, dtype=torch.float64, requires_grad=True)
            q(t, float(r)).backward()
            ref.append(float(t.grad))
    # the scalar-R profile-parameter gradient is matched to FD above
    numpy.testing.assert_allclose(g, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_array_R_under_jit(backend):
    # jax.jit / torch.compile (dynamo) of an array-R moment and its R-gradient
    # match eager (measured 7e-16 values, round-off gradients)
    def f(R):
        return _dehnen_b.meanvT(R, use_physical=False)

    R0 = numpy.array(_RARR)
    with use(backend, force=True):
        if backend == "jax":
            ve, vc = f(jnp.asarray(R0)), jax.jit(f)(jnp.asarray(R0))
            gf = jax.vmap(jax.vmap(jax.grad(f)))
            ge, gc = gf(jnp.asarray(R0)), jax.jit(gf)(jnp.asarray(R0))
        else:
            torch._dynamo.reset()
            cf = torch.compile(f, backend="eager")
            Re = _array(backend, R0, requires_grad=True)
            Rc = _array(backend, R0, requires_grad=True)
            ve, vc = f(Re), cf(Rc)
            ve.sum().backward()
            vc.sum().backward()
            ve, vc, ge, gc = ve.detach(), vc.detach(), Re.grad, Rc.grad
    numpy.testing.assert_allclose(as_numpy(vc), as_numpy(ve), rtol=1e-14, atol=0.0)
    numpy.testing.assert_allclose(as_numpy(gc), as_numpy(ge), rtol=1e-12, atol=0.0)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize(
    "dfname,fn",
    [
        ("dehnendf", "surfacemass"),
        ("shudf", "sigma2surfacemass"),
        ("dehnendf_beta", "meanvT"),
        ("shudf_beta", "meanvT"),
    ],
)
def test_moment_compiles_under_a_default_device(dfname, fn):
    # torch.compile under a default device (what --device cuda runs; a CPU
    # default device reproduces it) cannot take the numpy.float64 gamma or
    # log(sigma_R) as a left operand ("'ndarray' object has no attribute 'mul'")
    from backend_jit_helpers import no_torch_compile_deprecations

    df = dict(_DFS)[dfname]

    def f(R):
        return getattr(df, fn)(R, use_physical=False)

    torch._dynamo.reset()
    with (
        use("torch", force=True),
        torch.device("cpu"),
        torch._dynamo.config.patch(trace_numpy=False),
        no_torch_compile_deprecations(),
    ):
        R = torch.tensor(_RARR, dtype=torch.float64)
        ref, got = f(R), torch.compile(f, backend="eager")(R)
    numpy.testing.assert_allclose(as_numpy(got), as_numpy(ref), rtol=1e-14, atol=0.0)


# --------------------------------------------------------------------------
# d/dbeta (the rotation-curve power law): gamma = sqrt(2/(1+beta)) and the
# `if beta == 0` branches were numpy/Python, so a backend beta raised in every
# mode. Each case builds the df INSIDE the differentiated function, in data
# mode (backend coordinates) and FORCED mode (plain floats under use(...,
# force=True)); the reference is a Richardson-extrapolated central FD of the
# same backend computation at a plain-float beta.
# --------------------------------------------------------------------------
_BETA_CLASSES = ["dehnendf", "shudf", "schwarzschilddf"]
_BETA_OUTS = ["f", "surfacemass", "sigmaR2", "meanvT"]


@pytest.fixture
def beta_quantity():
    import contextlib

    import galpy.df

    def q(cls, out, beta, mode, backend):
        vR, vT, R = 0.1, 0.9, 0.9
        if mode == "data":
            vR, vT, R = (_scalar(backend, x) for x in (vR, vT, R))
        ctx = use(backend, force=True) if mode == "forced" else contextlib.nullcontext()
        with ctx:
            df = getattr(galpy.df, cls)(beta=beta, profileParams=(0.25, 1.0, 0.2))
            if out == "f":
                return df(*vRvTRToEL(vR, vT, R, beta, df._dftype))
            return getattr(df, out)(R, use_physical=False)

    return q


def _richardson(f, x0, h=1e-3):
    d = lambda h: (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    d1, d2 = d(h), d(h / 2.0)
    return (4.0 * d2 - d1) / 3.0, abs(d2 - d1)


@pytest.mark.parametrize("mode", ["data", "forced"])
@pytest.mark.parametrize("out", _BETA_OUTS)
@pytest.mark.parametrize("cls", _BETA_CLASSES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_beta_grad_vs_fd(backend, cls, out, mode, beta_quantity):
    beta0 = 0.1
    fd, fderr = _richardson(
        lambda b: float(as_numpy(beta_quantity(cls, out, b, mode, backend))), beta0
    )
    assert fderr < 1e-4 * abs(fd)  # FD converged
    if backend == "jax":
        f = lambda b: jnp.reshape(beta_quantity(cls, out, b, mode, "jax"), ())
        gs = [jax.grad(f)(jnp.asarray(beta0)), jax.jit(jax.grad(f))(jnp.asarray(beta0))]
    else:
        b = torch.tensor(beta0, requires_grad=True)
        (g,) = torch.autograd.grad(beta_quantity(cls, out, b, mode, "torch"), b)
        gs = [g]
    # FD (Richardson, h=1e-3) truncation ~1e-12: measured AD-vs-FD <= 2e-10
    for g in gs:
        numpy.testing.assert_allclose(float(g), fd, rtol=1e-8, atol=1e-12)


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("cls", _BETA_CLASSES)
def test_beta_grad_vmap(cls, beta_quantity):
    # a batched (traced) beta takes both _beta_branch arms through xp.where
    betas = numpy.array([0.1, -0.15])
    f = lambda b: jnp.reshape(beta_quantity(cls, "f", b, "data", "jax"), ())
    g = numpy.asarray(jax.vmap(jax.grad(f))(jnp.asarray(betas)))
    ref = [float(jax.grad(f)(jnp.asarray(b))) for b in betas]
    numpy.testing.assert_allclose(g, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("out", _BETA_OUTS)
@pytest.mark.parametrize("cls", _BETA_CLASSES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("beta", [0.0, 0.2])
def test_backend_beta_value_parity(backend, beta, cls, out, beta_quantity):
    # a backend beta (concretely 0 or not, and traced under jax.jit) matches numpy
    ref = float(beta_quantity(cls, out, beta, "numpy", "numpy"))
    vals = [beta_quantity(cls, out, _scalar(backend, beta), "data", backend)]
    if backend == "jax":
        vals.append(
            jax.jit(lambda b: beta_quantity(cls, out, b, "data", "jax"))(
                jnp.asarray(beta)
            )
        )
    for v in vals:
        assert is_backend_array(v)
        numpy.testing.assert_allclose(float(as_numpy(v)), ref, rtol=1e-10)


@pytest.mark.parametrize("cls", _BETA_CLASSES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_forced_plain_inputs_return_backend(backend, cls, monkeypatch):
    # CONTRACT CHANGE: under a forced backend, plain-float inputs to __call__ and
    # the moments are coerced onto it (they used to stay on numpy by design, so
    # a gradient through e.g. a backend profile parameter was the only way in).
    # The moments then use the backend quadrature, never scipy's dblquad.
    import importlib

    import galpy.df
    from galpy.orbit import Orbit

    _dd = importlib.import_module("galpy.df.diskdf")  # the class shadows the module

    df = getattr(galpy.df, cls)(beta=0.1, profileParams=(0.25, 1.0, 0.2))
    E, L = vRvTRToEL(0.1, 0.9, 0.9, 0.1, df._dftype)
    calls = {
        "f(E,L)": lambda: df(E, L),
        "f(Orbit)": lambda: df(Orbit([0.9, 0.1, 0.9, 0.0])),
        "eval": lambda: df.eval(E, L),
        "targetSigma2": lambda: df.targetSigma2(0.9, use_physical=False),
        "targetSurfacemass": lambda: df.targetSurfacemass(0.9, use_physical=False),
        "asymmetricdrift": lambda: df.asymmetricdrift(0.9, use_physical=False),
        "vmomentsurfacemass": lambda: df.vmomentsurfacemass(
            0.9, 2, 0, use_physical=False
        ),
        "vmomentsurfacemass_odd": lambda: df.vmomentsurfacemass(
            0.9, 1, 0, use_physical=False
        ),
    }
    for fn in ["surfacemass", "sigma2surfacemass", "sigma2", "sigmaR2", "sigmaT2"]:
        calls[fn] = lambda fn=fn: getattr(df, fn)(0.9, use_physical=False)
    for fn in ["meanvT", "meanvR", "oortA", "oortB", "oortC", "oortK"]:
        calls[fn] = lambda fn=fn: getattr(df, fn)(0.9, use_physical=False)
    for fn in ["skewvT", "skewvR", "kurtosisvT", "kurtosisvR"]:
        calls[fn] = lambda fn=fn: getattr(df, fn)(0.9)
    calls["targetSurfacemassLOS"] = lambda: df.targetSurfacemassLOS(
        0.3, 20.0, use_physical=False
    )
    calls["targetSurfacemassLOS_log"] = lambda: df.targetSurfacemassLOS(
        0.3, 20.0, log=True, use_physical=False
    )
    calls["surfacemassLOS"] = lambda: df.surfacemassLOS(0.3, 20.0, use_physical=False)
    calls["surfacemassLOS_df"] = lambda: df.surfacemassLOS(
        0.3, 20.0, target=False, use_physical=False
    )
    refs = {k: float(numpy.real(c())) for k, c in calls.items()}
    used = []
    monkeypatch.setattr(
        _dd.integrate, "dblquad", lambda *a, **k: used.append(1) or (0.0, 0.0)
    )
    with use(backend, force=True):
        for k, c in calls.items():
            got = c()
            assert is_backend_array(got), f"{k}: returned {type(got)}"
            # meanvR/skewvR vanish: absolute floor
            numpy.testing.assert_allclose(
                float(as_numpy(got)), refs[k], rtol=1e-9, atol=1e-12, err_msg=k
            )
    assert not used, "a forced-backend moment fell back to scipy's dblquad"


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("cls", _BETA_CLASSES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_grad_wrt_profile_parameter_at_plain_EL(backend, cls, forced):
    # a differentiated profile parameter with plain-float (E, L): the backend
    # eval used to resolve numpy from (E, L) alone and meet the traced parameter
    import contextlib

    import galpy.df

    E, L = vRvTRToEL(0.1, 0.9, 0.9, 0.0, getattr(galpy.df, cls)()._dftype)

    def f(hr, bk=None):
        ctx = use(bk, force=True) if forced and bk else contextlib.nullcontext()
        with ctx:
            return getattr(galpy.df, cls)(profileParams=(hr, 1.0, 0.2))(E, L)

    fd, _ = _richardson(lambda h: float(f(h)), 0.25)
    if backend == "jax":
        g = float(jax.grad(lambda h: jnp.reshape(f(h, "jax"), ()))(jnp.asarray(0.25)))
    else:
        t = torch.tensor(0.25, requires_grad=True)
        (g,) = torch.autograd.grad(f(t, "torch"), t)
        g = float(g)
    # closed form on both sides: FD (Richardson) floor ~1e-12
    numpy.testing.assert_allclose(g, fd, rtol=1e-9, atol=0.0)


# --------------------------------------------------------------------------
# correct=True (DFcorrection). Backend profile parameters / beta: the
# corrections are computed on the backend (niter fixed iterations; the
# parameter-keyed save file is skipped) and interpolated with the same
# not-a-knot cubic as InterpolatedUnivariateSpline; plain parameters keep the
# numpy (cached) corrections, evaluated on the backend through their exact
# piecewise polynomials.
# --------------------------------------------------------------------------
_CORR_KW = dict(correct=True, niter=2, npoints=21)


@pytest.fixture
def corr_outputs(tmp_path):
    import contextlib

    def q(hr, s0, R, backend=None, forced=False):
        ctx = use(backend, force=True) if forced else contextlib.nullcontext()
        with ctx:
            df = dehnendf(
                profileParams=(hr, 1.0, s0), savedir=str(tmp_path), **_CORR_KW
            )
            vals = [df(*vRvTRToEL(0.1, 0.9, R, 0.0))]
            vals += [
                getattr(df, fn)(R, use_physical=False)
                for fn in ("surfacemass", "sigmaR2", "meanvT")
            ]
        if backend is None:
            return numpy.array([float(numpy.real(v)) for v in vals])
        xp = jnp if backend == "jax" else torch
        return xp.stack([xp.reshape(xp.asarray(v), ()) for v in vals])

    return q


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_corrections_match_numpy(backend, tmp_path):
    from galpy.df.diskdf import DFcorrection

    ref = dehnendf(profileParams=(0.25, 1.0, 0.2), savedir=str(tmp_path), **_CORR_KW)
    files = sorted(tmp_path.iterdir())
    got = dehnendf(
        profileParams=(_scalar(backend, 0.25), 1.0, 0.2),
        savedir=str(tmp_path),
        **_CORR_KW,
    )
    # the backend corrections skip the (parameter-keyed) save file
    assert sorted(tmp_path.iterdir()) == files
    assert is_backend_array(got._corr._corrections)
    # GL (order 100) vs adaptive dblquad, compounded over the iterations
    numpy.testing.assert_allclose(
        as_numpy(got._corr._corrections), ref._corr._corrections, rtol=1e-8
    )
    # small-R slope: the least-squares cubic == UnivariateSpline(k=3, default s)
    # and the not-a-knot cubic == InterpolatedUnivariateSpline(k=3)
    c = got._corr._corrections
    ssp = ref._surfaceSigmaProfile
    npc = DFcorrection(corrections=as_numpy(c), surfaceSigmaProfile=ssp)
    bkc = DFcorrection(corrections=c, surfaceSigmaProfile=ssp)
    numpy.testing.assert_allclose(
        as_numpy(bkc._backend_tables[3]),
        [npc._surfaceDerivSmallR, npc._sigma2DerivSmallR],
        rtol=1e-10,
    )
    for R in [1e-12, 0.3, 1.1, 4.9, 7.0, 12.0]:
        numpy.testing.assert_allclose(
            as_numpy(bkc.correct(R, log=True)),
            npc.correct(R, log=True),
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("interp_k", [1, 3])
def test_numpy_corrections_on_backend_inputs(backend, interp_k, tmp_path):
    # cached numpy corrections, backend R: the scipy splines' own polynomials
    from galpy.df.diskdf import DFcorrection

    df = dehnendf(profileParams=(0.25, 1.0, 0.2), savedir=str(tmp_path), **_CORR_KW)
    corr = DFcorrection(
        corrections=df._corr._corrections,
        surfaceSigmaProfile=df._surfaceSigmaProfile,
        interp_k=interp_k,
    )
    R = numpy.array([1e-12, 0.3, 1.1, 4.9, 7.0, 12.0])
    for log in (True, False):
        got = corr.correct(_array(backend, R), log=log)
        assert is_backend_array(got)
        ref = numpy.array([corr.correct(r, log=log) for r in R]).T
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13, atol=1e-15)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_numpy_corrections_first_backend_call_compiled(tmp_path):
    # the backend tables exist before the first backend call, so a compiled
    # first call does not trace into scipy (dynamo fails inside splev)
    from galpy.df.diskdf import DFcorrection

    df = dehnendf(profileParams=(0.25, 1.0, 0.2), savedir=str(tmp_path), **_CORR_KW)
    corr = DFcorrection(
        corrections=df._corr._corrections,
        surfaceSigmaProfile=df._surfaceSigmaProfile,
    )
    R = numpy.array([1e-12, 0.3, 1.1, 4.9, 7.0, 12.0])
    torch._dynamo.reset()
    cf = torch.compile(lambda r: corr.correct(r, log=True), backend="eager")
    got = cf(torch.tensor(R, dtype=torch.float64))
    ref = numpy.array([corr.correct(r, log=True) for r in R]).T
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_corrections_interp_k(backend, tmp_path):
    from galpy.df.diskdf import DFcorrection

    df = dehnendf(profileParams=(0.25, 1.0, 0.2), savedir=str(tmp_path), **_CORR_KW)
    c = df._corr._corrections
    for k in (1, 3):
        npc = DFcorrection(
            corrections=c, surfaceSigmaProfile=df._surfaceSigmaProfile, interp_k=k
        )
        bkc = DFcorrection(
            corrections=_array(backend, c),
            surfaceSigmaProfile=df._surfaceSigmaProfile,
            interp_k=k,
        )
        for R in [0.3, 1.1, 4.9, 7.0]:
            numpy.testing.assert_allclose(
                as_numpy(bkc.correct(R, log=True)),
                npc.correct(R, log=True),
                rtol=1e-12,
                atol=1e-14,
            )
    with pytest.raises(NotImplementedError, match="interp_k"):
        DFcorrection(
            corrections=_array(backend, c),
            surfaceSigmaProfile=df._surfaceSigmaProfile,
            interp_k=2,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_corrected_forced_plain_inputs_match_numpy(backend, corr_outputs):
    # plain parameters under a forced backend: the cached numpy corrections,
    # evaluated (and the moments integrated) on the backend
    ref = corr_outputs(0.25, 0.2, 0.9)
    vals = corr_outputs(0.25, 0.2, 0.9, backend=backend, forced=True)
    assert is_backend_array(vals)
    # f: exact; moments: GL (order 100) vs adaptive dblquad across spline knots
    numpy.testing.assert_allclose(as_numpy(vals)[0], ref[0], rtol=1e-14)
    numpy.testing.assert_allclose(as_numpy(vals)[1:], ref[1:], rtol=1e-9)


@pytest.mark.parametrize("mode", ["data", "forced"])
@pytest.mark.parametrize("param", ["hR", "sigma0", "R"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_corrected_grad_vs_fd(backend, param, mode, corr_outputs):
    if param == "R" and mode == "forced":
        pytest.skip("R is the differentiated coordinate: data mode")
    p0 = {"hR": 0.25, "sigma0": 0.2, "R": 0.9}
    forced = mode == "forced"

    def f(x, xp):
        P = {k: (x if k == param else v) for k, v in p0.items()}
        if not forced:  # data mode: every input a backend array
            P = {k: (v if k == param else xp(v)) for k, v in P.items()}
        return corr_outputs(P["hR"], P["sigma0"], P["R"], backend, forced)

    if backend == "jax":
        jf = jax.jit(lambda x: f(x, jnp.asarray))
        g = numpy.asarray(
            jax.jacrev(lambda x: f(x, jnp.asarray))(jnp.asarray(p0[param]))
        )
        gj = numpy.asarray(
            jax.jit(jax.jacrev(lambda x: f(x, jnp.asarray)))(jnp.asarray(p0[param]))
        )
        numpy.testing.assert_allclose(gj, g, rtol=1e-10)
        # traced parameter: the backend-corrections route for the FD too
        fv = lambda x: numpy.asarray(jf(jnp.asarray(x)))
    else:
        tt = lambda v: torch.tensor(v)
        t = torch.tensor(p0[param], requires_grad=True)
        out = f(t, tt)
        g = numpy.array(
            [float(torch.autograd.grad(o, t, retain_graph=True)[0]) for o in out]
        )
        fv = lambda x: as_numpy(f(torch.tensor(x, requires_grad=True), tt).detach())
    d = lambda h: (fv(p0[param] + h) - fv(p0[param] - h)) / (2.0 * h)
    d1, d2 = d(1e-3), d(5e-4)
    fd = (4.0 * d2 - d1) / 3.0
    assert numpy.all(numpy.fabs(d2 - d1) < 1e-3 * numpy.fabs(fd) + 1e-10)
    # Richardson truncation ~1e-12; measured AD-vs-FD <= 2e-10
    numpy.testing.assert_allclose(g, fd, rtol=1e-8, atol=1e-11)


# (d, l in degrees): the near side, the far side (theta = pi - arcsin), and the
# Sun's position (R = 0 is nudged by 1e-4)
_LOSPTS = [(0.3, 20.0), (1.5, 10.0), (1.0, 0.0)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("d,l", _LOSPTS)
def test_targetSurfacemassLOS_backend_value_and_grad(backend, d, l):
    # the LOS geometry (_dlToRphi) on the backend: numpy's value at every
    # branch, and d/dd against a central finite difference of numpy
    df = _dehnen
    ref = df.targetSurfacemassLOS(d, l, use_physical=False)
    dd = 1e-6
    fd = (
        df.targetSurfacemassLOS(d + dd, l, use_physical=False)
        - df.targetSurfacemassLOS(d - dd, l, use_physical=False)
    ) / (2.0 * dd)
    if backend == "jax":
        f = lambda x: df.targetSurfacemassLOS(x, l, use_physical=False)
        got, grad = f(jnp.asarray(d)), jax.grad(f)(jnp.asarray(d))
    else:
        x = torch.tensor(d, requires_grad=True)
        got = df.targetSurfacemassLOS(x, l, use_physical=False)
        (grad,) = torch.autograd.grad(got, x)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-14)
    if d == 1.0 and l == 0.0:  # R = |1 - d|: a kink, so only a finite slope
        assert numpy.isfinite(float(as_numpy(grad)))
    else:
        numpy.testing.assert_allclose(float(as_numpy(grad)), fd, rtol=1e-8)


_PRIVATE = [
    ("_dlnfdR", (0.8, 0.1, 0.9)),
    ("_dlnfdvR", (0.8, 0.1, 0.9)),
    ("_dlnfdvT", (0.8, 0.1, 0.9)),
    ("_dlnfdRe", (0.8, 0.1, 0.9)),
    ("_dlnfdl", (0.8, 0.1, 0.9)),
    ("_asymmetricdrift", (0.8,)),
    ("_estimatemeanvT", (0.8,)),
    ("_estimateSigmaR2", (0.8,)),
    ("_estimateSigmaT2", (0.8,)),
]
_PRIVATE_DFS = [
    ("dehnen-flat", dehnendf, 0.0),
    ("dehnen-rise", dehnendf, 0.2),
    ("shu-flat", shudf, 0.0),
    ("shu-fall", shudf, -0.2),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "cls,beta", [d[1:] for d in _PRIVATE_DFS], ids=[d[0] for d in _PRIVATE_DFS]
)
@pytest.mark.parametrize("name,args", _PRIVATE, ids=[p[0] for p in _PRIVATE])
def test_forced_private_profile_helpers_on_backend(backend, cls, beta, name, args):
    # the derivative/estimate helpers at plain floats under a forced backend: on
    # it (they read the surface-density/dispersion profiles), numpy's values
    df = cls(beta=beta, profileParams=(1.0 / 4.0, 1.0, 0.2))
    if not hasattr(df, name):  # shudf has no _dlnfdRe/_dlnfdl
        pytest.skip(f"{cls.__name__} has no {name}")
    ref = getattr(df, name)(*args)
    with use(backend, force=True):
        got = getattr(df, name)(*args)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-14)


###############################################################################
# Inverse-CDF samplers (2026-10-28): sample / sampleVRVT / sampleLOS /
# sampledSurfacemassLOS on the backends. The chain of tables and inversions is a
# deterministic function of the uniforms, so (a) a forced backend with key=None
# (numpy's global stream) reproduces the numpy path's draws, (b) a key
# reproduces its own draws and returns backend arrays, (c) the draws follow the
# distribution (moments against the DF's own integrals), and (d) the samples
# are differentiable in the DF's parameters: AD against central finite
# differences on the continuous chain (the kappa/omega_R replication of
# sample() is discrete, so its public gradient is checked for finiteness only;
# the velocity and line-of-sight samplers are continuous end to end).
###############################################################################
from galpy.backend import random as grandom  # noqa: E402
from galpy.backend._namespaces import namespace_for_name  # noqa: E402
from galpy.backend.sampling import (  # noqa: E402
    linear_inverse_cdf_sample,
    trapezoid_cdf,
    trapezoid_sum,
)
from galpy.orbit import Orbit  # noqa: E402

# <vT>, sigma_R, sigma_T of _dehnen at R=0.7 on a 1601 x 3001 velocity mesh
_VTRUTH = (0.88098, 0.25266, 0.19583)
_LOS_L = numpy.pi / 4.0


def _los_truth(df, target=True, nnodes=1001, maxd=4.0):
    # mean and std of the distance along the line of sight, Riemann on a grid
    d = numpy.linspace(0.0, maxd, nnodes)[1:]
    p = numpy.array(
        [
            df.surfacemassLOS(x, _LOS_L, deg=False, target=target, use_physical=False)
            for x in d
        ]
    )
    m = numpy.sum(d * p) / numpy.sum(p)
    return m, numpy.sqrt(numpy.sum(d**2 * p) / numpy.sum(p) - m**2)


def _orbit_cols(o, cols=("R", "vR", "vT")):
    if isinstance(o, Orbit):
        return numpy.stack(
            [as_numpy(getattr(o, c)(use_physical=False)) for c in cols], axis=1
        )
    return numpy.array(
        [[float(getattr(x, c)(use_physical=False)) for c in cols] for x in o]
    )


def _with_hR(df, hR):
    """The same DF with the surface scale length hR (a differentiated value)"""
    p = df._surfaceSigmaProfile._params
    return type(df)(beta=df._beta, profileParams=(hR, p[1], p[2]))


def _chain(df, xp, u, hR=None):
    """(R, vR, vT, omega_R) of the draws at the fixed uniforms u (3, n): the
    continuous part of sample() (no kappa/omega_R replication)"""
    if hR is not None:
        df = _with_hR(df, hR)
    xg, F = df._sample_x_table(xp)
    x = linear_inverse_cdf_sample(xp, xg, F, xp.asarray(u[0]))
    E, L = df._sample_EL(x, xp.asarray(u[1]), xp)
    R, vR, vT, Or = df._sample_phase(E, L, x, xp.asarray(u[2]), xp)
    return xp.stack([R, vR, vT, Or])


def _grad_vs_fd(ad, fd):
    """Elementwise AD-vs-FD check; the action-angle code's epicycle screen is a
    branch, so a draw that switches branch inside the finite-difference step
    is a genuine FD failure, not an AD one: a few per cent are allowed."""
    assert numpy.all(numpy.isfinite(ad))
    err = numpy.abs(ad - fd) / (numpy.abs(fd) + 1e-6)
    assert numpy.median(err) < 1e-6, numpy.median(err)
    assert numpy.mean(err > 1e-5) <= 0.1, (numpy.mean(err > 1e-5), err.max())
    assert err.max() < 1e-3, err.max()


# --- parity with the numpy path on the same uniforms ---------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,df", _DFS)
def test_sample_forced_backend_matches_numpy(backend, name, df):
    numpy.random.seed(3)
    ref = _orbit_cols(df.sample(n=30))
    with use(backend, force=True):
        numpy.random.seed(3)
        got = _orbit_cols(df.sample(n=30))
    assert ref.shape == got.shape
    assert numpy.allclose(got, ref, rtol=1e-7, atol=1e-9), numpy.abs(got - ref).max()
    numpy.random.seed(4)
    ref = numpy.array(df.sample(n=12, returnROrbit=False, returnOrbit=False))
    with use(backend, force=True):
        numpy.random.seed(4)
        got = numpy.array(df.sample(n=12, returnROrbit=False, returnOrbit=False))
    assert numpy.allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_velocity_and_los_forced_backend_match_numpy(backend):
    numpy.random.seed(5)
    ref_v = _dehnen.sampleVRVT(0.7, n=40, use_physical=False)
    ref_d = _dehnen.sampledSurfacemassLOS(_LOS_L, n=40, use_physical=False)
    ref_o = _orbit_cols(
        _dehnen.sampleLOS(_LOS_L, n=20, deg=False), ("R", "vR", "vT", "phi")
    )
    with use(backend, force=True):
        numpy.random.seed(5)
        got_v = _dehnen.sampleVRVT(0.7, n=40, use_physical=False)
        got_d = _dehnen.sampledSurfacemassLOS(_LOS_L, n=40, use_physical=False)
        got_o = _orbit_cols(
            _dehnen.sampleLOS(_LOS_L, n=20, deg=False), ("R", "vR", "vT", "phi")
        )
    for got, ref in ((got_v, ref_v), (got_d, ref_d), (got_o, ref_o)):
        assert isinstance(got, numpy.ndarray)  # key=None: numpy out
        assert numpy.allclose(got, ref, rtol=1e-8, atol=1e-10), numpy.abs(
            got - ref
        ).max()


# --- a key: backend arrays, reproducible, the options --------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,df", _DFS)
def test_sample_key(backend, name, df):
    o = df.sample(n=40, key=grandom.key(1, backend))
    assert isinstance(o, Orbit) and len(o) == 40
    R = o.R(use_physical=False)
    assert is_backend_array(R)
    again = df.sample(n=40, key=grandom.key(1, backend)).R(use_physical=False)
    assert numpy.array_equal(as_numpy(R), as_numpy(again))
    # (another key: on the fixed-shape EL output, since eager jax compiles
    # every op again for each new replicated output size)
    EL1 = df.sample(
        n=40, returnROrbit=False, returnOrbit=False, key=grandom.key(1, backend)
    )
    EL2 = df.sample(
        n=40, returnROrbit=False, returnOrbit=False, key=grandom.key(2, backend)
    )
    assert not numpy.array_equal(as_numpy(EL1), as_numpy(EL2))
    o = df.sample(n=40, returnOrbit=True, key=grandom.key(1, backend))
    phi = as_numpy(o.phi(use_physical=False))
    assert len(o) == 40 and phi.min() >= 0.0 and phi.max() <= 2.0 * numpy.pi
    assert is_backend_array(EL1) and tuple(EL1.shape) == (40, 2)
    assert len(df.sample(n=40, nphi=3, key=grandom.key(1, backend))) == 120
    o = df.sample(n=40, rrange=[0.5, 1.5], key=grandom.key(1, backend))
    R = as_numpy(o.R(use_physical=False))
    assert len(o) == 40 and R.min() >= 0.5 and R.max() <= 1.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_key_physical(backend):
    df = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, 0.2), ro=8.0, vo=220.0)
    o = df.sample(n=10, key=grandom.key(1, backend))
    assert o._roSet and o._voSet
    assert numpy.allclose(as_numpy(o.R()), 8.0 * as_numpy(o.R(use_physical=False)))
    o = df.sampleLOS(45.0, n=10, key=grandom.key(1, backend))
    assert isinstance(o, Orbit) and o._roSet and len(o) == 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_velocity_and_los_key(backend):
    vv = _dehnen.sampleVRVT(
        0.7, n=20000, key=grandom.key(3, backend), use_physical=False
    )
    assert is_backend_array(vv) and tuple(vv.shape) == (20000, 2)
    v = as_numpy(vv)
    assert abs(v[:, 0].mean()) < 0.006  # 3 s.e.
    assert abs(v[:, 1].mean() - _VTRUTH[0]) < 0.005
    assert abs(v[:, 0].std() - _VTRUTH[1]) < 0.004
    assert abs(v[:, 1].std() - _VTRUTH[2]) < 0.004
    ds = _dehnen.sampledSurfacemassLOS(
        _LOS_L, n=20000, key=grandom.key(3, backend), use_physical=False
    )
    assert is_backend_array(ds) and tuple(ds.shape) == (20000,)
    m, s = _los_truth(_dehnen)
    d = as_numpy(ds)
    assert abs(d.mean() - m) < 0.01 and abs(d.std() - s) < 0.01  # 3 s.e.
    o = _dehnen.sampleLOS(_LOS_L, n=200, deg=False, key=grandom.key(3, backend))
    assert isinstance(o, Orbit) and len(o) == 200
    R, phi = as_numpy(o.R(use_physical=False)), as_numpy(o.phi(use_physical=False))
    assert numpy.all(R > 0.0) and numpy.all(numpy.abs(phi) < numpy.pi)
    again = _dehnen.sampleLOS(_LOS_L, n=200, deg=False, key=grandom.key(3, backend))
    assert numpy.array_equal(R, as_numpy(again.R(use_physical=False)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_key_moments(backend):
    # the DF's actual surface density (quadrature: <R> = 0.5115, its spread
    # 0.367; the sampler reproduces them to 0.001 at n = 20000), <vR> = 0,
    # and the DF's own velocity dispersions in the ring R in [0.6, 0.8]
    o = _dehnen.sample(n=1000, key=grandom.key(5, backend))
    R, vR, vT = _orbit_cols(o).T
    assert abs(R.mean() - 0.5115) < 0.04 and abs(R.std() - 0.367) < 0.04
    assert abs(vR.mean()) < 0.03
    m = (R > 0.6) & (R < 0.8)
    assert abs(vR[m].std() - numpy.sqrt(_dehnen.sigmaR2(0.7))) < 0.05
    assert abs(vT[m].std() - numpy.sqrt(_dehnen.sigmaT2(0.7))) < 0.05


# --- differentiability -----------------------------------------------------------
@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("name,df", _DFS[:2])
def test_sample_chain_grad_jax(name, df):
    u = numpy.random.default_rng(11).random((3, 120))

    def f(hR):
        return _chain(df, jnp, u, hR=hR)

    ad = numpy.asarray(jax.jacfwd(f)(0.25))
    h = 1e-5
    fd = numpy.asarray((f(0.25 + h) - f(0.25 - h)) / (2.0 * h))
    _grad_vs_fd(ad, fd)
    # the public sampler: finite gradients through the replicated points
    g = jax.grad(
        lambda hR: jnp.mean(
            _with_hR(df, hR)
            .sample(n=60, key=grandom.key(2, "jax"))
            .R(use_physical=False)
        )
    )(0.25)
    assert numpy.isfinite(float(g)) and float(g) != 0.0


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
@pytest.mark.parametrize("name,df", _DFS[:2])
def test_sample_chain_grad_torch(name, df):
    xp = namespace_for_name("torch")
    u = numpy.random.default_rng(11).random((3, 16))

    def f(hR):
        return _chain(df, xp, u, hR=hR)

    ad = torch.autograd.functional.jacobian(f, torch.tensor(0.25)).numpy()
    h = 1e-5
    fd = ((f(torch.tensor(0.25 + h)) - f(torch.tensor(0.25 - h))) / (2.0 * h)).numpy()
    _grad_vs_fd(ad, fd)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_velocity_and_los_grad_jax():
    # continuous end to end: the mean sampled vT w.r.t. the dispersion scale,
    # the mean sampled distance w.r.t. the surface scale length
    key = grandom.key(9, "jax")

    def mean_vT(sR):
        df = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, sR))
        return jnp.mean(df.sampleVRVT(0.7, n=400, key=key, use_physical=False)[:, 1])

    def mean_d(hR):
        return jnp.mean(
            _with_hR(_dehnen, hR).sampledSurfacemassLOS(
                _LOS_L, n=400, key=key, use_physical=False
            )
        )

    for fn, x0 in ((mean_vT, 0.2), (mean_d, 0.25)):
        ad = float(jax.grad(fn)(x0))
        h = 1e-4
        fd = float((fn(x0 + h) - fn(x0 - h)) / (2.0 * h))
        assert numpy.isfinite(ad) and abs(ad - fd) < 1e-3 * abs(fd) + 1e-6, (
            fn.__name__,
            ad,
            fd,
        )


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
def test_velocity_and_los_grad_torch():
    key = grandom.key(9, "torch")

    def mean_vT(sR):
        df = dehnendf(beta=0.0, profileParams=(1.0 / 4.0, 1.0, sR))
        return df.sampleVRVT(0.7, n=400, key=key, use_physical=False)[:, 1].mean()

    def mean_d(hR):
        return (
            _with_hR(_dehnen, hR)
            .sampledSurfacemassLOS(_LOS_L, n=400, key=key, use_physical=False)
            .mean()
        )

    for fn, x0 in ((mean_vT, 0.2), (mean_d, 0.25)):
        x = torch.tensor(x0, requires_grad=True)
        (ad,) = torch.autograd.grad(fn(x), x)
        h = 1e-4
        fd = float((fn(torch.tensor(x0 + h)) - fn(torch.tensor(x0 - h))) / (2.0 * h))
        assert (
            numpy.isfinite(float(ad)) and abs(float(ad) - fd) < 1e-3 * abs(fd) + 1e-6
        ), (fn.__name__, float(ad), fd)


# --- the CDF helpers ------------------------------------------------------------
def test_trapezoid_cdf_helpers():
    g = numpy.linspace(0.0, 2.0, 41)
    p = numpy.stack([numpy.exp(-g), g * 0.0, g**2])  # a tail, an empty slice, a rise
    assert numpy.allclose(trapezoid_sum(numpy, p, g, 1), numpy.trapezoid(p, g, axis=1))
    F = trapezoid_cdf(numpy, p, g, 1)
    assert (
        F.shape == p.shape
        and numpy.allclose(F[:, 0], 0.0)
        and numpy.allclose(F[:, -1], 1.0)
    )
    assert numpy.all(numpy.diff(F, axis=1) >= 0.0)
    c = numpy.concatenate(
        [[0.0], numpy.cumsum(0.5 * (p[0, 1:] + p[0, :-1]) * (g[1] - g[0]))]
    )
    assert numpy.allclose(F[0], c / c[-1])
    assert numpy.allclose(F[1], numpy.linspace(0.0, 1.0, 41))  # the empty slice: a ramp


@pytest.mark.parametrize("backend", BACKENDS)
def test_shudf_eval_vanishes_at_zero_angular_momentum(backend):
    # the backend guard's stand-in xL = 1 made f(E, L=0) finite and wrong
    E, L = _scalar(backend, 0.2), _scalar(backend, 0.0)
    assert float(_shu._eval_backend(E, L, 0.0, 0.0)) == 0.0
    assert float(_shu.eval(0.2, 0.0)) == 0.0
    assert float(_shu._eval_backend(E, _scalar(backend, 1e-3), 0.0, 0.0)) > 0.0
