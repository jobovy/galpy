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
def test_ssp_plain_scalar_stays_on_numpy(backend, fn, monkeypatch):
    # The PUBLIC entry points coerce plain inputs under a forced backend (see
    # test_forced_plain_inputs_return_backend); _ssp is what the numpy-only
    # internals (the _estimate* helpers evolveddiskdf calls per gridpoint, the
    # DFcorrection quadrature, the scipy samplers) call with PLAIN scalars,
    # ~133k times per scipy-quad oortA(). Under a forced backend a plain float
    # resolves to that backend, so if _ssp lets it through, every one of those
    # is a scalar coerced in and converted back out -- measured at an 8.7x
    # traced slowdown (25.3 s vs 2.9 s for one oortA) when it regressed.
    #
    # Asserting on the RETURN VALUE cannot see that: _ssp casts back with
    # as_numpy either way, so the round trip is invisible except as float noise
    # -- which jax happens to show and torch does not. Watch the coercion itself
    # instead, so the guard works the same on both backends.
    import galpy.backend._input as _bi

    real, entered = _bi.coerce_coords, []

    def spy(xp, *coords, **kwargs):
        # **kwargs: coerce_coords grew a device= anchor; a positional-only spy
        # turns every call through it into a TypeError.
        entered.append(getattr(xp, "__name__", str(xp)))
        return real(xp, *coords, **kwargs)

    monkeypatch.setattr(_bi, "coerce_coords", spy)

    ref = getattr(_dehnen._surfaceSigmaProfile, fn)(0.8)
    with use(backend, force=True):
        got = _dehnen._ssp(fn, 0.8)
    on_backend = [n for n in entered if "numpy" != n]
    assert not on_backend, f"{fn}: plain scalar coerced onto {on_backend}"
    assert not is_backend_array(got), f"{fn}: plain scalar leaked onto {backend}"
    numpy.testing.assert_array_equal(got, ref, err_msg=f"{fn}: value changed")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn", ["surfacemass", "surfacemassDerivative", "sigma2", "sigma2Derivative"]
)
def test_ssp_backend_array_uses_backend(backend, fn):
    # The other half of the same contract: pinning numpy for plain scalars must
    # not undo the migration, so a real backend array still gets the backend
    # path (and stays differentiable there).
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_numpy_only_consumers_stay_on_numpy_under_force(backend, monkeypatch):
    # the scipy/numpy-RNG methods keep their plain inputs on numpy under a forced
    # backend (otherwise every inner df()/sigma2() call is coerced in and out)
    import galpy.backend._input as _bi
    from galpy.df import dehnendf

    df = dehnendf(beta=0.1, profileParams=(0.25, 1.0, 0.2))
    numpy.random.seed(3)
    ref = df.sampleVRVT(0.9, n=2, target=False)
    real, entered = _bi.coerce_coords, []

    def spy(xp, *coords, **kwargs):
        entered.append(getattr(xp, "__name__", str(xp)))
        return real(xp, *coords, **kwargs)

    monkeypatch.setattr(_bi, "coerce_coords", spy)
    numpy.random.seed(3)
    with use(backend, force=True):
        got = df.sampleVRVT(0.9, n=2, target=False)
    assert not [n for n in entered if n != "numpy"]
    assert not is_backend_array(got)
    numpy.testing.assert_array_equal(got, ref)


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
