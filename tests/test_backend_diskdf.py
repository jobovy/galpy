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
