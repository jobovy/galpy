###############################################################################
# test_backend_df.py: Track F Pdf.1 -- backend (jax/torch) coverage for the df
# foundation, surfaceSigmaProfile + jeans (sigmar/sigmalos). The numpy path is
# byte-identical (test_diskdf/test_jeans unchanged); this exercises the
# resolved-namespace dispatch (parity numpy<->jax<->torch + grad-vs-FD) that
# makes them evaluate AND differentiate under every backend.
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

from galpy.df import jeans
from galpy.df.surfaceSigmaProfile import expSurfaceSigmaProfile
from galpy.potential import HernquistPotential


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


_SSP_R = numpy.array([0.5, 0.8, 1.0, 1.3, 1.7])
_SSP_METHODS = ("surfacemass", "surfacemassDerivative", "sigma2", "sigma2Derivative")


# (method, log): the log-derivatives of an exponential are CONSTANT in R, so
# these two return a plain scalar (backend-array-independent) -- correctly, since
# wrapping them in a backend array would break the numpy byte-identical scalar
# return. They broadcast against backend arrays downstream.
_SSP_R_INDEP = {("surfacemassDerivative", True), ("sigma2Derivative", True)}


@pytest.mark.parametrize("backend", BACKENDS)
def test_surfaceSigmaProfile_parity(backend):
    p = expSurfaceSigmaProfile(params=(1.0 / 3.0, 1.0, 0.2))
    R = _arr(backend, _SSP_R)
    for name in _SSP_METHODS:
        for lg in (False, True):
            ref = getattr(p, name)(_SSP_R, log=lg)
            got = getattr(p, name)(R, log=lg)
            if (name, lg) not in _SSP_R_INDEP:
                assert _is_backend_array(backend, got)
            numpy.testing.assert_allclose(
                _np(got), numpy.asarray(ref), rtol=1e-12, atol=1e-14
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_surfaceSigmaProfile_grad_matches_derivative(backend):
    # d(surfacemass)/dR == surfacemassDerivative -- an exact analytic identity.
    p = expSurfaceSigmaProfile(params=(1.0 / 3.0, 1.0, 0.2))
    R0 = 0.9
    if backend == "jax":
        g = float(jax.grad(lambda r: p.surfacemass(r))(jnp.asarray(R0)))
    else:
        t = torch.tensor(R0, requires_grad=True)
        p.surfacemass(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, p.surfacemassDerivative(R0), rtol=1e-8)


_HP = HernquistPotential(normalize=1.0, a=1.3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["sigmar", "sigmalos"])
def test_jeans_parity(backend, fn):
    # numpy<->backend parity of the fixed-order-GL semi-infinite integral (the
    # numpy path uses scipy.quad; the backend path fixed_quad_semiinfinite).
    f = getattr(jeans, fn)
    for r0 in (0.7, 1.1, 1.6):
        ref = f(_HP, r0, use_physical=False)
        got = f(_HP, _arr(backend, r0), use_physical=False)
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(
            _np(got), numpy.asarray(ref), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_jeans_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr via backend autodiff vs central finite difference -- the
    # differentiability that motivates the migration.
    r0, eps = 1.1, 1e-5
    fd = (
        jeans.sigmar(_HP, r0 + eps, use_physical=False)
        - jeans.sigmar(_HP, r0 - eps, use_physical=False)
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(
            jax.grad(lambda r: jeans.sigmar(_HP, r, use_physical=False))(
                jnp.asarray(r0)
            )
        )
    else:
        t = torch.tensor(r0, requires_grad=True)
        jeans.sigmar(_HP, t, use_physical=False).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-4)
