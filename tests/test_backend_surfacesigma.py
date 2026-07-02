###############################################################################
# test_backend_surfacesigma.py: Track F Pdf.1 -- backend (jax/torch) coverage
# for surfaceSigmaProfile. The numpy path is byte-identical (test_diskdf
# unchanged); this exercises the resolved-namespace dispatch (parity
# numpy<->jax<->torch + grad-vs-analytic) that makes the profiles evaluate AND
# differentiate under every backend. One file per df module (jeans is in
# test_backend_jeans.py; each later df family gets its own file).
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

from galpy.df.surfaceSigmaProfile import expSurfaceSigmaProfile


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
