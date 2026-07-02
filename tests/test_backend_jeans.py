###############################################################################
# test_backend_jeans.py: Track F Pdf.1 -- backend (jax/torch) coverage for
# jeans (sigmar/sigmalos). The numpy path is byte-identical (test_jeans
# unchanged); this exercises the resolved-namespace dispatch (parity
# numpy<->jax<->torch + grad-vs-FD) that makes them evaluate AND differentiate
# under every backend. One file per df module (surfaceSigmaProfile is in
# test_backend_surfacesigma.py; each later df family gets its own file).
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
def test_jeans_sigmar_callable_beta_parity(backend):
    # callable (r-dependent) anisotropy: exercises the backend intFactor path
    # exp(2 * quad(beta(y)/y)) and its numpy<->backend parity (the closed-form
    # power-law intFactor is used for the constant-beta case above).
    beta = lambda r: 0.2 / (1.0 + r)
    for r0 in (0.7, 1.1, 1.6):
        ref = jeans.sigmar(_HP, r0, beta=beta, use_physical=False)
        got = jeans.sigmar(_HP, _arr(backend, r0), beta=beta, use_physical=False)
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
