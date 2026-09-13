###############################################################################
# test_backend_linalg.py: backend-agnostic linear-algebra primitives
# (galpy.backend.linalg). psd_project = the differentiable nearest-PSD
# projection of a batch of symmetric matrices, used to sanitise streamTrack's
# smoothed covariance series. The forward matches the plain per-slice
# numpy.linalg.eigh loop; the backend gradient stays FINITE where a naive
# eigh(cov) in the grad path would NaN (repeated / clamped-to-zero eigenvalues).
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array
from galpy.backend.linalg import psd_project

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


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _psd_loop(cov):
    out = numpy.array(cov, dtype=float)
    for k in range(out.shape[0]):
        evals, evecs = numpy.linalg.eigh(out[k])
        out[k] = (evecs * numpy.clip(evals, 0.0, None)) @ evecs.T
    return out


def _cov_batch(seed=0, K=30, degenerate=False):
    rng = numpy.random.RandomState(seed)
    A = rng.randn(K, 6, 6)
    cov = numpy.einsum("kij,klj->kil", A, A)  # PSD
    cov[::3] -= 2e-3 * numpy.eye(6)  # inject some negative-eigenvalue slices
    if degenerate:
        cov[1::5] = 1e-2 * numpy.eye(6)[None]  # isotropic -> repeated eigenvalues
    return cov


def test_psd_project_numpy_matches_loop():
    # numpy path is the plain per-slice eigh loop (byte-identical to the inline
    # streamTrack loop it replaces).
    cov = _cov_batch()
    numpy.testing.assert_array_equal(psd_project(cov), _psd_loop(cov))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("degenerate", [False, True])
def test_psd_project_backend_parity(backend, degenerate):
    # the batched backend projection reproduces the numpy loop, incl. isotropic
    # (repeated-eigenvalue) slices; the result is a backend array.
    cov = _cov_batch(degenerate=degenerate)
    ref = _psd_loop(cov)
    got = psd_project(_arr(backend, cov))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11, atol=1e-12)
    # projection is idempotent and symmetric-PSD
    got_np = as_numpy(got)
    w = numpy.linalg.eigvalsh(got_np)
    assert w.min() > -1e-10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("degenerate", [False, True])
def test_psd_project_backend_grad_finite(backend, degenerate):
    # the gradient is FINITE everywhere -- including repeated / clamped-to-zero
    # eigenvalues, where a naive eigh(cov) in the grad path NaN-poisons -- and
    # jax and torch agree (frozen-eigenvector projection).
    cov = _cov_batch(seed=1, degenerate=degenerate)
    if backend == "jax":
        g = numpy.asarray(jax.grad(lambda c: jnp.sum(psd_project(c)))(jnp.asarray(cov)))
    else:
        ct = torch.tensor(cov, requires_grad=True)
        psd_project(ct).sum().backward()
        g = numpy.asarray(ct.grad.detach())
    assert numpy.isfinite(g).all()
    assert numpy.max(numpy.abs(g)) > 0


@pytest.mark.skipif(not BACKENDS, reason="no backend")
def test_psd_project_jax_torch_grad_agree():
    if "jax" not in BACKENDS or "torch" not in BACKENDS:
        pytest.skip("need both backends")
    cov = _cov_batch(seed=2)
    gj = numpy.asarray(jax.grad(lambda c: jnp.sum(psd_project(c)))(jnp.asarray(cov)))
    ct = torch.tensor(cov, requires_grad=True)
    psd_project(ct).sum().backward()
    gt = numpy.asarray(ct.grad.detach())
    numpy.testing.assert_allclose(gj, gt, rtol=1e-9, atol=1e-11)
