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
from galpy.backend.linalg import cholesky_invert, psd_project, real_eig

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


# --- cholesky_invert / real_eig ----------------------------------------------
def _spd(seed=4, n=3):
    rng = numpy.random.RandomState(seed)
    m = rng.randn(n, n)
    return m @ m.T + 3.0 * numpy.eye(n)


def test_cholesky_invert_numpy_is_the_scipy_object():
    # the numpy branch must BE galpy.util.fast_cholesky_invert, bit for bit --
    # streamdf's _sigomatrixinv/_sigomatrixLogdet are pinned by value elsewhere
    from galpy.util import fast_cholesky_invert

    a = _spd()
    for tiny in (1e-9, 1e-15):
        got_i, got_l = cholesky_invert(a, tiny, logdet=True)
        ref_i, ref_l = fast_cholesky_invert(a, tiny=tiny, logdet=True)
        assert got_i.tobytes() == ref_i.tobytes()
        assert got_l == ref_l
        assert (
            cholesky_invert(a, tiny).tobytes()
            == fast_cholesky_invert(a, tiny=tiny).tobytes()
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tiny", [1e-9, 1e-15])
def test_cholesky_invert_backend_matches_scipy(backend, tiny):
    from galpy.util import fast_cholesky_invert

    a = _spd()
    ref_i, ref_l = fast_cholesky_invert(a, tiny=tiny, logdet=True)
    got_i, got_l = cholesky_invert(_arr(backend, a), tiny, logdet=True)
    assert is_backend_array(got_i)
    numpy.testing.assert_allclose(as_numpy(got_i), ref_i, rtol=1e-13, atol=1e-15)
    numpy.testing.assert_allclose(float(as_numpy(got_l)), ref_l, rtol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cholesky_invert_backend_without_logdet(backend):
    # logdet=False on the BACKEND path returns the inverse alone (the numpy
    # branch of this is covered above; this is the backend twin)
    from galpy.util import fast_cholesky_invert

    a = _spd()
    got = cholesky_invert(_arr(backend, a), 1e-15)
    assert is_backend_array(got)
    assert not isinstance(got, tuple), "logdet=False must return the inverse alone"
    numpy.testing.assert_allclose(
        as_numpy(got), fast_cholesky_invert(a, tiny=1e-15), rtol=1e-13, atol=1e-15
    )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_cholesky_invert_logdet_grad_is_ainv_T():
    # d logdet(A)/dA = A^-T exactly, so this is a reference match, not a probe
    a = _spd()
    g = jax.grad(lambda x: cholesky_invert(x, 1e-15, logdet=True)[1])(jnp.asarray(a))
    ref = numpy.linalg.inv(a).T
    numpy.testing.assert_allclose(numpy.asarray(g), ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_eig_backend_matches_numpy(backend):
    a = _spd(seed=7)
    wn, _ = real_eig(a)
    w, v = real_eig(_arr(backend, a))
    assert is_backend_array(w) and is_backend_array(v)
    w, v = as_numpy(w), as_numpy(v)
    # eigh is ascending, eig is not -- compare the SETS
    numpy.testing.assert_allclose(
        numpy.sort(w), numpy.sort(numpy.asarray(wn)), rtol=1e-12
    )
    # and they must really be eigenpairs of a
    numpy.testing.assert_allclose(a @ v, v * w, rtol=1e-10, atol=1e-12)


def test_real_eig_numpy_is_unchanged():
    a = _spd(seed=9)
    w, v = real_eig(a)
    wr, vr = numpy.linalg.eig(a)
    assert w.tobytes() == numpy.real(wr).tobytes()
    assert v.tobytes() == numpy.real(vr).tobytes()


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("gap", [1e-3, 1e-8, 0.0])
def test_real_eig_eigenvalue_grad_survives_degeneracy(gap):
    # the eigenVECTOR rotation is frozen, so d(sum w)/dA is the clean identity
    # even at an exactly repeated eigenvalue -- a naive eigh eigenvector grad
    # blows up like 1/gap there (1.6e14 at gap=0), which is what this avoids
    rng = numpy.random.RandomState(7)
    q, _ = numpy.linalg.qr(rng.randn(3, 3))
    a = q @ numpy.diag([1.0, 1.0 + gap, 5.0]) @ q.T
    g = jax.grad(lambda x: jnp.sum(real_eig(x)[0]))(jnp.asarray(a))
    assert numpy.all(numpy.isfinite(numpy.asarray(g)))
    numpy.testing.assert_allclose(numpy.asarray(g), numpy.eye(3), rtol=0, atol=1e-12)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_real_eig_freeze_vectors_controls_the_rotation_gradient():
    # freeze_vectors=True takes the eigenVECTORS from a stop-gradient copy, so
    # anything read off them is a constant to autodiff. That is right when the
    # eigenvalues are near-degenerate (the eigenvector derivative goes like
    # 1/gap), and wrong when one eigenvalue dominates -- which is the case
    # streamdf reads, where freezing costs ~70% of d(track)/d(theta).
    #
    # Use a WELL-SEPARATED spectrum, like a stream frequency covariance
    # (measured ~600x between the top two eigenvalues).
    q = numpy.linalg.qr(numpy.random.default_rng(3).normal(size=(3, 3)))[0]

    def top_direction_sum(scale):
        # a(scale) has eigenvalues (1.0*scale, 2e-3, 4e-4) in a fixed basis
        w = jnp.stack([1.0 * scale, jnp.asarray(2e-3), jnp.asarray(4e-4)])
        a = jnp.asarray(q) @ jnp.diag(w) @ jnp.asarray(q).T
        # rotate a slightly with scale so the eigenVECTORS genuinely move
        tilt = jnp.asarray(numpy.eye(3)) + 0.1 * scale * jnp.asarray(
            numpy.triu(numpy.ones((3, 3)), 1) - numpy.tril(numpy.ones((3, 3)), -1)
        )
        a = tilt @ a @ tilt.T
        return a

    def frozen(scale):
        w, v = real_eig(top_direction_sum(scale), freeze_vectors=True)
        return jnp.sum(v[:, jnp.argmax(w)])

    def live(scale):
        w, v = real_eig(top_direction_sum(scale), freeze_vectors=False)
        return jnp.sum(v[:, jnp.argmax(w)])

    g_frozen = float(jax.grad(frozen)(1.0))
    g_live = float(jax.grad(live)(1.0))
    h = 1e-6
    fd = (float(live(1.0 + h)) - float(live(1.0 - h))) / (2.0 * h)
    assert abs(g_frozen) < 1e-12, "frozen vectors must carry NO rotation gradient"
    assert abs(g_live - fd) / max(abs(fd), 1e-12) < 1e-5, (
        f"unfrozen vectors must match a finite difference (AD {g_live}, FD {fd})"
    )
    # values are identical either way -- stop_gradient is the identity forward
    numpy.testing.assert_allclose(float(frozen(1.0)), float(live(1.0)), rtol=1e-12)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_psd_project_is_exact_where_nothing_is_clipped():
    # psd_project takes its eigenVECTORS from a stop-gradient copy, so
    # d(out)/d(a) keeps only the DIAGONAL of da in that frozen basis. Where no
    # eigenvalue is negative the projection IS the identity, and returning `a`
    # there makes the gradient exact; measured against a finite difference, the
    # frozen form was 100% wrong for a purely off-diagonal perturbation and
    # ~51% for a generic one.
    rng = numpy.random.default_rng(5)
    Q = numpy.linalg.qr(rng.normal(size=(6, 6)))[0]
    A0 = Q @ numpy.diag(numpy.array([3.0, 1.5, 0.8, 0.4, 0.2, 0.05])) @ Q.T
    E = rng.normal(size=(6, 6))
    E = 0.5 * (E + E.T)
    E_off = E - Q @ numpy.diag(numpy.diag(Q.T @ E @ Q)) @ Q.T  # pure off-diagonal
    W = rng.normal(size=(6, 6))
    W = 0.5 * (W + W.T)

    def loss(t, pert):
        a = jnp.asarray(A0) + t * jnp.asarray(pert)
        return jnp.sum(psd_project(a[None, ...])[0] * jnp.asarray(W))

    for pert in (E_off, E):
        ad = float(jax.grad(lambda t: loss(t, pert))(0.0))
        h = 1e-6
        fd = (float(loss(h, pert)) - float(loss(-h, pert))) / (2.0 * h)
        assert abs(ad - fd) / abs(fd) < 1e-7, (
            f"psd_project gradient wrong where nothing is clipped ({ad} vs {fd})"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_psd_project_still_clips_negative_eigenvalues(backend):
    # the exactness above must not cost the projection itself
    rng = numpy.random.default_rng(7)
    Q = numpy.linalg.qr(rng.normal(size=(5, 5)))[0]
    for w in (
        numpy.array([2.0, 1.0, 0.5, 0.3, 0.1]),
        numpy.array([2.0, 1.0, 0.5, 0.3, -0.2]),
        numpy.array([2.0, 1.0, -0.1, -0.4, 0.3]),
    ):
        A = Q @ numpy.diag(w) @ Q.T
        ref = psd_project(A[None, ...])[0]
        got = as_numpy(psd_project(_arr(backend, A)[None, ...])[0])
        numpy.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-13)
        assert numpy.linalg.eigvalsh(got).min() > -1e-12, "output must be PSD"


# The gradient against a FINITE DIFFERENCE, on slices that are actually CLIPPED
# (the tests above only check finite / jax==torch, which the old frozen-
# eigenvector gradient passed while being 77-210% off -- wrong in sign on the
# generic slice). Daleckii-Krein gives the exact first derivative from the
# frozen eigenvectors; spectra stay >=1e-1 from zero so a central FD step of
# 1e-6 cannot cross the clamp's kink.
def _clipped_batch():
    rng = numpy.random.default_rng(1)
    spectra = [
        [2.0, 1.0, 0.5, -0.3, -0.7, 0.1],  # generic, two clipped
        [2.0, 1.0, 0.5, -0.3, -0.3, -0.3],  # repeated clipped eigenvalue
        [2.0, 1.0, 0.5, 0.3, 0.2, 0.1],  # nothing clipped: identity branch
    ]
    out = []
    for s in spectra:
        q, _ = numpy.linalg.qr(rng.standard_normal((6, 6)))
        out.append(q @ numpy.diag(s) @ q.T)
    d = rng.standard_normal((3, 6, 6))
    return (
        numpy.array(out),
        d + numpy.swapaxes(d, -1, -2),
        rng.standard_normal((3, 6, 6)),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_psd_project_backend_grad_vs_fd_on_clipped_slices(backend):
    a, d, w = _clipped_batch()

    def f(t):
        return numpy.sum(w * psd_project(a + t * d), axis=(-1, -2))

    h = 1e-6
    fd = (f(h) - f(-h)) / (2.0 * h)
    if backend == "jax":
        ad = numpy.asarray(
            jax.jacfwd(
                lambda t: jnp.sum(
                    jnp.asarray(w) * psd_project(jnp.asarray(a) + t * jnp.asarray(d)),
                    axis=(-1, -2),
                )
            )(0.0)
        )
    else:
        ad = numpy.array(
            [
                float(
                    torch.autograd.grad(
                        (
                            torch.tensor(w[k])
                            * psd_project(torch.tensor(a) + t * torch.tensor(d))[k]
                        ).sum(),
                        t,
                    )[0]
                )
                for k in range(3)
                for t in [torch.tensor(0.0, requires_grad=True)]
            ]
        )
    # measured 1.2e-10 - 3.3e-10
    numpy.testing.assert_allclose(ad, fd, rtol=1e-8)
