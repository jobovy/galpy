###############################################################################
#   galpy.backend.linalg: backend-agnostic linear-algebra primitives.
#
#   numpy stays byte-identical to the plain numpy.linalg computation; jax/torch
#   evaluate the same operation natively so the result is autodifferentiable.
###############################################################################
import numpy

from ._namespaces import (
    asarray_on_device,
    device_of,
    is_backend_array,
    name_of_namespace,
)
from ._resolver import get_namespace

__all__ = ["cholesky_invert", "psd_project", "real_eig", "solve_tridiagonal"]


def cholesky_invert(a, tiny, logdet=False):
    """Inverse (and log-determinant) of a symmetric positive-definite ``a``.

    The backend twin of ``galpy.util.fast_cholesky_invert``, regularised the same
    way (``a + sum(diag(a)) * tiny * I``), so callers must pass the SAME ``tiny``
    they would give the scipy routine -- galpy's default is 1e-9 but streamdf
    asks for 1e-15.

    numpy ``a`` -> defers to the scipy routine, byte-identical. A backend ``a``
    -> ``cholesky``/triangular inverse on its own namespace, so the result (and
    the logdet) is differentiable in ``a``. Unlike ``eigh``, the Cholesky JVP is
    well defined for any SPD input, so no frozen-structure trick is needed here.
    """
    if not is_backend_array(a):
        from ..util import fast_cholesky_invert

        return fast_cholesky_invert(a, tiny=tiny, logdet=logdet)
    xp = get_namespace(a)
    n = a.shape[0]
    reg = xp.sum(xp.linalg.diagonal(a)) * tiny
    chol = xp.linalg.cholesky(a + reg * xp.eye(n, dtype=a.dtype))
    cholinv = xp.linalg.inv(chol)
    ainv = xp.matmul(xp.matrix_transpose(cholinv), cholinv)
    if logdet:
        return ainv, 2.0 * xp.sum(xp.log(xp.linalg.diagonal(chol)))
    return ainv


def real_eig(a, freeze_vectors=True):
    """Real eigenvalues/eigenvectors of a symmetric ``a``, as ``(w, v)``.

    numpy ``a`` -> ``numpy.linalg.eig`` with the real part taken, byte-identical
    to the inline expression it replaces (numpy>=2.5 returns a complex result
    even for real eigenvalues).

    A backend ``a`` -> ``eigh``, which is what these matrices (the symmetric
    dO/dJ = d^2H/dJ^2 and the frequency covariance built from it) actually want.
    The eigenVECTORS come from a stop-gradient copy and the eigenVALUES are
    re-derived live as ``diag(V^T a V)``, so the gradient flows through the
    eigenvalues only. Measured on a 3x3 with a controlled eigenvalue gap: the
    eigenVALUE gradient stays finite even at exact degeneracy, but the
    eigenVECTOR gradient diverges as 1/gap -- 6e2 at gap 1e-3, 6e7 at 1e-8, 1.6e14
    at gap 0 (finite, not NaN, but useless). A frequency covariance routinely has
    near-degenerate eigenvalues, and streamdf reads an eigenVECTOR off this
    (``_dsigomeanProgDirection``), so that rotation is kept a frozen
    hyperparameter -- which is what streamdf already assumes of it.

    NOTE the eigenvalue ORDER differs from ``eig``'s (``eigh`` is ascending);
    ``w`` and ``v`` stay paired, and every galpy caller sorts/argmaxes for
    itself rather than relying on the order.
    """
    if not is_backend_array(a):
        w, v = numpy.linalg.eig(a)
        return numpy.real(w), numpy.real(v)
    xp = get_namespace(a)
    name = name_of_namespace(xp)
    if freeze_vectors:
        _, evecs = xp.linalg.eigh(_stop_gradient(a, name))
        evecs = _stop_gradient(evecs, name)
    else:
        # The caller has established that the eigenvector it READS is separated
        # from the rest by a large eigenvalue ratio, so the 1/gap sensitivity
        # that motivates freezing does not bite and the rotation must carry its
        # gradient. Freezing it silently drops d(rotation)/d(theta).
        _, evecs = xp.linalg.eigh(a)
    evecsT = xp.matrix_transpose(evecs)
    evals = xp.sum(evecsT * xp.matrix_transpose(a @ evecs), axis=-1)
    return evals, evecs


def psd_project(a):
    """Nearest positive-semidefinite projection of a batch of symmetric matrices.

    For each ``(D, D)`` slice along the leading axis of ``a`` (shape
    ``(..., D, D)``), clamp the (symmetric) eigenvalues at zero and rebuild:
    ``V diag(max(w, 0)) V^T``. This is the standard nearest-PSD projection in
    Frobenius norm; galpy uses it to sanitise a smoothed covariance series whose
    small noise eigenvalues can dip slightly negative.

    numpy ``a`` -> the plain per-slice ``numpy.linalg.eigh`` computation
    (byte-identical to the inline loop it replaces). A backend ``a`` (jax/torch)
    -> a BATCHED, differentiable projection: the eigendecomposition is taken of
    a stop-gradient copy of ``a``, and the result carries the EXACT first
    derivative through the Daleckii-Krein divided
    differences (see below) WITHOUT the singular ``eigh`` JVP -- naive
    ``eigh(a)`` in the gradient path yields NaN gradients at repeated
    eigenvalues (routine once several noise eigenvalues are clamped to the same
    zero).
    """
    if not is_backend_array(a):
        out = numpy.array(a, dtype=float)
        flat = out.reshape((-1,) + out.shape[-2:])
        for k in range(flat.shape[0]):
            evals, evecs = numpy.linalg.eigh(flat[k])
            evals = numpy.clip(evals, 0.0, None)
            flat[k] = (evecs * evals) @ evecs.T
        return out
    xp = get_namespace(a)
    name = name_of_namespace(xp)
    lam, evecs = xp.linalg.eigh(_stop_gradient(a, name))
    lam, evecs = _stop_gradient(lam, name), _stop_gradient(evecs, name)
    evecsT = xp.swapaxes(evecs, -1, -2)
    projected = (evecs * xp.clip(lam, 0.0, None)[..., None, :]) @ evecsT
    # Exact first derivative with the eigenvectors FROZEN (no eigh JVP, which
    # blows up like 1/gap at the near-degenerate noise eigenvalues this
    # sanitises): Daleckii-Krein, dP = V (F o V^T dA V) V^T with divided
    # differences F_ij = (f(l_i) - f(l_j)) / (l_i - l_j) of f = max(l, 0) --
    # exactly 1 (both > 0), 0 (both <= 0), or l+/(l+ - l-) in (0, 1): bounded,
    # so repeated eigenvalues are harmless. Grafted: the value is `projected`.
    pos = lam > 0.0
    pi, pj = pos[..., :, None], pos[..., None, :]
    li, lj = lam[..., :, None], lam[..., None, :]
    mixed = pi != pj
    den = xp.where(mixed, li - lj, xp.ones_like(li - lj))
    F = xp.where(
        pi & pj,
        xp.ones_like(den),
        xp.where(mixed, xp.where(pi, li, lj) / xp.where(pi, den, -den), 0.0 * den),
    )
    donor = evecs @ (F * (evecsT @ a @ evecs)) @ evecsT
    projected = projected + (donor - _stop_gradient(donor, name))
    # Where NOTHING is clipped the projection IS the identity: return `a` itself
    anyneg = xp.any(lam < 0.0, axis=-1)
    return xp.where(anyneg[..., None, None], projected, a)


def _stop_gradient(a, name):
    # only called from the backend branch of psd_project (name is jax or torch)
    if name == "jax":
        import jax

        return jax.lax.stop_gradient(a)
    return a.detach()  # torch


def solve_tridiagonal(xp, a, b, c, d):
    """Solve a tridiagonal system by parallel cyclic reduction.

    Row ``i`` reads ``a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]`` (``a[0]``
    and ``c[-1]`` are ignored); ``d`` may carry trailing right-hand-side
    columns. Each of the ~log2(n) steps eliminates every row's neighbours at
    distance ``s`` with whole-array arithmetic, so the solve is ``O(n log n)``
    work in ~log2(n) vectorized passes -- no ``(n, n)`` matrix and no
    sequential loop over rows -- and differentiable in all four inputs. No
    pivoting: meant for DIAGONALLY DOMINANT systems (cubic-spline second
    derivatives), where it is stable.
    """
    n = b.shape[0]
    tail = tuple(d.shape[1:])

    def col(v):  # broadcast a per-row coefficient against d's trailing columns
        return v.reshape((n,) + (1,) * len(tail))

    idx = numpy.arange(n)
    dev = device_of(b)

    def mask(m):  # a 0/1 row mask on b's namespace, device and dtype
        return asarray_on_device(xp, m, dev) * (b[:1] * 0.0 + 1.0)

    one = b * 0.0 + 1.0
    a = a * mask(idx > 0)
    c = c * mask(idx < n - 1)
    s = 1
    while s < n:
        # neighbours at distance s by GATHER with a validity mask (outside:
        # the identity row 0, 1, 0 | 0). Every pass then has the same shapes,
        # so eager jax compiles its primitives once rather than per stride
        # (slicing at a new offset each pass was ~6 s of XLA compiles).
        im, ip = numpy.clip(idx - s, 0, n - 1), numpy.clip(idx + s, 0, n - 1)
        vm, vp = mask(idx - s >= 0), mask(idx + s < n)
        am, cm, dm = a[im] * vm, c[im] * vm, d[im] * col(vm)
        ap, cp, dp = a[ip] * vp, c[ip] * vp, d[ip] * col(vp)
        bm = b[im] * vm + one * (1.0 - vm)
        bp = b[ip] * vp + one * (1.0 - vp)
        alpha, gamma = -a / bm, -c / bp
        a, c = alpha * am, gamma * cp
        b = b + alpha * cm + gamma * ap
        d = d + col(alpha) * dm + col(gamma) * dp
        s *= 2
    return d / col(b)
