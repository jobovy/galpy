###############################################################################
#   galpy.backend.linalg: backend-agnostic linear-algebra primitives.
#
#   numpy stays byte-identical to the plain numpy.linalg computation; jax/torch
#   evaluate the same operation natively so the result is autodifferentiable.
###############################################################################
import numpy

from ._namespaces import is_backend_array, name_of_namespace
from ._resolver import get_namespace

__all__ = ["cholesky_invert", "psd_project", "real_eig"]


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


def real_eig(a):
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
    _, evecs = xp.linalg.eigh(_stop_gradient(a, name))
    evecs = _stop_gradient(evecs, name)
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
    -> a BATCHED, differentiable projection: the eigenvectors are taken from a
    stop-gradient copy of ``a`` (frozen structure) and the eigenvalues are
    re-derived from the live ``a`` as ``diag(V^T a V)``, so the gradient flows
    through the eigenvalue magnitudes (and the clamp) WITHOUT the singular
    ``eigh`` JVP -- naive ``eigh(a)`` in the gradient path yields NaN gradients
    at repeated eigenvalues (routine once several noise eigenvalues are clamped
    to the same zero). The eigenvector ROTATION is treated as frozen (a
    stop-gradient hyperparameter, like galpy's other frozen-structure backend
    reconstructions); the eigenvalue-magnitude sensitivity dominates.
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
    a_frozen = _stop_gradient(a, name)
    _, evecs = xp.linalg.eigh(a_frozen)
    evecs = _stop_gradient(evecs, name)
    evecsT = xp.swapaxes(evecs, -1, -2)
    # diag(V^T a V): differentiable in a, no eigh JVP
    evals_live = xp.sum(evecsT * xp.swapaxes(a @ evecs, -1, -2), axis=-1)
    evals = xp.clip(evals_live, 0.0, None)
    return (evecs * evals[..., None, :]) @ evecsT


def _stop_gradient(a, name):
    # only called from the backend branch of psd_project (name is jax or torch)
    if name == "jax":
        import jax

        return jax.lax.stop_gradient(a)
    return a.detach()  # torch
