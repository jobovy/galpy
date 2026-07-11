###############################################################################
#   galpy.backend.linalg: backend-agnostic linear-algebra primitives.
#
#   numpy stays byte-identical to the plain numpy.linalg computation; jax/torch
#   evaluate the same operation natively so the result is autodifferentiable.
###############################################################################
import numpy

from ._namespaces import is_backend_array, name_of_namespace
from ._resolver import get_namespace

__all__ = ["psd_project"]


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
