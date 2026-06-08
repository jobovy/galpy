###############################################################################
#   Fixed-order Gauss-Legendre quadrature helpers shared by the special-function
#   fallbacks (and, later, the backend-agnostic potential/df integrals). Nodes
#   and weights are numpy constants computed once; per call they are converted
#   into the active backend's namespace so the integrand differentiates.
###############################################################################
import numpy

# Cache of (nodes, weights) on [0, 1] keyed by order, as numpy float64 constants.
_GL01_CACHE = {}


def gauss_legendre_01(n):
    """Return (nodes, weights) for n-point Gauss-Legendre quadrature on [0, 1].

    numpy float64 constants (cached). The caller converts them into the backend
    namespace with ``xp.asarray`` so they pick up the backend's array type while
    the integrand stays differentiable.
    """
    cached = _GL01_CACHE.get(n)
    if cached is None:
        x, w = numpy.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (x + 1.0)  # map [-1, 1] -> [0, 1]
        weights = 0.5 * w
        cached = (nodes, weights)
        _GL01_CACHE[n] = cached
    return cached
