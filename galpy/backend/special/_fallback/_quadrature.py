###############################################################################
#   Thin shim: the fixed-order Gauss-Legendre helpers now live in the
#   backend-agnostic galpy.backend.quadrature module. ``gauss_legendre_01``
#   stays importable from this historical location (it is re-exported here)
#   so the special-function fallbacks (hyp2f1) keep working unchanged.
###############################################################################
from ...quadrature import gauss_legendre_01

__all__ = ["gauss_legendre_01"]
