###############################################################################
#   galpy.backend._reference: differentiable reference implementations.
#
#   In-backend ODE orbit integration (diffrax / torchdiffeq) of galpy's
#   backend-agnostic forces -- the fully-differentiable orbit path for jax/torch
#   and the independent correctness reference for the fast C state-transition-
#   matrix path.
###############################################################################
from .inbackend_ode import integrate_orbit

__all__ = ["integrate_orbit"]
