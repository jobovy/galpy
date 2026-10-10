###############################################################################
#   galpy.backend._jax.adiabatic_c
#
#   jax wrappers for the compiled Adiabatic C actions: the same custom_vjp ties
#   as the Staeckel ones (pure_callback forward returning the values AND the
#   C-assembled Jacobian; backward a matvec of it), so they are shared.
###############################################################################
from .staeckel_c import actions_with_jac, ecczmax_with_jac  # noqa: F401
