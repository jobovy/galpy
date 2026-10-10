###############################################################################
#   galpy.backend._torch.adiabatic_c
#
#   torch wrappers for the compiled Adiabatic C actions: the same
#   autograd.Function ties as the Staeckel ones (forward returning the values
#   AND the C-assembled Jacobian; backward a matvec of it), so they are shared.
###############################################################################
from .staeckel_c import actions_with_jac, ecczmax_with_jac  # noqa: F401
