###############################################################################
#   galpy.backend._jax.cond: a traced two-way branch that runs only one arm.
###############################################################################
import jax


def cond(pred, if_true, if_false):
    """``jax.lax.cond`` on zero-argument branches (a select under vmap)."""
    return jax.lax.cond(pred, if_true, if_false)
