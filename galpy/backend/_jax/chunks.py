###############################################################################
#   galpy.backend._jax.chunks: a row-wise function over chunks of rows, traced.
###############################################################################
import jax
import jax.numpy as jnp


def map_row_chunks(f, args, step):
    """``f(*args)`` as a ``jax.lax.map`` over chunks of ``step`` rows, the
    rows padded to a whole number of chunks with copies of the first row"""
    n = args[0].shape[0]
    nc = -(-n // step)

    def chunked(a):
        a = jnp.concatenate(
            [a, jnp.broadcast_to(a[:1], (nc * step - n,) + a.shape[1:])]
        )
        return jnp.reshape(a, (nc, step) + a.shape[1:])

    outs = jax.lax.map(lambda c: f(*c), tuple(chunked(a) for a in args))
    return tuple(jnp.reshape(o, (nc * step,) + o.shape[2:])[:n] for o in outs)
