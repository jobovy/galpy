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


def map_bucketed_rows(fs, bucket, args, step):
    """``fs[b](*rows)`` for the rows of bucket ``b``, under jax: the rows
    sorted by bucket, padded with copies of the last (largest-bucket) row to a
    whole number of chunks of ``step`` rows, and ``jax.lax.map`` over the
    chunks with ``jax.lax.switch`` to the function of each chunk's largest
    bucket (static shapes: one compiled branch per function); the outputs in
    the rows' original order"""
    n = args[0].shape[0]
    nc = -(-n // step)
    perm = jnp.argsort(bucket, stable=True)

    def chunked(a):
        a = a[perm]
        a = jnp.concatenate(
            [a, jnp.broadcast_to(a[-1:], (nc * step - n,) + a.shape[1:])]
        )
        return jnp.reshape(a, (nc, step) + a.shape[1:])

    cb = jnp.max(chunked(bucket), axis=1)
    branches = [lambda c, f=f: tuple(f(*c)) for f in fs]
    outs = jax.lax.map(
        lambda x: jax.lax.switch(x[0], branches, x[1]),
        (cb, tuple(chunked(a) for a in args)),
    )
    inv = jnp.argsort(perm)
    return tuple(jnp.reshape(o, (nc * step,) + o.shape[2:])[:n][inv] for o in outs)
