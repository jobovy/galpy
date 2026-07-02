###############################################################################
#   galpy.backend._jax.staeckel_c
#
#   jax.pure_callback wrapper forwarding the compiled Staeckel C wrappers'
#   VALUES under a jax trace (jit/grad/vmap). No JVP rule is defined here: the
#   actions' first-order gradients are grafted from the in-backend t^2 donor
#   quadrature in actionAngleStaeckel (graft_gradient stop-gradients this
#   callback's outputs), and frequency/angle gradients are second-order
#   objects (not implemented) -- differentiating them raises jax's standard
#   pure_callback-not-differentiable error.
###############################################################################
import numpy


def c_value(host, coords, nout):
    """Forward a numpy-in/numpy-out C `host` under a jax trace.

    host : callable taking len(coords) numpy (N,) arrays, returning `nout`
        numpy (N,) arrays (the Staeckel C wrapper closure).
    coords : tuple of traced jax (N,) coordinate arrays.
    nout : number of outputs; all share coords[0]'s shape and dtype.
    """
    import jax

    # Stop-gradient the coords INTO the callback: their tangents become
    # symbolic zeros, so grad/jvp never engages pure_callback's (unsupported)
    # JVP rule -- the actions' gradient is grafted from the t^2 donor instead.
    coords = tuple(jax.lax.stop_gradient(c) for c in coords)
    shape, dtype = coords[0].shape, coords[0].dtype

    def _host_np(*args):
        out = host(*(numpy.asarray(a, dtype=numpy.float64) for a in args))
        return tuple(numpy.asarray(o, dtype=dtype) for o in out)

    # vmap_method="sequential": the C wrappers are strictly 1-D (len(R) batch),
    # so a user vmap must call the host per batch element, not add axes.
    return jax.pure_callback(
        _host_np,
        tuple(jax.ShapeDtypeStruct(shape, dtype) for _ in range(nout)),
        *coords,
        vmap_method="sequential",
    )
