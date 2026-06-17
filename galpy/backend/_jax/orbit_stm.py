###############################################################################
#   galpy.backend._jax.orbit_stm
#
#   jax.custom_vjp wrapping galpy's compiled C variational integrator for
#   differentiable fast orbit integration. The forward is a jax.pure_callback to
#   the (non-differentiable) C integrator, which also returns the
#   state-transition matrix M(t)=d x(t)/d x(0); the custom vjp applies
#   sum_t M(t)^T cotangent[t] for the IC gradient. See
#   galpy.backend._reference.inbackend_stm for the convention (Orbit order,
#   IC-only gradients).
###############################################################################
import numpy

from .._reference.inbackend_stm import c_stm_forward


def integrate(pot, vxvv, ts, *, method="dop853_c", rtol=1e-10, atol=1e-10):
    """Differentiable C-STM orbit integration under jax.

    vxvv: jax array (6,) or (N,6), Orbit order. Returns (nt,6) or (N,nt,6).
    Differentiable w.r.t. vxvv. pot/ts/method are static (closed over).
    """
    import jax
    import jax.numpy as jnp

    ts_np = numpy.asarray(ts, dtype=numpy.float64)
    nt = ts_np.shape[0]

    def _host(vxvv_np):
        xt, M = c_stm_forward(
            pot, numpy.asarray(vxvv_np, dtype=numpy.float64), ts_np, method, rtol, atol
        )
        return xt, M

    def _call(v):
        single = v.ndim == 1
        n6 = (nt, 6) if single else (v.shape[0], nt, 6)
        n66 = (nt, 6, 6) if single else (v.shape[0], nt, 6, 6)
        return jax.pure_callback(
            _host,
            (jax.ShapeDtypeStruct(n6, v.dtype), jax.ShapeDtypeStruct(n66, v.dtype)),
            v,
            vmap_method="expand_dims",
        )

    @jax.custom_vjp
    def _stm(v):
        xt, _ = _call(v)
        return xt

    def _fwd(v):
        xt, M = _call(v)
        return xt, M  # residual = M

    def _bwd(M, ct):
        if M.ndim == 3:  # single: M (nt,6,6), ct (nt,6) -> (6,)
            vbar = jnp.einsum("tab,ta->b", M, ct)
        else:  # batch: M (N,nt,6,6), ct (N,nt,6) -> (N,6)
            vbar = jnp.einsum("ntab,nta->nb", M, ct)
        return (vbar,)

    _stm.defvjp(_fwd, _bwd)
    return _stm(vxvv)
