###############################################################################
#   galpy.backend._jax.adiabatic_c
#
#   jax wrapper for the compiled Adiabatic C actions. actions_with_jac is a
#   jax.custom_vjp whose forward is a pure_callback to the C entry returning
#   (jr, jz) AND the fused 2x5 Jacobian d(jr,jz)/d(R,vR,vT,z,vz) (assembled
#   natively in C); the backward is a matvec of that Jacobian. Mirrors the
#   Staeckel staeckel_c.actions_with_jac (#131 Adiabatic PR-2a). First-order.
###############################################################################
import numpy


def actions_with_jac(host_jac, coords):
    """Differentiable (jr, jz) with the native-C Jacobian as the vjp residual.

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr, jz, jac) with jr,jz (N,) and jac (N,2,5). coords : tuple of 5
        traced jax (N,) arrays (R,vR,vT,z,vz).
    """
    import jax

    shape, dtype = coords[0].shape, coords[0].dtype

    def _host(*cs):
        jr, jz, jac = host_jac(*(numpy.asarray(c, dtype=numpy.float64) for c in cs))
        return (
            numpy.asarray(jr, dtype=dtype),
            numpy.asarray(jz, dtype=dtype),
            numpy.asarray(jac, dtype=dtype),
        )

    def _call(cs):
        cs = tuple(jax.lax.stop_gradient(c) for c in cs)
        return jax.pure_callback(
            _host,
            (
                jax.ShapeDtypeStruct(shape, dtype),
                jax.ShapeDtypeStruct(shape, dtype),
                jax.ShapeDtypeStruct(shape + (2, 5), dtype),
            ),
            *cs,
            vmap_method="sequential",
        )

    @jax.custom_vjp
    def _actions(cs):
        jr, jz, _ = _call(cs)
        return jr, jz

    def _fwd(cs):
        jr, jz, jac = _call(cs)
        return (jr, jz), jac  # residual = the 2x5 Jacobian

    def _bwd(jac, ct):
        ct_jr, ct_jz = ct  # each (N,)
        g = ct_jr[:, None] * jac[:, 0, :] + ct_jz[:, None] * jac[:, 1, :]  # (N,5)
        return (tuple(g[:, k] for k in range(5)),)

    _actions.defvjp(_fwd, _bwd)
    return _actions(coords)


def ecczmax_with_jac(host_jac, coords):
    """Differentiable (ecc, zmax, rperi, rap) with the native-C (N,4,5) Jacobian
    as the vjp residual. Mirrors actions_with_jac (#131 Adiabatic PR-2c).

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (ecc, zmax, rperi, rap, jac) with the 4 values (N,) and jac (N,4,5).
    coords : tuple of 5 traced jax (N,) arrays (R,vR,vT,z,vz).
    """
    import jax

    shape, dtype = coords[0].shape, coords[0].dtype

    def _host(*cs):
        out = host_jac(*(numpy.asarray(c, dtype=numpy.float64) for c in cs))
        return tuple(numpy.asarray(o, dtype=dtype) for o in out)

    def _call(cs):
        cs = tuple(jax.lax.stop_gradient(c) for c in cs)
        return jax.pure_callback(
            _host,
            tuple(jax.ShapeDtypeStruct(shape, dtype) for _ in range(4))
            + (jax.ShapeDtypeStruct(shape + (4, 5), dtype),),
            *cs,
            vmap_method="sequential",
        )

    @jax.custom_vjp
    def _ez(cs):
        return _call(cs)[:4]

    def _fwd(cs):
        out = _call(cs)
        return out[:4], out[4]  # residual = the (N,4,5) Jacobian

    def _bwd(jac, ct):
        g = sum(ct[o][:, None] * jac[:, o, :] for o in range(4))  # (N,5)
        return (tuple(g[:, k] for k in range(5)),)

    _ez.defvjp(_fwd, _bwd)
    return _ez(coords)
