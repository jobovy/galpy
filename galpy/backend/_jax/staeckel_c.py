###############################################################################
#   galpy.backend._jax.staeckel_c
#
#   jax wrappers for the compiled Staeckel C actions. Two entry points:
#     * c_value: a jax.pure_callback forwarding a numpy-in/numpy-out C wrapper's
#       VALUES under a jax trace (used for the frequency/angle values, which are
#       not differentiated here -- their gradients are second-order objects).
#     * actions_with_jac: a jax.custom_vjp whose forward is a pure_callback to
#       the C entry that returns (jr, jz) AND the full 2x5 Jacobian
#       d(jr,jz)/d(R,vR,vT,z,vz) (assembled natively in C); the backward is a
#       matvec of that Jacobian. Registered exactly like the C-STM custom_vjp
#       (orbit_stm), so users' jit/grad stay traceable. First-order only.
###############################################################################
import numpy


def c_value(host, coords, nout):
    """Forward a numpy-in/numpy-out C `host` under a jax trace (value only).

    host : callable taking len(coords) numpy (N,) arrays, returning `nout`
        numpy (N,) arrays. coords : tuple of traced jax (N,) arrays. The coords
        are stop-gradient'd in, so no JVP rule is engaged (no gradient here).
    """
    import jax

    coords = tuple(jax.lax.stop_gradient(c) for c in coords)
    shape, dtype = coords[0].shape, coords[0].dtype

    def _host_np(*args):
        out = host(*(numpy.asarray(a, dtype=numpy.float64) for a in args))
        return tuple(numpy.asarray(o, dtype=dtype) for o in out)

    return jax.pure_callback(
        _host_np,
        tuple(jax.ShapeDtypeStruct(shape, dtype) for _ in range(nout)),
        *coords,
        vmap_method="sequential",
    )


def actions_with_jac(host_jac, coords):
    """Differentiable (jr, jz) with the native-C Jacobian as the vjp residual.

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr, jz, jac) with jr,jz (N,) and jac (N,2,5). coords : tuple of 5
        traced jax (N,) arrays (R,vR,vT,z,vz).
    """
    import jax
    import jax.numpy as jnp

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
        # grad_k = ct_jr * dJr/dx_k + ct_jz * dJz/dx_k
        g = ct_jr[:, None] * jac[:, 0, :] + ct_jz[:, None] * jac[:, 1, :]  # (N,5)
        return (tuple(g[:, k] for k in range(5)),)

    _actions.defvjp(_fwd, _bwd)
    return _actions(coords)


def actionsfreqs_with_jac(host_jac, coords):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz) with the native-C fused (5,5)
    Jacobian d(jr,jz,Omega)/d(R,vR,vT,z,vz) as the vjp residual (#131).

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr,jz,Or,Op,Oz,jac) with the values (N,) and jac (N,5,5).
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
            tuple(jax.ShapeDtypeStruct(shape, dtype) for _ in range(5))
            + (jax.ShapeDtypeStruct(shape + (5, 5), dtype),),
            *cs,
            vmap_method="sequential",
        )

    @jax.custom_vjp
    def _af(cs):
        out = _call(cs)
        return out[:5]

    def _fwd(cs):
        out = _call(cs)
        return out[:5], out[5]  # residual = the (5,5) Jacobian

    def _bwd(jac, ct):
        # grad_k = sum_o ct_o * jac[:,o,k]  (o over jr,jz,Or,Op,Oz)
        g = sum(ct[o][:, None] * jac[:, o, :] for o in range(5))  # (N,5)
        return (tuple(g[:, k] for k in range(5)),)

    _af.defvjp(_fwd, _bwd)
    return _af(coords)


def actionsfreqsangles_with_jac(host_jac, coords, phi):
    """Differentiable (jr,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez) with the
    native-C stacked (8,5) Jacobian d(that stack w/o phi)/d(R,vR,vT,z,vz) as the vjp
    residual (#131 PR-B). phi enters analytically (d anglephi/dphi==1) via the plain
    remainder(anglephi_raw+phi,2pi) below, so jax handles its gradient/wrap.

    host_jac : callable taking 5 numpy (N,) coord arrays, returning
        (jr,jz,Or,Op,Oz,angler,anglephi_raw,anglez, jac) with the 8 values (N,) --
        angler/anglez already wrapped to [0,2pi), anglephi WITHOUT phi -- and
        jac (N,8,5) = [ojac(5,5) stacked on ajac(3,5)]. coords : 5 traced (N,) arrays;
        phi : traced (N,) array.
    """
    import jax
    import jax.numpy as jnp

    shape, dtype = coords[0].shape, coords[0].dtype

    def _host(*cs):
        out = host_jac(*(numpy.asarray(c, dtype=numpy.float64) for c in cs))
        return tuple(numpy.asarray(o, dtype=dtype) for o in out)

    def _call(cs):
        cs = tuple(jax.lax.stop_gradient(c) for c in cs)
        return jax.pure_callback(
            _host,
            tuple(jax.ShapeDtypeStruct(shape, dtype) for _ in range(8))
            + (jax.ShapeDtypeStruct(shape + (8, 5), dtype),),
            *cs,
            vmap_method="sequential",
        )

    @jax.custom_vjp
    def _afa(cs):
        return _call(cs)[:8]

    def _fwd(cs):
        out = _call(cs)
        return out[:8], out[8]  # residual = the stacked (8,5) Jacobian

    def _bwd(jac, ct):
        # grad_k = sum_o ct_o * jac[:,o,k]  (o over jr,jz,Or,Op,Oz,angler,anglephi,anglez)
        g = sum(ct[o][:, None] * jac[:, o, :] for o in range(8))  # (N,5)
        return (tuple(g[:, k] for k in range(5)),)

    _afa.defvjp(_fwd, _bwd)
    raw = _afa(coords)  # 8 values, differentiable w.r.t. the 5 coords via ajac/ojac
    anglephi = jnp.remainder(raw[6] + phi, 2.0 * jnp.pi)
    return raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], anglephi, raw[7]
