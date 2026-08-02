###############################################################################
# test_backend_coords.py: multi-backend tests for the vector/coordinate leaf
# transforms in galpy.util.coords that the stream DFs, impulse kernels, and
# stream-track accessors sit on:
#
#   rect_to_cyl_vec  (rectangular -> cylindrical velocity vectors)
#   cyl_to_rect_vec  (cylindrical -> rectangular velocity vectors)
#   spher_to_cyl     (spherical -> cylindrical positions)
#
# Before this migration each did raw numpy.cos/numpy.sin on its argument, which
# raised TypeError the moment a torch tensor flowed in -- the confirmed root
# cause of the torch reds across streamspraydf / streamgapdf / streamTrack. Now
# they dispatch through get_namespace + promote_scalars, so this proves numpy /
# jax / torch value parity and that they are differentiable end-to-end (grad of
# an output component w.r.t. an input matches the closed form).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array, use
from galpy.util import coords

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

_rng = numpy.random.default_rng(20260707)
_N = 6
_VX, _VY, _VZ = _rng.normal(size=_N), _rng.normal(size=_N), _rng.normal(size=_N)
_X, _Y, _Z = _rng.normal(size=_N), _rng.normal(size=_N), _rng.normal(size=_N)
_VR, _VT = _rng.normal(size=_N), _rng.normal(size=_N)
_R = numpy.abs(_rng.normal(size=_N)) + 0.5
_THETA = _rng.uniform(0.1, numpy.pi - 0.1, _N)
_PHI = _rng.uniform(-3.0, 3.0, _N)


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_vector_transforms_value_parity(backend_name):
    A = lambda a: _asarray(backend_name, a)
    cases = [
        (
            coords.rect_to_cyl_vec(
                numpy.asarray(_VX),
                numpy.asarray(_VY),
                numpy.asarray(_VZ),
                numpy.asarray(_X),
                numpy.asarray(_Y),
                numpy.asarray(_Z),
            ),
            coords.rect_to_cyl_vec(A(_VX), A(_VY), A(_VZ), A(_X), A(_Y), A(_Z)),
            "rect_to_cyl_vec",
        ),
        (
            coords.rect_to_cyl_vec(
                numpy.asarray(_VX),
                numpy.asarray(_VY),
                numpy.asarray(_VZ),
                numpy.asarray(_R),
                numpy.asarray(_PHI),
                numpy.asarray(_Z),
                cyl=True,
            ),
            coords.rect_to_cyl_vec(
                A(_VX), A(_VY), A(_VZ), A(_R), A(_PHI), A(_Z), cyl=True
            ),
            "rect_to_cyl_vec[cyl]",
        ),
        (
            coords.cyl_to_rect_vec(
                numpy.asarray(_VR),
                numpy.asarray(_VT),
                numpy.asarray(_VZ),
                numpy.asarray(_PHI),
            ),
            coords.cyl_to_rect_vec(A(_VR), A(_VT), A(_VZ), A(_PHI)),
            "cyl_to_rect_vec",
        ),
        (
            coords.spher_to_cyl(
                numpy.asarray(_R), numpy.asarray(_THETA), numpy.asarray(_PHI)
            ),
            coords.spher_to_cyl(A(_R), A(_THETA), A(_PHI)),
            "spher_to_cyl",
        ),
    ]
    for ref, got, label in cases:
        for a, b in zip(ref, got):
            numpy.testing.assert_allclose(
                as_numpy(b),
                numpy.asarray(a),
                rtol=1e-13,
                atol=1e-14,
                err_msg=f"{label} ({backend_name})",
            )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_scalar_inputs(backend_name):
    # A plain python float must survive promotion on every backend (torch's
    # cos/sin reject bare floats -> promote_scalars must lift them).
    R, z, phi = coords.spher_to_cyl(
        _asarray(backend_name, 2.0) if backend_name != "numpy" else 2.0, 1.0, 0.5
    )
    numpy.testing.assert_allclose(as_numpy(R), 2.0 * numpy.sin(1.0), rtol=1e-13)
    numpy.testing.assert_allclose(as_numpy(z), 2.0 * numpy.cos(1.0), rtol=1e-13)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_through(backend_name):
    # spher_to_cyl: dR/dr = sin(theta); cyl_to_rect_vec: dvx/dvr = cos(phi);
    # rect_to_cyl_vec: dvr/dvx = cos(phi) with phi = arctan2(Y, X).
    theta0, phi0, X0, Y0 = 0.7, 0.9, 1.0, 2.0
    if backend_name == "jax":
        gR = jax.grad(
            lambda r: coords.spher_to_cyl(r, jnp.asarray(theta0), jnp.asarray(phi0))[0]
        )(jnp.asarray(2.0))
        gvx = jax.grad(
            lambda vr: coords.cyl_to_rect_vec(
                vr, jnp.asarray(0.4), jnp.asarray(0.5), jnp.asarray(phi0)
            )[0]
        )(jnp.asarray(0.3))
        gvr = jax.grad(
            lambda vx: coords.rect_to_cyl_vec(
                vx,
                jnp.asarray(0.2),
                jnp.asarray(0.3),
                jnp.asarray(X0),
                jnp.asarray(Y0),
                jnp.asarray(3.0),
            )[0]
        )(jnp.asarray(0.1))
        gR, gvx, gvr = float(gR), float(gvx), float(gvr)
    else:
        r = torch.tensor(2.0, requires_grad=True)
        coords.spher_to_cyl(r, torch.tensor(theta0), torch.tensor(phi0))[0].backward()
        gR = float(r.grad)
        vr = torch.tensor(0.3, requires_grad=True)
        coords.cyl_to_rect_vec(
            vr, torch.tensor(0.4), torch.tensor(0.5), torch.tensor(phi0)
        )[0].backward()
        gvx = float(vr.grad)
        vx = torch.tensor(0.1, requires_grad=True)
        coords.rect_to_cyl_vec(
            vx,
            torch.tensor(0.2),
            torch.tensor(0.3),
            torch.tensor(X0),
            torch.tensor(Y0),
            torch.tensor(3.0),
        )[0].backward()
        gvr = float(vx.grad)
    numpy.testing.assert_allclose(gR, numpy.sin(theta0), rtol=1e-10)
    numpy.testing.assert_allclose(gvx, numpy.cos(phi0), rtol=1e-10)
    numpy.testing.assert_allclose(gvr, numpy.cos(numpy.arctan2(Y0, X0)), rtol=1e-10)


###############################################################################
# Rz_to_lambdanu / _jac / _hess: the prolate-spheroidal (R,z)->(lambda,nu)
# transform behind KuzminKutuzovStaeckelPotential. These did raw numpy.sqrt and
# numpy.zeros + in-place assignment, so a jax/torch trace over the force eval
# crashed (TracerArrayConversionError). Now they namespace-swap the sqrt (numpy
# path byte-identical) and take an is_backend_array-gated xp.stack build for
# backend inputs. z=0 is included as an edge (the general formula is exact and
# differentiable there; the numpy z==0 special-case is skipped for backends).
###############################################################################
_LN_R = numpy.array([0.5, 1.0, 1.2, 2.3, 0.8])
_LN_Z = numpy.array([0.0, 0.3, -0.7, 1.5, 0.2])


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_lambdanu_value_parity(backend_name):
    A = lambda a: _asarray(backend_name, a)
    ref_l, ref_n = coords.Rz_to_lambdanu(_LN_R, _LN_Z, ac=5.0, Delta=1.4)
    ref_jac = coords.Rz_to_lambdanu_jac(_LN_R, _LN_Z, Delta=1.4)
    ref_hess = coords.Rz_to_lambdanu_hess(_LN_R, _LN_Z, Delta=1.4)
    l, n = coords.Rz_to_lambdanu(A(_LN_R), A(_LN_Z), ac=5.0, Delta=1.4)
    jac = coords.Rz_to_lambdanu_jac(A(_LN_R), A(_LN_Z), Delta=1.4)
    hess = coords.Rz_to_lambdanu_hess(A(_LN_R), A(_LN_Z), Delta=1.4)
    numpy.testing.assert_allclose(as_numpy(l), ref_l, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(n), ref_n, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(jac), ref_jac, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(hess), ref_hess, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_kuzminkutuzov_force_traces(backend_name):
    from galpy.potential import (
        KuzminKutuzovStaeckelPotential,
        evaluateR2derivs,
        evaluateRforces,
        evaluatezforces,
    )

    kp = KuzminKutuzovStaeckelPotential(amp=1.3, ac=5.0, Delta=1.0)
    R0, z0 = 1.2, 0.3
    if backend_name == "jax":
        R, z = jnp.asarray(R0), jnp.asarray(z0)
        assert isinstance(evaluateRforces(kp, R, z), jax.Array)
        gR = jax.jacfwd(lambda RR: evaluateRforces(kp, RR, z))(R)
        gz = jax.jacfwd(lambda zz: evaluatezforces(kp, R, zz))(z)
        assert bool(jnp.isfinite(gR)) and bool(jnp.isfinite(gz))
        jf = jax.jit(lambda RR, zz: evaluateRforces(kp, RR, zz))
        assert bool(jnp.isfinite(jf(R, z)))
        # autodiff self-consistency: d(Rforce)/dR == -R2deriv
        numpy.testing.assert_allclose(
            float(gR), -float(evaluateR2derivs(kp, R, z)), rtol=1e-10
        )
        # differentiable w.r.t. the Delta parameter (goes through the transform)
        gd = jax.jacfwd(
            lambda d: evaluateRforces(
                KuzminKutuzovStaeckelPotential(amp=1.3, ac=5.0, Delta=d), R, z
            )
        )(jnp.asarray(1.0))
        assert bool(jnp.isfinite(gd))
    else:
        R = torch.tensor(R0, requires_grad=True)
        z = torch.tensor(z0, requires_grad=True)
        f = evaluateRforces(kp, R, z)
        assert isinstance(f, torch.Tensor)
        (gR,) = torch.autograd.grad(f, R, create_graph=True)
        assert bool(torch.isfinite(gR))
        r2 = evaluateR2derivs(kp, R.detach(), z.detach())
        numpy.testing.assert_allclose(float(gR.detach()), -float(r2), rtol=1e-10)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_kuzminkutuzov_force_grad_fd(backend_name):
    from galpy.potential import KuzminKutuzovStaeckelPotential, evaluateRforces

    kp = KuzminKutuzovStaeckelPotential(amp=1.3, ac=5.0, Delta=1.2)
    R0, z0 = 1.3, 0.4
    if backend_name == "jax":
        g_ad = float(
            jax.jacfwd(lambda RR: evaluateRforces(kp, RR, jnp.asarray(z0)))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, requires_grad=True)
        (g,) = torch.autograd.grad(evaluateRforces(kp, R, torch.tensor(z0)), R)
        g_ad = float(g)
    # numpy central finite difference must show O(h^2) convergence to the AD grad
    f = lambda R: float(evaluateRforces(kp, R, z0))
    errs = [abs((f(R0 + h) - f(R0 - h)) / (2.0 * h) - g_ad) for h in (1e-3, 1e-4)]
    assert errs[0] < 1e-6
    assert errs[1] < errs[0] / 50.0


# --- uv_to_Rz individual-delta coercion (numpy delta under a forced backend) --
# actionAngleStaeckel's EccZmaxRperiRap with a per-point ``delta`` array feeds
# uv_to_Rz a numpy ``delta`` while u, v get promoted onto the (forced) backend;
# before promoting delta too the prolate/oblate ``delta * xp.sinh(u)`` raised
# ``numpy.ndarray * Tensor`` on torch. Reproduced under a *forced* backend (the
# resolver short-circuits to the backend, so a numpy delta reaches the backend
# ops) -- matching the ledgered test. Proves it now runs on the backend and
# matches the numpy reference (prolate & oblate, array & scalar delta).
_U = numpy.array([0.5, 0.7, 1.3, 0.2])
_V = numpy.array([1.1, 0.9, 0.4, 2.0])
_DELTA = numpy.array([0.2, 0.4, 0.6, 0.35])


@pytest.mark.parametrize("oblate", [False, True])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_uv_to_Rz_indiv_delta_forced_backend(backend_name, oblate):
    Rref, zref = coords.uv_to_Rz(_U, _V, delta=_DELTA, oblate=oblate)
    Rref2, zref2 = coords.uv_to_Rz(_U, _V, delta=0.85, oblate=oblate)
    with use(backend_name, force=True):
        # numpy u, v, delta under a forced backend: delta must be promoted too
        R, z = coords.uv_to_Rz(_U, _V, delta=_DELTA, oblate=oblate)
        assert is_backend_array(R) and is_backend_array(z)
        # a scalar (python-float) delta must also survive
        R2, z2 = coords.uv_to_Rz(_U, _V, delta=0.85, oblate=oblate)
    numpy.testing.assert_allclose(as_numpy(R), Rref, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(z), zref, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(R2), Rref2, rtol=1e-13, atol=1e-14)
    numpy.testing.assert_allclose(as_numpy(z2), zref2, rtol=1e-13, atol=1e-14)


# --- degreeDecorator's backend branch + lb_to_radec's rotation branch --------
# Two lines that a numpy-only run can never reach, and the coverage-uploading
# shards run numpy (see the backend-branch coverage note): degreeDecorator's
# `if is_backend_array(out): scale = asarray_on_device(...)` and the
# rotation-matrix branch of lb_to_radec, which numpy skips because astropy is
# present. Exercised here with a real backend array, which is the only way to
# reach them.
_LDEG = numpy.array([12.0, 200.0, 351.0])
_BDEG = numpy.array([-40.0, 5.0, 62.0])


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_lb_to_radec_backend_branch_matches_numpy(backend_name):
    # Backend-vs-numpy parity. NOTE the reference path is environment
    # dependent: with astropy present numpy uses SkyCoord, and on the
    # astropy-free test_backend shard it uses the same rotation matrix as the
    # backend. Both agree to roundoff (astropy-vs-rotation measured at
    # ~1.6e-15 rad), so the assertion holds either way.
    ref = coords.lb_to_radec(_LDEG, _BDEG, degree=True)
    with use(backend_name, force=True):
        got = coords.lb_to_radec(_LDEG, _BDEG, degree=True)
        assert is_backend_array(got), "lb_to_radec did not preserve the backend"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_degree_decorator_scales_backend_output(backend_name):
    # degree=True must scale the output columns for a backend array too; this
    # is the `is_backend_array(out)` branch of degreeDecorator.
    ref_deg = coords.lb_to_radec(_LDEG, _BDEG, degree=True)
    ref_rad = coords.lb_to_radec(
        numpy.radians(_LDEG), numpy.radians(_BDEG), degree=False
    )
    with use(backend_name, force=True):
        got_deg = coords.lb_to_radec(_LDEG, _BDEG, degree=True)
        got_rad = coords.lb_to_radec(
            numpy.radians(_LDEG), numpy.radians(_BDEG), degree=False
        )
    # the degree output is exactly 180/pi times the radian one, elementwise
    numpy.testing.assert_allclose(
        as_numpy(got_deg), numpy.degrees(as_numpy(got_rad)), rtol=1e-13, atol=0
    )
    numpy.testing.assert_allclose(as_numpy(got_deg), ref_deg, rtol=0, atol=1e-12)
    numpy.testing.assert_allclose(as_numpy(got_rad), ref_rad, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# The galcenrect -> heliocentric -> sky chain (#184).
#
# Each of these was plain @scalarDecorator, so it promoted its inputs through
# numpy and silently returned numpy even when handed a backend array. That is
# what made streamTrack's sky-frame cov() raise "Multiple namespaces" on an
# unforced mixed track and "Can't call numpy() on Tensor that requires grad"
# under a gradient.
#
# These assert the RETURN TYPE, not just the values, and that is deliberate:
# a byte-identity check cannot distinguish a working migration from one that
# does nothing, because the numpy path is unchanged either way. During this
# migration @backendNative was briefly placed as the OUTERMOST decorator, which
# marks the scalarDecorator wrapper rather than the function scalarDecorator
# inspects -- every migration was inert and every value check still passed.
# Only a type assertion catches that.
# ---------------------------------------------------------------------------
_CHAIN_CASES = {
    "galcenrect_to_XYZ": lambda c, a: c.galcenrect_to_XYZ(
        a["x"], a["y"], a["z"], Xsun=1.0, Zsun=0.02
    ),
    "galcenrect_to_vxvyvz": lambda c, a: c.galcenrect_to_vxvyvz(
        a["vx"], a["vy"], a["vz"], Xsun=1.0, Zsun=0.02
    ),
    "XYZ_to_lbd": lambda c, a: c.XYZ_to_lbd(a["x"], a["y"], a["z"], degree=False),
    "vxvyvz_to_vrpmllpmbb": lambda c, a: c.vxvyvz_to_vrpmllpmbb(
        a["vx"], a["vy"], a["vz"], a["x"], a["y"], a["z"], XYZ=True, degree=False
    ),
    "pmllpmbb_to_pmrapmdec": lambda c, a: c.pmllpmbb_to_pmrapmdec(
        a["pmll"], a["pmbb"], a["l"], a["b"], degree=False
    ),
}


def _chain_args():
    rng = numpy.random.default_rng(19)
    n = 7
    return {
        "x": rng.normal(size=n) + 1.5,
        "y": rng.normal(size=n),
        "z": 0.4 * rng.normal(size=n),
        "vx": rng.normal(size=n),
        "vy": rng.normal(size=n) + 1.0,
        "vz": 0.3 * rng.normal(size=n),
        "l": rng.random(size=n) * 5.0,
        "b": (rng.random(size=n) - 0.5),
        "pmll": rng.normal(size=n),
        "pmbb": rng.normal(size=n),
    }


def _as_backend(backend_name, arr):
    """A real backend array while the AMBIENT backend stays numpy."""
    if backend_name == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        return jnp.asarray(arr)
    import torch

    return torch.tensor(numpy.asarray(arr, dtype=float), dtype=torch.float64)


def _all_backend(res):
    """True when every array in ``res`` is a backend array (res may be a tuple)."""
    parts = res if isinstance(res, tuple) else (res,)
    return all(is_backend_array(a) for a in parts)


@pytest.mark.parametrize("shape", ["scalar", "array"])
@pytest.mark.parametrize("name", sorted(_CHAIN_CASES))
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sky_chain_preserves_backend_and_matches_numpy(backend_name, name, shape):
    # Two things this test is careful about, both learned the hard way:
    #
    # 1. UNFORCED. Under `use(..., force=True)` these recover even with the
    #    marker misplaced, because get_namespace reports the caller's intent and
    #    the body's promote_scalars re-coerces. Forcing hides the bug.
    # 2. SCALAR inputs. scalarDecorator only rewrites its arguments when they
    #    are 0-d (`if xp.asarray(args[0]).ndim == 0`), so with array inputs the
    #    marker is never consulted and the test passes either way. streamTrack
    #    calls these per-tp with scalars, which is why it was the thing that
    #    broke. Verified: with @backendNative moved outermost, scalar-in returns
    #    a tuple of numpy float64 and this test fails; array-in stays Tensor.
    args = _chain_args()
    if shape == "scalar":
        args = {k: v[0] for k, v in args.items()}
    call = _CHAIN_CASES[name]
    ref = numpy.asarray(call(coords, args))
    bargs = {k: _as_backend(backend_name, v) for k, v in args.items()}
    got = call(coords, bargs)
    assert _all_backend(got), f"{name} dropped back to numpy ({shape}, unforced)"
    flat = (
        numpy.asarray([as_numpy(a) for a in got])
        if isinstance(got, tuple)
        else as_numpy(got)
    )
    numpy.testing.assert_allclose(
        flat,
        ref,
        rtol=1e-13,
        atol=1e-13 * max(numpy.max(numpy.abs(ref)), 1.0),
        err_msg=f"{name} backend value does not match numpy ({shape})",
    )


@pytest.mark.parametrize("extra_rot", [True, False])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_galcen_rot_array_Xsun_and_extra_rot(backend_name, extra_rot):
    # _galcen_rot has two shapes: a (3, 3) rotation for scalar Xsun/Zsun and a
    # (3, 3, N) one when they are arrays, and _apply_galcen_rot contracts them
    # differently (matmul vs a per-point sum). The scalar/default case is
    # covered above; this exercises the batched branch and _extra_rot=False,
    # which the streamTrack path never reaches.
    rng = numpy.random.default_rng(23)
    n = 6
    x, y, z = rng.normal(size=n) + 1.5, rng.normal(size=n), 0.3 * rng.normal(size=n)
    vx, vy, vz = rng.normal(size=n), rng.normal(size=n) + 1.0, 0.2 * rng.normal(size=n)
    Xarr, Zarr = 1.0 + 0.1 * rng.random(n), 0.02 * rng.random(n)
    for Xs, Zs, tag in ((1.0, 0.02, "scalar"), (Xarr, Zarr, "array")):
        ref_x = numpy.asarray(
            coords.galcenrect_to_XYZ(x, y, z, Xsun=Xs, Zsun=Zs, _extra_rot=extra_rot)
        )
        ref_v = numpy.asarray(
            coords.galcenrect_to_vxvyvz(
                vx, vy, vz, Xsun=Xs, Zsun=Zs, _extra_rot=extra_rot
            )
        )
        bx, by, bz = (_as_backend(backend_name, a) for a in (x, y, z))
        bvx, bvy, bvz = (_as_backend(backend_name, a) for a in (vx, vy, vz))
        got_x = coords.galcenrect_to_XYZ(
            bx, by, bz, Xsun=Xs, Zsun=Zs, _extra_rot=extra_rot
        )
        got_v = coords.galcenrect_to_vxvyvz(
            bvx, bvy, bvz, Xsun=Xs, Zsun=Zs, _extra_rot=extra_rot
        )
        assert _all_backend(got_x), (
            f"XYZ dropped to numpy ({tag}, extra_rot={extra_rot})"
        )
        assert _all_backend(got_v), f"v dropped to numpy ({tag}, extra_rot={extra_rot})"
        for got, ref, what in ((got_x, ref_x, "XYZ"), (got_v, ref_v, "vxvyvz")):
            numpy.testing.assert_allclose(
                as_numpy(got),
                ref,
                rtol=1e-13,
                atol=1e-13 * max(numpy.max(numpy.abs(ref)), 1.0),
                err_msg=f"{what} mismatch ({tag}, _extra_rot={extra_rot})",
            )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_vxvyvz_to_vrpmllpmbb_XYZ_degree_backend(backend_name):
    # XYZ=True + degree=True is the one combination with its own backend branch:
    # degreeDecorator has already converted args 3/4 deg->rad, which is wrong when
    # they are X/Y rather than l/b, so the body undoes it. numpy undoes it in
    # place (`l *= ...`); backend arrays are immutable, so there is a separate
    # out-of-place branch. Nothing else in the suite passes both flags with
    # backend arrays, which left those two lines the only uncovered ones in
    # coords.py.
    X = numpy.array([1.2, -0.7, 0.3])
    Y = numpy.array([-0.4, 0.9, 1.1])
    Z = numpy.array([0.25, -0.6, 0.8])
    vx = numpy.array([10.0, -20.0, 5.0])
    vy = numpy.array([-30.0, 15.0, 25.0])
    vz = numpy.array([7.0, -3.0, 12.0])
    ref = coords.vxvyvz_to_vrpmllpmbb(vx, vy, vz, X, Y, Z, XYZ=True, degree=True)
    got = coords.vxvyvz_to_vrpmllpmbb(
        *[_as_backend(backend_name, a) for a in (vx, vy, vz, X, Y, Z)],
        XYZ=True,
        degree=True,
    )
    assert _all_backend(got), "XYZ+degree backend path fell back to numpy"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13, atol=1e-15)
