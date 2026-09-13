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
    # The FORWARD galcen transforms. Their backend branches are only reachable
    # through a test like this one: the coverage-measuring CI job runs numpy
    # only, and the --backend shards upload no coverage at all, so a branch
    # exercised solely under --backend reads as uncovered. Covers
    # _to_galcen_rot and both _as_vsun branches: the default vsun is a list,
    # the "vsun_array" case passes it pre-built.
    "XYZ_to_galcenrect": lambda c, a: c.XYZ_to_galcenrect(
        a["x"], a["y"], a["z"], Xsun=1.0, Zsun=0.02
    ),
    "XYZ_to_galcencyl": lambda c, a: c.XYZ_to_galcencyl(
        a["x"], a["y"], a["z"], Xsun=1.0, Zsun=0.02
    ),
    "vxvyvz_to_galcenrect": lambda c, a: c.vxvyvz_to_galcenrect(
        a["vx"], a["vy"], a["vz"], Xsun=1.0, Zsun=0.02
    ),
    "vxvyvz_to_galcenrect_vsun_array": lambda c, a: c.vxvyvz_to_galcenrect(
        a["vx"],
        a["vy"],
        a["vz"],
        Xsun=1.0,
        Zsun=0.02,
        vsun=numpy.array([-10.0, 240.0, 7.0]),
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


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_galcenrect_transforms_differentiate_through_Xsun_Zsun(backend_name):
    # Xsun/Zsun were assumed to be "configuration, never traced data", so
    # _galcen_rot built the rotation with hard-coded numpy. But the solar
    # position is a FIT PARAMETER, so a caller may hand in a backend array and
    # differentiate through it -- which raised
    #     RuntimeError: Can't call numpy() on Tensor that requires grad
    # on torch, and a tracer conversion error under jax.jit. The JACOBIAN of
    # the same transform (galcenrect_to_XYZ_jac) already handled backend
    # Xsun/Zsun, so this was an asymmetry within one transform pair.
    #
    # Bar is grad-vs-FD, not merely "it returns something": the whole point of
    # a backend Xsun is the derivative w.r.t. it.
    X0, Y0, Z0, Xs0, Zs0 = 1.0, 2.0, 3.0, 8.0, 0.02

    def total(fn, xs, zs):
        return sum(float(v) for v in fn(X0, Y0, Z0, Xsun=xs, Zsun=zs))

    for fn in (coords.galcenrect_to_XYZ, coords.galcenrect_to_vxvyvz):
        h = 1e-6
        fd_X = (total(fn, Xs0 + h, Zs0) - total(fn, Xs0 - h, Zs0)) / (2 * h)
        fd_Z = (total(fn, Xs0, Zs0 + h) - total(fn, Xs0, Zs0 - h)) / (2 * h)
        if backend_name == "jax":
            import jax

            def scalar(p):
                out = fn(
                    jnp.asarray(X0),
                    jnp.asarray(Y0),
                    jnp.asarray(Z0),
                    Xsun=p[0],
                    Zsun=p[1],
                )
                return out[0] + out[1] + out[2]

            ad = jax.grad(scalar)(jnp.asarray([Xs0, Zs0]))
            ad_X, ad_Z = float(ad[0]), float(ad[1])
        else:
            xs = torch.tensor(Xs0, dtype=torch.float64, requires_grad=True)
            zs = torch.tensor(Zs0, dtype=torch.float64, requires_grad=True)
            out = fn(
                torch.tensor(X0, dtype=torch.float64),
                torch.tensor(Y0, dtype=torch.float64),
                torch.tensor(Z0, dtype=torch.float64),
                Xsun=xs,
                Zsun=zs,
            )
            (out[0] + out[1] + out[2]).backward()
            ad_X, ad_Z = float(xs.grad), float(zs.grad)
        # FD with h=1e-6 is good to ~1e-9 here; anything looser would pass on a
        # wrong-but-plausible gradient.
        numpy.testing.assert_allclose(ad_X, fd_X, rtol=1e-7, atol=1e-9)
        numpy.testing.assert_allclose(ad_Z, fd_Z, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_galcenrect_backend_Xsun_with_numpy_coordinates(backend_name):
    # The namespace probe used to read only the COORDINATES, so a backend
    # Xsun alongside numpy X/Y/Z silently took the numpy path; and probing the
    # mix naively raises "Multiple namespaces for array inputs". The rule --
    # backend args decide, numpy ones are weak and coerce across -- is
    # galpy.backend.prefer_backend_namespace, shared with the @backend_input
    # boundary rather than re-spelled here.
    x, y, z = 1.0, 2.0, 3.0
    Xs = _as_backend(backend_name, 8.0)
    ref = coords.galcenrect_to_XYZ(x, y, z, Xsun=8.0, Zsun=0.02)
    got = coords.galcenrect_to_XYZ(x, y, z, Xsun=Xs, Zsun=0.02)
    assert is_backend_array(got[0]), "backend Xsun must select the backend path"
    for g, r in zip(got, ref):
        numpy.testing.assert_allclose(float(as_numpy(g)), float(r), rtol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_galcenrect_integer_Xsun_does_not_truncate_Zsun(backend_name):
    # SILENT WRONG ANSWER, not a crash. promote_scalars anchors every value on
    # the dtype of the first BACKEND array, so an INTEGER Xsun (an ordinary
    # thing to pass) dragged a python-float Zsun down to int64: 0.02 -> 0. The
    # transform then returned the Zsun = 0 answer with no error at all --
    # ~7.5e-3 off in X, which is far too small to look obviously broken and far
    # too large to be roundoff.
    #
    # Pin BOTH directions: an integer Xsun must equal the float answer, and it
    # must NOT equal the Zsun = 0 answer (which is what a re-truncation would
    # silently give back).
    x, y, z, Zs = 1.0, 2.0, 3.0, 0.02
    ref = coords.galcenrect_to_XYZ(x, y, z, Xsun=8.0, Zsun=Zs)
    flat = coords.galcenrect_to_XYZ(x, y, z, Xsun=8.0, Zsun=0.0)
    Xi = _as_backend(backend_name, 8)  # integer dtype on purpose
    got = coords.galcenrect_to_XYZ(x, y, z, Xsun=Xi, Zsun=Zs)
    for g, r in zip(got, ref):
        numpy.testing.assert_allclose(float(as_numpy(g)), float(r), rtol=1e-12)
    assert not numpy.allclose(
        [float(as_numpy(g)) for g in got], [float(v) for v in flat], rtol=1e-9
    ), "integer Xsun silently reproduced the Zsun=0 answer -- Zsun was truncated"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_lambdanu_to_Rz_is_traceable_and_matches_numpy(backend_name):
    # lambdanu_to_Rz clamped its roundoff-negative roots with a python `if` on
    # the data plus an in-place `r2[index] = 0.0` -- untraceable AND unwritable
    # on jax. Exactly the if -> xp.where array-input class. Its FORWARD
    # direction (Rz_to_lambdanu) was migrated long ago; this is the inverse
    # catching up.
    rng = numpy.random.default_rng(3)
    R = 0.2 + 2.5 * rng.random(7)
    z = rng.random(7) - 0.5
    lam, nu = coords.Rz_to_lambdanu(R, z, ac=5.0, Delta=1.0)
    ref_R, ref_z = coords.lambdanu_to_Rz(lam, nu, ac=5.0, Delta=1.0)

    bl, bn = _as_backend(backend_name, lam), _as_backend(backend_name, nu)
    got_R, got_z = coords.lambdanu_to_Rz(bl, bn, ac=5.0, Delta=1.0)
    assert is_backend_array(got_R), "backend input must not fall through to numpy"
    numpy.testing.assert_allclose(as_numpy(got_R), ref_R, rtol=1e-14)
    numpy.testing.assert_allclose(as_numpy(got_z), ref_z, rtol=1e-14)

    if backend_name == "jax":
        import jax

        # The clamp used to raise TracerBoolConversionError here. Grad-vs-FD
        # rather than "it traced": a where() with the mask inverted would still
        # trace happily and give the wrong derivative.
        def total(lv):
            return coords.lambdanu_to_Rz(lv, jnp.asarray(nu), ac=5.0, Delta=1.0)[
                0
            ].sum()

        ad = numpy.asarray(jax.jit(jax.grad(total))(jnp.asarray(lam)))
        h = 1e-6
        fd = numpy.empty_like(lam)
        for i in range(lam.size):
            up, dn = lam.copy(), lam.copy()
            up[i] += h
            dn[i] -= h
            fd[i] = (
                coords.lambdanu_to_Rz(up, nu, ac=5.0, Delta=1.0)[0].sum()
                - coords.lambdanu_to_Rz(dn, nu, ac=5.0, Delta=1.0)[0].sum()
            ) / (2 * h)
        numpy.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("degree", [True, False])
def test_lbd_to_XYZ_preserves_the_backend(backend_name, degree):
    # lbd_to_XYZ carried @scalarDecorator WITHOUT @backendNative, so the
    # decorator promoted every input through numpy before the body ran --
    # stripping the framework off a backend array. The body was raw numpy too.
    # Both had to move: marking the function alone leaves numpy.cos in the body,
    # and migrating the body alone leaves the decorator stripping the input.
    #
    # @backendNative is INNERMOST by construction (scalarDecorator reads the
    # attribute off the function it wraps); anywhere else it is silently dead.
    rng = numpy.random.default_rng(5)
    n = 6
    lo = rng.random(n) * (360.0 if degree else 6.2)
    ba = (rng.random(n) - 0.5) * (180.0 if degree else 3.1)
    da = rng.random(n) * 5 + 0.1
    ref = coords.lbd_to_XYZ(lo, ba, da, degree=degree)

    got = coords.lbd_to_XYZ(
        _as_backend(backend_name, lo),
        _as_backend(backend_name, ba),
        _as_backend(backend_name, da),
        degree=degree,
    )
    assert is_backend_array(got), "decorator stripped the backend off the input"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("XYZ", [False, True])
def test_vrpmllpmbb_to_vxvyvz_preserves_the_backend(backend_name, XYZ):
    # Same @scalarDecorator/@backendNative pair as lbd_to_XYZ, but the body had
    # two more blockers: a preallocated numpy.zeros((3,3,N)) filled by nine
    # INDEXED WRITES (jax arrays are immutable), and `l *= 180/pi` on the
    # XYZ=True path (in-place on an input). Both are out-of-place now.
    #
    # XYZ=True is parametrised precisely because it is the branch with the
    # in-place mutation -- XYZ=False never reaches it.
    rng = numpy.random.default_rng(19)
    n = 5
    vr, pmll, pmbb = rng.normal(size=n) * 30, rng.normal(size=n), rng.normal(size=n)
    if XYZ:
        lo, ba, da = rng.normal(size=n) + 2, rng.normal(size=n), rng.normal(size=n) + 5
    else:
        lo, ba, da = (
            rng.random(n) * 360,
            (rng.random(n) - 0.5) * 180,
            rng.random(n) * 5 + 0.5,
        )
    ref = coords.vrpmllpmbb_to_vxvyvz(vr, pmll, pmbb, lo, ba, da, XYZ=XYZ, degree=True)

    args = [_as_backend(backend_name, a) for a in (vr, pmll, pmbb, lo, ba, da)]
    got = coords.vrpmllpmbb_to_vxvyvz(*args, XYZ=XYZ, degree=True)
    assert is_backend_array(got), "decorator stripped the backend off the input"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-13)


def test_python_scalar_is_not_silently_single_precision():
    # REGRESSION, torch only. scalarDecorator's @backendNative path tested ndim
    # with a bare xp.asarray(arg), and torch.asarray(<python float>) follows
    # torch's DEFAULT dtype -- float32 for anyone who has not changed it. So a
    # call with plain Python scalars computed the whole transform in single
    # precision (~1e-7 relative) while the same call with numpy scalars or
    # arrays stayed float64.
    #
    # conftest sets torch.set_default_dtype(float64) for the whole suite, which
    # is why NOTHING here could see this: the harness silently supplies the
    # very thing real users lack. This test therefore restores torch's own
    # default for the duration, which is the configuration a user actually has.
    #
    # Pinned on the SPEED INVARIANT, exact independently of any reference: the
    # (vr, d*pmll*_K, d*pmbb*_K) triad is orthonormal, so |v| is preserved.
    torch = pytest.importorskip("torch")
    lo, ba = coords.radec_to_lb(20.0, 30.0, degree=True)
    vr, pmll, pmbb, dist = 30.0, -3.0, 5.0, 2.0
    want = vr**2 + (dist * pmll * coords._K) ** 2 + (dist * pmbb * coords._K) ** 2

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # what a user actually has
    try:
        with use("torch", force=True):
            got = coords.vrpmllpmbb_to_vxvyvz(
                vr, pmll, pmbb, float(lo), float(ba), dist, degree=True
            )
            # scalarDecorator unpacks a scalar call into a tuple of components
            speed2 = float(sum(float(as_numpy(c)) ** 2 for c in got))
    finally:
        torch.set_default_dtype(prev)
    assert abs(speed2 - want) / want < 1e-13, (
        f"python-scalar input fell to single precision: |v|^2 off by "
        f"{abs(speed2 - want) / want:.2e} (float64 sits at ~2e-16)"
    )
