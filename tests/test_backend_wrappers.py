###############################################################################
# test_backend_wrappers.py: backend-agnostic tests for the wrapper-potential
# family (P2.6): DehnenSmooth / GaussianAmplitude / TimeDependentAmplitude /
# SolidBodyRotation / CorotatingRotation / RotateAndTilt / KuzminLike /
# OblateStaeckel / CylindricallySeparable wrappers (wrapping already-migrated
# children) and AdiabaticContraction (which is an interpSphericalPotential at
# runtime, so it inherits the P2.5 backend-evaluable spline machinery without
# any wrapper-specific code).
#
# For every wrapper this proves:
#   1. numpy / jax / torch value parity (rtol=1e-12, atol=1e-14) for all
#      migrated methods over a grid of scalar (R, z, phi, t) points (the
#      wrappers are scalar-evaluation classes via their children / decorators);
#   2. jax/torch autodiff of _evaluate matches central finite differences and
#      the analytic identities AD(_evaluate) == -_Rforce / -_zforce;
#   3. jax jit + vmap survival of _Rforce;
#   4. torch.autograd.gradcheck of _evaluate;
#   5. the wrapper layer preserves the torch dtype its wrapped child produces
#      (float32 stays float32 unless the child itself promotes);
#   6. the time-modulation wrappers are differentiable wrt a *traced* t (the
#      in-backend diffrax/torchdiffeq integrator case), exercising the
#      branch-free xp.where smoothing path;
#   7. wrapper-of-wrapper chains (DehnenSmooth(GaussianAmplitude(Plummer)))
#      and planar (2D) wrappers evaluate identically across backends;
#   8. the RotateAndTilt R == inf branch agrees across backends.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager

from galpy.potential import (
    AdiabaticContractionWrapperPotential,
    CorotatingRotationWrapperPotential,
    CylindricallySeparablePotentialWrapper,
    DehnenSmoothWrapperPotential,
    GaussianAmplitudeWrapperPotential,
    HernquistPotential,
    KeplerPotential,
    KuzminLikeWrapperPotential,
    MiyamotoNagaiPotential,
    OblateStaeckelWrapperPotential,
    PlummerPotential,
    RotateAndTiltWrapperPotential,
    SolidBodyRotationWrapperPotential,
    SpiralArmsPotential,
    TimeDependentAmplitudeWrapperPotential,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

# Discover available backends
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

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

# --- wrapped (already-migrated) children --------------------------------------
_PP = PlummerPotential(amp=1.3, b=0.8)
_MN = MiyamotoNagaiPotential(amp=1.1, a=0.7, b=0.3)
_SP = SpiralArmsPotential(amp=1.0, N=2, alpha=0.2)
_KP = KeplerPotential(amp=1.2)

_AXI_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
]
# SpiralArms-wrapped wrappers also exercise the phi-direction methods; their
# _dens is omitted because the bare SpiralArms child itself only matches at
# ~1e-10 under torch (pre-existing child-level difference, not the wrapper).
_NONAXI_METHODS = _AXI_METHODS + ["_phitorque", "_phi2deriv", "_Rphideriv"]

# Each entry: (id, wrapper, [methods]).
_WRAPPERS = [
    (
        "DehnenSmooth-grow",
        DehnenSmoothWrapperPotential(pot=_PP, tform=-1.0, tsteady=2.0),
        _AXI_METHODS + ["_dens"],
    ),
    (
        "DehnenSmooth-decay",
        DehnenSmoothWrapperPotential(pot=_MN, tform=-1.0, tsteady=2.0, decay=True),
        _AXI_METHODS + ["_dens"],
    ),
    (
        "GaussianAmplitude",
        GaussianAmplitudeWrapperPotential(pot=_MN, to=0.5, sigma=1.3),
        _AXI_METHODS + ["_dens"],
    ),
    (
        "TimeDependentAmplitude",
        TimeDependentAmplitudeWrapperPotential(
            pot=_PP, A=lambda t: 1.0 + 0.1 * t + 0.02 * t**2
        ),
        _AXI_METHODS + ["_dens"],
    ),
    (
        "SolidBodyRotation",
        SolidBodyRotationWrapperPotential(pot=_SP, omega=1.1, pa=0.3),
        _NONAXI_METHODS,
    ),
    (
        "CorotatingRotation",
        CorotatingRotationWrapperPotential(pot=_SP, vpo=1.1, beta=0.2, to=0.1, pa=0.2),
        _NONAXI_METHODS,
    ),
    (
        "RotateAndTilt-norot",
        RotateAndTiltWrapperPotential(pot=_MN),
        _NONAXI_METHODS + ["_phizderiv", "_dens"],
    ),
    (
        "RotateAndTilt-zvecpa",
        RotateAndTiltWrapperPotential(pot=_MN, zvec=[0.1, 0.2, 0.97], galaxy_pa=0.3),
        _NONAXI_METHODS + ["_phizderiv", "_dens"],
    ),
    (
        "RotateAndTilt-offset",
        RotateAndTiltWrapperPotential(
            pot=_MN, zvec=[0.1, 0.2, 0.97], galaxy_pa=0.3, offset=[0.1, -0.05, 0.02]
        ),
        _NONAXI_METHODS + ["_phizderiv", "_dens"],
    ),
    (
        "RotateAndTilt-inclination",
        RotateAndTiltWrapperPotential(
            pot=_MN, inclination=0.4, galaxy_pa=0.3, sky_pa=0.2
        ),
        _NONAXI_METHODS + ["_phizderiv", "_dens"],
    ),
    (
        "KuzminLike",
        KuzminLikeWrapperPotential(pot=_KP, a=1.1, b=0.3),
        _AXI_METHODS + ["_phitorque"],
    ),
    (
        "OblateStaeckel",
        OblateStaeckelWrapperPotential(pot=_MN, delta=0.5, u0=1.2),
        _AXI_METHODS,
    ),
    (
        "CylindricallySeparable",
        CylindricallySeparablePotentialWrapper(pot=_MN, Rp=1.1),
        _AXI_METHODS,
    ),
    (
        "AdiabaticContraction",
        AdiabaticContractionWrapperPotential(
            pot=HernquistPotential(amp=1.4, a=1.1),
            baryonpot=MiyamotoNagaiPotential(amp=0.4, a=0.7, b=0.1),
        ),
        ["_evaluate", "_Rforce", "_zforce"],
    ),
]
_WRAPPER_IDS = [w[0] for w in _WRAPPERS]

# Scalar evaluation points: z = 0 hits the OblateStaeckel v = pi/2 line and the
# RotateAndTilt plane; t straddles the DehnenSmooth tform/tsteady window.
_POINTS = [
    (0.6, -0.3, 0.0, -2.0),
    (1.1, 0.0, 0.9, 0.15),
    (1.7, 0.23, 2.7, 3.0),
]


def _toscalar(backend_name, x):
    if backend_name == "numpy":
        return x
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


def _iter_parity_cases():
    for wid, w, methods in _WRAPPERS:
        for m in methods:
            yield pytest.param(w, m, id=f"{wid}-{m}")


# --- value parity --------------------------------------------------------------
@pytest.mark.parametrize("pot,method", list(_iter_parity_cases()))
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot, method):
    for R, z, phi, t in _POINTS:
        ref = float(getattr(pot, method)(R, z, phi=phi, t=t))
        got = float(
            getattr(pot, method)(
                _toscalar(backend_name, R),
                _toscalar(backend_name, z),
                phi=_toscalar(backend_name, phi),
                t=t,
            )
        )
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- autodiff vs finite differences + force identities ---------------------------
@pytest.mark.parametrize("pot", [w[1] for w in _WRAPPERS], ids=_WRAPPER_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_fd_and_forces(backend_name, pot):
    R0, z0, phi0, t0 = 1.1, 0.23, 0.4, 0.15
    eps = 1e-6

    def phi_np(R, z):
        return float(pot._evaluate(R, z, phi=phi0, t=t0))

    fdR = (phi_np(R0 + eps, z0) - phi_np(R0 - eps, z0)) / (2 * eps)
    fdz = (phi_np(R0, z0 + eps) - phi_np(R0, z0 - eps)) / (2 * eps)
    refFR = -float(pot._Rforce(R0, z0, phi=phi0, t=t0))
    refFz = -float(pot._zforce(R0, z0, phi=phi0, t=t0))
    if backend_name == "jax":
        adR = float(
            jax.grad(
                lambda R: pot._evaluate(R, jnp.asarray(z0), phi=jnp.asarray(phi0), t=t0)
            )(jnp.asarray(R0))
        )
        adz = float(
            jax.grad(
                lambda z: pot._evaluate(jnp.asarray(R0), z, phi=jnp.asarray(phi0), t=t0)
            )(jnp.asarray(z0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(
            R,
            torch.tensor(z0, dtype=torch.float64),
            phi=torch.tensor(phi0, dtype=torch.float64),
            t=t0,
        ).backward()
        adR = float(R.grad)
        z = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(
            torch.tensor(R0, dtype=torch.float64),
            z,
            phi=torch.tensor(phi0, dtype=torch.float64),
            t=t0,
        ).backward()
        adz = float(z.grad)
    numpy.testing.assert_allclose(adR, fdR, rtol=1e-5)
    numpy.testing.assert_allclose(adz, fdz, rtol=1e-5)
    # Analytic identities (OblateStaeckel's hand-coded forces carry a 1e-12
    # softening, so the identity holds to ~1e-12 rather than machine precision)
    numpy.testing.assert_allclose(adR, refFR, rtol=1e-9)
    numpy.testing.assert_allclose(adz, refFz, rtol=1e-9)


# --- jax jit + vmap survival -----------------------------------------------------
@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("pot", [w[1] for w in _WRAPPERS], ids=_WRAPPER_IDS)
def test_jax_jit_and_vmap(pot):
    Rs = numpy.array([0.8, 1.1, 1.5])
    zs = numpy.array([0.1, 0.23, -0.2])
    phis = numpy.array([0.4, 1.0, 2.2])
    ref = numpy.array(
        [float(pot._Rforce(R, z, phi=phi, t=0.15)) for R, z, phi in zip(Rs, zs, phis)]
    )
    f = lambda R, z, phi: pot._Rforce(R, z, phi=phi, t=0.15)
    jitted = numpy.array(
        [
            float(jax.jit(f)(jnp.asarray(R), jnp.asarray(z), jnp.asarray(phi)))
            for R, z, phi in zip(Rs, zs, phis)
        ]
    )
    numpy.testing.assert_allclose(jitted, ref, rtol=1e-12, atol=1e-14)
    vmapped = numpy.asarray(
        jax.vmap(f)(jnp.asarray(Rs), jnp.asarray(zs), jnp.asarray(phis))
    )
    numpy.testing.assert_allclose(vmapped, ref, rtol=1e-12, atol=1e-14)


# --- torch gradcheck ------------------------------------------------------------
@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
@pytest.mark.parametrize("pot", [w[1] for w in _WRAPPERS], ids=_WRAPPER_IDS)
def test_torch_gradcheck(pot):
    z0 = torch.tensor(0.23, dtype=torch.float64)
    phi0 = torch.tensor(0.4, dtype=torch.float64)

    def f(R):
        return pot._evaluate(R, z0, phi=phi0, t=0.15)

    R = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (R,), eps=1e-6, atol=1e-8, rtol=1e-6)


# --- torch dtype preservation -----------------------------------------------------
# The wrapper layer must not promote: its output dtype equals whatever its
# wrapped child produces for the same inputs (SpiralArms itself promotes
# float32 -> float64 -- a pre-existing child-level behavior, hence the
# child-relative assertion). AdiabaticContraction has no wrapped child at
# runtime (it *is* an interpSphericalPotential; see
# test_backend_interpspherical.py for that machinery's tests).
@pytest.mark.skipif("torch" not in BACKENDS, reason="torch not installed")
@pytest.mark.parametrize(
    "pot",
    [w[1] for w in _WRAPPERS if w[0] != "AdiabaticContraction"],
    ids=[w[0] for w in _WRAPPERS if w[0] != "AdiabaticContraction"],
)
def test_torch_dtype_preserved(pot):
    from galpy.potential.Potential import _evaluateRforces

    for dt in (torch.float64, torch.float32):
        args = (
            torch.tensor(1.1, dtype=dt),
            torch.tensor(0.23, dtype=dt),
        )
        kwargs = dict(phi=torch.tensor(0.4, dtype=dt), t=0.15)
        child_dtype = _evaluateRforces(pot._pot, *args, **kwargs).dtype
        assert pot._Rforce(*args, **kwargs).dtype == child_dtype


# --- traced t: the branch-free xp.where smoothing paths ----------------------------
_TDEP_WRAPPERS = [
    ("DehnenSmooth", _WRAPPERS[0][1]),
    ("GaussianAmplitude", _WRAPPERS[2][1]),
    ("TimeDependentAmplitude", _WRAPPERS[3][1]),
    ("SolidBodyRotation", _WRAPPERS[4][1]),
    ("CorotatingRotation", _WRAPPERS[5][1]),
]


@pytest.mark.parametrize(
    "pot", [w[1] for w in _TDEP_WRAPPERS], ids=[w[0] for w in _TDEP_WRAPPERS]
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_wrt_time(backend_name, pot):
    R0, z0, phi0, t0 = 1.1, 0.23, 0.4, 0.15
    eps = 1e-6
    fd = (
        float(pot._evaluate(R0, z0, phi=phi0, t=t0 + eps))
        - float(pot._evaluate(R0, z0, phi=phi0, t=t0 - eps))
    ) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(
                lambda t: pot._evaluate(
                    jnp.asarray(R0), jnp.asarray(z0), phi=jnp.asarray(phi0), t=t
                )
            )(jnp.asarray(t0))
        )
    else:
        t = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(
            torch.tensor(R0, dtype=torch.float64),
            torch.tensor(z0, dtype=torch.float64),
            phi=torch.tensor(phi0, dtype=torch.float64),
            t=t,
        ).backward()
        ad = float(t.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-12)


# Backend-array t value parity across the smoothing window (covers the
# xp.where branches at t < tform, tform <= t <= tsteady, and t > tsteady).
@pytest.mark.parametrize("t0", [-2.0, 0.15, 3.0])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_dehnensmooth_backend_t_window_parity(backend_name, t0):
    pot = _WRAPPERS[0][1]  # DehnenSmooth-grow, tform=-1, tsteady at 1
    ref = float(pot._evaluate(1.1, 0.23, phi=0.4, t=t0))
    got = float(
        pot._evaluate(
            _toscalar(backend_name, 1.1),
            _toscalar(backend_name, 0.23),
            phi=_toscalar(backend_name, 0.4),
            t=_toscalar(backend_name, t0),
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-15, atol=1e-15)


# Backend-array t value parity for GaussianAmplitude: a non-numpy, non-Python t
# takes the xp.exp branch of _smooth (the math.exp fast path handles only a
# concrete Python/numpy scalar t). Mirrors the DehnenSmooth window test above.
@pytest.mark.parametrize("t0", [-2.0, 0.15, 3.0])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_gaussianamplitude_backend_t_parity(backend_name, t0):
    pot = _WRAPPERS[2][1]  # GaussianAmplitude, to=0.5, sigma=1.3
    ref = float(pot._evaluate(1.1, 0.23, phi=0.4, t=t0))
    got = float(
        pot._evaluate(
            _toscalar(backend_name, 1.1),
            _toscalar(backend_name, 0.23),
            phi=_toscalar(backend_name, 0.4),
            t=_toscalar(backend_name, t0),
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-15, atol=1e-15)


# --- wrapper-of-wrapper chain -------------------------------------------------------
_CHAIN = DehnenSmoothWrapperPotential(
    pot=GaussianAmplitudeWrapperPotential(pot=_PP, to=0.5, sigma=1.3),
    tform=-1.0,
    tsteady=2.0,
)


@pytest.mark.parametrize("method", ["_evaluate", "_Rforce", "_zforce", "_R2deriv"])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_chain_value_parity(backend_name, method):
    for R, z, phi, t in _POINTS:
        ref = float(getattr(_CHAIN, method)(R, z, phi=phi, t=t))
        got = float(
            getattr(_CHAIN, method)(
                _toscalar(backend_name, R),
                _toscalar(backend_name, z),
                phi=_toscalar(backend_name, phi),
                t=t,
            )
        )
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_chain_jax_grad_and_jit():
    R0, z0 = 1.1, 0.23
    refF = -float(_CHAIN._Rforce(R0, z0, phi=0.0, t=0.15))
    g = float(
        jax.grad(lambda R: _CHAIN._evaluate(R, jnp.asarray(z0), phi=0.0, t=0.15))(
            jnp.asarray(R0)
        )
    )
    numpy.testing.assert_allclose(g, refF, rtol=1e-9)
    assert_jit_matches_eager(
        lambda R, z: _CHAIN._Rforce(R, z, phi=0.0, t=0.15),
        jnp.asarray(R0),
        jnp.asarray(z0),
        rtol=1e-12,
        atol=0.0,
        ref=-refF,
    )


# --- planar (2D) wrappers -------------------------------------------------------------
_PLANAR_WRAPPERS = [
    (
        "planar-DehnenSmooth",
        DehnenSmoothWrapperPotential(pot=_PP.toPlanar(), tform=-1.0, tsteady=2.0),
        ["_evaluate", "_Rforce", "_R2deriv"],
    ),
    (
        "planar-SolidBody",
        SolidBodyRotationWrapperPotential(pot=_SP.toPlanar(), omega=1.1, pa=0.3),
        ["_evaluate", "_Rforce", "_phitorque", "_R2deriv", "_phi2deriv", "_Rphideriv"],
    ),
]


def _iter_planar_cases():
    for wid, w, methods in _PLANAR_WRAPPERS:
        for m in methods:
            yield pytest.param(w, m, id=f"{wid}-{m}")


@pytest.mark.parametrize("pot,method", list(_iter_planar_cases()))
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_planar_value_parity(backend_name, pot, method):
    for R, _, phi, t in _POINTS:
        ref = float(getattr(pot, method)(R, phi=phi, t=t))
        got = float(
            getattr(pot, method)(
                _toscalar(backend_name, R), phi=_toscalar(backend_name, phi), t=t
            )
        )
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# --- coords transforms under a forced backend default --------------------------------
# All-scalar inputs with a forced (non-numpy) default exercise the
# nothing-to-anchor-on path of the coords scalar promotion (jax handles plain
# scalars natively; with x64 enabled the values match numpy's exactly).
@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_coords_forced_backend_scalar_inputs():
    from galpy import backend
    from galpy.util import coords

    ref_uv = coords.uv_to_Rz(1.2, 0.8, delta=0.5)
    ref_cyl = coords.cyl_to_rect(1.1, 0.4, 0.23)
    with backend.use("jax", force=True):
        got_uv = coords.uv_to_Rz(1.2, 0.8, delta=0.5)
        got_cyl = coords.cyl_to_rect(1.1, 0.4, 0.23)
    numpy.testing.assert_allclose(
        [float(g) for g in got_uv], [float(r) for r in ref_uv], rtol=1e-15
    )
    numpy.testing.assert_allclose(
        [float(g) for g in got_cyl], [float(r) for r in ref_cyl], rtol=1e-15
    )


# --- RotateAndTilt R == inf branch ------------------------------------------------------
@pytest.mark.parametrize(
    "pot",
    [w[1] for w in _WRAPPERS if w[0].startswith("RotateAndTilt")],
    ids=[w[0] for w in _WRAPPERS if w[0].startswith("RotateAndTilt")],
)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_rotateandtilt_Rinf(backend_name, pot):
    ref = float(pot._evaluate(numpy.inf, 0.0, phi=0.3, t=0.0))
    got = float(
        pot._evaluate(
            _toscalar(backend_name, numpy.inf),
            _toscalar(backend_name, 0.0),
            phi=_toscalar(backend_name, 0.3),
            t=0.0,
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    # finite-R autodiff is not poisoned by the guarded R == inf branch
    if backend_name == "jax":
        g = float(
            jax.grad(
                lambda R: pot._evaluate(
                    R, jnp.asarray(0.1), phi=jnp.asarray(0.3), t=0.0
                )
            )(jnp.asarray(1.1))
        )
        assert numpy.isfinite(g)


# --- scalar-only methods accept plain python floats under a forced backend ----
# The @check_potential_inputs_not_arrays methods do real backend arithmetic on
# the coordinates (xp.isinf/cos/sin/...); a plain python float reaching them
# under a forced backend used to crash torch ("isinf(): argument 'input' ... must
# be Tensor, not float"). The decorator now coerces R, z, phi onto the active
# backend. Pass plain python floats (NOT pre-made backend arrays) so the decorator
# coercion -- not the caller -- is what brings them onto the backend.
@pytest.mark.parametrize(
    "method", ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_dens"]
)
@pytest.mark.parametrize(
    "pot",
    [w[1] for w in _WRAPPERS if w[0].startswith("RotateAndTilt")],
    ids=[w[0] for w in _WRAPPERS if w[0].startswith("RotateAndTilt")],
)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_scalar_only_python_float_inputs_forced_backend(backend_name, pot, method):
    from galpy import backend

    R, z, phi = 0.75, 0.2, 1.76
    ref = float(getattr(pot, method)(R, z, phi=phi, t=0.0))
    with backend.use(backend_name, force=True):
        got = getattr(pot, method)(R, z, phi=phi, t=0.0)  # floats -> decorator coerces
    numpy.testing.assert_allclose(float(got), ref, rtol=1e-12, atol=1e-14)


# --- vertical-quadrature split point for displaced / tilted wrappers ----------
# The traced surfdens quadratures split the vertical interval at the potential's
# mid-plane and cluster nodes there. Splitting at z=0 is only right when the
# structure sits at z=0; an offset or tilted RotateAndTilt puts it elsewhere,
# and sampling a sharply-peaked density where it is flat costs many orders.
_MWLIKE = MiyamotoNagaiPotential(amp=1.0, a=3.0 / 8.0, b=0.28 / 8.0)
_S3 = numpy.sqrt(1.0 / 3.0)
_SPLIT_CASES = [
    ("offset", dict(zvec=None, galaxy_pa=None, offset=[1.0, 1.0, 1.0])),
    ("tilted", dict(zvec=[_S3, _S3, _S3], galaxy_pa=0.4, offset=None)),
    (
        "tilted+offset",
        dict(zvec=[_S3, _S3, _S3], galaxy_pa=0.4, offset=[1.0, 1.0, 1.0]),
    ),
    ("edge-on", dict(zvec=[0.0, 1.0, 0.0], galaxy_pa=0.3, offset=None)),
    # Degenerate but constructible: identity rotation and no offset, i.e. a
    # wrapper that displaces nothing. Takes the early return, which must hand
    # back a plain 0.0 -- callers use that to skip the xp.where entirely.
    ("no-op", dict(zvec=[0.0, 0.0, 1.0], galaxy_pa=0.0, offset=None)),
]


@pytest.mark.parametrize("case", _SPLIT_CASES, ids=[c[0] for c in _SPLIT_CASES])
def test_vertical_quad_split_is_the_mid_plane_crossing(case):
    """The split is where the WRAPPED potential's own z' vanishes.

    Checked against ``_rect_transformed`` rather than against a formula, so this
    would catch a sign error or a transposed rotation that a self-consistent
    re-derivation would not.
    """
    name, kw = case
    pot = RotateAndTiltWrapperPotential(pot=_MWLIKE, **kw)
    for R, phi in ((0.5, 0.0), (1.3, 0.9), (0.2, 2.7)):
        raw = pot._vertical_quad_split(R, phi=phi)
        c = float(raw)
        zp = float(pot._rect_transformed(numpy, R, c, phi)[2])
        if name == "no-op":
            # Nothing is displaced, so the mid-plane is z = 0 and the method
            # must say so via the early return -- a plain float, not an array
            # from the solve below, which is what lets callers skip the
            # xp.where and keeps this off the traced path entirely.
            assert type(raw) is float and raw == 0.0
        elif kw["zvec"] == [0.0, 1.0, 0.0]:
            # Edge-on: the wrapped plane contains the line of integration, so
            # there is no unique crossing and the split must stay at 0.
            assert c == 0.0
        else:
            assert abs(zp) < 1e-12, f"z' = {zp:e} at the split {c:e}"


def test_vertical_quad_interior_falls_back_to_zero_outside_the_range():
    """The clamp is a fall-back to 0, NOT a clamp to the boundary.

    Both arms of the ``xp.where`` are asserted. Pinning the reject arm is the
    point: clamping a runaway crossing to +/-|z| degenerates one panel to zero
    width and measured 2-6x WORSE than splitting at 0, so "outside the range"
    has to return 0.0 and not ``+/-absz``.
    """
    pot = RotateAndTiltWrapperPotential(
        pot=_MWLIKE, zvec=None, galaxy_pa=None, offset=[1.0, 1.0, 1.0]
    )
    R, phi = 0.5, 0.0
    c = float(pot._vertical_quad_split(R, phi=phi))
    assert c != 0.0, "this case is supposed to displace the mid-plane"

    # Crossing inside [-|z|, |z|]: used as-is, to full precision.
    inside = pot._vertical_quad_interior(numpy, R, 10.0 * abs(c), phi, 0.0)
    assert float(inside) == c

    # Crossing outside: 0.0, and specifically not the nearer boundary.
    outside = pot._vertical_quad_interior(numpy, R, 0.5 * abs(c), phi, 0.0)
    assert float(outside) == 0.0

    # The boundary itself is excluded (a split AT +/-|z| is the zero-width
    # panel the fall-back exists to avoid).
    edge = pot._vertical_quad_interior(numpy, R, abs(c), phi, 0.0)
    assert float(edge) == 0.0

    # An ordinary potential never gets here: it short-circuits to a plain
    # float, which is what keeps its quadrature byte-identical.
    assert type(_MWLIKE._vertical_quad_interior(numpy, R, 1.0, phi, 0.0)) is float


def test_vertical_split_makes_the_density_quadrature_accurate():
    """The split is worth ~8 orders on a displaced disk, and z=0 is not enough.

    Both arms are asserted: the fix must be accurate AND the unfixed rule must
    be visibly wrong, so this cannot pass by the integrand happening to be easy.
    """
    from scipy import integrate

    from galpy.backend import quadrature as bq

    pot = RotateAndTiltWrapperPotential(
        pot=_MWLIKE, zvec=None, galaxy_pa=None, offset=[1.0, 1.0, 1.0]
    )
    R, Z, phi = 0.5, 10.0, 0.0
    f = lambda x: pot._dens(R, x, phi=phi, t=0.0)  # noqa: E731
    fv = lambda s: numpy.array(  # noqa: E731
        [f(v) for v in numpy.atleast_1d(s).ravel()]
    ).reshape(numpy.shape(s))
    ref = integrate.quad(f, -Z, Z, limit=400)[0]

    split = float(pot._vertical_quad_split(R, phi=phi))
    got = bq.symmetric_quad(numpy, fv, Z, n=50, interior_point=split)
    at_zero = bq.symmetric_quad(numpy, fv, Z, n=50, interior_point=0.0)

    assert abs(got - ref) / abs(ref) < 1e-8, "split-at-mid-plane is not accurate"
    assert abs(at_zero - ref) / abs(ref) > 1e-2, "split-at-0 was supposed to be bad"


def test_composite_delegates_both_surfdens_routes():
    """A composite integrates each component separately, on both routes.

    Sigma of a sum is the sum of the Sigmas (and the Poisson integrand is
    additive too), and only the per-component route lets each component split at
    its own mid-plane -- integrating the summed density splits at the
    composite's plane, which is 0 for every composite.

    The structural assertions are the discriminating ones: a value comparison
    alone passes vacuously, because on numpy (and on an untraced backend) both
    routes go through the same adaptive scipy quadrature and agree regardless.
    """
    from galpy.potential import CompositePotential
    from galpy.potential.Potential import Potential as _P

    assert CompositePotential._surfdens is not _P._surfdens
    assert CompositePotential._surfdens_poisson is not _P._surfdens_poisson

    off = RotateAndTiltWrapperPotential(
        pot=_MWLIKE, zvec=None, galaxy_pa=None, offset=[1.0, 1.0, 1.0]
    )
    comp = CompositePotential([off, HernquistPotential(amp=0.4, a=0.6)])
    R, z, phi = 0.5, 2.0, 0.3
    # numpy must keep the inherited single quadrature, so that path stays
    # byte-identical; the delegation is a backend-only route.
    numpy.testing.assert_allclose(
        comp._surfdens(R, z, phi=phi, t=0.0),
        _P._surfdens(comp, R, z, phi=phi, t=0.0),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_composite_surfdens_delegation_runs_on_a_backend(backend_name):
    """Run both delegating branches, and pin what they must return.

    The structural test above proves the overrides exist; nothing was executing
    them, because ``get_namespace`` only resolves away from numpy once the
    inputs are backend arrays. The contract both routes owe is the sum over the
    NON-dissipative components, so a wrong component list, a missing
    ``use_physical=False``, or a dropped ``isDissipative`` filter each move the
    number; the friction term is here to make that last one bite.

    The private methods are called directly with backend scalars rather than
    through ``surfdens``: a composite holding a dissipative force is not
    backend-compatible, so ``@backend_input`` skips the coercion and ``z``
    would arrive as a float while ``get_namespace`` still resolves to the
    forced backend. That mismatch is real but pre-existing (it reproduces on
    develop without these overrides) and belongs to its own fix, not here.
    """
    from galpy import backend
    from galpy.backend import as_numpy
    from galpy.potential import (
        ChandrasekharDynamicalFrictionForce,
        CompositePotential,
    )

    off = RotateAndTiltWrapperPotential(
        pot=_MWLIKE, zvec=None, galaxy_pa=None, offset=[1.0, 1.0, 1.0]
    )
    hern = HernquistPotential(amp=0.4, a=0.6)
    comp = CompositePotential(
        [off, hern, ChandrasekharDynamicalFrictionForce(GMs=0.01, rhm=0.1, dens=hern)]
    )
    R, z, phi = 0.5, 2.0, 0.3
    parts = (off, hern)
    # The density route delegates to the public surfdens, which applies each
    # component's own _amp; the Poisson route delegates to _surfdens_poisson,
    # which already carries it. Mirror both exactly.
    want = sum(
        float(p.surfdens(R, z, phi=phi, t=0.0, use_physical=False)) for p in parts
    )
    want_poisson = sum(float(p._surfdens_poisson(R, z, phi=phi, t=0.0)) for p in parts)
    Rb, zb, phib = (_toscalar(backend_name, v) for v in (R, z, phi))
    with backend.use(backend_name, force=True):
        got = comp._surfdens(Rb, zb, phi=phib, t=0.0)
        got_poisson = comp._surfdens_poisson(Rb, zb, phi=phib, t=0.0)
    numpy.testing.assert_allclose(float(as_numpy(got)), want, rtol=1e-10, atol=0.0)
    numpy.testing.assert_allclose(
        float(as_numpy(got_poisson)), want_poisson, rtol=1e-8, atol=0.0
    )
