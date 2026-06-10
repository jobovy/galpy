###############################################################################
# test_backend_interpspherical.py: multi-backend tests for
# interpSphericalPotential and KingPotential (P2.5).
#
# These potentials evaluate a cubic spline of the radial force (built once at
# __init__ in numpy/scipy; for KingPotential after solving the King ODE). The
# numpy code path keeps calling the scipy splines exactly as before
# (byte-identical); the jax/torch path evaluates the *same* piecewise cubic
# through its exact power-basis (PPoly) coefficients with namespace-agnostic
# searchsorted + Horner ops, so values agree at the ~1 ulp level and the
# potential is exactly autodifferentiable.
#
# For every backend this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14)
#      for the seven 2D methods and the four 1D radial methods, on grids that
#      straddle the r >= rmax Kepler-extrapolation boundary and include an
#      exact spline knot, exactly r == rmax, and a point in the innermost
#      spline interval;
#   2. autodiff of _evaluate matches central finite differences (computed on
#      the numpy/scipy-spline path) both inside the spline domain and in the
#      Kepler-extrapolation region;
#   3. the analytic identities AD(_evaluate)==-_Rforce, AD(_Rforce)==-_R2deriv,
#      etc. (and their 1D radial counterparts) hold to rtol=1e-9, including at
#      a small radius where the dead (r >= rmax) side of the xp.where is the
#      guarded 1/r Kepler branch.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    HernquistPotential,
    KingPotential,
    interpSphericalPotential,
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

# All 2D methods derive from the SphericalPotential 1D interface, whose four
# radial methods are all migrated here.
_THREE_D = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]
_ONE_D = ["_revaluate", "_rforce", "_r2deriv", "_rdens"]

# KingPotential: rgrid starts at exactly 0 and ends at rt=1.4 (the __init__
# King-ODE solve is init-time numpy by design). The generic
# interpSphericalPotential interpolates an analytic parent through the
# galpy-Potential rforce flavor on a nonzero-origin grid ending at 1.5, so the
# shared test grids straddle both Kepler-extrapolation boundaries.
CASES = [
    KingPotential(W0=3.0, M=2.3, rt=1.4),
    interpSphericalPotential(
        rforce=HernquistPotential(amp=2.0, a=1.3),
        rgrid=numpy.geomspace(0.01, 1.5, 151),
    ),
]
CASE_IDS = ["KingPotential", "interpSphericalPotential"]

# 2D grid: r = 0.51, 1.02 (inside the spline domain), 2.02 (Kepler region).
_RS = [0.5, 1.0, 2.0]
_ZS = [0.1, 0.2, 0.3]


def _radial_grid(pot):
    # Innermost spline interval, generic interior point, an exact spline knot,
    # exactly rmax (first point of the Kepler branch), and beyond rmax.
    return [
        0.05,
        0.9,
        float(pot._rgrid[len(pot._rgrid) // 2]),
        float(pot._rmax),
        1.9,
    ]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _THREE_D:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = _tonumpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{type(pot).__name__}.{method}"
        )


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_radial_value_parity(backend_name, pot):
    rs = _radial_grid(pot)
    r = _asarray(backend_name, rs)
    for method in _ONE_D:
        ref = numpy.asarray(getattr(pot, method)(numpy.asarray(rs)))
        got = _tonumpy(getattr(pot, method)(r))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{type(pot).__name__}.{method}"
        )


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot):
    # The public Rforce (through the unit decorators and _amp) must give
    # identical values across backends.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = _tonumpy(pot.Rforce(R, z))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


# One (R,z) point inside the spline domain and one in the Kepler region.
_FD_POINTS = [(1.0, 0.4), (1.8, 0.4)]


@pytest.mark.parametrize("R0,z0", _FD_POINTS, ids=["inside", "kepler"])
@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot, R0, z0):
    # Independent (finite-difference, on the numpy scipy-spline path)
    # cross-check that the migrated _evaluate is differentiable end-to-end.
    eps = 1e-6

    def phi_np(R):
        return float(pot._evaluate(numpy.asarray(R), numpy.asarray(z0)))

    fd = (phi_np(R0 + eps) - phi_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._evaluate(R, torch.tensor(z0, dtype=torch.float64))
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


###############################################################################
# Analytic-identity autodiff checks (same conventions as
# test_backend_spherical.py):
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_Rforce  wrt R) == -_R2deriv      AD(_Rforce  wrt z) == -_Rzderiv
#   AD(_zforce  wrt z) == -_z2deriv
#   AD(_revaluate wrt r) == -_rforce     AD(_rforce wrt r) == -_r2deriv
# These hold exactly here because the backend PPoly antiderivative/derivative
# coefficients are the exact integrals/derivatives of the force pieces.
###############################################################################
_R, _Z = 0, 1
_ID_PAIRS_2D = [
    ("_evaluate", _R, "_Rforce", "R"),
    ("_evaluate", _Z, "_zforce", "z"),
    ("_Rforce", _R, "_R2deriv", "R"),
    ("_Rforce", _Z, "_Rzderiv", "z"),
    ("_zforce", _Z, "_z2deriv", "z"),
]
_ID_PAIRS_1D = [
    ("_revaluate", "_rforce"),
    ("_rforce", "_r2deriv"),
]
# 1D identity radii: innermost spline interval (exercises the guarded dead
# Kepler 1/r branch under AD), generic interior, and the Kepler region.
_ID_RS_1D = [0.05, 0.9, 1.9]


def _grad_wrt(backend_name, fn, *args, argnum=0):
    # AD of scalar-valued fn(*args) wrt args[argnum].
    if backend_name == "jax":
        jargs = [jnp.asarray(a) for a in args]

        def f(x):
            full = list(jargs)
            full[argnum] = x
            return fn(*full)

        return float(jax.grad(f)(jargs[argnum]))
    targs = [torch.tensor(a, dtype=torch.float64) for a in args]
    leaf = torch.tensor(args[argnum], dtype=torch.float64, requires_grad=True)
    targs[argnum] = leaf
    out = fn(*targs)
    out.backward()  # backward() needs a scalar output; all args here are scalars
    return float(leaf.grad)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities(backend_name, pot):
    R0, z0 = 1.0, 0.4
    for lower, argnum, higher, _vn in _ID_PAIRS_2D:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, _l=lower: getattr(pot, _l)(R, z),
            R0,
            z0,
            argnum=argnum,
        )
        # numpy reference through 0-d arrays (the masked-write numpy branch
        # needs array-like input, as everywhere in galpy).
        ref = -float(getattr(pot, higher)(numpy.asarray(R0), numpy.asarray(z0)))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            err_msg=f"{type(pot).__name__}: AD({lower}/{_vn})==-{higher}",
        )
    for r0 in _ID_RS_1D:
        for lower, higher in _ID_PAIRS_1D:
            ad = _grad_wrt(backend_name, lambda r, _l=lower: getattr(pot, _l)(r), r0)
            ref = -float(getattr(pot, higher)(numpy.asarray(r0)))
            numpy.testing.assert_allclose(
                ad,
                ref,
                rtol=1e-9,
                err_msg=f"{type(pot).__name__}: AD({lower})==-{higher} at r={r0}",
            )
