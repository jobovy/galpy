###############################################################################
# test_backend_scf.py: multi-backend tests for SCFPotential (P2.5), migrated
# to the galpy.backend namespace layer (Hernquist-basis expansion via the
# galpy.backend.special Gegenbauer + associated-Legendre router).
#
# For each case (axisymmetric and non-axisymmetric expansions) and each
# migrated compute method this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-12, atol=1e-14),
#      for scalar AND array inputs (the array path stacks per-point results),
#   2. autodiff (jax.grad / torch.autograd) of _evaluate matches central finite
#      differences (independent cross-check of differentiability),
#   3. the analytic force / Hessian identities hold under AD:
#      AD(_evaluate) == -force and AD(force) == -2nd-derivative, exact to
#      ~1e-9 (cross-validates the hand-coded _dphiTilde/_d2phiTilde chain),
#   4. the jax path is jit-compatible (the per-point md5 / float() caches are
#      numpy-only, so tracing never touches them).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
# The grids stay off the z-axis (R > 0): the backend associated-Legendre
# derivative recurrence (like the numpy chain rule it feeds) is singular at
# the poles, exactly as in the numpy implementation's nudged-pole handling.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager

from galpy.backend import as_numpy
from galpy.potential import SCFPotential

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

# All compute methods of SCFPotential are migrated (forces and the analytic
# second derivatives both ride on _phiTilde/_dphiTilde/_d2phiTilde).
METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_phitorque",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
    "_dens",
]


def _make_cases():
    rng = numpy.random.default_rng(42)
    cases = []
    # Hernquist limit: monopole-only (N=1, L=1); exercises the _d2C N==1 guard.
    cases.append(("hernquist", SCFPotential(amp=1.3, Acos=numpy.array([[[1.0]]]))))
    # axisymmetric, multi-(n,l): radial + costheta structure, M == 1.
    Acos_axi = numpy.zeros((5, 4, 1))
    Acos_axi[:, :, 0] = rng.normal(size=(5, 4)) * 0.1
    Acos_axi[0, 0, 0] = 1.0
    cases.append(("axi", SCFPotential(amp=2.6, Acos=Acos_axi, a=1.3)))
    # non-axisymmetric: full (n,l,m) structure incl. Asin, exercises the
    # phi-derivative (phitorque / phi-2nd-derivative) paths.
    Acos_na = numpy.tril(rng.normal(size=(3, 3, 3)) * 0.1)
    Asin_na = numpy.tril(rng.normal(size=(3, 3, 3)) * 0.1)
    Acos_na[0, 0, 0] = 1.0
    cases.append(("nonaxi", SCFPotential(amp=1.9, Acos=Acos_na, Asin=Asin_na, a=0.8)))
    return cases


CASES = _make_cases()
CASE_IDS = [name for name, _ in CASES]

# Evaluation grid: off-centre, off-axis, both signs of z, inside and outside
# the expansion scale radius.
_RS = numpy.array([0.3, 1.0, 2.7])
_ZS = numpy.array([-0.4, 0.2, 1.1])
_PHIS = numpy.array([0.3, 1.1, 2.2])
# Smooth scalar point for the AD checks.
_R0, _Z0, _PHI0 = 1.3, 0.4, 0.7


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_array(backend_name, name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    for mname in METHODS:
        method = getattr(pot, mname)
        ref = numpy.asarray(method(_RS, _ZS, _PHIS))
        got = as_numpy(method(R, z, phi))
        numpy.testing.assert_allclose(
            got,
            ref,
            rtol=1e-12,
            atol=1e-14,
            err_msg=f"SCF[{name}].{mname} array parity ({backend_name})",
        )


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_scalar(backend_name, name, pot):
    for mname in METHODS:
        method = getattr(pot, mname)
        for R0, z0, phi0 in zip(_RS, _ZS, _PHIS):
            ref = numpy.asarray(method(R0, z0, phi0))
            got = as_numpy(
                method(
                    _asarray(backend_name, R0),
                    _asarray(backend_name, z0),
                    _asarray(backend_name, phi0),
                )
            )
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"SCF[{name}].{mname} scalar parity ({backend_name})",
            )


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, name, pot):
    # Through the unit decorators and _amp (public Rforce), values must be
    # identical across backends.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    ref = numpy.asarray(pot.Rforce(_RS, _ZS, phi=_PHIS))
    got = as_numpy(pot.Rforce(R, z, phi=phi))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("var", ["R", "z", "phi"])
def test_grad_evaluate_vs_finite_difference(backend_name, name, pot, var):
    # Independent (finite-difference) cross-check that the migrated _evaluate
    # is differentiable end-to-end in every coordinate.
    eps = 1e-6
    argnum = {"R": 0, "z": 1, "phi": 2}[var]
    x0 = (_R0, _Z0, _PHI0)[argnum]

    def phi_np(x):
        args = [_R0, _Z0, _PHI0]
        args[argnum] = x
        return float(pot._evaluate(*[numpy.asarray(a) for a in args]))

    fd = (phi_np(x0 + eps) - phi_np(x0 - eps)) / (2 * eps)
    ad = _grad_wrt(backend_name, lambda R, z, p: pot._evaluate(R, z, p), argnum=argnum)
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-5, atol=1e-10, err_msg=f"SCF[{name}] d_evaluate/d{var}"
    )


def _grad_wrt(backend_name, fn, argnum=0):
    # AD of scalar-valued fn(R, z, phi) at the smooth point, wrt args[argnum].
    args = (_R0, _Z0, _PHI0)
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
    out.backward()
    return float(leaf.grad)


###############################################################################
# Analytic-identity autodiff checks; galpy's sign conventions give
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_evaluate wrt phi) == -_phitorque
#   AD(_Rforce wrt R) == -_R2deriv       AD(_Rforce wrt z) == -_Rzderiv
#   AD(_zforce wrt z) == -_z2deriv       AD(_phitorque wrt phi) == -_phi2deriv
#   AD(_phitorque wrt R) == -_Rphideriv  AD(_phitorque wrt z) == -_phizderiv
# Exact to ~1e-9, which cross-validates the hand-coded radial-derivative chain
# (_dphiTilde against _phiTilde, _d2phiTilde against _dphiTilde) and the
# spherical-to-cylindrical chain rule, not just the AD plumbing.
###############################################################################
_R, _Z, _PHI = 0, 1, 2
_ID_PAIRS = [
    ("_evaluate", _R, "_Rforce"),
    ("_evaluate", _Z, "_zforce"),
    ("_evaluate", _PHI, "_phitorque"),
    ("_Rforce", _R, "_R2deriv"),
    ("_Rforce", _Z, "_Rzderiv"),
    ("_zforce", _Z, "_z2deriv"),
    ("_phitorque", _PHI, "_phi2deriv"),
    ("_phitorque", _R, "_Rphideriv"),
    ("_phitorque", _Z, "_phizderiv"),
]


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities(backend_name, name, pot):
    for lower, argnum, higher in _ID_PAIRS:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, p, _l=lower: getattr(pot, _l)(R, z, p),
            argnum=argnum,
        )
        ref = -float(getattr(pot, higher)(_R0, _Z0, _PHI0))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"SCF[{name}]: AD({lower}/{argnum}) == -{higher} ({backend_name})",
        )


###############################################################################
# jit-compatibility: the per-point Python caches (md5 force hash, float()-keyed
# 2nd-derivative cache) are numpy-only, so the jax path must trace cleanly
# under jit for forces AND second derivatives.
###############################################################################
@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
def test_jax_jit(name, pot):
    if jax is None:  # pragma: no cover
        pytest.skip("jax not available")
    R = jnp.asarray(_R0)
    z = jnp.asarray(_Z0)
    phi = jnp.asarray(_PHI0)
    for mname in ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_R2deriv"]:
        method = getattr(pot, mname)
        assert_jit_matches_eager(
            method,
            R,
            z,
            phi,
            rtol=1e-12,
            atol=1e-14,
            ref=float(method(_R0, _Z0, _PHI0)),
            err_msg=f"SCF[{name}].{mname} jit",
        )
    # gradient under jit as well
    g = float(jax.jit(jax.grad(lambda R: pot._evaluate(R, z, phi)))(R))
    numpy.testing.assert_allclose(g, -float(pot._Rforce(_R0, _Z0, _PHI0)), rtol=1e-9)


###############################################################################
# r = 0 / r = inf guards: the backend branches handle the expansion centre and
# infinity branchlessly (xp.where with guarded dead branches); values must
# match numpy and reverse-mode gradients of smooth quantities through those
# guards must not be NaN-poisoned at ordinary points (checked above); here we
# check the special points evaluate finitely and identically.
###############################################################################
@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_centre_and_infinity_parity(backend_name, name, pot):
    # potential at the centre is finite (the -CC/a limit of _phiTilde)
    ref0 = float(pot._evaluate(0.0, 0.0, 0.3))
    got0 = float(
        as_numpy(
            pot._evaluate(
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.3),
            )
        )
    )
    assert numpy.isfinite(got0)
    numpy.testing.assert_allclose(got0, ref0, rtol=1e-12, atol=1e-14)
    # potential at infinity -> 0, through the xi = 1 guard of _RToxi
    gotinf = float(
        as_numpy(
            pot._evaluate(
                _asarray(backend_name, numpy.inf),
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.3),
            )
        )
    )
    numpy.testing.assert_allclose(gotinf, 0.0, atol=1e-14)
    # second derivatives at the centre are defined to be 0 (numpy convention)
    refc = float(pot._R2deriv(0.0, 0.0, 0.3))
    gotc = float(
        as_numpy(
            pot._R2deriv(
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.3),
            )
        )
    )
    numpy.testing.assert_allclose(gotc, refc, atol=1e-14)


###############################################################################
# Time-dependent SCF: the (Nt, N, L, M) expansion coefficients are cubic-spline
# interpolated in time. The numpy path evaluates the scipy CubicSpline (and a
# float-keyed cache); the backend (jax/torch) path evaluates the SAME piecewise
# cubic through the active namespace (SCFPotential._coeffs_at_time /
# _interp_ppoly_vec), so a time-dependent SCF
#   1. evaluates identically across backends at scalar AND array times, and
#   2. is DIFFERENTIABLE w.r.t. the evaluation time t (the coefficient
#      time-interpolation flows autodiff) -- the headline new capability --
#      cross-checked against a central finite difference of the numpy path.
###############################################################################
_TDEP_A = 1.7
# a smooth, non-polynomial time dependence: d/dt is non-trivial and the cubic
# spline genuinely interpolates between the (finely spaced) grid nodes.
_TDEP_TGRID = numpy.linspace(0.0, 5.0, 26)
_tdep_scale = lambda t: 1.0 + 0.05 * t + 0.02 * t**2 + 0.1 * numpy.sin(0.7 * t)


def _make_tdep_cases():
    rng = numpy.random.default_rng(7)
    cases = []
    # spherical/axisymmetric time-dependent SCF (monopole-dominated, M == 1)
    Acos_sph = numpy.zeros((5, 1, 1))
    Acos_sph[:, 0, 0] = rng.normal(size=5) * 0.1
    Acos_sph[0, 0, 0] = 1.0
    arr = numpy.array([Acos_sph * _tdep_scale(t) for t in _TDEP_TGRID])
    cases.append(
        ("tdep_axi", SCFPotential(amp=1.4, Acos=arr, a=_TDEP_A, tgrid=_TDEP_TGRID))
    )
    # non-axisymmetric time-dependent SCF (full n,l,m incl. Asin)
    Ac = numpy.tril(rng.normal(size=(3, 3, 3)) * 0.1)
    As = numpy.tril(rng.normal(size=(3, 3, 3)) * 0.1)
    Ac[0, 0, 0] = 1.0
    Aca = numpy.array([Ac * _tdep_scale(t) for t in _TDEP_TGRID])
    Asa = numpy.array([As * _tdep_scale(t) for t in _TDEP_TGRID])
    cases.append(
        (
            "tdep_nonaxi",
            SCFPotential(amp=1.1, Acos=Aca, Asin=Asa, a=_TDEP_A, tgrid=_TDEP_TGRID),
        )
    )
    return cases


TDEP_CASES = _make_tdep_cases()
TDEP_IDS = [name for name, _ in TDEP_CASES]
# evaluation times: off the grid nodes (interpolation genuinely exercised),
# spanning the grid, plus one beyond the last node (finite extrapolation).
_TDEP_TS = [0.37, 1.83, 3.14, 4.61, 5.4]
_TDEP_METHODS = ["_evaluate", "_dens", "_Rforce", "_zforce", "_phitorque", "_R2deriv"]


@pytest.mark.parametrize("name,pot", TDEP_CASES, ids=TDEP_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_tdep_value_parity_scalar_t(backend_name, name, pot):
    # A time-dependent SCF evaluated at a backend scalar time must match the numpy
    # (scipy-CubicSpline) evaluation at that time to ~1 ulp, for every method.
    for mname in _TDEP_METHODS:
        method = getattr(pot, mname)
        for t in _TDEP_TS:
            ref = numpy.asarray(method(_R0, _Z0, _PHI0, t=t))
            got = as_numpy(
                method(
                    _asarray(backend_name, _R0),
                    _asarray(backend_name, _Z0),
                    _asarray(backend_name, _PHI0),
                    t=_asarray(backend_name, t),
                )
            )
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-11,
                atol=1e-13,
                err_msg=f"tdep SCF[{name}].{mname} scalar-t parity "
                f"({backend_name}, t={t})",
            )


@pytest.mark.parametrize("name,pot", TDEP_CASES, ids=TDEP_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_tdep_value_parity_array_t(backend_name, name, pot):
    # A single point at an ARRAY of times: each entry picks up its own
    # time-interpolated coefficients (the batched per-point-coefficient path).
    t_arr = numpy.asarray(_TDEP_TS)
    ref = numpy.asarray(pot._evaluate(_R0, _Z0, _PHI0, t=t_arr))
    got = as_numpy(
        pot._evaluate(
            _asarray(backend_name, _R0),
            _asarray(backend_name, _Z0),
            _asarray(backend_name, _PHI0),
            t=_asarray(backend_name, t_arr),
        )
    )
    numpy.testing.assert_allclose(
        got,
        ref,
        rtol=1e-11,
        atol=1e-13,
        err_msg=f"tdep SCF[{name}] array-t parity ({backend_name})",
    )
    assert got.shape == t_arr.shape
    assert not numpy.allclose(got, got[0])  # genuinely time-dependent


@pytest.mark.parametrize("name,pot", TDEP_CASES, ids=TDEP_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_grad_in_t_vs_finite_difference(backend_name, name, pot):
    # HEADLINE: differentiability of a time-dependent SCF w.r.t. the evaluation
    # time t, flowing through the cubic-spline coefficient interpolation. AD of
    # _evaluate wrt t matches a central finite difference of the numpy _evaluate.
    eps = 1e-6
    for t0 in [0.37, 1.83, 3.14]:  # mid-segment points (away from the grid nodes)
        fd = (
            float(pot._evaluate(_R0, _Z0, _PHI0, t=t0 + eps))
            - float(pot._evaluate(_R0, _Z0, _PHI0, t=t0 - eps))
        ) / (2 * eps)
        ad = _grad_in_t(backend_name, pot, t0)
        assert abs(fd) > 1e-6  # the time dependence is non-trivial at this point
        numpy.testing.assert_allclose(
            ad,
            fd,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"tdep SCF[{name}] d_evaluate/dt ({backend_name}, t0={t0})",
        )


def _grad_in_t(backend_name, pot, t0):
    # AD of pot._evaluate(_R0, _Z0, _PHI0, t) wrt the scalar time t.
    R = _asarray(backend_name, _R0)
    z = _asarray(backend_name, _Z0)
    phi = _asarray(backend_name, _PHI0)
    if backend_name == "jax":
        return float(jax.grad(lambda t: pot._evaluate(R, z, phi, t=t))(jnp.asarray(t0)))
    leaf = torch.tensor(t0, dtype=torch.float64, requires_grad=True)
    out = pot._evaluate(R, z, phi, t=leaf)
    out.backward()
    return float(leaf.grad)


@pytest.mark.parametrize("name,pot", TDEP_CASES, ids=TDEP_IDS)
def test_tdep_jax_jit_in_t(name, pot):
    # The time-interpolation path is jit-compatible (searchsorted / clip / Horner,
    # no float()/md5 caches on the backend), including its gradient in t.
    if jax is None:  # pragma: no cover
        pytest.skip("jax not available")
    R, z, phi = jnp.asarray(_R0), jnp.asarray(_Z0), jnp.asarray(_PHI0)
    f = lambda t: pot._evaluate(R, z, phi, t=t)
    t0 = 2.4
    assert_jit_matches_eager(
        f,
        jnp.asarray(t0),
        rtol=1e-11,
        atol=1e-13,
        ref=float(pot._evaluate(_R0, _Z0, _PHI0, t=t0)),
        err_msg=f"SCF[{name}] tdep jit in t",
    )
    g_jit = float(jax.jit(jax.grad(f))(jnp.asarray(t0)))
    eps = 1e-6
    fd = (
        float(pot._evaluate(_R0, _Z0, _PHI0, t=t0 + eps))
        - float(pot._evaluate(_R0, _Z0, _PHI0, t=t0 - eps))
    ) / (2 * eps)
    numpy.testing.assert_allclose(g_jit, fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "name,pot,ts",
    [(n, p, [0.0]) for n, p in CASES] + [(n, p, _TDEP_TS) for n, p in TDEP_CASES],
    ids=CASE_IDS + TDEP_IDS,
)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_mass_backend_parity(backend_name, name, pot, ts):
    # SCF enclosed mass M(<R) = R^2 * sum(Acos[l=0,m=0] * dphiTilde) evaluated on a
    # backend must match numpy. Exercises the _mass backend branch and its
    # _coeffs_at_time coefficient source for BOTH static (fixed) and
    # time-dependent (spline-interpolated-at-t) potentials.
    for t in ts:
        ref = numpy.asarray(pot._mass(_R0, t=t))
        got = as_numpy(
            pot._mass(_asarray(backend_name, _R0), t=_asarray(backend_name, t))
        )
        numpy.testing.assert_allclose(
            got,
            ref,
            rtol=1e-11,
            atol=1e-13,
            err_msg=f"SCF[{name}]._mass backend parity ({backend_name}, t={t})",
        )


###############################################################################
# Forced-backend coercion regression tests (burndown wave: expansion family).
# Under a forced backend (use(..., force=True)) the expwhole* DiskSCF catalogue
# potentials used to crash: _RToxi hit xp.isinf on a python/numpy scalar, and
# the from_density coefficient quadrature multiplied numpy basis arrays by a
# backend-array density (a user density closing over a backend-amp potential
# returns backend arrays even in the numpy-force setup). These assert the fixes.
###############################################################################


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_RToxi_parity(backend_name):
    # _RToxi on a backend array (incl. the r = inf -> xi = 1 limit, handled by
    # the xp.where(isinf) branch) matches the numpy transform.
    from galpy.potential.SCFPotential import _RToxi

    rr = numpy.array([0.05, 0.5, 1.3, 5.0, numpy.inf])
    numpy.testing.assert_allclose(
        as_numpy(_RToxi(_asarray(backend_name, rr), a=numpy.pi)),
        _RToxi(rr, a=numpy.pi),
        rtol=1e-13,
        atol=1e-14,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_RToxi_data_dispatch(backend_name):
    # Leaf data-guard: under a forced backend a numpy/scalar r stays numpy (so
    # the numpy coefficient setup / numpy parents keep working), while a genuine
    # backend array routes to the backend path.
    from galpy import backend as _b
    from galpy.backend import is_backend_array
    from galpy.potential.SCFPotential import _RToxi

    with _b.use(backend_name, force=True):
        assert not is_backend_array(_RToxi(1.4, a=1.3))
        assert not is_backend_array(_RToxi(numpy.linspace(0.1, 5.0, 5), a=1.3))
        assert is_backend_array(_RToxi(_asarray(backend_name, 2.0), a=1.3))


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("symmetry", ["spherical", "axisymmetric"])
def test_from_density_backend_amp_closure(backend_name, symmetry):
    # from_density's coefficient quadrature runs on numpy, but a user density
    # closing over a backend-amp potential returns backend arrays even there; the
    # setup must cast them to numpy (and the .to() units probe must treat a
    # backend array as non-physical) so the coefficients match the numpy build.
    from galpy import backend as _b
    from galpy.potential import HernquistPotential

    def build():
        hp = HernquistPotential(amp=2.0, a=1.3)
        return SCFPotential.from_density(
            lambda R, z: hp.dens(R, z), 6, 2, a=1.3, symmetry=symmetry
        )

    ref = numpy.asarray(build()._Acos)
    with _b.use(backend_name, force=True):
        got = as_numpy(build()._Acos)
    numpy.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-13)


###############################################################################
# scf_compute_coeffs_spherical: the coefficient quadrature follows the ambient
# namespace, so the expansion is differentiable w.r.t. the DENSITY's parameters
# (the thing you need to fit a density model through its SCF potential). The
# whole function used to be pinned to numpy, which silently severed that
# gradient -- silently, because the values were still correct.
###############################################################################


def _plummer_dens(b, xp=numpy):
    # Finite at r = 0, so the density-arity autodetect resolves to one argument
    # (a density that raises at r=0 falls through to the 3-argument branch and
    # is then called with zeroed scale/mass, which silently returns all-zero
    # coefficients -- a trap worth avoiding in tests).
    def dens(r):
        return 3.0 / (4.0 * numpy.pi) * b**3 * (b**2 + r**2) ** -2.5

    return dens


def _scalar_grad(backend_name, fn, x0):
    """d fn / dx at x0 for a scalar backend parameter."""
    if backend_name == "jax":
        return float(jax.grad(fn)(jnp.asarray(x0)))
    leaf = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    out = fn(leaf)
    out.backward()
    return float(leaf.grad)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_scf_coeffs_spherical_backend_parity(backend_name):
    # A backend density yields BACKEND coefficients (under the old numpy pin the
    # values were right but the array was numpy, so gradients were already gone)
    # and they agree with the numpy result.
    from galpy import backend as _b
    from galpy.backend import is_backend_array
    from galpy.potential.SCFPotential import scf_compute_coeffs_spherical

    ref, _ = scf_compute_coeffs_spherical(_plummer_dens(1.3), 6, a=1.0)
    with _b.use(backend_name, force=True):
        got, _ = scf_compute_coeffs_spherical(
            _plummer_dens(_asarray(backend_name, 1.3)), 6, a=1.0
        )
        assert is_backend_array(got), (
            "coefficients came back numpy, so the density gradient is severed"
        )
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_scf_coeffs_spherical_grad_wrt_density_parameter(backend_name):
    # d A_000 / d b against central finite differences on the SAME code path.
    from galpy import backend as _b
    from galpy.potential.SCFPotential import scf_compute_coeffs_spherical

    def A0(b):
        A, _ = scf_compute_coeffs_spherical(_plummer_dens(b), 4, a=1.0)
        return A[0, 0, 0]

    b0, h = 1.3, 1e-6
    with _b.use(backend_name, force=True):
        fd = float(as_numpy(A0(b0 + h)) - as_numpy(A0(b0 - h))) / (2 * h)
        grad = _scalar_grad(backend_name, A0, b0)
    assert numpy.fabs(grad - fd) / numpy.fabs(fd) < 1e-6, (
        f"d A000/db = {grad!r} disagrees with finite differences {fd!r}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_scf_coeffs_axi_grad_wrt_density_parameter(backend_name):
    # Same check as the spherical case for the AXISYMMETRIC quadrature, which
    # has its own integrand (and its own time-dependent twin).
    from galpy import backend as _b
    from galpy.potential.SCFPotential import scf_compute_coeffs_axi

    def dens_of(b):
        def dens(R, z):
            r2 = R**2 + z**2
            return 3.0 / (4.0 * numpy.pi) * b**3 * (b**2 + r2) ** -2.5

        return dens

    def A0(b):
        A, _ = scf_compute_coeffs_axi(dens_of(b), 3, 2, a=1.0)
        return A[0, 0, 0]

    b0, h = 1.3, 1e-6
    with _b.use(backend_name, force=True):
        fd = float(as_numpy(A0(b0 + h)) - as_numpy(A0(b0 - h))) / (2 * h)
        grad = _scalar_grad(backend_name, A0, b0)
    assert numpy.fabs(grad - fd) / numpy.fabs(fd) < 1e-5, (
        f"axi d A000/db = {grad!r} disagrees with finite differences {fd!r}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_scf_coeffs_general_grad_wrt_density_parameter(backend_name):
    # The general (non-axisymmetric) quadrature. The cos(2 phi) term averages
    # out of the m=0 monopole, so A000 -- and its derivative -- must match the
    # spherical and axi routines; that agreement is a cross-routine check.
    from galpy import backend as _b
    from galpy.potential.SCFPotential import scf_compute_coeffs

    def dens_of(b):
        def dens(R, z, phi):
            r2 = R**2 + z**2
            return (
                3.0
                / (4.0 * numpy.pi)
                * b**3
                * (b**2 + r2) ** -2.5
                * (1.0 + 0.1 * numpy.cos(2 * phi))
            )

        return dens

    def A0(b):
        A, _ = scf_compute_coeffs(dens_of(b), 2, 2, a=1.0)
        return A[0, 0, 0]

    b0, h = 1.3, 1e-6
    with _b.use(backend_name, force=True):
        fd = float(as_numpy(A0(b0 + h)) - as_numpy(A0(b0 - h))) / (2 * h)
        grad = _scalar_grad(backend_name, A0, b0)
    assert numpy.fabs(grad - fd) / numpy.fabs(fd) < 1e-5, (
        f"general d A000/db = {grad!r} disagrees with finite differences {fd!r}"
    )
