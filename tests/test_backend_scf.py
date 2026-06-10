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


def _tonumpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return numpy.asarray(x)


@pytest.mark.parametrize("name,pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_array(backend_name, name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    for mname in METHODS:
        method = getattr(pot, mname)
        ref = numpy.asarray(method(_RS, _ZS, _PHIS))
        got = _tonumpy(method(R, z, phi))
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
            got = _tonumpy(
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
    got = _tonumpy(pot.Rforce(R, z, phi=phi))
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
        ref = float(method(_R0, _Z0, _PHI0))
        got = float(jax.jit(method)(R, z, phi))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"SCF[{name}].{mname} jit"
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
        _tonumpy(
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
        _tonumpy(
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
        _tonumpy(
            pot._R2deriv(
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.3),
            )
        )
    )
    numpy.testing.assert_allclose(gotc, refc, atol=1e-14)
