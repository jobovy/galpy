###############################################################################
# test_backend_multipole.py: multi-backend tests for the
# MultipoleExpansionPotential backend (jax/torch) evaluation path (P2.5).
#
# The numpy path evaluates the radial BPoly/spline objects point-by-point and
# is bit-for-bit unchanged; the backend path evaluates the same piecewise
# polynomials as power-basis (PPoly) coefficient stacks via searchsorted +
# Horner and the angular part via galpy.backend.special.assoc_legendre. This
# module proves, for spherical / axisymmetric / non-axisymmetric expansions:
#   1. numpy / jax / torch value parity for the potential, all three forces,
#      all six second derivatives, and the density -- on a grid that spans the
#      below-grid (r < rmin), in-grid, and above-grid (r > rmax) regions of the
#      radial expansion (so every piecewise branch and its dead-side guards
#      are exercised),
#   2. autodiff of _evaluate matches finite differences of the numpy path,
#   3. the analytic identities AD(_evaluate) == -force and AD(force) ==
#      -second-derivative, including the phi pairs of the non-axisymmetric
#      case and the below-/above-grid extrapolation regions,
#   4. jax.jit produces identical values,
#   5. time-dependent (tgrid=...) instances raise NotImplementedError on
#      backend arrays (that path is numpy-only for now).
#
# Value/force parity is asserted at 1e-12; second derivatives at 1e-9
# (power-basis Horner vs Bernstein de-Casteljau evaluation of d2I/dr2 differs
# at the ~1e-10 rounding level -- the C backend uses the same power-basis
# representation and is held to 1e-8 in test_potential).
###############################################################################
import numpy
import pytest

from galpy.potential import MiyamotoNagaiPotential, MultipoleExpansionPotential

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

###############################################################################
# Test potentials: a pure monopole (L=1), an axisymmetric expansion of a
# Miyamoto-Nagai disk (L=6), and a non-axisymmetric expansion with both
# cos(phi) and sin(2 phi) terms (L=4). The coarse radial grid keeps the
# expansion cheap; rmin=0.05 / rmax=10 leave room to test both extrapolation
# regions.
###############################################################################
_RGRID = numpy.geomspace(0.05, 10.0, 101)
_HERN = lambda r: 1.0 / (2.0 * numpy.pi) / r / (1 + r) ** 3

_SPH = MultipoleExpansionPotential.from_density(
    _HERN, L=1, symmetry="spherical", rgrid=_RGRID
)
_AXI = MultipoleExpansionPotential.from_density(
    MiyamotoNagaiPotential(amp=1.3, a=0.5, b=0.3),
    L=6,
    symmetry="axisymmetric",
    rgrid=_RGRID,
)
_NONAXI = MultipoleExpansionPotential.from_density(
    lambda R, z, phi: (
        (1.0 + 0.5 * numpy.cos(phi) + 0.3 * numpy.sin(2.0 * phi))
        * _HERN(numpy.sqrt(R**2 + z**2))
    ),
    L=4,
    rgrid=_RGRID,
)

CASES = [_SPH, _AXI, _NONAXI]
CASE_IDS = ["spherical", "axisymmetric", "nonaxisymmetric"]

_FIRST_ORDER = ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_dens"]
_SECOND_ORDER = [
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_phi2deriv",
    "_Rphideriv",
    "_phizderiv",
]

# Grid spanning below-grid (r < rmin = 0.05), in-grid, and above-grid
# (r > rmax = 10) points; phi spans all quadrants.
_RS = [0.03, 0.5, 1.0, 2.0, 12.0]
_ZS = [0.02, 0.1, 0.2, 0.3, 3.0]
_PHIS = [0.3, 1.1, 2.0, 4.0, 5.5]


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
    phi = _asarray(backend_name, _PHIS)
    for method in _FIRST_ORDER + _SECOND_ORDER:
        ref = numpy.asarray(
            getattr(pot, method)(
                numpy.asarray(_RS), numpy.asarray(_ZS), numpy.asarray(_PHIS)
            )
        )
        got = _tonumpy(getattr(pot, method)(R, z, phi))
        rtol = 1e-9 if method in _SECOND_ORDER else 1e-12
        numpy.testing.assert_allclose(
            got,
            ref,
            rtol=rtol,
            atol=1e-13,
            err_msg=f"{CASE_IDS[CASES.index(pot)]}.{method}",
        )


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity_scalar(backend_name, pot):
    # Scalar (0-d) inputs must work and agree with the numpy scalar path.
    for method in _FIRST_ORDER + _SECOND_ORDER:
        ref = float(numpy.asarray(getattr(pot, method)(1.3, 0.4, 0.7)))
        got = float(
            _tonumpy(
                getattr(pot, method)(
                    _asarray(backend_name, 1.3),
                    _asarray(backend_name, 0.4),
                    _asarray(backend_name, 0.7),
                )
            )
        )
        rtol = 1e-9 if method in _SECOND_ORDER else 1e-12
        numpy.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-13)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot):
    # The public Rforce (through the unit decorators and _amp) must give
    # identical values across backends.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    ref = numpy.asarray(
        pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS), phi=numpy.asarray(_PHIS))
    )
    got = _tonumpy(pot.Rforce(R, z, phi=phi))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_vs_finite_difference(backend_name, pot):
    # Independent (finite-difference, numpy-path) cross-check that the
    # backend _evaluate is differentiable end-to-end; the exact analytic
    # identity is asserted (more tightly) in test_force_hessian_identities.
    R0, z0, phi0 = 1.3, 0.4, 0.7
    eps = 1e-6

    def phi_np(R):
        return float(pot._evaluate(numpy.asarray(R), numpy.asarray(z0), phi0))

    fd = (phi_np(R0 + eps) - phi_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0), jnp.asarray(phi0)))(
                jnp.asarray(R0)
            )
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        y = pot._evaluate(
            R,
            torch.tensor(z0, dtype=torch.float64),
            torch.tensor(phi0, dtype=torch.float64),
        )
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


###############################################################################
# Analytic-identity autodiff checks (galpy sign conventions):
#   AD(_evaluate wrt R) == -_Rforce      AD(_evaluate wrt z) == -_zforce
#   AD(_evaluate wrt phi) == -_phitorque
#   AD(_Rforce wrt R) == -_R2deriv       AD(_Rforce wrt z) == -_Rzderiv
#   AD(_Rforce wrt phi) == -_Rphideriv   AD(_zforce wrt z) == -_z2deriv
#   AD(_zforce wrt phi) == -_phizderiv   AD(_phitorque wrt phi) == -_phi2deriv
# For the axisymmetric/spherical cases the phi pairs are identically zero,
# which the atol covers.
###############################################################################
_R, _Z, _PHI = 0, 1, 2
_ID_PAIRS = [
    ("_evaluate", _R, "_Rforce"),
    ("_evaluate", _Z, "_zforce"),
    ("_evaluate", _PHI, "_phitorque"),
    ("_Rforce", _R, "_R2deriv"),
    ("_Rforce", _Z, "_Rzderiv"),
    ("_Rforce", _PHI, "_Rphideriv"),
    ("_zforce", _Z, "_z2deriv"),
    ("_zforce", _PHI, "_phizderiv"),
    ("_phitorque", _PHI, "_phi2deriv"),
]


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
    out.backward()
    return float(leaf.grad)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_force_hessian_identities(backend_name, pot):
    R0, z0, phi0 = 1.3, 0.4, 0.7
    for lower, argnum, higher in _ID_PAIRS:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, phi, _l=lower: getattr(pot, _l)(R, z, phi),
            R0,
            z0,
            phi0,
            argnum=argnum,
        )
        ref = -float(numpy.asarray(getattr(pot, higher)(R0, z0, phi0)))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{CASE_IDS[CASES.index(pot)]}: AD({lower})==-{higher}",
        )


@pytest.mark.parametrize(
    "point", [(0.03, 0.02, 0.7), (11.0, 2.0, 0.7)], ids=["below-grid", "above-grid"]
)
@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_extrapolation_region_identities(backend_name, pot, point):
    # The constant-density (r < rmin) and point-mass (r > rmax) extrapolation
    # branches must also be consistent under autodiff (their dead sides are
    # guarded with safe radii).
    R0, z0, phi0 = point
    for lower, argnum, higher in [
        ("_evaluate", _R, "_Rforce"),
        ("_evaluate", _Z, "_zforce"),
        ("_Rforce", _R, "_R2deriv"),
    ]:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, phi, _l=lower: getattr(pot, _l)(R, z, phi),
            R0,
            z0,
            phi0,
            argnum=argnum,
        )
        ref = -float(numpy.asarray(getattr(pot, higher)(R0, z0, phi0)))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{CASE_IDS[CASES.index(pot)]}@{point}: AD({lower})==-{higher}",
        )


@pytest.mark.skipif(jax is None, reason="jax not available")
@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
def test_jax_jit_matches(pot):
    R = jnp.asarray(_RS)
    z = jnp.asarray(_ZS)
    phi = jnp.asarray(_PHIS)
    for method in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        fn = getattr(pot, method)
        ref = numpy.asarray(fn(R, z, phi))
        got = numpy.asarray(jax.jit(fn)(R, z, phi))
        numpy.testing.assert_allclose(got, ref, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_backend_raises(backend_name):
    # Time-dependent (tgrid=...) instances are numpy-only: backend arrays
    # must raise a clear NotImplementedError instead of silently failing.
    tdep = MultipoleExpansionPotential.from_density(
        lambda R, z, phi, t=0.0: (1.0 + 0.1 * t) * _HERN(numpy.sqrt(R**2 + z**2)),
        L=1,
        symmetry="spherical",
        rgrid=_RGRID,
        tgrid=numpy.linspace(0.0, 2.0, 5),
    )
    R = _asarray(backend_name, 1.3)
    z = _asarray(backend_name, 0.4)
    # numpy evaluation still works
    assert numpy.isfinite(float(numpy.asarray(tdep._evaluate(1.3, 0.4, 0.7, 0.5))))
    for method in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        with pytest.raises(NotImplementedError, match="time-dependent"):
            getattr(tdep, method)(R, z, 0.7, 0.5)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_axi_phi_none_default(backend_name):
    # The backend paths default phi=None to 0 for axisymmetric expansions in
    # four places (the _evaluate dispatch, _backend_cyl_force,
    # _backend_cyl_2nd_deriv, and _backend_dens); call each without phi and
    # check against the explicit phi=0 result
    R = _asarray(backend_name, [0.5, 1.0, 2.0])
    z = _asarray(backend_name, [0.1, 0.2, 0.3])
    for meth in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        nophi = numpy.asarray(getattr(_AXI, meth)(R, z))
        withphi = numpy.asarray(getattr(_AXI, meth)(R, z, phi=0.0))
        assert numpy.amax(numpy.fabs(nophi - withphi)) == 0.0, (
            f"backend {backend_name} {meth} with phi omitted differs from phi=0"
        )
    return None
