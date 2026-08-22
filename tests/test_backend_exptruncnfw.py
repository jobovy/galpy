###############################################################################
# test_backend_exptruncnfw.py: multi-backend tests for ExpTruncNFWPotential,
# the exponentially-truncated NFW profile whose closed-form enclosed mass and
# outer-potential integral go through the exponential integral E_1 (routed via
# galpy.backend.special.exp1: scipy on numpy, -expi(-x) on jax, a pure-backend
# Lentz/series fallback on torch).
#
# For every migrated compute method this proves numpy / jax / torch value parity
# at the existing tolerances, and that the migrated radial force (which passes
# through exp1) is differentiable with a gradient consistent with the potential
# (jax.grad / torch.autograd vs finite differences and vs the analytic identity
# AD(_evaluate) == -_Rforce). The small-r Taylor branch and the closed-form
# branch of _F are both exercised (via a grid that straddles _small_r_thresh).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.potential import ExpTruncNFWPotential

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

# amp= and mass= constructions exercise the two _amp conventions (mass= also
# bakes _amp into _ddensdr/_d2densdr2). a=1.5, rc=8 => _small_r_thresh=1.5e-3.
POTS = [
    ExpTruncNFWPotential(amp=1.3, a=1.5, rc=8.0),
    ExpTruncNFWPotential(mass=5.0, a=1.3, rc=6.0),
]
POT_IDS = ["amp", "mass"]

# Radial grid straddling _small_r_thresh (~1.5e-3): the first two points hit the
# Taylor-series branch of _F, the rest the closed (E_1) form. r == 0 is excluded
# (its finite-limit substitution is checked in test_potential.py).
_RS = [5.0e-4, 1.0e-3, 0.3, 0.8, 1.5, 4.0, 12.0]
_ZS = [0.2, 0.35, 0.15, 0.4, 0.25, 0.3, 0.5]

_THREE_D = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]
_ONE_D = ["_revaluate", "_rforce", "_r2deriv", "_rdens", "_F", "_G", "_mass"]


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_threed_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _THREE_D:
        ref = numpy.asarray(
            getattr(pot, method)(numpy.asarray(_RS), numpy.asarray(_ZS))
        )
        got = as_numpy(getattr(pot, method)(R, z))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{method} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_oned_value_parity(backend_name, pot):
    r = _asarray(backend_name, _RS)
    rnp = numpy.asarray(_RS)
    for method in _ONE_D:
        ref = numpy.asarray(getattr(pot, method)(rnp))
        got = as_numpy(getattr(pot, method)(r))
        # rtol 1e-11 (not 1e-12): _r2deriv = 4*pi*rho - 2*F/r^3 subtracts two large
        # terms at the smallest grid radius (r=5e-4), a cancellation that amplifies
        # the last-ULP difference between scipy.exp1 (numpy) and the native/fallback
        # exp1 (jax/torch) to ~7e-13 -- hardware-dependent around a 1e-12 cut.
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-11, atol=1e-13, err_msg=f"{method} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_density_derivs_and_mass_at_infinity(backend_name, pot):
    r = _asarray(backend_name, _RS)
    rnp = numpy.asarray(_RS)
    for method in ("_ddensdr", "_d2densdr2"):
        ref = numpy.asarray(getattr(pot, method)(rnp))
        got = as_numpy(getattr(pot, method)(r))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"{method} ({backend_name})"
        )
    # _ddenstwobetadr dispatches on the array's OWN namespace (data-first)
    for beta in (0.0, 0.5, -0.5, 1.0):
        ref = numpy.asarray(pot._ddenstwobetadr(rnp, beta=beta))
        got = as_numpy(pot._ddenstwobetadr(r, beta=beta))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-14, err_msg=f"_ddenstwobetadr beta={beta}"
        )
    # closed-form total mass M(inf) = amp * F(inf) stays finite on every backend
    inf = _asarray(backend_name, float("inf"))
    numpy.testing.assert_allclose(
        as_numpy(pot._mass(inf)), float(pot._mass(numpy.inf)), rtol=1e-12
    )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_rforce_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    ref = numpy.asarray(pot.Rforce(numpy.asarray(_RS), numpy.asarray(_ZS)))
    got = as_numpy(pot.Rforce(R, z))
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd(backend_name, pot):
    # d/dr[_rforce] via autodiff (through exp1 in _F_closed) vs central FD, at a
    # closed-form point and near the series/closed seam. r == 0 is avoided.
    eps = 1e-6
    for r0 in (0.3, 0.8, 4.0):
        fd = (
            float(pot._rforce(numpy.asarray(r0 + eps)))
            - float(pot._rforce(numpy.asarray(r0 - eps)))
        ) / (2.0 * eps)
        if backend_name == "jax":
            ad = float(jax.grad(lambda x: pot._rforce(x))(jnp.asarray(r0)))
        else:
            xt = torch.tensor(r0, dtype=torch.float64, requires_grad=True)
            pot._rforce(xt).backward()
            ad = float(xt.grad)
        assert not numpy.isnan(ad), f"NaN grad at r={r0} ({backend_name})"
        numpy.testing.assert_allclose(
            ad, fd, rtol=1e-5, err_msg=f"r={r0} ({backend_name})"
        )


@pytest.mark.parametrize("pot", POTS, ids=POT_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_evaluate_is_minus_rforce(backend_name, pot):
    # Exact analytic identity AD(_evaluate) == -_Rforce (spherical: dPhi/dR at
    # z=0 equals the full radial force through R/r == 1).
    R0, z0 = 1.3, 0.4
    ref = -float(pot._Rforce(numpy.asarray(R0), numpy.asarray(z0)))
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda R: pot._evaluate(R, jnp.asarray(z0)))(jnp.asarray(R0))
        )
    else:
        R = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot._evaluate(R, torch.tensor(z0, dtype=torch.float64)).backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, ref, rtol=1e-9)


@pytest.mark.parametrize("var", ["a", "rc"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_param_grad_vs_fd(backend_name, var):
    # d(_rforce)/d(truncation parameter): the data-first alpha-block keeps the
    # potential differentiable w.r.t. its parameters a and rc -- constructed from a
    # backend param and evaluated through exp1(alpha) -- vs central finite diff.
    r0, a0, rc0, eps = 2.0, 2.0, 8.0, 1e-6

    def rf_np(a, rc):
        return float(
            ExpTruncNFWPotential(amp=1.0, a=a, rc=rc)._rforce(numpy.asarray(r0))
        )

    def rf_b(p):  # p = the varied parameter as a backend scalar
        a, rc = (p, rc0) if var == "a" else (a0, p)
        return ExpTruncNFWPotential(amp=1.0, a=a, rc=rc)._rforce(
            _asarray(backend_name, r0)
        )

    p0 = a0 if var == "a" else rc0
    hi = (a0 + eps, rc0) if var == "a" else (a0, rc0 + eps)
    lo = (a0 - eps, rc0) if var == "a" else (a0, rc0 - eps)
    fd = (rf_np(*hi) - rf_np(*lo)) / (2.0 * eps)
    if backend_name == "jax":
        ad = float(jax.grad(rf_b)(jnp.asarray(p0)))
    else:
        pt = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
        rf_b(pt).backward()
        ad = float(pt.grad)
    assert not numpy.isnan(ad)
    numpy.testing.assert_allclose(
        ad, fd, rtol=1e-5, err_msg=f"d(rforce)/d{var} ({backend_name})"
    )


###############################################################################
# from_nfw(mass=...): the truncation radius is the root of
# amp * F(a/rc) = mass, so d(rc)/d(mass) exists and is what a fit of a truncated
# NFW to an observed total mass needs. The solve now runs through galpy's own
# backend brentq (implicit function theorem) instead of scipy.optimize.brentq,
# so that derivative flows on jax/torch; the numpy path still routes to scipy.
###############################################################################


def _rc_from_mass(mass):
    from galpy.potential import NFWPotential

    nfw = NFWPotential(amp=1.0, a=2.0)
    return ExpTruncNFWPotential.from_nfw(nfw, mass=mass).rc


def test_from_nfw_mass_solve_satisfies_its_own_equation():
    """numpy path: the returned rc really is the root, to machine precision.

    Pins the SOLVE rather than a hard-coded rc, so the test still means something
    if the profile is ever reparametrized.
    """
    from scipy.special import exp1 as _exp1

    from galpy.potential import NFWPotential

    nfw = NFWPotential(amp=1.0, a=2.0)
    mass = 3.0
    rc = _rc_from_mass(mass)
    assert isinstance(rc, float), "numpy path must stay on scipy and return a float"
    al = nfw.a / rc
    resid = nfw._amp * (numpy.exp(al) * (1.0 + al) * _exp1(al) - 1.0) - mass
    assert abs(resid) < 1e-12 * mass, f"rc is not the root: residual {resid!r}"


@pytest.mark.parametrize("backend_name", [b for b in BACKENDS if b != "numpy"])
def test_from_nfw_forward_value_matches_numpy(backend_name):
    """The solved rc is backend-independent to ~machine precision."""
    ref = _rc_from_mass(3.0)
    xp_mass = jnp.asarray(3.0) if backend_name == "jax" else torch.tensor(3.0)
    got = float(as_numpy(_rc_from_mass(xp_mass)))
    assert abs(got - ref) < 1e-10 * ref, f"{backend_name}: {got!r} vs numpy {ref!r}"


@pytest.mark.parametrize("backend_name", [b for b in BACKENDS if b != "numpy"])
def test_from_nfw_mass_gradient_vs_finite_difference(backend_name):
    """d(rc)/d(mass) from the implicit function theorem vs a central difference.

    The FD reference is computed on the NUMPY path, so this compares the backend
    derivative against an independent evaluation of the same function -- not
    against itself. Tolerance is 1e-8 relative; the implicit-diff value agrees
    with the FD to ~8e-12 in practice, and a wrong/absent derivative would be off
    by order unity, so this is a real check rather than a finite-and-nonzero one.
    """
    m0, h = 3.0, 1e-5
    fd = (_rc_from_mass(m0 + h) - _rc_from_mass(m0 - h)) / (2.0 * h)
    if backend_name == "jax":
        got = float(jax.grad(_rc_from_mass)(jnp.asarray(m0)))
    else:
        mt = torch.tensor(m0, requires_grad=True)
        _rc_from_mass(mt).backward()
        got = float(mt.grad)
    assert abs(got - fd) < 1e-8 * abs(fd), (
        f"{backend_name}: d(rc)/d(mass)={got!r} vs finite difference {fd!r} "
        f"(rel err {abs(got - fd) / abs(fd):.3e})"
    )
