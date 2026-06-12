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
#   5. the time-dependent (tgrid=...) path: numpy/jax/torch parity for the
#      potential, forces, second derivatives, and density at times on/between/
#      at-the-edges-of/outside the tgrid knots (the backend's clamped time
#      searchsorted must reproduce scipy's edge-interval cubic extrapolation),
#      for an axisymmetric and a non-axisymmetric time-dependent density;
#      autodiff and jax.jit at time-dependent times; the r = 0 and phi = None
#      guards; broadcasting over an array of times; and the end-to-end
#      composite DiskMultipoleExpansionPotential (KuijkenDubinski correction
#      layer on a time-dependent inner expansion) with jax arrays.
#
# Static value/force parity is asserted at 1e-12; static second derivatives at
# 1e-9 (power-basis Horner vs Bernstein de-Casteljau evaluation of d2I/dr2
# differs at the ~1e-10 rounding level -- the C backend uses the same
# power-basis representation and is held to 1e-8 in test_potential). The
# time-dependent numpy path evaluates the same power-basis stacks as the
# backend (via _fused_ppoly_eval), so ALL time-dependent parity -- second
# derivatives included -- is asserted at 1e-12.
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
def test_axi_phi_none_default(backend_name):
    # The backend paths default phi=None to 0 for axisymmetric expansions in
    # four places (the _evaluate dispatch, _backend_cyl_force,
    # _backend_cyl_2nd_deriv, and _backend_dens). The PUBLIC API forwards
    # phi=None when phi is omitted (the private-method signature default is
    # 0.0, NOT None -- so phi=None must be passed explicitly here to exercise
    # the guards) and the result must equal the explicit phi=0 one
    R = _asarray(backend_name, [0.5, 1.0, 2.0])
    z = _asarray(backend_name, [0.1, 0.2, 0.3])
    for meth in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        nophi = numpy.asarray(getattr(_AXI, meth)(R, z, phi=None))
        withphi = numpy.asarray(getattr(_AXI, meth)(R, z, phi=0.0))
        assert numpy.amax(numpy.fabs(nophi - withphi)) == 0.0, (
            f"backend {backend_name} {meth} with phi=None differs from phi=0"
        )
    return None


###############################################################################
# Time-dependent (tgrid=...) backend path. The numpy TD path evaluates
# CubicSpline-in-t stacks of power-basis radial PPoly coefficients through
# _fused_ppoly_eval / _eval_radial_lm_timedep; the backend path evaluates the
# exact same stacks via a clamped searchsorted in tgrid + cubic Horner in
# (t - t_k) followed by the static radial searchsorted + Horner, so parity is
# held at 1e-12 for everything, second derivatives included.
###############################################################################
_TD_TGRID = numpy.linspace(0.0, 2.0, 5)
# on-knot, between-knots, at-the-edges, and outside-the-grid times (the numpy
# path extrapolates outside tgrid with the edge-interval cubic, which the
# backend's clamped time-interval lookup must reproduce)
_TD_TS = [0.0, 0.37, 0.5, 1.21, 2.0, -0.2, 2.4]
_MN = MiyamotoNagaiPotential(amp=1.3, a=0.5, b=0.3)
_TD_AXI = MultipoleExpansionPotential.from_density(
    lambda R, z, t=0.0: (
        (1.0 + 0.2 * numpy.sin(0.7 * t)) * _MN.dens(R, z, use_physical=False)
    ),
    L=4,
    symmetry="axisymmetric",
    rgrid=_RGRID,
    tgrid=_TD_TGRID,
)
_TD_NONAXI = MultipoleExpansionPotential.from_density(
    lambda R, z, phi, t=0.0: (
        (
            1.0
            + 0.5 * numpy.cos(phi) * (1.0 + 0.1 * t)
            + 0.3 * numpy.sin(2.0 * phi - 0.5 * t)
        )
        * _HERN(numpy.sqrt(R**2 + z**2))
    ),
    L=4,
    rgrid=_RGRID,
    tgrid=_TD_TGRID,
)
_TD_CASES = [_TD_AXI, _TD_NONAXI]
_TD_IDS = ["tdep-axisymmetric", "tdep-nonaxisymmetric"]


@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_tdep_value_parity(backend_name, pot):
    # numpy/jax/torch parity of the potential, all three forces, all six
    # second derivatives, and the density on the below-/in-/above-grid radial
    # points at every kind of time relative to the tgrid knots; t is a Python
    # float here (the orbit-integration calling convention).
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    phi = _asarray(backend_name, _PHIS)
    for t in _TD_TS:
        for method in _FIRST_ORDER + _SECOND_ORDER:
            ref = numpy.asarray(
                getattr(pot, method)(
                    numpy.asarray(_RS), numpy.asarray(_ZS), numpy.asarray(_PHIS), t
                )
            )
            got = _tonumpy(getattr(pot, method)(R, z, phi, t))
            numpy.testing.assert_allclose(
                got,
                ref,
                rtol=1e-12,
                atol=1e-13,
                err_msg=f"{_TD_IDS[_TD_CASES.index(pot)]}.{method} at t={t}",
            )


@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_array_t_broadcast(backend_name, pot):
    # An array of times must broadcast against the coordinates exactly like
    # the numpy path does (one time-interval lookup per point).
    ts = numpy.asarray(_TD_TS[: len(_RS)])
    ref = numpy.asarray(
        pot._evaluate(numpy.asarray(_RS), numpy.asarray(_ZS), numpy.asarray(_PHIS), ts)
    )
    got = _tonumpy(
        pot._evaluate(
            _asarray(backend_name, _RS),
            _asarray(backend_name, _ZS),
            _asarray(backend_name, _PHIS),
            _asarray(backend_name, ts),
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_center(backend_name, pot):
    # Exact center at a between-knots time: _evaluate returns
    # R_00(rmin, t) * P_00 (the time-dependent R00 branch of
    # _backend_evaluate); forces and second derivatives are zero.
    t0 = 1.21
    ref = float(numpy.asarray(pot._evaluate(0.0, 0.0, 0.3, t0)))
    got = float(
        _tonumpy(
            pot._evaluate(
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.0),
                _asarray(backend_name, 0.3),
                t0,
            )
        )
    )
    numpy.testing.assert_allclose(got, ref, rtol=1e-12)
    for meth in ["_Rforce", "_zforce", "_phitorque", "_R2deriv"]:
        val = float(
            _tonumpy(
                getattr(pot, meth)(
                    _asarray(backend_name, 0.0),
                    _asarray(backend_name, 0.0),
                    _asarray(backend_name, 0.3),
                    t0,
                )
            )
        )
        assert val == 0.0, f"{meth} not zero at the center"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_axi_phi_none_default(backend_name):
    # Same phi=None guards as the static test, but through the
    # time-dependent branches (phi=None only arrives via the public API, so
    # it must be passed explicitly here).
    R = _asarray(backend_name, [0.5, 1.0, 2.0])
    z = _asarray(backend_name, [0.1, 0.2, 0.3])
    for meth in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        nophi = numpy.asarray(getattr(_TD_AXI, meth)(R, z, phi=None, t=0.83))
        withphi = numpy.asarray(getattr(_TD_AXI, meth)(R, z, phi=0.0, t=0.83))
        assert numpy.amax(numpy.fabs(nophi - withphi)) == 0.0, (
            f"backend {backend_name} {meth} with phi=None differs from phi=0"
        )
    return None


@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_grad_evaluate_vs_finite_difference(backend_name, pot):
    # AD of the backend _evaluate w.r.t. R, z (and phi for the
    # non-axisymmetric case) at a time-dependent time vs central finite
    # differences of the numpy path.
    R0, z0, phi0, t0 = 1.3, 0.4, 0.7, 0.83
    eps = 1e-6
    argnums = [_R, _Z] + ([_PHI] if pot.isNonAxi else [])
    for argnum in argnums:
        base = [R0, z0, phi0]

        def f_np(x):
            q = list(base)
            q[argnum] = x
            return float(
                pot._evaluate(numpy.asarray(q[0]), numpy.asarray(q[1]), q[2], t0)
            )

        fd = (f_np(base[argnum] + eps) - f_np(base[argnum] - eps)) / (2 * eps)
        ad = _grad_wrt(
            backend_name,
            lambda R, z, phi: pot._evaluate(R, z, phi, t0),
            R0,
            z0,
            phi0,
            argnum=argnum,
        )
        numpy.testing.assert_allclose(ad, fd, rtol=1e-5)


@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_tdep_force_hessian_identities(backend_name, pot):
    # The static analytic identities AD(_evaluate) == -force and
    # AD(force) == -second-derivative must also hold at time-dependent times
    # (this exercises the mode=1/2 time-dependent radial branches under AD).
    R0, z0, phi0, t0 = 1.3, 0.4, 0.7, 1.21
    for lower, argnum, higher in _ID_PAIRS:
        ad = _grad_wrt(
            backend_name,
            lambda R, z, phi, _l=lower: getattr(pot, _l)(R, z, phi, t0),
            R0,
            z0,
            phi0,
            argnum=argnum,
        )
        ref = -float(numpy.asarray(getattr(pot, higher)(R0, z0, phi0, t0)))
        numpy.testing.assert_allclose(
            ad,
            ref,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{_TD_IDS[_TD_CASES.index(pot)]}: AD({lower})==-{higher}",
        )


@pytest.mark.skipif(jax is None, reason="jax not available")
@pytest.mark.parametrize("pot", _TD_CASES, ids=_TD_IDS)
def test_tdep_jax_jit_matches(pot):
    # jit survival with t a traced argument; jitted == eager up to XLA's
    # fma/reassociation fuzz (~1e-14 absolute on the second derivatives).
    R = jnp.asarray(_RS)
    z = jnp.asarray(_ZS)
    phi = jnp.asarray(_PHIS)
    t = jnp.asarray(1.21)
    for method in ["_evaluate", "_Rforce", "_R2deriv", "_dens"]:
        fn = getattr(pot, method)
        ref = numpy.asarray(fn(R, z, phi, t))
        got = numpy.asarray(jax.jit(fn)(R, z, phi, t))
        numpy.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-13)


@pytest.mark.skipif(jax is None, reason="jax not available")
def test_tdep_diskmep_composite_jax():
    # End-to-end: a DiskMultipoleExpansionPotential whose inner multipole
    # expansion is time-dependent -- the #932 KuijkenDubinski correction
    # layer on top of the time-dependent backend path -- evaluated with jax
    # arrays must match the numpy path.
    from galpy.potential import DiskMultipoleExpansionPotential

    rgrid = numpy.geomspace(1e-2, 20, 51)
    dmep = DiskMultipoleExpansionPotential(
        dens=lambda R, z: 13.5 * numpy.exp(-3.0 * R) * numpy.exp(-27.0 * numpy.fabs(z)),
        Sigma={"h": 1.0 / 3.0, "type": "exp", "amp": 1.0},
        hz={"type": "exp", "h": 1.0 / 27.0},
        L=3,
        rgrid=rgrid,
    )
    # Replace the static inner expansion by a time-dependent one (same
    # pattern as the numpy-path tests in test_MultipoleExpansionPotential)
    static_me = dmep._me
    dmep._me = MultipoleExpansionPotential.from_density(
        dens=lambda R, z, phi, t=0.0: (
            static_me.dens(R, z, use_physical=False) * (1.0 + 0.1 * numpy.cos(t))
        ),
        L=static_me._L,
        rgrid=rgrid,
        tgrid=numpy.linspace(0.0, 10.0, 11),
        symmetry="axisymmetric",
    )
    assert dmep._me._tdep
    R = numpy.array([0.5, 1.0, 2.0])
    z = numpy.array([0.1, 0.2, 0.3])
    for meth in ["_evaluate", "_Rforce", "_zforce", "_R2deriv", "_z2deriv", "_dens"]:
        ref = numpy.asarray(getattr(dmep, meth)(R, z, 0.0, 0.9))
        got = numpy.asarray(
            getattr(dmep, meth)(jnp.asarray(R), jnp.asarray(z), 0.0, 0.9)
        )
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-12, atol=1e-12, err_msg=f"composite TD DiskMEP {meth}"
        )
