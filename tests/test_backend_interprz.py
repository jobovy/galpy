###############################################################################
# test_backend_interprz.py: multi-backend tests for interpRZPotential.
#
# interpRZPotential interpolates precomputed R-z grids of the potential, the two
# forces, the three interpolated 2nd derivatives, and the density with scipy
# ``RectBivariateSpline``. The numpy code path keeps calling the scipy splines'
# ``.ev`` exactly as before (byte-identical); the jax/torch path evaluates the
# SAME tensor-product piecewise polynomial (via ``rect_bivariate_to_ppoly`` +
# ``eval_rect_ppoly``: searchsorted + 2D Horner in namespace ops), so values
# agree with scipy to ~1 ulp and the potential is exactly autodifferentiable.
#
# For every backend this proves:
#   1. numpy / jax / torch produce identical values (rtol=1e-9) for the seven
#      interpolated 2D methods, on an interior grid that includes negative z
#      (the zsym odd-force branch) and both logR conventions;
#   2. jit + jacfwd over the public evaluatePotentials/evaluateRforces on backend
#      (R,z) return finite (traced safety);
#   3. autodiff of the interpolated force/potential matches central finite
#      differences computed on the numpy/scipy-spline path (grad-vs-FD).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager

from galpy.backend import as_numpy, is_backend_array
from galpy.potential import (
    MiyamotoNagaiPotential,
    MWPotential2014,
    evaluatePotentials,
    evaluateRforces,
    interpRZPotential,
)

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

_INTERP_KW = dict(
    interpPot=True,
    interpRforce=True,
    interpzforce=True,
    interpR2deriv=True,
    interpz2deriv=True,
    interpRzderiv=True,
    interpDens=True,
    zsym=True,
)


def _build(logR):
    base = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1)
    rgrid = (numpy.log(0.05), numpy.log(16.0), 21) if logR else (0.05, 16.0, 21)
    return interpRZPotential(
        RZPot=base, rgrid=rgrid, zgrid=(0.0, 1.0, 21), logR=logR, **_INTERP_KW
    )


# Built once (grid construction is the expensive part).
CASES = [_build(True), _build(False)]
CASE_IDS = ["logR", "linR"]

_METHODS = [
    "_evaluate",
    "_Rforce",
    "_zforce",
    "_R2deriv",
    "_z2deriv",
    "_Rzderiv",
    "_dens",
]

# Interior query grid; negative z exercises the zsym odd (sign-flipped) branch.
_RS = numpy.array([0.3, 0.8, 1.0, 1.3, 2.5, 8.0])
_ZS = numpy.array([-0.4, -0.1, 0.05, 0.15, 0.3, 0.6])


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    if backend_name == "torch":
        return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for method in _METHODS:
        ref = numpy.asarray(getattr(pot, method)(_RS, _ZS))
        raw = getattr(pot, method)(R, z)
        if backend_name != "numpy":
            assert is_backend_array(raw), f"{method} not a backend array"
        got = as_numpy(raw)
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-9, atol=1e-11, err_msg=f"{CASE_IDS}.{method}"
        )


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_public_value_parity(backend_name, pot):
    # The public evaluate* (through the unit decorators and _amp) agree too.
    R = _asarray(backend_name, _RS)
    z = _asarray(backend_name, _ZS)
    for fn in (evaluatePotentials, evaluateRforces):
        ref = numpy.asarray(fn(pot, _RS, _ZS))
        got = as_numpy(fn(pot, R, z))
        numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
def test_jax_traced_matches_numpy_and_fd(pot):
    # Finiteness says nothing: a trace that folded its arguments away is finite
    # too. Pin the traced VALUE to the plain-numpy one (and let
    # assert_jit_matches_eager check the jaxpr really consumes its arguments),
    # and the traced DERIVATIVE to central FD.
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    Rj = jnp.asarray(_RS)
    zj = jnp.asarray(_ZS)
    h = 1e-5
    for fn in (evaluatePotentials, evaluateRforces):
        assert_jit_matches_eager(
            lambda R_, z_, _f=fn: _f(pot, R_, z_),
            Rj,
            zj,
            rtol=1e-14,
            atol=0.0,
            ref=numpy.asarray(fn(pot, _RS, _ZS)),
            err_msg=fn.__name__,
        )
        # elementwise in R, so the Jacobian is exactly diagonal
        jac = as_numpy(jax.jacfwd(lambda R_, _f=fn: _f(pot, R_, zj))(Rj))
        offdiag = jac - numpy.diag(numpy.diag(jac))
        assert numpy.all(offdiag == 0.0), f"{fn.__name__}: Jacobian not diagonal"
        fd = (
            numpy.asarray(fn(pot, _RS + h, _ZS)) - numpy.asarray(fn(pot, _RS - h, _ZS))
        ) / (2.0 * h)
        numpy.testing.assert_allclose(
            numpy.diag(jac), fd, rtol=2e-9, err_msg=f"{fn.__name__} d/dR vs FD"
        )


# One interior (R,z) point; grad w.r.t. R (Rforce) and z (potential).
_FD_R0, _FD_Z0 = 1.15, 0.22


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_rforce_vs_fd(backend_name, pot):
    # d(Rforce)/dR: AD vs central FD on the numpy/scipy-spline path.
    eps = 1e-5

    def f_np(Rv):
        return float(evaluateRforces(pot, numpy.asarray(Rv), numpy.asarray(_FD_Z0)))

    fd = (f_np(_FD_R0 + eps) - f_np(_FD_R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda Rv: evaluateRforces(pot, Rv, jnp.asarray(_FD_Z0)))(
                jnp.asarray(_FD_R0)
            )
        )
    else:
        R = torch.tensor(_FD_R0, dtype=torch.float64, requires_grad=True)
        y = evaluateRforces(pot, R, torch.tensor(_FD_Z0, dtype=torch.float64))
        y.backward()
        ad = float(R.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("pot", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_pot_wrt_z_vs_fd(backend_name, pot):
    # d(Pot)/dz: AD vs central FD; -d(Pot)/dz should also track zforce.
    eps = 1e-5

    def f_np(zv):
        return float(evaluatePotentials(pot, numpy.asarray(_FD_R0), numpy.asarray(zv)))

    fd = (f_np(_FD_Z0 + eps) - f_np(_FD_Z0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda zv: evaluatePotentials(pot, jnp.asarray(_FD_R0), zv))(
                jnp.asarray(_FD_Z0)
            )
        )
    else:
        zt = torch.tensor(_FD_Z0, dtype=torch.float64, requires_grad=True)
        y = evaluatePotentials(pot, torch.tensor(_FD_R0, dtype=torch.float64), zt)
        y.backward()
        ad = float(zt.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


###############################################################################
# 1D interpolators (vcirc / dvcircdR / epifreq / verticalfreq). Each is a 1D
# InterpolatedUnivariateSpline of the corresponding original-potential quantity
# on the R-grid. numpy keeps calling the scipy spline (byte-identical); the
# jax/torch path evaluates the SAME frozen piecewise polynomial (spline_to_ppoly
# + eval_ppoly), so values match scipy to ~1 ulp and the quantity is exactly
# autodifferentiable in R.
###############################################################################
_1D_METHODS = ["vcirc", "dvcircdR", "epifreq", "verticalfreq"]


def _build_1d(logR):
    base = MiyamotoNagaiPotential(amp=1.0, a=0.5, b=0.1)
    rgrid = (numpy.log(0.05), numpy.log(16.0), 31) if logR else (0.05, 16.0, 31)
    return interpRZPotential(
        RZPot=base,
        rgrid=rgrid,
        zgrid=(0.0, 1.0, 5),
        logR=logR,
        interpvcirc=True,
        interpdvcircdr=True,
        interpepifreq=True,
        interpverticalfreq=True,
    )


CASES_1D = [_build_1d(True), _build_1d(False)]

# Interior R grid (inside [0.05, 16.0] so the on-grid interpolant is exercised).
_RS_1D = numpy.array([0.3, 0.8, 1.0, 1.3, 2.5, 8.0])


@pytest.mark.parametrize("pot", CASES_1D, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_1d_value_parity(backend_name, pot):
    R = _asarray(backend_name, _RS_1D)
    for method in _1D_METHODS:
        ref = numpy.asarray(getattr(pot, method)(_RS_1D, use_physical=False))
        raw = getattr(pot, method)(R, use_physical=False)
        if backend_name != "numpy":
            assert is_backend_array(raw), f"{method} not a backend array"
        numpy.testing.assert_allclose(
            as_numpy(raw), ref, rtol=1e-9, atol=1e-11, err_msg=method
        )


@pytest.mark.parametrize("pot", CASES_1D, ids=CASE_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_1d_grad_vcirc_vs_fd(backend_name, pot):
    # d(vcirc)/dR: AD vs central FD on the numpy/scipy-spline path.
    eps = 1e-5
    R0 = 1.15

    def f_np(Rv):
        return float(pot.vcirc(numpy.asarray([Rv]), use_physical=False)[0])

    fd = (f_np(R0 + eps) - f_np(R0 - eps)) / (2 * eps)
    if backend_name == "jax":
        ad = float(
            jax.grad(lambda Rv: pot.vcirc(Rv, use_physical=False))(jnp.asarray(R0))
        )
    else:
        Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
        pot.vcirc(Rt, use_physical=False).backward()
        ad = float(Rt.grad)
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-7)


def test_1d_jax_traced_matches_numpy():
    if jax is None:  # pragma: no cover
        pytest.skip("jax not installed")
    Rj = jnp.asarray(_RS_1D)
    for case_id, pot in zip(CASE_IDS, CASES_1D):
        for method in _1D_METHODS:
            fn = getattr(pot, method)
            assert_jit_matches_eager(
                lambda R_, _f=fn: _f(R_, use_physical=False),
                Rj,
                rtol=1e-14,
                atol=0.0,
                ref=numpy.asarray(fn(_RS_1D, use_physical=False)),
                err_msg=f"{method} ({case_id})",
            )


###############################################################################
# StaeckelGrid interpecc divergence (the payoff of the interpRZ backend
# migration). A grid built under a forced backend recomputes the bad (extreme)
# orbits' (ecc,zmax,rperi,rap) with the INTERP-potential Staeckel (self._aA) --
# exactly as the numpy build does -- now that interpRZPotential evaluates in the
# backend. Before the fix the backend build used the ORIGINAL-potential Staeckel
# (tmpaA), so the backend ecc/rperi tables diverged ~O(1) (>90%) from the numpy
# grid on off-grid orbits; here we assert backend == numpy at off-grid orbits.
###############################################################################
def _interprz_for_grid():
    return interpRZPotential(
        RZPot=MWPotential2014,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 41),
        zgrid=(0.0, 1.0, 41),
        logR=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_staeckelgrid_interpecc_divergence_fixed(backend_name):
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.backend import use

    if backend_name == "torch":
        # Building a StaeckelGrid on an interpRZPotential under forced torch hits a
        # SEPARATE pre-existing numpy-scalar-under-forced-torch issue in
        # _build_grid_backend (torch.sqrt rejects the numpy-scalar vcirc(Rmax));
        # jax tolerates numpy scalars). It crashes identically on the base branch
        # and is unrelated to the tmpaA->self._aA divergence fix under test here,
        # which is backend-agnostic and verified on jax.
        pytest.skip("StaeckelGrid-on-interpRZ build under forced torch: pre-existing")

    ipot = _interprz_for_grid()
    delta, nE, npsi, nLz = 0.45, 16, 16, 18
    g_np = actionAngleStaeckelGrid(
        pot=ipot, delta=delta, nE=nE, npsi=npsi, nLz=nLz, interpecc=True, c=True
    )
    with use(backend_name, force=True):
        g_be = actionAngleStaeckelGrid(
            pot=ipot, delta=delta, nE=nE, npsi=npsi, nLz=nLz, interpecc=True, c=True
        )
    # realistic bound, interior off-grid orbits (well inside the E/Lz grid)
    rng = numpy.random.default_rng(1)
    N = 25
    R = rng.uniform(0.6, 1.8, N)
    vR = rng.uniform(-0.12, 0.12, N)
    vT = rng.uniform(0.65, 1.05, N)
    z = rng.uniform(-0.18, 0.18, N)
    vz = rng.uniform(-0.12, 0.12, N)
    ref = g_np.EccZmaxRperiRap(R, vR, vT, z, vz)
    with use(backend_name, force=True):
        got = g_be.EccZmaxRperiRap(
            *[_asarray(backend_name, v) for v in (R, vR, vT, z, vz)]
        )
    for name, r, gg in zip(("ecc", "zmax", "rperi", "rap"), ref, got):
        assert is_backend_array(gg), f"{name} not a backend array"
        # after the fix backend==numpy at the shared query-path floor (~1e-2);
        # the pre-fix tmpaA bug diverged ecc/rperi by >90% (maxabs ~5), so this
        # 2e-2 tolerance cleanly separates fixed from buggy.
        numpy.testing.assert_allclose(
            as_numpy(gg), numpy.asarray(r), rtol=2e-2, atol=2e-2, err_msg=name
        )
