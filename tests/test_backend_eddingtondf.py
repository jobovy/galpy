###############################################################################
# test_backend_eddingtondf.py: Track F Pdf.2 -- backend (jax/torch) coverage
# for the Eddington-inversion isotropic DF family (eddingtondf). The numpy path
# is byte-identical (test_sphericaldf unchanged); this exercises the
# resolved-namespace dispatch: parity numpy<->jax<->torch of fE (the two
# GL-substituted half-integrals) / __call__ / moments / dM/dE, grad-vs-FD of fE
# and a moment, is-backend-array assertions, and the numpy-side sampling
# contract (numpy RNG draws unchanged under a forced backend, Spline1D f(E)).
###############################################################################
import numpy
import pytest

pytestmark = pytest.mark.backend_managed

BACKENDS = []
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

import galpy.backend
from galpy.backend import as_numpy
from galpy.df import eddingtondf
from galpy.potential import (
    DehnenCoreSphericalPotential,
    HernquistPotential,
    NFWPotential,
    PowerSphericalPotential,
)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(numpy.asarray(x, float))


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


# self-consistent DehnenCore (denspot == pot) and DehnenCore-in-NFW (denspot !=
# pot): the two eddington test regimes in test_sphericaldf
_DC = eddingtondf(pot=DehnenCoreSphericalPotential(amp=2.5, a=1.15))
_DCNFW = eddingtondf(
    pot=NFWPotential(amp=2.3, a=1.3),
    denspot=DehnenCoreSphericalPotential(amp=2.5, a=1.15),
)
_DFS = {"dehnencore": _DC, "dc_in_nfw": _DCNFW}


def _egrid(dfp):
    # in-bounds energies (fractionally between potInf and Emin)
    frac = numpy.linspace(0.05, 0.95, 15)
    return frac * (dfp._Emin - dfp._potInf) + dfp._potInf


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_parity(backend):
    # fE = the two GL-substituted half-integrals; N=100 GL agrees with scipy
    # adaptive quad to ~6e-8 (two independent methods). Higher GL order drifts
    # via the small-r turning-point (r=rphi, Phi-E->0) fp cancellation, so N=100
    # is the sweet spot; rtol 1e-6 covers the ~6e-8 gap with ~17x margin.
    for key, dfp in _DFS.items():
        Es = _egrid(dfp)
        ref = numpy.atleast_1d(dfp.fE(Es))
        got = dfp.fE(_arr(backend, Es))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_out_of_bounds(backend):
    # E > potInf (unbound) and E < Emin (below the inner cutoff) -> exactly 0 on
    # the dead branch, NaN-free (functional dummy-then-zero)
    dfp = _DC
    Eoob = numpy.array([dfp._potInf + 0.05, 0.5, dfp._Emin - 0.5])
    got = as_numpy(dfp.fE(_arr(backend, Eoob)))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # __call__ (E,) tuple form routes through _call_internal -> fE
    for key, dfp in _DFS.items():
        Es = _egrid(dfp)
        ref = dfp((Es,))
        got = dfp((_arr(backend, Es),))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moments_parity(backend):
    # sigmar (v-moment GL over the migrated base) and the isotropic beta==0
    rs = numpy.array([0.2, 0.5, 1.0, 2.0, 5.0])
    for key, dfp in _DFS.items():
        ref = numpy.array([dfp.sigmar(r) for r in rs])
        got = numpy.array([float(as_numpy(dfp.sigmar(_arr(backend, r)))) for r in rs])
        gotv = dfp.sigmar(_arr(backend, rs))
        assert _is_backend_array(backend, gotv)
        numpy.testing.assert_allclose(got, ref, rtol=1e-7)
        numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-7)
        b = dfp.beta(_arr(backend, 1.0))
        assert float(as_numpy(b)) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # dM/dE via the migrated base isotropic quadrature (r = rphi - s^2), using
    # the eddington fE + rphi interpolator
    for key, dfp in _DFS.items():
        Edm = numpy.linspace(0.15, 0.85, 7) * (dfp._Emin - dfp._potInf) + dfp._potInf
        ref = numpy.atleast_1d(dfp.dMdE(Edm))
        got = dfp.dMdE(_arr(backend, Edm))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)
    # out-of-bounds E -> exactly zero
    assert numpy.all(as_numpy(_DC.dMdE(_arr(backend, numpy.array([0.5])))) == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_grad_vs_fd(backend):
    # d(fE)/dE through the two GL half-integrals (limits + Phi(r) + rphi(E)); the
    # out-of-bounds grad is finite 0, not NaN (dead-branch guards)
    dfp = _DC
    E0 = 0.4 * (dfp._Emin - dfp._potInf) + dfp._potInf
    eps = 1e-6
    fd = (
        dfp.fE(numpy.atleast_1d(E0 + eps))[0] - dfp.fE(numpy.atleast_1d(E0 - eps))[0]
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda E: dfp.fE(E).sum())(jnp.asarray(E0)))
        goob = float(jax.grad(lambda E: dfp.fE(E).sum())(jnp.asarray(0.5)))
    else:
        t = torch.tensor(E0, requires_grad=True)
        dfp.fE(t).sum().backward()
        g = float(t.grad)
        t = torch.tensor(0.5, requires_grad=True)
        dfp.fE(t).sum().backward()
        goob = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)
    assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr through the GL moment integrals (limits + Phi(r) + fE)
    dfp = _DC
    r0, eps = 1.0, 1e-5
    fd = (dfp.sigmar(r0 + eps) - dfp.sigmar(r0 - eps)) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: dfp.sigmar(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        dfp.sigmar(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_numpy_side_forced(backend):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and the outputs are numpy arrays; the f(E)
    # interpolator (built via a forced-backend vectorized fE eval, pulled into a
    # Spline1D) and the vesc/mass grids run on the backend, so draws match the
    # pure-numpy ones to the grids' fp noise
    ref_df = eddingtondf(pot=DehnenCoreSphericalPotential(amp=2.5, a=1.15))
    numpy.random.seed(777)
    ref = ref_df.sample(n=200, return_orbit=False)
    dfb = eddingtondf(pot=DehnenCoreSphericalPotential(amp=2.5, a=1.15))
    numpy.random.seed(777)
    with galpy.backend.use(backend, force=True):
        got = dfb.sample(n=200, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ensure_fE_interp_forced(backend):
    # the f(E) interpolator: numpy builds a scipy spline, a forced backend builds
    # a Spline1D from the (backend-vectorized, numpy-pulled) fE grid; both give
    # the same f(E) and the Spline1D evaluates natively on backend queries
    ref_df = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
    ref_df._ensure_fE_interp()
    dfb = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
    with galpy.backend.use(backend, force=True):
        dfb._ensure_fE_interp()
    Es = _egrid(ref_df)
    numpy.testing.assert_allclose(dfb._fE_interp(Es), ref_df._fE_interp(Es), rtol=1e-6)
    gb = dfb._fE_interp(_arr(backend, Es))
    assert _is_backend_array(backend, gb)
    numpy.testing.assert_allclose(as_numpy(gb), ref_df._fE_interp(Es), rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ensure_fE_interp_forced_construction(backend):
    # DF CONSTRUCTED under a forced backend (the real harness case): _Emin/_potInf
    # are backend scalars, so the numpy interpolation-grid bounds must be pulled
    # numpy-side (else numpy_grid * tensor raises). Construction-outside (the test
    # above) leaves them numpy and misses this. The f(E) interp still matches
    # pure numpy.
    ref_df = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
    ref_df._ensure_fE_interp()
    with galpy.backend.use(backend, force=True):
        dfb = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
        dfb._ensure_fE_interp()
    Es = _egrid(ref_df)
    numpy.testing.assert_allclose(dfb._fE_interp(Es), ref_df._fE_interp(Es), rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_forced_construction(backend):
    # end-to-end sample() with the DF built under a forced backend: exercises the
    # _ensure_fE_interp grid-bound coercion through the public sampling entry.
    ref_df = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
    numpy.random.seed(321)
    ref = ref_df.sample(n=100, return_orbit=False)
    numpy.random.seed(321)
    with galpy.backend.use(backend, force=True):
        got = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3)).sample(
            n=100, return_orbit=False
        )
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_powerspherical_mass_array_fallback(backend):
    # PowerSpherical denspot: mass(denspot, array) raises under the backend in
    # _make_cmf_interpolator, so the CMF construction falls back to the per-r
    # backend loop (the except-RuntimeError branch). Samples stay numpy-side.
    pot = PowerSphericalPotential(amp=1.3, alpha=1.4)
    with galpy.backend.use(backend, force=True):
        numpy.random.seed(654)
        got = eddingtondf(pot=pot, rmax=5.0).sample(n=50, rmin=0.1, return_orbit=False)
    assert not _is_backend_array(backend, got[0])
    assert len(got[0]) == 50 and numpy.all(numpy.isfinite(got[0]))


# --------------------------------------------------------------------------
# d/d(potential parameter) through the DF construction.
#
# Building any spherical DF from a DIFFERENTIATED potential used to be blocked
# four times over, each a discrete test or a scalar-fill that cannot take a
# traced value:
#   1. _handle_rmin read a concrete Phi(0) (as_numpy) just to test divergence;
#   2. eddingtondf's _rInf did the same with numpy.isfinite to pick inf vs 1e12;
#   3. the _RphiRootFind guard tested under_trace ALONE, so eager torch autograd
#      fell through to the grid path and died on ndarray * Tensor; and
#   4. that root-find's bracket used xp.full(shape, r_lo), and r_lo is
#      r_a_min * scale -- a backend array once scale is differentiated, which
#      torch's full() rejects.
# Only (1) and (2) affect jax; (3) and (4) are torch-only, which is why this is
# parametrized over both rather than jax alone.
# --------------------------------------------------------------------------
_PGRAD_A0 = 1.3


def _edd_fE(a, cast, backend):
    from galpy.potential import HernquistPotential

    with galpy.backend.use(backend, force=True):
        df = eddingtondf(pot=HernquistPotential(amp=2.0, a=a))
        return df.fE(cast([-0.6]))[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_eddingtondf_fE_grad_wrt_potential_parameter(backend):
    h = 1e-5 * _PGRAD_A0
    fd = (
        float(_edd_fE(_PGRAD_A0 + h, numpy.array, "numpy"))
        - float(_edd_fE(_PGRAD_A0 - h, numpy.array, "numpy"))
    ) / (2.0 * h)
    if backend == "jax":
        ad = float(
            jax.grad(lambda t: _edd_fE(t, jnp.asarray, "jax"))(jnp.asarray(_PGRAD_A0))
        )
    else:
        t = torch.tensor(_PGRAD_A0, dtype=torch.float64, requires_grad=True)
        _edd_fE(
            t, lambda v: torch.as_tensor(numpy.asarray(v, dtype=float)), "torch"
        ).backward()
        ad = float(t.grad)
    assert numpy.isfinite(ad), "gradient must not be nan/inf"
    assert abs(ad) > 0.0, "gradient is identically zero (detached?)"
    # the DF construction runs a quadrature and a root-find, so this is not a
    # 1e-12 identity; 1e-5 still leaves ~400x margin on the observed ~2.3e-08
    numpy.testing.assert_allclose(ad, fd, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eddingtondf_grad_wrt_potential_parameter_without_backend_context(backend):
    # The traced potential parameter is the data: the constructor's Python-float
    # potential evaluations (Phi(0), Phi(rmax), ...) follow its backend with no
    # use() context (they used to resolve numpy and meet the traced parameter).
    from galpy.potential import HernquistPotential, NFWPotential

    def sigmar(a, r):
        df = eddingtondf(
            pot=NFWPotential(amp=1.0, a=a), denspot=HernquistPotential(amp=0.1, a=0.7)
        )
        return df.sigmar(r)

    if backend == "jax":
        ad = float(jax.grad(lambda a: sigmar(a, jnp.asarray(1.0)))(2.0))
    else:
        a = torch.tensor(2.0, requires_grad=True)
        ad = float(torch.autograd.grad(sigmar(a, torch.tensor(1.0)), a)[0])
    # central differences of the numpy build: O(h^2) down to h=1e-4 (measured
    # 1.4e-7 between 1e-3 and 1e-4), quadrature noise below; AD vs h=1e-4: 9e-10
    fd = [(sigmar(2.0 + h, 1.0) - sigmar(2.0 - h, 1.0)) / (2 * h) for h in (1e-3, 1e-4)]
    assert abs(fd[0] - fd[1]) < 3e-7 * abs(fd[1])
    numpy.testing.assert_allclose(ad, fd[1], rtol=2e-8)


# --- eddingtondf under jax.jit, differentiated w.r.t. the potential ------------
# Inside jit there is no concrete value: the boundary radius for the r -> inf
# limit is taken as 1e12 (which every standard profile already gets eagerly),
# and the r(Phi) inversion, sampling grids and f(E) table stay traced (#1558's
# machinery). Reference: EAGER construction under a gradient (same root-find).
# Measured jit vs eager-traced: fE exact, sigmar 1.7e-13, sampled v^2 7.7e-10;
# gradients <= 8e-10.
from galpy.backend import random as _grandom

_EJ_Q = {
    "fE": lambda d: d.fE(jnp.asarray([-0.9, -0.5, -0.1])),
    "sigmar": lambda d: d.sigmar(jnp.asarray(0.7)),
    "sample_v2": lambda d: sum(
        (x**2).sum()
        for i, x in enumerate(
            d.sample(n=3, key=_grandom.key(3, "jax"), return_orbit=False)
        )
        if i in (1, 2, 4)
    ),
}


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("which", list(_EJ_Q))
def test_eddingtondf_under_jit_matches_eager_traced(which):
    from galpy.potential import HernquistPotential

    def f(a):
        with galpy.backend.use("jax", force=True):
            d = eddingtondf(pot=HernquistPotential(amp=2.0, a=a), rmin=0.0)
            return jnp.sum(jnp.asarray(_EJ_Q[which](d)))

    v_jit, g_jit = float(jax.jit(f)(1.2)), float(jax.jit(jax.grad(f))(1.2))
    v_eager, g_eager = (float(x) for x in jax.jvp(f, (1.2,), (1.0,)))
    numpy.testing.assert_allclose(v_jit, v_eager, rtol=5e-9)
    numpy.testing.assert_allclose(g_jit, g_eager, rtol=5e-9)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_osipkovmerrittdf_sample_under_jit_matches_eager_traced():
    # the traced r(Phi) root-find reaches E -> Emin, so the f(Q) table is finite
    # at every knot under jit and sampling works there, value and gradient.
    # Measured: values and d/da <= 2.5e-8 (f(Q) ~ 1e13 at the end knots: an ulp
    # of E - Phi there, compiled vs eager, is ~1e-8 of f)
    from galpy.df import osipkovmerrittdf
    from galpy.potential import HernquistPotential

    def f(a):
        with galpy.backend.use("jax", force=True):
            d = osipkovmerrittdf(pot=HernquistPotential(amp=2.0, a=a), ra=1.5, rmin=0.0)
            out = d.sample(n=50, key=_grandom.key(3, "jax"), return_orbit=False)
            return jnp.stack([jnp.sum(x**2) for x in out[:5]])

    v_jit = numpy.asarray(jax.jit(f)(1.2))
    g_jit = numpy.asarray(jax.jit(jax.jacfwd(f))(1.2))
    v_eager, g_eager = (numpy.asarray(x) for x in jax.jvp(f, (1.2,), (1.0,)))
    assert numpy.all(g_eager != 0.0), "gradient disconnected"
    numpy.testing.assert_allclose(v_jit, v_eager, rtol=1e-7)
    numpy.testing.assert_allclose(g_jit, g_eager, rtol=1e-7)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
@pytest.mark.parametrize("profile", ["hernquist", "plummer"])
def test_traced_rphi_near_Emin(profile):
    # the traced r(Phi) (a root-find) reaches E -> Emin: its bracket starts
    # below the grid's first radius and it bisects in log r, so the root is
    # relative-exact down to r ~ 1e-8 a (it returned -1.2e12 there)
    from galpy.potential import (
        HernquistPotential,
        PlummerPotential,
        evaluatePotentials,
    )

    mk = {
        "hernquist": lambda a: HernquistPotential(amp=2.0, a=a),
        "plummer": lambda a: PlummerPotential(amp=2.0, b=a),
    }[profile]
    x = numpy.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-3])

    def rphi(a):
        with galpy.backend.use("jax", force=True):
            d = eddingtondf(pot=mk(a), rmin=0.0)
            E = d._Emin + jnp.asarray(x) * (d._potInf - d._Emin)
            return d._rphi(E), E, d._Emin * jnp.ones_like(E)

    from scipy import optimize

    got, Es, Emins = (numpy.asarray(v) for v in jax.jit(rphi)(1.2))
    pot = mk(1.2)
    for r, E, Emin, xx in zip(got, Es, Emins, x):
        # the residual is well-conditioned everywhere: Phi(r) = E to a few ulp
        assert r > 0.0 and numpy.isfinite(r), (xx, r)
        assert abs(evaluatePotentials(pot, r, 0.0) - E) <= 4e-16 * abs(E), (xx, r)
        # the radius to its conditioning: an ulp of E moves r by
        # ~eps |E| / |E - Emin| (x (1/2 for a core, whose Phi - Phi(0) ~ r^2))
        ref = optimize.brentq(
            lambda rr: evaluatePotentials(pot, rr, 0.0) - E,
            1e-20,
            1e7,
            xtol=1e-300,
            rtol=1e-15,
        )
        cond = 2.2e-16 * abs(E) / abs(E - Emin)
        assert abs(r / ref - 1.0) < 10.0 * cond + 1e-14, (xx, r, ref, cond)


# --- f(E) near Emin -------------------------------------------------------------
# NFW (amp=2.3, a=1.3) against a 40-digit mpmath Eddington integral at the exact
# double energies passed. The backend was NaN at 1e-8/1e-7 of the way from Emin
# and 7e-4 off at 1e-6: r(Phi) below the spline's first knot, and Phi(r) - E a
# difference of O(1) numbers. Now: Newton-refined r(Phi), and the small-r piece
# as 2/sqrt(mean dPhi/dr). Measured 5.9e-8 at 1e-8, <= 2.4e-10 above.
_EDD_NFW_GOLD = [  # (E, f(E))
    (-1.769230751559042, 1.4983431954507396e17),  # x = 1e-08
    (-1.769229002058064, 1498343239910.5228),  # x = 1e-06
    (-1.7690540519602862, 14983432.383219456),  # x = 0.0001
    (-1.7515590421824891, 149.81271594972208),  # x = 0.01
    (-1.2390789577823735, 0.02035422714067017),  # x = 0.3
]


@pytest.mark.parametrize("E,fref", _EDD_NFW_GOLD)
@pytest.mark.parametrize("backend", BACKENDS)
def test_nfw_fE_near_Emin(backend, E, fref):
    with galpy.backend.use(backend, force=True):
        d = eddingtondf(pot=NFWPotential(amp=2.3, a=1.3))
        got = float(as_numpy(d.fE(_arr(backend, [E])))[0])
    tol = 1e-7 if E < -1.7692 else 1e-9
    assert abs(got / fref - 1.0) < tol, f"E={E}: {got} vs {fref}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_nfw_fE_near_Emin_grad_wrt_potential(backend):
    # d f(E)/d a at fixed (E - Emin)/(Einf - Emin) = 1e-6, through the Newton
    # refinement of r(Phi): AD vs a Richardson central difference of the same
    # backend values
    def f(a):
        with galpy.backend.use(backend, force=True):
            d = eddingtondf(pot=NFWPotential(amp=2.3, a=a))
            E = d._Emin + 1e-6 * (d._potInf - d._Emin)
            return d.fE(E * _arr(backend, [1.0]))[0]

    if backend == "jax":
        ad = float(jax.grad(f)(1.3))
    else:
        a = torch.tensor(1.3, requires_grad=True)
        (g,) = torch.autograd.grad(f(a), a)
        ad = float(g)

    def cd(h):
        return (float(as_numpy(f(1.3 + h))) - float(as_numpy(f(1.3 - h)))) / (2 * h)

    # h^2 truncation dominates down to h ~ 1e-3; below, the values' ~1e-10
    # quadrature noise does. Richardson on (2e-3, 1e-3): measured jax 1.4e-8,
    # torch 2.1e-7 (the reference's noise floor, not the AD)
    fd = (4.0 * cd(1e-3) - cd(2e-3)) / 3.0
    numpy.testing.assert_allclose(ad, fd, rtol=5e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_at_near_Emin_and_rmax_inf(backend):
    # the endpoint E = Emin (inf for a cusp, finite for a core), the integrand's
    # bulk at r ~ scale when rphi << scale (log r, not 1/r out to 1/(2 rphi):
    # Plummer was 91% off at 1e-12 of the energy range, 8% at 1e-8), and
    # rmax = inf; against the analytic DFs
    from galpy.df import isotropicHernquistdf, isotropicPlummerdf
    from galpy.potential import HernquistPotential, PlummerPotential

    with galpy.backend.use(backend, force=True):
        dfh = eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3))
        Emin = float(as_numpy(dfh._Emin))
        assert as_numpy(dfh.fE(_arr(backend, [Emin])))[0] == numpy.inf
        E = numpy.array([-0.8, -0.3, -0.05])
        got = dfh.fE(_arr(backend, E))
        assert _is_backend_array(backend, got)
        got = as_numpy(
            eddingtondf(pot=HernquistPotential(amp=2.3, a=1.3), rmax=numpy.inf).fE(
                _arr(backend, E)
            )
        )
    ref = isotropicHernquistdf(pot=HernquistPotential(amp=2.3, a=1.3)).fE(E)
    numpy.testing.assert_allclose(got, ref, rtol=1e-13)
    pot = PlummerPotential(amp=2.3, b=1.3)
    with galpy.backend.use(backend, force=True):
        dfp = eddingtondf(pot=pot)
        Emin, Einf = float(as_numpy(dfp._Emin)), float(as_numpy(dfp._potInf))
        E = Emin + numpy.array([0.0, 1e-12, 1e-8, 1e-4]) * (Einf - Emin)
        got = as_numpy(dfp.fE(_arr(backend, E)))
    ref = isotropicPlummerdf(pot=pot).fE(E)
    numpy.testing.assert_allclose(got[0], ref[0], rtol=2e-8)  # fixed GL at Emin
    numpy.testing.assert_allclose(got[1:], ref[1:], rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_plummer_fE_near_Emin_grad_wrt_b(backend):
    # f(E) = 24 sqrt(2)/(7 pi^3) b^2 (-E)^(7/2) / (G M)^5 for Plummer, so at
    # fixed E d f/d b = 2 f/b exactly; E 1e-6 of the range above Emin (44x
    # off before; 5e-9 now, 3e-5 at 1e-8)
    from galpy.potential import PlummerPotential

    E0 = -2.3 / 1.3 * (1.0 - 1e-6)

    def f(b):
        with galpy.backend.use(backend, force=True):
            return eddingtondf(pot=PlummerPotential(amp=2.3, b=b)).fE(
                E0 * _arr(backend, [1.0])
            )[0]

    if backend == "jax":
        ad = float(jax.grad(f)(1.3))
    else:
        b = torch.tensor(1.3, requires_grad=True)
        (g,) = torch.autograd.grad(f(b), b)
        ad = float(g)
    val = float(as_numpy(f(1.3)))
    # 1e-6 above Emin amplifies last-digit differences: CUDA lands at 2.9e-8
    on_gpu = (
        jax.default_backend() == "gpu"
        if backend == "jax"
        else torch.get_default_device().type == "cuda"
    )
    numpy.testing.assert_allclose(ad, 2.0 * val / 1.3, rtol=5e-8 if on_gpu else 2e-8)
