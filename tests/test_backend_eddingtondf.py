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
def test_osipkovmerrittdf_sample_under_jit_raises_clearly():
    # f(Q) is non-finite at knots near Emin (numpy's too; eager drops them), so
    # a fixed-shape jit table would differ there -- refuse rather than differ
    from galpy.df import osipkovmerrittdf
    from galpy.potential import HernquistPotential

    def f(a):
        with galpy.backend.use("jax", force=True):
            d = osipkovmerrittdf(pot=HernquistPotential(amp=2.0, a=a), ra=1.5, rmin=0.0)
            return d.sample(n=2, key=_grandom.key(3, "jax"), return_orbit=False)[0]

    with pytest.raises(NotImplementedError, match="under jax.jit"):
        jax.jit(f)(1.2)
