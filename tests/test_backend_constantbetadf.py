###############################################################################
# test_backend_constantbetadf.py: Track F Pdf.2 -- backend (jax/torch)
# coverage for the constant-beta spherical-DF family: the closed-form
# constantbetaHernquistdf (algebraic beta=0,+-0.5 branches + the general-beta
# hyp2f1 branch, routed through galpy.backend.special via a Pfaff transform so
# the z<=0 fallback covers the DF's z=Etilde in [0,1] regime) and
# constantbetaPowerLawdf, plus the shared anisotropic _constantbetadf base
# machinery (_vmomentdensity, _dMdE, _p_v_at_r). The numpy path is
# byte-identical (test_sphericaldf unchanged); this exercises the
# resolved-namespace dispatch: parity numpy<->jax<->torch of fE / __call__ /
# moments / dM/dE, grad-vs-FD, and the numpy-side sampling contract.
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

from backend_jit_helpers import assert_jit_matches_eager

import galpy.backend
from galpy.backend import as_numpy, use
from galpy.df import constantbetadf, constantbetaHernquistdf, constantbetaPowerLawdf
from galpy.potential import (
    DehnenCoreSphericalPotential,
    HernquistPotential,
    PowerSphericalPotential,
)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


_HP = HernquistPotential(amp=2.3, a=1.3)
_PSI0 = float(-_HP(0, 0, use_physical=False))
_PP = PowerSphericalPotential(amp=1.0, alpha=2.5)
# smooth, finite core -> exercises the generic constantbetadf fE inversion
_DC = DehnenCoreSphericalPotential(amp=2.5, a=1.15)

# beta with closed-form (algebraic) fE and beta needing the hyp2f1 branch
_ALG_BETAS = [-0.5, 0.0, 0.5]
_HYP_BETAS = [-0.4, 0.3]

# in-bounds E grid (Etilde in (0,1)) + out-of-bounds points (E>0, E<-psi0). E==0
# is excluded: for beta=0 the numpy arcsin term is 0/0 -> NaN there (a known
# 1-D numpy edge), which the backend maps to the correct fE->0 limit (tested
# separately in test_fE_zero_edge)
_EGRID = numpy.concatenate(
    [numpy.linspace(0.99 * _HP(0, 0), -1e-4, 21), [0.5, -1.5 * _PSI0]]
)
_ENEG = numpy.linspace(-0.95 * _PSI0, -0.05 * _PSI0, 11)
_RS = numpy.array([0.13, 0.5, 1.3, 5.2, 13.0])
# power-law: relative energy eps=-E>0 is bound; out-of-bounds is E>=0
_EGRIDP = numpy.concatenate([numpy.linspace(-10.0, -0.1, 21), [0.5, 0.0, 1e-8]])
_RSP = numpy.array([0.5, 1.3, 5.0, 20.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_parity_hernquist(backend):
    # algebraic branches (beta=0,+-0.5) are exact closed forms; the general-beta
    # branch goes through the Pfaff-transformed backend hyp2f1 fallback, whose
    # floor is ~3e-7 over Etilde in [0,1] (validated vs scipy)
    for beta in _ALG_BETAS:
        dfh = constantbetaHernquistdf(pot=_HP, beta=beta)
        ref = dfh.fE(_EGRID)
        got = dfh.fE(_arr(backend, _EGRID))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    for beta in _HYP_BETAS:
        dfh = constantbetaHernquistdf(pot=_HP, beta=beta)
        ref = dfh.fE(_EGRID)
        got = dfh.fE(_arr(backend, _EGRID))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_parity_powerlaw(backend):
    # closed-form power-law fE = eta * (-E)^n; exact on the backend
    for beta in [-0.5, 0.0, 0.3, 0.5]:
        dfp = constantbetaPowerLawdf(pot=_PP, beta=beta, rmax=100.0, rmin=1e-4)
        ref = dfp.fE(_EGRIDP)
        got = dfp.fE(_arr(backend, _EGRIDP))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    # scalar (0-d) input: numpy returns out[0]; backend returns a 0-d array
    dfp = constantbetaPowerLawdf(pot=_PP, beta=0.3, rmax=100.0, rmin=1e-4)
    got = dfp.fE(_arr(backend, -1.0))
    assert float(as_numpy(got)) == pytest.approx(float(dfp.fE(-1.0)), rel=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_zero_edge(backend):
    # E == 0 (Etilde == 0) is 0/0 in the numpy arcsin term for beta=0 (NaN in
    # 1-D numpy); the backend branch implements the correct fE -> 0 limit,
    # NaN-free (special-fn edge testing)
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.0)
    got = as_numpy(dfh.fE(_arr(backend, numpy.array([0.0, -0.0, -1e-300]))))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_outofbounds(backend):
    # out-of-bounds E -> exactly 0 on the dead branch (functional masking)
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.3)
    oob = as_numpy(dfh.fE(_arr(backend, numpy.array([0.5, 1.0, -1.5 * _PSI0, 0.0]))))
    assert numpy.all(oob == 0.0)
    dfp = constantbetaPowerLawdf(pot=_PP, beta=0.3, rmax=100.0, rmin=1e-4)
    oobp = as_numpy(dfp.fE(_arr(backend, numpy.array([0.5, 0.0, 1e-8]))))
    assert numpy.all(oobp == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # anisotropic __call__: (E, L) tuple form (f = L^{-2beta} fE) and the
    # 6-coordinate form
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.3)
    E = numpy.linspace(0.99 * _HP(0, 0), -1e-4, 8)
    L = numpy.linspace(0.2, 1.5, 8)
    ref = dfh((E, L))
    got = dfh((_arr(backend, E), _arr(backend, L)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)
    R = numpy.array([0.5, 1.1, 1.7, 2.9])
    vR = numpy.array([0.1, -0.2, 0.3, 0.05])
    vT = numpy.array([0.3, 0.5, 0.2, 0.4])
    z = numpy.array([0.2, -0.3, 0.5, 0.1])
    vz = numpy.array([-0.1, 0.2, 0.05, 0.1])
    ref6 = dfh(R, vR, vT, z, vz, numpy.zeros_like(R))
    got6 = dfh(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got6)
    numpy.testing.assert_allclose(as_numpy(got6), ref6, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moments_parity(backend):
    # sigmar/sigmat (GL-vs-adaptive quadrature over v) and beta(r); the
    # anisotropic constant-beta base _vmomentdensity. beta=0 hits the GL floor
    # (~1e-9), the hyp2f1-fE betas ~1e-8 (measured); scalar + vector r
    for pot, betas, rs, ctor in (
        (_HP, [-0.5, 0.0, 0.3, 0.5], _RS, _mk_hern),
        (_PP, [0.3], _RSP, _mk_pl),
    ):
        for beta in betas:
            df = ctor(beta)
            for name in ("sigmar", "sigmat"):
                f = getattr(df, name)
                ref = numpy.array([f(r) for r in rs])
                got = numpy.array([float(f(_arr(backend, r))) for r in rs])
                gotv = f(_arr(backend, rs))
                assert _is_backend_array(backend, gotv)
                numpy.testing.assert_allclose(got, ref, rtol=1e-7)
                numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-7)
            refb = numpy.array([df.beta(r) for r in rs])
            gotb = numpy.array([float(as_numpy(df.beta(_arr(backend, r)))) for r in rs])
            numpy.testing.assert_allclose(gotb, refb, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vmomentdensity_divergent_potential(backend):
    # a divergent potential (alpha<2 -> Phi(inf)=inf -> v_esc=inf) drives the base
    # _vmomentdensity onto the fixed_quad_semiinfinite branch (integrate the v_esc=inf
    # tail via a reciprocal map instead of to an infinite fixed_quad limit -> NaN)
    # (correctness of the divergent case is covered by test_sphericaldf's
    # test_constantbeta_differentpotentials_dens_directint; here we just need the
    # backend semiinfinite branch to run and return a finite, positive density.)
    df = constantbetadf(pot=PowerSphericalPotential(amp=1.3, alpha=1.9), twobeta=-1)
    got = df.vmomentdensity(_arr(backend, 1.3), 0, 0)
    assert _is_backend_array(backend, got)
    val = float(as_numpy(got))
    assert numpy.isfinite(val) and val > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # anisotropic constant-beta dM/dE (GL after the r = rphi - s^2 turning-point
    # substitution). rtol 1e-5 is the numpy adaptive-quad floor at the
    # (E-Phi)^{1/2-beta} endpoint (measured ~1.3e-6 at beta=-0.4, ~4.9e-7 at
    # beta=0) plus the hyp2f1-fE floor; the algebraic betas hit ~1e-15
    for beta in [-0.5, -0.4, 0.0, 0.3, 0.5]:
        dfh = constantbetaHernquistdf(pot=_HP, beta=beta)
        ref = dfh.dMdE(_ENEG)
        got = dfh.dMdE(_arr(backend, _ENEG))
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-5)
    # out-of-bounds E -> exactly zero on the dead branch
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.3)
    assert numpy.all(as_numpy(dfh.dMdE(_arr(backend, numpy.array([0.5])))) == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_grad_vs_fd(backend):
    # d fE/dE through the closed-form and hyp2f1 branches; out-of-bounds grad
    # is a finite 0 (dead-branch guards)
    for pot, beta, E0, ctor in (
        (_HP, -0.5, -0.5 * _PSI0, _mk_hern),
        (_HP, 0.3, -0.5 * _PSI0, _mk_hern),
        (_PP, 0.3, -2.0, _mk_pl),
    ):
        df = ctor(beta)
        eps = 1e-6
        fd = (
            df.fE(numpy.atleast_1d(E0 + eps))[0] - df.fE(numpy.atleast_1d(E0 - eps))[0]
        ) / (2.0 * eps)
        if backend == "jax":
            g = float(jax.grad(lambda E: df.fE(E))(jnp.asarray(E0)))
            goob = float(jax.grad(lambda E: df.fE(E))(jnp.asarray(0.5)))
        else:
            t = torch.tensor(E0, requires_grad=True)
            df.fE(t).backward()
            g = float(t.grad)
            t = torch.tensor(0.5, requires_grad=True)
            df.fE(t).backward()
            goob = float(t.grad)
        numpy.testing.assert_allclose(g, fd, rtol=1e-6)
        assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_grad_vs_fd(backend):
    # d(sigma_r)/dr through the GL moment integrals (limits + Phi(r) + fE)
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.3)
    r0, eps = 1.3, 1e-5
    fd = (dfh.sigmar(r0 + eps) - dfh.sigmar(r0 - eps)) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: dfh.sigmar(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        dfh.sigmar(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_numpy_side_forced(backend):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and the outputs are numpy arrays; only the
    # deterministic sub-steps (fE/vesc/pvr grids, closed-form icmf) run on the
    # backend, so draws match the pure-numpy ones to the grids' fp noise
    # (Hernquist's pvr grid inherits the hyp2f1 floor -> looser tol)
    for ctor, kw, rtol, atol in (
        (constantbetaHernquistdf, dict(pot=_HP, beta=0.3), 1e-6, 1e-8),
        (
            constantbetaPowerLawdf,
            dict(pot=_PP, beta=0.3, rmax=100.0, rmin=1e-4),
            1e-9,
            1e-11,
        ),
    ):
        ref_df = ctor(**kw)
        numpy.random.seed(10)
        ref = ref_df.sample(n=100, return_orbit=False)
        dfb = ctor(**kw)
        numpy.random.seed(10)
        with galpy.backend.use(backend, force=True):
            got = dfb.sample(n=100, return_orbit=False)
        for g, r in zip(got, ref):
            assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
            numpy.testing.assert_allclose(g, r, rtol=rtol, atol=atol)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("beta", [0.5, 0.7, 0.9])
def test_general_beta_hernquist_past_the_old_hyp2f1_limit(backend, beta):
    # beta >= 0.5 gives b = 1-2beta <= 0, so BOTH 2F1 parameters are
    # non-positive and no Euler labeling exists. The fallback used to raise
    # NotImplementedError here, which made this whole DF family numpy-only; it
    # now takes the Pfaff series route (see the hyp2f1 fallback). scipy on the
    # numpy path is the reference, and the agreement is at double precision --
    # the series is exact at the |z| these DFs evaluate at -- so this is a tight
    # bound, not a smoke check.
    dfh = constantbetaHernquistdf(pot=_HP, beta=beta)
    Es = numpy.array([-0.8, -0.5, -0.3]) * _PSI0
    ref = dfh.fE(Es)
    assert numpy.all(numpy.isfinite(ref))
    got = as_numpy(dfh.fE(_arr(backend, Es)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_constantbetadf_fE_backend(backend):
    # The generic constantbetadf (no closed form): the fE inversion integral
    # (non-halfint beta) and the halfint derivative branch, via the backend GL
    # fixed_quad path (_fE_backend / _deriv / _raw_gradfunc). The DF is built
    # *under the forced backend*, so this also exercises the backend
    # construction branches (_autodiff_xp / _make_gradfunc / _make_func, the
    # backend _potInf/_Emin, and the _evalpot_asnumpy startt calibration). The
    # scipy-adaptive numpy path on the same DF is the reference; the fixed-order
    # GL floor is ~1e-5 for the integral inversion, ~1e-6 for the halfint case.
    for kw, rtol in ((dict(beta=0.25), 5e-5), (dict(twobeta=-1), 1e-5)):
        with galpy.backend.use(backend, force=True):
            df = constantbetadf(pot=_DC, **kw)
            Emin, pinf = float(df._Emin), float(df._potInf)
            Es = Emin + numpy.linspace(0.1, 0.9, 7) * (pinf - Emin)
            got = df.fE(_arr(backend, Es))
        assert _is_backend_array(backend, got)
        ref = df.fE(Es)  # numpy default context -> scipy-adaptive, same DF
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=rtol)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_generic_fE_jit_traceable():
    # The generic constantbetadf reads its fE integration limits off two frozen
    # interpolators, r(Phi) and log10(startt). Both used to be queried with a
    # concrete numpy energy (as_numpy of the clamped E), which made fE
    # untraceable: under jax.jit that clamped energy is a tracer and the
    # conversion raises. Both are Spline1D now, so the clamp stays on-backend.
    # Covers both branches -- twobeta=-1 (half-integer, queries only r(Phi)) and
    # beta=0.25 (the inversion integral, which also queries log10(startt)) --
    # and the out-of-bounds clamp, which is the line that used to force numpy.
    for kw in (dict(twobeta=-1), dict(beta=0.25)):
        with galpy.backend.use("jax", force=True):
            df = constantbetadf(pot=_DC, **kw)
            Emin, pinf = float(df._Emin), float(df._potInf)
            Es = numpy.concatenate(
                [
                    Emin + numpy.linspace(0.1, 0.9, 7) * (pinf - Emin),
                    [pinf + 1.0, Emin - 1.0],  # out of bounds -> exactly zero
                ]
            )
            # tracing must not perturb the value: same kernel, same GL nodes
            traced = assert_jit_matches_eager(
                df.fE, jnp.asarray(Es), rtol=1e-12, atol=0.0
            )
        assert numpy.all(traced[-2:] == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_autodiff_ops_dispatch(backend):
    # autodiff_ops returns the functional (grad, vmap) pair for the backend's
    # namespace and differentiates a simple scalar function correctly; numpy has
    # no autodiff and raises. This is the engine the generic constantbetadf fE
    # derivative chain is built with.
    from galpy.backend import autodiff_ops

    if backend == "jax":
        grad, vmap = autodiff_ops(jnp)
        assert grad is jax.grad and vmap is jax.vmap
        g = vmap(grad(lambda x: x**3))(jnp.asarray([1.0, 2.0]))
        numpy.testing.assert_allclose(as_numpy(g), [3.0, 12.0])
    else:
        grad, vmap = autodiff_ops(torch)
        assert grad is torch.func.grad and vmap is torch.func.vmap
        g = vmap(grad(lambda x: x**3))(torch.tensor([1.0, 2.0]))
        numpy.testing.assert_allclose(as_numpy(g), [3.0, 12.0])
    with pytest.raises(ValueError):
        autodiff_ops(numpy)


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_constantbetadf_cutoff_force(backend):
    # constantbetadf on PowerSphericalPotentialwCutoff: its _Rforce re-coerces
    # coords, so the evaluateRforces divisor is coerced twice under the fE
    # derivative chain's vmap -- which exposes the vmap-tracer's
    # SingleDeviceSharding device to asarray_on_device. Exercises that
    # backend-native force path (fE finite on jax/torch).
    from galpy.potential import PowerSphericalPotentialwCutoff

    pot = PowerSphericalPotentialwCutoff(amp=1.1, alpha=1.4, rc=2.0)
    with galpy.backend.use(backend, force=True):
        df = constantbetadf(pot=pot, beta=0.0)
        Emin, pinf = float(df._Emin), float(df._potInf)
        Es = Emin + numpy.linspace(0.15, 0.9, 5) * (pinf - Emin)
        got = as_numpy(df.fE(_arr(backend, Es)))
    assert numpy.all(numpy.isfinite(got))
    assert numpy.all(got[got != 0.0] > 0.0)


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch-only fallback needs torch")
def test_torch_only_install_autodiff_fallback(monkeypatch):
    # On a torch-only install (no jax) the numpy-eval fE derivative chain is built
    # with torch autodiff instead of jax: _autodiff_xp() returns the torch namespace
    # and _make_gradfunc wraps a torch-vmapped closure so it takes/returns numpy.
    # Force _JAX_LOADED=False so both torch branches run on the normal (jax+torch)
    # CI (the end-to-end torch-built DF is covered by the --backend torch tests).
    import sys

    cbd = sys.modules["galpy.df.constantbetadf"]
    monkeypatch.setattr(cbd, "_JAX_LOADED", False)
    assert cbd._autodiff_xp() is torch
    # _make_gradfunc's torch branch: numpy in -> torch grad -> numpy out.
    vmapped = torch.func.vmap(torch.func.grad(lambda x: x**3))
    gradfunc = cbd._make_gradfunc(vmapped, "torch")
    out = gradfunc(numpy.array([1.0, 2.0]))
    assert isinstance(out, numpy.ndarray)
    numpy.testing.assert_allclose(out, [3.0, 12.0])  # d(x^3)/dx = 3 x^2


def _mk_hern(beta):
    return constantbetaHernquistdf(pot=_HP, beta=beta)


def _mk_pl(beta):
    return constantbetaPowerLawdf(pot=_PP, beta=beta, rmax=100.0, rmin=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "mk,E0,name",
    [(_mk_hern, -0.1, "Hernquist"), (_mk_pl, -1.0, "PowerLaw")],
    ids=["Hernquist", "PowerLaw"],
)
def test_fE_differentiable_in_beta(backend, mk, E0, name):
    # d f_E / d beta -- differentiating the DF w.r.t. its own anisotropy
    # parameter, which is the point of having a constant-beta family.
    #
    # The Hernquist variant could not do this: its normalisation calls
    # scipy.special.gamma on beta, which raises TracerArrayConversionError on a
    # jax tracer and "Can't call numpy() on Tensor that requires grad" on torch.
    # The PowerLaw sibling was already routed, which is why only one of the two
    # worked. Both are checked here so the pair cannot drift apart again.
    #
    # E0 is chosen INSIDE the bound range: outside it f_E is identically 0 for
    # every beta, so the derivative is 0 too and the test would pass while
    # measuring nothing.
    b0, h = 0.3, 1e-6
    val = float(numpy.atleast_1d(mk(b0).fE(numpy.atleast_1d(E0)))[0])
    assert abs(val) > 1e-12, f"{name}: f_E is 0 at E={E0}; pick a bound energy"
    ref = (
        float(numpy.atleast_1d(mk(b0 + h).fE(numpy.atleast_1d(E0)))[0])
        - float(numpy.atleast_1d(mk(b0 - h).fE(numpy.atleast_1d(E0)))[0])
    ) / (2.0 * h)

    with use(backend, force=True):
        if backend == "jax":
            import jax
            import jax.numpy as jnp

            got = float(jax.grad(lambda b: mk(b).fE(jnp.asarray(E0)))(b0))
        else:
            import torch

            b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
            out = mk(b).fE(torch.tensor(E0, dtype=torch.float64))
            (grad,) = torch.autograd.grad(out, b)
            got = float(grad)
    # h=1e-6 central differences on a smooth function: good to ~1e-9
    assert got == pytest.approx(ref, rel=1e-6), f"{name} d f_E/d beta"


# (beta, rtol) -- each bar is ~10x the MEASURED numpy-vs-jit agreement, and the
# spread across betas is not noise: under a trace every beta takes the general
# hyp2f1 branch, and the fallback's accuracy depends on the (A, B, c) that
# branch happens to request. beta=0 and 0.3 land on the Gauss-Legendre route
# (1e-9..1e-10); the others land on parameters it handles exactly.
_JIT_BETAS = [
    (0.3, 1e-9),  # -> B=0.4, GL route; measured 7.8e-11
    (0.0, 1e-8),  # -> B=1.0, GL route; measured 6.8e-10
    (0.5, 1e-13),  # measured 1.3e-15
    (-0.5, 1e-13),  # measured 5.7e-15
    (-1.5, 1e-12),  # measured 2.3e-14
]


@pytest.mark.skipif(jax is None, reason="jax not installed")
@pytest.mark.parametrize("beta,rtol", _JIT_BETAS, ids=[str(b) for b, _ in _JIT_BETAS])
def test_fE_traceable_in_beta_under_external_jit(beta, rtol):
    # d f_E / d beta works eagerly, but the EXTERNAL-jit contract is stronger:
    # constructing and evaluating the DF with a dynamic beta inside jax.jit used
    # to raise TracerBoolConversionError at ``if self._beta == 0.0``. Those
    # exact-beta closed forms cannot be selected from a tracer, so they now go
    # through concretely_true and a traced beta falls through to the general
    # hyp2f1 branch -- of which they are all special cases.
    #
    # beta = 0, +-0.5 are parametrized precisely BECAUSE they are the branches
    # being skipped: they are where falling through could silently give a
    # different number, and they are checked against the numpy path, which does
    # take the closed form.
    with use("jax", force=True):
        got = jax.jit(lambda b: _mk_hern(b).fE(jnp.asarray(-0.1)))(jnp.asarray(beta))
    want = float(numpy.atleast_1d(_mk_hern(beta).fE(numpy.atleast_1d(-0.1)))[0])
    numpy.testing.assert_allclose(float(numpy.ravel(as_numpy(got))[0]), want, rtol=rtol)


# NOT TESTED HERE: d f_E/d beta THROUGH an external jit, i.e.
# jax.jit(jax.grad(...)). It is a real contract and it does work, but the trace
# costs >25 MINUTES to compile on this DF -- measured, after a CI runner died
# mid-test and made me time what I had already shipped. A 25-minute test does
# not belong in a shard, and the two halves of what it covered are covered
# cheaply and separately: the value under jit by the parametrized test above
# (~3 s per beta), and the gradient by test_fE_differentiable_in_beta (eager
# grad vs finite differences, both DF classes, both backends). What is left
# uncovered is specifically grad-composed-with-jit; closing it needs the
# compile cost brought down first, not a slower test.
