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

import galpy.backend
from galpy.backend import as_numpy
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
def test_hyp2f1_domain_limit(backend):
    # The shared galpy.backend.special.hyp2f1 fallback (Euler labeling requires
    # a positive parameter B with c-B>=1) cannot reach the general-beta
    # Hernquist DF for beta>=0.5 (b=1-2beta<=0 after the Pfaff transform); the
    # numpy path (scipy) is unaffected. This locks that documented boundary.
    dfh = constantbetaHernquistdf(pot=_HP, beta=0.7)
    assert numpy.isfinite(dfh.fE(numpy.array([-0.5 * _PSI0]))[0])  # numpy OK
    with pytest.raises(NotImplementedError):
        dfh.fE(_arr(backend, numpy.array([-0.5 * _PSI0])))


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_constantbetadf_fE_backend(backend):
    # The generic constantbetadf (no closed form): the fE inversion integral
    # (non-halfint beta) and the halfint derivative branch, via the backend GL
    # fixed_quad path (_fE_backend / _deriv / _gradfunc_for). The DF is built
    # *under the forced backend*, so this also exercises the backend
    # construction branches (_active_autodiff / _make_gradfunc / _make_func, the
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


def _mk_hern(beta):
    return constantbetaHernquistdf(pot=_HP, beta=beta)


def _mk_pl(beta):
    return constantbetaPowerLawdf(pot=_PP, beta=beta, rmax=100.0, rmin=1e-4)
