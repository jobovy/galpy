###############################################################################
# test_backend_osipkovmerrittdf.py: Track F Pdf.2 -- backend (jax/torch)
# coverage for the Osipkov-Merritt anisotropic-DF family (the general
# _osipkovmerrittdf superclass machinery + the three closed-form variants
# Hernquist/NFW/PowerLaw). The numpy path is byte-identical (test_sphericaldf
# unchanged); this exercises the resolved-namespace dispatch: parity
# numpy<->jax<->torch of fQ / __call__ / moments / dM/dE, grad-vs-FD, and the
# numpy-side sampling contract (numpy RNG draws unchanged under a forced
# backend). The general osipkovmerrittdf.fQ routes through the (not-yet-
# migrated) eddingtondf.fE, so only the closed-form variants are exercised on a
# backend here.
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
from galpy.df import (
    osipkovmerrittHernquistdf,
    osipkovmerrittNFWdf,
    osipkovmerrittPowerLawdf,
)
from galpy.potential import (
    HernquistPotential,
    NFWPotential,
    PowerSphericalPotential,
)


def _arr(backend, x):
    return (
        jnp.asarray(x)
        if backend == "jax"
        else torch.tensor(numpy.asarray(x, dtype=float))
    )


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


# --- family instances --------------------------------------------------------
_HP = HernquistPotential(amp=2.3, a=1.3)
_DFH = osipkovmerrittHernquistdf(pot=_HP, ra=2.3)
_PSI0 = float(_DFH._psi0)

_NP = NFWPotential(amp=2.3, a=1.3)
_DFN = osipkovmerrittNFWdf(pot=_NP, ra=2.3)
_QM = float(_DFN._Qtildemax)

_PP = PowerSphericalPotential(amp=1.0, alpha=2.5)
# non-self-consistent tracer (gamma=2.8 > alpha=2.5) has n2=+0.1: no Q^{-0.5}
# endpoint singularity, so the velocity-moment GL is tight (galpy's own
# test_osipkovmerritt_powerlaw_sigmar uses this case for the same reason)
_DPN = PowerSphericalPotential(amp=1.0, alpha=2.8)
_DFP = osipkovmerrittPowerLawdf(pot=_PP, denspot=_DPN, ra=2.0, rmax=100.0, rmin=1e-4)
# self-consistent (gamma=alpha=2.5) has n2=-0.5 -> integrable Q^{-0.5}
# endpoint singularity in the velocity moment (documented floor below)
_DFPS = osipkovmerrittPowerLawdf(pot=_PP, ra=2.0, rmax=100.0, rmin=1e-4)

# interior Q grids (closed-form regime, away from the boundary edges)
_QH = numpy.linspace(0.02, 0.98, 25) * _PSI0
_QN = numpy.linspace(0.02, 0.98, 25) * _QM
_QPL = numpy.linspace(0.05, 5.0, 25)

# (df, Qgrid, radii, interior Q0) for the three variants
_VARIANTS = {
    "hern": (_DFH, _QH, [0.2, 0.5, 1.3, 3.0, 7.0], 0.5 * _PSI0),
    "nfw": (_DFN, _QN, [0.2, 0.5, 1.3, 3.0, 7.0], 0.5 * _QM),
    "plaw": (_DFP, _QPL, [0.5, 1.0, 3.0, 8.0, 20.0], 1.3),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tag", list(_VARIANTS))
def test_fQ_parity(backend, tag):
    # closed-form f(Q): numpy<->backend parity on the interior grid
    df, Qgrid, _, _ = _VARIANTS[tag]
    ref = df.fQ(Qgrid)
    got = df.fQ(_arr(backend, Qgrid))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fQ_out_of_bounds(backend):
    # the Q<0 / Q>Qmax / Q==0 dead branches return exactly zero, NaN-free
    # (functional dummy-then-zero); Q==0 is the arcsin 0/0 edge for Hernquist
    for df, hi in ((_DFH, _PSI0), (_DFN, _QM)):
        got = as_numpy(df.fQ(_arr(backend, numpy.array([-1.0, -1e-8, 0.0, 1.2 * hi]))))
        assert numpy.all(got == 0.0)
    # PowerLaw is dead for Q<=0 only
    gotp = as_numpy(_DFP.fQ(_arr(backend, numpy.array([-1.0, -1e-8, 0.0]))))
    assert numpy.all(gotp == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tag", list(_VARIANTS))
def test_call_parity(backend, tag):
    # __call__ (E, L) tuple and the 6-coordinate form
    df, Qgrid, _, _ = _VARIANTS[tag]
    Eg = -Qgrid[:15] * 0.8
    Lg = numpy.linspace(0.05, 0.4, 15)
    ref = df((Eg, Lg))
    got = df((_arr(backend, Eg), _arr(backend, Lg)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    R = numpy.array([0.5, 1.1, 1.7, 2.9, 0.8])
    vR = numpy.array([0.1, -0.2, 0.3, 0.0, 0.15])
    vT = numpy.array([0.3, 0.5, 0.2, 0.4, 0.1])
    z = numpy.array([0.2, -0.3, 0.5, 0.0, 0.1])
    vz = numpy.array([-0.1, 0.2, 0.0, 0.1, -0.05])
    ref = df(R, vR, vT, z, vz, numpy.zeros_like(R))
    got = df(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tag", list(_VARIANTS))
def test_moments_parity(backend, tag):
    # sigmar/sigmat (fixed-order GL over v vs adaptive scipy; measured 4e-7
    # Hernquist / 4e-8 NFW / 2.9e-6 PowerLaw-nonself) and the analytic OM beta
    # = 1/(1+ra^2/r^2) (the velocity integral cancels in the ratio). numpy is a
    # scalar loop (integrate.quad); backend is vectorized (fixed_quad)
    df, _, rs, _ = _VARIANTS[tag]
    rsa = numpy.array(rs)
    for name in ("sigmar", "sigmat"):
        f = getattr(df, name)
        ref = numpy.array([f(r, use_physical=False) for r in rs])
        gots = numpy.array(
            [as_numpy(f(_arr(backend, r), use_physical=False)) for r in rs]
        )
        gotv = f(_arr(backend, rsa), use_physical=False)
        assert _is_backend_array(backend, gotv)
        numpy.testing.assert_allclose(gots, ref, rtol=1e-5)
        numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-5)
    refb = numpy.array([df.beta(r) for r in rs])
    gotb = df.beta(_arr(backend, rsa))
    assert _is_backend_array(backend, gotb)
    numpy.testing.assert_allclose(as_numpy(gotb), refb, rtol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_powerlaw_selfconsistent_sigmar_floor(backend):
    # self-consistent power law (gamma=alpha=2.5 -> n2=-0.5) has an integrable
    # Q^{-0.5} endpoint singularity in the velocity moment; the fixed-order GL
    # cannot resolve it (numpy adaptive quad refines there), so parity floors at
    # ~1.8e-3 -- a genuine quadrature-floor case (match-numpy-quadrature-floor).
    # Still exercises the n2<0 fractional-power fQ path on the backend.
    rs = [1.0, 3.0, 8.0]
    ref = numpy.array([_DFPS.sigmar(r, use_physical=False) for r in rs])
    got = _DFPS.sigmar(_arr(backend, numpy.array(rs)), use_physical=False)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=3e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # the general anisotropic OM dM/dE (nested GL after the r=rphi-s^2 and the
    # t=Lmax sin(phi) substitutions, phi clustered w^2 to cancel the sqrt(Q)
    # endpoint) vs the numpy nested adaptive quad, on the Hernquist variant
    Ed = numpy.array([-0.7 * _PSI0, -0.3 * _PSI0])
    ref = _DFH.dMdE(Ed, use_physical=False)
    got = _DFH.dMdE(_arr(backend, Ed), use_physical=False)
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)
    # out-of-bounds E -> exactly zero
    assert numpy.all(
        as_numpy(_DFH.dMdE(_arr(backend, numpy.array([0.5])), use_physical=False))
        == 0.0
    )


@pytest.mark.filterwarnings("ignore:.*requires_grad.*:UserWarning")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tag", list(_VARIANTS))
def test_fQ_grad_vs_fd(backend, tag):
    # d f(Q)/dQ through the closed forms; the out-of-bounds grad is a finite 0
    # (dead-branch guards), not NaN
    df, _, _, Q0 = _VARIANTS[tag]
    eps = 1e-6
    fd = float(
        (df.fQ(numpy.atleast_1d(Q0 + eps))[0] - df.fQ(numpy.atleast_1d(Q0 - eps))[0])
        / (2.0 * eps)
    )
    if backend == "jax":
        g = float(jax.grad(lambda q: jnp.sum(df.fQ(q)))(jnp.asarray(Q0)))
        goob = float(jax.grad(lambda q: jnp.sum(df.fQ(q)))(jnp.asarray(-1.0)))
    else:
        t = torch.tensor(Q0, requires_grad=True)
        torch.sum(df.fQ(t)).backward()
        g = float(t.grad)
        t = torch.tensor(-1.0, requires_grad=True)
        torch.sum(df.fQ(t)).backward()
        goob = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)
    assert goob == 0.0


@pytest.mark.filterwarnings("ignore:.*requires_grad.*:UserWarning")
@pytest.mark.parametrize("backend", BACKENDS)
def test_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr through the GL velocity-moment integrals (limits + Phi(r)),
    # on the Hernquist variant
    r0, eps = 1.3, 1e-5
    fd = (
        _DFH.sigmar(r0 + eps, use_physical=False)
        - _DFH.sigmar(r0 - eps, use_physical=False)
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(
            jax.grad(lambda r: _DFH.sigmar(r, use_physical=False))(jnp.asarray(r0))
        )
    else:
        t = torch.tensor(r0, requires_grad=True)
        _DFH.sigmar(t, use_physical=False).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, float(fd), rtol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tag", ["hern", "plaw"])
def test_sample_numpy_side_forced(backend, tag):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and the outputs are numpy arrays; only the
    # deterministic sub-steps (fQ/vesc grids, closed-form _icmf) run on the
    # backend, so draws match the pure-numpy ones to fp noise. hern covers the
    # closed-form _icmf + _p_v_at_r coerce; plaw covers _icmf + _vmax_at_r
    def make():
        if tag == "hern":
            return osipkovmerrittHernquistdf(pot=_HP, ra=2.3)
        return osipkovmerrittPowerLawdf(
            pot=_PP, denspot=_DPN, ra=2.0, rmax=100.0, rmin=1e-4
        )

    numpy.random.seed(10)
    ref = make().sample(n=80, return_orbit=False)
    numpy.random.seed(10)
    with galpy.backend.use(backend, force=True):
        got = make().sample(n=80, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-8, atol=1e-10)
