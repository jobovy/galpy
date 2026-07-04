###############################################################################
# test_backend_isotropicsphericaldf.py: Track F Pdf.2 -- backend (jax/torch)
# coverage for the isotropic closed-form spherical-DF family:
# isotropicPlummerdf, isotropicNFWdf (improved + Widrow fits), and
# isotropicPowerLawdf. The numpy path stays byte-identical (test_sphericaldf
# unchanged); this exercises the resolved-namespace dispatch: parity
# numpy<->jax<->torch of fE / __call__ / moments / dM/dE, grad-vs-FD of fE and
# sigmar, is-backend-array assertions, and the numpy-side seeded-sampling
# contract (numpy RNG draws unchanged under a forced backend).
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
from galpy.df import isotropicNFWdf, isotropicPlummerdf, isotropicPowerLawdf
from galpy.potential import (
    NFWPotential,
    PlummerPotential,
    PowerSphericalPotential,
)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


# --- fixtures: one instance of each closed-form DF + its energy grids --------
_PP = PlummerPotential(amp=1.7, b=1.3)
_PDF = isotropicPlummerdf(pot=_PP)
_PSI0 = float(-_PP(0.0, 0.0, use_physical=False))
_NP = NFWPotential(amp=2.0, a=1.5)
_NDF = isotropicNFWdf(pot=_NP, rmax=1e4)
_NDF_W = isotropicNFWdf(pot=_NP, widrow=True, rmax=1e4)
_ETMAX_N = float(_NP._amp / _NP.a)
_PLP = PowerSphericalPotential(amp=1.0, alpha=2.5)
_LDF = isotropicPowerLawdf(pot=_PLP, rmax=1e4, rmin=1e-6)

# fE grids: in-bounds + out-of-bounds (E>0 and E below the well floor)
_EGRID_P = numpy.concatenate(
    [numpy.linspace(-0.999 * _PSI0, -1e-4, 25), [0.5, -1.5 * _PSI0]]
)
_ETG = numpy.linspace(_NDF._Etildemin + 1e-3, 0.999, 25)
_EGRID_N = numpy.concatenate([-_ETG * _ETMAX_N, [0.5, -2.0 * _ETMAX_N]])
_EGRID_L = numpy.concatenate([numpy.linspace(-3.0, -1e-3, 25), [0.5, 1.0]])

_RS = numpy.array([0.13, 0.5, 1.3, 5.2, 13.0])

# (name, df, energy grid, an in-bounds scalar E, fE-scalar for grad-vs-FD)
_CASES = [
    ("plummer", _PDF, _EGRID_P, -0.4 * _PSI0),
    ("nfw", _NDF, _EGRID_N, -0.4 * _ETMAX_N),
    ("nfw_widrow", _NDF_W, _EGRID_N, -0.4 * _ETMAX_N),
    ("powerlaw", _LDF, _EGRID_L, -1.3),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,df,egrid,E0", _CASES)
def test_fE_parity(backend, name, df, egrid, E0):
    # closed-form fE: numpy<->backend parity on an E grid including the
    # out-of-bounds (functional dummy-then-zero) branches
    ref = df.fE(egrid)
    got = df.fE(_arr(backend, egrid))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    # scalar E returns a backend array too
    gs = df.fE(_arr(backend, E0))
    assert _is_backend_array(backend, gs)
    numpy.testing.assert_allclose(
        float(as_numpy(gs)), float(df.fE(numpy.atleast_1d(E0))[0]), rtol=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,df,egrid,E0", _CASES)
def test_fE_out_of_bounds_zero(backend, name, df, egrid, E0):
    # out-of-bounds E (E > 0) is exactly zero on the dead branch, NaN-free
    got = as_numpy(df.fE(_arr(backend, numpy.array([0.5, 1.0, 10.0]))))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_powerlaw_scalar_float_forced(backend):
    # forced backend + Python-float E (no .shape) hits the scalar out[0] return
    # on isotropicPowerLawdf's backend branch; result is a backend scalar array
    ref = float(_LDF.fE(numpy.atleast_1d(-1.3))[0])
    with galpy.backend.use(backend, force=True):
        got = _LDF.fE(-1.3)
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # __call__ tuple forms ((E,), (E, L)) and the 6-coordinate form (Plummer)
    ref = _PDF((_EGRID_P,))
    got = _PDF((_arr(backend, _EGRID_P),))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    got = _PDF((_arr(backend, _EGRID_P), _arr(backend, numpy.ones_like(_EGRID_P))))
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    R = numpy.array([0.5, 1.1, 1.7, 2.9])
    vR = numpy.array([0.1, -0.2, 0.3, 0.0])
    vT = numpy.array([0.3, 0.5, 0.2, 0.4])
    z = numpy.array([0.2, -0.3, 0.5, 0.0])
    vz = numpy.array([-0.1, 0.2, 0.0, 0.1])
    ref = _PDF(R, vR, vT, z, vz, numpy.zeros_like(R))
    got = _PDF(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,df", [("plummer", _PDF), ("nfw", _NDF), ("powerlaw", _LDF)]
)
def test_moments_parity(backend, name, df):
    # sigmar/sigmat GL-vs-adaptive quadrature parity (measured <=1.4e-10) and
    # the exactly-isotropic beta; scalar and vectorized r
    for mname in ("sigmar", "sigmat"):
        f = getattr(df, mname)
        ref = numpy.array([f(r) for r in _RS])
        got = numpy.array([float(f(_arr(backend, r))) for r in _RS])
        gotv = f(_arr(backend, _RS))
        assert _is_backend_array(backend, gotv)
        numpy.testing.assert_allclose(got, ref, rtol=1e-8)
        numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-8)
    b = df.beta(_arr(backend, 1.3))
    assert float(as_numpy(b)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # base-class quadrature dM/dE (Plummer: no closed-form _dMdE override); the
    # r = rphi - s^2 turning-point substitution + GL matches numpy adaptive quad
    # to 1.2e-13 (measured), no quadrature floor at these energies
    Eneg = numpy.linspace(0.95 * (-_PSI0), 0.05 * (-_PSI0), 7)
    ref = _PDF.dMdE(Eneg)
    got = _PDF.dMdE(_arr(backend, Eneg))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,df,egrid,E0", _CASES)
def test_fE_grad_vs_fd(backend, name, df, egrid, E0):
    eps = 1e-6
    fd = (
        df.fE(numpy.atleast_1d(E0 + eps))[0] - df.fE(numpy.atleast_1d(E0 - eps))[0]
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda E: df.fE(E).reshape(()))(jnp.asarray(E0)))
    else:
        t = torch.tensor(E0, requires_grad=True)
        df.fE(t).reshape(()).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)
    # out-of-bounds grad is a finite 0, not NaN (dead-branch guards)
    if backend == "jax":
        goob = float(jax.grad(lambda E: df.fE(E).reshape(()))(jnp.asarray(0.5)))
    else:
        t = torch.tensor(0.5, requires_grad=True)
        df.fE(t).reshape(()).backward()
        goob = float(t.grad)
    assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,df", [("plummer", _PDF), ("nfw", _NDF), ("powerlaw", _LDF)]
)
def test_sigmar_grad_vs_fd(backend, name, df):
    # d(sigma_r)/dr through the GL moment integrals (limits + Phi(r))
    r0, eps = 1.3, 1e-5
    fd = (df.sigmar(r0 + eps) - df.sigmar(r0 - eps)) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: df.sigmar(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        df.sigmar(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,mk,seed,rtol,atol",
    [
        ("plummer", lambda: isotropicPlummerdf(pot=_PP), 10, 1e-10, 1e-12),
        # NFW has no closed-form _icmf -> interpolated CMF; the mass grid is
        # built on the backend then pulled numpy-side into the Spline1D icdf, so
        # jax draws differ from pure numpy at the grid's fp floor (~2e-6)
        ("nfw", lambda: isotropicNFWdf(pot=_NP, rmax=1e4), 20, 1e-5, 1e-6),
        (
            "powerlaw",
            lambda: isotropicPowerLawdf(pot=_PLP, rmax=1e4, rmin=1e-6),
            30,
            1e-10,
            1e-12,
        ),
    ],
)
def test_sample_numpy_side_forced(backend, name, mk, seed, rtol, atol):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and outputs are numpy arrays; only the
    # deterministic sub-steps (fE/vesc grids, closed-form icmf) run on the
    # backend, so draws match the pure-numpy ones to fp noise
    numpy.random.seed(seed)
    ref = mk().sample(n=100, return_orbit=False)
    dfb = mk()
    numpy.random.seed(seed)
    with galpy.backend.use(backend, force=True):
        got = dfb.sample(n=100, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=rtol, atol=atol)
