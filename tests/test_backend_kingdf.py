###############################################################################
# test_backend_kingdf.py: Track F Pdf.2 -- backend (jax/torch) coverage for the
# King spherical-DF family (kingdf). The numpy path is byte-identical
# (test_sphericaldf unchanged); this exercises the resolved-namespace dispatch
# in kingdf's own fE / dens paths and the inherited moment/dM/dE machinery:
# parity numpy<->jax<->torch of fE / dens / __call__ / moments / dM/dE,
# grad-vs-FD, is-backend-array assertions, and the numpy-side sampling contract
# (seeded draws unchanged under a forced backend).
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
from galpy.df import kingdf
from galpy.df.sphericaldf import isotropicsphericaldf


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


_DF = kingdf(W0=3.0, M=2.3, rt=1.76)
_PI = float(_DF._potInf)  # cutoff energy; bound stars have E < _potInf
# in-bounds E grid (E < _potInf) + out-of-bounds points (E >= _potInf) + the
# exact boundary E == _potInf (varE == 0 -> fE == 0)
_EGRID = numpy.concatenate(
    [
        numpy.linspace(_PI - abs(_PI) - 1.0, _PI - 1e-3, 21),
        numpy.array([_PI, _PI - 1e-12, _PI + 1e-12, _PI + 0.5]),
    ]
)
# radii within [0, rt] for dens/moments
_RS = numpy.array([_DF._scale / 5.0, _DF._scale, _DF.rt * 0.3, _DF.rt * 0.7])
_DENSRS = numpy.linspace(0.01, _DF.rt * 0.999, 12)
# dM/dE energies strictly inside (_potInf - |_potInf|, _potInf)
_EDM = numpy.linspace(_PI - abs(_PI) * 0.9, _PI - abs(_PI) * 0.05, 7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_parity(backend):
    # king fE is a smooth exp(varE/sigma^2)-1 form: numpy<->backend parity on an
    # E grid including the out-of-bounds (functional dead-mask -> 0) branch
    ref = _DF.fE(_EGRID)
    got = _DF.fE(_arr(backend, _EGRID))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_dead_edge(backend):
    # E == _potInf exactly (varE == 0) and E > _potInf are the dead branch;
    # fE -> 0 there, NaN-free under the dummy-then-zero guard
    got = as_numpy(_DF.fE(_arr(backend, numpy.array([_PI, _PI + 1e-9, _PI + 1.0]))))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dens_parity(backend):
    # dens goes through the Spline1D W(r) table + the erf-based _dens_W; measured
    # ~3e-13 (frozen-table eval + backend erf vs scipy)
    ref = _DF.dens(_DENSRS)
    got = _DF.dens(_arr(backend, _DENSRS))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # __call__ tuple form (E,) and the 6-coordinate form
    ref = _DF((_EGRID,))
    got = _DF((_arr(backend, _EGRID),))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    R = numpy.array([0.3, 0.6, 0.9, 1.2])
    vR = numpy.array([0.1, -0.2, 0.3, 0.0])
    vT = numpy.array([0.2, 0.4, 0.1, 0.3])
    z = numpy.array([0.1, -0.2, 0.3, 0.0])
    vz = numpy.array([-0.1, 0.1, 0.0, 0.2])
    ref = _DF(R, vR, vT, z, vz, numpy.zeros_like(R))
    got = _DF(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moments_parity(backend):
    # sigmar/sigmat (fixed-order GL vs numpy adaptive quad): the King DF is a
    # smooth exponential so GL matches to ~1.5e-15 (no quadrature floor here);
    # the exactly-isotropic beta is 0. Scalar and vector r.
    for name in ("sigmar", "sigmat"):
        f = getattr(_DF, name)
        ref = numpy.array([f(r) for r in _RS])
        got = numpy.array([float(f(_arr(backend, r))) for r in _RS])
        gotv = f(_arr(backend, _RS))
        assert _is_backend_array(backend, gotv)
        numpy.testing.assert_allclose(got, ref, rtol=1e-9)
        numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-9)
    b = _DF.beta(_arr(backend, _DF.rt * 0.3))
    assert float(as_numpy(b)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_dMdE_parity(backend):
    # the inherited isotropic base-class dM/dE (GL after the r = rphi - s^2
    # turning-point substitution + backend Spline1D rphi eval); measured ~4e-10
    # vs the numpy adaptive quad (rtol 1e-6 leaves margin for numpy's own floor)
    ref = isotropicsphericaldf._dMdE(_DF, _EDM)
    got = isotropicsphericaldf._dMdE(_DF, _arr(backend, _EDM))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_grad_vs_fd(backend):
    E0, eps = _PI - 0.5 * abs(_PI), 1e-6
    fd = (
        _DF.fE(numpy.atleast_1d(E0 + eps))[0] - _DF.fE(numpy.atleast_1d(E0 - eps))[0]
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda E: _DF.fE(E))(jnp.asarray(E0)))
        goob = float(jax.grad(lambda E: _DF.fE(E))(jnp.asarray(_PI + 0.5)))
    else:
        t = torch.tensor(E0, requires_grad=True)
        _DF.fE(t).backward()
        g = float(t.grad)
        t = torch.tensor(_PI + 0.5, requires_grad=True)
        _DF.fE(t).backward()
        goob = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)
    # out-of-bounds grad is a finite 0, not NaN (dead-branch guard)
    assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_dens_grad_vs_fd(backend):
    r0, eps = _DF.rt * 0.4, 1e-6
    fd = (
        float(_DF.dens(numpy.atleast_1d(r0 + eps))[0])
        - float(_DF.dens(numpy.atleast_1d(r0 - eps))[0])
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: _DF.dens(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        _DF.dens(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr through the GL moment integrals (limits + Phi(r))
    r0, eps = _DF.rt * 0.4, 1e-5
    fd = (_DF.sigmar(r0 + eps) - _DF.sigmar(r0 - eps)) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: _DF.sigmar(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        _DF.sigmar(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_numpy_side_forced(backend):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and the outputs are numpy arrays; only the
    # deterministic sub-steps (icmf/pvr grids built at __init__) matter, so
    # draws match the pure-numpy ones to fp noise
    ref_df = kingdf(W0=3.0, M=2.3, rt=1.76)
    numpy.random.seed(10)
    ref = ref_df.sample(n=100, return_orbit=False)
    dfb = kingdf(W0=3.0, M=2.3, rt=1.76)
    numpy.random.seed(10)
    with galpy.backend.use(backend, force=True):
        got = dfb.sample(n=100, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_under_forced_backend(backend):
    # _dens_W is a leaf consumed by numpy construction (solve_ivp, rho0/r0);
    # its is_backend_array data-guard must keep scalar/numpy W on numpy even
    # under a forced backend, so construction stays byte-identical to numpy
    # (a backend-resolving _dens_W would make _scale a tensor and break the
    # base class's numpy grid arithmetic)
    ref = kingdf(W0=3.0, M=2.3, rt=1.76)
    with galpy.backend.use(backend, force=True):
        dfb = kingdf(W0=3.0, M=2.3, rt=1.76)
    assert not _is_backend_array(backend, dfb._scale)
    assert numpy.array_equal(dfb._scalefree_kdf._W, ref._scalefree_kdf._W)
    assert numpy.array_equal(
        dfb._scalefree_kdf._cumul_mass, ref._scalefree_kdf._cumul_mass
    )
