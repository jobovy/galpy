###############################################################################
# test_backend_sphericaldf.py: Track F Pdf.2 (BASE + PILOT) -- backend
# (jax/torch) coverage for the spherical-DF foundation: sphericaldf base
# classes + the closed-form isotropicHernquistdf pilot. The numpy path is
# byte-identical (test_sphericaldf unchanged); this exercises the
# resolved-namespace dispatch: parity numpy<->jax<->torch of fE / __call__ /
# moments / dM/dE, grad-vs-FD, and the numpy-side sampling contract (numpy RNG
# draws unchanged under a forced backend; outputs are numpy arrays).
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
from galpy.df import isotropicHernquistdf
from galpy.df.sphericaldf import (
    anisotropicsphericaldf,
    isotropicsphericaldf,
    sphericaldf,
)
from galpy.potential import HernquistPotential


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


_HP = HernquistPotential(amp=2.3, a=1.3)
_DF = isotropicHernquistdf(pot=_HP)
_PSI0 = float(_DF._psi0)
# in-bounds E grid + out-of-bounds points (E > 0 and E < -psi0)
_EGRID = numpy.concatenate(
    [numpy.linspace(-0.999 * _PSI0, -1e-4, 21), [0.5, -1.5 * _PSI0]]
)
_ENEG = numpy.linspace(-0.95 * _PSI0, -0.05 * _PSI0, 11)
_RS = numpy.array([0.13, 0.5, 1.3, 5.2, 13.0])


class _IsoAsAniso(anisotropicsphericaldf):
    """Isotropic Hernquist dressed as an anisotropic DF: exercises the
    anisotropic base-class (E, L)-machinery with a known isotropic answer."""

    def __init__(self, pot=None):
        anisotropicsphericaldf.__init__(self, pot=pot)
        self._iso = isotropicHernquistdf(pot=pot)

    def _call_internal(self, E, L, Lz):
        fE = self._iso.fE(E)
        return fE if L is None else fE * (1.0 + 0.0 * L)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_parity(backend):
    # closed-form pilot: numpy<->backend parity of fE on an E grid including
    # the out-of-bounds (functional dummy-then-zero) branches
    ref = _DF.fE(_EGRID)
    got = _DF.fE(_arr(backend, _EGRID))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_zero_edge(backend):
    # E == 0 exactly is 0/0 in the numpy arcsin term (masked there by a
    # historical row-quirk for 2-D grids); the backend branch implements the
    # correct fE -> 0 limit, NaN-free (special-fn edge testing)
    got = _np(_DF.fE(_arr(backend, numpy.array([0.0, -0.0, -1e-300]))))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # __call__ tuple forms ((E,), (E, L)) and the 6-coordinate form
    ref = _DF((_EGRID,))
    got = _DF((_arr(backend, _EGRID),))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-12)
    # with (ignored) L: exercises the backend L-parse branch
    got = _DF((_arr(backend, _EGRID), _arr(backend, numpy.ones_like(_EGRID))))
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-12)
    R = numpy.array([0.5, 1.1, 1.7, 2.9])
    vR = numpy.array([0.1, -0.2, 0.3, 0.0])
    vT = numpy.array([0.3, 0.5, 0.2, 0.4])
    z = numpy.array([0.2, -0.3, 0.5, 0.0])
    vz = numpy.array([-0.1, 0.2, 0.0, 0.1])
    ref = _DF(R, vR, vT, z, vz, numpy.zeros_like(R))
    got = _DF(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moments_parity(backend):
    # sigmar/sigmat (GL-vs-adaptive quadrature parity, measured 1.4e-9) and the
    # exactly-isotropic beta; scalar and vector r (vectorized GL)
    for name in ("sigmar", "sigmat"):
        f = getattr(_DF, name)
        ref = numpy.array([f(r) for r in _RS])
        got = numpy.array([float(f(_arr(backend, r))) for r in _RS])
        gotv = f(_arr(backend, _RS))
        assert _is_backend_array(backend, gotv)
        numpy.testing.assert_allclose(got, ref, rtol=1e-8)
        numpy.testing.assert_allclose(_np(gotv), ref, rtol=1e-8)
    b = _DF.beta(_arr(backend, 1.3))
    assert float(_np(b)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # Hernquist closed-form dM/dE (functional masking branch)
    ref = _DF.dMdE(_ENEG)
    got = _DF.dMdE(_arr(backend, _ENEG))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-11)
    # out-of-bounds E -> exactly zero on the dead branch
    assert numpy.all(_np(_DF.dMdE(_arr(backend, numpy.array([0.5])))) == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_isotropic_dMdE_parity(backend):
    # the base-class quadrature dM/dE (GL after the r = rphi - s^2 turning-point
    # substitution + backend Spline1D rphi eval). rtol 1e-6 is numpy's own
    # adaptive-quad floor at the sqrt endpoint: against a tight gold reference
    # the numpy path errs by 4.9e-7 at the deepest E while the backend GL is
    # accurate to 1.3e-14 (match-numpy-quadrature-floor)
    dfh = isotropicHernquistdf(pot=_HP)
    ref = isotropicsphericaldf._dMdE(dfh, _ENEG)
    got = isotropicsphericaldf._dMdE(dfh, _arr(backend, _ENEG))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_anisotropic_vmomentdensity_parity(backend):
    # the anisotropic base-class (v, eta) tensor-product GL vs numpy dblquad
    adf = _IsoAsAniso(pot=_HP)
    for n, m in ((0, 0), (2, 0), (0, 2)):
        ref = sphericaldf._vmomentdensity(adf, 1.3, n, m)
        got = sphericaldf._vmomentdensity(adf, _arr(backend, 1.3), n, m)
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(_np(got), ref, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_anisotropic_dMdE_parity(backend):
    # the anisotropic base-class dM/dE (nested GL after the r = rphi - s^2 and
    # t = Lmax sin(phi) substitutions) vs the numpy nested adaptive quad
    adf = _IsoAsAniso(pot=_HP)
    ref = anisotropicsphericaldf._dMdE(adf, _ENEG[::3])
    got = anisotropicsphericaldf._dMdE(adf, _arr(backend, _ENEG[::3]))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(_np(got), ref, rtol=1e-7)
    # and it agrees with the isotropic base-class machinery
    iso = isotropicsphericaldf._dMdE(isotropicHernquistdf(pot=_HP), _ENEG[::3])
    numpy.testing.assert_allclose(_np(got), iso, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_grad_vs_fd(backend):
    E0, eps = -0.5 * _PSI0, 1e-6
    fd = (
        _DF.fE(numpy.atleast_1d(E0 + eps))[0] - _DF.fE(numpy.atleast_1d(E0 - eps))[0]
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda E: _DF.fE(E))(jnp.asarray(E0)))
    else:
        t = torch.tensor(E0, requires_grad=True)
        _DF.fE(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)
    # out-of-bounds grad is finite 0, not NaN (dead-branch guards)
    if backend == "jax":
        goob = float(jax.grad(lambda E: _DF.fE(E))(jnp.asarray(0.5)))
    else:
        t = torch.tensor(0.5, requires_grad=True)
        _DF.fE(t).backward()
        goob = float(t.grad)
    assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr through the GL moment integrals (limits + Phi(r))
    r0, eps = 1.3, 1e-5
    fd = (_DF.sigmar(r0 + eps) - _DF.sigmar(r0 - eps)) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda r: _DF.sigmar(r))(jnp.asarray(r0)))
    else:
        t = torch.tensor(r0, requires_grad=True)
        _DF.sigmar(t).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_grad_vs_fd(backend):
    # closed-form and base-quadrature dM/dE gradients w.r.t. E; the
    # out-of-bounds grad is finite 0 (dead-branch guards)
    dfh = isotropicHernquistdf(pot=_HP)
    E0, eps = -0.4 * _PSI0, 1e-6
    fd = (
        dfh.dMdE(numpy.atleast_1d(E0 + eps))[0]
        - dfh.dMdE(numpy.atleast_1d(E0 - eps))[0]
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(jax.grad(lambda E: dfh.dMdE(E))(jnp.asarray(E0)))
        gbase = float(
            jax.grad(lambda E: isotropicsphericaldf._dMdE(dfh, E[None])[0])(
                jnp.asarray(E0)
            )
        )
        goob = float(jax.grad(lambda E: dfh.dMdE(E))(jnp.asarray(0.5)))
    else:
        t = torch.tensor(E0, requires_grad=True)
        dfh.dMdE(t).backward()
        g = float(t.grad)
        t = torch.tensor([E0], requires_grad=True)
        isotropicsphericaldf._dMdE(dfh, t)[0].backward()
        gbase = float(t.grad[0])
        t = torch.tensor(0.5, requires_grad=True)
        dfh.dMdE(t).backward()
        goob = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)
    numpy.testing.assert_allclose(gbase, fd, rtol=1e-6)
    assert goob == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_numpy_side_forced(backend):
    # sampling is numpy-side by design: under a forced backend the numpy RNG
    # draw sequence is unchanged and the outputs are numpy arrays; only the
    # deterministic sub-steps (fE/vesc grids, closed-form icmf) run on the
    # backend, so draws match the pure-numpy ones to fp noise
    ref_df = isotropicHernquistdf(pot=_HP)
    numpy.random.seed(10)
    ref = ref_df.sample(n=100, return_orbit=False)
    dfb = isotropicHernquistdf(pot=_HP)
    numpy.random.seed(10)
    with galpy.backend.use(backend, force=True):
        got = dfb.sample(n=100, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-10, atol=1e-12)
    # position-conditioned branch with backend-array R, z, phi inputs
    numpy.random.seed(11)
    refRz = ref_df.sample(R=1.1, z=0.3, phi=0.7, n=20, return_orbit=False)
    numpy.random.seed(11)
    gotRz = ref_df.sample(
        R=_arr(backend, 1.1),
        z=_arr(backend, 0.3),
        phi=_arr(backend, 0.7),
        n=20,
        return_orbit=False,
    )
    for g, r in zip(gotRz, refRz):
        assert isinstance(g, numpy.ndarray)
        numpy.testing.assert_allclose(g, r, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_cmf_interpolator_forced(backend):
    # the interpolated inverse-CMF route (no closed-form _icmf): the mass grid
    # is evaluated on the forced backend, pulled numpy-side into the Spline1D
    # icdf; draws match pure numpy to the grid's fp noise
    class NoICMF(isotropicHernquistdf):
        _icmf = property()

    ref_df = NoICMF(pot=_HP)
    numpy.random.seed(12)
    ref = ref_df.sample(n=50, return_orbit=False)
    dfb = NoICMF(pot=_HP)
    numpy.random.seed(12)
    with galpy.backend.use(backend, force=True):
        got = dfb.sample(n=50, return_orbit=False)
    for g, r in zip(got, ref):
        assert isinstance(g, numpy.ndarray) and not _is_backend_array(backend, g)
        numpy.testing.assert_allclose(g, r, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_handle_rmin_forced(backend):
    # the divergence probe Phi(0) coerces its scalar coordinate under a forced
    # backend (undecorated potential evals reject scalars under torch)
    from galpy.df.sphericaldf import _handle_rmin

    ref = _handle_rmin(None, _HP, _HP, _HP._scale, 8.0, "testdf")
    with galpy.backend.use(backend, force=True):
        got = _handle_rmin(None, _HP, _HP, _HP._scale, 8.0, "testdf")
    assert got == ref == 0.0  # Hernquist Phi(0) is finite -> rmin = 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_setup_rphi_interpolator_forced(backend):
    # forced backend vectorizes the r(Phi) grid construction (one call instead
    # of nra scalar dispatches); the resulting spline matches pure numpy
    ref = isotropicHernquistdf(pot=_HP)._setup_rphi_interpolator()
    dfb = isotropicHernquistdf(pot=_HP)
    with galpy.backend.use(backend, force=True):
        got = dfb._setup_rphi_interpolator()
    Es = numpy.linspace(-0.9 * _PSI0, -0.1 * _PSI0, 7)
    numpy.testing.assert_allclose(got(Es), ref(Es), rtol=1e-12)
    # and the frozen table evaluates natively on backend queries
    gb = got(_arr(backend, Es))
    assert _is_backend_array(backend, gb)
    numpy.testing.assert_allclose(_np(gb), ref(Es), rtol=1e-10)
