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

from galpy.backend import as_numpy

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
from galpy.backend import random as grandom
from galpy.backend.interpolate import interp_linear
from galpy.df import eddingtondf, isotropicHernquistdf
from galpy.df.sphericaldf import (
    anisotropicsphericaldf,
    isotropicsphericaldf,
    sphericaldf,
)
from galpy.potential import HernquistPotential

if torch is not None:
    import array_api_compat.torch as _TXP


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


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
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fE_zero_edge(backend):
    # E == 0 exactly is 0/0 in the numpy arcsin term (masked there by a
    # historical row-quirk for 2-D grids); the backend branch implements the
    # correct fE -> 0 limit, NaN-free (special-fn edge testing)
    got = as_numpy(_DF.fE(_arr(backend, numpy.array([0.0, -0.0, -1e-300]))))
    assert numpy.all(got == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_parity(backend):
    # __call__ tuple forms ((E,), (E, L)) and the 6-coordinate form
    ref = _DF((_EGRID,))
    got = _DF((_arr(backend, _EGRID),))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    # with (ignored) L: exercises the backend L-parse branch
    got = _DF((_arr(backend, _EGRID), _arr(backend, numpy.ones_like(_EGRID))))
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    R = numpy.array([0.5, 1.1, 1.7, 2.9])
    vR = numpy.array([0.1, -0.2, 0.3, 0.0])
    vT = numpy.array([0.3, 0.5, 0.2, 0.4])
    z = numpy.array([0.2, -0.3, 0.5, 0.0])
    vz = numpy.array([-0.1, 0.2, 0.0, 0.1])
    ref = _DF(R, vR, vT, z, vz, numpy.zeros_like(R))
    got = _DF(*(_arr(backend, c) for c in (R, vR, vT, z, vz)))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


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
        numpy.testing.assert_allclose(as_numpy(gotv), ref, rtol=1e-8)
    b = _DF.beta(_arr(backend, 1.3))
    assert float(as_numpy(b)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dMdE_parity(backend):
    # Hernquist closed-form dM/dE (functional masking branch)
    ref = _DF.dMdE(_ENEG)
    got = _DF.dMdE(_arr(backend, _ENEG))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11)
    # out-of-bounds E -> exactly zero on the dead branch
    assert numpy.all(as_numpy(_DF.dMdE(_arr(backend, numpy.array([0.5])))) == 0.0)


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
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_anisotropic_vmomentdensity_parity(backend):
    # the anisotropic base-class (v, eta) tensor-product GL vs numpy dblquad
    adf = _IsoAsAniso(pot=_HP)
    for n, m in ((0, 0), (2, 0), (0, 2)):
        ref = sphericaldf._vmomentdensity(adf, 1.3, n, m)
        got = sphericaldf._vmomentdensity(adf, _arr(backend, 1.3), n, m)
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_base_anisotropic_dMdE_parity(backend):
    # the anisotropic base-class dM/dE (nested GL after the r = rphi - s^2 and
    # t = Lmax sin(phi) substitutions) vs the numpy nested adaptive quad
    adf = _IsoAsAniso(pot=_HP)
    ref = anisotropicsphericaldf._dMdE(adf, _ENEG[::3])
    got = anisotropicsphericaldf._dMdE(adf, _arr(backend, _ENEG[::3]))
    assert _is_backend_array(backend, got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7)
    # and it agrees with the isotropic base-class machinery
    iso = isotropicsphericaldf._dMdE(isotropicHernquistdf(pot=_HP), _ENEG[::3])
    numpy.testing.assert_allclose(as_numpy(got), iso, rtol=1e-6)


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
    numpy.testing.assert_allclose(as_numpy(gb), ref(Es), rtol=1e-10)


# Orbit.E()/L() accessors trip a pre-existing numpy __array_wrap__ deprecation on
# torch tensors (Orbits.py, outside the df scope); ignore just that one here.
@pytest.mark.filterwarnings(
    "ignore:__array_wrap__ must accept context:DeprecationWarning"
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_call_orbit_forced(backend):
    # __call__'s Orbit branch builds |L| = sqrt(sum L^2); under a forced backend
    # Orbit.L() is a backend array, so the reduction runs in the active namespace
    # (numpy.sum(tensor) would raise TypeError). numpy path stays byte-identical.
    from galpy.orbit import Orbit

    ic = [0.6, 0.05, 0.2, 0.02, 0.03, 1.0]  # bound -> nonzero f
    ref = as_numpy(_DF(Orbit(ic)))
    assert numpy.all(ref > 0.0)
    with galpy.backend.use(backend, force=True):
        got = _DF(Orbit(ic))
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)
    # the anisotropic base actually consumes |L| in _call_internal
    ani = _IsoAsAniso(pot=_HP)
    ref_a = as_numpy(ani(Orbit(ic)))
    with galpy.backend.use(backend, force=True):
        got_a = ani(Orbit(ic))
    numpy.testing.assert_allclose(as_numpy(got_a), ref_a, rtol=1e-12)


###############################################################################
# Backend-native, differentiable radial + analytic-angle sampling via a backend
# ``key`` (interp_linear inverse-CDF). The numpy path (key=None) is byte-
# identical and covered by test_sphericaldf; here we exercise the backend key.
###############################################################################
def _key(backend, seed=7):
    return grandom.key(seed, backend)


def _ns(backend):
    return jnp if backend == "jax" else _TXP


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_r_interp_linear_same_u_parity(backend):
    # feed the SAME uniforms to interp_linear on the DF's (cdf, xi) grid under
    # numpy vs the backend -> identical inverse-CDF samples (the whole radial
    # sampler is a deterministic function of the uniforms)
    xp = _ns(backend)
    df = eddingtondf(pot=_HP)
    df.sample(n=1, return_orbit=False)  # build the (cdf, xi) grids
    ms, xis = df._get_cmf_grids()
    u = numpy.random.uniform(size=400)
    ref = interp_linear(numpy, ms, xis, u, extrapolate="clip")
    got = interp_linear(
        xp, _arr(backend, ms), _arr(backend, xis), _arr(backend, u), extrapolate="clip"
    )
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_backend_key_coords_and_distribution(backend):
    # a full sample() under a backend key returns backend-array coordinates and
    # reproduces the analytic Hernquist mass profile + azimuthal symmetry
    import tests.test_sphericaldf as T
    from galpy.orbit import Orbit

    df = isotropicHernquistdf(pot=_HP)
    R, vR, vT, z, vz, phi = df.sample(n=4000, return_orbit=False, key=_key(backend))
    for c in (R, z, phi, vR, vz, vT):
        assert _is_backend_array(backend, c)
    a = _HP.a
    samp = Orbit(
        vxvv=numpy.array(
            [
                as_numpy(R),
                as_numpy(vR),
                as_numpy(vT),
                as_numpy(z),
                as_numpy(vz),
                as_numpy(phi),
            ]
        ).T
    )
    T.check_spherical_massprofile(
        samp, lambda r: r**2.0 / (r + a) ** 2.0, 0.05, skip=1000
    )
    T.check_azimuthal_symmetry(samp, 1, 0.05)
    # king also samples r backend-native via its grid _icmf
    from galpy.df import kingdf

    Rk, _, _, zk, _, _ = kingdf(W0=3.0, M=2.0, rt=1.5).sample(
        n=1000, return_orbit=False, key=_key(backend)
    )
    assert _is_backend_array(backend, Rk) and _is_backend_array(backend, zk)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_r_grad_vs_fd_cdf_grid(backend):
    # d(sample_r)/d(cdf_grid): the interp_linear inverse-CDF is differentiable in
    # the CDF knots (the parameter-dependent quantity). Random directional AD
    # must h-converge to a central FD of the numpy path.
    df = eddingtondf(pot=_HP)
    df.sample(n=1, return_orbit=False)
    ms, xis = df._get_cmf_grids()
    scale = _HP._scale
    u = numpy.random.uniform(size=200)
    rng = numpy.random.default_rng(0)
    d = rng.standard_normal(ms.shape)
    d /= numpy.linalg.norm(d)

    def sumr_np(cdf):
        xi = interp_linear(numpy, cdf, xis, u, extrapolate="clip")
        return numpy.sum(scale * (1.0 + xi) / (1.0 - xi))

    if backend == "jax":

        def sumr(cdf):
            xi = interp_linear(
                jnp, cdf, jnp.asarray(xis), jnp.asarray(u), extrapolate="clip"
            )
            return jnp.sum(scale * (1.0 + xi) / (1.0 - xi))

        g = numpy.asarray(jax.grad(sumr)(jnp.asarray(ms)))
    else:
        c = torch.tensor(ms, requires_grad=True)
        xi = interp_linear(
            _TXP, c, torch.tensor(xis), torch.tensor(u), extrapolate="clip"
        )
        (scale * (1.0 + xi) / (1.0 - xi)).sum().backward()
        g = c.grad.numpy()
    ad = float(numpy.dot(g, d))
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (sumr_np(ms + h * d) - sumr_np(ms - h * d)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-4 * abs(ad) + 1e-7, f"cdf-grad {backend} best={best:.2e}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_r_grad_vs_fd_scale(backend):
    # d(sum sample_r)/d(a): Hernquist scale radius via the analytic (backend-
    # native) _icmf, differentiated through _sample_r with a fixed backend key
    # (CRN -> same uniforms). AD must h-converge to a central FD.
    key = _key(backend, 3)

    def make(aval):
        d = isotropicHernquistdf(pot=HernquistPotential(amp=2.3, a=1.3))
        d._pot.a = aval  # differentiable leaf on a fresh pot (no shared-obj leak)
        return d

    if backend == "jax":
        g = float(
            jax.grad(lambda a: jnp.sum(make(a)._sample_r(n=150, key=key)))(
                jnp.asarray(1.3)
            )
        )
        out = make(jnp.asarray(1.3))._sample_r(n=150, key=key)
        u = numpy.asarray(grandom.uniform(key, 150))
    else:
        at = torch.tensor(1.3, requires_grad=True)
        out = make(at)._sample_r(n=150, key=key)
        out.sum().backward()
        g = float(at.grad)
        u = grandom.uniform(key, 150).numpy()
    assert _is_backend_array(backend, out)
    assert numpy.isfinite(g) and abs(g) > 0
    sq = numpy.sqrt(u)

    def sumr(a):
        return numpy.sum(a * sq / (1.0 - sq))

    best = min(
        abs(g - (sumr(1.3 + h) - sumr(1.3 - h)) / (2 * h)) for h in (1e-3, 1e-4, 1e-5)
    )
    assert best < 1e-5 * abs(g) + 1e-7, f"scale-grad {backend} best={best:.2e}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_backend_key_angles_independent(backend):
    # regression: sample() must hand each angle sampler an INDEPENDENT sub-key.
    # A prior bug passed the SAME parent key to the position- and velocity-angle
    # samplers, which each re-split it identically -> the velocity polar angle
    # became a deterministic function of the azimuthal position angle -> a biased
    # joint (position, velocity) distribution. Isotropic -> E[vz] = 0, so the
    # correlation showed up as a systematic mean(vz) offset (was ~ +0.033).
    df = isotropicHernquistdf(pot=_HP)
    vz = numpy.concatenate(
        [
            as_numpy(df.sample(n=6000, return_orbit=False, key=_key(backend, s))[4])
            for s in (1, 2, 3)
        ]
    )
    assert abs(numpy.mean(vz)) < 0.012, (
        f"mean(vz)={numpy.mean(vz):.4f} -- correlated angle sub-keys?"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_eddington_general_backend_reachable(backend):
    # the general (no closed-form _icmf) interp_linear inverse-CDF branch is
    # reachable via the public eddingtondf.sample(key=...) and returns backend
    # coordinates -- isotropic velocity sampling is backend-native
    coords = eddingtondf(pot=_HP).sample(n=300, return_orbit=False, key=_key(backend))
    for c in coords:
        assert _is_backend_array(backend, c)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_v_native_pvr_parity(backend):
    # the native bilinear inverse-CDF pvr must reproduce the scipy
    # RectBivariateSpline(kx=1, ky=1) it dual-paths, at the SAME (log10 r/a, u)
    # query points, to ~1e-13 -- so a backend-key velocity equals the numpy one
    # given the same uniforms.
    df = isotropicHernquistdf(pot=_HP)
    df.sample(n=1, return_orbit=False)  # build the pvr interpolator
    pvr = df._v_vesc_pvr_interpolator
    rng = numpy.random.default_rng(5)
    X = rng.uniform(-2.0, 2.0, 300)  # log10(r/a)
    Y = rng.uniform(0.0, 1.0, 300)  # velocity uniform
    ref = pvr(X, Y, grid=False)  # scipy path
    got = as_numpy(pvr(_arr(backend, X), _arr(backend, Y), grid=False))  # native
    numpy.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sample_v_grad_vs_fd_r(backend):
    # d(sampled velocity)/d(r): the backend-key velocity magnitude is a
    # differentiable function of the backend r (through both the pvr query
    # log10(r/a) and vmax(r)). Random directional AD must h-converge to a central
    # FD of the numpy path (fixed velocity uniforms = common random numbers).
    xp = _ns(backend)
    df = isotropicHernquistdf(pot=_HP)
    df.sample(n=1, return_orbit=False)  # build the pvr interpolator
    pvr = df._v_vesc_pvr_interpolator
    scale = df._scale
    rng = numpy.random.default_rng(6)
    r0 = rng.uniform(0.3, 4.0, 20)
    u_v = rng.uniform(0.05, 0.95, 20)  # fixed velocity uniforms (CRN)
    d = rng.standard_normal(r0.shape)
    d /= numpy.linalg.norm(d)

    def sumv_np(r):
        v = pvr(numpy.log10(r / scale), u_v, grid=False) * as_numpy(
            df._vmax_at_r(df._pot, r)
        )
        return numpy.sum(v)

    def loss_b(r_b):
        v = pvr(xp.log10(r_b / scale), _arr(backend, u_v), grid=False) * df._vmax_at_r(
            df._pot, r_b
        )
        return xp.sum(v)

    with galpy.backend.use(backend, force=True):
        if backend == "jax":
            g = numpy.asarray(jax.grad(loss_b)(jnp.asarray(r0)))
        else:
            rt = torch.tensor(r0, requires_grad=True)
            loss_b(rt).backward()
            g = rt.grad.numpy()
    ad = float(numpy.dot(g, d))
    assert numpy.isfinite(ad) and abs(ad) > 0
    best = min(
        abs(ad - (sumv_np(r0 + h * d) - sumv_np(r0 - h * d)) / (2 * h))
        for h in (1e-4, 1e-5, 1e-6)
    )
    assert best < 1e-4 * abs(ad) + 1e-7, f"v-grad {backend} best={best:.2e}"


@pytest.mark.skipif(not BACKENDS, reason="no backend installed")
def test_sample_anisotropic_backend_key_raises():
    # anisotropic velocity-angle sampling on a backend key is deferred; a clear
    # NotImplementedError beats a confusing TypeError / a silent numpy result.
    # key=None (numpy) still works. Backend-independent (the guard keys off the
    # key type), so run it once.
    from galpy.df import constantbetadf, osipkovmerrittdf

    backend = BACKENDS[0]
    for df in (constantbetadf(pot=_HP, beta=-0.2), osipkovmerrittdf(pot=_HP, ra=1.5)):
        with pytest.raises(NotImplementedError):
            df.sample(n=10, return_orbit=False, key=_key(backend))
        df.sample(n=10, return_orbit=False)  # numpy path still works
