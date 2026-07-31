###############################################################################
# test_backend_qdf.py: Track F Pdf.4 PR-4a -- backend (jax/torch) coverage for
# the quasiisothermaldf __call__ + _rg core. The numpy path is byte-identical
# (test_qdf unchanged); this exercises the resolved-namespace dispatch:
#   (a) value parity numpy<->jax<->torch of __call__ (log + non-log), including
#       the lz<0 / NaN sentinel branches, and
#   (b) grad-vs-FD of __call__ w.r.t. (R,vR,vT,z,vz), flowing the #131 Staeckel
#       c=True action gradients through the pure-arithmetic DF core.
# The Staeckel action gradient is first-order-accurate (#1050), so grad-vs-FD is
# checked at rtol 8e-3 (>2x margin over the observed ~3.4e-3 floor).
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
from galpy.actionAngle import actionAngleStaeckel
from galpy.backend import as_numpy, is_backend_array
from galpy.df import quasiisothermaldf
from galpy.potential import MWPotential

# Staeckel-based qdf fixture, mirroring test_qdf.py's Staeckel qdf setup.
_aAS = actionAngleStaeckel(pot=MWPotential, c=True, delta=0.5)
_qdf = quasiisothermaldf(
    1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=_aAS, cutcounter=True
)

# Interior prograde orbits (bound, lz>0); includes vR=0/vz=0/z=0 edge orbits.
_ORBITS = numpy.array(
    [
        [0.9, 0.1, 0.9, 0.05, 0.02],
        [1.1, -0.05, 0.8, 0.1, -0.03],
        [0.7, 0.2, 1.0, -0.08, 0.05],
        [1.0, 0.0, 0.95, 0.0, 0.0],
    ]
)


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(numpy.asarray(x, float))


def _cols(backend, orbits):
    return [_arr(backend, orbits[:, i]) for i in range(5)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_value_parity(backend, log):
    # __call__ (R,vR,vT,z,vz) value byte-identity numpy<->backend, both modes.
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log))
    got = _qdf(*_cols(backend, _ORBITS), log=log)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_sentinel_lzneg(backend, log):
    # cutcounter lz<0 sentinel branch: retrograde orbit -> -finfo.max (log) / 0.
    retro = numpy.array([[0.9, 0.1, -1.5, 0.05, 0.02]])
    cn = [retro[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log))
    got = as_numpy(_qdf(*_cols(backend, retro), log=log))
    # exact match of the sentinel/zero (dtype-generic finfo max == float64 max)
    assert numpy.array_equal(got, ref)
    if log:
        assert ref[0] == -numpy.finfo(numpy.float64).max
    else:
        assert ref[0] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_sentinel_nan(backend):
    # NaN sentinel branch (log mode): a func returning a negative value makes
    # log(func) NaN -> masked to -finfo.max on both numpy and backend paths.
    negfunc = lambda jr, lz, jz: jr - 100.0
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=True, func=negfunc))
    got = as_numpy(_qdf(*_cols(backend, _ORBITS), log=True, func=negfunc))
    assert numpy.all(ref == -numpy.finfo(numpy.float64).max)  # NaN tripped
    assert numpy.array_equal(got, ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_func_parity(backend, log):
    # func-path value parity (moment weight lz**2 * jr) through xp.log/funcFactor.
    f = lambda jr, lz, jz: lz**2 * jr + 1e-3
    cn = [_ORBITS[:, i] for i in range(5)]
    ref = as_numpy(_qdf(*cn, log=log, func=f))
    got = _qdf(*_cols(backend, _ORBITS), log=log, func=f)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rg_backend_parity(backend):
    # _rg backend Spline1D eval matches the numpy scipy spline to ~1e-12 in the
    # physical (in-precompute-range) regime; output is a backend array.
    lzn = numpy.array([0.3, 0.5, 0.81, 1.0, 1.4])
    ref = as_numpy(_qdf._rg(lzn))
    got = _qdf._rg(_arr(backend, lzn))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rg_no_precompute(backend):
    # _precomputerg=False: the rg spline is never built (_rgInterpBackend is None),
    # so the backend _rg falls through to potential.rl everywhere -- matching the
    # numpy scalar path (which also always takes rl since Lzmin/max are +/-finfo).
    from galpy.potential import rl

    qdf0 = quasiisothermaldf(
        1.0 / 4.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=_aAS,
        cutcounter=True,
        _precomputerg=False,
    )
    assert qdf0._rgInterpBackend is None
    lzn = numpy.array([0.3, 0.81, 1.4])
    ref = numpy.array([float(rl(MWPotential, l)) for l in lzn])
    got = qdf0._rg(_arr(backend, lzn))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("log", [False, True])
def test_call_grad_vs_fd(backend, log):
    # jax.grad / torch.autograd of __call__ w.r.t. (R,vR,vT,z,vz) vs central FD.
    # Exercises the #131 Staeckel c=True action grads through jr/lz/jz; the
    # action grad is first-order-accurate (#1050) -> rtol 8e-3.
    def npval(p):
        a = [numpy.atleast_1d(x) for x in p]
        return float(as_numpy(_qdf(*a, log=log)).reshape(-1)[0])

    def fd(p, h=1e-6):
        g = numpy.zeros(5)
        for i in range(5):
            pp = p.copy()
            pp[i] += h
            pm = p.copy()
            pm[i] -= h
            g[i] = (npval(pp) - npval(pm)) / (2.0 * h)
        return g

    for ic in _ORBITS:
        ic = numpy.asarray(ic, float)
        gfd = fd(ic)
        if backend == "jax":

            def f(v):
                a = [v[i].reshape(1) for i in range(5)]
                return _qdf(*a, log=log).reshape(())

            g = as_numpy(jax.grad(f)(jnp.asarray(ic)))
        else:
            vt = torch.tensor(ic, requires_grad=True)
            a = [vt[i].reshape(1) for i in range(5)]
            _qdf(*a, log=log).reshape(()).backward()
            g = vt.grad.numpy()
        numpy.testing.assert_allclose(g, gfd, rtol=8e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# PR-4b: pv-projection (GL) + mc-path (_vmomentdensity) backend parity. These
# integrate the migrated __call__ over a GL / Monte-Carlo velocity grid; the
# numpy path stays byte-identical (test_qdf/test_pv2qdf unchanged). Scalar
# inputs under a forced backend exercise the `if xp is not numpy` promotion
# branches (promote_scalars, GL-table asarray, xp.concatenate/tile/permute_dims).
# ---------------------------------------------------------------------------
_PV_ARGS = {
    "pvR": (0.1, 0.9, 0.05),
    "pvT": (1.0, 0.9, 0.05),
    "pvz": (0.05, 0.9, 0.05),
    "pvRvT": (0.1, 1.0, 0.9, 0.05),
    "pvTvz": (1.0, 0.05, 0.9, 0.05),
    "pvRvz": (0.1, 0.05, 0.9, 0.05),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(_PV_ARGS))
def test_pv_projection_parity(backend, name):
    # GL pv-projection value parity numpy<->backend + backend-array output.
    args = _PV_ARGS[name]
    ref = as_numpy(getattr(_qdf, name)(*args, use_physical=False))
    with galpy.backend.use(backend, force=True):
        got = getattr(_qdf, name)(*args, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pvz_array_parity(backend):
    # pvz with array (vz,R,z) input: the permute_dims 3-D-.T tiling array path.
    vz = 0.05 * numpy.ones(2)
    R = 0.9 * numpy.ones(2)
    z = 0.05 * numpy.ones(2)
    ref = as_numpy(_qdf.pvz(vz, R, z, use_physical=False))
    with galpy.backend.use(backend, force=True):
        got = _qdf.pvz(vz, R, z, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pvRvz_array_parity(backend):
    # pvRvz with array (vR,vz,R,z) input: the 2-D-.T tiling array path.
    vR = 0.1 * numpy.ones(2)
    vz = 0.05 * numpy.ones(2)
    R = 0.9 * numpy.ones(2)
    z = 0.05 * numpy.ones(2)
    ref = as_numpy(_qdf.pvRvz(vR, vz, R, z, use_physical=False))
    with galpy.backend.use(backend, force=True):
        got = _qdf.pvRvz(vR, vz, R, z, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vmomentdensity_mc_parity(backend):
    # Monte-Carlo _vmomentdensity path: numpy random draws promoted to the
    # backend, xp.mean reduction, backend-array scalar out. Re-seed both sides;
    # the draws are then identical so parity is tight despite being an MC sum.
    numpy.random.seed(1)
    ref = as_numpy(_qdf._vmomentdensity(0.9, 0.05, 1, 0, 0, mc=True, nmc=10000))
    with galpy.backend.use(backend, force=True):
        numpy.random.seed(1)
        got = _qdf._vmomentdensity(0.9, 0.05, 1, 0, 0, mc=True, nmc=10000)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_jmomentdensity_mc_parity(backend):
    # Monte-Carlo _jmomentdensity path: promote_scalars on (R,z) and on the numpy
    # draws, the xp.where va-clamp, xp.mean reduction -- the backend-only branches
    # (the numpy result is byte-identical). Re-seed both sides for a tight MC match.
    numpy.random.seed(2)
    ref = as_numpy(_qdf._jmomentdensity(0.9, 0.05, 0, 0, 0, mc=True, nmc=10000))
    with galpy.backend.use(backend, force=True):
        numpy.random.seed(2)
        got = _qdf._jmomentdensity(0.9, 0.05, 0, 0, 0, mc=True, nmc=10000)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-9)


# --- moment-wrapper backend parity: the gl moment engine via the public API ---
_R0, _Z0 = 0.9, 0.08


def _moment_calls(q):
    return {
        "density": q.density(_R0, _Z0, use_physical=False),
        "sigmaR2": q.sigmaR2(_R0, _Z0, use_physical=False),
        "sigmaz2": q.sigmaz2(_R0, _Z0, use_physical=False),
        "sigmaRz": q.sigmaRz(_R0, _Z0, use_physical=False),
        "sigmaT2": q.sigmaT2(_R0, _Z0, use_physical=False),
        "meanvT": q.meanvT(_R0, _Z0, use_physical=False),
        "meanvz": q.meanvz(_R0, _Z0, use_physical=False),
        "tilt": q.tilt(_R0, _Z0, use_physical=False),
        "surfacemass_z": q.surfacemass_z(_R0, use_physical=False),
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_wrappers_parity(backend):
    # the moment engine's gl path through the public wrappers -- exercises the
    # sigma*/meanv*/tilt(xp.arctan)/surfacemass_z(backend.quadrature.fixed_quad)
    # branches on the backend and their value parity with numpy.
    ref = {k: float(as_numpy(v)) for k, v in _moment_calls(_qdf).items()}
    with galpy.backend.use(backend, force=True):
        got = _moment_calls(_qdf)
    for k, v in got.items():
        assert is_backend_array(v), f"{k} not a backend array"
        numpy.testing.assert_allclose(
            float(as_numpy(v)), ref[k], rtol=1e-6, atol=1e-9, err_msg=k
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_array_R_parity(backend):
    # array-R density -> the _vmomentdensity ndim-guard per-scalar recursion +
    # xp.stack collection branch (numpy path uses numpy.array).
    Rs = numpy.array([0.8, 1.0, 1.2])
    zs = numpy.array([0.05, 0.1, 0.02])
    ref = as_numpy(_qdf.density(Rs, zs, use_physical=False))
    with galpy.backend.use(backend, force=True):
        got = _qdf.density(Rs, zs, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6, atol=1e-9)


# --- sampler numpy boundary -------------------------------------------------
# The rejection samplers are numpy end to end (numpy.random proposals, numpy
# fancy-indexing, a numpy output array), but under a forced backend the DF value
# they accept/reject on comes back as a backend array, and `Tensor > ndarray`
# raises. quasiisothermaldf lands the DF value (and fmin_powell's optimum) on
# numpy where they enter that arithmetic. What the sampler returns is therefore
# plain numpy on every backend -- that is the contract these tests pin.
#
# The samples are NOT bit-identical to the numpy path: fmin_powell converges in a
# slightly different number of evaluations on a backend (46 vs 37 here), so maxVT
# differs in its last bits and shifts the proposals. Measured max |difference|
# over a 12-sample draw is 4.5e-10, so 1e-8 is a ~20x margin -- tight enough that
# a real regression in the boundary (wrong values, wrong draw order) cannot pass.
@pytest.mark.parametrize("backend", BACKENDS)
def test_sampleV_returns_numpy(backend):
    numpy.random.seed(17)
    ref = _qdf.sampleV(0.9, 0.05, n=12, use_physical=False)
    with galpy.backend.use(backend, force=True):
        numpy.random.seed(17)
        got = _qdf.sampleV(0.9, 0.05, n=12, use_physical=False)
    assert isinstance(got, numpy.ndarray), (
        f"sampleV under forced {backend} must return a numpy array, got {type(got)}"
    )
    assert not is_backend_array(got)
    assert got.shape == ref.shape == (12, 3)
    numpy.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sampleV_interpolate_returns_numpy(backend):
    # sampleV_interpolate drives the second sampler, _sampleV_preoptimized, whose
    # accept step is a separate site from sampleV's.
    # the (R, z) spread has to cover several R_pixel/z_pixel cells, or the
    # internal RectBivariateSpline gets a degenerate grid
    Rs = numpy.linspace(0.7, 1.3, 40)
    zs = numpy.linspace(0.02, 0.3, 40)
    numpy.random.seed(23)
    ref = _qdf.sampleV_interpolate(Rs, zs, 0.1, 0.05, use_physical=False)
    with galpy.backend.use(backend, force=True):
        numpy.random.seed(23)
        got = _qdf.sampleV_interpolate(Rs, zs, 0.1, 0.05, use_physical=False)
    assert isinstance(got, numpy.ndarray), (
        f"sampleV_interpolate under forced {backend} must return numpy, got {type(got)}"
    )
    assert not is_backend_array(got)
    assert got.shape == ref.shape == (40, 3)
    numpy.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-8)
