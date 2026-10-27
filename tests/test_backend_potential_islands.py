###############################################################################
# test_backend_potential_islands.py: potential-level functions that computed on
# numpy/scipy under a forced backend now run on the backend -- the root finders
# (rl, rE, zvc, zvc_range, rvir, rhalf, lindbladR), the Lin & Shu reduction
# factor, the Coulomb logarithm, SpiralArms' K, Irrgang13's halo density, and
# the quasi-isothermal DF's scale-length estimates. Values match numpy;
# derivatives match closed forms or converged finite differences.
###############################################################################
import numpy
import pytest

from galpy import potential
from galpy.backend import as_numpy, get_namespace, is_backend_array, use

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


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _grad(backend, f, x0):
    """df/dx at the float x0 through the backend's autodiff"""
    if backend == "jax":
        return float(jax.grad(lambda x: jnp.reshape(f(x), ()))(jnp.asarray(x0)))
    x = torch.tensor(x0, requires_grad=True)
    (g,) = torch.autograd.grad(torch.reshape(f(x), ()), x)
    return float(g)


_MW = potential.MWPotential2014
_LP = potential.LogarithmicHaloPotential(normalize=1.0)
_LZ = numpy.array([0.3, 0.8, 1.1, 1.6])
_E = numpy.array([-1.5, -1.3, -1.2, -1.0])
# forced, array inputs: one vectorized backend solve
_FORCED_ARRAYS = [
    ("rl", lambda: potential.rl(_MW, _LZ), lambda: [potential.rl(_MW, x) for x in _LZ]),
    ("rE", lambda: potential.rE(_MW, _E), lambda: [potential.rE(_MW, x) for x in _E]),
    (
        "LcE",
        lambda: potential.LcE(_MW, _E),
        lambda: [potential.LcE(_MW, x) for x in _E],
    ),
    (
        "zvc",
        lambda: potential.zvc(_MW, numpy.array([0.6, 0.7, 0.8]), -1.25, 0.6),
        lambda: [potential.zvc(_MW, R, -1.25, 0.6) for R in (0.6, 0.7, 0.8)],
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,call,ref", _FORCED_ARRAYS, ids=[n for n, _, _ in _FORCED_ARRAYS]
)
def test_forced_array_inputs_solve_on_the_backend(backend, name, call, ref):
    ref = numpy.array(ref())
    with use(backend, force=True):
        got = call()
    assert is_backend_array(got), f"{name}: {type(got)}"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-10)


_SCALARS = [
    ("rl", lambda: potential.rl(_MW, 1.1)),
    ("rE", lambda: potential.rE(_MW, -1.2)),
    ("zvc", lambda: potential.zvc(_MW, 0.7, -1.25, 0.6)),
    ("zvc_range", lambda: potential.zvc_range(_MW, -1.25, 0.6)),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,call", _SCALARS, ids=[n for n, _ in _SCALARS])
def test_forced_plain_scalars_solve_on_the_backend(backend, name, call):
    # a forced default with plain scalar inputs: the backend root, numpy's value
    ref = call()
    with use(backend, force=True):
        got = call()
    assert is_backend_array(got), f"{name}: {type(got)}"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11)


def _bk_param(backend, x):
    return _arr(backend, x)


_PARAM = [
    (
        "rl",
        lambda b: potential.rl(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)), 1.1
        ),
        lambda: potential.rl(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9), 1.1
        ),
    ),
    (
        "rE",
        lambda b: potential.rE(
            potential.HernquistPotential(a=_bk_param(b, 2.0), normalize=1.0), -0.8
        ),
        lambda: potential.rE(potential.HernquistPotential(a=2.0, normalize=1.0), -0.8),
    ),
    (
        "zvc",
        lambda b: potential.zvc(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)),
            0.7,
            -0.4,
            0.6,
        ),
        lambda: potential.zvc(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9), 0.7, -0.4, 0.6
        ),
    ),
    (
        "zvc_range",
        lambda b: potential.zvc_range(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)),
            -0.4,
            0.6,
        ),
        lambda: potential.zvc_range(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9), -0.4, 0.6
        ),
    ),
    (
        "rhalf",
        lambda b: potential.rhalf(potential.HernquistPotential(a=_bk_param(b, 2.0))),
        lambda: potential.rhalf(potential.HernquistPotential(a=2.0)),
    ),
    (
        "rvir",
        lambda b: potential.NFWPotential(a=_bk_param(b, 2.0), normalize=0.35).rvir(),
        lambda: potential.NFWPotential(a=2.0, normalize=0.35).rvir(),
    ),
    (
        "lindbladR",
        lambda b: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)),
            0.5,
            m=2,
        ),
        lambda: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9), 0.5, m=2
        ),
    ),
    (
        "corotation",
        lambda b: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)),
            0.5,
            m="corotation",
        ),
        lambda: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9),
            0.5,
            m="corotation",
        ),
    ),
    (
        "outerlindblad",
        lambda b: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=_bk_param(b, 0.9)),
            0.5,
            m=-2,
        ),
        lambda: potential.lindbladR(
            potential.LogarithmicHaloPotential(normalize=1.0, q=0.9), 0.5, m=-2
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name,call,ref", _PARAM, ids=[n for n, _, _ in _PARAM])
def test_backend_parameter_solves_on_the_backend(backend, name, call, ref):
    # a backend parameter turns the backend on, also at plain scalar inputs
    got = call(backend)
    assert is_backend_array(got), f"{name}: {type(got)}"
    numpy.testing.assert_allclose(as_numpy(got), ref(), rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zvc_backend_no_solution_and_unbound(backend):
    # no solution at R (Phi + Lz^2/2R^2 > E at z=0): nan, as numpy
    assert numpy.isnan(potential.zvc(_MW, 0.1, -1.25, 0.6))
    assert numpy.isnan(
        float(as_numpy(potential.zvc(_MW, _arr(backend, 0.1), -1.25, 0.6)))
    )
    # E above Phi at every z: no curve, a ValueError as numpy's
    with pytest.raises(ValueError, match="No solution for the zero-velocity"):
        potential.zvc(_MW, _arr(backend, 0.7), 198.75, 0.6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lindbladR_backend_no_resonance_is_none(backend):
    # corotation beyond the bracket (R = 1/OmegaP = 1e4 > 1000): None, after
    # numpy's retry from 0.01 found no sign change either
    assert potential.lindbladR(_LP, 1e-4, m="corotation") is None
    assert potential.lindbladR(_LP, _arr(backend, 1e-4), m="corotation") is None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m", [2, -2, "corotation"])
def test_lindbladR_grad_vs_fd(backend, m):
    # d R_res / d OmegaP: implicit differentiation of the resonance condition
    h = 1e-6
    fd = (
        potential.lindbladR(_LP, 0.5 + h, m=m) - potential.lindbladR(_LP, 0.5 - h, m=m)
    ) / (2.0 * h)
    g = _grad(backend, lambda x: potential.lindbladR(_LP, x, m=m), 0.5)
    numpy.testing.assert_allclose(g, fd, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rhalf_grad_closed_form(backend):
    # Hernquist: rhalf = (1 + sqrt 2) a, so d rhalf / d a = 1 + sqrt 2
    def f(a):
        return potential.HernquistPotential(amp=1.0, a=a).rhalf()

    numpy.testing.assert_allclose(
        _grad(backend, f, numpy.pi), 1.0 + numpy.sqrt(2.0), rtol=1e-10
    )


def _linshu_args():
    R, m, sr = 0.9, 2.0, 0.1
    tepi, tomegac = potential.epifreq(_LP, R), potential.omegac(_LP, R)
    OmegaP = tepi * numpy.sqrt(0.2) / m + tomegac
    k = numpy.sqrt(4.0) * tepi / sr
    return R, sr, m, k, OmegaP


@pytest.mark.parametrize("backend", BACKENDS)
def test_LinShuReductionFactor_backend_value_and_grad(backend):
    # the Gauss-Legendre integral on the backend: scipy's value, and
    # d F / d sigma_R against a central finite difference
    R, sr, m, k, OmegaP = _linshu_args()

    def F(sigmar):
        return potential.LinShuReductionFactor(_LP, R, sigmar, m=m, k=k, OmegaP=OmegaP)

    ref = F(sr)
    with use(backend, force=True):
        got = F(sr)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-13)
    h = 1e-6
    fd = (F(sr + h) - F(sr - h)) / (2.0 * h)
    numpy.testing.assert_allclose(_grad(backend, F, sr), fd, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rhm", [0.0, 0.125])
def test_lnLambda_backend(backend, rhm):
    cdf = potential.ChandrasekharDynamicalFrictionForce(
        GMs=0.01, rhm=rhm, dens=_LP, sigmar=lambda r: 1.0 / numpy.sqrt(2.0)
    )
    for r, v in [(1.0, 1.0), (0.3, 0.05)]:
        ref = cdf.lnLambda(r, v)
        got = cdf.lnLambda(_arr(backend, r), _arr(backend, v))
        assert is_backend_array(got)
        numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["_K", "_dK_dR"])
def test_spiralarms_K_default_ns_on_backend(backend, fn):
    sp = potential.SpiralArmsPotential(Cs=[1.0, 0.5, 0.2])
    ref = getattr(sp, fn)(0.9)
    got = getattr(sp, fn)(_arr(backend, 0.9))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_irrgang13_halo_density_on_backend(backend):
    from galpy.potential.Irrgang13 import Irrgang13I_halo_dens

    r = numpy.array([0.1, 1.0, 20.0, 25.0, 30.0])  # across the tanh cut-off
    got = Irrgang13I_halo_dens(_arr(backend, r))
    assert is_backend_array(got)
    # 1 - tanh in the cut-off amplifies the last-digit tanh differences
    numpy.testing.assert_allclose(as_numpy(got), Irrgang13I_halo_dens(r), rtol=1e-11)


_QDF_ESTIMATES = [
    ("estimate_hr", dict(z=0.0)),
    ("estimate_hr", dict(z=None)),
    ("estimate_hz", dict(z=0.1)),
    ("estimate_hsr", dict(z=0.0)),
    ("estimate_hsz", dict(z=0.0)),
]


@pytest.fixture(scope="module")
def _qdf():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf

    aA = actionAngleAdiabatic(pot=_MW, c=True)
    return quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=_MW, aA=aA, cutcounter=True
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "fn,kw", _QDF_ESTIMATES, ids=[f"{f}-{k['z']}" for f, k in _QDF_ESTIMATES]
)
def test_qdf_estimates_on_backend(backend, _qdf, fn, kw):
    # the finite-difference scale lengths: their logs on the backend
    ref = getattr(_qdf, fn)(0.9, **kw, use_physical=False)
    with use(backend, force=True):
        got = getattr(_qdf, fn)(0.9, **kw, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("precompute", [False, True])
def test_qdf_rg_forced_on_backend(backend, precompute):
    # _rg at numpy Lz under a forced backend solves (and interpolates) on it:
    # rl everywhere without the table, the table plus rl outside it with one
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf

    aA = actionAngleAdiabatic(pot=_MW, c=True)
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=_MW,
        aA=aA,
        cutcounter=True,
        _precomputerg=precompute,
    )
    # with the table, one Lz outside it
    lzs = numpy.array([0.5, 1.1, qdf._precomputergLzmax * 1.5 if precompute else 2.0])
    ref_s, ref_a = qdf._rg(1.1), qdf._rg(lzs)
    with use(backend, force=True):
        s = qdf._rg(1.1)
        a = qdf._rg(lzs)
    assert is_backend_array(s) and is_backend_array(a)
    numpy.testing.assert_allclose(as_numpy(s), ref_s, rtol=1e-11)
    numpy.testing.assert_allclose(as_numpy(a), ref_a, rtol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_qdf_surfacemass_z_adaptive_on_backend(backend, _qdf):
    # fixed_quad=False: the least-squares cubic that scipy's smoothing spline is
    # here (no interior knots), integrated on the backend; d/dR vs a central
    # difference of numpy
    ref = _qdf.surfacemass_z(0.9, fixed_quad=False, use_physical=False)
    with use(backend, force=True):
        got = _qdf.surfacemass_z(0.9, fixed_quad=False, use_physical=False)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-12)
    h = 1e-4
    fd = (
        _qdf.surfacemass_z(0.9 + h, fixed_quad=False, use_physical=False)
        - _qdf.surfacemass_z(0.9 - h, fixed_quad=False, use_physical=False)
    ) / (2.0 * h)
    grad = _grad(
        backend,
        lambda R: _qdf.surfacemass_z(R, fixed_quad=False, use_physical=False),
        0.9,
    )
    numpy.testing.assert_allclose(grad, fd, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_qdf_smoothing_cubic_misfit_takes_scipy(backend):
    # a log-density the cubic misfits by more than the smoothing: scipy's spline
    from scipy import integrate, interpolate

    from galpy.df.quasiisothermaldf import _int_exp_smoothing_cubic

    zs = numpy.linspace(0.0, 1.0, 7)
    lsf = 10.0 * numpy.sin(12.0 * zs)
    spline = interpolate.UnivariateSpline(zs, lsf, k=3)
    ref = integrate.quad(lambda x: numpy.exp(spline(x)), 0.0, 1.0)[0]
    lsf_bk = _arr(backend, lsf)
    got = _int_exp_smoothing_cubic(get_namespace(lsf_bk), zs, lsf_bk)
    assert is_backend_array(got)
    assert float(as_numpy(got)) == ref
