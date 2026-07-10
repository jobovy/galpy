###############################################################################
# test_backend_streamspraydf.py: multi-backend tests for the stream particle-
# spray DFs (fardal15spraydf / chen24spraydf).
#
# The sampler core (_sample_tail / _setup_rot / _calc_rtide / _calc_vc / both
# spray_df) is backend-agnostic: it resolves its namespace from the context and
# coerces the (exogenous, numpy) RNG draws onto the active backend, so the
# deterministic transforms run under numpy / jax / torch. The RNG itself stays
# numpy on every backend, so seeding numpy identically before each backend's
# sample makes the draws bit-identical and only the transform arithmetic differs
# in floating point -- hence we compare the actual integrate=False sample arrays
# at a tight rtol (not just summary statistics).
#
# integrate=True is exercised by test_sample_integrate_parity: under a backend the
# per-particle 2D-time-grid Orbit.integrate routes to the differentiable C-STM
# (dop853_c on a per-orbit 2-point [-dt_i, 0] grid), matching the numpy path.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy, is_backend_array, use
from galpy.df import chen24spraydf, fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)

    BACKENDS.append("jax")
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

_RO, _VO = 8.0, 220.0
_SEED = 20260707


def _build(cls, **kwargs):
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 4.5 / conversion.time_in_Gyr(_VO, _RO)
    return cls(mass, progenitor=obs, pot=lp, tdisrupt=td, **kwargs)


def _sample(df, backend_name, n, **kwargs):
    numpy.random.seed(_SEED)
    with use(backend_name, force=True):
        return df.sample(n=n, return_orbit=False, integrate=False, **kwargs)


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_array_parity(cls, backend_name):
    # Same numpy RNG seed on every backend -> identical draws -> the sampled
    # (R, vR, vT, z, vz, phi) arrays match the numpy path up to the tiny FP
    # differences of the deterministic transforms.
    df = _build(cls, tail="leading")
    ref = _sample(df, "numpy", 300)
    got = _sample(df, backend_name, 300)
    if backend_name != "numpy":
        assert is_backend_array(got), (
            f"{cls.__name__} sample should be a backend array under {backend_name}"
        )
    numpy.testing.assert_allclose(
        as_numpy(got),
        as_numpy(ref),
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"{cls.__name__} sample parity ({backend_name})",
    )


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_sample_stats_parity(cls, backend_name):
    # Summary-statistics parity (mean/std per phase-space coordinate).
    df = _build(cls, tail="leading")
    ref = as_numpy(_sample(df, "numpy", 1000))
    got = as_numpy(_sample(df, backend_name, 1000))
    numpy.testing.assert_allclose(
        got.mean(axis=1), ref.mean(axis=1), rtol=1e-6, atol=1e-8
    )
    numpy.testing.assert_allclose(
        got.std(axis=1), ref.std(axis=1), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_tail_both_parity(cls, backend_name):
    # tail='both' concatenates leading+trailing sub-samples; the concatenation is
    # backend-aware (xp.hstack for backend arrays, numpy.hstack for numpy).
    df = _build(cls, tail="both")
    ref = _sample(df, "numpy", 200, tail="both")
    got = _sample(df, backend_name, 200, tail="both")
    assert as_numpy(got).shape == (6, 200)
    numpy.testing.assert_allclose(as_numpy(got), as_numpy(ref), rtol=1e-6, atol=1e-8)


def _sample_integ(df, backend_name, n, **kwargs):
    numpy.random.seed(_SEED)
    with use(backend_name, force=True):
        return df.sample(n=n, return_orbit=False, integrate=True, **kwargs)


@pytest.mark.parametrize("cls", [fardal15spraydf, chen24spraydf])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sample_integrate_parity(cls, backend_name):
    # integrate=True: the per-particle sample orbits are integrated to the present
    # day. Under a backend the per-orbit (N, nt) integration routes to the
    # differentiable C-STM (an RK dxdv-C method, dop853_c) and the result is a
    # backend array matching the numpy path (which uses the fixed-step default
    # symplec4_c) up to the two integrators' agreement (~1e-8).
    df = _build(cls, tail="leading")
    ref = _sample_integ(df, "numpy", 200)
    got = _sample_integ(df, backend_name, 200)
    if backend_name != "numpy":
        assert is_backend_array(got), (
            f"{cls.__name__} integrated sample should be a backend array "
            f"under {backend_name}"
        )
    numpy.testing.assert_allclose(
        as_numpy(got),
        as_numpy(ref),
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"{cls.__name__} integrate=True parity ({backend_name})",
    )
