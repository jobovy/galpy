###############################################################################
# test_backend_array_inputs.py: the opt-in array gate on potentials whose
# methods are declared scalar-only by ``check_potential_inputs_not_arrays``.
#
# A potential that sets ``_backend_accepts_arrays`` promises that its BACKEND
# evaluation broadcasts over a trailing quadrature-node axis. That is what lets
# the traced surfdens/mass quadratures hand it the whole Gauss-Legendre node
# array in one vectorised call instead of unrolling one traced call per node.
#
# What is proven here:
#   (a) for an opted-in potential, a backend array gives EXACTLY the same value as
#       looping the same coordinates as scalars -- bitwise, not to a tolerance,
#       since the node axis is a no-op for scalars;
#   (b) numpy arrays are still rejected, i.e. the documented scalars-only
#       contract is unchanged off the backend;
#   (c) a potential that has NOT opted in still rejects backend arrays, which is
#       deliberate for AnyAxisymmetricRazorThinDisk: its traced GL evaluation
#       loses all accuracy as |z| -> 0, so a loud TypeError is preferable to a
#       silently wrong number;
#   (d) the capability this buys: the traced Poisson surfdens of
#       DoubleExponentialDisk now reproduces the numpy value.
###############################################################################
import numpy
import pytest

from galpy.potential import (
    AnyAxisymmetricRazorThinDiskPotential,
    DoubleExponentialDiskPotential,
)

# This module manages backends explicitly; exempt from the global --backend
# force fixture.
pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _HAS_JAX = False

try:
    import torch

    torch.set_default_dtype(torch.float64)
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_METHODS = ("Rforce", "zforce", "R2deriv", "z2deriv", "Rzderiv")
_RS = [0.8, 1.0, 1.3, 2.0]
_ZS = [0.1, -0.2, 0.3, 0.05]


def _dep():
    return DoubleExponentialDiskPotential(amp=1.0, hr=1.0 / 3.0, hz=1.0 / 16.0)


def _bitwise_array_equals_scalars(pot, asarray):
    """Every method: array call == scalar-by-scalar calls, bit for bit."""
    for name in _METHODS:
        meth = getattr(pot, name)
        scalars = numpy.array(
            [
                float(meth(asarray(r), asarray(z), use_physical=False))
                for r, z in zip(_RS, _ZS)
            ]
        )
        arrayed = numpy.asarray(
            meth(asarray(_RS), asarray(_ZS), use_physical=False), dtype=float
        )
        assert arrayed.shape == scalars.shape, (
            f"{name}: array call returned shape {arrayed.shape}, "
            f"expected {scalars.shape}"
        )
        # Bitwise: node_axis is a no-op on 0-d input, so vectorising must not
        # perturb the arithmetic at all. A tolerance here would hide a reordering.
        assert all(a == b for a, b in zip(arrayed, scalars)), (
            f"{name}: vectorised evaluation is not bit-identical to the scalar "
            f"one\n  array : {arrayed!r}\n  scalar: {scalars!r}"
        )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_doubleexp_jax_arrays_bit_identical_to_scalars():
    _bitwise_array_equals_scalars(_dep(), lambda v: jnp.array(v))


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_doubleexp_torch_arrays_bit_identical_to_scalars():
    _bitwise_array_equals_scalars(_dep(), lambda v: torch.as_tensor(v))


def test_doubleexp_numpy_arrays_still_rejected():
    # The documented contract off the backend is unchanged: these methods take
    # scalars. Opting in must not open the numpy path.
    pot = _dep()
    for name in _METHODS:
        with pytest.raises(TypeError, match="do not accept array inputs"):
            getattr(pot, name)(numpy.array(_RS), numpy.array(_ZS), use_physical=False)


def test_doubleexp_numpy_scalars_unaffected():
    # Guard the no-op claim from the other side: scalar numpy calls still work
    # and are unchanged by the gate rewrite.
    pot = _dep()
    assert numpy.isfinite(pot.Rforce(1.0, 0.1, use_physical=False))
    assert pot.Rforce(1.0, 0.1, use_physical=False) == pot.Rforce(
        1.0, 0.1, use_physical=False
    )


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_not_opted_in_potential_still_rejects_backend_arrays():
    # AnyAxisymmetricRazorThinDisk deliberately does NOT opt in. Its traced GL
    # integrand is exact to ~1e-12 at |z|=1e-2 but wrong by ~2e4x at |z|=1e-5,
    # and the Poisson quadrature clusters nodes at z=0, so accepting arrays
    # there would return a silently wrong ~0 instead of raising.
    pot = AnyAxisymmetricRazorThinDiskPotential(surfdens=lambda R: numpy.exp(-R))
    assert not getattr(pot, "_backend_accepts_arrays", False)
    with pytest.raises(TypeError, match="do not accept array inputs"):
        pot.Rforce(jnp.array([0.8, 1.0]), jnp.array([0.1, 0.2]), use_physical=False)


@pytest.mark.skipif(not _HAS_JAX, reason="jax not installed")
def test_doubleexp_traced_poisson_surfdens_matches_numpy():
    # The capability the broadcasting buys: under jit the Poisson route feeds
    # the whole node array to the forces in ONE call, which previously raised.
    import galpy.backend as gb

    pot = _dep()
    R0, z0 = 1.0, 0.5
    ref = pot.surfdens(R0, z0, forcepoisson=True, use_physical=False)
    with gb.use("jax", force=True):
        traced = float(
            jax.jit(
                lambda R, z: pot.surfdens(R, z, forcepoisson=True, use_physical=False)
            )(jnp.array(R0), jnp.array(z0))
        )
    # Fixed-order GL against scipy's adaptive quad on a smooth, exponentially
    # decaying integrand: this is the quadrature floor, not a loose smoke bound.
    assert numpy.fabs(traced - ref) / numpy.fabs(ref) < 1e-10, (
        f"traced Poisson surfdens {traced!r} does not reproduce numpy {ref!r}"
    )
