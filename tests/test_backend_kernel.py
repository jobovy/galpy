import numpy
import pytest

from galpy.backend import backend_kernel

# These tests manage their own backend (backend_managed), so the --backend fixture
# that sets float64 precision does not run; galpy's tolerances assume float64.
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None


class _Toy:
    """Minimal carrier exercising the decorator's coord-coercion + xp injection."""

    @backend_kernel("R", "z", "phi")
    def ev(self, R, z, phi=0.0, t=0.0, *, xp=None):
        return xp.sqrt(R**2 + z**2) + xp.sin(phi)

    @backend_kernel("R")
    def one(self, R, *, xp=None):
        return xp.log(R)


def _inline(R, z, phi=0.0):
    return numpy.sqrt(R**2 + z**2) + numpy.sin(phi)


def test_kernel_numpy_byte_identical():
    # numpy path: get_namespace -> numpy, coerce_coords is identity, so the
    # decorated kernel is bit-for-bit the hand-inlined equivalent.
    t = _Toy()
    for R, z, phi in [(1.3, 0.2, 0.0), (0.3, -0.9, 0.5), (5.0, 3.0, 2.1)]:
        got = t.ev(R, z, phi=phi)
        exp = _inline(R, z, phi)
        assert repr(float(got)) == repr(float(exp))


def test_kernel_arg_passing_positional_keyword_default():
    t = _Toy()
    base = t.ev(1.3, 0.2)
    assert t.ev(R=1.3, z=0.2) == base  # keyword
    assert t.ev(1.3, z=0.2) == base  # mixed
    # phi defaulted (sin(0)=0) vs passed explicitly
    assert t.ev(1.3, 0.2, phi=0.0) == base


def test_kernel_requires_xp_param():
    with pytest.raises(TypeError, match="must accept a keyword-only 'xp'"):

        @backend_kernel("R")
        def bad(self, R):
            return R


def test_kernel_unknown_coord_name():
    with pytest.raises(TypeError, match="names no parameter"):

        @backend_kernel("Q")
        def bad(self, R, *, xp=None):
            return R


@pytest.mark.backend_managed
@pytest.mark.parametrize("name", ["jax", "torch"])
def test_kernel_coerces_under_forced_backend(name):
    # Under a forced non-numpy backend the decorator must bring a numpy coord
    # onto the backend so xp.<fn> (torch rejects numpy) succeeds and the result
    # is a backend array -- the latent-torch-bug fix.
    pytest.importorskip(name)
    from galpy import backend
    from galpy.backend import is_backend_array

    t = _Toy()
    with backend.use(name, force=True):
        v = t.one(numpy.float64(2.0))
        assert is_backend_array(v)
        assert numpy.isclose(float(v), numpy.log(2.0))


@pytest.mark.backend_managed
def test_kernel_jit_mode_matches_eager():
    # The jit seam (galpy.backend._kernel._JIT_MODE, driven by the --backend jax-jit
    # test dimension): a decorated kernel traced through jax.jit must return the same
    # value as eager. Exercise the decorator directly (_Toy, multiple declared coords)
    # and a real potential entry point (Hernquist). (torch-compile is deferred: its
    # dynamo cannot trace the more complex kernels.)
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from galpy import backend
    from galpy.backend import _kernel, as_numpy
    from galpy.potential import (
        HernquistPotential,
        evaluatePotentials,
        evaluateRforces,
    )

    R = jnp.asarray(1.3, dtype=jnp.float64)
    z = jnp.asarray(0.2, dtype=jnp.float64)
    phi = jnp.asarray(0.4, dtype=jnp.float64)
    pot = HernquistPotential(amp=1.2, a=0.9)
    with backend.use("jax", force=True):
        calls = [
            lambda: _Toy().ev(R, z, phi=phi),
            lambda: evaluatePotentials(pot, R, z),
            lambda: evaluateRforces(pot, R, z),
        ]
        for call in calls:
            _kernel.set_jit_mode(None)
            eager = call()
            _kernel.set_jit_mode("jax")
            try:
                jitted = call()
            finally:
                _kernel.set_jit_mode(None)
            numpy.testing.assert_allclose(
                as_numpy(jitted), as_numpy(eager), rtol=1e-9, atol=1e-11
            )
