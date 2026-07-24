import numpy
import pytest

from galpy.backend import backend_kernel


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
