###############################################################################
# test_backend_coerce.py: backend data-coercion helpers (galpy.backend._coerce).
#
# Focus: promote_scalars must coerce all-numpy / all-python operands onto the
# active backend when there is NO backend array to anchor on but the resolved
# namespace is non-numpy (a forced default, or an array-API call). torch's
# functions reject numpy.float64 / python floats, so a pass-through there breaks
# every promote_scalars caller (the coords transforms, OblateStaeckel wrapper)
# under a forced torch backend. The numpy path stays an object-identical
# pass-through (byte-identical).
###############################################################################
import numpy
import pytest

from galpy import backend
from galpy.backend import (
    as_backend_constant,
    as_numpy,
    is_backend_array,
    promote_scalars,
)
from galpy.backend._namespaces import default_device

# This module manages backends explicitly, so it is exempt from the global force.
pytestmark = pytest.mark.backend_managed

_NS = {"numpy": numpy}
try:
    import jax

    jax.config.update("jax_enable_x64", True)

    _NS["jax"] = jax
except ImportError:  # pragma: no cover
    pass
try:
    import torch

    _NS["torch"] = torch
except ImportError:  # pragma: no cover
    pass

AD_BACKENDS = [b for b in _NS if b != "numpy"]


def test_promote_scalars_numpy_is_object_identical_passthrough():
    # numpy backend: strict object-identical pass-through (byte-identical path).
    a = numpy.array([1.0, 2.0])
    out = promote_scalars(numpy, a, 1.0, 2, None)
    assert out == (a, 1.0, 2, None)
    assert out[0] is a  # no copy


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_promote_scalars_coerces_all_numpy_under_forced_backend(backend_name):
    # The regression: no backend array among the inputs, but xp resolves to the
    # forced non-numpy backend. Each numpy.float64 / python scalar must become a
    # backend float64 array (so torch's xp.cos/sqrt/... accept them); previously
    # this branch passed through and torch crashed downstream.
    with backend.use(backend_name, force=True) as xp:
        out = promote_scalars(xp, numpy.float64(1.5), 2.0, 3)
    for v in out:
        assert is_backend_array(v), f"{backend_name}: {v!r} not coerced to backend"
        assert "float64" in str(v.dtype)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_promote_scalars_anchored_path_still_works(backend_name):
    # When a backend array IS present, the scalars anchor on its dtype/device
    # (the pre-existing mixed-input path) -- unchanged by this fix.
    with backend.use(backend_name, force=True) as xp:
        ref = xp.asarray([1.0, 2.0])
        a, b = promote_scalars(xp, ref, 3.0)
    assert a is ref
    # the scalar anchors on the reference array's dtype (not galpy's float64)
    assert is_backend_array(b) and str(b.dtype) == str(ref.dtype)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_coords_transform_all_numpy_under_forced_backend(backend_name):
    # Integration: a coords transform fed plain python/numpy scalars under a
    # forced backend (the exact failure of the migrated RotateAndTilt / Offset /
    # OblateStaeckel wrappers) now returns backend arrays matching numpy.
    from galpy.util import coords

    ref_cr = coords.cyl_to_rect(1.1, 0.4, 0.2)
    ref_rc = coords.rect_to_cyl(0.7, 0.3, 0.2)
    with backend.use(backend_name, force=True):
        got_cr = coords.cyl_to_rect(1.1, 0.4, 0.2)
        got_rc = coords.rect_to_cyl(0.7, 0.3, 0.2)
    for got, ref in ((got_cr, ref_cr), (got_rc, ref_rc)):
        for g, r in zip(got, ref):
            assert is_backend_array(g)
            numpy.testing.assert_allclose(float(g), float(r), rtol=1e-12, atol=1e-14)


# --- as_backend_constant: do not name the default device (galpy #1300) --------
# Asking asarray for an explicit device COMMITS the placement instead of letting
# the backend choose, measured in default_device's docstring at 397.6 us vs
# 134.6 us on jax CPU. coerce_coords already skips the keyword when the anchor
# IS the default device; as_backend_constant did not, so every stored constant
# (144 call sites: streamdf, streamgapdf, sphericaldf, NonInertialFrameForce,
# RotateAndTilt, ...) paid it on every evaluation.


def test_as_backend_constant_numpy_is_object_identical_passthrough():
    # numpy backend: the stored constant is returned untouched (byte-identical).
    value = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    out = as_backend_constant(numpy, value, numpy.array(1.0))
    assert out is value


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_as_backend_constant_does_not_name_the_default_device(backend_name):
    # The fix: for a ref sitting on the backend's DEFAULT device, asarray must be
    # called without a device= keyword. Asserted by spying on the namespace's
    # asarray, so the test fails if the guard is removed.
    value = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    seen = []
    with backend.use(backend_name, force=True) as xp:
        ref = xp.asarray(1.0)
        assert getattr(ref, "device", None) == default_device(xp), (
            f"{backend_name}: ref is not on the default device, test is vacuous"
        )
        real = xp.asarray

        def spy(*args, **kwargs):
            seen.append(kwargs.get("device", "ABSENT"))
            return real(*args, **kwargs)

        try:
            xp.asarray = spy
            as_backend_constant(xp, value, ref)
        finally:
            xp.asarray = real
    assert seen, f"{backend_name}: as_backend_constant did not call asarray"
    assert all(d in ("ABSENT", None) for d in seen), (
        f"{backend_name}: default device was named explicitly: {seen}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_as_backend_constant_preserves_values_dtype_and_device(backend_name):
    # Skipping the device keyword must not move the array or perturb a bit: the
    # constant is compared EXACTLY (no tolerance) against the stored numpy value.
    value = numpy.array([[1.0, -2.5], [3.25, 4.0e-17]])
    with backend.use(backend_name, force=True) as xp:
        ref = xp.asarray(numpy.float64(1.0))
        out = as_backend_constant(xp, value, ref)
        assert is_backend_array(out)
        assert out.dtype == ref.dtype
        assert getattr(out, "device", None) == getattr(ref, "device", None)
        assert numpy.array_equal(as_numpy(out), value)
