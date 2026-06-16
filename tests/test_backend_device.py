###############################################################################
# test_backend_device.py: device placement on mixed scalar/array coordinate
# inputs. A Python-scalar coordinate must be anchored on the device of a sibling
# backend-array coordinate, not left on the CPU default, or torch raises
# "Expected all tensors to be on the same device" on CUDA. The CPU cases here
# exercise the device-anchoring branch (a CPU torch/jax array still carries a
# .device); the CUDA case (skipped without a GPU) is where it is load-bearing.
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


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _np(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return numpy.asarray(x)


def _scf():
    from galpy.potential import SCFPotential, scf_compute_coeffs_spherical

    Acos, Asin = scf_compute_coeffs_spherical(lambda r: numpy.exp(-r), 5)
    return SCFPotential(Acos=Acos, Asin=Asin)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scf_mixed_scalar_array_coords(backend):
    # SCF with scalar R,z and a backend-ARRAY phi: the scalar coords must follow
    # phi's device. On CPU this just checks parity vs numpy (and covers the
    # device-anchoring branch); under CUDA it is the difference between working
    # and a mixed-device crash.
    from galpy.potential import evaluatephitorques, evaluateRforces, evaluatezforces

    scf = _scf()
    phin = numpy.array([0.1, 0.7, 1.3])
    phib = _arr(backend, [0.1, 0.7, 1.3])
    for name, fn in [
        ("pot", lambda p: scf(1.5, 0.3, phi=p)),
        ("Rforce", lambda p: evaluateRforces(scf, 1.5, 0.3, phi=p)),
        ("zforce", lambda p: evaluatezforces(scf, 1.5, 0.3, phi=p)),
        ("phitorque", lambda p: evaluatephitorques(scf, 1.5, 0.3, phi=p)),
        ("R2deriv", lambda p: scf.R2deriv(1.5, 0.3, phi=p)),
        ("phi2deriv", lambda p: scf.phi2deriv(1.5, 0.3, phi=p)),
    ]:
        ref = numpy.asarray(fn(phin))
        got = _np(fn(phib))
        numpy.testing.assert_allclose(
            got, ref, rtol=1e-10, atol=1e-12, err_msg=f"SCF {name} ({backend})"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cyl_to_spher_mixed(backend):
    # array phi + scalar R,z: r, theta, phi must come back on one namespace and
    # match the numpy values.
    from galpy.util import coords

    phib = _arr(backend, [0.1, 0.2, 0.3])
    r, th, ph = coords.cyl_to_spher(1.5, 0.3, phib)
    r_ref, theta_ref, phi_ref = coords.cyl_to_spher(
        1.5, 0.3, numpy.array([0.1, 0.2, 0.3])
    )
    numpy.testing.assert_allclose(_np(r), numpy.broadcast_to(r_ref, (3,)), rtol=1e-12)
    numpy.testing.assert_allclose(
        _np(th), numpy.broadcast_to(theta_ref, (3,)), rtol=1e-12
    )
    numpy.testing.assert_allclose(_np(ph), phi_ref, rtol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_doubleexp_scalar_R_array_z(backend):
    # DoubleExp._evaluate's scalar-R path. The backend coercion promotes the
    # scalar R to a 0-d array, which must still take the scalar path and return
    # the SAME value numpy does (DoubleExp's scalar-R path returns the R-scalar
    # result, out[0]) -- regression for the 0-d-R reshape error, and a device
    # check (scalar R + backend-array z must not mix devices).
    from galpy.potential import DoubleExponentialDiskPotential as DEDP

    dp = DEDP()
    zb = _arr(backend, [0.3, 0.6])
    ref = numpy.asarray(dp(1.5, numpy.array([0.3, 0.6])))
    numpy.testing.assert_allclose(_np(dp(1.5, zb)), ref, rtol=1e-8, atol=1e-10)
    # supported array combo (same-shape R, z) stays a full array, matching numpy
    Rb, zb2 = _arr(backend, [1.5, 2.0]), _arr(backend, [0.3, 0.6])
    ref2 = numpy.asarray(dp(numpy.array([1.5, 2.0]), numpy.array([0.3, 0.6])))
    numpy.testing.assert_allclose(_np(dp(Rb, zb2)), ref2, rtol=1e-8, atol=1e-10)


def test_promote_scalars_device_reject_fallback():
    # _promote_scalars_for must fall back to a device-less asarray when the
    # namespace rejects the ref's device value (array-api jax exposes .device as
    # the string 'cpu', and jnp.asarray(device='cpu') raises ValueError). Driven
    # deterministically here with a tiny stub so the fallback is covered on every
    # CI runner regardless of the installed jax's .device behaviour.
    from galpy.util.coords import _promote_scalars_for

    class _Xp:
        def asarray(self, v, dtype=None, device=None):
            if device is not None:
                raise ValueError(f"backend rejects device={device!r}")
            return numpy.asarray(v, dtype=dtype)

    class _Ref:
        ndim = 1
        dtype = float
        device = "string-device-the-namespace-rejects"

    ref = _Ref()
    out = _promote_scalars_for(_Xp(), ref, 2.5)
    assert out[0] is ref  # the array passes through untouched
    assert float(out[1]) == 2.5  # the scalar was promoted via the fallback path


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="needs a CUDA torch device",
)
def test_mixed_coords_cross_device_cuda():
    # The load-bearing case: scalar R,z + a CUDA-array phi. Pre-fix these raised
    # "Expected all tensors to be on the same device"; now they return on CUDA.
    from galpy.potential import DoubleExponentialDiskPotential as DEDP
    from galpy.potential import evaluateRforces
    from galpy.util import coords

    scf = _scf()
    phi = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    refp = numpy.asarray(scf(1.5, 0.3, phi=numpy.array([0.1, 0.2, 0.3])))
    out = scf(1.5, 0.3, phi=phi)
    assert out.device.type == "cuda"
    numpy.testing.assert_allclose(out.detach().cpu().numpy(), refp, rtol=1e-10)
    fr = evaluateRforces(scf, 1.5, 0.3, phi=phi)
    assert fr.device.type == "cuda"
    r, th, ph = coords.cyl_to_spher(1.5, 0.3, phi)
    assert r.device.type == "cuda" and th.device.type == "cuda"
    zc = torch.tensor([0.3, 0.6], device="cuda")
    de = DEDP()(1.5, zc)
    assert de.device.type == "cuda"
