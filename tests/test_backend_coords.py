###############################################################################
# test_backend_coords.py: multi-backend tests for the vector/coordinate leaf
# transforms in galpy.util.coords that the stream DFs, impulse kernels, and
# stream-track accessors sit on:
#
#   rect_to_cyl_vec  (rectangular -> cylindrical velocity vectors)
#   cyl_to_rect_vec  (cylindrical -> rectangular velocity vectors)
#   spher_to_cyl     (spherical -> cylindrical positions)
#
# Before this migration each did raw numpy.cos/numpy.sin on its argument, which
# raised TypeError the moment a torch tensor flowed in -- the confirmed root
# cause of the torch reds across streamspraydf / streamgapdf / streamTrack. Now
# they dispatch through get_namespace + promote_scalars, so this proves numpy /
# jax / torch value parity and that they are differentiable end-to-end (grad of
# an output component w.r.t. an input matches the closed form).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import as_numpy
from galpy.util import coords

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

BACKENDS = ["numpy"]
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

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]

_rng = numpy.random.default_rng(20260707)
_N = 6
_VX, _VY, _VZ = _rng.normal(size=_N), _rng.normal(size=_N), _rng.normal(size=_N)
_X, _Y, _Z = _rng.normal(size=_N), _rng.normal(size=_N), _rng.normal(size=_N)
_VR, _VT = _rng.normal(size=_N), _rng.normal(size=_N)
_R = numpy.abs(_rng.normal(size=_N)) + 0.5
_THETA = _rng.uniform(0.1, numpy.pi - 0.1, _N)
_PHI = _rng.uniform(-3.0, 3.0, _N)


def _asarray(backend_name, x):
    if backend_name == "numpy":
        return numpy.asarray(x, dtype=float)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_vector_transforms_value_parity(backend_name):
    A = lambda a: _asarray(backend_name, a)
    cases = [
        (
            coords.rect_to_cyl_vec(
                numpy.asarray(_VX),
                numpy.asarray(_VY),
                numpy.asarray(_VZ),
                numpy.asarray(_X),
                numpy.asarray(_Y),
                numpy.asarray(_Z),
            ),
            coords.rect_to_cyl_vec(A(_VX), A(_VY), A(_VZ), A(_X), A(_Y), A(_Z)),
            "rect_to_cyl_vec",
        ),
        (
            coords.rect_to_cyl_vec(
                numpy.asarray(_VX),
                numpy.asarray(_VY),
                numpy.asarray(_VZ),
                numpy.asarray(_R),
                numpy.asarray(_PHI),
                numpy.asarray(_Z),
                cyl=True,
            ),
            coords.rect_to_cyl_vec(
                A(_VX), A(_VY), A(_VZ), A(_R), A(_PHI), A(_Z), cyl=True
            ),
            "rect_to_cyl_vec[cyl]",
        ),
        (
            coords.cyl_to_rect_vec(
                numpy.asarray(_VR),
                numpy.asarray(_VT),
                numpy.asarray(_VZ),
                numpy.asarray(_PHI),
            ),
            coords.cyl_to_rect_vec(A(_VR), A(_VT), A(_VZ), A(_PHI)),
            "cyl_to_rect_vec",
        ),
        (
            coords.spher_to_cyl(
                numpy.asarray(_R), numpy.asarray(_THETA), numpy.asarray(_PHI)
            ),
            coords.spher_to_cyl(A(_R), A(_THETA), A(_PHI)),
            "spher_to_cyl",
        ),
    ]
    for ref, got, label in cases:
        for a, b in zip(ref, got):
            numpy.testing.assert_allclose(
                as_numpy(b),
                numpy.asarray(a),
                rtol=1e-13,
                atol=1e-14,
                err_msg=f"{label} ({backend_name})",
            )


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_scalar_inputs(backend_name):
    # A plain python float must survive promotion on every backend (torch's
    # cos/sin reject bare floats -> promote_scalars must lift them).
    R, z, phi = coords.spher_to_cyl(
        _asarray(backend_name, 2.0) if backend_name != "numpy" else 2.0, 1.0, 0.5
    )
    numpy.testing.assert_allclose(as_numpy(R), 2.0 * numpy.sin(1.0), rtol=1e-13)
    numpy.testing.assert_allclose(as_numpy(z), 2.0 * numpy.cos(1.0), rtol=1e-13)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_grad_through(backend_name):
    # spher_to_cyl: dR/dr = sin(theta); cyl_to_rect_vec: dvx/dvr = cos(phi);
    # rect_to_cyl_vec: dvr/dvx = cos(phi) with phi = arctan2(Y, X).
    theta0, phi0, X0, Y0 = 0.7, 0.9, 1.0, 2.0
    if backend_name == "jax":
        gR = jax.grad(
            lambda r: coords.spher_to_cyl(r, jnp.asarray(theta0), jnp.asarray(phi0))[0]
        )(jnp.asarray(2.0))
        gvx = jax.grad(
            lambda vr: coords.cyl_to_rect_vec(
                vr, jnp.asarray(0.4), jnp.asarray(0.5), jnp.asarray(phi0)
            )[0]
        )(jnp.asarray(0.3))
        gvr = jax.grad(
            lambda vx: coords.rect_to_cyl_vec(
                vx,
                jnp.asarray(0.2),
                jnp.asarray(0.3),
                jnp.asarray(X0),
                jnp.asarray(Y0),
                jnp.asarray(3.0),
            )[0]
        )(jnp.asarray(0.1))
        gR, gvx, gvr = float(gR), float(gvx), float(gvr)
    else:
        r = torch.tensor(2.0, requires_grad=True)
        coords.spher_to_cyl(r, torch.tensor(theta0), torch.tensor(phi0))[0].backward()
        gR = float(r.grad)
        vr = torch.tensor(0.3, requires_grad=True)
        coords.cyl_to_rect_vec(
            vr, torch.tensor(0.4), torch.tensor(0.5), torch.tensor(phi0)
        )[0].backward()
        gvx = float(vr.grad)
        vx = torch.tensor(0.1, requires_grad=True)
        coords.rect_to_cyl_vec(
            vx,
            torch.tensor(0.2),
            torch.tensor(0.3),
            torch.tensor(X0),
            torch.tensor(Y0),
            torch.tensor(3.0),
        )[0].backward()
        gvr = float(vx.grad)
    numpy.testing.assert_allclose(gR, numpy.sin(theta0), rtol=1e-10)
    numpy.testing.assert_allclose(gvx, numpy.cos(phi0), rtol=1e-10)
    numpy.testing.assert_allclose(gvr, numpy.cos(numpy.arctan2(Y0, X0)), rtol=1e-10)
