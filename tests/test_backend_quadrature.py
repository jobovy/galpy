###############################################################################
# test_backend_quadrature.py: galpy.backend.quadrature -- backend-agnostic
# fixed-order quadrature. Asserts value parity vs scipy.integrate.quad / exact,
# autodiff in the limits AND through integrand parameters, and that the promoted
# gauss_legendre_01 is unchanged (the special-function hyp2f1 fallback uses it).
###############################################################################
import numpy
import pytest

from galpy.backend.quadrature import (
    fixed_quad,
    fixed_quad_semiinfinite,
    gauss_legendre,
    gauss_legendre_01,
    gauss_legendre_nodes,
    nested_quad,
    quad,
    transformed_quad,
)

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
    import array_api_compat.torch as txp

    BACKENDS.append("torch")
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = [b for b in BACKENDS if b != "numpy"]


def _xp(backend):
    return {
        "numpy": numpy,
        "jax": jnp if jax else None,
        "torch": txp if torch else None,
    }[backend]


def test_gauss_legendre_01_unchanged():
    # byte-identical to the leggauss [0,1] remap, and still importable from the
    # special fallback's old path (hyp2f1 uses it).
    for n in (8, 20, 50):
        x, w = numpy.polynomial.legendre.leggauss(n)
        nodes, weights = gauss_legendre_01(n)
        numpy.testing.assert_array_equal(nodes, 0.5 * (x + 1))
        numpy.testing.assert_array_equal(weights, 0.5 * w)
    from galpy.backend.special._fallback._quadrature import gauss_legendre_01 as old

    assert old(20)[0] is gauss_legendre_01(20)[0]  # same cached object


def test_gauss_legendre_nodes_remap():
    nodes, weights = gauss_legendre_nodes(30, 2.0, 5.0)
    # integrate 1 over [2,5] -> 3
    numpy.testing.assert_allclose(numpy.sum(weights), 3.0, rtol=1e-13)
    numpy.testing.assert_allclose(
        numpy.sum(weights * nodes), 0.5 * (25.0 - 4.0), rtol=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_quad_parity(backend):
    xp = _xp(backend)
    # int_0.5^3 exp(-s) ds = exp(-0.5) - exp(-3)
    ref = numpy.exp(-0.5) - numpy.exp(-3.0)
    got = float(numpy.asarray(fixed_quad(xp, lambda s: xp.exp(-s), 0.5, 3.0, n=40)))
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_semiinfinite_parity(backend):
    xp = _xp(backend)
    # int_1^inf exp(-s) ds = exp(-1); int_0^inf 1/(1+s^2) ds = pi/2
    g1 = float(
        numpy.asarray(
            fixed_quad_semiinfinite(xp, lambda s: xp.exp(-s), 1.0, n=100, kind="recip")
        )
    )
    numpy.testing.assert_allclose(g1, numpy.exp(-1.0), rtol=1e-7)
    g2 = float(
        numpy.asarray(
            fixed_quad_semiinfinite(
                xp, lambda s: 1.0 / (1.0 + s**2), 0.0, n=100, kind="tan"
            )
        )
    )
    numpy.testing.assert_allclose(g2, numpy.pi / 2.0, rtol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_quad_parity(backend):
    xp = _xp(backend)
    # int_[0,1]^2 exp(x+y) dx dy = (e-1)^2
    got = float(
        numpy.asarray(
            nested_quad(xp, lambda x, y: xp.exp(x + y), [(0.0, 1.0), (0.0, 1.0)], n=20)
        )
    )
    numpy.testing.assert_allclose(got, (numpy.e - 1.0) ** 2, rtol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_quad_grad_in_limit_and_param(backend):
    xp = _xp(backend)
    a0, p0 = 0.7, 1.3
    # d/da int_a^3 exp(-s) ds = -exp(-a)
    if backend == "jax":
        ga = float(
            jax.grad(lambda a: fixed_quad(jnp, lambda s: jnp.exp(-s), a, 3.0, n=40))(
                jnp.asarray(a0)
            )
        )
        # d/dp int_0^2 exp(-p s) ds = -(d/dp)[(1-exp(-2p))/p]
        gp = float(
            jax.grad(
                lambda p: fixed_quad(jnp, lambda s: jnp.exp(-p * s), 0.0, 2.0, n=60)
            )(jnp.asarray(p0))
        )
    else:
        at = torch.tensor(a0, requires_grad=True)
        fixed_quad(txp, lambda s: txp.exp(-s), at, 3.0, n=40).backward()
        ga = float(at.grad)
        pt = torch.tensor(p0, requires_grad=True)
        fixed_quad(txp, lambda s: txp.exp(-pt * s), 0.0, 2.0, n=60).backward()
        gp = float(pt.grad)
    numpy.testing.assert_allclose(ga, -numpy.exp(-a0), rtol=1e-6)
    # analytic d/dp of (1-exp(-2p))/p
    ref_gp = (2.0 * numpy.exp(-2 * p0) * p0 - (1 - numpy.exp(-2 * p0))) / p0**2
    numpy.testing.assert_allclose(gp, ref_gp, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transformed_quad_interior_split(backend):
    xp = _xp(backend)
    # int_0^2 |s-1|^0.5 ds = 4/3, with a sqrt-kink at the interior point s=1
    got = float(
        numpy.asarray(
            transformed_quad(
                xp, lambda s: xp.abs(s - 1.0) ** 0.5, 0.0, 2.0, n=60, interior_point=1.0
            )
        )
    )
    numpy.testing.assert_allclose(got, 4.0 / 3.0, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transformed_quad_no_interior(backend):
    xp = _xp(backend)
    # interior_point=None falls through to plain fixed_quad: int_0^2 exp(s) = e^2-1
    got = float(
        numpy.asarray(transformed_quad(xp, lambda s: xp.exp(s), 0.0, 2.0, n=40))
    )
    numpy.testing.assert_allclose(got, numpy.exp(2.0) - 1.0, rtol=1e-10)


def test_boundary_layer_remap_identity():
    # k == 1 is the identity map: nodes/weights returned unchanged (same objects).
    from galpy.backend.quadrature import _boundary_layer_remap

    x01, w01 = gauss_legendre_01(12)
    X, wX = _boundary_layer_remap(numpy, x01, w01, 1.0)
    assert X is x01 and wX is w01


@pytest.mark.parametrize("backend", BACKENDS)
def test_nested_quad_per_dim_n(backend):
    xp = _xp(backend)
    # per-dimension n list: int_[0,1]x[0,2] exp(x+y) dx dy = (e-1)(e^2-1)
    got = float(
        numpy.asarray(
            nested_quad(
                xp, lambda x, y: xp.exp(x + y), [(0.0, 1.0), (0.0, 2.0)], n=[20, 30]
            )
        )
    )
    ref = (numpy.e - 1.0) * (numpy.exp(2.0) - 1.0)
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_semiinfinite_grad_in_limit(backend):
    # d/da int_a^inf exp(-s) ds = -exp(-a), through the recip semi-infinite map.
    a0 = 0.7
    if backend == "jax":
        ga = float(
            jax.grad(
                lambda a: fixed_quad_semiinfinite(
                    jnp, lambda s: jnp.exp(-s), a, n=120, kind="recip"
                )
            )(jnp.asarray(a0))
        )
    else:
        at = torch.tensor(a0, requires_grad=True)
        fixed_quad_semiinfinite(
            txp, lambda s: txp.exp(-s), at, n=120, kind="recip"
        ).backward()
        ga = float(at.grad)
    numpy.testing.assert_allclose(ga, -numpy.exp(-a0), rtol=1e-5)


# ---------------------------------------------------------------------------
# Public definite-integral API: quad / gauss_legendre. numpy -> scipy (byte-
# identical value); jax/torch -> fixed-order GL, differentiable in params AND
# limits (Leibniz). A scipy-style integrand func(x, *args).
# ---------------------------------------------------------------------------


def _integrand(backend):
    # f(x, p) = exp(-p x) x**2 in the backend's namespace.
    xp = _xp(backend)
    return lambda x, p: xp.exp(-p * x) * x**2


# int_0^b exp(-p x) x**2 dx and its exact derivatives in b and in p.
def _exact_val(b, p):
    return (2.0 - numpy.exp(-p * b) * (p * b * (p * b + 2.0) + 2.0)) / p**3


def _exact_db(b, p):  # d/db (Leibniz): the integrand at x=b
    return numpy.exp(-p * b) * b**2


def _exact_dp(b, p):  # d/dp under the integral: int_0^b -x * exp(-p x) x**2 dx
    fd = (_exact_val(b, p + 1e-7) - _exact_val(b, p - 1e-7)) / (2e-7)
    return fd


B0, P0 = 2.5, 1.3


def test_quad_numpy_equals_scipy():
    # numpy path delegates to scipy.integrate.quad and returns its value [0]:
    # byte-identical, and a plain Python float (what the call sites use).
    from scipy import integrate as sint

    f = _integrand("numpy")
    ref = sint.quad(f, 0.3, B0, args=(P0,))[0]
    got = quad(f, 0.3, B0, args=(P0,))
    assert isinstance(got, float)
    assert got == ref  # byte-identical
    # ... and matches the closed form to ~1e-8.
    numpy.testing.assert_allclose(
        got, _exact_val(B0, P0) - _exact_val(0.3, P0), atol=1e-8
    )


def test_gauss_legendre_numpy_value():
    # gauss_legendre runs the GL rule in numpy (does NOT call scipy) and still
    # matches the analytic integral; numpy in, numpy out.
    f = _integrand("numpy")
    got = gauss_legendre(f, 0.0, B0, args=(P0,), n=80)
    assert isinstance(got, (numpy.ndarray, numpy.floating, float))
    numpy.testing.assert_allclose(float(got), _exact_val(B0, P0), rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_quad_known_function(backend):
    # Integral of a known function vs analytic, in every backend.
    xp = _xp(backend)
    f = _integrand(backend)
    a = xp.asarray(0.0) if backend != "numpy" else 0.0
    b = xp.asarray(B0) if backend != "numpy" else B0
    p = xp.asarray(P0) if backend != "numpy" else P0
    got = float(numpy.asarray(quad(f, a, b, args=(p,), n=100)))
    numpy.testing.assert_allclose(got, _exact_val(B0, P0), rtol=1e-9)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_quad_returns_backend_array(backend):
    # eager jax/torch must return a BACKEND array (the discriminating check: a
    # bare-numpy compute path silently passes eager torch but detaches on jax).
    xp = _xp(backend)
    f = _integrand(backend)
    out = quad(f, xp.asarray(0.0), xp.asarray(B0), args=(xp.asarray(P0),), n=60)
    assert backend in type(out).__module__


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_quad_grad_param_and_limit(backend):
    # grad w.r.t. a parameter and w.r.t. the upper limit vs finite-difference.
    xp = _xp(backend)
    f = _integrand(backend)
    h = 1e-6
    if backend == "jax":
        gb = float(
            jax.grad(lambda b: quad(f, 0.0, b, args=(jnp.asarray(P0),), n=100))(
                jnp.asarray(B0)
            )
        )
        gp = float(
            jax.grad(lambda p: quad(f, 0.0, jnp.asarray(B0), args=(p,), n=100))(
                jnp.asarray(P0)
            )
        )
    else:
        bt = torch.tensor(B0, requires_grad=True)
        quad(f, torch.tensor(0.0), bt, args=(torch.tensor(P0),), n=100).backward()
        gb = float(bt.grad)
        pt = torch.tensor(P0, requires_grad=True)
        quad(f, torch.tensor(0.0), torch.tensor(B0), args=(pt,), n=100).backward()
        gp = float(pt.grad)
    # vs analytic
    numpy.testing.assert_allclose(gb, _exact_db(B0, P0), rtol=1e-6)
    numpy.testing.assert_allclose(gp, _exact_dp(B0, P0), rtol=1e-6)
    # vs finite-difference (numpy reference, independent of the backend AD)

    def npval(b, p):
        return float(
            numpy.asarray(
                quad(
                    _integrand("numpy"),
                    0.0,
                    b,
                    args=(p,),
                )
            )
        )

    fd_b = (npval(B0 + h, P0) - npval(B0 - h, P0)) / (2 * h)
    fd_p = (npval(B0, P0 + h) - npval(B0, P0 - h)) / (2 * h)
    numpy.testing.assert_allclose(gb, fd_b, rtol=1e-5)
    numpy.testing.assert_allclose(gp, fd_p, rtol=1e-5)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_quad_dispatches_on_args_only(backend):
    # A backend array ONLY in args (scalar Python limits) still routes to the
    # in-backend differentiable path.
    xp = _xp(backend)
    f = _integrand(backend)
    out = quad(f, 0.0, B0, args=(xp.asarray(P0),), n=60)
    assert backend in type(out).__module__
    numpy.testing.assert_allclose(
        float(numpy.asarray(out)), _exact_val(B0, P0), rtol=1e-9
    )


def test_quad_numpy_no_args():
    # The no-args branch (integrand is used as-is) on the numpy/scipy path.
    from scipy import integrate as sint

    g = lambda x: numpy.sin(x)  # noqa: E731
    ref = sint.quad(g, 0.0, numpy.pi)[0]
    got = quad(g, 0.0, numpy.pi)
    assert got == ref
    numpy.testing.assert_allclose(got, 2.0, atol=1e-10)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_device_hint_explicit(backend):
    # The device= hint anchors the node/weight tables AND the (possibly scalar)
    # limits on a caller-supplied device, for integrands that close over arrays
    # on a device the scalar limits cannot reveal. Exercised here on CPU; the
    # CUDA case (where it is load-bearing) is in test_device_hint_cuda.
    if backend == "jax":
        xp, dev, tonp = jnp, jax.devices("cpu")[0], float
    else:
        xp, dev, tonp = txp, torch.device("cpu"), (lambda v: float(v.detach()))
    e5 = 2.0 * (1.0 - numpy.exp(-5.0))
    numpy.testing.assert_allclose(
        tonp(fixed_quad(xp, lambda s: 2.0 * xp.exp(-s), 0.0, 5.0, n=60, device=dev)),
        e5,
        rtol=1e-9,
    )
    numpy.testing.assert_allclose(
        tonp(
            fixed_quad_semiinfinite(
                xp, lambda s: 2.0 * xp.exp(-s), 0.0, n=80, device=dev
            )
        ),
        2.0,
        rtol=1e-6,
    )
    numpy.testing.assert_allclose(
        tonp(
            transformed_quad(
                xp,
                lambda s: 2.0 * xp.exp(-s),
                0.0,
                5.0,
                n=40,
                interior_point=1.0,
                device=dev,
            )
        ),
        e5,
        rtol=1e-7,
    )
    numpy.testing.assert_allclose(
        tonp(
            nested_quad(
                xp,
                lambda x, y: xp.ones_like(x * y),
                [[0.0, 1.0], [0.0, 2.0]],
                n=15,
                device=dev,
            )
        ),
        2.0,
        rtol=1e-12,
    )


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="needs a CUDA torch device",
)
def test_device_hint_cuda():
    # Scalar limits + a CUDA-closure integrand: without device= the CPU node
    # tables meet the CUDA integrand and torch raises; device= fixes all four.
    cuda = torch.device("cuda:0")
    scale = torch.tensor(2.0, device=cuda)

    def integ(s):
        return scale * torch.exp(-s)

    e5 = 2.0 * (1.0 - numpy.exp(-5.0))
    with pytest.raises(RuntimeError):  # no hint -> mixed-device error
        fixed_quad(txp, integ, 0.0, 5.0, n=60)
    for out, ref in [
        (fixed_quad(txp, integ, 0.0, 5.0, n=60, device=cuda), e5),
        (fixed_quad_semiinfinite(txp, integ, 0.0, n=60, device=cuda), 2.0),
        (
            transformed_quad(
                txp, integ, 0.0, 5.0, n=40, interior_point=1.0, device=cuda
            ),
            e5,
        ),
        (
            nested_quad(
                txp,
                lambda x, y: scale * torch.ones_like(x * y),
                [[0.0, 1.0], [0.0, 1.0]],
                n=20,
                device=cuda,
            ),
            2.0,
        ),
    ]:
        assert out.device.type == "cuda"
        numpy.testing.assert_allclose(float(out.detach().cpu()), ref, atol=1e-4)
    sc = torch.tensor(2.0, device=cuda, requires_grad=True)
    fixed_quad(
        txp, lambda s: sc * torch.exp(-s), 0.0, 5.0, n=60, device=cuda
    ).backward()
    numpy.testing.assert_allclose(float(sc.grad.cpu()), e5 / 2.0, rtol=1e-6)
