###############################################################################
# test_backend_quadrature.py: galpy.backend.quadrature -- backend-agnostic
# fixed-order quadrature. Asserts value parity vs scipy.integrate.quad / exact,
# autodiff in the limits AND through integrand parameters, and that the promoted
# gauss_legendre_01 is unchanged (the special-function hyp2f1 fallback uses it).
###############################################################################
import numpy
import pytest

from galpy.backend.quadrature import (
    finite_part_quad,
    fixed_quad,
    fixed_quad_semiinfinite,
    gauss_legendre,
    gauss_legendre_01,
    gauss_legendre_nodes,
    nested_quad,
    quad,
    symmetric_quad,
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
def test_fixed_quad_vectorized_false(backend):
    # vectorized=False drives the quadrature node-by-node (like
    # scipy.integrate.quad does) for a scalar-only integrand -- one that REJECTS
    # an array argument. This is the contract Potential.mass uses for
    # scalar-only potentials (DoubleExponentialDiskPotential / AnySpherical).
    xp = _xp(backend)
    ref = numpy.exp(-0.5) - numpy.exp(-3.0)

    def scalar_only(s):
        # Mirror check_potential_inputs_not_arrays: reject a >1-element array.
        if hasattr(s, "shape") and s.shape != () and len(s) > 1:
            raise TypeError("scalar-only integrand got an array")
        return xp.exp(-s)

    # The default (vectorized=True) would feed scalar_only the whole node array
    # and raise; vectorized=False calls it per node and matches the analytic.
    with pytest.raises(TypeError):
        fixed_quad(xp, scalar_only, 0.5, 3.0, n=40)
    got = float(
        numpy.asarray(fixed_quad(xp, scalar_only, 0.5, 3.0, n=40, vectorized=False))
    )
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


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_quad_grad_vectorized_false(backend):
    # Gradients still flow through the node-by-node (vectorized=False) path:
    # d/dp int_0^2 exp(-p s) ds, with a scalar-only integrand called per node.
    p0 = 1.3
    ref_gp = (2.0 * numpy.exp(-2 * p0) * p0 - (1 - numpy.exp(-2 * p0))) / p0**2
    if backend == "jax":
        gp = float(
            jax.grad(
                lambda p: fixed_quad(
                    jnp, lambda s: jnp.exp(-p * s), 0.0, 2.0, n=60, vectorized=False
                )
            )(jnp.asarray(p0))
        )
    else:
        pt = torch.tensor(p0, requires_grad=True)
        fixed_quad(
            txp, lambda s: txp.exp(-pt * s), 0.0, 2.0, n=60, vectorized=False
        ).backward()
        gp = float(pt.grad)
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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("eps", [1.0, 1e-2, 1e-4, 1e-6])
def test_semiinfinite_scale_makes_the_map_resolve_small_structure(backend, eps):
    # The map's node spacing in s just above the lower limit is O(L/n**2), so
    # with the default L=1 an integrand whose structure sits on a scale eps<<1
    # is simply not sampled -- and no n rescues it. Passing scale=eps makes the
    # map scale-invariant, so the error becomes INDEPENDENT of eps.
    #   int_eps^inf ds / (1 + (s/eps)**2) = eps * pi/4
    xp = _xp(backend)
    got = float(
        numpy.asarray(
            fixed_quad_semiinfinite(
                xp, lambda s: 1.0 / (1.0 + (s / eps) ** 2), eps, n=50, scale=eps
            )
        )
    )
    numpy.testing.assert_allclose(got, eps * numpy.pi / 4.0, rtol=1e-13)
    #   int_0^inf ds / (eps**2 + s**2) = pi / (2 eps), via the 'tan' map
    got = float(
        numpy.asarray(
            fixed_quad_semiinfinite(
                xp,
                lambda s: 1.0 / (eps**2 + s**2),
                0.0,
                n=50,
                kind="tan",
                scale=eps,
            )
        )
    )
    numpy.testing.assert_allclose(got, numpy.pi / (2.0 * eps), rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_semiinfinite_scale_broadcasts_against_the_limit(backend):
    # One call, three lower limits spanning six decades, each with its own
    # scale: the whole point is that a batched caller (jeans.sigmar over an r
    # grid) gets every element resolved, not just the O(1) ones.
    xp = _xp(backend)
    eps = xp.asarray([1.0, 1e-3, 1e-6])
    got = numpy.asarray(
        fixed_quad_semiinfinite(
            xp,
            lambda s: 1.0 / (1.0 + (s / eps[..., None]) ** 2),
            eps,
            n=50,
            scale=eps,
        )
    )
    ref = numpy.asarray([1.0, 1e-3, 1e-6]) * numpy.pi / 4.0
    numpy.testing.assert_allclose(got, ref, rtol=1e-13)


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_semiinfinite_scale_grad_in_limit(backend):
    # d/da int_a^inf f = -f(a); at a = eps with f = 1/(1+(s/eps)**2) that is
    # exactly -1/2, independent of eps. Differentiating through a SCALED map is
    # the path jeans.sigmar's r-gradient takes.
    eps = 1e-5

    def f(s):
        return 1.0 / (1.0 + (s / eps) ** 2)

    if backend == "jax":
        ga = float(
            jax.grad(lambda a: fixed_quad_semiinfinite(jnp, f, a, n=50, scale=eps))(
                jnp.asarray(eps)
            )
        )
    else:
        at = torch.tensor(eps, requires_grad=True)
        fixed_quad_semiinfinite(txp, f, at, n=50, scale=eps).backward()
        ga = float(at.grad)
    numpy.testing.assert_allclose(ga, -0.5, rtol=1e-10)


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
        # constant integrand -> GL is exact (summation roundoff only)
        numpy.testing.assert_allclose(float(out.detach().cpu()), ref, atol=1e-9)
    sc = torch.tensor(2.0, device=cuda, requires_grad=True)
    fixed_quad(
        txp, lambda s: sc * torch.exp(-s), 0.0, 5.0, n=60, device=cuda
    ).backward()
    numpy.testing.assert_allclose(float(sc.grad.cpu()), e5 / 2.0, rtol=1e-6)


def test_symmetric_quad_default_order_resolves_extreme_aspect_ratio():
    """The default order must handle R << |z| on a real galpy integrand.

    This is the case the vertical surface-density quadrature actually ships:
    ``Potential._surfdens``/``_surfdens_poisson`` integrate the density over
    ``[-|z|, |z|]`` with this rule, and under a trace there is no scipy to fall
    back to. At R=0.01, |z|=50 the integrand has structure on scale ~R across a
    5000:1 range, which is where a fixed-order rule runs out first.

    All three arms are asserted, because the pass alone would not show the bar
    discriminates:

    * n=200 vs the reference -- validates the ARBITER. scipy is the reference
      here, so it has to be shown converging to the same value the rule does;
      otherwise a disagreement could be scipy's fault, not the rule's.
    * the default order -- the actual contract.
    * n=50 -- must FAIL the bar. Without this the test would still pass if the
      default were lowered back to 50, which is exactly the regression it
      exists to catch.
    """
    from scipy import integrate

    from galpy.potential import MWPotential2014

    p = MWPotential2014[0]  # PowerSphericalPotentialwCutoff: cusped and truncated
    R, Z = 0.01, 50.0
    f = lambda x: p._dens(R, float(x), phi=0.0, t=0.0)  # noqa: E731
    fv = lambda s: numpy.array(  # noqa: E731
        [f(v) for v in numpy.atleast_1d(s).ravel()]
    ).reshape(numpy.shape(s))

    # epsabs=0 -- a pure RELATIVE criterion. scipy's default epsabs=1.49e-8 is
    # absolute and silently truncates integrals whose value is near it (galpy
    # gh#1289); it does not bite at this R, but the reference must not depend on
    # that luck.
    ref = integrate.quad(f, -Z, Z, epsabs=0, epsrel=1e-13, limit=2000)[0]

    def rel(**kw):
        got = float(symmetric_quad(numpy, fv, Z, interior_point=0.0, **kw))
        return abs(got - ref) / abs(ref)

    assert rel(n=200) < 1e-12, "reference and the rule disagree -- arbiter suspect"
    # No n= : this asserts on the DEFAULT, which is the thing that can regress.
    # Passing n=_QUAD_N here instead would still pass if someone lowered the
    # default, since it would be pinning the constant rather than the default.
    assert rel() < 1e-9, f"symmetric_quad's DEFAULT order is too low here: {rel():.2e}"
    assert rel(n=50) > 1e-7, (
        f"n=50 was supposed to be visibly wrong here ({rel(n=50):.2e}); if it is "
        "not, this test no longer discriminates and the bar needs re-deriving"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_part_quad_matches_the_analytic_finite_part(backend):
    # Construct an integrand whose finite part is known in closed form, rather
    # than comparing against another quadrature: f(u) = c/u**2 + exp(-u) gives
    # sym(u) = 2c/u**2 + (exp(-u) + exp(u)), so the rule's subtraction removes
    # the singular model exactly and
    #     int_0^b [sym(u) - 2c/u**2] du - 2c/b = 2 sinh(b) - 2c/b.
    xp = _xp(backend)
    c, b = 0.75, 1.3

    def f(u):
        return c / (u * u) + xp.exp(-u)

    got = finite_part_quad(xp, f, xp.asarray(b), c=c, peak_width=xp.asarray(0.0), n=200)
    expected = 2.0 * numpy.sinh(b) - 2.0 * c / b
    numpy.testing.assert_allclose(float(got), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_part_quad_peaked_branch_is_the_plain_integral(backend):
    # peak_width > 0 selects u = w sinh(t) over [0, asinh(b/w)], which is exactly
    # int_0^b sym(u) du -- no finite part, because there is no singularity to
    # subtract. With a smooth f that integral is 2 sinh(b) again, and the answer
    # must not depend on w: the substitution only redistributes the nodes.
    xp = _xp(backend)
    b = 1.3

    def f(u):
        return xp.exp(-u)

    expected = 2.0 * numpy.sinh(b)
    for w in (1e-6, 1e-3, 0.5):
        got = finite_part_quad(
            xp, f, xp.asarray(b), c=0.0, peak_width=xp.asarray(w), n=200
        )
        numpy.testing.assert_allclose(
            float(got), expected, rtol=1e-9, atol=1e-11, err_msg=f"peak_width={w}"
        )


@pytest.mark.parametrize("backend", AD_BACKENDS)
def test_finite_part_quad_branches_on_a_traced_peak_width(backend):
    # peak_width is data, so the branch cannot be a python if. Both arms are
    # evaluated; the guarded width is what keeps asinh(b/0) from poisoning the
    # taken arm. Check the selection is right at w == 0 AND that no nan leaks.
    xp = _xp(backend)
    c, b = 0.75, 1.3

    def f(u):
        return c / (u * u) + xp.exp(-u)

    at_zero = float(
        finite_part_quad(xp, f, xp.asarray(b), c=c, peak_width=xp.asarray(0.0), n=200)
    )
    assert numpy.isfinite(at_zero)
    numpy.testing.assert_allclose(
        at_zero, 2.0 * numpy.sinh(b) - 2.0 * c / b, rtol=1e-10, atol=1e-12
    )
