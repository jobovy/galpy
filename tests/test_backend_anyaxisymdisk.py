###############################################################################
# test_backend_anyaxisymdisk.py: multi-backend tests for
# AnyAxisymmetricRazorThinDiskPotential.
#
# This potential integrates an arbitrary surface density Sigma(R) against the
# razor-thin-disk Green's function (complete elliptic integrals K/E). The numpy
# path keeps scipy's adaptive quad + scipy.special.ellipk/ellipe and is
# byte-identical; a jax/torch input -- concrete, traced or grad-tracking --
# routes to backend Gauss-Legendre quadrature (galpy.backend.quadrature) with the
# backend elliptic fallback (galpy.backend.special.ellipk/ellipe).
#
# The eager-only gap this closes: before migration the forces returned a bare
# python float under a forced backend, so downstream ``xp.sqrt`` (e.g. vcirc)
# crashed ("sqrt(): argument must be Tensor, not float") and jax.jacfwd could
# not trace them.
#
# Design note: the backend integrands are written in the exact offset u = a - R
# and folded about a = R (finite_part_quad), R2deriv is -dRforce/dR by autodiff
# and z2deriv follows from Laplace's equation, so the backend path matches an
# mpmath Hankel reference to ~1e-11 at and near the plane, where scipy's quad
# itself is good to 1e-8..1e-13 (test_backend_values_match_the_hankel_reference).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest
from backend_jit_helpers import assert_jit_matches_eager, no_torch_compile_deprecations

from galpy.backend import get_namespace, is_backend_array
from galpy.potential import (
    AnyAxisymmetricRazorThinDiskPotential,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluateRzderivs,
    evaluatez2derivs,
    evaluatezforces,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture. The -W error marks are the
# coverage-shard trap guard: a numpy.<ufunc> on a torch tensor emits a
# DeprecationWarning, so the namespace-agnostic surface density below must never
# fall back to bare numpy on the backend path.
pytestmark = [
    pytest.mark.backend_managed,
    pytest.mark.filterwarnings("error::DeprecationWarning"),
    pytest.mark.filterwarnings("error::FutureWarning"),
]

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


def _surfdens(R):
    # namespace-agnostic (numpy / jax / torch): differentiable on every backend
    return 1.5 * get_namespace(R).exp(-3.0 * R)


# One instance, shared: __init__ integrates surfdens once (scipy) for _pot_zero.
_POT = AnyAxisymmetricRazorThinDiskPotential(surfdens=_surfdens)

# Physical (R, z) grid with z != 0 for the differentiable checks; z == 0 is
# covered by the value-parity / scalar tests (its force is exact by symmetry).
_RS = [0.5, 0.9, 1.3, 2.1]
_ZS = [0.15, 0.3, 0.25, 0.4]


def _scalar(backend_name, x):
    if backend_name == "numpy":
        return numpy.float64(x)
    if backend_name == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return torch.tensor(x, dtype=torch.float64)


# Gold values: the Hankel-transform form of each quantity, evaluated with
# mpmath.quadosc at 25 dps (mpmath.quad for |z| >= 1), for Sigma(R) = exp(-R/0.3):
#     Phi    = -2 pi Int J0(kR) e^{-k|z|} St(k) dk
#     Rforce = -2 pi Int k J1(kR) e^{-k|z|} St(k) dk
#     zforce = -2 pi sgn(z) Int k J0(kR) e^{-k|z|} St(k) dk
#     R2deriv = 2 pi Int k^2 (J0(kR) - J1(kR)/(kR)) e^{-k|z|} St(k) dk
#     z2deriv = -2 pi Int k^2 J0(kR) e^{-k|z|} St(k) dk
#     Rzderiv = -2 pi sgn(z) Int k^2 J1(kR) e^{-k|z|} St(k) dk
# with St(k) = 0.09 (1 + 0.09 k^2)^(-3/2); at z = 0 the limits z -> 0+.
_HANKEL_GOLD = {
    "pot": {
        (0.2, 0.0): -1.6138577225030613,
        (0.2, 1e-08): -1.6138576902441126,
        (0.2, 1e-06): -1.613854496612939,
        (0.2, 0.01): -1.5820733448601418,
        (0.2, 1.0): -0.47652243083731066,
        (1.0, 0.0): -0.6286545960773888,
        (1.0, 1e-08): -0.6286545938359257,
        (1.0, 1e-06): -0.6286543719308844,
        (1.0, 0.01): -0.6263940553131371,
        (1.0, 1.0): -0.3810925503468679,
        (5.0, 0.0): -0.11373622112612981,
        (5.0, 1e-08): -0.11373622112612619,
        (5.0, 1e-06): -0.1137362211257644,
        (5.0, 0.01): -0.11373597865925376,
        (5.0, 1.0): -0.11142506885430359,
    },
    "Rforce": {
        (0.2, 0.0): -1.7898186952604456,
        (0.2, 1e-08): -1.7898185877306192,
        (0.2, 1e-06): -1.7898079423123356,
        (0.2, 0.01): -1.6857222206172744,
        (0.2, 1.0): -0.05366341989303102,
        (1.0, 0.0): -0.6398089000397701,
        (1.0, 1e-08): -0.6398088925682264,
        (1.0, 1e-06): -0.6398081528853387,
        (1.0, 0.01): -0.6323319963719337,
        (1.0, 1.0): -0.1483001842572871,
        (5.0, 0.0): -0.02301590950453551,
        (5.0, 1e-08): -0.02301590950452341,
        (5.0, 1e-06): -0.02301590950332393,
        (5.0, 0.01): -0.023015748148129966,
        (5.0, 1.0): -0.021605737586217724,
    },
    "zforce": {
        (0.2, 1e-08): -3.2258948032317742,
        (0.2, 1e-06): -3.225885345940858,
        (0.2, 0.01): -3.131295694966348,
        (0.2, 1.0): -0.3632586641804715,
        (1.0, 1e-08): -0.2241463147212866,
        (1.0, 1e-06): -0.22414669818781774,
        (1.0, 0.01): -0.22793302168526636,
        (1.0, 1.0): -0.19417544520871455,
        (5.0, 1e-08): -3.6307441366694755e-07,
        (5.0, 1e-06): -3.6780340695785897e-07,
        (5.0, 0.01): -4.813012465947153e-05,
        (5.0, 1.0): -0.004473541768393179,
    },
    "R2deriv": {
        (0.2, 0.0): 0.6037346813750064,
        (0.2, 1e-08): 0.6037350398077452,
        (0.2, 1e-06): 0.6037705243744419,
        (0.2, 0.01): 0.9352286829508506,
        (0.2, 1.0): 0.25226630538857053,
        (1.0, 0.0): -1.0271497108832743,
        (1.0, 1e-08): -1.0271496859781288,
        (1.0, 1e-06): -1.02714722037005,
        (1.0, 0.01): -1.0023807416394712,
        (1.0, 1.0): -0.005649784918531225,
        (5.0, 0.0): -0.009379942802732731,
        (5.0, 1e-08): -0.009379942802692396,
        (5.0, 1e-06): -0.00937994279869784,
        (5.0, 0.01): -0.009379776190087423,
        (5.0, 1.0): -0.008224481940673708,
    },
    "z2deriv": {
        (0.2, 0.0): -9.552828157677235,
        (0.2, 1e-08): -9.55282797846084,
        (0.2, 1e-06): -9.55281023593612,
        (0.2, 0.01): -9.363839786037222,
        (0.2, 1.0): -0.5205834048537257,
        (1.0, 0.0): 0.3873408108435042,
        (1.0, 1e-08): 0.38734079340990235,
        (1.0, 1e-06): 0.3873390674847113,
        (1.0, 0.01): 0.3700487452675374,
        (1.0, 1.0): -0.14265039933875587,
        (5.0, 0.0): 0.0047767609018256295,
        (5.0, 1e-08): 0.004776760901787713,
        (5.0, 1e-06): 0.004776760898033053,
        (5.0, 0.01): 0.004776626560461429,
        (5.0, 1.0): 0.0039033344234301634,
    },
    "Rzderiv": {
        (0.2, 1e-08): -10.752982298147986,
        (0.2, 1e-06): -10.752913224143162,
        (0.2, 0.01): -10.071879929275749,
        (0.2, 1.0): -0.11323144468734216,
        (1.0, 1e-08): -0.7471543707322954,
        (1.0, 1e-06): -0.7471544934271073,
        (1.0, 0.01): -0.7481433924238021,
        (1.0, 1.0): -0.21543894536387256,
        (5.0, 1e-08): -1.2101186718907986e-06,
        (5.0, 1e-06): -1.2130739899569089e-06,
        (5.0, 0.01): -3.106080790624548e-05,
        (5.0, 1.0): -0.00266372514039345,
    },
}
# measured worst over the grid, both backends, with ~3x headroom
_HANKEL_RTOL = {
    "pot": 3e-11,
    "Rforce": 2e-11,
    "zforce": 1.5e-13,
    "R2deriv": 3e-10,
    "z2deriv": 8e-11,
    "Rzderiv": 4e-9,
}


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("quantity", list(_HANKEL_GOLD))
def test_backend_values_match_the_hankel_reference(backend_name, quantity):
    # a concrete backend input takes the backend quadrature (not scipy), which
    # must be accurate on and near the plane: z = 0, 1e-8, 1e-6 straddle the
    # a = R singularity, where the old graded-panel rule was 2e-4 (Rforce) to
    # 1e2 (Rzderiv) off
    p = AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: get_namespace(R).exp(-R / 0.3)
    )
    fn = {
        "pot": lambda R, z: evaluatePotentials(p, R, z),
        "Rforce": p.Rforce,
        "zforce": p.zforce,
        "R2deriv": p.R2deriv,
        "z2deriv": p.z2deriv,
        "Rzderiv": p.Rzderiv,
    }[quantity]
    for (R, z), gold in _HANKEL_GOLD[quantity].items():
        got = fn(_scalar(backend_name, R), _scalar(backend_name, z))
        assert is_backend_array(got)
        assert abs(float(got) / gold - 1.0) < _HANKEL_RTOL[quantity], (
            f"{quantity}({R}, {z}) on {backend_name}: {float(got)!r}, gold {gold!r}"
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_eager_force_returns_backend_array_matching_numpy(backend_name):
    # Eager backend force is a backend array (the eager-only-gap fix) whose value
    # matches the numpy/scipy reference, INCLUDING the z == 0 plane (the vcirc /
    # normalize crash site).
    for R in _RS + [1.0]:
        for z in _ZS + [0.0]:
            refR = float(evaluateRforces(_POT, numpy.float64(R), numpy.float64(z)))
            refz = float(evaluatezforces(_POT, numpy.float64(R), numpy.float64(z)))
            Rb, zb = _scalar(backend_name, R), _scalar(backend_name, z)
            gotR = evaluateRforces(_POT, Rb, zb)
            gotz = evaluatezforces(_POT, Rb, zb)
            assert is_backend_array(gotR) and is_backend_array(gotz)
            numpy.testing.assert_allclose(float(gotR), refR, rtol=1e-10, atol=1e-12)
            numpy.testing.assert_allclose(float(gotz), refz, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_traced_potential_and_2ndderivs_backend(backend_name):
    # The backend path for the POTENTIAL and the three SECOND derivatives, traced
    # (jit for jax, requires_grad for torch), against the numpy/scipy reference.
    R0, z0 = 1.3, 0.25
    for fn in (
        evaluatePotentials,
        evaluateR2derivs,
        evaluatez2derivs,
        evaluateRzderivs,
    ):
        ref = float(fn(_POT, numpy.float64(R0), numpy.float64(z0)))
        if backend_name == "jax":
            got = jax.jit(lambda R, fn=fn: fn(_POT, R, jnp.asarray(z0)))(
                jnp.asarray(R0)
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            got = fn(_POT, Rt, torch.tensor(z0, dtype=torch.float64))
        assert is_backend_array(got), (fn.__name__, backend_name)
        numpy.testing.assert_allclose(
            float(got), ref, rtol=1e-6, atol=1e-8, err_msg=fn.__name__
        )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_traced_force_finite():
    # The exact failure that defined this gap: jax.jacfwd / jax.jit over the
    # forces must trace and return finite (they previously returned a bare float).
    R0, z0 = jnp.asarray(1.2), jnp.asarray(0.3)
    rf = lambda R: evaluateRforces(_POT, R, z0)
    zf = lambda z: evaluatezforces(_POT, R0, z)
    gR = jax.jacfwd(rf)(R0)
    jR = jax.jit(rf)(R0)
    gz = jax.jacfwd(zf)(z0)
    assert jnp.isfinite(gR) and jnp.isfinite(jR) and jnp.isfinite(gz)
    # jit value matches the eager scipy reference (z>0 -> GL is machine-accurate)
    numpy.testing.assert_allclose(
        float(jR), float(evaluateRforces(_POT, numpy.float64(1.2), numpy.float64(0.3)))
    )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_default_surfdens_jit_safe():
    # The DEFAULT surfdens is now backend-agnostic (was a bare ``numpy.exp`` that calls
    # __array__ on a jit tracer), so a DEFAULT-constructed disk -- how the potential-zoo
    # differentiability sweep builds it -- is jit/jacfwd-safe too. numpy path unchanged
    # (is_backend_array guard: numpy / python-float / Quantity R keep numpy.exp).
    pot = AnyAxisymmetricRazorThinDiskPotential()  # default surfdens
    R0, z0 = 1.1, 0.3
    ref = float(evaluateRforces(pot, numpy.float64(R0), numpy.float64(z0)))
    rf = lambda R: evaluateRforces(pot, R, jnp.asarray(z0))
    # ref= compares the traced array call against the plain-float numpy value;
    # the helper additionally rejects a trace that folded R away to a constant.
    assert_jit_matches_eager(rf, jnp.asarray(R0), rtol=1e-10, atol=1e-12, ref=ref)
    gR = float(jax.jacfwd(rf)(jnp.asarray(R0)))
    assert numpy.isfinite(gR)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_rforce_grad_vs_fd(backend_name):
    # d(Rforce)/dR via autodiff vs central finite differences, h-converging (not a
    # finite-and-nonzero check), at physical z > 0.
    for R0, z0 in zip(_RS, _ZS):
        znum = numpy.float64(z0)

        def fd(h):
            fp = float(evaluateRforces(_POT, numpy.float64(R0 + h), znum))
            fm = float(evaluateRforces(_POT, numpy.float64(R0 - h), znum))
            return (fp - fm) / (2.0 * h)

        if backend_name == "jax":
            ad = float(
                jax.grad(lambda R: evaluateRforces(_POT, R, jnp.asarray(z0)))(
                    jnp.asarray(R0)
                )
            )
        else:
            Rt = torch.tensor(R0, dtype=torch.float64, requires_grad=True)
            evaluateRforces(_POT, Rt, torch.tensor(z0, dtype=torch.float64)).backward()
            ad = float(Rt.grad)
        # central FD converges O(h^2); the two finest h bracket the AD tightly
        rels = [abs((ad - fd(h)) / ad) for h in (1e-4, 1e-5)]
        assert min(rels) < 1e-7, f"R={R0} z={z0} ({backend_name}): rels={rels}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_zforce_grad_vs_fd(backend_name):
    # d(zforce)/dz via autodiff vs central FD, h-converging, at physical z > 0.
    for R0, z0 in zip(_RS, _ZS):
        Rnum = numpy.float64(R0)

        def fd(h):
            fp = float(evaluatezforces(_POT, Rnum, numpy.float64(z0 + h)))
            fm = float(evaluatezforces(_POT, Rnum, numpy.float64(z0 - h)))
            return (fp - fm) / (2.0 * h)

        if backend_name == "jax":
            ad = float(
                jax.grad(lambda z: evaluatezforces(_POT, jnp.asarray(R0), z))(
                    jnp.asarray(z0)
                )
            )
        else:
            zt = torch.tensor(z0, dtype=torch.float64, requires_grad=True)
            evaluatezforces(_POT, torch.tensor(R0, dtype=torch.float64), zt).backward()
            ad = float(zt.grad)
        rels = [abs((ad - fd(h)) / ad) for h in (1e-4, 1e-5)]
        assert min(rels) < 1e-6, f"R={R0} z={z0} ({backend_name}): rels={rels}"


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_vcirc_no_longer_crashes(backend_name):
    # Regression for the reported eager crash: vcirc = xp.sqrt(R * -Rforce) fed a
    # bare python float under a forced backend. With the migrated (backend-array)
    # force it evaluates and matches the numpy value.
    pot = AnyAxisymmetricRazorThinDiskPotential(surfdens=_surfdens)
    pot.normalize(1.0)
    ref = float(pot.vcirc(1.3, use_physical=False))
    got = float(pot.vcirc(_scalar(backend_name, 1.3), use_physical=False))
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_torch_compile_takes_the_backend_gl_path():
    # Regression: under torch.compile the dispatch must pick the in-backend GL
    # quadrature, exactly as a jax tracer does. Its concreteness probe is
    # ``float(R)``, which a jax tracer answers by raising -- but dynamo makes it
    # a symbolic scalar, so the probe took the scipy branch and dynamo then
    # traced scipy's adaptive quad and died on numpy.unique's data-dependent
    # output shape (InductorError: DynamicOutputShapeException aten.unique_dim).
    # ``under_trace`` asks dynamo directly instead.
    #
    # The DEFAULT (inductor) compile backend is deliberate: with backend="eager"
    # a graph break silently rescues the scipy branch, so the bug would not
    # show. Only ``_evaluate`` is compiled -- the forces take minutes to codegen
    # the 3-panel/100-node GL graph and add no coverage.
    R0 = torch.tensor(1.1, dtype=torch.float64)
    z0 = torch.tensor(0.2, dtype=torch.float64)
    ref = float(evaluatePotentials(_POT, R0, z0))  # eager value
    with no_torch_compile_deprecations():
        torch._dynamo.reset()
        got = float(
            torch.compile(
                lambda R, z: evaluatePotentials(_POT, R, z),
                fullgraph=False,
                dynamic=False,
            )(R0, z0)
        )
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)


# --- degenerate radii under a trace -----------------------------------------
# The a=R split makes R=0 and R=inf special: at R=0 the [0,R] and [R,2R] panels
# have zero width while the integrand is 0/0 there, so they evaluate to 0*nan;
# at R=inf every panel spans an infinite range. Both returned NaN under a trace
# while the concrete scipy path was finite. Guarded in _bk_offset_quad.


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_degenerate_radii_traced_match_numpy_jax():
    """Phi(0) and Phi(inf) trace to the concrete values, not NaN.

    Traced, so the guards must hold without data-dependent branches. Compared
    by VALUE against the numpy path -- asserting merely 'not NaN' would pass on
    any finite garbage, and Phi(0) is a real number (~-2.79) worth pinning.
    """
    import galpy.backend as gb

    tp = AnyAxisymmetricRazorThinDiskPotential()
    tp.normalize(1.0)
    for R in (0.0, numpy.inf):
        ref = float(evaluatePotentials(tp, R, 0, phi=0.0, t=0.0))
        with gb.use("jax", force=True):
            got = float(
                jax.jit(
                    lambda Rv: evaluatePotentials(
                        tp,
                        Rv,
                        jnp.asarray(0.0),
                        phi=jnp.asarray(0.0),
                        t=jnp.asarray(0.0),
                    )
                )(jnp.asarray(R))
            )
        assert numpy.isfinite(got), f"R={R}: traced gave {got}"
        numpy.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_finite_radii_unchanged_by_degenerate_guards_jax():
    """The guards must not perturb ordinary radii -- they only select branches.

    Tolerance is the pre-existing traced-GL vs scipy-adaptive floor for this
    potential, not a licence for the guards to move anything: a guard that
    accidentally clamped a finite R would miss by far more than this.
    """
    import galpy.backend as gb

    tp = AnyAxisymmetricRazorThinDiskPotential()
    tp.normalize(1.0)
    for R in (0.3, 1.0, 3.0):
        ref = float(evaluatePotentials(tp, R, 0.2, phi=0.0, t=0.0))
        with gb.use("jax", force=True):
            got = float(
                jax.jit(
                    lambda Rv: evaluatePotentials(
                        tp,
                        Rv,
                        jnp.asarray(0.2),
                        phi=jnp.asarray(0.0),
                        t=jnp.asarray(0.0),
                    )
                )(jnp.asarray(R))
            )
        numpy.testing.assert_allclose(got, ref, rtol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_degenerate_guards_do_not_break_gradients_jax():
    """The xp.where guards must not poison AD.

    Both guards evaluate their dead branch (that is what xp.where does eagerly),
    so a nan there would reach the gradient even though the value is correct.
    Checked as grad-vs-central-FD with h-convergence rather than
    finite-and-nonzero: halving h must improve agreement, which a nan-poisoned or
    merely-plausible derivative would not do.
    """
    import galpy.backend as gb

    tp = AnyAxisymmetricRazorThinDiskPotential()
    tp.normalize(1.0)
    with gb.use("jax", force=True):

        def f(R):
            return evaluatePotentials(
                tp, R, jnp.asarray(0.2), phi=jnp.asarray(0.0), t=jnp.asarray(0.0)
            )

        g = jax.grad(f)
        for R0 in (0.3, 1.0, 3.0):
            ad = float(g(jnp.asarray(R0)))
            rels = []
            for h in (1e-4, 1e-5):
                fd = float(
                    (f(jnp.asarray(R0 + h)) - f(jnp.asarray(R0 - h))) / (2.0 * h)
                )
                rels.append(abs(ad - fd) / abs(fd))
            assert rels[-1] < 1e-9, f"R={R0}: AD vs FD rel={rels[-1]:g}"
            assert rels[-1] < rels[0], f"R={R0}: no h-convergence {rels}"


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_degenerate_guards_do_not_break_gradients_torch():
    """Same gradient check on torch: `requires_grad` also selects the GL path.

    Worth having on both backends rather than trusting jax to speak for torch --
    the guards are namespace-agnostic, so this is the assertion that says so.
    """
    import galpy.backend as gb

    tp = AnyAxisymmetricRazorThinDiskPotential()
    tp.normalize(1.0)
    with gb.use("torch", force=True):
        z, ph, t = torch.tensor(0.2), torch.tensor(0.0), torch.tensor(0.0)

        def f(Rv):
            return evaluatePotentials(tp, Rv, z, phi=ph, t=t)

        for R0 in (0.3, 1.0, 3.0):
            R = torch.tensor(R0, requires_grad=True)
            f(R).backward()
            ad = float(R.grad)
            h = 1e-5
            fd = float((f(torch.tensor(R0 + h)) - f(torch.tensor(R0 - h))) / (2.0 * h))
            rel = abs(ad - fd) / abs(fd)
            assert rel < 1e-9, f"torch R={R0}: AD vs FD rel={rel:g}"


# --- small-|z| quadrature: the regime every probe above skips --------------
# _ZS is [0.15, 0.3, 0.25, 0.4] and the traced 2nd-deriv test uses z=0.25, so no
# backend test ever entered 0 < |z| << R. That is exactly where the plain
# two-panel GL split failed: the integrand has a peak of width ~|z| at a=R, and
# once |z| drops below the Legendre node spacing near the panel edge the peak
# falls BETWEEN nodes. Measured before the graded split, R=1:
#     z/R    1e-06     1e-04     1e-03     3e-03
#     rel    3.67e+03  1.15e+03  8.38e-01  5.79e-04
# numpy is immune because scipy's quad(..., points=[R]) subdivides adaptively.
# This reaches users as a wrong GRADIENT too, not just a wrong jit value, since
# _bk_dispatch routes requires_grad input down the same GL path.
_SMALL_ZS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def _traced_call(backend_name, fn, R, z):
    """Evaluate fn on the GL path (jit for jax, requires_grad for torch)."""
    if backend_name == "jax":
        return float(
            jax.jit(lambda RR: fn(_POT, RR, jnp.asarray(z, dtype=jnp.float64)))(
                jnp.asarray(R, dtype=jnp.float64)
            )
        )
    Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
    return float(fn(_POT, Rt, torch.tensor(z, dtype=torch.float64)))


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("z", _SMALL_ZS)
def test_second_derivs_at_small_z_match_scipy(backend_name, z):
    """The GL path must track scipy down to |z|/R ~ 1e-6, not just at |z| >= 0.15."""
    R = 1.0
    for fn in (evaluateR2derivs, evaluatez2derivs):
        ref = float(fn(_POT, numpy.float64(R), numpy.float64(z)))
        got = _traced_call(backend_name, fn, R, z)
        rel = abs(got - ref) / abs(ref)
        assert rel < 1e-4, (
            f"{fn.__name__} on {backend_name} at z={z:g} is off by {rel:.2e}; "
            "the a=R peak (width ~|z|) is not resolved by the quadrature"
        )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("z", [0.0, 1e-3, 1e-2, 0.125])
def test_rforce_finite_difference_in_R_stays_clean(backend_name, z):
    """A finite difference of Rforce in R must still reproduce R2deriv.

    Panel edges scale with R, so refining them toward a=R can make the
    quadrature error vary non-smoothly in R -- harmless for the value, fatal for
    a caller's finite difference, which divides by dr=1e-8. An earlier
    unconditionally-graded version passed every accuracy check above and was
    wrong here by 193%. z=0 is the sensitive case and is included deliberately.
    """
    R = 1.0
    dr = 1e-8
    dr = (R + dr) - R  # representable
    f0 = _traced_call(backend_name, evaluateRforces, R, z)
    f1 = _traced_call(backend_name, evaluateRforces, R + dr, z)
    fd = (f0 - f1) / dr
    # At z=0 the analytic 2nd derivative is a divergent integral, so compare the
    # FD against the numpy/scipy value rather than the backend's own.
    ref = float(evaluateR2derivs(_POT, numpy.float64(R), numpy.float64(z)))
    rel = abs(fd - ref) / abs(ref)
    assert rel < 1e-3, (
        f"FD of Rforce on {backend_name} at z={z:g} gives {fd:.8e} vs "
        f"R2deriv {ref:.8e} (rel {rel:.2e}) -- the quadrature error is not "
        "varying smoothly in R"
    )


# The panel ladder graded toward a=R used to be dyadic (R/2, R/4, ... clamped at
# dmin ~ 4|z|), which bottoms out at R*2**-K: for |z| below that the innermost
# panel never reaches the peak and the force decays toward zero instead of
# toward the sheet limit. K=24 is 0.5% low at |z|=1e-10 and ~95% low at 1e-12.
# Spanning geometrically to dmin instead resolves the peak at any |z| with the
# same K (so the jit graph does not grow).
#
# This is checked against the ANALYTIC sheet limit rather than against numpy:
# scipy's own quad has the same peak problem here and is nan by |z|=1e-10
# (galpy #1276 is the sibling defect in RazorThinExponentialDiskPotential), so
# numpy is not a usable reference this far in.
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("R", [0.7, 1.0, 1.6])
@pytest.mark.parametrize("z,tol", [(1e-10, 1e-7), (1e-12, 2e-5)])
def test_zforce_at_tiny_z_reaches_sheet_limit(backend_name, R, z, tol):
    """zforce -> -2 pi Sigma(R) as |z| -> 0+, on the differentiable GL path.

    Bars are set just above the measured deviation (2.1e-8 at 1e-10, 3.9e-6 at
    1e-12); the dyadic ladder misses them by 2.2e-3 and 0.92 respectively, so
    this fails loudly if the grading regresses rather than merely drifting.
    """
    limit = -2.0 * numpy.pi * float(_surfdens(numpy.float64(R)))
    got = _traced_call(backend_name, evaluatezforces, R, z)
    rel = abs(got / limit - 1.0)
    assert rel < tol, (
        f"zforce on {backend_name} at R={R:g}, z={z:g} is {got:.8e} vs the "
        f"sheet limit {limit:.8e} (rel {rel:.2e}) -- the a=R peak is not "
        "resolved, so the graded panels are not reaching dmin"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_traced_second_derivs_match_numpy_through_the_midplane(backend_name):
    # Traced (concrete input takes the same backend path; the eager comparison
    # is test_backend_values_match_the_hankel_reference). The backend path used
    # to run ONE generic graded-panel rule for all six methods, while numpy
    # handles the a=R singularity analytically: at z == 0 traced R2deriv came out
    # ~3.8e+03 relative error, reachable from user code, since traced epifreq()
    # evaluates R2deriv at z=0.
    #
    # The tolerances are those of the numpy reference this compares against
    # (scipy's finite-part quad, ~1e-5 at z == 0 and tightening off the plane).
    import warnings

    import galpy.backend

    grid = ((0.0, 1e-4), (1e-10, 1e-4), (1e-8, 1e-6), (1e-6, 1e-8), (1e-4, 1e-10))
    for fn in (evaluateR2derivs, evaluatez2derivs):
        for R in (0.5, 1.0, 2.0):
            # numpy references OUTSIDE the forced context: taken inside, the
            # boundary coerces them onto the traced path and the comparison
            # becomes traced-vs-itself.
            refs = [
                float(fn(_POT, numpy.float64(R), numpy.float64(z))) for z, _ in grid
            ]
            # torch emits its own script_method DeprecationWarning the first
            # time torch.compile runs in a process. Under the numpy-default run
            # this test is that first use, and the suite turns warnings into
            # errors -- so filter that one warning narrowly rather than losing
            # the torch arm of the assertion.
            with (
                warnings.catch_warnings(),
                galpy.backend.use(backend_name, force=True),
                galpy.backend.jit(backend_name),
            ):
                warnings.filterwarnings(
                    "ignore", message=".*script_method.*", category=DeprecationWarning
                )
                for (z, tol), ref in zip(grid, refs):
                    got = fn(_POT, _scalar(backend_name, R), _scalar(backend_name, z))
                    assert is_backend_array(got), (
                        f"{fn.__name__} fell back off the backend at R={R}, z={z}"
                    )
                    rel = abs(1.0 - float(got) / ref)
                    assert rel < tol, (
                        f"{backend_name} traced {fn.__name__} at R={R}, z={z}: "
                        f"rel err {rel:.2e} exceeds {tol:.0e} (numpy {ref:.6e})"
                    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_construction_accepts_a_surfdens_that_returns_a_backend_array(backend_name):
    # __init__ decides whether surfdens carries astropy units by DUCK-TYPING on
    # .to(): call it and see whether the result converts to Msun/pc^2. A torch
    # tensor HAS a .to() -- with a completely different meaning -- so instead of
    # the expected AttributeError it raised
    #     TypeError: to() received an invalid combination of arguments
    #                - got (CompositeUnit)
    # which was not in the except clause, and constructing this potential with a
    # torch-returning surfdens died outright. jax escaped only because jax
    # arrays have no .to at all.
    #
    # The module-level _surfdens does not exercise this: it dispatches on its
    # ARGUMENT, and __init__ probes with a plain python float, so it comes back
    # numpy on every backend. The surface density here is what a torch user
    # actually writes -- backend array out, whatever goes in.
    if backend_name == "jax":

        def sdens(R):
            return 1.5 * jnp.exp(-3.0 * jnp.asarray(R))

    else:

        def sdens(R):
            return 1.5 * torch.exp(-3.0 * torch.as_tensor(R, dtype=torch.float64))

    pot = AnyAxisymmetricRazorThinDiskPotential(surfdens=sdens)
    # Constructing is most of the point, but not all of it: if the units probe
    # had gone the OTHER way and decided this surfdens was unit-carrying, the
    # amplitude would be silently rescaled. Pin the value against the
    # equivalent namespace-agnostic surface density on the same backend path
    # instead of just smoke-testing.
    for R, z in ((0.9, 0.3), (1.3, 0.0)):
        Rb, zb = _scalar(backend_name, R), _scalar(backend_name, z)
        ref = float(evaluateRforces(_POT, Rb, zb))
        got = float(evaluateRforces(pot, Rb, zb))
        numpy.testing.assert_allclose(got, ref, rtol=1e-14, err_msg=f"R={R}, z={z}")


# Under a forced backend (gap audit PR 28): the constructor's unit probes and its
# Phi(0) quadrature call surfdens with plain floats under numpy semantics, so an
# xp-agnostic surfdens works (forced torch raised on torch.exp(int)) and amp is
# differentiable under jit
from backend_param_grad import MODES, assert_param_grad  # noqa: E402


def _anydisk(amp):
    return AnyAxisymmetricRazorThinDiskPotential(
        amp=amp, surfdens=lambda R: 1.5 * get_namespace(R).exp(-3.0 * R)
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["Phi", "Rforce", "zforce"])
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_xp_agnostic_surfdens_amp_gradient(backend_name, mode, method):
    assert_param_grad(
        backend_name, mode, _anydisk, {"amp": 1.2}, "amp", method, [0.9, 0.2]
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_xp_agnostic_surfdens_forced_plain_values(backend_name):
    import galpy.backend as gb

    npot = _anydisk(1.2)
    with gb.use(backend_name, force=True):
        pot = _anydisk(1.2)
        out = [pot(0.9, 0.2), pot.Rforce(0.9, 0.2), pot(0.0, 0.0)]
    # forced plain values take the backend quadrature, numpy's scipy: both are
    # within ~1e-12 of the truth here
    for got, ref in zip(out, [npot(0.9, 0.2), npot.Rforce(0.9, 0.2), npot(0.0, 0.0)]):
        assert is_backend_array(got)
        numpy.testing.assert_allclose(float(got), ref, rtol=1e-11)
