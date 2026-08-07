###############################################################################
# test_backend_anyaxisymdisk.py: multi-backend tests for
# AnyAxisymmetricRazorThinDiskPotential.
#
# This potential integrates an arbitrary surface density Sigma(R) against the
# razor-thin-disk Green's function (complete elliptic integrals K/E). The numpy
# path keeps scipy's adaptive quad + scipy.special.ellipk/ellipe and is
# byte-identical; a jax/torch input routes to the backend split Gauss-Legendre
# quadrature (galpy.backend.quadrature) with the backend elliptic fallback
# (galpy.backend.special.ellipk/ellipe), so the FORCES are jit/grad-safe.
#
# The eager-only gap this closes: before migration the forces returned a bare
# python float under a forced backend, so downstream ``xp.sqrt`` (e.g. vcirc)
# crashed ("sqrt(): argument must be Tensor, not float") and jax.jacfwd could
# not trace them.
#
# Design note: a PLAIN concrete backend scalar reuses scipy's accurate value
# (wrapped as a backend array); native GL runs only when a gradient is actually
# taken (a tracer, or a grad-tracking torch tensor). GL cannot resolve the
# small-z sheet structure the derivative FD-probes hit, so the concrete path
# stays scipy-accurate while the differentiable path stays native.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

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
    # The backend GL path for the POTENTIAL and the three SECOND derivatives
    # (_evaluate_gl / _R2deriv_gl / _z2deriv_gl / _Rzderiv_gl) is taken only for a TRACER
    # (jax) or a requires_grad tensor (torch) -- a CONCRETE backend scalar reuses scipy
    # via _bk_dispatch. #1121 landed these ~36 GL lines with NO traced backend test (only
    # the forces were traced), so feat/backends coverage jumped 15->51. Trace over each
    # (jit for jax, requires_grad for torch) so the GL branch runs, and check the value
    # matches the numpy/scipy reference (z != 0: the 2nd derivs are singular in the plane).
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
    jR = float(jax.jit(rf)(jnp.asarray(R0)))
    gR = float(jax.jacfwd(rf)(jnp.asarray(R0)))
    assert numpy.isfinite(jR) and numpy.isfinite(gR)
    numpy.testing.assert_allclose(jR, ref, rtol=1e-10, atol=1e-12)


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


def _no_torch_compile_deprecations():
    """Suppress torch's own import-time DeprecationWarnings during a compile.

    ``torch.compile`` lazily imports ``torch._inductor``, whose mkldnn module
    warns on ``torch.jit.script_method`` at class-definition time. A per-test
    ``filterwarnings`` mark cannot suppress it (a module-level
    ``error::DeprecationWarning`` pytestmark is applied last and wins), so
    filter it here, around the call itself.
    """
    import contextlib
    import warnings

    @contextlib.contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            yield

    return _ctx()


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
    ref = float(evaluatePotentials(_POT, R0, z0))  # eager (scipy) value
    torch._dynamo.reset()
    with _no_torch_compile_deprecations():
        got = float(
            torch.compile(
                lambda R, z: evaluatePotentials(_POT, R, z),
                fullgraph=False,
                dynamic=False,
            )(R0, z0)
        )
    # GL vs scipy adaptive quad differ only at the quadrature floor
    numpy.testing.assert_allclose(got, ref, rtol=1e-10)


# --- degenerate radii under a trace -----------------------------------------
# The a=R split makes R=0 and R=inf special: at R=0 the [0,R] and [R,2R] panels
# have zero width while the integrand is 0/0 there, so they evaluate to 0*nan;
# at R=inf every panel spans an infinite range. Both returned NaN under a trace
# while the concrete scipy path was finite. Guarded in _bk_split_quad.


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_degenerate_radii_traced_match_numpy_jax():
    """Phi(0) and Phi(inf) trace to the concrete values, not NaN.

    Must jit: eagerly the input is concrete and the scipy branch is taken, so an
    eager run would never touch the guarded quadrature. Compared by VALUE
    against the numpy path -- asserting merely 'not NaN' would pass on any
    finite garbage, and Phi(0) is a real number (~-2.79) worth pinning.
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
