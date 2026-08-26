###############################################################################
# test_backend_potential_diff.py: a general PARAMETERIZED backend
# differentiability sweep over (almost) every galpy potential.
#
# The regular backend suite runs EAGER: it wraps a plain force call in
# `use(backend, force=True)` and checks the value. That eager path is BLIND to
# the differentiability / jit gap class, because a raw-numpy potential happily
# accepts a *concrete* jax/torch array through the __array__ protocol and
# quietly returns numpy -- losing the backend but never raising. The two
# failure modes that actually break differentiable / jit-compiled use only
# surface under a trace:
#
#   1. jit / trace-safety -- under an AD trace (jax.jacfwd / torch autograd) a
#      raw-numpy potential calls numpy on a *tracer* and crashes (or a
#      data-dependent branch on the tracer aborts).
#   2. amp-gradient -- differentiating a force w.r.t. a backend-array `amp`
#      trips the in-place `self._amp *= ...` leaf mutation (torch) or a
#      raw-numpy transform of amp in __init__ (jax/torch).
#
# This module makes both gaps VISIBLE across the whole potential zoo. It is
# TEST-ONLY (no source edits, so no numpy byte-identity risk): the currently
# broken (potential, backend) pairs are collected in KNOWN_JIT_GAPS /
# KNOWN_AMP_GAPS and xfail(strict=False)-ed, so the sweep is green today and
# burns down as the fix/backend-* PRs land (a flipped gap becomes an xpass, and
# is then removed from the set).
#
# jit-safety differentiates w.r.t. the potential's PRIMARY coordinate (R for
# 3d/planar, the 1d coordinate for linear) with a SCALAR input: a scalar tracer
# fully exercises the raw-numpy fall-through, while avoiding array-vectorization
# limitations (some potentials reject array inputs on numpy too) that are
# orthogonal to backend trace-safety. It uses jax.jacfwd (not jax.jit) on the
# jax side: jacfwd/torch-autograd operate on concrete primals, so a
# differentiable-but-not-jit-compilable potential (one whose python branch is
# taken on a concrete value) is correctly NOT counted as a gap -- the target
# class here is raw-numpy potentials that cannot be traced at all. Those
# remaining `if <traced>` branches are fixed one family at a time (the
# RazorThinExponentialDisk in-plane/panel selections became xp.where/xp.maximum;
# the jax.jit coverage lives in that potential's own backend test module).
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import inspect

import numpy
import pytest

# test-module potential-name discovery (75 names), byte-identical to test_potential
from test_potential import _discover_base_pots

from galpy.backend import use
from galpy.potential import (
    Potential,
    evaluatelinearForces,
    evaluateplanarRforces,
    evaluateRforces,
)
from galpy.potential.linearPotential import linearPotential
from galpy.potential.planarPotential import planarPotential

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture. The filterwarnings entry is a
# TARGETED ignore of a pre-existing numpy-2.0 source deprecation (coords.py
# __array_wrap__, tripped by torch tensors through the Staeckel uv-transform in
# KuzminKutuzovStaeckel amp-grad); it is unrelated to this test and would
# otherwise error under the coverage shard's -W error::DeprecationWarning.
pytestmark = [
    pytest.mark.backend_managed,
    pytest.mark.filterwarnings(
        "ignore:__array_wrap__ must accept context:DeprecationWarning"
    ),
]

BACKENDS = ["numpy"]
try:
    import jax

    jax.config.update("jax_enable_x64", True)

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

POT_NAMES = _discover_base_pots()

# Evaluation point (generic, non-degenerate: away from z=0 / turning points so
# gradients are well defined).
_R0, _Z0, _PHI0 = 1.1, 0.13, 0.2

# ---------------------------------------------------------------------------
# Known gaps -- (potential, backend) pairs that currently FAIL. Determined
# empirically (see the sweep) and cross-checked against the backend-audit gap
# lists. xfail(strict=False) so a fix flips the entry to an xpass rather than a
# red; remove the entry from the set in the same PR that fixes the source.
# ---------------------------------------------------------------------------

# jit / trace-safety: raw-numpy potentials that cannot be traced (identical set
# on jax and torch). Matches the audit jit-gap list (constructible subset).
# FULLY BURNED DOWN: every constructible potential in the zoo is now jit-safe. The
# last entry, AnyAxisymmetricRazorThinDiskPotential, was fixed in #1127 (its default
# surfdens made backend-agnostic so it no longer hits raw numpy.exp on a traced GL node).
KNOWN_JIT_GAP_NAMES = set()
KNOWN_JIT_GAPS = {(name, b) for name in KNOWN_JIT_GAP_NAMES for b in AD_BACKENDS}

# amp-gradient gaps -- ALL burned down. The in-place `self._amp *= ...`
# leaf-mutation family (torch) and the raw-numpy amp-__init__ cases (jax
# IsothermalDisk, #1123) were migrated by the fix/backend-* PRs; the sweep now
# asserts amp-grad works for every amp-bearing potential on both backends.
_AMP_GAP_TORCH = set()
_AMP_GAP_JAX = set()
KNOWN_AMP_GAPS = {(name, "torch") for name in _AMP_GAP_TORCH} | {
    (name, "jax") for name in _AMP_GAP_JAX
}

_JIT_REASON = (
    "eager-only raw-numpy potential: not trace-safe -- being fixed in fix/backend-* PRs"
)
_AMP_REASON = "in-place-_amp leaf mutation / raw-numpy amp transform -- being fixed in fix/backend-* PRs"


# ---------------------------------------------------------------------------
# Force-evaluation helpers (dispatch by potential dimensionality).
# ---------------------------------------------------------------------------
def _kind(pot):
    if isinstance(pot, Potential):
        return "3d"
    if isinstance(pot, planarPotential):
        return "planar"
    if isinstance(pot, linearPotential):
        return "linear"
    return None


def _x0(pot):
    # the potential's primary (differentiated) coordinate
    return _Z0 if _kind(pot) == "linear" else _R0


def _force_of_x(pot, x):
    k = _kind(pot)
    if k == "3d":
        return evaluateRforces(pot, x, _Z0, phi=_PHI0)
    if k == "planar":
        return evaluateplanarRforces(pot, x, phi=_PHI0)
    if k == "linear":
        return evaluatelinearForces(pot, x)
    raise RuntimeError(f"no force dispatch for {type(pot).__name__}")


def _construct(name):
    """Default-construct the named potential, or skip the param.

    A skip (needs args / is a test-module mock / needs pynbody / is an abstract
    base with no force / has an identically-zero force) is NOT a gap.
    """
    import galpy.potential as gp

    try:
        cls = getattr(gp, name)
    except AttributeError:  # a test-module mock, not a real potential
        pytest.skip(f"{name}: not a galpy.potential attribute (test-module mock)")
    try:
        pot = cls()
    except Exception as e:  # noqa: BLE001 -- needs args / pynbody / not callable
        pytest.skip(f"{name}: default construction failed ({type(e).__name__})")
    if _kind(pot) is None:
        pytest.skip(f"{name}: not a force-bearing potential")
    try:
        f0 = float(numpy.asarray(_force_of_x(pot, _x0(pot))))
        f1 = float(numpy.asarray(_force_of_x(pot, _x0(pot) + 0.7)))
    except Exception as e:  # noqa: BLE001 -- abstract base w/o _Rforce, etc.
        pytest.skip(f"{name}: force not evaluable on numpy ({type(e).__name__})")
    if f0 == 0.0 and f1 == 0.0:
        pytest.skip(f"{name}: identically-zero force (null / empty composite)")
    return cls, pot


def _amp_default(cls):
    """Default value of the constructor `amp` argument, or None if the potential
    has no `amp` free parameter (e.g. KingPotential, composites)."""
    try:
        p = inspect.signature(cls.__init__).parameters.get("amp")
    except (ValueError, TypeError):  # pragma: no cover
        return None
    if p is None or p.default is inspect.Parameter.empty:
        return None
    try:
        return float(p.default)
    except (TypeError, ValueError):  # pragma: no cover
        return None


def _ctx(backend_name):
    # Run the traced eval under the forced backend, matching how galpy's backend
    # suite runs (also makes the namespace unambiguous for migrated potentials).
    return use(backend_name, force=True)


# ---------------------------------------------------------------------------
# Parametrization: one (name, backend) param per test, with a per-pair xfail
# mark for the known gaps.
# ---------------------------------------------------------------------------
def _params(known_gaps, reason):
    out = []
    for backend_name in AD_BACKENDS:
        for name in POT_NAMES:
            marks = []
            if (name, backend_name) in known_gaps:
                marks = [pytest.mark.xfail(strict=False, reason=reason)]
            out.append(
                pytest.param(
                    name, backend_name, marks=marks, id=f"{name}-{backend_name}"
                )
            )
    return out


@pytest.mark.parametrize("name,backend_name", _params(KNOWN_JIT_GAPS, _JIT_REASON))
def test_jit_safety(name, backend_name):
    """Force eval must survive an AD trace with a backend scalar input.

    Raw-numpy potentials crash here (numpy called on a tracer); the eager suite
    silently tolerates them. Differentiate w.r.t. the potential's primary
    coordinate and assert the value and derivative are finite.
    """
    cls, pot = _construct(name)
    x0 = _x0(pot)
    if backend_name == "jax":

        def f(x):
            with _ctx("jax"):
                return _force_of_x(pot, x)

        deriv = jax.jacfwd(f)(jax.numpy.asarray(x0))
        val = f(jax.numpy.asarray(x0))
        assert numpy.isfinite(float(deriv)), f"{name}: jax jacfwd not finite"
        assert numpy.isfinite(float(val)), f"{name}: jax value not finite"
    else:  # torch
        x = torch.tensor(x0, requires_grad=True)
        with _ctx("torch"):
            force = _force_of_x(pot, x)
        force.backward()
        assert torch.isfinite(force.detach()), f"{name}: torch value not finite"
        assert torch.isfinite(x.grad), f"{name}: torch grad not finite"


@pytest.mark.parametrize("name,backend_name", _params(KNOWN_AMP_GAPS, _AMP_REASON))
def test_amp_gradient(name, backend_name):
    """d(force)/d(amp) with a backend-array amp must (a) not raise and (b) match
    a central finite-difference of the force w.r.t. amp, h-converged.

    Skips potentials without a free `amp` constructor parameter (a skip is not a
    gap). The amp is differentiated through construction (`cls(amp=a)`), which is
    exactly where the in-place-_amp leaf mutation / raw-numpy amp transform bite.
    """
    cls, pot = _construct(name)
    a0 = _amp_default(cls)
    if a0 is None:
        pytest.skip(f"{name}: no free `amp` constructor parameter")
    x0 = _x0(pot)

    def g_numpy(a):
        return float(numpy.asarray(_force_of_x(cls(amp=a), x0)))

    # central finite-difference reference, h-converged
    def fd(h):
        return (g_numpy(a0 + h) - g_numpy(a0 - h)) / (2.0 * h)

    fd_coarse, fd_fine = fd(1e-5), fd(5e-6)
    assert numpy.isfinite(fd_coarse) and numpy.isfinite(fd_fine)
    assert abs(fd_coarse - fd_fine) <= 1e-4 * (abs(fd_fine) + 1e-8), (
        f"{name}: FD not converged ({fd_coarse:.6g} vs {fd_fine:.6g})"
    )

    if backend_name == "jax":

        def gj(a):
            with _ctx("jax"):
                return _force_of_x(cls(amp=a), x0)

        grad = float(jax.grad(gj)(jax.numpy.asarray(a0)))
    else:  # torch
        a = torch.tensor(a0, requires_grad=True)
        with _ctx("torch"):
            force = _force_of_x(cls(amp=a), x0)
        force.backward()
        grad = float(a.grad)

    assert numpy.isfinite(grad), f"{name}: {backend_name} amp-grad not finite"
    tol = 1e-5 * (abs(fd_fine) + 1e-6) + 1e-7
    assert abs(grad - fd_fine) <= tol, (
        f"{name}: {backend_name} amp-grad {grad:.8g} != FD {fd_fine:.8g}"
    )


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
@pytest.mark.parametrize("forcepoisson", [True, False])
def test_surfdens_differentiates_in_R_not_only_z(backend_name, forcepoisson):
    # surfdens picks scipy-vs-backend-quadrature with _quad_needs_backend, which
    # used to ask only about the INTEGRATION LIMIT (absz). The integrand closes
    # over R, phi and t just as much, so differentiating w.r.t. R at fixed z
    # fell into scipy and raised a concretization error -- while the very same
    # derivative w.r.t. z worked. An asymmetry inside one function.
    #
    # Both surfdens routes are covered: forcepoisson=True goes through the
    # Rforce/R2deriv/phi2deriv combination, False through _dens.
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(a=0.5, b=0.1)
    R0, z0, h = 1.0, 0.3, 1e-5

    def val(R, z):
        return float(pot.surfdens(R, z, forcepoisson=forcepoisson, use_physical=False))

    fd_R = (val(R0 + h, z0) - val(R0 - h, z0)) / (2 * h)
    fd_z = (val(R0, z0 + h) - val(R0, z0 - h)) / (2 * h)

    def sd(R, z):
        return pot.surfdens(R, z, forcepoisson=forcepoisson, use_physical=False)

    # Both backends now. torch reaches this because _quad_needs_backend asks
    # requires_backend_grad as well as under_trace: eager torch autograd is the
    # one differentiating mode with no trace behind it, so scipy used to run and
    # silently detach -- returning a bare float here (forcepoisson=False) or a
    # tensor whose gradient was 23% wrong in R and 72% in z (forcepoisson=True).
    ad_R, live_R = _surfdens_ad(backend_name, pot, forcepoisson, "R", R0, z0)
    ad_z, live_z = _surfdens_ad(backend_name, pot, forcepoisson, "z", R0, z0)
    assert live_R and live_z, (
        f"{backend_name} forcepoisson={forcepoisson}: surfdens returned a bare "
        "float, severing the autograd graph"
    )

    # h=1e-5 central differences on a smooth integral: good to ~1e-9. d/dz is
    # the control -- it worked before this fix, so if it ever regresses the
    # cause is the gate, not the quadrature.
    numpy.testing.assert_allclose(ad_R, fd_R, rtol=1e-6, atol=1e-9)
    numpy.testing.assert_allclose(ad_z, fd_z, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# surfdens: BOTH routes must stay on the backend and stay differentiable.
#
# scipy.integrate.quad cannot be evaluated on a tracer at all, but on an eager
# grad-tracking torch tensor it does something worse: it silently DETACHES the
# integrated piece while the algebraic prefactor keeps its graph, so the value
# stays right and the gradient comes back wrong. Measured on develop, before
# `_quad_needs_backend` learned to ask `requires_backend_grad`:
#
#   forcepoisson=False -> a bare Python float (graph severed entirely)
#   forcepoisson=True  -> Tensor, requires_grad=True, grad_fn set,
#                         gradient 23.3% wrong in R and 71.8% wrong in z
#
# jax never had this: jax.grad passes a tracer, which `under_trace` already
# caught. Only eager torch autograd differentiates with no trace behind it.
# ---------------------------------------------------------------------------
_SURFDENS_R0, _SURFDENS_Z0 = 1.0, 0.3


def _surfdens_ad(backend_name, pot, forcepoisson, wrt, R0, z0):
    with _ctx(backend_name):
        if backend_name == "jax":
            import jax.numpy as jnp

            def f(v):
                R = v if wrt == "R" else jnp.asarray(R0)
                z = v if wrt == "z" else jnp.asarray(z0)
                return pot.surfdens(R, z, forcepoisson=forcepoisson, use_physical=False)

            x0 = R0 if wrt == "R" else z0
            return float(jax.grad(f)(jnp.asarray(x0))), True
        leaf = torch.tensor(R0 if wrt == "R" else z0, requires_grad=True)
        R = leaf if wrt == "R" else torch.tensor(R0)
        z = leaf if wrt == "z" else torch.tensor(z0)
        out = pot.surfdens(R, z, forcepoisson=forcepoisson, use_physical=False)
        # A bare float here IS the bug: the graph was severed, so there is no
        # gradient to be wrong about. Report it rather than raising obscurely.
        if not isinstance(out, torch.Tensor):
            return float("nan"), False
        (g,) = torch.autograd.grad(out, leaf)
        return float(g), True


def _surfdens_fd(pot, wrt, eps=1e-6):
    def at(R, z):
        return float(pot.surfdens(R, z, use_physical=False))

    if wrt == "R":
        hi, lo = (
            at(_SURFDENS_R0 + eps, _SURFDENS_Z0),
            at(_SURFDENS_R0 - eps, _SURFDENS_Z0),
        )
    else:
        hi, lo = (
            at(_SURFDENS_R0, _SURFDENS_Z0 + eps),
            at(_SURFDENS_R0, _SURFDENS_Z0 - eps),
        )
    return (hi - lo) / (2 * eps)


@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_surfdens_gradient_for_a_potential_that_opts_out_of_array_forces(
    backend_name,
):
    # AnySphericalPotential sets _force_accepts_arrays = False, the case that
    # motivated a possible "raise unless array-safe" guard. Measured: it needs
    # no guard -- the backend quadrature handles it and the gradient is right,
    # so the simpler predicate-only fix is enough.
    from galpy.potential import AnySphericalPotential

    pot = AnySphericalPotential(
        amp=1.0, dens=lambda r: 1.0 / (r * (1.0 + r) ** 3) / (4.0 * numpy.pi)
    )
    fd = _surfdens_fd(pot, "R")
    ad, on_backend = _surfdens_ad(
        backend_name, pot, False, "R", _SURFDENS_R0, _SURFDENS_Z0
    )
    assert on_backend, f"{backend_name}: bare float, graph severed"
    assert abs(ad - fd) < 1e-7 * abs(fd), (
        f"{backend_name} opt-out potential d/dR: {ad!r} vs FD {fd!r} "
        f"(rel {abs(ad - fd) / abs(fd):.3e})"
    )
