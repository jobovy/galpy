###############################################################################
# test_backend_constructor_jit.py: BUILDING a potential or a DF inside a traced,
# differentiated function.
#
# test_backend_potential_jit.py traces the EVALUATION of an already-built
# potential. This file covers the other half, which is what a fit actually
# needs: the scale radius / exponent / anisotropy being optimised is a tracer,
# so the object is constructed inside jax.jit(jax.grad(...)) and the constructor
# itself has to be traceable.
#
# Constructors are full of Python-level decisions -- min(a, rc) to size a
# series-expansion threshold, `if beta <= 2` validation, `if 2-alpha == 0` for a
# degenerate closed form, asserts on the DF exponents -- and every one of them
# needs a truth value that a tracer does not have. The rule adopted here is
# galpy.backend.concretely_true: a check fires only when it can be PROVEN to
# fire, and a special case is taken only when it can be PROVEN to apply, so
# concrete construction is unchanged and traced construction falls through to
# the generic (differentiable) branch.
#
# Gradients are checked against central finite differences, never merely for
# being finite and non-zero, and the validation errors are checked to still
# raise on concrete bad input -- a guard that silently disables the check would
# otherwise look like a pass.
#
# Backends that are not installed self-skip, so this is green on numpy alone.
###############################################################################
import numpy
import pytest

from galpy.backend import concretely_true, use
from galpy.df import constantbetaPowerLawdf
from galpy.potential import (
    ExpTruncNFWPotential,
    PowerSphericalPotential,
    TwoPowerTriaxialPotential,
)

# This module manages backends explicitly (parametrizes over them), so it is
# exempt from the global --backend force fixture.
pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None

requires_jax = pytest.mark.skipif(jax is None, reason="jax not installed")
requires_torch = pytest.mark.skipif(torch is None, reason="torch not installed")


def _central_diff(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _jit_grad(f, x):
    """d f/dx under jax.jit(jax.grad(...)) -- the constructor is TRACED."""
    return float(jax.jit(jax.grad(f))(x))


# ---------------------------------------------------------------------------
# the helper itself
# ---------------------------------------------------------------------------
def test_concretely_true_matches_bool_when_concrete():
    for v in (True, False, 1.0 > 0.0, 1.0 < 0.0, numpy.bool_(True), numpy.bool_(False)):
        assert concretely_true(v) is bool(v)
    # a multi-element array has no truth value either, and must not raise here
    assert concretely_true(numpy.array([True, True])) is False


@requires_jax
def test_concretely_true_is_false_on_both_branches_under_trace():
    # The property that makes it safe: while tracing it answers False for a
    # predicate AND for its negation, so `if concretely_true(p)` and
    # `if concretely_true(not p)` both fall through rather than one of them
    # silently firing on a tracer.
    seen = []

    def probe(x):
        seen.append((concretely_true(x > 0.0), concretely_true(x <= 0.0)))
        return x

    jax.jit(probe)(1.0)
    assert seen == [(False, False)]


# ---------------------------------------------------------------------------
# ExpTruncNFWPotential: min(a, rc) sizes the small-r series threshold
# ---------------------------------------------------------------------------
@requires_jax
@pytest.mark.parametrize("R", [1.2, 1e-5])
def test_exptruncnfw_scale_radius_gradient_under_jit(R):
    # R = 1.2 takes the closed form, R = 1e-5 the small-r series -- the two
    # sides of the _small_r_thresh = 1e-3 * min(a, rc) split that min() used to
    # make untraceable.
    with use("jax", force=True):
        Rb, zb = jnp.asarray(R), jnp.asarray(0.0 if R < 1e-3 else 0.1)

        def f(a):
            return ExpTruncNFWPotential(amp=1.0, a=a, rc=2.0).Rforce(Rb, zb)

        assert _jit_grad(f, 1.0) == pytest.approx(_central_diff(f, 1.0), rel=1e-7)


@requires_jax
def test_exptruncnfw_threshold_still_selects_the_series_branch():
    # The threshold has to keep its VALUE, not just trace: min(a, rc) is rc here.
    with use("jax", force=True):
        p = ExpTruncNFWPotential(amp=1.0, a=3.0, rc=0.5)
        assert float(p._small_r_thresh) == pytest.approx(1e-3 * 0.5, rel=1e-15)


@requires_torch
def test_exptruncnfw_scale_radius_gradient_torch():
    with use("torch", force=True):
        a = torch.tensor(1.0, requires_grad=True)
        out = ExpTruncNFWPotential(amp=1.0, a=a, rc=2.0).Rforce(
            torch.tensor(1.2), torch.tensor(0.1)
        )
        (g,) = torch.autograd.grad(out, a)

        def f(av):
            return float(ExpTruncNFWPotential(amp=1.0, a=av, rc=2.0).Rforce(1.2, 0.1))

        assert float(g) == pytest.approx(_central_diff(f, 1.0), rel=1e-7)


# ---------------------------------------------------------------------------
# TwoPowerTriaxialPotential: the alpha/beta validation and the alpha == 2 case
# ---------------------------------------------------------------------------
@requires_jax
@pytest.mark.parametrize(
    "name,x0", [("alpha", 1.0), ("beta", 3.5)], ids=["d/dalpha", "d/dbeta"]
)
def test_twopower_triaxial_exponent_gradient_under_jit(name, x0):
    with use("jax", force=True):
        Rb, zb = jnp.asarray(1.2), jnp.asarray(0.1)

        def f(x):
            kw = {"alpha": 1.0, "beta": 3.0, name: x}
            return TwoPowerTriaxialPotential(amp=1.0, a=1.0, b=0.9, c=0.8, **kw).Rforce(
                Rb, zb
            )

        assert _jit_grad(f, x0) == pytest.approx(_central_diff(f, x0), rel=1e-7)


def test_twopower_triaxial_validation_still_raises_when_concrete():
    # The guard must not have turned the check off for ordinary construction.
    with pytest.raises(OSError):
        TwoPowerTriaxialPotential(amp=1.0, a=1.0, alpha=1.0, beta=1.5)
    with pytest.raises(OSError):
        TwoPowerTriaxialPotential(amp=1.0, a=1.0, alpha=3.5, beta=4.0)


def test_twopower_triaxial_alpha_two_keeps_its_own_closed_form():
    # alpha == 2 makes 2-alpha == 0 and the generic psi has a 1/(2-alpha) pole,
    # so the degenerate branch must still be taken when alpha is concrete.
    p = TwoPowerTriaxialPotential(amp=1.0, a=1.0, alpha=2.0, beta=4.0, b=0.9, c=0.8)
    assert not hasattr(p, "psi_inf")
    assert numpy.isfinite(p.Rforce(1.2, 0.1))


@requires_torch
def test_twopower_triaxial_alpha_gradient_torch():
    with use("torch", force=True):
        alpha = torch.tensor(1.0, requires_grad=True)
        out = TwoPowerTriaxialPotential(
            amp=1.0, a=1.0, alpha=alpha, beta=3.0, b=0.9, c=0.8
        ).Rforce(torch.tensor(1.2), torch.tensor(0.1))
        (g,) = torch.autograd.grad(out, alpha)

        def f(av):
            return float(
                TwoPowerTriaxialPotential(
                    amp=1.0, a=1.0, alpha=av, beta=3.0, b=0.9, c=0.8
                ).Rforce(1.2, 0.1)
            )

        assert float(g) == pytest.approx(_central_diff(f, 1.0), rel=1e-7)


# ---------------------------------------------------------------------------
# constantbetaPowerLawdf: the anisotropy parameter itself
# ---------------------------------------------------------------------------
@requires_jax
def test_constantbetapowerlawdf_beta_gradient_under_jit():
    # d f_E / d beta with the DF BUILT inside the traced function: the four
    # normalizability asserts all read beta (directly or through the exponents
    # n and p), and _handle_rmin's Phi(0) probe sat in front of the explicit
    # rmin.
    with use("jax", force=True):
        pot = PowerSphericalPotential(amp=1.0, alpha=2.5)
        E = jnp.asarray(-1.0)

        def f(beta):
            return constantbetaPowerLawdf(pot=pot, beta=beta, rmax=100.0, rmin=1e-4).fE(
                E
            )

        assert _jit_grad(f, 0.3) == pytest.approx(_central_diff(f, 0.3), rel=1e-7)


def test_constantbetapowerlawdf_asserts_still_fire_when_concrete():
    with pytest.raises(AssertionError, match="beta must be < 1"):
        constantbetaPowerLawdf(
            pot=PowerSphericalPotential(amp=1.0, alpha=2.5), beta=1.5, rmin=1e-4
        )
    with pytest.raises(AssertionError, match="alpha must be > 2"):
        constantbetaPowerLawdf(
            pot=PowerSphericalPotential(amp=1.0, alpha=1.5), beta=0.3, rmin=1e-4
        )


@requires_torch
def test_constantbetapowerlawdf_beta_gradient_torch():
    with use("torch", force=True):
        pot = PowerSphericalPotential(amp=1.0, alpha=2.5)
        beta = torch.tensor(0.3, requires_grad=True)
        out = constantbetaPowerLawdf(pot=pot, beta=beta, rmax=100.0, rmin=1e-4).fE(
            torch.tensor(-1.0)
        )
        (g,) = torch.autograd.grad(out, beta)

        def f(b):
            return float(
                constantbetaPowerLawdf(
                    pot=PowerSphericalPotential(amp=1.0, alpha=2.5),
                    beta=b,
                    rmax=100.0,
                    rmin=1e-4,
                ).fE(-1.0)
            )

        assert float(g) == pytest.approx(_central_diff(f, 0.3), rel=1e-7)
