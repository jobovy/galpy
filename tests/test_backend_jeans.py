###############################################################################
# test_backend_jeans.py: Track F Pdf.1 -- backend (jax/torch) coverage for
# jeans (sigmar/sigmalos). The numpy path is byte-identical (test_jeans
# unchanged); this exercises the resolved-namespace dispatch (parity
# numpy<->jax<->torch + grad-vs-FD) that makes them evaluate AND differentiate
# under every backend. One file per df module (surfaceSigmaProfile is in
# test_backend_surfacesigma.py; each later df family gets its own file).
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

from galpy.backend import as_numpy
from galpy.df import jeans
from galpy.potential import HernquistPotential


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _is_backend_array(backend, x):
    if backend == "jax":
        return isinstance(x, jax.Array)
    return torch.is_tensor(x)


_HP = HernquistPotential(normalize=1.0, a=1.3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["sigmar", "sigmalos"])
def test_jeans_parity(backend, fn):
    # numpy<->backend parity of the fixed-order-GL semi-infinite integral (the
    # numpy path uses scipy.quad; the backend path fixed_quad_semiinfinite).
    f = getattr(jeans, fn)
    for r0 in (0.7, 1.1, 1.6):
        ref = f(_HP, r0, use_physical=False)
        got = f(_HP, _arr(backend, r0), use_physical=False)
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(
            as_numpy(got), numpy.asarray(ref), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_jeans_sigmar_callable_beta_parity(backend):
    # callable (r-dependent) anisotropy: exercises the backend intFactor path
    # exp(2 * quad(beta(y)/y)) and its numpy<->backend parity (the closed-form
    # power-law intFactor is used for the constant-beta case above).
    beta = lambda r: 0.2 / (1.0 + r)
    for r0 in (0.7, 1.1, 1.6):
        ref = jeans.sigmar(_HP, r0, beta=beta, use_physical=False)
        got = jeans.sigmar(_HP, _arr(backend, r0), beta=beta, use_physical=False)
        assert _is_backend_array(backend, got)
        numpy.testing.assert_allclose(
            as_numpy(got), numpy.asarray(ref), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_jeans_sigmar_grad_vs_fd(backend):
    # d(sigma_r)/dr via backend autodiff vs central finite difference -- the
    # differentiability that motivates the migration.
    r0, eps = 1.1, 1e-5
    fd = (
        jeans.sigmar(_HP, r0 + eps, use_physical=False)
        - jeans.sigmar(_HP, r0 - eps, use_physical=False)
    ) / (2.0 * eps)
    if backend == "jax":
        g = float(
            jax.grad(lambda r: jeans.sigmar(_HP, r, use_physical=False))(
                jnp.asarray(r0)
            )
        )
    else:
        t = torch.tensor(r0, requires_grad=True)
        jeans.sigmar(_HP, t, use_physical=False).backward()
        g = float(t.grad)
    numpy.testing.assert_allclose(g, fd, rtol=1e-4)


# --- forced-backend dispatch: numpy-typed input must NOT pin to scipy --------
# This file's header says it exercises "resolved-namespace dispatch", but the
# module actually guarded on the DATA:
#     xp = get_namespace(r) if is_backend_array(r) else numpy
# `get_namespace` already honours a forced context ("forced default beats the
# data"), so that test overrode it and pinned numpy-typed input to scipy -- a
# numpy island: under `use(backend, force=True)` a plain float went down the
# non-differentiable path, silently. Nothing else here catches it, because every
# other test passes an already-backend array, which the guard let through.
#
# ASSERT ON THE PATH, NOT THE RETURN TYPE. Under forced torch the scipy result is
# converted to a Tensor downstream, so `is_tensor(result)` is True even when the
# whole computation ran on scipy -- a type check passes for the wrong reason on
# torch while correctly failing on jax. Spy on the backend quadrature instead.
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fn", ["sigmar", "sigmalos"])
def test_forced_backend_takes_the_backend_path_for_numpy_input(
    backend, fn, monkeypatch
):
    from galpy.backend import use

    calls = []
    real = jeans.fixed_quad_semiinfinite
    monkeypatch.setattr(
        jeans,
        "fixed_quad_semiinfinite",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    with use(backend, force=True):
        getattr(jeans, fn)(_HP, numpy.float64(1.0))
    assert calls, (
        f"jeans.{fn} made no backend-quadrature call under a FORCED {backend} "
        "backend: numpy-typed input is still pinned to the scipy path"
    )


@pytest.mark.parametrize("fn", ["sigmar", "sigmalos"])
def test_numpy_path_unchanged_without_a_forced_backend(fn, monkeypatch):
    # The other half of the contract: with no forced backend, numpy input must
    # still resolve to numpy and go to scipy -- zero backend-quadrature calls.
    calls = []
    real = jeans.fixed_quad_semiinfinite
    monkeypatch.setattr(
        jeans,
        "fixed_quad_semiinfinite",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    got = getattr(jeans, fn)(_HP, numpy.float64(1.0))
    assert not calls, f"jeans.{fn} took the backend path with no forced backend"
    assert isinstance(got, numpy.floating), type(got).__name__


# --- accuracy at small r -----------------------------------------------------
# The semi-infinite map's default transition scale is 1, so as r -> 0 the nodes
# stop resolving the integrand and the backend answer diverges from the truth
# (1.1e-02 at r=1e-3 -- not a tolerance question, a wrong number). sigmar passes
# scale=r to make the map scale-invariant; this pins that down against the
# CLOSED FORM, not against numpy, so it cannot be satisfied by agreeing with an
# equally-wrong reference.
#
#   Logarithmic halo (vc=1), dens ~ r**gamma, beta(r) = -b*r  =>
#   sigma_r(r)**2 = Gamma(-gamma) * Gammainc_upper(-gamma, 2 b r)
#                   / ((2 b r)**-gamma * exp(-2 b r))
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("r", [1e-3, 1e-2, 1e-1, 1.0, 5.0])
def test_sigmar_small_r_matches_closed_form(backend, r):
    from scipy import special

    from galpy.backend import use
    from galpy.potential import LogarithmicHaloPotential

    gamma, b = -0.1, 3.0
    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    exact = numpy.sqrt(
        special.gamma(-gamma)
        * special.gammaincc(-gamma, 2.0 * b * r)
        / ((2.0 * b * r) ** -gamma * numpy.exp(-2.0 * b * r))
    )
    with use(backend, force=True):
        got = jeans.sigmar(
            lp, _arr(backend, r), beta=lambda x: -b * x, dens=lambda x: x**-gamma
        )
    assert _is_backend_array(backend, got)
    # 1e-10 absolute is the accuracy scipy's adaptive quad reaches here; the
    # backend now matches it rather than merely being close.
    assert numpy.fabs(float(as_numpy(got)) - exact) < 1e-10, (
        f"sigmar off by {numpy.fabs(float(as_numpy(got)) - exact):.3e} at r={r}"
    )
