###############################################################################
# test_backend_normalize.py: normalize= as a differentiable constructor
# parameter (gap audit, PR 16).
#
# Every constructor tested `if normalize or (...)`, Python truthiness that a
# traced normalize has not got (jax.jit: TracerBoolConversionError).
# _normalize_requested decides structurally. Rforce scales linearly with
# normalize, so d Rforce/d normalize = Rforce / normalize exactly: the check is
# against that, at 1e-12, plus the value against numpy's normalized potential.
###############################################################################
import contextlib

import numpy
import pytest

import galpy.potential as gp
from galpy import backend

pytestmark = pytest.mark.backend_managed

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

AD_BACKENDS = (["jax"] if jax is not None else []) + (
    ["torch"] if torch is not None else []
)

_NORMALIZABLE = [  # class, extra constructor kwargs
    (gp.PlummerPotential, {"b": 0.8}),
    (gp.MiyamotoNagaiPotential, {"a": 0.5, "b": 0.1}),
    (gp.DoubleExponentialDiskPotential, {"hr": 1.0 / 3.0, "hz": 1.0 / 16.0}),
    (gp.IsochronePotential, {"b": 0.7}),
    (gp.PseudoIsothermalPotential, {"a": 0.4}),
    (gp.LogarithmicHaloPotential, {"core": 0.1, "q": 0.9}),
    (gp.KuzminDiskPotential, {"a": 0.6}),
    (gp.KuzminKutuzovStaeckelPotential, {"ac": 3.0, "Delta": 0.4}),
    (gp.MN3ExponentialDiskPotential, {"hr": 0.3, "hz": 0.05}),
    (gp.RazorThinExponentialDiskPotential, {"hr": 0.3}),
    (gp.BurkertPotential, {"a": 2.0}),
    (gp.EinastoPotential, {"h": 2.0, "n": 1.3}),
    (gp.ExpTruncNFWPotential, {"a": 1.0, "rc": 2.0}),
    (gp.HomogeneousSpherePotential, {"R": 2.0}),
    (gp.SphericalShellPotential, {"a": 0.7}),
    (gp.RingPotential, {"a": 0.7}),
    (gp.PowerSphericalPotentialwCutoff, {"alpha": 1.0, "rc": 1.5}),
    (gp.PowerSphericalPotential, {"alpha": 1.5}),
    (gp.FlattenedPowerPotential, {"alpha": 0.5, "q": 0.9}),
    (gp.TwoPowerSphericalPotential, {"a": 2.0, "alpha": 1.5, "beta": 3.5}),
    (gp.NFWPotential, {"a": 2.0}),
    (gp.HernquistPotential, {"a": 2.0}),
    (gp.JaffePotential, {"a": 2.0}),
    (gp.DehnenSphericalPotential, {"a": 2.0, "alpha": 1.5}),
    (gp.DehnenCoreSphericalPotential, {"a": 2.0}),
    (gp.TwoPowerTriaxialPotential, {"a": 2.0, "b": 0.9, "c": 0.8}),
    (gp.TriaxialNFWPotential, {"a": 2.0, "b": 0.9, "c": 0.8}),
    (gp.TriaxialHernquistPotential, {"a": 2.0, "b": 0.9, "c": 0.8}),
    (gp.TriaxialJaffePotential, {"a": 2.0, "b": 0.9, "c": 0.8}),
    (gp.PowerTriaxialPotential, {"alpha": 1.5, "b": 0.9, "c": 0.8}),
    (gp.PerfectEllipsoidPotential, {"a": 2.0, "b": 0.9, "c": 0.8}),
    (gp.TriaxialGaussianPotential, {"sigma": 2.0, "b": 0.9, "c": 0.8}),
    (gp.FerrersPotential, {"a": 1.5, "b": 0.35, "c": 0.2375}),
    (gp.SoftenedNeedleBarPotential, {"a": 2.0, "b": 0.1, "c": 1.0}),
    (gp.AnySphericalPotential, {"dens": lambda r: 1.0 / (1.0 + r * r) ** 2.5}),
    (gp.SCFPotential, {"a": 1.2}),
]
_IDS = [c.__name__ for c, _ in _NORMALIZABLE]
_R, _Z = 1.1, 0.1
_N0 = 0.6


def _ctx(backend_name, mode):
    if mode == "forced":
        return backend.use(backend_name, force=True)
    return contextlib.nullcontext()


@pytest.mark.parametrize("mode", ["data", "forced"])
@pytest.mark.parametrize("cls,kw", _NORMALIZABLE, ids=_IDS)
@pytest.mark.parametrize("backend_name", AD_BACKENDS)
def test_normalize_gradient(backend_name, mode, cls, kw):
    ref = float(cls(normalize=_N0, **kw).Rforce(_R, _Z))
    if backend_name == "jax":
        R, z = (_R, _Z) if mode == "forced" else (jnp.asarray(_R), jnp.asarray(_Z))

        def f(n):
            with _ctx("jax", mode):
                return cls(normalize=n, **kw).Rforce(R, z)

        n = jnp.asarray(_N0)
        vals = [float(f(n)), float(jax.jit(f)(n))]
        grads = [float(jax.grad(f)(n)), float(jax.jit(jax.grad(f))(n))]
    else:
        torch.set_default_dtype(torch.float64)
        R, z = (_R, _Z) if mode == "forced" else (torch.tensor(_R), torch.tensor(_Z))
        n = torch.tensor(_N0, requires_grad=True)
        with _ctx("torch", mode):
            out = cls(normalize=n, **kw).Rforce(R, z)
        (g,) = torch.autograd.grad(out, n)
        vals, grads = [float(out.detach())], [float(g)]
    for v in vals:
        assert abs(v / ref - 1.0) < 1e-12, (vals, ref)
    for g in grads:
        assert abs(g / (ref / _N0) - 1.0) < 1e-12, (grads, ref / _N0)


def test_normalize_requested_matches_the_old_truthiness():
    from galpy.potential.Potential import _normalize_requested

    for n in (False, None, True, 0, 0.0, 1, 0.5, numpy.float64(0.0)):
        old = bool(n) or (isinstance(n, (int, float)) and not isinstance(n, bool))
        assert _normalize_requested(n) is old, n
    if jax is not None:
        assert _normalize_requested(jnp.asarray(0.0))
