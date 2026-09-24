###############################################################################
# test_backend_potential_edges.py: potential quantities differentiated w.r.t. a
# PARAMETER at the radial edges r = 0 and r = inf.
#
# Closed forms written in a/r or r/a ("so it works for r=numpy.inf") evaluate
# finite at the edge where that ratio is infinite, but their backward is
# 0 * inf = NaN there -- e.g. Hernquist's mass(0) = 0 with d/da = NaN. That
# poisons anything that touches the edge: a CMF grid starting at r=0, a
# potential-at-infinity reference, a density tail. galpy.backend.radial_limits
# evaluates at a benign radius and selects the known limit on a backend path.
#
# Reference: the gradient must equal a central finite difference of the NUMPY
# value at the edge (exactly 0 for parameter-independent limits), and the
# backend value must equal numpy's.
###############################################################################
import math

import numpy
import pytest

import galpy.potential as P
from galpy import backend
from galpy.backend import as_numpy, radial_limits
from galpy.potential import evaluateDensities, evaluatePotentials, mass

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

_POTS = {
    "Hernquist": lambda p: P.HernquistPotential(amp=2.0, a=p),
    "NFW": lambda p: P.NFWPotential(amp=2.0, a=p),
    "Jaffe": lambda p: P.JaffePotential(amp=2.0, a=p),
    "TwoPower": lambda p: P.TwoPowerSphericalPotential(
        amp=2.0, a=p, alpha=1.5, beta=3.5
    ),
    "Dehnen": lambda p: P.DehnenSphericalPotential(amp=2.0, a=p, alpha=1.5),
    "DehnenCore": lambda p: P.DehnenCoreSphericalPotential(amp=2.0, a=p),
    "Plummer": lambda p: P.PlummerPotential(amp=2.0, b=p),
    "PowerSphCut": lambda p: P.PowerSphericalPotentialwCutoff(amp=2.0, alpha=1.5, rc=p),
    "PowerSph": lambda p: P.PowerSphericalPotential(amp=2.0, alpha=p),
    "Burkert": lambda p: P.BurkertPotential(amp=2.0, a=p),
    "Einasto": lambda p: P.EinastoPotential(amp=2.0, h=p, n=2.0),
    "PseudoIsothermal": lambda p: P.PseudoIsothermalPotential(amp=2.0, a=p),
}
_Q = {
    "mass": lambda pot, r: mass(pot, r, use_physical=False),
    "Phi": lambda pot, r: evaluatePotentials(pot, r, 0.0, use_physical=False),
    "dens": lambda pot, r: evaluateDensities(pot, r, 0.0, use_physical=False),
}
# (potential, quantity, edge): every case whose VALUE is finite at the edge
# but whose parameter gradient was NaN
_CASES = [
    ("Hernquist", "mass", 0.0),
    ("Hernquist", "Phi", math.inf),
    ("Hernquist", "dens", math.inf),
    ("NFW", "dens", math.inf),
    ("Jaffe", "mass", 0.0),
    ("Jaffe", "dens", math.inf),
    ("TwoPower", "Phi", math.inf),
    ("TwoPower", "dens", math.inf),
    ("Dehnen", "mass", 0.0),
    ("Dehnen", "Phi", 0.0),
    ("Dehnen", "dens", math.inf),
    ("DehnenCore", "mass", 0.0),
    ("DehnenCore", "Phi", 0.0),
    ("DehnenCore", "dens", math.inf),
    ("Plummer", "mass", 0.0),
    ("PowerSphCut", "mass", 0.0),
    ("PowerSphCut", "mass", math.inf),
    ("PowerSphCut", "Phi", math.inf),
    ("PowerSphCut", "dens", math.inf),
    ("PowerSph", "dens", math.inf),
    ("Burkert", "Phi", math.inf),
    ("Burkert", "dens", math.inf),
    ("Einasto", "Phi", math.inf),
    ("Einasto", "dens", 0.0),
    ("Einasto", "dens", math.inf),
    ("PseudoIsothermal", "dens", math.inf),
]
_P0 = 1.2


def _numpy_value(name, q, r, p):
    return float(_Q[q](_POTS[name](p), numpy.array([r]))[0])


@pytest.mark.parametrize("name,q,r", _CASES, ids=[f"{n}-{q}@{r}" for n, q, r in _CASES])
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_edge_value_and_parameter_gradient(backend_name, name, q, r):
    with backend.use(backend_name, force=True):
        if backend_name == "jax":
            f = lambda p: _Q[q](_POTS[name](p), jnp.asarray(r))  # noqa: E731
            val, ad = float(f(_P0)), float(jax.grad(f)(_P0))
        else:
            p = torch.tensor(_P0, requires_grad=True)
            out = _Q[q](_POTS[name](p), torch.tensor(r))
            (g,) = torch.autograd.grad(out, p)
            val, ad = float(out), float(g)
    ref = _numpy_value(name, q, r, _P0)
    assert math.isfinite(ad), f"{name} d{q}/dp at r={r} is {ad}"
    numpy.testing.assert_allclose(val, ref, rtol=1e-12, atol=0.0)
    h = 1e-6
    fd = (_numpy_value(name, q, r, _P0 + h) - _numpy_value(name, q, r, _P0 - h)) / (
        2.0 * h
    )
    numpy.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_radial_limits_passes_non_backend_input_through(backend_name):
    # numpy -- and a plain scalar under a FORCED backend -- must get fn(r)
    # itself: byte-identical, and no backend op applied to a python float
    sentinel = object()
    for r in (numpy.array([0.0, 1.0, numpy.inf]), 0.0, math.inf):
        with backend.use(backend_name, force=True):
            assert radial_limits(r, lambda r: sentinel, at0=1.0, atinf=2.0) is sentinel
    arr = jnp.asarray if backend_name == "jax" else torch.tensor
    got = radial_limits(
        arr([0.0, 2.0, math.inf]), lambda r: 1.0 / r, at0=5.0, atinf=7.0
    )
    numpy.testing.assert_array_equal(as_numpy(got), [5.0, 0.5, 7.0])
