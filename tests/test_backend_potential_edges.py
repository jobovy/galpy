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
    "Isochrone": lambda p: P.IsochronePotential(amp=2.0, b=p),
    "Logarithmic": lambda p: P.LogarithmicHaloPotential(amp=2.0, core=p),
    "Kepler": lambda p: P.KeplerPotential(amp=p),
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
    # NaN on numpy too before (the numpy value now carries the limit as well)
    ("TwoPower", "mass", math.inf),
    ("Burkert", "mass", 0.0),
    ("Burkert", "Phi", 0.0),
    ("Einasto", "mass", 0.0),
    ("Einasto", "mass", math.inf),
    ("Isochrone", "dens", math.inf),
    ("Logarithmic", "dens", math.inf),
    ("Kepler", "dens", 0.0),
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


def test_radial_limits_numpy_too():
    # numpy input gets the limits too, but only at the edge entries; without an
    # edge present it is fn(r) itself (byte-identical)
    sentinel = object()
    assert (
        radial_limits(numpy.array([1.0, 2.0]), lambda r: sentinel, 1.0, 2.0, True)
        is sentinel
    )
    got = radial_limits(
        numpy.array([0.0, 2.0, math.inf]),
        lambda r: 1.0 / r,
        at0=5.0,
        atinf=7.0,
        numpy_too=True,
    )
    numpy.testing.assert_array_equal(got, [5.0, 0.5, 7.0])
    assert radial_limits(0.0, lambda r: 1.0 / r, at0=5.0, numpy_too=True) == 5.0


# --- small r: the backend matches numpy, and d/d(parameter) is not noise -------
# The closed forms lost ~eps/x^k at x << 1; their parameter gradients inherited
# that noise. (potential, quantity, x = r/scale)
_SMALL_R = {
    "NFW": (
        lambda p: P.NFWPotential(amp=2.0, a=p),
        ("Phi", "Rforce", "R2deriv", "mass"),
    ),
    "Burkert": (lambda p: P.BurkertPotential(amp=2.0, a=p), ("Phi", "Rforce")),
    "Einasto": (lambda p: P.EinastoPotential(amp=2.0, h=p, n=2.0), ("Phi", "Rforce")),
    "TwoPower": (
        lambda p: P.TwoPowerSphericalPotential(amp=2.0, a=p, alpha=0.5, beta=4.0),
        ("Phi",),
    ),
}
_SQ = {
    "Phi": lambda pot, r: evaluatePotentials(pot, r, 0.0, use_physical=False),
    "Rforce": lambda pot, r: P.evaluateRforces(pot, r, 0.0, use_physical=False),
    "R2deriv": lambda pot, r: P.evaluateR2derivs(pot, r, 0.0, use_physical=False),
    "mass": lambda pot, r: mass(pot, r, use_physical=False),
}
# (potential, quantity, x, value, d/dp) at fixed r = x * 1.2, p = 1.2, amp = 2:
# 50-digit mpmath references (a finite difference of the numpy value cannot
# resolve a d/dp of ~6e-16 at x = 1e-8)
_SMALL_CASES = [
    ("NFW", "Phi", 1e-8, -1.6666666583333334, 1.3888888750000001),
    ("NFW", "Phi", 1e-4, -1.6665833388884723, 1.3887500138875001),
    ("NFW", "Rforce", 1e-8, -6.9444443518518529e-1, 1.1574073842592596),
    ("NFW", "Rforce", 1e-4, -6.9435186226740752e-1, 1.1571759606435191),
    ("NFW", "R2deriv", 1e-8, -7.716049209104941e-1, 1.9290122878086431),
    ("NFW", "R2deriv", 1e-4, -7.7143135493441408e-1, 1.9284337576967622),
    ("NFW", "mass", 1e-8, 9.9999998666666682e-17, -1.6666666333333338e-16),
    ("NFW", "mass", 1e-4, 9.9986668166506683e-9, -1.6663333833266675e-8),
    ("Burkert", "Phi", 1e-8, -2.8424460675137352e1, -4.7374101125228921e1),
    ("Burkert", "Phi", 1e-4, -2.842446061482179e1, -4.7374101125226408e1),
    ("Burkert", "Rforce", 1e-8, -1.0053096416089115e-7, -6.2831853071795865e-16),
    ("Burkert", "Rforce", 1e-4, -1.0052342509250477e-3, -6.2831853071652265e-8),
    ("Einasto", "Phi", 1e-8, -4.3429376843225302e2, -7.2382294738708836e2),
    ("Einasto", "Phi", 1e-4, -4.3429376837234655e2, -7.2382294738691728e2),
    ("Einasto", "Rforce", 1e-8, -1.0052234835200349e-7, -3.590077458799477e-12),
    ("Einasto", "Rforce", 1e-4, -9.9673029696052965e-4, -3.5591148859781171e-6),
    ("TwoPower", "Phi", 1e-8, -4.44444444444e-1, 3.7037037036944444e-1),
    ("TwoPower", "Phi", 1e-4, -4.4444400006665833e-1, 3.7036944463885764e-1),
]


@pytest.mark.parametrize(
    "name,q,x,ref,dref",
    _SMALL_CASES,
    ids=[f"{c[0]}-{c[1]}@{c[2]}" for c in _SMALL_CASES],
)
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_small_r_value_and_parameter_gradient(backend_name, name, q, x, ref, dref):
    build = _SMALL_R[name][0]
    r = x * _P0
    with backend.use(backend_name, force=True):
        if backend_name == "jax":
            f = lambda p: _SQ[q](build(p), jnp.asarray(r))  # noqa: E731
            val, ad = float(f(_P0)), float(jax.grad(f)(_P0))
        else:
            p = torch.tensor(_P0, requires_grad=True)
            out = _SQ[q](build(p), torch.tensor(r))
            (g,) = torch.autograd.grad(out, p)
            val, ad = float(out), float(g)
    numpy.testing.assert_allclose(val, ref, rtol=2e-14)
    numpy.testing.assert_allclose(ad, dref, rtol=1e-12)
