###############################################################################
# test_backend_compile_workflows.py: whole galpy workflows under jax.jit and
# torch.compile, value AND gradient vs eager.
#
# The --jit CI rows compile each @backend_input entry point for its VALUE only
# (torch: dynamo with backend="eager"). They cannot see a workflow that is not
# one entry point (Orbit.integrate, a DF build, spray -> track), a failure that
# appears only in the backward pass, or a wrong derivative. This file compiles
# the workflows themselves; add a case whenever a new one turns up.
#
# A known gap is a STRICT xfail naming its cause: fixing it turns the case red
# (XPASS), so the entry is removed in the fixing PR. Workflows already compiled
# elsewhere are not repeated: jax.jit of spray sample/track
# (test_backend_streamspraydf), torch.compile of the C integrators
# (test_backend_orbit_stm) and of torchode (test_backend_torchode), kingdf W0
# (test_backend_kingdf).
###############################################################################
import warnings

import numpy
import pytest

import galpy.backend
from galpy.actionAngle import actionAngleSpherical, actionAngleStaeckel
from galpy.backend import random as grandom
from galpy.df import fardal15spraydf, isotropicHernquistdf, quasiisothermaldf
from galpy.orbit import Orbit
from galpy.potential import (
    HernquistPotential,
    LogarithmicHaloPotential,
    MWPotential2014,
    NFWPotential,
    evaluatePotentials,
    vcirc,
)

pytestmark = pytest.mark.backend_managed

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)
jnp = jax.numpy

_IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]


def _arr(bk, x):
    return torch.tensor(x) if bk == "torch" else jnp.asarray(x)


def _one(bk, x):  # a differentiated scalar as a length-1 array
    return x.reshape(1) if bk == "torch" else jnp.reshape(x, (1,))


# --- the workflows: f(backend, x) -> scalar, differentiated w.r.t. x -----------
def orbit_c_integrator(bk, x):
    # IC gradient through the C integrator + the energy accessor
    ic = [1.0, x, 1.1, 0.05, 0.05, 0.3]
    o = Orbit(
        torch.stack([torch.as_tensor(v) for v in ic])
        if bk == "torch"
        else jnp.stack(ic)
    )
    ts = _arr(bk, numpy.linspace(0.0, 5.0, 11))
    o.integrate(ts, MWPotential2014, method="dop853_c")
    return o.R(ts[-1]) + o.E(ts[-1])


def orbit_potential_parameter(bk, x):
    # potential-parameter gradient through the in-backend ODE
    ts = _arr(bk, numpy.linspace(0.0, 5.0, 11))
    o = Orbit(_arr(bk, _IC))
    method = "diffrax" if bk == "jax" else "torchode"
    o.integrate(ts, LogarithmicHaloPotential(amp=x, q=0.9), method=method)
    return o.x(ts[-1]) + o.vz(ts[-1])


def _spray(bk, x):
    return fardal15spraydf(
        3e-5,
        progenitor=Orbit(_IC),
        pot=LogarithmicHaloPotential(amp=x, q=0.9),
        tdisrupt=2.0,
    )


def spray_sample(bk, x):
    out = _spray(bk, x).sample(
        40, return_orbit=False, tail="leading", key=grandom.key(7, backend=bk)
    )
    return out[0].sum()


def spray_track(bk, x):
    tr = _spray(bk, x).streamTrack(
        n=40, tail="leading", velocity_weight=1.0, order=2, key=grandom.key(7, bk)
    )
    return tr._track_xyz.sum() + tr._cov_xyz.sum()


def actions_staeckel(bk, x):
    aA = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    c = [_arr(bk, [v]) for v in (0.1, 1.1, 0.05, 0.05)]
    jr, _, jz = aA(_one(bk, x), *c)
    return (jr + jz).sum()


def actions_spherical(bk, x):
    aA = actionAngleSpherical(pot=HernquistPotential(amp=x, a=1.5))
    c = [_arr(bk, [v]) for v in (1.0, 0.1, 1.1, 0.05, 0.05)]
    jr, _, jz = aA(*c)
    return (jr + jz).sum()


def sphericaldf_sample(bk, x):
    with galpy.backend.use(bk, force=True):
        d = isotropicHernquistdf(pot=HernquistPotential(amp=x, a=1.3))
        R, vR, *_ = d.sample(n=20, return_orbit=False, key=grandom.key(3, bk))
    return (R + vR**2).sum()


def qdf_density(bk, x):
    aA = actionAngleStaeckel(pot=MWPotential2014, c=True, delta=0.5)
    q = quasiisothermaldf(
        1.0 / 4.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential2014, aA=aA, cutcounter=True
    )
    c = [_arr(bk, [v]) for v in (0.1, 0.9, 0.05, 0.02)]
    return q(_one(bk, x), *c).sum()


def potential_evaluations(bk, x):
    p = NFWPotential(amp=x, a=2.0)
    return evaluatePotentials(p, _arr(bk, 1.1), _arr(bk, 0.2)) + vcirc(p, _arr(bk, 1.3))


def _gap(reason):
    return pytest.mark.xfail(reason=reason, strict=True)


_T = "torch.compile gap: "
_CASES = [
    # (backend, workflow, x0, marks)
    ("torch", orbit_c_integrator, 0.1, ()),
    (
        "jax",
        orbit_c_integrator,
        0.1,
        _gap("jax.jit gap: Orbit.E at a traced time (numpy.atleast_1d/tile)"),
    ),
    ("torch", orbit_potential_parameter, 1.1, ()),
    ("jax", orbit_potential_parameter, 1.1, ()),
    ("torch", spray_sample, 1.1, ()),
    ("torch", spray_track, 1.1, _gap(_T + "streamTrack's numpy fit path")),
    ("torch", actions_staeckel, 1.0, ()),
    ("jax", actions_staeckel, 1.0, ()),
    ("torch", actions_spherical, 2.0, ()),
    ("jax", actions_spherical, 2.0, ()),
    ("torch", sphericaldf_sample, 1.7, ()),
    ("jax", sphericaldf_sample, 1.7, ()),
    ("torch", qdf_density, 1.0, _gap(_T + "fake-tensor 0-d indexing, interpolate.py")),
    ("jax", qdf_density, 1.0, ()),
    ("torch", potential_evaluations, 2.0, ()),
    ("jax", potential_evaluations, 2.0, ()),
]


def _compiled(bk, workflow, x0):
    if bk == "jax":
        return jax.jit(jax.value_and_grad(lambda x: workflow("jax", x)))(x0)
    torch._dynamo.reset()
    with warnings.catch_warnings():
        # torch-internal deprecations under CI's -W error
        for msg in (".*script_method.*", ".*should not be instantiated.*"):
            warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        fc = torch.compile(lambda x: workflow("torch", x), backend="eager")
        xc = torch.tensor(x0, requires_grad=True)
        vc = fc(xc)
        return vc, torch.autograd.grad(vc, xc)[0]


@pytest.mark.parametrize(
    "bk,workflow,x0",
    [pytest.param(b, w, x, marks=m) for b, w, x, m in _CASES],
    ids=[f"{b}-{w.__name__}" for b, w, _, _ in _CASES],
)
def test_workflow_compiled_matches_eager(bk, workflow, x0):
    if bk == "jax":
        ve, ge = jax.value_and_grad(lambda x: workflow("jax", x))(x0)
    else:
        xe = torch.tensor(x0, requires_grad=True)
        ve = workflow("torch", xe)
        (ge,) = torch.autograd.grad(ve, xe)
    try:
        vc, gc = _compiled(bk, workflow, x0)
    except Exception as e:
        # re-raised from here: under coverage (py3.14 sys.monitoring) a dynamo
        # frame can carry tb_lineno=None, which crashes pytest's failure report
        # (INTERNALERROR) and with it the whole session
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:2000]}") from None
    assert float(ge) != 0.0, "gradient disconnected"
    # compiled == eager up to op reordering; measured <= 5e-14 value, 1e-12 grad
    numpy.testing.assert_allclose(float(vc), float(ve), rtol=1e-12)
    numpy.testing.assert_allclose(float(gc), float(ge), rtol=1e-10)
