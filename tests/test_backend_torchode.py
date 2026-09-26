###############################################################################
# test_backend_torchode.py: Orbit.integrate(method='torchode'), the torch
# in-backend ODE integrator that torch.compile (inductor) can compile. Checked
# against galpy's C dop853_c integrator (values), finite differences of it
# (gradients), and eager torchode (compiled values/gradients).
#
# Self-skips unless torch and torchode are installed.
###############################################################################
import warnings

import numpy
import pytest

torch = pytest.importorskip("torch")
to = pytest.importorskip("torchode")
torch.set_default_dtype(torch.float64)

from galpy.backend._reference import integrate_orbit  # noqa: E402
from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import (  # noqa: E402
    DehnenBarPotential,
    HernquistPotential,
    MiyamotoNagaiPotential,
    MWPotential2014,
    PlummerPotential,
    toVerticalPotential,
)

pytestmark = pytest.mark.backend_managed

_POT = MiyamotoNagaiPotential(normalize=0.6, a=0.5, b=0.05) + HernquistPotential(
    normalize=0.4, a=2.0
)
_TS = numpy.linspace(0.0, 10.0, 21)
_IC = [1.0, 0.1, 1.1, 0.05, 0.05, 0.3]
_ICS = [_IC, [0.9, -0.1, 1.0, 0.1, 0.0, 1.0]]


def _c_orbit(ic, ts, pot):
    o = Orbit(ic)
    o.integrate(ts, pot, method="dop853_c")
    return o.getOrbit()


def _torchode_orbit(ic, ts, pot, **kw):
    o = Orbit(torch.tensor(ic))
    o.integrate(torch.tensor(ts), pot, method="torchode", **kw)
    return o.getOrbit().numpy()


@pytest.mark.parametrize(
    "ic,ts,pot",
    [
        (_IC, _TS, _POT),  # 6D single
        (_ICS, _TS, _POT),  # 6D batch, shared grid
        (_ICS, numpy.stack([_TS, 2.0 * _TS]), _POT),  # per-orbit grids
        (_IC, _TS[::-1].copy(), _POT),  # backward in time
        ([1.0, 0.1, 1.1, 0.2], _TS, _POT),  # planar
        ([0.1, 0.2], _TS, toVerticalPotential(_POT, 1.0)),  # 1D
        (_ICS, _TS, MWPotential2014 + DehnenBarPotential()),  # time-dependent
    ],
    ids=["6d", "batch", "per_orbit_t", "backward", "planar", "1d", "tdep"],
)
def test_torchode_matches_c(ic, ts, pot):
    got = _torchode_orbit(ic, ts, pot)
    ref = _c_orbit(ic, ts, pot)
    numpy.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_torchode_tsit5_matches_c():
    got = _torchode_orbit(_ICS, _TS, _POT, inbackend_kwargs={"solver": "tsit5"})
    numpy.testing.assert_allclose(got, _c_orbit(_ICS, _TS, _POT), rtol=1e-9, atol=1e-9)


def test_torchode_grad_ic_matches_fd():
    # d(R + vz)(t_end)/d(IC) vs central differences of the C integrator
    x = torch.tensor(_IC, requires_grad=True)
    o = Orbit(x)
    o.integrate(torch.tensor(_TS), _POT, method="torchode")
    (g,) = torch.autograd.grad(o.R(_TS[-1]) + o.vz(_TS[-1]), x)
    h = 1e-6
    fd = []
    for i in range(6):
        p, m = list(_IC), list(_IC)
        p[i] += h
        m[i] -= h
        a, b = _c_orbit(p, _TS, _POT)[-1], _c_orbit(m, _TS, _POT)[-1]
        fd.append(((a[0] + a[4]) - (b[0] + b[4])) / (2.0 * h))
    fd = numpy.array(fd)
    assert numpy.fabs(g.numpy() - fd).max() < 1e-8 * numpy.fabs(fd).max()


def _final_x(ic, amp, ts):
    pot = MiyamotoNagaiPotential(normalize=0.6, a=0.5, b=0.05) + HernquistPotential(
        amp=amp, a=2.0
    )
    o = Orbit(ic)
    o.integrate(ts, pot, method="dop853_c" if isinstance(ic, list) else "torchode")
    return o.getOrbit()[-1, 0]


def test_torchode_grad_potential_parameter_matches_fd():
    ts = _TS[:11]
    amp = torch.tensor(1.3, requires_grad=True)
    (g,) = torch.autograd.grad(_final_x(torch.tensor(_IC), amp, torch.tensor(ts)), amp)
    h = 1e-6
    fd = (_final_x(_IC, 1.3 + h, ts) - _final_x(_IC, 1.3 - h, ts)) / (2.0 * h)
    assert abs(float(g) - fd) < 1e-8 * abs(fd)


def test_torchode_hessian_matches_fd_of_grad():
    ts = torch.tensor(_TS[:3])

    def s(x):
        o = Orbit(x)
        o.integrate(ts, _POT, method="torchode")
        return o.getOrbit()[-1, 0]

    def grad(x):
        x = x.detach().requires_grad_()
        return torch.autograd.grad(s(x), x)[0]

    x = torch.tensor(_IC)
    H = torch.autograd.functional.hessian(s, x)
    h = 1e-5
    e = torch.eye(6)
    fdH = torch.stack(
        [(grad(x + h * e[i]) - grad(x - h * e[i])) / (2 * h) for i in range(6)]
    )
    assert float((H - fdH).abs().max()) < 1e-7 * float(H.abs().max())


def test_torchode_errors():
    ic = torch.tensor(_IC)
    ts = torch.tensor(_TS)
    with pytest.raises(ValueError, match="torchode solver must be one of"):
        integrate_orbit(_POT, ic, ts, engine="torchode", solver="dopri8")
    o = Orbit(ic)
    with pytest.raises(RuntimeError, match="torchode integration failed"):
        o.integrate(ts, _POT, method="torchode", inbackend_kwargs={"max_steps": 10})


def test_torch_compile_torchode_orbit_matches_eager():
    # inductor, value AND gradient, at two tolerances in ONE process: torchode's
    # 0-d-tensor rtol (as torch.add's alpha=) is baked into the first compile
    # unguarded, so without galpy's workaround the second tolerance silently
    # reused the first one's
    ts = torch.tensor(_TS[:11])

    def f(x, tol):
        o = Orbit(x)
        o.integrate(
            ts, PlummerPotential(normalize=1.0), method="torchode", rtol=tol, atol=tol
        )
        return o.x(ts[-1]) + o.vz(ts[-1])

    torch._dynamo.reset()
    with warnings.catch_warnings():
        # torch-internal deprecations under CI's -W error: inductor's import
        # uses torch.jit.script_method, its lowering torch._prims_common.check
        warnings.filterwarnings(
            "ignore", message=".*script_method.*", category=DeprecationWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*_prims_common.check.*", category=FutureWarning
        )
        fc = torch.compile(f)
        for tol in (1e-6, 1e-12):
            xe = torch.tensor(_IC, requires_grad=True)
            ve = f(xe, tol)
            (ge,) = torch.autograd.grad(ve, xe)
            xc = torch.tensor(_IC, requires_grad=True)
            vc = fc(xc, tol)
            assert vc.grad_fn is not None
            (gc,) = torch.autograd.grad(vc, xc)
            numpy.testing.assert_allclose(
                float(vc.detach()), float(ve.detach()), rtol=1e-12
            )
            numpy.testing.assert_allclose(
                gc.numpy(), ge.numpy(), rtol=1e-10, atol=1e-12
            )
