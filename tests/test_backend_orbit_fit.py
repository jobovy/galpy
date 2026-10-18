###############################################################################
# test_backend_orbit_fit.py: Orbit.from_fit on the backend. A backend input (the
# data, their errors, the initial guess, a potential parameter) or a forced
# backend fits with the backend likelihood and its gradient (scipy's L-BFGS on
# their values) and returns a backend Orbit; one Newton step from the optimum
# carries the implicit derivative of the fit with respect to the data and the
# potential. The likelihood is flat along the orbit, so the derivative is that of
# quantities that are constant along it (E, Lz), checked against re-fits.
###############################################################################
import numpy
import pytest

from galpy import potential
from galpy.backend import as_numpy, is_backend_array, use
from galpy.orbit import Orbit
from galpy.util import coords

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

_FIT = dict(tintJ=1.5, ntintJ=300)


def _pot(q=0.9):
    return potential.LogarithmicHaloPotential(normalize=1.0, q=q)


def _truth():
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0])
    o.integrate(numpy.linspace(0.0, 1.0, 1001), _pot())
    return o


_O = _truth()
_DATA = _O.getOrbit()[::100, :]
_INIT = _O.vxvv[0] * numpy.array([1.02, 1.05, 0.98, 1.03, 0.95, 1.01])


def _arr(backend, x):
    return jnp.asarray(x) if backend == "jax" else torch.tensor(x)


def _mlogl(of, data, pot, **kw):
    # the numpy likelihood at a fitted orbit's initial condition
    from galpy.actionAngle import actionAngle, actionAngleIsochroneApprox
    from galpy.orbit.Orbits import _fit_orbit_mlogl

    class _aA(actionAngleIsochroneApprox):
        def __init__(self):
            actionAngle.__init__(self)
            self._tintJ, self._ntintJ = _FIT["tintJ"], _FIT["ntintJ"]
            self._tsJ = numpy.linspace(0.0, self._tintJ, self._ntintJ)
            self._integrate_dt, self._pot = None, pot
            self._integrate_method = "dopr54_c"

    p = numpy.asarray(
        as_numpy(of._ic_backend if of._ic_backend is not None else of.vxvv[0]),
        dtype=float,
    ).reshape(6)
    return _fit_orbit_mlogl(
        p,
        data,
        None,
        pot,
        False,
        False,
        False,
        None,
        None,
        _aA(),
        None,
        None,
        None,
        **kw,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("how", ["forced", "data", "potential"])
def test_fit_on_backend_returns_backend_orbit(backend, how):
    # every way in: a forced backend, backend data, a backend potential parameter
    pot = _pot()
    data = _DATA
    if how == "forced":
        with use(backend, force=True):
            of = Orbit.from_fit(_INIT, data, pot=pot, **_FIT)
    elif how == "data":
        of = Orbit.from_fit(_INIT, _arr(backend, data), pot=pot, **_FIT)
    else:
        pot = potential.LogarithmicHaloPotential(normalize=1.0, q=_arr(backend, 0.9))
        of = Orbit.from_fit(_INIT, data, pot=pot, **_FIT)
    assert is_backend_array(of._ic_backend)
    assert is_backend_array(of.R())
    # at least as good a fit as numpy's Powell from the same start
    ofn = Orbit.from_fit(_INIT, data, pot=_pot(), **_FIT)
    assert _mlogl(of, data, _pot()) <= _mlogl(ofn, data, _pot()) + 1e-8


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sky", ["lb", "radec", "customsky"])
def test_fit_on_backend_sky_coordinates(backend, sky):
    # data in (l, b, ...), (ra, dec, ...), or a custom sky frame: the likelihood's
    # coordinate chain on the backend; a fit at least as good as numpy's
    ro, vo = 8.0, 220.0
    o = Orbit([0.8, 0.3, 1.3, 0.4, 0.2, 2.0], ro=ro, vo=vo)
    o.integrate(numpy.linspace(0.0, 1.0, 1001), _pot())
    ts = numpy.linspace(0.0, 1.0, 11)
    if sky == "lb":
        names = ["ll", "bb", "dist", "pmll", "pmbb", "vlos"]
        kw = dict(lb=True)
    else:
        names = ["ra", "dec", "dist", "pmra", "pmdec", "vlos"]
        kw = dict(radec=True)
    data = numpy.array([getattr(o, n)(ts) for n in names]).T
    if sky == "customsky":
        # (ra, dec) given as a custom frame
        kw = dict(
            customsky=True,
            lb_to_customsky=lambda l, b, degree=False: coords.lb_to_radec(
                l, b, degree=degree
            ),
            pmllpmbb_to_customsky=lambda pmll, pmbb, l, b, degree=False: (
                coords.pmllpmbb_to_pmrapmdec(pmll, pmbb, l, b, degree=degree)
            ),
        )
    init = data[0] * numpy.array([1.0005, 1.0005, 1.01, 1.01, 1.01, 1.01])
    err = numpy.array([1e-4, 1e-4, 1e-2, 1e-2, 1e-2, 1.0]) * numpy.ones_like(data)
    ofn = Orbit.from_fit(
        init, data, vxvv_err=err, pot=_pot(), ro=ro, vo=vo, **kw, **_FIT
    )
    with use(backend, force=True):
        of = Orbit.from_fit(
            init, data, vxvv_err=err, pot=_pot(), ro=ro, vo=vo, **kw, **_FIT
        )
    assert is_backend_array(of._ic_backend)
    # as good a fit of the data as numpy's: the chi^2 of each datum against the
    # nearest point of the fitted orbit's track, in the data's own coordinates
    tt = numpy.linspace(-1.5, 1.5, 6001)

    def chi2(f):
        g = Orbit(
            numpy.asarray(
                as_numpy(f._ic_backend if f._ic_backend is not None else f.vxvv[0]),
                dtype=float,
            ).reshape(6),
            ro=ro,
            vo=vo,
        )
        g.integrate(tt, _pot(), method="dop853_c")
        if sky == "lb":
            tr = numpy.array(
                [g.ll(tt), g.bb(tt), g.dist(tt), g.pmll(tt), g.pmbb(tt), g.vlos(tt)]
            ).T
        else:
            tr = numpy.array(
                [g.ra(tt), g.dec(tt), g.dist(tt), g.pmra(tt), g.pmdec(tt), g.vlos(tt)]
            ).T
        return numpy.sum(
            numpy.min(
                numpy.sum(((tr[None] - data[:, None]) / err[:, None]) ** 2, axis=2),
                axis=1,
            )
        )

    assert chi2(of) <= 1.01 * chi2(ofn) + 1e-6


def _fit_EL(data=_DATA, q=0.9, backend="torch"):
    with use(backend, force=True):
        of = Orbit.from_fit(_INIT, data, pot=_pot(q), **_FIT)
        return numpy.array(
            [
                float(as_numpy(of.E(pot=_pot(q), use_physical=False))),
                float(as_numpy(of.Lz(use_physical=False))),
            ]
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_derivative_wrt_data_vs_refits(backend):
    # d(E, Lz of the fit) / d(one data value), against central differences of
    # re-fits (E and Lz are constant along the likelihood's flat direction)
    k, j, h = 3, 1, 1e-4
    vp, vm = _DATA.copy(), _DATA.copy()
    vp[k, j] += h
    vm[k, j] -= h
    fd = (_fit_EL(vp) - _fit_EL(vm)) / (2.0 * h)
    if backend == "jax":

        def f(v):
            of = Orbit.from_fit(_INIT, v, pot=_pot(), **_FIT)
            return jnp.stack(
                [
                    of.E(pot=_pot(), use_physical=False).reshape(()),
                    of.Lz(use_physical=False).reshape(()),
                ]
            )

        g = jax.jacrev(f)(jnp.asarray(_DATA))[:, k, j]
    else:
        v = torch.tensor(_DATA, requires_grad=True)
        of = Orbit.from_fit(_INIT, v, pot=_pot(), **_FIT)
        g = torch.stack(
            [
                torch.autograd.grad(
                    of.E(pot=_pot(), use_physical=False).reshape(()),
                    v,
                    retain_graph=True,
                )[0][k, j],
                torch.autograd.grad(of.Lz(use_physical=False).reshape(()), v)[0][k, j],
            ]
        )
    numpy.testing.assert_allclose(as_numpy(g), fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_derivative_wrt_initial_guess_vanishes(backend):
    # the optimum does not depend on where the optimizer starts
    if backend == "jax":
        g = jax.grad(
            lambda x: (
                Orbit.from_fit(x, _DATA, pot=_pot(), **_FIT)
                .Lz(use_physical=False)
                .reshape(())
            )
        )(jnp.asarray(_INIT))
    else:
        x = torch.tensor(_INIT, requires_grad=True)
        of = Orbit.from_fit(x, _DATA, pot=_pot(), **_FIT)
        (g,) = torch.autograd.grad(
            of.Lz(use_physical=False).reshape(()), x, allow_unused=True
        )
        g = torch.zeros(6) if g is None else g
    assert numpy.all(as_numpy(g) == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_derivative_wrt_potential_parameter(backend):
    # d Lz(fit) / d q: the optimizer runs on a host copy of the potential, the
    # Newton step differentiates the gradient once more through the in-backend
    # solve (jax: diffrax's direct adjoint); against central re-fits
    fit = dict(tintJ=1.5, ntintJ=300)
    data = _DATA[::2]

    def Lz(q, bk):
        with use(bk, force=True):
            of = Orbit.from_fit(_INIT, data, pot=_pot(q), **fit)
            return float(as_numpy(of.Lz(use_physical=False)))

    h = 1e-4
    fd = (Lz(0.9 + h, backend) - Lz(0.9 - h, backend)) / (2.0 * h)
    if backend == "jax":
        g = jax.grad(
            lambda q: (
                Orbit.from_fit(_INIT, data, pot=_pot(q), **fit)
                .Lz(use_physical=False)
                .reshape(())
            )
        )(0.9)
    else:
        q = torch.tensor(0.9, requires_grad=True)
        of = Orbit.from_fit(_INIT, data, pot=_pot(q), **fit)
        (g,) = torch.autograd.grad(of.Lz(use_physical=False).reshape(()), q)
    numpy.testing.assert_allclose(float(g), fd, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fit_disp_prints_the_optimizer_message(backend, capsys):
    with use(backend, force=True):
        Orbit.from_fit(_INIT, _DATA, pot=_pot(), disp=True, **_FIT)
    assert "CONVERGENCE" in capsys.readouterr().out


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_copy_of_a_backend_potential(backend):
    # host_copy: every backend-array attribute of the (nested) object becomes
    # its host value; the original keeps its backend parameters
    from galpy.backend import host_copy
    from galpy.potential import DehnenSmoothWrapperPotential

    b = _arr(backend, 0.6)
    pots = [
        potential.PlummerPotential(amp=1.0, b=b),
        DehnenSmoothWrapperPotential(
            pot=potential.MiyamotoNagaiPotential(a=b, b=0.3), tform=-1.0, tsteady=1.0
        ),
    ]
    ref = [float(as_numpy(p(1.0, 0.1))) for p in pots]  # also fills parameter caches
    hpots = host_copy(pots)
    assert isinstance(hpots, list) and len(hpots) == 2
    # anything but a galpy object passes through
    assert host_copy((1.5, "x")) == (1.5, "x")
    for hp, p, r in zip(hpots, pots, ref):
        assert hp is not p
        assert not any(is_backend_array(v) for v in vars(hp).values())
        assert any(is_backend_array(v) for v in vars(p).values()) or any(
            is_backend_array(v) for v in vars(p._pot).values()
        )
        got = hp(1.0, 0.1)
        assert not is_backend_array(got)
        numpy.testing.assert_allclose(got, r, rtol=1e-14)
    assert not any(is_backend_array(v) for v in vars(hpots[1]._pot).values())
