###############################################################################
# test_backend_evolveddiskdf.py: Track F Pdf.3 -- backend (jax/torch) coverage
# for evolveddiskdf's velocity-moment grid, DIFFERENTIABLE THROUGH THE ORBIT
# EVOLUTION.
#
# evolveddiskdf builds f(x,t) = initdf(orbit integrated back to t0) on a velocity
# grid and reduces that grid to moments <vR^n vT^m Sigma>. Under a backend (or a
# forced backend), _buildvgrid_backend builds the whole grid in ONE vectorized
# multi-orbit integrate -- in plain C for concrete input, through the C
# state-transition matrix or the in-backend ODE solver for a differentiated one --
# so grid.df is a backend array and the moment is differentiable w.r.t. R (and
# potential parameters) THROUGH the orbit evolution. The numpy path (_buildvgrid,
# per-gridpoint C loop) is byte-identical (test_evolveddiskdf unchanged).
#
# Checks:
#   (a) moment value parity numpy<->backend for every moment method, on scalar-t
#       and time-list grids, both integrated with the same accurate integrator
#       (dop853_c / diffrax at rtol=atol=1e-12) so they agree tightly;
#   (b) grad-vs-FD of a moment w.r.t. R -- differentiability THROUGH the evolution;
#   (c) deriv= grids match numpy's (rejected only for a differentiated moment);
#   (d) the grid reduction (_vmomentsurfacemassGrid) is itself differentiable w.r.t.
#       the grid df values (closed-form check).
###############################################################################
import copy

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

from backend_jit_helpers import assert_jit_matches_eager

from galpy.backend import as_numpy, is_backend_array, use
from galpy.df import dehnendf, evolveddiskdf
from galpy.potential import (
    EllipticalDiskPotential,
    LogarithmicHaloPotential,
    SteadyLogSpiralPotential,
)

_GP = 9  # small grid for speed
_R, _PHI = 0.9, 0.2
# dop853_c on both: a concrete backend orbit integrates with the same C
# integrator as numpy, so the grid moments agree to rounding.
_IM = "dop853_c"


def _make_edf():
    idf = dehnendf(beta=0.0)
    # mild non-axisymmetric, C-capable planar composite potential
    pot = LogarithmicHaloPotential(normalize=1.0) + SteadyLogSpiralPotential(
        A=-0.005, omegas=0.2
    )
    return evolveddiskdf(idf, pot=pot, to=-10.0)


def _scalar(backend, x):
    return jnp.asarray(float(x)) if backend == "jax" else torch.tensor(float(x))


# Moment methods, keyed by name -> callable(edf, R) at fixed phi, grid-built.
_MOMENTS = {
    "surfacemass": lambda edf, R: edf.vmomentsurfacemass(
        R, 0, 0, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM
    ),
    "sigmaR2": lambda edf, R: edf.sigmaR2(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
    "sigmaT2": lambda edf, R: edf.sigmaT2(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
    "sigmaRT": lambda edf, R: edf.sigmaRT(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
    "meanvR": lambda edf, R: edf.meanvR(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
    "meanvT": lambda edf, R: edf.meanvT(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
    "vertexdev": lambda edf, R: edf.vertexdev(
        R, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, use_physical=False
    ),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("moment", list(_MOMENTS))
def test_moment_parity_native_backend_grid(backend, moment):
    # moment on a NATIVELY backend-built scalar-t grid (one vectorized multi-orbit
    # integrate) vs numpy; the same C integrator, so they agree to rounding.
    edf = _make_edf()
    ref = numpy.asarray(_MOMENTS[moment](edf, _R))
    with use(backend, force=True):
        got = _MOMENTS[moment](edf, _scalar(backend, _R))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_parity_tlist_native_backend_grid(backend):
    # time-list grid -> 3D df; one moment per time, natively backend-built.
    edf = _make_edf()
    ts = numpy.array([0.0, -2.5, -5.0])
    ref = numpy.asarray(
        edf.vmomentsurfacemass(
            _R, 1, 0, t=ts, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM
        )
    )
    with use(backend, force=True):
        got = edf.vmomentsurfacemass(
            _scalar(backend, _R),
            1,
            0,
            t=ts,
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
        )
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("moment", ["surfacemass", "sigmaR2", "meanvT", "vertexdev"])
def test_moment_grad_through_evolution_vs_fd(backend, moment):
    # d(moment)/dR via autodiff THROUGH the orbit evolution vs central FD. FD is on
    # the backend itself (integrator-consistent), so the floor is autodiff/FD, ~2e-4.
    edf = _make_edf()
    fn = _MOMENTS[moment]
    R0, h = _R, 1e-5

    def val(R):
        with use(backend, force=True):
            return as_numpy(
                jnp.asarray(fn(edf, R)).reshape(())
                if backend == "jax"
                else fn(edf, R).reshape(())
            )

    gfd = (
        float(val(_scalar(backend, R0 + h))) - float(val(_scalar(backend, R0 - h)))
    ) / (2.0 * h)
    if backend == "jax":
        with use("jax", force=True):
            g = float(
                jax.grad(lambda R: jnp.asarray(fn(edf, R)).reshape(()))(jnp.asarray(R0))
            )
    else:
        Rt = torch.tensor(R0, requires_grad=True)
        with use("torch", force=True):
            fn(edf, Rt).reshape(()).backward()
        g = float(Rt.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=2e-4, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonaxi_vertexdev_native_backend(backend):
    # a genuinely non-axisymmetric potential gives sigmaRT != 0, so vertexdev's
    # arctan combiner is non-trivial; native backend build vs numpy.
    idf = dehnendf(beta=0.0)
    pot = LogarithmicHaloPotential(normalize=1.0) + EllipticalDiskPotential(
        twophio=0.05, phib=25.0 / 180.0 * numpy.pi, p=0.0, tform=-150.0, tsteady=125.0
    )
    edf = evolveddiskdf(idf, pot=pot, to=-10.0)
    ref = float(
        edf.vertexdev(
            _R,
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
            use_physical=False,
        )
    )
    with use(backend, force=True):
        got = edf.vertexdev(
            _scalar(backend, _R),
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
            use_physical=False,
        )
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "tkw",
    [{"t": -10.0}, {"t": [-10.0]}, {"t": [0.0]}],
    ids=["scalar-no-evolution", "tlist-len1-no-evolution", "tlist-len1-evolved"],
)
def test_no_evolution_and_len1_branches(backend, tkw):
    # cover the self._to == t (no integration) branches and the tlist len-1
    # (single evolved endpoint) branch of _buildvgrid_backend; to == -10.0.
    edf = _make_edf()  # to = -10.0
    ref = numpy.asarray(
        edf.vmomentsurfacemass(
            _R, 0, 0, phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, **tkw
        )
    )
    with use(backend, force=True):
        got = edf.vmomentsurfacemass(
            _scalar(backend, _R),
            0,
            0,
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
            **tkw,
        )
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("deriv", ["R", "phi"])
@pytest.mark.parametrize(
    "gkw",
    [
        {"t": 0.0},
        {"t": [0.0, -2.5, -5.0]},
        {"t": 0.0, "hierarchgrid": True, "nlevels": 2},
    ],
    ids=["scalar-t", "tlist", "hierarchical"],
)
def test_grid_deriv_matches_numpy(backend, deriv, gkw):
    # the deriv= grids (the Oort functions' input) on the backend: numpy's
    # variational equations, batched over the gridpoints. The hierarchical grid
    # used to drop deriv= on the backend and return the moment itself.
    edf = _make_edf()
    kw = dict(phi=_PHI, grid=True, gridpoints=_GP, integrate_method=_IM, deriv=deriv)
    ref = numpy.asarray(edf.vmomentsurfacemass(_R, 0, 0, **kw, **gkw))
    with use(backend, force=True):
        got = edf.vmomentsurfacemass(_scalar(backend, _R), 0, 0, **kw, **gkw)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_deriv_of_a_differentiated_moment_rejected(backend):
    # deriv= of a moment that is itself being differentiated is not supported
    # (autodiff replaces it): the grid build must raise a clear error.
    edf = _make_edf()

    def moment(R):
        return edf.vmomentsurfacemass(
            R,
            0,
            0,
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
            deriv="R",
        )

    with use(backend, force=True):
        with pytest.raises(NotImplementedError):
            if backend == "jax":
                jax.grad(lambda R: jnp.asarray(moment(R)).reshape(()))(jnp.asarray(_R))
            else:
                moment(torch.tensor(_R, requires_grad=True))


@pytest.mark.parametrize("backend", BACKENDS)
def test_concrete_orbits_integrate_in_c(backend, monkeypatch):
    # a forced backend with concrete input integrates every gridpoint's orbit
    # with the requested C integrator (plain C), not with the in-backend ODE or
    # the C state-transition matrix, which only a gradient needs
    from galpy.orbit import Orbits

    def refuse(*args, **kwargs):
        raise AssertionError("a concrete orbit took a differentiable integrator")

    monkeypatch.setattr(Orbits.Orbit, "_integrate_inbackend", refuse)
    monkeypatch.setattr(Orbits.Orbit, "_integrate_cstm", refuse)
    edf = _make_edf()
    ref = edf.meanvT(
        _R,
        phi=_PHI,
        grid=True,
        gridpoints=_GP,
        integrate_method=_IM,
        use_physical=False,
    )
    with use(backend, force=True):
        got = edf.meanvT(
            _R,
            phi=_PHI,
            grid=True,
            gridpoints=_GP,
            integrate_method=_IM,
            use_physical=False,
        )
    assert is_backend_array(got)
    numpy.testing.assert_allclose(float(as_numpy(got)), float(ref), rtol=1e-13)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduction_grad_vs_df(backend):
    # the grid reduction _vmomentsurfacemassGrid is differentiable w.r.t. the grid
    # df values (closed form for <vR^1 vT^1 Sigma>: grad = vR_i vT_j dvR dvT).
    edf = _make_edf()
    _, g_np = edf.vmomentsurfacemass(
        _R,
        0,
        0,
        phi=_PHI,
        integrate_method="rk6_c",
        grid=True,
        gridpoints=_GP,
        returnGrid=True,
    )
    df0 = numpy.asarray(g_np.df)
    vR = numpy.asarray(g_np.vRgrid)
    vT = numpy.asarray(g_np.vTgrid)

    def moment_of_df(df_arr):
        g = copy.copy(g_np)
        g.df = df_arr
        g.vRgrid = jnp.asarray(vR) if backend == "jax" else torch.as_tensor(vR)
        g.vTgrid = jnp.asarray(vT) if backend == "jax" else torch.as_tensor(vT)
        with use(backend, force=True):
            return edf._vmomentsurfacemassGrid(1, 1, g)

    idx = (_GP // 3, _GP // 3)  # off-centre so vR_i*vT_j != 0
    expect = vR[idx[0]] * vT[idx[1]] * (vR[1] - vR[0]) * (vT[1] - vT[0])
    if backend == "jax":
        grad = jax.grad(lambda d: moment_of_df(d))(jnp.asarray(df0))
        g_elem = float(as_numpy(grad)[idx])
    else:
        dt = torch.tensor(df0, requires_grad=True)
        moment_of_df(dt).backward()
        g_elem = float(dt.grad[idx])
    numpy.testing.assert_allclose(g_elem, expect, rtol=1e-8)


# --- direct (grid-free) moment on a backend -------------------------------
# Everything above exercises grid=True. The direct path had NO backend coverage
# at all, which is why its non-differentiability went unnoticed: scipy's dblquad
# computes the VALUE correctly under a backend (it just evaluates eagerly with
# concrete floats), so any value-only check passes while jax.grad raises
# ConcretizationTypeError because scipy calls float() on the tracer.


@pytest.mark.parametrize("backend", BACKENDS)
def test_direct_moment_value_matches_numpy(backend):
    # Polar-GL direct path vs numpy's adaptive dblquad. These are different
    # quadrature rules, so this is a correctness check at quadrature tolerance,
    # not a parity check.
    edf = _make_edf()
    ref = edf.vmomentsurfacemass(_R, 0, 0, phi=_PHI, grid=False, nsigma=3.0)
    with use(backend, force=True):
        got = edf.vmomentsurfacemass(
            _scalar(backend, _R), 0, 0, phi=_PHI, grid=False, nsigma=3.0
        )
        assert is_backend_array(got), "direct path fell back to numpy"
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_direct_moment_grad_through_evolution_vs_fd(backend):
    # The point of the whole exercise: d(moment)/dR through the ORBIT EVOLUTION
    # on the direct path. Before the backend branch this raised
    # ConcretizationTypeError. FD is taken on the backend itself so the
    # comparison is integrator- and quadrature-consistent (see
    # test_moment_grad_through_evolution_vs_fd).
    edf = _make_edf()
    h = 1e-5

    def val(R):
        with use(backend, force=True):
            return float(
                as_numpy(
                    edf.vmomentsurfacemass(R, 0, 0, phi=_PHI, grid=False, nsigma=3.0)
                )
            )

    gfd = (val(_scalar(backend, _R + h)) - val(_scalar(backend, _R - h))) / (2.0 * h)
    if backend == "jax":
        with use("jax", force=True):
            g = float(
                jax.grad(
                    lambda R: jnp.asarray(
                        edf.vmomentsurfacemass(
                            R, 0, 0, phi=_PHI, grid=False, nsigma=3.0
                        )
                    ).reshape(())
                )(jnp.asarray(_R))
            )
    else:
        Rt = torch.tensor(_R, requires_grad=True)
        with use("torch", force=True):
            edf.vmomentsurfacemass(Rt, 0, 0, phi=_PHI, grid=False, nsigma=3.0).reshape(
                ()
            ).backward()
        g = float(Rt.grad)
    numpy.testing.assert_allclose(g, gfd, rtol=2e-4, atol=1e-7)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax required for jit")
def test_direct_moment_jit_matches_eager():
    # The direct path is not merely differentiable but TRACEABLE: the
    # `if initvmoment == 0.0` guard is selected with xp.where on the backend
    # (numpy keeps the branch, so its float return type is unchanged). Without
    # that this raises TracerBoolConversionError.
    edf = _make_edf()

    def call(R):
        return jnp.asarray(
            edf.vmomentsurfacemass(R, 0, 0, phi=_PHI, grid=False, nsigma=3.0)
        ).reshape(())

    with use("jax", force=True):
        # assert_jit_matches_eager also checks the jaxpr CONSUMES its argument: a
        # trace that folded it away returns a constant and would match eager
        # vacuously, which a bare allclose cannot see. The traced orbits integrate
        # through the C state-transition matrix, the eager ones with the plain C
        # integrator: the two agree to the integrator's tolerance (2e-11 here),
        # not to rounding.
        assert_jit_matches_eager(call, jnp.asarray(_R), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n,m", [(1, 0), (0, 1), (0, 0)])
def test_direct_moment_with_a_zero_initdf_moment(backend, n, m):
    # The vanishing-normalisation guard: when the INITIAL df's (n,m) moment is
    # exactly 0 the code substitutes 1.0 so the ratio stays finite. For the
    # ODD radial moment (n=1,m=0) dehnendf's <vR> is 0 BY SYMMETRY, so that
    # branch is the live one -- and every other direct-path test here uses
    # (0,0), whose initdf moment is never zero, so the branch was never
    # executed on a backend. On torch it raised
    #   TypeError: where() received an invalid combination of arguments
    # because initvmoment is a numpy scalar, making the condition a numpy bool.
    edf = _make_edf()
    initvmoment = edf._initdf.vmomentsurfacemass(_R, n, m, nsigma=3.0, phi=_PHI)
    assert (float(initvmoment) == 0.0) == (n == 1), "fixture no longer pins the branch"
    ref = edf.vmomentsurfacemass(_R, n, m, phi=_PHI, grid=False, nsigma=3.0)
    with use(backend, force=True):
        got = edf.vmomentsurfacemass(
            _scalar(backend, _R), n, m, phi=_PHI, grid=False, nsigma=3.0
        )
        assert is_backend_array(got), "direct path fell back to numpy"
    # Tolerance from measurement, not guesswork: the adaptive-dblquad and
    # polar-GL rules agree to |diff| = 3.1e-06 (0,0), 7.2e-08 (1,0), 2.0e-06
    # (0,1). The ODD moment is near-cancelling (~3.2e-05 against a 6.6e-02
    # surface density), so its RELATIVE error is 2.2e-03 while its ABSOLUTE
    # error is the smallest of the three -- the backend rule is not the looser
    # one. Hence rtol for the even moments and a small atol for the odd one.
    numpy.testing.assert_allclose(
        float(as_numpy(got)), float(ref), rtol=1e-4, atol=1e-6
    )


# --- the gradient in the POTENTIAL, with R a plain float -------------------
# The moment gate used to ask only `is_backend_array(R)`. A gradient carried by
# a potential PARAMETER is invisible to that question, so with a plain-float R
# the whole moment was computed in numpy and came back a bare float64 -- the
# gradient silently lost, with no error. These pin the three routes.
_GRAD_A = 0.6


def _grad_edf(a):
    """evolveddiskdf whose POTENTIAL carries the differentiated parameter."""
    from galpy.potential import MiyamotoNagaiPotential

    return evolveddiskdf(
        dehnendf(beta=0.0),
        pot=MiyamotoNagaiPotential(normalize=1.0, a=a, b=0.3),
        to=-1.0,
    )


_GRAD_ROUTES = {
    # R stays a PLAIN FLOAT in every one of these -- that is the point.
    "direct": dict(grid=False),
    "grid": dict(grid=True, gridpoints=8, integrate_method="rk6_c", returnGrid=False),
}


def _grad_moment(a, route):
    return _grad_edf(a).vmomentsurfacemass(
        _R, 0, 0, phi=_PHI, nsigma=3.0, **_GRAD_ROUTES[route]
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("route", sorted(_GRAD_ROUTES))
def test_moment_grad_wrt_potential_parameter_with_float_R(backend, route):
    # AD vs FD, each route against ITSELF: the direct (polar-GL) and grid
    # routes are different quadratures and legitimately give different
    # derivatives (0.0712 vs 0.1087 here), so comparing one against the other's
    # FD would report a ~50% "error" that is not a bug.
    with use(backend, force=True):
        if backend == "jax":
            g = float(
                jax.grad(lambda t: jnp.asarray(_grad_moment(t, route)).reshape(()))(
                    jnp.asarray(_GRAD_A)
                )
            )
        else:
            t = torch.tensor(_GRAD_A, requires_grad=True)
            out = _grad_moment(t, route)
            assert is_backend_array(out), "moment fell back to numpy"
            out.reshape(()).backward()
            g = float(t.grad)
    h = 1e-5
    fd = (
        float(_grad_moment(_GRAD_A + h, route))
        - float(_grad_moment(_GRAD_A - h, route))
    ) / (2.0 * h)
    assert abs(g) > 1e-6, "zero gradient: the potential parameter is disconnected"
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-8)


def _hierarch(pot_a, R, **kw):
    return _grad_edf(pot_a).vmomentsurfacemass(
        R,
        0,
        0,
        phi=_PHI,
        nsigma=3.0,
        grid=True,
        gridpoints=8,
        integrate_method="rk6_c",
        hierarchgrid=True,
        returnGrid=False,
        **kw,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nlevels", [2, 0])
def test_hierarchical_grid_matches_numpy_on_the_backend(backend, nlevels):
    # The hierarchical levels now build with one vectorised multi-orbit
    # integrate each (plus a static hole mask where the finer level takes
    # over), so the whole grid is a backend array. The backend ODE solver is
    # not dop853_c, so this agrees at quadrature level, not bit-for-bit.
    # nlevels=0 is the no-subgrid case (the recursion stops at nlevels > 1, so
    # only an explicit 0 reaches it); the numpy suite covers it the same way.
    ref = float(_hierarch(_GRAD_A, _R, nlevels=nlevels))
    with use(backend, force=True):
        got = _hierarch(_GRAD_A, _scalar(backend, _R), nlevels=nlevels)
        assert is_backend_array(got), "hierarchical grid fell back to numpy"
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hierarchical_grid_grad_wrt_potential_parameter(backend):
    # The point of migrating it: hierarchgrid=True used to WARN that it could
    # carry no gradient. FD is taken on the same route so the comparison is
    # quadrature-consistent.
    with use(backend, force=True):
        if backend == "jax":
            g = float(
                jax.grad(lambda t: jnp.asarray(_hierarch(t, _R)).reshape(()))(
                    jnp.asarray(_GRAD_A)
                )
            )
        else:
            t = torch.tensor(_GRAD_A, requires_grad=True)
            out = _hierarch(t, _R)
            assert is_backend_array(out), "hierarchical grid fell back to numpy"
            out.reshape(()).backward()
            g = float(t.grad)
    h = 1e-5
    fd = (float(_hierarch(_GRAD_A + h, _R)) - float(_hierarch(_GRAD_A - h, _R))) / (
        2.0 * h
    )
    assert abs(g) > 1e-6, "zero gradient: the potential parameter is disconnected"
    numpy.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nlevels", [2, 0])
def test_hierarchical_grid_time_list_matches_numpy_on_the_backend(backend, nlevels):
    # a list of times: each level's one integrate returns (N, nt) and the hole
    # mask broadcasts over the times
    ts = [0.0, -0.5, -1.0]
    ref = _hierarch(_GRAD_A, _R, nlevels=nlevels, t=ts)
    with use(backend, force=True):
        got = _hierarch(_GRAD_A, _scalar(backend, _R), nlevels=nlevels, t=ts)
        assert is_backend_array(got), "hierarchical grid fell back to numpy"
    assert got.shape == (3,)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6)


# The direct route is a different quadrature on the backend (polar GL) from
# numpy's (dblquad at its default epsrel=1e-2); the grid route is the same grid,
# integrated with the same C integrator.
_ROUTE_RTOL = {"direct": 1e-4, "grid": 1e-12}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("route", sorted(_GRAD_ROUTES))
def test_merely_forced_backend_runs_on_the_backend(backend, route):
    # a forced backend with a plain potential and a plain R runs the moment on
    # the backend (its orbits integrate in C), at numpy's value
    ref = _grad_edf(_GRAD_A).vmomentsurfacemass(
        _R, 0, 0, phi=_PHI, nsigma=3.0, **_GRAD_ROUTES[route]
    )
    with use(backend, force=True):
        got = _grad_moment(_GRAD_A, route)  # plain float a -> no gradient anywhere
    assert is_backend_array(got), "a forced backend fell back to numpy"
    numpy.testing.assert_allclose(
        float(as_numpy(got)), float(ref), rtol=_ROUTE_RTOL[route], atol=0.0
    )


# --- the gradient in the INITIAL DF, with R a plain float ------------------
# A differentiated initdf profile parameter (hR) with a plain-float R and a
# plain potential: the moment gate asked only R and the potential, so the
# numpy route met the traced initdf moments (TracerArrayConversionError in
# numpy.linspace under a forced backend).
_HR = 1.0 / 3.0


@pytest.fixture
def hr_moment():
    import contextlib

    from galpy.potential import MiyamotoNagaiPotential

    def q(hr, route, backend=None):
        ctx = (
            use(backend, force=True)
            if backend is not None
            else contextlib.nullcontext()
        )
        with ctx:
            edf = evolveddiskdf(
                dehnendf(beta=0.0, profileParams=(hr, 1.0, 0.2)),
                pot=MiyamotoNagaiPotential(normalize=1.0, a=_GRAD_A, b=0.3),
                to=-1.0,
            )
            return edf.vmomentsurfacemass(
                _R, 0, 0, phi=_PHI, nsigma=3.0, **_GRAD_ROUTES[route]
            )

    return q


@pytest.mark.parametrize("route", sorted(_GRAD_ROUTES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_forced_backend_with_a_concrete_initdf_parameter_runs_on_the_backend(
    backend, route, hr_moment
):
    # a concrete backend hR under a forced backend runs on the backend, at
    # numpy's value
    hr = jnp.asarray(_HR) if backend == "jax" else torch.tensor(_HR)
    got = hr_moment(hr, route, backend)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(
        float(as_numpy(got)),
        float(hr_moment(_HR, route)),
        rtol=_ROUTE_RTOL[route],
        atol=0.0,
    )


# --- __call__ and the marginalizations on a backend ------------------------
# numpy's __call__ algorithm on the backend: the same integrators and output
# times, so the same values; the marginalizations replace scipy's quad by GL.
_CALL_CASES = {
    "scalar": ((0.0,), {}),
    "no-evolution": ((-10.0,), {}),
    "tlist-log": (([0.0, -2.5, -5.0],), {"log": True}),
    "tlist-len1": (([-2.5],), {}),
    "deriv-R": ((0.0,), {"deriv": "R"}),
    "deriv-phi-rk6c": ((0.0,), {"deriv": "phi", "integrate_method": "rk6_c"}),
    "deriv-R-tlist": (([0.0, -2.5, -5.0],), {"deriv": "R"}),
    "deriv-R-no-evolution": ((-10.0,), {"deriv": "R"}),
    "odeint-log": ((0.0,), {"integrate_method": "odeint", "log": True}),
    "odeint-deriv-tlist": (
        ([0.0, -2.5, -5.0],),
        {"integrate_method": "odeint", "deriv": "R"},
    ),
    "leapfrog_c": ((-2.5,), {"integrate_method": "leapfrog_c"}),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(_CALL_CASES))
def test_call_matches_numpy(backend, case):
    from galpy.orbit import Orbit

    edf = _make_edf()  # to = -10.0
    args, kw = _CALL_CASES[case]
    ref = numpy.asarray(edf(Orbit([_R, 0.1, 1.05, _PHI]), *args, **kw))
    with use(backend, force=True):
        got = edf(Orbit([_R, 0.1, 1.05, _PHI]), *args, **kw)
    assert is_backend_array(got), "__call__ fell back to numpy"
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_input_errors(backend):
    # numpy's errors: one orbit at a time, and no time list when marginalizing
    from galpy.orbit import Orbit

    edf = _make_edf()
    with use(backend, force=True):
        with pytest.raises(RuntimeError, match="single-object"):
            edf(Orbit([[_R, 0.1, 1.05, _PHI], [_R, 0.0, 1.0, _PHI]]))
        for marg in ["marginalizeVperp", "marginalizeVlos"]:
            with pytest.raises(RuntimeError, match="list"):
                edf(Orbit([_R, 0.1, 1.05, _PHI]), [0.0, -1.0], **{marg: True})


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("integrate_method", ["dopr54_c", "leapfrog_c"])
def test_call_without_c_matches_numpy(backend, integrate_method):
    # a potential without C: numpy's fallback to odeint / leapfrog, on the host
    from galpy.orbit import Orbit

    edf = _make_edf()
    edf._pot.hasC = False
    edf._pot.hasC_dxdv = False
    kw = dict(integrate_method=integrate_method, log=True)
    ref = numpy.asarray(edf(Orbit([_R, 0.1, 1.05, _PHI]), 0.0, **kw))
    with use(backend, force=True):
        got = edf(Orbit([_R, 0.1, 1.05, _PHI]), 0.0, **kw)
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_grad_with_a_non_c_method(backend):
    # a differentiated moment with a method that has no differentiable C route
    # integrates with the backend's own solver; AD vs FD of that same route
    edf = _make_edf()

    def moment(R):
        return edf.meanvT(
            R,
            phi=_PHI,
            grid=True,
            gridpoints=4,
            integrate_method="leapfrog_c",
            use_physical=False,
        )

    h = 1e-4
    with use(backend, force=True):
        if backend == "jax":
            f = jax.jit(lambda R: jnp.reshape(jnp.asarray(moment(R)), ()))
            g = float(jax.grad(lambda R: jnp.reshape(jnp.asarray(moment(R)), ()))(_R))
            fd = (float(f(_R + h)) - float(f(_R - h))) / (2.0 * h)
        else:
            Rt = torch.tensor(_R, requires_grad=True)
            out = moment(Rt)
            assert is_backend_array(out) and out.requires_grad
            (g,) = torch.autograd.grad(out.reshape(()), Rt)
            g = float(g)
            val = lambda x: float(moment(torch.tensor(x, requires_grad=True)).detach())
            fd = (val(_R + h) - val(_R - h)) / (2.0 * h)
    assert g != 0.0
    # the adaptive in-backend solver's AD and its FD agree to ~1e-5 (as in
    # test_moment_grad_through_evolution_vs_fd); 1.3e-5 here on both backends
    numpy.testing.assert_allclose(g, fd, rtol=5e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_r_negative_and_vanishing_df(backend):
    # a radial orbit through the centre and a far-out orbit with a vanishing
    # DF: numpy's eps / -max conventions
    from galpy.orbit import Orbit

    edf = _make_edf()
    for vxvv, kw in [
        ([1.0, -1.0, 0.0, 0.0], {}),
        ([1.0, -1.0, 0.0, 0.0], {"log": True}),
        ([3.0, 1.0, -1.0, 2.0], {"log": True}),
    ]:
        ref = numpy.asarray(edf(Orbit(vxvv), 0.0, **kw))
        with use(backend, force=True):
            got = edf(Orbit(vxvv), 0.0, **kw)
        assert is_backend_array(got)
        numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("marg", ["marginalizeVperp", "marginalizeVlos"])
@pytest.mark.parametrize("phi", [0.2, 1.5])  # both branches of |sin(alpha)|
def test_call_marginalize_matches_numpy(backend, marg, phi):
    from galpy.orbit import Orbit

    edf = _make_edf()
    o = lambda: Orbit([_R, 0.1, 1.05, phi])
    ref = float(edf(o(), **{marg: True}))
    with use(backend, force=True):
        got = edf(o(), **{marg: True})
        got_log = edf(o(), log=True, **{marg: True})
    assert is_backend_array(got)
    # GL against scipy's adaptive quad, both converged
    numpy.testing.assert_allclose(float(as_numpy(got)), ref, rtol=1e-12)
    numpy.testing.assert_allclose(
        float(as_numpy(got_log)), numpy.log(ref), rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("marg", ["marginalizeVperp", "marginalizeVlos"])
@pytest.mark.parametrize("phi", [0.2, 1.5])  # both branches of |sin(alpha)|
def test_call_marginalize_grad_vs_fd(backend, marg, phi):
    # d/d(R, vR, vT, phi) of a marginalized DF through a differentiated orbit
    # (its observed velocity and local circular speed) vs central FD on the
    # backend; AD runs through the C state-transition matrix, FD through plain
    # C, which agree to the integrator's tolerance
    from galpy.orbit import Orbit

    edf = _make_edf()
    x0 = numpy.array([_R, 0.1, 1.05, phi])

    def f(x):
        return edf(Orbit([x[0], x[1], x[2], x[3]]), **{marg: True})

    with use(backend, force=True):
        if backend == "jax":
            g = numpy.asarray(
                jax.grad(lambda x: jnp.reshape(f(x), ()))(jnp.asarray(x0))
            )
            val = lambda x: float(f(jnp.asarray(x)))
        else:
            xt = torch.tensor(x0, requires_grad=True)
            out = f(xt)
            assert is_backend_array(out) and out.requires_grad
            (g,) = torch.autograd.grad(out.reshape(()), xt)
            g = as_numpy(g)
            val = lambda x: float(f(torch.tensor(x)))
        h = 1e-5
        fd = numpy.array(
            [(val(x0 + h * e) - val(x0 - h * e)) / (2.0 * h) for e in numpy.eye(4)]
        )
    assert numpy.all(g != 0.0)
    # worst 2.1e-7 over both backends and branches
    numpy.testing.assert_allclose(g, fd, rtol=1e-6)


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("route", sorted(_GRAD_ROUTES))
@pytest.mark.parametrize("backend", BACKENDS)
def test_moment_grad_wrt_initdf_parameter_with_float_R(
    backend, route, forced, hr_moment
):
    fbk = backend if forced else None
    if backend == "jax":
        f = lambda t: jnp.reshape(jnp.asarray(hr_moment(t, route, fbk)), ())
        g = float(jax.grad(f)(jnp.asarray(_HR)))
        jf = jax.jit(f)  # a traced hR keeps the backend route for the FD too
        fdf = lambda x: float(jf(jnp.asarray(x)))
    else:
        t = torch.tensor(_HR, requires_grad=True)
        out = hr_moment(t, route, fbk)
        assert is_backend_array(out), "moment fell back to numpy"
        (g,) = torch.autograd.grad(out.reshape(()), t)
        g = float(g)
        fdf = lambda x: float(
            hr_moment(torch.tensor(x, requires_grad=True), route, fbk).detach()
        )
    # Richardson FD of the same backend route (h=1e-3)
    d = lambda h: (fdf(_HR + h) - fdf(_HR - h)) / (2.0 * h)
    d1, d2 = d(1e-3), d(5e-4)
    fd = (4.0 * d2 - d1) / 3.0
    assert abs(d2 - d1) < 1e-3 * abs(fd)
    numpy.testing.assert_allclose(g, fd, rtol=1e-8, atol=0.0)
