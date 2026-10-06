###############################################################################
# test_backend_evolveddiskdf.py: Track F Pdf.3 -- backend (jax/torch) coverage
# for evolveddiskdf's velocity-moment grid, DIFFERENTIABLE THROUGH THE ORBIT
# EVOLUTION.
#
# evolveddiskdf builds f(x,t) = initdf(orbit integrated back to t0) on a velocity
# grid and reduces that grid to moments <vR^n vT^m Sigma>. Under a backend (or a
# forced backend), _buildvgrid_backend builds the whole grid in ONE vectorized
# multi-orbit integrate through the in-backend ODE solver, so grid.df is a backend
# array and the moment is differentiable w.r.t. R (and potential parameters) THROUGH
# the orbit evolution. The numpy path (_buildvgrid, per-gridpoint C loop) is
# byte-identical (test_evolveddiskdf unchanged).
#
# Checks:
#   (a) moment value parity numpy<->backend for every moment method, on scalar-t
#       and time-list grids, both integrated with the same accurate integrator
#       (dop853_c / diffrax at rtol=atol=1e-12) so they agree tightly;
#   (b) grad-vs-FD of a moment w.r.t. R -- differentiability THROUGH the evolution;
#   (c) deriv= (the finite-difference moment derivative) is rejected on the backend;
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
# dop853_c on numpy and diffrax/torchdiffeq at rtol=atol=1e-12 on the backend are
# both ~1e-12-accurate, so the moments agree to ~1e-8 (the integrators are
# different methods; a looser integrator would show its own truncation, not a bug).
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
    # integrate) vs numpy; both at high accuracy so they agree to integrator tol.
    edf = _make_edf()
    ref = numpy.asarray(_MOMENTS[moment](edf, _R))
    with use(backend, force=True):
        got = _MOMENTS[moment](edf, _scalar(backend, _R))
    assert is_backend_array(got)
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7, atol=1e-9)


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
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7, atol=1e-9)


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
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-6, atol=1e-9)


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
    numpy.testing.assert_allclose(as_numpy(got), ref, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grid_deriv_rejected_on_backend(backend):
    # the finite-difference moment derivative (deriv=) is not supported on the
    # backend -- autodiff replaces it; the grid build must raise a clear error.
    edf = _make_edf()
    with use(backend, force=True):
        with pytest.raises(NotImplementedError):
            edf.vmomentsurfacemass(
                _scalar(backend, _R),
                0,
                0,
                phi=_PHI,
                grid=True,
                gridpoints=_GP,
                integrate_method=_IM,
                deriv="R",
            )


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
        # vacuously, which a bare allclose cannot see.
        assert_jit_matches_eager(call, jnp.asarray(_R), rtol=1e-12, atol=1e-14)


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
def test_hierarchical_grid_time_list_says_it_is_unsupported(backend):
    # Only the scalar-t hierarchical build is vectorised. A time list must say
    # so rather than fall into the numpy loop and fail obscurely on a backend
    # velocity grid.
    with use(backend, force=True):
        with pytest.raises(NotImplementedError, match="list of times"):
            _hierarch(_GRAD_A, _scalar(backend, _R), t=[0.0, -0.5])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("route", sorted(_GRAD_ROUTES))
def test_merely_forced_backend_with_a_plain_potential_stays_numpy(backend, route):
    # The NEGATIVE CONTROL for the widened gate, and the whole reason the guard
    # exists: under a merely FORCED backend a plain numpy potential carries no
    # gradient, so the fast numpy path must still be taken and the public
    # return type must still be a float. If this ever flips, the numpy suite is
    # silently running on diffrax/torchdiffeq.
    ref = _grad_edf(_GRAD_A).vmomentsurfacemass(
        _R, 0, 0, phi=_PHI, nsigma=3.0, **_GRAD_ROUTES[route]
    )
    with use(backend, force=True):
        got = _grad_moment(_GRAD_A, route)  # plain float a -> no gradient anywhere
    assert not is_backend_array(got), "a merely-forced backend flipped the numpy path"
    numpy.testing.assert_allclose(float(got), float(ref), rtol=0.0, atol=0.0)
