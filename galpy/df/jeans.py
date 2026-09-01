# jeans.py: utilities related to the Jeans equations
import time

import numpy
from scipy import integrate

from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    evaluateDensities,
    evaluaterforces,
    evaluateSurfaceDensities,
)
from ..util.conversion import physical_conversion, potential_physical_input

_INVSQRTTWO = 1.0 / numpy.sqrt(2.0)


@potential_physical_input
@physical_conversion("velocity", pop=True)
def sigmar(Pot, r, dens=None, beta=0.0):
    """
    Compute the radial velocity dispersion using the spherical Jeans equation

    Parameters
    ----------
    Pot : potential or a combined potential formed using addition (pot1+pot2+…)
        Gravitational potential; evaluated at R=r/sqrt(2),z=r/sqrt(2), sphericity not checked.
    r : float or Quantity
        Galactocentric radius
    dens : function, optional
        tracer density profile (function of r); if None, the density is assumed to be that corresponding to the potential
    beta : float or function, optional
        anisotropy; can be a constant or a function of r

    Returns
    -------
    float
        sigma_r(r)

    Notes
    -----
    - 2018-07-05 - Written - Bovy (UofT)
    """
    Pot = _check_potential_list_and_deprecate(Pot)
    if dens is None:
        dens = lambda r: evaluateDensities(
            Pot,
            r * _INVSQRTTWO,
            r * _INVSQRTTWO,
            phi=numpy.pi / 4.0,
            use_physical=False,
        )
    if callable(beta):
        intFactor = lambda x: numpy.exp(
            2.0 * integrate.quad(lambda y: beta(y) / y, 1.0, x)[0]
        )
    else:  # assume to be number
        intFactor = lambda x: x ** (2.0 * beta)
    return numpy.sqrt(
        integrate.quad(
            lambda x: (
                -intFactor(x)
                * dens(x)
                * evaluaterforces(
                    Pot,
                    x * _INVSQRTTWO,
                    x * _INVSQRTTWO,
                    phi=numpy.pi / 4.0,
                    use_physical=False,
                )
            ),
            r,
            numpy.inf,
        )[0]
        / dens(r)
        / intFactor(r)
    )


def _eval_on_grid(fn, xs):
    """Evaluate fn on the array xs, falling back to a loop for potentials whose
    methods reject array input (e.g. DoubleExponentialDiskPotential)."""
    try:
        out = numpy.asarray(fn(xs), dtype=float)
        if out.shape == xs.shape:
            return out
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    return numpy.array([float(fn(x)) for x in xs])


def _sigmar_on_grid(Pot, rs, dens=None, beta=0.0, nquad=100001):
    """
    sigma_r at every radius in rs, from ONE cumulative integral.

    Same quantity as calling ``sigmar`` at each radius, but the integrand
    int_r^inf -x^(2 beta) rho(x) F_r(x) dx differs between radii only in its
    LOWER limit, so every radius can be read off a single cumulative pass over
    a shared grid instead of one adaptive quadrature per radius.

    Returns None when the fast path does not apply, so callers fall back to the
    per-radius loop.

    Notes
    -----
    - The grid is geometric: rho ~ 1/r near the centre varies over decades, and
      a uniform grid there is catastrophically inaccurate (measured 22% error).
    - The cumulative sum runs from the OUTSIDE IN. Accumulating left-to-right
      and taking I(r) = C[-1] - C(r) subtracts two nearly-equal large sums at
      large r; that cancellation costs ~5e-06 on MWPotential2014 (and is
      invisible on Hernquist/NFW, which are insensitive to it).
    """
    rs = numpy.asarray(rs, dtype=float)
    if callable(beta) or rs[0] <= 0.0 or rs[-1] <= rs[0]:
        return None  # caller falls back to the per-radius loop
    Pot = _check_potential_list_and_deprecate(Pot)
    if dens is None:
        dens = lambda r: evaluateDensities(
            Pot,
            r * _INVSQRTTWO,
            r * _INVSQRTTWO,
            phi=numpy.pi / 4.0,
            use_physical=False,
        )
    integrand = lambda x: (
        -(x ** (2.0 * beta))
        * dens(x)
        * evaluaterforces(
            Pot,
            x * _INVSQRTTWO,
            x * _INVSQRTTWO,
            phi=numpy.pi / 4.0,
            use_physical=False,
        )
    )
    # Whether this is actually faster depends entirely on the potential. The
    # grid costs nquad integrand evaluations; the per-radius loop costs len(rs)
    # ADAPTIVE quadratures, i.e. only ~50 evaluations each. So the grid does
    # several times MORE work and wins only when vectorization more than makes
    # up for it -- true for cheap analytic potentials (measured ~100x on NFW and
    # MWPotential2014), false for SCF/Multipole, where every point costs
    # spherical harmonics (measured 0.33x, i.e. 3x SLOWER). Rather than guess or
    # keep a per-potential list, time both and pick the winner. The probe also
    # doubles as the array-capability check: potentials whose methods reject
    # arrays (e.g. DoubleExponentialDiskPotential) fall back here.
    probe = numpy.geomspace(rs[0], rs[-1], 256)
    try:
        _t0 = time.perf_counter()
        if numpy.shape(integrand(probe)) != probe.shape:
            return None
        per_point = (time.perf_counter() - _t0) / probe.size
    except (TypeError, ValueError, IndexError, AttributeError):
        return None
    _t0 = time.perf_counter()
    integrate.quad(integrand, rs[rs.size // 2], numpy.inf)
    per_quad = time.perf_counter() - _t0
    if per_point * nquad > 0.5 * per_quad * rs.size:
        return None  # the per-radius loop is cheaper for this potential
    xs = numpy.geomspace(rs[0], rs[-1], nquad)
    gs = numpy.asarray(integrand(xs), dtype=float)
    seg = 0.5 * (gs[1:] + gs[:-1]) * numpy.diff(xs)
    I = numpy.concatenate((numpy.cumsum(seg[::-1])[::-1], [0.0]))
    I = I + integrate.quad(integrand, rs[-1], numpy.inf)[0]  # r > max(rs) tail
    Ir = numpy.interp(numpy.log(rs), numpy.log(xs), I)
    return numpy.sqrt(Ir / _eval_on_grid(dens, rs) / rs ** (2.0 * beta))


@potential_physical_input
@physical_conversion("velocity", pop=True)
def sigmalos(Pot, R, dens=None, surfdens=None, beta=0.0, sigma_r=None):
    """
    Compute the line-of-sight velocity dispersion using the spherical Jeans equation

    Parameters
    ----------
    Pot : potential or a combined potential formed using addition (pot1+pot2+…)
        Gravitational potential; evaluated at R=r/sqrt(2),z=r/sqrt(2), sphericity not checked.
    R : float or Quantity
        Galactocentric projected radius
    dens : function, optional
        tracer density profile (function of r); if None, the density is assumed to be that corresponding to the potential
    surfdens : float or function, optional
        tracer surface density profile (value at R or function of R); if None, the surface density is assumed to be that corresponding to the density
    beta : float or function, optional
        anisotropy; can be a constant or a function of r
    sigma_r : float or function, optional
        if given, the solution of the spherical Jeans equation sigma_r(r) (used instead of solving the Jeans equation as part of this routine)

    Returns
    -------
    float
        sigma_los(R)

    Notes
    -----
    - 2018-08-27 - Written - Bovy (UofT)
    """
    Pot = _check_potential_list_and_deprecate(Pot)
    if dens is None:
        densPot = True
        dens = lambda r: evaluateDensities(
            Pot, r * _INVSQRTTWO, r * _INVSQRTTWO, use_physical=False
        )
    else:
        densPot = False
    if callable(surfdens):
        called_surfdens = surfdens(R)
    elif surfdens is None:
        if densPot:
            called_surfdens = evaluateSurfaceDensities(
                Pot, R, numpy.inf, use_physical=False
            )
        if not densPot or numpy.isnan(called_surfdens):
            called_surfdens = (
                2.0
                * integrate.quad(
                    lambda x: dens(numpy.sqrt(R**2.0 + x**2.0)), 0.0, numpy.inf
                )[0]
            )
    else:
        called_surfdens = surfdens
    if callable(beta):
        call_beta = beta
    else:
        call_beta = lambda x: beta
    if sigma_r is None:
        call_sigma_r = lambda r: sigmar(
            Pot, r, dens=dens, beta=beta, use_physical=False
        )
    elif not callable(sigma_r):
        call_sigma_r = lambda x: sigma_r
    else:
        call_sigma_r = sigma_r
    return numpy.sqrt(
        2.0
        * integrate.quad(
            lambda x: (
                (1.0 - call_beta(x) * R**2.0 / x**2.0)
                * x
                * dens(x)
                * call_sigma_r(x) ** 2.0
                / numpy.sqrt(x**2.0 - R**2.0)
            ),
            R,
            numpy.inf,
        )[0]
        / called_surfdens
    )
