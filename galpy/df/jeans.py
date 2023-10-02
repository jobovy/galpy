# jeans.py: utilities related to the Jeans equations
import numpy
from scipy import integrate

from ..potential.Potential import (
    evaluateDensities,
    evaluaterforces,
    evaluateSurfaceDensities,
)
from ..potential.Potential import flatten as flatten_pot
from ..util.conversion import physical_conversion, potential_physical_input

_INVSQRTTWO = 1.0 / numpy.sqrt(2.0)


@potential_physical_input
@physical_conversion("velocity", pop=True)
def sigmar(Pot, r, dens=None, beta=0.0):
    """
    Compute the radial velocity dispersion using the spherical Jeans equation

    Parameters
    ----------
    Pot : potential or list of potentials
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
    Pot = flatten_pot(Pot)
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
            lambda x: -intFactor(x)
            * dens(x)
            * evaluaterforces(
                Pot,
                x * _INVSQRTTWO,
                x * _INVSQRTTWO,
                phi=numpy.pi / 4.0,
                use_physical=False,
            ),
            r,
            numpy.inf,
        )[0]
        / dens(r)
        / intFactor(r)
    )


@potential_physical_input
@physical_conversion("velocity", pop=True)
def sigmalos(Pot, R, dens=None, surfdens=None, beta=0.0, sigma_r=None):
    """
    Compute the line-of-sight velocity dispersion using the spherical Jeans equation

    Parameters
    ----------
    Pot : potential or list of potentials
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
    Pot = flatten_pot(Pot)
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
        call_sigma_r = lambda r: sigmar(Pot, r, dens=dens, beta=beta)
    elif not callable(sigma_r):
        call_sigma_r = lambda x: sigma_r
    else:
        call_sigma_r = sigma_r
    return numpy.sqrt(
        2.0
        * integrate.quad(
            lambda x: (1.0 - call_beta(x) * R**2.0 / x**2.0)
            * x
            * dens(x)
            * call_sigma_r(x) ** 2.0
            / numpy.sqrt(x**2.0 - R**2.0),
            R,
            numpy.inf,
        )[0]
        / called_surfdens
    )
