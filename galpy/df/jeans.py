# jeans.py: utilities related to the Jeans equations
import numpy
from scipy import integrate

from ..backend import get_namespace, is_backend_array
from ..backend.quadrature import fixed_quad_semiinfinite, quad
from ..potential.Potential import (
    _check_potential_list_and_deprecate,
    evaluateDensities,
    evaluaterforces,
    evaluateSurfaceDensities,
)
from ..util.conversion import physical_conversion, potential_physical_input

_INVSQRTTWO = 1.0 / numpy.sqrt(2.0)


@potential_physical_input(coerce_backend=False)
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
    xp = get_namespace(r) if is_backend_array(r) else numpy
    if xp is numpy:
        # numpy path: scipy.integrate.quad over [r, inf) -- byte-identical.
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
    # jax/torch: fixed-order Gauss-Legendre on the mapped semi-infinite range,
    # differentiable in r and through the potential's parameters.
    if callable(beta):
        intFactor = lambda x: xp.exp(2.0 * quad(lambda y: beta(y) / y, 1.0, x))
    else:
        intFactor = lambda x: x ** (2.0 * beta)
    return xp.sqrt(
        fixed_quad_semiinfinite(
            xp,
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
        )
        / dens(r)
        / intFactor(r)
    )


@potential_physical_input(coerce_backend=False)
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
    xp = get_namespace(R) if is_backend_array(R) else numpy
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
        # xp.isnan == numpy.isnan on the numpy path (byte-identical); the
        # surfdens is a scalar here (scipy.quad / the R-scalar backend path).
        if not densPot or xp.isnan(called_surfdens):
            if xp is numpy:
                called_surfdens = (
                    2.0
                    * integrate.quad(
                        lambda x: dens(numpy.sqrt(R**2.0 + x**2.0)), 0.0, numpy.inf
                    )[0]
                )
            else:
                called_surfdens = 2.0 * fixed_quad_semiinfinite(
                    xp, lambda x: dens(xp.sqrt(R**2.0 + x**2.0)), 0.0
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
    if xp is numpy:
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

    # Substitute x = sqrt(R^2 + s^2): dx = (s/x) ds and sqrt(x^2-R^2) = s cancel
    # the endpoint singularity, leaving a smooth integrand over s in [0, inf) that
    # fixed-order GL integrates accurately (the raw 1/sqrt(x^2-R^2) form loses ~1%).
    def _los_integrand(s):
        x = xp.sqrt(R**2.0 + s**2.0)
        return (1.0 - call_beta(x) * R**2.0 / x**2.0) * dens(x) * call_sigma_r(x) ** 2.0

    return xp.sqrt(
        2.0 * fixed_quad_semiinfinite(xp, _los_integrand, 0.0) / called_surfdens
    )
