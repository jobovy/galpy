# jeans.py: utilities related to the Jeans equations
import numpy
from scipy import integrate
from galpy.potential.Potential import evaluateDensities, \
    evaluaterforces, evaluateSurfaceDensities
from galpy.potential.Potential import flatten as flatten_pot
from galpy.util.bovy_conversion import physical_conversion, \
    potential_physical_input
_INVSQRTTWO= 1./numpy.sqrt(2.)
@potential_physical_input
@physical_conversion('velocity',pop=True)
def sigmar(Pot,r,dens=None,beta=0.):
    """
    NAME:

       sigmar

    PURPOSE:

       Compute the radial velocity dispersion using the spherical Jeans equation

    INPUT:

       Pot - potential or list of potentials (evaluated at R=r/sqrt(2),z=r/sqrt(2), sphericity not checked)

       r - Galactocentric radius (can be Quantity)

       dens= (None) tracer density profile (function of r); if None, the density is assumed to be that corresponding to the potential

       beta= (0.) anisotropy; can be a constant or a function of r
       
    OUTPUT:

       sigma_r(r)

    HISTORY:

       2018-07-05 - Written - Bovy (UofT)

    """
    Pot= flatten_pot(Pot)
    if dens is None:
        dens= lambda r: evaluateDensities(Pot,r*_INVSQRTTWO,r*_INVSQRTTWO,
                                          use_physical=False)
    if callable(beta):
        intFactor= lambda x: numpy.exp(2.*integrate.quad(lambda y: beta(y)/y,
                                       1.,x)[0])
    else: # assume to be number
        intFactor= lambda x: x**(2.*beta)
    return numpy.sqrt(integrate.quad(lambda x: -intFactor(x)*dens(x)
                                     *evaluaterforces(Pot,
                                                      x*_INVSQRTTWO,
                                                      x*_INVSQRTTWO,
                                                      use_physical=False),
                                     r,numpy.inf)[0]/
                      dens(r)/intFactor(r))

@potential_physical_input
@physical_conversion('velocity',pop=True)
def sigmalos(Pot,R,dens=None,surfdens=None,beta=0.,sigma_r=None):
    """
    NAME:

       sigmalos

    PURPOSE:

       Compute the line-of-sight velocity dispersion using the spherical Jeans equation

    INPUT:

       Pot - potential or list of potentials (evaluated at R=r/sqrt(2),z=r/sqrt(2), sphericity not checked)

       R - Galactocentric projected radius (can be Quantity)

       dens= (None) tracer density profile (function of r); if None, the density is assumed to be that corresponding to the potential

       surfdens= (None) tracer surface density profile (value at R or function of R); if None, the surface density is assumed to be that corresponding to the density

       beta= (0.) anisotropy; can be a constant or a function of r

       sigma_r= (None) if given, the solution of the spherical Jeans equation sigma_r(r) (used instead of solving the Jeans equation as part of this routine)
       
    OUTPUT:

       sigma_los(R)

    HISTORY:

       2018-08-27 - Written - Bovy (UofT)

    """
    Pot= flatten_pot(Pot)
    if dens is None:
        densPot= True
        dens= lambda r: evaluateDensities(Pot,r*_INVSQRTTWO,r*_INVSQRTTWO,
                                          use_physical=False)
    else:
        densPot= False
    if callable(surfdens):
        called_surfdens= surfdens(R)
    elif surfdens is None:
        if densPot:
            called_surfdens= evaluateSurfaceDensities(Pot,R,numpy.inf,
                                                    use_physical=False)
        if not densPot or numpy.isnan(called_surfdens):
            called_surfdens=\
                    2.*integrate.quad(lambda x: dens(numpy.sqrt(R**2.+x**2.)),
                                      0.,numpy.inf)[0]
    else:
        called_surfdens= surfdens
    if callable(beta):
        call_beta= beta
    else:
        call_beta= lambda x: beta
    if sigma_r is None:
        call_sigma_r= lambda r: sigmar(Pot,r,dens=dens,beta=beta)
    elif not callable(sigma_r):
        call_sigma_r= lambda x: sigma_r
    else:
        call_sigma_r= sigma_r
    return numpy.sqrt(2.*integrate.quad(\
            lambda x: (1.-call_beta(x)*R**2./x**2.)*x*dens(x)\
                *call_sigma_r(x)**2./numpy.sqrt(x**2.-R**2.),R,numpy.inf)[0]\
                          /called_surfdens)
