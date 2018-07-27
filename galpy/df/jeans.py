# jeans.py: utilities related to the Jeans equations
import numpy
from scipy import integrate
from galpy.potential.Potential import evaluateDensities, \
    evaluaterforces
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
