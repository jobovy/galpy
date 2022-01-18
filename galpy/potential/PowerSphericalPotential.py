###############################################################################
#   PowerSphericalPotential.py: General class for potentials derived from 
#                               densities with two power-laws
#
#                                     amp
#                          rho(r)= ---------
#                                   r^\alpha
###############################################################################
import numpy
from scipy import special
from ..util import conversion
from .Potential import Potential
class PowerSphericalPotential(Potential):
    """Class that implements spherical potentials that are derived from power-law density models

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{r_1^3}\\,\\left(\\frac{r_1}{r}\\right)^{\\alpha}

    """
    def __init__(self,amp=1.,alpha=1.,normalize=False,r1=1.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a power-law-density potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass or Gxmass

           alpha - power-law exponent

           r1= (1.) reference radius for amplitude (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        r1= conversion.parse_length(r1,ro=self._ro)
        self.alpha= alpha
        # Back to old definition
        if self.alpha != 3.:
            self._amp*= r1**(self.alpha-3.)*4.*numpy.pi/(3.-self.alpha)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2010-07-10 - Started - Bovy (NYU)
        """
        r2= R**2.+z**2.
        if self.alpha == 2.:
            return numpy.log(r2)/2. 
        elif isinstance(r2,(float,int)) and r2 == 0 and self.alpha > 2:
            return -numpy.inf
        else:
            out= -r2**(1.-self.alpha/2.)/(self.alpha-2.)
            if isinstance(r2,numpy.ndarray) and self.alpha > 2:
                out[r2 == 0]= -numpy.inf
            return out                

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return -R/(R**2.+z**2.)**(self.alpha/2.)

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return -z/(R**2.+z**2.)**(self.alpha/2.)

    def _rforce_jax(self,r):
        """
        NAME:
           _rforce_jax
        PURPOSE:
           evaluate the spherical radial force for this potential using JAX
        INPUT:
           r - Galactocentric spherical radius
        OUTPUT:
           the radial force
        HISTORY:
           2021-02-14 - Written - Bovy (UofT)
        """
        # No need for actual JAX!
        return -self._amp/r**(self.alpha-1.)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rderiv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2011-10-09 - Written - Bovy (NYU)
        """
        return 1./(R**2.+z**2.)**(self.alpha/2.)\
            -self.alpha*R**2./(R**2.+z**2.)**(self.alpha/2.+1.)

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t- time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2012-07-26 - Written - Bovy (IAS@MPIA)
        """
        return self._R2deriv(z,R) #Spherical potential

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAs)
        """
        return -self.alpha*R*z*(R**2.+z**2.)**(-1.-self.alpha/2.)

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2013-01-09 - Written - Bovy (IAS)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return (3.-self.alpha)/4./numpy.pi/r**self.alpha

    def _ddensdr(self,r,t=0.):
        """
        NAME:
           _ddensdr
        PURPOSE:
           evaluate the radial density derivative for this potential
        INPUT:
           r - spherical radius
           t= time
        OUTPUT:
           the density derivative
        HISTORY:
           2021-02-25 - Written - Bovy (UofT)
        """
        return -self._amp\
            *self.alpha*(3.-self.alpha)/4./numpy.pi/r**(self.alpha+1.)

    def _d2densdr2(self,r,t=0.):
        """
        NAME:
           _d2densdr2
        PURPOSE:
           evaluate the second radial density derivative for this potential
        INPUT:
           r - spherical radius
           t= time
        OUTPUT:
           the 2nd density derivative
        HISTORY:
           2021-02-25 - Written - Bovy (UofT)
        """
        return self._amp*(self.alpha+1.)*self.alpha\
            *(3.-self.alpha)/4./numpy.pi/r**(self.alpha+2.)

    def _ddenstwobetadr(self,r,beta=0):
        """
        NAME:
           _ddenstwobetadr
        PURPOSE:
           evaluate the radial density derivative x r^(2beta) for this potential
        INPUT:
           r - spherical radius
           beta= (0)
        OUTPUT:
           d (rho x r^{2beta} ) / d r
        HISTORY:
           2021-02-14 - Written - Bovy (UofT)
        """
        return -self._amp*(self.alpha-2.*beta)\
            *(3.-self.alpha)/4./numpy.pi/r**(self.alpha+1.-2.*beta)
    
    def _surfdens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _surfdens
        PURPOSE:
           evaluate the surface density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        return (3.-self.alpha)/2./numpy.pi*z*R**-self.alpha\
            *special.hyp2f1(0.5,self.alpha/2.,1.5,-(z/R)**2)

class KeplerPotential(PowerSphericalPotential):
    """Class that implements the Kepler (point mass) potential

    .. math::

        \\Phi(r) = -\\frac{\\mathrm{amp}}{r}

    with :math:`\\mathrm{amp} = GM` the mass.
    """
    def __init__(self,amp=1.,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Kepler, point-mass potential

        INPUT:

           amp - amplitude to be applied to the potential, the mass of the point mass (default: 1); can be a Quantity with units of mass density or Gxmass density

           alpha - inner power

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        PowerSphericalPotential.__init__(self,amp=amp,normalize=normalize,
                                         alpha=3.,ro=ro,vo=vo)

    def _mass(self,R,z=None,t=0.):
        """
        NAME:
           _mass
        PURPOSE:
           evaluate the mass within R for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           t - time
        OUTPUT:
           the mass enclosed
        HISTORY:
           2014-07-02 - Written - Bovy (IAS)
        """
        return 1.
