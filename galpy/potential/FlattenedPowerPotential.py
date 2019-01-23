###############################################################################
#   FlattenedPowerPotential.py: Power-law potential that is flattened in the 
#                               potential (NOT the density)
#
#                                     amp
#                          phi(R,z)= --------- ; m^2 = R^2 + z^2/q^2
#                                   m^\alpha
###############################################################################
import numpy as nu
from scipy import special, integrate
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
_CORE=10**-8
class FlattenedPowerPotential(Potential):
    """Class that implements a power-law potential that is flattened in the potential (NOT the density)

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}\,r_1^\\alpha}{\\alpha\\,\\left(R^2+(z/q)^2+\\mathrm{core}^2\\right)^{\\alpha/2}}

    and the same as LogarithmicHaloPotential for :math:`\\alpha=0`

    See Figure 1 in `Evans (1994) <http://adsabs.harvard.edu/abs/1994MNRAS.267..333E>`_ for combinations of alpha and q that correspond to positive densities

    """
    def __init__(self,amp=1.,alpha=0.5,q=0.9,core=_CORE,normalize=False,r1=1.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a flattened power-law potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units of velocity-squared

           alpha - power

           q - flattening

           core - core radius (can be Quantity)

           r1= (1.) reference radius for amplitude (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2013-01-09 - Written - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='velocity2')
        if _APY_LOADED and isinstance(core,units.Quantity):
            core= core.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(r1,units.Quantity):
            r1= r1.to(units.kpc).value/self._ro
        self.alpha= alpha
        self.q2= q**2.
        self.core2= core**2.
        # Back to old definition
        self._amp*= r1**self.alpha
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True

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
           2013-01-09 - Started - Bovy (IAS)
        """
        if self.alpha == 0.:
            return 1./2.*nu.log(R**2.+z**2./self.q2+self.core2)
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return -m2**(-self.alpha/2.)/self.alpha

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
        if self.alpha == 0.:
            return -R/(R**2.+z**2./self.q2+self.core2)
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return -m2**(-self.alpha/2.-1.)*R

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
        if self.alpha == 0.:
            return -z/self.q2/(R**2.+z**2./self.q2+self.core2)
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return -m2**(-self.alpha/2.-1.)*z/self.q2

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
        if self.alpha == 0.:
            denom= 1./(R**2.+z**2./self.q2+self.core2)
            return denom-2.*R**2.*denom**2.
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return -m2**(-self.alpha/2.-1.)*((self.alpha+2)*R**2./m2-1.)

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
        if self.alpha == 0.:
            denom= 1./(R**2.+z**2./self.q2+self.core2)
            return denom/self.q2-2.*z**2.*denom**2./self.q2**2.
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return -1./self.q2*m2**(-self.alpha/2.-1.)*((self.alpha+2)*z**2./m2/self.q2-1.)

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density force for this potential
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
        if self.alpha == 0.:
            return 1./4./nu.pi/self.q2*((2.*self.q2+1.)*self.core2+R**2.\
                                       +(2.-1./self.q2)*z**2.)/\
                                       (R**2.+z**2./self.q2+self.core2)**2.
        else:
            m2= self.core2+R**2.+z**2./self.q2
            return 1./self.q2*(self.core2*(1.+2.*self.q2)+R**2.*(1.-self.alpha*self.q2)+z**2.*(2.-(1.+self.alpha)/self.q2))*m2**(-self.alpha/2.-2.)/4./nu.pi
