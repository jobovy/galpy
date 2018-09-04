###############################################################################
#   PlummerPotential.py: class that implements the Plummer potential
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                                    \sqrt(R^2+z^2+b^2)
###############################################################################
import numpy as nu
from .Potential import Potential, kms_to_kpcGyrDecorator, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class PlummerPotential(Potential):
    """Class that implements the Plummer potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2+z^2+b^2}}

    with :math:`\\mathrm{amp} = GM` the total mass.
    """
    def __init__(self,amp=1.,b=0.8,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Plummer potential

        INPUT:

           amp - amplitude to be applied to the potential, the total mass (default: 1); can be a Quantity with units of mass or Gxmass

           b - scale parameter (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2015-06-15 - Written - Bovy (IAS)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(b,units.Quantity):
            b= b.to(units.kpc).value/self._ro
        self._b= b
        self._scale= self._b
        self._b2= self._b**2.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= True
        self._nemo_accname= 'Plummer'

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
           2015-06-15 - Started - Bovy (IAS)
        """
        return -1./nu.sqrt(R**2.+z**2.+self._b2)

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
           2015-06-15 - Written - Bovy (IAS)
        """
        dPhidrr= -(R**2.+z**2.+self._b2)**-1.5
        return dPhidrr*R

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
           2015-06-15 - Written - Bovy (IAS)
        """
        dPhidrr= -(R**2.+z**2.+self._b2)**-1.5
        return dPhidrr*z

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
           2015-06-15 - Written - Bovy (IAS)
        """
        return 3./4./nu.pi*self._b2*(R**2.+z**2.+self._b2)**-2.5

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
           the density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        Rb= R**2.+self._b2
        return self._b2*z*(3.*Rb+2.*z**2.)/Rb**2.*(Rb+z**2.)**-1.5/2./nu.pi

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
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
           2015-06-15 - Written - Bovy (IAS)
        """
        return (self._b2-2.*R**2.+z**2.)*(R**2.+z**2.+self._b2)**-2.5

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
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2015-06-15 - Written - Bovy (IAS)
        """
        return (self._b2+R**2.-2.*z**2.)*(R**2.+z**2.+self._b2)**-2.5

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
           2015-06-15 - Written - Bovy (IAS)
        """
        return -3.*R*z*(R**2.+z**2.+self._b2)**-2.5

    @kms_to_kpcGyrDecorator
    def _nemo_accpars(self,vo,ro):
        """
        NAME:

           _nemo_accpars

        PURPOSE:

           return the accpars potential parameters for use of this potential with NEMO

        INPUT:

           vo - velocity unit in km/s

           ro - length unit in kpc

        OUTPUT:

           accpars string

        HISTORY:

           2014-12-18 - Written - Bovy (IAS)

        """
        ampl= self._amp*vo**2.*ro
        return "0,%s,%s" % (ampl,self._b*ro)
