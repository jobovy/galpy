###############################################################################
#   LogarithmicHaloPotential.py: class that implements the logarithmic halo
#                                halo potential Phi(r) = vc**2 ln(r)
###############################################################################
import numpy as nu
from Potential import Potential
_CORE=10**-8
class LogarithmicHaloPotential(Potential):
    """Class that implements the logarithmic halo potential

    .. math::

        \\Phi(R,z) = \\frac{\\mathrm{amp}}{2}\\,\\ln\\left(R^2+(z/q)^2+\\mathrm{core}^2\\right)

    """
    def __init__(self,amp=1.,core=_CORE,q=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Logarithmic Halo potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1)

           core - core radius at which the logarithm is cut

           q - potential flattening (z/q)**2.

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2010-04-02 - Started - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self.hasC= True
        self.hasC_dxdv= True
        self._core2= core**2.
        self._q= q
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover 
            self.normalize(normalize)
        return None

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
           2010-04-02 - Started - Bovy (NYU)
           2010-04-30 - Adapted for R,z - Bovy (NYU)
        """
        return 1./2.*nu.log(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return -R/(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return -z/self._q**2./(R**2.+(z/self._q)**2.+self._core2)

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
        """
        return 1./4./nu.pi/self._q**2.*((2.*self._q**2.+1.)*self._core2+R**2.\
                                           +(2.-self._q**-2.)*z**2.)/\
                                           (R**2.+(z/self._q)**2.+self._core2)**2.

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
           2011-10-09 - Written - Bovy (IAS)
        """
        denom= 1./(R**2.+(z/self._q)**2.+self._core2)
        return denom-2.*R**2.*denom**2.

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
           2012-07-25 - Written - Bovy (IAS@MPIA)
        """
        denom= 1./(R**2.+(z/self._q)**2.+self._core2)
        return denom/self._q**2.-2.*z**2.*denom**2./self._q**4.

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
           d2Phi/dR/dz
        HISTORY:
           2013-08-28 - Written - Bovy (IAS)
        """
        return -2.*R*z/self._q**2./(R**2.+(z/self._q)**2.+self._core2)**2.

