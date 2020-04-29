###############################################################################
#   HenonHeilesPotential: the Henon-Heiles (1964) potential
###############################################################################
import numpy
from .planarPotential import planarPotential
class HenonHeilesPotential(planarPotential):
    """Class that implements a the `Henon & Heiles (1964) <http://adsabs.harvard.edu/abs/1964AJ.....69...73H>`__ potential
    
    .. math::

        \\Phi(R,\\phi) = \\frac{\\mathrm{amp}}{2}\\,\\left[R^2 + \\frac{2\\,R^3}{3}\\,\\sin\\left(3\,\phi\\right)\\right]

    """
    def __init__(self,amp=1.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a Henon-Heiles potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

        OUTPUT:

           (none)

        HISTORY:

           2017-10-16 - Written - Bovy (UofT)

        """
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        self.hasC= True
        self.hasC_dxdv= True

    def _evaluate(self,R,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,phi,t
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,phi,t)
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return 0.5*R*R*(1.+2./3.*R*numpy.sin(3.*phi))

    def _Rforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return -R*(1.+R*numpy.sin(3.*phi))
       
    def _phiforce(self,R,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return -R**3.*numpy.cos(3.*phi)

    def _R2deriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return 1.+2.*R*numpy.sin(3.*phi)
       
    def _phi2deriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return -3.*R**3.*numpy.sin(3.*phi)

    def _Rphideriv(self,R,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed radial, azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, azimuthal derivative
        HISTORY:
           2017-10-16 - Written - Bovy (UofT)
        """
        return 3.*R**2.*numpy.cos(3.*phi)

