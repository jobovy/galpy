###############################################################################
#   SphericalShellPotential.py: The gravitational potential of a thin, 
#                               spherical shell
###############################################################################
import numpy as nu
from .Potential import Potential, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class SphericalShellPotential(Potential):
    """Class that implements the potential of an infinitesimally-thin, spherical shell

    .. math::

        \\rho(r) = \\frac{\\mathrm{amp}}{4\pi\,a^2}\\,\\delta(r-a)

    with :math:`\\mathrm{amp} = GM` the mass of the shell.
    """
    def __init__(self,amp=1.,a=0.75,normalize=False,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a spherical shell potential

        INPUT:

           amp - mass of the shell (default: 1); can be a Quantity with units of mass or Gxmass

           a= (0.75) radius of the shell (can be Quantity)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.; note that because the force is always zero at r < a, this does not work if a > 1

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2018-08-04 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        if _APY_LOADED and isinstance(a,units.Quantity):
            a= a.to(units.kpc).value/self._ro
        self.a= a
        self.a2= a**2
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            if self.a > 1.:
                raise ValueError('SphericalShellPotential with normalize= for a > 1 is not supported (because the force is always 0 at r=1)')
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv= False

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
           2018-08-04 - Written - Bovy (UofT)
        """
        r2= R**2+z**2
        if r2 <= self.a2:
            return -1./self.a
        else:
            return -1./nu.sqrt(r2)

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
           2018-08-04 - Written - Bovy (UofT)
        """
        r= nu.sqrt(R**2+z**2)
        if r <= self.a:
            return 0.
        else:
            return -R/r**3

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
           2018-08-04 - Written - Bovy (UofT)
        """
        r= nu.sqrt(R**2+z**2)
        if r <= self.a:
            return 0.
        else:
            return -z/r**3

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
           2018-08-04 - Written - Bovy (UofT)
        """
        r= nu.sqrt(R**2+z**2)
        if r <= self.a:
            return 0.
        else:
            return (z**2-2*R**2)/r**5

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
           2018-08-04 - Written - Bovy (UofT)
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
           2018-08-04 - Written - Bovy (UofT)
        """
        r= nu.sqrt(R**2+z**2)
        if r <= self.a:
            return 0.
        else:
            return -3*R*z/r**5

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
           the surface density
        HISTORY:
           2018-08-19 - Written - Bovy (UofT)
        """
        r2= R**2+z**2
        if r2 != self.a2:
            return 0.
        else: # pragma: no cover
            return nu.infty

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
           2018-08-04 - Written - Bovy (UofT)
        """
        if R > self.a: return 0.
        h= nu.sqrt(self.a2-R**2)
        if z < h: return 0.
        else: return 1./(2.*nu.pi*self.a*h)
