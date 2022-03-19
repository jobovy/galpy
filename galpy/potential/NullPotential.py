###############################################################################
#   NullPotential.py: class that implements a constant potential
###############################################################################
from .Potential import Potential
class NullPotential(Potential):
    """Class that implements a constant potential with, thus, zero forces. Can be used, for example, for integrating orbits in the absence of forces or for adjusting the value of the total gravitational potential, say, at infinity"""
    normalize= property() # turn off normalize
    def __init__(self,amp=1.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a null potential: a constant potential with, thus, zero forces

        INPUT:

           amp - constant value of the potential (default: 1); can be a Quantity with units of velocity-squared

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2022-03-18 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='velocity2')
        self.hasC= True
        self.hasC_dxdv= True
        self.hasC_dens= True
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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 1.
     
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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.

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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.

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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.
     
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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.

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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.

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
           2022-03-18 - Written - Bovy (UofT)
        """
        return 0.
