###############################################################################
#   HomogeneousSpherePotential.py: The potential of a homogeneous sphere
###############################################################################
import numpy
from ..util import conversion
from .Potential import Potential
class HomogeneousSpherePotential(Potential):
    """Class that implements the homogeneous sphere potential for :math:`\\rho(r) = \\rho_0 = \\mathrm{constant}` for all :math:`r < R` and zero otherwise. The potential is given by

    .. math::

        \\Phi(r) = \\mathrm{amp}\\times\\left\\{\\begin{array}{lr}
        (r^2-3R^2), & \\text{for } r < R\\\\
        -\\frac{2R^3}{r} & \\text{for } r \\geq R
        \\end{array}\\right.

    We have that :math:`\\rho_0 = 3\\,\\mathrm{amp}/[2\\pi G]`.
    """
    def __init__(self,amp=1.,R=1.1,normalize=False,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a homogeneous sphere potential

        INPUT:

           amp= amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

           R= size of the sphere (can be Quantity)

           normalize= if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2019-12-20 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='density')
        R= conversion.parse_length(R,ro=self._ro)
        self.R= R
        self._R2= self.R**2.
        self._R3= self.R**3.
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): #pragma: no cover
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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return r2-3.*self._R2
        else:
            return -2.*self._R3/numpy.sqrt(r2)

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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return -2.*R
        else:
            return -2.*self._R3*R/r2**1.5

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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return -2.*z
        else:
            return -2.*self._R3*z/r2**1.5

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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return 2.
        else:
            return 2.*self._R3/r2**1.5-6.*self._R3*R**2./r2**2.5

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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return 2.
        else:
            return 2.*self._R3/r2**1.5-6.*self._R3*z**2./r2**2.5

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
           t- time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return 0.
        else:
            return -6.*self._R3*R*z/r2**2.5

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
           2019-12-20 - Written - Bovy (UofT)
        """
        r2= R**2.+z**2.
        if r2 < self._R2:
            return 1.5/numpy.pi
        else:
            return 0.
