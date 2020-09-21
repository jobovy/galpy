###############################################################################
#   SphericalShellPotential.py: The gravitational potential of a thin, 
#                               spherical shell
###############################################################################
import numpy
from ..util import conversion
from .SphericalPotential import SphericalPotential
class SphericalShellPotential(SphericalPotential):
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

           2020-03-30 - Re-implemented using SphericalPotential - Bovy (UofT)

        """
        SphericalPotential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
        a= conversion.parse_length(a,ro=self._ro)
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

    def _revaluate(self,r,t=0.):
        """The potential as a function of r"""
        if r <= self.a:
            return -1./self.a
        else:
            return -1./r

    def _rforce(self,r,t=0.):
        """The force as a function of r"""
        if r <= self.a:
            return 0.
        else:
            return -1/r**2.

    def _r2deriv(self,r,t=0.):
        """The second radial derivative as a function of r"""
        if r <= self.a:
            return 0.
        else:
            return -2./r**3.

    def _rdens(self,r,t=0.):
        """The density as a function of r"""
        if r != self.a:
            return 0.
        else: # pragma: no cover
            return numpy.infty

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
        h= numpy.sqrt(self.a2-R**2)
        if z < h: return 0.
        else: return 1./(2.*numpy.pi*self.a*h)
