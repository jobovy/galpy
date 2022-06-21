###############################################################################
#   SphericalPotential.py: base class for potentials corresponding to 
#                          spherical density profiles
###############################################################################
import numpy
from scipy import integrate
from .Potential import Potential
class SphericalPotential(Potential):
    """Base class for spherical potentials.

Implement a specific spherical density distribution with this form by inheriting from this class and defining functions:

* ``_revaluate(self,r,t=0.)``: the potential as a function of ``r`` and time;

* ``_rforce(self,r,t=0.)``: the radial force as a function of ``r`` and time;

* ``_r2deriv(self,r,t=0.)``: the second radial derivative of the potential as a function of ``r`` and time;

* ``_rdens(self,r,t=0.)``: the density as a function of ``r`` and time (if *not* implemented, calculated using the Poisson equation).
    """
    def __init__(self,amp=1.,ro=None,vo=None,amp_units=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a spherical potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1); can be a Quantity with units that depend on the specific spherical potential

           amp_units - ('mass', 'velocity2', 'density') type of units that amp should have if it has units (passed to Potential.__init__)

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2020-03-30 - Written - Bovy (UofT)

        """
        Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units=amp_units)
        return None

    def _rdens(self,r,t=0.):
        """Implement using the Poisson equation in case this isn't implemented"""
        return (self._r2deriv(r,t=t)-2.*self._rforce(r,t=t)/r)/4./numpy.pi
    
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
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._revaluate(r,t=t)

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
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._rforce(r,t=t)*R/r

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
           the radial force
        HISTORY:
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._rforce(r,t=t)*z/r

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
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._r2deriv(r,t=t)*R**2./r**2.-self._rforce(r,t=t)*z**2./r**3.

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
           the second radial derivative
        HISTORY:
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._r2deriv(r,t=t)*z**2./r**2.-self._rforce(r,t=t)*R**2./r**3.

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed radial, vertical  derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial, vertical derivative
        HISTORY:
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._r2deriv(r,t=t)*R*z/r**2.+self._rforce(r,t=t)*R*z/r**3.

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
           2020-03-30 - Written - Bovy (UofT)
        """
        r= numpy.sqrt(R**2.+z**2.)
        return self._rdens(r,t=t)

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
           2021-03-15 - Written - Bovy (UofT)
        """
        if z is not None: raise AttributeError # use general implementation
        R= numpy.float64(R) # Avoid indexing issues
        return -R**2.*self._rforce(R,t=t)
