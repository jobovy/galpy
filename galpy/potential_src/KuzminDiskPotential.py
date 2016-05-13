###############################################################################
#   KuzminDiskPotential.py: class that implements Kuzmin disk potential
#
#                                   - amp                   
#               Phi(R, z)=  ---------------------------
#                            \sqrt{R^2 + (a + |z|)^2} 
###############################################################################

import numpy as nu
import warnings
from scipy import special, integrate
from galpy.util import galpyWarning
from galpy.potential_src.Potential import Potential, kms_to_kpcGyrDecorator, \
    _APY_LOADED
if _APY_LOADED:
    from astropy import units

class KuzminDiskPotential(Potential):
	"""Class that implements the Kuzmin Disk potential

    .. math::

        \\Phi(R,z) = -\\frac{\\mathrm{amp}}{\\sqrt{R^2 + (a + |z|)^2}}
	"""
	def __init__(self, amp=1., a=1. ,normalize=False, ro=None,vo=None):
		"""
        NAME:

            __init__

        PURPOSE:

            initialize a Kuzmin disk Potential

        INPUT:

            amp       - amplitude to be applied to the potential (default: 1); can be a Quantity with units of mass density or Gxmass density

            a        - Parameter
    
            normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           KuzminDiskPotential object

        HISTORY:

           2016-05-09 - Written - Aladdin 

        """
		Potential.__init__(self,amp=amp,ro=ro,vo=vo,amp_units='mass')
		if _APY_LOADED and isinstance(a,units.Quantity): 
			a= a.to(units.kpc).value/self._ro 

		self._a = a ## a must be greator or equal to 0. 
		##Should there be a check for this? Or should I just take the absolute value of a?
		
		if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)): 
			self.normalize(normalize)

                self.hasC = True



	def _evaluate(self,R,z,phi=0.,t=0.):
		"""
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at (R,z)
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           potential at (R,z)
        HISTORY:
           2016-05-09 - Written - Aladdin 
        """
		return -self._denom(R, z)**-0.5

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
            the radial force = -dphi/dR
        HISTORY:
            2016-05-09 - Written - Aladdin 
		"""
		return -self._denom(R, z)**-1.5 * R

	def _zforce(self, R, z, phi=0., t=0.):
		"""
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force  for this potential
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force = -dphi/dz
        HISTORY:
           2016-05-09 - Written - Aladdin 
		"""
		return -nu.sign(z) * self._denom(R,z)**-1.5 * (self._a + nu.fabs(z))
		

	def _denom(self, R, z):
		"""
        NAME:
           _denom
        PURPOSE:
           evaluate R^2 + (a + |z|)^2 which is used in the denominator
		   of most equations
        INPUT:
           R - Cylindrical Galactocentric radius
           z - vertical height
        OUTPUT:
           R^2 + (a + |z|)^2
        HISTORY:
           2016-05-09 - Written - Aladdin 
		"""
		return (R**2. + (self._a + nu.fabs(z))**2.)

