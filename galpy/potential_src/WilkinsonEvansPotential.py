##########################################################################################################################################
#   WilkinsonEvansPotential.py: class that implements the Wilkinson and Evans 
#                               (1991) potential in Irrgang et al. (2013) modification
# 
#                                            Mh    ( sqrt(r^2 + ah^2) + ah )
#                              phi(R,z) = - --- ln (-----------------------)
#                                            ah    (           R           )
#
###########################################################################################################################################
import numpy as nu
from galpy.potential_src.Potential import Potential, kms_to_kpcGyrDecorator
class WilkinsonEvansPotential(Potential):
    """Class that implements the Wilkinson and Evans (1999) halo potential

    .. math::

   \\Phi_h(r, z) = -{{ M_h \\log \\left({{\\sqrt{z^2+r^2+a_h^2}+a_h}\\over{ \\sqrt{z^2+r^2}}}\\right)}\\over{ a_h}}

    """
    def __init__(self,Mh=69725.0,ah=200, normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize the Wilkinson and Evans (1999) halo potential

        INPUT:

           Mh - amplitude to be applied to the potential - M_h (default: 1)

           ah - "scale length" (in terms of Ro)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if given as a number, such that the force is this fraction of the force necessary to make vc(1.,0.)=1.

        OUTPUT:

           (none)

        HISTORY:

           2015-12-26 - Written - Igoshev A.P. (RU)

        """
        Potential.__init__(self,amp=1.0)
        self._Mh= Mh
        self._ah= ah
        if normalize or \
                (isinstance(normalize,(int,float)) \
                     and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= True
        self.hasC_dxdv= False
        self._nemo_accname= 'WilkinsonEvans'

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
           2015-12-26 - Written - Igoshev A.P. (RU)
        """

        return -self._Mh/self._ah * nu.log((nu.sqrt(R*R+z*z + self._ah*self._ah) + self._ah) / nu.sqrt(R*R + z*z))

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
           2015-12-26 - Written - Igoshev A.P. (RU)
        """
#        return -R/(R**2.+(self._a+nu.sqrt(z**2.+self._b2))**2.)**(3./2.)
        return self._Mh*nu.sqrt(z**2+R**2)*(R/(nu.sqrt(z**2+R**2)*nu.sqrt(z**2+R**2+self._ah**2))-R*(z**2+R**2)**((-3.0)/2.0)*(nu.sqrt(z**2+R**2+self._ah**2)+self._ah))/(self._ah*(nu.sqrt(z**2+R**2+self._ah**2)+self._ah))

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
           2015-12-26 - Written - Igoshev A.P. (RU)
        """
        return self._Mh*nu.sqrt(z**2+R**2)*(z/(nu.sqrt(z**2+R**2)*nu.sqrt(z**2+R**2+self._ah**2))-z*(z**2+R**2)**((-3.0)/2.0)*(nu.sqrt(z**2+R**2+self._ah**2)+self._ah))/(self._ah*(nu.sqrt(z**2+R**2+self._ah**2)+self._ah))

