###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                                        distance
###############################################################################
import numpy as nu
from Potential import Potential
class MovingObjectPotential(Potential):
    """Class that implements the potential coming from a moving object
                                 GM
    phi(R,z) = -  ---------------------------------
                               distance
    """
    def __init__(self,orbit,amp=1.,GM=1.,normalize=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a MovingObjectPotential

        INPUT:

           orbit - the Orbit of the object (Orbit object)

           amp= - amplitude to be applied to the potential (default: 1)

           GM - 'mass' of the object (degenerate with amp)

           normalize - if True, normalize such that vc(1.,0.)=1., or, if 
                       given as a number, such that the force is this fraction 
                       of the force necessary to make vc(1.,0.)=1. (at t=0)

        OUTPUT:

           (none)

        HISTORY:

           2011-04-10 - Started - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self._gm= GM
        self._orb= orbit
        if normalize:
            self.normalize(normalize)

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z, phi
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z,phi)
        HISTORY:
           2010104-10 - Started - Bovy (NYU)
        """
        #Calculate distance
        dist= _cyldist(R,phi,z,
                       self._orb.R(t),self._orb.phi(t),self._orb.z(t))
        #Evaluate potential
        return -self._gm/dist

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
           2011-04-10 - Written - Bovy (NYU)
        """
        #Calculate distance and difference vector
        (xd,yd,zd,dist)= _cyldiffdist(self._orb.R(t),self._orb.phi(t),
                                   self._orb.z(t),
                                   R,phi,z)
                                   
        #Evaluate force
        return self._gm*(nu.cos(phi)*xd+nu.sin(phi)*yd)/dist**3.

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
           2011-04-10 - Written - Bovy (NYU)
        """
        #Calculate distance and difference vector
        (xd,yd,zd,dist)= _cyldiffdist(self._orb.R(t),self._orb.phi(t),
                                   self._orb.z(t),
                                   R,phi,z)
                                   
        #Evaluate force
        return self._gm*zd/dist**3.

    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
           2011-04-10 - Written - Bovy (NYU)
        """
        #Calculate distance and difference vector
        (xd,yd,zd,dist)= _cyldiffdist(self._orb.R(t),self._orb.phi(t),
                                   self._orb.z(t),
                                   R,phi,z)
                                   
        #Evaluate force
        return self._gm*R*(nu.cos(phi)*yd-nu.sin(phi)*xd)/dist**3.


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
           2010-08-08 - Written - Bovy (NYU)
        """
        raise AttributeError("This is a delta function, not implemented yet")

def _cyldist(R1,phi1,z1,R2,phi2,z2):
    return nu.sqrt( (R1*nu.cos(phi1)-R2*nu.cos(phi2))**2.
                    +(R1*nu.sin(phi1)-R2*nu.sin(phi2))**2.
                    +(z1-z2)**2.)     

def _cyldiffdist(R1,phi1,z1,R2,phi2,z2):
    x= R1*nu.cos(phi1)-R2*nu.cos(phi2)
    y= R1*nu.sin(phi1)-R2*nu.sin(phi2)
    z= z1-z2
    return (x,y,z,nu.sqrt(x**2.+y**2.+z**2.))

