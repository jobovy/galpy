###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
#                                                           GM
#                              phi(R,z) = -  ---------------------------------
#                                                        distance
###############################################################################
import numpy as nu
from Potential import Potential
from galpy.potential_src.ForceSoftening import PlummerSoftening
class MovingObjectPotential(Potential):
    """Class that implements the potential coming from a moving object

    .. math::

        \\Phi(R,z,\\phi,t) = -\\mathrm{amp}\\,GM\\,S(d)

    where :math:`d` is the distance between :math:`(R,z,\\phi)` and the moving object at time :math:`t` and :math:`S(\\cdot)` is a softening kernel. In the case of Plummer softening, this kernel is

    .. math::

        S(d) = \\frac{1}{\\sqrt{d^2+\\mathrm{softening\_length}^2}}

    Plummer is currently the only implemented softening.

    """
    def __init__(self,orbit,amp=1.,GM=.06,
                 softening=None,
                 softening_model='plummer',softening_length=0.01):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a MovingObjectPotential

        INPUT:

           orbit - the Orbit of the object (Orbit object)

           amp= - amplitude to be applied to the potential (default: 1)

           GM - 'mass' of the object (degenerate with amp)

           Softening: either provide

              a) softening= with a ForceSoftening-type object

              b) softening_model=  type of softening to use ('plummer')

                 softening_length= (optional)

        OUTPUT:

           (none)

        HISTORY:

           2011-04-10 - Started - Bovy (NYU)

        """
        Potential.__init__(self,amp=amp)
        self._gm= GM
        self._orb= orbit
        if softening is None:
            if softening_model.lower() == 'plummer':
                self._softening= PlummerSoftening(softening_length=softening_length)
        else:
            self._softening= softening
        self.isNonAxi= True
        return None

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
        return -self._gm*self._softening.potential(dist)

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
        return self._gm*(nu.cos(phi)*xd+nu.sin(phi)*yd)/dist\
            *self._softening(dist)

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
        return self._gm*zd/dist*self._softening(dist)

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
        return self._gm*R*(nu.cos(phi)*yd-nu.sin(phi)*xd)/dist\
            *self._softening(dist)

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
        dist= _cyldist(R,phi,z,
                       self._orb.R(t),self._orb.phi(t),self._orb.z(t))
        return self._gm*self._softening.density(dist)

def _cyldist(R1,phi1,z1,R2,phi2,z2):
    return nu.sqrt( (R1*nu.cos(phi1)-R2*nu.cos(phi2))**2.
                    +(R1*nu.sin(phi1)-R2*nu.sin(phi2))**2.
                    +(z1-z2)**2.)     

def _cyldiffdist(R1,phi1,z1,R2,phi2,z2):
    x= R1*nu.cos(phi1)-R2*nu.cos(phi2)
    y= R1*nu.sin(phi1)-R2*nu.sin(phi2)
    z= z1-z2
    return (x,y,z,nu.sqrt(x**2.+y**2.+z**2.))

