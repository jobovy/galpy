###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
###############################################################################
import copy
import numpy
from .Potential import Potential, _isNonAxi, flatten, \
    evaluatePotentials, evaluateRforces, evaluatezforces, evaluateDensities, _check_c
from .PlummerPotential import PlummerPotential
class MovingObjectPotential(Potential):
    """
    Class that implements the potential coming from a moving object by combining
    any galpy potential with an integrated galpy orbit.
    """
    def __init__(self,orbit,pot=None,amp=1.0,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a MovingObjectPotential

        INPUT:

           orbit - the Orbit of the object (Orbit object)

           pot - A potential object or list of potential objects representing the potential of the moving object; should be spherical, but this is not checked [default= PlummerPotential(amp=0.06,b=0.01)]
           
           amp (=1.) another amplitude to apply to the potential

           ro=, vo= distance and velocity scales for translation into internal units (default from configuration file)

        OUTPUT:

           (none)

        HISTORY:

           2011-04-10 - Started - Bovy (NYU)

           2018-10-18 - Re-implemented to represent general object potentials using galpy potential models - James Lane (UofT)

        """

        Potential.__init__(self,amp=amp,ro=ro,vo=vo)       
        # If no potential supplied use a default Plummer sphere
        if pot is None:
            pot=PlummerPotential(amp=0.06,b=0.01)
            self._pot = pot
        else:
            pot=flatten(pot)
            if _isNonAxi(pot):
                raise NotImplementedError('MovingObjectPotential for non-axisymmetric potentials is not currently supported')
            self._pot=pot
        self._orb= copy.deepcopy(orbit)
        self._orb.turn_physical_off()
        self.isNonAxi= True
        self.hasC= _check_c(self._pot)
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
           2011-04-10 - Started - Bovy (NYU)
           2018-10-18 - Updated for general object potential - James Lane (UofT)
        """
        #Cylindrical distance
        Rdist = _cylR(R,phi,self._orb.R(t),self._orb.phi(t))
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        #Evaluate potential
        return evaluatePotentials(self._pot,Rdist,orbz-z,t=t,
                                  use_physical=False)

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
           2018-10-18 - Updated for general object potential - James Lane (UofT)
        """
        #Cylindrical distance
        Rdist = _cylR(R,phi,self._orb.R(t),self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd,yd,zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz,
            R, phi, z)
        #Evaluate cylindrical radial force
        RF = evaluateRforces(self._pot,Rdist,zd,t=t,use_physical=False)

        # Return R force, negative of radial vector to evaluation location.
        return -RF*(numpy.cos(phi)*xd+numpy.sin(phi)*yd)/Rdist

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
           2018-10-18 - Updated for general object potential - James Lane (UofT)
        """
        #Cylindrical distance
        Rdist = _cylR(R,phi,self._orb.R(t),self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd,yd,zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz,
            R, phi, z)
        #Evaluate and return z force
        return -evaluatezforces(self._pot,Rdist,zd,t=t,use_physical=False)

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
           2018-10-18 - Updated for general object potential - James Lane (UofT)
        """
        #Cylindrical distance
        Rdist = _cylR(R,phi,self._orb.R(t),self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd,yd,zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz,
            R, phi, z)
        #Evaluate cylindrical radial force.
        RF = evaluateRforces(self._pot,Rdist,zd,t=t,use_physical=False)
        # Return phi force, negative of phi vector to evaluate location
        return -RF*R*(numpy.cos(phi)*yd-numpy.sin(phi)*xd)/Rdist

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
        #Cylindrical distance
        Rdist = _cylR(R,phi,self._orb.R(t),self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd,yd,zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz,
            R, phi, z)
        # Return the density
        return evaluateDensities(self._pot,Rdist,zd,t=t,use_physical=False)

def _cylR(R1,phi1,R2,phi2):
    return numpy.sqrt(R1**2.+R2**2.-2.*R1*R2*numpy.cos(phi1-phi2)) # Cosine law

def _cyldiff(R1,phi1,z1,R2,phi2,z2):
    dx = R1*numpy.cos(phi1)-R2*numpy.cos(phi2)
    dy = R1*numpy.sin(phi1)-R2*numpy.sin(phi2)
    dz = z1-z2
    return (dx,dy,dz)
