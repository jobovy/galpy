###############################################################################
#   NonInertialFrameForce: Class that implements the fictitious forces
#                          present when integrating orbits in a non-intertial
#                          frame
###############################################################################
import numpy
import numpy.linalg
import hashlib
from ..util import coords, conversion
from .DissipativeForce import DissipativeForce
class NonInertialFrameForce(DissipativeForce):
    """Class that implements the fictitious forces present when integrating 
    orbits in a non-intertial frame. Coordinates in the inertial frame 
    :math:`\mathbf{x}` and in the non-inertial frame :math:`\mathbf{r}` are
    related through rotation and linear motion as
    
    .. math::
    
        \mathbf{x} = \mathbf{R}\,\mathbf{r} + \mathbf{x}_0
        
    where :math:`\mathbf{R}` is a rotation matrix and :math:`\mathbf{x}_{\mathrm{CM}}`
    is the motion of the origin. The rotation matrix has angular frequencies
    :math:`\\boldsymbol{\Omega}` with time derivative :math:`\dot{\\boldsymbol{\Omega}}`;
    the latter is assumed to be constant. The motion of the origin can be any function
    of time.    
    This leads to the fictitious force
    
    .. math::
    
        \mathbf{F} = -\mathbf{R}^T\,\mathbf{a}_0 - \\boldsymbol{\Omega} \\times ( \\boldsymbol{\Omega} \\times \mathbf{r}) - \dot{\\boldsymbol{\Omega}} \\times \mathbf{r} -2\\boldsymbol{\Omega}\\times \dot{\mathbf{r}}
        
    where :math:`\mathbf{a}_0` is the acceleration of the origin. To avoid having 
    to specify the rotation matrix, we use :math:`\mathbf{R}^T\,\mathbf{a}_0` as the 
    acceleration input.
    """
    def __init__(self,amp=1.,Omega=1.,Omegadot=None,RTacm=None,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a NonInertialFrameForce

        INPUT:

           amp= (1.) amplitude to be applied to the potential (default: 1)

           Omega= (1.) Angular frequency of the rotation of the non-inertial frame in an inertial one; can be a vector (numpy array) [Omega_x,Omega_y,Omega_z] or a single value Omega_z (can be a Quantity)
           
           Omegadot= (None) Time derivative of the angular frequency of the non-intertial frame's rotation. Vector or scalar should match Omega input
           
           RTacm= (None) Acceleration vector a_cm (cartesian) of the center of mass of the non-intertial frame, transformed to the rotating frame as R^T a_cm where R^T is the inverse rotation matrix; constant or a function

        OUTPUT:

           (none)

        HISTORY:

           2022-03-02 - Started - Bovy (UofT)

        """
        DissipativeForce.__init__(self,amp=amp,ro=ro,vo=vo)
        self._Omega= conversion.parse_frequency(Omega,ro=self._ro,vo=self._vo)
        self._omegaz_only= len(numpy.atleast_1d(self._Omega)) == 1
        self._const_freq= Omegadot is None
        self._Omegadot= conversion.parse_frequency(Omegadot,ro=self._ro,vo=self._vo)
        self._lin_acc= not (RTacm is None)
        if not callable(RTacm):
            self._RTacm= lambda t, copy=RTacm: copy
        else: 
            self._RTacm= RTacm
        # Useful derived quantities
        self._Omega2= numpy.linalg.norm(self._Omega)**2.
        if not self._omegaz_only:
            self._Omega_for_cross= numpy.array([[0.,-self._Omega[2],self._Omega[1]],
                                                [self._Omega[2],0.,-self._Omega[0]],
                                                [-self._Omega[1],self._Omega[0],0.]])
            if not self._const_freq:
                self._Omegadot_for_cross= \
                    numpy.array([[0.,-self._Omegadot[2],self._Omegadot[1]],
                                [self._Omegadot[2],0.,-self._Omegadot[0]],
                                [-self._Omegadot[1],self._Omegadot[0],0.]])
        self._force_hash= None            
        self.hasC= False
        return None

    def _force(self,R,z,phi,t,v):
        """Internal function that computes the fictitious forces in rectangular
        coordinates"""
        new_hash= hashlib.md5(numpy.array([R,phi,z,v[0],v[1],v[2],t])).hexdigest()
        if new_hash == self._force_hash:
            return self._cached_force
        x,y,z= coords.cyl_to_rect(R,phi,z)
        vx,vy,vz= coords.cyl_to_rect_vec(v[0],v[1],v[2],phi)
        if self._const_freq:
            tOmega= self._Omega
            tOmega2= self._Omega2
        else:
            tOmega= self._Omega+self._Omegadot*t
            tOmega2= numpy.linalg.norm(tOmega)**2.            
        if self._omegaz_only:
            force= -2.*tOmega*numpy.array([-vy,vx,0.])\
                +tOmega2*numpy.array([x,y,0.])
        else:
            force= -2.*numpy.dot(self._Omega_for_cross,numpy.array([vx,vy,vz]))\
                +tOmega2*numpy.array([x,y,z])\
                -tOmega*numpy.dot(tOmega,numpy.array([x,y,z]))
            if not self._const_freq:
                force-= 2.*t*numpy.dot(self._Omegadot_for_cross,numpy.array([vx,vy,vz]))
        if not self._const_freq:
            force-= numpy.dot(self._Omegadot_for_cross,numpy.array([x,y,z]))           
        if self._lin_acc:
            force-= self._RTacm(t)
        self._force_hash= new_hash
        self._cached_force= force
        return force       

    def _Rforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this Force
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v= current velocity in cylindrical coordinates
        OUTPUT:
           the radial force
        HISTORY:
           2022-03-02 - Written - Bovy (UofT)
        """
        force= self._force(R,z,phi,t,v)
        return numpy.cos(phi)*force[0]+numpy.sin(phi)*force[1]

    def _phiforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this Force
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v= current velocity in cylindrical coordinates
        OUTPUT:
           the azimuthal force
        HISTORY:
           2016-06-09 - Written - Bovy (UofT)
        """
        force= self._force(R,z,phi,t,v)
        return R*(-numpy.sin(phi)*force[0]+numpy.cos(phi)*force[1])

    def _zforce(self,R,z,phi=0.,t=0.,v=None):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this Force
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
           v= current velocity in cylindrical coordinates
        OUTPUT:
           the vertical force
        HISTORY:
           2016-06-09 - Written - Bovy (UofT)
        """
        return self._force(R,z,phi,t,v)[2]
