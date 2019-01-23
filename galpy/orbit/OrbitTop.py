import warnings
import json
from random import choice
from string import ascii_lowercase
import math as m
import numpy as nu
import scipy
from scipy import interpolate
_APY_LOADED= True
try:
    from astropy import units, coordinates
except ImportError:
    _APY_LOADED= False
if _APY_LOADED:
    import astropy
    _APY3= astropy.__version__ > '3'
from galpy import actionAngle
from galpy.potential import PotentialError
import galpy.util.bovy_plot as plot
import galpy.util.bovy_coords as coords
from galpy.util.bovy_conversion import physical_conversion
from galpy.util import bovy_conversion, galpyWarning
from galpy.util import config
if int(scipy.__version__.split('.')[0]) < 1 and \
        int(scipy.__version__.split('.')[1]) < 15: #pragma: no cover
    _OLD_SCIPY= True
    _KWINTERP= {}  #for scipy version <1.15
else:
    _OLD_SCIPY= False
    _KWINTERP= {'ext':2}  #for scipy version >=1.15
class OrbitTop(object):
    """General class that holds orbits and integrates them"""
    def __init__(self,vxvv=None,vo=None,ro=None,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize an orbit instance

        INPUT:

           vxvv - initial condition

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

        """
        # If you change the way an Orbit object is setup, also change each of
        # the methods that return Orbits
        self.vxvv= vxvv
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= vo
            self._voSet= True
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            self._ro= ro
            self._roSet= True
        self._zo= zo
        self._solarmotion= solarmotion
        return None

    def turn_physical_off(self):
        """
        NAME:
           turn_physical_off
        PURPOSE:
           turn off automatic returning of outputs in physical units
        INPUT:
           (none)
        OUTPUT:
           (none)
        HISTORY:
           2014-06-17 - Written - Bovy (IAS)
        """
        self._roSet= False
        self._voSet= False
        return None

    def turn_physical_on(self,ro=None,vo=None):
        """
        NAME:
           turn_physical_on
        PURPOSE:
           turn on automatic returning of outputs in physical units
        INPUT:
           ro= reference distance (kpc)
           vo= reference velocity (km/s)
        OUTPUT:
           (none)
        HISTORY:
           2016-01-19 - Written - Bovy (UofT)
        """
        self._roSet= True
        self._voSet= True
        if not ro is None:
            self._ro= ro
        if not vo is None:
            self._vo= vo
        return None

    def integrate(self,t,pot,method='symplec4_c',dt=None):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - Potential instance or list of instances
        OUTPUT:
           (none) (get the actual orbit using self.getOrbit()
        HISTORY:
           2010-07-10
        """
        raise NotImplementedError

    def getOrbit(self):
        """
        NAME:
           getOrbit
        PURPOSE:
           return a previously calculated orbit
        INPUT:
           (none)
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self.orbit

    def getOrbit_dxdv(self):
        """
        NAME:
           getOrbit_dxdv
        PURPOSE:
           return a previously calculated orbit_dxdv
        INPUT:
           (none)
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self.orbit_dxdv[:,4:]

    @physical_conversion('time')
    def time(self,*args,**kwargs):
        """
        NAME:
           time
        PURPOSE:
           return the times at which the orbit is sampled
        INPUT:
           t - (default: integration times) time at which to get the time (for consistency reasons); default is to return the list of times at which the orbit is sampled
           ro= (Object-wide default) physical scale for distances to use to convert
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           t(t)
        HISTORY:
           2014-06-11 - Written - Bovy (IAS)
        """
        if len(args) == 0:
            try:
                return self.t
            except AttributeError:
                return 0.
        else: return args[0]

    @physical_conversion('position')
    def R(self,*args,**kwargs):
        """
        NAME:
           R
        PURPOSE:
           return cylindrical radius at time t
        INPUT:
           t - (optional) time at which to get the radius
           ro= (Object-wide default) physical scale for distances to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           R(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[0]
        else: return thiso[0,:]

    @physical_conversion('position')
    def r(self,*args,**kwargs):
        """
        NAME:
           r
        PURPOSE:
           return spherical radius at time t
        INPUT:
           t - (optional) time at which to get the radius
           ro= (Object-wide default) physical scale for distances to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           r(t)
        HISTORY:
           2016-04-19 - Written - Bovy (UofT)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return nu.sqrt(thiso[0]**2.+thiso[3]**2.)
        else: return nu.sqrt(thiso[0,:]**2.+thiso[3,:]**2.)

    @physical_conversion('velocity')
    def vR(self,*args,**kwargs):
        """
        NAME:
           vR
        PURPOSE:
           return radial velocity at time t
        INPUT:
           t - (optional) time at which to get the radial velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vR(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[1]
        else: return thiso[1,:]

    @physical_conversion('velocity')
    def vT(self,*args,**kwargs):
        """
        NAME:
           vT
        PURPOSE:
           return tangential velocity at time t
        INPUT:
           t - (optional) time at which to get the tangential velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vT(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[2]
        else: return thiso[2,:]

    @physical_conversion('position')
    def z(self,*args,**kwargs):
        """
        NAME:
           z
        PURPOSE:
           return vertical height
        INPUT:
           t - (optional) time at which to get the vertical height
           ro= (Object-wide default) physical scale for distances to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           z(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        if len(self.vxvv) < 5:
            raise AttributeError("linear and planar orbits do not have z()")
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[3]
        else: return thiso[3,:]

    @physical_conversion('velocity')
    def vz(self,*args,**kwargs):
        """
        NAME:
           vz
        PURPOSE:
           return vertical velocity
        INPUT:
           t - (optional) time at which to get the vertical velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vz(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        if len(self.vxvv) < 5:
            raise AttributeError("linear and planar orbits do not have vz()")
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[4]
        else: return thiso[4,:]
        
    @physical_conversion('angle')
    def phi(self,*args,**kwargs):
        """
        NAME:
           phi
        PURPOSE:
           return azimuth
        INPUT:
           t - (optional) time at which to get the azimuth
        OUTPUT:
           phi(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        if len(self.vxvv) != 4 and len(self.vxvv) != 6:
            raise AttributeError("orbit must track azimuth to use phi()")
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[-1]
        else: return thiso[-1,:]

    @physical_conversion('position')
    def x(self,*args,**kwargs):
        """
        NAME:
           x
        PURPOSE:
           return x
        INPUT:
           t - (optional) time at which to get x
           ro= (Object-wide default) physical scale for distances to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           x(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) == 2:
            return thiso[0,:]
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use x()")
        elif len(thiso[:,0]) == 4:
            return thiso[0,:]*nu.cos(thiso[3,:])
        else:
            return thiso[0,:]*nu.cos(thiso[5,:])

    @physical_conversion('position')
    def y(self,*args,**kwargs):
        """
        NAME:
           y
        PURPOSE:
           return y
        INPUT:
           t - (optional) time at which to get y
           ro= (Object-wide default) physical scale for distances to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           y(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use x()")
        elif len(thiso[:,0]) == 4:
            return thiso[0,:]*nu.sin(thiso[3,:])
        else:
            return thiso[0,:]*nu.sin(thiso[5,:])

    @physical_conversion('velocity')
    def vx(self,*args,**kwargs):
        """
        NAME:
           vx
        PURPOSE:
           return x velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vx(t)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) == 2:
            return thiso[1,:]
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use vx()")
        elif len(thiso[:,0]) == 4:
            theta= thiso[3,:]
        else:
            theta= thiso[5,:]
        return thiso[1,:]*nu.cos(theta)-thiso[2,:]*nu.sin(theta)

    @physical_conversion('velocity')
    def vy(self,*args,**kwargs):
        """
        NAME:
           vy
        PURPOSE:
           return y velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vy(t)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use vx()")
        elif len(thiso[:,0]) == 4:
            theta= thiso[3,:]
        else:
            theta= thiso[5,:]
        return thiso[2,:]*nu.cos(theta)+thiso[1,:]*nu.sin(theta)

    @physical_conversion('velocity')
    def vphi(self,*args,**kwargs):
        """
        NAME:
           vphi
        PURPOSE:
           return angular velocity
        INPUT:
           t - (optional) time at which to get the angular velocity
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output
        OUTPUT:
           vphi(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        return thiso[2,:]/thiso[0,:]

    @physical_conversion('angle_deg')
    def ra(self,*args,**kwargs):
        """
        NAME:
           ra
        PURPOSE:
           return the right ascension
        INPUT:
           t - (optional) time at which to get ra
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)
        OUTPUT:
           ra(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'ra')
        radec= self._radec(*args,**kwargs)
        return radec[:,0]

    @physical_conversion('angle_deg')
    def dec(self,*args,**kwargs):
        """
        NAME:
           dec
        PURPOSE:
           return the declination
        INPUT:
           t - (optional) time at which to get dec
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)
        OUTPUT:
           dec(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'dec')
        radec= self._radec(*args,**kwargs)
        return radec[:,1]

    @physical_conversion('angle_deg')
    def ll(self,*args,**kwargs):
        """
        NAME:
           ll
        PURPOSE:
           return Galactic longitude
        INPUT:
           t - (optional) time at which to get ll
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           l(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'ll')
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,0]

    @physical_conversion('angle_deg')
    def bb(self,*args,**kwargs):
        """
        NAME:
           bb
        PURPOSE:
           return Galactic latitude
        INPUT:
           t - (optional) time at which to get bb
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           b(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'bb')
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,1]

    @physical_conversion('position_kpc')
    def dist(self,*args,**kwargs):
        """
        NAME:
           dist
        PURPOSE:
           return distance from the observer in kpc
        INPUT:
           t - (optional) time at which to get dist
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           dist(t) in kpc
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'dist')
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,2].astype('float64')

    @physical_conversion('proper-motion_masyr')
    def pmra(self,*args,**kwargs):
        """
        NAME:
           pmra
        PURPOSE:
           return proper motion in right ascension (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmra
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)    
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           pm_ra(t) in mas / yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'pmra')
        _check_voSet(self,kwargs,'pmra')
        pmrapmdec= self._pmrapmdec(*args,**kwargs)
        return pmrapmdec[:,0]

    @physical_conversion('proper-motion_masyr')
    def pmdec(self,*args,**kwargs):
        """
        NAME:
           pmdec
        PURPOSE:
           return proper motion in declination (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmdec
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           pm_dec(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'pmdec')
        _check_voSet(self,kwargs,'pmdec')
        pmrapmdec= self._pmrapmdec(*args,**kwargs)
        return pmrapmdec[:,1]

    @physical_conversion('proper-motion_masyr')
    def pmll(self,*args,**kwargs):
        """
        NAME:
           pmll
        PURPOSE:
           return proper motion in Galactic longitude (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmll
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           pm_l(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'pmll')
        _check_voSet(self,kwargs,'pmll')
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,4]

    @physical_conversion('proper-motion_masyr')
    def pmbb(self,*args,**kwargs):
        """
        NAME:
           pmbb
        PURPOSE:
           return proper motion in Galactic latitude (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmbb
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           pm_b(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'pmbb')
        _check_voSet(self,kwargs,'pmbb')
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,5]

    @physical_conversion('velocity_kms')
    def vlos(self,*args,**kwargs):
        """
        NAME:
           vlos
        PURPOSE:
           return the line-of-sight velocity (in km/s)
        INPUT:
           t - (optional) time at which to get vlos
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           vlos(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'vlos')
        _check_voSet(self,kwargs,'vlos')
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,3]

    @physical_conversion('position_kpc')
    def helioX(self,*args,**kwargs):
        """
        NAME:
           helioX
        PURPOSE:
           return Heliocentric Galactic rectangular x-coordinate (aka "X")
        INPUT:
           t - (optional) time at which to get X
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           helioX(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'helioX')
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return X

    @physical_conversion('position_kpc')
    def helioY(self,*args,**kwargs):
        """
        NAME:
           helioY
        PURPOSE:
           return Heliocentric Galactic rectangular y-coordinate (aka "Y")
        INPUT:
           t - (optional) time at which to get Y
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           helioY(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'helioY')
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return Y

    @physical_conversion('position_kpc')
    def helioZ(self,*args,**kwargs):
        """
        NAME:
           helioZ
        PURPOSE:
           return Heliocentric Galactic rectangular z-coordinate (aka "Z")
        INPUT:
           t - (optional) time at which to get Z
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
        OUTPUT:
           helioZ(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'helioZ')
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return Z

    @physical_conversion('velocity_kms')
    def U(self,*args,**kwargs):
        """
        NAME:
           U
        PURPOSE:
           return Heliocentric Galactic rectangular x-velocity (aka "U")
        INPUT:
           t - (optional) time at which to get U
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           U(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'U')
        _check_voSet(self,kwargs,'U')
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return U

    @physical_conversion('velocity_kms')
    def V(self,*args,**kwargs):
        """
        NAME:
           V
        PURPOSE:
           return Heliocentric Galactic rectangular y-velocity (aka "V")
        INPUT:
           t - (optional) time at which to get U
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           V(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'V')
        _check_voSet(self,kwargs,'V')
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return V

    @physical_conversion('velocity_kms')
    def W(self,*args,**kwargs):
        """
        NAME:
           W
        PURPOSE:
           return Heliocentric Galactic rectangular z-velocity (aka "W")
        INPUT:
           t - (optional) time at which to get W
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)         
           vo= velocity in km/s corresponding to v=1. (default=Object-wide default)
        OUTPUT:
           W(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        _check_roSet(self,kwargs,'W')
        _check_voSet(self,kwargs,'W')
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return W

    def SkyCoord(self,*args,**kwargs):
        """
        NAME:
           SkyCoord
        PURPOSE:
           return the position as an astropy SkyCoord
        INPUT:
           t - (optional) time at which to get the position
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=Object-wide default)
                         OR Orbit object that corresponds to the orbit
                         of the observer
                         Y is ignored and always assumed to be zero
           ro= distance in kpc corresponding to R=1. (default=Object-wide default)
        OUTPUT:
           SkyCoord(t)
        HISTORY:
           2015-06-02 - Written - Bovy (IAS)
        """
        kwargs.pop('quantity',None) # rm useless keyword to no conflict later
        _check_roSet(self,kwargs,'SkyCoord')
        radec= self._radec(*args,**kwargs)
        tdist= self.dist(quantity=False,*args,**kwargs)
        if not _APY3: # pragma: no cover
            return coordinates.SkyCoord(radec[:,0]*units.degree,
                                        radec[:,1]*units.degree,
                                        distance=tdist*units.kpc,
                                        frame='icrs')
        pmrapmdec= self._pmrapmdec(*args,**kwargs)
        vlos= self._lbdvrpmllpmbb(*args,**kwargs)[:,3]
        # Also return the Galactocentric frame used
        v_sun= coordinates.CartesianDifferential(\
            nu.array([-self._solarmotion[0],
                       self._solarmotion[1]+self._vo,
                       self._solarmotion[2]])*units.km/units.s)
        return coordinates.SkyCoord(radec[:,0]*units.degree,
                                    radec[:,1]*units.degree,
                                    distance=tdist*units.kpc,
                                    pm_ra_cosdec=pmrapmdec[:,0]\
                                        *units.mas/units.yr,
                                    pm_dec=pmrapmdec[:,1]*units.mas/units.yr,
                                    radial_velocity=vlos*units.km/units.s,
                                    frame='icrs',
                                    galcen_distance=\
                                        nu.sqrt(self._ro**2.+self._zo**2.)\
                                        *units.kpc,
                                    z_sun=self._zo*units.kpc,
                                    galcen_v_sun=v_sun)

    def _radec(self,*args,**kwargs):
        """Calculate ra and dec"""
        lbd= self._lbd(*args,**kwargs)
        return coords.lb_to_radec(lbd[:,0],lbd[:,1],degree=True,epoch=None)

    def _pmrapmdec(self,*args,**kwargs):
        """Calculate pmra and pmdec"""
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return coords.pmllpmbb_to_pmrapmdec(lbdvrpmllpmbb[:,4],
                                            lbdvrpmllpmbb[:,5],
                                            lbdvrpmllpmbb[:,0],
                                            lbdvrpmllpmbb[:,1],degree=True,
                                            epoch=None)

    def _lbd(self,*args,**kwargs):
        """Calculate l,b, and d"""
        obs, ro, vo= self._parse_radec_kwargs(kwargs,dontpop=True)
        X,Y,Z= self._helioXYZ(*args,**kwargs)
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx:
            X[bad_indx]+= ro/10000.
        return coords.XYZ_to_lbd(X,Y,Z,degree=True)

    def _helioXYZ(self,*args,**kwargs):
        """Calculate heliocentric rectangular coordinates"""
        obs, ro, vo= self._parse_radec_kwargs(kwargs)
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6: #pragma: no cover
            raise AttributeError("orbit must track azimuth to use radeclbd functions")
        elif len(thiso[:,0]) == 4: #planarOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-nu.arctan2(obs[1],obs[0]),0.,
                    Xsun=nu.sqrt(obs[0]**2.+obs[1]**2.)/ro,
                    Zsun=obs[2]/ro,_extra_rot=False).T
            else: #Orbit instance
                obs.turn_physical_off()
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                        nu.zeros_like(thiso[0]),
                        Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                        nu.zeros_like(thiso[0]),
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
                obs.turn_physical_on()
        else: #FullOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-nu.arctan2(obs[1],obs[0]),
                    thiso[3,:],
                    Xsun=nu.sqrt(obs[0]**2.+obs[1]**2.)/ro,
                    Zsun=obs[2]/ro).T
            else: #Orbit instance
                obs.turn_physical_off()
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                        thiso[3,:],
                        Xsun=obs.R(*args,**kwargs),Zsun=0.).T
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                        thiso[3,:],
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs)).T
                obs.turn_physical_on()
        return (X*ro,Y*ro,Z*ro)

    def _lbdvrpmllpmbb(self,*args,**kwargs):
        """Calculate l,b,d,vr,pmll,pmbb"""
        obs, ro, vo= self._parse_radec_kwargs(kwargs,dontpop=True)
        X,Y,Z,vX,vY,vZ= self._XYZvxvyvz(*args,**kwargs)
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx:
            X[bad_indx]+= ro/10000.
        return coords.rectgal_to_sphergal(X,Y,Z,vX,vY,vZ,degree=True)

    def _XYZvxvyvz(self,*args,**kwargs):
        """Calculate X,Y,Z,U,V,W"""
        obs, ro, vo= self._parse_radec_kwargs(kwargs,vel=True)
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6: #pragma: no cover
            raise AttributeError("orbit must track azimuth to use radeclbduvw functions")
        elif len(thiso[:,0]) == 4: #planarOrbit
            if isinstance(obs,(nu.ndarray,list)):
                Xsun= nu.sqrt(obs[0]**2.+obs[1]**2.)
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[3,:]-nu.arctan2(obs[1],obs[0]),
                    nu.zeros_like(thiso[0]),
                    Xsun=Xsun/ro,Zsun=obs[2]/ro,_extra_rot=False).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],nu.zeros_like(thiso[0]),
                    thiso[3,:]-nu.arctan2(obs[1],obs[0]),
                    vsun=nu.array(# have to rotate
                        [obs[3]*obs[0]/Xsun+obs[4]*obs[1]/Xsun,
                         -obs[3]*obs[1]/Xsun+obs[4]*obs[0]/Xsun,
                         obs[5]])/vo,
                    Xsun=Xsun/ro,Zsun=obs[2]/ro,_extra_rot=False).T
            else: #Orbit instance
                obs.turn_physical_off()
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                        nu.zeros_like(thiso[0]),
                        Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                        thiso[1,:],thiso[2,:],nu.zeros_like(thiso[0]),
                        thiso[3,:]-obs.phi(*args,**kwargs),
                        vsun=nu.array([\
                                obs.vR(*args,**kwargs),obs.vT(*args,**kwargs),
                                0.]),
                        Xsun=obs.R(*args,**kwargs),Zsun=0.,_extra_rot=False).T
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[3,:]-obs.phi(*args,**kwargs),
                        nu.zeros_like(thiso[0]),
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                        thiso[1,:],thiso[2,:],nu.zeros_like(thiso[0]),
                        thiso[3,:]-obs.phi(*args,**kwargs),
                        vsun=nu.array([\
                                obs.vR(*args,**kwargs),
                                obs.vT(*args,**kwargs),
                                obs.vz(*args,**kwargs)]),
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs),_extra_rot=False).T
                obs.turn_physical_on()
        else: #FullOrbit
            if isinstance(obs,(nu.ndarray,list)):
                Xsun= nu.sqrt(obs[0]**2.+obs[1]**2.)
                X,Y,Z = coords.galcencyl_to_XYZ(\
                    thiso[0,:],thiso[5,:]-nu.arctan2(obs[1],obs[0]),thiso[3,:],
                    Xsun=Xsun/ro,Zsun=obs[2]/ro).T
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                    thiso[1,:],thiso[2,:],thiso[4,:],
                    thiso[5,:]-nu.arctan2(obs[1],obs[0]),
                    vsun=nu.array(# have to rotate
                        [obs[3]*obs[0]/Xsun+obs[4]*obs[1]/Xsun,
                         -obs[3]*obs[1]/Xsun+obs[4]*obs[0]/Xsun,
                         obs[5]])/vo,
                    Xsun=Xsun/ro,Zsun=obs[2]/ro).T
            else: #Orbit instance
                obs.turn_physical_off()
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                        thiso[3,:],
                        Xsun=obs.R(*args,**kwargs),Zsun=0.).T
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                        thiso[1,:],thiso[2,:],thiso[4,:],
                        thiso[5,:]-obs.phi(*args,**kwargs),
                        vsun=nu.array([\
                                obs.vR(*args,**kwargs),obs.vT(*args,**kwargs),
                                0.]),
                        Xsun=obs.R(*args,**kwargs),Zsun=0.).T
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(\
                        thiso[0,:],thiso[5,:]-obs.phi(*args,**kwargs),
                        thiso[3,:],
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs)).T
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(\
                        thiso[1,:],thiso[2,:],thiso[4,:],
                        thiso[5,:]-obs.phi(*args,**kwargs),
                        vsun=nu.array([\
                                obs.vR(*args,**kwargs),
                                obs.vT(*args,**kwargs),
                                obs.vz(*args,**kwargs)]),
                        Xsun=obs.R(*args,**kwargs),
                        Zsun=obs.z(*args,**kwargs)).T
                obs.turn_physical_on()
        return (X*ro,Y*ro,Z*ro,vX*vo,vY*vo,vZ*vo)

    def _parse_radec_kwargs(self,kwargs,vel=False,dontpop=False):
        if 'obs' in kwargs:
            obs= kwargs['obs']
            if not dontpop:
                kwargs.pop('obs')
            if isinstance(obs,(list,nu.ndarray)):
                if len(obs) == 2:
                    obs= [obs[0],obs[1],0.]
                elif len(obs) == 4:
                    obs= [obs[0],obs[1],0.,obs[2],obs[3],0.]
                for ii in range(len(obs)):
                    if _APY_LOADED and isinstance(obs[ii],units.Quantity):
                        if ii < 3:
                            obs[ii]= obs[ii].to(units.kpc).value
                        else:
                            obs[ii]= obs[ii].to(units.km/units.s).value
        else:
            if vel:
                obs= [self._ro,0.,self._zo,
                      self._solarmotion[0],self._solarmotion[1]+self._vo,
                      self._solarmotion[2]]
            else:
                obs= [self._ro,0.,self._zo]
        if 'ro' in kwargs:
            ro= kwargs['ro']
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            if not dontpop:
                kwargs.pop('ro')
        else:
            ro= self._ro
        if 'vo' in kwargs:
            vo= kwargs['vo']
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            if not dontpop:
                kwargs.pop('vo')
        else:
            vo= self._vo
        return (obs,ro,vo)

    def Jacobi(self,Omega,t=0.,pot=None):
        """
        NAME:
           Jacobi
        PURPOSE:
           calculate the Jacobi integral E - Omega L
        INPUT:
           Omega - pattern speed         
           t= time at which to evaluate the Jacobi integral
           Pot= potential instance or list of such instances
        OUTPUT:
           Jacobi integral
        HISTORY:
           2011-04-18 - Written - Bovy (NYU)
        """
        raise NotImplementedError("'Jacobi' for this Orbit type is not implemented yet")

    @physical_conversion('action')
    def L(self,*args,**kwargs):
        """
        NAME:
           L
        PURPOSE:
           calculate the angular momentum
        INPUT:
           (none)
        OUTPUT:
           angular momentum
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        #Make sure you are not using physical coordinates
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        Omega= kwargs.pop('Omega',None)
        t= kwargs.pop('t',None)
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) < 3:
            raise AttributeError("'linearOrbit has no angular momentum")
        elif len(thiso[:,0]) == 3 or len(thiso[:,0]) == 4:
            if Omega is None:
                out= thiso[0,:]*thiso[2,:]
            else:
                out= thiso[0,:]*(thiso[2,:]-Omega*t*thiso[0,:])
        elif len(thiso[:,0]) == 5:
            raise AttributeError("You must track the azimuth to get the angular momentum of a 3D Orbit")
        else: #len(thiso[:,0]) == 6
            vx= self.vx(*args,**kwargs)
            vy= self.vy(*args,**kwargs)
            vz= self.vz(*args,**kwargs)
            x= self.x(*args,**kwargs)
            y= self.y(*args,**kwargs)
            z= self.z(*args,**kwargs)
            out= nu.zeros((len(x),3))
            out[:,0]= y*vz-z*vy
            out[:,1]= z*vx-x*vz
            out[:,2]= x*vy-y*vx
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    @physical_conversion('action')
    def Lz(self,*args,**kwargs):
        """
        NAME:
           Lz
        PURPOSE:
           calculate the z component of the angular momentum
        INPUT:
           (none)
        OUTPUT:
           z-component of the angular momentum
        HISTORY:
           2018-08-29 - Written - Bovy (UofT)
        """
        tL= self.L(*args,use_physical=False)
        if len(tL.shape) == 2: # 3D orbit
            return tL[:,2]
        else: # 2D orbit
            return tL

    def _resetaA(self,pot=None,type=None):
        """
        NAME:
           _resetaA
        PURPOSE:
           re-set up an actionAngle module for this Orbit
           ONLY TO BE CALLED FROM WITHIN SETUPAA
        INPUT:
           pot - potential
        OUTPUT:
           True if reset happened, False otherwise
        HISTORY:
           2012-06-01 - Written - Bovy (IAS)
        """
        if (not pot is None and pot != self._aAPot) \
                or (not type is None and type != self._aAType):
            delattr(self,'_aA')
            return True
        else:
            pass #Already set up

    def _setupaA(self,pot=None,type='staeckel',**kwargs):
        """
        NAME:
           _setupaA
        PURPOSE:
           set up an actionAngle module for this Orbit
        INPUT:
           pot - potential
           type= ('staeckel') type of actionAngle module to use
              1) 'adiabatic'
              2) 'staeckel'
              3) 'isochroneApprox'
              4) 'spherical'
        OUTPUT:
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
           2013-11-27 - Re-written in terms of new actionAngle modules - Bovy (IAS)
           2017-12-25 - Changed default method to 'staeckel' and automatic delta estimation - Bovy (UofT)
        """
        if hasattr(self,'_aA'):
            if not self._resetaA(pot=pot,type=type): return None
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        self._aAPot= pot
        self._aAType= type
        #Setup
        if self._aAType.lower() == 'adiabatic':
            self._aA= actionAngle.actionAngleAdiabatic(pot=self._aAPot,
                                                       **kwargs)
        elif self._aAType.lower() == 'staeckel':
            try:
                delta= \
                    kwargs.pop('delta',
                               actionAngle.estimateDeltaStaeckel(\
                        self._aAPot,self.R(use_physical=False),
                        self.z(use_physical=False)+(nu.fabs(self.z(use_physical=False)) < 1e-8) * (2.*(self.z(use_physical=False) >= 0)-1.)*1e-10)) # try to make sure this is not 0
            except PotentialError as e:
                if 'deriv' in str(e):
                    raise PotentialError('Automagic calculation of delta parameter for Staeckel approximation failed because the necessary second derivatives of the given potential are not implemented; set delta= explicitly')
                elif 'non-axi' in str(e):
                    raise PotentialError('Automagic calculation of delta parameter for Staeckel approximation failed because the given potential is not axisymmetric; pass an axisymmetric potential instead')
                else: #pragma: no cover
                    raise
            if delta < 1e-6:
                self._setupaA(pot=pot,type='spherical')
            else:
                self._aA= actionAngle.actionAngleStaeckel(pot=self._aAPot,
                                                          delta=delta,
                                                          **kwargs)
        elif self._aAType.lower() == 'isochroneapprox':
            from galpy.actionAngle import actionAngleIsochroneApprox
            self._aA= actionAngleIsochroneApprox(pot=self._aAPot,
                                                 **kwargs)
        elif self._aAType.lower() == 'spherical':
            self._aA= actionAngle.actionAngleSpherical(pot=self._aAPot,
                                                       **kwargs)
        return None

    def _xw(self,*args,**kwargs): #pragma: no cover
        """
        NAME:
           xw
        PURPOSE:
           return the Fourier transform of xx
        INPUT:
           t - (optional) time at which to get xw
        OUTPUT:
           xw(t)
        HISTORY:
           2011-01-04 - Written - Bovy (NYU)
        """
        #BOVY: REPLACE WITH CALCULATION FUNCTION
        x= self.x(self.t)
        xw= nu.fft.fft(x)#-nu.mean(x))
        xw= nu.abs(xw[0:len(xw)/2])*(self.t[1]-self.t[0])/(self.t[-1]-self.t[0])
        return xw

    def _plotxw(self,*args,**kwargs): #pragma: no cover
        """
        NAME:
           plotxw
        PURPOSE:
           plot the spectrum of x
        INPUT:
           bovy_plot.bovy_plot args and kwargs
        OUTPUT:
           x(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        xw= self.xw()
        #BOVY: CHECK THAT THIS IS CORRECT
        plot.bovy_plot(2.*m.pi*nu.fft.fftfreq(len(self.t),
                                              d=(self.t[1]-self.t[0]))\
                           [0:len(xw)],
                       xw,*args,**kwargs)

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           return the orbit vector at time t
        INPUT:
           t - desired time
        OUTPUT:
           [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on the orbit
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if len(args) == 0:
            return nu.array(self.vxvv)
        else:
            t= args[0]
        # Parse t
        if _APY_LOADED and isinstance(t,units.Quantity):
            t= t.to(units.Gyr).value\
                /bovy_conversion.time_in_Gyr(self._vo,self._ro)
        elif hasattr(self,'_integrate_t_asQuantity') \
                    and self._integrate_t_asQuantity \
                    and not nu.all(t == self.t):
            warnings.warn("You specified integration times as a Quantity, but are evaluating at times not specified as a Quantity; assuming that time given is in natural (internal) units (multiply time by unit to get output at physical time)",galpyWarning)
        if isinstance(t,(int,float)) and hasattr(self,'t') \
                and t in list(self.t):
            return self.orbit[list(self.t).index(t),:]
        else:
            if isinstance(t,(int,float)): 
                nt= 1
                t= [t]
            else: 
                nt= len(t)
            dim= len(self.vxvv)
            try:
                self._setupOrbitInterp()
            except:
                out= nu.zeros((dim,nt))
                for jj in range(nt):
                    try:
                        indx= list(self.t).index(t[jj])
                    except ValueError:
                        raise LookupError("Orbit interpolaton failed; integrate on finer grid")
                    for ii in range(dim):
                        out[ii,jj]= self.orbit[indx,ii]
                return out #should always have nt > 1, bc otherwise covered by above
            out= []
            if _OLD_SCIPY and not isinstance(self._orbInterp[0],_fakeInterp) \
                    and nu.any((nu.array(t) < self._orbInterp[0]._data[3])\
                               +(nu.array(t) > self._orbInterp[0]._data[4])): #pragma: no cover
                raise ValueError("One or more requested time is not within the integrated range")
            if dim == 4 or dim == 6:
                #Unpack interpolated x and y to R and phi
                x= self._orbInterp[0](t)
                y= self._orbInterp[-1](t)
                R= nu.sqrt(x*x+y*y)
                phi= nu.arctan2(y,x) % (2.*nu.pi)
                for ii in range(dim):
                    if ii == 0:
                        out.append(R) 
                    elif ii == dim-1:
                        out.append(phi) 
                    else:
                        out.append(self._orbInterp[ii](t))
            else:
                for ii in range(dim):
                    out.append(self._orbInterp[ii](t))
            if nt == 1:
                return nu.array(out).reshape(dim)
            else:
                return nu.array(out).reshape((dim,nt))

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot aspects of an Orbit
        INPUT:
           bovy_plot args and kwargs
           ro= (Object-wide default) physical scale for distances to use to convert
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output

           +kwargs for ra,dec,ll,bb, etc. functions
        OUTPUT:
           plot
        HISTORY:
           2010-07-26 - Written - Bovy (NYU)
           2010-09-22 - Adapted to more general framework - Bovy (NYU)
           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)
           2014-06-11 - Support for plotting in physical coordinates - Bovy (IAS)
           2017-11-28 - Allow arbitrary functions of time to be plotted - Bovy (UofT)
        """
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':r'$t\ (\mathrm{Gyr})$','R':r'$R\ (\mathrm{kpc})$',
                        'vR':r'$v_R\ (\mathrm{km\,s}^{-1})$',
                        'vT':r'$v_T\ (\mathrm{km\,s}^{-1})$',
                        'z':r'$z\ (\mathrm{kpc})$',
                        'vz':r'$v_z\ (\mathrm{km\,s}^{-1})$','phi':r'$\phi$',
                        'r':r'$r\ (\mathrm{kpc})$',
                        'x':r'$x\ (\mathrm{kpc})$','y':r'$y\ (\mathrm{kpc})$',
                        'vx':r'$v_x\ (\mathrm{km\,s}^{-1})$',
                        'vy':r'$v_y\ (\mathrm{km\,s}^{-1})$',
                        'E':r'$E\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Ez':r'$E_z\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'ER':r'$E_R\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Enorm':r'$E(t)/E(0.)$',
                        'Eznorm':r'$E_z(t)/E_z(0.)$',
                        'ERnorm':r'$E_R(t)/E_R(0.)$',
                        'Jacobi':r'$E-\Omega_p\,L\,(\mathrm{km}^2\,\mathrm{s}^{-2})$',
                        'Jacobinorm':r'$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$'}
        else:
            labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                        'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                        'r':r'$r$',
                        'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$',
                        'E':r'$E$','Enorm':r'$E(t)/E(0.)$',
                        'Ez':r'$E_z$','Eznorm':r'$E_z(t)/E_z(0.)$',
                        'ER':r'$E_R$','ERnorm':r'$E_R(t)/E_R(0.)$',
                        'Jacobi':r'$E-\Omega_p\,L$',
                        'Jacobinorm':r'$(E-\Omega_p\,L)(t)/(E-\Omega_p\,L)(0)$'}
        labeldict.update({'ra':r'$\alpha\ (\mathrm{deg})$',
                          'dec':r'$\delta\ (\mathrm{deg})$',
                          'll':r'$l\ (\mathrm{deg})$',
                          'bb':r'$b\ (\mathrm{deg})$',
                          'dist':r'$d\ (\mathrm{kpc})$',
                          'pmra':r'$\mu_\alpha\ (\mathrm{mas\,yr}^{-1})$',
                          'pmdec':r'$\mu_\delta\ (\mathrm{mas\,yr}^{-1})$',
                          'pmll':r'$\mu_l\ (\mathrm{mas\,yr}^{-1})$',
                          'pmbb':r'$\mu_b\ (\mathrm{mas\,yr}^{-1})$',
                          'vlos':r'$v_\mathrm{los}\ (\mathrm{km\,s}^{-1})$',
                          'helioX':r'$X\ (\mathrm{kpc})$',
                          'helioY':r'$Y\ (\mathrm{kpc})$',
                          'helioZ':r'$Z\ (\mathrm{kpc})$',
                          'U':r'$U\ (\mathrm{km\,s}^{-1})$',
                          'V':r'$V\ (\mathrm{km\,s}^{-1})$',
                          'W':r'$W\ (\mathrm{km\,s}^{-1})$'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs:
            if len(self.vxvv) == 3:
                d1= 'R'
                d2= 'vR'
            elif len(self.vxvv) == 4:
                d1= 'x'
                d2= 'y'
            elif len(self.vxvv) == 2:
                d1= 'x'
                d2= 'vx'
            elif len(self.vxvv) == 5 or len(self.vxvv) == 6:
                d1= 'R'
                d2= 'z'
        elif not 'd1' in kwargs:
            d2=  kwargs.pop('d2')
            d1= 't'
        elif not 'd2' in kwargs:
            d1= kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
        #Get x and y
        if callable(d1):
            x= d1(self.t)
        elif d1 == 't':
            x= self.time(self.t,**kwargs)
        elif d1 == 'R':
            x= self.R(self.t,**kwargs)
        elif d1 == 'r':
            x= nu.sqrt(self.R(self.t,**kwargs)**2.
                       +self.z(self.t,**kwargs)**2.)
        elif d1 == 'z':
            x= self.z(self.t,**kwargs)
        elif d1 == 'vz':
            x= self.vz(self.t,**kwargs)
        elif d1 == 'vR':
            x= self.vR(self.t,**kwargs)
        elif d1 == 'vT':
            x= self.vT(self.t,**kwargs)
        elif d1 == 'x':
            x= self.x(self.t,**kwargs)
        elif d1 == 'y':
            x= self.y(self.t,**kwargs)
        elif d1 == 'vx':
            x= self.vx(self.t,**kwargs)
        elif d1 == 'vy':
            x= self.vy(self.t,**kwargs)
        elif d1 == 'phi':
            x= self.phi(self.t,**kwargs)
        elif d1.lower() == 'ra':
            x= self.ra(self.t,**kwargs)
        elif d1.lower() == 'dec':
            x= self.dec(self.t,**kwargs)
        elif d1 == 'll':
            x= self.ll(self.t,**kwargs)
        elif d1 == 'bb':
            x= self.bb(self.t,**kwargs)
        elif d1 == 'dist':
            x= self.dist(self.t,**kwargs)
        elif d1 == 'pmra':
            x= self.pmra(self.t,**kwargs)
        elif d1 == 'pmdec':
            x= self.pmdec(self.t,**kwargs)
        elif d1 == 'pmll':
            x= self.pmll(self.t,**kwargs)
        elif d1 == 'pmbb':
            x= self.pmbb(self.t,**kwargs)
        elif d1 == 'vlos':
            x= self.vlos(self.t,**kwargs)
        elif d1 == 'helioX':
            x= self.helioX(self.t,**kwargs)
        elif d1 == 'helioY':
            x= self.helioY(self.t,**kwargs)
        elif d1 == 'helioZ':
            x= self.helioZ(self.t,**kwargs)
        elif d1 == 'U':
            x= self.U(self.t,**kwargs)
        elif d1 == 'V':
            x= self.V(self.t,**kwargs)
        elif d1 == 'W':
            x= self.W(self.t,**kwargs)
        elif d1 == 'E':
            x= self.E(self.t,**kwargs)
        elif d1 == 'Enorm':
            x= self.E(self.t,**kwargs)/self.E(0.,**kwargs)
        elif d1 == 'Ez':
            x= self.Ez(self.t,**kwargs)
        elif d1 == 'Eznorm':
            x= self.Ez(self.t,**kwargs)/self.Ez(0.,**kwargs)
        elif d1 == 'ER':
            x= self.ER(self.t,**kwargs)
        elif d1 == 'ERnorm':
            x= self.ER(self.t,**kwargs)/self.ER(0.,**kwargs)
        elif d1 == 'Jacobi':
            x= self.Jacobi(self.t,**kwargs)
        elif d1 == 'Jacobinorm':
            x= self.Jacobi(self.t,**kwargs)/self.Jacobi(0.,**kwargs)
        if callable(d2):
            y= d2(self.t)
        elif d2 == 't':
            y= self.time(self.t,**kwargs)
        elif d2 == 'R':
            y= self.R(self.t,**kwargs)
        elif d2 == 'r':
            y= nu.sqrt(self.R(self.t,**kwargs)**2.
                       +self.z(self.t,**kwargs)**2.)
        elif d2 == 'z':
            y= self.z(self.t,**kwargs)
        elif d2 == 'vz':
            y= self.vz(self.t,**kwargs)
        elif d2 == 'vR':
            y= self.vR(self.t,**kwargs)
        elif d2 == 'vT':
            y= self.vT(self.t,**kwargs)
        elif d2 == 'x':
            y= self.x(self.t,**kwargs)
        elif d2 == 'y':
            y= self.y(self.t,**kwargs)
        elif d2 == 'vx':
            y= self.vx(self.t,**kwargs)
        elif d2 == 'vy':
            y= self.vy(self.t,**kwargs)
        elif d2 == 'phi':
            y= self.phi(self.t,**kwargs)
        elif d2.lower() == 'ra':
            y= self.ra(self.t,**kwargs)
        elif d2.lower() == 'dec':
            y= self.dec(self.t,**kwargs)
        elif d2 == 'll':
            y= self.ll(self.t,**kwargs)
        elif d2 == 'bb':
            y= self.bb(self.t,**kwargs)
        elif d2 == 'dist':
            y= self.dist(self.t,**kwargs)
        elif d2 == 'pmra':
            y= self.pmra(self.t,**kwargs)
        elif d2 == 'pmdec':
            y= self.pmdec(self.t,**kwargs)
        elif d2 == 'pmll':
            y= self.pmll(self.t,**kwargs)
        elif d2 == 'pmbb':
            y= self.pmbb(self.t,**kwargs)
        elif d2 == 'vlos':
            y= self.vlos(self.t,**kwargs)
        elif d2 == 'helioX':
            y= self.helioX(self.t,**kwargs)
        elif d2 == 'helioY':
            y= self.helioY(self.t,**kwargs)
        elif d2 == 'helioZ':
            y= self.helioZ(self.t,**kwargs)
        elif d2 == 'U':
            y= self.U(self.t,**kwargs)
        elif d2 == 'V':
            y= self.V(self.t,**kwargs)
        elif d2 == 'W':
            y= self.W(self.t,**kwargs)
        elif d2 == 'E':
            y= self.E(self.t,**kwargs)
        elif d2 == 'Enorm':
            y= self.E(self.t,**kwargs)/self.E(0.,**kwargs)
        elif d2 == 'Ez':
            y= self.Ez(self.t,**kwargs)
        elif d2 == 'Eznorm':
            y= self.Ez(self.t,**kwargs)/self.Ez(0.,**kwargs)
        elif d2 == 'ER':
            y= self.ER(self.t,**kwargs)
        elif d2 == 'ERnorm':
            y= self.ER(self.t,**kwargs)/self.ER(0.,**kwargs)
        elif d2 == 'Jacobi':
            y= self.Jacobi(self.t,**kwargs)
        elif d2 == 'Jacobinorm':
            y= self.Jacobi(self.t,**kwargs)/self.Jacobi(0.,**kwargs)
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('pot',None)
        kwargs.pop('OmegaP',None)
        kwargs.pop('quantity',None)
        #Plot
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= labeldict.get(d1,'\mathrm{No\ xlabel\ specified}')
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= labeldict.get(d2,'\mathrm{No\ ylabel\ specified}')
        return plot.bovy_plot(x,y,*args,**kwargs)

    def plot3d(self,*args,**kwargs):
        """
        NAME:
           plot3d
        PURPOSE:
           plot 3D aspects of an Orbit
        INPUT:
           ro= (Object-wide default) physical scale for distances to use to convert
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output

           bovy_plot args and kwargs
        OUTPUT:
           plot
        HISTORY:
           2010-07-26 - Written - Bovy (NYU)
           2010-09-22 - Adapted to more general framework - Bovy (NYU)
           2010-01-08 - Adapted to 3D - Bovy (NYU)
           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)
           2014-06-11 - Support for plotting in physical coordinates - Bovy (IAS)
           2017-11-28 - Allow arbitrary functions of time to be plotted - Bovy (UofT)
        """
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':r'$t\ (\mathrm{Gyr})$','R':r'$R\ (\mathrm{kpc})$',
                        'vR':r'$v_R\ (\mathrm{km\,s}^{-1})$',
                        'vT':r'$v_T\ (\mathrm{km\,s}^{-1})$',
                        'z':r'$z\ (\mathrm{kpc})$',
                        'vz':r'$v_z\ (\mathrm{km\,s}^{-1})$','phi':r'$\phi$',
                        'r':r'$r\ (\mathrm{kpc})$',
                        'x':r'$x\ (\mathrm{kpc})$','y':r'$y\ (\mathrm{kpc})$',
                        'vx':r'$v_x\ (\mathrm{km\,s}^{-1})$',
                        'vy':r'$v_y\ (\mathrm{km\,s}^{-1})$'}
        else:
            labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                        'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                        'r':r'$r$','x':r'$x$','y':r'$y$',
                        'vx':r'$v_x$','vy':r'$v_y$'}
        labeldict.update({'ra':r'$\alpha\ (\mathrm{deg})$',
                          'dec':r'$\delta\ (\mathrm{deg})$',
                          'll':r'$l\ (\mathrm{deg})$',
                          'bb':r'$b\ (\mathrm{deg})$',
                          'dist':r'$d\ (\mathrm{kpc})$',
                          'pmra':r'$\mu_\alpha\ (\mathrm{mas\,yr}^{-1})$',
                          'pmdec':r'$\mu_\delta\ (\mathrm{mas\,yr}^{-1})$',
                          'pmll':r'$\mu_l\ (\mathrm{mas\,yr}^{-1})$',
                          'pmbb':r'$\mu_b\ (\mathrm{mas\,yr}^{-1})$',
                          'vlos':r'$v_\mathrm{los}\ (\mathrm{km\,s}^{-1})$',
                          'helioX':r'$X\ (\mathrm{kpc})$',
                          'helioY':r'$Y\ (\mathrm{kpc})$',
                          'helioZ':r'$Z\ (\mathrm{kpc})$',
                          'U':r'$U\ (\mathrm{km\,s}^{-1})$',
                          'V':r'$V\ (\mathrm{km\,s}^{-1})$',
                          'W':r'$W\ (\mathrm{km\,s}^{-1})$'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs and not 'd3' in kwargs:
            if len(self.vxvv) == 3:
                d1= 'R'
                d2= 'vR'
                d3= 'vT'
            elif len(self.vxvv) == 4:
                d1= 'x'
                d2= 'y'
                d3= 'vR'
            elif len(self.vxvv) == 2:
                raise AttributeError("Cannot plot 3D aspects of 1D orbits")
            elif len(self.vxvv) == 5:
                d1= 'R'
                d2= 'vR'
                d3= 'z'
            elif len(self.vxvv) == 6:
                d1= 'x'
                d2= 'y'
                d3= 'z'
        elif not ('d1' in kwargs and 'd2' in kwargs and 'd3' in kwargs):
            raise AttributeError("Please provide 'd1', 'd2', and 'd3'")
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
            d3= kwargs.pop('d3')
        #Get x, y, and z
        if callable(d1):
            x= d1(self.t)
        elif d1 == 't':
            x= self.time(self.t,**kwargs)
        elif d1 == 'R':
            x= self.R(self.t,**kwargs)
        elif d1 == 'r':
            x= nu.sqrt(self.R(self.t,**kwargs)**2.
                       +self.z(self.t,**kwargs)**2.)
        elif d1 == 'z':
            x= self.z(self.t,**kwargs)
        elif d1 == 'vz':
            x= self.vz(self.t,**kwargs)
        elif d1 == 'vR':
            x= self.vR(self.t,**kwargs)
        elif d1 == 'vT':
            x= self.vT(self.t,**kwargs)
        elif d1 == 'x':
            x= self.x(self.t,**kwargs)
        elif d1 == 'y':
            x= self.y(self.t,**kwargs)
        elif d1 == 'vx':
            x= self.vx(self.t,**kwargs)
        elif d1 == 'vy':
            x= self.vy(self.t,**kwargs)
        elif d1 == 'phi':
            x= self.phi(self.t,**kwargs)
        elif d1.lower() == 'ra':
            x= self.ra(self.t,**kwargs)
        elif d1.lower() == 'dec':
            x= self.dec(self.t,**kwargs)
        elif d1 == 'll':
            x= self.ll(self.t,**kwargs)
        elif d1 == 'bb':
            x= self.bb(self.t,**kwargs)
        elif d1 == 'dist':
            x= self.dist(self.t,**kwargs)
        elif d1 == 'pmra':
            x= self.pmra(self.t,**kwargs)
        elif d1 == 'pmdec':
            x= self.pmdec(self.t,**kwargs)
        elif d1 == 'pmll':
            x= self.pmll(self.t,**kwargs)
        elif d1 == 'pmbb':
            x= self.pmbb(self.t,**kwargs)
        elif d1 == 'vlos':
            x= self.vlos(self.t,**kwargs)
        elif d1 == 'helioX':
            x= self.helioX(self.t,**kwargs)
        elif d1 == 'helioY':
            x= self.helioY(self.t,**kwargs)
        elif d1 == 'helioZ':
            x= self.helioZ(self.t,**kwargs)
        elif d1 == 'U':
            x= self.U(self.t,**kwargs)
        elif d1 == 'V':
            x= self.V(self.t,**kwargs)
        elif d1 == 'W':
            x= self.W(self.t,**kwargs)
        if callable(d2):
            y= d2(self.t)
        elif d2 == 't':
            y= self.time(self.t,**kwargs)
        elif d2 == 'R':
            y= self.R(self.t,**kwargs)
        elif d2 == 'r':
            y= nu.sqrt(self.R(self.t,**kwargs)**2.
                       +self.z(self.t,**kwargs)**2.)
        elif d2 == 'z':
            y= self.z(self.t,**kwargs)
        elif d2 == 'vz':
            y= self.vz(self.t,**kwargs)
        elif d2 == 'vR':
            y= self.vR(self.t,**kwargs)
        elif d2 == 'vT':
            y= self.vT(self.t,**kwargs)
        elif d2 == 'x':
            y= self.x(self.t,**kwargs)
        elif d2 == 'y':
            y= self.y(self.t,**kwargs)
        elif d2 == 'vx':
            y= self.vx(self.t,**kwargs)
        elif d2 == 'vy':
            y= self.vy(self.t,**kwargs)
        elif d2 == 'phi':
            y= self.phi(self.t,**kwargs)
        elif d2.lower() == 'ra':
            y= self.ra(self.t,**kwargs)
        elif d2.lower() == 'dec':
            y= self.dec(self.t,**kwargs)
        elif d2 == 'll':
            y= self.ll(self.t,**kwargs)
        elif d2 == 'bb':
            y= self.bb(self.t,**kwargs)
        elif d2 == 'dist':
            y= self.dist(self.t,**kwargs)
        elif d2 == 'pmra':
            y= self.pmra(self.t,**kwargs)
        elif d2 == 'pmdec':
            y= self.pmdec(self.t,**kwargs)
        elif d2 == 'pmll':
            y= self.pmll(self.t,**kwargs)
        elif d2 == 'pmbb':
            y= self.pmbb(self.t,**kwargs)
        elif d2 == 'vlos':
            y= self.vlos(self.t,**kwargs)
        elif d2 == 'helioX':
            y= self.helioX(self.t,**kwargs)
        elif d2 == 'helioY':
            y= self.helioY(self.t,**kwargs)
        elif d2 == 'helioZ':
            y= self.helioZ(self.t,**kwargs)
        elif d2 == 'U':
            y= self.U(self.t,**kwargs)
        elif d2 == 'V':
            y= self.V(self.t,**kwargs)
        elif d2 == 'W':
            y= self.W(self.t,**kwargs)
        if callable(d3):
            z= d3(self.t)
        elif d3 == 't':
            z= self.time(self.t,**kwargs)
        elif d3 == 'R':
            z= self.R(self.t,**kwargs)
        elif d3 == 'r':
            z= nu.sqrt(self.R(self.t,**kwargs)**2.
                       +self.z(self.t,**kwargs)**2.)
        elif d3 == 'z':
            z= self.z(self.t,**kwargs)
        elif d3 == 'vz':
            z= self.vz(self.t,**kwargs)
        elif d3 == 'vR':
            z= self.vR(self.t,**kwargs)
        elif d3 == 'vT':
            z= self.vT(self.t,**kwargs)
        elif d3 == 'x':
            z= self.x(self.t,**kwargs)
        elif d3 == 'y':
            z= self.y(self.t,**kwargs)
        elif d3 == 'vx':
            z= self.vx(self.t,**kwargs)
        elif d3 == 'vy':
            z= self.vy(self.t,**kwargs)
        elif d3 == 'phi':
            z= self.phi(self.t,**kwargs)
        elif d3.lower() == 'ra':
            z= self.ra(self.t,**kwargs)
        elif d3.lower() == 'dec':
            z= self.dec(self.t,**kwargs)
        elif d3 == 'll':
            z= self.ll(self.t,**kwargs)
        elif d3 == 'bb':
            z= self.bb(self.t,**kwargs)
        elif d3 == 'dist':
            z= self.dist(self.t,**kwargs)
        elif d3 == 'pmra':
            z= self.pmra(self.t,**kwargs)
        elif d3 == 'pmdec':
            z= self.pmdec(self.t,**kwargs)
        elif d3 == 'pmll':
            z= self.pmll(self.t,**kwargs)
        elif d3 == 'pmbb':
            z= self.pmbb(self.t,**kwargs)
        elif d3 == 'vlos':
            z= self.vlos(self.t,**kwargs)
        elif d3 == 'helioX':
            z= self.helioX(self.t,**kwargs)
        elif d3 == 'helioY':
            z= self.helioY(self.t,**kwargs)
        elif d3 == 'helioZ':
            z= self.helioZ(self.t,**kwargs)
        elif d3 == 'U':
            z= self.U(self.t,**kwargs)
        elif d3 == 'V':
            z= self.V(self.t,**kwargs)
        elif d3 == 'W':
            z= self.W(self.t,**kwargs)
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('quantity',None)
        #Plot
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= labeldict.get(d1,'\mathrm{No\ xlabel\ specified}')
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= labeldict.get(d2,'\mathrm{No\ ylabel\ specified}')
        if not 'zlabel' in kwargs:
            kwargs['zlabel']= labeldict.get(d3,'\mathrm{No\ zlabel\ specified}')
        return plot.bovy_plot3d(x,y,z,*args,**kwargs)

    def plotR(self,*args,**kwargs):
        """
        NAME:
           plotR
        PURPOSE:
           plot R(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'R'
        return self.plot(*args,**kwargs)

    def plotz(self,*args,**kwargs):
        """
        NAME:
           plotz
        PURPOSE:
           plot z(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'z'
        return self.plot(*args,**kwargs)

    def plotx(self,*args,**kwargs):
        """
        NAME:
           plotx
        PURPOSE:
           plot x(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'x'
        return self.plot(*args,**kwargs)

    def plotvx(self,*args,**kwargs):
        """
        NAME:
           plotvx
        PURPOSE:
           plot vx(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vx'
        return self.plot(*args,**kwargs)

    def ploty(self,*args,**kwargs):
        """
        NAME:
           ploty
        PURPOSE:
           plot y(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'y'
        return self.plot(*args,**kwargs)

    def plotvy(self,*args,**kwargs):
        """
        NAME:
           plotvy
        PURPOSE:
           plot vy(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vy'
        return self.plot(*args,**kwargs)

    def plotvR(self,*args,**kwargs):
        """
        NAME:
           plotvR
        PURPOSE:
           plot vR(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vR'
        return self.plot(*args,**kwargs)

    def plotvT(self,*args,**kwargs):
        """
        NAME:
           plotvT
        PURPOSE:
           plot vT(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vT'
        return self.plot(*args,**kwargs)
        
    def plotphi(self,*args,**kwargs):
        """
        NAME:
           plotphi
        PURPOSE:
           plot \phi(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'phi'
        return self.plot(*args,**kwargs)

    def plotvz(self,*args,**kwargs):
        """
        NAME:
           plotvz
        PURPOSE:
           plot vz(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        kwargs['d2']= 'vz'
        return self.plot(*args,**kwargs)
        
    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2014-06-16 - Written - Bovy (IAS)
        """
        if kwargs.pop('normed',False):
            kwargs['d2']= 'Enorm'
        else:
            kwargs['d2']= 'E'
        return self.plot(*args,**kwargs)
        
    def plotJacobi(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot Jacobi(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2014-06-16 - Written - Bovy (IAS)
        """
        if kwargs.pop('normed',False):
            kwargs['d2']= 'Jacobinorm'
        else:
            kwargs['d2']= 'Jacobi'
        return self.plot(*args,**kwargs)
        
    def _setupOrbitInterp(self):
        if not hasattr(self,"_orbInterp"):
            # First check that times increase
            if hasattr(self,"t"): #Orbit has been integrated
                if self.t[1] < self.t[0]: #must be backward
                    sindx= nu.argsort(self.t)
                    # sort
                    self.t= self.t[sindx]
                    self.orbit= self.orbit[sindx]
                    usindx= nu.argsort(sindx) # to later unsort
            orbInterp= []
            for ii in range(len(self.vxvv)):
                if (len(self.vxvv) == 4 or len(self.vxvv) == 6) and ii == 0:
                   #Interpolate x and y rather than R and phi to avoid issues w/ phase wrapping
                    if not hasattr(self,"t"): #Orbit has not been integrated
                        orbInterp.append(_fakeInterp(self.vxvv[0]*nu.cos(self.vxvv[-1])))
                    else:
                        orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                                self.t,self.orbit[:,0]*nu.cos(self.orbit[:,-1]),
                                **_KWINTERP))
                elif (len(self.vxvv) == 4 or len(self.vxvv) == 6) and \
                        ii == len(self.vxvv)-1:
                    if not hasattr(self,"t"): #Orbit has not been integrated
                        orbInterp.append(_fakeInterp(self.vxvv[0]*nu.sin(self.vxvv[-1])))
                    else:
                        orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                                self.t,self.orbit[:,0]*nu.sin(self.orbit[:,-1]),**_KWINTERP))
                else:
                    if not hasattr(self,"t"): #Orbit has not been integrated
                        orbInterp.append(_fakeInterp(self.vxvv[ii]))
                    else:
                        orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                                self.t,self.orbit[:,ii],**_KWINTERP))
            self._orbInterp= orbInterp
            try: #unsort
                self.t= self.t[usindx]
                self.orbit= self.orbit[usindx]
            except: pass
        return None

    def animate(self,*args,**kwargs): #pragma: no cover
        """
        NAME:
           animate
        PURPOSE:
           animate an Orbit
        INPUT:
           d1= first dimension to plot ('x', 'y', 'R', 'vR', 'vT', 'z', 'vz', ...); can be list with up to three entries for three subplots
           d2= second dimension to plot; can be list with up to three entries for three subplots
           width= (600) width of output div in px
           height= (400) height of output div in px
           json_filename= (None) if set, save the data necessary for the figure in this filename (e.g.,  json_filename= 'orbit_data/orbit.json'); this path is also used in the output HTML, so needs to be accessible
           load_jslibs= (True) if True, load the require and jQuery Javascript libraries (necessary in Jupyterlab, not necessary but harmless in notebooks; if embedding on a webpage one typically wants to load these libraries in the header)
           ro= (Object-wide default) physical scale for distances to use to convert
           vo= (Object-wide default) physical scale for velocities to use to convert
           use_physical= use to override Object-wide default for using a physical scale for output

           +kwargs for ra,dec,ll,bb, etc. functions
        OUTPUT:
           IPython.display.HTML object with code to animate the orbit; can be directly shown in jupyter notebook or embedded in HTML pages; get a text version of the HTML using the _repr_html_() function
        HISTORY:
           2017-09-17-24 - Written - Bovy (UofT)
           2017-11-28 - Allow arbitrary functions of time to be plotted - Bovy (UofT)
        """
        try:
            from IPython.display import HTML
        except ImportError:
            raise ImportError("Orbit.animate requires ipython/jupyter to be installed")
        if (kwargs.get('use_physical',False) \
                and kwargs.get('ro',self._roSet)) or \
                (not 'use_physical' in kwargs \
                     and kwargs.get('ro',self._roSet)):
            labeldict= {'t':'t (Gyr)',
                        'R':'R (kpc)',
                        'vR':'v_R (km/s)',
                        'vT':'v_T (km/s)',
                        'z':'z (kpc)',
                        'vz':'v_z (km/s)',
                        'phi':'azimuthal angle',
                        'r':'r (kpc)',
                        'x':'x (kpc)',
                        'y':'y (kpc)',
                        'vx':'v_x (km/s)',
                        'vy':'v_y (km/s)',
                        'E':'E (km^2/s^2)',
                        'Ez':'E_z (km^2/s^2)',
                        'ER':'E_R (km^2/s^2)',
                        'Enorm':'E(t)/E(0.)',
                        'Eznorm':'E_z(t)/E_z(0.)',
                        'ERnorm':'E_R(t)/E_R(0.)',
                        'Jacobi':'E-Omega_p L (km^2/s^2)',
                        'Jacobinorm':'(E-Omega_p L)(t)/(E-Omega_p L)(0)'}
        else:
            labeldict= {'t':'t','R':'R','vR':'v_R','vT':'v_T',
                        'z':'z','vz':'v_z','phi':r'azimuthal angle',
                        'r':'r',
                        'x':'x','y':'y','vx':'v_x','vy':'v_y',
                        'E':'E','Enorm':'E(t)/E(0.)',
                        'Ez':'E_z','Eznorm':'E_z(t)/E_z(0.)',
                        'ER':r'E_R','ERnorm':r'E_R(t)/E_R(0.)',
                        'Jacobi':r'E-Omega_p L',
                        'Jacobinorm':r'(E-Omega_p L)(t)/(E-Omega_p L)(0)'}
        labeldict.update({'ra':'RA (deg)',
                          'dec':'Dec (deg)',
                          'll':'Galactic lon (deg)',
                          'bb':'Galactic lat (deg)',
                          'dist':'distance (kpc)',
                          'pmra':'pmRA (mas/yr)',
                          'pmdec':'pmDec (mas/yr)',
                          'pmll':'pmGlon (mas/yr)',
                          'pmbb':'pmGlat (mas/yr)',
                          'vlos':'line-of-sight vel (km/s)',
                          'helioX':'X (kpc)',
                          'helioY':'Y (kpc)',
                          'helioZ':'Z (kpc)',
                          'U':'U (km/s)',
                          'V':'V (km/s)',
                          'W':'W (km/s)'})
        # Cannot be using Quantity output
        kwargs['quantity']= False
        #Defaults
        if not 'd1' in kwargs and not 'd2' in kwargs:
            if len(self.vxvv) == 3:
                d1= 'R'
                d2= 'vR'
            elif len(self.vxvv) == 4:
                d1= 'x'
                d2= 'y'
            elif len(self.vxvv) == 2:
                d1= 'x'
                d2= 'vx'
            elif len(self.vxvv) == 5 or len(self.vxvv) == 6:
                d1= 'R'
                d2= 'z'
        elif not 'd1' in kwargs:
            d2=  kwargs.pop('d2')
            d1= 't'
        elif not 'd2' in kwargs:
            d1= kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs.pop('d1')
            d2= kwargs.pop('d2')
        xs= []
        ys= []
        xlabels= []
        ylabels= []
        if isinstance(d1,str) or callable(d1):
            d1s= [d1]
            d2s= [d2]
        else:
            d1s= d1
            d2s= d2
        if len(d1s) > 3:
            raise ValueError('Orbit.animate only works for up to three subplots')
        all_xlabel= kwargs.get('xlabel',[None for d in d1])
        all_ylabel= kwargs.get('ylabel',[None for d in d2])
        for d1,d2, xlabel, ylabel in zip(d1s,d2s,all_xlabel,all_ylabel):
           #Get x and y for each subplot
            if callable(d1):
                x= d1(self.t)
            elif d1 == 't':
                x= self.time(self.t,**kwargs)
            elif d1 == 'R':
                x= self.R(self.t,**kwargs)
            elif d1 == 'r':
                x= nu.sqrt(self.R(self.t,**kwargs)**2.
                           +self.z(self.t,**kwargs)**2.)
            elif d1 == 'z':
                x= self.z(self.t,**kwargs)
            elif d1 == 'vz':
                x= self.vz(self.t,**kwargs)
            elif d1 == 'vR':
                x= self.vR(self.t,**kwargs)
            elif d1 == 'vT':
                x= self.vT(self.t,**kwargs)
            elif d1 == 'x':
                x= self.x(self.t,**kwargs)
            elif d1 == 'y':
                x= self.y(self.t,**kwargs)
            elif d1 == 'vx':
                x= self.vx(self.t,**kwargs)
            elif d1 == 'vy':
                x= self.vy(self.t,**kwargs)
            elif d1 == 'phi':
                x= self.phi(self.t,**kwargs)
            elif d1.lower() == 'ra':
                x= self.ra(self.t,**kwargs)
            elif d1.lower() == 'dec':
                x= self.dec(self.t,**kwargs)
            elif d1 == 'll':
                x= self.ll(self.t,**kwargs)
            elif d1 == 'bb':
                x= self.bb(self.t,**kwargs)
            elif d1 == 'dist':
                x= self.dist(self.t,**kwargs)
            elif d1 == 'pmra':
                x= self.pmra(self.t,**kwargs)
            elif d1 == 'pmdec':
                x= self.pmdec(self.t,**kwargs)
            elif d1 == 'pmll':
                x= self.pmll(self.t,**kwargs)
            elif d1 == 'pmbb':
                x= self.pmbb(self.t,**kwargs)
            elif d1 == 'vlos':
                x= self.vlos(self.t,**kwargs)
            elif d1 == 'helioX':
                x= self.helioX(self.t,**kwargs)
            elif d1 == 'helioY':
                x= self.helioY(self.t,**kwargs)
            elif d1 == 'helioZ':
                x= self.helioZ(self.t,**kwargs)
            elif d1 == 'U':
                x= self.U(self.t,**kwargs)
            elif d1 == 'V':
                x= self.V(self.t,**kwargs)
            elif d1 == 'W':
                x= self.W(self.t,**kwargs)
            elif d1 == 'E':
                x= self.E(self.t,**kwargs)
            elif d1 == 'Enorm':
                x= self.E(self.t,**kwargs)/self.E(0.,**kwargs)
            elif d1 == 'Ez':
                x= self.Ez(self.t,**kwargs)
            elif d1 == 'Eznorm':
                x= self.Ez(self.t,**kwargs)/self.Ez(0.,**kwargs)
            elif d1 == 'ER':
                x= self.ER(self.t,**kwargs)
            elif d1 == 'ERnorm':
                x= self.ER(self.t,**kwargs)/self.ER(0.,**kwargs)
            elif d1 == 'Jacobi':
                x= self.Jacobi(self.t,**kwargs)
            elif d1 == 'Jacobinorm':
                x= self.Jacobi(self.t,**kwargs)/self.Jacobi(0.,**kwargs)
            if callable(d2):
                y= d2(self.t)
            elif d2 == 't':
                y= self.time(self.t,**kwargs)
            elif d2 == 'R':
                y= self.R(self.t,**kwargs)
            elif d2 == 'r':
                y= nu.sqrt(self.R(self.t,**kwargs)**2.
                           +self.z(self.t,**kwargs)**2.)
            elif d2 == 'z':
                y= self.z(self.t,**kwargs)
            elif d2 == 'vz':
                y= self.vz(self.t,**kwargs)
            elif d2 == 'vR':
                y= self.vR(self.t,**kwargs)
            elif d2 == 'vT':
                y= self.vT(self.t,**kwargs)
            elif d2 == 'x':
                y= self.x(self.t,**kwargs)
            elif d2 == 'y':
                y= self.y(self.t,**kwargs)
            elif d2 == 'vx':
                y= self.vx(self.t,**kwargs)
            elif d2 == 'vy':
                y= self.vy(self.t,**kwargs)
            elif d2 == 'phi':
                y= self.phi(self.t,**kwargs)
            elif d2.lower() == 'ra':
                y= self.ra(self.t,**kwargs)
            elif d2.lower() == 'dec':
                y= self.dec(self.t,**kwargs)
            elif d2 == 'll':
                y= self.ll(self.t,**kwargs)
            elif d2 == 'bb':
                y= self.bb(self.t,**kwargs)
            elif d2 == 'dist':
                y= self.dist(self.t,**kwargs)
            elif d2 == 'pmra':
                y= self.pmra(self.t,**kwargs)
            elif d2 == 'pmdec':
                y= self.pmdec(self.t,**kwargs)
            elif d2 == 'pmll':
                y= self.pmll(self.t,**kwargs)
            elif d2 == 'pmbb':
                y= self.pmbb(self.t,**kwargs)
            elif d2 == 'vlos':
                y= self.vlos(self.t,**kwargs)
            elif d2 == 'helioX':
                y= self.helioX(self.t,**kwargs)
            elif d2 == 'helioY':
                y= self.helioY(self.t,**kwargs)
            elif d2 == 'helioZ':
                y= self.helioZ(self.t,**kwargs)
            elif d2 == 'U':
                y= self.U(self.t,**kwargs)
            elif d2 == 'V':
                y= self.V(self.t,**kwargs)
            elif d2 == 'W':
                y= self.W(self.t,**kwargs)
            elif d2 == 'E':
                y= self.E(self.t,**kwargs)
            elif d2 == 'Enorm':
                y= self.E(self.t,**kwargs)/self.E(0.,**kwargs)
            elif d2 == 'Ez':
                y= self.Ez(self.t,**kwargs)
            elif d2 == 'Eznorm':
                y= self.Ez(self.t,**kwargs)/self.Ez(0.,**kwargs)
            elif d2 == 'ER':
                y= self.ER(self.t,**kwargs)
            elif d2 == 'ERnorm':
                y= self.ER(self.t,**kwargs)/self.ER(0.,**kwargs)
            elif d2 == 'Jacobi':
                y= self.Jacobi(self.t,**kwargs)
            elif d2 == 'Jacobinorm':
                y= self.Jacobi(self.t,**kwargs)/self.Jacobi(0.,**kwargs)
            xs.append(x)
            ys.append(y)
            if xlabel is None:
                xlabels.append(labeldict.get(d1,'\mathrm{No\ xlabel\ specified}'))
            else:
                xlabels.append(xlabel)
            if ylabel is None:
                ylabels.append(labeldict.get(d2,'\mathrm{No\ ylabel\ specified}'))
            else:
                ylabels.append(ylabel)
        kwargs.pop('ro',None)
        kwargs.pop('vo',None)
        kwargs.pop('obs',None)
        kwargs.pop('use_physical',None)
        kwargs.pop('pot',None)
        kwargs.pop('OmegaP',None)
        kwargs.pop('quantity',None)
        width= kwargs.pop('width',600)
        height= kwargs.pop('height',400)
        load_jslibs= kwargs.pop('load_jslibs',True)
        if load_jslibs:
            load_jslibs_code= """</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.5/require.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script>
"""
        else:
            load_jslibs_code= ""
        # Dump data to HTML
        nplots= len(xs)
        jsonDict= {}
        jsonDict['x']= xs[0].tolist()
        jsonDict['y']= ys[0].tolist()
        for ii in range(1,nplots):
            jsonDict['x%i' % (ii+1)]= xs[ii].tolist()
            jsonDict['y%i' % (ii+1)]= ys[ii].tolist()
        json_filename= kwargs.pop('json_filename',None)
        if json_filename is None:
            jd= json.dumps(jsonDict)
            json_code= """  let data= JSON.parse('{jd}');""".format(jd=jd)
            close_json_code= ""
        else:
            with open(json_filename,'w') as jfile:
                json.dump(jsonDict,jfile)
            json_code= """Plotly.d3.json('{jfilename}',function(data){{""".format(jfilename=json_filename)
            close_json_code= "});"
        self.divid= 'galpy-'\
            +''.join(choice(ascii_lowercase) for i in range(24))
        button_width= 419.51+4.*10.
        button_margin_left= int(nu.round((width-button_width)/2.))
        if button_margin_left < 0: button_margin_left= 0
        # Layout for multiple plots
        if len(d1s) == 1:
            xmin= [0,0,0]
            xmax= [1,1,1]
        elif len(d1s) == 2:
            xmin= [0,0.55,0]
            xmax= [0.45,1,1]
        elif len(d1s) == 3:
            xmin= [0,0.365,0.73]
            xmax= [0.27,0.635,1]
        layout= """{{
  xaxis: {{
    title: '{xlabel}',
    domain: [{xmin},{xmax}],
}},
  yaxis: {{title: '{ylabel}'}},
  margin: {{t: 20}},
  hovermode: 'closest',
  showlegend: false,
""".format(xlabel=xlabels[0],ylabel=ylabels[0],xmin=xmin[0],xmax=xmax[0])
        for ii in range(1,nplots):
            layout+= """  xaxis{idx}: {{
    title: '{xlabel}',
    anchor: 'y{idx}',
    domain: [{xmin},{xmax}],
}},
  yaxis{idx}: {{
    title: '{ylabel}',
    anchor: 'x{idx}',
}},
""".format(idx=ii+1,xlabel=xlabels[ii],ylabel=ylabels[ii],
           xmin=xmin[ii],xmax=xmax[ii])
        layout+="""}"""
        # Additional traces for additional plots
        if len(d1s) > 1:
            setup_trace2= """
    let trace3= {{
      x: data.x2.slice(0,numPerFrame),
      y: data.y2.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '#1f77b4',
      }},
    }};

    let trace4= {{
      x: data.x2.slice(0,numPerFrame), 
      y: data.y2.slice(0,numPerFrame),
      xaxis: 'x2',
      yaxis: 'y2',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '#d62728',
      }},
    }};
""".format(divid=self.divid) # not used!
            delete_trace4= """Plotly.deleteTraces('{divid}',3);""".format(divid=self.divid)
            delete_trace3= """Plotly.deleteTraces('{divid}',0);""".format(divid=self.divid)
            update_trace34= """
      trace_slice_begin+= trace_slice_len;
      Plotly.extendTraces('{divid}', {{
        x: [data.x2.slice(trace_slice_begin,trace_slice_end)],
        y: [data.y2.slice(trace_slice_begin,trace_slice_end)],
      }}, [2]);

      trace_slice_begin-= trace_slice_len;
      trace4= {{
        x: [data.x2.slice(trace_slice_begin,trace_slice_end)], 
        y: [data.y2.slice(trace_slice_begin,trace_slice_end)],
      }},
      Plotly.restyle('{divid}',trace4,[3]);
""".format(divid=self.divid)
        else:
            setup_trace2= """
    let traces= [trace1,trace2];
"""
            delete_trace4= ""
            delete_trace3= ""
            update_trace34= ""
        if len(d1s) > 2:
            setup_trace3= """
    let trace5= {{
      x: data.x3.slice(0,numPerFrame),
      y: data.y3.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '#1f77b4',
      }},
    }};

    let trace6= {{
      x: data.x3.slice(0,numPerFrame), 
      y: data.y3.slice(0,numPerFrame),
      xaxis: 'x3',
      yaxis: 'y3',
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 3.,
        color: '#d62728',
      }},
    }};

    let traces= [trace1,trace2,trace3,trace4,trace5,trace6];
""".format(divid=self.divid)
            delete_trace6= """Plotly.deleteTraces('{divid}',5);""".format(divid=self.divid)
            delete_trace5= """Plotly.deleteTraces('{divid}',0);""".format(divid=self.divid)
            update_trace56= """
      trace_slice_begin+= trace_slice_len;
      Plotly.extendTraces('{divid}', {{
        x: [data.x3.slice(trace_slice_begin,trace_slice_end)],
        y: [data.y3.slice(trace_slice_begin,trace_slice_end)],
      }}, [4]);

      trace_slice_begin-= trace_slice_len;
      trace6= {{
        x: [data.x3.slice(trace_slice_begin,trace_slice_end)], 
        y: [data.y3.slice(trace_slice_begin,trace_slice_end)],
      }},
      Plotly.restyle('{divid}',trace6,[5]);
""".format(divid=self.divid)
        elif len(d1s) > 1:
            setup_trace3= """
    let traces= [trace1,trace2,trace3,trace4];
"""
            delete_trace5= ""
            delete_trace6= ""
            update_trace56= ""
        else:
            setup_trace3= ""
            delete_trace5= ""
            delete_trace6= ""
            update_trace56= ""

        return HTML("""
<style>
.galpybutton {{
    background-color:#ffffff;
    -moz-border-radius:16px;
    -webkit-border-radius:16px;
    border-radius:16px;
    border:1px solid #1f77b4;
    display:inline-block;
    cursor:pointer;
    color:#1f77b4;
    font-family:Courier;
    font-size:17px;
    padding:8px 10px;
    text-decoration:none;
    text-shadow:0px 1px 0px #2f6627;
}}
.galpybutton:hover {{
    background-color:#ffffff;
}}
.galpybutton:active {{
    position:relative;
    top:1px;
}}
.galpybutton:focus{{
    outline:0;
}}
</style>

<div id='{divid}' style='width:{width}px;height:{height}px;'></div>
<div class="controlbutton" id="{divid}-play" style="margin-left:{button_margin_left}px;display: inline-block;">
<button class="galpybutton">Play</button></div>
<div class="controlbutton" id="{divid}-pause" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Pause</button></div>
<div class="controlbutton" id="{divid}-timestwo" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Speed<font face="Arial">&thinsp;</font>x<font face="Arial">&thinsp;</font>2</button></div>
<div class="controlbutton" id="{divid}-timeshalf" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Speed<font face="Arial">&thinsp;</font>/<font face="Arial">&thinsp;</font>2</button></div>
<div class="controlbutton" id="{divid}-replay" style="margin-left:10px;display: inline-block;">
<button class="galpybutton">Replay</button></div>

<script>
require.config({{
  paths: {{
    Plotly: 'https://cdn.plot.ly/plotly-latest.min',
  }}
}});
{load_jslibs_code}
require(['Plotly'], function (Plotly) {{
{json_code}
  let layout = {layout};
  let numPerFrame= 5;    
  let cnt= 1;
  let interval;
  let trace_slice_len;
  let trace_slice_begin;
  let trace_slice_end;

  setup_trace();
  
  $('.controlbutton button').click(function() {{
    let button_type= this.parentNode.id;
    if ( button_type === '{divid}-play' ) {{
      clearInterval(interval);
      interval= animate_trace();
    }}
    else if ( button_type === '{divid}-pause' )
      clearInterval(interval);
    else if ( button_type === '{divid}-timestwo' ) {{
      cnt/= 2;
      numPerFrame*= 2;
    }}
    else if ( button_type === '{divid}-timeshalf' ) {{
      cnt*= 2;
      numPerFrame/= 2;
    }}
    else if ( button_type === '{divid}-replay' ) {{
      cnt= 1;
      try {{ // doesn't exist if animation has already ended
        {delete_trace6}
        {delete_trace4}
        Plotly.deleteTraces('{divid}',1);
      }}
      catch (err) {{
      }}
      Plotly.deleteTraces('{divid}',0);
      {delete_trace3}
      {delete_trace5}
      clearInterval(interval);
      setup_trace();
      interval= animate_trace();
    }}
  }});
    
  function setup_trace() {{
    let trace1= {{
      x: data.x.slice(0,numPerFrame), 
      y: data.y.slice(0,numPerFrame),
      mode: 'lines',
      line: {{
        shape: 'spline',
        width: 0.8,
        color: '#1f77b4',
       }},
    }};

   let trace2= {{
     x: data.x.slice(0,numPerFrame), 
     y: data.y.slice(0,numPerFrame),
     mode: 'lines',
     line: {{
       shape: 'spline',
       width: 3.,
       color: '#d62728',
       }},
     }};

    {setup_trace2}

    {setup_trace3}

    Plotly.plot('{divid}',traces,layout);
  }}

  function animate_trace() {{
    return setInterval(function() {{
      // Make sure narrow and thick trace end in the same 
      // and the highlighted length has constant length
      trace_slice_len= Math.floor(numPerFrame);
      if ( trace_slice_len < 1) trace_slice_len= 1;
      trace_slice_begin= Math.floor(cnt*numPerFrame);
      trace_slice_end= Math.floor(Math.min(cnt*numPerFrame+trace_slice_len,data.x.length-1));
      Plotly.extendTraces('{divid}', {{
        x: [data.x.slice(trace_slice_begin,trace_slice_end)],
        y: [data.y.slice(trace_slice_begin,trace_slice_end)],
      }}, [0]);
      trace_slice_begin-= trace_slice_len;
      trace2= {{
        x: [data.x.slice(trace_slice_begin,trace_slice_end)], 
        y: [data.y.slice(trace_slice_begin,trace_slice_end)],
      }};
      Plotly.restyle('{divid}',trace2,[1]);

      {update_trace34}
      {update_trace56}
      cnt+= 1;
      if(cnt*numPerFrame+trace_slice_len > data.x.length/1) {{
          clearInterval(interval);
          {delete_trace6}
          {delete_trace4}
          Plotly.deleteTraces('{divid}',1);
      }}
    }}, 30);
    }}
{close_json_code}}});
</script>""".format(json_code=json_code,close_json_code=close_json_code,
                    divid=self.divid,width=width,height=height,
                    button_margin_left=button_margin_left,
                    layout=layout,load_jslibs_code=load_jslibs_code,
                    setup_trace2=setup_trace2,setup_trace3=setup_trace3,
                    delete_trace4=delete_trace4,delete_trace6=delete_trace6,
                    delete_trace3=delete_trace3,delete_trace5=delete_trace5,
                    update_trace34=update_trace34,
                    update_trace56=update_trace56))

class _fakeInterp(object): 
    """Fake class to simulate interpolation when orbit was not integrated"""
    def __init__(self,x):
        self.x= x
    def __call__(self,t):
        if nu.any(nu.array(t) != 0.):
            raise ValueError("Integrate instance before evaluating it at non-zero time")
        else:
            return nu.array([self.x for i in t])

def _check_roSet(orb,kwargs,funcName):
    """Function to check whether ro is set, because it's required for funcName"""
    if not orb._roSet and kwargs.get('ro',None) is None:
        warnings.warn("Method %s(.) requires ro to be given at Orbit initialization or at method evaluation; using default ro which is %f kpc" % (funcName,orb._ro),
                      galpyWarning)

def _check_voSet(orb,kwargs,funcName):
    """Function to check whether vo is set, because it's required for funcName"""
    if not orb._voSet and kwargs.get('vo',None) is None:
        warnings.warn("Method %s(.) requires vo to be given at Orbit initialization or at method evaluation; using default vo which is %f km/s" % (funcName,orb._vo),
                      galpyWarning)
