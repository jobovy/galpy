import math as m
import numpy as nu
from scipy import interpolate, optimize
from galpy import actionAngle
import galpy.util.bovy_plot as plot
import galpy.util.bovy_coords as coords
from galpy.potential_src.planarPotential import RZToplanarPotential
class OrbitTop:
    """General class that holds orbits and integrates them"""
    def __init__(self,vxvv=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize an orbit instance
        INPUT:
           vxvv - initial condition
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return None

    def integrate(self,t,pot,method='leapfrog_c'):
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

    def integrateBC(self,pot,bc=None,method='odeint'):
        """
        NAME:
           integrateBC
        PURPOSE:
           integrate the orbit subject to a final boundary condition
        INPUT:
           pot - potential instance or list of instances
           bc= boundary condition, takes array of phase-space position (in the manner that is relevant to the type of Orbit) and outputs the condition that should be zero; default: z=0
           method= 'odeint' for scipy's odeint integrator, 'leapfrog' for
                   a simple symplectic integrator
        OUTPUT:
           Another Orbit instance, time at which the BC is reached
        HISTORY:
           2011-09-30
        """
        #Parse potential
        if len(self.vxvv) == 3 or len(self.vxvv) == 4:
            thispot= RZToplanarPotential(pot)
        else:
            thispot= pot
        #First find the interval; initialize
        dt= 1.
        a,b=  0., dt
        vxvv_a= self.vxvv
        bc_a= bc(vxvv_a)
        if bc_a == 0.:
            return (nu.array([vxvv_a,vxvv_a]),0.)
        tmp_orb= self._BCIntegrateFunction(vxvv_a,thispot,nu.array([0.,b-a]),method)
        vxvv_b= tmp_orb[1,:]
        bc_b= bc(vxvv_b)
        if bc_b*bc_a < 0.: found_init_interval= True
        else: found_init_interval= False
        while not found_init_interval:
            #Repeat!
            a= b
            b= a+1.
            bc_a= bc_b
            vxvv_a= vxvv_b
            tmp_orb= self._BCIntegrateFunction(vxvv_a,thispot,nu.array([0.,b-a]),method)
            vxvv_b= tmp_orb[1,:]
            bc_b= bc(vxvv_b)
            if bc_b*bc_a < 0.: found_init_interval= True
        tout= optimize.brentq(_BCZeroFunction,a,b,
                              args=(vxvv_a,thispot,method,bc,a,self._BCIntegrateFunction))
        t= nu.array([a,tout])
        return (self._BCIntegrateFunction(vxvv_a,thispot,t,method),a+tout)
    
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

    def R(self,*args,**kwargs):
        """
        NAME:
           R
        PURPOSE:
           return cylindrical radius at time t
        INPUT:
           t - (optional) time at which to get the radius
        OUTPUT:
           R(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[0]
        else: return thiso[0,:]

    def vR(self,*args,**kwargs):
        """
        NAME:
           vR
        PURPOSE:
           return radial velocity at time t
        INPUT:
           t - (optional) time at which to get the radial velocity
        OUTPUT:
           vR(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[1]
        else: return thiso[1,:]

    def vT(self,*args,**kwargs):
        """
        NAME:
           vT
        PURPOSE:
           return tangential velocity at time t
        INPUT:
           t - (optional) time at which to get the tangential velocity
        OUTPUT:
           vT(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet: return thiso[2]
        else: return thiso[2,:]

    def z(self,*args,**kwargs):
        """
        NAME:
           z
        PURPOSE:
           return vertical height
        INPUT:
           t - (optional) time at which to get the vertical height
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

    def vz(self,*args,**kwargs):
        """
        NAME:
           vz
        PURPOSE:
           return vertical velocity
        INPUT:
           t - (optional) time at which to get the vertical velocity
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

    def x(self,*args,**kwargs):
        """
        NAME:
           x
        PURPOSE:
           return x
        INPUT:
           t - (optional) time at which to get x
        OUTPUT:
           x(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) == 2:
            return thiso[:,0]
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use x()")
        elif len(thiso[:,0]) == 4:
            return thiso[0,:]*nu.cos(thiso[3,:])
        else:
            return thiso[0,:]*nu.cos(thiso[5,:])

    def y(self,*args,**kwargs):
        """
        NAME:
           y
        PURPOSE:
           return y
        INPUT:
           t - (optional) time at which to get y
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

    def vx(self,*args,**kwargs):
        """
        NAME:
           vx
        PURPOSE:
           return x velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
        OUTPUT:
           vx(t)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) == 2:
            return thiso[:,1]
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use vx()")
        elif len(thiso[:,0]) == 4:
            theta= thiso[3,:]
        else:
            theta= thiso[5,:]
        return thiso[1,:]*nu.cos(theta)-thiso[2,:]*nu.sin(theta)

    def vy(self,*args,**kwargs):
        """
        NAME:
           vy
        PURPOSE:
           return y velocity at time t
        INPUT:
           t - (optional) time at which to get the velocity
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

    def vphi(self,*args,**kwargs):
        """
        NAME:
           vphi
        PURPOSE:
           return angular velocity
        INPUT:
           t - (optional) time at which to get the angular velocity
        OUTPUT:
           vphi(t)
        HISTORY:
           2010-09-21 - Written - Bovy (NYU)
        """
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        return thiso[2,:]/thiso[0,:]

    def ra(self,*args,**kwargs):
        """
        NAME:
           ra
        PURPOSE:
           return the right ascension
        INPUT:
           t - (optional) time at which to get ra
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=[8.5,0.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)
        OUTPUT:
           ra(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        radec= self._radec(*args,**kwargs)
        return radec[:,0]

    def dec(self,*args,**kwargs):
        """
        NAME:
           dec
        PURPOSE:
           return the declination
        INPUT:
           t - (optional) time at which to get dec
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=[8.5,0.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)
        OUTPUT:
           dec(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        radec= self._radec(*args,**kwargs)
        return radec[:,1]

    def ll(self,*args,**kwargs):
        """
        NAME:
           ll
        PURPOSE:
           return Galactic longitude
        INPUT:
           t - (optional) time at which to get ll
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=[8.5,0.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           l(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,0]

    def bb(self,*args,**kwargs):
        """
        NAME:
           bb
        PURPOSE:
           return Galactic latitude
        INPUT:
           t - (optional) time at which to get bb
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=[8.5,0.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           b(t)
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,1]

    def dist(self,*args,**kwargs):
        """
        NAME:
           dist
        PURPOSE:
           return distance from the observer in kpc
        INPUT:
           t - (optional) time at which to get dist
           obs=[X,Y,Z] - (optional) position of observer (in kpc) 
                         (default=[8.5,0.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           dist(t) in kpc
        HISTORY:
           2011-02-23 - Written - Bovy (NYU)
        """
        lbd= self._lbd(*args,**kwargs)
        return lbd[:,2].astype('float64')

    def pmra(self,*args,**kwargs):
        """
        NAME:
           pmra
        PURPOSE:
           return proper motion in right ascension (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmra
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           pm_ra(t) in mas / yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        pmrapmdec= self._pmrapmdec(*args,**kwargs)
        return pmrapmdec[:,0]

    def pmdec(self,*args,**kwargs):
        """
        NAME:
           pmdec
        PURPOSE:
           return proper motion in declination (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmdec
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           pm_dec(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        pmrapmdec= self._pmrapmdec(*args,**kwargs)
        return pmrapmdec[:,1]

    def pmll(self,*args,**kwargs):
        """
        NAME:
           pmll
        PURPOSE:
           return proper motion in Galactic longitude (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmll
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           pm_l(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,4]

    def pmbb(self,*args,**kwargs):
        """
        NAME:
           pmbb
        PURPOSE:
           return proper motion in Galactic latitude (in mas/yr)
        INPUT:
           t - (optional) time at which to get pmbb
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           pm_b(t) in mas/yr
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,5]

    def vlos(self,*args,**kwargs):
        """
        NAME:
           vlos
        PURPOSE:
           return the line-of-sight velocity (in km/s)
        INPUT:
           t - (optional) time at which to get vlos
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           vlos(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return lbdvrpmllpmbb[:,3]

    def helioX(self,*args,**kwargs):
        """
        NAME:
           helioX
        PURPOSE:
           return Heliocentric Galactic rectangular x-coordinate (aka "X")
        INPUT:
           t - (optional) time at which to get X
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           helioX(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return X

    def helioY(self,*args,**kwargs):
        """
        NAME:
           helioY
        PURPOSE:
           return Heliocentric Galactic rectangular y-coordinate (aka "Y")
        INPUT:
           t - (optional) time at which to get Y
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           helioY(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return Y

    def helioZ(self,*args,**kwargs):
        """
        NAME:
           helioZ
        PURPOSE:
           return Heliocentric Galactic rectangular z-coordinate (aka "Z")
        INPUT:
           t - (optional) time at which to get Z
           obs=[X,Y,Z] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
        OUTPUT:
           helioZ(t) in kpc
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z= self._helioXYZ(*args,**kwargs)
        return Z

    def U(self,*args,**kwargs):
        """
        NAME:
           U
        PURPOSE:
           return Heliocentric Galactic rectangular x-velocity (aka "U")
        INPUT:
           t - (optional) time at which to get U
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           U(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return U

    def V(self,*args,**kwargs):
        """
        NAME:
           V
        PURPOSE:
           return Heliocentric Galactic rectangular y-velocity (aka "V")
        INPUT:
           t - (optional) time at which to get U
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           V(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return V

    def W(self,*args,**kwargs):
        """
        NAME:
           W
        PURPOSE:
           return Heliocentric Galactic rectangular z-velocity (aka "W")
        INPUT:
           t - (optional) time at which to get W
           obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                         (in kpc and km/s) (default=[8.5,0.,0.,0.,235.,0.])
                         OR Orbit object that corresponds to the orbit
                         of the observer
           ro= distance in kpc corresponding to R=1. (default: 8.5)         
           vo= velocity in km/s corresponding to v=1. (default: 235.)
        OUTPUT:
           W(t) in km/s
        HISTORY:
           2011-02-24 - Written - Bovy (NYU)
        """
        X, Y, Z, U, V, W= self._XYZvxvyvz(*args,**kwargs)
        return W

    def _radec(self,*args,**kwargs):
        """Calculate ra and dec"""
        lbd= self._lbd(*args,**kwargs)
        return coords.lb_to_radec(lbd[:,0],lbd[:,1],degree=True)

    def _pmrapmdec(self,*args,**kwargs):
        """Calculate pmra and pmdec"""
        lbdvrpmllpmbb= self._lbdvrpmllpmbb(*args,**kwargs)
        return coords.pmllpmbb_to_pmrapmdec(lbdvrpmllpmbb[:,4],
                                            lbdvrpmllpmbb[:,5],
                                            lbdvrpmllpmbb[:,0],
                                            lbdvrpmllpmbb[:,1],degree=True)

    def _lbd(self,*args,**kwargs):
        """Calculate l,b, and d"""
        obs, ro, vo= _parse_radec_kwargs(kwargs,dontpop=True)
        X,Y,Z= self._helioXYZ(*args,**kwargs)
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx:
            X[bad_indx]+= ro/10000.
        return coords.XYZ_to_lbd(X*ro,Y*ro,Z*ro,degree=True)

    def _helioXYZ(self,*args,**kwargs):
        """Calculate heliocentric rectangular coordinates"""
        obs, ro, vo= _parse_radec_kwargs(kwargs)
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use radeclbd functions")
        elif len(thiso[:,0]) == 4: #planarOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                Xsun=obs[0]/ro,
                                                Ysun=obs[1]/ro,
                                                Zsun=obs[2]/ro)
            else: #Orbit instance
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=0.)
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=obs.z(*args,**kwargs))
        else: #FullOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                thiso[3,:],
                                                Xsun=obs[0]/ro,
                                                Ysun=obs[1]/ro,
                                                Zsun=obs[2]/ro)
            else: #Orbit instance
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                    thiso[3,:],
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=0.)
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                    thiso[3,:],
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=obs.z(*args,**kwargs))
        return (X,Y,Z)

    def _lbdvrpmllpmbb(self,*args,**kwargs):
        """Calculate l,b,d,vr,pmll,pmbb"""
        obs, ro, vo= _parse_radec_kwargs(kwargs,dontpop=True)
        X,Y,Z,vX,vY,vZ= self._XYZvxvyvz(*args,**kwargs)
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx:
            X[bad_indx]+= ro/10000.
        return coords.rectgal_to_sphergal(X,Y,Z,vX,vY,vZ,degree=True)

    def _XYZvxvyvz(self,*args,**kwargs):
        """Calculate X,Y,Z,U,V,W"""
        obs, ro, vo= _parse_radec_kwargs(kwargs,vel=True)
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) != 4 and len(thiso[:,0]) != 6:
            raise AttributeError("orbit must track azimuth to use radeclbduvw functions")
        elif len(thiso[:,0]) == 4: #planarOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                Xsun=obs[0]/ro,
                                                Ysun=obs[1]/ro,
                                                Zsun=obs[2]/ro)
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],thiso[2,:],0.,
                                                      thiso[3,:],
                                                      vsun=nu.array(\
                        obs[3:6])/vo)
            else: #Orbit instance
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=0.)
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],
                                                          thiso[2,:],
                                                          0.,
                                                          thiso[3,:],
                                                          vsun=nu.array([\
                                obs.vx(*args,**kwargs),obs.vy(*args,**kwargs),
                                nu.zeros(len(thiso[0,:]))]))
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[3,:],0.,
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=obs.z(*args,**kwargs))
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],
                                                          thiso[2,:],
                                                          0.,
                                                          thiso[3,:],
                                                          vsun=nu.array([\
                                obs.vx(*args,**kwargs),
                                obs.vy(*args,**kwargs),
                                obs.vz(*args,**kwargs)]))
        else: #FullOrbit
            if isinstance(obs,(nu.ndarray,list)):
                X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                thiso[3,:],
                                                Xsun=obs[0]/ro,
                                                Ysun=obs[1]/ro,
                                                Zsun=obs[2]/ro)
                vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],
                                                      thiso[2,:],
                                                      thiso[4,:],
                                                      thiso[5,:],
                                                      vsun=nu.array(\
                        obs[3:6])/vo)
            else: #Orbit instance
                if obs.dim() == 2:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                    thiso[3,:],
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=0.)
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],
                                                          thiso[2,:],
                                                          thiso[4,:],
                                                          thiso[5,:],
                                                          vsun=nu.array([\
                                obs.vx(*args,**kwargs),obs.vy(*args,**kwargs),
                                nu.zeros(len(thiso[0,:]))]))
                else:
                    X,Y,Z = coords.galcencyl_to_XYZ(thiso[0,:],thiso[5,:],
                                                    thiso[3,:],
                                                    Xsun=obs.x(*args,**kwargs),
                                                    Ysun=obs.y(*args,**kwargs),
                                                    Zsun=obs.z(*args,**kwargs))
                    vX,vY,vZ = coords.galcencyl_to_vxvyvz(thiso[1,:],
                                                          thiso[2,:],
                                                          thiso[4,:],
                                                          thiso[5,:],
                                                          vsun=nu.array([\
                                obs.vx(*args,**kwargs),
                                obs.vy(*args,**kwargs),
                                obs.vx(*args,**kwargs)]))
        return (X*ro,Y*ro,Z*ro,vX*vo,vY*vo,vZ*vo)

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
        if kwargs.has_key('Omega'):
            Omega= kwargs['Omega']
            kwargs.pop('Omega')
        else:
            Omega= None
        if kwargs.has_key('t'):
            t= kwargs['t']
            kwargs.pop('t')
        thiso= self(*args,**kwargs)
        if not len(thiso.shape) == 2: thiso= thiso.reshape((thiso.shape[0],1))
        if len(thiso[:,0]) < 3:
            raise AttributeError("'linearOrbit has no angular momentum")
        elif len(thiso[:,0]) == 3 or len(thiso[:,0]) == 4:
            if Omega is None:
                return thiso[0,:]*thiso[2,:]
            else:
                return thiso[0,:]*(thiso[2,:]-Omega*t*thiso[0,:])
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
            return out

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

    def _setupaA(self,pot=None,type='adiabatic',**kwargs):
        """
        NAME:
           _setupaA
        PURPOSE:
           set up an actionAngle module for this Orbit
        INPUT:
           pot - potential
           type= ('adiabatic') type of actionAngle module to use
              1) 'adiabatic'
              2) 'staeckel'
              3) 'isochroneApprox'
              4) 'spherical'
        OUTPUT:
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
           2013-11-27 - Re-written in terms of new actionAngle modules - Bovy (IAS)
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
            self._aA= actionAngle.actionAngleStaeckel(pot=self._aAPot,
                                                      **kwargs)
        elif self._aAType.lower() == 'isochroneapprox':
            from galpy.actionAngle_src.actionAngleIsochroneApprox import actionAngleIsochroneApprox
            self._aA= actionAngleIsochroneApprox(pot=self._aAPot,
                                                 **kwargs)
        elif self._aAType.lower() == 'spherical':
            self._aA= actionAngle.actionAngleSpherical(pot=self._aAPot,
                                                       **kwargs)
        return None

    def xw(self,*args,**kwargs):
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

    def plotxw(self,*args,**kwargs):
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
        if isinstance(t,(int,float)) and t in list(self.t):
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
                if nt == 1:
                    return nu.array(out).reshape(dim)
                else:
                    return out
            out= []
            for ii in range(dim):
                out.append(self._orbInterp[ii](t))
            if nt == 1:
                return nu.array(out).reshape(dim)
            else:
                return nu.array(out).reshape((dim,nt))

    def plotE(self,pot,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot E vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        raise NotImplementedError

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot aspects of an Orbit
        INPUT:
           bovy_plot args and kwargs
           +kwargs for ra,dec,ll,bb, etc. functions
        OUTPUT:
           plot
        HISTORY:
           2010-07-26 - Written - Bovy (NYU)
           2010-09-22 - Adapted to more general framework - Bovy (NYU)
           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$',
                    'ra':r'$\alpha\ (\mathrm{deg})$',
                    'dec':r'$\delta\ (\mathrm{deg})$',
                    'll':r'$l\ (\mathrm{deg})$',
                    'bb':r'$b\ (\mathrm{deg})$',
                    'dist':r'$d\ [\mathrm{kpc}]$',
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
                    'W':r'$W\ (\mathrm{km\,s}^{-1})$'}
        #Defaults
        if not kwargs.has_key('d1') and not kwargs.has_key('d2'):
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
        elif not kwargs.has_key('d1'):
            d2= kwargs['d2']
            kwargs.pop('d2')
            d1= 't'
        elif not kwargs.has_key('d2'):
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= 't'
        else:
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= kwargs['d2']
            kwargs.pop('d2')
        #Get x and y
        if d1 == 't':
            x= nu.array(self.t)
        elif d1 == 'R':
            x= self.R(self.t)
        elif d1 == 'z':
            x= self.orbit[:,3]
        elif d1 == 'vz':
            x= self.orbit[:,4]
        elif d1 == 'vR':
            x= self.vR(self.t)
        elif d1 == 'vT':
            x= self.orbit[:,2]
        elif d1 == 'x':
            if len(self.vxvv) == 2:
                x= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d1 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d1 == 'vx':
            x= self.vx(self.t)
        elif d1 == 'vy':
            x= self.vy(self.t)
        elif d1 == 'phi':
            x= self.phi(self.t)
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
        if d2 == 't':
            y= nu.array(self.t)
        elif d2 == 'R':
            y= self.R(self.t)
        elif d2 == 'z':
            y= self.orbit[:,3]
        elif d2 == 'vz':
            y= self.orbit[:,4]
        elif d2 == 'vR':
            y= self.vR(self.t)
        elif d2 == 'vT':
            y= self.orbit[:,2]
        elif d2 == 'x':
            if len(self.vxvv) == 2:
                y= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d2 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d2 == 'vx':
            y= self.vx(self.t)
        elif d2 == 'vy':
            y= self.vy(self.t)
        elif d2 == 'phi':
            y= self.phi(self.t)
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
        if kwargs.has_key('ro'): kwargs.pop('ro')
        if kwargs.has_key('vo'): kwargs.pop('vo')
        if kwargs.has_key('obs'): kwargs.pop('obs')
        #Plot
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= labeldict[d2]
        plot.bovy_plot(x,y,*args,**kwargs)

    def plot3d(self,*args,**kwargs):
        """
        NAME:
           plot3d
        PURPOSE:
           plot 3D aspects of an Orbit
        INPUT:
           bovy_plot args and kwargs
        OUTPUT:
           plot
        HISTORY:
           2010-07-26 - Written - Bovy (NYU)
           2010-09-22 - Adapted to more general framework - Bovy (NYU)
           2010-01-08 - Adapted to 3D - Bovy (NYU)
           2013-11-29 - added ra,dec kwargs and other derived quantities - Bovy (IAS)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$',
                    'ra':r'$\alpha\ (\mathrm{deg})$',
                    'dec':r'$\delta\ (\mathrm{deg})$',
                    'll':r'$l\ (\mathrm{deg})$',
                    'bb':r'$b\ (\mathrm{deg})$',
                    'dist':r'$d\ [\mathrm{kpc}]$',
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
                    'W':r'$W\ (\mathrm{km\,s}^{-1})$'}
        #Defaults
        if not kwargs.has_key('d1') and not kwargs.has_key('d2') \
                and not kwargs.has_key('d3'):
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
        elif not (kwargs.has_key('d1') and kwargs.has_key('d2') \
                      and kwargs.has_key('d3')):
            raise AttributeError("Please provide 'd1', 'd2', and 'd3'")
        else:
            d1= kwargs['d1']
            kwargs.pop('d1')
            d2= kwargs['d2']
            kwargs.pop('d2')
            d3= kwargs['d3']
            kwargs.pop('d3')
        #Get x, y, and z
        if d1 == 't':
            x= nu.array(self.t)
        elif d1 == 'R':
            x= self.orbit[:,0]
        elif d1 == 'z':
            x= self.orbit[:,3]
        elif d1 == 'vz':
            x= self.orbit[:,4]
        elif d1 == 'vR':
            x= self.orbit[:,1]
        elif d1 == 'vT':
            x= self.orbit[:,2]
        elif d1 == 'x':
            if len(self.vxvv) == 2:
                x= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d1 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                x= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d1 == 'vx':
            x= self.vx(self.t)
        elif d1 == 'vy':
            x= self.vy(self.t)
        elif d1 == 'phi':
            x= self.phi(self.t)
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
        if d2 == 't':
            y= nu.array(self.t)
        elif d2 == 'R':
            y= self.orbit[:,0]
        elif d2 == 'z':
            y= self.orbit[:,3]
        elif d2 == 'vz':
            y= self.orbit[:,4]
        elif d2 == 'vR':
            y= self.orbit[:,1]
        elif d2 == 'vT':
            y= self.orbit[:,2]
        elif d2 == 'x':
            if len(self.vxvv) == 2:
                y= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d2 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                y= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d2 == 'vx':
            y= self.vx(self.t)
        elif d2 == 'vy':
            y= self.vy(self.t)
        elif d2 == 'phi':
            y= self.phi(self.t)
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
        if d3 == 't':
            z= nu.array(self.t)
        elif d3 == 'R':
            z= self.orbit[:,0]
        elif d3 == 'z':
            z= self.orbit[:,3]
        elif d3 == 'vz':
            z= self.orbit[:,4]
        elif d3 == 'vR':
            z= self.orbit[:,1]
        elif d3 == 'vT':
            z= self.orbit[:,2]
        elif d3 == 'x':
            if len(self.vxvv) == 2:
                z= self.orbit[:,0]
            elif len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want x you need to track phi")
            elif len(self.vxvv) == 4:
                z= self.orbit[:,0]*nu.cos(self.orbit[:,3])
            else:
                z= self.orbit[:,0]*nu.cos(self.orbit[:,5])                
        elif d3 == 'y':
            if len(self.vxvv) != 4 and len(self.vxvv) != 6:
                raise AttributeError("If you want y you need to track phi")
            elif len(self.vxvv) == 4:
                z= self.orbit[:,0]*nu.sin(self.orbit[:,3])
            else:
                z= self.orbit[:,0]*nu.sin(self.orbit[:,5])                
        elif d3 == 'vx':
            z= self.vx(self.t)
        elif d3 == 'vy':
            z= self.vy(self.t)
        elif d3 == 'phi':
            z= self.phi(self.t)
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
        if kwargs.has_key('ro'): kwargs.pop('ro')
        if kwargs.has_key('vo'): kwargs.pop('vo')
        if kwargs.has_key('obs'): kwargs.pop('obs')
        #Plot
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= labeldict[d2]
        if not kwargs.has_key('zlabel'):
            kwargs['zlabel']= labeldict[d3]
        plot.bovy_plot3d(x,y,z,*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)

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
        self.plot(*args,**kwargs)
        
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
        self.plot(*args,**kwargs)

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
        kwargs['d2']= 'phi'
        self.plot(*args,**kwargs)
        
    def _setupOrbitInterp(self):
        if not hasattr(self,"_orbInterp"):
            orbInterp= []
            for ii in range(len(self.vxvv)):
                if not hasattr(self,"t"): #Orbit has not been integrated
                    orbInterp.append(_fakeInterp(self.vxvv[ii]))
                else:
                    orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                            self.t,self.orbit[:,ii]))
            self._orbInterp= orbInterp
        return None


class _fakeInterp: 
    """Fake class to simulate interpolation when orbit was not integrated"""
    def __init__(self,x):
        self.x= x
    def __call__(self,t):
        return self.x

def _parse_radec_kwargs(kwargs,vel=False,dontpop=False):
    if kwargs.has_key('obs'):
        obs= kwargs['obs']
        if not dontpop:
            kwargs.pop('obs')
        if isinstance(obs,(list,nu.ndarray)):
            if len(obs) == 2:
                obs= [obs[0],obs[1],0.]
            elif len(obs) == 4:
                obs= [obs[0],obs[1],0.,obs[2],obs[3],0.]
    else:
        if vel:
            obs= [8.5,0.,0.,0.,235.,0.]
        else:
            obs= [8.5,0.,0.]
    if kwargs.has_key('ro'):
        ro= kwargs['ro']
        if not dontpop:
            kwargs.pop('ro')
    else:
        ro= 8.5
    if kwargs.has_key('vo'):
        vo= kwargs['vo']
        if not dontpop:
            kwargs.pop('vo')
    else:
        vo= 235.
    return (obs,ro,vo)

def _BCZeroFunction(t,vxvv,pot,method,bc,to,BCIntegrateFunc):
    if t == to: return bc(vxvv)
    #Determine number of ts
    nts= int(nu.ceil(t-to))+1 #very simple estimate
    tin= nu.linspace(to,t,nts)
    orb= BCIntegrateFunc(vxvv,pot,tin,method)
    return bc(orb[nts-1,:])
