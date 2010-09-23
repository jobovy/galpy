import math as m
import numpy as nu
from scipy import integrate, interpolate
import galpy.util.bovy_plot as plot
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

    def integrate(self,t,pot):
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
        raise AttributeError

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
        return thiso[0]

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
        return thiso[1]

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
        return thiso[2]

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
        thiso= self(*args,**kwargs)
        if len(thiso) < 5:
            raise AttributeError("linear and planar orbits do not have z()")
        return thiso[3]

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
        thiso= self(*args,**kwargs)
        if len(thiso) < 5:
            raise AttributeError("linear and planar orbits do not have vz()")
        return thiso[4]

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
        thiso= self(*args,**kwargs)
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use phi()")
        elif len(thiso) == 4:
            return thiso[3]
        else:
            return thiso[5]

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
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use x()")
        elif len(thiso) == 4:
            return thiso[0]*m.cos(thiso[3])
        else:
            return thiso[0]*m.cos(thiso[5])

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
        if len(thiso) != 4 and len(thiso) != 6:
            raise AttributeError("orbit must track azimuth to use y()")
        elif len(thiso) == 4:
            return thiso[0]*m.sin(thiso[3])
        else:
            return thiso[0]*m.sin(thiso[5])

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
        return thiso[2]/thiso[0]

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           return the orbit vector at time t
        INPUT:
           t - desired time
           rect - if true, return rectangular coordinates
        OUTPUT:
           [R,vR,vT,z,vz(,phi)] or [R,vR,vT(,phi)] depending on the orbit
        BUGS:
           currently only works for times at which the orbit was requested
           during integration; use interpolation in between?
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        #Options
        if kwargs.has_key('rect'):
            rect= kwargs['rect']
        else:
            rect= False
        if rect:
            return self._callRect(*args)
        if len(args) == 0:
            return self.vxvv
        else:
            t= args
        if t in list(self.t):
            return self.orbit[list(self.t).index(t),:]
        else:
            dim= len(self.vxvv)
            if not hasattr(self,"_orbInterp"):
                orbInterp= []
                for ii in range(dim):
                    orbInterp.append(interpolate.InterpolatedUnivariateSpline(\
                            self.t,self.orbit[:,ii]))
                self._orbInterp= orbInterp
            out= []
            for ii in range(dim):
                out.append(self._orbInterp[ii](t))
            return nu.array(out).reshape(dim)

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
        raise AttributeError

    def plotRt(self,*args,**kwargs):
        """
        NAME:
           plotRt
        PURPOSE:
           plot R(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(nu.array(self.t),self.orbit[:,0],*args,**kwargs)

    def plotzt(self,*args,**kwargs):
        """
        NAME:
           plotzt
        PURPOSE:
           plot z(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(nu.array(self.t),self.orbit[:,3],*args,**kwargs)

    def plotvRt(self,*args,**kwargs):
        """
        NAME:
           plotvRt
        PURPOSE:
           plot vR(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(nu.array(self.t),self.orbit[:,1],*args,**kwargs)

    def plotvTt(self,*args,**kwargs):
        """
        NAME:
           plotvTt
        PURPOSE:
           plot vT(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(nu.array(self.t),self.orbit[:,2],*args,**kwargs)

    def plotphit(self,*args,**kwargs):
        """
        NAME:
           plotphit
        PURPOSE:
           plot \phi(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if self.orbit.shape[1] != 4 and self.orbit.shape[1] != 6:
            raise AttributeError
        elif self.orbit.shape[1] == 4:
            plot.bovy_plot(nu.array(self.t),self.orbit[:,3],*args,**kwargs)
        else:
            plot.bovy_plot(nu.array(self.t),self.orbit[:,5],*args,**kwargs)

    def plotvzt(self,*args,**kwargs):
        """
        NAME:
           plotvzt
        PURPOSE:
           plot vz(t) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(nu.array(self.t),self.orbit[:,4],*args,**kwargs)

