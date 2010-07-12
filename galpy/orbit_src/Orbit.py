import numpy as nu
from scipy import integrate
from galpy.potential import evaluateRforces, evaluatezforces, evaluatePotentials
import galpy.util.bovy_plot as plot
class Orbit:
    """General class that holds orbits and integrates them"""
    def __init__(self,vxvv):
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

    def __call__(self,t):
        """
        NAME:
           __call__
        PURPOSE:
           return the orbit vector at time t
        INPUT:
           t - desired time
        OUTPUT:
           [R,vR,vT,z,vz(,phi)]
        BUGS:
           currently only works for times at which the orbit was requested
           during integration; use interpolation in between?
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        return self.orbit[list(self.t).index(t),:]

    def plotEt(self,pot,*args,**kwargs):
        """
        NAME:
           plotEt
        PURPOSE:
           plot E(t) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
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
        if self.orbit.shape[1] < 6:
            raise AttributeError
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

