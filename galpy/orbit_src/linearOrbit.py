import numpy as nu
from scipy import integrate
from Orbit import Orbit
from galpy.potential_src.linearPotential import evaluateForces
import galpy.util.bovy_plot as plot
class linearOrbit(Orbit):
    """Class that represents an orbit in a (effectively) one-dimensional potential"""
    def __init__(self,vxvv=[1.,0.]):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a linear orbit
        INPUT:
           vxvv - [x,vx]
        OUTPUT:
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        self.vxvv= vxvv
        return None

    def integrate(self,t,pot):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        self.t= nu.array(t)
        self.orbit= _integrateLinearOrbit(self.vxvv,pot,t)

    def plot(self,*args,**kwargs):
        """
        NAME:
           plot
        PURPOSE:
           plot a previously calculated orbit
        INPUT:
           matplotlib.plot inputs+bovy_plot.plot inputs
        OUTPUT:
           sends plot to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        plot.bovy_plot(self.orbit[:,0],self.orbit[:,1],*args,**kwargs)

def _integrateLinearOrbit(vxvv,pot,t):
    """
    NAME:
       integrateLinearOrbit
    PURPOSE:
       integrate a one-dimensional orbit
    INPUT:
       vxvv - initial condition [x,vx]
       pot - linearPotential or list of linearPotentials
       t - list of times at which to output (0 has to be in this!)
    OUTPUT:
       [:,2] array of [x,vx] at each t
    HISTORY:
       2010-07-13- Written - Bovy (NYU)
    """
    return integrate.odeint(_linearEOM,vxvv,t,args=(pot,),rtol=10.**-8.)

def _linearEOM(y,t,pot):
    """
    NAME:
       linearEOM
    PURPOSE:
       the one-dimensional equation-of-motion
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) linearPotential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2010-07-13 - Bovy (NYU)
    """
    return [y[1],evaluateForces(y[0],pot)]
