###############################################################################
#   integrateOrbit: integrate orbits in axisymmetric potentials
###############################################################################
import numpy as nu
from scipy import integrate
from Potential import evaluateRforces, evaluatezforces, evaluatePotentials
import bovy_plot as plot
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

class RZOrbit(Orbit):
    """Class that holds and integrates orbits in axisymetric potentials 
    in the (R,z) plane"""
    def __init__(self,vxvv=[1.,0.,0.9,0.,0.1]):
        """
        NAME:
           __init__
        PURPOSE:
           intialize an RZ-orbit
        INPUT:
           vxvv - initial condition [R,vR,vT,z,vz]
        OUTPUT:
           (none)
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
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
           2010-07-10
        """
        self.t= t
        self.orbit= integrateRZOrbit(self.vxvv,pot,t)

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
        plot.bovy_plot(self.orbit[:,0],self.orbit[:,3],*args,**kwargs)

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
        self.E= [evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],pot)+
                 self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.+
                 self.orbit[ii,4]**2./2. for ii in range(len(self.t))]
        plot.bovy_plot(nu.array(self.t),nu.array(self.E)/self.E[0],
                       *args,**kwargs)

def integrateRZOrbit(vxvv,pot,t):
    """
    NAME:
       integrateRZOrbit
    PURPOSE:
       integrate an orbit in a Phi(R,z) potential in the (R,z) plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT,z,vz]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
    OUTPUT:
       [:,5] array of [R,vR,vT,z,vz] at each t
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    l= vxvv[0]*vxvv[2]
    l2= l**2.
    init= [vxvv[0],vxvv[1],vxvv[3],vxvv[4]]
    intOut= integrate.odeint(RZEOM,init,t,args=(pot,l2),
                             rtol=10.**-8.)#,mxstep=100000000)
    out= nu.zeros((len(t),5))
    out[:,0]= intOut[:,0]
    out[:,1]= intOut[:,1]
    out[:,3]= intOut[:,2]
    out[:,4]= intOut[:,3]
    out[:,2]= l/out[:,0]
    return out

def RZEOM(y,t,pot,l2):
    """
    NAME:
       RZEOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential 
       equation
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
       l2 - angular momentum squared
    OUTPUT:
       dy/dt
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    return [y[1],
            l2/y[0]**3.+evaluateRforces(y[0],y[2],pot),
            y[3],
            evaluatezforces(y[0],y[2],pot)]

if __name__ == '__main__':
    #from DoubleExponentialDiskPotential import DoubleExponentialDiskPotential
    #pot= DoubleExponentialDiskPotential()
    from MiyamotoNagaiPotential import MiyamotoNagaiPotential
    pot= MiyamotoNagaiPotential(a=1.,b=0.2)
    rhoo= 1./nu.fabs(pot.Rforce(1.,0))
    pot= MiyamotoNagaiPotential(a=1.,b=0.2,amp=rhoo)  
    #pot= DoubleExponentialDiskPotential(rhoo=rhoo) #Not quite what we want?
    t= nu.arange(11)/10.*nu.pi*2.

    vxvv= [1.,0.05,1.,0.,0.05]
    out= integrateRZOrbit(vxvv,pot,t)
    print out
