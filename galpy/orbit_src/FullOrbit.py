import math as m
import numpy as nu
from scipy import integrate
from galpy.potential_src.Potential import evaluateRforces, evaluatezforces,\
    evaluatePotentials, evaluatephiforces, evaluateDensities
import galpy.util.bovy_plot as plot
from OrbitTop import OrbitTop
class FullOrbit(OrbitTop):
    """Class that holds and integrates orbits in full 3D potentials"""
    def __init__(self,vxvv=[1.,0.,0.9,0.,0.1]):
        """
        NAME:
           __init__
        PURPOSE:
           intialize a full orbit
        INPUT:
           vxvv - initial condition [R,vR,vT,z,vz,phi]
        OUTPUT:
           (none)
        HISTORY:
           2010-08-01 - Written - Bovy (NYU)
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
           2010-08-01 - Written - Bovy (NYU)
        """
        self.t= nu.array(t)
        self._pot= pot
        self.orbit= _integrateFullOrbit(self.vxvv,pot,t)

    def E(self,pot=None):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        return evaluatePotentials(self.vxvv[0],self.vxvv[3],pot,
                                  phi=self.vxvv[5])+\
            self.vxvv[1]**2./2.+self.vxvv[2]**2./2.+\
            self.vxvv[4]**2./2.

    def e(self):
        """
        NAME:
           e
        PURPOSE:
           calculate the eccentricity
        INPUT:
        OUTPUT:
           eccentricity
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

    def rap(self):
        """
        NAME:
           rap
        PURPOSE:
           return the apocenter radius
        INPUT:
        OUTPUT:
           R_ap
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return nu.amax(self.rs)

    def rperi(self):
        """
        NAME:
           rperi
        PURPOSE:
           return the pericenter radius
        INPUT:
        OUTPUT:
           R_peri
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return nu.amin(self.rs)

    def zmax(self):
        """
        NAME:
           zmax
        PURPOSE:
           return the maximum vertical height
        INPUT:
        OUTPUT:
           Z_max
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        return nu.amax(nu.fabs(self.orbit[:,3]))

    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot= - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if not kwargs.has_key('pot'):
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit first or specify pot=")
        else:
            pot= kwargs['pot']
            kwargs.pop('pot')
        if kwargs.has_key('d1'):
            d1= kwargs['d1']
            kwargs.pop('d1')
        else:
            d1= 't'
        self.Es= [evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],
                                     pot,phi=self.orbit[ii,5])+
                  self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.+
                  self.orbit[ii,4]**2./2. for ii in range(len(self.t))]
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'z':
            plot.bovy_plot(self.orbit[:,3],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'R':
            plot.bovy_plot(self.orbit[:,0],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'vR':
            plot.bovy_plot(self.orbit[:,1],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'vT':
            plot.bovy_plot(self.orbit[:,2],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'vz':
            plot.bovy_plot(self.orbit[:,4],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'phi':
            plot.bovy_plot(self.orbit[:,5],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)

    def plotEz(self,*args,**kwargs):
        """
        NAME:
           plotEz
        PURPOSE:
           plot E_z(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        if not kwargs.has_key('pot'):
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit first or specify pot=")
        else:
            pot= kwargs['pot']
            kwargs.pop('pot')
        if kwargs.has_key('d1'):
            d1= kwargs['d1']
            kwargs.pop('d1')
        else:
            d1= 't'
        self.Ezs= [evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],
                                      pot,phi=self.orbit[ii,5])-
                   evaluatePotentials(self.orbit[ii,0],0.,pot,
                                      phi=self.orbit[ii,5])+
                  self.orbit[ii,4]**2./2. for ii in range(len(self.t))]
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'z':
            plot.bovy_plot(self.orbit[:,3],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'R':
            plot.bovy_plot(self.orbit[:,0],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'vR':
            plot.bovy_plot(self.orbit[:,1],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'vT':
            plot.bovy_plot(self.orbit[:,2],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'vz':
            plot.bovy_plot(self.orbit[:,4],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
        elif d1 == 'phi':
            plot.bovy_plot(self.orbit[:,5],nu.array(self.Ezs)/self.Ezs[0],
                           *args,**kwargs)
    def plotEzJz(self,*args,**kwargs):
        """
        NAME:
           plotEzJz
        PURPOSE:
           plot E_z(.)/sqrt(dens(R)) along the orbit
        INPUT:
           pot= Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-08-08 - Written - Bovy (NYU)
        """
        if not kwargs.has_key('pot'):
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit first or specify pot=")
        else:
            pot= kwargs['pot']
            kwargs.pop('pot')
        if kwargs.has_key('d1'):
            d1= kwargs['d1']
            kwargs.pop('d1')
        else:
            d1= 't'
        self.EzJz= [(evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],pot)-
                     evaluatePotentials(self.orbit[ii,0],0.,pot,
                                        phi= self.orbit[ii,5])+
                     self.orbit[ii,4]**2./2.)/\
                        nu.sqrt(evaluateDensities(self.orbit[ii,0],0.,pot,phi=self.orbit[ii,5]))\
                        for ii in range(len(self.t))]
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)
        elif d1 == 'z':
            plot.bovy_plot(self.orbit[:,3],nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)
        elif d1 == 'R':
            plot.bovy_plot(self.orbit[:,0],nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)
        elif d1 == 'vR':
            plot.bovy_plot(self.orbit[:,1],nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)
        elif d1 == 'vT':
            plot.bovy_plot(self.orbit[:,2],nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)
        elif d1 == 'vz':
            plot.bovy_plot(self.orbit[:,4],nu.array(self.EzJz)/self.EzJz[0],
                           *args,**kwargs)

    def _callRect(self,*args):
        kwargs= {}
        kwargs['rect']= False
        vxvv= self.__call__(*args,**kwargs)
        x= vxvv[0]*m.cos(vxvv[5])
        y= vxvv[0]*m.sin(vxvv[5])
        vx= vxvv[1]*m.cos(vxvv[5])-vxvv[2]*m.sin(vxvv[5])
        vy= -vxvv[1]*m.sin(vxvv[5])-vxvv[2]*m.cos(vxvv[5])
        return nu.array([x,y,vxvv[3],vx,vy,vxvv[4]])

def _integrateFullOrbit(vxvv,pot,t):
    """
    NAME:
       _integrateFullOrbit
    PURPOSE:
       integrate an orbit in a Phi(R,z,phi) potential
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT,z,vz,phi]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
    OUTPUT:
       [:,5] array of [R,vR,vT,z,vz,phi] at each t
    HISTORY:
       2010-08-01 - Written - Bovy (NYU)
    """
    vphi= vxvv[2]/vxvv[0]
    init= [vxvv[0],vxvv[1],vxvv[5],vphi,vxvv[3],vxvv[4]]
    intOut= integrate.odeint(_FullEOM,init,t,args=(pot,),
                             rtol=10.**-8.)#,mxstep=100000000)
    out= nu.zeros((len(t),6))
    out[:,0]= intOut[:,0]
    out[:,1]= intOut[:,1]
    out[:,2]= out[:,0]*intOut[:,3]
    out[:,3]= intOut[:,4]
    out[:,4]= intOut[:,5]
    out[:,5]= intOut[:,2]
    return out

def _FullEOM(y,t,pot):
    """
    NAME:
       _FullEOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential 
       equation
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    l2= (y[0]**2.*y[3])**2.
    return [y[1],
            l2/y[0]**3.+evaluateRforces(y[0],y[4],pot,phi=y[2],t=t),
            y[3],
            1./y[0]**2.*(evaluatephiforces(y[0],y[4],pot,phi=y[2],t=t)-
                         2.*y[0]*y[1]*y[3]),
            y[5],
            evaluatezforces(y[0],y[4],pot,phi=y[2],t=t)]
