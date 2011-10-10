import numpy as nu
from scipy import integrate
from OrbitTop import OrbitTop
from galpy.potential_src.linearPotential import evaluatelinearForces,\
    evaluatelinearPotentials
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
class linearOrbit(OrbitTop):
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
        #For boundary-condition integration
        self._BCIntegrateFunction= _integrateLinearOrbit
        return None

    def integrate(self,t,pot,method='odeint'):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint'= scipy's odeint, or 'leapfrog'
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-13 - Written - Bovy (NYU)
        """
        if method == 'leapfrog_c': method= 'odeint'
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        self.t= nu.array(t)
        self._pot= pot
        self.orbit= _integrateLinearOrbit(self.vxvv,pot,t,method)

    def E(self,*args,**kwargs):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           t - (optional) time at which to get the radius
           pot= linearPotential instance or list thereof
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        if not kwargs.has_key('pot') or kwargs['pot'] is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
            if kwargs.has_key('pot') and kwargs['pot'] is None:
                kwargs.pop('pot')          
        else:
            pot= kwargs['pot']
            kwargs.pop('pot')
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatelinearPotentials(thiso[0],thispot,
                                            t=t)\
                                            +thiso[1]**2./2.
        else:
            return nu.array([evaluatelinearPotentials(thiso[0,ii],thispot,
                                                      t=t[ii])\
                                 +thiso[1,ii]**2./2.\
                                 for ii in range(len(t))])

    def e(self,analytic=False,pot=None):
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
        raise AttributeError("linearOrbit does not have an eccentricity")

    def rap(self,analytic=False,pot=None):
        raise AttributeError("linearOrbit does not have an apocenter")

    def rperi(self,analytic=False,pot=None):
        raise AttributeError("linearOrbit does not have a pericenter")

    def zmax(self):
        raise AttributeError("linearOrbit does not have a zmax")

    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'x', 'vx'
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2010-07-10 - Written - Bovy (NYU)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$'}
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
        self.Es= [evaluatelinearPotentials(self.orbit[ii,0],pot,t=self.t[ii])+
                 self.orbit[ii,1]**2./2.
                 for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E$'
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'x':
            plot.bovy_plot(self.orbit[:,0],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)
        elif d1 == 'vx':
            plot.bovy_plot(self.orbit[:,1],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)

    def _callRect(self,*args):
        kwargs['rect']= False
        vxvv= self.__call__(*args,**kwargs)     

def _integrateLinearOrbit(vxvv,pot,t,method):
    """
    NAME:
       integrateLinearOrbit
    PURPOSE:
       integrate a one-dimensional orbit
    INPUT:
       vxvv - initial condition [x,vx]
       pot - linearPotential or list of linearPotentials
       t - list of times at which to output (0 has to be in this!)
       method - 'odeint' or 'leapfrog'
    OUTPUT:
       [:,2] array of [x,vx] at each t
    HISTORY:
       2010-07-13- Written - Bovy (NYU)
    """
    if method.lower() == 'leapfrog':
        return symplecticode.leapfrog(evaluatelinearForces,nu.array(vxvv),
                                      t,args=(pot,),rtol=10.**-8)
    elif method.lower() == 'odeint':
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
    return [y[1],evaluatelinearForces(y[0],pot,t=t)]
