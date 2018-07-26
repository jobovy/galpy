import warnings
import math as m
import numpy as nu
from scipy import integrate
from galpy.potential.Potential import _evaluateRforces, _evaluatezforces,\
    evaluatePotentials, evaluateDensities, _check_c
from galpy.util import galpyWarning
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
from .FullOrbit import _integrateFullOrbit
from .integrateFullOrbit import _ext_loaded as ext_loaded
from galpy.util.bovy_conversion import physical_conversion
from .OrbitTop import OrbitTop
class RZOrbit(OrbitTop):
    """Class that holds and integrates orbits in axisymetric potentials 
    in the (R,z) plane"""
    def __init__(self,vxvv=[1.,0.,0.9,0.,0.1],vo=220.,ro=8.0,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])):
        """
        NAME:

           __init__

        PURPOSE:

           intialize an RZ-orbit

        INPUT:

           vxvv - initial condition [R,vR,vT,z,vz]

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

           (none)

        HISTORY:

           2010-07-10 - Written - Bovy (NYU)

           2014-06-11 - Added conversion kwargs to physical coordinates - Bovy (IAS)

        """
        OrbitTop.__init__(self,vxvv=vxvv,
                          ro=ro,zo=zo,vo=vo,solarmotion=solarmotion)
        return None

    def integrate(self,t,pot,method='symplec4_c',dt=None):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint' for scipy's odeint
                   'leapfrog' for a simple leapfrog implementation
                   'leapfrog_c' for a simple leapfrog implementation in C
                   'rk4_c' for a 4th-order Runge-Kutta integrator in C
                   'rk6_c' for a 6-th order Runge-Kutta integrator in C
                   'dopr54_c' for a Dormand-Prince integrator in C (generally the fastest)
           dt= (None) if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-10
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        self.t= nu.array(t)
        self._pot= pot
        self.orbit= _integrateRZOrbit(self.vxvv,pot,t,method,dt)

    @physical_conversion('energy')
    def E(self,*args,**kwargs):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           t - (optional) time at which to get the radius
           pot= RZPotential instance or list thereof
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        if not 'pot' in kwargs or kwargs['pot'] is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
            if 'pot' in kwargs and kwargs['pot'] is None:
                kwargs.pop('pot')          
        else:
            pot= kwargs.pop('pot')
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(pot,thiso[0],thiso[3],
                                      t=t,use_physical=False)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.\
                                      +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(pot,thiso[0,ii],thiso[3,ii],
                                                t=t[ii],use_physical=False)\
                                 +thiso[1,ii]**2./2.\
                                 +thiso[2,ii]**2./2.\
                                 +thiso[4,ii]**2./2. for ii in range(len(t))])

    @physical_conversion('energy')
    def ER(self,*args,**kwargs):
        """
        NAME:
           ER
        PURPOSE:
           calculate the radial energy
        INPUT:
           t - (optional) time at which to get the energy
           pot= potential instance or list of such instances
        OUTPUT:
           radial energy
        HISTORY:
           2013-11-30 - Written - Bovy (IAS)
        """
        if not 'pot' in kwargs or kwargs['pot'] is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
            if 'pot' in kwargs and kwargs['pot'] is None:
                kwargs.pop('pot')          
        else:
            pot= kwargs.pop('pot')
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(pot,thiso[0],0.,
                                      t=t,use_physical=False)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.
        else:
            return nu.array([evaluatePotentials(pot,thiso[0,ii],0.,
                                                t=t[ii],use_physical=False)\
                                 +thiso[1,ii]**2./2.\
                                 +thiso[2,ii]**2./2. for ii in range(len(t))])

    @physical_conversion('energy')
    def Ez(self,*args,**kwargs):
        """
        NAME:
           Ez
        PURPOSE:
           calculate the vertical energy
        INPUT:
           t - (optional) time at which to get the energy
           pot= potential instance or list of such instances
        OUTPUT:
           vertical energy
        HISTORY:
           2013-11-30 - Written - Bovy (IAS)
        """
        if not 'pot' in kwargs or kwargs['pot'] is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
            if 'pot' in kwargs and kwargs['pot'] is None:
                kwargs.pop('pot')          
        else:
            pot= kwargs.pop('pot')
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(pot,thiso[0],thiso[3],
                                      t=t,use_physical=False)\
                                      -evaluatePotentials(pot,thiso[0],0.,
                                                          t=t,
                                                          use_physical=False)\
                                                          +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(pot,thiso[0,ii],thiso[3,ii],
                                                t=t[ii],use_physical=False)\
                                 -evaluatePotentials(pot,thiso[0,ii],0.,
                                                t=t[ii],use_physical=False)\
                                 +thiso[4,ii]**2./2. for ii in range(len(t))])

    @physical_conversion('energy')
    def Jacobi(self,*args,**kwargs):
        """
        NAME:
           Jacobi
        PURPOSE:
           calculate the Jacobi integral of the motion
        INPUT:
           t - (optional) time at which to get the radius
           OmegaP= pattern speed of rotating frame (scalar)
           pot= potential instance or list of such instances
        OUTPUT:
           Jacobi integral
        HISTORY:
           2011-04-18 - Written - Bovy (NYU)
        """
        if not 'OmegaP' in kwargs or kwargs['OmegaP'] is None:
            OmegaP= 1.
            if not 'pot' in kwargs or kwargs['pot'] is None:
                try:
                    pot= self._pot
                except AttributeError:
                    raise AttributeError("Integrate orbit or specify pot=")
            else:
                pot= kwargs['pot']
            if isinstance(pot,list):
                for p in pot:
                    if hasattr(p,'OmegaP'):
                        OmegaP= p.OmegaP()
                        break
            else:
                if hasattr(pot,'OmegaP'):
                    OmegaP= pot.OmegaP()
            kwargs.pop('OmegaP',None)
        else:
            OmegaP= kwargs.pop('OmegaP')
        #Make sure you are not using physical coordinates
        old_physical= kwargs.get('use_physical',None)
        kwargs['use_physical']= False
        thiso= self(*args,**kwargs)
        out= self.E(*args,**kwargs)-OmegaP*thiso[0]*thiso[2]
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    def e(self,analytic=False,pot=None,**kwargs):
        """
        NAME:
           e
        PURPOSE:
           calculate the eccentricity
        INPUT:
           analytic - compute this analytically
           pot - potential to use for analytical calculation
        OUTPUT:
           eccentricity
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
        """
        if analytic:
            self._setupaA(pot=pot,**kwargs)
            return float(self._aA.EccZmaxRperiRap(self)[0])
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

    @physical_conversion('position')
    def rap(self,analytic=False,pot=None,**kwargs):
        """
        NAME:
           rap
        PURPOSE:
           return the apocenter radius
        INPUT:
           analytic - compute this analytically
           pot - potential to use for analytical calculation
        OUTPUT:
           R_ap
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        if analytic:
            self._setupaA(pot=pot,**kwargs)
            return float(self._aA.EccZmaxRperiRap(self)[3])
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate rap")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return nu.amax(self.rs)

    @physical_conversion('position')
    def rperi(self,analytic=False,pot=None,**kwargs):
        """
        NAME:
           rperi
        PURPOSE:
           return the pericenter radius
        INPUT:
           analytic - compute this analytically
           pot - potential to use for analytical calculation
        OUTPUT:
           R_peri
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
        """
        if analytic:
            self._setupaA(pot=pot,**kwargs)
            return float(self._aA.EccZmaxRperiRap(self)[2])
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate rperi")
        if not hasattr(self,'rs'):
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return nu.amin(self.rs)

    @physical_conversion('position')
    def zmax(self,analytic=False,pot=None,**kwargs):
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
        if analytic:
            self._setupaA(pot=pot,**kwargs)
            return float(self._aA.EccZmaxRperiRap(self)[1])
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate zmax")
        return nu.amax(nu.fabs(self.orbit[:,3]))

    def plotEz(self,*args,**kwargs):
        """
        NAME:
           plotEz
        PURPOSE:
           plot Ez(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2014-06-16 - Written - Bovy (IAS)
        """
        if kwargs.pop('normed',False):
            kwargs['d2']= 'Eznorm'
        else:
            kwargs['d2']= 'Ez'
        return self.plot(*args,**kwargs)
        
    def plotER(self,*args,**kwargs):
        """
        NAME:
           plotER
        PURPOSE:
           plot ER(.) along the orbit
        INPUT:
           bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2014-06-16 - Written - Bovy (IAS)
        """
        if kwargs.pop('normed',False):
            kwargs['d2']= 'ERnorm'
        else:
            kwargs['d2']= 'ER'
        return self.plot(*args,**kwargs)
        
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
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$'}
        if not 'pot' in kwargs:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit first or specify pot=")
        else:
            pot= kwargs.pop('pot')
        d1= kwargs.pop('d1','t')
        self.EzJz= [(evaluatePotentials(pot,self.orbit[ii,0],self.orbit[ii,3],
                                        t=self.t[ii],use_physical=False)-
                     evaluatePotentials(pot,self.orbit[ii,0],0.,t=self.t[ii],
                                        use_physical=False)+
                     self.orbit[ii,4]**2./2.)/\
                        nu.sqrt(evaluateDensities(pot,self.orbit[ii,0],0.,
                                                  t=self.t[ii],
                                                  use_physical=False))\
                        for ii in range(len(self.t))]
        if not 'xlabel' in kwargs:
            kwargs['xlabel']= labeldict[d1]
        if not 'ylabel' in kwargs:
            kwargs['ylabel']= r'$E_z/\sqrt{\rho}$'
        if d1 == 't':
            return plot.bovy_plot(nu.array(self.t),
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)
        elif d1 == 'z':
            return plot.bovy_plot(self.orbit[:,3],
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)
        elif d1 == 'R':
            return plot.bovy_plot(self.orbit[:,0],
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)
        elif d1 == 'vR':
            return plot.bovy_plot(self.orbit[:,1],
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)
        elif d1 == 'vT':
            return plot.bovy_plot(self.orbit[:,2],
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)
        elif d1 == 'vz':
            return plot.bovy_plot(self.orbit[:,4],
                                  nu.array(self.EzJz)/self.EzJz[0],
                                  *args,**kwargs)

def _integrateRZOrbit(vxvv,pot,t,method,dt):
    """
    NAME:
       _integrateRZOrbit
    PURPOSE:
       integrate an orbit in a Phi(R,z) potential in the (R,z) plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT,z,vz]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
       method - 'odeint' or 'leapfrog'
       dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize
    OUTPUT:
       [:,5] array of [R,vR,vT,z,vz] at each t
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    #First check that the potential has C
    if '_c' in method:
        if not ext_loaded or not _check_c(pot):
            if ('leapfrog' in method or 'symplec' in method):
                method= 'leapfrog'
            else:
                method= 'odeint'
            if not ext_loaded: # pragma: no cover
                warnings.warn("Cannot use C integration because C extension not loaded (using %s instead)" % (method), galpyWarning)
            else:
                warnings.warn("Cannot use C integration because some of the potentials are not implemented in C (using %s instead)" % (method), galpyWarning)
    if method.lower() == 'leapfrog' \
            or method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c':
        #We hack this by upgrading to a FullOrbit
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out= _integrateFullOrbit(this_vxvv,pot,t,method,dt)
        #tmp_out is (nt,6)
        out= tmp_out[:,0:5]
    elif method.lower() == 'odeint':
        l= vxvv[0]*vxvv[2]
        l2= l**2.
        init= [vxvv[0],vxvv[1],vxvv[3],vxvv[4]]
        intOut= integrate.odeint(_RZEOM,init,t,args=(pot,l2),
                                 rtol=10.**-8.)#,mxstep=100000000)
        out= nu.zeros((len(t),5))
        out[:,0]= intOut[:,0]
        out[:,1]= intOut[:,1]
        out[:,3]= intOut[:,2]
        out[:,4]= intOut[:,3]
        out[:,2]= l/out[:,0]
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    return out

def _RZEOM(y,t,pot,l2):
    """
    NAME:
       _RZEOM
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
            l2/y[0]**3.+_evaluateRforces(pot,y[0],y[2],t=t),
            y[3],
            _evaluatezforces(pot,y[0],y[2],t=t)]

