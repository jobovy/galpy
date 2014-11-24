import math as m
import numpy as nu
from scipy import integrate
from galpy.potential_src.Potential import evaluateRforces, evaluatezforces,\
    evaluatePotentials, evaluateDensities
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
from galpy.orbit_src.FullOrbit import _integrateFullOrbit
from galpy.util.bovy_conversion import physical_conversion
from OrbitTop import OrbitTop
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

    def integrate(self,t,pot,method='symplec4_c'):
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
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-10
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        self.t= nu.array(t)
        self._pot= pot
        self.orbit= _integrateRZOrbit(self.vxvv,pot,t,method)

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
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(thiso[0],thiso[3],pot,
                                      t=t)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.\
                                      +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],thiso[3,ii],
                                                pot,
                                                t=t[ii])\
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
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(thiso[0],0.,pot,
                                      t=t)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],0.,
                                                pot,
                                                t=t[ii])\
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
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return evaluatePotentials(thiso[0],thiso[3],pot,
                                      t=t)\
                                      -evaluatePotentials(thiso[0],0.,pot,
                                                          t=t)\
                                                          +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],thiso[3,ii],
                                                pot,
                                                t=t[ii])\
                                 -evaluatePotentials(thiso[0,ii],0.,
                                                     pot,
                                                t=t[ii])\
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
        if not kwargs.has_key('OmegaP') or kwargs['OmegaP'] is None:
            OmegaP= 1.
            if not kwargs.has_key('pot') or kwargs['pot'] is None:
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
            if kwargs.has_key('OmegaP'):
                kwargs.pop('OmegaP')         
        else:
            OmegaP= kwargs['OmegaP']
            kwargs.pop('OmegaP')
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

    def e(self,analytic=False,pot=None):
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
            self._setupaA(pot=pot,type='adiabatic')
            (rperi,rap)= self._aA.calcRapRperi(self)
            return (rap-rperi)/(rap+rperi)
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
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
            self._setupaA(pot=pot,type='adiabatic')
            (rperi,rap)= self._aA.calcRapRperi(self)
            return rap
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
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
            self._setupaA(pot=pot,type='adiabatic')
            (rperi,rap)= self._aA.calcRapRperi(self)
            return rperi
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
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
            self._setupaA(pot=pot,type='adiabatic')
            zmax= self._aA.calczmax(self)
            return zmax
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
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
        if kwargs.get('normed',False):
            kwargs['d2']= 'Eznorm'
        else:
            kwargs['d2']= 'Ez'
        if kwargs.has_key('normed'): kwargs.pop('normed')
        self.plot(*args,**kwargs)
        
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
        if kwargs.get('normed',False):
            kwargs['d2']= 'ERnorm'
        else:
            kwargs['d2']= 'ER'
        if kwargs.has_key('normed'): kwargs.pop('normed')
        self.plot(*args,**kwargs)
        
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
        self.EzJz= [(evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],
                                        pot,t=self.t[ii])-
                     evaluatePotentials(self.orbit[ii,0],0.,pot,t=self.t[ii])+
                     self.orbit[ii,4]**2./2.)/\
                        nu.sqrt(evaluateDensities(self.orbit[ii,0],0.,pot,
                                                  t=self.t[ii]))\
                        for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E_z/\sqrt{\rho}$'
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

def _integrateRZOrbit(vxvv,pot,t,method):
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
    OUTPUT:
       [:,5] array of [R,vR,vT,z,vz] at each t
    HISTORY:
       2010-04-16 - Written - Bovy (NYU)
    """
    #First check that the potential has C
    if '_c' in method:
        if isinstance(pot,list):
            allHasC= nu.prod([p.hasC for p in pot])
        else:
            allHasC= pot.hasC
        if not allHasC and ('leapfrog' in method or 'symplec' in method):
            method= 'leapfrog'
        elif not allHasC:
            method= 'odeint'
    if method.lower() == 'leapfrog' \
            or method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c':
        #We hack this by upgrading to a FullOrbit
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out= _integrateFullOrbit(this_vxvv,pot,t,method)
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
            l2/y[0]**3.+evaluateRforces(y[0],y[2],pot,t=t),
            y[3],
            evaluatezforces(y[0],y[2],pot,t=t)]

