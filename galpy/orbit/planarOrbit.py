import math as m
import warnings
import numpy as nu
from scipy import integrate
import galpy.util.bovy_symplecticode as symplecticode
from galpy.util.bovy_conversion import physical_conversion
from .OrbitTop import OrbitTop
from galpy.potential.planarPotential import _evaluateplanarRforces,\
    RZToplanarPotential, toPlanarPotential, _evaluateplanarphiforces,\
    _evaluateplanarPotentials
from galpy.potential.Potential import Potential, _check_c
from galpy.util import galpyWarning, galpyWarningVerbose
#try:
from .integratePlanarOrbit import integratePlanarOrbit_c,\
    integratePlanarOrbit_dxdv_c, _ext_loaded
ext_loaded= _ext_loaded
class planarOrbitTop(OrbitTop):
    """Top-level class representing a planar orbit (i.e., one in the plane 
    of a galaxy)"""
    def __init__(self,vxvv=None,vo=220.,ro=8.0,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])): #pragma: no cover (never used)
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a planar orbit

        INPUT:

           vxvv - [R,vR,vT(,phi)]

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

           2014-06-11 - Added conversion kwargs to physical coordinates - Bovy (IAS)

        """
        OrbitTop.__init__(self,vxvv=vxvv,
                          ro=ro,zo=zo,vo=vo,solarmotion=solarmotion)
        return None

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
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

    @physical_conversion('energy')
    def Jacobi(self,*args,**kwargs):
        """
        NAME:
           Jacobi
        PURPOSE:
           calculate the Jacobi integral of the motion
        INPUT:
           t - (optional) time at which to get the radius
           OmegaP= pattern speed of rotating frame
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
        out= self.E(*args,**kwargs)-OmegaP*self.L(*args,**kwargs)
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

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
            self.rs= self.orbit[:,0]
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
            self.rs= self.orbit[:,0]
        return nu.amin(self.rs)

    @physical_conversion('position')
    def zmax(self,pot=None,analytic=False,**kwargs):
        raise AttributeError("planarOrbit does not have a zmax")    

class planarROrbit(planarOrbitTop):
    """Class representing a planar orbit, without \phi. Useful for 
    orbit-integration in axisymmetric potentials when you don't care about the
    azimuth"""
    def __init__(self,vxvv=[1.,0.,1.],vo=220.,ro=8.0,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a planarROrbit

        INPUT:

           vxvv - [R,vR,vT]

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

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
           error message number (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-20
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        thispot= RZToplanarPotential(pot)
        self.t= nu.array(t)
        self._pot= thispot
        self.orbit, msg= _integrateROrbit(self.vxvv,thispot,t,method,dt)
        return msg

    @physical_conversion('energy')
    def E(self,*args,**kwargs):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           t - (optional) time at which to get the radius
           pot= potential instance or list of such instances
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
           2011-04-18 - Added t - Bovy (NYU)
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
        if isinstance(pot,Potential):
            thispot= RZToplanarPotential(pot)
        elif isinstance(pot,list):
            thispot= []
            for p in pot:
                if isinstance(p,Potential): thispot.append(RZToplanarPotential(p))
                else: thispot.append(p)
        else:
            thispot= pot
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return _evaluateplanarPotentials(thispot,thiso[0],
                                             t=t)\
                                            +thiso[1]**2./2.\
                                            +thiso[2]**2./2.
        else:
            return nu.array([_evaluateplanarPotentials(thispot,thiso[0,ii],
                                                      t=t[ii])\
                                 +thiso[1,ii]**2./2.\
                                 +thiso[2,ii]**2./2. for ii in range(len(t))])
        
class planarOrbit(planarOrbitTop):
    """Class representing a full planar orbit (R,vR,vT,phi)"""
    def __init__(self,vxvv=[1.,0.,1.,0.],vo=220.,ro=8.0,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a planarOrbit

        INPUT:

           vxvv - [R,vR,vT,phi]

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

        HISTORY:

           2010-07-12 - Written - Bovy (NYU)

           2014-06-11 - Added conversion kwargs to physical coordinates - Bovy (IAS)

        """
        if len(vxvv) == 3: #pragma: no cover
            raise ValueError("You only provided R,vR, & vT, but not phi; you probably want planarROrbit")
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
           2010-07-20
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        thispot= toPlanarPotential(pot)
        self.t= nu.array(t)
        self._pot= thispot
        self.orbit, msg= _integrateOrbit(self.vxvv,thispot,t,method,dt)
        return msg

    def integrate_dxdv(self,dxdv,t,pot,method='dopr54_c',
                       rectIn=False,rectOut=False):
        """
        NAME:
           integrate_dxdv
        PURPOSE:
           integrate the orbit and a small area of phase space
        INPUT:
           dxdv - [dR,dvR,dvT,dphi]
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint' for scipy's odeint
                   'rk4_c' for a 4th-order Runge-Kutta integrator in C
                   'rk6_c' for a 6-th order Runge-Kutta integrator in C
                   'dopr54_c' for a Dormand-Prince integrator in C (generally the fastest)
           rectIn= (False) if True, input dxdv is in rectangular coordinates
           rectOut= (False) if True, output dxdv (that in orbit_dxdv) is in rectangular coordinates
        OUTPUT:
           (none) (get the actual orbit using getOrbit_dxdv()
        HISTORY:
           2010-10-17 - Written - Bovy (IAS)
           2014-06-29 - Added rectIn and rectOut - Bovy (IAS)
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        thispot= toPlanarPotential(pot)
        self.t= nu.array(t)
        self._pot_dxdv= thispot
        self._pot= thispot
        self.orbit_dxdv, msg= _integrateOrbit_dxdv(self.vxvv,dxdv,thispot,t,
                                                   method,rectIn,rectOut)
        self.orbit= self.orbit_dxdv[:,:4]
        return msg

    @physical_conversion('energy')
    def E(self,*args,**kwargs):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           pot=
           t= time at which to evaluate E
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
        if isinstance(pot,Potential):
            thispot= toPlanarPotential(pot)
        elif isinstance(pot,list):
            thispot= []
            for p in pot:
                if isinstance(p,Potential): thispot.append(toPlanarPotential(p))
                else: thispot.append(p)
        else:
            thispot= pot
        if len(args) > 0:
            t= args[0]
        else:
            t= 0.
        #Get orbit
        thiso= self(*args,**kwargs)
        onet= (len(thiso.shape) == 1)
        if onet:
            return _evaluateplanarPotentials(thispot,thiso[0],
                                            phi=thiso[3],t=t)\
                                            +thiso[1]**2./2.\
                                            +thiso[2]**2./2.
        else:
            return nu.array([_evaluateplanarPotentials(thispot,thiso[0,ii],
                                                      phi=thiso[3,ii],
                                                      t=t[ii])\
                                 +thiso[1,ii]**2./2.\
                                 +thiso[2,ii]**2./2. for ii in range(len(t))])

    def e(self,analytic=False,pot=None):
        """
        NAME:
           e
        PURPOSE:
           calculate the eccentricity
        INPUT:
           analytic - calculate e analytically
           pot - potential used to analytically calculate e
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
            raise AttributeError("Integrate the orbit first or use analytic=True for approximate eccentricity")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

def _integrateROrbit(vxvv,pot,t,method,dt):
    """
    NAME:
       _integrateROrbit
    PURPOSE:
       integrate an orbit in a Phi(R) potential in the R-plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
       method - 'odeint' or 'leapfrog'
       dt - if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize
    OUTPUT:
       [:,3] array of [R,vR,vT] at each t
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
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
    if method.lower() == 'leapfrog':
        #We hack this by putting in a dummy phi
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out, msg= _integrateOrbit(this_vxvv,pot,t,method,dt)
        #tmp_out is (nt,4)
        out= tmp_out[:,0:3]
    elif ext_loaded and \
            (method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c'):
        #We hack this by putting in a dummy phi
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out, msg= _integrateOrbit(this_vxvv,pot,t,method,dt)
        #tmp_out is (nt,4)
        out= tmp_out[:,0:3]
    elif method.lower() == 'odeint' or not ext_loaded:
        l= vxvv[0]*vxvv[2]
        l2= l**2.
        init= [vxvv[0],vxvv[1]]
        intOut= integrate.odeint(_REOM,init,t,args=(pot,l2),
                                 rtol=10.**-8.)#,mxstep=100000000)
        out= nu.zeros((len(t),3))
        out[:,0]= intOut[:,0]
        out[:,1]= intOut[:,1]
        out[:,2]= l/out[:,0]
        msg= 0
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    _parse_warnmessage(msg)
    return (out,msg)

def _REOM(y,t,pot,l2):
    """
    NAME:
       _REOM
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
       2010-07-20 - Written - Bovy (NYU)
    """
    return [y[1],
            l2/y[0]**3.+_evaluateplanarRforces(pot,y[0],t=t)]

def _integrateOrbit(vxvv,pot,t,method,dt):
    """
    NAME:
       _integrateOrbit
    PURPOSE:
       integrate an orbit in a Phi(R) potential in the (R,phi)-plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT,phi]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
       method - 'odeint' or 'leapfrog'
       dt- if set, force the integrator to use this basic stepsize; must be an integer divisor of output stepsize
    OUTPUT:
       [:,4] array of [R,vR,vT,phi] at each t
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
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
    if method.lower() == 'leapfrog':
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                             vxvv[0]*nu.sin(vxvv[3]),
                             vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                             vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
        #integrate
        tmp_out= symplecticode.leapfrog(_rectForce,this_vxvv,
                                        t,args=(pot,),rtol=10.**-8)
        #go back to the cylindrical frame
        R= nu.sqrt(tmp_out[:,0]**2.+tmp_out[:,1]**2.)
        phi= nu.arccos(tmp_out[:,0]/R)
        phi[(tmp_out[:,1] < 0.)]= 2.*nu.pi-phi[(tmp_out[:,1] < 0.)]
        vR= tmp_out[:,2]*nu.cos(phi)+tmp_out[:,3]*nu.sin(phi)
        vT= tmp_out[:,3]*nu.cos(phi)-tmp_out[:,2]*nu.sin(phi)
        out= nu.zeros((len(t),4))
        out[:,0]= R
        out[:,1]= vR
        out[:,2]= vT
        out[:,3]= phi
        msg= 0
    elif ext_loaded and \
            (method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c'):
        warnings.warn("Using C implementation to integrate orbits",galpyWarningVerbose)
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                             vxvv[0]*nu.sin(vxvv[3]),
                             vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                             vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
        #integrate
        tmp_out, msg= integratePlanarOrbit_c(pot,this_vxvv,
                                             t,method,dt=dt)
        #go back to the cylindrical frame
        R= nu.sqrt(tmp_out[:,0]**2.+tmp_out[:,1]**2.)
        phi= nu.arccos(tmp_out[:,0]/R)
        phi[(tmp_out[:,1] < 0.)]= 2.*nu.pi-phi[(tmp_out[:,1] < 0.)]
        vR= tmp_out[:,2]*nu.cos(phi)+tmp_out[:,3]*nu.sin(phi)
        vT= tmp_out[:,3]*nu.cos(phi)-tmp_out[:,2]*nu.sin(phi)
        out= nu.zeros((len(t),4))
        out[:,0]= R
        out[:,1]= vR
        out[:,2]= vT
        out[:,3]= phi
    elif method.lower() == 'odeint' or not ext_loaded:
        vphi= vxvv[2]/vxvv[0]
        init= [vxvv[0],vxvv[1],vxvv[3],vphi]
        intOut= integrate.odeint(_EOM,init,t,args=(pot,),
                                 rtol=10.**-8.)#,mxstep=100000000)
        out= nu.zeros((len(t),4))
        out[:,0]= intOut[:,0]
        out[:,1]= intOut[:,1]
        out[:,3]= intOut[:,2]
        out[:,2]= out[:,0]*intOut[:,3]
        msg= 0
    else:
        raise NotImplementedError("requested integration method does not exist")
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    out[neg_radii,3]+= m.pi
    _parse_warnmessage(msg)
    return (out,msg)

def _integrateOrbit_dxdv(vxvv,dxdv,pot,t,method,rectIn,rectOut):
    """
    NAME:
       _integrateOrbit_dxdv
    PURPOSE:
       integrate an orbit and area of phase space in a Phi(R) potential 
       in the (R,phi)-plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT,phi]; vR outward!
       dxdv - difference to integrate [dR,dvR,dvT,dphi]
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
       method - 'odeint' or 'leapfrog'
       rectIn= (False) if True, input dxdv is in rectangular coordinates
       rectOut= (False) if True, output dxdv (that in orbit_dxdv) is in rectangular coordinates
    OUTPUT:
       [:,8] array of [R,vR,vT,phi,dR,dvR,dvT,dphi] at each t
       error message from integrator
    HISTORY:
       2010-10-17 - Written - Bovy (IAS)
    """
    #First check that the potential has C
    if '_c' in method:
        allHasC= _check_c(pot) and _check_c(pot,dxdv=True)
        if not ext_loaded or \
        (not allHasC and not 'leapfrog' in method and not 'symplec' in method):
            method= 'odeint'
            if not ext_loaded: # pragma: no cover
                warnings.warn("Using odeint because C extension not loaded",galpyWarning)
            else:
                warnings.warn("Using odeint because not all used potential have adequate C implementations to integrate phase-space volumes",galpyWarning)
    #go to the rectangular frame
    this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                         vxvv[0]*nu.sin(vxvv[3]),
                         vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                         vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
    if not rectIn:
        this_dxdv= nu.array([nu.cos(vxvv[3])*dxdv[0]
                             -vxvv[0]*nu.sin(vxvv[3])*dxdv[3],
                             nu.sin(vxvv[3])*dxdv[0]
                             +vxvv[0]*nu.cos(vxvv[3])*dxdv[3],
                             -(vxvv[1]*nu.sin(vxvv[3])
                               +vxvv[2]*nu.cos(vxvv[3]))*dxdv[3]
                             +nu.cos(vxvv[3])*dxdv[1]-nu.sin(vxvv[3])*dxdv[2],
                             (vxvv[1]*nu.cos(vxvv[3])
                              -vxvv[2]*nu.sin(vxvv[3]))*dxdv[3]
                             +nu.sin(vxvv[3])*dxdv[1]+nu.cos(vxvv[3])*dxdv[2]])
    else:
        this_dxdv= dxdv
    if 'leapfrog' in method.lower() or 'symplec' in method.lower():
        raise TypeError('Symplectic integration for phase-space volume is not possible')
    elif ext_loaded and \
            (method.lower() == 'rk4_c' or method.lower() == 'rk6_c' \
            or method.lower() == 'dopr54_c'):
        warnings.warn("Using C implementation to integrate orbits",galpyWarningVerbose)
        #integrate
        tmp_out, msg= integratePlanarOrbit_dxdv_c(pot,this_vxvv,this_dxdv,
                                                  t,method)
    elif method.lower() == 'odeint' or not ext_loaded:
        init= [this_vxvv[0],this_vxvv[1],this_vxvv[2],this_vxvv[3],
               this_dxdv[0],this_dxdv[1],this_dxdv[2],this_dxdv[3]]
        #integrate
        tmp_out= integrate.odeint(_EOM_dxdv,init,t,args=(pot,),
                                  rtol=10.**-8.)#,mxstep=100000000)
        msg= 0
    else:
        raise NotImplementedError("requested integration method does not exist")
    #go back to the cylindrical frame
    R= nu.sqrt(tmp_out[:,0]**2.+tmp_out[:,1]**2.)
    phi= nu.arccos(tmp_out[:,0]/R)
    phi[(tmp_out[:,1] < 0.)]= 2.*nu.pi-phi[(tmp_out[:,1] < 0.)]
    vR= tmp_out[:,2]*nu.cos(phi)+tmp_out[:,3]*nu.sin(phi)
    vT= tmp_out[:,3]*nu.cos(phi)-tmp_out[:,2]*nu.sin(phi)
    cp= nu.cos(phi)
    sp= nu.sin(phi)
    dR= cp*tmp_out[:,4]+sp*tmp_out[:,5]
    dphi= (cp*tmp_out[:,5]-sp*tmp_out[:,4])/R
    dvR= cp*tmp_out[:,6]+sp*tmp_out[:,7]+vT*dphi
    dvT= cp*tmp_out[:,7]-sp*tmp_out[:,6]-vR*dphi
    out= nu.zeros((len(t),8))
    out[:,0]= R
    out[:,1]= vR
    out[:,2]= vT
    out[:,3]= phi
    if rectOut:
        out[:,4:]= tmp_out[:,4:]
    else:
        out[:,4]= dR
        out[:,7]= dphi
        out[:,5]= dvR
        out[:,6]= dvT
    _parse_warnmessage(msg)
    return (out,msg)

def _EOM_dxdv(x,t,pot):
    """
    NAME:
       _EOM_dxdv
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential 
       equation, for integrating phase space differences, rectangular
    INPUT:
       x - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       dy/dt
    HISTORY:
       2011-10-18 - Written - Bovy (NYU)
    """
    #x is rectangular so calculate R and phi
    R= nu.sqrt(x[0]**2.+x[1]**2.)
    phi= nu.arccos(x[0]/R)
    sinphi= x[1]/R
    cosphi= x[0]/R
    if x[1] < 0.: phi= 2.*nu.pi-phi
    #calculate forces
    Rforce= _evaluateplanarRforces(pot,R,phi=phi,t=t)
    phiforce= _evaluateplanarphiforces(pot,R,phi=phi,t=t)
    R2deriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dR=2)
    phi2deriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dphi=2)
    Rphideriv= _evaluateplanarPotentials(pot,R,phi=phi,t=t,dR=1,dphi=1)
    #Calculate derivatives and derivatives+time derivatives
    dFxdx= -cosphi**2.*R2deriv\
           +2.*cosphi*sinphi/R**2.*phiforce\
           +sinphi**2./R*Rforce\
           +2.*sinphi*cosphi/R*Rphideriv\
           -sinphi**2./R**2.*phi2deriv
    dFxdy= -sinphi*cosphi*R2deriv\
           +(sinphi**2.-cosphi**2.)/R**2.*phiforce\
           -cosphi*sinphi/R*Rforce\
           -(cosphi**2.-sinphi**2.)/R*Rphideriv\
           +cosphi*sinphi/R**2.*phi2deriv
    dFydx= -cosphi*sinphi*R2deriv\
           +(sinphi**2.-cosphi**2.)/R**2.*phiforce\
           +(sinphi**2.-cosphi**2.)/R*Rphideriv\
           -sinphi*cosphi/R*Rforce\
           +sinphi*cosphi/R**2.*phi2deriv
    dFydy= -sinphi**2.*R2deriv\
           -2.*sinphi*cosphi/R**2.*phiforce\
           -2.*sinphi*cosphi/R*Rphideriv\
           +cosphi**2./R*Rforce\
           -cosphi**2./R**2.*phi2deriv
    return nu.array([x[2],x[3],
                     cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce,
                     x[6],x[7],
                     dFxdx*x[4]+dFxdy*x[5],
                     dFydx*x[4]+dFydy*x[5]])

def _EOM(y,t,pot):
    """
    NAME:
       _EOM
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
       2010-07-20 - Written - Bovy (NYU)
    """
    l2= (y[0]**2.*y[3])**2.
    return [y[1],
            l2/y[0]**3.+_evaluateplanarRforces(pot,y[0],phi=y[2],t=t),
            y[3],
            1./y[0]**2.*(_evaluateplanarphiforces(pot,y[0],phi=y[2],t=t)-
                         2.*y[0]*y[1]*y[3])]

def _rectForce(x,pot,t=0.):
    """
    NAME:
       _rectForce
    PURPOSE:
       returns the force in the rectangular frame
    INPUT:
       x - current position
       t - current time
       pot - (list of) Potential instance(s)
    OUTPUT:
       force
    HISTORY:
       2011-02-02 - Written - Bovy (NYU)
    """
    #x is rectangular so calculate R and phi
    R= nu.sqrt(x[0]**2.+x[1]**2.)
    phi= nu.arccos(x[0]/R)
    sinphi= x[1]/R
    cosphi= x[0]/R
    if x[1] < 0.: phi= 2.*nu.pi-phi
    #calculate forces
    Rforce= _evaluateplanarRforces(pot,R,phi=phi,t=t)
    phiforce= _evaluateplanarphiforces(pot,R,phi=phi,t=t)
    return nu.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce])

def _parse_warnmessage(msg):
    if msg == 1: #pragma: no cover
        warnings.warn("During numerical integration, steps smaller than the smallest step were requested; integration might not be accurate",galpyWarning)
        
