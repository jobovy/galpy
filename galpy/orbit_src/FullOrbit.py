import warnings
import copy
import math as m
import numpy as nu
from scipy import integrate, optimize
import scipy
if int(scipy.__version__.split('.')[1]) < 10: #pragma: no cover
    from scipy.maxentropy import logsumexp
else:
    from scipy.misc import logsumexp
from galpy.potential_src.Potential import evaluateRforces, evaluatezforces,\
    evaluatePotentials, evaluatephiforces, evaluateDensities
from galpy.util import galpyWarning
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
import galpy.util.bovy_coords as coords
#try:
from galpy.orbit_src.integrateFullOrbit import integrateFullOrbit_c, _ext_loaded
ext_loaded= _ext_loaded
from galpy.util.bovy_conversion import physical_conversion
from OrbitTop import OrbitTop
_ORBFITNORMRADEC= 360.
_ORBFITNORMDIST= 10.
_ORBFITNORMPMRADEC= 4.
_ORBFITNORMVLOS= 200.
class FullOrbit(OrbitTop):
    """Class that holds and integrates orbits in full 3D potentials"""
    def __init__(self,vxvv=[1.,0.,0.9,0.,0.1],vo=220.,ro=8.0,zo=0.025,
                 solarmotion=nu.array([-10.1,4.0,6.7])):
        """
        NAME:

           __init__

        PURPOSE:

           intialize a full orbit

        INPUT:

           vxvv - initial condition [R,vR,vT,z,vz,phi]

           vo - circular velocity at ro (km/s)

           ro - distance from vantage point to GC (kpc)

           zo - offset toward the NGP of the Sun wrt the plane (kpc)

           solarmotion - value in [-U,V,W] (km/s)

        OUTPUT:

           (none)

        HISTORY:

           2010-08-01 - Written - Bovy (NYU)

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
           2010-08-01 - Written - Bovy (NYU)
        """
        #Reset things that may have been defined by a previous integration
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        self.t= nu.array(t)
        self._pot= pot
        self.orbit= _integrateFullOrbit(self.vxvv,pot,t,method)

    @physical_conversion('energy')
    def Jacobi(self,*args,**kwargs):
        """
        NAME:
           Jacobi
        PURPOSE:
           calculate the Jacobi integral of the motion
        INPUT:
           Omega - pattern speed of rotating frame
           t= time
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
        if not isinstance(OmegaP,(int,float)) and len(OmegaP) == 3:
            if isinstance(OmegaP,list): thisOmegaP= nu.array(OmegaP)
            else: thisOmegaP= OmegaP
            out= self.E(*args,**kwargs)-nu.dot(thisOmegaP,
                                                 self.L(*args,**kwargs).T).T
        else:
            out= self.E(*args,**kwargs)-OmegaP*self.L(*args,**kwargs)[:,2]
        if not old_physical is None:
            kwargs['use_physical']= old_physical
        else:
            kwargs.pop('use_physical')
        return out

    @physical_conversion('energy')
    def E(self,*args,**kwargs):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           t - (optional) time at which to get the energy
           pot= potential instance or list of such instances
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
                                      phi=thiso[5],t=t)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.\
                                      +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],thiso[3,ii],
                                                pot,phi=thiso[5,ii],
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
                                      phi=thiso[5],t=t)\
                                      +thiso[1]**2./2.\
                                      +thiso[2]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],0.,
                                                pot,phi=thiso[5,ii],
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
                                      phi=thiso[5],t=t)\
                                      -evaluatePotentials(thiso[0],0.,pot,
                                                          phi=thiso[5],t=t)\
                                                          +thiso[4]**2./2.
        else:
            return nu.array([evaluatePotentials(thiso[0,ii],thiso[3,ii],
                                                pot,phi=thiso[5,ii],
                                                t=t[ii])\
                                 -evaluatePotentials(thiso[0,ii],0.,
                                                     pot,phi=thiso[5,ii],
                                                t=t[ii])\
                                 +thiso[4,ii]**2./2. for ii in range(len(t))])

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
           analytic - compute this analytically
           pot - potential to use for analytical calculation
        OUTPUT:
           Z_max
        HISTORY:
           2010-09-20 - Written - Bovy (NYU)
           2012-06-01 - Added analytic calculation - Bovy (IAS)
        """
        if analytic:
            self._setupaA(pot=pot,type='adiabatic')
            zmax= self._aA.calczmax(self)
            return zmax
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        return nu.amax(nu.fabs(self.orbit[:,3]))

    def fit(self,vxvv,vxvv_err=None,pot=None,radec=False,lb=False,
            tintJ=10,ntintJ=1000,integrate_method='dopr54_c',
            **kwargs):
        """
        NAME:
           fit
        PURPOSE:
           fit an Orbit to data using the current orbit as the initial 
           condition
        INPUT:
           vxvv - [:,6] array of positions and velocities along the orbit
           vxvv_err= [:,6] array of errors on positions and velocities along the orbit (if None, these are set to 0.01)
           pot= Potential to fit the orbit in

           Keywords related to the input data:
               radec= if True, input vxvv and vxvv are [ra,dec,d,mu_ra, mu_dec,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (all J2000.0; mu_ra = mu_ra * cos dec); the attributes of the current Orbit are used to convert between these coordinates and Galactocentric coordinates
               lb= if True, input vxvv and vxvv are [long,lat,d,mu_ll, mu_bb,vlos] in [deg,deg,kpc,mas/yr,mas/yr,km/s] (mu_ll = mu_ll * cos lat); the attributes of the current Orbit are used to convert between these coordinates and Galactocentric coordinates
               obs=[X,Y,Z,vx,vy,vz] - (optional) position and velocity of observer 
                                      (in kpc and km/s) (default=Object-wide default)
                                      Cannot be an Orbit instance with the orbit of the reference point, as w/ the ra etc. functions
                ro= distance in kpc corresponding to R=1. (default: taken from object)
                vo= velocity in km/s corresponding to v=1. (default: taken from object)

           Keywords related to the orbit integrations:
               tintJ= (default: 10) time to integrate orbits for fitting the orbit
               ntintJ= (default: 1000) number of time-integration points
               integrate_method= (default: 'dopr54_c') integration method to use

        OUTPUT:
           max of log likelihood
        HISTORY:
           2014-06-17 - Written - Bovy (IAS)

        TEST:
        from galpy.potential import LogarithmicHaloPotential; lp= LogarithmicHaloPotential(normalize=1.); from galpy.orbit import Orbit; o= Orbit(vxvv=[1.,0.1,1.1,0.1,0.02,0.]); ts= numpy.linspace(0,10,1000); o.integrate(ts,lp); outts= [0.,0.1,0.2,0.3,0.4]; vxvv= numpy.array([o.R(outts),o.vR(outts),o.vT(outts),o.z(outts),o.vz(outts),o.phi(outts)]).T; of= Orbit(vxvv=[1.02,0.101,1.101,0.101,0.0201,0.001]); of._orb.fit(vxvv,pot=lp,radec=False,tintJ=10,ntintJ=1000)

        """
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit first or specify pot=")
        if radec or lb:
            obs, ro, vo= self._parse_radec_kwargs(kwargs,vel=True,dontpop=True)
        else:
            obs, ro, vo= None, None, None
        new_vxvv, maxLogL= _fit_orbit(self,vxvv,vxvv_err,pot,radec=radec,lb=lb,
                                      tintJ=tintJ,ntintJ=ntintJ,
                                      integrate_method=integrate_method,
                                      ro=ro,vo=vo,obs=obs)
        #Setup with these new initial conditions
        self.vxvv= new_vxvv
        return maxLogL

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
                     evaluatePotentials(self.orbit[ii,0],0.,pot,
                                        phi= self.orbit[ii,5],t=self.t[ii])+
                     self.orbit[ii,4]**2./2.)/\
                        nu.sqrt(evaluateDensities(self.orbit[ii,0],0.,pot,phi=self.orbit[ii,5],t=self.t[ii]))\
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

def _integrateFullOrbit(vxvv,pot,t,method):
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
       method - 'odeint' or 'leapfrog'
    OUTPUT:
       [:,5] array of [R,vR,vT,z,vz,phi] at each t
    HISTORY:
       2010-08-01 - Written - Bovy (NYU)
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
    if method.lower() == 'leapfrog':
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[5]),
                             vxvv[0]*nu.sin(vxvv[5]),
                             vxvv[3],
                             vxvv[1]*nu.cos(vxvv[5])-vxvv[2]*nu.sin(vxvv[5]),
                             vxvv[2]*nu.cos(vxvv[5])+vxvv[1]*nu.sin(vxvv[5]),
                             vxvv[4]])
        #integrate
        out= symplecticode.leapfrog(_rectForce,this_vxvv,
                                    t,args=(pot,),rtol=10.**-8)
        #go back to the cylindrical frame
        R= nu.sqrt(out[:,0]**2.+out[:,1]**2.)
        phi= nu.arccos(out[:,0]/R)
        phi[(out[:,1] < 0.)]= 2.*nu.pi-phi[(out[:,1] < 0.)]
        vR= out[:,3]*nu.cos(phi)+out[:,4]*nu.sin(phi)
        vT= out[:,4]*nu.cos(phi)-out[:,3]*nu.sin(phi)
        out[:,3]= out[:,2]
        out[:,4]= out[:,5]
        out[:,0]= R
        out[:,1]= vR
        out[:,2]= vT
        out[:,5]= phi
    elif ext_loaded and \
            (method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c'):
        warnings.warn("Using C implementation to integrate orbits",
                      galpyWarning)
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[5]),
                             vxvv[0]*nu.sin(vxvv[5]),
                             vxvv[3],
                             vxvv[1]*nu.cos(vxvv[5])-vxvv[2]*nu.sin(vxvv[5]),
                             vxvv[2]*nu.cos(vxvv[5])+vxvv[1]*nu.sin(vxvv[5]),
                             vxvv[4]])
        #integrate
        tmp_out, msg= integrateFullOrbit_c(pot,this_vxvv,
                                           t,method)
        #go back to the cylindrical frame
        R= nu.sqrt(tmp_out[:,0]**2.+tmp_out[:,1]**2.)
        phi= nu.arccos(tmp_out[:,0]/R)
        phi[(tmp_out[:,1] < 0.)]= 2.*nu.pi-phi[(tmp_out[:,1] < 0.)]
        vR= tmp_out[:,3]*nu.cos(phi)+tmp_out[:,4]*nu.sin(phi)
        vT= tmp_out[:,4]*nu.cos(phi)-tmp_out[:,3]*nu.sin(phi)
        out= nu.zeros((len(t),6))
        out[:,0]= R
        out[:,1]= vR
        out[:,2]= vT
        out[:,5]= phi
        out[:,3]= tmp_out[:,2]
        out[:,4]= tmp_out[:,5]
    elif method.lower() == 'odeint' or not ext_loaded:
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
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    out[neg_radii,5]+= m.pi
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
    Rforce= evaluateRforces(R,x[2],pot,phi=phi,t=t)
    phiforce= evaluatephiforces(R,x[2],pot,phi=phi,t=t)
    return nu.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce,
                     evaluatezforces(R,x[2],pot,phi=phi,t=t)])

def _fit_orbit(orb,vxvv,vxvv_err,pot,radec=False,lb=False,
               tintJ=100,ntintJ=1000,integrate_method='dopr54_c',
               ro=None,vo=None,obs=None):
    """Fit an orbit to data in a given potential"""
    #Import here, because otherwise there is an infinite loop of imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    #Mock this up, bc we want to use its orbit-integration routines
    class mockActionAngleIsochroneApprox(actionAngleIsochroneApprox):
        def __init__(self,tintJ,ntintJ,pot,integrate_method='dopr54_c'):
            self._tintJ= tintJ
            self._ntintJ=ntintJ
            self._tsJ= nu.linspace(0.,self._tintJ,self._ntintJ)
            self._pot= pot
            self._integrate_method= integrate_method
            return None
    tmockAA= mockActionAngleIsochroneApprox(tintJ,ntintJ,pot,
                                            integrate_method=integrate_method)
    opt_vxvv= optimize.fmin_powell(_fit_orbit_mlogl,orb.vxvv,
                                   args=(vxvv,vxvv_err,pot,radec,lb,tmockAA,
                                         ro,vo,obs))
    maxLogL= -_fit_orbit_mlogl(opt_vxvv,vxvv,vxvv_err,pot,radec,lb,tmockAA,
                               ro,vo,obs)
    return (opt_vxvv,maxLogL)

def _fit_orbit_mlogl(new_vxvv,vxvv,vxvv_err,pot,radec,lb,tmockAA,
                     ro,vo,obs):
    """The log likelihood for fitting an orbit"""
    #Use this _parse_args routine, which does forward and backward integration
    iR,ivR,ivT,iz,ivz,iphi= tmockAA._parse_args(True,False,
                                                new_vxvv[0],
                                                new_vxvv[1],
                                                new_vxvv[2],
                                                new_vxvv[3],
                                                new_vxvv[4],
                                                new_vxvv[5])
    if radec or lb:
        #Need to transform to ra,dec
        #First transform to X,Y,Z,vX,vY,vZ (Galactic)
        X,Y,Z = coords.galcencyl_to_XYZ(iR.flatten(),iphi.flatten(),
                                        iz.flatten(),
                                        Xsun=obs[0]/ro,
                                        Ysun=obs[1]/ro,
                                        Zsun=obs[2]/ro)
        vX,vY,vZ = coords.galcencyl_to_vxvyvz(ivR.flatten(),ivT.flatten(),
                                              ivz.flatten(),iphi.flatten(),
                                              vsun=nu.array(\
                obs[3:6])/vo)
        bad_indx= (X == 0.)*(Y == 0.)*(Z == 0.)
        if True in bad_indx: X[bad_indx]+= ro/10000.
        lbdvrpmllpmbb= coords.rectgal_to_sphergal(X*ro,Y*ro,Z*ro,
                                                  vX*vo,vY*vo,vZ*vo,
                                                  degree=True)
        if lb:
            orb_vxvv= nu.array([lbdvrpmllpmbb[:,0],
                                lbdvrpmllpmbb[:,1],
                                lbdvrpmllpmbb[:,2],
                                lbdvrpmllpmbb[:,4],
                                lbdvrpmllpmbb[:,5],
                                lbdvrpmllpmbb[:,3]]).T
        else:
            #Further transform to ra,dec,pmra,pmdec
            radec= coords.lb_to_radec(lbdvrpmllpmbb[:,0],
                                      lbdvrpmllpmbb[:,1],degree=True)
            pmrapmdec= coords.pmllpmbb_to_pmrapmdec(lbdvrpmllpmbb[:,4],
                                                    lbdvrpmllpmbb[:,5],
                                                    lbdvrpmllpmbb[:,0],
                                                    lbdvrpmllpmbb[:,1],
                                                    degree=True)
            orb_vxvv= nu.array([radec[:,0],radec[:,1],
                                lbdvrpmllpmbb[:,2],
                                pmrapmdec[:,0],pmrapmdec[:,1],
                                lbdvrpmllpmbb[:,3]]).T
    else:
        #shape=(2tintJ-1,6)
        orb_vxvv= nu.array([iR.flatten(),ivR.flatten(),ivT.flatten(),
                            iz.flatten(),ivz.flatten(),iphi.flatten()]).T 
    out= 0.
    for ii in range(vxvv.shape[0]):
        sub_vxvv= (orb_vxvv-vxvv[ii,:].flatten())**2.
        #print sub_vxvv[nu.argmin(nu.sum(sub_vxvv,axis=1))]
        if not vxvv_err is None:
            sub_vxvv/= vxvv_err[ii,:]**2.
        else:
            sub_vxvv/= 0.01**2.
        out+= logsumexp(-0.5*nu.sum(sub_vxvv,axis=1))
    return -out

