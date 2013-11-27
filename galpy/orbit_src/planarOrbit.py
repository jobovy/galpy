import math as m
import warnings
import numpy as nu
from scipy import integrate
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
from OrbitTop import OrbitTop
from galpy.potential_src.planarPotential import evaluateplanarRforces,\
    RZToplanarPotential, evaluateplanarphiforces,\
    evaluateplanarPotentials
from galpy.potential_src.Potential import Potential
from galpy.util import galpyWarning
try:
    from galpy.orbit_src.integratePlanarOrbit import integratePlanarOrbit_c,\
        integratePlanarOrbit_dxdv_c
except IOError:
    warnings.warn("integratePlanarOrbit_c extension module not loaded",
                  galpyWarning)
    ext_loaded= False
else:
    ext_loaded= True   
class planarOrbitTop(OrbitTop):
    """Top-level class representing a planar orbit (i.e., one in the plane 
    of a galaxy)"""
    def __init__(self,vxvv=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planar orbit
        INPUT:
           vxvv - [R,vR,vT(,phi)]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
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
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

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
        return self.E(*args,**kwargs)-OmegaP*self.L(*args,**kwargs)

    def rap(self,analytic=False,pot=None):
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

    def rperi(self,analytic=False,pot=None):
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

    def zmax(self,pot=None,analytic=False):
        raise AttributeError("planarOrbit does not have a zmax")
    
    def plotJacobi(self,*args,**kwargs):
        """
        NAME:
           plotJacobi
        PURPOSE:
           plot Jacobi integratl(.) along the orbit
        INPUT:
           OmegaP= pattern speed
           pot= Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Jacobi vs d1: e.g., 't', 'R', 'vR', 'vT'
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2011-10-09 - Written - Bovy (IAS)
        """
        labeldict= {'t':r'$t$','R':r'$R$','vR':r'$v_R$','vT':r'$v_T$',
                    'z':r'$z$','vz':r'$v_z$','phi':r'$\phi$',
                    'x':r'$x$','y':r'$y$','vx':r'$v_x$','vy':r'$v_y$'}
        Js= self.Jacobi(self.t,**kwargs)
        if kwargs.has_key('OmegaP'): kwargs.pop('OmegaP')
        if kwargs.has_key('pot'): kwargs.pop('pot')
        if kwargs.has_key('d1'):
            d1= kwargs['d1']
            kwargs.pop('d1')
        else:
            d1= 't'
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E-\Omega_p\,L$'
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),Js/Js[0],
                           *args,**kwargs)
        elif d1 == 'R':
            plot.bovy_plot(self.orbit[:,0],Js/Js[0],
                           *args,**kwargs)
        elif d1 == 'vR':
            plot.bovy_plot(self.orbit[:,1],Js/Js[0],
                           *args,**kwargs)
        elif d1 == 'vT':
            plot.bovy_plot(self.orbit[:,2],Js/Js[0],
                           *args,**kwargs)

class planarROrbit(planarOrbitTop):
    """Class representing a planar orbit, without \phi. Useful for 
    orbit-integration in axisymmetric potentials when you don't care about the
    azimuth"""
    def __init__(self,vxvv=[1.,0.,1.]):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planarROrbit
        INPUT:
           vxvv - [R,vR,vT]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        self.vxvv= vxvv
        #For boundary-condition integration
        self._BCIntegrateFunction= _integrateROrbit
        return None

    def integrate(self,t,pot,method='leapfrog_c'):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint' for scipy's odeint, 'leapfrog' for a simple 
                   leapfrog implementation, 'leapfrog_c' for a simple leapfrog
                   in C (if possible)
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
        if isinstance(pot,list):
            c_possible= True
            for p in pot:
                if not p.hasC:
                    c_possible= False
                    break
        else:
            c_possible= pot.hasC
        c_possible*= ext_loaded
        if '_c' in method and not c_possible:
            method= 'odeint'
        self.orbit, msg= _integrateROrbit(self.vxvv,thispot,t,method)
        return msg

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
            return evaluateplanarPotentials(thiso[0],thispot,
                                            t=t)\
                                            +thiso[1]**2./2.\
                                            +thiso[2]**2./2.
        else:
            return nu.array([evaluateplanarPotentials(thiso[0,ii],thispot,
                                                     t=t[ii])\
                                 +thiso[1,ii]**2./2.\
                                 +thiso[2,ii]**2./2. for ii in range(len(t))])
        
    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'R', 'vR', 'vT'
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
        if len(self.vxvv) == 4:
            self.Es= [evaluateplanarPotentials(self.orbit[ii,0],pot,
                                               phi=self.orbit[ii,3],
                                               t=self.t[ii])+
                      self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.
                      for ii in range(len(self.t))]
        else:
            self.Es= [evaluateplanarPotentials(self.orbit[ii,0],pot,
                                               t=self.t[ii])+
                      self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.
                      for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E$'
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.Es)/self.Es[0],
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

    def _callRect(self,*args):
        raise AttributeError("Cannot transform R-only planar orbit to rectangular coordinates")

class planarOrbit(planarOrbitTop):
    """Class representing a full planar orbit (R,vR,vT,phi)"""
    def __init__(self,vxvv=[1.,0.,1.,0.]):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planarOrbit
        INPUT:
           vxvv - [R,vR,vT,phi]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        if len(vxvv) == 3:
            raise ValueError("You only provided R,vR, & vT, but not phi; you probably want planarROrbit")
        self.vxvv= vxvv
        #For boundary-condition integration
        self._BCIntegrateFunction= _integrateOrbit
        return None

    def integrate(self,t,pot,method='leapfrog_c'):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint' for scipy's odeint, 'leapfrog' for a simple
                   leapfrog implementation, 'leapfrog_c' for a simple
                   leapfrog implemenation in C (if possible)
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-20
        """
        if hasattr(self,'_orbInterp'): delattr(self,'_orbInterp')
        if hasattr(self,'rs'): delattr(self,'rs')
        thispot= RZToplanarPotential(pot)
        self.t= nu.array(t)
        self._pot= thispot
        if isinstance(pot,list):
            c_possible= True
            for p in pot:
                if not p.hasC:
                    c_possible= False
                    break
        else:
            c_possible= pot.hasC
        c_possible*= ext_loaded
        if '_c' in method and not c_possible:
            method= 'odeint'
        self.orbit, msg= _integrateOrbit(self.vxvv,thispot,t,method)
        return msg

    def integrate_dxdv(self,dxdv,t,pot,method='dopr54_c'):
        """
        NAME:
           integrate_dxdv
        PURPOSE:
           integrate the orbit and a small area of phase space
        INPUT:
           dxdv - [dR,dvR,dvT,dphi]
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
           method= 'odeint' for scipy's odeint, 'leapfrog' for a simple
                   leapfrog implementation, 'leapfrog_c' for a simple
                   leapfrog implemenation in C (if possible)
        OUTPUT:
           (none) (get the actual orbit using getOrbit_dxdv()
        HISTORY:
           2010-10-17 - Written - Bovy (IAS)
        """
        thispot= RZToplanarPotential(pot)
        self.t= nu.array(t)
        self._pot_dxdv= thispot
        if isinstance(pot,list):
            c_possible= True
            for p in pot:
                if not p.hasC:
                    c_possible= False
                    break
        else:
            c_possible= pot.hasC
        c_possible*= ext_loaded
        if '_c' in method and not c_possible:
            method= 'odeint'
        self.orbit_dxdv, msg= _integrateOrbit_dxdv(self.vxvv,dxdv,thispot,t,method)
        return msg

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
            return evaluateplanarPotentials(thiso[0],thispot,
                                            phi=thiso[3],t=t)\
                                            +thiso[1]**2./2.\
                                            +thiso[2]**2./2.
        else:
            return nu.array([evaluateplanarPotentials(thiso[0,ii],thispot,
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
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

    def plotE(self,*args,**kwargs):
        """
        NAME:
           plotE
        PURPOSE:
           plot E(.) along the orbit
        INPUT:
           pot - Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Ez vs d1: e.g., 't', 'R', 'vR', 'vT', 'phi'
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
        self.Es= [evaluateplanarPotentials(self.orbit[ii,0],pot,
                                           phi=self.orbit[ii,3])+
                  self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.
                  for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E$'
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),nu.array(self.Es)/self.Es[0],
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
        elif d1 == 'phi':
            plot.bovy_plot(self.orbit[:,3],nu.array(self.Es)/self.Es[0],
                           *args,**kwargs)


    def _callRect(self,*args):
        kwargs= {}
        kwargs['rect']= False
        vxvv= self.__call__(*args,**kwargs)
        x= vxvv[0]*m.cos(vxvv[3])
        y= vxvv[0]*m.sin(vxvv[3])
        vx= vxvv[1]*m.cos(vxvv[5])-vxvv[2]*m.sin(vxvv[5])
        vy= -vxvv[1]*m.sin(vxvv[5])-vxvv[2]*m.cos(vxvv[5])
        return nu.array([x,y,vx,vy])

def _integrateROrbit(vxvv,pot,t,method):
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
    OUTPUT:
       [:,3] array of [R,vR,vT] at each t
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
    if method.lower() == 'leapfrog':
        #We hack this by putting in a dummy phi
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out= _integrateOrbit(this_vxvv,pot,t,method)
        #tmp_out is (nt,4)
        out= tmp_out[:,0:3]
        msg= 0
    elif method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c':
        #We hack this by putting in a dummy phi
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out, msg= _integrateOrbit(this_vxvv,pot,t,method)
        #tmp_out is (nt,4)
        out= tmp_out[:,0:3]
    elif method.lower() == 'odeint':
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
            l2/y[0]**3.+evaluateplanarRforces(y[0],pot,t=t)]

def _integrateOrbit(vxvv,pot,t,method):
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
    OUTPUT:
       [:,4] array of [R,vR,vT,phi] at each t
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
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
    elif method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c':
        warnings.warn("Using C implementation to integrate orbits",galpyWarning)
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                             vxvv[0]*nu.sin(vxvv[3]),
                             vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                             vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
        #integrate
        tmp_out, msg= integratePlanarOrbit_c(pot,this_vxvv,
                                             t,method)
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
    elif method.lower() == 'odeint':
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

def _integrateOrbit_dxdv(vxvv,dxdv,pot,t,method):
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
    OUTPUT:
       [:,8] array of [R,vR,vT,phi,dR,dvR,dvT,dphi] at each t
       error message from integrator
    HISTORY:
       2010-10-17 - Written - Bovy (IAS)
    """
    #go to the rectangular frame
    this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                         vxvv[0]*nu.sin(vxvv[3]),
                         vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                         vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
    this_dxdv= nu.array([nu.cos(vxvv[3])*dxdv[0]-vxvv[0]*nu.sin(vxvv[3])*dxdv[3],
                         nu.sin(vxvv[3])*dxdv[0]+vxvv[0]*nu.cos(vxvv[3])*dxdv[3],
                         -(vxvv[1]*nu.sin(vxvv[3])+vxvv[2]*nu.cos(vxvv[3]))*dxdv[3]
                         +nu.cos(vxvv[3])*dxdv[1]-nu.sin(vxvv[3])*dxdv[2],
                         (vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]))*dxdv[3]
                         +nu.sin(vxvv[3])*dxdv[1]+nu.cos(vxvv[3])*dxdv[2]])
    if method.lower() == 'leapfrog_c' or method.lower() == 'rk4_c' \
            or method.lower() == 'rk6_c' or method.lower() == 'symplec4_c' \
            or method.lower() == 'symplec6_c' or method.lower() == 'dopr54_c':
        #raise NotImplementedError("C implementation of phase space integration not implemented yet")
        warnings.warn("Using C implementation to integrate orbits",galpyWarning)
        #integrate
        tmp_out, msg= integratePlanarOrbit_dxdv_c(pot,this_vxvv,this_dxdv,
                                                  t,method)
    elif method.lower() == 'odeint':
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
    Rforce= evaluateplanarRforces(R,pot,phi=phi,t=t)
    phiforce= evaluateplanarphiforces(R,pot,phi=phi,t=t)
    R2deriv= evaluateplanarPotentials(R,pot,phi=phi,t=t,dR=2)
    phi2deriv= evaluateplanarPotentials(R,pot,phi=phi,t=t,dphi=2)
    Rphideriv= evaluateplanarPotentials(R,pot,phi=phi,t=t,dR=1,dphi=1)
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
            l2/y[0]**3.+evaluateplanarRforces(y[0],pot,phi=y[2],t=t),
            y[3],
            1./y[0]**2.*(evaluateplanarphiforces(y[0],pot,phi=y[2],t=t)-
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
    Rforce= evaluateplanarRforces(R,pot,phi=phi,t=t)
    phiforce= evaluateplanarphiforces(R,pot,phi=phi,t=t)
    return nu.array([cosphi*Rforce-1./R*sinphi*phiforce,
                     sinphi*Rforce+1./R*cosphi*phiforce])

def _parse_warnmessage(msg):
    if msg == 1:
        warnings.warn("During numerical integration, steps smaller than the smallest step were requested; integration might not be accurate",galpyWarning)
        
