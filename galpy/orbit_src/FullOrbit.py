import warnings
import math as m
import numpy as nu
from scipy import integrate
from galpy.potential_src.Potential import evaluateRforces, evaluatezforces,\
    evaluatePotentials, evaluatephiforces, evaluateDensities
from galpy.util import galpyWarning
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
try:
    from galpy.orbit_src.integrateFullOrbit import integrateFullOrbit_c
except IOError:
    warnings.warn("integrateFullOrbit_c extension module not loaded",
                  galpyWarning)
    ext_loaded= False
else:
    ext_loaded= True
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
        #For boundary-condition integration
        self._BCIntegrateFunction= _integrateFullOrbit
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
           method= 'odeint' for scipy's odeint integration, 'leapfrog' for
                    a simple symplectic integrator
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
        if not isinstance(OmegaP,(int,float)) and len(OmegaP) == 3:
            if isinstance(OmegaP,list): thisOmegaP= nu.array(OmegaP)
            else: thisOmegaP= OmegaP
            return self.E(*args,**kwargs)-nu.dot(thisOmegaP,
                                                 self.L(*args,**kwargs))
        else:
            return self.E(*args,**kwargs)-OmegaP*self.L(*args,**kwargs)[:,2]

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
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
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
            self.rs= nu.sqrt(self.orbit[:,0]**2.+self.orbit[:,3]**2.)
        return nu.amin(self.rs)

    def zmax(self,analytic=False,pot=None):
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
        try:
            self.Es= evaluatePotentials(self.orbit[:,0],self.orbit[:,3],
                                        pot,phi=self.orbit[:,5],
                                        t=self.t)\
                                        +self.orbit[:,1]**2./2.\
                                        +self.orbit[:,2]**2./2.\
                                        +self.orbit[:,4]**2./2.
        except TypeError:
            self.Es= [evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],
                                         pot,phi=self.orbit[ii,5],
                                         t=self.t[ii])+
                      self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.+
                      self.orbit[ii,4]**2./2. for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E$'
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

    def plotJacobi(self,*args,**kwargs):
        """
        NAME:
           plotJacobi
        PURPOSE:
           plot the Jacobi integral along the orbit
        INPUT:
           OmegaP= pattern speed
           pot= Potential instance or list of instances in which the orbit was
                 integrated
           d1= - plot Jacobi vs d1: e.g., 't', 'z', 'R', 'vR', 'vT', 'vz'      
           +bovy_plot.bovy_plot inputs
        OUTPUT:
           figure to output device
        HISTORY:
           2011-10-10 - Written - Bovy (IAS)
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
            kwargs['ylabel']= r'$E$'
        if d1 == 't':
            plot.bovy_plot(nu.array(self.t),Js/Js[0],
                           *args,**kwargs)
        elif d1 == 'z':
            plot.bovy_plot(self.orbit[:,3],Js/Js[0],
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
        elif d1 == 'vz':
            plot.bovy_plot(self.orbit[:,4],Js/Js[0],
                           *args,**kwargs)
        elif d1 == 'phi':
            plot.bovy_plot(self.orbit[:,5],Js/Js[0],
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
        self.Ezs= [evaluatePotentials(self.orbit[ii,0],self.orbit[ii,3],
                                      pot,phi=self.orbit[ii,5],t=self.t[ii])-
                   evaluatePotentials(self.orbit[ii,0],0.,pot,
                                      phi=self.orbit[ii,5],t=self.t[ii])+
                  self.orbit[ii,4]**2./2. for ii in range(len(self.t))]
        if not kwargs.has_key('xlabel'):
            kwargs['xlabel']= labeldict[d1]
        if not kwargs.has_key('ylabel'):
            kwargs['ylabel']= r'$E_z$'
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

    def _callRect(self,*args):
        kwargs= {}
        kwargs['rect']= False
        vxvv= self.__call__(*args,**kwargs)
        x= vxvv[0]*nu.cos(vxvv[5])
        y= vxvv[0]*nu.sin(vxvv[5])
        vx= vxvv[1]*nu.cos(vxvv[5])-vxvv[2]*nu.sin(vxvv[5])
        vy= -vxvv[1]*nu.sin(vxvv[5])-vxvv[2]*nu.cos(vxvv[5])
        return nu.array([x,y,vxvv[3],vx,vy,vxvv[4]])

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


