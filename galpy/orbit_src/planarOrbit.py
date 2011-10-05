import math as m
import numpy as nu
from scipy import integrate
import galpy.util.bovy_plot as plot
import galpy.util.bovy_symplecticode as symplecticode
from galpy import actionAngle
from galpy.potential import LogarithmicHaloPotential, PowerSphericalPotential,\
    KeplerPotential
from OrbitTop import OrbitTop
from RZOrbit import RZOrbit
from galpy.potential_src.planarPotential import evaluateplanarRforces,\
    planarPotential, RZToplanarPotential, evaluateplanarphiforces,\
    evaluateplanarPotentials, planarPotentialFromRZPotential
from galpy.potential_src.Potential import Potential
from galpy.orbit_src.integratePlanarOrbit import integratePlanarOrbit_leapfrog
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
            if not hasattr(self,'_aA'):
                self._setupaA(pot=pot)
            (rperi,rap)= self._aA.calcRapRperi()
            return (rap-rperi)/(rap+rperi)
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return (nu.amax(self.rs)-nu.amin(self.rs))/(nu.amax(self.rs)+nu.amin(self.rs))

    def Jacobi(self,Omega,t=0.,pot=None):
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
        return self.E(pot=pot,t=t)-Omega*self.L()

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
            if not hasattr(self,'_aA'):
                self._setupaA(pot=pot)
            (rperi,rap)= self._aA.calcRapRperi()
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
            if not hasattr(self,'_aA'):
                self._setupaA(pot=pot)
            (rperi,rap)= self._aA.calcRapRperi()
            return rperi
        if not hasattr(self,'orbit'):
            raise AttributeError("Integrate the orbit first")
        if not hasattr(self,'rs'):
            self.rs= self.orbit[:,0]
        return nu.amin(self.rs)

    def zmax(self):
        raise AttributeError("planarOrbit does not have a zmax")
    
    def wp(self,pot=None):
        """
        NAME:
           wp
        PURPOSE:
           calculate the azimuthal angle
        INPUT:
           pot - potential
        OUTPUT:
           wp
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        if len(self.vxvv) < 4:
            raise AttributeError("'Orbit' does not track azimuth")
        else:
            return self.vxvv[-1]

    def _setupaA(self,pot=None):
        """
        NAME:
           _setupaA
        PURPOSE:
           set up an actionAngle module for this Orbit
        INPUT:
           pot - potential
        OUTPUT:
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        if isinstance(pot,Potential) or isinstance(pot,list):
            thispot= RZToplanarPotential(pot)
        else:
            thispot= pot
        if isinstance(thispot,LogarithmicHaloPotential) or \
                (isinstance(thispot,planarPotentialFromRZPotential) and \
                     isinstance(thispot._RZPot,LogarithmicHaloPotential)):
            self._aA= actionAngle.actionAngleFlat(self.vxvv[0],
                                                  self.vxvv[1],
                                                  self.vxvv[2])
        elif isinstance(thispot,KeplerPotential) or \
                (isinstance(thispot,planarPotentialFromRZPotential) and \
                     isinstance(thispot._RZPot,KeplerPotential)):
            self._aA= actionAngle.actionAnglePower(self.vxvv[0],
                                                   self.vxvv[1],
                                                   self.vxvv[2],
                                                   beta=-0.5)
        elif isinstance(thispot,PowerSphericalPotential) or \
                (isinstance(thispot,planarPotentialFromRZPotential) and \
                     isinstance(thispot._RZPot,PowerSphericalPotential)):
            if isinstance(thispot,planarPotentialFromRZPotential) and \
                    isinstance(thispot._RZPot,PowerSphericalPotential):
                thispot= thispot._RZPot
            if thispot.alpha == 2.:
                self._aA= actionAngle.actionAngleFlat(self.vxvv[0],
                                                      self.vxvv[1],
                                                      self.vxvv[2])
            else:
                self._aA= actionAngle.actionAnglePower(self.vxvv[0],
                                                       self.vxvv[1],
                                                       self.vxvv[2],
                                                       beta=1.-thispot.alpha/2.)
        else:
            
            self._aA= actionAngle.actionAngleAxi(self.vxvv[0],self.vxvv[1],
                                                 self.vxvv[2],pot=thispot)

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
        if '_c' in method and not c_possible:
            method= 'odeint'
        self.orbit= _integrateROrbit(self.vxvv,thispot,t,method)

    def E(self,pot=None,t=0.):
        """
        NAME:
           E
        PURPOSE:
           calculate the energy
        INPUT:
           pot= potential instance or list of such instances
           t= time
        OUTPUT:
           energy
        HISTORY:
           2010-09-15 - Written - Bovy (NYU)
           2011-04-18 - Added t - Bovy (NYU)
        """
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        if isinstance(pot,Potential):
            thispot= RZToplanarPotential(pot)
        elif isinstance(pot,list):
            thispot= []
            for p in pot:
                if isinstance(p,Potential): thispot.append(RZToplanarPotential(p))
                else: thispot.append(p)
        else:
            thispot= pot
        if len(self.vxvv) == 4:
            return evaluateplanarPotentials(self.vxvv[0],thispot,
                                            phi=self.vxvv[3],t=t)+\
                                            self.vxvv[1]**2./2.\
                                            +self.vxvv[2]**2./2.
        else:
            return evaluateplanarPotentials(self.vxvv[0],thispot,t=t)+\
                self.vxvv[1]**2./2.+self.vxvv[2]**2./2.
        
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
                                               phi=self.orbit[ii,3])+
                      self.orbit[ii,1]**2./2.+self.orbit[ii,2]**2./2.
                      for ii in range(len(self.t))]
        else:
            self.Es= [evaluateplanarPotentials(self.orbit[ii,0],pot)+
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
        if '_c' in method and not c_possible:
            method= 'odeint'
        self.orbit= _integrateOrbit(self.vxvv,thispot,t,method)

    def E(self,pot=None,t=0.):
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
        if pot is None:
            try:
                pot= self._pot
            except AttributeError:
                raise AttributeError("Integrate orbit or specify pot=")
        if isinstance(pot,Potential):
            thispot= RZToplanarPotential(pot)
        elif isinstance(pot,list):
            thispot= []
            for p in pot:
                if isinstance(p,Potential): thispot.append(RZToplanarPotential(p))
                else: thispot.append(p)
        else:
            thispot= pot
        return evaluateplanarPotentials(self.vxvv[0],thispot,
                                        phi=self.vxvv[3],t=t)+\
                                        self.vxvv[1]**2./2.+self.vxvv[2]**2./2.

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
            if not hasattr(self,'_aA'):
                self._setupaA(pot=pot)
            (rperi,rap)= self._aA.calcRapRperi()
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
    elif method.lower() == 'leapfrog_c':
        #We hack this by putting in a dummy phi
        this_vxvv= nu.zeros(len(vxvv)+1)
        this_vxvv[0:len(vxvv)]= vxvv
        tmp_out= _integrateOrbit(this_vxvv,pot,t,method)
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
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    return out

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
    elif method.lower() == 'leapfrog_c':
        print "Using C implementation"
        #go to the rectangular frame
        this_vxvv= nu.array([vxvv[0]*nu.cos(vxvv[3]),
                             vxvv[0]*nu.sin(vxvv[3]),
                             vxvv[1]*nu.cos(vxvv[3])-vxvv[2]*nu.sin(vxvv[3]),
                             vxvv[2]*nu.cos(vxvv[3])+vxvv[1]*nu.sin(vxvv[3])])
        #integrate
        tmp_out= integratePlanarOrbit_leapfrog(pot,this_vxvv,
                                               t,rtol=10.**-8)
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
    #post-process to remove negative radii
    neg_radii= (out[:,0] < 0.)
    out[neg_radii,0]= -out[neg_radii,0]
    out[neg_radii,3]+= m.pi
    return out

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

