###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleStaeckel
#
#             Use Binney (2012; MNRAS 426, 1324)'s Staeckel approximation for 
#             calculating the actions
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from actionAngle import actionAngle
from galpy.potential import evaluatePotentials, evaluateRforces, \
    evaluatezforces
from galpy.util import bovy_coords #for prolate confocal transforms
class actionAngleStaeckel():
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleStaeckel object
        INPUT:
           pot= potential or list of potentials (3D)
           delta= focus
        OUTPUT:
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleStaeckel")
        self._pot= kwargs['pot']
        if not kwargs.has_key('delta'):
            raise IOError("Must specify delta= for actionAngleStaeckel")
        self._delta= kwargs['delta']
        return None
    
    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           (jr,lz,jz), where jr=[jr,jrerr], and jz=[jz,jzerr]
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        #Set up the actionAngleStaeckelSingle object
        meta= actionAngle(*args)
        aASingle= actionAngleStaeckelSingle(*args,pot=self._pot,
                                             delta=self._delta)
        return (aASingle.JR(**kwargs),aASingle._R*aAAxi._vT,
                aASingle.Jz(**kwargs))

    def JR(self,*args,**kwargs):
        """
        NAME:
           JR
        PURPOSE:
           evaluate the action jr
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           Jr
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        #Set up the actionAngleStaeckelSingle object
        meta= actionAngle(*args)
        aASingle= actionAngleStaeckelSingle(*args,pot=self._pot)
        return aASingle.JR(**kwargs)

    def Jz(self,*args,**kwargs):
        """
        NAME:
           Jz
        PURPOSE:
           evaluate the action jz
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
           jz,jzerr
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        #Set up the actionAngleStaeckelSingle object
        meta= actionAngle(*args)
        aASingle= actionAngleStaeckelSingle(*args,pot=self._pot)
        return aASingle.Jz(**kwargs)

class actionAngleStaeckelSingle(actionAngle):
    """Action-angle formalism for axisymmetric potentials using Binney (2012)'s Staeckel approximation"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleStaeckelSingle object
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
              pot= potential or list of potentials
        OUTPUT:
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        actionAngle.__init__(self,*args,**kwargs)
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleStaeckelSingle")
        self._pot= kwargs['pot']
        if not kwargs.has_key('delta'):
            raise IOError("Must specify delta= for actionAngleStaeckel")
        self._delta= kwargs['delta']
        #Pre-calculate everything
        self._ux, self._vx= bovy_coords.Rz_to_uv(self._R,self._z,
                                                 delta=self._delta)
        self._sinvx= nu.sin(self._vx)
        self._cosvx= nu.cos(self._vx)
        self._coshux= nu.cosh(self._ux)
        self._sinhux= nu.sinh(self._ux)
        self._pux= self._delta*(self._vR*self._coshux*self._sinvx
                                +self._vz*self._sinhux*self._cosvx)
        self._pvx= self._delta*(self._vR*self._sinhux*self._cosvx
                                -self._vz*self._coshux*self._sinvx)
        EL= self.calcEL()
        self._E= EL[0]
        self._Lz= EL[1]
        #Determine umin and umax
        self._u0= self._ux #first guess
        self._potu0v0= potentialStaeckel(self._u0,self._vx,
                                         self._pot,self._delta)
        self._I3U= self._E*self._sinhux**2.-self._pux**2./2./self._delta**2.\
            -self._Lz**2./2./self._delta**2./self._sinhux**2.
        self.calcUminUmax()
        return None
    
    def angleR(self,**kwargs):
        """
        NAME:
           AngleR
        PURPOSE:
           Calculate the radial angle
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           w_R(R,vT,vT) in radians + 
           estimate of the error (does not include TR error)
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'angleR' not yet implemented for Staeckel approximation")

    def TR(self,**kwargs):
        """
        NAME:
           TR
        PURPOSE:
           Calculate the radial period for a power-law rotation curve
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           T_R(R,vT,vT)*vc/ro + estimate of the error
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'TR' not implemented yet for Staeckel approximation")

    def Tphi(self,**kwargs):
        """
        NAME:
           Tphi
        PURPOSE:
           Calculate the azimuthal period
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           T_phi(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'Tphi' not implemented yet for Staeckel approxximation")

    def I(self,**kwargs):
        """
        NAME:
           I
        PURPOSE:
           Calculate I, the 'ratio' between the radial and azimutha period
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           I(R,vT,vT) + estimate of the error
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'I' not implemented yet for Staeckel approxximation")

    def Jphi(self):
        """
        NAME:
           Jphi
        PURPOSE:
           Calculate the azimuthal action
        INPUT:
        OUTPUT:
           J_R(R,vT,vT)/ro/vc
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        return nu.array([self._R*self._vT,0.])

    def JR(self,**kwargs):
        """
        NAME:
           JR
        PURPOSE:
           Calculate the radial action
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           J_R(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'JR' not implemented yet for Staeckel approxximation")
        if hasattr(self,'_JR'):
            return self._JR
        (rperi,rap)= self.calcRapRperi(**kwargs)
        EL= self.calcEL(**kwargs)
        E, L= EL
        self._JR= (1./nu.pi*nu.array(integrate.quad(_JRAxiIntegrand,rperi,rap,
                                                    args=(E,L,self._pot),
                                                    **kwargs)))
        return self._JR

    def Jz(self,**kwargs):
        """
        NAME:
           Jz
        PURPOSE:
           Calculate the vertical action
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           J_z(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("'JR' not implemented yet for Staeckel approxximation")
        if hasattr(self,'_JZ'):
            return self._JZ
        (rperi,rap)= self.calcRapRperi(**kwargs)
        EL= self.calcEL(**kwargs)
        E, L= EL
        self._JZ= (1./nu.pi*nu.array(integrate.quad(_JRAxiIntegrand,rperi,rap,
                                                    args=(E,L,self._pot),
                                                    **kwargs)))
        return self._JZ

    def calcEL(self,**kwargs):
        """
        NAME:
           calcEL
        PURPOSE:
           calculate the energy and angular momentum
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           (E,L)
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """                           
        E,L= calcELStaeckel(self._R,self._vR,self._vT,self._z,self._vz,
                            self._pot)
        return (E,L)

    def calcu0(self):
        """
        NAME:
           calcu0
        PURPOSE:
           calculate the minimum of dU(u;v)
        INPUT:
        OUTPUT:
           u0
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """                           
        if hasattr(self,'_u0'):
            return self._u0
        self._u0= optimize.brentq(_u0Eq,0.,100.,
                                  args=(self._sinvx,self._cosvx,
                                        self._vx,self._delta,self._pot))
        #Also update 
        self._potu0v0= potentialStaeckel(self._u0,self._vx,
                                         self._pot,self._delta)
        return self._u0

    def calcUminUmax(self,**kwargs):
        """
        NAME:
           calcUminUmax
        PURPOSE:
           calculate the u 'apocenter' and 'pericenter'
        INPUT:
        OUTPUT:
           (umin,umax)
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        if hasattr(self,'_uminumax'):
            return self._uminumax
        E, L= self._E, self._Lz
        if True: #if True, such that we can later special-case
            startsign= 1.
            #rstart= _rapRperiAxiFindStart(self._R,E,L,self._pot,
            #                              startsign=startsign)
            rstart= 0.1
            if rstart == 0.: umin= 0.
            else: 
                try:
                    umin= optimize.brentq(_JRStaeckelIntegrandSquared,
                                          rstart,self._ux,
                                          (E,L,self._I3U,self._delta,
                                           self._u0,self._sinhux**2.,
                                           self._vx,self._sinvx**2.,
                                           self._potu0v0,self._pot),
                                           maxiter=200)
                except RuntimeError:
                    raise UnboundError("Orbit seems to be unbound")
            #rend= _rapRperiAxiFindStart(self._R,E,L,self._pot,rap=True,
            #                            startsign=startsign)
            rend= 30.
            umax= optimize.brentq(_JRStaeckelIntegrandSquared,
                                          self._ux,rend,
                                          (E,L,self._I3U,self._delta,
                                           self._u0,self._sinhux**2.,
                                           self._vx,self._sinvx**2.,
                                           self._potu0v0,self._pot),
                                           maxiter=200)
        self._uminumax= (umin,umax)
        return self._uminumax

def calcRapRperiFromELAxi(E,L,pot,vc=1.,ro=1.):
    """
    NAME:
       calcRapRperiFromELAxi
    PURPOSE:
       calculate the apocenter and pericenter radii
    INPUT:
       E - energy
       L - angular momemtum
       pot - potential
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       (rperi,rap)
    HISTORY:
       2010-12-01 - Written - Bovy (NYU)
    """
    rstart= _rapRperiAxiFindStart(L,E,L)
    rperi= optimize.brentq(_rapRperiAxiEq,rstart,L,(E,L,pot))
    rend= _rapRperiAxiFindStart(L,E,L,rap=True)
    rap= optimize.brentq(_rapRperiAxiEq,L,rend,(E,L,pot))
    return (rperi,rap)

def calcELStaeckel(R,vR,vT,z,vz,pot,vc=1.,ro=1.):
    """
    NAME:
       calcELStaeckel
    PURPOSE:
       calculate the energy and angular momentum
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       (E,L)
    HISTORY:
       2012-11-30 - Written - Bovy (IAS)
    """                           
    return (evaluatePotentials(R,z,pot)+vR**2./2.+vT**2./2.+vz**2./2.,R*vT)

def potentialStaeckel(u,v,pot,delta):
    """
    NAME:
       potentialStaeckel
    PURPOSE:
       return the potential
    INPUT:
       u - confocal u
       v - confocal v
       pot - potential
       delta - focus
    OUTPUT:
       Phi(u,v)
    HISTORY:
       2012-11-29 - Written - Bovy (IAS)
    """
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    return evaluatePotentials(R,z,pot)

def FRStaeckel(u,v,pot,delta):
    """
    NAME:
       FRStaeckel
    PURPOSE:
       return the radial force
    INPUT:
       u - confocal u
       v - confocal v
       pot - potential
       delta - focus
    OUTPUT:
       FR(u,v)
    HISTORY:
       2012-11-30 - Written - Bovy (IAS)
    """
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    return evaluateRforces(R,z,pot)

def FZStaeckel(u,v,pot,delta):
    """
    NAME:
       FZStaeckel
    PURPOSE:
       return the vertical force
    INPUT:
       u - confocal u
       v - confocal v
       pot - potential
       delta - focus
    OUTPUT:
       FZ(u,v)
    HISTORY:
       2012-11-30 - Written - Bovy (IAS)
    """
    R,z= bovy_coords.uv_to_Rz(u,v,delta=delta)
    return evaluatezforces(R,z,pot)

def _u0Eq(u,sinv0,cosv0,v0,delta,pot):
    """The equation that needs to be solved to find u0"""
    sinhu= nu.sinh(u)
    coshu= nu.cosh(u)
    dUdu= 2.*sinhu*coshu*potentialStaeckel(u,v0,pot,delta)\
        -delta*(sinhu**2.+sinv0**2.)\
        *(FRStaeckel(u,v0,pot,delta)*coshu*sinv0
          +FZStaeckel(u,v0,pot,delta)*sinhu*cosv0)
    return dUdu

def _JRStaeckelIntegrand(u,E,Lz,I3U,delta,u0,sinh2u0,v0,sin2v0,
                         potu0v0,pot):
    return nu.sqrt(_JRStaeckelIntegrandSquared(u,E,Lz,I3U,delta,u0,sinh2u0,
                                               v0,sin2v0,
                                               potu0v0,pot))
def _JRStaeckelIntegrandSquared(u,E,Lz,I3U,delta,u0,sinh2u0,v0,sin2v0,
                                potu0v0,pot):
    #potu0v0= potentialStaeckel(u0,v0,pot,delta)
    """The J_R integrand: p_u(u)/2/delta^2"""
    sinh2u= nu.sinh(u)**2.
    dU= (sinh2u+sin2v0)*potentialStaeckel(u,v0,pot,delta)\
        -(sinh2u0+sin2v0)*potu0v0
    print u, dU, E*sinh2u-I3U-dU-Lz**2./2./delta**2./sinh2u
    return E*sinh2u-I3U-dU-Lz**2./2./delta**2./sinh2u

def _JzStaeckelIntegrand(v,E,Lz,I3V,delta,u0,cosh2u0,sinh2u0,
                         potu0pi2,pot):
    #potu0pi2= potentialStaeckel(u0,nu.pi/2.,pot,delta)
    """The J_z integrand: p_v(v)/2/delta^2"""
    sin2v= nu.sin(v)**2.
    dV= cosh2u0*potu0pi2\
        -(sinh2u0+sin2v)*potentialStaeckel(u0,v,pot,delta)
    return nu.sqrt(E*sin2v**2.+I3V+dV-Lz**2./2./delta**2./sin2v)

def _rapRperiAxiEq(R,E,L,pot):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E-potentialAxi(R,pot)-L**2./2./R**2.

def _rapRperiAxiDeriv(R,E,L,pot):
    """The derivative of the vr=0 equation that needs to be solved to find 
    apo- and pericenter"""
    return evaluateplanarRforces(R,pot)+L**2./R**3.

def _rapRperiAxiFindStart(R,E,L,pot,rap=False,startsign=1.):
    """
    NAME:
       _rapRperiAxiFindStart
    PURPOSE:
       Find adequate start or end points to solve for rap and rperi
    INPUT:
       R - Galactocentric radius
       E - energy
       L - angular momentum
       pot - potential
       rap - if True, find the rap end-point
       startsign= set to -1 if the function is not positive (due to gamma)
    OUTPUT:
       rstart or rend
    HISTORY:
       2010-12-01 - Written - Bovy (NYU)
    """
    if rap:
        rtry= 2.*R
    else:
        rtry= R/2.
    while startsign*_rapRperiAxiEq(rtry,E,L,pot) > 0. \
            and rtry > 0.000000001:
        if rap:
            if rtry > 100.:
                raise UnboundError("Orbit seems to be unbound")
            rtry*= 2.
        else:
            rtry/= 2.
    if rtry < 0.000000001: return 0.
    return rtry

