###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleAxi
#
#      methods:
#              JR
#              Jphi
#              angleR
#              TR
#              Tphi
#              I
#              calcRapRperi
#              calcEL
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from .actionAngle import *
from .actionAngleVertical import actionAngleVertical
from galpy.potential.planarPotential import _evaluateplanarPotentials
from galpy.potential.Potential import epifreq
from galpy.potential import vcirc
_EPS= 10.**-15.
class actionAngleAxi(actionAngleVertical):
    """Action-angle formalism for axisymmetric potentials"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleAxi object
        INPUT:
           Either:
              a) R,vR,vT
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
              pot= potential or list of potentials (planarPotentials)
              verticalPot= the vertical Potential
              gamma= (default=1.) replace Lz by Lz+gamma Jz in effective potential (if there is no vertical potential, this is set to zero)
        OUTPUT:
        HISTORY:
           2010-12-01 - Written - Bovy (NYU)
        """
        self._parse_eval_args(*args,_noOrbUnitsCheck=True,**kwargs)
        self._R= self._eval_R
        self._vR= self._eval_vR
        self._vT= self._eval_vT
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleAxi")
        self._pot= kwargs['pot']
        if 'verticalPot' in kwargs:
            kwargs.pop('pot')
            actionAngleVertical.__init__(self,*args,pot=kwargs['verticalPot'],
                                         **kwargs)
            self._gamma= kwargs.get('gamma',1.)
        else:
            self._gamma= 0.
        return None
    
    def angleR(self,**kwargs): #pragma: no cover
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
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angleR'):
            return self._angleR
        (rperi,rap)= self.calcRapRperi(**kwargs)
        if rap == rperi:
            return 0.
        TR= self.TR(**kwargs)[0]
        EL= self.calcEL(**kwargs)
        E, L= EL
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if self._R < Rmean:
            if self._R > rperi:
                wR= (2.*m.pi/TR*
                     nu.array(integrate.quadrature(_TRAxiIntegrandSmall,
                                                   0.,m.sqrt(self._R-rperi),
                                                   args=(E,L,self._pot,rperi),
                                                   **kwargs)))\
                                                   +nu.array([m.pi,0.])
            else:
                wR= nu.array([m.pi,0.])
        else:
            if self._R < rap:
                wR= -(2.*m.pi/TR*
                      nu.array(integrate.quadrature(_TRAxiIntegrandLarge,
                                                    0.,m.sqrt(rap-self._R),
                                                    args=(E,L,self._pot,rap),
                                                    **kwargs)))
            else:
                wR= nu.array([0.,0.])
        if self._vR < 0.:
            wR[0]+= m.pi
        self._angleR= nu.array([wR[0] % (2.*m.pi),wR[1]])
        return self._angleR

    def TR(self,**kwargs): #pragma: no cover
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
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_TR'):
            return self._TR
        (rperi,rap)= self.calcRapRperi(**kwargs)
        if nu.fabs(rap-rperi)/rap < 10.**-4.: #Rough limit
            self._TR= 2.*m.pi/epifreq(self._pot,self._R,use_physical=False)
            return self._TR
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        EL= self.calcEL(**kwargs)
        E, L= EL
        TR= 0.
        if Rmean > rperi:
            TR+= integrate.quadrature(_TRAxiIntegrandSmall,
                                      0.,m.sqrt(Rmean-rperi),
                                      args=(E,L,self._pot,rperi),
                                      **kwargs)[0]
        if Rmean < rap:
            TR+= integrate.quadrature(_TRAxiIntegrandLarge,
                                      0.,m.sqrt(rap-Rmean),
                                      args=(E,L,self._pot,rap),
                                      **kwargs)[0]
        self._TR= 2.*TR
        return self._TR

    def Tphi(self,**kwargs): #pragma: no cover
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
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_Tphi'):
            return self._Tphi
        (rperi,rap)= self.calcRapRperi(**kwargs)
        if rap == rperi:#Circular orbit
            return 2.*m.pi*self._R/self._vT
        TR= self.TR(**kwargs)
        I= self.I(**kwargs)
        Tphi= TR/I*m.pi
        self._Tphi= Tphi
        return self._Tphi

    def I(self,**kwargs): #pragma: no cover
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
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_I'):
            return self._I
        (rperi,rap)= self.calcRapRperi(**kwargs)
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if nu.fabs(rap-rperi)/rap < 10.**-4.: #Rough limit
            TR= self.TR()[0]
            Tphi= self.Tphi()[0]
            self._I= TR/Tphi*m.pi
            return self._I
        EL= self.calcEL(**kwargs)
        E, L= EL
        I= 0.
        if Rmean > rperi:
            I+= integrate.quadrature(_IAxiIntegrandSmall,
                                     0.,m.sqrt(Rmean-rperi),
                                     args=(E,L,self._pot,rperi),
                                     **kwargs)[0]
        if Rmean < rap:
            I+= integrate.quadrature(_IAxiIntegrandLarge,
                                     0.,m.sqrt(rap-Rmean),
                                     args=(E,L,self._pot,rap),
                                     **kwargs)[0]
        self._I= I*self._R*self._vT
        return self._I

    def Jphi(self): #pragma: no cover
        """
        NAME:
           Jphi
        PURPOSE:
           Calculate the azimuthal action
        INPUT:
        OUTPUT:
           J_R(R,vT,vT)/ro/vc
        HISTORY:
           2010-12-01 - Written - Bovy (NYU)
        """
        return self._R*self._vT

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
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_JR'): #pragma: no cover
            return self._JR
        (rperi,rap)= self.calcRapRperi(**kwargs)
        EL= self.calcEL(**kwargs)
        E, L= EL
        self._JR= 1./nu.pi*integrate.quad(_JRAxiIntegrand,rperi,rap,
                                          args=(E,L,self._pot),
                                          **kwargs)[0]
        return self._JR

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
           2012-07-26 - Written - Bovy (IAS)
        """                           
        E,L= calcELAxi(self._R,self._vR,self._vT,self._pot)
        if self._gamma != 0.:
            #Adjust E
            E-= self._vT**2./2.
            L= m.fabs(L)+self._gamma*self.Jz(**kwargs)
            E+= L**2./2./self._R**2.
        return (E,L)

    def calcRapRperi(self,**kwargs):
        """
        NAME:
           calcRapRperi
        PURPOSE:
           calculate the apocenter and pericenter radii
        INPUT:
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_rperirap'): #pragma: no cover
            return self._rperirap
        EL= self.calcEL(**kwargs)
        E, L= EL
        if self._vR == 0. and m.fabs(self._vT - vcirc(self._pot,self._R,use_physical=False)) < _EPS: #We are on a circular orbit
            rperi= self._R
            rap = self._R
        elif self._vR == 0. and self._vT > vcirc(self._pot,self._R,use_physical=False): #We are exactly at pericenter
            rperi= self._R
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(self._R+10.**-8.,E,L,self._pot)
                startsign/= m.fabs(startsign)
            else: startsign= 1.
            rend= _rapRperiAxiFindStart(self._R,E,L,self._pot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,rperi+0.00001,rend,
                                 args=(E,L,self._pot))
#                                   fprime=_rapRperiAxiDeriv)
        elif self._vR == 0. and self._vT < vcirc(self._pot,self._R,use_physical=False): #We are exactly at apocenter
            rap= self._R
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(self._R-10.**-8.,E,L,self._pot)
                startsign/= m.fabs(startsign)
            else: startsign= 1.
            rstart= _rapRperiAxiFindStart(self._R,E,L,self._pot,
                                          startsign=startsign)
            if rstart == 0.: rperi= 0.
            else:
                rperi= optimize.brentq(_rapRperiAxiEq,rstart,rap-0.000001,
                                       args=(E,L,self._pot))
#                                   fprime=_rapRperiAxiDeriv)
        else:
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(self._R,E,L,self._pot)
                startsign/= m.fabs(startsign)
            else:
                startsign= 1.
            rstart= _rapRperiAxiFindStart(self._R,E,L,self._pot,
                                          startsign=startsign)
            if rstart == 0.: rperi= 0.
            else: 
                try:
                    rperi= optimize.brentq(_rapRperiAxiEq,rstart,self._R,
                                           (E,L,self._pot),
                                           maxiter=200)
                except RuntimeError: #pragma: no cover
                    raise UnboundError("Orbit seems to be unbound")
            rend= _rapRperiAxiFindStart(self._R,E,L,self._pot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,self._R,rend,
                                 (E,L,self._pot))
        self._rperirap= (rperi,rap)
        return self._rperirap

def calcELAxi(R,vR,vT,pot,vc=1.,ro=1.):
    """
    NAME:
       calcELAxi
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
       2010-11-30 - Written - Bovy (NYU)
    """                           
    return (potentialAxi(R,pot)+vR**2./2.+vT**2./2.,R*vT)

def potentialAxi(R,pot,vc=1.,ro=1.):
    """
    NAME:
       potentialAxi
    PURPOSE:
       return the potential
    INPUT:
       R - Galactocentric radius (/ro)
       pot - potential
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       Phi(R)
    HISTORY:
       2010-11-30 - Written - Bovy (NYU)
    """
    return _evaluateplanarPotentials(pot,R)

def _JRAxiIntegrand(r,E,L,pot):
    """The J_R integrand"""
    return nu.sqrt(2.*(E-potentialAxi(r,pot))-L**2./r**2.)

def _TRAxiIntegrandSmall(t,E,L,pot,rperi): #pragma: no cover
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)

def _TRAxiIntegrandLarge(t,E,L,pot,rap): #pragma: no cover
    r= rap-t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)

def _IAxiIntegrandSmall(t,E,L,pot,rperi): #pragma: no cover
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)/r**2.

def _IAxiIntegrandLarge(t,E,L,pot,rap): #pragma: no cover
    r= rap-t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)/r**2.

def _rapRperiAxiEq(R,E,L,pot):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E-potentialAxi(R,pot)-L**2./2./R**2.

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
            if rtry > 100.: #pragma: no cover
                raise UnboundError("Orbit seems to be unbound")
            rtry*= 2.
        else:
            rtry/= 2.
    if rtry < 0.000000001: return 0.
    return rtry

