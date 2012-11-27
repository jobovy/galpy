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
from actionAngle import actionAngle
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
        OUTPUT:
        HISTORY:
           2012-11-27 - Written - Bovy (IAS)
        """
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleStaeckel")
        self._pot= kwargs['pot']
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
        aASingle= actionAngleStaeckelSingle(*args,pot=self._pot)
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
              a) R,vR,vT
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
        raise NotImplementedError("adjust calcEL")
        E,L= calcELAxi(self._R,self._vR,self._vT,self._pot)
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
           2012-11-27 - Written - Bovy (IAS)
        """
        raise NotImplementedError("not sure this is even necessary")
        if hasattr(self,'_rperirap'):
            return self._rperirap
        EL= self.calcEL(**kwargs)
        E, L= EL
        if self._vR == 0. and m.fabs(self._vT - vcirc(self._pot,self._R)) < _EPS: #We are on a circular orbit
            rperi= self._R
            rap = self._R
        elif self._vR == 0. and self._vT > vcirc(self._pot,self._R): #We are exactly at pericenter
            rperi= self._R
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(self._R,E,L,self._pot)
                startsign/= m.fabs(startsign)
            else: startsign= 1.
            rend= _rapRperiAxiFindStart(self._R,E,L,self._pot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,rperi+0.00001,rend,
                                 args=(E,L,self._pot))
#                                   fprime=_rapRperiAxiDeriv)
        elif self._vR == 0. and self._vT < vcirc(self._pot,self._R): #We are exactly at apocenter
            rap= self._R
            if self._gamma != 0.:
                startsign= _rapRperiAxiEq(self._R,E,L,self._pot)
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
                except RuntimeError:
                    raise UnboundError("Orbit seems to be unbound")
            rend= _rapRperiAxiFindStart(self._R,E,L,self._pot,rap=True,
                                        startsign=startsign)
            rap= optimize.brentq(_rapRperiAxiEq,self._R,rend,
                                 (E,L,self._pot))
        self._rperirap= (rperi,rap)
        return self._rperirap

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
    return evaluateplanarPotentials(R,pot)

def _JRAxiIntegrand(r,E,L,pot):
    """The J_R integrand"""
    return nu.sqrt(2.*(E-potentialAxi(r,pot))-L**2./r**2.)

def _TRAxiIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)

def _TRAxiIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)

def _IAxiIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)/r**2.

def _IAxiIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_JRAxiIntegrand(r,E,L,pot)/r**2.

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


