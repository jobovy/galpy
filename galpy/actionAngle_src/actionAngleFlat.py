###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleFlat
#
#      methods:
#              JR
#              Jphi
#              angleR
#              TR
#              Tphi
#              I
#              calcRapRperi
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from actionAngle import *
from actionAngleVertical import actionAngleVertical
class actionAngleFlat(actionAngle,actionAngleVertical):
    """Action-angle formalism for flat rotation curves"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngle object
        INPUT:
           Either:
              a) R,vR,vT
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
        OUTPUT:
        HISTORY:
           2010-07-11 - Written - Bovy (NYU)
        """
        actionAngle.__init__(self,*args,**kwargs)
        if kwargs.has_key('verticalPot'):
            actionAngleVertical.__init__(self,*args,pot=kwargs['verticalPot'],
                                         **kwargs)
        return None
    
    def angleR(self,**kwargs):
        """
        NAME:
           AngleR
        PURPOSE:
           Calculate the radial angle for a flat rotation curve
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           w_R(R,vT,vT) in radians + 
           estimate of the error (does not include TR error)
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angleR'):
            return self._angleR
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi:
            return 0.
        TR= self.TR(**kwargs)[0]
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if self._R < Rmean:
            if self._R > rperi:
                wR= (2.*m.pi/TR/m.sqrt(2.)*rperi*
                     nu.array(integrate.quadrature(_TRFlatIntegrandSmall,
                                                   0.,m.sqrt(self._R/rperi-1.),
                                                   args=((self._R*self._vT)**2/rperi**2.,),
                                                   **kwargs)))+nu.array([m.pi,0.])
            else:
                wR= nu.array([m.pi,0.])
        else:
            if self._R < rap:
                wR= -(2.*m.pi/TR/m.sqrt(2.)*rap*
                      nu.array(integrate.quadrature(_TRFlatIntegrandLarge,
                                                    0.,m.sqrt(1.-self._R/rap),
                                                    args=((self._R*self._vT)**2/rap**2.,),
                                                    **kwargs)))
            else:
                wR= nu.array([0.,0.])
        if self._vR < 0.:
            wR[0]+= m.pi
        self._angleR= nu.array([wR[0] % (2.*m.pi),wR[1]])
        return self._angleR

    def TR(self,**kwargs):
        """
        NAME:
           TR
        PURPOSE:
           Calculate the radial period for a flat rotation curve
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           T_R(R,vT,vT)*vc/ro + estimate of the error
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_TR'):
            return self._TR
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi: #Rough limit
            #TR=kappa
            kappa= m.sqrt(2.)/self._R
            self._TR= nu.array([2.*m.pi/kappa,0.])
            return self._TR
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        TR= 0.
        if Rmean > rperi:
            TR+= rperi*nu.array(integrate.quadrature(_TRFlatIntegrandSmall,
                                                     0.,m.sqrt(Rmean/rperi-1.),
                                                     args=((self._R*self._vT)**2/rperi**2.,),
                                                     **kwargs))
        if Rmean < rap:
            TR+= rap*nu.array(integrate.quadrature(_TRFlatIntegrandLarge,
                                                   0.,m.sqrt(1.-Rmean/rap),
                                                   args=((self._R*self._vT)**2/rap**2.,),
                                                   **kwargs))
        self._TR= m.sqrt(2.)*TR
        return self._TR

    def Tphi(self,**kwargs):
        """
        NAME:
           Tphi
        PURPOSE:
           Calculate the azimuthal period for a flat rotation curve
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           T_phi(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_Tphi'):
            return self._Tphi
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi:
            return nu.array([2.*m.pi*self._R/self._vT,0.])
        TR= self.TR(**kwargs)
        I= self.I(**kwargs)
        Tphi= nu.zeros(2)
        Tphi[0]= TR[0]/I[0]*m.pi
        Tphi[1]= Tphi[0]*m.sqrt((I[1]/I[0])**2.+(TR[1]/TR[0])**2.)
        self._Tphi= Tphi
        return self._Tphi

    def I(self,**kwargs):
        """
        NAME:
           I
        PURPOSE:
           Calculate I, the 'ratio' between the radial and azimutha period, 
           for a flat rotation curve
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           I(R,vT,vT) + estimate of the error
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_I'):
            return self._I
        (rperi,rap)= self.calcRapRperi()
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if rap == rperi: #Rough limit
            TR= self.TR()[0]
            Tphi= self.Tphi()[0]
            self._I= nu.array([TR/Tphi,0.])
            return self._I
        I= 0.
        if Rmean > rperi:
            I+= nu.array(integrate.quadrature(_IFlatIntegrandSmall,
                                              0.,m.sqrt(Rmean/rperi-1.),
                                              args=((self._R*self._vT)**2/rperi**2.,),
                                              **kwargs))/rperi
        if Rmean < rap:
            I+= nu.array(integrate.quadrature(_IFlatIntegrandLarge,
                                              0.,m.sqrt(1.-Rmean/rap),
                                              args=((self._R*self._vT)**2/rap**2.,),
                                              **kwargs))/rap
        self._I= I/m.sqrt(2.)*self._R*self._vT
        return self._I

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
           2010-05-13 - Written - Bovy (NYU)
        """
        return nu.array([self._R*self._vT,0.])

    def JR(self,**kwargs):
        """
        NAME:
           JR
        PURPOSE:
           Calculate the radial action for a flat rotation curve
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           J_R(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_JR'):
            return self._JR
        (rperi,rap)= self.calcRapRperi()
        self._JR= (2.*m.sqrt(2.)*rperi*
                   nu.array(integrate.quad(_JRFlatIntegrand,1.,rap/rperi,
                                           args=((self._R*self._vT)**2./rperi**2.),**kwargs)))
        return self._JR

    def calcRapRperi(self):
        """
        NAME:
           calcRapRperi
        PURPOSE:
           calculate the apocenter and pericenter radii for a flat rotation 
           curve
        INPUT:
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_rperirap'):
            return self._rperirap
        EL= calcELFlat(self._R,self._vR,self._vT,vc=1.,ro=1.)
        E, L= EL
        if self._vR == 0. and self._vT > 1.: #We are exactly at pericenter
            rperi= self._R
            rend= _rapRperiFlatFindStart(self._R,E,L,rap=True)
            rap= optimize.brentq(_rapRperiFlatEq,rperi+0.000001,rend,args=EL)
#                                 fprime=_rapRperiFlatDeriv)
        elif self._vR == 0. and self._vT < 1.: #We are exactly at apocenter
            rap= self._R
            rstart= _rapRperiFlatFindStart(self._R,E,L)
            rperi= optimize.brentq(_rapRperiFlatEq,rstart,rap-0.000001,args=EL)
#                                   fprime=_rapRperiFlatDeriv)
        elif self._vR == 0. and self._vT == 1.: #We are on a circular orbit
            rperi= self._R
            rap = self._R
        else:
            rstart= _rapRperiFlatFindStart(self._R,E,L)
            rperi= optimize.brentq(_rapRperiFlatEq,rstart,self._R,EL)
            rend= _rapRperiFlatFindStart(self._R,E,L,rap=True)
            rap= optimize.brentq(_rapRperiFlatEq,self._R,rend,EL)
        self._rperirap= (rperi,rap)
        return self._rperirap

#
# Deprecated interface
#
def angleRFlat(R,vR,vT,vc=1.,ro=1.,rap=None,rperi=None,TR=None,**kwargs):
    """
    NAME:
       AngleRFlat
    PURPOSE:
       Calculate the radial angle for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
       +scipy.integrate.quadrature keywords
    OPTIONAL INPUT:
       rap - apocenter radius (/ro)
       rperi - pericenter radius (/ro)
       TR - radial period (/ro/vc)
    OUTPUT:
       w_R(R,vT,vT) in radians + 
       estimate of the error (does not include TR error)
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap == None or rperi == None:
        (rperi,rap)= calcRapRperiFlat(R,vR,vT,vc=vc,ro=ro)
    if rap == rperi:
        return 0.
    if TR == None:
        TR= TRFlat(R,vR,vT,vc=vc,ro=ro,rap=rap,rperi=rperi,**kwargs)[0]
    Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
    if R < Rmean:
        wR= (2.*m.pi/TR*m.sqrt(2.)*rperi*
             nu.array(integrate.quadrature(_TRFlatIntegrandSmall,
                                           0.,m.sqrt(R/rperi-1.),
                                           args=((R*vT)**2/rperi**2.,),
                                            **kwargs)))+nu.array([m.pi,0.])
    else:
        wR= -(2.*m.pi/TR*m.sqrt(2.)*rap*
              nu.array(integrate.quadrature(_TRFlatIntegrandLarge,
                                            0.,m.sqrt(1.-R/rap),
                                            args=((R*vT)**2/rap**2.,),
                                            **kwargs)))
    if vR < 0.:
        wR[0]+= m.pi
    return nu.array([wR[0] % (2.*m.pi),wR[1]])

def TRFlat(R,vR,vT,vc=1.,ro=1.,rap=None,rperi=None,**kwargs):
    """
    NAME:
       TRFlat
    PURPOSE:
       Calculate the radial period for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
       +scipy.integrate.quadrature keywords
    OPTIONAL INPUT:
       rap - apocenter radius (/ro)
       rperi - pericenter radius (/ro)
    OUTPUT:
       T_R(R,vT,vT)*vc/ro + estimate of the error
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap == None or rperi == None:
        (rperi,rap)= calcRapRperiFlat(R,vR,vT,vc=vc,ro=ro)
    if rap == rperi: #Rough limit
        e= 10.**-6.
        rap= R*(1+e)
        rperi= R*(1-e)
    Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
    TR= 0.
    if Rmean > rperi:
        TR+= rperi*nu.array(integrate.quadrature(_TRFlatIntegrandSmall,
                                                0.,m.sqrt(Rmean/rperi-1.),
                                                args=((R*vT)**2/rperi**2.,),
                                                **kwargs))
    if Rmean < rap:
        TR+= rap*nu.array(integrate.quadrature(_TRFlatIntegrandLarge,
                                               0.,m.sqrt(1.-Rmean/rap),
                                               args=((R*vT)**2/rap**2.,),
                                               **kwargs))
    return m.sqrt(2.)*TR

def TphiFlat(R,vR,vT,vc=1.,ro=1.,rap=None,rperi=None,TR=None,I=None,**kwargs):
    """
    NAME:
       TphiFlat
    PURPOSE:
       Calculate the azimuthal period for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
       +scipy.integrate.quadrature keywords
    OPTIONAL INPUT:
       rap - apocenter radius (/ro)
       rperi - pericenter radius (/ro)
       TR - radial period
       I - "ratio" between radial and azimuthal period (computed using IFlat)
    OUTPUT:
       T_phi(R,vT,vT)/ro/vc + estimate of the error
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap == None or rperi == None:
        (rperi,rap)= calcRapRperiFlat(R,vR,vT,vc=vc,ro=ro)
    if rap == rperi:
        return nu.array([2.*m.pi*R/vT,0.])
    if TR == None:
        TR= TRFlat(R,vR,vT,vc=1.,ro=1.,rap=rap,rperi=rperi,**kwargs)
    if I == None:
        I= IFlat(R,vR,vT,vc=1.,ro=1.,rap=rap,rperi=rperi,**kwargs)
    Tphi= nu.zeros(2)
    Tphi[0]= TR[0]/I[0]*m.pi
    Tphi[1]= Tphi[0]*m.sqrt((I[1]/I[0])**2.+(TR[1]/TR[0])**2.)
    return Tphi

def IFlat(R,vR,vT,vc=1.,ro=1.,rap=None,rperi=None,**kwargs):
    """
    NAME:
       IFlat
    PURPOSE:
       Calculate I, the 'ratio' between the radial and azimutha period, 
       for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
       +scipy.integrate.quadrature keywords
    OPTIONAL INPUT:
       rap - apocenter radius (/ro)
       rperi - pericenter radius (/ro)
    OUTPUT:
       I(R,vT,vT) + estimate of the error
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap == None or rperi == None:
        (rperi,rap)= calcRapRperiFlat(R,vR,vT,vc=vc,ro=ro)
    Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
    if rap == rperi: #Rough limit
        e= 10.**-6.
        rap= R*(1+e)
        rperi= R*(1-e)
    I= 0.
    if Rmean > rperi:
        I+= nu.array(integrate.quadrature(_IFlatIntegrandSmall,
                                          0.,m.sqrt(Rmean/rperi-1.),
                                          args=((R*vT)**2/rperi**2.,),
                                          **kwargs))/rperi
    if Rmean < rap:
        I+= nu.array(integrate.quadrature(_IFlatIntegrandLarge,
                                          0.,m.sqrt(1.-Rmean/rap),
                                          args=((R*vT)**2/rap**2.,),
                                          **kwargs))/rap
    return I/m.sqrt(2.)*R*vT

def Jphi(R,vR,vT):
    """
    NAME:
       Jphi
    PURPOSE:
       Calculate the azimuthal action
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
    OUTPUT:
       J_R(R,vT,vT)/ro/vc
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    return R*vT

def JphiFlat(R,vR,vT):
    """
    NAME:
       Jphi
    PURPOSE:
       Calculate the azimuthal action for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
    OUTPUT:
       J_R(R,vT,vT)/ro/vc
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    return Jphi(R,vR,vT)

def JRFlat(R,vR,vT,vc=1.,ro=1.,rap=None,rperi=None,**kwargs):
    """
    NAME:
       JRFlat
    PURPOSE:
       Calculate the radial action for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
       +scipy.integrate.quad keywords
    OPTIONAL INPUT:
       rap - apocenter radius (/ro)
       rperi - pericenter radius (/ro)
    OUTPUT:
       J_R(R,vT,vT)/ro/vc + estimate of the error
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap == None or rperi == None:
        (rperi,rap)= calcRapRperiFlat(R,vR,vT,vc=vc,ro=ro)
    return (2.*m.sqrt(2.)*rperi*
            nu.array(integrate.quad(_JRFlatIntegrand,1.,rap/rperi,
                           args=((R*vT)**2/rperi**2.),**kwargs)))

def calcRapRperiFlat(R,vR,vT,vc=1.,ro=1.):
    """
    NAME:
       calcRapRperiFlat
    PURPOSE:
       calculate the apocenter and pericenter radii for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       (rperi,rap)
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    EL= calcELFlat(R,vR,vT,vc=vc,ro=ro)
    E, L= EL
    if vR == 0. and vT > vc: #We are exactly at pericenter
        rperi= R
        rend= _rapRperiFlatFindStart(R,E,L,rap=True)
        rap= optimize.newton(_rapRperiFlatEq,rend,args=EL,
                             fprime=_rapRperiFlatDeriv)
    elif vR == 0. and vT < vc: #We are exactly at apocenter
        rap= R
        rstart= _rapRperiFlatFindStart(R,E,L)
        rperi= optimize.newton(_rapRperiFlatEq,rstart,args=EL,
                               fprime=_rapRperiFlatDeriv)
    elif vR == 0. and vT == vc: #We are on a circular orbit
        rperi= R
        rap = R
    else:
        rstart= _rapRperiFlatFindStart(R,E,L)
        rperi= optimize.brentq(_rapRperiFlatEq,rstart,R,EL)
        rend= _rapRperiFlatFindStart(R,E,L,rap=True)
        rap= optimize.brentq(_rapRperiFlatEq,R,rend,EL)
    return (rperi,rap)

def calcRapRperiFromELFlat(E,L,vc=1.,ro=1.):
    """
    NAME:
       calcRapRperiFromELFlat
    PURPOSE:
       calculate the apocenter and pericenter radii for a flat rotation curve
    INPUT:
       E - energy
       L - angular momemtum
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       (rperi,rap)
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    rstart= _rapRperiFlatFindStart(L,E,L)
    rperi= optimize.brentq(_rapRperiFlatEq,rstart,L,(E,L))
    rend= _rapRperiFlatFindStart(L,E,L,rap=True)
    rap= optimize.brentq(_rapRperiFlatEq,L,rend,(E,L))
    return (rperi,rap)

def calcELFlat(R,vR,vT,vc=1.,ro=1.):
    """
    NAME:
       calcELFlat
    PURPOSE:
       calculate the energy and angular momentum for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vR - radial part of the velocity (/vc)
       vT - azimuthal part of the velocity (/vc)
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       (E,L)
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """                           
    return (potentialFlat(R)+vR**2./2.+vT**2./2.,R*vT)

def potentialFlat(R,vc=1.,ro=1.):
    """
    NAME:
       potentialFlat
    PURPOSE:
       return the potential for a flat rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       Phi(R)
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    return vc**2.*m.log(R/ro)

def _JRFlatIntegrand(r,L2rperi2):
    """The J_R integrand for a flat rotation curve"""
    return nu.sqrt(L2rperi2*(1.-1./r**2)/2.-nu.log(r))

def _TRFlatIntegrandSmall(t,L2rperi2):
    r= 1.+t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rperi2)

def _TRFlatIntegrandLarge(t,L2rap2):
    r= 1.-t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rap2) #same integrand

def _IFlatIntegrandSmall(t,L2rperi2):
    r= 1.+t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rperi2)/r**2.

def _IFlatIntegrandLarge(t,L2rap2):
    r= 1.-t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rap2)/r**2.

def _rapRperiFlatEq(R,E,L):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E-potentialFlat(R)-L**2./2./R**2.

def _rapRperiFlatDeriv(R,E,L):
    """The derivative of the vr=0 equation that needs to be solved to find 
    apo- and pericenter"""
    return -1./R+L**2./R**3.

def _rapRperiFlatFindStart(R,E,L,rap=False):
    """
    NAME:
       _rapRperiFlatFindStart
    PURPOSE:
       Find adequate start or end points to solve for rap and rperi
    INPUT:
       R - Galactocentric radius
       E - energy
       L - angular momentum
       rap - if True, find the rap end-point
    OUTPUT:
       rstart or rend
    HISTORY:
       2010-05-13 - Written - Bovy (NYU)
    """
    if rap:
        rtry= 2.*R
    else:
        rtry= R/2.
    while (E-potentialFlat(rtry)-L**2./2./rtry**2) > 0.:
        if rap:
            if rtry > 100.:
                raise UnboundError("Orbit seems to be unbound")
            rtry*= 2.
        else:
            rtry/= 2.
    return rtry

