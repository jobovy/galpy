###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAnglePower
#
#      methods:
#              JR
#              Jphi
#              angleR
#              TR
#              Tphi
#              I
#              calcRapRperi
#              potential
#              calcEL
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from actionAngle import *
class actionAnglePower(actionAngle):
    """Action-angle formalism for power-law rotation curves"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAnglePower object, power-law is vc=vo(r)^beta
        INPUT:
           Either:
              a) R,vR,vT
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           beta= power-law index
        OUTPUT:
        HISTORY:
           2010-07-11 - Written - Bovy (NYU)
        """
        actionAngle.__init__(self,*args,**kwargs)
        if not kwargs.has_key('beta'):
            raise InputError("Must specify beta= for actionAnglePower")
        self._beta= kwargs['beta']
        if self._beta < 0.:
            self._signbeta= -1
        else:
            self._signbeta= 1
        self._absbeta= nu.fabs(self._beta)
        (rperi,rap)= self.calcRapRperi()
        self._e= (rap-rperi)/(rap+rperi)
        self._X= (1.-self._e)**2./4./self._e\
            *(((1.+self._e)/(1.-self._e))**(2.self._beta+2.)-1.)
        self._Y= (1.+self._e)**2./4./self._e\
            *(1.-((1.-self._e)/(1.+self._e))**(2.self._beta+2.))
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
           2010-05-13 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angleR'):
            return self._angleR
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi:
            return 0.
        TR= self.TR(**kwargs)[0]
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if self._R < Rmean:BOVY
            if self._R > rperi:
                wR= (2.*m.pi/TR*m.sqrt(self._absbeta)*rperi**(1.-self._beta)*
                     nu.array(integrate.quadrature(_TRPowerIntegrandSmall,
                                                   0.,m.sqrt(self._R/rperi-1.),
                                                   args=((self._R*self._vT)**2/rperi**2.,),
                                                   **kwargs)))+nu.array([m.pi,0.])
            else:
                wR= nu.array([m.pi,0.])
        else:
            if self._R < rap:
                wR= -(2.*m.pi/TR*m.sqrt(2.)*rap*
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
        return (self._R*self._vT,0.)

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
           calculate the apocenter and pericenter radii for a power-law 
           rotation curve
        INPUT:
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2010-11-30 - Written - Bovy (NYU)
        """
        if hasattr(self,'_rperirap'):
            return self._rperirap
        EL= calcELFlat(self._R,self._vR,self._vT,self._beta,vc=1.,ro=1.)
        E, L= EL
        if self._vR == 0. and self._vT > 1.: #We are exactly at pericenter
            rperi= self._R
            rend= _rapRperiPowerFindStart(self._R,E,L,self._beta,rap=True)
            rap= optimize.newton(_rapRperiPowerEq,rend,args=(E,L,self._beta),
                                 fprime=_rapRperiPowerDeriv)
        elif self._vR == 0. and self._vT < 1.: #We are exactly at apocenter
            rap= self._R
            rstart= _rapRperiPowerFindStart(self._R,E,L,self._beta)
            rperi= optimize.newton(_rapRperiPowerEq,rstart,
                                   args=(E,L,self._beta),
                                   fprime=_rapRperiPowerDeriv)
        elif self._vR == 0. and self._vT == 1.: #We are on a circular orbit
            rperi= self._R
            rap = self._R
        else:
            rstart= _rapRperiPowerFindStart(self._R,E,L,self._beta)
            rperi= optimize.brentq(_rapRperiPowerEq,rstart,self._R,
                                   (E,L,self._beta))
            rend= _rapRperiPowerFindStart(self._R,E,L,self._beta,rap=True)
            rap= optimize.brentq(_rapRperiPowerEq,self._R,rend,
                                 (E,L,self._beta))
        self._rperirap= (rperi,rap)
        return self._rperirap

def calcELpower(R,vR,vT,beta,vc=1.,ro=1.):
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
       2010-11-30 - Written - Bovy (NYU)
    """                           
    return (potentialPower(R,beta,vc=vc,ro=ro)+vR**2./2.+vT**2./2.,R*vT)

def potentialPower(R,beta,vc=1.,ro=1.):
    """
    NAME:
       potentialPower
    PURPOSE:
       return the potential for a power-law rotation curve
    INPUT:
       R - Galactocentric radius (/ro)
       beta - power-law index
       vc - circular velocity
       ro - reference radius
    OUTPUT:
       Phi(R)
    HISTORY:
       2010-11-30 - Written - Bovy (NYU)
    """
    return vc**2./2./beta*(R/ro)**(2.*beta)

def _JRFlatIntegrand(r,L2rperi2):
    """The J_R integrand for a flat rotation curve"""
    return nu.sqrt(L2rperi2*(1.-1./r**2)/2.-nu.log(r))

def _TRFlatIntegrandSmall(t,L2rperi2):
    r= 1.+t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rperi2)

def _TRPowerIntegrandLarge(t,L2rap2):
    r= 1.-t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rap2) #same integrand

def _IFlatIntegrandSmall(t,L2rperi2):
    r= 1.+t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rperi2)/r**2.

def _IFlatIntegrandLarge(t,L2rap2):
    r= 1.-t**2.#part of the transformation
    return 2.*t/_JRFlatIntegrand(r,L2rap2)/r**2.

def _rapRperiPowerEq(R,E,L,beta):
    """The vr=0 equation that needs to be solved to find apo- and pericenter"""
    return E-potentialPower(R,beta)-L**2./2./R**2.

def _rapRperiPowerDeriv(R,E,L,beta):
    """The derivative of the vr=0 equation that needs to be solved to find 
    apo- and pericenter"""
    return -R**(2.*beta-1.)+L**2./R**3.

def _rapRperiFlatPowerStart(R,E,L,beta,rap=False):
    """
    NAME:
       _rapRperiPowerFindStart
    PURPOSE:
       Find adequate start or end points to solve for rap and rperi
    INPUT:
       R - Galactocentric radius
       E - energy
       L - angular momentum
       beta - power-law index of rotation curve
       rap - if True, find the rap end-point
    OUTPUT:
       rstart or rend
    HISTORY:
       2010-11-30 - Written - Bovy (NYU)
    """
    if rap:
        rtry= 2.*R
    else:
        rtry= R/2.
    while (E-potentialPower(rtry,beta)-L**2./2./rtry**2) > 0.:
        if rap:
            rtry*= 2.
        else:
            rtry/= 2.
    return rtry

