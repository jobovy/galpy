###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSpherical
#
#      methods:
#              J1 DONE
#              J2 DONE
#              J3 DONE
#              angle1 DONE
#              angle2
#              angle3
#              T1 DONE
#              T2 DONE
#              T3 DONE
#              I DONE
#              calcRapRperi DONE
#              calcEL DONE
###############################################################################
import math as m
import numpy as nu
from scipy import optimize, integrate
from actionAngle import *
from actionAngleAxi import actionAngleAxi, potentialAxi
from galpy.potential_src.planarPotential import evaluateplanarRforces,\
    planarPotential, evaluateplanarPotentials
class actionAngleSpherical(actionAngle):
    """Action-angle formalism for spherical potentials"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__ DONE
        PURPOSE:
           initialize an actionAngleSpherical object
        INPUT:
           Either:
              a) R,vR,vT
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
              pot= potential or list of potentials (planarPotentials)
        OUTPUT:
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        actionAngle.__init__(self,*args,**kwargs)
        if not kwargs.has_key('pot'):
            raise IOError("Must specify pot= for actionAngleSpherical")
        self._pot= kwargs['pot']
        #Also set up an actionAngleAxi object for EL and rap/rperi calculations
        axiR= m.sqrt(self._R**2.+self._z**2.)
        axivT= self.J2()[0]/axiR
        axivR= (self._x*self._vx+self._y*self._vy+self._z*self._vz)/axiR
        self._axi= actionAngleAxi(axiR,axivR,axivT,pot=self._pot)
        return None
    
    def angle1(self,**kwargs):
        """
        NAME:
           angle1 DONE
        PURPOSE:
           Calculate the longitude of the ascending node
        INPUT:
        OUTPUT:
           angle1 in radians + 
           estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angle1'): return self._angle1
        if not hasattr(self,'_i'): self._i= m.acos(self._J1/self._J2)
        sinu= 1./self.tan(self._i)/self.tan(self._theta)
        u= m.asin(sinu)
        O= self._phi-u
        if O < 0.: O+= 2.*m.pi
        if O > (2.*m.pi): O-= 2.*m.pi
        if self._J1 >= 0.:
            if (self._z*self._vx-self._x*self._vz) >= 0.: #Ly >= 0
                #second quadrant
                if O < m.pi/2. or O > m.pi: 
                    O= self._phi-m.pi+u
                    if O < 0.: O+= 2.*m.pi
                    if O > (2.*m.pi): O-= 2.*m.pi
            else: #Ly < 0
                #First quadrant
                if O > m.pi/2.:
                    O= self._phi-m.pi+u
                    if O < 0.: O+= 2.*m.pi
                    if O > (2.*m.pi): O-= 2.*m.pi
        else: #Lz < 0
            if (self._z*self._vx-self._x*self._vz) >= 0.: #Ly >= 0
                #third quadrant
                if O < m.pi or O > 3.*m.pi/2.: 
                    O= self._phi-m.pi+u
                    if O < 0.: O+= 2.*m.pi
                    if O > (2.*m.pi): O-= 2.*m.pi
            else:
                #fourth quadrant
                if O > 3.*m.pi/2.:
                    O= self._phi-m.pi+u
                    if O < 0.: O+= 2.*m.pi
                    if O > (2.*m.pi): O-= 2.*m.pi
        self._angle1= nu.array([O,0.])
        return self._angle1

    def angle2(self,**kwargs):
        """
        NAME:
           angle2 DONE
        PURPOSE:
           Calculate the longitude of the ascending node
        INPUT:
        OUTPUT:
           angle2 in radians + 
           estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angle2'): return self._angle2
        if not hasattr(self,'_i'): self._i= m.acos(self._J1/self._J2)
        if self._i == 0.: dstdj2= m.pi/2.
        else: dstdj2= m.asin(m.cos(self._theta)/m.sin(self._i))
        out= self._angle3(**kwargs)*self.T3(**kwargs)[0]/self.T2(**kwargs)[0]
        out[0]+= dstdj2
        #Now add the final piece dsrdj2
        EL= self.calcEL()
        E, L= EL
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if self._axi._R < Rmean:
            if self._axi._R > self.rperi:
                out+= L*nu.array(integrate.quadrature(_ISphericalIntegrandSmall,
                                                      0.,m.sqrt(self._axi._R-rperi),
                                                      args=(E,L,self._pot,rperi),
                                                      **kwargs))
        else:
            if self._axi._R < self._rap:
                out+= L*nu.array(integrate.quadrature(_ISphericalIntegrandLarge,
                                                      0.,m.sqrt(rap-self._axi._R),
                                                      args=(E,L,self._pot,rap),
                                                      **kwargs))
            else:
                out[0]+= m.pi*self.T3(**kwargs)[0]/self.T2(**kwargs)[0]
        if self._axi._vR < 0.:
            out[0]+= m.pi*self.T3(**kwargs)[0]/self.T2(**kwargs)[0]
        self._angle2= out
        return self._angle2

    def angle3(self,**kwargs):
        """
        NAME:
           angle3 DONE
        PURPOSE:
           Calculate the radial angle
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           angle_3 in radians + 
           estimate of the error (does not include TR error)
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_angle3'):
            return self._angle3
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi:
            return 0.
        T3= self.T3(**kwargs)[0]
        EL= self.calcEL()
        E, L= EL
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if self._r < Rmean:
            if self._r > rperi:
                wR= (2.*m.pi/T3*
                     nu.array(integrate.quadrature(_T3SphericalIntegrandSmall,
                                                   0.,m.sqrt(self._axi._R-rperi),
                                                   args=(E,L,self._pot,rperi),
                                                   **kwargs)))\
                                                   +nu.array([m.pi,0.])
            else:
                wR= nu.array([m.pi,0.])
        else:
            if self._r < rap:
                wR= -(2.*m.pi/T3*
                      nu.array(integrate.quadrature(_TRAxiIntegrandLarge,
                                                    0.,m.sqrt(rap-self._axi._R),
                                                    args=(E,L,self._pot,rap),
                                                    **kwargs)))
            else:
                wR= nu.array([0.,0.])
        if self._axi._vR < 0.:
            wR[0]+= m.pi
        self._angle3= nu.array([wR[0] % (2.*m.pi),wR[1]])
        return self._angle3

    def T1(self,**kwargs):
        """
        NAME:
           T1 DONE
        PURPOSE:
           Calculate the period corresponding to the 
           longitude of the ascending node
        INPUT:
        OUTPUT:
           T_1*vc/ro + estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        return [0.,0.]

    def T2(self,**kwargs):
        """
        NAME:
           T2 DONE
        PURPOSE:
           Calculate the second period
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           T_2(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_T2'):
            return self._T2
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi:#Circular orbit
            return nu.array([2.*m.pi*self._axi._R/self._axi._vT,0.])
        T3= self.T3(**kwargs)
        I= self.I(**kwargs)
        T2= nu.zeros(2)
        T2[0]= T3[0]/I[0]*m.pi
        T2[1]= T2[0]*m.sqrt((I[1]/I[0])**2.+(T3[1]/T3[0])**2.)
        self._T2= T2
        return self._T2

    def T3(self,**kwargs):
        """
        NAME:
           T3 DONE
        PURPOSE:
           Calculate the radial period
        INPUT:
           scipy.integrate.quadrature keywords
        OUTPUT:
           T_3(R,vT,vT)*vc/ro + estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_T3'):
            return self._T3
        (rperi,rap)= self.calcRapRperi()
        if rap == rperi: #Rough limit
            raise AttributeError("Not implemented yet")
            #TR=kappa
            gamma= m.sqrt(2./(1.+self._beta))
            kappa= 2.*self._R**(self._beta-1.)/gamma
            self._TR= nu.array([2.*m.pi/kappa,0.])
            return self._TR
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        EL= self.calcEL()
        E, L= EL
        T3= 0.
        if Rmean > rperi:
            T3+= nu.array(integrate.quadrature(_T3SphericalIntegrandSmall,
                                               0.,m.sqrt(Rmean-rperi),
                                               args=(E,L,self._pot,rperi),
                                               **kwargs))
        if Rmean < rap:
            T3+= nu.array(integrate.quadrature(_T3SphericalAxiIntegrandLarge,
                                               0.,m.sqrt(rap-Rmean),
                                               args=(E,L,self._pot,rap),
                                               **kwargs))
        self._T3= 2.*T3
        return m.fabs(self._T3)

    def I(self,**kwargs):
        """
        NAME:
           I DONE
        PURPOSE:
           Calculate I, the 'ratio' between the radial and azimutha period
        INPUT:
           +scipy.integrate.quadrature keywords
        OUTPUT:
           I + estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_I'):
            return self._I
        (rperi,rap)= self.calcRapRperi()
        Rmean= m.exp((m.log(rperi)+m.log(rap))/2.)
        if rap == rperi: #Rough limit
            TR= self.TR()[0]
            Tphi= self.Tphi()[0]
            self._I= nu.array([T3/T2*m.pi,0.])
            return self._I
        EL= self.calcEL()
        E, L= EL
        I= 0.
        if Rmean > rperi:
            I+= nu.array(integrate.quadrature(_ISphericalIntegrandSmall,
                                              0.,m.sqrt(Rmean-rperi),
                                              args=(E,L,self._pot,rperi),
                                              **kwargs))
        if Rmean < rap:
            I+= nu.array(integrate.quadrature(_ISphericalIntegrandLarge,
                                               0.,m.sqrt(rap-Rmean),
                                               args=(E,L,self._pot,rap),
                                               **kwargs))
        self._I= I*self._J2()
        return m.fabs(self._I)

    def J1(self):
        """
        NAME:
           J1 DONE
        PURPOSE:
           Calculate the azimuthal action
        INPUT:
        OUTPUT:
           J_1(R,vT,vT)/ro/vc
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        return nu.array([self._R*self._vT,0.])

    def J2(self):
        """
        NAME:
           J2 DONE
        PURPOSE:
           Calculate the second action
        INPUT:
        OUTPUT:
           J_2(R,vT,vT)/ro/vc
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_J2'):
            return self._J2
        x= self._R*m.cos(self._phi)
        y= self._R*m.sin(self._phi)
        z= self._z
        vx= self._vR*m.cos(self._phi)-self._vT*m.sin(self._phi)
        vy= self._vT*m.cos(self._phi)+self._vR*m.sin(self._phi)
        vx= self._vz
        self._x= x
        self._y= y
        self._vx= vx
        self._vy= vy
        self._J2= m.sqrt((y*vz-z*vy)**2.+(z*vx-x*vz)**2.+(x*vy-y*vx)**2.)
        return nu.array([self._J2,0.])

    def J3(self,**kwargs):
        """
        NAME:
           J3 DONE
        PURPOSE:
           Calculate the radial action
        INPUT:
           +scipy.integrate.quad keywords
        OUTPUT:
           J_3(R,vT,vT)/ro/vc + estimate of the error
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        if hasattr(self,'_J3'):
            return self._J3
        (rperi,rap)= self.calcRapRperi()
        EL= self.calcEL()
        E, L= EL
        self._J3= (2.*nu.array(integrate.quad(_J3SphericalIntegrand,rperi,rap,
                                              args=(E,L,self._pot),
                                              **kwargs)))
        return self._J3

    def calcEL(self):
        """
        NAME:
           calcEL DONE
        PURPOSE:
           calculate the energy and the angular momentum
        INPUT:
        OUTPUT:
           (E.L)
        HISTORY:
           2011-03-03 - Written - Bovy (NYU)
        """
        return (potentialAxi(m.sqrt(self._R**2.+self._z**2.),self._pot)\
                    +self._vR**2./2.+self._vT**2./2.+self._vz**2./2.,
                self._J2()[0])

    def calcRapRperi(self):
        """
        NAME:
           calcRapRperi DONE
        PURPOSE:
           calculate the apocenter and pericenter radii
        INPUT:
        OUTPUT:
           (rperi,rap)
        HISTORY:
           2010-12-01 - Written - Bovy (NYU)
        """
        if hasattr(self,'_rperirap'):
            return self._rperirap
        #Use the actionAngleAxi rap and rperi
        self._rperirap= self._axi.calcRapRperi()
        return self._rperirap

def _J3SphericalIntegrand(r,E,L,pot):
    """The J_3 integrand DONE"""
    return nu.sqrt(2.*(E-potentialAxi(r,pot))-L**2./r**2.)

def _T3SphericalIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_J3SphericalIntegrand(r,E,L,pot)

def _T3SphericalIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_J3SphericalIntegrand(r,E,L,pot)

def _ISphericalIntegrandSmall(t,E,L,pot,rperi):
    r= rperi+t**2.#part of the transformation
    return 2.*t/_J3SphericalIntegrand(r,E,L,pot)/r**2.

def _ISphericalIntegrandLarge(t,E,L,pot,rap):
    r= rap-t**2.#part of the transformation
    return 2.*t/_J3SphericalIntegrand(r,E,L,pot)/r**2.
