###############################################################################
#   OblateStaeckelWrapperPotential.py: Wrapper to turn an axisymmetric 
#                                      potential into an oblate Staeckel
#                                      potential following Binney (2012)
#
#   NOT A TYPICAL WRAPPER, SO DON'T USE THIS BLINDLY AS A TEMPLATE FOR NEW 
#   WRAPPERS
#
###############################################################################
import numpy
from galpy.potential_src.Potential import _evaluatePotentials, \
    _evaluateRforces, _evaluatezforces, evaluateR2derivs, evaluateRzderivs, \
    evaluatez2derivs
from galpy.potential_src.WrapperPotential import parentWrapperPotential
from galpy.potential_src.Potential import _APY_LOADED
from galpy.util import bovy_coords
if _APY_LOADED:
    from astropy import units
class OblateStaeckelWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that approximates a given axisymmetric potential as an oblate Staeckel potential, following the scheme of Binney (2012)"""
    def __init__(self,amp=1.,pot=None,delta=0.5,u0=0.,ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a OblateStaeckelWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; this potential is made to rotate around the z axis by the wrapper

           delta= (0.5) the focal length

           u0= (None) reference u value

        OUTPUT:

           (none)

        HISTORY:

           2017-12-15 - Started - Bovy (UofT)

        """
        if _APY_LOADED and isinstance(delta,units.Quantity):
            delta= delta.to(units.kpc).value/self._ro
        self._delta= delta
        if u0 is None: # pragma: no cover
            raise ValueError('u0= needs to be given to setup OblateStaeckelWrapperPotential')
        self._u0= u0
        self._v0= numpy.pi/2. # so we know when we're using this
        R0,z0= bovy_coords.uv_to_Rz(self._u0,self._v0,delta=self._delta)
        self._refpot= _evaluatePotentials(self._pot,R0,z0)\
            *numpy.cosh(self._u0)**2.
        self.hasC= True
        self.hasC_dxdv= False

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        return (self._U(u)-self._V(v))/_staeckel_prefactor(u,v)

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        prefac= _staeckel_prefactor(u,v)
        dprefacdu, dprefacdv= _dstaeckel_prefactordudv(u,v)
        return ((-self._dUdu(u)*self._delta*numpy.sin(v)*numpy.cosh(u)
                  +self._dVdv(v)*numpy.tanh(u)*z
                  +(self._U(u)-self._V(v))
                  *(dprefacdu*self._delta*numpy.sin(v)*numpy.cosh(u)
                    +dprefacdv*numpy.tanh(u)*z)/prefac)
                /self._delta**2./prefac**2.)

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2017-12-15 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        prefac= _staeckel_prefactor(u,v)
        dprefacdu, dprefacdv= _dstaeckel_prefactordudv(u,v)
        return ((-self._dUdu(u)*R/numpy.tan(v)
                  -self._dVdv(v)*self._delta*numpy.sin(v)*numpy.cosh(u)
                  +(self._U(u)-self._V(v))
                  *(dprefacdu/numpy.tan(v)*R
                    -dprefacdv*self._delta*numpy.sin(v)*numpy.cosh(u))/prefac)
                /self._delta**2./prefac**2.)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the 2nd radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the 2nd radial derivative
        HISTORY:
           2017-01-21 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        prefac= _staeckel_prefactor(u,v)
        dprefacdu, dprefacdv= _dstaeckel_prefactordudv(u,v)
        d2prefacdu2, d2prefacdv2= _dstaeckel_prefactord2ud2v(u,v)
        umvfac= (dprefacdu*self._delta*numpy.sin(v)*numpy.cosh(u)
                  +dprefacdv*numpy.tanh(u)*z)/prefac # xs (U-V) in Rforce
        U= self._U(u)
        dUdu= self._dUdu(u)
        d2Udu2= self._d2Udu2(u)
        V= self._V(v)
        dVdv= self._dVdv(v)
        d2Vdv2= self._d2Vdv2(v)
        return ((d2Udu2*numpy.sin(v)**2.*numpy.cosh(u)**2.
                  +dUdu*numpy.sinh(u)*numpy.cosh(u)
                 -d2Vdv2*numpy.sinh(u)**2.*numpy.cos(v)**2.
                 -dVdv*numpy.sin(v)*numpy.cos(v)
                 +((-dUdu*numpy.cosh(u)*numpy.sin(v)
                     +dVdv*numpy.sinh(u)*numpy.cos(v))/self._delta*umvfac
                  +(U-V)*(-d2prefacdu2*numpy.cosh(u)**2.*numpy.sin(v)**2.
                           -dprefacdu*numpy.sinh(u)*numpy.cosh(u)
                           -d2prefacdv2*numpy.sinh(u)**2.*numpy.cos(v)**2.
                           -dprefacdv*numpy.sin(v)*numpy.cos(v))/prefac
                  +(U-V)*umvfac/prefac/self._delta*(dprefacdu*numpy.cosh(u)*numpy.sin(v)+dprefacdv*numpy.sinh(u)*numpy.cos(v))))
                /self._delta**2./prefac**3.
                +2.*self._Rforce(R,z,phi=phi,t=t)/prefac**2.*(dprefacdu*numpy.cosh(u)*numpy.sin(v)+dprefacdv*numpy.sinh(u)*numpy.cos(v))/self._delta)
   
    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the 2nd vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the 2nd vertical derivative
        HISTORY:
           2017-01-21 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        prefac= _staeckel_prefactor(u,v)
        dprefacdu, dprefacdv= _dstaeckel_prefactordudv(u,v)
        d2prefacdu2, d2prefacdv2= _dstaeckel_prefactord2ud2v(u,v)
        umvfac= (dprefacdu/numpy.tan(v)*R # xs (U-V) in zforce
                 -dprefacdv*self._delta*numpy.sin(v)*numpy.cosh(u))/prefac
        U= self._U(u)
        dUdu= self._dUdu(u)
        d2Udu2= self._d2Udu2(u)
        V= self._V(v)
        dVdv= self._dVdv(v)
        d2Vdv2= self._d2Vdv2(v)
        return ((d2Udu2*numpy.sinh(u)**2.*numpy.cos(v)**2.
                 +dUdu*numpy.cosh(u)*numpy.sinh(u)
                -d2Vdv2*numpy.sin(v)**2.*numpy.cosh(u)**2.
                -dVdv*numpy.cos(v)*numpy.sin(v)
                 +((-dUdu*numpy.sinh(u)*numpy.cos(v)
                    -dVdv*numpy.cosh(u)*numpy.sin(v))/self._delta*umvfac
                  +(U-V)*(-d2prefacdu2*numpy.sinh(u)**2.*numpy.cos(v)**2.
                           -dprefacdu*numpy.sinh(u)*numpy.cosh(u)
                           -d2prefacdv2*numpy.sin(v)**2.*numpy.cosh(u)**2.
                           -dprefacdv*numpy.cos(v)*numpy.sin(v))/prefac
                   -(U-V)*umvfac/prefac/self._delta*(-dprefacdu*numpy.sinh(u)*numpy.cos(v)+dprefacdv*numpy.cosh(u)*numpy.sin(v))))
                /self._delta**2./prefac**3.
                -2.*self._zforce(R,z,phi=phi,t=t)/prefac**2.*(-dprefacdu*numpy.sinh(u)*numpy.cos(v)+dprefacdv*numpy.cosh(u)*numpy.sin(v))/self._delta)

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed radial and vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the mixed radial and vertical derivative
        HISTORY:
           2017-01-22 - Written - Bovy (UofT)
        """
        u,v= bovy_coords.Rz_to_uv(R,z,delta=self._delta)
        prefac= _staeckel_prefactor(u,v)
        dprefacdu, dprefacdv= _dstaeckel_prefactordudv(u,v)
        d2prefacdu2, d2prefacdv2= _dstaeckel_prefactord2ud2v(u,v)
        umvfac= (dprefacdu/numpy.tan(v)*R # xs (U-V) in zforce
                 -dprefacdv*self._delta*numpy.sin(v)*numpy.cosh(u))/prefac
        U= self._U(u)
        dUdu= self._dUdu(u)
        d2Udu2= self._d2Udu2(u)
        V= self._V(v)
        dVdv= self._dVdv(v)
        d2Vdv2= self._d2Vdv2(v)
        return (((d2Udu2+d2Vdv2)*numpy.cosh(u)*numpy.sin(v)*numpy.cos(v)*numpy.sinh(u)
                 +dUdu*numpy.sin(v)*numpy.cos(v)
                 +dVdv*numpy.sinh(u)*numpy.cosh(u)
                 +((-dUdu*numpy.cosh(u)*numpy.sin(v)
                     +dVdv*numpy.sinh(u)*numpy.cos(v))/self._delta*umvfac
                  +(U-V)*((-d2prefacdu2+d2prefacdv2)*numpy.sin(v)*numpy.cosh(u)*numpy.sinh(u)*numpy.cos(v)
                          -dprefacdu*numpy.sin(v)*numpy.cos(v)
                          +dprefacdv*numpy.cosh(u)*numpy.sinh(u))/prefac
                   +(U-V)*umvfac/prefac/self._delta*(dprefacdu*numpy.cosh(u)*numpy.sin(v)+dprefacdv*numpy.sinh(u)*numpy.cos(v))))
                /self._delta**2./prefac**3.
                +2.*self._zforce(R,z,phi=phi,t=t)/prefac**2.*(dprefacdu*numpy.cosh(u)*numpy.sin(v)+dprefacdv*numpy.sinh(u)*numpy.cos(v))/self._delta)

    def _U(self,u):
        """Approximated U(u) = cosh^2(u) Phi(u,pi/2)"""
        Rz0= bovy_coords.uv_to_Rz(u,self._v0,delta=self._delta)        
        return numpy.cosh(u)**2.*_evaluatePotentials(self._pot,Rz0[0],Rz0[1])

    def _dUdu(self,u):
        Rz0= bovy_coords.uv_to_Rz(u,self._v0,delta=self._delta)        
        # 1e-12 bc force should win the 0/0 battle
        return 2.*numpy.cosh(u)*numpy.sinh(u)\
            *_evaluatePotentials(self._pot,Rz0[0],Rz0[1])\
            -numpy.cosh(u)**2.\
            *(_evaluateRforces(self._pot,Rz0[0],Rz0[1])*Rz0[0]/(numpy.tanh(u)+1e-12)
              +_evaluatezforces(self._pot,Rz0[0],Rz0[1])*Rz0[1]*numpy.tanh(u))

    def _d2Udu2(self,u):
        Rz0= bovy_coords.uv_to_Rz(u,self._v0,delta=self._delta)
        tRforce= _evaluateRforces(self._pot,Rz0[0],Rz0[1])
        tzforce= _evaluatezforces(self._pot,Rz0[0],Rz0[1])
        return 2.*numpy.cosh(2*u)*_evaluatePotentials(self._pot,Rz0[0],Rz0[1])\
            -4.*numpy.cosh(u)*numpy.sinh(u)\
            *(tRforce*Rz0[0]/(numpy.tanh(u)+1e-12)
              +tzforce*Rz0[1]*numpy.tanh(u))\
              -numpy.cosh(u)**2.\
              *(-evaluateR2derivs(self._pot,Rz0[0],Rz0[1],use_physical=False)*Rz0[0]**2./(numpy.tanh(u)+1e-12)**2.
                 -2.*evaluateRzderivs(self._pot,Rz0[0],Rz0[1],use_physical=False)*Rz0[0]*Rz0[1]
                 +tRforce*Rz0[0]
                 -evaluatez2derivs(self._pot,Rz0[0],Rz0[1],use_physical=False)*Rz0[1]**2.*numpy.tanh(u)**2.
                 +tzforce*Rz0[1])

    def _V(self,v):
        """Approximated 
        V(v) = cosh^2(u0) Phi(u0,pi/2) - (sinh^2(u0)+sin^2(v)) Phi(u0,v)"""
        R0z= bovy_coords.uv_to_Rz(self._u0,v,delta=self._delta)        
        return self._refpot-_staeckel_prefactor(self._u0,v)\
            *_evaluatePotentials(self._pot,R0z[0],R0z[1])

    def _dVdv(self,v):
        R0z= bovy_coords.uv_to_Rz(self._u0,v,delta=self._delta)        
        return -2.*numpy.sin(v)*numpy.cos(v)\
            *_evaluatePotentials(self._pot,R0z[0],R0z[1])\
            +_staeckel_prefactor(self._u0,v)\
            *(_evaluateRforces(self._pot,R0z[0],R0z[1])*R0z[0]/numpy.tan(v)
              -_evaluatezforces(self._pot,R0z[0],R0z[1])*R0z[1]*numpy.tan(v))

    def _d2Vdv2(self,v):
        R0z= bovy_coords.uv_to_Rz(self._u0,v,delta=self._delta)        
        tRforce= _evaluateRforces(self._pot,R0z[0],R0z[1])
        tzforce= _evaluatezforces(self._pot,R0z[0],R0z[1])
        return -2.*numpy.cos(2.*v)\
            *_evaluatePotentials(self._pot,R0z[0],R0z[1])\
            +2.*numpy.sin(2.*v)\
            *(tRforce*R0z[0]/numpy.tan(v)
              -tzforce*R0z[1]*numpy.tan(v))\
              +_staeckel_prefactor(self._u0,v)\
              *(-evaluateR2derivs(self._pot,R0z[0],R0z[1],use_physical=False)*R0z[0]**2./numpy.tan(v)**2.
                 +2.*evaluateRzderivs(self._pot,R0z[0],R0z[1],use_physical=False)*R0z[0]*R0z[1]
                 -tRforce*R0z[0]
                 -evaluatez2derivs(self._pot,R0z[0],R0z[1],use_physical=False)*R0z[1]**2.*numpy.tan(v)**2.
                 -tzforce*R0z[1])
    
def _staeckel_prefactor(u,v):
    return numpy.sinh(u)**2.+numpy.sin(v)**2.
def _dstaeckel_prefactordudv(u,v):
    return (2.*numpy.sinh(u)*numpy.cosh(u),2.*numpy.sin(v)*numpy.cos(v))
def _dstaeckel_prefactord2ud2v(u,v):
    return (2.*numpy.cosh(2.*u),2.*numpy.cos(2.*v))
