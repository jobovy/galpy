###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochrone
#
#             Calculate actions-angle coordinates for the Isochrone potential
#
#      methods:
#             __call__: returns (jr,lz,jz)
#
###############################################################################
import numpy as nu
from actionAngle import actionAngle, UnboundError
from galpy.potential import IsochronePotential
class actionAngleIsochrone():
    """Action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008)"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleIsochrone object
        INPUT:
           Either:
              b= scale parameter of the isochrone parameter
              ip= instance of a IsochronePotential
        OUTPUT:
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        if not kwargs.has_key('b') and not kwargs.has_key('ip'):
            raise IOError("Must specify b= for actionAngleIsochrone")
        if kwargs.has_key('ip'):
            ip= kwargs['ip']
            if not isinstance(ip,IsochronePotential):
                raise IOError("'Provided ip= does not appear to be an instance of an IsochronePotential")
            self.b= ip.b
            self.amp= ip._amp
        else:
            self.b= kwargs['b']
            rb= nu.sqrt(self.b**2.+1.)
            self.amp= (self.b+rb)**2.*rb
        self._c= False
        ext_loaded= False
        if ext_loaded and ((kwargs.has_key('c') and kwargs['c'])
                           or not kwargs.has_key('c')):
            self._c= True
        else:
            self._c= False
        if not self._c:
            self._ip= IsochronePotential(normalize=1.,b=self.b)
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
           (jr,lz,jz)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        if len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            meta= actionAngle(*args)
            R= meta._R
            vR= meta._vR
            vT= meta._vT
            z= meta._z
            vz= meta._vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c:
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            return (Jr,Jphi,Jz)

    def actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        if len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            meta= actionAngle(*args)
            R= meta._R
            vR= meta._vR
            vT= meta._vT
            z= meta._z
            vz= meta._vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c:
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            Omegar= (-2.*E)**1.5/self.amp
            Omegaz= 0.5*(1.+L/nu.sqrt(L2+4.*self.amp*self.b))*Omegar
            if Lz > 0.:
                Omegaphi= Omegaz
            else:
                Omegaphi= -Omegaz
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz)

    def actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles
        PURPOSE:
           evaluate the actions, frequencies, and angles 
           (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        INPUT:
           Either:
              a) R,vR,vT,z,vz,phi (MUST HAVE PHI)
              b) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well
           scipy.integrate.quadrature keywords
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        if len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            meta= actionAngle(*args)
            R= meta._R
            vR= meta._vR
            vT= meta._vT
            z= meta._z
            vz= meta._vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c:
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            Omegar= (-2.*E)**1.5/self.amp
            Omegaz= 0.5*(1.+L/nu.sqrt(L2+4.*self.amp*self.b))*Omegar
            Omegaphi= Omegaz
            indx= Lz < 0.
            Omegaphi[indx]*= -1.
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz)

