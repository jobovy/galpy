###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochrone
#
#             Calculate actions-angle coordinates for the Isochrone potential
#
#      methods:
#             __call__: returns (jr,lz,jz)
#             actionsFreqs: returns (jr,lz,jz,Or,Op,Oz)
#             actionsFreqsAngles: returns (jr,lz,jz,Or,Op,Oz,ar,ap,az)
#
###############################################################################
import copy
import warnings
import numpy as nu
from .actionAngle import actionAngle
from galpy.potential import IsochronePotential
from galpy.util import galpyWarning
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleIsochrone(actionAngle):
    """Action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008)"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngleIsochrone object
        INPUT:
           Either:

              b= scale parameter of the isochrone parameter (can be Quantity)

              ip= instance of a IsochronePotential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        actionAngle.__init__(self,
                             ro=kwargs.get('ro',None),vo=kwargs.get('vo',None))
        if not 'b' in kwargs and not 'ip' in kwargs: #pragma: no cover
            raise IOError("Must specify b= for actionAngleIsochrone")
        if 'ip' in kwargs:
            ip= kwargs['ip']
            if not isinstance(ip,IsochronePotential): #pragma: no cover
                raise IOError("'Provided ip= does not appear to be an instance of an IsochronePotential")
            # Check the units
            self._pot= ip
            self._check_consistent_units()
            self.b= ip.b
            self.amp= ip._amp
        else:
            self.b= kwargs['b']
            if _APY_LOADED and isinstance(self.b,units.Quantity):
                self.b= self.b.to(units.kpc).value/self._ro
            rb= nu.sqrt(self.b**2.+1.)
            self.amp= (self.b+rb)**2.*rb
        self._c= False
        ext_loaded= False
        if ext_loaded and (('c' in kwargs and kwargs['c'])
                           or not 'c' in kwargs): #pragma: no cover
            self._c= True
        else:
            self._c= False
        if not self._c:
            self._ip= IsochronePotential(amp=self.amp,b=self.b)
        #Define _pot, because some functions that use actionAngle instances need this
        self._pot= IsochronePotential(amp=self.amp,b=self.b)
        # Check the units
        self._check_consistent_units()
        return None
    
    def _evaluate(self,*args,**kwargs):
        """
        NAME:
           __call__ (_evaluate)
        PURPOSE:
           evaluate the actions (jr,lz,jz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
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
            self._parse_eval_args(*args)
            R= self._eval_R
            vR= self._eval_vR
            vT= self._eval_vT
            z= self._eval_z
            vz= self._eval_vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            return (Jr,Jphi,Jz)

    def _actionsFreqs(self,*args,**kwargs):
        """
        NAME:
           actionsFreqs (_actionsFreqs)
        PURPOSE:
           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
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
            self._parse_eval_args(*args)
            R= self._eval_R
            vR= self._eval_vR
            vT= self._eval_vT
            z= self._eval_z
            vz= self._eval_vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            #Frequencies
            Omegar= (-2.*E)**1.5/self.amp
            Omegaz= 0.5*(1.+L/nu.sqrt(L2+4.*self.amp*self.b))*Omegar
            Omegaphi= copy.copy(Omegaz)
            indx= Lz < 0.
            Omegaphi[indx]*= -1.
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz)

    def _actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:
           actionsFreqsAngles (_actionsFreqsAngles)
        PURPOSE:
           evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)
        HISTORY:
           2013-09-08 - Written - Bovy (IAS)
        """
        if len(args) == 5: #R,vR.vT, z, vz pragma: no cover
            raise IOError("You need to provide phi when calculating angles")
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            self._parse_eval_args(*args)
            R= self._eval_R
            vR= self._eval_vR
            vT= self._eval_vT
            z= self._eval_z
            vz= self._eval_vz
            phi= self._eval_phi
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
            phi= nu.array([phi])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            L= nu.sqrt(L2)
            #Actions
            Jphi= Lz
            Jz= L-nu.fabs(Lz)
            Jr= self.amp/nu.sqrt(-2.*E)\
                -0.5*(L+nu.sqrt((L2+4.*self.amp*self.b)))
            #Frequencies
            Omegar= (-2.*E)**1.5/self.amp
            Omegaz= 0.5*(1.+L/nu.sqrt(L2+4.*self.amp*self.b))*Omegar
            Omegaphi= copy.copy(Omegaz)
            indx= Lz < 0.
            Omegaphi[indx]*= -1.
            #Angles
            c= -self.amp/2./E-self.b
            e2= 1.-L2/self.amp/c*(1.+self.b/c)
            e= nu.sqrt(e2)
            if self.b == 0.:
                coseta= 1/e*(1.-nu.sqrt(R**2.+z**2.)/c)
            else:
                s= 1.+nu.sqrt(1.+(R**2.+z**2.)/self.b**2.)
                coseta= 1/e*(1.-self.b/c*(s-2.))
            pindx= (coseta > 1.)*(coseta < (1.+10.**-7.))
            coseta[pindx]= 1.
            pindx= (coseta < -1.)*(coseta > (-1.-10.**-7.))
            coseta[pindx]= -1.           
            eta= nu.arccos(coseta)
            costheta= z/nu.sqrt(R**2.+z**2.)
            sintheta= R/nu.sqrt(R**2.+z**2.)
            vrindx= (vR*sintheta+vz*costheta) < 0.
            eta[vrindx]= 2.*nu.pi-eta[vrindx]
            angler= eta-e*c/(c+self.b)*nu.sin(eta)
            tan11= nu.arctan(nu.sqrt((1.+e)/(1.-e))*nu.tan(0.5*eta))
            tan12= nu.arctan(nu.sqrt((1.+e+2.*self.b/c)/(1.-e+2.*self.b/c))*nu.tan(0.5*eta))
            vzindx= (-vz*sintheta+vR*costheta) > 0.
            tan11[tan11 < 0.]+= nu.pi
            tan12[tan12 < 0.]+= nu.pi
            pindx= (Lz/L > 1.)*(Lz/L < (1.+10.**-7.))
            Lz[pindx]= L[pindx]
            pindx= (Lz/L < -1.)*(Lz/L > (-1.-10.**-7.))
            Lz[pindx]= -L[pindx]
            i= nu.arccos(Lz/L)
            sinpsi= costheta/nu.sin(i)
            pindx= (sinpsi > 1.)*(sinpsi < (1.+10.**-7.))
            sinpsi[pindx]= 1.
            pindx= (sinpsi < -1.)*(sinpsi > (-1.-10.**-7.))
            sinpsi[pindx]= -1.           
            psi= nu.arcsin(sinpsi)
            psi[vzindx]= nu.pi-psi[vzindx]
            psi= psi % (2.*nu.pi)
            anglez= psi+Omegaz/Omegar*angler\
                -tan11-1./nu.sqrt(1.+4*self.amp*self.b/L2)*tan12
            sinu= z/R/nu.tan(i)
            pindx= (sinu > 1.)*(sinu < (1.+10.**-7.))
            sinu[pindx]= 1.
            pindx= (sinu < -1.)*(sinu > (-1.-10.**-7.))
            sinu[pindx]= -1.           
            u= nu.arcsin(sinu)
            u[vzindx]= nu.pi-u[vzindx]
            Omega= phi-u
            anglephi= Omega
            anglephi[indx]-= anglez[indx]
            anglephi[True^indx]+= anglez[True^indx]
            angler= angler % (2.*nu.pi)
            anglephi= anglephi % (2.*nu.pi)
            anglez= anglez % (2.*nu.pi)
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

    def _EccZmaxRperiRap(self,*args,**kwargs):
        """
        NAME:
           _EccZmaxRperiRap
        PURPOSE:
           evaluate the eccentricity, maximum height above the plane, peri- and apocenter for an isochrone potential
        INPUT:
           Either:
              a) R,vR,vT,z,vz[,phi]:
                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)
                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)
              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
        OUTPUT:
           (e,zmax,rperi,rap)
        HISTORY:
           2017-12-22 - Written - Bovy (UofT)
        """
        if len(args) == 5: #R,vR.vT, z, vz pragma: no cover
            R,vR,vT, z, vz= args
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
        else:
            self._parse_eval_args(*args)
            R= self._eval_R
            vR= self._eval_vR
            vT= self._eval_vT
            z= self._eval_z
            vz= self._eval_vz
        if isinstance(R,float):
            R= nu.array([R])
            vR= nu.array([vR])
            vT= nu.array([vT])
            z= nu.array([z])
            vz= nu.array([vz])
        if self._c: #pragma: no cover
            pass
        else:
            Lz= R*vT
            Lx= -z*vT
            Ly= z*vR-R*vz
            L2= Lx*Lx+Ly*Ly+Lz*Lz
            E= self._ip(R,z)+vR**2./2.+vT**2./2.+vz**2./2.
            if self.b == 0:
                warnings.warn("zmax for point-mass (b=0) isochrone potential is only approximate, because it assumes that zmax is attained at rap, which is not necessarily the case",galpyWarning)
                a= -self.amp/2./E
                me2= L2/self.amp/a
                e= nu.sqrt(1.-me2)
                rperi= a*(1.-e)
                rap= a*(1.+e)
            else:
                smin= 0.5*((2.*E-self.amp/self.b)\
                               +nu.sqrt((2.*E-self.amp/self.b)**2.
                                   +2.*E*(4.*self.amp/self.b+L2/self.b**2.)))/E
                smax= 2.-self.amp/E/self.b-smin
                rperi= smin*nu.sqrt(1.-2./smin)*self.b
                rap= smax*nu.sqrt(1.-2./smax)*self.b
            return ((rap-rperi)/(rap+rperi),rap*nu.sqrt(1.-Lz**2./L2),
                    rperi,rap)
