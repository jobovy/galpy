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
import numpy as nu
from galpy.actionAngle_src.actionAngle import actionAngle
from galpy.potential import IsochronePotential
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
           _evaluate
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
           _actionsFreqs
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
           _actionsFreqsAngles
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
            pindx= (coseta > 1.)
            coseta[pindx]= 1.
            pindx= (coseta < -1.)
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
            pindx= (Lz/L > 1.)
            Lz[pindx]= L[pindx]
            pindx= (Lz/L < -1.)
            Lz[pindx]= -L[pindx]
            sini= nu.sqrt(L**2.-Lz**2.)/L
            tani= nu.sqrt(L**2.-Lz**2.)/Lz
            sinpsi= costheta/sini
            pindx= (sinpsi > 1.)*nu.isfinite(sinpsi)
            sinpsi[pindx]= 1.
            pindx= (sinpsi < -1.)*nu.isfinite(sinpsi)
            sinpsi[pindx]= -1.           
            psi= nu.arcsin(sinpsi)
            psi[vzindx]= nu.pi-psi[vzindx]
            # For non-inclined orbits, we set Omega=0 by convention
            psi[True^nu.isfinite(psi)]= phi[True^nu.isfinite(psi)]
            psi= psi % (2.*nu.pi)
            anglez= psi+Omegaz/Omegar*angler\
                -tan11-1./nu.sqrt(1.+4*self.amp*self.b/L2)*tan12
            sinu= z/R/tani
            pindx= (sinu > 1.)*nu.isfinite(sinu)
            sinu[pindx]= 1.
            pindx= (sinu < -1.)*nu.isfinite(sinu)
            sinu[pindx]= -1.           
            u= nu.arcsin(sinu)
            u[vzindx]= nu.pi-u[vzindx]
            # For non-inclined orbits, we set Omega=0 by convention
            u[True^nu.isfinite(u)]= phi[True^nu.isfinite(u)]
            Omega= phi-u
            anglephi= Omega
            anglephi[indx]-= anglez[indx]
            anglephi[True^indx]+= anglez[True^indx]
            angler= angler % (2.*nu.pi)
            anglephi= anglephi % (2.*nu.pi)
            anglez= anglez % (2.*nu.pi)
            return (Jr,Jphi,Jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

class _actionAngleIsochroneHelper(object):
    """Simplified version of the actionAngleIsochrone transformations, for use in actionAngleSphericalInverse"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an _actionAngleIsochroneHelper object

        INPUT:

           ip= instance of a IsochronePotential

        OUTPUT:
        
           instance

        HISTORY:

           2017-11-30 - Written - Bovy (UofT)

        """
        if not 'ip' in kwargs: #pragma: no cover
            raise IOError("Must specify ip= for _actionAngleIsochroneHelper")
        else:
            ip= kwargs['ip']
            if not isinstance(ip,IsochronePotential): #pragma: no cover
                raise IOError("'Provided ip= does not appear to be an instance of an IsochronePotential")
            # Check the units
            self.b= ip.b
            self.amp= ip._amp
        self._ip= ip
        return None
    
    def angler(self,r,vr2,L,reuse=False,vrneg=False):
        """
        NAME:
           angler
        PURPOSE:
           calculate the radial angle
        INPUT:
           r - radius
           vr2 - radial velocity squared
           L - angular momentum
           vrneg= (False) True if vr is negative
           reuse= (False) if True, re-use all relevant quantities for computing the radial angle that were computed prviously as part of danglerdr_constant_L)
        OUTPUT:
           radial angle
        HISTORY:
           2017-11-30 - Written - Bovy (UofT)
        """
        if reuse: 
            return (self._eta-self._e*self._c/(self._c+self.b)*self._sineta) % (2.*nu.pi)
        E= self._ip(r,0.)+vr2/2.+L**2./2./r**2.
        if E > 0.: return -1.
        c= -self.amp/2./E-self.b
        e2= 1.-L*L/self.amp/c*(1.+self.b/c)
        e= nu.sqrt(e2)
        if self.b == 0.:
            coseta= 1/e*(1.-r/c)
        else:
            s= 1.+nu.sqrt(1.+r*r/self.b**2.)
            coseta= 1/e*(1.-self.b/c*(s-2.))
        if coseta > 1. and coseta < (1.+10.**-7.): coseta= 1.
        elif coseta < -1. and coseta > (-1.-10.**-7.): coseta= -1.
        eta= nu.arccos(coseta)
        if vrneg: eta= 2.*nu.pi-eta
        angler= (eta-e*c/(c+self.b)*nu.sin(eta)) % (2.*nu.pi)
        return angler

    def danglerdr_constant_L(self,r,vr2,L,dEdr,vrneg=False):
        """Function used in actionAngleSphericalInverse when finding r at which angler has a particular value on the isochrone torus"""
        E= self._ip(r,0.)+vr2/2.+L**2./2./r**2.
        L2= L**2.
        self._c= -self.amp/2./E-self.b
        L2overampc= L2/self.amp/self._c
        e2= 1.-L2overampc*(1.+self.b/self._c)
        self._e= nu.sqrt(e2)
        if self.b == 0.:
            coseta= 1/self._e*(1.-r/self._c)
        else:
            s= 1.+nu.sqrt(1.+r**2./self.b**2.)
            coseta= 1/self._e*(1.-self.b/self._c*(s-2.))
        if coseta > 1. and coseta < (1.+10.**-7.): coseta= 1.
        elif coseta < -1. and coseta > (-1.-10.**-7.): coseta= -1.
        self._eta= nu.arccos(coseta)
        if vrneg: self._eta= 2.*nu.pi-self._eta
        self._sineta= nu.sin(self._eta)
        L2overampc*= (1.+2.*self.b/self._c)/(2.*self._e) # from now on need L2/(2GM c e)
        dcdr= self.amp/2./E**2.*dEdr
        dsdrtimesb= r/nu.sqrt(r**2.+self.b**2.)
        detadr= (dsdrtimesb+(coseta*(self._e+L2overampc)-1.)*dcdr)/(self._e*self._c*self._sineta)
        return detadr*(1.-self._e*self._c*coseta/(self._c+self.b))\
            -self._sineta/(self._c+self.b)*(self._e*self.b/(self._c+self.b)+L2overampc)*dcdr

    def Jr(self,E,L):
        return self.amp/nu.sqrt(-2.*E)\
            -0.5*(L+nu.sqrt((L*L+4.*self.amp*self.b)))
        
    def Or(self,E):
        return (-2.*E)**1.5/self.amp
        
    def drdEL_constant_angler(self,r,vr2,E,L,dEdr,vrneg=False):
        """Function used in actionAngleSphericalInverse to determine dEA/dE and dEA/dL: derivative of the radius r wrt E and L necessary to have constant angler"""
        L2= L**2.
        c= -self.amp/2./E-self.b
        e2= 1.-L2/self.amp/c*(1.+self.b/c)
        e= nu.sqrt(e2)
        if self.b == 0.:
            coseta= 1/e*(1.-r/c)
        else:
            s= 1.+nu.sqrt(1.+r**2./self.b**2.)
            coseta= 1/e*(1.-self.b/c*(s-2.))
        if coseta > 1. and coseta < (1.+10.**-7.): coseta= 1.
        elif coseta < -1. and coseta > (-1.-10.**-7.): coseta= -1.
        eta= nu.arccos(coseta)
        if vrneg: eta= 2.*nu.pi-eta
        sineta= nu.sin(eta)
        bcmecce= (self.b+c-e*c*coseta)
        c2e2ob= c**2.*sineta**2./self.b
        dcdLfac= (1.-e*coseta)/self.b+e2*c2e2ob/bcmecce*(1./c-1./(self.b+c))
        dcdLoverdrdL= self.amp/2./E**2.*dEdr
        dedLfac= -c*coseta/e/self.b+c2e2ob/bcmecce
        numfordrdE= dcdLfac*self.amp/2./E**2.+dedLfac*L2/4./c**2./E**2*(1.+2.*self.b/c)
        return (numfordrdE/(r/self.b**2./(s-1.)-numfordrdE*dEdr),
                -dedLfac*L/self.amp/c*(1.+self.b/c)\
                    /(r/self.b**2./(s-1.)-dcdLfac*dcdLoverdrdL
                      -dedLfac*L2/2./self.amp/c**2.*(1.+2.*self.b/c)*dcdLoverdrdL))
        
