###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleIsochroneInverse
#
#             Calculate (x,v) coordinates for the Isochrone potential from 
#             given actions-angle coordinates
#
###############################################################################
import numpy
from scipy import optimize
from ..util import conversion
from ..potential import IsochronePotential
from .actionAngleInverse import actionAngleInverse
class actionAngleIsochroneInverse(actionAngleInverse):
    """Inverse action-angle formalism for the isochrone potential, on the Jphi, Jtheta system of Binney & Tremaine (2008); following McGill & Binney (1990) for transformations"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleIsochroneInverse object

        INPUT:

           Either:

              b= scale parameter of the isochrone parameter (can be Quantity)

              ip= instance of a IsochronePotential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2017-11-14 - Started - Bovy (UofT)

        """
        actionAngleInverse.__init__(self,*args,**kwargs)
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
            self.b= conversion.parse_length(kwargs['b'],ro=self._ro)
            rb= numpy.sqrt(self.b**2.+1.)
            self.amp= (self.b+rb)**2.*rb
        # In case we ever decide to implement this in C...
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
    
    def _evaluate(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])

           tol= (object-wide value) goal for |dJ|/|J| along the torus

        OUTPUT:

           [R,vR,vT,z,vz,phi]

        HISTORY:

           2017-11-14 - Written - Bovy (UofT)

        """
        return self._xvFreqs(jr,jphi,jz,angler,anglephi,anglez,**kwargs)[:6]
        
    def _xvFreqs(self,jr,jphi,jz,angler,anglephi,anglez,**kwargs):
        """
        NAME:

           xvFreqs

        PURPOSE:

           evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

           angler - radial angle (array [N])

           anglephi - azimuthal angle (array [N])

           anglez - vertical angle (array [N])


        OUTPUT:

           ([R,vR,vT,z,vz,phi],OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-11-15 - Written - Bovy (UofT)

        """
        L= jz+numpy.fabs(jphi) # total angular momentum
        L2= L**2.
        sqrtfourbkL2= numpy.sqrt(L2+4.*self.b*self.amp)
        H= -2.*self.amp**2./(2.*jr+L+sqrtfourbkL2)**2.
        # Calculate the frequencies
        omegar= (-2.*H)**1.5/self.amp
        omegaz= (1.+L/sqrtfourbkL2)/2.*omegar
        # Start on getting the coordinates
        a= -self.amp/2./H-self.b
        ab= a+self.b
        e= numpy.sqrt(1.+L2/(2.*H*a**2.))
        # Solve Kepler's-ish equation; ar must be between 0 and 2pi
        angler= (numpy.atleast_1d(angler) % (-2.*numpy.pi)) % (2.*numpy.pi)
        anglephi= numpy.atleast_1d(anglephi)
        anglez= numpy.atleast_1d(anglez)
        eta= numpy.empty(len(angler))
        for ii,ar in enumerate(angler):
            try:
                eta[ii]= optimize.newton(lambda x: x-a*e/ab*numpy.sin(x)-ar,
                                         0.,
                                         lambda x: 1-a*e/ab*numpy.cos(x))
            except RuntimeError:
                # Newton-Raphson did not converge, this has to work, 
                # bc 0 <= ra < 2pi the following start x have different signs
                eta[ii]= optimize.brentq(lambda x: x-a*e/ab*numpy.sin(x)-ar,
                                         0.,2.*numpy.pi)
        coseta= numpy.cos(eta)
        r= a*numpy.sqrt((1.-e*coseta)*(1.-e*coseta+2.*self.b/a))
        vr= numpy.sqrt(self.amp/ab)*a*e*numpy.sin(eta)/r
        taneta2= numpy.tan(eta/2.)
        tan11= numpy.arctan(numpy.sqrt((1.+e)/(1.-e))*taneta2)
        tan12= numpy.arctan(\
            numpy.sqrt((a*(1.+e)+2.*self.b)/(a*(1.-e)+2.*self.b))*taneta2)
        tan11[tan11 < 0.]+= numpy.pi
        tan12[tan12 < 0.]+= numpy.pi
        Lambdaeta= tan11+L/sqrtfourbkL2*tan12
        psi= anglez-omegaz/omegar*angler+Lambdaeta
        lowerl= numpy.sqrt(1.-jphi**2./L2)
        sintheta= numpy.sin(psi)*lowerl
        costheta= numpy.sqrt(1.-sintheta**2.)
        vtheta= L*lowerl*numpy.cos(psi)/costheta/r
        R= r*costheta
        z= r*sintheta
        vR= vr*costheta-vtheta*sintheta
        vz= vr*sintheta+vtheta*costheta
        sinu= sintheta/costheta*jphi/L/lowerl
        u= numpy.arcsin(sinu)
        u[vtheta < 0.]= numpy.pi-u[vtheta < 0.]
        phi= anglephi-numpy.sign(jphi)*anglez+u
        # For non-inclined orbits, phi == psi
        phi[True^numpy.isfinite(phi)]= psi[True^numpy.isfinite(phi)]
        phi= phi % (2.*numpy.pi)
        phi[phi < 0.]+= 2.*numpy.pi
        return (R,vR,jphi/R,z,vz,phi,
                omegar,numpy.sign(jphi)*omegaz,omegaz)
        
    def _Freqs(self,jr,jphi,jz,**kwargs):
        """
        NAME:

           Freqs

        PURPOSE:

           return the frequencies corresponding to a torus

        INPUT:

           jr - radial action (scalar)

           jphi - azimuthal action (scalar)

           jz - vertical action (scalar)

        OUTPUT:

           (OmegaR,Omegaphi,Omegaz)

        HISTORY:

           2017-11-15 - Written - Bovy (UofT)

        """
        L= jz+numpy.fabs(jphi) # total angular momentum
        sqrtfourbkL2= numpy.sqrt(L**2.+4.*self.b*self.amp)
        H= -2.*self.amp**2./(2.*jr+L+sqrtfourbkL2)**2.
        # Calculate the frequencies
        omegar= (-2.*H)**1.5/self.amp
        omegaz= (1.+L/sqrtfourbkL2)/2.*omegar
        return (omegar,numpy.sign(jphi)*omegaz,omegaz)

