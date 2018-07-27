###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSphericalInverse
#
#             Calculate (x,v) coordinates for any spherical potential from 
#             given actions-angle coordinates
#
###############################################################################
import numpy
from scipy import optimize
from galpy.potential import IsochronePotential, vcirc, dvcircdR, \
    evaluatePotentials, evaluateRforces, toPlanarPotential
from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
from galpy.actionAngle import actionAngleSpherical
from .actionAngleAxi import actionAngleAxi
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import _actionAngleIsochroneHelper
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleSphericalInverse(actionAngleInverse):
    """Inverse action-angle formalism for spherical potentials"""
    def __init__(self,*args,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSphericalInverse object

        INPUT:

           pot= a Spherical potential

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OUTPUT:
        
           instance

        HISTORY:

           2017-11-21 - Started - Bovy (UofT)

        """
        actionAngleInverse.__init__(self,*args,**kwargs)
        if not 'pot' in kwargs: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSphericalInverse")
        self._pot= kwargs['pot']
        #Also store a 'planar' (2D) version of the potential
        if isinstance(self._pot,list):
            self._2dpot= [p.toPlanar() for p in self._pot]
        else:
            self._2dpot= self._pot.toPlanar()
        #The following for if we ever implement this code in C
        self._c= False
        ext_loaded= False
        if ext_loaded and (('c' in kwargs and kwargs['c'])
                           or not 'c' in kwargs):
            self._c= True #pragma: no cover
        else:
            self._c= False
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

           2017-11-21 - Written - Bovy (UofT)

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
        raise NotImplementedError("Method not implemented yet")

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

           2017-11-21 - Written - Bovy (UofT)

        """
        raise NotImplementedError("Method not implemented yet")

class actionAngleSphericalInverseSingle(actionAngleInverse):
    """Invert the action-angle formalism for a spherical potential for a single torus"""
    def __init__(self,E,L,
                 jr=None,jphi=None,jz=None,
                 Omegar=None,Omegaphi=None,Omegaz=None,
                 rperi=None,rap=None,
                 pot=None,ntr='auto',max_ntr=512,
                 use_newton=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSphericalInverseSingle object

        INPUT:

           E- energy

           L- total angular momentun

           pot= spherical potential instance or list of such instances

           Optionally, also supply the following:
           
               jr,jphi,jz= actions (one obviously redundant)

               Omegar, Omegaphi, Omegaz= frequencies (one obviously redundant)

               rperi, rap= peri- and apocenter radius

           ntr= 'auto'

           max_ntr= (512) Maximum number of radial angles to use to determine the torus mapping when using ntr='auto'

           use_newton= (False) if True, solve for radii for which angler = given angler using Newton-Raphson instead of using Brent's method (might be faster sometimes)

        OUTPUT:

           instance

        HISTORY:

           2017-11-21 - Written - Bovy (UofT)

        """
        if pot is None: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSphericalInverse")
        self._pot= pot
        self._E= E
        self._L= L
        L2= L**2.
        # Calculate jr if not given
        if jr is None or rperi is None or rap is None or Omegar is None \
                or Omegaphi is None or Omegaz is None:
            # Just setup an actionAngleAxi instance for all of this
            # Setup the orbit at R_L s.t. R_L vc(R_L) = L
            rl= optimize.newton(lambda x: x*vcirc(self._pot,x)-L,1.,
                                lambda x: dvcircdR(self._pot,x)+x)
            aAS= actionAngleAxi(rl,
                                numpy.sqrt(2.*(E
                                               -evaluatePotentials(self._pot,
                                                                   rl,0.))
                                           -L2/rl**2.),
                                L/rl,pot=toPlanarPotential(self._pot))
            self._jr= aAS.JR()
            # Assume that the orbit is in the plane unless otherwise specified
            if jphi is None and jz is None:
                self._jphi= L
                self._jz= 0.
            else:
                # Should probably check that jphi and jz are consistent with L
                self._jphi= jphi
                self._jz= jz
            self._rperi,self._rap= aAS.calcRapRperi()
            # Also compute the frequencies, use internal actionAngleSpherical
            aAS= actionAngleSpherical(pot=self._pot)
            Rmean= numpy.exp((numpy.log(self._rperi)+numpy.log(self._rap))/2.)
            self._Omegar= aAS._calc_or(Rmean,self._rperi,self._rap,
                                       self._E,self._L,False)
            self._Omegaz= aAS._calc_op(self._Omegar,Rmean,
                                       self._rperi,self._rap,
                                       self._E,self._L,False)
        else:
            # Store everything
            self._jr= jr
            self._jphi= jphi
            self._jz= jz
            self._Omegar= Omegar
            self._Omegaphi= Omegaphi
            self._Omegaz= Omegaz
            self._rperi= rperi
            self._rap= rap
        self._OmegazoverOmegar= self._Omegaz/self._Omegar
        # First need to determine an appropriate IsochronePotential
        ampb= L2*self._Omegaz*(self._Omegar-self._Omegaz)\
            /(2.*self._Omegaz-self._Omegar)**2.
        if ampb < 0.:
            raise NotImplementedError('actionAngleSphericalInverse not implemented for the case where Omegaz > Omegar or Omegaz < Omegar/2')
        amp= numpy.sqrt(self._Omegar
                        *(self._jr+numpy.sqrt(L2+4.*ampb)
                          *self._Omegaz/self._Omegar)**3.)
        self._ip= IsochronePotential(amp=amp,b=ampb/amp)
        self._isoaa= actionAngleIsochrone(ip=self._ip)
        self._isoaa_helper= _actionAngleIsochroneHelper(ip=self._ip)
        self._isoaainv= actionAngleIsochroneInverse(ip=self._ip)
        # Now need to determine the Sn and dSn mapping       
        if ntr == 'auto': ntr= 128 # BOVY: IMPLEMENT CORRECTLY
        thetara= numpy.linspace(0.,2.*numpy.pi*(1.-1./ntr),ntr)
        solvera= numpy.empty_like(thetara)
        jra= numpy.empty_like(thetara)
        ora= numpy.empty_like(thetara)
        dEdL= numpy.empty_like(thetara)
        for ii,tra in enumerate(thetara[:ntr//2+1]):
            if ii == 0:
                tr= self._rperi
            elif ii == ntr//2:
                tr= self._rap
            else:
                use_brent= not use_newton
                if use_newton:
                    if ii == ntr//2-1: tr= self._rap # Hack?
                    try:
                        tr= optimize.newton(lambda r: self._isoaa_helper.angler(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L2,reuse=True)-tra,
                                            tr,
                                            lambda r: self._isoaa_helper.danglerdr_constant_L(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L2,evaluateRforces(self._pot,r,0.)-evaluateRforces(self._ip,r,0.)))
                    except RuntimeError:
                        use_brent= True # fallback for non-convergence
                if use_brent:
                    tr= optimize.brentq(lambda r: self._isoaa_helper.angler(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L2,reuse=False)-tra,tr,self._rap)
            tE= E+self._ip(tr,0.)-evaluatePotentials(self._pot,tr,0.)
            tjra= self._isoaa_helper.Jr(tE,L)
            tora= self._isoaa_helper.Or(tE)
            solvera[ii]= tr
            jra[ii]= tjra
            # Compute dEA/dE and dEA/dL
            dEAdr= evaluateRforces(self._pot,tr,0.)\
                -evaluateRforces(self._ip,tr,0.)
            drdE,drdL= self._isoaa_helper.drdEL_constant_angler(\
                tr,2.*(E-evaluatePotentials(self._pot,tr,0.))-L2/tr**2.,
                tE,L,dEAdr)
            dEdE= drdE*dEAdr+1.
            dEdL[ii]= drdL*dEAdr/tora
            ora[ii]= tora/dEdE
        # don't need solvera, just for sanity checking
        solvera[ntr//2+1:]= solvera[::-1][ntr//2:-1]
        jra[ntr//2+1:]= jra[::-1][ntr//2:-1]
        ora[ntr//2+1:]= ora[::-1][ntr//2:-1]
        dEdL[ntr//2+1:]= dEdL[::-1][ntr//2:-1]
        self._thetara= thetara
        self._solvera= solvera
        self._jra= jra
        self._ora= ora
        self._dEdL= dEdL
        # Compute Sn and dSn/dJr, remove n=0
        self._nforSn= numpy.arange(len(self._ora)//2+1)
        self._nSn= numpy.real(numpy.fft.rfft(self._jra-self._jr))[1:]/len(self._jra)
        self._dSndJr= (numpy.real(numpy.fft.rfft(self._Omegar/self._ora-1.))/self._nforSn)[1:]/len(self._ora)
        self._dSndLish= (numpy.real(numpy.fft.rfft(self._dEdL))/self._nforSn)[1:]/len(self._ora)
        self._nforSn= self._nforSn[1:]
        return None

    def __call__(self,angler,anglephi,anglez,jphi=None):
        """
        NAME:
           __call__
        PURPOSE:
           convert angles --> (x,v) for this torus
        INPUT:
           angler, anglephi, anglez - angles on the torus
           jphi= (object-wide default) z-component of the angular momentum
        OUTPUT:
           (R,vR,vT,z,vz,phi)
        HISTORY:
           2017-11-21 - Written - Bovy (UofT)
        """
        # First we need to solve for anglera
        anglera= optimize.newton(\
            lambda ar: ar+2.*numpy.sum(self._dSndJr*numpy.sin(self._nforSn*ar))-angler,
            0.,
            lambda ar: 1.+2.*numpy.sum(self._nforSn*self._dSndJr
                                       *numpy.cos(self._nforSn*ar)))
        # Then compute the auxiliary action
        jra= self._jr+2.*numpy.sum(self._nSn*numpy.cos(self._nforSn*anglera))
        angleza= anglez+self._OmegazoverOmegar*(anglera-angler)\
            -2.*numpy.sum(self._dSndLish*numpy.sin(self._nforSn*anglera))
        if jphi is None: 
            jphi= self._jphi
            jz= self._jz
        else:
            jz= self._L-numpy.fabs(jphi)
        anglephia= anglephi+numpy.sign(jphi)*(angleza-anglez)
        return self._isoaainv(jra,jphi,jz,anglera,anglephia,angleza)
