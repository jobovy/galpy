###############################################################################
#   actionAngle: a Python module to calculate  actions, angles, and frequencies
#
#      class: actionAngleSphericalInverse
#
#             Calculate (x,v) coordinates for any spherical potential from 
#             given actions-angle coordinates
#
###############################################################################
import copy
import numpy
import warnings
from scipy import optimize
from galpy.util import galpyWarning
from galpy.potential import IsochronePotential, vcirc, dvcircdR, \
    evaluatePotentials, evaluateRforces
from .actionAngleIsochrone import actionAngleIsochrone
from .actionAngleIsochroneInverse import actionAngleIsochroneInverse
from galpy.actionAngle import actionAngleSpherical
from .actionAngleInverse import actionAngleInverse
from .actionAngleIsochrone import _actionAngleIsochroneHelper
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
class actionAngleSphericalInverse(actionAngleInverse):
    """Inverse action-angle formalism for spherical potentials"""
    def __init__(self,pot=None,Es=[0.1,0.3],Ls=[1.,1.2],grid=False,
                 nta=128,
                 setup_interp=False,maxiter=100,angle_tol=1e-12,bisect=False):
        """
        NAME:

           __init__

        PURPOSE:

           initialize an actionAngleSphericalInverse object

        INPUT:

           pot= a Potential or list thereof, should be a spherical potential

           Es= energies of the orbits to map the tori for, will be forcibly sorted when grid=True (needs to be a dense grid when setting up the object for interpolation with setup_interp=True)

           Ls= angular momenta of the orbits to map the tori for, will be forcibly sorted when grid=True (needs to be a dense grid when setting up the object for interpolation with setup_interp=True)

           grid= (False) if True, make a 2D grid out of provide 1D Es and Ls, to fully map (E,L)

           nta= (128) number of auxiliary angles to sample the torus at when mapping the torus

           setup_interp= (False) if True, setup interpolation grids that allow any torus within the E range to be accessed through interpolation

           maxiter= (100) maximum number of iterations of root-finding algorithms

           angle_tol= (1e-12) tolerance for angle root-finding (f(x) is within tol of desired value)

           bisect= (False) if True, use simple bisection for root-finding, otherwise first try Newton-Raphson (mainly useful for testing the bisection fallback)
           
        OUTPUT:
        
           instance

        HISTORY:

           2017-11-21 - Started initial implementation that works for single (E,L) - Bovy (UofT)

           2017-11-02 - Started efficient implementation for multiple (E,L), like actionAngleVerticalInverse - Bovy (UofT)

        """
        #actionAngleInverse.__init__(self,*args,**kwargs)
        if pot is None: #pragma: no cover
            raise IOError("Must specify pot= for actionAngleSphericalInverse")
        self._pot= pot
        self._aAS= actionAngleSpherical(pot=self._pot)
        # Determine gridding options
        self._Es= numpy.atleast_1d(Es)
        self._Ls= numpy.atleast_1d(Ls)
        self._nE= len(self._Es)
        self._nL= len(self._Ls)
        if not grid and self._nE != self._nL:
            raise ValueError("When grid=False, len(Es) has to equal len(Ls)")
        if grid:
            # Make grid, flatten so we can treat it as regular 1D input
            self._Es= numpy.sort(self._Es)
            self._Ls= numpy.sort(self._Ls)
            self._internal_Es,self._internal_Ls= \
                numpy.meshgrid(self._Es,self._Ls,indexing='ij')
            self._internal_Es= self._internal_Es.flatten()
            self._internal_Ls= self._internal_Ls.flatten()
        else:
            self._internal_Es= copy.copy(self._Es)
            self._internal_Ls= copy.copy(self._Ls)
        self._L2= self._internal_Ls**2
        # Compute actions, frequencies, and rperi/rap for each (E,L), to do
        # this, setup orbit at radius of circular orbit for given L
        rls= numpy.array([optimize.newton(lambda x: x*vcirc(self._pot,x)-L,1.,
                                          lambda x: dvcircdR(self._pot,x)+x)
                          for L in self._internal_Ls])
        self._jr,_,_,self._Omegar,_,self._Omegaz= self._aAS.actionsFreqs(\
            rls,numpy.sqrt(2.*(self._internal_Es\
                                  -evaluatePotentials(self._pot,rls,
                                                      numpy.zeros_like(rls)))
                          -self._L2/rls**2.),
            self._internal_Ls/rls,numpy.zeros_like(rls),numpy.zeros_like(rls))
        # Also need rperi and rap
        _,_,self._rperi,self._rap= self._aAS.EccZmaxRperiRap(\
            rls,numpy.sqrt(2.*(self._internal_Es\
                                  -evaluatePotentials(self._pot,rls,
                                                      numpy.zeros_like(rls)))
                          -self._L2/rls**2.),
            self._internal_Ls/rls,numpy.zeros_like(rls),numpy.zeros_like(rls))
        self._OmegazoverOmegar= self._Omegaz/self._Omegar
        # First need to determine appropriate IsochronePotentials
        ampb= self._L2*self._Omegaz*(self._Omegar-self._Omegaz)\
            /(2.*self._Omegaz-self._Omegar)**2.
        if numpy.any(ampb < 0.):
            raise NotImplementedError('actionAngleSphericalInverse not implemented for the case where Omegaz > Omegar or Omegaz < Omegar/2')
        amp= numpy.sqrt(self._Omegar
                        *(self._jr+numpy.sqrt(self._L2+4.*ampb)
                          *self._Omegaz/self._Omegar)**3.)
        self._amp= amp
        self._b= ampb/amp
        # This sets up objects with arrays of parameters, which is not
        # generally supported in galpy, but works here as long as the object
        # is evaluated with the same number of phase-space points as the 
        # length of the parameter array
        self._ip= IsochronePotential(amp=self._amp,b=self._b)
        self._isoaa= actionAngleIsochrone(ip=self._ip)
        self._isoaainv= actionAngleIsochroneInverse(ip=self._ip)
        # Now map all tori
        self._nta= nta
        self._thetaa= numpy.linspace(0.,2.*numpy.pi*(1.-1./nta),nta)
        self._maxiter= maxiter
        self._angle_tol= angle_tol
        self._bisect= bisect
        # Determine the r grid for even-spaced theta_r grid
        self._rgrid= self._create_rgrid()
        # Compute mapping coefficients
        isoaa_helper= _actionAngleIsochroneHelper(\
            ip=IsochronePotential(amp=self._ampgrid,b=self._bgrid))
        tE= self._Egrid+isoaa_helper._ip(self._rgrid,
                                         numpy.zeros_like(self._rgrid))\
            -evaluatePotentials(self._pot,self._rgrid,
                                numpy.zeros_like(self._rgrid))
        self._jra= isoaa_helper.Jr(tE,self._Lgrid)
        self._ora= isoaa_helper.Or(tE)
        # Compute dEA/dE and dEA/dL for dJr^A/d(Jr,L)
        dEAdr= evaluateRforces(self._pot,
                               self._rgrid,numpy.zeros_like(self._rgrid))\
            -evaluateRforces(isoaa_helper._ip,
                             self._rgrid,numpy.zeros_like(self._rgrid))
        drdE,drdL= isoaa_helper.drdEL_constant_angler(\
            self._rgrid,
            2.*(self._Egrid
                -evaluatePotentials(self._pot,
                                    self._rgrid,
                                    numpy.zeros_like(self._rgrid)))\
                -self._Lgrid**2/self._rgrid**2.,
            tE,self._Lgrid,dEAdr)
        self._dEdE= drdE*dEAdr+1.
        self._dEdL= drdL*dEAdr/self._ora
        self._djradjr= numpy.tile(self._Omegar,(self._nta,1)).T\
            /self._ora*self._dEdE
        self._djradLish= self._dEdL
        # Store mean(jra) as probably a better approx. of jr
        self._jr_orig= copy.copy(self._jr)
        self._jr= numpy.mean(self._jra,axis=1)
        # Compute Fourier expansions
        self._nforSn= numpy.arange(self._jra.shape[1]//2+1)
        self._nSn= numpy.real(numpy.fft.rfft(self._jra
                                             -numpy.atleast_2d(self._jr).T,
                                             axis=1))[:,1:]/self._jra.shape[1]
        self._dSndJr= numpy.real(numpy.fft.rfft(self._djradjr-1.,axis=1))[:,1:]\
                          /self._jra.shape[1]
        self._dSndLish= numpy.real(numpy.fft.rfft(self._djradLish,axis=1))[:,1:]\
                          /self._jra.shape[1]
        self._dSndJr/= numpy.atleast_2d(self._nforSn)[:,1:]
        self._dSndLish/= numpy.atleast_2d(self._nforSn)[:,1:]
        self._nforSn= self._nforSn[1:]
        # Check the units
        #self._check_consistent_units()
        return None
           
    def _create_rgrid(self):
        # Find r grid for regular grid in auxiliary angle (thetara)
        # in practice only need to map 0 < thetara < pi  to r with +v bc symm
        # To efficiently start the search, first compute thetara for a dense
        # grid in r (at +v); also don't allow points to be exactly at 
        # rperi or rap, because Newton derivative is inf there...
        rgrid= numpy.linspace(0.,1.,2*self._nta)
        rs= rgrid*numpy.atleast_2d(self._rap-self._rperi-2*1e-8).T\
            +numpy.atleast_2d(self._rperi+1e-8).T
        # Setup helper for computing angles, and derivative
        isoaa_helper= _actionAngleIsochroneHelper(\
            ip=IsochronePotential(amp=numpy.tile(self._amp,(rs.shape[1],1)).T,
                                  b=numpy.tile(self._b,(rs.shape[1],1)).T))
        rta= isoaa_helper.angler(\
            rs,2.*(numpy.tile(self._internal_Es,(rs.shape[1],1)).T
                   -evaluatePotentials(self._pot,rs,numpy.zeros_like(rs)))
            -numpy.tile(self._L2,(rs.shape[1],1)).T/rs**2.,
            numpy.tile(self._internal_Ls,(rs.shape[1],1)).T,reuse=False)
        rta[numpy.isnan(rta)]= 0. # Zero energy orbit -> NaN
        # Now use Newton-Raphson to iterate to a regular grid
        cindx= numpy.nanargmin(numpy.fabs(\
                (rta-numpy.rollaxis(numpy.atleast_3d(self._thetaa),1)
                 +numpy.pi) % (2.*numpy.pi)-numpy.pi),axis=2)
        rgrid= rgrid[cindx].T*numpy.atleast_2d(self._rap-self._rperi-2*1e-8).T\
            +numpy.atleast_2d(self._rperi+1e-8).T
        Egrid= numpy.tile(self._internal_Es,(self._nta,1)).T
        Lgrid= numpy.tile(self._internal_Ls,(self._nta,1)).T
        L2grid= Lgrid**2
        # Force rperi and rap to be thetar=0 and pi and don't optimize later
        rgrid[:,0]= self._rperi
        rgrid[:,self._nta//2]= self._rap
        # Need to adjust parameters of helpers
        ampgrid= numpy.tile(self._amp,(self._nta,1)).T
        bgrid= numpy.tile(self._b,(self._nta,1)).T
        isoaa_helper._ip= IsochronePotential(amp=ampgrid,b=bgrid)
        isoaa_helper.amp= ampgrid
        isoaa_helper.b= bgrid
        rperigrid= numpy.tile(self._rperi,(self._nta,1)).T
        rapgrid= numpy.tile(self._rap,(self._nta,1)).T
        ta= isoaa_helper.angler(\
            rgrid,2.*(Egrid
                   -evaluatePotentials(self._pot,rgrid,
                                       numpy.zeros_like(rgrid)))
            -L2grid/rgrid**2.,Lgrid,reuse=False)
        mta= numpy.tile(self._thetaa,(len(self._internal_Es),1))
        # Now iterate
        cntr= 0
        unconv= numpy.ones(rgrid.shape,dtype='bool')
        # We'll fill in the -v part using the +v, also remove rperi/rap
        unconv[:,0]= False
        unconv[:,self._nta//2:]= False
        dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
        unconv[unconv]= numpy.fabs(dta) > self._angle_tol
        # Don't allow too big steps
        maxdr= numpy.tile((self._rap-self._rperi)/float(self._nta),
                          (self._nta,1)).T
        isoaa_helper._ip= IsochronePotential(amp=ampgrid[unconv],
                                             b=bgrid[unconv])
        isoaa_helper.amp= ampgrid[unconv]
        isoaa_helper.b= bgrid[unconv]
        while not self._bisect:
            dtadr= isoaa_helper.danglerdr_constant_L(\
                rgrid[unconv],
                2.*(Egrid[unconv]-evaluatePotentials(self._pot,rgrid[unconv],
                                                     numpy.zeros_like(rgrid[unconv])))
                -L2grid[unconv]/rgrid[unconv]**2.,Lgrid[unconv],
                evaluateRforces(self._pot,rgrid[unconv],0.)
                -evaluateRforces(isoaa_helper._ip,rgrid[unconv],
                                 numpy.zeros_like(rgrid[unconv])))
            dta= (ta[unconv]-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
            dr= -dta/dtadr
            dr[numpy.fabs(dr) > maxdr[unconv]]=\
                (numpy.sign(dr)*maxdr[unconv])[numpy.fabs(dr) > maxdr[unconv]]
            rgrid[unconv]+= dr
            rgrid[unconv*(rgrid > rapgrid)]=\
                rapgrid[unconv*(rgrid > rapgrid)]
            rgrid[unconv*(rgrid < rperigrid)]=\
                rperigrid[unconv*(rgrid < rperigrid)]
            unconv[unconv]= numpy.fabs(dta) > self._angle_tol
            isoaa_helper._ip= IsochronePotential(amp=ampgrid[unconv],
                                                 b=bgrid[unconv])
            isoaa_helper.amp= ampgrid[unconv]
            isoaa_helper.b= bgrid[unconv]
            newta= isoaa_helper.angler(\
                rgrid[unconv],2.*(Egrid[unconv]
                          -evaluatePotentials(self._pot,rgrid[unconv],
                                              numpy.zeros_like(rgrid[unconv])))
                -L2grid[unconv]/rgrid[unconv]**2.,Lgrid[unconv],reuse=False)
            ta[unconv]= newta
            cntr+= 1
            if numpy.sum(unconv) == 0:
                break
            if cntr > self._maxiter:
                warnings.warn(\
                    "Torus mapping with Newton-Raphson did not converge in {} iterations, falling back onto simple bisection (increase maxiter to try harder with Newton-Raphson)"\
                        .format(self._maxiter),galpyWarning)
                break
        if False:#self._bisect or cntr > self._maxiter:
            # Reset cntr
            cntr= 0
            # Start from nearest guess from below
            new_rgrid= numpy.linspace(0.,1.,2*self._nta)
            da=(rta-numpy.rollaxis(numpy.atleast_3d(self._thetaa),1)+numpy.pi)\
                % (2.*numpy.pi) - numpy.pi
            da[da >= 0.]= -numpy.nanmax(numpy.fabs(da))-0.1
            cindx= numpy.nanargmax(da,axis=2)
            tryr_min= (new_rgrid[cindx].T
                       *numpy.atleast_2d(self._rap-self._rperi-2*1e-8).T
                       +numpy.atleast_2d(self._rperi+1e-8).T)[unconv]
            dr= 2./(2.*self._nta-1)*(rapgrid-rperigrid) # delta of initial x grid above
            while True:
                dr*= 0.5
                rgrid[unconv]= tryr_min+dr[unconv]
                isoaa_helper._ip= IsochronePotential(amp=ampgrid[unconv],
                                                     b=bgrid[unconv])
                isoaa_helper.amp= ampgrid[unconv]
                isoaa_helper.b= bgrid[unconv]
                newta= (isoaa_helper.angler(\
                    rgrid[unconv],2.*(Egrid[unconv]
                          -evaluatePotentials(self._pot,rgrid[unconv],
                                              numpy.zeros_like(rgrid[unconv])))
                    -L2grid[unconv]/rgrid[unconv]**2.,Lgrid[unconv],
                    reuse=False)+2.*numpy.pi) \
                                % (2.*numpy.pi)
                ta[unconv]= newta
                dta= (newta-mta[unconv]+numpy.pi) % (2.*numpy.pi)-numpy.pi
                tryr_min[newta < mta[unconv]]=\
                    rgrid[unconv][newta < mta[unconv]]
                unconv[unconv]= numpy.fabs(dta) > self._angle_tol
                tryr_min= tryr_min[numpy.fabs(dta) > self._angle_tol]
                cntr+= 1
                if numpy.sum(unconv) == 0:
                    break
                if cntr > self._maxiter:
                    warnings.warn(\
                        "Torus mapping with bisection did not converge in {} iterations"\
                            .format(self._maxiter)
                        +" for energies:"+""\
                  .join(' {:g}'.format(k) for k in sorted(set(Egrid[unconv]))),
                    galpyWarning)
                    break
        rgrid[:,self._nta//2+1:]= rgrid[:,1:self._nta//2][:,::-1]
        isoaa_helper._ip= IsochronePotential(amp=ampgrid[:,self._nta//2+1:],
                                             b=bgrid[:,self._nta//2+1:])
        isoaa_helper.amp= ampgrid[:,self._nta//2+1:]
        isoaa_helper.b= bgrid[:,self._nta//2+1:]
        ta[:,self._nta//2+1:]= isoaa_helper.angler(\
            rgrid[:,self._nta//2+1:],2.*(Egrid[:,self._nta//2+1:]
                          -evaluatePotentials(self._pot,rgrid[:,self._nta//2+1:],
                                              numpy.zeros_like(rgrid[:,self._nta//2+1:])))
                    -L2grid[:,self._nta//2+1:]/rgrid[:,self._nta//2+1:]**2.,Lgrid[:,self._nta//2+1:],
                    reuse=False,vrneg=True)
        self._dta= (ta-mta+numpy.pi) % (2.*numpy.pi)-numpy.pi
        self._mta= mta
        # Store these, they are useful (obv. arbitrary to return rgrid 
        # and not just store it...)
        self._Egrid= Egrid
        self._Lgrid= Lgrid
        self._ampgrid= ampgrid
        self._bgrid= bgrid
        self._rperigrid= rperigrid
        self._rapgrid= rapgrid
        return rgrid

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
            # Just setup an actionAngleSpherical instance for all of this
            # Setup the orbit at R_L s.t. R_L vc(R_L) = L
            rl= optimize.newton(lambda x: x*vcirc(self._pot,x)-L,1.,
                                lambda x: dvcircdR(self._pot,x)+x)
            aAS= actionAngleSpherical(pot=self._pot)
            self._jr,_,_,self._Omegar,_,self._Omegaz= \
                aAS.actionsFreqs(rl,
                                numpy.sqrt(2.*(E
                                               -evaluatePotentials(self._pot,
                                                                   rl,0.))
                                           -L2/rl**2.),
                                L/rl,0.,0.)
            # Assume that the orbit is in the plane unless otherwise specified
            if jphi is None and jz is None:
                self._jphi= L
                self._jz= 0.
            else:
                # Should probably check that jphi and jz are consistent with L
                self._jphi= jphi
                self._jz= jz
            # Also need rperi and rap
            _,_,self._rperi,self._rap= aAS.EccZmaxRperiRap(rl,
                                numpy.sqrt(2.*(E
                                               -evaluatePotentials(self._pot,
                                                                   rl,0.))
                                           -L2/rl**2.),
                                                           L/rl,0.,0.)
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
        solvedta= numpy.empty_like(thetara)
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
                        tr= optimize.newton(lambda r: self._isoaa_helper.angler(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L,reuse=True)-tra,
                                            tr,
                                            lambda r: self._isoaa_helper.danglerdr_constant_L(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L,evaluateRforces(self._pot,r,0.)-evaluateRforces(self._ip,r,0.)))
                    except RuntimeError:
                        use_brent= True # fallback for non-convergence
                if use_brent:
                    tr= optimize.brentq(lambda r: self._isoaa_helper.angler(r,2.*(E-evaluatePotentials(self._pot,r,0.))-L2/r**2.,L,reuse=False)-tra,tr,self._rap)
            tE= E+self._ip(tr,0.)-evaluatePotentials(self._pot,tr,0.)
            tjra= self._isoaa_helper.Jr(tE,L)
            tora= self._isoaa_helper.Or(tE)
            solvera[ii]= tr
            solvedta[ii]= self._isoaa_helper.angler(tr,2.*(E-evaluatePotentials(self._pot,tr,0.))-L2/tr**2.,L,reuse=False)-tra
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
        solvedta[ntr//2+1:]= solvedta[::-1][ntr//2:-1]
        jra[ntr//2+1:]= jra[::-1][ntr//2:-1]
        ora[ntr//2+1:]= ora[::-1][ntr//2:-1]
        dEdL[ntr//2+1:]= dEdL[::-1][ntr//2:-1]
        self._thetara= thetara
        self._solvera= solvera
        self._solvedta= solvedta
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
