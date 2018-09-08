#A 'Binney' quasi-isothermal DF
import math
import warnings
import numpy
import hashlib
from scipy import optimize, interpolate, integrate
from galpy import potential
from galpy import actionAngle
from galpy.actionAngle import actionAngleIsochrone
from galpy.potential import IsochronePotential
from galpy.potential import flatten as flatten_potential
from galpy.orbit import Orbit
from .df import df, _APY_LOADED
from galpy.util import galpyWarning
from galpy.util.bovy_conversion import physical_conversion, \
    potential_physical_input, actionAngle_physical_input, _APY_UNITS
if _APY_LOADED:
    from astropy import units
_NSIGMA=4
_DEFAULTNGL=10
_DEFAULTNGL2=20
class quasiisothermaldf(df):
    """Class that represents a 'Binney' quasi-isothermal DF"""
    def __init__(self,hr,sr,sz,hsr,hsz,pot=None,aA=None,
                 cutcounter=False,
                 _precomputerg=True,_precomputergrmax=None,
                 _precomputergnLz=51,
                 refr=1.,lo=10./220./8.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize a quasi-isothermal DF

        INPUT:

           hr - radial scale length (can be Quantity)

           sr - radial velocity dispersion at the solar radius (can be Quantity)

           sz - vertical velocity dispersion at the solar radius (can be Quantity)

           hsr - radial-velocity-dispersion scale length (can be Quantity)

           hsz - vertial-velocity-dispersion scale length (can be Quantity)

           pot= Potential instance or list thereof

           aA= actionAngle instance used to convert (x,v) to actions [must be an instance of an actionAngle class that computes (J,Omega,angle) for a given (x,v)]

           cutcounter= if True, set counter-rotating stars' DF to zero

           refr= reference radius for dispersions (can be different from ro) (can be Quantity)

           lo= reference angular momentum below where there are significant numbers of retrograde stars (can be Quantity)

           ro= distance from vantage point to GC (kpc; can be Quantity)

           vo= circular velocity at ro (km/s; can be Quantity)

        OTHER INPUTS:

           _precomputerg= if True (default), pre-compute the rL(L)

           _precomputergrmax= if set, this is the maximum R for which to pre-compute rg (default: 5*hr)

           _precomputergnLz if set, number of Lz to pre-compute rg for (default: 51)

        OUTPUT:

           object

        HISTORY:

           2012-07-25 - Started - Bovy (IAS@MPIA)

        """
        df.__init__(self,ro=ro,vo=vo)
        if _APY_LOADED and isinstance(hr,units.Quantity):
            hr= hr.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(sr,units.Quantity):
            sr= sr.to(units.km/units.s).value/self._vo
        if _APY_LOADED and isinstance(sz,units.Quantity):
            sz= sz.to(units.km/units.s).value/self._vo
        if _APY_LOADED and isinstance(hsr,units.Quantity):
            hsr= hsr.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(hsz,units.Quantity):
            hsz= hsz.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(refr,units.Quantity):
            refr= refr.to(units.kpc).value/self._ro
        if _APY_LOADED and isinstance(lo,units.Quantity):
            lo= lo.to(units.kpc*units.km/units.s).value/self._ro/self._vo
        self._hr= hr
        self._sr= sr
        self._sz= sz
        self._hsr= hsr
        self._hsz= hsz
        self._refr= refr
        self._lo= lo
        self._lnsr= math.log(self._sr)
        self._lnsz= math.log(self._sz)
        self._maxVT_hash= None
        self._maxVT_ip= None
        if pot is None:
            raise IOError("pot= must be set")
        self._pot= flatten_potential(pot)
        if aA is None:
            raise IOError("aA= must be set")
        self._aA= aA
        if not self._aA._pot == self._pot:
            if not isinstance(self._aA,actionAngleIsochrone):
                raise IOError("Potential in aA does not appear to be the same as given potential pot")
            elif isinstance(self._pot,IsochronePotential) and \
                    not self._aA.b == self._pot.b and \
                    not self._aA.amp == self._pot._amp:
                raise IOError("Potential in aA does not appear to be the same as given potential pot")
        self._cutcounter= cutcounter
        if _precomputerg:
            if _precomputergrmax is None:
                _precomputergrmax= 5*self._hr
            self._precomputergrmax= _precomputergrmax
            self._precomputergnLz= _precomputergnLz
            self._precomputergLzmin= 0.01
            self._precomputergLzmax= self._precomputergrmax\
                *potential.vcirc(self._pot,self._precomputergrmax)
            self._precomputergLzgrid= numpy.linspace(self._precomputergLzmin,self._precomputergLzmax,self._precomputergnLz)
            self._rls= numpy.array([potential.rl(self._pot,l) for l in self._precomputergLzgrid])
            #Spline interpolate
            self._rgInterp= interpolate.InterpolatedUnivariateSpline(self._precomputergLzgrid,self._rls,k=3)
        else:
            self._precomputergrmax= 0.
            self._rgInterp= None
            self._rls= None
            self._precomputergnr= None
            self._precomputergLzgrid= None
            self._precomputergLzmin= \
                numpy.finfo(numpy.dtype(numpy.float64)).max
            self._precomputergLzmax= \
                numpy.finfo(numpy.dtype(numpy.float64)).min
        self._precomputerg= _precomputerg
        self._glxdef, self._glwdef= \
            numpy.polynomial.legendre.leggauss(_DEFAULTNGL)
        self._glxdef2, self._glwdef2= \
            numpy.polynomial.legendre.leggauss(_DEFAULTNGL2)
        self._glxdef12, self._glwdef12= \
            numpy.polynomial.legendre.leggauss(_DEFAULTNGL//2)
        return None

    @physical_conversion('phasespacedensity',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           return the DF

        INPUT:

           Either:

              a)(jr,lz,jz) tuple; each can be a Quantity
                 where:
                    jr - radial action
                    lz - z-component of angular momentum
                    jz - vertical action

              b) R,vR,vT,z,vz

              c) Orbit instance: initial condition used if that's it, orbit(t)
                 if there is a time given as well

           log= if True, return the natural log

           +scipy.integrate.quadrature kwargs

           func= function of (jr,lz,jz) to multiply f with (useful for moments)

        OUTPUT:

           value of DF

        HISTORY:

           2012-07-25 - Written - Bovy (IAS@MPIA)

        NOTE:

           For Miyamoto-Nagai/adiabatic approximation this seems to take 
           about 30 ms / evaluation in the extended Solar neighborhood
           For a MWPotential/adiabatic approximation this takes about 
           50 ms / evaluation in the extended Solar neighborhood

           For adiabatic-approximation grid this seems to take 
           about 0.67 to 0.75 ms / evaluation in the extended Solar 
           neighborhood (includes some out of the grid)

           up to 200x faster when called with vector R,vR,vT,z,vz

        """
        #First parse log
        log= kwargs.pop('log',False)
        _return_actions= kwargs.pop('_return_actions',False)
        _return_freqs= kwargs.pop('_return_freqs',False)
        _func= kwargs.pop('func',None)
        if 'rg' in kwargs:
            thisrg= kwargs.pop('rg')
            kappa= kwargs.pop('kappa')
            nu= kwargs.pop('nu')
            Omega= kwargs.pop('Omega')
        else:
            thisrg= None
            kappa= None
            nu= None
            Omega= None
        #First parse args
        if len(args) == 1 and not isinstance(args[0],Orbit): #(jr,lz,jz)
            jr,lz,jz= args[0]
            if _APY_LOADED and isinstance(jr,units.Quantity):
                jr= jr.to(units.kpc*units.km/units.s).value/self._ro/self._vo
            if _APY_LOADED and isinstance(lz,units.Quantity):
                lz= lz.to(units.kpc*units.km/units.s).value/self._ro/self._vo
            if _APY_LOADED and isinstance(jz,units.Quantity):
                jz= jz.to(units.kpc*units.km/units.s).value/self._ro/self._vo
        else:
            #Use self._aA to calculate the actions
            try:
                jr,lz,jz= self._aA(*args,use_physical=False,**kwargs)
            except actionAngle.UnboundError:
                if log: return -numpy.finfo(numpy.dtype(numpy.float64)).max
                else: return 0.
            #if isinstance(jr,(list,numpy.ndarray)) and len(jr) > 1: jr= jr[0]
            #if isinstance(jz,(list,numpy.ndarray)) and len(jz) > 1: jz= jz[0]
        if not isinstance(lz,numpy.ndarray) and self._cutcounter and lz < 0.:
            if log: return -numpy.finfo(numpy.dtype(numpy.float64)).max
            else: return 0.
        #First calculate rg
        if thisrg is None:
            thisrg= self._rg(lz)
            #Then calculate the epicycle and vertical frequencies
            kappa, nu= self._calc_epifreq(thisrg), self._calc_verticalfreq(thisrg)
            Omega= numpy.fabs(lz)/thisrg/thisrg
        #calculate surface-densities and sigmas
        lnsurfmass= (self._refr-thisrg)/self._hr
        lnsr= self._lnsr+(self._refr-thisrg)/self._hsr
        lnsz= self._lnsz+(self._refr-thisrg)/self._hsz
        #Calculate func
        if not _func is None:
            if log:
                funcTerm= numpy.log(_func(jr,lz,jz))
            else:
                funcFactor= _func(jr,lz,jz)
        #Calculate fsr
        else:
            if log:
                funcTerm= 0.
            else:
                funcFactor= 1.            
        if log:
            lnfsr= numpy.log(Omega)+lnsurfmass-2.*lnsr-math.log(math.pi)\
                -numpy.log(kappa)\
                +numpy.log(1.+numpy.tanh(lz/self._lo))\
                -kappa*jr*numpy.exp(-2.*lnsr)
            lnfsz= numpy.log(nu)-math.log(2.*math.pi)\
                -2.*lnsz-nu*jz*numpy.exp(-2.*lnsz)
            out= lnfsr+lnfsz+funcTerm
            if isinstance(lz,numpy.ndarray):
                out[numpy.isnan(out)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
                if self._cutcounter: out[(lz < 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            elif numpy.isnan(out): out= -numpy.finfo(numpy.dtype(numpy.float64)).max
        else:
            srm2= numpy.exp(-2.*lnsr)
            fsr= Omega*numpy.exp(lnsurfmass)*srm2/math.pi/kappa\
                *(1.+numpy.tanh(lz/self._lo))\
                *numpy.exp(-kappa*jr*srm2)
            szm2= numpy.exp(-2.*lnsz)
            fsz= nu/2./math.pi*szm2*numpy.exp(-nu*jz*szm2)
            out= fsr*fsz*funcFactor
            if isinstance(lz,numpy.ndarray):
                out[numpy.isnan(out)]= 0.
                if self._cutcounter: out[(lz < 0.)]= 0.
            elif numpy.isnan(out): out= 0.
        if _return_actions and _return_freqs:
            return (out,jr,lz,jz,thisrg,kappa,nu,Omega)
        elif _return_actions:
            return (out,jr,lz,jz)
        elif _return_freqs:
            return (out,thisrg,kappa,nu,Omega)
        else:
            return out

    @potential_physical_input
    @physical_conversion('position',pop=True)
    def estimate_hr(self,R,z=0.,dR=10.**-8.,**kwargs):
        """
        NAME:

           estimate_hr

        PURPOSE:

           estimate the exponential scale length at R

        INPUT:

           R - Galactocentric radius (can be Quantity)

           z= height (default: 0 pc) (can be Quantity)

           dR- range in R to use (can be Quantity)

           density kwargs

        OUTPUT:

           estimated hR

        HISTORY:

           2012-09-11 - Written - Bovy (IAS)

           2013-01-28 - Re-written - Bovy

        """
        Rs= [R-dR/2.,R+dR/2.]
        if z is None:
            sf= numpy.array([self.surfacemass_z(r,use_physical=False,
                                                **kwargs) for r in Rs])
        else:
            sf= numpy.array([self.density(r,z,use_physical=False,
                                          **kwargs) for r in Rs])
        lsf= numpy.log(sf)
        return -dR/(lsf[1]-lsf[0])

    @potential_physical_input
    @physical_conversion('position',pop=True)
    def estimate_hz(self,R,z,dz=10.**-8.,**kwargs):
        """
        NAME:

           estimate_hz

        PURPOSE:

           estimate the exponential scale height at R

        INPUT:

           R - Galactocentric radius (can be Quantity)

           dz - z range to use (can be Quantity)

           density kwargs

        OUTPUT:

           estimated hz

        HISTORY:

           2012-08-30 - Written - Bovy (IAS)

           2013-01-28 - Re-written - Bovy

        """
        if z == 0.:
            zs= [z,z+dz]
        else:
            zs= [z-dz/2.,z+dz/2.]
        sf= numpy.array([self.density(R,zz,use_physical=False,
                                      **kwargs) for zz in zs])
        lsf= numpy.log(sf)
        return -dz/(lsf[1]-lsf[0])

    @potential_physical_input
    @physical_conversion('position',pop=True)
    def estimate_hsr(self,R,z=0.,dR=10.**-8.,**kwargs):
        """
        NAME:

           estimate_hsr

        PURPOSE:

           estimate the exponential scale length of the radial dispersion at R

        INPUT:

           R - Galactocentric radius (can be Quantity)

           z= height (default: 0 pc) (can be Quantity)

           dR- range in R to use (can be Quantity)

           density kwargs

        OUTPUT:

           estimated hsR

        HISTORY:

           2013-03-08 - Written - Bovy (IAS)

        """
        Rs= [R-dR/2.,R+dR/2.]
        sf= numpy.array([self.sigmaR2(r,z,use_physical=False,
                                      **kwargs) for r in Rs])
        lsf= numpy.log(sf)/2.
        return -dR/(lsf[1]-lsf[0])

    @potential_physical_input
    @physical_conversion('position',pop=True)
    def estimate_hsz(self,R,z=0.,dR=10.**-8.,**kwargs):
        """
        NAME:

           estimate_hsz

        PURPOSE:

           estimate the exponential scale length of the vertical dispersion at R

        INPUT:

           R - Galactocentric radius (can be Quantity)

           z= height (default: 0 pc) (can be Quantity)

           dR- range in R to use (can be Quantity)

           density kwargs

        OUTPUT:

           estimated hsz

        HISTORY:

           2013-03-08 - Written - Bovy (IAS)

        """
        Rs= [R-dR/2.,R+dR/2.]
        sf= numpy.array([self.sigmaz2(r,z,use_physical=False,
                                      **kwargs) for r in Rs])
        lsf= numpy.log(sf)/2.
        return -dR/(lsf[1]-lsf[0])

    @potential_physical_input
    @physical_conversion('numbersurfacedensity',pop=True)
    def surfacemass_z(self,R,nz=7,zmax=1.,fixed_quad=True,fixed_order=8,
                      **kwargs):
        """
        NAME:

           surfacemass_z

        PURPOSE:

           calculate the vertically-integrated surface density

        INPUT:

           R - Galactocentric radius (can be Quantity)

           fixed_quad= if True (default), use Gauss-Legendre integration

           fixed_order= (20), order of GL integration to use

           nz= number of zs to use to estimate

           zmax= maximum z to use (can be Quantity)

           density kwargs

        OUTPUT:

           \Sigma(R)

        HISTORY:

           2012-08-30 - Written - Bovy (IAS)

        """
        if fixed_quad:
            return 2.*integrate.fixed_quad(lambda x: self.density(R*numpy.ones(fixed_order),x,use_physical=False),
                                           0.,.5,n=fixed_order)[0]
        zs= numpy.linspace(0.,zmax,nz)
        sf= numpy.array([self.density(R,z,use_physical=False,
                                      **kwargs) for z in zs])
        lsf= numpy.log(sf)
        #Interpolate
        lsfInterp= interpolate.UnivariateSpline(zs,
                                                lsf,
                                                k=3)
        #Integrate
        return 2.*integrate.quad((lambda x: numpy.exp(lsfInterp(x))),
                                 0.,1.)[0]

    def vmomentdensity(self,*args,**kwargs):
        """
        NAME:

           vmomentdensity

        PURPOSE:

           calculate the an arbitrary moment of the velocity distribution 
           at R times the density

        INPUT:

           R - radius at which to calculate the moment(/ro)

           n - vR^n

           m - vT^m

           o - vz^o

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the vR and vz velocities over (when doing explicit numerical integral; default: 4)
           
           vTmax - upper limit for integration over vT (default: 1.5)

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= use Gauss-Legendre

           _returngl= if True, return the evaluated DF

           _return_actions= if True, return the evaluated actions (does not work with _returngl currently)

           _return_freqs= if True, return the evaluated frequencies and rg (does not work with _returngl currently)

        OUTPUT:

           <vR^n vT^m  x density> at R,z (no support for units)

        HISTORY:

           2012-08-06 - Written - Bovy (IAS@MPIA)

        """
        use_physical= kwargs.pop('use_physical',True)
        ro= kwargs.pop('ro',None)
        if ro is None and hasattr(self,'_roSet') and self._roSet:
            ro= self._ro
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        vo= kwargs.pop('vo',None)
        if vo is None and hasattr(self,'_voSet') and self._voSet:
            vo= self._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        if use_physical and not vo is None and not ro is None:
            fac= vo**(args[2]+args[3]+args[4])/ro**3
            if _APY_UNITS:
                u= 1/units.kpc**3*(units.km/units.s)**(args[2]+args[3]+args[4])
            out= self._vmomentdensity(*args,**kwargs)
            if _APY_UNITS:
                return units.Quantity(out*fac,unit=u)
            else:
                return out*fac
        else:
            return self._vmomentdensity(*args,**kwargs)  

    def _vmomentdensity(self,R,z,n,m,o,nsigma=None,mc=False,nmc=10000,
                       _returnmc=False,_vrs=None,_vts=None,_vzs=None,
                       _rawgausssamples=False,
                       gl=False,ngl=_DEFAULTNGL,_returngl=False,_glqeval=None,
                       _return_actions=False,_jr=None,_lz=None,_jz=None,
                       _return_freqs=False,
                       _rg=None,_kappa=None,_nu=None,_Omega=None,
                       _sigmaR1=None,_sigmaz1=None,
                       **kwargs):
        """Non-physical version of vmomentdensity, otherwise the same"""
        if isinstance(R,numpy.ndarray):
            return numpy.array([self._vmomentdensity(r,zz,n,m,o,nsigma=nsigma,
                                                    mc=mc,nmc=nmc,
                                                    gl=gl,ngl=ngl,**kwargs) for r,zz in zip(R,z)])
        if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                actionAngle.actionAngleAdiabaticGrid)):
            if n % 2 == 1. or o % 2 == 1.:
                return 0. #we know this must be the case
        if nsigma == None:
            nsigma= _NSIGMA
        if _sigmaR1 is None:
            sigmaR1= self._sr*numpy.exp((self._refr-R)/self._hsr)
        else:
            sigmaR1= _sigmaR1
        if _sigmaz1 is None:
            sigmaz1= self._sz*numpy.exp((self._refr-R)/self._hsz)
        else:
            sigmaz1= _sigmaz1
        thisvc= potential.vcirc(self._pot,R,use_physical=False)
        #Use the asymmetric drift equation to estimate va
        gamma= numpy.sqrt(0.5)
        va= sigmaR1**2./2./thisvc\
            *(gamma**2.-1. #Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
               +R*(1./self._hr+2./self._hsr))
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            if not _glqeval is None and ngl != _glqeval.shape[0]:
                _glqeval= None
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vRgl= nsigma*sigmaR1/2.*(glx+1.)
                vzgl= nsigma*sigmaz1/2.*(glx+1.)
                vRglw= glw
                vzglw= glw
            else:
                vRgl= nsigma*sigmaR1/2.*(glx12+1.)
                #vRgl= 1.5/2.*(glx12+1.)
                vRgl= list(vRgl)
                vRgl.extend(-nsigma*sigmaR1/2.*(glx12+1.))
                #vRgl.extend(-1.5/2.*(glx12+1.))
                vRgl= numpy.array(vRgl)
                vzgl= nsigma*sigmaz1/2.*(glx12+1.)
                #vzgl= 1.5/2.*(glx12+1.)
                vzgl= list(vzgl)
                vzgl.extend(-nsigma*sigmaz1/2.*(glx12+1.))
                #vzgl.extend(-1.5/2.*(glx12+1.))
                vzgl= numpy.array(vzgl)
                vRglw= glw12
                vRglw= list(vRglw)
                vRglw.extend(glw12)
                vRglw= numpy.array(vRglw)
                vzglw= glw12
                vzglw= list(vzglw)
                vzglw.extend(glw12)
                vzglw= numpy.array(vzglw)
            if 'vTmax' in kwargs: vTmax = kwargs['vTmax']
            else:                 vTmax = 1.5
            vTgl= vTmax/2.*(glx+1.)
            #Tile everything
            vTgl= numpy.tile(vTgl,(ngl,ngl,1)).T
            vRgl= numpy.tile(numpy.reshape(vRgl,(1,ngl)).T,(ngl,1,ngl))
            vzgl= numpy.tile(vzgl,(ngl,ngl,1))
            vTglw= numpy.tile(glw,(ngl,ngl,1)).T #also tile weights
            vRglw= numpy.tile(numpy.reshape(vRglw,(1,ngl)).T,(ngl,1,ngl))
            vzglw= numpy.tile(vzglw,(ngl,ngl,1))
            #evaluate
            if _glqeval is None and _jr is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self(R+numpy.zeros(ngl*ngl*ngl),
                                           vRgl.flatten(),
                                           vTgl.flatten(),
                                           z+numpy.zeros(ngl*ngl*ngl),
                                           vzgl.flatten(),
                                           log=True,
                                           _return_actions=True,
                                           _return_freqs=True,
                                           use_physical=False)
                logqeval= numpy.reshape(logqeval,(ngl,ngl,ngl))
            elif not _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self((_jr,_lz,_jz),
                                                                 log=True,
                                                                 _return_actions=True,
                                                                 _return_freqs=True,
                                                                 use_physical=False)
                logqeval= numpy.reshape(logqeval,(ngl,ngl,ngl))
            elif not _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self((_jr,_lz,_jz),
                                           rg=_rg,kappa=_kappa,nu=_nu,
                                           Omega=_Omega,
                                           log=True,
                                           _return_actions=True,
                                           _return_freqs=True,
                                           use_physical=False)
                logqeval= numpy.reshape(logqeval,(ngl,ngl,ngl))
            else:
                logqeval= _glqeval
            if _returngl:
                return (numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                  *vTglw*vRglw*vzglw)*sigmaR1*sigmaz1*0.125*vTmax*nsigma**2,
                        logqeval)
            elif _return_actions and _return_freqs:
                return (numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                  *vTglw*vRglw*vzglw)*sigmaR1*sigmaz1*0.125*vTmax*nsigma**2,
                        jr,lz,jz,
                        rg,kappa,nu,Omega)
            elif _return_actions:
                return (numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                  *vTglw*vRglw*vzglw)*sigmaR1*sigmaz1*0.125*vTmax*nsigma**2,
                        jr,lz,jz)
            else:
                return numpy.sum(numpy.exp(logqeval)*vRgl**n*vTgl**m*vzgl**o
                                 *vTglw*vRglw*vzglw*sigmaR1*sigmaz1*0.125*vTmax*nsigma**2)
        elif mc:
            mvT= (thisvc-va)/gamma/sigmaR1
            if _vrs is None:
                vrs= numpy.random.normal(size=nmc)
            else:
                vrs= _vrs
            if _vts is None:
                vts= numpy.random.normal(size=nmc)+mvT
            else:
                if _rawgausssamples:
                    vts= _vts+mvT
                else:
                    vts= _vts
            if _vzs is None:
                vzs= numpy.random.normal(size=nmc)
            else:
                vzs= _vzs
            Is= _vmomentsurfaceMCIntegrand(vzs,vrs,vts,numpy.ones(nmc)*R,
                                           numpy.ones(nmc)*z,
                                           self,sigmaR1,gamma,sigmaz1,mvT,
                                           n,m,o)
            if _returnmc:
                if _rawgausssamples:
                    return (numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o),
                        vrs,vts-mvT,vzs)
                else:
                    return (numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o),
                            vrs,vts,vzs)
            else:
                return numpy.mean(Is)*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o)
        else: #pragma: no cover because this is too slow; a warning is shown
            warnings.warn("Calculations using direct numerical integration using tplquad is not recommended and extremely slow; it has also not been carefully tested",galpyWarning)
            return integrate.tplquad(_vmomentsurfaceIntegrand,
                                     1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                     1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     lambda x,y: 0., lambda x,y: nsigma,
                                     (R,z,self,sigmaR1,gamma,sigmaz1,n,m,o),
                                     **kwargs)[0]*sigmaR1**(2.+n+m)*gamma**(1.+m)*sigmaz1**(1.+o)
        
    def jmomentdensity(self,*args,**kwargs):
        """
        NAME:

           jmomentdensity
        PURPOSE:

           calculate the an arbitrary moment of an action
           of the velocity distribution 
           at R times the surfacmass
        INPUT:

           R - radius at which to calculate the moment(/ro)

           n - jr^n

           m - lz^m

           o - jz^o

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over (when doing explicit numerical integral)

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

        OUTPUT:

           <jr^n lz^m jz^o  x density> at R (no support for units)

        HISTORY:

           2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        use_physical= kwargs.pop('use_physical',True)
        ro= kwargs.pop('ro',None)
        if ro is None and hasattr(self,'_roSet') and self._roSet:
            ro= self._ro
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        vo= kwargs.pop('vo',None)
        if vo is None and hasattr(self,'_voSet') and self._voSet:
            vo= self._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        if use_physical and not vo is None and not ro is None:
            fac= (ro*vo)**(args[2]+args[3]+args[4])/ro**3
            if _APY_UNITS:
                u= 1/units.kpc**3*(units.kpc*units.km/units.s)**(args[2]+args[3]+args[4])
            out= self._jmomentdensity(*args,**kwargs)
            if _APY_UNITS:
                return units.Quantity(out*fac,unit=u)
            else:
                return out*fac
        else:
            return self._jmomentdensity(*args,**kwargs)  

    def _jmomentdensity(self,R,z,n,m,o,nsigma=None,mc=True,nmc=10000,
                       _returnmc=False,_vrs=None,_vts=None,_vzs=None,
                       **kwargs):
        """Non-physical version of jmomentdensity, otherwise the same"""
        if nsigma == None:
            nsigma= _NSIGMA
        sigmaR1= self._sr*numpy.exp((self._refr-R)/self._hsr)
        sigmaz1= self._sz*numpy.exp((self._refr-R)/self._hsz)
        thisvc= potential.vcirc(self._pot,R,use_physical=False)
        #Use the asymmetric drift equation to estimate va
        gamma= numpy.sqrt(0.5)
        va= sigmaR1**2./2./thisvc\
            *(gamma**2.-1. #Assume close to flat rotation curve, sigphi2/sigR2 =~ 0.5
               +R*(1./self._hr+2./self._hsr))
        if math.fabs(va) > sigmaR1: va = 0.#To avoid craziness near the center
        if mc:
            mvT= (thisvc-va)/gamma/sigmaR1
            if _vrs is None:
                vrs= numpy.random.normal(size=nmc)
            else:
                vrs= _vrs
            if _vts is None:
                vts= numpy.random.normal(size=nmc)+mvT
            else:
                vts= _vts
            if _vzs is None:
                vzs= numpy.random.normal(size=nmc)
            else:
                vzs= _vzs
            Is= _jmomentsurfaceMCIntegrand(vzs,vrs,vts,numpy.ones(nmc)*R,numpy.ones(nmc)*z,self,sigmaR1,gamma,sigmaz1,mvT,n,m,o)
            if _returnmc:
                return (numpy.mean(Is)*sigmaR1**2.*gamma*sigmaz1,
                        vrs,vts,vzs)
            else:
                return numpy.mean(Is)*sigmaR1**2.*gamma*sigmaz1
        else: #pragma: no cover because this is too slow; a warning is shown
            warnings.warn("Calculations using direct numerical integration using tplquad is not recommended and extremely slow; it has also not been carefully tested",galpyWarning)
            return integrate.tplquad(_jmomentsurfaceIntegrand,
                                     1./gamma*(thisvc-va)/sigmaR1-nsigma,
                                     1./gamma*(thisvc-va)/sigmaR1+nsigma,
                                     lambda x: 0., lambda x: nsigma,
                                     lambda x,y: 0., lambda x,y: nsigma,
                                     (R,z,self,sigmaR1,gamma,sigmaz1,n,m,o),
                                     **kwargs)[0]*sigmaR1**2.*gamma*sigmaz1
        
    @potential_physical_input
    @physical_conversion('numberdensity',pop=True)
    def density(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           density

        PURPOSE:

           calculate the density at R,z by marginalizing over velocity

        INPUT:

           R - radius at which to calculate the density (can be Quantity)

           z - height at which to calculate the density (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           density at (R,z)

        HISTORY:

           2012-07-26 - Written - Bovy (IAS@MPIA)

        """
        return self._vmomentdensity(R,z,0.,0.,0.,
                                   nsigma=nsigma,mc=mc,nmc=nmc,
                                   gl=gl,ngl=ngl,
                                   **kwargs)
    
    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaR2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           sigmaR2

        PURPOSE:

           calculate sigma_R^2 by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           sigma_R^2

        HISTORY:

           2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,2.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,2.,0.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,2.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaRz(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           sigmaRz

        PURPOSE:

           calculate sigma_RZ^2 by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           sigma_Rz^2

        HISTORY:

           2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,1.,0.,1.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,1.,0.,1.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,1.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('angle',pop=True)
    def tilt(self,R,z,nsigma=None,mc=False,nmc=10000,
             gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           tilt

        PURPOSE:

           calculate the tilt of the velocity ellipsoid by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           tilt in rad

        HISTORY:

           2012-12-23 - Written - Bovy (IAS)

           2017-10-28 - Changed return unit to rad - Bovy (UofT)

        """
        warnings.warn("In versions >1.3, the output unit of quasiisothermaldf.tilt has been changed to radian (from degree before)",galpyWarning)
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            tsigmar2= self._vmomentdensity(R,z,2.,0.,0.,
                                              nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                              _vrs=vrs,_vts=vts,_vzs=vzs,
                                              **kwargs)/surfmass
            tsigmaz2= self._vmomentdensity(R,z,0.,0.,2.,
                                              nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                              _vrs=vrs,_vts=vts,_vzs=vzs,
                                              **kwargs)/surfmass
            tsigmarz= self._vmomentdensity(R,z,1.,0.,1.,
                                              nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                              _vrs=vrs,_vts=vts,_vzs=vzs,
                                              **kwargs)/surfmass
            return 0.5*numpy.arctan(2.*tsigmarz/(tsigmar2-tsigmaz2))
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            tsigmar2= self._vmomentdensity(R,z,2.,0.,0.,
                                             ngl=ngl,gl=gl,
                                             _glqeval=glqeval,
                                             **kwargs)/surfmass
            tsigmaz2= self._vmomentdensity(R,z,0.,0.,2.,
                                              ngl=ngl,gl=gl,
                                              _glqeval=glqeval,
                                              **kwargs)/surfmass
            tsigmarz= self._vmomentdensity(R,z,1.,0.,1.,
                                              ngl=ngl,gl=gl,
                                              _glqeval=glqeval,
                                              **kwargs)/surfmass
            return 0.5*numpy.arctan(2.*tsigmarz/(tsigmar2-tsigmaz2))
        else:
            raise NotImplementedError("Use either mc=True or gl=True")
        
    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaz2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           sigmaz2

        PURPOSE:

           calculate sigma_z^2 by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           sigma_z^2

        HISTORY:

           2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,0.,0.,2.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,0.,0.,2.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,0.,0.,2.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvT(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           meanvT

        PURPOSE:

           calculate the mean rotational velocity by marginalizing over velocity 

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           meanvT

        HISTORY:

           2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,0.,1.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,0.,1.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvR(self,R,z,nsigma=None,mc=False,nmc=10000,
               gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           meanvR

        PURPOSE:

           calculate the mean radial velocity by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           meanvR

        HISTORY:

           2012-12-23 - Written - Bovy (IAS)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,1.,0.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('velocity',pop=True)
    def meanvz(self,R,z,nsigma=None,mc=False,nmc=10000,
               gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           meanvz

        PURPOSE:

           calculate the mean vertical velocity by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           meanvz

        HISTORY:

           2012-12-23 - Written - Bovy (IAS)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._vmomentdensity(R,z,0.,0.,1.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            return self._vmomentdensity(R,z,0.,0.,1.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._vmomentdensity(R,z,0.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('velocity2',pop=True)
    def sigmaT2(self,R,z,nsigma=None,mc=False,nmc=10000,
                gl=True,ngl=_DEFAULTNGL,**kwargs):
        """
        NAME:

           sigmaT2

        PURPOSE:

           calculate sigma_T^2 by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

           gl= if True, calculate using Gauss-Legendre integration

           ngl= if gl, use ngl-th order Gauss-Legendre integration for each dimension

        OUTPUT:

           sigma_T^2

        HISTORY:

           2012-07-30 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            mvt= self._vmomentdensity(R,z,0.,1.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
            return self._vmomentdensity(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                           **kwargs)/surfmass\
                                           -mvt**2.
        elif gl:
            surfmass, glqeval= self._vmomentdensity(R,z,0.,0.,0.,
                                                       gl=gl,ngl=ngl,
                                                       _returngl=True,
                                                       **kwargs)
            mvt= self._vmomentdensity(R,z,0.,1.,0.,
                                         ngl=ngl,gl=gl,
                                         _glqeval=glqeval,
                                         **kwargs)/surfmass
            return self._vmomentdensity(R,z,0.,2.,0.,
                                           ngl=ngl,gl=gl,
                                           _glqeval=glqeval,
                                           **kwargs)/surfmass-mvt**2.

        else: #pragma: no cover because this is too slow; a warning is shown
            surfmass= self._vmomentdensity(R,z,0.,0.,0.,
                                              nsigma=nsigma,mc=mc,nmc=nmc,
                                              **kwargs)
            return (self._vmomentdensity(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/surfmass\
                        -(self._vmomentdensity(R,z,0.,2.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/surfmass)**2.)

    @potential_physical_input
    @physical_conversion('action',pop=True)
    def meanjr(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:

           meanjr

        PURPOSE:

           calculate the mean radial action by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

        OUTPUT:

           meanjr

        HISTORY:

           2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._jmomentdensity(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._jmomentdensity(R,z,1.,0.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('action',pop=True)
    def meanlz(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:

           meanlz

        PURPOSE:

           calculate the mean angular momemtum by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

        OUTPUT:

           meanlz

        HISTORY:

           2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._jmomentdensity(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._jmomentdensity(R,z,0.,1.,0.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    @physical_conversion('action',pop=True)
    def meanjz(self,R,z,nsigma=None,mc=True,nmc=10000,**kwargs):
        """
        NAME:

           meanjz

        PURPOSE:

           calculate the mean vertical action by marginalizing over velocity

        INPUT:

           R - radius at which to calculate this (can be Quantity)

           z - height at which to calculate this (can be Quantity)

        OPTIONAL INPUT:

           nsigma - number of sigma to integrate the velocities over

           scipy.integrate.tplquad kwargs epsabs and epsrel

           mc= if True, calculate using Monte Carlo integration

           nmc= if mc, use nmc samples

        OUTPUT:

           meanjz

        HISTORY:

           2012-08-09 - Written - Bovy (IAS@MPIA)

        """
        if mc:
            surfmass, vrs, vts, vzs= self._vmomentdensity(R,z,0.,0.,0.,
                                                             nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=True,
                                                             **kwargs)
            return self._jmomentdensity(R,z,0.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,_returnmc=False,
                                           _vrs=vrs,_vts=vts,_vzs=vzs,
                                                             **kwargs)/surfmass
        else: #pragma: no cover because this is too slow; a warning is shown
            return (self._jmomentdensity(R,z,0.,0.,1.,
                                           nsigma=nsigma,mc=mc,nmc=nmc,
                                           **kwargs)/
                    self._vmomentdensity(R,z,0.,0.,0.,
                                            nsigma=nsigma,mc=mc,nmc=nmc,
                                            **kwargs))
        
    @potential_physical_input
    def sampleV(self,R,z,n=1,**kwargs):
        """
        NAME:

           sampleV

        PURPOSE:

           sample a radial, azimuthal, and vertical velocity at R,z

        INPUT:

           R - Galactocentric distance (can be Quantity)

           z - height (can be Quantity)

           n= number of distances to sample

        OUTPUT:

           list of samples

        HISTORY:

           2012-12-17 - Written - Bovy (IAS)

        """
        use_physical= kwargs.pop('use_physical',True)
        vo= kwargs.pop('vo',None)
        if vo is None and hasattr(self,'_voSet') and self._voSet:
            vo= self._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        #Determine the maximum of the velocity distribution
        maxVR= 0.
        maxVz= 0.
        maxVT= optimize.fmin_powell((lambda x: -self(R,0.,x,z,0.,log=True,
                                                     use_physical=False)),
                                    1.)
        logmaxVD= self(R,maxVR,maxVT,z,maxVz,log=True,use_physical=False)
        #Now rejection-sample
        vRs= []
        vTs= []
        vzs= []
        while len(vRs) < n:
            nmore= n-len(vRs)+1
            #sample
            propvR= numpy.random.normal(size=nmore)*2.*self._sr
            propvT= numpy.random.normal(size=nmore)*2.*self._sr+maxVT
            propvz= numpy.random.normal(size=nmore)*2.*self._sz
            VDatprop= self(R+numpy.zeros(nmore),
                           propvR,propvT,z+numpy.zeros(nmore),
                           propvz,log=True,use_physical=False)-logmaxVD
            VDatprop-= -0.5*(propvR**2./4./self._sr**2.+propvz**2./4./self._sz**2.\
                                 +(propvT-maxVT)**2./4./self._sr**2.)
            VDatprop= numpy.reshape(VDatprop,(nmore))
            indx= (VDatprop > numpy.log(numpy.random.random(size=nmore))) #accept
            vRs.extend(list(propvR[indx]))
            vTs.extend(list(propvT[indx]))
            vzs.extend(list(propvz[indx]))
        out= numpy.empty((n,3))
        out[:,0]= vRs[0:n]
        out[:,1]= vTs[0:n]
        out[:,2]= vzs[0:n]
        if use_physical and not vo is None:
            if _APY_UNITS:
                return units.Quantity(out*vo,unit=units.km/units.s)
            else:
                return out*vo
        else:
            return out

    @potential_physical_input
    def sampleV_interpolate(self,R,z,R_pixel,z_pixel,num_std=3,R_min=None,
                            R_max=None,z_max=None,**kwargs):
        """
        NAME:
            
            sampleV_interpolate
    
        PURPOSE:
            
            Given an array of R and z coordinates of stars, return the 
            positions and their radial, azimuthal, and vertical velocity.
    
        INPUT:
            
            R - array of Galactocentric distance (can be Quantity)

            z - array of height (can be Quantity)
            
            R_pixel, z_pixel= the pixel size for creating the grid for
                   interpolation (in natural unit)
            
            num_std= number of standard deviation to be considered outliers
                      sampled separately from interpolation
                      
            R_min, R_max, z_max= optional edges of the grid
    
        OUTPUT:
            
            coord_v= a numpy array containing the sampled velocity, (vR, vT, vz),
                     where each row correspond to the row of (R,z)
    
        HISTORY:
            
            2018-08-10 - Written - Samuel Wong (University of Toronto)
            
        """
        use_physical= kwargs.pop('use_physical',True)
        vo= kwargs.pop('vo',None)
        if vo is None and hasattr(self,'_voSet') and self._voSet:
            vo= self._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        #Initialize output array
        coord_v= numpy.empty((numpy.size(R), 3))
        #Since the sign of z doesn't matter, work with absolute value of z
        z= numpy.abs(z)
        # Grid edges
        if R_min is None:
            R_min= numpy.amax([numpy.mean(R)-num_std*numpy.std(R),
                               numpy.amin(R)])
        if R_max is None:
            R_max= numpy.amin([numpy.mean(R)+num_std*numpy.std(R),
                               numpy.amax(R)])
        if z_max is None:
            z_max= numpy.amin([numpy.mean(z)+num_std*numpy.std(z),
                               numpy.amax(z)])
        z_min= 0. #Always start grid at z=0 for stars close to plane
        #Separate the coodinates into outliers and normal points
        #Define outliers as points outside of grid
        mask= numpy.any([R < R_min, R > R_max, z > z_max],axis = 0)
        outliers_R= R[mask]
        outliers_z= z[mask]
        normal_R= R[~mask]
        normal_z= z[~mask]
        #Sample the velocity of outliers directly (without interpolation)
        outlier_coord_v= numpy.empty((outliers_R.size, 3))
        for i in range(outliers_R.size):
            outlier_coord_v[i]= self.sampleV(outliers_R[i], outliers_z[i],
                                             use_physical=False)[0]
        #Prepare for optimizing maxVT on a grid
         #Get the new hash of the parameters of grid
        new_hash= hashlib.md5(numpy.array([R_min,R_max,z_max,R_pixel,z_pixel])).hexdigest()
        #Reuse old interpolated object if new hash matches the old one
        if new_hash == self._maxVT_hash:
            ip_max_vT= self._maxVT_ip
        #Generate a new interpolation object if different from before
        else:
            R_number= int((R_max - R_min)/R_pixel)
            z_number= int((z_max - z_min)/z_pixel)
            R_linspace= numpy.linspace(R_min, R_max, R_number)
            z_linspace= numpy.linspace(z_min, z_max, z_number)
            Rv, zv= numpy.meshgrid(R_linspace, z_linspace)
            grid= numpy.dstack((Rv, zv)) #This grid stores (R,z) coordinate
            #Grid is a 3 dimensional array since it stores pairs of values, but 
            #grid max vT is a 2 dimensinal array
            grid_max_vT= numpy.empty((grid.shape[0], grid.shape[1]))
            #Optimize max_vT on the grid
            for i in range(z_number):
                for j in range(R_number):
                    R, z= grid[i][j]
                    grid_max_vT[i][j]= optimize.fmin_powell((lambda x: -self(
                            R,0.,x,z,0.,log=True, use_physical=False)),1.)
            #Determine degree of interpolation
            ky= numpy.min([R_number-1,3])
            kx= numpy.min([z_number-1,3])
            #Generate interpolation object
            ip_max_vT= interpolate.RectBivariateSpline(z_linspace,R_linspace,
                                                       grid_max_vT,kx=kx,ky=ky)
            #Store interpolation object
            self._maxVT_ip= ip_max_vT
            #Update hash of parameters
            self._maxVT_hash= new_hash
        #Evaluate interpolation object to get maxVT at the normal coordinates
        normal_max_vT= ip_max_vT.ev(normal_z, normal_R)
        #Sample all 3 velocities at a normal point and use interpolated vT
        normal_coord_v= \
            self._sampleV_preoptimized(normal_R,normal_z,normal_max_vT)
        #Combine normal and outlier result, preserving original order
        coord_v[mask]= outlier_coord_v
        coord_v[~mask]= normal_coord_v
        if use_physical and not vo is None:
            if _APY_UNITS:
                return units.Quantity(coord_v*vo,unit=units.km/units.s)
            else:
                return coord_v*vo
        else:
            return coord_v

    def _sampleV_preoptimized(self,R,z,maxVT):
        """
        NAME:

           _sampleV_preoptimized

        PURPOSE:

           sample a radial, azimuthal, and vertical velocity at R,z;
           R,z can be an array of positions maxVT is already optimized

        INPUT:

           R - Galactocentric distance (can be Quantity)

           z - height (can be Quantity)
           
           maxVT - an array of pre-optimized maximum vT at corresponding R,z

        OUTPUT:

           a numpy array containing the sampled velocity, (vR, vT, vz),
           where each row correspond to the row of (R,z)

        HISTORY:

           2018-08-09 - Written - Samuel Wong (University of Toronto)

        """
        length = numpy.size(R)
        out= numpy.empty((length,3)) #Initialize output
        #Determine the maximum of the velocity distribution
        maxVR= numpy.zeros(length)
        maxVz= numpy.zeros(length)
        logmaxVD= self(R,maxVR,maxVT,z,maxVz,log=True,use_physical=False)
        #Now rejection-sample
        #Intiialize boolean index of position remaining to be sampled
        remain_indx = numpy.full(length,True)
        while numpy.any(remain_indx):
            nmore= numpy.sum(remain_indx)
            propvR= numpy.random.normal(size=nmore)*2.*self._sr
            propvT= numpy.random.normal(size=nmore)*2.*self._sr+maxVT[remain_indx]
            propvz= numpy.random.normal(size=nmore)*2.*self._sz
            VDatprop= self(R[remain_indx],propvR,propvT,z[remain_indx],propvz,
                           log=True, use_physical=False)-logmaxVD[remain_indx]
            VDatprop-= -0.5*(propvR**2./4./self._sr**2.+
                             propvz**2./4./self._sz**2.+
                             (propvT-maxVT[remain_indx])**2./4./self._sr**2.)
            accept_indx= (VDatprop > numpy.log(numpy.random.random(size=nmore)))
            vR_accept= propvR[accept_indx]
            vT_accept= propvT[accept_indx]
            vz_accept= propvz[accept_indx]
            #Get the indexing of rows of output array that need to be updated
            #with newly accepted velocity
            to_change= numpy.copy(remain_indx)
            to_change[remain_indx]= accept_indx
            out[to_change]= numpy.stack((vR_accept,vT_accept,vz_accept), axis = 1)
            #Removing accepted sampled from remain index
            remain_indx[remain_indx]= ~accept_indx
        return out

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity2',pop=True)
    def pvR(self,vR,R,z,gl=True,ngl=_DEFAULTNGL2,nsigma=4.,vTmax=1.5):
        """
        NAME:

           pvR

        PURPOSE:

           calculate the marginalized vR probability at this location (NOT normalized by the density)

        INPUT:

           vR - radial velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration

           nsigma - sets integration limits to [-1,+1]*nsigma*sigma_z(R) for integration over vz (default: 4)

           vTmax - sets integration limits to [0,vTmax] for integration over vT (default: 1.5)

        OUTPUT:

           p(vR,R,z)

        HISTORY:

           2012-12-22 - Written - Bovy (IAS)

        """
        sigmaz1= self._sz*numpy.exp((self._refr-R)/self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vzgl= nsigma*sigmaz1/2.*(glx+1.)
                vzglw= glw
                vzfac= nsigma*sigmaz1  #2 x integration over [0,nsigma*sigmaz1]
            else:
                vzgl= nsigma*sigmaz1/2.*(glx12+1.)
                vzgl= list(vzgl)
                vzgl.extend(-nsigma*sigmaz1/2.*(glx12+1.))
                vzgl= numpy.array(vzgl)
                vzglw= glw12
                vzglw= list(vzglw)
                vzglw.extend(glw12)
                vzglw= numpy.array(vzglw)
                vzfac = 0.5*nsigma*sigmaz1   #integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            vTgl= vTmax/2.*(glx+1.)
            vTfac= 0.5 * vTmax  #integration over [0.,vTmax]
            #Tile everything
            vTgl= numpy.tile(vTgl,(ngl,1)).T
            vzgl= numpy.tile(vzgl,(ngl,1))
            vTglw= numpy.tile(glw,(ngl,1)).T #also tile weights
            vzglw= numpy.tile(vzglw,(ngl,1))
            #evaluate
            logqeval= numpy.reshape(self(R+numpy.zeros(ngl*ngl),
                                         vR+numpy.zeros(ngl*ngl),
                                         vTgl.flatten(),
                                         z+numpy.zeros(ngl*ngl),
                                         vzgl.flatten(),
                                         log=True,
                                         use_physical=False),
                                    (ngl,ngl))
            return numpy.sum(numpy.exp(logqeval)*vTglw*vzglw*vzfac)*vTfac

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity2',pop=True)
    def pvT(self,vT,R,z,gl=True,ngl=_DEFAULTNGL2,nsigma=4.):
        """
        NAME:

           pvT

        PURPOSE:

           calculate the marginalized vT probability at this location (NOT normalized by the density)

        INPUT:

           vT - tangential velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration

           nsigma - sets integration limits to [-1,+1]*nsigma*sigma(R) for integration over vz and vR (default: 4)

        OUTPUT:

           p(vT,R,z)

        HISTORY:

           2012-12-22 - Written - Bovy (IAS)
           2018-01-12 - Added Gauss-Legendre integration prefactor nsigma^2/4 - Trick (MPA)

        """
        sigmaR1= self._sr*numpy.exp((self._refr-R)/self._hsr)
        sigmaz1= self._sz*numpy.exp((self._refr-R)/self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vRgl= nsigma*sigmaR1/2.*(glx+1.)
                vzgl= nsigma*sigmaz1/2.*(glx+1.)
                vRglw= glw
                vzglw= glw
                vRfac= nsigma*sigmaR1  #2 x integration over [0,nsigma*sigmaR1]
                vzfac= nsigma*sigmaz1  #2 x integration over [0,nsigma*sigmaz1]
            else:
                vRgl= nsigma*sigmaR1/2.*(glx12+1.)
                vRgl= list(vRgl)
                vRgl.extend(-nsigma*sigmaR1/2.*(glx12+1.))
                vRgl= numpy.array(vRgl)
                vzgl= nsigma*sigmaz1/2.*(glx12+1.)
                vzgl= list(vzgl)
                vzgl.extend(-nsigma*sigmaz1/2.*(glx12+1.))
                vzgl= numpy.array(vzgl)
                vRglw= glw12
                vRglw= list(vRglw)
                vRglw.extend(glw12)
                vRglw= numpy.array(vRglw)
                vzglw= glw12
                vzglw= list(vzglw)
                vzglw.extend(glw12)
                vzglw= numpy.array(vzglw)
                vRfac = 0.5*nsigma*sigmaR1   #integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
                vzfac = 0.5*nsigma*sigmaz1   #integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            #Tile everything
            vRgl= numpy.tile(vRgl,(ngl,1)).T
            vzgl= numpy.tile(vzgl,(ngl,1))
            vRglw= numpy.tile(vRglw,(ngl,1)).T #also tile weights
            vzglw= numpy.tile(vzglw,(ngl,1))
            #evaluate
            logqeval= numpy.reshape(self(R+numpy.zeros(ngl*ngl),
                                         vRgl.flatten(),
                                         vT+numpy.zeros(ngl*ngl),
                                         z+numpy.zeros(ngl*ngl),
                                         vzgl.flatten(),
                                         log=True,
                                         use_physical=False),
                                    (ngl,ngl))
            return numpy.sum(numpy.exp(logqeval)*vRglw*vzglw*vRfac*vzfac)

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity2',pop=True)
    def pvz(self,vz,R,z,gl=True,ngl=_DEFAULTNGL2,
            nsigma=4.,vTmax=1.5,
            _return_actions=False,_jr=None,_lz=None,_jz=None,
            _return_freqs=False,
            _rg=None,_kappa=None,_nu=None,_Omega=None,
            _sigmaR1=None):
        """
        NAME:

           pvz

        PURPOSE:

           calculate the marginalized vz probability at this location (NOT normalized by the density)

        INPUT:

           vz - vertical velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration

           nsigma - sets integration limits to [-1,+1]*nsigma*sigma_R(R) for integration over vR (default: 4)

           vTmax - sets integration limits to [0,vTmax] for integration over vT (default: 1.5)

        OUTPUT:

           p(vz,R,z)

        HISTORY:

           2012-12-22 - Written - Bovy (IAS)

        """
        if _sigmaR1 is None:
            sigmaR1= self._sr*numpy.exp((self._refr-R)/self._hsr)
        else:
            sigmaR1= _sigmaR1
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vRgl= (glx+1.)
                vRglw= glw
                vRfac= nsigma*sigmaR1  #2 x integration over [0,nsigma*sigmaR1]
            else:
                vRgl= (glx12+1.)
                vRgl= list(vRgl)
                vRgl.extend(-(glx12+1.))
                vRgl= numpy.array(vRgl)
                vRglw= glw12
                vRglw= list(vRglw)
                vRglw.extend(glw12)
                vRglw= numpy.array(vRglw)
                vRfac = 0.5*nsigma*sigmaR1   #integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
            vTgl= vTmax/2.*(glx+1.)
            vTfac= 0.5 * vTmax  #integration over [0.,vTmax]
            #Tile everything
            vTgl= numpy.tile(vTgl,(ngl,1)).T
            vRgl= numpy.tile(vRgl,(ngl,1))
            vTglw= numpy.tile(glw,(ngl,1)).T #also tile weights
            vRglw= numpy.tile(vRglw,(ngl,1))
            #If inputs are arrays, tile
            if isinstance(R,numpy.ndarray):
                nR= len(R)
                R= numpy.tile(R,(ngl,ngl,1)).T.flatten()
                z= numpy.tile(z,(ngl,ngl,1)).T.flatten()
                vz= numpy.tile(vz,(ngl,ngl,1)).T.flatten()
                vTgl= numpy.tile(vTgl,(nR,1,1)).flatten()
                vRgl= numpy.tile(vRgl,(nR,1,1)).flatten()
                vTglw= numpy.tile(vTglw,(nR,1,1))
                vRglw= numpy.tile(vRglw,(nR,1,1))
                scalarOut= False
            else:
                R= R+numpy.zeros(ngl*ngl)
                z= z+numpy.zeros(ngl*ngl)
                vz= vz+numpy.zeros(ngl*ngl)
                nR= 1
                scalarOut= True
                vRgl= vRgl.flatten()
            vRgl*= numpy.tile(nsigma*sigmaR1/2.,(ngl,ngl,1)).T.flatten()
            #evaluate
            if _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self(R,
                                                                 vRgl.flatten(),
                                                                 vTgl.flatten(),
                                                                 z,
                                                                 vz,
                                                                 log=True,
                                                                 _return_actions=True,
                                                                 _return_freqs=True,
                                                                 use_physical=False)
                logqeval= numpy.reshape(logqeval,(nR,ngl*ngl))
            elif not _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self((_jr,_lz,_jz),
                                           rg=_rg,kappa=_kappa,nu=_nu,
                                           Omega=_Omega,
                                           log=True,
                                           _return_actions=True,
                                           _return_freqs=True,
                                           use_physical=False)
                logqeval= numpy.reshape(logqeval,(nR,ngl*ngl))
            elif not _jr is None and _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self((_jr,_lz,_jz),
                                           log=True,
                                           _return_actions=True,
                                           _return_freqs=True,
                                           use_physical=False)
                logqeval= numpy.reshape(logqeval,(nR,ngl*ngl))
            elif _jr is None and not _rg is None:
                logqeval, jr, lz, jz, rg, kappa, nu, Omega= self(R,
                                             vRgl.flatten(),
                                             vTgl.flatten(),
                                             z,
                                             vz,
                                             rg=_rg,kappa=_kappa,nu=_nu,
                                             Omega=_Omega,
                                             log=True,
                                             _return_actions=True,
                                             _return_freqs=True,
                                             use_physical=False)
                logqeval= numpy.reshape(logqeval,(nR,ngl*ngl))
            vRglw= numpy.reshape(vRglw,(nR,ngl*ngl))
            vTglw= numpy.reshape(vTglw,(nR,ngl*ngl))
            if scalarOut:
                result= numpy.sum(numpy.exp(logqeval)*vTglw*vRglw,axis=1)[0]*vRfac*vTfac
            else:
                result= numpy.sum(numpy.exp(logqeval)*vTglw*vRglw,axis=1)*vRfac*vTfac
            if _return_actions and _return_freqs:
                return (result,
                        jr,lz,jz,
                        rg, kappa, nu, Omega)          
            elif _return_freqs:
                return (result,
                        rg, kappa, nu, Omega)
            elif _return_actions:
                return (result,
                        jr,lz,jz)
            else:
                return result

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity',pop=True)
    def pvRvT(self,vR,vT,R,z,gl=True,ngl=_DEFAULTNGL2,nsigma=4.):
        """
        NAME:

           pvRvT

        PURPOSE:

           calculate the marginalized (vR,vT) probability at this location (NOT normalized by the density)

        INPUT:

           vR - radial velocity (can be Quantity)

           vT - tangential velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration
           
           nsigma - sets integration limits to [-1,+1]*nsigma*sigma_z(R) for integration over vz (default: 4)

        OUTPUT:

           p(vR,vT,R,z)

        HISTORY:

           2013-01-02 - Written - Bovy (IAS)           
           2018-01-12 - Added Gauss-Legendre integration prefactor nsigma/2 - Trick (MPA)

        """
        sigmaz1= self._sz*numpy.exp((self._refr-R)/self._hsz)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vzgl= nsigma*sigmaz1/2.*(glx+1.)
                vzglw= glw
                vzfac= nsigma*sigmaz1  #2 x integration over [0,nsigma*sigmaz1]
            else:
                vzgl= nsigma*sigmaz1/2.*(glx12+1.)
                vzgl= list(vzgl)
                vzgl.extend(-nsigma*sigmaz1/2.*(glx12+1.))
                vzgl= numpy.array(vzgl)
                vzglw= glw12
                vzglw= list(vzglw)
                vzglw.extend(glw12)
                vzglw= numpy.array(vzglw)
                vzfac = 0.5*nsigma*sigmaz1   #integration over [-nsigma*sigmaz1,0] and [0,nsigma*sigmaz1]
            #evaluate
            logqeval= self(R+numpy.zeros(ngl),
                           vR+numpy.zeros(ngl),
                           vT+numpy.zeros(ngl),
                           z+numpy.zeros(ngl),
                           vzgl,
                           log=True,use_physical=False)
            return numpy.sum(numpy.exp(logqeval)*vzglw*vzfac)

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity',pop=True)
    def pvTvz(self,vT,vz,R,z,gl=True,ngl=_DEFAULTNGL2,nsigma=4.):
        """
        NAME:

           pvTvz

        PURPOSE:

           calculate the marginalized (vT,vz) probability at this location (NOT normalized by the density)

        INPUT:

           vT - tangential velocity (can be Quantity)

           vz - vertical velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration
           
           nsigma - sets integration limits to [-1,+1]*nsigma*sigma_R(R) for integration over vR (default: 4)

        OUTPUT:

           p(vT,vz,R,z)

        HISTORY:

           2012-12-22 - Written - Bovy (IAS)           
           2018-01-12 - Added Gauss-Legendre integration prefactor nsigma/2 - Trick (MPA)

        """
        sigmaR1= self._sr*numpy.exp((self._refr-R)/self._hsr)
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            if isinstance(self._aA,(actionAngle.actionAngleAdiabatic,
                                    actionAngle.actionAngleAdiabaticGrid)):
                vRgl= nsigma*sigmaR1/2.*(glx+1.)
                vRglw= glw
                vRfac= nsigma*sigmaR1  #2 x integration over [0,nsigma*sigmaR1]
            else:
                vRgl= nsigma*sigmaR1/2.*(glx12+1.)
                vRgl= list(vRgl)
                vRgl.extend(-nsigma*sigmaR1/2.*(glx12+1.))
                vRgl= numpy.array(vRgl)
                vRglw= glw12
                vRglw= list(vRglw)
                vRglw.extend(glw12)
                vRglw= numpy.array(vRglw)
                vRfac = 0.5*nsigma*sigmaR1   #integration over [-nsigma*sigmaR1,0] and [0,nsigma*sigmaR1]
            #evaluate
            logqeval= self(R+numpy.zeros(ngl),
                           vRgl,
                           vT+numpy.zeros(ngl),
                           z+numpy.zeros(ngl),
                           vz+numpy.zeros(ngl),
                           log=True,use_physical=False)
            return numpy.sum(numpy.exp(logqeval)*vRglw*vRfac)

    @actionAngle_physical_input
    @physical_conversion('phasespacedensityvelocity',pop=True)
    def pvRvz(self,vR,vz,R,z,gl=True,ngl=_DEFAULTNGL2,vTmax=1.5):
        """
        NAME:

           pvR

        PURPOSE:

           calculate the marginalized (vR,vz) probability at this location (NOT normalized by the density)

        INPUT:

           vR - radial velocity (can be Quantity)

           vz - vertical velocity (can be Quantity)

           R - radius (can be Quantity)

           z - height (can be Quantity)

           gl - use Gauss-Legendre integration (True, currently the only option)

           ngl - order of Gauss-Legendre integration
           
           vTmax - sets integration limits to [0,vTmax] for integration over vT (default: 1.5)

        OUTPUT:

           p(vR,vz,R,z)

        HISTORY:

           2013-01-02 - Written - Bovy (IAS)
           2018-01-12 - Added Gauss-Legendre integration prefactor vTmax/2 - Trick (MPA)

        """
        if gl:
            if ngl % 2 == 1:
                raise ValueError("ngl must be even")
            #Use Gauss-Legendre integration for all
            if ngl == _DEFAULTNGL:
                glx, glw= self._glxdef, self._glwdef
                glx12, glw12= self._glxdef12, self._glwdef12
            elif ngl == _DEFAULTNGL2:
                glx, glw= self._glxdef2, self._glwdef2
                glx12, glw12= self._glxdef, self._glwdef
            else:
                glx, glw= numpy.polynomial.legendre.leggauss(ngl)
                glx12, glw12= numpy.polynomial.legendre.leggauss(ngl//2)
            #Evaluate everywhere
            vTgl= vTmax/2.*(glx+1.)
            vTglw= glw
            vTfac= 0.5 * vTmax  #integration over [0.,vTmax]
            #If inputs are arrays, tile
            if isinstance(R,numpy.ndarray):
                nR= len(R)
                R= numpy.tile(R,(ngl,1)).T.flatten()
                z= numpy.tile(z,(ngl,1)).T.flatten()
                vR= numpy.tile(vR,(ngl,1)).T.flatten()
                vz= numpy.tile(vz,(ngl,1)).T.flatten()
                vTgl= numpy.tile(vTgl,(nR,1)).flatten()
                vTglw= numpy.tile(vTglw,(nR,1))
                scalarOut= False
            else:
                R= R+numpy.zeros(ngl)
                vR= vR+numpy.zeros(ngl)
                z= z+numpy.zeros(ngl)
                vz= vz+numpy.zeros(ngl)
                nR= 1
                scalarOut= True
            #evaluate
            logqeval= numpy.reshape(self(R,
                                         vR,
                                         vTgl,
                                         z,
                                         vz,
                                         log=True,
                                         use_physical=False),
                                    (nR,ngl))
            out= numpy.sum(numpy.exp(logqeval)*vTglw*vTfac,axis=1)
            if scalarOut: return out[0]
            else: return out

    def _calc_epifreq(self,r):
        """
        NAME:
           _calc_epifreq
        PURPOSE:
           calculate the epicycle frequency at r
        INPUT:
           r - radius
        OUTPUT:
           kappa
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.1 ms for a Miyamoto-Nagai potential
        """
        return potential.epifreq(self._pot,r)

    def _calc_verticalfreq(self,r):
        """
        NAME:
           _calc_verticalfreq
        PURPOSE:
           calculate the vertical frequency at r
        INPUT:
           r - radius
        OUTPUT:
           nu
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           takes about 0.05 ms for a Miyamoto-Nagai potential
        """
        return potential.verticalfreq(self._pot,r)

    def _rg(self,lz):
        """
        NAME:
           _rg
        PURPOSE:
           calculate the radius of a circular orbit of Lz
        INPUT:
           lz - Angular momentum
        OUTPUT:
           radius
        HISTORY:
           2012-07-25 - Written - Bovy (IAS@MPIA)
        NOTE:
           seems to take about ~0.5 ms for a Miyamoto-Nagai potential; 
           ~0.75 ms for a MWPotential
           about the same with or without interpolation of the rotation curve

           Not sure what to do about negative lz...
        """
        if isinstance(lz,numpy.ndarray):
            indx= (lz > self._precomputergLzmax)*(lz < self._precomputergLzmin)
            indxc= True^indx
            out= numpy.empty(lz.shape)
            out[indxc]= self._rgInterp(lz[indxc])
            out[indx]= numpy.array([potential.rl(self._pot,lz[indx][ii]) for ii in range(numpy.sum(indx))])
            return out
        else:
            if lz > self._precomputergLzmax or lz < self._precomputergLzmin:
                return potential.rl(self._pot,lz)
            return numpy.atleast_1d(self._rgInterp(lz))

def _vmomentsurfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,n,m,o): #pragma: no cover because this is too slow; a warning is shown
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return vR**n*vT**m*vz**o*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,
                                use_physical=False)

def _vmomentsurfaceMCIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,mvT,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return vR**n*vT**m*vz**o*df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,use_physical=False)*numpy.exp(vR**2./2.+(vT-mvT)**2./2.+vz**2./2.)

def _jmomentsurfaceIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,n,m,o): #pragma: no cover because this is too slow; a warning is shown
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,use_physical=False,
              func= (lambda x,y,z: x**n*y**m*z**o))

def _jmomentsurfaceMCIntegrand(vz,vR,vT,R,z,df,sigmaR1,gamma,sigmaz1,mvT,n,m,o):
    """Internal function that is the integrand for the vmomentsurface mass integration"""
    return df(R,vR*sigmaR1,vT*sigmaR1*gamma,z,vz*sigmaz1,use_physical=False,
              func=(lambda x,y,z: x**n*y**m*z**o))\
              *numpy.exp(vR**2./2.+(vT-mvT)**2./2.+vz**2./2.)
