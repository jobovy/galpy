# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter
import numpy
from scipy import interpolate, integrate, special
from ..util import conversion
from ..potential.Potential import _evaluatePotentials
from .sphericaldf import anisotropicsphericaldf, sphericaldf

# This is the general constantbeta superclass, implementation of general
# formula can be found following this class
class _constantbetadf(anisotropicsphericaldf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter"""
    def __init__(self,pot=None,denspot=None,beta=None,rmax=None,
                 scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a spherical DF with constant anisotropy parameter

        INPUT:

            pot - Spherical potential which determines the DF

           denspot= (None) Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot)

           rmax= (None) maximum radius to consider (can be Quantity); DF is cut off at E = Phi(rmax)

            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot if 
                available.

        """
        anisotropicsphericaldf.__init__(self,pot=pot,denspot=denspot,
                                        rmax=rmax,scale=scale,ro=ro,vo=vo)
        self._beta= beta

    def _call_internal(self,*args):
        """
        NAME:

            _call_internal

        PURPOSE:

            Evaluate the DF for a constant anisotropy Hernquist

        INPUT:

            E - The energy

            L - The angular momentum

        OUTPUT:

            fH - The value of the DF

        HISTORY:

            2020-07-22 - Written - Lane (UofT)
        """
        E, L, _= args
        return L**(-2*self._beta)*self.fE(E)

    def _sample_eta(self,r,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        if not hasattr(self,'_coseta_icmf_interp'):
            # Cumulative dist for cos(eta) =
            # 0.5 + x 2F1(0.5,beta,1.5,x^2)/sqrt(pi)/Gamma(1-beta)*Gamma(1.5-beta)
            cosetas= numpy.linspace(-1.,1.,20001)
            coseta_cmf= cosetas*special.hyp2f1(0.5,self._beta,1.5,cosetas**2.)\
                /numpy.sqrt(numpy.pi)/special.gamma(1.-self._beta)\
                *special.gamma(1.5-self._beta)+0.5
            self._coseta_icmf_interp= interpolate.interp1d(\
                                coseta_cmf,cosetas,
                                bounds_error=False,fill_value='extrapolate')
        return numpy.arccos(self._coseta_icmf_interp(\
                                                numpy.random.uniform(size=n)))

    def _p_v_at_r(self,v,r):
        if hasattr(self,'_fE_interp'):
            return self._fE_interp(_evaluatePotentials(self._pot,r,0)\
                                   +0.5*v**2.)*v**(2.-2.*self._beta)
        else:
            return self.fE(_evaluatePotentials(self._pot,r,0)\
                           +0.5*v**2.)*v**(2.-2.*self._beta)
    
    def _vmomentdensity(self,r,n,m):
         if m%2 == 1 or n%2 == 1:
             return 0.
         return 2.*numpy.pi*r**(-2.*self._beta)\
             *integrate.quad(lambda v: v**(2.-2.*self._beta+m+n)
                             *self.fE(_evaluatePotentials(self._pot,r,0)
                                      +0.5*v**2.),
                             0.,self._vmax_at_r(self._pot,r))[0]\
            *special.gamma(m/2.-self._beta+1.)*special.gamma((n+1)/2.)/\
            special.gamma(0.5*(m+n-2.*self._beta+3.))

class constantbetadf(_constantbetadf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter for a given density profile"""
    def __init__(self,pot=None,denspot=None,beta=0.,twobeta=None,
                 rmax=None,scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a spherical DF with constant anisotropy parameter

        INPUT:

            pot= (None) Potential instance or list thereof

            denspot= (None) Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot)

           beta= (0.) anisotropy parameter

           twobeta= (None) twice the anisotropy parameter (useful for \beta = half-integer, which is a special case); has priority over beta

           rmax= (None) maximum radius to consider (can be Quantity); DF is cut off at E = Phi(rmax)

           scale - Characteristic scale radius to aid sampling calculations. Optionaland will also be overridden by value from pot if available.

           ro=, vo= galpy unit parameters

        OUTPUT:

            None

        HISTORY:

            2021-02-14 - Written - Bovy (UofT)

        """
        try:
            from jax import grad, vmap
        except ImportError: # pragma: no cover
            raise ImportError("galpy.df.constantbetadf requires the google/jax library")
        # Parse twobeta
        if not twobeta is None:
            beta= twobeta/2.
        else:
            twobeta= 2.*beta
        _constantbetadf.__init__(self,pot=pot,denspot=denspot,beta=beta,
                                 rmax=rmax,scale=scale,ro=ro,vo=vo)
        self._twobeta= twobeta
        self._halfint= False
        if isinstance(self._twobeta,int) and self._twobeta % 2 == 1:
            self._halfint= True
            self._m= (3-self._twobeta)//2
            ii= self._m-1
            # Compute d^m (dens x r^2beta) / d Psi^m as successive
            # d / dr ( ...) / F_r
            func= lambda r: self._denspot._ddenstwobetadr(r,beta=self._beta)\
                    /self._pot._rforce_jax(r)
            while ii > 0:
                func= lambda r,func=func: grad(func)(r)\
                    /self._pot._rforce_jax(r)
                ii-= 1
        else:
            self._m= int(numpy.floor(1.5-self._beta))
            self._alpha= 1.5-self._beta-self._m
            self._fE_prefactor= 2.**self._beta/(2.*numpy.pi)**1.5\
                /special.gamma(1.-self._alpha)/special.gamma(1.-self._beta)
            ii= self._m
            # Similar d^m (dens x r^2beta) / d Psi^m as above,
            # but because integral necessary now is over psi, we can omit
            # the final 1/Fr to do the integral over r
            if ii == 0:
                func= lambda r: self._denspot._ddenstwobetadr(r,
                                                              beta=self._beta)
            else:
                func= lambda r: self._denspot._ddenstwobetadr(r,
                                                              beta=self._beta)\
                                                /self._pot._rforce_jax(r)
            while ii > 0:
                if ii == 1:
                    func= lambda r,func=func: grad(func)(r)
                else:
                    func= lambda r,func=func: grad(func)(r)\
                        /self._pot._rforce_jax(r)
                ii-= 1
        self._gradfunc= vmap(func)
        # Min and max energy
        self._potInf= _evaluatePotentials(self._pot,self._rmax,0)
        self._Emin= _evaluatePotentials(self._pot,0.,0)
        # Build interpolator r(pot)
        r_a_values= numpy.concatenate(\
                        (numpy.array([0.]),
                         numpy.geomspace(1e-6,1e6,10001)))
        self._rphi= interpolate.InterpolatedUnivariateSpline(\
                        [_evaluatePotentials(self._pot,r*self._scale,0)
                         for r in r_a_values],r_a_values*self._scale,k=3)
        # Build interpolator for the lower limit of the integration (near the
        # 1/(Phi-E)^alpha divergence; at the end, we slightly adjust it up
        # to be sure to be above the point where things go haywire...
        if not self._halfint:
            Es= numpy.linspace(self._Emin,
                               self._potInf+1e-3*(self._Emin-self._potInf),51)
            guesspow= -17
            guesst= 10.**(guesspow*(1.-self._alpha))
            startt= numpy.ones_like(Es)*guesst
            startval= numpy.zeros_like(Es)
            while numpy.any(startval == 0.):
                guesspow+= 1
                guesst= 10.**(guesspow*(1.-self._alpha))
                indx= startval == 0.
                startt[indx]= guesst
                startval[indx]= _fEintegrand_smallr(startt[indx],
                                                    self._pot,Es[indx],
                                                    self._gradfunc,
                                                    self._alpha,
                                                    self._rphi(Es[indx]))
            self._logstartt= interpolate.InterpolatedUnivariateSpline(\
                            Es,numpy.log10(startt)+10./3.*(1.-self._alpha),k=3)
        
    def sample(self,R=None,z=None,phi=None,n=1,return_orbit=True,rmin=0.):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        if not hasattr(self,'_fE_interp'):
            Es4interp= numpy.hstack((numpy.geomspace(1e-8,0.5,101,
                                                     endpoint=False),
                                     sorted(1.-numpy.geomspace(1e-4,0.5,101))))
            Es4interp= (Es4interp*(self._Emin-self._potInf)+self._potInf)[::-1]
            fE4interp= self.fE(Es4interp)
            iindx= numpy.isfinite(fE4interp)
            self._fE_interp= interpolate.InterpolatedUnivariateSpline(\
                                        Es4interp[iindx],fE4interp[iindx],
                                        k=3,ext=3)
        return sphericaldf.sample(self,R=R,z=z,phi=phi,n=n,
                                  return_orbit=return_orbit,rmin=rmin)

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of a constant-beta distribution function

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2021-02-14 - Written - Bovy (UofT)
        """
        Eint= numpy.atleast_1d(conversion.parse_energy(E,vo=self._vo))
        out= numpy.zeros_like(Eint)
        indx= (Eint < self._potInf)*(Eint >= self._Emin)
        if self._halfint:
            # fE is simply given by the relevant derivative
            out[indx]= self._gradfunc(self._rphi(Eint[indx]))
            return out\
           /(2.*numpy.pi**1.5*2**(0.5-self._beta)*special.gamma(1.-self._beta))
        else:
            # Now need to integrate to get fE
            # Split integral at twice the lower limit to deal with divergence
            # at the lower end and infinity at the upper end
            out[indx]= numpy.array([integrate.quadrature(
                lambda t: _fEintegrand_smallr(t,self._pot,tE,self._gradfunc,
                                              self._alpha,self._rphi(tE)),
                10.**self._logstartt(tE),self._rphi(tE)**(1.-self._alpha))[0]
                                    for tE in Eint[indx]])
            # Add constant part at the beginning
            out[indx]+= 10.**self._logstartt(Eint[indx])\
                *_fEintegrand_smallr(10.**self._logstartt(Eint[indx]),
                                     self._pot,Eint[indx],
                                     self._gradfunc,
                                     self._alpha,
                                     self._rphi(Eint[indx]))
            # 2nd half of the integral
            out[indx]+= numpy.array([integrate.quadrature(
                lambda t: _fEintegrand_larger(t,self._pot,tE,self._gradfunc,
                                              self._alpha),
                0.,0.5/self._rphi(tE))[0] for tE in Eint[indx]])
            return -out*self._fE_prefactor
  
def _fEintegrand_raw(r,pot,E,dmp1nudrmp1,alpha):
    # The 'raw', i.e., direct integrand in the constant-beta inversion
    out= numpy.zeros_like(r) # Avoid JAX item assignment issues
    #print("r",r,dmp1nudrmp1(r),(_evaluatePotentials(pot,r,0)-E))
    out[:]= dmp1nudrmp1(r)/(_evaluatePotentials(pot,r,0)-E)**alpha
    out[True^numpy.isfinite(out)]= 0. # assume these are where denom is slightly neg.
    return out

def _fEintegrand_smallr(t,pot,E,dmp1nudrmp1,alpha,rmin):
    # The integrand at small r, using transformation to deal with divergence
    #print("t",t,rmin,t**(1./(1.-alpha))+rmin)
    return 1./(1.-alpha)*t**(alpha/(1.-alpha))\
        *_fEintegrand_raw(t**(1./(1.-alpha))+rmin,pot,E,dmp1nudrmp1,alpha)

def _fEintegrand_larger(t,pot,E,dmp1nudrmp1,alpha):
    # The integrand at large r, using transformation to deal with infinity
    return 1./t**2*_fEintegrand_raw(1./t,pot,E,dmp1nudrmp1,alpha)

