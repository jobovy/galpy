# Class that implements isotropic spherical DFs computed using the Eddington
# formula
import numpy
from scipy import interpolate, integrate
from ..util import conversion
from ..potential import evaluateR2derivs
from ..potential.Potential import _evaluatePotentials, _evaluateRforces
from .sphericaldf import isotropicsphericaldf, sphericaldf

class eddingtondf(isotropicsphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula"""
    def __init__(self,pot=None,denspot=None,rmax=1e4,
                 scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize an isotropic distribution function computed using the Eddington inversion

        INPUT:

           pot= (None) Potential instance or list thereof that represents the gravitational potential (assumed to be spherical)

           denspot= (None) Potential instance or list thereof that represent the density of the tracers (assumed to be spherical; if None, set equal to pot)

           rmax= (1e4) when sampling, maximum radius to consider (can be Quantity)

           ro=, vo= galpy unit parameters

        OUTPUT:
        
            None

        HISTORY:

            2021-02-04 - Written - Bovy (UofT)

        """
        isotropicsphericaldf.__init__(self,pot=pot,denspot=denspot,rmax=rmax,
                                      scale=scale,ro=ro,vo=vo)
        self._dnudr= self._denspot._ddensdr \
            if not isinstance(self._denspot,list) \
            else lambda r: numpy.sum([p._ddensdr(r) for p in self._denspot])
        self._d2nudr2= self._denspot._d2densdr2 \
            if not isinstance(self._denspot,list) \
            else lambda r: numpy.sum([p._d2densdr2(r) for p in self._denspot])
        self._potInf= _evaluatePotentials(pot,self._rmax,0)
        self._Emin= _evaluatePotentials(pot,0,0)
        # Build interpolator r(pot)
        r_a_values= numpy.concatenate(\
                        (numpy.array([0.]),
                         numpy.geomspace(1e-6,1e6,10001)))
        self._rphi= interpolate.InterpolatedUnivariateSpline(\
                        [_evaluatePotentials(self._pot,r*self._scale,0)
                         for r in r_a_values],r_a_values*self._scale,k=3)
        
    def sample(self,R=None,z=None,phi=None,n=1,return_orbit=True):
        # Slight over-write of superclass method to first build f(E) interp
        # No docstring so superclass' is used
        if not hasattr(self,'_fE_interp'):
            Es4interp= numpy.hstack((numpy.linspace(0,0.5,51,endpoint=False),
                                     sorted(1.-numpy.geomspace(1e-4,0.5,51))))
            Es4interp= (Es4interp*(self._Emin-self._potInf)+self._potInf)[::-1]
            fE4interp= self.fE(Es4interp)
            iindx= True^numpy.isnan(fE4interp)
            self._fE_interp= interpolate.InterpolatedUnivariateSpline(\
                                        Es4interp[iindx],fE4interp[iindx],k=3)
        return sphericaldf.sample(self,R=R,z=z,phi=phi,n=n,
                                  return_orbit=return_orbit)

    def fE(self,E):
        """
        NAME:

            fE

        PURPOSE

            Calculate the energy portion of a DF computed using the Eddington inversion

        INPUT:

            E - The energy (can be Quantity)

        OUTPUT:

            fE - The value of the energy portion of the DF

        HISTORY:

            2021-02-04 - Written - Bovy (UofT)
        """
        Eint= conversion.parse_energy(E,vo=self._vo)
        out= numpy.zeros_like(Eint)
        indx= (Eint < self._potInf)*(Eint >= self._Emin)
        # Split integral at twice the lower limit to deal with divergence at
        # the lower end and infinity at the upper end
        out[indx]= numpy.array([integrate.quad(
            lambda t: _fEintegrand_smallr(t,self._pot,tE,
                                          self._dnudr,self._d2nudr2,
                                          self._rphi(tE)),
            0.,numpy.sqrt(self._rphi(tE)),
            points=[0.])[0] for tE in Eint[indx]])
        out[indx]+= numpy.array([integrate.quad(
            lambda t: _fEintegrand_larger(t,self._pot,tE,
                                          self._dnudr,self._d2nudr2),
            0.,0.5/self._rphi(tE))[0] for tE in Eint[indx]])
        return -out/(numpy.sqrt(8.)*numpy.pi**2.)
  
def _fEintegrand_raw(r,pot,E,dnudr,d2nudr2):
    # The 'raw', i.e., direct integrand in the Eddington inversion
    Fr= _evaluateRforces(pot,r,0)
    return (Fr*d2nudr2(r)
            +dnudr(r)*evaluateR2derivs(pot,r,0,use_physical=False))\
            /Fr**2.\
            /numpy.sqrt(_evaluatePotentials(pot,r,0)-E)

def _fEintegrand_smallr(t,pot,E,dnudr,d2nudr2,rmin):
    # The integrand at small r, using transformation to deal with sqrt diverge
    return 2.*t*_fEintegrand_raw(t**2.+rmin,pot,E,dnudr,d2nudr2)

def _fEintegrand_larger(t,pot,E,dnudr,d2nudr2):
    # The integrand at large r, using transformation to deal with infinity
    return 1./t**2*_fEintegrand_raw(1./t,pot,E,dnudr,d2nudr2)

