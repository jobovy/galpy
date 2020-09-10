# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter
import numpy
import scipy.interpolate
from scipy import integrate, special
from ..potential import evaluatePotentials
from .sphericaldf import anisotropicsphericaldf

class constantbetadf(anisotropicsphericaldf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter"""
    def __init__(self,pot=None,beta=None,scale=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a spherical DF with constant anisotropy parameter

        INPUT:

            pot - Spherical potential which determines the DF

            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot if 
                available.

        """
        anisotropicsphericaldf.__init__(self,pot=pot,scale=scale,ro=ro,vo=vo)
        self._beta= beta
        self._potInf= evaluatePotentials(pot,10**12,0)

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

    def _sample_eta(self,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        deta = 0.00005*numpy.pi
        etas = (numpy.arange(0, numpy.pi, deta)+deta/2)
        eta_pdf_cml = numpy.cumsum(numpy.power(numpy.sin(etas),1.-2.*self._beta))
        eta_pdf_cml_norm = eta_pdf_cml / eta_pdf_cml[-1]
        eta_icml_interp = scipy.interpolate.interp1d(eta_pdf_cml_norm, etas, 
            bounds_error=False, fill_value='extrapolate')
        eta_samples = eta_icml_interp(numpy.random.uniform(size=n))
        return eta_samples

    def _p_v_at_r(self,v,r):
        return self.fE(evaluatePotentials(self._pot,r,0,use_physical=False)\
                       +0.5*v**2.)*v**(2.-2.*self._beta)
    
    def vmomentdensity(self,r,n,m):
         """
        NAME:

           vmomentdensity

        PURPOSE:

           calculate the an arbitrary moment of the velocity distribution 
           at r times the density

        INPUT:

           r - spherical radius at which to calculate the moment

           n - vr^n, where vr = v x cos eta

           m - vt^m, where vt = v x sin eta

        OUTPUT:

           <vr^n vt^m x density> at r (no support for units)

        HISTORY:
         
            2020-09-04 - Written - Bovy (UofT)
         """
         if m%2 == 1 or n%2 == 1:
             return 0.
         return 2.*numpy.pi\
             *integrate.quad(lambda v: v**(2.-2.*self._beta+m+n)
                             *self.fE(evaluatePotentials(self._pot,r,0,
                                                         use_physical=False)
                                      +0.5*v**2.),
                             0.,self._vmax_at_r(self._pot,r))[0]\
            *special.gamma(m/2.-self._beta+1.)*special.gamma((n+1)/2.)/\
               2./special.gamma(0.5*(m+n-2.*self._beta+3.))
