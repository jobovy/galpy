# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter
from .sphericaldf import anisotropicsphericaldf
import numpy
import scipy.interpolate

class constantbetadf(anisotropicsphericaldf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter"""
    def __init__(self,pot=None,beta=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initialize a spherical DF with constant anisotropy parameter

        INPUT:

            pot - Spherical potential which determines the DF
        """
        anisotropicsphericaldf.__init__(self,ro=ro,vo=vo)
        self.beta = beta
        if pot is None:
            raise IOError("pot= must be set")
        # Some sort of check for spherical symmetry in the potential?
        assert not isinstance(pot,(list,tuple)), 'Lists of potentials not yet supported'
        self._pot = pot
        self._potInf = evaluatePotentials(pot,10**12,0)
        try:
            self._scale = pot._scale
        except AttributeError:
            if scale is not None:
                if _APY_LOADED and isinstance(scale,units.Quantity):
                    scale= scale.to(u.kpc).value/self._ro
                self._scale = scale
            else:
                self._scale = 1.
        self._xi_cmf_interpolator = self._make_cmf_interpolator()
        self._v_vesc_pvr_interpolator = self._make_pvr_interpolator()

    def _call_internal(self,*args):
        # Stub for calling
        return None

    def fE(self,E):
        # Stub for computing f_1(E) in BT08 nomenclature
        return None

    def _sample_eta(self,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        deta = 0.00005*numpy.pi
        etas = (numpy.arange(0, numpy.pi, deta)+deta/2)
        eta_pdf_cml = numpy.cumsum(numpy.power(numpy.sin(etas),1.-2.*self.beta))
        eta_pdf_cml_norm = eta_pdf_cml / eta_pdf_cml[-1]
        eta_icml_interp = scipy.interpolate.interp1d(eta_pdf_cml_norm, etas, 
            bounds_error=False, fill_value='extrapolate')
        eta_samples = eta_icml_interp(numpy.random.uniform(size=n))
        return eta_samples
