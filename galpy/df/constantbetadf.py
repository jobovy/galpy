# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter
from .sphericaldf import anisotropicsphericaldf

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
        self.beta = beta
        anisotropicsphericaldf.__init__(self,pot=pot,dftype='constant',
            ro=ro,vo=vo)

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
