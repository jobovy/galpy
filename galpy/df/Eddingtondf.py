# Class that implements isotropic spherical DFs computed using the Eddington
# formula
from .sphericaldf import sphericaldf

class Eddingtondf(sphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula"""
    def __init__(self,ro=None,vo=None):
        sphericaldf.__init__(self,ro=ro,vo=vo)

    def fE(self,E):
        # Stub for computing f(E)
        return None

    def _sample_eta(self,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        deta = 0.00005*numpy.pi
        etas = (numpy.arange(0, numpy.pi, deta)+deta/2)
        eta_pdf_cml = numpy.cumsum(numpy.sin(etas))
        eta_pdf_cml_norm = eta_pdf_cml / eta_pdf_cml[-1]
        eta_icml_interp = scipy.interpolate.interp1d(eta_pdf_cml_norm, etas, 
            bounds_error=False, fill_value='extrapolate')
        eta_samples = eta_icml_interp(numpy.random.uniform(size=n))
        return eta_samples
        
