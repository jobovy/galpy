# Class that implements isotropic spherical DFs computed using the Eddington
# formula
from .sphericaldf import sphericaldf
import numpy

class Eddingtondf(sphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula"""
    def __init__(self,pot=None,ro=None,vo=None):
        sphericaldf.__init__(self,pot=pot,ro=ro,vo=vo)

    def _call_internal(self,*args):
        # Stub for calling
        return None

    def fE(self,E):
        # Stub for computing f(E)
        return None

    def _sample_eta(self,n=1):
        """Sample the angle eta which defines radial vs tangential velocities"""
        return numpy.arccos(1.-2.*numpy.random.uniform(size=n))
