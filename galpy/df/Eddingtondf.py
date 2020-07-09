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

    def _sample_eta(self):
        # Stub for function that samples eta
        return None
        
