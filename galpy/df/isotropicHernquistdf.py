# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
from .sphericaldf import sphericaldf
from .Eddingtondf import Eddingtondf

class isotropicHernquistdf(Eddingtondf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""
    def __init__(self,ro=None,vo=None):
        # Initialize using sphericaldf rather than Eddingtondf, because
        # Eddingtondf will have code specific to computing the Eddington
        # integral, which is not necessary for Hernquist
        sphericaldf.__init__(self,ro=ro,vo=vo)

    def fE(self,E):
        # Stub for computing f(E)
        return None

    
