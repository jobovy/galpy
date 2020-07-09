# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
from .constantbetadf import constantbetadf

class constantbetaHernquistdf(constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""
    def __init__(self,ro=None,vo=None):
        constantbetadf.__init__(self,ro=ro,vo=vo)

    def f1E(self,E):
        # Stub for computing f_1(E)
        return None

    
