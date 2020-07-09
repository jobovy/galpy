# Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant
# beta anisotropy parameter
from .sphericaldf import anisotropicsphericaldf

class constantbetadf(anisotropicsphericaldf):
    """Class that implements DFs of the form f(E,L) = L^{-2\beta} f(E) with constant beta anisotropy parameter"""
    def __init__(self,ro=None,vo=None):
        anisotropicsphericaldf.__init__(self,ro=ro,vo=vo)

    def f1E(self,E):
        # Stub for computing f_1(E) in BT08 nomenclature
        return None
