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
        anisotropicsphericaldf.__init__(self,pot=pot,dftype='constant',
            ro=ro,vo=vo)
        self.beta = beta

    def f1E(self,E):
        # Stub for computing f_1(E) in BT08 nomenclature
        return None
