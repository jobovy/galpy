# Class that implements isotropic spherical DFs computed using the Eddington
# formula
from ..potential import evaluatePotentials
from .sphericaldf import isotropicsphericaldf

class eddingtondf(isotropicsphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula"""
    def __init__(self,pot=None,scale=None,ro=None,vo=None):
        """
            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot if 
                available.
        """
        isotropicsphericaldf.__init__(self,pot=pot,scale=scale,ro=ro,vo=vo)
        self._potInf= evaluatePotentials(pot,10**12,0)
