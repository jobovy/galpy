# Class that implements isotropic spherical DFs computed using the Eddington
# formula
from ..potential import evaluatePotentials
from .sphericaldf import isotropicsphericaldf, _APY_LOADED
if _APY_LOADED:
    from astropy import units

class Eddingtondf(isotropicsphericaldf):
    """Class that implements isotropic spherical DFs computed using the Eddington formula"""
    def __init__(self,pot=None,scale=None,ro=None,vo=None):
        """
            scale - Characteristic scale radius to aid sampling calculations. 
                Not necessary, and will also be overridden by value from pot if 
                available.
        """
        isotropicsphericaldf.__init__(self,ro=ro,vo=vo)
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
                    scale= scale.to(units.kpc).value/self._ro
                self._scale = scale
            else:
                self._scale = 1.
        self._xi_cmf_interpolator = self._make_cmf_interpolator()
        self._v_vesc_pvr_interpolator = self._make_pvr_interpolator()

    def _call_internal(self,*args):
        # Stub for calling
        return None

    def fE(self,E):
        # Stub for computing f(E)
        return None
