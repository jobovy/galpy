# Superclass for spherical distribution functions, contains
#   - sphericaldf: superclass of all spherical DFs
#   - anisotropicsphericaldf: superclass of all anisotropic spherical DFs
import numpy
import pdb
import scipy.interpolate
from .df import df, _APY_LOADED
from ..potential import flatten as flatten_potential
from ..potential import evaluatePotentials
from ..orbit import Orbit
from ..util.bovy_conversion import physical_conversion
if _APY_LOADED:
    from astropy import units

class sphericaldf(df):
    """Superclass for spherical distribution functions"""
    def __init__(self,ro=None,vo=None):
        df.__init__(self,ro=ro,vo=vo)

############################## EVALUATING THE DF###############################
    def __call__(self,*args,**kwargs):
        # Stub for calling the DF as a function of either a) R,vR,vT,z,vz,phi,
        # b) Orbit, c) E,L (Lz?) --> maybe depends on the actual form?
        return None
        
############################### SAMPLING THE DF################################
    def sample(self):
        # Stub for main sampling function, which will return (x,v) or Orbits...
        return None
        
    def _sample_r(self,n=1):
        # Stub for sampling the radius from M(<r)
        return None

    def _sample_position_angles(self,n=1):
        # Stub for sampling the spherical angles
        return None

    def _sample_v(self,r,n=1):
        # Stub for sampling the magnitude of the velocity at a given r
        # Uses methods for defining how the mag of the velocity is sampled
        # defined in subclasses
        return None

    def _sample_velocity_angles(self,r,n=1):
        # Stub for sampling the angles eta and psi for the velocities
        # Uses _sample_eta implemented in subclasses
        return None

class anisotropicsphericaldf(sphericaldf):
    """Superclass for anisotropic spherical distribution functions"""
    def __init__(self,ro=None,vo=None):
        sphericaldf.__init__(self,ro=ro,vo=vo)
