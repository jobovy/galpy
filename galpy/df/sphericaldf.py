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
    def __init__(self,pot=None,ro=None,vo=None):
        """
        NAME:

            __init__

        PURPOSE:

            Initializes a spherical DF

        INPUT:

            pot - Spherical potential which determines the DF

        OUTPUT:

            None

        HISTORY:

            2020-07-22 - Written - 

        """
        df.__init__(self,ro=ro,vo=vo)
        if pot is None:
            raise IOError("pot= must be set")
        # Some sort of check for spherical symmetry in the potential?
        self._pot = flatten_potential(pot)

############################## EVALUATING THE DF###############################
    @physical_conversion('phasespacedensity',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

            __call__

        PURPOSE:

            return the DF

        INPUT:

            Either:

                a) (E,L,Lz): tuple of E and (optionally) L and (optionally) Lz.
                    Each may be Quantity
                    
                b) R,vR,vT,z,vz,phi: 

                c) Orbit instance: orbit.Orbit instance and if specific time 
                    then orbit.Orbit(t) 

        KWARGS:

            return_fE= if True then return the full distribution function plus 
                just the energy component (for e.g. an anisotropic DF)

        OUTPUT:

            Value of DF

        HISTORY:

            2020-07-22 - Written -

        """
        # Stub for calling the DF as a function of either a) R,vR,vT,z,vz,phi,
        # b) Orbit, c) E,L (Lz?) --> maybe depends on the actual form?
        
        # Get E,L,Lz. Generic requirements for any possible spherical DF?
        if len(args) == 1:
            if not isinstance(args[0],Orbit): # Assume tuple (E,L,Lz)
                if len(args[0]) == 1:
                    E = args[0][0]
                    L,Lz = None,None
                elif len(args[0]) == 2:
                    E,L = args[0]
                    Lz = None
                elif len(args[0]) == 3:
                    E,L,Lz = args[0]
            else: # Orbit
                E = args[0].E(pot=self._pot)
                L = numpy.sqrt(numpy.sum(numpy.square(args[0].L())))
                Lz = args[0].Lz()
            if _APY_LOADED and isinstance(E,units.Quantity):
                E.to(units.km**2/units.s**2).value/self._vo**2.
            if _APY_LOADED and isinstance(L,units.Quantity):
                L.to(units.kpc*units.km/units.s).value/self._ro/self._vo
            if _APY_LOADED and isinstance(Lz,units.Quantity):
                Lz.to(units.kpc*units.km/units.s).value/self._ro/self._vo
        else: # Assume R,vR,vT,z,vz,(phi)
            if len(args) == 5:
                R,vR,vT,z,vz = args
                phi = None
            else:
                R,vR,vT,z,vz,phi = args
            if _APY_LOADED and isinstance(R,units.Quantity):
                R.to(units.kpc).value/self._ro
            if _APY_LOADED and isinstance(vR,units.Quantity):
                vR.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(vT,units.Quantity):
                vT.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(z,units.Quantity):
                z.to(units.kpc).value/self._ro
            if _APY_LOADED and isinstance(vz,units.Quantity):
                vz.to(units.km/units.s).value/self._vo
            if _APY_LOADED and isinstance(phi,units.Quantity):
                phi.to(units.rad).value
            vtotSq = vR**2.+vT**2.+vz**2.
            E = 0.5*vtotSq + evaluatePotentials(R,z)
            Lz = R*vT
            r = numpy.sqrt(R**2.+z**2.)
            vrad = (R*vR+z*vz)/r
            L = numpy.sqrt(vtotSq-vrad**2.)*r
        f = self.__call_internal__(E,L,Lz) # Some function for each sub-class
        return f

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