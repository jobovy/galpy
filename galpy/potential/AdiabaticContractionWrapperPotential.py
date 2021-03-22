###############################################################################
#   AdiabaticContractionWrapperPotential.py: Wrapper to adiabatically
#                                            contract a DM halo in response
#                                            to the growth of a baryonic
#                                            component
###############################################################################
import numpy
from scipy import integrate
from .Force import Force
from .interpSphericalPotential import interpSphericalPotential
from ..util import conversion
# Note: not actually implemented as a WrapperPotential!
class AdiabaticContractionWrapperPotential(interpSphericalPotential):
    """AdiabaticContractionWrapperPotential: Wrapper to adiabatically contract a DM halo in response to the growth of a baryonic component.
    """
    def __init__(self,amp=1.,pot=None,baryonpot=None,
                 method='cautun',f_bar=None,rmin=None,rmax=50.,
                 ro=None,vo=None):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a AdiabaticContractionWrapper Potential

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof representing the density that is adiabatically contracted

           baryonpot - Potential instance or list thereof representing the density of baryons whose growth causes the contraction

           method= ('cautun') Type of adiabatic-contraction formula: 'cautun' for that from Cautun et al. 2020 (2020MNRAS.494.4291C)

           f_bar= (None) baryon fraction; if None, computed from pot and baryonpot

           rmin= (None) minimum radius to consider (default: rmax/2500)

           rmax= (50.) maximum radius to consider (can be Quantity)

           ro, vo= standard unit-conversion parameters

        OUTPUT:

           (none)

        HISTORY:

           2021-03-21 - Started based on Marius Cautun's code - Bovy (UofT)

        """
        # Initialize with Force just to parse (ro,vo)
        Force.__init__(self,ro=ro,vo=vo)
        rmax= conversion.parse_length(rmax,ro=self._ro)
        rmin= conversion.parse_length(rmin,ro=self._ro) if not rmin is None \
            else rmax/2500.
        # Compute baryon and DM enclosed masses on radial grid
        from ..potential import mass
        rgrid= numpy.concatenate((numpy.array([0.]),
                                  numpy.geomspace(rmin,rmax,301)))
        baryon_mass= numpy.array([mass(baryonpot,r,use_physical=False)
                                  for r in rgrid])
        dm_mass= numpy.array([mass(pot,r,use_physical=False)
                              for r in rgrid])
        # Adiabatic contraction
        if f_bar is None:
            f_bar= baryon_mass[-1]/(baryon_mass[-1]+dm_mass[-1])
        eta_bar = baryon_mass/dm_mass*(1.-f_bar)/f_bar
        if method.lower() == 'cautun':
            new_rforce= dm_mass*(0.45+0.38*(eta_bar+1.16)**0.53)/rgrid**2.
        new_rforce[0]= 0.
        new_rforce_func= lambda r: -numpy.interp(r,rgrid,new_rforce)
        # Potential at zero = int_0^inf dr rforce
        Phi0= integrate.quad(new_rforce_func,0.,numpy.inf)[0]
        interpSphericalPotential.__init__(self,
                                          rforce=new_rforce_func,
                                          rgrid=rgrid,
                                          Phi0=Phi0,
                                          ro=ro,vo=vo)
        
