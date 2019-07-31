# galpy.potential.mwpotentials: Milky-Way-like potentials and tools for 
# working with MW-like potentials (bars, spirals, ...)
from . import HernquistPotential
from . import MiyamotoNagaiPotential
from . import NFWPotential
from . import PowerSphericalPotentialwCutoff

############################ MILKY WAY MODELS #################################
# galpy's first version of a MW-like potential, kept for backwards 
# compatibility and reproducibility-of-results-in-the-literature reasons, 
# underscore it here to avoid use
_MWPotential= [MiyamotoNagaiPotential(a=0.5,b=0.0375,normalize=.6),
               NFWPotential(a=4.5,normalize=.35),
               HernquistPotential(a=0.6/8,normalize=0.05)]
# See Table 1 in galpy paper: Bovy (2014)
MWPotential2014= [PowerSphericalPotentialwCutoff(normalize=0.05,alpha=1.8,
                                                 rc=1.9/8.),
                  MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=0.6),
                  NFWPotential(a=2.,normalize=0.35)]
