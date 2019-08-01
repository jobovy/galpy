# galpy.potential.mwpotentials: Milky-Way-like potentials and tools for 
# working with MW-like potentials (bars, spirals, ...)
import sys
import os
import copy
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
# Following class allows potentials that are expensive to setup to be 
# lazily-loaded (see _mcmillan17.py)
def _setup_globals(): # this func necessary to transfer *all* globals in Py2
    out= copy.copy(globals())
    out['__path__']= [os.path.dirname(__file__)]
    return out
class _ExpensivePotentials(object):
    def __init__(self):
        # Initialize all expensive potentials as None, filled in when loaded
        self._mcmillan17= None
        # This is necessary to transfer *all* globals in Py2
        self.__globals__= _setup_globals()
        return None

    # For tab completion
    def __dir__(self):
        return ['McMillan17']

    @property
    def McMillan17(self):
        if not self._mcmillan17:
            # In python 3 this can be a relative import, but for some reason
            # in python 2 it cannot
            from galpy.potential._mcmillan17 import McMillan17 as _McMillan17
            self._mcmillan17= _McMillan17
        return self._mcmillan17

    def __getattr__(self,name):
        try:
            #In Py3 you can just do 'return globals()[name]', but not in Py2
            return self.__globals__[name]
        except: return None

__all__= ['MWPotential2014']

# The magic to make lazy-loading of expensive potentials possible
sys.modules[__name__] = _ExpensivePotentials()

    
