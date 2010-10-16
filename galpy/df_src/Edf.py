###############################################################################
#   Edf.py: top-level class for distribution functions that are only a 
#           function of the energy
###############################################################################
from galpy.orbit import Orbit
class Edf:
    """Top-level class for distribution functions that are only a function of
    the energy"""
    def __init__(self):
        return None

    def __call__(self,call_in,t=None):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the DF
        INPUT:
           Either:
              a) E - the energy
              b) Some orbit instance, whose E(t) method will be called; t is an
                 optional second input
        OUTPUT:
           df(E) or df(Orbit(t))
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        if isinstance(call_in,Orbit):
            if not t == None:
                E= call_in.E(t)
            else:
                E= call_in.E()
            return self.eval(E)
        else:
            return self.eval(call_in)

    def eval(self,E):
        """
        NAME:
           eval
        PURPOSE:
           evaluate the DF
        INPUT:
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        raise AttributeError("eval() function of this DF is not implemented")

    def sample(self,pot,n=1):
        """
        NAME:
           sample
        PURPOSE:
           sample from this df
        INPUT:
           pot - potential
           n - number of samples desired
        OUTPUT:
           depending on the dimension, 
           list of [linearOrbit,planarOrbit,Orbit]s
           or a single orbit if n=1
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        BUGS:
           sampling is bad and inaccurate at large x
        """
        raise AttributeError("'sample' method for this ergodic df is not implemented")
