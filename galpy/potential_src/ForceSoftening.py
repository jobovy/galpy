###############################################################################
#   ForceSoftening: class representing a force softening kernel
###############################################################################
import numpy as nu
class ForceSoftening:
    """class representing a force softening kernel"""
    def __init__(self): #pragma: no cover
        pass

    def __call__(self,d): #pragma: no cover
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the force of the softening kernel
        INPUT:
           d - distance
        OUTPUT:
           softened force (amplitude; without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'__call__' not implemented for this softening kernel")

    def potential(self,d): #pragma: no cover
        """
        NAME:
           potential
        PURPOSE:
           return the potential corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           potential (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'potential' not implemented for this softening kernel")

    def density(self,d): #pragma: no cover
        """
        NAME:
           density
        PURPOSE:
           return the density corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           density (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        raise AttributeError("'density' not implemented for this softening kernel")

class PlummerSoftening (ForceSoftening):
    """class representing a Plummer softening kernel"""
    def __init__(self,softening_length=0.01):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a Plummer softening kernel
        INPUT:
           softening_length=
        OUTPUT:
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        self._softening_length= softening_length

    def __call__(self,d):
        """
        NAME:
           __call__
        PURPOSE:
           evaluate the force of the softening kernel
        INPUT:
           d - distance
        OUTPUT:
           softened force (amplitude; without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        return d/(d**2.+self._softening_length**2.)**1.5

    def potential(self,d):
        """
        NAME:
           potential
        PURPOSE:
           return the potential corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           potential (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        return (d**2.+self._softening_length**2.)**-0.5

    def density(self,d):
        """
        NAME:
           density
        PURPOSE:
           return the density corresponding to this softening kernel
        INPUT:
           d - distance
        OUTPUT:
           density (without GM)
        HISTORY:
           2011-04-13 - Written - Bovy (NYU)
        """
        return 3./4./nu.pi*self._softening_length**2.\
            *(d**2.+self._softening_length**2.)**-2.5
