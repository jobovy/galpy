import numpy as nu
from Orbit import Orbit
class planarOrbitTop(Orbit):
    """Top-level class representing a planar orbit (i.e., one in the plane 
    of a galaxy)"""
    def __init__(self,vxvv=None):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planar orbit
        INPUT:
           vxvv - [R,vR,vT(,phi)]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        return None

class planarROrbit(planarOrbitTop):
    """Class representing a planar orbit, without \phi. Useful for 
    orbit-integration in axisymmetric potentials when you don't care about the
    azimuth"""
    def __init__(self,vxvv=[1.,0.,1.]):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planarROrbit
        INPUT:
           vxvv - [R,vR,vT]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        self.vxvv= vxvv
        return None

class planarOrbit(planarOrbitTop):
    """Class representing a full planar orbit (R,vR,vT,phi)"""
    def __init__(self,vxvv=[1.,0.,1.,0.]):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a planarOrbit
        INPUT:
           vxvv - [R,vR,vT,phi]
        OUTPUT:
        HISTORY:
           2010-07-12 - Written - Bovy (NYU)
        """
        if len(vxvv) == 3:
            raise ValueError("You only provided R,vR, & vT, but not phi; you probably want planarROrbit")
        self.vxvv= vxvv
        return None
    
