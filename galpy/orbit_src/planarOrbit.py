import numpy as nu
from scipy import integrate
from Orbit import Orbit
from galpy.potential_src.planarPotential import evaluateplanarRforces,\
    planarPotential, RZToplanarPotential
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

    def integrate(self,t,pot):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the orbit
        INPUT:
           t - list of times at which to output (0 has to be in this!)
           pot - potential instance or list of instances
        OUTPUT:
           (none) (get the actual orbit using getOrbit()
        HISTORY:
           2010-07-20
        """
        thispot= RZToplanarPotential(pot)
        self.t= nu.array(t)
        self.orbit= _integrateROrbit(self.vxvv,thispot,t)

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
    

def _integrateROrbit(vxvv,pot,t):
    """
    NAME:
       _integrateRZOrbit
    PURPOSE:
       integrate an orbit in a Phi(R) potential in the R-plane
    INPUT:
       vxvv - array with the initial conditions stacked like
              [R,vR,vT]; vR outward!
       pot - Potential instance
       t - list of times at which to output (0 has to be in this!)
    OUTPUT:
       [:,3] array of [R,vR,vT] at each t
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
    l= vxvv[0]*vxvv[2]
    l2= l**2.
    init= [vxvv[0],vxvv[1]]
    intOut= integrate.odeint(_REOM,init,t,args=(pot,l2),
                             rtol=10.**-8.)#,mxstep=100000000)
    out= nu.zeros((len(t),3))
    out[:,0]= intOut[:,0]
    out[:,1]= intOut[:,1]
    out[:,2]= l/out[:,0]
    return out

def _REOM(y,t,pot,l2):
    """
    NAME:
       _REOM
    PURPOSE:
       implements the EOM, i.e., the right-hand side of the differential 
       equation
    INPUT:
       y - current phase-space position
       t - current time
       pot - (list of) Potential instance(s)
       l2 - angular momentum squared
    OUTPUT:
       dy/dt
    HISTORY:
       2010-07-20 - Written - Bovy (NYU)
    """
    return [y[1],
            l2/y[0]**3.+evaluateplanarRforces(y[0],pot)]
