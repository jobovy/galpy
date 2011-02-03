import numpy as nu
from galpy.orbit import Orbit
class Snapshot:
    """General snapshot = collection of particles class"""
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a snapshot object
        INPUT:
           Initialize using:
              1) list of orbits, list of masses (masses=)
              Coming soon:
              2) observations
              3) DFs to draw from
        OUTPUT:
        HISTORY:
           2011-02-02 - Started - Bovy
        """
        if isinstance(args[0],list) and isinstance(args[0][0],Orbit):
            self.orbits= args[0]
            if kwargs.has_key('masses'):
                self.masses= kwargs['masses']
            else:
                self.masses= nu.ones(len(self.orbits))
        return None

    def integrate(self,t,pot=None,method='test-particle'):
        """
        NAME:
           integrate
        PURPOSE:
           integrate the snapshot in time
        INPUT:
           t - numpy.array of times to save the snapshots at (must start at 0)
           pot= potential object or list of such objects (default=None)
           method= method to use ('test-particle' or 'direct-python' for now)
        OUTPUT:
           list of snapshots at times t
        HISTORY:
           2011-02-02 - Written - Bovy (NYU)
        """
        if method.lower() == 'test-particle':
            return self._integrate_test_particle(t,pot)
        elif method.lower() == 'direct-python':
            raise AttributeError("'direct-python' not implemented yet")

    def _integrate_test_particle(self,t,pot):
        """Integrate the snapshot as a set of test particles in an external \
        potential"""
        #Integrate all the orbits
        for o in self.orbits:
            o.integrate(t,pot)
        #Return them as a set of snapshots
        out= []
        for ii in range(len(t)):
            outOrbits= []
            for o in self.orbits:
                outOrbits.append(o(t[ii]))
            out.append(Snapshot(outOrbits,self.masses))
        return out
