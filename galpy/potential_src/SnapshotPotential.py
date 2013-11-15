from pynbody import grav_omp
import numpy as np
from galpy.potential import Potential
import hashlib

class SnapshotPotential(Potential):
    """Create a snapshot potential object. The potential and forces are 
    calculated as needed through the _evaluate and _Rforce methods. 
    Requires an installation of [pynbody](http://pynbody.github.io).
    
    `_evaluate` and `_Rforce` calculate a hash for the array of points
    that is passed in by the user. The hash and corresponding
    potential/force arrays are stored -- if a subsequent request
    matches a previously computed hash, the previous results are
    returned and note recalculated.
    
    **Input**:
    
    *s* : a simulation snapshot loaded with pynbody

    **Optional Keywords**:
    
    *num_threads* (4): number of threads to use for calculation

    """

    def __init__(self, s, num_threads=4) : 
        self.s = s
        self._point_hash = {}
        self._amp = 1.0
    
    def __call__(self, R, z) : 
        return self._evaluate(R,z)

    def _evaluate(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        pot, acc = self._setup_potential(R,z)
        return pot
        
    def _Rforce(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        pot, acc = self._setup_potential(R,z)
        return acc[:,0]

    def _setup_potential(self, R, z) : 
        # cast the points into arrays for compatibility
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        # compute the hash for the requested grid
        new_hash = hashlib.md5(np.array([R,z])).hexdigest()

        # if we computed for these points before, return; otherwise compute
        if new_hash in self._point_hash : 
            pot, rz_acc = self._point_hash[new_hash]
        else : 
            # set up the four points per R,z pair to mimic axisymmetry
            points = np.zeros((len(R),len(z),4,3))
        
            for i in xrange(len(R)) :
                for j in xrange(len(z)) : 
                    points[i,j] = [(R[i],0,z[j]),
                                   (0,R[i],z[j]),
                                   (-R[i],0,z[j]),
                                   (0,-R[i],z[j])]

            points_new = points.reshape(points.size/3,3)
            pot, acc = grav_omp.direct(self.s,points_new,num_threads=4)

            pot = pot.reshape(len(R)*len(z),4)
            acc = acc.reshape(len(R)*len(z),4,3)

            # need to average the potentials
            if len(pot) > 1:
                pot = pot.mean(axis=1)
            else : 
                pot = pot.mean()


            # get the radial accelerations
            rz_acc = np.zeros((len(R)*len(z),2))
            rvecs = [(1.0,0.0,0.0),
                     (0.0,1.0,0.0),
                     (-1.0,0.0,0.0),
                     (0.0,-1.0,0.0)]
        
            # reshape the acc to make sure we have a leading index even
            # if we are only evaluating a single point, i.e. we have
            # shape = (1,4,3) not (4,3)
            acc = acc.reshape((len(rz_acc),4,3))

            for i in xrange(len(R)) : 
                for j,rvec in enumerate(rvecs) : 
                    rz_acc[i,0] += acc[i,j].dot(rvec)
                    rz_acc[i,1] += acc[i,j,2]
            rz_acc /= 4.0
            
            # store the computed values for reuse
            self._point_hash[new_hash] = [pot,rz_acc]

        return pot, rz_acc
