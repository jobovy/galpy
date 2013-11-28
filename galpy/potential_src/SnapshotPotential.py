try: 
    from pynbody import grav_omp
except ImportError: 
    raise ImportError("This class is designed to work with pynbody snapshots -- obtain from pynbody.github.io")

import pynbody
from pynbody import grav_omp
import numpy as np
from Potential import Potential
import hashlib
from scipy.misc import derivative
import interpRZPotential
from scipy import interpolate 
from os import system
from galpy.util import multi

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

    def __init__(self, s, num_threads=pynbody.config['number_of_threads']) : 
        self.s = s
        self._point_hash = {}
        self._amp = 1.0
        self._num_threads = num_threads
    
    def __call__(self, R, z, phi = None, t = None) : 
        # cast the points into arrays for compatibility
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        return self._evaluate(R,z)

    def _evaluate(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        pot, acc = self._setup_potential(R,z)
        return pot
        
    def _Rforce(self, R,z,phi=None,t=None,dR=None,dphi=None) : 
        pot, acc = self._setup_potential(R,z)
        return acc[:,0]

#    def _R2deriv(self, R,z,phi=None,t=None) : 
        

    def _setup_potential(self, R, z, use_pkdgrav = False) : 
        from galpy.potential import vcirc
        # cast the points into arrays for compatibility
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        # compute the hash for the requested grid
        new_hash = hashlib.md5(np.array([R,z])).hexdigest()

        # if we computed for these points before, return; otherwise compute
        if new_hash in self._point_hash : 
            pot, r_acc = self._point_hash[new_hash]

#        if use_pkdgrav :
            

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
            pot, acc = grav_omp.direct(self.s,points_new,num_threads=self._num_threads)

            pot = pot.reshape(len(R)*len(z),4)
            acc = acc.reshape(len(R)*len(z),4,3)

            # need to average the potentials
            if len(pot) > 1:
                pot = pot.mean(axis=1)
            else : 
                pot = pot.mean()


            # get the radial accelerations
            r_acc = np.zeros((len(R)*len(z),2))
            rvecs = [(1.0,0.0,0.0),
                     (0.0,1.0,0.0),
                     (-1.0,0.0,0.0),
                     (0.0,-1.0,0.0)]
        
            # reshape the acc to make sure we have a leading index even
            # if we are only evaluating a single point, i.e. we have
            # shape = (1,4,3) not (4,3)
            acc = acc.reshape((len(r_acc),4,3))

            for i in xrange(len(R)) : 
                for j,rvec in enumerate(rvecs) : 
                    r_acc[i,0] += acc[i,j].dot(rvec)
                    r_acc[i,1] += acc[i,j,2]
            r_acc /= 4.0
            
            # store the computed values for reuse
            self._point_hash[new_hash] = [pot,r_acc]

        return pot, r_acc


class InterpSnapshotPotential(interpRZPotential.interpRZPotential) : 
    """
    Interpolated potential extracted from a simulation output. 

    
    
    """

    
    def __init__(self, s, 
                 rgrid=(0.01,2.,101), zgrid=(0.,0.2,101), 
                 interpepifreq = False, interpverticalfreq = False, interpvcirc = False, interpPot = False,
                 interpDens = False, interpdvcircdr = False,
                 enable_c = True, logR = False, zsym = True, num_threads=pynbody.config['number_of_threads'], use_pkdgrav = False) : 
        self._num_threads = num_threads
        self.s = s
        self._amp = 1.0
        
        self._interpPot = True
        self._interpRforce = True
        self._interpzforce = True
        self._interpvcirc = True
        self._interpepifreq = True
        self._interpverticalfreq = True

        self._zsym = zsym
        self._enable_c = enable_c

        self._logR = logR

        # make the potential accessible at points beyond the grid
        self._origPot = SnapshotPotential(s, num_threads)

        # setup the grid
        self._rgrid = np.linspace(*rgrid)
        if logR : 
            self._rgrid = np.exp(self._rgrid)
            self._logrgrid = np.log(self._rgrid)
            rs = self._logrgrid
        else : 
            rs = self._rgrid

        self._zgrid = np.linspace(*zgrid)

        # calculate the grids
        self._setup_potential(self._rgrid,self._zgrid,use_pkdgrav=use_pkdgrav)

        if enable_c : 
            self._potGrid_splinecoeffs    = interpRZPotential.calc_2dsplinecoeffs_c(self._potGrid)
            self._rforceGrid_splinecoeffs = interpRZPotential.calc_2dsplinecoeffs_c(self._rforceGrid)
            self._zforceGrid_splinecoeffs = interpRZPotential.calc_2dsplinecoeffs_c(self._zforceGrid)

        else :
            self._potInterp= interpolate.RectBivariateSpline(rs,
                                                             self._zgrid,
                                                             self._potGrid,
                                                             kx=3,ky=3,s=0.)
            self._rforceInterp= interpolate.RectBivariateSpline(rs,
                                                                self._zgrid,
                                                                self._rforceGrid,
                                                                kx=3,ky=3,s=0.)
            self._zforceInterp= interpolate.RectBivariateSpline(rs,
                                                                self._zgrid,
                                                                self._zforceGrid,
                                                                kx=3,ky=3,s=0.)

        self._R2interp = interpolate.RectBivariateSpline(rs,
                                                         self._zgrid,
                                                         self._R2derivGrid,
                                                         kx=3,ky=3,s=0.)
            
        self._z2interp = interpolate.RectBivariateSpline(rs,
                                                         self._zgrid,
                                                         self._z2derivGrid,
                                                         kx=3,ky=3,s=0.)
         
        # setup the derived quantities

        self._vcircGrid = np.sqrt(self._rgrid*(-self._rforceGrid[:,0]))
        self._epifreqGrid = np.sqrt(self._R2derivGrid[:,0]-
                                    3./self._rgrid*self._rforceGrid[:,0])

        self._verticalfreqGrid = np.sqrt(np.abs(self._z2derivGrid[:,0]))
        
        self._vcircInterp = interpolate.InterpolatedUnivariateSpline(rs, self._vcircGrid, k=3)
        self._epifreqInterp = interpolate.InterpolatedUnivariateSpline(rs, self._epifreqGrid, k=3)
        self._verticalfreqInterp = interpolate.InterpolatedUnivariateSpline(rs, self._verticalfreqGrid, k=3)

            

           

    def __call__(self, R, z, phi = 0.0, t = 0.0) : 
        # cast the points into arrays for compatibility
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        return self._evaluate(R,z)

        
    def _setup_potential(self, R, z, use_pkdgrav = False, dr = 0.1) : 
        from galpy.potential import vcirc

        # cast the points into arrays for compatibility
        if isinstance(R,float) : 
            R = np.array([R])
        if isinstance(z, float) : 
            z = np.array([z])

        # set up the four points per R,z pair to mimic axisymmetry
        points = np.zeros((len(R),len(z),4,3))
        
        for i in xrange(len(R)) :
            for j in xrange(len(z)) : 
                points[i,j] = [(R[i],0,z[j]),
                               (0,R[i],z[j]),
                               (-R[i],0,z[j]),
                               (0,-R[i],z[j])]

        points_new = points.reshape(points.size/3,3)

        # set up the points to calculate the second derivatives
        zgrad_points = np.zeros((len(points_new)*2,3))
        rgrad_points = np.zeros((len(points_new)*2,3))
        for i,p in enumerate(points_new) : 
            zgrad_points[i*2] = p
            zgrad_points[i*2][2] -= dr
            zgrad_points[i*2+1] = p
            zgrad_points[i*2+1][2] += dr
            
            rgrad_points[i*2] = p
            rgrad_points[i*2][:2] -= p[:2]/np.sqrt(np.dot(p[:2],p[:2]))*dr
            rgrad_points[i*2+1] = p
            rgrad_points[i*2+1][:2] += p[:2]/np.sqrt(np.dot(p[:2],p[:2]))*dr
                        

        if use_pkdgrav :
            raise RuntimeError("using pkdgrav not currently implemented")
            sn = pynbody.snapshot._new(len(self.s.d)+len(self.s.g)+len(self.s.s)+len(points_new))
            print "setting up %d grid points"%(len(points_new))
            #sn['pos'][0:len(self.s)] = self.s['pos']
            #sn['mass'][0:len(self.s)] = self.s['mass']
            #sn['phi'] = 0.0
            #sn['eps'] = 1e3
            #sn['eps'][0:len(self.s)] = self.s['eps']
            #sn['vel'][0:len(self.s)] = self.s['vel']
            #sn['mass'][len(self.s):] = 1e-10
            sn['pos'][len(self.s):] = points_new
            sn['mass'][len(self.s):] = 0.0
            
                
            sn.write(fmt=pynbody.tipsy.TipsySnap, filename='potgridsnap')
            command = '~/bin/pkdgrav2_pthread -sz %d -n 0 +std -o potgridsnap -I potgridsnap +potout +overwrite %s'%(self._num_threads, self.s._paramfile['filename'])
            print command
            system(command)
            sn = pynbody.load('potgridsnap')
            acc = sn['accg'][len(self.s):].reshape(len(R)*len(z),4,3)
            pot = sn['pot'][len(self.s):].reshape(len(R)*len(z),4)
            

        else : 
  
            pot, acc = grav_omp.direct(self.s,points_new,num_threads=self._num_threads)

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

            for i in xrange(len(R)*len(z)) : 
                for j,rvec in enumerate(rvecs) : 
                    rz_acc[i,0] += acc[i,j].dot(rvec)
                    rz_acc[i,1] += acc[i,j,2]
            rz_acc /= 4.0
            

            # compute the force gradients

            # first get the accelerations
            zgrad_pot, zgrad_acc = grav_omp.direct(self.s,zgrad_points,num_threads=self._num_threads)
            rgrad_pot, rgrad_acc = grav_omp.direct(self.s,rgrad_points,num_threads=self._num_threads)

            # each point from the points used above for pot and acc is straddled by 
            # two points to get the gradient across it. Compute the gradient by 
            # using a finite difference 

            zgrad = np.zeros(len(points_new))
            rgrad = np.zeros(len(points_new))

            # do a loop through the pairs of points -- reshape the array
            # so that each item is the pair of acceleration vectors
            # then calculate the gradient from the two points
            for i,zacc in enumerate(zgrad_acc.reshape((len(zgrad_acc)/2,2,3))) :
                zgrad[i] = ((zacc[1]-zacc[0])/(dr*2.0))[2]

            for i,racc in enumerate(rgrad_acc.reshape((len(rgrad_acc)/2,2,3))) :
                point = points_new[i]
                point[2] = 0.0
                rvec = point/np.sqrt(np.dot(point,point))
                rgrad_vec = (np.dot(racc[1],rvec)-
                             np.dot(racc[0],rvec)) / (dr*2.0)
                rgrad[i] = rgrad_vec


                        
            self.zgrad_acc = zgrad_acc
            self.rgrad_acc = rgrad_acc
            self.zgrad_points = zgrad_points
            self.rgrad_points = rgrad_points
            # reshape the arrays
            self._z2derivGrid = zgrad.reshape((len(zgrad)/4,4)).mean(axis=1).reshape((len(R),len(z)))
            self._R2derivGrid = rgrad.reshape((len(rgrad)/4,4)).mean(axis=1).reshape((len(R),len(z)))
            self.points = points_new
    
        self._potGrid = pot.reshape((len(R),len(z)))
        self._rforceGrid = rz_acc[:,0].reshape((len(R),len(z)))
        self._zforceGrid = rz_acc[:,1].reshape((len(R),len(z)))
    
    def _R2deriv(self,R,Z,phi=0.,t=0.): 
        if not phi == 0.0 or not t == 0.0 : 
            raise RuntimeError("Only axisymmetric potentials are supported")
        if self._zsym: Z = np.abs(Z)
        return self._R2interp(R,Z)

    def _z2deriv(self,R,Z,phi=None,t=None):
        if not phi == 0.0 or not t == 0.0 : 
            raise RuntimeError("Only axisymmetric potentials are supported")
        if self._zsym: Z = np.abs(Z)
        return self._z2interp(R,Z)
