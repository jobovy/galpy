import numpy as nu
from galpy.orbit import Orbit
from galpy.potential_src.planarPotential import RZToplanarPotential
from directnbody import direct_nbody
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

    def integrate(self,t,pot=None,method='test-particle',
                  **kwargs):
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
            return self._integrate_direct_python(t,pot,**kwargs)

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

    def _integrate_direct_python(self,t,pot,**kwargs):
        """Integrate the snapshot using a direct force summation method \
        written entirely in python"""
        #Prepare input for direct_nbody
        q= []
        p= []
        nq= len(self.orbits)
        dim= self.orbits[0].dim()
        if pot is None:
            thispot= None
        elif dim == 2:
            thispot= RZToplanarPotential(pot)
        else:
            thispot= pot
        for ii in range(nq):
            #Transform to rectangular frame
            if dim == 1:
                thisq= nu.array([self.orbits[ii].x()]).flatten()
                thisp= nu.array([self.orbits[ii].vx()]).flatten()
            elif dim == 2:
                thisq= nu.array([self.orbits[ii].x(),
                                 self.orbits[ii].y()]).flatten()
                thisp= nu.array([self.orbits[ii].vx(),
                                 self.orbits[ii].vy()]).flatten()
            elif dim == 3:
                thisq= nu.array([self.orbits[ii].x(),
                                 self.orbits[ii].y(),
                                 self.orbits[ii].z()]).flatten()
                thisp= nu.array([self.orbits[ii].vx(),
                                 self.orbits[ii].vy(),
                                 self.orbits[ii].vz()]).flatten()
            q.append(thisq)
            p.append(thisp)
        #Run simulation
        nbody_out= direct_nbody(q,p,self.masses,t,pot=thispot,**kwargs)
        #Post-process output
        nt= len(nbody_out)
        out= []
        for ii in range(nt):
            snap_orbits= []
            for jj in range(nq):
                if dim == 3:
                    #go back to the cylindrical frame
                    R= nu.sqrt(nbody_out[ii][0][jj][0]**2.
                               +nbody_out[ii][0][jj][1]**2.)
                    phi= nu.arccos(nbody_out[ii][0][jj][0]/R)
                    if nbody_out[ii][0][jj][1] < 0.: phi= 2.*nu.pi-phi
                    vR= nbody_out[ii][1][jj][0]*nu.cos(phi)\
                        +nbody_out[ii][1][jj][1]*nu.sin(phi)
                    vT= nbody_out[ii][1][jj][1]*nu.cos(phi)\
                        -nbody_out[ii][1][jj][0]*nu.sin(phi)
                    vxvv= nu.zeros(dim*2)
                    vxvv[3]= nbody_out[ii][0][jj][2]
                    vxvv[4]= nbody_out[ii][1][jj][2]
                    vxvv[0]= R
                    vxvv[1]= vR
                    vxvv[2]= vT
                    vxvv[5]= phi
                if dim == 2:
                    #go back to the cylindrical frame
                    R= nu.sqrt(nbody_out[ii][0][jj][0]**2.
                               +nbody_out[ii][0][jj][1]**2.)
                    phi= nu.arccos(nbody_out[ii][0][jj][0]/R)
                    if nbody_out[ii][0][jj][1] < 0.: phi= 2.*nu.pi-phi
                    vR= nbody_out[ii][1][jj][0]*nu.cos(phi)\
                        +nbody_out[ii][1][jj][1]*nu.sin(phi)
                    vT= nbody_out[ii][1][jj][1]*nu.cos(phi)\
                        -nbody_out[ii][1][jj][0]*nu.sin(phi)
                    vxvv= nu.zeros(dim*2)
                    vxvv[0]= R
                    vxvv[1]= vR
                    vxvv[2]= vT
                    vxvv[3]= phi
                if dim == 1:
                    vxvv= [nbody_out[ii][0][jj],nbody_out[ii][1][jj]]
                snap_orbits.append(Orbit(vxvv))
            out.append(Snapshot(snap_orbits,self.masses))
        return out
