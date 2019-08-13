# Test consistency between galpy and amuse
from __future__ import print_function, division
import os
import numpy
import subprocess
from galpy.orbit import Orbit
from galpy import potential
from galpy.util import bovy_conversion,bovy_coords

from galpy.potential import to_amuse

from amuse.lab import *
from amuse.couple import bridge
from amuse.datamodel import Particles

def test_amuse_MN3ExponentialDiskPotential():
    mn= potential.MN3ExponentialDiskPotential(normalize=1.,hr=0.5,hz=0.1)
    tmax= 3.
    vo,ro= 215., 8.75
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mn,tmax,vo,ro)
    return None

def test_amuse_MiyamotoNagaiPotential():
    mp= potential.MiyamotoNagaiPotential(normalize=1.,a=0.5,b=0.1)
    tmax= 4.
    vo,ro= 220., 8.
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mp,tmax,vo,ro)
    return None

def test_amuse_NFWPotential():
    np= potential.NFWPotential(normalize=1.,a=3.)
    tmax= 3.
    vo,ro= 200., 7.
    o= Orbit([1.,0.5,1.3,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,np,tmax,vo,ro)
    return None

def test_amuse_HernquistPotential():
    hp= potential.HernquistPotential(normalize=1.,a=3.)
    tmax= 3.
    vo,ro= 210., 7.5
    o= Orbit([1.,0.25,1.4,0.3,-0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,hp,tmax,vo,ro,tol=0.02)
    return None

def test_amuse_PowerSphericalPotentialwCutoffPotential():
    pp= potential.PowerSphericalPotentialwCutoff(normalize=1.,alpha=1.,rc=0.4)
    tmax= 2.
    vo,ro= 180., 9.
    o= Orbit([1.,0.03,1.03,0.2,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,pp,tmax,vo,ro)
    return None

def test_amuse_LogarithmicHaloPotential():
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    tmax= 2.
    vo,ro= 210., 8.5
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,lp,tmax,vo,ro,tol=0.03)
    return None

def test_amuse_PlummerPotential():
    pp= potential.PlummerPotential(normalize=1.,b=2.)
    tmax= 3.
    vo,ro= 213., 8.23
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,pp,tmax,vo,ro,tol=0.03)
    return None

def test_amuse_MWPotential2014():
    mp= potential.MWPotential2014
    tmax= 3.5
    vo,ro= 220., 8.
    o= Orbit([1.,0.1,1.1,0.2,0.1,1.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mp,tmax,vo,ro)
    return None

def run_orbitIntegration_comparison(orb,pot,tmax,vo,ro,tol=0.01):
    # Integrate in galpy
    ts= numpy.linspace(0.,tmax/bovy_conversion.time_in_Gyr(vo,ro),1001)
    orb.integrate(ts,pot)

    # Integrate with amuse
    x,y,z,vx,vy,vz=integrate_amuse(orb,pot,tmax | units.Gyr, vo,ro)

    # Read and compare

    xdiff= numpy.fabs((x-orb.x(ts[-1]))/x)
    ydiff= numpy.fabs((y-orb.y(ts[-1]))/y)
    zdiff= numpy.fabs((z-orb.z(ts[-1]))/z)
    vxdiff= numpy.fabs((vx-orb.vx(ts[-1]))/vx)
    vydiff= numpy.fabs((vy-orb.vy(ts[-1]))/vy)
    vzdiff= numpy.fabs((vz-orb.vz(ts[-1]))/vz)
    assert xdiff < tol, 'galpy and amuse orbit integration inconsistent for x by %g' % xdiff
    assert ydiff < tol, 'galpy and amuse orbit integration inconsistent for y by %g' % ydiff
    assert zdiff < tol, 'galpy and amuse orbit integration inconsistent for z by %g' % zdiff
    assert vxdiff < tol, 'galpy and amuse orbit integration inconsistent for vx by %g' % vxdiff
    assert vydiff < tol, 'galpy and amuse orbit integration inconsistent for vy by %g' % vydiff
    assert vzdiff < tol, 'galpy and amuse orbit integration inconsistent for vz by %g' % vzdiff

    return None
    
def integrate_amuse(orb,pot,tmax,vo,ro):
    """Integrate a snapshot in infile until tmax in Gyr, save to outfile"""

    time=0.0 | tmax.unit
    dt = tmax/10001.

    orbit = Particles(1)

    orbit.mass= 1. | units.MSun
    orbit.radius = 1. |units.RSun

    orbit.position=[orb.x(),orb.y(),orb.z()] | units.kpc
    orbit.velocity=[orb.vx(),orb.vy(),orb.vz()] | units.kms
    galaxy_code = to_amuse(pot,ro=ro,vo=vo)
    
    orbit_gravity=drift_without_gravity(orbit)
    orbit_gravity.particles.add_particles(orbit)
    channel_from_gravity_to_orbit= orbit_gravity.particles.new_channel_to(orbit)

    gravity = bridge.Bridge(use_threading=False)
    gravity.add_system(orbit_gravity, (galaxy_code,))
    gravity.add_system(galaxy_code,)
    gravity.timestep = dt

    while time <= tmax:
        time += dt
        gravity.evolve_model(time)

    channel_from_gravity_to_orbit.copy()
    gravity.stop()

    return orbit.x[0].value_in(units.kpc),orbit.y[0].value_in(units.kpc),orbit.z[0].value_in(units.kpc),orbit.vx[0].value_in(units.kms),orbit.vy[0].value_in(units.kms),orbit.vz[0].value_in(units.kms)


class drift_without_gravity(object):
    def __init__(self, convert_nbody, time= 0 |units.Myr):
        self.model_time= time
        self.particles=Particles()

    def evolve_model(self, t_end):
        dt= t_end- self.model_time
        self.particles.position += self.particles.velocity*dt
        self.model_time= t_end
    @property
    def potential_energy(self):
        return quantities.zero
    @property
    def get_potential_at_point(self):
        return quantities.zero
    @property
    def kinetic_energy(self):
        return (0.5*self.particles.mass*self.particles.velocity.lengths()**2).sum()
    @property
    def angular_momenum(self):
        return np.cross(self.particles.position,self.particles.velocity)
    def stop(self):
        pass
