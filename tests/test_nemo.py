# Test consistency between galpy and NEMO
from __future__ import print_function, division
import os
import numpy
import subprocess
from galpy.orbit import Orbit
from galpy import potential
from galpy.util import bovy_conversion
def test_nemo_MN3ExponentialDiskPotential():
    mn= potential.MN3ExponentialDiskPotential(normalize=1.,hr=0.5,hz=0.1)
    tmax= 3.
    vo,ro= 215., 8.75
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mn,tmax,vo,ro)
    return None

def test_nemo_MiyamotoNagaiPotential():
    mp= potential.MiyamotoNagaiPotential(normalize=1.,a=0.5,b=0.1)
    tmax= 4.
    vo,ro= 220., 8.
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mp,tmax,vo,ro)
    return None

def test_nemo_NFWPotential():
    np= potential.NFWPotential(normalize=1.,a=3.)
    tmax= 3.
    vo,ro= 200., 7.
    o= Orbit([1.,0.5,1.3,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,np,tmax,vo,ro)
    return None

def test_nemo_HernquistPotential():
    hp= potential.HernquistPotential(normalize=1.,a=3.)
    tmax= 3.
    vo,ro= 210., 7.5
    o= Orbit([1.,0.25,1.4,0.3,-0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,hp,tmax,vo,ro)
    return None

def test_nemo_PowerSphericalPotentialwCutoffPotential():
    pp= potential.PowerSphericalPotentialwCutoff(normalize=1.,alpha=1.,rc=0.4)
    tmax= 2.
    vo,ro= 180., 9.
    o= Orbit([1.,0.03,1.03,0.2,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,pp,tmax,vo,ro)
    return None

def test_nemo_LogarithmicHaloPotential():
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    tmax= 2.
    vo,ro= 210., 8.5
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,lp,tmax,vo,ro,tol=0.03)
    return None

def test_nemo_PlummerPotential():
    pp= potential.PlummerPotential(normalize=1.,b=2.)
    tmax= 3.
    vo,ro= 213., 8.23
    o= Orbit([1.,0.1,1.1,0.3,0.1,0.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,pp,tmax,vo,ro,tol=0.03)
    return None

def test_nemo_MWPotential2014():
    mp= potential.MWPotential2014
    tmax= 3.5
    vo,ro= 220., 8.
    o= Orbit([1.,0.1,1.1,0.2,0.1,1.4],ro=ro,vo=vo)
    run_orbitIntegration_comparison(o,mp,tmax,vo,ro,isList=True)
    return None

def run_orbitIntegration_comparison(orb,pot,tmax,vo,ro,isList=False,
                                    tol=0.01):
    # Integrate in galpy
    ts= numpy.linspace(0.,tmax/bovy_conversion.time_in_Gyr(vo,ro),1001)
    orb.integrate(ts,pot)
    # Now setup a NEMO snapshot in the correct units ([x] = kpc, [v] = kpc/Gyr)
    numpy.savetxt('orb.dat',
                  numpy.array([[10.**-6.,orb.x(),orb.y(),orb.z(),
                                orb.vx(use_physical=False)\
                                    *bovy_conversion.velocity_in_kpcGyr(vo,ro),
                                orb.vy(use_physical=False)\
                                    *bovy_conversion.velocity_in_kpcGyr(vo,ro),
                                orb.vz(use_physical=False)\
                                    *bovy_conversion.velocity_in_kpcGyr(vo,ro)]]))
    # Now convert to NEMO format
    try:
        convert_to_nemo('orb.dat','orb.nemo')
    finally:
        os.remove('orb.dat')
    # Integrate with gyrfalcON
    try:
        if isList:
            integrate_gyrfalcon('orb.nemo','orb_evol.nemo',tmax,
                                potential.nemo_accname(pot),
                                potential.nemo_accpars(pot,vo,ro))
        else:
            integrate_gyrfalcon('orb.nemo','orb_evol.nemo',tmax,
                                pot.nemo_accname(),pot.nemo_accpars(vo,ro))
    finally:
        os.remove('orb.nemo')
        os.remove('gyrfalcON.log')
    # Convert back to ascii
    try:
        convert_from_nemo('orb_evol.nemo','orb_evol.dat')
    finally:
        os.remove('orb_evol.nemo')
    # Read and compare
    try:
        nemodata= numpy.loadtxt('orb_evol.dat',comments='#')
        xdiff= numpy.fabs((nemodata[-1,1]-orb.x(ts[-1]))/nemodata[-1,1])
        ydiff= numpy.fabs((nemodata[-1,2]-orb.y(ts[-1]))/nemodata[-1,2])
        zdiff= numpy.fabs((nemodata[-1,3]-orb.z(ts[-1]))/nemodata[-1,3])
        vxdiff= numpy.fabs((nemodata[-1,4]-orb.vx(ts[-1],use_physical=False)*bovy_conversion.velocity_in_kpcGyr(vo,ro))/nemodata[-1,4])
        vydiff= numpy.fabs((nemodata[-1,5]-orb.vy(ts[-1],use_physical=False)*bovy_conversion.velocity_in_kpcGyr(vo,ro))/nemodata[-1,5])
        vzdiff= numpy.fabs((nemodata[-1,6]-orb.vz(ts[-1],use_physical=False)*bovy_conversion.velocity_in_kpcGyr(vo,ro))/nemodata[-1,6])
        assert xdiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for x by %g' % xdiff
        assert ydiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for y by %g' % ydiff
        assert zdiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for z by %g' % zdiff
        assert vxdiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for vx by %g' % vxdiff
        assert vydiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for vy by %g' % vydiff
        assert vzdiff < tol, 'galpy and NEMO gyrfalcON orbit integration inconsistent for vz by %g' % vzdiff
    finally:
        os.remove('orb_evol.dat')
    return None

def convert_to_nemo(infile,outfile):
    subprocess.check_call(['a2s','in=%s'% infile,'out=%s' % outfile,'N=1',
                           'read=mxv'])
    
def convert_from_nemo(infile,outfile):
    subprocess.check_call(['s2a','in=%s' % infile,'out=%s' % outfile])
    
def integrate_gyrfalcon(infile,outfile,tmax,nemo_accname,nemo_accpars):
    """Integrate a snapshot in infile until tmax in Gyr, save to outfile"""
    with open('gyrfalcON.log','w') as f:
        subprocess.check_call(['gyrfalcON',
                               'in=%s' % infile,
                               'out=%s' % outfile,
                               'tstop=%g' % tmax,
                               'eps=0.0015',
                               'step=0.01',
                               'kmax=10',
                               'Nlev=8',
                               'fac=0.01',
                               'accname=%s' % nemo_accname,
                               'accpars=%s' % nemo_accpars],
                              stdout=f)
    return None
