##############################TESTS ON ORBITS##################################
import sys
import numpy
import os
from galpy import potential
_TRAVIS= bool(os.getenv('TRAVIS'))
if not _TRAVIS:
    _QUICKTEST= True #Run a more limited set of tests

# Test whether the energy of simple orbits is conserved for different
# integrators
def test_energy_conservation():
    #Basic parameters for the test
    times= numpy.linspace(0.,280.,10001) #~10 Gyr at the Solar circle
    integrators= ['odeint','leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c',
                  'dopr54_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    if _TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -10.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    for p in pots:
        if _QUICKTEST and not 'NFW' in p: continue #For testing the test
        #Setup instance of potential
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        for integrator in integrators:
            o= setup_orbit_energy(tp)
            o.integrate(times,tp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            try:
                assert((numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol)
            except AssertionError:
                raise AssertionError("Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator))
#    raise AssertionError
    return None

# Test some long-term integrations for the symplectic integrators
def test_energy_symplec_longterm():
    #Basic parameters for the test
    times= numpy.linspace(0.,10000.,100001) #~360 Gyr at the Solar circle
    integrators= ['leapfrog_c', #don't do leapfrog, because it takes too long
                  'symplec4_c','symplec6_c']
    #Only use KeplerPotential
    #Grab all of the potentials
    pots= ['KeplerPotential']
    #tolerances in log10
    tol= {}
    tol['default']= -20.
    tol['leapfrog_c']= -16.
    tol['leapfrog']= -16.
    for p in pots:
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        for integrator in integrators:
            if integrator in tol.keys(): ttol= tol[integrator]
            else: ttol= tol['default']
            o= setup_orbit_energy(tp)
            o.integrate(times,tp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
#            print p, ((numpy.mean(o.E(times[0:20]))-numpy.mean(o.E(times[-20:-1])))/numpy.mean(tEs))**2.
            try:
                assert((numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol)
            except AssertionError:
                raise AssertionError("Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator))
            #Check whether there is a trend
            try:
                assert((numpy.mean(o.E(times[0:20]))-numpy.mean(o.E(times[-20:-1])))/numpy.mean(tEs))**2. < 10.**ttol
            except AssertionError:
                raise AssertionError("Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator))
#    raise AssertionError
    return None
   
# Test that the eccentricity of circular orbits is zero

# Test that the pericenter of orbits launched with vR=0 and vT > vc is the starting radius

# Test that the apocenter of orbits launched with vR=0 and vT < vc is the starting radius

# Test that the zmax of orbits launched with vz=0 is the starting height

# Test that vR of circular orbits is always zero

# Test the vT of circular orbits is always vc

# Test that the eccentricity, apo-, and pericenters of orbits calculated analytically agrees with the numerical calculation

# Check that adding a linear orbit to a planar orbit gives a FullOrbit

# Check that ER + Ez = E for orbits that stay close to the plane for the MWPotential

# Check that getOrbit returns the orbit properly (agrees with the input and with vR, ...)

# Check that toLiner and toPlanar work

def setup_orbit_energy(tp):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.linearPotential): 
        o= Orbit([1.,1.])
    elif isinstance(tp,potential.planarPotential): 
        o= Orbit([1.,1.1,1.1,0.])
    else:
        o= Orbit([1.,1.1,1.1,0.1,0.1,0.])
    return o
