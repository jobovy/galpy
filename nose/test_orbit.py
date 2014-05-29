##############################TESTS ON ORBITS##################################
import sys
import numpy
import os
from galpy import potential
from test_potential import testplanarMWPotential, testMWPotential
_TRAVIS= bool(os.getenv('TRAVIS'))
if not _TRAVIS:
    _QUICKTEST= True #Run a more limited set of tests
else:
    _QUICKTEST= True #Also do this for Travis, bc otherwise it takes too long
_NOLONGINTEGRATIONS= False

# Test whether the energy of simple orbits is conserved for different
# integrators
def test_energy_jacobi_conservation():
    if _NOLONGINTEGRATIONS: return None
    #Basic parameters for the test
    times= numpy.linspace(0.,280.,10001) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockFlatEllipticalDiskPotential')
    pots.append('mockFlatLopsidedDiskPotential')
    pots.append('mockFlatDehnenBarPotential')
    pots.append('mockFlatSteadyLogSpiralPotential')
    pots.append('mockFlatTransientLogSpiralPotential')
    pots.append('testMWPotential')
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
    jactol= {}
    jactol['default']= -10.
    jactol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    jactol['FlattenedPowerPotential']= -8. #these are more difficult
    jactol['mockFlatDehnenBarPotential']= -8. #these are more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        if p in jactol.keys(): tjactol= jactol[p]
        else: tjactol= jactol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
        for integrator in integrators:
            #First track azimuth
            o= setup_orbit_energy(tp,axi=False)
            if isinstance(tp,testplanarMWPotential):
                o.integrate(times,tp._potlist,method=integrator)
            else:
                o.integrate(times,tp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            if not 'DehnenBar' in p and not 'LogSpiral' in p:
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Jacobi
            if 'Elliptical' in p or 'Lopsided' in p:
                tJacobis= o.Jacobi(times,pot=tp)
            else:
                tJacobis= o.Jacobi(times)
#            print p, (numpy.std(tJacobis)/numpy.mean(tJacobis))**2.
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            if firstTest or 'MWPotential' in p:
                #Some basic checking of the energy and Jacobi functions
                assert (o.E(pot=None)-o.E(pot=tp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
                assert (o.E()-o.E(0.))**2. < 10.**ttol, \
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                assert (o.Jacobi(OmegaP=None)-o.Jacobi())**2. < 10.**ttol, \
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=tp))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=[tp]))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=[the Potential the orbit was integrated with] do not agree"
                if not tp.isNonAxi:
                    assert (o.Jacobi(OmegaP=1.)-o.Jacobi())**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                    assert (o.Jacobi(OmegaP=[0.,0.,1.])-o.Jacobi(OmegaP=1.))**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=[0,0,1] for axisymmetric potential is not equal to o.Jacobi with OmegaP=1"
                    assert (o.Jacobi(OmegaP=numpy.array([0.,0.,1.]))-o.Jacobi(OmegaP=1.))**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=[0,0,1] for axisymmetric potential is not equal to o.Jacobi with OmegaP=1"
                o= setup_orbit_energy(tp,axi=False)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.Jacobi() before the orbit was integrated did not throw an AttributeError")
            if ptp is None and tp.isNonAxi:
                if _QUICKTEST and not 'NFW' in p: break
                else: continue
            #Now do axisymmetric
            o= setup_orbit_energy(tp,axi=True)
            o.integrate(times,tp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Jacobi
            tJacobis= o.Jacobi(times)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            if firstTest or 'MWPotential' in p:
                #Some basic checking of the energy function
                assert (o.E(pot=None)-o.E(pot=tp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
                assert (o.E()-o.E(0.))**2. < 10.**ttol, \
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                assert (o.Jacobi(OmegaP=None)-o.Jacobi())**2. < 10.**ttol, \
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=tp))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=[tp]))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                if not tp.isNonAxi:
                    assert (o.Jacobi(OmegaP=1.)-o.Jacobi())**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                o= setup_orbit_energy(tp,axi=True)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.Jacobi() before the orbit was integrated did not throw an AttributeError")
            if ptp is None:
                if _QUICKTEST and not 'NFW' in p: break
                else: continue
            #Same for a planarPotential
#            print integrator
            o= setup_orbit_energy(ptp,axi=True)
            o.integrate(times,ptp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Jacobi
            tJacobis= o.Jacobi(times)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            if firstTest or 'MWPotential' in p:
                #Some basic checking of the energy function
                assert (o.E(pot=None)-o.E(pot=ptp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the planarPotential the orbit was integrated with do not agree for planarPotential"
                assert (o.E(pot=None)-o.E(pot=tp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree for planarPotential"
                assert (o.E()-o.E(0.))**2. < 10.**ttol, \
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                assert (o.Jacobi(OmegaP=None)-o.Jacobi())**2. < 10.**ttol, \
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=tp))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=[tp]))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                if not tp.isNonAxi:
                    assert (o.Jacobi(OmegaP=1.)-o.Jacobi())**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                o= setup_orbit_energy(ptp,axi=True)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.Jacobi() before the orbit was integrated did not throw an AttributeError")
            #Same for a planarPotential, track azimuth
            o= setup_orbit_energy(ptp,axi=False)
            o.integrate(times,ptp,method=integrator)
            tEs= o.E(times)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Jacobi
            tJacobis= o.Jacobi(times)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            if firstTest or 'MWPotential' in p:
                #Some basic checking of the energy function
                assert (o.E(pot=None)-o.E(pot=ptp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the planarPotential the orbit was integrated with do not agree for planarPotential"
                assert (o.E(pot=None)-o.E(pot=tp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree for planarPotential"
                assert (o.E()-o.E(0.))**2. < 10.**ttol, \
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                assert (o.Jacobi(OmegaP=None)-o.Jacobi())**2. < 10.**ttol, \
                    "o.Jacobi calculated with OmegaP=None is not equal to o.Jacobi"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=tp))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                assert (o.Jacobi(pot=None)-o.Jacobi(pot=[tp]))**2. < 10.**ttol, \
                    "o.Jacobi calculated with pot=None is not equal to o.Jacobi with pot=the Potential the orbit was integrated with do not agree"
                if not tp.isNonAxi:
                    assert (o.Jacobi(OmegaP=1.)-o.Jacobi())**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                o= setup_orbit_energy(ptp,axi=False)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
                try:
                    o.Jacobi()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.Jacobi() before the orbit was integrated did not throw an AttributeError")
                firstTest= False
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None

# Test some long-term integrations for the symplectic integrators
def test_energy_symplec_longterm():
    if _NOLONGINTEGRATIONS: return None
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
            assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                "Energy conservation during the orbit integration fails for potential %s and integrator %s by %.20f" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs))**2)
            #Check whether there is a trend
            linfit= numpy.polyfit(times,tEs,1)
#            print p
            assert linfit[0]**2. < 10.**ttol, \
                "Absence of secular trend in energy conservation fails for potential %s and symplectic integrator %s" %(p,integrator)
    #raise AssertionError
    return None
   
# Test that the eccentricity of circular orbits is zero
def test_eccentricity():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
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
    tol['default']= -16.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['NFWPotential']= -12. #these are more difficult
    firstTest= True
    for p in pots:
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
        ptp= tp.toPlanar()
        for integrator in integrators:
            #First do axi
            o= setup_orbit_eccentricity(tp,axi=True)
            if firstTest:
                try:
                    o.e() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.e() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tecc= o.e()
#            print p, integrator, tecc
            assert tecc**2. < 10.**ttol, \
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s" %(p,integrator)
            #add tracking azimuth
            o= setup_orbit_eccentricity(tp,axi=False)
            if firstTest:
                try:
                    o.e() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.e() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tecc= o.e()
#            print p, integrator, tecc
            assert tecc**2. < 10.**ttol, \
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential
#            print integrator
            o= setup_orbit_eccentricity(ptp,axi=True)
            if firstTest:
                try:
                    o.e() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.e() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,ptp,method=integrator)
            tecc= o.e()
#            print p, integrator, tecc
            assert tecc**2. < 10.**ttol, \
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential, track azimuth
            o= setup_orbit_eccentricity(ptp,axi=False)
            if firstTest:
                try:
                    o.e() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.e() before the orbit was integrated did not throw an AttributeError")
                firstTest= True
            o.integrate(times,ptp,method=integrator)
            tecc= o.e()
#            print p, integrator, tecc
            assert tecc**2. < 10.**ttol, \
                "Eccentricity of a circular orbit is not equal to zero for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None
    
# Test that the pericenter of orbits launched with vR=0 and vT > vc is the starting radius
def test_pericenter():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
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
    tol['default']= -16.
#    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
#    tol['NFWPotential']= -12. #these are more difficult
    firstTest= True
    for p in pots:
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
        ptp= tp.toPlanar()
        for integrator in integrators:
            #First do axi
            o= setup_orbit_pericenter(tp,axi=True)
            if firstTest:
                try:
                    o.rperi() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rperi() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tperi= o.rperi()
#            print p, integrator, tperi
            assert (tperi-o.R())**2. < 10.**ttol, \
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #add tracking azimuth
            o= setup_orbit_pericenter(tp,axi=False)
            if firstTest:
                try:
                    o.rperi() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rperi() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tperi= o.rperi()
#            print p, integrator, tperi
            assert (tperi-o.R())**2. < 10.**ttol, \
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential
#            print integrator
            o= setup_orbit_pericenter(ptp,axi=True)
            if firstTest:
                try:
                    o.rperi() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rperi() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,ptp,method=integrator)
            tperi= o.rperi()
#            print p, integrator, tperi
            assert (tperi-o.R())**2. < 10.**ttol, \
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential, track azimuth
            o= setup_orbit_pericenter(ptp,axi=False)
            if firstTest:
                try:
                    o.rperi() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rperi() before the orbit was integrated did not throw an AttributeError")
                firstTest= False
            o.integrate(times,ptp,method=integrator)
            tperi= o.rperi()
#            print p, integrator, tperi
            assert (tperi-o.R())**2. < 10.**ttol, \
                "Pericenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None

# Test that the apocenter of orbits launched with vR=0 and vT < vc is the starting radius
def test_apocenter():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    rmpots.append('FlattenedPowerPotential') #odd behavior, issue #148
    if _TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -16.
#    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
#    tol['NFWPotential']= -12. #these are more difficult
    firstTest= True
    for p in pots:
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
        ptp= tp.toPlanar()
        for integrator in integrators:
            #First do axi
            o= setup_orbit_apocenter(tp,axi=True)
            if firstTest:
                try:
                    o.rap() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rap() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tapo= o.rap()
#            print p, integrator, tapo
            assert (tapo-o.R())**2. < 10.**ttol, \
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #add tracking azimuth
            o= setup_orbit_apocenter(tp,axi=False)
            if firstTest:
                try:
                    o.rap() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rap() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tapo= o.rap()
#            print p, integrator, tapo
            assert (tapo-o.R())**2. < 10.**ttol, \
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential
#            print integrator
            o= setup_orbit_apocenter(ptp,axi=True)
            if firstTest:
                try:
                    o.rap() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rap() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,ptp,method=integrator)
            tapo= o.rap()
#            print p, integrator, tapo
            assert (tapo-o.R())**2. < 10.**ttol, \
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential, track azimuth
            o= setup_orbit_apocenter(ptp,axi=False)
            if firstTest:
                try:
                    o.rap() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.rap() before the orbit was integrated did not throw an AttributeError")
                firstTest= False
            o.integrate(times,ptp,method=integrator)
            tapo= o.rap()
#            print p, integrator, tapo
            assert (tapo-o.R())**2. < 10.**ttol, \
                "Apocenter radius for an orbit launched with vR=0 and vT > Vc is not equal to the initial radius for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None

# Test that the zmax of orbits launched with vz=0 is the starting height
def test_zmax():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    rmpots.append('FlattenedPowerPotential') #odd behavior, issue #148
    if _TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -16.
    tol['RazorThinExponentialDiskPotential']= -6. #these are more difficult
#    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    firstTest= True
    for p in pots:
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
        ptp= tp.toPlanar()
        for integrator in integrators:
            #First do axi
            o= setup_orbit_zmax(tp,axi=True)
            if firstTest:
                try:
                    o.zmax() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.zmax() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tzmax= o.zmax()
#            print p, integrator, tzmax
            assert (tzmax-o.z())**2. < 10.**ttol, \
                "Zmax for an orbit launched with vR=0 and vT > Vc is not equal to the initial height for potential %s and integrator %s" %(p,integrator)
            #add tracking azimuth
            o= setup_orbit_zmax(tp,axi=False)
            if firstTest:
                try:
                    o.zmax() #This should throw an AttributeError
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.zmax() before the orbit was integrated did not throw an AttributeError")
            o.integrate(times,tp,method=integrator)
            tzmax= o.zmax()
#            print p, integrator, tzmax
            assert (tzmax-o.z())**2. < 10.**ttol, \
                "Zmax for an orbit launched with vR=0 and vT > Vc is not equal to the initial height for potential %s and integrator %s" %(p,integrator)
            if firstTest:
                ptp= tp.toPlanar()
                o= setup_orbit_energy(ptp,axi=False)
                try:
                    o.zmax() #This should throw an AttributeError, bc there is no zmax
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.zmax() for a planarOrbit did not throw an AttributeError")
                o= setup_orbit_energy(ptp,axi=True)
                try:
                    o.zmax() #This should throw an AttributeError, bc there is no zmax
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.zmax() for a planarROrbit did not throw an AttributeError")
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None

# Test that vR of circular orbits is always zero

# Test the vT of circular orbits is always vc

# Test that the eccentricity, apo-, and pericenters of orbits calculated analytically agrees with the numerical calculation
def test_analytic_ecc_rperi_rap():
    #Basic parameters for the test
    times= numpy.linspace(0.,20.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    rmpots.append('FlattenedPowerPotential') #odd behavior, issue #148
    if _TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -10.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['IsochronePotential']= -6. #these are more difficult
    tol['JaffePotential']= -6. #these are more difficult
    tol['PowerSphericalPotential']= -8. #these are more difficult
    tol['PowerSphericalPotentialwCutoff']= -8. #these are more difficult
    for p in pots:
        #Setup instance of potential
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        if p == 'MWPotential':
            tp= potential.MWPotential
            ptp= [ttp.toPlanar() for ttp in tp]
        else:
            try:
                tclass= getattr(potential,p)
            except AttributeError:
                tclass= getattr(sys.modules[__name__],p)
            tp= tclass()
            if not hasattr(tp,'normalize'): continue #skip these
            tp.normalize(1.)
            ptp= tp.toPlanar()
        for integrator in integrators:
            for ii in range(4):
                if ii == 0: #axi, full
                    #First do axi
                    o= setup_orbit_analytic(tp,axi=True)
                    o.integrate(times,tp,method=integrator)
                elif ii == 1: #track azimuth, full
                    #First do axi
                    o= setup_orbit_analytic(tp,axi=False)
                    o.integrate(times,tp,method=integrator)
                elif ii == 2: #axi, planar
                    #First do axi
                    o= setup_orbit_analytic(ptp,axi=True)
                    o.integrate(times,ptp,method=integrator)
                elif ii == 3: #track azimuth, full
                    #First do axi
                    o= setup_orbit_analytic(ptp,axi=False)
                    o.integrate(times,ptp,method=integrator)
                #Eccentricity
                tecc= o.e()
                tecc_analytic= o.e(analytic=True)
#                print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc-tecc_analytic)**2. < 10.**ttol, \
                    "Analytically computed eccentricity does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                #Pericenter radius
                trperi= o.rperi()
                trperi_analytic= o.rperi(analytic=True)
#                print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi-trperi_analytic)**2. < 10.**ttol, \
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                #Apocenter radius
                trap= o.rap()
                trap_analytic= o.rap(analytic=True)
#                print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap-trap_analytic)**2. < 10.**ttol, \
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None
    
# Check that zmax calculated analytically agrees with numerical calculation
def test_analytic_zmax():
    #Basic parameters for the test
    times= numpy.linspace(0.,20.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    rmpots.append('FlattenedPowerPotential') #odd behavior, issue #148
    if _TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -10.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['RazorThinExponentialDiskPotential']= -4. #these are more difficult
    tol['HernquistPotential']= -8. #these are more difficult
    tol['JaffePotential']= -8. #these are more difficult
    tol['MiyamotoNagaiPotential']= -7. #these are more difficult
    tol['LogarithmicHaloPotential']= -7. #these are more difficult
    tol['KeplerPotential']= -7. #these are more difficult
    tol['PowerSphericalPotentialwCutoff']= -8. #these are more difficult
    for p in pots:
        #Setup instance of potential
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        if p == 'MWPotential':
            tp= potential.MWPotential
        else:
            try:
                tclass= getattr(potential,p)
            except AttributeError:
                tclass= getattr(sys.modules[__name__],p)
            tp= tclass()
            if not hasattr(tp,'normalize'): continue #skip these
            tp.normalize(1.)
        for integrator in integrators:
            for ii in range(2):
                if ii == 0: #axi, full
                    #First do axi
                    o= setup_orbit_analytic_zmax(tp,axi=True)
                elif ii == 1: #track azimuth, full
                    #First do axi
                    o= setup_orbit_analytic_zmax(tp,axi=False)
                o.integrate(times,tp,method=integrator)
                tzmax= o.zmax()
                tzmax_analytic= o.zmax(analytic=True)
#                print p, integrator, tzmax, tzmax_analytic, (tzmax-tzmax_analytic)**2.
                assert (tzmax-tzmax_analytic)**2. < 10.**ttol, \
                    "Analytically computed zmax does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not 'NFW' in p: break
    #raise AssertionError
    return None

# Check that adding a linear orbit to a planar orbit gives a FullOrbit
def test_add_linear_planar_orbit():
    from galpy.orbit_src import FullOrbit, RZOrbit
    kg= potential.KGPotential()
    ol= setup_orbit_energy(kg)
    #w/ azimuth
    plp= potential.NFWPotential().toPlanar()
    op= setup_orbit_energy(plp)
    of= ol+op
    assert isinstance(of._orb,FullOrbit.FullOrbit), \
        "Sum of linearOrbit and planarOrbit does not give a FullOrbit"
    #w/o azimuth
    op= setup_orbit_energy(plp,axi=True)
    of= ol+op
    assert isinstance(of._orb,RZOrbit.RZOrbit), \
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    return None

# Check that ER + Ez = E and that ER and EZ are separately conserved for orbits that stay close to the plane for the MWPotential
def test_ER_EZ():
    from galpy.potential import MWPotential
    ona= setup_orbit_analytic_EREz(MWPotential,axi=False)
    oa= setup_orbit_analytic_EREz(MWPotential,axi=True)
    os= [ona,oa]
    for o in os:
        times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
        o.integrate(times,MWPotential)
        ERs= o.ER(times)
        Ezs= o.Ez(times)
        ERdiff= numpy.fabs(numpy.std(ERs-numpy.mean(ERs))/numpy.mean(ERs))
        assert ERdiff < 10.**-4., \
            'ER conservation for orbits close to the plane in MWPotential fails at %g%%' % (100.*ERdiff)
        Ezdiff= numpy.fabs(numpy.std(Ezs-numpy.mean(Ezs))/numpy.mean(Ezs))
        assert Ezdiff < 10.**-1.7, \
            'Ez conservation for orbits close to the plane in MWPotential fails at %g%%' % (100.*Ezdiff)
        #Some basic checking
        assert numpy.fabs(o.ER()-o.ER(pot=MWPotential)) < 10.**-16., \
            'o.ER() not equal to o.ER(pot=)'
        assert numpy.fabs(o.Ez()-o.Ez(pot=MWPotential)) < 10.**-16., \
            'o.ER() not equal to o.Ez(pot=)'
        assert numpy.fabs(o.ER(pot=None)-o.ER(pot=MWPotential)) < 10.**-16., \
            'o.ER() not equal to o.ER(pot=)'
        assert numpy.fabs(o.Ez(pot=None)-o.Ez(pot=MWPotential)) < 10.**-16., \
            'o.ER() not equal to o.Ez(pot=)'
    o= setup_orbit_analytic_EREz(MWPotential,axi=False)
    try:
        o.Ez()
    except AttributeError:
        pass
    else:
        raise AssertionError('o.Ez() w/o potential before the orbit was integrated did not raise AttributeError')
    try:
        o.ER()
    except AttributeError:
        pass
    else:
        raise AssertionError('o.ER() w/o potential before the orbit was integrated did not raise AttributeError')
    o= setup_orbit_analytic_EREz(MWPotential,axi=True)
    try:
        o.Ez()
    except AttributeError:
        pass
    else:
        raise AssertionError('o.Ez() w/o potential before the orbit was integrated did not raise AttributeError')
    try:
        o.ER()
    except AttributeError:
        pass
    else:
        raise AssertionError('o.ER() w/o potential before the orbit was integrated did not raise AttributeError')
    return None

# Check that the different setups work
def test_orbit_setup_linear():
    from galpy.orbit import Orbit
    # linearOrbit
    o= Orbit([1.,0.1])
    assert o.dim() == 1, 'linearOrbit does not have dim == 1'
    assert numpy.fabs(o.x()-1.) < 10.**-16., 'linearOrbit x setup does not agree with o.x()'
    assert numpy.fabs(o.vx()-0.1) < 10.**-16., 'linearOrbit vx setup does not agree with o.vx()'
    try:
        o.setphi(3.)
    except AttributeError:
        pass
    else:
        raise AssertionError('setphi applied to linearOrbit did not raise AttributeError')
    return None

def test_orbit_setup_planar():
    from galpy.orbit import Orbit
    from galpy.orbit_src.planarOrbit import planarROrbit
    o= Orbit([1.,0.1,1.1])
    assert o.dim() == 2, 'planarROrbit does not have dim == 2'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'planarOrbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'planarOrbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'planarOrbit vT setup does not agree with o.vT()'
    o.setphi(3.)
    assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    assert not isinstance(o._orb,planarROrbit), 'After applying setphi, planarROrbit did not become planarOrbit'
    o= Orbit([1.,0.1,1.1,2.])
    assert o.dim() == 2, 'planarOrbit does not have dim == 2'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'planarOrbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'planarOrbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'planarOrbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.phi()-2.) < 10.**-16., 'planarOrbit phi setup does not agree with o.phi()'
    o.setphi(3.)
    assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    #lb, plane w/ default
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.])
    obs= [8.5,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.5,0.,-10.,245.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/235.,239./235,0.])
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit in the plane
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/235.,239./235,0.,0.,0.])
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    return None

def test_orbit_setup():
    from galpy.orbit import Orbit
    from galpy.orbit_src.FullOrbit import FullOrbit
    o= Orbit([1.,0.1,1.1,0.2,0.3])
    assert o.dim() == 3, 'RZOrbitOrbit does not have dim == 3'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'Orbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'Orbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.vphi()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vphi()'
    assert numpy.fabs(o.z()-0.2) < 10.**-16., 'Orbit z setup does not agree with o.z()'
    assert numpy.fabs(o.vz()-0.3) < 10.**-16., 'Orbit vz setup does not agree with o.vz()'
    o.setphi(3.)
    assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    assert isinstance(o._orb,FullOrbit), 'After applying setphi, RZOrbit did not become FullOrbit'
    o= Orbit([1.,0.1,1.1,0.2,0.3,2.])
    assert o.dim() == 3, 'FullOrbit does not have dim == 3'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'Orbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'Orbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.z()-0.2) < 10.**-16., 'Orbit z setup does not agree with o.z()'
    assert numpy.fabs(o.vz()-0.3) < 10.**-16., 'Orbit vz setup does not agree with o.vz()'
    assert numpy.fabs(o.phi()-2.) < 10.**-16., 'Orbit phi setup does not agree with o.phi()'
    o.setphi(3.)
    assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    #Radec w/ default
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    assert numpy.fabs(o.ra()-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #Radec w/ hogg
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='hogg')
    assert numpy.fabs(o.ra()-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #Radec w/ dehnen and diff ro,vo
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='dehnen',vo=220.,
             ro=8.,zo=0.01)
    obs= [8.,0.,0.01,-10.,225.25,7.17]
    assert numpy.fabs(o.ra(obs=obs,ro=8.)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=8.)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=8.)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=8.,vo=220.)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=8.,vo=220.)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=8.,vo=220.)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #Radec w/ schoenrich and diff ro,vo
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='schoenrich',
             vo=240.,ro=7.5,zo=0.035)
    obs= [7.5,0.,0.035,-11.1,252.24,7.25]
    assert numpy.fabs(o.ra(obs=obs,ro=7.5)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=7.5)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=7.5)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=7.5,vo=240.)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=7.5,vo=240.)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=7.5,vo=240.)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #Radec w/ custom solarmotion and diff ro,vo
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion=[10.,20.,15.],
             vo=220.,ro=7.5,zo=0.035)
    obs= [7.5,0.,0.035,10.,240.,15.]
    assert numpy.fabs(o.ra(obs=obs,ro=7.5)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=7.5)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=7.5)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=7.5,vo=220.)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=7.5,vo=220.)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=7.5,vo=220.)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb w/ default
    o= Orbit([120.,60.,2.,0.5,0.4,30.],lb=True)
    assert numpy.fabs(o.ll()-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-60.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll()-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb()-0.4) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb w/ default at the Sun
    o= Orbit([120.,60.,0.,10.,20.,30.],uvw=True,lb=True)
    assert numpy.fabs(o.dist()-0.) < 10.**-2., 'Orbit dist setup does not agree with o.dist()' #because of tweak in the code to deal with at the Sun
    assert (o.U()**2.+o.V()**2.+o.W()**2.-10.**2.-20.**2.-30.**2.) < 10.**-10., 'Velocity wrt the Sun when looking at Orbit at the Sun does not agree'
    assert (o.vlos()**2.-10.**2.-20.**2.-30.**2.) < 10.**-10., 'Velocity wrt the Sun when looking at Orbit at the Sun does not agree'
    #lb w/ default and UVW
    o= Orbit([120.,60.,2.,-10.,20.,-25.],lb=True,uvw=True)
    assert numpy.fabs(o.ll()-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-60.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.U()+10.) < 10.**-13., 'Orbit U setup does not agree with o.U()'
    assert numpy.fabs(o.V()-20.) < 10.**-13., 'Orbit V setup does not agree with o.V()'
    assert numpy.fabs(o.W()+25.) < 10.**-13., 'Orbit W setup does not agree with o.W()'
    #Radec w/ default and obs=Orbit
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    obs= Orbit([1.,-10.1/235.,239./235,0.025/8.5,6.7/235.,0.])
    assert numpy.fabs(o.ra(obs=obs)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb, plane w/ default
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.])
    obs= [8.5,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.5,0.,-10.,245.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/235.,239./235,0.])
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    return None

# Check that getOrbit returns the orbit properly (agrees with the input and with vR, ...)

# Check that toLinear and toPlanar work

# Check that some relevant errors are being raised
def test_attributeerrors():
    from galpy.orbit import Orbit
    #Vertical functions for planarOrbits
    o= Orbit([1.,0.1,1.,0.1])
    try:
        o.z()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.z() for planarOrbit should have raised AttributeError, but did not")
    try:
        o.vz()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vz() for planarOrbit should have raised AttributeError, but did not")
    #phi, x, y, vx, vy for Orbits that don't track phi
    o= Orbit([1.,0.1,1.1,0.1,0.2])
    try:
        o.phi()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.phi() for RZOrbit should have raised AttributeError, but did not")
    try:
        o.x()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.x() for RZOrbit should have raised AttributeError, but did not")
    try:
        o.y()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.y() for RZOrbit should have raised AttributeError, but did not")
    try:
        o.vx()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vx() for RZOrbit should have raised AttributeError, but did not")
    try:
        o.vy()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vy() for RZOrbit should have raised AttributeError, but did not")
    o= Orbit([1.,0.1,1.1])
    try:
        o.phi()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.phi() for planarROrbit should have raised AttributeError, but did not")
    try:
        o.x()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.x() for planarROrbit should have raised AttributeError, but did not")
    try:
        o.y()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.y() for planarROrbit should have raised AttributeError, but did not")
    try:
        o.vx()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vx() for planarROrbit should have raised AttributeError, but did not")
    try:
        o.vy()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vy() for planarROrbit should have raised AttributeError, but did not")
    return None

# Check plotting routines

# Setup the orbit for the energy test
def setup_orbit_energy(tp,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.linearPotential): 
        o= Orbit([1.,1.])
    elif isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,1.1,1.1])
        else:
            o= Orbit([1.,1.1,1.1,0.])
    else:
        if axi:
            o= Orbit([1.,1.1,1.1,0.1,0.1])
        else:
            o= Orbit([1.,1.1,1.1,0.1,0.1,0.])
    return o

# Setup the orbit for the eccentricity test
def setup_orbit_eccentricity(tp,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,0.,1.])
        else:
            o= Orbit([1.,0.,1.,0.])
    else:
        if axi:
            o= Orbit([1.,0.,1.,0.,0.])
        else:
            o= Orbit([1.,0.,1.,0.,0.,0.])
    return o

# Setup the orbit for the pericenter test
def setup_orbit_pericenter(tp,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,0.,1.1])
        else:
            o= Orbit([1.,0.,1.1,0.])
    else:
        if axi:
            o= Orbit([1.,0.,1.1,0.,0.])
        else:
            o= Orbit([1.,0.,1.1,0.,0.,0.])
    return o

# Setup the orbit for the apocenter test
def setup_orbit_apocenter(tp,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,0.,0.9])
        else:
            o= Orbit([1.,0.,0.9,0.])
    else:
        if axi:
            o= Orbit([1.,0.,0.9,0.,0.])
        else:
            o= Orbit([1.,0.,0.9,0.,0.,0.])
    return o

# Setup the orbit for the zmax test
def setup_orbit_zmax(tp,axi=False):
    from galpy.orbit import Orbit
    if axi:
        o= Orbit([1.,0.,0.98,0.05,0.])
    else:
        o= Orbit([1.,0.,0.98,0.05,0.,0.])
    return o

# Setup the orbit for the apocenter test
def setup_orbit_analytic(tp,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,0.1,0.9])
        else:
            o= Orbit([1.,0.1,0.9,0.])
    else:
        if axi:
            o= Orbit([1.,0.1,0.9,0.,0.])
        else:
            o= Orbit([1.,0.1,0.9,0.,0.,0.])
    return o

# Setup the orbit for the zmax test
def setup_orbit_analytic_zmax(tp,axi=False):
    from galpy.orbit import Orbit
    if axi:
        o= Orbit([1.,0.,1.,0.05,0.03])
    else:
        o= Orbit([1.,0.,1.,0.05,0.03,0.0])
    return o

# Setup the orbit for the ER, EZ test
def setup_orbit_analytic_EREz(tp,axi=False):
    from galpy.orbit import Orbit
    if axi:
        o= Orbit([1.,0.03,1.,0.05,0.03])
    else:
        o= Orbit([1.,0.03,1.,0.05,0.03,0.0])
    return o

class mockFlatEllipticalDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.EllipticalDiskPotential(phib=numpy.pi/2.,p=0.,tform=None,tsteady=None,twophio=14./220.)])
    def OmegaP(self):
        return 0.
class mockFlatLopsidedDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.LopsidedDiskPotential(phib=numpy.pi/2.,p=0.,tform=None,tsteady=None,phio=10./220.)])
    def OmegaP(self):
        return 0.
class mockFlatDehnenBarPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.DehnenBarPotential()])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockFlatSteadyLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.SteadyLogSpiralPotential()])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockFlatTransientLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.TransientLogSpiralPotential(to=-10.)]) #this way, it's basically a steady spiral
    def OmegaP(self):
        return self._potlist[1].OmegaP()
