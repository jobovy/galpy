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
                assert (o.rperi(ro=8.)/8.-trperi_analytic)**2. < 10.**ttol, \
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Apocenter radius
                trap= o.rap()
                trap_analytic= o.rap(analytic=True)
#                print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap-trap_analytic)**2. < 10.**ttol, \
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                assert (o.rap(ro=8.)/8.-trap_analytic)**2. < 10.**ttol, \
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
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
                assert (o.zmax(ro=8.)/8.-tzmax_analytic)**2. < 10.**ttol, \
                    "Zmax in physical coordinates does not agree with physical-scale times zmax in normalized coordinates for potential %s and integrator %s" %(p,integrator)
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
    of= op+ol
    assert isinstance(of._orb,FullOrbit.FullOrbit), \
        "Sum of linearOrbit and planarOrbit does not give a FullOrbit"
    #w/o azimuth
    op= setup_orbit_energy(plp,axi=True)
    of= ol+op
    assert isinstance(of._orb,RZOrbit.RZOrbit), \
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    of= op+ol
    assert isinstance(of._orb,RZOrbit.RZOrbit), \
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    # op + op shouldn't work
    try:
        of= op+op
    except AttributeError:
        pass
    else:
        raise AssertionError('Adding a planarOrbit to a planarOrbit did not raise AttributeError')
    return None

# Check that pickling orbits works
def test_pickle():
    import pickle
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.2,2.])
    po= pickle.dumps(o)
    upo= pickle.loads(po)
    assert o.R() == upo.R(), "Pickled/unpickled orbit does not agree with original orbut for R"
    assert o.vR() == upo.vR(), "Pickled/unpickled orbit does not agree with original orbut for vR"
    assert o.vT() == upo.vT(), "Pickled/unpickled orbit does not agree with original orbut for vT"
    assert o.z() == upo.z(), "Pickled/unpickled orbit does not agree with original orbut for z"
    assert o.vz() == upo.vz(), "Pickled/unpickled orbit does not agree with original orbut for vz"
    assert o.phi() == upo.phi(), "Pickled/unpickled orbit does not agree with original orbut for phi"
    return None

# Basic checks of the angular momentum function
def test_angularmomentum():
    from galpy.orbit import Orbit
    # Shouldn't work for a 1D orbit
    o= Orbit([1.,0.1])
    try:
        o.L()
    except AttributeError:
        pass
    else:
        raise AssertionError('Orbit.L() for linearOrbit did not raise AttributeError')
    # Also shouldn't work for an RZOrbit
    o= Orbit([1.,0.1,1.1,0.1,0.2])
    try:
        o.L()
    except AttributeError:
        pass
    else:
        raise AssertionError('Orbit.L() for RZOrbit did not raise AttributeError')
    # For a planarROrbit, should return Lz
    o= Orbit([1.,0.1,1.1])
    assert len(o.L()) == 1, "planarOrbit's angular momentum isn't 1D"
    assert o.L() == 1.1, "planarOrbit's angular momentum isn't correct"
    # If Omega is given, then it should be subtracted
    times= numpy.linspace(0.,2.,51)
    from galpy.potential import MWPotential
    o.integrate(times,MWPotential)
    assert numpy.fabs(o.L(t=1.,Omega=1.)-0.1) < 10.**-16., 'o.L() w/ Omega does not work'
    # For a FullOrbit, angular momentum should be 3D
    o= Orbit([1.,0.1,1.1,0.1,0.,numpy.pi/2.])
    assert o.L().shape[1] == 3, "FullOrbit's angular momentum is not 3D"
    assert numpy.fabs(o.L()[0,2]-1.1) < 10.**-16., "FullOrbit's Lz is not correct"
    assert numpy.fabs(o.L()[0,0]+0.01) < 10.**-16., "FullOrbit's Lx is not correct"
    assert numpy.fabs(o.L()[0,1]+0.11) < 10.**-16., "FullOrbit's Ly is not correct"
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
    obs= [8.,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.,0.,-10.,230.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #also check that the ro,vo,solarmotion values are stored and used properly (issue #158 solution)
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.],
             ro=7.5)
    assert numpy.fabs(o.ll()-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.5,0.,-10.,245.]
    assert numpy.fabs(o.pmll()-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb()-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.])
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit in the plane
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.,0.,0.])
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
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='dehnen',vo=240.,
             ro=7.5,zo=0.01)
    obs= [7.5,0.,0.01,-10.,245.25,7.17]
    assert numpy.fabs(o.ra(obs=obs,ro=7.5)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=7.5)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=7.5)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=7.5,vo=240.)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=7.5,vo=240.)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=7.5,vo=240.)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #also check that the ro,vo,solarmotion values are stored and used properly (issue #158 solution)
    assert numpy.fabs(o.ra()-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
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
             vo=240.,ro=7.5,zo=0.035)
    obs= [7.5,0.,0.035,10.,260.,15.]
    assert numpy.fabs(o.ra(obs=obs,ro=7.5)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=7.5)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=7.5)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=7.5,vo=240.)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=7.5,vo=240.)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=7.5,vo=240.)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb w/ default
    o= Orbit([120.,60.,2.,0.5,0.4,30.],lb=True)
    assert numpy.fabs(o.ll()-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-60.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll()-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vll()-4.74047) < 10.**-13., 'Orbit pmll setup does not agree with o.vll()'
    assert numpy.fabs(o.pmbb()-0.4) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vbb()-0.8*4.74047) < 10.**-13., 'Orbit pmbb setup does not agree with o.vbb()'
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
    #lb w/ default and UVW, test wrt helioXYZ
    o= Orbit([180.,0.,2.,-10.,20.,-25.],lb=True,uvw=True)
    assert numpy.fabs(o.helioX()+2./8.) < 10.**-13., 'Orbit ll setup does not agree with o.helioX()'
    assert numpy.fabs(o.helioY()-0.) < 10.**-13., 'Orbit bb setup does not agree with o.helioY()'
    assert numpy.fabs(o.helioZ()-0.) < 10.**-13., 'Orbit dist setup does not agree with o.helioZ()'
    assert numpy.fabs(o.U()+10.) < 10.**-13., 'Orbit U setup does not agree with o.U()'
    assert numpy.fabs(o.V()-20.) < 10.**-13., 'Orbit V setup does not agree with o.V()'
    assert numpy.fabs(o.W()+25.) < 10.**-13., 'Orbit W setup does not agree with o.W()'
    #Radec w/ default and obs=Orbit
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    obs= Orbit([1.,-10.1/220.,224./220,0.025/8.,6.7/220.,0.])
    assert numpy.fabs(o.ra(obs=obs)-120.) < 10.**-13., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs)-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs)-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.vra(obs=obs)-4.74047) < 10.**-13., 'Orbit pmra setup does not agree with o.vra()'
    assert numpy.fabs(o.pmdec(obs=obs)-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vdec(obs=obs)-0.8*4.74047) < 10.**-13., 'Orbit pmdec setup does not agree with o.vdec()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb, plane w/ default
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.])
    obs= [8.,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.,0.,-10.,230.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.])
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-13., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-13., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-13., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-13., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    return None

# Check that toPlanar works
def test_toPlanar():
    from galpy.orbit import Orbit
    obs= Orbit([1.,0.1,1.1,0.3,0.,2.])
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for FullOrbit'
    assert obsp.R() == obs.R(), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert obsp.vR() == obs.vR(), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert obsp.vT() == obs.vT(), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    assert obsp.phi() == obs.phi(), 'Planar orbit generated w/ toPlanar does not have the correct phi'
    obs= Orbit([1.,0.1,1.1,0.3,0.])
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for RZOrbit'
    assert obsp.R() == obs.R(), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert obsp.vR() == obs.vR(), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert obsp.vT() == obs.vT(), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    obs= Orbit([1.,0.1,1.1,2.])
    try:
        obs.toPlanar()
    except AttributeError:
        pass
    else:
        raise AttributeError('toPlanar() applied to a planar Orbit did not raise an AttributeError')        
    return None

# Check that toLinear works
def test_toLinear():
    from galpy.orbit import Orbit
    obs= Orbit([1.,0.1,1.1,0.3,0.,2.])
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinwar does not generate an Orbit w/ dim=1 for FullOrbit'
    assert obsl.x() == obs.z(), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert obsl.vx() == obs.vz(), 'Linear orbit generated w/ toLinear does not have the correct vx'
    obs= Orbit([1.,0.1,1.1,0.3,0.])
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinear does not generate an Orbit w/ dim=1 for FullOrbit'
    assert obsl.x() == obs.z(), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert obsl.vx() == obs.vz(), 'Linear orbit generated w/ toLinear does not have the correct vx'
    obs= Orbit([1.,0.1,1.1,2.])
    try:
        obs.toLinear()
    except AttributeError:
        pass
    else:
        raise AttributeError('toLinear() applied to a planar Orbit did not raise an AttributeError')        
    return None

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

# Test reversing an orbit
def test_reverse():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([1.,0.1,1.2,0.3,0.2,2.])
    times= numpy.linspace(0.,7.,251)
    o.integrate(times,lp)
    Rs= o.R(times)
    vRs= o.vR(times)
    vTs= o.vT(times)
    zs= o.z(times)
    vzs= o.vz(times)
    phis= o.phi(times)
    o.reverse()
    assert numpy.all(numpy.fabs(Rs-o.R(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.R'
    assert numpy.all(numpy.fabs(vRs-o.vR(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.vR'
    assert numpy.all(numpy.fabs(vTs-o.vT(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.vT'
    assert numpy.all(numpy.fabs(zs-o.z(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.z'
    assert numpy.all(numpy.fabs(vzs-o.vz(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.vz'
    assert numpy.all(numpy.fabs(phis-o.phi(times)[::-1])) < 10.**-16., \
        'Orbit.reverse does not work as expected for o.phi'
    return None

# test getOrbit
def test_getOrbit():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([1.,0.1,1.2,0.3,0.2,2.])
    times= numpy.linspace(0.,7.,251)
    o.integrate(times,lp)
    Rs= o.R(times)
    vRs= o.vR(times)
    vTs= o.vT(times)
    zs= o.z(times)
    vzs= o.vz(times)
    phis= o.phi(times)
    orbarray= o.getOrbit()
    assert numpy.all(numpy.fabs(Rs-orbarray[:,0])) < 10.**-16., \
        'getOrbit does not work as expected for R'
    assert numpy.all(numpy.fabs(vRs-orbarray[:,1])) < 10.**-16., \
        'getOrbit does not work as expected for vR'
    assert numpy.all(numpy.fabs(vTs-orbarray[:,2])) < 10.**-16., \
        'getOrbit does not work as expected for vT'
    assert numpy.all(numpy.fabs(zs-orbarray[:,3])) < 10.**-16., \
        'getOrbit does not work as expected for z'
    assert numpy.all(numpy.fabs(vzs-orbarray[:,4])) < 10.**-16., \
        'getOrbit does not work as expected for vz'
    assert numpy.all(numpy.fabs(phis-orbarray[:,5])) < 10.**-16., \
        'getOrbit does not work as expected for phi'
    return None

# Check the routines that should return physical coordinates
def test_physical_output():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.)
    plp= lp.toPlanar()
    for ii in range(4):
        ro, vo= 7., 200.
        if ii == 0: #axi, full
            o= setup_orbit_physical(lp,axi=True,ro=ro,vo=vo)
        elif ii == 1: #track azimuth, full
            o= setup_orbit_physical(lp,axi=False,ro=ro,vo=vo)
        elif ii == 2: #axi, planar
            o= setup_orbit_physical(plp,axi=True,ro=ro,vo=vo)
        elif ii == 3: #track azimuth, full
            o= setup_orbit_physical(plp,axi=False,ro=ro,vo=vo)
        #Test positions
        assert numpy.fabs(o.R()/ro-o.R(use_physical=False)) < 10.**-10., 'o.R() output for Orbit setup with ro= does not work as expected'
        if ii % 2 == 1:
            assert numpy.fabs(o.x()/ro-o.x(use_physical=False)) < 10.**-10., 'o.x() output for Orbit setup with ro= does not work as expected'
            assert numpy.fabs(o.y()/ro-o.y(use_physical=False)) < 10.**-10., 'o.y() output for Orbit setup with ro= does not work as expected'
        if ii < 2:
            assert numpy.fabs(o.z()/ro-o.z(use_physical=False)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected'
        #Test velocities
        assert numpy.fabs(o.vR()/vo-o.vR(use_physical=False)) < 10.**-10., 'o.vR() output for Orbit setup with ro= does not work as expected'
        assert numpy.fabs(o.vT()/vo-o.vT(use_physical=False)) < 10.**-10., 'o.vT() output for Orbit setup with ro= does not work as expected'
        assert numpy.fabs(o.vphi()/vo-o.vphi(use_physical=False)) < 10.**-10., 'o.vphi() output for Orbit setup with ro= does not work as expected'
        if ii % 2 == 1:
            assert numpy.fabs(o.vx()/vo-o.vx(use_physical=False)) < 10.**-10., 'o.vx() output for Orbit setup with ro= does not work as expected'
            assert numpy.fabs(o.vy()/vo-o.vy(use_physical=False)) < 10.**-10., 'o.vy() output for Orbit setup with ro= does not work as expected'
        if ii < 2:
            assert numpy.fabs(o.vz()/vo-o.vz(use_physical=False)) < 10.**-10., 'o.vz() output for Orbit setup with ro= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    assert numpy.fabs((o.time(1.,ro=ro,vo=vo)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    assert numpy.fabs((o.time(1.,use_physical=False)-1.)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    return None

# Check plotting routines
def test_planar_plotting():
    from galpy.orbit import Orbit
    from galpy.potential_src.planarPotential import RZToplanarPotential
    o= Orbit([1.,0.1,1.1,2.])
    oa= Orbit([1.,0.1,1.1])
    times= numpy.linspace(0.,7.,251)
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.8)
    try: o.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotJacobi()
    except AttributeError: pass
    else: raise AssertionError('o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: oa.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarROrbit')
    try: oa.plotJacobi()
    except AttributeError: pass
    else: raise AssertionError('o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarROrbit')
    # Integrate
    o.integrate(times,lp)
    oa.integrate(times,lp)
    # Energy
    o.plotE()
    o.plotE(pot=lp,d1='R')
    o.plotE(pot=lp,d1='vR')
    o.plotE(pot=lp,d1='phi')
    o.plotE(pot=[lp,RZToplanarPotential(lp)],d1='vT')
    oa.plotE()
    oa.plotE(pot=lp,d1='R')
    oa.plotE(pot=lp,d1='vR')
    oa.plotE(pot=[lp,RZToplanarPotential(lp)],d1='vT')
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    o.plotJacobi(pot=lp,d1='vR')
    o.plotJacobi(pot=lp,d1='phi')
    o.plotJacobi(pot=[lp,RZToplanarPotential(lp)],d1='vT')
    oa.plotJacobi()
    oa.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    oa.plotJacobi(pot=lp,d1='vR')
    oa.plotJacobi(pot=[lp,RZToplanarPotential(lp)],d1='vT')
    # Plot the orbit itself, defaults
    o.plot()
    o.plot(ro=8.)
    oa.plot()
    o.plotx(d1='vx')
    o.plotvx(d1='y')
    o.ploty(d1='vy')
    o.plotvy(d1='x')
    # Plot the orbit itself in 3D, defaults
    o.plot3d()
    o.plot3d(ro=8.)
    oa.plot3d()
    o.plot3d(d1='x',d2='vx',d3='y')
    o.plot3d(d1='vx',d2='y',d3='vy')
    o.plot3d(d1='y',d2='vy',d3='x')
    o.plot3d(d1='vy',d2='x',d3='vx')
    return None

# Check plotting routines
def test_full_plotting():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.2,2.])
    oa= Orbit([1.,0.1,1.1,0.1,0.2])
    times= numpy.linspace(0.,7.,251)
    from galpy.potential import LogarithmicHaloPotential
    if not _TRAVIS:
        from galpy.potential import DoubleExponentialDiskPotential
        dp= DoubleExponentialDiskPotential(normalize=1.)
    lp= LogarithmicHaloPotential(normalize=1.,q=0.8)
    try: o.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotEz()
    except AttributeError: pass
    else: raise AssertionError('o.plotEz() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotEzJz()
    except AttributeError: pass
    else: raise AssertionError('o.plotJzEz() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotJacobi()
    except AttributeError: pass
    else: raise AssertionError('o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: oa.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarROrbit')
    try: oa.plotEz()
    except AttributeError: pass
    else: raise AssertionError('o.plotEz() before the orbit was integrated did not raise AttributeError for planarROrbit')
    try: oa.plotEzJz()
    except AttributeError: pass
    else: raise AssertionError('o.plotEzJz() before the orbit was integrated did not raise AttributeError for planarROrbit')
    try: oa.plotJacobi()
    except AttributeError: pass
    else: raise AssertionError('o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarROrbit')
    # Integrate
    o.integrate(times,lp)
    oa.integrate(times,lp)
    # Energy
    o.plotE()
    o.plotE(normed=True)
    o.plotE(pot=lp,d1='R')
    o.plotE(pot=lp,d1='vR')
    o.plotE(pot=lp,d1='vT')
    o.plotE(pot=lp,d1='z')
    o.plotE(pot=lp,d1='vz')
    o.plotE(pot=lp,d1='phi')
    if not _TRAVIS:
        o.plotE(pot=dp,d1='phi')
    oa.plotE()
    oa.plotE(pot=lp,d1='R')
    oa.plotE(pot=lp,d1='vR')
    oa.plotE(pot=lp,d1='vT')
    oa.plotE(pot=lp,d1='z')
    oa.plotE(pot=lp,d1='vz')
    # Vertical energy
    o.plotEz()
    o.plotEz(normed=True)
    o.plotEz(pot=lp,d1='R')
    o.plotEz(pot=lp,d1='vR')
    o.plotEz(pot=lp,d1='vT')
    o.plotEz(pot=lp,d1='z')
    o.plotEz(pot=lp,d1='vz')
    o.plotEz(pot=lp,d1='phi')
    if not _TRAVIS:
        o.plotEz(pot=dp,d1='phi')
    oa.plotEz()
    oa.plotEz(pot=lp,d1='R')
    oa.plotEz(pot=lp,d1='vR')
    oa.plotEz(pot=lp,d1='vT')
    oa.plotEz(pot=lp,d1='z')
    oa.plotEz(pot=lp,d1='vz')
    # Radial energy
    o.plotER()
    o.plotER(normed=True)
    # EzJz
    o.plotEzJz()
    o.plotEzJz(pot=lp,d1='R')
    o.plotEzJz(pot=lp,d1='vR')
    o.plotEzJz(pot=lp,d1='vT')
    o.plotEzJz(pot=lp,d1='z')
    o.plotEzJz(pot=lp,d1='vz')
    o.plotEzJz(pot=lp,d1='phi')
    if not _TRAVIS:
        o.plotEzJz(pot=dp,d1='phi')
    oa.plotEzJz()
    oa.plotEzJz(pot=lp,d1='R')
    oa.plotEzJz(pot=lp,d1='vR')
    oa.plotEzJz(pot=lp,d1='vT')
    oa.plotEzJz(pot=lp,d1='z')
    oa.plotEzJz(pot=lp,d1='vz')
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(normed=True)
    o.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    o.plotJacobi(pot=lp,d1='vR')
    o.plotJacobi(pot=lp,d1='vT')
    o.plotJacobi(pot=lp,d1='z')
    o.plotJacobi(pot=lp,d1='vz')
    o.plotJacobi(pot=lp,d1='phi')
    oa.plotJacobi()
    oa.plotJacobi(pot=lp,d1='R',OmegaP=1.)
    oa.plotJacobi(pot=lp,d1='vR')
    oa.plotJacobi(pot=lp,d1='vT')
    oa.plotJacobi(pot=lp,d1='z')
    oa.plotJacobi(pot=lp,d1='vz')
    # Plot the orbit itself
    o.plot() #defaults
    oa.plot()
    o.plot(d1='vR')
    o.plotR()
    o.plotvR(d1='vT')
    o.plotvT(d1='z')
    o.plotz(d1='vz')
    o.plotvz(d1='phi')
    o.plotphi(d1='vR')
    o.plotx(d1='vx')
    o.plotvx(d1='y')
    o.ploty(d1='vy')
    o.plotvy(d1='x')
    # Remaining attributes
    o.plot(d1='ra',d2='dec')
    o.plot(d2='ra',d1='dec')
    o.plot(d1='pmra',d2='pmdec')
    o.plot(d2='pmra',d1='pmdec')
    o.plot(d1='ll',d2='bb')
    o.plot(d2='ll',d1='bb')
    o.plot(d1='pmll',d2='pmbb')
    o.plot(d2='pmll',d1='pmbb')
    o.plot(d1='vlos',d2='dist')
    o.plot(d2='vlos',d1='dist')
    o.plot(d1='helioX',d2='U')
    o.plot(d2='helioX',d1='U')
    o.plot(d1='helioY',d2='V')
    o.plot(d2='helioY',d1='V')
    o.plot(d1='helioZ',d2='W')
    o.plot(d2='helioZ',d1='W')
    # Some more energies etc.
    o.plot(d1='E',d2='R')
    o.plot(d1='Enorm',d2='R')
    o.plot(d1='Ez',d2='R')
    o.plot(d1='Eznorm',d2='R')
    o.plot(d1='ER',d2='R')
    o.plot(d1='ERnorm',d2='R')
    o.plot(d1='Jacobi',d2='R')
    o.plot(d1='Jacobinorm',d2='R')
    # Test AttributeErrors
    try: oa.plotx()
    except AttributeError: pass
    else: raise AssertionError('plotx() applied to RZOrbit did not raise AttributeError')
    try: oa.plotvx()
    except AttributeError: pass
    else: raise AssertionError('plotvx() applied to RZOrbit did not raise AttributeError')
    try: oa.ploty()
    except AttributeError: pass
    else: raise AssertionError('ploty() applied to RZOrbit did not raise AttributeError')
    try: oa.plotvy()
    except AttributeError: pass
    else: raise AssertionError('plotvy() applied to RZOrbit did not raise AttributeError')
    try: oa.plot(d1='x')
    except AttributeError: pass
    else: raise AssertionError("plot(d1='x') applied to RZOrbit did not raise AttributeError")
    try: oa.plot(d1='vx')
    except AttributeError: pass
    else: raise AssertionError("plot(d1='vx') applied to RZOrbit did not raise AttributeError")
    try: oa.plot(d1='y')
    except AttributeError: pass
    else: raise AssertionError("plot(d1='y') applied to RZOrbit did not raise AttributeError")
    try: oa.plot(d1='vy')
    except AttributeError: pass
    else: raise AssertionError("plot(d1='vy') applied to RZOrbit did not raise AttributeError")
    # Plot the orbit itself in 3D
    o.plot3d() #defaults
    oa.plot3d()
    o.plot3d(d1='t',d2='z',d3='R')
    o.plot3d(d1='R',d2='t',d3='phi')
    o.plot3d(d1='vT',d2='vR',d3='t')
    o.plot3d(d1='z',d2='vT',d3='vz')
    o.plot3d(d1='vz',d2='z',d3='phi')
    o.plot3d(d1='phi',d2='vz',d3='R')
    o.plot3d(d1='vR',d2='phi',d3='vR')
    o.plot3d(d1='vx',d2='x',d3='y')
    o.plot3d(d1='y',d2='vx',d3='vy')
    o.plot3d(d1='vy',d2='y',d3='x')
    o.plot3d(d1='x',d2='vy',d3='vx')
    # Remaining attributes
    o.plot3d(d1='ra',d2='dec',d3='pmra')
    o.plot3d(d2='ra',d1='dec',d3='pmdec')
    o.plot3d(d1='pmra',d2='pmdec',d3='ra')
    o.plot3d(d2='pmra',d1='pmdec',d3='dec')
    o.plot3d(d1='ll',d2='bb',d3='pmll')
    o.plot3d(d2='ll',d1='bb',d3='pmbb')
    o.plot3d(d1='pmll',d2='pmbb',d3='ll')
    o.plot3d(d2='pmll',d1='pmbb',d3='bb')
    o.plot3d(d1='vlos',d2='dist',d3='vlos')
    o.plot3d(d2='vlos',d1='dist',d3='dist')
    o.plot3d(d1='helioX',d2='U',d3='V')
    o.plot3d(d2='helioX',d1='U',d3='helioY')
    o.plot3d(d1='helioY',d2='V',d3='W')
    o.plot3d(d2='helioY',d1='V',d3='helioZ')
    o.plot3d(d1='helioZ',d2='W',d3='U')
    o.plot3d(d2='helioZ',d1='W',d3='helioX')
    # Test AttributeErrors
    try: oa.plot3d(d2='x',d1='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d2='x') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d2='vx',d1='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d2='vx') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d2='y',d1='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d2='y') applied to RZOrbit did not raise AttributeError")
    try: oa.plot(d2='vy',d1='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d2='vy') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d1='x',d2='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d1='x') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d1='vx',d2='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d1='vx') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d1='y',d2='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d1='y') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d1='vy',d2='R',d3='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d1='vy') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d3='x',d2='R',d1='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d3='x') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d3='vx',d2='R',d1='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d3='vx') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d3='y',d2='R',d1='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d3='y') applied to RZOrbit did not raise AttributeError")
    try: oa.plot3d(d3='vy',d2='R',d1='t')
    except AttributeError: pass
    else: raise AssertionError("plot3d(d3='vy') applied to RZOrbit did not raise AttributeError")
    return None

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

# Setup the orbit for the physical-coordinates test
def setup_orbit_physical(tp,axi=False,ro=None,vo=None):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.planarPotential): 
        if axi:
            o= Orbit([1.,1.1,1.1],ro=ro,vo=vo)
        else:
            o= Orbit([1.,1.1,1.1,0.],ro=ro,vo=vo)
    else:
        if axi:
            o= Orbit([1.,1.1,1.1,0.1,0.1],ro=ro,vo=vo)
        else:
            o= Orbit([1.,1.1,1.1,0.1,0.1,0.],ro=ro,vo=vo)
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
