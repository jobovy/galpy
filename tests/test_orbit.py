##############################TESTS ON ORBITS##################################
from __future__ import print_function, division
import warnings
import os, os.path
import sys
import platform
WIN32= platform.system() == 'Windows'
import time
import signal
import subprocess
import pytest
import numpy
import astropy
PY2= sys.version < '3'
_APY3= astropy.__version__ > '3'
from galpy import potential
from galpy.potential.Potential import  _check_c
from galpy.util import galpyWarning
from galpy.util.coords import _K
from test_actionAngle import reset_warning_registry
from test_potential import testplanarMWPotential, testMWPotential, \
    testlinearMWPotential, \
    mockFlatEllipticalDiskPotential, \
    mockFlatLopsidedDiskPotential, \
    mockFlatCosmphiDiskPotential, \
    mockFlatCosmphiDiskwBreakPotential, \
    mockSlowFlatEllipticalDiskPotential, \
    mockFlatDehnenBarPotential, \
    mockSlowFlatDehnenBarPotential, \
    mockFlatSteadyLogSpiralPotential, \
    mockSlowFlatSteadyLogSpiralPotential, \
    mockFlatTransientLogSpiralPotential, \
    mockCombLinearPotential, \
    mockSimpleLinearPotential, \
    mockMovingObjectLongIntPotential, \
    mockSCFZeeuwPotential, \
    mockSCFNFWPotential, \
    mockSCFAxiDensity1Potential, \
    mockSCFAxiDensity2Potential, \
    mockSCFDensityPotential, \
    specialFlattenedPowerPotential, \
    specialMiyamotoNagaiPotential, \
    BurkertPotentialNoC, \
    oblateHernquistPotential, \
    oblateNFWPotential, \
    prolateNFWPotential, \
    prolateJaffePotential, \
    triaxialNFWPotential, \
    NFWTwoPowerTriaxialPotential, \
    fullyRotatedTriaxialNFWPotential, \
    sech2DiskSCFPotential, \
    expwholeDiskSCFPotential, \
    altExpwholeDiskSCFPotential, \
    mockFlatSpiralArmsPotential, \
    mockRotatingFlatSpiralArmsPotential, \
    mockSpecialRotatingFlatSpiralArmsPotential, \
    expwholeDiskSCFPotential, \
    mockFlatDehnenSmoothBarPotential, \
    mockSlowFlatDehnenSmoothBarPotential, \
    mockSlowFlatDecayingDehnenSmoothBarPotential, \
    mockFlatSolidBodyRotationSpiralArmsPotential, \
    mockFlatSolidBodyRotationPlanarSpiralArmsPotential, \
    triaxialLogarithmicHaloPotential, \
    testorbitHenonHeilesPotential, \
    mockFlatCorotatingRotationSpiralArmsPotential, \
    mockFlatTrulyCorotatingRotationSpiralArmsPotential, \
    mockFlatGaussianAmplitudeBarPotential, \
    mockFlatTrulyGaussianAmplitudeBarPotential, \
    testorbitHenonHeilesPotential, \
    nestedListPotential, \
    mockInterpSphericalPotential, \
    mockAdiabaticContractionMWP14WrapperPotential
_TRAVIS= bool(os.getenv('TRAVIS'))
if not _TRAVIS:
    _QUICKTEST= True #Run a more limited set of tests
else:
    _QUICKTEST= True #Also do this for Travis, bc otherwise it takes too long
_NOLONGINTEGRATIONS= False
# Don't show all warnings, to reduce log output
warnings.simplefilter("always",galpyWarning)

# Test whether the energy of simple orbits is conserved for different
# integrators; tests 3D and 2D orbits, 1D orbits done separately below
def test_energy_jacobi_conservation():
    if _NOLONGINTEGRATIONS: return None
    #Basic parameters for the test
    times= numpy.linspace(0.,210.,5001) #~7.5 Gyr at the Solar circle
    growtimes= numpy.linspace(0.,280.,5001) # for pots that grow slowly
    fasttimes= numpy.linspace(0.,14.,501) #~0.5 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p 
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('mockFlatEllipticalDiskPotential')
    pots.append('mockFlatLopsidedDiskPotential')
    pots.append('mockFlatCosmphiDiskPotential')
    pots.append('mockFlatCosmphiDiskwBreakPotential')
    pots.append('mockSlowFlatEllipticalDiskPotential')
    pots.append('mockFlatDehnenBarPotential')
    pots.append('mockSlowFlatDehnenBarPotential')
    pots.append('mockFlatSteadyLogSpiralPotential')
    pots.append('mockSlowFlatSteadyLogSpiralPotential')
    pots.append('mockFlatTransientLogSpiralPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('BurkertPotentialNoC')
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('mockMovingObjectLongIntPotential')
    pots.append('oblateHernquistPotential')
    pots.append('oblateNFWPotential')
    pots.append('prolateNFWPotential')
    pots.append('prolateJaffePotential')
    pots.append('triaxialNFWPotential')
    pots.append('fullyRotatedTriaxialNFWPotential')
    pots.append('NFWTwoPowerTriaxialPotential') # for planar-from-full
    pots.append('mockSCFZeeuwPotential')
    pots.append('mockSCFNFWPotential')
    pots.append('mockSCFAxiDensity1Potential')
    pots.append('mockSCFAxiDensity2Potential')
    pots.append('mockSCFDensityPotential')
    pots.append('sech2DiskSCFPotential')
    pots.append('expwholeDiskSCFPotential')
    pots.append('altExpwholeDiskSCFPotential')
    pots.append('mockFlatSpiralArmsPotential')
    pots.append('mockRotatingFlatSpiralArmsPotential')
    pots.append('mockSpecialRotatingFlatSpiralArmsPotential')
    pots.append('mockFlatDehnenSmoothBarPotential')
    pots.append('mockSlowFlatDehnenSmoothBarPotential')
    pots.append('mockSlowFlatDecayingDehnenSmoothBarPotential')
    pots.append('mockFlatSolidBodyRotationSpiralArmsPotential')
    pots.append('mockFlatSolidBodyRotationPlanarSpiralArmsPotential')
    pots.append('triaxialLogarithmicHaloPotential')   
    pots.append('testorbitHenonHeilesPotential')   
    pots.append('mockFlatCorotatingRotationSpiralArmsPotential')
    pots.append('mockFlatGaussianAmplitudeBarPotential')
    pots.append('nestedListPotential')
    pots.append('mockInterpSphericalPotential')
    pots.append('mockAdiabaticContractionMWP14WrapperPotential')   
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    if False: #_TRAVIS: #travis CI
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
    jactol['RazorThinExponentialDiskPotential']= -9. #these are more difficult
    jactol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    jactol['mockFlatDehnenBarPotential']= -8. #these are more difficult
    jactol['mockFlatDehnenSmoothBarPotential']= -8. #these are more difficult
    jactol['mockMovingObjectLongIntPotential']= -8. #these are more difficult
    jactol['mockSlowFlatEllipticalDiskPotential']= -6. #these are more difficult (and also not quite conserved)
    jactol['mockSlowFlatSteadyLogSpiralPotential']= -8. #these are more difficult (and also not quite conserved)
    jactol['mockSlowFlatDehnenSmoothBarPotential']= -8. #these are more difficult (and also not quite conserved)
    jactol['mockSlowFlatDecayingDehnenSmoothBarPotential']= -8. #these are more difficult (and also not quite conserved)
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        if p in list(jactol.keys()): tjactol= jactol[p]
        else: tjactol= jactol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        #if not p == 'NFWPotential' and not p == 'mockSlowFlatDecayingDehnenSmoothBarPotential': continue
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
        for integrator in integrators:
            if integrator == 'dopr54_c' \
                    and ('Spiral' in p or 'Lopsided' in p \
                             or 'Dehnen' in p or 'Cosmphi' in p):
                ttimes= growtimes
            elif integrator == 'dopr54_c' \
                    and not 'MovingObject' in p \
                    and not p == 'FerrersPotential': ttimes= times
            else: ttimes= fasttimes
            #First track azimuth
            o= setup_orbit_energy(tp,axi=False,henon='Henon' in p)
            if isinstance(tp,testMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            elif isinstance(tp,testplanarMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            else:
                o.integrate(ttimes,tp,method=integrator)
            tEs= o.E(ttimes)
            #print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
            if not 'Bar' in p and not 'Spiral' in p \
                    and not 'MovingObject' in p and not 'Slow' in p:
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs)))
            #Jacobi
            if 'Elliptical' in p or 'Lopsided' in p \
                    or 'DehnenSmoothBar' in p  \
                    or 'SolidBodyRotation' in p \
                    or 'CorotatingRotation' in p \
                    or 'GaussianAmplitudeBar' in p  \
                    or p == 'mockMovingObjectLongIntPotential' \
                    or 'Cosmphi' in p or 'triaxialLog' in p \
                    or 'Henon' in p:
                tJacobis= o.Jacobi(ttimes,pot=tp)
            else:
                tJacobis= o.Jacobi(ttimes)
#            print(p, (numpy.std(tJacobis)/numpy.mean(tJacobis))**2.)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s at the %g level" %(p,integrator,(numpy.std(tJacobis)/numpy.mean(tJacobis))**2.)
            if firstTest or 'testMWPotential' in p:
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
                o= setup_orbit_energy(tp,axi=False,henon='Henon' in p)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
                if not isinstance(tp,potential.linearPotential):
                    try:
                        o.Jacobi()
                    except AttributeError:
                        pass
                    else:
                        raise AssertionError("o.Jacobi() before the orbit was integrated did not throw an AttributeError")
            if 'MovingObject' in p:
                if _QUICKTEST \
                        and not (('NFW' in p and not tp.isNonAxi and 'SCF' not in p)
                                 or 'linearMWPotential' in p \
                                     or ('Burkert' in p and not tp.hasC)):
                    break
                else: continue
            # AnyAxisymmetricRazorThinDiskPot has bad behavior for odeint and isn't really meant for orbit integration
            if 'AnyAxisymmetricRazorThinDiskPotential' in p: break
            #Now do axisymmetric
            if not tp.isNonAxi:
                o= setup_orbit_energy(tp,axi=True,henon='Henon' in p)
                if isinstance(tp,testMWPotential) \
                        or isinstance(tp,testplanarMWPotential):
                    o.integrate(ttimes,tp._potlist,method=integrator)
                else:
                    o.integrate(ttimes,tp,method=integrator)
                tEs= o.E(ttimes)
    #            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs))**2.)
                #Jacobi
                tJacobis= o.Jacobi(ttimes)
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
                    assert (o.Jacobi(OmegaP=1.)-o.Jacobi())**2. < 10.**ttol, \
                        "o.Jacobi calculated with OmegaP=1. for axisymmetric potential is not equal to o.Jacobi (OmegaP=1 is the default for potentials without a pattern speed"
                    o= setup_orbit_energy(tp,axi=True,henon='Henon' in p)
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
                if _QUICKTEST and not (('NFW' in p and not tp.isNonAxi and 'SCF' not in p) \
                                           or ('Burkert' in p and not tp.hasC)): break
                else: continue
            #Same for a planarPotential
#            print integrator
            if not ptp is None and not ptp.isNonAxi:
                o= setup_orbit_energy(ptp,axi=True)
                if isinstance(tp,testMWPotential) \
                        or isinstance(tp,testplanarMWPotential):
                    o.integrate(ttimes,
                                potential.toPlanarPotential(tp._potlist),
                                method=integrator)
                else:
                    o.integrate(ttimes,ptp,method=integrator)
                tEs= o.E(ttimes)
#                print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
                #Jacobi
                tJacobis= o.Jacobi(ttimes)
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
            if isinstance(tp,testMWPotential) \
                    or isinstance(tp,testplanarMWPotential):
                o.integrate(ttimes,
                            potential.toPlanarPotential(tp._potlist),
                            method=integrator)
            else:
                o.integrate(ttimes,ptp,method=integrator)
            tEs= o.E(ttimes)
            #print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
            if not 'Bar' in p and not 'Spiral' in p:
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs))**2.)
            #Jacobi
            if 'DehnenSmoothBar' in p or 'SolidBodyRotation' in p \
                    or 'CorotatingRotation' in p \
                    or 'GaussianAmplitudeBar' in p:
                tJacobis= o.Jacobi(ttimes,pot=tp)
            else:
                tJacobis= o.Jacobi(ttimes)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails by %g for potential %s and integrator %s" %((numpy.std(tJacobis)/numpy.mean(tJacobis))**2.,p,integrator)
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
            #Same for a planarPotential, but integrating w/ the potential directly, rather than the toPlanar instance; this tests that those potential attributes are passed to C correctly
#            print integrator
            if not ptp is None and not ptp.isNonAxi:
                o= setup_orbit_energy(ptp,axi=True)
                if isinstance(tp,testMWPotential) \
                        or isinstance(tp,testplanarMWPotential):
                    o.integrate(ttimes,tp._potlist,method=integrator)
                else:
                    o.integrate(ttimes,tp,method=integrator)
                tEs= o.E(ttimes)
                #print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs))**2.)
                #Jacobi
                if 'DehnenSmoothBar' in p or 'SolidBodyRotation' in p \
                        or 'CorotatingRotation' in p \
                        or 'GaussianAmplitudeBar' in p:
                    tJacobis= o.Jacobi(ttimes,pot=tp)
                else:
                    tJacobis= o.Jacobi(ttimes)
                assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                    "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Same for a planarPotential, track azimuth
            o= setup_orbit_energy(ptp,axi=False)
            if isinstance(tp,testMWPotential) \
                    or isinstance(tp,testplanarMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            else:
                o.integrate(ttimes,tp,method=integrator)
            tEs= o.E(ttimes)
#            print p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.
            if not 'Bar' in p and not 'Spiral' in p:
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            #Jacobi
            if 'DehnenSmoothBar' in p or 'SolidBodyRotation' in p \
                    or 'CorotatingRotation' in p \
                    or 'GaussianAmplitudeBar' in p:
                tJacobis= o.Jacobi(ttimes,pot=tp)
            else:
                tJacobis= o.Jacobi(ttimes)
            assert (numpy.std(tJacobis)/numpy.mean(tJacobis))**2. < 10.**tjactol, \
                "Jacobi integral conservation during the orbit integration fails for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and not (('NFW' in p and not tp.isNonAxi and 'SCF' not in p) \
                                     or ('Burkert' in p and not tp.hasC)): break
    #raise AssertionError
    return None

# Test whether the energy of 1D orbits is conserved for different integrators
def test_energy_conservation_linear():
    if _NOLONGINTEGRATIONS: return None
    #Basic parameters for the test
    times= numpy.linspace(0.,210.,5001) #~7.5 Gyr at the Solar circle
    growtimes= numpy.linspace(0.,280.,5001) # for pots that grow slowly
    fasttimes= numpy.linspace(0.,14.,501) #~0.5 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p 
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('BurkertPotentialNoC')
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('testlinearMWPotential')
    pots.append('mockCombLinearPotential')
    pots.append('mockSimpleLinearPotential')
    pots.append('oblateNFWPotential')
    pots.append('prolateNFWPotential')
    pots.append('triaxialNFWPotential')
    pots.append('fullyRotatedTriaxialNFWPotential')
    pots.append('NFWTwoPowerTriaxialPotential') # for planar-from-full
    pots.append('mockSCFZeeuwPotential')
    pots.append('mockSCFNFWPotential')
    pots.append('mockSCFAxiDensity1Potential')
    pots.append('mockSCFAxiDensity2Potential')
    pots.append('sech2DiskSCFPotential')
    pots.append('expwholeDiskSCFPotential')
    pots.append('altExpwholeDiskSCFPotential')
    pots.append('triaxialLogarithmicHaloPotential')   
    pots.append('nestedListPotential')
    pots.append('mockInterpSphericalPotential')
    pots.append('mockAdiabaticContractionMWP14WrapperPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    rmpots.append('SoftenedNeedleBarPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -10.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        #if not p == 'NFWPotential' and not p == 'mockFlatGaussianAmplitudeBarPotential': continue
        tp= tclass()
        if hasattr(tp,'toVertical'):
            if not hasattr(tp,'normalize'): continue #skip these
            tp.normalize(1.)
            tp= tp.toVertical(1.2,phi=0.3)
        elif isinstance(tp,potential.linearPotential):
            pass
        else: # not 3D --> 1D or 1D, so skip
            continue
        for integrator in integrators:
            if integrator == 'dopr54_c' \
                    and ('Spiral' in p or 'Lopsided' in p \
                             or 'Dehnen' in p or 'Cosmphi' in p):
                ttimes= growtimes
            elif integrator == 'dopr54_c' \
                    and not 'MovingObject' in p \
                    and not p == 'FerrersPotential' \
                    and not p == 'AnyAxisymmetricRazorThinDiskPotential': ttimes= times
            else: ttimes= fasttimes
            #First track azimuth
            o= setup_orbit_energy(tp)
            if isinstance(tp,testMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            elif isinstance(tp,testplanarMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            elif isinstance(tp,testlinearMWPotential):
                o.integrate(ttimes,tp._potlist,method=integrator)
            else:
                o.integrate(ttimes,tp,method=integrator)
            tEs= o.E(ttimes)
            #print(p, integrator, (numpy.std(tEs)/numpy.mean(tEs))**2.)
            if not 'Bar' in p and not 'Spiral' in p \
                    and not 'MovingObject' in p and not 'Slow' in p:
                assert (numpy.std(tEs)/numpy.mean(tEs))**2. < 10.**ttol, \
                    "Energy conservation during the orbit integration fails for potential %s and integrator %s by %g" %(p,integrator,(numpy.std(tEs)/numpy.mean(tEs)))
            if firstTest or 'testMWPotential' in p or 'linearMWPotential' in p:
                #Some basic checking of the energy function
                assert (o.E(pot=None)-o.E(pot=tp))**2. < 10.**ttol, \
                    "Energy calculated with pot=None and pot=the Potential the orbit was integrated with do not agree"
                assert (o.E()-o.E(0.))**2. < 10.**ttol, \
                    "Energy calculated with o.E() and o.E(0.) do not agree"
                o= setup_orbit_energy(tp,axi=False,henon='Henon' in p)
                try:
                    o.E()
                except AttributeError:
                    pass
                else:
                    raise AssertionError("o.E() before the orbit was integrated did not throw an AttributeError")
            if _QUICKTEST and not (p == 'NFWPotential' \
                                  or ('Burkert' in p and not tp.hasC)): break
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
            if integrator in list(tol.keys()): ttol= tol[integrator]
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
   
def test_liouville_planar():
    if _NOLONGINTEGRATIONS: return None
    #Basic parameters for the test
    times= numpy.linspace(0.,28.,1001) #~1 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'dop853_c', 'dop853',
                  'odeint', #direct python solver
                  'rk4_c','rk6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('mockFlatEllipticalDiskPotential')
    pots.append('mockFlatLopsidedDiskPotential')
    pots.append('mockFlatCosmphiDiskPotential')
    pots.append('mockFlatCosmphiDiskwBreakPotential')
    pots.append('mockSlowFlatEllipticalDiskPotential')
    pots.append('mockFlatDehnenBarPotential')
    pots.append('mockFlatDehnenBarPotential')
    pots.append('mockSlowFlatDehnenBarPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('BurkertPotentialNoC')
    pots.append('NFWTwoPowerTriaxialPotential') # for planar-from-full
    pots.append('mockSCFZeeuwPotential')
    pots.append('mockSCFNFWPotential')
    pots.append('mockSCFAxiDensity1Potential')
    pots.append('mockSCFAxiDensity2Potential')
    pots.append('mockSCFDensityPotential')
    pots.append('mockFlatSpiralArmsPotential')
    pots.append('mockRotatingFlatSpiralArmsPotential')
    pots.append('mockSpecialRotatingFlatSpiralArmsPotential')
    #pots.append('mockFlatSteadyLogSpiralPotential')
    #pots.append('mockFlatTransientLogSpiralPotential')
    pots.append('mockFlatDehnenSmoothBarPotential')
    pots.append('mockSlowFlatDehnenSmoothBarPotential') 
    pots.append('mockSlowFlatDecayingDehnenSmoothBarPotential') 
    pots.append('mockFlatSolidBodyRotationSpiralArmsPotential')
    pots.append('triaxialLogarithmicHaloPotential')   
    pots.append('testorbitHenonHeilesPotential')   
    pots.append('mockFlatTrulyCorotatingRotationSpiralArmsPotential')
    pots.append('mockFlatTrulyGaussianAmplitudeBarPotential')
    pots.append('nestedListPotential')
    pots.append('mockInterpSphericalPotential')
    pots.append('mockAdiabaticContractionMWP14WrapperPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    #rmpots.append('BurkertPotential')
    #Don't have C implementations of the relevant 2nd derivatives
    rmpots.append('DoubleExponentialDiskPotential')
    rmpots.append('RazorThinExponentialDiskPotential')
    # Doesn't have C at all
    rmpots.append('AnyAxisymmetricRazorThinDiskPotential')
    #rmpots.append('PowerSphericalPotentialwCutoff')
    #Doesn't have the R2deriv
    rmpots.append('TwoPowerSphericalPotential')
    rmpots.append('TwoPowerTriaxialPotential')
    rmpots.append('TriaxialHernquistPotential')
    rmpots.append('TriaxialJaffePotential')
    rmpots.append('SoftenedNeedleBarPotential')
    rmpots.append('DiskSCFPotential')
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    rmpots.append('PerfectEllipsoidPotential')
    rmpots.append('TriaxialGaussianPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -8.
    tol['KeplerPotential']= -7. #more difficult
    tol['NFWPotential']= -6. #more difficult for rk4_c, only one that does this
    tol['TriaxialNFWPotential']= -4. #more difficult
    tol['triaxialLogarithmicHaloPotential']= -7. #more difficult
    tol['FerrersPotential']= -2.
    tol['HomogeneousSpherePotential']= -4.
    tol['KingPotential']= -6.
    tol['mockInterpSphericalPotential']= -4. # == HomogeneousSpherePotential
    tol['mockFlatCosmphiDiskwBreakPotential']= -7. # more difficult
    tol['mockFlatTrulyCorotatingRotationSpiralArmsPotential']= -5. # more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        #if not p == 'NFWPotential' and not p == 'mockSlowFlatDecayingDehnenSmoothBarPotential': continue
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        for integrator in integrators:
            if p in list(tol.keys()): ttol= tol[p]
            else: ttol= tol['default']
            if isinstance(tp,testMWPotential) \
                        or isinstance(tp,testplanarMWPotential):
                thasC= _check_c(tp._potlist)
            else:
                thasC= _check_c(tp)
            if (integrator == 'odeint' or not thasC) \
                and not p == 'FerrersPotential' : ttol= -4.
            if True: ttimes= times
            o= setup_orbit_liouville(ptp,axi=False,henon='Henon' in p)
            #Calculate the Jacobian d x / d x
            if hasattr(tp,'_potlist'):
                if isinstance(tp,testMWPotential):
                    plist= potential.toPlanarPotential(tp._potlist)
                else:
                    plist= tp._potlist
                o.integrate_dxdv([1.,0.,0.,0.],ttimes,plist,
                                 method=integrator,
                                 rectIn=True,rectOut=True)
                dx= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,1.,0.,0.],ttimes,plist,
                                 method=integrator,
                                 rectIn=True,rectOut=True)
                dy= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,0.,1.,0.],ttimes,plist,
                                 method=integrator,
                                 rectIn=True,rectOut=True)
                dvx= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,0.,0.,1.],ttimes,plist,
                                 method=integrator,
                                 rectIn=True,rectOut=True)
                dvy= o.getOrbit_dxdv()[-1,:]
            else:
                o.integrate_dxdv([1.,0.,0.,0.],ttimes,ptp,method=integrator,
                                 rectIn=True,rectOut=True)
                dx= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,1.,0.,0.],ttimes,ptp,method=integrator,
                                 rectIn=True,rectOut=True)
                dy= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,0.,1.,0.],ttimes,ptp,method=integrator,
                                 rectIn=True,rectOut=True)
                dvx= o.getOrbit_dxdv()[-1,:]
                o.integrate_dxdv([0.,0.,0.,1.],ttimes,ptp,method=integrator,
                                 rectIn=True,rectOut=True)
                dvy= o.getOrbit_dxdv()[-1,:]
            tjac= numpy.linalg.det(numpy.array([dx,dy,dvx,dvy]))
            #print(p, integrator, numpy.fabs(tjac-1.),ttol)
            assert numpy.fabs(tjac-1.) < 10.**ttol, 'Liouville theorem jacobian differs from one by %g for %s and integrator %s' % (numpy.fabs(tjac-1.),p,integrator)
            if firstTest or ('Burkert' in p and not ptp.hasC):
                #Some one time tests
                #Test non-rectangular in- and output
                try:
                    o.integrate_dxdv([0.,0.,0.,1.],ttimes,ptp,
                                     method='leapfrog',
                                     rectIn=True,rectOut=True)
                except ValueError: pass
                else: raise AssertionError("integrate_dxdv with symplectic integrator should have raised ValueError, but didn't")
                firstTest= False
            if _QUICKTEST and not (('NFW' in p and not ptp.isNonAxi and 'SCF' not in p) \
                                       or ('Burkert' in p and not ptp.hasC)): break
    return None

# Test that the eccentricity of circular orbits is zero
def test_eccentricity():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('mockInterpSphericalPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -16.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['NFWPotential']= -12. #these are more difficult
    tol['TriaxialNFWPotential']= -12. #these are more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'isNonAxi') and tp.isNonAxi:
            continue # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
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
            if isinstance(tp,testplanarMWPotential) \
                    or isinstance(tp,testMWPotential):
                o.integrate(times,tp._potlist,method=integrator)
            else:
                o.integrate(times,tp,method=integrator)
            tecc= o.e()
#            print p, integrator, tecc
            assert tecc**2. < 10.**ttol, \
                "Eccentricity of a circular orbit is not equal to zero by %g for potential %s and integrator %s" %(tecc**2.,p,integrator)
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
            if ptp is None:
                if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
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
            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
    #raise AssertionError
    return None
    
# Test that the pericenter of orbits launched with vR=0 and vT > vc is the starting radius
def test_pericenter():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('mockInterpSphericalPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    if False: #_TRAVIS: #travis CI
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
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'isNonAxi') and tp.isNonAxi:
            continue # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
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
            if isinstance(tp,testplanarMWPotential) \
                    or isinstance(tp,testMWPotential):
                o.integrate(times,tp._potlist,method=integrator)
            else:
                o.integrate(times,tp,method=integrator)
            tperi= o.rperi()
#               print p, integrator, tperi
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
            if ptp is None:
                if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
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
            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
    #raise AssertionError
    return None

# Test that the apocenter of orbits launched with vR=0 and vT < vc is the starting radius
def test_apocenter():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential)
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('mockInterpSphericalPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -16.
    tol['FlattenedPowerPotential']= -14. #these are more difficult
#    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
#    tol['NFWPotential']= -12. #these are more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'isNonAxi') and tp.isNonAxi:
            continue # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
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
            if isinstance(tp,testplanarMWPotential) \
                    or isinstance(tp,testMWPotential):
                o.integrate(times,tp._potlist,method=integrator)
            else:
                o.integrate(times,tp,method=integrator)
            tapo= o.rap()
            #print p, integrator, tapo, (tapo-o.R())**2.
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
            if ptp is None:
                if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
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
            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
    #raise AssertionError
    return None

# Test that the zmax of orbits launched with vz=0 is the starting height
def test_zmax():
    #return None
    #Basic parameters for the test
    times= numpy.linspace(0.,7.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    pots.append('mockInterpSphericalPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    # No C and therefore annoying
    rmpots.append('AnyAxisymmetricRazorThinDiskPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -16.
    tol['RazorThinExponentialDiskPotential']= -6. #these are more difficult
    tol['KuzminDiskPotential']= -6. #these are more difficult
#    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    firstTest= True
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'isNonAxi') and tp.isNonAxi:
            continue # skip, bc eccentricity of circ. =/= 0
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if hasattr(tp,'toPlanar'):
            ptp= tp.toPlanar()
        else:
            ptp= None
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
            if isinstance(tp,testMWPotential):
                o.integrate(times,tp._potlist,method=integrator)
            else:
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
            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
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
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    rmpots.append('HomogeneousSpherePotential') # fails currently, because delta esimation gives a NaN due to a 0/0; delta should just be zero, but don't want to special-case
    # No C and therefore annoying
    rmpots.append('AnyAxisymmetricRazorThinDiskPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -10.
    tol['NFWPotential']= -9. #these are more difficult
    tol['PlummerPotential']= -9. #these are more difficult
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['RazorThinExponentialDiskPotential']= -8. #these are more difficult
    tol['IsochronePotential']= -6. #these are more difficult
    tol['DehnenSphericalPotential']= -8. #these are more difficult
    tol['DehnenCoreSphericalPotential']= -8. #these are more difficult
    tol['JaffePotential']= -6. #these are more difficult
    tol['TriaxialHernquistPotential']= -8. #these are more difficult
    tol['TriaxialJaffePotential']= -8. #these are more difficult
    tol['TriaxialNFWPotential']= -8. #these are more difficult
    tol['PowerSphericalPotential']= -8. #these are more difficult
    tol['PowerSphericalPotentialwCutoff']= -8. #these are more difficult
    tol['FlattenedPowerPotential']= -8. #these are more difficult
    tol['KeplerPotential']= -8. #these are more difficult
    tol['PseudoIsothermalPotential']= -7. #these are more difficult
    tol['KuzminDiskPotential'] = -8.  #these are more difficult
    tol['DiskSCFPotential'] = -8.  #these are more difficult
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
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
            if hasattr(tp,'isNonAxi') and tp.isNonAxi:
                continue # skip, bc eccentricity of circ. =/= 0
            if not hasattr(tp,'normalize'): continue #skip these
            tp.normalize(1.)
            if hasattr(tp,'toPlanar'):
                ptp= tp.toPlanar()
            else:
                ptp= None
        for integrator in integrators:
            for ii in range(4):
                if ii == 0: #axi, full
                    #First do axi
                    o= setup_orbit_analytic(tp,axi=True)
                    if isinstance(tp,testplanarMWPotential) \
                            or isinstance(tp,testMWPotential):
                        o.integrate(times,tp._potlist,method=integrator)
                    else:
                        o.integrate(times,tp,method=integrator)
                elif ii == 1: #track azimuth, full
                    #First do axi
                    o= setup_orbit_analytic(tp,axi=False)
                    if isinstance(tp,testplanarMWPotential) \
                            or isinstance(tp,testMWPotential):
                        o.integrate(times,tp._potlist,method=integrator)
                    else:
                        o.integrate(times,tp,method=integrator)
                elif ii == 2: #axi, planar
                    if ptp is None: continue
                    #First do axi
                    o= setup_orbit_analytic(ptp,axi=True)
                    if isinstance(ptp,testplanarMWPotential) \
                            or isinstance(ptp,testMWPotential):
                        o.integrate(times,ptp._potlist,method=integrator)
                    else:
                        o.integrate(times,ptp,method=integrator)
                elif ii == 3: #track azimuth, full
                    if ptp is None: continue
                    #First do axi
                    o= setup_orbit_analytic(ptp,axi=False)
                    if isinstance(ptp,testplanarMWPotential) \
                            or isinstance(ptp,testMWPotential):
                        o.integrate(times,ptp._potlist,method=integrator)
                    else:
                        o.integrate(times,ptp,method=integrator)
                #Eccentricity
                tecc= o.e()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    tecc_analytic= o.e(analytic=True,type='adiabatic')
                else:
                    tecc_analytic= o.e(analytic=True)
                #print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc-tecc_analytic)**2. < 10.**ttol, \
                    "Analytically computed eccentricity does not agree with numerical estimate for potential %s and integrator %s, by %g" %(p,integrator,(tecc-tecc_analytic)**2.)
                #Pericenter radius
                trperi= o.rperi()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trperi_analytic= o.rperi(analytic=True,type='adiabatic')
                else:
                    trperi_analytic= o.rperi(analytic=True)
                #print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi-trperi_analytic)**2. < 10.**ttol, \
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                assert (o.rperi(ro=8.)/8.-trperi_analytic)**2. < 10.**ttol, \
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Apocenter radius
                trap= o.rap()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trap_analytic= o.rap(analytic=True,type='adiabatic')
                else:
                    trap_analytic= o.rap(analytic=True)
                #print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap-trap_analytic)**2. < 10.**ttol, \
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s by %g" %(p,integrator,(trap-trap_analytic)**2.)
                assert (o.rap(ro=8.)/8.-trap_analytic)**2. < 10.**ttol, \
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Do this also for an orbit starting at pericenter
                if ii == 0: #axi, full
                    #First do axi
                    o= setup_orbit_pericenter(tp,axi=True)
                    o.integrate(times,tp,method=integrator)
                elif ii == 1: #track azimuth, full
                    #First do axi
                    o= setup_orbit_pericenter(tp,axi=False)
                    o.integrate(times,tp,method=integrator)
                elif ii == 2: #axi, planar
                    #First do axi
                    o= setup_orbit_pericenter(ptp,axi=True)
                    o.integrate(times,ptp,method=integrator)
                elif ii == 3: #track azimuth, full
                    #First do axi
                    o= setup_orbit_pericenter(ptp,axi=False)
                    o.integrate(times,ptp,method=integrator)
                #Eccentricity
                tecc= o.e()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    tecc_analytic= o.e(analytic=True,type='adiabatic')
                else:
                    tecc_analytic= o.e(analytic=True)
                #print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc-tecc_analytic)**2. < 10.**ttol, \
                    "Analytically computed eccentricity does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                #Pericenter radius
                trperi= o.rperi()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trperi_analytic= o.rperi(analytic=True,type='adiabatic')
                else:
                    trperi_analytic= o.rperi(analytic=True)
                #print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi-trperi_analytic)**2. < 10.**ttol, \
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                assert (o.rperi(ro=8.)/8.-trperi_analytic)**2. < 10.**ttol, \
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Apocenter radius
                trap= o.rap()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trap_analytic= o.rap(analytic=True,type='adiabatic')
                else:
                    trap_analytic= o.rap(analytic=True)
                #print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap-trap_analytic)**2. < 10.**ttol, \
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s by %g" %(p,integrator,(trap-trap_analytic))
                assert (o.rap(ro=8.)/8.-trap_analytic)**2. < 10.**ttol, \
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Do this also for an orbit starting at apocenter
                if ii == 0: #axi, full
                    #First do axi
                    o= setup_orbit_apocenter(tp,axi=True)
                    o.integrate(times,tp,method=integrator)
                elif ii == 1: #track azimuth, full
                    #First do axi
                    o= setup_orbit_apocenter(tp,axi=False)
                    o.integrate(times,tp,method=integrator)
                elif ii == 2: #axi, planar
                    #First do axi
                    o= setup_orbit_apocenter(ptp,axi=True)
                    o.integrate(times,ptp,method=integrator)
                elif ii == 3: #track azimuth, full
                    #First do axi
                    o= setup_orbit_apocenter(ptp,axi=False)
                    o.integrate(times,ptp,method=integrator)
                #Eccentricity
                tecc= o.e()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    tecc_analytic= o.e(analytic=True,type='adiabatic')
                else:
                    tecc_analytic= o.e(analytic=True)
                #print p, integrator, tecc, tecc_analytic, (tecc-tecc_analytic)**2.
                assert (tecc-tecc_analytic)**2. < 10.**ttol, \
                    "Analytically computed eccentricity does not agree with numerical estimate by %g for potential %s and integrator %s" %((tecc-tecc_analytic)**2.,p,integrator)
                #Pericenter radius
                trperi= o.rperi()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trperi_analytic= o.rperi(analytic=True,type='adiabatic')
                else:
                    trperi_analytic= o.rperi(analytic=True)
                #print p, integrator, trperi, trperi_analytic, (trperi-trperi_analytic)**2.
                assert (trperi-trperi_analytic)**2. < 10.**ttol, \
                    "Analytically computed pericenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                assert (o.rperi(ro=8.)/8.-trperi_analytic)**2. < 10.**ttol, \
                    "Pericenter in physical coordinates does not agree with physical-scale times pericenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)
                #Apocenter radius
                trap= o.rap()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    trap_analytic= o.rap(analytic=True,type='adiabatic')
                else:
                    trap_analytic= o.rap(analytic=True)
                #print p, integrator, trap, trap_analytic, (trap-trap_analytic)**2.
                assert (trap-trap_analytic)**2. < 10.**ttol, \
                    "Analytically computed apocenter radius does not agree with numerical estimate for potential %s and integrator %s" %(p,integrator)
                assert (o.rap(ro=8.)/8.-trap_analytic)**2. < 10.**ttol, \
                    "Apocenter in physical coordinates does not agree with physical-scale times apocenter in normalized coordinates for potential %s and integrator %s" %(p,integrator)

            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
    #raise AssertionError
    return None
    
def test_orbit_rguiding():
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014, \
        TriaxialNFWPotential, rl
    from galpy.orbit import Orbit
    # For a single potential
    lp= LogarithmicHaloPotential(normalize=1.)
    R,Lz= 1.,1.4
    o= Orbit([R,0.4,Lz/R,0.,0.1,0.])
    assert numpy.fabs(o.rguiding(pot=lp)-rl(lp,Lz)) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl'
    # For a list of potentials
    R,Lz= 1.4,0.9
    o= Orbit([R,0.4,Lz/R,0.,0.1,0.])
    assert numpy.fabs(o.rguiding(pot=MWPotential2014)-rl(MWPotential2014,Lz)) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl'
    # For an orbit integrated in a non-axisymmetric potential, such that Lz varies
    np= TriaxialNFWPotential(amp=20.,c=0.8,b=0.7)
    npaxi= TriaxialNFWPotential(amp=20.,c=0.8)
    R,Lz= 1.2,2.4
    o= Orbit([R,0.4,Lz/R,0.,0.1,0.])
    ts= numpy.linspace(0.,10.,101)
    o.integrate(ts,np)
    assert numpy.amax(numpy.fabs(o.rguiding(ts,pot=npaxi)-numpy.array([rl(npaxi,o.Lz(t)) for t in ts]))) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl for integrated orbit'
    return None

def test_orbit_rguiding_planar():
    from galpy.potential import LogarithmicHaloPotential, MWPotential2014, \
        TriaxialNFWPotential, rl
    from galpy.orbit import Orbit
    # For a single potential
    lp= LogarithmicHaloPotential(normalize=1.)
    R,Lz= 1.,1.4
    o= Orbit([R,0.4,Lz/R,0.])
    assert numpy.fabs(o.rguiding(pot=lp)-rl(lp,Lz)) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl'
    # For a list of potentials
    R,Lz= 1.4,0.9
    o= Orbit([R,0.4,Lz/R,0.])
    assert numpy.fabs(o.rguiding(pot=MWPotential2014)-rl(MWPotential2014,Lz)) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl'
    # For an orbit integrated in a non-axisymmetric potential, such that Lz varies
    np= TriaxialNFWPotential(amp=20.,c=0.8,b=0.7)
    npaxi= TriaxialNFWPotential(amp=20.,c=0.8)
    R,Lz= 1.2,2.4
    o= Orbit([R,0.4,Lz/R,0.])
    ts= numpy.linspace(0.,10.,101)
    o.integrate(ts,np)
    assert numpy.amax(numpy.fabs(o.rguiding(ts,pot=npaxi)-numpy.array([rl(npaxi,o.Lz(t)) for t in ts]))) < 1e-10, 'Guiding center radius returned by Orbit interface rguiding is different from that returned by potential interface rl for integrated orbit'
    return None

# Check that zmax calculated analytically agrees with numerical calculation
def test_analytic_zmax():
    #Basic parameters for the test
    times= numpy.linspace(0.,20.,251) #~10 Gyr at the Solar circle
    integrators= ['dopr54_c', #first, because we do it for all potentials
                  'odeint', #direct python solver
                  'dop853', 'dop853_c',
                  'leapfrog','leapfrog_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'FullTo' in p and not 'toPlanar' in p
               and not 'evaluate' in p and not 'Wrapper' in p
               and not 'toVertical' in p)]
    pots.append('testMWPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential',
             'EllipsoidalPotential','NumericalPotentialDerivativesMixin',
             'SphericalPotential','interpSphericalPotential']
    rmpots.append('SphericalShellPotential')
    rmpots.append('RingPotential')
    rmpots.append('HomogeneousSpherePotential') # fails currently, because delta esimation gives a NaN due to a 0/0; delta should just be zero, but don't want to special-case
    # No C and therefore annoying
    rmpots.append('AnyAxisymmetricRazorThinDiskPotential')
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -9.
    tol['IsochronePotential']= -4. #these are more difficult
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['RazorThinExponentialDiskPotential']= -4. #these are more difficult
    tol['KuzminKutuzovStaeckelPotential']= -4. #these are more difficult
    tol['PlummerPotential']= -4. #these are more difficult
    tol['PseudoIsothermalPotential']= -4. #these are more difficult
    tol['DehnenSphericalPotential']= -8. #these are more difficult
    tol['DehnenCoreSphericalPotential']= -8. #these are more difficult
    tol['HernquistPotential']= -8. #these are more difficult
    tol['TriaxialHernquistPotential']= -8. #these are more difficult
    tol['JaffePotential']= -8. #these are more difficult
    tol['TriaxialJaffePotential']= -8. #these are more difficult
    tol['TriaxialNFWPotential']= -8. #these are more difficult
    tol['MiyamotoNagaiPotential']= -7. #these are more difficult
    tol['MN3ExponentialDiskPotential']= -6. #these are more difficult
    tol['LogarithmicHaloPotential']= -7. #these are more difficult
    tol['KeplerPotential']= -7. #these are more difficult
    tol['PowerSphericalPotentialwCutoff']= -8. #these are more difficult
    tol['FlattenedPowerPotential']= -8. #these are more difficult
    tol['testMWPotential']= -6. #these are more difficult
    tol['KuzminDiskPotential']=-4 #these are more difficult
    tol['SCFPotential']= -8. #these are more difficult
    tol['DiskSCFPotential']= -6. #these are more difficult
    for p in pots:
        #Setup instance of potential
        if p in list(tol.keys()): ttol= tol[p]
        else: ttol= tol['default']
        if p == 'MWPotential':
            tp= potential.MWPotential
        else:
            try:
                tclass= getattr(potential,p)
            except AttributeError:
                tclass= getattr(sys.modules[__name__],p)
            tp= tclass()
            if hasattr(tp,'isNonAxi') and tp.isNonAxi:
                continue # skip, bc eccentricity of circ. =/= 0
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
                if isinstance(tp,testMWPotential):
                    o.integrate(times,tp._potlist,method=integrator)
                else:
                    o.integrate(times,tp,method=integrator)
                tzmax= o.zmax()
                if ii < 2 and (p == 'BurkertPotential'
                               or 'SCFPotential' in p
                               or 'FlattenedPower' in p
                               or 'RazorThinExponential' in p
                               or 'TwoPowerSpherical' in p): # no Rzderiv currently
                    tzmax_analytic= o.zmax(analytic=True,type='adiabatic')
                else:
                    tzmax_analytic= o.zmax(analytic=True)
                #print(p, integrator, tzmax, tzmax_analytic, (tzmax-tzmax_analytic)**2.)
                assert (tzmax-tzmax_analytic)**2. < 10.**ttol, \
                    "Analytically computed zmax does not agree by %g with numerical estimate for potential %s and integrator %s" %(numpy.fabs(tzmax-tzmax_analytic),p,integrator)
                assert (o.zmax(ro=8.)/8.-tzmax_analytic)**2. < 10.**ttol, \
                    "Zmax in physical coordinates does not agree with physical-scale times zmax in normalized coordinates for potential %s and integrator %s" %(p,integrator)
            if _QUICKTEST and (not 'NFW' in p or tp.isNonAxi): break
    #raise AssertionError
    return None

# Test the error for when explicit stepsize does not divide the output stepsize
def test_check_integrate_dt():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([1.,0.1,1.2,0.3,0.2,2.])
    times= numpy.linspace(0.,7.,251)
    # This shouldn't work
    try:
        o.integrate(times,lp,dt=(times[1]-times[0])/4.*1.1)
    except ValueError: pass
    else: raise AssertionError('dt that is not an integer divisor of the output step size does not raise a ValueError')
    # This should
    try:
        o.integrate(times,lp,dt=(times[1]-times[0])/4.)
    except ValueError:
        raise AssertionError('dt that is an integer divisor of the output step size raises a ValueError')
    return None

# Test that fixing the stepsize works, issue #207
def test_fixedstepsize():
    if WIN32: return None # skip on appveyor, because fails for reason that I can't figure out (runtimes[0] == 0.) and not that important
    from galpy.potential import LogarithmicHaloPotential
    import time
    # Integrators for which it should work
    integrators= ['leapfrog_c','rk4_c','rk6_c','symplec4_c','symplec6_c']
    # Somewhat long time
    times= numpy.linspace(0.,100.,30001)
    # Test the following multiples
    mults= [1.,10.]
    # Just do this for LogarithmicHaloPotential
    pot= LogarithmicHaloPotential(normalize=1.)
    planarpot= pot.toPlanar()
    types= ['full','rz','planar','r']
    # Loop through integrators and different types of orbits
    for integrator in integrators:
        for type in types:
            if type == 'full':
                o= setup_orbit_energy(pot,axi=False)
            elif type == 'rz':
                o= setup_orbit_energy(pot,axi=True)
            elif type == 'planar':
                o= setup_orbit_energy(planarpot,axi=False)
            elif type == 'r':
                o= setup_orbit_energy(planarpot,axi=True)
            runtimes= numpy.empty(len(mults))
            for ii,mult in enumerate(mults):
                start= time.time()
                o.integrate(times,pot,dt=(times[1]-times[0])/mult,
                            method=integrator)
                runtimes[ii]= time.time()-start
            for ii,mult in enumerate(mults):
                if ii == 0: continue
                # Pretty loose test, because hard to get exactly right with overhead
                assert numpy.fabs(runtimes[ii]/runtimes[0]/mults[ii]*mults[0]-1.) < 0.85, 'Runtime of integration with fixed stepsize for integrator %s, type or orbit %s, stepsize reduction %i is not %i times less (residual is %g, times %g and %g)' % (integrator,type,mults[ii],mults[ii],
numpy.fabs(runtimes[ii]/runtimes[0]/mults[ii]*mults[0]-1.),mults[ii]/mults[0],runtimes[ii]/runtimes[0])
    return None

# Test that fixing the stepsize works for integrate_dxdv
def test_fixedstepsize_dxdv():
    if WIN32: return None # skip on appveyor, because test_fixedstepsize fails for reason that I can't figure out (runtimes[0] == 0.) and not that important
    from galpy.potential import LogarithmicHaloPotential
    import time
    # Integrators for which it should work
    integrators= ['rk4_c','rk6_c']
    # Somewhat long time
    from astropy import units
    times= numpy.linspace(0.,100.,90001)/280.*units.Gyr
    # Test the following multiples
    mults= [1.,10.]
    # Just do this for LogarithmicHaloPotential
    pot= LogarithmicHaloPotential(normalize=1.)
    planarpot= pot.toPlanar()
    # Loop through integrators and different types of orbits
    for integrator in integrators:
        o= setup_orbit_energy(planarpot,axi=False)
        runtimes= numpy.empty(len(mults))
        for ii,mult in enumerate(mults):
            start= time.time()
            o.integrate_dxdv(1e-2*numpy.ones(4),
                             times,planarpot,dt=(times[1]-times[0])/mult,
                             method=integrator)
            runtimes[ii]= time.time()-start
        for ii,mult in enumerate(mults):
            if ii == 0: continue
            # Pretty loose test, because hard to get exactly right with overhead
            assert numpy.fabs(runtimes[ii]/runtimes[0]/mults[ii]*mults[0]-1.) < 0.85, 'Runtime of integration with fixed stepsize for integrator %s, type or orbit %s, stepsize reduction %i is not %i times less (residual is %g, times %g and %g)' % (integrator,type,mults[ii],mults[ii],
numpy.fabs(runtimes[ii]/runtimes[0]/mults[ii]*mults[0]-1.),mults[ii]/mults[0],runtimes[ii]/runtimes[0])
    return None

# Check that adding a linear orbit to a planar orbit gives a FullOrbit
@pytest.mark.skip(reason="Not implemented for Orbits currently")
def test_add_linear_planar_orbit():
    from galpy.orbit import FullOrbit, RZOrbit
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
    #w/ physical scale and coordinate-transformation parameters
    ro,vo,zo,solarmotion= 10.,300.,0.01,'dehnen'
    op= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=True)
    of= op+ol
    assert isinstance(of._orb,RZOrbit.RZOrbit), \
        "Sum of linearOrbit and planarROrbit does not give a FullOrbit"
    assert numpy.fabs(op._orb._ro-of._orb._ro) < 10.**-15., 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
    assert numpy.fabs(op._orb._vo-of._orb._vo) < 10.**-15., 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
    assert numpy.fabs(op._orb._zo-of._orb._zo) < 10.**-15., 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
    assert numpy.all(numpy.fabs(op._orb._solarmotion-of._orb._solarmotion) < 10.**-15.), 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
    assert op._orb._roSet == of._orb._roSet, 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
    assert op._orb._voSet == of._orb._voSet, 'Sum of orbits does not properly propagate physical scales and coordinate-transformation parameters'
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
    assert (True^o._roSet)*(True^upo._roSet), "Pickled/unpickled orbit does not agree with original orbut for roSet"
    assert (True^o._voSet)*(True^upo._voSet), "Pickled/unpickled orbit does not agree with original orbut for voSet"
    # w/ physical scales etc.
    o= Orbit([1.,0.1,1.1,0.1,0.2,2.],ro=10.,vo=300.)
    po= pickle.dumps(o)
    upo= pickle.loads(po)
    assert o.R() == upo.R(), "Pickled/unpickled orbit does not agree with original orbut for R"
    assert o.vR() == upo.vR(), "Pickled/unpickled orbit does not agree with original orbut for vR"
    assert o.vT() == upo.vT(), "Pickled/unpickled orbit does not agree with original orbut for vT"
    assert o.z() == upo.z(), "Pickled/unpickled orbit does not agree with original orbut for z"
    assert o.vz() == upo.vz(), "Pickled/unpickled orbit does not agree with original orbut for vz"
    assert o.phi() == upo.phi(), "Pickled/unpickled orbit does not agree with original orbut for phi"
    assert o._ro == upo._ro, "Pickled/unpickled orbit does not agree with original orbut for ro"
    assert o._vo == upo._vo, "Pickled/unpickled orbit does not agree with original orbut for vo"
    assert o._zo == upo._zo, "Pickled/unpickled orbit does not agree with original orbut for zo"
    assert numpy.all(o._solarmotion == upo._solarmotion), "Pickled/unpickled orbit does not agree with original orbut for solarmotion"
    assert (o._roSet)*(upo._roSet), "Pickled/unpickled orbit does not agree with original orbut for roSet"
    assert (o._voSet)*(upo._voSet), "Pickled/unpickled orbit does not agree with original orbut for voSet"
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
    assert numpy.ndim(o.L()) == 0, "planarOrbit's angular momentum isn't 1D"
    assert o.L() == 1.1, "planarOrbit's angular momentum isn't correct"
    if False: 
        #JB 5/23/2019 isn'tn sure why he ever implemented the Omega
        # keyword for L, so decided not to support this in new Orbits
        # If Omega is given, then it should be subtracted
        times= numpy.linspace(0.,2.,51)
        from galpy.potential import MWPotential
        o.integrate(times,MWPotential)
        assert numpy.fabs(o.L(t=1.,Omega=1.)-0.1) < 10.**-16., 'o.L() w/ Omega does not work'
    # For a FullOrbit, angular momentum should be 3D
    o= Orbit([1.,0.1,1.1,0.1,0.,numpy.pi/2.])
    assert o.L().shape[0] == 3, "FullOrbit's angular momentum is not 3D"
    assert numpy.fabs(o.L()[2]-1.1) < 10.**-16., "FullOrbit's Lz is not correct"
    assert numpy.fabs(o.L()[0]+0.01) < 10.**-16., "FullOrbit's Lx is not correct"
    assert numpy.fabs(o.L()[1]+0.11) < 10.**-16., "FullOrbit's Ly is not correct"
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
    assert numpy.fabs(o.vr()-0.1) < 10.**-16., 'linearOrbit vx setup does not agree with o.vr()'
    if False:
        # setphi was deprecated when moving to Orbits
        try:
            o.setphi(3.)
        except AttributeError:
            pass
        else:
            raise AssertionError('setphi applied to linearOrbit did not raise AttributeError')
    return None

def test_orbit_setup_planar():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1])
    assert o.dim() == 2, 'planarROrbit does not have dim == 2'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'planarOrbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'planarOrbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'planarOrbit vT setup does not agree with o.vT()'
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.)
        assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
        # planarROrbit no longer exists after moving to Orbits
        #assert not isinstance(o._orb,planarROrbit), 'After applying setphi, planarROrbit did not become planarOrbit'
    o= Orbit([1.,0.1,1.1,2.])
    assert o.dim() == 2, 'planarOrbit does not have dim == 2'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'planarOrbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'planarOrbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'planarOrbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.phi()-2.) < 10.**-16., 'planarOrbit phi setup does not agree with o.phi()'
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.)
        assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    #lb, plane w/ default
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.])
    obs= [8.,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.,0.,-10.,230.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-5.5, 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #also check that the ro,vo,solarmotion values are stored and used properly (issue #158 solution)
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.],
             ro=7.5)
    assert numpy.fabs(o.ll()-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.5,0.,-10.,245.]
    assert numpy.fabs(o.pmll()-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.pmbb()-0.) < 10.**-5.5, 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.],solarmotion='hogg')
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-5.5, 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit in the plane
    o= Orbit([120.,2.,0.5,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.,0.,0.],solarmotion='hogg')
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-5.5, 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    return None

def test_orbit_setup():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.2,0.3])
    assert o.dim() == 3, 'RZOrbitOrbit does not have dim == 3'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'Orbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'Orbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.vphi()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vphi()'
    assert numpy.fabs(o.z()-0.2) < 10.**-16., 'Orbit z setup does not agree with o.z()'
    assert numpy.fabs(o.vz()-0.3) < 10.**-16., 'Orbit vz setup does not agree with o.vz()'
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.)
        assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
        # FullOrbit no longer exists after switch to Orbits
        #assert isinstance(o._orb,FullOrbit), 'After applying setphi, RZOrbit did not become FullOrbit'
    o= Orbit([1.,0.1,1.1,0.2,0.3,2.])
    assert o.dim() == 3, 'FullOrbit does not have dim == 3'
    assert numpy.fabs(o.R()-1.) < 10.**-16., 'Orbit R setup does not agree with o.R()'
    assert numpy.fabs(o.vR()-0.1) < 10.**-16., 'Orbit vR setup does not agree with o.vR()'
    assert numpy.fabs(o.vT()-1.1) < 10.**-16., 'Orbit vT setup does not agree with o.vT()'
    assert numpy.fabs(o.z()-0.2) < 10.**-16., 'Orbit z setup does not agree with o.z()'
    assert numpy.fabs(o.vz()-0.3) < 10.**-16., 'Orbit vz setup does not agree with o.vz()'
    assert numpy.fabs(o.phi()-2.) < 10.**-16., 'Orbit phi setup does not agree with o.phi()'
    if False:
        # setphi was deprecated when moving to Orbits
        o.setphi(3.)
        assert numpy.fabs(o.phi()-3.) < 10.**-16., 'Orbit setphi does not agree with o.phi()'
    #Radec w/ default
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    assert numpy.fabs(o.ra()-120.) < 10.**-12., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-13., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-13., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-13., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit vlos setup does not agree with o.vlos()'
    #Radec w/ hogg
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='hogg')
    assert numpy.fabs(o.ra()-120.) < 10.**-12., 'Orbit ra setup does not agree with o.ra()'
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
    assert numpy.fabs(o.ll()-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-60.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll()-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vll()-_K) < 10.**-10., 'Orbit pmll setup does not agree with o.vll()'
    assert numpy.fabs(o.pmbb()-0.4) < 10.**-10., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vbb()-0.8*_K) < 10.**-10., 'Orbit pmbb setup does not agree with o.vbb()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #lb w/ default at the Sun
    o= Orbit([120.,60.,0.,10.,20.,30.],uvw=True,lb=True,zo=0.)
    assert numpy.fabs(o.dist()-0.) < 10.**-2., 'Orbit dist setup does not agree with o.dist()' #because of tweak in the code to deal with at the Sun
    assert (o.U()**2.+o.V()**2.+o.W()**2.-10.**2.-20.**2.-30.**2.) < 10.**-10., 'Velocity wrt the Sun when looking at Orbit at the Sun does not agree'
    assert (o.vlos()**2.-10.**2.-20.**2.-30.**2.) < 10.**-10., 'Velocity wrt the Sun when looking at Orbit at the Sun does not agree'
    #lb w/ default and UVW
    o= Orbit([120.,60.,2.,-10.,20.,-25.],lb=True,uvw=True)
    assert numpy.fabs(o.ll()-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb()-60.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist()-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.U()+10.) < 10.**-10., 'Orbit U setup does not agree with o.U()'
    assert numpy.fabs(o.V()-20.) < 10.**-10., 'Orbit V setup does not agree with o.V()'
    assert numpy.fabs(o.W()+25.) < 10.**-10., 'Orbit W setup does not agree with o.W()'
    #lb w/ default and UVW, test wrt helioXYZ
    o= Orbit([180.,0.,2.,-10.,20.,-25.],lb=True,uvw=True)
    assert numpy.fabs(o.helioX()+2.) < 10.**-10., 'Orbit helioX setup does not agree with o.helioX()'
    assert numpy.fabs(o.helioY()-0.) < 10.**-10., 'Orbit helioY setup does not agree with o.helioY()'
    assert numpy.fabs(o.helioZ()-0.) < 10.**-10., 'Orbit helioZ setup does not agree with o.helioZ()'
    assert numpy.fabs(o.U()+10.) < 10.**-10., 'Orbit U setup does not agree with o.U()'
    assert numpy.fabs(o.V()-20.) < 10.**-10., 'Orbit V setup does not agree with o.V()'
    assert numpy.fabs(o.W()+25.) < 10.**-10., 'Orbit W setup does not agree with o.W()'
    #Radec w/ hogg and obs=Orbit
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,solarmotion='hogg')
    obs= Orbit([1.,-10.1/220.,224./220,0.0208/8.,6.7/220.,0.],
               solarmotion='hogg')
    assert numpy.fabs(o.ra(obs=obs)-120.) < 10.**-10., 'Orbit ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs)-60.) < 10.**-10., 'Orbit dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs)-0.5) < 10.**-10., 'Orbit pmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.vra(obs=obs)-_K) < 10.**-10., 'Orbit pmra setup does not agree with o.vra()'
    assert numpy.fabs(o.pmdec(obs=obs)-0.4) < 10.**-10., 'Orbit pmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vdec(obs=obs)-0.8*_K) < 10.**-10., 'Orbit pmdec setup does not agree with o.vdec()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #lb, plane w/ default
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.,10.,0.])
    obs= [8.,0.]
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    obs= [8.,0.,-10.,230.]
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-10., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #lb in plane and obs=Orbit
    o= Orbit([120.,0.,2.,0.5,0.,30.],lb=True,zo=0.,solarmotion=[-10.1,4.,0.])
    obs= Orbit([1.,-10.1/220.,224./220,0.],solarmotion='hogg')
    assert numpy.fabs(o.ll(obs=obs)-120.) < 10.**-10., 'Orbit ll setup does not agree with o.ll()'
    assert numpy.fabs(o.bb(obs=obs)-0.) < 10.**-10., 'Orbit bb setup does not agree with o.bb()'
    assert numpy.fabs(o.dist(obs=obs)-2.) < 10.**-10., 'Orbit dist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmll(obs=obs)-0.5) < 10.**-10., 'Orbit pmll setup does not agree with o.pmll()'
    assert numpy.fabs(o.pmbb(obs=obs)-0.) < 10.**-10., 'Orbit pmbb setup does not agree with o.pmbb()'
    assert numpy.fabs(o.vlos(obs=obs)-30.) < 10.**-10., 'Orbit vlos setup does not agree with o.vlos()'
    #test galactocentric spherical coordinates
    o= Orbit([2.,2.**0.5,1.,2.,2.**0.5,0.5])
    assert numpy.fabs(o.vr()-2.) < 10.**-10., 'Orbit galactocentric spherical coordinates are not correct'
    assert numpy.fabs(o.vtheta()-0.) < 10.**-10., 'Orbit galactocentric spherical coordinates are not correct'
    assert numpy.fabs(o.theta()-numpy.pi/4) < 10.**-10.,  'Orbit galactocentric spherical coordinates are not correct'
    return None

def test_orbit_setup_SkyCoord():
    # Only run this for astropy>3
    if not _APY3: return None
    from galpy.orbit import Orbit
    import astropy.coordinates as apycoords
    import astropy.units as u
    ra= 120.*u.deg
    dec= 60.*u.deg
    distance= 2.*u.kpc
    pmra= 0.5*u.mas/u.yr
    pmdec= 0.4*u.mas/u.yr
    vlos= 30.*u.km/u.s
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs')
    #w/ default
    o= Orbit(c)
    # galpy's sky is not exactly aligned with astropy's sky, so celestials are slightly off
    assert numpy.fabs(o.ra()-120.) < 10.**-8., 'Orbit SkyCoord ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-8., 'Orbit SkyCoord dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-14., 'Orbit SkyCoorddist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-8., 'Orbit SkyCoordpmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-8., 'Orbit SkyCoordpmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit SkyCoordvlos setup does not agree with o.vlos()'
    #Radec w/ hogg
    o= Orbit(c,solarmotion='hogg')
    assert numpy.fabs(o.ra()-120.) < 10.**-8., 'Orbit SkyCoordra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-8., 'Orbit SkyCoorddec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-13., 'Orbit SkyCoorddist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-8., 'Orbit SkyCoordpmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-8., 'Orbit SkyCoordpmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit SkyCoordvlos setup does not agree with o.vlos()'
    #Radec w/ dehnen and diff ro,vo
    o= Orbit(c,solarmotion='dehnen',vo=240.,ro=7.5,zo=0.01)
    obs= [7.5,0.,0.01,-10.,245.25,7.17]
    assert numpy.fabs(o.ra(obs=obs,ro=7.5)-120.) < 10.**-8., 'Orbit SkyCoordra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec(obs=obs,ro=7.5)-60.) < 10.**-8., 'Orbit SkyCoorddec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist(obs=obs,ro=7.5)-2.) < 10.**-13., 'Orbit SkyCoorddist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra(obs=obs,ro=7.5,vo=240.)-0.5) < 10.**-8., 'Orbit SkyCoordpmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec(obs=obs,ro=7.5,vo=240.)-0.4) < 10.**-8., 'Orbit SkyCoordpmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos(obs=obs,ro=7.5,vo=240.)-30.) < 10.**-13., 'Orbit SkyCoordvlos setup does not agree with o.vlos()'
    # Now specifying the coordinate conversion parameters in the SkyCoord
    v_sun= apycoords.CartesianDifferential([-11.1,215.,3.25]*u.km/u.s)
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                          galcen_v_sun=v_sun)
    o= Orbit(c)
    assert numpy.fabs(o.ra()-120.) < 10.**-8., 'Orbit SkyCoord ra setup does not agree with o.ra()'
    assert numpy.fabs(o.dec()-60.) < 10.**-8., 'Orbit SkyCoord dec setup does not agree with o.dec()'
    assert numpy.fabs(o.dist()-2.) < 10.**-14., 'Orbit SkyCoorddist setup does not agree with o.dist()'
    assert numpy.fabs(o.pmra()-0.5) < 10.**-8., 'Orbit SkyCoordpmra setup does not agree with o.pmra()'
    assert numpy.fabs(o.pmdec()-0.4) < 10.**-8., 'Orbit SkyCoordpmdec setup does not agree with o.pmdec()'
    assert numpy.fabs(o.vlos()-30.) < 10.**-13., 'Orbit SkyCoordvlos setup does not agree with o.vlos()'
    # Also test that the coordinate-transformation parameters are properly read
    assert numpy.fabs(o._ro-numpy.sqrt(10.**2.-1.**2.)) < 1e-12, 'Orbit SkyCoord setup does not properly store ro'
    assert numpy.fabs(o._ro-numpy.sqrt(10.**2.-1.**2.)) < 1e-12, 'Orbit SkyCoord setup does not properly store ro'
    assert numpy.fabs(o._zo-1.) < 1e-12, 'Orbit SkyCoord setup does not properly store zo'
    assert numpy.all(numpy.fabs(o._solarmotion-numpy.array([[11.1,-5.,3.25]])) < 1e-12), 'Orbit SkyCoord setup does not properly store solarmotion'
    # If we only specify galcen_distance, but not z_sun, zo --> 0
    # Now specifying the coordinate conversion parameters in the SkyCoord
    v_sun= apycoords.CartesianDifferential([-11.1,215.,3.25]*u.km/u.s)
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,
                          galcen_v_sun=v_sun)
    o= Orbit(c)
    assert numpy.fabs(o._zo-0.) < 1e-12, 'Orbit SkyCoord setup does not properly store zo'
    # If we specify both z_sun and zo, they need to be consistent
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                          galcen_v_sun=v_sun)
    with pytest.raises(ValueError) as excinfo:
        o= Orbit(c,zo=0.025)
    # If ro and galcen_distance are both specified, warn if they are not consistent
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                          galcen_v_sun=v_sun)
    with pytest.warns(None) as record:
        o= Orbit(c,ro=10.)
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "Orbit's initialization normalization ro and zo are incompatible with SkyCoord's galcen_distance (should have galcen_distance^2 = ro^2 + zo^2)")
    assert raisedWarning, "Orbit initialization with SkyCoord with galcen_distance incompatible with ro should have raised a warning, but didn't"
    # If ro and galcen_distance are both specified, don't warn if they *are* consistent (issue #370)
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,z_sun=1.*u.kpc,
                          galcen_v_sun=v_sun)
    with pytest.warns(None) as record:
        o= Orbit(c,ro=numpy.sqrt(10.**2.-1.**2.))
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "Orbit's initialization normalization ro and zo are incompatible with SkyCoord's galcen_distance (should have galcen_distance^2 = ro^2 + zo^2)")
    assert not raisedWarning, "Orbit initialization with SkyCoord with galcen_distance compatible with ro shouldn't have raised a warning, but did"
    # If we specify both v_sun and solarmotion, they need to be consistent
    v_sun= apycoords.CartesianDifferential([-11.1,215.,3.25]*u.km/u.s)
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          pm_ra_cosdec=pmra,pm_dec=pmdec,radial_velocity=vlos,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,
                          galcen_v_sun=v_sun)
    # This should be fine
    o= Orbit(c,solarmotion=[11.1,-5.,3.25])
    # This shouldn't be
    with pytest.raises(ValueError) as excinfo:
        o= Orbit(c,solarmotion=[11.,-4.,2.25])
    # Should get error if we give a SkyCoord without velocities
    c= apycoords.SkyCoord(ra=ra,dec=dec,distance=distance,
                          frame='icrs',
                          galcen_distance=10.*u.kpc,
                          galcen_v_sun=v_sun)
    with pytest.raises(TypeError) as excinfo:
        o= Orbit(c)
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
    ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
    obs= Orbit([1.,0.1,1.1,0.3,0.],ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    obsp= obs.toPlanar()
    assert obsp.dim() == 2, 'toPlanar does not generate an Orbit w/ dim=2 for RZOrbit'
    assert obsp.R() == obs.R(), 'Planar orbit generated w/ toPlanar does not have the correct R'
    assert obsp.vR() == obs.vR(), 'Planar orbit generated w/ toPlanar does not have the correct vR'
    assert obsp.vT() == obs.vT(), 'Planar orbit generated w/ toPlanar does not have the correct vT'
    assert numpy.fabs(obs._ro-obsp._ro) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._vo-obsp._vo) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._zo-obsp._zo) < 10.**-15., 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.all(numpy.fabs(obs._solarmotion-obsp._solarmotion) < 10.**-15.), 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._roSet == obsp._roSet, 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._voSet == obsp._voSet, 'Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it'
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
    assert obsl.dim() == 1, 'toLinear does not generate an Orbit w/ dim=1 for FullOrbit'
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
    # w/ scales
    ro,vo= 10.,300.
    obs= Orbit([1.,0.1,1.1,0.3,0.,2.],ro=ro,vo=vo)
    obsl= obs.toLinear()
    assert obsl.dim() == 1, 'toLinear does not generate an Orbit w/ dim=1 for FullOrbit'
    assert obsl.x() == obs.z(), 'Linear orbit generated w/ toLinear does not have the correct z'
    assert obsl.vx() == obs.vz(), 'Linear orbit generated w/ toLinear does not have the correct vx'
    assert numpy.fabs(obs._ro-obsl._ro) < 10.**-15., 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert numpy.fabs(obs._vo-obsl._vo) < 10.**-15., 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert (obsl._zo is None), 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert (obsl._solarmotion is None), 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._roSet == obsl._roSet, 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
    assert obs._voSet == obsl._voSet, 'Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it'
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
    try:
        o.theta()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.theta() for planarROrbit should have raise AttributeError, but did not")
    try:
        o.vtheta()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vtheta() for planarROrbit should have raise AttributeError, but did not")
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
    try:
        o.theta()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.theta() for planarROrbit should have raise AttributeError, but did not")
    try:
        o.vtheta()
    except AttributeError:
        pass
    else:
        raise AssertionError("o.vtheta() for planarROrbit should have raise AttributeError, but did not")
    return None

# Test reversing an orbit
def test_flip():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    for ii in range(5):
        #Scales to test that these are properly propagated to the new Orbit
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbit_flip(llp,ro,vo,zo,solarmotion,axi=False)
        of= o.flip()
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.abs(o.x()-of.x()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vx()+of.vx()) < 10.**-10., 'o.flip() did not work as expected'
        else:
            assert numpy.abs(o.R()-of.R()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vR()+of.vR()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vT()+of.vT()) < 10.**-10., 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.abs(o.phi()-of.phi()) < 10.**-10., 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.abs(o.z()-of.z()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vz()+of.vz()) < 10.**-10., 'o.flip() did not work as expected'
    return None

# Test reversing an orbit inplace
def test_flip_inplace():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbit_flip(llp,ro,vo,zo,solarmotion,axi=False)
        of= o()
        of.flip(inplace=True)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.abs(o.x()-of.x()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vx()+of.vx()) < 10.**-10., 'o.flip() did not work as expected'
        else:
            assert numpy.abs(o.R()-of.R()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vR()+of.vR()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vT()+of.vT()) < 10.**-10., 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.abs(o.phi()-of.phi()) < 10.**-10., 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.abs(o.z()-of.z()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vz()+of.vz()) < 10.**-10., 'o.flip() did not work as expected'
    return None

# Test reversing an orbit inplace after orbit integration
def test_flip_inplace_integrated():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    ts= numpy.linspace(0.,1.,11)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbit_flip(llp,ro,vo,zo,solarmotion,axi=False)
        of= o()
        if ii < 2 or ii == 3:
            o.integrate(ts,lp)
            of.integrate(ts,lp)
        elif ii == 2:
            o.integrate(ts,plp)
            of.integrate(ts,plp)
        else:
            o.integrate(ts,llp)
            of.integrate(ts,llp)
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o= o(0.5)
        of= of(0.5)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.abs(o.x()-of.x()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vx()+of.vx()) < 10.**-10., 'o.flip() did not work as expected'
        else:
            assert numpy.abs(o.R()-of.R()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vR()+of.vR()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vT()+of.vT()) < 10.**-10., 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.abs(o.phi()-of.phi()) < 10.**-10., 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.abs(o.z()-of.z()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vz()+of.vz()) < 10.**-10., 'o.flip() did not work as expected'
    return None

# Test reversing an orbit inplace after orbit integration, and after having
# once evaluated the orbit before flipping inplace (#345)
# only difference wrt previous test is a line that evaluates of before
# flipping
def test_flip_inplace_integrated_evaluated():
    from galpy.potential import LogarithmicHaloPotential
    lp= LogarithmicHaloPotential(normalize=1.,q=0.9)
    plp= lp.toPlanar()
    llp= lp.toVertical(1.)
    ts= numpy.linspace(0.,1.,11)
    for ii in range(5):
        #Scales (not really necessary for this test)
        ro,vo,zo,solarmotion= 10.,300.,0.01,'schoenrich'
        if ii == 0: #axi, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 1: #track azimuth, full
            o= setup_orbit_flip(lp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 2: #axi, planar
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=True)
        elif ii == 3: #track azimuth, full
            o= setup_orbit_flip(plp,ro,vo,zo,solarmotion,axi=False)
        elif ii == 4: #linear orbit
            o= setup_orbit_flip(llp,ro,vo,zo,solarmotion,axi=False)
        of= o()
        if ii < 2 or ii == 3:
            o.integrate(ts,lp)
            of.integrate(ts,lp)
        elif ii == 2:
            o.integrate(ts,plp)
            of.integrate(ts,plp)
        else:
            o.integrate(ts,llp)
            of.integrate(ts,llp)
        # Evaluate, make sure it is at an interpolated time!
        dum= of.R(0.52)
        # Now flip
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o= o(0.52)
        of= of(0.52)
        #First check that the scales have been propagated properly
        assert numpy.fabs(o._ro-of._ro) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert numpy.fabs(o._vo-of._vo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert (o._zo is None)*(of._zo is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert (o._solarmotion is None)*(of._solarmotion is None), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        else:
            assert numpy.fabs(o._zo-of._zo) < 10.**-15., 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
            assert numpy.all(numpy.fabs(o._solarmotion-of._solarmotion) < 10.**-15.), 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._roSet == of._roSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        assert o._voSet == of._voSet, 'o.flip() did not conserve physical scales and coordinate-transformation parameters'
        if ii == 4:
            assert numpy.abs(o.x()-of.x()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vx()+of.vx()) < 10.**-10., 'o.flip() did not work as expected'
        else:
            assert numpy.abs(o.R()-of.R()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vR()+of.vR()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vT()+of.vT()) < 10.**-10., 'o.flip() did not work as expected'
        if ii % 2 == 1:
            assert numpy.abs(o.phi()-of.phi()) < 10.**-10., 'o.flip() did not work as expected'
        if ii < 2:
            assert numpy.abs(o.z()-of.z()) < 10.**-10., 'o.flip() did not work as expected'
            assert numpy.abs(o.vz()+of.vz()) < 10.**-10., 'o.flip() did not work as expected'
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

# Test new orbits formed from __call__
def test_newOrbit():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.,0.])
    ts= numpy.linspace(0.,1.,21) #v. quick orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o.integrate(ts,lp)
    no= o(ts[-1]) #new orbit
    assert no.R() == o.R(ts[-1]), "New orbit formed from calling an old orbit does not have the correct R"
    assert no.vR() == o.vR(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vR"
    assert no.vT() == o.vT(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vT"
    assert no.z() == o.z(ts[-1]), "New orbit formed from calling an old orbit does not have the correct z"
    assert no.vz() == o.vz(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vz"
    assert no.phi() == o.phi(ts[-1]), "New orbit formed from calling an old orbit does not have the correct phi"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    #Also test this for multiple time outputs
    nos= o(ts[-2:]) #new orbits
    #First t
    assert numpy.fabs(nos[0].R()-o.R(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.fabs(nos[0].vR()-o.vR(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.fabs(nos[0].vT()-o.vT(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.fabs(nos[0].z()-o.z(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.fabs(nos[0].vz()-o.vz(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.fabs(nos[0].phi()-o.phi(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct phi"
    assert not nos[0]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    #Second t
    assert numpy.fabs(nos[1].R()-o.R(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.fabs(nos[1].vR()-o.vR(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.fabs(nos[1].vT()-o.vT(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.fabs(nos[1].z()-o.z(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.fabs(nos[1].vz()-o.vz(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.fabs(nos[1].phi()-o.phi(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct phi"
    assert not nos[1]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    return None

# Test new orbits formed from __call__, before integration
def test_newOrbit_b4integration():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.,0.])
    no= o(0.) #New orbit formed before integration
    assert numpy.fabs(no.R()-o.R()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.fabs(no.vR()-o.vR()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.fabs(no.vT()-o.vT()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.fabs(no.z()-o.z()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.fabs(no.vz()-o.vz()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.fabs(no.phi()-o.phi()) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct phi"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    return None

# Test that we can still get outputs when there aren't enough points for an actual interpolation
def test_newOrbit_badinterpolation():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.,0.])
    ts= numpy.linspace(0.,1.,3) #v. quick orbit integration, w/ not enough points for interpolation
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o.integrate(ts,lp)
    no= o(ts[-1]) #new orbit
    print("Done")
    assert no.R() == o.R(ts[-1]), "New orbit formed from calling an old orbit does not have the correct R"
    assert no.vR() == o.vR(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vR"
    assert no.vT() == o.vT(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vT"
    assert no.z() == o.z(ts[-1]), "New orbit formed from calling an old orbit does not have the correct z"
    assert no.vz() == o.vz(ts[-1]), "New orbit formed from calling an old orbit does not have the correct vz"
    assert no.phi() == o.phi(ts[-1]), "New orbit formed from calling an old orbit does not have the correct phi"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not no._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    #Also test this for multiple time outputs
    nos= o(ts[-2:]) #new orbits
    #First t
    assert numpy.fabs(nos[0].R()-o.R(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.fabs(nos[0].vR()-o.vR(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.fabs(nos[0].vT()-o.vT(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.fabs(nos[0].z()-o.z(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.fabs(nos[0].vz()-o.vz(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.fabs(nos[0].phi()-o.phi(ts[-2])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct phi"
    assert not nos[0]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[0]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    #Second t
    assert numpy.fabs(nos[1].R()-o.R(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.fabs(nos[1].vR()-o.vR(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.fabs(nos[1].vT()-o.vT(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.fabs(nos[1].z()-o.z(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.fabs(nos[1].vz()-o.vz(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.fabs(nos[1].phi()-o.phi(ts[-1])) < 10.**-10., "New orbit formed from calling an old orbit does not have the correct phi"
    assert not nos[1]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._roSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    assert not nos[1]._voSet, "New orbit formed from calling an old orbit does not have the correct roSet"
    #Try point in between, shouldn't work
    try: no= o(0.6)
    except LookupError: pass
    else: raise AssertionError('Orbit interpolation with not enough points to interpolate should raise LookUpError, but did not')
    return None

# Check the routines that should return physical coordinates
def test_physical_output():
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion
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
            assert numpy.fabs(o.r()/ro-o.r(use_physical=False)) < 10.**-10., 'o.r() output for Orbit setup with ro= does not work as expected'
            assert numpy.fabs(o.z()/ro-o.z(use_physical=False)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected'
        #Test velocities
        assert numpy.fabs(o.vR()/vo-o.vR(use_physical=False)) < 10.**-10., 'o.vR() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.vT()/vo-o.vT(use_physical=False)) < 10.**-10., 'o.vT() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.vphi()*ro/vo-o.vphi(use_physical=False)) < 10.**-10., 'o.vphi() output for Orbit setup with vo= does not work as expected'
        if ii % 2 == 1:
            assert numpy.fabs(o.vx()/vo-o.vx(use_physical=False)) < 10.**-10., 'o.vx() output for Orbit setup with vo= does not work as expected'
            assert numpy.fabs(o.vy()/vo-o.vy(use_physical=False)) < 10.**-10., 'o.vy() output for Orbit setup with vo= does not work as expected'
        if ii < 2:
            assert numpy.fabs(o.vz()/vo-o.vz(use_physical=False)) < 10.**-10., 'o.vz() output for Orbit setup with vo= does not work as expected'
        #Test energies
        assert numpy.fabs(o.E(pot=lp)/vo**2.-o.E(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.Jacobi(pot=lp)/vo**2.-o.Jacobi(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected'
        if ii < 2:
            assert numpy.fabs(o.ER(pot=lp)/vo**2.-o.ER(pot=lp,use_physical=False)) < 10.**-10., 'o.ER() output for Orbit setup with vo= does not work as expected'
            assert numpy.fabs(o.Ez(pot=lp)/vo**2.-o.Ez(pot=lp,use_physical=False)) < 10.**-10., 'o.Ez() output for Orbit setup with vo= does not work as expected'
        #Test angular momentun
        if ii > 0:
            assert numpy.all(numpy.fabs(o.L()/vo/ro-o.L(use_physical=False)) < 10.**-10.), 'o.L() output for Orbit setup with ro=,vo= does not work as expected'
        # Test action-angle functions
        if ii == 1:
            assert numpy.fabs(o.jr(pot=lp,type='staeckel',delta=0.5)/vo/ro-o.jr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jp(pot=lp,type='staeckel',delta=0.5)/vo/ro-o.jp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jz(pot=lp,type='staeckel',delta=0.5)/vo/ro-o.jz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tr(pot=lp,type='staeckel',delta=0.5)/conversion.time_in_Gyr(vo,ro)-o.Tr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tp(pot=lp,type='staeckel',delta=0.5)/conversion.time_in_Gyr(vo,ro)-o.Tp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tz(pot=lp,type='staeckel',delta=0.5)/conversion.time_in_Gyr(vo,ro)-o.Tz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Or(pot=lp,type='staeckel',delta=0.5)/conversion.freq_in_Gyr(vo,ro)-o.Or(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Or() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Op(pot=lp,type='staeckel',delta=0.5)/conversion.freq_in_Gyr(vo,ro)-o.Op(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Op() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Oz(pot=lp,type='staeckel',delta=0.5)/conversion.freq_in_Gyr(vo,ro)-o.Oz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Oz() output for Orbit setup with ro=,vo= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    assert numpy.fabs((o.time(1.,ro=ro,vo=vo)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    assert numpy.fabs((o.time(1.,use_physical=False)-1.)) < 10.**-10., 'o.time() in physical coordinates does not work as expected'
    return None

# Check that the routines that should return physical coordinates are turned off by turn_physical_off
def test_physical_output_off():
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
        #turn off
        o.turn_physical_off()
        #Test positions
        assert numpy.fabs(o.R()-o.R(use_physical=False)) < 10.**-10., 'o.R() output for Orbit setup with ro= does not work as expected when turned off'
        if ii % 2 == 1:
            assert numpy.fabs(o.x()-o.x(use_physical=False)) < 10.**-10., 'o.x() output for Orbit setup with ro= does not work as expected when turned off'
            assert numpy.fabs(o.y()-o.y(use_physical=False)) < 10.**-10., 'o.y() output for Orbit setup with ro= does not work as expected when turned off'
        if ii < 2:
            assert numpy.fabs(o.z()-o.z(use_physical=False)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected when turned off'
            assert numpy.fabs(o.r()-o.r(use_physical=False)) < 10.**-10., 'o.r() output for Orbit setup with ro= does not work as expected when turned off'
        #Test velocities
        assert numpy.fabs(o.vR()-o.vR(use_physical=False)) < 10.**-10., 'o.vR() output for Orbit setup with vo= does not work as expected when turned off'
        assert numpy.fabs(o.vT()-o.vT(use_physical=False)) < 10.**-10., 'o.vT() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.vphi()-o.vphi(use_physical=False)) < 10.**-10., 'o.vphi() output for Orbit setup with vo= does not work as expected when turned off'
        if ii % 2 == 1:
            assert numpy.fabs(o.vx()-o.vx(use_physical=False)) < 10.**-10., 'o.vx() output for Orbit setup with vo= does not work as expected when turned off'
            assert numpy.fabs(o.vy()-o.vy(use_physical=False)) < 10.**-10., 'o.vy() output for Orbit setup with vo= does not work as expected when turned off'
        if ii < 2:
            assert numpy.fabs(o.vz()-o.vz(use_physical=False)) < 10.**-10., 'o.vz() output for Orbit setup with vo= does not work as expected when turned off'
        #Test energies
        assert numpy.fabs(o.E(pot=lp)-o.E(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned off'
        assert numpy.fabs(o.Jacobi(pot=lp)-o.Jacobi(pot=lp,use_physical=False)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned off'
        if ii < 2:
            assert numpy.fabs(o.ER(pot=lp)-o.ER(pot=lp,use_physical=False)) < 10.**-10., 'o.ER() output for Orbit setup with vo= does not work as expected when turned off'
            assert numpy.fabs(o.Ez(pot=lp)-o.Ez(pot=lp,use_physical=False)) < 10.**-10., 'o.Ez() output for Orbit setup with vo= does not work as expected when turned off'
        #Test angular momentun
        if ii > 0:
            assert numpy.all(numpy.fabs(o.L()-o.L(use_physical=False)) < 10.**-10.), 'o.L() output for Orbit setup with ro=,vo= does not work as expected when turned off'
        # Test action-angle functions
        if ii == 1:
            assert numpy.fabs(o.jr(pot=lp,type='staeckel',delta=0.5)-o.jr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jp(pot=lp,type='staeckel',delta=0.5)-o.jp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jz(pot=lp,type='staeckel',delta=0.5)-o.jz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.jz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tr(pot=lp,type='staeckel',delta=0.5)-o.Tr(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tp(pot=lp,type='staeckel',delta=0.5)-o.Tp(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tz(pot=lp,type='staeckel',delta=0.5)-o.Tz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Tz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Or(pot=lp,type='staeckel',delta=0.5)-o.Or(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Or() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Op(pot=lp,type='staeckel',delta=0.5)-o.Op(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Op() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Oz(pot=lp,type='staeckel',delta=0.5)-o.Oz(pot=lp,type='staeckel',delta=0.5,use_physical=False)) < 10.**-10., 'o.Oz() output for Orbit setup with ro=,vo= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-1.)) < 10.**-10., 'o.time() in physical coordinates does not work as expected when turned off'
    assert numpy.fabs((o.time(1.,ro=ro,vo=vo)-ro/vo/1.0227121655399913)) < 10.**-10., 'o.time() in physical coordinates does not work as expected when turned off'
    return None

# Check that the routines that should return physical coordinates are turned
# back on by turn_physical_on
def test_physical_output_on():
    from galpy.potential import LogarithmicHaloPotential
    from astropy import units
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
        o_orig= o()
        #turn off and on
        o.turn_physical_off()
        if ii == 0:
            o.turn_physical_on(ro=ro,vo=vo)
        elif ii == 1:
            o.turn_physical_on(ro=ro*units.kpc,vo=vo*units.km/units.s)
        else:
            o.turn_physical_on()
        #Test positions
        assert numpy.fabs(o.R()-o_orig.R(use_physical=True)) < 10.**-10., 'o.R() output for Orbit setup with ro= does not work as expected when turned back on'
        if ii % 2 == 1:
            assert numpy.fabs(o.x()-o_orig.x(use_physical=True)) < 10.**-10., 'o.x() output for Orbit setup with ro= does not work as expected when turned back on'
            assert numpy.fabs(o.y()-o_orig.y(use_physical=True)) < 10.**-10., 'o.y() output for Orbit setup with ro= does not work as expected when turned back on'
        if ii < 2:
            assert numpy.fabs(o.z()-o_orig.z(use_physical=True)) < 10.**-10., 'o.z() output for Orbit setup with ro= does not work as expected when turned back on'
        #Test velocities
        assert numpy.fabs(o.vR()-o_orig.vR(use_physical=True)) < 10.**-10., 'o.vR() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.vT()-o_orig.vT(use_physical=True)) < 10.**-10., 'o.vT() output for Orbit setup with vo= does not work as expected'
        assert numpy.fabs(o.vphi()-o_orig.vphi(use_physical=True)) < 10.**-10., 'o.vphi() output for Orbit setup with vo= does not work as expected when turned back on'
        if ii % 2 == 1:
            assert numpy.fabs(o.vx()-o_orig.vx(use_physical=True)) < 10.**-10., 'o.vx() output for Orbit setup with vo= does not work as expected when turned back on'
            assert numpy.fabs(o.vy()-o_orig.vy(use_physical=True)) < 10.**-10., 'o.vy() output for Orbit setup with vo= does not work as expected when turned back on'
        if ii < 2:
            assert numpy.fabs(o.vz()-o_orig.vz(use_physical=True)) < 10.**-10., 'o.vz() output for Orbit setup with vo= does not work as expected when turned back on'
        #Test energies
        assert numpy.fabs(o.E(pot=lp)-o_orig.E(pot=lp,use_physical=True)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned back on'
        assert numpy.fabs(o.Jacobi(pot=lp)-o_orig.Jacobi(pot=lp,use_physical=True)) < 10.**-10., 'o.E() output for Orbit setup with vo= does not work as expected when turned back on'
        if ii < 2:
            assert numpy.fabs(o.ER(pot=lp)-o_orig.ER(pot=lp,use_physical=True)) < 10.**-10., 'o.ER() output for Orbit setup with vo= does not work as expected when turned back on'
            assert numpy.fabs(o.Ez(pot=lp)-o_orig.Ez(pot=lp,use_physical=True)) < 10.**-10., 'o.Ez() output for Orbit setup with vo= does not work as expected when turned back on'
        #Test angular momentun
        if ii > 0:
            assert numpy.all(numpy.fabs(o.L()-o_orig.L(use_physical=True)) < 10.**-10.), 'o.L() output for Orbit setup with ro=,vo= does not work as expected when turned back on'
        # Test action-angle functions
        if ii == 1:
            assert numpy.fabs(o.jr(pot=lp,type='staeckel',delta=0.5)-o_orig.jr(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jp(pot=lp,type='staeckel',delta=0.5)-o_orig.jp(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.jz(pot=lp,type='staeckel',delta=0.5)-o_orig.jz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.jz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tr(pot=lp,type='staeckel',delta=0.5)-o_orig.Tr(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tr() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tp(pot=lp,type='staeckel',delta=0.5)-o_orig.Tp(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tp() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Tz(pot=lp,type='staeckel',delta=0.5)-o_orig.Tz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Tz() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Or(pot=lp,type='staeckel',delta=0.5)-o_orig.Or(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Or() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Op(pot=lp,type='staeckel',delta=0.5)-o_orig.Op(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Op() output for Orbit setup with ro=,vo= does not work as expected'
            assert numpy.fabs(o.Oz(pot=lp,type='staeckel',delta=0.5)-o_orig.Oz(pot=lp,type='staeckel',delta=0.5,use_physical=True)) < 10.**-10., 'o.Oz() output for Orbit setup with ro=,vo= does not work as expected'
    #Also test the times
    assert numpy.fabs((o.time(1.)-o_orig.time(1.,use_physical=True))) < 10.**-10., 'o_orig.time() in physical coordinates does not work as expected when turned back on'
    return None

# Test that physical scales are propagated correctly when a new orbit is formed by calling an old orbit
def test_physical_newOrbit():
    from galpy.orbit import Orbit
    o= Orbit([1.,0.1,1.1,0.1,0.,0.],ro=9.,vo=230.,
             zo=0.02,solarmotion=[-5.,15.,25.])
    ts= numpy.linspace(0.,1.,21) #v. quick orbit integration
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o.integrate(ts,lp)
    no= o(ts[-1]) #new orbit
    assert no._ro == 9., "New orbit formed from calling old orbit's ro is not that of the old orbit"
    assert no._vo == 230., "New orbit formed from calling old orbit's vo is not that of the old orbit"
    assert no._ro == 9., "New orbit formed from calling old orbit's ro is not that of the old orbit"
    assert no._vo == 230., "New orbit formed from calling old orbit's vo is not that of the old orbit"
    assert no._roSet, "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    assert no._voSet, "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    assert no._roSet, "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    assert no._voSet, "New orbit formed from calling old orbit's roSet is not that of the old orbit"
    assert no._zo == 0.02, "New orbit formed from calling old orbit's zo is not that of the old orbit"
    assert no._solarmotion[0] == -5., "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    assert no._solarmotion[1] == 15., "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    assert no._solarmotion[2] == 25., "New orbit formed from calling old orbit's solarmotion is not that of the old orbit"
    return None

#Test the orbit interpolation
def test_interpolation_issue187():
    #Test that fails because of issue 187 reported by Mark Fardal
    from galpy.orbit import Orbit
    from scipy import interpolate
    pot = potential.IsochronePotential(b=1./7.,normalize=True)
    R, vR, vT, z, vz, phi = 1.,0.0,0.8,0.,0.,0.
    orb = Orbit(vxvv=[R, vR, vT, z, vz, phi])
    ts = numpy.linspace(0.,10.,1000)
    orb.integrate(ts, pot)
    orbpts = orb.getOrbit()
    #detect phase wrap
    pdiff= orbpts[:,5]-numpy.roll(orbpts[:,5],1)
    phaseWrapIndx= numpy.where(pdiff < -6.)[0][0]
    tsPreWrap = numpy.linspace(ts[phaseWrapIndx]-5.e-2,
                               ts[phaseWrapIndx]-0.002,100)
    tsPostWrap = numpy.linspace(ts[phaseWrapIndx]+0.002,
                                ts[phaseWrapIndx]+5.e-2,100)
    #Interpolate just before and after the phase-wrap
    preWrapInterpolate=\
        interpolate.InterpolatedUnivariateSpline(ts[phaseWrapIndx-11:phaseWrapIndx-1],
                                                 orbpts[phaseWrapIndx-11:phaseWrapIndx-1,5])
    postWrapInterpolate=\
        interpolate.InterpolatedUnivariateSpline(ts[phaseWrapIndx:phaseWrapIndx+10],
                                                 orbpts[phaseWrapIndx:phaseWrapIndx+10,5])
    assert numpy.all(numpy.fabs((((preWrapInterpolate(tsPreWrap)+numpy.pi) % (2.*numpy.pi) - numpy.pi))-orb.phi(tsPreWrap)) < 10.**-5.), 'phase interpolation near a phase-wrap does not work'
    assert numpy.all(numpy.fabs((((postWrapInterpolate(tsPostWrap)+numpy.pi) % (2.*numpy.pi) - numpy.pi))-orb.phi(tsPostWrap)) < 10.**-5.), 'phase interpolation near a phase-wrap does not work'
    return None

# Test that fitting an orbit works
def test_orbitfit():
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([0.8,0.3,1.3,0.4,0.2,2.])
    ts= numpy.linspace(0.,1.,1001)
    o.integrate(ts,lp)
    #Create orbit points from this integrated orbit, each 100th point
    vxvv= o.getOrbit()[::100,:]
    #now fit
    of= Orbit.from_fit(o.vxvv[0],vxvv,pot=lp,tintJ=1.5)
    assert numpy.all(comp_orbfit(of,vxvv,numpy.linspace(0.,2.,1001),lp) < 10.**-7.), 'Orbit fit in configuration space does not work'
    return None

def test_orbitfit_potinput():
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([0.8,0.3,1.3,0.4,0.2,2.])
    ts= numpy.linspace(0.,1.,1001)
    o.integrate(ts,lp)
    #Create orbit points from this integrated orbit, each 100th point
    vxvv= o.getOrbit()[::100,:]
    #now fit, using another orbit instance, without potential, should error
    of= o()
    try:
        Orbit.from_fit(o.vxvv[0],vxvv,pot=None,tintJ=1.5)
    except AttributeError: pass
    else: raise AssertionError('Orbit fit w/o potential does not raise AttributeError')
    #Now give a potential to of
    of= Orbit.from_fit(o.vxvv[0],vxvv,pot=lp,tintJ=1.5)
    assert numpy.all(comp_orbfit(of,vxvv,numpy.linspace(0.,2.,1001),lp) < 10.**-7.), 'Orbit fit in configuration space does not work'
    return None

# Test orbit fit in observed Galactic coordinates
def test_orbitfit_lb():
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    o= Orbit([0.8,0.3,1.3,0.4,0.2,2.])
    ts= numpy.linspace(0.,1.,1001)
    o.integrate(ts,lp)
    #Create orbit points from this integrated orbit, each 100th point
    vxvv= []
    for ii in range(10):
        vxvv.append([o.ll(ii/10.),o.bb(ii/10.),o.dist(ii/10.),
                     o.pmll(ii/10.),o.pmbb(ii/10.),o.vlos(ii/10.)])
    vxvv= numpy.array(vxvv)
    #now fit
    of= Orbit.from_fit([o.ll(),o.bb(),o.dist(),o.pmll(),o.pmbb(),o.vlos()],
                       vxvv,pot=lp,tintJ=1.5,lb=True,
                       vxvv_err=0.01*numpy.ones_like(vxvv))
    compf= comp_orbfit(of,vxvv,numpy.linspace(0.,2.,1001),lp,lb=True)
    assert numpy.all(compf < 10.**-4.), 'Orbit fit in lb space does not work'
    return None

# Test orbit fit in observed equatorial coordinates
def test_orbitfit_radec():
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    ro, vo= 9., 230.
    o= Orbit([0.8,0.3,1.3,0.4,0.2,2.],ro=ro,vo=vo)
    ts= numpy.linspace(0.,1.,1001)
    o.integrate(ts,lp)
    #Create orbit points from this integrated orbit, each 100th point
    vxvv= []
    for ii in range(10):
        vxvv.append([o.ra(ii/10.),o.dec(ii/10.),
                     o.dist(ii/10.),o.pmra(ii/10.),
                     o.pmdec(ii/10.),o.vlos(ii/10.)])
    vxvv= numpy.array(vxvv)
    #now fit
    of= Orbit.from_fit([o.ra(),o.dec(),o.dist(),o.pmra(),o.pmdec(),o.vlos()],
                       vxvv,pot=lp,tintJ=1.5,radec=True,ro=ro,vo=vo)
    compf= comp_orbfit(of,vxvv,numpy.linspace(0.,2.,1001),lp,lb=False,radec=True,
                       ro=ro,vo=vo)
    assert numpy.all(compf < 10.**-4.), 'Orbit fit in radec space does not work'
    return None

# Test orbit fit in custom coordinates (using Equatorial for testing)
def test_orbitfit_custom():
    from galpy.orbit import Orbit
    from galpy.util import coords
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
    ro, vo= 9., 230.
    o= Orbit([0.8,0.3,1.3,0.4,0.2,2.],ro=ro,vo=vo)
    ts= numpy.linspace(0.,1.,1001)
    o.integrate(ts,lp)
    #Create orbit points from this integrated orbit, each 100th point
    vxvv= []
    for ii in range(10):
        vxvv.append([o.ra(ii/10.),o.dec(ii/10.),
                     o.dist(ii/10.),o.pmra(ii/10.),
                     o.pmdec(ii/10.),o.vlos(ii/10.)])
    vxvv= numpy.array(vxvv)
    #now fit
    #First test the exception
    try:
        Orbit.from_fit([o.ra(),o.dec(),o.dist(),o.pmra(),o.pmdec(),o.vlos()],
                       vxvv,pot=lp,tintJ=1.5,customsky=True,
                       ro=ro,vo=vo)
    except IOError: pass
    else: raise AssertionError('Orbit fit with custom sky coordinates but without the necessary coordinate-transformation functions did not raise an exception')
    of= Orbit.from_fit([o.ra(),o.dec(),o.dist(),o.pmra(),o.pmdec(),o.vlos()],
                       vxvv,pot=lp,tintJ=1.5,customsky=True,
                       lb_to_customsky=coords.lb_to_radec,
                       pmllpmbb_to_customsky=coords.pmllpmbb_to_pmrapmdec,
                       ro=ro,vo=vo)
    compf= comp_orbfit(of,vxvv,numpy.linspace(0.,2.,1001),lp,lb=False,radec=True,
                       ro=ro,vo=vo)
    assert numpy.all(compf < 10.**-4.), 'Orbit fit in radec space does not work'
    return None

def comp_orbfit(of,vxvv,ts,pot,lb=False,radec=False,ro=None,vo=None):
    """Compare the output of the orbit fit properly, ro and vo only implemented for radec"""
    from galpy.util import coords
    coords._APY_COORDS_ORIG= coords._APY_COORDS
    coords._APY_COORDS= False # too slow otherwise
    of.integrate(ts,pot)
    off= of.flip()
    off.integrate(ts,pot)
    #Flip velocities again
    off.vxvv[...,1]*= -1.
    off.vxvv[...,2]*= -1.
    off.vxvv[...,4]*= -1.
    if lb:
        allvxvv= []
        for ii in range(len(ts)):
            allvxvv.append([of.ll(ts[ii]),of.bb(ts[ii]),
                            of.dist(ts[ii]),of.pmll(ts[ii]),
                            of.pmbb(ts[ii]),of.vlos(ts[ii])])
            allvxvv.append([off.ll(ts[ii]),off.bb(ts[ii]),
                            off.dist(ts[ii]),off.pmll(ts[ii]),
                            off.pmbb(ts[ii]),off.vlos(ts[ii])])
        allvxvv= numpy.array(allvxvv)
    elif radec:
        allvxvv= []
        for ii in range(len(ts)):
            allvxvv.append([of.ra(ts[ii],ro=ro,vo=vo),of.dec(ts[ii],ro=ro,vo=vo),
                            of.dist(ts[ii],ro=ro,vo=vo),of.pmra(ts[ii],ro=ro,vo=vo),
                            of.pmdec(ts[ii],ro=ro,vo=vo),of.vlos(ts[ii],ro=ro,vo=vo)])
            allvxvv.append([off.ra(ts[ii]),off.dec(ts[ii],ro=ro,vo=vo),
                            off.dist(ts[ii],ro=ro,vo=vo),off.pmra(ts[ii],ro=ro,vo=vo),
                            off.pmdec(ts[ii],ro=ro,vo=vo),off.vlos(ts[ii],ro=ro,vo=vo)])
        allvxvv= numpy.array(allvxvv)
    else:
        allvxvv= numpy.concatenate((of.getOrbit(),off.getOrbit()),axis=0)
    out= []
    for ii in range(vxvv.shape[0]):
        out.append(numpy.amin(numpy.sum((allvxvv-vxvv[ii])**2.,axis=1)))
    coords._APY_COORDS= coords._APY_COORDS_ORIG
    return numpy.array(out)

def test_MWPotential_warning():
    # Test that using MWPotential throws a warning, see #229
    ts= numpy.linspace(0.,100.,1001)
    o= setup_orbit_energy(potential.MWPotential,axi=False)
    with pytest.warns(None) as record:
        if PY2: reset_warning_registry('galpy')
        warnings.simplefilter("always",galpyWarning)
        o.integrate(ts,potential.MWPotential)
        # Should raise warning bc of MWPotential, might raise others
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "Use of MWPotential as a Milky-Way-like potential is deprecated; galpy.potential.MWPotential2014, a potential fit to a large variety of dynamical constraints (see Bovy 2015), is the preferred Milky-Way-like potential in galpy")
    assert raisedWarning, "Orbit integration with MWPotential should have thrown a warning, but didn't"
    return None

# Test the new Orbit.time function
def test_time():
    # Setup orbit
    o= setup_orbit_energy(potential.MWPotential,axi=False)
    # Prior to integration, should return zero
    assert numpy.fabs(o.time()-0.) < 10.**-10., "Orbit.time before integration does not return zero"
    # Then integrate
    times= numpy.linspace(0.,10.,1001)
    o.integrate(times,potential.MWPotential)
    assert numpy.all((o.time()-times) < 10.**-8.), "Orbit.time after integration does not return the integration times"
    return None

# Test interpolation with backwards orbit integration
def test_backinterpolation_issue204():
    # Setup orbit and its flipped version
    o= setup_orbit_energy(potential.MWPotential,axi=False)
    of= o.flip()
    # Times to integrate backward and forward of flipped (should agree)
    ntimes= numpy.linspace(0.,-10.,1001)
    ptimes= -ntimes
    # Integrate the orbits
    o.integrate(ntimes,potential.MWPotential)
    of.integrate(ptimes,potential.MWPotential)
    # Test that interpolation works and gives the same result
    nitimes= numpy.linspace(0.,-10.,2501)
    pitimes= -nitimes
    assert numpy.all((o.R(nitimes)-of.R(pitimes)) < 10.**-8.), 'Forward and backward integration with interpolation do not agree'
    assert numpy.all((o.z(nitimes)-of.z(pitimes)) < 10.**-8.), 'Forward and backward integration with interpolation do not agree'
    # Velocities should be flipped
    assert numpy.all((o.vR(nitimes)+of.vR(pitimes)) < 10.**-8.), 'Forward and backward integration with interpolation do not agree'
    assert numpy.all((o.vT(nitimes)+of.vT(pitimes)) < 10.**-8.), 'Forward and backward integration with interpolation do not agree'
    assert numpy.all((o.vT(nitimes)+of.vT(pitimes)) < 10.**-8.), 'Forward and backward integration with interpolation do not agree'
    return None

# Test that Orbit.x .y .vx and .vy return a scalar for scalar time input
def test_scalarxyvzvz_issue247():
    # Setup an orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o= setup_orbit_energy(lp,axi=False)
    assert isinstance(o.x(),float), 'Orbit.x() does not return a scalar'
    assert isinstance(o.y(),float), 'Orbit.y() does not return a scalar'
    assert isinstance(o.vx(),float), 'Orbit.vx() does not return a scalar'
    assert isinstance(o.vy(),float), 'Orbit.vy() does not return a scalar'
    # Also integrate and then test
    times= numpy.linspace(0.,10.,1001)
    o.integrate(times,lp)
    assert isinstance(o.x(5.),float), 'Orbit.x() does not return a scalar'
    assert isinstance(o.y(5.),float), 'Orbit.y() does not return a scalar'
    assert isinstance(o.vx(5.),float), 'Orbit.vx() does not return a scalar'
    assert isinstance(o.vy(5.),float), 'Orbit.vy() does not return a scalar'
    return None

# Test that all Orbit methods return a scalar for scalar time input (mentioned
# in #294)
def test_scalar_all():
    # Setup an orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    o= setup_orbit_energy(lp,axi=False)
    assert isinstance(o.R(),float), 'Orbit.R() does not return a scalar'
    assert isinstance(o.vR(),float), 'Orbit.vR() does not return a scalar'
    assert isinstance(o.vT(),float), 'Orbit.vT() does not return a scalar'
    assert isinstance(o.z(),float), 'Orbit.z() does not return a scalar'
    assert isinstance(o.vz(),float), 'Orbit.vz() does not return a scalar'
    assert isinstance(o.phi(),float), 'Orbit.phi() does not return a scalar'
    assert isinstance(o.r(),float), 'Orbit.r() does not return a scalar'
    assert isinstance(o.x(),float), 'Orbit.x() does not return a scalar'
    assert isinstance(o.y(),float), 'Orbit.y() does not return a scalar'
    assert isinstance(o.vx(),float), 'Orbit.vx() does not return a scalar'
    assert isinstance(o.vy(),float), 'Orbit.vy() does not return a scalar'
    assert isinstance(o.theta(),float), 'Orbit.theta() does not return a scalar'
    assert isinstance(o.vtheta(),float), 'Orbit.vtheta() does not return a scalar'
    assert isinstance(o.vr(),float), 'Orbit.vr() does not return a scalar'
    assert isinstance(o.ra(),float), 'Orbit.ra() does not return a scalar'
    assert isinstance(o.dec(),float), 'Orbit.dec() does not return a scalar'
    assert isinstance(o.ll(),float), 'Orbit.ll() does not return a scalar'
    assert isinstance(o.bb(),float), 'Orbit.bb() does not return a scalar'
    assert isinstance(o.dist(),float), 'Orbit.dist() does not return a scalar'
    assert isinstance(o.pmra(),float), 'Orbit.pmra() does not return a scalar'
    assert isinstance(o.pmdec(),float), 'Orbit.pmdec() does not return a scalar'
    assert isinstance(o.pmll(),float), 'Orbit.pmll() does not return a scalar'
    assert isinstance(o.pmbb(),float), 'Orbit.pmbb() does not return a scalar'
    assert isinstance(o.vra(),float), 'Orbit.vra() does not return a scalar'
    assert isinstance(o.vdec(),float), 'Orbit.vdec() does not return a scalar'
    assert isinstance(o.vll(),float), 'Orbit.vll() does not return a scalar'
    assert isinstance(o.vbb(),float), 'Orbit.vbb() does not return a scalar'
    assert isinstance(o.vlos(),float), 'Orbit.vlos() does not return a scalar'
    assert isinstance(o.helioX(),float), 'Orbit.helioX() does not return a scalar'
    assert isinstance(o.helioY(),float), 'Orbit.helioY() does not return a scalar'
    assert isinstance(o.helioZ(),float), 'Orbit.helioZ() does not return a scalar'
    assert isinstance(o.U(),float), 'Orbit.U() does not return a scalar'
    assert isinstance(o.V(),float), 'Orbit.V() does not return a scalar'
    assert isinstance(o.W(),float), 'Orbit.W() does not return a scalar'
    assert isinstance(o.E(pot=lp),float), 'Orbit.E() does not return a scalar'
    assert isinstance(o.Jacobi(pot=lp),float), 'Orbit.Jacobi() does not return a scalar'
    assert isinstance(o.ER(pot=lp),float), 'Orbit.ER() does not return a scalar'
    assert isinstance(o.Ez(pot=lp),float), 'Orbit.Ez() does not return a scalar'
    # Also integrate and then test
    times= numpy.linspace(0.,10.,1001)
    o.integrate(times,lp)
    assert isinstance(o.R(5.),float), 'Orbit.R() does not return a scalar'
    assert isinstance(o.vR(5.),float), 'Orbit.vR() does not return a scalar'
    assert isinstance(o.vT(5.),float), 'Orbit.vT() does not return a scalar'
    assert isinstance(o.z(5.),float), 'Orbit.z() does not return a scalar'
    assert isinstance(o.vz(5.),float), 'Orbit.vz() does not return a scalar'
    assert isinstance(o.phi(5.),float), 'Orbit.phi() does not return a scalar'
    assert isinstance(o.r(5.),float), 'Orbit.r() does not return a scalar'
    assert isinstance(o.x(5.),float), 'Orbit.x() does not return a scalar'
    assert isinstance(o.y(5.),float), 'Orbit.y() does not return a scalar'
    assert isinstance(o.vx(5.),float), 'Orbit.vx() does not return a scalar'
    assert isinstance(o.vy(5.),float), 'Orbit.vy() does not return a scalar'
    assert isinstance(o.theta(5.),float), 'Orbit.theta() does not return a scalar'
    assert isinstance(o.vtheta(5.),float), 'Orbit.vtheta() does not return a scalar'
    assert isinstance(o.vr(5.),float), 'Orbit.vr() does not return a scalar'
    assert isinstance(o.ra(5.),float), 'Orbit.ra() does not return a scalar'
    assert isinstance(o.dec(5.),float), 'Orbit.dec() does not return a scalar'
    assert isinstance(o.ll(5.),float), 'Orbit.ll() does not return a scalar'
    assert isinstance(o.bb(5.),float), 'Orbit.bb() does not return a scalar'
    assert isinstance(o.dist(5.),float), 'Orbit.dist() does not return a scalar'
    assert isinstance(o.pmra(5.),float), 'Orbit.pmra() does not return a scalar'
    assert isinstance(o.pmdec(5.),float), 'Orbit.pmdec() does not return a scalar'
    assert isinstance(o.pmll(5.),float), 'Orbit.pmll() does not return a scalar'
    assert isinstance(o.pmbb(5.),float), 'Orbit.pmbb() does not return a scalar'
    assert isinstance(o.vra(5.),float), 'Orbit.vra() does not return a scalar'
    assert isinstance(o.vdec(5.),float), 'Orbit.vdec() does not return a scalar'
    assert isinstance(o.vll(5.),float), 'Orbit.vll() does not return a scalar'
    assert isinstance(o.vbb(5.),float), 'Orbit.vbb() does not return a scalar'
    assert isinstance(o.vlos(5.),float), 'Orbit.vlos() does not return a scalar'
    assert isinstance(o.helioX(5.),float), 'Orbit.helioX() does not return a scalar'
    assert isinstance(o.helioY(5.),float), 'Orbit.helioY() does not return a scalar'
    assert isinstance(o.helioZ(5.),float), 'Orbit.helioZ() does not return a scalar'
    assert isinstance(o.U(5.),float), 'Orbit.U() does not return a scalar'
    assert isinstance(o.V(5.),float), 'Orbit.V() does not return a scalar'
    assert isinstance(o.W(5.),float), 'Orbit.W() does not return a scalar'
    assert isinstance(o.E(5.),float), 'Orbit.E() does not return a scalar'
    assert isinstance(o.Jacobi(5.),float), 'Orbit.Jacobi() does not return a scalar'
    assert isinstance(o.ER(5.),float), 'Orbit.ER() does not return a scalar'
    assert isinstance(o.Ez(5.),float), 'Orbit.Ez() does not return a scalar'
    return None

def test_call_issue256():
    # Reported by Semyeong Oh: non-integrated orbit with t=/=0 should return eror
    from galpy.orbit import Orbit
    o = Orbit(vxvv=[5.,-1.,0.8, 3, -0.1, 0])
    # no integration of the orbit
    with pytest.raises(ValueError) as excinfo:
        o.R(30)
    return None

# Test whether the output from the SkyCoord function is correct
def test_SkyCoord():
    from galpy.orbit import Orbit
    from astropy import units
    # In ra, dec
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    assert numpy.fabs(o.SkyCoord().ra.degree-o.ra()) < 10.**-13., 'Orbit SkyCoord ra and direct ra do not agree'
    assert numpy.fabs(o.SkyCoord().dec.degree-o.dec()) < 10.**-13., 'Orbit SkyCoord dec and direct dec do not agree'
    assert numpy.fabs(o.SkyCoord().distance.kpc-o.dist()) < 10.**-13., 'Orbit SkyCoord distance and direct distance do not agree'
    # For a list
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True)
    times= numpy.linspace(0.,2.,51)
    from galpy.potential import MWPotential
    o.integrate(times,MWPotential)
    ras= numpy.array([s.ra.degree for s in o.SkyCoord(times)])
    decs= numpy.array([s.dec.degree for s in o.SkyCoord(times)])
    dists= numpy.array([s.distance.kpc for s in o.SkyCoord(times)])
    assert numpy.all(numpy.fabs(ras-o.ra(times)) < 10.**-13.), 'Orbit SkyCoord ra and direct ra do not agree'
    assert numpy.all(numpy.fabs(decs-o.dec(times)) < 10.**-13.), 'Orbit SkyCoord dec and direct dec do not agree'
    assert numpy.all(numpy.fabs(dists-o.dist(times)) < 10.**-13.), 'Orbit SkyCoord distance and direct distance do not agree'
    # Check that the GC frame parameters are correctly propagated
    if not _APY3: return None # not done in python 2
    o= Orbit([120.,60.,2.,0.5,0.4,30.],radec=True,ro=10.,zo=1.,
             solarmotion=[-10.,34.,12.])
    assert numpy.fabs(o.SkyCoord().galcen_distance.to(units.kpc).value-numpy.sqrt(10.**2.+1.**2.)) < 10.**-13., 'Orbit SkyCoord GC frame attributes are incorrect'
    assert numpy.fabs(o.SkyCoord().z_sun.to(units.kpc).value-1.) < 10.**-13., 'Orbit SkyCoord GC frame attributes are incorrect'
    assert numpy.all(numpy.fabs(o.SkyCoord().galcen_v_sun.d_xyz.to(units.km/units.s).value-numpy.array([10.,220.+34.,12.])) < 10.**-13.), 'Orbit SkyCoord GC frame attributes are incorrect'
    return None

def test_orbit_obs_list_issue322():
    # Further tests of obs= list parameter for orbit output, mainly in relation
    # to issue #322
    from galpy.orbit import Orbit
    # The basic case, for a planar orbit
    o= Orbit([0.9,0.1,1.2,0.])
    assert numpy.fabs(o.helioX(obs=[1.,0.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[1.,0.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=[0.,1.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[0.,1.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,3.*numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=[0.,-1.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[0.,-1.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Full orbit
    # The basic case, for a full orbit
    o= Orbit([0.9,0.1,1.2,0.,0.,0.])
    assert numpy.fabs(o.helioX(obs=[1.,0.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[1.,0.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=[0.,1.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[0.,1.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,3.*numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=[0.,-1.,0.],ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=[0.,-1.,0.],ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    return None

def test_orbit_obs_Orbit_issue322():
    #Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322
    from galpy.orbit import Orbit
    # The basic case, for a planar orbit
    o= Orbit([0.9,0.1,1.2,0.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,0.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,0.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,numpy.pi/2.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,numpy.pi/2.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,3.*numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,3.*numpy.pi/2.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,3.*numpy.pi/2.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Full orbit
    # The basic case, for a full orbit
    o= Orbit([0.9,0.1,1.2,0.,0.,0.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,0.,0.,0.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,0.,0.,0.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,0.,0.,numpy.pi/2.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,0.,0.,numpy.pi/2.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,3.*numpy.pi/2.])
    assert numpy.fabs(o.helioX(obs=Orbit([1.,0.,0.,0.,0.,3.*numpy.pi/2.]),ro=1.)-0.1) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=Orbit([1.,0.,0.,0.,0.,3.*numpy.pi/2.]),ro=1.)) < 10.**-7., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    return None

def test_orbit_obs_Orbits_issue322():
    #Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322; specific case where the orbit gets evaluated at multiple
    # times
    from galpy.orbit import Orbit
    # Do non-zero Ysun case for planarOrbit
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.,0.,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.helioX(times,obs=obs,ro=1.)[ii]-o.helioX(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),0.],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.helioY(times,obs=obs,ro=1.)[ii]-o.helioY(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),0.],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Do non-zero Ysun case for planarOrbit, but giving FullOrbit for obs
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.,0.,0.,0.,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.helioX(times,obs=obs,ro=1.)[ii]-o.helioX(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.helioY(times,obs=obs,ro=1.)[ii]-o.helioY(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Do non-zero Ysun case for FullOrbit
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.,0.,0.,0.,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.helioX(times,obs=obs,ro=1.)[ii]-o.helioX(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.helioY(times,obs=obs,ro=1.)[ii]-o.helioY(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    return None

def test_orbit_obsvel_list_issue322():
    # Further tests of obs= list parameter for orbit output, incl. velocity
    # mainly in relation to issue #322
    from galpy.orbit import Orbit
    # The basic case, for a planar orbit
    o= Orbit([0.9,0.1,1.2,0.])
    assert numpy.fabs(o.U(obs=[1.,0.,0.,0.,0.7,0.],ro=1.,vo=1.)+0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[1.,0.,0.,0.,0.7,0.],ro=1.,vo=1.)-0.5) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.])
    assert numpy.fabs(o.U(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-0.7) < 10.**-5.7, 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-1.8) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,3.*numpy.pi/2.])
    assert numpy.fabs(o.U(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-0.9) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)+0.6) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Full orbit
    # The basic case, for a full orbit
    o= Orbit([0.9,0.1,1.2,0.,0.,0.])
    assert numpy.fabs(o.U(obs=[1.,0.,0.,0.,0.7,0.],ro=1.,vo=1.)+0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[1.,0.,0.,0.,0.7,0.],ro=1.,vo=1.)-0.5) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.])
    assert numpy.fabs(o.U(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-0.7) < 10.**-5.5, 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-1.8) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,3.*numpy.pi/2.])
    assert numpy.fabs(o.U(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)-0.9) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=[0.,1.,0.,0.6,0.8,0.],ro=1.,vo=1.)+0.6) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'

    return None

def test_orbit_obsvel_Orbit_issue322():
    # Further tests of obs= Orbit parameter for orbit output, incl. velocity
    # mainly in relation to issue #322
    from galpy.orbit import Orbit
    # The basic case, for a planar orbit
    o= Orbit([0.9,0.1,1.2,0.])
    obs= Orbit([1.,0.,0.7,0.,0.,0.],ro=1.,vo=1.)
    assert numpy.fabs(o.U(obs=obs,ro=1.,vo=1.)+0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=obs,ro=1.,vo=1.)-0.5) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.])
    obs= Orbit([1.,0.,0.7,0.,0.,numpy.pi/2.],ro=1.,vo=1.)
    assert numpy.fabs(o.helioX(obs=obs,ro=1.)-0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=obs,ro=1.)) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,3.*numpy.pi/2.])
    obs= Orbit([1.,0.,0.7,0.,0.,3.*numpy.pi/2.],ro=1.,vo=1.)
    assert numpy.fabs(o.helioX(obs=obs,ro=1.)-0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=obs,ro=1.)) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Full orbit
    # The basic case, for a full orbit
    o= Orbit([0.9,0.1,1.2,0.,0.,0.])
    obs= Orbit([1.,0.,0.7,0.,0.,0.],ro=1.,vo=1.)
    assert numpy.fabs(o.U(obs=obs,ro=1.,vo=1.)+0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.V(obs=obs,ro=1.,vo=1.)-0.5) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.])
    obs= Orbit([1.,0.,0.7,0.,0.,numpy.pi/2.],ro=1.,vo=1.)
    assert numpy.fabs(o.helioX(obs=obs,ro=1.)-0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=obs,ro=1.)) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Now use non-zero Ysun
    o= Orbit([0.9,0.1,1.2,0.,0.,3.*numpy.pi/2.])
    obs= Orbit([1.,0.,0.7,0.,0.,3.*numpy.pi/2.],ro=1.,vo=1.)
    assert numpy.fabs(o.helioX(obs=obs,ro=1.)-0.1) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    assert numpy.fabs(o.helioY(obs=obs,ro=1.)) < 10.**-6., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    return None

def test_orbit_obsvel_Orbits_issue322():
    #Further tests of obs= Orbit parameter for orbit output, mainly in relation
    # to issue #322; specific case where the orbit gets evaluated at multiple
    # times; for velocity
    from galpy.orbit import Orbit
    # Do non-zero Ysun case for planarOrbit
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.5,1.3,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.U(times,obs=obs,ro=1.)[ii]-o.U(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),0.,obs.vx(times[ii]),obs.vy(times[ii]),0.],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.V(times,obs=obs,ro=1.)[ii]-o.V(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),0.,obs.vx(times[ii]),obs.vy(times[ii]),0.],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Do non-zero Ysun case for planarOrbit, but giving FullOrbit for obs
    o= Orbit([0.9,0.1,1.2,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.5,1.3,0.,0.,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.U(times,obs=obs,ro=1.)[ii]-o.U(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii]),obs.vx(times[ii]),obs.vy(times[ii]),obs.vz(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.V(times,obs=obs,ro=1.)[ii]-o.V(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii]),obs.vx(times[ii]),obs.vy(times[ii]),obs.vz(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    # Do non-zero Ysun case for FullOrbit
    o= Orbit([0.9,0.1,1.2,0.,0.,numpy.pi/2.],ro=1.)
    obs= Orbit([1.,0.5,1.3,0.,0.,numpy.pi/2.],ro=1.)
    times= numpy.linspace(0.,2.,2)
    from galpy.potential import MWPotential2014
    o.integrate(times,MWPotential2014)
    obs.integrate(times,MWPotential2014)
    for ii in range(len(times)):
        # Test against individual
        assert numpy.fabs(o.U(times,obs=obs,ro=1.)[ii]-o.U(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii]),obs.vx(times[ii]),obs.vy(times[ii]),obs.vz(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
        assert numpy.fabs(o.V(times,obs=obs,ro=1.)[ii]-o.V(times[ii],obs=[obs.x(times[ii]),obs.y(times[ii]),obs.z(times[ii]),obs.vx(times[ii]),obs.vy(times[ii]),obs.vz(times[ii])],ro=1.)) < 10.**-10., 'Relative position wrt the Sun from using obs= keyword does not work as expected'
    return None

def test_orbit_dim_2dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit (using ~Plevne's example)
    from galpy.util import conversion
    from galpy.orbit import Orbit
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    ell_p= potential.EllipticalDiskPotential()
    pota=[b_p,ell_p]
    o= Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],radec=True,ro=8.0,vo=220.0)
    ts= numpy.linspace(0.,3.5/conversion.time_in_Gyr(vo=220.0,ro=8.0),
                       1000,endpoint=True)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="odeint")
    return None

def test_orbit_dim_1dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.util import conversion
    from galpy.orbit import Orbit
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    pota= potential.RZToverticalPotential(b_p,1.1)
    o= Orbit(vxvv=[20.,10.,2.,3.2,3.4,-100.],radec=True,ro=8.0,vo=220.0)
    ts= numpy.linspace(0.,3.5/conversion.time_in_Gyr(vo=220.0,ro=8.0),
                       1000,endpoint=True)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="odeint")
    return None

def test_orbit_dim_1dPot_2dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit
    b_p= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    pota= [b_p.toVertical(1.1)]
    o= Orbit(vxvv=[1.1,0.1,1.1,0.1])
    ts= numpy.linspace(0.,10.,1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="dop853")
    return None

def test_orbit_dim_3dPot_1dOrb():
    # Test that orbit integration throws an error when using a 3D potential
    # for a 1D orbit
    from galpy.orbit import Orbit
    pota= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05)
    o= Orbit(vxvv=[1.1,0.1])
    ts= numpy.linspace(0.,10.,1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="dop853")
    return None

def test_orbit_dim_2dPot_1dOrb():
    # Test that orbit integration throws an error when using a 2D potential
    # for a 1D orbit
    from galpy.orbit import Orbit
    pota= potential.PowerSphericalPotentialwCutoff(\
        alpha=1.8,rc=1.9/8.,normalize=0.05).toPlanar()
    o= Orbit(vxvv=[1.1,0.1])
    ts= numpy.linspace(0.,10.,1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts,pota,method="dop853")
    return None

# Test whether ro warning is sounded when calling ra etc.
def test_orbit_radecetc_roWarning():
    from galpy.orbit import Orbit
    o= Orbit([1.1,0.1,1.1,0.1,0.1,0.2])
    check_radecetc_roWarning(o,'ra')
    check_radecetc_roWarning(o,'dec')
    check_radecetc_roWarning(o,'ll')
    check_radecetc_roWarning(o,'bb')
    check_radecetc_roWarning(o,'dist')
    check_radecetc_roWarning(o,'pmra')
    check_radecetc_roWarning(o,'pmdec')
    check_radecetc_roWarning(o,'pmll')
    check_radecetc_roWarning(o,'pmbb')
    check_radecetc_roWarning(o,'vra')
    check_radecetc_roWarning(o,'vdec')
    check_radecetc_roWarning(o,'vll')
    check_radecetc_roWarning(o,'vbb')
    check_radecetc_roWarning(o,'helioX')
    check_radecetc_roWarning(o,'helioY')
    check_radecetc_roWarning(o,'helioZ')
    check_radecetc_roWarning(o,'U')
    check_radecetc_roWarning(o,'V')
    check_radecetc_roWarning(o,'W')
    return None

# Test whether vo warning is sounded when calling pmra etc.
def test_orbit_radecetc_voWarning():
    from galpy.orbit import Orbit
    o= Orbit([1.1,0.1,1.1,0.1,0.1,0.2])
    check_radecetc_voWarning(o,'pmra')
    check_radecetc_voWarning(o,'pmdec')
    check_radecetc_voWarning(o,'pmll')
    check_radecetc_voWarning(o,'pmbb')
    check_radecetc_voWarning(o,'vra')
    check_radecetc_voWarning(o,'vdec')
    check_radecetc_voWarning(o,'vll')
    check_radecetc_voWarning(o,'vbb')
    check_radecetc_voWarning(o,'U')
    check_radecetc_voWarning(o,'V')
    check_radecetc_voWarning(o,'W')
    return None

# Test whether orbit evaluation methods sound warning when called with
# unitless time when orbit is integrated with unitfull times
def test_orbit_method_integrate_t_asQuantity_warning():
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    from astropy import units
    # Setup and integrate orbit
    ts= numpy.linspace(0.,10.,1001)*units.Gyr
    o= Orbit([1.1,0.1,1.1,0.1,0.1,0.2])
    o.integrate(ts,MWPotential2014)
    # Now check
    check_integrate_t_asQuantity_warning(o,'R')
    check_integrate_t_asQuantity_warning(o,'vR')
    check_integrate_t_asQuantity_warning(o,'vT')
    check_integrate_t_asQuantity_warning(o,'z')
    check_integrate_t_asQuantity_warning(o,'vz')
    check_integrate_t_asQuantity_warning(o,'phi')
    check_integrate_t_asQuantity_warning(o,'r')
    check_integrate_t_asQuantity_warning(o,'x')
    check_integrate_t_asQuantity_warning(o,'y')
    check_integrate_t_asQuantity_warning(o,'vx')
    check_integrate_t_asQuantity_warning(o,'vy')
    check_integrate_t_asQuantity_warning(o,'theta')
    check_integrate_t_asQuantity_warning(o,'vtheta')
    check_integrate_t_asQuantity_warning(o,'vr')
    check_integrate_t_asQuantity_warning(o,'ra')
    check_integrate_t_asQuantity_warning(o,'dec')
    check_integrate_t_asQuantity_warning(o,'ll')
    check_integrate_t_asQuantity_warning(o,'bb')
    check_integrate_t_asQuantity_warning(o,'dist')
    check_integrate_t_asQuantity_warning(o,'pmra')
    check_integrate_t_asQuantity_warning(o,'pmdec')
    check_integrate_t_asQuantity_warning(o,'pmll')
    check_integrate_t_asQuantity_warning(o,'pmbb')
    check_integrate_t_asQuantity_warning(o,'vra')
    check_integrate_t_asQuantity_warning(o,'vdec')
    check_integrate_t_asQuantity_warning(o,'vll')
    check_integrate_t_asQuantity_warning(o,'vbb')
    check_integrate_t_asQuantity_warning(o,'vlos')
    check_integrate_t_asQuantity_warning(o,'helioX')
    check_integrate_t_asQuantity_warning(o,'helioY')
    check_integrate_t_asQuantity_warning(o,'helioZ')
    check_integrate_t_asQuantity_warning(o,'U')
    check_integrate_t_asQuantity_warning(o,'V')
    check_integrate_t_asQuantity_warning(o,'W')
    check_integrate_t_asQuantity_warning(o,'E')
    check_integrate_t_asQuantity_warning(o,'L')
    check_integrate_t_asQuantity_warning(o,'Jacobi')
    check_integrate_t_asQuantity_warning(o,'ER')
    check_integrate_t_asQuantity_warning(o,'Ez')
    return None

# Test whether ro in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputro_quantity():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from astropy import units
    o= Orbit([1.1,0.1,1.1,0.2,0.3,0.3])
    ro= 11.
    assert numpy.fabs(o.E(pot=MWPotential2014,ro=ro*units.kpc)-o.E(pot=MWPotential2014,ro=ro)) < 10.**-8., 'Orbit method E does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.ER(pot=MWPotential2014,ro=ro*units.kpc)-o.ER(pot=MWPotential2014,ro=ro)) < 10.**-8., 'Orbit method ER does not return the correct value as Quantity'
    assert numpy.fabs(o.Ez(pot=MWPotential2014,ro=ro*units.kpc)-o.Ez(pot=MWPotential2014,ro=ro)) < 10.**-8., 'Orbit method Ez does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Jacobi(pot=MWPotential2014,ro=ro*units.kpc)-o.Jacobi(pot=MWPotential2014,ro=ro)) < 10.**-8., 'Orbit method Jacobi does not return the correct value when input ro is Quantity'
    assert numpy.all(numpy.fabs(o.L(pot=MWPotential2014,ro=ro*units.kpc)-o.L(pot=MWPotential2014,ro=ro)) < 10.**-8.), 'Orbit method L does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.rap(pot=MWPotential2014,analytic=True,ro=ro*units.kpc)-o.rap(pot=MWPotential2014,analytic=True,ro=ro)) < 10.**-8., 'Orbit method rap does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.rperi(pot=MWPotential2014,analytic=True,ro=ro*units.kpc)-o.rperi(pot=MWPotential2014,analytic=True,ro=ro)) < 10.**-8., 'Orbit method rperi does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.zmax(pot=MWPotential2014,analytic=True,ro=ro*units.kpc)-o.zmax(pot=MWPotential2014,analytic=True,ro=ro)) < 10.**-8., 'Orbit method zmax does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.jr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.jr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method jr does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.jp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.jp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method jp does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.jz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.jz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method jz does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.wr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.wr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method wr does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.wp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.wp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method wp does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.wz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.wz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method wz does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method Tr does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method Tp does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method Tz does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Or(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Or(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Orbit method Or does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Op(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Op(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Opbit method Or does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro*units.kpc)-o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5,ro=ro)) < 10.**-8., 'Ozbit method Or does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.time(ro=ro*units.kpc)-o.time(ro=ro)) < 10.**-8., 'Orbit method time does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.R(ro=ro*units.kpc)-o.R(ro=ro)) < 10.**-8., 'Orbit method R does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vR(ro=ro*units.kpc)-o.vR(ro=ro)) < 10.**-8., 'Orbit method vR does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vT(ro=ro*units.kpc)-o.vT(ro=ro)) < 10.**-8., 'Orbit method vT does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.z(ro=ro*units.kpc)-o.z(ro=ro)) < 10.**-8., 'Orbit method z does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vz(ro=ro*units.kpc)-o.vz(ro=ro)) < 10.**-8., 'Orbit method vz does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.phi(ro=ro*units.kpc)-o.phi(ro=ro)) < 10.**-8., 'Orbit method phi does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vphi(ro=ro*units.kpc)-o.vphi(ro=ro)) < 10.**-8., 'Orbit method vphi does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.x(ro=ro*units.kpc)-o.x(ro=ro)) < 10.**-8., 'Orbit method x does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.y(ro=ro*units.kpc)-o.y(ro=ro)) < 10.**-8., 'Orbit method y does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vx(ro=ro*units.kpc)-o.vx(ro=ro)) < 10.**-8., 'Orbit method vx does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vy(ro=ro*units.kpc)-o.vy(ro=ro)) < 10.**-8., 'Orbit method vy does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.theta(ro=ro*units.kpc)-o.theta(ro=ro)) < 10.**-8., 'Orbit method theta does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vtheta(ro=ro*units.kpc)-o.vtheta(ro=ro)) < 10.**-8., 'Orbit method vtheta does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vr(ro=ro*units.kpc)-o.vr(ro=ro)) < 10.**-8., 'Orbit method vr does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.ra(ro=ro*units.kpc)-o.ra(ro=ro)) < 10.**-8., 'Orbit method ra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.dec(ro=ro*units.kpc)-o.dec(ro=ro)) < 10.**-8., 'Orbit method dec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.ll(ro=ro*units.kpc)-o.ll(ro=ro)) < 10.**-8., 'Orbit method ll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.bb(ro=ro*units.kpc)-o.bb(ro=ro)) < 10.**-8., 'Orbit method bb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.dist(ro=ro*units.kpc)-o.dist(ro=ro)) < 10.**-8., 'Orbit method dist does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmra(ro=ro*units.kpc)-o.pmra(ro=ro)) < 10.**-8., 'Orbit method pmra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmdec(ro=ro*units.kpc)-o.pmdec(ro=ro)) < 10.**-8., 'Orbit method pmdec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmll(ro=ro*units.kpc)-o.pmll(ro=ro)) < 10.**-8., 'Orbit method pmll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmbb(ro=ro*units.kpc)-o.pmbb(ro=ro)) < 10.**-8., 'Orbit method pmbb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vlos(ro=ro*units.kpc)-o.vlos(ro=ro)) < 10.**-8., 'Orbit method vlos does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vra(ro=ro*units.kpc)-o.vra(ro=ro)) < 10.**-8., 'Orbit method vra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vdec(ro=ro*units.kpc)-o.vdec(ro=ro)) < 10.**-8., 'Orbit method vdec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vll(ro=ro*units.kpc)-o.vll(ro=ro)) < 10.**-8., 'Orbit method vll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vbb(ro=ro*units.kpc)-o.vbb(ro=ro)) < 10.**-8., 'Orbit method vbb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioX(ro=ro*units.kpc)-o.helioX(ro=ro)) < 10.**-8., 'Orbit method helioX does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioY(ro=ro*units.kpc)-o.helioY(ro=ro)) < 10.**-8., 'Orbit method helioY does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioZ(ro=ro*units.kpc)-o.helioZ(ro=ro)) < 10.**-8., 'Orbit method helioZ does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.U(ro=ro*units.kpc)-o.U(ro=ro)) < 10.**-8., 'Orbit method U does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.V(ro=ro*units.kpc)-o.V(ro=ro)) < 10.**-8., 'Orbit method V does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.W(ro=ro*units.kpc)-o.W(ro=ro)) < 10.**-8., 'Orbit method W does not return the correct value when input ro is Quantity'
    return None

# Test whether vo in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputvo_quantity():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from astropy import units
    o= Orbit([1.1,0.1,1.1,0.2,0.3,0.3])
    vo= 222.
    assert numpy.fabs(o.E(pot=MWPotential2014,vo=vo*units.km/units.s)-o.E(pot=MWPotential2014,vo=vo)) < 10.**-8., 'Orbit method E does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.ER(pot=MWPotential2014,vo=vo*units.km/units.s)-o.ER(pot=MWPotential2014,vo=vo)) < 10.**-8., 'Orbit method ER does not return the correct value as Quantity'
    assert numpy.fabs(o.Ez(pot=MWPotential2014,vo=vo*units.km/units.s)-o.Ez(pot=MWPotential2014,vo=vo)) < 10.**-8., 'Orbit method Ez does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Jacobi(pot=MWPotential2014,vo=vo*units.km/units.s)-o.Jacobi(pot=MWPotential2014,vo=vo)) < 10.**-8., 'Orbit method Jacobi does not return the correct value when input vo is Quantity'
    assert numpy.all(numpy.fabs(o.L(pot=MWPotential2014,vo=vo*units.km/units.s)-o.L(pot=MWPotential2014,vo=vo)) < 10.**-8.), 'Orbit method L does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.rap(pot=MWPotential2014,analytic=True,vo=vo*units.km/units.s)-o.rap(pot=MWPotential2014,analytic=True,vo=vo)) < 10.**-8., 'Orbit method rap does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.rperi(pot=MWPotential2014,analytic=True,vo=vo*units.km/units.s)-o.rperi(pot=MWPotential2014,analytic=True,vo=vo)) < 10.**-8., 'Orbit method rperi does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.zmax(pot=MWPotential2014,analytic=True,vo=vo*units.km/units.s)-o.zmax(pot=MWPotential2014,analytic=True,vo=vo)) < 10.**-8., 'Orbit method zmax does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.jr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.jr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method jr does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.jp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.jp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method jp does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.jz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.jz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method jz does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.wr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.wr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method wr does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.wp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.wp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method wp does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.wz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.wz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method wz does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Tr(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method Tr does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Tp(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method Tp does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Tz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method Tz does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Or(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Or(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Orbit method Or does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Op(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Op(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Opbit method Or does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo*units.km/units.s)-o.Oz(pot=MWPotential2014,type='staeckel',delta=0.5,vo=vo)) < 10.**-8., 'Ozbit method Or does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.time(vo=vo*units.km/units.s)-o.time(vo=vo)) < 10.**-8., 'Orbit method time does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.R(vo=vo*units.km/units.s)-o.R(vo=vo)) < 10.**-8., 'Orbit method R does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vR(vo=vo*units.km/units.s)-o.vR(vo=vo)) < 10.**-8., 'Orbit method vR does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vT(vo=vo*units.km/units.s)-o.vT(vo=vo)) < 10.**-8., 'Orbit method vT does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.z(vo=vo*units.km/units.s)-o.z(vo=vo)) < 10.**-8., 'Orbit method z does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vz(vo=vo*units.km/units.s)-o.vz(vo=vo)) < 10.**-8., 'Orbit method vz does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.phi(vo=vo*units.km/units.s)-o.phi(vo=vo)) < 10.**-8., 'Orbit method phi does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vphi(vo=vo*units.km/units.s)-o.vphi(vo=vo)) < 10.**-8., 'Orbit method vphi does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.x(vo=vo*units.km/units.s)-o.x(vo=vo)) < 10.**-8., 'Orbit method x does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.y(vo=vo*units.km/units.s)-o.y(vo=vo)) < 10.**-8., 'Orbit method y does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vx(vo=vo*units.km/units.s)-o.vx(vo=vo)) < 10.**-8., 'Orbit method vx does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vy(vo=vo*units.km/units.s)-o.vy(vo=vo)) < 10.**-8., 'Orbit method vy does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.theta(vo=vo*units.km/units.s)-o.theta(vo=vo)) < 10.**-8., 'Orbit method theta does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vtheta(vo=vo*units.km/units.s)-o.vtheta(vo=vo)) < 10.**-8., 'Orbit method vtheta does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vr(vo=vo*units.km/units.s)-o.vr(vo=vo)) < 10.**-8., 'Orbit method vr does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.ra(vo=vo*units.km/units.s)-o.ra(vo=vo)) < 10.**-8., 'Orbit method ra does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.dec(vo=vo*units.km/units.s)-o.dec(vo=vo)) < 10.**-8., 'Orbit method dec does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.ll(vo=vo*units.km/units.s)-o.ll(vo=vo)) < 10.**-8., 'Orbit method ll does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.bb(vo=vo*units.km/units.s)-o.bb(vo=vo)) < 10.**-8., 'Orbit method bb does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.dist(vo=vo*units.km/units.s)-o.dist(vo=vo)) < 10.**-8., 'Orbit method dist does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.pmra(vo=vo*units.km/units.s)-o.pmra(vo=vo)) < 10.**-8., 'Orbit method pmra does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.pmdec(vo=vo*units.km/units.s)-o.pmdec(vo=vo)) < 10.**-8., 'Orbit method pmdec does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.pmll(vo=vo*units.km/units.s)-o.pmll(vo=vo)) < 10.**-8., 'Orbit method pmll does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.pmbb(vo=vo*units.km/units.s)-o.pmbb(vo=vo)) < 10.**-8., 'Orbit method pmbb does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vlos(vo=vo*units.km/units.s)-o.vlos(vo=vo)) < 10.**-8., 'Orbit method vlos does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vra(vo=vo*units.km/units.s)-o.vra(vo=vo)) < 10.**-8., 'Orbit method vra does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vdec(vo=vo*units.km/units.s)-o.vdec(vo=vo)) < 10.**-8., 'Orbit method vdec does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vll(vo=vo*units.km/units.s)-o.vll(vo=vo)) < 10.**-8., 'Orbit method vll does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.vbb(vo=vo*units.km/units.s)-o.vbb(vo=vo)) < 10.**-8., 'Orbit method vbb does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.helioX(vo=vo*units.km/units.s)-o.helioX(vo=vo)) < 10.**-8., 'Orbit method helioX does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.helioY(vo=vo*units.km/units.s)-o.helioY(vo=vo)) < 10.**-8., 'Orbit method helioY does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.helioZ(vo=vo*units.km/units.s)-o.helioZ(vo=vo)) < 10.**-8., 'Orbit method helioZ does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.U(vo=vo*units.km/units.s)-o.U(vo=vo)) < 10.**-8., 'Orbit method U does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.V(vo=vo*units.km/units.s)-o.V(vo=vo)) < 10.**-8., 'Orbit method V does not return the correct value when input vo is Quantity'
    assert numpy.fabs(o.W(vo=vo*units.km/units.s)-o.W(vo=vo)) < 10.**-8., 'Orbit method W does not return the correct value when input vo is Quantity'
    return None

# Test whether obs in methods using physical_conversion can be specified
# as a Quantity
def test_orbit_method_inputobs_quantity():
    from galpy.orbit import Orbit
    from astropy import units
    o= Orbit([1.1,0.1,1.1,0.2,0.3,0.3])
    obs= [11.,0.1,0.2,-10.,245.,7.]
    obs_units= [11.*units.kpc,0.1*units.kpc,0.2*units.kpc,
                -10.*units.km/units.s,245.*units.km/units.s,7.*units.km/units.s]
    assert numpy.fabs(o.ra(obs=obs_units)-o.ra(obs=obs)) < 10.**-8., 'Orbit method ra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.dec(obs=obs_units)-o.dec(obs=obs)) < 10.**-8., 'Orbit method dec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.ll(obs=obs_units)-o.ll(obs=obs)) < 10.**-8., 'Orbit method ll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.bb(obs=obs_units)-o.bb(obs=obs)) < 10.**-8., 'Orbit method bb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.dist(obs=obs_units)-o.dist(obs=obs)) < 10.**-8., 'Orbit method dist does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmra(obs=obs_units)-o.pmra(obs=obs)) < 10.**-8., 'Orbit method pmra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmdec(obs=obs_units)-o.pmdec(obs=obs)) < 10.**-8., 'Orbit method pmdec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmll(obs=obs_units)-o.pmll(obs=obs)) < 10.**-8., 'Orbit method pmll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.pmbb(obs=obs_units)-o.pmbb(obs=obs)) < 10.**-8., 'Orbit method pmbb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vlos(obs=obs_units)-o.vlos(obs=obs)) < 10.**-8., 'Orbit method vlos does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vra(obs=obs_units)-o.vra(obs=obs)) < 10.**-8., 'Orbit method vra does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vdec(obs=obs_units)-o.vdec(obs=obs)) < 10.**-8., 'Orbit method vdec does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vll(obs=obs_units)-o.vll(obs=obs)) < 10.**-8., 'Orbit method vll does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.vbb(obs=obs_units)-o.vbb(obs=obs)) < 10.**-8., 'Orbit method vbb does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioX(obs=obs_units)-o.helioX(obs=obs)) < 10.**-8., 'Orbit method helioX does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioY(obs=obs_units)-o.helioY(obs=obs)) < 10.**-8., 'Orbit method helioY does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.helioZ(obs=obs_units)-o.helioZ(obs=obs)) < 10.**-8., 'Orbit method helioZ does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.U(obs=obs_units)-o.U(obs=obs)) < 10.**-8., 'Orbit method U does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.V(obs=obs_units)-o.V(obs=obs)) < 10.**-8., 'Orbit method V does not return the correct value when input ro is Quantity'
    assert numpy.fabs(o.W(obs=obs_units)-o.W(obs=obs)) < 10.**-8., 'Orbit method W does not return the correct value when input ro is Quantity'
    return None

# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_full():
    integrators= ['dopr54_c',
                  'leapfrog_c',
                  'dop853_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    scriptpath= 'orbitint4sigint.py'
    if not 'tests' in os.getcwd():
        scriptpath= os.path.join('tests',scriptpath)
    ntries= 10
    for integrator in integrators:
        p= subprocess.Popen(['python',scriptpath,integrator,'full'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid,signal.SIGINT)
        time.sleep(1)
        cnt= 0
        while p.poll() is None and cnt < ntries: # wait a little longer
            time.sleep(4)
            cnt+= 1

        if p.poll() == 2 and WIN32: break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None: msg= -100
            else: msg= p.poll()
            raise AssertionError("Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i" % (integrator,msg))
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None

# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_planar():
    integrators= ['dopr54_c',
                  'leapfrog_c',
                  'dop853_c',
                  'rk4_c','rk6_c',
                  'symplec4_c','symplec6_c']
    scriptpath= 'orbitint4sigint.py'
    if not 'tests' in os.getcwd():
        scriptpath= os.path.join('tests',scriptpath)
    ntries= 10
    for integrator in integrators:
        p= subprocess.Popen(['python',scriptpath,integrator,'planar'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid,signal.SIGINT)
        time.sleep(1)
        cnt= 0
        while p.poll() is None and cnt < ntries: # wait a little longer
            time.sleep(4)
            cnt+= 1

        if p.poll() == 2 and WIN32: break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None: msg= -100
            else: msg= p.poll()
            raise AssertionError("Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i" % (integrator,msg))
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None

# Test that orbit integration in C gets interrupted by SIGINT (CTRL-C)
def test_orbit_c_sigint_planardxdv():
    integrators= ['dopr54_c','rk4_c','rk6_c', 'dop853_c',]
    scriptpath= 'orbitint4sigint.py'
    if not 'tests' in os.getcwd():
        scriptpath= os.path.join('tests',scriptpath)
    ntries= 10
    for integrator in integrators:
        p= subprocess.Popen(['python',scriptpath,integrator,'planardxdv'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        for line in iter(p.stdout.readline, b''):
            if line.startswith(b"Starting long C integration ..."):
                break
        time.sleep(2)
        os.kill(p.pid,signal.SIGINT)
        time.sleep(1)
        cnt= 0
        while p.poll() is None and cnt < ntries: # wait a little longer
            time.sleep(4)
            cnt+= 1

        if p.poll() == 2 and WIN32: break

        if p.poll() is None or (p.poll() != 1 and p.poll() != -2):
            if p.poll() is None: msg= -100
            else: msg= p.poll()
            raise AssertionError("Full orbit integration using %s should have been interrupted by SIGINT (CTRL-C), but was not because p.poll() == %i" % (integrator,msg))
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
    return None

def test_orbitint_pythonfallback():
    # Check if a warning is raised when the potential has no C integrator
    from galpy.orbit import Orbit
    bp= BurkertPotentialNoC() # BurkertPotentialNoC is already imported at the top of test_orbit.py
    bp.normalize(1.)
    ts= numpy.linspace(0.,1.,101)
    for orb in [Orbit([1.,0.1,1.1,0.1,0.,1.]),Orbit([1.,0.1,1.1,0.1,0.]),
                Orbit([1.,0.1,1.1,1.]),Orbit([1.,0.1,1.1])]:
        with pytest.warns(None) as record:
            if PY2: reset_warning_registry('galpy')
            warnings.simplefilter("always",galpyWarning)
            #Test w/ dopr54_c
            orb.integrate(ts,bp, method='dopr54_c')
        raisedWarning= False
        for rec in record:
            # check that the message matches
            print(rec.message.args[0])
            raisedWarning+= (str(rec.message.args[0]) == "Cannot use C integration because some of the potentials are not implemented in C (using odeint instead)")
        assert raisedWarning, "Orbit integration did not raise fallback warning"
    return None

def test_orbitint_dissipativefallback():
    # Check if a warning is raised when one tries to integrate an orbit
    # in a dissipative force law with a symplectic integrator
    from galpy.orbit import Orbit
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    cdf= potential.ChandrasekharDynamicalFrictionForce(\
        GMs=0.01,
        dens=lp,sigmar=lambda r: 1./numpy.sqrt(2.))
    ts= numpy.linspace(0.,1.,101)
    for orb in [Orbit([1.,0.1,1.1,0.1,0.,1.])]:
        with pytest.warns(None) as record:
            orb.integrate(ts,[lp,cdf], method='leapfrog')
        raisedWarning= False
        for rec in record:
            # check that the message matches
            raisedWarning+= (str(rec.message.args[0]) == "Cannot use symplectic integration because some of the included forces are dissipative (using non-symplectic integrator odeint instead)")
        assert raisedWarning, "Orbit integration with symplectic integrator for dissipative force did not raise fallback warning"
    return None

# Test that the functions that supposedly *always* return output in physical
# units actually do so; see issue #294
def test_intrinsic_physical_output():
    from galpy.orbit import Orbit
    from galpy.util import coords
    o= Orbit([0.9,0.,1.,0.,0.2,0.],ro=8.,vo=220.,zo=0.,
             solarmotion=[-20.,30.,40.])
    # 04/2018: not quite anylonger w/ astropy def. of plane, but close
    l_true= 0.
    b_true= 0.
    ra_true, dec_true= coords.lb_to_radec(l_true,b_true,degree=True,
                                               epoch=None)
    assert numpy.fabs(o.ra()-ra_true) < 10.**-3.8, 'Orbit.ra does not return correct ra in degree'
    assert numpy.fabs(o.dec()-dec_true) < 10.**-3.8, 'Orbit.dec does not return correct dec in degree'
    assert numpy.fabs(o.ll()-l_true) < 10.**-4., 'Orbit.ll does not return correct l in degree'
    assert numpy.fabs(o.bb()-b_true) < 10.**-4., 'Orbit.bb does not return correct b in degree'
    assert numpy.fabs(o.dist()-0.8) < 10.**-8., 'Orbit.dist does not return correct dist in kpc'
    pmll_true= -30./0.8/_K
    pmbb_true= 4./0.8/_K
    pmra_true, pmdec_true= coords.pmllpmbb_to_pmrapmdec(pmll_true,
                                                             pmbb_true,
                                                             l_true,b_true,
                                                             degree=True,
                                                             epoch=None)
    assert numpy.fabs(o.pmra()-pmra_true) < 10.**-5., 'Orbit.pmra does not return correct pmra in mas/yr'
    assert numpy.fabs(o.pmdec()-pmdec_true) < 10.**-5., 'Orbit.pmdec does not return correct pmdec in mas/yr'
    assert numpy.fabs(o.pmll()-pmll_true) < 10.**-5., 'Orbit.pmll does not return correct pmll in mas/yr'
    assert numpy.fabs(o.pmbb()-pmbb_true) < 10.**-4.7, 'Orbit.pmbb does not return correct pmbb in mas/yr'
    assert numpy.fabs(o.vra()-pmra_true*0.8*_K) < 10.**-4.8, 'Orbit.vra does not return correct vra in km/s'
    assert numpy.fabs(o.vdec()-pmdec_true*0.8*_K) < 10.**-4.6, 'Orbit.vdec does not return correct vdec in km/s'
    assert numpy.fabs(o.vll()-pmll_true*0.8*_K) < 10.**-5., 'Orbit.vll does not return correct vll in km/s'
    assert numpy.fabs(o.vbb()-pmbb_true*0.8*_K) < 10.**-4., 'Orbit.vbb does not return correct vbb in km/s'
    assert numpy.fabs(o.vlos()+20.) < 10.**-8., 'Orbit.vlos does not return correct vlos in km/s'
    assert numpy.fabs(o.U()+20.) < 10.**-4., 'Orbit.U does not return correct U in km/s'
    assert numpy.fabs(o.V()-pmll_true*0.8*_K) < 10.**-4.8, 'Orbit.V does not return correct V in km/s'
    assert numpy.fabs(o.W()-pmbb_true*0.8*_K) < 10.**-4., 'Orbit.W does not return correct W in km/s'
    assert numpy.fabs(o.helioX()-0.8) < 10.**-8., 'Orbit.helioX does not return correct helioX in kpc'
    # For non-trivial helioY and helioZ tests
    o= Orbit([1./numpy.sqrt(2.),0.,1.,0.,0.2,numpy.pi/4.],
             ro=8.,vo=220.,zo=0.,
             solarmotion=[-20.,30.,40.])
    assert numpy.fabs(o.helioY()-4.) < 10.**-5., 'Orbit.helioY does not return correct helioY in kpc'
    o= Orbit([0.9,0.,1.,0.3,0.2,numpy.pi/4.],
             ro=8.,vo=220.,zo=0.,
             solarmotion=[-20.,30.,40.])
    assert numpy.fabs(o.helioZ()-0.3*8.) < 10.**-4.8, 'Orbit.helioZ does not return correct helioZ in kpc'
    return None

def test_doublewrapper_2d():
    # Test that a doubly-wrapped potential gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential
    # potential= flat vc + doubly-wrapped bar
    pot= [LogarithmicHaloPotential(normalize=1.),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_doublewrapper_3d():
    # Test that a doubly-wrapped potential gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential
    # potential= flat vc + doubly-wrapped bar
    pot= [LogarithmicHaloPotential(normalize=1.),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1,-0.03,numpy.pi])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.z()-oc.z()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vz()-oc.vz()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_wrapper_followedbyanotherpotential_2d():
    # Test that a wrapped potential that gets followed by another potential
    # gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential, \
        SpiralArmsPotential
    # potential= flat vc + doubly-wrapped bar
    pot= [LogarithmicHaloPotential(normalize=1.),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3),
          SpiralArmsPotential(N=4,omega=0.79,amp=0.9)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_wrapper_followedbyanotherpotential_3d():
    # Test that a wrapped potential that gets followed by another potential
    # gets passed to C correctly,
    # by comparing orbit integrated in C to that in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential, \
        SpiralArmsPotential
    # potential= flat vc + doubly-wrapped bar
    pot= [LogarithmicHaloPotential(normalize=1.),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3),
          SpiralArmsPotential(N=4,omega=0.79,amp=0.9)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1,-0.03,numpy.pi])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.z()-oc.z()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vz()-oc.vz()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_wrapper_complicatedsequence_2d():
    # Test that a complicated combination of potentials and wrapped potentials
    # gets passed to C correctly, by comparing orbit integrated in C to that
    # in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential, \
        SpiralArmsPotential
    # potential= flat vc + doubly-wrapped bar + spiral-arms
    pot= [LogarithmicHaloPotential(normalize=0.2),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3),
          DehnenSmoothWrapperPotential(\
            pot=SpiralArmsPotential(N=4,omega=0.79,amp=0.9),
            tform=5.,tsteady=15.),
          LogarithmicHaloPotential(normalize=0.8)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_wrapper_complicatedsequence_3d():
    # Test that a complicated combination of potentials and wrapped potentials
    # gets passed to C correctly, by comparing orbit integrated in C to that
    # in python
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential, \
        DehnenBarPotential, \
        SolidBodyRotationWrapperPotential, \
        DehnenSmoothWrapperPotential, \
        SpiralArmsPotential
    # potential= flat vc + doubly-wrapped bar + spiral-arms
    pot= [LogarithmicHaloPotential(normalize=0.2),
          SolidBodyRotationWrapperPotential(\
            pot=DehnenSmoothWrapperPotential(\
                pot=DehnenBarPotential(omegab=1.,rb=5./8.,Af=1./100.),
                tform=5.,tsteady=15.),omega=1.3),
          DehnenSmoothWrapperPotential(\
            pot=SpiralArmsPotential(N=4,omega=0.79,amp=0.9),
            tform=5.,tsteady=15.),
          LogarithmicHaloPotential(normalize=0.8)]
    # Integrate orbit in C and python
    o= Orbit([1.,0.1,1.1,0.1,-0.03,numpy.pi])
    oc= o()
    ts= numpy.linspace(0.,20.,1001)
    o.integrate(ts,pot,method='leapfrog')
    oc.integrate(ts,pot,method='leapfrog_c')
    # Check that they end up in the same point
    o= o(ts[-1])
    oc= oc(ts[-1])
    assert numpy.fabs(o.x()-oc.x()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.y()-oc.y()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.z()-oc.z()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vx()-oc.vx()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vy()-oc.vy()) < 10.**-4.,  'Final orbit velocity between C and Python integration of a doubly-wrapped orbit is too large'
    assert numpy.fabs(o.vz()-oc.vz()) < 10.**-4.,  'Final orbit position between C and Python integration of a doubly-wrapped orbit is too large'
    return None

def test_orbit_sun_setup():
    # Test that setting up an Orbit with no vxvv returns the Orbit of the Sun
    from galpy.orbit import Orbit
    o= Orbit()
    assert numpy.fabs(o.dist()) < 1e-10, 'Orbit with no vxvv does not produce an orbit with zero distance'
    assert numpy.fabs(o.vll()) < 1e-10, 'Orbit with no vxvv does not produce an orbit with zero velocity in the Galactic longitude direction'
    assert numpy.fabs(o.vbb()) < 1e-10, 'Orbit with no vxvv does not produce an orbit with zero velocity in the Galactic latitude direction'
    assert numpy.fabs(o.vlos()) < 1e-10, 'Orbit with no vxvv does not produce an orbit with zero line-of-sight velocity'

def test_integrate_dxdv_errors():
    from galpy.orbit import Orbit
    ts= numpy.linspace(0.,10.,1001)
    # Test that attempting to use integrate_dxdv with a non-phasedim==4 orbit 
    # raises error
    o= Orbit([1.,0.1])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None,ts,potential.toVertical(potential.MWPotential,1.))
    o= Orbit([1.,0.1,1.])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None,ts,potential.MWPotential)
    o= Orbit([1.,0.1,1.,0.1,0.1])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None,ts,potential.MWPotential)
    o= Orbit([1.,0.1,1.,0.1,0.1,3.])
    with pytest.raises(AttributeError) as excinfo:
        o.integrate_dxdv(None,ts,potential.MWPotential)
    # Test that a random string as the integrator doesn't work
    o= Orbit([1.,0.1,1.,3.])
    with pytest.raises(ValueError) as excinfo:
        o.integrate_dxdv(None,ts,potential.MWPotential,method='some non-existent integrator')
    return None

def test_linear_plotting():
    from galpy.orbit import Orbit
    from galpy.potential.verticalPotential import RZToverticalPotential
    o= Orbit([1.,1.])
    times= numpy.linspace(0.,7.,251)
    from galpy.potential import LogarithmicHaloPotential
    lp= RZToverticalPotential(LogarithmicHaloPotential(normalize=1.,q=0.8),1.)
    try: o.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit')
    # Integrate
    o.integrate(times,lp)
    # Energy
    o.plotE()
    o.plotE(pot=lp,d1='x',xlabel=r'$xlabel$')
    o.plotE(pot=lp,d1='vx',ylabel=r'$ylabel$')
    # Plot the orbit itself, defaults
    o.plot()
    o.plot(ro=8.)
    # Plot the orbit itself in 3D
    try: o.plot3d()
    except AttributeError: pass
    else: raise AssertionError('o.plot3d for linearOrbit did not raise Exception')
    return None

# Check plotting routines
def test_planar_plotting():
    from galpy.orbit import Orbit
    from galpy.potential.planarPotential import RZToplanarPotential
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
    if True: #not _TRAVIS:
        from galpy.potential import DoubleExponentialDiskPotential
        dp= DoubleExponentialDiskPotential(normalize=1.)
    lp= LogarithmicHaloPotential(normalize=1.,q=0.8)
    try: o.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotEz()
    except AttributeError: pass
    else: raise AssertionError('o.plotEz() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: o.plotJacobi()
    except AttributeError: pass
    else: raise AssertionError('o.plotJacobi() before the orbit was integrated did not raise AttributeError for planarOrbit')
    try: oa.plotE()
    except AttributeError: pass
    else: raise AssertionError('o.plotE() before the orbit was integrated did not raise AttributeError for planarROrbit')
    try: oa.plotEz()
    except AttributeError: pass
    else: raise AssertionError('o.plotEz() before the orbit was integrated did not raise AttributeError for planarROrbit')
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
    if True: #not _TRAVIS:
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
    if True: #not _TRAVIS:
        o.plotEz(pot=dp,d1='phi')
    oa.plotEz()
    oa.plotEz(normed=True)
    oa.plotEz(pot=lp,d1='R')
    oa.plotEz(pot=lp,d1='vR')
    oa.plotEz(pot=lp,d1='vT')
    oa.plotEz(pot=lp,d1='z')
    oa.plotEz(pot=lp,d1='vz')
    # Radial energy
    o.plotER()
    o.plotER(normed=True)
    # Radial energy
    oa.plotER()
    oa.plotER(normed=True)
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
    o.plot(d2='vR')
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
    o.plot(d2='r',d1='R')
    o.plot(d2='R',d1='r')
    # Some more energies etc.
    o.plot(d1='E',d2='R')
    o.plot(d1='Enorm',d2='R')
    o.plot(d1='Ez',d2='R')
    o.plot(d1='Eznorm',d2='R')
    o.plot(d1='ER',d2='R')
    o.plot(d1='ERnorm',d2='R')
    o.plot(d1='Jacobi',d2='R')
    o.plot(d1='Jacobinorm',d2='R')
    # callables
    o.plot(d1=lambda t: t,d2=lambda t: o.R(t))
    # Expressions
    o.plot(d1='t',d2='r*R/vR')
    o.plot(d1='R*cos(phi-{:f}*t)'\
               .format(o.Op(quantity=False)-o.Or(quantity=False)/2),
           d2='R*sin(phi-{:f}*t)'\
               .format(o.Op(quantity=False)-o.Or(quantity=False)/2))
    with pytest.raises(TypeError) as excinfo:
        # Unparseable expression gives TypeError
        o.plot(d1='t',d2='r^2')
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
    o.plot3d(d1='r',d2='t',d3='phi')
    o.plot3d(d1='vT',d2='vR',d3='t')
    o.plot3d(d1='z',d2='vT',d3='vz')
    o.plot3d(d1='vz',d2='z',d3='phi')
    o.plot3d(d1='phi',d2='vz',d3='R')
    o.plot3d(d1='vR',d2='phi',d3='vR')
    o.plot3d(d1='vx',d2='x',d3='y')
    o.plot3d(d1='y',d2='vx',d3='vy')
    o.plot3d(d1='vy',d2='y',d3='x')
    o.plot3d(d1='x',d2='vy',d3='vx')
    o.plot3d(d1='x',d2='r',d3='vx')
    o.plot3d(d1='x',d2='vy',d3='r')
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
    # callables don't work
    o.plot3d(d1=lambda t: t,d2=lambda t: o.R(t),d3=lambda t: o.z(t))
    # Test AttributeErrors
    try: o.plot3d(d1='R') #shouldn't work, bc there is no default
    except AttributeError: pass
    else: raise AssertionError('plot3d with just d1= set should have raised AttributeError, but did not')
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

def test_from_name_values():
    from galpy.orbit import Orbit

    # test Vega
    o = Orbit.from_name('Vega')
    assert numpy.isclose(o.ra(), 279.23473479), \
        "RA of Vega does not match SIMBAD value"
    assert numpy.isclose(o.dec(), 38.78368896), \
        "DEC of Vega does not match SIMBAD value"
    assert numpy.isclose(o.dist(), 1/130.23), \
        "Parallax of Vega does not match SIMBAD value"
    assert numpy.isclose(o.pmra(), 200.94), \
        "PMRA of Vega does not match SIMBAD value"
    assert numpy.isclose(o.pmdec(), 286.23), \
        "PMDec of Vega does not match SIMBAD value"
    assert numpy.isclose(o.vlos(), -20.60), \
        "radial velocity of Vega does not match SIMBAD value"

    # test Lacaille 8760
    o = Orbit.from_name('Lacaille 8760')
    assert numpy.isclose(o.ra(), 319.31362024), \
        "RA of Lacaille 8760 does not match SIMBAD value"
    assert numpy.isclose(o.dec(), -38.86736390), \
        "DEC of Lacaille 8760 does not match SIMBAD value"
    assert numpy.isclose(o.dist(), 1/251.8295), \
        "Parallax of Lacaille 8760 does not match SIMBAD value"
    assert numpy.isclose(o.pmra(), -3258.553), \
        "PMRA of Lacaille 8760 does not match SIMBAD value"
    assert numpy.isclose(o.pmdec(), -1145.396), \
        "PMDec of Lacaille 8760 does not match SIMBAD value"
    assert numpy.isclose(o.vlos(), 20.56), \
        "radial velocity of Lacaille 8760 does not match SIMBAD value"

    # test LMC
    o = Orbit.from_name('LMC')
    assert numpy.isclose(o.ra(), 78.77), \
        "RA of LMC does not match value on file"
    assert numpy.isclose(o.dec(), -69.01), \
        "DEC of LMC does not match value on file"
    # Remove distance for now, because SIMBAD has the wrong distance (100 Mpc)
    assert numpy.isclose(o.dist(), 50.1), \
       "Parallax of LMC does not match value on file"
    assert numpy.isclose(o.pmra(), 1.850), \
        "PMRA of LMC does not match value on file"
    assert numpy.isclose(o.pmdec(), 0.234), \
        "PMDec of LMC does not match value on file"
    assert numpy.isclose(o.vlos(), 262.2), \
        "radial velocity of LMC does not match value on file"

    # test a distant hypervelocity star
    o = Orbit.from_name('[BGK2006] HV 5')
    assert numpy.isclose(o.ra(), 139.498), \
        "RA of [BGK2006] HV 5 does not match value on file"
    assert numpy.isclose(o.dec(), 67.377), \
        "DEC of [BGK2006] HV 5 does not match SIMBAD value"
    assert numpy.isclose(o.dist(), 55.), \
        "Parallax of [BGK2006] HV 5 does not match SIMBAD value"
    assert numpy.isclose(o.pmra(), -0.023), \
        "PMRA of [BGK2006] HV 5 does not match SIMBAD value"
    assert numpy.isclose(o.pmdec(), -1.179), \
        "PMDec of [BGK2006] HV 5 does not match SIMBAD value"
    assert numpy.isclose(o.vlos(), 553.), \
        "radial velocity of [BGK2006] HV 5 does not match SIMBAD value"

def test_from_name_errors():
    from galpy.orbit import Orbit

    # test GJ 440
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name('GJ 440')
    msg = "failed to find some coordinates for GJ 440 in SIMBAD"
    assert str(excinfo.value) == msg, \
        "expected message '{}' but got '{}' instead".format(msg, str(excinfo.value))

    # test with a fake object
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name('abc123')
    msg = "failed to find abc123 in SIMBAD"
    assert str(excinfo.value) == msg, \
        "expected message '{}' but got '{}' instead".format(msg, str(excinfo.value))

    # test GRB 090423
    with pytest.raises(ValueError) as excinfo:
        Orbit.from_name('GRB 090423')
    msg = "failed to find some coordinates for GRB 090423 in SIMBAD"
    assert str(excinfo.value) == msg, \
        "expected message '{}' but got '{}' instead".format(msg, str(excinfo.value))

def test_from_name_named():
    # Test that the values from the JSON file are correctly transferred
    from galpy.orbit import Orbit
    import galpy.orbit
    # Read the JSON file
    import os
    import json
    named_objects_file= os.path.join(os.path.dirname(os.path.realpath(\
                galpy.orbit.__file__)),'named_objects.json')
    with open(named_objects_file,'r') as json_file:
        named_data= json.load(json_file)
    del named_data['_collections']
    del named_data['_synonyms']
    for obj in named_data:
        o= Orbit.from_name(obj)
        for attr in named_data[obj]:
            if 'source' in attr: continue
            if attr == 'ro' or attr == 'vo' or attr == 'zo' \
                    or attr == 'solarmotion':
                assert numpy.all(numpy.isclose(getattr(o,'_{:s}'.format(attr)),
                                               named_data[obj][attr]))
            elif attr == 'distance':
                assert numpy.isclose(o.dist(),named_data[obj][attr])
            else:
                assert numpy.isclose(getattr(o,'{:s}'.format(attr))(),
                                     named_data[obj][attr])
    return None    

def test_from_name_collections():
    # Test that the values from the JSON file are correctly transferred, 
    # for collections of objects
    from galpy.orbit import Orbit
    from galpy.orbit.Orbits import _known_objects_collections_original_keys
    import galpy.orbit
    # Read the JSON file
    import os
    import json
    named_objects_file= os.path.join(os.path.dirname(os.path.realpath(\
                galpy.orbit.__file__)),'named_objects.json')
    with open(named_objects_file,'r') as json_file:
        named_data= json.load(json_file)
    for obj in _known_objects_collections_original_keys:
        o= Orbit.from_name(obj)
        for ii,individual_obj in enumerate(named_data['_collections'][obj]):
            for attr in named_data[individual_obj]:
                if 'source' in attr: continue
                if attr == 'ro' or attr == 'vo' or attr == 'zo' \
                        or attr == 'solarmotion':
                    continue # don't test these here
                elif attr == 'distance':
                    assert numpy.isclose(o.dist()[ii],
                                         named_data[individual_obj][attr])
                else:
                    assert numpy.isclose(getattr(o,'{:s}'.format(attr))()[ii],
                                         named_data[individual_obj][attr])
    return None    

def test_from_name_solarsystem():
    # Test that the solar system matches Bovy et al. (2010)'s input data
    from astropy import units
    from galpy.orbit import Orbit
    correct_xyz= numpy.array(
        [[0.324190175,0.090955208,-0.022920510,-4.627851589,10.390063716,1.273504997],
         [-0.701534590,-0.168809218,0.037947785,1.725066954,-7.205747212,-0.198268558],
         [-0.982564148,-0.191145980,-0.000014724,1.126784520,-6.187988860,0.000330572],
         [1.104185888,-0.826097003,-0.044595990,3.260215854,4.524583075,0.014760239],
         [3.266443877,-3.888055863,-0.057015321,2.076140727,1.904040630,-0.054374153],
         [-9.218802228,1.788299816,0.335737817,-0.496457364,-2.005021061,0.054667082],
         [19.930781147,-2.555241579,-0.267710968,0.172224285,1.357933443,0.002836325],
         [24.323085642,-17.606227355,-0.197974999,0.664855006,0.935497207,-0.034716967]])
    os= Orbit.from_name('solar system')
    for (ii,o) in enumerate(os):
        assert numpy.fabs((o.x()*units.kpc).to(units.AU).value-correct_xyz[ii,0]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert numpy.fabs((o.y()*units.kpc).to(units.AU).value-correct_xyz[ii,1]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert numpy.fabs((o.z()*units.kpc).to(units.AU).value-correct_xyz[ii,2]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert numpy.fabs((o.vx()*units.km/units.s).to(units.AU/units.yr).value-correct_xyz[ii,3]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert numpy.fabs((o.vy()*units.km/units.s).to(units.AU/units.yr).value-correct_xyz[ii,4]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
        assert numpy.fabs((o.vz()*units.km/units.s).to(units.AU/units.yr).value-correct_xyz[ii,5]) < 1e-8, "Orbit.from_name('solar system') does not agree with Bovy et al. (2010) data"
    return None

def test_rguiding_errors():
    from galpy.potential import TriaxialNFWPotential
    from galpy.orbit import Orbit
    R,Lz= 1.,1.4
    o= Orbit([R,0.4,Lz/R,0.])
    # No potential raises error
    with pytest.raises(RuntimeError) as excinfo:
        o.rguiding()
    # non-axi potential raises error
    np= TriaxialNFWPotential(amp=20.,c=0.8,b=0.7)
    with pytest.raises(RuntimeError) as excinfo:
        o.rguiding(pot=np)
    return None

def test_phi_range():
    # Test that the range returned by Orbit.phi is [-pi,pi], 
    # example from Jeremy Webb
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    o=Orbit()
    ts= numpy.linspace(0.0,-30,5000)
    o.integrate(ts,MWPotential2014)
    assert numpy.all(o.phi(ts) <= numpy.pi), 'o.phi does not return values <= pi'
    assert numpy.all(o.phi(ts) >= -numpy.pi), 'o.phi does not return values >= pi'
    assert numpy.all(o.phi(ts[::-1]) <= numpy.pi), 'o.phi does not return values <= pi'
    assert numpy.all(o.phi(ts[::-1]) >= -numpy.pi), 'o.phi does not return values >= pi'
    # Also really interpolated
    its= numpy.linspace(0.0,-30,5001)
    assert numpy.all(o.phi(its) <= numpy.pi), 'o.phi does not return values <= pi'
    assert numpy.all(o.phi(its) >= -numpy.pi), 'o.phi does not return values >= pi'
    return None

def test_orbit_time():
    # Test that Orbit.time returns the time correctly, with units when that's
    # required
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion
    from astropy import units as u
    ts= numpy.linspace(0.,1.,1001)*u.Gyr
    o= Orbit()
    o.integrate(ts,MWPotential2014)
    # No argument, in this case should be times in Gyr
    assert numpy.all(numpy.fabs((ts-o.time(quantity=True))/ts[-1]).value < 1e-10), 'Orbit.time does not return the correct times'
    # with argument, should be time in Gyr
    assert numpy.all(numpy.fabs((ts-o.time(ts,quantity=True))/ts[-1]).value < 1e-10), 'Orbit.time does not return the correct times'
    assert numpy.fabs((ts[-1]-o.time(ts[-1],quantity=True))/ts[-1]).value < 1e-10, 'Orbit.time does not return the correct times'
    # with argument without units --> units
    assert numpy.all(numpy.fabs((ts-o.time(ts.to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro),quantity=True))).value < 1e-10), 'Orbit.time does not return the correct times'
    assert numpy.fabs((ts[-1]-o.time(ts[-1].to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro),quantity=True))).value < 1e-10, 'Orbit.time does not return the correct times'
    # Now should get without units
    o.turn_physical_off()
    assert numpy.all(numpy.fabs((ts.to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro)-o.time(ts.to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro)))) < 1e-10), 'Orbit.time does not return the correct times'
    assert numpy.fabs((ts[-1].to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro)-o.time(ts[-1].to(u.Gyr).value/conversion.time_in_Gyr(MWPotential2014[0]._vo,MWPotential2014[0]._ro)))) < 1e-10, 'Orbit.time does not return the correct times'
    return None

def test_noDeprecationWarning_timeInCall():
    # The short-cut in calling an orbit to check whether the given times are
    # exactly the same as the input times should not raise a DeprecationWarning
    # (in the first implementation as 'numpy.all(t == self.t)' it did)
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from astropy import units
    ts= numpy.linspace(0.,10.,1001)
    orb= Orbit()
    orb.integrate(ts,MWPotential)
    with pytest.warns(None) as record:
        orb.R(ts[:2])
        raisedWarning= False
        for rec in record:
            # check that the message matches
            raisedWarning+= (str(rec.message.args[0]) == "elementwise == comparison failed; this will raise an error in the future.")
        assert not raisedWarning, "Orbit evaluation with array times raises the DeprecationWarning 'elementwise == comparison failed; this will raise an error in the future.'"
    # Also when using time units and evaluating without
    ts= numpy.linspace(0.,0.1,1001)*units.Gyr
    orb.integrate(ts,MWPotential)
    with pytest.warns(None) as record:
        orb.R(ts[:2].to(units.Gyr).value)
        raisedWarning= False
        for rec in record:
            # check that the message matches
            raisedWarning+= (str(rec.message.args[0]) == "elementwise == comparison failed; this will raise an error in the future.")
        assert not raisedWarning, "Orbit evaluation with array times raises the DeprecationWarning 'elementwise == comparison failed; this will raise an error in the future.'"
    return None

# Test that issue 402 is resolved: initialization with a SkyCoord when radec=True should work fine
def test_SkyCoord_init_with_radecisTrue():
    if not _APY3: return None # not done in python 2
    from galpy.orbit import Orbit
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    # Example is for NGC5466 from @jjensen4571
    rv_ngc5466= 106.93
    pmra_ngc5466= -5.41
    pmdec_ngc5466= -0.79
    ra_ngc5466= 211.363708
    dec_ngc5466= 28.534445
    rhel_ngc5466= 16.0
    mean_ngc5466= SkyCoord(ra =ra_ngc5466*u.deg,
                           dec=dec_ngc5466*u.deg,
                           distance=rhel_ngc5466*u.kpc,
                           pm_ra_cosdec=pmra_ngc5466*u.mas/u.yr,
                           pm_dec=pmdec_ngc5466*u.mas/u.yr,
                           radial_velocity=rv_ngc5466*u.km/u.s)
    o_sky= Orbit(mean_ngc5466,radec=True,ro=8.1,vo=229.0,
                 solarmotion="schoenrich" )
    o_radec= Orbit([ra_ngc5466,dec_ngc5466,rhel_ngc5466,
                    pmra_ngc5466,pmdec_ngc5466,rv_ngc5466],
                   radec=True,ro=8.1,vo=229.0,
                   solarmotion = "schoenrich")
    assert numpy.fabs(o_sky.ra()-o_radec.ra()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    assert numpy.fabs(o_sky.dec()-o_radec.dec()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    assert numpy.fabs(o_sky.dist()-o_radec.dist()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    assert numpy.fabs(o_sky.pmra()-o_radec.pmra()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    assert numpy.fabs(o_sky.pmdec()-o_radec.pmdec()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    assert numpy.fabs(o_sky.vlos()-o_radec.vlos()) < 1e-8, 'Orbit setup with SkyCoord and radec=True does not agree with Orbit setup directly with radec'
    # Let's also test lb=True for good measure
    o_sky= Orbit(mean_ngc5466,lb=True,ro=8.1,vo=229.0,
                 solarmotion="schoenrich" )
    assert numpy.fabs(o_sky.ra()-o_radec.ra()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    assert numpy.fabs(o_sky.dec()-o_radec.dec()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    assert numpy.fabs(o_sky.dist()-o_radec.dist()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    assert numpy.fabs(o_sky.pmra()-o_radec.pmra()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    assert numpy.fabs(o_sky.pmdec()-o_radec.pmdec()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    assert numpy.fabs(o_sky.vlos()-o_radec.vlos()) < 1e-8, 'Orbit setup with SkyCoord and lb=True does not agree with Orbit setup directly with lb'
    return None

# Test related to issue #415: calling an Orbit with a single int time does not 
# work properly
# Test from @jamesmlane
def test_orbit_call_single_time_as_int():
    from galpy import potential, orbit
    pot = potential.MWPotential2014
    o= orbit.Orbit()
    times = numpy.array([0,1,2])
    o.integrate(times,pot)
    # Make sure this does not raise TypeErrpr
    try:
        o.x(times[0])
    except TypeError:
        raise
    # Test that the value makes sense
    assert numpy.fabs(o.x(times[0])-o.x()) < 1e-10
    return None

# Test related to issue #415: calling an Orbit with a single Quantity time 
# does not work properly
# Test from @jamesmlane
def test_orbit_call_single_time_as_Quantity():
    from galpy import potential, orbit
    from astropy import units as u
    pot = potential.MWPotential2014
    o= orbit.Orbit()
    times = numpy.array([0,1,2])*u.Gyr
    o.integrate(times,pot)
    # Make sure this does not raise TypeErrpr
    try:
        o.x(times[0])
    except TypeError:
        raise
    # Test that the value makes sense
    assert numpy.fabs(o.x(times[0])-o.x()) < 1e-10
    return None

# Setup the orbit for the energy test
def setup_orbit_energy(tp,axi=False,henon=False):
    # Need to treat Henon sep. here, bc cannot be scaled to be reasonable
    from galpy.orbit import Orbit
    if isinstance(tp,potential.linearPotential):
        o= Orbit([1.,1.])
    elif isinstance(tp,potential.planarPotential):
        if henon:
            if axi:
                o= Orbit([0.1,0.3,0.,])
            else:
                o= Orbit([0.1,0.3,0.,numpy.pi])
        else:
            if axi:
                o= Orbit([1.,1.1,1.1])
            else:
                o= Orbit([1.,1.1,1.1,numpy.pi/2.])
    else:
        if axi:
            o= Orbit([1.,1.1,1.1,0.1,0.1])
        else:
            o= Orbit([1.,1.1,1.1,0.1,0.1,0.])
    return o

# Setup the orbit for the Liouville test
def setup_orbit_liouville(tp,axi=False,henon=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.linearPotential):
        o= Orbit([1.,1.])
    elif isinstance(tp,potential.planarPotential):
        if henon:
            if axi:
                o= Orbit([0.1,0.3,0.,])
            else:
                o= Orbit([0.1,0.3,0.,numpy.pi])
        else:
            if axi:
                o= Orbit([1.,0.1,1.1])
            else:
                o= Orbit([1.,0.1,1.1,0.])
    else:
        if axi:
            o= Orbit([1.,0.1,1.1,0.1,0.1])
        else:
            o= Orbit([1.,0.1,1.1,0.1,0.1,0.])
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

# Setup the orbit for the energy test
def setup_orbit_flip(tp,ro,vo,zo,solarmotion,axi=False):
    from galpy.orbit import Orbit
    if isinstance(tp,potential.linearPotential):
        o= Orbit([1.,1.],ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    elif isinstance(tp,potential.planarPotential):
        if axi:
            o= Orbit([1.,1.1,1.1],ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
        else:
            o= Orbit([1.,1.1,1.1,0.],ro=ro,vo=vo,zo=zo,solarmotion=solarmotion)
    else:
        if axi:
            o= Orbit([1.,1.1,1.1,0.1,0.1],ro=ro,vo=vo,zo=zo,
                     solarmotion=solarmotion)
        else:
            o= Orbit([1.,1.1,1.1,0.1,0.1,0.],ro=ro,vo=vo,zo=zo,
                     solarmotion=solarmotion)
    return o

def check_radecetc_roWarning(o,funcName):
    # Convenience function to check whether the ro-needs-to-be-specified
    # warning is sounded
    with pytest.warns(None) as record:
        if PY2: reset_warning_registry('galpy')
        warnings.simplefilter("always",galpyWarning)
        getattr(o,funcName)()
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "Method %s(.) requires ro to be given at Orbit initialization or at method evaluation; using default ro which is %f kpc" % (funcName,8.))
    assert raisedWarning, "Orbit method %s without ro specified should have thrown a warning, but didn't" % funcName
    return None

def check_radecetc_voWarning(o,funcName):
    # Convenience function to check whether the vo-needs-to-be-specified
    # warning is sounded
    with pytest.warns(None) as record:
        if PY2: reset_warning_registry('galpy')
        warnings.simplefilter("always",galpyWarning)
        getattr(o,funcName)()
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "Method %s(.) requires vo to be given at Orbit initialization or at method evaluation; using default vo which is %f km/s" % (funcName,220.))
    assert raisedWarning, "Orbit method %s without vo specified should have thrown a warning, but didn't" % funcName
    return None

def check_integrate_t_asQuantity_warning(o,funcName):
    with pytest.warns(None) as record:
        getattr(o,funcName)(1.)
    raisedWarning= False
    for rec in record:
        # check that the message matches
        raisedWarning+= (str(rec.message.args[0]) == "You specified integration times as a Quantity, but are evaluating at times not specified as a Quantity; assuming that time given is in natural (internal) units (multiply time by unit to get output at physical time)")
    assert raisedWarning, "Orbit method %s wit unitless time after integrating with unitful time should have thrown a warning, but didn't" % funcName
    return None

def test_integrate_method_warning():
    """ Test Orbit.integrate raises an error if method is unvalid """
    from galpy.potential import MWPotential2014
    from galpy.orbit import Orbit
    o = Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0])
    t = numpy.arange(0.0, 10.0, 0.001)
    with pytest.raises(ValueError):
        o.integrate(t, MWPotential2014, method='rk4')

def test_MovingObjectPotential_orbit():
    # Test integration of an object with a MovingObjectPotential
    # Test that orbits integrated by C and Python are the same
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, HernquistPotential, MovingObjectPotential

    tmax = 5.
    times = numpy.linspace(0, tmax, 101)

    orbit_pot = HernquistPotential(amp=1E-1, a=1E-1, ro=8, vo=220)
    o = Orbit([1., 0.03, 1., 0.05, 0.03, 0.0])
    o.integrate(times, MWPotential2014)
    orbit_potential = MovingObjectPotential(o, pot=orbit_pot)

    total_potential = [MWPotential2014, orbit_potential]

    oc = Orbit([0.5, 0.5, 0.5, 0.05, 0.03, 0.0])
    op = Orbit([0.5, 0.5, 0.5, 0.05, 0.03, 0.0])
    oc.integrate(times, total_potential, method='leapfrog_c')
    op.integrate(times, total_potential, method='leapfrog')
    assert numpy.fabs(oc.x(tmax)-op.x(tmax)) < 10.**-3.,  'Final orbit position between C and Python integration in a MovingObjectPotential is too large'
    assert numpy.fabs(oc.y(tmax)-op.y(tmax)) < 10.**-3.,  'Final orbit position between C and Python integration in a MovingObjectPotential is too large'
    assert numpy.fabs(oc.z(tmax)-op.z(tmax)) < 10.**-3.,  'Final orbit position between C and Python integration in a MovingObjectPotential is too large'
    assert numpy.fabs(oc.vx(tmax)-op.vx(tmax)) < 10.**-3.,  'Final orbit velocity between C and Python integration in a MovingObjectPotential is too large'
    assert numpy.fabs(oc.vy(tmax)-op.vy(tmax)) < 10.**-3.,  'Final orbit velocity between C and Python integration in a MovingObjectPotential is too large'
    assert numpy.fabs(oc.vz(tmax)-op.vz(tmax)) < 10.**-3.,  'Final orbit velocity between C and Python integration in a MovingObjectPotential is too large'
    return None

def test_MovingObjectPotential_planar_orbit():
    # Test integration of an object with a MovingObjectPotential
    # Test that orbits integrated by C and Python are the same
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, HernquistPotential, MovingObjectPotential

    tmax = 5.
    times = numpy.linspace(0, tmax, 101)

    orbit_pot = HernquistPotential(amp=0.1, a=0.1, ro=8., vo=220.)
    o = Orbit([0.4, 0.1, 0.6, 0.])
    o.integrate(times, MWPotential2014)
    orbit_potential = MovingObjectPotential(o, pot=orbit_pot)

    total_potential = [MWPotential2014, orbit_potential]

    oc = Orbit([0.5, -0.1, 0.5, 1.])
    op = Orbit([0.5, -0.1, 0.5, 1.])
    oc.integrate(times, total_potential, method='leapfrog_c')
    op.integrate(times, total_potential, method='leapfrog')

    assert numpy.fabs(oc.x(tmax)-op.x(tmax)) < 10.**-3.,  'Final orbit position between C and Python integration in a planar MovingObjectPotential is too large'
    assert numpy.fabs(oc.y(tmax)-op.y(tmax)) < 10.**-3.,  'Final orbit position between C and Python integration in a planar MovingObjectPotential is too large'
    assert numpy.fabs(oc.vx(tmax)-op.vx(tmax)) < 10.**-3.,  'Final orbit velocity between C and Python integration in a planar MovingObjectPotential is too large'
    assert numpy.fabs(oc.vy(tmax)-op.vy(tmax)) < 10.**-3.,  'Final orbit velocity between C and Python integration in a planar MovingObjectPotential is too large'
    return None

