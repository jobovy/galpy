############################TESTS ON POTENTIALS################################
import os
import sys
import numpy
import pynbody
from galpy import potential
_TRAVIS= bool(os.getenv('TRAVIS'))

#Test whether the normalization of the potential works
def test_normalize_potential():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockTwoPowerIntegerSphericalPotential')
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerIntegerSphericalPotential')
    pots.append('JaffeTwoPowerIntegerSphericalPotential')
    pots.append('NFWTwoPowerIntegerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('specialFlattenedPowerPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        #if not 'NFW' in p: continue #For testing the test
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue
        tp.normalize(1.)
        assert (tp.Rforce(1.,0.)+1.)**2. < 10.**-16., \
            "Normalization of %s potential fails" % p
        assert (tp.vcirc(1.)**2.-1.)**2. < 10.**-16., \
            "Normalization of %s potential fails" % p
        tp.normalize(.5)
        assert (tp.Rforce(1.,0.)+.5)**2. < 10.**-16., \
            "Normalization of %s potential fails" % p
        assert (tp.vcirc(1.)**2.-0.5)**2. < 10.**-16., \
            "Normalization of %s potential fails" % p

#Test whether the derivative of the potential is minus the force
def test_forceAsDeriv_potential():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockTwoPowerIntegerSphericalPotential')
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerIntegerSphericalPotential')
    pots.append('JaffeTwoPowerIntegerSphericalPotential')
    pots.append('NFWTwoPowerIntegerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('testlinearMWPotential')
    pots.append('mockInterpRZPotential')
    pots.append('mockSnapshotRZPotential')
    pots.append('mockInterpSnapshotRZPotential')
    pots.append('mockCosmphiDiskPotentialT1')
    pots.append('mockCosmphiDiskPotentialTm1')
    pots.append('mockCosmphiDiskPotentialTm5')
    pots.append('mockDehnenBarPotentialT1')
    pots.append('mockDehnenBarPotentialTm1')
    pots.append('mockDehnenBarPotentialTm5')
    pots.append('mockEllipticalDiskPotentialT1')
    pots.append('mockEllipticalDiskPotentialTm1')
    pots.append('mockEllipticalDiskPotentialTm5')
    pots.append('mockSteadyLogSpiralPotentialT1')
    pots.append('mockSteadyLogSpiralPotentialTm1')
    pots.append('mockSteadyLogSpiralPotentialTm5')
    pots.append('mockTransientLogSpiralPotential')
    pots.append('mockFlatEllipticalDiskPotential') #for evaluate w/ nonaxi lists
    pots.append('mockMovingObjectPotential')
    pots.append('mockMovingObjectExplSoftPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    Rs= numpy.array([0.5,1.,2.])
    Zs= numpy.array([0.,.125,-.125,0.25,-0.25])
    phis= numpy.array([0.,0.5,-0.5,1.,-1.,
                       numpy.pi,0.5+numpy.pi,
                       1.+numpy.pi])
    #tolerances in log10
    tol= {}
    tol['default']= -8.
    tol['DoubleExponentialDiskPotential']= -6. #these are more difficult
    tol['RazorThinExponentialDiskPotential']= -6.
    tol['mockInterpRZPotential']= -4.
    for p in pots:
        #if not 'NFW' in p: continue #For testing the test
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'normalize'): tp.normalize(1.)
        #Set tolerance
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        #Radial force
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                dr= 10.**-8.
                newR= Rs[ii]+dr
                dr= newR-Rs[ii] #Representable number
                if isinstance(tp,potential.linearPotential): 
                    mpotderivR= (potential.evaluatelinearPotentials(Rs[ii],tp)
                                 -potential.evaluatelinearPotentials(Rs[ii]+dr,
                                                                     tp))/dr
                    tRforce= potential.evaluatelinearForces(Rs[ii],tp)
                elif isinstance(tp,potential.planarPotential):
                    mpotderivR= (potential.evaluateplanarPotentials(Rs[ii],tp,phi=Zs[jj])-potential.evaluateplanarPotentials(Rs[ii]+dr,tp,phi=Zs[jj]))/dr
                    tRforce= potential.evaluateplanarRforces(Rs[ii],tp,
                                                             phi=Zs[jj])
                else:
                    mpotderivR= (potential.evaluatePotentials(Rs[ii],Zs[jj],tp)
                                 -potential.evaluatePotentials(Rs[ii]+dr,Zs[jj],
                                                               tp))/dr
                    tRforce= potential.evaluateRforces(Rs[ii],Zs[jj],tp)
                if tRforce**2. < 10.**ttol:
                    assert mpotderivR**2. < 10.**ttol, \
                        "Calculation of the Radial force as the Radial derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRforce-mpotderivR), numpy.fabs((tRforce-mpotderivR)/tRforce))
                else:
                    assert (tRforce-mpotderivR)**2./tRforce**2. < 10.**ttol, \
                        "Calculation of the Radial force as the Radial derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRforce-mpotderivR), numpy.fabs((tRforce-mpotderivR)/tRforce))
        #Azimuthal force, if it exists
        if isinstance(tp,potential.linearPotential): continue
        for ii in range(len(Rs)):
            for jj in range(len(phis)):
                dphi= 10.**-8.
                newphi= phis[jj]+dphi
                dphi= newphi-phis[jj] #Representable number
                if isinstance(tp,potential.planarPotential):
                    mpotderivphi= (tp(Rs[ii],phi=phis[jj])-tp(Rs[ii],phi=phis[jj]+dphi))/dphi
                    tphiforce= potential.evaluateplanarphiforces(Rs[ii],tp,
                                                                 phi=phis[jj])
                else:
                    mpotderivphi= (tp(Rs[ii],0.05,phi=phis[jj])-tp(Rs[ii],0.05,phi=phis[jj]+dphi))/dphi
                    tphiforce= potential.evaluatephiforces(Rs[ii],0.05,tp,
                                                           phi=phis[jj])
                try:
                    if tphiforce**2. < 10.**ttol:
                        assert(mpotderivphi**2. < 10.**ttol)
                    else:
                        assert((tphiforce-mpotderivphi)**2./tphiforce**2. < 10.**ttol)
                except AssertionError:
                    if isinstance(tp,potential.planarPotential):
                        raise AssertionError("Calculation of the azimuthal force as the azimuthal derivative of the %s potential fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(mpotderivphi),numpy.fabs((tphiforce-mpotderivphi)/tphiforce)))
                    else:
                        raise AssertionError("Calculation of the azimuthal force as the azimuthal derivative of the %s potential fails at (R,Z,phi) = (%.3f,0.05,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(mpotderivphi),numpy.fabs((tphiforce-mpotderivphi)/tphiforce)))
        #Vertical force, if it exists
        if isinstance(tp,potential.planarPotential) \
                or isinstance(tp,potential.linearPotential): continue
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                dz= 10.**-8.
                newZ= Zs[jj]+dz
                dz= newZ-Zs[jj] #Representable number
                mpotderivz= (tp(Rs[ii],Zs[jj])-tp(Rs[ii],Zs[jj]+dz))/dz
                tzforce= potential.evaluatezforces(Rs[ii],Zs[jj],tp)
                if tzforce**2. < 10.**ttol:
                    assert mpotderivz**2. < 10.**ttol, \
                        "Calculation of the vertical force as the vertical derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(mpotderivz),numpy.fabs((tzforce-mpotderivz)/tzforce))
                else:
                    assert (tzforce-mpotderivz)**2./tzforce**2. < 10.**ttol, \
"Calculation of the vertical force as the vertical derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(mpotderivz),numpy.fabs((tzforce-mpotderivz)/tzforce))

#Test whether the second derivative of the potential is minus the derivative of the force
def test_2ndDeriv_potential():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockTwoPowerIntegerSphericalPotential')
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerIntegerSphericalPotential')
    pots.append('JaffeTwoPowerIntegerSphericalPotential')
    pots.append('NFWTwoPowerIntegerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('testlinearMWPotential')
    pots.append('mockInterpRZPotential')
    pots.append('mockCosmphiDiskPotentialT1')
    pots.append('mockCosmphiDiskPotentialTm1')
    pots.append('mockCosmphiDiskPotentialTm5')
    pots.append('mockDehnenBarPotentialT1')
    pots.append('mockDehnenBarPotentialTm1')
    pots.append('mockDehnenBarPotentialTm5')
    pots.append('mockEllipticalDiskPotentialT1')
    pots.append('mockEllipticalDiskPotentialTm1')
    pots.append('mockEllipticalDiskPotentialTm5')
    pots.append('mockSteadyLogSpiralPotentialT1')
    pots.append('mockSteadyLogSpiralPotentialTm1')
    pots.append('mockSteadyLogSpiralPotentialTm5')
    pots.append('mockTransientLogSpiralPotential')
    pots.append('mockFlatEllipticalDiskPotential') #for evaluate w/ nonaxi lists
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    Rs= numpy.array([0.5,1.,2.])
    Zs= numpy.array([0.,.125,-.125,0.25,-0.25])
    phis= numpy.array([0.,0.5,-0.5,1.,-1.,
                       numpy.pi,0.5+numpy.pi,
                       1.+numpy.pi])
    #tolerances in log10
    tol= {}
    tol['default']= -8.
    tol['DoubleExponentialDiskPotential']= -3. #these are more difficult
    tol['RazorThinExponentialDiskPotential']= -6.
    tol['mockInterpRZPotential']= -4.
    for p in pots:
        #if not 'NFW' in p: continue #For testing the test
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'normalize'): tp.normalize(1.)
        #Set tolerance
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        #2nd radial
        if hasattr(tp,'_R2deriv'):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
                    if p == 'RazorThinExponentialDiskPotential' and numpy.fabs(Zs[jj]) > 0.: continue #Not implemented
                    dr= 10.**-8.
                    newR= Rs[ii]+dr
                    dr= newR-Rs[ii] #Representable number
                    if isinstance(tp,potential.linearPotential): 
                        mRforcederivR= (tp.Rforce(Rs[ii])-tp.Rforce(Rs[ii]+dr))/dr
                        tR2deriv= tp.R2deriv(Rs[ii])
                    elif isinstance(tp,potential.planarPotential): 
                        mRforcederivR= (tp.Rforce(Rs[ii],Zs[jj])-tp.Rforce(Rs[ii]+dr,Zs[jj]))/dr
                        tR2deriv= potential.evaluateplanarR2derivs(Rs[ii],tp,
                                                                   phi=Zs[jj])
                    else:
                        mRforcederivR= (tp.Rforce(Rs[ii],Zs[jj])-tp.Rforce(Rs[ii]+dr,Zs[jj]))/dr
                        tR2deriv= potential.evaluateR2derivs(Rs[ii],Zs[jj],tp)
                    if tR2deriv**2. < 10.**ttol:
                        assert mRforcederivR**2. < 10.**ttol, \
                            "Calculation of the second Radial derivative of the potential as the Radial derivative of the %s Radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tR2deriv-mRforcederivR), numpy.fabs((tR2deriv-mRforcederivR)/tR2deriv))
                    else:
                        assert (tR2deriv-mRforcederivR)**2./tR2deriv**2. < 10.**ttol, \
                            "Calculation of the second Radial derivative of the potential as the Radial derivative of the %s Radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tR2deriv-mRforcederivR), numpy.fabs((tR2deriv-mRforcederivR)/tR2deriv))
        #2nd azimuthal
        if not isinstance(tp,potential.linearPotential) \
                and hasattr(tp,'_phi2deriv'):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    dphi= 10.**-8.
                    newphi= phis[jj]+dphi
                    dphi= newphi-phis[jj] #Representable number
                    if isinstance(tp,potential.planarPotential):
                        mphiforcederivphi= (tp.phiforce(Rs[ii],phi=phis[jj])-tp.phiforce(Rs[ii],phi=phis[jj]+dphi))/dphi
                        tphi2deriv= tp.phi2deriv(Rs[ii],phi=phis[jj])
                    else:
                        mphiforcederivphi= (tp.phiforce(Rs[ii],0.05,phi=phis[jj])-tp.phiforce(Rs[ii],0.05,phi=phis[jj]+dphi))/dphi
                        tphi2deriv= tp.phi2deriv(Rs[ii],0.05,phi=phis[jj])
                    try:
                        if tphi2deriv**2. < 10.**ttol:
                            assert(mphiforcederivphi**2. < 10.**ttol)
                        else:
                            assert((tphi2deriv-mphiforcederivphi)**2./tphi2deriv**2. < 10.**ttol)
                    except AssertionError:
                        if isinstance(tp,potential.planarPotential):
                            raise AssertionError("Calculation of the second azimuthal derivative of the potential as the azimuthal derivative of the %s azimuthal force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tphi2deriv-mphiforcederivphi), numpy.fabs((tphi2deriv-mphiforcederivphi)/tphi2deriv)))
                        else:
                            raise AssertionError("Calculation of the second azimuthal derivative of the potential as the azimuthal derivative of the %s azimuthal force fails at (R,Z,phi) = (%.3f,0.05,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tphi2deriv-mphiforcederivphi), numpy.fabs((tphi2deriv-mphiforcederivphi)/tphi2deriv)))
        #mixed radial azimuthal
        if not isinstance(tp,potential.linearPotential) \
                and hasattr(tp,'_Rphideriv'):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
                    dphi= 10.**-8.
                    newphi= phis[jj]+dphi
                    dphi= newphi-phis[jj] #Representable number
                    if isinstance(tp,potential.planarPotential):
                        mRforcederivphi= (tp.Rforce(Rs[ii],phi=phis[jj])-tp.Rforce(Rs[ii],phi=phis[jj]+dphi))/dphi
                        tRphideriv= tp.Rphideriv(Rs[ii],phi=phis[jj])
                    else:
                        mRforcederivphi= (tp.Rforce(Rs[ii],0.05,phi=phis[jj])-tp.Rforce(Rs[ii],0.05,phi=phis[jj]+dphi))/dphi
                        tRphideriv= tp.Rphideriv(Rs[ii],0.05,phi=phis[jj])
                    try:
                        if tRphideriv**2. < 10.**ttol:
                            assert(mRforcederivphi**2. < 10.**ttol)
                        else:
                            assert((tRphideriv-mRforcederivphi)**2./tRphideriv**2. < 10.**ttol)
                    except AssertionError:
                        if isinstance(tp,potential.planarPotential):
                            raise AssertionError("Calculation of the mixed radial, azimuthal derivative of the potential as the azimuthal derivative of the %s Radial force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tRphideriv-mRforcederivphi), numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv)))
                        else:
                            raise AssertionError("Calculation of the second azimuthal derivative of the potential as the azimuthal derivative of the %s azimuthal force fails at (R,Z,phi) = (%.3f,0.05,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tphi2deriv-mphiforcederivphi), numpy.fabs((tphi2deriv-mphiforcederivphi)/tphi2deriv)))
        #2nd vertical
        if not isinstance(tp,potential.planarPotential) \
                and not isinstance(tp,potential.linearPotential) \
                and hasattr(tp,'_z2deriv'):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    if p == 'TwoPowerSphericalPotential': continue #Not implemented, or badly defined
                    if p == 'mockTwoPowerIntegerSphericalPotential': continue #Not implemented, or badly defined
                    if p == 'specialTwoPowerSphericalPotential': continue #Not implemented, or badly defined
                    if p == 'HernquistTwoPowerIntegerSphericalPotential': continue #Not implemented, or badly defined
                    if p == 'JaffeTwoPowerIntegerSphericalPotential': continue #Not implemented, or badly defined
                    if p == 'NFWTwoPowerIntegerSphericalPotential': continue #Not implemented, or badly defined
                    dz= 10.**-8.
                    newz= Zs[jj]+dz
                    dz= newz-Zs[jj] #Representable number
                    mzforcederivz= (tp.zforce(Rs[ii],Zs[jj])-tp.zforce(Rs[ii],Zs[jj]+dz))/dz
                    tz2deriv= potential.evaluatez2derivs(Rs[ii],Zs[jj],tp)
                    if tz2deriv**2. < 10.**ttol:
                        assert mzforcederivz**2. < 10.**ttol, \
                            "Calculation of the second vertical derivative of the potential as the vertical derivative of the %s vertical force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tz2deriv-mzforcederivz), numpy.fabs((tz2deriv-mzforcederivz)/tz2deriv))
                    else:
                        assert (tz2deriv-mzforcederivz)**2./tz2deriv**2. < 10.**ttol, \
                            "Calculation of the second vertical derivative of the potential as the vertical derivative of the %s vertical force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tz2deriv-mzforcederivz), numpy.fabs((tz2deriv-mzforcederivz)/tz2deriv))
        #mixed radial vertical
        if not isinstance(tp,potential.planarPotential) \
                and not isinstance(tp,potential.linearPotential) \
                and hasattr(tp,'_Rzderiv'):
            for ii in range(len(Rs)):
                for jj in range(len(Zs)):
#                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    dz= 10.**-8.
                    newz= Zs[jj]+dz
                    dz= newz-Zs[jj] #Representable number
                    mRforcederivz= (tp.Rforce(Rs[ii],Zs[jj])-tp.Rforce(Rs[ii],Zs[jj]+dz))/dz
                    tRzderiv= potential.evaluateRzderivs(Rs[ii],Zs[jj],tp)
                    if tRzderiv**2. < 10.**ttol:
                        assert mRforcederivz**2. < 10.**ttol, \
                            "Calculation of the mixed radial vertical derivative of the potential as the vertical derivative of the %s radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRzderiv-mRforcederivz), numpy.fabs((tRzderiv-mRforcederivz)/tRzderiv))
                    else:
                        assert (tRzderiv-mRforcederivz)**2./tRzderiv**2. < 10.**ttol, \
"Calculation of the mixed radial vertical derivative of the potential as the vertical derivative of the %s radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRzderiv-mRforcederivz), numpy.fabs((tRzderiv-mRforcederivz)/tRzderiv))                        
        #mixed radial, azimuthal
        if not isinstance(tp,potential.linearPotential) \
                and hasattr(tp,'_Rphideriv'):
            for ii in range(len(Rs)):
                for jj in range(len(phis)):
#                    if p == 'RazorThinExponentialDiskPotential': continue #Not implemented, or badly defined
                    dphi= 10.**-8.
                    newphi= phis[jj]+dphi
                    dphi= newphi-phis[jj] #Representable number
                    if isinstance(tp,potential.planarPotential):
                        mRforcederivphi= (tp.Rforce(Rs[ii],phi=phis[jj])\
                                              -tp.Rforce(Rs[ii],phi=phis[jj]+dphi))/dphi
                        tRphideriv= potential.evaluateplanarPotentials(Rs[ii],tp,
                                                                       phi=phis[jj],dR=1,dphi=1)
                    else:
                        mRforcederivphi= (tp.Rforce(Rs[ii],0.1,phi=phis[jj])\
                                              -tp.Rforce(Rs[ii],0.1,phi=phis[jj]+dphi))/dphi
                        tRphideriv= potential.evaluatePotentials(Rs[ii],0.1,tp,
                                                                 phi=phis[jj],dR=1,dphi=1)
                    if tRphideriv**2. < 10.**ttol:
                        assert mRforcederivphi**2. < 10.**ttol, \
                            "Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the %s radial force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tRphideriv-mRforcederivphi), numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv))
                    else:
                        assert (tRphideriv-mRforcederivphi)**2./tRphideriv**2. < 10.**ttol, \
"Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the %s radial force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tRphideriv-mRforcederivphi), numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv))

#Test whether the Poisson equation is satisfied if _dens and the relevant second derivatives are implemented
def test_poisson_potential():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockTwoPowerIntegerSphericalPotential')
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerIntegerSphericalPotential')
    pots.append('JaffeTwoPowerIntegerSphericalPotential')
    pots.append('NFWTwoPowerIntegerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    pots.append('testlinearMWPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    Rs= numpy.array([0.5,1.,2.])
    Zs= numpy.array([0.,.125,-.125,0.25,-0.25])
    phis= numpy.array([0.,0.5,-0.5,1.,-1.,
                       numpy.pi,0.5+numpy.pi,
                       1.+numpy.pi])
    #tolerances in log10
    tol= {}
    tol['default']= -8.
    tol['DoubleExponentialDiskPotential']= -3. #these are more difficult
    #tol['RazorThinExponentialDiskPotential']= -6.
    for p in pots:
        #if not 'NFW' in p: continue #For testing the test
        #if 'Isochrone' in p: continue #For testing the test
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'normalize'): tp.normalize(1.)
        #Set tolerance
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        #2nd radial
        if not hasattr(tp,'_dens') or not hasattr(tp,'_R2deriv') \
                or not hasattr(tp,'_Rforce') or not hasattr(tp,'phi2deriv') \
                or not hasattr(tp,'_z2deriv'):
            continue
        for ii in range(len(Rs)):
            for jj in range(len(Zs)):
                for kk in range(len(phis)):
                    tpoissondens= tp.dens(Rs[ii],Zs[jj],phi=phis[kk],
                                         forcepoisson=True)
                    tdens= potential.evaluateDensities(Rs[ii],Zs[jj],tp,
                                                       phi=phis[kk],
                                                       forcepoisson=False)
                    if tdens**2. < 10.**ttol:
                        assert tpoissondens**2. < 10.**ttol, \
                            "Poisson equation relation between the derivatives of the potential and the implemented density is not satisfied for the %s potential at (R,Z,phi) = (%.3f,%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],phis[kk],numpy.fabs(tdens-tpoissondens), numpy.fabs((tdens-tpoissondens)/tdens))
                    else:
                        assert (tpoissondens-tdens)**2./tdens**2. < 10.**ttol, \
                            "Poisson equation relation between the derivatives of the potential and the implemented density is not satisfied for the %s potential at (R,Z,phi) = (%.3f,%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],phis[kk],numpy.fabs(tdens-tpoissondens), numpy.fabs((tdens-tpoissondens)/tdens))
    return None
                        
#Test whether the _evaluate function is correctly implemented in specifying derivatives
def test_evaluateAndDerivs_potential():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    pots.append('mockTwoPowerIntegerSphericalPotential')
    pots.append('specialTwoPowerSphericalPotential')
    pots.append('HernquistTwoPowerIntegerSphericalPotential')
    pots.append('JaffeTwoPowerIntegerSphericalPotential')
    pots.append('NFWTwoPowerIntegerSphericalPotential')
    pots.append('specialMiyamotoNagaiPotential')
    pots.append('specialFlattenedPowerPotential')
    pots.append('specialPowerSphericalPotential')
    pots.append('mockCosmphiDiskPotentialT1')
    pots.append('mockCosmphiDiskPotentialTm1')
    pots.append('mockCosmphiDiskPotentialTm5')
    pots.append('mockDehnenBarPotentialT1')
    pots.append('mockDehnenBarPotentialTm1')
    pots.append('mockDehnenBarPotentialTm5')
    pots.append('mockEllipticalDiskPotentialT1')
    pots.append('mockEllipticalDiskPotentialTm1')
    pots.append('mockEllipticalDiskPotentialTm5')
    pots.append('mockSteadyLogSpiralPotentialT1')
    pots.append('mockSteadyLogSpiralPotentialTm1')
    pots.append('mockSteadyLogSpiralPotentialTm5')
    pots.append('mockTransientLogSpiralPotential')
    pots.append('mockMovingObjectPotential')
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    #tolerances in log10
    tol= {}
    tol['default']= -12.
    #tol['DoubleExponentialDiskPotential']= -3. #these are more difficult
    #tol['RazorThinExponentialDiskPotential']= -6.
    for p in pots:
        #if 'Isochrone' in p: continue #For testing the test
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if hasattr(tp,'normalize'): tp.normalize(1.)
        #Set tolerance
        if p in tol.keys(): ttol= tol[p]
        else: ttol= tol['default']
        #1st radial
        if isinstance(tp,potential.linearPotential): 
            continue
        elif isinstance(tp,potential.planarPotential): 
            tevaldr= tp(1.2,phi=0.1,dR=1)
            trforce= tp.Rforce(1.2,phi=0.1)
        else:
            tevaldr= tp(1.2,0.1,phi=0.1,dR=1)
            trforce= tp.Rforce(1.2,0.1,phi=0.1)
        if not tevaldr is None:
            if tevaldr**2. < 10.**ttol:
                assert trforce**2. < 10.**ttol, \
"Calculation of radial derivative through _evaluate and Rforce inconsistent for the %s potential" % p
            else:
                assert (tevaldr+trforce)**2./tevaldr**2. < 10.**ttol, \
                    "Calculation of radial derivative through _evaluate and Rforce inconsistent for the %s potential" % p                
        #2nd radial
        hasR2= True
        from galpy.potential import PotentialError
        if 'RazorThin' in p: R2z= 0.
        else: R2z= 0.1
        try:
            if isinstance(tp,potential.planarPotential): 
                tp.R2deriv(1.2)
            else:
                tp.R2deriv(1.2,R2z)
        except PotentialError:
            hasR2= False
        if hasR2:
            if isinstance(tp,potential.planarPotential): 
                tevaldr2= tp(1.2,phi=0.1,dR=2)
                tr2deriv= tp.R2deriv(1.2,phi=0.1)
            else:
                tevaldr2= tp(1.2,R2z,phi=0.1,dR=2)
                tr2deriv= tp.R2deriv(1.2,R2z,phi=0.1)
            if not tevaldr2 is None:
                if tevaldr2**2. < 10.**ttol:
                    assert tr2deriv*2. < 10.**ttol, \
                        "Calculation of 2nd radial derivative through _evaluate and R2deriv inconsistent for the %s potential" % p
                else:
                    assert (tevaldr2-tr2deriv)**2./tevaldr2**2. < 10.**ttol, \
                        "Calculation of 2nd radial derivative through _evaluate and R2deriv inconsistent for the %s potential" % p                    
        #1st phi
        if isinstance(tp,potential.planarPotential): 
            tevaldphi= tp(1.2,phi=0.1,dphi=1)
            tphiforce= tp.phiforce(1.2,phi=0.1)
        else:
            tevaldphi= tp(1.2,0.1,phi=0.1,dphi=1)
            tphiforce= tp.phiforce(1.2,0.1,phi=0.1)
        if not tevaldphi is None:
            if tevaldphi**2. < 10.**ttol:
                assert tphiforce**2. < 10.**ttol, \
                    "Calculation of azimuthal derivative through _evaluate and phiforce inconsistent for the %s potential" % p
            else:
                assert (tevaldphi+tphiforce)**2./tevaldphi**2. < 10.**ttol, \
                    "Calculation of azimuthal derivative through _evaluate and phiforce inconsistent for the %s potential" % p
        #2nd phi
        hasphi2= True
        try:
            if isinstance(tp,potential.planarPotential): 
                tp.phi2deriv(1.2,phi=0.1)
            else:
                tp.phi2deriv(1.2,0.1,phi=0.1)
        except (PotentialError,AttributeError):
            hasphi2= False
        if hasphi2 and hasattr(tp,'_phi2deriv'):
            if isinstance(tp,potential.planarPotential): 
                tevaldphi2= tp(1.2,phi=0.1,dphi=2)
                tphi2deriv= tp.phi2deriv(1.2,phi=0.1)
            else:
                tevaldphi2= tp(1.2,0.1,phi=0.1,dphi=2)
                tphi2deriv= tp.phi2deriv(1.2,0.1,phi=0.1)
            if not tevaldphi2 is None:
                if tevaldphi2**2. < 10.**ttol:
                    assert tphi2deriv*2. < 10.**ttol, \
                        "Calculation of 2nd azimuthal derivative through _evaluate and phi2deriv inconsistent for the %s potential" % p
                else:
                    assert (tevaldphi2-tphi2deriv)**2./tevaldphi2**2. < 10.**ttol, \
                        "Calculation of 2nd azimuthal derivative through _evaluate and phi2deriv inconsistent for the %s potential" % p
        continue
        #mixed radial,vertical
        if isinstance(tp,potential.planarPotential): 
            tevaldrz= tp(1.2,0.1,phi=0.1,dR=1,dz=1)
            trzderiv= tp.Rzderiv(1.2,0.1,phi=0.1)
        else:
            tevaldrz= tp(1.2,0.1,phi=0.1,dR=1,dz=1)
            trzderiv= tp.Rzderiv(1.2,0.1,phi=0.1)
        if not tevaldrz is None:
            if tevaldrz**2. < 10.**ttol:
                assert trzderiv*2. < 10.**ttol, \
                    "Calculation of mixed radial,vertical derivative through _evaluate and z2deriv inconsistent for the %s potential" % p
            else:
                assert (tevaldrz-trzderiv)**2./tevaldrz**2. < 10.**ttol, \
"Calculation of mixed radial,vertical derivative through _evaluate and z2deriv inconsistent for the %s potential" % p
    #Finally test that much higher derivatives are not implemented
    try: tp(1.2,0.1,dR=4,dphi=10)
    except NotImplementedError: pass
    else: raise AssertionError('Higher-order derivative request in potential __call__ does not raise NotImplementedError')
    return None

# Check that the masses are calculated correctly for spherical potentials
def test_mass_spher():
    #PowerPotential close to Kepler should be very steep
    pp= potential.PowerSphericalPotential(amp=2.,alpha=3.001)
    kp= potential.KeplerPotential(amp=2.)
    assert numpy.fabs(((pp.mass(10.)-kp.mass(10.)))/kp.mass(10.)) < 10.**-2., "Mass for PowerSphericalPotential close to KeplerPotential is not close to KeplerPotential's mass"
    pp= potential.PowerSphericalPotential(amp=2.)
    #mass = amp x r^(3-alpha)
    tR= 1.
    assert numpy.fabs(pp.mass(tR,forceint=True)-pp._amp*tR**(3.-pp.alpha)) < 10.**-10., 'Mass for PowerSphericalPotential not as expected'
    tR= 2.
    assert numpy.fabs(pp.mass(tR,forceint=True)-pp._amp*tR**(3.-pp.alpha)) < 10.**-10., 'Mass for PowerSphericalPotential not as expected'
    tR= 20.
    assert numpy.fabs(pp.mass(tR,forceint=True)-pp._amp*tR**(3.-pp.alpha)) < 10.**-10., 'Mass for PowerSphericalPotential not as expected'
    #Test that for a cut-off potential, the mass far beyond the cut-off is 
    # 2pi rc^(3-alpha) gamma(1.5-alpha/2)
    pp= potential.PowerSphericalPotentialwCutoff(amp=2.)
    from scipy import special
    expecMass= 2.*pp._amp*numpy.pi*pp.rc**(3.-pp.alpha)*special.gamma(1.5-pp.alpha/2.)
    tR= 5.
    assert numpy.fabs((pp.mass(tR,forceint=True)-expecMass)/expecMass) < 10.**-6., 'Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected'
    tR= 15.
    assert numpy.fabs((pp.mass(tR,forceint=True)-expecMass)/expecMass) < 10.**-6., 'Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected'
    tR= 50.
    assert numpy.fabs((pp.mass(tR,forceint=True)-expecMass)/expecMass) < 10.**-6., 'Mass of PowerSphericalPotentialwCutoff far beyond the cut-off not as expected'
    #Jaffe and Hernquist both have finite masses, NFW diverges logarithmically
    jp= potential.JaffePotential(amp=2.,a=0.1)
    hp= potential.HernquistPotential(amp=2.,a=0.1)
    np= potential.NFWPotential(amp=2.,a=0.1)
    tR= 10.
    # Limiting behavior
    jaffemass= jp._amp*(1.-jp.a/tR)
    hernmass= hp._amp/2.*(1.-2.*hp.a/tR)
    nfwmass= np._amp*(numpy.log(tR/np.a)-1.+np.a/tR)
    assert numpy.fabs((jp.mass(tR,forceint=True)-jaffemass)/jaffemass) < 10.**-3., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((hp.mass(tR,forceint=True)-hernmass)/hernmass) < 10.**-3., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((np.mass(tR,forceint=True)-nfwmass)/nfwmass) < 10.**-2., 'Limit mass for NFW potential not as expected'
    tR= 200.
    # Limiting behavior, add z, to test that too
    jaffemass= jp._amp*(1.-jp.a/tR)
    hernmass= hp._amp/2.*(1.-2.*hp.a/tR)
    nfwmass= np._amp*(numpy.log(tR/np.a)-1.+np.a/tR)
    assert numpy.fabs((jp.mass(tR,forceint=True)-jaffemass)/jaffemass) < 10.**-6., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((hp.mass(tR,forceint=True)-hernmass)/hernmass) < 10.**-6., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((np.mass(tR,forceint=True)-nfwmass)/nfwmass) < 10.**-4., 'Limit mass for NFW potential not as expected'
    tR, tz= 200., 10.
    tr= numpy.sqrt(tR**2.+tz**2.)
    # Limiting behavior, add z, to test that too
    jaffemass= jp._amp*(1.-jp.a/tr)
    hernmass= hp._amp/2.*(1.-2.*hp.a/tr)
    nfwmass= np._amp*(numpy.log(tr/np.a)-1.+np.a/tr)
    assert numpy.fabs((jp.mass(tR,z=tz,forceint=False)-jaffemass)/jaffemass) < 10.**-6., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((hp.mass(tR,z=tz,forceint=False)-hernmass)/hernmass) < 10.**-6., 'Limit mass for Jaffe potential not as expected'
    assert numpy.fabs((np.mass(tR,z=tz,forceint=False)-nfwmass)/nfwmass) < 10.**-4., 'Limit mass for NFW potential not as expected'
    return None

# Check that the masses are implemented correctly for spherical potentials
def test_mass_spher_analytic():
    #TwoPowerSphericalPotentials all have explicitly implemented masses
    jp= potential.JaffePotential(amp=2.)
    hp= potential.HernquistPotential(amp=2.)
    np= potential.NFWPotential(amp=2.)
    tp= potential.TwoPowerSphericalPotential(amp=2.)
    tR= 2.
    assert numpy.fabs(jp.mass(tR,forceint=True)-jp.mass(tR)) < 10.**-10., 'Explicit mass does not agree with integral of the density for Jaffe potential'
    assert numpy.fabs(hp.mass(tR,forceint=True)-hp.mass(tR)) < 10.**-10., 'Explicit mass does not agree with integral of the density for Hernquist potential'
    assert numpy.fabs(np.mass(tR,forceint=True)-np.mass(tR)) < 10.**-10., 'Explicit mass does not agree with integral of the density for NFW potential'
    assert numpy.fabs(tp.mass(tR,forceint=True)-tp.mass(tR)) < 10.**-10., 'Explicit mass does not agree with integral of the density for TwoPowerSpherical potential'
    assert numpy.fabs(tp.mass(tR,forceint=True)-tp.mass(numpy.sqrt(tR**2.-1**2.),z=1.)) < 10.**-10., 'Explicit mass does not agree with integral of the density for TwoPowerSpherical potential, for not z is None'
    return None

# Check that the masses are calculated correctly for axisymmetric potentials
def test_mass_axi():
    #For Miyamoto-Nagai, we know that mass integrated over everything should be equal to amp, so
    mp= potential.MiyamotoNagaiPotential(amp=1.)
    assert numpy.fabs(mp.mass(200.,20.)-1.) < 0.01, 'Total mass of Miyamoto-Nagai potential w/ amp=1 is not equal to 1'
    #For a double-exponential disk potential, the 
    # mass(R,z) = amp x hR^2 x hz x (1-(1+R/hR)xe^(-R/hR)) x (1-e^(-Z/hz)
    dp= potential.DoubleExponentialDiskPotential(amp=2.)
    def dblexpmass(r,z,dp):
        return 4.*numpy.pi*dp._amp*dp._hr**2.*dp._hz*(1.-(1.+r/dp._hr)*numpy.exp(-r/dp._hr))*(1.-numpy.exp(-z/dp._hz))
    tR,tz= 0.01,0.01
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-10., 'Mass for DoubleExponentialDiskPotential incorrect'
    tR,tz= 0.1,0.05
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-10., 'Mass for DoubleExponentialDiskPotential incorrect'
    tR,tz= 1.,0.1
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-10., 'Mass for DoubleExponentialDiskPotential incorrect'
    tR,tz= 5.,0.1
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-10., 'Mass for DoubleExponentialDiskPotential incorrect'
    tR,tz= 5.,1.
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-10., 'Mass for DoubleExponentialDiskPotential incorrect'
    tR,tz= 100.,100.
    assert numpy.fabs((dp.mass(tR,tz,forceint=True)-dblexpmass(tR,tz,dp))/dblexpmass(tR,tz,dp)) < 10.**-6., 'Mass for DoubleExponentialDiskPotential incorrect'
    #Test that nonAxi raises error
    from galpy.orbit import Orbit
    mop= potential.MovingObjectPotential(Orbit([1.,0.1,1.1,0.1,0.,0.]))
    try: mop.mass(1.,0.)
    except NotImplementedError: pass
    else: raise AssertionError('mass for non-axisymmetric potential should have raised NotImplementedError, but did not')
    return None

# Check that toVertical and toPlanar work
def test_toVertical_toPlanar():
    #Grab all of the potentials
    pots= [p for p in dir(potential) 
           if ('Potential' in p and not 'plot' in p and not 'RZTo' in p 
               and not 'evaluate' in p)]
    rmpots= ['Potential','MWPotential','MWPotential2014',
             'MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError',
             'SnapshotRZPotential','InterpSnapshotRZPotential']
    if False: #_TRAVIS: #travis CI
        rmpots.append('DoubleExponentialDiskPotential')
        rmpots.append('RazorThinExponentialDiskPotential')
    for p in rmpots:
        pots.remove(p)
    for p in pots:
        #Setup instance of potential
        try:
            tclass= getattr(potential,p)
        except AttributeError:
            tclass= getattr(sys.modules[__name__],p)
        tp= tclass()
        if not hasattr(tp,'normalize'): continue #skip these
        tp.normalize(1.)
        if isinstance(tp,potential.linearPotential) or \
                isinstance(tp,potential.planarPotential):
            continue
        tpp= tp.toPlanar()
        assert isinstance(tpp,potential.planarPotential), \
            "Conversion into planar potential of potential %s fails" % p
        tlp= tp.toVertical(1.)
        assert isinstance(tlp,potential.linearPotential), \
            "Conversion into linear potential of potential %s fails" % p

def test_RZToplanarPotential():
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    plp= potential.RZToplanarPotential(lp)
    assert isinstance(plp,potential.planarPotential), 'Running an RZPotential through RZToplanarPotential does not produce a planarPotential'
    #Check that a planarPotential through RZToplanarPotential is still planar
    pplp= potential.RZToplanarPotential(lp)
    assert isinstance(pplp,potential.planarPotential), 'Running a planarPotential through RZToplanarPotential does not produce a planarPotential'
    try:
        plp= potential.RZToplanarPotential('something else')
    except potential.PotentialError:
        pass
    else:
        raise AssertionError('Using RZToplanarPotential with a string rather than an RZPotential or a planarPotential did not raise PotentialError')
    return None

# Sanity check the derivative of the rotation curve and the frequencies in the plane
def test_dvcircdR_omegac_epifreq_rl_vesc():
    #Derivative of rotation curve
    #LogarithmicHaloPotential: rotation everywhere flat
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    assert lp.dvcircdR(1.)**2. < 10.**-16., \
        "LogarithmicHaloPotential's rotation curve is not flat at R=1"
    assert lp.dvcircdR(0.5)**2. < 10.**-16., \
        "LogarithmicHaloPotential's rotation curve is not flat at R=0.5"
    assert lp.dvcircdR(2.)**2. < 10.**-16., \
        "LogarithmicHaloPotential's rotation curve is not flat at R=2"
    #Kepler potential, vc = vc_0(R/R0)^-0.5 -> dvcdR= -0.5 vc_0 (R/R0)**-1.5
    kp= potential.KeplerPotential(normalize=1.)
    assert (kp.dvcircdR(1.)+0.5)**2. < 10.**-16., \
        "KeplerPotential's rotation curve is not what it should be at R=1"
    assert (kp.dvcircdR(0.5)+0.5**-0.5)**2. < 10.**-16., \
        "KeplerPotential's rotation curve is not what it should be at R=0.5"
    assert (kp.dvcircdR(2.)+0.5**2.5)**2. < 10.**-16., \
        "KeplerPotential's rotation curve is not what it should be at R=2"
    #Rotational frequency
    assert (lp.omegac(1.)-1.)**2. < 10.**-16., \
        "LogarithmicHalo's rotational frequency is off at R=1"
    assert (lp.omegac(0.5)-2.)**2. < 10.**-16., \
        "LogarithmicHalo's rotational frequency is off at R=0.5"
    assert (lp.omegac(2.)-0.5)**2. < 10.**-16., \
        "LogarithmicHalo's rotational frequency is off at R=2"
    assert (lp.toPlanar().omegac(2.)-0.5)**2. < 10.**-16., \
        "LogarithmicHalo's rotational frequency is off at R=2 through planarPotential"
    #Epicycle frequency, flat rotation curve
    assert (lp.epifreq(1.)-numpy.sqrt(2.)*lp.omegac(1.))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=1"
    assert (lp.epifreq(0.5)-numpy.sqrt(2.)*lp.omegac(0.5))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=0.5"
    assert (lp.epifreq(2.0)-numpy.sqrt(2.)*lp.omegac(2.0))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=2"
    assert (lp.toPlanar().epifreq(2.0)-numpy.sqrt(2.)*lp.omegac(2.0))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=, through planar2"
    #Epicycle frequency, Kepler
    assert (kp.epifreq(1.)-kp.omegac(1.))**2. < 10.**-16., \
        "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=1"
    assert (kp.epifreq(0.5)-kp.omegac(0.5))**2. < 10.**-16., \
        "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=0.5"
    assert (kp.epifreq(2.)-kp.omegac(2.))**2. < 10.**-16., \
        "KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=2"
    #Check radius of circular orbit, Kepler
    assert (kp.rl(1.)-1.)**2. < 10.**-16., \
        "KeplerPotential's radius of a circular orbit is wrong at Lz=1."
    assert (kp.rl(0.5)-1./4.)**2. < 10.**-16., \
        "KeplerPotential's radius of a circular orbit is wrong at Lz=0.5"
    assert (kp.rl(2.)-4.)**2. < 10.**-16., \
        "KeplerPotential's radius of a circular orbit is wrong at Lz=2."
    #Check radius of circular orbit, PowerSphericalPotential with close-to-flat rotation curve
    pp= potential.PowerSphericalPotential(alpha=1.8,normalize=1.)
    assert (pp.rl(1.)-1.)**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=1."
    assert (pp.rl(0.5)-0.5**(10./11.))**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=0.5"
    assert (pp.rl(2.)-2.**(10./11.))**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=2."
    #Check radius of circular orbit, PowerSphericalPotential with steeper rotation curve
    pp= potential.PowerSphericalPotential(alpha=0.5,normalize=1.)
    assert (pp.rl(1.)-1.)**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=1."
    assert (pp.rl(0.0625)-0.0625**(4./7.))**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=0.0625"
    assert (pp.rl(16.)-16.**(4./7.))**2. < 10.**-16., \
        "PowerSphericalPotential's radius of a circular orbit is wrong at Lz=16."
    #Escape velocity of Kepler potential
    assert (kp.vesc(1.)**2.-2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=1"
    assert (kp.vesc(0.5)**2.-2.*kp.vcirc(0.5)**2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=0.5"
    assert (kp.vesc(2.)**2.-2.*kp.vcirc(2.)**2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=2"
    assert (kp.toPlanar().vesc(2.)**2.-2.*kp.vcirc(2.)**2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=2, through planar"
    # W/ different interface
    assert (kp.vcirc(1.)-potential.vcirc(kp,1.))**2. < 10.**-16., \
        "KeplerPotential's circular velocity does not agree between kp.vcirc and vcirc(kp)"
    assert (kp.vcirc(1.)-potential.vcirc(kp.toPlanar(),1.))**2. < 10.**-16., \
        "KeplerPotential's circular velocity does not agree between kp.vcirc and vcirc(kp.toPlanar)"
    assert (kp.vesc(1.)-potential.vesc(kp,1.))**2. < 10.**-16., \
        "KeplerPotential's escape velocity does not agree between kp.vesc and vesc(kp)"
    assert (kp.vesc(1.)-potential.vesc(kp.toPlanar(),1.))**2. < 10.**-16., \
        "KeplerPotential's escape velocity does not agree between kp.vesc and vesc(kp.toPlanar)"
    return None

def test_vcirc_vesc_special():
    #Test some special cases of vcirc and vesc
    dp= potential.DehnenBarPotential()
    try:
        potential.plotRotcurve([dp])
    except AttributeError: #should be raised
        pass
    else:
        raise AssertionError("plotRotcurve for non-axisymmetric potential should have raised AttributeError, but didn't")
    try:
        potential.plotEscapecurve([dp])
    except AttributeError: #should be raised
        pass
    else:
        raise AssertionError("plotEscapecurve for non-axisymmetric potential should have raised AttributeError, but didn't")
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    assert numpy.fabs(potential.calcRotcurve(lp,0.8)-lp.vcirc(0.8)) < 10.**-16., 'Circular velocity calculated with calcRotcurve not the same as that calculated with vcirc'
    assert numpy.fabs(potential.calcEscapecurve(lp,0.8)-lp.vesc(0.8)) < 10.**-16., 'Escape velocity calculated with calcEscapecurve not the same as that calculated with vcirc'
    return None        

def test_lindbladR():
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    assert numpy.fabs(lp.lindbladR(0.5,'corotation')-2.) < 10.**-10., 'Location of co-rotation resonance is wrong for LogarithmicHaloPotential'
    assert numpy.fabs(lp.omegac(lp.lindbladR(0.5,2))-2./(2.-numpy.sqrt(2.))*0.5) < 10.**-14., 'Location of m=2 resonance is wrong for LogarithmicHaloPotential'
    assert numpy.fabs(lp.omegac(lp.lindbladR(0.5,-2))+2./(-2.-numpy.sqrt(2.))*0.5) < 10.**-14., 'Location of m=-2 resonance is wrong for LogarithmicHaloPotential'
    #Also through general interface
    assert numpy.fabs(lp.omegac(potential.lindbladR(lp,0.5,-2))+2./(-2.-numpy.sqrt(2.))*0.5) < 10.**-14., 'Location of m=-2 resonance is wrong for LogarithmicHaloPotential'
    #Also for planar
    assert numpy.fabs(lp.omegac(lp.toPlanar().lindbladR(0.5,-2))+2./(-2.-numpy.sqrt(2.))*0.5) < 10.**-14., 'Location of m=-2 resonance is wrong for LogarithmicHaloPotential'
    #Test non-existent ones
    mp= potential.MiyamotoNagaiPotential(normalize=1.,a=0.3)
    assert mp.lindbladR(3.,2) is None, 'MiyamotoNagai w/ OmegaP=3 should not have a inner m=2 LindbladR'
    assert mp.lindbladR(6.,'corotation') is None, 'MiyamotoNagai w/ OmegaP=6 should not have a inner m=2 LindbladR'
    #Test error
    try:
        lp.lindbladR(0.5,'wrong resonance')
    except IOError:
        pass
    else:
        raise AssertionError("lindbladR w/ wrong m input should have raised IOError, but didn't")
    return None

def test_vterm():
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    assert numpy.fabs(lp.vterm(30.,deg=True)-0.5*(lp.omegac(0.5)-1.)) < 10.**-10., 'vterm for LogarithmicHaloPotential at l=30 is incorrect'
    assert numpy.fabs(lp.vterm(numpy.pi/3.,deg=False)-numpy.sqrt(3.)/2.*(lp.omegac(numpy.sqrt(3.)/2.)-1.)) < 10.**-10., 'vterm for LogarithmicHaloPotential at l=60 in rad is incorrect'
    #Also using general interface
    assert numpy.fabs(potential.vterm(lp,30.,deg=True)-0.5*(lp.omegac(0.5)-1.)) < 10.**-10., 'vterm for LogarithmicHaloPotential at l=30 is incorrect'
    assert numpy.fabs(potential.vterm(lp,numpy.pi/3.,deg=False)-numpy.sqrt(3.)/2.*(lp.omegac(numpy.sqrt(3.)/2.)-1.)) < 10.**-10., 'vterm for LogarithmicHaloPotential at l=60 in rad is incorrect'
    return None

def test_flattening():
    #Simple tests: LogarithmicHalo
    qs= [0.75,1.,1.25]
    for q in qs:
        lp= potential.LogarithmicHaloPotential(normalize=1.,q=q)
        assert (lp.flattening(1.,0.001)-q)**2. < 10.**-16., \
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (1.,0.001)" % q
        assert (lp.flattening(1.,0.1)-q)**2. < 10.**-16., \
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (1.,0.1)" % q
        assert (lp.flattening(0.5,0.001)-q)**2. < 10.**-16., \
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.001)" % q
        assert (lp.flattening(0.5,0.1)-q)**2. < 10.**-16., \
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.1)" % q
        #One test with the general interface
        assert (potential.flattening(lp,0.5,0.1)-q)**2. < 10.**-16., \
            "Flattening of LogarithmicHaloPotential w/ q= %f is not equal to q  at (R,z) = (0.5,0.1), through potential.flattening" % q
    #Check some spherical potentials
    kp= potential.KeplerPotential(normalize=1.)
    assert (kp.flattening(1.,0.02)-1.)**2. < 10.**-16., \
        "Flattening of KeplerPotential is not equal to 1 at (R,z) = (1.,0.02)"
    np= potential.NFWPotential(normalize=1.,a=5.)
    assert (np.flattening(1.,0.02)-1.)**2. < 10.**-16., \
        "Flattening of NFWPotential is not equal to 1 at (R,z) = (1.,0.02)"
    hp= potential.HernquistPotential(normalize=1.,a=5.)
    assert (hp.flattening(1.,0.02)-1.)**2. < 10.**-16., \
        "Flattening of HernquistPotential is not equal to 1 at (R,z) = (1.,0.02)"
    #Disk potentials should be oblate everywhere
    mp= potential.MiyamotoNagaiPotential(normalize=1.,a=0.5,b=0.05)
    assert mp.flattening(1.,0.1) <= 1., \
        "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (1.,0.1)"
    assert mp.flattening(1.,2.) <= 1., \
        "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (1.,2.)"
    assert mp.flattening(3.,3.) <= 1., \
        "Flattening of MiyamotoNagaiPotential w/ a=0.5, b=0.05 is > 1 at (R,z) = (3.,3.)"
    return None

def test_verticalfreq():
    #For spherical potentials, vertical freq should be equal to rotational freq
    lp= potential.LogarithmicHaloPotential(normalize=1.,q=1.)
    kp= potential.KeplerPotential(normalize=1.)
    np= potential.NFWPotential(normalize=1.)
    bp= potential.BurkertPotential(normalize=1.)
    rs= numpy.linspace(0.2,2.,21)
    for r in rs:
        assert numpy.fabs(lp.verticalfreq(r)-lp.omegac(r)) < 10.**-10., \
            'Verticalfreq for spherical potential does not equal rotational freq'
        assert numpy.fabs(kp.verticalfreq(r)-kp.omegac(r)) < 10.**-10., \
            'Verticalfreq for spherical potential does not equal rotational freq'
        #Through general interface
        assert numpy.fabs(potential.verticalfreq(np,r)-np.omegac(r)) < 10.**-10., \
            'Verticalfreq for spherical potential does not equal rotational freq'
        assert numpy.fabs(potential.verticalfreq([bp],r)-bp.omegac(r)) < 10.**-10., \
            'Verticalfreq for spherical potential does not equal rotational freq'
    #For Double-exponential disk potential, epi^2+vert^2-2*rot^2 =~ 0 (explicitly, because we use a Kepler potential)
    if True: #not _TRAVIS:
        dp= potential.DoubleExponentialDiskPotential(normalize=1.,hr=0.05,hz=0.01)
        assert numpy.fabs(dp.epifreq(1.)**2.+dp.verticalfreq(1.)**2.-2.*dp.omegac(1.)**2.) < 10.**-6., 'epi^2+vert^2-2*rot^2 !=~ 0 for dblexp potential, very far from center'
        #Closer to the center, this becomes the Poisson eqn.
        assert numpy.fabs(dp.epifreq(.125)**2.+dp.verticalfreq(.125)**2.-2.*dp.omegac(.125)**2.-4.*numpy.pi*dp.dens(0.125,0.))/4./numpy.pi/dp.dens(0.125,0.) < 10.**-3., 'epi^2+vert^2-2*rot^2 !=~ dens for dblexp potential'
    return None

def test_planar_nonaxi():
    dp= potential.DehnenBarPotential()
    try:
        potential.evaluateplanarPotentials(1.,dp)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError('evaluateplanarPotentials for non-axisymmetric potential w/o specifying phi did not raise PotentialError')
    try:
        potential.evaluateplanarRforces(1.,dp)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError('evaluateplanarRforces for non-axisymmetric potential w/o specifying phi did not raise PotentialError')
    try:
        potential.evaluateplanarphiforces(1.,dp)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError('evaluateplanarphiforces for non-axisymmetric potential w/o specifying phi did not raise PotentialError')
    try:
        potential.evaluateplanarR2derivs(1.,dp)
    except potential.PotentialError:
        pass
    else:
        raise AssertionError('evaluateplanarR2derivs for non-axisymmetric potential w/o specifying phi did not raise PotentialError')
    return None

def test_ExpDisk_special():
    #Test some special cases for the ExponentialDisk potentials
    #if _TRAVIS: return None
    #Test that array input works
    dp= potential.DoubleExponentialDiskPotential(normalize=1.)
    rs= numpy.linspace(0.1,2.11)
    zs= numpy.ones_like(rs)*0.1
    #Potential itself
    dpevals= numpy.array([dp(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential evaluation does not work as expected for array inputs'
    #Rforce
    dpevals= numpy.array([dp.Rforce(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp.Rforce(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential Rforce evaluation does not work as expected for array inputs'
    #zforce
    dpevals= numpy.array([dp.zforce(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp.zforce(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential zforce evaluation does not work as expected for array inputs'
    #R2deriv
    dpevals= numpy.array([dp.R2deriv(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp.R2deriv(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential R2deriv evaluation does not work as expected for array inputs'
    #z2deriv
    dpevals= numpy.array([dp.z2deriv(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp.z2deriv(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential z2deriv evaluation does not work as expected for array inputs'
    #Rzderiv
    dpevals= numpy.array([dp.Rzderiv(r,z) for (r,z) in zip(rs,zs)])
    assert numpy.all(numpy.fabs(dp.Rzderiv(rs,zs)-dpevals) < 10.**-10.), \
        'DoubleExppnentialDiskPotential Rzderiv evaluation does not work as expected for array inputs'
    #Check the PotentialError for z=/=0 evaluation of R2deriv of RazorThinDiskPotential
    rp= potential.RazorThinExponentialDiskPotential(normalize=1.)
    try: rp.R2deriv(1.,0.1)
    except potential.PotentialError: pass
    else: raise AssertionError("RazorThinExponentialDiskPotential's R2deriv did not raise AttributeError for z=/= 0 input")
    return None

def test_MovingObject_density():
    mp= mockMovingObjectPotential()
    #Just test that the density far away from the object is close to zero
    assert numpy.fabs(mp.dens(5.,0.)) < 10.**-8., 'Density far away from MovingObject is not close to zero'
    return None

# Test that MWPotential is what it's supposed to be
def test_MWPotential2014():
    pot= potential.MWPotential2014
    V0, R0= 220., 8.
    #Check the parameters of the bulge
    assert pot[0].rc == 1.9/R0, "MWPotential2014's bulge cut-off radius is incorrect"
    assert pot[0].alpha == 1.8, "MWPotential2014's bulge power-law exponent is incorrect"
    assert numpy.fabs(pot[0].Rforce(1.,0.)+0.05) < 10.**-14., "MWPotential2014's bulge amplitude is incorrect"
    #Check the parameters of the disk
    assert numpy.fabs(pot[1]._a-3./R0) < 10.**-14., "MWPotential2014's disk scale length is incorrect"
    assert numpy.fabs(pot[1]._b-0.28/R0) < 10.**-14., "MWPotential2014's disk scale heigth is incorrect"
    assert numpy.fabs(pot[1].Rforce(1.,0.)+0.60) < 10.**-14., "MWPotential2014's disk amplitude is incorrect"
    #Check the parameters of the halo
    assert numpy.fabs(pot[2].a-16./R0) < 10.**-14., "MWPotential2014's halo scale radius is incorrect"
    assert numpy.fabs(pot[2].Rforce(1.,0.)+0.35) < 10.**-14., "MWPotential2014's halo amplitude is incorrect"
    return None

# Test that the virial setup of NFW works
def test_NFW_virialsetup_wrtmeanmatter():
    from galpy.util import bovy_conversion
    H, Om, overdens, wrtcrit= 71., 0.32, 201., False
    ro, vo= 220., 8.
    conc, mvir= 12., 1.1
    np= potential.NFWPotential(conc=conc,mvir=mvir,vo=vo,ro=ro,
                               H=H,Om=Om,overdens=overdens,
                               wrtcrit=wrtcrit)
    assert numpy.fabs(conc-np.conc(vo,ro,H=H,Om=Om,overdens=overdens,
                                   wrtcrit=wrtcrit)) < 10.**-6., "NFWPotential virial setup's concentration does not work"
    assert numpy.fabs(mvir*100./bovy_conversion.mass_in_1010msol(vo,ro)\
                          -np.mvir(vo,ro,H=H,Om=Om,overdens=overdens,
                                   wrtcrit=wrtcrit)) < 10.**-6., "NFWPotential virial setup's virial mass does not work"
    return None

def test_NFW_virialsetup_wrtcrit():
    from galpy.util import bovy_conversion
    H, Om, overdens, wrtcrit= 71., 0.32, 201., True
    ro, vo= 220., 8.
    conc, mvir= 12., 1.1
    np= potential.NFWPotential(conc=conc,mvir=mvir,vo=vo,ro=ro,
                               H=H,Om=Om,overdens=overdens,
                               wrtcrit=wrtcrit)
    assert numpy.fabs(conc-np.conc(vo,ro,H=H,Om=Om,overdens=overdens,
                                   wrtcrit=wrtcrit)) < 10.**-6., "NFWPotential virial setup's concentration does not work"
    assert numpy.fabs(mvir*100./bovy_conversion.mass_in_1010msol(vo,ro)\
                          -np.mvir(vo,ro,H=H,Om=Om,overdens=overdens,
                                   wrtcrit=wrtcrit)) < 10.**-6., "NFWPotential virial setup's virial mass does not work"
    return None

def test_conc_attributeerror():
    pp= potential.PowerSphericalPotential(normalize=1.)
    #This potential doesn't have a scale, so we cannot calculate the concentration
    try: pp.conc(220.,8.)
    except AttributeError: pass
    else: raise AssertionError('conc function for potential w/o scale did not raise AttributeError')
    return None

def test_mvir_attributeerror():
    mp= potential.MiyamotoNagaiPotential(normalize=1.)
    #Don't think I will ever implement the virial radius for this
    try: mp.mvir(220.,8.)
    except AttributeError: pass
    else: raise AssertionError('mvir function for potential w/o rvir did not raise AttributeError')
    return None

def test_LinShuReductionFactor():
    #Test that the LinShuReductionFactor is implemented correctly, by comparing to figure 1 in Lin & Shu (1966)
    from galpy.potential import LinShuReductionFactor, \
        LogarithmicHaloPotential, omegac, epifreq
    lp= LogarithmicHaloPotential(normalize=1.) #work in flat rotation curve
    #nu^2 = 0.2, x=4 for m=2,sigmar=0.1 
    # w/ nu = m(OmegaP-omegac)/epifreq, x=sr^2*k^2/epifreq^2
    R,m,sr = 0.9,2.,0.1
    tepi, tomegac= epifreq(lp,R), omegac(lp,R)
    OmegaP= tepi*numpy.sqrt(0.2)/m+tomegac #leads to nu^2 = 0.2
    k= numpy.sqrt(4.)*tepi/sr
    assert numpy.fabs(LinShuReductionFactor(lp,R,sr,m=m,k=k,OmegaP=OmegaP)-0.18) < 0.01, 'LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)'
    #nu^2 = 0.8, x=10
    OmegaP= tepi*numpy.sqrt(0.8)/m+tomegac #leads to nu^2 = 0.8
    k= numpy.sqrt(10.)*tepi/sr
    assert numpy.fabs(LinShuReductionFactor(lp,R,sr,m=m,k=k,OmegaP=OmegaP)-0.04) < 0.01, 'LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)'   
    #Similar test, but using a nonaxiPot= input
    from galpy.potential import SteadyLogSpiralPotential
    sp= SteadyLogSpiralPotential(m=2.,omegas=OmegaP,alpha=k*R)
    assert numpy.fabs(LinShuReductionFactor(lp,R,sr,nonaxiPot=sp)-0.04) < 0.01, 'LinShuReductionFactor does not agree w/ Figure 1 from Lin & Shu (1966)'   
    #Test exception
    try:
        LinShuReductionFactor(lp,R,sr)
    except IOError: pass
    else: raise AssertionError("LinShuReductionFactor w/o nonaxiPot set or k=,m=,OmegaP= set did not raise IOError")
    return None

def test_plotting():
    import tempfile
    #Some tests of the plotting routines, to make sure they don't fail
    kp= potential.KeplerPotential(normalize=1.)
    #Plot the rotation curve
    kp.plotRotcurve()
    kp.toPlanar().plotRotcurve() #through planar interface
    kp.plotRotcurve(Rrange=[0.01,10.],
                    grid=101,
                    savefilename=None)
    potential.plotRotcurve([kp])
    potential.plotRotcurve([kp],Rrange=[0.01,10.],
                           grid=101,
                           savefilename=None)
    #Also while saving the result
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        kp.plotRotcurve(Rrange=[0.01,10.],
                        grid=101,
                        savefilename=tmp_savefilename)
        #Then plot using the saved file
        kp.plotRotcurve(Rrange=[0.01,10.],
                        grid=101,
                        savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    #Plot the escape-velocity curve
    kp.plotEscapecurve()
    kp.toPlanar().plotEscapecurve() #Through planar interface
    kp.plotEscapecurve(Rrange=[0.01,10.],
                       grid=101,
                       savefilename=None)
    potential.plotEscapecurve([kp])
    potential.plotEscapecurve([kp],Rrange=[0.01,10.],
                              grid=101,
                              savefilename=None)
    #Also while saving the result
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        kp.plotEscapecurve(Rrange=[0.01,10.],
                           grid=101,
                           savefilename=tmp_savefilename)
        #Then plot using the saved file
        kp.plotEscapecurve(Rrange=[0.01,10.],
                           grid=101,
                           savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    #Plot the potential itself
    kp.plot()
    kp.plot(t=1.,rmin=0.01,rmax=1.8,nrs=11,zmin=-0.55,zmax=0.55,nzs=11, 
            effective=False,Lz=None, 
            xrange=[0.01,1.8],yrange=[-0.55,0.55],justcontours=True,
            ncontours=11,savefilename=None)
    #Also while saving the result
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        kp.plot(t=1.,rmin=0.01,rmax=1.8,nrs=11,zmin=-0.55,zmax=0.55,nzs=11, 
                effective=False,Lz=None, 
                xrange=[0.01,1.8],yrange=[-0.55,0.55], 
                ncontours=11,savefilename=tmp_savefilename)
        #Then plot using the saved file
        kp.plot(t=1.,rmin=0.01,rmax=1.8,nrs=11,zmin=-0.55,zmax=0.55,nzs=11, 
                effective=False,Lz=None, 
                xrange=[0.01,1.8],yrange=[-0.55,0.55], 
                ncontours=11,savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    potential.plotPotentials([kp])
    #Also while saving the result
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        potential.plotPotentials([kp],
                                 rmin=0.01,rmax=1.8,nrs=11,
                                 zmin=-0.55,zmax=0.55,nzs=11, 
                                 justcontours=True,
                                 ncontours=11,savefilename=tmp_savefilename)
        #Then plot using the saved file
        potential.plotPotentials([kp],
                                 rmin=0.01,rmax=1.8,nrs=11,
                                 zmin=-0.55,zmax=0.55,nzs=11, 
                                 ncontours=11,savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    #Plot the effective potential
    kp.plot()
    kp.plot(effective=True,Lz=1.)
    try:
        kp.plot(effective=True,Lz=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Potential.plot with effective=True, but Lz=None did not return a RuntimeError")
    #Plot the density of a LogarithmicHaloPotential
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    lp.plotDensity()
    lp.plotDensity(rmin=0.05,rmax=1.8,nrs=11,zmin=-0.55,zmax=0.55,nzs=11, 
                   aspect=1.,log=True,justcontours=True,
                   ncontours=11,savefilename=None)
    #Also while saving the result
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        lp.plotDensity(savefilename=tmp_savefilename)
        #Then plot using the saved file
        lp.plotDensity(savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    potential.plotDensities([lp])
    potential.plotDensities([lp],
                            rmin=0.05,rmax=1.8,nrs=11,
                            zmin=-0.55,zmax=0.55,nzs=11, 
                            aspect=1.,log=True,
                            justcontours=True,
                            ncontours=11,savefilename=None)
    #Plot the potential itself for a 2D potential
    kp.toPlanar().plot()
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        kp.toPlanar().plot(Rrange=[0.01,1.8],grid=11,
                           savefilename=tmp_savefilename)
        #Then plot using the saved file
        kp.toPlanar().plot(Rrange=[0.01,1.8],grid=11,
                           savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    dp= potential.DehnenBarPotential()
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        dp.plot(xrange=[0.01,1.8],yrange=[0.01,1.8],gridx=11,gridy=11,
                ncontours=11,savefilename=tmp_savefilename)
        #Then plot using the saved file
        dp.plot(xrange=[0.01,1.8],yrange=[0.01,1.8],gridx=11,gridy=11,
                ncontours=11,savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    potential.plotplanarPotentials([dp],gridx=11,gridy=11)
    #Tests of linearPotential plotting
    lip= potential.RZToverticalPotential(potential.MiyamotoNagaiPotential(normalize=1.),1.)
    lip.plot()
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        lip.plot(t=0.,min=-15.,max=15,ns=21,savefilename=tmp_savefilename)
        #Then plot using the saved file
        lip.plot(t=0.,min=-15.,max=15,ns=21,savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    savefile, tmp_savefilename= tempfile.mkstemp()
    try:
        os.close(savefile) #Easier this way 
        os.remove(tmp_savefilename)
        #First save
        potential.plotlinearPotentials(lip,t=0.,min=-15.,max=15,ns=21,
                                       savefilename=tmp_savefilename)
        #Then plot using the saved file
        potential.plotlinearPotentials(lip,t=0.,min=-15.,max=15,ns=21,
                                       savefilename=tmp_savefilename)
    finally:
        os.remove(tmp_savefilename)
    return None

#Classes for testing Integer TwoSphericalPotential and for testing special
# cases of some other potentials
from galpy.potential import TwoPowerSphericalPotential, \
    MiyamotoNagaiPotential, PowerSphericalPotential, interpRZPotential, \
    MWPotential, FlattenedPowerPotential
class mockTwoPowerIntegerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self,amp=1.,a=5.,alpha=2.,beta=5.)
        return None
class specialTwoPowerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self,amp=1.,a=5.,alpha=1.5,beta=3.)
        return None
class HernquistTwoPowerIntegerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self,amp=1.,a=5.,alpha=1.,beta=4.)
        return None
class JaffeTwoPowerIntegerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self,amp=1.,a=5.,alpha=2.,beta=4.)
        return None
class NFWTwoPowerIntegerSphericalPotential(TwoPowerSphericalPotential):
    def __init__(self):
        TwoPowerSphericalPotential.__init__(self,amp=1.,a=5.,alpha=1.,beta=3.)
        return None
class specialPowerSphericalPotential(PowerSphericalPotential):
    def __init__(self):
        PowerSphericalPotential.__init__(self,amp=1.,alpha=2.)
        return None
class specialMiyamotoNagaiPotential(MiyamotoNagaiPotential):
    def __init__(self):
        MiyamotoNagaiPotential.__init__(self,amp=1.,a=0.,b=0.1)
        return None
class specialFlattenedPowerPotential(FlattenedPowerPotential):
    def __init__(self):
        FlattenedPowerPotential.__init__(self,alpha=0.)
        return None
class mockInterpRZPotential(interpRZPotential):
    def __init__(self):
        interpRZPotential.__init__(self,RZPot=MWPotential,
                                   rgrid=(0.01,2.1,101),zgrid=(0.,0.26,101),
                                   logR=True,
                                   interpPot=True,interpRforce=True,
                                   interpzforce=True,interpDens=True)
class mockSnapshotRZPotential(potential.SnapshotRZPotential):
    def __init__(self):
        # Test w/ equivalent of KeplerPotential: one mass
        kp= potential.KeplerPotential(amp=1.)
        s= pynbody.new(star=1)
        s['mass']= 1./numpy.fabs(kp.Rforce(1.,0.)) #forces vc(1,0)=1
        s['eps']= 0.
        potential.SnapshotRZPotential.__init__(self,s)
class mockInterpSnapshotRZPotential(potential.InterpSnapshotRZPotential):
    def __init__(self):
        # Test w/ equivalent of KeplerPotential: one mass
        kp= potential.KeplerPotential(amp=1.)
        s= pynbody.new(star=1)
        s['mass']= 1./numpy.fabs(kp.Rforce(1.,0.)) #forces vc(1,0)=1
        s['eps']= 0.
        potential.InterpSnapshotRZPotential.__init__(self,s,
                                                   rgrid=(0.01,2.,101),
                                                   zgrid=(0.,0.3,101),
                                                   logR=False,
                                                   interpPot=True,
                                                   zsym=True)
# Some special cases of 2D, non-axisymmetric potentials, to make sure they
# are covered; need 3 to capture all of the transient behavior
from galpy.potential import CosmphiDiskPotential, DehnenBarPotential, \
    EllipticalDiskPotential, SteadyLogSpiralPotential, \
    TransientLogSpiralPotential
class mockCosmphiDiskPotentialT1(CosmphiDiskPotential):
    def __init__(self):
        CosmphiDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                      p=1.,phio=0.01,m=1., 
                                      tform=1.,tsteady=2.,
                                      cp=0.05,sp=0.05)
class mockCosmphiDiskPotentialTm1(CosmphiDiskPotential):
    def __init__(self):
        CosmphiDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                      p=1.,phio=0.01,m=1., 
                                      tform=-1.,tsteady=None,
                                      cp=-0.05,sp=0.05)
class mockCosmphiDiskPotentialTm5(CosmphiDiskPotential):
    def __init__(self):
        CosmphiDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                      p=1.,phio=0.01,m=1., 
                                      tform=-5.,tsteady=-1.,
                                      cp=-0.05,sp=0.05)
class mockDehnenBarPotentialT1(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(self,omegab=1.9,rb=0.4,
                                    barphi=25.*numpy.pi/180.,beta=0.,
                                    tform=1.,tsteady=1.,
                                    alpha=0.01,Af=0.04)
class mockDehnenBarPotentialTm1(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(self,omegab=1.9,rb=0.6,
                                    barphi=25.*numpy.pi/180.,beta=0.,
                                    tform=-1.,tsteady=2.,
                                    alpha=0.01,Af=0.04)
class mockDehnenBarPotentialTm5(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(self,omegab=1.9,rb=0.4,
                                    barphi=25.*numpy.pi/180.,beta=0.,
                                    tform=-5.,tsteady=4.,
                                    alpha=0.01,Af=0.04)
class mockEllipticalDiskPotentialT1(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                         p=1.,twophio=0.02, 
                                         tform=1.,tsteady=2.,
                                         cp=0.05,sp=0.05)
class mockEllipticalDiskPotentialTm1(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                         p=1.,twophio=0.02,
                                         tform=-1.,tsteady=None,
                                         cp=-0.05,sp=0.05)
class mockEllipticalDiskPotentialTm5(EllipticalDiskPotential):
    def __init__(self):
        EllipticalDiskPotential.__init__(self,amp=1.,phib=25.*numpy.pi/180.,
                                         p=1.,twophio=0.02,
                                         tform=-5.,tsteady=-1.,
                                         cp=-0.05,sp=0.05)
class mockSteadyLogSpiralPotentialT1(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(self,amp=1.,omegas=0.65,A=-0.035, 
                                          m=2,gamma=numpy.pi/4.,
                                          p=-0.3, 
                                          tform=1.,tsteady=2.)
class mockSteadyLogSpiralPotentialTm1(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(self,amp=1.,omegas=0.65,A=-0.035, 
                                          m=2,gamma=numpy.pi/4.,
                                          p=-0.3, 
                                          tform=-1.,tsteady=None)
class mockSteadyLogSpiralPotentialTm5(SteadyLogSpiralPotential):
    def __init__(self):
        SteadyLogSpiralPotential.__init__(self,amp=1.,omegas=0.65,A=-0.035, 
                                          m=2,gamma=numpy.pi/4.,
                                          p=-0.3, 
                                          tform=-1.,tsteady=-5.)
class mockTransientLogSpiralPotential(TransientLogSpiralPotential):
    def __init__(self):
        TransientLogSpiralPotential.__init__(self,amp=1.,omegas=0.65,A=-0.035, 
                                             m=2,gamma=numpy.pi/4.,
                                             p=-0.3)
#Class to test potentials given as lists, st we can use their methods as class.
from galpy.potential import Potential, \
    evaluatePotentials, evaluateRforces, evaluatezforces, evaluatephiforces, \
    evaluateR2derivs, evaluatez2derivs, evaluateRzderivs, \
    evaluateDensities
class testMWPotential(Potential):
    """Initialize with potential in natural units"""
    def __init__(self,potlist=MWPotential):
        self._potlist= potlist
        Potential.__init__(self,amp=1.)
        return None
    def _evaluate(self,R,z,phi=0,t=0,dR=0,dphi=0):
        return evaluatePotentials(R,z,self._potlist,phi=phi,t=t,
                                  dR=dR,dphi=dphi)
    def _Rforce(self,R,z,phi=0.,t=0.):
        return evaluateRforces(R,z,self._potlist,phi=phi,t=t)
    def _phiforce(self,R,z,phi=0.,t=0.):
        return evaluatephiforces(R,z,self._potlist,phi=phi,t=t)
    def _zforce(self,R,z,phi=0.,t=0.):
        return evaluatezforces(R,z,self._potlist,phi=phi,t=t)
    def _R2deriv(self,R,z,phi=0.,t=0.):
        return evaluateR2derivs(R,z,self._potlist,phi=phi,t=t)
    def _z2deriv(self,R,z,phi=0.,t=0.):
        return evaluatez2derivs(R,z,self._potlist,phi=phi,t=t)
    def _Rzderiv(self,R,z,phi=0.,t=0.):
        return evaluateRzderivs(R,z,self._potlist,phi=phi,t=t)
    def _dens(self,R,z,phi=0.,t=0.,forcepoisson=False):
        return evaluateDensities(R,z,self._potlist,phi=phi,t=t,
                                 forcepoisson=forcepoisson)
    def vcirc(self,R):
        return potential.vcirc(self._potlist,R)
    def normalize(self,norm,t=0.):
        self._amp= norm
    def OmegaP(self):
        return 1.
#Class to test lists of planarPotentials
from galpy.potential import planarPotential, \
    evaluateplanarPotentials, evaluateplanarRforces, evaluateplanarphiforces, \
    evaluateplanarR2derivs
class testplanarMWPotential(planarPotential):
    """Initialize with potential in natural units"""
    def __init__(self,potlist=MWPotential):
        self._potlist= [p.toPlanar() for p in potlist if isinstance(p,Potential)]
        self._potlist.extend([p for p in potlist if isinstance(p,planarPotential)])
        planarPotential.__init__(self,amp=1.)
        self.isNonAxi= True-numpy.prod([True-p.isNonAxi for p in self._potlist])
        return None
    def _evaluate(self,R,phi=0,t=0,dR=0,dphi=0):
        return evaluateplanarPotentials(R,self._potlist,phi=phi,t=t)
    def _Rforce(self,R,phi=0.,t=0.):
        return evaluateplanarRforces(R,self._potlist,phi=phi,t=t)
    def _phiforce(self,R,phi=0.,t=0.):
        return evaluateplanarphiforces(R,self._potlist,phi=phi,t=t)
    def _R2deriv(self,R,phi=0.,t=0.):
        return evaluateplanarR2derivs(R,self._potlist,phi=phi,t=t)
    def _phi2deriv(self,R,phi=0.,t=0.):
        return evaluateplanarPotentials(R,self._potlist,phi=phi,t=t,dphi=2)
    def _Rphideriv(self,R,phi=0.,t=0.):
        return evaluateplanarPotentials(R,self._potlist,phi=phi,t=t,dR=1,
                                        dphi=1)
    def vcirc(self,R):
        return potential.vcirc(self._potlist,R)
    def normalize(self,norm,t=0.):
        self._amp= norm
    def OmegaP(self):
        return 1.

class mockFlatEllipticalDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.EllipticalDiskPotential(phib=numpy.pi/2.,p=0.,tform=None,tsteady=None,twophio=14./220.)])
    def OmegaP(self):
        return 0.
class mockSlowFlatEllipticalDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.EllipticalDiskPotential(phib=numpy.pi/2.,p=0.,twophio=14./220.,tform=1.,tsteady=250.)])
    def OmegaP(self):
        return 0.
class mockFlatLopsidedDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.LopsidedDiskPotential(phib=numpy.pi/2.,p=0.,tform=None,tsteady=None,phio=10./220.)])
    def OmegaP(self):
        return 0.
class mockSlowFlatLopsidedDiskPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.LopsidedDiskPotential(phib=numpy.pi/2.,p=0.,tform=1.,tsteady=250.,phio=10./220.)])
    def OmegaP(self):
        return 0.
class mockFlatDehnenBarPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.DehnenBarPotential()])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockSlowFlatDehnenBarPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.DehnenBarPotential(tform=1.,tsteady=250.,rolr=2.5)])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockFlatSteadyLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.SteadyLogSpiralPotential()])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockSlowFlatSteadyLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.SteadyLogSpiralPotential(tform=.1,tsteady=25.)])
    def OmegaP(self):
        return self._potlist[1].OmegaP()
class mockFlatTransientLogSpiralPotential(testplanarMWPotential):
    def __init__(self):
        testplanarMWPotential.__init__(self,
                                       potlist=[potential.LogarithmicHaloPotential(normalize=1.),
                                                potential.TransientLogSpiralPotential(to=-10.)]) #this way, it's basically a steady spiral
    def OmegaP(self):
        return self._potlist[1].OmegaP()

#Class to test lists of linearPotentials
from galpy.potential import linearPotential, \
    evaluatelinearPotentials, evaluatelinearForces, \
    RZToverticalPotential
class testlinearMWPotential(linearPotential):
    """Initialize with potential in natural units"""
    def __init__(self,potlist=MWPotential):
        self._potlist= RZToverticalPotential(potlist,1.)
        linearPotential.__init__(self,amp=1.)
        return None
    def _evaluate(self,R,phi=0,t=0,dR=0,dphi=0):
        return evaluatelinearPotentials(R,self._potlist,t=t)
    def _force(self,R,t=0.):
        return evaluatelinearForces(R,self._potlist,t=t)
    def normalize(self,norm,t=0.):
        self._amp= norm

class mockCombLinearPotential(testlinearMWPotential):
    def __init__(self):
        testlinearMWPotential.__init__(self,
                                       potlist=[potential.MWPotential[0],
                                                potential.MWPotential[1].toVertical(1.),
                                                potential.MWPotential[2].toVertical(1.)])

class mockSimpleLinearPotential(testlinearMWPotential):
    def __init__(self):
        testlinearMWPotential.__init__(self,
                                       potlist=potential.MiyamotoNagaiPotential(normalize=1.).toVertical(1.))

class mockMovingObjectPotential(testMWPotential):
    def __init__(self,rc=0.75,maxt=1.,nt=50):
        from galpy.orbit import Orbit
        self._rc= rc
        o1= Orbit([self._rc,0.,1.,0.,0.,0.])
        o2= Orbit([self._rc,0.,1.,0.,0.,numpy.pi])
        lp= potential.LogarithmicHaloPotential(normalize=1.)
        times= numpy.linspace(0.,maxt,nt)
        o1.integrate(times,lp,method='dopr54_c')
        o2.integrate(times,lp,method='dopr54_c')
        self._o1p= potential.MovingObjectPotential(o1)
        self._o2p= potential.MovingObjectPotential(o2)
        testMWPotential.__init__(self,[self._o1p,self._o2p])
        self.isNonAxi= True
        return None
    def phi2deriv(self,R,z,phi=0.,t=0.):
        raise AttributeError
    def OmegaP(self):
        return 1./self._rc
from galpy.potential_src.ForceSoftening import PlummerSoftening
class mockMovingObjectExplSoftPotential(testMWPotential):
    def __init__(self,rc=0.75,maxt=1.,nt=50):
        from galpy.orbit import Orbit
        self._rc= rc
        o1= Orbit([self._rc,0.,1.,0.,0.,0.])
        o2= Orbit([self._rc,0.,1.,0.,0.,numpy.pi])
        lp= potential.LogarithmicHaloPotential(normalize=1.)
        times= numpy.linspace(0.,maxt,nt)
        o1.integrate(times,lp,method='dopr54_c')
        o2.integrate(times,lp,method='dopr54_c')
        self._o1p= potential.MovingObjectPotential(o1,
                                                   softening=PlummerSoftening(softening_length=0.05))
        self._o2p= potential.MovingObjectPotential(o2,
                                                   softening=PlummerSoftening(softening_length=0.05))
        testMWPotential.__init__(self,[self._o1p,self._o2p])
        self.isNonAxi= True
        return None
class mockMovingObjectLongIntPotential(mockMovingObjectPotential):
    def __init__(self,rc=0.75):
        mockMovingObjectPotential.__init__(self,rc=rc,maxt=28.,nt=1001)
        return None
