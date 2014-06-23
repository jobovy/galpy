############################TESTS ON POTENTIALS################################
import sys
import numpy
import os
_TRAVIS= bool(os.getenv('TRAVIS'))

#Test whether the normalization of the potential works
def test_normalize_potential():
    from galpy import potential
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
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    if _TRAVIS: #travis CI
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
    from galpy import potential
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
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
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
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    if _TRAVIS: #travis CI
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
    from galpy import potential
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
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
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
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    if _TRAVIS: #travis CI
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
                        raise NotImplementedError("Rphideriv for Potentials is not implemented")
                    if tRphideriv**2. < 10.**ttol:
                        assert mRforcederivphi**2. < 10.**ttol, \
                            "Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the %s radial force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tRphideriv-mRforcederivphi), numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv))
                    else:
                        assert (tRphideriv-mRforcederivphi)**2./tRphideriv**2. < 10.**ttol, \
"Calculation of the mixed radial azimuthal derivative of the potential as the azimuthal derivative of the %s radial force fails at (R,phi) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],phis[jj],numpy.fabs(tRphideriv-mRforcederivphi), numpy.fabs((tRphideriv-mRforcederivphi)/tRphideriv))

#Test whether the Poisson equation is satisfied if _dens and the relevant second derivatives are implemented
def test_poisson_potential():
    from galpy import potential
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
    pots.append('testMWPotential')
    pots.append('testplanarMWPotential')
    rmpots= ['Potential','MWPotential','MovingObjectPotential',
             'interpRZPotential', 'linearPotential', 'planarAxiPotential',
             'planarPotential', 'verticalPotential','PotentialError']
    if _TRAVIS: #travis CI
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
    from galpy import potential
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
        try:
            if isinstance(tp,potential.planarPotential): 
                tp.R2deriv(1.2)
            else:
                tp.R2deriv(1.2,0.1)
        except PotentialError:
            hasR2= False
        if hasR2:
            if isinstance(tp,potential.planarPotential): 
                tevaldr2= tp(1.2,phi=0.1,dR=2)
                tr2deriv= tp.R2deriv(1.2,phi=0.1)
            else:
                tevaldr2= tp(1.2,0.1,phi=0.1,dR=2)
                tr2deriv= tp.R2deriv(1.2,0.1,phi=0.1)
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
        if hasattr(tp,'_phi2deriv'):
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

# Check that toVertical and toPlanar work
def test_toVertical_toPlanar():
    from galpy import potential
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

# Sanity check the derivative of the rotation curve and the frequencies in the plane
def test_dvcircdR_omegac_epifreq_rl_vesc():
    from galpy import potential
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
    #Epicycle frequency, flat rotation curve
    assert (lp.epifreq(1.)-numpy.sqrt(2.)*lp.omegac(1.))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=1"
    assert (lp.epifreq(0.5)-numpy.sqrt(2.)*lp.omegac(0.5))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=0.5"
    assert (lp.epifreq(2.0)-numpy.sqrt(2.)*lp.omegac(2.0))**2. < 10.**-16., \
        "LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=2"
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
    #Escape velocity of Kepler potential
    assert (kp.vesc(1.)**2.-2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=1"
    assert (kp.vesc(0.5)**2.-2.*kp.vcirc(0.5)**2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=0.5"
    assert (kp.vesc(2.)**2.-2.*kp.vcirc(2.)**2.)**2. < 10.**-16., \
        "KeplerPotential's escape velocity is wrong at R=2"
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
    from galpy import potential
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
    assert numpy.fabs(potential.calcEscapecurve(lp,0.8)-lp.vesc(0.8)) < 10.**-16., 'Escape velocity calculated with calcRotcurve not the same as that calculated with vcirc'
    return None        

def test_flattening():
    #Simple tests: LogarithmicHalo
    from galpy import potential
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

def test_plotting():
    import tempfile
    from galpy import potential
    #Some tests of the plotting routines, to make sure they don't fail
    kp= potential.KeplerPotential(normalize=1.)
    #Plot the rotation curve
    kp.plotRotcurve()
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
            xrange=[0.01,1.8],yrange=[-0.55,0.55], 
            ncontours=11,savefilename=None)
    potential.plotPotentials([kp])
    potential.plotPotentials([kp],
                             rmin=0.01,rmax=1.8,nrs=11,
                             zmin=-0.55,zmax=0.55,nzs=11, 
                             ncontours=11,savefilename=None)
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
                   aspect=1.,log=True,
                   ncontours=11,savefilename=None)
    potential.plotDensities([lp])
    potential.plotDensities([lp],
                            rmin=0.05,rmax=1.8,nrs=11,
                            zmin=-0.55,zmax=0.55,nzs=11, 
                            aspect=1.,log=True,
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
    return None

#Classes for testing Integer TwoSphericalPotential and for testing special
# cases of some other potentials
from galpy.potential import TwoPowerSphericalPotential, \
    MiyamotoNagaiPotential, PowerSphericalPotential, interpRZPotential, \
    MWPotential
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
class mockInterpRZPotential(interpRZPotential):
    def __init__(self):
        interpRZPotential.__init__(self,RZPot=MWPotential,
                                   rgrid=(0.01,2.1,101),zgrid=(0.,0.26,101),
                                   logR=True,
                                   interpPot=True,interpRforce=True,
                                   interpzforce=True,interpDens=True)
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
                                    tform=1.,tsteady=2.,
                                    alpha=0.01,Af=0.04)
class mockDehnenBarPotentialTm1(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(self,omegab=1.9,rb=0.6,
                                    barphi=25.*numpy.pi/180.,beta=0.,
                                    tform=-1.,tsteady=1.,
                                    alpha=0.01,Af=0.04)
class mockDehnenBarPotentialTm5(DehnenBarPotential):
    def __init__(self):
        DehnenBarPotential.__init__(self,omegab=1.9,rb=0.4,
                                    barphi=25.*numpy.pi/180.,beta=0.,
                                    tform=-5.,tsteady=-1.,
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
    def __init__(self,potlist=MWPotential):
        self._potlist= potlist
        Potential.__init__(self,amp=1.)
        return None
    def _evaluate(self,R,z,phi=0,t=0,dR=0,dphi=0):
        return evaluatePotentials(R,z,self._potlist,phi=phi,t=t)
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
    def normalize(self,norm,t=0.):
        self._amp= norm
    def OmegaP(self):
        return 1.
#Class to test lists of planarPotentials
from galpy.potential import planarPotential, \
    evaluateplanarPotentials, evaluateplanarRforces, evaluateplanarphiforces, \
    evaluateplanarR2derivs
class testplanarMWPotential(planarPotential):
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
    def normalize(self,norm,t=0.):
        self._amp= norm
    def OmegaP(self):
        return 1.
