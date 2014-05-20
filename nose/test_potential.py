############################TESTS ON POTENTIALS################################
import sys
import numpy
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
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
        try:
            assert((tp.Rforce(1.,0.)+1.)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential fails" % p)
        try:
            assert((tp.vcirc(1.)**2.-1.)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential fails" % p)
        tp.normalize(.5)
        try:
            assert((tp.Rforce(1.,0.)+.5)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential fails" % p)
        try:
            assert((tp.vcirc(1.)**2.-0.5)**2. < 10.**-16.)
        except AssertionError:
            raise AssertionError("Normalization of %s potential fails" % p)

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
                    mpotderivR= (tp(Rs[ii])-tp(Rs[ii]+dr))/dr
                    tRforce= tp.force(Rs[ii])
                else:
                    mpotderivR= (tp(Rs[ii],Zs[jj])-tp(Rs[ii]+dr,Zs[jj]))/dr
                    tRforce= tp.Rforce(Rs[ii],Zs[jj])
                try:
                    if tRforce**2. < 10.**ttol:
                        assert(mpotderivR**2. < 10.**ttol)
                    else:
                        assert((tRforce-mpotderivR)**2./tRforce**2. < 10.**ttol)
                except AssertionError:
                    raise AssertionError("Calculation of the Radial force as the Radial derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRforce-mpotderivR), numpy.fabs((tRforce-mpotderivR)/tRforce)))
        #Azimuthal force, if it exists
        if isinstance(tp,potential.linearPotential): continue
        for ii in range(len(Rs)):
            for jj in range(len(phis)):
                dphi= 10.**-8.
                newphi= phis[jj]+dphi
                dphi= newphi-phis[jj] #Representable number
                if isinstance(tp,potential.planarPotential):
                    mpotderivphi= (tp(Rs[ii],phi=phis[jj])-tp(Rs[ii],phi=phis[jj]+dphi))/dphi
                    tphiforce= tp.phiforce(Rs[ii],phi=phis[jj])
                else:
                    mpotderivphi= (tp(Rs[ii],0.05,phi=phis[jj])-tp(Rs[ii],0.05,phi=phis[jj]+dphi))/dphi
                    tphiforce= tp.phiforce(Rs[ii],0.05,phi=phis[jj])
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
                tzforce= tp.zforce(Rs[ii],Zs[jj])
                try:
                    if tzforce**2. < 10.**ttol:
                        assert(mpotderivz**2. < 10.**ttol)
                    else:
                        assert((tzforce-mpotderivz)**2./tzforce**2. < 10.**ttol)
                except AssertionError:
                    raise AssertionError("Calculation of the vertical force as the vertical derivative of the %s potential fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(mpotderivz),numpy.fabs((tzforce-mpotderivz)/tzforce)))

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
                    else:
                        mRforcederivR= (tp.Rforce(Rs[ii],Zs[jj])-tp.Rforce(Rs[ii]+dr,Zs[jj]))/dr
                        tR2deriv= tp.R2deriv(Rs[ii],Zs[jj])
                    try:
                        if tR2deriv**2. < 10.**ttol:
                            assert(mRforcederivR**2. < 10.**ttol)
                        else:
                            assert((tR2deriv-mRforcederivR)**2./tR2deriv**2. < 10.**ttol)
                    except AssertionError:
                        raise AssertionError("Calculation of the second Radial derivative of the potential as the Radial derivative of the %s Radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tR2deriv-mRforcederivR), numpy.fabs((tR2deriv-mRforcederivR)/tR2deriv)))
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
                    tz2deriv= tp.z2deriv(Rs[ii],Zs[jj])
                    try:
                        if tz2deriv**2. < 10.**ttol:
                            assert(mzforcederivz**2. < 10.**ttol)
                        else:
                            assert((tz2deriv-mzforcederivz)**2./tz2deriv**2. < 10.**ttol)
                    except AssertionError:
                        raise AssertionError("Calculation of the second vertical derivative of the potential as the vertical derivative of the %s vertical force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tz2deriv-mzforcederivz), numpy.fabs((tz2deriv-mzforcederivz)/tz2deriv)))
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
                    tRzderiv= tp.Rzderiv(Rs[ii],Zs[jj])
                    try:
                        if tRzderiv**2. < 10.**ttol:
                            assert(mRforcederivz**2. < 10.**ttol)
                        else:
                            assert((tRzderiv-mRforcederivz)**2./tRzderiv**2. < 10.**ttol)
                    except AssertionError:
                        raise AssertionError("Calculation of the mixed radial vertical derivative of the potential as the vertical derivative of the %s radial force fails at (R,Z) = (%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],numpy.fabs(tRzderiv-mRforcederivz), numpy.fabs((tRzderiv-mRforcederivz)/tRzderiv)))

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
                    tdens= tp.dens(Rs[ii],Zs[jj],phi=phis[kk],
                                   forcepoisson=False)
                    try:
                        if tdens**2. < 10.**ttol:
                            assert(tpoissondens**2. < 10.**ttol)
                        else:
                            assert((tpoissondens-tdens)**2./tdens**2. < 10.**ttol)
                    except AssertionError:
                        raise AssertionError("Poisson equation relation between the derivatives of the potential and the implemented density is not satisfied for the %s potential at (R,Z,phi) = (%.3f,%.3f,%.3f); diff = %e, rel. diff = %e" % (p,Rs[ii],Zs[jj],phis[kk],numpy.fabs(tdens-tpoissondens), numpy.fabs((tdens-tpoissondens)/tdens)))

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
            try:
                if tevaldr**2. < 10.**ttol:
                    assert(trforce**2. < 10.**ttol)
                else:
                    assert((tevaldr+trforce)**2./tevaldr**2. < 10.**ttol)
            except AssertionError:
                raise AssertionError("Calculation of radial derivative through _evaluate and Rforce inconsistent for the %s potential" % p)
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
                try:
                    if tevaldr2**2. < 10.**ttol:
                        assert(tr2deriv*2. < 10.**ttol)
                    else:
                        assert((tevaldr2-tr2deriv)**2./tevaldr2**2. < 10.**ttol)
                except AssertionError:
                    raise AssertionError("Calculation of 2nd radial derivative through _evaluate and R2deriv inconsistent for the %s potential" % p)
        #1st phi
        if isinstance(tp,potential.planarPotential): 
            tevaldphi= tp(1.2,phi=0.1,dphi=1)
            tphiforce= tp.phiforce(1.2,phi=0.1)
        else:
            tevaldphi= tp(1.2,0.1,phi=0.1,dphi=1)
            tphiforce= tp.phiforce(1.2,0.1,phi=0.1)
        if not tevaldphi is None:
            try:
                if tevaldphi**2. < 10.**ttol:
                    assert(tphiforce**2. < 10.**ttol)
                else:
                    assert((tevaldphi+tphiforce)**2./tevaldphi**2. < 10.**ttol)
            except AssertionError:
                raise AssertionError("Calculation of azimuthal derivative through _evaluate and phiforce inconsistent for the %s potential" % p)
        #2nd phi
        if hasattr(tp,'_phi2deriv'):
            if isinstance(tp,potential.planarPotential): 
                tevaldphi2= tp(1.2,phi=0.1,dphi=2)
                tphi2deriv= tp.phi2deriv(1.2,phi=0.1)
            else:
                tevaldphi2= tp(1.2,0.1,phi=0.1,dphi=2)
                tphi2deriv= tp.phi2deriv(1.2,0.1,phi=0.1)
            if not tevaldphi2 is None:
                try:
                    if tevaldphi2**2. < 10.**ttol:
                        assert(tphi2deriv*2. < 10.**ttol)
                    else:
                        assert((tevaldphi2-tphi2deriv)**2./tevaldphi2**2. < 10.**ttol)
                except AssertionError:
                    raise AssertionError("Calculation of 2nd azimuthal derivative through _evaluate and phi2deriv inconsistent for the %s potential" % p)
        continue
        #mixed radial,vertical
        if isinstance(tp,potential.planarPotential): 
            tevaldrz= tp(1.2,0.1,phi=0.1,dR=1,dz=1)
            trzderiv= tp.Rzderiv(1.2,0.1,phi=0.1)
        else:
            tevaldrz= tp(1.2,0.1,phi=0.1,dR=1,dz=1)
            trzderiv= tp.Rzderiv(1.2,0.1,phi=0.1)
        if not tevaldrz is None:
            try:
                if tevaldrz**2. < 10.**ttol:
                    assert(trzderiv*2. < 10.**ttol)
                else:
                    assert((tevaldrz-trzderiv)**2./tevaldrz**2. < 10.**ttol)
            except AssertionError:
                raise AssertionError("Calculation of mixed radial,vertical derivative through _evaluate and z2deriv inconsistent for the %s potential" % p)

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
        try:
            assert(isinstance(tpp,potential.planarPotential))
        except AssertionError:
            raise AssertionError("Conversion into planar potential of potential %s fails" % p)
        tlp= tp.toVertical(1.)
        try:
            assert(isinstance(tlp,potential.linearPotential))
        except AssertionError:
            raise AssertionError("Conversion into linear potential of potential %s fails" % p)

# Sanity check the derivative of the rotation curve and the frequencies in the plane
def test_dvcircdR_omegac_epifreq_rl():
    from galpy import potential
    #Derivative of rotation curve
    #LogarithmicHaloPotential: rotation everywhere flat
    lp= potential.LogarithmicHaloPotential(normalize=1.)
    try:
        assert(lp.dvcircdR(1.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHaloPotential's rotation curve is not flat at R=1")
    try:
        assert(lp.dvcircdR(0.5)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHaloPotential's rotation curve is not flat at R=0.5")
    try:
        assert(lp.dvcircdR(2.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHaloPotential's rotation curve is not flat at R=2")
    #Kepler potential, vc = vc_0(R/R0)^-0.5 -> dvcdR= -0.5 vc_0 (R/R0)**-1.5
    kp= potential.KeplerPotential(normalize=1.)
    try:
        assert((kp.dvcircdR(1.)+0.5)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's rotation curve is not what it should be at R=1")
    try:
        assert((kp.dvcircdR(0.5)+0.5**-0.5)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's rotation curve is not what it should be at R=0.5")
    try:
        assert((kp.dvcircdR(2.)+0.5**2.5)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's rotation curve is not what it should be at R=2")
    #Rotational frequency
    try:
        assert((lp.omegac(1.)-1.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's rotational frequency is off at R=1")
    try:
        assert((lp.omegac(0.5)-2.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's rotational frequency is off at R=0.5")
    try:
        assert((lp.omegac(2.)-0.5)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's rotational frequency is off at R=2")
    #Epicycle frequency, flat rotation curve
    try:
        assert((lp.epifreq(1.)-numpy.sqrt(2.)*lp.omegac(1.))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=1")
    try:
        assert((lp.epifreq(0.5)-numpy.sqrt(2.)*lp.omegac(0.5))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=0.5")
    try:
        assert((lp.epifreq(2.0)-numpy.sqrt(2.)*lp.omegac(2.0))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("LogarithmicHalo's epicycle and rotational frequency are inconsistent with kappa = sqrt(2) Omega at R=2")
    #Epicycle frequency, Kepler
    try:
        assert((kp.epifreq(1.)-kp.omegac(1.))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=1")
    try:
        assert((kp.epifreq(0.5)-kp.omegac(0.5))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=0.5")
    try:
        assert((kp.epifreq(2.)-kp.omegac(2.))**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's epicycle and rotational frequency are inconsistent with kappa = Omega at R=2")
    #Check radius of circular orbit, Kepler
    try:
        assert((kp.rl(1.)-1.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's radius of a circular orbit is wrong at Lz=1.")
    try:
        assert((kp.rl(0.5)-1./4.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's radius of a circular orbit is wrong at Lz=0.5")
    try:
        assert((kp.rl(2.)-4.)**2. < 10.**-16.)
    except AssertionError:
        raise AssertionError("KeplerPotential's radius of a circular orbit is wrong at Lz=2.")
    return None

def test_plotting():
    from galpy import potential
    #Some tests of the plotting routines, to make sure they don't fail
    kp= potential.KeplerPotential(normalize=1.)
    #Plot the rotation curve
    kp.plotRotcurve()
    kp.plotRotcurve(Rrange=[0.01,10.],
                    grid=101,
                    savefilename=None)
    #Plot the escape-velocity curve
    kp.plotEscapecurve()
    kp.plotEscapecurve(Rrange=[0.01,10.],
                       grid=101,
                       savefilename=None)
    #Plot the potential itself
    kp.plot()
    kp.plot(t=1.,rmin=0.01,rmax=1.8,nrs=11,zmin=-0.55,zmax=0.55,nzs=11, 
            effective=False,Lz=None, 
            xrange=[0.01,1.8],yrange=[-0.55,0.55], 
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
    return None

#Classes for testing Integer TwoSphericalPotential and for testing special
# cases of some other potentials
from galpy.potential import TwoPowerSphericalPotential, \
    MiyamotoNagaiPotential, PowerSphericalPotential
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
