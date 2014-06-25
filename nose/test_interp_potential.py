import numpy
from galpy import potential

def test_interpolation_potential():
    #Test the interpolation of the potential
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=True)
    #This just tests on the grid
    rs= numpy.linspace(0.01,2.,21)
    zs= numpy.linspace(0.0,0.2,21)
    for r in rs:
        for z in zs:
            assert numpy.fabs((rzpot(r,z)
                              -potential.evaluatePotentials(r,z,potential.MWPotential))/potential.evaluatePotentials(r,z,potential.MWPotential)) < 10.**-10., 'RZPot interpolation w/ interpRZPotential fails at (R,z) = (%g,%g)' % (r,z)
    #This just tests within the grid
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(0.0,0.2,20)
    for r in rs:
        for z in zs:
            assert numpy.fabs((rzpot(r,z)
                              -potential.evaluatePotentials(r,z,potential.MWPotential))/potential.evaluatePotentials(r,z,potential.MWPotential)) < 10.**-6., 'RZPot interpolation w/ interpRZPotential fails at (R,z) = (%g,%g)' % (r,z)
    #Test all at the same time to use vector evaluation
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input'
    #Test the interpolation of the potential, now with logR
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(numpy.log(0.01),numpy.log(20.),
                                              201),
                                       logR=True,
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=True)
    rs= numpy.linspace(0.01,20.,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, w/ logR'
    #Test the interpolation of the potential, w/o zsym
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(-0.2,0.2,101),
                                       interpPot=True,
                                       zsym=False)
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(-0.2,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, w/o zsym'
    #Test the interpolation of the potential, w/o zsym and with logR
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(numpy.log(0.01),numpy.log(20.),
                                              201),
                                       logR=True,
                                       zgrid=(-0.2,0.2,101),
                                       interpPot=True,
                                       zsym=False)
    rs= numpy.linspace(0.01,20.,20)
    zs= numpy.linspace(-0.2,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input w/o zsym and w/ logR'
    return None

def test_interpolation_potential_diffinputs():
    #Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=True)
    #Test all at the same time to use vector evaluation
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(0.0,0.2,20)
    #R vector, z scalar
    assert numpy.all(numpy.fabs((rzpot(rs,zs[10])-potential.evaluatePotentials(rs,zs[10]*numpy.ones(len(rs)),potential.MWPotential))/potential.evaluatePotentials(rs,zs[10]*numpy.ones(len(rs)),potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z'
    #R scalar, z vector
    assert numpy.all(numpy.fabs((rzpot(rs[10],zs)-potential.evaluatePotentials(rs[10]*numpy.ones(len(zs)),zs,potential.MWPotential))/potential.evaluatePotentials(rs[10]*numpy.ones(len(zs)),zs,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z'
    return None

def test_interpolation_potential_c():
    #Test the interpolation of the potential
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,151),
                                       zgrid=(0.,0.2,151),
                                       interpPot=True,
                                       enable_c=True,
                                       zsym=True)
    #Test within the grid, using vector evaluation
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(0.0,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, using C'
    #now w/o zsym
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,151),
                                       zgrid=(-0.2,0.2,301),
                                       interpPot=True,
                                       enable_c=True,
                                       zsym=False)
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/o zsym'
    #now with logR
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(numpy.log(0.01),numpy.log(20.),
                                              251),
                                       logR= True,
                                       zgrid=(0.,0.2,151),
                                       interpPot=True,
                                       enable_c=True,
                                       zsym=True)
    rs= numpy.linspace(0.01,10.,20) #don't go too far
    zs= numpy.linspace(0.0,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR'
    #now with logR and w/o zsym
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(numpy.log(0.01),numpy.log(20.),
                                              251),
                                       logR= True,
                                       zgrid=(-0.2,0.2,301),
                                       interpPot=True,
                                       enable_c=True,
                                       zsym=False)
    rs= numpy.linspace(0.01,10.,20) #don't go too far
    zs= numpy.linspace(-0.2,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym'
    return None

def test_interpolation_potential_diffinputs_c():
    #Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,151),
                                       zgrid=(0.,0.2,151),
                                       interpPot=True,
                                       zsym=True,enable_c=True)
    #Test all at the same time to use vector evaluation
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(0.0,0.2,20)
    #R vector, z scalar
    assert numpy.all(numpy.fabs((rzpot(rs,zs[10])-potential.evaluatePotentials(rs,zs[10]*numpy.ones(len(rs)),potential.MWPotential))/potential.evaluatePotentials(rs,zs[10]*numpy.ones(len(rs)),potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z'
    #R scalar, z vector
    assert numpy.all(numpy.fabs((rzpot(rs[10],zs)-potential.evaluatePotentials(rs[10]*numpy.ones(len(zs)),zs,potential.MWPotential))/potential.evaluatePotentials(rs[10]*numpy.ones(len(zs)),zs,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z'
    return None

def test_interpolation_potential_c_vdiffgridsizes():
    #Test the interpolation of the potential
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,271),
                                       zgrid=(0.,0.2,162),
                                       interpPot=True,
                                       enable_c=True,
                                       zsym=True)
    #Test within the grid, using vector evaluation
    rs= numpy.linspace(0.01,2.,20)
    zs= numpy.linspace(0.0,0.2,20)
    mr,mz= numpy.meshgrid(rs,zs)
    mr= mr.flatten()
    mz= mz.flatten()
    assert numpy.all(numpy.fabs((rzpot(mr,mz)-potential.evaluatePotentials(mr,mz,potential.MWPotential))/potential.evaluatePotentials(mr,mz,potential.MWPotential)) < 10.**-6.), 'RZPot interpolation w/ interpRZPotential fails for vector input, using C'
    return None

def test_interpolation_potential_use_c():
    #Test the interpolation of the potential, using C to calculate the grid
    rzpot_c= potential.interpRZPotential(RZPot=potential.MWPotential,
                                         rgrid=(0.01,2.,101),
                                         zgrid=(0.,0.2,101),
                                         interpPot=True,
                                         zsym=True,
                                         use_c=False)
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=True,
                                       use_c=True)
    assert numpy.all(numpy.fabs(rzpot._potGrid-rzpot_c._potGrid) < 10.**-14.), \
        'Potential interpolation grid calculated with use_c does not agree with that calculated in python'
    return None

# Test evaluation outside the grid
def test_interpolation_potential_outsidegrid():
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=False)
    rs= [0.005,2.5]
    zs= [-0.1,0.3]
    for r in rs:
        for z in zs:
            assert numpy.fabs((rzpot(r,z)
                               -potential.evaluatePotentials(r,z,potential.MWPotential))/potential.evaluatePotentials(r,z,potential.MWPotential)) < 10.**-10., 'RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = (%g,%g)' % (r,z)
    return None

# Test evaluation outside the grid in C
def test_interpolation_potential_outsidegrid_c():
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=True,
                                       zsym=False,
                                       enable_c=True)
    rs= [0.005,2.5]
    zs= [-0.1,0.3]
    for r in rs:
        for z in zs:
            assert numpy.fabs((rzpot(r,z)
                               -potential.evaluatePotentials(r,z,potential.MWPotential))/potential.evaluatePotentials(r,z,potential.MWPotential)) < 10.**-10., 'RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = (%g,%g)' % (r,z)
    return None

def test_interpolation_potential_notinterpolated():
    rzpot= potential.interpRZPotential(RZPot=potential.MWPotential,
                                       rgrid=(0.01,2.,101),
                                       zgrid=(0.,0.2,101),
                                       interpPot=False,
                                       zsym=True)
    rs= [0.5,1.5]
    zs= [0.075,0.15]
    for r in rs:
        for z in zs:
            assert numpy.fabs((rzpot(r,z)
                               -potential.evaluatePotentials(r,z,potential.MWPotential))/potential.evaluatePotentials(r,z,potential.MWPotential)) < 10.**-10., 'RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at (R,z) = (%g,%g)' % (r,z)
    return None

# Test Rforce

# Test Rforce in C

# Test zforce

# Test zforce in C

# Test density

# Test density in C

# Test the circular velocity

# Test dvcircdR

# Test epifreq

# Test verticalfreq

