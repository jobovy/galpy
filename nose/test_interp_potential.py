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
                                       rgrid=(0.01,20.,201),
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
                                       rgrid=(0.01,2.,201),
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
