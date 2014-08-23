# Tests of the evolveddiskdf module
import numpy
from galpy.df import evolveddiskdf, dehnendf
from galpy.potential import LogarithmicHaloPotential

def test_axi_meanvr_grid():
    # Test that for an axisymmetric potential, the mean vr is zero
    idf= dehnendf(beta=0.)
    edf= evolveddiskdf(idf,pot=LogarithmicHaloPotential(normalize=1.),
                       to=-10.)
    mvr, grid= edf.meanvR(0.9,phi=0.2,integrate_method='rk6_c',grid=True,
                          returnGrid=True)
    assert numpy.fabs(mvr) < 0.001, 'meanvR of evolveddiskdf for axisymmetric potential is not equal to zero'
    mvr= edf.meanvR(0.9,phi=0.2,integrate_method='rk6_c',grid=grid)
    assert numpy.fabs(mvr) < 0.001, 'meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated with pre-computed grid'
    mvr= edf.meanvR(0.9,phi=0.2,integrate_method='rk6_c',grid=False)
    assert numpy.fabs(mvr) < 0.001, 'meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated directly'
    return None
                       
def test_axi_meanvr_direct():
    # Test that for an axisymmetric potential, the mean vr is zero
    idf= dehnendf(beta=0.)
    edf= evolveddiskdf(idf,pot=LogarithmicHaloPotential(normalize=1.),
                       to=-10.)
    mvr= edf.meanvR(0.9,phi=0.2,integrate_method='rk6_c',grid=False)
    assert numpy.fabs(mvr) < 0.001, 'meanvR of evolveddiskdf for axisymmetric potential is not equal to zero when calculated directly'
    return None
                       
def test_axi_meanvt_grid():
    # Test that for an axisymmetric potential, the mean vt is equal to that of the initial DF
    idf= dehnendf(beta=0.)
    edf= evolveddiskdf(idf,pot=LogarithmicHaloPotential(normalize=1.),
                       to=-10.)
    mvt, grid= edf.meanvT(0.9,phi=0.2,integrate_method='rk6_c',grid=True,
                          returnGrid=True)
    assert numpy.fabs(mvt-idf.meanvT(0.9)) < 0.001, 'meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf'
    mvt= edf.meanvT(0.9,phi=0.2,integrate_method='rk6_c',grid=grid)
    assert numpy.fabs(mvt-idf.meanvT(0.9)) < 0.001, 'meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid'
    return None
                       
def test_axi_meanvt_hierarchgrid():
    # Test that for an axisymmetric potential, the mean vt is equal to that of the initial DF
    idf= dehnendf(beta=0.)
    edf= evolveddiskdf(idf,pot=LogarithmicHaloPotential(normalize=1.),
                       to=-10.)
    mvt, grid= edf.meanvT(0.9,phi=0.2,integrate_method='rk6_c',grid=True,
                          hierarchgrid=True,
                          returnGrid=True)
    assert numpy.fabs(mvt-idf.meanvT(0.9)) < 0.001, 'meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using hierarchgrid'
    mvt= edf.meanvT(0.9,phi=0.2,integrate_method='rk6_c',grid=grid)
    assert numpy.fabs(mvt-idf.meanvT(0.9)) < 0.001, 'meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when calculated with pre-computed grid when using hierarchgrid'
    return None
                       
def test_axi_meanvt_direct():
    # Test that for an axisymmetric potential, the mean vt is equal to that of the initial DF
    idf= dehnendf(beta=0.)
    edf= evolveddiskdf(idf,pot=LogarithmicHaloPotential(normalize=1.),
                       to=-10.)
    mvt= edf.meanvT(0.9,phi=0.2,integrate_method='rk6_c',grid=False)
    assert numpy.fabs(mvt-idf.meanvT(0.9)) < 0.001, 'meanvT of evolveddiskdf for axisymmetric potential is not equal to that of the initial dehnendf when using direct integration'
    return None
                       
