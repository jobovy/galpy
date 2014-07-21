# Tests of the diskdf module: distribution functions from Dehnen (1999)
import numpy
from galpy.df import dehnendf, shudf

# Tests for cold population, flat rotation curve: <vt> =~ v_c
def test_dehnendf_cold_flat_vt():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.meanvT(1.)-1.) < 10.**-3., 'mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.meanvT(0.5)-1.) < 10.**-3., 'mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.meanvT(2.)-1.) < 10.**-3., 'mean vT of cold dehnendf in a flat rotation curve is not close to V_c at R=2'
    return None

# Tests for cold population, power-law rotation curve: <vt> =~ v_c
def test_dehnendf_cold_powerrise_vt():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.meanvT(1.)-1.) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.meanvT(0.5)-(0.5)**beta) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.meanvT(2.)-(2.)**beta) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=2'

def test_dehnendf_cold_powerfall_vt():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.meanvT(1.)-1.) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.meanvT(0.5)-(0.5)**beta) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.meanvT(2.)-(2.)**beta) < 10.**-3., 'mean vT of cold dehnendf in a power-law rotation curve is not close to V_c at R=2'
    return None

# Tests for cold population, flat rotation curve: <vr> = 0
def test_dehnendf_cold_flat_vr():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.meanvR(1.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.meanvR(0.5)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.meanvR(2.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, power-law rotation curve: <vr> = 0
def test_dehnendf_cold_powerrise_vr():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.meanvR(1.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.meanvR(0.5)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.meanvR(2.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'

def test_dehnendf_cold_powerfall_vr():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.meanvR(1.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.meanvR(0.5)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.meanvR(2.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, flat rotation curve: A = 0.5
def test_dehnendf_cold_flat_oortA():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*1./0.5) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*1./2.) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to V_c at R=2'
    return None

# Tests for cold population, power-law rotation curve: A = 0.5
def test_dehnendf_cold_powerrise_oortA():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*(0.5)**beta/0.5*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*(2.)**beta/2.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=2'
    return None

def test_dehnendf_cold_powerfall_oortA():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*(0.5)**beta/0.5*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*(2.)**beta/2.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to V_c at R=2'
    return None

