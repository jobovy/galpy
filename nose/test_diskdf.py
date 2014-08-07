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

# Tests for cold population, flat rotation curve: <vt> =~ v_c
def test_dehnendf_cold_flat_skewvt():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.skewvT(1.)) < 1./20., 'skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvT(0.5)) < 1./20., 'skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvT(2.)) < 1./20., 'skew vT of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, power-law rotation curve: <vt> =~ v_c
def test_dehnendf_cold_powerrise_skewvt():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.skewvT(1.)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvT(0.5)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvT(2.)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
    return None

def test_dehnendf_cold_powerfall_skewvt():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.skewvT(1.)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvT(0.5)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvT(2.)) < 1./20., 'skew vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, flat rotation curve: <vr> = 0
def test_dehnendf_cold_flat_vr():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.meanvR(1.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.meanvR(0.5)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.meanvR(2.)-0.) < 10.**-3., 'mean vR of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, flat rotation curve: kurtosis = 0
def test_dehnendf_cold_flat_kurtosisvt():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.kurtosisvT(1.)) < 1./20., 'kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvT(0.5)) < 1./20., 'kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvT(2.)) < 1./20., 'kurtosis vT of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, power-law rotation curve: kurtosis = 0
def test_dehnendf_cold_powerrise_kurtosisvt():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.kurtosisvT(1.)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvT(0.5)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvT(2.)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2'

def test_dehnendf_cold_powerfall_kurtosisvt():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.kurtosisvT(1.)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvT(0.5)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvT(2.)) < 1./20., 'kurtosis vT of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
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

# Tests for cold population, flat rotation curve: <vr> = 0
def test_dehnendf_cold_flat_skewvr():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.skewvR(1.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvR(0.5)-0.) < 10.**-3., 'skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvR(2.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, power-law rotation curve: <vr> = 0
def test_dehnendf_cold_powerrise_skewvr():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.skewvR(1.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvR(0.5)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvR(2.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'

def test_dehnendf_cold_powerfall_skewvr():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.skewvR(1.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.skewvR(0.5)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.skewvR(2.)-0.) < 10.**-3., 'skew vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, flat rotation curve: kurtosis = 0
def test_dehnendf_cold_flat_kurtosisvr():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.kurtosisvR(1.)) < 1./20., 'kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvR(0.5)) < 1./20., 'kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvR(2.)) < 1./20., 'kurtosis vR of cold dehnendf in a flat rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, power-law rotation curve: kurtosis = 0
def test_dehnendf_cold_powerrise_kurtosisvr():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.kurtosisvR(1.)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvR(0.5)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvR(2.)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'

def test_dehnendf_cold_powerfall_kurtosisvr():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.kurtosisvR(1.)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=1'
    assert numpy.fabs(df.kurtosisvR(0.5)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=0.5'
    assert numpy.fabs(df.kurtosisvR(2.)) < 1./20., 'kurtosis vR of cold dehnendf in a power-law rotation curve is not close to zero at R=2'
    return None

# Tests for cold population, flat rotation curve: A = 0.5
def test_dehnendf_cold_flat_oortA():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*1./0.5) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*1./2.) < 10.**-3., 'Oort A of cold dehnendf in a flat rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, power-law rotation curve: A 
def test_dehnendf_cold_powerrise_oortA():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*(0.5)**beta/0.5*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*(2.)**beta/2.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

def test_dehnendf_cold_powerfall_oortA():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortA(1.)-0.5*1./1.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortA(0.5)-0.5*(0.5)**beta/0.5*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortA(2.)-0.5*(2.)**beta/2.*(1.-beta)) < 10.**-3., 'Oort A of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, flat rotation curve: B = -0.5
def test_dehnendf_cold_flat_oortB():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.oortB(1.)+0.5*1./1.) < 10.**-3., 'Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortB(0.5)+0.5*1./0.5) < 10.**-3., 'Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortB(2.)+0.5*1./2.) < 10.**-3., 'Oort B of cold dehnendf in a flat rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, power-law rotation curve: B 
def test_dehnendf_cold_powerrise_oortB():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortB(1.)+0.5*1./1.*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortB(0.5)+0.5*(0.5)**beta/0.5*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortB(2.)+0.5*(2.)**beta/2.*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

def test_dehnendf_cold_powerfall_oortB():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortB(1.)+0.5*1./1.*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortB(0.5)+0.5*(0.5)**beta/0.5*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortB(2.)+0.5*(2.)**beta/2.*(1.+beta)) < 10.**-3., 'Oort B of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, flat rotation curve: C = 0
def test_dehnendf_cold_flat_oortC():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.oortC(1.)) < 10.**-3., 'Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortC(0.5)) < 10.**-3., 'Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortC(2.)) < 10.**-3., 'Oort C of cold dehnendf in a flat rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, power-law rotation curve: C
def test_dehnendf_cold_powerrise_oortC():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortC(1.)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortC(0.5)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortC(2.)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

def test_dehnendf_cold_powerfall_oortC():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortC(1.)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortC(0.5)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortC(2.)) < 10.**-3., 'Oort C of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, flat rotation curve: K = 0
def test_dehnendf_cold_flat_oortK():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.oortK(1.)) < 10.**-3., 'Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortK(0.5)) < 10.**-3., 'Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortK(2.)) < 10.**-3., 'Oort K of cold dehnendf in a flat rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, power-law rotation curve: K
def test_dehnendf_cold_powerrise_oortK():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortK(1.)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortK(0.5)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortK(2.)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

def test_dehnendf_cold_powerfall_oortK():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.oortK(1.)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.oortK(0.5)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=0.5'
    assert numpy.fabs(df.oortK(2.)) < 10.**-3., 'Oort K of cold dehnendf in a power-law rotation curve is not close to expected at R=2'
    return None

# Tests for cold population, flat rotation curve: sigma_R^2 / sigma_T^2 = kappa^2 / Omega^2
def test_dehnendf_cold_flat_srst():
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=0.,correct=False)
    assert numpy.fabs(df.sigmaR2(1.)/df.sigmaT2(1.)-2.) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(0.5)/df.sigmaT2(0.5)-2.) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(2.)/df.sigmaT2(2.)-2.) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    return None

# Tests for cold population, power-law rotation curve: sigma_R^2 / sigma_T^2 = kappa^2 / Omega^2
def test_dehnendf_cold_powerrise_srst():
    # Rising rotation curve
    beta= 0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.sigmaR2(1.)/df.sigmaT2(1.)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(0.5)/df.sigmaT2(0.5)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(2.)/df.sigmaT2(2.)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    return None

def test_dehnendf_cold_powerfall_srst():
    # Falling rotation curve
    beta= -0.2
    df= dehnendf(profileParams=(0.3333333333333333,1.0, 0.01),
                 beta=beta,correct=False)
    assert numpy.fabs(df.sigmaR2(1.)/df.sigmaT2(1.)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(0.5)/df.sigmaT2(0.5)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    assert numpy.fabs(df.sigmaR2(2.)/df.sigmaT2(2.)-2./(1.+beta)) < 10.**-2., 'sigma_R^2 / sigma_T^2 of cool dehnendf in a flat rotation curve is not close to expected at R=1'
    return None

def test_targetSigma2():
    beta= 0.
    df= dehnendf(profileParams=(0.3333333333333333,1.0,0.1),
                 beta=beta,correct=False)
    assert numpy.fabs(df.targetSigma2(1.)-0.1**2.) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    assert numpy.fabs(df.targetSigma2(.3)-0.1**2.*numpy.exp(-(0.3-1.)/0.5)) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    assert numpy.fabs(df.targetSigma2(3.,log=True)-numpy.log(0.1)*2.+(3.-1.)/0.5) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    return None

def test_targetSurfacemass():
    beta= 0.
    df= dehnendf(profileParams=(0.3333333333333333,1.0,0.1),
                 beta=beta,correct=False)
    assert numpy.fabs(df.targetSurfacemass(1.)-numpy.exp(-1./0.3333333333333333)) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    assert numpy.fabs(df.targetSurfacemass(.3)-numpy.exp(-0.3/0.3333333333333333)) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    assert numpy.fabs(df.targetSurfacemass(3.,log=True)+3./0.3333333333333333) < 10.**-8., 'targetSigma2 for dehnendf does not agree with input'
    return None
