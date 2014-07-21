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
