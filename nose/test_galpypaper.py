# Test that all of the examples in the galpy paper run
import numpy

def test_overview():
    from galpy.potential import NFWPotential
    np= NFWPotential(normalize=1.)
    from galpy.orbit import Orbit
    o= Orbit(vxvv=[1.,0.1,1.1,0.1,0.02,0.])
    from galpy.actionAngle import actionAngleSpherical
    aA= actionAngleSpherical(pot=np)
    js= aA(o)
    assert numpy.fabs((js[0]-0.00980542)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    assert numpy.fabs((js[1]-1.1)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    assert numpy.fabs((js[2]-0.00553155)/js[0]) < 10.**-3., 'Action calculation in the overview section has changed'
    from galpy.df import quasiisothermaldf
    qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                           pot=np,aA=aA)
    assert numpy.fabs((qdf(o)-61.57476085)/61.57476085) < 10.**-3., 'qdf calculation in the overview section has changed'
    return None

def test_import():
    import galpy
    import galpy.potential
    import galpy.orbit
    import galpy.actionAngle
    import galpy.df
    import galpy.util
    return None

def test_units():
    import galpy.util.bovy_conversion as conversion
    print conversion.force_in_pcMyr2(220.,8.)#pc/Myr^2
    assert numpy.fabs(conversion.force_in_pcMyr2(220.,8.)-6.32793804994) < 10.**-4., 'unit conversion has changed'
    print conversion.dens_in_msolpc3(220.,8.)#Msolar/pc^3
    assert numpy.fabs(conversion.dens_in_msolpc3(220.,8.)-0.175790330079) < 10.**-4., 'unit conversion has changed'
    print conversion.surfdens_in_msolpc2(220.,8.)#Msolar/pc^2
    assert numpy.fabs(conversion.surfdens_in_msolpc2(220.,8.)-1406.32264063) < 10.**-4., 'unit conversion has changed'
    print conversion.mass_in_1010msol(220.,8.)#10^10 Msolar
    assert numpy.fabs(conversion.mass_in_1010msol(220.,8.)-9.00046490005) < 10.**-4., 'unit conversion has changed'
    print conversion.freq_in_Gyr(220.,8.)#1/Gyr
    assert numpy.fabs(conversion.freq_in_Gyr(220.,8.)-28.1245845523) < 10.**-4., 'unit conversion has changed'
    print conversion.time_in_Gyr(220.,8.)#Gyr
    assert numpy.fabs(conversion.time_in_Gyr(220.,8.)-0.0355560807712) < 10.**-4., 'unit conversion has changed'
    return None
