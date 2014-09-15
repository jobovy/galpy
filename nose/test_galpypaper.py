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

def test_potmethods():
    from galpy.potential import DoubleExponentialDiskPotential
    dp= DoubleExponentialDiskPotential(normalize=1.,
                                       hr=3./8.,hz=0.3/8.)
    dp(1.,0.1) # The potential itself at R=1., z=0.1
    assert numpy.fabs(dp(1.,0.1)+1.1037196286636572) < 10.**-4., 'potmethods has changed'
    dp.Rforce(1.,0.1) # The radial force
    assert numpy.fabs(dp.Rforce(1.,0.1)+0.9147659436328015) < 10.**-4., 'potmethods has changed'
    dp.zforce(1.,0.1) # The vertical force
    assert numpy.fabs(dp.zforce(1.,0.1)+0.50056789703079607) < 10.**-4., 'potmethods has changed'
    dp.R2deriv(1.,0.1) # The second radial derivative
    assert numpy.fabs(dp.R2deriv(1.,0.1)+1.0189440730205248) < 10.**-4., 'potmethods has changed'
    dp.z2deriv(1.,0.1) # The second vertical derivative
    assert numpy.fabs(dp.z2deriv(1.,0.1)-1.0648350937842703) < 10.**-4., 'potmethods has changed'
    dp.Rzderiv(1.,0.1) # The mixed radial,vertical derivative
    assert numpy.fabs(dp.Rzderiv(1.,0.1)+1.1872449759212851) < 10.**-4., 'potmethods has changed'
    dp.dens(1.,0.1) # The density
    assert numpy.fabs(dp.dens(1.,0.1)-0.076502355610946121) < 10.**-4., 'potmethods has changed'
    dp.dens(1.,0.1,forcepoisson=True) # Using Poisson's eqn.
    assert numpy.fabs(dp.dens(1.,0.1,forcepoisson=True)-0.076446652249682681) < 10.**-4., 'potmethods has changed'
    dp.mass(1.,0.1) # The mass
    assert numpy.fabs(dp.mass(1.,0.1)-0.7281629803939751) < 10.**-4., 'potmethods has changed'
    dp.vcirc(1.) # The circular velocity at R=1.
    assert numpy.fabs(dp.vcirc(1.)-1.0) < 10.**-4., 'potmethods has changed' # By definition, because of normalize=1.
    dp.omegac(1.) # The rotational frequency
    assert numpy.fabs(dp.omegac(1.)-1.0) < 10.**-4., 'potmethods has changed' # Also because of normalize=1.
    dp.epifreq(1.) # The epicycle frequency
    assert numpy.fabs(dp.epifreq(1.)-1.3301123099210266) < 10.**-4., 'potmethods has changed'
    dp.verticalfreq(1.) # The vertical frequency
    assert numpy.fabs(dp.verticalfreq(1.)-3.7510872575640293) < 10.**-4., 'potmethods has changed'
    dp.flattening(1.,0.1) #The flattening (see caption)
    assert numpy.fabs(dp.flattening(1.,0.1)-0.42748757564198159) < 10.**-4., 'potmethods has changed'
    dp.lindbladR(1.75,m='corotation') # co-rotation resonance
    assert numpy.fabs(dp.lindbladR(1.75,m='corotation')-0.540985051273488) < 10.**-4., 'potmethods has changed'
    return None

