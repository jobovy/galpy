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

from galpy.potential import Potential

def smoothInterp(t,dt,tform):
    """Smooth interpolation in time, following Dehnen (2000)"""
    if t < tform: smooth= 0.
    elif t > (tform+dt): smooth= 1.
    else:
        xi= 2.*(t-tform)/dt-1.
        smooth= (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5)
    return smooth
    
class TimeInterpPotential(Potential):
    """Potential that smoothly interpolates in time between two static potentials"""
    def __init__(self,pot1,pot2,dt=100.,tform=50.):
        """pot1= potential for t < tform, pot2= potential for t > tform+dt, dt: time over which to turn on pot2,
        tform: time at which the interpolation is switched on"""
        Potential.__init__(self,amp=1.)
        self._pot1= pot1
        self._pot2= pot2
        self._tform= tform
        self._dt= dt
        return None
    
    def _Rforce(self,R,z,phi=0.,t=0.):
        smooth= smoothInterp(t,self._dt,self._tform)
        return (1.-smooth)*self._pot1.Rforce(R,z)+smooth*self._pot2.Rforce(R,z)
    
    def _zforce(self,R,z,phi=0.,t=0.):
        smooth= smoothInterp(t,self._dt,self._tform)
        return (1.-smooth)*self._pot1.zforce(R,z)+smooth*self._pot2.zforce(R,z)

def test_TimeInterpPotential():
    #Just to check that the code above has run properly
    from galpy.potential import LogarithmicHaloPotential, \
        MiyamotoNagaiPotential
    lp= LogarithmicHaloPotential(normalize=1.)
    mp= MiyamotoNagaiPotential(normalize=1.)
    tip= TimeInterpPotential(lp,mp)
    assert numpy.fabs(tip.Rforce(1.,0.1,t=10.)-lp.Rforce(1.,0.1)) < 10.**-8., 'TimeInterPotential does not work as expected'
    assert numpy.fabs(tip.Rforce(1.,0.1,t=200.)-mp.Rforce(1.,0.1)) < 10.**-8., 'TimeInterPotential does not work as expected'
    return None

def test_orbitint():
    import numpy
    from galpy.potential import MWPotential2014
    from galpy.potential import evaluatePotentials as evalPot
    from galpy.orbit import Orbit
    E, Lz= -1.25, 0.6
    o1= Orbit([0.8,0.,Lz/0.8,0.,numpy.sqrt(2.*(E-evalPot(0.8,0.,MWPotential2014)-(Lz/0.8)**2./2.)),0.])
    ts= numpy.linspace(0.,100.,2001)
    o1.integrate(ts,MWPotential2014)
    o1.plot(xrange=[0.3,1.],yrange=[-0.2,0.2],color='k')
    o2= Orbit([0.8,0.3,Lz/0.8,0.,numpy.sqrt(2.*(E-evalPot(0.8,0.,MWPotential2014)-(Lz/0.8)**2./2.-0.3**2./2.)),0.])
    o2.integrate(ts,MWPotential2014)
    o2.plot(xrange=[0.3,1.],yrange=[-0.2,0.2],color='k')
    return None

def test_orbmethods():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    o= Orbit([0.8,0.3,0.75,0.,0.2,0.]) # setup R,vR,vT,z,vz,phi
    times= numpy.linspace(0.,10.,1001) # Output times
    o.integrate(times,MWPotential2014) # Integrate
    o.E() # Energy
    assert numpy.fabs(o.E()+1.2547650648697966) < 10.**-5., 'Orbit method does not work as expected'
    o.L() # Angular momentum
    assert numpy.all(numpy.fabs(o.L()-numpy.array([[ 0.  , -0.16,  0.6 ]])) < 10.**-5.), 'Orbit method does not work as expected'
    o.Jacobi(OmegaP=0.65) #Jacobi integral E-OmegaP Lz
    assert numpy.fabs(o.Jacobi(OmegaP=0.65)-numpy.array([-1.64476506])) < 10.**-5., 'Orbit method does not work as expected'
    o.ER(times[-1]), o.Ez(times[-1]) # Rad. and vert. E at end
    assert numpy.fabs(o.ER(times[-1])+1.27601734263047) < 10.**-5., 'Orbit method does not work as expected'
    assert numpy.fabs(o.Ez(times[-1])-0.021252201847851909) < 10.**-5.,  'Orbit method does not work as expected'
    o.rperi(), o.rap(), o.zmax() # Peri-/apocenter r, max. |z|
    assert numpy.fabs(o.rperi()-0.44231993168097) < 10.**-5., 'Orbit method does not work as expected'
    assert numpy.fabs(o.rap()-0.87769030382105) < 10.**-5., 'Orbit method does not work as expected'
    assert numpy.fabs(o.zmax()-0.077452357289016) < 10.**-5., 'Orbit method does not work as expected'
    o.e() # eccentricity (rap-rperi)/(rap+rperi)
    assert numpy.fabs(o.e()-0.32982348199330563) < 10.**-5., 'Orbit method does not work as expected'
    o.R(2.,ro=8.) # Cylindrical radius at time 2. in kpc
    assert numpy.fabs(o.R(2.,ro=8.)-3.5470772876920007) < 10.**-3., 'Orbit method does not work as expected'
    o.vR(5.,vo=220.) # Cyl. rad. velocity at time 5. in km/s
    assert numpy.fabs(o.vR(5.,vo=220.)-45.202530965094553) < 10.**-3., 'Orbit method does not work as expected'
    o.ra(1.), o.dec(1.) # RA and Dec at t=1. (default settings)
    assert numpy.fabs(o.ra(1.)-numpy.array([ 288.19277])) < 10.**-3., 'Orbit method does not work as expected'
    assert numpy.fabs(o.dec(1.)-numpy.array([ 18.98069155])) < 10.**-3., 'Orbit method does not work as expected'
    o.jr(type='adiabatic'), o.jz() # R/z actions (ad. approx.)
    assert numpy.fabs(o.jr(type='adiabatic')-0.05285302231137586) < 10.**-3., 'Orbit method does not work as expected'
    assert numpy.fabs(o.jz()-0.006637988850751242) < 10.**-3., 'Orbit method does not work as expected'
    # Rad. period w/ Staeckel approximation w/ focal length 0.5,
    o.Tr(type='staeckel',delta=0.5,ro=8.,vo=220.) # in Gyr  
    assert numpy.fabs(o.Tr(type='staeckel',delta=0.5,ro=8.,vo=220.)-0.1039467864018446) < 10.**-3., 'Orbit method does not work as expected'
    o.plot(d1='R',d2='z') # Plot the orbit in (R,z)
    o.plot3d() # Plot the orbit in 3D, w/ default [x,y,z]
    return None

def test_orbsetup():
    from galpy.orbit import Orbit
    o= Orbit([25.,10.,2.,5.,-2.,50.],radec=True,ro=8.,
             vo=220.,solarmotion=[-11.1,25.,7.25])
    return None

def test_surfacesection():
    #Preliminary code
    import numpy
    from galpy.potential import MWPotential2014
    from galpy.potential import evaluatePotentials as evalPot
    from galpy.orbit import Orbit
    E, Lz= -1.25, 0.6
    o1= Orbit([0.8,0.,Lz/0.8,0.,numpy.sqrt(2.*(E-evalPot(0.8,0.,MWPotential2014)-(Lz/0.8)**2./2.)),0.])
    ts= numpy.linspace(0.,100.,2001)
    o1.integrate(ts,MWPotential2014)
    o2= Orbit([0.8,0.3,Lz/0.8,0.,numpy.sqrt(2.*(E-evalPot(0.8,0.,MWPotential2014)-(Lz/0.8)**2./2.-0.3**2./2.)),0.])
    o2.integrate(ts,MWPotential2014)
    def surface_section(Rs,zs,vRs):
        # Find points where the orbit crosses z from - to +
        shiftzs= numpy.roll(zs,-1)
        indx= (zs[:-1] < 0.)*(shiftzs[:-1] > 0.)
        return (Rs[:-1][indx],vRs[:-1][indx])
    # Calculate and plot the surface of section
    ts= numpy.linspace(0.,1000.,20001) # long integration
    o1.integrate(ts,MWPotential2014)
    o2.integrate(ts,MWPotential2014)
    sect1Rs,sect1vRs=surface_section(o1.R(ts),o1.z(ts),o1.vR(ts))
    sect2Rs,sect2vRs=surface_section(o2.R(ts),o2.z(ts),o2.vR(ts))
    from matplotlib.pyplot import plot, xlim, ylim
    plot(sect1Rs,sect1vRs,'bo',mec='none')
    xlim(0.3,1.); ylim(-0.69,0.69)
    plot(sect2Rs,sect2vRs,'yo',mec='none')
    return None
