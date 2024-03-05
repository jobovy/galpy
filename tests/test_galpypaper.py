"""Test that all of the examples in the galpy paper run

isort:skip_file
"""

import os
import numpy
import pytest


def test_overview():
    from galpy.potential import NFWPotential

    np = NFWPotential(normalize=1.0)
    from galpy.orbit import Orbit

    o = Orbit(vxvv=[1.0, 0.1, 1.1, 0.1, 0.02, 0.0])
    from galpy.actionAngle import actionAngleSpherical

    aA = actionAngleSpherical(pot=np)
    js = aA(o)
    assert (
        numpy.fabs((js[0] - 0.00980542) / js[0]) < 10.0**-3.0
    ), "Action calculation in the overview section has changed"
    assert (
        numpy.fabs((js[1] - 1.1) / js[0]) < 10.0**-3.0
    ), "Action calculation in the overview section has changed"
    assert (
        numpy.fabs((js[2] - 0.00553155) / js[0]) < 10.0**-3.0
    ), "Action calculation in the overview section has changed"
    from galpy.df import quasiisothermaldf

    qdf = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=np, aA=aA)
    assert (
        numpy.fabs((qdf(o) - 61.57476085) / 61.57476085) < 10.0**-3.0
    ), "qdf calculation in the overview section has changed"
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
    # Import changed because of bovy_conversion --> conversion name change
    from galpy.util import conversion

    print(conversion.force_in_pcMyr2(220.0, 8.0))  # pc/Myr^2
    assert (
        numpy.fabs(conversion.force_in_pcMyr2(220.0, 8.0) - 6.32793804994) < 10.0**-4.0
    ), "unit conversion has changed"
    print(conversion.dens_in_msolpc3(220.0, 8.0))  # Msolar/pc^3
    # Loosen tolerances including mass bc of 0.025% change in Msun in astropyv2
    assert (
        numpy.fabs(
            (conversion.dens_in_msolpc3(220.0, 8.0) - 0.175790330079) / 0.175790330079
        )
        < 0.0003
    ), "unit conversion has changed"
    print(conversion.surfdens_in_msolpc2(220.0, 8.0))  # Msolar/pc^2
    assert (
        numpy.fabs(
            (conversion.surfdens_in_msolpc2(220.0, 8.0) - 1406.32264063) / 1406.32264063
        )
        < 0.0003
    ), "unit conversion has changed"
    print(conversion.mass_in_1010msol(220.0, 8.0))  # 10^10 Msolar
    assert (
        numpy.fabs(
            (conversion.mass_in_1010msol(220.0, 8.0) - 9.00046490005) / 9.00046490005
        )
        < 0.0003
    ), "unit conversion has changed"
    print(conversion.freq_in_Gyr(220.0, 8.0))  # 1/Gyr
    assert (
        numpy.fabs(conversion.freq_in_Gyr(220.0, 8.0) - 28.1245845523) < 10.0**-4.0
    ), "unit conversion has changed"
    print(conversion.time_in_Gyr(220.0, 8.0))  # Gyr
    assert (
        numpy.fabs(conversion.time_in_Gyr(220.0, 8.0) - 0.0355560807712) < 10.0**-4.0
    ), "unit conversion has changed"
    return None


def test_potmethods():
    from galpy.potential import DoubleExponentialDiskPotential

    dp = DoubleExponentialDiskPotential(normalize=1.0, hr=3.0 / 8.0, hz=0.3 / 8.0)
    dp(1.0, 0.1)  # The potential itself at R=1., z=0.1
    assert (
        numpy.fabs(dp(1.0, 0.1) + 1.1037196286636572) < 10.0**-4.0
    ), "potmethods has changed"
    dp.Rforce(1.0, 0.1)  # The radial force
    assert (
        numpy.fabs(dp.Rforce(1.0, 0.1) + 0.9147659436328015) < 10.0**-4.0
    ), "potmethods has changed"
    dp.zforce(1.0, 0.1)  # The vertical force
    assert (
        numpy.fabs(dp.zforce(1.0, 0.1) + 0.50056789703079607) < 10.0**-4.0
    ), "potmethods has changed"
    dp.R2deriv(1.0, 0.1)  # The second radial derivative
    # Loosened tolerance, because new (more precise) calculation differs by 3e-4
    assert (
        numpy.fabs(dp.R2deriv(1.0, 0.1) + 1.0189440730205248) < 3 * 10.0**-4.0
    ), "potmethods has changed"
    dp.z2deriv(1.0, 0.1)  # The second vertical derivative
    # Loosened tolerance, because new (more precise) calculation differs by 4e-4
    assert (
        numpy.fabs(dp.z2deriv(1.0, 0.1) - 1.0648350937842703) < 4 * 10.0**-4.0
    ), "potmethods has changed"
    dp.Rzderiv(1.0, 0.1)  # The mixed radial,vertical derivative
    assert (
        numpy.fabs(dp.Rzderiv(1.0, 0.1) + 1.1872449759212851) < 10.0**-4.0
    ), "potmethods has changed"
    dp.dens(1.0, 0.1)  # The density
    assert (
        numpy.fabs(dp.dens(1.0, 0.1) - 0.076502355610946121) < 10.0**-4.0
    ), "potmethods has changed"
    dp.dens(1.0, 0.1, forcepoisson=True)  # Using Poisson's eqn.
    assert (
        numpy.fabs(dp.dens(1.0, 0.1, forcepoisson=True) - 0.076446652249682681)
        < 10.0**-4.0
    ), "potmethods has changed"
    dp.mass(1.0, 0.1)  # The mass
    assert (
        numpy.fabs(dp.mass(1.0, 0.1) - 0.7281629803939751) < 10.0**-4.0
    ), "potmethods has changed"
    dp.vcirc(1.0)  # The circular velocity at R=1.
    assert (
        numpy.fabs(dp.vcirc(1.0) - 1.0) < 10.0**-4.0
    ), "potmethods has changed"  # By definition, because of normalize=1.
    dp.omegac(1.0)  # The rotational frequency
    assert (
        numpy.fabs(dp.omegac(1.0) - 1.0) < 10.0**-4.0
    ), "potmethods has changed"  # Also because of normalize=1.
    dp.epifreq(1.0)  # The epicycle frequency
    # Loosened tolerance, because new (more precise) calculation differs by 1e-3
    assert (
        numpy.fabs(dp.epifreq(1.0) - 1.3301123099210266) < 2 * 10.0**-3.0
    ), "potmethods has changed"
    dp.verticalfreq(1.0)  # The vertical frequency
    # Loosened tolerance, because new (more precise) calculation differs by 1e-3
    assert (
        numpy.fabs(dp.verticalfreq(1.0) - 3.7510872575640293) < 10.0**-3.0
    ), "potmethods has changed"
    dp.flattening(1.0, 0.1)  # The flattening (see caption)
    assert (
        numpy.fabs(dp.flattening(1.0, 0.1) - 0.42748757564198159) < 10.0**-4.0
    ), "potmethods has changed"
    dp.lindbladR(1.75, m="corotation")  # co-rotation resonance
    assert (
        numpy.fabs(dp.lindbladR(1.75, m="corotation") - 0.540985051273488) < 10.0**-4.0
    ), "potmethods has changed"
    return None


from galpy.potential import Potential


def smoothInterp(t, dt, tform):
    """Smooth interpolation in time, following Dehnen (2000)"""
    if t < tform:
        smooth = 0.0
    elif t > (tform + dt):
        smooth = 1.0
    else:
        xi = 2.0 * (t - tform) / dt - 1.0
        smooth = 3.0 / 16.0 * xi**5.0 - 5.0 / 8 * xi**3.0 + 15.0 / 16.0 * xi + 0.5
    return smooth


class TimeInterpPotential(Potential):
    """Potential that smoothly interpolates in time between two static potentials"""

    def __init__(self, pot1, pot2, dt=100.0, tform=50.0):
        """pot1= potential for t < tform, pot2= potential for t > tform+dt, dt: time over which to turn on pot2,
        tform: time at which the interpolation is switched on"""
        Potential.__init__(self, amp=1.0)
        self._pot1 = pot1
        self._pot2 = pot2
        self._tform = tform
        self._dt = dt
        return None

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        smooth = smoothInterp(t, self._dt, self._tform)
        return (1.0 - smooth) * self._pot1.Rforce(R, z) + smooth * self._pot2.Rforce(
            R, z
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        smooth = smoothInterp(t, self._dt, self._tform)
        return (1.0 - smooth) * self._pot1.zforce(R, z) + smooth * self._pot2.zforce(
            R, z
        )


def test_TimeInterpPotential():
    # Just to check that the code above has run properly
    from galpy.potential import LogarithmicHaloPotential, MiyamotoNagaiPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    mp = MiyamotoNagaiPotential(normalize=1.0)
    tip = TimeInterpPotential(lp, mp)
    assert (
        numpy.fabs(tip.Rforce(1.0, 0.1, t=10.0) - lp.Rforce(1.0, 0.1)) < 10.0**-8.0
    ), "TimeInterPotential does not work as expected"
    assert (
        numpy.fabs(tip.Rforce(1.0, 0.1, t=200.0) - mp.Rforce(1.0, 0.1)) < 10.0**-8.0
    ), "TimeInterPotential does not work as expected"
    return None


@pytest.mark.skip(reason="Test does not work correctly")
def test_potentialAPIChange_warning():
    # Test that a warning is displayed about the API change for evaluatePotentials etc. functions from what is given in the galpy paper
    # Turn warnings into errors to test for them
    import warnings
    from galpy.util import galpyWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        import galpy.potential

        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "A major change in versions > 1.1 is that all galpy.potential functions and methods take the potential as the first argument; previously methods such as evaluatePotentials, evaluateDensities, etc. would be called with (R,z,Pot), now they are called as (Pot,R,z) for greater consistency across the codebase"
            )
            if raisedWarning:
                break
        assert raisedWarning, "Importing galpy.potential does not raise warning about evaluatePotentials API change"
    return None


def test_orbitint():
    import numpy
    from galpy.potential import MWPotential2014
    from galpy.potential import evaluatePotentials as evalPot
    from galpy.orbit import Orbit

    E, Lz = -1.25, 0.6
    o1 = Orbit(
        [
            0.8,
            0.0,
            Lz / 0.8,
            0.0,
            numpy.sqrt(
                2.0 * (E - evalPot(MWPotential2014, 0.8, 0.0) - (Lz / 0.8) ** 2.0 / 2.0)
            ),
            0.0,
        ]
    )
    ts = numpy.linspace(0.0, 100.0, 2001)
    o1.integrate(ts, MWPotential2014)
    o1.plot(xrange=[0.3, 1.0], yrange=[-0.2, 0.2], color="k")
    o2 = Orbit(
        [
            0.8,
            0.3,
            Lz / 0.8,
            0.0,
            numpy.sqrt(
                2.0
                * (
                    E
                    - evalPot(MWPotential2014, 0.8, 0.0)
                    - (Lz / 0.8) ** 2.0 / 2.0
                    - 0.3**2.0 / 2.0
                )
            ),
            0.0,
        ]
    )
    o2.integrate(ts, MWPotential2014)
    o2.plot(xrange=[0.3, 1.0], yrange=[-0.2, 0.2], color="k")
    return None


def test_orbmethods():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # 8/17/2019: added explicit z=0.025, because that was the default at the
    # time of the galpy paper, but the default has been changed
    o = Orbit([0.8, 0.3, 0.75, 0.0, 0.2, 0.0], zo=0.025)  # setup R,vR,vT,z,vz,phi
    times = numpy.linspace(0.0, 10.0, 1001)  # Output times
    o.integrate(times, MWPotential2014)  # Integrate
    o.E()  # Energy
    assert (
        numpy.fabs(o.E() + 1.2547650648697966) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.L()  # Angular momentum
    assert numpy.all(
        numpy.fabs(o.L() - numpy.array([[0.0, -0.16, 0.6]])) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.Jacobi(OmegaP=0.65)  # Jacobi integral E-OmegaP Lz
    assert (
        numpy.fabs(o.Jacobi(OmegaP=0.65) - numpy.array([-1.64476506])) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.ER(times[-1]), o.Ez(times[-1])  # Rad. and vert. E at end
    assert (
        numpy.fabs(o.ER(times[-1]) + 1.27601734263047) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    assert (
        numpy.fabs(o.Ez(times[-1]) - 0.021252201847851909) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.rperi(), o.rap(), o.zmax()  # Peri-/apocenter r, max. |z|
    assert (
        numpy.fabs(o.rperi() - 0.44231993168097) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    assert (
        numpy.fabs(o.rap() - 0.87769030382105) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    assert (
        numpy.fabs(o.zmax() - 0.077452357289016) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.e()  # eccentricity (rap-rperi)/(rap+rperi)
    assert (
        numpy.fabs(o.e() - 0.32982348199330563) < 10.0**-5.0
    ), "Orbit method does not work as expected"
    o.R(2.0, ro=8.0)  # Cylindrical radius at time 2. in kpc
    assert (
        numpy.fabs(o.R(2.0, ro=8.0) - 3.5470772876920007) < 10.0**-3.0
    ), "Orbit method does not work as expected"
    o.vR(5.0, vo=220.0)  # Cyl. rad. velocity at time 5. in km/s
    assert (
        numpy.fabs(o.vR(5.0, vo=220.0) - 45.202530965094553) < 10.0**-3.0
    ), "Orbit method does not work as expected"
    o.ra(1.0), o.dec(1.0)  # RA and Dec at t=1. (default settings)
    # 5/12/2016: test weakened, because improved galcen<->heliocen
    #            transformation has changed these, but still close
    assert (
        numpy.fabs(o.ra(1.0) - numpy.array([288.19277])) < 10.0**-1.0
    ), "Orbit method does not work as expected"
    assert (
        numpy.fabs(o.dec(1.0) - numpy.array([18.98069155])) < 10.0**-1.0
    ), "Orbit method does not work as expected"
    o.jr(type="adiabatic"), o.jz()  # R/z actions (ad. approx.)
    assert (
        numpy.fabs(o.jr(type="adiabatic") - 0.05285302231137586) < 10.0**-3.0
    ), "Orbit method does not work as expected"
    assert (
        numpy.fabs(o.jz() - 0.006637988850751242) < 10.0**-3.0
    ), "Orbit method does not work as expected"
    # Rad. period w/ Staeckel approximation w/ focal length 0.5,
    o.Tr(type="staeckel", delta=0.5, ro=8.0, vo=220.0)  # in Gyr
    assert (
        numpy.fabs(
            o.Tr(type="staeckel", delta=0.5, ro=8.0, vo=220.0) - 0.1039467864018446
        )
        < 10.0**-3.0
    ), "Orbit method does not work as expected"
    o.plot(d1="R", d2="z")  # Plot the orbit in (R,z)
    o.plot3d()  # Plot the orbit in 3D, w/ default [x,y,z]
    return None


def test_orbsetup():
    from galpy.orbit import Orbit

    o = Orbit(
        [25.0, 10.0, 2.0, 5.0, -2.0, 50.0],
        radec=True,
        ro=8.0,
        vo=220.0,
        solarmotion=[-11.1, 25.0, 7.25],
    )
    return None


def test_surfacesection():
    # Preliminary code
    import numpy
    from galpy.potential import MWPotential2014
    from galpy.potential import evaluatePotentials as evalPot
    from galpy.orbit import Orbit

    E, Lz = -1.25, 0.6
    o1 = Orbit(
        [
            0.8,
            0.0,
            Lz / 0.8,
            0.0,
            numpy.sqrt(
                2.0 * (E - evalPot(MWPotential2014, 0.8, 0.0) - (Lz / 0.8) ** 2.0 / 2.0)
            ),
            0.0,
        ]
    )
    ts = numpy.linspace(0.0, 100.0, 2001)
    o1.integrate(ts, MWPotential2014)
    o2 = Orbit(
        [
            0.8,
            0.3,
            Lz / 0.8,
            0.0,
            numpy.sqrt(
                2.0
                * (
                    E
                    - evalPot(MWPotential2014, 0.8, 0.0)
                    - (Lz / 0.8) ** 2.0 / 2.0
                    - 0.3**2.0 / 2.0
                )
            ),
            0.0,
        ]
    )
    o2.integrate(ts, MWPotential2014)

    def surface_section(Rs, zs, vRs):
        # Find points where the orbit crosses z from - to +
        shiftzs = numpy.roll(zs, -1)
        indx = (zs[:-1] < 0.0) * (shiftzs[:-1] > 0.0)
        return (Rs[:-1][indx], vRs[:-1][indx])

    # Calculate and plot the surface of section
    ts = numpy.linspace(0.0, 1000.0, 20001)  # long integration
    o1.integrate(ts, MWPotential2014)
    o2.integrate(ts, MWPotential2014)
    sect1Rs, sect1vRs = surface_section(o1.R(ts), o1.z(ts), o1.vR(ts))
    sect2Rs, sect2vRs = surface_section(o2.R(ts), o2.z(ts), o2.vR(ts))
    from matplotlib.pyplot import plot, xlim, ylim

    plot(sect1Rs, sect1vRs, "bo", mec="none")
    xlim(0.3, 1.0)
    ylim(-0.69, 0.69)
    plot(sect2Rs, sect2vRs, "yo", mec="none")
    return None


def test_adinvariance():
    from galpy.potential import IsochronePotential
    from galpy.orbit import Orbit
    from galpy.actionAngle import actionAngleIsochrone

    # Initialize two different IsochronePotentials
    ip1 = IsochronePotential(normalize=1.0, b=1.0)
    ip2 = IsochronePotential(normalize=0.5, b=1.0)
    # Use TimeInterpPotential to interpolate smoothly
    tip = TimeInterpPotential(ip1, ip2, dt=100.0, tform=50.0)
    # Integrate: 1) Orbit in the first isochrone potential
    o1 = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
    ts = numpy.linspace(0.0, 50.0, 1001)
    o1.integrate(ts, tip)
    o1.plot(d1="x", d2="y", xrange=[-1.6, 1.6], yrange=[-1.6, 1.6], color="b")
    # 2) Orbit in the transition
    o2 = o1(ts[-1])  # Last time step => initial time step
    ts2 = numpy.linspace(50.0, 150.0, 1001)
    o2.integrate(ts2, tip)
    o2.plot(d1="x", d2="y", overplot=True, color="g")
    # 3) Orbit in the second isochrone potential
    o3 = o2(ts2[-1])
    ts3 = numpy.linspace(150.0, 200.0, 1001)
    o3.integrate(ts3, tip)
    o3.plot(d1="x", d2="y", overplot=True, color="r")
    # Now we calculate energy, maximum height, and mean radius
    print(o1.E(pot=ip1), (o1.rperi() + o1.rap()) / 2, o1.zmax())
    assert (
        numpy.fabs(o1.E(pot=ip1) + 2.79921356237) < 10.0**-4.0
    ), "Energy in the adiabatic invariance test is different"
    assert (
        numpy.fabs((o1.rperi() + o1.rap()) / 2 - 1.07854158141) < 10.0**-4.0
    ), "mean radius in the adiabatic invariance test is different"
    assert (
        numpy.fabs(o1.zmax() - 0.106331362938) < 10.0**-4.0
    ), "zmax in the adiabatic invariance test is different"
    print(o3.E(pot=ip2), (o3.rperi() + o3.rap()) / 2, o3.zmax())
    assert (
        numpy.fabs(o3.E(pot=ip2) + 1.19677002624) < 10.0**-4.0
    ), "Energy in the adiabatic invariance test is different"
    assert (
        numpy.fabs((o3.rperi() + o3.rap()) / 2 - 1.39962036137) < 10.0**-4.0
    ), "mean radius in the adiabatic invariance test is different"
    assert (
        numpy.fabs(o3.zmax() - 0.138364269321) < 10.0**-4.0
    ), "zmax in the adiabatic invariance test is different"
    # The orbit has clearly moved to larger radii,
    # the actions are however conserved from beginning to end
    aAI1 = actionAngleIsochrone(ip=ip1)
    print(aAI1(o1))
    js = aAI1(o1)
    assert (
        numpy.fabs(js[0] - numpy.array([0.00773779])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    assert (
        numpy.fabs(js[1] - numpy.array([1.1])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    assert (
        numpy.fabs(js[2] - numpy.array([0.0045361])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    aAI2 = actionAngleIsochrone(ip=ip2)
    print(aAI2(o3))
    js = aAI2(o3)
    assert (
        numpy.fabs(js[0] - numpy.array([0.00773812])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    assert (
        numpy.fabs(js[1] - numpy.array([1.1])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    assert (
        numpy.fabs(js[2] - numpy.array([0.0045361])) < 10.0**-4.0
    ), "action in the adiabatic invariance test is different"
    return None


def test_diskdf():
    from galpy.df import dehnendf

    # Init. dehnendf w/ flat rot., hr=1/3, hs=1, and sr(1)=0.2
    df = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.2))
    # Same, w/ correction factors to scale profiles
    dfc = dehnendf(
        beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.2), correct=True, niter=20
    )
    if True:
        # Log. diff. between scale and DF surf. dens.
        numpy.log(df.surfacemass(0.5) / df.targetSurfacemass(0.5))
        assert (
            numpy.fabs(
                numpy.log(df.surfacemass(0.5) / df.targetSurfacemass(0.5))
                + 0.056954077791649592
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Same for corrected DF
        numpy.log(dfc.surfacemass(0.5) / dfc.targetSurfacemass(0.5))
        assert (
            numpy.fabs(
                numpy.log(dfc.surfacemass(0.5) / dfc.targetSurfacemass(0.5))
                + 4.1440377205802041e-06
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Log. diff between scale and DF sr
        numpy.log(df.sigmaR2(0.5) / df.targetSigma2(0.5))
        assert (
            numpy.fabs(
                numpy.log(df.sigmaR2(0.5) / df.targetSigma2(0.5)) + 0.12786083001363127
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Same for corrected DF
        numpy.log(dfc.sigmaR2(0.5) / dfc.targetSigma2(0.5))
        assert (
            numpy.fabs(
                numpy.log(dfc.sigmaR2(0.5) / dfc.targetSigma2(0.5))
                + 6.8065001252214986e-06
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Evaluate DF w/ R,vR,vT
        df(numpy.array([0.9, 0.1, 0.8]))
        assert (
            numpy.fabs(
                df(numpy.array([0.9, 0.1, 0.8])) - numpy.array(0.1740247246180417)
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Evaluate corrected DF w/ Orbit instance
        from galpy.orbit import Orbit

        dfc(Orbit([0.9, 0.1, 0.8]))
        assert (
            numpy.fabs(dfc(Orbit([0.9, 0.1, 0.8])) - numpy.array(0.16834863725552207))
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate the mean velocities
        df.meanvR(0.9), df.meanvT(0.9)
        assert (
            numpy.fabs(df.meanvR(0.9)) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(df.meanvT(0.9) - 0.91144428051168291) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate the velocity dispersions
        numpy.sqrt(dfc.sigmaR2(0.9)), numpy.sqrt(dfc.sigmaT2(0.9))
        assert (
            numpy.fabs(numpy.sqrt(dfc.sigmaR2(0.9)) - 0.22103383792719539) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(numpy.sqrt(dfc.sigmaT2(0.9)) - 0.17613725303902811) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate the skew of the velocity distribution
        df.skewvR(0.9), df.skewvT(0.9)
        assert (
            numpy.fabs(df.skewvR(0.9)) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(df.skewvT(0.9) + 0.47331638366025863) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate the kurtosis of the velocity distribution
        df.kurtosisvR(0.9), df.kurtosisvT(0.9)
        assert (
            numpy.fabs(df.kurtosisvR(0.9) + 0.13561300880237059) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(df.kurtosisvT(0.9) - 0.12612702099300721) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate a higher-order moment of the velocity DF
        df.vmomentsurfacemass(1.0, 6.0, 2.0) / df.surfacemass(1.0)
        assert (
            numpy.fabs(
                df.vmomentsurfacemass(1.0, 6.0, 2.0) / df.surfacemass(1.0)
                - 0.00048953492205559054
            )
            < 10.0**-4.0
        ), "diskdf does not behave as expected"
        # Calculate the Oort functions
        dfc.oortA(1.0), dfc.oortB(1.0), dfc.oortC(1.0), dfc.oortK(1.0)
        assert (
            numpy.fabs(dfc.oortA(1.0) - 0.40958989067012197) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(dfc.oortB(1.0) + 0.49396172114486514) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(dfc.oortC(1.0)) < 10.0**-4.0
        ), "diskdf does not behave as expected"
        assert (
            numpy.fabs(dfc.oortK(1.0)) < 10.0**-4.0
        ), "diskdf does not behave as expected"
    # Sample Orbits from the DF, returns list of Orbits
    numpy.random.seed(1)
    os = dfc.sample(n=100, returnOrbit=True, nphi=1)
    # check that these have the right mean radius = 2hr=2/3
    rs = numpy.array([o.R() for o in os])
    assert numpy.fabs(numpy.mean(rs) - 2.0 / 3.0) < 0.1
    # Sample vR and vT at given R, check their mean
    vrvt = dfc.sampleVRVT(0.7, n=500, target=True)
    vt = vrvt[:, 1]
    assert numpy.fabs(numpy.mean(vrvt[:, 0])) < 0.05
    assert numpy.fabs(numpy.mean(vt) - dfc.meanvT(0.7)) < 0.01
    # Sample Orbits along a given line-of-sight
    os = dfc.sampleLOS(45.0, n=1000)
    return None


def test_oort():
    from galpy.df import dehnendf

    df = dehnendf(beta=0.0, correct=True, niter=20, profileParams=(1.0 / 3.0, 1.0, 0.1))
    va = 1.0 - df.meanvT(1.0)  # asymmetric drift
    A = df.oortA(1.0)
    B = df.oortB(1.0)
    return None


def test_qdf():
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential2014
    from galpy.actionAngle import actionAngleStaeckel

    # Setup actionAngle instance for action calcs
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    # Quasi-iso df w/ hr=1/3, hsr/z=1, sr(1)=0.2, sz(1)=0.1
    df = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, aA=aAS, pot=MWPotential2014)
    # Evaluate DF w/ R,vR,vT,z,vz
    df(0.9, 0.1, 0.8, 0.05, 0.02)
    assert (
        numpy.fabs(df(0.9, 0.1, 0.8, 0.05, 0.02) - numpy.array([123.57158928]))
        < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Evaluate DF w/ Orbit instance, return ln
    from galpy.orbit import Orbit

    df(Orbit([0.9, 0.1, 0.8, 0.05, 0.02]), log=True)
    assert (
        numpy.fabs(
            df(Orbit([0.9, 0.1, 0.8, 0.05, 0.02]), log=True) - numpy.array([4.81682066])
        )
        < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Evaluate DF marginalized over vz
    df.pvRvT(0.1, 0.9, 0.9, 0.05)
    assert (
        numpy.fabs(df.pvRvT(0.1, 0.9, 0.9, 0.05) - 2.0 * 23.273310451852243)
        < 10.0**-4.0
    ), "qdf does not behave as expected"
    # NOTE: The pvRvT() function has changed with respect to the version used in Bovy (2015).
    #      As of January 2018, a prefactor of 2 has been added (=nsigma/2 with default nsigma=4),
    #      to account for the correct Gauss-Legendre integration normalization.
    # Evaluate DF marginalized over vR,vT
    df.pvz(0.02, 0.9, 0.05)
    assert (
        numpy.fabs(df.pvz(0.02, 0.9, 0.05) - 50.949586235238172) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Calculate the density
    df.density(0.9, 0.05)
    assert (
        numpy.fabs(df.density(0.9, 0.05) - 12.73725936526167) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Estimate the DF's actual density scale length at z=0
    df.estimate_hr(0.9, 0.0)
    assert (
        numpy.fabs(df.estimate_hr(0.9, 0.0) - 0.322420336223) < 10.0**-2.0
    ), "qdf does not behave as expected"
    # Estimate the DF's actual surface-density scale length
    df.estimate_hr(0.9, None)
    assert (
        numpy.fabs(df.estimate_hr(0.9, None) - 0.38059909132766462) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Estimate the DF's density scale height
    df.estimate_hz(0.9, 0.02)
    assert (
        numpy.fabs(df.estimate_hz(0.9, 0.02) - 0.064836202345657207) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Calculate the mean velocities
    (
        df.meanvR(0.9, 0.05),
        df.meanvT(0.9, 0.05),
    )
    df.meanvz(0.9, 0.05)
    assert (
        numpy.fabs(df.meanvR(0.9, 0.05) - 3.8432265354618213e-18) < 10.0**-4.0
    ), "qdf does not behave as expected"
    assert (
        numpy.fabs(df.meanvT(0.9, 0.05) - 0.90840425173325279) < 10.0**-4.0
    ), "qdf does not behave as expected"
    assert (
        numpy.fabs(df.meanvz(0.9, 0.05) + 4.3579787517991084e-19) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Calculate the velocity dispersions
    from numpy import sqrt

    sqrt(df.sigmaR2(0.9, 0.05)), sqrt(df.sigmaz2(0.9, 0.05))
    assert (
        numpy.fabs(sqrt(df.sigmaR2(0.9, 0.05)) - 0.22695537077102387) < 10.0**-4.0
    ), "qdf does not behave as expected"
    assert (
        numpy.fabs(sqrt(df.sigmaz2(0.9, 0.05)) - 0.094215523962105044) < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Calculate the tilt of the velocity ellipsoid
    # 2017/10-28: CHANGED bc tilt now returns angle in rad, no longer in deg
    df.tilt(0.9, 0.05)
    assert (
        numpy.fabs(df.tilt(0.9, 0.05) - 2.5166061974413765 / 180.0 * numpy.pi)
        < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Calculate a higher-order moment of the velocity DF
    df.vmomentdensity(0.9, 0.05, 6.0, 2.0, 2.0, gl=True)
    assert (
        numpy.fabs(
            df.vmomentdensity(0.9, 0.05, 6.0, 2.0, 2.0, gl=True) - 0.0001591100892366438
        )
        < 10.0**-4.0
    ), "qdf does not behave as expected"
    # Sample velocities at given R,z, check mean
    numpy.random.seed(1)
    vs = df.sampleV(0.9, 0.05, n=500)
    mvt = numpy.mean(vs[:, 1])
    assert numpy.fabs(numpy.mean(vs[:, 0])) < 0.05  # vR
    assert numpy.fabs(mvt - df.meanvT(0.9, 0.05)) < 0.01  # vT
    assert numpy.fabs(numpy.mean(vs[:, 2])) < 0.05  # vz
    return None


def test_coords():
    from galpy.util import coords

    ra, dec, dist = 161.0, 50.0, 8.5
    pmra, pmdec, vlos = -6.8, -10.0, -115.0
    # Convert to Galactic and then to rect. Galactic
    ll, bb = coords.radec_to_lb(ra, dec, degree=True)
    pmll, pmbb = coords.pmrapmdec_to_pmllpmbb(pmra, pmdec, ra, dec, degree=True)
    X, Y, Z = coords.lbd_to_XYZ(ll, bb, dist, degree=True)
    vX, vY, vZ = coords.vrpmllpmbb_to_vxvyvz(vlos, pmll, pmbb, X, Y, Z, XYZ=True)
    # Convert to cylindrical Galactocentric
    # Assuming Sun's distance to GC is (8,0.025) in (R,z)
    R, phi, z = coords.XYZ_to_galcencyl(X, Y, Z, Xsun=8.0, Zsun=0.025)
    vR, vT, vz = coords.vxvyvz_to_galcencyl(
        vX, vY, vZ, R, phi, Z, vsun=[-10.1, 244.0, 6.7], galcen=True
    )
    # 5/12/2016: test weakened, because improved galcen<->heliocen
    #            transformation has changed these, but still close
    print(R, phi, z, vR, vT, vz)
    assert (
        numpy.fabs(R - 12.51328515156942) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    assert (
        numpy.fabs(phi - 0.12177409073433249) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    assert (
        numpy.fabs(z - 7.1241282354856228) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    assert (
        numpy.fabs(vR - 78.961682923035966) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    assert (
        numpy.fabs(vT + 241.49247772351964) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    assert (
        numpy.fabs(vz + 102.83965442188689) < 10.0**-1.0
    ), "Coordinate transformation has changed"
    return None
