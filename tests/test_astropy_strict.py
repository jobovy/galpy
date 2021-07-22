# Tests of the astropy strict mode that requires non-dimensionless inputs
# to be given as quantities and all outputs to have units turned on
import pytest
from galpy.util import config
# Don't set astropy_strict here, because then it affects all test files run
# through pytest at the same time
import numpy
from astropy import units, constants

def check_apy_strict_inputs_error_msg(excinfo,input_value,input_type,
                                      weak=False):
    if weak:
        assert "with units of {} is not a Quantity"\
            .format(input_type) in str(excinfo.value), \
            """astropy_strict error message incorrect"""
    else:
        assert str(excinfo.value) == \
            """astropy-strict config mode set that requires all"""\
            """ non-dimensionless inputs to be specified as """\
            """Quantities, but """\
            """input {} with units of {} is not a Quantity"""\
            .format(input_value,input_type), \
            """astropy_strict error message incorrect"""

def test_potential_ampunits():
    config.__config__.set('astropy','astropy-strict','True')
    # Test that input units for potential amplitudes behave as expected
    # ~clone of the same test_quantity function
    from galpy import potential
    ro, vo= 9.*units.kpc, 210.*units.km/units.s
    # Burkert
    with pytest.raises(ValueError) as excinfo:
        pot= potential.BurkertPotential(amp=0.1,
                                        a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.1,'density')
    pot= potential.BurkertPotential(amp=0.1*units.Msun/units.pc**3.,
                                    a=2.*units.kpc,ro=ro,vo=vo)
    # DoubleExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DoubleExponentialDiskPotential(\
                        amp=0.1,hr=2.*units.kpc,
                        hz=0.2*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.1,'density')
    pot= potential.DoubleExponentialDiskPotential(\
                        amp=0.1*units.Msun/units.pc**3.,hr=2.*units.kpc,
                        hz=0.2*units.kpc,ro=ro,vo=vo)
    # TwoPowerSphericalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TwoPowerSphericalPotential(amp=20.,a=2.*units.kpc,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,a=2.*units.kpc,
                                              alpha=1.5,beta=3.5,ro=ro,vo=vo)
    # JaffePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.JaffePotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.JaffePotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo)
    # HernquistPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.HernquistPotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.HernquistPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo)
    # NFWPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.NFWPotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.NFWPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo)
    # TwoPowerTriaxialPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TwoPowerTriaxialPotential(amp=20.,a=2.*units.kpc,
                                                 b=0.3,c=1.4,
                                                 alpha=1.5,beta=3.5,
                                                 ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.TwoPowerTriaxialPotential(amp=20.*units.Msun,a=2.*units.kpc,
                                             b=0.3,c=1.4,
                                             alpha=1.5,beta=3.5,ro=ro,vo=vo)
    # TriaxialJaffePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialJaffePotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo,
                                          b=0.3,c=1.4)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.TriaxialJaffePotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo,
                                          b=0.3,c=1.4)
    # TriaxialHernquistPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialHernquistPotential(amp=20.,a=2.*units.kpc,
                                                  b=0.4,c=1.4,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.TriaxialHernquistPotential(amp=20.*units.Msun,a=2.*units.kpc,
                                              b=0.4,c=1.4,ro=ro,vo=vo)
    # TriaxialNFWPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialNFWPotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo,
                                            b=1.3,c=0.4)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.TriaxialNFWPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo,
                                        b=1.3,c=0.4)
    # SCFPotential, default = spherical Hernquist
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SCFPotential(amp=20.,a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'mass') # because /2
    pot= potential.SCFPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo)
    # FlattenedPowerPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FlattenedPowerPotential(amp=40000.,
                                               r1=1.*units.kpc,q=0.9,alpha=0.5,core=0.*units.kpc,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,40000.,'velocity2')
    pot= potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s**2,
                                           r1=1.*units.kpc,q=0.9,alpha=0.5,core=0.*units.kpc,
                                           ro=ro,vo=vo)
    # IsochronePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.IsochronePotential(amp=20.,b=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.IsochronePotential(amp=20.*units.Msun,b=2.*units.kpc,ro=ro,vo=vo)
    # KeplerPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KeplerPotential(amp=20.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.KeplerPotential(amp=20.*units.Msun,ro=ro,vo=vo)
    # KuzminKutuzovStaeckelPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KuzminKutuzovStaeckelPotential(amp=20.,
                                                      Delta=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')        
    pot= potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun,
                                                  Delta=2.*units.kpc,ro=ro,vo=vo)
    # LogarithmicHaloPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.LogarithmicHaloPotential(amp=40000.,
                                                core=0.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,40000.,'velocity2')        
    pot= potential.LogarithmicHaloPotential(amp=40000*units.km**2/units.s**2,
                                            core=0.*units.kpc,ro=ro,vo=vo)
    # MiyamotoNagaiPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MiyamotoNagaiPotential(amp=20.,
                                              a=2.*units.kpc,b=0.5*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun,
                                          a=2.*units.kpc,b=0.5*units.kpc,ro=ro,vo=vo)
    # KuzminDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KuzminDiskPotential(amp=20.,
                                           a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.KuzminDiskPotential(amp=20*units.Msun,
                                       a=2.*units.kpc,ro=ro,vo=vo)
    # MN3ExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MN3ExponentialDiskPotential(\
                        amp=0.1,hr=2.*units.kpc,hz=0.2*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.1,'density')
    pot= potential.MN3ExponentialDiskPotential(\
        amp=0.1*units.Msun/units.pc**3.,hr=2.*units.kpc,hz=0.2*units.kpc,ro=ro,vo=vo)
    # PlummerPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PlummerPotential(amp=20.,
                                    b=0.5*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'mass')
    pot= potential.PlummerPotential(amp=20*units.Msun,
                                    b=0.5*units.kpc,ro=ro,vo=vo)
    # PowerSphericalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PowerSphericalPotential(amp=10.**10.,
                                               r1=1.*units.kpc,
                                               alpha=2.,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.**10.,'mass')
    pot= potential.PowerSphericalPotential(amp=10.**10.*units.Msun,
                                           r1=1.*units.kpc,
                                           alpha=2.,ro=ro,vo=vo)
    # PowerSphericalPotentialwCutoff
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PowerSphericalPotentialwCutoff(amp=0.1,
                                                      rc=2.*units.kpc,
                                                      r1=1.*units.kpc,alpha=2.,
                                                      ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.1,'density')
    pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**3,
                                                  rc=2.*units.kpc,
                                                  r1=1.*units.kpc,alpha=2.,
                                                  ro=ro,vo=vo)
    # PseudoIsothermalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PseudoIsothermalPotential(amp=10.**10.,
                                                 a=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.**10.,'mass')
    pot= potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun,
                                             a=2.*units.kpc,ro=ro,vo=vo)
    # RazorThinExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RazorThinExponentialDiskPotential(amp=40.,
                                                         hr=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,40.,'surfacedensity')
    pot= potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun/units.pc**2,
                                                     hr=2.*units.kpc,ro=ro,vo=vo)
    # SoftenedNeedleBarPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.,
                                                  a=1.*units.kpc,b=2.*units.kpc,c=3.*units.kpc,pa=0.*units.deg,omegab=0./units.Gyr,
                                              ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                              a=1.*units.kpc,b=2.*units.kpc,c=3.*units.kpc,pa=0.*units.deg,omegab=0./units.Gyr,
                                              ro=ro,vo=vo)
    # FerrersPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FerrersPotential(amp=4.*10.**10.,
                                        a=1.*units.kpc,b=2.,c=3.,pa=0.*units.deg,omegab=0./units.Gyr,
                                        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.FerrersPotential(amp=4.*10.**10.*units.Msun,
                                    a=1.*units.kpc,b=2.,c=3.,pa=0.*units.deg,omegab=0./units.Gyr,
                                    ro=ro,vo=vo)
    # # SpiralArmsPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SpiralArmsPotential(amp=0.3,
                                           ro=ro, vo=vo,
                                           N=2, alpha=13*units.deg,
                                           r_ref=0.8*units.kpc,
                                           phi_ref=90.*units.deg,
                                           Rs=8*units.kpc,
                                           H=0.1*units.kpc,
                                           omega=20.*units.km/units.s/units.kpc,
                                           Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,0.3,'density')
    pot= potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                       ro=ro, vo=vo,
                                       N=2, alpha=13*units.deg,
                                       r_ref=0.8*units.kpc,
                                       phi_ref=90.*units.deg,
                                       Rs=8*units.kpc,
                                       H=0.1*units.kpc,
                                       omega=20.*units.km/units.s/units.kpc,
                                       Cs=[1])   
    # SphericalShellPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SphericalShellPotential(amp=4.*10.**10.,
                                               a=1*units.kpc,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.SphericalShellPotential(amp=4.*10.**10.*units.Msun,
                                           a=1*units.kpc,
                                           ro=ro,vo=vo)
    # RingPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RingPotential(amp=4.*10.**10.,
                                     a=1*units.kpc,
                                     ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.RingPotential(amp=4.*10.**10.*units.Msun,
                                 a=1*units.kpc,
                                 ro=ro,vo=vo)
    # PerfectEllipsoidPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PerfectEllipsoidPotential(amp=4.*10.**10.,
                                                 a=2.*units.kpc,ro=ro,vo=vo,
                                                 b=1.3,c=0.4)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.PerfectEllipsoidPotential(amp=4.*10.**10.*units.Msun,
                                             a=2.*units.kpc,ro=ro,vo=vo,
                                             b=1.3,c=0.4)
    # HomogeneousSpherePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.HomogeneousSpherePotential(amp=0.1,
                                                  R=2.*units.kpc,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.1,'density')
    pot= potential.HomogeneousSpherePotential(amp=0.1*units.Msun/units.pc**3.,
                                              R=2.*units.kpc,ro=ro,vo=vo)
    # TriaxialGaussianPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialGaussianPotential(amp=4.*10.**10.,
                                                 sigma=2.*units.kpc,ro=ro,vo=vo,
                                                 b=1.3,c=0.4)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**10.,'mass')
    pot= potential.TriaxialGaussianPotential(amp=4.*10.**10.*units.Msun,
                                             sigma=2.*units.kpc,ro=ro,vo=vo,
                                             b=1.3,c=0.4)
    return None

def test_potential_paramunits():
    config.__config__.set('astropy','astropy-strict','True')
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import conversion
    ro, vo= 7.*units.kpc, 230.*units.km/units.s
    # Burkert
    with pytest.raises(ValueError) as excinfo:
        pot= potential.BurkertPotential(amp=0.1*units.Msun/units.pc**3.,
                                        a=2.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,2.,'length')
    # DoubleExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DoubleExponentialDiskPotential(\
                        amp=0.1*units.Msun/units.pc**3.,hr=4.,hz=200.*units.pc,
        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DoubleExponentialDiskPotential(\
                        amp=0.1*units.Msun/units.pc**3.,hr=4.*units.kpc,hz=200.,
        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,200.,'length')
    # TwoPowerSphericalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,
                                                  a=10.,
                                                  alpha=1.5,beta=3.5,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # TwoPowerSphericalPotential with integer powers
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TwoPowerSphericalPotential(amp=20.*units.Msun,
                                                  a=10.,
                                                  alpha=2.,
                                                  beta=5.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # JaffePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.JaffePotential(amp=20.*units.Msun,a=10.,
                                      ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # HernquistPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.HernquistPotential(amp=20.*units.Msun,a=10.,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # NFWPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.NFWPotential(amp=20.*units.Msun,a=15.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,15.,'length')
    # NFWPotential, rmax,vmax
    with pytest.raises(ValueError) as excinfo:
        pot= potential.NFWPotential(rmax=10.,
                                    vmax=175.*units.km/units.s,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.NFWPotential(rmax=10.*units.kpc,
                                    vmax=175.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,175.,'velocity')
    # SCFPotential, default = Hernquist
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SCFPotential(amp=20.*units.Msun,a=10.,
                                    ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # TwoPowerTriaxialPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TwoPowerTriaxialPotential(amp=20.*units.Msun,
                                                 a=10.,
                                                 alpha=1.5,beta=3.5,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # TriaxialJaffePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialJaffePotential(amp=20.*units.Msun,
                                              a=0.02,
                                              b=0.2,c=0.8,
                                              ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,0.02,'length')
    # TriaxialHernquistPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialHernquistPotential(amp=20.*units.Msun,
                                                  a=10.,
                                                  b=0.7,c=0.9,
                                              ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # TriaxialNFWPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialNFWPotential(amp=20.*units.Msun,a=15.,
                                            b=1.3,c=0.2,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,15.,'length')
    # Also do pa
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialNFWPotential(amp=20.*units.Msun,
                                            a=15.*units.kpc,
                                            pa=30.,
                                            b=1.3,c=0.2,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,30.,'angle')
    # FlattenedPowerPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s**2,
                                               r1=10.,
                                           q=0.9,alpha=0.5,core=1.*units.kpc,
                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FlattenedPowerPotential(amp=40000.*units.km**2/units.s**2,
                                               r1=10.*units.kpc,
                                               q=0.9,alpha=0.5,core=1.,
                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1.,'length')
    # IsochronePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.IsochronePotential(amp=20.*units.Msun,b=10.,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # KuzminKutuzovStaeckelPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KuzminKutuzovStaeckelPotential(amp=20.*units.Msun,
                                                      Delta=10.,
                                                      ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # LogarithmicHaloPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.LogarithmicHaloPotential(amp=40000*units.km**2/units.s**2,
                                                core=1.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1.,'length')
    # DehnenBarPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          omegab=50.,
                                          rb=4.*units.kpc,
                                          Af=1290.*units.km**2/units.s**2,
                                          barphi=20.*units.deg,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,50.,'frequency')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          omegab=50.*units.km/units.s/units.kpc,
                                          rb=4.,
                                          Af=1290.*units.km**2/units.s**2,
                                          barphi=20.*units.deg,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          omegab=50.*units.km/units.s/units.kpc,
                                          rb=4.*units.kpc,
                                          Af=1290.,
                                          barphi=20.*units.deg,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1290.,'energy')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          omegab=50.*units.km/units.s/units.kpc,
                                          rb=4.*units.kpc,
                                          Af=1290.*units.km**2/units.s**2,
                                          barphi=20.,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'angle')
    # DehnenBarPotential, alternative setup
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          rolr=8.,
                                          chi=0.8,
                                          alpha=0.02,
                                          beta=0.2,
                                          barphi=20.*units.deg,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,8.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenBarPotential(amp=1.,
                                          rolr=8.*units.kpc,
                                          chi=0.8,
                                          alpha=0.02,
                                          beta=0.2,
                                          barphi=20.,
                                          ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'angle')
    # MiyamotoNagaiPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun,
                                              a=5.,b=300.*units.pc,
                                              ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MiyamotoNagaiPotential(amp=20*units.Msun,
                                              a=5.*units.kpc,b=300.,
                                              ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,300.,'length')
    # KuzminDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KuzminDiskPotential(amp=20*units.Msun,
                                           a=5.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # MN3ExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MN3ExponentialDiskPotential(\
            amp=0.1*units.Msun/units.pc**3.,hr=6.,hz=300.*units.pc,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,6.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.MN3ExponentialDiskPotential(\
            amp=0.1*units.Msun/units.pc**3.,hr=6.*units.kpc,hz=300.,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,300.,'length')
    # PlummerPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PlummerPotential(amp=20*units.Msun,
                                        b=5.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # PowerSphericalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PowerSphericalPotential(amp=10.**10.*units.Msun,
                                               r1=10.,
                                               alpha=2.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # PowerSphericalPotentialwCutoff
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**3,
                                                      r1=10.,
                                                      alpha=2.,rc=12.*units.kpc,
                                                      ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PowerSphericalPotentialwCutoff(amp=0.1*units.Msun/units.pc**3,
                                                      r1=10.*units.kpc,
                                                      alpha=2.,rc=12.,
                                                      ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,12.,'length')
    # PseudoIsothermalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PseudoIsothermalPotential(amp=10.**10.*units.Msun,
                                                 a=20.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'length')
    # RazorThinExponentialDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RazorThinExponentialDiskPotential(amp=40.*units.Msun/units.pc**2,
                                                         hr=10.,
                                                         ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # SoftenedNeedleBarPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                                  a=10.,
                                                  b=2.*units.kpc,
                                                  c=3.*units.kpc,
                                                  pa=10.*units.deg,
                                                  omegab=20.*units.km/units.s/units.kpc,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                                  a=10.*units.kpc,
                                                  b=2.,
                                                  c=3.*units.kpc,
                                                  pa=10.*units.deg,
                                                  omegab=20.*units.km/units.s/units.kpc,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,2.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                                  a=10.*units.kpc,
                                                  b=2.*units.kpc,
                                                  c=3.,
                                                  pa=10.*units.deg,
                                                  omegab=20.*units.km/units.s/units.kpc,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,3.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                                  a=10.*units.kpc,
                                                  b=2.*units.kpc,
                                                  c=3.*units.kpc,
                                                  pa=10.,
                                                  omegab=20.*units.km/units.s/units.kpc,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SoftenedNeedleBarPotential(amp=4.*10.**10.*units.Msun,
                                                  a=10.*units.kpc,
                                                  b=2.*units.kpc,
                                                  c=3.*units.kpc,
                                                  pa=10.*units.deg,
                                                  omegab=20.,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'frequency')
    # FerrersPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FerrersPotential(amp=4.*10.**10.*units.Msun,
                                        a=10.,
                                        b=2.,
                                        c=3.,
                                        pa=10.*units.deg,
                                        omegab=20.*units.km/units.s/units.kpc,
                                        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FerrersPotential(amp=4.*10.**10.*units.Msun,
                                        a=10.*units.kpc,
                                        b=2.,
                                        c=3.,
                                        pa=10.,
                                        omegab=20.*units.km/units.s/units.kpc,
                                        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.FerrersPotential(amp=4.*10.**10.*units.Msun,
                                        a=10.*units.kpc,
                                        b=2.,
                                        c=3.,
                                        pa=10.*units.deg,
                                        omegab=20.,
                                        ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'frequency')
    # DiskSCFPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DiskSCFPotential(dens=lambda R,z: 1.,# doesn't matter
                                        Sigma=[{'type':'exp','h':1./3.,'amp':1.},
                                               {'type':'expwhole','h':1./3.,
                                                'amp':1.,'Rhole':0.5}],
                                        hz=[{'type':'exp','h':1./27.},
                                            {'type':'sech2','h':1./27.}],
                                        a=8.,N=2,L=2,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,8.,'length')
    # SpiralArmsPotential
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13.,
                                            r_ref=0.8*units.kpc,
                                            phi_ref=90.*units.deg,
                                            Rs=8*units.kpc,
                                            H=0.1*units.kpc,
                                            omega=20.*units.km/units.s/units.kpc, Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,13.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13*units.deg,
                                            r_ref=0.8,
                                            phi_ref=90.*units.deg,
                                            Rs=8*units.kpc,
                                            H=0.1*units.kpc,
                                            omega=20.*units.km/units.s/units.kpc, Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,0.8,'length')
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13.*units.deg,
                                            r_ref=0.8*units.kpc,
                                            phi_ref=90.,
                                            Rs=8*units.kpc,
                                            H=0.1*units.kpc,
                                            omega=20.*units.km/units.s/units.kpc, Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,90.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13*units.deg,
                                            r_ref=0.8*units.kpc,
                                            phi_ref=90.*units.deg,
                                            Rs=8.,
                                            H=0.1*units.kpc,
                                            omega=20.*units.km/units.s/units.kpc, Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,8.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13*units.deg,
                                            r_ref=0.8*units.kpc,
                                            phi_ref=90.*units.deg,
                                            Rs=8*units.kpc,
                                            H=0.1,
                                            omega=20.*units.km/units.s/units.kpc, Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,0.1,'length')
    with pytest.raises(ValueError) as excinfo:
        pot = potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                            ro=ro, vo=vo,
                                            N=2, alpha=13*units.deg,
                                            r_ref=0.8*units.kpc,
                                            phi_ref=90.*units.deg,
                                            Rs=8*units.kpc,
                                            H=0.1*units.kpc,
                                            omega=20., Cs=[1])
    check_apy_strict_inputs_error_msg(excinfo,20.,'frequency')
    # DehnenSmoothWrapperPotential
    dpn= potential.DehnenBarPotential(amp=1.,
                                      omegab=50./units.Gyr,
                                      rb=4.*units.kpc,
                                      Af=1290.*units.km**2/units.s**2,
                                      barphi=20.*units.deg,
                                      ro=ro,vo=vo)
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenSmoothWrapperPotential(pot=dpn,
                                                    tform=-1.,
                                                    tsteady=3.*units.Gyr,
                                                    ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,-1.,'time')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.DehnenSmoothWrapperPotential(pot=dpn,
                                                    tform=-1.*units.Gyr,
                                                    tsteady=3.,
                                                    ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,3.,'time')
    # SolidBodyRotationWrapperPotential
    spn= potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                       ro=ro, vo=vo,
                                       N=2, alpha=13*units.deg,
                                       r_ref=0.8*units.kpc,
                                       phi_ref=90.*units.deg,
                                       Rs=8*units.kpc,
                                       H=0.1*units.kpc,
                                       omega=20.*units.km/units.s/units.kpc,
                                       Cs=[1])   
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SolidBodyRotationWrapperPotential(pot=spn,\
                                                         omega=20.,
                                                         pa=30.*units.deg,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,20.,'frequency')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SolidBodyRotationWrapperPotential(pot=spn,\
                                                         omega=20.*units.km/units.s/units.kpc,
                                                         pa=30.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,30.,'angle')
    # CorotatingRotationWrapperPotential
    spn= potential.SpiralArmsPotential(amp=0.3*units.Msun/units.pc**3,
                                       ro=ro, vo=vo,
                                       N=2, alpha=13*units.deg,
                                       r_ref=0.8*units.kpc,
                                       phi_ref=90.*units.deg,
                                       Rs=8*units.kpc,
                                       H=0.1*units.kpc,
                                       omega=20.*units.km/units.s/units.kpc,
                                       Cs=[1])
    with pytest.raises(ValueError) as excinfo:
        pot= potential.CorotatingRotationWrapperPotential(pot=spn,\
                                                          vpo=200.,
                                                          to=1.*units.Gyr,
                                                          pa=30.*units.deg,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,200.,'velocity')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.CorotatingRotationWrapperPotential(pot=spn,\
                                                          vpo=200.*units.km/units.s,
                                                          to=1.,
                                                          pa=30.*units.deg,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1.,'time')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.CorotatingRotationWrapperPotential(pot=spn,\
                                                          vpo=200.*units.km/units.s,
                                                          to=1.*units.Gyr,
                                                          pa=30.,ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,30.,'angle')
    # GaussianAmplitudeWrapperPotential
    dpn= potential.DehnenBarPotential(amp=1.,
                                      omegab=50./units.Gyr,
                                      rb=4.*units.kpc,
                                      Af=1290.*units.km**2/units.s**2,
                                      barphi=20.*units.deg,
                                      ro=ro,vo=vo)
    with pytest.raises(ValueError) as excinfo:
        pot= potential.GaussianAmplitudeWrapperPotential(pot=dpn,
                                                         to=-1.,
                                                         sigma=10.*units.Gyr,
                                                         ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,-1.,'time')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.GaussianAmplitudeWrapperPotential(pot=dpn,
                                                         to=-1.*units.Gyr,
                                                         sigma=10.,
                                                         ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'time')
    # ChandrasekharDynamicalFrictionForce
    with pytest.raises(ValueError) as excinfo:
        pot= potential.ChandrasekharDynamicalFrictionForce(GMs=10.,
                                                           rhm=1.2*units.kpc,
                                                           minr=1.*units.pc,
                                                           maxr=100.*units.kpc,
                                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'mass')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.ChandrasekharDynamicalFrictionForce(GMs=10.**9.*units.Msun,
                                                           rhm=1.2,
                                                           minr=1.*units.pc,
                                                           maxr=100.*units.kpc,
                                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1.2,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.ChandrasekharDynamicalFrictionForce(GMs=10.**9.*units.Msun,
                                                           rhm=1.2*units.kpc,
                                                           minr=1.,
                                                           maxr=100.*units.kpc,
                                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,1.,'length')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.ChandrasekharDynamicalFrictionForce(GMs=10.**9.*units.Msun,
                                                           rhm=1.2*units.kpc,
                                                           minr=1.*units.pc,
                                                           maxr=100.,
                                                           ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,100.,'length')
    # SphericalShellPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.SphericalShellPotential(amp=4.*10.**10.*units.Msun,
                                               a=5.,
                                               ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # RingPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RingPotential(amp=4.*10.**10.*units.Msun,
                                     a=5.,
                                     ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # PerfectEllipsoidPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.PerfectEllipsoidPotential(amp=4.*10.**10.*units.Msun,
                                                 a=5.,
                                                 ro=ro,vo=vo,
                                                 b=1.3,c=0.4)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # HomogeneousSpherePotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.HomogeneousSpherePotential(amp=0.1*units.Msun/units.pc**3,
                                                  R=10.,
                                                  ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # TriaxialGaussianPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.TriaxialGaussianPotential(amp=4.*10.**10.*units.Msun,
                                                 sigma=5.,
                                                 ro=ro,vo=vo,
                                                 b=1.3,c=0.4)
    check_apy_strict_inputs_error_msg(excinfo,5.,'length')
    # KingPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KingPotential(W0=3.,M=4.*10.**6.,
                                     rt=10.*units.pc,
                                     ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,4.*10.**6.,'mass')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.KingPotential(W0=3.,M=4.*10.**6.*units.Msun,
                                     rt=10.,
                                     ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'length')
    # AnyAxisymmetricRazorThinDiskPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.AnyAxisymmetricRazorThinDiskPotential(\
                                                             surfdens=lambda R: 1.5*conversion.surfdens_in_msolpc2(vo.to_value(units.km/units.s),ro.to_value(units.kpc))
                                                             *numpy.exp(-R),
                                                             ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'surfacedensity',weak=True)
    # AnySphericalPotential
    with pytest.raises(ValueError) as excinfo:
        pot= potential.AnySphericalPotential(
            dens=lambda r: 0.64/r/(1+r)**3*conversion.dens_in_msolpc3(vo.to_value(units.km/units.s),ro.to_value(units.kpc))\
            ,
                                                         ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,10.,'density',weak=True)
    # RotateAndTiltWrapperPotential, zvec, pa
    wrappot= potential.TriaxialNFWPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo,
                                        b=1.3,c=0.4)
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RotateAndTiltWrapperPotential(pot=wrappot,zvec=[0,1.,0],galaxy_pa=30.,
                                                 ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,30.,'angle')
    # RotateAndTiltWrapperPotential, inclination, galaxy_pa, sky_pa
    wrappot= potential.TriaxialNFWPotential(amp=20.*units.Msun,a=2.*units.kpc,ro=ro,vo=vo,
                                        b=1.3,c=0.4)
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RotateAndTiltWrapperPotential(pot=wrappot,galaxy_pa=30.,
                                                     inclination=60.*units.deg,sky_pa=-45.*units.deg,
                                                 ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,30.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RotateAndTiltWrapperPotential(pot=wrappot,galaxy_pa=30.*units.deg,
                                                     inclination=60.,sky_pa=-45.*units.deg,
                                                 ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,60.,'angle')
    with pytest.raises(ValueError) as excinfo:
        pot= potential.RotateAndTiltWrapperPotential(pot=wrappot,galaxy_pa=30.*units.deg,
                                                     inclination=60.*units.deg,sky_pa=-45.,
                                                 ro=ro,vo=vo)
    check_apy_strict_inputs_error_msg(excinfo,-45.,'angle')
    return None


