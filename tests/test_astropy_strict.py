# Tests of the astropy strict mode that requires non-dimensionless inputs
# to be given as quantities and all outputs to have units turned on
import pytest
from galpy.util import config
# Don't set astropy_strict here, because then it affects all test files run
# through pytest at the same time
import numpy
from astropy import units, constants

def check_apy_strict_inputs_error_msg(excinfo,input_value,input_type):
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
    # pot= potential.SpiralArmsPotential(amp=0.3*units.Msun / units.pc**3)
    # assert numpy.fabs(pot(1.,0.,phi=1.,use_physical=False)*) < 10.**-8., "SpiralArmsPotential w/ amp w/ units does not behave as expected"
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


