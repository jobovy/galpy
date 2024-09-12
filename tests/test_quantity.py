# Make sure to set configuration, needs to be before any galpy imports
import pytest
from packaging.version import parse as parse_version

from galpy.util import config

config.__config__.set("astropy", "astropy-units", "True")
import numpy

_NUMPY_VERSION = parse_version(numpy.__version__)
_NUMPY_1_22 = (_NUMPY_VERSION > parse_version("1.21")) * (
    _NUMPY_VERSION < parse_version("1.23")
) + (_NUMPY_VERSION > parse_version("1.23")) * (
    _NUMPY_VERSION < parse_version("1.25")
)  # For testing 1.22/1.24 precision issues
from astropy import constants, units

sdf_sanders15 = None  # so we can set this up and then use in other tests
sdf_sanders15_nou = None  # so we can set this up and then use in other tests


def test_parsers():
    from galpy.util import conversion

    # Unitless
    assert (
        numpy.fabs(conversion.parse_length(2.0) - 2.0) < 1e-10
    ), "parse_length does not parse unitless position correctly"
    assert (
        numpy.fabs(conversion.parse_energy(3.0) - 3.0) < 1e-10
    ), "parse_energy does not parse unitless energy correctly"
    assert (
        numpy.fabs(conversion.parse_angmom(-1.5) + 1.5) < 1e-10
    ), "parse_angmom does not parse unitless angular momentum correctly"
    # Quantity input
    ro, vo = 7.0, 230.0
    assert (
        numpy.fabs(
            conversion.parse_length(2.0 * units.parsec, ro=ro, vo=vo) - (0.002 / ro)
        )
        < 1e-10
    ), "parse_length does parse Quantity position correctly"
    assert (
        numpy.fabs(
            conversion.parse_energy(-30.0 * units.km**2 / units.s**2, ro=ro, vo=vo)
            - (-30.0 / vo**2)
        )
        < 1e-10
    ), "parse_energy does parse Quantity energy correctly"
    assert (
        numpy.fabs(
            conversion.parse_angmom(
                2200.0 * units.kpc * units.km / units.s, ro=ro, vo=vo
            )
            - (2200.0 / ro / vo)
        )
        < 1e-10
    ), "parse_angmom does parse Quantity angular momentum correctly"
    return None


def test_parsers_with_unrecognized_inputs():
    # Test related to $542: test that an error is raised when parsing an object
    # that is not a float/... or an astropy Quantity (e.g., a different unit system)
    from galpy.util import conversion

    # Just some object
    class other_quantity_object:
        def __init__(self):
            return None

    obj = other_quantity_object()
    ro, vo = 7.0, 230.0
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_length(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_length_kpc(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_velocity(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_velocity_kms(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_angle(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_time(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_mass(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_energy(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_angmom(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_frequency(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_force(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_dens(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_surfdens(obj, ro=ro, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_numdens(obj, ro=ro, vo=vo)
    return None


def test_parsers_rovo_input():
    # Test that providing ro in kpc and vo in km/s to the parsers works
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    assert (
        numpy.fabs(
            conversion.parse_length(2.0 * units.parsec, ro=ro, vo=vo)
            - conversion.parse_length(
                2.0 * units.parsec, ro=ro * units.kpc, vo=vo * units.km / units.s
            )
        )
        < 1e-10
    ), "parse_length does parse Quantity position correctly when specifying ro and vo as Quantities"
    assert (
        numpy.fabs(
            conversion.parse_energy(-30.0 * units.km**2 / units.s**2, ro=ro, vo=vo)
            - conversion.parse_energy(
                -30.0 * units.km**2 / units.s**2,
                ro=(ro * units.kpc).to(units.m),
                vo=(vo * units.km / units.s).to(units.pc / units.Myr),
            )
        )
        < 1e-10
    ), "parse_energy does parse Quantity energy correctly when specifying ro and vo as Quantities"
    return None


def test_parsers_rovo_wronginputtype():
    # Test that giving ro and vo that can't be understood gives an error
    from galpy.util import conversion

    # Just some object
    class other_quantity_object:
        def __init__(self):
            return None

    obj = other_quantity_object()
    ro, vo = 7.0, 230.0
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_length(8.0 * units.kpc, ro=obj, vo=vo)
    with pytest.raises(
        RuntimeError, match="should either be a number or an astropy Quantity"
    ):
        assert conversion.parse_length(8.0 * units.kpc, ro=ro, vo=obj)
    return None


def test_warn_internal_when_use_physical():
    import warnings

    from galpy import potential
    from galpy.util import galpyWarning

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        potential.evaluateRforces(
            potential.MWPotential2014, 1.0, 0.0, use_physical=True
        )
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "Returning output(s) in internal units even though use_physical=True, because ro and/or vo not set"
            )
            if raisedWarning:
                break
        assert (
            raisedWarning
        ), "No warning raised when returning internal-units with use_physical=True"
    return None


def test_orbit_setup_radec_basic():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.deg,
            -20.0 * units.deg,
            3.0 * units.kpc,
            -3.0 * units.mas / units.yr,
            2.0 * units.mas / units.yr,
            130.0 * units.km / units.s,
        ],
        radec=True,
    )
    assert (
        numpy.fabs(o.ra(quantity=False) - 10.0) < 10.0**-8.0
    ), "Orbit initialization with RA as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dec(quantity=False) + 20.0) < 10.0**-8.0
    ), "Orbit initialization with Dec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.pmra(quantity=False) + 3.0) < 10.0**-8.0
    ), "Orbit initialization with pmra as Quantity does not work as expected"
    assert (
        numpy.fabs(o.pmdec(quantity=False) - 2.0) < 10.0**-8.0
    ), "Orbit initialization with pmdec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.vlos(quantity=False) - 130.0) < 10.0**-8.0
    ), "Orbit initialization with vlos as Quantity does not work as expected"
    return None


def test_orbit_setup_radec_oddunits():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.lyr,
            -3.0 * units.mas / units.s,
            2.0 * units.mas / units.kyr,
            130.0 * units.pc / units.Myr,
        ],
        radec=True,
    )
    assert (
        numpy.fabs(o.ra(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with RA as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dec(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with Dec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0 / 3.26156) < 10.0**-5.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(
            (o.pmra(quantity=False) + 3.0 * units.yr.to(units.s))
            / o.pmra(quantity=False)
        )
        < 10.0**-8.0
    ), "Orbit initialization with pmra as Quantity does not work as expected"
    assert (
        numpy.fabs(
            (o.pmdec(quantity=False) - 2.0 / 10.0**3.0) / o.pmdec(quantity=False)
        )
        < 10.0**-4.0
    ), "Orbit initialization with pmdec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.vlos(quantity=False) - 130.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with vlos as Quantity does not work as expected"
    return None


def test_orbit_setup_radec_uvw():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.pc,
            -30.0 * units.km / units.s,
            20.0 * units.km / units.s,
            130.0 * units.km / units.s,
        ],
        radec=True,
        uvw=True,
    )
    assert (
        numpy.fabs(o.ra(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with RA as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dec(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with Dec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.U(quantity=False) + 30.0) < 10.0**-8.0
    ), "Orbit initialization with U as Quantity does not work as expected"
    assert (
        numpy.fabs(o.V(quantity=False) - 20.0) < 10.0**-8.0
    ), "Orbit initialization with V as Quantity does not work as expected"
    assert (
        numpy.fabs(o.W(quantity=False) - 130.0) < 10.0**-8.0
    ), "Orbit initialization with W as Quantity does not work as expected"
    return None


def test_orbit_setup_radec_uvw_oddunits():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.pc,
            -30.0 * units.pc / units.Myr,
            20.0 * units.pc / units.Myr,
            130.0 * units.pc / units.Myr,
        ],
        radec=True,
        uvw=True,
    )
    assert (
        numpy.fabs(o.ra(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with RA as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dec(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with Dec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.U(quantity=False) + 30.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with U as Quantity does not work as expected"
    assert (
        numpy.fabs(o.V(quantity=False) - 20.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with V as Quantity does not work as expected"
    assert (
        numpy.fabs(o.W(quantity=False) - 130.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with W as Quantity does not work as expected"
    return None


def test_orbit_setup_lb_basic():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.deg,
            -20.0 * units.deg,
            3.0 * units.kpc,
            -3.0 * units.mas / units.yr,
            2.0 * units.mas / units.yr,
            130.0 * units.km / units.s,
        ],
        lb=True,
    )
    assert (
        numpy.fabs(o.ll(quantity=False) - 10.0) < 10.0**-8.0
    ), "Orbit initialization with ll as Quantity does not work as expected"
    assert (
        numpy.fabs(o.bb(quantity=False) + 20.0) < 10.0**-8.0
    ), "Orbit initialization with bb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.pmll(quantity=False) + 3.0) < 10.0**-8.0
    ), "Orbit initialization with pmra as Quantity does not work as expected"
    assert (
        numpy.fabs(o.pmbb(quantity=False) - 2.0) < 10.0**-8.0
    ), "Orbit initialization with pmdec as Quantity does not work as expected"
    assert (
        numpy.fabs(o.vlos(quantity=False) - 130.0) < 10.0**-8.0
    ), "Orbit initialization with vlos as Quantity does not work as expected"
    return None


def test_orbit_setup_lb_oddunits():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.lyr,
            -3.0 * units.mas / units.s,
            2.0 * units.mas / units.kyr,
            130.0 * units.pc / units.Myr,
        ],
        lb=True,
    )
    assert (
        numpy.fabs(o.ll(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with ll as Quantity does not work as expected"
    assert (
        numpy.fabs(o.bb(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with bb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0 / 3.26156) < 10.0**-5.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(
            (o.pmll(quantity=False) + 3.0 * units.yr.to(units.s))
            / o.pmll(quantity=False)
        )
        < 10.0**-8.0
    ), "Orbit initialization with pmll as Quantity does not work as expected"
    assert (
        numpy.fabs((o.pmbb(quantity=False) - 2.0 / 10.0**3.0) / o.pmbb(quantity=False))
        < 10.0**-4.0
    ), "Orbit initialization with pmbb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.vlos(quantity=False) - 130.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with vlos as Quantity does not work as expected"
    return None


def test_orbit_setup_lb_uvw():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.pc,
            -30.0 * units.km / units.s,
            20.0 * units.km / units.s,
            130.0 * units.km / units.s,
        ],
        lb=True,
        uvw=True,
    )
    assert (
        numpy.fabs(o.ll(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with ll as Quantity does not work as expected"
    assert (
        numpy.fabs(o.bb(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with bb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.U(quantity=False) + 30.0) < 10.0**-8.0
    ), "Orbit initialization with pmll as Quantity does not work as expected"
    assert (
        numpy.fabs(o.V(quantity=False) - 20.0) < 10.0**-8.0
    ), "Orbit initialization with pmbb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.W(quantity=False) - 130.0) < 10.0**-8.0
    ), "Orbit initialization with W as Quantity does not work as expected"
    return None


def test_orbit_setup_lb_uvw_oddunits():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            1.0 * units.rad,
            -0.25 * units.rad,
            3000.0 * units.pc,
            -30.0 * units.pc / units.Myr,
            20.0 * units.pc / units.Myr,
            130.0 * units.pc / units.Myr,
        ],
        lb=True,
        uvw=True,
    )
    assert (
        numpy.fabs(o.ll(quantity=False) - 1.0 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with ll as Quantity does not work as expected"
    assert (
        numpy.fabs(o.bb(quantity=False) + 0.25 / numpy.pi * 180.0) < 10.0**-8.0
    ), "Orbit initialization with bb as Quantity does not work as expected"
    assert (
        numpy.fabs(o.dist(quantity=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with distance as Quantity does not work as expected"
    assert (
        numpy.fabs(o.U(quantity=False) + 30.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with U as Quantity does not work as expected"
    assert (
        numpy.fabs(o.V(quantity=False) - 20.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with V as Quantity does not work as expected"
    assert (
        numpy.fabs(o.W(quantity=False) - 130.0 / 1.0227121655399913) < 10.0**-5.0
    ), "Orbit initialization with W as Quantity does not work as expected"
    return None


def test_orbit_setup_vxvv_fullorbit():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert (
        numpy.fabs(o.R(use_physical=False) * o._ro - 10.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    assert (
        numpy.fabs(o.vR(use_physical=False) * o._vo + 20.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    assert (
        numpy.fabs(o.vT(use_physical=False) * o._vo - 210.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    assert (
        numpy.fabs(o.z(use_physical=False) * o._ro - 0.5) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    assert (
        numpy.fabs(o.vz(use_physical=False) * o._vo + 12) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    assert (
        numpy.fabs(o.phi(use_physical=False) - 45.0 / 180.0 * numpy.pi) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    return None


def test_orbit_setup_vxvv_rzorbit():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10000.0 * units.lyr,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.pc / units.Myr,
        ]
    )
    assert (
        numpy.fabs(o.R(use_physical=False) * o._ro - 10.0 / 3.26156) < 10.0**-5.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vR(use_physical=False) * o._vo + 20.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vT(use_physical=False) * o._vo - 210.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.z(use_physical=False) * o._ro - 0.5) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vz(use_physical=False) * o._vo + 12.0 / 1.0227121655399913)
        < 10.0**-5.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    return None


def test_orbit_setup_vxvv_planarorbit():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10000.0 * units.lyr,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            3.0 * units.rad,
        ]
    )
    assert (
        numpy.fabs(o.R(use_physical=False) * o._ro - 10.0 / 3.26156) < 10.0**-5.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vR(use_physical=False) * o._vo + 20.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vT(use_physical=False) * o._vo - 210.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.phi(use_physical=False) - 3.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for FullOrbit"
    return None


def test_orbit_setup_vxvv_planarrorbit():
    from galpy.orbit import Orbit

    o = Orbit(
        [7.0 * units.kpc, -2.0 * units.km / units.s, 210.0 * units.km / units.s],
        ro=10.0,
        vo=150.0,
    )
    assert (
        numpy.fabs(o.R(use_physical=False) * o._ro - 7.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vR(use_physical=False) * o._vo + 2.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vT(use_physical=False) * o._vo - 210.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    return None


def test_orbit_setup_vxvv_linearorbit():
    from galpy.orbit import Orbit

    o = Orbit([7.0 * units.kpc, -21.0 * units.pc / units.Myr])
    assert (
        numpy.fabs(o.x(use_physical=False) * o._ro - 7.0) < 10.0**-8.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    assert (
        numpy.fabs(o.vx(use_physical=False) * o._vo + 21.0 / 1.0227121655399913)
        < 10.0**-5.0
    ), "Orbit initialization with vxvv as Quantity does not work as expected for RZOrbit"
    return None


def test_orbit_setup_solarmotion():
    from galpy.orbit import Orbit

    o = Orbit(
        [1.0, 0.1, 1.1, 0.2, 0.1, 0.0],
        solarmotion=units.Quantity([13.0, 25.0, 8.0], unit=units.km / units.s),
    )
    assert (
        numpy.fabs(o._solarmotion[0] - 13.0) < 10.0**-8.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._solarmotion[1] - 25.0) < 10.0**-8.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._solarmotion[2] - 8.0) < 10.0**-8.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_solarmotion_oddunits():
    from galpy.orbit import Orbit

    o = Orbit(
        [1.0, 0.1, 1.1, 0.2, 0.1, 0.0],
        solarmotion=units.Quantity([13.0, 25.0, 8.0], unit=units.kpc / units.Gyr),
    )
    assert (
        numpy.fabs(o._solarmotion[0] - 13.0 / 1.0227121655399913) < 10.0**-5.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._solarmotion[1] - 25.0 / 1.0227121655399913) < 10.0**-5.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._solarmotion[2] - 8.0 / 1.0227121655399913) < 10.0**-5.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_roAsQuantity():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], ro=11 * units.kpc)
    assert (
        numpy.fabs(o._ro - 11.0) < 10.0**-10.0
    ), "ro in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._ro - 11.0) < 10.0**-10.0
    ), "ro in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_roAsQuantity_oddunits():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], ro=11 * units.lyr)
    assert (
        numpy.fabs(o._ro - 11.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._ro - 11.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_voAsQuantity():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], vo=210 * units.km / units.s)
    assert (
        numpy.fabs(o._vo - 210.0) < 10.0**-10.0
    ), "vo in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._vo - 210.0) < 10.0**-10.0
    ), "vo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_voAsQuantity_oddunits():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], vo=210 * units.pc / units.Myr)
    assert (
        numpy.fabs(o._vo - 210.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in Orbit setup as Quantity does not work as expected"
    assert (
        numpy.fabs(o._vo - 210.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_zoAsQuantity():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], zo=12 * units.pc)
    assert (
        numpy.fabs(o._zo - 0.012) < 10.0**-10.0
    ), "zo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_setup_zoAsQuantity_oddunits():
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.2, 0.1, 0.0], zo=13 * units.lyr)
    assert (
        numpy.fabs(o._zo - 13.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "zo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbit_method_returntype_scalar():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    from galpy.potential import MWPotential2014

    assert isinstance(
        o.E(pot=MWPotential2014), units.Quantity
    ), "Orbit method E does not return Quantity when it should"
    assert isinstance(
        o.ER(pot=MWPotential2014), units.Quantity
    ), "Orbit method ER does not return Quantity when it should"
    assert isinstance(
        o.Ez(pot=MWPotential2014), units.Quantity
    ), "Orbit method Ez does not return Quantity when it should"
    assert isinstance(
        o.Jacobi(pot=MWPotential2014), units.Quantity
    ), "Orbit method Jacobi does not return Quantity when it should"
    assert isinstance(
        o.L(), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.Lz(), units.Quantity
    ), "Orbit method Lz does not return Quantity when it should"
    assert isinstance(
        o.rap(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method rap does not return Quantity when it should"
    assert isinstance(
        o.rperi(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method rperi does not return Quantity when it should"
    assert isinstance(
        o.rguiding(pot=MWPotential2014), units.Quantity
    ), "Orbit method rguiding does not return Quantity when it should"
    assert isinstance(
        o.rE(pot=MWPotential2014), units.Quantity
    ), "Orbit method rE does not return Quantity when it should"
    assert isinstance(
        o.LcE(pot=MWPotential2014), units.Quantity
    ), "Orbit method LcE does not return Quantity when it should"
    assert isinstance(
        o.zmax(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method zmax does not return Quantity when it should"
    assert isinstance(
        o.jr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jr does not return Quantity when it should"
    assert isinstance(
        o.jp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jp does not return Quantity when it should"
    assert isinstance(
        o.jz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jz does not return Quantity when it should"
    assert isinstance(
        o.wr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wr does not return Quantity when it should"
    assert isinstance(
        o.wp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wp does not return Quantity when it should"
    assert isinstance(
        o.wz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wz does not return Quantity when it should"
    assert isinstance(
        o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tr does not return Quantity when it should"
    assert isinstance(
        o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tp does not return Quantity when it should"
    assert isinstance(
        o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tz does not return Quantity when it should"
    assert isinstance(
        o.Or(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Or does not return Quantity when it should"
    assert isinstance(
        o.Op(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Op does not return Quantity when it should"
    assert isinstance(
        o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Oz does not return Quantity when it should"
    assert isinstance(
        o.time(), units.Quantity
    ), "Orbit method time does not return Quantity when it should"
    assert isinstance(
        o.R(), units.Quantity
    ), "Orbit method R does not return Quantity when it should"
    assert isinstance(
        o.r(), units.Quantity
    ), "Orbit method r does not return Quantity when it should"
    assert isinstance(
        o.vR(), units.Quantity
    ), "Orbit method vR does not return Quantity when it should"
    assert isinstance(
        o.vT(), units.Quantity
    ), "Orbit method vT does not return Quantity when it should"
    assert isinstance(
        o.z(), units.Quantity
    ), "Orbit method z does not return Quantity when it should"
    assert isinstance(
        o.vz(), units.Quantity
    ), "Orbit method vz does not return Quantity when it should"
    assert isinstance(
        o.phi(), units.Quantity
    ), "Orbit method phi does not return Quantity when it should"
    assert isinstance(
        o.vphi(), units.Quantity
    ), "Orbit method vphi does not return Quantity when it should"
    assert isinstance(
        o.x(), units.Quantity
    ), "Orbit method x does not return Quantity when it should"
    assert isinstance(
        o.y(), units.Quantity
    ), "Orbit method y does not return Quantity when it should"
    assert isinstance(
        o.vx(), units.Quantity
    ), "Orbit method vx does not return Quantity when it should"
    assert isinstance(
        o.vy(), units.Quantity
    ), "Orbit method vy does not return Quantity when it should"
    assert isinstance(
        o.ra(), units.Quantity
    ), "Orbit method ra does not return Quantity when it should"
    assert isinstance(
        o.dec(), units.Quantity
    ), "Orbit method dec does not return Quantity when it should"
    assert isinstance(
        o.ll(), units.Quantity
    ), "Orbit method ll does not return Quantity when it should"
    assert isinstance(
        o.bb(), units.Quantity
    ), "Orbit method bb does not return Quantity when it should"
    assert isinstance(
        o.dist(), units.Quantity
    ), "Orbit method dist does not return Quantity when it should"
    assert isinstance(
        o.pmra(), units.Quantity
    ), "Orbit method pmra does not return Quantity when it should"
    assert isinstance(
        o.pmdec(), units.Quantity
    ), "Orbit method pmdec does not return Quantity when it should"
    assert isinstance(
        o.pmll(), units.Quantity
    ), "Orbit method pmll does not return Quantity when it should"
    assert isinstance(
        o.pmbb(), units.Quantity
    ), "Orbit method pmbb does not return Quantity when it should"
    assert isinstance(
        o.vlos(), units.Quantity
    ), "Orbit method vlos does not return Quantity when it should"
    assert isinstance(
        o.vra(), units.Quantity
    ), "Orbit method vra does not return Quantity when it should"
    assert isinstance(
        o.vdec(), units.Quantity
    ), "Orbit method vdec does not return Quantity when it should"
    assert isinstance(
        o.vll(), units.Quantity
    ), "Orbit method vll does not return Quantity when it should"
    assert isinstance(
        o.vbb(), units.Quantity
    ), "Orbit method vbb does not return Quantity when it should"
    assert isinstance(
        o.helioX(), units.Quantity
    ), "Orbit method helioX does not return Quantity when it should"
    assert isinstance(
        o.helioY(), units.Quantity
    ), "Orbit method helioY does not return Quantity when it should"
    assert isinstance(
        o.helioZ(), units.Quantity
    ), "Orbit method helioZ does not return Quantity when it should"
    assert isinstance(
        o.U(), units.Quantity
    ), "Orbit method U does not return Quantity when it should"
    assert isinstance(
        o.V(), units.Quantity
    ), "Orbit method V does not return Quantity when it should"
    assert isinstance(
        o.W(), units.Quantity
    ), "Orbit method W does not return Quantity when it should"
    return None


def test_orbit_method_returntype():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    from galpy.potential import MWPotential2014

    ts = numpy.linspace(0.0, 6.0, 1001)
    o.integrate(ts, MWPotential2014)
    assert isinstance(
        o.E(ts), units.Quantity
    ), "Orbit method E does not return Quantity when it should"
    assert isinstance(
        o.ER(ts), units.Quantity
    ), "Orbit method ER does not return Quantity when it should"
    assert isinstance(
        o.Ez(ts), units.Quantity
    ), "Orbit method Ez does not return Quantity when it should"
    assert isinstance(
        o.Jacobi(ts), units.Quantity
    ), "Orbit method Jacobi does not return Quantity when it should"
    assert isinstance(
        o.L(ts), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.Lz(ts), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.time(ts), units.Quantity
    ), "Orbit method time does not return Quantity when it should"
    assert isinstance(
        o.R(ts), units.Quantity
    ), "Orbit method R does not return Quantity when it should"
    assert isinstance(
        o.r(ts), units.Quantity
    ), "Orbit method r does not return Quantity when it should"
    assert isinstance(
        o.vR(ts), units.Quantity
    ), "Orbit method vR does not return Quantity when it should"
    assert isinstance(
        o.vT(ts), units.Quantity
    ), "Orbit method vT does not return Quantity when it should"
    assert isinstance(
        o.z(ts), units.Quantity
    ), "Orbit method z does not return Quantity when it should"
    assert isinstance(
        o.vz(ts), units.Quantity
    ), "Orbit method vz does not return Quantity when it should"
    assert isinstance(
        o.phi(ts), units.Quantity
    ), "Orbit method phi does not return Quantity when it should"
    assert isinstance(
        o.vphi(ts), units.Quantity
    ), "Orbit method vphi does not return Quantity when it should"
    assert isinstance(
        o.x(ts), units.Quantity
    ), "Orbit method x does not return Quantity when it should"
    assert isinstance(
        o.y(ts), units.Quantity
    ), "Orbit method y does not return Quantity when it should"
    assert isinstance(
        o.vx(ts), units.Quantity
    ), "Orbit method vx does not return Quantity when it should"
    assert isinstance(
        o.vy(ts), units.Quantity
    ), "Orbit method vy does not return Quantity when it should"
    assert isinstance(
        o.ra(ts), units.Quantity
    ), "Orbit method ra does not return Quantity when it should"
    assert isinstance(
        o.dec(ts), units.Quantity
    ), "Orbit method dec does not return Quantity when it should"
    assert isinstance(
        o.ll(ts), units.Quantity
    ), "Orbit method ll does not return Quantity when it should"
    assert isinstance(
        o.bb(ts), units.Quantity
    ), "Orbit method bb does not return Quantity when it should"
    assert isinstance(
        o.dist(ts), units.Quantity
    ), "Orbit method dist does not return Quantity when it should"
    assert isinstance(
        o.pmra(ts), units.Quantity
    ), "Orbit method pmra does not return Quantity when it should"
    assert isinstance(
        o.pmdec(ts), units.Quantity
    ), "Orbit method pmdec does not return Quantity when it should"
    assert isinstance(
        o.pmll(ts), units.Quantity
    ), "Orbit method pmll does not return Quantity when it should"
    assert isinstance(
        o.pmbb(ts), units.Quantity
    ), "Orbit method pmbb does not return Quantity when it should"
    assert isinstance(
        o.vlos(ts), units.Quantity
    ), "Orbit method vlos does not return Quantity when it should"
    assert isinstance(
        o.vra(ts), units.Quantity
    ), "Orbit method vra does not return Quantity when it should"
    assert isinstance(
        o.vdec(ts), units.Quantity
    ), "Orbit method vdec does not return Quantity when it should"
    assert isinstance(
        o.vll(ts), units.Quantity
    ), "Orbit method vll does not return Quantity when it should"
    assert isinstance(
        o.vbb(ts), units.Quantity
    ), "Orbit method vbb does not return Quantity when it should"
    assert isinstance(
        o.helioX(ts), units.Quantity
    ), "Orbit method helioX does not return Quantity when it should"
    assert isinstance(
        o.helioY(ts), units.Quantity
    ), "Orbit method helioY does not return Quantity when it should"
    assert isinstance(
        o.helioZ(ts), units.Quantity
    ), "Orbit method helioZ does not return Quantity when it should"
    assert isinstance(
        o.U(ts), units.Quantity
    ), "Orbit method U does not return Quantity when it should"
    assert isinstance(
        o.V(ts), units.Quantity
    ), "Orbit method V does not return Quantity when it should"
    assert isinstance(
        o.W(ts), units.Quantity
    ), "Orbit method W does not return Quantity when it should"
    return None


def test_orbit_method_returnunit():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    from galpy.potential import MWPotential2014

    try:
        o.E(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method E does not return Quantity with the right units"
        )
    try:
        o.ER(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ER does not return Quantity with the right units"
        )
    try:
        o.Ez(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Ez does not return Quantity with the right units"
        )
    try:
        o.Jacobi(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Jacobi does not return Quantity with the right units"
        )
    try:
        o.L().to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method L does not return Quantity with the right units"
        )
    try:
        o.Lz().to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Lz does not return Quantity with the right units"
        )
    try:
        o.rap(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rap does not return Quantity with the right units"
        )
    try:
        o.rperi(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rperi does not return Quantity with the right units"
        )
    try:
        o.rguiding(pot=MWPotential2014).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rguiding does not return Quantity with the right units"
        )
    try:
        o.rE(pot=MWPotential2014).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rE does not return Quantity with the right units"
        )
    try:
        o.LcE(pot=MWPotential2014).to(units.kpc * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method LcE does not return Quantity with the right units"
        )
    try:
        o.zmax(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method zmax does not return Quantity with the right units"
        )
    try:
        o.jr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jr does not return Quantity with the right units"
        )
    try:
        o.jp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jp does not return Quantity with the right units"
        )
    try:
        o.jz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jz does not return Quantity with the right units"
        )
    try:
        o.wr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wr does not return Quantity with the right units"
        )
    try:
        o.wp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wp does not return Quantity with the right units"
        )
    try:
        o.wz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wz does not return Quantity with the right units"
        )
    try:
        o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tr does not return Quantity with the right units"
        )
    try:
        o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tp does not return Quantity with the right units"
        )
    try:
        o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tz does not return Quantity with the right units"
        )
    try:
        o.Or(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Or does not return Quantity with the right units"
        )
    try:
        o.Op(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Op does not return Quantity with the right units"
        )
    try:
        o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Oz does not return Quantity with the right units"
        )
    try:
        o.time().to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method time does not return Quantity with the right units"
        )
    try:
        o.R().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method R does not return Quantity with the right units"
        )
    try:
        o.r().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method r does not return Quantity with the right units"
        )
    try:
        o.vR().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vR does not return Quantity with the right units"
        )
    try:
        o.vT().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vT does not return Quantity with the right units"
        )
    try:
        o.z().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method z does not return Quantity with the right units"
        )
    try:
        o.vz().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vz does not return Quantity with the right units"
        )
    try:
        o.phi().to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method phi does not return Quantity with the right units"
        )
    try:
        o.vphi().to(units.km / units.s / units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vphi does not return Quantity with the right units"
        )
    try:
        o.x().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method x does not return Quantity with the right units"
        )
    try:
        o.y().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method y does not return Quantity with the right units"
        )
    try:
        o.vx().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vx does not return Quantity with the right units"
        )
    try:
        o.vy().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vy does not return Quantity with the right units"
        )
    try:
        o.ra().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ra does not return Quantity with the right units"
        )
    try:
        o.dec().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method dec does not return Quantity with the right units"
        )
    try:
        o.ll().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ll does not return Quantity with the right units"
        )
    try:
        o.bb().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method bb does not return Quantity with the right units"
        )
    try:
        o.dist().to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method dist does not return Quantity with the right units"
        )
    try:
        o.pmra().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmra does not return Quantity with the right units"
        )
    try:
        o.pmdec().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmdec does not return Quantity with the right units"
        )
    try:
        o.pmll().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmll does not return Quantity with the right units"
        )
    try:
        o.pmbb().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmbb does not return Quantity with the right units"
        )
    try:
        o.vlos().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vlos does not return Quantity with the right units"
        )
    try:
        o.vra().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vra does not return Quantity with the right units"
        )
    try:
        o.vdec().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vdec does not return Quantity with the right units"
        )
    try:
        o.vll().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vll does not return Quantity with the right units"
        )
    try:
        o.vbb().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vbb does not return Quantity with the right units"
        )
    try:
        o.helioX().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioX does not return Quantity with the right units"
        )
    try:
        o.helioY().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioY does not return Quantity with the right units"
        )
    try:
        o.helioZ().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioZ does not return Quantity with the right units"
        )
    try:
        o.U().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method U does not return Quantity with the right units"
        )
    try:
        o.V().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method V does not return Quantity with the right units"
        )
    try:
        o.W().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method W does not return Quantity with the right units"
        )
    return None


def test_orbit_method_value():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    oc = o()
    oc.turn_physical_off()
    assert (
        numpy.fabs(
            o.E(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.E(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.ER(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.ER(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Ez(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.Ez(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.Jacobi(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014).to(units.km / units.s * units.kpc).value
            - oc.L(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Lz(pot=MWPotential2014).to(units.km / units.s * units.kpc).value
            - oc.Lz(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.rap(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.rperi(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.rguiding(pot=MWPotential2014).to(units.kpc).value
            - oc.rguiding(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rguiding does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.rE(pot=MWPotential2014).to(units.kpc).value
            - oc.rE(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rE does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.LcE(pot=MWPotential2014).to(units.kpc * units.km / units.s).value
            - oc.LcE(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method LcE does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.zmax(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.jr(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jr(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.jp(pot=MWPotential2014, type="staeckel", delta=4.0 * units.kpc)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jp(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8 * 8.0 * units.kpc)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.wr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wr(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.wp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wp(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.wz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wz(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tr(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tp(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.time().to(units.Gyr).value
            - oc.time() * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method time does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.R().to(units.kpc).value - oc.R() * o._ro) < 10.0**-8.0
    ), "Orbit method R does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.r().to(units.kpc).value - oc.r() * o._ro) < 10.0**-8.0
    ), "Orbit method r does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vR().to(units.km / units.s).value - oc.vR() * o._vo) < 10.0**-8.0
    ), "Orbit method vR does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vT().to(units.km / units.s).value - oc.vT() * o._vo) < 10.0**-8.0
    ), "Orbit method vT does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.z().to(units.kpc).value - oc.z() * o._ro) < 10.0**-8.0
    ), "Orbit method z does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vz().to(units.km / units.s).value - oc.vz() * o._vo) < 10.0**-8.0
    ), "Orbit method vz does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.phi().to(units.rad).value - oc.phi()) < 10.0**-8.0
    ), "Orbit method phi does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            o.vphi().to(units.km / units.s / units.kpc).value
            - oc.vphi() * o._vo / o._ro
        )
        < 10.0**-8.0
    ), "Orbit method vphi does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.x().to(units.kpc).value - oc.x() * o._ro) < 10.0**-8.0
    ), "Orbit method x does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.y().to(units.kpc).value - oc.y() * o._ro) < 10.0**-8.0
    ), "Orbit method y does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vx().to(units.km / units.s).value - oc.vx() * o._vo) < 10.0**-8.0
    ), "Orbit method vx does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vy().to(units.km / units.s).value - oc.vy() * o._vo) < 10.0**-8.0
    ), "Orbit method vy does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.ra().to(units.deg).value - oc.ra(quantity=False)) < 10.0**-8.0
    ), "Orbit method ra does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.dec().to(units.deg).value - oc.dec(quantity=False)) < 10.0**-8.0
    ), "Orbit method dec does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.ll().to(units.deg).value - oc.ll(quantity=False)) < 10.0**-8.0
    ), "Orbit method ll does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.bb().to(units.deg).value - oc.bb(quantity=False)) < 10.0**-8.0
    ), "Orbit method bb does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.dist().to(units.kpc).value - oc.dist(quantity=False)) < 10.0**-8.0
    ), "Orbit method dist does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.pmra().to(units.mas / units.yr).value - oc.pmra(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmra does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.pmdec().to(units.mas / units.yr).value - oc.pmdec(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmdec does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.pmll().to(units.mas / units.yr).value - oc.pmll(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmll does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.pmbb().to(units.mas / units.yr).value - oc.pmbb(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmbb does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vlos().to(units.km / units.s).value - oc.vlos(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vlos does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vra().to(units.km / units.s).value - oc.vra(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vra does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vdec().to(units.km / units.s).value - oc.vdec(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vdec does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vll().to(units.km / units.s).value - oc.vll(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vll does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.vbb().to(units.km / units.s).value - oc.vbb(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vbb does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.helioX().to(units.kpc).value - oc.helioX(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioX does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.helioY().to(units.kpc).value - oc.helioY(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioY does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.helioZ().to(units.kpc).value - oc.helioZ(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioZ does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.U().to(units.km / units.s).value - oc.U(quantity=False))
        < 10.0**-8.0
    ), "Orbit method U does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.V().to(units.km / units.s).value - oc.V(quantity=False))
        < 10.0**-8.0
    ), "Orbit method V does not return the correct value as Quantity"
    assert (
        numpy.fabs(o.W().to(units.km / units.s).value - oc.W(quantity=False))
        < 10.0**-8.0
    ), "Orbit method W does not return the correct value as Quantity"
    return None


def test_orbit_method_value_turnquantityoff():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    oc = o()
    oc.turn_physical_off()
    assert (
        numpy.fabs(
            o.E(pot=MWPotential2014, quantity=False)
            - oc.E(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.ER(pot=MWPotential2014, quantity=False)
            - oc.ER(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Ez(pot=MWPotential2014, quantity=False)
            - oc.Ez(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014, quantity=False)
            - oc.Jacobi(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014, quantity=False)
            - oc.L(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Lz(pot=MWPotential2014, quantity=False)
            - oc.Lz(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.rap(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.rperi(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.rguiding(pot=MWPotential2014, quantity=False)
            - oc.rguiding(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rguiding does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.rE(pot=MWPotential2014, quantity=False)
            - oc.rE(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rE does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.LcE(pot=MWPotential2014, quantity=False)
            - oc.LcE(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method LcE does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.zmax(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.jr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.jr(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.jp(
                pot=MWPotential2014,
                type="staeckel",
                delta=4.0 * units.kpc,
                quantity=False,
            )
            - oc.jp(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.jz(
                pot=MWPotential2014,
                type="isochroneapprox",
                b=0.8 * 8.0 * units.kpc,
                quantity=False,
            )
            - oc.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.wr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wr(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.wp(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wp(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.wz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wz(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tr(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tp(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Or(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Op(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(
            o.time(quantity=False) - oc.time() * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method time does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.R(quantity=False) - oc.R() * o._ro) < 10.0**-8.0
    ), "Orbit method R does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.r(quantity=False) - oc.r() * o._ro) < 10.0**-8.0
    ), "Orbit method r does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vR(quantity=False) - oc.vR() * o._vo) < 10.0**-8.0
    ), "Orbit method vR does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vT(quantity=False) - oc.vT() * o._vo) < 10.0**-8.0
    ), "Orbit method vT does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.z(quantity=False) - oc.z() * o._ro) < 10.0**-8.0
    ), "Orbit method z does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vz(quantity=False) - oc.vz() * o._vo) < 10.0**-8.0
    ), "Orbit method vz does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.phi(quantity=False) - oc.phi()) < 10.0**-8.0
    ), "Orbit method phi does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vphi(quantity=False) - oc.vphi() * o._vo / o._ro) < 10.0**-8.0
    ), "Orbit method vphi does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.x(quantity=False) - oc.x() * o._ro) < 10.0**-8.0
    ), "Orbit method x does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.y(quantity=False) - oc.y() * o._ro) < 10.0**-8.0
    ), "Orbit method y does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vx(quantity=False) - oc.vx() * o._vo) < 10.0**-8.0
    ), "Orbit method vx does not return the correct value when Quantity turned off"
    assert (
        numpy.fabs(o.vy(quantity=False) - oc.vy() * o._vo) < 10.0**-8.0
    ), "Orbit method vy does not return the correct value when Quantity turned off"
    return None


def test_integrate_timeAsQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Gyr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential)
    oc.integrate(ts_nounits, MWPotential)
    assert numpy.all(
        numpy.fabs(o.x(ts) - oc.x(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.y(ts) - oc.y(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.z(ts) - oc.z(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vx(ts) - oc.vx(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vy(ts) - oc.vy(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vz(ts) - oc.vz(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_integrate_timeAsQuantity_Myr():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    ts_nounits = numpy.linspace(0.0, 1000.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Myr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro) * 1000.0
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential)
    oc.integrate(ts_nounits, MWPotential)
    assert numpy.all(
        numpy.fabs(o.x(ts) - oc.x(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.y(ts) - oc.y(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.z(ts) - oc.z(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vx(ts) - oc.vx(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vy(ts) - oc.vy(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vz(ts) - oc.vz(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_integrate_dtimeAsQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    dt_nounits = (ts_nounits[1] - ts_nounits[0]) / 10.0
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Gyr)
    dt = dt_nounits * units.Gyr
    ts_nounits /= conversion.time_in_Gyr(vo, ro)
    dt_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential, dt=dt)
    oc.integrate(ts_nounits, MWPotential, dt=dt_nounits)
    assert numpy.all(
        numpy.fabs(o.x(ts) - oc.x(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.y(ts) - oc.y(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.z(ts) - oc.z(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vx(ts) - oc.vx(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vy(ts) - oc.vy(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vz(ts) - oc.vz(ts_nounits)).value < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_integrate_dxdv_timeAsQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Gyr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate_dxdv([1.0, 0.3, 0.4, 0.2], ts, MWPotential, rectIn=True, rectOut=True)
    oc.integrate_dxdv(
        [1.0, 0.3, 0.4, 0.2], ts_nounits, MWPotential, rectIn=True, rectOut=True
    )
    dx = o.getOrbit_dxdv()
    dxc = oc.getOrbit_dxdv()
    assert numpy.all(
        numpy.fabs(dx - dxc) < 10.0**-8.0
    ), "Orbit integrated_dxdv with times specified as Quantity does not agree with Orbit integrated_dxdv with time specified as array"
    return None


def test_integrate_dxdv_timeAsQuantity_Myr():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Myr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro) * 1000.0
    # Integrate both with Quantity time and with unitless time
    o.integrate_dxdv([1.0, 0.3, 0.4, 0.2], ts, MWPotential, rectIn=True, rectOut=True)
    oc.integrate_dxdv(
        [1.0, 0.3, 0.4, 0.2], ts_nounits, MWPotential, rectIn=True, rectOut=True
    )
    dx = o.getOrbit_dxdv()
    dxc = oc.getOrbit_dxdv()
    assert numpy.all(
        numpy.fabs(dx - dxc) < 10.0**-8.0
    ), "Orbit integrated_dxdv with times specified as Quantity does not agree with Orbit integrated_dxdv with time specified as array"
    return None


def test_integrate_SOS_psiQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    oc = o()
    psis_nounits = numpy.linspace(0.0, 400.0, 1001)
    psis = units.Quantity(copy.copy(psis_nounits), unit=units.deg)
    psis_nounits /= 180.0 / numpy.pi
    t0_nounits = 1.0
    t0 = units.Quantity(copy.copy(t0_nounits), unit=units.Gyr)
    t0_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate_SOS(psis, MWPotential, t0=t0)
    oc.integrate_SOS(psis_nounits, MWPotential, t0=t0_nounits)
    assert numpy.all(
        numpy.fabs(o.x(o.t) - oc.x(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.y(o.t) - oc.y(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.z(o.t) - oc.z(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vx(o.t) - oc.vx(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vy(o.t) - oc.vy(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(o.vz(o.t) - oc.vz(oc.t)).value < 10.0**-8.0
    ), "Orbit SOS integrated with psis specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_orbit_inconsistentPotentialUnits_error():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ro, vo = 9.0, 220.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    ts = numpy.linspace(0.0, 10.0, 1001) * units.Gyr
    # single, ro wrong
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pot)
    # list, ro wrong
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, [pot])
    # single, vo wrong
    pot = IsochronePotential(normalize=1.0, ro=9.0, vo=250.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pot)
    # list, vo wrong
    pot = IsochronePotential(normalize=1.0, ro=9.0, vo=250.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, [pot])
    return None


def test_orbits_setup_roAsQuantity():
    from galpy.orbit import Orbit

    ro = 7.0 * units.kpc
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], ro=ro),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], ro=ro),
    ]
    orbits = Orbit(orbits_list, ro=ro)
    assert (
        numpy.fabs(orbits._ro - 7.0) < 10.0**-10.0
    ), "ro in Orbit setup as Quantity does not work as expected"
    return None


def test_orbits_setup_voAsQuantity():
    from galpy.orbit import Orbit

    vo = 230.0 * units.km / units.s
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], vo=vo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], vo=vo),
    ]
    orbits = Orbit(orbits_list, vo=vo)
    assert (
        numpy.fabs(orbits._vo - 230.0) < 10.0**-10.0
    ), "vo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbits_setup_zoAsQuantity():
    from galpy.orbit import Orbit

    zo = 23.0 * units.pc
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], zo=zo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], zo=zo),
    ]
    orbits = Orbit(orbits_list, zo=zo)
    assert (
        numpy.fabs(orbits._zo - 0.023) < 10.0**-10.0
    ), "zo in Orbit setup as Quantity does not work as expected"
    return None


def test_orbits_setup_solarmotionAsQuantity():
    from galpy.orbit import Orbit

    solarmotion = numpy.array([-10.0, 20.0, 30.0]) * units.kpc / units.Gyr
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], solarmotion=solarmotion),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], solarmotion=solarmotion),
    ]
    orbits = Orbit(orbits_list, solarmotion=solarmotion)
    assert numpy.all(
        numpy.fabs(orbits._solarmotion - solarmotion.to(units.km / units.s).value)
        < 10.0**-10.0
    ), "solarmotion in Orbit setup as Quantity does not work as expected"
    return None


def test_orbits_method_returntype_scalar():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            [
                10.0 * units.kpc,
                -20.0 * units.km / units.s,
                210.0 * units.km / units.s,
                500.0 * units.pc,
                -12.0 * units.km / units.s,
                45.0 * units.deg,
            ],
            [
                -20.0 * units.kpc,
                10.0 * units.km / units.s,
                230.0 * units.km / units.s,
                -300.0 * units.pc,
                12.0 * units.km / units.s,
                125.0 * units.deg,
            ],
        ]
    )
    from galpy.potential import MWPotential2014

    assert isinstance(
        o.E(pot=MWPotential2014), units.Quantity
    ), "Orbit method E does not return Quantity when it should"
    assert isinstance(
        o.ER(pot=MWPotential2014), units.Quantity
    ), "Orbit method ER does not return Quantity when it should"
    assert isinstance(
        o.Ez(pot=MWPotential2014), units.Quantity
    ), "Orbit method Ez does not return Quantity when it should"
    assert isinstance(
        o.Jacobi(pot=MWPotential2014), units.Quantity
    ), "Orbit method Jacobi does not return Quantity when it should"
    assert isinstance(
        o.L(), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.Lz(), units.Quantity
    ), "Orbit method Lz does not return Quantity when it should"
    assert isinstance(
        o.rap(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method rap does not return Quantity when it should"
    assert isinstance(
        o.rperi(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method rperi does not return Quantity when it should"
    assert isinstance(
        o.rguiding(pot=MWPotential2014), units.Quantity
    ), "Orbit method rguiding does not return Quantity when it should"
    assert isinstance(
        o.rE(pot=MWPotential2014), units.Quantity
    ), "Orbit method rE does not return Quantity when it should"
    assert isinstance(
        o.LcE(pot=MWPotential2014), units.Quantity
    ), "Orbit method LcE does not return Quantity when it should"
    assert isinstance(
        o.zmax(pot=MWPotential2014, analytic=True), units.Quantity
    ), "Orbit method zmax does not return Quantity when it should"
    assert isinstance(
        o.jr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jr does not return Quantity when it should"
    assert isinstance(
        o.jp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jp does not return Quantity when it should"
    assert isinstance(
        o.jz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method jz does not return Quantity when it should"
    assert isinstance(
        o.wr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wr does not return Quantity when it should"
    assert isinstance(
        o.wp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wp does not return Quantity when it should"
    assert isinstance(
        o.wz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method wz does not return Quantity when it should"
    assert isinstance(
        o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tr does not return Quantity when it should"
    assert isinstance(
        o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tp does not return Quantity when it should"
    assert isinstance(
        o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Tz does not return Quantity when it should"
    assert isinstance(
        o.Or(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Or does not return Quantity when it should"
    assert isinstance(
        o.Op(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Op does not return Quantity when it should"
    assert isinstance(
        o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5), units.Quantity
    ), "Orbit method Oz does not return Quantity when it should"
    assert isinstance(
        o.time(), units.Quantity
    ), "Orbit method time does not return Quantity when it should"
    assert isinstance(
        o.R(), units.Quantity
    ), "Orbit method R does not return Quantity when it should"
    assert isinstance(
        o.r(), units.Quantity
    ), "Orbit method r does not return Quantity when it should"
    assert isinstance(
        o.vR(), units.Quantity
    ), "Orbit method vR does not return Quantity when it should"
    assert isinstance(
        o.vT(), units.Quantity
    ), "Orbit method vT does not return Quantity when it should"
    assert isinstance(
        o.z(), units.Quantity
    ), "Orbit method z does not return Quantity when it should"
    assert isinstance(
        o.vz(), units.Quantity
    ), "Orbit method vz does not return Quantity when it should"
    assert isinstance(
        o.phi(), units.Quantity
    ), "Orbit method phi does not return Quantity when it should"
    assert isinstance(
        o.vphi(), units.Quantity
    ), "Orbit method vphi does not return Quantity when it should"
    assert isinstance(
        o.x(), units.Quantity
    ), "Orbit method x does not return Quantity when it should"
    assert isinstance(
        o.y(), units.Quantity
    ), "Orbit method y does not return Quantity when it should"
    assert isinstance(
        o.vx(), units.Quantity
    ), "Orbit method vx does not return Quantity when it should"
    assert isinstance(
        o.vy(), units.Quantity
    ), "Orbit method vy does not return Quantity when it should"
    assert isinstance(
        o.ra(), units.Quantity
    ), "Orbit method ra does not return Quantity when it should"
    assert isinstance(
        o.dec(), units.Quantity
    ), "Orbit method dec does not return Quantity when it should"
    assert isinstance(
        o.ll(), units.Quantity
    ), "Orbit method ll does not return Quantity when it should"
    assert isinstance(
        o.bb(), units.Quantity
    ), "Orbit method bb does not return Quantity when it should"
    assert isinstance(
        o.dist(), units.Quantity
    ), "Orbit method dist does not return Quantity when it should"
    assert isinstance(
        o.pmra(), units.Quantity
    ), "Orbit method pmra does not return Quantity when it should"
    assert isinstance(
        o.pmdec(), units.Quantity
    ), "Orbit method pmdec does not return Quantity when it should"
    assert isinstance(
        o.pmll(), units.Quantity
    ), "Orbit method pmll does not return Quantity when it should"
    assert isinstance(
        o.pmbb(), units.Quantity
    ), "Orbit method pmbb does not return Quantity when it should"
    assert isinstance(
        o.vlos(), units.Quantity
    ), "Orbit method vlos does not return Quantity when it should"
    assert isinstance(
        o.vra(), units.Quantity
    ), "Orbit method vra does not return Quantity when it should"
    assert isinstance(
        o.vdec(), units.Quantity
    ), "Orbit method vdec does not return Quantity when it should"
    assert isinstance(
        o.vll(), units.Quantity
    ), "Orbit method vll does not return Quantity when it should"
    assert isinstance(
        o.vbb(), units.Quantity
    ), "Orbit method vbb does not return Quantity when it should"
    assert isinstance(
        o.helioX(), units.Quantity
    ), "Orbit method helioX does not return Quantity when it should"
    assert isinstance(
        o.helioY(), units.Quantity
    ), "Orbit method helioY does not return Quantity when it should"
    assert isinstance(
        o.helioZ(), units.Quantity
    ), "Orbit method helioZ does not return Quantity when it should"
    assert isinstance(
        o.U(), units.Quantity
    ), "Orbit method U does not return Quantity when it should"
    assert isinstance(
        o.V(), units.Quantity
    ), "Orbit method V does not return Quantity when it should"
    assert isinstance(
        o.W(), units.Quantity
    ), "Orbit method W does not return Quantity when it should"
    return None


def test_orbits_method_returntype():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            [
                10.0 * units.kpc,
                -20.0 * units.km / units.s,
                210.0 * units.km / units.s,
                500.0 * units.pc,
                -12.0 * units.km / units.s,
                45.0 * units.deg,
            ],
            [
                -20.0 * units.kpc,
                10.0 * units.km / units.s,
                230.0 * units.km / units.s,
                -300.0 * units.pc,
                12.0 * units.km / units.s,
                125.0 * units.deg,
            ],
        ]
    )
    from galpy.potential import MWPotential2014

    ts = numpy.linspace(0.0, 6.0, 1001)
    o.integrate(ts, MWPotential2014)
    assert isinstance(
        o.E(ts), units.Quantity
    ), "Orbit method E does not return Quantity when it should"
    assert isinstance(
        o.ER(ts), units.Quantity
    ), "Orbit method ER does not return Quantity when it should"
    assert isinstance(
        o.Ez(ts), units.Quantity
    ), "Orbit method Ez does not return Quantity when it should"
    assert isinstance(
        o.Jacobi(ts), units.Quantity
    ), "Orbit method Jacobi does not return Quantity when it should"
    assert isinstance(
        o.L(ts), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.Lz(ts), units.Quantity
    ), "Orbit method L does not return Quantity when it should"
    assert isinstance(
        o.time(ts), units.Quantity
    ), "Orbit method time does not return Quantity when it should"
    assert isinstance(
        o.R(ts), units.Quantity
    ), "Orbit method R does not return Quantity when it should"
    assert isinstance(
        o.r(ts), units.Quantity
    ), "Orbit method r does not return Quantity when it should"
    assert isinstance(
        o.vR(ts), units.Quantity
    ), "Orbit method vR does not return Quantity when it should"
    assert isinstance(
        o.vT(ts), units.Quantity
    ), "Orbit method vT does not return Quantity when it should"
    assert isinstance(
        o.z(ts), units.Quantity
    ), "Orbit method z does not return Quantity when it should"
    assert isinstance(
        o.vz(ts), units.Quantity
    ), "Orbit method vz does not return Quantity when it should"
    assert isinstance(
        o.phi(ts), units.Quantity
    ), "Orbit method phi does not return Quantity when it should"
    assert isinstance(
        o.vphi(ts), units.Quantity
    ), "Orbit method vphi does not return Quantity when it should"
    assert isinstance(
        o.x(ts), units.Quantity
    ), "Orbit method x does not return Quantity when it should"
    assert isinstance(
        o.y(ts), units.Quantity
    ), "Orbit method y does not return Quantity when it should"
    assert isinstance(
        o.vx(ts), units.Quantity
    ), "Orbit method vx does not return Quantity when it should"
    assert isinstance(
        o.vy(ts), units.Quantity
    ), "Orbit method vy does not return Quantity when it should"
    assert isinstance(
        o.ra(ts), units.Quantity
    ), "Orbit method ra does not return Quantity when it should"
    assert isinstance(
        o.dec(ts), units.Quantity
    ), "Orbit method dec does not return Quantity when it should"
    assert isinstance(
        o.ll(ts), units.Quantity
    ), "Orbit method ll does not return Quantity when it should"
    assert isinstance(
        o.bb(ts), units.Quantity
    ), "Orbit method bb does not return Quantity when it should"
    assert isinstance(
        o.dist(ts), units.Quantity
    ), "Orbit method dist does not return Quantity when it should"
    assert isinstance(
        o.pmra(ts), units.Quantity
    ), "Orbit method pmra does not return Quantity when it should"
    assert isinstance(
        o.pmdec(ts), units.Quantity
    ), "Orbit method pmdec does not return Quantity when it should"
    assert isinstance(
        o.pmll(ts), units.Quantity
    ), "Orbit method pmll does not return Quantity when it should"
    assert isinstance(
        o.pmbb(ts), units.Quantity
    ), "Orbit method pmbb does not return Quantity when it should"
    assert isinstance(
        o.vlos(ts), units.Quantity
    ), "Orbit method vlos does not return Quantity when it should"
    assert isinstance(
        o.vra(ts), units.Quantity
    ), "Orbit method vra does not return Quantity when it should"
    assert isinstance(
        o.vdec(ts), units.Quantity
    ), "Orbit method vdec does not return Quantity when it should"
    assert isinstance(
        o.vll(ts), units.Quantity
    ), "Orbit method vll does not return Quantity when it should"
    assert isinstance(
        o.vbb(ts), units.Quantity
    ), "Orbit method vbb does not return Quantity when it should"
    assert isinstance(
        o.helioX(ts), units.Quantity
    ), "Orbit method helioX does not return Quantity when it should"
    assert isinstance(
        o.helioY(ts), units.Quantity
    ), "Orbit method helioY does not return Quantity when it should"
    assert isinstance(
        o.helioZ(ts), units.Quantity
    ), "Orbit method helioZ does not return Quantity when it should"
    assert isinstance(
        o.U(ts), units.Quantity
    ), "Orbit method U does not return Quantity when it should"
    assert isinstance(
        o.V(ts), units.Quantity
    ), "Orbit method V does not return Quantity when it should"
    assert isinstance(
        o.W(ts), units.Quantity
    ), "Orbit method W does not return Quantity when it should"
    return None


def test_orbits_method_returnunit():
    from galpy.orbit import Orbit

    o = Orbit(
        [
            [
                10.0 * units.kpc,
                -20.0 * units.km / units.s,
                210.0 * units.km / units.s,
                500.0 * units.pc,
                -12.0 * units.km / units.s,
                45.0 * units.deg,
            ],
            [
                -20.0 * units.kpc,
                10.0 * units.km / units.s,
                230.0 * units.km / units.s,
                -300.0 * units.pc,
                12.0 * units.km / units.s,
                125.0 * units.deg,
            ],
        ]
    )
    from galpy.potential import MWPotential2014

    try:
        o.E(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method E does not return Quantity with the right units"
        )
    try:
        o.ER(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ER does not return Quantity with the right units"
        )
    try:
        o.Ez(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Ez does not return Quantity with the right units"
        )
    try:
        o.Jacobi(pot=MWPotential2014).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Jacobi does not return Quantity with the right units"
        )
    try:
        o.L().to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method L does not return Quantity with the right units"
        )
    try:
        o.Lz().to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Lz does not return Quantity with the right units"
        )
    try:
        o.rap(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rap does not return Quantity with the right units"
        )
    try:
        o.rperi(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rperi does not return Quantity with the right units"
        )
    try:
        o.rguiding(pot=MWPotential2014).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rguiding does not return Quantity with the right units"
        )
    try:
        o.rE(pot=MWPotential2014).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method rE does not return Quantity with the right units"
        )
    try:
        o.LcE(pot=MWPotential2014).to(units.kpc * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method LcE does not return Quantity with the right units"
        )
    try:
        o.zmax(pot=MWPotential2014, analytic=True).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method zmax does not return Quantity with the right units"
        )
    try:
        o.jr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jr does not return Quantity with the right units"
        )
    try:
        o.jp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jp does not return Quantity with the right units"
        )
    try:
        o.jz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.km**2 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method jz does not return Quantity with the right units"
        )
    try:
        o.wr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wr does not return Quantity with the right units"
        )
    try:
        o.wp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wp does not return Quantity with the right units"
        )
    try:
        o.wz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method wz does not return Quantity with the right units"
        )
    try:
        o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tr does not return Quantity with the right units"
        )
    try:
        o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tp does not return Quantity with the right units"
        )
    try:
        o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Tz does not return Quantity with the right units"
        )
    try:
        o.Or(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Or does not return Quantity with the right units"
        )
    try:
        o.Op(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Op does not return Quantity with the right units"
        )
    try:
        o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5).to(1 / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method Oz does not return Quantity with the right units"
        )
    try:
        o.time().to(units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method time does not return Quantity with the right units"
        )
    try:
        o.R().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method R does not return Quantity with the right units"
        )
    try:
        o.r().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method r does not return Quantity with the right units"
        )
    try:
        o.vR().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vR does not return Quantity with the right units"
        )
    try:
        o.vT().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vT does not return Quantity with the right units"
        )
    try:
        o.z().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method z does not return Quantity with the right units"
        )
    try:
        o.vz().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vz does not return Quantity with the right units"
        )
    try:
        o.phi().to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method phi does not return Quantity with the right units"
        )
    try:
        o.vphi().to(units.km / units.s / units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vphi does not return Quantity with the right units"
        )
    try:
        o.x().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method x does not return Quantity with the right units"
        )
    try:
        o.y().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method y does not return Quantity with the right units"
        )
    try:
        o.vx().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vx does not return Quantity with the right units"
        )
    try:
        o.vy().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vy does not return Quantity with the right units"
        )
    try:
        o.ra().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ra does not return Quantity with the right units"
        )
    try:
        o.dec().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method dec does not return Quantity with the right units"
        )
    try:
        o.ll().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method ll does not return Quantity with the right units"
        )
    try:
        o.bb().to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method bb does not return Quantity with the right units"
        )
    try:
        o.dist().to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method dist does not return Quantity with the right units"
        )
    try:
        o.pmra().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmra does not return Quantity with the right units"
        )
    try:
        o.pmdec().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmdec does not return Quantity with the right units"
        )
    try:
        o.pmll().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmll does not return Quantity with the right units"
        )
    try:
        o.pmbb().to(units.mas / units.yr)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method pmbb does not return Quantity with the right units"
        )
    try:
        o.vlos().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vlos does not return Quantity with the right units"
        )
    try:
        o.vra().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vra does not return Quantity with the right units"
        )
    try:
        o.vdec().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vdec does not return Quantity with the right units"
        )
    try:
        o.vll().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vll does not return Quantity with the right units"
        )
    try:
        o.vbb().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method vbb does not return Quantity with the right units"
        )
    try:
        o.helioX().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioX does not return Quantity with the right units"
        )
    try:
        o.helioY().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioY does not return Quantity with the right units"
        )
    try:
        o.helioZ().to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method helioZ does not return Quantity with the right units"
        )
    try:
        o.U().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method U does not return Quantity with the right units"
        )
    try:
        o.V().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method V does not return Quantity with the right units"
        )
    try:
        o.W().to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method W does not return Quantity with the right units"
        )
    return None


def test_orbits_method_value():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    o = Orbit(
        [
            [
                10.0 * units.kpc,
                -20.0 * units.km / units.s,
                210.0 * units.km / units.s,
                500.0 * units.pc,
                -12.0 * units.km / units.s,
                45.0 * units.deg,
            ],
            [
                -20.0 * units.kpc,
                10.0 * units.km / units.s,
                230.0 * units.km / units.s,
                -300.0 * units.pc,
                12.0 * units.km / units.s,
                125.0 * units.deg,
            ],
        ]
    )
    oc = o()
    oc.turn_physical_off()
    assert numpy.all(
        numpy.fabs(
            o.E(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.E(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.ER(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.ER(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Ez(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.Ez(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014).to(units.km**2 / units.s**2).value
            - oc.Jacobi(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014).to(units.km / units.s * units.kpc).value
            - oc.L(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Lz(pot=MWPotential2014).to(units.km / units.s * units.kpc).value
            - oc.Lz(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.rap(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.rperi(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.rguiding(pot=MWPotential2014).to(units.kpc).value
            - oc.rguiding(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rguiding does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.rE(pot=MWPotential2014).to(units.kpc).value
            - oc.rE(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rE does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.LcE(pot=MWPotential2014).to(units.kpc * units.km / units.s).value
            - oc.LcE(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method LcE does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True).to(units.kpc).value
            - oc.zmax(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.jr(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jr(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.jp(pot=MWPotential2014, type="staeckel", delta=4.0 * units.kpc)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jp(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8 * 8.0 * units.kpc)
            .to(units.km / units.s * units.kpc)
            .value
            - oc.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.wr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wr(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.wp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wp(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.wz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.rad).value
            - oc.wz(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tr(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tp(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5).to(units.Gyr).value
            - oc.Tz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            .to(1 / units.Gyr)
            .value
            - oc.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.time().to(units.Gyr).value
            - oc.time() * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method time does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.R().to(units.kpc).value - oc.R() * o._ro) < 10.0**-8.0
    ), "Orbit method R does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.r().to(units.kpc).value - oc.r() * o._ro) < 10.0**-8.0
    ), "Orbit method r does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vR().to(units.km / units.s).value - oc.vR() * o._vo) < 10.0**-8.0
    ), "Orbit method vR does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vT().to(units.km / units.s).value - oc.vT() * o._vo) < 10.0**-8.0
    ), "Orbit method vT does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.z().to(units.kpc).value - oc.z() * o._ro) < 10.0**-8.0
    ), "Orbit method z does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vz().to(units.km / units.s).value - oc.vz() * o._vo) < 10.0**-8.0
    ), "Orbit method vz does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.phi().to(units.rad).value - oc.phi()) < 10.0**-8.0
    ), "Orbit method phi does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            o.vphi().to(units.km / units.s / units.kpc).value
            - oc.vphi() * o._vo / o._ro
        )
        < 10.0**-8.0
    ), "Orbit method vphi does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.x().to(units.kpc).value - oc.x() * o._ro) < 10.0**-8.0
    ), "Orbit method x does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.y().to(units.kpc).value - oc.y() * o._ro) < 10.0**-8.0
    ), "Orbit method y does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vx().to(units.km / units.s).value - oc.vx() * o._vo) < 10.0**-8.0
    ), "Orbit method vx does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vy().to(units.km / units.s).value - oc.vy() * o._vo) < 10.0**-8.0
    ), "Orbit method vy does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.ra().to(units.deg).value - oc.ra(quantity=False)) < 10.0**-8.0
    ), "Orbit method ra does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.dec().to(units.deg).value - oc.dec(quantity=False)) < 10.0**-8.0
    ), "Orbit method dec does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.ll().to(units.deg).value - oc.ll(quantity=False)) < 10.0**-8.0
    ), "Orbit method ll does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.bb().to(units.deg).value - oc.bb(quantity=False)) < 10.0**-8.0
    ), "Orbit method bb does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.dist().to(units.kpc).value - oc.dist(quantity=False)) < 10.0**-8.0
    ), "Orbit method dist does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.pmra().to(units.mas / units.yr).value - oc.pmra(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmra does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.pmdec().to(units.mas / units.yr).value - oc.pmdec(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmdec does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.pmll().to(units.mas / units.yr).value - oc.pmll(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmll does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.pmbb().to(units.mas / units.yr).value - oc.pmbb(quantity=False))
        < 10.0**-8.0
    ), "Orbit method pmbb does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vlos().to(units.km / units.s).value - oc.vlos(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vlos does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vra().to(units.km / units.s).value - oc.vra(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vra does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vdec().to(units.km / units.s).value - oc.vdec(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vdec does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vll().to(units.km / units.s).value - oc.vll(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vll does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.vbb().to(units.km / units.s).value - oc.vbb(quantity=False))
        < 10.0**-8.0
    ), "Orbit method vbb does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.helioX().to(units.kpc).value - oc.helioX(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioX does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.helioY().to(units.kpc).value - oc.helioY(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioY does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.helioZ().to(units.kpc).value - oc.helioZ(quantity=False))
        < 10.0**-8.0
    ), "Orbit method helioZ does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.U().to(units.km / units.s).value - oc.U(quantity=False))
        < 10.0**-8.0
    ), "Orbit method U does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.V().to(units.km / units.s).value - oc.V(quantity=False))
        < 10.0**-8.0
    ), "Orbit method V does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(o.W().to(units.km / units.s).value - oc.W(quantity=False))
        < 10.0**-8.0
    ), "Orbit method W does not return the correct value as Quantity"
    return None


def test_orbits_method_value_turnquantityoff():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import conversion

    o = Orbit(
        [
            [
                10.0 * units.kpc,
                -20.0 * units.km / units.s,
                210.0 * units.km / units.s,
                500.0 * units.pc,
                -12.0 * units.km / units.s,
                45.0 * units.deg,
            ],
            [
                -20.0 * units.kpc,
                10.0 * units.km / units.s,
                230.0 * units.km / units.s,
                -300.0 * units.pc,
                12.0 * units.km / units.s,
                125.0 * units.deg,
            ],
        ]
    )
    oc = o()
    oc.turn_physical_off()
    assert numpy.all(
        numpy.fabs(
            o.E(pot=MWPotential2014, quantity=False)
            - oc.E(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method E does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.ER(pot=MWPotential2014, quantity=False)
            - oc.ER(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method ER does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Ez(pot=MWPotential2014, quantity=False)
            - oc.Ez(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Ez does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Jacobi(pot=MWPotential2014, quantity=False)
            - oc.Jacobi(pot=MWPotential2014) * o._vo**2.0
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.L(pot=MWPotential2014, quantity=False)
            - oc.L(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Lz(pot=MWPotential2014, quantity=False)
            - oc.Lz(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method L does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.rap(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.rap(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rap does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.rperi(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.rperi(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rperi does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.rguiding(pot=MWPotential2014, quantity=False)
            - oc.rguiding(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rguiding does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.rE(pot=MWPotential2014, quantity=False)
            - oc.rE(pot=MWPotential2014) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method rE does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.LcE(pot=MWPotential2014, quantity=False)
            - oc.LcE(pot=MWPotential2014) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method LcE does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.zmax(pot=MWPotential2014, analytic=True, quantity=False)
            - oc.zmax(pot=MWPotential2014, analytic=True) * o._ro
        )
        < 10.0**-8.0
    ), "Orbit method zmax does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.jr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.jr(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jr does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.jp(
                pot=MWPotential2014,
                type="staeckel",
                delta=4.0 * units.kpc,
                quantity=False,
            )
            - oc.jp(pot=MWPotential2014, type="staeckel", delta=0.5) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jp does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.jz(
                pot=MWPotential2014,
                type="isochroneapprox",
                b=0.8 * 8.0 * units.kpc,
                quantity=False,
            )
            - oc.jz(pot=MWPotential2014, type="isochroneapprox", b=0.8) * o._ro * o._vo
        )
        < 10.0**-8.0
    ), "Orbit method jz does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.wr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wr(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wr does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.wp(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wp(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wp does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.wz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.wz(pot=MWPotential2014, type="staeckel", delta=0.5)
        )
        < 10.0**-8.0
    ), "Orbit method wz does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Tr(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tr(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tr does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Tp(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tp(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tp does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Tz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Tz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Tz does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Or(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Or(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method Or does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Op(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Op(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Opbit method Or does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.Oz(pot=MWPotential2014, type="staeckel", delta=0.5, quantity=False)
            - oc.Oz(pot=MWPotential2014, type="staeckel", delta=0.5)
            * conversion.freq_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Ozbit method Or does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(
            o.time(quantity=False) - oc.time() * conversion.time_in_Gyr(o._vo, o._ro)
        )
        < 10.0**-8.0
    ), "Orbit method time does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.R(quantity=False) - oc.R() * o._ro) < 10.0**-8.0
    ), "Orbit method R does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.r(quantity=False) - oc.r() * o._ro) < 10.0**-8.0
    ), "Orbit method r does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vR(quantity=False) - oc.vR() * o._vo) < 10.0**-8.0
    ), "Orbit method vR does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vT(quantity=False) - oc.vT() * o._vo) < 10.0**-8.0
    ), "Orbit method vT does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.z(quantity=False) - oc.z() * o._ro) < 10.0**-8.0
    ), "Orbit method z does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vz(quantity=False) - oc.vz() * o._vo) < 10.0**-8.0
    ), "Orbit method vz does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.phi(quantity=False) - oc.phi()) < 10.0**-8.0
    ), "Orbit method phi does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vphi(quantity=False) - oc.vphi() * o._vo / o._ro) < 10.0**-8.0
    ), "Orbit method vphi does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.x(quantity=False) - oc.x() * o._ro) < 10.0**-8.0
    ), "Orbit method x does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.y(quantity=False) - oc.y() * o._ro) < 10.0**-8.0
    ), "Orbit method y does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vx(quantity=False) - oc.vx() * o._vo) < 10.0**-8.0
    ), "Orbit method vx does not return the correct value when Quantity turned off"
    assert numpy.all(
        numpy.fabs(o.vy(quantity=False) - oc.vy() * o._vo) < 10.0**-8.0
    ), "Orbit method vy does not return the correct value when Quantity turned off"
    return None


def test_integrate_orbits_timeAsQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    oc = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Gyr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential)
    oc.integrate(ts_nounits, MWPotential)
    # Turn physical units off for ease
    o.turn_physical_off()
    oc.turn_physical_off()
    assert numpy.all(
        numpy.fabs(numpy.array(o.x(ts)) - numpy.array(oc.x(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.y(ts)) - numpy.array(oc.y(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.z(ts)) - numpy.array(oc.z(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vx(ts)) - numpy.array(oc.vx(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vy(ts)) - numpy.array(oc.vy(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vz(ts)) - numpy.array(oc.vz(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_orbits_integrate_timeAsQuantity_Myr():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    oc = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    ts_nounits = numpy.linspace(0.0, 1000.0, 1001)
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Myr)
    ts_nounits /= conversion.time_in_Gyr(vo, ro) * 1000.0
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential)
    oc.integrate(ts_nounits, MWPotential)
    # Turn physical units off for ease
    o.turn_physical_off()
    oc.turn_physical_off()
    assert numpy.all(
        numpy.fabs(numpy.array(o.x(ts)) - numpy.array(oc.x(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.y(ts)) - numpy.array(oc.y(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.z(ts)) - numpy.array(oc.z(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vx(ts)) - numpy.array(oc.vx(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vy(ts)) - numpy.array(oc.vy(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vz(ts)) - numpy.array(oc.vz(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_orbits_integrate_dtimeAsQuantity():
    import copy

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro, vo = 8.0, 200.0
    o = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    oc = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    500.0 * units.pc,
                    -12.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    ts_nounits = numpy.linspace(0.0, 1.0, 1001)
    dt_nounits = (ts_nounits[1] - ts_nounits[0]) / 10.0
    ts = units.Quantity(copy.copy(ts_nounits), unit=units.Gyr)
    dt = dt_nounits * units.Gyr
    ts_nounits /= conversion.time_in_Gyr(vo, ro)
    dt_nounits /= conversion.time_in_Gyr(vo, ro)
    # Integrate both with Quantity time and with unitless time
    o.integrate(ts, MWPotential, dt=dt)
    oc.integrate(ts_nounits, MWPotential, dt=dt_nounits)
    # Turn physical units off for ease
    o.turn_physical_off()
    oc.turn_physical_off()
    assert numpy.all(
        numpy.fabs(numpy.array(o.x(ts)) - numpy.array(oc.x(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.y(ts)) - numpy.array(oc.y(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.z(ts)) - numpy.array(oc.z(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vx(ts)) - numpy.array(oc.vx(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vy(ts)) - numpy.array(oc.vy(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    assert numpy.all(
        numpy.fabs(numpy.array(o.vz(ts)) - numpy.array(oc.vz(ts_nounits))) < 10.0**-8.0
    ), "Orbit integrated with times specified as Quantity does not agree with Orbit integrated with time specified as array"
    return None


def test_orbits_inconsistentPotentialUnits_error():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential

    ro, vo = 9.0, 220.0
    o = Orbit(
        [
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
            Orbit(
                [
                    10.0 * units.kpc,
                    -20.0 * units.km / units.s,
                    210.0 * units.km / units.s,
                    45.0 * units.deg,
                ],
                ro=ro,
                vo=vo,
            ),
        ]
    )
    ts = numpy.linspace(0.0, 10.0, 1001) * units.Gyr
    # single, ro wrong
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pot)
    # list, ro wrong
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, [pot])
    # single, vo wrong
    pot = IsochronePotential(normalize=1.0, ro=9.0, vo=250.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pot)
    # list, vo wrong
    pot = IsochronePotential(normalize=1.0, ro=9.0, vo=250.0)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, [pot])
    return None


def test_orbit_method_inputAsQuantity():
    from galpy import potential
    from galpy.orbit import Orbit

    ro, vo = 7.0, 210.0
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            500.0 * units.pc,
            -12.0 * units.km / units.s,
            45.0 * units.deg,
        ],
        ro=ro,
        vo=vo,
    )
    assert (
        numpy.fabs(
            o.Jacobi(
                pot=potential.MWPotential,
                OmegaP=41 * units.km / units.s / units.kpc,
                use_physical=False,
            )
            - o.Jacobi(
                pot=potential.MWPotential, OmegaP=41.0 * ro / vo, use_physical=False
            )
        )
        < 10.0**-8.0
    ), "Orbit method Jacobi does not return the correct value when input OmegaP is Quantity"
    return None


def test_change_ro_config():
    from galpy.orbit import Orbit
    from galpy.util import config

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert numpy.fabs(o._ro - 8.0) < 10.0**-10.0, "Default ro value not as expected"
    # Change value
    newro = 9.0
    config.set_ro(newro)
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert numpy.fabs(o._ro - newro) < 10.0**-10.0, "Default ro value not as expected"
    # Change value as Quantity
    newro = 9.0 * units.kpc
    config.set_ro(newro)
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert (
        numpy.fabs(o._ro - newro.value) < 10.0**-10.0
    ), "Default ro value not as expected"
    # Back to default
    config.set_ro(8.0)
    return None


def test_change_vo_config():
    from galpy.orbit import Orbit
    from galpy.util import config

    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert numpy.fabs(o._vo - 220.0) < 10.0**-10.0, "Default ro value not as expected"
    # Change value
    newvo = 250.0
    config.set_vo(newvo)
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert numpy.fabs(o._vo - newvo) < 10.0**-10.0, "Default ro value not as expected"
    # Change value as Quantity
    newvo = 250.0 * units.km / units.s
    config.set_vo(newvo)
    o = Orbit(
        [
            10.0 * units.kpc,
            -20.0 * units.km / units.s,
            210.0 * units.km / units.s,
            45.0 * units.deg,
        ]
    )
    assert (
        numpy.fabs(o._vo - newvo.value) < 10.0**-10.0
    ), "Default ro value not as expected"
    # Back to default
    config.set_vo(220.0)
    return None


def test_potential_method_returntype():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0)
    assert isinstance(
        pot(1.1, 0.1), units.Quantity
    ), "Potential method __call__ does not return Quantity when it should"
    assert isinstance(
        pot.Rforce(1.1, 0.1), units.Quantity
    ), "Potential method Rforce does not return Quantity when it should"
    assert isinstance(
        pot.rforce(1.1, 0.1), units.Quantity
    ), "Potential method rforce does not return Quantity when it should"
    assert isinstance(
        pot.zforce(1.1, 0.1), units.Quantity
    ), "Potential method zforce does not return Quantity when it should"
    assert isinstance(
        pot.phitorque(1.1, 0.1), units.Quantity
    ), "Potential method phitorque does not return Quantity when it should"
    assert isinstance(
        pot.dens(1.1, 0.1), units.Quantity
    ), "Potential method dens does not return Quantity when it should"
    assert isinstance(
        pot.surfdens(1.1, 0.1), units.Quantity
    ), "Potential method surfdens does not return Quantity when it should"
    assert isinstance(
        pot.mass(1.1, 0.1), units.Quantity
    ), "Potential method mass does not return Quantity when it should"
    assert isinstance(
        pot.R2deriv(1.1, 0.1), units.Quantity
    ), "Potential method R2deriv does not return Quantity when it should"
    assert isinstance(
        pot.z2deriv(1.1, 0.1), units.Quantity
    ), "Potential method z2deriv does not return Quantity when it should"
    assert isinstance(
        pot.Rzderiv(1.1, 0.1), units.Quantity
    ), "Potential method Rzderiv does not return Quantity when it should"
    assert isinstance(
        pot.Rphideriv(1.1, 0.1), units.Quantity
    ), "Potential method Rphideriv does not return Quantity when it should"
    assert isinstance(
        pot.phi2deriv(1.1, 0.1), units.Quantity
    ), "Potential method phi2deriv does not return Quantity when it should"
    assert isinstance(
        pot.phizderiv(1.1, 0.1), units.Quantity
    ), "Potential method phizderiv does not return Quantity when it should"
    assert isinstance(
        pot.flattening(1.1, 0.1), units.Quantity
    ), "Potential method flattening does not return Quantity when it should"
    assert isinstance(
        pot.vcirc(1.1), units.Quantity
    ), "Potential method vcirc does not return Quantity when it should"
    assert isinstance(
        pot.dvcircdR(1.1), units.Quantity
    ), "Potential method dvcircdR does not return Quantity when it should"
    assert isinstance(
        pot.omegac(1.1), units.Quantity
    ), "Potential method omegac does not return Quantity when it should"
    assert isinstance(
        pot.epifreq(1.1), units.Quantity
    ), "Potential method epifreq does not return Quantity when it should"
    assert isinstance(
        pot.verticalfreq(1.1), units.Quantity
    ), "Potential method verticalfreq does not return Quantity when it should"
    assert (
        pot.lindbladR(0.9) is None
    ), "Potential method lindbladR does not return None, even when it should return a Quantity, when it should"
    assert isinstance(
        pot.lindbladR(0.9, m="corot"), units.Quantity
    ), "Potential method lindbladR does not return Quantity when it should"
    assert isinstance(
        pot.vesc(1.3), units.Quantity
    ), "Potential method vesc does not return Quantity when it should"
    assert isinstance(
        pot.rl(1.3), units.Quantity
    ), "Potential method rl does not return Quantity when it should"
    assert isinstance(
        pot.rE(-1.14), units.Quantity
    ), "Potential method rE does not return Quantity when it should"
    assert isinstance(
        pot.LcE(-1.14), units.Quantity
    ), "Potential method LcE does not return Quantity when it should"
    assert isinstance(
        pot.vterm(45.0), units.Quantity
    ), "Potential method vterm does not return Quantity when it should"
    assert isinstance(
        pot.rtide(1.0, 0.0, M=1.0), units.Quantity
    ), "Potential method rtide does not return Quantity when it should"
    assert isinstance(
        pot.ttensor(1.0, 0.0), units.Quantity
    ), "Potential method ttensor does not return Quantity when it should"
    assert isinstance(
        pot.ttensor(1.0, 0.0, eigenval=True), units.Quantity
    ), "Potential method ttensor does not return Quantity when it should"
    assert isinstance(
        pot.zvc_range(-1.9, 0.2), units.Quantity
    ), "Potential method zvc_range does not return Quantity when it should"
    assert isinstance(
        pot.zvc(0.4, -1.9, 0.2), units.Quantity
    ), "Potential method zvc does not return Quantity when it should"
    assert isinstance(
        pot.rhalf(), units.Quantity
    ), "Potential method rhalf does not return Quantity when it should"
    assert isinstance(
        pot.tdyn(1.1), units.Quantity
    ), "Potential method tdyn does not return Quantity when it should"
    return None


def test_dissipativeforce_method_returntype():
    from galpy.potential import ChandrasekharDynamicalFrictionForce

    pot = ChandrasekharDynamicalFrictionForce(GMs=0.1, rhm=1.2 / 8.0, ro=8.0, vo=220.0)
    assert isinstance(
        pot.phitorque(1.1, 0.1, phi=2.0, v=[0.1, 1.2, 0.3]), units.Quantity
    ), "Potential method phitorque does not return Quantity when it should"
    assert isinstance(
        pot.Rforce(1.1, 0.1, phi=2.0, v=[0.1, 1.2, 0.3]), units.Quantity
    ), "Potential method Rforce does not return Quantity when it should"
    assert isinstance(
        pot.zforce(1.1, 0.1, phi=2.0, v=[0.1, 1.2, 0.3]), units.Quantity
    ), "Potential method zforce does not return Quantity when it should"
    return None


def test_planarPotential_method_returntype():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0).toPlanar()
    assert isinstance(
        pot(1.1), units.Quantity
    ), "Potential method __call__ does not return Quantity when it should"
    assert isinstance(
        pot.Rforce(1.1), units.Quantity
    ), "Potential method Rforce does not return Quantity when it should"
    assert isinstance(
        pot.phitorque(1.1), units.Quantity
    ), "Potential method phitorque does not return Quantity when it should"
    assert isinstance(
        pot.R2deriv(1.1), units.Quantity
    ), "Potential method R2deriv does not return Quantity when it should"
    assert isinstance(
        pot.Rphideriv(1.1), units.Quantity
    ), "Potential method Rphideriv does not return Quantity when it should"
    assert isinstance(
        pot.phi2deriv(1.1), units.Quantity
    ), "Potential method phi2deriv does not return Quantity when it should"
    assert isinstance(
        pot.vcirc(1.1), units.Quantity
    ), "Potential method vcirc does not return Quantity when it should"
    assert isinstance(
        pot.omegac(1.1), units.Quantity
    ), "Potential method omegac does not return Quantity when it should"
    assert isinstance(
        pot.epifreq(1.1), units.Quantity
    ), "Potential method epifreq does not return Quantity when it should"
    assert (
        pot.lindbladR(0.9) is None
    ), "Potential method lindbladR does not return None, even when it should return a Quantity, when it should"
    assert isinstance(
        pot.lindbladR(0.9, m="corot"), units.Quantity
    ), "Potential method lindbladR does not return Quantity when it should"
    assert isinstance(
        pot.vesc(1.3), units.Quantity
    ), "Potential method vesc does not return Quantity when it should"
    return None


def test_linearPotential_method_returntype():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0).toVertical(1.1)
    assert isinstance(
        pot(1.1), units.Quantity
    ), "Potential method __call__ does not return Quantity when it should"
    assert isinstance(
        pot.force(1.1), units.Quantity
    ), "Potential method Rforce does not return Quantity when it should"
    return None


def test_potential_method_returnunit():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0)
    try:
        pot(1.1, 0.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method __call__ does not return Quantity with the right units"
        )
    try:
        pot.Rforce(1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method Rforce does not return Quantity with the right units"
        )
    try:
        pot.rforce(1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method rforce does not return Quantity with the right units"
        )
    try:
        pot.zforce(1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method zforce does not return Quantity with the right units"
        )
    try:
        pot.phitorque(1.1, 0.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method phitorque does not return Quantity with the right units"
        )
    try:
        pot.dens(1.1, 0.1).to(units.kg / units.m**3)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method dens does not return Quantity with the right units"
        )
    try:
        pot.surfdens(1.1, 0.1).to(units.kg / units.m**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method surfdens does not return Quantity with the right units"
        )
    try:
        pot.mass(1.1, 0.1).to(units.kg)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method mass does not return Quantity with the right units"
        )
    try:
        pot.R2deriv(1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method R2deriv does not return Quantity with the right units"
        )
    try:
        pot.z2deriv(1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method z2deriv does not return Quantity with the right units"
        )
    try:
        pot.Rzderiv(1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method Rzderiv does not return Quantity with the right units"
        )
    try:
        pot.phi2deriv(1.1, 0.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method phi2deriv does not return Quantity with the right units"
        )
    try:
        pot.Rphideriv(1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method Rphideriv does not return Quantity with the right units"
        )
    try:
        pot.phizderiv(1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method phizderiv does not return Quantity with the right units"
        )
    try:
        pot.flattening(1.1, 0.1).to(units.dimensionless_unscaled)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method flattening does not return Quantity with the right units"
        )
    try:
        pot.vcirc(1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method vcirc does not return Quantity with the right units"
        )
    try:
        pot.dvcircdR(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method dvcircdR does not return Quantity with the right units"
        )
    try:
        pot.omegac(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method omegac does not return Quantity with the right units"
        )
    try:
        pot.epifreq(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method epifreq does not return Quantity with the right units"
        )
    try:
        pot.verticalfreq(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method verticalfreq does not return Quantity with the right units"
        )
    try:
        pot.lindbladR(0.9, m="corot").to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method lindbladR does not return Quantity with the right units"
        )
    try:
        pot.vesc(1.3).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method vesc does not return Quantity with the right units"
        )
    try:
        pot.rl(1.3).to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method rl does not return Quantity with the right units"
        )
    try:
        pot.rE(-1.14).to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method rE does not return Quantity with the right units"
        )
    try:
        pot.LcE(-1.14).to(units.km / units.s * units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method LcE does not return Quantity with the right units"
        )
    try:
        pot.vterm(45.0).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method vterm does not return Quantity with the right units"
        )
    try:
        pot.rtide(1.0, 0.0, M=1.0).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method rtide does not return Quantity with the right units"
        )
    try:
        pot.ttensor(1.0, 0.0).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method ttensor does not return Quantity with the right units"
        )
    try:
        pot.ttensor(1.0, 0.0, eigenval=True).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method ttensor does not return Quantity with the right units"
        )
    try:
        pot.zvc_range(-1.9, 0.2).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method zvc_range does not return Quantity with the right units"
        )
    try:
        pot.zvc(0.4, -1.9, 0.2).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method zvc does not return Quantity with the right units"
        )
    try:
        pot.rhalf().to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method rhalf does not return Quantity with the right units"
        )
    try:
        pot.tdyn(1.4).to(units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method tdyn does not return Quantity with the right units"
        )
    return None


def test_planarPotential_method_returnunit():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0).toPlanar()
    try:
        pot(1.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method __call__ does not return Quantity with the right units"
        )
    try:
        pot.Rforce(1.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method Rforce does not return Quantity with the right units"
        )
    try:
        pot.phitorque(1.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method phitorque does not return Quantity with the right units"
        )
    try:
        pot.R2deriv(1.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method R2deriv does not return Quantity with the right units"
        )
    try:
        pot.phi2deriv(1.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method phi2deriv does not return Quantity with the right units"
        )
    try:
        pot.Rphideriv(1.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method Rphideriv does not return Quantity with the right units"
        )
    try:
        pot.vcirc(1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method vcirc does not return Quantity with the right units"
        )
    try:
        pot.omegac(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method omegac does not return Quantity with the right units"
        )
    try:
        pot.epifreq(1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method epifreq does not return Quantity with the right units"
        )
    try:
        pot.lindbladR(0.9, m="corot").to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method lindbladR does not return Quantity with the right units"
        )
    try:
        pot.vesc(1.3).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method vesc does not return Quantity with the right units"
        )
    return None


def test_linearPotential_method_returnunit():
    from galpy.potential import PlummerPotential

    pot = PlummerPotential(normalize=True, ro=8.0, vo=220.0).toVertical(1.1)
    try:
        pot(1.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method __call__ does not return Quantity with the right units"
        )
    try:
        pot.force(1.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential method force does not return Quantity with the right units"
        )
    return None


def test_potential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    potu = PlummerPotential(normalize=True)
    assert (
        numpy.fabs(
            pot(1.1, 0.1).to(units.km**2 / units.s**2).value - potu(1.1, 0.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(1.1, 0.1).to(units.km / units.s**2).value * 10.0**13.0
            - potu.Rforce(1.1, 0.1) * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.rforce(1.1, 0.1).to(units.km / units.s**2).value * 10.0**13.0
            - potu.rforce(1.1, 0.1) * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential method rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.zforce(1.1, 0.1).to(units.km / units.s**2).value * 10.0**13.0
            - potu.zforce(1.1, 0.1) * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential method zforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.phitorque(1.1, 0.1).to(units.km**2 / units.s**2).value
            - potu.phitorque(1.1, 0.1) * vo**2
        )
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.dens(1.1, 0.1).to(units.Msun / units.pc**3).value
            - potu.dens(1.1, 0.1) * conversion.dens_in_msolpc3(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential method dens does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.surfdens(1.1, 0.1).to(units.Msun / units.pc**2).value
            - potu.surfdens(1.1, 0.1) * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential method surfdens does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.mass(1.1, 0.1).to(units.Msun).value / 10.0**10.0
            - potu.mass(1.1, 0.1) * conversion.mass_in_1010msol(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential method mass does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.R2deriv(1.1, 0.1).to(units.km**2 / units.s**2.0 / units.kpc**2).value
            - potu.R2deriv(1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.z2deriv(1.1, 0.1).to(units.km**2 / units.s**2.0 / units.kpc**2).value
            - potu.z2deriv(1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method z2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.Rzderiv(1.1, 0.1).to(units.km**2 / units.s**2.0 / units.kpc**2).value
            - potu.Rzderiv(1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method Rzderiv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.Rphideriv(1.1, 0.1).to(units.km**2 / units.s**2.0 / units.kpc).value
            - potu.Rphideriv(1.1, 0.1) * vo**2.0 / ro
        )
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.phi2deriv(1.1, 0.1).to(units.km**2 / units.s**2.0).value
            - potu.phi2deriv(1.1, 0.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.phizderiv(1.1, 0.1).to(units.km**2 / units.s**2.0 / units.kpc).value
            - potu.phizderiv(1.1, 0.1) * vo**2.0 / ro
        )
        < 10.0**-8.0
    ), "Potential method phizderiv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.flattening(1.1, 0.1).value - potu.flattening(1.1, 0.1))
        < 10.0**-8.0
    ), "Potential method flattening does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vcirc(1.1).to(units.km / units.s).value - potu.vcirc(1.1) * vo)
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.dvcircdR(1.1).to(units.km / units.s / units.kpc).value
            - potu.dvcircdR(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method dvcircdR does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.omegac(1.1).to(units.km / units.s / units.kpc).value
            - potu.omegac(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.epifreq(1.1).to(units.km / units.s / units.kpc).value
            - potu.epifreq(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.verticalfreq(1.1).to(units.km / units.s / units.kpc).value
            - potu.verticalfreq(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method verticalfreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.lindbladR(0.9, m="corot").to(units.kpc).value
            - potu.lindbladR(0.9, m="corot") * ro
        )
        < 10.0**-8.0
    ), "Potential method lindbladR does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vesc(1.1).to(units.km / units.s).value - potu.vesc(1.1) * vo)
        < 10.0**-8.0
    ), "Potential method vesc does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.rl(1.1).to(units.kpc).value - potu.rl(1.1) * ro) < 10.0**-8.0
    ), "Potential method rl does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.rE(-1.14).to(units.kpc).value - potu.rE(-1.14) * ro) < 10.0**-8.0
    ), "Potential method rE does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.LcE(-1.14).to(units.kpc * units.km / units.s).value
            - potu.LcE(-1.14) * ro * vo
        )
        < 10.0**-8.0
    ), "Potential method LcE does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vterm(45.0).to(units.km / units.s).value - potu.vterm(45.0) * vo)
        < 10.0**-8.0
    ), "Potential method vterm does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.rtide(1.0, 0.0, M=1.0).to(units.kpc).value
            - potu.rtide(1.0, 0.0, M=1.0) * ro
        )
        < 10.0**-8.0
    ), "Potential method rtide does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(1.0, 0.0).to(units.km**2 / units.s**2.0 / units.kpc**2).value
            - potu.ttensor(1.0, 0.0) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(1.0, 0.0, eigenval=True)
            .to(units.km**2 / units.s**2.0 / units.kpc**2)
            .value
            - potu.ttensor(1.0, 0.0, eigenval=True) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.zvc_range(-1.9, 0.2).to(units.kpc).value
            - potu.zvc_range(-1.9, 0.2) * ro
        )
        < 10.0**-8.0
    ), "Potential method zvc_range does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.zvc(0.4, -1.9, 0.2).to(units.kpc).value - potu.zvc(0.4, -1.9, 0.2) * ro
        )
        < 10.0**-8.0
    ), "Potential method zvc_range does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.rhalf().to(units.kpc).value - potu.rhalf() * ro) < 10.0**-8.0
    ), "Potential method rhalf does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.tdyn(1.4).to(units.Gyr).value
            - potu.tdyn(1.4) * conversion.time_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential method tdyn does not return the correct value as Quantity"
    return None


def test_planarPotential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo).toPlanar()
    potu = PlummerPotential(normalize=True).toPlanar()
    assert (
        numpy.fabs(pot(1.1).to(units.km**2 / units.s**2).value - potu(1.1) * vo**2.0)
        < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(1.1).to(units.km / units.s**2).value * 10.0**13.0
            - potu.Rforce(1.1) * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.phitorque(1.1).to(units.km**2 / units.s**2).value
            - potu.phitorque(1.1) * vo**2
        )
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.R2deriv(1.1).to(units.km**2 / units.s**2.0 / units.kpc**2).value
            - potu.R2deriv(1.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.Rphideriv(1.1).to(units.km**2 / units.s**2.0 / units.kpc).value
            - potu.Rphideriv(1.1) * vo**2.0 / ro
        )
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.phi2deriv(1.1).to(units.km**2 / units.s**2.0).value
            - potu.phi2deriv(1.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vcirc(1.1).to(units.km / units.s).value - potu.vcirc(1.1) * vo)
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.omegac(1.1).to(units.km / units.s / units.kpc).value
            - potu.omegac(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.epifreq(1.1).to(units.km / units.s / units.kpc).value
            - potu.epifreq(1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vesc(1.1).to(units.km / units.s).value - potu.vesc(1.1) * vo)
        < 10.0**-8.0
    ), "Potential method vesc does not return the correct value as Quantity"
    return None


def test_linearPotential_method_value():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo).toVertical(1.1)
    potu = PlummerPotential(normalize=True).toVertical(1.1)
    assert (
        numpy.fabs(pot(1.1).to(units.km**2 / units.s**2).value - potu(1.1) * vo**2.0)
        < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.force(1.1).to(units.km / units.s**2).value * 10.0**13.0
            - potu.force(1.1) * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential method force does not return the correct value as Quantity"
    return None


def test_potential_function_returntype():
    from galpy import potential
    from galpy.potential import PlummerPotential

    pot = [PlummerPotential(normalize=True, ro=8.0, vo=220.0)]
    assert isinstance(
        potential.evaluatePotentials(pot, 1.1, 0.1), units.Quantity
    ), "Potential function __call__ does not return Quantity when it should"
    assert isinstance(
        potential.evaluateRforces(pot, 1.1, 0.1), units.Quantity
    ), "Potential function Rforce does not return Quantity when it should"
    assert isinstance(
        potential.evaluaterforces(pot, 1.1, 0.1), units.Quantity
    ), "Potential function rforce does not return Quantity when it should"
    assert isinstance(
        potential.evaluatezforces(pot, 1.1, 0.1), units.Quantity
    ), "Potential function zforce does not return Quantity when it should"
    assert isinstance(
        potential.evaluatephitorques(pot, 1.1, 0.1), units.Quantity
    ), "Potential function phitorque does not return Quantity when it should"
    assert isinstance(
        potential.evaluateDensities(pot, 1.1, 0.1), units.Quantity
    ), "Potential function dens does not return Quantity when it should"
    assert isinstance(
        potential.evaluateSurfaceDensities(pot, 1.1, 0.1), units.Quantity
    ), "Potential function surfdens does not return Quantity when it should"
    assert isinstance(
        potential.evaluateR2derivs(pot, 1.1, 0.1), units.Quantity
    ), "Potential function R2deriv does not return Quantity when it should"
    assert isinstance(
        potential.evaluatez2derivs(pot, 1.1, 0.1), units.Quantity
    ), "Potential function z2deriv does not return Quantity when it should"
    assert isinstance(
        potential.evaluateRzderivs(pot, 1.1, 0.1), units.Quantity
    ), "Potential function Rzderiv does not return Quantity when it should"
    assert isinstance(
        potential.flattening(pot, 1.1, 0.1), units.Quantity
    ), "Potential function flattening does not return Quantity when it should"
    assert isinstance(
        potential.vcirc(pot, 1.1), units.Quantity
    ), "Potential function vcirc does not return Quantity when it should"
    assert isinstance(
        potential.dvcircdR(pot, 1.1), units.Quantity
    ), "Potential function dvcircdR does not return Quantity when it should"
    assert isinstance(
        potential.omegac(pot, 1.1), units.Quantity
    ), "Potential function omegac does not return Quantity when it should"
    assert isinstance(
        potential.epifreq(pot, 1.1), units.Quantity
    ), "Potential function epifreq does not return Quantity when it should"
    assert isinstance(
        potential.verticalfreq(pot, 1.1), units.Quantity
    ), "Potential function verticalfreq does not return Quantity when it should"
    assert (
        potential.lindbladR(pot, 0.9) is None
    ), "Potential function lindbladR does not return None, even when it should return a Quantity, when it should"
    assert isinstance(
        potential.lindbladR(pot, 0.9, m="corot"), units.Quantity
    ), "Potential function lindbladR does not return Quantity when it should"
    assert isinstance(
        potential.vesc(pot, 1.3), units.Quantity
    ), "Potential function vesc does not return Quantity when it should"
    assert isinstance(
        potential.rl(pot, 1.3), units.Quantity
    ), "Potential function rl does not return Quantity when it should"
    assert isinstance(
        potential.rE(pot, -1.14), units.Quantity
    ), "Potential function rE does not return Quantity when it should"
    assert isinstance(
        potential.LcE(pot, -1.14), units.Quantity
    ), "Potential function LcE does not return Quantity when it should"
    assert isinstance(
        potential.vterm(pot, 45.0), units.Quantity
    ), "Potential function vterm does not return Quantity when it should"
    assert isinstance(
        potential.rtide(pot, 1.0, 0.0, M=1.0), units.Quantity
    ), "Potential function rtide does not return Quantity when it should"
    assert isinstance(
        potential.ttensor(pot, 1.0, 0.0), units.Quantity
    ), "Potential function ttensor does not return Quantity when it should"
    assert isinstance(
        potential.ttensor(pot, 1.0, 0.0, eigenval=True), units.Quantity
    ), "Potential function ttensor does not return Quantity when it should"
    assert isinstance(
        potential.zvc_range(pot, -1.9, 0.2), units.Quantity
    ), "Potential function zvc_range does not return Quantity when it should"
    assert isinstance(
        potential.zvc(pot, 0.4, -1.9, 0.2), units.Quantity
    ), "Potential function zvc does not return Quantity when it should"
    assert isinstance(
        potential.rhalf(pot), units.Quantity
    ), "Potential function rhalf does not return Quantity when it should"
    assert isinstance(
        potential.tdyn(pot, 1.4), units.Quantity
    ), "Potential function tdyn does not return Quantity when it should"
    return None


def test_planarPotential_function_returntype():
    from galpy import potential
    from galpy.potential import PlummerPotential

    pot = [PlummerPotential(normalize=True, ro=8.0, vo=220.0).toPlanar()]
    assert isinstance(
        potential.evaluateplanarPotentials(pot, 1.1), units.Quantity
    ), "Potential function __call__ does not return Quantity when it should"
    assert isinstance(
        potential.evaluateplanarRforces(pot, 1.1), units.Quantity
    ), "Potential function Rforce does not return Quantity when it should"
    assert isinstance(
        potential.evaluateplanarphitorques(pot, 1.1), units.Quantity
    ), "Potential function phitorque does not return Quantity when it should"
    assert isinstance(
        potential.evaluateplanarR2derivs(pot, 1.1), units.Quantity
    ), "Potential function R2deriv does not return Quantity when it should"
    assert isinstance(
        potential.vcirc(pot, 1.1), units.Quantity
    ), "Potential function vcirc does not return Quantity when it should"
    assert isinstance(
        potential.omegac(pot, 1.1), units.Quantity
    ), "Potential function omegac does not return Quantity when it should"
    assert isinstance(
        potential.epifreq(pot, 1.1), units.Quantity
    ), "Potential function epifreq does not return Quantity when it should"
    assert (
        potential.lindbladR(pot, 0.9) is None
    ), "Potential function lindbladR does not return None, even when it should return a Quantity, when it should"
    assert isinstance(
        potential.lindbladR(pot, 0.9, m="corot"), units.Quantity
    ), "Potential function lindbladR does not return Quantity when it should"
    assert isinstance(
        potential.vesc(pot, 1.3), units.Quantity
    ), "Potential function vesc does not return Quantity when it should"
    return None


def test_linearPotential_function_returntype():
    from galpy import potential
    from galpy.potential import PlummerPotential

    pot = [PlummerPotential(normalize=True, ro=8.0, vo=220.0).toVertical(1.1)]
    assert isinstance(
        potential.evaluatelinearPotentials(pot, 1.1), units.Quantity
    ), "Potential function __call__ does not return Quantity when it should"
    assert isinstance(
        potential.evaluatelinearForces(pot, 1.1), units.Quantity
    ), "Potential function Rforce does not return Quantity when it should"
    return None


def test_potential_function_returnunit():
    from galpy import potential
    from galpy.potential import PlummerPotential

    pot = [PlummerPotential(normalize=True, ro=8.0, vo=220.0)]
    try:
        potential.evaluatePotentials(pot, 1.1, 0.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function __call__ does not return Quantity with the right units"
        )
    try:
        potential.evaluateRforces(pot, 1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function Rforce does not return Quantity with the right units"
        )
    try:
        potential.evaluaterforces(pot, 1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function rforce does not return Quantity with the right units"
        )
    try:
        potential.evaluatezforces(pot, 1.1, 0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function zforce does not return Quantity with the right units"
        )
    try:
        potential.evaluatephitorques(pot, 1.1, 0.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function phitorque does not return Quantity with the right units"
        )
    try:
        potential.evaluateDensities(pot, 1.1, 0.1).to(units.kg / units.m**3)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function dens does not return Quantity with the right units"
        )
    try:
        potential.evaluateSurfaceDensities(pot, 1.1, 0.1).to(units.kg / units.m**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function surfdens does not return Quantity with the right units"
        )
    try:
        potential.evaluateR2derivs(pot, 1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function R2deriv does not return Quantity with the right units"
        )
    try:
        potential.evaluatez2derivs(pot, 1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function z2deriv does not return Quantity with the right units"
        )
    try:
        potential.evaluateRzderivs(pot, 1.1, 0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function Rzderiv does not return Quantity with the right units"
        )
    try:
        potential.flattening(pot, 1.1, 0.1).to(units.dimensionless_unscaled)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function flattening does not return Quantity with the right units"
        )
    try:
        potential.vcirc(pot, 1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function vcirc does not return Quantity with the right units"
        )
    try:
        potential.dvcircdR(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function dvcircdR does not return Quantity with the right units"
        )
    try:
        potential.omegac(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function omegac does not return Quantity with the right units"
        )
    try:
        potential.epifreq(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function epifreq does not return Quantity with the right units"
        )
    try:
        potential.verticalfreq(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function verticalfreq does not return Quantity with the right units"
        )
    try:
        potential.lindbladR(pot, 0.9, m="corot").to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function lindbladR does not return Quantity with the right units"
        )
    try:
        potential.vesc(pot, 1.3).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function vesc does not return Quantity with the right units"
        )
    try:
        potential.rl(pot, 1.3).to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function rl does not return Quantity with the right units"
        )
    try:
        potential.rE(pot, -1.14).to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function rE does not return Quantity with the right units"
        )
    try:
        potential.LcE(pot, -1.14).to(units.km / units.s * units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function LcE does not return Quantity with the right units"
        )
    try:
        potential.vterm(pot, 45.0).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function vterm does not return Quantity with the right units"
        )
    try:
        potential.rtide(pot, 1.0, 0.0, M=1.0).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function rtide does not return Quantity with the right units"
        )
    try:
        potential.ttensor(pot, 1.0, 0.0).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function ttensor does not return Quantity with the right units"
        )
    try:
        potential.ttensor(pot, 1.0, 0.0, eigenval=True).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function ttensor does not return Quantity with the right units"
        )
    try:
        potential.zvc_range(pot, -1.9, 0.2).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function zvc_range does not return Quantity with the right units"
        )
    try:
        potential.zvc(pot, 0.4, -1.9, 0.2).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function zvc does not return Quantity with the right units"
        )
    try:
        potential.rhalf(pot).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function rhalf does not return Quantity with the right units"
        )
    try:
        potential.tdyn(pot, 1.4).to(units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function tdyn does not return Quantity with the right units"
        )
    return None


def test_planarPotential_function_returnunit():
    from galpy import potential
    from galpy.potential import LopsidedDiskPotential, PlummerPotential

    pot = [
        PlummerPotential(normalize=True, ro=8.0, vo=220.0).toPlanar(),
        LopsidedDiskPotential(ro=8.0 * units.kpc, vo=220.0 * units.km / units.s),
    ]
    try:
        potential.evaluateplanarPotentials(pot, 1.1, phi=0.1).to(
            units.km**2 / units.s**2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function __call__ does not return Quantity with the right units"
        )
    try:
        potential.evaluateplanarRforces(pot, 1.1, phi=0.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function Rforce does not return Quantity with the right units"
        )
    try:
        potential.evaluateplanarphitorques(pot, 1.1, phi=0.1).to(
            units.km**2 / units.s**2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function phitorque does not return Quantity with the right units"
        )
    try:
        potential.evaluateplanarR2derivs(pot, 1.1, phi=0.1).to(1 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function R2deriv does not return Quantity with the right units"
        )
    pot.pop()
    try:
        potential.vcirc(pot, 1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function vcirc does not return Quantity with the right units"
        )
    try:
        potential.omegac(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function omegac does not return Quantity with the right units"
        )
    try:
        potential.epifreq(pot, 1.1).to(1.0 / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function epifreq does not return Quantity with the right units"
        )
    try:
        potential.lindbladR(pot, 0.9, m="corot").to(units.km)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function lindbladR does not return Quantity with the right units"
        )
    try:
        potential.vesc(pot, 1.3).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function vesc does not return Quantity with the right units"
        )
    return None


def test_linearPotential_function_returnunit():
    from galpy import potential
    from galpy.potential import KGPotential

    pot = [KGPotential(ro=8.0 * units.kpc, vo=220.0 * units.km / units.s)]
    try:
        potential.evaluatelinearPotentials(pot, 1.1).to(units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function __call__ does not return Quantity with the right units"
        )
    try:
        potential.evaluatelinearForces(pot, 1.1).to(units.km / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "Potential function force does not return Quantity with the right units"
        )
    return None


def test_potential_function_value():
    from galpy import potential
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo)]
    potu = [PlummerPotential(normalize=True)]
    assert (
        numpy.fabs(
            potential.evaluatePotentials(pot, 1.1, 0.1)
            .to(units.km**2 / units.s**2)
            .value
            - potential.evaluatePotentials(potu, 1.1, 0.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRforces(pot, 1.1, 0.1).to(units.km / units.s**2).value
            * 10.0**13.0
            - potential.evaluateRforces(potu, 1.1, 0.1)
            * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluaterforces(pot, 1.1, 0.1).to(units.km / units.s**2).value
            * 10.0**13.0
            - potential.evaluaterforces(potu, 1.1, 0.1)
            * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential function rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatezforces(pot, 1.1, 0.1).to(units.km / units.s**2).value
            * 10.0**13.0
            - potential.evaluatezforces(potu, 1.1, 0.1)
            * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential function zforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatephitorques(pot, 1.1, 0.1)
            .to(units.km**2 / units.s**2)
            .value
            - potential.evaluatephitorques(potu, 1.1, 0.1) * vo**2
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateDensities(pot, 1.1, 0.1)
            .to(units.Msun / units.pc**3)
            .value
            - potential.evaluateDensities(potu, 1.1, 0.1)
            * conversion.dens_in_msolpc3(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential function dens does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateSurfaceDensities(pot, 1.1, 0.1)
            .to(units.Msun / units.pc**2)
            .value
            - potential.evaluateSurfaceDensities(potu, 1.1, 0.1)
            * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential function surfdens does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateR2derivs(pot, 1.1, 0.1)
            .to(units.km**2 / units.s**2.0 / units.kpc**2)
            .value
            - potential.evaluateR2derivs(potu, 1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatez2derivs(pot, 1.1, 0.1)
            .to(units.km**2 / units.s**2.0 / units.kpc**2)
            .value
            - potential.evaluatez2derivs(potu, 1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential function z2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRzderivs(pot, 1.1, 0.1)
            .to(units.km**2 / units.s**2.0 / units.kpc**2)
            .value
            - potential.evaluateRzderivs(potu, 1.1, 0.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential function Rzderiv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.flattening(pot, 1.1, 0.1).value
            - potential.flattening(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function flattening does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, 1.1).to(units.km / units.s).value
            - potential.vcirc(potu, 1.1) * vo
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.dvcircdR(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.dvcircdR(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function dvcircdR does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.omegac(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.epifreq(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.verticalfreq(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.verticalfreq(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function verticalfreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.lindbladR(pot, 0.9, m="corot").to(units.kpc).value
            - potential.lindbladR(potu, 0.9, m="corot") * ro
        )
        < 10.0**-8.0
    ), "Potential function lindbladR does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, 1.1).to(units.km / units.s).value
            - potential.vesc(potu, 1.1) * vo
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.rl(pot, 1.1).to(units.kpc).value - potential.rl(potu, 1.1) * ro
        )
        < 10.0**-8.0
    ), "Potential function rl does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.rE(pot, -1.14).to(units.kpc).value
            - potential.rE(potu, -1.14) * ro
        )
        < 10.0**-8.0
    ), "Potential function rE does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.LcE(pot, -1.14).to(units.kpc * units.km / units.s).value
            - potential.LcE(potu, -1.14) * ro * vo
        )
        < 10.0**-8.0
    ), "Potential function LcE does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vterm(pot, 45.0).to(units.km / units.s).value
            - potential.vterm(potu, 45.0) * vo
        )
        < 10.0**-8.0
    ), "Potential function vterm does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.rtide(pot, 1.0, 0.0, M=1.0).to(units.kpc).value
            - potential.rtide(potu, 1.0, 0.0, M=1.0) * ro
        )
        < 10.0**-8.0
    ), "Potential function rtide does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(pot, 1.0, 0.0)
            .to(units.km**2 / units.s**2 / units.kpc**2)
            .value
            - potential.ttensor(potu, 1.0, 0.0) * vo**2 / ro**2
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(pot, 1.0, 0.0, eigenval=True)
            .to(units.km**2 / units.s**2 / units.kpc**2)
            .value
            - potential.ttensor(potu, 1.0, 0.0, eigenval=True) * vo**2 / ro**2
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.zvc_range(pot, -1.9, 0.2).to(units.kpc).value
            - potential.zvc_range(potu, -1.9, 0.2) * ro
        )
        < 10.0**-8.0
    ), "Potential function zvc_range does not return the correct value as Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.zvc(pot, 0.4, -1.9, 0.2).to(units.kpc).value
            - potential.zvc(potu, 0.4, -1.9, 0.2) * ro
        )
        < 10.0**-8.0
    ), "Potential function zvc_range does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.rhalf(pot).to(units.kpc).value - potential.rhalf(potu) * ro
        )
        < 10.0**-8.0
    ), "Potential function rhalf does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.tdyn(pot, 1.4).to(units.Gyr).value
            - potential.tdyn(potu, 1.4) * conversion.time_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "Potential function tdyn does not return the correct value as Quantity"
    return None


def test_planarPotential_function_value():
    from galpy import potential
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toPlanar()]
    potu = [PlummerPotential(normalize=True).toPlanar()]
    assert (
        numpy.fabs(
            potential.evaluateplanarPotentials(pot, 1.1)
            .to(units.km**2 / units.s**2)
            .value
            - potential.evaluateplanarPotentials(potu, 1.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(pot, 1.1).to(units.km / units.s**2).value
            * 10.0**13.0
            - potential.evaluateplanarRforces(potu, 1.1)
            * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(pot, 1.1)
            .to(units.km**2 / units.s**2)
            .value
            - potential.evaluateplanarphitorques(potu, 1.1) * vo**2
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarR2derivs(pot, 1.1)
            .to(units.km**2 / units.s**2.0 / units.kpc**2)
            .value
            - potential.evaluateplanarR2derivs(potu, 1.1) * vo**2.0 / ro**2.0
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, 1.1).to(units.km / units.s).value
            - potential.vcirc(potu, 1.1) * vo
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.omegac(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, 1.1).to(units.km / units.s / units.kpc).value
            - potential.epifreq(potu, 1.1) * vo / ro
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, 1.1).to(units.km / units.s).value
            - potential.vesc(potu, 1.1) * vo
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value as Quantity"
    return None


def test_linearPotential_function_value():
    from galpy import potential
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toVertical(1.1)]
    potu = [PlummerPotential(normalize=True).toVertical(1.1)]
    assert (
        numpy.fabs(
            potential.evaluatelinearPotentials(pot, 1.1)
            .to(units.km**2 / units.s**2)
            .value
            - potential.evaluatelinearPotentials(potu, 1.1) * vo**2.0
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatelinearForces(pot, 1.1).to(units.km / units.s**2).value
            * 10.0**13.0
            - potential.evaluatelinearForces(potu, 1.1)
            * conversion.force_in_10m13kms2(vo, ro)
        )
        < 10.0**-4.0
    ), "Potential function force does not return the correct value as Quantity"
    return None


def test_potential_method_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    potu = PlummerPotential(normalize=True)
    assert (
        numpy.fabs(
            pot(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    # Few more cases for Rforce
    assert (
        numpy.fabs(
            pot.Rforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=9.0,
                use_physical=False,
            )
            - potu.Rforce(1.1 * 8.0 / 9.0, 0.1 * 8.0 / 9.0)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                vo=230.0,
                use_physical=False,
            )
            - potu.Rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.zforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.zforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phitorque(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phitorque(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.dens(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.dens(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method dens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.surfdens(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.surfdens(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method surfdens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.mass(1.1 * ro, 0.1 * ro, use_physical=False) - potu.mass(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method mass does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.R2deriv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.R2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.z2deriv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.z2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method z2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rzderiv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rzderiv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method Rzderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rphideriv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rphideriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phi2deriv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phi2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phizderiv(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phizderiv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method phizderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.flattening(1.1 * ro, 0.1 * ro, use_physical=False)
            - potu.flattening(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method flattening does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.vcirc(1.1 * ro, use_physical=False) - potu.vcirc(1.1))
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.dvcircdR(1.1 * ro, use_physical=False) - potu.dvcircdR(1.1))
        < 10.0**-8.0
    ), "Potential method dvcircdR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.omegac(1.1 * ro, use_physical=False) - potu.omegac(1.1))
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.epifreq(1.1 * ro, use_physical=False) - potu.epifreq(1.1))
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.verticalfreq(1.1 * ro, use_physical=False) - potu.verticalfreq(1.1)
        )
        < 10.0**-8.0
    ), "Potential method verticalfreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.vesc(1.1 * ro, use_physical=False) - potu.vesc(1.1)) < 10.0**-8.0
    ), "Potential method vesc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.lindbladR(
                0.9 * conversion.freq_in_Gyr(vo, ro.value) / units.Gyr,
                m="corot",
                use_physical=False,
            )
            - potu.lindbladR(0.9, m="corot")
        )
        < 10.0**-8.0
    ), "Potential method lindbladR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rl(1.1 * vo * ro * units.km / units.s, use_physical=False)
            - potu.rl(1.1)
        )
        < 10.0**-8.0
    ), "Potential method rl does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rE(-1.14 * vo**2 * units.km**2 / units.s**2, use_physical=False)
            - potu.rE(-1.14)
        )
        < 10.0**-8.0
    ), "Potential method rE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.LcE(-1.14 * vo**2 * units.km**2 / units.s**2, use_physical=False)
            - potu.LcE(-1.14)
        )
        < 10.0**-8.0
    ), "Potential method LcE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.vterm(45.0 * units.deg, use_physical=False) - potu.vterm(45.0))
        < 10.0**-8.0
    ), "Potential method vterm does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rtide(1.1 * ro, 0.1 * ro, M=10.0**9.0 * units.Msun, use_physical=False)
            - potu.rtide(1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value))
        )
        < 10.0**-8.0
    ), "Potential method rtide does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(1.1 * ro, 0.1 * ro, use_physical=False) - potu.ttensor(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(1.1 * ro, 0.1 * ro, eigenval=True, use_physical=False)
            - potu.ttensor(1.1, 0.1, eigenval=True)
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.zvc_range(
                -92000 * units.km**2 / units.s**2,
                45.0 * units.kpc * units.km / units.s,
                use_physical=False,
            )
            - potu.zvc_range(-92000 / vo**2, 45.0 / ro.to_value(units.kpc) / vo)
        )
        < 10.0**-8.0
    ), "Potential method zvc_range does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.zvc(
                0.4 * ro,
                -92000 * units.km**2 / units.s**2,
                45.0 * units.kpc * units.km / units.s,
                use_physical=False,
            )
            - potu.zvc(0.4, -92000 / vo**2, 45.0 / ro.to_value(units.kpc) / vo)
        )
        < 10.0**-8.0
    ), "Potential method zvc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.tdyn(1.1 * ro, use_physical=False) - potu.tdyn(1.1)) < 10.0**-8.0
    ), "Potential method tdyn does not return the correct value when input is Quantity"
    return None


def test_potential_method_inputAsQuantity_Rzaskwargs():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    potu = PlummerPotential(normalize=True)
    assert (
        numpy.fabs(
            pot(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    # Few more cases for Rforce
    assert (
        numpy.fabs(
            pot.Rforce(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=9.0,
                use_physical=False,
            )
            - potu.Rforce(1.1 * 8.0 / 9.0, 0.1 * 8.0 / 9.0)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rforce(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                vo=230.0,
                use_physical=False,
            )
            - potu.Rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rforce(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.rforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.zforce(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.zforce(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phitorque(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phitorque(1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.dens(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.dens(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method dens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.surfdens(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.surfdens(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method surfdens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.mass(R=1.1 * ro, z=0.1 * ro, use_physical=False) - potu.mass(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method mass does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.R2deriv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.R2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.z2deriv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.z2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method z2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rzderiv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rzderiv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method Rzderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.Rphideriv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.Rphideriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phi2deriv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phi2deriv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phizderiv(
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potu.phizderiv(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method phizderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.flattening(R=1.1 * ro, z=0.1 * ro, use_physical=False)
            - potu.flattening(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method flattening does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.vcirc(R=1.1 * ro, use_physical=False) - potu.vcirc(1.1))
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.dvcircdR(R=1.1 * ro, use_physical=False) - potu.dvcircdR(1.1))
        < 10.0**-8.0
    ), "Potential method dvcircdR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.omegac(R=1.1 * ro, use_physical=False) - potu.omegac(1.1))
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.epifreq(R=1.1 * ro, use_physical=False) - potu.epifreq(1.1))
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.verticalfreq(R=1.1 * ro, use_physical=False) - potu.verticalfreq(1.1)
        )
        < 10.0**-8.0
    ), "Potential method verticalfreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.vesc(R=1.1 * ro, use_physical=False) - potu.vesc(1.1))
        < 10.0**-8.0
    ), "Potential method vesc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.rtide(
                R=1.1 * ro, z=0.1 * ro, M=10.0**9.0 * units.Msun, use_physical=False
            )
            - potu.rtide(1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value))
        )
        < 10.0**-8.0
    ), "Potential method rtide does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(R=1.1 * ro, z=0.1 * ro, use_physical=False)
            - potu.ttensor(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            pot.ttensor(R=1.1 * ro, z=0.1 * ro, eigenval=True, use_physical=False)
            - potu.ttensor(1.1, 0.1, eigenval=True)
        )
        < 10.0**-8.0
    ), "Potential method ttensor does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(pot.tdyn(R=1.1 * ro, use_physical=False) - potu.tdyn(1.1))
        < 10.0**-8.0
    ), "Potential method tdyn does not return the correct value when input is Quantity"
    return None


def test_planarPotential_method_inputAsQuantity():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    # Force planarPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toPlanar()
    potu = PlummerPotential(normalize=True).toPlanar()
    assert (
        numpy.fabs(pot(1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.Rforce(1.1 * ro, use_physical=False) - potu.Rforce(1.1))
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.phitorque(1.1 * ro, use_physical=False) - potu.phitorque(1.1))
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.R2deriv(1.1 * ro, use_physical=False) - potu.R2deriv(1.1))
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.Rphideriv(1.1 * ro, use_physical=False) - potu.Rphideriv(1.1))
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.phi2deriv(1.1 * ro, use_physical=False) - potu.phi2deriv(1.1))
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vcirc(1.1 * ro, use_physical=False) - potu.vcirc(1.1))
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.omegac(1.1 * ro, use_physical=False) - potu.omegac(1.1))
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.epifreq(1.1 * ro, use_physical=False) - potu.epifreq(1.1))
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vesc(1.1 * ro, use_physical=False) - potu.vesc(1.1)) < 10.0**-8.0
    ), "Potential method vesc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            pot.lindbladR(
                0.9 * conversion.freq_in_Gyr(vo, ro.value) / units.Gyr,
                m="corot",
                use_physical=False,
            )
            - potu.lindbladR(0.9, m="corot")
        )
        < 10.0**-8.0
    ), "Potential method lindbladR does not return the correct value when input is Quantity"
    return None


def test_planarPotential_method_inputAsQuantity_Raskwarg():
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    # Force planarPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toPlanar()
    potu = PlummerPotential(normalize=True).toPlanar()
    assert (
        numpy.fabs(pot(R=1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.Rforce(R=1.1 * ro, use_physical=False) - potu.Rforce(1.1))
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.phitorque(R=1.1 * ro, use_physical=False) - potu.phitorque(1.1))
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.R2deriv(R=1.1 * ro, use_physical=False) - potu.R2deriv(1.1))
        < 10.0**-8.0
    ), "Potential method R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.Rphideriv(R=1.1 * ro, use_physical=False) - potu.Rphideriv(1.1))
        < 10.0**-8.0
    ), "Potential method Rphideriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.phi2deriv(R=1.1 * ro, use_physical=False) - potu.phi2deriv(1.1))
        < 10.0**-8.0
    ), "Potential method phi2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vcirc(R=1.1 * ro, use_physical=False) - potu.vcirc(1.1))
        < 10.0**-8.0
    ), "Potential method vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.omegac(R=1.1 * ro, use_physical=False) - potu.omegac(1.1))
        < 10.0**-8.0
    ), "Potential method omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.epifreq(R=1.1 * ro, use_physical=False) - potu.epifreq(1.1))
        < 10.0**-8.0
    ), "Potential method epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.vesc(R=1.1 * ro, use_physical=False) - potu.vesc(1.1))
        < 10.0**-8.0
    ), "Potential method vesc does not return the correct value as Quantity"
    return None


def test_linearPotential_method_inputAsQuantity():
    from galpy import potential
    from galpy.potential import PlummerPotential, SpiralArmsPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0 * units.km / units.s
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    # Force linearPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toVertical(1.1)
    potu = potential.RZToverticalPotential(PlummerPotential(normalize=True), 1.1 * ro)
    assert (
        numpy.fabs(pot(1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.force(1.1 * ro, use_physical=False) - potu.force(1.1))
        < 10.0**-4.0
    ), "Potential method force does not return the correct value as Quantity"
    # also toVerticalPotential w/ non-axi
    pot = SpiralArmsPotential(ro=ro, vo=vo)
    # Force linearPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toVertical(
        1.1,
        10.0 / 180.0 * numpy.pi,
        t0=1.0
        / conversion.time_in_Gyr(
            vo.to(units.km / units.s).value, ro.to(units.kpc).value
        ),
    )
    potu = potential.toVerticalPotential(
        SpiralArmsPotential(), 1.1 * ro, phi=10 * units.deg, t0=1.0 * units.Gyr
    )
    assert (
        numpy.fabs(pot(1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.force(1.1 * ro, use_physical=False) - potu.force(1.1))
        < 10.0**-4.0
    ), "Potential method force does not return the correct value as Quantity"
    return None


def test_linearPotential_method_inputAsQuantity_xaskwarg():
    from galpy import potential
    from galpy.potential import PlummerPotential, SpiralArmsPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0 * units.km / units.s
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    # Force linearPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toVertical(1.1)
    potu = potential.RZToverticalPotential(PlummerPotential(normalize=True), 1.1 * ro)
    assert (
        numpy.fabs(pot(x=1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.force(x=1.1 * ro, use_physical=False) - potu.force(1.1))
        < 10.0**-4.0
    ), "Potential method force does not return the correct value as Quantity"
    # also toVerticalPotential w/ non-axi
    pot = SpiralArmsPotential(ro=ro, vo=vo)
    # Force linearPotential setup with default
    pot._ro = None
    pot._roSet = False
    pot._vo = None
    pot._voSet = False
    pot = pot.toVertical(
        1.1,
        10.0 / 180.0 * numpy.pi,
        t0=1.0
        / conversion.time_in_Gyr(
            vo.to(units.km / units.s).value, ro.to(units.kpc).value
        ),
    )
    potu = potential.toVerticalPotential(
        SpiralArmsPotential(), 1.1 * ro, phi=10 * units.deg, t0=1.0 * units.Gyr
    )
    assert (
        numpy.fabs(pot(x=1.1 * ro, use_physical=False) - potu(1.1)) < 10.0**-8.0
    ), "Potential method __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(pot.force(x=1.1 * ro, use_physical=False) - potu.force(1.1))
        < 10.0**-4.0
    ), "Potential method force does not return the correct value as Quantity"
    return None


def test_dissipativeforce_method_inputAsQuantity():
    from galpy.potential import ChandrasekharDynamicalFrictionForce
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = ChandrasekharDynamicalFrictionForce(GMs=0.1, rhm=1.2 / 8.0, ro=ro, vo=vo)
    potu = ChandrasekharDynamicalFrictionForce(GMs=0.1, rhm=1.2 / 8.0)
    assert (
        numpy.fabs(
            pot.Rforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potu.Rforce(
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential method Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.zforce(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potu.zforce(
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential method zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            pot.phitorque(
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potu.phitorque(
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential method phitorque does not return the correct value when input is Quantity"
    return None


def test_potential_function_inputAsQuantity():
    from galpy import potential
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo)]
    potu = [PlummerPotential(normalize=True)]
    assert (
        numpy.fabs(
            potential.evaluatePotentials(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatePotentials(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRforces(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluateRforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluaterforces(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluaterforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatezforces(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatezforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatephitorques(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatephitorques(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateDensities(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateDensities(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function dens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateSurfaceDensities(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateSurfaceDensities(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function surfdens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateR2derivs(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateR2derivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatez2derivs(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatez2derivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function z2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRzderivs(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateRzderivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function Rzderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.flattening(pot, 1.1 * ro, 0.1 * ro, use_physical=False)
            - potential.flattening(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function flattening does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, 1.1 * ro, use_physical=False)
            - potential.vcirc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.dvcircdR(pot, 1.1 * ro, use_physical=False)
            - potential.dvcircdR(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function dvcircdR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, 1.1 * ro, use_physical=False)
            - potential.omegac(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, 1.1 * ro, use_physical=False)
            - potential.epifreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.verticalfreq(pot, 1.1 * ro, use_physical=False)
            - potential.verticalfreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function verticalfreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, 1.1 * ro, use_physical=False)
            - potential.vesc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.lindbladR(
                pot,
                0.9 * conversion.freq_in_Gyr(vo, ro.value) / units.Gyr,
                m="corot",
                use_physical=False,
            )
            - potential.lindbladR(potu, 0.9, m="corot")
        )
        < 10.0**-8.0
    ), "Potential method lindbladR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.lindbladR(
                pot[0],
                0.9 * conversion.freq_in_Gyr(vo, ro.value) / units.Gyr,
                m="corot",
                use_physical=False,
            )
            - potential.lindbladR(potu, 0.9, m="corot")
        )
        < 10.0**-8.0
    ), "Potential method lindbladR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rl(pot, 1.1 * vo * ro * units.km / units.s, use_physical=False)
            - potential.rl(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function rl does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rl(pot[0], 1.1 * vo * ro * units.km / units.s, use_physical=False)
            - potential.rl(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function rl does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rE(
                pot, -1.14 * vo**2 * units.km**2 / units.s**2, use_physical=False
            )
            - potential.rE(potu, -1.14)
        )
        < 10.0**-8.0
    ), "Potential function rE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rE(
                pot[0],
                -1.14 * vo**2 * units.km**2 / units.s**2,
                use_physical=False,
            )
            - potential.rE(potu, -1.14)
        )
        < 10.0**-8.0
    ), "Potential function rE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.LcE(
                pot, -1.14 * vo**2 * units.km**2 / units.s**2, use_physical=False
            )
            - potential.LcE(potu, -1.14)
        )
        < 10.0**-8.0
    ), "Potential function LcE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.LcE(
                pot[0],
                -1.14 * vo**2 * units.km**2 / units.s**2,
                use_physical=False,
            )
            - potential.LcE(potu, -1.14)
        )
        < 10.0**-8.0
    ), "Potential function LcE does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.vterm(pot, 45.0 * units.deg, use_physical=False)
            - potential.vterm(potu, 45.0)
        )
        < 10.0**-8.0
    ), "Potential function vterm does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rtide(
                pot, 1.1 * ro, 0.1 * ro, M=10.0**9.0 * units.Msun, use_physical=False
            )
            - potential.rtide(
                potu, 1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value)
            )
        )
        < 10.0**-8.0
    ), "Potential function rtide does not return the correct value when input is Quantity"
    # Test non-list for M as well, bc units done in rtide special, and do GM
    assert (
        numpy.fabs(
            potential.rtide(
                pot[0],
                1.1 * ro,
                0.1 * ro,
                M=constants.G * 10.0**9.0 * units.Msun,
                use_physical=False,
            )
            - potential.rtide(
                potu, 1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value)
            )
        )
        < 10.0**-8.0
    ), "Potential function rtide does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(pot, 1.1 * ro, 0.1 * ro, use_physical=False)
            - potential.ttensor(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(
                pot, 1.1 * ro, 0.1 * ro, eigenval=True, use_physical=False
            )
            - potential.ttensor(potu, 1.1, 0.1, eigenval=True)
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.zvc_range(
                pot,
                -92000 * units.km**2 / units.s**2,
                45.0 * units.kpc * units.km / units.s,
                use_physical=False,
            )
            - potential.zvc_range(
                potu, -92000 / vo**2, 45.0 / ro.to_value(units.kpc) / vo
            )
        )
        < 10.0**-8.0
    ), "Potential function zvc_range does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.zvc(
                pot,
                0.4 * ro,
                -92000 * units.km**2 / units.s**2,
                45.0 * units.kpc * units.km / units.s,
                use_physical=False,
            )
            - potential.zvc(
                potu, 0.4, -92000 / vo**2, 45.0 / ro.to_value(units.kpc) / vo
            )
        )
        < 10.0**-8.0
    ), "Potential function zvc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.tdyn(pot, 1.1 * ro, use_physical=False)
            - potential.tdyn(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function tdyn does not return the correct value when input is Quantity"
    return None


def test_potential_function_inputAsQuantity_Rzaskwargs():
    from galpy import potential
    from galpy.potential import PlummerPotential
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo)]
    potu = [PlummerPotential(normalize=True)]
    assert (
        numpy.fabs(
            potential.evaluatePotentials(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatePotentials(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRforces(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluateRforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluaterforces(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluaterforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatezforces(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatezforces(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatephitorques(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatephitorques(potu, 1.1, 0.1)
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateDensities(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateDensities(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function dens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateSurfaceDensities(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateSurfaceDensities(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function surfdens does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateR2derivs(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateR2derivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatez2derivs(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluatez2derivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function z2deriv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRzderivs(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                use_physical=False,
            )
            - potential.evaluateRzderivs(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function Rzderiv does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.flattening(pot, R=1.1 * ro, z=0.1 * ro, use_physical=False)
            - potential.flattening(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function flattening does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, R=1.1 * ro, use_physical=False)
            - potential.vcirc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.dvcircdR(pot, R=1.1 * ro, use_physical=False)
            - potential.dvcircdR(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function dvcircdR does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, R=1.1 * ro, use_physical=False)
            - potential.omegac(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, R=1.1 * ro, use_physical=False)
            - potential.epifreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.verticalfreq(pot, R=1.1 * ro, use_physical=False)
            - potential.verticalfreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function verticalfreq does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, R=1.1 * ro, use_physical=False)
            - potential.vesc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.rtide(
                pot,
                R=1.1 * ro,
                z=0.1 * ro,
                M=10.0**9.0 * units.Msun,
                use_physical=False,
            )
            - potential.rtide(
                potu, 1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value)
            )
        )
        < 10.0**-8.0
    ), "Potential function rtide does not return the correct value when input is Quantity"
    # Test non-list for M as well, bc units done in rtide special, and do GM
    assert (
        numpy.fabs(
            potential.rtide(
                pot[0],
                R=1.1 * ro,
                z=0.1 * ro,
                M=constants.G * 10.0**9.0 * units.Msun,
                use_physical=False,
            )
            - potential.rtide(
                potu, 1.1, 0.1, M=10.0**9.0 / conversion.mass_in_msol(vo, ro.value)
            )
        )
        < 10.0**-8.0
    ), "Potential function rtide does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(pot, R=1.1 * ro, z=0.1 * ro, use_physical=False)
            - potential.ttensor(potu, 1.1, 0.1)
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value when input is Quantity"
    assert numpy.all(
        numpy.fabs(
            potential.ttensor(
                pot, R=1.1 * ro, z=0.1 * ro, eigenval=True, use_physical=False
            )
            - potential.ttensor(potu, 1.1, 0.1, eigenval=True)
        )
        < 10.0**-8.0
    ), "Potential function ttensor does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.tdyn(pot, R=1.1 * ro, use_physical=False)
            - potential.tdyn(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function tdyn does not return the correct value when input is Quantity"
    return None


def test_dissipativeforce_function_inputAsQuantity():
    from galpy import potential
    from galpy.potential import ChandrasekharDynamicalFrictionForce
    from galpy.util import conversion

    ro, vo = 8.0 * units.kpc, 220.0
    pot = ChandrasekharDynamicalFrictionForce(GMs=0.1, rhm=1.2 / 8.0, ro=ro, vo=vo)
    potu = ChandrasekharDynamicalFrictionForce(GMs=0.1, rhm=1.2 / 8.0)
    assert (
        numpy.fabs(
            potential.evaluatezforces(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluatezforces(
                potu,
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential function zforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluateRforces(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluateRforces(
                potu,
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            potential.evaluatephitorques(
                pot,
                1.1 * ro,
                0.1 * ro,
                phi=10.0 * units.deg,
                t=10.0 * units.Gyr,
                ro=8.0 * units.kpc,
                vo=220.0 * units.km / units.s,
                v=numpy.array([10.0, 200.0, -20.0]) * units.km / units.s,
                use_physical=False,
            )
            - potential.evaluatephitorques(
                potu,
                1.1,
                0.1,
                phi=10.0 / 180.0 * numpy.pi,
                v=numpy.array([10.0, 200.0, -20.0]) / vo,
            )
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value when input is Quantity"
    return None


def test_planarPotential_function_inputAsQuantity():
    from galpy import potential
    from galpy.potential import PlummerPotential

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toPlanar()]
    potu = [PlummerPotential(normalize=True).toPlanar()]
    assert (
        numpy.fabs(
            potential.evaluateplanarPotentials(pot, 1.1 * ro, use_physical=False)
            - potential.evaluateplanarPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(pot, 1.1 * ro, use_physical=False)
            - potential.evaluateplanarRforces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(pot, 1.1 * ro, use_physical=False)
            - potential.evaluateplanarphitorques(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarR2derivs(pot, 1.1 * ro, use_physical=False)
            - potential.evaluateplanarR2derivs(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, 1.1 * ro, use_physical=False)
            - potential.vcirc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, 1.1 * ro, use_physical=False)
            - potential.omegac(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, 1.1 * ro, use_physical=False)
            - potential.epifreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, 1.1 * ro, use_physical=False)
            - potential.vesc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value as Quantity"
    return None


def test_planarPotential_function_inputAsQuantity_Raskwarg():
    from galpy import potential
    from galpy.potential import PlummerPotential

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toPlanar()]
    potu = [PlummerPotential(normalize=True).toPlanar()]
    assert (
        numpy.fabs(
            potential.evaluateplanarPotentials(pot, R=1.1 * ro, use_physical=False)
            - potential.evaluateplanarPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarRforces(pot, R=1.1 * ro, use_physical=False)
            - potential.evaluateplanarRforces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function Rforce does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarphitorques(pot, R=1.1 * ro, use_physical=False)
            - potential.evaluateplanarphitorques(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function phitorque does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluateplanarR2derivs(pot, R=1.1 * ro, use_physical=False)
            - potential.evaluateplanarR2derivs(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function R2deriv does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vcirc(pot, R=1.1 * ro, use_physical=False)
            - potential.vcirc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vcirc does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.omegac(pot, R=1.1 * ro, use_physical=False)
            - potential.omegac(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function omegac does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.epifreq(pot, R=1.1 * ro, use_physical=False)
            - potential.epifreq(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function epifreq does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.vesc(pot, R=1.1 * ro, use_physical=False)
            - potential.vesc(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function vesc does not return the correct value as Quantity"
    return None


def test_linearPotential_function_inputAsQuantity():
    from galpy import potential
    from galpy.potential import PlummerPotential, SpiralArmsPotential

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toVertical(1.1 * ro)]
    potu = potential.RZToverticalPotential([PlummerPotential(normalize=True)], 1.1 * ro)
    assert (
        numpy.fabs(
            potential.evaluatelinearPotentials(pot, 1.1 * ro, use_physical=False)
            - potential.evaluatelinearPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatelinearForces(pot, 1.1 * ro, use_physical=False)
            - potential.evaluatelinearForces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function force does not return the correct value as Quantity"
    # Also toVerticalPotential, with non-axi
    pot = [
        SpiralArmsPotential(ro=ro, vo=vo).toVertical(
            (1.1 * ro).to(units.kpc).value / 8.0,
            phi=20.0 * units.deg,
            t0=1.0 * units.Gyr,
        )
    ]
    potu = potential.toVerticalPotential(
        [SpiralArmsPotential()], 1.1 * ro, phi=20.0 * units.deg, t0=1.0 * units.Gyr
    )
    assert (
        numpy.fabs(
            potential.evaluatelinearPotentials(pot, 1.1 * ro, use_physical=False)
            - potential.evaluatelinearPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatelinearForces(pot, 1.1 * ro, use_physical=False)
            - potential.evaluatelinearForces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function force does not return the correct value as Quantity"
    return None


def test_linearPotential_function_inputAsQuantity_xaskwarg():
    from galpy import potential
    from galpy.potential import PlummerPotential, SpiralArmsPotential

    ro, vo = 8.0 * units.kpc, 220.0
    pot = [PlummerPotential(normalize=True, ro=ro, vo=vo).toVertical(1.1 * ro)]
    potu = potential.RZToverticalPotential([PlummerPotential(normalize=True)], 1.1 * ro)
    assert (
        numpy.fabs(
            potential.evaluatelinearPotentials(pot, x=1.1 * ro, use_physical=False)
            - potential.evaluatelinearPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatelinearForces(pot, x=1.1 * ro, use_physical=False)
            - potential.evaluatelinearForces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function force does not return the correct value as Quantity"
    # Also toVerticalPotential, with non-axi
    pot = [
        SpiralArmsPotential(ro=ro, vo=vo).toVertical(
            (1.1 * ro).to(units.kpc).value / 8.0,
            phi=20.0 * units.deg,
            t0=1.0 * units.Gyr,
        )
    ]
    potu = potential.toVerticalPotential(
        [SpiralArmsPotential()], 1.1 * ro, phi=20.0 * units.deg, t0=1.0 * units.Gyr
    )
    assert (
        numpy.fabs(
            potential.evaluatelinearPotentials(pot, x=1.1 * ro, use_physical=False)
            - potential.evaluatelinearPotentials(potu, 1.1)
        )
        < 10.0**-8.0
    ), "Potential function __call__ does not return the correct value as Quantity"
    assert (
        numpy.fabs(
            potential.evaluatelinearForces(pot, x=1.1 * ro, use_physical=False)
            - potential.evaluatelinearForces(potu, 1.1)
        )
        < 10.0**-4.0
    ), "Potential function force does not return the correct value as Quantity"
    return None


def test_plotting_inputAsQuantity():
    from galpy import potential
    from galpy.potential import PlummerPotential

    ro, vo = 8.0 * units.kpc, 220.0
    pot = PlummerPotential(normalize=True, ro=ro, vo=vo)
    pot.plot(
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    pot.plotDensity(
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    pot.plotSurfaceDensity(
        xmin=1.0 * units.kpc,
        xmax=4.0 * units.kpc,
        ymin=-4.0 * units.kpc,
        ymax=4.0 * units.kpc,
    )
    potential.plotPotentials(
        pot,
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    potential.plotPotentials(
        [pot],
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    potential.plotDensities(
        pot,
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    potential.plotDensities(
        [pot],
        rmin=1.0 * units.kpc,
        rmax=4.0 * units.kpc,
        zmin=-4.0 * units.kpc,
        zmax=4.0 * units.kpc,
    )
    potential.plotSurfaceDensities(
        pot,
        xmin=1.0 * units.kpc,
        xmax=4.0 * units.kpc,
        ymin=-4.0 * units.kpc,
        ymax=4.0 * units.kpc,
    )
    potential.plotSurfaceDensities(
        [pot],
        xmin=1.0 * units.kpc,
        xmax=4.0 * units.kpc,
        ymin=-4.0 * units.kpc,
        ymax=4.0 * units.kpc,
    )
    # Planar
    plpot = pot.toPlanar()
    plpot.plot(
        Rrange=[1.0 * units.kpc, 8.0 * units.kpc],
        xrange=[-4.0 * units.kpc, 4.0 * units.kpc],
        yrange=[-6.0 * units.kpc, 7.0 * units.kpc],
    )
    potential.plotplanarPotentials(
        plpot,
        Rrange=[1.0 * units.kpc, 8.0 * units.kpc],
        xrange=[-4.0 * units.kpc, 4.0 * units.kpc],
        yrange=[-6.0 * units.kpc, 7.0 * units.kpc],
    )
    potential.plotplanarPotentials(
        [plpot],
        Rrange=[1.0 * units.kpc, 8.0 * units.kpc],
        xrange=[-4.0 * units.kpc, 4.0 * units.kpc],
        yrange=[-6.0 * units.kpc, 7.0 * units.kpc],
    )
    # Rotcurve
    pot.plotRotcurve(Rrange=[1.0 * units.kpc, 8.0 * units.kpc], ro=10.0, vo=250.0)
    plpot.plotRotcurve(
        Rrange=[1.0 * units.kpc, 8.0 * units.kpc],
        ro=10.0 * units.kpc,
        vo=250.0 * units.km / units.s,
    )
    potential.plotRotcurve(pot, Rrange=[1.0 * units.kpc, 8.0 * units.kpc])
    potential.plotRotcurve([pot], Rrange=[1.0 * units.kpc, 8.0 * units.kpc])
    # Escapecurve
    pot.plotEscapecurve(Rrange=[1.0 * units.kpc, 8.0 * units.kpc], ro=10.0, vo=250.0)
    plpot.plotEscapecurve(
        Rrange=[1.0 * units.kpc, 8.0 * units.kpc],
        ro=10.0 * units.kpc,
        vo=250.0 * units.km / units.s,
    )
    potential.plotEscapecurve(pot, Rrange=[1.0 * units.kpc, 8.0 * units.kpc])
    potential.plotEscapecurve([pot], Rrange=[1.0 * units.kpc, 8.0 * units.kpc])
    return None


def test_potential_ampunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential
    from galpy.util import conversion

    ro, vo = 9.0, 210.0
    # Burkert
    pot = potential.BurkertPotential(
        amp=0.1 * units.Msun / units.pc**3.0, a=2.0, ro=ro, vo=vo
    )
    # density at r=a should be amp/4
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 / 4.0
        )
        < 10.0**-8.0
    ), "BurkertPotential w/ amp w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot = potential.DoubleExponentialDiskPotential(
        amp=0.1 * units.Msun / units.pc**3.0, hr=2.0, hz=0.2, ro=ro, vo=vo
    )
    # density at zero should be amp
    assert (
        numpy.fabs(
            pot.dens(0.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1
        )
        < 10.0**-8.0
    ), "DoubleExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun, a=2.0, alpha=1.5, beta=3.5, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun, a=2.0, alpha=2.0, beta=5.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # JaffePotential
    pot = potential.JaffePotential(amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo)
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "JaffePotential w/ amp w/ units does not behave as expected"
    # HernquistPotential
    pot = potential.HernquistPotential(amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo)
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "HernquistPotential w/ amp w/ units does not behave as expected"
    # NFWPotential
    pot = potential.NFWPotential(amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo)
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "NFWPotential w/ amp w/ units does not behave as expected"
    # TwoPowerTriaxialPotential
    pot = potential.TwoPowerTriaxialPotential(
        amp=20.0 * units.Msun, a=2.0, b=0.3, c=1.4, alpha=1.5, beta=3.5, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerTriaxialPotential w/ amp w/ units does not behave as expected"
    # TwoPowerTriaxialPotential with integer powers
    pot = potential.TwoPowerTriaxialPotential(
        amp=20.0 * units.Msun, a=2.0, b=0.3, c=1.4, alpha=2.0, beta=5.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TwoPowerTriaxialPotential w/ amp w/ units does not behave as expected"
    # TriaxialJaffePotential
    pot = potential.TriaxialJaffePotential(
        amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo, b=0.3, c=1.4
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialJaffePotential w/ amp w/ units does not behave as expected"
    # TriaxialHernquistPotential
    pot = potential.TriaxialHernquistPotential(
        amp=20.0 * units.Msun, a=2.0, b=0.4, c=1.4, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TriaxialHernquistPotential w/ amp w/ units does not behave as expected"
    # TriaxialNFWPotential
    pot = potential.TriaxialNFWPotential(
        amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo, b=1.3, c=0.4
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialNFWPotential w/ amp w/ units does not behave as expected"
    # SCFPotential, default = spherical Hernquist
    pot = potential.SCFPotential(amp=20.0 * units.Msun, a=2.0, ro=ro, vo=vo)
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "SCFPotential w/ amp w/ units does not behave as expected"
    # FlattenedPowerPotential
    pot = potential.FlattenedPowerPotential(
        amp=40000.0 * units.km**2 / units.s**2,
        r1=1.0,
        q=0.9,
        alpha=0.5,
        core=0.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(2.0, 1.0, use_physical=False) * vo**2.0
            + 40000.0 / 0.5 / (2.0**2.0 + (1.0 / 0.9) ** 2.0) ** 0.25
        )
        < 10.0**-8.0
    ), "FlattenedPowerPotential w/ amp w/ units does not behave as expected"
    # IsochronePotential
    pot = potential.IsochronePotential(amp=20.0 * units.Msun, b=2.0, ro=ro, vo=vo)
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / (2.0 + numpy.sqrt(4.0 + 16.0))
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "IsochronePotential w/ amp w/ units does not behave as expected"
    # KeplerPotential
    pot = potential.KeplerPotential(amp=20.0 * units.Msun, ro=ro, vo=vo)
    # Check mass
    assert (
        numpy.fabs(
            pot.mass(100.0, use_physical=False) * conversion.mass_in_msol(vo, ro) - 20.0
        )
        < 10.0**-8.0
    ), "KeplerPotential w/ amp w/ units does not behave as expected"
    # KuzminKutuzovStaeckelPotential
    pot = potential.KuzminKutuzovStaeckelPotential(
        amp=20.0 * units.Msun, Delta=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.KuzminKutuzovStaeckelPotential(
        amp=(20.0 * units.Msun * constants.G)
        .to(units.kpc * units.km**2 / units.s**2)
        .value
        / ro
        / vo**2,
        Delta=2.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "KuzminKutuzovStaeckelPotential w/ amp w/ units does not behave as expected"
    # LogarithmicHaloPotential
    pot = potential.LogarithmicHaloPotential(
        amp=40000 * units.km**2 / units.s**2, core=0.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0 - 20000 * numpy.log(16.0)
        )
        < 10.0**-8.0
    ), "LogarithmicHaloPotential w/ amp w/ units does not behave as expected"
    # MiyamotoNagaiPotential
    pot = potential.MiyamotoNagaiPotential(
        amp=20 * units.Msun, a=2.0, b=0.5, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (2.0 + numpy.sqrt(1.0 + 0.25)) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "MiyamotoNagaiPotential( w/ amp w/ units does not behave as expected"
    # KuzminDiskPotential
    pot = potential.KuzminDiskPotential(amp=20 * units.Msun, a=2.0, ro=ro, vo=vo)
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (2.0 + 1.0) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "KuzminDiskPotential( w/ amp w/ units does not behave as expected"
    # MN3ExponentialDiskPotential
    pot = potential.MN3ExponentialDiskPotential(
        amp=0.1 * units.Msun / units.pc**3.0, hr=2.0, hz=0.2, ro=ro, vo=vo
    )
    # density at hr should be
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.2, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-2.0)
        )
        < 10.0**-3.0
    ), "MN3ExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # PlummerPotential
    pot = potential.PlummerPotential(amp=20 * units.Msun, b=0.5, ro=ro, vo=vo)
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + 0.25)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "PlummerPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotential
    pot = potential.PowerSphericalPotential(
        amp=10.0**10.0 * units.Msun, r1=1.0, alpha=2.0, ro=ro, vo=vo
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(1.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / ro**3.0
        )
        < 10.0**-8.0
    ), "PowerSphericalPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot = potential.PowerSphericalPotentialwCutoff(
        amp=0.1 * units.Msun / units.pc**3, r1=1.0, alpha=2.0, rc=2.0, ro=ro, vo=vo
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(1.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-0.25)
        )
        < 10.0**-8.0
    ), "PowerSphericalPotentialwCutoff w/ amp w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot = potential.PseudoIsothermalPotential(
        amp=10.0**10.0 * units.Msun, a=2.0, ro=ro, vo=vo
    )
    # density at a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / 4.0 / numpy.pi / 8.0 / 2.0 / ro**3.0
        )
        < 10.0**-8.0
    ), "PseudoIsothermalPotential w/ amp w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot = potential.RazorThinExponentialDiskPotential(
        amp=40.0 * units.Msun / units.pc**2, hr=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.RazorThinExponentialDiskPotential(
        amp=(40.0 * units.Msun / units.pc**2 * constants.G)
        .to(1 / units.kpc * units.km**2 / units.s**2)
        .value
        * ro
        / vo**2,
        hr=2.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RazorThinExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # SoftenedNeedleBarPotential
    pot = potential.SoftenedNeedleBarPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SoftenedNeedleBarPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SoftenedNeedleBarPotential w/ amp w/ units does not behave as expected"
    # FerrersPotential
    pot = potential.FerrersPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.FerrersPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "FerrersPotential w/ amp w/ units does not behave as expected"
    # # SpiralArmsPotential
    # pot= potential.SpiralArmsPotential(amp=0.3*units.Msun / units.pc**3)
    # assert numpy.fabs(pot(1.,0.,phi=1.,use_physical=False)*) < 10.**-8., "SpiralArmsPotential w/ amp w/ units does not behave as expected"
    # SphericalShellPotential
    pot = potential.SphericalShellPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, ro=ro, vo=vo
    )
    pot_nounits = potential.SphericalShellPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SphericalShellPotential w/ amp w/ units does not behave as expected"
    # RingPotential
    pot = potential.RingPotential(amp=4.0 * 10.0**10.0 * units.Msun, ro=ro, vo=vo)
    pot_nounits = potential.RingPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RingPotential w/ amp w/ units does not behave as expected"
    # PerfectEllipsoidPotential
    pot = potential.PerfectEllipsoidPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=2.0, ro=ro, vo=vo, b=1.3, c=0.4
    )
    pot_nounits = potential.PerfectEllipsoidPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), a=2.0, ro=ro, vo=vo, b=1.3, c=0.4
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "PerfectEllipsoidPotential w/ amp w/ units does not behave as expected"
    # HomogeneousSpherePotential
    pot = potential.HomogeneousSpherePotential(
        amp=0.1 * units.Msun / units.pc**3.0, R=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.HomogeneousSpherePotential(
        amp=0.1 / conversion.dens_in_msolpc3(vo, ro), R=2.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.1, 0.2, phi=1.0, use_physical=False)
            - pot_nounits(1.1, 0.2, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "HomogeneousSpherePotential w/ amp w/ units does not behave as expected"
    # TriaxialGaussianPotential
    pot = potential.TriaxialGaussianPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, sigma=2.0, ro=ro, vo=vo, b=1.3, c=0.4
    )
    pot_nounits = potential.TriaxialGaussianPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        sigma=2.0,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "TriaxialGaussianPotential w/ amp w/ units does not behave as expected"
    # NullPotential
    pot = potential.NullPotential(amp=(200.0 * units.km / units.s) ** 2, ro=ro, vo=vo)
    pot_nounits = potential.NullPotential(amp=(200 / vo) ** 2.0, ro=ro, vo=vo)
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "NullPotential w/ amp w/ units does not behave as expected"
    return None


def test_potential_ampunits_altunits():
    # Test that input units for potential amplitudes behave as expected, alternative where G*M is given
    from galpy import potential
    from galpy.util import conversion

    ro, vo = 9.0, 210.0
    # Burkert
    pot = potential.BurkertPotential(
        amp=0.1 * units.Msun / units.pc**3.0 * constants.G, a=2.0, ro=ro, vo=vo
    )
    # density at r=a should be amp/4
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 / 4.0
        )
        < 10.0**-8.0
    ), "BurkertPotential w/ amp w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot = potential.DoubleExponentialDiskPotential(
        amp=0.1 * units.Msun / units.pc**3.0 * constants.G,
        hr=2.0,
        hz=0.2,
        ro=ro,
        vo=vo,
    )
    # density at zero should be amp
    assert (
        numpy.fabs(
            pot.dens(0.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1
        )
        < 10.0**-8.0
    ), "DoubleExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, alpha=1.5, beta=3.5, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, alpha=2.0, beta=5.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ amp w/ units does not behave as expected"
    # JaffePotential
    pot = potential.JaffePotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "JaffePotential w/ amp w/ units does not behave as expected"
    # HernquistPotential
    pot = potential.HernquistPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "HernquistPotential w/ amp w/ units does not behave as expected"
    # NFWPotential
    pot = potential.NFWPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "NFWPotential w/ amp w/ units does not behave as expected"
    # SCFPotential, default = Hernquist
    pot = potential.SCFPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "SCFPotential w/ amp w/ units does not behave as expected"
    # TwoPowerTriaxialPotential
    pot = potential.TwoPowerTriaxialPotential(
        amp=20.0 * units.Msun * constants.G,
        a=2.0,
        b=0.3,
        c=1.4,
        alpha=1.5,
        beta=3.5,
        ro=ro,
        vo=vo,
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerTriaxialPotential w/ amp w/ units does not behave as expected"
    # TwoPowerTriaxialPotential with integer powers
    pot = potential.TwoPowerTriaxialPotential(
        amp=20.0 * units.Msun * constants.G,
        a=2.0,
        b=0.5,
        c=0.3,
        alpha=2.0,
        beta=5.0,
        ro=ro,
        vo=vo,
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TwoPowerTriaxialPotential w/ amp w/ units does not behave as expected"
    # TriaxialJaffePotential
    pot = potential.TriaxialJaffePotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, b=0.4, c=0.9, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialJaffePotential w/ amp w/ units does not behave as expected"
    # TriaxialHernquistPotential
    pot = potential.TriaxialHernquistPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, b=1.3, c=0.3, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TriaxialHernquistPotential w/ amp w/ units does not behave as expected"
    # TriaxialNFWPotential
    pot = potential.TriaxialNFWPotential(
        amp=20.0 * units.Msun * constants.G, a=2.0, b=1.2, c=0.6, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialNFWPotential w/ amp w/ units does not behave as expected"
    # IsochronePotential
    pot = potential.IsochronePotential(
        amp=20.0 * units.Msun * constants.G, b=2.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / (2.0 + numpy.sqrt(4.0 + 16.0))
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "IsochronePotential w/ amp w/ units does not behave as expected"
    # KeplerPotential
    pot = potential.KeplerPotential(amp=20.0 * units.Msun * constants.G, ro=ro, vo=vo)
    # Check mass
    assert (
        numpy.fabs(
            pot.mass(100.0, use_physical=False) * conversion.mass_in_msol(vo, ro) - 20.0
        )
        < 10.0**-8.0
    ), "KeplerPotential w/ amp w/ units does not behave as expected"
    # KuzminKutuzovStaeckelPotential
    pot = potential.KuzminKutuzovStaeckelPotential(
        amp=20.0 * units.Msun * constants.G, Delta=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.KuzminKutuzovStaeckelPotential(
        amp=(20.0 * units.Msun * constants.G)
        .to(units.kpc * units.km**2 / units.s**2)
        .value
        / ro
        / vo**2,
        Delta=2.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "KuzminKutuzovStaeckelPotential w/ amp w/ units does not behave as expected"
    # MiyamotoNagaiPotential
    pot = potential.MiyamotoNagaiPotential(
        amp=20 * units.Msun * constants.G, a=2.0, b=0.5, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (2.0 + numpy.sqrt(1.0 + 0.25)) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "MiyamotoNagaiPotential( w/ amp w/ units does not behave as expected"
    # KuzminDiskPotential
    pot = potential.KuzminDiskPotential(
        amp=20 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (2.0 + 1.0) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "KuzminDiskPotential( w/ amp w/ units does not behave as expected"
    # MN3ExponentialDiskPotential
    pot = potential.MN3ExponentialDiskPotential(
        amp=0.1 * units.Msun * constants.G / units.pc**3.0,
        hr=2.0,
        hz=0.2,
        ro=ro,
        vo=vo,
    )
    # density at hr should be
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.2, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-2.0)
        )
        < 10.0**-3.0
    ), "MN3ExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # PlummerPotential
    pot = potential.PlummerPotential(
        amp=20 * units.Msun * constants.G, b=0.5, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + 0.25)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "PlummerPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotential
    pot = potential.PowerSphericalPotential(
        amp=10.0**10.0 * units.Msun * constants.G, r1=1.0, alpha=2.0, ro=ro, vo=vo
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(1.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / ro**3.0
        )
        < 10.0**-8.0
    ), "PowerSphericalPotential w/ amp w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot = potential.PowerSphericalPotentialwCutoff(
        amp=0.1 * units.Msun * constants.G / units.pc**3,
        r1=1.0,
        alpha=2.0,
        rc=2.0,
        ro=ro,
        vo=vo,
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(1.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-0.25)
        )
        < 10.0**-8.0
    ), "PowerSphericalPotentialwCutoff w/ amp w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot = potential.PseudoIsothermalPotential(
        amp=10.0**10.0 * units.Msun * constants.G, a=2.0, ro=ro, vo=vo
    )
    # density at a
    assert (
        numpy.fabs(
            pot.dens(2.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / 4.0 / numpy.pi / 8.0 / 2.0 / ro**3.0
        )
        < 10.0**-8.0
    ), "PseudoIsothermalPotential w/ amp w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot = potential.RazorThinExponentialDiskPotential(
        amp=40.0 * units.Msun * constants.G / units.pc**2, hr=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.RazorThinExponentialDiskPotential(
        amp=(40.0 * units.Msun / units.pc**2 * constants.G)
        .to(1 / units.kpc * units.km**2 / units.s**2)
        .value
        * ro
        / vo**2,
        hr=2.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RazorThinExponentialDiskPotential w/ amp w/ units does not behave as expected"
    # SoftenedNeedleBarPotential
    pot = potential.SoftenedNeedleBarPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G,
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SoftenedNeedleBarPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SoftenedNeedleBarPotential w/ amp w/ units does not behave as expected"
    # FerrersPotential
    pot = potential.FerrersPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G,
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.FerrersPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        a=1.0,
        b=2.0,
        c=3.0,
        pa=0.0,
        omegab=0.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "FerrersPotential w/ amp w/ units does not behave as expected"
    # SphericalShellPotential
    pot = potential.SphericalShellPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G, ro=ro, vo=vo
    )
    pot_nounits = potential.SphericalShellPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SphericalShellPotential w/ amp w/ units does not behave as expected"
    # RingPotential
    pot = potential.RingPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G, ro=ro, vo=vo
    )
    pot_nounits = potential.RingPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RingPotential w/ amp w/ units does not behave as expected"
    # PerfectEllipsoidPotential
    pot = potential.PerfectEllipsoidPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G,
        a=2.0,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    pot_nounits = potential.PerfectEllipsoidPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro), a=2.0, ro=ro, vo=vo, b=1.3, c=0.4
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "PerfectEllipsoidPotential w/ amp w/ units does not behave as expected"
    # HomogeneousSpherePotential
    pot = potential.HomogeneousSpherePotential(
        amp=0.1 * units.Msun / units.pc**3.0 * constants.G, R=2.0, ro=ro, vo=vo
    )
    pot_nounits = potential.HomogeneousSpherePotential(
        amp=0.1 / conversion.dens_in_msolpc3(vo, ro), R=2.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.1, 0.2, phi=1.0, use_physical=False)
            - pot_nounits(1.1, 0.2, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "HomogeneousSpherePotential w/ amp w/ units does not behave as expected"
    # TriaxialGaussianPotential
    pot = potential.TriaxialGaussianPotential(
        amp=4.0 * 10.0**10.0 * units.Msun * constants.G,
        sigma=2.0,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    pot_nounits = potential.TriaxialGaussianPotential(
        amp=4.0 / conversion.mass_in_1010msol(vo, ro),
        sigma=2.0,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "TriaxialGaussianPotential w/ amp w/ units does not behave as expected"
    return None


def test_potential_ampunits_wrongunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential

    ro, vo = 9.0, 210.0
    # Burkert
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.BurkertPotential(
            amp=0.1 * units.Msun / units.pc**2.0, a=2.0, ro=ro, vo=vo
        )
    # DoubleExponentialDiskPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.DoubleExponentialDiskPotential(
            amp=0.1 * units.Msun / units.pc**2.0 * constants.G,
            hr=2.0,
            hz=0.2,
            ro=ro,
            vo=vo,
        )
    # TwoPowerSphericalPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TwoPowerSphericalPotential(
            amp=20.0 * units.Msun / units.pc**3,
            a=2.0,
            alpha=1.5,
            beta=3.5,
            ro=ro,
            vo=vo,
        )
    # TwoPowerSphericalPotential with integer powers
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TwoPowerSphericalPotential(
            amp=20.0 * units.Msun / units.pc**3 * constants.G,
            a=2.0,
            alpha=2.0,
            beta=5.0,
            ro=ro,
            vo=vo,
        )
    # JaffePotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.JaffePotential(amp=20.0 * units.kpc, a=2.0, ro=ro, vo=vo)
    # HernquistPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.HernquistPotential(
            amp=20.0 * units.Msun / units.pc**3, a=2.0, ro=ro, vo=vo
        )
    # NFWPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.NFWPotential(amp=20.0 * units.km**2 / units.s**2, a=2.0, ro=ro, vo=vo)
    # SCFPotential, default = Hernquist
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.SCFPotential(amp=20.0 * units.Msun / units.pc**3, a=2.0, ro=ro, vo=vo)
    # TwoPowerTriaxialPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TwoPowerTriaxialPotential(
            amp=20.0 * units.Msun / units.pc**3,
            a=2.0,
            alpha=1.5,
            beta=3.5,
            ro=ro,
            vo=vo,
        )
    # TriaxialJaffePotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TriaxialJaffePotential(amp=20.0 * units.kpc, a=2.0, ro=ro, vo=vo)
    # TriaxialHernquistPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TriaxialHernquistPotential(
            amp=20.0 * units.Msun / units.pc**3, a=2.0, ro=ro, vo=vo
        )
    # TriaxialNFWPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TriaxialNFWPotential(
            amp=20.0 * units.km**2 / units.s**2, a=2.0, ro=ro, vo=vo
        )
    # FlattenedPowerPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.FlattenedPowerPotential(
            amp=40000.0 * units.km**2 / units.s,
            r1=1.0,
            q=0.9,
            alpha=0.5,
            core=0.0,
            ro=ro,
            vo=vo,
        )
    # IsochronePotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.IsochronePotential(
            amp=20.0 * units.km**2 / units.s**2, b=2.0, ro=ro, vo=vo
        )
    # KeplerPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.KeplerPotential(amp=20.0 * units.Msun / units.pc**3, ro=ro, vo=vo)
    # KuzminKutuzovStaeckelPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.KuzminKutuzovStaeckelPotential(
            amp=20.0 * units.Msun / units.pc**2, Delta=2.0, ro=ro, vo=vo
        )
    # LogarithmicHaloPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.LogarithmicHaloPotential(amp=40 * units.Msun, core=0.0, ro=ro, vo=vo)
    # MiyamotoNagaiPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.MiyamotoNagaiPotential(
            amp=20 * units.km**2 / units.s**2, a=2.0, b=0.5, ro=ro, vo=vo
        )
    # KuzminDiskPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.KuzminDiskPotential(
            amp=20 * units.km**2 / units.s**2, a=2.0, ro=ro, vo=vo
        )
    # MN3ExponentialDiskPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.MN3ExponentialDiskPotential(
            amp=0.1 * units.Msun * constants.G, hr=2.0, hz=0.2, ro=ro, vo=vo
        )
    # PlummerPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.PlummerPotential(
            amp=20 * units.km**2 / units.s**2, b=0.5, ro=ro, vo=vo
        )
    # PowerSphericalPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.PowerSphericalPotential(
            amp=10.0**10.0 * units.Msun / units.pc**3,
            r1=1.0,
            alpha=2.0,
            ro=ro,
            vo=vo,
        )
    # PowerSphericalPotentialwCutoff
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.PowerSphericalPotentialwCutoff(
            amp=0.1 * units.Msun / units.pc**2,
            r1=1.0,
            alpha=2.0,
            rc=2.0,
            ro=ro,
            vo=vo,
        )
    # PseudoIsothermalPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.PseudoIsothermalPotential(
            amp=10.0**10.0 * units.Msun / units.pc**3, a=2.0, ro=ro, vo=vo
        )
    # RazorThinExponentialDiskPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.RazorThinExponentialDiskPotential(
            amp=40.0 * units.Msun / units.pc**3, hr=2.0, ro=ro, vo=vo
        )
    # SoftenedNeedleBarPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.SoftenedNeedleBarPotential(
            amp=40.0 * units.Msun / units.pc**2, a=2.0, ro=ro, vo=vo
        )
    # FerrersPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.FerrersPotential(
            amp=40.0 * units.Msun / units.pc**2, a=2.0, ro=ro, vo=vo
        )
    # DiskSCFPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.DiskSCFPotential(amp=40.0 * units.Msun / units.pc**2)
    # SpiralArmsPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.SpiralArmsPotential(amp=10**10 * units.Msun)
    # SphericalShellPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.SphericalShellPotential(
            amp=40.0 * units.Msun / units.pc**2, a=2.0, ro=ro, vo=vo
        )
    # RingPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.RingPotential(
            amp=40.0 * units.Msun / units.pc**2, a=2.0, ro=ro, vo=vo
        )
    # PerfectEllipsoidPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.PerfectEllipsoidPotential(
            amp=40.0 * units.Msun / units.pc**2, a=2.0, ro=ro, vo=vo, b=1.3, c=0.4
        )
    # HomogeneousSpherePotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.HomogeneousSpherePotential(amp=40.0 * units.Msun, R=2.0, ro=ro, vo=vo)
    # TriaxialGaussianPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.TriaxialGaussianPotential(
            amp=40.0 * units.Msun / units.pc**2, sigma=2.0, ro=ro, vo=vo, b=1.3, c=0.4
        )
    # NullPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.NullPotential(amp=40.0 * units.Msun, ro=ro, vo=vo)
    return None


def test_potential_paramunits():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    # Burkert
    pot = potential.BurkertPotential(
        amp=0.1 * units.Msun / units.pc**3.0, a=2.0 * units.kpc, ro=ro, vo=vo
    )
    # density at r=a should be amp/4
    assert (
        numpy.fabs(
            pot.dens(2.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 / 4.0
        )
        < 10.0**-8.0
    ), "BurkertPotential w/ parameters w/ units does not behave as expected"
    # DoubleExponentialDiskPotential
    pot = potential.DoubleExponentialDiskPotential(
        amp=0.1 * units.Msun / units.pc**3.0,
        hr=4.0 * units.kpc,
        hz=200.0 * units.pc,
        ro=ro,
        vo=vo,
    )
    # density at zero should be amp
    assert (
        numpy.fabs(
            pot.dens(0.0, 0.0, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1
        )
        < 10.0**-8.0
    ), "DoubleExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # density at 1. is...
    assert (
        numpy.fabs(
            pot.dens(1.0, 0.1, use_physical=False) * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-1.0 / 4.0 * ro - 0.1 / 0.2 * ro)
        )
        < 10.0**-8.0
    ), "DoubleExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # TwoPowerSphericalPotential
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun, a=10.0 * units.kpc, alpha=1.5, beta=3.5, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # TwoPowerSphericalPotential with integer powers
    pot = potential.TwoPowerSphericalPotential(
        amp=20.0 * units.Msun, a=12000.0 * units.lyr, alpha=2.0, beta=5.0, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(
                (12000.0 * units.lyr).to(units.kpc).value / ro, 0.0, use_physical=False
            )
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TwoPowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # JaffePotential
    pot = potential.JaffePotential(
        amp=20.0 * units.Msun, a=0.02 * units.Mpc, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(20.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "JaffePotential w/ parameters w/ units does not behave as expected"
    # HernquistPotential
    pot = potential.HernquistPotential(
        amp=20.0 * units.Msun, a=10.0 * units.kpc, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "HernquistPotential w/ parameters w/ units does not behave as expected"
    # NFWPotential
    pot = potential.NFWPotential(
        amp=20.0 * units.Msun, a=15.0 * units.kpc, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(15.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "NFWPotential w/ parameters w/ units does not behave as expected"
    # NFWPotential, rmax,vmax
    pot = potential.NFWPotential(
        rmax=10.0 * units.kpc, vmax=175.0 * units.km / units.s, ro=ro, vo=vo
    )
    # Check velocity at r=rmax
    assert (
        numpy.fabs(pot.vcirc(10.0 / ro, use_physical=False) * vo - 175.0) < 10.0**-8.0
    ), "NFWPotential w/ parameters w/ units does not behave as expected"
    # SCFPotential, default = Hernquist
    pot = potential.SCFPotential(
        amp=20.0 * units.Msun, a=10.0 * units.kpc, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "SCFPotential w/ parameters w/ units does not behave as expected"
    # TwoPowerTriaxialPotential
    pot = potential.TwoPowerTriaxialPotential(
        amp=20.0 * units.Msun, a=10.0 * units.kpc, alpha=1.5, beta=3.5, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TwoPowerTriaxialPotential w/ parameters w/ units does not behave as expected"
    # TriaxialJaffePotential
    pot = potential.TriaxialJaffePotential(
        amp=20.0 * units.Msun, a=0.02 * units.Mpc, b=0.2, c=0.8, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(20.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialJaffePotential w/ parameters w/ units does not behave as expected"
    # TriaxialHernquistPotential
    pot = potential.TriaxialHernquistPotential(
        amp=20.0 * units.Msun, a=10.0 * units.kpc, b=0.7, c=0.9, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 8.0
        )
        < 10.0**-8.0
    ), "TriaxialHernquistPotential w/ parameters w/ units does not behave as expected"
    # TriaxialNFWPotential
    pot = potential.TriaxialNFWPotential(
        amp=20.0 * units.Msun, a=15.0 * units.kpc, b=1.3, c=0.2, ro=ro, vo=vo
    )
    # Check density at r=a
    assert (
        numpy.fabs(
            pot.dens(15.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 20.0 / 4.0 / numpy.pi / 8.0 / ro**3.0 / 10.0**9.0 / 4.0
        )
        < 10.0**-8.0
    ), "TriaxialNFWPotential w/ parameters w/ units does not behave as expected"
    # Also do pa
    pot = potential.TriaxialNFWPotential(
        amp=20.0 * units.Msun,
        a=15.0 * units.kpc,
        pa=30.0 * units.deg,
        b=1.3,
        c=0.2,
        ro=ro,
        vo=vo,
    )
    assert (
        numpy.fabs(numpy.arccos(pot._rot[0, 0]) - 30.0 / 180.0 * numpy.pi) < 10.0**-8.0
    ), "TriaxialNFWPotential w/ parameters w/ units does not behave as expected"
    # FlattenedPowerPotential
    pot = potential.FlattenedPowerPotential(
        amp=40000.0 * units.km**2 / units.s**2,
        r1=10.0 * units.kpc,
        q=0.9,
        alpha=0.5,
        core=1.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(2.0, 1.0, use_physical=False) * vo**2.0
            + 40000.0
            * (10.0 / ro) ** 0.5
            / 0.5
            / (2.0**2.0 + (1.0 / 0.9) ** 2.0 + (1.0 / ro) ** 2.0) ** 0.25
        )
        < 10.0**-8.0
    ), "FlattenedPowerPotential w/ parameters w/ units does not behave as expected"
    # IsochronePotential
    pot = potential.IsochronePotential(
        amp=20.0 * units.Msun, b=10.0 * units.kpc, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / (10.0 / ro + numpy.sqrt((10.0 / ro) ** 2.0 + 16.0))
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "IsochronePotential w/ parameters w/ units does not behave as expected"
    # KuzminKutuzovStaeckelPotential
    pot = potential.KuzminKutuzovStaeckelPotential(
        amp=20.0 * units.Msun, Delta=10.0 * units.kpc, ro=ro, vo=vo
    )
    pot_nounits = potential.KuzminKutuzovStaeckelPotential(
        amp=(20.0 * units.Msun * constants.G)
        .to(units.kpc * units.km**2 / units.s**2)
        .value
        / ro
        / vo**2,
        Delta=10.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "KuzminKutuzovStaeckelPotential w/ parameters w/ units does not behave as expected"
    # LogarithmicHaloPotential
    pot = potential.LogarithmicHaloPotential(
        amp=40000 * units.km**2 / units.s**2, core=1.0 * units.kpc, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            - 20000 * numpy.log(16.0 + (1.0 / ro) ** 2.0)
        )
        < 10.0**-8.0
    ), "LogarithmicHaloPotential w/ parameters w/ units does not behave as expected"
    # DehnenBarPotential
    pot = potential.DehnenBarPotential(
        amp=1.0,
        omegab=50.0 * units.km / units.s / units.kpc,
        rb=4.0 * units.kpc,
        Af=1290.0 * units.km**2 / units.s**2,
        barphi=20.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.DehnenBarPotential(
        amp=1.0,
        omegab=50.0 * ro / vo,
        rb=4.0 / ro,
        Af=1290.0 / vo**2.0,
        barphi=20.0 / 180.0 * numpy.pi,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "DehnenBarPotential w/ parameters w/ units does not behave as expected"
    # DehnenBarPotential, alternative setup
    pot = potential.DehnenBarPotential(
        amp=1.0,
        rolr=8.0 * units.kpc,
        chi=0.8,
        alpha=0.02,
        beta=0.2,
        barphi=20.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.DehnenBarPotential(
        amp=1.0,
        rolr=8.0 / ro,
        chi=0.8,
        alpha=0.02,
        beta=0.2,
        barphi=20.0 / 180.0 * numpy.pi,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "DehnenBarPotential w/ parameters w/ units does not behave as expected"
    # MiyamotoNagaiPotential
    pot = potential.MiyamotoNagaiPotential(
        amp=20 * units.Msun, a=5.0 * units.kpc, b=300.0 * units.pc, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (5.0 / ro + numpy.sqrt(1.0 + (0.3 / ro) ** 2.0)) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "MiyamotoNagaiPotential( w/ parameters w/ units does not behave as expected"
    # KuzminDiskPotential
    pot = potential.KuzminDiskPotential(
        amp=20 * units.Msun, a=5.0 * units.kpc, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 1.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (5.0 / ro + 1.0) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "KuzminDiskPotential( w/ parameters w/ units does not behave as expected"
    # MN3ExponentialDiskPotential
    pot = potential.MN3ExponentialDiskPotential(
        amp=0.1 * units.Msun / units.pc**3.0,
        hr=6.0 * units.kpc,
        hz=300.0 * units.pc,
        ro=ro,
        vo=vo,
    )
    # density at hr should be
    assert (
        numpy.fabs(
            pot.dens(6.0 / ro, 0.3 / ro, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-2.0)
        )
        < 10.0**-3.0
    ), "MN3ExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # PlummerPotential
    pot = potential.PlummerPotential(
        amp=20 * units.Msun, b=5.0 * units.kpc, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False) * vo**2.0
            + (20.0 * units.Msun * constants.G)
            .to(units.pc * units.km**2 / units.s**2)
            .value
            / numpy.sqrt(16.0 + (5.0 / ro) ** 2.0)
            / ro
            / 1000.0
        )
        < 10.0**-8.0
    ), "PlummerPotential w/ parameters w/ units does not behave as expected"
    # PowerSphericalPotential
    pot = potential.PowerSphericalPotential(
        amp=10.0**10.0 * units.Msun, r1=10.0 * units.kpc, alpha=2.0, ro=ro, vo=vo
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / ro**3.0 / (10.0 / ro) ** 3.0
        )
        < 10.0**-8.0
    ), "PowerSphericalPotential w/ parameters w/ units does not behave as expected"
    # PowerSphericalPotentialwCutoff
    pot = potential.PowerSphericalPotentialwCutoff(
        amp=0.1 * units.Msun / units.pc**3,
        r1=10.0 * units.kpc,
        alpha=2.0,
        rc=12.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    # density at r1
    assert (
        numpy.fabs(
            pot.dens(10.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 0.1 * numpy.exp(-((10.0 / 12.0) ** 2.0))
        )
        < 10.0**-8.0
    ), "PowerSphericalPotentialwCutoff w/ parameters w/ units does not behave as expected"
    # PseudoIsothermalPotential
    pot = potential.PseudoIsothermalPotential(
        amp=10.0**10.0 * units.Msun, a=20.0 * units.kpc, ro=ro, vo=vo
    )
    # density at a
    assert (
        numpy.fabs(
            pot.dens(20.0 / ro, 0.0, use_physical=False)
            * conversion.dens_in_msolpc3(vo, ro)
            - 10.0 / 4.0 / numpy.pi / (20.0 / ro) ** 3.0 / 2.0 / ro**3.0
        )
        < 10.0**-8.0
    ), "PseudoIsothermalPotential w/ parameters w/ units does not behave as expected"
    # RazorThinExponentialDiskPotential
    pot = potential.RazorThinExponentialDiskPotential(
        amp=40.0 * units.Msun / units.pc**2, hr=10.0 * units.kpc, ro=ro, vo=vo
    )
    pot_nounits = potential.RazorThinExponentialDiskPotential(
        amp=(40.0 * units.Msun / units.pc**2 * constants.G)
        .to(1 / units.kpc * units.km**2 / units.s**2)
        .value
        * ro
        / vo**2,
        hr=10.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, use_physical=False)
            - pot_nounits(4.0, 0.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RazorThinExponentialDiskPotential w/ parameters w/ units does not behave as expected"
    # SoftenedNeedleBarPotential
    pot = potential.SoftenedNeedleBarPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=10.0 * units.kpc,
        b=2.0 * units.kpc,
        c=3.0 * units.kpc,
        pa=10.0 * units.deg,
        omegab=20.0 * units.km / units.s / units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SoftenedNeedleBarPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=10.0 / ro,
        b=2.0 / ro,
        c=3.0 / ro,
        pa=10.0 / 180.0 * numpy.pi,
        omegab=20.0 / conversion.freq_in_kmskpc(vo, ro),
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SoftenedNeedleBarPotential w/ amp w/ units does not behave as expected"
    # FerrersPotential
    pot = potential.FerrersPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=10.0 * units.kpc,
        b=2.0,
        c=3.0,
        pa=10.0 * units.deg,
        omegab=20.0 * units.km / units.s / units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.FerrersPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=10.0 / ro,
        b=2.0,
        c=3.0,
        pa=10.0 / 180.0 * numpy.pi,
        omegab=20.0 / conversion.freq_in_kmskpc(vo, ro),
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "FerrersPotential w/ amp w/ units does not behave as expected"
    # DiskSCFPotential
    pot = potential.DiskSCFPotential(
        dens=lambda R, z: 1.0,  # doesn't matter
        Sigma=[
            {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
            {"type": "expwhole", "h": 1.0 / 3.0, "amp": 1.0, "Rhole": 0.5},
        ],
        hz=[{"type": "exp", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 27.0}],
        a=8.0 * units.kpc,
        N=2,
        L=2,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.DiskSCFPotential(
        dens=lambda R, z: 1.0,  # doesn't matter
        Sigma=[
            {"type": "exp", "h": 1.0 / 3.0, "amp": 1.0},
            {"type": "expwhole", "h": 1.0 / 3.0, "amp": 1.0, "Rhole": 0.5},
        ],
        hz=[{"type": "exp", "h": 1.0 / 27.0}, {"type": "sech2", "h": 1.0 / 27.0}],
        a=8.0 / ro,
        N=2,
        L=2,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "DiskSCFPotential w/ a w/ units does not behave as expected"
    # SpiralArmsPotential
    pot = potential.SpiralArmsPotential(
        amp=1,
        ro=ro,
        vo=vo,
        N=2,
        alpha=13 * units.deg,
        r_ref=0.8 * units.kpc,
        phi_ref=90.0 * units.deg,
        Rs=8 * units.kpc,
        H=0.1 * units.kpc,
        omega=20.0 * units.km / units.s / units.kpc,
        Cs=[1],
    )
    pot_nounits = potential.SpiralArmsPotential(
        amp=1,
        ro=ro,
        vo=vo,
        N=2,
        alpha=13 * numpy.pi / 180.0,
        r_ref=0.8 / ro,
        phi_ref=numpy.pi / 2,
        Rs=8.0 / ro,
        H=0.1 / ro,
        omega=20.0 / conversion.freq_in_kmskpc(vo, ro),
        Cs=[1],
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "SpiralArmsPotential w/ parameters w/ units does not behave as expected"
    # DehnenSmoothWrapperPotential
    dpn = potential.DehnenBarPotential(tform=-100.0, tsteady=1.0)
    pot = potential.DehnenSmoothWrapperPotential(
        pot=dpn, tform=-1.0 * units.Gyr, tsteady=3.0 * units.Gyr, ro=ro, vo=vo
    )
    pot_nounits = potential.DehnenSmoothWrapperPotential(
        pot=dpn,
        tform=-1.0 / conversion.time_in_Gyr(vo, ro),
        tsteady=3.0 / conversion.time_in_Gyr(vo, ro),
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "DehnenSmoothWrapperPotential w/ parameters w/ units does not behave as expected"
    # SolidBodyRotationWrapperPotential
    spn = potential.SpiralArmsPotential(omega=0.0, phi_ref=0.0)
    pot = potential.SolidBodyRotationWrapperPotential(
        pot=spn,
        omega=20.0 * units.km / units.s / units.kpc,
        pa=30.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SolidBodyRotationWrapperPotential(
        pot=spn,
        omega=20.0 / conversion.freq_in_kmskpc(vo, ro),
        pa=30.0 / 180.0 * numpy.pi,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "SolidBodyRotationWrapperPotential w/ parameters w/ units does not behave as expected"
    # CorotatingRotationWrapperPotential
    spn = potential.SpiralArmsPotential(omega=0.0, phi_ref=0.0)
    pot = potential.CorotatingRotationWrapperPotential(
        pot=spn,
        vpo=200.0 * units.km / units.s,
        to=1.0 * units.Gyr,
        pa=30.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.CorotatingRotationWrapperPotential(
        pot=spn,
        vpo=200.0 / vo,
        to=1.0 / conversion.time_in_Gyr(vo, ro),
        pa=30.0 / 180.0 * numpy.pi,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "CorotatingRotationWrapperPotential w/ parameters w/ units does not behave as expected"
    # GaussianAmplitudeWrapperPotential
    dpn = potential.DehnenBarPotential(tform=-100.0, tsteady=1.0)
    pot = potential.GaussianAmplitudeWrapperPotential(
        pot=dpn, to=-1.0 * units.Gyr, sigma=10.0 * units.Gyr, ro=ro, vo=vo
    )
    pot_nounits = potential.GaussianAmplitudeWrapperPotential(
        pot=dpn,
        to=-1.0 / conversion.time_in_Gyr(vo, ro),
        sigma=10.0 / conversion.time_in_Gyr(vo, ro),
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.5, 0.3, phi=0.1, use_physical=False)
            - pot_nounits(1.5, 0.3, phi=0.1, use_physical=False)
        )
        < 10.0**-8.0
    ), "GaussianAmplitudeWrapperPotential w/ parameters w/ units does not behave as expected"
    # ChandrasekharDynamicalFrictionForce
    pot = potential.ChandrasekharDynamicalFrictionForce(
        GMs=10.0**9.0 * units.Msun,
        rhm=1.2 * units.kpc,
        minr=1.0 * units.pc,
        maxr=100.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.ChandrasekharDynamicalFrictionForce(
        GMs=10.0**9.0 / conversion.mass_in_msol(vo, ro),
        rhm=1.2 / ro,
        minr=1.0 / ro / 1000.0,
        maxr=100.0 / ro,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - pot_nounits.Rforce(
                1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False
            )
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # Also check that this works after changing GMs and rhm on the fly (specific to ChandrasekharDynamicalFrictionForce
    old_force = pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
    pot.GMs = 10.0**8.0 * units.Msun
    pot_nounits.GMs = 10.0**8.0 / conversion.mass_in_msol(vo, ro)
    # units should still work
    assert (
        numpy.fabs(
            pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - pot_nounits.Rforce(
                1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False
            )
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # and now for GMs
    pot.GMs = 10.0**8.0 * units.Msun * constants.G
    pot_nounits.GMs = 10.0**8.0 / conversion.mass_in_msol(vo, ro)
    # units should still work
    assert (
        numpy.fabs(
            pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - pot_nounits.Rforce(
                1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False
            )
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # Quick test that other units don't work
    with pytest.raises(units.UnitConversionError) as excinfo:
        pot.GMs = 10.0**8.0 * units.Msun / units.pc**2
    # and force should be /10 of previous (because linear in mass
    assert (
        numpy.fabs(
            pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - old_force / 10.0
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # Now do rhm
    pot.rhm = 12 * units.kpc
    pot_nounits.rhm = 12 / ro
    assert (
        numpy.fabs(
            pot.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - pot_nounits.Rforce(
                1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False
            )
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # Compare changed rhm to one that has rhm directly set to this value
    # to make sure that the change is okay
    pot_nounits_direct = potential.ChandrasekharDynamicalFrictionForce(
        GMs=10.0**8.0 / conversion.mass_in_msol(vo, ro),
        rhm=12 / ro,
        minr=1.0 / ro / 1000.0,
        maxr=100.0 / ro,
    )
    assert (
        numpy.fabs(
            pot_nounits.Rforce(1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False)
            - pot_nounits_direct.Rforce(
                1.5, 0.3, phi=0.1, v=[1.0, 0.0, 0.0], use_physical=False
            )
        )
        < 10.0**-8.0
    ), "ChandrasekharDynamicalFrictionForce w/ parameters w/ units does not behave as expected"
    # SphericalShellPotential
    pot = potential.SphericalShellPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=5.0 * units.kpc, ro=ro, vo=vo
    )
    pot_nounits = potential.SphericalShellPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=5.0 / ro, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "SphericalShellPotential w/ amp w/ units does not behave as expected"
    # RingPotential
    pot = potential.RingPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=5.0 * units.kpc, ro=ro, vo=vo
    )
    pot_nounits = potential.RingPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=5.0 / ro, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RingPotential w/ amp w/ units does not behave as expected"
    # If you add one here, don't base it on ChandrasekharDynamicalFrictionForce!!
    # PerfectEllipsoidPotential
    pot = potential.PerfectEllipsoidPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        a=5.0 * units.kpc,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    pot_nounits = potential.PerfectEllipsoidPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, a=5.0 / ro, ro=ro, vo=vo, b=1.3, c=0.4
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "PerfectEllipsoidPotential w/ amp w/ units does not behave as expected"
    # If you add one here, don't base it on ChandrasekharDynamicalFrictionForce!!
    # HomogeneousSpherePotential
    pot = potential.HomogeneousSpherePotential(
        amp=0.1 * units.Msun / units.pc**3, R=10.0 * units.kpc, ro=ro, vo=vo
    )
    pot_nounits = potential.HomogeneousSpherePotential(
        amp=0.1 * units.Msun / units.pc**3, R=10.0 / ro, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(1.1, 0.2, phi=1.0, use_physical=False)
            - pot_nounits(1.1, 0.2, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "HomogeneousSpherePotential w/ amp w/ units does not behave as expected"
    # TriaxialGaussianPotential
    pot = potential.TriaxialGaussianPotential(
        amp=4.0 * 10.0**10.0 * units.Msun,
        sigma=5.0 * units.kpc,
        ro=ro,
        vo=vo,
        b=1.3,
        c=0.4,
    )
    pot_nounits = potential.TriaxialGaussianPotential(
        amp=4.0 * 10.0**10.0 * units.Msun, sigma=5.0 / ro, ro=ro, vo=vo, b=1.3, c=0.4
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "TriaxialGaussianPotential w/ amp w/ units does not behave as expected"
    # If you add one here, don't base it on ChandrasekharDynamicalFrictionForce!!
    # KingPotential
    pot = potential.KingPotential(
        W0=3.0, M=4.0 * 10.0**6.0 * units.Msun, rt=10.0 * units.pc, ro=ro, vo=vo
    )
    pot_nounits = potential.KingPotential(
        W0=3.0,
        M=4.0 * 10.0**6.0 / conversion.mass_in_msol(vo, ro),
        rt=10.0 / 1000 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "KingPotential w/ amp w/ units does not behave as expected"
    # AnyAxisymmetricRazorThinDiskPotential
    pot = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5
        * conversion.surfdens_in_msolpc2(vo, ro)
        * units.Msun
        / units.pc**2
        * numpy.exp(-R),
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5 * numpy.exp(-R), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "AnyAxisymmetricRazorThinDiskPotential w/ parameters w/ units does not behave as expected"
    # AnyAxisymmetricRazorThinDiskPotential, r in surfdens also has units
    pot = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5
        * conversion.surfdens_in_msolpc2(vo, ro)
        * units.Msun
        / units.pc**2
        * numpy.exp(-R / ro / units.kpc),
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5 * numpy.exp(-R), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "AnyAxisymmetricRazorThinDiskPotential w/ parameters w/ units does not behave as expected"
    # AnyAxisymmetricRazorThinDiskPotential, r in surfdens only has units
    pot = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5 * numpy.exp(-R / ro / units.kpc), ro=ro, vo=vo
    )
    pot_nounits = potential.AnyAxisymmetricRazorThinDiskPotential(
        surfdens=lambda R: 1.5 * numpy.exp(-R), ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "AnyAxisymmetricRazorThinDiskPotential w/ parameters w/ units does not behave as expected"
    # AnySphericalPotential
    pot = potential.AnySphericalPotential(
        dens=lambda r: 0.64
        / r
        / (1 + r) ** 3
        * conversion.dens_in_msolpc3(vo, ro)
        * units.Msun
        / units.pc**3,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.AnySphericalPotential(
        dens=lambda r: 0.64 / r / (1 + r) ** 3, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "potential.AnySphericalPotential w/ parameters w/ units does not behave as expected"
    # AnySphericalPotential, r in dens also has units
    pot = potential.AnySphericalPotential(
        dens=lambda r: 0.64
        / (r / ro / units.kpc)
        / (1 + r / ro / units.kpc) ** 3
        * conversion.dens_in_msolpc3(vo, ro)
        * units.Msun
        / units.pc**3,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.AnySphericalPotential(
        dens=lambda r: 0.64 / r / (1 + r) ** 3, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "AnyAxisymmetricRazorThinDiskPotential w/ parameters w/ units does not behave as expected"
    # AnySphericalPotential, r in dens only has units
    pot = potential.AnySphericalPotential(
        dens=lambda r: 0.64 / (r / ro / units.kpc) / (1 + r / ro / units.kpc) ** 3,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.AnySphericalPotential(
        dens=lambda r: 0.64 / r / (1 + r) ** 3, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "AnyAxisymmetricRazorThinDiskPotential w/ parameters w/ units does not behave as expected"
    # If you add one here, don't base it on ChandrasekharDynamicalFrictionForce!!
    # RotateAndTiltWrapperPotential, zvec, pa
    wrappot = potential.TriaxialNFWPotential(amp=1.0, a=3.0, b=0.7, c=0.5)
    pot = potential.RotateAndTiltWrapperPotential(
        pot=wrappot, zvec=[0, 1.0, 0], galaxy_pa=30.0 * units.deg, ro=ro, vo=vo
    )
    pot_nounits = potential.RotateAndTiltWrapperPotential(
        pot=wrappot, zvec=[0, 1.0, 0], galaxy_pa=numpy.pi / 6.0, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RotateAndTiltWrapperPotential w/ pa w/ units does not behave as expected"
    # RotateAndTiltWrapperPotential, inclination, galaxy_pa, sky_pa
    wrappot = potential.TriaxialNFWPotential(amp=1.0, a=3.0, b=0.7, c=0.5)
    pot = potential.RotateAndTiltWrapperPotential(
        pot=wrappot,
        galaxy_pa=30.0 * units.deg,
        inclination=60.0 * units.deg,
        sky_pa=-45.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.RotateAndTiltWrapperPotential(
        pot=wrappot,
        galaxy_pa=numpy.pi / 6.0,
        inclination=numpy.pi / 3.0,
        sky_pa=-numpy.pi / 4.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(4.0, 0.0, phi=1.0, use_physical=False)
            - pot_nounits(4.0, 0.0, phi=1.0, use_physical=False)
        )
        < 10.0**-8.0
    ), "RotateAndTiltWrapperPotential w/ pa w/ units does not behave as expected"
    # If you add one here, don't base it on ChandrasekharDynamicalFrictionForce!!
    return None


def test_potential_paramunits_2d():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import conversion

    ro, vo = 11.0, 180.0
    # CosmphiDiskPotential
    pot = potential.CosmphiDiskPotential(
        amp=1.0,
        m=3,
        phib=20.0 * units.deg,
        phio=1290.0 * units.km**2 / units.s**2,
        r1=8.0 * units.kpc,
        rb=7.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.CosmphiDiskPotential(
        amp=1.0,
        m=3,
        phib=20.0 / 180.0 * numpy.pi,
        phio=1290.0 / vo**2.0,
        r1=8.0 / ro,
        rb=7.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "CosmphiDiskPotential w/ parameters w/ units does not behave as expected"
    # CosmphiDiskPotential, alternative setup
    pot = potential.CosmphiDiskPotential(
        amp=1.0,
        m=3,
        cp=1000.0 * units.km**2 / units.s**2.0,
        sp=300.0 * units.km**2 / units.s**2.0,
        r1=8.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.CosmphiDiskPotential(
        amp=1.0,
        m=3,
        cp=1000.0 / vo**2.0,
        sp=300.0 / vo**2.0,
        r1=8.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "CosmphiDiskPotential w/ parameters w/ units does not behave as expected"
    # EllipticalDiskPotential
    pot = potential.EllipticalDiskPotential(
        amp=1.0,
        tform=1.0 * units.Gyr,
        tsteady=3.0 * units.Gyr,
        phib=20.0 * units.deg,
        twophio=1290.0 * units.km**2 / units.s**2,
        r1=8.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.EllipticalDiskPotential(
        amp=1.0,
        tform=1.0 / conversion.time_in_Gyr(vo, ro),
        tsteady=3.0 / conversion.time_in_Gyr(vo, ro),
        phib=20.0 / 180.0 * numpy.pi,
        twophio=1290.0 / vo**2.0,
        r1=8.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "EllipticalDiskPotential w/ parameters w/ units does not behave as expected"
    # EllipticalDiskPotential, alternative setup
    pot = potential.EllipticalDiskPotential(
        amp=1.0,
        tform=1.0 * units.Gyr,
        tsteady=3.0 * units.Gyr,
        cp=1000.0 * units.km**2 / units.s**2.0,
        sp=300.0 * units.km**2 / units.s**2.0,
        r1=8.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.EllipticalDiskPotential(
        amp=1.0,
        tform=1.0 / conversion.time_in_Gyr(vo, ro),
        tsteady=3.0 / conversion.time_in_Gyr(vo, ro),
        cp=1000.0 / vo**2.0,
        sp=300.0 / vo**2.0,
        r1=8.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "EllipticalDiskPotential w/ parameters w/ units does not behave as expected"
    # LopsidedDiskPotential
    pot = potential.LopsidedDiskPotential(
        amp=1.0,
        phib=20.0 * units.deg,
        phio=1290.0 * units.km**2 / units.s**2,
        r1=8.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.LopsidedDiskPotential(
        amp=1.0,
        phib=20.0 / 180.0 * numpy.pi,
        phio=1290.0 / vo**2.0,
        r1=8.0 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"
    # LopsidedDiskPotential, alternative setup
    pot = potential.LopsidedDiskPotential(
        amp=1.0,
        cp=1000.0 * units.km**2 / units.s**2.0,
        sp=300.0 * units.km**2 / units.s**2.0,
        r1=8.0 * units.kpc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.LopsidedDiskPotential(
        amp=1.0, cp=1000.0 / vo**2.0, sp=300.0 / vo**2.0, r1=8.0 / ro, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "LopsidedDiskPotential w/ parameters w/ units does not behave as expected"
    # SteadyLogSpiralPotential
    pot = potential.SteadyLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * units.km / units.s / units.kpc,
        A=1700.0 * units.km**2 / units.s**2,
        gamma=21.0 * units.deg,
        alpha=-9.0,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SteadyLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * ro / vo,
        A=1700.0 / vo**2.0,
        gamma=21.0 / 180.0 * numpy.pi,
        alpha=-9.0,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "SteadyLogSpiralPotential w/ parameters w/ units does not behave as expected"
    # SteadyLogSpiralPotential, alternative setup
    pot = potential.SteadyLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * units.km / units.s / units.kpc,
        A=1700.0 * units.km**2 / units.s**2,
        gamma=21.0 * units.deg,
        p=10.0 * units.deg,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.SteadyLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * ro / vo,
        A=1700.0 / vo**2.0,
        gamma=21.0 / 180.0 * numpy.pi,
        p=10.0 / 180.0 * numpy.pi,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "SteadyLogSpiralPotential w/ parameters w/ units does not behave as expected"
    # TransientLogSpiralPotential
    pot = potential.TransientLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * units.km / units.s / units.kpc,
        A=1700.0 * units.km**2 / units.s**2,
        gamma=21.0 * units.deg,
        alpha=-9.0,
        to=2.0 * units.Gyr,
        sigma=1.0 * units.Gyr,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.TransientLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * ro / vo,
        A=1700.0 / vo**2.0,
        gamma=21.0 / 180.0 * numpy.pi,
        alpha=-9.0,
        to=2.0 / conversion.time_in_Gyr(vo, ro),
        sigma=1.0 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "TransientLogSpiralPotential w/ parameters w/ units does not behave as expected"
    # TransientLogSpiralPotential, alternative setup
    pot = potential.TransientLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * units.km / units.s / units.kpc,
        A=1700.0 * units.km**2 / units.s**2,
        gamma=21.0 * units.deg,
        p=10.0 * units.deg,
        to=2.0 * units.Gyr,
        sigma=1.0 * units.Gyr,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.TransientLogSpiralPotential(
        amp=1.0,
        m=4,
        omegas=50.0 * ro / vo,
        A=1700.0 / vo**2.0,
        gamma=21.0 / 180.0 * numpy.pi,
        p=10.0 / 180.0 * numpy.pi,
        to=2.0 / conversion.time_in_Gyr(vo, ro),
        sigma=1.0 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(
            pot(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
            - pot_nounits(
                1.5, phi=0.1, t=2.0 / conversion.time_in_Gyr(vo, ro), use_physical=False
            )
        )
        < 10.0**-8.0
    ), "TransientLogSpiralPotential w/ parameters w/ units does not behave as expected"
    return None


def test_potential_paramunits_1d():
    # Test that input units for potential parameters other than the amplitude
    # behave as expected
    from galpy import potential
    from galpy.util import conversion

    ro, vo = 10.5, 195.0
    # KGPotential
    pot = potential.KGPotential(
        amp=1.0,
        K=40.0 * units.Msun / units.pc**2,
        F=0.02 * units.Msun / units.pc**3,
        D=200 * units.pc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.KGPotential(
        amp=1.0,
        K=40.0 / conversion.surfdens_in_msolpc2(vo, ro) * 2.0 * numpy.pi,
        F=0.02 / conversion.dens_in_msolpc3(vo, ro) * 4.0 * numpy.pi,
        D=0.2 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(pot(1.5, use_physical=False) - pot_nounits(1.5, use_physical=False))
        < 10.0**-8.0
    ), "KGPotential w/ parameters w/ units does not behave as expected"
    # KGPotential, alternative setup
    pot = potential.KGPotential(
        amp=1.0,
        K=40.0 * units.Msun / units.pc**2 * constants.G,
        F=0.02 * units.Msun / units.pc**3 * constants.G,
        D=200 * units.pc,
        ro=ro,
        vo=vo,
    )
    pot_nounits = potential.KGPotential(
        amp=1.0,
        K=40.0 / conversion.surfdens_in_msolpc2(vo, ro),
        F=0.02 / conversion.dens_in_msolpc3(vo, ro),
        D=0.2 / ro,
        ro=ro,
        vo=vo,
    )
    # Check potential
    assert (
        numpy.fabs(pot(1.5, use_physical=False) - pot_nounits(1.5, use_physical=False))
        < 10.0**-8.0
    ), "KGPotential w/ parameters w/ units does not behave as expected"
    # IsothermalDiskPotential
    pot = potential.IsothermalDiskPotential(
        amp=1.2, sigma=30.0 * units.km / units.s, ro=ro, vo=vo
    )
    pot_nounits = potential.IsothermalDiskPotential(
        amp=1.2, sigma=30.0 / vo, ro=ro, vo=vo
    )
    # Check potential
    assert (
        numpy.fabs(pot(1.5, use_physical=False) - pot_nounits(1.5, use_physical=False))
        < 10.0**-8.0
    ), "IsothermalDiskPotential w/ parameters w/ units does not behave as expected"
    return None


def test_potential_paramunits_1d_wrongunits():
    # Test that input units for potential amplitudes behave as expected
    from galpy import potential

    ro, vo = 9.0, 210.0
    # KGPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.KGPotential(
            amp=1.0,
            K=40.0 * units.Msun / units.pc**3,
            F=0.02 * units.Msun / units.pc**3,
            D=200 * units.pc,
            ro=ro,
            vo=vo,
        )
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.KGPotential(
            amp=1.0,
            K=40.0 * units.Msun / units.pc**2,
            F=0.02 * units.Msun / units.pc**2,
            D=200 * units.pc,
            ro=ro,
            vo=vo,
        )
    # IsothermalDiskPotential
    with pytest.raises(units.UnitConversionError) as excinfo:
        potential.IsothermalDiskPotential(amp=1.0, sigma=10 * units.kpc, ro=ro, vo=vo)
    return None


def test_potential_method_turnphysicalon():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    pot.turn_physical_on()
    assert isinstance(
        pot(1.1, 0.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 7.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 220.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    pot.turn_physical_on(ro=6.0, vo=210.0)
    assert isinstance(
        pot(1.1, 0.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 210.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    pot.turn_physical_on(ro=6.0 * units.kpc, vo=210.0 * units.km / units.s)
    assert isinstance(
        pot(1.1, 0.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 210.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.kpc)
    pot.turn_physical_on(ro=6.0, vo=210.0)
    assert isinstance(
        pot(1.1, phi=0.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 210.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    pot.turn_physical_on(ro=6.0 * units.kpc, vo=210.0 * units.km / units.s)
    assert isinstance(
        pot(1.1, phi=0.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 210.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.kpc)
    pot.turn_physical_on(ro=9, vo=230)
    assert isinstance(
        pot(1.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 9.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    assert (
        numpy.fabs(pot._vo - 230.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    pot.turn_physical_on(ro=9 * units.kpc, vo=230 * units.km / units.s)
    assert isinstance(
        pot(1.1), units.Quantity
    ), "Potential method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 9.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    assert (
        numpy.fabs(pot._vo - 230.0) < 10.0**-10.0
    ), "Potential method turn_physical_on does not work as expected"
    return None


def test_potential_method_turnphysicaloff():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    pot.turn_physical_off()
    assert isinstance(
        pot(1.1, 0.1), float
    ), "Potential method does not return float when turn_physical_off has been called"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.kpc)
    pot.turn_physical_off()
    assert isinstance(
        pot(1.1, phi=0.1), float
    ), "Potential method does not return float when turn_physical_off has been called"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.kpc)
    pot.turn_physical_off()
    assert isinstance(
        pot(1.1), float
    ), "Potential method does not return float when turn_physical_off has been called"
    return None


def test_potential_function_turnphysicalon():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(
        potential.evaluatePotentials(pot, 1.1, 0.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 7.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    potential.turn_physical_on([pot])
    assert isinstance(
        potential.evaluatePotentials([pot], 1.1, 0.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 7.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 220.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(
        potential.evaluateplanarPotentials(pot, 1.1, phi=0.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    potential.turn_physical_on([pot], ro=9.0, vo=230.0)
    assert isinstance(
        potential.evaluateplanarPotentials([pot], 1.1, phi=0.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 9.0) < 10.0**-10.0
    ), "Potential method does not work as expected"
    assert (
        numpy.fabs(pot._vo - 230.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.kpc)
    potential.turn_physical_on(pot)
    assert isinstance(
        potential.evaluatelinearPotentials(pot, 1.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 5.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    assert (
        numpy.fabs(pot._vo - 220.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    potential.turn_physical_on([pot], ro=6.0 * units.kpc, vo=250.0 * units.km / units.s)
    assert isinstance(
        potential.evaluatelinearPotentials([pot], 1.1), units.Quantity
    ), "Potential function does not return Quantity when function turn_physical_on has been called"
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    assert (
        numpy.fabs(pot._vo - 250.0) < 10.0**-10.0
    ), "Potential function turn_physical_on does not work as expected"
    return None


def test_potential_function_turnphysicaloff():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(
        potential.evaluatePotentials(pot, 1.1, 0.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    potential.turn_physical_off([pot])
    assert isinstance(
        potential.evaluatePotentials([pot], 1.1, 0.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(
        potential.evaluateplanarPotentials(pot, 1.1, phi=0.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    potential.turn_physical_off([pot])
    assert isinstance(
        potential.evaluateplanarPotentials([pot], 1.1, phi=0.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.kpc)
    potential.turn_physical_off(pot)
    assert isinstance(
        potential.evaluatelinearPotentials(pot, 1.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    potential.turn_physical_off([pot])
    assert isinstance(
        potential.evaluatelinearPotentials([pot], 1.1), float
    ), "Potential function does not return float when function turn_physical_off has been called"
    return None


def test_potential_setup_roAsQuantity():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.kpc)
    assert (
        numpy.fabs(pot._ro - 7.0) < 10.0**-10.0
    ), "ro in 3D potential setup as Quantity does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.kpc)
    assert (
        numpy.fabs(pot._ro - 6.0) < 10.0**-10.0
    ), "ro in 2D potential setup as Quantity does not work as expected"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.kpc)
    assert (
        numpy.fabs(pot._ro - 5.0) < 10.0**-10.0
    ), "ro in 1D potential setup as Quantity does not work as expected"
    return None


def test_potential_setup_roAsQuantity_oddunits():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(ro=7.0 * units.lyr)
    assert (
        numpy.fabs(pot._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in 3D potential setup as Quantity does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(ro=6.0 * units.lyr)
    assert (
        numpy.fabs(pot._ro - 6.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in 2D potential setup as Quantity does not work as expected"
    # 1D
    pot = potential.KGPotential(ro=5.0 * units.lyr)
    assert (
        numpy.fabs(pot._ro - 5.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in 1D potential setup as Quantity does not work as expected"
    return None


def test_potential_setup_voAsQuantity():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(vo=210.0 * units.km / units.s)
    assert (
        numpy.fabs(pot._vo - 210.0) < 10.0**-10.0
    ), "vo in 3D potential setup as Quantity does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(vo=230.0 * units.km / units.s)
    assert (
        numpy.fabs(pot._vo - 230.0) < 10.0**-10.0
    ), "vo in 2D potential setup as Quantity does not work as expected"
    # 1D
    pot = potential.KGPotential(vo=250.0 * units.km / units.s)
    assert (
        numpy.fabs(pot._vo - 250.0) < 10.0**-10.0
    ), "vo in 1D potential setup as Quantity does not work as expected"
    return None


def test_potential_setup_voAsQuantity_oddunits():
    from galpy import potential

    # 3D
    pot = potential.BurkertPotential(vo=210.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(pot._vo - 210.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in 3D potential setup as Quantity does not work as expected"
    # 2D
    pot = potential.EllipticalDiskPotential(vo=230.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(pot._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in 2D potential setup as Quantity does not work as expected"
    # 1D
    pot = potential.KGPotential(vo=250.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(pot._vo - 250.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in 1D potential setup as Quantity does not work as expected"
    return None


def test_interpRZPotential_ro():
    # Test that ro is correctly propagated to interpRZPotential
    from galpy.potential import BurkertPotential, interpRZPotential

    ro = 9.0
    # ro on, single pot
    bp = BurkertPotential(ro=ro)
    ip = interpRZPotential(bp)
    assert (
        numpy.fabs(ip._ro - bp._ro) < 10.0**-10.0
    ), "ro not correctly propagated to interpRZPotential"
    assert ip._roSet, "roSet not correctly propagated to interpRZPotential"
    # ro on, list pot
    ip = interpRZPotential([bp])
    assert (
        numpy.fabs(ip._ro - bp._ro) < 10.0**-10.0
    ), "ro not correctly propagated to interpRZPotential"
    assert ip._roSet, "roSet not correctly propagated to interpRZPotential"
    # ro off, single pot
    bp = BurkertPotential()
    ip = interpRZPotential(bp)
    assert (
        numpy.fabs(ip._ro - bp._ro) < 10.0**-10.0
    ), "ro not correctly propagated to interpRZPotential"
    assert not ip._roSet, "roSet not correctly propagated to interpRZPotential"
    # ro off, list pot
    bp = BurkertPotential()
    ip = interpRZPotential([bp])
    assert (
        numpy.fabs(ip._ro - bp._ro) < 10.0**-10.0
    ), "ro not correctly propagated to interpRZPotential"
    assert not ip._roSet, "roSet not correctly propagated to interpRZPotential"
    return None


def test_interpRZPotential_vo():
    # Test that vo is correctly propagated to interpRZPotential
    from galpy.potential import BurkertPotential, interpRZPotential

    vo = 200.0
    # vo on, single pot
    bp = BurkertPotential(vo=vo)
    ip = interpRZPotential(bp)
    assert (
        numpy.fabs(ip._vo - bp._vo) < 10.0**-10.0
    ), "vo not correctly propagated to interpRZPotential"
    assert ip._voSet, "voSet not correctly propagated to interpRZPotential"
    # vo on, list pot
    ip = interpRZPotential([bp])
    assert (
        numpy.fabs(ip._vo - bp._vo) < 10.0**-10.0
    ), "vo not correctly propagated to interpRZPotential"
    assert ip._voSet, "voSet not correctly propagated to interpRZPotential"
    # vo off, single pot
    bp = BurkertPotential()
    ip = interpRZPotential(bp)
    assert (
        numpy.fabs(ip._vo - bp._vo) < 10.0**-10.0
    ), "vo not correctly propagated to interpRZPotential"
    assert not ip._voSet, "voSet not correctly propagated to interpRZPotential"
    # vo off, list pot
    bp = BurkertPotential()
    ip = interpRZPotential([bp])
    assert (
        numpy.fabs(ip._vo - bp._vo) < 10.0**-10.0
    ), "vo not correctly propagated to interpRZPotential"
    assert not ip._voSet, "voSet not correctly propagated to interpRZPotential"
    return None


def test_SCFPotential_from_density():
    from galpy import potential

    a = 5.0 * units.kpc
    hp = potential.HernquistPotential(amp=2 * 1e11 * units.Msun, a=a)
    # Spherical
    sp = potential.SCFPotential.from_density(
        lambda r, **kw: hp.dens(r, 0.0, **kw), 10, a=a, symmetry="spherical"
    )
    rs = numpy.geomspace(1.0, 100.0, 101) * units.kpc
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, rs, use_physical=False) / hp.dens(rs, rs, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree when initialized with density with units"
    assert numpy.all(
        numpy.fabs(1.0 - sp.dens(rs, rs) / hp.dens(rs, rs)) < 1e-10
    ), "SCF density does not agree when initialized with density with units"
    # Output density should have units of density, can just test for Quantity, other tests ensure that this is a density
    assert isinstance(
        sp.dens(1.0, 0.1), units.Quantity
    ), "SCF density does not return Quantity when initialized with density with units"
    # Axisymmetry, use non-physical input
    sp = potential.SCFPotential.from_density(
        lambda R, z: hp.dens(R, z, use_physical=False), 10, L=3, a=a, symmetry="axisym"
    )
    rs = numpy.geomspace(1.0, 100.0, 101) * units.kpc
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, rs, use_physical=False) / hp.dens(rs, rs, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree when initialized with density with units"
    # Output density should not have units of density
    assert not isinstance(
        sp.dens(1.0, 0.1), units.Quantity
    ), "SCF density does not return Quantity when initialized with density with units"
    # General
    sp = potential.SCFPotential.from_density(
        lambda R, z, phi, **kw: hp.dens(R, z, phi=phi, **kw), 10, L=3, a=a
    )
    rs = numpy.geomspace(1.0, 100.0, 101) * units.kpc
    assert numpy.all(
        numpy.fabs(
            1.0
            - sp.dens(rs, rs, use_physical=False) / hp.dens(rs, rs, use_physical=False)
        )
        < 1e-10
    ), "SCF density does not agree when initialized with density with units"
    assert numpy.all(
        numpy.fabs(1.0 - sp.dens(rs, rs) / hp.dens(rs, rs)) < 1e-10
    ), "SCF density does not agree when initialized with density with units"
    # Output density should have units of density, can just test for Quantity, other tests ensure that this is a density
    assert isinstance(
        sp.dens(1.0, 0.1), units.Quantity
    ), "SCF density does not return Quantity when initialized with density with units"
    return None


def test_actionAngle_method_returntype():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=8.0, vo=220.0
    )
    assert isinstance(
        aA(-0.2, 0.1), units.Quantity
    ), "actionAngleHarmonic method __call__ does not return Quantity when it should"
    for ii in range(2):
        assert isinstance(
            aA.actionsFreqs(-0.2, 0.1)[ii], units.Quantity
        ), "actionAngleHarmonic method actionsFreqs does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.actionsFreqsAngles(-0.2, 0.1)[ii], units.Quantity
        ), "actionAngleHarmonic method actionsFreqsAngles does not return Quantity when it should"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=8.0, vo=220.0)
    for ii in range(3):
        assert isinstance(
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method __call__ does not return Quantity when it should"
    for ii in range(6):
        assert isinstance(
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqs does not return Quantity when it should"
    for ii in range(9):
        assert isinstance(
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method EccZmaxRperiRap does not return Quantity when it should"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    for ii in range(3):
        assert isinstance(
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method __call__ does not return Quantity when it should"
    for ii in range(6):
        assert isinstance(
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqs does not return Quantity when it should"
    for ii in range(9):
        assert isinstance(
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method EccZmaxRperiRap does not return Quantity when it should"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=8.0, vo=220.0)
    for ii in range(3):
        assert isinstance(
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method __call__ does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method EccZmaxRperiRap does not return Quantity when it should"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=8.0, vo=220.0)
    for ii in range(3):
        assert isinstance(
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method __call__ does not return Quantity when it should"
    for ii in range(6):
        assert isinstance(
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqs does not return Quantity when it should"
    for ii in range(9):
        assert isinstance(
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method EccZmaxRperiRap does not return Quantity when it should"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=8.0, vo=220.0)
    for ii in range(3):
        assert isinstance(
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method __call__ does not return Quantity when it should"
    for ii in range(6):
        assert isinstance(
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqs does not return Quantity when it should"
    for ii in range(9):
        assert isinstance(
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochrone method actionsFreqsAngles does not return Quantity when it should"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=8.0, vo=220.0
    )
    for ii in range(2):
        assert isinstance(
            aA(0.1, -0.2)[ii], units.Quantity
        ), "actionAngleHarmonicInverse method __call__ does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.xvFreqs(0.1, -0.2)[ii], units.Quantity
        ), "actionAngleHarmonicInverse method xvFreqs does not return Quantity when it should"
    assert isinstance(
        aA.Freqs(0.1), units.Quantity
    ), "actionAngleIsochroneInverse method Freqs does not return Quantity when it should"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, ro=8.0, vo=220.0)
    for ii in range(6):
        assert isinstance(
            aA(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochroneInverse method __call__ does not return Quantity when it should"
    for ii in range(9):
        assert isinstance(
            aA.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii], units.Quantity
        ), "actionAngleIsochroneInverse method xvFreqs does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            aA.Freqs(0.1, 1.1, 0.1)[ii], units.Quantity
        ), "actionAngleIsochroneInverse method Freqs does not return Quantity when it should"
    return None


def test_actionAngle_method_returnunit():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=8.0, vo=220.0
    )
    try:
        aA(-0.2, 0.1).to(units.kpc * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function __call__ does not return Quantity with the right units"
        )
    try:
        aA.actionsFreqs(-0.2, 0.1)[0].to(units.kpc * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function actionsFreqs does not return Quantity with the right units"
        )
    try:
        aA.actionsFreqs(-0.2, 0.1)[1].to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function actionsFreqs does not return Quantity with the right units"
        )
    try:
        aA.actionsFreqsAngles(-0.2, 0.1)[0].to(units.kpc * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
        )
    try:
        aA.actionsFreqsAngles(-0.2, 0.1)[1].to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
        )
    try:
        aA.actionsFreqsAngles(-0.2, 0.1)[2].to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
        )
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=8.0, vo=220.0)
    for ii in range(3):
        try:
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc * units.km / units.s)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function __call__ does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(6, 9):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    try:
        aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0].to(
            units.dimensionless_unscaled
        )
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
        )
    for ii in range(1, 4):
        try:
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
            )
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    for ii in range(3):
        try:
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc * units.km / units.s)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function __call__ does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(6, 9):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    try:
        aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0].to(
            units.dimensionless_unscaled
        )
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
        )
    for ii in range(1, 4):
        try:
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
            )
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=8.0, vo=220.0)
    for ii in range(3):
        try:
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc * units.km / units.s)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function __call__ does not return Quantity with the right units"
            )
    try:
        aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0].to(
            units.dimensionless_unscaled
        )
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
        )
    for ii in range(1, 4):
        try:
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
            )
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=8.0, vo=220.0)
    for ii in range(3):
        try:
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc * units.km / units.s)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function __call__ does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(6, 9):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    try:
        aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0].to(
            units.dimensionless_unscaled
        )
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
        )
    for ii in range(1, 4):
        try:
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function EccZmaxRperiRap does not return Quantity with the right units"
            )
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=8.0, vo=220.0)
    for ii in range(3):
        try:
            aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc * units.km / units.s)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function __call__ does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(
                units.kpc * units.km / units.s
            )
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(3, 6):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    for ii in range(6, 9):
        try:
            aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.rad)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngle function actionsFreqsAngles does not return Quantity with the right units"
            )
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=8.0, vo=220.0
    )
    correct_unit = [units.m, units.m / units.s]
    for ii in range(2):
        try:
            aA(0.1, -0.2)[ii].to(correct_unit[ii])
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngleInverse function __call__ does not return Quantity with the right units"
            )
    correct_unit = [units.m, units.m / units.s, 1 / units.Gyr]
    for ii in range(3):
        try:
            aA.xvFreqs(0.1, -0.2)[ii].to(correct_unit[ii])
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngleInverse function actionsFreqs does not return Quantity with the right units"
            )
    try:
        aA.Freqs(0.1).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "actionAngleInverse function Freqs does not return Quantity with the right units"
        )
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, ro=8.0, vo=220.0)
    correct_unit = [
        units.m,
        units.m / units.s,
        units.m / units.s,
        units.m,
        units.m / units.s,
        units.deg,
    ]
    for ii in range(6):
        try:
            aA(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii].to(correct_unit[ii])
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngleInverse function __call__ does not return Quantity with the right units"
            )
    correct_unit = [
        units.m,
        units.m / units.s,
        units.m / units.s,
        units.m,
        units.m / units.s,
        units.deg,
        1 / units.Gyr,
        1 / units.Gyr,
        1 / units.Gyr,
    ]
    for ii in range(9):
        try:
            aA.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii].to(correct_unit[ii])
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngleInverse function actionsFreqs does not return Quantity with the right units"
            )
    for ii in range(3):
        try:
            aA.Freqs(0.1, 1.1, 0.1)[ii].to(1 / units.Gyr)
        except units.UnitConversionError:
            raise AssertionError(
                "actionAngleInverse function Freqs does not return Quantity with the right units"
            )
    return None


def test_actionAngle_method_value():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential
    from galpy.util import conversion

    ro, vo = 9.0, 230.0
    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=ro, vo=vo
    )
    aAnu = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    assert (
        numpy.fabs(
            aA(-0.2, 0.1).to(units.kpc * units.km / units.s).value
            - aAnu(-0.2, 0.1) * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function __call__ does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.actionsFreqs(-0.2, 0.1)[0].to(units.kpc * units.km / units.s).value
            - aAnu.actionsFreqs(-0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.actionsFreqs(-0.2, 0.1)[1].to(1 / units.Gyr).value
            - aAnu.actionsFreqs(-0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(-0.2, 0.1)[0].to(units.kpc * units.km / units.s).value
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[0] * ro * vo
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(-0.2, 0.1)[1].to(1 / units.Gyr).value
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[1] * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(-0.2, 0.1)[2].to(units.rad).value
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=ro, vo=vo)
    aAnu = actionAngleIsochrone(b=0.8)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.rad)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
            .to(units.dimensionless_unscaled)
            .value
            - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
        )
        < 10.0**-8.0
    ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    for ii in range(1, 4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc).value
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro
            )
            < 10.0**-8.0
        ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=ro, vo=vo)
    aAnu = actionAngleSpherical(pot=pot)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0, ro=9.0 * units.kpc)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * 9.0 * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1, 0.1, 1.1, 0.1, 0.2, 0.0, vo=230.0 * units.km / units.s
                )[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * 230.0
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.rad)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
            .to(units.dimensionless_unscaled)
            .value
            - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
        )
        < 10.0**-8.0
    ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    for ii in range(1, 4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc).value
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro
            )
            < 10.0**-8.0
        ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=ro, vo=vo)
    aAnu = actionAngleAdiabatic(pot=MWPotential)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
            .to(units.dimensionless_unscaled)
            .value
            - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
        )
        < 10.0**-8.0
    ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    for ii in range(1, 4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc).value
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro
            )
            < 10.0**-8.0
        ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=ro, vo=vo)
    aAnu = actionAngleStaeckel(pot=MWPotential, delta=0.45)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.rad)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
            .to(units.dimensionless_unscaled)
            .value
            - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0]
        )
        < 10.0**-8.0
    ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    for ii in range(1, 4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii].to(units.kpc).value
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro
            )
            < 10.0**-8.0
        ), "actionAngle function EccZmaxRperiRap does not return Quantity with the right value"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=ro, vo=vo)
    aAnu = actionAngleIsochroneApprox(pot=MWPotential, b=0.8)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function __call__ does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.kpc * units.km / units.s)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii] * ro * vo
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(1 / units.Gyr)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
                .to(units.rad)
                .value
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle function actionsFreqsAngles does not return Quantity with the right value"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    # Omega = sqrt(4piG density / 3)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0),
        ro=ro * units.kpc,
        vo=vo * units.km / units.s,
    )
    aAnu = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    correct_unit = [units.kpc, units.km / units.s]
    correct_fac = [ro, vo]
    for ii in range(2):
        assert (
            numpy.fabs(
                aA(0.1, -0.2, ro=ro * units.kpc, vo=vo * units.km / units.s)[ii]
                .to(correct_unit[ii])
                .value
                - aAnu(0.1, -0.2)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse function __call__ does not return Quantity with the right value"
    correct_unit = [units.kpc, units.km / units.s, 1 / units.Gyr]
    correct_fac = [ro, vo, conversion.freq_in_Gyr(vo, ro)]
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.xvFreqs(0.1, -0.2)[ii].to(correct_unit[ii]).value
                - aAnu.xvFreqs(0.1, -0.2)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse function xvFreqs does not return Quantity with the right value"
    assert (
        numpy.fabs(
            aA.Freqs(0.1).to(1 / units.Gyr).value
            - aAnu.Freqs(0.1) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "actionAngleInverse function Freqs does not return Quantity with the right value"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(
        b=0.8, ro=ro * units.kpc, vo=vo * units.km / units.s
    )
    aAnu = actionAngleIsochroneInverse(b=0.8)
    correct_unit = [
        units.kpc,
        units.km / units.s,
        units.km / units.s,
        units.kpc,
        units.km / units.s,
        units.rad,
    ]
    correct_fac = [ro, vo, vo, ro, vo, 1.0]
    for ii in range(6):
        assert (
            numpy.fabs(
                aA(
                    0.1,
                    1.1,
                    0.1,
                    0.1,
                    0.2,
                    0.0,
                    ro=ro * units.kpc,
                    vo=vo * units.km / units.s,
                )[ii]
                .to(correct_unit[ii])
                .value
                - aAnu(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse function __call__ does not return Quantity with the right value"
    correct_unit = [
        units.kpc,
        units.km / units.s,
        units.km / units.s,
        units.kpc,
        units.km / units.s,
        units.rad,
        1 / units.Gyr,
        1 / units.Gyr,
        1 / units.Gyr,
    ]
    correct_fac = [
        ro,
        vo,
        vo,
        ro,
        vo,
        1.0,
        conversion.freq_in_Gyr(vo, ro),
        conversion.freq_in_Gyr(vo, ro),
        conversion.freq_in_Gyr(vo, ro),
    ]
    for ii in range(9):
        assert (
            numpy.fabs(
                aA.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii].to(correct_unit[ii]).value
                - aAnu.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii] * correct_fac[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse function xvFreqs does not return Quantity with the right value"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.Freqs(0.1, 1.1, 0.1)[ii].to(1 / units.Gyr).value
                - aAnu.Freqs(0.1, 1.1, 0.1)[ii] * conversion.freq_in_Gyr(vo, ro)
            )
            < 10.0**-8.0
        ), "actionAngleInverse function Freqs does not return Quantity with the right value"
    return None


def test_actionAngle_setup_roAsQuantity():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonicc
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=7.0 * units.kpc
    )
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=7.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=7.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=9.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 9.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=7.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=7.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=7.0 * units.kpc
    )
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, ro=7.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    return None


def test_actionAngle_setup_roAsQuantity_oddunits():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=7.0 * units.lyr
    )
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=7.0 * units.lyr
    )
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, ro=7.0 * units.lyr)
    assert (
        numpy.fabs(aA._ro - 7.0 * units.lyr.to(units.kpc)) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    return None


def test_actionAngle_setup_voAsQuantity():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0),
        vo=230.0 * units.km / units.s,
    )
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, vo=230.0 * units.km / units.s)
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, vo=230.0 * units.km / units.s)
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=9.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 9.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, vo=230.0 * units.km / units.s)
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(
        pot=MWPotential, b=0.8, vo=230.0 * units.km / units.s
    )
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0),
        vo=230.0 * units.km / units.s,
    )
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, vo=230.0 * units.km / units.s)
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    return None


def test_actionAngle_setup_voAsQuantity_oddunits():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0),
        vo=230.0 * units.pc / units.Myr,
    )
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, vo=230.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, vo=230.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=9.0 * units.kpc)
    assert (
        numpy.fabs(aA._ro - 9.0) < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(
        pot=MWPotential, delta=0.45, vo=230.0 * units.pc / units.Myr
    )
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(
        pot=MWPotential, b=0.8, vo=230.0 * units.pc / units.Myr
    )
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleHarmonicInverse
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0),
        vo=230.0 * units.pc / units.Myr,
    )
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, vo=230.0 * units.pc / units.Myr)
    assert (
        numpy.fabs(aA._vo - 230.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "ro in actionAngle setup as Quantity does not work as expected"
    return None


def test_actionAngle_method_turnphysicalon():
    from galpy.actionAngle import actionAngleIsochrone

    aA = actionAngleIsochrone(b=0.8, ro=7.0 * units.kpc, vo=230.0 * units.km / units.s)
    aA.turn_physical_on()
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert isinstance(
        aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert isinstance(
        aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(aA._ro - 7.0) < 10.0**-10.0
    ), "actionAngle method does not work as expected"
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "actionAngle method turn_physical_on does not work as expected"
    aA.turn_physical_on(ro=8.0)
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(aA._ro - 8.0) < 10.0**-10.0
    ), "actionAngle method does not work as expected"
    assert (
        numpy.fabs(aA._vo - 230.0) < 10.0**-10.0
    ), "actionAngle method turn_physical_on does not work as expected"
    aA.turn_physical_on(vo=210.0)
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(aA._ro - 8.0) < 10.0**-10.0
    ), "actionAngle method does not work as expected"
    assert (
        numpy.fabs(aA._vo - 210.0) < 10.0**-10.0
    ), "actionAngle method turn_physical_on does not work as expected"
    aA.turn_physical_on(ro=9.0 * units.kpc)
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(aA._ro - 9.0) < 10.0**-10.0
    ), "actionAngle method does not work as expected"
    assert (
        numpy.fabs(aA._vo - 210.0) < 10.0**-10.0
    ), "actionAngle method turn_physical_on does not work as expected"
    aA.turn_physical_on(vo=200.0 * units.km / units.s)
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0], units.Quantity
    ), "actionAngle method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(aA._ro - 9.0) < 10.0**-10.0
    ), "actionAngle method does not work as expected"
    assert (
        numpy.fabs(aA._vo - 200.0) < 10.0**-10.0
    ), "actionAngle method turn_physical_on does not work as expected"
    return None


def test_actionAngle_method_turnphysicaloff():
    from galpy.actionAngle import actionAngleIsochrone

    aA = actionAngleIsochrone(b=0.8, ro=7.0 * units.kpc, vo=230.0 * units.km / units.s)
    aA.turn_physical_off()
    assert isinstance(
        aA(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0][0], float
    ), "actionAngle method does not return float when turn_physical_off has been called"
    assert isinstance(
        aA.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0][0], float
    ), "actionAngle method does not return float when turn_physical_off has been called"
    assert isinstance(
        aA.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[0][0], float
    ), "actionAngle method does not return float when turn_physical_off has been called"
    return None


def test_actionAngleHarmonic_setup_omega_units():
    from galpy.actionAngle import actionAngleHarmonic
    from galpy.util import conversion

    ro, vo = 9.0, 230.0
    aA = actionAngleHarmonic(omega=0.1 / units.Gyr, ro=ro, vo=vo)
    aAu = actionAngleHarmonic(omega=0.1 / conversion.freq_in_Gyr(vo, ro))
    assert (
        numpy.fabs(aA._omega - aAu._omega) < 10.0**-10.0
    ), "omega with units in actionAngleHarmonic setup does not work as expected"
    return None


def test_actionAngleHarmonicInverse_setup_omega_units():
    from galpy.actionAngle import actionAngleHarmonicInverse
    from galpy.util import conversion

    ro, vo = 9.0, 230.0
    aA = actionAngleHarmonicInverse(omega=0.1 / units.Gyr, ro=ro, vo=vo)
    aAu = actionAngleHarmonicInverse(omega=0.1 / conversion.freq_in_Gyr(vo, ro))
    assert (
        numpy.fabs(aA._omega - aAu._omega) < 10.0**-10.0
    ), "omega with units in actionAngleHarmonic setup does not work as expected"
    return None


def test_actionAngleStaeckel_setup_delta_units():
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential

    ro = 9.0
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45 * ro * units.kpc, ro=ro)
    aAu = actionAngleStaeckel(pot=MWPotential, delta=0.45)
    assert (
        numpy.fabs(aA._delta - aAu._delta) < 10.0**-10.0
    ), "delta with units in actionAngleStaeckel setup does not work as expected"
    return None


def test_actionAngleStaeckelGrid_setup_delta_units():
    from galpy.actionAngle import actionAngleStaeckelGrid
    from galpy.potential import MWPotential

    ro = 9.0
    aA = actionAngleStaeckelGrid(
        pot=MWPotential, delta=0.45 * ro * units.kpc, ro=ro, nE=5, npsi=5, nLz=5
    )
    aAu = actionAngleStaeckelGrid(pot=MWPotential, delta=0.45, nE=5, npsi=5, nLz=5)
    assert (
        numpy.fabs(aA._delta - aAu._delta) < 10.0**-10.0
    ), "delta with units in actionAngleStaeckel setup does not work as expected"
    return None


def test_actionAngleIsochrone_setup_b_units():
    from galpy.actionAngle import actionAngleIsochrone

    ro = 9.0
    aA = actionAngleIsochrone(b=0.7 * ro * units.kpc, ro=ro)
    aAu = actionAngleIsochrone(b=0.7)
    assert (
        numpy.fabs(aA.b - aAu.b) < 10.0**-10.0
    ), "b with units in actionAngleIsochrone setup does not work as expected"
    return None


def test_actionAngleIsochroneApprox_setup_b_units():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential

    ro = 9.0
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.7 * ro * units.kpc, ro=ro)
    aAu = actionAngleIsochroneApprox(pot=MWPotential, b=0.7)
    assert (
        numpy.fabs(aA._aAI.b - aAu._aAI.b) < 10.0**-10.0
    ), "b with units in actionAngleIsochroneApprox setup does not work as expected"
    return None


def test_actionAngleIsochroneInverse_setup_b_units():
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import MWPotential

    ro = 9.0
    aA = actionAngleIsochroneInverse(pot=MWPotential, b=0.7 * ro * units.kpc, ro=ro)
    aAu = actionAngleIsochroneInverse(pot=MWPotential, b=0.7)
    assert (
        numpy.fabs(aA.b - aAu.b) < 10.0**-10.0
    ), "b with units in actionAngleIsochroneInverse setup does not work as expected"
    return None


def test_actionAngleIsochroneApprix_setup_tintJ_units():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.potential import MWPotential
    from galpy.util import conversion

    ro = 9.0
    vo = 230.0
    aA = actionAngleIsochroneApprox(
        pot=MWPotential, b=0.7, tintJ=11.0 * units.Gyr, ro=ro, vo=vo
    )
    aAu = actionAngleIsochroneApprox(
        pot=MWPotential, b=0.7, tintJ=11.0 / conversion.time_in_Gyr(vo, ro)
    )
    assert (
        numpy.fabs(aA._tintJ - aAu._tintJ) < 10.0**-10.0
    ), "tintJ with units in actionAngleIsochroneApprox setup does not work as expected"
    return None


def test_actionAngle_method_inputAsQuantity():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleHarmonic,
        actionAngleHarmonicInverse,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, MWPotential, PlummerPotential

    ro, vo = 9.0, 230.0
    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=ro, vo=vo
    )
    aAnu = actionAngleHarmonic(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    assert (
        numpy.fabs(
            aA(-0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False)
            - aAnu(-0.2, 0.1)
        )
        < 10.0**-8.0
    ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            aA.actionsFreqs(
                -0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False
            )[0]
            - aAnu.actionsFreqs(-0.2, 0.1)[0]
        )
        < 10.0**-8.0
    ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            aA.actionsFreqs(
                -0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False
            )[1]
            - aAnu.actionsFreqs(-0.2, 0.1)[1]
        )
        < 10.0**-8.0
    ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(
                -0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False
            )[0]
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[0]
        )
        < 10.0**-8.0
    ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(
                -0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False
            )[1]
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[1]
        )
        < 10.0**-8.0
    ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(
            aA.actionsFreqsAngles(
                -0.2 * ro * units.kpc, 0.1 * vo * units.km / units.s, use_physical=False
            )[2]
            - aAnu.actionsFreqsAngles(-0.2, 0.1)[2]
        )
        < 10.0**-8.0
    ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    # actionAngleIsochrone
    aA = actionAngleIsochrone(b=0.8, ro=ro, vo=vo)
    aAnu = actionAngleIsochrone(b=0.8)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method EccZmaxRperiRap does not return the correct value when input is Quantity"
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=ro, vo=vo)
    aAnu = actionAngleSpherical(pot=pot)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                    ro=ro * units.kpc,
                )[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                    vo=vo * units.km / units.s,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=MWPotential, ro=ro, vo=vo)
    aAnu = actionAngleAdiabatic(pot=MWPotential)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    for ii in range(4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method EccZmaxRperiRap does not return the correct value when input is Quantity"
    # actionAngleStaeckel
    aA = actionAngleStaeckel(pot=MWPotential, delta=0.45, ro=ro, vo=vo)
    aAnu = actionAngleStaeckel(pot=MWPotential, delta=0.45)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(4):
        assert (
            numpy.fabs(
                aA.EccZmaxRperiRap(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.EccZmaxRperiRap(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method EccZmaxRperiRap does not return the correct value when input is Quantity"
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(pot=MWPotential, b=0.8, ro=ro, vo=vo)
    aAnu = actionAngleIsochroneApprox(pot=MWPotential, b=0.8)
    for ii in range(3):
        assert (
            numpy.fabs(
                aA(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method __call__ does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqs(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqs(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqs does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(3, 6):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    for ii in range(6, 9):
        assert (
            numpy.fabs(
                aA.actionsFreqsAngles(
                    1.1 * ro * units.kpc,
                    0.1 * vo * units.km / units.s,
                    1.1 * vo * units.km / units.s,
                    0.1 * ro * units.kpc,
                    0.2 * vo * units.km / units.s,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.actionsFreqsAngles(1.1, 0.1, 1.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngle method actionsFreqsAngles does not return the correct value when input is Quantity"
    # actionAngleHarmonic
    ip = IsochronePotential(normalize=5.0, b=10000.0)
    aA = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0), ro=ro, vo=vo
    )
    aAnu = actionAngleHarmonicInverse(
        omega=numpy.sqrt(4.0 * numpy.pi * ip.dens(1.2, 0.0) / 3.0)
    )
    actionsUnit = ro * vo * units.kpc * units.km / units.s
    for ii in range(2):
        assert (
            numpy.fabs(
                aA(0.1 * actionsUnit, -0.2 * units.rad, use_physical=False)[ii]
                - aAnu(0.1, -0.2)[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse method __call__ does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.xvFreqs(0.1 * actionsUnit, -0.2 * units.rad, use_physical=False)[ii]
                - aAnu.xvFreqs(0.1, -0.2)[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse method xvFreqs does not return the correct value when input is Quantity"
    assert (
        numpy.fabs(aA.Freqs(0.1 * actionsUnit, use_physical=False) - aAnu.Freqs(0.1))
        < 10.0**-8.0
    ), "actionAngleInverse method Freqs does not return the correct value when input is Quantity"
    # actionAngleIsochroneInverse
    aA = actionAngleIsochroneInverse(b=0.8, ro=ro, vo=vo)
    aAnu = actionAngleIsochroneInverse(b=0.8)
    actionsUnit = ro * vo * units.kpc * units.km / units.s
    for ii in range(6):
        assert (
            numpy.fabs(
                aA(
                    0.1 * actionsUnit,
                    1.1 * actionsUnit,
                    0.1 * actionsUnit,
                    0.1 * units.rad,
                    0.2 * units.rad,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse method __call__ does not return the correct value when input is Quantity"
    for ii in range(9):
        assert (
            numpy.fabs(
                aA.xvFreqs(
                    0.1 * actionsUnit,
                    1.1 * actionsUnit,
                    0.1 * actionsUnit,
                    0.1 * units.rad,
                    0.2 * units.rad,
                    0.0 * units.rad,
                    use_physical=False,
                )[ii]
                - aAnu.xvFreqs(0.1, 1.1, 0.1, 0.1, 0.2, 0.0)[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse method xvFreqs does not return the correct value when input is Quantity"
    for ii in range(3):
        assert (
            numpy.fabs(
                aA.Freqs(
                    0.1 * actionsUnit,
                    1.1 * actionsUnit,
                    0.1 * actionsUnit,
                    use_physical=False,
                )[ii]
                - aAnu.Freqs(0.1, 1.1, 0.1)[ii]
            )
            < 10.0**-8.0
        ), "actionAngleInverse method Freqs does not return the correct value when input is Quantity"
    return None


def test_actionAngleIsochroneApprox_method_ts_units():
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential
    from galpy.util import conversion

    ip = IsochronePotential(normalize=1.0, b=1.2)
    ro, vo = 7.5, 215.0
    aAIA = actionAngleIsochroneApprox(pot=ip, b=0.8, ro=ro, vo=vo)
    R, vR, vT, z, vz, phi = 1.1, 0.3, 1.2, 0.2, 0.5, 2.0
    # Setup an orbit, and integrated it first
    o = Orbit([R, vR, vT, z, vz, phi])
    ts = (
        numpy.linspace(0.0, 10.0, 25000) * units.Gyr
    )  # Integrate for a long time, not the default
    o.integrate(ts, ip)
    jiaO = aAIA.actionsFreqs(o, ts=ts)
    jiaOu = aAIA.actionsFreqs(o, ts=ts.value / conversion.time_in_Gyr(vo, ro))
    dOr = numpy.fabs((jiaO[3] - jiaOu[3]) / jiaO[3])
    dOp = numpy.fabs((jiaO[4] - jiaOu[4]) / jiaO[4])
    dOz = numpy.fabs((jiaO[5] - jiaOu[5]) / jiaO[5])
    assert dOr < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    assert dOp < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    assert dOz < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    # Same for actionsFreqsAngles
    jiaO = aAIA.actionsFreqsAngles(o, ts=ts)
    jiaOu = aAIA.actionsFreqsAngles(o, ts=ts.value / conversion.time_in_Gyr(vo, ro))
    dOr = numpy.fabs((jiaO[3] - jiaOu[3]) / jiaO[3])
    dOp = numpy.fabs((jiaO[4] - jiaOu[4]) / jiaO[4])
    dOz = numpy.fabs((jiaO[5] - jiaOu[5]) / jiaO[5])
    assert dOr < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    assert dOp < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    assert dOz < 10.0**-6.0, "actionAngleIsochroneApprox with ts with units fails"
    return None


def test_actionAngle_inconsistentPotentialUnits_error():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleIsochroneInverse,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.potential import IsochronePotential, PlummerPotential

    # actionAngleIsochrone
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochrone(ip=pot, ro=8.0, vo=220.0)
    pot = IsochronePotential(normalize=1.0, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochrone(ip=pot, ro=8.0, vo=220.0)
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    # actionAngleAdiabatic
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleAdiabatic(pot=[pot], ro=8.0, vo=220.0)
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleAdiabatic(pot=[pot], ro=8.0, vo=220.0)
    # actionAngleStaeckel
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleStaeckel(delta=0.45, pot=pot, ro=8.0, vo=220.0)
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleStaeckel(delta=0.45, pot=pot, ro=8.0, vo=220.0)
    # actionAngleIsochroneApprox
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochroneApprox(b=0.8, pot=pot, ro=8.0, vo=220.0)
    pot = PlummerPotential(normalize=1.0, b=0.7, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochroneApprox(b=0.8, pot=pot, ro=8.0, vo=220.0)
    # actionAngleIsochroneInverse
    pot = IsochronePotential(normalize=1.0, ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochroneInverse(ip=pot, ro=8.0, vo=220.0)
    pot = IsochronePotential(normalize=1.0, ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        actionAngleIsochroneInverse(ip=pot, ro=8.0, vo=220.0)
    return None


def test_actionAngle_inconsistentOrbitUnits_error():
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleIsochrone,
        actionAngleIsochroneApprox,
        actionAngleSpherical,
        actionAngleStaeckel,
    )
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, PlummerPotential

    # actionAngleIsochrone
    pot = IsochronePotential(normalize=1)
    aA = actionAngleIsochrone(ip=pot, ro=8.0, vo=220.0)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    # actionAngleAdiabatic
    aA = actionAngleAdiabatic(pot=[pot], ro=8.0, vo=220.0)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    # actionAngleStaeckel
    aA = actionAngleStaeckel(delta=0.45, pot=pot, ro=8.0, vo=220.0)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    # actionAngleIsochroneApprox
    aA = actionAngleIsochroneApprox(b=0.8, pot=pot, ro=8.0, vo=220.0)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=7.0, vo=220.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    o = Orbit([1.1, 0.2, 1.2, 0.1, 0.2, 0.2], ro=8.0, vo=230.0)
    with pytest.raises(AssertionError) as excinfo:
        aA(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqs(o)
    with pytest.raises(AssertionError) as excinfo:
        aA.actionsFreqsAngles(o)
    return None


def test_actionAngle_input_wrongunits():
    from galpy.actionAngle import actionAngleSpherical
    from galpy.potential import PlummerPotential

    # actionAngleSpherical
    pot = PlummerPotential(normalize=1.0, b=0.7)
    aA = actionAngleSpherical(pot=pot, ro=8.0, vo=220.0)
    with pytest.raises(units.UnitConversionError) as excinfo:
        aA(
            1.0 * units.Gyr,
            0.1 * units.km / units.s,
            1.1 * units.km / units.s,
            0.1 * units.kpc,
            0.2 * units.km / units.s,
            0.1 * units.rad,
        )
    with pytest.raises(units.UnitConversionError) as excinfo:
        aA(
            1.0 * units.kpc,
            0.1 * units.Gyr,
            1.1 * units.km / units.s,
            0.1 * units.kpc,
            0.2 * units.km / units.s,
            0.1 * units.rad,
        )
    return None


def test_actionAngleInverse_input_wrongunits():
    from galpy.actionAngle import actionAngleIsochroneInverse
    from galpy.potential import IsochronePotential

    ip = IsochronePotential(normalize=1.0, b=0.7)
    aAII = actionAngleIsochroneInverse(ip=ip, ro=8.0, vo=220.0)
    with pytest.raises(units.UnitConversionError) as excinfo:
        aAII(
            1.0 * units.Gyr,
            0.1 * units.kpc * units.km / units.s,
            1.1 * units.kpc * units.km / units.s,
            0.1 * units.rad,
            0.2 * units.rad,
            0.1 * units.rad,
        )
    with pytest.raises(units.UnitConversionError) as excinfo:
        aAII(
            1.0 * units.Gyr,
            0.1 * units.kpc * units.km / units.s,
            1.1 * units.kpc * units.km / units.s,
            0.1 * units.km,
            0.2 * units.rad,
            0.1 * units.rad,
        )
    return None


def test_estimateDeltaStaeckel_method_returntype():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=True, ro=8.0, vo=220.0)
    assert isinstance(
        estimateDeltaStaeckel(pot, 1.1, 0.1), units.Quantity
    ), "estimateDeltaStaeckel function does not return Quantity when it should"
    assert isinstance(
        estimateDeltaStaeckel(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3)),
        units.Quantity,
    ), "estimateDeltaStaeckel function does not return Quantity when it should"
    return None


def test_estimateDeltaStaeckel_method_returnunit():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=True, ro=8.0, vo=220.0)
    try:
        estimateDeltaStaeckel(pot, 1.1, 0.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "estimateDeltaStaeckel function does not return Quantity with the right units"
        )
    try:
        estimateDeltaStaeckel(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3)).to(
            units.kpc
        )
    except units.UnitConversionError:
        raise AssertionError(
            "estimateDeltaStaeckel function does not return Quantity with the right units"
        )
    return None


def test_estimateDeltaStaeckel_method_value():
    from galpy.actionAngle import estimateDeltaStaeckel
    from galpy.potential import MiyamotoNagaiPotential

    ro, vo = 9.0, 230.0
    pot = MiyamotoNagaiPotential(normalize=True, ro=ro, vo=vo)
    potu = MiyamotoNagaiPotential(normalize=True)
    assert (
        numpy.fabs(
            estimateDeltaStaeckel(pot, 1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(units.kpc)
            .value
            - estimateDeltaStaeckel(potu, 1.1, 0.1) * ro
        )
        < 10.0**-8.0
    ), "estimateDeltaStaeckel function does not return Quantity with the right value"
    assert numpy.all(
        numpy.fabs(
            estimateDeltaStaeckel(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3))
            .to(units.kpc)
            .value
            - estimateDeltaStaeckel(potu, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3)) * ro
        )
        < 10.0**-8.0
    ), "estimateDeltaStaeckel function does not return Quantity with the right value"
    return None


def test_estimateBIsochrone_method_returntype():
    from galpy.actionAngle import estimateBIsochrone
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=True, ro=8.0, vo=220.0)
    assert isinstance(
        estimateBIsochrone(pot, 1.1, 0.1), units.Quantity
    ), "estimateBIsochrone function does not return Quantity when it should"
    for ii in range(3):
        assert isinstance(
            estimateBIsochrone(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3))[ii],
            units.Quantity,
        ), "estimateBIsochrone function does not return Quantity when it should"
    return None


def test_estimateBIsochrone_method_returnunit():
    from galpy.actionAngle import estimateBIsochrone
    from galpy.potential import MiyamotoNagaiPotential

    pot = MiyamotoNagaiPotential(normalize=True, ro=8.0, vo=220.0)
    try:
        estimateBIsochrone(pot, 1.1, 0.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "estimateBIsochrone function does not return Quantity with the right units"
        )
    for ii in range(3):
        try:
            estimateBIsochrone(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3))[ii].to(
                units.kpc
            )
        except units.UnitConversionError:
            raise AssertionError(
                "estimateBIsochrone function does not return Quantity with the right units"
            )
    return None


def test_estimateBIsochrone_method_value():
    from galpy.actionAngle import estimateBIsochrone
    from galpy.potential import MiyamotoNagaiPotential

    ro, vo = 9.0, 230.0
    pot = MiyamotoNagaiPotential(normalize=True, ro=ro, vo=vo)
    potu = MiyamotoNagaiPotential(normalize=True)
    assert (
        numpy.fabs(
            estimateBIsochrone(pot, 1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(units.kpc)
            .value
            - estimateBIsochrone(potu, 1.1, 0.1) * ro
        )
        < 10.0**-8.0
    ), "estimateBIsochrone function does not return Quantity with the right value"
    for ii in range(3):
        assert numpy.all(
            numpy.fabs(
                estimateBIsochrone(pot, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3))[ii]
                .to(units.kpc)
                .value
                - estimateBIsochrone(potu, 1.1 * numpy.ones(3), 0.1 * numpy.ones(3))[ii]
                * ro
            )
            < 10.0**-8.0
        ), "estimateBIsochrone function does not return Quantity with the right value"
    return None


def test_df_method_turnphysicalon():
    from galpy.df import dehnendf
    from galpy.orbit import Orbit

    df = dehnendf(ro=7.0, vo=230.0)
    df.turn_physical_on()
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "df method does not return Quantity when turn_physical_on has been called"
    assert numpy.fabs(df._ro - 7.0) < 10.0**-10.0, "df method does not work as expected"
    assert (
        numpy.fabs(df._vo - 230.0) < 10.0**-10.0
    ), "df method turn_physical_on does not work as expected"
    df.turn_physical_on(ro=9.0)
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "df method does not return Quantity when turn_physical_on has been called"
    assert numpy.fabs(df._ro - 9.0) < 10.0**-10.0, "df method does not work as expected"
    assert (
        numpy.fabs(df._vo - 230.0) < 10.0**-10.0
    ), "df method turn_physical_on does not work as expected"
    df.turn_physical_on(vo=210.0)
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "df method does not return Quantity when turn_physical_on has been called"
    assert numpy.fabs(df._ro - 9.0) < 10.0**-10.0, "df method does not work as expected"
    assert (
        numpy.fabs(df._vo - 210.0) < 10.0**-10.0
    ), "df method turn_physical_on does not work as expected"
    df.turn_physical_on(ro=10.0 * units.kpc)
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "df method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(df._ro - 10.0) < 10.0**-10.0
    ), "df method does not work as expected"
    assert (
        numpy.fabs(df._vo - 210.0) < 10.0**-10.0
    ), "df method turn_physical_on does not work as expected"
    df.turn_physical_on(vo=190.0 * units.km / units.s)
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "df method does not return Quantity when turn_physical_on has been called"
    assert (
        numpy.fabs(df._ro - 10.0) < 10.0**-10.0
    ), "df method does not work as expected"
    assert (
        numpy.fabs(df._vo - 190.0) < 10.0**-10.0
    ), "df method turn_physical_on does not work as expected"
    return None


def test_df_method_turnphysicaloff():
    from galpy.df import dehnendf
    from galpy.orbit import Orbit

    df = dehnendf(ro=7.0, vo=230.0)
    df.turn_physical_off()
    assert isinstance(
        numpy.atleast_1d(df(Orbit([1.1, 0.1, 1.1])))[0], float
    ), "df method does not return float when turn_physical_off has been called"
    return None


def test_diskdf_method_returntype():
    from galpy.df import dehnendf, shudf
    from galpy.orbit import Orbit

    df = dehnendf(ro=8.0, vo=220.0)
    dfs = shudf(ro=8.0, vo=220.0)
    assert isinstance(
        df(Orbit([1.1, 0.1, 1.1])), units.Quantity
    ), "diskdf method __call__ does not return Quantity when it should"
    assert isinstance(
        df.targetSigma2(1.2), units.Quantity
    ), "diskdf method targetSigma2 does not return Quantity when it should"
    assert isinstance(
        df.targetSurfacemass(1.2), units.Quantity
    ), "diskdf method targetSurfacemass does not return Quantity when it should"
    assert isinstance(
        df.targetSurfacemassLOS(1.2, 40.0), units.Quantity
    ), "diskdf method targetSurfacemassLOS does not return Quantity when it should"
    assert isinstance(
        df.surfacemassLOS(1.2, 35.0), units.Quantity
    ), "diskdf method surfacemassLOS does not return Quantity when it should"
    assert isinstance(
        df.sampledSurfacemassLOS(1.2), units.Quantity
    ), "diskdf method sampledSurfacemassLOS does not return Quantity when it should"
    assert isinstance(
        df.sampleVRVT(1.1), units.Quantity
    ), "diskdf method sampleVRVT does not return Quantity when it should"
    assert isinstance(
        df.sampleLOS(12.0)[0].R(), units.Quantity
    ), "diskdf method sampleLOS does not return Quantity when it should"
    assert isinstance(
        df.sample()[0].R(), units.Quantity
    ), "diskdf method sample does not return Quantity when it should"
    assert isinstance(
        dfs.sample()[0].R(), units.Quantity
    ), "diskdf method sample does not return Quantity when it should"
    assert isinstance(
        df.asymmetricdrift(0.8), units.Quantity
    ), "diskdf method asymmetricdrift does not return Quantity when it should"
    assert isinstance(
        df.surfacemass(1.1), units.Quantity
    ), "diskdf method  does not return Quantity when it should"
    assert isinstance(
        df.sigma2surfacemass(1.2), units.Quantity
    ), "diskdf method sigma2surfacemass does not return Quantity when it should"
    assert isinstance(
        df.oortA(1.2), units.Quantity
    ), "diskdf method oortA does not return Quantity when it should"
    assert isinstance(
        df.oortB(1.2), units.Quantity
    ), "diskdf method oortB does not return Quantity when it should"
    assert isinstance(
        df.oortC(1.2), units.Quantity
    ), "diskdf method oortC does not return Quantity when it should"
    assert isinstance(
        df.oortK(1.2), units.Quantity
    ), "diskdf method oortK does not return Quantity when it should"
    assert isinstance(
        df.sigma2(1.2), units.Quantity
    ), "diskdf method sigma2 does not return Quantity when it should"
    assert isinstance(
        df.sigmaT2(1.2), units.Quantity
    ), "diskdf method sigmaT2 does not return Quantity when it should"
    assert isinstance(
        df.sigmaR2(1.2), units.Quantity
    ), "diskdf method sigmaR2 does not return Quantity when it should"
    assert isinstance(
        df.meanvT(1.2), units.Quantity
    ), "diskdf method meanvT does not return Quantity when it should"
    assert isinstance(
        df.meanvR(1.2), units.Quantity
    ), "diskdf method meanvR does not return Quantity when it should"
    assert isinstance(
        df.vmomentsurfacemass(1.1, 0, 0), units.Quantity
    ), "diskdf method vmomentsurfacemass does not return Quantity when it should"
    assert isinstance(
        df.vmomentsurfacemass(1.1, 1, 0), units.Quantity
    ), "diskdf method vmomentsurfacemass does not return Quantity when it should"
    assert isinstance(
        df.vmomentsurfacemass(1.1, 1, 1), units.Quantity
    ), "diskdf method vmomentsurfacemass does not return Quantity when it should"
    return None


def test_diskdf_method_returnunit():
    from galpy.df import dehnendf
    from galpy.orbit import Orbit

    df = dehnendf(ro=8.0, vo=220.0)
    try:
        df(Orbit([1.1, 0.1, 1.1])).to(1 / (units.km / units.s) ** 2 / units.kpc**2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method __call__ does not return Quantity with the right units"
        )
    try:
        df.targetSigma2(1.2).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method targetSigma2 does not return Quantity with the right units"
        )
    try:
        df.targetSurfacemass(1.2).to(units.Msun / units.pc**2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method targetSurfacemass does not return Quantity with the right units"
        )
    try:
        df.targetSurfacemassLOS(1.2, 30.0).to(units.Msun / units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method targetSurfacemassLOS does not return Quantity with the right units"
        )
    try:
        df.surfacemassLOS(1.2, 40.0).to(units.Msun / units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method surfacemassLOS does not return Quantity with the right units"
        )
    try:
        df.sampledSurfacemassLOS(1.2).to(units.pc)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method sampledSurfacemassLOS does not return Quantity with the right units"
        )
    try:
        df.sampleVRVT(1.2).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method sampleVRVT does not return Quantity with the right units"
        )
    try:
        df.asymmetricdrift(1.2).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method asymmetricdrift does not return Quantity with the right units"
        )
    try:
        df.surfacemass(1.2).to(units.Msun / units.pc**2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method surfacemass does not return Quantity with the right units"
        )
    try:
        df.sigma2surfacemass(1.2).to(
            units.Msun / units.pc**2 * (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method surfacemass does not return Quantity with the right units"
        )
    try:
        df.oortA(1.2).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method oortA does not return Quantity with the right units"
        )
    try:
        df.oortB(1.2).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method oortB does not return Quantity with the right units"
        )
    try:
        df.oortC(1.2).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method oortC does not return Quantity with the right units"
        )
    try:
        df.oortK(1.2).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method oortK does not return Quantity with the right units"
        )
    try:
        df.sigma2(1.2).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method sigma2 does not return Quantity with the right units"
        )
    try:
        df.sigmaT2(1.2).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method sigmaT2 does not return Quantity with the right units"
        )
    try:
        df.sigmaR2(1.2).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method sigmaR2 does not return Quantity with the right units"
        )
    try:
        df.meanvR(1.2).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method meanvR does not return Quantity with the right units"
        )
    try:
        df.meanvT(1.2).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method meanvT does not return Quantity with the right units"
        )
    try:
        df.vmomentsurfacemass(1.1, 0, 0).to(units.Msun / units.pc**2)
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method vmomentsurfacemass does not return Quantity with the right units"
        )
    try:
        df.vmomentsurfacemass(1.1, 1, 0).to(
            units.Msun / units.pc**2 * (units.km / units.s)
        )
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method vmomentsurfacemass does not return Quantity with the right units"
        )
    try:
        df.vmomentsurfacemass(1.1, 1, 1).to(
            units.Msun / units.pc**2 * (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method vmomentsurfacemass does not return Quantity with the right units"
        )
    try:
        df.vmomentsurfacemass(1.1, 0, 2).to(
            units.Msun / units.pc**2 * (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "diskdf method vmomentsurfacemass does not return Quantity with the right units"
        )
    return None


def test_diskdf_method_value():
    from galpy.df import dehnendf
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    df = dehnendf(ro=ro, vo=vo)
    dfnou = dehnendf()
    assert (
        numpy.fabs(
            df(Orbit([1.1, 0.1, 1.1]))
            .to(1 / units.kpc**2 / (units.km / units.s) ** 2)
            .value
            - dfnou(Orbit([1.1, 0.1, 1.1])) / vo**2 / ro**2
        )
        < 10.0**-8.0
    ), "diskdf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            df.targetSigma2(1.2).to((units.km / units.s) ** 2).value
            - dfnou.targetSigma2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method targetSigma2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.targetSurfacemass(1.2).to(units.Msun / units.pc**2).value
            - dfnou.targetSurfacemass(1.2) * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method targetSurfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(1.2, 40.0).to(units.Msun / units.pc).value
            - dfnou.targetSurfacemassLOS(1.2, 40.0)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * ro
            * 1000.0
        )
        < 10.0**-8.0
    ), "diskdf method targetSurfacemassLOS does not return correct Quantity"
    assert (
        numpy.fabs(
            df.surfacemassLOS(1.2, 35.0).to(units.Msun / units.pc).value
            - dfnou.surfacemassLOS(1.2, 35.0)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * ro
            * 1000.0
        )
        < 10.0**-8.0
    ), "diskdf method surfacemassLOS does not return correct Quantity"
    assert (
        numpy.fabs(
            df.asymmetricdrift(0.8).to(units.km / units.s).value
            - dfnou.asymmetricdrift(0.8) * vo
        )
        < 10.0**-8.0
    ), "diskdf method asymmetricdrift does not return correct Quantity"
    assert (
        numpy.fabs(
            df.surfacemass(1.1).to(units.Msun / units.pc**2).value
            - dfnou.surfacemass(1.1) * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method  does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigma2surfacemass(1.2)
            .to(units.Msun / units.pc**2 * (units.km / units.s) ** 2)
            .value
            - dfnou.sigma2surfacemass(1.2)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigma2surfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortA(1.2).to(1 / units.Gyr).value
            - dfnou.oortA(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortA does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortB(1.2).to(1 / units.Gyr).value
            - dfnou.oortB(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortB does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortC(1.2).to(1 / units.Gyr).value
            - dfnou.oortC(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortC does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortK(1.2).to(1 / units.Gyr).value
            - dfnou.oortK(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortK does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigma2(1.2).to((units.km / units.s) ** 2).value
            - dfnou.sigma2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigma2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigmaT2(1.2).to((units.km / units.s) ** 2).value
            - dfnou.sigmaT2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigmaT2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigmaR2(1.2).to((units.km / units.s) ** 2).value
            - dfnou.sigmaR2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigmaR2 does not return correct Quantity"
    assert (
        numpy.fabs(df.meanvT(1.2).to(units.km / units.s).value - dfnou.meanvT(1.2) * vo)
        < 10.0**-8.0
    ), "diskdf method meanvT does not return correct Quantity"
    assert (
        numpy.fabs(df.meanvR(1.2).to(units.km / units.s).value - dfnou.meanvR(1.2) * vo)
        < 10.0**-8.0
    ), "diskdf method meanvT does not return correct Quantity"
    assert (
        numpy.fabs(
            df.vmomentsurfacemass(1.1, 0, 0).to(units.Msun / units.pc**2).value
            - dfnou.vmomentsurfacemass(1.1, 0, 0)
            * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method vmomentsurfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.vmomentsurfacemass(1.1, 0, 1)
            .to(units.Msun / units.pc**2 * (units.km / units.s) ** 1)
            .value
            - dfnou.vmomentsurfacemass(1.1, 0, 1)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * vo
        )
        < 10.0**-8.0
    ), "diskdf method vmomentsurfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.vmomentsurfacemass(1.1, 1, 1)
            .to(units.Msun / units.pc**2 * (units.km / units.s) ** 2)
            .value
            - dfnou.vmomentsurfacemass(1.1, 1, 1)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method vmomentsurfacemass does not return correct Quantity"
    return None


def test_diskdf_sample():
    # Test that the sampling routines work with Quantity output
    from galpy.df import dehnendf, shudf

    ro, vo = 7.0, 230.0
    df = dehnendf(ro=ro, vo=vo)
    dfnou = dehnendf()
    dfs = shudf(ro=ro, vo=vo)
    dfsnou = shudf()
    # sampledSurfacemassLOS
    numpy.random.seed(1)
    du = (
        df.sampledSurfacemassLOS(11.0 * units.deg, n=1, maxd=10.0 * units.kpc)
        .to(units.kpc)
        .value
        / ro
    )
    numpy.random.seed(1)
    dnou = dfnou.sampledSurfacemassLOS(11.0 * numpy.pi / 180.0, n=1, maxd=10.0 / ro)
    assert (
        numpy.fabs(du - dnou) < 10.0**-8.0
    ), "diskdf sampling method sampledSurfacemassLOS does not return expected Quantity"
    # sampleVRVT
    numpy.random.seed(1)
    du = df.sampleVRVT(1.1, n=1).to(units.km / units.s).value / vo
    numpy.random.seed(1)
    dnou = dfnou.sampleVRVT(1.1, n=1)
    assert numpy.all(
        numpy.fabs(du - dnou) < 10.0**-8.0
    ), "diskdf sampling method sampleVRVT does not return expected Quantity"
    # sampleLOS
    numpy.random.seed(1)
    du = df.sampleLOS(11.0 * units.deg, n=1)
    numpy.random.seed(1)
    dnou = dfnou.sampleLOS(11.0, n=1, deg=True)
    assert numpy.all(
        numpy.fabs(numpy.array(du[0].vxvv) - numpy.array(dnou[0].vxvv)) < 10.0**-8.0
    ), "diskdf sampling method sampleLOS does not work as expected with Quantity input"
    # sample
    numpy.random.seed(1)
    du = df.sample(rrange=[4.0 * units.kpc, 12.0 * units.kpc], n=1)
    numpy.random.seed(1)
    dnou = dfnou.sample(rrange=[4.0 / ro, 12.0 / ro], n=1)
    assert numpy.all(
        numpy.fabs(numpy.array(du[0].vxvv) - numpy.array(dnou[0].vxvv)) < 10.0**-8.0
    ), "diskdf sampling method sample does not work as expected with Quantity input"
    # sample for Shu
    numpy.random.seed(1)
    du = dfs.sample(rrange=[4.0 * units.kpc, 12.0 * units.kpc], n=1)
    numpy.random.seed(1)
    dnou = dfsnou.sample(rrange=[4.0 / ro, 12.0 / ro], n=1)
    assert numpy.all(
        numpy.fabs(numpy.array(du[0].vxvv) - numpy.array(dnou[0].vxvv)) < 10.0**-8.0
    ), "diskdf sampling method sample does not work as expected with Quantity input"
    return None


def test_diskdf_method_inputAsQuantity():
    # Using the decorator
    from galpy.df import dehnendf
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    df = dehnendf(ro=ro, vo=vo)
    dfnou = dehnendf()
    assert (
        numpy.fabs(
            df.targetSigma2(1.2 * ro * units.kpc).to((units.km / units.s) ** 2).value
            - dfnou.targetSigma2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method targetSigma2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.targetSurfacemass(1.2 * ro * units.kpc)
            .to(units.Msun / units.pc**2)
            .value
            - dfnou.targetSurfacemass(1.2) * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method targetSurfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.asymmetricdrift(0.8 * ro * units.kpc).to(units.km / units.s).value
            - dfnou.asymmetricdrift(0.8) * vo
        )
        < 10.0**-8.0
    ), "diskdf method asymmetricdrift does not return correct Quantity"
    assert (
        numpy.fabs(
            df.surfacemass(1.1 * ro * units.kpc).to(units.Msun / units.pc**2).value
            - dfnou.surfacemass(1.1) * conversion.surfdens_in_msolpc2(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method  does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigma2surfacemass(1.2 * ro * units.kpc)
            .to(units.Msun / units.pc**2 * (units.km / units.s) ** 2)
            .value
            - dfnou.sigma2surfacemass(1.2)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigma2surfacemass does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortA(1.2 * ro * units.kpc).to(1 / units.Gyr).value
            - dfnou.oortA(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortA does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortB(1.2 * ro * units.kpc).to(1 / units.Gyr).value
            - dfnou.oortB(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortB does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortC(1.2 * ro * units.kpc).to(1 / units.Gyr).value
            - dfnou.oortC(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortC does not return correct Quantity"
    assert (
        numpy.fabs(
            df.oortK(1.2 * ro * units.kpc).to(1 / units.Gyr).value
            - dfnou.oortK(1.2) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "diskdf method oortK does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigma2(1.2 * ro * units.kpc).to((units.km / units.s) ** 2).value
            - dfnou.sigma2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigma2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigmaT2(1.2 * ro * units.kpc).to((units.km / units.s) ** 2).value
            - dfnou.sigmaT2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigmaT2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.sigmaR2(1.2 * ro * units.kpc).to((units.km / units.s) ** 2).value
            - dfnou.sigmaR2(1.2) * vo**2
        )
        < 10.0**-8.0
    ), "diskdf method sigmaR2 does not return correct Quantity"
    assert (
        numpy.fabs(
            df.meanvT(1.2 * ro * units.kpc).to(units.km / units.s).value
            - dfnou.meanvT(1.2) * vo
        )
        < 10.0**-8.0
    ), "diskdf method meanvT does not return correct Quantity"
    assert (
        numpy.fabs(
            df.meanvR(1.2 * ro * units.kpc).to(units.km / units.s).value
            - dfnou.meanvR(1.2) * vo
        )
        < 10.0**-8.0
    ), "diskdf method meanvT does not return correct Quantity"
    return None


def test_diskdf_method_inputAsQuantity_special():
    from galpy.df import dehnendf, shudf
    from galpy.util import conversion

    ro, vo = 7.0, 230.0
    df = dehnendf(ro=ro, vo=vo)
    dfnou = dehnendf()
    dfs = shudf(ro=ro, vo=vo)
    dfsnou = shudf()
    assert (
        numpy.fabs(
            df(
                0.6 * vo**2.0 * units.km**2 / units.s**2,
                1.1 * vo * ro * units.kpc * units.km / units.s,
            )
            .to(1 / units.kpc**2 / (units.km / units.s) ** 2)
            .value
            - dfnou(0.6, 1.1) / vo**2 / ro**2
        )
        < 10.0**-6.0
    ), "diskdf method __call__ with Quantity input does not return correct Quantity"
    assert (
        numpy.fabs(
            dfs(
                0.6 * vo**2.0 * units.km**2 / units.s**2,
                1.1 * vo * ro * units.kpc * units.km / units.s,
            )
            .to(1 / units.kpc**2 / (units.km / units.s) ** 2)
            .value
            - dfsnou(0.6, 1.1) / vo**2 / ro**2
        )
        < 10.0**-6.0
    ), "diskdf method __call__ with Quantity input does not return correct Quantity"
    assert (
        numpy.fabs(
            df.targetSurfacemassLOS(1.2 * ro * units.kpc, 40.0 * units.deg)
            .to(units.Msun / units.pc)
            .value
            - dfnou.targetSurfacemassLOS(1.2, 40.0)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * ro
            * 1000.0
        )
        < 10.0**-8.0
    ), "diskdf method targetSurfacemassLOS with Quantity input does not return correct Quantity"
    assert (
        numpy.fabs(
            df.surfacemassLOS(1.2 * ro * units.kpc, 35.0 * units.deg)
            .to(units.Msun / units.pc)
            .value
            - dfnou.surfacemassLOS(1.2, 35.0)
            * conversion.surfdens_in_msolpc2(vo, ro)
            * ro
            * 1000.0
        )
        < 10.0**-8.0
    ), "diskdf method surfacemassLOS does with Quantity input not return correct Quantity"
    assert (
        numpy.fabs(
            df.vmomentsurfacemass(
                1.1, 0, 0, ro=9.0 * units.kpc, vo=245.0 * units.km / units.s
            )
            .to(units.Msun / units.pc**2)
            .value
            - dfnou.vmomentsurfacemass(1.1, 0, 0)
            * conversion.surfdens_in_msolpc2(245, 9.0)
        )
        < 10.0**-8.0
    ), "diskdf method vmomentsurfacemass does with Quantity input not return correct Quantity"
    return None


def test_diskdf_setup_roAsQuantity():
    from galpy.df import dehnendf

    ro = 7.0
    df = dehnendf(ro=ro * units.kpc)
    assert (
        numpy.fabs(df._ro - ro) < 10.0**-10.0
    ), "ro in diskdf setup as Quantity does not work as expected"
    return None


def test_diskdf_setup_roAsQuantity_oddunits():
    from galpy.df import dehnendf

    ro = 7000.0
    df = dehnendf(ro=ro * units.lyr)
    assert (
        numpy.fabs(df._ro - ro * (units.lyr).to(units.kpc)) < 10.0**-10.0
    ), "ro in diskdf setup as Quantity does not work as expected"
    return None


def test_diskdf_setup_voAsQuantity():
    from galpy.df import dehnendf

    vo = 230.0
    df = dehnendf(vo=vo * units.km / units.s)
    assert (
        numpy.fabs(df._vo - vo) < 10.0**-10.0
    ), "vo in diskdf setup as Quantity does not work as expected"
    return None


def test_diskdf_setup_voAsQuantity_oddunits():
    from galpy.df import dehnendf

    vo = 230.0
    df = dehnendf(vo=vo * units.pc / units.Myr)
    assert (
        numpy.fabs(df._vo - vo * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in diskdf setup as Quantity does not work as expected"
    return None


def test_diskdf_setup_profileAsQuantity():
    from galpy.df import dehnendf, shudf
    from galpy.orbit import Orbit

    df = dehnendf(
        ro=8.0,
        vo=220.0,
        profileParams=(9.0 * units.kpc, 10.0 * units.kpc, 20.0 * units.km / units.s),
    )
    dfs = shudf(
        ro=8.0,
        vo=220.0,
        profileParams=(9.0 * units.kpc, 10.0 * units.kpc, 20.0 * units.km / units.s),
    )
    assert (
        numpy.fabs(df._surfaceSigmaProfile._params[0] - 9.0 / 8.0) < 10.0**-10.0
    ), "hR in diskdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._surfaceSigmaProfile._params[1] - 10.0 / 8.0) < 10.0**-10.0
    ), "hsR in diskdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._surfaceSigmaProfile._params[2] - 20.0 / 220.0) < 10.0**-10.0
    ), "sR in diskdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(dfs._surfaceSigmaProfile._params[0] - 9.0 / 8.0) < 10.0**-10.0
    ), "hR in diskdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(dfs._surfaceSigmaProfile._params[1] - 10.0 / 8.0) < 10.0**-10.0
    ), "hsR in diskdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(dfs._surfaceSigmaProfile._params[2] - 20.0 / 220.0) < 10.0**-10.0
    ), "sR in diskdf setup as Quantity does not work as expected"
    return None


def test_evolveddiskdf_method_returntype():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=8.0, vo=220.0)
    from galpy.df import evolveddiskdf

    edfwarm = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1])
    assert isinstance(
        edfwarm(o), units.Quantity
    ), "evolveddiskdf method __call__ does not return Quantity when it should"
    assert isinstance(
        edfwarm.oortA(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ),
        units.Quantity,
    ), "evolveddiskdf method oortA does not return Quantity when it should"
    assert isinstance(
        edfwarm.oortB(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ),
        units.Quantity,
    ), "evolveddiskdf method oortB does not return Quantity when it should"
    assert isinstance(
        edfwarm.oortC(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ),
        units.Quantity,
    ), "evolveddiskdf method oortC does not return Quantity when it should"
    assert isinstance(
        edfwarm.oortK(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ),
        units.Quantity,
    ), "evolveddiskdf method oortK does not return Quantity when it should"
    assert isinstance(
        edfwarm.sigmaT2(1.2, grid=True, returnGrid=False, gridpoints=3), units.Quantity
    ), "evolveddiskdf method sigmaT2 does not return Quantity when it should"
    assert isinstance(
        edfwarm.sigmaR2(1.2, grid=True, returnGrid=False, gridpoints=3), units.Quantity
    ), "evolveddiskdf method sigmaR2 does not return Quantity when it should"
    assert isinstance(
        edfwarm.sigmaRT(1.2, grid=True, returnGrid=False, gridpoints=3), units.Quantity
    ), "evolveddiskdf method sigmaRT does not return Quantity when it should"
    assert isinstance(
        edfwarm.vertexdev(1.2, grid=True, returnGrid=False, gridpoints=3),
        units.Quantity,
    ), "evolveddiskdf method vertexdev does not return Quantity when it should"
    assert isinstance(
        edfwarm.meanvT(1.2, grid=True, returnGrid=False, gridpoints=3), units.Quantity
    ), "evolveddiskdf method meanvT does not return Quantity when it should"
    assert isinstance(
        edfwarm.meanvR(1.2, grid=True, returnGrid=False, gridpoints=3), units.Quantity
    ), "evolveddiskdf method meanvR does not return Quantity when it should"
    return None


def test_evolveddiskdf_method_returnunit():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=8.0, vo=220.0)
    from galpy.df import evolveddiskdf

    edfwarm = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    from galpy.orbit import Orbit

    try:
        edfwarm(Orbit([1.1, 0.1, 1.1, 0.2])).to(
            1 / (units.km / units.s) ** 2 / units.kpc**2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method __call__ does not return Quantity with the right units"
        )
    try:
        edfwarm.oortA(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method oortA does not return Quantity with the right units"
        )
    try:
        edfwarm.oortB(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method oortB does not return Quantity with the right units"
        )
    try:
        edfwarm.oortC(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method oortC does not return Quantity with the right units"
        )
    try:
        edfwarm.oortK(
            1.2,
            grid=True,
            returnGrids=False,
            gridpoints=3,
            derivRGrid=True,
            derivphiGrid=True,
            derivGridpoints=3,
        ).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method oortK does not return Quantity with the right units"
        )
    try:
        edfwarm.sigmaT2(1.2, grid=True, returnGrid=False, gridpoints=3).to(
            (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method sigmaT2 does not return Quantity with the right units"
        )
    try:
        edfwarm.sigmaR2(1.2, grid=True, returnGrid=False, gridpoints=3).to(
            (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method sigmaR2 does not return Quantity with the right units"
        )
    try:
        edfwarm.sigmaRT(1.2, grid=True, returnGrid=False, gridpoints=3).to(
            (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method sigmaRT does not return Quantity with the right units"
        )
    try:
        edfwarm.vertexdev(1.2, grid=True, returnGrid=False, gridpoints=3).to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method vertexdev does not return Quantity with the right units"
        )
    try:
        edfwarm.meanvR(1.2, grid=True, returnGrid=False, gridpoints=3).to(
            units.km / units.s
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method meanvR does not return Quantity with the right units"
        )
    try:
        edfwarm.meanvT(1.2, grid=True, returnGrid=False, gridpoints=3).to(
            units.km / units.s
        )
    except units.UnitConversionError:
        raise AssertionError(
            "evolveddiskdf method meanvT does not return Quantity with the right units"
        )
    return None


def test_evolveddiskdf_method_value():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro, vo = 6.0, 230.0
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=ro, vo=vo)
    from galpy.df import evolveddiskdf

    edfwarm = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    idfwarmnou = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15))
    edfwarmnou = evolveddiskdf(idfwarmnou, [lp, ep], to=-150.0)
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1])
    assert (
        numpy.fabs(
            edfwarm(o).to(1 / units.kpc**2 / (units.km / units.s) ** 2).value
            - edfwarmnou(o) / ro**2 / vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method __call__ does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortA(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortA(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortA does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortB(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortB(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortB does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortC(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortC(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortC does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortK(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortK(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortK does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaT2(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaT2(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaT2 does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaR2(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaR2(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaR2 does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaRT(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaRT(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaRT does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.vertexdev(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to(units.rad)
            .value
            - edfwarmnou.vertexdev(1.2, grid=True, returnGrid=False, gridpoints=3)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method vertexdev does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.meanvT(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to(units.km / units.s)
            .value
            - edfwarmnou.meanvT(1.2, grid=True, returnGrid=False, gridpoints=3) * vo
        )
        < 10.0**-8.0
    ), "evolveddiskdf method meanvT does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.meanvR(1.2, grid=True, returnGrid=False, gridpoints=3)
            .to(units.km / units.s)
            .value
            - edfwarmnou.meanvR(1.2, grid=True, returnGrid=False, gridpoints=3) * vo
        )
        < 10.0**-8.0
    ), "evolveddiskdf method meanvR does not return correct Quantity when it should"
    return None


def test_evolveddiskdf_method_inputAsQuantity():
    # Those that use the decorator
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro, vo = 6.0, 230.0
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=ro, vo=vo)
    from galpy.df import evolveddiskdf

    edfwarm = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    idfwarmnou = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15))
    edfwarmnou = evolveddiskdf(idfwarmnou, [lp, ep], to=-150.0)
    from galpy.orbit import Orbit

    assert (
        numpy.fabs(
            edfwarm.oortA(
                1.2 * ro * units.kpc,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortA(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortA does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortB(
                1.2 * ro * units.kpc,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortB(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortB does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortC(
                1.2 * ro * units.kpc,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortC(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortC does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.oortK(
                1.2 * ro * units.kpc,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            .to(1 / units.Gyr)
            .value
            - edfwarmnou.oortK(
                1.2,
                grid=True,
                returnGrids=False,
                gridpoints=3,
                derivRGrid=True,
                derivphiGrid=True,
                derivGridpoints=3,
            )
            * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method oortK does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaT2(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaT2(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaT2 does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaR2(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaR2(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaR2 does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.sigmaRT(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to((units.km / units.s) ** 2)
            .value
            - edfwarmnou.sigmaRT(1.2, grid=True, returnGrid=False, gridpoints=3) * vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method sigmaRT does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.vertexdev(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to(units.rad)
            .value
            - edfwarmnou.vertexdev(1.2, grid=True, returnGrid=False, gridpoints=3)
        )
        < 10.0**-8.0
    ), "evolveddiskdf method vertexdev does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.meanvT(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to(units.km / units.s)
            .value
            - edfwarmnou.meanvT(1.2, grid=True, returnGrid=False, gridpoints=3) * vo
        )
        < 10.0**-8.0
    ), "evolveddiskdf method meanvT does not return correct Quantity when it should"
    assert (
        numpy.fabs(
            edfwarm.meanvR(
                1.2 * ro * units.kpc, grid=True, returnGrid=False, gridpoints=3
            )
            .to(units.km / units.s)
            .value
            - edfwarmnou.meanvR(1.2, grid=True, returnGrid=False, gridpoints=3) * vo
        )
        < 10.0**-8.0
    ), "evolveddiskdf method meanvR does not return correct Quantity when it should"
    return None


def test_evolveddiskdf_method_inputAsQuantity_special():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro, vo = 6.0, 230.0
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=ro, vo=vo)
    from galpy.df import evolveddiskdf

    edfwarm = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    idfwarmnou = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15))
    edfwarmnou = evolveddiskdf(idfwarmnou, [lp, ep], to=-150.0)
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1])
    ts = numpy.linspace(0.0, -150.0, 101)
    assert numpy.all(
        numpy.fabs(
            edfwarm(o, ts * conversion.time_in_Gyr(vo, ro) * units.Gyr)
            .to(1 / units.kpc**2 / (units.km / units.s) ** 2)
            .value
            - edfwarmnou(o, ts) / ro**2 / vo**2
        )
        < 10.0**-8.0
    ), "evolveddiskdf method __call__ does not return correct Quantity when it should"
    return None


def test_evolveddiskdf_setup_roAsQuantity():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro = 7.0
    idfwarm = dehnendf(
        beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=ro * units.kpc
    )
    from galpy.df import evolveddiskdf

    df = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    assert (
        numpy.fabs(df._ro - ro) < 10.0**-10.0
    ), "ro in evolveddiskdf setup as Quantity does not work as expected"
    return None


def test_evolveddiskdf_setup_roAsQuantity_oddunits():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro = 7000.0
    idfwarm = dehnendf(
        beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), ro=ro * units.lyr
    )
    from galpy.df import evolveddiskdf

    df = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    assert (
        numpy.fabs(df._ro - ro * (units.lyr).to(units.kpc)) < 10.0**-10.0
    ), "ro in evolveddiskdf setup as Quantity does not work as expected"
    return None


def test_evolveddiskdf_setup_voAsQuantity():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    vo = 230.0
    idfwarm = dehnendf(
        beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), vo=vo * units.km / units.s
    )
    from galpy.df import evolveddiskdf

    df = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    assert (
        numpy.fabs(df._vo - vo) < 10.0**-10.0
    ), "vo in evolveddiskdf setup as Quantity does not work as expected"
    return None


def test_evolveddiskdf_setup_voAsQuantity_oddunits():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    vo = 230.0
    idfwarm = dehnendf(
        beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), vo=vo * units.pc / units.Myr
    )
    from galpy.df import evolveddiskdf

    df = evolveddiskdf(idfwarm, [lp, ep], to=-150.0)
    assert (
        numpy.fabs(df._vo - vo * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in evolveddiskdf setup as Quantity does not work as expected"
    return None


def test_evolveddiskdf_setup_toAsQuantity():
    from galpy.df import dehnendf
    from galpy.potential import EllipticalDiskPotential, LogarithmicHaloPotential
    from galpy.util import conversion

    lp = LogarithmicHaloPotential(normalize=1.0)
    ep = EllipticalDiskPotential(
        twophio=0.05, phib=0.0, p=0.0, tform=-150.0, tsteady=125.0
    )
    ro, vo = 7.0, 230.0
    idfwarm = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.15), vo=vo, ro=ro)
    from galpy.df import evolveddiskdf

    df = evolveddiskdf(idfwarm, [lp, ep], to=-3.0 * units.Gyr)
    assert (
        numpy.fabs(df._to + 3.0 / conversion.time_in_Gyr(vo, ro)) < 10.0**-10.0
    ), "to in evolveddiskdf setup as Quantity does not work as expected"
    return None


def test_quasiisothermaldf_method_returntype():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=8.0,
        vo=220.0,
    )
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    R = numpy.array([1.0, 1.1, 1.2, 1.3])
    z = numpy.array([-0.1, 0.0, 0.1, 0.2])
    assert isinstance(
        qdf(o), units.Quantity
    ), "quasiisothermaldf method __call__ does not return Quantity when it should"
    assert isinstance(
        qdf.estimate_hr(1.1), units.Quantity
    ), "quasiisothermaldf method estimate_hr does not return Quantity when it should"
    assert isinstance(
        qdf.estimate_hz(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method estimate_hz does not return Quantity when it should"
    assert isinstance(
        qdf.estimate_hsr(1.1), units.Quantity
    ), "quasiisothermaldf method estimate_hsr does not return Quantity when it should"
    assert isinstance(
        qdf.estimate_hsz(1.1), units.Quantity
    ), "quasiisothermaldf method estimate_hsz does not return Quantity when it should"
    assert isinstance(
        qdf.surfacemass_z(1.1), units.Quantity
    ), "quasiisothermaldf method surfacemass_z does not return Quantity when it should"
    assert isinstance(
        qdf.density(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method density does not return Quantity when it should"
    assert isinstance(
        qdf.sigmaR2(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sigmaR2 does not return Quantity when it should"
    assert isinstance(
        qdf.sigmaT2(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sigmaT2 does not return Quantity when it should"
    assert isinstance(
        qdf.sigmaz2(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sigmaz2 does not return Quantity when it should"
    assert isinstance(
        qdf.sigmaRz(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sigmaRz does not return Quantity when it should"
    assert isinstance(
        qdf.tilt(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method tilt does not return Quantity when it should"
    assert isinstance(
        qdf.meanvR(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanvR does not return Quantity when it should"
    assert isinstance(
        qdf.meanvT(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanvT does not return Quantity when it should"
    assert isinstance(
        qdf.meanvz(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanvz does not return Quantity when it should"
    assert isinstance(
        qdf.meanjr(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanjr does not return Quantity when it should"
    assert isinstance(
        qdf.meanlz(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanlz does not return Quantity when it should"
    assert isinstance(
        qdf.meanjz(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method meanjz does not return Quantity when it should"
    assert isinstance(
        qdf.sampleV(1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sampleV does not return Quantity when it should"
    assert isinstance(
        qdf.sampleV_interpolate(R, z, 0.1, 0.1), units.Quantity
    ), "quasiisothermaldf method sampleV_interpolate does not return Quantity when it should"
    assert isinstance(
        qdf.pvR(0.1, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvR does not return Quantity when it should"
    assert isinstance(
        qdf.pvT(1.1, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvT does not return Quantity when it should"
    assert isinstance(
        qdf.pvz(0.1, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvz does not return Quantity when it should"
    assert isinstance(
        qdf.pvRvT(0.1, 1.1, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvRvT does not return Quantity when it should"
    assert isinstance(
        qdf.pvRvz(0.1, 0.2, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvRvz does not return Quantity when it should"
    assert isinstance(
        qdf.pvTvz(1.1, 1.1, 1.1, 0.1), units.Quantity
    ), "quasiisothermaldf method pvTvz does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 0, 0, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 1, 0, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 0, 1, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 0, 0, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 1, 1, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.vmomentdensity(1.1, 0.1, 2, 1, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 0, 0, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 1, 0, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 0, 1, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 0, 0, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 1, 1, 0, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    assert isinstance(
        qdf.jmomentdensity(1.1, 0.1, 2, 1, 1, gl=True), units.Quantity
    ), "quasiisothermaldf method jmomentdensity does not return Quantity when it should"
    return None


def test_quasiisothermaldf_method_returnunit():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=8.0,
        vo=220.0,
    )
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    R = numpy.array([0.6, 0.7, 0.8, 0.9, 1.0])
    z = numpy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    try:
        qdf(o).to(1 / (units.km / units.s) ** 3 / units.kpc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method __call__ does not return Quantity with the right units"
        )
    try:
        qdf.estimate_hr(1.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method estimate_hr does not return Quantity with the right units"
        )
    try:
        qdf.estimate_hz(1.1, 0.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method estimate_hz does not return Quantity with the right units"
        )
    try:
        qdf.estimate_hsr(1.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method estimate_hsr does not return Quantity with the right units"
        )
    try:
        qdf.estimate_hsz(1.1).to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method estimate_hsz does not return Quantity with the right units"
        )
    try:
        qdf.surfacemass_z(1.1).to(1 / units.pc**2)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method surfacemass_z does not return Quantity with the right units"
        )
    try:
        qdf.density(1.1, 0.1).to(1 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method density does not return Quantity with the right units"
        )
    try:
        qdf.sigmaR2(1.1, 0.1).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sigmaR2 does not return Quantity with the right units"
        )
    try:
        qdf.sigmaRz(1.1, 0.1).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sigmaRz does not return Quantity with the right units"
        )
    try:
        qdf.sigmaT2(1.1, 0.1).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sigmaT2 does not return Quantity with the right units"
        )
    try:
        qdf.sigmaz2(1.1, 0.1).to((units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sigmaz2 does not return Quantity with the right units"
        )
    try:
        qdf.tilt(1.1, 0.1).to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method tilt does not return Quantity with the right units"
        )
    try:
        qdf.meanvR(1.1, 0.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanvR does not return Quantity with the right units"
        )
    try:
        qdf.meanvT(1.1, 0.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanvT does not return Quantity with the right units"
        )
    try:
        qdf.meanvz(1.1, 0.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanvz does not return Quantity with the right units"
        )
    try:
        qdf.meanjr(1.1, 0.1).to(units.kpc * (units.km / units.s))
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanjr does not return Quantity with the right units"
        )
    try:
        qdf.meanlz(1.1, 0.1).to(units.kpc * (units.km / units.s))
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanlz does not return Quantity with the right units"
        )
    try:
        qdf.meanjz(1.1, 0.1).to(units.kpc * (units.km / units.s))
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method meanjz does not return Quantity with the right units"
        )
    try:
        qdf.sampleV(1.1, 0.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sampleV does not return Quantity with the right units"
        )
    try:
        qdf.sampleV_interpolate(R, z, 0.1, 0.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method sampleV_interpolate does not return Quantity with the right units"
        )
    try:
        qdf.pvR(0.1, 1.1, 0.1).to(1 / (units.km / units.s) / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvR does not return Quantity with the right units"
        )
    try:
        qdf.pvz(0.1, 1.1, 0.1).to(1 / (units.km / units.s) / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvz does not return Quantity with the right units"
        )
    try:
        qdf.pvT(1.1, 1.1, 0.1).to(1 / (units.km / units.s) / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvT does not return Quantity with the right units"
        )
    try:
        qdf.pvRvT(0.1, 1.1, 1.1, 0.1).to(1 / (units.km / units.s) ** 2 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvRvT does not return Quantity with the right units"
        )
    try:
        qdf.pvRvz(0.1, 0.2, 1.1, 0.1).to(1 / (units.km / units.s) ** 2 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvRvz does not return Quantity with the right units"
        )
    try:
        qdf.pvTvz(1.1, 0.2, 1, 1.1, 0.1).to(1 / (units.km / units.s) ** 2 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method pvTvz does not return Quantity with the right units"
        )
    try:
        qdf.vmomentdensity(1.1, 0.2, 0, 0, 0, gl=True).to(1 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        qdf.vmomentdensity(1.1, 0.2, 1, 0, 0, gl=True).to(
            1 / units.pc**3 * (units.km / units.s)
        )
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        qdf.vmomentdensity(1.1, 0.2, 1, 1, 0, gl=True).to(
            1 / units.pc**3 * (units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        qdf.jmomentdensity(1.1, 0.2, 0, 0, 0, gl=True).to(1 / units.pc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method jmomentdensity does not return Quantity with the right units"
        )
    try:
        qdf.jmomentdensity(1.1, 0.2, 1, 0, 0, gl=True).to(
            1 / units.pc**3 * (units.kpc * units.km / units.s)
        )
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method jmomentdensity does not return Quantity with the right units"
        )
    try:
        qdf.jmomentdensity(1.1, 0.2, 1, 1, 0, gl=True).to(
            1 / units.pc**3 * (units.kpc * units.km / units.s) ** 2
        )
    except units.UnitConversionError:
        raise AssertionError(
            "quasiisothermaldf method jmomentdensity does not return Quantity with the right units"
        )
    return None


def test_quasiisothermaldf_method_value():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 9.0, 210.0
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    qdfnou = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aA, cutcounter=True
    )
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    assert (
        numpy.fabs(
            qdf(o).to(1 / units.kpc**3 / (units.km / units.s) ** 3).value
            - qdfnou(o) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hr(1.1).to(units.kpc).value - qdfnou.estimate_hr(1.1) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hz(1.1, 0.1).to(units.kpc).value
            - qdfnou.estimate_hz(1.1, 0.1) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hsr(1.1).to(units.kpc).value - qdfnou.estimate_hsr(1.1) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hsr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hsz(1.1).to(units.kpc).value - qdfnou.estimate_hsz(1.1) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hsz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.surfacemass_z(1.1).to(1 / units.kpc**2).value
            - qdfnou.surfacemass_z(1.1) / ro**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method surfacemass_z does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.density(1.1, 0.1).to(1 / units.kpc**3).value
            - qdfnou.density(1.1, 0.1) / ro**3
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method density does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaR2(1.1, 0.1).to((units.km / units.s) ** 2).value
            - qdfnou.sigmaR2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaR2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaT2(1.1, 0.1).to((units.km / units.s) ** 2).value
            - qdfnou.sigmaT2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaT2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaz2(1.1, 0.1).to((units.km / units.s) ** 2).value
            - qdfnou.sigmaz2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaz2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaRz(1.1, 0.1).to((units.km / units.s) ** 2).value
            - qdfnou.sigmaRz(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaRz does not return correct Quantity"
    assert (
        numpy.fabs(qdf.tilt(1.1, 0.1).to(units.rad).value - qdfnou.tilt(1.1, 0.1))
        < 10.0**-8.0
    ), "quasiisothermaldf method tilt does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvR(1.1, 0.1).to(units.km / units.s).value
            - qdfnou.meanvR(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvR does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvT(1.1, 0.1).to(units.km / units.s).value
            - qdfnou.meanvT(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvz(1.1, 0.1).to(units.km / units.s).value
            - qdfnou.meanvz(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvz does not return correct Quantity"
    # Lower tolerance, because determined through sampling
    assert (
        numpy.fabs(
            qdf.meanjr(1.1, 0.1, nmc=100000).to(units.kpc * units.km / units.s).value
            - qdfnou.meanjr(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 10.0
    ), "quasiisothermaldf method meanjr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanlz(1.1, 0.1, nmc=100000).to(units.kpc * units.km / units.s).value
            - qdfnou.meanlz(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 100.0
    ), "quasiisothermaldf method meanlz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanjz(1.1, 0.1, nmc=100000).to(units.kpc * units.km / units.s).value
            - qdfnou.meanjz(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 10.0
    ), "quasiisothermaldf method meanjz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvR(0.1, 1.1, 0.1).to(1 / units.kpc**3 / (units.km / units.s)).value
            - qdfnou.pvR(0.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvR does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvT(1.1, 1.1, 0.1).to(1 / units.kpc**3 / (units.km / units.s)).value
            - qdfnou.pvT(1.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvz(0.1, 1.1, 0.1).to(1 / units.kpc**3 / (units.km / units.s)).value
            - qdfnou.pvz(0.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvRvT(0.1, 1.1, 1.1, 0.1)
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvRvT(0.1, 1.1, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvRvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvRvz(0.1, 0.2, 1.1, 0.1)
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvRvz(0.1, 0.2, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvRvz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvTvz(1.1, 1.1, 1.1, 0.1)
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvTvz(1.1, 1.1, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvTvz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.vmomentdensity(
                1.1,
                0.1,
                0,
                0,
                0,
                gl=True,
                ro=ro * units.kpc,
                vo=vo * units.km / units.s,
            )
            .to(1 / units.kpc**3 * (units.km / units.s) ** 0)
            .value
            - qdfnou.vmomentdensity(1.1, 0.1, 0, 0, 0, gl=True) / ro**3 * vo**0
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.vmomentdensity(1.1, 0.1, 1, 0, 0, gl=True)
            .to(1 / units.kpc**3 * (units.km / units.s) ** 1)
            .value
            - qdfnou.vmomentdensity(1.1, 0.1, 1, 0, 0, gl=True) / ro**3 * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.vmomentdensity(1.1, 0.1, 0, 1, 1, gl=True)
            .to(1 / units.kpc**3 * (units.km / units.s) ** 2)
            .value
            - qdfnou.vmomentdensity(1.1, 0.1, 0, 1, 1, gl=True) / ro**3 * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.vmomentdensity(1.1, 0.1, 1, 1, 0, gl=True)
            .to(1 / units.kpc**3 * (units.km / units.s) ** 2)
            .value
            - qdfnou.vmomentdensity(1.1, 0.1, 1, 1, 0, gl=True) / ro**3 * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.vmomentdensity(1.1, 0.1, 2, 1, 1, gl=True)
            .to(1 / units.kpc**3 * (units.km / units.s) ** 4)
            .value
            - qdfnou.vmomentdensity(1.1, 0.1, 2, 1, 1, gl=True) / ro**3 * vo**4
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.jmomentdensity(
                1.1,
                0.1,
                0,
                0,
                0,
                nmc=100000,
                ro=ro * units.kpc,
                vo=vo * units.km / units.s,
            )
            .to(1 / units.kpc**3 * (units.kpc * units.km / units.s) ** 0)
            .value
            - qdfnou.jmomentdensity(1.1, 0.1, 0, 0, 0, nmc=100000)
            / ro**3
            * (ro * vo) ** 0
        )
        < 10.0**-4.0
    ), "quasiisothermaldf method jmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.jmomentdensity(1.1, 0.1, 1, 0, 0, nmc=100000)
            .to(1 / units.kpc**3 * (units.kpc * units.km / units.s) ** 1)
            .value
            - qdfnou.jmomentdensity(1.1, 0.1, 1, 0, 0, nmc=100000)
            / ro**3
            * (ro * vo) ** 1
        )
        < 10.0**-2.0
    ), "quasiisothermaldf method jmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.jmomentdensity(1.1, 0.1, 0, 1, 1, nmc=100000)
            .to(1 / units.kpc**3 * (units.kpc * units.km / units.s) ** 2)
            .value
            - qdfnou.jmomentdensity(1.1, 0.1, 0, 1, 1, nmc=100000)
            / ro**3
            * (ro * vo) ** 2
        )
        < 1.0
    ), "quasiisothermaldf method jmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.jmomentdensity(1.1, 0.1, 1, 1, 0, nmc=100000)
            .to(1 / units.kpc**3 * (units.kpc * units.km / units.s) ** 2)
            .value
            - qdfnou.jmomentdensity(1.1, 0.1, 1, 1, 0, nmc=100000)
            / ro**3
            * (ro * vo) ** 2
        )
        < 10.0
    ), "quasiisothermaldf method jmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.jmomentdensity(1.1, 0.1, 2, 1, 1, nmc=100000)
            .to(1 / units.kpc**3 * (units.kpc * units.km / units.s) ** 4)
            .value
            - qdfnou.jmomentdensity(1.1, 0.1, 2, 1, 1, nmc=100000)
            / ro**3
            * (ro * vo) ** 4
        )
        < 10000.0
    ), "quasiisothermaldf method jmomentdensity does not return correct Quantity"
    return None


def test_quasiisothermaldf_sample():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 9.0, 210.0
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    qdfnou = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aA, cutcounter=True
    )
    numpy.random.seed(1)
    vu = qdf.sampleV(1.1, 0.1, n=1).to(units.km / units.s).value / vo
    numpy.random.seed(1)
    vnou = qdfnou.sampleV(1.1, 0.1, n=1)
    assert numpy.all(
        numpy.fabs(vu - vnou) < 10.0**-8.0
    ), "quasiisothermaldf sampleV does not return correct Quantity"
    # Also when giving vo with units itself
    numpy.random.seed(1)
    vu = (
        qdf.sampleV(1.1, 0.1, n=1, vo=vo * units.km / units.s)
        .to(units.km / units.s)
        .value
        / vo
    )
    numpy.random.seed(1)
    vnou = qdfnou.sampleV(1.1, 0.1, n=1)
    assert numpy.all(
        numpy.fabs(vu - vnou) < 10.0**-8.0
    ), "quasiisothermaldf sampleV does not return correct Quantity"
    return None


def test_quasiisothermaldf_interpolate_sample():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 9.0, 210.0
    R = numpy.array([0.6, 0.7, 0.8, 0.9, 1.0])
    z = numpy.array([0.0, 0.1, 0.2, 0.3, 0.4])
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    qdfnou = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aA, cutcounter=True
    )
    numpy.random.seed(1)
    vu = qdf.sampleV_interpolate(R, z, 0.1, 0.1).to(units.km / units.s).value / vo
    numpy.random.seed(1)
    vnou = qdfnou.sampleV_interpolate(R, z, 0.1, 0.1)
    assert numpy.all(
        numpy.fabs(vu - vnou) < 10.0**-8.0
    ), "quasiisothermaldf sampleV_interpolate does not return correct Quantity"
    # Also when giving vo with units itself
    numpy.random.seed(1)
    vu = (
        qdf.sampleV_interpolate(R, z, 0.1, 0.1, vo=vo * units.km / units.s)
        .to(units.km / units.s)
        .value
        / vo
    )
    numpy.random.seed(1)
    vnou = qdfnou.sampleV_interpolate(R, z, 0.1, 0.1)
    assert numpy.all(
        numpy.fabs(vu - vnou) < 10.0**-8.0
    ), "quasiisothermaldf sampleV_interpolate does not return correct Quantity"
    return None


def test_quasiisothermaldf_method_inputAsQuantity():
    # Those that use the decorator
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 9.0, 210.0
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    qdfnou = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aA, cutcounter=True
    )
    assert (
        numpy.fabs(
            qdf.estimate_hr(1.1 * ro * units.kpc, z=100.0 * units.pc, dR=1.0 * units.pc)
            .to(units.kpc)
            .value
            - qdfnou.estimate_hr(1.1, 0.1 / ro, dR=10.0**-3.0 / ro) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hz(
                1.1 * ro * units.kpc, 0.1 * ro * units.kpc, dz=1.0 * units.pc
            )
            .to(units.kpc)
            .value
            - qdfnou.estimate_hz(1.1, 0.1, dz=10.0**-3.0 / ro) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hsr(
                1.1 * ro * units.kpc, z=100.0 * units.pc, dR=1.0 * units.pc
            )
            .to(units.kpc)
            .value
            - qdfnou.estimate_hsr(1.1, 0.1 / ro, dR=10.0**-3.0 / ro) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hsr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.estimate_hsz(
                1.1 * ro * units.kpc, z=100.0 * units.pc, dR=1.0 * units.pc
            )
            .to(units.kpc)
            .value
            - qdfnou.estimate_hsz(1.1, 0.1 / ro, dR=10.0**-3.0 / ro) * ro
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method estimate_hsz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.surfacemass_z(1.1 * ro * units.kpc, zmax=2.0 * units.kpc)
            .to(1 / units.kpc**2)
            .value
            - qdfnou.surfacemass_z(1.1, zmax=2.0 / ro) / ro**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method surfacemass_z does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.density(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(1 / units.kpc**3)
            .value
            - qdfnou.density(1.1, 0.1) / ro**3
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method density does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaR2(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to((units.km / units.s) ** 2)
            .value
            - qdfnou.sigmaR2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaR2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaT2(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to((units.km / units.s) ** 2)
            .value
            - qdfnou.sigmaT2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaT2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaz2(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to((units.km / units.s) ** 2)
            .value
            - qdfnou.sigmaz2(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaz2 does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.sigmaRz(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to((units.km / units.s) ** 2)
            .value
            - qdfnou.sigmaRz(1.1, 0.1) * vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method sigmaRz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.tilt(1.1 * ro * units.kpc, 0.1 * ro * units.kpc).to(units.rad).value
            - qdfnou.tilt(1.1, 0.1)
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method tilt does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvR(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(units.km / units.s)
            .value
            - qdfnou.meanvR(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvR does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvT(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(units.km / units.s)
            .value
            - qdfnou.meanvT(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanvz(1.1 * ro * units.kpc, 0.1 * ro * units.kpc)
            .to(units.km / units.s)
            .value
            - qdfnou.meanvz(1.1, 0.1) * vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method meanvz does not return correct Quantity"
    # Lower tolerance, because determined through sampling
    assert (
        numpy.fabs(
            qdf.meanjr(1.1 * ro * units.kpc, 0.1 * ro * units.kpc, nmc=100000)
            .to(units.kpc * units.km / units.s)
            .value
            - qdfnou.meanjr(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 10.0
    ), "quasiisothermaldf method meanjr does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanlz(1.1 * ro * units.kpc, 0.1 * ro * units.kpc, nmc=100000)
            .to(units.kpc * units.km / units.s)
            .value
            - qdfnou.meanlz(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 100.0
    ), "quasiisothermaldf method meanlz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.meanjz(1.1 * ro * units.kpc, 0.1 * ro * units.kpc, nmc=100000)
            .to(units.kpc * units.km / units.s)
            .value
            - qdfnou.meanjz(1.1, 0.1, nmc=100000) * ro * vo
        )
        < 10.0
    ), "quasiisothermaldf method meanjz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvR(
                0.1 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s))
            .value
            - qdfnou.pvR(0.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvR does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvT(
                1.1 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s))
            .value
            - qdfnou.pvT(1.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvz(
                0.1 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s))
            .value
            - qdfnou.pvz(0.1, 1.1, 0.1) / ro**3 / vo
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvRvT(
                0.1 * vo * units.km / units.s,
                1.1 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvRvT(0.1, 1.1, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvRvT does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvRvz(
                0.1 * vo * units.km / units.s,
                0.2 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvRvz(0.1, 0.2, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvRvz does not return correct Quantity"
    assert (
        numpy.fabs(
            qdf.pvTvz(
                1.1 * vo * units.km / units.s,
                0.1 * vo * units.km / units.s,
                1.1 * ro * units.kpc,
                0.1 * ro * units.kpc,
            )
            .to(1 / units.kpc**3 / (units.km / units.s) ** 2)
            .value
            - qdfnou.pvTvz(1.1, 0.1, 1.1, 0.1) / ro**3 / vo**2
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method pvTvz does not return correct Quantity"
    return None


def test_quasiisothermaldf_method_inputAsQuantity_special():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 9.0, 210.0
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    qdfnou = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential, aA=aA, cutcounter=True
    )
    assert (
        numpy.fabs(
            qdf(
                (
                    0.05 * ro * vo * units.kpc * units.km / units.s,
                    1.1 * ro * vo * units.kpc * units.km / units.s,
                    0.025 * ro * vo * units.kpc * units.km / units.s,
                )
            )
            .to(1 / units.kpc**3 / (units.km / units.s) ** 3)
            .value
            - qdfnou((0.05, 1.1, 0.025)) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "quasiisothermaldf method __call__ does not return correct Quantity"
    return None


def test_quasiisothermaldf_setup_roAsQuantity():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro = 9.0
    df = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro * units.kpc,
    )
    assert (
        numpy.fabs(df._ro - ro) < 10.0**-10.0
    ), "ro in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_quasiisothermaldf_setup_roAsQuantity_oddunits():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro = 9000.0
    df = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro * units.lyr,
    )
    assert (
        numpy.fabs(df._ro - ro * (units.lyr).to(units.kpc)) < 10.0**-10.0
    ), "ro in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_quasiisothermaldf_setup_voAsQuantity():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    vo = 230.0
    df = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        vo=vo * units.km / units.s,
    )
    assert (
        numpy.fabs(df._vo - vo) < 10.0**-10.0
    ), "vo in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_quasiisothermaldf_setup_voAsQuantity_oddunits():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    vo = 230.0
    df = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        vo=vo * units.pc / units.Myr,
    )
    assert (
        numpy.fabs(df._vo - vo * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_test_quasiisothermaldf_setup_profileAsQuantity():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 7.0, 250.0
    qdf = quasiisothermaldf(
        3.0 * units.kpc,
        30.0 * units.km / units.s,
        20.0 * units.pc / units.Myr,
        10.0 * units.kpc,
        8000.0 * units.lyr,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        ro=ro,
        vo=vo,
    )
    assert (
        numpy.fabs(qdf._hr - 3.0 / ro) < 10.0**-10.0
    ), "hr in quasiisothermaldf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(qdf._sr - 30.0 / vo) < 10.0**-10.0
    ), "sr in quasiisothermaldf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(qdf._sz - 20.0 * (units.pc / units.Myr).to(units.km / units.s) / vo)
        < 10.0**-10.0
    ), "sz in quasiisothermaldf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(qdf._hsr - 10.0 / ro) < 10.0**-10.0
    ), "hr in quasiisothermaldf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(qdf._hsz - 8000.0 * (units.lyr).to(units.kpc) / ro) < 10.0**-10.0
    ), "hsz in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_test_quasiisothermaldf_setup_refrloAsQuantity():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential

    aA = actionAngleAdiabatic(pot=MWPotential, c=True)
    ro, vo = 7.0, 250.0
    qdf = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=MWPotential,
        aA=aA,
        cutcounter=True,
        refr=9.0 * units.kpc,
        lo=10.0 * units.kpc * units.km / units.s,
        ro=ro,
        vo=vo,
    )
    assert (
        numpy.fabs(qdf._refr - 9.0 / ro) < 10.0**-10.0
    ), "refr in quasiisothermaldf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(qdf._lo - 10.0 / vo / ro) < 10.0**-10.0
    ), "lo in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_sphericaldf_method_returntype():
    from galpy import potential
    from galpy.df import constantbetaHernquistdf, isotropicHernquistdf
    from galpy.orbit import Orbit

    pot = potential.HernquistPotential(amp=2.0, a=1.3, ro=8.0, vo=220.0)
    dfh = isotropicHernquistdf(pot=pot)
    dfa = constantbetaHernquistdf(pot=pot, beta=-0.2)
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    assert isinstance(
        dfh(o), units.Quantity
    ), "sphericaldf method __call__ does not return Quantity when it should"
    assert isinstance(
        dfh((o.E(pot=pot),)), units.Quantity
    ), "sphericaldf method __call__ does not return Quantity when it should"
    assert isinstance(
        dfh(o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()), units.Quantity
    ), "sphericaldf method __call__ does not return Quantity when it should"
    assert isinstance(
        dfh.dMdE(o.E(pot=pot)), units.Quantity
    ), "sphericaldf method dMdE does not return Quantity when it should"
    assert isinstance(
        dfh.vmomentdensity(1.1, 0, 0), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfa.vmomentdensity(1.1, 0, 0), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfh.vmomentdensity(1.1, 1, 0), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfa.vmomentdensity(1.1, 1, 0), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfh.vmomentdensity(1.1, 0, 2), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfa.vmomentdensity(1.1, 0, 2), units.Quantity
    ), "sphericaldf method vmomentdensity does not return Quantity when it should"
    assert isinstance(
        dfh.sigmar(1.1), units.Quantity
    ), "sphericaldf method sigmar does not return Quantity when it should"
    assert isinstance(
        dfh.sigmat(1.1), units.Quantity
    ), "sphericaldf method sigmar does not return Quantity when it should"
    # beta should not be a quantity
    assert not isinstance(
        dfh.beta(1.1), units.Quantity
    ), "sphericaldf method beta returns Quantity when it shouldn't"
    return None


def test_sphericaldf_method_returnunit():
    from galpy import potential
    from galpy.df import constantbetaHernquistdf, isotropicHernquistdf
    from galpy.orbit import Orbit

    pot = potential.HernquistPotential(amp=2.0, a=1.3, ro=8.0, vo=220.0)
    dfh = isotropicHernquistdf(pot=pot)
    dfa = constantbetaHernquistdf(pot=pot, beta=-0.2)
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    try:
        dfh(o).to(1 / units.kpc**3 / (units.km / units.s) ** 3)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method __call__ does not return Quantity with the right units"
        )
    try:
        dfh((o.E(pot=pot),)).to(1 / units.kpc**3 / (units.km / units.s) ** 3)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method __call__ does not return Quantity with the right units"
        )
    try:
        dfh(o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()).to(
            1 / units.kpc**3 / (units.km / units.s) ** 3
        )
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method __call__ does not return Quantity with the right units"
        )
    try:
        dfh.dMdE(o.E(pot=pot)).to(1 / (units.km / units.s) ** 2)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method dMdE does not return Quantity with the right units"
        )
    try:
        dfh.vmomentdensity(1.1, 0, 0).to(1 / units.kpc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfa.vmomentdensity(1.1, 0, 0).to(1 / units.kpc**3)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfh.vmomentdensity(1.1, 1, 0).to(1 / units.kpc**3 * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfa.vmomentdensity(1.1, 1, 0).to(1 / units.kpc**3 * units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfh.vmomentdensity(1.1, 0, 2).to(1 / units.kpc**3 * units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfa.vmomentdensity(1.1, 0, 2).to(1 / units.kpc**3 * units.km**2 / units.s**2)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method vmomentdensity does not return Quantity with the right units"
        )
    try:
        dfh.sigmar(1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method sigmar does not return Quantity with the right units"
        )
    try:
        dfh.sigmat(1.1).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "sphericaldf method sigmar does not return Quantity with the right units"
        )
    return None


def test_sphericaldf_method_value():
    from galpy import potential
    from galpy.df import constantbetaHernquistdf, isotropicHernquistdf
    from galpy.orbit import Orbit

    ro, vo = 8.0, 220.0
    pot = potential.HernquistPotential(amp=2.0, a=1.3)
    dfh = isotropicHernquistdf(pot=pot, ro=ro, vo=vo)
    dfh_nou = isotropicHernquistdf(pot=pot)
    dfa = constantbetaHernquistdf(pot=pot, beta=-0.2, ro=ro, vo=vo)
    dfa_nou = constantbetaHernquistdf(pot=pot, beta=-0.2)
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4])
    assert (
        numpy.fabs(
            dfh(o).to(1 / units.kpc**3 / (units.km / units.s) ** 3).value
            - dfh_nou(o) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "sphericaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh((o.E(pot=pot),)).to(1 / units.kpc**3 / (units.km / units.s) ** 3).value
            - dfh_nou((o.E(pot=pot),)) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "sphericaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh(o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi())
            .to(1 / units.kpc**3 / (units.km / units.s) ** 3)
            .value
            - dfh_nou(o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi()) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "sphericaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.dMdE(o.E(pot=pot)).to(1 / (units.km / units.s) ** 2).value
            - dfh_nou.dMdE(o.E(pot=pot)) / vo**2
        )
        < 10.0**-8.0
    ), "sphericaldf method dMdE does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1, 0, 0).to(1 / units.kpc**3).value
            - dfh_nou.vmomentdensity(1.1, 0, 0) / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1, 0, 0).to(1 / units.kpc**3).value
            - dfa_nou.vmomentdensity(1.1, 0, 0) / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1, 1, 0)
            .to(1 / units.kpc**3 * units.km / units.s)
            .value
            - dfh_nou.vmomentdensity(1.1, 1, 0) * vo / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1, 1, 0)
            .to(1 / units.kpc**3 * units.km / units.s)
            .value
            - dfa_nou.vmomentdensity(1.1, 1, 0) * vo / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    # One with no quantity output
    import galpy.util._optional_deps

    galpy.util._optional_deps._APY_UNITS = False  # Hack
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1, 0, 2)
            - dfh_nou.vmomentdensity(1.1, 0, 2) * vo**2 / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    galpy.util._optional_deps._APY_UNITS = True  # Hack
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1, 0, 2)
            .to(1 / units.kpc**3 * units.km**2 / units.s**2)
            .value
            - dfh_nou.vmomentdensity(1.1, 0, 2) * vo**2 / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1, 0, 2)
            .to(1 / units.kpc**3 * units.km**2 / units.s**2)
            .value
            - dfa_nou.vmomentdensity(1.1, 0, 2) * vo**2 / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.sigmar(1.1).to(units.km / units.s).value - dfh_nou.sigmar(1.1) * vo
        )
        < 10.0**-8.0
    ), "sphericaldf method sigmar does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.sigmat(1.1).to(units.km / units.s).value - dfh_nou.sigmat(1.1) * vo
        )
        < 10.0**-8.0
    ), "sphericaldf method sigmat does not return correct Quantity"
    return None


def test_sphericaldf_method_inputAsQuantity():
    from galpy import potential
    from galpy.df import constantbetaHernquistdf, isotropicHernquistdf
    from galpy.orbit import Orbit

    ro, vo = 8.0, 220.0
    pot = potential.HernquistPotential(amp=2.0, a=1.3)
    dfh = isotropicHernquistdf(pot=pot, ro=ro, vo=vo)
    dfh_nou = isotropicHernquistdf(pot=pot)
    dfa = constantbetaHernquistdf(pot=pot, beta=-0.2, ro=ro, vo=vo)
    dfa_nou = constantbetaHernquistdf(pot=pot, beta=-0.2)
    o = Orbit([1.1, 0.1, 1.1, 0.1, 0.03, 0.4], ro=ro, vo=vo)
    assert (
        numpy.fabs(
            dfh((o.E(pot=pot),)).to(1 / units.kpc**3 / (units.km / units.s) ** 3).value
            - dfh_nou((o.E(pot=pot, use_physical=False),)) / ro**3 / vo**3
        )
        < 10.0**-8.0
    ), "sphericaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh(o.R(), o.vR(), o.vT(), o.z(), o.vz(), o.phi())
            .to(1 / units.kpc**3 / (units.km / units.s) ** 3)
            .value
            - dfh_nou(
                o.R(use_physical=False),
                o.vR(use_physical=False),
                o.vT(use_physical=False),
                o.z(use_physical=False),
                o.vz(use_physical=False),
                o.phi(use_physical=False),
            )
            / ro**3
            / vo**3
        )
        < 10.0**-8.0
    ), "sphericaldf method __call__ does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.dMdE(o.E(pot=pot)).to(1 / (units.km / units.s) ** 2).value
            - dfh_nou.dMdE(o.E(pot=pot, use_physical=False)) / vo**2
        )
        < 10.0**-8.0
    ), "sphericaldf method dMdE does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1 * ro * units.kpc, 0, 0).to(1 / units.kpc**3).value
            - dfh_nou.vmomentdensity(1.1, 0, 0) / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1 * ro * units.kpc, 0, 0, ro=ro * units.kpc)
            .to(1 / units.kpc**3)
            .value
            - dfa_nou.vmomentdensity(1.1, 0, 0) / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.vmomentdensity(
                1.1 * ro * units.kpc, 1, 0, ro=ro, vo=vo * units.km / units.s
            )
            .to(1 / units.kpc**3 * units.km / units.s)
            .value
            - dfh_nou.vmomentdensity(1.1, 1, 0) * vo / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1 * ro * units.kpc, 1, 0, vo=vo * units.km / units.s)
            .to(1 / units.kpc**3 * units.km / units.s)
            .value
            - dfa_nou.vmomentdensity(1.1, 1, 0) * vo / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.vmomentdensity(1.1 * ro * units.kpc, 0, 2)
            .to(1 / units.kpc**3 * units.km**2 / units.s**2)
            .value
            - dfh_nou.vmomentdensity(1.1, 0, 2) * vo**2 / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfa.vmomentdensity(1.1 * ro * units.kpc, 0, 2)
            .to(1 / units.kpc**3 * units.km**2 / units.s**2)
            .value
            - dfa_nou.vmomentdensity(1.1, 0, 2) * vo**2 / ro**3
        )
        < 10.0**-8.0
    ), "sphericaldf method vmomentdensity does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.sigmar(1.1 * ro * units.kpc).to(units.km / units.s).value
            - dfh_nou.sigmar(1.1) * vo
        )
        < 10.0**-8.0
    ), "sphericaldf method sigmar does not return correct Quantity"
    assert (
        numpy.fabs(
            dfh.sigmat(1.1 * ro * units.kpc).to(units.km / units.s).value
            - dfh_nou.sigmat(1.1) * vo
        )
        < 10.0**-8.0
    ), "sphericaldf method sigmat does not return correct Quantity"
    return None


def test_sphericaldf_sample():
    from galpy import potential
    from galpy.df import isotropicHernquistdf
    from galpy.orbit import Orbit

    ro, vo = 8.0, 220.0
    pot = potential.HernquistPotential(amp=2.0, a=1.3)
    dfh = isotropicHernquistdf(pot=pot, ro=ro, vo=vo)
    numpy.random.seed(10)
    sam = dfh.sample(R=1.0 * units.kpc, z=0.0 * units.kpc, phi=10.0 * units.deg, n=2)
    numpy.random.seed(10)
    sam_nou = dfh.sample(R=1.0 / ro, z=0.0 / ro, phi=10.0 / 180.0 * numpy.pi, n=2)
    assert numpy.all(
        numpy.fabs(sam.r(use_physical=False) - sam_nou.r(use_physical=False)) < 1e-8
    ), "Sample returned by sphericaldf.sample with input R,z,phi with units does not agree with that returned by sampline with input R,z,phi without units"
    assert numpy.all(
        numpy.fabs(sam.vr(use_physical=False) - sam_nou.vr(use_physical=False)) < 1e-8
    ), "Sample returned by sphericaldf.sample with input R,z,phi with units does not agree with that returned by sampline with input R,z,phi without units"
    # Array input
    arr = numpy.array([1.0, 2.0])
    numpy.random.seed(10)
    sam = dfh.sample(
        R=arr * units.kpc,
        z=arr * 0.0 * units.kpc,
        phi=arr * 10.0 * units.deg,
        n=len(arr),
    )
    numpy.random.seed(10)
    sam_nou = dfh.sample(
        R=arr / ro, z=arr * 0.0 / ro, phi=arr * 10.0 / 180.0 * numpy.pi, n=len(arr)
    )
    assert numpy.all(
        numpy.fabs(sam.r(use_physical=False) - sam_nou.r(use_physical=False)) < 1e-8
    ), "Sample returned by sphericaldf.sample with input R,z,phi with units does not agree with that returned by sampline with input R,z,phi without units"
    assert numpy.all(
        numpy.fabs(sam.vr(use_physical=False) - sam_nou.vr(use_physical=False)) < 1e-8
    ), "Sample returned by sphericaldf.sample with input R,z,phi with units does not agree with that returned by sampline with input R,z,phi without units"
    # rmin
    numpy.random.seed(10)
    sam = dfh.sample(n=2, rmin=1.1 * units.kpc)
    numpy.random.seed(10)
    sam_nou = dfh.sample(n=2, rmin=1.1 / ro)
    assert numpy.all(
        numpy.fabs(sam.r(use_physical=False) - sam_nou.r(use_physical=False)) < 1e-8
    ), "Sample returned by sphericaldf.sample with input rmin with units does not agree with that returned by sampline with input rmin without units"
    return None


def test_sphericaldf_sample_outputunits():
    from galpy import potential
    from galpy.df import isotropicHernquistdf

    ro, vo = 8.0, 220.0
    pot = potential.HernquistPotential(amp=2.0, a=1.3)
    dfh = isotropicHernquistdf(pot=pot, ro=ro, vo=vo)
    dfh_nou = isotropicHernquistdf(pot=pot)
    numpy.random.seed(10)
    sam = dfh.sample(
        R=1.0 * units.kpc,
        z=0.0 * units.kpc,
        phi=10.0 * units.deg,
        n=2,
        return_orbit=False,
    )
    numpy.random.seed(10)
    sam_nou = dfh_nou.sample(
        R=1.0 / ro, z=0.0 / ro, phi=10.0 / 180.0 * numpy.pi, n=2, return_orbit=False
    )
    assert numpy.all(
        numpy.fabs(sam[0].to_value(units.kpc) / ro - sam_nou[0]) < 1e-8
    ), "Sample returned by sphericaldf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[1].to_value(units.km / units.s) / vo - sam_nou[1]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[2].to_value(units.km / units.s) / vo - sam_nou[2]) < 1e-8
    ), "Sample returned by sphericaldf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[3].to_value(units.kpc) / ro - sam_nou[3]) < 1e-8
    ), "Sample returned by sphericaldf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[4].to_value(units.km / units.s) / vo - sam_nou[4]) < 1e-8
    ), "Sample returned by sphericaldf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[5].to_value(units.rad) - sam_nou[5]) < 1e-8
    ), "Sample returned by sphericaldf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    return None


def test_kingdf_setup_wunits():
    from galpy.df import kingdf
    from galpy.util import conversion

    ro, vo = 9.0, 210.0
    dfk = kingdf(W0=3.0, M=4 * 1e4 * units.Msun, rt=10.0 * units.pc, ro=ro, vo=vo)
    dfk_nou = kingdf(
        W0=3.0,
        M=4 * 1e4 / conversion.mass_in_msol(vo, ro),
        rt=10.0 / ro / 1000,
        ro=ro,
        vo=vo,
    )
    assert (
        numpy.fabs(
            dfk.sigmar(1.0 * units.pc, use_physical=False)
            - dfk_nou.sigmar(1.0 * units.pc, use_physical=False)
        )
        < 1e-8
    ), "kingdf set up with parameters with units does not agree with kingdf not set up with parameters with units"
    return None


def test_streamdf_method_returntype():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro,
        vo=vo,
        nosetup=True,
    )
    assert isinstance(
        sdf_bovy14.misalignment(), units.Quantity
    ), "streamdf method misalignment does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.estimateTdisrupt(0.1), units.Quantity
    ), "streamdf method estimateTdisrupt does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.meanOmega(0.1), units.Quantity
    ), "streamdf method meanOmega does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.sigOmega(0.1), units.Quantity
    ), "streamdf method sigOmega does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.meantdAngle(0.1), units.Quantity
    ), "streamdf method meantdAngle does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.sigtdAngle(0.1), units.Quantity
    ), "streamdf method sigtdAngle does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.meanangledAngle(0.1), units.Quantity
    ), "streamdf method meanangledAngle does not return Quantity when it should"
    assert isinstance(
        sdf_bovy14.sigangledAngle(0.1), units.Quantity
    ), "streamdf method sigangledAngle does not return Quantity when it should"
    return None


def test_streamdf_method_returnunit():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro,
        vo=vo,
        nosetup=True,
    )
    try:
        sdf_bovy14.misalignment().to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method misalignment does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.estimateTdisrupt(0.1).to(units.Myr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method estimateTdisrupt does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.meanOmega(0.1).to(1 / units.Myr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method meanOmega does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.sigOmega(0.1).to(1 / units.Myr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method sigOmega does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.meantdAngle(0.1).to(units.Myr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method meantdAngle does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.sigtdAngle(0.1).to(units.Myr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method sigtdAngle does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.meanangledAngle(0.1).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method meanangledAngle does not return Quantity with the right units"
        )
    try:
        sdf_bovy14.sigangledAngle(0.1).to(units.rad)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method sigangledAngle does not return Quantity with the right units"
        )
    return None


def test_streamdf_method_value():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro,
        vo=vo,
        nosetup=True,
    )
    sdf_bovy14_nou = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nosetup=True,
    )
    assert (
        numpy.fabs(
            sdf_bovy14.misalignment().to(units.rad).value
            - sdf_bovy14_nou.misalignment()
        )
        < _NUMPY_1_22 * 1e-7 + (1 - _NUMPY_1_22) * 1e-8
    ), "streamdf method misalignment does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.estimateTdisrupt(0.1).to(units.Gyr).value
            - sdf_bovy14_nou.estimateTdisrupt(0.1) * conversion.time_in_Gyr(vo, ro)
        )
        < _NUMPY_1_22 * 1e-7 + (1 - _NUMPY_1_22) * 1e-8
    ), "streamdf method estimateTdisrupt does not return correct Quantity"
    assert numpy.all(
        numpy.fabs(
            sdf_bovy14.meanOmega(0.1).to(1 / units.Gyr).value
            - sdf_bovy14_nou.meanOmega(0.1) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "streamdf method meanOmega does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.sigOmega(0.1).to(1 / units.Gyr).value
            - sdf_bovy14_nou.sigOmega(0.1) * conversion.freq_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "streamdf method sigOmega does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.meantdAngle(0.1).to(units.Gyr).value
            - sdf_bovy14_nou.meantdAngle(0.1) * conversion.time_in_Gyr(vo, ro)
        )
        < 10.0**-7.0
    ), "streamdf method meantdAngle does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.sigtdAngle(0.1).to(units.Gyr).value
            - sdf_bovy14_nou.sigtdAngle(0.1) * conversion.time_in_Gyr(vo, ro)
        )
        < 10.0**-8.0
    ), "streamdf method sigtdAngle does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.meanangledAngle(0.1).to(units.rad).value
            - sdf_bovy14_nou.meanangledAngle(0.1)
        )
        < 10.0**-8.0
    ), "streamdf method meanangledAngle does not return correct Quantity"
    assert (
        numpy.fabs(
            sdf_bovy14.sigangledAngle(0.1).to(units.rad).value
            - sdf_bovy14_nou.sigangledAngle(0.1)
        )
        < 10.0**-8.0
    ), "streamdf method sigangledAngle does not return correct Quantity"
    return None


def test_streamdf_method_inputAsQuantity():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro,
        vo=vo,
        nosetup=True,
    )
    sdf_bovy14_nou = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nosetup=True,
    )
    assert (
        numpy.fabs(
            sdf_bovy14.subhalo_encounters(
                venc=200.0 * units.km / units.s,
                sigma=150.0 * units.km / units.s,
                nsubhalo=38.35 / (4.0 * (25.0 * units.kpc) ** 3.0 * numpy.pi / 3.0),
                bmax=1.0 * units.kpc,
                yoon=False,
            )
            - sdf_bovy14_nou.subhalo_encounters(
                venc=200.0 / vo,
                sigma=150.0 / vo,
                nsubhalo=38.35 / (4.0 * 25.0**3.0 * numpy.pi / 3.0) * ro**3.0,
                bmax=1.0 / ro,
                yoon=False,
            )
        )
        < 1e-6 * _NUMPY_1_22 + 1e-8 * (1 - _NUMPY_1_22)
    ), "streamdf method subhalo_encounters with Quantity input does not return correct Quantity"
    assert numpy.fabs(
        sdf_bovy14.pOparapar(0.2 / units.Gyr, 30.0 * units.deg)
        - sdf_bovy14_nou.pOparapar(
            0.2 / conversion.freq_in_Gyr(vo, ro), 30.0 * numpy.pi / 180.0
        )
    ) < 1e-5 * _NUMPY_1_22 + 1e-8 * (
        1 - _NUMPY_1_22
    ), "streamdf method pOparapar with Quantity input does not return correct Quantity"
    return None


def test_streamdf_sample():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    sdf_bovy14 = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro,
        vo=vo,
        nosetup=True,
    )
    sdf_bovy14_nou = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        nosetup=True,
    )
    # aa
    numpy.random.seed(1)
    acfsdt = sdf_bovy14.sample(1, returnaAdt=True)
    numpy.random.seed(1)
    acfsdtnou = sdf_bovy14_nou.sample(1, returnaAdt=True)
    assert numpy.all(
        numpy.fabs(
            acfsdt[0].to(1 / units.Gyr).value / conversion.freq_in_Gyr(vo, ro)
            - acfsdtnou[0]
        )
        < 10.0**-8.0
    ), "streamdf sample returnaAdt does not return correct Quantity"
    assert numpy.all(
        numpy.fabs(acfsdt[1].to(units.rad).value - acfsdtnou[1]) < 10.0**-8.0
    ), "streamdf sample returnaAdt does not return correct Quantity"
    assert numpy.all(
        numpy.fabs(
            acfsdt[2].to(units.Gyr).value / conversion.time_in_Gyr(vo, ro)
            - acfsdtnou[2]
        )
        < 10.0**-8.0
    ), "streamdf sample returnaAdt does not return correct Quantity"
    # Test others as part of streamgapdf
    return None


def test_streamdf_setup_roAsQuantity():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro = 9.0
    df = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro * units.kpc,
        nosetup=True,
    )
    assert (
        numpy.fabs(df._ro - ro) < 10.0**-10.0
    ), "ro in streamdf setup as Quantity does not work as expected"
    return None


def test_streamdf_setup_roAsQuantity_oddunits():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro = 9000.0
    df = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        ro=ro * units.lyr,
        nosetup=True,
    )
    assert (
        numpy.fabs(df._ro - ro * (units.lyr).to(units.kpc)) < 10.0**-10.0
    ), "ro in quasiisothermaldf setup as Quantity does not work as expected"
    return None


def test_streamdf_setup_voAsQuantity():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    vo = 250.0
    df = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        vo=vo * units.km / units.s,
        nosetup=True,
    )
    assert (
        numpy.fabs(df._vo - vo) < 10.0**-10.0
    ), "vo in streamdf setup as Quantity does not work as expected"
    return None


def test_streamdf_setup_voAsQuantity_oddunits():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    vo = 250.0
    df = streamdf(
        sigv / 220.0,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
        vo=vo * units.pc / units.Myr,
        nosetup=True,
    )
    assert (
        numpy.fabs(df._vo - vo * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vo in streamdf setup as Quantity does not work as expected"
    return None


def test_streamdf_setup_paramsAsQuantity():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365 * units.km / units.s
    ro, vo = 9.0, 230.0
    df = streamdf(
        sigv,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 * units.Gyr,
        ro=ro,
        vo=vo,
        sigangle=0.01 * units.deg,
        deltaAngleTrack=170.0 * units.deg,
        nosetup=True,
    )
    assert (
        numpy.fabs(df._sigv - 0.365 / vo) < 10.0**-10.0
    ), "sigv in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._tdisrupt - 4.5 / conversion.time_in_Gyr(vo, ro)) < 10.0**-10.0
    ), "tdisrupt in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._sigangle - 0.01 * (units.deg).to(units.rad)) < 10.0**-10.0
    ), "sigangle in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._deltaAngleTrack - 170.0 * (units.deg).to(units.rad))
        < 10.0**-10.0
    ), "deltaAngleTrack in streamdf setup as Quantity does not work as expected"
    return None


def test_streamdf_setup_coordtransformparamsAsQuantity():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 230.0
    df = streamdf(
        sigv / vo,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
        nosetup=True,
        R0=8.0 * units.kpc,
        Zsun=25.0 * units.pc,
        vsun=units.Quantity(
            [
                -10.0 * units.km / units.s,
                240.0 * units.pc / units.Myr,
                7.0 * units.km / units.s,
            ]
        ),
    )
    assert (
        numpy.fabs(df._R0 - 8.0) < 10.0**-10.0
    ), "R0 in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._Zsun - 0.025) < 10.0**-10.0
    ), "Zsun in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._vsun[0] + 10.0) < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._vsun[1] - 240.0 * (units.pc / units.Myr).to(units.km / units.s))
        < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._vsun[2] - 7.0) < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    # Now with vsun as Quantity
    df = streamdf(
        sigv / vo,
        progenitor=obs,
        pot=lp,
        aA=aAI,
        leading=True,
        nTrackChunks=11,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
        nosetup=True,
        R0=8.0 * units.kpc,
        Zsun=25.0 * units.pc,
        vsun=units.Quantity([-10.0, 240.0, 7.0], unit=units.km / units.s),
    )
    assert (
        numpy.fabs(df._vsun[0] + 10.0) < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._vsun[1] - 240.0) < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    assert (
        numpy.fabs(df._vsun[2] - 7.0) < 10.0**-10.0
    ), "vsun in streamdf setup as Quantity does not work as expected"
    return None


def test_streamdf_RnormWarning():
    import warnings

    from galpy.actionAngle import actionAngleIsochroneApprox

    # Imports
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions
    from galpy.util import galpyWarning

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            Rnorm=ro,
            nosetup=True,
        )
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "WARNING: Rnorm keyword input to streamdf is deprecated in favor of the standard ro keyword"
            )
            if raisedWarning:
                break
        assert raisedWarning, "Rnorm warning not raised when it should have been"
    return None


def test_streamdf_VnormWarning():
    import warnings

    from galpy.actionAngle import actionAngleIsochroneApprox

    # Imports
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions
    from galpy.util import galpyWarning

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    ro, vo = 9.0, 250.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", galpyWarning)
        sdf_bovy14 = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=4.5 / conversion.time_in_Gyr(220.0, 8.0),
            Vnorm=vo,
            nosetup=True,
        )
        raisedWarning = False
        for wa in w:
            raisedWarning = (
                str(wa.message)
                == "WARNING: Vnorm keyword input to streamdf is deprecated in favor of the standard vo keyword"
            )
            if raisedWarning:
                break
        assert raisedWarning, "Vnorm warning not raised when it should have been"
    return None


def test_streamgapdf_method_returntype():
    # Imports
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8)
    prog_unp_peri = Orbit(
        [
            2.6556151742081835,
            0.2183747276300308,
            0.67876510797240575,
            -2.0143395648974671,
            -0.3273737682604374,
            0.24218273922966019,
        ]
    )
    global sdf_sanders15, sdf_sanders15_nou
    V0, R0 = 220.0, 8.0
    sigv = 0.365 * (10.0 / 2.0) ** (1.0 / 3.0) * units.km / units.s
    # bare-bones setup, only interested in testing consistency between units
    # and no units
    sdf_sanders15 = streamgapdf(
        sigv,
        progenitor=prog_unp_peri,
        pot=lp,
        aA=aAI,
        leading=False,
        nTrackChunks=5,
        nTrackIterations=1,
        nTrackChunksImpact=5,
        sigMeanOffset=4.5,
        tdisrupt=10.88 * units.Gyr,
        Vnorm=V0,
        Rnorm=R0,
        impactb=0.1 * units.kpc,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464])
        * units.km
        / units.s,
        timpact=0.88 * units.Gyr,
        impact_angle=-2.34 * units.rad,
        GM=10.0**8.0 * units.Msun,
        rs=625.0 * units.pc,
    )
    # Setup nounit version for later
    sdf_sanders15_nou = streamgapdf(
        sigv.to(units.km / units.s).value / V0,
        progenitor=prog_unp_peri,
        pot=lp,
        aA=aAI,
        leading=False,
        nTrackChunks=5,
        nTrackIterations=1,
        nTrackChunksImpact=5,
        Vnorm=V0,
        Rnorm=R0,
        sigMeanOffset=4.5,
        tdisrupt=10.88 / conversion.time_in_Gyr(V0, R0),
        impactb=0.1 / R0,
        subhalovel=numpy.array([6.82200571, 132.7700529, 149.4174464]) / V0,
        timpact=0.88 / conversion.time_in_Gyr(V0, R0),
        impact_angle=-2.34,
        GM=10.0**-2.0 / conversion.mass_in_1010msol(V0, R0),
        rs=0.625 / R0,
    )
    # turn off units
    sdf_sanders15_nou._roSet = False
    sdf_sanders15_nou._voSet = False
    assert isinstance(
        sdf_sanders15.meanOmega(0.1), units.Quantity
    ), "streamgapdf method meanOmega does not return Quantity when it should"
    return None


def test_streamgapdf_method_returnunit():
    try:
        sdf_sanders15.meanOmega(0.1).to(1 / units.Gyr)
    except units.UnitConversionError:
        raise AssertionError(
            "streamdf method meanOmega does not return Quantity with the right units"
        )
    return None


def test_streamgapdf_method_value():
    from galpy.util import conversion

    assert numpy.all(
        numpy.fabs(
            sdf_sanders15.meanOmega(0.1).to(1 / units.Gyr).value
            / conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
            - sdf_sanders15_nou.meanOmega(0.1)
        )
        < 10.0**-8.0
    ), "streamgapdf method meanOmega does not return correct Quantity"
    return None


def test_streamgapdf_setup_impactparamsAsQuantity():
    assert (
        numpy.fabs(sdf_sanders15._impactb - sdf_sanders15_nou._impactb) < 10.0**-8.0
    ), "impactb specified as Quantity for streamgapdf does not work as expected"
    assert (
        numpy.fabs(sdf_sanders15._impact_angle - sdf_sanders15_nou._impact_angle)
        < 10.0**-8.0
    ), "impact_angle specified as Quantity for streamgapdf does not work as expected"
    assert (
        numpy.fabs(sdf_sanders15._timpact - sdf_sanders15_nou._timpact) < 10.0**-8.0
    ), "timpact specified as Quantity for streamgapdf does not work as expected"
    assert numpy.all(
        numpy.fabs(sdf_sanders15._subhalovel - sdf_sanders15_nou._subhalovel)
        < 10.0**-8.0
    ), "subhalovel specified as Quantity for streamgapdf does not work as expected"
    # GM and rs are not currently stored in streamgapdf, so just check kick
    assert numpy.all(
        numpy.fabs(sdf_sanders15._kick_deltav - sdf_sanders15_nou._kick_deltav)
        < 10.0**-8.0
    ), "Calculated kick from parameters specified as Quantity for streamgapdf does not work as expected"
    return None


def test_streamgapdf_inputAsQuantity():
    from galpy.util import conversion

    assert (
        numpy.fabs(
            sdf_sanders15.pOparapar(0.2 / units.Gyr, 30.0 * units.deg)
            - sdf_sanders15_nou.pOparapar(
                0.2 / conversion.freq_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro),
                30.0 * numpy.pi / 180.0,
            )
        )
        < 1e-4
    ), "streamgapdf method pOparapar with Quantity input does not return correct Quantity"
    return None


def test_streamgapdf_sample():
    from galpy.util import conversion

    # RvR
    numpy.random.seed(1)
    RvR = sdf_sanders15.sample(1)
    numpy.random.seed(1)
    RvRnou = sdf_sanders15_nou.sample(1)
    assert (
        numpy.fabs(RvR[0].to(units.kpc).value / sdf_sanders15._ro - RvRnou[0])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    assert (
        numpy.fabs(RvR[3].to(units.kpc).value / sdf_sanders15._ro - RvRnou[3]) < 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    assert (
        numpy.fabs(RvR[1].to(units.km / units.s).value / sdf_sanders15._vo - RvRnou[1])
        < 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    assert (
        numpy.fabs(RvR[2].to(units.km / units.s).value / sdf_sanders15._vo - RvRnou[2])
        < 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    assert (
        numpy.fabs(RvR[4].to(units.km / units.s).value / sdf_sanders15._vo - RvRnou[4])
        < 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    assert (
        numpy.fabs(RvR[5].to(units.rad).value - RvRnou[5]) < 1e-6
    ), "streamgapdf sample RvR does not return a correct Quantity"
    # RvR,dt
    numpy.random.seed(1)
    RvRdt = sdf_sanders15.sample(1, returndt=True)
    numpy.random.seed(1)
    RvRdtnou = sdf_sanders15_nou.sample(1, returndt=True)
    assert (
        numpy.fabs(RvRdt[0].to(units.kpc).value / sdf_sanders15._ro - RvRdtnou[0])
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(RvRdt[3].to(units.kpc).value / sdf_sanders15._ro - RvRdtnou[3])
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(
            RvRdt[1].to(units.km / units.s).value / sdf_sanders15._vo - RvRdtnou[1]
        )
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(
            RvRdt[2].to(units.km / units.s).value / sdf_sanders15._vo - RvRdtnou[2]
        )
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(
            RvRdt[4].to(units.km / units.s).value / sdf_sanders15._vo - RvRdtnou[4]
        )
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(RvRdt[5].to(units.rad).value - RvRdtnou[5]) < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    assert (
        numpy.fabs(
            RvRdt[6].to(units.Gyr).value
            / conversion.time_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
            - RvRdtnou[6]
        )
        < 1e-6
    ), "streamgapdf sample RvRdt does not return a correct Quantity"
    # xy
    numpy.random.seed(1)
    xy = sdf_sanders15.sample(1, xy=True)
    numpy.random.seed(1)
    xynou = sdf_sanders15_nou.sample(1, xy=True)
    assert (
        numpy.fabs(xy[0].to(units.kpc).value / sdf_sanders15._ro - xynou[0]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[1].to(units.kpc).value / sdf_sanders15._ro - xynou[1]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[2].to(units.kpc).value / sdf_sanders15._ro - xynou[2]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[3].to(units.km / units.s).value / sdf_sanders15._vo - xynou[3])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[4].to(units.km / units.s).value / sdf_sanders15._vo - xynou[4])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[5].to(units.km / units.s).value / sdf_sanders15._vo - xynou[5])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    # xydt
    numpy.random.seed(1)
    xydt = sdf_sanders15.sample(1, xy=True, returndt=True)
    numpy.random.seed(1)
    xydtnou = sdf_sanders15_nou.sample(1, xy=True, returndt=True)
    assert (
        numpy.fabs(xy[0].to(units.kpc).value / sdf_sanders15._ro - xynou[0]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[1].to(units.kpc).value / sdf_sanders15._ro - xynou[1]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[2].to(units.kpc).value / sdf_sanders15._ro - xynou[2]) < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[3].to(units.km / units.s).value / sdf_sanders15._vo - xynou[3])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[4].to(units.km / units.s).value / sdf_sanders15._vo - xynou[4])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(xy[5].to(units.km / units.s).value / sdf_sanders15._vo - xynou[5])
        < 1e-6
    ), "streamgapdf sample xy does not return a correct Quantity"
    assert (
        numpy.fabs(
            xydt[6].to(units.Gyr).value
            / conversion.time_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
            - xydtnou[6]
        )
        < 1e-6
    ), "streamgapdf sample xydt does not return a correct Quantity"
    # lb
    numpy.random.seed(1)
    lb = sdf_sanders15.sample(1, lb=True)
    numpy.random.seed(1)
    lbnou = sdf_sanders15_nou.sample(1, lb=True)
    assert (
        numpy.fabs(lb[0].to(units.deg).value - lbnou[0])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-5
    ), "streamgapdf sample lb does not return a correct Quantity"
    assert (
        numpy.fabs(lb[1].to(units.deg).value - lbnou[1])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-5
    ), "streamgapdf sample lb does not return a correct Quantity"
    assert (
        numpy.fabs(lb[2].to(units.kpc).value - lbnou[2])
        < _NUMPY_1_22 * 1e-5 + (1 - _NUMPY_1_22) * 1e-8
    ), "streamgapdf sample lb does not return a correct Quantity"
    assert (
        numpy.fabs(lb[3].to(units.km / units.s).value - lbnou[3])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-5
    ), "streamgapdf sample lb does not return a correct Quantity"
    assert (
        numpy.fabs(lb[4].to(units.mas / units.yr).value - lbnou[4])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-5
    ), "streamgapdf sample lb does not return a correct Quantity"
    assert (
        numpy.fabs(lb[5].to(units.mas / units.yr).value - lbnou[5])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-5
    ), "streamgapdf sample lb does not return a correct Quantity"
    # lbdt
    numpy.random.seed(1)
    lbdt = sdf_sanders15.sample(1, lb=True, returndt=True)
    numpy.random.seed(1)
    lbdtnou = sdf_sanders15_nou.sample(1, lb=True, returndt=True)
    assert (
        numpy.fabs(lbdt[0].to(units.deg).value - lbdtnou[0])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(lbdt[1].to(units.deg).value - lbdtnou[1])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(lbdt[2].to(units.kpc).value - lbdtnou[2])
        < _NUMPY_1_22 * 1e-5 + (1 - _NUMPY_1_22) * 1e-8
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(lbdt[3].to(units.km / units.s).value - lbdtnou[3])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(lbdt[4].to(units.mas / units.yr).value - lbdtnou[4])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(lbdt[5].to(units.mas / units.yr).value - lbdtnou[5])
        < _NUMPY_1_22 * 1e-4 + (1 - _NUMPY_1_22) * 1e-6
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    assert (
        numpy.fabs(
            lbdt[6].to(units.Gyr).value
            / conversion.time_in_Gyr(sdf_sanders15._vo, sdf_sanders15._ro)
            - lbdtnou[6]
        )
        < _NUMPY_1_22 * 1e-6 + (1 - _NUMPY_1_22) * 1e-8
    ), "streamgapdf sample lbdt does not return a correct Quantity"
    return None


def test_streamspraydf_setup_paramsAsQuantity():
    # Imports
    from galpy.df import streamspraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion  # for unit conversions

    ro, vo = 8.0, 220.0
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    Mass = 2 * 10.0**4.0 * units.Msun
    tdisrupt = 4.5 * units.Gyr
    # Object with physical inputs off
    spdf_bovy14_nou = streamspraydf(
        Mass.to_value(units.Msun) / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=tdisrupt.to_value(units.Gyr) / conversion.time_in_Gyr(vo, ro),
    )
    # Object with physical on
    spdf_bovy14 = streamspraydf(
        Mass, progenitor=obs, pot=lp, tdisrupt=tdisrupt, ro=ro, vo=vo
    )
    numpy.random.seed(10)
    sam = spdf_bovy14.sample(n=2)
    numpy.random.seed(10)
    sam_nou = spdf_bovy14_nou.sample(n=2)
    assert numpy.all(
        numpy.fabs(sam.r(use_physical=False) - sam_nou.r(use_physical=False)) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam.vr(use_physical=False) - sam_nou.vr(use_physical=False)) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    return None


def test_streamspraydf_sample_orbit():
    from galpy import potential
    from galpy.df import streamspraydf
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    # Object with physical off
    spdf_bovy14_nou = streamspraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    # Object with physical on
    spdf_bovy14 = streamspraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
    )
    numpy.random.seed(10)
    sam = spdf_bovy14.sample(n=2)
    numpy.random.seed(10)
    sam_nou = spdf_bovy14_nou.sample(n=2)
    assert numpy.all(
        numpy.fabs(sam.r(use_physical=False) - sam_nou.r(use_physical=False)) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam.vr(use_physical=False) - sam_nou.vr(use_physical=False)) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    return None


def test_streamspraydf_sample_RvR():
    from galpy import potential
    from galpy.df import streamspraydf
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    # Object with physical off
    spdf_bovy14_nou = streamspraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
    )
    # Object with physical on
    spdf_bovy14 = streamspraydf(
        2 * 10.0**4.0 / conversion.mass_in_msol(vo, ro),
        progenitor=obs,
        pot=lp,
        tdisrupt=4.5 / conversion.time_in_Gyr(vo, ro),
        ro=ro,
        vo=vo,
    )
    numpy.random.seed(10)
    sam, dt = spdf_bovy14.sample(n=2, return_orbit=False, returndt=True)
    numpy.random.seed(10)
    sam_nou, dt_nou = spdf_bovy14_nou.sample(n=2, return_orbit=False, returndt=True)
    assert numpy.all(
        numpy.fabs(sam[0].to_value(units.kpc) / ro - sam_nou[0]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[1].to_value(units.km / units.s) / vo - sam_nou[1]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[2].to_value(units.km / units.s) / vo - sam_nou[2]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[3].to_value(units.kpc) / ro - sam_nou[3]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[4].to_value(units.km / units.s) / vo - sam_nou[4]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(sam[5].to_value(units.rad) - sam_nou[5]) < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    assert numpy.all(
        numpy.fabs(dt.to_value(units.Gyr) / conversion.time_in_Gyr(vo, ro) - dt_nou)
        < 1e-8
    ), "Sample returned by streamspraydf.sample with with unit output is inconsistenty with the same sample sampled without unit output"
    return None


def test_df_inconsistentPotentialUnits_error():
    from galpy.actionAngle import actionAngleAdiabatic
    from galpy.df import quasiisothermaldf, streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    ro, vo = 9.0, 220.0
    # quasiisothermaldf
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9, ro=ro, vo=vo)
    aA = actionAngleAdiabatic(pot=lp, c=True, ro=ro, vo=vo)
    with pytest.raises(AssertionError) as excinfo:
        qdf = quasiisothermaldf(
            1.0 / 3.0,
            0.2,
            0.1,
            1.0,
            1.0,
            pot=lp,
            aA=aA,
            cutcounter=True,
            ro=ro * 1.1,
            vo=vo,
        )
    with pytest.raises(AssertionError) as excinfo:
        qdf = quasiisothermaldf(
            1.0 / 3.0,
            0.2,
            0.1,
            1.0,
            1.0,
            pot=lp,
            aA=aA,
            cutcounter=True,
            ro=ro,
            vo=vo * 1.1,
        )
    with pytest.raises(AssertionError) as excinfo:
        qdf = quasiisothermaldf(
            1.0 / 3.0,
            0.2,
            0.1,
            1.0,
            1.0,
            pot=lp,
            aA=aA,
            cutcounter=True,
            ro=ro * 1.1,
            vo=vo * 1.1,
        )
    # streamdf
    from galpy.actionAngle import actionAngleIsochroneApprox

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9, ro=ro, vo=vo)
    aAI = actionAngleIsochroneApprox(pot=lp, b=0.8, ro=ro, vo=vo)
    obs = Orbit(
        [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    )
    sigv = 0.365  # km/s
    with pytest.raises(AssertionError) as excinfo:
        sdf = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=30.0,
            ro=ro * 1.1,
            vo=vo,
        )
    with pytest.raises(AssertionError) as excinfo:
        sdf = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=30.0,
            ro=ro,
            vo=vo * 1.1,
        )
    with pytest.raises(AssertionError) as excinfo:
        sdf = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=lp,
            aA=aAI,
            leading=True,
            nTrackChunks=11,
            tdisrupt=30.0,
            ro=ro * 1.1,
            vo=vo * 1.1,
        )
    return None


def test_jeans_sigmar_returntype():
    from galpy.df import jeans
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    ro, vo = 8.5, 240.0
    assert isinstance(
        jeans.sigmar(lp, 2.0, ro=ro, vo=vo), units.Quantity
    ), "jeans.sigmar does not return Quantity when it should"
    return None


def test_jeans_sigmar_returnunit():
    from galpy.df import jeans
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    ro, vo = 8.5, 240.0
    try:
        jeans.sigmar(lp, 2.0, ro=ro, vo=vo).to(units.km / units.s)
    except units.UnitConversionError:
        raise AssertionError(
            "jeans.sigmar does not return Quantity with the right units"
        )
    return None


def test_jeans_sigmar_value():
    from galpy.df import jeans
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    ro, vo = 8.5, 240.0
    assert (
        numpy.fabs(
            jeans.sigmar(lp, 2.0, ro=ro, vo=vo).to(units.km / units.s).value
            - jeans.sigmar(lp, 2.0) * vo
        )
        < 10.0**-8.0
    ), "jeans.sigmar does not return correct Quantity"
    return None


def test_jeans_sigmar_inputAsQuantity():
    from galpy.df import jeans
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=1.0)
    ro, vo = 8.5, 240.0
    assert (
        numpy.fabs(
            jeans.sigmar(lp, 2.0 * ro * units.kpc, ro=ro, vo=vo)
            .to(units.km / units.s)
            .value
            - jeans.sigmar(lp, 2.0) * vo
        )
        < 10.0**-8.0
    ), "jeans.sigmar does not return correct Quantity"
    return None


def test_orbitmethodswunits_quantity_issue326():
    # Methods that *always* return a number with implied units
    # (like Orbit.dist), should return always return a Quantity when
    # apy-units=True in the configuration file (see issue 326)
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 0.0])
    # First make sure we're testing what we want to test
    assert not o._roSet, "Test of whether or not Orbit methods that should always return a Quantity do so cannot run meaningfully when _roSet is True"
    assert not o._voSet, "Test of whether or not Orbit methods that should always return a Quantity do so cannot run meaningfully when _voSet is True"
    # Then test methods
    assert isinstance(
        o.ra(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.dec(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.ll(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.bb(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.dist(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmra(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmdec(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmll(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmbb(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.vlos(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioX(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioY(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioZ(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.U(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.V(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.W(), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    return None


def test_orbitmethodswunits_quantity_overrideusephysical_issue326():
    # Methods that *always* return a number with implied units
    # (like Orbit.dist), should return always return a Quantity when
    # apy-units=True in the configuration file (see issue 326)
    # This test: *even* when use_physical=False
    from galpy.orbit import Orbit

    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 0.0])
    # First make sure we're testing what we want to test
    assert not o._roSet, "Test of whether or not Orbit methods that should always return a Quantity do so cannot run meaningfully when _roSet is True"
    assert not o._voSet, "Test of whether or not Orbit methods that should always return a Quantity do so cannot run meaningfully when _voSet is True"
    # Then test methods
    assert isinstance(
        o.ra(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.dec(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.ll(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.bb(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.dist(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmra(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmdec(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmll(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.pmbb(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.vlos(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioX(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioY(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.helioZ(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.U(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.V(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    assert isinstance(
        o.W(use_physical=False), units.Quantity
    ), "Orbit method ra does not return Quantity when called for orbit with _roSet = False / _voSet = False"
    return None


def test_SkyCoord_nodoubleunits_issue325():
    # make sure that SkyCoord doesn't return distances with units like kpc^2
    # which happened before, because it would use a distance with units of
    # kpc and then again multiply with kpc
    from galpy.orbit import Orbit

    o = Orbit(vxvv=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], radec=True)
    # Check return units of SkyCoord
    try:
        o.SkyCoord().ra.to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method SkyCoord has the wrong units for the right ascension"
        )
    try:
        o.SkyCoord().dec.to(units.deg)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method SkyCoord has the wrong units for the declination"
        )
    try:
        o.SkyCoord().distance.to(units.kpc)
    except units.UnitConversionError:
        raise AssertionError(
            "Orbit method SkyCoord has the wrong units for the distance"
        )
    return None
