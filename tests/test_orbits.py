##########################TESTS ON MULTIPLE ORBITS#############################
import astropy
import astropy.coordinates as apycoords
import astropy.units as u
import numpy
import pytest

from galpy import potential

_APY3 = astropy.__version__ > "3"


# Test Orbits initialization
def test_initialization_vxvv():
    from galpy.orbit import Orbit

    # 1D
    vxvvs = [[1.0, 0.1], [0.1, 3.0]]
    orbits = Orbit(vxvvs)
    assert (
        orbits.dim() == 1
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    assert (
        orbits.phasedim() == 2
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    assert (
        numpy.fabs(orbits.x()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    assert (
        numpy.fabs(orbits.x()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    assert (
        numpy.fabs(orbits.vx()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    assert (
        numpy.fabs(orbits.vx()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv in 1D does not work as expected"
    # 2D, 3 phase-D
    vxvvs = [[1.0, 0.1, 1.0], [0.1, 3.0, 1.1]]
    orbits = Orbit(vxvvs)
    assert (
        orbits.dim() == 2
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 3
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[1] - 1.1) < 1e-10
    ), "Orbits initialization with vxvv in 2D, 3 phase-D does not work as expected"
    # 2D, 4 phase-D
    vxvvs = [[1.0, 0.1, 1.0, 1.5], [0.1, 3.0, 1.1, 2.0]]
    orbits = Orbit(vxvvs)
    assert (
        orbits.dim() == 2
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 4
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[1] - 1.1) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.phi()[0] - 1.5) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.phi()[1] - 2.0) < 1e-10
    ), "Orbits initialization with vxvv 2D, 4 phase-D does not work as expected"
    # 3D, 5 phase-D
    vxvvs = [[1.0, 0.1, 1.0, 0.1, -0.2], [0.1, 3.0, 1.1, -0.3, 0.4]]
    orbits = Orbit(vxvvs)
    assert (
        orbits.dim() == 3
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 5
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[1] - 1.1) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.z()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.z()[1] + 0.3) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vz()[0] + 0.2) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vz()[1] - 0.4) < 1e-10
    ), "Orbits initialization with vxvv 3D, 5 phase-D does not work as expected"
    # 3D, 6 phase-D
    vxvvs = [[1.0, 0.1, 1.0, 0.1, -0.2, 1.5], [0.1, 3.0, 1.1, -0.3, 0.4, 2.0]]
    orbits = Orbit(vxvvs)
    assert (
        orbits.dim() == 3
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 6
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.R()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vR()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vT()[1] - 1.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.z()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.z()[1] + 0.3) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vz()[0] + 0.2) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.vz()[1] - 0.4) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.phi()[0] - 1.5) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits.phi()[1] - 2.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    return None


def test_initialization_SkyCoord():
    # Only run this for astropy>3
    if not _APY3:
        return None
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s
    # Without any custom coordinate-transformation parameters
    co = apycoords.SkyCoord(
        ra=ras,
        dec=decs,
        distance=dists,
        pm_ra_cosdec=pmras,
        pm_dec=pmdecs,
        radial_velocity=vloss,
        frame="icrs",
    )
    orbits = Orbit(co)
    assert (
        orbits.dim() == 3
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 6
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    for ii in range(nrand):
        to = Orbit(co[ii])
        assert (
            numpy.fabs(orbits.R()[ii] - to.R()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vR()[ii] - to.vR()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vT()[ii] - to.vT()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.z()[ii] - to.z()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vz()[ii] - to.vz()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.phi()[ii] - to.phi()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    # Also test list of Quantities
    orbits = Orbit([ras, decs, dists, pmras, pmdecs, vloss], radec=True)
    for ii in range(nrand):
        to = Orbit(co[ii])
        assert (
            numpy.fabs((orbits.R()[ii] - to.R()) / to.R()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vR()[ii] - to.vR()) / to.vR()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vT()[ii] - to.vT()) / to.vT()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.z()[ii] - to.z()) / to.z()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vz()[ii] - to.vz()) / to.vz()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(
                ((orbits.phi()[ii] - to.phi() + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    # With custom coordinate-transformation parameters
    v_sun = apycoords.CartesianDifferential([-11.1, 215.0, 3.25] * u.km / u.s)
    co = apycoords.SkyCoord(
        ra=ras,
        dec=decs,
        distance=dists,
        pm_ra_cosdec=pmras,
        pm_dec=pmdecs,
        radial_velocity=vloss,
        frame="icrs",
        galcen_distance=10.0 * u.kpc,
        z_sun=1.0 * u.kpc,
        galcen_v_sun=v_sun,
    )
    orbits = Orbit(co)
    assert (
        orbits.dim() == 3
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        orbits.phasedim() == 6
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    for ii in range(nrand):
        to = Orbit(co[ii])
        assert (
            numpy.fabs(orbits.R()[ii] - to.R()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vR()[ii] - to.vR()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vT()[ii] - to.vT()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.z()[ii] - to.z()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.vz()[ii] - to.vz()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(orbits.phi()[ii] - to.phi()) < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    return None


def test_initialization_list_of_arrays():
    # Test that initialization using a list of arrays works (see #548)
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    dists = numpy.random.uniform(size=nrand) * 10.0
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    orbits = Orbit([ras, decs, dists, pmras, pmdecs, vloss], radec=True)
    for ii in range(nrand):
        to = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]], radec=True
        )
        assert (
            numpy.fabs((orbits.R()[ii] - to.R()) / to.R()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vR()[ii] - to.vR()) / to.vR()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vT()[ii] - to.vT()) / to.vT()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.z()[ii] - to.z()) / to.z()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vz()[ii] - to.vz()) / to.vz()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(
                ((orbits.phi()[ii] - to.phi() + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi
            )
            < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    # Also test with R,vR, etc. input, badly
    orbits = Orbit([ras, decs, dists, pmras, pmdecs, vloss])
    for ii in range(nrand):
        to = Orbit([ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]])
        assert (
            numpy.fabs((orbits.R()[ii] - to.R()) / to.R()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vR()[ii] - to.vR()) / to.vR()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vT()[ii] - to.vT()) / to.vT()) < 1e-7
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.z()[ii] - to.z()) / to.z()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs((orbits.vz()[ii] - to.vz()) / to.vz()) < 1e-9
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
        assert (
            numpy.fabs(
                ((orbits.phi()[ii] - to.phi() + numpy.pi) % (2.0 * numpy.pi)) - numpy.pi
            )
            < 1e-10
        ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    return None


def test_initialization_diffro():
    # Test that supplying an array of ro values works as expected
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], ro=ros, radec=True)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            radec=True,
        )
        assert (
            numpy.fabs((all_orbs.R()[ii] - orb.R()) / orb.R()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.vR()[ii] - orb.vR()) / orb.vR()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.vT()[ii] - orb.vT()) / orb.vT()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.z()[ii] - orb.z()) / orb.z()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.vz()[ii] - orb.vz()) / orb.vz()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs(
                ((all_orbs.phi()[ii] - orb.phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        # Also some observed values like ra, dec, ...
        assert (
            numpy.fabs((all_orbs.ra()[ii] - orb.ra()) / orb.ra()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.dec()[ii] - orb.dec()) / orb.dec()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.dist()[ii] - orb.dist()) / orb.dist()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmra()[ii] - orb.pmra()) / orb.pmra()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmdec()[ii] - orb.pmdec()) / orb.pmdec()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert (
            numpy.fabs((all_orbs.vlos()[ii] - orb.vlos()) / orb.vlos()) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
    return None


def test_initialization_diffzo():
    # Test that supplying an array of zo values works as expected
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    zos = (-1.0 + numpy.random.uniform(size=nrand) * 2.0) * 50.0 * u.pc

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], zo=zos, radec=True)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            zo=zos[ii],
            radec=True,
        )
        assert (
            numpy.fabs((all_orbs.R()[ii] - orb.R()) / orb.R()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vR()[ii] - orb.vR()) / orb.vR()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vT()[ii] - orb.vT()) / orb.vT()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.z()[ii] - orb.z()) / orb.z()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vz()[ii] - orb.vz()) / orb.vz()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs(
                ((all_orbs.phi()[ii] - orb.phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        # Also some observed values like ra, dec, ...
        assert (
            numpy.fabs((all_orbs.ra()[ii] - orb.ra()) / orb.ra()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.dec()[ii] - orb.dec()) / orb.dec()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.dist()[ii] - orb.dist()) / orb.dist()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmra()[ii] - orb.pmra()) / orb.pmra()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmdec()[ii] - orb.pmdec()) / orb.pmdec()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vlos()[ii] - orb.vlos()) / orb.vlos()) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
    return None


def test_initialization_diffvo():
    # Test that supplying a single vo value works as expected
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], vo=vos, radec=True)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            vo=vos[ii],
            radec=True,
        )
        assert (
            numpy.fabs((all_orbs.R()[ii] - orb.R()) / orb.R()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vR()[ii] - orb.vR()) / orb.vR()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vT()[ii] - orb.vT()) / orb.vT()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.z()[ii] - orb.z()) / orb.z()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vz()[ii] - orb.vz()) / orb.vz()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs(
                ((all_orbs.phi()[ii] - orb.phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        # Also some observed values like ra, dec, ...
        assert (
            numpy.fabs((all_orbs.ra()[ii] - orb.ra()) / orb.ra()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.dec()[ii] - orb.dec()) / orb.dec()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.dist()[ii] - orb.dist()) / orb.dist()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmra()[ii] - orb.pmra()) / orb.pmra()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmdec()[ii] - orb.pmdec()) / orb.pmdec()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
        assert (
            numpy.fabs((all_orbs.vlos()[ii] - orb.vlos()) / orb.vlos()) < 1e-7
        ), "Orbits initialization with single vo does not work as expected"
    return None


def test_initialization_diffsolarmotion():
    # Test that supplying an array of solarmotion values works as expected
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss], solarmotion=solarmotions, radec=True
    )
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            solarmotion=solarmotions[:, ii],
            radec=True,
        )
        assert (
            numpy.fabs((all_orbs.R()[ii] - orb.R()) / orb.R()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.vR()[ii] - orb.vR()) / orb.vR()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.vT()[ii] - orb.vT()) / orb.vT()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.z()[ii] - orb.z()) / orb.z()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.vz()[ii] - orb.vz()) / orb.vz()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs(
                ((all_orbs.phi()[ii] - orb.phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        # Also some observed values like ra, dec, ...
        assert (
            numpy.fabs((all_orbs.ra()[ii] - orb.ra()) / orb.ra()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.dec()[ii] - orb.dec()) / orb.dec()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.dist()[ii] - orb.dist()) / orb.dist()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmra()[ii] - orb.pmra()) / orb.pmra()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmdec()[ii] - orb.pmdec()) / orb.pmdec()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert (
            numpy.fabs((all_orbs.vlos()[ii] - orb.vlos()) / orb.vlos()) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
    return None


def test_initialization_allsolarparams():
    # Test that supplying all parameters works as expected
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc
    zos = (-1.0 + numpy.random.uniform(size=nrand) * 2.0) * 50.0 * u.pc
    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s
    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss],
        ro=ros,
        zo=zos,
        vo=vos,
        solarmotion=solarmotions,
        radec=True,
    )
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            zo=zos[ii],
            vo=vos[ii],
            solarmotion=solarmotions[:, ii],
            radec=True,
        )
        assert (
            numpy.fabs((all_orbs.R()[ii] - orb.R()) / orb.R()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.vR()[ii] - orb.vR()) / orb.vR()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.vT()[ii] - orb.vT()) / orb.vT()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.z()[ii] - orb.z()) / orb.z()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.vz()[ii] - orb.vz()) / orb.vz()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs(
                ((all_orbs.phi()[ii] - orb.phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        # Also some observed values like ra, dec, ...
        assert (
            numpy.fabs((all_orbs.ra()[ii] - orb.ra()) / orb.ra()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.dec()[ii] - orb.dec()) / orb.dec()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.dist()[ii] - orb.dist()) / orb.dist()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmra()[ii] - orb.pmra()) / orb.pmra()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.pmdec()[ii] - orb.pmdec()) / orb.pmdec()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
        assert (
            numpy.fabs((all_orbs.vlos()[ii] - orb.vlos()) / orb.vlos()) < 1e-7
        ), "Orbits initialization with all parameters does not work as expected"
    return None


def test_initialization_diffsolarparams_shape_error():
    # Test that we get the correct error when providing the wrong shape for array
    # ro/zo/vo/solarmotion inputs
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 30
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand * 2) * 2.0) * u.kpc
    with pytest.raises(RuntimeError) as excinfo:
        Orbit([ras, decs, dists, pmras, pmdecs, vloss], ro=ros, radec=True)
    assert (
        excinfo.value.args[0]
        == "ro must have the same shape as the input orbits for an array of orbits"
    ), "Orbits initialization with wrong shape for ro does not raise correct error"

    zos = (-1.0 + numpy.random.uniform(size=2 * nrand) * 2.0) * 50.0 * u.pc
    with pytest.raises(RuntimeError) as excinfo:
        Orbit([ras, decs, dists, pmras, pmdecs, vloss], zo=zos, radec=True)
    assert (
        excinfo.value.args[0]
        == "zo must have the same shape as the input orbits for an array of orbits"
    ), "Orbits initialization with wrong shape for zo does not raise correct error"

    vos = (200.0 + numpy.random.uniform(size=2 * nrand) * 40.0) * u.km / u.s
    with pytest.raises(RuntimeError) as excinfo:
        Orbit([ras, decs, dists, pmras, pmdecs, vloss], vo=vos, radec=True)
    assert (
        excinfo.value.args[0]
        == "vo must have the same shape as the input orbits for an array of orbits"
    ), "Orbits initialization with wrong shape for vo does not raise correct error"

    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, 2 * nrand)) - 1.0) * 10.0 * u.km / u.s
    )
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(
            [ras, decs, dists, pmras, pmdecs, vloss],
            solarmotion=solarmotions,
            radec=True,
        )
    assert (
        excinfo.value.args[0]
        == "solarmotion must have the shape [3,...] where the ... matches the shape of the input orbits for an array of orbits"
    ), "Orbits initialization with wrong shape for solarmotion does not raise correct error"

    return None


# Tests that integrating Orbits agrees with integrating multiple Orbit
# instances
def test_integration_1d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [Orbit([1.0, 0.1]), Orbit([0.1, 1.0]), Orbit([-0.2, 0.3])]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(
        times, potential.toVerticalPotential(potential.MWPotential2014, 1.0)
    )
    orbits.integrate(
        times, potential.toVerticalPotential(potential.MWPotential2014, 1.0)
    )
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(
            times, potential.toVerticalPotential(potential.MWPotential2014, 1.0)
        )
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_2d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3]),
        Orbit([1.2, -0.3, 0.7, 5.0]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times, potential.MWPotential)
    orbits.integrate(times, potential.MWPotential)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].y(times) - orbits.y(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vy(times) - orbits.vy(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        (orbits_list[ii].phi(times) - orbits.phi(times)[ii] + numpy.pi)
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_p3d():
    # 3D phase-space integration
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0]),
        Orbit([0.9, 0.3, 1.0]),
        Orbit([1.2, -0.3, 0.7]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times, potential.MWPotential2014)
    orbits.integrate(times, potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_3d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times, potential.MWPotential2014)
    orbits.integrate(times, potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].y(times) - orbits.y(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vy(times) - orbits.vy(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].z(times) - orbits.z(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vz(times) - orbits.vz(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        (
                            (orbits_list[ii].phi(times) - orbits.phi(times)[ii])
                            + numpy.pi
                        )
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_p5d():
    # 5D phase-space integration
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits, twice to make sure initial cond. isn't changed
    orbits.integrate(times, potential.MWPotential2014)
    orbits.integrate(times, potential.MWPotential2014)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].z(times) - orbits.z(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vz(times) - orbits.vz(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integrate_3d_diffro():
    # Test that supplying an array of ro values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], ro=ros, radec=True)

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            radec=True,
        )
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.z(times)[ii] - orb.z(times)) / orb.z(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vz(times)[ii] - orb.vz(times)) / orb.vz(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
    return None


def test_integrate_3d_diffzo():
    # Test that supplying an array of zo values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    zos = (-1.0 + numpy.random.uniform(size=nrand) * 2.0) * 100.0 * u.pc

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], zo=zos, radec=True)

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            zo=zos[ii],
            radec=True,
        )
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.z(times)[ii] - orb.z(times)) / orb.z(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vz(times)[ii] - orb.vz(times)) / orb.vz(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of zo does not work as expected"
    return None


def test_integrate_3d_diffvo():
    # Test that supplying an array of zo values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s

    all_orbs = Orbit([ras, decs, dists, pmras, pmdecs, vloss], vo=vos, radec=True)

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            vo=vos[ii],
            radec=True,
        )
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.z(times)[ii] - orb.z(times)) / orb.z(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vz(times)[ii] - orb.vz(times)) / orb.vz(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
    return None


def test_integrate_3d_diffsolarmotion():
    # Test that supplying an array of zo values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss], solarmotion=solarmotions, radec=True
    )

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            solarmotion=solarmotions[:, ii],
            radec=True,
        )
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.z(times)[ii] - orb.z(times)) / orb.z(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vz(times)[ii] - orb.vz(times)) / orb.vz(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
    return None


def test_integrate_3d_diffallsolarparams():
    # Test that supplying an array of solar values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc
    zos = (-1.0 + numpy.random.uniform(size=nrand) * 2.0) * 100.0 * u.pc
    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s
    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss],
        ro=ros,
        zo=zos,
        vo=vos,
        solarmotion=solarmotions,
        radec=True,
    )

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            zo=zos[ii],
            vo=vos[ii],
            solarmotion=solarmotions[:, ii],
            radec=True,
        )
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.z(times)[ii] - orb.z(times)) / orb.z(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vz(times)[ii] - orb.vz(times)) / orb.vz(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
    return None


def test_integrate_2d_diffro():
    # Test that supplying an array of ro values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss], ro=ros, radec=True
    ).toPlanar()

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            radec=True,
        ).toPlanar()
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of ro does not work as expected"
    return None


def test_integrate_2d_diffvo():
    # Test that supplying an array of zo values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss], vo=vos, radec=True
    ).toPlanar()

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            vo=vos[ii],
            radec=True,
        ).toPlanar()
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of vo does not work as expected"
    return None


def test_integrate_2d_diffsolarmotion():
    # Test that supplying an array of zo values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss], solarmotion=solarmotions, radec=True
    ).toPlanar()

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            solarmotion=solarmotions[:, ii],
            radec=True,
        ).toPlanar()
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
    return None


def test_integrate_2d_diffallsolarparams():
    # Test that supplying an array of solar values works as expected when integrating an orbit
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 4
    ras = numpy.random.uniform(size=nrand) * 360.0 * u.deg
    decs = 90.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.deg
    dists = numpy.random.uniform(size=nrand) * 10.0 * u.kpc
    pmras = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    pmdecs = 20.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * 20.0 * u.mas / u.yr
    vloss = 200.0 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) * u.km / u.s

    ros = (6.0 + numpy.random.uniform(size=nrand) * 2.0) * u.kpc
    zos = (-1.0 + numpy.random.uniform(size=nrand) * 2.0) * 100.0 * u.pc
    vos = (200.0 + numpy.random.uniform(size=nrand) * 40.0) * u.km / u.s
    solarmotions = (
        (2.0 * numpy.random.uniform(size=(3, nrand)) - 1.0) * 10.0 * u.km / u.s
    )

    all_orbs = Orbit(
        [ras, decs, dists, pmras, pmdecs, vloss],
        ro=ros,
        zo=zos,
        vo=vos,
        solarmotion=solarmotions,
        radec=True,
    ).toPlanar()

    times = numpy.linspace(0.0, 10.0, 1001)

    all_orbs.integrate(times, potential.MWPotential2014)
    for ii in range(nrand):
        orb = Orbit(
            [ras[ii], decs[ii], dists[ii], pmras[ii], pmdecs[ii], vloss[ii]],
            ro=ros[ii],
            zo=zos[ii],
            vo=vos[ii],
            solarmotion=solarmotions[:, ii],
            radec=True,
        ).toPlanar()
        orb.integrate(times, potential.MWPotential2014)
        assert numpy.all(
            numpy.fabs((all_orbs.R(times)[ii] - orb.R(times)) / orb.R(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vR(times)[ii] - orb.vR(times)) / orb.vR(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vT(times)[ii] - orb.vT(times)) / orb.vT(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (
                    (all_orbs.phi(times)[ii] - orb.phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        # Also some observed values like ra, dec, ...
        assert numpy.all(
            numpy.fabs((all_orbs.ra(times)[ii] - orb.ra(times)) / orb.ra(times)) < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dec(times)[ii] - orb.dec(times)) / orb.dec(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.dist(times)[ii] - orb.dist(times)) / orb.dist(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.pmra(times)[ii] - orb.pmra(times)) / orb.pmra(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs(
                (all_orbs.pmdec(times)[ii] - orb.pmdec(times)) / orb.pmdec(times)
            )
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
        assert numpy.all(
            numpy.fabs((all_orbs.vlos(times)[ii] - orb.vlos(times)) / orb.vlos(times))
            < 1e-7
        ), "Orbits initialization with array of solarmotion does not work as expected"
    return None


# Tests that integrating Orbits agrees with integrating multiple Orbit
# instances when using parallel_map Python parallelization
def test_integration_forcemap_1d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [Orbit([1.0, 0.1]), Orbit([0.1, 1.0]), Orbit([-0.2, 0.3])]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits
    orbits.integrate(
        times,
        potential.toVerticalPotential(potential.MWPotential2014, 1.0),
        force_map=True,
    )
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(
            times, potential.toVerticalPotential(potential.MWPotential2014, 1.0)
        )
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_forcemap_2d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3]),
        Orbit([1.2, -0.3, 0.7, 5.0]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits
    orbits.integrate(times, potential.MWPotential2014, force_map=True)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].y(times) - orbits.y(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vy(times) - orbits.vy(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        (
                            (orbits_list[ii].phi(times) - orbits.phi(times)[ii])
                            + numpy.pi
                        )
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_forcemap_3d():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits
    orbits.integrate(times, potential.MWPotential2014, force_map=True)
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, potential.MWPotential2014)
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].x(times) - orbits.x(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vx(times) - orbits.vx(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].y(times) - orbits.y(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vy(times) - orbits.vy(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].z(times) - orbits.z(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vz(times) - orbits.vz(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        (
                            (orbits_list[ii].phi(times) - orbits.phi(times)[ii])
                            + numpy.pi
                        )
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integration_dxdv_2d():
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3]),
        Orbit([1.2, -0.3, 0.7, 5.0]),
    ]
    orbits = Orbit(orbits_list)
    numpy.random.seed(1)
    dxdv = (2.0 * numpy.random.uniform(size=orbits.shape + (4,)) - 1) / 10.0
    # Default, C integration
    integrator = "dopr54_c"
    orbits.integrate_dxdv(dxdv, times, lp, method=integrator)
    # Integrate as multiple Orbits
    for o, tdxdv in zip(orbits_list, dxdv):
        o.integrate_dxdv(tdxdv, times, lp, method=integrator)
    assert (
        numpy.amax(
            numpy.fabs(
                orbits.getOrbit_dxdv()
                - numpy.array([o.getOrbit_dxdv() for o in orbits_list])
            )
        )
        < 1e-8
    ), "Integration of the phase-space volume of multiple orbits as Orbits does not agree with integrating the phase-space volume of multiple orbits"
    # Python integration
    integrator = "odeint"
    orbits.integrate_dxdv(dxdv, times, lp, method=integrator)
    # Integrate as multiple Orbits
    for o, tdxdv in zip(orbits_list, dxdv):
        o.integrate_dxdv(tdxdv, times, lp, method=integrator)
    assert (
        numpy.amax(
            numpy.fabs(
                orbits.getOrbit_dxdv()
                - numpy.array([o.getOrbit_dxdv() for o in orbits_list])
            )
        )
        < 1e-8
    ), "Integration of the phase-space volume of multiple orbits as Orbits does not agree with integrating the phase-space volume of multiple orbits"
    return None


def test_integration_dxdv_2d_rectInOut():
    from galpy.orbit import Orbit

    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3]),
        Orbit([1.2, -0.3, 0.7, 5.0]),
    ]
    orbits = Orbit(orbits_list)
    numpy.random.seed(1)
    dxdv = (2.0 * numpy.random.uniform(size=orbits.shape + (4,)) - 1) / 10.0
    # Default, C integration
    integrator = "dopr54_c"
    orbits.integrate_dxdv(dxdv, times, lp, method=integrator, rectIn=True, rectOut=True)
    # Integrate as multiple Orbits
    for o, tdxdv in zip(orbits_list, dxdv):
        o.integrate_dxdv(tdxdv, times, lp, method=integrator, rectIn=True, rectOut=True)
    assert (
        numpy.amax(
            numpy.fabs(
                orbits.getOrbit_dxdv()
                - numpy.array([o.getOrbit_dxdv() for o in orbits_list])
            )
        )
        < 1e-8
    ), "Integration of the phase-space volume of multiple orbits as Orbits does not agree with integrating the phase-space volume of multiple orbits"
    # Python integration
    integrator = "odeint"
    orbits.integrate_dxdv(dxdv, times, lp, method=integrator, rectIn=True, rectOut=True)
    # Integrate as multiple Orbits
    for o, tdxdv in zip(orbits_list, dxdv):
        o.integrate_dxdv(tdxdv, times, lp, method=integrator, rectIn=True, rectOut=True)
    assert (
        numpy.amax(
            numpy.fabs(
                orbits.getOrbit_dxdv()
                - numpy.array([o.getOrbit_dxdv() for o in orbits_list])
            )
        )
        < 1e-8
    ), "Integration of the phase-space volume of multiple orbits as Orbits does not agree with integrating the phase-space volume of multiple orbits"
    return None


# Test that the 3D SOS function returns points with z=0, vz > 0
def test_SOS_3D():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.MWPotential2014
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
            t0=numpy.arange(len(orbits)),
        )
        zs = orbits.z(orbits.t)
        vzs = orbits.vz(orbits.t)
        assert (
            numpy.fabs(zs) < 10.0**-6.0
        ).all(), f"z on SOS is not zero for integrate_sos for method={method}"
        assert (
            vzs > 0.0
        ).all(), f"vz on SOS is not positive for integrate_sos for method={method}"
    return None


# Test that the 2D SOS function returns points with x=0, vx > 0
def test_SOS_2Dx():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, 3.0]),
        Orbit([1.2, -0.3, 0.7, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
            t0=numpy.arange(len(orbits)),
            surface="x",
        )
        xs = orbits.x(orbits.t)
        vxs = orbits.vx(orbits.t)
        assert (
            numpy.fabs(xs) < 10.0**-6.0
        ).all(), f"x on SOS is not zero for integrate_sos for method={method}"
        assert (
            vxs > 0.0
        ).all(), f"vx on SOS is not positive for integrate_sos for method={method}"
    return None


# Test that the 2D SOS function returns points with y=0, vy > 0
def test_SOS_2Dy():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, 3.0]),
        Orbit([1.2, -0.3, 0.7, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.SOS(
            pot,
            method=method,
            ncross=500 if "_c" in method else 20,
            force_map="rk" in method,
            t0=numpy.arange(len(orbits)),
            surface="y",
        )
        ys = orbits.y(orbits.t)
        vys = orbits.vy(orbits.t)
        assert (
            numpy.fabs(ys) < 10.0**-6.0
        ).all(), f"y on SOS is not zero for integrate_sos for method={method}"
        assert (
            vys > 0.0
        ).all(), f"vy on SOS is not positive for integrate_sos for method={method}"
    return None


# Test that the SOS integration returns an error
# when one orbit does not leave the surface
def test_SOS_onsurfaceerror_3D():
    from galpy.orbit import Orbit

    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.0, 0.0], [1.0, 0.1, 1.1, 0.0, 0.0, 0.0]])
    with pytest.raises(
        RuntimeError,
        match="An orbit appears to be within the SOS surface. Refusing to perform specialized SOS integration, please use normal integration instead",
    ):
        o.SOS(potential.MWPotential2014)
    return None


# Test that the 3D bruteSOS function returns points with z=0, vz > 0
def test_bruteSOS_3D():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.MWPotential2014
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
        )
        zs = orbits.z(orbits.t)
        vzs = orbits.vz(orbits.t)
        assert (
            numpy.fabs(zs[~numpy.isnan(zs)]) < 10.0**-3.0
        ).all(), f"z on bruteSOS is not zero for bruteSOS for method={method}"
        assert (
            vzs[~numpy.isnan(zs)] > 0.0
        ).all(), f"vz on bruteSOS is not positive for bruteSOS for method={method}"
    return None


# Test that the 2D bruteSOS function returns points with x=0, vx > 0
def test_bruteSOS_2Dx():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, 3.0]),
        Orbit([1.2, -0.3, 0.7, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
            surface="x",
        )
        xs = orbits.x(orbits.t)
        vxs = orbits.vx(orbits.t)
        assert (
            numpy.fabs(xs[~numpy.isnan(xs)]) < 10.0**-3.0
        ).all(), f"x on SOS is not zero for bruteSOS for method={method}"
        assert (
            vxs[~numpy.isnan(xs)] > 0.0
        ).all(), f"vx on SOS is not zero for bruteSOS for method={method}"
    return None


# Test that the 2D bruteSOS function returns points with y=0, vy > 0
def test_bruteSOS_2Dy():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0]),
        Orbit([0.9, 0.3, 1.0, 3.0]),
        Orbit([1.2, -0.3, 0.7, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.LogarithmicHaloPotential(normalize=1.0, q=0.9).toPlanar()
    for method in ["dopr54_c", "dop853_c", "rk4_c", "rk6_c", "dop853", "odeint"]:
        orbits.bruteSOS(
            numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
            pot,
            method=method,
            force_map="rk" in method,
            surface="y",
        )
        ys = orbits.y(orbits.t)
        vys = orbits.vy(orbits.t)
        assert (
            numpy.fabs(ys[~numpy.isnan(ys)]) < 10.0**-3.0
        ).all(), f"y on SOS is not zero for bruteSOS for method={method}"
        assert (
            vys[~numpy.isnan(ys)] > 0.0
        ).all(), f"vy on SOS is not zero for bruteSOS for method={method}"
    return None


# Test slicing of orbits
def test_slice_singleobject():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    orbits.integrate(times, potential.MWPotential2014)
    indices = [0, 1, -1]
    for ii in indices:
        assert (
            numpy.amax(numpy.fabs(orbits[ii].x(times) - orbits.x(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vx(times) - orbits.vx(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].y(times) - orbits.y(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vy(times) - orbits.vy(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].z(times) - orbits.z(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vz(times) - orbits.vz(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].R(times) - orbits.R(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vR(times) - orbits.vR(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vT(times) - orbits.vT(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        ((orbits[ii].phi(times) - orbits.phi(times)[ii]) + numpy.pi)
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


# Test slicing of orbits
def test_slice_multipleobjects():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
        Orbit([0.6, -0.4, 0.4, 0.25, -0.5, 6.0]),
        Orbit([1.1, -0.13, 0.17, 0.35, -0.5, 2.0]),
    ]
    orbits = Orbit(orbits_list)
    # Pre-integration
    orbits_slice = orbits[1:4]
    for ii in range(3):
        assert (
            numpy.amax(numpy.fabs(orbits_slice.x()[ii] - orbits.x()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.vx()[ii] - orbits.vx()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.y()[ii] - orbits.y()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.vy()[ii] - orbits.vy()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.z()[ii] - orbits.z()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.vz()[ii] - orbits.vz()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.R()[ii] - orbits.R()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.vR()[ii] - orbits.vR()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.vT()[ii] - orbits.vT()[ii + 1])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.phi()[ii] - orbits.phi()[ii + 1]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    # After integration
    orbits.integrate(times, potential.MWPotential2014)
    orbits_slice = orbits[1:4]
    for ii in range(3):
        assert (
            numpy.amax(numpy.fabs(orbits_slice.x(times)[ii] - orbits.x(times)[ii + 1]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.vx(times)[ii] - orbits.vx(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.y(times)[ii] - orbits.y(times)[ii + 1]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.vy(times)[ii] - orbits.vy(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.z(times)[ii] - orbits.z(times)[ii + 1]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.vz(times)[ii] - orbits.vz(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_slice.R(times)[ii] - orbits.R(times)[ii + 1]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.vR(times)[ii] - orbits.vR(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.vT(times)[ii] - orbits.vT(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(orbits_slice.phi(times)[ii] - orbits.phi(times)[ii + 1])
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


# Test slicing of orbits with non-trivial shapes
def test_slice_singleobject_multidim():
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = (5, 1, 3)
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vxvv = numpy.rollaxis(numpy.array([Rs, vRs, vTs, zs, vzs, phis]), 0, 4)
    orbits = Orbit(vxvv)
    times = numpy.linspace(0.0, 10.0, 1001)
    orbits.integrate(times, potential.MWPotential2014)
    indices = [(0, 0, 0), (1, 0, 2), (-1, 0, 1)]
    for ii in indices:
        assert (
            numpy.amax(numpy.fabs(orbits[ii].x(times) - orbits.x(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vx(times) - orbits.vx(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].y(times) - orbits.y(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vy(times) - orbits.vy(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].z(times) - orbits.z(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vz(times) - orbits.vz(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].R(times) - orbits.R(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vR(times) - orbits.vR(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vT(times) - orbits.vT(times)[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        ((orbits[ii].phi(times) - orbits.phi(times)[ii]) + numpy.pi)
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


# Test slicing of orbits with non-trivial shapes
def test_slice_multipleobjects_multidim():
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = (5, 1, 3)
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vxvv = numpy.rollaxis(numpy.array([Rs, vRs, vTs, zs, vzs, phis]), 0, 4)
    orbits = Orbit(vxvv)
    times = numpy.linspace(0.0, 10.0, 1001)
    # Pre-integration
    orbits_slice = orbits[1:4, 0, :2]
    for ii in range(3):
        for jj in range(1):
            for kk in range(2):
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.x()[ii, kk] - orbits.x()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vx()[ii, kk] - orbits.vx()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.y()[ii, kk] - orbits.y()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vy()[ii, kk] - orbits.vy()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.z()[ii, kk] - orbits.z()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vz()[ii, kk] - orbits.vz()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.R()[ii, kk] - orbits.R()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vR()[ii, kk] - orbits.vR()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vT()[ii, kk] - orbits.vT()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.phi()[ii, kk] - orbits.phi()[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    # After integration
    orbits.integrate(times, potential.MWPotential2014)
    orbits_slice = orbits[1:4, 0, :2]
    for ii in range(3):
        for jj in range(1):
            for kk in range(2):
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.x(times)[ii, kk]
                            - orbits.x(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vx(times)[ii, kk]
                            - orbits.vx(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.y(times)[ii, kk]
                            - orbits.y(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vy(times)[ii, kk]
                            - orbits.vy(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.z(times)[ii, kk]
                            - orbits.z(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vz(times)[ii, kk]
                            - orbits.vz(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.R(times)[ii, kk]
                            - orbits.R(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vR(times)[ii, kk]
                            - orbits.vR(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.vT(times)[ii, kk]
                            - orbits.vT(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
                assert (
                    numpy.amax(
                        numpy.fabs(
                            orbits_slice.phi(times)[ii, kk]
                            - orbits.phi(times)[ii + 1, jj, kk]
                        )
                    )
                    < 1e-10
                ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_slice_integratedorbit_wrapperpot_367():
    # Test related to issue 367: slicing orbits with a potential that includes
    # a wrapper potential (from Ted Mackereth)
    from galpy.orbit import Orbit
    from galpy.potential import (
        DehnenBarPotential,
        DehnenSmoothWrapperPotential,
        LogarithmicHaloPotential,
    )

    # initialise a wrapper potential
    tform = -10.0
    tsteady = 5.0
    omega = 1.85
    angle = 25.0 / 180.0 * numpy.pi
    dp = DehnenBarPotential(
        omegab=omega,
        rb=3.5 / 8.0,
        Af=(1.0 / 75.0),
        tform=tform,
        tsteady=tsteady,
        barphi=angle,
    )
    lhp = LogarithmicHaloPotential(normalize=1.0)
    dswp = DehnenSmoothWrapperPotential(
        pot=dp,
        tform=-4.0 * 2.0 * numpy.pi / dp.OmegaP(),
        tsteady=2.0 * 2.0 * numpy.pi / dp.OmegaP(),
    )
    pot = [lhp, dswp]
    # initialise 2 random orbits
    r = numpy.random.randn(2) * 0.01 + 1.0
    z = numpy.random.randn(2) * 0.01 + 0.2
    phi = numpy.random.randn(2) * 0.01 + 0.0
    vR = numpy.random.randn(2) * 0.01 + 0.0
    vT = numpy.random.randn(2) * 0.01 + 1.0
    vz = numpy.random.randn(2) * 0.01 + 0.02
    vxvv = numpy.dstack([r, vR, vT, z, vz, phi])[0]
    os = Orbit(vxvv)
    times = numpy.linspace(0.0, 100.0, 3000)
    os.integrate(times, pot)
    # This failed in #367
    assert (
        not os[0] is None
    ), "Slicing an integrated Orbits instance with a WrapperPotential does not work"
    return None


# Test that slicing of orbits propagates unit info
def test_slice_physical_issue385():
    from galpy.orbit import Orbit

    ra = [17.2875, 302.2875, 317.79583333, 306.60833333, 9.65833333, 147.2]
    dec = [61.54730278, 42.86525833, 17.72774722, 9.45011944, -7.69072222, 13.74425556]
    dist = [0.16753225, 0.08499065, 0.03357057, 0.05411548, 0.11946004, 0.0727802]
    pmra = [633.01, 119.536, -122.216, 116.508, 20.34, 373.05]
    pmdec = [65.303, 540.224, -899.263, -548.329, -546.373, -774.38]
    vlos = [-317.86, -195.44, -44.15, -246.76, -46.79, -15.17]
    orbits = Orbit(
        numpy.column_stack([ra, dec, dist, pmra, pmdec, vlos]),
        radec=True,
        ro=9.0,
        vo=230.0,
        solarmotion=[-11.1, 24.0, 7.25],
    )
    assert (
        orbits._roSet
    ), "Test Orbit instance that was supposed to have physical output turned does not"
    assert (
        orbits._voSet
    ), "Test Orbit instance that was supposed to have physical output turned does not"
    assert (
        numpy.fabs(orbits._ro - 9.0) < 1e-10
    ), "Test Orbit instance that was supposed to have physical output turned does not have the right ro"
    assert (
        numpy.fabs(orbits._vo - 230.0) < 1e-10
    ), "Test Orbit instance that was supposed to have physical output turned does not have the right vo"
    for ii in range(orbits.size):
        assert orbits[
            ii
        ]._roSet, "Sliced Orbit instance that was supposed to have physical output turned does not"
        assert orbits[
            ii
        ]._voSet, "Sliced Orbit instance that was supposed to have physical output turned does not"
        assert (
            numpy.fabs(orbits[ii]._ro - 9.0) < 1e-10
        ), "Sliced Orbit instance that was supposed to have physical output turned does not have the right ro"
        assert (
            numpy.fabs(orbits[ii]._vo - 230.0) < 1e-10
        ), "Sliced Orbit instance that was supposed to have physical output turned does not have the right vo"
        assert (
            numpy.fabs(orbits[ii]._zo - orbits._zo) < 1e-10
        ), "Sliced Orbit instance that was supposed to have physical output turned does not have the right zo"
        assert numpy.all(
            numpy.fabs(orbits[ii]._solarmotion - orbits._solarmotion) < 1e-10
        ), "Sliced Orbit instance that was supposed to have physical output turned does not have the right zo"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].x() - orbits.x()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vx() - orbits.vx()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].y() - orbits.y()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vy() - orbits.vy()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].z() - orbits.z()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vz() - orbits.vz()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].R() - orbits.R()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vR() - orbits.vR()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits[ii].vT() - orbits.vT()[ii])) < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(
                numpy.fabs(
                    (
                        ((orbits[ii].phi() - orbits.phi()[ii]) + numpy.pi)
                        % (2.0 * numpy.pi)
                    )
                    - numpy.pi
                )
            )
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


# Test that slicing in the case of individual time arrays works as expected
# Currently, the only way individual time arrays occur is through SOS integration
# so we implementing this test using SOS integration
def test_slice_indivtimes():
    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.0, 0.1, 0.0]),
        Orbit([0.9, 0.3, 1.0, -0.3, 0.4, 3.0]),
        Orbit([1.2, -0.3, 0.7, 0.5, -0.5, 6.0]),
    ]
    orbits = Orbit(orbits_list)
    pot = potential.MWPotential2014
    orbits.SOS(pot, t0=numpy.arange(len(orbits)))
    # First check that we actually have individual times
    assert (
        len(orbits.t.shape) >= len(orbits.orbit.shape) - 1
    ), "Test should be using individual time arrays, but a single time array was found"
    # Now slice single and multiple
    assert numpy.all(
        orbits[0].t == orbits.t[0]
    ), "Individually sliced orbit with individual time arrays does not produce the correct time array in the slice"
    assert numpy.all(
        orbits[1].t == orbits.t[1]
    ), "Individually sliced orbit with individual time arrays does not produce the correct time array in the slice"
    assert numpy.all(
        orbits[:2].t == orbits.t[:2]
    ), "Multiply-sliced orbit with individual time arrays does not produce the correct time array in the slice"
    assert numpy.all(
        orbits[1:4].t == orbits.t[1:4]
    ), "Multiply-sliced orbit with individual time arrays does not produce the correct time array in the slice"
    return None


# Test that initializing Orbits with orbits with different phase-space
# dimensions raises an error
def test_initialize_diffphasedim_error():
    from galpy.orbit import Orbit

    # 2D with 3D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1], [1.0, 0.1, 1.0]])
    # 2D with 4D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1], [1.0, 0.1, 1.0, 0.1]])
    # 2D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1], [1.0, 0.1, 1.0, 0.1, 0.2]])
    # 2D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1], [1.0, 0.1, 1.0, 0.1, 0.2, 3.0]])
    # 3D with 4D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0], [1.0, 0.1, 1.0, 0.1]])
    # 3D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0], [1.0, 0.1, 1.0, 0.1, 0.2]])
    # 3D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0], [1.0, 0.1, 1.0, 0.1, 0.2, 6.0]])
    # 4D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0, 2.0], [1.0, 0.1, 1.0, 0.1, 0.2]])
    # 4D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0, 2.0], [1.0, 0.1, 1.0, 0.1, 0.2, 6.0]])
    # 5D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([[1.0, 0.1, 1.0, 0.2, -0.2], [1.0, 0.1, 1.0, 0.1, 0.2, 6.0]])

    # Also as Orbit inputs
    # 2D with 3D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1]), Orbit([1.0, 0.1, 1.0])])
    # 2D with 4D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1]), Orbit([1.0, 0.1, 1.0, 0.1])])
    # 2D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2])])
    # 2D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2, 3.0])])
    # 3D with 4D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1, 1.0]), Orbit([1.0, 0.1, 1.0, 0.1])])
    # 3D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1, 1.0]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2])])
    # 3D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1, 1.0]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2, 6.0])])
    # 4D with 5D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1, 1.0, 2.0]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2])])
    # 4D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit([Orbit([1.0, 0.1, 1.0, 2.0]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2, 6.0])])
    # 5D with 6D
    with pytest.raises(
        (RuntimeError, ValueError),
        match="All individual orbits in an Orbit class must have the same phase-space dimensionality",
    ) as excinfo:
        Orbit(
            [Orbit([1.0, 0.1, 1.0, 0.2, -0.2]), Orbit([1.0, 0.1, 1.0, 0.1, 0.2, 6.0])]
        )

    return None


# Test that initializing Orbits with a list of non-scalar Orbits raises an error
def test_initialize_listorbits_error():
    from galpy.orbit import Orbit

    with pytest.raises(RuntimeError) as excinfo:
        Orbit([Orbit([[1.0, 0.1], [1.0, 0.1]]), Orbit([[1.0, 0.1], [1.0, 0.1]])])
    return None


# Test that initializing Orbits with an array of the wrong shape raises an error, that is, the phase-space dim part is > 6 or 1
def test_initialize_wrongshape():
    from galpy.orbit import Orbit

    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(2, 12)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(3, 12)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(4, 12)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(2, 1)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(5, 12)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(6, 12)))
    with pytest.raises(RuntimeError) as excinfo:
        Orbit(numpy.random.uniform(size=(7, 12)))
    return None


def test_orbits_consistentro():
    from galpy.orbit import Orbit

    ro = 7.0
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], ro=ro),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], ro=ro),
    ]
    orbits = Orbit(orbits_list)
    # Check that ro is taken correctly
    assert (
        numpy.fabs(orbits._ro - orbits_list[0]._ro) < 1e-10
    ), "Orbits' ro not correctly taken from input list of Orbit instances"
    assert (
        orbits._roSet
    ), "Orbits' ro not correctly taken from input list of Orbit instances"
    # Check that consistency of ros is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, ro=6.0)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], ro=ro),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], ro=ro * 1.2),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, ro=ro)
    return None


def test_orbits_consistentvo():
    from galpy.orbit import Orbit

    vo = 230.0
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], vo=vo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], vo=vo),
    ]
    orbits = Orbit(orbits_list)
    # Check that vo is taken correctly
    assert (
        numpy.fabs(orbits._vo - orbits_list[0]._vo) < 1e-10
    ), "Orbits' vo not correctly taken from input list of Orbit instances"
    assert (
        orbits._voSet
    ), "Orbits' vo not correctly taken from input list of Orbit instances"
    # Check that consistency of vos is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, vo=210.0)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], vo=vo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], vo=vo * 1.2),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, vo=vo)
    return None


def test_orbits_consistentzo():
    from galpy.orbit import Orbit

    zo = 0.015
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], zo=zo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], zo=zo),
    ]
    orbits = Orbit(orbits_list)
    # Check that zo is taken correctly
    assert (
        numpy.fabs(orbits._zo - orbits_list[0]._zo) < 1e-10
    ), "Orbits' zo not correctly taken from input list of Orbit instances"
    # Check that consistency of zos is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, zo=0.045)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], zo=zo),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], zo=zo * 1.2),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, zo=zo)
    return None


def test_orbits_consistentsolarmotion():
    from galpy.orbit import Orbit

    solarmotion = numpy.array([-10.0, 20.0, 30.0])
    # Initialize Orbits from list of Orbit instances
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], solarmotion=solarmotion),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], solarmotion=solarmotion),
    ]
    orbits = Orbit(orbits_list)
    # Check that solarmotion is taken correctly
    assert numpy.all(
        numpy.fabs(orbits._solarmotion - orbits_list[0]._solarmotion) < 1e-10
    ), "Orbits' solarmotion not correctly taken from input list of Orbit instances"
    # Check that consistency of solarmotions is enforced
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, solarmotion=numpy.array([15.0, 20.0, 30]))
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, solarmotion=numpy.array([-10.0, 25.0, 30]))
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, solarmotion=numpy.array([-10.0, 20.0, -30]))
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], solarmotion=solarmotion),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], solarmotion=solarmotion * 1.2),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        orbits = Orbit(orbits_list, solarmotion=solarmotion)
    return None


def test_orbits_stringsolarmotion():
    from galpy.orbit import Orbit

    solarmotion = "hogg"
    orbits_list = [
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -3.0], solarmotion=solarmotion),
        Orbit([1.0, 0.1, 1.0, 0.1, 0.2, -4.0], solarmotion=solarmotion),
    ]
    orbits = Orbit(orbits_list, solarmotion="hogg")
    assert numpy.all(
        numpy.fabs(orbits._solarmotion - numpy.array([-10.1, 4.0, 6.7])) < 1e-10
    ), "String solarmotion not parsed correctly"
    return None


def test_orbits_dim_2dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit (using ~Plevne's example)
    from galpy.orbit import Orbit
    from galpy.util import conversion

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    ell_p = potential.EllipticalDiskPotential()
    pota = [b_p, ell_p]
    o = Orbit(
        [
            Orbit(
                vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0
            ),
            Orbit(
                vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0
            ),
        ]
    )
    ts = numpy.linspace(
        0.0, 3.5 / conversion.time_in_Gyr(vo=220.0, ro=8.0), 1000, endpoint=True
    )
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="odeint")
    return None


def test_orbit_dim_1dPot_3dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit
    from galpy.util import conversion

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    pota = potential.RZToverticalPotential(b_p, 1.1)
    o = Orbit(
        [
            Orbit(
                vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0
            ),
            Orbit(
                vxvv=[20.0, 10.0, 2.0, 3.2, 3.4, -100.0], radec=True, ro=8.0, vo=220.0
            ),
        ]
    )
    ts = numpy.linspace(
        0.0, 3.5 / conversion.time_in_Gyr(vo=220.0, ro=8.0), 1000, endpoint=True
    )
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="odeint")
    return None


def test_orbit_dim_1dPot_2dOrb():
    # Test that orbit integration throws an error when using a potential that
    # is lower dimensional than the orbit, for a 1D potential
    from galpy.orbit import Orbit

    b_p = potential.PowerSphericalPotentialwCutoff(
        alpha=1.8, rc=1.9 / 8.0, normalize=0.05
    )
    pota = [b_p.toVertical(1.1)]
    o = Orbit([Orbit(vxvv=[1.1, 0.1, 1.1, 0.1]), Orbit(vxvv=[1.1, 0.1, 1.1, 0.1])])
    ts = numpy.linspace(0.0, 10.0, 1001)
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="leapfrog")
    with pytest.raises(AssertionError) as excinfo:
        o.integrate(ts, pota, method="dop853")
    return None


# Test the error for when explicit stepsize does not divide the output stepsize
def test_check_integrate_dt():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit(
        [Orbit([1.0, 0.1, 1.2, 0.3, 0.2, 2.0]), Orbit([1.0, 0.1, 1.2, 0.3, 0.2, 2.0])]
    )
    times = numpy.linspace(0.0, 7.0, 251)
    # This shouldn't work
    try:
        o.integrate(times, lp, dt=(times[1] - times[0]) / 4.0 * 1.1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "dt that is not an integer divisor of the output step size does not raise a ValueError"
        )
    # This should
    try:
        o.integrate(times, lp, dt=(times[1] - times[0]) / 4.0)
    except ValueError:
        raise AssertionError(
            "dt that is an integer divisor of the output step size raises a ValueError"
        )
    return None


# Test that evaluating coordinate functions for integrated orbits works
def test_coordinate_interpolation():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # Before integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time() - list_os[ii].time()) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R()[ii] - list_os[ii].R()) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r()[ii] - list_os[ii].r()) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR()[ii] - list_os[ii].vR()) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT()[ii] - list_os[ii].vT()) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z()[ii] - list_os[ii].z()) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz()[ii] - list_os[ii].vz()) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                ((os.phi()[ii] - list_os[ii].phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.x()[ii] - list_os[ii].x()) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.y()[ii] - list_os[ii].y()) < 1e-10
        ), "Evaluating Orbits y does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx()[ii] - list_os[ii].vx()) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vy()[ii] - list_os[ii].vy()) < 1e-10
        ), "Evaluating Orbits vy does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi()[ii] - list_os[ii].vphi()) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ra()[ii] - list_os[ii].ra()) < 1e-10
        ), "Evaluating Orbits ra  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dec()[ii] - list_os[ii].dec()) < 1e-10
        ), "Evaluating Orbits dec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dist()[ii] - list_os[ii].dist()) < 1e-10
        ), "Evaluating Orbits dist does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ll()[ii] - list_os[ii].ll()) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.bb()[ii] - list_os[ii].bb()) < 1e-10
        ), "Evaluating Orbits bb  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmra()[ii] - list_os[ii].pmra()) < 1e-10
        ), "Evaluating Orbits pmra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmdec()[ii] - list_os[ii].pmdec()) < 1e-10
        ), "Evaluating Orbits pmdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmll()[ii] - list_os[ii].pmll()) < 1e-10
        ), "Evaluating Orbits pmll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmbb()[ii] - list_os[ii].pmbb()) < 1e-10
        ), "Evaluating Orbits pmbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vra()[ii] - list_os[ii].vra()) < 1e-10
        ), "Evaluating Orbits vra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vdec()[ii] - list_os[ii].vdec()) < 1e-10
        ), "Evaluating Orbits vdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vll()[ii] - list_os[ii].vll()) < 1e-10
        ), "Evaluating Orbits vll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vbb()[ii] - list_os[ii].vbb()) < 1e-10
        ), "Evaluating Orbits vbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vlos()[ii] - list_os[ii].vlos()) < 1e-10
        ), "Evaluating Orbits vlos does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioX()[ii] - list_os[ii].helioX()) < 1e-10
        ), "Evaluating Orbits helioX does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioY()[ii] - list_os[ii].helioY()) < 1e-10
        ), "Evaluating Orbits helioY does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioZ()[ii] - list_os[ii].helioZ()) < 1e-10
        ), "Evaluating Orbits helioZ does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.U()[ii] - list_os[ii].U()) < 1e-10
        ), "Evaluating Orbits U does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.V()[ii] - list_os[ii].V()) < 1e-10
        ), "Evaluating Orbits V does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.W()[ii] - list_os[ii].W()) < 1e-10
        ), "Evaluating Orbits W does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord().ra[ii] - list_os[ii].SkyCoord().ra).to(u.deg).value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord().dec[ii] - list_os[ii].SkyCoord().dec)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord().distance[ii] - list_os[ii].SkyCoord().distance)
            .to(u.kpc)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        if _APY3:
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord().pm_ra_cosdec[ii] - list_os[ii].SkyCoord().pm_ra_cosdec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(os.SkyCoord().pm_dec[ii] - list_os[ii].SkyCoord().pm_dec)
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord().radial_velocity[ii]
                    - list_os[ii].SkyCoord().radial_velocity
                )
                .to(u.km / u.s)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time(times) - list_os[ii].time(times)) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R(times)[ii] - list_os[ii].R(times)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times)[ii] - list_os[ii].r(times)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times)[ii] - list_os[ii].vR(times)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times)[ii] - list_os[ii].vT(times)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times)[ii] - list_os[ii].z(times)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times)[ii] - list_os[ii].vz(times)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(times)[ii] - list_os[ii].phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.x(times)[ii] - list_os[ii].x(times)) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.y(times)[ii] - list_os[ii].y(times)) < 1e-10
        ), "Evaluating Orbits y does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(times)[ii] - list_os[ii].vx(times)) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vy(times)[ii] - list_os[ii].vy(times)) < 1e-10
        ), "Evaluating Orbits vy does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times)[ii] - list_os[ii].vphi(times)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ra(times)[ii] - list_os[ii].ra(times)) < 1e-10
        ), "Evaluating Orbits ra  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dec(times)[ii] - list_os[ii].dec(times)) < 1e-10
        ), "Evaluating Orbits dec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dist(times)[ii] - list_os[ii].dist(times)) < 1e-10
        ), "Evaluating Orbits dist does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ll(times)[ii] - list_os[ii].ll(times)) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.bb(times)[ii] - list_os[ii].bb(times)) < 1e-10
        ), "Evaluating Orbits bb  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmra(times)[ii] - list_os[ii].pmra(times)) < 1e-10
        ), "Evaluating Orbits pmra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmdec(times)[ii] - list_os[ii].pmdec(times)) < 1e-10
        ), "Evaluating Orbits pmdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmll(times)[ii] - list_os[ii].pmll(times)) < 1e-10
        ), "Evaluating Orbits pmll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmbb(times)[ii] - list_os[ii].pmbb(times)) < 1e-10
        ), "Evaluating Orbits pmbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vra(times)[ii] - list_os[ii].vra(times)) < 1e-10
        ), "Evaluating Orbits vra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vdec(times)[ii] - list_os[ii].vdec(times)) < 1e-10
        ), "Evaluating Orbits vdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vll(times)[ii] - list_os[ii].vll(times)) < 1e-10
        ), "Evaluating Orbits vll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vbb(times)[ii] - list_os[ii].vbb(times)) < 1e-10
        ), "Evaluating Orbits vbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vlos(times)[ii] - list_os[ii].vlos(times)) < 1e-9
        ), "Evaluating Orbits vlos does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioX(times)[ii] - list_os[ii].helioX(times)) < 1e-10
        ), "Evaluating Orbits helioX does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioY(times)[ii] - list_os[ii].helioY(times)) < 1e-10
        ), "Evaluating Orbits helioY does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioZ(times)[ii] - list_os[ii].helioZ(times)) < 1e-10
        ), "Evaluating Orbits helioZ does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.U(times)[ii] - list_os[ii].U(times)) < 1e-10
        ), "Evaluating Orbits U does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.V(times)[ii] - list_os[ii].V(times)) < 1e-10
        ), "Evaluating Orbits V does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.W(times)[ii] - list_os[ii].W(times)) < 1e-10
        ), "Evaluating Orbits W does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord(times).ra[ii] - list_os[ii].SkyCoord(times).ra)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord(times).dec[ii] - list_os[ii].SkyCoord(times).dec)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(times).distance[ii] - list_os[ii].SkyCoord(times).distance
            )
            .to(u.kpc)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        if _APY3:
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times).pm_ra_cosdec[ii]
                    - list_os[ii].SkyCoord(times).pm_ra_cosdec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times).pm_dec[ii] - list_os[ii].SkyCoord(times).pm_dec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times).radial_velocity[ii]
                    - list_os[ii].SkyCoord(times).radial_velocity
                )
                .to(u.km / u.s)
                .value
                < 1e-9
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        # Also a single time in the array ...
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time(times[1]) - list_os[ii].time(times[1])) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R(times[1])[ii] - list_os[ii].R(times[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times[1])[ii] - list_os[ii].r(times[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times[1])[ii] - list_os[ii].vR(times[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times[1])[ii] - list_os[ii].vT(times[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times[1])[ii] - list_os[ii].z(times[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times[1])[ii] - list_os[ii].vz(times[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(times[1])[ii] - list_os[ii].phi(times[1]) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.x(times[1])[ii] - list_os[ii].x(times[1])) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.y(times[1])[ii] - list_os[ii].y(times[1])) < 1e-10
        ), "Evaluating Orbits y does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(times[1])[ii] - list_os[ii].vx(times[1])) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vy(times[1])[ii] - list_os[ii].vy(times[1])) < 1e-10
        ), "Evaluating Orbits vy does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times[1])[ii] - list_os[ii].vphi(times[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ra(times[1])[ii] - list_os[ii].ra(times[1])) < 1e-10
        ), "Evaluating Orbits ra  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dec(times[1])[ii] - list_os[ii].dec(times[1])) < 1e-10
        ), "Evaluating Orbits dec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dist(times[1])[ii] - list_os[ii].dist(times[1])) < 1e-10
        ), "Evaluating Orbits dist does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ll(times[1])[ii] - list_os[ii].ll(times[1])) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.bb(times[1])[ii] - list_os[ii].bb(times[1])) < 1e-10
        ), "Evaluating Orbits bb  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmra(times[1])[ii] - list_os[ii].pmra(times[1])) < 1e-10
        ), "Evaluating Orbits pmra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmdec(times[1])[ii] - list_os[ii].pmdec(times[1])) < 1e-10
        ), "Evaluating Orbits pmdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmll(times[1])[ii] - list_os[ii].pmll(times[1])) < 1e-10
        ), "Evaluating Orbits pmll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmbb(times[1])[ii] - list_os[ii].pmbb(times[1])) < 1e-10
        ), "Evaluating Orbits pmbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vra(times[1])[ii] - list_os[ii].vra(times[1])) < 1e-10
        ), "Evaluating Orbits vra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vdec(times[1])[ii] - list_os[ii].vdec(times[1])) < 1e-10
        ), "Evaluating Orbits vdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vll(times[1])[ii] - list_os[ii].vll(times[1])) < 1e-10
        ), "Evaluating Orbits vll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vbb(times[1])[ii] - list_os[ii].vbb(times[1])) < 1e-10
        ), "Evaluating Orbits vbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vlos(times[1])[ii] - list_os[ii].vlos(times[1])) < 1e-10
        ), "Evaluating Orbits vlos does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioX(times[1])[ii] - list_os[ii].helioX(times[1])) < 1e-10
        ), "Evaluating Orbits helioX does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioY(times[1])[ii] - list_os[ii].helioY(times[1])) < 1e-10
        ), "Evaluating Orbits helioY does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioZ(times[1])[ii] - list_os[ii].helioZ(times[1])) < 1e-10
        ), "Evaluating Orbits helioZ does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.U(times[1])[ii] - list_os[ii].U(times[1])) < 1e-10
        ), "Evaluating Orbits U does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.V(times[1])[ii] - list_os[ii].V(times[1])) < 1e-10
        ), "Evaluating Orbits V does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.W(times[1])[ii] - list_os[ii].W(times[1])) < 1e-10
        ), "Evaluating Orbits W does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord(times[1]).ra[ii] - list_os[ii].SkyCoord(times[1]).ra)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(times[1]).dec[ii] - list_os[ii].SkyCoord(times[1]).dec
            )
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(times[1]).distance[ii]
                - list_os[ii].SkyCoord(times[1]).distance
            )
            .to(u.kpc)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        if _APY3:
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times[1]).pm_ra_cosdec[ii]
                    - list_os[ii].SkyCoord(times[1]).pm_ra_cosdec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times[1]).pm_dec[ii]
                    - list_os[ii].SkyCoord(times[1]).pm_dec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(times[1]).radial_velocity[ii]
                    - list_os[ii].SkyCoord(times[1]).radial_velocity
                )
                .to(u.km / u.s)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes)[ii] - list_os[ii].r(itimes)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes)[ii] - list_os[ii].vR(itimes)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes)[ii] - list_os[ii].vT(itimes)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes)[ii] - list_os[ii].z(itimes)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes)[ii] - list_os[ii].vz(itimes)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(itimes)[ii] - list_os[ii].phi(itimes) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.x(itimes)[ii] - list_os[ii].x(itimes)) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.y(itimes)[ii] - list_os[ii].y(itimes)) < 1e-10
        ), "Evaluating Orbits y does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(itimes)[ii] - list_os[ii].vx(itimes)) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vy(itimes)[ii] - list_os[ii].vy(itimes)) < 1e-10
        ), "Evaluating Orbits vy does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes)[ii] - list_os[ii].vphi(itimes)) < 1e-10
        ), "Evaluating Orbits vphidoes not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ra(itimes)[ii] - list_os[ii].ra(itimes)) < 1e-10
        ), "Evaluating Orbits ra  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dec(itimes)[ii] - list_os[ii].dec(itimes)) < 1e-10
        ), "Evaluating Orbits dec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dist(itimes)[ii] - list_os[ii].dist(itimes)) < 1e-10
        ), "Evaluating Orbits dist does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ll(itimes)[ii] - list_os[ii].ll(itimes)) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.bb(itimes)[ii] - list_os[ii].bb(itimes)) < 1e-10
        ), "Evaluating Orbits bb  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmra(itimes)[ii] - list_os[ii].pmra(itimes)) < 1e-10
        ), "Evaluating Orbits pmra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmdec(itimes)[ii] - list_os[ii].pmdec(itimes)) < 1e-10
        ), "Evaluating Orbits pmdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmll(itimes)[ii] - list_os[ii].pmll(itimes)) < 1e-10
        ), "Evaluating Orbits pmll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmbb(itimes)[ii] - list_os[ii].pmbb(itimes)) < 1e-10
        ), "Evaluating Orbits pmbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vra(itimes)[ii] - list_os[ii].vra(itimes)) < 1e-10
        ), "Evaluating Orbits vra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vdec(itimes)[ii] - list_os[ii].vdec(itimes)) < 1e-10
        ), "Evaluating Orbits vdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vll(itimes)[ii] - list_os[ii].vll(itimes)) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vbb(itimes)[ii] - list_os[ii].vbb(itimes)) < 1e-10
        ), "Evaluating Orbits vbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vlos(itimes)[ii] - list_os[ii].vlos(itimes)) < 1e-10
        ), "Evaluating Orbits vlos does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioX(itimes)[ii] - list_os[ii].helioX(itimes)) < 1e-10
        ), "Evaluating Orbits helioX does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioY(itimes)[ii] - list_os[ii].helioY(itimes)) < 1e-10
        ), "Evaluating Orbits helioY does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioZ(itimes)[ii] - list_os[ii].helioZ(itimes)) < 1e-10
        ), "Evaluating Orbits helioZ does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.U(itimes)[ii] - list_os[ii].U(itimes)) < 1e-10
        ), "Evaluating Orbits U does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.V(itimes)[ii] - list_os[ii].V(itimes)) < 1e-10
        ), "Evaluating Orbits V does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.W(itimes)[ii] - list_os[ii].W(itimes)) < 1e-10
        ), "Evaluating Orbits W does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord(itimes).ra[ii] - list_os[ii].SkyCoord(itimes).ra)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.SkyCoord(itimes).dec[ii] - list_os[ii].SkyCoord(itimes).dec)
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(itimes).distance[ii] - list_os[ii].SkyCoord(itimes).distance
            )
            .to(u.kpc)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        if _APY3:
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes).pm_ra_cosdec[ii]
                    - list_os[ii].SkyCoord(itimes).pm_ra_cosdec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes).pm_dec[ii] - list_os[ii].SkyCoord(itimes).pm_dec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes).radial_velocity[ii]
                    - list_os[ii].SkyCoord(itimes).radial_velocity
                )
                .to(u.km / u.s)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes[1])[ii] - list_os[ii].r(itimes[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes[1])[ii] - list_os[ii].vR(itimes[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes[1])[ii] - list_os[ii].vT(itimes[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes[1])[ii] - list_os[ii].z(itimes[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes[1])[ii] - list_os[ii].vz(itimes[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(itimes[1])[ii] - list_os[ii].phi(itimes[1]) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ra(itimes[1])[ii] - list_os[ii].ra(itimes[1])) < 1e-10
        ), "Evaluating Orbits ra  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dec(itimes[1])[ii] - list_os[ii].dec(itimes[1])) < 1e-10
        ), "Evaluating Orbits dec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.dist(itimes[1])[ii] - list_os[ii].dist(itimes[1])) < 1e-10
        ), "Evaluating Orbits dist does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.ll(itimes[1])[ii] - list_os[ii].ll(itimes[1])) < 1e-10
        ), "Evaluating Orbits ll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.bb(itimes[1])[ii] - list_os[ii].bb(itimes[1])) < 1e-10
        ), "Evaluating Orbits bb  does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmra(itimes[1])[ii] - list_os[ii].pmra(itimes[1])) < 1e-10
        ), "Evaluating Orbits pmra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmdec(itimes[1])[ii] - list_os[ii].pmdec(itimes[1])) < 1e-10
        ), "Evaluating Orbits pmdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmll(itimes[1])[ii] - list_os[ii].pmll(itimes[1])) < 1e-10
        ), "Evaluating Orbits pmll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.pmbb(itimes[1])[ii] - list_os[ii].pmbb(itimes[1])) < 1e-10
        ), "Evaluating Orbits pmbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vra(itimes[1])[ii] - list_os[ii].vra(itimes[1])) < 1e-10
        ), "Evaluating Orbits vra does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vdec(itimes[1])[ii] - list_os[ii].vdec(itimes[1])) < 1e-10
        ), "Evaluating Orbits vdec does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vll(itimes[1])[ii] - list_os[ii].vll(itimes[1])) < 1e-10
        ), "Evaluating Orbits vll does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vbb(itimes[1])[ii] - list_os[ii].vbb(itimes[1])) < 1e-10
        ), "Evaluating Orbits vbb does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vlos(itimes[1])[ii] - list_os[ii].vlos(itimes[1])) < 1e-10
        ), "Evaluating Orbits vlos does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioX(itimes[1])[ii] - list_os[ii].helioX(itimes[1])) < 1e-10
        ), "Evaluating Orbits helioX does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioY(itimes[1])[ii] - list_os[ii].helioY(itimes[1])) < 1e-10
        ), "Evaluating Orbits helioY does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.helioZ(itimes[1])[ii] - list_os[ii].helioZ(itimes[1])) < 1e-10
        ), "Evaluating Orbits helioZ does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.U(itimes[1])[ii] - list_os[ii].U(itimes[1])) < 1e-10
        ), "Evaluating Orbits U does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.V(itimes[1])[ii] - list_os[ii].V(itimes[1])) < 1e-10
        ), "Evaluating Orbits V does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.W(itimes[1])[ii] - list_os[ii].W(itimes[1])) < 1e-10
        ), "Evaluating Orbits W does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(itimes[1]).ra[ii] - list_os[ii].SkyCoord(itimes[1]).ra
            )
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(itimes[1]).dec[ii] - list_os[ii].SkyCoord(itimes[1]).dec
            )
            .to(u.deg)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                os.SkyCoord(itimes[1]).distance[ii]
                - list_os[ii].SkyCoord(itimes[1]).distance
            )
            .to(u.kpc)
            .value
            < 1e-10
        ), "Evaluating Orbits SkyCoord does not agree with Orbit"
        if _APY3:
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes[1]).pm_ra_cosdec[ii]
                    - list_os[ii].SkyCoord(itimes[1]).pm_ra_cosdec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes[1]).pm_dec[ii]
                    - list_os[ii].SkyCoord(itimes[1]).pm_dec
                )
                .to(u.mas / u.yr)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.SkyCoord(itimes[1]).radial_velocity[ii]
                    - list_os[ii].SkyCoord(itimes[1]).radial_velocity
                )
                .to(u.km / u.s)
                .value
                < 1e-10
            ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    return None


# Test that evaluating coordinate functions for integrated orbits works,
# for 5D orbits
def test_coordinate_interpolation_5d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 20
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs)))
    list_os = [
        Orbit([R, vR, vT, z, vz]) for R, vR, vT, z, vz in zip(Rs, vRs, vTs, zs, vzs)
    ]
    # Before integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R()[ii] - list_os[ii].R()) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r()[ii] - list_os[ii].r()) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR()[ii] - list_os[ii].vR()) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT()[ii] - list_os[ii].vT()) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z()[ii] - list_os[ii].z()) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz()[ii] - list_os[ii].vz()) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi()[ii] - list_os[ii].vphi()) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(times)[ii] - list_os[ii].R(times)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times)[ii] - list_os[ii].r(times)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times)[ii] - list_os[ii].vR(times)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times)[ii] - list_os[ii].vT(times)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times)[ii] - list_os[ii].z(times)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times)[ii] - list_os[ii].vz(times)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times)[ii] - list_os[ii].vphi(times)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(times[1])[ii] - list_os[ii].R(times[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times[1])[ii] - list_os[ii].r(times[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times[1])[ii] - list_os[ii].vR(times[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times[1])[ii] - list_os[ii].vT(times[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times[1])[ii] - list_os[ii].z(times[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times[1])[ii] - list_os[ii].vz(times[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times[1])[ii] - list_os[ii].vphi(times[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes)[ii] - list_os[ii].r(itimes)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes)[ii] - list_os[ii].vR(itimes)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes)[ii] - list_os[ii].vT(itimes)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes)[ii] - list_os[ii].z(itimes)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes)[ii] - list_os[ii].vz(itimes)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes)[ii] - list_os[ii].vphi(itimes)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes[1])[ii] - list_os[ii].r(itimes[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes[1])[ii] - list_os[ii].vR(itimes[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes[1])[ii] - list_os[ii].vT(itimes[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes[1])[ii] - list_os[ii].z(itimes[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes[1])[ii] - list_os[ii].vz(itimes[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes[1])[ii] - list_os[ii].vphi(itimes[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    with pytest.raises(AttributeError):
        os.phi()
    return None


# Test that evaluating coordinate functions for integrated orbits works,
# for 4D orbits
def test_coordinate_interpolation_4d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 20
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, phis)))
    list_os = [Orbit([R, vR, vT, phi]) for R, vR, vT, phi in zip(Rs, vRs, vTs, phis)]
    # Before integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R()[ii] - list_os[ii].R()) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r()[ii] - list_os[ii].r()) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR()[ii] - list_os[ii].vR()) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT()[ii] - list_os[ii].vT()) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.phi()[ii] - list_os[ii].phi()) < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi()[ii] - list_os[ii].vphi()) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(times)[ii] - list_os[ii].R(times)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times)[ii] - list_os[ii].r(times)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times)[ii] - list_os[ii].vR(times)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times)[ii] - list_os[ii].vT(times)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.phi(times)[ii] - list_os[ii].phi(times)) < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times)[ii] - list_os[ii].vphi(times)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(times[1])[ii] - list_os[ii].R(times[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times[1])[ii] - list_os[ii].r(times[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times[1])[ii] - list_os[ii].vR(times[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times[1])[ii] - list_os[ii].vT(times[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.phi(times[1])[ii] - list_os[ii].phi(times[1])) < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times[1])[ii] - list_os[ii].vphi(times[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes)[ii] - list_os[ii].r(itimes)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes)[ii] - list_os[ii].vR(itimes)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes)[ii] - list_os[ii].vT(itimes)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.phi(itimes)[ii] - list_os[ii].phi(itimes)) < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes)[ii] - list_os[ii].vphi(itimes)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes[1])[ii] - list_os[ii].r(itimes[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes[1])[ii] - list_os[ii].vR(itimes[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes[1])[ii] - list_os[ii].vT(itimes[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.phi(itimes[1])[ii] - list_os[ii].phi(itimes[1])) < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes[1])[ii] - list_os[ii].vphi(itimes[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    with pytest.raises(AttributeError):
        os.z()
    with pytest.raises(AttributeError):
        os.vz()
    return None


# Test that evaluating coordinate functions for integrated orbits works,
# for 3D orbits
def test_coordinate_interpolation_3d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 20
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    os = Orbit(list(zip(Rs, vRs, vTs)))
    list_os = [Orbit([R, vR, vT]) for R, vR, vT in zip(Rs, vRs, vTs)]
    # Before integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R()[ii] - list_os[ii].R()) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r()[ii] - list_os[ii].r()) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR()[ii] - list_os[ii].vR()) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT()[ii] - list_os[ii].vT()) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi()[ii] - list_os[ii].vphi()) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(times)[ii] - list_os[ii].R(times)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times)[ii] - list_os[ii].r(times)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times)[ii] - list_os[ii].vR(times)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times)[ii] - list_os[ii].vT(times)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times)[ii] - list_os[ii].vphi(times)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(times[1])[ii] - list_os[ii].R(times[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times[1])[ii] - list_os[ii].r(times[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times[1])[ii] - list_os[ii].vR(times[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times[1])[ii] - list_os[ii].vT(times[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(times[1])[ii] - list_os[ii].vphi(times[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes)[ii] - list_os[ii].r(itimes)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes)[ii] - list_os[ii].vR(itimes)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes)[ii] - list_os[ii].vT(itimes)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes)[ii] - list_os[ii].vphi(itimes)) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes[1])[ii] - list_os[ii].r(itimes[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes[1])[ii] - list_os[ii].vR(itimes[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes[1])[ii] - list_os[ii].vT(itimes[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vphi(itimes[1])[ii] - list_os[ii].vphi(itimes[1])) < 1e-10
        ), "Evaluating Orbits vphi does not agree with Orbit"
    with pytest.raises(AttributeError):
        os.phi()
    with pytest.raises(AttributeError):
        os.x()
    with pytest.raises(AttributeError):
        os.vx()
    with pytest.raises(AttributeError):
        os.y()
    with pytest.raises(AttributeError):
        os.vy()
    return None


# Test that evaluating coordinate functions for integrated orbits works,
# for 2D orbits
def test_coordinate_interpolation_2d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014, toVerticalPotential

    MWPotential2014 = toVerticalPotential(MWPotential2014, 1.0)
    numpy.random.seed(1)
    nrand = 20
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(zs, vzs)))
    list_os = [Orbit([z, vz]) for z, vz in zip(zs, vzs)]
    # Before integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.x()[ii] - list_os[ii].x()) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx()[ii] - list_os[ii].vx()) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.x(times)[ii] - list_os[ii].x(times)) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(times)[ii] - list_os[ii].vx(times)) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.x(times[1])[ii] - list_os[ii].x(times[1])) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(times[1])[ii] - list_os[ii].vx(times[1])) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.x(itimes)[ii] - list_os[ii].x(itimes)) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(itimes)[ii] - list_os[ii].vx(itimes)) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.x(itimes[1])[ii] - list_os[ii].x(itimes[1])) < 1e-10
        ), "Evaluating Orbits x does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vx(itimes[1])[ii] - list_os[ii].vx(itimes[1])) < 1e-10
        ), "Evaluating Orbits vx does not agree with Orbit"
    return None


# Test interpolation with backwards orbit integration
def test_backinterpolation():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 20
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # Integrate all
    times = numpy.linspace(0.0, -10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
    return None


# Test that evaluating coordinate functions for integrated orbits works for
# a single orbit
def test_coordinate_interpolation_oneorbit():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 1
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # Before integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time() - list_os[ii].time()) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R()[ii] - list_os[ii].R()) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r()[ii] - list_os[ii].r()) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR()[ii] - list_os[ii].vR()) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT()[ii] - list_os[ii].vT()) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z()[ii] - list_os[ii].z()) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz()[ii] - list_os[ii].vz()) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                ((os.phi()[ii] - list_os[ii].phi() + numpy.pi) % (2.0 * numpy.pi))
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    # Test exact times of integration
    for ii in range(nrand):
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time(times) - list_os[ii].time(times)) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R(times)[ii] - list_os[ii].R(times)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times)[ii] - list_os[ii].r(times)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times)[ii] - list_os[ii].vR(times)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times)[ii] - list_os[ii].vT(times)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times)[ii] - list_os[ii].z(times)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times)[ii] - list_os[ii].vz(times)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(times)[ii] - list_os[ii].phi(times) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        # Also a single time in the array ...
        # .time is special, just a single array
        assert numpy.all(
            numpy.fabs(os.time(times[1]) - list_os[ii].time(times[1])) < 1e-10
        ), "Evaluating Orbits time does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.R(times[1])[ii] - list_os[ii].R(times[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(times[1])[ii] - list_os[ii].r(times[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(times[1])[ii] - list_os[ii].vR(times[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(times[1])[ii] - list_os[ii].vT(times[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(times[1])[ii] - list_os[ii].z(times[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(times[1])[ii] - list_os[ii].vz(times[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(times[1])[ii] - list_os[ii].phi(times[1]) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
    # Test actual interpolated
    itimes = times[:-2] + (times[1] - times[0]) / 2.0
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.R(itimes)[ii] - list_os[ii].R(itimes)) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes)[ii] - list_os[ii].r(itimes)) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes)[ii] - list_os[ii].vR(itimes)) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes)[ii] - list_os[ii].vT(itimes)) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes)[ii] - list_os[ii].z(itimes)) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes)[ii] - list_os[ii].vz(itimes)) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(itimes)[ii] - list_os[ii].phi(itimes) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
        # Also a single time in the array ...
        assert numpy.all(
            numpy.fabs(os.R(itimes[1])[ii] - list_os[ii].R(itimes[1])) < 1e-10
        ), "Evaluating Orbits R does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.r(itimes[1])[ii] - list_os[ii].r(itimes[1])) < 1e-10
        ), "Evaluating Orbits r does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vR(itimes[1])[ii] - list_os[ii].vR(itimes[1])) < 1e-10
        ), "Evaluating Orbits vR does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vT(itimes[1])[ii] - list_os[ii].vT(itimes[1])) < 1e-10
        ), "Evaluating Orbits vT does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.z(itimes[1])[ii] - list_os[ii].z(itimes[1])) < 1e-10
        ), "Evaluating Orbits z does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.vz(itimes[1])[ii] - list_os[ii].vz(itimes[1])) < 1e-10
        ), "Evaluating Orbits vz does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(
                (
                    (os.phi(itimes[1])[ii] - list_os[ii].phi(itimes[1]) + numpy.pi)
                    % (2.0 * numpy.pi)
                )
                - numpy.pi
            )
            < 1e-10
        ), "Evaluating Orbits phi does not agree with Orbit"
    return None


# Test that an error is raised when evaluating an orbit outside of the
# integration range
def test_interpolate_outsiderange():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 3
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    with pytest.raises(ValueError) as excinfo:
        os.R(11.0)
    with pytest.raises(ValueError) as excinfo:
        os.R(-1.0)
    # Also for arrays that partially overlap
    with pytest.raises(ValueError) as excinfo:
        os.R(numpy.linspace(5.0, 11.0, 1001))
    with pytest.raises(ValueError) as excinfo:
        os.R(numpy.linspace(-5.0, 5.0, 1001))


def test_output_shape():
    # Test that the output shape is correct and that the shaped output is correct
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = (3, 1, 2)
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vxvv = numpy.rollaxis(numpy.array([Rs, vRs, vTs, zs, vzs, phis]), 0, 4)
    os = Orbit(vxvv)
    list_os = [
        [
            [
                Orbit(
                    [
                        Rs[ii, jj, kk],
                        vRs[ii, jj, kk],
                        vTs[ii, jj, kk],
                        zs[ii, jj, kk],
                        vzs[ii, jj, kk],
                        phis[ii, jj, kk],
                    ]
                )
                for kk in range(nrand[2])
            ]
            for jj in range(nrand[1])
        ]
        for ii in range(nrand[0])
    ]
    # Before integration
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                # .time is special, just a single array
                assert numpy.all(
                    numpy.fabs(os.time() - list_os[ii][jj][kk].time()) < 1e-10
                ), "Evaluating Orbits time does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.R()[ii, jj, kk] - list_os[ii][jj][kk].R()) < 1e-10
                ), "Evaluating Orbits R does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.r()[ii, jj, kk] - list_os[ii][jj][kk].r()) < 1e-10
                ), "Evaluating Orbits r does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vR()[ii, jj, kk] - list_os[ii][jj][kk].vR()) < 1e-10
                ), "Evaluating Orbits vR does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vT()[ii, jj, kk] - list_os[ii][jj][kk].vT()) < 1e-10
                ), "Evaluating Orbits vT does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.z()[ii, jj, kk] - list_os[ii][jj][kk].z()) < 1e-10
                ), "Evaluating Orbits z does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vz()[ii, jj, kk] - list_os[ii][jj][kk].vz()) < 1e-10
                ), "Evaluating Orbits vz does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        (
                            (
                                os.phi()[ii, jj, kk]
                                - list_os[ii][jj][kk].phi()
                                + numpy.pi
                            )
                            % (2.0 * numpy.pi)
                        )
                        - numpy.pi
                    )
                    < 1e-10
                ), "Evaluating Orbits phi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.x()[ii, jj, kk] - list_os[ii][jj][kk].x()) < 1e-10
                ), "Evaluating Orbits x does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.y()[ii, jj, kk] - list_os[ii][jj][kk].y()) < 1e-10
                ), "Evaluating Orbits y does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vx()[ii, jj, kk] - list_os[ii][jj][kk].vx()) < 1e-10
                ), "Evaluating Orbits vx does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vy()[ii, jj, kk] - list_os[ii][jj][kk].vy()) < 1e-10
                ), "Evaluating Orbits vy does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vphi()[ii, jj, kk] - list_os[ii][jj][kk].vphi())
                    < 1e-10
                ), "Evaluating Orbits vphi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ra()[ii, jj, kk] - list_os[ii][jj][kk].ra()) < 1e-10
                ), "Evaluating Orbits ra  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.dec()[ii, jj, kk] - list_os[ii][jj][kk].dec()) < 1e-10
                ), "Evaluating Orbits dec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.dist()[ii, jj, kk] - list_os[ii][jj][kk].dist())
                    < 1e-10
                ), "Evaluating Orbits dist does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ll()[ii, jj, kk] - list_os[ii][jj][kk].ll()) < 1e-10
                ), "Evaluating Orbits ll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.bb()[ii, jj, kk] - list_os[ii][jj][kk].bb()) < 1e-10
                ), "Evaluating Orbits bb  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmra()[ii, jj, kk] - list_os[ii][jj][kk].pmra())
                    < 1e-10
                ), "Evaluating Orbits pmra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmdec()[ii, jj, kk] - list_os[ii][jj][kk].pmdec())
                    < 1e-10
                ), "Evaluating Orbits pmdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmll()[ii, jj, kk] - list_os[ii][jj][kk].pmll())
                    < 1e-10
                ), "Evaluating Orbits pmll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmbb()[ii, jj, kk] - list_os[ii][jj][kk].pmbb())
                    < 1e-10
                ), "Evaluating Orbits pmbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vra()[ii, jj, kk] - list_os[ii][jj][kk].vra()) < 1e-10
                ), "Evaluating Orbits vra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vdec()[ii, jj, kk] - list_os[ii][jj][kk].vdec())
                    < 1e-10
                ), "Evaluating Orbits vdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vll()[ii, jj, kk] - list_os[ii][jj][kk].vll()) < 1e-10
                ), "Evaluating Orbits vll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vbb()[ii, jj, kk] - list_os[ii][jj][kk].vbb()) < 1e-10
                ), "Evaluating Orbits vbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vlos()[ii, jj, kk] - list_os[ii][jj][kk].vlos())
                    < 1e-10
                ), "Evaluating Orbits vlos does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioX()[ii, jj, kk] - list_os[ii][jj][kk].helioX())
                    < 1e-10
                ), "Evaluating Orbits helioX does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioY()[ii, jj, kk] - list_os[ii][jj][kk].helioY())
                    < 1e-10
                ), "Evaluating Orbits helioY does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioZ()[ii, jj, kk] - list_os[ii][jj][kk].helioZ())
                    < 1e-10
                ), "Evaluating Orbits helioZ does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.U()[ii, jj, kk] - list_os[ii][jj][kk].U()) < 1e-10
                ), "Evaluating Orbits U does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.V()[ii, jj, kk] - list_os[ii][jj][kk].V()) < 1e-10
                ), "Evaluating Orbits V does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.W()[ii, jj, kk] - list_os[ii][jj][kk].W()) < 1e-10
                ), "Evaluating Orbits W does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().ra[ii, jj, kk] - list_os[ii][jj][kk].SkyCoord().ra
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().dec[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord().dec
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().distance[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord().distance
                    )
                    .to(u.kpc)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                if _APY3:
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().pm_ra_cosdec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().pm_ra_cosdec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().pm_dec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().pm_dec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().radial_velocity[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().radial_velocity
                        )
                        .to(u.km / u.s)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                list_os[ii][jj][kk].integrate(times, MWPotential2014)
    # Test exact times of integration
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                # .time is special, just a single array
                assert numpy.all(
                    numpy.fabs(os.time(times) - list_os[ii][jj][kk].time(times)) < 1e-10
                ), "Evaluating Orbits time does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.R(times)[ii, jj, kk] - list_os[ii][jj][kk].R(times))
                    < 1e-10
                ), "Evaluating Orbits R does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.r(times)[ii, jj, kk] - list_os[ii][jj][kk].r(times))
                    < 1e-10
                ), "Evaluating Orbits r does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vR(times)[ii, jj, kk] - list_os[ii][jj][kk].vR(times))
                    < 1e-10
                ), "Evaluating Orbits vR does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vT(times)[ii, jj, kk] - list_os[ii][jj][kk].vT(times))
                    < 1e-10
                ), "Evaluating Orbits vT does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.z(times)[ii, jj, kk] - list_os[ii][jj][kk].z(times))
                    < 1e-10
                ), "Evaluating Orbits z does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vz(times)[ii, jj, kk] - list_os[ii][jj][kk].vz(times))
                    < 1e-10
                ), "Evaluating Orbits vz does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        (
                            (
                                os.phi(times)[ii, jj, kk]
                                - list_os[ii][jj][kk].phi(times)
                                + numpy.pi
                            )
                            % (2.0 * numpy.pi)
                        )
                        - numpy.pi
                    )
                    < 1e-10
                ), "Evaluating Orbits phi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.x(times)[ii, jj, kk] - list_os[ii][jj][kk].x(times))
                    < 1e-10
                ), "Evaluating Orbits x does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.y(times)[ii, jj, kk] - list_os[ii][jj][kk].y(times))
                    < 1e-10
                ), "Evaluating Orbits y does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vx(times)[ii, jj, kk] - list_os[ii][jj][kk].vx(times))
                    < 1e-10
                ), "Evaluating Orbits vx does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vy(times)[ii, jj, kk] - list_os[ii][jj][kk].vy(times))
                    < 1e-10
                ), "Evaluating Orbits vy does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vphi(times)[ii, jj, kk] - list_os[ii][jj][kk].vphi(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vphi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ra(times)[ii, jj, kk] - list_os[ii][jj][kk].ra(times))
                    < 1e-10
                ), "Evaluating Orbits ra  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.dec(times)[ii, jj, kk] - list_os[ii][jj][kk].dec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits dec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.dist(times)[ii, jj, kk] - list_os[ii][jj][kk].dist(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits dist does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ll(times)[ii, jj, kk] - list_os[ii][jj][kk].ll(times))
                    < 1e-10
                ), "Evaluating Orbits ll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.bb(times)[ii, jj, kk] - list_os[ii][jj][kk].bb(times))
                    < 1e-10
                ), "Evaluating Orbits bb  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmra(times)[ii, jj, kk] - list_os[ii][jj][kk].pmra(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmdec(times)[ii, jj, kk] - list_os[ii][jj][kk].pmdec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmll(times)[ii, jj, kk] - list_os[ii][jj][kk].pmll(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmbb(times)[ii, jj, kk] - list_os[ii][jj][kk].pmbb(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vra(times)[ii, jj, kk] - list_os[ii][jj][kk].vra(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vdec(times)[ii, jj, kk] - list_os[ii][jj][kk].vdec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vll(times)[ii, jj, kk] - list_os[ii][jj][kk].vll(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vbb(times)[ii, jj, kk] - list_os[ii][jj][kk].vbb(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vlos(times)[ii, jj, kk] - list_os[ii][jj][kk].vlos(times)
                    )
                    < 1e-9
                ), "Evaluating Orbits vlos does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioX(times)[ii, jj, kk] - list_os[ii][jj][kk].helioX(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioX does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioY(times)[ii, jj, kk] - list_os[ii][jj][kk].helioY(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioY does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioZ(times)[ii, jj, kk] - list_os[ii][jj][kk].helioZ(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioZ does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.U(times)[ii, jj, kk] - list_os[ii][jj][kk].U(times))
                    < 1e-10
                ), "Evaluating Orbits U does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.V(times)[ii, jj, kk] - list_os[ii][jj][kk].V(times))
                    < 1e-10
                ), "Evaluating Orbits V does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.W(times)[ii, jj, kk] - list_os[ii][jj][kk].W(times))
                    < 1e-10
                ), "Evaluating Orbits W does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).ra[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).ra
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).dec[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).dec
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).distance[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).distance
                    )
                    .to(u.kpc)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                if _APY3:
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).pm_ra_cosdec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).pm_ra_cosdec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).pm_dec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).pm_dec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).radial_velocity[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).radial_velocity
                        )
                        .to(u.km / u.s)
                        .value
                        < 1e-9
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    return None


def test_output_reshape():
    # Test that the output shape is correct and that the shaped output is correct
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = (3, 1, 2)
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vxvv = numpy.rollaxis(numpy.array([Rs, vRs, vTs, zs, vzs, phis]), 0, 4)
    os = Orbit(vxvv)
    # NOW RESHAPE
    # First try a shape that doesn't work to test the error
    with pytest.raises(ValueError) as excinfo:
        os.reshape((4, 2, 1))
    # then do one that should work and also setup the list of indiv orbits
    # with the new shape
    newshape = (3, 2, 1)
    os.reshape(newshape)
    Rs = Rs.reshape(newshape)
    vRs = vRs.reshape(newshape)
    vTs = vTs.reshape(newshape)
    zs = zs.reshape(newshape)
    vzs = vzs.reshape(newshape)
    phis = phis.reshape(newshape)
    nrand = newshape
    list_os = [
        [
            [
                Orbit(
                    [
                        Rs[ii, jj, kk],
                        vRs[ii, jj, kk],
                        vTs[ii, jj, kk],
                        zs[ii, jj, kk],
                        vzs[ii, jj, kk],
                        phis[ii, jj, kk],
                    ]
                )
                for kk in range(nrand[2])
            ]
            for jj in range(nrand[1])
        ]
        for ii in range(nrand[0])
    ]
    # Before integration
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                # .time is special, just a single array
                assert numpy.all(
                    numpy.fabs(os.time() - list_os[ii][jj][kk].time()) < 1e-10
                ), "Evaluating Orbits time does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.R()[ii, jj, kk] - list_os[ii][jj][kk].R()) < 1e-10
                ), "Evaluating Orbits R does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.r()[ii, jj, kk] - list_os[ii][jj][kk].r()) < 1e-10
                ), "Evaluating Orbits r does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vR()[ii, jj, kk] - list_os[ii][jj][kk].vR()) < 1e-10
                ), "Evaluating Orbits vR does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vT()[ii, jj, kk] - list_os[ii][jj][kk].vT()) < 1e-10
                ), "Evaluating Orbits vT does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.z()[ii, jj, kk] - list_os[ii][jj][kk].z()) < 1e-10
                ), "Evaluating Orbits z does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vz()[ii, jj, kk] - list_os[ii][jj][kk].vz()) < 1e-10
                ), "Evaluating Orbits vz does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        (
                            (
                                os.phi()[ii, jj, kk]
                                - list_os[ii][jj][kk].phi()
                                + numpy.pi
                            )
                            % (2.0 * numpy.pi)
                        )
                        - numpy.pi
                    )
                    < 1e-10
                ), "Evaluating Orbits phi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.x()[ii, jj, kk] - list_os[ii][jj][kk].x()) < 1e-10
                ), "Evaluating Orbits x does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.y()[ii, jj, kk] - list_os[ii][jj][kk].y()) < 1e-10
                ), "Evaluating Orbits y does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vx()[ii, jj, kk] - list_os[ii][jj][kk].vx()) < 1e-10
                ), "Evaluating Orbits vx does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vy()[ii, jj, kk] - list_os[ii][jj][kk].vy()) < 1e-10
                ), "Evaluating Orbits vy does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vphi()[ii, jj, kk] - list_os[ii][jj][kk].vphi())
                    < 1e-10
                ), "Evaluating Orbits vphi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ra()[ii, jj, kk] - list_os[ii][jj][kk].ra()) < 1e-10
                ), "Evaluating Orbits ra  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.dec()[ii, jj, kk] - list_os[ii][jj][kk].dec()) < 1e-10
                ), "Evaluating Orbits dec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.dist()[ii, jj, kk] - list_os[ii][jj][kk].dist())
                    < 1e-10
                ), "Evaluating Orbits dist does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ll()[ii, jj, kk] - list_os[ii][jj][kk].ll()) < 1e-10
                ), "Evaluating Orbits ll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.bb()[ii, jj, kk] - list_os[ii][jj][kk].bb()) < 1e-10
                ), "Evaluating Orbits bb  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmra()[ii, jj, kk] - list_os[ii][jj][kk].pmra())
                    < 1e-10
                ), "Evaluating Orbits pmra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmdec()[ii, jj, kk] - list_os[ii][jj][kk].pmdec())
                    < 1e-10
                ), "Evaluating Orbits pmdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmll()[ii, jj, kk] - list_os[ii][jj][kk].pmll())
                    < 1e-10
                ), "Evaluating Orbits pmll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.pmbb()[ii, jj, kk] - list_os[ii][jj][kk].pmbb())
                    < 1e-10
                ), "Evaluating Orbits pmbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vra()[ii, jj, kk] - list_os[ii][jj][kk].vra()) < 1e-10
                ), "Evaluating Orbits vra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vdec()[ii, jj, kk] - list_os[ii][jj][kk].vdec())
                    < 1e-10
                ), "Evaluating Orbits vdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vll()[ii, jj, kk] - list_os[ii][jj][kk].vll()) < 1e-10
                ), "Evaluating Orbits vll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vbb()[ii, jj, kk] - list_os[ii][jj][kk].vbb()) < 1e-10
                ), "Evaluating Orbits vbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vlos()[ii, jj, kk] - list_os[ii][jj][kk].vlos())
                    < 1e-10
                ), "Evaluating Orbits vlos does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioX()[ii, jj, kk] - list_os[ii][jj][kk].helioX())
                    < 1e-10
                ), "Evaluating Orbits helioX does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioY()[ii, jj, kk] - list_os[ii][jj][kk].helioY())
                    < 1e-10
                ), "Evaluating Orbits helioY does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.helioZ()[ii, jj, kk] - list_os[ii][jj][kk].helioZ())
                    < 1e-10
                ), "Evaluating Orbits helioZ does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.U()[ii, jj, kk] - list_os[ii][jj][kk].U()) < 1e-10
                ), "Evaluating Orbits U does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.V()[ii, jj, kk] - list_os[ii][jj][kk].V()) < 1e-10
                ), "Evaluating Orbits V does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.W()[ii, jj, kk] - list_os[ii][jj][kk].W()) < 1e-10
                ), "Evaluating Orbits W does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().ra[ii, jj, kk] - list_os[ii][jj][kk].SkyCoord().ra
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().dec[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord().dec
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord().distance[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord().distance
                    )
                    .to(u.kpc)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                if _APY3:
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().pm_ra_cosdec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().pm_ra_cosdec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().pm_dec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().pm_dec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord().radial_velocity[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord().radial_velocity
                        )
                        .to(u.km / u.s)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                list_os[ii][jj][kk].integrate(times, MWPotential2014)
    # Test exact times of integration
    for ii in range(nrand[0]):
        for jj in range(nrand[1]):
            for kk in range(nrand[2]):
                # .time is special, just a single array
                assert numpy.all(
                    numpy.fabs(os.time(times) - list_os[ii][jj][kk].time(times)) < 1e-10
                ), "Evaluating Orbits time does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.R(times)[ii, jj, kk] - list_os[ii][jj][kk].R(times))
                    < 1e-10
                ), "Evaluating Orbits R does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.r(times)[ii, jj, kk] - list_os[ii][jj][kk].r(times))
                    < 1e-10
                ), "Evaluating Orbits r does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vR(times)[ii, jj, kk] - list_os[ii][jj][kk].vR(times))
                    < 1e-10
                ), "Evaluating Orbits vR does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vT(times)[ii, jj, kk] - list_os[ii][jj][kk].vT(times))
                    < 1e-10
                ), "Evaluating Orbits vT does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.z(times)[ii, jj, kk] - list_os[ii][jj][kk].z(times))
                    < 1e-10
                ), "Evaluating Orbits z does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vz(times)[ii, jj, kk] - list_os[ii][jj][kk].vz(times))
                    < 1e-10
                ), "Evaluating Orbits vz does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        (
                            (
                                os.phi(times)[ii, jj, kk]
                                - list_os[ii][jj][kk].phi(times)
                                + numpy.pi
                            )
                            % (2.0 * numpy.pi)
                        )
                        - numpy.pi
                    )
                    < 1e-10
                ), "Evaluating Orbits phi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.x(times)[ii, jj, kk] - list_os[ii][jj][kk].x(times))
                    < 1e-10
                ), "Evaluating Orbits x does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.y(times)[ii, jj, kk] - list_os[ii][jj][kk].y(times))
                    < 1e-10
                ), "Evaluating Orbits y does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vx(times)[ii, jj, kk] - list_os[ii][jj][kk].vx(times))
                    < 1e-10
                ), "Evaluating Orbits vx does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.vy(times)[ii, jj, kk] - list_os[ii][jj][kk].vy(times))
                    < 1e-10
                ), "Evaluating Orbits vy does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vphi(times)[ii, jj, kk] - list_os[ii][jj][kk].vphi(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vphi does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ra(times)[ii, jj, kk] - list_os[ii][jj][kk].ra(times))
                    < 1e-10
                ), "Evaluating Orbits ra  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.dec(times)[ii, jj, kk] - list_os[ii][jj][kk].dec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits dec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.dist(times)[ii, jj, kk] - list_os[ii][jj][kk].dist(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits dist does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.ll(times)[ii, jj, kk] - list_os[ii][jj][kk].ll(times))
                    < 1e-10
                ), "Evaluating Orbits ll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.bb(times)[ii, jj, kk] - list_os[ii][jj][kk].bb(times))
                    < 1e-10
                ), "Evaluating Orbits bb  does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmra(times)[ii, jj, kk] - list_os[ii][jj][kk].pmra(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmdec(times)[ii, jj, kk] - list_os[ii][jj][kk].pmdec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmll(times)[ii, jj, kk] - list_os[ii][jj][kk].pmll(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.pmbb(times)[ii, jj, kk] - list_os[ii][jj][kk].pmbb(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits pmbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vra(times)[ii, jj, kk] - list_os[ii][jj][kk].vra(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vra does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vdec(times)[ii, jj, kk] - list_os[ii][jj][kk].vdec(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vdec does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vll(times)[ii, jj, kk] - list_os[ii][jj][kk].vll(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vll does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vbb(times)[ii, jj, kk] - list_os[ii][jj][kk].vbb(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits vbb does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.vlos(times)[ii, jj, kk] - list_os[ii][jj][kk].vlos(times)
                    )
                    < 1e-9
                ), "Evaluating Orbits vlos does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioX(times)[ii, jj, kk] - list_os[ii][jj][kk].helioX(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioX does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioY(times)[ii, jj, kk] - list_os[ii][jj][kk].helioY(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioY does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.helioZ(times)[ii, jj, kk] - list_os[ii][jj][kk].helioZ(times)
                    )
                    < 1e-10
                ), "Evaluating Orbits helioZ does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.U(times)[ii, jj, kk] - list_os[ii][jj][kk].U(times))
                    < 1e-10
                ), "Evaluating Orbits U does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.V(times)[ii, jj, kk] - list_os[ii][jj][kk].V(times))
                    < 1e-10
                ), "Evaluating Orbits V does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(os.W(times)[ii, jj, kk] - list_os[ii][jj][kk].W(times))
                    < 1e-10
                ), "Evaluating Orbits W does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).ra[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).ra
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).dec[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).dec
                    )
                    .to(u.deg)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                assert numpy.all(
                    numpy.fabs(
                        os.SkyCoord(times).distance[ii, jj, kk]
                        - list_os[ii][jj][kk].SkyCoord(times).distance
                    )
                    .to(u.kpc)
                    .value
                    < 1e-10
                ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                if _APY3:
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).pm_ra_cosdec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).pm_ra_cosdec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).pm_dec[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).pm_dec
                        )
                        .to(u.mas / u.yr)
                        .value
                        < 1e-10
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
                    assert numpy.all(
                        numpy.fabs(
                            os.SkyCoord(times).radial_velocity[ii, jj, kk]
                            - list_os[ii][jj][kk].SkyCoord(times).radial_velocity
                        )
                        .to(u.km / u.s)
                        .value
                        < 1e-9
                    ), "Evaluating Orbits SkyCoord does not agree with Orbit"
    return None


def test_output_specialshapes():
    # Test that the output shape is correct and that the shaped output is correct, for 'special' inputs (single objects, ...)
    from galpy.orbit import Orbit

    # vxvv= list of [R,vR,vT,z,...] should be shape == () and scalar output
    os = Orbit([1.0, 0.1, 1.0, 0.1, 0.0, 0.1])
    assert os.shape == (), "Shape of Orbits with list of [R,vR,...] input is not empty"
    assert (
        numpy.ndim(os.R()) == 0
    ), "Orbits with list of [R,vR,...] input does not return scalar"
    # Similar for list [ra,dec,...]
    os = Orbit([1.0, 0.1, 1.0, 0.1, 0.0, 0.1], radec=True)
    assert (
        os.shape == ()
    ), "Shape of Orbits with list of [ra,dec,...] input is not empty"
    assert (
        numpy.ndim(os.R()) == 0
    ), "Orbits with list of [ra,dec,...] input does not return scalar"
    # Also with units
    os = Orbit(
        [
            1.0 * u.deg,
            0.1 * u.rad,
            1.0 * u.pc,
            0.1 * u.mas / u.yr,
            0.0 * u.arcsec / u.yr,
            0.1 * u.pc / u.Myr,
        ],
        radec=True,
    )
    assert (
        os.shape == ()
    ), "Shape of Orbits with list of [ra,dec,...] w/units input is not empty"
    assert (
        numpy.ndim(os.R()) == 0
    ), "Orbits with list of [ra,dec,...] w/units input does not return scalar"
    # Also from_name
    os = Orbit.from_name("LMC")
    assert os.shape == (), "Shape of Orbits with from_name single object is not empty"
    assert (
        numpy.ndim(os.R()) == 0
    ), "Orbits with from_name single object does not return scalar"
    # vxvv= list of list of [R,vR,vT,z,...] should be shape == (1,) and array output
    os = Orbit([[1.0, 0.1, 1.0, 0.1, 0.0, 0.1]])
    assert os.shape == (
        1,
    ), "Shape of Orbits with list of list of [R,vR,...] input is not (1,)"
    assert (
        numpy.ndim(os.R()) == 1
    ), "Orbits with list of list of [R,vR,...] input does not return array"
    # vxvv= array of [R,vR,vT,z,...] should be shape == () and scalar output
    os = Orbit(numpy.array([1.0, 0.1, 1.0, 0.1, 0.0, 0.1]))
    assert os.shape == (), "Shape of Orbits with array of [R,vR,...] input is not empty"
    assert (
        numpy.ndim(os.R()) == 0
    ), "Orbits with array of [R,vR,...] input does not return scalar"
    if _APY3:
        # vxvv= single SkyCoord should be shape == () and scalar output
        co = apycoords.SkyCoord(
            ra=1.0 * u.deg,
            dec=0.5 * u.rad,
            distance=2.0 * u.kpc,
            pm_ra_cosdec=-0.1 * u.mas / u.yr,
            pm_dec=10.0 * u.mas / u.yr,
            radial_velocity=10.0 * u.km / u.s,
            frame="icrs",
        )
        os = Orbit(co)
        assert (
            os.shape == co.shape
        ), "Shape of Orbits with SkyCoord does not agree with shape of SkyCoord"
        # vxvv= single SkyCoord, but as array should be shape == (1,) and array output
        s = numpy.ones(1)
        co = apycoords.SkyCoord(
            ra=s * 1.0 * u.deg,
            dec=s * 0.5 * u.rad,
            distance=s * 2.0 * u.kpc,
            pm_ra_cosdec=-0.1 * u.mas / u.yr * s,
            pm_dec=10.0 * u.mas / u.yr * s,
            radial_velocity=10.0 * u.km / u.s * s,
            frame="icrs",
        )
        os = Orbit(co)
        assert (
            os.shape == co.shape
        ), "Shape of Orbits with SkyCoord does not agree with shape of SkyCoord"
        # vxvv= None should be shape == (1,) and array output
        os = Orbit()
        assert os.shape == (), "Shape of Orbits with vxvv=None input is not empty"
        assert (
            numpy.ndim(os.R()) == 0
        ), "Orbits with with vxvv=None input does not return scalar"
    return None


def test_call_issue256():
    # Same as for Orbit instances: non-integrated orbit with t=/=0 should return error
    from galpy.orbit import Orbit

    o = Orbit(vxvv=[[5.0, -1.0, 0.8, 3, -0.1, 0]])
    # no integration of the orbit
    with pytest.raises(ValueError) as excinfo:
        o.R(30)
    return None


# Test that the energy, angular momentum, and Jacobi functions work as expected
def test_energy_jacobi_angmom():
    from galpy.orbit import Orbit

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    # 6D
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    _check_energy_jacobi_angmom(os, list_os)
    # 5D
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs)))
    list_os = [
        Orbit([R, vR, vT, z, vz]) for R, vR, vT, z, vz in zip(Rs, vRs, vTs, zs, vzs)
    ]
    _check_energy_jacobi_angmom(os, list_os)
    # 4D
    os = Orbit(list(zip(Rs, vRs, vTs, phis)))
    list_os = [Orbit([R, vR, vT, phi]) for R, vR, vT, phi in zip(Rs, vRs, vTs, phis)]
    _check_energy_jacobi_angmom(os, list_os)
    # 3D
    os = Orbit(list(zip(Rs, vRs, vTs)))
    list_os = [Orbit([R, vR, vT]) for R, vR, vT in zip(Rs, vRs, vTs)]
    _check_energy_jacobi_angmom(os, list_os)
    # 2D
    os = Orbit(list(zip(zs, vzs)))
    list_os = [Orbit([z, vz]) for z, vz in zip(zs, vzs)]
    _check_energy_jacobi_angmom(os, list_os)
    return None


def _check_energy_jacobi_angmom(os, list_os):
    nrand = len(os)
    from galpy.potential import (
        DehnenBarPotential,
        DoubleExponentialDiskPotential,
        MWPotential2014,
        SpiralArmsPotential,
    )

    sp = SpiralArmsPotential()
    dp = DehnenBarPotential()
    lp = DoubleExponentialDiskPotential(normalize=1.0)
    if os.dim() == 1:
        from galpy.potential import toVerticalPotential

        MWPotential2014 = toVerticalPotential(MWPotential2014, 1.0)
        lp = toVerticalPotential(lp, 1.0)
    # Before integration
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(
                os.E(pot=MWPotential2014)[ii] / list_os[ii].E(pot=MWPotential2014) - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits E does not agree with Orbit"
        if os.dim() == 3:
            assert numpy.all(
                numpy.fabs(
                    os.ER(pot=MWPotential2014)[ii] / list_os[ii].ER(pot=MWPotential2014)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits ER does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.Ez(pot=MWPotential2014)[ii] / list_os[ii].Ez(pot=MWPotential2014)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Ez does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.L()[ii] / list_os[ii].L() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits L does not agree with Orbit"
        if os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Lz()[ii] / list_os[ii].Lz() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits Lz does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=MWPotential2014)[ii]
                    / list_os[ii].Jacobi(pot=MWPotential2014)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            # Also explicitly set OmegaP
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=MWPotential2014, OmegaP=0.6)[ii]
                    / list_os[ii].Jacobi(pot=MWPotential2014, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
    # Potential for which array evaluation definitely does not work
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.E(pot=lp)[ii] / list_os[ii].E(pot=lp) - 1.0) < 10.0**-10.0
        ), "Evaluating Orbits E does not agree with Orbit"
        if os.dim() == 3:
            assert numpy.all(
                numpy.fabs(os.ER(pot=lp)[ii] / list_os[ii].ER(pot=lp) - 1.0)
                < 10.0**-10.0
            ), "Evaluating Orbits ER does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(os.Ez(pot=lp)[ii] / list_os[ii].Ez(pot=lp) - 1.0)
                < 10.0**-10.0
            ), "Evaluating Orbits Ez does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.L()[ii] / list_os[ii].L() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits L does not agree with Orbit"
        if os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Lz()[ii] / list_os[ii].Lz() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits Lz does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Jacobi(pot=lp)[ii] / list_os[ii].Jacobi(pot=lp) - 1.0)
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            # Also explicitly set OmegaP
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=lp, OmegaP=0.6)[ii]
                    / list_os[ii].Jacobi(pot=lp, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
        if os.phasedim() == 6:
            # Also in 3D
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=lp, OmegaP=[0.0, 0.0, 0.6])[ii]
                    / list_os[ii].Jacobi(pot=lp, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=lp, OmegaP=numpy.array([0.0, 0.0, 0.6]))[ii]
                    / list_os[ii].Jacobi(pot=lp, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
    # o.E before integration gives AttributeError
    with pytest.raises(AttributeError):
        os.E()
    with pytest.raises(AttributeError):
        os.Jacobi()
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    for ii in range(nrand):
        # Don't have to specify the potential or set to None
        assert numpy.all(
            numpy.fabs(
                os.E(times)[ii] / list_os[ii].E(times, pot=MWPotential2014) - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits E does not agree with Orbit"
        if os.dim() == 3:
            assert numpy.all(
                numpy.fabs(
                    os.ER(times, pot=None)[ii]
                    / list_os[ii].ER(times, pot=MWPotential2014)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits ER does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.Ez(times)[ii] / list_os[ii].Ez(times, pot=MWPotential2014) - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Ez does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.L(times)[ii] / list_os[ii].L(times) - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits L does not agree with Orbit"
        if os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Lz(times)[ii] / list_os[ii].Lz(times) - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits Lz does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Jacobi(times)[ii] / list_os[ii].Jacobi(times) - 1.0)
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            # Also explicitly set OmegaP
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(times, pot=MWPotential2014, OmegaP=0.6)[ii]
                    / list_os[ii].Jacobi(times, pot=MWPotential2014, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
        if os.phasedim() == 6:
            # Also in 3D
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(times, pot=MWPotential2014, OmegaP=[0.0, 0.0, 0.6])[ii]
                    / list_os[ii].Jacobi(times, pot=MWPotential2014, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(
                        times, pot=MWPotential2014, OmegaP=numpy.array([0.0, 0.0, 0.6])
                    )[ii]
                    / list_os[ii].Jacobi(times, pot=MWPotential2014, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
    # Don't do non-axi for odd-D Orbits or 1D
    if os.phasedim() % 2 == 1 or os.dim() == 1:
        return None
    # Add bar and spiral
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(
                os.E(pot=MWPotential2014 + dp + sp)[ii]
                / list_os[ii].E(pot=MWPotential2014 + dp + sp)
                - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits E does not agree with Orbit"
        if os.dim() == 3:
            assert numpy.all(
                numpy.fabs(
                    os.ER(pot=MWPotential2014 + dp + sp)[ii]
                    / list_os[ii].ER(pot=MWPotential2014 + dp + sp)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits ER does not agree with Orbit"
            assert numpy.all(
                numpy.fabs(
                    os.Ez(pot=MWPotential2014 + dp + sp)[ii]
                    / list_os[ii].Ez(pot=MWPotential2014 + dp + sp)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Ez does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.L()[ii] / list_os[ii].L() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits L does not agree with Orbit"
        if os.dim() != 1:
            assert numpy.all(
                numpy.fabs(os.Lz()[ii] / list_os[ii].Lz() - 1.0) < 10.0**-10.0
            ), "Evaluating Orbits Lz does not agree with Orbit"
        if os.phasedim() % 2 == 0 and os.dim() != 1:
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=MWPotential2014 + dp + sp)[ii]
                    / list_os[ii].Jacobi(pot=MWPotential2014 + dp + sp)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
            # Also explicitly set OmegaP
            assert numpy.all(
                numpy.fabs(
                    os.Jacobi(pot=MWPotential2014 + dp + sp, OmegaP=0.6)[ii]
                    / list_os[ii].Jacobi(pot=MWPotential2014 + dp + sp, OmegaP=0.6)
                    - 1.0
                )
                < 10.0**-10.0
            ), "Evaluating Orbits Jacobi does not agree with Orbit"
    return None


# Test that L cannot be computed for (a) linearOrbits and (b) 5D orbits
def test_angmom_errors():
    from galpy.orbit import Orbit

    o = Orbit([[1.0, 0.1]])
    with pytest.raises(AttributeError):
        o.L()
    o = Orbit([[1.0, 0.1, 1.1, 0.1, -0.2]])
    with pytest.raises(AttributeError):
        o.L()
    return None


# Test that we can still get outputs when there aren't enough points for an actual interpolation
# Test whether Orbits evaluation methods sound warning when called with
# unitless time when orbit is integrated with unitfull times
def test_orbits_method_integrate_t_asQuantity_warning():
    from astropy import units
    from test_orbit import check_integrate_t_asQuantity_warning

    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    # Setup and integrate orbit
    ts = numpy.linspace(0.0, 10.0, 1001) * units.Gyr
    o = Orbit([[1.1, 0.1, 1.1, 0.1, 0.1, 0.2], [1.1, 0.1, 1.1, 0.1, 0.1, 0.2]])
    o.integrate(ts, MWPotential2014)
    # Now check
    check_integrate_t_asQuantity_warning(o, "R")
    return None


# Test new orbits formed from __call__
def test_newOrbits():
    from galpy.orbit import Orbit

    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.0, 0.0], [1.1, 0.3, 0.9, -0.2, 0.3, 2.0]])
    ts = numpy.linspace(0.0, 1.0, 21)  # v. quick orbit integration
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o.integrate(ts, lp)
    no = o(ts[-1])  # new Orbits
    assert numpy.all(
        no.R() == o.R(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(
        no.vR() == o.vR(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(
        no.vT() == o.vT(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(
        no.z() == o.z(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(
        no.vz() == o.vz(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(
        no.phi() == o.phi(ts[-1])
    ), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert (
        not no._roSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert (
        not no._voSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    # Also test this for multiple time outputs
    nos = o(ts[-2:])  # new Orbits
    assert numpy.all(
        numpy.fabs(nos.R() - o.R(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(
        numpy.fabs(nos.vR() - o.vR(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(
        numpy.fabs(nos.vT() - o.vT(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(
        numpy.fabs(nos.z() - o.z(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(
        numpy.fabs(nos.vz() - o.vz(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(
        numpy.fabs(nos.phi() - o.phi(ts[-2:])) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert (
        not nos._roSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert (
        not nos._voSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    return None


# Test new orbits formed from __call__, before integration
def test_newOrbit_b4integration():
    from galpy.orbit import Orbit

    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.0, 0.0], [1.1, 0.3, 0.9, -0.2, 0.3, 2.0]])
    no = o()  # New Orbits formed before integration
    assert numpy.all(
        numpy.fabs(no.R() - o.R()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct R"
    assert numpy.all(
        numpy.fabs(no.vR() - o.vR()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vR"
    assert numpy.all(
        numpy.fabs(no.vT() - o.vT()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vT"
    assert numpy.all(
        numpy.fabs(no.z() - o.z()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct z"
    assert numpy.all(
        numpy.fabs(no.vz() - o.vz()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct vz"
    assert numpy.all(
        numpy.fabs(no.phi() - o.phi()) < 10.0**-10.0
    ), "New Orbits formed from calling an old orbit does not have the correct phi"
    assert (
        not no._roSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    assert (
        not no._voSet
    ), "New Orbits formed from calling an old orbit does not have the correct roSet"
    return None


# Test that we can still get outputs when there aren't enough points for an actual interpolation
def test_badinterpolation():
    from galpy.orbit import Orbit

    o = Orbit([[1.0, 0.1, 1.1, 0.1, 0.0, 0.0], [1.1, 0.3, 0.9, -0.2, 0.3, 2.0]])
    ts = numpy.linspace(
        0.0, 1.0, 3
    )  # v. quick orbit integration, w/ not enough points for interpolation
    lp = potential.LogarithmicHaloPotential(normalize=1.0)
    o.integrate(ts, lp)
    no = o(ts[-1])  # new orbit
    assert numpy.all(
        no.R() == o.R(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.all(
        no.vR() == o.vR(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.all(
        no.vT() == o.vT(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.all(
        no.z() == o.z(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.all(
        no.vz() == o.vz(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.all(
        no.phi() == o.phi(ts[-1])
    ), "New orbit formed from calling an old orbit does not have the correct phi"
    assert (
        not no._roSet
    ), "New orbit formed from calling an old orbit does not have the correct roSet"
    assert (
        not no._voSet
    ), "New orbit formed from calling an old orbit does not have the correct roSet"
    # Also test this for multiple time outputs
    nos = o(ts[-2:])  # new Orbits
    # First t
    assert numpy.all(
        numpy.fabs(nos.R() - o.R(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct R"
    assert numpy.all(
        numpy.fabs(nos.vR() - o.vR(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct vR"
    assert numpy.all(
        numpy.fabs(nos.vT() - o.vT(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct vT"
    assert numpy.all(
        numpy.fabs(nos.z() - o.z(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct z"
    assert numpy.all(
        numpy.fabs(nos.vz() - o.vz(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct vz"
    assert numpy.all(
        numpy.fabs(nos.phi() - o.phi(ts[-2:])) < 10.0**-10.0
    ), "New orbit formed from calling an old orbit does not have the correct phi"
    assert (
        not nos._roSet
    ), "New orbit formed from calling an old orbit does not have the correct roSet"
    assert (
        not nos._voSet
    ), "New orbit formed from calling an old orbit does not have the correct roSet"
    # Try point in between, shouldn't work
    with pytest.raises(LookupError) as exc_info:
        no = o(0.6)
    return None


# Check plotting routines
def test_plotting():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    o = Orbit(
        [Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0]), Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0])]
    )
    oa = Orbit([Orbit([1.0, 0.1, 1.1, 0.1, 0.2]), Orbit([1.0, 0.1, 1.1, 0.1, 0.2])])
    # Interesting shape
    os = Orbit(
        numpy.array(
            [
                [[1.0, 0.1, 1.1, -0.1, -0.2, 0.0], [1.0, 0.2, 1.2, 0.0, -0.1, 1.0]],
                [[1.0, -0.2, 0.9, 0.2, 0.2, 2.0], [1.2, -0.4, 1.1, -0.1, 0.0, -2.0]],
                [[1.0, 0.2, 0.9, 0.3, -0.2, 0.1], [1.2, 0.4, 1.1, -0.2, 0.05, 4.0]],
            ]
        )
    )
    times = numpy.linspace(0.0, 7.0, 251)
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.8)
    # Integrate
    o.integrate(times, lp)
    oa.integrate(times, lp)
    os.integrate(times, lp)
    # Some plots
    # Energy
    o.plotE()
    o.plotE(normed=True)
    o.plotE(pot=lp, d1="R")
    o.plotE(pot=lp, d1="vR")
    o.plotE(pot=lp, d1="vT")
    o.plotE(pot=lp, d1="z")
    o.plotE(pot=lp, d1="vz")
    o.plotE(pot=lp, d1="phi")
    oa.plotE()
    oa.plotE(pot=lp, d1="R")
    oa.plotE(pot=lp, d1="vR")
    oa.plotE(pot=lp, d1="vT")
    oa.plotE(pot=lp, d1="z")
    oa.plotE(pot=lp, d1="vz")
    os.plotE()
    os.plotE(pot=lp, d1="R")
    os.plotE(pot=lp, d1="vR")
    os.plotE(pot=lp, d1="vT")
    os.plotE(pot=lp, d1="z")
    os.plotE(pot=lp, d1="vz")
    # Vertical energy
    o.plotEz()
    o.plotEz(normed=True)
    o.plotEz(pot=lp, d1="R")
    o.plotEz(pot=lp, d1="vR")
    o.plotEz(pot=lp, d1="vT")
    o.plotEz(pot=lp, d1="z")
    o.plotEz(pot=lp, d1="vz")
    o.plotEz(pot=lp, d1="phi")
    oa.plotEz()
    oa.plotEz(normed=True)
    oa.plotEz(pot=lp, d1="R")
    oa.plotEz(pot=lp, d1="vR")
    oa.plotEz(pot=lp, d1="vT")
    oa.plotEz(pot=lp, d1="z")
    oa.plotEz(pot=lp, d1="vz")
    os.plotEz()
    os.plotEz(normed=True)
    os.plotEz(pot=lp, d1="R")
    os.plotEz(pot=lp, d1="vR")
    os.plotEz(pot=lp, d1="vT")
    os.plotEz(pot=lp, d1="z")
    os.plotEz(pot=lp, d1="vz")
    # Radial energy
    o.plotER()
    o.plotER(normed=True)
    # Radial energy
    oa.plotER()
    oa.plotER(normed=True)
    os.plotER()
    os.plotER(normed=True)
    # Jacobi
    o.plotJacobi()
    o.plotJacobi(normed=True)
    o.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    o.plotJacobi(pot=lp, d1="vR")
    o.plotJacobi(pot=lp, d1="vT")
    o.plotJacobi(pot=lp, d1="z")
    o.plotJacobi(pot=lp, d1="vz")
    o.plotJacobi(pot=lp, d1="phi")
    oa.plotJacobi()
    oa.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    oa.plotJacobi(pot=lp, d1="vR")
    oa.plotJacobi(pot=lp, d1="vT")
    oa.plotJacobi(pot=lp, d1="z")
    oa.plotJacobi(pot=lp, d1="vz")
    os.plotJacobi()
    os.plotJacobi(pot=lp, d1="R", OmegaP=1.0)
    os.plotJacobi(pot=lp, d1="vR")
    os.plotJacobi(pot=lp, d1="vT")
    os.plotJacobi(pot=lp, d1="z")
    os.plotJacobi(pot=lp, d1="vz")
    # Plot the orbit itself
    o.plot()  # defaults
    oa.plot()
    os.plot()
    o.plot(d1="vR")
    o.plotR()
    o.plotvR(d1="vT")
    o.plotvT(d1="z")
    o.plotz(d1="vz")
    o.plotvz(d1="phi")
    o.plotphi(d1="vR")
    o.plotx(d1="vx")
    o.plotvx(d1="y")
    o.ploty(d1="vy")
    o.plotvy(d1="x")
    # Remaining attributes
    o.plot(d1="ra", d2="dec")
    o.plot(d2="ra", d1="dec")
    o.plot(d1="pmra", d2="pmdec")
    o.plot(d2="pmra", d1="pmdec")
    o.plot(d1="ll", d2="bb")
    o.plot(d2="ll", d1="bb")
    o.plot(d1="pmll", d2="pmbb")
    o.plot(d2="pmll", d1="pmbb")
    o.plot(d1="vlos", d2="dist")
    o.plot(d2="vlos", d1="dist")
    o.plot(d1="helioX", d2="U")
    o.plot(d2="helioX", d1="U")
    o.plot(d1="helioY", d2="V")
    o.plot(d2="helioY", d1="V")
    o.plot(d1="helioZ", d2="W")
    o.plot(d2="helioZ", d1="W")
    o.plot(d2="r", d1="R")
    o.plot(d2="R", d1="r")
    # Some more energies etc.
    o.plot(d1="E", d2="R")
    o.plot(d1="Enorm", d2="R")
    o.plot(d1="Ez", d2="R")
    o.plot(d1="Eznorm", d2="R")
    o.plot(d1="ER", d2="R")
    o.plot(d1="ERnorm", d2="R")
    o.plot(d1="Jacobi", d2="R")
    o.plot(d1="Jacobinorm", d2="R")
    # callables
    o.plot(d1=lambda t: t, d2=lambda t: o.R(t))
    # Expressions
    o.plot(d1="t", d2="r*R/vR")
    os.plot(d1="t", d2="r*R/vR")
    return None


def test_plotSOS():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    # 3D
    o = Orbit(
        [Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0]), Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0])]
    )
    pot = potential.MWPotential2014
    o.plotSOS(pot)
    o.plotSOS(pot, use_physical=True, ro=8.0, vo=220.0)
    # 2D
    o = Orbit([Orbit([1.0, 0.1, 1.1, 2.0]), Orbit([1.0, 0.1, 1.1, 2.0])])
    pot = LogarithmicHaloPotential(normalize=1.0).toPlanar()
    o.plotSOS(pot)
    o.plotSOS(pot, use_physical=True, ro=8.0, vo=220.0)
    return None


def test_plotBruteSOS():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    # 3D
    o = Orbit(
        [Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0]), Orbit([1.0, 0.1, 1.1, 0.1, 0.2, 2.0])]
    )
    pot = potential.MWPotential2014
    o.plotBruteSOS(numpy.linspace(0.0, 20.0 * numpy.pi, 100001), pot)
    o.plotBruteSOS(
        numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
        pot,
        use_physical=True,
        ro=8.0,
        vo=220.0,
    )
    # 2D
    o = Orbit([Orbit([1.0, 0.1, 1.1, 2.0]), Orbit([1.0, 0.1, 1.1, 2.0])])
    pot = LogarithmicHaloPotential(normalize=1.0).toPlanar()
    o.plotBruteSOS(numpy.linspace(0.0, 20.0 * numpy.pi, 100001), pot)
    o.plotBruteSOS(
        numpy.linspace(0.0, 20.0 * numpy.pi, 100001),
        pot,
        use_physical=True,
        ro=8.0,
        vo=220.0,
    )
    return None


def test_integrate_method_warning():
    """Test Orbits.integrate raises an error if method is invalid"""
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit(
        [
            Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0]),
            Orbit(vxvv=[1.0, 0.1, 0.1, 0.5, 0.1, 0.0]),
        ]
    )
    t = numpy.arange(0.0, 10.0, 0.001)
    with pytest.raises(ValueError):
        o.integrate(t, MWPotential2014, method="rk4")


# Test that fallback onto Python integrators works for Orbits
def test_integrate_Cfallback_symplec():
    from test_potential import BurkertPotentialNoC

    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0]),
        Orbit([0.9, 0.3, 1.0]),
        Orbit([1.2, -0.3, 0.7]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits
    pot = BurkertPotentialNoC()
    pot.normalize(1.0)
    orbits.integrate(times, pot, method="symplec4_c")
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, pot, method="symplec4_c")
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


def test_integrate_Cfallback_nonsymplec():
    from test_potential import BurkertPotentialNoC

    from galpy.orbit import Orbit

    times = numpy.linspace(0.0, 10.0, 1001)
    orbits_list = [
        Orbit([1.0, 0.1, 1.0]),
        Orbit([0.9, 0.3, 1.0]),
        Orbit([1.2, -0.3, 0.7]),
    ]
    orbits = Orbit(orbits_list)
    # Integrate as Orbits
    pot = BurkertPotentialNoC()
    pot.normalize(1.0)
    orbits.integrate(times, pot, method="dop853_c")
    # Integrate as multiple Orbits
    for o in orbits_list:
        o.integrate(times, pot, method="dop853_c")
    # Compare
    for ii in range(len(orbits)):
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].R(times) - orbits.R(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vR(times) - orbits.vR(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
        assert (
            numpy.amax(numpy.fabs(orbits_list[ii].vT(times) - orbits.vT(times)[ii]))
            < 1e-10
        ), "Integration of multiple orbits as Orbits does not agree with integrating multiple orbits"
    return None


# Test flippingg an orbit
def setup_orbits_flip(tp, ro, vo, zo, solarmotion, axi=False):
    from galpy.orbit import Orbit

    if isinstance(tp, potential.linearPotential):
        o = Orbit(
            [[1.0, 1.0], [0.2, -0.3]], ro=ro, vo=vo, zo=zo, solarmotion=solarmotion
        )
    elif isinstance(tp, potential.planarPotential):
        if axi:
            o = Orbit(
                [[1.0, 1.1, 1.1], [1.1, -0.1, 0.9]],
                ro=ro,
                vo=vo,
                zo=zo,
                solarmotion=solarmotion,
            )
        else:
            o = Orbit(
                [[1.0, 1.1, 1.1, 0.0], [1.1, -1.2, -0.9, 2.0]],
                ro=ro,
                vo=vo,
                zo=zo,
                solarmotion=solarmotion,
            )
    else:
        if axi:
            o = Orbit(
                [[1.0, 1.1, 1.1, 0.1, 0.1], [1.1, -0.7, 1.4, -0.1, 0.3]],
                ro=ro,
                vo=vo,
                zo=zo,
                solarmotion=solarmotion,
            )
        else:
            o = Orbit(
                [[1.0, 1.1, 1.1, 0.1, 0.1, 0.0], [0.6, -0.4, -1.0, -0.3, -0.5, 2.0]],
                ro=ro,
                vo=vo,
                zo=zo,
                solarmotion=solarmotion,
            )
    return o


def test_flip():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    for ii in range(5):
        # Scales to test that these are properly propagated to the new Orbit
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbits_flip(llp, ro, vo, None, None, axi=False)
        of = o.flip()
        # First check that the scales have been propagated properly
        assert (
            numpy.fabs(o._ro - of._ro) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            numpy.fabs(o._vo - of._vo) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert (
                (o._zo is None) * (of._zo is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert (
                (o._solarmotion is None) * (of._solarmotion is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        else:
            assert (
                numpy.fabs(o._zo - of._zo) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._roSet == of._roSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._voSet == of._voSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert numpy.all(
                numpy.abs(o.x() - of.x()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vx() + of.vx()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        else:
            assert numpy.all(
                numpy.abs(o.R() - of.R()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vR() + of.vR()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vT() + of.vT()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii % 2 == 1:
            assert numpy.all(
                numpy.abs(o.phi() - of.phi()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii < 2:
            assert numpy.all(
                numpy.abs(o.z() - of.z()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vz() + of.vz()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
    return None


# Test flippingg an orbit inplace
def test_flip_inplace():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbits_flip(llp, ro, vo, None, None, axi=False)
        of = o()
        of.flip(inplace=True)
        # First check that the scales have been propagated properly
        assert (
            numpy.fabs(o._ro - of._ro) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            numpy.fabs(o._vo - of._vo) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert (
                (o._zo is None) * (of._zo is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert (
                (o._solarmotion is None) * (of._solarmotion is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        else:
            assert (
                numpy.fabs(o._zo - of._zo) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._roSet == of._roSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._voSet == of._voSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert numpy.all(
                numpy.abs(o.x() - of.x()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vx() + of.vx()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        else:
            assert numpy.all(
                numpy.abs(o.R() - of.R()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vR() + of.vR()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vT() + of.vT()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii % 2 == 1:
            assert numpy.all(
                numpy.abs(o.phi() - of.phi()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii < 2:
            assert numpy.all(
                numpy.abs(o.z() - of.z()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vz() + of.vz()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
    return None


# Test flippingg an orbit inplace after orbit integration
def test_flip_inplace_integrated():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    ts = numpy.linspace(0.0, 1.0, 11)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbits_flip(llp, ro, vo, None, None, axi=False)
        of = o()
        if ii < 2 or ii == 3:
            o.integrate(ts, lp)
            of.integrate(ts, lp)
        elif ii == 2:
            o.integrate(ts, plp)
            of.integrate(ts, plp)
        else:
            o.integrate(ts, llp)
            of.integrate(ts, llp)
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o = o(0.5)
        of = of(0.5)
        # First check that the scales have been propagated properly
        assert (
            numpy.fabs(o._ro - of._ro) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            numpy.fabs(o._vo - of._vo) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert (
                (o._zo is None) * (of._zo is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert (
                (o._solarmotion is None) * (of._solarmotion is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        else:
            assert (
                numpy.fabs(o._zo - of._zo) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._roSet == of._roSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._voSet == of._voSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert numpy.all(
                numpy.abs(o.x() - of.x()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vx() + of.vx()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        else:
            assert numpy.all(
                numpy.abs(o.R() - of.R()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vR() + of.vR()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vT() + of.vT()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii % 2 == 1:
            assert numpy.all(
                numpy.abs(o.phi() - of.phi()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii < 2:
            assert numpy.all(
                numpy.abs(o.z() - of.z()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vz() + of.vz()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
    return None


# Test flippingg an orbit inplace after orbit integration, and after having
# once evaluated the orbit before flipping inplace (#345)
# only difference wrt previous test is a line that evaluates of before
# flipping
def test_flip_inplace_integrated_evaluated():
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    plp = lp.toPlanar()
    llp = lp.toVertical(1.0)
    ts = numpy.linspace(0.0, 1.0, 11)
    for ii in range(5):
        # Scales (not really necessary for this test)
        ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
        if ii == 0:  # axi, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 1:  # track azimuth, full
            o = setup_orbits_flip(lp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 2:  # axi, planar
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=True)
        elif ii == 3:  # track azimuth, full
            o = setup_orbits_flip(plp, ro, vo, zo, solarmotion, axi=False)
        elif ii == 4:  # linear orbit
            o = setup_orbits_flip(llp, ro, vo, None, None, axi=False)
        of = o()
        if ii < 2 or ii == 3:
            o.integrate(ts, lp)
            of.integrate(ts, lp)
        elif ii == 2:
            o.integrate(ts, plp)
            of.integrate(ts, plp)
        else:
            o.integrate(ts, llp)
            of.integrate(ts, llp)
        # Evaluate, make sure it is at an interpolated time!
        dumb = of.R(0.52)
        # Now flip
        of.flip(inplace=True)
        # Just check one time, allows code duplication!
        o = o(0.52)
        of = of(0.52)
        # First check that the scales have been propagated properly
        assert (
            numpy.fabs(o._ro - of._ro) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            numpy.fabs(o._vo - of._vo) < 10.0**-15.0
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert (
                (o._zo is None) * (of._zo is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert (
                (o._solarmotion is None) * (of._solarmotion is None)
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        else:
            assert (
                numpy.fabs(o._zo - of._zo) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
            assert numpy.all(
                numpy.fabs(o._solarmotion - of._solarmotion) < 10.0**-15.0
            ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._roSet == of._roSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        assert (
            o._voSet == of._voSet
        ), "o.flip() did not conserve physical scales and coordinate-transformation parameters"
        if ii == 4:
            assert numpy.all(
                numpy.abs(o.x() - of.x()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vx() + of.vx()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        else:
            assert numpy.all(
                numpy.abs(o.R() - of.R()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vR() + of.vR()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vT() + of.vT()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii % 2 == 1:
            assert numpy.all(
                numpy.abs(o.phi() - of.phi()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
        if ii < 2:
            assert numpy.all(
                numpy.abs(o.z() - of.z()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
            assert numpy.all(
                numpy.abs(o.vz() + of.vz()) < 10.0**-10.0
            ), "o.flip() did not work as expected"
    return None


# test getOrbit
def test_getOrbit():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    o = Orbit([[1.0, 0.1, 1.2, 0.3, 0.2, 2.0], [1.0, -0.1, 1.1, -0.3, 0.2, 5.0]])
    times = numpy.linspace(0.0, 7.0, 251)
    o.integrate(times, lp)
    Rs = o.R(times)
    vRs = o.vR(times)
    vTs = o.vT(times)
    zs = o.z(times)
    vzs = o.vz(times)
    phis = o.phi(times)
    orbarray = o.getOrbit()
    assert (
        numpy.all(numpy.fabs(Rs - orbarray[..., 0])) < 10.0**-16.0
    ), "getOrbit does not work as expected for R"
    assert (
        numpy.all(numpy.fabs(vRs - orbarray[..., 1])) < 10.0**-16.0
    ), "getOrbit does not work as expected for vR"
    assert (
        numpy.all(numpy.fabs(vTs - orbarray[..., 2])) < 10.0**-16.0
    ), "getOrbit does not work as expected for vT"
    assert (
        numpy.all(numpy.fabs(zs - orbarray[..., 3])) < 10.0**-16.0
    ), "getOrbit does not work as expected for z"
    assert (
        numpy.all(numpy.fabs(vzs - orbarray[..., 4])) < 10.0**-16.0
    ), "getOrbit does not work as expected for vz"
    assert (
        numpy.all(numpy.fabs(phis - orbarray[..., 5])) < 10.0**-16.0
    ), "getOrbit does not work as expected for phi"
    return None


# Test that the eccentricity, zmax, rperi, and rap calculated numerically by
# Orbits agrees with that calculated numerically using Orbit
def test_EccZmaxRperiRap_num_againstorbit_3d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test AttributeError when not integrated
    with pytest.raises(AttributeError):
        os.e()
    with pytest.raises(AttributeError):
        os.zmax()
    with pytest.raises(AttributeError):
        os.rperi()
    with pytest.raises(AttributeError):
        os.rap()
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.e()[ii] - list_os[ii].e()) < 1e-10
        ), "Evaluating Orbits e does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.zmax()[ii] - list_os[ii].zmax()) < 1e-10
        ), "Evaluating Orbits zmax does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.rperi()[ii] - list_os[ii].rperi()) < 1e-10
        ), "Evaluating Orbits rperi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.rap()[ii] - list_os[ii].rap()) < 1e-10
        ), "Evaluating Orbits rap does not agree with Orbit"
    return None


def test_EccZmaxRperiRap_num_againstorbit_2d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, phis)))
    list_os = [Orbit([R, vR, vT, phi]) for R, vR, vT, phi in zip(Rs, vRs, vTs, phis)]
    # Integrate all
    times = numpy.linspace(0.0, 10.0, 1001)
    os.integrate(times, MWPotential2014)
    [o.integrate(times, MWPotential2014) for o in list_os]
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(os.e()[ii] - list_os[ii].e()) < 1e-10
        ), "Evaluating Orbits e does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.rperi()[ii] - list_os[ii].rperi()) < 1e-10
        ), "Evaluating Orbits rperi does not agree with Orbit"
        assert numpy.all(
            numpy.fabs(os.rap()[ii] - list_os[ii].rap()) < 1e-10
        ), "Evaluating Orbits rap does not agree with Orbit"
    return None


# Test that the eccentricity, zmax, rperi, and rap calculated analytically by
# Orbits agrees with that calculated analytically using Orbit
def test_EccZmaxRperiRap_analytic_againstorbit_3d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test AttributeError when no potential and not integrated
    with pytest.raises(AttributeError):
        os.e(analytic=True)
    with pytest.raises(AttributeError):
        os.zmax(analytic=True)
    with pytest.raises(AttributeError):
        os.rperi(analytic=True)
    with pytest.raises(AttributeError):
        os.rap(analytic=True)
    for type in ["spherical", "staeckel", "adiabatic"]:
        for ii in range(nrand):
            assert numpy.all(
                numpy.fabs(
                    os.e(pot=MWPotential2014, analytic=True, type=type)[ii]
                    - list_os[ii].e(pot=MWPotential2014, analytic=True, type=type)
                )
                < 1e-10
            ), f"Evaluating Orbits e analytically does not agree with Orbit for type={type}"
        assert numpy.all(
            numpy.fabs(
                os.zmax(pot=MWPotential2014, analytic=True, type=type)[ii]
                - list_os[ii].zmax(pot=MWPotential2014, analytic=True, type=type)
            )
            < 1e-10
        ), f"Evaluating Orbits zmax analytically does not agree with Orbit for type={type}"
        assert numpy.all(
            numpy.fabs(
                os.rperi(pot=MWPotential2014, analytic=True, type=type)[ii]
                - list_os[ii].rperi(pot=MWPotential2014, analytic=True, type=type)
            )
            < 1e-10
        ), f"Evaluating Orbits rperi analytically does not agree with Orbit for type={type}"
        assert numpy.all(
            numpy.fabs(
                os.rap(pot=MWPotential2014, analytic=True, type=type)[ii]
                - list_os[ii].rap(pot=MWPotential2014, analytic=True, type=type)
            )
            < 1e-10
        ), f"Evaluating Orbits rap analytically does not agree with Orbit for type={type}"
    return None


def test_EccZmaxRperiRap_analytic_againstorbit_2d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, phis)))
    list_os = [Orbit([R, vR, vT, phi]) for R, vR, vT, phi in zip(Rs, vRs, vTs, phis)]
    # No matter the type, should always be using adiabtic, not specified in
    # Orbit
    for type in ["spherical", "staeckel", "adiabatic"]:
        for ii in range(nrand):
            assert numpy.all(
                numpy.fabs(
                    os.e(pot=MWPotential2014, analytic=True, type=type)[ii]
                    - list_os[ii].e(pot=MWPotential2014, analytic=True)
                )
                < 1e-10
            ), f"Evaluating Orbits e analytically does not agree with Orbit for type={type}"
        assert numpy.all(
            numpy.fabs(
                os.rperi(pot=MWPotential2014, analytic=True, type=type)[ii]
                - list_os[ii].rperi(pot=MWPotential2014, analytic=True, type=type)
            )
            < 1e-10
        ), f"Evaluating Orbits rperi analytically does not agree with Orbit for type={type}"
        assert numpy.all(
            numpy.fabs(
                os.rap(pot=MWPotential2014, analytic=True, type=type)[ii]
                - list_os[ii].rap(pot=MWPotential2014, analytic=True)
            )
            < 1e-10
        ), f"Evaluating Orbits rap analytically does not agree with Orbit for type={type}"
    return None


def test_rguiding():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test that if potential is not given, error is raised
    with pytest.raises(RuntimeError):
        os.rguiding()
    # With small number, calculation is direct
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(
                os.rguiding(pot=MWPotential2014)[ii]
                / list_os[ii].rguiding(pot=MWPotential2014)
                - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits rguiding analytically does not agree with Orbit"
    # With large number, calculation is interpolated
    nrand = 1002
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    rgs = os.rguiding(pot=MWPotential2014)
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(rgs[ii] / list_os[ii].rguiding(pot=MWPotential2014) - 1.0)
            < 10.0**-10.0
        ), "Evaluating Orbits rguiding analytically does not agree with Orbit"
    # rguiding for non-axi potential fails
    with pytest.raises(
        RuntimeError,
        match="Potential given to rguiding is non-axisymmetric, but rguiding requires an axisymmetric potential",
    ) as exc_info:
        os.rguiding(pot=MWPotential2014 + potential.DehnenBarPotential())
    return None


def test_rE():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test that if potential is not given, error is raised
    with pytest.raises(RuntimeError):
        os.rE()
    # With small number, calculation is direct
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(
                os.rE(pot=MWPotential2014)[ii] / list_os[ii].rE(pot=MWPotential2014)
                - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits rE analytically does not agree with Orbit"
    # With large number, calculation is interpolated
    nrand = 1002
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    rgs = os.rE(pot=MWPotential2014)
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(rgs[ii] / list_os[ii].rE(pot=MWPotential2014) - 1.0)
            < 10.0**-10.0
        ), "Evaluating Orbits rE analytically does not agree with Orbit"
    # rE for non-axi potential fails
    with pytest.raises(
        RuntimeError,
        match="Potential given to rE is non-axisymmetric, but rE requires an axisymmetric potential",
    ) as exc_info:
        os.rE(pot=MWPotential2014 + potential.DehnenBarPotential())
    return None


def test_LcE():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test that if potential is not given, error is raised
    with pytest.raises(RuntimeError):
        os.LcE()
    # With small number, calculation is direct
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(
                os.LcE(pot=MWPotential2014)[ii] / list_os[ii].LcE(pot=MWPotential2014)
                - 1.0
            )
            < 10.0**-10.0
        ), "Evaluating Orbits LcE analytically does not agree with Orbit"
    # With large number, calculation is interpolated
    nrand = 1002
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    rgs = os.LcE(pot=MWPotential2014)
    for ii in range(nrand):
        assert numpy.all(
            numpy.fabs(rgs[ii] / list_os[ii].LcE(pot=MWPotential2014) - 1.0)
            < 10.0**-10.0
        ), "Evaluating Orbits LcE analytically does not agree with Orbit"
    # LcE for non-axi potential fails
    with pytest.raises(
        RuntimeError,
        match="Potential given to LcE is non-axisymmetric, but LcE requires an axisymmetric potential",
    ) as exc_info:
        os.LcE(pot=MWPotential2014 + potential.DehnenBarPotential())
    return None


# Test that the actions, frequencies/periods, and angles calculated
# analytically by Orbits agrees with that calculated analytically using Orbit
def test_actionsFreqsAngles_againstorbit_3d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, zs, vzs, phis)))
    list_os = [
        Orbit([R, vR, vT, z, vz, phi])
        for R, vR, vT, z, vz, phi in zip(Rs, vRs, vTs, zs, vzs, phis)
    ]
    # First test AttributeError when no potential and not integrated
    with pytest.raises(AttributeError):
        os.jr()
    with pytest.raises(AttributeError):
        os.jp()
    with pytest.raises(AttributeError):
        os.jz()
    with pytest.raises(AttributeError):
        os.wr()
    with pytest.raises(AttributeError):
        os.wp()
    with pytest.raises(AttributeError):
        os.wz()
    with pytest.raises(AttributeError):
        os.Or()
    with pytest.raises(AttributeError):
        os.Op()
    with pytest.raises(AttributeError):
        os.Oz()
    with pytest.raises(AttributeError):
        os.Tr()
    with pytest.raises(AttributeError):
        os.Tp()
    with pytest.raises(AttributeError):
        os.TrTp()
    with pytest.raises(AttributeError):
        os.Tz()
    # Tolerance for jr, jp, jz, diff. for isochroneApprox, because currently
    # not implemented in exactly the same way in Orbit and Orbits (Orbit uses
    # __call__ for the actions, Orbits uses actionsFreqsAngles, which is diff.)
    tol = {}
    tol["spherical"] = -12.0
    tol["staeckel"] = -12.0
    tol["adiabatic"] = -12.0
    tol["isochroneApprox"] = -2.0
    # For now we skip adiabatic here, because frequencies and angles not
    # implemented yet
    #    for type in ['spherical','staeckel','adiabatic']:
    for type in ["spherical", "staeckel", "isochroneApprox"]:
        for ii in range(nrand):
            assert numpy.all(
                numpy.fabs(
                    os.jr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].jr(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 10.0 ** tol[type]
            ), f"Evaluating Orbits jr analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.jp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].jp(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 10.0 ** tol[type]
            ), f"Evaluating Orbits jp analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.jz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].jz(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 10.0 ** tol[type]
            ), f"Evaluating Orbits jz analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.wr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].wr(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits wr analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.wp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].wp(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits wp analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.wz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].wz(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits wz analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Or(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Or(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Or analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Op(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Op(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Op analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Oz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Oz(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Oz analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Tr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Tr(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Tr analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Tp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Tp(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Tp analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.TrTp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].TrTp(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits TrTp analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.Tz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].Tz(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 1e-10
            ), f"Evaluating Orbits Tz analytically does not agree with Orbit for type={type}"
            if type == "isochroneApprox":
                break  # otherwise takes too long
    return None


# Test that the actions, frequencies/periods, and angles calculated
# analytically by Orbits agrees with that calculated analytically using Orbit
def test_actionsFreqsAngles_againstorbit_2d():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = 10
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    os = Orbit(list(zip(Rs, vRs, vTs, phis)))
    list_os = [Orbit([R, vR, vT, phi]) for R, vR, vT, phi in zip(Rs, vRs, vTs, phis)]
    # First test AttributeError when no potential and not integrated
    with pytest.raises(AttributeError):
        os.jr()
    with pytest.raises(AttributeError):
        os.jp()
    with pytest.raises(AttributeError):
        os.jz()
    with pytest.raises(AttributeError):
        os.wr()
    with pytest.raises(AttributeError):
        os.wp()
    with pytest.raises(AttributeError):
        os.wz()
    with pytest.raises(AttributeError):
        os.Or()
    with pytest.raises(AttributeError):
        os.Op()
    with pytest.raises(AttributeError):
        os.Oz()
    with pytest.raises(AttributeError):
        os.Tr()
    with pytest.raises(AttributeError):
        os.Tp()
    with pytest.raises(AttributeError):
        os.TrTp()
    with pytest.raises(AttributeError):
        os.Tz()
    # Tolerance for jr, jp, jz, diff. for isochroneApprox, because currently
    # not implemented in exactly the same way in Orbit and Orbits (Orbit uses
    # __call__ for the actions, Orbits uses actionsFreqsAngles, which is diff.)
    tol = {}
    tol["spherical"] = -12.0
    tol["staeckel"] = -12.0
    tol["adiabatic"] = -12.0
    tol["isochroneApprox"] = -2.0
    # For now we skip adiabatic here, because frequencies and angles not
    # implemented yet
    #    for type in ['spherical','staeckel','adiabatic']:
    for type in ["spherical", "staeckel", "isochroneApprox"]:
        for ii in range(nrand):
            assert numpy.all(
                numpy.fabs(
                    os.jr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].jr(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 10.0 ** tol[type]
            ), f"Evaluating Orbits jr analytically does not agree with Orbit for type={type}"
            assert numpy.all(
                numpy.fabs(
                    os.jp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                    / list_os[ii].jp(
                        pot=MWPotential2014, analytic=True, type=type, b=0.8
                    )
                    - 1.0
                )
                < 10.0 ** tol[type]
            ), f"Evaluating Orbits jp analytically does not agree with Orbit for type={type}"
            # zero, so don't divide, also doesn't work for isochroneapprox now
            if not type == "isochroneApprox":
                assert numpy.all(
                    numpy.fabs(
                        os.jz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        - list_os[ii].jz(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                    )
                    < 10.0 ** tol[type]
                ), f"Evaluating Orbits jz analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.wr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].wr(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits wr analytically does not agree with Orbit for type={type}"
                # Think I may have fixed wp = NaN?
                # assert numpy.all(numpy.fabs(os.wp(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].wp(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits wp analytically does not agree with Orbit for type={}'.format(type)
                # assert numpy.all(numpy.fabs(os.wz(pot=MWPotential2014,analytic=True,type=type,b=0.8)[ii]/list_os[ii].wz(pot=MWPotential2014,analytic=True,type=type,b=0.8)-1.) < 1e-10), 'Evaluating Orbits wz analytically does not agree with Orbit for type={}'.format(type)
                assert numpy.all(
                    numpy.fabs(
                        os.Or(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Or(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Or analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.Op(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Op(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Op analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.Oz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Oz(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Oz analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.Tr(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Tr(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Tr analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.Tp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Tp(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Tp analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.TrTp(pot=MWPotential2014, analytic=True, type=type, b=0.8)[
                            ii
                        ]
                        / list_os[ii].TrTp(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits TrTp analytically does not agree with Orbit for type={type}"
                assert numpy.all(
                    numpy.fabs(
                        os.Tz(pot=MWPotential2014, analytic=True, type=type, b=0.8)[ii]
                        / list_os[ii].Tz(
                            pot=MWPotential2014, analytic=True, type=type, b=0.8
                        )
                        - 1.0
                    )
                    < 1e-10
                ), f"Evaluating Orbits Tz analytically does not agree with Orbit for type={type}"
            if type == "isochroneApprox":
                break  # otherwise takes too long
    return None


def test_actionsFreqsAngles_output_shape():
    # Test that the output shape is correct and that the shaped output is correct for actionAngle methods
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    numpy.random.seed(1)
    nrand = (3, 1, 2)
    Rs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    vRs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vTs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0) + 1.0
    zs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vzs = 0.2 * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    phis = 2.0 * numpy.pi * (2.0 * numpy.random.uniform(size=nrand) - 1.0)
    vxvv = numpy.rollaxis(numpy.array([Rs, vRs, vTs, zs, vzs, phis]), 0, 4)
    os = Orbit(vxvv)
    list_os = [
        [
            [
                Orbit(
                    [
                        Rs[ii, jj, kk],
                        vRs[ii, jj, kk],
                        vTs[ii, jj, kk],
                        zs[ii, jj, kk],
                        vzs[ii, jj, kk],
                        phis[ii, jj, kk],
                    ]
                )
                for kk in range(nrand[2])
            ]
            for jj in range(nrand[1])
        ]
        for ii in range(nrand[0])
    ]
    # Tolerance for jr, jp, jz, diff. for isochroneApprox, because currently
    # not implemented in exactly the same way in Orbit and Orbits (Orbit uses
    # __call__ for the actions, Orbits uses actionsFreqsAngles, which is diff.)
    tol = {}
    tol["spherical"] = -12.0
    tol["staeckel"] = -12.0
    tol["adiabatic"] = -12.0
    tol["isochroneApprox"] = -2.0
    # For now we skip adiabatic here, because frequencies and angles not
    # implemented yet
    #    for type in ['spherical','staeckel','adiabatic']:
    for type in ["spherical", "staeckel", "isochroneApprox"]:
        # Evaluate Orbits once to not be too slow...
        tjr = os.jr(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tjp = os.jp(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tjz = os.jz(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        twr = os.wr(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        twp = os.wp(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        twz = os.wz(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tOr = os.Or(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tOp = os.Op(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tOz = os.Oz(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tTr = os.Tr(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tTp = os.Tp(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tTrTp = os.TrTp(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        tTz = os.Tz(pot=MWPotential2014, analytic=True, type=type, b=0.8)
        for ii in range(nrand[0]):
            for jj in range(nrand[1]):
                for kk in range(nrand[2]):
                    assert numpy.all(
                        numpy.fabs(
                            tjr[ii, jj, kk]
                            / list_os[ii][jj][kk].jr(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 10.0 ** tol[type]
                    ), f"Evaluating Orbits jr analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tjp[ii, jj, kk]
                            / list_os[ii][jj][kk].jp(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 10.0 ** tol[type]
                    ), f"Evaluating Orbits jp analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tjz[ii, jj, kk]
                            / list_os[ii][jj][kk].jz(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 10.0 ** tol[type]
                    ), f"Evaluating Orbits jz analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            twr[ii, jj, kk]
                            / list_os[ii][jj][kk].wr(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits wr analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            twp[ii, jj, kk]
                            / list_os[ii][jj][kk].wp(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits wp analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            twz[ii, jj, kk]
                            / list_os[ii][jj][kk].wz(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits wz analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tOr[ii, jj, kk]
                            / list_os[ii][jj][kk].Or(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Or analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tOp[ii, jj, kk]
                            / list_os[ii][jj][kk].Op(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Op analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tOz[ii, jj, kk]
                            / list_os[ii][jj][kk].Oz(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Oz analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tTr[ii, jj, kk]
                            / list_os[ii][jj][kk].Tr(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Tr analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tTp[ii, jj, kk]
                            / list_os[ii][jj][kk].Tp(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Tp analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tTrTp[ii, jj, kk]
                            / list_os[ii][jj][kk].TrTp(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits TrTp analytically does not agree with Orbit for type={type}"
                    assert numpy.all(
                        numpy.fabs(
                            tTz[ii, jj, kk]
                            / list_os[ii][jj][kk].Tz(
                                pot=MWPotential2014, analytic=True, type=type, b=0.8
                            )
                            - 1.0
                        )
                        < 1e-10
                    ), f"Evaluating Orbits Tz analytically does not agree with Orbit for type={type}"
                    if type == "isochroneApprox":
                        break  # otherwise takes too long
    return None


# Test that the delta parameter is properly dealt with when using the staeckel
# approximation: when it changes, need to re-do the aA calcs.
def test_actionsFreqsAngles_staeckeldelta():
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    os = Orbit([None, None])  # Just twice the Sun!
    # First with delta
    jr = os.jr(delta=0.4, pot=MWPotential2014)
    # Now without, should be different
    jrn = os.jr(pot=MWPotential2014)
    assert numpy.all(
        numpy.fabs(jr - jrn) > 1e-4
    ), "Action calculation in Orbits using Staeckel approximation not updated when going from specifying delta to not specifying it"
    # Again, now the other way around
    os = Orbit([None, None])  # Just twice the Sun!
    # First without delta
    jrn = os.jr(pot=MWPotential2014)
    # Now with, should be different
    jr = os.jr(delta=0.4, pot=MWPotential2014)
    assert numpy.all(
        numpy.fabs(jr - jrn) > 1e-4
    ), "Action calculation in Orbits using Staeckel approximation not updated when going from specifying delta to not specifying it"
    return None


# Test that actionAngleStaeckel for a spherical potential is the same
# as actionAngleSpherical
def test_actionsFreqsAngles_staeckeldeltaequalzero():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    os = Orbit([None, None])  # Just twice the Sun!
    lp = LogarithmicHaloPotential(normalize=1.0)
    assert numpy.all(
        numpy.fabs(os.jr(pot=lp, type="staeckel") - os.jr(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.jp(pot=lp, type="staeckel") - os.jp(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.jz(pot=lp, type="staeckel") - os.jz(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.wr(pot=lp, type="staeckel") - os.wr(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.wp(pot=lp, type="staeckel") - os.wp(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.wz(pot=lp, type="staeckel") - os.wz(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.Tr(pot=lp, type="staeckel") - os.Tr(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.Tp(pot=lp, type="staeckel") - os.Tp(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    assert numpy.all(
        numpy.fabs(os.Tz(pot=lp, type="staeckel") - os.Tz(pot=lp, type="spherical"))
        < 1e-8
    ), "Action-angle function for staeckel method with spherical potential is not equal to actionAngleSpherical"
    return None


# Test that the b / ip parameters are properly dealt with when using the
# isochroneapprox approximation: when they change, need to re-do the aA calcs.
def test_actionsFreqsAngles_isochroneapproxb():
    from galpy.orbit import Orbit
    from galpy.potential import IsochronePotential, MWPotential2014

    os = Orbit([None, None])  # Just twice the Sun!
    # First with one b
    jr = os.jr(type="isochroneapprox", b=0.8, pot=MWPotential2014)
    # Now with another b, should be different
    jrn = os.jr(type="isochroneapprox", b=1.8, pot=MWPotential2014)
    assert numpy.all(
        numpy.fabs(jr - jrn) > 1e-4
    ), "Action calculation in Orbits using isochroneapprox approximation not updated when going from specifying b to not specifying it"
    # Again, now specifying ip
    os = Orbit([None, None])  # Just twice the Sun!
    # First with one
    jrn = os.jr(
        pot=MWPotential2014,
        type="isochroneapprox",
        ip=IsochronePotential(normalize=1.1, b=0.8),
    )
    # Now with another one, should be different
    jr = os.jr(
        pot=MWPotential2014,
        type="isochroneapprox",
        ip=IsochronePotential(normalize=0.99, b=1.8),
    )
    assert numpy.all(
        numpy.fabs(jr - jrn) > 1e-4
    ), "Action calculation in Orbits using isochroneapprox approximation not updated when going from specifying delta to not specifying it"
    return None


def test_actionsFreqsAngles_RuntimeError_1d():
    from galpy.orbit import Orbit

    os = Orbit([[1.0, 0.1], [0.2, 0.3]])
    with pytest.raises(RuntimeError):
        os.jz(analytic=True)
    return None


def test_ChandrasekharDynamicalFrictionForce_constLambda():
    # Test from test_potential for Orbits now!
    #
    # Test that the ChandrasekharDynamicalFrictionForce with constant Lambda
    # agrees with analytical solutions for circular orbits:
    # assuming that a mass remains on a circular orbit in an isothermal halo
    # with velocity dispersion sigma and for constant Lambda:
    # r_final^2 - r_initial^2 = -0.604 ln(Lambda) GM/sigma t
    # (e.g., B&T08, p. 648)
    from galpy.orbit import Orbit
    from galpy.util import conversion

    ro, vo = 8.0, 220.0
    # Parameters
    GMs = 10.0**9.0 / conversion.mass_in_msol(vo, ro)
    const_lnLambda = 7.0
    r_inits = [2.0, 2.5]
    dt = 2.0 / conversion.time_in_Gyr(vo, ro)
    # Compute
    lp = potential.LogarithmicHaloPotential(normalize=1.0, q=1.0)
    cdfc = potential.ChandrasekharDynamicalFrictionForce(
        GMs=GMs, const_lnLambda=const_lnLambda, dens=lp
    )  # don't provide sigmar, so it gets computed using galpy.df.jeans
    o = Orbit(
        [
            Orbit([r_inits[0], 0.0, 1.0, 0.0, 0.0, 0.0]),
            Orbit([r_inits[1], 0.0, 1.0, 0.0, 0.0, 0.0]),
        ]
    )
    ts = numpy.linspace(0.0, dt, 1001)
    o.integrate(ts, [lp, cdfc], method="leapfrog")  # also tests fallback onto odeint
    r_pred = numpy.sqrt(
        numpy.array(o.r()) ** 2.0 - 0.604 * const_lnLambda * GMs * numpy.sqrt(2.0) * dt
    )
    assert numpy.all(
        numpy.fabs(r_pred - numpy.array(o.r(ts[-1]))) < 0.015
    ), "ChandrasekharDynamicalFrictionForce with constant lnLambda for circular orbits does not agree with analytical prediction"
    return None


# Check that toPlanar works
def test_toPlanar():
    from galpy.orbit import Orbit

    obs = Orbit([[1.0, 0.1, 1.1, 0.3, 0.0, 2.0], [1.0, -0.2, 1.3, -0.3, 0.0, 5.0]])
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for FullOrbit"
    assert numpy.all(
        obsp.R() == obs.R()
    ), "Planar orbit generated w/ toPlanar does not have the correct R"
    assert numpy.all(
        obsp.vR() == obs.vR()
    ), "Planar orbit generated w/ toPlanar does not have the correct vR"
    assert numpy.all(
        obsp.vT() == obs.vT()
    ), "Planar orbit generated w/ toPlanar does not have the correct vT"
    assert numpy.all(
        obsp.phi() == obs.phi()
    ), "Planar orbit generated w/ toPlanar does not have the correct phi"
    obs = Orbit([[1.0, 0.1, 1.1, 0.3, 0.0], [1.0, -0.2, 1.3, -0.3, 0.0]])
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for RZOrbit"
    assert numpy.all(
        obsp.R() == obs.R()
    ), "Planar orbit generated w/ toPlanar does not have the correct R"
    assert numpy.all(
        obsp.vR() == obs.vR()
    ), "Planar orbit generated w/ toPlanar does not have the correct vR"
    assert numpy.all(
        obsp.vT() == obs.vT()
    ), "Planar orbit generated w/ toPlanar does not have the correct vT"
    ro, vo, zo, solarmotion = 10.0, 300.0, 0.01, "schoenrich"
    obs = Orbit(
        [[1.0, 0.1, 1.1, 0.3, 0.0, 2.0], [1.0, -0.2, 1.3, -0.3, 0.0, 5.0]],
        ro=ro,
        vo=vo,
        zo=zo,
        solarmotion=solarmotion,
    )
    obsp = obs.toPlanar()
    assert obsp.dim() == 2, "toPlanar does not generate an Orbit w/ dim=2 for RZOrbit"
    assert numpy.all(
        obsp.R() == obs.R()
    ), "Planar orbit generated w/ toPlanar does not have the correct R"
    assert numpy.all(
        obsp.vR() == obs.vR()
    ), "Planar orbit generated w/ toPlanar does not have the correct vR"
    assert numpy.all(
        obsp.vT() == obs.vT()
    ), "Planar orbit generated w/ toPlanar does not have the correct vT"
    assert (
        numpy.fabs(obs._ro - obsp._ro) < 10.0**-15.0
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        numpy.fabs(obs._vo - obsp._vo) < 10.0**-15.0
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        numpy.fabs(obs._zo - obsp._zo) < 10.0**-15.0
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert numpy.all(
        numpy.fabs(obs._solarmotion - obsp._solarmotion) < 10.0**-15.0
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obs._roSet == obsp._roSet
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obs._voSet == obsp._voSet
    ), "Planar orbit generated w/ toPlanar does not have the proper physical scale and coordinate-transformation parameters associated with it"
    obs = Orbit([[1.0, 0.1, 1.1, 0.3], [1.0, -0.2, 1.3, -0.3]])
    try:
        obs.toPlanar()
    except AttributeError:
        pass
    else:
        raise AttributeError(
            "toPlanar() applied to a planar Orbit did not raise an AttributeError"
        )
    return None


# Check that toLinear works
def test_toLinear():
    from galpy.orbit import Orbit

    obs = Orbit([[1.0, 0.1, 1.1, 0.3, 0.0, 2.0], [1.0, -0.2, 1.3, -0.3, 0.0, 5.0]])
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinear does not generate an Orbit w/ dim=1 for FullOrbit"
    assert numpy.all(
        obsl.x() == obs.z()
    ), "Linear orbit generated w/ toLinear does not have the correct z"
    assert numpy.all(
        obsl.vx() == obs.vz()
    ), "Linear orbit generated w/ toLinear does not have the correct vx"
    obs = Orbit([[1.0, 0.1, 1.1, 0.3, 0.0], [1.0, -0.2, 1.3, -0.3, 0.0]])
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinear does not generate an Orbit w/ dim=1 for FullOrbit"
    assert numpy.all(
        obsl.x() == obs.z()
    ), "Linear orbit generated w/ toLinear does not have the correct z"
    assert numpy.all(
        obsl.vx() == obs.vz()
    ), "Linear orbit generated w/ toLinear does not have the correct vx"
    obs = Orbit([[1.0, 0.1, 1.1, 0.3], [1.0, -0.2, 1.3, -0.3]])
    try:
        obs.toLinear()
    except AttributeError:
        pass
    else:
        raise AttributeError(
            "toLinear() applied to a planar Orbit did not raise an AttributeError"
        )
    # w/ scales
    ro, vo = 10.0, 300.0
    obs = Orbit(
        [[1.0, 0.1, 1.1, 0.3, 0.0, 2.0], [1.0, -0.2, 1.3, -0.3, 0.0, 5.0]], ro=ro, vo=vo
    )
    obsl = obs.toLinear()
    assert obsl.dim() == 1, "toLinwar does not generate an Orbit w/ dim=1 for FullOrbit"
    assert numpy.all(
        obsl.x() == obs.z()
    ), "Linear orbit generated w/ toLinear does not have the correct z"
    assert numpy.all(
        obsl.vx() == obs.vz()
    ), "Linear orbit generated w/ toLinear does not have the correct vx"
    assert (
        numpy.fabs(obs._ro - obsl._ro) < 10.0**-15.0
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        numpy.fabs(obs._vo - obsl._vo) < 10.0**-15.0
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obsl._zo is None
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obsl._solarmotion is None
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obs._roSet == obsl._roSet
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    assert (
        obs._voSet == obsl._voSet
    ), "Linear orbit generated w/ toLinear does not have the proper physical scale and coordinate-transformation parameters associated with it"
    return None


# Check that the routines that should return physical coordinates are turned off by turn_physical_off
def test_physical_output_off():
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    o = Orbit()
    ro = o._ro
    vo = o._vo
    # turn off
    o.turn_physical_off()
    # Test positions
    assert (
        numpy.fabs(o.R() - o.R(use_physical=False)) < 10.0**-10.0
    ), "o.R() output for Orbit setup with ro= does not work as expected when turned off"
    assert (
        numpy.fabs(o.x() - o.x(use_physical=False)) < 10.0**-10.0
    ), "o.x() output for Orbit setup with ro= does not work as expected when turned off"
    assert (
        numpy.fabs(o.y() - o.y(use_physical=False)) < 10.0**-10.0
    ), "o.y() output for Orbit setup with ro= does not work as expected when turned off"
    assert (
        numpy.fabs(o.z() - o.z(use_physical=False)) < 10.0**-10.0
    ), "o.z() output for Orbit setup with ro= does not work as expected when turned off"
    assert (
        numpy.fabs(o.r() - o.r(use_physical=False)) < 10.0**-10.0
    ), "o.r() output for Orbit setup with ro= does not work as expected when turned off"
    # Test velocities
    assert (
        numpy.fabs(o.vR() - o.vR(use_physical=False)) < 10.0**-10.0
    ), "o.vR() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.vT() - o.vT(use_physical=False)) < 10.0**-10.0
    ), "o.vT() output for Orbit setup with vo= does not work as expected"
    assert (
        numpy.fabs(o.vphi() - o.vphi(use_physical=False)) < 10.0**-10.0
    ), "o.vphi() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.vx() - o.vx(use_physical=False)) < 10.0**-10.0
    ), "o.vx() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.vy() - o.vy(use_physical=False)) < 10.0**-10.0
    ), "o.vy() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.vz() - o.vz(use_physical=False)) < 10.0**-10.0
    ), "o.vz() output for Orbit setup with vo= does not work as expected when turned off"
    # Test energies
    assert (
        numpy.fabs(o.E(pot=lp) - o.E(pot=lp, use_physical=False)) < 10.0**-10.0
    ), "o.E() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.Jacobi(pot=lp) - o.Jacobi(pot=lp, use_physical=False))
        < 10.0**-10.0
    ), "o.E() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.ER(pot=lp) - o.ER(pot=lp, use_physical=False)) < 10.0**-10.0
    ), "o.ER() output for Orbit setup with vo= does not work as expected when turned off"
    assert (
        numpy.fabs(o.Ez(pot=lp) - o.Ez(pot=lp, use_physical=False)) < 10.0**-10.0
    ), "o.Ez() output for Orbit setup with vo= does not work as expected when turned off"
    # Test angular momentun
    assert numpy.all(
        numpy.fabs(o.L() - o.L(use_physical=False)) < 10.0**-10.0
    ), "o.L() output for Orbit setup with ro=,vo= does not work as expected when turned off"
    # Test action-angle functions
    assert (
        numpy.fabs(
            o.jr(pot=lp, type="staeckel", delta=0.5)
            - o.jr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.jr() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.jp(pot=lp, type="staeckel", delta=0.5)
            - o.jp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.jp() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.jz(pot=lp, type="staeckel", delta=0.5)
            - o.jz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.jz() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Tr(pot=lp, type="staeckel", delta=0.5)
            - o.Tr(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Tr() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Tp(pot=lp, type="staeckel", delta=0.5)
            - o.Tp(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Tp() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Tz(pot=lp, type="staeckel", delta=0.5)
            - o.Tz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Tz() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Or(pot=lp, type="staeckel", delta=0.5)
            - o.Or(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Or() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Op(pot=lp, type="staeckel", delta=0.5)
            - o.Op(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Op() output for Orbit setup with ro=,vo= does not work as expected"
    assert (
        numpy.fabs(
            o.Oz(pot=lp, type="staeckel", delta=0.5)
            - o.Oz(pot=lp, type="staeckel", delta=0.5, use_physical=False)
        )
        < 10.0**-10.0
    ), "o.Oz() output for Orbit setup with ro=,vo= does not work as expected"
    # Also test the times
    assert (
        numpy.fabs(o.time(1.0) - 1.0) < 10.0**-10.0
    ), "o.time() in physical coordinates does not work as expected when turned off"
    assert (
        numpy.fabs(o.time(1.0, ro=ro, vo=vo) - ro / vo / 1.0227121655399913)
        < 10.0**-10.0
    ), "o.time() in physical coordinates does not work as expected when turned off"
    return None


# Check that the routines that should return physical coordinates are turned
# back on by turn_physical_on
def test_physical_output_on():
    from astropy import units

    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential

    lp = LogarithmicHaloPotential(normalize=1.0)
    o = Orbit()
    ro = o._ro
    vo = o._vo
    o_orig = o()
    # turn off and on
    o.turn_physical_off()
    for ii in range(3):
        if ii == 0:
            o.turn_physical_on(ro=ro, vo=vo)
        elif ii == 1:
            o.turn_physical_on(ro=ro * units.kpc, vo=vo * units.km / units.s)
        else:
            o.turn_physical_on()
        # Test positions
        assert (
            numpy.fabs(o.R() - o_orig.R(use_physical=True)) < 10.0**-10.0
        ), "o.R() output for Orbit setup with ro= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.x() - o_orig.x(use_physical=True)) < 10.0**-10.0
        ), "o.x() output for Orbit setup with ro= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.y() - o_orig.y(use_physical=True)) < 10.0**-10.0
        ), "o.y() output for Orbit setup with ro= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.z() - o_orig.z(use_physical=True)) < 10.0**-10.0
        ), "o.z() output for Orbit setup with ro= does not work as expected when turned back on"
        # Test velocities
        assert (
            numpy.fabs(o.vR() - o_orig.vR(use_physical=True)) < 10.0**-10.0
        ), "o.vR() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.vT() - o_orig.vT(use_physical=True)) < 10.0**-10.0
        ), "o.vT() output for Orbit setup with vo= does not work as expected"
        assert (
            numpy.fabs(o.vphi() - o_orig.vphi(use_physical=True)) < 10.0**-10.0
        ), "o.vphi() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.vx() - o_orig.vx(use_physical=True)) < 10.0**-10.0
        ), "o.vx() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.vy() - o_orig.vy(use_physical=True)) < 10.0**-10.0
        ), "o.vy() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.vz() - o_orig.vz(use_physical=True)) < 10.0**-10.0
        ), "o.vz() output for Orbit setup with vo= does not work as expected when turned back on"
        # Test energies
        assert (
            numpy.fabs(o.E(pot=lp) - o_orig.E(pot=lp, use_physical=True)) < 10.0**-10.0
        ), "o.E() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.Jacobi(pot=lp) - o_orig.Jacobi(pot=lp, use_physical=True))
            < 10.0**-10.0
        ), "o.E() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.ER(pot=lp) - o_orig.ER(pot=lp, use_physical=True))
            < 10.0**-10.0
        ), "o.ER() output for Orbit setup with vo= does not work as expected when turned back on"
        assert (
            numpy.fabs(o.Ez(pot=lp) - o_orig.Ez(pot=lp, use_physical=True))
            < 10.0**-10.0
        ), "o.Ez() output for Orbit setup with vo= does not work as expected when turned back on"
        # Test angular momentun
        assert numpy.all(
            numpy.fabs(o.L() - o_orig.L(use_physical=True)) < 10.0**-10.0
        ), "o.L() output for Orbit setup with ro=,vo= does not work as expected when turned back on"
        # Test action-angle functions
        assert (
            numpy.fabs(
                o.jr(pot=lp, type="staeckel", delta=0.5)
                - o_orig.jr(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.jr() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.jp(pot=lp, type="staeckel", delta=0.5)
                - o_orig.jp(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.jp() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.jz(pot=lp, type="staeckel", delta=0.5)
                - o_orig.jz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.jz() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Tr(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Tr(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Tr() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Tp(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Tp(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Tp() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Tz(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Tz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Tz() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Or(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Or(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Or() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Op(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Op(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Op() output for Orbit setup with ro=,vo= does not work as expected"
        assert (
            numpy.fabs(
                o.Oz(pot=lp, type="staeckel", delta=0.5)
                - o_orig.Oz(pot=lp, type="staeckel", delta=0.5, use_physical=True)
            )
            < 10.0**-10.0
        ), "o.Oz() output for Orbit setup with ro=,vo= does not work as expected"
    # Also test the times
    assert (
        numpy.fabs(o.time(1.0) - o_orig.time(1.0, use_physical=True)) < 10.0**-10.0
    ), "o_orig.time() in physical coordinates does not work as expected when turned back on"
    return None


# Test that Orbits can be pickled
def test_pickling():
    import pickle

    from galpy.orbit import Orbit

    # Just test most common setup: 3D, 6 phase-D
    vxvvs = [[1.0, 0.1, 1.0, 0.1, -0.2, 1.5], [0.1, 3.0, 1.1, -0.3, 0.4, 2.0]]
    orbits = Orbit(vxvvs)
    pickled = pickle.dumps(orbits)
    orbits_unpickled = pickle.loads(pickled)
    # Tests
    assert (
        orbits_unpickled.dim() == 3
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        orbits_unpickled.phasedim() == 6
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.R()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.R()[1] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vR()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vR()[1] - 3.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vT()[0] - 1.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vT()[1] - 1.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.z()[0] - 0.1) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.z()[1] + 0.3) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vz()[0] + 0.2) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.vz()[1] - 0.4) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.phi()[0] - 1.5) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    assert (
        numpy.fabs(orbits_unpickled.phi()[1] - 2.0) < 1e-10
    ), "Orbits initialization with vxvv in 3D, 6 phase-D does not work as expected"
    return None


def test_from_name_values():
    from galpy.orbit import Orbit

    # test Vega and Lacaille 8760
    o = Orbit.from_name("Vega", "Lacaille 8760")
    assert numpy.allclose(
        o.ra(), [279.23473479, 319.31362024]
    ), "RA of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.dec(), [38.78368896, -38.86736390]
    ), "DEC of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.dist(), [1 / 130.23, 1 / 251.9124]
    ), "Parallax of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.pmra(), [200.94, -3258.996]
    ), "PMRA of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.pmdec(), [286.23, -1145.862]
    ), "PMDec of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.vlos(), [-20.60, 20.56]
    ), "radial velocity of Vega/Lacaille 8760  does not match SIMBAD value"
    # test Vega and Lacaille 8760, as a list
    o = Orbit.from_name(["Vega", "Lacaille 8760"])
    assert numpy.allclose(
        o.ra(), [279.23473479, 319.31362024]
    ), "RA of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.dec(), [38.78368896, -38.86736390]
    ), "DEC of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.dist(), [1 / 130.23, 1 / 251.9124]
    ), "Parallax of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.pmra(), [200.94, -3258.996]
    ), "PMRA of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.pmdec(), [286.23, -1145.862]
    ), "PMDec of Vega/Lacaille 8760  does not match SIMBAD value"
    assert numpy.allclose(
        o.vlos(), [-20.60, 20.56]
    ), "radial velocity of Vega/Lacaille 8760  does not match SIMBAD value"
    return None


def test_from_name_name():
    # Test that o.name gives the expected output
    from galpy.orbit import Orbit

    assert (
        Orbit.from_name("LMC").name == "LMC"
    ), "Orbit.from_name does not appear to set the name attribute correctly"
    assert numpy.char.equal(
        Orbit.from_name(["LMC"]).name, numpy.char.array("LMC")
    ), "Orbit.from_name does not appear to set the name attribute correctly"
    assert numpy.all(
        numpy.char.equal(
            Orbit.from_name(["LMC", "SMC"]).name, numpy.char.array(["LMC", "SMC"])
        )
    ), "Orbit.from_name does not appear to set the name attribute correctly"
    # Also slice
    assert (
        Orbit.from_name(["LMC", "SMC", "Fornax"])[-1].name == "Fornax"
    ), "Orbit.from_name does not appear to set the name attribute correctly"
    assert numpy.all(
        numpy.char.equal(
            Orbit.from_name(["LMC", "SMC", "Fornax"])[:2].name,
            numpy.char.array(["LMC", "SMC"]),
        )
    ), "Orbit.from_name does not appear to set the name attribute correctly"
    return None
