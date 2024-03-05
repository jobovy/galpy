################ TESTS OF THE SNAPSHOTPOTENTIAL CLASS AND RELATED #############
import numpy
import pynbody

from galpy import potential


def test_snapshotKeplerPotential_eval():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s, num_threads=1)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    assert (
        numpy.fabs(sp(1.0, 0.0) - kp(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp(0.5, 0.0) - kp(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp(1.0, 0.5) - kp(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp(1.0, -0.5) - kp(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_Rforce():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) - kp.Rforce(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.Rforce(0.5, 0.0) - kp.Rforce(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.5) - kp.Rforce(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.Rforce(1.0, -0.5) - kp.Rforce(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_zforce():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    assert (
        numpy.fabs(sp.zforce(1.0, 0.0) - kp.zforce(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.zforce(0.5, 0.0) - kp.zforce(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.zforce(1.0, 0.5) - kp.zforce(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp.zforce(1.0, -0.5) - kp.zforce(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_hash():
    # Test that hashing the previous grid works
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    assert (
        numpy.fabs(sp(1.0, 0.0) - kp(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    assert (
        numpy.fabs(sp(1.0, 0.0) - kp(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_grid():
    # Test that evaluating on a grid works
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=2.0)  # should be the same
    rs = numpy.arange(3) + 1
    zs = 0.1
    assert numpy.all(
        numpy.fabs(sp(rs, zs) - kp(rs, zs)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_eval_array():
    # Test evaluating the snapshotPotential with array input
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    rs = numpy.ones(3) * 0.5 + 0.5
    zs = (numpy.zeros(3) - 1.0) / 2.0
    assert numpy.all(
        numpy.fabs(sp(rs, zs) - kp(rs, zs)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_Rforce_array():
    # Test evaluating the snapshotPotential with array input
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    rs = numpy.ones(3) * 0.5 + 0.5
    zs = (numpy.zeros(3) - 1.0) / 2.0
    assert numpy.all(
        numpy.fabs(sp.Rforce(rs, zs) - kp.Rforce(rs, zs)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


def test_snapshotKeplerPotential_zforce_array():
    # Test evaluating the snapshotPotential with array input
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s)
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    rs = numpy.ones(3) * 0.5 + 0.5
    zs = (numpy.zeros(3) - 1.0) / 2.0
    assert numpy.all(
        numpy.fabs(sp.zforce(rs, zs) - kp.zforce(rs, zs)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass does not correspond to KeplerPotential"
    return None


# Test that using different numbers of azimuths to average over gives the same result
def test_snapshotKeplerPotential_eval_naz():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s, num_threads=1)
    spaz = potential.SnapshotRZPotential(s, num_threads=1, nazimuths=12)
    assert (
        numpy.fabs(sp(1.0, 0.0) - spaz(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp(0.5, 0.0) - spaz(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp(1.0, 0.5) - spaz(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp(1.0, -0.5) - spaz(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    return None


def test_snapshotKeplerPotential_Rforce_naz():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s, num_threads=1)
    spaz = potential.SnapshotRZPotential(s, num_threads=1, nazimuths=12)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) - spaz.Rforce(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.Rforce(0.5, 0.0) - spaz.Rforce(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.5) - spaz.Rforce(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.Rforce(1.0, -0.5) - spaz.Rforce(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    return None


def test_snapshotKeplerPotential_zforce_naz():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.SnapshotRZPotential(s, num_threads=1)
    spaz = potential.SnapshotRZPotential(s, num_threads=1, nazimuths=12)
    assert (
        numpy.fabs(sp.zforce(1.0, 0.0) - spaz.zforce(1.0, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.zforce(0.5, 0.0) - spaz.zforce(0.5, 0.0)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.zforce(1.0, 0.5) - spaz.zforce(1.0, 0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    assert (
        numpy.fabs(sp.zforce(1.0, -0.5) - spaz.zforce(1.0, -0.5)) < 10.0**-8.0
    ), "SnapshotRZPotential with single unit mass for naz=4 does not agree with naz=12"
    return None


def test_interpsnapshotKeplerPotential_eval():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpPot=True,
        zsym=True,
        numcores=1,
    )
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp(r, z) - kp(r, z)) / kp(r, z)) < 10.0**-10.0
            ), f"RZPot interpolation w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g})"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)
    zs = numpy.linspace(-0.2, 0.2, 20)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp(r, z) - kp(r, z)) / kp(r, z)) < 10.0**-5.0
            ), f"RZPot interpolation w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp(r,z)-kp(r,z))/kp(r,z)):g}"
    # Test all at the same time to use vector evaluation
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs((sp(mr, mz) - kp(mr, mz)) / kp(mr, mz)) < 10.0**-5.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input"
    return None


def test_interpsnapshotKeplerPotential_logR_eval():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        logR=True,
        zgrid=(0.0, 0.2, 201),
        interpPot=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    rs = numpy.linspace(0.02, 16.0, 20)
    zs = numpy.linspace(-0.15, 0.15, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs((sp(mr, mz) - kp(mr, mz)) / kp(mr, mz)) < 10.0**-5.0
    ), f"RZPot interpolation w/ interpRZPotential fails for vector input, w/ logR at (R,z) = ({mr[numpy.argmax(numpy.fabs((sp(mr,mz)-kp(mr,mz))/kp(mr,mz)))]:f},{mz[numpy.argmax(numpy.fabs((sp(mr,mz)-kp(mr,mz))/kp(mr,mz)))]:f}) by {numpy.amax(numpy.fabs((sp(mr,mz)-kp(mr,mz))/kp(mr,mz))):g}"
    return None


def test_interpsnapshotKeplerPotential_noc_eval():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpPot=True,
        zsym=True,
        enable_c=False,
    )
    kp = potential.KeplerPotential(amp=1.0)  # should be the same
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 10)
    zs = numpy.linspace(-0.2, 0.2, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs((sp(mr, mz) - kp(mr, mz)) / kp(mr, mz)) < 10.0**-5.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, without enable_c"
    return None


# Test that using different numbers of azimuths to average over gives the same result
def test_interpsnapshotKeplerPotential_eval_naz():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 1.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 51),
        zgrid=(0.0, 0.2, 51),
        logR=False,
        interpPot=True,
        zsym=True,
        numcores=1,
    )
    spaz = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 51),
        zgrid=(0.0, 0.2, 51),
        logR=False,
        interpPot=True,
        zsym=True,
        numcores=1,
        nazimuths=12,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp(r, z) - spaz(r, z)) / sp(r, z)) < 10.0**-10.0
            ), f"RZPot interpolation w/ InterpSnapShotPotential of KeplerPotential with different nazimuths fails at (R,z) = ({r:g},{z:g})"
    # This tests within the grid, with vector evaluation
    rs = numpy.linspace(0.01, 2.0, 10)
    zs = numpy.linspace(-0.2, 0.2, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs((sp(mr, mz) - spaz(mr, mz)) / sp(mr, mz)) < 10.0**-5.0
    ), "RZPot interpolation w/ interpRZPotential with different nazimimuths fails for vector input"
    return None


def test_interpsnapshotKeplerPotential_normalize():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 4.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 3.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpPot=True,
        interpepifreq=True,
        interpverticalfreq=True,
        zsym=True,
    )
    nkp = potential.KeplerPotential(normalize=1.0)  # normalized Kepler
    ukp = potential.KeplerPotential(amp=4.0)  # Un-normalized Kepler
    # Currently unnormalized
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp(1.0, 0.1) - ukp(1.0, 0.1)) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.epifreq(1.0) - ukp.epifreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.verticalfreq(1.0) - ukp.verticalfreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # Normalize
    sp.normalize(R0=1.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    assert (
        numpy.fabs(sp(1.0, 0.1) - nkp(1.0, 0.1)) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.epifreq(1.0) - nkp.epifreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.verticalfreq(1.0) - nkp.verticalfreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    assert (
        numpy.fabs(sp(1.0, 0.1) - ukp(1.0, 0.1)) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.epifreq(1.0) - ukp.epifreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.verticalfreq(1.0) - ukp.verticalfreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # Also test when R0 =/= 1
    sp.normalize(R0=2.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    assert (
        numpy.fabs(sp(1.0, 0.1) - nkp(1.0, 0.1)) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.epifreq(1.0) - nkp.epifreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.verticalfreq(1.0) - nkp.verticalfreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    assert (
        numpy.fabs(sp(1.0, 0.1) - ukp(1.0, 0.1)) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.epifreq(1.0) - ukp.epifreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    assert (
        numpy.fabs(sp.verticalfreq(1.0) - ukp.verticalfreq(1.0)) < 10.0**-4.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    return None


def test_interpsnapshotKeplerPotential_noc_normalize():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 4.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(numpy.log(0.01), numpy.log(3.0), 201),
        zgrid=(0.0, 0.2, 201),
        logR=True,
        interpPot=True,
        enable_c=False,
        zsym=True,
    )
    # Currently unnormalized
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-6.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # Normalize
    sp.normalize(R0=1.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-6.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-6.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # Also test when R0 =/= 1
    sp.normalize(R0=2.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-6.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-6.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    return None


def test_interpsnapshotKeplerPotential_normalize_units():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 4.0
    s["eps"] = 0.0
    s["pos"].units = "kpc"
    s["vel"].units = "km s**-1"
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 3.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpPot=True,
        zsym=True,
    )
    # Currently unnormalized
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be unnormalized doesn't behave as expected"
    # Normalize
    sp.normalize(R0=1.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # Also test when R0 =/= 1
    sp.normalize(R0=2.0)
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 1.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    # De normalize
    sp.denormalize()
    assert (
        numpy.fabs(sp.Rforce(1.0, 0.0) + 4.0) < 10.0**-7.0
    ), "InterpSnapShotPotential that is assumed to be normalized doesn't behave as expected"
    return None


def test_interpsnapshotKeplerPotential_epifreq():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpepifreq=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(normalize=2.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)[1:]
    for r in rs:
        assert (
            numpy.fabs((sp.epifreq(r) - kp.epifreq(r)) / kp.epifreq(r)) < 10.0**-4.0
        ), f"RZPot interpolation of epifreq w/ InterpSnapShotPotential of KeplerPotential fails at R = {r:g} by {numpy.fabs((sp.epifreq(r)-kp.epifreq(r))/kp.epifreq(r)):g}"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)[1:]
    for r in rs:
        assert (
            numpy.fabs((sp.epifreq(r) - kp.epifreq(r)) / kp.epifreq(r)) < 10.0**-4.0
        ), f"RZPot interpolation of epifreq w/ InterpSnapShotPotential of KeplerPotential fails at R = {r:g} by {numpy.fabs((sp.epifreq(r)-kp.epifreq(r))/kp.epifreq(r)):g}"
    return None


def test_interpsnapshotKeplerPotential_verticalfreq():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpverticalfreq=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(normalize=2.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)[1:]
    for r in rs:
        assert (
            numpy.fabs((sp.verticalfreq(r) - kp.verticalfreq(r)) / kp.verticalfreq(r))
            < 10.0**-4.0
        ), f"RZPot interpolation of verticalfreq w/ InterpSnapShotPotential of KeplerPotential fails at R = {r:g} by {numpy.fabs((sp.verticalfreq(r)-kp.verticalfreq(r))/kp.verticalfreq(r)):g}"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)[1:]
    for r in rs:
        assert (
            numpy.fabs((sp.verticalfreq(r) - kp.verticalfreq(r)) / kp.verticalfreq(r))
            < 10.0**-4.0
        ), f"RZPot interpolation of verticalfreq w/ InterpSnapShotPotential of KeplerPotential fails at R = {r:g} by {numpy.fabs((sp.verticalfreq(r)-kp.verticalfreq(r))/kp.verticalfreq(r)):g}"
    return None


def test_interpsnapshotKeplerPotential_R2deriv():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpepifreq=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(amp=2.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)[1:]
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.R2deriv(r, z) - kp.R2deriv(r, z)) / kp.R2deriv(r, z))
                < 10.0**-4.0
            ), f"RZPot interpolation of R2deriv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.R2deriv(r,z)-kp.R2deriv(r,z))/kp.R2deriv(r,z)):g}"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)[1:]
    zs = numpy.linspace(-0.2, 0.2, 20)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.R2deriv(r, z) - kp.R2deriv(r, z)) / kp.R2deriv(r, z))
                < 10.0**-4.0
            ), f"RZPot interpolation of R2deriv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.R2deriv(r,z)-kp.R2deriv(r,z))/kp.R2deriv(r,z)):g}"
    return None


def test_interpsnapshotKeplerPotential_z2deriv():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpverticalfreq=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(amp=2.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)[1:]
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.z2deriv(r, z) - kp.z2deriv(r, z)) / kp.z2deriv(r, z))
                < 10.0**-4.0
            ), f"RZPot interpolation of z2deriv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.z2deriv(r,z)-kp.z2deriv(r,z))/kp.z2deriv(r,z)):g}"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)[1:]
    zs = numpy.linspace(-0.2, 0.2, 20)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.z2deriv(r, z) - kp.z2deriv(r, z)) / kp.z2deriv(r, z))
                < 2.0 * 10.0**-4.0
            ), f"RZPot interpolation of z2deriv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.z2deriv(r,z)-kp.z2deriv(r,z))/kp.z2deriv(r,z)):g}"
    return None


def test_interpsnapshotKeplerpotential_Rzderiv():
    # Set up a snapshot with just one unit mass at the origin
    s = pynbody.new(star=1)
    s["mass"] = 2.0
    s["eps"] = 0.0
    sp = potential.InterpSnapshotRZPotential(
        s,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpepifreq=True,
        interpverticalfreq=True,
        zsym=True,
    )
    kp = potential.KeplerPotential(amp=2.0)  # should be the same
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)[1:]
    zs = numpy.linspace(-0.2, 0.2, 41)
    zs = zs[zs != 0.0]  # avoid zero
    # Test, but give small |z| a less constraining
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.Rzderiv(r, z) - kp.Rzderiv(r, z)) / kp.Rzderiv(r, z))
                < 10.0** -4.0 * (1.0 + 19.0 * (numpy.fabs(z) < 0.05))
            ), f"RZPot interpolation of Rzderiv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.Rzderiv(r,z)-kp.Rzderiv(r,z))/kp.Rzderiv(r,z)):g}; value is {kp.Rzderiv(r,z):g}"
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 10)[1:]
    zs = numpy.linspace(-0.2, 0.2, 20)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs((sp.Rzderiv(r, z) - kp.Rzderiv(r, z)) / kp.Rzderiv(r, z))
                < 10.0** -4.0 * (1.0 + 19.0 * (numpy.fabs(z) < 0.05))
            ), f"RZPot interpolation of Rzderiv w/ InterpSnapShotPotential of KeplerPotential fails at (R,z) = ({r:g},{z:g}) by {numpy.fabs((sp.Rzderiv(r,z)-kp.Rzderiv(r,z))/kp.Rzderiv(r,z)):g}"
    return None
