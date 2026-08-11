import numpy

from galpy import potential


def test_errors():
    # Test that when we set up an interpRZPotential w/ another interpRZPotential, we get an error
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 11),
        zgrid=(0.0, 0.2, 11),
        logR=False,
        interpPot=True,
        zsym=True,
    )
    try:
        rzpot2 = potential.interpRZPotential(
            RZPot=rzpot,
            rgrid=(0.01, 2.0, 11),
            zgrid=(0.0, 0.2, 11),
            logR=False,
            interpPot=True,
            zsym=True,
        )
    except potential.PotentialError:
        pass
    else:
        raise AssertionError(
            "Setting up an interpRZPotential w/ another interpRZPotential did not raise PotentialError"
        )


def test_interpolation_potential():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot(r, z)
                        - potential.evaluatePotentials(potential.MWPotential, r, z)
                    )
                    / potential.evaluatePotentials(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
            )
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot(r, z)
                        - potential.evaluatePotentials(potential.MWPotential, r, z)
                    )
                    / potential.evaluatePotentials(potential.MWPotential, r, z)
                )
                < 10.0**-6.0
            ), (
                f"RZPot interpolation w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
            )
    # Test all at the same time to use vector evaluation
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(0.0, 0.2, 101),
        interpPot=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, w/ logR"
    # Test the interpolation of the potential, w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(-0.2, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, w/o zsym"
    # Test the interpolation of the potential, w/o zsym and with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(-0.2, 0.2, 101),
        interpPot=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-6.0
    ), (
        "RZPot interpolation w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
    )
    return None


def test_interpolation_potential_diffinputs():
    # Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=True,
    )
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    # R vector, z scalar
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(rs, zs[10])
                - potential.evaluatePotentials(
                    potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
                )
            )
            / potential.evaluatePotentials(
                potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    # R scalar, z vector
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(rs[10], zs)
                - potential.evaluatePotentials(
                    potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
                )
            )
            / potential.evaluatePotentials(
                potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    return None


def test_interpolation_potential_c():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 151),
        zgrid=(0.0, 0.2, 151),
        logR=False,
        interpPot=True,
        enable_c=True,
        zsym=True,
    )
    # Test within the grid, using vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, using C"
    # now w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 151),
        zgrid=(-0.2, 0.2, 301),
        logR=False,
        interpPot=True,
        enable_c=True,
        zsym=False,
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-6.0
    ), (
        "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/o zsym"
    )
    # now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        logR=True,
        zgrid=(0.0, 0.2, 151),
        interpPot=True,
        enable_c=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 10.0, 20)  # don't go too far
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), (
        "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR"
    )
    # now with logR and w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        logR=True,
        zgrid=(-0.2, 0.2, 301),
        interpPot=True,
        enable_c=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 10.0, 20)  # don't go too far
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-6.0
    ), (
        "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
    )
    return None


def test_interpolation_potential_diffinputs_c():
    # Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 151),
        zgrid=(0.0, 0.2, 151),
        logR=False,
        interpPot=True,
        zsym=True,
        enable_c=True,
    )
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    # R vector, z scalar
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(rs, zs[10])
                - potential.evaluatePotentials(
                    potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
                )
            )
            / potential.evaluatePotentials(
                potential.MWPotential,
                rs,
                zs[10] * numpy.ones(len(rs)),
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    # R scalar, z vector
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(rs[10], zs)
                - potential.evaluatePotentials(
                    potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
                )
            )
            / potential.evaluatePotentials(
                potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    return None


def test_interpolation_potential_c_vdiffgridsizes():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 271),
        zgrid=(0.0, 0.2, 162),
        logR=False,
        interpPot=True,
        enable_c=True,
        zsym=True,
    )
    # Test within the grid, using vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot(mr, mz)
                - potential.evaluatePotentials(potential.MWPotential, mr, mz)
            )
            / potential.evaluatePotentials(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, using C"
    return None


def test_interpolation_potential_use_c():
    # Test the interpolation of the potential, using C to calculate the grid
    rzpot_c = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=True,
        use_c=False,
    )
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=True,
        use_c=True,
    )
    assert numpy.all(numpy.fabs(rzpot._potGrid - rzpot_c._potGrid) < 10.0**-14.0), (
        "Potential interpolation grid calculated with use_c does not agree with that calculated in python"
    )
    return None


# Test evaluation outside the grid
def test_interpolation_potential_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    zs = [-0.1, 0.3]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot(r, z)
                        - potential.evaluatePotentials(potential.MWPotential, r, z)
                    )
                    / potential.evaluatePotentials(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
    return None


# Test evaluation outside the grid in C
def test_interpolation_potential_outsidegrid_c():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        zsym=False,
        enable_c=True,
    )
    rs = [0.005, 2.5]
    zs = [-0.1, 0.3]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot(r, z)
                        - potential.evaluatePotentials(potential.MWPotential, r, z)
                    )
                    / potential.evaluatePotentials(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
    return None


def test_interpolation_potential_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    zs = [0.075, 0.15]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot(r, z)
                        - potential.evaluatePotentials(potential.MWPotential, r, z)
                    )
                    / potential.evaluatePotentials(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
            )
    return None


# Test Rforce and zforce
def test_interpolation_potential_force():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.Rforce(r, z)
                        - potential.evaluateRforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of Rforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
            )
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of zforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
            )
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    for r in rs:
        for z in zs:
            rforcediff = numpy.fabs(
                (
                    rzpot.Rforce(r, z)
                    - potential.evaluateRforces(potential.MWPotential, r, z)
                )
                / potential.evaluateRforces(potential.MWPotential, r, z)
            )
            assert rforcediff < 10.0**-5.0, (
                f"RZPot interpolation of Rforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {rforcediff:g}"
            )
            zforcediff = numpy.fabs(
                (
                    rzpot.zforce(r, z)
                    - potential.evaluatezforces(potential.MWPotential, r, z)
                )
                / potential.evaluatezforces(potential.MWPotential, r, z)
            )
            assert zforcediff < 5.0 * 10.0**-5.0, (
                f"RZPot interpolation of zforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {zforcediff:g}"
            )
    # Test all at the same time to use vector evaluation
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(0.0, 0.2, 201),
        interpRforce=True,
        interpzforce=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, w/ logR"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(-0.2, 0.2, 301),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, w/o zsym"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, w/o zsym"
    )
    # Test the interpolation of the potential, w/o zsym and with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        zgrid=(-0.2, 0.2, 201),
        interpRforce=True,
        interpzforce=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
    )
    return None


def test_interpolation_potential_force_diffinputs():
    # Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=True,
    )
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    # R vector, z scalar
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(rs, zs[10])
                - potential.evaluateRforces(
                    potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
                )
            )
            / potential.evaluateRforces(
                potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
            )
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of of Rforce w/ interpRZPotential fails for vector R and scalar Z"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(rs, zs[10])
                - potential.evaluatezforces(
                    potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
                )
            )
            / potential.evaluatezforces(
                potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
            )
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of of zforce w/ interpRZPotential fails for vector R and scalar Z"
    )
    # R scalar, z vector
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(rs[10], zs)
                - potential.evaluateRforces(
                    potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
                )
            )
            / potential.evaluateRforces(
                potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(rs[10], zs)
                - potential.evaluatezforces(
                    potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
                )
            )
            / potential.evaluatezforces(
                potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
            )
        )
        < 10.0**-6.0
    ), "RZPot interpolation w/ interpRZPotential fails for vector R and scalar Z"
    return None


# Test Rforce in C
def test_interpolation_potential_force_c():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 251),
        zgrid=(0.0, 0.2, 251),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
        zsym=True,
    )
    # Test within the grid, using vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C"
    )
    # now w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 251),
        zgrid=(-0.2, 0.2, 351),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
        zsym=False,
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(
                    potential.MWPotential,
                    mr,
                    mz,
                )
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C, w/o zsym"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C, w/o zsym"
    )
    # now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 351),
        logR=True,
        zgrid=(0.0, 0.2, 251),
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 10.0, 20)  # don't go too far
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation Rforcew/ interpRZPotential fails for vector input, using C, w/ logR"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation zforcew/ interpRZPotential fails for vector input, using C, w/ logR"
    )
    # now with logR and w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 351),
        logR=True,
        zgrid=(-0.2, 0.2, 351),
        interpRforce=True,
        interpzforce=True,
        enable_c=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 10.0, 20)  # don't go too far
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
    )
    return None


def test_interpolation_potential_force_c_vdiffgridsizes():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 391),
        zgrid=(0.0, 0.2, 262),
        logR=False,
        interpPot=True,
        enable_c=True,
        zsym=True,
    )
    # Test within the grid, using vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rforce(mr, mz)
                - potential.evaluateRforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), (
        "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C"
    )
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), (
        "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C"
    )
    return None


def test_interpolation_potential_force_use_c():
    # Test the interpolation of the potential, using C to calculate the grid
    rzpot_c = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=True,
        use_c=False,
    )
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=True,
        use_c=True,
    )
    assert numpy.all(
        numpy.fabs(rzpot._rforceGrid - rzpot_c._rforceGrid) < 10.0**-13.0
    ), (
        f"Potential interpolation grid of Rforce  calculated with use_c does not agree with that calculated in python, max diff = {numpy.amax(numpy.fabs(rzpot._rforceGrid - rzpot_c._rforceGrid))}"
    )
    assert numpy.all(
        numpy.fabs(rzpot._zforceGrid - rzpot_c._zforceGrid) < 10.0**-13.0
    ), (
        f"Potential interpolation grid of zforce  calculated with use_c does not agree with that calculated in python, max diff = {numpy.amax(numpy.fabs(rzpot._zforceGrid - rzpot_c._zforceGrid))}"
    )
    return None


# Test evaluation outside the grid
def test_interpolation_potential_force_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    zs = [-0.1, 0.3]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.Rforce(r, z)
                        - potential.evaluateRforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of Rforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of zforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
    return None


# Test evaluation outside the grid in C
def test_interpolation_potential_force_outsidegrid_c():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpRforce=True,
        interpzforce=True,
        zsym=False,
        enable_c=True,
    )
    rs = [0.005, 2.5]
    zs = [-0.1, 0.3]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.Rforce(r, z)
                        - potential.evaluateRforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of Rforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of zforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
    return None


def test_interpolation_potential_force_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpRforce=False,
        interpzforce=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    zs = [0.075, 0.15]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.Rforce(r, z)
                        - potential.evaluateRforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of Rforce w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
            )
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of zforce w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
            )
    return None


# Test RZderiv, taken from the origPot, so quite trivial
def test_interpolation_potential_rzderiv():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        zsym=True,
    )
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.Rzderiv(mr, mz)
                - potential.evaluateRzderivs(potential.MWPotential, mr, mz)
            )
            / potential.evaluateRzderivs(potential.MWPotential, mr, mz)
        )
        < 10.0**-10.0
    ), (
        "RZPot interpolation of Rzderiv (which is not an interpolation at all) w/ interpRZPotential fails for vector input"
    )
    return None


# Test the interpolated 2nd derivatives (R2deriv/z2deriv/Rzderiv; together the
# full 3D Hessian): like the forces, each is a precomputed grid of exact values
# interpolated with a 2D cubic spline; these checks are interpolation-limited
# (cf. the tol=-4 entries for mockInterpRZPotential in test_potential.py)
def test_interpolation_potential_secondderivs():
    # Python (RectBivariateSpline) interpolation path, linear R grid
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=True,
    )
    # On- and off-grid points, both z signs (Rzderiv is odd in z)
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    for name, finterp, fdirect in [
        ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
        ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
        ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
    ]:
        vinterp = finterp(mr, mz)
        vdirect = fdirect(potential.MWPotential, mr, mz)
        # Rzderiv crosses zero (odd in z), so guard the relative error
        relerr = numpy.amax(
            numpy.fabs(vinterp - vdirect) / (numpy.fabs(vdirect) + 1e-8)
        )
        assert relerr < 10.0**-3.0, (
            f"RZPot interpolation of {name} w/ interpRZPotential fails for vector input by {relerr:g}"
        )
    # logR grid
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        zgrid=(0.0, 0.2, 101),
        logR=True,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    for name, finterp, fdirect in [
        ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
        ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
        ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
    ]:
        vinterp = finterp(mr, mz)
        vdirect = fdirect(potential.MWPotential, mr, mz)
        relerr = numpy.amax(
            numpy.fabs(vinterp - vdirect) / (numpy.fabs(vdirect) + 1e-8)
        )
        assert relerr < 10.0**-5.0, (
            f"RZPot interpolation of {name} w/ interpRZPotential fails for vector input, w/ logR by {relerr:g}"
        )
    # w/o zsym (grid covers z<0 directly; covers the no-mirroring branch)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(-0.2, 0.2, 201),
        logR=False,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=False,
    )
    mr, mz = numpy.meshgrid(numpy.linspace(0.01, 2.0, 20), zs)
    mr = mr.flatten()
    mz = mz.flatten()
    for name, finterp, fdirect in [
        ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
        ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
        ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
    ]:
        vinterp = finterp(mr, mz)
        vdirect = fdirect(potential.MWPotential, mr, mz)
        relerr = numpy.amax(
            numpy.fabs(vinterp - vdirect) / (numpy.fabs(vdirect) + 1e-8)
        )
        # coarser z grid (same number of points over twice the range) ->
        # interpolation-limited at the few-x-1e-3 level near the sharp
        # disk midplane structure at small R
        assert relerr < 10.0**-2.0, (
            f"RZPot interpolation of {name} w/ interpRZPotential fails for vector input, w/o zsym by {relerr:g}"
        )
    return None


def test_interpolation_potential_secondderivs_c():
    # C (2D cubic B-spline) interpolation path: the same splines the C orbit
    # integrator uses for the 3D variational equations (hasC_dxdv3d)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        zgrid=(0.0, 0.2, 101),
        logR=True,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=True,
        use_c=True,
        enable_c=True,
    )
    assert rzpot.hasC_dxdv3d, (
        "interpRZPotential w/ all of pot/forces/2nd derivs interpolated and enable_c should advertise hasC_dxdv3d"
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)  # both z signs: Rzderiv is odd in z
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    for name, finterp, fdirect in [
        ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
        ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
        ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
    ]:
        vinterp = finterp(mr, mz)
        vdirect = fdirect(potential.MWPotential, mr, mz)
        relerr = numpy.amax(
            numpy.fabs(vinterp - vdirect) / (numpy.fabs(vdirect) + 1e-8)
        )
        # the C cubic B-spline uses mirror boundary conditions, which add an
        # O(h^2) boundary layer at the grid edges (and, for the odd-in-z
        # Rzderiv, at the z=0 mirror), so the C path is interpolation-limited
        # at the few-x-1e-3 level there (the interior agrees to ~1e-5); any
        # sign/wiring error would instead show up at O(1)
        assert relerr < 10.0**-2.0, (
            f"RZPot interpolation of {name} w/ interpRZPotential fails for vector input, w/ logR, in C by {relerr:g}"
        )
    # Unit-level finite differences of the C-interpolated forces vs the
    # C-interpolated 2nd derivatives at interior points: both interpolate the
    # same underlying potential very accurately there, so they agree to ~1e-6
    # relative (any sign/factor error in the Hessian wiring would be O(1)).
    # NB: points at exactly z=0 are excluded NOT because the interpolated
    # z2deriv is wrong there (it interpolates the exact z2deriv grid and is
    # correct), but because the FD of the C *zforce* spline degenerates there:
    # the B-spline mirror boundary condition at z=0 is even, so the zforce
    # spline has zero slope at exactly z=0 (a sub-grid-cell boundary layer;
    # the precomputed exact-z2deriv grid -- rather than a spline derivative of
    # the zforce -- is what makes the C Hessian correct at the midplane,
    # where orbits live).
    dx = 1e-6
    for r, z in [(0.7, 0.05), (1.0, -0.1), (2.3, 0.12), (5.0, 0.03)]:
        fdr2 = -(rzpot.Rforce(r + dx, z) - rzpot.Rforce(r - dx, z)) / (2.0 * dx)
        fdz2 = -(rzpot.zforce(r, z + dx) - rzpot.zforce(r, z - dx)) / (2.0 * dx)
        fdrz = -(rzpot.Rforce(r, z + dx) - rzpot.Rforce(r, z - dx)) / (2.0 * dx)
        for name, fd, vinterp in [
            ("R2deriv", fdr2, rzpot.R2deriv(r, z)),
            ("z2deriv", fdz2, rzpot.z2deriv(r, z)),
            ("Rzderiv", fdrz, rzpot.Rzderiv(r, z)),
        ]:
            assert numpy.fabs(fd - vinterp) / (numpy.fabs(vinterp) + 1e-4) < 1e-4, (
                f"C-interpolated {name} of interpRZPotential does not match the finite difference of the C-interpolated forces at (R,z) = ({r:g},{z:g}): {vinterp:g} vs {fd:g}"
            )
    # also exercise the linear-R (logR=False) branch of the C evaluators,
    # away from the grid edges (interior accuracy)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.5, 1.5, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=True,
        use_c=True,
        enable_c=True,
    )
    mr, mz = numpy.meshgrid(
        numpy.linspace(0.6, 1.4, 11), numpy.linspace(-0.15, 0.15, 11)
    )
    mr = mr.flatten()
    mz = mz.flatten()
    for name, finterp, fdirect in [
        ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
        ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
        ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
    ]:
        vinterp = finterp(mr, mz)
        vdirect = fdirect(potential.MWPotential, mr, mz)
        relerr = numpy.amax(
            numpy.fabs(vinterp - vdirect) / (numpy.fabs(vdirect) + 1e-8)
        )
        assert relerr < 10.0**-2.0, (
            f"RZPot interpolation of {name} w/ interpRZPotential fails for vector input, w/o logR, in C by {relerr:g}"
        )
    return None


def test_interpolation_potential_secondderivs_outsidegrid():
    # Outside the grid the 2nd derivatives fall back to the original potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.5, 1.5, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpR2deriv=True,
        interpz2deriv=True,
        interpRzderiv=True,
        zsym=True,
    )
    rs = [0.2, 1.8]
    zs = [-0.1, 0.1, 0.25, -0.25]
    for r in rs:
        for z in zs:
            for name, finterp, fdirect in [
                ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
                ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
                ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
            ]:
                assert (
                    numpy.fabs(
                        (finterp(r, z) - fdirect(potential.MWPotential, r, z))
                        / fdirect(potential.MWPotential, r, z)
                    )
                    < 10.0**-10.0
                ), (
                    f"RZPot interpolation of {name} w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
                )
    return None


def test_interpolation_potential_secondderivs_notinterpolated():
    # Without interpR2deriv/interpz2deriv/interpRzderiv the 2nd derivatives
    # pass through to the original potential (and there is no 3D C Hessian)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpPot=True,
        interpRforce=True,
        interpzforce=True,
        zsym=True,
        use_c=True,
        enable_c=True,
    )
    assert not rzpot.hasC_dxdv3d, (
        "interpRZPotential w/o interpolated 2nd derivatives should not advertise hasC_dxdv3d"
    )
    rs = [0.5, 1.5]
    zs = [0.075, 0.15, -0.15]
    for r in rs:
        for z in zs:
            for name, finterp, fdirect in [
                ("R2deriv", rzpot.R2deriv, potential.evaluateR2derivs),
                ("z2deriv", rzpot.z2deriv, potential.evaluatez2derivs),
                ("Rzderiv", rzpot.Rzderiv, potential.evaluateRzderivs),
            ]:
                assert (
                    numpy.fabs(
                        (finterp(r, z) - fdirect(potential.MWPotential, r, z))
                        / fdirect(potential.MWPotential, r, z)
                    )
                    < 10.0**-10.0
                ), (
                    f"RZPot {name} w/ interpRZPotential fails when the 2nd derivatives were not interpolated at (R,z) = ({r:g},{z:g})"
                )
    return None


# Test density
def test_interpolation_potential_dens():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpDens=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    zs = numpy.linspace(-0.2, 0.2, 41)
    for r in rs:
        for z in zs:
            densdiff = numpy.fabs(
                (
                    rzpot.dens(r, z)
                    - potential.evaluateDensities(potential.MWPotential, r, z)
                )
                / potential.evaluateDensities(potential.MWPotential, r, z)
            )
            assert densdiff < 10.0**-10.0, (
                f"RZPot interpolation of density of density w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {densdiff:g}"
            )
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    for r in rs:
        for z in zs:
            densdiff = numpy.fabs(
                (
                    rzpot.dens(r, z)
                    - potential.evaluateDensities(potential.MWPotential, r, z)
                )
                / potential.evaluateDensities(potential.MWPotential, r, z)
            )
            assert densdiff < 4.0 * 10.0**-6.0, (
                f"RZPot interpolation of density w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {densdiff:g}"
            )
    # Test all at the same time to use vector evaluation
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(mr, mz)
                - potential.evaluateDensities(potential.MWPotential, mr, mz)
            )
            / potential.evaluateDensities(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-6.0
    ), "RZPot interpolation of density w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        logR=True,
        zgrid=(0.0, 0.2, 201),
        interpDens=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(mr, mz)
                - potential.evaluateDensities(potential.MWPotential, mr, mz)
            )
            / potential.evaluateDensities(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-6.0
    ), (
        "RZPot interpolation of density w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, w/o zsym
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(-0.2, 0.2, 251),
        logR=False,
        interpDens=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(mr, mz)
                - potential.evaluateDensities(potential.MWPotential, mr, mz)
            )
            / potential.evaluateDensities(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-6.0
    ), (
        "RZPot interpolation of density w/ interpRZPotential fails for vector input, w/o zsym"
    )
    # Test the interpolation of the potential, w/o zsym and with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 251),
        logR=True,
        zgrid=(-0.2, 0.2, 201),
        interpDens=True,
        zsym=False,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    mr, mz = numpy.meshgrid(rs, zs)
    mr = mr.flatten()
    mz = mz.flatten()
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(mr, mz)
                - potential.evaluateDensities(potential.MWPotential, mr, mz)
            )
            / potential.evaluateDensities(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-6.0
    ), (
        "RZPot interpolation of density w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
    )
    return None


def test_interpolation_potential_dens_diffinputs():
    # Test the interpolation of the potential for different inputs: combination of vector and scalar (we've already done both scalars and both vectors above)
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        zgrid=(0.0, 0.2, 201),
        logR=False,
        interpDens=True,
        zsym=True,
    )
    # Test all at the same time to use vector evaluation
    rs = numpy.linspace(0.01, 2.0, 20)
    zs = numpy.linspace(-0.2, 0.2, 40)
    # R vector, z scalar
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(rs, zs[10])
                - potential.evaluateDensities(
                    potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
                )
            )
            / potential.evaluateDensities(
                potential.MWPotential, rs, zs[10] * numpy.ones(len(rs))
            )
        )
        < 4.0 * 10.0**-6.0
    ), (
        "RZPot interpolation of the density w/ interpRZPotential fails for vector R and scalar Z"
    )
    # R scalar, z vector
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.dens(rs[10], zs)
                - potential.evaluateDensities(
                    potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
                )
            )
            / potential.evaluateDensities(
                potential.MWPotential, rs[10] * numpy.ones(len(zs)), zs
            )
        )
        < 4.0 * 10.0**-6.0
    ), (
        "RZPot interpolation of the density w/ interpRZPotential fails for vector R and scalar Z"
    )
    return None


# Test evaluation outside the grid
def test_interpolation_potential_dens_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpDens=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    zs = [-0.1, 0.3]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.dens(r, z)
                        - potential.evaluateDensities(potential.MWPotential, r, z)
                    )
                    / potential.evaluateDensities(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of the density w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            )
    return None


def test_interpolation_potential_density_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 101),
        zgrid=(0.0, 0.2, 101),
        logR=False,
        interpDens=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    zs = [0.075, 0.15]
    for r in rs:
        for z in zs:
            assert (
                numpy.fabs(
                    (
                        rzpot.dens(r, z)
                        - potential.evaluateDensities(potential.MWPotential, r, z)
                    )
                    / potential.evaluateDensities(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), (
                f"RZPot interpolation of the density w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
            )
    return None


# Test the circular velocity
def test_interpolation_potential_vcirc():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpvcirc=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    for r in rs:
        assert (
            numpy.fabs(
                (rzpot.vcirc(r) - potential.vcirc(potential.MWPotential, r))
                / potential.vcirc(potential.MWPotential, r)
            )
            < 10.0**-10.0
        ), "RZPot interpolation of vcirc w/ interpRZPotential fails at R = %g" % (r)
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    for r in rs:
        vcdiff = numpy.fabs(
            (rzpot.vcirc(r) - potential.vcirc(potential.MWPotential, r))
            / potential.vcirc(potential.MWPotential, r)
        )
        assert vcdiff < 10.0**-6.0, (
            f"RZPot interpolation of vcirc w/ interpRZPotential fails at R = {r:g} by {vcdiff:g}"
        )
    # Test all at the same time to use vector evaluation
    assert numpy.all(
        numpy.fabs(
            (rzpot.vcirc(rs) - potential.vcirc(potential.MWPotential, rs))
            / potential.vcirc(potential.MWPotential, rs)
        )
        < 10.0**-6.0
    ), "RZPot interpolation of vcirc w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        interpvcirc=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.vcirc(rs) - potential.vcirc(potential.MWPotential, rs))
            / potential.vcirc(potential.MWPotential, rs)
        )
        < 10.0**-6.0
    ), (
        "RZPot interpolation of vcirc w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, with numcores
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpvcirc=True,
        numcores=1,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.vcirc(rs) - potential.vcirc(potential.MWPotential, rs))
            / potential.vcirc(potential.MWPotential, rs)
        )
        < 10.0**-6.0
    ), "RZPot interpolation of vcirc w/ interpRZPotential fails for vector input"
    return None


# Test evaluation outside the grid
def test_interpolation_potential_vcirc_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpvcirc=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    for r in rs:
        vcdiff = numpy.fabs(
            (rzpot.vcirc(r) - potential.vcirc(potential.MWPotential, r))
            / potential.vcirc(potential.MWPotential, r)
        )
        assert vcdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {vcdiff:g}"
        )
    return None


def test_interpolation_potential_vcirc_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpvcirc=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    for r in rs:
        vcdiff = numpy.fabs(
            (rzpot.vcirc(r) - potential.vcirc(potential.MWPotential, r))
            / potential.vcirc(potential.MWPotential, r)
        )
        assert vcdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {vcdiff:g}"
        )
    return None


# Test dvcircdR
def test_interpolation_potential_dvcircdR():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpdvcircdr=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    for r in rs:
        assert (
            numpy.fabs(
                (rzpot.dvcircdR(r) - potential.dvcircdR(potential.MWPotential, r))
                / potential.dvcircdR(potential.MWPotential, r)
            )
            < 10.0**-10.0
        ), "RZPot interpolation of dvcircdR w/ interpRZPotential fails at R = %g" % (r)
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    for r in rs:
        dvcdrdiff = numpy.fabs(
            (rzpot.dvcircdR(r) - potential.dvcircdR(potential.MWPotential, r))
            / potential.dvcircdR(potential.MWPotential, r)
        )
        assert dvcdrdiff < 10.0**-5.0, (
            f"RZPot interpolation of dvcircdR w/ interpRZPotential fails at R = {r:g} by {dvcdrdiff:g}"
        )
    # Test all at the same time to use vector evaluation
    assert numpy.all(
        numpy.fabs(
            (rzpot.dvcircdR(rs) - potential.dvcircdR(potential.MWPotential, rs))
            / potential.dvcircdR(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of dvcircdR w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        interpdvcircdr=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.dvcircdR(rs) - potential.dvcircdR(potential.MWPotential, rs))
            / potential.dvcircdR(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of dvcircdR w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, with numcores
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpdvcircdr=True,
        numcores=1,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.dvcircdR(rs) - potential.dvcircdR(potential.MWPotential, rs))
            / potential.dvcircdR(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of dvcircdR w/ interpRZPotential fails for vector input"
    return None


# Test evaluation outside the grid
def test_interpolation_potential_dvcircdR_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpdvcircdr=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    for r in rs:
        dvcdrdiff = numpy.fabs(
            (rzpot.dvcircdR(r) - potential.dvcircdR(potential.MWPotential, r))
            / potential.dvcircdR(potential.MWPotential, r)
        )
        assert dvcdrdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {dvcdrdiff:g}"
        )
    return None


def test_interpolation_potential_dvcircdR_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpdvcircdr=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    for r in rs:
        dvcdrdiff = numpy.fabs(
            (rzpot.dvcircdR(r) - potential.dvcircdR(potential.MWPotential, r))
            / potential.dvcircdR(potential.MWPotential, r)
        )
        assert dvcdrdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {dvcdrdiff:g}"
        )
    return None


# Test epifreq
def test_interpolation_potential_epifreq():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpepifreq=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    for r in rs:
        assert (
            numpy.fabs(
                (rzpot.epifreq(r) - potential.epifreq(potential.MWPotential, r))
                / potential.epifreq(potential.MWPotential, r)
            )
            < 10.0**-10.0
        ), "RZPot interpolation of epifreq w/ interpRZPotential fails at R = %g" % (r)
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    for r in rs:
        epidiff = numpy.fabs(
            (rzpot.epifreq(r) - potential.epifreq(potential.MWPotential, r))
            / potential.epifreq(potential.MWPotential, r)
        )
        assert epidiff < 10.0**-5.0, (
            f"RZPot interpolation of epifreq w/ interpRZPotential fails at R = {r:g} by {epidiff:g}"
        )
    # Test all at the same time to use vector evaluation
    assert numpy.all(
        numpy.fabs(
            (rzpot.epifreq(rs) - potential.epifreq(potential.MWPotential, rs))
            / potential.epifreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        interpepifreq=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.epifreq(rs) - potential.epifreq(potential.MWPotential, rs))
            / potential.epifreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, with numcores
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpepifreq=True,
        numcores=1,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.epifreq(rs) - potential.epifreq(potential.MWPotential, rs))
            / potential.epifreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input"
    return None


# Test epifreq setup when the number of r points is small
def test_interpolation_potential_epifreq_smalln():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(1.0, 1.3, 3),
        logR=False,
        interpepifreq=True,
        zsym=False,
    )
    rs = numpy.linspace(1.1, 1.2, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.epifreq(rs) - potential.epifreq(potential.MWPotential, rs))
            / potential.epifreq(potential.MWPotential, rs)
        )
        < 10.0**-2.0
    ), (
        "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input"
    )  # not as harsh, bc we don't have many points
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(1.0), numpy.log(1.3), 3),
        logR=True,
        interpepifreq=True,
        zsym=False,
    )
    rs = numpy.linspace(1.1, 1.2, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.epifreq(rs) - potential.epifreq(potential.MWPotential, rs))
            / potential.epifreq(potential.MWPotential, rs)
        )
        < 10.0**-2.0
    ), (
        "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input"
    )  # not as harsh, bc we don't have many points
    return None


# Test evaluation outside the grid
def test_interpolation_potential_epifreq_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpepifreq=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    for r in rs:
        epidiff = numpy.fabs(
            (rzpot.epifreq(r) - potential.epifreq(potential.MWPotential, r))
            / potential.epifreq(potential.MWPotential, r)
        )
        assert epidiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {epidiff:g}"
        )
    return None


def test_interpolation_potential_epifreq_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpepifreq=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    for r in rs:
        epidiff = numpy.fabs(
            (rzpot.epifreq(r) - potential.epifreq(potential.MWPotential, r))
            / potential.epifreq(potential.MWPotential, r)
        )
        assert epidiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {epidiff:g}"
        )
    return None


# Test verticalfreq
def test_interpolation_potential_verticalfreq():
    # Test the interpolation of the potential
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpverticalfreq=True,
        zsym=True,
    )
    # This just tests on the grid
    rs = numpy.linspace(0.01, 2.0, 21)
    for r in rs:
        assert (
            numpy.fabs(
                (
                    rzpot.verticalfreq(r)
                    - potential.verticalfreq(potential.MWPotential, r)
                )
                / potential.verticalfreq(potential.MWPotential, r)
            )
            < 10.0**-10.0
        ), (
            "RZPot interpolation of verticalfreq w/ interpRZPotential fails at R = %g"
            % (r)
        )
    # This tests within the grid
    rs = numpy.linspace(0.01, 2.0, 20)
    for r in rs:
        vfdiff = numpy.fabs(
            (rzpot.verticalfreq(r) - potential.verticalfreq(potential.MWPotential, r))
            / potential.verticalfreq(potential.MWPotential, r)
        )
        assert vfdiff < 10.0**-5.0, (
            f"RZPot interpolation of verticalfreq w/ interpRZPotential fails at R = {r:g} by {vfdiff:g}"
        )
    # Test all at the same time to use vector evaluation
    assert numpy.all(
        numpy.fabs(
            (rzpot.verticalfreq(rs) - potential.verticalfreq(potential.MWPotential, rs))
            / potential.verticalfreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of verticalfreq w/ interpRZPotential fails for vector input"
    # Test the interpolation of the potential, now with logR
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(numpy.log(0.01), numpy.log(20.0), 201),
        logR=True,
        interpverticalfreq=True,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 20.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.verticalfreq(rs) - potential.verticalfreq(potential.MWPotential, rs))
            / potential.verticalfreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), (
        "RZPot interpolation of verticalfreq w/ interpRZPotential fails for vector input, w/ logR"
    )
    # Test the interpolation of the potential, with numcores
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpverticalfreq=True,
        numcores=1,
        zsym=True,
    )
    rs = numpy.linspace(0.01, 2.0, 20)
    assert numpy.all(
        numpy.fabs(
            (rzpot.verticalfreq(rs) - potential.verticalfreq(potential.MWPotential, rs))
            / potential.verticalfreq(potential.MWPotential, rs)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of verticalfreq w/ interpRZPotential fails for vector input"
    return None


# Test evaluation outside the grid
def test_interpolation_potential_verticalfreq_outsidegrid():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpverticalfreq=True,
        zsym=False,
    )
    rs = [0.005, 2.5]
    for r in rs:
        vfdiff = numpy.fabs(
            (rzpot.verticalfreq(r) - potential.verticalfreq(potential.MWPotential, r))
            / potential.verticalfreq(potential.MWPotential, r)
        )
        assert vfdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {vfdiff:g}"
        )
    return None


def test_interpolation_potential_verticalfreq_notinterpolated():
    rzpot = potential.interpRZPotential(
        RZPot=potential.MWPotential,
        rgrid=(0.01, 2.0, 201),
        logR=False,
        interpverticalfreq=False,
        zsym=True,
    )
    rs = [0.5, 1.5]
    for r in rs:
        vfdiff = numpy.fabs(
            (rzpot.verticalfreq(r) - potential.verticalfreq(potential.MWPotential, r))
            / potential.verticalfreq(potential.MWPotential, r)
        )
        assert vfdiff < 10.0**-10.0, (
            f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {vfdiff:g}"
        )
    return None


# Regression test for the units bug: interpRZPotential built from a potential
# with ro/vo set stored PHYSICAL values in its grids (and in the off-grid
# fallback), while the interpolator is queried in internal units. The
# potential came out a factor vo^2 too large, forces by force_in_kmsMyr, and
# dens/R2deriv/epifreq/verticalfreq were wrong regardless of use_c because
# those grids have no C implementation. See galpy #212.
def test_interpRZPotential_units_set_matches_unitless():
    import numpy

    from galpy.potential import MiyamotoNagaiPotential, interpRZPotential

    grid = dict(rgrid=(0.5, 1.5, 21), zgrid=(0.0, 0.2, 11), logR=False)
    with_units = MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0, ro=8.0, vo=220.0)
    unitless = MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0)

    # (R, z) on the grid and off it -- the off-grid branch falls back to the
    # original potential and had the same defect.
    checks = [
        (
            dict(interpPot=True, use_c=False),
            lambda p, R, z: p(R, z, use_physical=False),
        ),
        (dict(interpPot=True, use_c=True), lambda p, R, z: p(R, z, use_physical=False)),
        (
            dict(interpRforce=True, use_c=False),
            lambda p, R, z: p.Rforce(R, z, use_physical=False),
        ),
        (
            dict(interpzforce=True, use_c=False),
            lambda p, R, z: p.zforce(R, z, use_physical=False),
        ),
        (dict(interpDens=True), lambda p, R, z: p.dens(R, z, use_physical=False)),
        (dict(interpR2deriv=True), lambda p, R, z: p.R2deriv(R, z, use_physical=False)),
        (dict(interpepifreq=True), lambda p, R, z: p.epifreq(R, use_physical=False)),
        (
            dict(interpverticalfreq=True),
            lambda p, R, z: p.verticalfreq(R, use_physical=False),
        ),
    ]
    for kwargs, evaluate in checks:
        ip_units = interpRZPotential(RZPot=with_units, **grid, **kwargs)
        ip_plain = interpRZPotential(RZPot=unitless, **grid, **kwargs)
        for R, z in [(1.05, 0.07), (3.0, 0.5)]:  # on-grid, then off-grid
            got = float(evaluate(ip_units, R, z))
            want = float(evaluate(ip_plain, R, z))
            assert numpy.fabs(got - want) < 1e-12 * numpy.fabs(want), (
                f"interpRZPotential with ro/vo set disagrees with the unitless "
                f"build for {kwargs} at (R,z)=({R},{z}): {got} vs {want}"
            )


# The vectorised grid build must fall back to cell-by-cell sampling for a
# potential that does not broadcast. AnySphericalPotential is the hard case: it
# neither raises nor broadcasts correctly for the forces -- it silently returns
# different numbers for array input -- so a try/except alone is not enough and
# the whole interpolation grid would be built from wrong values.
def test_interpRZPotential_grid_falls_back_for_nonbroadcasting_potential():
    import importlib

    import numpy

    from galpy.potential import AnySphericalPotential, interpRZPotential

    M = importlib.import_module("galpy.potential.interpRZPotential")
    pot = AnySphericalPotential(
        dens=lambda r: 1.0 / 4.0 / numpy.pi / r**2.0 / (1.0 + r) ** 2.0, normalize=1.0
    )
    kw = dict(
        interpRforce=True,
        interpzforce=True,
        rgrid=(numpy.log(0.05), numpy.log(5.0), 11),
        zgrid=(0.0, 0.5, 7),
        logR=True,
        use_c=False,
        enable_c=False,
        zsym=True,
    )
    got = interpRZPotential(RZPot=pot, **kw)
    # Reference: the pure cell-by-cell path, which is what the grids must equal.
    real = M._grid_eval
    M._grid_eval = lambda ev, p, rg, zg: numpy.array(
        [[ev(p, r, z, use_physical=False) for z in zg] for r in rg]
    )
    try:
        ref = interpRZPotential(RZPot=pot, **kw)
    finally:
        M._grid_eval = real
    for g in ("_rforceGrid", "_zforceGrid"):
        assert numpy.array_equal(getattr(got, g), getattr(ref, g), equal_nan=True), (
            f"interpRZPotential {g} does not match the cell-by-cell reference for "
            "a non-broadcasting potential; the vectorised path was accepted when "
            "it should have fallen back"
        )
    return None


def test_grid_spot_check_cells_covers_every_grid_size():
    """`_spot_check_cells` must be in-bounds and useful at ANY grid size.

    The sample replaced a two-full-row check (2*nz cells, 502 at the default
    251x251) with ~19 cells, so its behaviour at the extremes is the thing that
    needs pinning, not its behaviour at 251.
    """
    import importlib

    M = importlib.import_module("galpy.potential.interpRZPotential")

    # 1. in-bounds and duplicate-free everywhere, including 1-wide grids
    for nR in list(range(1, 14)) + [21, 51, 251, 1001]:
        for nz in list(range(1, 14)) + [21, 51, 251, 1001]:
            cells = M._spot_check_cells(nR, nz)
            assert len(cells) == len(set(cells)), f"duplicates at {nR}x{nz}"
            for ii, jj in cells:
                assert 0 <= ii < nR and 0 <= jj < nz, (
                    f"_spot_check_cells({nR},{nz}) returned out-of-bounds ({ii},{jj})"
                )

    # 2. degenerate grid samples NOTHING: nR-1 would be -1, and an empty grid is
    #    already rejected downstream by the spline fitter, which names the real
    #    problem ("(mx>kx) failed ... mx=0"). Sampling here would preempt that
    #    with a bare IndexError from an internal helper.
    for nR, nz in ((0, 5), (5, 0), (0, 0)):
        assert M._spot_check_cells(nR, nz) == [], f"{nR}x{nz} should sample nothing"

    # 3. small grids are covered EXHAUSTIVELY (the index set dedupes), so the
    #    cheap sample never trades away coverage where checking is cheap anyway
    for nR, nz in ((1, 1), (2, 2), (3, 3), (2, 5), (3, 5)):
        assert len(M._spot_check_cells(nR, nz)) == nR * nz, (
            f"{nR}x{nz} should be exhaustive"
        )

    # 4. and it saturates rather than growing with the grid -- the whole point
    assert len(M._spot_check_cells(251, 251)) == len(M._spot_check_cells(1001, 1001))
    assert len(M._spot_check_cells(251, 251)) < 25, "sample should stay ~19 cells"

    # 5. both R edges and both z edges are always sampled: a broadcasting bug
    #    that only bites at a boundary must not sit between samples
    cells = M._spot_check_cells(251, 251)
    assert any(ii == 0 for ii, _ in cells) and any(ii == 250 for ii, _ in cells)
    assert any(jj == 0 for _, jj in cells) and any(jj == 250 for _, jj in cells)
    return None


def test_grid_eval_falls_back_when_the_vectorised_call_returns_a_bad_shape():
    """A vectorised call that neither raises nor returns (nR, nz) must fall back.

    `_grid_eval` has three fallbacks and the other two are exercised by real
    potentials: `AnySphericalPotential` RAISES on an array (the try/except), and
    the non-broadcasting case above is caught by the bit-for-bit spot check. The
    third -- a call that returns successfully with the WRONG SHAPE -- has no
    potential in the zoo that does it, so it needs a synthetic evaluator or it
    stays untested.

    That branch matters: without it a wrong-shaped result would flow into
    RectBivariateSpline and fail far from the cause, or silently broadcast.
    """
    import importlib

    import numpy

    M = importlib.import_module("galpy.potential.interpRZPotential")
    rg = numpy.linspace(0.3, 1.4, 5)
    zg = numpy.linspace(0.0, 0.2, 4)
    pot = potential.MiyamotoNagaiPotential(normalize=1.0)

    calls = {"n": 0}

    def bad_shape(p, R, z, use_physical=False):
        # Array call -> return a transposed/flat result of the wrong shape;
        # scalar calls (the loop) behave normally so the fallback can succeed.
        if numpy.ndim(R) > 0:
            calls["n"] += 1
            return numpy.zeros(numpy.size(R) + 1)
        return potential.evaluatePotentials(p, R, z, use_physical=False)

    got = M._grid_eval(bad_shape, pot, rg, zg)
    assert calls["n"] == 1, "the vectorised call should be attempted exactly once"
    assert got.shape == (len(rg), len(zg)), "fallback must produce the (nR, nz) grid"
    # and it must be the true cell-by-cell answer, not the zeros the bad call gave
    ref = numpy.array(
        [
            [potential.evaluatePotentials(pot, r, z, use_physical=False) for z in zg]
            for r in rg
        ]
    )
    assert numpy.array_equal(got, ref), (
        "wrong-shaped vectorised result was accepted instead of falling back"
    )
    assert not numpy.any(got == 0.0), "fallback returned the bad call's zeros"
    return None
