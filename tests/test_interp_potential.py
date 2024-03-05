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
            ), f"RZPot interpolation w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
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
    ), "RZPot interpolation w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
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
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/o zsym"
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
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR"
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
    ), "RZPot interpolation w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
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
    assert numpy.all(
        numpy.fabs(rzpot._potGrid - rzpot_c._potGrid) < 10.0**-14.0
    ), "Potential interpolation grid calculated with use_c does not agree with that calculated in python"
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
            ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation of Rforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluateRforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), f"RZPot interpolation of zforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g})"
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
            assert (
                rforcediff < 10.0**-5.0
            ), f"RZPot interpolation of Rforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {rforcediff:g}"
            zforcediff = numpy.fabs(
                (
                    rzpot.zforce(r, z)
                    - potential.evaluatezforces(potential.MWPotential, r, z)
                )
                / potential.evaluatezforces(potential.MWPotential, r, z)
            )
            assert (
                zforcediff < 5.0 * 10.0**-5.0
            ), f"RZPot interpolation of zforce w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {zforcediff:g}"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, w/ logR"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, w/ logR"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, w/o zsym"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 4.0 * 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, w/o zsym"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
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
    ), "RZPot interpolation of of Rforce w/ interpRZPotential fails for vector R and scalar Z"
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
    ), "RZPot interpolation of of zforce w/ interpRZPotential fails for vector R and scalar Z"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C, w/o zsym"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C, w/o zsym"
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
    ), "RZPot interpolation Rforcew/ interpRZPotential fails for vector input, using C, w/ logR"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-5.0
    ), "RZPot interpolation zforcew/ interpRZPotential fails for vector input, using C, w/ logR"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 2.0 * 10.0**-5.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C, w/ logR, and w/o zsym"
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
    ), "RZPot interpolation of Rforce w/ interpRZPotential fails for vector input, using C"
    assert numpy.all(
        numpy.fabs(
            (
                rzpot.zforce(mr, mz)
                - potential.evaluatezforces(potential.MWPotential, mr, mz)
            )
            / potential.evaluatezforces(potential.MWPotential, mr, mz)
        )
        < 10.0**-6.0
    ), "RZPot interpolation of zforce w/ interpRZPotential fails for vector input, using C"
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
    ), f"Potential interpolation grid of Rforce  calculated with use_c does not agree with that calculated in python, max diff = {numpy.amax(numpy.fabs(rzpot._rforceGrid-rzpot_c._rforceGrid))}"
    assert numpy.all(
        numpy.fabs(rzpot._zforceGrid - rzpot_c._zforceGrid) < 10.0**-13.0
    ), f"Potential interpolation grid of zforce  calculated with use_c does not agree with that calculated in python, max diff = {numpy.amax(numpy.fabs(rzpot._zforceGrid-rzpot_c._zforceGrid))}"
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
            ), f"RZPot interpolation of Rforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), f"RZPot interpolation of zforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation of Rforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), f"RZPot interpolation of zforce w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation of Rforce w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
            assert (
                numpy.fabs(
                    (
                        rzpot.zforce(r, z)
                        - potential.evaluatezforces(potential.MWPotential, r, z)
                    )
                    / potential.evaluatezforces(potential.MWPotential, r, z)
                )
                < 10.0**-10.0
            ), f"RZPot interpolation of zforce w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
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
    ), "RZPot interpolation of Rzderiv (which is not an interpolation at all) w/ interpRZPotential fails for vector input"
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
            assert (
                densdiff < 10.0**-10.0
            ), f"RZPot interpolation of density of density w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {densdiff:g}"
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
            assert (
                densdiff < 4.0 * 10.0**-6.0
            ), f"RZPot interpolation of density w/ interpRZPotential fails at (R,z) = ({r:g},{z:g}) by {densdiff:g}"
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
    ), "RZPot interpolation of density w/ interpRZPotential fails for vector input, w/ logR"
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
    ), "RZPot interpolation of density w/ interpRZPotential fails for vector input, w/o zsym"
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
    ), "RZPot interpolation of density w/ interpRZPotential fails for vector input w/o zsym and w/ logR"
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
    ), "RZPot interpolation of the density w/ interpRZPotential fails for vector R and scalar Z"
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
    ), "RZPot interpolation of the density w/ interpRZPotential fails for vector R and scalar Z"
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
            ), f"RZPot interpolation of the density w/ interpRZPotential fails outside the grid at (R,z) = ({r:g},{z:g})"
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
            ), f"RZPot interpolation of the density w/ interpRZPotential fails when the potential was not interpolated at (R,z) = ({r:g},{z:g})"
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
        assert (
            vcdiff < 10.0**-6.0
        ), f"RZPot interpolation of vcirc w/ interpRZPotential fails at R = {r:g} by {vcdiff:g}"
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
    ), "RZPot interpolation of vcirc w/ interpRZPotential fails for vector input, w/ logR"
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
        assert (
            vcdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {vcdiff:g}"
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
        assert (
            vcdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {vcdiff:g}"
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
        assert (
            dvcdrdiff < 10.0**-5.0
        ), f"RZPot interpolation of dvcircdR w/ interpRZPotential fails at R = {r:g} by {dvcdrdiff:g}"
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
    ), "RZPot interpolation of dvcircdR w/ interpRZPotential fails for vector input, w/ logR"
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
        assert (
            dvcdrdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {dvcdrdiff:g}"
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
        assert (
            dvcdrdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {dvcdrdiff:g}"
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
        assert (
            epidiff < 10.0**-5.0
        ), f"RZPot interpolation of epifreq w/ interpRZPotential fails at R = {r:g} by {epidiff:g}"
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
    ), "RZPot interpolation of epifreq w/ interpRZPotential fails for vector input, w/ logR"
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
        assert (
            epidiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {epidiff:g}"
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
        assert (
            epidiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {epidiff:g}"
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
        assert (
            vfdiff < 10.0**-5.0
        ), f"RZPot interpolation of verticalfreq w/ interpRZPotential fails at R = {r:g} by {vfdiff:g}"
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
    ), "RZPot interpolation of verticalfreq w/ interpRZPotential fails for vector input, w/ logR"
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
        assert (
            vfdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails outside the grid at R = {r:g} by {vfdiff:g}"
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
        assert (
            vfdiff < 10.0**-10.0
        ), f"RZPot interpolation w/ interpRZPotential fails when the potential was not interpolated at R = {r:g} by {vfdiff:g}"
    return None
