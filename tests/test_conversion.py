import numpy

from galpy.util import conversion


def test_dens_in_criticaldens():
    # Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.dens_in_criticaldens(vofid, rofid)
            / conversion.dens_in_criticaldens(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_criticaldens did not work as expected"
    assert (
        numpy.fabs(
            0.25
            * conversion.dens_in_criticaldens(vofid, rofid)
            / conversion.dens_in_criticaldens(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_critical did not work as expected"
    return None


def test_dens_in_meanmatterdens():
    # Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.dens_in_meanmatterdens(vofid, rofid)
            / conversion.dens_in_meanmatterdens(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_meanmatterdens did not work as expected"
    assert (
        numpy.fabs(
            0.25
            * conversion.dens_in_meanmatterdens(vofid, rofid)
            / conversion.dens_in_meanmatterdens(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_meanmatter did not work as expected"
    return None


def test_dens_in_gevcc():
    # Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.dens_in_gevcc(vofid, rofid)
            / conversion.dens_in_gevcc(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_gevcc did not work as expected"
    assert (
        numpy.fabs(
            0.25
            * conversion.dens_in_gevcc(vofid, rofid)
            / conversion.dens_in_gevcc(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_gevcc did not work as expected"
    return None


def test_dens_in_msolpc3():
    # Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.dens_in_msolpc3(vofid, rofid)
            / conversion.dens_in_msolpc3(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_msolpc3 did not work as expected"
    assert (
        numpy.fabs(
            0.25
            * conversion.dens_in_msolpc3(vofid, rofid)
            / conversion.dens_in_msolpc3(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "dens_in_msolpc3 did not work as expected"
    return None


def test_force_in_2piGmsolpc2():
    # Test the scaling, as a 1st derivative of the potential / G, should scale as velocity^2/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.force_in_2piGmsolpc2(vofid, rofid)
            / conversion.force_in_2piGmsolpc2(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_2piGmsolpc2 did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.force_in_2piGmsolpc2(vofid, rofid)
            / conversion.force_in_2piGmsolpc2(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_2piGmsolpc2 did not work as expected"
    return None


def test_force_in_pcMyr2():
    # Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.force_in_pcMyr2(vofid, rofid)
            / conversion.force_in_pcMyr2(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_pcMyr2 did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.force_in_pcMyr2(vofid, rofid)
            / conversion.force_in_pcMyr2(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_pcMyr2 did not work as expected"
    return None


def test_force_in_kmsMyr():
    # Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.force_in_kmsMyr(vofid, rofid)
            / conversion.force_in_kmsMyr(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_kmsMyr did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.force_in_kmsMyr(vofid, rofid)
            / conversion.force_in_kmsMyr(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_kmsMyr did not work as expected"
    return None


def test_force_in_10m13kms2():
    # Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.force_in_10m13kms2(vofid, rofid)
            / conversion.force_in_10m13kms2(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_10m13kms2 did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.force_in_10m13kms2(vofid, rofid)
            / conversion.force_in_10m13kms2(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "force_in_10m13kms2 did not work as expected"
    return None


def test_freq_in_Gyr():
    # Test the scaling, as 1/time, should scale as velocity/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            2.0
            * conversion.freq_in_Gyr(vofid, rofid)
            / conversion.freq_in_Gyr(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "freq_in_Gyr did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.freq_in_Gyr(vofid, rofid)
            / conversion.freq_in_Gyr(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "freq_in_Gyr did not work as expected"
    return None


def test_freq_in_kmskpc():
    # Test the scaling, as 1/time, should scale as velocity/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            2.0
            * conversion.freq_in_kmskpc(vofid, rofid)
            / conversion.freq_in_kmskpc(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "freq_in_kmskpc did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.freq_in_kmskpc(vofid, rofid)
            / conversion.freq_in_kmskpc(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "freq_in_kmskpc did not work as expected"
    return None


def test_surfdens_in_msolpc2():
    # Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.surfdens_in_msolpc2(vofid, rofid)
            / conversion.surfdens_in_msolpc2(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "surfdens_in_msolpc2 did not work as expected"
    assert (
        numpy.fabs(
            0.5
            * conversion.surfdens_in_msolpc2(vofid, rofid)
            / conversion.surfdens_in_msolpc2(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "surfdens_in_msolpc2 did not work as expected"
    return None


def test_mass_in_msol():
    # Test the scaling, should be velocity^2 x position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.mass_in_msol(vofid, rofid)
            / conversion.mass_in_msol(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "mass_in_msol did not work as expected"
    assert (
        numpy.fabs(
            2.0
            * conversion.mass_in_msol(vofid, rofid)
            / conversion.mass_in_msol(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "mass_in_msol did not work as expected"
    return None


def test_mass_in_1010msol():
    # Test the scaling, should be velocity^2 x position
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            4.0
            * conversion.mass_in_1010msol(vofid, rofid)
            / conversion.mass_in_1010msol(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "mass_in_1010msol did not work as expected"
    assert (
        numpy.fabs(
            2.0
            * conversion.mass_in_1010msol(vofid, rofid)
            / conversion.mass_in_1010msol(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "mass_in_1010msol did not work as expected"
    return None


def test_time_in_Gyr():
    # Test the scaling, should scale as position/velocity
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            0.5
            * conversion.time_in_Gyr(vofid, rofid)
            / conversion.time_in_Gyr(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "time_in_Gyr did not work as expected"
    assert (
        numpy.fabs(
            2.0
            * conversion.time_in_Gyr(vofid, rofid)
            / conversion.time_in_Gyr(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "time_in_Gyr did not work as expected"
    return None


def test_velocity_in_kpcGyr():
    # Test the scaling, should scale as velocity
    vofid, rofid = 200.0, 8.0
    assert (
        numpy.fabs(
            2.0
            * conversion.velocity_in_kpcGyr(vofid, rofid)
            / conversion.velocity_in_kpcGyr(2.0 * vofid, rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "velocity_in_kpcGyr did not work as expected"
    assert (
        numpy.fabs(
            conversion.velocity_in_kpcGyr(vofid, rofid)
            / conversion.velocity_in_kpcGyr(vofid, 2 * rofid)
            - 1.0
        )
        < 10.0**-10.0
    ), "velocity_in_kpcGyr did not work as expected"
    return None


def test_get_physical():
    # Test that the get_physical function returns the right scaling parameters
    # Potential and variations thereof
    from galpy.potential import DehnenBarPotential, MWPotential2014
    from galpy.util.conversion import get_physical

    dp = DehnenBarPotential
    assert numpy.fabs(get_physical(MWPotential2014[0]).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014[0]).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    ro, vo = 9.0, 230.0
    dp = DehnenBarPotential(ro=ro, vo=vo)
    assert numpy.fabs(get_physical(dp).get("ro") - ro) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(dp).get("vo") - vo) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014 + dp).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014 + dp).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014 + dp).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    assert numpy.fabs(get_physical(MWPotential2014 + dp).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a Potential"
    )
    # Orbits
    from galpy.orbit import Orbit

    ro, vo = 10.0, 210.0
    o = Orbit(ro=ro, vo=vo)
    assert numpy.fabs(get_physical(o).get("ro") - ro) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an Orbit"
    )
    assert numpy.fabs(get_physical(o).get("vo") - vo) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an Orbit"
    )
    # even though one shouldn't do this, let's test a list
    assert numpy.fabs(get_physical([o, o]).get("ro") - ro) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an Orbit"
    )
    assert numpy.fabs(get_physical([o, o]).get("vo") - vo) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an Orbit"
    )
    # actionAngle
    from galpy.actionAngle import actionAngleStaeckel

    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    assert numpy.fabs(get_physical(aAS).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    )
    assert numpy.fabs(get_physical(aAS).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    )
    # This doesn't make much sense, but let's test...
    ro, vo = 19.0, 130.0
    dp = DehnenBarPotential(ro=ro, vo=vo)
    aAS = actionAngleStaeckel(pot=dp, delta=0.45, ro=ro, vo=vo)
    assert numpy.fabs(get_physical(aAS).get("ro") - ro) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    )
    assert numpy.fabs(get_physical(aAS).get("vo") - vo) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    )
    # DF
    from galpy.df import quasiisothermaldf

    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    qdf = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, aA=aAS, pot=MWPotential2014)
    assert numpy.fabs(get_physical(qdf).get("ro") - 8.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a DF instance"
    )
    assert numpy.fabs(get_physical(qdf).get("vo") - 220.0) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a DF instance"
    )
    # non-standard ro,vo
    from galpy.potential import MiyamotoNagaiPotential

    ro, vo = 4.0, 330.0
    mp = MiyamotoNagaiPotential(a=0.5, b=0.1, ro=ro, vo=vo)
    aAS = actionAngleStaeckel(pot=mp, delta=0.45, ro=ro, vo=vo)
    qdf = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, aA=aAS, pot=mp, ro=ro, vo=vo)
    assert numpy.fabs(get_physical(qdf).get("ro") - ro) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a DF instance"
    )
    assert numpy.fabs(get_physical(qdf).get("vo") - vo) < 1e-10, (
        "get_physical does not return the correct unit conversion parameter for a DF instance"
    )
    return None


def test_physical_compatible_potential():
    # Test that physical_compatible acts as expected
    from galpy.potential import HernquistPotential
    from galpy.util.conversion import physical_compatible

    # Set up potentials for all possible cases
    pot_default_phys = HernquistPotential(amp=0.55, a=2.0, ro=8.0, vo=220.0)
    pot_nonstandardro = HernquistPotential(amp=0.55, a=2.0, ro=9.0, vo=220.0)
    pot_nonstandardvo = HernquistPotential(amp=0.55, a=2.0, ro=8.0, vo=230.0)
    pot_nonstandardrovo = HernquistPotential(amp=0.55, a=2.0, ro=9.0, vo=230.0)
    pot_nophys = HernquistPotential(amp=0.55)
    pot_default_noro = HernquistPotential(amp=0.55, vo=220.0)
    pot_default_novo = HernquistPotential(amp=0.55, ro=8.0)
    pot_nonstandardro_novo = HernquistPotential(amp=0.55, ro=9.0)
    pot_nonstandardvo_noro = HernquistPotential(amp=0.55, vo=230.0)
    # Test expected behavior for single potentials
    assert physical_compatible(pot_default_phys, pot_default_phys), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_default_phys, pot_nonstandardro), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_nonstandardro, pot_default_phys), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_default_phys, pot_nonstandardvo), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_default_phys, pot_nonstandardrovo), (
        "pot_default_phys does not behave as expected"
    )
    assert physical_compatible(pot_default_phys, pot_nophys), (
        "pot_default_phys does not behave as expected"
    )
    assert physical_compatible(pot_default_phys, pot_default_noro), (
        "pot_default_phys does not behave as expected"
    )
    assert physical_compatible(pot_default_phys, pot_default_novo), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_default_phys, pot_nonstandardro_novo), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(pot_default_phys, pot_nonstandardvo_noro), (
        "pot_default_phys does not behave as expected"
    )
    # Test expected behavior for single,list pairs
    assert physical_compatible(
        pot_default_phys, [pot_default_phys, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardro, pot_nonstandardro]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardro, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_nonstandardro, [pot_default_phys, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardvo, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardrovo, pot_nonstandardro]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(pot_default_phys, [pot_nophys, pot_nophys]), (
        "pot_default_phys does not behave as expected"
    )
    assert physical_compatible(
        pot_default_phys, [pot_default_noro, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(pot_default_phys, [pot_default_novo, pot_nophys]), (
        "pot_default_phys does not behave as expected"
    )
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardro_novo, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, [pot_nonstandardvo_noro, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    # Test expected behavior for list,list pairs
    assert physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_default_phys, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardro, pot_nonstandardro]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardro, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_nonstandardro, pot_default_phys], [pot_default_phys, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardvo, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardrovo, pot_nonstandardro]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nophys, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_default_noro, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_default_novo, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardro_novo, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        [pot_default_phys, pot_default_phys], [pot_nonstandardvo_noro, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    return None


# ADD OTHER COMBINATIONS, e.g., potential and orbit
def test_physical_compatible_combos():
    # Test that physical_compatible acts as expected for combinations of
    # different types of objects
    from galpy.actionAngle import actionAngleSpherical
    from galpy.df import quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import HernquistPotential
    from galpy.util.conversion import physical_compatible

    # Set up different objects for possible cases
    # Potentials
    pot_default_phys = HernquistPotential(amp=0.55, a=2.0, ro=8.0, vo=220.0)
    pot_nonstandardro = HernquistPotential(amp=0.55, a=2.0, ro=9.0, vo=220.0)
    pot_nonstandardvo = HernquistPotential(amp=0.55, a=2.0, ro=8.0, vo=230.0)
    pot_nonstandardrovo = HernquistPotential(amp=0.55, a=2.0, ro=9.0, vo=230.0)
    pot_nophys = HernquistPotential(amp=0.55)
    pot_default_noro = HernquistPotential(amp=0.55, vo=220.0)
    pot_default_novo = HernquistPotential(amp=0.55, ro=8.0)
    pot_nonstandardro_novo = HernquistPotential(amp=0.55, ro=9.0)
    pot_nonstandardvo_noro = HernquistPotential(amp=0.55, vo=230.0)
    pot_nonstandardvo_noro = HernquistPotential(amp=0.55, vo=230.0)
    # Orbits
    orb_default_phys = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], ro=8.0, vo=220.0)
    orb_nonstandardro = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], ro=9.0, vo=220.0)
    orb_nonstandardvo = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], ro=8.0, vo=230.0)
    orb_nonstandardrovo = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], ro=9.0, vo=230.0)
    orb_nophys = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9])
    orb_default_noro = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], vo=220.0)
    orb_nonstandardvo_noro = Orbit([1.0, 0.1, 1.1, 0.1, 0.3, -0.9], vo=230.0)
    # aAs
    aA_default_phys = actionAngleSpherical(pot=pot_default_phys, ro=8.0, vo=220.0)
    aA_nonstandardro = actionAngleSpherical(pot=pot_nonstandardro, ro=9.0, vo=220.0)
    aA_nonstandardvo = actionAngleSpherical(pot=pot_nonstandardvo, ro=8.0, vo=230.0)
    aA_nonstandardrovo = actionAngleSpherical(pot=pot_nonstandardrovo, ro=9.0, vo=230.0)
    aA_nophys = actionAngleSpherical(pot=pot_nophys)
    aA_default_novo = actionAngleSpherical(pot=pot_default_novo, ro=8.0)
    aA_nonstandardvo_noro = actionAngleSpherical(pot=pot_nonstandardvo_noro, vo=230.0)
    # DFs
    qdf_default_phys = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot_default_phys,
        aA=aA_default_phys,
        ro=8.0,
        vo=220.0,
    )
    qdf_nonstandardro = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot_nonstandardro,
        aA=aA_nonstandardro,
        ro=9.0,
        vo=220.0,
    )
    qdf_nonstandardvo = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot_nonstandardvo,
        aA=aA_nonstandardvo,
        ro=8.0,
        vo=230.0,
    )
    qdf_nonstandardrovo = quasiisothermaldf(
        1.0 / 3.0,
        0.2,
        0.1,
        1.0,
        1.0,
        pot=pot_nonstandardrovo,
        aA=aA_nonstandardrovo,
        ro=9.0,
        vo=230.0,
    )
    qdf_nophys = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=pot_nophys, aA=aA_nophys
    )
    # Now do some tests!
    assert physical_compatible(pot_default_phys, orb_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_default_phys, aA_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_default_phys, qdf_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nonstandardro, orb_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nonstandardro, aA_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nonstandardro, qdf_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, orb_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, aA_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, qdf_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, orb_nonstandardvo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, aA_nonstandardvo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, qdf_nonstandardvo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, orb_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, aA_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(pot_default_phys, qdf_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible([pot_nophys, pot_nophys], orb_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, aA_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, qdf_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, orb_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, aA_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, qdf_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, orb_nophys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, orb_nophys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nophys, qdf_nophys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_default_noro, qdf_nonstandardro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_nonstandardro_novo, orb_default_noro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(aA_nonstandardvo_noro, orb_nonstandardvo_noro), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert not physical_compatible(aA_default_novo, qdf_nonstandardrovo), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    # Also test agained None!
    assert physical_compatible(None, pot_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(None, orb_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(None, aA_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(None, qdf_default_phys), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(pot_default_phys, None), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(orb_default_phys, None), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(aA_default_phys, None), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    assert physical_compatible(qdf_default_phys, None), (
        "pot_default_phys does not behave as expected for combinations of different objects"
    )
    return None


# Tests for parse functions
def test_parse_length_scalar():
    """Test parse_length with scalar float input."""
    from galpy.util.conversion import parse_length

    # Scalar float should be returned unchanged
    result = parse_length(5.0, ro=8.0, vo=220.0)
    assert result == 5.0, f"parse_length scalar failed: expected 5.0, got {result}"
    return None


def test_parse_length_array():
    """Test parse_length with numpy array input."""
    from galpy.util.conversion import parse_length

    arr = numpy.array([1.0, 2.0, 3.0])
    result = parse_length(arr, ro=8.0, vo=220.0)
    assert numpy.allclose(result, arr), (
        f"parse_length array failed: expected {arr}, got {result}"
    )
    return None


def test_parse_length_quantity():
    """Test parse_length with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_length

    # 16 kpc with ro=8 kpc should give 2.0 in internal units
    result = parse_length(16.0 * units.kpc, ro=8.0, vo=220.0)
    assert numpy.fabs(result - 2.0) < 1e-10, (
        f"parse_length Quantity failed: expected 2.0, got {result}"
    )
    # 8000 pc with ro=8 kpc should give 1.0 in internal units
    result2 = parse_length(8000.0 * units.pc, ro=8.0, vo=220.0)
    assert numpy.fabs(result2 - 1.0) < 1e-10, (
        f"parse_length Quantity pc failed: expected 1.0, got {result2}"
    )
    return None


def test_parse_length_kpc_quantity():
    """Test parse_length_kpc with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_length_kpc

    # 5000 pc should give 5.0 kpc
    result = parse_length_kpc(5000.0 * units.pc)
    assert numpy.fabs(result - 5.0) < 1e-10, (
        f"parse_length_kpc Quantity failed: expected 5.0, got {result}"
    )
    return None


def test_parse_velocity_quantity():
    """Test parse_velocity with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_velocity

    # 220 km/s with vo=220 km/s should give 1.0 in internal units
    result = parse_velocity(220.0 * units.km / units.s, ro=8.0, vo=220.0)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_velocity Quantity failed: expected 1.0, got {result}"
    )
    # 440 km/s with vo=220 km/s should give 2.0 in internal units
    result2 = parse_velocity(440.0 * units.km / units.s, ro=8.0, vo=220.0)
    assert numpy.fabs(result2 - 2.0) < 1e-10, (
        f"parse_velocity Quantity failed: expected 2.0, got {result2}"
    )
    return None


def test_parse_velocity_kms_quantity():
    """Test parse_velocity_kms with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_velocity_kms

    # 220000 m/s should give 220.0 km/s
    result = parse_velocity_kms(220000.0 * units.m / units.s)
    assert numpy.fabs(result - 220.0) < 1e-10, (
        f"parse_velocity_kms Quantity failed: expected 220.0, got {result}"
    )
    return None


def test_parse_angle_scalar():
    """Test parse_angle with scalar float input (already in radians)."""
    from galpy.util.conversion import parse_angle

    result = parse_angle(numpy.pi / 2)
    assert numpy.fabs(result - numpy.pi / 2) < 1e-10, (
        f"parse_angle scalar failed: expected {numpy.pi / 2}, got {result}"
    )
    return None


def test_parse_angle_quantity():
    """Test parse_angle with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_angle

    # 90 degrees should give pi/2 radians
    result = parse_angle(90.0 * units.deg)
    assert numpy.fabs(result - numpy.pi / 2) < 1e-10, (
        f"parse_angle Quantity failed: expected {numpy.pi / 2}, got {result}"
    )
    # 180 degrees should give pi radians
    result2 = parse_angle(180.0 * units.deg)
    assert numpy.fabs(result2 - numpy.pi) < 1e-10, (
        f"parse_angle Quantity failed: expected {numpy.pi}, got {result2}"
    )
    return None


def test_parse_time_quantity():
    """Test parse_time with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_time, time_in_Gyr

    ro, vo = 8.0, 220.0
    # 1 Gyr should convert to internal units
    result = parse_time(1.0 * units.Gyr, ro=ro, vo=vo)
    expected = 1.0 / time_in_Gyr(vo, ro)
    assert numpy.fabs(result - expected) < 1e-10, (
        f"parse_time Quantity failed: expected {expected}, got {result}"
    )
    return None


def test_parse_energy_quantity():
    """Test parse_energy with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_energy

    vo = 220.0
    # vo^2 km^2/s^2 should give 1.0 in internal units
    result = parse_energy(vo**2 * units.km**2 / units.s**2, ro=8.0, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_energy Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_angmom_quantity():
    """Test parse_angmom with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_angmom

    ro, vo = 8.0, 220.0
    # ro*vo kpc*km/s should give 1.0 in internal units
    result = parse_angmom(ro * vo * units.kpc * units.km / units.s, ro=ro, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_angmom Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_frequency_quantity():
    """Test parse_frequency with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import freq_in_kmskpc, parse_frequency

    ro, vo = 8.0, 220.0
    freq_internal = freq_in_kmskpc(vo, ro)  # internal units to km/s/kpc
    # freq_internal km/s/kpc should give 1.0 in internal units
    result = parse_frequency(
        freq_internal * units.km / units.s / units.kpc, ro=ro, vo=vo
    )
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_frequency Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_mass_quantity():
    """Test parse_mass with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import mass_in_msol, parse_mass

    ro, vo = 8.0, 220.0
    mass_internal = mass_in_msol(vo, ro)  # internal units to Msun
    # mass_internal Msun should give 1.0 in internal units
    result = parse_mass(mass_internal * units.Msun, ro=ro, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_mass Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_dens_quantity():
    """Test parse_dens with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import dens_in_msolpc3, parse_dens

    ro, vo = 8.0, 220.0
    dens_internal = dens_in_msolpc3(vo, ro)  # internal units to Msun/pc^3
    # dens_internal Msun/pc^3 should give 1.0 in internal units
    result = parse_dens(dens_internal * units.Msun / units.pc**3, ro=ro, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_dens Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_surfdens_quantity():
    """Test parse_surfdens with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_surfdens, surfdens_in_msolpc2

    ro, vo = 8.0, 220.0
    surfdens_internal = surfdens_in_msolpc2(vo, ro)  # internal units to Msun/pc^2
    # surfdens_internal Msun/pc^2 should give 1.0 in internal units
    result = parse_surfdens(surfdens_internal * units.Msun / units.pc**2, ro=ro, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_surfdens Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_force_quantity():
    """Test parse_force with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import force_in_pcMyr2, parse_force

    ro, vo = 8.0, 220.0
    force_internal = force_in_pcMyr2(vo, ro)  # internal units to pc/Myr^2
    # force_internal pc/Myr^2 should give 1.0 in internal units
    result = parse_force(force_internal * units.pc / units.Myr**2, ro=ro, vo=vo)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_force Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_numdens_quantity():
    """Test parse_numdens with astropy Quantity input."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_numdens

    ro = 8.0
    # 1/ro^3 kpc^-3 should give 1.0 in internal units
    result = parse_numdens((1.0 / ro**3) / units.kpc**3, ro=ro, vo=220.0)
    assert numpy.fabs(result - 1.0) < 1e-10, (
        f"parse_numdens Quantity failed: expected 1.0, got {result}"
    )
    return None


def test_parse_invalid_input_string():
    """Test that parse functions raise RuntimeError for invalid input."""
    from galpy.util.conversion import parse_length

    try:
        parse_length("invalid", ro=8.0, vo=220.0)
        raise AssertionError(
            "parse_length should have raised RuntimeError for string input"
        )
    except RuntimeError as e:
        assert "not understood" in str(e), f"Unexpected error message: {e}"
    return None


def test_parse_invalid_ro():
    """Test that parse functions raise RuntimeError for invalid ro."""
    from galpy.util.conversion import parse_length

    try:
        parse_length(5.0, ro="invalid", vo=220.0)
        raise AssertionError(
            "parse_length should have raised RuntimeError for invalid ro"
        )
    except RuntimeError as e:
        assert "ro=" in str(e) and "not understood" in str(e), (
            f"Unexpected error message: {e}"
        )
    return None


def test_parse_invalid_vo():
    """Test that parse functions raise RuntimeError for invalid vo."""
    from galpy.util.conversion import parse_velocity

    try:
        parse_velocity(5.0, ro=8.0, vo="invalid")
        raise AssertionError(
            "parse_velocity should have raised RuntimeError for invalid vo"
        )
    except RuntimeError as e:
        assert "vo=" in str(e) and "not understood" in str(e), (
            f"Unexpected error message: {e}"
        )
    return None


def test_parse_none_input():
    """Test that parse functions handle None input."""
    from galpy.util.conversion import parse_length

    result = parse_length(None, ro=8.0, vo=220.0)
    assert result is None, f"parse_length None failed: expected None, got {result}"
    return None


def test_parse_ro_vo_quantity():
    """Test that parse functions correctly convert ro/vo Quantities."""
    from galpy.util._optional_deps import _APY_LOADED

    if not _APY_LOADED:
        return None
    from astropy import units

    from galpy.util.conversion import parse_length

    # 16 kpc with ro=8000 pc should give 2.0 in internal units
    result = parse_length(16.0 * units.kpc, ro=8000.0 * units.pc, vo=220.0)
    assert numpy.fabs(result - 2.0) < 1e-10, (
        f"parse_length with ro Quantity failed: expected 2.0, got {result}"
    )
    return None


def test_parse_empty_array():
    """Test that parse functions handle empty arrays."""
    from galpy.util.conversion import parse_length

    arr = numpy.array([])
    result = parse_length(arr, ro=8.0, vo=220.0)
    assert len(result) == 0, (
        f"parse_length empty array failed: expected empty, got {result}"
    )
    return None
