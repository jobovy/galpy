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
    assert (
        numpy.fabs(get_physical(MWPotential2014[0]).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014[0]).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    ro, vo = 9.0, 230.0
    dp = DehnenBarPotential(ro=ro, vo=vo)
    assert (
        numpy.fabs(get_physical(dp).get("ro") - ro) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(dp).get("vo") - vo) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014 + dp).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014 + dp).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014 + dp).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    assert (
        numpy.fabs(get_physical(MWPotential2014 + dp).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a Potential"
    # Orbits
    from galpy.orbit import Orbit

    ro, vo = 10.0, 210.0
    o = Orbit(ro=ro, vo=vo)
    assert (
        numpy.fabs(get_physical(o).get("ro") - ro) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an Orbit"
    assert (
        numpy.fabs(get_physical(o).get("vo") - vo) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an Orbit"
    # even though one shouldn't do this, let's test a list
    assert (
        numpy.fabs(get_physical([o, o]).get("ro") - ro) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an Orbit"
    assert (
        numpy.fabs(get_physical([o, o]).get("vo") - vo) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an Orbit"
    # actionAngle
    from galpy.actionAngle import actionAngleStaeckel

    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    assert (
        numpy.fabs(get_physical(aAS).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    assert (
        numpy.fabs(get_physical(aAS).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    # This doesn't make much sense, but let's test...
    ro, vo = 19.0, 130.0
    dp = DehnenBarPotential(ro=ro, vo=vo)
    aAS = actionAngleStaeckel(pot=dp, delta=0.45, ro=ro, vo=vo)
    assert (
        numpy.fabs(get_physical(aAS).get("ro") - ro) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    assert (
        numpy.fabs(get_physical(aAS).get("vo") - vo) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for an actionAngle instance"
    # DF
    from galpy.df import quasiisothermaldf

    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45)
    qdf = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, aA=aAS, pot=MWPotential2014)
    assert (
        numpy.fabs(get_physical(qdf).get("ro") - 8.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a DF instance"
    assert (
        numpy.fabs(get_physical(qdf).get("vo") - 220.0) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a DF instance"
    # non-standard ro,vo
    from galpy.potential import MiyamotoNagaiPotential

    ro, vo = 4.0, 330.0
    mp = MiyamotoNagaiPotential(a=0.5, b=0.1, ro=ro, vo=vo)
    aAS = actionAngleStaeckel(pot=mp, delta=0.45, ro=ro, vo=vo)
    qdf = quasiisothermaldf(1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, aA=aAS, pot=mp, ro=ro, vo=vo)
    assert (
        numpy.fabs(get_physical(qdf).get("ro") - ro) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a DF instance"
    assert (
        numpy.fabs(get_physical(qdf).get("vo") - vo) < 1e-10
    ), "get_physical does not return the correct unit conversion parameter for a DF instance"
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
    assert physical_compatible(
        pot_default_phys, pot_default_phys
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, pot_nonstandardro
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_nonstandardro, pot_default_phys
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, pot_nonstandardvo
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, pot_nonstandardrovo
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        pot_default_phys, pot_nophys
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        pot_default_phys, pot_default_noro
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        pot_default_phys, pot_default_novo
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, pot_nonstandardro_novo
    ), "pot_default_phys does not behave as expected"
    assert not physical_compatible(
        pot_default_phys, pot_nonstandardvo_noro
    ), "pot_default_phys does not behave as expected"
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
    assert physical_compatible(
        pot_default_phys, [pot_nophys, pot_nophys]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        pot_default_phys, [pot_default_noro, pot_default_phys]
    ), "pot_default_phys does not behave as expected"
    assert physical_compatible(
        pot_default_phys, [pot_default_novo, pot_nophys]
    ), "pot_default_phys does not behave as expected"
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
    assert physical_compatible(
        pot_default_phys, orb_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_default_phys, aA_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_default_phys, qdf_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nonstandardro, orb_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nonstandardro, aA_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nonstandardro, qdf_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, orb_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, aA_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, qdf_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, orb_nonstandardvo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, aA_nonstandardvo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, qdf_nonstandardvo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, orb_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, aA_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        pot_default_phys, qdf_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        [pot_nophys, pot_nophys], orb_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, aA_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, qdf_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, orb_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, aA_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, qdf_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, orb_nophys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, orb_nophys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nophys, qdf_nophys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_default_noro, qdf_nonstandardro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_nonstandardro_novo, orb_default_noro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        aA_nonstandardvo_noro, orb_nonstandardvo_noro
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert not physical_compatible(
        aA_default_novo, qdf_nonstandardrovo
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    # Also test agained None!
    assert physical_compatible(
        None, pot_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        None, orb_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        None, aA_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        None, qdf_default_phys
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        pot_default_phys, None
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        orb_default_phys, None
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        aA_default_phys, None
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    assert physical_compatible(
        qdf_default_phys, None
    ), "pot_default_phys does not behave as expected for combinations of different objects"
    return None
