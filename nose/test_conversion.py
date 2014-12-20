import numpy
from galpy.util import bovy_conversion

def test_dens_in_criticaldens():
    #Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.dens_in_criticaldens(vofid,rofid)/bovy_conversion.dens_in_criticaldens(2.*vofid,rofid)-1.) < 10.**-10., 'dens_in_criticaldens did not work as expected'
    assert numpy.fabs(.25*bovy_conversion.dens_in_criticaldens(vofid,rofid)/bovy_conversion.dens_in_criticaldens(vofid,2*rofid)-1.) < 10.**-10., 'dens_in_critical did not work as expected'
    return None
    
def test_dens_in_meanmatterdens():
    #Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.dens_in_meanmatterdens(vofid,rofid)/bovy_conversion.dens_in_meanmatterdens(2.*vofid,rofid)-1.) < 10.**-10., 'dens_in_meanmatterdens did not work as expected'
    assert numpy.fabs(.25*bovy_conversion.dens_in_meanmatterdens(vofid,rofid)/bovy_conversion.dens_in_meanmatterdens(vofid,2*rofid)-1.) < 10.**-10., 'dens_in_meanmatter did not work as expected'
    return None
    
def test_dens_in_gevcc():
    #Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.dens_in_gevcc(vofid,rofid)/bovy_conversion.dens_in_gevcc(2.*vofid,rofid)-1.) < 10.**-10., 'dens_in_gevcc did not work as expected'
    assert numpy.fabs(.25*bovy_conversion.dens_in_gevcc(vofid,rofid)/bovy_conversion.dens_in_gevcc(vofid,2*rofid)-1.) < 10.**-10., 'dens_in_gevcc did not work as expected'
    return None

def test_dens_in_msolpc3():
    #Test the scaling, as a 2nd derivative of the potential / G, should scale as velocity^2/position^2
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.dens_in_msolpc3(vofid,rofid)/bovy_conversion.dens_in_msolpc3(2.*vofid,rofid)-1.) < 10.**-10., 'dens_in_msolpc3 did not work as expected'
    assert numpy.fabs(.25*bovy_conversion.dens_in_msolpc3(vofid,rofid)/bovy_conversion.dens_in_msolpc3(vofid,2*rofid)-1.) < 10.**-10., 'dens_in_msolpc3 did not work as expected'
    return None

def test_force_in_2piGmsolpc2():
    #Test the scaling, as a 1st derivative of the potential / G, should scale as velocity^2/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.force_in_2piGmsolpc2(vofid,rofid)/bovy_conversion.force_in_2piGmsolpc2(2.*vofid,rofid)-1.) < 10.**-10., 'force_in_2piGmsolpc2 did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.force_in_2piGmsolpc2(vofid,rofid)/bovy_conversion.force_in_2piGmsolpc2(vofid,2*rofid)-1.) < 10.**-10., 'force_in_2piGmsolpc2 did not work as expected'
    return None

def test_force_in_pcMyr2():
    #Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.force_in_pcMyr2(vofid,rofid)/bovy_conversion.force_in_pcMyr2(2.*vofid,rofid)-1.) < 10.**-10., 'force_in_pcMyr2 did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.force_in_pcMyr2(vofid,rofid)/bovy_conversion.force_in_pcMyr2(vofid,2*rofid)-1.) < 10.**-10., 'force_in_pcMyr2 did not work as expected'
    return None

def test_force_in_kmsMyr():
    #Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.force_in_kmsMyr(vofid,rofid)/bovy_conversion.force_in_kmsMyr(2.*vofid,rofid)-1.) < 10.**-10., 'force_in_kmsMyr did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.force_in_kmsMyr(vofid,rofid)/bovy_conversion.force_in_kmsMyr(vofid,2*rofid)-1.) < 10.**-10., 'force_in_kmsMyr did not work as expected'
    return None

def test_force_in_10m13kms2():
    #Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.force_in_10m13kms2(vofid,rofid)/bovy_conversion.force_in_10m13kms2(2.*vofid,rofid)-1.) < 10.**-10., 'force_in_10m13kms2 did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.force_in_10m13kms2(vofid,rofid)/bovy_conversion.force_in_10m13kms2(vofid,2*rofid)-1.) < 10.**-10., 'force_in_10m13kms2 did not work as expected'
    return None

def test_freq_in_Gyr():
    #Test the scaling, as 1/time, should scale as velocity/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(2.*bovy_conversion.freq_in_Gyr(vofid,rofid)/bovy_conversion.freq_in_Gyr(2.*vofid,rofid)-1.) < 10.**-10., 'freq_in_Gyr did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.freq_in_Gyr(vofid,rofid)/bovy_conversion.freq_in_Gyr(vofid,2*rofid)-1.) < 10.**-10., 'freq_in_Gyr did not work as expected'
    return None

def test_freq_in_kmskpc():
    #Test the scaling, as 1/time, should scale as velocity/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(2.*bovy_conversion.freq_in_kmskpc(vofid,rofid)/bovy_conversion.freq_in_kmskpc(2.*vofid,rofid)-1.) < 10.**-10., 'freq_in_kmskpc did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.freq_in_kmskpc(vofid,rofid)/bovy_conversion.freq_in_kmskpc(vofid,2*rofid)-1.) < 10.**-10., 'freq_in_kmskpc did not work as expected'
    return None

def test_surfdens_in_msolpc2():
    #Test the scaling, as a 1st derivative of the potential, should scale as velocity^2/position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.surfdens_in_msolpc2(vofid,rofid)/bovy_conversion.surfdens_in_msolpc2(2.*vofid,rofid)-1.) < 10.**-10., 'surfdens_in_msolpc2 did not work as expected'
    assert numpy.fabs(.5*bovy_conversion.surfdens_in_msolpc2(vofid,rofid)/bovy_conversion.surfdens_in_msolpc2(vofid,2*rofid)-1.) < 10.**-10., 'surfdens_in_msolpc2 did not work as expected'
    return None

def test_mass_in_msol():
    #Test the scaling, should be velocity^2 x position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.mass_in_msol(vofid,rofid)/bovy_conversion.mass_in_msol(2.*vofid,rofid)-1.) < 10.**-10., 'mass_in_msol did not work as expected'
    assert numpy.fabs(2.*bovy_conversion.mass_in_msol(vofid,rofid)/bovy_conversion.mass_in_msol(vofid,2*rofid)-1.) < 10.**-10., 'mass_in_msol did not work as expected'
    return None

def test_mass_in_1010msol():
    #Test the scaling, should be velocity^2 x position
    vofid, rofid= 200., 8.
    assert numpy.fabs(4.*bovy_conversion.mass_in_1010msol(vofid,rofid)/bovy_conversion.mass_in_1010msol(2.*vofid,rofid)-1.) < 10.**-10., 'mass_in_1010msol did not work as expected'
    assert numpy.fabs(2.*bovy_conversion.mass_in_1010msol(vofid,rofid)/bovy_conversion.mass_in_1010msol(vofid,2*rofid)-1.) < 10.**-10., 'mass_in_1010msol did not work as expected'
    return None

def test_time_in_Gyr():
    #Test the scaling, should scale as position/velocity
    vofid, rofid= 200., 8.
    assert numpy.fabs(0.5*bovy_conversion.time_in_Gyr(vofid,rofid)/bovy_conversion.time_in_Gyr(2.*vofid,rofid)-1.) < 10.**-10., 'time_in_Gyr did not work as expected'
    assert numpy.fabs(2.*bovy_conversion.time_in_Gyr(vofid,rofid)/bovy_conversion.time_in_Gyr(vofid,2*rofid)-1.) < 10.**-10., 'time_in_Gyr did not work as expected'
    return None
    
def test_velocity_in_kpcGyr():
    #Test the scaling, should scale as velocity
    vofid, rofid= 200., 8.
    assert numpy.fabs(0.5*bovy_conversion.velocity_in_kpcGyr(vofid,rofid)/bovy_conversion.velocity_in_kpcGyr(2.*vofid,rofid)-1.) < 10.**-10., 'velocity_in_kpcGyr did not work as expected'
    assert numpy.fabs(bovy_conversion.velocity_in_kpcGyr(vofid,rofid)/bovy_conversion.velocity_in_kpcGyr(vofid,2*rofid)-1.) < 10.**-10., 'velocity_in_kpcGyr did not work as expected'
    return None
    
