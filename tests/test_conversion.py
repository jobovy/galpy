from __future__ import print_function, division
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
    assert numpy.fabs(2.*bovy_conversion.velocity_in_kpcGyr(vofid,rofid)/bovy_conversion.velocity_in_kpcGyr(2.*vofid,rofid)-1.) < 10.**-10., 'velocity_in_kpcGyr did not work as expected'
    assert numpy.fabs(bovy_conversion.velocity_in_kpcGyr(vofid,rofid)/bovy_conversion.velocity_in_kpcGyr(vofid,2*rofid)-1.) < 10.**-10., 'velocity_in_kpcGyr did not work as expected'
    return None
    
def test_get_physical():
    #Test that the get_physical function returns the right scaling parameters
    from galpy.util.bovy_conversion import get_physical
    # Potential and variations thereof
    from galpy.potential import MWPotential2014, DehnenBarPotential
    dp= DehnenBarPotential
    assert numpy.fabs(get_physical(MWPotential2014[0]).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014[0]).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    ro,vo= 9., 230.
    dp= DehnenBarPotential(ro=ro,vo=vo)
    assert numpy.fabs(get_physical(dp).get('ro')-ro) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(dp).get('vo')-vo) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014+dp).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014+dp).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014+dp).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    assert numpy.fabs(get_physical(MWPotential2014+dp).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a Potential'
    # Orbits
    from galpy.orbit import Orbit
    ro,vo= 10., 210.
    o= Orbit(ro=ro,vo=vo)
    assert numpy.fabs(get_physical(o).get('ro')-ro) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an Orbit'
    assert numpy.fabs(get_physical(o).get('vo')-vo) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an Orbit'
    # even though one shouldn't do this, let's test a list
    assert numpy.fabs(get_physical([o,o]).get('ro')-ro) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an Orbit'
    assert numpy.fabs(get_physical([o,o]).get('vo')-vo) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an Orbit'
    # actionAngle
    from galpy.actionAngle import actionAngleStaeckel
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.45)
    assert numpy.fabs(get_physical(aAS).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an actionAngle instance'
    assert numpy.fabs(get_physical(aAS).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an actionAngle instance'
    # This doesn't make much sense, but let's test...
    ro,vo= 19., 130.
    dp= DehnenBarPotential(ro=ro,vo=vo)
    aAS= actionAngleStaeckel(pot=dp,delta=0.45,ro=ro,vo=vo)
    assert numpy.fabs(get_physical(aAS).get('ro')-ro) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an actionAngle instance'
    assert numpy.fabs(get_physical(aAS).get('vo')-vo) < 1e-10, 'get_physical does not return the correct unit conversion parameter for an actionAngle instance'
    # DF
    from galpy.df import quasiisothermaldf
    aAS= actionAngleStaeckel(pot=MWPotential2014,delta=0.45)
    qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,aA=aAS,pot=MWPotential2014)
    assert numpy.fabs(get_physical(qdf).get('ro')-8.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a DF instance'
    assert numpy.fabs(get_physical(qdf).get('vo')-220.) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a DF instance'
    # non-standard ro,vo
    from galpy.potential import MiyamotoNagaiPotential
    ro,vo= 4., 330.
    mp= MiyamotoNagaiPotential(a=0.5,b=0.1,ro=ro,vo=vo)
    aAS= actionAngleStaeckel(pot=mp,delta=0.45,ro=ro,vo=vo)
    qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,aA=aAS,pot=mp,ro=ro,vo=vo)
    assert numpy.fabs(get_physical(qdf).get('ro')-ro) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a DF instance'
    assert numpy.fabs(get_physical(qdf).get('vo')-vo) < 1e-10, 'get_physical does not return the correct unit conversion parameter for a DF instance'
    return None

    
