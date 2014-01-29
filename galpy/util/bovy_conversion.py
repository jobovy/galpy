###############################################################################
#
# bovy_conversion: utilities to convert from galpy 'natural units' to physical
#                  units
#
###############################################################################
import math as m
_G= 4.302*10.**-3. #pc / Msolar (km/s)^2
_kmsInPcMyr= 1.0227121655399913
_MyrIn1013Sec= 3.65242198*0.24*3.6 #use tropical year, like for pms
_TWOPI= 2.*m.pi
def dens_in_criticaldens(vo,ro,H=70.):
    """
    NAME:

       dens_in_criticaldens

    PURPOSE:

       convert density to units of the critical density

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

       H= (default: 70) Hubble constant in km/s/Mpc
       
    OUTPUT:

       conversion from units where vo=1. at ro=1. to units of the critical density

    HISTORY:

       2014-01-28 - Written - Bovy (IAS)

    """
    return vo**2./ro**2.*10.**6./H**2.*8.*m.pi/3.

def dens_in_meanmatterdens(vo,ro,H=70.,Om=0.3):
    """
    NAME:

       dens_in_meanmatterdens

    PURPOSE:

       convert density to units of the mean matter density

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

       H= (default: 70) Hubble constant in km/s/Mpc

       Om= (default: 0.3) Omega matter
       
    OUTPUT:

       conversion from units where vo=1. at ro=1. to units of the mean matter density

    HISTORY:

       2014-01-28 - Written - Bovy (IAS)

    """
    return dens_in_criticaldens(vo,ro,H=H)/Om

def dens_in_msolpc3(vo,ro):
    """
    NAME:

       dens_in_msolpc3

    PURPOSE:

       convert density to Msolar / pc^3

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1. to Msolar/pc^3

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2./ro**2./_G*10.**-6.

def force_in_2piGmsolpc2(vo,ro):
    """
    NAME:

       force_in_2piGmsolpc2

    PURPOSE:

       convert a force or acceleration to 2piG x Msolar / pc^2

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2./ro/_G*10.**-3./_TWOPI

def force_in_pcMyr2(vo,ro):
    """
    NAME:

       force_in_pcMyr2

    PURPOSE:

       convert a force or acceleration to pc/Myr^2

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2./ro*_kmsInPcMyr**2.*10.**-3.

def force_in_kmsMyr(vo,ro):
    """
    NAME:

       force_in_kmsMyr

    PURPOSE:

       convert a force or acceleration to km/s/Myr

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2./ro*_kmsInPcMyr*10.**-3.

def force_in_10m13kms2(vo,ro):
    """
    NAME:

       force_in_10m13kms2

    PURPOSE:

       convert a force or acceleration to 10^(-13) km/s^2

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2014-01-22 - Written - Bovy (IAS)

    """
    return vo**2./ro*_kmsInPcMyr*10.**-3./_MyrIn1013Sec

def freq_in_Gyr(vo,ro):
    """
    NAME:

       freq_in_Gyr

    PURPOSE:

       convert a frequency to 1/Gyr

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo/ro*_kmsInPcMyr

def freq_in_kmskpc(vo,ro):
    """
    NAME:

       freq_in_kmskpc

    PURPOSE:

       convert a frequency to km/s/kpc

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo/ro

def surfdens_in_msolpc2(vo,ro):
    """
    NAME:

       surfdens_in_msolpc2

    PURPOSE:

       convert a surface density to Msolar / pc^2

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2./ro/_G*10.**-3.

def mass_in_msol(vo,ro):
    """
    NAME:

       mass_in_msol

    PURPOSE:

       convert a mass to Msolar

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.*ro/_G*10.**3.

def mass_in_1010msol(vo,ro):
    """
    NAME:

       mass_in_1010msol

    PURPOSE:

       convert a mass to 10^10 x Msolar

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return vo**2.*ro/_G*10.**-7.

def time_in_Gyr(vo,ro):
    """
    NAME:

       time_in_Gyr

    PURPOSE:

       convert a time to Gyr

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2013-09-01 - Written - Bovy (IAS)

    """
    return ro/vo/_kmsInPcMyr

