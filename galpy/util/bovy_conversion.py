###############################################################################
#
# bovy_conversion: utilities to convert from galpy 'natural units' to physical
#                  units
#
###############################################################################
from functools import wraps
import warnings
import copy
import math as m
from galpy.util import galpyWarning
_G= 4.302*10.**-3. #pc / Msolar (km/s)^2
_kmsInPcMyr= 1.0227121655399913
_MyrIn1013Sec= 3.65242198*0.24*3.6 #use tropical year, like for pms
_PCIN10p18CM= 3.08567758 #10^18 cm
_CIN10p5KMS= 2.99792458 #10^5 km/s
_MSOLAR10p30KG= 1.9891 #10^30 kg
_EVIN10m19J= 1.60217657 #10^-19 J
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

def dens_in_gevcc(vo,ro):
    """
    NAME:

       dens_in_gevcc

    PURPOSE:

       convert density to GeV / cm^3

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1. to GeV/cm^3

    HISTORY:

       2014-06-16 - Written - Bovy (IAS)

    """
    return vo**2./ro**2./_G*_MSOLAR10p30KG*_CIN10p5KMS**2./_EVIN10m19J/ _PCIN10p18CM**3.*10.**-4.

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

#Decorator to apply these transformations
def print_physical_warning():
    warnings.warn("The behavior of Orbit member functions has changed in versions > 0.1 to return positions in kpc, velocities in km/s, energies and the Jacobi integral in (km/s)^2, the angular momentum o.L() and actions in km/s kpc, frequencies in 1/Gyr, and times and periods in Gyr if a distance and velocity scale was specified upon Orbit initialization with ro=...,vo=...; you can turn this off by specifying use_physical=False when calling the function (e.g., o=Orbit(...); o.R(use_physical=False)",
                  galpyWarning)   
_roNecessary= {'time': True,
               'position': True,
               'velocity': False,
               'energy': False,
               'density': True,
               'force': True,
               'surfacedensity': True,
               'mass': True,
               'action': True,
               'frequency':True}
_voNecessary= copy.copy(_roNecessary)
_voNecessary['position']= False
_voNecessary['velocity']= True
_voNecessary['energy']= True
def physical_conversion(quantity,pop=False):
    """Decorator to convert to physical coordinates: 
    quantity = [position,velocity,time]"""
    def wrapper(method):
        @wraps(method)
        def wrapped(*args,**kwargs):
            if kwargs.has_key('use_physical'):
                use_physical= kwargs['use_physical']
            else:
                use_physical= True
            if kwargs.has_key('ro'):
                ro= kwargs['ro']
            elif hasattr(args[0],'_roSet') and args[0]._roSet:
                ro= args[0]._ro
            else:
                ro= None
            if kwargs.has_key('vo'):
                vo= kwargs['vo']
            elif hasattr(args[0],'_voSet') and args[0]._voSet:
                vo= args[0]._vo
            else:
                vo= None
            #Remove ro and vo kwargs if necessary
            if pop and kwargs.has_key('ro'): kwargs.pop('ro')
            if pop and kwargs.has_key('vo'): kwargs.pop('vo')
            if use_physical and \
                    not (_voNecessary[quantity.lower()] and vo is None) and \
                    not (_roNecessary[quantity.lower()] and ro is None):
                print_physical_warning()
                if quantity.lower() == 'time':
                    fac= time_in_Gyr(vo,ro)
                elif quantity.lower() == 'position':
                    fac= ro
                elif quantity.lower() == 'velocity':
                    fac= vo
                elif quantity.lower() == 'frequency':
                    if kwargs.get('kmskpc',False):
                        fac= freq_in_kmskpc(vo,ro)
                    else:
                        fac= freq_in_Gyr(vo,ro)
                elif quantity.lower() == 'action':
                    fac= ro*vo
                elif quantity.lower() == 'energy':
                    fac= vo**2.
                return method(*args,**kwargs)*fac
            else:
                return method(*args,**kwargs)
        return wrapped
    return wrapper
