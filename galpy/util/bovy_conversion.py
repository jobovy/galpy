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
from galpy.util.config import __config__
_APY_UNITS= __config__.getboolean('astropy','astropy-units')
_APY_LOADED= True
try:
    from astropy import units, constants
except ImportError:
    _APY_UNITS= False
    _APY_LOADED= False
    _G= 4.302*10.**-3. #pc / Msolar (km/s)^2
    _kmsInPcMyr= 1.0227121655399913
    _PCIN10p18CM= 3.08567758 #10^18 cm
    _CIN10p5KMS= 2.99792458 #10^5 km/s
    _MSOLAR10p30KG= 1.9891 #10^30 kg
    _EVIN10m19J= 1.60217657 #10^-19 J
else:
    _G= constants.G.to(units.pc/units.Msun*units.km**2/units.s**2).value
    _kmsInPcMyr= (units.km/units.s).to((units.pc/units.Myr))
    _PCIN10p18CM= units.pc.to(units.cm)/10.**18. #10^18 cm
    _CIN10p5KMS= constants.c.to((units.km/units.s)).value/10.**5. #10^5 km/s
    _MSOLAR10p30KG= units.Msun.to(units.kg)/10.**30. #10^30 kg
    _EVIN10m19J= units.eV.to(units.J)*10.**19. #10^-19 J
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

def velocity_in_kpcGyr(vo,ro):
    """
    NAME:

       velocity_in_kpcGyr

    PURPOSE:

       convert a velocity to kpc/Gyr

    INPUT:

       vo - velocity unit in km/s

       ro - length unit in kpc

    OUTPUT:

       conversion from units where vo=1. at ro=1.

    HISTORY:

       2014-12-19 - Written - Bovy (IAS)

    """
    return vo*_kmsInPcMyr

#Decorator to apply these transformations
def print_physical_warning():
    warnings.warn("The behavior of Orbit member functions has changed in versions > 0.1 to return positions in kpc, velocities in km/s, energies and the Jacobi integral in (km/s)^2, the angular momentum o.L() and actions in km/s kpc, frequencies in 1/Gyr, and times and periods in Gyr if a distance and velocity scale was specified upon Orbit initialization with ro=...,vo=...; you can turn this off by specifying use_physical=False when calling the function (e.g., o=Orbit(...); o.R(use_physical=False)",
                  galpyWarning)   
# NOTE: names with underscores in them signify return values that *always* have
# units, which is depended on in the Orbit returns (see issue #326)
_roNecessary= {'time': True,
               'position': True,
               'position_kpc': True,
               'velocity': False,
               'velocity2': False,
               'velocity2surfacendensity': False,
               'velocity_kms': False,
               'energy': False,
               'density': True,
               'numberdensity': True,
               'force': True,
               'velocity2surfacedensity': True,
               'surfacedensity': True,
               'numbersurfacedensity': True,
               'surfacedensitydistance': True,
               'mass': True,
               'action': True,
               'frequency':True,
               'frequency-kmskpc':True,
               'forcederivative':True,
               'angle':True,
               'angle_deg':True,
               'proper-motion_masyr':True,
               'phasespacedensity':True,
               'phasespacedensity2d':True,
               'phasespacedensityvelocity':True,
               'phasespacedensityvelocity2':True,
               'dimensionless':False}
_voNecessary= copy.copy(_roNecessary)
_voNecessary['position']= False
_voNecessary['position_kpc']= False
_voNecessary['angle']= False
_voNecessary['angle_deg']= False
_voNecessary['velocity']= True
_voNecessary['velocity2']= True
_voNecessary['velocity_kms']= True
_voNecessary['energy']= True
def physical_conversion(quantity,pop=False):
    """Decorator to convert to physical coordinates: 
    quantity = [position,velocity,time]"""
    def wrapper(method):
        @wraps(method)
        def wrapped(*args,**kwargs):
            use_physical= kwargs.get('use_physical',True) and \
                not kwargs.get('log',False)
            # Parse whether ro or vo should be considered to be set, because 
            # the return value will have units anyway
            # (like in Orbit methods that return numbers with units, like ra)
            roSet= '_' in quantity # _ in quantity name means always units
            voSet= '_' in quantity # _ in quantity name means always units
            use_physical= use_physical or '_' in quantity # _ in quantity name means always units
            ro= kwargs.get('ro',None)
            if ro is None and \
                    (roSet or (hasattr(args[0],'_roSet') and args[0]._roSet)):
                ro= args[0]._ro
            if ro is None and isinstance(args[0],list) \
                    and hasattr(args[0][0],'_roSet') and args[0][0]._roSet:
                # For lists of Potentials
                ro= args[0][0]._ro
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            vo= kwargs.get('vo',None)
            if vo is None and \
                    (voSet or (hasattr(args[0],'_voSet') and args[0]._voSet)):
                vo= args[0]._vo
            if vo is None and isinstance(args[0],list) \
                    and hasattr(args[0][0],'_voSet') and args[0][0]._voSet:
                # For lists of Potentials
                vo= args[0][0]._vo
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            # Override Quantity output?
            _apy_units= kwargs.get('quantity',_APY_UNITS)
            #Remove ro and vo kwargs if necessary
            if pop and 'use_physical' in kwargs: kwargs.pop('use_physical')
            if pop and 'ro' in kwargs: kwargs.pop('ro')
            if pop and 'vo' in kwargs: kwargs.pop('vo')
            if use_physical and \
                    not (_voNecessary[quantity.lower()] and vo is None) and \
                    not (_roNecessary[quantity.lower()] and ro is None):
                from galpy.orbit import Orbit
                if isinstance(args[0],Orbit): print_physical_warning()
                if quantity.lower() == 'time':
                    fac= time_in_Gyr(vo,ro)
                    if _apy_units:
                        u= units.Gyr
                elif quantity.lower() == 'position':
                    fac= ro
                    if _apy_units:
                        u= units.kpc
                elif quantity.lower() == 'position_kpc': # already in kpc
                    fac= 1.
                    if _apy_units:
                        u= units.kpc
                elif quantity.lower() == 'velocity':
                    fac= vo
                    if _apy_units:
                        u= units.km/units.s
                elif quantity.lower() == 'velocity2':
                    fac= vo**2.
                    if _apy_units:
                        u= (units.km/units.s)**2
                elif quantity.lower() == 'velocity_kms': # already in km/s
                    fac= 1.
                    if _apy_units:
                        u= units.km/units.s
                elif quantity.lower() == 'frequency':
                    if kwargs.get('kmskpc',False) and not _apy_units:
                        fac= freq_in_kmskpc(vo,ro)
                    else:
                        fac= freq_in_Gyr(vo,ro)
                        if _apy_units:
                            u= units.Gyr**-1.
                elif quantity.lower() == 'frequency-kmskpc':
                    fac= freq_in_kmskpc(vo,ro)
                    if _apy_units:
                        u= units.km/units.s/units.kpc
                elif quantity.lower() == 'action':
                    fac= ro*vo
                    if _apy_units:
                        u= units.kpc*units.km/units.s
                elif quantity.lower() == 'energy':
                    fac= vo**2.
                    if _apy_units:
                        u= units.km**2./units.s**2.
                elif quantity.lower() == 'angle': # in rad
                    fac= 1.
                    if _apy_units:
                        u= units.rad
                elif quantity.lower() == 'angle_deg': # already in deg
                    fac= 1.
                    if _apy_units:
                        u= units.deg
                elif quantity.lower() == 'proper-motion_masyr': # already in mas/yr
                    fac= 1.
                    if _apy_units:
                        u= units.mas/units.yr
                elif quantity.lower() == 'force':
                    fac= force_in_kmsMyr(vo,ro)
                    if _apy_units:
                        u= units.km/units.s/units.Myr
                elif quantity.lower() == 'density':
                    fac= dens_in_msolpc3(vo,ro)
                    if _apy_units:
                        u= units.Msun/units.pc**3
                elif quantity.lower() == 'numberdensity':
                    fac= 1/ro**3.
                    if _apy_units:
                        u= 1/units.kpc**3
                elif quantity.lower() == 'velocity2surfacedensity':
                    fac= surfdens_in_msolpc2(vo,ro)*vo**2
                    if _apy_units:
                        u= units.Msun/units.pc**2*(units.km/units.s)**2
                elif quantity.lower() == 'surfacedensity':
                    fac= surfdens_in_msolpc2(vo,ro)
                    if _apy_units:
                        u= units.Msun/units.pc**2
                elif quantity.lower() == 'numbersurfacedensity':
                    fac= 1./ro**2.
                    if _apy_units:
                        u= 1/units.kpc**2
                elif quantity.lower() == 'surfacedensitydistance':
                    fac= surfdens_in_msolpc2(vo,ro)*ro*1000.
                    if _apy_units:
                        u= units.Msun/units.pc
                elif quantity.lower() == 'mass':
                    fac= mass_in_msol(vo,ro)
                    if _apy_units:
                        u= units.Msun
                elif quantity.lower() == 'forcederivative':
                    fac= freq_in_Gyr(vo,ro)**2.
                    if _apy_units:
                        u= units.Gyr**-2.
                elif quantity.lower() == 'phasespacedensity':
                    fac= 1./vo**3./ro**3.
                    if _apy_units:
                        u= 1/(units.km/units.s)**3/units.kpc**3
                elif quantity.lower() == 'phasespacedensity2d':
                    fac= 1./vo**2./ro**2.
                    if _apy_units:
                        u= 1/(units.km/units.s)**2/units.kpc**2
                elif quantity.lower() == 'phasespacedensityvelocity':
                    fac= 1./vo**2./ro**3.
                    if _apy_units:
                        u= 1/(units.km/units.s)**2/units.kpc**3
                elif quantity.lower() == 'phasespacedensityvelocity2':
                    fac= 1./vo/ro**3.
                    if _apy_units:
                        u= 1/(units.km/units.s)/units.kpc**3
                elif quantity.lower() == 'dimensionless':
                    fac= 1.
                    if _apy_units:
                        u= units.dimensionless_unscaled
                out= method(*args,**kwargs)
                if out is None:
                    return out
                if _apy_units:
                    return units.Quantity(out*fac,unit=u)
                else:
                    return out*fac
            else:
                return method(*args,**kwargs)
        return wrapped
    return wrapper
def potential_physical_input(method):
    """Decorator to convert inputs to Potential functions from physical 
    to internal coordinates"""
    @wraps(method)
    def wrapper(*args,**kwargs):
        from galpy.potential import flatten as flatten_potential
        Pot= flatten_potential(args[0])
        ro= kwargs.get('ro',None)
        if ro is None and hasattr(Pot,'_ro'):
            ro= Pot._ro
        if ro is None and isinstance(Pot,list) \
                and hasattr(Pot[0],'_ro'):
            # For lists of Potentials
            ro= Pot[0]._ro
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        if 't' in kwargs or 'M' in kwargs:
            vo= kwargs.get('vo',None)
            if vo is None and hasattr(Pot,'_vo'):
                vo= Pot._vo
            if vo is None and isinstance(Pot,list) \
                    and hasattr(Pot[0],'_vo'):
                # For lists of Potentials
                vo= Pot[0]._vo
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
        # Loop through args
        newargs= (Pot,)
        for ii in range(1,len(args)):
            if _APY_LOADED and isinstance(args[ii],units.Quantity):
                newargs= newargs+(args[ii].to(units.kpc).value/ro,)
            else:
                newargs= newargs+(args[ii],)
        args= newargs
        # phi and t kwargs
        if 'phi' in kwargs and _APY_LOADED \
                and isinstance(kwargs['phi'],units.Quantity):
            kwargs['phi']= kwargs['phi'].to(units.rad).value
        if 't' in kwargs and _APY_LOADED \
                and isinstance(kwargs['t'],units.Quantity):
            kwargs['t']= kwargs['t'].to(units.Gyr).value\
                /time_in_Gyr(vo,ro)
        # v kwarg for dissipative forces
        if 'v' in kwargs and _APY_LOADED \
                and isinstance(kwargs['v'],units.Quantity):
            kwargs['v']= kwargs['v'].to(units.km/units.s).value/vo
        # Mass kwarg for rtide
        if 'M' in kwargs and _APY_LOADED \
                and isinstance(kwargs['M'],units.Quantity):
            try:
                kwargs['M']= kwargs['M'].to(units.Msun).value\
                    /mass_in_msol(vo,ro)
            except units.UnitConversionError:
                kwargs['M']= kwargs['M'].to(units.pc*units.km**2/units.s**2)\
                    .value/mass_in_msol(vo,ro)/_G
        # kwargs that come up in quasiisothermaldf    
        if 'z' in kwargs and _APY_LOADED \
                and isinstance(kwargs['z'],units.Quantity):
            kwargs['z']= kwargs['z'].to(units.kpc).value/ro
        if 'dz' in kwargs and _APY_LOADED \
                and isinstance(kwargs['dz'],units.Quantity):
            kwargs['dz']= kwargs['dz'].to(units.kpc).value/ro
        if 'dR' in kwargs and _APY_LOADED \
                and isinstance(kwargs['dR'],units.Quantity):
            kwargs['dR']= kwargs['dR'].to(units.kpc).value/ro
        if 'zmax' in kwargs and _APY_LOADED \
                and isinstance(kwargs['zmax'],units.Quantity):
            kwargs['zmax']= kwargs['zmax'].to(units.kpc).value/ro
        return method(*args,**kwargs)
    return wrapper
def physical_conversion_actionAngle(quantity,pop=False):
    """Decorator to convert to physical coordinates for the actionAngle methods: 
    quantity= call, actionsFreqs, or actionsFreqsAngles (or EccZmaxRperiRap for actionAngleStaeckel)"""
    def wrapper(method):
        @wraps(method)
        def wrapped(*args,**kwargs):
            use_physical= kwargs.get('use_physical',True)
            ro= kwargs.get('ro',None)
            if ro is None and hasattr(args[0],'_roSet') and args[0]._roSet:
                ro= args[0]._ro
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            vo= kwargs.get('vo',None)
            if vo is None and hasattr(args[0],'_voSet') and args[0]._voSet:
                vo= args[0]._vo
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            #Remove ro and vo kwargs if necessary
            if pop and 'use_physical' in kwargs: kwargs.pop('use_physical')
            if pop and 'ro' in kwargs: kwargs.pop('ro')
            if pop and 'vo' in kwargs: kwargs.pop('vo')
            if use_physical and not vo is None and not ro is None:
                if 'call' in quantity or 'actions' in quantity:
                    fac= [ro*vo,ro*vo,ro*vo]
                    if _APY_UNITS:
                        u= [units.kpc*units.km/units.s,
                            units.kpc*units.km/units.s,
                            units.kpc*units.km/units.s]
                if 'Freqs' in quantity:
                    FreqsFac= freq_in_Gyr(vo,ro)
                    fac.extend([FreqsFac,FreqsFac,FreqsFac])
                    if _APY_UNITS:
                        Freqsu= units.Gyr**-1.
                        u.extend([Freqsu,Freqsu,Freqsu])
                if 'Angles' in quantity:
                    fac.extend([1.,1.,1.])
                    if _APY_UNITS:
                        Freqsu= units.Gyr**-1.
                        u.extend([units.rad,units.rad,units.rad])
                if 'EccZmaxRperiRap' in quantity:
                    fac= [1.,ro,ro,ro]
                    if _APY_UNITS:
                        u= [1.,
                            units.kpc,
                            units.kpc,
                            units.kpc]
                out= method(*args,**kwargs)
                if _APY_UNITS:
                    newOut= ()
                    for ii in range(len(out)):
                        newOut= newOut+(units.Quantity(out[ii]*fac[ii],
                                                       unit=u[ii]),)
                else:
                    newOut= ()
                    for ii in range(len(out)):
                        newOut= newOut+(out[ii]*fac[ii],)
                return newOut
            else:
                return method(*args,**kwargs)
        return wrapped
    return wrapper

def actionAngle_physical_input(method):
    """Decorator to convert inputs to actionAngle functions from physical 
    to internal coordinates"""
    @wraps(method)
    def wrapper(*args,**kwargs):
        if len(args) < 3: # orbit input
            return method(*args,**kwargs)
        ro= kwargs.get('ro',None)
        if ro is None and hasattr(args[0],'_ro'):
            ro= args[0]._ro
        if _APY_LOADED and isinstance(ro,units.Quantity):
            ro= ro.to(units.kpc).value
        vo= kwargs.get('vo',None)
        if vo is None and hasattr(args[0],'_vo'):
            vo= args[0]._vo
        if _APY_LOADED and isinstance(vo,units.Quantity):
            vo= vo.to(units.km/units.s).value
        # Loop through args
        newargs= ()
        for ii in range(len(args)):
            if _APY_LOADED and isinstance(args[ii],units.Quantity):
                try:
                    targ= args[ii].to(units.kpc).value/ro
                except units.UnitConversionError:
                    try:
                        targ= args[ii].to(units.km/units.s).value/vo
                    except units.UnitConversionError:
                        try:
                            targ= args[ii].to(units.rad).value
                        except units.UnitConversionError:
                            raise units.UnitConversionError("Input units not understood")               
                newargs= newargs+(targ,)
            else:
                newargs= newargs+(args[ii],)
        args= newargs
        return method(*args,**kwargs)
    return wrapper
