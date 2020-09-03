###############################################################################
#   Force.py: top-level class for a 3D force, conservative (Potential) or 
#             not (DissipativeForce)
#
###############################################################################
import copy
import numpy
from ..util import config
from ..util import conversion
from ..util.conversion import physical_conversion, \
    potential_physical_input, physical_compatible, _APY_LOADED
if _APY_LOADED:
    from astropy import units
class Force(object):
    """Top-level class for any force, conservative or dissipative"""
    def __init__(self,amp=1.,ro=None,vo=None,amp_units=None):
        """
        NAME:

           __init__

        PURPOSE:

           Initialize Force

        INPUT:

           amp - amplitude to be applied when evaluating the potential and its forces
           
           ro - physical distance scale (in kpc or as Quantity)

           vo - physical velocity scale (in km/s or as Quantity)

           amp_units - ('mass', 'velocity2', 'density') type of units that amp should have if it has units
           
        OUTPUT:

        HISTORY:
           2018-03-18 - Written to generalize Potential to force that may or may not be conservative - Bovy (UofT)
        """
        self._amp= amp
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            self._ro= conversion.parse_length_kpc(ro)
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= conversion.parse_velocity_kms(vo)
            self._voSet= True
        # Parse amp if it has units
        if _APY_LOADED and isinstance(self._amp,units.Quantity):
            # Try a bunch of possible units
            unitFound= False
            # velocity^2
            try:
                self._amp= conversion.parse_energy(self._amp,vo=self._vo)
            except units.UnitConversionError: pass
            else:
                unitFound= True
                if not amp_units == 'velocity2':
                    raise units.UnitConversionError('amp= parameter of %s should have units of %s, but has units of velocity2 instead' % (type(self).__name__,amp_units))
            if not unitFound:
                # mass
                try:
                    self._amp= conversion.parse_mass(self._amp,ro=self._ro,vo=self._vo)
                except units.UnitConversionError: pass
                else:
                    unitFound= True
                    if not amp_units == 'mass':
                        raise units.UnitConversionError('amp= parameter of %s should have units of %s, but has units of mass instead' % (type(self).__name__,amp_units))
            if not unitFound:
                # density
                try:
                    self._amp= conversion.parse_dens(self._amp,ro=self._ro,vo=self._vo)
                except units.UnitConversionError: pass
                else:
                    unitFound= True
                    if not amp_units == 'density':
                        raise units.UnitConversionError('amp= parameter of %s should have units of %s, but has units of density instead' % (type(self).__name__,amp_units))
            if not unitFound:
                # surface density
                try:
                    self._amp= conversion.parse_surfdens(self._amp,ro=self._ro,vo=self._vo)
                except units.UnitConversionError: pass
                else:
                    unitFound= True
                    if not amp_units == 'surfacedensity':
                        raise units.UnitConversionError('amp= parameter of %s should have units of %s, but has units of surface density instead' % (type(self).__name__,amp_units))
            if not unitFound:
                raise units.UnitConversionError('amp= parameter of %s should have units of %s; given units are not understood' % (type(self).__name__,amp_units))    
            else:
                # When amplitude is given with units, turn on physical output
                self._roSet= True
                self._voSet= True
        return None

    def __mul__(self,b):
        """
        NAME:

           __mul__

        PURPOSE:

           Multiply a Force or Potential's amplitude by a number

        INPUT:

           b - number

        OUTPUT:

           New instance with amplitude = (old amplitude) x b

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

        """
        if not isinstance(b,(int,float)):
            raise TypeError("Can only multiply a Force or Potential instance with a number")
        out= copy.deepcopy(self)
        out._amp*= b
        return out
    # Similar functions
    __rmul__= __mul__
    def __div__(self,b): return self.__mul__(1./b)
    __truediv__= __div__

    def __add__(self,b):
        """
        NAME:

           __add__

        PURPOSE:

           Add Force or Potential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        INPUT:

           b - Force or Potential instance or a list thereof

        OUTPUT:

           List of Force or Potential instances that represents the combined potential

        HISTORY:

           2019-01-27 - Written - Bovy (UofT)

           2020-04-22 - Added check that unit systems of combined potentials are compatible - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot
        from ..potential import planarPotential
        if not isinstance(flatten_pot([b])[0],(Force,planarPotential)):
            raise TypeError("""Can only combine galpy Force objects with """
                            """other Force objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        if isinstance(b,list):
            return [self]+b
        else:
            return [self,b]
    # Define separately to keep order
    def __radd__(self,b):
        from ..potential import flatten as flatten_pot
        from ..potential import planarPotential
        if not isinstance(flatten_pot([b])[0],(Force,planarPotential)):
            raise TypeError("""Can only combine galpy Force objects with """
                            """other Force objects or lists thereof""")
        assert physical_compatible(self,b), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between potentials to be combined"""
        # If we get here, b has to be a list
        return b+[self]

    def turn_physical_off(self):
        """
        NAME:

           turn_physical_off

        PURPOSE:

           turn off automatic returning of outputs in physical units

        INPUT:

           (none)

        OUTPUT:

           (none)

        HISTORY:

           2016-01-30 - Written - Bovy (UofT)

        """
        self._roSet= False
        self._voSet= False
        return None

    def turn_physical_on(self,ro=None,vo=None):
        """
        NAME:

           turn_physical_on

        PURPOSE:

           turn on automatic returning of outputs in physical units

        INPUT:

           ro= reference distance (kpc; can be Quantity)

           vo= reference velocity (km/s; can be Quantity)

        OUTPUT:

           (none)

        HISTORY:

           2016-01-30 - Written - Bovy (UofT)

           2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False: self._roSet= True
        if not vo is False: self._voSet= True
        if not ro is None and ro:
            self._ro= conversion.parse_length_kpc(ro)
        if not vo is None and vo:
            self._vo= conversion.parse_velocity_kms(vo)
        return None

    @potential_physical_input
    @physical_conversion('force',pop=True)
    def rforce(self,*args,**kwargs):
        """
        NAME:

           rforce

        PURPOSE:

           evaluate spherical radial force F_r  (R,z)

        INPUT:

           R - Cylindrical Galactocentric radius (can be Quantity)

           z - vertical height (can be Quantity)

           phi - azimuth (optional; can be Quantity)

           t - time (optional; can be Quantity)

           v - current velocity in cylindrical coordinates (optional, but required when including dissipative forces; can be a Quantity)

        OUTPUT:

           F_r (R,z,phi,t)

        HISTORY:

           2016-06-20 - Written - Bovy (UofT)

        """
        R,z= args
        r= numpy.sqrt(R**2.+z**2.)
        kwargs['use_physical']= False
        return self.Rforce(*args,**kwargs)*R/r+self.zforce(*args,**kwargs)*z/r
