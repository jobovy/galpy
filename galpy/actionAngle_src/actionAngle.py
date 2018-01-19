from six import with_metaclass
import types
import copy
import math as m
from galpy.util import config
from galpy.util.bovy_conversion import physical_conversion_actionAngle, \
    actionAngle_physical_input
_APY_LOADED= True
try:
    from astropy import units
except ImportError:
    _APY_LOADED= False
# Metaclass for copying docstrings from subclass methods, first func 
# to copy func
def copyfunc(func):
    return types.FunctionType(func.__code__,func.__globals__,
                              name=func.__name__,
                              argdefs=func.__defaults__,
                              closure=func.__closure__)
class MetaActionAngle(type):
    """Metaclass to assign subclass' docstrings for methods _evaluate, _actionsFreqs, _actionsFreqsAngles, and _EccZmaxRperiRap to their public cousins __call__, actionsFreqs, etc."""
    def __new__(meta,name,bases,attrs):
        for key in copy.copy(attrs): # copy bc size changes
            if key[0] == '_':
                skey= copy.copy(key[1:])
                if skey == 'evaluate': skey= '__call__'
                for base in bases:
                    original= getattr(base,skey,None)
                    if original is not None:
                        funccopy= copyfunc(original)
                        funccopy.__doc__= attrs[key].__doc__
                        attrs[skey]= funccopy
                        break
        return type.__new__(meta,name,bases,attrs)

# Python 2 & 3 compatible way to have a metaclass
class actionAngle(with_metaclass(MetaActionAngle,object)):
    """Top-level class for actionAngle classes"""
    def __init__(self,ro=None,vo=None):
        """
        NAME:
           __init__
        PURPOSE:
           initialize an actionAngle object
        INPUT:
           ro= (None) distance scale
           vo= (None) velocity scale
        OUTPUT:
        HISTORY:
           2016-02-18 - Written - Bovy (UofT)
        """
        # Parse ro and vo
        if ro is None:
            self._ro= config.__config__.getfloat('normalization','ro')
            self._roSet= False
        else:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
            self._voSet= True
        return None

    def _check_consistent_units(self):
        """Internal function to check that the set of units for this object is consistent with that for the potential"""
        if isinstance(self._pot,list):
            if self._roSet and self._pot[0]._roSet:
                assert m.fabs(self._ro-self._pot[0]._ro) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Potential given to it'
            if self._voSet and self._pot[0]._voSet:
                assert m.fabs(self._vo-self._pot[0]._vo) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Potential given to it'
        else:
            if self._roSet and self._pot._roSet:
                assert m.fabs(self._ro-self._pot._ro) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Potential given to it'
            if self._voSet and self._pot._voSet:
                assert m.fabs(self._vo-self._pot._vo) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Potential given to it'
        return None
            
    def _check_consistent_units_orbitInput(self,orb):
        """Internal function to check that the set of units for this object is consistent with that for an input orbit"""
        if self._roSet and orb._roSet:
            assert m.fabs(self._ro-orb._ro) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Orbit given to it'
        if self._voSet and orb._voSet:
            assert m.fabs(self._vo-orb._vo) < 10.**-10., 'Physical conversion for the actionAngle object is not consistent with that of the Orbit given to it'
        return None
            
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

           2017-06-05 - Written - Bovy (UofT)

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

           2016-06-05 - Written - Bovy (UofT)

        """
        self._roSet= True
        self._voSet= True
        if not ro is None:
            if _APY_LOADED and isinstance(ro,units.Quantity):
                ro= ro.to(units.kpc).value
            self._ro= ro
        if not vo is None:
            if _APY_LOADED and isinstance(vo,units.Quantity):
                vo= vo.to(units.km/units.s).value
            self._vo= vo
        return None  

    def _parse_eval_args(self,*args,**kwargs):
        """
        NAME:
           _parse_eval_args
        PURPOSE:
           Internal function to parse the arguments given for an action/frequency/angle evaluation
        INPUT:
        OUTPUT:
        HISTORY:
           2010-07-11 - Written - Bovy (NYU)
        """
        if len(args) == 3: #R,vR.vT
            R,vR,vT= args
            self._eval_R= R
            self._eval_vR= vR
            self._eval_vT= vT
            self._eval_z= 0.
            self._eval_vz= 0.
        elif len(args) == 5: #R,vR.vT, z, vz
            R,vR,vT, z, vz= args
            self._eval_R= R
            self._eval_vR= vR
            self._eval_vT= vT
            self._eval_z= z
            self._eval_vz= vz
        elif len(args) == 6: #R,vR.vT, z, vz, phi
            R,vR,vT, z, vz, phi= args
            self._eval_R= R
            self._eval_vR= vR
            self._eval_vT= vT
            self._eval_z= z
            self._eval_vz= vz
            self._eval_phi= phi
        else:
            if not kwargs.get('_noOrbUnitsCheck',False):
                self._check_consistent_units_orbitInput(args[0])
            if len(args) == 2:
                vxvv= args[0](args[1])._orb.vxvv
            else:
                try:
                    vxvv= args[0]._orb.vxvv
                except AttributeError: #if we're given an OrbitTop instance
                    vxvv= args[0].vxvv
            self._eval_R= vxvv[0]
            self._eval_vR= vxvv[1]
            self._eval_vT= vxvv[2]
            if len(vxvv) > 4:
                self._eval_z= vxvv[3]
                self._eval_vz= vxvv[4]
                if len(vxvv) > 5:
                    self._eval_phi= vxvv[5]
            elif len(vxvv) > 3:
                self._eval_phi= vxvv[3]
                self._eval_z= 0.
                self._eval_vz= 0.
            else:
                self._eval_z= 0.
                self._eval_vz= 0.
        if hasattr(self,'_eval_z'): #calculate the polar angle
            if self._eval_z == 0.: self._eval_theta= m.pi/2.
            else: self._eval_theta= m.atan(self._eval_R/self._eval_z)
        return None

    @actionAngle_physical_input
    @physical_conversion_actionAngle('__call__',pop=True)
    def __call__(self,*args,**kwargs):
        """
        NAME:

           __call__

        PURPOSE:

           evaluate the actions (jr,lz,jz)

        INPUT:

           Either:

              a) R,vR,vT,z,vz[,phi]:

                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)

                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:

           (jr,lz,jz)

        HISTORY:

           2014-01-03 - Written for top level - Bovy (IAS)

        """
        try:
            return self._evaluate(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'__call__' method not implemented for this actionAngle module")

    @actionAngle_physical_input
    @physical_conversion_actionAngle('actionsFreqs',pop=True)
    def actionsFreqs(self,*args,**kwargs):
        """
        NAME:

           actionsFreqs

        PURPOSE:

           evaluate the actions and frequencies (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        INPUT:

           Either:

              a) R,vR,vT,z,vz[,phi]:

                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)

                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:

            (jr,lz,jz,Omegar,Omegaphi,Omegaz)

        HISTORY:

           2014-01-03 - Written for top level - Bovy (IAS)

        """
        try:
            return self._actionsFreqs(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'actionsFreqs' method not implemented for this actionAngle module")

    @actionAngle_physical_input
    @physical_conversion_actionAngle('actionsFreqsAngles',pop=True)
    def actionsFreqsAngles(self,*args,**kwargs):
        """
        NAME:

           actionsFreqsAngles

        PURPOSE:

           evaluate the actions, frequencies, and angles (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        INPUT:

           Either:

              a) R,vR,vT,z,vz,phi:

                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)

                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:

            (jr,lz,jz,Omegar,Omegaphi,Omegaz,angler,anglephi,anglez)

        HISTORY:

           2014-01-03 - Written for top level - Bovy (IAS)

        """
        try:
            return self._actionsFreqsAngles(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'actionsFreqsAngles' method not implemented for this actionAngle module")

    @actionAngle_physical_input
    @physical_conversion_actionAngle('EccZmaxRperiRap',pop=True)
    def EccZmaxRperiRap(self,*args,**kwargs):
        """
        NAME:

           EccZmaxRperiRap

        PURPOSE:

           evaluate the eccentricity, maximum height above the plane, peri- and apocenter

        INPUT:

           Either:

              a) R,vR,vT,z,vz[,phi]:

                 1) floats: phase-space value for single object (phi is optional) (each can be a Quantity)

                 2) numpy.ndarray: [N] phase-space values for N objects (each can be a Quantity)

              b) Orbit instance: initial condition used if that's it, orbit(t) if there is a time given as well as the second argument
                 
        OUTPUT:

           (e,zmax,rperi,rap)

        HISTORY:

           2017-12-12 - Written - Bovy (UofT)

        """
        try:
            return self._EccZmaxRperiRap(*args,**kwargs)
        except AttributeError: #pragma: no cover
            raise NotImplementedError("'EccZmaxRperiRap' method not implemented for this actionAngle module")

class UnboundError(Exception): #pragma: no cover
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
