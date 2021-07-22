from six import with_metaclass
import types
import copy
import numpy
from ..util import config, conversion
from ..util.conversion import physical_conversion_actionAngle, \
    actionAngle_physical_input, physical_compatible
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
            self._ro= conversion.parse_length_kpc(ro)
            self._roSet= True
        if vo is None:
            self._vo= config.__config__.getfloat('normalization','vo')
            self._voSet= False
        else:
            self._vo= conversion.parse_velocity_kms(vo)
            self._voSet= True
        return None

    def _check_consistent_units(self):
        """Internal function to check that the set of units for this object is consistent with that for the potential"""
        assert physical_compatible(self,self._pot),  'Physical conversion for the actionAngle object is not consistent with that of the Potential given to it'
            
    def _check_consistent_units_orbitInput(self,orb):
        """Internal function to check that the set of units for this object is consistent with that for an input orbit"""
        assert physical_compatible(self,orb),  'Physical conversion for the actionAngle object is not consistent with that of the Orbit given to it'
     
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

           2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False: self._roSet= True
        if not vo is False: self._voSet= True
        if not ro is None and ro:
            self._ro= conversion.parse_length_kpc(ro)
        if not vo is None and vo:
            self._vo= conversion.parse_velocity_kms(vo)
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
        if len(args) == 3: #R, vR, vT
            R,vR,vT= args
            self._eval_R= R
            self._eval_vR= vR
            self._eval_vT= vT
            self._eval_z= numpy.zeros_like(R)
            self._eval_vz= numpy.zeros_like(R)
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
        else: # Orbit instance
            if not kwargs.get('_noOrbUnitsCheck',False):
                self._check_consistent_units_orbitInput(args[0])
            if len(args) == 2:
                orb= args[0](args[1])
            else:
                orb= args[0]
            if len(orb.shape) > 1:
                raise RuntimeError("Evaluating actionAngle methods with Orbit instances with multi-dimensional shapes is not supported")
            self._eval_R= orb.R(use_physical=False)
            self._eval_vR= orb.vR(use_physical=False)
            self._eval_vT= orb.vT(use_physical=False)
            if args[0].phasedim() > 4:
                self._eval_z= orb.z(use_physical=False)
                self._eval_vz= orb.vz(use_physical=False)
                if args[0].phasedim() > 5:
                    self._eval_phi= orb.phi(use_physical=False)
            else:
                if args[0].phasedim() > 3:
                    self._eval_phi= orb.phi(use_physical=False)
                self._eval_z= numpy.zeros_like(self._eval_R)
                self._eval_vz= numpy.zeros_like(self._eval_R)
        if hasattr(self,'_eval_z'): #calculate the polar angle
            self._eval_theta= numpy.arctan2(self._eval_R,self._eval_z)
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
