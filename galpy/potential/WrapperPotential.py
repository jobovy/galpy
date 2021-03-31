###############################################################################
#   WrapperPotential.py: Super-class for wrapper potentials
###############################################################################
from .Potential import Potential, _isNonAxi, _dim
from .planarPotential import planarPotential
from .Potential import _evaluatePotentials, \
    _evaluateRforces, _evaluatephiforces, _evaluatezforces, \
    evaluateR2derivs, evaluatez2derivs, \
    evaluateRzderivs, evaluateDensities
from .planarPotential import _evaluateplanarPotentials, \
    _evaluateplanarRforces, _evaluateplanarphiforces, \
    evaluateplanarR2derivs
from ..util.conversion import physical_compatible, get_physical
def _new_obj(cls, kwargs, args):
    """Maps kwargs to cls.__new__"""
    return cls.__new__(cls, *args, **kwargs)

class parentWrapperPotential(object):
    """'Dummy' class only used to delegate wrappers to either 2D planarWrapperPotential or 3D WrapperPotential based on pot's dimensionality, using a little python object creation magic..."""
    def __new__(cls,*args,**kwargs):
        if kwargs.pop('_init',False):
            # When we get here recursively, just create new object
            return object.__new__(cls)
        # Decide whether superclass is Wrapper or planarWrapper based on dim
        pot= kwargs.get('pot',None)
        if _dim(pot) == 2:
            parentWrapperPotential= planarWrapperPotential
        elif _dim(pot) == 3:
            parentWrapperPotential= WrapperPotential
        else:
            raise ValueError("WrapperPotentials are only supported in 3D and 2D")
        # Create object from custom class that derives from correct wrapper,
        # make sure to turn off normalization for all wrappers
        kwargs['_init']= True # to break recursion above
        # __reduce__ method to allow pickling
        reduce= lambda self: (_new_obj, (cls, kwargs, args), self.__dict__)
        out= type.__new__(type,'_%s' % cls.__name__,
                          (parentWrapperPotential,cls),
                          {'normalize':property(),
                           '__reduce__':reduce})(*args,**kwargs)
        kwargs.pop('_init',False)
        # This runs init for the subclass (the specific wrapper)
        cls.__init__(out,*args,**kwargs)
        return out

class WrapperPotential(Potential):
    def __init__(self,amp=1.,pot=None,ro=None,vo=None,_init=None,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a WrapperPotential, a super-class for wrapper potentials

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof

        OUTPUT:

           (none)

        HISTORY:

           2017-06-26 - Started - Bovy (UofT)

        """
        if not _init: return None # Don't run __init__ at the end of setup
        Potential.__init__(self,amp=amp,ro=ro,vo=vo)
        self._pot= pot
        self.isNonAxi= _isNonAxi(self._pot)
        # Check whether units are consistent between the wrapper and the 
        # wrapped potential
        assert physical_compatible(self,self._pot), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between this wrapper and the wrapped potential"""
        # Transfer unit system if set for wrapped potential, but not here
        phys_wrapped= get_physical(self._pot,include_set=True)
        if not self._roSet and phys_wrapped['roSet']:
            self.turn_physical_on(ro=phys_wrapped['ro'],vo=False)
        if not self._voSet and phys_wrapped['voSet']:
            self.turn_physical_on(vo=phys_wrapped['vo'],ro=False)

    def __repr__(self):
        wrapped_repr= repr(self._pot)
        return Potential.__repr__(self) + ', wrapper of' \
            + ''.join(['\n\t{}'.format(s) for s in wrapped_repr.split('\n')])

    def __getattr__(self,attribute):
        if attribute == '_evaluate' \
                or attribute == '_Rforce' or attribute == '_zforce' \
                or attribute == '_phiforce' \
                or attribute == '_R2deriv' or attribute == '_z2deriv' \
                or attribute == '_Rzderiv' or attribute == '_phi2deriv' \
                or attribute == '_Rphideriv' or attribute == '_dens':
            return lambda R,Z,phi=0.,t=0.: \
                self._wrap(attribute,R,Z,phi=phi,t=t)
        else:
            return super(WrapperPotential,self).__getattr__(attribute)

    def _wrap_pot_func(self,attribute):
        if attribute == '_evaluate':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluatePotentials(p,R,Z,phi=phi,t=t)
        elif attribute == '_dens':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluateDensities(p,R,Z,phi=phi,t=t,use_physical=False)
        elif attribute == '_Rforce':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluateRforces(p,R,Z,phi=phi,t=t)
        elif attribute == '_zforce':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluatezforces(p,R,Z,phi=phi,t=t)
        elif attribute == '_phiforce':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluatephiforces(p,R,Z,phi=phi,t=t)
        elif attribute == '_R2deriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluateR2derivs(p,R,Z,phi=phi,t=t,use_physical=False)
        elif attribute == '_z2deriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluatez2derivs(p,R,Z,phi=phi,t=t,use_physical=False)
        elif attribute == '_Rzderiv':
            return lambda p,R,Z,phi=0.,t=0.: \
                evaluateRzderivs(p,R,Z,phi=phi,t=t,use_physical=False)
        elif attribute == '_phi2deriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluatePotentials(p,R,Z,phi=phi,t=t,dphi=2)
        elif attribute == '_Rphideriv':
            return lambda p,R,Z,phi=0.,t=0.: \
                _evaluatePotentials(p,R,Z,phi=phi,t=t,dR=1,dphi=1)
        else: #pragma: no cover
            raise AttributeError("Attribute %s not found in for this WrapperPotential" % attribute)

class planarWrapperPotential(planarPotential):
    def __init__(self,amp=1.,pot=None,ro=None,vo=None,_init=None,**kwargs):
        """
        NAME:

           __init__

        PURPOSE:

           initialize a WrapperPotential, a super-class for wrapper potentials

        INPUT:

           amp - amplitude to be applied to the potential (default: 1.)

           pot - Potential instance or list thereof; the amplitude of this will be grown by this wrapper

        OUTPUT:

           (none)

        HISTORY:

           2017-06-26 - Started - Bovy (UofT)

        """
        if not _init: return None # Don't run __init__ at the end of setup
        planarPotential.__init__(self,amp=amp,ro=ro,vo=vo)
        self._pot= pot
        self.isNonAxi= _isNonAxi(self._pot)
        # Check whether units are consistent between the wrapper and the 
        # wrapped potential
        assert physical_compatible(self,self._pot), \
            """Physical unit conversion parameters (ro,vo) are not """\
            """compatible between this wrapper and the wrapped potential"""
        # Transfer unit system if set for wrapped potential, but not here
        phys_wrapped= get_physical(self._pot,include_set=True)
        if not self._roSet and phys_wrapped['roSet']:
            self.turn_physical_on(ro=phys_wrapped['ro'],vo=False)
        if not self._voSet and phys_wrapped['voSet']:
            self.turn_physical_on(vo=phys_wrapped['vo'],ro=False)

    def __repr__(self):
        wrapped_repr= repr(self._pot)
        return Potential.__repr__(self) + ', wrapper of' \
            + ''.join(['\n\t{}'.format(s) for s in wrapped_repr.split('\n')])

    def __getattr__(self,attribute):
        if attribute == '_evaluate' \
                or attribute == '_Rforce' \
                or attribute == '_phiforce' \
                or attribute == '_R2deriv' \
                or attribute == '_phi2deriv' \
                or attribute == '_Rphideriv':
            return lambda R,phi=0.,t=0.: \
                self._wrap(attribute,R,phi=phi,t=t)
        else:
            return super(planarWrapperPotential,self).__getattr__(attribute)

    def _wrap_pot_func(self,attribute):
        if attribute == '_evaluate':
            return lambda p,R,phi=0.,t=0.: \
                _evaluateplanarPotentials(p,R,phi=phi,t=t)
        elif attribute == '_Rforce':
            return lambda p,R,phi=0.,t=0.: \
                _evaluateplanarRforces(p,R,phi=phi,t=t)
        elif attribute == '_phiforce':
            return lambda p,R,phi=0.,t=0.: \
                _evaluateplanarphiforces(p,R,phi=phi,t=t)
        elif attribute == '_R2deriv':
            return lambda p,R,phi=0.,t=0.: \
                evaluateplanarR2derivs(p,R,phi=phi,t=t,use_physical=False)
        elif attribute == '_phi2deriv':
            return lambda p,R,phi=0.,t=0.: \
                _evaluateplanarPotentials(p,R,phi=phi,t=t,dphi=2)
        elif attribute == '_Rphideriv':
            return lambda p,R,phi=0.,t=0.: \
                _evaluateplanarPotentials(p,R,phi=phi,t=t,dR=1,dphi=1)
        else: #pragma: no cover
            raise AttributeError("Attribute %s not found in for this WrapperPotential" % attribute)

