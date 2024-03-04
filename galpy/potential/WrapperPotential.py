###############################################################################
#   WrapperPotential.py: Super-class for wrapper potentials
###############################################################################
from ..util.conversion import get_physical, physical_compatible
from .planarPotential import (
    _evaluateplanarphitorques,
    _evaluateplanarPotentials,
    _evaluateplanarRforces,
    evaluateplanarR2derivs,
    planarPotential,
)
from .Potential import (
    Force,
    Potential,
    _dim,
    _evaluatephitorques,
    _evaluatePotentials,
    _evaluateRforces,
    _evaluatezforces,
    _isNonAxi,
    evaluateDensities,
    evaluateR2derivs,
    evaluateRzderivs,
    evaluatez2derivs,
)


def _new_obj(cls, kwargs, args):
    """Maps kwargs to cls.__new__"""
    return cls.__new__(cls, *args, **kwargs)


class parentWrapperPotential:
    """'Dummy' class only used to delegate wrappers to either 2D planarWrapperPotential or 3D WrapperPotential based on pot's dimensionality, using a little python object creation magic..."""

    def __new__(cls, *args, **kwargs):
        if kwargs.pop("_init", False):
            # When we get here recursively, just create new object
            return object.__new__(cls)
        # Decide whether superclass is Wrapper or planarWrapper based on dim
        pot = kwargs.get("pot", None)
        if _dim(pot) == 2:
            parentWrapperPotential = planarWrapperPotential
        elif _dim(pot) == 3:
            parentWrapperPotential = WrapperPotential
        else:
            raise ValueError("WrapperPotentials are only supported in 3D and 2D")
        # Create object from custom class that derives from correct wrapper,
        # make sure to turn off normalization for all wrappers
        kwargs["_init"] = True  # to break recursion above
        # __reduce__ method to allow pickling
        reduce = lambda self: (_new_obj, (cls, kwargs, args), self.__dict__)
        out = type.__new__(
            type,
            "_%s" % cls.__name__,
            (parentWrapperPotential, cls),
            {"normalize": property(), "__reduce__": reduce},
        )(*args, **kwargs)
        kwargs.pop("_init", False)
        # This runs init for the subclass (the specific wrapper)
        cls.__init__(out, *args, **kwargs)
        return out


class WrapperPotential(Potential):
    def __init__(self, amp=1.0, pot=None, ro=None, vo=None, _init=None, **kwargs):
        """
        Initialize a WrapperPotential, a super-class for wrapper potentials.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        pot : Potential instance or list thereof
            Potential instance or list thereof.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        _init : bool, optional
            If True, run __init__ at the end of setup (default: True).
        **kwargs
            Any other keyword arguments are passed to the Potential superclass.

        Notes
        -----
        - 2017-06-26 - Started - Bovy (UofT)

        """
        if not _init:
            return None  # Don't run __init__ at the end of setup
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._pot = pot
        # Check that we are not wrapping a non-potential Force object
        if (
            isinstance(self._pot, list)
            and any(
                [
                    isinstance(p, Force) and not isinstance(p, Potential)
                    for p in self._pot
                ]
            )
        ) or (isinstance(self._pot, Force) and not isinstance(self._pot, Potential)):
            raise RuntimeError(
                "WrapperPotential cannot currently wrap non-Potential Force objects"
            )
        self.isNonAxi = _isNonAxi(self._pot)
        # Check whether units are consistent between the wrapper and the
        # wrapped potential
        assert physical_compatible(self, self._pot), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between this wrapper and the wrapped potential"""
        )
        # Transfer unit system if set for wrapped potential, but not here
        phys_wrapped = get_physical(self._pot, include_set=True)
        if not self._roSet and phys_wrapped["roSet"]:
            self.turn_physical_on(ro=phys_wrapped["ro"], vo=False)
        if not self._voSet and phys_wrapped["voSet"]:
            self.turn_physical_on(vo=phys_wrapped["vo"], ro=False)

    def __repr__(self):
        wrapped_repr = repr(self._pot)
        return (
            Potential.__repr__(self)
            + ", wrapper of"
            + "".join([f"\n\t{s}" for s in wrapped_repr.split("\n")])
        )

    def __getattr__(self, attribute):
        if (
            attribute == "_evaluate"
            or attribute == "_Rforce"
            or attribute == "_zforce"
            or attribute == "_phitorque"
            or attribute == "_R2deriv"
            or attribute == "_z2deriv"
            or attribute == "_Rzderiv"
            or attribute == "_phi2deriv"
            or attribute == "_Rphideriv"
            or attribute == "_dens"
        ):
            return lambda R, Z, phi=0.0, t=0.0: self._wrap(
                attribute, R, Z, phi=phi, t=t
            )
        else:
            return super().__getattr__(attribute)

    def _wrap_pot_func(self, attribute):
        if attribute == "_evaluate":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluatePotentials(
                p, R, Z, phi=phi, t=t
            )
        elif attribute == "_dens":
            return lambda p, R, Z, phi=0.0, t=0.0: evaluateDensities(
                p, R, Z, phi=phi, t=t, use_physical=False
            )
        elif attribute == "_Rforce":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluateRforces(
                p, R, Z, phi=phi, t=t
            )
        elif attribute == "_zforce":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluatezforces(
                p, R, Z, phi=phi, t=t
            )
        elif attribute == "_phitorque":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluatephitorques(
                p, R, Z, phi=phi, t=t
            )
        elif attribute == "_R2deriv":
            return lambda p, R, Z, phi=0.0, t=0.0: evaluateR2derivs(
                p, R, Z, phi=phi, t=t, use_physical=False
            )
        elif attribute == "_z2deriv":
            return lambda p, R, Z, phi=0.0, t=0.0: evaluatez2derivs(
                p, R, Z, phi=phi, t=t, use_physical=False
            )
        elif attribute == "_Rzderiv":
            return lambda p, R, Z, phi=0.0, t=0.0: evaluateRzderivs(
                p, R, Z, phi=phi, t=t, use_physical=False
            )
        elif attribute == "_phi2deriv":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluatePotentials(
                p, R, Z, phi=phi, t=t, dphi=2
            )
        elif attribute == "_Rphideriv":
            return lambda p, R, Z, phi=0.0, t=0.0: _evaluatePotentials(
                p, R, Z, phi=phi, t=t, dR=1, dphi=1
            )
        else:  # pragma: no cover
            raise AttributeError(
                "Attribute %s not found in for this WrapperPotential" % attribute
            )


class planarWrapperPotential(planarPotential):
    def __init__(self, amp=1.0, pot=None, ro=None, vo=None, _init=None, **kwargs):
        """
        Initialize a WrapperPotential, a super-class for wrapper potentials.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        pot : Potential instance or list thereof, optional
            The potential instance or list thereof; the amplitude of this will be grown by this wrapper.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).
        _init : bool, optional
            If True, run __init__ at the end of setup. Default is None.
        **kwargs
            Any other keyword arguments are passed to the Potential class.

        Notes
        -----
        - 2017-06-26 - Started - Bovy (UofT)

        """
        if not _init:
            return None  # Don't run __init__ at the end of setup
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        self._pot = pot
        self.isNonAxi = _isNonAxi(self._pot)
        # Check whether units are consistent between the wrapper and the
        # wrapped potential
        assert physical_compatible(self, self._pot), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between this wrapper and the wrapped potential"""
        )
        # Transfer unit system if set for wrapped potential, but not here
        phys_wrapped = get_physical(self._pot, include_set=True)
        if not self._roSet and phys_wrapped["roSet"]:
            self.turn_physical_on(ro=phys_wrapped["ro"], vo=False)
        if not self._voSet and phys_wrapped["voSet"]:
            self.turn_physical_on(vo=phys_wrapped["vo"], ro=False)

    def __repr__(self):
        wrapped_repr = repr(self._pot)
        return (
            Potential.__repr__(self)
            + ", wrapper of"
            + "".join([f"\n\t{s}" for s in wrapped_repr.split("\n")])
        )

    def __getattr__(self, attribute):
        if (
            attribute == "_evaluate"
            or attribute == "_Rforce"
            or attribute == "_phitorque"
            or attribute == "_R2deriv"
            or attribute == "_phi2deriv"
            or attribute == "_Rphideriv"
        ):
            return lambda R, phi=0.0, t=0.0: self._wrap(attribute, R, phi=phi, t=t)
        else:
            return super().__getattr__(attribute)

    def _wrap_pot_func(self, attribute):
        if attribute == "_evaluate":
            return lambda p, R, phi=0.0, t=0.0: _evaluateplanarPotentials(
                p, R, phi=phi, t=t
            )
        elif attribute == "_Rforce":
            return lambda p, R, phi=0.0, t=0.0: _evaluateplanarRforces(
                p, R, phi=phi, t=t
            )
        elif attribute == "_phitorque":
            return lambda p, R, phi=0.0, t=0.0: _evaluateplanarphitorques(
                p, R, phi=phi, t=t
            )
        elif attribute == "_R2deriv":
            return lambda p, R, phi=0.0, t=0.0: evaluateplanarR2derivs(
                p, R, phi=phi, t=t, use_physical=False
            )
        elif attribute == "_phi2deriv":
            return lambda p, R, phi=0.0, t=0.0: _evaluateplanarPotentials(
                p, R, phi=phi, t=t, dphi=2
            )
        elif attribute == "_Rphideriv":
            return lambda p, R, phi=0.0, t=0.0: _evaluateplanarPotentials(
                p, R, phi=phi, t=t, dR=1, dphi=1
            )
        else:  # pragma: no cover
            raise AttributeError(
                "Attribute %s not found in for this WrapperPotential" % attribute
            )
