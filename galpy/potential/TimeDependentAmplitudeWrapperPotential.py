###############################################################################
#   TimeDependentAmplitudeWrapperPotential.py: Wrapper to change the amplitude
#                                              of any potential with an
#                                              arbitrary function of time
###############################################################################
from inspect import signature
from numbers import Number

from numpy import empty

from .WrapperPotential import parentWrapperPotential


class TimeDependentAmplitudeWrapperPotential(parentWrapperPotential):
    """Potential wrapper class that allows the amplitude of any potential to
    be any function of time. That is, the amplitude of a potential gets
    modulated to

    .. math::

        \\mathrm{amp} \\rightarrow \\mathrm{amp} \\times A(t)

    where :math:`A(t)` is an arbitrary function of time.
    Note that `amp` itself can already be a function of time.
    """

    def __init__(self, amp=1.0, A=None, pot=None, ro=None, vo=None):
        """
        Initialize a TimeDependentAmplitudeWrapperPotential.

        Parameters
        ----------
        amp : float, optional
            Amplitude to be applied to the potential (default: 1.).
        A : function, optional
            Function of time giving the time-dependence of the amplitude; should be able to be called with a single time and return a numbers.Number (that is, a number); input time is in internal units (see galpy.util.conversion.time_in_Gyr to convert) and output is a dimensionless amplitude modulation.
        pot : Potential instance or list thereof
            The amplitude of this will modified by this wrapper.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - Started - 2022-03-29 - Bovy (UofT)

        """
        if not callable(A):
            raise TypeError(
                "A= input to TimeDependentAmplitudeWrapperPotential should be a function"
            )
        # Check whether there is at least a single parameter and not more than a single
        # argument, such that the function can be called as A(t)
        Aparams = signature(A).parameters
        nparams = 0
        for param in Aparams.keys():
            if Aparams[param].default == Aparams[param].empty or nparams == 0:
                nparams += 1
        if nparams != 1:
            raise TypeError(
                "A= input to TimeDependentAmplitudeWrapperPotential should be a function that can be called with a single parameter"
            )
        # Finally, check that A only returns a single value
        if not isinstance(A(0.0), Number):
            raise TypeError(
                "A= function needs to return a number (specifically, a numbers.Number)"
            )
        self._A = A
        self.hasC = True
        self.hasC_dxdv = True

    def _wrap(self, attribute, *args, **kwargs):
        return self._A(kwargs.get("t", 0.0)) * self._wrap_pot_func(attribute)(
            self._pot, *args, **kwargs
        )
