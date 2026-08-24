###############################################################################
# actionAngleInverse.py: top-level class with common routines for inverse
#                        actionAngle methods
###############################################################################
from ..util.conversion import (
    actionAngleInverse_integral_input,
    actionAngleInverse_physical_input,
    physical_conversion_actionAngleInverse,
)
from .actionAngle import actionAngle


class actionAngleInverse(actionAngle):
    """actionAngleInverse; top-level class with common routines for inverse actionAngle methods"""

    # Subclasses that can label a torus by its integrals of motion as well as
    # by its actions list them here as (name, dimension) pairs; see
    # actionAngleInverse_integral_input
    _integral_labels = ()

    def __init__(self, *args, **kwargs):
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))

    @actionAngleInverse_integral_input
    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("__call__", pop=True)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus.

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : numpy.ndarray
            Vertical angle.

        Returns
        -------
        numpy.ndarray
            [R,vR,vT,z,vz,phi]

        Notes
        -----
        Notes
        -----
        Subclasses that can also label a torus by its integrals of motion
        accept those by name instead of the actions, e.g., ``E=``, ``Lz=``,
        ``I3=`` for actionAngleStaeckelInverse; the actions are then omitted
        and only the angles are given positionally. Naming them works the
        same with and without units, which their dimensions alone cannot do,
        because in internal units the integrals are indistinguishable from
        the actions.

        - 2017-11-14 - Written - Bovy (UofT)

        """
        try:
            return self._evaluate(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'__call__' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_integral_input
    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("xvFreqs", pop=True)
    def xvFreqs(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) for a number of angles on a single torus as well as the frequencies

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.
        angler : numpy.ndarray
            Radial angle.
        anglephi : numpy.ndarray
            Azimuthal angle.
        anglez : numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            A tuple containing the phase-space coordinates (R,vR,vT,z,vz,phi) and the frequencies (OmegaR,Omegaphi,Omegaz).

        Notes
        -----
        Subclasses that can also label a torus by its integrals of motion
        accept those by name instead of the actions, e.g., ``E=``, ``Lz=``,
        ``I3=`` for actionAngleStaeckelInverse; the actions are then omitted
        and only the angles are given positionally. Naming them works the
        same with and without units, which their dimensions alone cannot do,
        because in internal units the integrals are indistinguishable from
        the actions.

        - 2017-11-15 - Written - Bovy (UofT)

        """
        try:
            return self._xvFreqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'xvFreqs' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_integral_input
    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("Freqs", pop=True)
    def Freqs(self, *args, **kwargs):
        """
        Return the frequencies corresponding to a torus

        Parameters
        ----------
        jr : float
            Radial action.
        jphi : float
            Azimuthal action.
        jz : float
            Vertical action.

        Returns
        -------
        tuple
            A tuple of three floats representing the frequencies (OmegaR, Omegaphi, Omegaz).

        Notes
        -----
        Subclasses that can also label a torus by its integrals of motion
        accept those by name instead of the actions, e.g., ``E=``, ``Lz=``,
        ``I3=`` for actionAngleStaeckelInverse; the actions are then omitted
        and only the angles are given positionally. Naming them works the
        same with and without units, which their dimensions alone cannot do,
        because in internal units the integrals are indistinguishable from
        the actions.

        - 2017-11-15 - Written - Bovy (UofT)

        """
        try:
            return self._Freqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'Freqs' method not implemented for this actionAngle module"
            )
