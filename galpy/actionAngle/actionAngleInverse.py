###############################################################################
# actionAngleInverse.py: top-level class with common routines for inverse
#                        actionAngle methods
###############################################################################
from ..util.conversion import (
    actionAngleInverse_physical_input,
    physical_conversion_actionAngleInverse,
)
from .actionAngle import actionAngle


class actionAngleInverse(actionAngle):
    """actionAngleInverse; top-level class with common routines for inverse actionAngle methods"""

    def __init__(self, *args, **kwargs):
        actionAngle.__init__(self, ro=kwargs.get("ro", None), vo=kwargs.get("vo", None))

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("__call__", pop=True)
    def __call__(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) at the given actions and angles, which broadcast against each other: one torus at many angles, or many tori each at their own angle.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            (R,vR,vT,z,vz,phi), each in the broadcast shape of the actions and angles.

        Notes
        -----
        - 2017-11-14 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions, broadcast against the angles - Bovy (UofT)

        """
        try:
            return self._evaluate(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'__call__' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("xvFreqs", pop=True)
    def xvFreqs(self, *args, **kwargs):
        """
        Evaluate the phase-space coordinates (x,v) and the frequencies at the given actions and angles, which broadcast against each other: one torus at many angles, or many tori each at their own angle.

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.
        angler : float or numpy.ndarray
            Radial angle.
        anglephi : float or numpy.ndarray
            Azimuthal angle.
        anglez : float or numpy.ndarray
            Vertical angle.

        Returns
        -------
        tuple
            A tuple containing the phase-space coordinates (R,vR,vT,z,vz,phi) and the frequencies (OmegaR,Omegaphi,Omegaz), each in the broadcast shape of the actions and angles; the frequencies are floats for a single torus.

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions, broadcast against the angles - Bovy (UofT)

        """
        try:
            return self._xvFreqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'xvFreqs' method not implemented for this actionAngle module"
            )

    @actionAngleInverse_physical_input
    @physical_conversion_actionAngleInverse("Freqs", pop=True)
    def Freqs(self, *args, **kwargs):
        """
        Return the frequencies of a torus or of an array of tori

        Parameters
        ----------
        jr : float or numpy.ndarray
            Radial action.
        jphi : float or numpy.ndarray
            Azimuthal action.
        jz : float or numpy.ndarray
            Vertical action.

        Returns
        -------
        tuple
            The frequencies (OmegaR, Omegaphi, Omegaz): three floats for a single torus, three arrays in the broadcast shape of the actions otherwise.

        Notes
        -----
        - 2017-11-15 - Written - Bovy (UofT)
        - 2026-11-08 - Arrays of actions - Bovy (UofT)

        """
        try:
            return self._Freqs(*args, **kwargs)
        except AttributeError:  # pragma: no cover
            raise NotImplementedError(
                "'Freqs' method not implemented for this actionAngle module"
            )
