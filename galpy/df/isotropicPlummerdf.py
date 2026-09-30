# Class that implements isotropic spherical Plummer DF
import numpy

from ..backend import resolve_namespace
from ..potential import PlummerPotential
from ..util import conversion
from .sphericaldf import isotropicsphericaldf


class isotropicPlummerdf(isotropicsphericaldf):
    """Class that implements isotropic spherical Plummer DF:

    .. math::

        f(E) = {24\\sqrt{2} \\over 7\\pi^3}\\,{b^2\\over (GM)^5}\\,(-E)^{7/2}

    for :math:`-GM/b \\leq E \\leq 0` and zero otherwise. The parameter :math:`GM` is the total mass and :math:`b` the Plummer profile's scale parameter.
    """

    def __init__(self, pot=None, ro=None, vo=None):
        """
        Initialize an isotropic Plummer distribution function

        Parameters
        ----------
        pot : Potential object
            Plummer Potential instance
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-10-01 - Written - Bovy (UofT).
        """
        assert isinstance(pot, PlummerPotential), (
            "pot= must be potential.PlummerPotential"
        )
        isotropicsphericaldf.__init__(self, pot=pot, ro=ro, vo=vo)
        self._Etildemax = pot._amp / pot._b
        # /amp^4 instead of /amp^5 to make the DF that of mass density
        self._fEnorm = (
            24.0 * numpy.sqrt(2.0) / 7.0 / numpy.pi**3.0 * pot._b**2.0 / pot._amp**4.0
        )

    def fE(self, E):
        """
        Calculate the energy portion of an isotropic Plummer distribution function.

        Parameters
        ----------
        E : float or Quantity
            The energy.

        Returns
        -------
        ndarray
            The value of the energy portion of the DF.

        Notes
        -----
        - 2020-10-01 - Written - Bovy (UofT)

        """
        Ei = conversion.parse_energy(E, vo=self._vo)
        # resolve on _Etildemax too so backend-built potential params keep grads
        xp = resolve_namespace(Ei, self._Etildemax)
        if xp is numpy:
            Etilde = -Ei
            out = numpy.zeros_like(Etilde)
            indx = (Etilde > 0) * (Etilde <= self._Etildemax)
            out[indx] = self._fEnorm * (Etilde[indx]) ** 3.5
            return out
        # jax/torch: functional dead-mask (negative base -> NaN power under AD)
        Etilde = -xp.asarray(Ei) * 1.0
        dead = (Etilde <= 0) | (Etilde > self._Etildemax)
        Esafe = xp.where(dead, 0.5 * self._Etildemax, Etilde)
        out = self._fEnorm * Esafe**3.5
        return xp.where(dead, xp.zeros_like(out), out)

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        xp = resolve_namespace(ms)
        if xp is numpy:
            return self._pot._b / numpy.sqrt(ms ** (-2.0 / 3.0) - 1.0)
        msb = xp.asarray(ms) * 1.0  # coerce: torch rejects numpy scalars
        return self._pot._b / xp.sqrt(msb ** (-2.0 / 3.0) - 1.0)
