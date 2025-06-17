# Class that implements isotropic spherical Plummer DF
import numpy

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
        assert isinstance(
            pot, PlummerPotential
        ), "pot= must be potential.PlummerPotential"
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
        Etilde = -conversion.parse_energy(E, vo=self._vo)
        out = numpy.zeros_like(Etilde)
        indx = (Etilde > 0) * (Etilde <= self._Etildemax)
        out[indx] = self._fEnorm * (Etilde[indx]) ** 3.5
        return out

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        return self._pot._b / numpy.sqrt(ms ** (-2.0 / 3.0) - 1.0)
