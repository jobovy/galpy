# Class that implements isotropic spherical Hernquist DF
# computed using the Eddington formula
import numpy

from ..potential import HernquistPotential, evaluatePotentials
from ..util import conversion
from .sphericaldf import isotropicsphericaldf


class isotropicHernquistdf(isotropicsphericaldf):
    """Class that implements isotropic spherical Hernquist DF computed using the Eddington formula"""

    def __init__(self, pot=None, ro=None, vo=None):
        """
        Initialize an isotropic Hernquist distribution function.

        Parameters
        ----------
        pot : HernquistPotential instance
            Hernquist potential instance.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-08-09 - Written - Lane (UofT)
        """
        assert isinstance(
            pot, HernquistPotential
        ), "pot= must be potential.HernquistPotential"
        isotropicsphericaldf.__init__(self, pot=pot, ro=ro, vo=vo)
        self._psi0 = -evaluatePotentials(self._pot, 0, 0, use_physical=False)
        self._GMa = self._psi0 * self._pot.a**2.0
        # first factor = mass to make the DF that of mass density
        self._fEnorm = (
            self._psi0
            * self._pot.a
            / numpy.sqrt(2.0)
            / (2 * numpy.pi) ** 3
            / self._GMa**1.5
        )

    def fE(self, E):
        """
        Calculate the energy portion of an isotropic Hernquist distribution function

        Parameters
        ----------
        E : float or Quantity
            The energy.

        Returns
        -------
        numpy.ndarray
            The value of the energy portion of the DF.

        Notes
        -----
        - 2020-08-09 - Written - James Lane (UofT)
        """
        Etilde = -numpy.atleast_1d(conversion.parse_energy(E, vo=self._vo) / self._psi0)
        # Handle E out of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde < 0, Etilde > 1))[0]
        if len(Etilde_out) > 0:
            # Set to dummy and 0 later, prevents functions throwing errors?
            Etilde[Etilde_out] = 0.5
        sqrtEtilde = numpy.sqrt(Etilde)
        fE = (
            self._fEnorm
            * sqrtEtilde
            / (1 - Etilde) ** 2.0
            * (
                (1.0 - 2.0 * Etilde) * (8.0 * Etilde**2.0 - 8.0 * Etilde - 3.0)
                + (
                    (3.0 * numpy.arcsin(sqrtEtilde))
                    / numpy.sqrt(Etilde * (1.0 - Etilde))
                )
            )
        )
        # Fix out of bounds values
        if len(Etilde_out) > 0:
            fE[Etilde_out] = 0
        return fE.reshape(E.shape)

    def _dMdE(self, E):
        # E already in internal units here
        fE = self.fE(E)
        A = -self._psi0 / E[fE > 0.0]
        out = numpy.zeros_like(E)
        out[fE > 0.0] = (
            (4.0 * numpy.pi) ** 2.0
            * fE[fE > 0.0]
            * self._pot.a**3.0
            * numpy.sqrt(-2.0 * E[fE > 0.0])
            * (
                numpy.sqrt(A - 1.0) * (0.125 * A**2.0 - 5.0 / 12.0 * A - 1.0 / 3.0)
                + 0.125 * A * (A**2.0 - 4.0 * A + 8.0) * numpy.arccos(A**-0.5)
            )
        )
        return out

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        return self._pot.a * numpy.sqrt(ms) / (1 - numpy.sqrt(ms))
