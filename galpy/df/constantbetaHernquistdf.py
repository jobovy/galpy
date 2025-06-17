# Class that implements the anisotropic spherical Hernquist DF with constant
# beta parameter
import numpy
import scipy.integrate
import scipy.special

from ..potential import HernquistPotential, evaluatePotentials
from ..util import conversion
from .constantbetadf import _constantbetadf


class constantbetaHernquistdf(_constantbetadf):
    """Class that implements the anisotropic spherical Hernquist DF with constant beta parameter"""

    def __init__(self, pot=None, beta=0, ro=None, vo=None):
        """
        Initialize a Hernquist DF with constant anisotropy.

        Parameters
        ----------
        pot : HernquistPotential
            Hernquist potential which determines the DF.
        beta : float
            Anisotropy parameter.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-22 - Written - Lane (UofT)
        """
        assert isinstance(
            pot, HernquistPotential
        ), "pot= must be potential.HernquistPotential"
        _constantbetadf.__init__(self, pot=pot, beta=beta, ro=ro, vo=vo)
        self._psi0 = -evaluatePotentials(self._pot, 0, 0, use_physical=False)
        self._potInf = 0.0
        self._GMa = self._psi0 * self._pot.a**2.0
        # Final factor is mass to make the DF that of the mass density
        self._fEnorm = (
            (2.0**self._beta / (2.0 * numpy.pi) ** 2.5)
            * scipy.special.gamma(5.0 - 2.0 * self._beta)
            / scipy.special.gamma(1.0 - self._beta)
            / scipy.special.gamma(3.5 - self._beta)
            / self._GMa ** (1.5 - self._beta)
            * self._psi0
            * self._pot.a
        )

    def fE(self, E):
        """
        Calculate the energy portion of a Hernquist distribution function

        Parameters
        ----------
        E : float, numpy.ndarray, or Quantity
            The energy.

        Returns
        -------
        float or numpy.ndarray
            The value of the energy portion of the DF

        Notes
        -----
        - 2020-07-22 - Written

        """
        Etilde = -numpy.atleast_1d(conversion.parse_energy(E, vo=self._vo) / self._psi0)
        # Handle potential E outside of bounds
        Etilde_out = numpy.where(numpy.logical_or(Etilde < 0, Etilde > 1))[0]
        if len(Etilde_out) > 0:
            # Dummy variable now and 0 later, prevents numerical issues?
            Etilde[Etilde_out] = 0.5
        # First check algebraic solutions, all adjusted such that DF = mass den
        if self._beta == 0.0:  # isotropic case
            sqrtEtilde = numpy.sqrt(Etilde)
            fE = (
                self._psi0
                * self._pot.a
                / numpy.sqrt(2.0)
                / (2 * numpy.pi) ** 3
                / self._GMa**1.5
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
        elif self._beta == 0.5:
            fE = (3.0 * Etilde**2.0) / (4.0 * numpy.pi**3.0 * self._pot.a)
        elif self._beta == -0.5:
            fE = (
                (20.0 * Etilde**3.0 - 20.0 * Etilde**4.0 + 6.0 * Etilde**5.0)
                / (1.0 - Etilde) ** 4
            ) / (4.0 * numpy.pi**3.0 * self._GMa * self._pot.a)
        else:
            fE = (
                self._fEnorm
                * numpy.power(Etilde, 2.5 - self._beta)
                * scipy.special.hyp2f1(
                    5.0 - 2.0 * self._beta,
                    1.0 - 2.0 * self._beta,
                    3.5 - self._beta,
                    Etilde,
                )
            )
        if len(Etilde_out) > 0:
            fE[Etilde_out] = 0.0
        return fE.reshape(E.shape)

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        return self._pot.a * numpy.sqrt(ms) / (1 - numpy.sqrt(ms))
