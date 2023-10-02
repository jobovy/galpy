# Class that implements the anisotropic spherical Hernquist DF with radially
# varying anisotropy of the Osipkov-Merritt type
import numpy

from ..potential import HernquistPotential, evaluatePotentials
from ..util import conversion
from .osipkovmerrittdf import _osipkovmerrittdf


class osipkovmerrittHernquistdf(_osipkovmerrittdf):
    """Class that implements the anisotropic spherical Hernquist DF with radially varying anisotropy of the Osipkov-Merritt type

    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anistropy radius.

    """

    def __init__(self, pot=None, ra=1.4, ro=None, vo=None):
        """
        Initialize a Hernquist DF with Osipkov-Merritt anisotropy

        Parameters
        ----------
        pot : potential.HernquistPotential
            Hernquist potential which determines the DF
        ra : float or Quantity, optional
            Anisotropy radius
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-11-12 - Written - Bovy (UofT)
        """
        assert isinstance(
            pot, HernquistPotential
        ), "pot= must be potential.HernquistPotential"
        _osipkovmerrittdf.__init__(self, pot=pot, ra=ra, ro=ro, vo=vo)
        self._psi0 = -evaluatePotentials(self._pot, 0, 0, use_physical=False)
        self._GMa = self._psi0 * self._pot.a**2.0
        self._a2overra2 = self._pot.a**2.0 / self._ra2
        # First factor is the mass to make the DF that of the mass density
        self._fQnorm = (
            self._psi0
            * self._pot.a
            / numpy.sqrt(2.0)
            / (2 * numpy.pi) ** 3
            / self._GMa**1.5
        )

    def fQ(self, Q):
        """
        Calculate the f(Q) portion of an Osipkov-Merritt Hernquist distribution function

        Parameters
        ----------
        Q : float or numpy.ndarray
            The Osipkov-Merritt 'energy' E-L^2/[2ra^2] (can be Quantity)

        Returns
        -------
        float or numpy.ndarray
            The value of the f(Q) portion of the DF

        Notes
        -----
        - 2020-11-12 - Written - Bovy (UofT)
        """
        Qtilde = numpy.atleast_1d(conversion.parse_energy(Q, vo=self._vo) / self._psi0)
        # Handle potential Q outside of bounds
        Qtilde_out = numpy.where(numpy.logical_or(Qtilde < 0.0, Qtilde > 1.0))[0]
        if len(Qtilde_out) > 0:
            # Dummy variable now and 0 later, prevents numerical issues
            Qtilde[Qtilde_out] = 0.5
        sqrtQtilde = numpy.sqrt(Qtilde)
        # The 'ergodic' part
        fQ = (
            sqrtQtilde
            / (1.0 - Qtilde) ** 2.0
            * (
                (1.0 - 2.0 * Qtilde) * (8.0 * Qtilde**2.0 - 8.0 * Qtilde - 3.0)
                + (
                    (3.0 * numpy.arcsin(sqrtQtilde))
                    / numpy.sqrt(Qtilde * (1.0 - Qtilde))
                )
            )
        )
        # The other part
        fQ += 8.0 * self._a2overra2 * sqrtQtilde * (1.0 - 2.0 * Qtilde)
        if len(Qtilde_out) > 0:
            fQ[Qtilde_out] = 0.0
        return self._fQnorm * fQ.reshape(Q.shape)

    def _icmf(self, ms):
        """Analytic expression for the normalized inverse cumulative mass
        function. The argument ms is normalized mass fraction [0,1]"""
        return self._pot.a * numpy.sqrt(ms) / (1 - numpy.sqrt(ms))
