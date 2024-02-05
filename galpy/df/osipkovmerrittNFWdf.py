# Class that implements the anisotropic spherical NFW DF with radially
# varying anisotropy of the Osipkov-Merritt type
import numpy

from ..potential import NFWPotential
from ..util import conversion
from .isotropicNFWdf import isotropicNFWdf
from .osipkovmerrittdf import _osipkovmerrittdf

_COEFFS = numpy.array(
    [
        -0.9958957901383353,
        4.2905266124525259,
        -7.6069046709185919,
        7.0313234865878806,
        -3.6920719890718035,
        0.8313023634615980,
        -0.2179687331774083,
        -0.0408426627412238,
        0.0802975743915827,
    ]
)


class osipkovmerrittNFWdf(_osipkovmerrittdf):
    """Class that implements the anisotropic spherical NFW DF with radially varying anisotropy of the Osipkov-Merritt type

    .. math::

        \\beta(r) = \\frac{1}{1+r_a^2/r^2}

    with :math:`r_a` the anistropy radius.

    """

    def __init__(self, pot=None, ra=1.4, rmax=1e4, ro=None, vo=None):
        """
        Initialize a NFW DF with Osipkov-Merritt anisotropy

        Parameters
        ----------
        pot : potential.NFWPotential
            NFW potential which determines the DF
        ra : float or Quantity, optional
            Anisotropy radius
        rmax : float or Quantity, optional
            Maximum radius to consider (set to numpy.inf to evaluate NFW w/o cut-off)
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-11-12 - Written - Bovy (UofT)
        """
        assert isinstance(pot, NFWPotential), "pot= must be potential.NFWPotential"
        _osipkovmerrittdf.__init__(self, pot=pot, ra=ra, rmax=rmax, ro=ro, vo=vo)
        self._Qtildemax = pot._amp / pot.a
        self._Qtildemin = -pot(self._rmax, 0, use_physical=False) / self._Qtildemax
        self._a2overra2 = self._pot.a**2.0 / self._ra2
        self._fQnorm = self._a2overra2 / (4.0 * numpy.pi) / pot.a**1.5 / pot._amp**0.5
        # Initialize isotropic version to use as part of fQ
        self._idf = isotropicNFWdf(pot=pot, rmax=rmax, ro=ro, vo=vo)

    def fQ(self, Q):
        """
        Calculate the f(Q) portion of an Osipkov-Merritt NFW distribution function

        Parameters
        ----------
        Q : float or Quantity
            The Osipkov-Merritt 'energy' E-L^2/[2ra^2]

        Returns
        -------
        ndarray
            The value of the f(Q) portion of the DF

        Notes
        -----
        - 2021-02-09 - Written - Bovy (UofT)

        """
        Qtilde = conversion.parse_energy(Q, vo=self._vo) / self._Qtildemax
        out = numpy.zeros_like(Qtilde)
        indx = (Qtilde > self._Qtildemin) * (Qtilde <= 1.0)
        # The 'ergodic' part
        out[indx] = self._idf.fE(-Q[indx])
        # The other part
        out[indx] += (
            self._fQnorm
            * numpy.polyval(_COEFFS, Qtilde[indx])
            / (
                Qtilde[indx] ** (2.0 / 3.0)
                * (numpy.log(Qtilde[indx]) / (1 - Qtilde[indx])) ** 2.0
            )
        )
        return out
