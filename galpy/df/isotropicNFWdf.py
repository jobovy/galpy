# Class that implements isotropic spherical NFW DF
import numpy

from ..potential import NFWPotential
from ..util import conversion
from .sphericaldf import isotropicsphericaldf

# Coefficients of the improved analytical approximation that JB made
_COEFFS = numpy.array(
    [
        7.8480631889123114,
        -41.0268009529575863,
        92.5144063082258157,
        -117.6477872907975382,
        92.6397009471828170,
        -46.6587221550257851,
        14.9776586391246376,
        -2.9784827749197880,
        0.2583468299241013,
        0.0232272797489981,
        0.0926081086527954,
    ]
)


class isotropicNFWdf(isotropicsphericaldf):
    """Class that implements the approximate isotropic spherical NFW DF (either `Widrow 2000 <https://ui.adsabs.harvard.edu/abs/2000ApJS..131...39W/abstract>`__ or an improved fit by Lane et al. 2021)."""

    def __init__(self, pot=None, widrow=False, rmax=1e4, ro=None, vo=None):
        """
        Initialize an isotropic NFW distribution function

        Parameters
        ----------
        pot : NFWPotential instance
            NFW Potential instance
        widrow : bool, optional
            If True, use the approximate form from Widrow (2000), otherwise use improved fit that has <~1e-5 relative density errors
        rmax : float or Quantity, optional
            Maximum radius to consider; set to numpy.inf to evaluate NFW w/o cut-off
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2021-02-01 - Written - Bovy (UofT)

        """
        assert isinstance(pot, NFWPotential), "pot= must be potential.NFWPotential"
        isotropicsphericaldf.__init__(self, pot=pot, rmax=rmax, ro=ro, vo=vo)
        self._Etildemax = pot._amp / pot.a
        self._fEnorm = (
            (9.1968e-2) ** widrow / (4.0 * numpy.pi) / pot.a**1.5 / pot._amp**0.5
        )
        self._widrow = widrow
        self._Etildemin = -pot(self._rmax, 0, use_physical=False) / self._Etildemax

    def fE(self, E):
        """
        Calculate the energy portion of an isotropic NFW distribution function

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
        - 2021-02-01 - Written - Bovy (UofT)
        """
        Etilde = -conversion.parse_energy(E, vo=self._vo) / self._Etildemax
        out = numpy.zeros_like(Etilde)
        indx = (Etilde > self._Etildemin) * (Etilde <= 1.0)
        if self._widrow:
            out[indx] = (
                self._fEnorm
                * Etilde[indx] ** 1.5
                * (1 - Etilde[indx]) ** -2.5
                * (-numpy.log(Etilde[indx]) / (1.0 - Etilde[indx])) ** -2.7419
                * numpy.exp(
                    0.3620 * Etilde[indx]
                    - 0.5639 * Etilde[indx] ** 2.0
                    - 0.0859 * Etilde[indx] ** 3.0
                    - 0.4912 * Etilde[indx] ** 4.0
                )
            )
        else:
            out[indx] = (
                self._fEnorm
                * Etilde[indx] ** 1.5
                * (1 - Etilde[indx]) ** -2.5
                * (-numpy.log(Etilde[indx]) / (1.0 - Etilde[indx])) ** -2.75
                * numpy.polyval(_COEFFS, Etilde[indx])
            )
        return out
