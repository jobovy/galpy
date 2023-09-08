###############################################################################
#   KingPotential.py: Potential of a King profile
###############################################################################
import numpy

from ..util import conversion
from .Force import Force
from .interpSphericalPotential import interpSphericalPotential


class KingPotential(interpSphericalPotential):
    """Potential of a King profile, defined from the distribution function

    .. math::

      f(\\mathcal{E}) = \\begin{cases} \\rho_1\\,(2\\pi\\sigma^2)^{-3/2}\\,\\left(e^{\\mathcal{E}/\\sigma^2}-1\\right), & \\mathcal{E} > 0\\\\0, & \\mathcal{E} \\leq 0\\end{cases}

    where :math:`\\mathcal{E}` is the binding energy. See also :ref:`King DF <king_df_api>`.
    """

    def __init__(self, W0=2.0, M=3.0, rt=1.5, npt=1001, _sfkdf=None, ro=None, vo=None):
        """
        Initialize a King potential

        Parameters
        ----------
        W0 : float, optional
            Dimensionless central potential W0 = Psi(0)/sigma^2 (in practice, needs to be <~ 200, where the DF is essentially isothermal). Default: 2.
        M : float or Quantity, optional
            Total mass. Default: 1.
        rt : float or Quantity, optional
            Tidal radius. Default: 1.
        npt : int, optional
            Number of points to use to solve for Psi(r) when solving the King DF. Default: 1001.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2020-07-11 - Written - Bovy (UofT)

        """
        # Initialize with Force just to parse (ro,vo)
        Force.__init__(self, ro=ro, vo=vo)
        newM = conversion.parse_mass(M, ro=self._ro, vo=self._vo)
        if newM != M:
            self.turn_physical_on(ro=self._ro, vo=self._vo)
        M = newM
        rt = conversion.parse_length(rt, ro=self._ro)
        # Set up King DF
        if _sfkdf is None:
            from ..df.kingdf import _scalefreekingdf

            sfkdf = _scalefreekingdf(W0)
            sfkdf.solve(npt)
        else:
            sfkdf = _sfkdf
        mass_scale = M / sfkdf.mass
        radius_scale = rt / sfkdf.rt
        # Remember whether to turn units on
        ro = self._ro if self._roSet else ro
        vo = self._vo if self._voSet else vo
        interpSphericalPotential.__init__(
            self,
            rforce=lambda r: mass_scale
            / radius_scale**2.0
            * numpy.interp(r / radius_scale, sfkdf._r, sfkdf._dWdr),
            rgrid=sfkdf._r * radius_scale,
            Phi0=-W0 * mass_scale / radius_scale,
            ro=ro,
            vo=vo,
        )
