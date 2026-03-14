###############################################################################
#   RARPotential.py: wrapper potential implementing the Radial Acceleration
#                     Relation (RAR) with multiple interpolation methods
#
#   Wraps a baryonic potential and boosts forces to match observed
#   rotation curves in the low-acceleration regime.
###############################################################################
import numpy

from scipy import integrate

from ..util import conversion
from ..util._optional_deps import _APY_LOADED

if _APY_LOADED:
    from astropy import units

from .Potential import Potential


# Physical constants (SI)
_C_KMS = 299792.458  # speed of light [km/s]
_H0_DEFAULT = 67.4  # Hubble constant [km/s/Mpc]
_MPC_TO_KPC = 1000.0
_KPC_TO_M = 3.0857e19
_KMS_TO_MS = 1e3


def _a0_natural(ro_kpc, vo_kms, H0=_H0_DEFAULT):
    """Compute the LFM acceleration scale a0 = c*H0/(2*pi) in natural units.

    Parameters
    ----------
    ro_kpc : float
        Distance scale in kpc.
    vo_kms : float
        Velocity scale in km/s.
    H0 : float, optional
        Hubble constant in km/s/Mpc. Default: 67.4.

    Returns
    -------
    float
        a0 in natural units (vo^2/ro).
    """
    a0_si = (
        _C_KMS * _KMS_TO_MS * H0 * _KMS_TO_MS / (_MPC_TO_KPC * _KPC_TO_M)
        / (2.0 * numpy.pi)
    )
    vo_ms = vo_kms * _KMS_TO_MS
    ro_m = ro_kpc * _KPC_TO_M
    return a0_si * ro_m / vo_ms**2


class RARPotential(Potential):
    """Class that implements modified gravity using the Radial Acceleration
    Relation (RAR) with multiple interpolation functions.

    .. math::

        g_\\mathrm{obs} = g_\\mathrm{bar}\\,\\nu(g_\\mathrm{bar}/a_0)

    where :math:`g_\\mathrm{bar}` is the baryonic gravitational acceleration
    and :math:`\\nu(y)` is the interpolation function. Four methods are
    available:

    - ``'simple'``: :math:`\\nu(y) = \\sqrt{1 + 1/y}` [1]_
    - ``'standard'``: :math:`\\nu(y) = \\frac{1 + \\sqrt{1 + 4/y}}{2}` [1]_
    - ``'exp'``: :math:`\\nu(y) = \\frac{1}{1 - e^{-\\sqrt{y}}}` [2]_
    - ``'lfm'``: same as ``'simple'`` but :math:`a_0 = c H_0/(2\\pi)` is
      derived from the Hubble constant with zero free parameters [3]_

    .. [1] Famaey & Binney (2005), MNRAS, 363, 603. ADS: https://ui.adsabs.harvard.edu/abs/2005MNRAS.363..603F

    .. [2] McGaugh, Lelli & Schombert (2016), Physical Review Letters, 117, 201101. ADS: https://ui.adsabs.harvard.edu/abs/2016PhRvL.117t1101M

    .. [3] Partin (2026), LFM-PAPER-045. Zenodo. https://zenodo.org/records/14838179
    """

    normalize = property()  # no intrinsic mass

    def __init__(
        self, pot=None, a0=None, H0=_H0_DEFAULT, method="simple",
        rmax=1000.0, ro=None, vo=None,
    ):
        """
        Initialize a RARPotential.

        Parameters
        ----------
        pot : Potential instance or list of Potential instances, optional
            The baryonic gravitational potential(s) to be boosted.
        a0 : float or Quantity, optional
            Acceleration scale. If a float, assumed to be in natural units
            (vo^2/ro). If an astropy Quantity with acceleration dimensions,
            converted automatically. Default: 1.2e-10 m/s^2 for ``'simple'``,
            ``'standard'``, and ``'exp'``; derived from H0 for ``'lfm'``.
        H0 : float or Quantity, optional
            Hubble constant in km/s/Mpc (used when ``a0`` is not given and
            ``method='lfm'``). Default: 67.4.
        method : str, optional
            Interpolation function to use: ``'simple'``, ``'standard'``,
            ``'exp'``, or ``'lfm'``. Default: ``'simple'``.
        rmax : float, optional
            Integration cutoff (natural units) for potential evaluation.
            Default: 1000.
        ro : float or Quantity, optional
            Distance scale for translation into internal units (default
            from configuration file).
        vo : float or Quantity, optional
            Velocity scale for translation into internal units (default
            from configuration file).

        Notes
        -----
        - 2026-03-13 - Written - Partin

        References
        ----------
        .. [1] Famaey & Binney (2005), MNRAS, 363, 603. ADS: https://ui.adsabs.harvard.edu/abs/2005MNRAS.363..603F

        .. [2] McGaugh, Lelli & Schombert (2016), Physical Review Letters, 117, 201101. ADS: https://ui.adsabs.harvard.edu/abs/2016PhRvL.117t1101M

        .. [3] Partin (2026), LFM-PAPER-045. Zenodo. https://zenodo.org/records/14838179

        Examples
        --------
        >>> from galpy.potential import MiyamotoNagaiPotential, RARPotential
        >>> bp = MiyamotoNagaiPotential(amp=1., a=3./8., b=0.28/8.)
        >>> rp = RARPotential(bp, method='simple')
        >>> rp.Rforce(1., 0.)  # RAR-boosted radial force at R=1, z=0
        """
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
        if pot is None:
            self._pot = []
        elif not isinstance(pot, list):
            self._pot = [pot]
        else:
            self._pot = list(pot)
        # Validate method
        valid_methods = ("simple", "standard", "exp", "lfm")
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}; got {method!r}"
            )
        self._method = method
        # Acceleration scale
        if method == "lfm":
            if a0 is not None:
                raise ValueError(
                    "a0 cannot be set explicitly when method='lfm'; "
                    "it is derived from H0"
                )
            if _APY_LOADED and isinstance(H0, units.Quantity):
                H0 = H0.to(units.km / units.s / units.Mpc).value
            self._a0 = _a0_natural(self._ro, self._vo, H0=H0)
        elif a0 is not None:
            if _APY_LOADED and isinstance(a0, units.Quantity):
                a0_si = a0.to(units.m / units.s**2).value
                vo_ms = self._vo * _KMS_TO_MS
                ro_m = self._ro * _KPC_TO_M
                self._a0 = a0_si * ro_m / vo_ms**2
            else:
                self._a0 = float(a0)
        else:
            # Default a0 = 1.2e-10 m/s^2 (empirical MOND value)
            a0_si = 1.2e-10
            vo_ms = self._vo * _KMS_TO_MS
            ro_m = self._ro * _KPC_TO_M
            self._a0 = a0_si * ro_m / vo_ms**2
        self._rmax = rmax
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        self.isNonAxi = False

    def __repr__(self):
        if len(self._pot) == 0:
            pot_repr = "None"
        elif len(self._pot) == 1:
            pot_repr = repr(self._pot[0])
        else:
            pot_repr = repr(self._pot)
        return (
            f"RARPotential(method={self._method!r}, "
            f"a0={self._a0:.6g}, rmax={self._rmax}, pot={pot_repr})"
        )

    def _nu(self, y):
        """Evaluate the interpolation function nu(y) where y = g_bar / a0.

        Parameters
        ----------
        y : float or ndarray
            Dimensionless baryonic acceleration g_bar / a0.

        Returns
        -------
        float or ndarray
            The interpolation function value.
        """
        if self._method in ("simple", "lfm"):
            return numpy.sqrt(1.0 + 1.0 / y)
        elif self._method == "standard":
            return 0.5 * (1.0 + numpy.sqrt(1.0 + 4.0 / y))
        else:  # exp
            sqrty = numpy.sqrt(y)
            return 1.0 / (1.0 - numpy.exp(-sqrty))

    def _bar_Rforce(self, R, z, phi=0.0, t=0.0):
        """Sum of baryonic R-forces."""
        return sum(p._Rforce_nodecorator(R, z, phi=phi, t=t) for p in self._pot)

    def _bar_zforce(self, R, z, phi=0.0, t=0.0):
        """Sum of baryonic z-forces."""
        return sum(p._zforce_nodecorator(R, z, phi=phi, t=t) for p in self._pot)

    def _boost(self, R, z, phi=0.0, t=0.0):
        """Compute the RAR boost factor nu(g_bar / a0).

        Returns 1 when g_bar is effectively zero.
        """
        FR = self._bar_Rforce(R, z, phi=phi, t=t)
        Fz = self._bar_zforce(R, z, phi=phi, t=t)
        g_bar = numpy.sqrt(FR * FR + Fz * Fz)
        if isinstance(g_bar, numpy.ndarray):
            out = numpy.ones_like(g_bar)
            mask = g_bar > 1e-20
            out[mask] = self._nu(g_bar[mask] / self._a0)
            return out
        else:
            if g_bar < 1e-20:
                return 1.0
            return self._nu(g_bar / self._a0)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if isinstance(R, numpy.ndarray):
            return numpy.array(
                [self._evaluate(Ri, zi, phi=phi, t=t)
                 for Ri, zi in zip(R.flat, numpy.broadcast_to(z, R.shape).flat)]
            ).reshape(R.shape)
        r = numpy.sqrt(R**2.0 + z**2.0)
        r = max(r, 1e-12)
        cos_theta = R / r
        sin_theta = z / r
        rmax = self._rmax

        def integrand(rp):
            Rp = rp * cos_theta
            zp = rp * sin_theta
            Rp = max(Rp, 1e-16)
            FR = self._bar_Rforce(Rp, zp, t=t)
            Fz = self._bar_zforce(Rp, zp, t=t)
            g_bar = numpy.sqrt(FR * FR + Fz * Fz)
            if g_bar < 1e-20:
                return 0.0
            boost = self._nu(g_bar / self._a0)
            Fr = FR * cos_theta + Fz * sin_theta
            return Fr * boost

        result, _ = integrate.quad(integrand, r, rmax, limit=200)
        return result

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        boost = self._boost(R, z, phi=phi, t=t)
        return self._bar_Rforce(R, z, phi=phi, t=t) * boost

    def _zforce(self, R, z, phi=0.0, t=0.0):
        boost = self._boost(R, z, phi=phi, t=t)
        return self._bar_zforce(R, z, phi=phi, t=t) * boost

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        boost = self._boost(R, z, phi=phi, t=t)
        return (
            sum(p._amp * p._phitorque(R, z, phi=phi, t=t) for p in self._pot)
            * boost
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        dr = 1e-4 * numpy.maximum(numpy.sqrt(R**2.0 + z**2.0), 0.01)
        FR_p = self._Rforce(R + dr, z, phi=phi, t=t)
        Rm = numpy.maximum(R - dr, 1e-8)
        FR_m = self._Rforce(Rm, z, phi=phi, t=t)
        dFRdR = (FR_p - FR_m) / (R + dr - Rm)
        Fz_p = self._zforce(R, z + dr, phi=phi, t=t)
        Fz_m = self._zforce(R, z - dr, phi=phi, t=t)
        dFzdz = (Fz_p - Fz_m) / (2.0 * dr)
        FR_here = self._Rforce(R, z, phi=phi, t=t)
        R_safe = numpy.maximum(R, 1e-10)
        return -(dFRdR + FR_here / R_safe + dFzdz) / (4.0 * numpy.pi)
