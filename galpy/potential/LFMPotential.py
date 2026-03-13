###############################################################################
#   LFMPotential.py: modified-gravity potential using the Lattice Field
#                     Medium Radial Acceleration Relation
#
#   g_obs = sqrt(g_bar^2 + g_bar * a0)
#
#   where g_bar is the baryonic acceleration and a0 = c*H0/(2*pi)
###############################################################################
import numpy

from scipy import integrate

from ..util import conversion
from ..util._optional_deps import _APY_LOADED

if _APY_LOADED:
    from astropy import units

from .Potential import (
    Potential,
)


# Physical constants for a0 derivation (SI)
_C_KMS = 299792.458  # speed of light [km/s]
_H0_DEFAULT = 67.4  # Hubble constant [km/s/Mpc]
_MPC_TO_KPC = 1000.0  # 1 Mpc = 1000 kpc
_KPC_TO_M = 3.0857e19  # 1 kpc [m]
_KMS_TO_MS = 1e3  # 1 km/s [m/s]


def _a0_natural(ro_kpc, vo_kms, H0=_H0_DEFAULT):
    """
    Compute the LFM acceleration scale in galpy natural units.

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
        a0 in natural units (vo^2 / ro).
    """
    # a0 = c * H0 / (2*pi) in m/s^2
    a0_si = _C_KMS * _KMS_TO_MS * H0 * _KMS_TO_MS / (_MPC_TO_KPC * _KPC_TO_M) / (
        2.0 * numpy.pi
    )
    # Convert to natural units: a0_nat = a0_si / (vo_ms^2 / ro_m)
    vo_ms = vo_kms * _KMS_TO_MS
    ro_m = ro_kpc * _KPC_TO_M
    return a0_si * ro_m / vo_ms**2


class LFMPotential(Potential):
    """Class that implements modified gravity using the Lattice Field Medium
    Radial Acceleration Relation [1]_.

    .. math::

        g_\\mathrm{obs} = \\sqrt{g_\\mathrm{bar}^2 + g_\\mathrm{bar}\\,a_0}

    where :math:`g_\\mathrm{bar}` is the baryonic gravitational acceleration
    magnitude and :math:`a_0 = c H_0 / (2\\pi)` is the LFM fundamental
    acceleration scale (derived, not a free parameter).

    The force direction is preserved; only the magnitude is boosted. The
    potential energy is computed via numerical quadrature (the deep-field force
    decays as :math:`1/r`, so the potential diverges logarithmically; a finite
    integration cutoff ``rmax`` is used).

    .. [1] Partin (2026), LFM-PAPER-045. Zenodo. https://zenodo.org/records/14838179

    .. [2] McGaugh, Lelli & Schombert (2016), Physical Review Letters, 117, 201101. ADS: https://ui.adsabs.harvard.edu/abs/2016PhRvL.117t1101M
    """

    normalize = property()  # turn off normalize (no intrinsic mass)

    def __init__(self, pot, a0=None, H0=_H0_DEFAULT, rmax=1000.0, ro=None, vo=None):
        """
        Initialize an LFMPotential.

        Parameters
        ----------
        pot : Potential instance or list of Potential instances
            The baryonic gravitational potential(s) to be boosted.
        a0 : float or Quantity, optional
            Acceleration scale. If a float, assumed to be in natural units
            (vo^2/ro). If an astropy Quantity with acceleration dimensions,
            converted automatically. Default: c*H0/(2*pi) using ``H0`` and
            the potential's unit system.
        H0 : float or Quantity, optional
            Hubble constant in km/s/Mpc (only used when ``a0`` is not
            given). Default: 67.4.
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
        - 2026-03-09 - Written - Partin

        References
        ----------
        .. [1] Partin (2026), LFM-PAPER-045. Zenodo. https://zenodo.org/records/14838179

        .. [2] McGaugh, Lelli & Schombert (2016), Physical Review Letters, 117, 201101. ADS: https://ui.adsabs.harvard.edu/abs/2016PhRvL.117t1101M

        Examples
        --------
        >>> from galpy.potential import PlummerPotential, LFMPotential
        >>> bp = PlummerPotential(amp=1., b=0.5)
        >>> lp = LFMPotential(bp)
        >>> lp.Rforce(1., 0.)  # LFM-boosted radial force at R=1, z=0
        """
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)
        # Store the baryonic potential(s)
        if not isinstance(pot, list):
            self._pot = [pot]
        else:
            self._pot = pot
        # Acceleration scale
        if a0 is not None:
            if _APY_LOADED and isinstance(a0, units.Quantity):
                # Convert from physical acceleration to natural units
                a0_si = a0.to(units.m / units.s**2).value
                vo_ms = self._vo * _KMS_TO_MS
                ro_m = self._ro * _KPC_TO_M
                self._a0 = a0_si * ro_m / vo_ms**2
            else:
                self._a0 = float(a0)
        else:
            if _APY_LOADED and isinstance(H0, units.Quantity):
                H0 = H0.to(units.km / units.s / units.Mpc).value
            self._a0 = _a0_natural(self._ro, self._vo, H0=H0)
        self._rmax = rmax
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        self.isNonAxi = False

    def __repr__(self):
        pot_repr = repr(self._pot) if len(self._pot) > 1 else repr(self._pot[0])
        return (
            f"LFMPotential with internal parameters: "
            f"a0={self._a0:.6g}, rmax={self._rmax}, pot={pot_repr}"
        )

    def _bar_Rforce(self, R, z, phi=0.0, t=0.0):
        """Sum of baryonic R-forces (includes amp of each sub-potential)."""
        return sum(p._Rforce_nodecorator(R, z, phi=phi, t=t) for p in self._pot)

    def _bar_zforce(self, R, z, phi=0.0, t=0.0):
        """Sum of baryonic z-forces (includes amp of each sub-potential)."""
        return sum(p._zforce_nodecorator(R, z, phi=phi, t=t) for p in self._pot)

    def _boost(self, R, z, phi=0.0, t=0.0):
        """
        Compute the LFM boost factor sqrt(1 + a0 / g_bar).

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth.
        t : float, optional
            Time.

        Returns
        -------
        float
            The boost factor.  Returns 1 when g_bar is effectively zero.
        """
        FR = self._bar_Rforce(R, z, phi=phi, t=t)
        Fz = self._bar_zforce(R, z, phi=phi, t=t)
        g_bar = numpy.sqrt(FR * FR + Fz * Fz)
        if isinstance(g_bar, numpy.ndarray):
            out = numpy.ones_like(g_bar)
            mask = g_bar > 1e-20
            out[mask] = numpy.sqrt(1.0 + self._a0 / g_bar[mask])
            return out
        else:
            if g_bar < 1e-20:
                return 1.0
            return numpy.sqrt(1.0 + self._a0 / g_bar)

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        r = numpy.sqrt(R**2.0 + z**2.0)
        if r < 1e-12:
            r = 1e-12
        cos_theta = R / r
        sin_theta = z / r
        rmax = self._rmax

        def integrand(rp):
            Rp = rp * cos_theta
            zp = rp * sin_theta
            # Guard against Rp very close to zero
            Rp = max(Rp, 1e-16)
            FR = self._bar_Rforce(Rp, zp, t=t)
            Fz = self._bar_zforce(Rp, zp, t=t)
            g_bar = numpy.sqrt(FR * FR + Fz * Fz)
            if g_bar < 1e-20:
                return 0.0
            boost = numpy.sqrt(1.0 + self._a0 / g_bar)
            # Project modified force onto radial direction
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
            sum(p._amp * p._phitorque(R, z, phi=phi, t=t) for p in self._pot) * boost
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        # Effective (phantom) density from Poisson equation applied to the
        # LFM-modified potential.  Computed numerically via finite differences.
        dr = 1e-4 * max(numpy.sqrt(R**2.0 + z**2.0), 0.01)
        # R-derivatives
        FR_p = self._Rforce(R + dr, z, phi=phi, t=t)
        FR_m = self._Rforce(R - dr, z, phi=phi, t=t) if R > dr else self._Rforce(
            1e-8, z, phi=phi, t=t
        )
        dFRdR = (FR_p - FR_m) / (2.0 * dr) if R > dr else (FR_p - FR_m) / (R + dr - 1e-8)
        # z-derivatives
        Fz_p = self._zforce(R, z + dr, phi=phi, t=t)
        Fz_m = self._zforce(R, z - dr, phi=phi, t=t)
        dFzdz = (Fz_p - Fz_m) / (2.0 * dr)
        # Poisson: 4*pi*rho = -(dFR/dR + FR/R + dFz/dz)
        FR_here = self._Rforce(R, z, phi=phi, t=t)
        R_safe = max(R, 1e-10)
        return -(dFRdR + FR_here / R_safe + dFzdz) / (4.0 * numpy.pi)

    def vcirc(self, R, phi=0.0, t=0.0, use_physical=True):
        """
        Circular velocity at radius R in the midplane.

        This is a convenience method that gives
        :math:`v_c(R) = \\sqrt{R\\,|F_R(R,0)|}`.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        phi : float or Quantity, optional
            Azimuth. Default: 0.
        t : float or Quantity, optional
            Time. Default: 0.
        use_physical : bool, optional
            If True and physical output is on, return Quantity. Default: True.

        Returns
        -------
        float or Quantity
            Circular velocity.
        """
        # Bypass parent's vcirc which evaluates sqrt(R * Rforce) but uses the
        # public interface (with amp and unit decorators).  We go through the
        # public interface ourselves.
        return Potential.vcirc(self, R, phi=phi, t=t, use_physical=use_physical)
