###############################################################################
#  SpiralArmsPotential.py: class that implements the spiral arms potential
#                           from Cox and Gomez (2002)
#
#  https://arxiv.org/abs/astro-ph/0207635
#
#  Phi(r, phi, z) = -4*pi*G*H*rho0*exp(-(r-r0)/Rs)*sum(Cn/(Kn*Dn)*cos(n*gamma)*sech(Kn*z/Bn)^Bn)
#  NOTE: Methods do not take array inputs.
###############################################################################

from __future__ import division
from functools import wraps
from .Potential import Potential, _APY_LOADED
from galpy.util import bovy_conversion
import numpy as np

if _APY_LOADED:
    from astropy import units


def check_inputs_not_arrays(func):
    """
    Decorator to check inputs and throw TypeError if any of the inputs are arrays.
    Methods potentially return with silent errors if inputs are not checked.
    """
    @wraps(func)
    def func_wrapper(self, R, z, phi, t):
        if hasattr(R, '__len__') or hasattr(z, '__len__') or hasattr(phi, '__len__') or hasattr(t, '__len__'):
            raise TypeError('Methods in SpiralArmsPotential do not accept array inputs. Please input scalars.')
        return func(self, R, z, phi, t)

    return func_wrapper


class SpiralArmsPotential(Potential):
    """Class that implements the spiral arms potential from (`Cox and Gomez 2002 <https://arxiv.org/abs/astro-ph/0207635>`__). Should be used to modulate an existing potential (density is positive in the arms, negative outside).
    
    .. math::
    
        \\Phi(R, \\phi, z) = -4 \\pi GH \\,\\rho_0 exp \\left( -\\frac{R-r_{ref}}{R_s} \\right) \\sum{\\frac{C_n}{K_n D_n} \\,\cos(n \\gamma) \\,\\mathrm{sech}^{B_n} \\left( \\frac{K_n z}{B_n} \\right)}

    where

    .. math::
        K_n &= \\frac{n N}{R \sin(\\alpha)} \\\\
        B_n &= K_n H (1 + 0.4 K_n H) \\\\
        D_n &= \\frac{1 + K_n H + 0.3 (K_n H)^2}{1 + 0.3 K_n H} \\\\

    and

    .. math::
        \\gamma = N \\left[\\phi - \\phi_{ref} - \\frac{\\ln(R/r_{ref})}{\\tan(\\alpha)} \\right]

    The default of :math:`C_n=[1]` gives a sinusoidal profile for the potential. An alternative from `Cox and Gomez (2002) <https://arxiv.org/abs/astro-ph/0207635>`__  creates a density that behaves approximately as a cosine squared in the arms but is separated by a flat interarm region by setting 

     .. math::
        C_n = \\left[\\frac{8}{3 \\pi}\,,\\frac{1}{2} \\,, \\frac{8}{15 \\pi}\\right]

    """
    normalize= property() # turn off normalize
    def __init__(self, amp=1, ro=None, vo=None, amp_units='density',
                 N=2, alpha=0.2, r_ref=1, phi_ref=0, Rs=0.3, H=0.125, omega=0, Cs=[1]):

        """
        NAME:       
            __init__
        PURPOSE:
            initialize a spiral arms potential
        INPUT:
            :amp: amplitude to be applied to the potential (default: 1); 
                        can be a Quantity with units of density. (:math:`amp = 4 \\pi G \\rho_0`)
            :ro: distance scales for translation into internal units (default from configuration file)
            :vo: velocity scales for translation into internal units (default from configuration file)
            :N: number of spiral arms
            :alpha: pitch angle of the logarithmic spiral arms in radians (can be Quantity)
            :r_ref: fiducial radius where :math:`\\rho = \\rho_0` (:math:`r_0` in the paper by Cox and Gomez) (can be Quantity)
            :phi_ref: reference angle (:math:`\\phi_p(r_0)` in the paper by Cox and Gomez) (can be Quantity)
            :Rs: radial scale length of the drop-off in density amplitude of the arms (can be Quantity)
            :H: scale height of the stellar arm perturbation (can be Quantity)
            :Cs: list of constants multiplying the :math:`\cos(n \\gamma)` terms
            :omega: rotational pattern speed of the spiral arms (can be Quantity)
        OUTPUT:
            (none)
        HISTORY:
            Started - 2017-05-12  Jack Hong (UBC)

            Completed - 2017-07-04 Jack Hong (UBC)
        """

        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        if _APY_LOADED:
            if isinstance(alpha, units.Quantity):
                alpha = alpha.to(units.rad).value
            if isinstance(r_ref, units.Quantity):
                r_ref = r_ref.to(units.kpc).value / self._ro
            if isinstance(phi_ref, units.Quantity):
                phi_ref = phi_ref.to(units.rad).value
            if isinstance(Rs, units.Quantity):
                Rs = Rs.to(units.kpc).value / self._ro
            if isinstance(H, units.Quantity):
                H = H.to(units.kpc).value / self._ro
            if isinstance(omega, units.Quantity):
                omega = omega.to(units.km / units.s / units.kpc).value \
                        / bovy_conversion.freq_in_kmskpc(self._vo, self._ro)

        self._N = -N  # trick to flip to left handed coordinate system; flips sign for phi and phi_ref, but also alpha.
        self._alpha = -alpha  # we don't want sign for alpha to change, so flip alpha. (see eqn. 3 in the paper)
        self._sin_alpha = np.sin(-alpha)
        self._tan_alpha = np.tan(-alpha)
        self._r_ref = r_ref
        self._phi_ref = phi_ref
        self._Rs = Rs
        self._H = H
        self._Cs = np.array(Cs)
        self._ns = np.arange(1, len(Cs) + 1)
        self._omega = omega
        self._rho0 = 1 / (4 * np.pi)
        self._HNn = self._H * self._N * self._ns

        self.isNonAxi = True   # Potential is not axisymmetric
        self.hasC = True       # Potential has C implementation to speed up orbit integrations
        self.hasC_dxdv = True  # Potential has C implementation of second derivatives

    @check_inputs_not_arrays
    def _evaluate(self, R, z, phi=0, t=0):
        """
        NAME:
            _evaluate
        PURPOSE:
            Evaluate the potential at the given coordinates. (without the amp factor; handled by super class)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: Phi(R, z, phi, t)
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        return -self._H * np.exp(-(R-self._r_ref) / self._Rs) \
               * np.sum(self._Cs / Ks / Ds * np.cos(self._ns * self._gamma(R, phi - self._omega * t)) / np.cosh(Ks * z / Bs) ** Bs)

    @check_inputs_not_arrays
    def _Rforce(self, R, z, phi=0, t=0):
        """
        NAME:
            _Rforce
        PURPOSE:
            Evaluate the radial force for this potential at the given coordinates. (-dPhi/dR)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the radial force
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """

        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = np.cos(self._ns * g)
        sin_ng = np.sin(self._ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)

        return -He * np.sum(self._Cs * sechzKB**Bs / Ds * ((self._ns * dg_dR / Ks * sin_ng
                                                            + cos_ng * (z * np.tanh(zKB) * (dKs_dR/Ks - dBs_dR/Bs)
                                                                        - dBs_dR / Ks * np.log(sechzKB)
                                                                        + dKs_dR / Ks**2
                                                                        + dDs_dR / Ds / Ks))
                                                           + cos_ng / Ks / self._Rs))

    @check_inputs_not_arrays
    def _zforce(self, R, z, phi=0, t=0):
        """
        NAME:
            _zforce
        PURPOSE:
            Evaluate the vertical force for this potential at the given coordinates. (-dPhi/dz)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the vertical force
        HISTORY:
            2017-05-25  Jack Hong (UBC) 
        """

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        zK_B = z * Ks / Bs

        return -self._H * np.exp(-(R-self._r_ref) / self._Rs) \
               * np.sum(self._Cs / Ds * np.cos(self._ns * self._gamma(R, phi - self._omega * t))
                        * np.tanh(zK_B) / np.cosh(zK_B)**Bs)

    @check_inputs_not_arrays
    def _phiforce(self, R, z, phi=0, t=0):
        """
        NAME:
            _phiforce
        PURPOSE:
            Evaluate the azimuthal force in cylindrical coordinates. (-dPhi/dphi)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the azimuthal force
        HISTORY:
            2017-05-25  Jack Hong (UBC)
        """

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        return -self._H * np.exp(-(R-self._r_ref) / self._Rs) \
               * np.sum(self._N * self._ns * self._Cs / Ds / Ks / np.cosh(z * Ks / Bs)**Bs * np.sin(self._ns * g))

    @check_inputs_not_arrays
    def _R2deriv(self, R, z, phi=0, t=0):
        """
        NAME:
            _R2deriv
        PURPOSE:
            Evaluate the second (cylindrical) radial derivative of the potential.
             (d^2 potential / d R^2)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the second radial derivative
        HISTORY:
            2017-05-31  Jack Hong (UBC)
        """

        Rs = self._Rs
        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        R_sina = R * self._sin_alpha
        HNn_R_sina = self._HNn / R_sina
        HNn_R_sina_2 = HNn_R_sina**2
        x = R * (0.3 * HNn_R_sina + 1) * self._sin_alpha

        d2Ks_dR2 = 2 * self._N * self._ns / R**3 / self._sin_alpha
        d2Bs_dR2 = HNn_R_sina / R**2 * (2.4 * HNn_R_sina + 2)
        d2Ds_dR2 = self._sin_alpha / R / x * (self._HNn* (0.18 * self._HNn * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x**2
                                                          + 2 / R_sina
                                                          - 0.6 * HNn_R_sina * (1 + 0.6 * HNn_R_sina) / x
                                                          - 0.6 * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x
                                                          + 1.8 * self._HNn / R_sina**2))

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)
        d2g_dR2 = self._N / R**2 / self._tan_alpha

        sin_ng = np.sin(self._ns * g)
        cos_ng = np.cos(self._ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = np.log(sechzKB)
        tanhzKB = np.tanh(zKB)
        ztanhzKB = z * tanhzKB

        return -He / Rs * (np.sum(self._Cs * sechzKB_Bs / Ds
                                  * ((self._ns * dg_dR / Ks * sin_ng
                                      + cos_ng * (ztanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                  - dBs_dR / Ks * log_sechzKB
                                                  + dKs_dR / Ks**2
                                                  + dDs_dR / Ds / Ks))
                                     - (Rs * (1 / Ks * ((ztanhzKB * (dBs_dR / Bs * Ks - dKs_dR)
                                                         + log_sechzKB * dBs_dR)
                                                        - dDs_dR / Ds) * (self._ns * dg_dR * sin_ng
                                                                          + cos_ng * (ztanhzKB * Ks * (dKs_dR/Ks - dBs_dR/Bs)
                                                                                      - dBs_dR * log_sechzKB
                                                                                      + dKs_dR / Ks
                                                                                      + dDs_dR / Ds))
                                              + (self._ns * (sin_ng * (d2g_dR2 / Ks - dg_dR / Ks**2 * dKs_dR)
                                                             + dg_dR**2 / Ks * cos_ng * self._ns)
                                                 + z * (-sin_ng * self._ns * dg_dR * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                        + cos_ng * (z * (dKs_dR/Bs - dBs_dR/Bs**2 * Ks) * (1-tanhzKB**2) * (dKs_dR/Ks - dBs_dR/Bs)
                                                                    + tanhzKB * (d2Ks_dR2/Ks-(dKs_dR/Ks)**2 - d2Bs_dR2/Bs + (dBs_dR/Bs)**2)))
                                                 + (cos_ng * (dBs_dR/Ks * ztanhzKB * (dKs_dR/Bs - dBs_dR/Bs**2*Ks)
                                                              -(d2Bs_dR2/Ks-dBs_dR*dKs_dR/Ks**2) * log_sechzKB)
                                                    + dBs_dR/Ks * log_sechzKB * sin_ng * self._ns * dg_dR)
                                                 + ((cos_ng * (d2Ks_dR2 / Ks**2 - 2 * dKs_dR**2 / Ks**3)
                                                     - dKs_dR / Ks**2 * sin_ng * self._ns * dg_dR)
                                                    + (cos_ng * (d2Ds_dR2 / Ds / Ks
                                                                 - (dDs_dR/Ds)**2 / Ks
                                                                 - dDs_dR / Ds / Ks**2 * dKs_dR)
                                                       - sin_ng * self._ns * dg_dR * dDs_dR / Ds / Ks))))
                                        - 1 / Ks * (cos_ng / Rs
                                                    + (cos_ng * ((dDs_dR * Ks + Ds * dKs_dR) / (Ds * Ks)
                                                                 -  (ztanhzKB * (dBs_dR / Bs * Ks - dKs_dR)
                                                                     + log_sechzKB * dBs_dR))
                                                       + sin_ng * self._ns * dg_dR))))))

    @check_inputs_not_arrays
    def _z2deriv(self, R, z, phi=0, t=0):
        """
        NAME:
            _z2deriv
        PURPOSE:
            Evaluate the second (cylindrical) vertical derivative of the potential.
             (d^2 potential / d z^2)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the second vertical derivative
        HISTORY:
            2017-05-26  Jack Hong (UBC) 
        """

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        zKB = z * Ks / Bs
        tanh2_zKB = np.tanh(zKB)**2

        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._Cs * Ks / Ds * ((tanh2_zKB - 1) / Bs + tanh2_zKB) * np.cos(self._ns * g) / np.cosh(zKB)**Bs)

    @check_inputs_not_arrays
    def _phi2deriv(self, R, z, phi=0, t=0):
        """
        NAME:
            _phi2deriv
        PURPOSE:
            Evaluate the second azimuthal derivative of the potential in cylindrical coordinates.
            (d^2 potential / d phi^2)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: d^2 potential / d phi^2
        HISTORY:
            2017-05-29 Jack Hong (UBC)
        """

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        return self._H * np.exp(-(R-self._r_ref) / self._Rs) \
               * np.sum(self._Cs * self._N**2. * self._ns**2. / Ds / Ks / np.cosh(z*Ks/Bs)**Bs * np.cos(self._ns*g))

    @check_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _Rzderiv
        PURPOSE:
            Evaluate the mixed (cylindrical) radial and vertical derivative of the potential
            (d^2 potential / dR dz).
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: d^2 potential / dR dz
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """

        Rs = self._Rs
        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = np.cos(self._ns * g)
        sin_ng = np.sin(self._ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = np.log(sechzKB)
        tanhzKB = np.tanh(zKB)

        return - He * np.sum(sechzKB_Bs * self._Cs / Ds * (Ks * tanhzKB * (self._ns * dg_dR / Ks * sin_ng
                                                                           + cos_ng * (z * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                                                       - dBs_dR / Ks * log_sechzKB
                                                                                       + dKs_dR / Ks**2
                                                                                       + dDs_dR / Ds / Ks))
                                                           - cos_ng * ((zKB * (dKs_dR/Ks - dBs_dR/Bs) * (1 - tanhzKB**2)
                                                                        + tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                                        + dBs_dR / Bs * tanhzKB)
                                                                       - tanhzKB / Rs)))

    @check_inputs_not_arrays
    def _Rphideriv(self, R, z, phi=0,t=0):
        """
        NAME:
            _Rphideriv
        PURPOSE:
            Return the mixed radial and azimuthal derivative of the potential in cylindrical coordinates
             (d^2 potential / dR dphi)
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the mixed radial and azimuthal derivative
        HISTORY:
            2017-06-09  Jack Hong (UBC)
        """

        He = self._H * np.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = np.cos(self._ns * g)
        sin_ng = np.sin(self._ns * g)
        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB ** Bs

        return - He * np.sum(self._Cs * sechzKB_Bs / Ds * self._ns * self._N
                           * (- self._ns * dg_dR / Ks * cos_ng
                              + sin_ng * (z * np.tanh(zKB) * (dKs_dR / Ks - dBs_dR / Bs)
                                          + 1/Ks * (-dBs_dR * np.log(sechzKB)
                                                    + dKs_dR / Ks
                                                    + dDs_dR / Ds
                                                    + 1 / self._Rs))))

    @check_inputs_not_arrays
    def _dens(self, R, z, phi=0, t=0):
        """
        NAME:
            _dens
        PURPOSE:
            Evaluate the density. If not given, the density is computed using the Poisson equation
            from the first and second derivatives of the potential (if all are implemented).
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: the density
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """

        g = self._gamma(R, phi - self._omega * t)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        ng = self._ns * g
        zKB = z * Ks / Bs
        sech_zKB = 1 / np.cosh(zKB)
        tanh_zKB = np.tanh(zKB)
        log_sech_zKB = np.log(sech_zKB)

        # numpy of E as defined in the appendix of the paper.
        E = 1 + Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H) ** 2) - R / self._Rs \
            - (Ks * self._H) * (1 + 0.8 * Ks * self._H) * log_sech_zKB \
            - 0.4 * (Ks * self._H) ** 2 * zKB * tanh_zKB

        # numpy array of rE' as define in the appendix of the paper.
        rE = -Ks * self._H / Ds * (1 - 0.3 * (1 - 0.3 * Ks * self._H) / (1 + 0.3 * Ks * self._H) ** 3) \
             + (Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H) ** 2)) - R / self._Rs \
             + Ks * self._H * (1 + 1.6 * Ks * self._H) * log_sech_zKB \
             - (0.4 * (Ks * self._H) ** 2 * zKB * sech_zKB) ** 2 / Bs \
             + 1.2 * (Ks * self._H) ** 2 * zKB * tanh_zKB

        return np.sum(self._Cs * self._rho0 * (self._H / (Ds * R)) * np.exp(-(R - self._r_ref) / self._Rs)
                      * sech_zKB**Bs * (np.cos(ng) * (Ks * R * (Bs + 1) / Bs * sech_zKB**2
                                                      - 1 / Ks / R * (E**2 + rE))
                                        - 2 * np.sin(ng)* E * np.cos(self._alpha)))

    def OmegaP(self):
        """
        NAME:
            OmegaP
        PURPOSE:
            Return the pattern speed. (used to compute the Jacobi integral for orbits).
        INPUT:
            :param self
        OUTPUT:
            :return: the pattern speed
        HISTORY:
            2017-06-09  Jack Hong (UBC)
        """
        return self._omega

    def _gamma(self, R, phi):
        """Return gamma. (eqn 3 in the paper)"""
        return self._N * (phi - self._phi_ref - np.log(R / self._r_ref) / self._tan_alpha)

    def _dgamma_dR(self, R):
        """Return the first derivative of gamma wrt R."""
        return -self._N / R / self._tan_alpha

    def _K(self, R):
        """Return numpy array from K1 up to and including Kn. (eqn. 5)"""
        return self._ns * self._N / R / self._sin_alpha

    def _dK_dR(self, R):
        """Return numpy array of dK/dR from K1 up to and including Kn."""
        return -self._ns * self._N / R**2 / self._sin_alpha

    def _B(self, R):
        """Return numpy array from B1 up to and including Bn. (eqn. 6)"""
        HNn_R = self._HNn / R

        return HNn_R / self._sin_alpha * (0.4 * HNn_R / self._sin_alpha + 1)

    def _dB_dR(self, R):
        """Return numpy array of dB/dR from B1 up to and including Bn."""
        return -self._HNn / R**3 / self._sin_alpha**2 * (0.8 * self._HNn + R * self._sin_alpha)

    def _D(self, R):
        """Return numpy array from D1 up to and including Dn. (eqn. 7)"""
        return (0.3 * self._HNn**2 / self._sin_alpha / R
                + self._HNn + R * self._sin_alpha) / (0.3 * self._HNn + R * self._sin_alpha)

    def _dD_dR(self, R):
        """Return numpy array of dD/dR from D1 up to and including Dn."""
        HNn_R_sina = self._HNn / R / self._sin_alpha

        return HNn_R_sina * (0.3 * (HNn_R_sina + 0.3 * HNn_R_sina**2. + 1) / R / (0.3 * HNn_R_sina + 1)**2
                             - (1/R * (1 + 0.6 * HNn_R_sina) / (0.3 * HNn_R_sina + 1)))
