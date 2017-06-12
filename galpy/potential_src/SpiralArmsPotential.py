###############################################################################
#  SpiralArmsPotential.py: class that implements the spiral arms potential
#                           from Cox and Gomez (2002)
#
#  https://arxiv.org/abs/astro-ph/0207635v1
#
#  Phi(r, phi, z) = -4*pi*G*H*rho0*exp(-(r-r0)/Rs)*sum(Cn/(Kn*Dn)*cos(n*gamma)*sech(Kn*z/Bn)^Bn)
#  NOTE: Methods do not take array inputs.
###############################################################################

from __future__ import division

import numpy as np
from galpy.potential_src.Potential import Potential, _APY_LOADED
from galpy.util import bovy_coords, bovy_conversion

if _APY_LOADED:
    from astropy import units


def check_inputs_not_arrays(func):
    """Decorator to check inputs and throw TypeError if any of the inputs are arrays.
    Methods potentially return with silent errors if inputs are not checked."""
    def func_wrapper(self, R, z, phi, t):
        if hasattr(R, '__len__') or hasattr(z, '__len__') or hasattr(phi, '__len__') or hasattr(t, '__len__'):
            raise TypeError('Methods in SpiralArmsPotential do not accept array inputs. Please input scalars.')
        return func(self, R, z, phi, t)

    return func_wrapper


class SpiralArmsPotential(Potential):
    """Class that implements the spiral arms potential from Cox and Gomez (2002). Should be used to modulate an existing
    potential. Unhandled division by zero error if R == 0.
    
    .. math::
    
        \\Phi(r, \\phi, z) = -4 \\pi GH \\rho_0 exp(-\\frac{r-r_0}{R_s}) \\sum(\\frac{C_n}{(K_n D_n} cos(n \\gamma) sech(\\frac{K_n z}{B_n})^B_n)

    """

    def __init__(self, amp=1, ro=None, vo=None, amp_units='density', normalize=False,
                 N=2, alpha=0.2, r_ref=1, phi_ref=0, Rs=0.5, H=0.5, Cs=[1], omega=0):
        """
        NAME:       
            __init__
        PURPOSE:
            initialize a spiral arm potential
        INPUT:
            :param amp: amplitude to be applied to the potential (default: 1); 
                        can be a Quantity with units of density. (amp = 4 * pi * G *rho0)
            :param normalize: if True, normalize such that vc(1.,0.)=1., or, if given as a number
                              such that the force is this fraction of the force necessary to make vc(1.,0.)=1.
            :param ro: distance scales for translation into internal units (default from configuration file)
            :param vo: velocity scales for translation into internal units (default from configuration file)
            :param N: number of spiral arms
            :param alpha: pitch angle of the logarithmic spiral arms in radians (can be Quantity)
            :param r_ref: fiducial radius where rho = rho0 (r_0 in the paper by Cox and Gomez) (can be Quantity)
            :param phi_ref: reference angle (phi_p(r_0) in the paper by Cox and Gomez) (can be Quantity)
            :param Rs: radial scale length of the drop-off in density amplitude of the arms (can be Quantity)
            :param H: scale height of the stellar arm perturbation (can be Quantity)
            :param Cs: list of constants multiplying the cos(n*gamma) term in the mass density expression
            :param omega: rotational speed of the spiral arms (can be Quantity)
        OUTPUT:
            (none)
        HISTORY:
            2017-05-12  Jack Hong (UBC)
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
        self._r_ref = r_ref
        self._phi_ref = phi_ref
        self._Rs = Rs
        self._H = H
        self._Cs = np.array(Cs)
        self._ns = np.arange(1, len(Cs) + 1)
        self._omega = omega
        self._rho0 = amp / (4 * np.pi)

        if normalize or (isinstance(normalize, (int, float)) and not isinstance(normalize, bool)):
            self.normalize(normalize)

        self.isNonAxi = True  # Potential is not axisymmetric

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._Cs/(Ks * Ds) * np.cos(self._ns * self._gamma(R, phi, t)) * 1 / np.cosh(Ks * z / Bs) ** Bs)

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        Cs = self._Cs
        Rs = self._Rs
        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)
        g = self._gamma(R, phi, t)
        dg_dR = self._dgamma_dR(R)

        n = self._ns
        cos_ng = np.cos(n * g)
        sin_ng = np.sin(n * g)
        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs

        return He * np.sum(-Cs * sechzKB_Bs / Ds * (n * dg_dR / Ks * sin_ng
                                                    + z * cos_ng * np.tanh(zKB) * (dKs_dR/Ks - dBs_dR/Bs)
                                                    - dBs_dR / Ks * np.log(sechzKB) * cos_ng
                                                    + dKs_dR / Ks**2 * cos_ng
                                                    + cos_ng * dDs_dR / Ds / Ks)) \
               - He / Rs * np.sum(Cs / Ds / Ks * sechzKB_Bs * cos_ng)

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._Cs/Ds * np.cos(self._ns * self._gamma(R, phi, t))
                        * np.tanh(z * Ks / Bs) * 1 / np.cosh(z * Ks / Bs)**Bs)

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._N * self._ns * self._Cs / Ds / Ks / np.cosh(z*Ks/Bs)**Bs * np.sin(self._ns * g))

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        Cs = self._Cs
        Rs = self._Rs
        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        C_DKs = Cs / (Ds * Ks)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        d2Ks_dR2 = self._d2K_dR2(R)
        d2Bs_dR2 = self._d2B_dR2(R)
        d2Ds_dR2 = self._d2D_dR2(R)

        dKs2_dR = self._dK2_dR(R)
        dBs2_dR = self._dB2_dR(R)
        dDs2_dR = self._dD2_dR(R)

        g = self._gamma(R, phi, t)
        dg_dR = self._dgamma_dR(R)
        d2g_dR2 = self._d2gamma_dR2(R)

        n = self._ns
        ng = n * g

        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = np.log(sechzKB)
        tanhzKB = np.tanh(zKB)
        ztanhzKB = z * tanhzKB
        cos_ng = np.cos(ng)
        sin_ng = np.sin(ng)

        def a():
            """Return the derivative of sech(z*K_n / B_n)**B_n wrt R."""
            return (-(z*dKs_dR / Bs - z * dBs_dR / Bs**2 * Ks) * Bs * tanhzKB + log_sechzKB * dBs_dR) * sechzKB_Bs

        return -He / Rs * (np.sum(Cs * sechzKB_Bs / Ds * (n * dg_dR / Ks * sin_ng
                                                          + ztanhzKB * cos_ng * (dKs_dR/Ks - dBs_dR/Bs)
                                                          - dBs_dR / Ks * log_sechzKB * cos_ng
                                                          + dKs_dR / Ks**2 * cos_ng
                                                          + cos_ng * dDs_dR / Ds / Ks))
                           - Rs * np.sum(Cs * ((a() / Ds - sechzKB_Bs / Ds**2 * dDs_dR) * (n * dg_dR / Ks * sin_ng
                                                                                           + z * cos_ng * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                                                           - dBs_dR / Ks * log_sechzKB * cos_ng
                                                                                           + dKs_dR / Ks**2 * cos_ng
                                                                                           + cos_ng * dDs_dR / Ds / Ks)
                                               + sechzKB_Bs / Ds * (n * (d2g_dR2 / Ks * sin_ng
                                                                         - dg_dR / Ks**2 * dKs_dR * sin_ng
                                                                         + dg_dR / Ks * cos_ng * n * dg_dR)
                                                                    + z * (-sin_ng * n * dg_dR * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                                           + cos_ng * z * (dKs_dR/Bs - dBs_dR/Bs**2 * Ks) * (1-tanhzKB**2) * (dKs_dR/Ks - dBs_dR/Bs)
                                                                           + cos_ng * tanhzKB * (d2Ks_dR2/Ks-(dKs_dR/Ks)**2 - d2Bs_dR2/Bs + (dBs_dR/Bs)**2))
                                                                    - ((d2Bs_dR2/Ks-dBs_dR*dKs_dR/Ks**2) * log_sechzKB * cos_ng
                                                                       - dBs_dR/Ks * z*(dKs_dR/Bs - dBs_dR/Bs**2*Ks)*tanhzKB * cos_ng
                                                                       - dBs_dR/Ks * log_sechzKB * sin_ng*n*dg_dR)
                                                                    + (d2Ks_dR2 / Ks**2 * cos_ng
                                                                       - 2 * dKs_dR**2 / Ks**3 * cos_ng
                                                                       - dKs_dR / Ks**2 * sin_ng * n * dg_dR)
                                                                    + (-sin_ng*n*dg_dR * dDs_dR / Ds / Ks
                                                                       + cos_ng * d2Ds_dR2 / Ds / Ks
                                                                       - cos_ng * (dDs_dR/Ds)**2 / Ks
                                                                       - cos_ng * dDs_dR / Ds / Ks**2 * dKs_dR))))
                           + (np.sum(C_DKs / Rs * sechzKB_Bs * cos_ng
                                     + Cs * (1 / (Ds * Ks)**2 * (dDs_dR * Ks + Ds * dKs_dR) * sechzKB_Bs * cos_ng
                                             - 1 / (Ds * Ks) * a() * cos_ng
                                             + 1 / (Ds * Ks) * sechzKB_Bs * sin_ng * n * dg_dR))))

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._Cs*Ks/Ds * ((1 / Bs) * (np.tanh(z*Ks/Bs)**2 - 1)
                                          + np.tanh(z * Ks / Bs)**2)
                        * np.cos(self._ns * g)
                        * 1 / np.cosh(z * Ks / Bs) ** Bs)

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        return self._H * np.exp(-(R-self._r_ref)/self._Rs) \
               * np.sum(self._Cs * self._N**2. * self._ns**2. / Ds / Ks / np.cosh(z*Ks/Bs)**Bs * np.cos(self._ns*g))

    @check_inputs_not_arrays
    def _Rzderiv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _Rzderiv
        PURPOSE:
            Evaluate the mixed (cylindrical) radial and vertical derivative of the potential (d^2 potential / d R d z).
        INPUT:
            :param R: galactocentric cylindrical radius (must be scalar, not array)
            :param z: vertical height (must be scalar, not array)
            :param phi: azimuth (must be scalar, not array)
            :param t: time (must be scalar, not array)
        OUTPUT:
            :return: d^2 potential / d R d z
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        R = max(1e-8, R)
        phi = phi - self._omega * t

        n = self._ns
        Cs = self._Cs
        Rs = self._Rs
        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)

        g = self._gamma(R, phi, t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = np.cos(n * g)
        sin_ng = np.sin(n * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = np.log(sechzKB)
        tanhzKB = np.tanh(zKB)
        ztanhzKB = z * tanhzKB

        return - He * np.sum(Cs * Ks * tanhzKB * sechzKB_Bs / Ds * (n * dg_dR / Ks * sin_ng
                                                                    + z * cos_ng * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                                    - dBs_dR / Ks * log_sechzKB * cos_ng
                                                                    + dKs_dR / Ks**2 * cos_ng
                                                                    + cos_ng * dDs_dR / Ds / Ks)
                             - Cs * sechzKB_Bs / Ds * np.sum(zKB * cos_ng * (dKs_dR/Ks - dBs_dR/Bs) * (1 - tanhzKB**2)
                                                             + cos_ng * tanhzKB * (dKs_dR/Ks - dBs_dR/Bs)
                                                             + dBs_dR * cos_ng / Bs * tanhzKB)) \
               - He / Rs * np.sum(Cs / Ds * tanhzKB * sechzKB_Bs * cos_ng)

    @check_inputs_not_arrays
    def _Rphideriv(self, R, z, phi=0,t=0):
        """
        NAME:
            _Rphideriv
        PURPOSE:
            Return the mixed radial and azimuthal derivative of the potential in cylindrical coordinates
             (d^2 potential / d R d phi
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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        Cs = self._Cs
        Rs = self._Rs
        He = self._H * np.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(R)
        dDs_dR = self._dD_dR(R)
        g = self._gamma(R, phi, t)
        dg_dR = self._dgamma_dR(R)

        n = self._ns
        N = self._N
        cos_ng = np.cos(n * g)
        sin_ng = np.sin(n * g)
        zKB = z * Ks / Bs
        sechzKB = 1 / np.cosh(zKB)
        sechzKB_Bs = sechzKB ** Bs

        return He * np.sum(Cs * sechzKB_Bs / Ds * (n * dg_dR / Ks * cos_ng * n * N
                                                   - z * sin_ng * n * N * np.tanh(zKB) * (dKs_dR / Ks - dBs_dR / Bs)
                                                   + dBs_dR / Ks * np.log(sechzKB) * sin_ng * n * N
                                                   - dKs_dR / Ks**2 * sin_ng * n * N
                                                   - sin_ng * n * N * dDs_dR / Ds / Ks)) \
               - He / Rs * np.sum(Cs / Ds / Ks * sechzKB_Bs * sin_ng * n * N)

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
        R = max(1e-8, R)
        phi = phi - self._omega * t

        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(R)
        Ds = self._D(R)
        E = self._E(R, z, Ks, Bs, Ds)
        rE = self._rE(R, z, Ks, Bs, Ds)  # actually rE' from the paper

        return np.sum(self._Cs * self._rho0 * (self._H / (Ds * R)) * np.exp(-(R - self._r_ref) / self._Rs)
                      * (1 / np.cosh(Ks * z / Bs))**Bs
                      * (((Ks * R * (Bs + 1) / Bs * (1 / np.cosh(Ks * z / Bs))**2)
                          - 1 / Ks / R * (E**2 + rE)) * np.cos(self._ns * g)
                         - 2 * E * np.cos(self._alpha) * np.sin(self._ns * g)))

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

    def _gamma(self, R, phi, t):
        """Return gamma. (eqn 3 in the paper)"""
        return self._N * (phi - self._phi_ref - np.log(R / self._r_ref) / np.tan(self._alpha))

    def _dgamma_dR(self, R):
        """Return the first derivative of gamma wrt R."""
        return -self._N / R / np.tan(self._alpha)

    def _d2gamma_dR2(self, R):
        """Return the second derivative of gamma wrt R."""
        return self._N / R**2 / np.tan(self._alpha)

    def _K(self, R):
        """Return numpy array from K1 up to and including Kn. (eqn. 5)"""
        return self._ns * self._N / R / np.sin(self._alpha)

    def _dK_dR(self, R):
        """Return numpy array of dK/dR from K1 up to and including Kn."""
        return -self._ns * self._N / R**2 / np.sin(self._alpha)

    def _d2K_dR2(self, R):
        """Return numpy array of the second derivative of K wrt R."""
        return 2 * self._N * self._ns / R**3 / np.sin(self._alpha)

    def _dK2_dR(self, R):
        """Return numpy array of the first derivative of K**2 wrt R."""
        return - 2 * self._N**2 * self._ns**2 / R**3 / np.sin(self._alpha)**2

    def _B(self, R):
        """Return numpy array from B1 up to and including Bn. (eqn. 6)"""
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        return HNn / R / sin_a * (0.4 * HNn / R / sin_a + 1)

    def _dB_dR(self, R):
        """Return numpy array of constants from """
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        return -HNn / R**3 / sin_a**2 * (0.8 * HNn + R * sin_a)

    def _d2B_dR2(self, R):
        """Return numpy array of the second derivative of B wrt R."""
        return self._H * self._N * self._ns / R**3 / np.sin(self._alpha) \
               * (2.4 * self._H * self._N * self._ns / R / np.sin(self._alpha) + 2)

    def _dB2_dR(self, R):
        """Return numpy array of the first derivative of B**2 wrt R."""
        HNn_Rsina = self._H * self._N * self._ns / R / np.sin(self._alpha)

        return -(0.4 * HNn_Rsina + 1) / R * (0.8 * HNn_Rsina**3 + 2 * HNn_Rsina**2 * (0.4 * HNn_Rsina + 1))

    def _D(self, R):
        """Return numpy array from D1 up to and including Dn. (eqn. 7)"""
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        return (0.3 * HNn**2 / sin_a + HNn * R + R**2 * sin_a) / R / (0.3 * HNn + 1 * R * sin_a)

    def _dD_dR(self, R):
        """Return numpy array of dD/dR from D1 up to and including Dn."""
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        HNn_Rsina = HNn / R / sin_a
        return 0.3 * HNn * (HNn_Rsina + 0.3 * HNn_Rsina**2. + 1) / R**2 / (0.3 * HNn_Rsina + 1)**2 / sin_a \
               - ((HNn_Rsina * (1/R + 0.6 / R * HNn_Rsina)) / (0.3 * HNn_Rsina + 1))

    def _d2D_dR2(self, R):
        """Return numpy array of the second derivative of D wrt R."""
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        HNn_Rsina = HNn / R / sin_a
        HNn_Rsina_2 = HNn_Rsina**2
        x = R * (0.3 * HNn_Rsina + 1) * sin_a

        return sin_a / R / x * (0.18 * HNn**2 * (HNn_Rsina + 0.3 * HNn_Rsina_2 + 1) / x**2
                                + 2 * HNn_Rsina
                                - 0.6 * HNn * (HNn_Rsina + 0.6*HNn_Rsina_2) / x
                                - 0.6 * HNn * (HNn_Rsina + 0.3 * HNn_Rsina_2 + 1) / x
                                + 1.8 * HNn_Rsina_2)

    def _dD2_dR(self, R):
        """Return numpy array of the first derivative of D**2 wrt R."""
        HNn = self._H * self._N * self._ns
        sin_a = np.sin(self._alpha)
        HNn_Rsina = HNn / R / sin_a

        return 0.6 * HNn * (HNn_Rsina + 0.3 * HNn_Rsina**2 + 1)**2 / R**2 / (0.3 * HNn_Rsina + 1)**3 / sin_a \
               - 1 / (0.3 * HNn_Rsina + 1)**2 * (HNn_Rsina / R * (2 + 1.2 * HNn_Rsina)) \
                 * (HNn_Rsina + 0.3 * HNn_Rsina**2 + 1)

    def _E(self, R, z, Ks, Bs, Ds):
        """Return numpy of E as defined in the paper."""
        return 1 + Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H)**2) - R / self._Rs \
               - (Ks * self._H) * (1 + 0.8 * Ks * self._H) * np.log(1 / np.cosh(Ks*z/Bs)) \
               - 0.4 * (Ks * self._H)**2 * (Ks * z / Bs) * np.tanh(Ks * z / Bs)

    def _rE(self, R, z, Ks, Bs, Ds):
        """Return numpy array of rE' as define in the paper."""
        return -Ks * self._H/Ds * (1 - 0.3 * (1 - 0.3 * Ks * self._H) / (1 + 0.3 * Ks * self._H)**3) \
               + (Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H)**2)) - R / self._Rs \
               + Ks * self._H * (1 + 1.6 * Ks * self._H) * np.log(1 / np.cosh(Ks*z/Bs)) \
               - (0.4 * (Ks * self._H)**2 * (Ks * z / Bs) * 1 / np.cosh(Ks * z / Bs))**2 / Bs \
               + 1.2 * (Ks * self._H)**2 * (Ks * z / Bs) * np.tanh(Ks * z / Bs)
