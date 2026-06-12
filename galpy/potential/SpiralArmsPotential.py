###############################################################################
#  SpiralArmsPotential.py: class that implements the spiral arms potential
#                           from Cox and Gomez (2002)
#
#  https://arxiv.org/abs/astro-ph/0207635
#
#  Phi(r, phi, z) = -4*pi*G*H*rho0*exp(-(r-r0)/Rs)*sum(Cn/(Kn*Dn)*cos(n*gamma)*sech(Kn*z/Bn)^Bn)
###############################################################################
import math

import numpy

from ..backend import get_namespace
from ..util import conversion
from .Potential import Potential


class SpiralArmsPotential(Potential):
    """Class that implements the spiral arms potential from (`Cox and Gomez 2002 <https://arxiv.org/abs/astro-ph/0207635>`__). Should be used to modulate an existing potential (density is positive in the arms, negative outside; note that because of this, a contour plot of this potential will appear to have twice as many arms, where half are the underdense regions).

    .. math::

        \\Phi(R, \\phi, z) = -4 \\pi GH \\,\\rho_0 exp \\left( -\\frac{R-r_{ref}}{R_s} \\right) \\sum{\\frac{C_n}{K_n D_n} \\,\\cos(n \\gamma) \\,\\mathrm{sech}^{B_n} \\left( \\frac{K_n z}{B_n} \\right)}

    where

    .. math::
        K_n &= \\frac{n N}{R \\sin(\\alpha)} \\\\
        B_n &= K_n H (1 + 0.4 K_n H) \\\\
        D_n &= \\frac{1 + K_n H + 0.3 (K_n H)^2}{1 + 0.3 K_n H} \\\\

    and

    .. math::
        \\gamma = N \\left[\\phi - \\phi_{ref} - \\frac{\\ln(R/r_{ref})}{\\tan(\\alpha)} \\right]

    The default of :math:`C_n=[1]` gives a sinusoidal profile for the potential. An alternative from `Cox and Gomez (2002) <https://arxiv.org/abs/astro-ph/0207635>`__  creates a density that behaves approximately as a cosine squared in the arms but is separated by a flat interarm region by setting

     .. math::
        C_n = \\left[\\frac{8}{3 \\pi}\\,,\\frac{1}{2} \\,, \\frac{8}{15 \\pi}\\right]

    """

    normalize = property()  # turn off normalize

    def __init__(
        self,
        amp=1,
        ro=None,
        vo=None,
        amp_units="density",
        N=2,
        alpha=0.2,
        r_ref=1,
        phi_ref=0,
        Rs=0.3,
        H=0.125,
        omega=0,
        Cs=[1],
    ):
        """
        Initialize a spiral arms potential

        Parameters
        ----------
        amp : float or Quantity, optional
            amplitude to be applied to the potential (default: 1); can be a Quantity with units of density. (:math:`amp = 4 \\pi G \\rho_0`)
        ro : float or Quantity, optional
            distance scales for translation into internal units (default from configuration file)
        vo : float or Quantity, optional
            velocity scales for translation into internal units (default from configuration file)
        N : int, optional
            number of spiral arms.
        alpha : float or Quantity, optional
            pitch angle of the logarithmic spiral arms.
        r_ref : float or Quantity, optional
            fiducial radius where :math:`\\rho = \\rho_0` (:math:`r_0` in the paper by Cox and Gomez).
        phi_ref : float or Quantity, optional
            reference angle (:math:`\\phi_p(r_0)` in the paper by Cox and Gomez).
        Rs : float or Quantity, optional
            radial scale length of the drop-off in density amplitude of the arms.
        H : float or Quantity, optional
            scale height of the stellar arm perturbation.
        Cs : list of floats, optional
            constants multiplying the :math:`\\cos(n \\gamma)` terms.
        omega : float or Quantity, optional
            rotational pattern speed of the spiral arms.

        Notes
        -----
        - 2017-05-12 - Started - Jack Hong (UBC)
        - 2020-03-30 - Re-implemented using Potential - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        alpha = conversion.parse_angle(alpha)
        r_ref = conversion.parse_length(r_ref, ro=self._ro)
        phi_ref = conversion.parse_angle(phi_ref)
        Rs = conversion.parse_length(Rs, ro=self._ro)
        H = conversion.parse_length(H, ro=self._ro)
        omega = conversion.parse_frequency(omega, ro=self._ro, vo=self._vo)
        self._N = -N  # trick to flip to left handed coordinate system; flips sign for phi and phi_ref, but also alpha.
        self._alpha = -alpha  # we don't want sign for alpha to change, so flip alpha. (see eqn. 3 in the paper)
        self._sin_alpha = numpy.sin(-alpha)
        self._tan_alpha = numpy.tan(-alpha)
        self._r_ref = r_ref
        self._phi_ref = phi_ref
        self._Rs = Rs
        self._H = H
        self._Cs = self._Cs0 = numpy.array(Cs)
        self._ns = self._ns0 = numpy.arange(1, len(Cs) + 1)
        self._omega = omega
        self._rho0 = 1 / (4 * numpy.pi)
        self._HNn = self._HNn0 = self._H * self._N * self._ns0

        self.isNonAxi = True  # Potential is not axisymmetric
        self.hasC = (
            True  # Potential has C implementation to speed up orbit integrations
        )
        self.hasC_dxdv = True  # Potential has C implementation of second derivatives
        self.hasC_dxdv3d = True  # Full 3D Hessian (incl. zphideriv) now in C

    def _evaluate(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        return (
            -self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                Cs
                / Ks
                / Ds
                * xp.cos(ns * self._gamma(R, phi - self._omega * t))
                / xp.cosh(Ks * z / Bs) ** Bs,
                axis=0,
            )
        )

    def _Rforce(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        He = self._H * xp.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        dKs_dR = self._dK_dR(R, ns)
        dBs_dR = self._dB_dR(R, HNn)
        dDs_dR = self._dD_dR(R, HNn)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = xp.cos(ns * g)
        sin_ng = xp.sin(ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / xp.cosh(zKB)

        return -He * xp.sum(
            Cs
            * sechzKB**Bs
            / Ds
            * (
                (
                    ns * dg_dR / Ks * sin_ng
                    + cos_ng
                    * (
                        z * xp.tanh(zKB) * (dKs_dR / Ks - dBs_dR / Bs)
                        - dBs_dR / Ks * xp.log(sechzKB)
                        + dKs_dR / Ks**2
                        + dDs_dR / Ds / Ks
                    )
                )
                + cos_ng / Ks / self._Rs
            ),
            axis=0,
        )

    def _zforce(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)
        zK_B = z * Ks / Bs

        return (
            -self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                Cs
                / Ds
                * xp.cos(ns * self._gamma(R, phi - self._omega * t))
                * xp.tanh(zK_B)
                / xp.cosh(zK_B) ** Bs,
                axis=0,
            )
        )

    def _phitorque(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        return (
            -self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                self._N
                * ns
                * Cs
                / Ds
                / Ks
                / xp.cosh(z * Ks / Bs) ** Bs
                * xp.sin(ns * g),
                axis=0,
            )
        )

    def _R2deriv(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        Rs = self._Rs
        He = self._H * xp.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        dKs_dR = self._dK_dR(R, ns)
        dBs_dR = self._dB_dR(R, HNn)
        dDs_dR = self._dD_dR(R, HNn)

        R_sina = R * self._sin_alpha
        HNn_R_sina = HNn / R_sina
        HNn_R_sina_2 = HNn_R_sina**2
        x = R * (0.3 * HNn_R_sina + 1) * self._sin_alpha

        d2Ks_dR2 = 2 * self._N * ns / R**3 / self._sin_alpha
        d2Bs_dR2 = HNn_R_sina / R**2 * (2.4 * HNn_R_sina + 2)
        d2Ds_dR2 = (
            self._sin_alpha
            / R
            / x
            * (
                HNn
                * (
                    0.18 * HNn * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x**2
                    + 2 / R_sina
                    - 0.6 * HNn_R_sina * (1 + 0.6 * HNn_R_sina) / x
                    - 0.6 * (HNn_R_sina + 0.3 * HNn_R_sina_2 + 1) / x
                    + 1.8 * HNn / R_sina**2
                )
            )
        )

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)
        d2g_dR2 = self._N / R**2 / self._tan_alpha

        sin_ng = xp.sin(ns * g)
        cos_ng = xp.cos(ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / xp.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = xp.log(sechzKB)
        tanhzKB = xp.tanh(zKB)
        ztanhzKB = z * tanhzKB

        return (
            -He
            / Rs
            * (
                xp.sum(
                    Cs
                    * sechzKB_Bs
                    / Ds
                    * (
                        (
                            ns * dg_dR / Ks * sin_ng
                            + cos_ng
                            * (
                                ztanhzKB * (dKs_dR / Ks - dBs_dR / Bs)
                                - dBs_dR / Ks * log_sechzKB
                                + dKs_dR / Ks**2
                                + dDs_dR / Ds / Ks
                            )
                        )
                        - (
                            Rs
                            * (
                                1
                                / Ks
                                * (
                                    (
                                        ztanhzKB * (dBs_dR / Bs * Ks - dKs_dR)
                                        + log_sechzKB * dBs_dR
                                    )
                                    - dDs_dR / Ds
                                )
                                * (
                                    ns * dg_dR * sin_ng
                                    + cos_ng
                                    * (
                                        ztanhzKB * Ks * (dKs_dR / Ks - dBs_dR / Bs)
                                        - dBs_dR * log_sechzKB
                                        + dKs_dR / Ks
                                        + dDs_dR / Ds
                                    )
                                )
                                + (
                                    ns
                                    * (
                                        sin_ng * (d2g_dR2 / Ks - dg_dR / Ks**2 * dKs_dR)
                                        + dg_dR**2 / Ks * cos_ng * ns
                                    )
                                    + z
                                    * (
                                        -sin_ng
                                        * ns
                                        * dg_dR
                                        * tanhzKB
                                        * (dKs_dR / Ks - dBs_dR / Bs)
                                        + cos_ng
                                        * (
                                            z
                                            * (dKs_dR / Bs - dBs_dR / Bs**2 * Ks)
                                            * (1 - tanhzKB**2)
                                            * (dKs_dR / Ks - dBs_dR / Bs)
                                            + tanhzKB
                                            * (
                                                d2Ks_dR2 / Ks
                                                - (dKs_dR / Ks) ** 2
                                                - d2Bs_dR2 / Bs
                                                + (dBs_dR / Bs) ** 2
                                            )
                                        )
                                    )
                                    + (
                                        cos_ng
                                        * (
                                            dBs_dR
                                            / Ks
                                            * ztanhzKB
                                            * (dKs_dR / Bs - dBs_dR / Bs**2 * Ks)
                                            - (d2Bs_dR2 / Ks - dBs_dR * dKs_dR / Ks**2)
                                            * log_sechzKB
                                        )
                                        + dBs_dR
                                        / Ks
                                        * log_sechzKB
                                        * sin_ng
                                        * ns
                                        * dg_dR
                                    )
                                    + (
                                        (
                                            cos_ng
                                            * (d2Ks_dR2 / Ks**2 - 2 * dKs_dR**2 / Ks**3)
                                            - dKs_dR / Ks**2 * sin_ng * ns * dg_dR
                                        )
                                        + (
                                            cos_ng
                                            * (
                                                d2Ds_dR2 / Ds / Ks
                                                - (dDs_dR / Ds) ** 2 / Ks
                                                - dDs_dR / Ds / Ks**2 * dKs_dR
                                            )
                                            - sin_ng * ns * dg_dR * dDs_dR / Ds / Ks
                                        )
                                    )
                                )
                            )
                            - 1
                            / Ks
                            * (
                                cos_ng / Rs
                                + (
                                    cos_ng
                                    * (
                                        (dDs_dR * Ks + Ds * dKs_dR) / (Ds * Ks)
                                        - (
                                            ztanhzKB * (dBs_dR / Bs * Ks - dKs_dR)
                                            + log_sechzKB * dBs_dR
                                        )
                                    )
                                    + sin_ng * ns * dg_dR
                                )
                            )
                        )
                    ),
                    axis=0,
                )
            )
        )

    def _z2deriv(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)
        zKB = z * Ks / Bs
        tanh2_zKB = xp.tanh(zKB) ** 2

        return (
            -self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                Cs
                * Ks
                / Ds
                * ((tanh2_zKB - 1) / Bs + tanh2_zKB)
                * xp.cos(ns * g)
                / xp.cosh(zKB) ** Bs,
                axis=0,
            )
        )

    def _phi2deriv(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        g = self._gamma(R, phi - self._omega * t)
        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        return (
            self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                Cs
                * self._N**2.0
                * ns**2.0
                / Ds
                / Ks
                / xp.cosh(z * Ks / Bs) ** Bs
                * xp.cos(ns * g),
                axis=0,
            )
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        Rs = self._Rs
        He = self._H * xp.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        dKs_dR = self._dK_dR(R, ns)
        dBs_dR = self._dB_dR(R, HNn)
        dDs_dR = self._dD_dR(R, HNn)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = xp.cos(ns * g)
        sin_ng = xp.sin(ns * g)

        zKB = z * Ks / Bs
        sechzKB = 1 / xp.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs
        log_sechzKB = xp.log(sechzKB)
        tanhzKB = xp.tanh(zKB)

        return -He * xp.sum(
            sechzKB_Bs
            * Cs
            / Ds
            * (
                Ks
                * tanhzKB
                * (
                    ns * dg_dR / Ks * sin_ng
                    + cos_ng
                    * (
                        z * tanhzKB * (dKs_dR / Ks - dBs_dR / Bs)
                        - dBs_dR / Ks * log_sechzKB
                        + dKs_dR / Ks**2
                        + dDs_dR / Ds / Ks
                    )
                )
                - cos_ng
                * (
                    (
                        zKB * (dKs_dR / Ks - dBs_dR / Bs) * (1 - tanhzKB**2)
                        + tanhzKB * (dKs_dR / Ks - dBs_dR / Bs)
                        + dBs_dR / Bs * tanhzKB
                    )
                    - tanhzKB / Rs
                )
            ),
            axis=0,
        )

    def _Rphideriv(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        He = self._H * xp.exp(-(R - self._r_ref) / self._Rs)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        dKs_dR = self._dK_dR(R, ns)
        dBs_dR = self._dB_dR(R, HNn)
        dDs_dR = self._dD_dR(R, HNn)

        g = self._gamma(R, phi - self._omega * t)
        dg_dR = self._dgamma_dR(R)

        cos_ng = xp.cos(ns * g)
        sin_ng = xp.sin(ns * g)
        zKB = z * Ks / Bs
        sechzKB = 1 / xp.cosh(zKB)
        sechzKB_Bs = sechzKB**Bs

        return -He * xp.sum(
            Cs
            * sechzKB_Bs
            / Ds
            * ns
            * self._N
            * (
                -ns * dg_dR / Ks * cos_ng
                + sin_ng
                * (
                    z * xp.tanh(zKB) * (dKs_dR / Ks - dBs_dR / Bs)
                    + 1
                    / Ks
                    * (
                        -dBs_dR * xp.log(sechzKB)
                        + dKs_dR / Ks
                        + dDs_dR / Ds
                        + 1 / self._Rs
                    )
                )
            ),
            axis=0,
        )

    def _phizderiv(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)
        zK_B = z * Ks / Bs

        return (
            -self._H
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * xp.sum(
                Cs
                / Ds
                * ns
                * self._N
                * xp.sin(ns * self._gamma(R, phi - self._omega * t))
                * xp.tanh(zK_B)
                / xp.cosh(zK_B) ** Bs,
                axis=0,
            )
        )

    def _dens(self, R, z, phi=0, t=0):
        xp = get_namespace(R, z, phi)
        Cs, ns, HNn = self._nvectors(R, z, xp)

        g = self._gamma(R, phi - self._omega * t)

        Ks = self._K(R, ns)
        Bs = self._B(R, HNn)
        Ds = self._D(R, HNn)

        ng = ns * g
        zKB = z * Ks / Bs
        sech_zKB = 1 / xp.cosh(zKB)
        tanh_zKB = xp.tanh(zKB)
        log_sech_zKB = xp.log(sech_zKB)

        # numpy of E as defined in the appendix of the paper.
        E = (
            1
            + Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H) ** 2)
            - R / self._Rs
            - (Ks * self._H) * (1 + 0.8 * Ks * self._H) * log_sech_zKB
            - 0.4 * (Ks * self._H) ** 2 * zKB * tanh_zKB
        )

        # numpy array of rE' as define in the appendix of the paper.
        rE = (
            -Ks
            * self._H
            / Ds
            * (1 - 0.3 * (1 - 0.3 * Ks * self._H) / (1 + 0.3 * Ks * self._H) ** 3)
            + (Ks * self._H / Ds * (1 - 0.3 / (1 + 0.3 * Ks * self._H) ** 2))
            - R / self._Rs
            + Ks * self._H * (1 + 1.6 * Ks * self._H) * log_sech_zKB
            - (0.4 * (Ks * self._H) ** 2 * zKB * sech_zKB) ** 2 / Bs
            + 1.2 * (Ks * self._H) ** 2 * zKB * tanh_zKB
        )

        return xp.sum(
            Cs
            * self._rho0
            * (self._H / (Ds * R))
            * xp.exp(-(R - self._r_ref) / self._Rs)
            * sech_zKB**Bs
            * (
                xp.cos(ng)
                * (Ks * R * (Bs + 1) / Bs * sech_zKB**2 - 1 / Ks / R * (E**2 + rE))
                - 2 * xp.sin(ng) * E * math.cos(self._alpha)
            ),
            axis=0,
        )

    def OmegaP(self):
        return self._omega

    def _nvectors(self, R, z, xp):
        """Return the (Cs, ns, HNn) summation vectors, in the active namespace and
        broadcast against the input shape (column vectors for array inputs, 1-D for
        scalar inputs), so that the per-mode sum (over axis 0) is backend-agnostic
        and reads no per-instance mutable state (safe under jax/torch tracing)."""
        ndR = getattr(R, "ndim", 0)
        ndz = getattr(z, "ndim", 0)
        if ndR > 1 or ndz > 1:
            raise ValueError("Array R and z with more than one dimension not supported")
        # Anchor the stored float constants on the input dtype so that float32
        # inputs are not promoted to float64 by these (strong-typed under torch)
        # float64 tensors, and so that the default integer Cs=[1] does not route
        # `Cs * self._rho0` through torch's default dtype. dtype=None for
        # python-scalar inputs (and numpy.asarray(x, dtype=None) is a no-op
        # pass-through), so the numpy path is untouched.
        dtype = getattr(R, "dtype", None)
        Cs = xp.asarray(self._Cs0, dtype=dtype)
        ns = xp.asarray(self._ns0)  # keep integer: int tensors never promote floats
        HNn = xp.asarray(self._HNn0, dtype=dtype)
        if ndR == 1 or ndz == 1:
            return (
                xp.reshape(Cs, (-1, 1)),
                xp.reshape(ns, (-1, 1)),
                xp.reshape(HNn, (-1, 1)),
            )
        return Cs, ns, HNn

    def _gamma(self, R, phi):
        """Return gamma. (eqn 3 in the paper)"""
        xp = get_namespace(R, phi)
        return self._N * (
            phi - self._phi_ref - xp.log(R / self._r_ref) / self._tan_alpha
        )

    def _dgamma_dR(self, R):
        """Return the first derivative of gamma wrt R."""
        return -self._N / R / self._tan_alpha

    def _K(self, R, ns=None):
        """Return numpy array from K1 up to and including Kn. (eqn. 5)"""
        if ns is None:
            ns = self._ns0
        return ns * self._N / R / self._sin_alpha

    def _dK_dR(self, R, ns=None):
        """Return numpy array of dK/dR from K1 up to and including Kn."""
        if ns is None:
            ns = self._ns0
        return -ns * self._N / R**2 / self._sin_alpha

    def _B(self, R, HNn=None):
        """Return numpy array from B1 up to and including Bn. (eqn. 6)"""
        if HNn is None:
            HNn = self._HNn0
        HNn_R = HNn / R

        return HNn_R / self._sin_alpha * (0.4 * HNn_R / self._sin_alpha + 1)

    def _dB_dR(self, R, HNn=None):
        """Return numpy array of dB/dR from B1 up to and including Bn."""
        if HNn is None:
            HNn = self._HNn0
        return -HNn / R**3 / self._sin_alpha**2 * (0.8 * HNn + R * self._sin_alpha)

    def _D(self, R, HNn=None):
        """Return numpy array from D1 up to and including Dn. (eqn. 7)"""
        if HNn is None:
            HNn = self._HNn0
        return (0.3 * HNn**2 / self._sin_alpha / R + HNn + R * self._sin_alpha) / (
            0.3 * HNn + R * self._sin_alpha
        )

    def _dD_dR(self, R, HNn=None):
        """Return numpy array of dD/dR from D1 up to and including Dn."""
        if HNn is None:
            HNn = self._HNn0
        HNn_R_sina = HNn / R / self._sin_alpha

        return HNn_R_sina * (
            0.3
            * (HNn_R_sina + 0.3 * HNn_R_sina**2.0 + 1)
            / R
            / (0.3 * HNn_R_sina + 1) ** 2
            - (1 / R * (1 + 0.6 * HNn_R_sina) / (0.3 * HNn_R_sina + 1))
        )
