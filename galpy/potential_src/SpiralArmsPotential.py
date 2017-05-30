###############################################################################
#  SpiralArmsPotential.py: class that implements the spiral arms potential
#                           from Cox and Gomez (2002)
#
#  Phi(r, phi, z) = -4*pi*G*H*rho0*exp(-(r-r0)/Rs)*sum(Cn/(Kn*Dn)*cos(n*gamma)*sech(Kn*z/Bn)^Bn)
###############################################################################

import numpy as np
from galpy.util import bovy_coords, bovy_conversion
from galpy.potential_src.Potential import Potential, _APY_LOADED

if _APY_LOADED:
    from astropy import units


class SpiralArmsPotential(Potential):
    """Class that implements the spiral arms potential from Cox and Gomez (2002).
    
    .. math::
    
        \\Phi(r, \\phi, z) = -4 \\pi GH \\rho_0 exp(-\\frac{r-r_0}{R_s}) \\sum(\\frac{C_n}{(K_n D_n} cos(n \\gamma) sech(\\frac{K_n z}{B_n})^B_n)
        
    where
    
    .. math::
    
        
        
    """

    def __init__(self, amp=1., ro=None, vo=None, amp_units='density', normalize=False,
                  N=2, alpha=0.3, r_ref=1, phi_ref=0, Rs=0.5, H=0.5, Cs=[1], omega=0):
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
            :param r_ref: fiducial radius where rho = rho0 (r_0 in the paper by Cox and Gomez)
            :param phi_ref: reference angle (phi_p(r_0) in the paper by Cox and Gomez)
            :param Rs: radial scale length of the drop-off in density amplitude of the arms
            :param H: scale height of the stellar arm perturbation
            :param Cs: list of constants multiplying the cos(n*gamma) term in the mass density expression
            :param omega: rotational speed of the spiral arms
        OUTPUT:
            (none)
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        if _APY_LOADED:
            if isinstance(alpha, units.Quantity):
                alpha = alpha.to(units.rad).value

        self._N = N
        self._alpha = alpha
        self._r_ref = r_ref
        self._phi_ref = phi_ref
        self._Rs = Rs
        self._H = H
        self._Cs = np.array(Cs)
        self._ns = np.arange(1, len(Cs)+1)
        self._omega = omega
        self._rho0 = amp / (4. * np.pi)
        self.isNonAxi = True  # Potential is not axisymmetric

    def _evaluate(self, R, z, phi=0., t=0.):
        """
        NAME:
            _evaluate
        PURPOSE:
            Evaluate the potential at the given coordinates. (without the amp factor; handled by super class)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: Phi(R, z, phi, t)
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        R = np.maximum(R, 1e-7)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
            * np.sum(self._Cs/(Ks * Ds) * np.cos(self._ns * self._gamma(R, phi, t)) * 1./np.cosh(Ks * z / Bs) ** Bs)

    def _Rforce(self, R, z, phi=0., t=0.):
        """
        NAME:
            _Rforce
        PURPOSE:
            Evaluate the radial force for this potential at the given coordinates. (-dPhi/dR)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: the radial force
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        R = np.maximum(R, 1e-7)

        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)

        dKs_dR = self._dK_dR(R)
        dBs_dR = self._dB_dR(Ks, dKs_dR)
        dDs_dR = self._dD_dR(Ks, dKs_dR)
        g = self._gamma(R, phi, t)
        dg_dR = self._dgamma_dR(R)

        He = self._H * np.exp(-(R-self._r_ref)/self._Rs)
        Csech_DK = self._Cs / Ds / Ks / np.cosh(z*Ks/Bs)**Bs

        n = self._ns
        return He * np.sum(-Csech_DK*self._ns*np.sin(n*g)*dg_dR
                           + Csech_DK * (z*dBs_dR*Ks/Bs**2 - z*dKs_dR/Bs) * Bs * np.tanh(z*Ks/Bs)
                           + np.log(np.abs(dBs_dR/np.cosh(z*Ks/Bs))) * np.cos(n*g)
                           - Csech_DK * np.cos(n*g) * (dKs_dR/Ks + dDs_dR/Ds)) \
            - He/self._Rs * np.sum(Csech_DK * np.cos(n*g))

    def _zforce(self, R, z, phi=0., t=0.):
        """
        NAME:
            _zforce
        PURPOSE:
            Evaluate the vertical force for this potential at the given coordinates. (-dPhi/dz)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: the vertical force
        HISTORY:
            2017-05-25  Jack Hong (UBC) 
        """
        R = np.maximum(R, 1e-7)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
            * np.sum(self._Cs/Ds * np.cos(self._ns * self._gamma(R, phi, t))
                     * np.tanh(z*Ks/Bs) * 1./np.cosh(z*Ks/Bs)**Bs)

    def _phiforce(self, R, z, phi=0., t=0.):
        """
        NAME:
            _phiforce
        PURPOSE:
            Evaluate the azimuthal force in cylindrical coordinates. (-dPhi/dphi)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: the azimuthal force
        HISTORY:
            2017-05-25  Jack Hong (UBC)
        """
        R = np.maximum(R, 1e-7)
        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs) \
            * np.sum(self._N * self._ns * self._Cs / Ds / Ks / np.cosh(z*Ks/Bs)**Bs * np.sin(self._ns * g))

    def _R2deriv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _R2deriv
        PURPOSE:
            Evaluate the second (cylindrical) radial derivative of the potential.
             (d^2 potential / d R^2)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: the second radial derivative
        HISTORY:
            2017-05-12  Jack Hong (UBC)
        """
        return 0.0

    def _z2deriv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _z2deriv
        PURPOSE:
            Evaluate the second (cylindrical) vertical derivative of the potential.
             (d^2 potential / d z^2)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: the second vertical derivative
        HISTORY:
            2017-05-26  Jack Hong (UBC) 
        """
        R = np.maximum(R, 1e-7)
        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        return -self._H * np.exp(-(R-self._r_ref)/self._Rs)\
            * np.sum(self._Cs*Ks/Ds * ((1./Bs) * (np.tanh(z*Ks/Bs)**2. - 1.)
                                       + np.tanh(z*Ks/Bs)**2.)
                     * np.cos(self._ns*g)
                     * 1./np.cosh(z*Ks/Bs)**Bs)

    def _phi2deriv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _phi2deriv
        PURPOSE:
            Evaluate the second azimuthal derivative of the potential in cylindrical coordinates.
            (d^2 potential / d phi^2)
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: d^2 potential / d phi^2
        HISTORY:
            2017-05-29 Jack Hong (UBC)
        """
        R = np.maximum(R, 1e-7)
        g = self._gamma(R, phi, t)
        dg2_dphi = self._dgamma2_dphi(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        return self._H * np.exp(-(R-self._r_ref)/self._Rs) \
            * np.sum(self._Cs*self._N**2*self._ns**2/Ds/Ks/np.cosh(z*Ks/Bs)**Bs * np.cos(self._N*self._ns*g))

    def _Rzderiv(self, R, z, phi=0., t=0.):
        """
        NAME:
            _Rzderiv
        PURPOSE:
            Evaluate the mixed (cylindrical) radial and vertical derivative of the potential (d^2 potential / d R d z).
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
        OUTPUT:
            :return: d^2 potential / d R d z
        HISTORY:
            2017-05-12  Jack Hong (UBC) 
        """
        return 0.0

    def _dens(self, R, z, phi=0., t=0., approx=False):
        """
        NAME:
            _dens
        PURPOSE:
            Evaluate the density. If not given, the density is computed using the Poisson equation
            from the first and second derivatives of the potential (if all are implemented).
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param phi: azimuth
            :param t: time
            :param approx: if True, the approximate density is calculated (eqn. 10 in paper)
        OUTPUT:
            :return: the density
        HISTORY:
            2017-05-12  Jack Hong (UBC) 
        """
        R = np.maximum(R, 1e-7)
        g = self._gamma(R, phi, t)
        Ks = self._K(R)
        Bs = self._B(Ks)
        Ds = self._D(Ks)
        E = self._E(R, z, Ks, Bs, Ds)
        rE = self._rE(R, z, Ks, Bs, Ds)  # actually rE'
        if approx:
            return self._rho0 * np.exp(-(R-self._r_ref)/self._Rs) \
                * np.sum(self._Cs*(Ks*self._H*(Bs+1.)/(Ds*Bs)) * np.cos(self._ns*self._gamma(R, phi, t))
                         * 1./np.cosh(Ks*z/Bs)**(2.+Bs))

        return np.sum(self._Cs*self._rho0*(self._H/(Ds*R))*np.exp(-(R-self._r_ref)/self._Rs)*(1./np.cosh(Ks*z/Bs))**Bs
                      * (((Ks*R*(Bs+1)/Bs*(1./np.cosh(Ks*z/Bs))**2) - 1/Ks/R*(E**2+rE))*np.cos(self._ns*g)
                         - 2*E*np.cos(self._alpha)*np.sin(self._ns*g)))

    def _mass(self, R, z=0, t=0.):
        """
        NAME:
            _mass
        PURPOSE:
            Evaluate the mass. Return the mass up to R and between -z and z. 
            If not given, the mass is computed by integrating the density (if it is implemented or can be 
            calculated from the Poisson equation).
        INPUT:
            :param R: galactocentric cylindrical radius
            :param z: vertical height
            :param t: time
        OUTPUT:
            :return: the mass
        HISTORY:
            2017-05-12  Jack Hong (UBC) 
        """
        # TODO: implement mass
        return 0.0

    def _gamma(self, R, phi, t):
        """Return gamma."""
        return self._N * (phi - self._phi_ref - np.log(R / self._ro) / np.tan(self._alpha) + self._omega * t)

    def _dgamma_dR(self, R):
        """Return the first derivative of gamma wrt R."""
        return -self._N / R / np.tan(self._alpha)

    def _dgamma2_dphi(self, R, phi, t):
        """Return the first derivative of gamma^2 wrt phi."""
        return self._N**2. * (2.*self._omega*t + 2.*phi - 2.*self._phi_ref
                              - 2.*np.log(R/self._r_ref) / np.tan(self._alpha))

    def _K(self, R):
        """Return numpy array from K1 up to and including Kn."""
        return self._ns * self._N / R / np.sin(self._alpha)

    def _dK_dR(self, R):
        """Return numpy array of dK/dR from K1 up to and including Kn."""
        return -self._ns * self._N / R**2. / np.sin(self._alpha)

    def _B(self, Ks):
        """Return numpy array from B1 up to and including Bn."""
        return Ks * self._H * (1. + 0.4 * Ks * self._H)

    def _dB_dR(self, Ks, dKs_dR):
        """Return numpy array of constants from """
        return self._H * dKs_dR * (1. + 0.8*self._H*Ks)

    def _D(self, Ks):
        """Return numpy array from D1 up to and including Dn."""
        return (1. + Ks * self._H + 0.3 * (Ks * self._H) ** 2.) / (1. + 0.3 * Ks * self._H)

    def _dD_dR(self, Ks, dKs_dR):
        """Return numpy array of dD/dR from D1 up to and including Dn"""
        return ((self._H*dKs_dR + 0.6*self._H**2*Ks*dKs_dR) * (1. + 0.3*self._H*Ks)
                - (1. + Ks*self._H + 0.3*(self._H*Ks)**2.) * (0.3*self._H*dKs_dR)) / (1. + 0.3*Ks*self._H)**2.

    def _E(self, R, z, Ks, Bs, Ds):
        """Return numpy of E as defined in the paper."""
        return 1. + Ks*self._H/Ds*(1. - 0.3/(1.+0.3*Ks*self._H)**2.) - R/self._Rs \
            - (Ks*self._H)*(1. + 0.8*Ks*self._H)*np.log(1./np.cosh(Ks*z/Bs)) \
            - 0.4*(Ks*self._H)**2. * (Ks*z/Bs) * np.tanh(Ks*z/Bs)

    def _rE(self, R, z, Ks, Bs, Ds):
        """Return numpy array of rE' as define in the paper."""
        return -Ks*self._H/Ds * (1. - 0.3*(1-0.3*Ks*self._H)/(1.+0.3*Ks*self._H)**3.) \
            + (Ks*self._H/Ds * (1. - 0.3/(1.+0.3*Ks*self._H)**2.)) - R/self._Rs \
            + Ks*self._H * (1. + 1.6*Ks*self._H) * np.log(1./np.cosh(Ks*z/Bs)) \
            - (0.4*(Ks*self._H)**2. * (Ks*z/Bs) * 1./np.cosh(Ks*z/Bs))**2. / Bs \
            + 1.2 * (Ks*self._H)**2 * (Ks*z/Bs) * np.tanh(Ks*z/Bs)
