###############################################################################
#   SphericalHarmonicPotentialMixin.py: Mixin providing the spherical-to-
#   cylindrical force chain rule for spherical-harmonic-based potentials
###############################################################################
import numpy

from ..util import coords


class SphericalHarmonicPotentialMixin:
    """Mixin that provides cylindrical force evaluation (_Rforce, _zforce,
    _phitorque) via the chain rule from spherical force components.

    Subclasses must implement ``_compute_spher_forces_at_point(self, R, z, phi)``
    returning ``(dPhi_dr, dPhi_dtheta, dPhi_dphi)``.
    """

    def _evaluate_cyl_force(self, dr_dx, dtheta_dx, dphi_dx, R, z, phi):
        """
        Evaluate a cylindrical force component over an array of coordinates.

        Transforms spherical force components (dPhi/dr, dPhi/dtheta, dPhi/dphi) to a
        cylindrical component using the chain rule derivatives dr/dx, dtheta/dx, dphi/dx.

        Parameters
        ----------
        dr_dx : float or numpy.ndarray
            The derivative of r with respect to the chosen cylindrical variable x.
        dtheta_dx : float or numpy.ndarray
            The derivative of theta with respect to x.
        dphi_dx : float or numpy.ndarray
            The derivative of phi with respect to x.
        R : float or numpy.ndarray
            Cylindrical Galactocentric radius.
        z : float or numpy.ndarray
            Vertical height.
        phi : float or numpy.ndarray
            Azimuth.

        Returns
        -------
        float or numpy.ndarray
            The force component in the x direction.

        Notes
        -----
        - 2016-06-02 - Written - Aladdin Seaifan (UofT)
        - 2026-02-11 - Simplified - Bovy (UofT)
        - 2026-02-13 - Moved to SphericalHarmonicPotentialMixin - Bovy (UofT)
        """
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        shape = (R * z * phi).shape
        if shape == ():
            dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                R, z, phi
            )
            return dr_dx * dPhi_dr + dtheta_dx * dPhi_dtheta + dPhi_dphi * dphi_dx
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        force = numpy.zeros(shape, float)
        dr_dx = dr_dx * numpy.ones(shape)
        dtheta_dx = dtheta_dx * numpy.ones(shape)
        dphi_dx = dphi_dx * numpy.ones(shape)
        for idx in numpy.ndindex(*shape):
            dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                R[idx], z[idx], phi[idx]
            )
            force[idx] = (
                dr_dx[idx] * dPhi_dr
                + dtheta_dx[idx] * dPhi_dtheta
                + dPhi_dphi * dphi_dx[idx]
            )
        return force

    def _Rforce(self, R, z, phi=0, t=0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        dr_dR = numpy.divide(R, r)
        dtheta_dR = numpy.divide(z, r**2)
        return self._evaluate_cyl_force(dr_dR, dtheta_dR, 0, R, z, phi)

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        dr_dz = numpy.divide(z, r)
        dtheta_dz = numpy.divide(-R, r**2)
        return self._evaluate_cyl_force(dr_dz, dtheta_dz, 0, R, z, phi)

    def _phitorque(self, R, z, phi=0, t=0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        return self._evaluate_cyl_force(0, 0, 1, R, z, phi)

    def _evaluate_cyl_2nd_deriv(self, deriv_type, R, z, phi):
        """
        Evaluate a cylindrical second derivative over an array of coordinates.

        Parameters
        ----------
        deriv_type : str
            One of 'R2', 'z2', 'Rz', 'phi2', 'Rphi', 'phiz'.
        R : float or numpy.ndarray
            Cylindrical Galactocentric radius.
        z : float or numpy.ndarray
            Vertical height.
        phi : float or numpy.ndarray
            Azimuth.

        Returns
        -------
        float or numpy.ndarray
            The second derivative.

        Notes
        -----
        - 2026-02-18 - Written - Bovy (UofT)
        """
        R = numpy.array(R, dtype=float)
        z = numpy.array(z, dtype=float)
        phi = numpy.array(phi, dtype=float)
        shape = (R * z * phi).shape
        if shape == ():
            return self._cyl_2nd_deriv_at_point(deriv_type, R, z, phi)
        R = R * numpy.ones(shape)
        z = z * numpy.ones(shape)
        phi = phi * numpy.ones(shape)
        result = numpy.zeros(shape, float)
        for idx in numpy.ndindex(*shape):
            result[idx] = self._cyl_2nd_deriv_at_point(
                deriv_type, R[idx], z[idx], phi[idx]
            )
        return result

    def _cyl_2nd_deriv_at_point(self, deriv_type, R, z, phi):
        """
        Compute a single cylindrical second derivative at a point using the chain rule.

        Notes
        -----
        - 2026-02-18 - Written - Bovy (UofT)
        """
        (
            Phi_rr,
            Phi_tt,
            Phi_pp,
            Phi_rt,
            Phi_rp,
            Phi_tp,
            Phi_r,
            Phi_t,
        ) = self._compute_spher_2nd_derivs_at_point(R, z, phi)
        r2 = R * R + z * z
        r = numpy.sqrt(r2)
        if r == 0.0:
            return 0.0
        r3 = r * r2
        r4 = r2 * r2
        if deriv_type == "R2":
            return (
                Phi_rr * R * R / r2
                + Phi_r * z * z / r3
                + 2.0 * Phi_rt * R * z / r3
                + Phi_tt * z * z / r4
                + Phi_t * (-2.0 * R * z) / r4
            )
        elif deriv_type == "z2":
            return (
                Phi_rr * z * z / r2
                + Phi_r * R * R / r3
                - 2.0 * Phi_rt * z * R / r3
                + Phi_tt * R * R / r4
                + Phi_t * 2.0 * R * z / r4
            )
        elif deriv_type == "Rz":
            return (
                Phi_rr * R * z / r2
                + Phi_r * (-R * z) / r3
                + Phi_rt * (z * z - R * R) / r3
                + Phi_tt * (-z * R) / r4
                + Phi_t * (R * R - z * z) / r4
            )
        elif deriv_type == "phi2":
            return Phi_pp
        elif deriv_type == "Rphi":
            return Phi_rp * R / r + Phi_tp * z / r2
        elif deriv_type == "phiz":
            return Phi_rp * z / r + Phi_tp * (-R) / r2
