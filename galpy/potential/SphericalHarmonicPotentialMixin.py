###############################################################################
#   SphericalHarmonicPotentialMixin.py: Mixin providing the spherical-to-
#   cylindrical force chain rule for spherical-harmonic-based potentials
###############################################################################
import numpy

from ..backend import get_namespace, match_input_dtype
from ..util import coords


class SphericalHarmonicPotentialMixin:
    """Mixin that provides cylindrical force evaluation (_Rforce, _zforce,
    _phitorque) via the chain rule from spherical force components.

    Subclasses must implement ``_compute_spher_forces_at_point(self, R, z, phi, t=0)``
    returning ``(dPhi_dr, dPhi_dtheta, dPhi_dphi)``.
    """

    def _evaluate_cyl_force(self, dr_dx, dtheta_dx, dphi_dx, R, z, phi, t=0.0):
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
        t : float, optional
            Time. Default: 0.0.

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
        xp = get_namespace(R, z, phi)
        if xp is numpy:
            R = numpy.array(R, dtype=float)
            z = numpy.array(z, dtype=float)
            phi = numpy.array(phi, dtype=float)
            t = numpy.array(t, dtype=float)
            shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape, t.shape)
            if shape == ():
                dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                    R, z, phi, t=t
                )
                return dr_dx * dPhi_dr + dtheta_dx * dPhi_dtheta + dPhi_dphi * dphi_dx
            R = R * numpy.ones(shape)
            z = z * numpy.ones(shape)
            phi = phi * numpy.ones(shape)
            t = t * numpy.ones(shape)
            force = numpy.zeros(shape, float)
            dr_dx = dr_dx * numpy.ones(shape)
            dtheta_dx = dtheta_dx * numpy.ones(shape)
            dphi_dx = dphi_dx * numpy.ones(shape)
            for idx in numpy.ndindex(*shape):
                dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                    R[idx], z[idx], phi[idx], t=t[idx]
                )
                force[idx] = (
                    dr_dx[idx] * dPhi_dr
                    + dtheta_dx[idx] * dPhi_dtheta
                    + dPhi_dphi * dphi_dx[idx]
                )
            return force
        # backend path: identical per-point evaluation, assembled functionally
        # (stack instead of in-place writes) so it traces and differentiates.
        R = xp.asarray(R) * 1.0
        z = xp.asarray(z) * 1.0
        phi = xp.asarray(phi) * 1.0
        t = xp.asarray(t) * 1.0
        shape = numpy.broadcast_shapes(
            tuple(R.shape), tuple(z.shape), tuple(phi.shape), tuple(t.shape)
        )
        if shape == ():
            dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                R, z, phi, t=t
            )
            return dr_dx * dPhi_dr + dtheta_dx * dPhi_dtheta + dPhi_dphi * dphi_dx
        R = xp.broadcast_to(R, shape)
        z = xp.broadcast_to(z, shape)
        phi = xp.broadcast_to(phi, shape)
        t = xp.broadcast_to(t, shape)
        dr_dx = xp.broadcast_to(xp.asarray(dr_dx) * 1.0, shape)
        dtheta_dx = xp.broadcast_to(xp.asarray(dtheta_dx) * 1.0, shape)
        dphi_dx = xp.broadcast_to(xp.asarray(dphi_dx) * 1.0, shape)
        force = []
        for idx in numpy.ndindex(*shape):
            dPhi_dr, dPhi_dtheta, dPhi_dphi = self._compute_spher_forces_at_point(
                R[idx], z[idx], phi[idx], t=t[idx]
            )
            force.append(
                dr_dx[idx] * dPhi_dr
                + dtheta_dx[idx] * dPhi_dtheta
                + dPhi_dphi * dphi_dx[idx]
            )
        return xp.reshape(xp.stack(force), shape)

    def _Rforce(self, R, z, phi=0, t=0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        xp = get_namespace(R, z, phi)
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        dr_dR = xp.divide(R, r)
        dtheta_dR = xp.divide(z, r**2)
        # the expansion tables of the implementing classes (SCF /
        # MultipoleExpansion) are deliberately float64 (precision); cast the
        # result to the input dtype at exit (no-op for float64/scalar inputs)
        return match_input_dtype(
            self._evaluate_cyl_force(dr_dR, dtheta_dR, 0, R, z, phi, t=t),
            R,
            z,
            phi,
            t,
        )

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        xp = get_namespace(R, z, phi)
        r, theta, phi = coords.cyl_to_spher(R, z, phi)
        dr_dz = xp.divide(z, r)
        dtheta_dz = xp.divide(-R, r**2)
        # float64 interior, input-dtype exit cast (see _Rforce)
        return match_input_dtype(
            self._evaluate_cyl_force(dr_dz, dtheta_dz, 0, R, z, phi, t=t),
            R,
            z,
            phi,
            t,
        )

    def _phitorque(self, R, z, phi=0, t=0):
        if not self.isNonAxi and phi is None:
            phi = 0.0
        # float64 interior, input-dtype exit cast (see _Rforce)
        return match_input_dtype(
            self._evaluate_cyl_force(0, 0, 1, R, z, phi, t=t), R, z, phi, t
        )

    def _evaluate_cyl_2nd_deriv(self, deriv_type, R, z, phi, t=0.0):
        # float64 interior, input-dtype exit cast (see _Rforce); the core
        # method below reassigns R/z/phi/t, so the cast wraps it here, where
        # the original inputs (whose dtype is to be matched) are available
        return match_input_dtype(
            self._evaluate_cyl_2nd_deriv_core(deriv_type, R, z, phi, t=t),
            R,
            z,
            phi,
            t,
        )

    def _evaluate_cyl_2nd_deriv_core(self, deriv_type, R, z, phi, t=0.0):
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
        t : float, optional
            Time. Default: 0.0.

        Returns
        -------
        float or numpy.ndarray
            The second derivative.

        Notes
        -----
        - 2026-02-18 - Written - Bovy (UofT)
        """
        # axisymmetric potentials are evaluated with phi=None (phi irrelevant)
        if not self.isNonAxi and phi is None:
            phi = 0.0
        xp = get_namespace(R, z, phi)
        if xp is numpy:
            R = numpy.array(R, dtype=float)
            z = numpy.array(z, dtype=float)
            phi = numpy.array(phi, dtype=float)
            t = numpy.array(t, dtype=float)
            shape = numpy.broadcast_shapes(R.shape, z.shape, phi.shape, t.shape)
            if shape == ():
                return self._cyl_2nd_deriv_at_point(deriv_type, R, z, phi, t=t)
            R = R * numpy.ones(shape)
            z = z * numpy.ones(shape)
            phi = phi * numpy.ones(shape)
            t = t * numpy.ones(shape)
            result = numpy.zeros(shape, float)
            for idx in numpy.ndindex(*shape):
                result[idx] = self._cyl_2nd_deriv_at_point(
                    deriv_type, R[idx], z[idx], phi[idx], t=t[idx]
                )
            return result
        # backend path: identical per-point evaluation, assembled functionally.
        R = xp.asarray(R) * 1.0
        z = xp.asarray(z) * 1.0
        phi = xp.asarray(phi) * 1.0
        t = xp.asarray(t) * 1.0
        shape = numpy.broadcast_shapes(
            tuple(R.shape), tuple(z.shape), tuple(phi.shape), tuple(t.shape)
        )
        if shape == ():
            return self._cyl_2nd_deriv_at_point(deriv_type, R, z, phi, t=t)
        R = xp.broadcast_to(R, shape)
        z = xp.broadcast_to(z, shape)
        phi = xp.broadcast_to(phi, shape)
        t = xp.broadcast_to(t, shape)
        return xp.reshape(
            xp.stack(
                [
                    self._cyl_2nd_deriv_at_point(
                        deriv_type, R[idx], z[idx], phi[idx], t=t[idx]
                    )
                    for idx in numpy.ndindex(*shape)
                ]
            ),
            shape,
        )

    def _cyl_2nd_deriv_at_point(self, deriv_type, R, z, phi, t=0.0):
        """
        Compute a single cylindrical second derivative at a point using the chain rule.

        Notes
        -----
        - 2026-02-18 - Written - Bovy (UofT)
        """
        xp = get_namespace(R, z, phi)
        (
            Phi_rr,
            Phi_tt,
            Phi_pp,
            Phi_rt,
            Phi_rp,
            Phi_tp,
            Phi_r,
            Phi_t,
        ) = self._compute_spher_2nd_derivs_at_point(R, z, phi, t=t)
        r2 = R * R + z * z
        r = xp.sqrt(r2)
        if xp is numpy:
            if r == 0.0:
                return 0.0
        else:
            # backend path: branchless r == 0 guard. Both xp.where branches are
            # evaluated under tracing/eager AD, so the chain rule below runs at
            # a guarded radius and the centre is zeroed at the end.
            degenerate = r == 0.0
            r2 = xp.where(degenerate, 1.0, r2)
            r = xp.where(degenerate, 1.0, r)
        r3 = r * r2
        r4 = r2 * r2
        if deriv_type == "R2":
            out = (
                Phi_rr * R * R / r2
                + Phi_r * z * z / r3
                + 2.0 * Phi_rt * R * z / r3
                + Phi_tt * z * z / r4
                + Phi_t * (-2.0 * R * z) / r4
            )
        elif deriv_type == "z2":
            out = (
                Phi_rr * z * z / r2
                + Phi_r * R * R / r3
                - 2.0 * Phi_rt * z * R / r3
                + Phi_tt * R * R / r4
                + Phi_t * 2.0 * R * z / r4
            )
        elif deriv_type == "Rz":
            out = (
                Phi_rr * R * z / r2
                + Phi_r * (-R * z) / r3
                + Phi_rt * (z * z - R * R) / r3
                + Phi_tt * (-z * R) / r4
                + Phi_t * (R * R - z * z) / r4
            )
        elif deriv_type == "phi2":
            out = Phi_pp
        elif deriv_type == "Rphi":
            out = Phi_rp * R / r + Phi_tp * z / r2
        elif deriv_type == "phiz":
            out = Phi_rp * z / r + Phi_tp * (-R) / r2
        return out if xp is numpy else xp.where(degenerate, 0.0, out)
