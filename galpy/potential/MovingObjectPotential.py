###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
###############################################################################
import copy

import numpy

from ..potential.Potential import _check_potential_list_and_deprecate
from .PlummerPotential import PlummerPotential
from .Potential import (
    Potential,
    _check_c,
    _isNonAxi,
    evaluateDensities,
    evaluatePotentials,
    evaluateR2derivs,
    evaluateRforces,
    evaluateRzderivs,
    evaluatez2derivs,
    evaluatezforces,
)
from .RotateAndTiltWrapperPotential import RotateAndTiltWrapperPotential


class MovingObjectPotential(Potential):
    """
    Class that implements the potential coming from a moving object by combining
    any galpy potential with an integrated galpy orbit.
    """

    def __init__(self, orbit, pot=None, amp=1.0, ro=None, vo=None, _general=False):
        """
        Initialize a MovingObjectPotential.

        Parameters
        ----------
        orbit : galpy.orbit.Orbit
            The orbit of the object.
        pot : Potential object or a combined potential formed using addition (pot1+pot2+…)
            A potential object or combination of potential objects representing the potential of the moving object, in a frame centered on the object with axes parallel to those of the orbit's frame. Default is `PlummerPotential(amp=0.06,b=0.01)`.
        amp : float, optional
            Another amplitude to apply to the potential. Default is 1.0.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - An axisymmetric `pot` is evaluated at the cylindrical radius and the height relative to the object, with the height measured downwards (`z_object - z`), which is the same for potentials that are symmetric under z -> -z. A non-axisymmetric `pot` is evaluated at the position relative to the object, x - x_object(t), as ``RotateAndTiltWrapperPotential(pot=pot, center=orbit)``; use that wrapper directly to also rotate `pot`, with a constant or time-dependent orientation.
        - 2011-04-10 - Started - Bovy (NYU)
        - 2018-10-18 - Re-implemented to represent general object potentials using galpy potential models - James Lane (UofT)
        - 2026-11-14 - Non-axisymmetric potentials, through RotateAndTiltWrapperPotential - Bovy (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        # If no potential supplied use a default Plummer sphere
        if pot is None:
            pot = PlummerPotential(amp=0.06, b=0.01)
            self._pot = pot
        else:
            pot = _check_potential_list_and_deprecate(pot)
            self._pot = pot
        self._orb = copy.deepcopy(orbit)
        self._orb.turn_physical_off()
        # The general path (a non-axisymmetric pot, or forced through the
        # private _general for comparisons with the axisymmetric fast path):
        # a RotateAndTiltWrapperPotential centered on the orbit
        self._general = _general or _isNonAxi(self._pot)
        if self._general:
            self._gen = RotateAndTiltWrapperPotential(
                pot=self._pot, center=self._orb, ro=self._ro, vo=self._vo
            )
            # the planar C implementation is the axisymmetric one
            self.hasC_planar = False
        self.isNonAxi = True
        self.hasC = _check_c(self._pot)
        # The second derivatives (for the variational equations) are the
        # kernel's second derivatives at the shifted point x-x_obj(t), both in
        # Python (below) and in C, where they are evaluated through the
        # wrapped potential's pointers exactly like the forces. They are thus
        # available in C exactly when the kernel's are (same gating as hasC).
        self.hasC_dxdv = _check_c(self._pot, dxdv=True)
        self.hasC_dxdv3d = _check_c(self._pot, dxdv3d=True)
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._evaluate(R, z, phi=phi, t=t)
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        # Evaluate potential
        return evaluatePotentials(self._pot, Rdist, orbz - z, t=t, use_physical=False)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._Rforce(R, z, phi=phi, t=t)
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Evaluate cylindrical radial force
        RF = evaluateRforces(self._pot, Rdist, zd, t=t, use_physical=False)

        # Return R force, negative of radial vector to evaluation location.
        return -RF * (numpy.cos(phi) * xd + numpy.sin(phi) * yd) / Rdist

    def _zforce(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._zforce(R, z, phi=phi, t=t)
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Evaluate and return z force
        return -evaluatezforces(self._pot, Rdist, zd, t=t, use_physical=False)

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._phitorque(R, z, phi=phi, t=t)
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Evaluate cylindrical radial force.
        RF = evaluateRforces(self._pot, Rdist, zd, t=t, use_physical=False)
        # Return phi force, negative of phi vector to evaluate location
        return -RF * R * (numpy.cos(phi) * yd - numpy.sin(phi) * xd) / Rdist

    def _xyzHess(self, R, z, phi, t):
        """Cartesian first (phix, phiy; phiz is never needed) and second
        derivatives of the potential at the field point (R, z, phi) at time t
        (without the overall amp, which the Potential base class applies).

        Phi(x,t) = Psi(R'(x,y), z'(z)) with R'^2 = (x-x_o(t))^2+(y-y_o(t))^2
        and z' = z_o(t)-z (the conventions of the force methods above). The
        moving-object shift is a pure translation of the evaluation point, so
        the Hessian is the kernel's Hessian at the shifted point chain-ruled
        to the field point's coordinates: the time-dependence enters only
        through (R', z'), with no extra terms. The kernel Psi is required to
        be axisymmetric (enforced in __init__), so its phi derivatives vanish.
        """
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Kernel cylindrical derivatives at the shifted point (R',z')=(Rdist,zd)
        RF = evaluateRforces(self._pot, Rdist, zd, t=t, use_physical=False)
        R2d = evaluateR2derivs(self._pot, Rdist, zd, t=t, use_physical=False)
        Rzd = evaluateRzderivs(self._pot, Rdist, zd, t=t, use_physical=False)
        z2d = evaluatez2derivs(self._pot, Rdist, zd, t=t, use_physical=False)
        # Chain rule with dR'/dx = (x-x_o)/R' = -xd/R' and dz'/dz = -1; the
        # minus signs cancel pairwise in the pure-second-derivative terms
        phix = RF * xd / Rdist
        phiy = RF * yd / Rdist
        phixx = R2d * xd**2.0 / Rdist**2.0 - RF * yd**2.0 / Rdist**3.0
        phiyy = R2d * yd**2.0 / Rdist**2.0 - RF * xd**2.0 / Rdist**3.0
        phixy = (R2d + RF / Rdist) * xd * yd / Rdist**2.0
        phixz = Rzd * xd / Rdist
        phiyz = Rzd * yd / Rdist
        phizz = z2d
        return (phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz)

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._R2deriv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        return cp**2.0 * phixx + 2.0 * cp * sp * phixy + sp**2.0 * phiyy

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._z2deriv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        return phizz

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._Rzderiv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        return cp * phixz + sp * phiyz

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._phi2deriv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        return R**2.0 * (
            sp**2.0 * phixx - 2.0 * cp * sp * phixy + cp**2.0 * phiyy
        ) - R * (cp * phix + sp * phiy)

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._Rphideriv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        return (
            R * (cp * sp * (phiyy - phixx) + (cp**2.0 - sp**2.0) * phixy)
            - sp * phix
            + cp * phiy
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._phizderiv(R, z, phi=phi, t=t)
        phix, phiy, phixx, phixy, phiyy, phixz, phiyz, phizz = self._xyzHess(
            R, z, phi, t
        )
        cp, sp = numpy.cos(phi), numpy.sin(phi)
        return R * (cp * phiyz - sp * phixz)

    def _dens(self, R, z, phi=0.0, t=0.0):
        if self._general:
            return self._gen._dens(R, z, phi=phi, t=t)
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Return the density
        return evaluateDensities(self._pot, Rdist, zd, t=t, use_physical=False)


def _cylR(R1, phi1, R2, phi2):
    return numpy.sqrt(
        R1**2.0 + R2**2.0 - 2.0 * R1 * R2 * numpy.cos(phi1 - phi2)
    )  # Cosine law


def _cyldiff(R1, phi1, z1, R2, phi2, z2):
    dx = R1 * numpy.cos(phi1) - R2 * numpy.cos(phi2)
    dy = R1 * numpy.sin(phi1) - R2 * numpy.sin(phi2)
    dz = z1 - z2
    return (dx, dy, dz)
