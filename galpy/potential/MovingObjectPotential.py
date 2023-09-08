###############################################################################
#   MovingObjectPotential.py: class that implements the potential coming from
#                             a moving object
###############################################################################
import copy

import numpy

from .PlummerPotential import PlummerPotential
from .Potential import (
    Potential,
    _check_c,
    _isNonAxi,
    evaluateDensities,
    evaluatePotentials,
    evaluateRforces,
    evaluatezforces,
    flatten,
)


class MovingObjectPotential(Potential):
    """
    Class that implements the potential coming from a moving object by combining
    any galpy potential with an integrated galpy orbit.
    """

    def __init__(self, orbit, pot=None, amp=1.0, ro=None, vo=None):
        """
        Initialize a MovingObjectPotential.

        Parameters
        ----------
        orbit : galpy.orbit.Orbit
            The orbit of the object.
        pot : Potential object or list of Potential objects
            A potential object or list of potential objects representing the potential of the moving object; should be spherical, but this is not checked. Default is `PlummerPotential(amp=0.06,b=0.01)`.
        amp : float, optional
            Another amplitude to apply to the potential. Default is 1.0.
        ro : float, optional
            Distance scale for translation into internal units (default from configuration file).
        vo : float, optional
            Velocity scale for translation into internal units (default from configuration file).

        Notes
        -----
        - 2011-04-10 - Started - Bovy (NYU)
        - 2018-10-18 - Re-implemented to represent general object potentials using galpy potential models - James Lane (UofT)
        """
        Potential.__init__(self, amp=amp, ro=ro, vo=vo)
        # If no potential supplied use a default Plummer sphere
        if pot is None:
            pot = PlummerPotential(amp=0.06, b=0.01)
            self._pot = pot
        else:
            pot = flatten(pot)
            if _isNonAxi(pot):
                raise NotImplementedError(
                    "MovingObjectPotential for non-axisymmetric potentials is not currently supported"
                )
            self._pot = pot
        self._orb = copy.deepcopy(orbit)
        self._orb.turn_physical_off()
        self.isNonAxi = True
        self.hasC = _check_c(self._pot)
        return None

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        # Evaluate potential
        return evaluatePotentials(self._pot, Rdist, orbz - z, t=t, use_physical=False)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
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
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Evaluate and return z force
        return -evaluatezforces(self._pot, Rdist, zd, t=t, use_physical=False)

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        # Cylindrical distance
        Rdist = _cylR(R, phi, self._orb.R(t), self._orb.phi(t))
        # Difference vector
        orbz = self._orb.z(t) if self._orb.dim() == 3 else 0
        (xd, yd, zd) = _cyldiff(self._orb.R(t), self._orb.phi(t), orbz, R, phi, z)
        # Evaluate cylindrical radial force.
        RF = evaluateRforces(self._pot, Rdist, zd, t=t, use_physical=False)
        # Return phi force, negative of phi vector to evaluate location
        return -RF * R * (numpy.cos(phi) * yd - numpy.sin(phi) * xd) / Rdist

    def _dens(self, R, z, phi=0.0, t=0.0):
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
