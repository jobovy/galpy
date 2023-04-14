###############################################################################
#   NumericalPotentialDerivativesMixin: helper class to add numerical derivs
###############################################################################
class NumericalPotentialDerivativesMixin:
    """Mixin to add numerical derivatives to a Potential class, use as, e.g.,

    .. highlight:: python
    .. code-block:: python

        class PotWithNumericalDerivs(Potential,NumericalPotentialDerivativesMixin):
            def __init__(self,*args,**kwargs):
                NumericalPotentialDerivativesMixin.__init__(self,kwargs) # *not* **kwargs!
                # Remainder of initialization
                ...

            def _evaluate(self,R,z,phi=0.,t=0.):
                # Evaluate the potential

            # All forces and second derivatives then computed by NumericalPotentialDerivativesMixin

    to add numerical derivatives to a new potential class ``PotWithNumericalDerivs`` that only implements the potential itself, but not the forces. The class may implement any of the forces or second derivatives, all non-implemented forces/second-derivatives will be computed numerically by adding this Mixin

    The step used to compute the first (force) and second derivatives can be controlled at object instantiation by the keyword arguments ``dR``, ``dz``, ``dphi`` (for the forces; 1e-8 default) and ``dR2``, ``dz2``, and ``dphi2`` (for the second derivaives; 1e-4 default)
    """

    def __init__(self, kwargs):  # no **kwargs to get a reference, not a copy!
        # For first derivatives
        self._dR = kwargs.pop("dR", 1e-8)
        self._dphi = kwargs.pop("dphi", 1e-8)
        self._dz = kwargs.pop("dz", 1e-8)
        # For second derivatives
        self._dR2 = kwargs.pop("dR2", 1e-4)
        self._dphi2 = kwargs.pop("dphi2", 1e-4)
        self._dz2 = kwargs.pop("dz2", 1e-4)

    def _Rforce(self, R, z, phi=0.0, t=0.0):
        # Do forward difference because R cannot be negative
        RplusdR = R + self._dR
        Rplus2dR = R + 2.0 * self._dR
        dR = (Rplus2dR - R) / 2.0
        return (
            1.5 * self._evaluate(R, z, phi=phi, t=t)
            - 2.0 * self._evaluate(RplusdR, z, phi=phi, t=t)
            + 0.5 * self._evaluate(Rplus2dR, z, phi=phi, t=t)
        ) / dR

    def _zforce(self, R, z, phi=0.0, t=0.0):
        # Central difference to get derivative at z=0 right
        zplusdz = z + self._dz
        zminusdz = z - self._dz
        dz = zplusdz - zminusdz
        return (
            self._evaluate(R, zminusdz, phi=phi, t=t)
            - self._evaluate(R, zplusdz, phi=phi, t=t)
        ) / dz

    def _phitorque(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            return 0.0
        # Central difference
        phiplusdphi = phi + self._dphi
        phiminusdphi = phi - self._dphi
        dphi = phiplusdphi - phiminusdphi
        return (
            self._evaluate(R, z, phi=phiminusdphi, t=t)
            - self._evaluate(R, z, phi=phiplusdphi, t=t)
        ) / dphi

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        # Do forward difference because R cannot be negative
        RplusdR = R + self._dR2
        Rplus2dR = R + 2.0 * self._dR2
        Rplus3dR = R + 3.0 * self._dR2
        dR = (Rplus3dR - R) / 3.0
        return (
            2.0 * self._evaluate(R, z, phi=phi, t=t)
            - 5.0 * self._evaluate(RplusdR, z, phi=phi, t=t)
            + 4.0 * self._evaluate(Rplus2dR, z, phi=phi, t=t)
            - 1.0 * self._evaluate(Rplus3dR, z, phi=phi, t=t)
        ) / dR**2.0

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        # Central derivative
        zplusdz = z + self._dz2
        zminusdz = z - self._dz2
        dz = (zplusdz - zminusdz) / 2.0
        return (
            self._evaluate(R, zplusdz, phi=phi, t=t)
            + self._evaluate(R, zminusdz, phi=phi, t=t)
            - 2.0 * self._evaluate(R, z, phi=phi, t=t)
        ) / dz**2.0

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            return 0.0
        # Central derivative
        phiplusdphi = phi + self._dphi2
        phiminusdphi = phi - self._dphi2
        dphi = (phiplusdphi - phiminusdphi) / 2.0
        return (
            self._evaluate(R, z, phi=phiplusdphi, t=t)
            + self._evaluate(R, z, phi=phiminusdphi, t=t)
            - 2.0 * self._evaluate(R, z, phi=phi, t=t)
        ) / dphi**2.0

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        # Do forward difference in R because R cannot be negative
        RplusdR = R + self._dR2
        Rplus2dR = R + 2.0 * self._dR2
        dR = (Rplus2dR - R) / 2.0
        zplusdz = z + self._dz2
        zminusdz = z - self._dz2
        dz = zplusdz - zminusdz
        return (
            (
                -1.5 * self._evaluate(R, zplusdz, phi=phi, t=t)
                + 2.0 * self._evaluate(RplusdR, zplusdz, phi=phi, t=t)
                - 0.5 * self._evaluate(Rplus2dR, zplusdz, phi=phi, t=t)
                + 1.5 * self._evaluate(R, zminusdz, phi=phi, t=t)
                - 2.0 * self._evaluate(RplusdR, zminusdz, phi=phi, t=t)
                + 0.5 * self._evaluate(Rplus2dR, zminusdz, phi=phi, t=t)
            )
            / dR
            / dz
        )

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            return 0.0
        # Do forward difference in R because R cannot be negative
        RplusdR = R + self._dR2
        Rplus2dR = R + 2.0 * self._dR2
        dR = (Rplus2dR - R) / 2.0
        phiplusdphi = phi + self._dphi2
        phiminusdphi = phi - self._dphi2
        dphi = phiplusdphi - phiminusdphi
        return (
            (
                -1.5 * self._evaluate(R, z, phi=phiplusdphi, t=t)
                + 2.0 * self._evaluate(RplusdR, z, phi=phiplusdphi, t=t)
                - 0.5 * self._evaluate(Rplus2dR, z, phi=phiplusdphi, t=t)
                + 1.5 * self._evaluate(R, z, phi=phiminusdphi, t=t)
                - 2.0 * self._evaluate(RplusdR, z, phi=phiminusdphi, t=t)
                + 0.5 * self._evaluate(Rplus2dR, z, phi=phiminusdphi, t=t)
            )
            / dR
            / dphi
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        if not self.isNonAxi:
            return 0.0
        # Central derivative
        phiplusdphi = phi + self._dphi2
        phiminusdphi = phi - self._dphi2
        dphi = (phiplusdphi - phiminusdphi) / 2.0
        zplusdz = z + self._dz2
        zminusdz = z - self._dz2
        dz = zplusdz - zminusdz
        return (
            (
                self._evaluate(R, zplusdz, phi=phiplusdphi, t=t)
                - self._evaluate(R, zplusdz, phi=phiminusdphi, t=t)
                - self._evaluate(R, zminusdz, phi=phiplusdphi, t=t)
                + self._evaluate(R, zminusdz, phi=phiminusdphi, t=t)
            )
            / dz
            / dphi
            / 2.0
        )
