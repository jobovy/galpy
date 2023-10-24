###############################################################################
#   DissipativeForce.py: top-level class for non-conservative forces
###############################################################################
import numpy

from ..util.conversion import physical_conversion, potential_physical_input
from .Force import Force


class DissipativeForce(Force):
    """Top-level class for non-conservative forces (cannot be derived from a potential function)"""

    def __init__(self, amp=1.0, ro=None, vo=None, amp_units=None):
        """
        Initialize a DissipativeForce object.

        Parameters
        ----------
        amp : float
            Amplitude to be applied when evaluating the potential and its forces.
        ro : float or Quantity, optional
            Distance from the Galactic center that corresponds to the zero point of the potential. Default is from galpy.util.const.
        vo : float or Quantity, optional
            Velocity that corresponds to the zero point of the velocity. Default is from galpy.util.const.
        amp_units : str or None, optional
            Units of the input amplitude. If None, default unit is used.

        Returns
        -------
        None

        Notes
        -----
        - 2018-03-16 - Started - Bovy (UofT)

        """
        Force.__init__(self, amp=amp, ro=ro, vo=vo, amp_units=amp_units)
        self.dim = 3
        self.isNonAxi = True  # Default: are non-axisymmetric
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate cylindrical radial force F_R  (R,z).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth. Default is 0.0.
        t : float or Quantity, optional
            Time. Default is 0.0.
        v : numpy.ndarray, optional
            3D velocity. Default is None.

        Returns
        -------
        float or Quantity
            Cylindrical radial force F_R (R,z,phi,t,v).

        Notes
        -----
        - 2018-03-18 - Written - Bovy (UofT)

        """
        return self._Rforce_nodecorator(R, z, phi=phi, t=t, v=v)

    def _Rforce_nodecorator(self, R, z, phi=0.0, t=0.0, v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, z, phi=phi, t=t, v=v)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_Rforce' function not implemented for this DissipativeForce"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def zforce(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate the vertical force F_z  (R,z,t).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth. Default is 0.0.
        t : float or Quantity, optional
            Time. Default is 0.0.
        v : numpy.ndarray, optional
            3D velocity. Default is None.

        Returns
        -------
        float or Quantity
            Vertical force F_z (R,z,phi,t,v).

        Notes
        -----
        - 2018-03-16 - Written - Bovy (UofT)

        """
        return self._zforce_nodecorator(R, z, phi=phi, t=t, v=v)

    def _zforce_nodecorator(self, R, z, phi=0.0, t=0.0, v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._zforce(R, z, phi=phi, t=t, v=v)
        except AttributeError:  # pragma: no cover
            from .Potential import PotentialError

            raise PotentialError(
                "'_zforce' function not implemented for this DissipativeForce"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate the azimuthal torque F_phi  (R,z,phi,t,v).

        Parameters
        ----------
        R : float or Quantity
            Cylindrical Galactocentric radius.
        z : float or Quantity
            Vertical height.
        phi : float or Quantity, optional
            Azimuth. Default is 0.0.
        t : float or Quantity, optional
            Time. Default is 0.0.
        v : numpy.ndarray, optional
            3D velocity. Default is None.

        Returns
        -------
        float or Quantity
            Azimuthal torque F_phi (R,z,phi,t,v).

        Notes
        -----
        - 2018-03-16 - Written - Bovy (UofT)

        """
        return self._phitorque_nodecorator(R, z, phi=phi, t=t, v=v)

    def _phitorque_nodecorator(self, R, z, phi=0.0, t=0.0, v=None):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, z, phi=phi, t=t, v=v)
        except AttributeError:  # pragma: no cover
            if self.isNonAxi:
                from .Potential import PotentialError

                raise PotentialError(
                    "'_phitorque' function not implemented for this DissipativeForce"
                )
            return 0.0


def _isDissipative(obj):
    """
    Determine whether this combination of potentials and forces is Dissipative

    Parameters
    ----------
    obj : Potential/DissipativeForce instance or list of such instances

    Returns
    -------
    bool
        True or False depending on whether the object is dissipative

    Notes
    -----
    - 2018-03-16 - Written - Bovy (UofT)
    - 2023-05-29 - Adjusted to also take planarDissipativeForces into account - Bovy (UofT)

    """
    from .planarDissipativeForce import planarDissipativeForce
    from .Potential import flatten

    obj = flatten(obj)
    isList = isinstance(obj, list)
    if isList:
        isCons = [
            not isinstance(p, DissipativeForce)
            and not isinstance(p, planarDissipativeForce)
            for p in obj
        ]
        nonCons = not numpy.prod(numpy.array(isCons))
    else:
        nonCons = isinstance(obj, DissipativeForce) or isinstance(
            obj, planarDissipativeForce
        )
    return nonCons
