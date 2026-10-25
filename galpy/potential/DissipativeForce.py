###############################################################################
#   DissipativeForce.py: top-level class for forces evaluated without a potential
###############################################################################
import numpy

from ..util.conversion import physical_conversion, potential_physical_input
from .Force import Force


class DissipativeForce(Force):
    """Top-level class for forces that galpy evaluates directly, with the velocity, rather than from a potential: dynamical friction and the fictitious forces of non-inertial frames.

    Most of these forces are not conservative: they do not derive from a potential, and they need non-symplectic orbit integrators. The exception is a ``NonInertialFrameForce`` in a frame that only accelerates, whose force :math:`-\\mathbf{a}_0(t)` is minus the gradient of :math:`\\mathbf{a}_0(t)\\cdot\\mathbf{r}` (galpy does not implement that potential). The ``isConservative`` attribute says whether a force derives from a potential; ``isDissipative`` is True for every ``DissipativeForce``.
    """

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
        self.isDissipative = True
        self.isConservative = False  # a subclass can derive from a potential
        self.hasC = False
        self.hasC_dxdv = False
        # Whether the force's rectangular Jacobian (dF/dx, dF/dv) is wired in C
        # for the 3D variational equations (integrate_dxdv); subclasses that
        # implement RectDissipativeForceJacobian in C set this to True.
        self.hasC_dxdv3d = False
        self.hasC_dens = False
        # Whether the C implementation (if hasC) extends to the planar
        # (z=0) version of this force obtained through toPlanar/
        # toPlanarPotential; classes with C support that is *not* hooked up
        # in the planar C integrator should set this to False
        self.hasC_planar = True

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

    @potential_physical_input
    @physical_conversion("energy", pop=True)
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


def _isDissipative(obj):
    """
    Determine whether this combination of potentials and forces includes a DissipativeForce

    In galpy, a dissipative force is one that is evaluated directly, with the velocity, rather than from a potential; this does not mean that it does not derive from a potential (see _isConservative).

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
    from .Potential import PotentialError, flatten

    obj = flatten(obj)
    isList = isinstance(obj, list)
    try:
        if isList:
            isCons = [not p.isDissipative for p in obj]
            nonCons = not numpy.prod(numpy.array(isCons))
        else:
            nonCons = obj.isDissipative
    except AttributeError:
        raise PotentialError(
            "'isDissipative' attribute has not been set for this potential/force"
        )

    return nonCons


def _isConservative(obj):
    """
    Determine whether all forces in this combination of potentials and forces derive from a potential

    Parameters
    ----------
    obj : Potential/DissipativeForce instance or list of such instances

    Returns
    -------
    bool
        True if every force is minus the gradient of a (possibly time-dependent) potential, False otherwise

    Notes
    -----
    - Every potential is conservative, and dissipative forces are not, except a NonInertialFrameForce without rotation, whose force -a0(t) is minus the gradient of a0(t).r (galpy does not implement that potential, so the force is still dissipative in galpy's sense: it is evaluated as a force, with the velocity).
    - Symplectic integrators require conservative forces.
    - 2026-10-25 - Written - Bovy (UofT)

    """
    from .Potential import PotentialError, flatten

    obj = flatten(obj)
    try:
        if isinstance(obj, list):
            return all(p.isConservative for p in obj)
        return obj.isConservative
    except AttributeError:
        raise PotentialError(
            "'isConservative' attribute has not been set for this potential/force"
        )
