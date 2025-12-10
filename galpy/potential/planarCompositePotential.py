###############################################################################
#   planarCompositePotential.py: class that represents a combination of
#                                planar potentials
###############################################################################
from ..util.conversion import physical_compatible
from .baseCompositePotential import baseCompositePotential
from .DissipativeForce import _isDissipative
from .planarDissipativeForce import planarDissipativeForce
from .planarForce import planarForce
from .planarPotential import planarPotential
from .Potential import _check_c, _isNonAxi, flatten


class planarCompositePotential(
    baseCompositePotential, planarDissipativeForce, planarPotential
):
    """Class that represents a combination of planar potentials and allows them
    to be called with method functions in the same way as individual planar
    potentials."""

    def __init__(self, *args, ro=None, vo=None):
        """
        Initialize a planarCompositePotential.

        Parameters
        ----------
        *args : planarForce, planarPotential, or list of such instances
            Planar forces/potentials to combine. Can be individual forces and
            potentials, lists, or nested lists.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from
            the first planarForce.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from
            the first planarForce.

        Notes
        -----
        - 2024-11-27 - Written - Copilot

        """
        # Flatten the input arguments into a list of potentials
        if len(args) == 1 and isinstance(args[0], list):
            pot_list = args[0]
        else:
            pot_list = list(args)
        # Flatten nested lists
        self._potlist = flatten(pot_list)
        
        # Check that all potentials are 2D FIRST (before calling _isNonAxi)
        for pot in self._potlist:
            if hasattr(pot, "dim") and pot.dim != 2:
                raise ValueError(
                    f"All potentials in planarCompositePotential must be 2D; "
                    f"got potential with dimensionality {pot.dim}"
                )
        
        # Check that unit systems of all forces are compatible
        if len(self._potlist) > 1:
            for pot in self._potlist[1:]:
                assert physical_compatible(self._potlist[0], pot), (
                    """Physical unit conversion parameters (ro,vo) are not """
                    """compatible between potentials to be combined"""
                )

        # Get ro/vo and _roSet/_voSet from first potential (standard behavior)
        first_pot = self._potlist[0] if len(self._potlist) > 0 else None
        if ro is None and first_pot is not None:
            ro = first_pot._ro
            roSet = first_pot._roSet
        else:
            roSet = ro is not None
        if vo is None and first_pot is not None:
            vo = first_pot._vo
            voSet = first_pot._voSet
        else:
            voSet = vo is not None

        # Initialize planarPotential with amp=1.0 (amplitude is in individual
        # potentials)
        planarPotential.__init__(self, amp=1.0, ro=ro, vo=vo)

        # Override _roSet/_voSet based on first potential's settings
        # (unless explicitly provided)
        self._roSet = roSet
        self._voSet = voSet

        # Set properties based on constituent potentials using existing
        # functions
        self.isNonAxi = _isNonAxi(self._potlist)
        self.isDissipative = _isDissipative(self._potlist)
        # Set dimensionality to 2 (already checked above)
        self.dim = 2
        # Use _check_c to determine C support based on constituent potentials
        self.hasC = _check_c(self._potlist)
        self.hasC_dxdv = _check_c(self._potlist, dxdv=True)
        self.hasC_dens = _check_c(self._potlist, dens=True)
        return None

    def __add__(self, other):
        """
        Add another planar potential or planarCompositePotential to this one.

        Parameters
        ----------
        other : planarForce or planarCompositePotential
            Potential(s) to add.

        Returns
        -------
        planarCompositePotential
            New planarCompositePotential with combined potentials.

        """
        from .Force import Force

        # Check type first before checking unit compatibility
        if not isinstance(other, (Force, planarForce, planarCompositePotential)):
            raise TypeError(
                "Can only add planarPotential, planarCompositePotential, or "
                "3D Force to planarCompositePotential"
            )

        # Check unit compatibility
        assert physical_compatible(self, other), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )

        # If adding a 3D Force, convert it to planar
        if isinstance(other, Force) and hasattr(other, "dim") and other.dim == 3:
            return planarCompositePotential(self._potlist + [other.toPlanar()])

        if isinstance(other, planarCompositePotential):
            return planarCompositePotential(self._potlist + other._potlist)
        else:  # isinstance(other, planarForce)
            return planarCompositePotential(self._potlist + [other])

    def _evaluate(self, R, phi=0.0, t=0.0):
        """
        Evaluate the potential at (R,phi,t).

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Potential at (R,phi,t).

        """
        return sum(
            pot._call_nodecorator(R, phi=phi, t=t)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _Rforce(self, R, phi=0.0, t=0.0, v=None):
        """
        Evaluate the cylindrical radial force.

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).
        v : numpy.ndarray, optional
            Current velocity (required for dissipative forces).

        Returns
        -------
        float
            Radial force at (R,phi,t).

        """
        return sum(
            (
                pot._Rforce_nodecorator(R, phi=phi, t=t, v=v)
                if pot.isDissipative
                else pot._Rforce_nodecorator(R, phi=phi, t=t)
            )
            for pot in self._potlist
        )

    def _phitorque(self, R, phi=0.0, t=0.0, v=None):
        """
        Evaluate the azimuthal torque.

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).
        v : numpy.ndarray, optional
            Current velocity (required for dissipative forces).

        Returns
        -------
        float
            Azimuthal torque at (R,phi,t).

        """
        return sum(
            (
                pot._phitorque_nodecorator(R, phi=phi, t=t, v=v)
                if pot.isDissipative
                else pot._phitorque_nodecorator(R, phi=phi, t=t)
            )
            for pot in self._potlist
        )

    def _R2deriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative.

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second radial derivative at (R,phi,t).

        """
        return sum(
            pot.R2deriv(R, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _phi2deriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the second azimuthal derivative.

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second azimuthal derivative at (R,phi,t).

        """
        return sum(
            pot.phi2deriv(R, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _Rphideriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the mixed radial, azimuthal derivative.

        Parameters
        ----------
        R : float
            Cylindrical radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Mixed radial, azimuthal derivative at (R,phi,t).

        """
        return sum(
            pot.Rphideriv(R, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )
