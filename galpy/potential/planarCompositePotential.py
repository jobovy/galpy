###############################################################################
#   planarCompositePotential.py: class that represents a combination of
#                                planar potentials
###############################################################################
from ..util.conversion import physical_compatible
from .DissipativeForce import _isDissipative
from .planarDissipativeForce import planarDissipativeForce
from .planarForce import planarForce
from .planarPotential import planarPotential
from .Potential import _check_c, _isNonAxi, flatten


class planarCompositePotential(planarDissipativeForce, planarPotential):
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
        # Use _check_c to determine C support based on constituent potentials
        self.hasC = _check_c(self._potlist)
        self.hasC_dxdv = _check_c(self._potlist, dxdv=True)
        self.hasC_dens = _check_c(self._potlist, dens=True)
        return None

    def __len__(self):
        """Return the number of potentials in the composite."""
        return len(self._potlist)

    def __getitem__(self, key):
        """
        Access individual potentials in the composite.

        Parameters
        ----------
        key : int or slice
            Index or slice to access.

        Returns
        -------
        planarPotential or planarCompositePotential
            If key is an integer, returns the single planarPotential at that
            index. If key is a slice, returns a planarCompositePotential
            containing the sliced potentials.

        """
        result = self._potlist[key]
        # If the result is a list (from slicing), return a
        # planarCompositePotential
        if isinstance(result, list):
            return planarCompositePotential(result)
        # Otherwise return the single potential
        return result

    def __setitem__(self, key, value):
        """
        Set individual potentials in the composite.

        Parameters
        ----------
        key : int or slice
            Index or slice to set.
        value : planarPotential or planarForce
            Potential to set at the given index.

        """
        # Validate input
        if not isinstance(value, planarForce):
            raise TypeError(
                "Can only assign planarPotential or planarForce instances "
                "to planarCompositePotential"
            )

        # Check unit compatibility with first potential (if not replacing first)
        if isinstance(key, int) and key != 0 and len(self._potlist) > 0:
            assert physical_compatible(self._potlist[0], value), (
                """Physical unit conversion parameters (ro,vo) are not """
                """compatible with existing potentials"""
            )

        # Set the item
        self._potlist[key] = value

        # Recalculate properties based on updated potential list
        self.isNonAxi = _isNonAxi(self._potlist)
        self.isDissipative = _isDissipative(self._potlist)
        self.hasC = _check_c(self._potlist)
        self.hasC_dxdv = _check_c(self._potlist, dxdv=True)
        self.hasC_dens = _check_c(self._potlist, dens=True)

    def __iter__(self):
        """Iterate over potentials in the composite."""
        return iter(self._potlist)

    def __eq__(self, other):
        """
        Check equality with another planarCompositePotential or list.

        Parameters
        ----------
        other : planarCompositePotential or list
            Object to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.

        """
        if isinstance(other, planarCompositePotential):
            return self._potlist == other._potlist
        elif isinstance(other, list):
            return self._potlist == other
        else:
            return False

    def __mul__(self, b):
        """
        Multiply a planarCompositePotential's amplitudes by a number.

        Applies the multiplication to each component potential.

        Parameters
        ----------
        b : int or float
            Number to multiply the amplitudes with.

        Returns
        -------
        planarCompositePotential
            New planarCompositePotential with each component's amplitude
            multiplied by b.

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                "Can only multiply a planarCompositePotential instance with a number"
            )
        return planarCompositePotential([force * b for force in self._potlist])

    __rmul__ = __mul__

    def __div__(self, b):
        """
        Divide a planarCompositePotential's amplitudes by a number.

        Parameters
        ----------
        b : int or float
            Number to divide the amplitudes by.

        Returns
        -------
        planarCompositePotential
            New planarCompositePotential with each component's amplitude
            divided by b.

        """
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

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

        if isinstance(other, planarCompositePotential):
            return planarCompositePotential(self._potlist + other._potlist)
        else:  # isinstance(other, (planarForce, Force))
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

    def __repr__(self):
        """
        Return a string representation of the planarCompositePotential.

        If only one potential, show its class name.
        If multiple, show a list of class names.
        """
        if len(self._potlist) == 1:
            pot = self._potlist[0]
            return (
                f"<planarCompositePotential: single potential "
                f"({pot.__class__.__name__})>"
            )
        else:
            pot_reprs = [repr(pot) for pot in self._potlist]
            return (
                f"<planarCompositePotential: {len(self._potlist)} potentials "
                f"({', '.join(pot_reprs)})>"
            )
