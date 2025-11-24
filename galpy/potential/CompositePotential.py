###############################################################################
#   CompositePotential.py: class that represents a combination of potentials
###############################################################################
from ..util.conversion import physical_compatible
from .DissipativeForce import DissipativeForce, _isDissipative
from .Potential import Potential, _check_c, _isNonAxi, flatten


class CompositePotential(DissipativeForce, Potential):
    """Class that represents a combination of potentials and allows them to be
    called with method functions in the same way as individual potentials."""

    def __init__(self, *args, ro=None, vo=None):
        """
        Initialize a CompositePotential.

        Parameters
        ----------
        *args : Force, Potential, or list of such instances
            Forces/Potentials to combine. Can be individual forces and potentials, lists, or nested lists.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the first Force.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the first Force.

        Notes
        -----
        - 2024-11-24 - Written - Bovy (UofT)

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

        # Initialize Potential with amp=1.0 (amplitude is in individual potentials)
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)

        # Override _roSet/_voSet based on first potential's settings
        # (unless explicitly provided)
        self._roSet = roSet
        self._voSet = voSet

        # Set properties based on constituent potentials using existing functions
        self.isNonAxi = _isNonAxi(self._potlist)
        self.isDissipative = _isDissipative(self._potlist)
        # Determine the dimensionality
        self.dim = min([p.dim for p in self._potlist])
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
        Potential or CompositePotential
            If key is an integer, returns the single Potential at that index.
            If key is a slice, returns a CompositePotential containing the sliced potentials.

        Notes
        -----
        - 2024-11-26 - Updated to return CompositePotential for slices - Copilot

        """
        result = self._potlist[key]
        # If the result is a list (from slicing), return a CompositePotential
        if isinstance(result, list):
            return CompositePotential(result)
        # Otherwise return the single potential
        return result

    def __setitem__(self, key, value):
        """
        Set individual potentials in the composite.

        Parameters
        ----------
        key : int or slice
            Index or slice to set.
        value : Potential or Force
            Potential to set at the given index.

        Notes
        -----
        - 2024-11-25 - Written - Copilot

        """
        from .Force import Force

        # Validate input
        if not isinstance(value, Force):
            raise TypeError(
                "Can only assign Potential or Force instances to CompositePotential"
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
        Check equality with another CompositePotential or list.

        Parameters
        ----------
        other : CompositePotential or list
            Object to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.

        """
        if isinstance(other, CompositePotential):
            return self._potlist == other._potlist
        elif isinstance(other, list):
            return self._potlist == other
        else:
            return False

    def __mul__(self, b):
        """
        Multiply a CompositePotential's amplitudes by a number.

        Applies the multiplication to each component potential.

        Parameters
        ----------
        b : int or float
            Number to multiply the amplitudes with.

        Returns
        -------
        CompositePotential
            New CompositePotential with each component's amplitude multiplied by b.

        Notes
        -----
        - 2024-11-24 - Written - Copilot

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                "Can only multiply a CompositePotential instance with a number"
            )
        return CompositePotential([force * b for force in self._potlist])

    __rmul__ = __mul__

    def __div__(self, b):
        """
        Divide a CompositePotential's amplitudes by a number.

        Parameters
        ----------
        b : int or float
            Number to divide the amplitudes by.

        Returns
        -------
        CompositePotential
            New CompositePotential with each component's amplitude divided by b.

        """
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __add__(self, other):
        """
        Add another potential or CompositePotential to this one.

        Parameters
        ----------
        other : Potential or CompositePotential
            Potential(s) to add.

        Returns
        -------
        CompositePotential
            New CompositePotential with combined potentials.

        """
        from .Force import Force

        # Check type first before checking unit compatibility
        if not isinstance(other, (Force, CompositePotential)):
            raise TypeError(
                "Can only add Potential or CompositePotential to CompositePotential"
            )

        # Check unit compatibility
        assert physical_compatible(self, other), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )

        if isinstance(other, CompositePotential):
            return CompositePotential(self._potlist + other._potlist)
        else:  # isinstance(other, Force)
            return CompositePotential(self._potlist + [other])

    def _evaluate(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the potential at (R,z,phi,t).

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Potential at (R,z,phi,t).

        """
        return sum(
            pot._call_nodecorator(R, z, phi=phi, t=t)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _Rforce(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate the cylindrical radial force.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).
        v : numpy.ndarray, optional
            Current velocity (required for dissipative forces).

        Returns
        -------
        float
            Radial force at (R,z,phi,t).

        """
        return sum(
            (
                pot._Rforce_nodecorator(R, z, phi=phi, t=t, v=v)
                if pot.isDissipative
                else pot._Rforce_nodecorator(R, z, phi=phi, t=t)
            )
            for pot in self._potlist
        )

    def _zforce(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate the vertical force.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).
        v : numpy.ndarray, optional
            Current velocity (required for dissipative forces).

        Returns
        -------
        float
            Vertical force at (R,z,phi,t).

        """
        return sum(
            (
                pot._zforce_nodecorator(R, z, phi=phi, t=t, v=v)
                if pot.isDissipative
                else pot._zforce_nodecorator(R, z, phi=phi, t=t)
            )
            for pot in self._potlist
        )

    def _phitorque(self, R, z, phi=0.0, t=0.0, v=None):
        """
        Evaluate the azimuthal torque.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).
        v : numpy.ndarray, optional
            Current velocity (required for dissipative forces).

        Returns
        -------
        float
            Azimuthal torque at (R,z,phi,t).

        """
        return sum(
            (
                pot._phitorque_nodecorator(R, z, phi=phi, t=t, v=v)
                if pot.isDissipative
                else pot._phitorque_nodecorator(R, z, phi=phi, t=t)
            )
            for pot in self._potlist
        )

    def _dens(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the density.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Density at (R,z,phi,t).

        """
        return sum(
            pot.dens(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _mass(self, R, z=None, t=0.0):
        """
        Evaluate the mass enclosed.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float, optional
            Vertical height (default: None).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Mass enclosed.

        """
        return sum(
            pot.mass(R, z=z, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _R2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second radial derivative at (R,z,phi,t).

        """
        return sum(
            pot.R2deriv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _z2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second vertical derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second vertical derivative at (R,z,phi,t).

        """
        return sum(
            pot.z2deriv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _Rzderiv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed R,z derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Mixed R,z derivative at (R,z,phi,t).

        """
        return sum(
            pot.Rzderiv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _phi2deriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the second azimuthal derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second azimuthal derivative at (R,z,phi,t).

        """
        return sum(
            pot.phi2deriv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _Rphideriv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed radial, azimuthal derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Mixed radial, azimuthal derivative at (R,z,phi,t).

        """
        return sum(
            pot.Rphideriv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def _phizderiv(self, R, z, phi=0.0, t=0.0):
        """
        Evaluate the mixed azimuthal, vertical derivative.

        Parameters
        ----------
        R : float
            Cylindrical Galactocentric radius.
        z : float
            Vertical height.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Mixed azimuthal, vertical derivative at (R,z,phi,t).

        """
        return sum(
            pot.phizderiv(R, z, phi=phi, t=t, use_physical=False)
            for pot in self._potlist
            if not pot.isDissipative
        )

    def nemo_accname(self):
        """
        Return the accname potential name for use of this CompositePotential with NEMO.
        """
        return "+".join(pot.nemo_accname() for pot in self._potlist)

    def nemo_accpars(self, vo, ro):
        """
        Return the accpars potential parameters for use of this CompositePotential with NEMO.
        """
        return "#".join(pot.nemo_accpars(vo, ro) for pot in self._potlist)

    def __repr__(self):
        """
        Return a string representation of the CompositePotential.

        If only one potential, show its class name.
        If multiple, show a list of class names.
        """
        if len(self._potlist) == 1:
            pot = self._potlist[0]
            return f"<CompositePotential: single potential ({pot.__class__.__name__})>"
        else:
            pot_reprs = [repr(pot) for pot in self._potlist]
            return (
                f"<CompositePotential: {len(self._potlist)} potentials "
                f"({', '.join(pot_reprs)})>"
            )
