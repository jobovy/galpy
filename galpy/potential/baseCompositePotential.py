# Common methods for CompositePotential-type classes
from ..util.conversion import physical_compatible
from .DissipativeForce import _isDissipative
from .Potential import _check_c, _isNonAxi


class baseCompositePotential:
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
        (planar)Potential or (planar)CompositePotential
            If key is an integer, returns the single (planar)Potential at that
            index. If key is a slice, returns a (planar)CompositePotential
            containing the sliced potentials.

        """
        result = self._potlist[key]
        # If the result is a list (from slicing), return a
        # (planar)CompositePotential
        if isinstance(result, list):
            return self.__class__(result)
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
        # Validate input
        if not isinstance(value, self.__class__.__mro__[-2]):
            raise TypeError(
                f"Can only assign {self.__class__.__mro__[-2].__name__} instances to a {self.__class__.__name__}"
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
        Check equality with another (planar)CompositePotential or list.

        Parameters
        ----------
        other : (planar)CompositePotential or list
            Object to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.

        """
        if isinstance(other, self.__class__):
            return self._potlist == other._potlist
        elif isinstance(other, list):
            return self._potlist == other
        else:
            return False

    def __mul__(self, b):
        """
        Multiply a (planar)CompositePotential's amplitudes by a number.

        Applies the multiplication to each component potential.

        Parameters
        ----------
        b : int or float
            Number to multiply the amplitudes with.

        Returns
        -------
        (planar)CompositePotential
            New (planar)CompositePotential with each component's amplitude
            multiplied by b.

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                f"Can only multiply a {self.__class__.__name__} instance with a number"
            )
        return self.__class__([force * b for force in self._potlist])

    __rmul__ = __mul__

    def __div__(self, b):
        """
        Divide a (planar)CompositePotential's amplitudes by a number.

        Parameters
        ----------
        b : int or float
            Number to divide the amplitudes by.

        Returns
        -------
        (planar)CompositePotential
            New CompositePotential with each component's amplitude divided by b.

        """
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __repr__(self):
        """
        Return a string representation of the (planar)CompositePotential.

        If only one potential, show its class name.
        If multiple, show a list of class names.
        """
        if len(self._potlist) == 1:
            pot = self._potlist[0]
            return (
                f"<{self.__class__.__name__}: single potential "
                f"({pot.__class__.__name__})>"
            )
        else:
            pot_reprs = [repr(pot) for pot in self._potlist]
            return (
                f"<{self.__class__.__name__}: {len(self._potlist)} potentials "
                f"({', '.join(pot_reprs)})>"
            )
