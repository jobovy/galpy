###############################################################################
#   linearCompositePotential.py: class that represents a combination of
#                                linear (1D) potentials
###############################################################################
from ..util.conversion import physical_compatible
from .baseCompositePotential import baseCompositePotential
from .linearPotential import linearPotential
from .Potential import _check_c, flatten


class linearCompositePotential(baseCompositePotential, linearPotential):
    """Class that represents a combination of linear (1D) potentials and allows
    them to be called with method functions in the same way as individual linear
    potentials."""

    def __init__(self, *args, ro=None, vo=None):
        """
        Initialize a linearCompositePotential.

        Parameters
        ----------
        *args : linearPotential or a combined potential formed using addition (pot1+pot2+…)
            Linear potentials to combine. Can be individual potentials,
            lists, or nested lists.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from
            the first linearPotential.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from
            the first linearPotential.

        Notes
        -----
        - 2024-12-01 - Written

        """
        # Flatten the input arguments into a list of potentials
        if len(args) == 1 and isinstance(args[0], list):
            pot_list = args[0]
        else:
            pot_list = list(args)
        # Flatten nested lists
        self._potlist = flatten(pot_list)
        # Check that unit systems of all potentials are compatible
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

        # Initialize linearPotential with amp=1.0 (amplitude is in individual
        # potentials)
        linearPotential.__init__(self, amp=1.0, ro=ro, vo=vo)

        # Override _roSet/_voSet based on first potential's settings
        # (unless explicitly provided)
        self._roSet = roSet
        self._voSet = voSet

        # Use _check_c to determine C support based on constituent potentials
        self.hasC = _check_c(self._potlist)
        self.hasC_dxdv = _check_c(self._potlist, dxdv=True)
        self.hasC_dens = _check_c(self._potlist, dens=True)
        return None

    def __add__(self, other):
        """
        Add another linear potential or linearCompositePotential to this one.

        Parameters
        ----------
        other : linearPotential or linearCompositePotential
            Potential(s) to add.

        Returns
        -------
        linearCompositePotential
            New linearCompositePotential with combined potentials.

        """
        # Check type first before checking unit compatibility
        if not isinstance(other, (linearPotential, linearCompositePotential)):
            raise TypeError(
                "Can only add linearPotential or linearCompositePotential to "
                "linearCompositePotential"
            )

        # Check unit compatibility
        assert physical_compatible(self, other), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )

        if isinstance(other, linearCompositePotential):
            return linearCompositePotential(self._potlist + other._potlist)
        else:  # isinstance(other, linearPotential)
            return linearCompositePotential(self._potlist + [other])

    def _evaluate(self, x, t=0.0):
        """
        Evaluate the potential at (x,t).

        Parameters
        ----------
        x : float
            Position.
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Potential at (x,t).

        """
        return sum(pot._call_nodecorator(x, t=t) for pot in self._potlist)

    def _force(self, x, t=0.0):
        """
        Evaluate the force at (x,t).

        Parameters
        ----------
        x : float
            Position.
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Force at (x,t).

        """
        return sum(pot._force_nodecorator(x, t=t) for pot in self._potlist)
