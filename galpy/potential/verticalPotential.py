import numpy

from ..util import conversion
from .DissipativeForce import _isDissipative
from .linearPotential import linearPotential
from .planarPotential import planarPotential
from .Potential import Potential, PotentialError, flatten


class verticalPotential(linearPotential):
    """Class that represents a vertical potential derived from a 3D Potential:
    phi(z,t;R,phi)= phi(R,z,phi,t)-phi(R,0.,phi,t0)"""

    def __init__(self, Pot, R=1.0, phi=None, t0=0.0):
        """
        Initialize a verticalPotential instance.

        Parameters
        ----------
        Pot : Potential instance
            The potential to convert to a vertical potential.
        R : float, optional
            Galactocentric radius at which to create the vertical potential. Default is 1.0.
        phi : float, optional
            Galactocentric azimuth at which to create the vertical potential (rad); necessary for non-axisymmetric potentials. Default is None.
        t0 : float, optional
            Time at which to create the vertical potential. Default is 0.0.

        Returns
        -------
        verticalPotential instance

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)
        - 2018-10-07 - Added support for non-axi potentials - Bovy (UofT)
        - 2019-08-19 - Added support for time-dependent potentials - Bovy (UofT)
        """
        linearPotential.__init__(self, amp=1.0, ro=Pot._ro, vo=Pot._vo)
        self._Pot = Pot
        self._R = R
        if phi is None:
            if Pot.isNonAxi:
                raise PotentialError(
                    "The Potential instance to convert to a verticalPotential is non-axisymmetric, but you did not provide phi"
                )
            self._phi = 0.0
        else:
            self._phi = phi
        self._midplanePot = self._Pot(
            self._R, 0.0, phi=self._phi, t=t0, use_physical=False
        )
        self.hasC = Pot.hasC
        # Also transfer roSet and voSet
        self._roSet = Pot._roSet
        self._voSet = Pot._voSet
        return None

    def _evaluate(self, z, t=0.0):
        """
        Evaluate the potential.

        Parameters
        ----------
        z : float
            Height above the midplane.
        t : float, optional
            Time at which to evaluate the potential. Default is 0.0.

        Returns
        -------
        float
            The potential at (R, z, phi, t) - phi(R, 0., phi, t0).

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)
        - 2018-10-07 - Added support for non-axi potentials - Bovy (UofT)
        - 2019-08-19 - Added support for time-dependent potentials - Bovy (UofT)
        """
        tR = self._R if not hasattr(z, "__len__") else self._R * numpy.ones_like(z)
        tphi = (
            self._phi if not hasattr(z, "__len__") else self._phi * numpy.ones_like(z)
        )
        return self._Pot(tR, z, phi=tphi, t=t, use_physical=False) - self._midplanePot

    def _force(self, z, t=0.0):
        """
        Evaluate the force.

        Parameters
        ----------
        z : float
            Height above the midplane.
        t : float, optional
            Time at which to evaluate the force. Default is 0.0.

        Returns
        -------
        float
            The vertical force at (R, z, phi, t).

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)
        - 2018-10-07 - Added support for non-axi potentials - Bovy (UofT)
        - 2019-08-19 - Added support for time-dependent potentials - Bovy (UofT)

        """
        tR = self._R if not hasattr(z, "__len__") else self._R * numpy.ones_like(z)
        tphi = (
            self._phi if not hasattr(z, "__len__") else self._phi * numpy.ones_like(z)
        )
        return self._Pot.zforce(tR, z, phi=tphi, t=t, use_physical=False)


def RZToverticalPotential(RZPot, R):
    """
    Convert a 3D azisymmetric potential to a vertical potential at a given R.

    Parameters
    ----------
    Pot : Potential instance
        The 3D potential to convert.
    R : float or Quantity
        Galactocentric radius at which to evaluate the vertical potential.

    Returns
    -------
    verticalPotential instance
        The vertical potential at (R, z, phi, t).

    Notes
    -----
    - 2010-07-21 - Written - Bovy (NYU)

    """
    RZPot = flatten(RZPot)
    if _isDissipative(RZPot):
        raise NotImplementedError(
            "Converting dissipative forces to 1D vertical potentials is currently not supported"
        )
    try:
        conversion.get_physical(RZPot)
    except:
        raise PotentialError(
            "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances"
        )
    R = conversion.parse_length(R, **conversion.get_physical(RZPot))
    if isinstance(RZPot, list):
        out = []
        for pot in RZPot:
            if isinstance(pot, linearPotential):
                out.append(pot)
            elif isinstance(pot, Potential):
                out.append(verticalPotential(pot, R))
            elif isinstance(pot, planarPotential):
                raise PotentialError(
                    "Input to 'RZToverticalPotential' cannot be a planarPotential"
                )
            else:
                raise PotentialError(
                    "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances"
                )
        return out
    elif isinstance(RZPot, Potential):
        return verticalPotential(RZPot, R)
    elif isinstance(RZPot, linearPotential):
        return RZPot
    elif isinstance(RZPot, planarPotential):
        raise PotentialError(
            "Input to 'RZToverticalPotential' cannot be a planarPotential"
        )
    else:  # pragma: no cover
        # All other cases should have been caught by the
        # conversion.get_physical test above
        raise PotentialError(
            "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a list of such instances"
        )


def toVerticalPotential(Pot, R, phi=None, t0=0.0):
    """
    Convert a 3D azisymmetric potential to a vertical potential at a given R.

    Parameters
    ----------
    Pot : Potential instance
        The 3D potential to convert.
    R : float or Quantity
        Galactocentric radius at which to evaluate the vertical potential.
    phi : float, optional
        Azimuth at which to evaluate the vertical potential. Default is None.
    t0 : float, optional
        Time at which to evaluate the vertical potential. Default is 0.0.

    Returns
    -------
    verticalPotential instance
        The vertical potential at (R, z, phi, t).

    Notes
    -----
    - 2010-07-21 - Written - Bovy (NYU)

    """
    Pot = flatten(Pot)
    if _isDissipative(Pot):
        raise NotImplementedError(
            "Converting dissipative forces to 1D vertical potentials is currently not supported"
        )
    try:
        conversion.get_physical(Pot)
    except:
        raise PotentialError(
            "Input to 'toVerticalPotential' is neither an Potential-instance or a list of such instances"
        )
    R = conversion.parse_length(R, **conversion.get_physical(Pot))
    phi = conversion.parse_angle(phi)
    t0 = conversion.parse_time(t0, **conversion.get_physical(Pot))
    if isinstance(Pot, list):
        out = []
        for pot in Pot:
            if isinstance(pot, linearPotential):
                out.append(pot)
            elif isinstance(pot, Potential):
                out.append(verticalPotential(pot, R, phi=phi, t0=t0))
            elif isinstance(pot, planarPotential):
                raise PotentialError(
                    "Input to 'toVerticalPotential' cannot be a planarPotential"
                )
            else:  # pragma: no cover
                # All other cases should have been caught by the
                # conversion.get_physical test above
                raise PotentialError(
                    "Input to 'toVerticalPotential' is neither an RZPotential-instance or a list of such instances"
                )
        return out
    elif isinstance(Pot, Potential):
        return verticalPotential(Pot, R, phi=phi, t0=t0)
    elif isinstance(Pot, linearPotential):
        return Pot
    elif isinstance(Pot, planarPotential):
        raise PotentialError(
            "Input to 'toVerticalPotential' cannot be a planarPotential"
        )
    else:  # pragma: no cover
        # All other cases should have been caught by the
        # conversion.get_physical test above
        raise PotentialError(
            "Input to 'toVerticalPotential' is neither an Potential-instance or a list of such instances"
        )
