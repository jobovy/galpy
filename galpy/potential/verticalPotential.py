import numpy

from ..backend import get_namespace, is_backend_array
from ..util import conversion
from ._repr_utils import _build_physical_output_string, _strip_physical_output_info
from .DissipativeForce import _isDissipative
from .linearPotential import linearPotential
from .planarPotential import planarPotential
from .Potential import (
    Potential,
    PotentialError,
    _check_backend_compatible,
    _check_potential_list_and_deprecate,
)


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
        # Backend-aware iff the wrapped 3D potential is (this wrapper itself is).
        self._backend_compatible = _check_backend_compatible(Pot)
        # Also transfer roSet and voSet
        self._roSet = Pot._roSet
        self._voSet = Pot._voSet
        return None

    def __repr__(self):
        # Get base potential representation
        if isinstance(self._Pot, list):  # pragma: no cover
            base_repr = "of list of potentials"
        else:
            base_repr_full = repr(self._Pot)
            # Remove physical output info from nested representation
            base_repr_full = _strip_physical_output_info(base_repr_full)
            base_repr = "of " + "".join(
                [f"\n\t{s}" for s in base_repr_full.split("\n")]
            )

        # Build physical output status string
        physical_str = _build_physical_output_string(self)
        if physical_str:
            physical_str = f"and {physical_str}"

        # Combine everything
        class_name = type(self).__name__
        return (
            f"{class_name} at R={self._R}"
            + (f", phi={self._phi} " if self._Pot.isNonAxi else " ")
            + f"{base_repr}\n{physical_str}"
        )

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
        # Follow z's actual type: a backend array (also under a forced-backend
        # context) -> the backend; a numpy/scalar z (e.g. an un-migrated parent)
        # -> numpy, so the broadcast matches what the parent will receive.
        xp = get_namespace(z) if is_backend_array(z) else numpy
        tR = self._R if not hasattr(z, "__len__") else self._R * xp.ones_like(z)
        tphi = self._phi if not hasattr(z, "__len__") else self._phi * xp.ones_like(z)
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
        # Follow z's actual type: a backend array (also under a forced-backend
        # context) -> the backend; a numpy/scalar z (e.g. an un-migrated parent)
        # -> numpy, so the broadcast matches what the parent will receive.
        xp = get_namespace(z) if is_backend_array(z) else numpy
        tR = self._R if not hasattr(z, "__len__") else self._R * xp.ones_like(z)
        tphi = self._phi if not hasattr(z, "__len__") else self._phi * xp.ones_like(z)
        return self._Pot.zforce(tR, z, phi=tphi, t=t, use_physical=False)


class _BatchedVerticalPotential(verticalPotential):
    """verticalPotential with a per-element (array) R, broadcast against z.

    Used by actionAngleAdiabatic's jax/torch backend path, which needs the
    effective vertical potential ``Phi(R_i,z)-Phi(R_i,0)`` for a whole BATCH of
    objects at once (each with its own ``R_i``), so it can reuse
    actionAngleVertical's vectorised backend Gauss-Legendre / root-find machinery.
    The parent pins a single scalar ``_R``; here ``_R`` is the ``(N,)`` batch
    array, broadcast against ``z``'s leading (object) axis on each call (``z`` is
    ``(N,)`` for calcxmax / angle limits or ``(N, n)`` for the quadrature node
    grid), and the midplane term is recomputed per call so it follows ``z``'s
    shape. Backend-agnostic: runs under numpy too (the all-backend suite forces
    numpy through the adiabatic backend branch); numpy is otherwise untouched
    because actionAngleAdiabatic only builds it on the is_backend_array-gated path.
    """

    def __init__(self, Pot, R, phi=None, t0=0.0):
        # Init via the scalar parent at a representative R (R[0]) for the
        # _Pot/_phi/amp/unit bookkeeping, then overwrite _R with the batch array
        # (_midplanePot from the parent is unused here; _evaluate recomputes it).
        xp = get_namespace(R)
        R0 = float(xp.reshape(R, (-1,))[0]) if hasattr(R, "shape") else float(R)
        verticalPotential.__init__(self, Pot, R=R0, phi=phi, t0=t0)
        self._R = R

    def _Rb(self, z):
        # Reshape the batch R to broadcast against z (leading-axis aligned).
        xp = get_namespace(z)
        R = self._R
        extra = z.ndim - R.ndim
        if extra > 0:
            R = xp.reshape(R, R.shape + (1,) * extra)
        return R

    def _evaluate(self, z, t=0.0):
        xp = get_namespace(z)
        Rb = self._Rb(z)
        tphi = self._phi * xp.ones_like(z)
        full = self._Pot(Rb, z, phi=tphi, t=t, use_physical=False)
        mid = self._Pot(Rb, 0.0 * z, phi=tphi, t=t, use_physical=False)
        return full - mid


def RZToverticalPotential(RZPot, R):
    """
    Convert a 3D azisymmetric potential to a vertical potential at a given R.

    Parameters
    ----------
    Pot : Potential instance or CompositePotential or a combined potential formed using addition (pot1+pot2+…)
        The 3D potential to convert.
    R : float or Quantity
        Galactocentric radius at which to evaluate the vertical potential.

    Returns
    -------
    verticalPotential or linearCompositePotential
        The vertical potential at (R, z, phi, t). Returns a
        linearCompositePotential when input is a list or CompositePotential
        with multiple components.

    Notes
    -----
    - 2010-07-21 - Written - Bovy (NYU)
    - 2024-12-01 - Updated to return linearCompositePotential

    """
    from .CompositePotential import CompositePotential
    from .linearCompositePotential import linearCompositePotential

    RZPot = _check_potential_list_and_deprecate(RZPot)
    try:
        conversion.get_physical(RZPot)
    except:
        raise PotentialError(
            "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a combination of such instances"
        )
    if _isDissipative(RZPot):
        raise NotImplementedError(
            "Converting dissipative forces to 1D vertical potentials is currently not supported"
        )
    R = conversion.parse_length(R, **conversion.get_physical(RZPot))
    if isinstance(RZPot, CompositePotential):
        out = []
        for pot in RZPot:
            if isinstance(pot, Potential):
                out.append(verticalPotential(pot, R))
            else:  # pragma: no cover
                raise PotentialError(
                    "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a combination of such instances"
                )
        return linearCompositePotential(out)
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
            "Input to 'RZToverticalPotential' is neither an RZPotential-instance or a combination of such instances"
        )


def toVerticalPotential(Pot, R, phi=None, t0=0.0):
    """
    Convert a 3D azisymmetric potential to a vertical potential at a given R.

    Parameters
    ----------
    Pot : Potential instance or CompositePotential or a combined potential formed using addition (pot1+pot2+…)
        The 3D potential to convert.
    R : float or Quantity
        Galactocentric radius at which to evaluate the vertical potential.
    phi : float, optional
        Azimuth at which to evaluate the vertical potential. Default is None.
    t0 : float, optional
        Time at which to evaluate the vertical potential. Default is 0.0.

    Returns
    -------
    verticalPotential or linearCompositePotential
        The vertical potential at (R, z, phi, t). Returns a
        linearCompositePotential when input is a list or CompositePotential
        with multiple components.

    Notes
    -----
    - 2010-07-21 - Written - Bovy (NYU)
    - 2024-12-01 - Updated to return linearCompositePotential

    """
    from .CompositePotential import CompositePotential
    from .linearCompositePotential import linearCompositePotential

    Pot = _check_potential_list_and_deprecate(Pot)
    try:
        conversion.get_physical(Pot)
    except:
        raise PotentialError(
            "Input to 'toVerticalPotential' is neither an Potential-instance or a combination of such instances"
        )
    if _isDissipative(Pot):
        raise NotImplementedError(
            "Converting dissipative forces to 1D vertical potentials is currently not supported"
        )
    R = conversion.parse_length(R, **conversion.get_physical(Pot))
    phi = conversion.parse_angle(phi)
    t0 = conversion.parse_time(t0, **conversion.get_physical(Pot))
    if isinstance(Pot, CompositePotential):
        out = []
        for pot in Pot:
            if isinstance(pot, Potential):
                out.append(verticalPotential(pot, R, phi=phi, t0=t0))
            else:  # pragma: no cover
                # All other cases should have been caught by the
                # conversion.get_physical test above
                raise PotentialError(
                    "Input to 'toVerticalPotential' is neither an RZPotential-instance or a combination of such instances"
                )
        return linearCompositePotential(out)
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
            "Input to 'toVerticalPotential' is neither an Potential-instance or a combination of such instances"
        )
