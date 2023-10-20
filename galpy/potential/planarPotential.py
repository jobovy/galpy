import os
import pickle

import numpy
from scipy import integrate

from ..util import config, conversion, plot
from ..util.conversion import (
    physical_compatible,
    physical_conversion,
    potential_physical_input,
)
from .DissipativeForce import DissipativeForce, _isDissipative
from .planarDissipativeForce import planarDissipativeForceFromFullDissipativeForce
from .planarForce import planarForce
from .plotEscapecurve import _INF, plotEscapecurve
from .plotRotcurve import plotRotcurve
from .Potential import (
    Potential,
    PotentialError,
    flatten,
    lindbladR,
    potential_positional_arg,
)


class planarPotential(planarForce):
    r"""Class representing 2D (R,\phi) potentials"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        planarForce.__init__(self, amp=amp, ro=ro, vo=vo)

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def __call__(self, R, phi=0.0, t=0.0, dR=0, dphi=0):
        """
        Evaluate the potential.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default: 0).
        t : float or Quantity, optional
            Time (default: 0).
        dR : int, optional
            Order of radial derivative (default: 0).
        dphi : int, optional
            Order of azimuthal derivative (default: 0).

        Returns
        -------
        float or Quantity
            Potential at (R, phi, t) or its derivative.

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(R, phi=phi, t=t, dR=dR, dphi=dphi)

    def _call_nodecorator(self, R, phi=0.0, t=0.0, dR=0, dphi=0):
        # Separate, so it can be used during orbit integration
        if dR == 0 and dphi == 0:
            try:
                return self._amp * self._evaluate(R, phi=phi, t=t)
            except AttributeError:  # pragma: no cover
                raise PotentialError(
                    "'_evaluate' function not implemented for this potential"
                )
        elif dR == 1 and dphi == 0:
            return -self.Rforce(R, phi=phi, t=t, use_physical=False)
        elif dR == 0 and dphi == 1:
            return -self.phitorque(R, phi=phi, t=t, use_physical=False)
        elif dR == 2 and dphi == 0:
            return self.R2deriv(R, phi=phi, t=t, use_physical=False)
        elif dR == 0 and dphi == 2:
            return self.phi2deriv(R, phi=phi, t=t, use_physical=False)
        elif dR == 1 and dphi == 1:
            return self.Rphideriv(R, phi=phi, t=t, use_physical=False)
        elif dR != 0 or dphi != 0:
            raise NotImplementedError(
                "Higher-order derivatives not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rforce(self, R, phi=0.0, t=0.0):
        r"""
        Evaluate the radial force.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default 0.0).
        t : float  or Quantity, optional
            Time (default 0.0).

        Returns
        -------
        float or Quantity
            Cylindrical radial force F_R(R, (\phi, t)).

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return self._Rforce_nodecorator(R, phi=phi, t=t)

    def _Rforce_nodecorator(self, R, phi=0.0, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._Rforce(R, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_Rforce' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def phitorque(self, R, phi=0.0, t=0.0):
        """
        Evaluate the azimuthal torque = - d Phi / d phi.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default 0.0).
        t : float or Quantity, optional
            Time (default 0.0).

        Returns
        -------
        float or Quantity
            Azimuthal torque tau_phi(R, (phi, t)).

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return self._phitorque_nodecorator(R, phi=phi, t=t)

    def _phitorque_nodecorator(self, R, phi=0.0, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._phitorque(R, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_phitorque' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("forcederivative", pop=True)
    def R2deriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default 0.0).
        t : float or Quantity, optional
            Time (default 0.0).

        Returns
        -------
        float or Quantity
            Second radial derivative d2phi/dR2 of the potential.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)
        """
        try:
            return self._amp * self._R2deriv(R, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_R2deriv' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def phi2deriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the second azimuthal derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default 0.0).
        t : float or Quantity, optional
            Time (default 0.0).

        Returns
        -------
        float or Quantity
            Second azimuthal derivative d2phi/dazi2 of the potential.

        Notes
        -----
        - 2014-04-06 - Written - Bovy (IAS)

        """
        try:
            return self._amp * self._phi2deriv(R, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_phi2deriv' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def Rphideriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the mixed radial and azimuthal derivative.

        Parameters
        ----------
        R : float or Quantity
            Cylindrical radius.
        phi : float or Quantity, optional
            Azimuth (default 0.0).
        t : float or Quantity, optional
            Time (default 0.0).

        Returns
        -------
        float or Quantity
            Mixed radial and azimuthal derivative d2phi/dR dazi of the potential.

        Notes
        -----
        - 2014-05-21 - Written - Bovy (IAS)

        """
        try:
            return self._amp * self._Rphideriv(R, phi=phi, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_Rphideriv' function not implemented for this potential"
            )

    def plot(self, *args, **kwargs):
        """
        Plot the potential.

        Parameters
        ----------
        Rrange : float or Quantity, optional
            Range (can be Quantity).
        grid : int, optional
            Number of points to plot.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        *args : list
            Arguments to be passed to `galpy.util.plot.plot`.
        **kwargs : dict
            Keyword arguments to be passed to `galpy.util.plot.plot`.

        Returns
        -------
        plot

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return plotplanarPotentials(self, *args, **kwargs)


class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        self.isNonAxi = False

    def _phitorque(self, R, phi=0.0, t=0.0):
        return 0.0

    def _phi2deriv(self, R, phi=0.0, t=0.0):  # pragma: no cover
        """
        Evaluate the second azimuthal derivative for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        phi : float, optional
            Azimuth.
        t : float, optional
            Time.

        Returns
        -------
        float
            The second azimuthal derivative.

        Notes
        -----
        - 2011-10-17 - Written - Bovy (IAS)

        """
        return 0.0

    def _Rphideriv(self, R, phi=0.0, t=0.0):  # pragma: no cover
        """
        Evaluate the radial+azimuthal derivative for this potential.

        Parameters
        ----------
        R : float
            Galactocentric cylindrical radius.
        phi : float, optional
            Azimuth.
        t : float, optional
            Time.

        Returns
        -------
        float
            The radial+azimuthal derivative.

        Notes
        -----
        - 2011-10-17 - Written - Bovy (IAS)

        """
        return 0.0

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def vcirc(self, R, phi=None, t=0.0):
        """
        Calculate the circular velocity at R in potential Pot.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        phi : float or Quantity, optional
            Azimuth to use for non-axisymmetric potentials.
        t : float or Quantity, optional
            Time (default: 0.0)

        Returns
        -------
        float or Quantity
            Circular rotation velocity.

        Notes
        -----
        - 2011-10-09 - Written - Bovy (IAS)
        - 2016-06-15 - Added phi= keyword for non-axisymmetric potential - Bovy (UofT)

        """
        return numpy.sqrt(R * -self.Rforce(R, phi=phi, t=t, use_physical=False))

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def omegac(self, R, t=0.0):
        """
        Calculate the circular angular speed at R in potential Pot.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time (default: 0.0)

        Returns
        -------
        float or Quantity
            Circular angular speed.

        Notes
        -----
        - Written on 2011-10-09 by Bovy (IAS).

        """
        return numpy.sqrt(-self.Rforce(R, t=t, use_physical=False) / R)

    @potential_physical_input
    @physical_conversion("frequency", pop=True)
    def epifreq(self, R, t=0.0):
        """
        Calculate the epicycle frequency at R in this potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            Epicycle frequency.

        Notes
        -----
        - Written on 2011-10-09 by Bovy (IAS).

        """
        return numpy.sqrt(
            self.R2deriv(R, t=t, use_physical=False)
            - 3.0 / R * self.Rforce(R, t=t, use_physical=False)
        )

    @physical_conversion("position", pop=True)
    def lindbladR(self, OmegaP, m=2, t=0.0, **kwargs):
        """
        Calculate the radius of a Lindblad resonance.

        Parameters
        ----------
        OmegaP : float or Quantity
            Pattern speed.
        m : int or str, optional
            Order of the resonance (as in m(O-Op)=kappa (negative m for outer)).
            Use m='corotation' for corotation.
        t : float or Quantity, optional
            Time (default: 0.0).
        **kwargs
            Additional arguments passed to `scipy.optimize.brentq`.

        Returns
        -------
        float or Quantity or None
            Radius of Lindblad resonance. None if there is no resonance.

        Notes
        -----
        - Written on 2011-10-09 by Bovy (IAS).

        """
        OmegaP = conversion.parse_frequency(OmegaP, ro=self._ro, vo=self._vo)
        return lindbladR(self, OmegaP, m=m, t=t, use_physical=False, **kwargs)

    @potential_physical_input
    @physical_conversion("velocity", pop=True)
    def vesc(self, R, t=0.0):
        """
        Calculate the escape velocity at R for the potential.

        Parameters
        ----------
        R : float or Quantity
            Galactocentric radius.
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            Escape velocity.

        Notes
        -----
        - Written on 2011-10-09 by Bovy (IAS).

        """
        return numpy.sqrt(
            2.0
            * (self(_INF, t=t, use_physical=False) - self(R, t=t, use_physical=False))
        )

    def plotRotcurve(self, *args, **kwargs):
        """
        Plot the rotation curve for this potential.

        Parameters
        ----------
        Rrange : list or Quantity, optional
            Range to plot.
        grid : int, optional
            Number of points to plot.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        *args, **kwargs :
            Arguments and keyword arguments for `galpy.util.plot.plot`.

        Returns
        -------
        matplotlib plot
            Plot to output device.

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return plotRotcurve(self, *args, **kwargs)

    def plotEscapecurve(self, *args, **kwargs):
        """
        Plot the escape velocity curve for this potential.

        Parameters
        ----------
        Rrange : list or Quantity, optional
            Range to plot.
        grid : int, optional
            Number of points to plot.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).
        *args, **kwargs :
            Arguments and keyword arguments for `galpy.util.plot.plot`.

        Returns
        -------
        matplotlib plot
            Plot to output device.

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        return plotEscapecurve(self, *args, **kwargs)


class planarPotentialFromRZPotential(planarAxiPotential):
    """Class that represents an axisymmetic planar potential derived from a
    RZPotential"""

    def __init__(self, RZPot):
        """
        Initialize.

        Parameters
        ----------
        RZPot : RZPotential instance
            RZPotential instance.

        Returns
        -------
        planarAxiPotential instance

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        planarAxiPotential.__init__(self, amp=1.0, ro=RZPot._ro, vo=RZPot._vo)
        # Also transfer roSet and voSet
        self._roSet = RZPot._roSet
        self._voSet = RZPot._voSet
        self._Pot = RZPot
        self.hasC = RZPot.hasC
        self.hasC_dxdv = RZPot.hasC_dxdv
        self.hasC_dens = RZPot.hasC_dens
        return None

    def _evaluate(self, R, phi=0.0, t=0.0):
        """
        Evaluate the potential.

        Parameters
        ----------
        R : float
            Galactocentric radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Potential at (R, phi, t).

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)
        """
        return self._Pot(R, 0.0, t=t, use_physical=False)

    def _Rforce(self, R, phi=0.0, t=0.0):
        """
        Evaluate the radial force.

        Parameters
        ----------
        R : float
            Galactocentric radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Radial force at (R, phi, t).

        Notes
        -----
        - Written on 2010-07-13 by Bovy (NYU).
        """
        return self._Pot.Rforce(R, 0.0, t=t, use_physical=False)

    def _R2deriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the second radial derivative.

        Parameters
        ----------
        R : float
            Galactocentric radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Second radial derivative at (R, phi, t).

        Notes
        -----
        - 2011-10-09: Written by Bovy (IAS).

        """
        return self._Pot.R2deriv(R, 0.0, t=t, use_physical=False)


def RZToplanarPotential(RZPot):
    """
    Convert an RZPotential to a planarPotential in the mid-plane (z=0).

    Parameters
    ----------
    RZPot : RZPotential instance or list of such instances
        Existing planarPotential instances are just copied to the output.

    Returns
    -------
    planarPotential instance(s)

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)

    """
    RZPot = flatten(RZPot)
    if _isDissipative(RZPot):
        raise NotImplementedError(
            "Converting dissipative forces to 2D axisymmetric potentials is currently not supported"
        )
    if isinstance(RZPot, list):
        out = []
        for pot in RZPot:
            if isinstance(pot, planarPotential) and not pot.isNonAxi:
                out.append(pot)
            elif isinstance(pot, Potential) and not pot.isNonAxi:
                out.append(planarPotentialFromRZPotential(pot))
            else:
                raise PotentialError(
                    "Input to 'RZToplanarPotential' is neither an RZPotential-instance or a list of such instances"
                )
        return out
    elif isinstance(RZPot, Potential) and not RZPot.isNonAxi:
        return planarPotentialFromRZPotential(RZPot)
    elif isinstance(RZPot, planarPotential) and not RZPot.isNonAxi:
        return RZPot
    else:
        raise PotentialError(
            "Input to 'RZToplanarPotential' is neither an RZPotential-instance or a list of such instances"
        )


class planarPotentialFromFullPotential(planarPotential):
    """Class that represents a planar potential derived from a non-axisymmetric
    3D potential"""

    def __init__(self, Pot):
        """
        Initialize.

        Parameters
        ----------
        Pot : Potential instance
            Potential instance.

        Returns
        -------
        planarPotential instance

        Notes
        -----
        - 2016-06-02 - Written - Bovy (UofT)

        """
        planarPotential.__init__(self, amp=1.0, ro=Pot._ro, vo=Pot._vo)
        # Also transfer roSet and voSet
        self._roSet = Pot._roSet
        self._voSet = Pot._voSet
        self._Pot = Pot
        self.hasC = Pot.hasC
        self.hasC_dxdv = Pot.hasC_dxdv
        self.hasC_dens = Pot.hasC_dens
        return None

    def _evaluate(self, R, phi=0.0, t=0.0):
        """
        Evaluate the potential.

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
            Potential at (R, phi, t).

        Notes
        -----
        - 2016-06-02: Written - Bovy (UofT)

        """
        return self._Pot(R, 0.0, phi=phi, t=t, use_physical=False)

    def _Rforce(self, R, phi=0.0, t=0.0):
        """
        Evaluate the radial force.

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
            Radial force at (R, phi, t).

        Notes
        -----
        - Written on 2016-06-02 by Bovy (UofT)

        """
        return self._Pot.Rforce(R, 0.0, phi=phi, t=t, use_physical=False)

    def _phitorque(self, R, phi=0.0, t=0.0):
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

        Returns
        -------
        float
            Azimuthal torque at (R, phi, t).

        Notes
        -----
        - 2016-06-02: Written - Bovy (UofT)

        """
        return self._Pot.phitorque(R, 0.0, phi=phi, t=t, use_physical=False)

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
            Second radial derivative at (R, phi, t).

        Notes
        -----
        - 2016-06-02: Written - Bovy (UofT)

        """
        return self._Pot.R2deriv(R, 0.0, phi=phi, t=t, use_physical=False)

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
            Second azimuthal derivative at (R, phi, t).

        Notes
        -----
        - 2016-06-02: Written - Bovy (UofT)

        """
        return self._Pot.phi2deriv(R, 0.0, phi=phi, t=t, use_physical=False)

    def _Rphideriv(self, R, phi=0.0, t=0.0):
        """
        Evaluate the mixed radial-azimuthal derivative.

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
            Mixed radial-azimuthal derivative at (R, phi, t).

        Notes
        -----
        - 2016-06-02: Written - Bovy (UofT)

        """
        return self._Pot.Rphideriv(R, 0.0, phi=phi, t=t, use_physical=False)

    def OmegaP(self):
        """
        Return the pattern speed.

        Returns
        -------
        float
            Pattern speed.

        Notes
        -----
        - 2016-05-31: Written - Bovy (UofT)
        """
        return self._Pot.OmegaP()


def toPlanarPotential(Pot):
    """
    Convert an Potential to a planarPotential in the mid-plane (z=0).

    Parameters
    ----------
    Pot : Potential instance or list of such instances
        Existing planarPotential instances are just copied to the output.

    Returns
    -------
    planarPotential, planarAxiPotential, or planarDissipativeForce instance(s)

    Notes
    -----
    - 2016-06-11: Written - Bovy (UofT)

    """
    Pot = flatten(Pot)
    if isinstance(Pot, list):
        out = []
        for pot in Pot:
            if isinstance(pot, planarForce):
                out.append(pot)
            elif isinstance(pot, Potential) and pot.isNonAxi:
                out.append(planarPotentialFromFullPotential(pot))
            elif isinstance(pot, Potential):
                out.append(planarPotentialFromRZPotential(pot))
            elif isinstance(pot, DissipativeForce):
                out.append(planarDissipativeForceFromFullDissipativeForce(pot))
            else:
                raise PotentialError(
                    "Input to 'toPlanarPotential' is neither an Potential-instance or a list of such instances"
                )
        return out
    elif isinstance(Pot, Potential) and Pot.isNonAxi:
        return planarPotentialFromFullPotential(Pot)
    elif isinstance(Pot, Potential):
        return planarPotentialFromRZPotential(Pot)
    elif isinstance(Pot, planarPotential):
        return Pot
    elif isinstance(Pot, DissipativeForce):
        return planarDissipativeForceFromFullDissipativeForce(Pot)
    else:
        raise PotentialError(
            "Input to 'toPlanarPotential' is neither an Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluateplanarPotentials(Pot, R, phi=None, t=0.0, dR=0, dphi=0):
    """
    Evaluate a (list of) planarPotential instance(s).

    Parameters
    ----------
    Pot : planarPotential or list of planarPotential
        A (list of) planarPotential instance(s).
    R : float or Quantity
        Cylindrical radius.
    phi : float or Quantity, optional
        Azimuth (default None).
    t : float or Quantity, optional
        Time (default 0.0).
    dR : int, optional
        If set to a non-zero integer, return the dR derivative instead. Default is 0.
    dphi : int, optional
        If set to a non-zero integer, return the dphi derivative instead. Default is 0.

    Returns
    -------
    float or Quantity
        Potential Phi(R(,phi,t)).

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluateplanarPotentials(Pot, R, phi=phi, t=t, dR=dR, dphi=dphi)


def _evaluateplanarPotentials(Pot, R, phi=None, t=0.0, dR=0, dphi=0):
    from .Potential import _isNonAxi

    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi"
        )
    if isList and numpy.all([isinstance(p, planarPotential) for p in Pot]):
        sum = 0.0
        for pot in Pot:
            if nonAxi:
                sum += pot._call_nodecorator(R, phi=phi, t=t, dR=dR, dphi=dphi)
            else:
                sum += pot._call_nodecorator(R, t=t, dR=dR, dphi=dphi)
        return sum
    elif isinstance(Pot, planarPotential):
        if nonAxi:
            return Pot._call_nodecorator(R, phi=phi, t=t, dR=dR, dphi=dphi)
        else:
            return Pot._call_nodecorator(R, t=t, dR=dR, dphi=dphi)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluateplanarRforces(Pot, R, phi=None, t=0.0, v=None):
    """
    Evaluate the cylindrical radial force of a (list of) planarPotential instance(s).

    Parameters
    ----------
    Pot : (list of) planarPotential instance(s)
        The potential(s) to evaluate.
    R : float or Quantity
        Cylindrical radius.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates (default: None).
        Required when including dissipative forces.

    Returns
    -------
    float or Quantity
        The cylindrical radial force F_R(R, phi, t).

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)
    - 2023-05-29 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluateplanarRforces(Pot, R, phi=phi, t=t, v=v)


def _evaluateplanarRforces(Pot, R, phi=None, t=0.0, v=None):
    """Raw, undecorated function for internal use"""
    from .Potential import _isNonAxi

    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) planarForce instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) planarForce instances includes dissipative components, but you did not provide the 2D velocity (required for dissipative forces)"
        )
    if isinstance(Pot, list) and numpy.all([isinstance(p, planarForce) for p in Pot]):
        sum = 0.0
        for pot in Pot:
            if _isDissipative(pot):
                sum += pot._Rforce_nodecorator(R, phi=phi, t=t, v=v)
            elif nonAxi:
                sum += pot._Rforce_nodecorator(R, phi=phi, t=t)
            else:
                sum += pot._Rforce_nodecorator(R, t=t)
        return sum
    elif dissipative:
        return Pot._Rforce_nodecorator(R, phi=phi, t=t, v=v)
    elif isinstance(Pot, planarPotential):
        if nonAxi:
            return Pot._Rforce_nodecorator(R, phi=phi, t=t)
        else:
            return Pot._Rforce_nodecorator(R, t=t)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluateplanarphitorques(Pot, R, phi=None, t=0.0, v=None):
    """
    Evaluate the phi torque of a (list of) planarPotential instance(s).

    Parameters
    ----------
    Pot : (list of) planarPotential instance(s)
        The potential(s) to evaluate.
    R : float or Quantity
        Cylindrical radius
    phi : float or Quantity, optional
        Azimuth (default: None)
    t : float or Quantity, optional
        Time (default: 0.0)
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates (default: None)
        Required when including dissipative forces.

    Returns
    -------
    float or Quantity
        The phitorque tau_phi(R, phi, t).

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)
    - 2023-05-29 - Added velocity input for dissipative forces - Bovy (UofT)

    """
    return _evaluateplanarphitorques(Pot, R, phi=phi, t=t, v=v)


def _evaluateplanarphitorques(Pot, R, phi=None, t=0.0, v=None):
    from .Potential import _isNonAxi

    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi"
        )
    dissipative = _isDissipative(Pot)
    if dissipative and v is None:
        raise PotentialError(
            "The (list of) planarForce instances includes dissipative components, but you did not provide the 2D velocity (required for dissipative forces)"
        )
    if isinstance(Pot, list) and numpy.all([isinstance(p, planarForce) for p in Pot]):
        sum = 0.0
        for pot in Pot:
            if _isDissipative(pot):
                sum += pot._phitorque_nodecorator(R, phi=phi, t=t, v=v)
            elif nonAxi:
                sum += pot._phitorque_nodecorator(R, phi=phi, t=t)
            else:
                sum += pot._phitorque_nodecorator(R, t=t)
        return sum
    elif dissipative:
        return Pot._phitorque_nodecorator(R, phi=phi, t=t, v=v)
    elif isinstance(Pot, planarPotential):
        if nonAxi:
            return Pot._phitorque_nodecorator(R, phi=phi, t=t)
        else:
            return Pot._phitorque_nodecorator(R, t=t)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("forcederivative", pop=True)
def evaluateplanarR2derivs(Pot, R, phi=None, t=0.0):
    """
    Evaluate the second radial derivative of planarPotential instance(s).

    Parameters
    ----------
    Pot : (list of) planarPotential instance(s)
        The potential(s) to evaluate.
    R : float or Quantity
        Cylindrical radius
    phi : float or Quantity, optional
        Azimuth (default: None)
    t : float or Quantity, optional
        Time (default: 0.0)

    Returns
    -------
    float or Quantity
        The second potential derivative d2Phi/dR2(R, phi, t).

    Notes
    -----
    - 2010-10-09 - Written - Bovy (IAS)

    """
    from .Potential import _isNonAxi

    isList = isinstance(Pot, list)
    nonAxi = _isNonAxi(Pot)
    if nonAxi and phi is None:
        raise PotentialError(
            "The (list of) planarPotential instances is non-axisymmetric, but you did not provide phi"
        )
    if isinstance(Pot, list) and numpy.all(
        [isinstance(p, planarPotential) for p in Pot]
    ):
        sum = 0.0
        for pot in Pot:
            if nonAxi:
                sum += pot.R2deriv(R, phi=phi, t=t, use_physical=False)
            else:
                sum += pot.R2deriv(R, t=t, use_physical=False)
        return sum
    elif isinstance(Pot, planarPotential):
        if nonAxi:
            return Pot.R2deriv(R, phi=phi, t=t, use_physical=False)
        else:
            return Pot.R2deriv(R, t=t, use_physical=False)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatePotentials' is neither a Potential-instance or a list of such instances"
        )


def LinShuReductionFactor(
    axiPot, R, sigmar, nonaxiPot=None, k=None, m=None, OmegaP=None
):
    r"""
    Calculate the Lin & Shu (1966) reduction factor: the reduced linear response of a kinematically-warm stellar disk to a perturbation

    Parameters
    ----------
    axiPot : Potential or list of Potential instances
        The background, axisymmetric potential
    R : float or Quantity
        Cylindrical radius
    sigmar : float or Quantity
        Radial velocity dispersion of the population
    nonaxiPot : Potential object, optional
        A non-axisymmetric Potential instance (such as SteadyLogSpiralPotential) that has functions that return OmegaP, m, and wavenumber. Either provide nonaxiPot or m, k, OmegaP.
    k : float or Quantity, optional
        Wavenumber (see Binney & Tremaine 2008). Either provide nonaxiPot or m, k, OmegaP.
    m : int, optional
        m in the perturbation's m x phi (number of arms for a spiral). Either provide nonaxiPot or m, k, OmegaP.
    OmegaP : float or Quantity, optional
        Pattern speed. Note that in the usual Lin-Shu formula \omega = m x OmegaP. Either provide nonaxiPot or m, k, OmegaP.

    Returns
    -------
    float
        The reduction factor

    Notes
    -----
    - 2014-08-23 - Written - Bovy (IAS)

    """
    axiPot = flatten(axiPot)
    from ..potential import epifreq, omegac

    if nonaxiPot is None and (OmegaP is None or k is None or m is None):
        raise OSError(
            "Need to specify either nonaxiPot= or m=, k=, OmegaP= for LinShuReductionFactor"
        )
    elif not nonaxiPot is None:
        OmegaP = nonaxiPot.OmegaP()
        k = nonaxiPot.wavenumber(R)
        m = nonaxiPot.m()
    tepif = epifreq(axiPot, R)
    # We define omega = m x OmegaP in the usual Lin-Shu formula
    s = m * (OmegaP - omegac(axiPot, R)) / tepif
    chi = sigmar**2.0 * k**2.0 / tepif**2.0
    return (
        (1.0 - s**2.0)
        / numpy.sin(numpy.pi * s)
        * integrate.quad(
            lambda t: numpy.exp(-chi * (1.0 + numpy.cos(t)))
            * numpy.sin(s * t)
            * numpy.sin(t),
            0.0,
            numpy.pi,
        )[0]
    )


def plotplanarPotentials(Pot, *args, **kwargs):
    """
    Plot a planar potential.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential or list of potentials to plot
    Rrange : list or Quantity, optional
        Range in R to plot (default is [0.01, 5.0])
    xrange, yrange : list, optional
        Range in x and y to plot (can be Quantity) (default is [-5.0, 5.0])
    grid, gridx, gridy : int, optional
        Number of points to plot (default is 100). grid for 1D plots, gridx and gridy for 2D plots
    savefilename : str, optional
        Save to or restore from this savefile (pickle)
    ncontours : int, optional
        Number of contours to plot (if applicable)
    *args, **kwargs :
        Arguments and keyword arguments for `galpy.util.plot.plot` or `galpy.util.plot.dens2d`

    Returns
    -------
    plot to output device

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)

    """
    Pot = flatten(Pot)
    Rrange = kwargs.pop("Rrange", [0.01, 5.0])
    xrange = kwargs.pop("xrange", [-5.0, 5.0])
    yrange = kwargs.pop("yrange", [-5.0, 5.0])
    if hasattr(Pot, "_ro"):
        tro = Pot._ro
    else:
        tro = Pot[0]._ro
    Rrange[0] = conversion.parse_length(Rrange[0], ro=tro)
    Rrange[1] = conversion.parse_length(Rrange[1], ro=tro)
    xrange[0] = conversion.parse_length(xrange[0], ro=tro)
    xrange[1] = conversion.parse_length(xrange[1], ro=tro)
    yrange[0] = conversion.parse_length(yrange[0], ro=tro)
    yrange[1] = conversion.parse_length(yrange[1], ro=tro)
    grid = kwargs.pop("grid", 100)
    gridx = kwargs.pop("gridx", 100)
    gridy = kwargs.pop("gridy", gridx)
    savefilename = kwargs.pop("savefilename", None)
    isList = isinstance(Pot, list)
    nonAxi = (isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi)
    if not savefilename is None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        potR = pickle.load(savefile)
        if nonAxi:
            xs = pickle.load(savefile)
            ys = pickle.load(savefile)
        else:
            Rs = pickle.load(savefile)
        savefile.close()
    else:
        if nonAxi:
            xs = numpy.linspace(xrange[0], xrange[1], gridx)
            ys = numpy.linspace(yrange[0], yrange[1], gridy)
            potR = numpy.zeros((gridx, gridy))
            for ii in range(gridx):
                for jj in range(gridy):
                    thisR = numpy.sqrt(xs[ii] ** 2.0 + ys[jj] ** 2.0)
                    if xs[ii] >= 0.0:
                        thisphi = numpy.arcsin(ys[jj] / thisR)
                    else:
                        thisphi = -numpy.arcsin(ys[jj] / thisR) + numpy.pi
                    potR[ii, jj] = evaluateplanarPotentials(
                        Pot, thisR, phi=thisphi, use_physical=False
                    )
        else:
            Rs = numpy.linspace(Rrange[0], Rrange[1], grid)
            potR = numpy.zeros(grid)
            for ii in range(grid):
                potR[ii] = evaluateplanarPotentials(Pot, Rs[ii], use_physical=False)
        if not savefilename is None:
            print("Writing planar savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(potR, savefile)
            if nonAxi:
                pickle.dump(xs, savefile)
                pickle.dump(ys, savefile)
            else:
                pickle.dump(Rs, savefile)
            savefile.close()
    if nonAxi:
        if not "origin" in kwargs:
            kwargs["origin"] = "lower"
        if not "cmap" in kwargs:
            kwargs["cmap"] = "gist_yarg"
        if not "contours" in kwargs:
            kwargs["contours"] = True
        if not "xlabel" in kwargs:
            kwargs["xlabel"] = r"$x / R_0$"
        if not "ylabel" in kwargs:
            kwargs["ylabel"] = "$y / R_0$"
        if not "aspect" in kwargs:
            kwargs["aspect"] = 1.0
        if not "cntrls" in kwargs:
            kwargs["cntrls"] = "-"
        ncontours = kwargs.pop("ncontours", 10)
        if not "levels" in kwargs:
            kwargs["levels"] = numpy.linspace(
                numpy.nanmin(potR), numpy.nanmax(potR), ncontours
            )
        return plot.dens2d(potR.T, xrange=xrange, yrange=yrange, **kwargs)
    else:
        kwargs["xlabel"] = r"$R/R_0$"
        kwargs["ylabel"] = r"$\Phi(R)$"
        kwargs["xrange"] = Rrange
        return plot.plot(Rs, potR, *args, **kwargs)
