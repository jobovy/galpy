import copy
import os
import os.path
import pickle

import numpy

from ..util import config, conversion, plot
from ..util.conversion import (
    physical_compatible,
    physical_conversion,
    potential_physical_input,
)
from .Potential import PotentialError, flatten, potential_positional_arg


class linearPotential:
    """Class representing 1D potentials"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        self._amp = amp
        self.dim = 1
        self.isRZ = False
        self.hasC = False
        self.hasC_dxdv = False
        self.hasC_dens = False
        # Parse ro and vo
        if ro is None:
            self._ro = config.__config__.getfloat("normalization", "ro")
            self._roSet = False
        else:
            ro = conversion.parse_length_kpc(ro)
            self._ro = ro
            self._roSet = True
        if vo is None:
            self._vo = config.__config__.getfloat("normalization", "vo")
            self._voSet = False
        else:
            vo = conversion.parse_velocity_kms(vo)
            self._vo = vo
            self._voSet = True
        return None

    def __mul__(self, b):
        """
        Multiply a linearPotential's amplitude by a number

        Parameters
        ----------
        b : int or float
            Number to multiply the amplitude of the linearPotential instance with.

        Returns
        -------
        linearPotential instance
            New instance with amplitude = (old amplitude) x b.

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)

        """
        if not isinstance(b, (int, float)):
            raise TypeError(
                "Can only multiply a planarPotential instance with a number"
            )
        out = copy.deepcopy(self)
        out._amp *= b
        return out

    # Similar functions
    __rmul__ = __mul__

    def __div__(self, b):
        return self.__mul__(1.0 / b)

    __truediv__ = __div__

    def __add__(self, b):
        """
        Add linearPotential instances together to create a multi-component potential (e.g., pot= pot1+pot2+pot3)

        Parameters
        ----------
        b : linearPotential instance or a list thereof

        Returns
        -------
        list of linearPotential instances
            Represents the combined potential

        Notes
        -----
        - 2019-01-27 - Written - Bovy (UofT)

        """
        from ..potential import flatten as flatten_pot

        if not isinstance(flatten_pot([b])[0], linearPotential):
            raise TypeError(
                """Can only combine galpy linearPotential"""
                """ objects with """
                """other such objects or lists thereof"""
            )
        assert physical_compatible(self, b), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )
        if isinstance(b, list):
            return [self] + b
        else:
            return [self, b]

    # Define separately to keep order
    def __radd__(self, b):
        from ..potential import flatten as flatten_pot

        if not isinstance(flatten_pot([b])[0], linearPotential):
            raise TypeError(
                """Can only combine galpy linearPotential"""
                """ objects with """
                """other such objects or lists thereof"""
            )
        assert physical_compatible(self, b), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )
        # If we get here, b has to be a list
        return b + [self]

    def turn_physical_off(self):
        """
        Turn off automatic returning of outputs in physical units.

        Returns
        -------
        None

        Notes
        -----
        - 2016-01-30 - Written - Bovy (UofT)

        """
        self._roSet = False
        self._voSet = False
        return None

    def turn_physical_on(self, ro=None, vo=None):
        """
        Turn on automatic returning of outputs in physical units.

        Parameters
        ----------
        ro : float or Quantity, optional
            Reference distance (kpc).
        vo : float or Quantity, optional
            Reference velocity (km/s).

        Returns
        -------
        None

        Notes
        -----
        - 2016-01-30 - Written - Bovy (UofT)
        - 2020-04-22 - Don't turn on a parameter when it is False - Bovy (UofT)

        """
        if not ro is False:
            self._roSet = True
            ro = conversion.parse_length_kpc(ro)
            if not ro is None:
                self._ro = ro
        if not vo is False:
            self._voSet = True
            vo = conversion.parse_velocity_kms(vo)
            if not vo is None:
                self._vo = vo
        return None

    @potential_physical_input
    @physical_conversion("energy", pop=True)
    def __call__(self, x, t=0.0):
        """
        Evaluate the potential.

        Parameters
        ----------
        x : float or Quantity
            Position.
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            Potential at position x and time t.

        Notes
        -----
        - 2010-07-12 - Written - Bovy (NYU)

        """
        return self._call_nodecorator(x, t=t)

    def _call_nodecorator(self, x, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._evaluate(x, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError(
                "'_evaluate' function not implemented for this potential"
            )

    @potential_physical_input
    @physical_conversion("force", pop=True)
    def force(self, x, t=0.0):
        """
        Evaluate the force.

        Parameters
        ----------
        x : float or Quantity
            Position.
        t : float or Quantity, optional
            Time (default: 0.0).

        Returns
        -------
        float or Quantity
            Force at position x and time t.

        Notes
        -----
        - 2010-07-12 - Written - Bovy (NYU)

        """
        return self._force_nodecorator(x, t=t)

    def _force_nodecorator(self, x, t=0.0):
        # Separate, so it can be used during orbit integration
        try:
            return self._amp * self._force(x, t=t)
        except AttributeError:  # pragma: no cover
            raise PotentialError("'_force' function not implemented for this potential")

    def plot(self, t=0.0, min=-15.0, max=15, ns=21, savefilename=None):
        """
        Plot the potential

        Parameters
        ----------
        t : float or Quantity, optional
            The time at which to evaluate the forces. Default is 0.0.
        min : float, optional
            Minimum x.
        max : float, optional
            Maximum x.
        ns : int, optional
            Grid in x.
        savefilename : str, optional
            Save to or restore from this savefile (pickle).

        Returns
        -------
        plot to output device

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)
        """
        if not savefilename == None and os.path.exists(savefilename):
            print("Restoring savefile " + savefilename + " ...")
            savefile = open(savefilename, "rb")
            potx = pickle.load(savefile)
            xs = pickle.load(savefile)
            savefile.close()
        else:
            xs = numpy.linspace(min, max, ns)
            potx = numpy.zeros(ns)
            for ii in range(ns):
                potx[ii] = self._evaluate(xs[ii], t=t)
            if not savefilename == None:
                print("Writing savefile " + savefilename + " ...")
                savefile = open(savefilename, "wb")
                pickle.dump(potx, savefile)
                pickle.dump(xs, savefile)
                savefile.close()
        return plot.plot(
            xs, potx, xlabel=r"$x/x_0$", ylabel=r"$\Phi(x)$", xrange=[min, max]
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("energy", pop=True)
def evaluatelinearPotentials(Pot, x, t=0.0):
    """
    Evaluate the sum of a list of potentials.

    Parameters
    ----------
    Pot : list of linearPotential instance(s)
        The list of potentials to evaluate.
    x : float or Quantity
        The position at which to evaluate the potentials.
    t : float or Quantity, optional
        The time at which to evaluate the potentials. Default is 0.0.

    Returns
    -------
    float or Quantity
        The value of the potential at the given position and time.

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)

    """
    return _evaluatelinearPotentials(Pot, x, t=t)


def _evaluatelinearPotentials(Pot, x, t=0.0):
    """Raw, undecorated function for internal use"""
    if isinstance(Pot, list):
        sum = 0.0
        for pot in Pot:
            sum += pot._call_nodecorator(x, t=t)
        return sum
    elif isinstance(Pot, linearPotential):
        return Pot._call_nodecorator(x, t=t)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluatelinearPotentials' is neither a linearPotential-instance or a list of such instances"
        )


@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
def evaluatelinearForces(Pot, x, t=0.0):
    """
    Evaluate the forces due to a list of potentials.

    Parameters
    ----------
    Pot : list of linearPotential instance(s)
        The list of potentials to evaluate.
    x : float or Quantity
        The position at which to evaluate the forces.
    t : float or Quantity, optional
        The time at which to evaluate the forces. Default is 0.0.

    Returns
    -------
    float or Quantity
        The value of the forces at the given position and time.

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)
    """
    return _evaluatelinearForces(Pot, x, t=t)


def _evaluatelinearForces(Pot, x, t=0.0):
    """Raw, undecorated function for internal use"""
    if isinstance(Pot, list):
        sum = 0.0
        for pot in Pot:
            sum += pot._force_nodecorator(x, t=t)
        return sum
    elif isinstance(Pot, linearPotential):
        return Pot._force_nodecorator(x, t=t)
    else:  # pragma: no cover
        raise PotentialError(
            "Input to 'evaluateForces' is neither a linearPotential-instance or a list of such instances"
        )


def plotlinearPotentials(Pot, t=0.0, min=-15.0, max=15, ns=21, savefilename=None):
    """
    Plot a combination of potentials

    Parameters
    ----------
    Pot : list of linearPotential instance(s)
        The list of potentials to evaluate.
    t : float or Quantity, optional
        The time at which to evaluate the forces. Default is 0.0.
    min : float, optional
        Minimum x.
    max : float, optional
        Maximum x.
    ns : int, optional
        Grid in x.
    savefilename : str, optional
        Save to or restore from this savefile (pickle).

    Returns
    -------
    plot to output device

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)
    """
    Pot = flatten(Pot)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        potx = pickle.load(savefile)
        xs = pickle.load(savefile)
        savefile.close()
    else:
        xs = numpy.linspace(min, max, ns)
        potx = numpy.zeros(ns)
        for ii in range(ns):
            potx[ii] = evaluatelinearPotentials(Pot, xs[ii], t=t)
        if not savefilename == None:
            print("Writing savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(potx, savefile)
            pickle.dump(xs, savefile)
            savefile.close()
    return plot.plot(
        xs, potx, xlabel=r"$x/x_0$", ylabel=r"$\Phi(x)$", xrange=[min, max]
    )
