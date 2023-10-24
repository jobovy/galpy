import os
import pickle

import numpy

from ..util import conversion, plot
from ..util.conversion import (
    parse_length,
    parse_velocity,
    physical_conversion,
    potential_physical_input,
)

_INF = 10**12.0


def plotEscapecurve(Pot, *args, **kwargs):
    """
    Plot the escape velocity curve for this potential (in the z=0 plane for non-spherical potentials).

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential(s) for which to plot the escape velocity curve.
    Rrange : numpy.ndarray or Quantity, optional
        Range in R to consider (can be Quantity).
    grid : int, optional
        Grid in R.
    savefilename : str, optional
        Save to or restore from this savefile (pickle).
    *args, **kwargs : dict
        Arguments and keyword arguments for `galpy.util.plot.plot`.

    Returns
    -------
    matplotlib.AxesSubplot
        Plot to output device.

    Notes
    -----
    - 2010-08-08 - Written by Bovy (NYU).

    """
    # Using physical units or not?
    if isinstance(Pot, list):
        potro = Pot[0]._ro
        roSet = Pot[0]._roSet
        potvo = Pot[0]._vo
        voSet = Pot[0]._voSet
    else:
        potro = Pot._ro
        roSet = Pot._roSet
        potvo = Pot._vo
        voSet = Pot._voSet
    # Following just to deal with Quantity ro/vo and check whether they are set
    _ro = parse_length(kwargs.get("ro", None), ro=potro)
    _vo = parse_velocity(kwargs.get("vo", None), vo=potvo)
    if (
        kwargs.get("use_physical", True)
        and (not _ro is None or roSet)
        and (not _vo is None or voSet)
    ):
        use_physical = True
        potro = kwargs.get("ro", potro)
        potvo = kwargs.get("vo", potvo)
        xlabel = r"$R\,(\mathrm{kpc})$"
        ylabel = r"$v_e(R)\,(\mathrm{km\,s}^{-1})$"
        Rrange = kwargs.pop("Rrange", [0.01 * potro, 5.0 * potro])
    else:
        use_physical = False
        xlabel = r"$R/R_0$"
        ylabel = r"$v_e(R)/v_c(R_0)$"
        Rrange = kwargs.pop("Rrange", [0.01, 5.0])
    # Parse ro
    potro = conversion.parse_length_kpc(potro)
    potvo = conversion.parse_velocity_kms(potvo)
    Rrange[0] = conversion.parse_length_kpc(Rrange[0])
    Rrange[1] = conversion.parse_length_kpc(Rrange[1])
    if use_physical:
        Rrange[0] /= potro
        Rrange[1] /= potro
    grid = kwargs.pop("grid", 1001)
    savefilename = kwargs.pop("savefilename", None)
    if not savefilename == None and os.path.exists(savefilename):
        print("Restoring savefile " + savefilename + " ...")
        savefile = open(savefilename, "rb")
        esccurve = pickle.load(savefile)
        Rs = pickle.load(savefile)
        savefile.close()
    else:
        Rs = numpy.linspace(Rrange[0], Rrange[1], grid)
        esccurve = calcEscapecurve(Pot, Rs)
        if not savefilename == None:
            print("Writing savefile " + savefilename + " ...")
            savefile = open(savefilename, "wb")
            pickle.dump(esccurve, savefile)
            pickle.dump(Rs, savefile)
            savefile.close()
    if use_physical:
        Rs *= potro
        esccurve *= potvo
        Rrange[0] *= potro
        Rrange[1] *= potro
    if not "xlabel" in kwargs:
        kwargs["xlabel"] = xlabel
    if not "ylabel" in kwargs:
        kwargs["ylabel"] = ylabel
    if not "xrange" in kwargs:
        kwargs["xrange"] = Rrange
    if not "yrange" in kwargs:
        kwargs["yrange"] = [
            0.0,
            1.2 * numpy.amax(esccurve[True ^ numpy.isnan(esccurve)]),
        ]
    kwargs.pop("ro", None)
    kwargs.pop("vo", None)
    kwargs.pop("use_physical", None)
    return plot.plot(Rs, esccurve, *args, **kwargs)


def calcEscapecurve(Pot, Rs, t=0.0):
    """
    Calculate the escape velocity curve for this potential (in the z=0 plane for non-spherical potentials).

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential or list of Potential instances.
    Rs : numpy.ndarray or Quantity
        Radius(i).
    t : float, optional
        Instantaneous time (default is 0.0).

    Returns
    -------
    numpy.ndarray or Quantity
        Array of v_esc.

    Raises
    ------
    AttributeError
        Escape velocity curve plotting for non-axisymmetric potentials is not currently supported.

    Notes
    -----
    - 2011-04-16 - Written - Bovy (NYU)

    """
    isList = isinstance(Pot, list)
    isNonAxi = (isList and Pot[0].isNonAxi) or (not isList and Pot.isNonAxi)
    if isNonAxi:
        raise AttributeError(
            "Escape velocity curve plotting for non-axisymmetric potentials is not currently supported"
        )
    try:
        grid = len(Rs)
    except TypeError:
        grid = 1
        Rs = numpy.array([Rs])
    esccurve = numpy.zeros(grid)
    for ii in range(grid):
        esccurve[ii] = vesc(Pot, Rs[ii], t=t, use_physical=False)
    return esccurve


@potential_physical_input
@physical_conversion("velocity", pop=True)
def vesc(Pot, R, t=0.0):
    """
    Calculate the escape velocity at R for potential Pot.

    Parameters
    ----------
    Pot : Potential or list of Potential instances
        Potential or list of Potential instances.
    R : numpy.ndarray or Quantity
        Galactocentric radius.
    t : float, optional
        Time (default is 0.0).

    Returns
    -------
    numpy.ndarray or Quantity
        Escape velocity.

    Notes
    -----
    - 2011-10-09 - Written - Bovy (IAS)

    """
    from ..potential import PotentialError, evaluateplanarPotentials

    try:
        return numpy.sqrt(
            2.0
            * (
                evaluateplanarPotentials(Pot, _INF, t=t, use_physical=False)
                - evaluateplanarPotentials(Pot, R, t=t, use_physical=False)
            )
        )
    except PotentialError:
        from ..potential import RZToplanarPotential

        Pot = RZToplanarPotential(Pot)
        return numpy.sqrt(
            2.0
            * (
                evaluateplanarPotentials(Pot, _INF, t=t, use_physical=False)
                - evaluateplanarPotentials(Pot, R, t=t, use_physical=False)
            )
        )
