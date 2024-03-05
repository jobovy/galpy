##############################################################################
#
#   plot.py: general wrappers for matplotlib plotting
#
#       'public' methods:
#                         end_print
#                         dens2d
#                         hist
#                         plot
#                         start_print
#                         scatterplot (like hogg_scatterplot)
#                         text
#
#                         this module also defines a custom matplotlib
#                         projection in which the polar azimuth increases
#                         clockwise (as in, the Galaxy viewed from the NGP)
#
#############################################################################
#############################################################################
# Copyright (c) 2010 - 2020, Jo Bovy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import re

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as pyplot
import matplotlib.ticker as ticker
import numpy
from matplotlib import rc
from matplotlib.projections import PolarAxes, register_projection
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting (projection = '3d')
from packaging.version import parse as parse_version
from scipy import interpolate, ndimage, special

_MPL_VERSION = parse_version(matplotlib.__version__)
from ..util.config import __config__

if __config__.getboolean("plot", "seaborn-bovy-defaults"):
    try:
        import seaborn as sns
    except:
        pass
    else:
        sns.set_style(
            "ticks",
            {
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.labelsize": 18.0,
                "axes.titlesize": 18.0,
                "figure.figsize": numpy.array([6.64, 4.0]),
                "grid.linewidth": 2.0,
                "legend.fontsize": 18.0,
                "lines.linewidth": 2.0,
                "lines.markeredgewidth": 0.0,
                "lines.markersize": 14.0,
                "patch.linewidth": 0.6,
                "xtick.labelsize": 16.0,
                "xtick.major.pad": 14.0,
                "xtick.major.width": 2.0,
                "xtick.minor.width": 1.0,
                "ytick.labelsize": 16.0,
                "ytick.major.pad": 14.0,
                "ytick.major.width": 2.0,
            },
        )
_DEFAULTNCNTR = 10


def end_print(filename, **kwargs):
    """
    Save the current figure(s) to a file.

    Parameters
    ----------
    filename : str
        Filename for the plot (with extension).
    **kwargs
        Additional keyword arguments to pass to `pyplot.savefig`.

    Notes
    -----
    - 2009-12-23 - Written - Bovy (NYU)
    """
    if "format" in kwargs:
        pyplot.savefig(filename, **kwargs)
    else:
        pyplot.savefig(filename, format=re.split(r"\.", filename)[-1], **kwargs)
    pyplot.close()


def hist(x, xlabel=None, ylabel=None, overplot=False, **kwargs):
    """
    Plot a histogram of the input array using matplotlib's hist function.

    Parameters
    ----------
    x : numpy.ndarray
        Array to histogram.
    xlabel : str, optional
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str, optional
        y-axis label, LaTeX math mode, no $s needed.
    overplot : bool, optional
        If True, plot on top of the current figure.
    **kwargs
        All other keyword arguments are passed to ``pyplot.hist``.

    Returns
    -------
    tuple
        Output from ``pyplot.hist``

    Notes
    -----
    - 2009-12-23 - Written - Bovy (NYU)
    """
    if not overplot:
        pyplot.figure()
    if "xrange" in kwargs:
        xlimits = kwargs.pop("xrange")
        if not "range" in kwargs:
            kwargs["range"] = xlimits
        xrangeSet = True
    else:
        xrangeSet = False
    if "yrange" in kwargs:
        ylimits = kwargs.pop("yrange")
        yrangeSet = True
    else:
        yrangeSet = False
    out = pyplot.hist(x, **kwargs)
    if overplot:
        return out
    _add_axislabels(xlabel, ylabel)
    if not "range" in kwargs and not xrangeSet:
        if isinstance(x, list):
            xlimits = (numpy.nanmin(numpy.array(x)), numpy.nanmax(numpy.array(x)))
        else:
            pyplot.xlim(numpy.nanmin(x), numpy.nanmax(x))
    elif xrangeSet:
        pyplot.xlim(xlimits)
    else:
        pyplot.xlim(kwargs["range"])
    if yrangeSet:
        pyplot.ylim(ylimits)
    _add_ticks()
    return out


def plot(*args, **kwargs):
    """
    Wrapper around matplotlib's plot function.

    Parameters
    ----------
    *args:
        Inputs to ``pyplot.plot``.
    xlabel : str, optional
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str, optional
        y-axis label, LaTeX math mode, no $s needed.
    xrange : tuple, optional
        x range to plot over.
    yrange : tuple, optional
        y range to plot over.
    overplot : bool, optional
        If True, plot on top of the current figure.
    gcf : bool, optional
        If True, do not start a new figure.
    onedhists : bool, optional
        If True, make one-d histograms on the sides.
    onedhistcolor : str, optional
        Histogram color.
    onedhistfc : str, optional
        Histogram fill color.
    onedhistec : str, optional
        Histogram edge color.
    onedhistxnormed : bool, optional
        If True, normalize the x-axis histogram.
    onedhistynormed : bool, optional
        If True, normalize the y-axis histogram.
    onedhistxweights : numpy.ndarray, optional
        Weights for the x-axis histogram.
    onedhistyweights : numpy.ndarray, optional
        Weights for the y-axis histogram.
    bins : int, optional
        Number of bins for the one-d histograms.
    semilogx : bool, optional
        If True, plot the x-axis on a log scale.
    semilogy : bool, optional
        If True, plot the y-axis on a log scale.
    loglog : bool, optional
        If True, plot both axes on a log scale.
    scatter : bool, optional
        If True, use ``pyplot.scatter`` instead of ``pyplot.plot``.
    colorbar : bool, optional
        If True, add a colorbar.
    crange : tuple, optional
        Range for the colorbar.
    clabel : str, optional
        Label for the colorbar.
    **kwargs
        All other keyword arguments are passed to ``pyplot.plot`` or ``pyplot.scatter``.

    Returns
    -------
    tuple
        Output from ``pyplot.plot``/``pyplot.scatter`` or 3 Axes instances if ``onedhists=True``.

    Notes
    -----
    - 2009-12-28 - Written - Bovy (NYU)
    """
    overplot = kwargs.pop("overplot", False)
    gcf = kwargs.pop("gcf", False)
    onedhists = kwargs.pop("onedhists", False)
    scatter = kwargs.pop("scatter", False)
    loglog = kwargs.pop("loglog", False)
    semilogx = kwargs.pop("semilogx", False)
    semilogy = kwargs.pop("semilogy", False)
    colorbar = kwargs.pop("colorbar", False)
    onedhisttype = kwargs.pop("onedhisttype", "step")
    onedhistcolor = kwargs.pop("onedhistcolor", "k")
    onedhistfc = kwargs.pop("onedhistfc", "w")
    onedhistec = kwargs.pop("onedhistec", "k")
    onedhistxnormed = kwargs.pop("onedhistxnormed", True)
    onedhistynormed = kwargs.pop("onedhistynormed", True)
    onedhistxweights = kwargs.pop("onedhistxweights", None)
    onedhistyweights = kwargs.pop("onedhistyweights", None)
    if "bins" in kwargs:
        bins = kwargs["bins"]
        kwargs.pop("bins")
    elif onedhists:
        if isinstance(args[0], numpy.ndarray):
            bins = round(0.3 * numpy.sqrt(args[0].shape[0]))
        elif isinstance(args[0], list):
            bins = round(0.3 * numpy.sqrt(len(args[0])))
        else:
            bins = 30
    if onedhists:
        if overplot or gcf:
            fig = pyplot.gcf()
        else:
            fig = pyplot.figure()
        nullfmt = NullFormatter()  # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = pyplot.axes(rect_scatter)
        axHistx = pyplot.axes(rect_histx)
        axHisty = pyplot.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        axHisty.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        fig.sca(axScatter)
    elif not overplot and not gcf:
        pyplot.figure()
    ax = pyplot.gca()
    ax.set_autoscale_on(False)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    clabel = kwargs.pop("clabel", None)
    xlimits = kwargs.pop("xrange", None)
    if xlimits is None:
        if isinstance(args[0], list):
            xlimits = (
                numpy.nanmin(numpy.array(args[0])),
                numpy.nanmax(numpy.array(args[0])),
            )
        else:
            xlimits = (numpy.nanmin(args[0]), numpy.nanmax(args[0]))
    ylimits = kwargs.pop("yrange", None)
    if ylimits is None:
        if isinstance(args[1], list):
            ylimits = (
                numpy.nanmin(numpy.array(args[1])),
                numpy.nanmax(numpy.array(args[1])),
            )
        else:
            ylimits = (numpy.nanmin(args[1]), numpy.nanmax(args[1]))
    climits = kwargs.pop("crange", None)
    if climits is None and scatter:
        if "c" in kwargs and isinstance(kwargs["c"], list):
            climits = (
                numpy.nanmin(numpy.array(kwargs["c"])),
                numpy.nanmax(numpy.array(kwargs["c"])),
            )
        elif "c" in kwargs:
            climits = (numpy.nanmin(kwargs["c"]), numpy.nanmax(kwargs["c"].nanmax()))
        else:
            climits = None
    if scatter:
        out = pyplot.scatter(*args, **kwargs)
    elif loglog:
        out = pyplot.loglog(*args, **kwargs)
    elif semilogx:
        out = pyplot.semilogx(*args, **kwargs)
    elif semilogy:
        out = pyplot.semilogy(*args, **kwargs)
    else:
        out = pyplot.plot(*args, **kwargs)
    if overplot:
        pass
    else:
        if semilogy:
            ax = pyplot.gca()
            ax.set_yscale("log")
        elif semilogx:
            ax = pyplot.gca()
            ax.set_xscale("log")
        elif loglog:
            ax = pyplot.gca()
            ax.set_xscale("log")
            ax.set_yscale("log")
        pyplot.xlim(*xlimits)
        pyplot.ylim(*ylimits)
        _add_axislabels(xlabel, ylabel)
        if not semilogy and not semilogx and not loglog:
            _add_ticks()
        elif semilogy:
            _add_ticks(xticks=True, yticks=False)
        elif semilogx:
            _add_ticks(yticks=True, xticks=False)
    # Add colorbar
    if colorbar:
        cbar = pyplot.colorbar(out, fraction=0.15)
        if _MPL_VERSION < parse_version("3.1"):  # pragma: no cover
            # https://matplotlib.org/3.1.0/api/api_changes.html#colorbarbase-inheritance
            cbar.set_clim(*climits)
        else:
            cbar.mappable.set_clim(*climits)
        if not clabel is None:
            cbar.set_label(clabel)
    # Add onedhists
    if not onedhists:
        return out
    histx, edges, patches = axHistx.hist(
        args[0],
        bins=bins,
        normed=onedhistxnormed,
        weights=onedhistxweights,
        histtype=onedhisttype,
        range=sorted(xlimits),
        color=onedhistcolor,
        fc=onedhistfc,
        ec=onedhistec,
    )
    histy, edges, patches = axHisty.hist(
        args[1],
        bins=bins,
        orientation="horizontal",
        weights=onedhistyweights,
        normed=onedhistynormed,
        histtype=onedhisttype,
        range=sorted(ylimits),
        color=onedhistcolor,
        fc=onedhistfc,
        ec=onedhistec,
    )
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axHistx.set_ylim(0, 1.2 * numpy.amax(histx))
    axHisty.set_xlim(0, 1.2 * numpy.amax(histy))
    return (axScatter, axHistx, axHisty)


def plot3d(*args, **kwargs):
    """
    Wrapper around ``pyplot.plot`` for 3D plots, much like plot is a wrapper around ``pyplot.plot`` for 2D plots.

    Parameters
    ----------
    *args:
        Inputs to ``pyplot.plot3d``.
    xlabel : str, optional
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str, optional
        y-axis label, LaTeX math mode, no $s needed.
    zlabel : str, optional
        z-axis label, LaTeX math mode, no $s needed.
    xrange : tuple, optional
        x range to plot over.
    yrange : tuple, optional
        y range to plot over.
    zrange : tuple, optional
        z range to plot over.
    overplot : bool, optional
        If True, plot on top of the current figure.

    Returns
    -------
    tuple
        Output from ``pyplot.plot3d``.

    Notes
    -----
    - 2011-01-08 - Written - Bovy (NYU)
    """
    overplot = kwargs.pop("overplot", False)
    if not overplot:
        pyplot.figure()
    ax = pyplot.gcf().add_subplot(projection="3d")
    ax.set_autoscale_on(False)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    zlabel = kwargs.pop("zlabel", None)
    if "xrange" in kwargs:
        xlimits = kwargs.pop("xrange")
    else:
        if isinstance(args[0], list):
            xlimits = (
                numpy.nanmin(numpy.array(args[0])),
                numpy.nanmax(numpy.array(args[0])),
            )
        else:
            xlimits = (numpy.nanmin(args[0]), numpy.nanmax(args[0]))
    if "yrange" in kwargs:
        ylimits = kwargs.pop("yrange")
    else:
        if isinstance(args[1], list):
            ylimits = (
                numpy.nanmin(numpy.array(args[1])),
                numpy.nanmax(numpy.array(args[1])),
            )
        else:
            ylimits = (numpy.nanmin(args[1]), numpy.nanmax(args[1]))
    if "zrange" in kwargs:
        zlimits = kwargs.pop("zrange")
    else:
        if isinstance(args[2], list):
            zlimits = (
                numpy.nanmin(numpy.array(args[2])),
                numpy.nanmax(numpy.array(args[2])),
            )
        else:
            zlimits = (numpy.nanmin(args[2]), numpy.nanmax(args[2]))
    out = pyplot.plot(*args, **kwargs)
    if overplot:
        pass
    else:
        if xlabel != None:
            if xlabel[0] != "$":
                thisxlabel = r"$" + xlabel + "$"
            else:
                thisxlabel = xlabel
            ax.set_xlabel(thisxlabel)
        if ylabel != None:
            if ylabel[0] != "$":
                thisylabel = r"$" + ylabel + "$"
            else:
                thisylabel = ylabel
            ax.set_ylabel(thisylabel)
        if zlabel != None:
            if zlabel[0] != "$":
                thiszlabel = r"$" + zlabel + "$"
            else:
                thiszlabel = zlabel
            ax.set_zlabel(thiszlabel)
        ax.set_xlim3d(*xlimits)
        ax.set_ylim3d(*ylimits)
        ax.set_zlim3d(*zlimits)
    return out


def dens2d(X, **kwargs):
    """
    Plot a 2d density with optional contours.

    Parameters
    ----------
    X : numpy.ndarray
        The density to plot.
    *args :
        Arguments for ``pyplot.imshow``.
    xlabel : str, optional
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str, optional
        y-axis label, LaTeX math mode, no $s needed.
    xrange : tuple, optional
        x range to plot over.
    yrange : tuple, optional
        y range to plot over.
    noaxes : bool, optional
        If True, don't plot any axes.
    overplot : bool, optional
        If True, overplot.
    gcf : bool, optional
        If True, do not start a new figure.
    colorbar : bool, optional
        If True, add colorbar.
    shrink : float, optional
        Colorbar shrink factor.
    conditional : bool, optional
        Normalize each column separately (for probability densities, i.e., ``cntrmass=True``).
    justcontours : bool, optional
        If True, only draw contours.
    contours : bool, optional
        If True, draw contours (10 by default).
    levels : numpy.ndarray, optional
        Contour levels.
    cntrmass : bool, optional
        If True, the density is a probability and the levels are probability masses contained within the contour.
    cntrcolors : str or list, optional
        Colors for contours (single color or array).
    cntrlabel : bool, optional
        Label the contours.
    cntrlw : float, optional
        Linewidths for contour.
    cntrls : str, optional
        Linestyles for contour.
    cntrlabelsize : float, optional
        Size of contour labels.
    cntrlabelcolors : str, optional
        Color of contour labels.
    cntrinline : bool, optional
        If True, put contour labels inline with contour.
    cntrSmooth : float, optional
        Use ``ndimage.gaussian_filter`` to smooth before contouring.
    retAxes : bool, optional
        Return all Axes instances.
    retCont : bool, optional
        Return the contour instance.

    Returns
    -------
    Axes or tuple
        Plot to output device, Axes instances depending on input.

    Notes
    -----
    - 2010-03-09 - Written - Bovy (NYU)
    """
    overplot = kwargs.pop("overplot", False)
    gcf = kwargs.pop("gcf", False)
    if not overplot and not gcf:
        pyplot.figure()
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    zlabel = kwargs.pop("zlabel", None)
    if "extent" in kwargs:
        extent = kwargs.pop("extent")
    else:
        xlimits = kwargs.pop("xrange", [0, X.shape[1]])
        ylimits = kwargs.pop("yrange", [0, X.shape[0]])
        extent = xlimits + ylimits
    if not "aspect" in kwargs:
        kwargs["aspect"] = (xlimits[1] - xlimits[0]) / float(ylimits[1] - ylimits[0])
    noaxes = kwargs.pop("noaxes", False)
    justcontours = kwargs.pop("justcontours", False)
    if (
        ("contours" in kwargs and kwargs["contours"])
        or "levels" in kwargs
        or justcontours
        or ("cntrmass" in kwargs and kwargs["cntrmass"])
    ):
        contours = True
    else:
        contours = False
    kwargs.pop("contours", None)
    if "levels" in kwargs:
        levels = kwargs["levels"]
        kwargs.pop("levels")
    elif contours:
        if "cntrmass" in kwargs and kwargs["cntrmass"]:
            levels = numpy.linspace(0.0, 1.0, _DEFAULTNCNTR)
        elif True in numpy.isnan(numpy.array(X)):
            levels = numpy.linspace(numpy.nanmin(X), numpy.nanmax(X), _DEFAULTNCNTR)
        else:
            levels = numpy.linspace(numpy.amin(X), numpy.amax(X), _DEFAULTNCNTR)
    cntrmass = kwargs.pop("cntrmass", False)
    conditional = kwargs.pop("conditional", False)
    cntrcolors = kwargs.pop("cntrcolors", "k")
    cntrlabel = kwargs.pop("cntrlabel", False)
    cntrlw = kwargs.pop("cntrlw", None)
    cntrls = kwargs.pop("cntrls", None)
    cntrSmooth = kwargs.pop("cntrSmooth", None)
    cntrlabelsize = kwargs.pop("cntrlabelsize", None)
    cntrlabelcolors = kwargs.pop("cntrlabelcolors", None)
    cntrinline = kwargs.pop("cntrinline", None)
    retCumImage = kwargs.pop("retCumImage", False)
    cb = kwargs.pop("colorbar", False)
    shrink = kwargs.pop("shrink", None)
    onedhists = kwargs.pop("onedhists", False)
    onedhistcolor = kwargs.pop("onedhistcolor", "k")
    retAxes = kwargs.pop("retAxes", False)
    retCont = kwargs.pop("retCont", False)
    if onedhists:
        if overplot or gcf:
            fig = pyplot.gcf()
        else:
            fig = pyplot.figure()
        nullfmt = NullFormatter()  # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = pyplot.axes(rect_scatter)
        axHistx = pyplot.axes(rect_histx)
        axHisty = pyplot.axes(rect_histy)
        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        axHisty.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        fig.sca(axScatter)
    ax = pyplot.gca()
    ax.set_autoscale_on(False)
    if conditional:
        plotthis = X / numpy.tile(numpy.sum(X, axis=0), (X.shape[1], 1))
    else:
        plotthis = X
    if not justcontours:
        out = pyplot.imshow(plotthis, extent=extent, **kwargs)
    if not overplot:
        pyplot.axis(extent)
        _add_axislabels(xlabel, ylabel)
        _add_ticks()
    # Add colorbar
    if cb and not justcontours:
        if shrink is None:
            shrink = numpy.amin([float(kwargs.pop("aspect", 1.0)) * 0.87, 1.0])
        CB1 = pyplot.colorbar(out, shrink=shrink)
        if not zlabel is None:
            if zlabel[0] != "$":
                thiszlabel = r"$" + zlabel + "$"
            else:
                thiszlabel = zlabel
            CB1.set_label(thiszlabel)
    if contours or retCumImage:
        aspect = kwargs.get("aspect", None)
        origin = kwargs.get("origin", None)
        if cntrmass:
            # Sum from the top down!
            plotthis[numpy.isnan(plotthis)] = 0.0
            sortindx = numpy.argsort(plotthis.flatten())[::-1]
            cumul = numpy.cumsum(numpy.sort(plotthis.flatten())[::-1]) / numpy.sum(
                plotthis.flatten()
            )
            cntrThis = numpy.zeros(numpy.prod(plotthis.shape))
            cntrThis[sortindx] = cumul
            cntrThis = numpy.reshape(cntrThis, plotthis.shape)
        else:
            cntrThis = plotthis
        if contours:
            if not cntrSmooth is None:
                cntrThis = ndimage.gaussian_filter(cntrThis, cntrSmooth, mode="nearest")
            cont = pyplot.contour(
                cntrThis,
                levels,
                colors=cntrcolors,
                linewidths=cntrlw,
                extent=extent,
                linestyles=cntrls,
                origin=origin,
            )
            if cntrlabel:
                pyplot.clabel(
                    cont,
                    fontsize=cntrlabelsize,
                    colors=cntrlabelcolors,
                    inline=cntrinline,
                )
    if noaxes:
        ax.set_axis_off()
    # Add onedhists
    if not onedhists:
        if retCumImage:
            return cntrThis
        elif retAxes:
            return pyplot.gca()
        elif retCont:
            return cont
        elif justcontours:
            return cntrThis
        else:
            return out
    histx = (
        numpy.nansum(X.T, axis=1) * numpy.fabs(ylimits[1] - ylimits[0]) / X.shape[1]
    )  # nansum bc nan is *no dens value*
    histy = numpy.nansum(X.T, axis=0) * numpy.fabs(xlimits[1] - xlimits[0]) / X.shape[0]
    histx[numpy.isnan(histx)] = 0.0
    histy[numpy.isnan(histy)] = 0.0
    dx = (extent[1] - extent[0]) / float(len(histx))
    axHistx.plot(
        numpy.linspace(extent[0] + dx, extent[1] - dx, len(histx)),
        histx,
        drawstyle="steps-mid",
        color=onedhistcolor,
    )
    dy = (extent[3] - extent[2]) / float(len(histy))
    axHisty.plot(
        histy,
        numpy.linspace(extent[2] + dy, extent[3] - dy, len(histy)),
        drawstyle="steps-mid",
        color=onedhistcolor,
    )
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axHistx.set_ylim(0, 1.2 * numpy.amax(histx))
    axHisty.set_xlim(0, 1.2 * numpy.amax(histy))
    if retCumImage:
        return cntrThis
    elif retAxes:
        return (axScatter, axHistx, axHisty)
    elif justcontours:
        return cntrThis
    else:
        return out


def start_print(
    fig_width=5,
    fig_height=5,
    axes_labelsize=16,
    text_fontsize=11,
    legend_fontsize=12,
    xtick_labelsize=10,
    ytick_labelsize=10,
    xtick_minor_size=2,
    ytick_minor_size=2,
    xtick_major_size=4,
    ytick_major_size=4,
):
    """
    Set up a figure for plotting.

    Parameters
    ----------
    fig_width : float, optional
        Width in inches. Default is 5.
    fig_height : float, optional
        Height in inches. Default is 5.
    axes_labelsize : int, optional
        Size of the axis-labels. Default is 16.
    text_fontsize : int, optional
        Font-size of the text (if any). Default is 11.
    legend_fontsize : int, optional
        Font-size of the legend (if any). Default is 12.
    xtick_labelsize : int, optional
        Size of the x-axis labels. Default is 10.
    ytick_labelsize : int, optional
        Size of the y-axis labels. Default is 10.
    xtick_minor_size : int, optional
        Size of the minor x-ticks. Default is 2.
    ytick_minor_size : int, optional
        Size of the minor y-ticks. Default is 2.
    xtick_major_size : int, optional
        Size of the major x-ticks. Default is 4.
    ytick_major_size : int, optional
        Size of the major y-ticks. Default is 4.

    Notes
    -----
    - 2009-12-23 - Written - Bovy (NYU).
    """
    fig_size = [fig_width, fig_height]
    params = {
        "axes.labelsize": axes_labelsize,
        "font.size": text_fontsize,
        "legend.fontsize": legend_fontsize,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
        "text.usetex": True,
        "figure.figsize": fig_size,
        "xtick.major.size": xtick_major_size,
        "ytick.major.size": ytick_major_size,
        "xtick.minor.size": xtick_minor_size,
        "ytick.minor.size": ytick_minor_size,
        "legend.numpoints": 1,
        "xtick.top": True,
        "xtick.direction": "in",
        "ytick.right": True,
        "ytick.direction": "in",
    }
    pyplot.rcParams.update(params)
    rc("text.latex", preamble=r"\usepackage{amsmath}" + "\n" + r"\usepackage{amssymb}")


def text(*args, **kwargs):
    """
    Thin wrapper around matplotlib's text and annotate.

    Parameters
    ----------
    *args :
        See matplotlib's text
        (http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.text).
    **kwargs :
        'bottom_left=True', 'bottom_right=True', 'top_left=True', 'top_right=True', 'title=True'
        to place the text in one of the corners or use it as the title.

    Notes
    -----
    - 2010-01-26 - Written - Bovy (NYU)
    """
    if kwargs.pop("title", False):
        pyplot.annotate(
            args[0],
            (0.5, 1.05),
            xycoords="axes fraction",
            horizontalalignment="center",
            verticalalignment="top",
            **kwargs,
        )
    elif kwargs.pop("bottom_left", False):
        pyplot.annotate(args[0], (0.05, 0.05), xycoords="axes fraction", **kwargs)
    elif kwargs.pop("bottom_right", False):
        pyplot.annotate(
            args[0],
            (0.95, 0.05),
            xycoords="axes fraction",
            horizontalalignment="right",
            **kwargs,
        )
    elif kwargs.pop("top_right", False):
        pyplot.annotate(
            args[0],
            (0.95, 0.95),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
            **kwargs,
        )
    elif kwargs.pop("top_left", False):
        pyplot.annotate(
            args[0],
            (0.05, 0.95),
            xycoords="axes fraction",
            verticalalignment="top",
            **kwargs,
        )
    else:
        pyplot.text(*args, **kwargs)


def scatterplot(x, y, *args, **kwargs):
    """
    Make a 'smart' scatterplot that is a density plot in high-density regions and a regular scatterplot for outliers.

    Parameters
    ----------
    x : numpy.ndarray
        x data.
    y : numpy.ndarray
        y data.
    xlabel : str, optional
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str, optional
        y-axis label, LaTeX math mode, no $s needed.
    xrange : tuple, optional
        x range to plot over.
    yrange : tuple, optional
        y range to plot over.
    bins : int, optional
        Number of bins to use in each dimension.
    weights : numpy.ndarray, optional
        Data-weights.
    aspect : float, optional
        Aspect ratio.
    conditional : bool, optional
        Normalize each column separately (for probability densities, i.e., ``cntrmass=True``).
    overplot : bool, optional
        If True, overplot.
    gcf : bool, optional
        Do not start a new figure (does change the ranges and labels).
    contours : bool, optional
        If False, don't plot contours.
    justcontours : bool, optional
        If True, only draw contours, no density.
    cntrcolors : str or list, optional
        Color of contours (can be array as for dens2d).
    cntrlw : float, optional
        Linewidths for contour.
    cntrls : str, optional
        Linestyles for contour.
    cntrSmooth : float, optional
        Use ``ndimage.gaussian_filter`` to smooth before contouring.
    levels : numpy.ndarray, optional
        Contour-levels; data points outside of the last level will be individually shown (so, e.g., if this list is descending, contours and data points will be overplotted).
    onedhists : bool, optional
        If True, make one-d histograms on the sides.
    onedhistx : bool, optional
        If True, make one-d histograms on the side of the x distribution.
    onedhisty : bool, optional
        If True, make one-d histograms on the side of the y distribution.
    onedhistcolor : str, optional
        Color of one-d histograms.
    onedhistfc : str, optional
        Facecolor of one-d histograms.
    onedhistec : str, optional
        Edgecolor of one-d histograms.
    onedhistxnormed : bool, optional
        Normed keyword for one-d histograms.
    onedhistynormed : bool, optional
        Normed keyword for one-d histograms.
    onedhistxweights : numpy.ndarray, optional
        Weights keyword for one-d histograms.
    onedhistyweights : numpy.ndarray, optional
        Weights keyword for one-d histograms.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for density plot.
    hist : numpy.ndarray, optional
        You can supply the histogram of the data yourself, this can be useful if you want to censor the data, both need to be set and calculated using scipy.histogramdd with the given range.
    edges : numpy.ndarray, optional
        You can supply the histogram of the data yourself, this can be useful if you want to censor the data, both need to be set and calculated using scipy.histogramdd with the given range.
    retAxes : bool, optional
        Return all Axes instances.

    Returns
    -------
    Axes or tuple
        Plot to output device, Axes instance(s) or not, depending on input.

    Notes
    -----
    - 2010-04-15 - Written - Bovy (NYU)
    """
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    if "xrange" in kwargs:
        xrange = kwargs.pop("xrange")
    else:
        if isinstance(x, list):
            xrange = [numpy.amin(x), numpy.amax(x)]
        else:
            xrange = [x.min(), x.max()]
    if "yrange" in kwargs:
        yrange = kwargs.pop("yrange")
    else:
        if isinstance(y, list):
            yrange = [numpy.amin(y), numpy.amax(y)]
        else:
            yrange = [y.min(), y.max()]
    ndata = len(x)
    bins = kwargs.pop("bins", round(0.3 * numpy.sqrt(ndata)))
    weights = kwargs.pop("weights", None)
    levels = kwargs.pop("levels", special.erf(numpy.arange(1, 4) / numpy.sqrt(2.0)))
    aspect = kwargs.pop("aspect", (xrange[1] - xrange[0]) / (yrange[1] - yrange[0]))
    conditional = kwargs.pop("conditional", False)
    contours = kwargs.pop("contours", True)
    justcontours = kwargs.pop("justcontours", False)
    cntrcolors = kwargs.pop("cntrcolors", "k")
    cntrlw = kwargs.pop("cntrlw", None)
    cntrls = kwargs.pop("cntrls", None)
    cntrSmooth = kwargs.pop("cntrSmooth", None)
    onedhists = kwargs.pop("onedhists", False)
    onedhistx = kwargs.pop("onedhistx", onedhists)
    onedhisty = kwargs.pop("onedhisty", onedhists)
    onedhisttype = kwargs.pop("onedhisttype", "step")
    onedhistcolor = kwargs.pop("onedhistcolor", "k")
    onedhistfc = kwargs.pop("onedhistfc", "w")
    onedhistec = kwargs.pop("onedhistec", "k")
    onedhistls = kwargs.pop("onedhistls", "solid")
    onedhistlw = kwargs.pop("onedhistlw", None)
    onedhistsbins = kwargs.pop("onedhistsbins", round(0.3 * numpy.sqrt(ndata)))
    overplot = kwargs.pop("overplot", False)
    gcf = kwargs.pop("gcf", False)
    cmap = kwargs.pop("cmap", cm.gist_yarg)
    onedhistxnormed = kwargs.pop("onedhistxnormed", True)
    onedhistynormed = kwargs.pop("onedhistynormed", True)
    onedhistxweights = kwargs.pop("onedhistxweights", weights)
    onedhistyweights = kwargs.pop("onedhistyweights", weights)
    retAxes = kwargs.pop("retAxes", False)
    if onedhists or onedhistx or onedhisty:
        if overplot or gcf:
            fig = pyplot.gcf()
        else:
            fig = pyplot.figure()
        nullfmt = NullFormatter()  # no labels
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]
        axScatter = pyplot.axes(rect_scatter)
        if onedhistx:
            axHistx = pyplot.axes(rect_histx)
            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHistx.yaxis.set_major_formatter(nullfmt)
        if onedhisty:
            axHisty = pyplot.axes(rect_histy)
            # no labels
            axHisty.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)
        fig.sca(axScatter)
    data = numpy.array([x, y]).T
    if "hist" in kwargs and "edges" in kwargs:
        hist = kwargs["hist"]
        kwargs.pop("hist")
        edges = kwargs["edges"]
        kwargs.pop("edges")
    else:
        hist, edges = numpy.histogramdd(
            data, bins=bins, range=[xrange, yrange], weights=weights
        )
    if contours:
        cumimage = dens2d(
            hist.T,
            contours=contours,
            levels=levels,
            cntrmass=contours,
            cntrSmooth=cntrSmooth,
            cntrcolors=cntrcolors,
            cmap=cmap,
            origin="lower",
            xrange=xrange,
            yrange=yrange,
            xlabel=xlabel,
            ylabel=ylabel,
            interpolation="nearest",
            retCumImage=True,
            aspect=aspect,
            conditional=conditional,
            cntrlw=cntrlw,
            cntrls=cntrls,
            justcontours=justcontours,
            zorder=5 * justcontours,
            overplot=(gcf or onedhists or overplot or onedhistx or onedhisty),
        )
    else:
        cumimage = dens2d(
            hist.T,
            contours=contours,
            cntrcolors=cntrcolors,
            cmap=cmap,
            origin="lower",
            xrange=xrange,
            yrange=yrange,
            xlabel=xlabel,
            ylabel=ylabel,
            interpolation="nearest",
            conditional=conditional,
            retCumImage=True,
            aspect=aspect,
            cntrlw=cntrlw,
            cntrls=cntrls,
            overplot=(gcf or onedhists or overplot or onedhistx or onedhisty),
        )
    # Set axes and labels
    pyplot.axis(list(xrange) + list(yrange))
    if not overplot:
        _add_axislabels(xlabel, ylabel)
        _add_ticks()
    binxs = []
    xedge = edges[0]
    for ii in range(len(xedge) - 1):
        binxs.append((xedge[ii] + xedge[ii + 1]) / 2.0)
    binxs = numpy.array(binxs)
    binys = []
    yedge = edges[1]
    for ii in range(len(yedge) - 1):
        binys.append((yedge[ii] + yedge[ii + 1]) / 2.0)
    binys = numpy.array(binys)
    cumInterp = interpolate.RectBivariateSpline(binxs, binys, cumimage.T, kx=1, ky=1)
    cums = []
    for ii in range(len(x)):
        cums.append(cumInterp(x[ii], y[ii])[0, 0])
    cums = numpy.array(cums)
    plotx = x[cums > levels[-1]]
    ploty = y[cums > levels[-1]]
    if not len(plotx) == 0:
        if not weights == None:
            w8 = weights[cums > levels[-1]]
            for ii in range(len(plotx)):
                plot(
                    plotx[ii],
                    ploty[ii],
                    overplot=True,
                    color="%.2f" % (1.0 - w8[ii]),
                    *args,
                    **kwargs,
                )
        else:
            plot(plotx, ploty, overplot=True, zorder=1, *args, **kwargs)
    # Add onedhists
    if not (onedhists or onedhistx or onedhisty):
        if retAxes:
            return pyplot.gca()
        else:
            return None
    if onedhistx:
        histx, edges, patches = axHistx.hist(
            x,
            bins=onedhistsbins,
            normed=onedhistxnormed,
            weights=onedhistxweights,
            histtype=onedhisttype,
            range=sorted(xrange),
            color=onedhistcolor,
            fc=onedhistfc,
            ec=onedhistec,
            ls=onedhistls,
            lw=onedhistlw,
        )
    if onedhisty:
        histy, edges, patches = axHisty.hist(
            y,
            bins=onedhistsbins,
            orientation="horizontal",
            weights=onedhistyweights,
            normed=onedhistynormed,
            histtype=onedhisttype,
            range=sorted(yrange),
            color=onedhistcolor,
            fc=onedhistfc,
            ec=onedhistec,
            ls=onedhistls,
            lw=onedhistlw,
        )
    if onedhistx and not overplot:
        axHistx.set_xlim(axScatter.get_xlim())
        axHistx.set_ylim(0, 1.2 * numpy.amax(histx))
    if onedhisty and not overplot:
        axHisty.set_ylim(axScatter.get_ylim())
        axHisty.set_xlim(0, 1.2 * numpy.amax(histy))
    if not onedhistx:
        axHistx = None
    if not onedhisty:
        axHisty = None
    if retAxes:
        return (axScatter, axHistx, axHisty)
    else:
        return None


def _add_axislabels(xlabel, ylabel):
    """
    Add axis labels to the current figure.

    Parameters
    ----------
    xlabel : str
        x-axis label, LaTeX math mode, no $s needed.
    ylabel : str
        y-axis label, LaTeX math mode, no $s needed.

    Notes
    -----
    = 2009-12-23 - Written - Bovy (NYU).
    """
    if xlabel != None:
        if xlabel[0] != "$":
            thisxlabel = r"$" + xlabel + "$"
        else:
            thisxlabel = xlabel
        pyplot.xlabel(thisxlabel)
    if ylabel != None:
        if ylabel[0] != "$":
            thisylabel = r"$" + ylabel + "$"
        else:
            thisylabel = ylabel
        pyplot.ylabel(thisylabel)


def _add_ticks(xticks=True, yticks=True):
    """
    Add minor axis ticks to a plot.

    Parameters
    ----------
    xticks : bool, optional
        If True, add minor ticks to the x-axis. Default is True.
    yticks : bool, optional
        If True, add minor ticks to the y-axis. Default is True.

    Notes
    -----
    - 2009-12-23 - Written - Bovy (NYU)
    """
    ax = pyplot.gca()
    if xticks:
        xstep = ax.xaxis.get_majorticklocs()
        xstep = xstep[1] - xstep[0]
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xstep / 5.0))
    if yticks:
        ystep = ax.yaxis.get_majorticklocs()
        ystep = ystep[1] - ystep[0]
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(ystep / 5.0))


class GalPolarAxes(PolarAxes):
    """
    A variant of PolarAxes where theta increases clockwise
    """

    name = "galpolar"

    class GalPolarTransform(PolarAxes.PolarTransform):
        def transform(self, tr):
            xy = numpy.zeros(tr.shape, numpy.float64)
            t = tr[:, 0:1]
            r = tr[:, 1:2]
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            x[:] = r * numpy.cos(t)
            y[:] = -r * numpy.sin(t)
            return xy

        transform_non_affine = transform

        def inverted(self):
            return GalPolarAxes.InvertedGalPolarTransform()

    class InvertedGalPolarTransform(PolarAxes.InvertedPolarTransform):
        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = numpy.sqrt(x * x + y * y)
            theta = numpy.arctan2(y, x)
            return numpy.concatenate((theta, r), 1)

        def inverted(self):
            return GalPolarAxes.GalPolarTransform()

    def _set_lim_and_transforms(self):
        PolarAxes._set_lim_and_transforms(self)
        self.transProjection = self.GalPolarTransform()
        self.transData = (
            self.transScale
            + self.transProjection
            + (self.transProjectionAffine + self.transAxes)
        )
        self._xaxis_transform = (
            self.transProjection
            + self.PolarAffine(IdentityTransform(), Bbox.unit())
            + self.transAxes
        )
        self._xaxis_text1_transform = (
            self._theta_label1_position + self._xaxis_transform
        )
        self._yaxis_transform = Affine2D().scale(numpy.pi * 2.0, 1.0) + self.transData
        self._yaxis_text1_transform = (
            self._r_label1_position
            + Affine2D().scale(1.0 / 360.0, 1.0)
            + self._yaxis_transform
        )


register_projection(GalPolarAxes)
